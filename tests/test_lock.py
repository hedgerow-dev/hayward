"""`hayward lock`: pin a Hub repo's files for one loading profile, and verify a snapshot.

The Hub is faked (`hub._api_get`, `hub._download`), so these run offline.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hayward import hub, lock

SHA = "c" * 40


def git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()  # noqa: S324 - git's own id


class FakeRepo:
    def __init__(self, monkeypatch, files: dict[str, bytes], lfs: tuple[str, ...] = ()):
        self.files, self.lfs = files, set(lfs)
        monkeypatch.setattr(hub, "_api_get", self.api_get)
        monkeypatch.setattr(hub, "_download", self.download)

    def api_get(self, url: str):
        assert url.startswith("https://huggingface.co/api/models/")
        siblings = []
        for name, data in self.files.items():
            entry = {"rfilename": name, "size": len(data), "blobId": git_blob(data)}
            if name in self.lfs:
                entry["lfs"] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
            siblings.append(entry)
        return {"sha": SHA, "siblings": siblings}

    def download(self, url: str, dest: Path, max_bytes: int):
        name = url.split(f"/resolve/{SHA}/", 1)[1]
        dest.write_bytes(self.files[name])

    def snapshot(self, root: Path) -> Path:
        for name, data in self.files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return root


def make_lock(argv, tmp_path, capsys):
    out = tmp_path / "hayward.lock.json"
    code = lock.main([*argv, "-o", str(out)])
    err = capsys.readouterr().err
    return code, (json.loads(out.read_text()) if out.exists() else None), err


BASIC = {"config.json": b'{"model_type": "bert"}', "pytorch_model.bin": b"weights" * 10,
         "README.md": b"hi"}


def test_a_lock_pins_the_commit_and_every_file(monkeypatch, tmp_path, capsys):
    FakeRepo(monkeypatch, BASIC, lfs=("pytorch_model.bin",))
    code, data, _ = make_lock(["org/name"], tmp_path, capsys)
    assert code == 0 and data["status"] == "complete"
    assert data["repo"] == "org/name" and data["revision"] == SHA
    files = {f["path"]: f for f in data["files"]}
    assert set(files) == set(BASIC)
    assert files["pytorch_model.bin"]["sha256"] == hashlib.sha256(BASIC["pytorch_model.bin"]).hexdigest()
    assert files["config.json"]["git_blob"] == git_blob(BASIC["config.json"])
    assert data["profile"] == "transformers.from_pretrained"


def test_local_custom_code_is_listed_and_flagged(monkeypatch, tmp_path, capsys):
    files = {**BASIC, "config.json": json.dumps({"auto_map": {"AutoModel": "modeling_x.XModel"}}).encode(),
             "modeling_x.py": b"print(1)"}
    FakeRepo(monkeypatch, files)
    code, data, _ = make_lock(["org/name"], tmp_path, capsys)
    assert code == 0 and data["requires_trust_remote_code"] is True
    assert data["code_files"] == ["modeling_x.py"]


@pytest.mark.parametrize("auto_map", [
    {"AutoModel": "other/repo--modeling_x.XModel"},
    {"AutoTokenizer": ["other/repo--tok.T", None]},
])
def test_code_from_another_repo_needs_review(monkeypatch, tmp_path, capsys, auto_map):
    FakeRepo(monkeypatch, {**BASIC, "config.json": json.dumps({"auto_map": auto_map}).encode()})
    code, data, err = make_lock(["org/name"], tmp_path, capsys)
    assert code == 1 and data["status"] == "needs_review"
    assert data["unresolved"][0]["kind"] == "remote_code_repo" and "other/repo" in data["unresolved"][0]["ref"]
    assert "needs review" in err


def test_an_adapter_base_model_needs_review(monkeypatch, tmp_path, capsys):
    files = {**BASIC, "adapter_config.json": json.dumps({"base_model_name_or_path": "meta/base"}).encode()}
    FakeRepo(monkeypatch, files)
    code, data, _ = make_lock(["org/name"], tmp_path, capsys)
    assert code == 1 and data["unresolved"][0] == {
        "kind": "base_model", "ref": "meta/base",
        "why": "an adapter loads a base model that this lock does not pin"}


def test_an_unreadable_config_needs_review(monkeypatch, tmp_path, capsys):
    FakeRepo(monkeypatch, {**BASIC, "config.json": b"{not json"})
    code, data, _ = make_lock(["org/name"], tmp_path, capsys)
    assert code == 1 and data["unresolved"][0]["kind"] == "unreadable"


def test_an_unsafe_file_name_needs_review(monkeypatch, tmp_path, capsys):
    FakeRepo(monkeypatch, {**BASIC, "../escape.bin": b"x"})
    code, data, _ = make_lock(["org/name"], tmp_path, capsys)
    assert code == 1 and "../escape.bin" not in {f["path"] for f in data["files"]}
    assert data["unresolved"][0]["kind"] == "unsafe_name"


def test_a_malformed_repo_id_is_refused(capsys, tmp_path):
    assert lock.main(["../x", "-o", str(tmp_path / "l.json")]) == 2


def write_lock(monkeypatch, tmp_path, capsys, files, lfs=()):
    fake = FakeRepo(monkeypatch, files, lfs)
    make_lock(["org/name"], tmp_path, capsys)
    return fake, tmp_path / "hayward.lock.json"


def test_verify_accepts_a_matching_snapshot(monkeypatch, tmp_path, capsys):
    fake, path = write_lock(monkeypatch, tmp_path, capsys, BASIC, lfs=("pytorch_model.bin",))
    snap = fake.snapshot(tmp_path / "snap")
    assert lock.main(["--verify", str(path), str(snap)]) == 0
    assert "matches" in capsys.readouterr().out


@pytest.mark.parametrize("change,word", [
    (lambda d: (d / "pytorch_model.bin").write_bytes(b"other"), "changed"),
    (lambda d: (d / "config.json").unlink(), "missing"),
    (lambda d: (d / "modeling_evil.py").write_bytes(b"import os"), "not in the lock"),
])
def test_verify_refuses_changed_missing_and_extra_files(monkeypatch, tmp_path, capsys, change, word):
    fake, path = write_lock(monkeypatch, tmp_path, capsys, BASIC, lfs=("pytorch_model.bin",))
    snap = fake.snapshot(tmp_path / "snap")
    change(snap)
    assert lock.main(["--verify", str(path), str(snap)]) == 1
    assert word in capsys.readouterr().out


def test_verify_checks_small_files_by_git_blob_id(monkeypatch, tmp_path, capsys):
    fake, path = write_lock(monkeypatch, tmp_path, capsys, BASIC)
    snap = fake.snapshot(tmp_path / "snap")
    (snap / "README.md").write_bytes(b"hj")  # same size, different bytes
    assert lock.main(["--verify", str(path), str(snap)]) == 1


def test_verify_refuses_a_lock_that_needs_review(monkeypatch, tmp_path, capsys):
    files = {**BASIC, "adapter_config.json": json.dumps({"base_model_name_or_path": "meta/base"}).encode()}
    fake, path = write_lock(monkeypatch, tmp_path, capsys, files)
    snap = fake.snapshot(tmp_path / "snap")
    assert lock.main(["--verify", str(path), str(snap)]) == 1
    assert "needs review" in capsys.readouterr().out
