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
OTHER_SHA = "d" * 40


def git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()  # noqa: S324 - git's own id


class FakeRepo:
    """org/name at SHA; `others` are further repos, each at OTHER_SHA."""

    def __init__(self, monkeypatch, files: dict[str, bytes], lfs: tuple[str, ...] = (),
                 others: dict[str, dict[str, bytes]] | None = None):
        self.files, self.lfs, self.others = files, set(lfs), others or {}
        monkeypatch.setattr(hub, "_api_get", self.api_get)
        monkeypatch.setattr(hub, "_download", self.download)

    def api_get(self, url: str):
        assert url.startswith("https://huggingface.co/api/models/")
        repo = "/".join(url.removeprefix("https://huggingface.co/api/models/").split("?")[0].split("/")[:2])
        if repo != "org/name":
            if repo not in self.others:
                raise hub.HubError(f"{repo}: not found")
            return {"sha": OTHER_SHA, "siblings": [
                {"rfilename": n, "size": len(d), "blobId": git_blob(d)} for n, d in self.others[repo].items()]}
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


OTHER = {"other/repo": {"modeling_x.py": b"from .util import f", "util.py": b"f = 1",
                       "tok.py": b"T = 1", "proc.py": b"P = 1", "README.md": b"x"}}


@pytest.mark.parametrize("auto_map", [
    {"AutoModel": "other/repo--modeling_x.XModel"},
    {"AutoTokenizer": ["other/repo--tok.T", None]},
])
def test_code_from_one_other_repo_is_pinned(monkeypatch, tmp_path, capsys, auto_map):
    FakeRepo(monkeypatch, {**BASIC, "config.json": json.dumps({"auto_map": auto_map}).encode()}, others=OTHER)
    code, data, err = make_lock(["org/name"], tmp_path, capsys)
    assert code == 0 and data["status"] == "complete" and data["unresolved"] == []
    [dep] = data["dependencies"]
    assert dep["kind"] == "remote_code_repo" and dep["repo"] == "other/repo" and dep["revision"] == OTHER_SHA
    # Every code file, since a module's relative imports load from the same commit.
    assert {f["path"] for f in dep["code_files"]} == {"modeling_x.py", "util.py", "tok.py", "proc.py"}
    assert dep["code_files"][0]["git_blob"] == git_blob(OTHER["other/repo"][dep["code_files"][0]["path"]])
    assert data["load_with"] == {"revision": SHA, "code_revision": OTHER_SHA}
    assert f"code_revision={OTHER_SHA}" in err


@pytest.mark.parametrize("auto_map,why", [
    ({"AutoModel": "other/repo--modeling_x.XModel", "AutoConfig": "third/repo--c.C"}, "more than one"),
    ({"AutoModel": "other/repo--modeling_x.XModel", "AutoConfig": "configuration_x.C"}, "own code"),
    ({"AutoModel": "missing/repo--modeling_x.XModel"}, "did not return"),
    ({"AutoModel": "other/repo--nothere.XModel"}, "nothere.py"),
    ({"AutoModel": "../x--modeling_x.XModel"}, "not a repo id"),
])
def test_code_from_another_repo_that_cannot_be_pinned_needs_review(monkeypatch, tmp_path, capsys, auto_map, why):
    others = {**OTHER, "third/repo": {"c.py": b"C = 1"}}
    FakeRepo(monkeypatch, {**BASIC, "config.json": json.dumps({"auto_map": auto_map}).encode()}, others=others)
    code, data, err = make_lock(["org/name"], tmp_path, capsys)
    assert code == 1 and data["status"] == "needs_review" and data["dependencies"] == []
    assert "code_revision" not in data["load_with"]
    kinds = {u["kind"] for u in data["unresolved"]}
    assert kinds == {"remote_code_repo"} and any(why in u["why"] for u in data["unresolved"])
    assert "needs review" in err


def test_code_named_by_this_repos_own_id_is_local(monkeypatch, tmp_path, capsys):
    files = {**BASIC, "config.json": json.dumps({"auto_map": {"AutoModel": "org/name--modeling_x.X"}}).encode(),
             "modeling_x.py": b"X = 1"}
    FakeRepo(monkeypatch, files)
    code, data, _ = make_lock(["org/name"], tmp_path, capsys)
    assert code == 0 and data["dependencies"] == [] and data["load_with"] == {"revision": SHA}


def test_an_adapter_base_model_needs_review(monkeypatch, tmp_path, capsys):
    files = {**BASIC, "adapter_config.json": json.dumps({"base_model_name_or_path": "meta/base"}).encode()}
    FakeRepo(monkeypatch, files)
    code, data, _ = make_lock(["org/name"], tmp_path, capsys)
    assert code == 1 and data["unresolved"][0] == {
        "kind": "base_model", "ref": "meta/base",
        "why": "transformers loads an adapter's base model at the revision passed for the adapter, "
               "so one revision cannot pin both"}


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


@pytest.mark.parametrize("name", ["preprocessor_config.json", "processor_config.json",
                                  "video_preprocessor_config.json"])
def test_remote_code_in_any_auto_class_config_is_read(monkeypatch, tmp_path, capsys, name):
    # AutoProcessor and the image and feature-extractor Auto classes read auto_map from
    # their own config files, not only from config.json.
    files = {**BASIC, name: json.dumps({"auto_map": {"AutoProcessor": "third/repo--proc.P"}}).encode(),
             "config.json": json.dumps({"auto_map": {"AutoModel": "other/repo--modeling_x.X"}}).encode()}
    FakeRepo(monkeypatch, files, others={**OTHER, "third/repo": {"proc.py": b"P = 1"}})
    code, data, _ = make_lock(["org/name"], tmp_path, capsys)
    assert code == 1 and "third/repo--proc.P" in {u["ref"] for u in data["unresolved"]}
    assert data["requires_trust_remote_code"] is True


@pytest.mark.parametrize("tok_map,local", [
    (["other/repo--tok.T", None], False),
    (["tok.T", None], True),
])
def test_a_list_shaped_tokenizer_auto_map_is_read(monkeypatch, tmp_path, capsys, tok_map, local):
    # The older tokenizer_config.json format: AutoTokenizer reads a list, not a dict.
    files = {**BASIC, "tokenizer_config.json": json.dumps({"auto_map": tok_map}).encode(), "tok.py": b"T = 1"}
    if local:
        files["config.json"] = json.dumps({"auto_map": {"AutoModel": "other/repo--modeling_x.X"}}).encode()
    FakeRepo(monkeypatch, files, others=OTHER)
    code, data, _ = make_lock(["org/name"], tmp_path, capsys)
    if local:  # this repo's code and another's: one code_revision cannot pin both
        assert code == 1 and data["status"] == "needs_review"
    else:
        assert code == 0 and data["load_with"]["code_revision"] == OTHER_SHA


@pytest.mark.parametrize("auto_map", ["tok.T", 7, {"AutoModel": {"nested": "x--y.Z"}}])
def test_an_auto_map_shape_the_lock_does_not_know_needs_review(monkeypatch, tmp_path, capsys, auto_map):
    FakeRepo(monkeypatch, {**BASIC, "config.json": json.dumps({"auto_map": auto_map}).encode()})
    code, data, _ = make_lock(["org/name"], tmp_path, capsys)
    assert code == 1 and data["unresolved"][0]["kind"] == "unreadable"


def test_a_hub_answer_that_is_not_an_object_needs_review(monkeypatch, tmp_path, capsys):
    fake = FakeRepo(monkeypatch, {**BASIC, "config.json": json.dumps(
        {"auto_map": {"AutoModel": "other/repo--modeling_x.X"}}).encode()}, others=OTHER)
    real = fake.api_get
    monkeypatch.setattr(hub, "_api_get", lambda url: [] if "other/repo" in url else real(url))
    code, data, _ = make_lock(["org/name"], tmp_path, capsys)
    assert code == 1 and data["unresolved"][0]["kind"] == "remote_code_repo"
