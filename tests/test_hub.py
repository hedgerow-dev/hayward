"""`hayward hub org/name`: fetch a Hub repo's model files into a temp folder and scan them.

The network is faked: `hub._api_get` and `hub._download` are replaced, so these run offline.
"""

import hashlib
import io
import pickle
import tarfile
import zipfile
from pathlib import Path

import pytest

from hayward import hub

# builtins.eval("40+2") as a protocol 4 pickle: CRITICAL, harmless if ever run.
EVIL = (b"\x80\x04\x8c\x08builtins\x8c\x04eval\x93\x8c\x0440+2\x85R.")
CLEAN = pickle.dumps({"w": [1.0, 2.0]}, protocol=2)


class FakeHub:
    def __init__(self, monkeypatch, files: dict[str, bytes], *, sha="a" * 40, sizes=None, lfs=None):
        self.files, self.sha, self.sizes, self.lfs = files, sha, sizes or {}, lfs or {}
        self.downloaded: list[str] = []
        self.range_bytes = 0
        self.range_calls = 0
        monkeypatch.setattr(hub, "_api_get", self.api_get)
        monkeypatch.setattr(hub, "_download", self.download)
        monkeypatch.setattr(hub, "_fetch_range", self.fetch_range)

    def api_get(self, url: str):
        assert url.startswith("https://huggingface.co/api/models/")
        siblings = []
        for name, data in self.files.items():
            entry = {"rfilename": name, "size": self.sizes.get(name, len(data))}
            if name in self.lfs:
                entry["lfs"] = {"sha256": self.lfs[name]}
            siblings.append(entry)
        return {"sha": self.sha, "siblings": siblings}

    def fetch_range(self, url: str, start: int, stop: int, size: int) -> bytes:
        assert url.startswith("https://huggingface.co/")
        name = url.split(f"/resolve/{self.sha}/", 1)[1]
        data = self.files[name]
        assert 0 <= start < stop <= len(data), (start, stop, len(data))
        self.range_calls += 1
        self.range_bytes += stop - start
        return data[start:stop]

    def download(self, url: str, dest: Path, max_bytes: int):
        assert url.startswith("https://huggingface.co/")
        name = url.split(f"/resolve/{self.sha}/", 1)[1]
        self.downloaded.append(name)
        dest.write_bytes(self.files[name])


def run(argv, capsys):
    code = hub.main(argv)
    out = capsys.readouterr()
    return code, out.out, out.err


def test_only_scannable_files_are_fetched(monkeypatch, capsys):
    fake = FakeHub(monkeypatch, {
        "model.pkl": CLEAN, "README.md": b"hi", "config.json": b"{}", "tokenizer_config.json": b"{}",
        "other.json": b"{}", "weights.safetensors": b"x", "cat.png": b"x",
    })
    run(["org/name", "--quiet"], capsys)
    assert sorted(fake.downloaded) == ["config.json", "model.pkl", "tokenizer_config.json", "weights.safetensors"]


def test_a_malicious_checkpoint_fails_the_scan(monkeypatch, capsys):
    FakeHub(monkeypatch, {"sub/model.pkl": EVIL})
    code, out, _ = run(["org/name", "--quiet"], capsys)
    assert code == 1
    assert "sub/model.pkl" in out.replace("\\", "/") and "MFV-PICKLE-001" in out  # Windows prints sub\\model.pkl


def test_a_clean_repo_exits_zero(monkeypatch, capsys):
    FakeHub(monkeypatch, {"model.pkl": CLEAN})
    code, _, _ = run(["org/name", "--quiet"], capsys)
    assert code == 0


def test_a_file_over_the_cap_is_named_as_not_scanned(monkeypatch, capsys):
    fake = FakeHub(monkeypatch, {"big.pt": CLEAN, "small.pkl": CLEAN}, sizes={"big.pt": 900 * 1024 * 1024})
    code, _, err = run(["org/name", "--max-file-mb", "50", "--quiet"], capsys)
    assert fake.downloaded == ["small.pkl"]
    assert "NOT SCANNED" in err and "big.pt" in err
    assert code == 0  # findings decide the exit code, as for every other scan


@pytest.mark.parametrize("name", ["../escape.pkl", "/abs.pkl", "a/../../b.pkl", "a//b.pkl", ".hidden/../../x.pt"])
def test_unsafe_file_names_from_the_api_are_never_written(monkeypatch, tmp_path, capsys, name):
    fake = FakeHub(monkeypatch, {name: CLEAN, "ok.pkl": CLEAN})
    code, _, err = run(["org/name", "--quiet"], capsys)
    assert name not in fake.downloaded
    assert "ok.pkl" in fake.downloaded
    assert "refusing" in err


@pytest.mark.parametrize("repo", ["", "../x", "a/b/c", "a b/c", "org", "org/", "/name", "org/na me", "http://evil/x", "org/name\n"])
def test_malformed_repo_ids_are_refused_before_any_request(monkeypatch, capsys, repo):
    monkeypatch.setattr(hub, "_api_get", lambda url: pytest.fail("made a request"))
    code, _, err = run([repo, "--quiet"], capsys)
    assert code == 2 and "repo id" in err


@pytest.mark.parametrize("rev", ["../main", "a b", "x" * 200, "a/b", "main\n"])
def test_malformed_revisions_are_refused(monkeypatch, capsys, rev):
    monkeypatch.setattr(hub, "_api_get", lambda url: pytest.fail("made a request"))
    code, _, err = run(["org/name", "--revision", rev, "--quiet"], capsys)
    assert code == 2 and "revision" in err


def test_downloads_are_pinned_to_the_commit_the_api_returned(monkeypatch, capsys):
    fake = FakeHub(monkeypatch, {"model.pkl": CLEAN}, sha="b" * 40)
    seen = []
    orig = fake.download
    monkeypatch.setattr(hub, "_download", lambda url, dest, mb: (seen.append(url), orig(url, dest, mb)))
    run(["org/name", "--revision", "main", "--quiet"], capsys)
    assert seen and f"/resolve/{'b' * 40}/model.pkl" in seen[0]


def test_an_lfs_hash_mismatch_is_an_error(monkeypatch, capsys):
    FakeHub(monkeypatch, {"model.pkl": CLEAN}, lfs={"model.pkl": "0" * 64})
    code, _, err = run(["org/name", "--quiet"], capsys)
    assert code == 2 and "sha256" in err


def test_a_matching_lfs_hash_passes(monkeypatch, capsys):
    FakeHub(monkeypatch, {"model.pkl": CLEAN}, lfs={"model.pkl": hashlib.sha256(CLEAN).hexdigest()})
    code, _, _ = run(["org/name", "--quiet"], capsys)
    assert code == 0


def test_scan_flags_pass_through(monkeypatch, capsys):
    FakeHub(monkeypatch, {"model.pkl": EVIL})
    code, out, _ = run(["org/name", "--quiet", "--format", "json", "--fail-on", "critical"], capsys)
    assert code == 1 and out.lstrip().startswith(("{", "["))


def test_a_legacy_tar_checkpoint_in_a_repo_is_flagged(monkeypatch, capsys):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as t:
        info = tarfile.TarInfo("storages")
        info.size = len(EVIL)
        t.addfile(info, io.BytesIO(EVIL))
    FakeHub(monkeypatch, {"pytorch_model.bin": buf.getvalue()})
    code, out, _ = run(["org/name", "--quiet"], capsys)
    assert code == 1 and "pytorch_model.bin" in out


def test_the_token_goes_only_to_the_hub(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test_value")
    req = hub._request("https://huggingface.co/api/models/a/b")
    assert "Authorization" not in req.headers  # unredirected: not copied to a CDN redirect
    assert req.unredirected_hdrs.get("Authorization") == "Bearer hf_test_value"
    with pytest.raises(ValueError):
        hub._request("https://evil.example/api/models/a/b")
    with pytest.raises(ValueError):
        hub._request("http://huggingface.co/api/models/a/b")


def test_a_clean_report_names_the_repo_not_the_temp_folder(monkeypatch, capsys):
    FakeHub(monkeypatch, {"model.pkl": CLEAN}, sha="c" * 40)
    code, out, _ = run(["org/name", "--quiet"], capsys)
    assert code == 0
    assert out.strip() == f"No findings in hf://org/name@{'c' * 12}"
    assert "hayward-hub-" not in out


# ---- range reads: big zip checkpoints, only the pickle member is fetched ----------

FILLER = 3_000_000  # a stand-in for tensor data: the whole "checkpoint" is bigger than the 1 MB cap


def zip_checkpoint(pickle_bytes: bytes, member="archive/data.pkl", extra=()) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        z.writestr(member, pickle_bytes)
        for name, data in extra:
            z.writestr(name, data)
        z.writestr("archive/data/0", b"\0" * FILLER)
    return buf.getvalue()


BIG = ["org/name", "--max-file-mb", "1", "--quiet"]


def test_a_big_zip_checkpoint_is_range_read_and_flagged(monkeypatch, capsys):
    fake = FakeHub(monkeypatch, {"pytorch_model.bin": zip_checkpoint(EVIL)})
    code, out, err = run(BIG, capsys)
    assert code == 1 and "MFV-PICKLE-001" in out and "pytorch_model.bin" in out
    assert fake.downloaded == []                       # never fetched whole
    assert 0 < fake.range_bytes < 200_000              # a small slice of a 3 MB file
    assert "pickle member" in err and "pytorch_model.bin" in err
    assert "NOT SCANNED" not in err


def test_a_big_clean_zip_checkpoint_passes_with_a_note(monkeypatch, capsys):
    FakeHub(monkeypatch, {"model.pt": zip_checkpoint(CLEAN)})
    code, _, err = run(BIG, capsys)
    assert code == 0
    assert "only the pickle" in err and "not the tensor data" in err


def test_a_big_legacy_tar_is_not_scanned_and_says_why(monkeypatch, capsys):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as t:
        info = tarfile.TarInfo("storages")
        info.size = FILLER
        t.addfile(info, io.BytesIO(b"\0" * FILLER))
    FakeHub(monkeypatch, {"pytorch_model.bin": buf.getvalue()})
    code, _, err = run(BIG, capsys)
    assert "NOT SCANNED" in err and "pytorch_model.bin" in err and "not a zip" in err


def test_other_big_formats_are_not_range_read(monkeypatch, capsys):
    fake = FakeHub(monkeypatch, {"model.safetensors": b"x" * FILLER})
    _, _, err = run(BIG, capsys)
    assert fake.range_calls == 0
    assert "NOT SCANNED" in err and "model.safetensors" in err


def test_a_pickle_member_over_the_budget_is_not_scanned(monkeypatch, capsys):
    monkeypatch.setattr(hub, "RANGE_BUDGET", 20_000)
    FakeHub(monkeypatch, {"model.pt": zip_checkpoint(CLEAN + b"\0" * 100_000)})
    _, _, err = run(BIG, capsys)
    assert "NOT SCANNED" in err and "budget" in err


@pytest.mark.parametrize("member", ["../escape.pkl", "/abs.pkl", "a/../../b.pkl"])
def test_an_unsafe_zip_member_name_is_refused(monkeypatch, capsys, member):
    FakeHub(monkeypatch, {"model.pt": zip_checkpoint(EVIL, member=member)})
    code, out, err = run(BIG, capsys)
    assert "NOT SCANNED" in err and "member name" in err
    assert "MFV-PICKLE-001" not in out


def test_too_many_pickle_members_are_not_scanned(monkeypatch, capsys):
    extra = [(f"m/{i}.pkl", CLEAN) for i in range(hub.MAX_PICKLE_MEMBERS + 1)]
    FakeHub(monkeypatch, {"model.pt": zip_checkpoint(CLEAN, extra=extra)})
    _, _, err = run(BIG, capsys)
    assert "NOT SCANNED" in err and "pickle members" in err


def test_a_zip_with_no_pickle_member_is_not_scanned(monkeypatch, capsys):
    FakeHub(monkeypatch, {"model.pt": zip_checkpoint(CLEAN, member="archive/readme.txt")})
    _, _, err = run(BIG, capsys)
    assert "NOT SCANNED" in err and "no pickle member" in err


def test_a_damaged_zip_is_not_scanned_and_does_not_crash(monkeypatch, capsys):
    data = bytearray(zip_checkpoint(CLEAN))
    data[-30:] = b"\xff" * 30  # corrupt the end-of-central-directory record
    FakeHub(monkeypatch, {"model.pt": bytes(data)})
    code, _, err = run(BIG, capsys)
    assert code in (0, 2) and "NOT SCANNED" in err


class FakeResponse:
    def __init__(self, body=b"", status=206, headers=None, url="https://cdn.example/x"):
        self.body, self.status, self.headers, self._url = body, status, headers or {}, url

    def geturl(self):
        return self._url

    def read(self, n=-1):
        return self.body if n < 0 else self.body[:n]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _range_response(monkeypatch, resp):
    monkeypatch.setattr(hub, "_open", lambda url, byte_range=None: resp)


def test_fetch_range_accepts_an_exact_partial_response(monkeypatch):
    _range_response(monkeypatch, FakeResponse(b"abcd", headers={"Content-Range": "bytes 10-13/100"}))
    assert hub._fetch_range("https://huggingface.co/x", 10, 14, 100) == b"abcd"


@pytest.mark.parametrize("resp,why", [
    (FakeResponse(b"abcd", status=200), "range"),
    (FakeResponse(b"abcd", headers={"Content-Range": "bytes 0-3/100"}), "range"),
    (FakeResponse(b"abcd", headers={"Content-Range": "bytes 10-13/999"}), "range"),
    (FakeResponse(b"abc", headers={"Content-Range": "bytes 10-13/100"}), "range"),
    (FakeResponse(b"abcd", headers={"Content-Range": "bytes 10-13/100"}, url="http://cdn.example/x"), "https"),
])
def test_fetch_range_refuses_anything_but_the_exact_range_over_https(monkeypatch, resp, why):
    _range_response(monkeypatch, resp)
    with pytest.raises(hub.RangeSkip):
        hub._fetch_range("https://huggingface.co/x", 10, 14, 100)


def test_compressed_tars_with_a_double_suffix_are_fetched(monkeypatch, capsys):
    fake = FakeHub(monkeypatch, {"data.tar.gz": b"x", "b.tar.xz": b"x", "c.tgz": b"x", "notes.gz": b"x"})
    run(["org/name", "--quiet"], capsys)
    assert sorted(fake.downloaded) == ["b.tar.xz", "c.tgz", "data.tar.gz"]
