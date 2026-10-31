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
        "generation_config.json": b"{}", "weights.safetensors": b"x", "cat.png": b"x",
        "chat_template.jinja": b"{{ m }}", "merges.txt": b"a b", "notes.txt": b"x", "modeling_x.py": b"x",
    })
    run(["org/name", "--quiet"], capsys)
    assert sorted(fake.downloaded) == ["chat_template.jinja", "config.json", "generation_config.json",
                                       "merges.txt", "model.pkl", "tokenizer_config.json", "weights.safetensors"]


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
    fake = FakeHub(monkeypatch, {"model.gguf": b"x" * FILLER})
    _, _, err = run(BIG, capsys)
    assert fake.range_calls == 0
    assert "NOT SCANNED" in err and "model.gguf" in err


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


# ---- big SafeTensors: only the header is fetched ----------------------------------------

import json  # noqa: E402
import struct  # noqa: E402

GB = 3 * 1024 ** 3


def safetensors(metadata=None):
    """The header of a 3 GB SafeTensors file, plus a few bytes of its data."""
    header = {"w": {"dtype": "U8", "shape": [0], "data_offsets": [0, 0]}}
    if metadata:
        header["__metadata__"] = metadata
    for _ in range(2):  # the data size depends on the header length, which depends on it
        raw = json.dumps(header).encode()
        data_bytes = GB - 8 - len(raw)
        header["w"].update(shape=[data_bytes], data_offsets=[0, data_bytes])
    raw = json.dumps(header).encode()
    assert 8 + len(raw) + header["w"]["data_offsets"][1] == GB
    return struct.pack("<Q", len(raw)) + raw + b"\0" * 4096


def big_safetensors_hub(monkeypatch, data: bytes, lfs=True):
    sha = {"model.safetensors": "b" * 64} if lfs else {}
    return FakeHub(monkeypatch, {"model.safetensors": data}, sizes={"model.safetensors": GB}, lfs=sha)


def test_a_big_safetensors_file_is_read_by_its_header(monkeypatch, capsys):
    fake = big_safetensors_hub(monkeypatch, safetensors())
    code, _, err = run(BIG, capsys)
    assert code == 0
    assert fake.downloaded == []
    assert 0 < fake.range_bytes < 10_000
    assert "NOT SCANNED" not in err and "header" in err and "model.safetensors" in err


def test_a_big_safetensors_file_with_bad_metadata_is_caught(monkeypatch, capsys):
    big_safetensors_hub(monkeypatch, safetensors(metadata={"__reduce__": "os.system"}))
    code, out, _ = run(BIG, capsys)
    assert code == 1 and "MFV-ST-005" in out


def _receipt(tmp_path, capsys):
    out = tmp_path / "r.json"
    run([*BIG, "--receipt", str(out), "-f", "json", "-o", str(tmp_path / "rep.json")], capsys)
    return json.loads(out.read_text())


def test_a_receipt_uses_the_hubs_hash_for_a_header_read_file(monkeypatch, tmp_path, capsys):
    big_safetensors_hub(monkeypatch, safetensors())
    statement = _receipt(tmp_path, capsys)
    assert statement["subject"] == [{"name": "model.safetensors", "digest": {"sha256": "b" * 64}}]
    entry = statement["predicate"]["files"]["model.safetensors"]
    assert entry["read"] == "header" and entry["hash_source"] == "hub"


def test_a_receipt_uses_the_hubs_hash_for_a_range_read_checkpoint(monkeypatch, tmp_path, capsys):
    FakeHub(monkeypatch, {"model.pt": zip_checkpoint(CLEAN)}, lfs={"model.pt": "c" * 64})
    statement = _receipt(tmp_path, capsys)
    assert statement["subject"] == [{"name": "model.pt", "digest": {"sha256": "c" * 64}}]
    assert statement["predicate"]["files"]["model.pt"]["read"] == "pickle members"


def test_a_partly_read_file_with_no_hub_hash_is_left_out_of_the_receipt(monkeypatch, tmp_path, capsys):
    FakeHub(monkeypatch, {"model.pt": zip_checkpoint(CLEAN), "small.pkl": CLEAN})
    statement = _receipt(tmp_path, capsys)
    assert [s["name"] for s in statement["subject"]] == ["small.pkl"]
    assert "model.pt" not in statement["predicate"]["files"]


def test_a_malformed_hub_hash_is_not_put_in_a_receipt(monkeypatch, tmp_path, capsys):
    FakeHub(monkeypatch, {"model.pt": zip_checkpoint(CLEAN), "small.pkl": CLEAN}, lfs={"model.pt": "not-a-hash\n"})
    statement = _receipt(tmp_path, capsys)
    assert [s["name"] for s in statement["subject"]] == ["small.pkl"]


def test_a_stale_receipt_is_never_rewritten(monkeypatch, tmp_path, capsys):
    stale = tmp_path / "r.json"
    stale.write_text("not a receipt")
    FakeHub(monkeypatch, {"model.pt": zip_checkpoint(CLEAN)}, lfs={"model.pt": "c" * 64})
    monkeypatch.setattr(hub, "scan_main", lambda argv, label=None: 2)  # the scan fails before writing
    run([*BIG, "--receipt", str(stale)], capsys)
    assert not stale.exists()


def test_the_last_receipt_flag_is_the_one_used(monkeypatch, tmp_path, capsys):
    FakeHub(monkeypatch, {"model.pt": zip_checkpoint(CLEAN)}, lfs={"model.pt": "c" * 64})
    first, last = tmp_path / "a.json", tmp_path / "b.json"
    run([*BIG, "--receipt", str(first), "--receipt", str(last), "-f", "json", "-o", str(tmp_path / "x.json")], capsys)
    assert json.loads(last.read_text())["subject"][0]["digest"]["sha256"] == "c" * 64


def test_a_hub_receipt_names_the_repo_and_commit(monkeypatch, tmp_path, capsys):
    FakeHub(monkeypatch, {"model.pkl": CLEAN}, sha="d" * 40)
    statement = _receipt(tmp_path, capsys)
    assert statement["predicate"]["source"] == {"host": "huggingface.co", "repo": "org/name", "revision": "d" * 40}


def test_a_chat_template_json_is_fetched(monkeypatch, capsys):
    fake = FakeHub(monkeypatch, {"chat_template.json": b"{}", "model.pkl": CLEAN})
    run(["org/name", "--quiet"], capsys)
    assert sorted(fake.downloaded) == ["chat_template.json", "model.pkl"]


def test_files_are_fetched_several_at_a_time_and_reported_in_hub_order(monkeypatch, tmp_path):
    import threading
    import time

    names = [f"model-{i:05d}.pkl" for i in range(20)]
    fake = FakeHub(monkeypatch, {name: CLEAN for name in names})
    lock, active, peak = threading.Lock(), [0], [0]
    orig = fake.download

    def slow_download(url, dest, max_bytes):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.02)
        orig(url, dest, max_bytes)
        with lock:
            active[0] -= 1

    monkeypatch.setattr(hub, "_download", slow_download)
    got = hub.fetch("org/name", "main", tmp_path, 1 << 20)
    assert got["fetched"] == names
    assert 1 < peak[0] <= hub.FETCH_WORKERS


def test_names_that_differ_only_by_case_are_refused(monkeypatch, tmp_path):
    fake = FakeHub(monkeypatch, {"Model.pkl": EVIL, "model.pkl": CLEAN, "other.pkl": CLEAN})
    got = hub.fetch("org/name", "main", tmp_path, 1 << 20)
    assert sorted(got["refused"]) == ["Model.pkl", "model.pkl"]
    assert fake.downloaded == ["other.pkl"]


def test_a_hash_mismatch_stops_without_fetching_the_rest(monkeypatch, tmp_path):
    names = [f"m-{i:03d}.pkl" for i in range(200)]
    fake = FakeHub(monkeypatch, {n: CLEAN for n in names}, lfs={names[0]: "0" * 64})
    with pytest.raises(hub.HubError):
        hub.fetch("org/name", "main", tmp_path, 1 << 20)
    assert len(fake.downloaded) < len(names)


def test_names_that_differ_only_by_unicode_form_are_refused(monkeypatch, tmp_path):
    composed, decomposed = "café.pkl", "café.pkl"
    FakeHub(monkeypatch, {composed: EVIL, decomposed: CLEAN})
    got = hub.fetch("org/name", "main", tmp_path, 1 << 20)
    assert sorted(got["refused"]) == sorted([composed, decomposed])
