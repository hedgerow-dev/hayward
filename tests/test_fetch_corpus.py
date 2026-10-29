"""scripts/fetch_corpus.py treats its manifest as untrusted data."""

import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "fetch_corpus", Path(__file__).resolve().parent.parent / "scripts" / "fetch_corpus.py")
fetch_corpus = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fetch_corpus)


def _entry(path: str, expected: str = "benign") -> dict:
    return {"path": path, "expected": expected, "repo": "a/b", "revision": "c" * 40,
            "hub_path": "model.bin", "sha256": "0" * 64}


def test_a_path_that_escapes_the_corpus_directory_is_refused(tmp_path, monkeypatch, capsys):
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps([_entry("../escaped.bin")]))
    dest = tmp_path / "corpus"
    dest.mkdir()
    monkeypatch.setattr(fetch_corpus, "download", lambda *a: (_ for _ in ()).throw(AssertionError("fetched")))
    monkeypatch.setattr("sys.argv", ["fetch_corpus.py", str(manifest), str(dest)])
    assert fetch_corpus.main() == 1
    assert "escapes the corpus directory" in capsys.readouterr().err
    assert not (tmp_path / "escaped.bin").exists()


def test_a_hash_mismatch_deletes_the_file(tmp_path, monkeypatch, capsys):
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps([_entry("a__b/model.bin")]))
    dest = tmp_path / "corpus"
    dest.mkdir()
    monkeypatch.setattr(fetch_corpus, "download", lambda url, target: target.write_bytes(b"not the pinned bytes"))
    monkeypatch.setattr(fetch_corpus, "PAUSE_SECONDS", 0)
    monkeypatch.setattr("sys.argv", ["fetch_corpus.py", str(manifest), str(dest)])
    assert fetch_corpus.main() == 1
    assert "sha256 differs" in capsys.readouterr().err
    assert not (dest / "a__b" / "model.bin").exists()


def test_only_filters_by_label(tmp_path, monkeypatch, capsys):
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps([_entry("x/benign.bin"), _entry("x/bad.bin", "malicious")]))
    dest = tmp_path / "corpus"
    dest.mkdir()
    seen = []
    monkeypatch.setattr(fetch_corpus, "download", lambda url, target: seen.append(target.name) or target.write_bytes(b""))
    monkeypatch.setattr(fetch_corpus, "PAUSE_SECONDS", 0)
    monkeypatch.setattr("sys.argv", ["fetch_corpus.py", str(manifest), str(dest), "--only", "malicious"])
    fetch_corpus.main()
    assert seen == ["bad.bin"]
