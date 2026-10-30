"""Issue #38: scan receipts, written by `hayward scan --receipt` and checked by
`hayward receipt verify`.

A receipt is an unsigned in-toto Statement: evidence of what the scanner saw in
each file, by hash. Signing is left to Sigstore tooling (GitHub's attest action,
cosign). It is analysis evidence, not a safety certificate.
"""

from __future__ import annotations

import hashlib
import json
import pickle

from hayward import __version__, cli
from hayward.receipt import PREDICATE_TYPE

# builtins.eval("40+2"), protocol 4: CRITICAL, harmless if ever executed.
EVAL = b"\x80\x04\x8c\x08builtins\x8c\x04eval\x93\x8c\x0440+2\x85R."
LFS = b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"a" * 64 + b"\nsize 12345\n"


def _models(tmp_path):
    d = tmp_path / "models"
    d.mkdir()
    (d / "clean.pkl").write_bytes(pickle.dumps({"weights": [1, 2, 3]}, protocol=4))
    (d / "bad.pkl").write_bytes(EVAL)
    (d / "pointer.bin").write_bytes(LFS)
    (d / "README.md").write_text("not a model")
    return d


def _receipt(tmp_path, d, *extra):
    out = tmp_path / "receipt.intoto.json"
    cli.main(["scan", str(d), "--receipt", str(out), "-f", "json", "-o", str(tmp_path / "r.json"), *extra])
    return out, json.loads(out.read_text())


def test_the_receipt_is_an_in_toto_statement_over_every_scanned_file(tmp_path):
    d = _models(tmp_path)
    _, statement = _receipt(tmp_path, d)
    assert statement["_type"] == "https://in-toto.io/Statement/v1"
    assert statement["predicateType"] == PREDICATE_TYPE
    subjects = {s["name"]: s["digest"]["sha256"] for s in statement["subject"]}
    assert set(subjects) == {"clean.pkl", "bad.pkl", "pointer.bin"}  # README.md is not a model
    for name, digest in subjects.items():
        assert digest == hashlib.sha256((d / name).read_bytes()).hexdigest()


def test_the_predicate_records_each_files_verdict_and_what_was_not_read(tmp_path):
    d = _models(tmp_path)
    _, statement = _receipt(tmp_path, d)
    predicate = statement["predicate"]
    assert predicate["scanner"]["version"] == __version__
    files = predicate["files"]
    assert files["clean.pkl"]["verdict"] == "clean"
    assert files["bad.pkl"]["verdict"] == "flagged"
    assert "MFV-PICKLE-001" in files["bad.pkl"]["rules"]
    assert files["pointer.bin"]["verdict"] == "not_verified"
    assert files["pointer.bin"]["coverage_gaps"] == ["MFV-LFS-001"]
    assert "not a safety certificate" in predicate["note"]


def test_the_receipt_is_scanner_evidence_before_any_acceptance_policy(tmp_path):
    # An allowlist is the organisation's acceptance policy, kept out of the evidence.
    d = _models(tmp_path)
    digest = hashlib.sha256(EVAL).hexdigest()
    allow = tmp_path / "allow.json"
    allow.write_text(json.dumps([
        {"sha256": digest, "rule_id": "MFV-PICKLE-001", "justification": "test"}]))
    _, statement = _receipt(tmp_path, d, "--allowlist", str(allow))
    assert statement["predicate"]["files"]["bad.pkl"]["verdict"] == "flagged"


def test_verify_passes_on_the_same_files(tmp_path):
    d = _models(tmp_path)
    out, _ = _receipt(tmp_path, d)
    assert cli.main(["receipt", "verify", str(out), str(d)]) == 0


def test_verify_fails_on_a_changed_file(tmp_path, capsys):
    d = _models(tmp_path)
    out, _ = _receipt(tmp_path, d)
    (d / "clean.pkl").write_bytes(EVAL)
    assert cli.main(["receipt", "verify", str(out), str(d)]) == 1
    assert "changed: clean.pkl" in capsys.readouterr().out


def test_verify_fails_on_a_missing_file(tmp_path, capsys):
    d = _models(tmp_path)
    out, _ = _receipt(tmp_path, d)
    (d / "bad.pkl").unlink()
    assert cli.main(["receipt", "verify", str(out), str(d)]) == 1
    assert "missing: bad.pkl" in capsys.readouterr().out


def test_verify_fails_on_a_model_file_the_receipt_does_not_cover(tmp_path, capsys):
    d = _models(tmp_path)
    out, _ = _receipt(tmp_path, d)
    (d / "new.pkl").write_bytes(EVAL)
    (d / "notes.txt").write_text("not a model, so not required")
    assert cli.main(["receipt", "verify", str(out), str(d)]) == 1
    assert capsys.readouterr().out.count("not in the receipt") == 1


def test_verify_rejects_something_that_is_not_a_receipt(tmp_path):
    d = _models(tmp_path)
    bad = tmp_path / "x.json"
    bad.write_text(json.dumps({"_type": "https://in-toto.io/Statement/v1", "predicateType": "other"}))
    assert cli.main(["receipt", "verify", str(bad), str(d)]) == 2


def test_a_subject_name_cannot_escape_the_directory(tmp_path):
    d = _models(tmp_path)
    out, statement = _receipt(tmp_path, d)
    statement["subject"].append({"name": "../outside.pkl", "digest": {"sha256": "0" * 64}})
    out.write_text(json.dumps(statement))
    assert cli.main(["receipt", "verify", str(out), str(d)]) == 2


def test_a_malformed_digest_is_a_mismatch_not_a_crash(tmp_path):
    d = _models(tmp_path)
    out, statement = _receipt(tmp_path, d)
    statement["subject"][0]["digest"] = "not an object"
    out.write_text(json.dumps(statement))
    assert cli.main(["receipt", "verify", str(out), str(d)]) == 1
