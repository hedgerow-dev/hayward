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


# ---- phase 2: signing with sigstore (hayward[sign]) ------------------------------------
# The two functions that call sigstore are replaced here, so these tests need neither
# the extra nor the network. The real calls are exercised by a live run in CI.

import importlib.util  # noqa: E402

import pytest  # noqa: E402

from hayward import receipt  # noqa: E402

BUNDLE_TYPE = "application/vnd.dev.sigstore.bundle.v0.3+json"


def _fake_bundle(payload: bytes) -> bytes:
    return json.dumps({"mediaType": BUNDLE_TYPE, "fake_payload": payload.decode()}).encode()


@pytest.fixture
def fake_sigstore(monkeypatch):
    signed = {}

    def sign(raw: bytes) -> str:
        signed["raw"] = raw
        return _fake_bundle(raw).decode()

    def open_bundle(raw: bytes, identity: str, issuer: str) -> bytes:
        if identity != "ci@example.com" or issuer != "https://issuer.example":
            raise receipt.SignatureError("the signer is not the one named")
        return json.loads(raw)["fake_payload"].encode()

    monkeypatch.setattr(receipt, "_sigstore_sign", sign)
    monkeypatch.setattr(receipt, "_sigstore_open", open_bundle)
    monkeypatch.setattr(receipt, "_ambient_credential", lambda: True)
    return signed


def test_sign_writes_a_bundle_over_the_exact_receipt_bytes(tmp_path, fake_sigstore):
    d = _models(tmp_path)
    out, _ = _receipt(tmp_path, d)
    assert cli.main(["receipt", "sign", str(out)]) == 0
    bundle = tmp_path / "receipt.intoto.json.sigstore.json"
    assert json.loads(bundle.read_text())["mediaType"] == BUNDLE_TYPE
    assert fake_sigstore["raw"] == out.read_bytes()


def test_a_signed_receipt_verifies_with_the_right_signer(tmp_path, fake_sigstore):
    d = _models(tmp_path)
    out, _ = _receipt(tmp_path, d)
    cli.main(["receipt", "sign", str(out), "-o", str(tmp_path / "b.json")])
    assert cli.main(["receipt", "verify", str(tmp_path / "b.json"), str(d),
                     "--identity", "ci@example.com", "--issuer", "https://issuer.example"]) == 0


def test_the_wrong_signer_fails(tmp_path, fake_sigstore, capsys):
    d = _models(tmp_path)
    out, _ = _receipt(tmp_path, d)
    cli.main(["receipt", "sign", str(out), "-o", str(tmp_path / "b.json")])
    assert cli.main(["receipt", "verify", str(tmp_path / "b.json"), str(d),
                     "--identity", "someone@else.com", "--issuer", "https://issuer.example"]) == 2
    assert "not the one named" in capsys.readouterr().err


def test_a_signed_receipt_still_checks_the_files(tmp_path, fake_sigstore):
    d = _models(tmp_path)
    out, _ = _receipt(tmp_path, d)
    cli.main(["receipt", "sign", str(out), "-o", str(tmp_path / "b.json")])
    (d / "clean.pkl").write_bytes(EVAL)
    assert cli.main(["receipt", "verify", str(tmp_path / "b.json"), str(d),
                     "--identity", "ci@example.com", "--issuer", "https://issuer.example"]) == 1


def test_a_bundle_needs_a_named_signer(tmp_path, fake_sigstore, capsys):
    d = _models(tmp_path)
    out, _ = _receipt(tmp_path, d)
    cli.main(["receipt", "sign", str(out), "-o", str(tmp_path / "b.json")])
    assert cli.main(["receipt", "verify", str(tmp_path / "b.json"), str(d)]) == 2
    assert "--identity" in capsys.readouterr().err


def test_naming_a_signer_for_an_unsigned_receipt_fails(tmp_path, fake_sigstore, capsys):
    d = _models(tmp_path)
    out, _ = _receipt(tmp_path, d)
    assert cli.main(["receipt", "verify", str(out), str(d),
                     "--identity", "ci@example.com", "--issuer", "https://issuer.example"]) == 2
    assert "not signed" in capsys.readouterr().err


def test_signing_on_a_laptop_asks_first_because_the_log_is_public(tmp_path, fake_sigstore, monkeypatch, capsys):
    d = _models(tmp_path)
    out, _ = _receipt(tmp_path, d)
    monkeypatch.setattr(receipt, "_ambient_credential", lambda: False)
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    assert cli.main(["receipt", "sign", str(out)]) == 2
    assert "public" in capsys.readouterr().err
    assert "raw" not in fake_sigstore


def test_sign_refuses_something_that_is_not_a_receipt(tmp_path, fake_sigstore):
    bad = tmp_path / "x.json"
    bad.write_text(json.dumps({"_type": "https://in-toto.io/Statement/v1", "predicateType": "other"}))
    assert cli.main(["receipt", "sign", str(bad)]) == 2


@pytest.mark.skipif(importlib.util.find_spec("sigstore") is not None, reason="the extra is installed")
def test_without_the_extra_signing_says_how_to_get_it(tmp_path, capsys):
    d = _models(tmp_path)
    out, _ = _receipt(tmp_path, d)
    assert cli.main(["receipt", "sign", str(out)]) == 2
    assert 'hayward[sign]' in capsys.readouterr().err


def test_signing_with_no_terminal_to_ask_on_does_not_sign(tmp_path, fake_sigstore, monkeypatch):
    d = _models(tmp_path)
    out, _ = _receipt(tmp_path, d)
    monkeypatch.setattr(receipt, "_ambient_credential", lambda: False)

    def no_terminal(prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", no_terminal)
    assert cli.main(["receipt", "sign", str(out)]) == 2
    assert "raw" not in fake_sigstore


def test_a_ci_identity_problem_is_reported_not_a_crash(tmp_path, fake_sigstore, monkeypatch, capsys):
    d = _models(tmp_path)
    out, _ = _receipt(tmp_path, d)

    def broken():
        raise receipt.SignatureError("no usable CI identity: missing id-token permission")

    monkeypatch.setattr(receipt, "_ambient_credential", broken)
    assert cli.main(["receipt", "sign", str(out)]) == 2
    assert "id-token" in capsys.readouterr().err
