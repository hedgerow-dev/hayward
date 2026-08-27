"""Tests for hayward.signatures.

All tests inject the directory listing and the file opener, so nothing here
touches a real filesystem except the two that deliberately use tmp_path to
prove the default filesystem path also works. The point under test is
detection plus honest, unverified structural reporting, never verification.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

from hayward.findings import Category, Severity
from hayward.signatures import (
    RULE_ID,
    ArtifactKind,
    find_signature_artifacts,
    signature_findings,
)


def _fake_env(files: dict[str, str]):
    """Build (listing, opener) backed by an in-memory {name: content} map.

    The map is keyed by bare filename; the model and its siblings are assumed
    to live in one directory, which is exactly what find_signature_artifacts
    scans.
    """

    def listing(directory: Path) -> list[str]:
        return list(files.keys())

    def opener(path: Path) -> str:
        try:
            return files[path.name]
        except KeyError:
            raise FileNotFoundError(str(path)) from None

    return listing, opener


def _sigstore_bundle_json(digest_hex: str) -> str:
    """A minimal but realistic Sigstore protobuf-JSON bundle."""
    return json.dumps(
        {
            "mediaType": "application/vnd.dev.sigstore.bundle+json;version=0.3",
            "verificationMaterial": {
                "certificate": {"rawBytes": "MIIB..."},
            },
            "messageSignature": {
                "messageDigest": {"algorithm": "SHA2_256", "digest": digest_hex},
                "signature": "MEUCIQ...",
            },
        }
    )


def _dsse_intoto_envelope(digest_hex: str) -> str:
    """A DSSE envelope whose payload is an in-toto SLSA provenance statement."""
    statement = {
        "_type": "https://in-toto.io/Statement/v1",
        "predicateType": "https://slsa.dev/provenance/v1",
        "subject": [{"name": "model.safetensors", "digest": {"sha256": digest_hex}}],
        "predicate": {"buildType": "https://example/build"},
    }
    payload_b64 = base64.b64encode(json.dumps(statement).encode()).decode()
    return json.dumps(
        {
            "payloadType": "application/vnd.in-toto+json",
            "payload": payload_b64,
            "signatures": [{"sig": "MEQCIA..."}],
        }
    )


def test_sigstore_bundle_detected_and_reported_unverified():
    digest = "a" * 64
    listing, opener = _fake_env(
        {
            "model.safetensors": "<weights>",
            "model.safetensors.sigstore.json": _sigstore_bundle_json(digest),
        }
    )
    artifacts = find_signature_artifacts(
        Path("/models/model.safetensors"), listing=listing, opener=opener
    )
    assert len(artifacts) == 1
    art = artifacts[0]
    assert art.kind == ArtifactKind.SIGSTORE_BUNDLE
    assert art.verified is False
    # The signed digest was parsed structurally out of the bundle.
    assert art.claims.get("message_digest") == digest
    assert art.claims.get("has_certificate") is True

    findings = signature_findings(Path("/models/model.safetensors"), artifacts)
    assert len(findings) == 1
    f = findings[0]
    assert f.rule_id == RULE_ID
    assert f.severity == Severity.INFO
    assert f.category == Category.AI_ML
    # The finding must state plainly that it did NOT verify the artifact.
    assert "not" in f.message.lower() and "verif" in f.message.lower()
    assert "Presence is not proof" in f.message
    assert f.metadata["verified"] is False
    assert f.metadata["artifact_kind"] == ArtifactKind.SIGSTORE_BUNDLE.value


def test_intoto_dsse_envelope_payload_type_parsed():
    digest = "b" * 64
    listing, opener = _fake_env(
        {
            "model.onnx": "<graph>",
            "model.onnx.intoto.jsonl": _dsse_intoto_envelope(digest),
        }
    )
    artifacts = find_signature_artifacts(
        Path("/m/model.onnx"), listing=listing, opener=opener
    )
    assert len(artifacts) == 1
    art = artifacts[0]
    assert art.kind == ArtifactKind.INTOTO_JSONL
    # payloadType is the discriminator and must be surfaced.
    assert art.claims.get("payload_type") == "application/vnd.in-toto+json"
    assert art.claims.get("predicate_type") == "https://slsa.dev/provenance/v1"
    assert art.claims.get("subject_digests") == [{"sha256": digest}]

    findings = signature_findings(Path("/m/model.onnx"), artifacts)
    assert findings[0].message.count("in-toto") >= 1 or "DSSE" in findings[0].message


def test_plain_json_dsse_envelope_promoted_by_content():
    # A generic .json name is only reported when its content proves it is a
    # signer. This exercises the content-sniff promotion path.
    digest = "c" * 64
    listing, opener = _fake_env(
        {
            "model.pt": "<weights>",
            "attestation.json": _dsse_intoto_envelope(digest),
            "notes.json": json.dumps({"unrelated": True}),
        }
    )
    artifacts = find_signature_artifacts(
        Path("/x/model.pt"), listing=listing, opener=opener
    )
    kinds = {a.kind for a in artifacts}
    assert ArtifactKind.DSSE_ENVELOPE in kinds
    # The unrelated JSON must NOT be reported as an artifact.
    assert all(a.path.name != "notes.json" for a in artifacts)


def test_detached_sig_and_model_sig_manifest():
    listing, opener = _fake_env(
        {
            "model.safetensors": "<weights>",
            "model.safetensors.sig": "<opaque signature bytes>",
            "model.sig": "<manifest bytes>",
        }
    )
    artifacts = find_signature_artifacts(
        Path("/d/model.safetensors"), listing=listing, opener=opener
    )
    kinds = {a.kind for a in artifacts}
    # model.sig is the more specific manifest, not a bare detached signature.
    assert ArtifactKind.MODEL_SIG_MANIFEST in kinds
    assert ArtifactKind.DETACHED_SIG in kinds


def test_no_artifacts_yields_nothing():
    listing, opener = _fake_env(
        {
            "model.safetensors": "<weights>",
            "config.json": json.dumps({"hidden_size": 768}),
            "README.md": "# model",
        }
    )
    artifacts = find_signature_artifacts(
        Path("/n/model.safetensors"), listing=listing, opener=opener
    )
    assert artifacts == []
    assert signature_findings(Path("/n/model.safetensors"), artifacts) == []


def test_malformed_bundle_json_handled_without_raising():
    # A file named like a bundle but containing broken JSON must still be
    # detected (by name) and must not raise; claims are simply empty.
    listing, opener = _fake_env(
        {
            "model.safetensors": "<weights>",
            "model.safetensors.sigstore.json": "{ this is not valid json ",
        }
    )
    artifacts = find_signature_artifacts(
        Path("/e/model.safetensors"), listing=listing, opener=opener
    )
    assert len(artifacts) == 1
    art = artifacts[0]
    assert art.kind == ArtifactKind.SIGSTORE_BUNDLE
    assert art.claims == {}
    # A finding is still emitted, still INFO, still explicitly unverified.
    findings = signature_findings(Path("/e/model.safetensors"), artifacts)
    assert len(findings) == 1
    assert findings[0].severity == Severity.INFO
    assert findings[0].metadata["verified"] is False


def test_malformed_dsse_payload_does_not_raise():
    # Envelope with a non-base64 payload: the type is still read, the inner
    # statement just cannot be decoded. No exception.
    envelope = json.dumps(
        {"payloadType": "application/vnd.in-toto+json", "payload": "!!!not-base64!!!"}
    )
    listing, opener = _fake_env(
        {"model.bin": "<w>", "model.bin.intoto.jsonl": envelope}
    )
    artifacts = find_signature_artifacts(
        Path("/p/model.bin"), listing=listing, opener=opener
    )
    assert artifacts[0].claims.get("payload_type") == "application/vnd.in-toto+json"
    # No subject digests, because the payload could not be decoded.
    assert "subject_digests" not in artifacts[0].claims


def test_defaults_hit_the_filesystem(tmp_path: Path):
    # Prove the default listing/opener work against real files too.
    model = tmp_path / "model.safetensors"
    model.write_text("<weights>")
    (tmp_path / "model.safetensors.sigstore.json").write_text(
        _sigstore_bundle_json("d" * 64)
    )
    artifacts = find_signature_artifacts(model)
    assert len(artifacts) == 1
    assert artifacts[0].kind == ArtifactKind.SIGSTORE_BUNDLE
    assert artifacts[0].claims.get("message_digest") == "d" * 64


def test_directory_with_no_siblings_is_safe(tmp_path: Path):
    # A model whose parent cannot be listed (points nowhere) yields nothing,
    # not an exception.
    artifacts = find_signature_artifacts(tmp_path / "missing" / "model.pt")
    assert artifacts == []
