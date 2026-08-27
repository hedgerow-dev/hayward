"""Model-signature and attestation awareness.

The signing ecosystem for ML models is young and fragmented: Sigstore
bundles, detached `.sig` files, in-toto / SLSA provenance carried in DSSE
envelopes, and the model-transparency project's `model.sig` manifest all
coexist. Hayward does NO network I/O and holds NO trust root, so this module
deliberately stops short of cryptographic verification. It answers one honest,
bounded question: does this model ship a signature or attestation alongside
it, and what does that artifact *structurally claim*?

Presence is not proof. A `.sig` file next to a model tells you someone
produced a signature; it does not tell you the signature is valid, that it was
made by anyone you trust, or even that it covers this model's bytes. Verifying
any of that needs a trust root and (for Sigstore's transparency log and
timestamping) network access, both out of scope here. Every finding this
module emits says so in plain words, and the metadata never carries a
`verified` flag set to anything but False.

The value is the inventory and the structural read: "this model does (or does
not) carry an attestation, and here is the digest / identity / predicate it
names, unverified." That is useful on its own, and it is all a
no-network scanner can truthfully offer.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import stat
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from hayward.findings import Category, Finding, Severity

# The single rule this module emits. INFORMATIONAL: it reports that a signature
# artifact exists, never that verification succeeded or failed. It is NOT a
# coverage gap (analysis of the model completed), so it stays out of
# COVERAGE_RULE_IDS deliberately.
RULE_ID = "MFV-SIG-001"

# Cap on how many bytes of a candidate artifact we will read for structural
# parsing. Signature and attestation metadata are tiny (kilobytes); a file
# claiming to be one but weighing megabytes is either not what it says or not
# worth loading into memory. We still report its presence, just without claims.
_MAX_READ_BYTES = 5_000_000

# The mediaType string a Sigstore protobuf bundle carries. The version suffix
# (`;version=0.x`) changes over releases, so we match on the stable prefix.
_SIGSTORE_BUNDLE_MEDIA_PREFIX = "application/vnd.dev.sigstore.bundle"

# The payloadType a DSSE envelope carrying an in-toto statement declares. This
# is the discriminator that separates SLSA provenance from an arbitrary JSON
# file that merely happens to have a "payload" key.
_INTOTO_PAYLOAD_TYPE = "application/vnd.in-toto+json"


class ArtifactKind(str, Enum):
    """What flavour of signing artifact a sibling file appears to be.

    Classification is by filename first and file content second. The content
    read only ever *refines* or *confirms* a name-based guess; it never fails
    the detection, because a malformed or unreadable artifact is still an
    artifact worth reporting.
    """

    DETACHED_SIG = "detached_signature"          # a bare <model>.sig blob
    SIGSTORE_BUNDLE = "sigstore_bundle"          # *.sigstore / *.sigstore.json
    INTOTO_JSONL = "intoto_dsse_jsonl"           # *.intoto.jsonl (DSSE lines)
    DSSE_ENVELOPE = "dsse_envelope"              # JSON DSSE with in-toto payload
    MODEL_SIG_MANIFEST = "model_sig_manifest"    # model-transparency model.sig


@dataclass
class SignatureArtifact:
    """One detected signing artifact and whatever it structurally claims.

    `claims` holds fields parsed out of the artifact (a signed digest, a
    certificate identity, a SLSA predicate type). It is best-effort and may be
    empty when the file is unreadable or malformed. `verified` is always False
    and exists only to make the no-verification stance explicit to any
    consumer that inspects the object rather than the finding text.
    """

    kind: ArtifactKind
    path: Path
    claims: dict[str, Any] = field(default_factory=dict)
    verified: bool = False


# A directory-listing function: given a directory Path, return the names it
# contains. Injectable so tests need no real filesystem.
Listing = Callable[[Path], list[str]]
# A file opener: given a Path, return its text content. Injectable for the same
# reason. May raise (missing / binary / oversized); callers must tolerate that.
Opener = Callable[[Path], str]


def _default_listing(directory: Path) -> list[str]:
    """List a directory, returning [] when it is missing or not a directory.

    Detection must never raise merely because the model's parent cannot be
    listed; an empty inventory is the correct answer in that case.
    """
    try:
        return os.listdir(directory)
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return []


def _default_opener(path: Path) -> str:
    """Read a small text file, refusing anything over the size cap.

    `errors="replace"` keeps a stray non-UTF-8 byte from raising: we would
    rather parse what we can of a slightly malformed artifact than treat a
    decode error as a hard failure.
    """
    st = path.stat()
    # Only a regular file has a meaningful size. A FIFO or device named like an
    # artifact reports size 0, sails past the cap, and then read_text blocks
    # forever or streams without end. Refuse anything that is not a plain file.
    if not stat.S_ISREG(st.st_mode):
        raise ValueError("artifact is not a regular file")
    if st.st_size > _MAX_READ_BYTES:
        raise ValueError("artifact exceeds structural-parse size cap")
    return path.read_text(encoding="utf-8", errors="replace")


def _classify_by_name(name: str) -> ArtifactKind | None:
    """Guess an artifact kind from a filename alone, or None if it is not one.

    Name-based classification is what makes detection work without reading a
    single byte. Content parsing later confirms and enriches, but the presence
    signal comes from here. Order matters: the most specific patterns win, so
    `model.sig` is a manifest rather than a bare detached signature.
    """
    lower = name.lower()
    # model-transparency writes a manifest named exactly `model.sig`. It is
    # more specific than the generic `.sig` case, so test it first.
    if lower == "model.sig":
        return ArtifactKind.MODEL_SIG_MANIFEST
    # Sigstore's own CLI writes `<name>.sigstore.json`; older tooling wrote
    # `<name>.sigstore`. Check the compound suffix via endswith rather than
    # Path.suffix, which would only see the final `.json`.
    if lower.endswith(".sigstore.json") or lower.endswith(".sigstore"):
        return ArtifactKind.SIGSTORE_BUNDLE
    # A JSONL of DSSE envelopes, the shape `slsa-verifier` and cosign emit for
    # provenance. One envelope per line.
    if lower.endswith(".intoto.jsonl"):
        return ArtifactKind.INTOTO_JSONL
    # A bare detached signature. Kept last among the `.sig` family so the more
    # specific manifest case above claims `model.sig` first.
    if lower.endswith(".sig"):
        return ArtifactKind.DETACHED_SIG
    # Plain `.json` / `.jsonl` are ambiguous by name. We return None here and
    # let content sniffing (below) promote them if they are actually a DSSE
    # envelope or a bundle; a name alone is not enough evidence to report one.
    return None


def _load_json(text: str) -> Any | None:
    """Parse JSON, returning None instead of raising on malformed input.

    RecursionError is caught alongside the value errors: a deeply nested
    artifact would otherwise escape this module, and this whole file promises
    never to raise on a hostile artifact.
    """
    try:
        return json.loads(text)
    except (ValueError, TypeError, RecursionError):
        return None


def _b64_json(value: str) -> Any | None:
    """Decode a base64 field to JSON, tolerating any malformed layer.

    DSSE envelopes and in-toto statements nest base64-encoded JSON. Any of the
    decode or parse steps can fail on a corrupt artifact; a failure means "we
    could not read this claim", not an error to propagate.
    """
    try:
        raw = base64.b64decode(value, validate=False)
    except (binascii.Error, ValueError, TypeError):
        return None
    try:
        return json.loads(raw.decode("utf-8", errors="replace"))
    except (ValueError, TypeError, RecursionError):
        return None


def _intoto_statement_claims(statement: Any) -> dict[str, Any]:
    """Pull the reportable fields out of a decoded in-toto statement.

    An in-toto statement names its `predicateType` (for SLSA provenance,
    something like `https://slsa.dev/provenance/v1`) and a `subject` list, each
    subject carrying the digest(s) of an artifact the attestation covers. Those
    digests are exactly what a verifier would later match against the model's
    own hash, so surfacing them (unverified) is the useful structural read.
    """
    claims: dict[str, Any] = {}
    if not isinstance(statement, dict):
        return claims
    predicate_type = statement.get("predicateType")
    if isinstance(predicate_type, str):
        claims["predicate_type"] = predicate_type
    subjects = statement.get("subject")
    if isinstance(subjects, list):
        digests: list[dict[str, str]] = []
        for subject in subjects:
            if not isinstance(subject, dict):
                continue
            digest = subject.get("digest")
            if isinstance(digest, dict):
                # Keep only str->str pairs (e.g. {"sha256": "abcd..."}); a
                # subject can name several algorithms.
                clean = {k: v for k, v in digest.items() if isinstance(v, str)}
                if clean:
                    digests.append(clean)
        if digests:
            claims["subject_digests"] = digests
    return claims


def _dsse_envelope_claims(envelope: Any) -> dict[str, Any]:
    """Read a DSSE envelope: its payloadType and, if in-toto, the statement.

    Returns {} for anything that is not a DSSE envelope so callers can use the
    presence of a `payload_type` key as the "this really is one" signal.
    """
    if not isinstance(envelope, dict):
        return {}
    payload_type = envelope.get("payloadType")
    if not isinstance(payload_type, str):
        return {}
    claims: dict[str, Any] = {"payload_type": payload_type}
    # The signed statement lives base64-encoded in `payload`. Decode it only
    # when the envelope claims to carry in-toto, since that is the one payload
    # shape we know how to read.
    if payload_type == _INTOTO_PAYLOAD_TYPE:
        payload = envelope.get("payload")
        if isinstance(payload, str):
            statement = _b64_json(payload)
            claims.update(_intoto_statement_claims(statement))
    return claims


def _sigstore_bundle_claims(bundle: Any) -> dict[str, Any]:
    """Read a Sigstore protobuf-JSON bundle for its digest and identity.

    A bundle either signs a raw message (`messageSignature.messageDigest`) or
    wraps a DSSE envelope (`dsseEnvelope`). Either way it carries verification
    material, and when that material is an X.509 certificate we surface its
    presence (the certificate is what binds a signature to an identity, though
    reading the identity out of the DER is beyond stdlib and beyond scope).
    Returns {} when the object does not look like a bundle at all.
    """
    if not isinstance(bundle, dict):
        return {}
    media_type = bundle.get("mediaType")
    is_bundle = isinstance(media_type, str) and media_type.startswith(
        _SIGSTORE_BUNDLE_MEDIA_PREFIX
    )
    # `dsseEnvelope` is a strong secondary signal for older bundles that
    # omitted or varied the mediaType string.
    has_dsse = isinstance(bundle.get("dsseEnvelope"), dict)
    if not is_bundle and not has_dsse:
        return {}
    claims: dict[str, Any] = {}
    if isinstance(media_type, str):
        claims["media_type"] = media_type
    # A raw-message signature names the digest it signed.
    message_sig = bundle.get("messageSignature")
    if isinstance(message_sig, dict):
        digest = message_sig.get("messageDigest")
        if isinstance(digest, dict):
            algorithm = digest.get("algorithm")
            value = digest.get("digest")
            if isinstance(algorithm, str):
                claims["message_digest_algorithm"] = algorithm
            if isinstance(value, str):
                claims["message_digest"] = value
    # A DSSE-wrapping bundle carries the in-toto statement one layer down.
    dsse = bundle.get("dsseEnvelope")
    if isinstance(dsse, dict):
        claims.update(_dsse_envelope_claims(dsse))
    # Note whether a signer certificate is present. Its mere presence is the
    # honest claim; parsing the identity from DER needs a crypto lib we do not
    # depend on.
    material = bundle.get("verificationMaterial")
    if isinstance(material, dict):
        if "certificate" in material or "x509CertificateChain" in material:
            claims["has_certificate"] = True
    return claims


def _read_claims(kind: ArtifactKind, path: Path, opener: Opener) -> dict[str, Any]:
    """Best-effort structural parse for one artifact. Never raises.

    A parse failure (missing file, oversized, malformed JSON, wrong shape)
    yields {} so the artifact is still reported as present with no claims. This
    is the function that must swallow the "malformed bundle" case the tests
    exercise.
    """
    # A detached blob and (in practice) a model.sig manifest are opaque bytes,
    # not JSON we can read structurally. Report them by presence alone.
    if kind in (ArtifactKind.DETACHED_SIG, ArtifactKind.MODEL_SIG_MANIFEST):
        return {}
    try:
        text = opener(path)
    except (OSError, ValueError):
        return {}

    if kind == ArtifactKind.INTOTO_JSONL:
        # One DSSE envelope per line; read the first non-empty line, which is
        # enough to confirm the type and pull a representative claim.
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            return _dsse_envelope_claims(_load_json(line))
        return {}

    parsed = _load_json(text)
    if kind == ArtifactKind.SIGSTORE_BUNDLE:
        return _sigstore_bundle_claims(parsed)
    if kind == ArtifactKind.DSSE_ENVELOPE:
        return _dsse_envelope_claims(parsed)
    return {}


def _sniff_ambiguous_json(path: Path, opener: Opener) -> ArtifactKind | None:
    """Promote a plain `.json` / `.jsonl` sibling if its content is a signer.

    Name-based classification skips generic JSON because a name alone is not
    evidence. This second pass reads the file and promotes it to a bundle or
    DSSE envelope only when the content proves it. Anything unreadable or
    unrecognised stays unclassified (returns None) and is not reported.
    """
    try:
        text = opener(path)
    except (OSError, ValueError):
        return None
    parsed = _load_json(text)
    if parsed is None:
        # Try JSONL: a first line that is a DSSE envelope.
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            candidate = _load_json(line)
            if _dsse_envelope_claims(candidate):
                return ArtifactKind.DSSE_ENVELOPE
            break
        return None
    if _sigstore_bundle_claims(parsed):
        return ArtifactKind.SIGSTORE_BUNDLE
    if _dsse_envelope_claims(parsed):
        return ArtifactKind.DSSE_ENVELOPE
    return None


def find_signature_artifacts(
    model_path: Path,
    listing: Listing | None = None,
    opener: Opener | None = None,
) -> list[SignatureArtifact]:
    """Detect signing artifacts sitting beside a model file.

    Looks at the siblings in `model_path`'s own directory. `listing` and
    `opener` are injectable so this runs against an in-memory inventory with no
    real files (the tests rely on that); by default they hit the filesystem.

    The model file itself is never returned, and detection is content-tolerant:
    an artifact whose JSON is malformed is still reported, just with empty
    claims. Ordering is stable (sorted by name) so output is deterministic.
    """
    model_path = Path(model_path)
    listing = listing or _default_listing
    opener = opener or _default_opener
    directory = model_path.parent

    try:
        names = listing(directory)
    except (OSError, ValueError):
        names = []

    artifacts: list[SignatureArtifact] = []
    for name in sorted(names):
        # Never treat the model as its own signature.
        if name == model_path.name:
            continue
        sibling = directory / name
        kind = _classify_by_name(name)
        if kind is None:
            # Ambiguous JSON: only reported if its content proves it is a signer.
            lower = name.lower()
            if lower.endswith(".json") or lower.endswith(".jsonl"):
                kind = _sniff_ambiguous_json(sibling, opener)
            if kind is None:
                continue
        claims = _read_claims(kind, sibling, opener)
        artifacts.append(SignatureArtifact(kind=kind, path=sibling, claims=claims))
    return artifacts


# Human-readable labels for each kind, used in the finding message.
_KIND_LABEL = {
    ArtifactKind.DETACHED_SIG: "a detached signature",
    ArtifactKind.SIGSTORE_BUNDLE: "a Sigstore bundle",
    ArtifactKind.INTOTO_JSONL: "an in-toto/SLSA DSSE attestation (JSONL)",
    ArtifactKind.DSSE_ENVELOPE: "an in-toto/SLSA DSSE envelope",
    ArtifactKind.MODEL_SIG_MANIFEST: "a model-transparency model.sig manifest",
}


def _clean(value: object, limit: int = 128) -> str:
    """Sanitize an attacker-controlled claim value for a finding message.

    The values here come from a hostile artifact (a crafted .sigstore.json can
    put anything in a predicateType or a digest). They flow into the finding
    message and from there into a Markdown or text report, which do not escape.
    So strip control characters (newlines that would forge extra report lines,
    terminal escapes) and truncate: a real digest is <=128 hex chars and a type
    is a short URI, so nothing legitimate is lost.
    """
    text = str(value)
    text = "".join(ch for ch in text if ch.isprintable())
    return text[:limit] + ("..." if len(text) > limit else "")


def _describe_claims(claims: dict[str, Any]) -> str:
    """Render parsed claims into a short clause for the finding message.

    Only the fields a reader cares about, and only when present. Returns "" so
    the caller can append it unconditionally. Every interpolated value is
    attacker-controlled and passes through `_clean`.
    """
    parts: list[str] = []
    if "predicate_type" in claims:
        parts.append(f"predicate {_clean(claims['predicate_type'])}")
    if "payload_type" in claims:
        parts.append(f"payloadType {_clean(claims['payload_type'])}")
    digests = claims.get("subject_digests")
    if isinstance(digests, list) and digests and isinstance(digests[0], dict) and digests[0]:
        # Show the first subject's first digest; enough to identify what the
        # attestation covers without dumping the whole list into a message.
        algo, value = next(iter(digests[0].items()))
        parts.append(f"signed {_clean(algo, 32)}:{_clean(value)}")
    if "message_digest" in claims:
        algo = claims.get("message_digest_algorithm", "digest")
        parts.append(f"signed {_clean(algo, 32)}:{_clean(claims['message_digest'])}")
    if claims.get("has_certificate"):
        parts.append("carries a signer certificate")
    if not parts:
        return ""
    return " It structurally claims: " + "; ".join(parts) + "."


def signature_findings(
    model_path: Path,
    artifacts: list[SignatureArtifact],
) -> list[Finding]:
    """Turn detected artifacts into INFO findings, one per artifact.

    Each finding says the artifact is present and, crucially, that Hayward did
    NOT cryptographically verify it: presence is not authenticity without a
    trust root and network access, both out of scope for this offline scanner.
    Returns [] when nothing was detected, so a model with no signing material
    produces no noise.
    """
    findings: list[Finding] = []
    for artifact in artifacts:
        label = _KIND_LABEL.get(artifact.kind, "a signing artifact")
        claims_clause = _describe_claims(artifact.claims)
        message = (
            f"Found {label} beside this model ({artifact.path.name}). "
            f"Hayward records its presence but does NOT verify it: confirming "
            f"authenticity needs a trust root and network access (a Sigstore "
            f"transparency log, a signer identity policy), both out of scope "
            f"for this offline scanner. Presence is not proof of "
            f"authenticity.{claims_clause}"
        )
        findings.append(
            Finding(
                rule_id=RULE_ID,
                message=message,
                severity=Severity.INFO,
                category=Category.AI_ML,
                file_path=str(model_path),
                # Metadata mirrors the structural read for machine consumers and
                # makes the no-verification stance explicit and inspectable.
                metadata={
                    "artifact_kind": artifact.kind.value,
                    "artifact_path": str(artifact.path),
                    "verified": False,
                    "claims": artifact.claims,
                },
            )
        )
    return findings
