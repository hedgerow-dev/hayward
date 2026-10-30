"""Scan receipts (issue #38): what Hayward found in each file, by hash.

    hayward scan DIR --receipt receipt.intoto.json
    hayward receipt verify receipt.intoto.json DIR
    hayward receipt sign receipt.intoto.json            (needs hayward[sign])
    hayward receipt verify receipt.intoto.json.sigstore.json DIR --identity WHO --issuer URL

A receipt is an unsigned in-toto Statement. Its subjects are the scanned files
and their SHA-256 digests; its predicate records the scanner version and
settings, and for each file its verdict, the rules that fired and the coverage
gaps (what was not fully read). Hayward does not sign it: GitHub's attest action
or cosign wraps it in a Sigstore-signed DSSE envelope, and so does
`hayward receipt sign` with the optional sigstore-python extra. Only signing and
signed verification use the network.

The receipt is scanner evidence, taken before any --policy or --allowlist, so an
organisation's acceptance decisions stay separate from what the scanner saw.
Receipts are never rewritten: a rescan writes a new one, and the newest receipt
for a digest is the current word. A receipt is analysis evidence, not a safety
certificate.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .findings import Finding, file_verdicts, is_coverage_gap
from .hub import _safe_relative, _sha256

STATEMENT_TYPE = "https://in-toto.io/Statement/v1"
PREDICATE_TYPE = "https://hedgerow.dev/hayward/scan-receipt/v1"
SIGNED_PAYLOAD_TYPE = "application/vnd.in-toto+json"
BUNDLE_MEDIA_PREFIX = "application/vnd.dev.sigstore.bundle"
PUBLIC_LOG = ("Signing writes your identity (the email you log in with) to Sigstore's "
              "public transparency log, permanently.")
NOTE = ("Analysis evidence: what Hayward found in each file, identified by hash. "
        "It is not a safety certificate.")


def build(files: list[Path], root: Path, findings: list[Finding], settings: dict[str, Any]) -> dict:
    verdicts = file_verdicts(findings)
    by_file: dict[str, list[Finding]] = {}
    for finding in findings:
        by_file.setdefault(finding.file_path, []).append(finding)
    subjects, entries = [], {}
    for path in sorted(files, key=lambda p: p.relative_to(root).as_posix()):
        name = path.relative_to(root).as_posix()
        own = by_file.get(str(path), [])
        subjects.append({"name": name, "digest": {"sha256": _sha256(path)}})
        entries[name] = {
            "verdict": verdicts.get(str(path), "clean"),
            "rules": sorted({f.rule_id for f in own if not is_coverage_gap(f)}),
            "coverage_gaps": sorted({f.rule_id for f in own if is_coverage_gap(f)}),
        }
        # Over the cap, the scanner reads a SafeTensors file by its header only.
        if path.suffix.lower() == ".safetensors" and path.stat().st_size > settings.get("max_scan_bytes", 0):
            entries[name]["read"] = "header"
    return {
        "_type": STATEMENT_TYPE,
        "subject": subjects,
        "predicateType": PREDICATE_TYPE,
        "predicate": {
            "scanner": {"name": "hayward", "version": __version__, "settings": settings},
            "scanned": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "note": NOTE,
            "files": entries,
        },
    }


class SignatureError(Exception):
    """A Sigstore bundle did not verify for the named signer."""


def _require_receipt(statement: Any) -> None:
    if (not isinstance(statement, dict) or statement.get("_type") != STATEMENT_TYPE
            or statement.get("predicateType") != PREDICATE_TYPE):
        raise ValueError("not a Hayward scan receipt")


# The three functions below are the only ones that touch sigstore (hayward[sign]),
# imported here so everything else works without it.

def _ambient_credential() -> bool:
    """True in CI with a workload identity (GitHub Actions, GitLab and others)."""
    from sigstore.errors import Error
    from sigstore.oidc import detect_credential

    try:
        return detect_credential() is not None
    except Error as exc:  # e.g. GitHub Actions without `id-token: write`
        raise SignatureError(f"no usable CI identity: {exc}") from exc


def _sigstore_sign(raw: bytes) -> str:
    """Sign the exact receipt bytes as a DSSE envelope; return the bundle JSON."""
    from sigstore.dsse import Statement
    from sigstore.errors import Error
    from sigstore.models import ClientTrustConfig
    from sigstore.oidc import IdentityToken, Issuer, IssuerError, detect_credential
    from sigstore.sign import SigningContext

    try:
        config = ClientTrustConfig.production()
        ambient = detect_credential()
        token = IdentityToken(ambient) if ambient else Issuer(config.signing_config.get_oidc_url()).identity_token()
        with SigningContext.from_trust_config(config).signer(token) as signer:
            return signer.sign_dsse(Statement(raw)).to_json()
    except (Error, IssuerError) as exc:
        raise SignatureError(f"signing failed: {exc}") from exc


def _sigstore_open(raw: bytes, identity: str, issuer: str) -> bytes:
    """Verify a bundle against the named signer; return the signed payload."""
    from sigstore.errors import Error
    from sigstore.models import Bundle
    from sigstore.verify import Verifier, policy

    try:
        payload_type, payload = Verifier.production().verify_dsse(
            Bundle.from_json(raw), policy.Identity(identity=identity, issuer=issuer))
    except Error as exc:
        raise SignatureError(str(exc)) from exc
    if payload_type != SIGNED_PAYLOAD_TYPE:
        raise SignatureError(f"the bundle signs {payload_type}, not an in-toto statement")
    return payload


def verify(statement: dict, root: Path, model_files: list[Path]) -> list[str]:
    """Problems with ``root`` against the receipt; an empty list means it matches.

    ``model_files`` is what the scanner would pick up under ``root`` today, so a
    model file added after the receipt was written is reported too. Raises
    ValueError for a file that is not a Hayward receipt or names a path outside
    ``root``.
    """
    _require_receipt(statement)
    subjects = statement.get("subject")
    if not isinstance(subjects, list):
        raise ValueError("the receipt has no subject list")
    covered = {}
    for subject in subjects:
        name = subject.get("name") if isinstance(subject, dict) else None
        if not isinstance(name, str) or not _safe_relative(name):
            raise ValueError(f"the receipt names a path outside the directory: {name!r}")
        digest = subject.get("digest")
        covered[name] = digest.get("sha256") if isinstance(digest, dict) else None
    problems = []
    for name, digest in covered.items():
        path = root / name
        if not path.is_file():
            problems.append(f"missing: {name}")
        elif _sha256(path) != digest:
            problems.append(f"changed: {name}")
    for path in model_files:
        name = path.relative_to(root).as_posix()
        if name not in covered:
            problems.append(f"not in the receipt: {name}")
    return problems


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="hayward receipt", description=(__doc__ or "").splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("verify", help="check a directory against a scan receipt, signed or not")
    check.add_argument("receipt", type=Path, help="a receipt, or a Sigstore bundle of one")
    check.add_argument("directory", type=Path)
    check.add_argument("--identity", help="for a signed receipt: who must have signed it (an email or workflow URI)")
    check.add_argument("--issuer", help="for a signed receipt: the OIDC issuer of that identity")
    signer = sub.add_parser("sign", help="sign a scan receipt with Sigstore (needs hayward[sign])")
    signer.add_argument("receipt", type=Path)
    signer.add_argument("-o", "--output", type=Path, help="bundle path (default: RECEIPT.sigstore.json)")
    signer.add_argument("--yes", action="store_true", help="sign without asking, even with a personal login")
    args = parser.parse_args(argv)
    try:
        return _sign(args) if args.command == "sign" else _verify(args)
    except ImportError:
        print('hayward receipt: signing needs: pip install "hayward[sign]"', file=sys.stderr)
    except (OSError, ValueError, SignatureError) as exc:
        print(f"hayward receipt: {exc}", file=sys.stderr)
    return 2


def _sign(args: argparse.Namespace) -> int:
    raw = args.receipt.read_bytes()
    _require_receipt(json.loads(raw))
    if not _ambient_credential() and not args.yes:
        print(f"hayward receipt: {PUBLIC_LOG}", file=sys.stderr)
        try:
            answer = input("Sign anyway? [y/N] ")
        except EOFError:  # no terminal to ask on: do not sign
            answer = ""
        if answer.strip().lower() != "y":
            print("hayward receipt: not signed", file=sys.stderr)
            return 2
    output = args.output or args.receipt.with_name(args.receipt.name + ".sigstore.json")
    output.write_text(_sigstore_sign(raw), encoding="utf-8")
    print(f"hayward: signed {args.receipt} -> {output}", file=sys.stderr)
    return 0


def _verify(args: argparse.Namespace) -> int:
    from .scanner import ModelFileScanner

    raw = args.receipt.read_bytes()
    data = json.loads(raw)
    signed = isinstance(data, dict) and str(data.get("mediaType", "")).startswith(BUNDLE_MEDIA_PREFIX)
    if signed:
        if not (args.identity and args.issuer):
            raise ValueError("a signed receipt needs --identity and --issuer: "
                             "a valid signature from anyone proves nothing")
        statement = json.loads(_sigstore_open(raw, args.identity, args.issuer))
    elif args.identity or args.issuer:
        raise ValueError("this receipt is not signed; --identity and --issuer apply to a Sigstore bundle")
    else:
        statement = data
    _require_receipt(statement)
    problems = verify(statement, args.directory, ModelFileScanner().discover_files(args.directory))
    for problem in problems:
        print(f"  {problem}")
    verdict = "matches" if not problems else "does NOT match"
    who = f", signed by {args.identity}" if signed else ""
    print(f"{args.directory}: {verdict} the receipt ({len(statement['subject'])} file(s){who})")
    return 1 if problems else 0
