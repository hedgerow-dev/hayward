"""Scan receipts (issue #38): what Hayward found in each file, by hash.

    hayward scan DIR --receipt receipt.intoto.json
    hayward receipt verify receipt.intoto.json DIR

A receipt is an unsigned in-toto Statement. Its subjects are the scanned files
and their SHA-256 digests; its predicate records the scanner version and
settings, and for each file its verdict, the rules that fired and the coverage
gaps (what was not fully read). Hayward does not sign it: GitHub's attest action
or cosign wraps it in a Sigstore-signed DSSE envelope.

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


def verify(statement: dict, root: Path, model_files: list[Path]) -> list[str]:
    """Problems with ``root`` against the receipt; an empty list means it matches.

    ``model_files`` is what the scanner would pick up under ``root`` today, so a
    model file added after the receipt was written is reported too. Raises
    ValueError for a file that is not a Hayward receipt or names a path outside
    ``root``.
    """
    if statement.get("_type") != STATEMENT_TYPE or statement.get("predicateType") != PREDICATE_TYPE:
        raise ValueError("not a Hayward scan receipt")
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
    check = sub.add_parser("verify", help="check a directory against a scan receipt")
    check.add_argument("receipt", type=Path)
    check.add_argument("directory", type=Path)
    args = parser.parse_args(argv)

    from .scanner import ModelFileScanner

    try:
        statement = json.loads(args.receipt.read_text(encoding="utf-8"))
        if not isinstance(statement, dict):
            raise ValueError("not a Hayward scan receipt")
        problems = verify(statement, args.directory, ModelFileScanner().discover_files(args.directory))
    except (OSError, ValueError) as exc:
        print(f"hayward receipt: {exc}", file=sys.stderr)
        return 2
    for problem in problems:
        print(f"  {problem}")
    verdict = "matches" if not problems else "does NOT match"
    print(f"{args.directory}: {verdict} the receipt ({len(statement['subject'])} file(s))")
    return 1 if problems else 0
