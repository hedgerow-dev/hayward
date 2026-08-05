"""Command-line interface.

    hayward scan path/to/models

Exit codes are the contract a CI job depends on, so they are deliberate:

    0   nothing at or above the fail threshold
    1   findings at or above the fail threshold
    2   the scan itself could not run

A coverage gap does not fail the build on its own, but `--fail-on-coverage`
makes it do so. That option exists because a file the scanner could not read
is the one case where a passing build tells you nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hayward import __version__
from hayward.findings import Finding, Severity, is_coverage_gap
from hayward.scanner import ModelFileScanner

_THRESHOLDS = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
    "info": 4,
    "never": 99,
}

_COLOURS = {
    Severity.CRITICAL: "\033[1;31m",
    Severity.HIGH: "\033[31m",
    Severity.MEDIUM: "\033[33m",
    Severity.LOW: "\033[36m",
    Severity.INFO: "\033[2m",
}
_RESET = "\033[0m"


def _print_text(findings: list[Finding], root: Path, colour: bool) -> None:
    if not findings:
        print(f"No findings in {root}")
        return

    for finding in sorted(findings, key=lambda f: (f.severity_order, f.file_path)):
        try:
            where = Path(finding.file_path).relative_to(root)
        except ValueError:
            where = Path(finding.file_path)
        tint = _COLOURS.get(finding.severity, "") if colour else ""
        end = _RESET if colour else ""
        label = finding.severity.value.upper()
        print(f"{tint}{label:<8}{end} {finding.rule_id:<16} {where}")
        print(f"         {finding.message}")

    counts: dict[str, int] = {}
    for finding in findings:
        counts[finding.severity.value] = counts.get(finding.severity.value, 0) + 1
    order = ["critical", "high", "medium", "low", "info"]
    summary = ", ".join(f"{counts[s]} {s}" for s in order if s in counts)
    print(f"\n{len(findings)} finding(s): {summary}")

    gaps = [f for f in findings if is_coverage_gap(f)]
    if gaps:
        print(
            f"{len(gaps)} file(s) could not be fully read. "
            "Those are not clean verdicts; see docs/rules.md."
        )


def _print_json(findings: list[Finding], root: Path) -> None:
    gaps = [f for f in findings if is_coverage_gap(f)]
    print(json.dumps({
        "tool": "hayward",
        "version": __version__,
        "root": str(root),
        "findings": [f.to_dict() for f in findings],
        "coverage_gaps": [f.file_path for f in gaps],
    }, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hayward",
        description=(
            "Security scanner for machine-learning model files. Detects code "
            "execution and unsafe content in checkpoints without loading them."
        ),
    )
    parser.add_argument("--version", action="version", version=f"hayward {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="scan a file or directory")
    scan.add_argument("target", type=Path, help="model file or directory of them")
    scan.add_argument(
        "-f", "--format", choices=("text", "json"), default="text",
        help="output format (default: text)",
    )
    scan.add_argument(
        "--fail-on", choices=tuple(_THRESHOLDS), default="high",
        help="lowest severity that exits non-zero (default: high)",
    )
    scan.add_argument(
        "--fail-on-coverage", action="store_true",
        help="also exit non-zero when a file could not be fully read",
    )
    scan.add_argument("--no-colour", action="store_true", help="disable colour output")

    args = parser.parse_args(argv)

    target: Path = args.target
    if not target.exists():
        print(f"hayward: no such file or directory: {target}", file=sys.stderr)
        return 2

    scanner = ModelFileScanner()
    try:
        findings = (
            scanner.scan_file(target) if target.is_file()
            else scanner.scan_directory(target)
        )
    except OSError as exc:
        print(f"hayward: {exc}", file=sys.stderr)
        return 2

    root = target if target.is_dir() else target.parent
    if args.format == "json":
        _print_json(findings, root)
    else:
        _print_text(findings, root, colour=not args.no_colour and sys.stdout.isatty())

    limit = _THRESHOLDS[args.fail_on]
    if any(f.severity_order <= limit for f in findings):
        return 1
    if args.fail_on_coverage and any(is_coverage_gap(f) for f in findings):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
