"""Command-line interface.

    hayward scan path/to/models

Exit codes are the contract a CI job depends on, so they are deliberate:

    0   nothing at or above the fail threshold
    1   findings at or above the fail threshold
    2   the scan itself could not run: bad usage, an unreadable target, or a
        crash inside the scanner

A coverage gap does not fail the build on its own, but `--fail-on-coverage`
makes it do so. That option exists because a file the scanner could not read
is the one case where a passing build tells you nothing.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import TextIO

from hayward import __version__
from hayward.allowlist import load_allowlist
from hayward.baseline import diff as baseline_diff
from hayward.baseline import load_baseline, new_findings_fail
from hayward.cache import ScanCache
from hayward.findings import Finding, Severity, is_coverage_gap
from hayward.policy import load_policy
from hayward.report import render
from hayward.scanner import ModelFileScanner, _path_excluded

_THRESHOLDS = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
    "info": 4,
    # Below every severity_order, so no finding can ever reach it.
    "never": -1,
}

_COLOURS = {
    Severity.CRITICAL: "\033[1;31m",
    Severity.HIGH: "\033[31m",
    Severity.MEDIUM: "\033[33m",
    Severity.LOW: "\033[36m",
    Severity.INFO: "\033[2m",
}
_RESET = "\033[0m"


def _colour_enabled(args: argparse.Namespace, stream: TextIO) -> bool:
    mode = "never" if args.no_colour else args.color
    if mode == "always":
        return True
    if mode == "never":
        return False
    if "NO_COLOR" in os.environ:
        return False
    return stream.isatty()


def _render_text(findings: list[Finding], root: Path | None, colour: bool) -> str:
    if not findings:
        return f"No findings in {root}" if root is not None else "No findings"

    lines: list[str] = []
    for finding in sorted(findings, key=lambda f: (f.severity_order, f.file_path)):
        where: Path
        if root is None:
            where = Path(finding.file_path)
        else:
            try:
                where = Path(finding.file_path).relative_to(root)
            except ValueError:
                where = Path(finding.file_path)
        tint = _COLOURS.get(finding.severity, "") if colour else ""
        end = _RESET if colour else ""
        label = finding.severity.value.upper()
        lines.append(f"{tint}{label:<8}{end} {finding.rule_id:<16} {where}")
        lines.append(f"         {finding.message}")

    counts: dict[str, int] = {}
    for finding in findings:
        counts[finding.severity.value] = counts.get(finding.severity.value, 0) + 1
    order = ["critical", "high", "medium", "low", "info"]
    summary = ", ".join(f"{counts[s]} {s}" for s in order if s in counts)
    lines += ["", f"{len(findings)} finding(s): {summary}"]

    gaps = [f for f in findings if is_coverage_gap(f)]
    if gaps:
        lines.append(
            f"{len(gaps)} file(s) could not be fully read. "
            "Those are not clean verdicts; see docs/rules.md."
        )
    return "\n".join(lines)


def _parse_size(text: str) -> int:
    """Parse a --max-size value into a byte count.

    Accepts a plain integer (bytes) or an integer with a k/m/g suffix
    (case-insensitive). The multipliers are decimal (1000-based) to match the
    scanner's own 500MB default, which is 500_000_000, not 500 * 2**20. Raises
    ValueError on anything that is not one of those forms, or on a value <= 0
    (a zero or negative cap would make every file oversized, which is never
    what the operator meant).
    """
    cleaned = text.strip().lower()
    multiplier = 1
    if cleaned and cleaned[-1] in _SIZE_SUFFIXES:
        multiplier = _SIZE_SUFFIXES[cleaned[-1]]
        cleaned = cleaned[:-1]
    value = int(cleaned) * multiplier  # int() rejects junk with ValueError
    if value <= 0:
        raise ValueError(f"size must be positive, got {text!r}")
    return value


_SIZE_SUFFIXES = {"k": 1000, "m": 1000_000, "g": 1000_000_000}


def _report_root(targets: list[Path]) -> Path | None:
    """Pick the root that relative paths in the report are shown against.

    A single target keeps today's behavior exactly: a directory is its own
    root, a file is reported relative to its parent. Several targets are shown
    relative to their common ancestor so the paths stay meaningful; when they
    share no ancestor (or the ancestor is empty), we return None and the report
    falls back to absolute paths.
    """
    if len(targets) == 1:
        only = targets[0]
        return only if only.is_dir() else only.parent
    try:
        common = os.path.commonpath([str(t) for t in targets])
    except ValueError:
        return None
    return Path(common) if common else None


def _make_progress(args: argparse.Namespace):
    """Build the per-file progress callback, or None if nothing should print.

    Everything here writes to stderr only: stdout carries the report and must
    stay clean. --quiet wins over the others. --verbose names each file as it
    finishes; --progress keeps a running counter on one rewritten line.
    """
    if args.quiet:
        return None
    if args.verbose:
        def on_file(done: int, total: int, path: Path) -> None:
            print(f"hayward: scanned {path}", file=sys.stderr)
        return on_file
    if args.progress:
        def on_file(done: int, total: int, path: Path) -> None:
            end = "\n" if done == total else ""
            print(f"\rhayward: scanned {done}/{total}", end=end, file=sys.stderr)
        return on_file
    return None


def _collect_files(
    targets: list[Path], scanner: ModelFileScanner,
    exclude: list[str] | None, root: Path | None,
) -> list[Path]:
    """Flatten every target into the concrete file list to scan.

    A directory target is expanded through the scanner's discovery (which
    applies the same skip rules and --exclude filtering a library caller gets).
    A file target is taken as-is, unless --exclude matches it: an explicitly
    named file is still subject to exclusion so `--exclude '*.pkl'` means the
    same thing however the file was reached.
    """
    files: list[Path] = []
    for target in targets:
        if target.is_dir():
            files.extend(scanner.discover_files(target, exclude))
        elif exclude and _path_excluded(target, root or target.parent, exclude):
            continue
        else:
            files.append(target)
    return files


def _run_scan(
    scanner: ModelFileScanner, targets: list[Path], jobs: int,
    exclude: list[str] | None, root: Path | None,
    cache: ScanCache | None, progress,
) -> list[Finding]:
    """Scan the targets, composing --jobs and --cache in the main process.

    Without a cache, each directory target goes through scan_directory (which
    fans out across --jobs itself) and each file through scan_file, so the
    single-target paths stay byte-identical to before. With a cache, the file
    list is materialised so the cache can be read and written only here (never
    shared into a worker): a sequential run lets get_or_scan serve hits and
    scan misses in one pass, while a parallel run partitions first so only the
    misses are sent to the pool, then stores each miss under its content hash.
    Either way the progress counter spans hits and misses alike.
    """
    if cache is None:
        findings: list[Finding] = []
        for target in targets:
            if target.is_dir():
                findings.extend(scanner.scan_directory(
                    target, jobs=jobs, exclude=exclude, progress=progress))
            elif exclude and _path_excluded(target, root or target.parent, exclude):
                continue
            else:
                findings.extend(scanner.scan_file(target))
        return findings

    files = _collect_files(targets, scanner, exclude, root)
    total = len(files)
    findings = []
    if jobs <= 1:
        # Sequential + cache: get_or_scan returns cached findings on a content
        # hit and scans on a miss, storing the result for next time.
        for index, target in enumerate(files, 1):
            file_findings = cache.get_or_scan(target, scanner.scan_file)
            if progress is not None:
                progress(index, total, target)
            findings.extend(file_findings)
        return findings

    # Parallel + cache: partition on the cache key in the main process so a
    # worker never touches the cache, scan only the misses in the pool, then
    # fold their findings back into the cache. The key must be built the same
    # way get_or_scan builds it (path plus content hash), so we call the cache's
    # own entry_key rather than hashing here.
    hits: list[Path] = []
    miss_keys: dict[Path, str] = {}
    uncacheable: list[Path] = []
    for target in files:
        try:
            key = ScanCache.entry_key(target)
        except OSError:
            # Unreadable: it cannot be cached, and it must not abort the run.
            # Scan it directly below, through the firewall, so it degrades to a
            # coverage gap exactly as it would without --cache.
            uncacheable.append(target)
            continue
        if key in cache.entries:
            hits.append(target)
        else:
            miss_keys[target] = key
    done = 0
    for target in hits:
        file_findings = cache.get_or_scan(target, scanner.scan_file)
        done += 1
        if progress is not None:
            progress(done, total, target)
        findings.extend(file_findings)
    for target in uncacheable:
        file_findings = scanner.scan_file(target)
        done += 1
        if progress is not None:
            progress(done, total, target)
        findings.extend(file_findings)
    miss_results = scanner._scan_paths(list(miss_keys), jobs=jobs, progress=None)
    for target, file_findings in miss_results:
        cache.entries[miss_keys[target]] = [f.to_dict() for f in file_findings]
        done += 1
        if progress is not None:
            progress(done, total, target)
        findings.extend(file_findings)
    return findings


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
    scan.add_argument(
        "target", type=Path, nargs="+",
        help="one or more model files or directories of them",
    )
    scan.add_argument(
        "-f", "--format",
        choices=("text", "json", "html", "markdown", "sarif", "cyclonedx"),
        default="text", help="output format (default: text)",
    )
    scan.add_argument(
        "-o", "--output", type=Path, metavar="FILE",
        help="write the report to a file instead of stdout",
    )
    scan.add_argument(
        "--fail-on", choices=tuple(_THRESHOLDS), default="high",
        help="lowest severity that exits non-zero (default: high)",
    )
    scan.add_argument(
        "--fail-on-coverage", action="store_true",
        help="also exit non-zero when a file could not be fully read",
    )
    scan.add_argument(
        "--color", choices=("auto", "always", "never"), default="auto",
        help="colour output: auto follows the terminal and NO_COLOR (default: auto)",
    )
    scan.add_argument(
        "--no-colour", action="store_true",
        help="disable colour output (same as --color never)",
    )
    scan.add_argument(
        "--allowlist", type=Path, metavar="FILE",
        help="JSON allowlist of findings to suppress, each keyed by the file's "
             "sha256 and rule id and requiring a justification (see docs)",
    )
    scan.add_argument(
        "--baseline", type=Path, metavar="FILE",
        help="a prior JSON report; --fail-on then applies only to findings that "
             "are new relative to it (brownfield mode)",
    )
    scan.add_argument(
        "--check-signatures", action="store_true",
        help="also report sibling signature/attestation artifacts (Sigstore, "
             "in-toto/SLSA). Detection only, not cryptographic verification",
    )
    scan.add_argument(
        "--exclude", action="append", metavar="PATTERN",
        help="glob pattern (fnmatch) matched against each file's path as "
             "walked (relative to the scan root) and against any path "
             "component, so matching files and directories are skipped. "
             "Repeatable",
    )
    scan.add_argument(
        "--max-size", metavar="BYTES",
        help="override the 500MB per-file scan cap for this run; accepts a "
             "plain byte count or a k/m/g suffix (e.g. 200M, 1G, 500k)",
    )
    scan.add_argument(
        "--jobs", type=int, default=1, metavar="N",
        help="scan a directory's files across N worker processes "
             "(default: 1, fully sequential)",
    )
    scan.add_argument(
        "--policy", type=Path, metavar="FILE",
        help="JSON file of per-rule severity overrides, applied before the "
             "allowlist, baseline and the fail-on gate",
    )
    scan.add_argument(
        "--cache", type=Path, metavar="FILE",
        help="content-hash scan cache; a re-run skips files whose bytes are "
             "unchanged since the cache was written",
    )
    scan.add_argument(
        "--progress", action="store_true",
        help="write a running scanned N/total counter to stderr",
    )
    scan.add_argument(
        "--verbose", action="store_true",
        help="log each file to stderr as it is scanned",
    )
    scan.add_argument(
        "--quiet", action="store_true",
        help="suppress progress and informational lines on stderr",
    )

    args = parser.parse_args(argv)

    targets: list[Path] = args.target
    for target in targets:
        if not target.exists():
            print(f"hayward: no such file or directory: {target}", file=sys.stderr)
            return 2

    scanner = ModelFileScanner()
    scanner.check_signatures = args.check_signatures
    if args.max_size is not None:
        # Setting the instance attribute shadows the class constant for this
        # run, and the value is carried to worker processes via _ScanConfig.
        try:
            scanner.MAX_SCAN_BYTES = _parse_size(args.max_size)
        except ValueError:
            print(
                f"hayward: --max-size: invalid size {args.max_size!r}; "
                "use a positive byte count or a k/m/g suffix",
                file=sys.stderr,
            )
            return 2

    # A single target keeps today's root exactly; several targets are reported
    # against their common ancestor (see _report_root).
    root = _report_root(targets)

    progress = _make_progress(args)

    cache: ScanCache | None = None
    if args.cache is not None:
        # Namespace the cache by the config that changes a file's verdict.
        # ScanCache already keys on file bytes plus the package version, but
        # --check-signatures adds sibling-dependent findings and --max-size
        # changes what counts as oversized, and neither is in the file's bytes.
        # Folding them into the version tag means a run with different flags
        # misses a cache written under another setting instead of trusting it.
        tag = (
            f"{__version__}|sig={int(args.check_signatures)}"
            f"|max={scanner.MAX_SCAN_BYTES}"
        )
        # load() tolerates a missing or corrupt file (or a tag mismatch) by
        # returning an empty cache, so a first run or a changed config simply
        # scans everything.
        cache = ScanCache.load(args.cache, version_tag=tag)

    try:
        findings = _run_scan(
            scanner, targets, args.jobs, args.exclude, root, cache, progress)
    except OSError as exc:
        print(f"hayward: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        # Exit 1 means "findings at/above threshold"; a crash must not be
        # indistinguishable from that.
        print(f"hayward: scan crashed: {exc}", file=sys.stderr)
        return 2

    if cache is not None:
        # A save failure is a persistence problem, not a scan problem: warn but
        # keep whatever exit code the scan itself earns below.
        try:
            cache.save(args.cache)
        except OSError as exc:
            print(f"hayward: cache: {exc}", file=sys.stderr)

    # Policy remaps per-rule severities before anything else looks at them, so
    # the report, the allowlist, the baseline diff and the fail-on gate all see
    # the operator's severities rather than the scanner's defaults.
    if args.policy is not None:
        try:
            policy = load_policy(args.policy)
        except (OSError, ValueError) as exc:
            print(f"hayward: policy: {exc}", file=sys.stderr)
            return 2
        findings = policy.apply(findings)

    # Allowlist suppression runs before the report and the exit gate, so a
    # suppressed finding is absent from both. Every suppression is still
    # announced on stderr, so the audit trail is never silent: the report is
    # clean but the operator sees exactly what was withheld and why.
    if args.allowlist:
        try:
            allowlist = load_allowlist(args.allowlist)
            findings, suppressions = allowlist.apply(findings)
        except (OSError, ValueError) as exc:
            print(f"hayward: allowlist: {exc}", file=sys.stderr)
            return 2
        for suppression in suppressions:
            print(f"hayward: {suppression.audit_line()}", file=sys.stderr)

    # Baseline mode changes only what fails the build, never what the report
    # shows: the report is the full current state, while --fail-on is measured
    # against the findings that are new relative to the baseline.
    gate_findings = findings
    if args.baseline:
        try:
            baseline_keys = load_baseline(args.baseline, root=root)
        except (OSError, ValueError) as exc:
            print(f"hayward: baseline: {exc}", file=sys.stderr)
            return 2
        delta = baseline_diff(baseline_keys, findings, root=root)
        gate_findings = delta.new
        print(
            f"hayward: baseline: {len(delta.new)} new, "
            f"{len(delta.unchanged)} unchanged, {len(delta.fixed)} fixed",
            file=sys.stderr,
        )

    if args.format == "text" and not args.output:
        print(_render_text(findings, root, colour=_colour_enabled(args, sys.stdout)))
    else:
        if args.format == "text":
            report = _render_text(findings, root, colour=False)
        else:
            report = render(args.format, findings, root, __version__)
        if args.output:
            try:
                args.output.write_text(report, encoding="utf-8")
            except OSError as exc:
                print(f"hayward: could not write report: {exc}", file=sys.stderr)
                return 2
            print(f"Report written to {args.output}")
        else:
            print(report)

    limit = _THRESHOLDS[args.fail_on]
    if args.baseline:
        # Only findings new since the baseline can fail the build; the existing
        # backlog stays visible in the report but does not block a change.
        if new_findings_fail(gate_findings, limit):
            return 1
    elif any(f.severity_order <= limit for f in gate_findings):
        return 1
    if args.fail_on_coverage and any(is_coverage_gap(f) for f in gate_findings):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
