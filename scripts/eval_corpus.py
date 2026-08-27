#!/usr/bin/env python3
"""Accuracy harness: score Hayward against a labelled corpus you supply.

    python scripts/eval_corpus.py CORPUS_DIR MANIFEST [--threshold high]

This runs `hayward`'s ModelFileScanner over every sample named in a labels
manifest and reports detection metrics: true/false positive and negative
counts, per-rule hit counts, the malicious samples it missed, and the benign
samples it flagged. It prints only what it measured on the corpus you give it.
No corpus ships in this repository and none is fabricated here; see
docs/accuracy-harness.md for the open decision on corpus ownership.

Two deliberate design points, because they are the things that quietly move
the numbers:

* Threshold. A sample counts as "detected" only when it carries a real finding
  at or above the chosen severity (`--threshold`, default `high`). Whether the
  INFO tier counts is the single biggest lever on any scanner's false-positive
  rate, so the threshold is stated at the top of every report.

* Coverage gaps. A file the scanner could not fully read (a Git LFS pointer, a
  7z with no extractor, a parser failure) is neither a detection nor a clean
  verdict. Those samples go in a coverage bucket of their own and are excluded
  from the TP/FP/TN/FN arithmetic, so a "read but not detected" miss is never
  confused with "never read". This mirrors how Hayward reports coverage
  everywhere else (docs/coverage.md).

hayward is imported lazily inside main(), so this module imports and
`py_compile`s without a built package present. Run `--selftest` to exercise the
pure classification and metric helpers with synthetic findings (it imports
nothing from hayward).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# Severity order, mirrored from hayward.findings.Severity.severity_order so the
# pure helpers below stay importable without the package. Lower is more severe;
# a finding "counts" at a threshold when its rank is <= the threshold's rank.
# Kept as plain data (five entries) rather than importing hayward at module
# load, which is the whole point of the lazy-import contract.
_SEVERITY_RANK: dict[str, int] = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
    "info": 4,
}

# The severities a caller may pass to --threshold. "never" is deliberately not
# offered here: a run where nothing can count is not a measurement.
_THRESHOLD_RANK: dict[str, int] = dict(_SEVERITY_RANK)


def _severity_rank(value: str) -> int:
    """Rank a severity string; unknown strings sort after INFO so a future
    severity never silently counts as a detection."""
    return _SEVERITY_RANK.get(value.lower(), 99)


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------

@dataclass
class SampleSpec:
    """One labelled sample from the manifest.

    `expected` is "malicious" or "benign". `rule` and `sink` are optional: when
    present they say which rule id, or which sink token, a correct detection
    ought to name, so the harness can report not just that a malicious file was
    caught but that it was caught for the right reason.
    """

    path: str
    expected: str
    rule: str | None = None
    sink: str | None = None


def _normalise_expected(raw: str, where: str) -> str:
    value = raw.strip().lower()
    # Accept the common spellings a hand-written manifest is likely to use, but
    # refuse anything ambiguous rather than guessing a label, because a
    # mislabelled sample corrupts every metric downstream in silence.
    if value in {"malicious", "mal", "bad", "positive", "1", "true"}:
        return "malicious"
    if value in {"benign", "clean", "good", "negative", "0", "false"}:
        return "benign"
    raise ValueError(
        f"{where}: expected label must be malicious or benign, got {raw!r}"
    )


def _spec_from_record(record: dict, where: str) -> SampleSpec:
    path = record.get("path") or record.get("file") or record.get("sample")
    if not path:
        raise ValueError(f"{where}: record has no 'path' field")
    expected = record.get("expected") or record.get("label")
    if expected is None:
        raise ValueError(f"{where}: record for {path!r} has no 'expected' field")
    rule = record.get("rule") or record.get("rule_id") or None
    sink = record.get("sink") or None
    return SampleSpec(
        path=str(path),
        expected=_normalise_expected(str(expected), where),
        rule=str(rule) if rule else None,
        sink=str(sink) if sink else None,
    )


def load_manifest(manifest_path: Path) -> list[SampleSpec]:
    """Read a labels manifest into SampleSpecs.

    Two formats, chosen by extension so a caller does not have to declare it:

    * JSON (.json): either a list of records, or an object with a "samples"
      list, or a flat mapping of path -> label ("benign"/"malicious") or
      path -> record object. A record is {"path", "expected"[, "rule"][,
      "sink"]}.

    * CSV (.csv): a header row naming at least `path` and `expected`, with
      optional `rule` and `sink` columns.
    """
    suffix = manifest_path.suffix.lower()
    if suffix == ".csv":
        return _load_csv_manifest(manifest_path)
    if suffix in {".json", ".jsonl"}:
        return _load_json_manifest(manifest_path)
    raise ValueError(
        f"unrecognised manifest extension {suffix!r}; use .json or .csv"
    )


def _load_csv_manifest(manifest_path: Path) -> list[SampleSpec]:
    specs: list[SampleSpec] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{manifest_path}: empty CSV, no header row")
        for row_number, row in enumerate(reader, start=2):  # header is line 1
            # Skip fully blank rows a spreadsheet export tends to leave behind.
            if not any((value or "").strip() for value in row.values()):
                continue
            specs.append(_spec_from_record(row, f"{manifest_path}:{row_number}"))
    return specs


def _load_json_manifest(manifest_path: Path) -> list[SampleSpec]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "samples" in data:
        data = data["samples"]

    specs: list[SampleSpec] = []
    if isinstance(data, dict):
        # Flat mapping: path -> "benign"/"malicious", or path -> record object.
        for path, value in data.items():
            where = f"{manifest_path}:{path}"
            if isinstance(value, str):
                specs.append(
                    SampleSpec(path=str(path),
                               expected=_normalise_expected(value, where))
                )
            elif isinstance(value, dict):
                record = dict(value)
                record.setdefault("path", path)
                specs.append(_spec_from_record(record, where))
            else:
                raise ValueError(
                    f"{where}: value must be a label string or a record object"
                )
    elif isinstance(data, list):
        for index, record in enumerate(data):
            where = f"{manifest_path}[{index}]"
            if not isinstance(record, dict):
                raise ValueError(f"{where}: list entries must be record objects")
            specs.append(_spec_from_record(record, where))
    else:
        raise ValueError(
            f"{manifest_path}: JSON must be a list, a mapping, or an object "
            "with a 'samples' list"
        )
    return specs


# ---------------------------------------------------------------------------
# Pure classification core (no hayward import; exercised by --selftest)
# ---------------------------------------------------------------------------

@dataclass
class FindingLite:
    """The slice of a Finding the harness reasons about.

    Decoupled from hayward.findings.Finding on purpose: the classification
    logic is pure so it can be unit-checked without a built package, and main()
    is the only place that turns real Findings into these.
    """

    rule_id: str
    severity: str
    is_coverage: bool
    text: str  # message plus stringified metadata, for sink token matching


@dataclass
class SampleResult:
    spec: SampleSpec
    outcome: str  # "tp" | "fp" | "tn" | "fn" | "coverage" | "missing"
    max_real_severity: str | None  # most severe non-coverage finding, if any
    real_rule_ids: list[str] = field(default_factory=list)
    coverage_rule_ids: list[str] = field(default_factory=list)
    reason_match: bool | None = None  # did it name the expected rule/sink?


def _reason_matches(spec: SampleSpec, real: list[FindingLite]) -> bool | None:
    """Whether a detection named the rule id or sink the manifest expected.

    None when the manifest asked for neither, so "no expectation" is never
    reported as a failed match. A rule expectation matches an exact rule id; a
    sink expectation matches a case-insensitive substring of the finding text
    (message plus metadata), which is where sink tokens like `os.system` land.
    """
    if not spec.rule and not spec.sink:
        return None
    if spec.rule and any(f.rule_id == spec.rule for f in real):
        return True
    if spec.sink:
        needle = spec.sink.lower()
        if any(needle in f.text.lower() for f in real):
            return True
    return False


def classify_sample(
    spec: SampleSpec, findings: list[FindingLite], limit: int
) -> SampleResult:
    """Bucket one sample given its findings and the threshold rank `limit`.

    A sample is "detected" only through a non-coverage finding at or above the
    threshold. A sample whose only findings are coverage gaps is bucketed as
    coverage and kept out of the confusion matrix, because it was never read
    and so cannot be a clean verdict nor a miss.
    """
    real = [f for f in findings if not f.is_coverage]
    coverage = [f for f in findings if f.is_coverage]

    detected = any(_severity_rank(f.severity) <= limit for f in real)
    coverage_only = not real and bool(coverage)

    max_real_severity: str | None = None
    if real:
        # Most severe means smallest rank; report it so a near-miss (a
        # malicious file that fired only INFO under a HIGH threshold) is legible
        # in the misses list.
        max_real_severity = min(
            (f.severity for f in real), key=_severity_rank
        )

    if spec.expected == "malicious":
        if detected:
            outcome = "tp"
        elif coverage_only:
            outcome = "coverage"
        else:
            outcome = "fn"
    else:  # benign
        if detected:
            outcome = "fp"
        elif coverage_only:
            outcome = "coverage"
        else:
            outcome = "tn"

    return SampleResult(
        spec=spec,
        outcome=outcome,
        max_real_severity=max_real_severity,
        real_rule_ids=[f.rule_id for f in real],
        coverage_rule_ids=[f.rule_id for f in coverage],
        reason_match=_reason_matches(spec, real) if outcome == "tp" else None,
    )


@dataclass
class Metrics:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    coverage_malicious: int = 0
    coverage_benign: int = 0
    missing: int = 0

    @property
    def precision(self) -> float | None:
        denom = self.tp + self.fp
        return self.tp / denom if denom else None

    @property
    def recall(self) -> float | None:
        # Recall is over samples that were actually read: coverage-bucket
        # malicious files are excluded from tp+fn, so this answers "of the
        # malicious files it read, how many did it catch", not "of all
        # malicious files". The coverage counts sit beside it for the rest.
        denom = self.tp + self.fn
        return self.tp / denom if denom else None

    @property
    def false_positive_rate(self) -> float | None:
        denom = self.fp + self.tn
        return self.fp / denom if denom else None


def tally(results: list[SampleResult]) -> Metrics:
    metrics = Metrics()
    for result in results:
        if result.outcome == "tp":
            metrics.tp += 1
        elif result.outcome == "fp":
            metrics.fp += 1
        elif result.outcome == "tn":
            metrics.tn += 1
        elif result.outcome == "fn":
            metrics.fn += 1
        elif result.outcome == "missing":
            metrics.missing += 1
        elif result.outcome == "coverage":
            if result.spec.expected == "malicious":
                metrics.coverage_malicious += 1
            else:
                metrics.coverage_benign += 1
    return metrics


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def _fmt_rate(rate: float | None) -> str:
    # A rate is only defined when its denominator is non-empty; say so rather
    # than printing a misleading 0.0 for an empty class.
    return "n/a (no samples)" if rate is None else f"{rate:.4f}"


def render_report(
    corpus_dir: Path,
    manifest_path: Path,
    threshold: str,
    results: list[SampleResult],
) -> str:
    limit = _THRESHOLD_RANK[threshold]
    counts_info = "counts INFO" if limit >= _SEVERITY_RANK["info"] else "excludes INFO"
    metrics = tally(results)

    total = len(results)
    n_mal = sum(1 for r in results if r.spec.expected == "malicious")
    n_ben = sum(1 for r in results if r.spec.expected == "benign")

    lines: list[str] = []
    lines.append("Hayward accuracy harness")
    lines.append("=" * 72)
    lines.append(f"corpus:    {corpus_dir}")
    lines.append(f"manifest:  {manifest_path}")
    lines.append(
        f"threshold: {threshold} (a sample is detected only by a non-coverage "
        f"finding at or above {threshold}; this {counts_info})"
    )
    lines.append(
        f"samples:   {total} total, {n_mal} malicious, {n_ben} benign"
    )
    lines.append("")
    lines.append(
        "All figures below are measured on the corpus above at the threshold "
        "above. They describe that corpus and nothing else."
    )
    lines.append("")

    # Confusion matrix and derived rates.
    lines.append("Counts")
    lines.append("-" * 72)
    lines.append(f"  true positives   (malicious, detected)     {metrics.tp}")
    lines.append(f"  false negatives  (malicious, missed)       {metrics.fn}")
    lines.append(f"  true negatives   (benign, clean)           {metrics.tn}")
    lines.append(f"  false positives  (benign, flagged)         {metrics.fp}")
    lines.append(
        f"  coverage gaps    (never fully read)        "
        f"{metrics.coverage_malicious + metrics.coverage_benign} "
        f"({metrics.coverage_malicious} malicious, "
        f"{metrics.coverage_benign} benign)"
    )
    if metrics.missing:
        lines.append(
            f"  missing on disk  (in manifest, not found)  {metrics.missing}"
        )
    lines.append("")
    lines.append("Rates (computed from the counts above)")
    lines.append("-" * 72)
    lines.append(
        f"  precision  tp/(tp+fp)          {_fmt_rate(metrics.precision)}"
    )
    lines.append(
        f"  recall     tp/(tp+fn)          {_fmt_rate(metrics.recall)}  "
        "(over malicious files that were read; coverage gaps excluded)"
    )
    lines.append(
        f"  fp rate    fp/(fp+tn)          "
        f"{_fmt_rate(metrics.false_positive_rate)}"
    )
    lines.append("")

    # Per-rule hit counts across every finding the run produced. Coverage rule
    # ids are marked so a reader does not mistake a skip for a detection.
    rule_hits: Counter[str] = Counter()
    coverage_rules: set[str] = set()
    for result in results:
        for rule_id in result.real_rule_ids:
            rule_hits[rule_id] += 1
        for rule_id in result.coverage_rule_ids:
            rule_hits[rule_id] += 1
            coverage_rules.add(rule_id)
    lines.append("Per-rule hit counts (findings, all severities)")
    lines.append("-" * 72)
    if rule_hits:
        for rule_id, count in sorted(
            rule_hits.items(), key=lambda kv: (-kv[1], kv[0])
        ):
            tag = "  [coverage]" if rule_id in coverage_rules else ""
            lines.append(f"  {rule_id:<20} {count}{tag}")
    else:
        lines.append("  (no findings)")
    lines.append("")

    # Reason-match summary: of the detections whose manifest named a rule or
    # sink, how many named it correctly. This is the "right reason" column, and
    # it is only meaningful for samples that carried an expectation.
    with_reason = [
        r for r in results if r.outcome == "tp" and r.reason_match is not None
    ]
    if with_reason:
        matched = sum(1 for r in with_reason if r.reason_match)
        lines.append("Detection reason (where the manifest named a rule/sink)")
        lines.append("-" * 72)
        lines.append(
            f"  {matched} of {len(with_reason)} detections named the expected "
            "rule id or sink"
        )
        mismatched = [r for r in with_reason if not r.reason_match]
        for result in mismatched:
            want = result.spec.rule or result.spec.sink
            got = ", ".join(sorted(set(result.real_rule_ids))) or "(none)"
            lines.append(f"    {result.spec.path}: expected {want}, got {got}")
        lines.append("")

    # Misses: malicious samples that were read but scored below the threshold.
    misses = [r for r in results if r.outcome == "fn"]
    lines.append(f"Misses ({len(misses)} malicious samples below threshold)")
    lines.append("-" * 72)
    if misses:
        for result in sorted(misses, key=lambda r: r.spec.path):
            top = result.max_real_severity or "no findings"
            lines.append(f"  {result.spec.path}  (highest: {top})")
    else:
        lines.append("  (none)")
    lines.append("")

    # False positives: benign samples flagged at or above the threshold.
    false_positives = [r for r in results if r.outcome == "fp"]
    lines.append(
        f"False positives ({len(false_positives)} benign samples at/above "
        "threshold)"
    )
    lines.append("-" * 72)
    if false_positives:
        for result in sorted(false_positives, key=lambda r: r.spec.path):
            rules = ", ".join(sorted(set(result.real_rule_ids))) or "(unknown)"
            top = result.max_real_severity or "?"
            lines.append(f"  {result.spec.path}  ({top}: {rules})")
    else:
        lines.append("  (none)")
    lines.append("")

    # Coverage bucket: named explicitly so a reader can decide whether the
    # corpus is measuring detection or measuring the reader environment.
    coverage_samples = [r for r in results if r.outcome == "coverage"]
    if coverage_samples:
        lines.append(
            f"Coverage gaps ({len(coverage_samples)} samples never fully read)"
        )
        lines.append("-" * 72)
        for result in sorted(coverage_samples, key=lambda r: r.spec.path):
            rules = ", ".join(sorted(set(result.coverage_rule_ids))) or "(none)"
            lines.append(
                f"  {result.spec.path}  ({result.spec.expected}: {rules})"
            )
        lines.append("")

    missing = [r for r in results if r.outcome == "missing"]
    if missing:
        lines.append(f"Missing on disk ({len(missing)} manifest entries)")
        lines.append("-" * 72)
        for result in sorted(missing, key=lambda r: r.spec.path):
            lines.append(f"  {result.spec.path}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Runner (the only place that touches hayward)
# ---------------------------------------------------------------------------

def _findings_to_lite(findings: list) -> list[FindingLite]:
    """Turn real hayward Findings into the pure shape the classifier uses.

    Imported lazily by the caller; this function only receives objects, so it
    stays independent of the package too. `is_coverage_gap` is passed in rather
    than imported here to keep the seam in one place.
    """
    from hayward.findings import is_coverage_gap  # lazy: no import at module load

    lite: list[FindingLite] = []
    for finding in findings:
        # Fold message and metadata into one searchable string so a sink
        # expectation can match a token that Hayward records in metadata rather
        # than in the human message.
        text = f"{finding.message} {finding.metadata}"
        lite.append(
            FindingLite(
                rule_id=finding.rule_id,
                severity=finding.severity.value,
                is_coverage=is_coverage_gap(finding),
                text=text,
            )
        )
    return lite


def run_corpus(
    corpus_dir: Path,
    specs: list[SampleSpec],
    threshold: str,
    check_signatures: bool = False,
) -> list[SampleResult]:
    """Scan every sample and classify it. Imports hayward lazily."""
    from hayward.scanner import ModelFileScanner  # lazy: keeps module importable

    scanner = ModelFileScanner()
    scanner.check_signatures = check_signatures
    limit = _THRESHOLD_RANK[threshold]

    results: list[SampleResult] = []
    for spec in specs:
        sample_path = Path(spec.path)
        if not sample_path.is_absolute():
            sample_path = corpus_dir / sample_path
        if not sample_path.exists():
            # A manifest entry with no file is a manifest error, not a miss; it
            # gets its own bucket so it cannot silently deflate recall.
            results.append(
                SampleResult(spec=spec, outcome="missing", max_real_severity=None)
            )
            continue
        findings = scanner.scan_file(sample_path)
        results.append(classify_sample(spec, _findings_to_lite(findings), limit))
    return results


# ---------------------------------------------------------------------------
# Self-test: pure helpers only, no hayward import
# ---------------------------------------------------------------------------

def _selftest() -> int:
    """Exercise the classification and metric arithmetic with synthetic
    findings. Deliberately imports nothing from hayward so it runs on a bare
    checkout, and checks behaviour rather than any specific accuracy figure.

    Uses an explicit `_expect` that raises rather than the `assert` statement,
    because this project's ruff config runs the bandit ruleset (S101) which
    bans `assert` outside the test tree.
    """

    def lite(rule: str, sev: str, coverage: bool = False, text: str = "") -> FindingLite:
        return FindingLite(rule_id=rule, severity=sev, is_coverage=coverage, text=text)

    def _expect(condition: bool, message: str) -> None:
        if not condition:
            raise RuntimeError(f"selftest failed: {message}")

    high = _THRESHOLD_RANK["high"]

    # Malicious with a CRITICAL finding is a true positive at a HIGH threshold.
    mal = SampleSpec(path="a", expected="malicious")
    _expect(
        classify_sample(mal, [lite("R1", "critical")], high).outcome == "tp",
        "critical malicious -> tp",
    )

    # Malicious that fired only INFO is a miss when the threshold excludes INFO.
    _expect(
        classify_sample(mal, [lite("R2", "info")], high).outcome == "fn",
        "info-only malicious under HIGH -> fn",
    )

    # ...but is a detection when the threshold counts INFO.
    info = _THRESHOLD_RANK["info"]
    _expect(
        classify_sample(mal, [lite("R2", "info")], info).outcome == "tp",
        "info-only malicious under INFO -> tp",
    )

    # Malicious whose only finding is a coverage gap is bucketed as coverage,
    # never as a miss, even though a LOW coverage rule outranks a HIGH gate.
    cov = classify_sample(mal, [lite("MFV-SKIP-003", "low", coverage=True)], high)
    _expect(cov.outcome == "coverage", f"coverage-only -> coverage, got {cov.outcome}")

    # Benign with no findings is a true negative.
    ben = SampleSpec(path="b", expected="benign")
    _expect(classify_sample(ben, [], high).outcome == "tn", "clean benign -> tn")

    # Benign flagged at/above threshold is a false positive.
    _expect(
        classify_sample(ben, [lite("R3", "high")], high).outcome == "fp",
        "high benign -> fp",
    )

    # Benign that fired only INFO under a HIGH threshold is NOT a false
    # positive: the INFO tier not counting is exactly the threshold's job.
    _expect(
        classify_sample(ben, [lite("R4", "info")], high).outcome == "tn",
        "info-only benign under HIGH -> tn",
    )

    # Reason match: a detection that names the expected sink token in metadata.
    mal_sink = SampleSpec(path="c", expected="malicious", sink="os.system")
    res = classify_sample(
        mal_sink, [lite("R5", "critical", text="calls os.system")], high
    )
    _expect(
        res.outcome == "tp" and res.reason_match is True,
        "sink token match -> reason_match True",
    )

    # Reason mismatch: detected, but not for the named rule.
    mal_rule = SampleSpec(path="d", expected="malicious", rule="EXPECTED-1")
    res = classify_sample(mal_rule, [lite("OTHER-1", "critical")], high)
    _expect(
        res.outcome == "tp" and res.reason_match is False,
        "wrong rule -> reason_match False",
    )

    # Metric arithmetic on a hand-built set of outcomes.
    results = [
        SampleResult(mal, "tp", "critical"),
        SampleResult(mal, "tp", "high"),
        SampleResult(mal, "fn", "info"),
        SampleResult(ben, "tn", None),
        SampleResult(ben, "fp", "high"),
        SampleResult(mal, "coverage", None),
        SampleResult(ben, "coverage", None),
        SampleResult(mal, "missing", None),
    ]
    m = tally(results)
    _expect((m.tp, m.fp, m.tn, m.fn) == (2, 1, 1, 1), f"confusion matrix {m}")
    _expect(
        m.coverage_malicious == 1 and m.coverage_benign == 1,
        "coverage split 1/1",
    )
    _expect(m.missing == 1, "missing count 1")
    _expect(abs(m.precision - 2 / 3) < 1e-9, "precision 2/3")
    _expect(abs(m.recall - 2 / 3) < 1e-9, "recall 2/3")
    _expect(abs(m.false_positive_rate - 1 / 2) < 1e-9, "fp rate 1/2")

    # An empty class yields an undefined rate, not a misleading zero.
    _expect(Metrics().precision is None, "empty precision is None")

    print("selftest: ok")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="eval_corpus.py",
        description=(
            "Score Hayward against a labelled corpus you supply. Runs the "
            "ModelFileScanner over each sample named in a labels manifest and "
            "reports true/false positive and negative counts, per-rule hit "
            "counts, misses, and false positives. Measures only the corpus you "
            "pass; no corpus ships in this repository."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "manifest formats:\n"
            "  JSON  a list of {\"path\",\"expected\"[,\"rule\"][,\"sink\"]} "
            "records,\n"
            "        an object with a \"samples\" list, or a flat mapping of\n"
            "        path -> \"benign\"/\"malicious\".\n"
            "  CSV   a header row with columns path,expected[,rule][,sink].\n"
            "\n"
            "expected is malicious or benign. Sample paths are resolved "
            "relative to\nCORPUS_DIR unless absolute. See "
            "docs/accuracy-harness.md."
        ),
    )
    parser.add_argument(
        "corpus_dir", type=Path, nargs="?",
        help="directory holding the sample files named in the manifest",
    )
    parser.add_argument(
        "manifest", type=Path, nargs="?",
        help="labels manifest (.json or .csv) mapping each sample to a label",
    )
    parser.add_argument(
        "--threshold", choices=tuple(_THRESHOLD_RANK), default="high",
        help=(
            "lowest severity that counts as a detection (default: high). "
            "Whether INFO counts is the single biggest lever on the numbers, "
            "so it is stated in the report."
        ),
    )
    parser.add_argument(
        "--check-signatures", action="store_true",
        help="also run Hayward's signature/attestation detection pass",
    )
    parser.add_argument(
        "-o", "--output", type=Path, metavar="FILE",
        help="write the report to a file instead of stdout",
    )
    parser.add_argument(
        "--selftest", action="store_true",
        help="run the built-in checks of the pure helpers and exit (no corpus, "
             "no hayward import)",
    )

    args = parser.parse_args(argv)

    if args.selftest:
        return _selftest()

    if args.corpus_dir is None or args.manifest is None:
        parser.error("corpus_dir and manifest are required unless --selftest")

    if not args.corpus_dir.is_dir():
        parser.error(f"corpus_dir is not a directory: {args.corpus_dir}")
    if not args.manifest.is_file():
        parser.error(f"manifest is not a file: {args.manifest}")

    try:
        specs = load_manifest(args.manifest)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"eval_corpus: manifest: {exc}", file=sys.stderr)
        return 2
    if not specs:
        print("eval_corpus: manifest names no samples", file=sys.stderr)
        return 2

    results = run_corpus(
        args.corpus_dir, specs, args.threshold,
        check_signatures=args.check_signatures,
    )
    report = render_report(args.corpus_dir, args.manifest, args.threshold, results)

    if args.output:
        try:
            args.output.write_text(report, encoding="utf-8")
        except OSError as exc:
            print(f"eval_corpus: could not write report: {exc}", file=sys.stderr)
            return 2
        print(f"Report written to {args.output}")
    else:
        sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
