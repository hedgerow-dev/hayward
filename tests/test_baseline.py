"""Tests for baseline / diff mode (hayward.baseline).

Baselines are built from small in-memory findings and written to a tmp_path file
so each test states the exact snapshot it compares against. load_baseline reads a
path (a real snapshot lives on disk), so its tests go through tmp files; the
finding_key and diff tests work purely in memory.
"""

from __future__ import annotations

import json
from pathlib import Path

from hayward.baseline import (
    diff,
    finding_key,
    load_baseline,
    new_findings_fail,
)
from hayward.findings import Category, Finding, Severity


def make_finding(
    rule_id: str = "MFV-PICKLE-001",
    file_path: str = "models/m.bin",
    message: str = "unsafe global",
    severity: Severity = Severity.HIGH,
    metadata: dict | None = None,
) -> Finding:
    """A Finding with sensible defaults; each test overrides only what it probes."""
    return Finding(
        rule_id=rule_id,
        message=message,
        severity=severity,
        category=Category.DESERIALIZATION,
        file_path=file_path,
        metadata=metadata or {},
    )


def write_report(
    tmp_path: Path, findings: list[Finding], root: str | None = None
) -> Path:
    """Write the subset of `report.to_json` that load_baseline reads: the
    envelope with a "findings" list of to_dict() entries and a top-level "root".
    Returns the file path."""
    path = tmp_path / "baseline.json"
    path.write_text(
        json.dumps({"root": root, "findings": [f.to_dict() for f in findings]}),
        encoding="utf-8",
    )
    return path


def test_finding_present_in_both_is_unchanged():
    f = make_finding()
    baseline = {finding_key(f)}
    result = diff(baseline, [f])
    assert result.unchanged == [f]
    assert result.new == []
    assert result.fixed == []


def test_finding_only_in_current_is_new():
    old = make_finding(file_path="models/old.bin")
    new = make_finding(file_path="models/new.bin")
    baseline = {finding_key(old)}
    result = diff(baseline, [old, new])
    assert result.new == [new]
    assert result.unchanged == [old]
    assert result.fixed == []


def test_finding_only_in_baseline_is_fixed():
    old = make_finding(file_path="models/removed.bin")
    still = make_finding(file_path="models/still.bin")
    baseline = {finding_key(old), finding_key(still)}
    result = diff(baseline, [still])
    assert result.unchanged == [still]
    assert result.new == []
    # The removed finding surfaces in .fixed, by key.
    assert result.fixed == [finding_key(old)]


def test_absolute_current_matches_relative_baseline(tmp_path):
    """Path normalization: a baseline captured with absolute paths (and its own
    root recorded) matches a later scan that produced relative paths."""
    # Baseline: `hayward scan /ci/checkout/models` (root == the scanned dir).
    abs_finding = make_finding(file_path="/ci/checkout/models/sub/m.bin")
    snapshot = write_report(tmp_path, [abs_finding], root="/ci/checkout/models")
    baseline = load_baseline(snapshot)
    # Later run: `hayward scan models` in a different checkout, relative paths.
    rel_finding = make_finding(file_path="models/sub/m.bin")
    result = diff(baseline, [rel_finding], root="models")
    # Both normalise to "sub/m.bin", so the finding is unchanged, not new.
    assert result.unchanged == [rel_finding]
    assert result.new == []
    assert result.fixed == []


def test_message_change_without_metadata_is_new():
    """Documented keying, no-metadata case: with nothing structured to key on,
    the message is the only per-hit discriminator, so a reworded message reads
    as a new finding (the old one becomes fixed). This errs loud, which is the
    safe direction for a security gate: a possibly-changed issue fails the build
    rather than being absorbed as unchanged."""
    before = make_finding(message="unsafe global os.system", metadata={})
    after = make_finding(message="unsafe global subprocess.Popen", metadata={})
    baseline = {finding_key(before)}
    result = diff(baseline, [after])
    assert result.new == [after]
    assert result.fixed == [finding_key(before)]
    assert result.unchanged == []


def test_message_change_with_metadata_is_unchanged():
    """Documented keying, metadata case: when a rule emits structured metadata
    we key on that, so a cosmetic message edit does not churn the finding."""
    meta = {"offset": 42, "global": "os.system"}
    before = make_finding(message="unsafe global at offset 42", metadata=meta)
    after = make_finding(message="reworded but same locus", metadata=meta)
    baseline = {finding_key(before)}
    result = diff(baseline, [after])
    assert result.unchanged == [after]
    assert result.new == []
    assert result.fixed == []


def test_load_baseline_parses_report_envelope(tmp_path):
    f = make_finding()
    keys = load_baseline(write_report(tmp_path, [f]))
    assert keys == {finding_key(f)}


def test_load_baseline_tolerates_bare_list(tmp_path):
    f = make_finding()
    path = tmp_path / "bare.json"
    path.write_text(json.dumps([f.to_dict()]), encoding="utf-8")
    keys = load_baseline(path)
    assert keys == {finding_key(f)}


def test_load_baseline_accepts_path_and_str(tmp_path):
    """load_baseline takes the snapshot path as a Path or a str."""
    f = make_finding()
    snapshot = write_report(tmp_path, [f])
    assert load_baseline(snapshot) == {finding_key(f)}
    assert load_baseline(str(snapshot)) == {finding_key(f)}


def test_new_findings_fail_respects_threshold():
    # fail_on_order = 1 means critical(0) and high(1) fail; medium(2) does not.
    high = make_finding(severity=Severity.HIGH)
    medium = make_finding(severity=Severity.MEDIUM)
    assert new_findings_fail([high], fail_on_order=1) is True
    assert new_findings_fail([medium], fail_on_order=1) is False
    # An empty new-set never fails, whatever the threshold.
    assert new_findings_fail([], fail_on_order=0) is False
