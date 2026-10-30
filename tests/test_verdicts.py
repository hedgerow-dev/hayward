"""Issue #14: every file with findings gets one verdict, separate from severity.

flagged       a detection at LOW or above (wins over a coverage gap)
not_verified  only coverage gaps: the file was not fully read
clean         nothing, or only INFO notes
"""

from __future__ import annotations

import json

from hayward import Category, Finding, Severity
from hayward.findings import file_verdicts
from hayward.report import to_json, to_sarif


def _f(rule: str, severity: Severity, path: str) -> Finding:
    return Finding(rule_id=rule, message="m", severity=severity, category=Category.AI_ML, file_path=path)


FINDINGS = [
    _f("MFV-PICKLE-001", Severity.CRITICAL, "bad.pkl"),
    _f("MFV-SKIP-003", Severity.LOW, "bad.pkl"),  # a gap does not soften a detection
    _f("MFV-SKIP-003", Severity.LOW, "broken.pkl"),
    _f("MFV-LFS-001", Severity.INFO, "pointer.bin"),
    _f("MFV-PICKLE-004", Severity.INFO, "noted.pkl"),  # an INFO note is not a detection
]


def test_each_file_gets_one_verdict():
    assert file_verdicts(FINDINGS) == {
        "bad.pkl": "flagged",
        "broken.pkl": "not_verified",
        "pointer.bin": "not_verified",
        "noted.pkl": "clean",
    }


def test_json_report_carries_the_verdicts():
    report = json.loads(to_json(FINDINGS, None, "test"))
    assert report["verdicts"]["broken.pkl"] == "not_verified"
    assert report["verdicts"]["bad.pkl"] == "flagged"


def test_sarif_results_carry_the_file_verdict():
    results = json.loads(to_sarif(FINDINGS, None, "test"))["runs"][0]["results"]
    by_rule_file = {(r["ruleId"], r["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]):
                    r["properties"]["verdict"] for r in results}
    assert by_rule_file[("MFV-SKIP-003", "broken.pkl")] == "not_verified"
    assert by_rule_file[("MFV-SKIP-003", "bad.pkl")] == "flagged"
    assert by_rule_file[("MFV-PICKLE-004", "noted.pkl")] == "clean"
