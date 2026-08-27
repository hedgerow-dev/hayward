"""Tests for hayward.policy: per-rule severity overrides (HW-153 part A)."""

from __future__ import annotations

import json

import pytest

from hayward.findings import Category, Finding, Severity
from hayward.policy import Policy, load_policy


def _finding(rule_id: str, severity: Severity) -> Finding:
    return Finding(
        rule_id=rule_id,
        message="x",
        severity=severity,
        category=Category.DESERIALIZATION,
        file_path="model.pkl",
    )


def _write_policy(tmp_path, document: dict):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_override_remaps_only_the_named_rule(tmp_path):
    path = _write_policy(
        tmp_path,
        {"severity_overrides": {"MFV-PICKLE-004": "low", "MFV-HF-002": "critical"}},
    )
    policy = load_policy(path)

    findings = [
        _finding("MFV-PICKLE-004", Severity.INFO),
        _finding("MFV-HF-002", Severity.MEDIUM),
        _finding("MFV-ST-005", Severity.HIGH),
    ]
    result = policy.apply(findings)

    by_rule = {f.rule_id: f.severity for f in result}
    assert by_rule["MFV-PICKLE-004"] == Severity.LOW
    assert by_rule["MFV-HF-002"] == Severity.CRITICAL
    # Un-listed rule is untouched.
    assert by_rule["MFV-ST-005"] == Severity.HIGH


def test_unlisted_findings_pass_through_by_identity(tmp_path):
    path = _write_policy(tmp_path, {"severity_overrides": {"MFV-PICKLE-004": "low"}})
    policy = load_policy(path)

    untouched = _finding("MFV-ST-005", Severity.HIGH)
    result = policy.apply([untouched])
    # Nothing to change, so the same object comes back.
    assert result[0] is untouched


def test_apply_does_not_mutate_caller_inputs(tmp_path):
    path = _write_policy(tmp_path, {"severity_overrides": {"MFV-PICKLE-004": "low"}})
    policy = load_policy(path)

    original = _finding("MFV-PICKLE-004", Severity.INFO)
    result = policy.apply([original])
    # The caller's finding keeps its original severity; the override lands on a
    # new object.
    assert original.severity == Severity.INFO
    assert result[0].severity == Severity.LOW
    assert result[0] is not original


def test_override_changes_severity_order(tmp_path):
    """The fail-on ordering must reflect the new severity, not the old."""
    path = _write_policy(tmp_path, {"severity_overrides": {"MFV-PICKLE-004": "critical"}})
    policy = load_policy(path)

    (remapped,) = policy.apply([_finding("MFV-PICKLE-004", Severity.INFO)])
    assert remapped.severity_order == _finding("x", Severity.CRITICAL).severity_order
    # And it now sorts ahead of a genuine HIGH.
    high = _finding("y", Severity.HIGH)
    assert remapped.severity_order < high.severity_order


def test_empty_policy_is_a_no_op(tmp_path):
    path = _write_policy(tmp_path, {})
    policy = load_policy(path)
    assert policy.overrides == {}

    findings = [_finding("MFV-ST-005", Severity.HIGH)]
    assert policy.apply(findings) == findings


def test_unknown_severity_is_rejected_at_load(tmp_path):
    path = _write_policy(tmp_path, {"severity_overrides": {"MFV-PICKLE-004": "criticl"}})
    with pytest.raises(ValueError, match="unknown severity"):
        load_policy(path)


def test_non_object_document_is_rejected(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
    with pytest.raises(ValueError, match="must be a JSON object"):
        load_policy(path)


def test_non_object_overrides_is_rejected(tmp_path):
    path = _write_policy(tmp_path, {"severity_overrides": "high"})
    with pytest.raises(ValueError, match="must be an object"):
        load_policy(path)


def test_malformed_json_is_rejected(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_policy(path)


def test_policy_is_reusable_across_scans(tmp_path):
    path = _write_policy(tmp_path, {"severity_overrides": {"MFV-PICKLE-004": "low"}})
    policy = load_policy(path)

    first = policy.apply([_finding("MFV-PICKLE-004", Severity.INFO)])
    second = policy.apply([_finding("MFV-PICKLE-004", Severity.INFO)])
    assert first[0].severity == Severity.LOW
    assert second[0].severity == Severity.LOW
    assert isinstance(policy, Policy)
