"""Per-rule severity overrides applied after a scan.

A team accepts some findings and cares more about others without any say over
the scanner's built-in severities. This module lets them remap the severity of
named rules through a small JSON file, so an accepted INFO rule can be silenced
in a fail-on gate, or a rule they treat as release-blocking can be raised, all
without touching the curated rule set.

The format is deliberately tiny and stdlib-only (json):

    {"severity_overrides": {"MFV-PICKLE-004": "low", "MFV-HF-002": "critical"}}

Only the severity is remapped. Nothing else about a finding is recomputed, and
findings whose rule_id is not listed pass through untouched.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

from hayward.findings import Finding, Severity

# The accepted severity strings, taken straight from the Severity enum so this
# stays in step if a tier is ever added or renamed. A value outside this set is
# a policy authoring mistake, and we reject it at load time rather than let it
# silently match nothing later.
_VALID_SEVERITIES = {s.value for s in Severity}


@dataclass(frozen=True)
class Policy:
    """A loaded, validated set of per-rule severity overrides.

    `overrides` maps a rule_id to the Severity it should be reported as. The
    object is immutable: apply() reads it and never changes it, so one loaded
    Policy can be reused across many scans.
    """

    overrides: dict[str, Severity]

    def apply(self, findings: list[Finding]) -> list[Finding]:
        """Return findings with severities remapped per the override map.

        Callers' Finding objects are left untouched: a finding whose rule_id is
        overridden is returned as a shallow copy with only `severity` changed,
        and a finding with no override is passed through by identity (there is
        nothing to change, so copying it would be waste). The returned list is
        always new. This matters because a caller may hold the pre-policy
        findings for its own reporting, and mutating them in place would corrupt
        that view surprisingly.
        """
        result: list[Finding] = []
        for finding in findings:
            new_severity = self.overrides.get(finding.rule_id)
            if new_severity is None:
                result.append(finding)
                continue
            # dataclasses.replace builds a new Finding, copying every other
            # field, so severity_order and any downstream fail-on ordering
            # reflect the override immediately.
            result.append(dataclasses.replace(finding, severity=new_severity))
        return result


def load_policy(path: Path) -> Policy:
    """Load and validate a JSON policy file.

    Raises ValueError with a clear message on any malformed input: bad JSON, a
    non-object document, a `severity_overrides` that is not an object, or a
    severity string outside the Severity enum. We reject rather than swallow so
    a typo like "criticl" surfaces at load, not as a rule that quietly never
    matches.
    """
    raw = path.read_text(encoding="utf-8")
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"policy file {path} is not valid JSON: {exc}") from exc

    if not isinstance(document, dict):
        raise ValueError(
            f"policy file {path} must be a JSON object, got {type(document).__name__}"
        )

    overrides_raw = document.get("severity_overrides", {})
    if not isinstance(overrides_raw, dict):
        raise ValueError(
            f"policy file {path}: 'severity_overrides' must be an object mapping "
            f"rule ids to severities, got {type(overrides_raw).__name__}"
        )

    overrides: dict[str, Severity] = {}
    for rule_id, severity_value in overrides_raw.items():
        if not isinstance(severity_value, str) or severity_value not in _VALID_SEVERITIES:
            allowed = ", ".join(sorted(_VALID_SEVERITIES))
            raise ValueError(
                f"policy file {path}: unknown severity {severity_value!r} for rule "
                f"{rule_id!r}; allowed severities are {allowed}"
            )
        overrides[rule_id] = Severity(severity_value)

    return Policy(overrides=overrides)
