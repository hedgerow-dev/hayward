"""The finding record produced by a scan."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Severity(str, Enum):
    """How much a finding should cost you.

    INFO is the unknown bucket: something was recognised as unfamiliar rather
    than as dangerous. Report it, but do not fail a build on it without
    deciding that deliberately. Every scanner in this space has an equivalent
    tier, and whether it counts is the single biggest lever on a
    false-positive rate.
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class Category(str, Enum):
    AI_ML = "ai_ml"
    DESERIALIZATION = "deserialization"
    INJECTION = "injection"
    PATH_TRAVERSAL = "path_traversal"
    SSTI = "ssti"


@dataclass
class Finding:
    """One statement about one file.

    `rule_id` identifies what was checked; the catalogue is in
    `docs/rules.md`. `confidence` is the scanner's own estimate and is
    reported rather than used to filter, so a consumer can set its own bar.
    """

    rule_id: str
    message: str
    severity: Severity
    category: Category
    file_path: str
    start_line: int = 0
    confidence: float = 1.0
    cwe_ids: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    engine: str = "hayward"

    @property
    def severity_order(self) -> int:
        order = {
            Severity.CRITICAL: 0,
            Severity.HIGH: 1,
            Severity.MEDIUM: 2,
            Severity.LOW: 3,
            Severity.INFO: 4,
        }
        return order.get(self.severity, 5)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "file": self.file_path,
            "confidence": self.confidence,
            "cwe_ids": self.cwe_ids,
            "metadata": self.metadata,
        }


# Rules that mean "this file was not fully read". They are not clean verdicts
# and they are not detections. Anything scoring a scan should count them in a
# coverage column of their own; see docs/rules.md.
COVERAGE_RULE_IDS = frozenset({
    "MFV-SKIP-001",
    "MFV-SKIP-002",
    "MFV-SKIP-003",
    "MFV-7Z-001",
    "MFV-GGUF-004",
})


def is_coverage_gap(finding: Finding) -> bool:
    """True when the finding reports that analysis did not complete."""
    return finding.rule_id in COVERAGE_RULE_IDS
