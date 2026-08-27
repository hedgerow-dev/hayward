"""Hayward: a security scanner for machine-learning model files.

Detects code execution, deserialisation attacks and malformed containers in
model checkpoints before anything loads them.

    from hayward import ModelFileScanner

    for finding in ModelFileScanner().scan_file(path):
        print(finding.rule_id, finding.severity.value, finding.message)

Nothing in this package imports, deserialises or executes the files it reads.
Pickle streams are walked as opcodes with `pickletools`; containers are read
member by member; XML is parsed with `defusedxml`.
"""

from hayward.findings import (
    COVERAGE_RULE_IDS,
    Category,
    Finding,
    Severity,
    is_coverage_gap,
)
from hayward.scanner import ModelFileScanner

__version__ = "1.2.0"

__all__ = [
    "COVERAGE_RULE_IDS",
    "Category",
    "Finding",
    "ModelFileScanner",
    "Severity",
    "__version__",
    "is_coverage_gap",
]
