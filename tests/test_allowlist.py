"""Tests for the hash-keyed finding allowlist.

Hashing is injected throughout via small in-memory byte fixtures, so nothing
here touches the filesystem or needs a real large model artifact.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pytest

from hayward.allowlist import (
    Allowlist,
    Entry,
    Suppression,
    load_allowlist,
)
from hayward.findings import Category, Finding, Severity


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _finding(rule_id: str, file_path: str) -> Finding:
    return Finding(
        rule_id=rule_id,
        message="example",
        severity=Severity.HIGH,
        category=Category.DESERIALIZATION,
        file_path=file_path,
    )


def _write(tmp_path: Path, entries: list[dict]) -> Path:
    path = tmp_path / "allowlist.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


def test_matching_entry_suppresses_and_others_pass_through(tmp_path: Path) -> None:
    model_bytes = b"pickle-model-bytes"
    other_bytes = b"a-different-file"
    file_map = {
        "model.pkl": _sha(model_bytes),
        "other.pkl": _sha(other_bytes),
    }

    allowlist = load_allowlist(
        _write(
            tmp_path,
            [
                {
                    "sha256": _sha(model_bytes),
                    "rule_id": "MFV-PICKLE-001",
                    "justification": "vetted internal checkpoint",
                    "approved_by": "kenneth",
                }
            ],
        )
    )

    suppressed_finding = _finding("MFV-PICKLE-001", "model.pkl")
    # Same file, different rule: not covered by the entry.
    other_rule = _finding("MFV-PICKLE-002", "model.pkl")
    # Same rule, different file: not covered either.
    other_file = _finding("MFV-PICKLE-001", "other.pkl")

    remaining, suppressions = allowlist.apply(
        [suppressed_finding, other_rule, other_file],
        file_sha256=file_map,
    )

    assert remaining == [other_rule, other_file]
    assert len(suppressions) == 1
    assert suppressions[0].finding is suppressed_finding


def test_missing_justification_rejected_at_load(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        [{"sha256": "ab" * 32, "rule_id": "MFV-PICKLE-001"}],
    )
    with pytest.raises(ValueError, match="justification"):
        load_allowlist(path)


def test_empty_justification_rejected_at_load(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        [
            {
                "sha256": "ab" * 32,
                "rule_id": "MFV-PICKLE-001",
                "justification": "   ",
            }
        ],
    )
    with pytest.raises(ValueError, match="justification"):
        load_allowlist(path)


def test_changed_file_hash_no_longer_matches(tmp_path: Path) -> None:
    original = b"original-model"
    allowlist = load_allowlist(
        _write(
            tmp_path,
            [
                {
                    "sha256": _sha(original),
                    "rule_id": "MFV-PICKLE-001",
                    "justification": "approved when it was the original bytes",
                }
            ],
        )
    )

    finding = _finding("MFV-PICKLE-001", "model.pkl")
    # The file's bytes changed since approval, so its digest no longer equals
    # the entry's. The finding must survive and force a re-review.
    changed_map = {"model.pkl": _sha(b"edited-model")}

    remaining, suppressions = allowlist.apply([finding], file_sha256=changed_map)

    assert remaining == [finding]
    assert suppressions == []


def test_expired_entry_does_not_suppress(tmp_path: Path) -> None:
    content = b"model"
    allowlist = load_allowlist(
        _write(
            tmp_path,
            [
                {
                    "sha256": _sha(content),
                    "rule_id": "MFV-PICKLE-001",
                    "justification": "temporary waiver",
                    "expires": "2026-01-01",
                }
            ],
        )
    )

    finding = _finding("MFV-PICKLE-001", "model.pkl")
    file_map = {"model.pkl": _sha(content)}

    # A day after expiry: the waiver lapsed, finding resurfaces.
    remaining, suppressions = allowlist.apply(
        [finding], file_sha256=file_map, today=date(2026, 6, 1)
    )
    assert remaining == [finding]
    assert suppressions == []

    # Expiry is inclusive: on the expiry date itself the entry still suppresses.
    remaining, suppressions = allowlist.apply(
        [finding], file_sha256=file_map, today=date(2026, 1, 1)
    )
    assert remaining == []
    assert len(suppressions) == 1


def test_suppression_carries_justification_and_audit_line(tmp_path: Path) -> None:
    content = b"model"
    allowlist = load_allowlist(
        _write(
            tmp_path,
            [
                {
                    "sha256": _sha(content),
                    "rule_id": "MFV-PICKLE-001",
                    "justification": "reviewed by appsec, ticket SEC-42",
                    "approved_by": "kenneth",
                }
            ],
        )
    )

    finding = _finding("MFV-PICKLE-001", "model.pkl")
    _, suppressions = allowlist.apply(
        [finding], file_sha256={"model.pkl": _sha(content)}
    )

    assert len(suppressions) == 1
    record = suppressions[0]
    assert isinstance(record, Suppression)
    assert record.justification == "reviewed by appsec, ticket SEC-42"
    line = record.audit_line()
    assert "MFV-PICKLE-001" in line
    assert "model.pkl" in line
    assert "kenneth" in line
    assert "reviewed by appsec, ticket SEC-42" in line


def test_hasher_callable_is_used_when_no_map(tmp_path: Path) -> None:
    content = b"model"
    allowlist = load_allowlist(
        _write(
            tmp_path,
            [
                {
                    "sha256": _sha(content),
                    "rule_id": "MFV-PICKLE-001",
                    "justification": "vetted",
                }
            ],
        )
    )

    calls: list[str] = []

    def hasher(file_path: str) -> str:
        calls.append(file_path)
        return _sha(content)

    # Two findings on the same file should hash it at most once (digest cache).
    findings = [
        _finding("MFV-PICKLE-001", "model.pkl"),
        _finding("MFV-PICKLE-002", "model.pkl"),
    ]
    remaining, suppressions = allowlist.apply(findings, hasher=hasher)

    assert calls == ["model.pkl"]
    assert len(suppressions) == 1
    assert remaining == [findings[1]]


def test_duplicate_key_rejected_at_load(tmp_path: Path) -> None:
    entry = {
        "sha256": "cd" * 32,
        "rule_id": "MFV-PICKLE-001",
        "justification": "first",
    }
    dup = dict(entry)
    dup["justification"] = "second"
    path = _write(tmp_path, [entry, dup])
    with pytest.raises(ValueError, match="duplicate"):
        load_allowlist(path)


def test_invalid_expires_rejected_at_load(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        [
            {
                "sha256": "ef" * 32,
                "rule_id": "MFV-PICKLE-001",
                "justification": "vetted",
                "expires": "not-a-date",
            }
        ],
    )
    with pytest.raises(ValueError, match="expires"):
        load_allowlist(path)


def test_non_array_top_level_rejected(tmp_path: Path) -> None:
    path = tmp_path / "allowlist.json"
    path.write_text(json.dumps({"sha256": "x"}), encoding="utf-8")
    with pytest.raises(ValueError, match="array"):
        load_allowlist(path)


def test_apply_with_direct_construction() -> None:
    # Allowlist can also be built directly (bypassing load) for callers that
    # already hold validated Entry objects.
    content = b"model"
    allowlist = Allowlist(
        [
            Entry(
                sha256=_sha(content),
                rule_id="MFV-PICKLE-001",
                justification="vetted",
            )
        ]
    )
    finding = _finding("MFV-PICKLE-001", "model.pkl")
    remaining, suppressions = allowlist.apply(
        [finding], file_sha256={"model.pkl": _sha(content)}
    )
    assert remaining == []
    assert len(suppressions) == 1
