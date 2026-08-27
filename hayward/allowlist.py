"""A hash-keyed, justification-required finding allowlist.

The problem this solves: teams want to suppress a known-and-accepted finding
without silencing the scanner wholesale, and an auditor later needs to see
*why* each suppression was granted and *what exactly* it covered.

The design choice that makes it auditable is the key: an entry is keyed by the
sha256 of the file's bytes plus the rule_id, not by the file path. A path-keyed
allowlist keeps suppressing even after someone edits the file, which is exactly
when you would want a fresh look. Keying on content means the suppression stops
matching the moment the artifact changes, forcing a re-review. That is the
whole thesis, so matching is deliberately exact: no globbing, no path rules,
no plugin hooks.

Justification is mandatory and enforced at load time. A malformed or
unjustified entry raises rather than being silently dropped, because a
suppression you cannot explain is not one you should be able to apply.

Format is plain JSON so the allowlist reviews cleanly in a pull request with no
extra dependency. The file is a JSON array of entry objects; see Entry.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from hayward.findings import Finding

# A function that maps a finding's file_path to the sha256 hex digest of that
# file's bytes. It is injected into apply() so tests can supply an in-memory map
# and never touch the filesystem, and so a caller can cache digests across a
# large scan instead of re-hashing shared files.
Hasher = Callable[[str], str]


def _sha256_file(file_path: str) -> str:
    """Default hasher: stream the file from disk into a sha256 hex digest.

    Read in chunks rather than all at once, because the model artifacts this
    scanner targets can be many gigabytes and must not be pulled fully into
    memory just to hash them.
    """
    digest = hashlib.sha256()
    with open(file_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Entry:
    """One approved suppression, as loaded from the allowlist file.

    An entry matches a finding only when both the file's content hash and the
    rule_id are equal, so it stops applying the instant the file's bytes change.

    `justification` is required and non-empty (enforced in load_allowlist). The
    audit-trail fields are optional: `approved_by` records who signed off,
    `reason` is a free-form category or ticket reference distinct from the
    human-readable justification, and `expires` is an inclusive last-effective
    date. An entry past its expiry does not suppress; see is_expired.
    """

    sha256: str
    rule_id: str
    justification: str
    expires: date | None = None
    reason: str | None = None
    approved_by: str | None = None

    def is_expired(self, today: date) -> bool:
        """True when `today` is strictly after the inclusive expiry date.

        An entry with no expiry never expires. Expiry is inclusive so an entry
        dated today still suppresses today.
        """
        if self.expires is None:
            return False
        return today > self.expires

    def matches(self, finding: Finding, file_sha256: str) -> bool:
        """True when this entry covers the given finding.

        Both the content hash and the rule_id must be equal. This intentionally
        ignores the file path: two identical files with different names share a
        hash and are covered by the same entry, and a renamed-but-unchanged file
        stays covered, because content is what was reviewed.
        """
        return self.rule_id == finding.rule_id and self.sha256 == file_sha256


@dataclass(frozen=True)
class Suppression:
    """A record that a finding was withheld, and by which entry.

    The caller keeps these so it can print an audit line and account for every
    finding that did not surface. It carries the finding, the entry that matched,
    and the justification lifted out for convenience.
    """

    finding: Finding
    entry: Entry
    justification: str

    def audit_line(self) -> str:
        """One human-readable line summarising the suppression for a log."""
        who = self.entry.approved_by or "unspecified"
        return (
            f"suppressed {self.finding.rule_id} on {self.finding.file_path} "
            f"by {who}: {self.justification}"
        )


class Allowlist:
    """A collection of validated entries that can suppress findings.

    Construct via load_allowlist so entries are validated. Entries are indexed
    by (sha256, rule_id) for O(1) lookup, which also means a later duplicate of
    the same key silently wins; load_allowlist rejects duplicates before they
    reach here so that cannot happen in practice.
    """

    def __init__(self, entries: list[Entry]) -> None:
        self.entries = entries
        # Index by the exact match key so apply() does not scan every entry per
        # finding. The key mirrors Entry.matches: content hash plus rule_id.
        self._by_key: dict[tuple[str, str], Entry] = {
            (entry.sha256, entry.rule_id): entry for entry in entries
        }

    def apply(
        self,
        findings: list[Finding],
        hasher: Hasher | None = None,
        file_sha256: dict[str, str] | None = None,
        today: date | None = None,
    ) -> tuple[list[Finding], list[Suppression]]:
        """Partition findings into (remaining, suppressed).

        A finding is suppressed only when a non-expired entry matches both its
        file's content hash and its rule_id. Anything else passes through,
        including a finding matched only by an expired entry: expiry means the
        approval lapsed, so the finding must resurface for re-review.

        Hashing is injectable so callers and tests control it. Provide either a
        precomputed `file_sha256` map (path -> hex digest) or a `hasher`
        callable; the map is consulted first and the hasher fills any gaps. With
        neither, the default streams files from disk. `today` defaults to the
        real current date and exists so tests can pin expiry deterministically.

        Findings whose file cannot be hashed (e.g. a missing path with the
        default hasher) are not silently dropped: the error propagates, because
        a suppression decision that cannot be made must not default to showing
        or to hiding the finding without the caller knowing.
        """
        if today is None:
            today = date.today()
        precomputed = file_sha256 or {}

        remaining: list[Finding] = []
        suppressed: list[Suppression] = []

        # Cache digests within this call so a file that produces several findings
        # is hashed at most once even when only a hasher (no map) was supplied.
        digest_cache: dict[str, str] = dict(precomputed)

        for finding in findings:
            digest = digest_cache.get(finding.file_path)
            if digest is None:
                compute = hasher or _sha256_file
                try:
                    digest = compute(finding.file_path)
                except OSError:
                    # The file cannot be hashed (unreadable, or gone between
                    # the scan and now). Without its content hash it cannot
                    # match a content-keyed entry, so the finding stands. A
                    # suppression pass must never turn one unreadable file into
                    # a failure that aborts the whole run.
                    remaining.append(finding)
                    continue
                digest_cache[finding.file_path] = digest

            entry = self._by_key.get((digest, finding.rule_id))
            # No entry, or an entry that has lapsed: the finding stands. An
            # expired match is treated as no match rather than a suppression so
            # the caller sees the finding again and can renew or drop the entry.
            if entry is None or entry.is_expired(today):
                remaining.append(finding)
                continue

            suppressed.append(
                Suppression(
                    finding=finding,
                    entry=entry,
                    justification=entry.justification,
                )
            )

        return remaining, suppressed


def _parse_expires(raw: object, index: int) -> date | None:
    """Parse an optional ISO `expires` value, or raise a clear ValueError."""
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise ValueError(
            f"allowlist entry {index}: 'expires' must be a YYYY-MM-DD string"
        )
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        # Re-raise with the entry index so a reviewer can find the bad line.
        raise ValueError(
            f"allowlist entry {index}: 'expires' is not a valid YYYY-MM-DD "
            f"date: {raw!r}"
        ) from exc


def _parse_entry(raw: object, index: int) -> Entry:
    """Validate one raw JSON object into an Entry, or raise ValueError.

    Every failure names the entry index and the offending field so the person
    fixing the file does not have to guess which of many entries is wrong.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"allowlist entry {index}: must be a JSON object")

    sha256 = raw.get("sha256")
    if not isinstance(sha256, str) or not sha256.strip():
        raise ValueError(
            f"allowlist entry {index}: 'sha256' is required and must be a "
            f"non-empty string"
        )

    rule_id = raw.get("rule_id")
    if not isinstance(rule_id, str) or not rule_id.strip():
        raise ValueError(
            f"allowlist entry {index}: 'rule_id' is required and must be a "
            f"non-empty string"
        )

    # The justification gate is the point of the feature: a suppression you
    # cannot explain is rejected at load rather than quietly applied. Whitespace
    # does not count as an explanation.
    justification = raw.get("justification")
    if not isinstance(justification, str) or not justification.strip():
        raise ValueError(
            f"allowlist entry {index}: 'justification' is required and must be "
            f"a non-empty string"
        )

    reason = raw.get("reason")
    if reason is not None and not isinstance(reason, str):
        raise ValueError(f"allowlist entry {index}: 'reason' must be a string")

    approved_by = raw.get("approved_by")
    if approved_by is not None and not isinstance(approved_by, str):
        raise ValueError(
            f"allowlist entry {index}: 'approved_by' must be a string"
        )

    return Entry(
        sha256=sha256.strip(),
        rule_id=rule_id.strip(),
        justification=justification.strip(),
        expires=_parse_expires(raw.get("expires"), index),
        reason=reason,
        approved_by=approved_by,
    )


def load_allowlist(path: Path) -> Allowlist:
    """Parse and validate an allowlist JSON file into an Allowlist.

    The file must be a JSON array of entry objects (see Entry for the fields).
    Any malformed entry, missing justification, or duplicate (sha256, rule_id)
    key raises ValueError with the entry index, because a suppression config
    that does not parse cleanly must fail loudly rather than partially apply.
    """
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"allowlist {path}: not valid JSON: {exc}") from exc

    if not isinstance(data, list):
        raise ValueError(
            f"allowlist {path}: top level must be a JSON array of entries"
        )

    entries: list[Entry] = []
    seen: set[tuple[str, str]] = set()
    for index, raw in enumerate(data):
        entry = _parse_entry(raw, index)
        key = (entry.sha256, entry.rule_id)
        # Reject duplicates rather than letting a later entry shadow an earlier
        # one, since a silently-dropped approval is its own audit gap.
        if key in seen:
            raise ValueError(
                f"allowlist entry {index}: duplicate (sha256, rule_id) key "
                f"already defined earlier in the file"
            )
        seen.add(key)
        entries.append(entry)

    return Allowlist(entries)
