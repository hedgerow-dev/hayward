"""Content-hash scan cache.

Re-scanning a file that has not changed since the last run is wasted work in a
large tree. This cache stores a file's findings keyed by the sha256 of its
bytes, so a second run over an unchanged file returns the stored findings
instead of parsing it again.

Two things gate a hit, and both matter for correctness:

- The file's content sha256 (not its mtime). An mtime can be preserved across
  an edit (a restore, a `touch -r`, a checkout), which would let a changed file
  keep a stale verdict. Hashing the bytes means any edit misses.

- A version tag that changes when the scanner logic changes. If the scanner
  gains a rule that would now flag a file, but the cache still holds the old
  "clean" result under the same content hash, the new finding would be hidden.
  We use hayward.__version__ as the tag, so every release bump invalidates the
  whole cache and forces a re-scan. Importing __version__ is cheap: it is
  defined in hayward/__init__.py and does not require the scanner to do work.

The scanner is never imported here. get_or_scan takes the scan function as an
injected callable, which keeps this module decoupled from scanner internals and
trivially testable with a fake.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from hayward import __version__
from hayward.findings import Category, Finding, Severity

# The callable get_or_scan invokes on a miss: it is handed the file path and
# returns that file's findings. The cli passes the real scanner's scan_file;
# tests pass a fake.
ScanCallable = Callable[[Path], list[Finding]]


def _finding_from_dict(data: dict[str, Any], file_path: str) -> Finding:
    """Rebuild a Finding from its to_dict() form.

    Finding.to_dict emits the severity and category as their enum *values*
    (strings) and stores the path under the key "file"; we map those back to
    the enums here. It does not emit `engine`, so the rebuilt Finding keeps the
    dataclass default. That is acceptable for a cache: the engine of a cached
    hayward scan is hayward, which is the default.

    `file_path` is the path being scanned right now, and it (not the stored
    "file") becomes the finding's path. The cache key includes the path, so on
    a hit they are equal anyway; stamping the live path is belt and suspenders
    against a finding ever being reported against the wrong file.
    """
    return Finding(
        rule_id=data["rule_id"],
        message=data["message"],
        severity=Severity(data["severity"]),
        category=Category(data["category"]),
        file_path=file_path,
        confidence=data.get("confidence", 1.0),
        cwe_ids=list(data.get("cwe_ids", [])),
        metadata=dict(data.get("metadata", {})),
    )


class ScanCache:
    """A store of scan findings keyed by (path, content sha256), persisted as
    JSON.

    A finding is NOT a pure function of a file's bytes: the scanner dispatches
    on the file's name and suffix (keras_metadata.pb, saved_model.pb, .7z, ...),
    and a finding carries its own path. So the key is the path together with the
    content hash, not the hash alone. That means a cached result is served only
    when the same path holds the same bytes it did last time, which is exactly
    "this file has not changed". Two byte-identical files at different paths, or
    the same path with different bytes, both miss and are scanned on their own.

    `version_tag` stamps the store; a cache loaded under a different tag is
    treated as empty, because scanner logic (or a scan-affecting flag folded
    into the tag by the caller) may have changed under it.
    """

    def __init__(
        self,
        version_tag: str = __version__,
        entries: dict[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        self.version_tag = version_tag
        # "<path>\0<sha256>" -> list of finding dicts (to_dict form).
        self.entries: dict[str, list[dict[str, Any]]] = entries or {}

    @classmethod
    def load(cls, path: Path, version_tag: str = __version__) -> ScanCache:
        """Load a cache from JSON at `path`.

        A missing or unreadable file, or one written under a different version
        tag, yields an empty cache stamped with the current tag. The stale
        entries are simply dropped: keeping them would risk serving a verdict
        from a scanner that no longer exists.
        """
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls(version_tag=version_tag)

        if not isinstance(document, dict) or document.get("version") != version_tag:
            return cls(version_tag=version_tag)

        stored = document.get("entries", {})
        if not isinstance(stored, dict):
            return cls(version_tag=version_tag)

        return cls(version_tag=version_tag, entries=stored)

    def save(self, path: Path) -> None:
        """Persist the cache to JSON at `path`, stamped with the version tag."""
        document = {"version": self.version_tag, "entries": self.entries}
        path.write_text(json.dumps(document), encoding="utf-8")

    def get_or_scan(self, file_path: Path, scan_callable: ScanCallable) -> list[Finding]:
        """Return cached findings for `file_path`, or scan and store on a miss.

        On a hit (this path held these bytes before) the stored findings are
        rebuilt and returned without calling scan_callable. On a miss,
        scan_callable(file_path) runs, its findings are stored, and returned.

        A file that cannot be hashed (an OSError: unreadable, vanished, a
        directory) is scanned directly and not cached, so the scanner's own
        per-file firewall degrades it to a coverage gap. Hashing must not turn
        one unreadable file into a hard failure that aborts the whole run.
        """
        try:
            key = self.entry_key(file_path)
        except OSError:
            return scan_callable(file_path)
        cached = self.entries.get(key)
        if cached is not None:
            rebuilt = _rebuild_findings(cached, str(file_path))
            if rebuilt is not None:
                return rebuilt
            # A corrupt or tampered entry must not abort the run: drop it and
            # rescan, so the file is analysed rather than the scan aborting.
        findings = scan_callable(file_path)
        self.entries[key] = [f.to_dict() for f in findings]
        return findings

    @classmethod
    def entry_key(cls, file_path: Path) -> str:
        """The cache key for a file: its path joined to its content hash.

        Raises OSError if the file cannot be read. Callers that partition a
        file list against the cache (the parallel path in the cli) use this so
        their keys match what get_or_scan stores.
        """
        return f"{file_path}\x00{cls._hash_file(file_path)}"

    @staticmethod
    def _hash_file(file_path: Path) -> str:
        """sha256 of the file's bytes, read in chunks to bound memory."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()


def _rebuild_findings(items: object, file_path: str) -> list[Finding] | None:
    """Rebuild a cached findings list, or None if the entry is malformed.

    A cache file is data on disk that could be truncated, hand-edited or
    tampered with. A bad entry is treated as a miss (rescan), never as a reason
    to crash the scan.
    """
    if not isinstance(items, list):
        return None
    try:
        return [_finding_from_dict(item, file_path) for item in items]
    except (KeyError, ValueError, TypeError):
        return None
