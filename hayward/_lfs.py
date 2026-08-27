"""Git LFS pointer detection.

Extracted verbatim from ``hayward.scanner`` (HW-147). A Git LFS pointer is a
few lines of text standing in for content stored elsewhere: the bytes on disk
are not the model, so scanning them scans nothing.

``_read_file_magic`` stays defined on ``hayward.scanner`` (it is shared with the
LFS probe and the dispatch/discovery code) and is referenced lazily through
``_scanner._read_file_magic`` so the circular import stays safe.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import hayward.scanner as _scanner
from hayward.findings import Category, Finding, Severity

# A Git LFS pointer is a few lines of text standing in for content stored
# elsewhere: the bytes on disk are not the model, so scanning them scans
# nothing. The version line is the identifier; `oid` and `size` follow, and
# the spec tolerates extension keys and trailing whitespace.
_LFS_VERSION_LINE = "version https://git-lfs.github.com/spec/v1"
_LFS_OID_RE = re.compile(r"oid sha256:([0-9a-f]{64})")
_LFS_SIZE_RE = re.compile(r"size ([0-9]+)")
_LFS_PROBE_BYTES = 1024


def _parse_lfs_pointer(data: bytes) -> dict[str, Any] | None:
    """Parse a Git LFS pointer prefix.

    Returns None unless the first line is the LFS version marker (anything
    else is not a pointer, whatever it claims). Otherwise returns `oid` and
    `size`, each None when that key is absent or malformed.
    """
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    lines = [line.strip() for line in text.splitlines()]
    if not lines or lines[0] != _LFS_VERSION_LINE:
        return None
    fields: dict[str, Any] = {"oid": None, "size": None}
    for line in lines[1:]:
        match = _LFS_OID_RE.fullmatch(line)
        if match:
            fields["oid"] = match.group(1)
            continue
        match = _LFS_SIZE_RE.fullmatch(line)
        if match:
            fields["size"] = int(match.group(1))
    return fields


def _lfs_pointer_finding(file_path: Path) -> Finding | None:
    """MFV-LFS-001: the file is a Git LFS pointer, a placeholder a few lines
    long that replaces the real content until `git lfs pull` fetches it. The
    scanner read the file completely and found no model in it, because the
    model was never there: a coverage gap, not a verdict."""
    head = _scanner._read_file_magic(file_path, _LFS_PROBE_BYTES)
    if not head.startswith(b"version "):
        return None
    pointer = _parse_lfs_pointer(head)
    if pointer is None:
        return None
    malformed = pointer["oid"] is None or pointer["size"] is None
    declared = []
    if pointer["size"] is not None:
        declared.append(f"declared size {pointer['size']} bytes")
    if pointer["oid"] is not None:
        declared.append(f"oid sha256:{pointer['oid']}")
    where = f" ({'; '.join(declared)})" if declared else ""
    message = (
        f"Git LFS pointer: the file on disk is a placeholder{where} for content "
        "stored elsewhere, so the real bytes were never fetched and nothing was "
        "scanned. NOT a clean verdict."
    )
    if malformed:
        message += " The pointer itself is malformed: oid or size is missing or invalid."
    metadata: dict[str, Any] = {"lfs_malformed": malformed}
    if pointer["oid"] is not None:
        metadata["lfs_oid"] = pointer["oid"]
    if pointer["size"] is not None:
        metadata["lfs_declared_size"] = pointer["size"]
    return Finding(
        rule_id="MFV-LFS-001",
        message=message,
        severity=Severity.INFO,
        category=Category.AI_ML,
        file_path=str(file_path),
        confidence=0.75 if malformed else 0.95,
        metadata=metadata,
    )
