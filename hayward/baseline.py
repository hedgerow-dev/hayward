"""Baseline / diff mode for brownfield CI.

A team adopting the scanner on an existing codebase starts with a pile of
findings that predate this change. Failing every build on the whole backlog
teaches people to ignore the gate, which defeats it. The useful contract is:
fail only on findings the *current* change introduced, and let the backlog burn
down separately.

So we snapshot the backlog once (`hayward scan -f json > baseline.json`), and on
every later run compare a fresh scan against that snapshot:

    baseline_keys = load_baseline(Path("baseline.json"))
    result = diff(baseline_keys, current_findings, root=scan_root)
    # result.new    -> introduced by this change
    # result.fixed  -> in the baseline but gone now
    # result.unchanged -> pre-existing, still present

The whole design rests on one question: when are two findings, produced by two
different scans, "the same finding"? That is `finding_key`, and its trade-off is
documented there.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hayward.findings import Finding

# A finding identity is a 3-tuple. Kept as a plain tuple (not a string) so it is
# both hashable for set membership and legible when logged or listed in .fixed.
FindingKey = tuple[str, str, str]


def _fields(finding_like: Finding | dict[str, Any]) -> tuple[str, str, dict, str]:
    """Pull (rule_id, file_path, metadata, message) from either a live Finding
    or a decoded JSON-report entry.

    The two shapes disagree on one field name: the dataclass calls it
    `file_path`, but `report.to_json` serialises it as `file` (see
    Finding.to_dict). Tolerating both is what lets the same key function run over
    a baseline loaded from disk and over the Finding objects of a fresh scan.
    """
    if isinstance(finding_like, Finding):
        return (
            finding_like.rule_id,
            finding_like.file_path,
            finding_like.metadata,
            finding_like.message,
        )
    rule_id = finding_like.get("rule_id", "")
    # "file" is the report shape; "file_path" is a courtesy fallback for a
    # caller that handed us a Finding.to_dict()-like dict under the other name.
    file_path = finding_like.get("file", finding_like.get("file_path", ""))
    # metadata may be absent or explicitly null in a report; normalise to {}.
    metadata = finding_like.get("metadata") or {}
    message = finding_like.get("message", "")
    return (rule_id, file_path, metadata, message)


def _normalize_path(file_path: str, root: Path | str | None) -> str:
    """Reduce a finding's path to a root-relative POSIX string.

    Two noise sources have to be cancelled so the same file matches across runs:

    1. Absolute vs relative invocation. A baseline captured with
       `hayward scan /abs/checkout/models` stores absolute paths; a CI run in a
       different checkout stores different absolute paths, or relative ones.
       Relativising each report against *its own* root collapses both to the
       same tail (e.g. "models/m.bin"), because the scanner's file_path always
       carries the scanned-root prefix (it walks with root.rglob).
    2. Path-separator differences. `as_posix()` makes a Windows-authored
       baseline match a Unix scan and vice versa.

    When the path is not under `root` (or no root is known) we keep it as-is,
    only POSIX-normalised. That is the honest fallback: we would rather compare
    two unrelativised paths than silently invent a match.
    """
    path = Path(file_path)
    if root is not None:
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            # file_path is not under root: fall through to the raw form.
            pass
    return path.as_posix()


def _detail_hash(metadata: dict[str, Any], message: str) -> str:
    """The third component of a key: what distinguishes two hits of the same
    rule in the same file.

    rule_id + file alone is too loose: a rule can fire twice in one file (two
    unsafe globals, two embedded streams), and collapsing those to one key would
    hide a genuinely new second issue behind a pre-existing first one.

    So we add a per-hit discriminator, preferring the rule's structured
    `metadata` over the prose `message` when metadata exists:

    - metadata is the machine-stable locus of the finding (an offset, a symbol,
      a global name). Keying on it means a rule that later rewords its message
      does NOT churn every historical finding into new+fixed pairs.
    - message is the fallback when a rule emits no metadata. A reworded message
      then reads as a new finding. That errs *loud*, which is the safe direction
      for a security gate: a possibly-changed issue fails the build rather than
      being silently absorbed as unchanged. The cost is churn on cosmetic edits,
      accepted only where there is nothing more stable to key on.

    metadata is serialised with sorted keys so dict ordering is not part of
    identity; default=str keeps non-JSON values (e.g. an Enum) from raising.
    """
    if metadata:
        basis = json.dumps(metadata, sort_keys=True, default=str)
    else:
        basis = message
    # 16 hex chars (64 bits) is far below any plausible collision rate for the
    # handful of findings in one file, and keeps a logged key compact.
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]


def finding_key(
    finding_like: Finding | dict[str, Any], root: Path | str | None = None
) -> FindingKey:
    """Stable identity for one finding across scans.

    (rule_id, root-relative POSIX path, detail hash). Robust against path shape
    and finding ordering; sensitive to a real change of rule, file, or the
    per-hit detail. See `_normalize_path` and `_detail_hash` for the two
    non-obvious halves and their trade-offs.
    """
    rule_id, file_path, metadata, message = _fields(finding_like)
    return (rule_id, _normalize_path(file_path, root), _detail_hash(metadata, message))


def load_baseline(
    path: Path | str, root: Path | str | None = None
) -> set[FindingKey]:
    """Read a baseline snapshot into a set of finding keys.

    Parses the `{"findings": [...]}` envelope that `hayward scan -f json`
    produces, and also tolerates a bare `[...]` list for a hand-written or
    post-processed baseline.

    The envelope records the scan `root` it was captured with, so the baseline
    is self-describing: we relativise its paths against that embedded root
    automatically, which is what makes an absolute-path baseline match a later
    relative-path scan. An explicit `root` argument overrides the embedded one
    for the unusual case where the snapshot was moved relative to its tree.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        findings = data.get("findings", [])
        embedded_root = data.get("root")
    else:
        # Bare list: no envelope, so no embedded root to normalise against.
        findings = data
        embedded_root = None
    base_root = root if root is not None else embedded_root
    return {finding_key(f, base_root) for f in findings}


@dataclass
class BaselineDiff:
    """The three-way split of a scan against a baseline.

    `new` and `unchanged` carry the live Finding objects from the current scan,
    so a caller can render or threshold them. `fixed` carries only keys: the
    baseline is loaded as keys, so the original Finding objects are not
    available, and listing what disappeared by key is enough to report progress.
    """

    new: list[Finding] = field(default_factory=list)
    unchanged: list[Finding] = field(default_factory=list)
    fixed: list[FindingKey] = field(default_factory=list)


def diff(
    baseline_keys: set[FindingKey],
    current_findings: list[Finding],
    root: Path | str | None = None,
) -> BaselineDiff:
    """Compare a fresh scan against a baseline key set.

    `root` is the current scan's root, used to relativise the current findings'
    paths the same way `load_baseline` relativised the baseline's. Pass the
    value the cli already computes for the scan (see cli `root`).
    """
    result = BaselineDiff()
    seen: set[FindingKey] = set()
    for finding in current_findings:
        key = finding_key(finding, root)
        seen.add(key)
        if key in baseline_keys:
            result.unchanged.append(finding)
        else:
            result.new.append(finding)
    # In the baseline but not in this scan: the issue is gone (or the file was).
    result.fixed = sorted(baseline_keys - seen)
    return result


def new_findings_fail(new: list[Finding], fail_on_order: int) -> bool:
    """Exit contract for --baseline mode.

    True when any NEW finding sits at or above the fail threshold. `fail_on_order`
    is the integer limit the cli already derives from --fail-on (critical=0,
    high=1, ... info=4, and never=-1 which nothing can reach). A finding fails
    when `severity_order <= fail_on_order`, exactly the test the non-baseline
    path applies, only now scoped to the findings this change introduced.

    Passing the integer the cli already has, rather than the --fail-on name,
    avoids duplicating its threshold table (including the "never" sentinel) here.
    """
    return any(f.severity_order <= fail_on_order for f in new)
