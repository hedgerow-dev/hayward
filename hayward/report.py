"""Report rendering.

Three formats, one renderer each, shared by the command line and the desktop
app so a report is identical whichever produced it.

HTML is the default for sharing: one self-contained file with no external
stylesheet, font or script, so it opens from an email attachment on a machine
with no network and renders the same. That is the same constraint the scanner
works under, and it is deliberate rather than incidental.

Markdown is for pasting into a ticket or a pull request. JSON is for machines.
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path

from hayward.findings import Finding, is_coverage_gap

_ORDER = ("critical", "high", "medium", "low", "info")

_SEVERITY_COLOUR = {
    "critical": "#c0392b",
    "high": "#d35400",
    "medium": "#b7950b",
    "low": "#2874a6",
    "info": "#7b8794",
}


def _counts(findings: list[Finding]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for finding in findings:
        counts[finding.severity.value] = counts.get(finding.severity.value, 0) + 1
    return counts


def _ordered(findings: list[Finding]) -> list[Finding]:
    return sorted(findings, key=lambda f: (f.severity_order, f.file_path))


def _relative(finding: Finding, root: Path | None) -> str:
    path = Path(finding.file_path)
    if root is not None:
        try:
            return str(path.relative_to(root))
        except ValueError:
            pass
    return str(path)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def to_json(findings: list[Finding], root: Path | None, version: str) -> str:
    return json.dumps({
        "tool": "hayward",
        "version": version,
        "generated": _timestamp(),
        "root": str(root) if root else None,
        "counts": _counts(findings),
        "findings": [f.to_dict() for f in _ordered(findings)],
        "coverage_gaps": [f.file_path for f in findings if is_coverage_gap(f)],
    }, indent=2)


def to_markdown(findings: list[Finding], root: Path | None, version: str) -> str:
    counts = _counts(findings)
    lines = [
        "# Model file scan",
        "",
        f"- **Scanned:** `{root}`" if root else "- **Scanned:** (unspecified)",
        f"- **Generated:** {_timestamp()}",
        f"- **Tool:** hayward {version}",
        "",
    ]

    if not findings:
        lines += ["No findings.", ""]
        return "\n".join(lines)

    summary = ", ".join(f"{counts[s]} {s}" for s in _ORDER if s in counts)
    lines += [f"**{len(findings)} finding(s):** {summary}", ""]

    gaps = [f for f in findings if is_coverage_gap(f)]
    if gaps:
        lines += [
            f"> {len(gaps)} file(s) could not be fully read. Those are not clean",
            "> verdicts: the content was never analysed.",
            "",
        ]

    lines += ["| Severity | Rule | File |", "|---|---|---|"]
    for finding in _ordered(findings):
        lines.append(
            f"| {finding.severity.value.upper()} | `{finding.rule_id}` | "
            f"`{_relative(finding, root)}` |"
        )
    lines += ["", "## Detail", ""]

    for finding in _ordered(findings):
        cwe = (
            "  \n**CWE:** " + ", ".join(f"CWE-{c}" for c in finding.cwe_ids)
            if finding.cwe_ids else ""
        )
        lines += [
            f"### {finding.severity.value.upper()}  `{finding.rule_id}`",
            "",
            f"**File:** `{_relative(finding, root)}`{cwe}",
            "",
            finding.message,
            "",
        ]

    return "\n".join(lines)


def to_html(findings: list[Finding], root: Path | None, version: str) -> str:
    """A single self-contained page. No external requests of any kind."""
    counts = _counts(findings)
    esc = html.escape

    chips = "".join(
        f'<span class="chip" style="--c:{_SEVERITY_COLOUR[s]}">'
        f'<b>{counts[s]}</b> {s}</span>'
        for s in _ORDER if s in counts
    ) or '<span class="chip" style="--c:#2874a6"><b>0</b> findings</span>'

    gaps = [f for f in findings if is_coverage_gap(f)]
    gap_note = (
        f'<p class="note"><b>{len(gaps)} file(s) could not be fully read.</b> '
        "Those are not clean verdicts: the content was never analysed.</p>"
        if gaps else ""
    )

    if findings:
        rows = "".join(
            "<tr>"
            f'<td><span class="dot" style="--c:{_SEVERITY_COLOUR[f.severity.value]}"></span>'
            f"{f.severity.value.upper()}</td>"
            f"<td><code>{esc(f.rule_id)}</code></td>"
            f"<td><code>{esc(_relative(f, root))}</code></td>"
            f"<td>{esc(f.message)}"
            + (
                f'<div class="cwe">{esc(", ".join(f"CWE-{c}" for c in f.cwe_ids))}</div>'
                if f.cwe_ids else ""
            )
            + "</td></tr>"
            for f in _ordered(findings)
        )
        table = (
            "<table><thead><tr><th>Severity</th><th>Rule</th><th>File</th>"
            f"<th>Detail</th></tr></thead><tbody>{rows}</tbody></table>"
        )
    else:
        table = '<p class="empty">No findings.</p>'

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Model file scan</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
         Helvetica, Arial, sans-serif; margin: 0; padding: 40px 32px;
         color: #1f2933; background: #fff; }}
  main {{ max-width: 1100px; margin: 0 auto; }}
  h1 {{ font-size: 22px; margin: 0 0 4px; }}
  .meta {{ color: #7b8794; font-size: 13px; margin: 0 0 20px; }}
  .meta code {{ color: inherit; }}
  .chip {{ display: inline-block; border: 1px solid var(--c); color: var(--c);
           border-radius: 999px; padding: 3px 11px; margin: 0 6px 6px 0;
           font-size: 13px; }}
  .note {{ border-left: 3px solid #b7950b; background: #fdf9ec;
           padding: 10px 14px; margin: 18px 0; font-size: 14px; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 20px;
           font-size: 14px; }}
  th {{ text-align: left; font-size: 11px; letter-spacing: .06em;
        text-transform: uppercase; color: #7b8794; font-weight: 600;
        border-bottom: 1px solid #dfe3e8; padding: 0 12px 8px 0; }}
  td {{ border-bottom: 1px solid #eef1f4; padding: 12px 12px 12px 0;
        vertical-align: top; }}
  td:nth-child(1) {{ white-space: nowrap; font-size: 13px; }}
  td:nth-child(2), td:nth-child(3) {{ white-space: nowrap; }}
  code {{ font: 13px/1.4 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
  .dot {{ display: inline-block; width: 7px; height: 7px; border-radius: 50%;
          background: var(--c); margin-right: 8px; vertical-align: middle; }}
  .cwe {{ color: #7b8794; font-size: 12px; margin-top: 4px; }}
  .empty {{ color: #7b8794; margin-top: 24px; }}
  footer {{ color: #7b8794; font-size: 12px; margin-top: 32px;
            border-top: 1px solid #dfe3e8; padding-top: 14px; }}
  @media (prefers-color-scheme: dark) {{
    body {{ background: #14181d; color: #e4e7eb; }}
    th {{ border-bottom-color: #2b333c; }}
    td {{ border-bottom-color: #232a31; }}
    .note {{ background: #241f10; }}
    footer {{ border-top-color: #2b333c; }}
  }}
</style></head><body><main>
<h1>Model file scan</h1>
<p class="meta">{esc(str(root)) if root else "unspecified target"}
 &middot; {_timestamp()} &middot; hayward {esc(version)}</p>
{chips}
{gap_note}
{table}
<footer>Files were read as bytes. Nothing was loaded, deserialised or
executed. INFO marks content the scanner could not verify rather than content
it judged dangerous.</footer>
</main></body></html>
"""


RENDERERS = {"json": to_json, "markdown": to_markdown, "html": to_html}
SUFFIXES = {"json": ".json", "markdown": ".md", "html": ".html"}


def render(fmt: str, findings: list[Finding], root: Path | None, version: str) -> str:
    return RENDERERS[fmt](findings, root, version)
