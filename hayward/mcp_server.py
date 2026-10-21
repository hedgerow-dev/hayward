"""A stdio MCP server so a coding agent can scan a model file before loading it.

    hayward mcp

It exposes one tool, `scan_path`. The verdict logic is the plain function of
the same name, which needs nothing beyond the core package; only `serve()`
imports the MCP SDK (the optional `hayward[mcp]` extra). Like every other
entry point it reads bytes, never loads a model, and makes no network calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hayward.cli import _THRESHOLDS
from hayward.findings import is_coverage_gap
from hayward.scanner import ModelFileScanner


def scan_path(path: str, fail_on: str = "high") -> dict[str, Any]:
    """Scan a file or directory and return a verdict with its findings.

    `blocked` means a finding at or above `fail_on`, the same gate as
    `hayward scan --fail-on`. `incomplete` means nothing reached the gate but
    some file was not fully read, so the result says nothing about it.
    Raises ValueError for a missing path or an unknown threshold.
    """
    if fail_on not in _THRESHOLDS:
        raise ValueError(f"fail_on must be one of: {', '.join(_THRESHOLDS)}")
    target = Path(path)
    if not target.exists():
        raise ValueError(f"no such file or directory: {path}")

    scanner = ModelFileScanner()
    if target.is_dir():
        findings = scanner.scan_directory(target)
    else:
        findings = scanner.scan_file(target)

    limit = _THRESHOLDS[fail_on]
    if any(f.severity_order <= limit for f in findings):
        verdict = "blocked"
    elif any(is_coverage_gap(f) for f in findings):
        verdict = "incomplete"
    else:
        verdict = "clean"
    return {
        "verdict": verdict,
        "fail_on": fail_on,
        "findings": [f.to_dict() for f in findings],
    }


def serve() -> None:
    """Run the MCP server on stdin/stdout until the client disconnects."""
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
    from mcp.types import ToolAnnotations

    from hayward import __version__

    server = MCPServer("hayward", version=__version__)

    @server.tool(
        name="scan_path",
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
    )
    def scan_path_tool(path: str, fail_on: str = "high") -> str:
        """Scan a model file or directory for unsafe content without loading it.

        Returns JSON with a verdict (blocked, clean or incomplete) and the
        findings. Call this before loading any model file.
        """
        try:
            return json.dumps(scan_path(path, fail_on))
        except (OSError, ValueError) as exc:
            raise ToolError(str(exc)) from exc

    server.run("stdio")
