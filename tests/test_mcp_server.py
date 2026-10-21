"""Spec 2b: the `hayward mcp` server and the `scan_path` verdict behind it.

The verdict is a plain function, so most of this runs without the MCP SDK.
Only the stdio round trip needs the `hayward[mcp]` extra and skips without it.
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys

import pytest

from hayward.mcp_server import scan_path


class _Evil:
    def __reduce__(self):
        return (os.system, ("echo pwned",))


def _evil(tmp_path):
    target = tmp_path / "evil.pkl"
    target.write_bytes(pickle.dumps(_Evil(), protocol=2))
    return target


def _benign(tmp_path):
    target = tmp_path / "ok.pkl"
    target.write_bytes(pickle.dumps({"weights": [1, 2, 3]}, protocol=2))
    return target


def test_malicious_file_is_blocked(tmp_path):
    result = scan_path(str(_evil(tmp_path)))
    assert result["verdict"] == "blocked"
    assert result["findings"]


def test_benign_file_is_clean(tmp_path):
    result = scan_path(str(_benign(tmp_path)))
    assert result == {"verdict": "clean", "fail_on": "high", "findings": []}


def test_directory_is_scanned(tmp_path):
    _benign(tmp_path)
    _evil(tmp_path)
    assert scan_path(str(tmp_path))["verdict"] == "blocked"


def test_missing_path_raises_value_error(tmp_path):
    with pytest.raises(ValueError, match="no such file"):
        scan_path(str(tmp_path / "absent.pkl"))


def test_unknown_threshold_raises_value_error(tmp_path):
    with pytest.raises(ValueError, match="fail_on"):
        scan_path(str(_benign(tmp_path)), fail_on="severe")


@pytest.mark.skipif(
    sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="chmod 000 does not make a file unreadable here",
)
def test_unreadable_file_is_incomplete(tmp_path):
    target = _benign(tmp_path)
    target.chmod(0)
    try:
        result = scan_path(str(target))
        strict = scan_path(str(target), fail_on="low")
    finally:
        target.chmod(0o600)
    # The scanner reports an unreadable file as MFV-SKIP-003 at low severity:
    # under the default gate that is a coverage gap, not a block.
    assert result["verdict"] == "incomplete"
    assert [f["rule_id"] for f in result["findings"]] == ["MFV-SKIP-003"]
    assert strict["verdict"] == "blocked"


def test_core_works_without_the_sdk(tmp_path):
    """`import hayward`, `hayward scan` and the plain verdict never need mcp;
    `hayward mcp` says how to install it and exits 2."""
    benign = _benign(tmp_path)
    code = (
        "import sys\n"
        "sys.modules['mcp'] = None\n"  # makes any `import mcp` raise ImportError
        "import hayward\n"
        "from hayward import cli\n"
        "from hayward.mcp_server import scan_path\n"
        f"assert cli.main(['scan', {str(benign)!r}]) == 0\n"
        f"assert scan_path({str(benign)!r})['verdict'] == 'clean'\n"
        "sys.exit(cli.main(['mcp']))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 2, proc.stderr
    assert 'pip install "hayward[mcp]"' in proc.stderr


def test_stdio_server_lists_one_tool_and_returns_the_same_json(tmp_path):
    pytest.importorskip("mcp")
    import anyio
    from mcp.client.client import Client
    from mcp.client.stdio import StdioServerParameters

    evil = str(_evil(tmp_path))
    params = StdioServerParameters(command=sys.executable, args=["-m", "hayward", "mcp"])

    async def session():
        async with Client(params) as client:
            tools = (await client.list_tools()).tools
            hit = await client.call_tool("scan_path", {"path": evil})
            miss = await client.call_tool("scan_path", {"path": str(tmp_path / "absent")})
            return tools, hit, miss

    tools, hit, miss = anyio.run(session)
    assert [tool.name for tool in tools] == ["scan_path"]
    assert not hit.is_error
    assert json.loads(hit.content[0].text) == scan_path(evil)
    assert miss.is_error
    assert "no such file" in miss.content[0].text
