"""Scanner-evasion gadgets found on the Hub (November 2026) must be CRITICAL.

Each one reaches a shell or exec without naming os.system or eval in a way the
plain deny list saw. Before 1.18 they came out HIGH, and four (runsource,
trace, codeop, torch.hub) came out INFO when the payload text did not look like
a shell command. Fixtures are opcode-built in tests/gadget_pickles.py; nothing
is unpickled.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from gadget_pickles import GADGETS, Call, build  # noqa: E402

from hayward import ModelFileScanner  # noqa: E402

ORDER = ["info", "low", "medium", "high", "critical"]


def worst(tmp_path: Path, data: bytes) -> str:
    p = tmp_path / "model.pkl"
    p.write_bytes(data)
    found = ModelFileScanner().scan_file(p)
    return max((f.severity.value for f in found), key=ORDER.index, default="clean")


@pytest.mark.parametrize("name", sorted(GADGETS))
def test_gadget_is_critical(tmp_path, name):
    assert worst(tmp_path, GADGETS[name]) == "critical"


@pytest.mark.parametrize("module", ["numpy", "torch.nn", "collections"])
def test_importing_an_ordinary_module_is_not_critical(tmp_path, module):
    assert worst(tmp_path, build(Call("importlib.import_module", module))) != "critical"


def test_a_project_package_named_code_is_not_denied(tmp_path):
    # `code` and `trace` are also common names for a project's own package.
    assert worst(tmp_path, build(Call("code.models.Net"))) != "critical"
    assert worst(tmp_path, build(Call("trace.utils.Config"))) != "critical"


def test_plain_copyreg_reconstruction_stays_clean(tmp_path):
    assert worst(tmp_path, pickle.dumps({"a": [1, 2], "b": (3,)}, protocol=2)) in ("clean", "info")
