"""HW-149: format-breadth additions.

Covers the high-value, low-risk subset of the format-breadth work:

- `.ptl` (PyTorch Lite / mobile) is a zip wrapping a pickle exactly like
  `.pt`, so a malicious pickle inside it must be convicted the same way.
- MFV-TORCH-001: a torch zip that carries executable Python *source*
  (torch.package's `.data/` layout or a TorchScript `code/` directory) is a
  code-execution surface a pickle scan alone misses.

Fixtures are hand-built zips, the same builder pattern as
test_new_formats.py, so this file stands alone.
"""

from __future__ import annotations

import os
import pickle
import zipfile
from pathlib import Path

from hayward.findings import Severity
from hayward.scanner import ModelFileScanner


class _Evil:
    def __reduce__(self):
        # builtins/os.system is on the pickle deny list, so this resolves to
        # an MFV-PICKLE-001 conviction wherever the stream is walked.
        return (os.system, ("echo pwned",))


def _scan(path: Path):
    return ModelFileScanner().scan_file(path)


# ── .ptl: PyTorch Lite / mobile ─────────────────────────────────────


class TestPtlMobileCheckpoint:
    """A `.ptl` zip wrapping a malicious pickle is convicted as a `.pt` is."""

    def test_ptl_zip_with_malicious_pickle_is_convicted(self, tmp_path):
        p = tmp_path / "model.ptl"
        with zipfile.ZipFile(p, "w") as zf:
            # torch mobile export writes the pickle at archive/data.pkl, the
            # same member name the desktop checkpoint uses.
            zf.writestr("archive/data.pkl", pickle.dumps(_Evil(), protocol=2))
            zf.writestr("archive/version", b"3\n")
        findings = _scan(p)
        pickle001 = [f for f in findings if f.rule_id == "MFV-PICKLE-001"]
        assert pickle001, [(f.rule_id, f.message) for f in findings]
        assert pickle001[0].severity == Severity.CRITICAL

    def test_benign_ptl_zip_is_clean(self, tmp_path):
        p = tmp_path / "clean.ptl"
        with zipfile.ZipFile(p, "w") as zf:
            zf.writestr("archive/data.pkl", pickle.dumps({"w": [1.0]}, protocol=2))
            zf.writestr("archive/version", b"3\n")
        findings = _scan(p)
        assert not any(f.rule_id == "MFV-PICKLE-001" for f in findings)
        assert not any(f.rule_id == "MFV-TORCH-001" for f in findings)


# ── MFV-TORCH-001: executable Python source in a torch zip ──────────


class TestTorchSourceMembers:
    """MFV-TORCH-001 fires on packaged source, never on a plain state_dict."""

    def test_torchscript_code_dir_is_flagged(self, tmp_path):
        # A TorchScript / torch.package archive: a `code/` directory of .py
        # that torch.jit compiles and runs on load, alongside the pickle.
        p = tmp_path / "scripted.pt"
        with zipfile.ZipFile(p, "w") as zf:
            zf.writestr("scripted/data.pkl", pickle.dumps({"w": [1.0]}, protocol=2))
            zf.writestr("scripted/code/__torch__/foo.py", b"import os\nos.system('id')\n")
        findings = _scan(p)
        torch001 = [f for f in findings if f.rule_id == "MFV-TORCH-001"]
        assert torch001, [(f.rule_id, f.message) for f in findings]
        assert torch001[0].severity == Severity.HIGH
        assert any(m.endswith("foo.py") for m in torch001[0].metadata["source_members"])

    def test_torch_package_data_layout_is_flagged(self, tmp_path):
        # torch.package's `.data/` layout with a python module the
        # PackageImporter imports on load.
        p = tmp_path / "packaged.pt"
        with zipfile.ZipFile(p, "w") as zf:
            zf.writestr("packaged/.data/extern_modules", b"os\n")
            zf.writestr("packaged/mypkg/model.py", b"print('hi')\n")
            zf.writestr("packaged/data.pkl", pickle.dumps({"w": [1.0]}, protocol=2))
        findings = _scan(p)
        assert any(f.rule_id == "MFV-TORCH-001" for f in findings), (
            [(f.rule_id, f.message) for f in findings]
        )

    def test_plain_state_dict_is_not_flagged(self, tmp_path):
        # The common case: a state_dict checkpoint has no `.py` members, so
        # MFV-TORCH-001 must stay silent.
        p = tmp_path / "state_dict.pt"
        with zipfile.ZipFile(p, "w") as zf:
            zf.writestr("archive/data.pkl", pickle.dumps({"w": [1.0]}, protocol=2))
            zf.writestr("archive/data/0", b"\x00" * 16)
            zf.writestr("archive/version", b"3\n")
        findings = _scan(p)
        assert not any(f.rule_id == "MFV-TORCH-001" for f in findings)
