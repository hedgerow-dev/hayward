"""MFV-ARCHIVE-001: archive member-name path traversal (zip slip, CWE-22).

The container walks read members into memory and never extract to disk, so
Hayward is not itself the zip-slip sink. The consumer that later unpacks the
same file is, and a member named `../../x`, `/etc/x`, a Windows drive path, or
a UNC path escapes the target directory on extraction. The rule fires across
every container that carries member names: the torch zip (`.pt`), a nested zip
inside a `.mar`, and the tar-based `.nemo`. A normal nested member such as
`archive/data.pkl` must never fire.

Fixtures are hand-built and tiny. Member bodies are a 4-byte pickle-shaped blob
so the walk has something to read; the rule keys only on the names, so the
bodies are irrelevant to what is asserted here.
"""

from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

from hayward.scanner import ModelFileScanner

_MEMBER_BODY = b"\x80\x04}."  # short, pickle-opener shaped; content does not matter


def _zip(names: list[str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name in names:
            zf.writestr(name, _MEMBER_BODY)
    return buf.getvalue()


def _tar(names: list[str]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name in names:
            info = tarfile.TarInfo(name)
            info.size = len(_MEMBER_BODY)
            tf.addfile(info, io.BytesIO(_MEMBER_BODY))
    return buf.getvalue()


def _mar_wrapping_pt(inner_names: list[str], outer_names: list[str] | None = None) -> bytes:
    """A `.mar` (zip) whose `model.pt` member is itself a torch zip. The rule
    must reach the inner names one container deeper via the nested-zip walk."""
    inner = _zip(inner_names)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("model.pt", inner)
        for name in outer_names or ["MAR-INF/MANIFEST.json"]:
            zf.writestr(name, b"{}")
    return buf.getvalue()


def _scan(tmp_path: Path, name: str, blob: bytes) -> list[str]:
    p = tmp_path / name
    p.write_bytes(blob)
    return [f.rule_id for f in ModelFileScanner().scan_file(p)]


class TestTorchZip:
    def test_traversing_member_fires(self, tmp_path):
        rules = _scan(tmp_path, "evil.pt", _zip(["../../etc/cron.d/x", "archive/data.pkl"]))
        assert "MFV-ARCHIVE-001" in rules

    def test_absolute_member_fires(self, tmp_path):
        assert "MFV-ARCHIVE-001" in _scan(tmp_path, "abs.pt", _zip(["/etc/passwd"]))

    def test_drive_absolute_member_fires(self, tmp_path):
        assert "MFV-ARCHIVE-001" in _scan(tmp_path, "drv.pt", _zip(["C:\\windows\\x"]))

    def test_unc_member_fires(self, tmp_path):
        assert "MFV-ARCHIVE-001" in _scan(tmp_path, "unc.pt", _zip(["\\\\host\\share\\x"]))

    def test_benign_nested_member_does_not_fire(self, tmp_path):
        rules = _scan(tmp_path, "ok.pt", _zip(["archive/data.pkl", "archive/version"]))
        assert "MFV-ARCHIVE-001" not in rules


class TestNemoTar:
    def test_traversing_member_fires(self, tmp_path):
        rules = _scan(tmp_path, "evil.nemo", _tar(["../../../etc/x", "model_weights.ckpt"]))
        assert "MFV-ARCHIVE-001" in rules

    def test_absolute_member_fires(self, tmp_path):
        assert "MFV-ARCHIVE-001" in _scan(tmp_path, "abs.nemo", _tar(["/etc/passwd"]))

    def test_benign_members_do_not_fire(self, tmp_path):
        rules = _scan(tmp_path, "ok.nemo", _tar(["model_config.yaml", "model_weights.ckpt"]))
        assert "MFV-ARCHIVE-001" not in rules


class TestNestedMar:
    def test_inner_traversal_fires_one_container_deeper(self, tmp_path):
        rules = _scan(tmp_path, "evil.mar", _mar_wrapping_pt(["../../evil", "archive/data.pkl"]))
        assert "MFV-ARCHIVE-001" in rules

    def test_outer_traversal_fires(self, tmp_path):
        rules = _scan(
            tmp_path, "outer.mar",
            _mar_wrapping_pt(["archive/data.pkl"], outer_names=["../../../etc/x"]),
        )
        assert "MFV-ARCHIVE-001" in rules

    def test_benign_nested_mar_does_not_fire(self, tmp_path):
        rules = _scan(tmp_path, "ok.mar", _mar_wrapping_pt(["archive/data.pkl"]))
        assert "MFV-ARCHIVE-001" not in rules


class TestFindingShape:
    def test_severity_category_and_cwe(self, tmp_path):
        p = tmp_path / "evil.pt"
        p.write_bytes(_zip(["../../etc/x"]))
        finding = next(
            f for f in ModelFileScanner().scan_file(p) if f.rule_id == "MFV-ARCHIVE-001"
        )
        assert finding.severity.name == "MEDIUM"
        assert finding.category.value == "path_traversal"
        assert 22 in (finding.cwe_ids or [])
