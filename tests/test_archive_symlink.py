"""MFV-ARCHIVE-002: archive member is a link to an unsafe path (CWE-22/CWE-59).

Distinct from MFV-ARCHIVE-001, which flags an unsafe member *name*. Here the
member *is* a symlink or hard link, and its target is what escapes: an
extractor first creates the link, then a later regular member written through
it lands wherever the link points (`../../outside`, `/etc/cron.d/x`). That is
the classic tar/zip symlink extraction attack (CVE-2007-4559 family). Hayward
never extracts, so it is not the sink; the downstream loader/unpacker is, so a
link with a traversing or absolute target is a strong crafted-archive signal.

The rule must fire across the containers Hayward walks:
  - tar (`.nemo`): a member whose `member.type` is SYMTYPE or LNKTYPE, its
    target in `member.linkname`.
  - zip (`.pt`/`.mar`): a member whose unix mode (the high 16 bits of
    external_attr) has the `S_IFLNK` bit set, its body the link target.

It must stay quiet on an ordinary regular member and on a link to a safe
in-tree relative target such as `weights/shard1`. Fixtures are hand-built and
tiny; a benign regular member is included so each archive is a plausible
checkpoint container.
"""

from __future__ import annotations

import io
import stat
import tarfile
import zipfile
from pathlib import Path

from hayward.scanner import ModelFileScanner

_MEMBER_BODY = b"\x80\x04}."  # short, pickle-opener shaped; content does not matter
_SYMLINK_MODE = (0o120000 | 0o777) << 16  # S_IFLNK | 0777 in external_attr's high word


def _zip_with_link(
    link_name: str, target: str, regular: tuple[str, ...] = ("archive/data.pkl",),
) -> bytes:
    """A zip carrying one S_IFLNK member (`target` as its body) plus benign
    regular members, so the container reads as an ordinary checkpoint zip."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zi = zipfile.ZipInfo(link_name)
        zi.external_attr = _SYMLINK_MODE
        zf.writestr(zi, target)
        for name in regular:
            zf.writestr(name, _MEMBER_BODY)
    return buf.getvalue()


def _zip_regular_only(names: tuple[str, ...]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name in names:
            zf.writestr(name, _MEMBER_BODY)
    return buf.getvalue()


def _tar_with_link(
    link_name: str, target: str, link_type: bytes = tarfile.SYMTYPE,
    regular: tuple[str, ...] = ("model_weights.ckpt",),
) -> bytes:
    """A tar carrying one sym/hard-link member (`target` in linkname) plus
    benign regular members."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        link = tarfile.TarInfo(link_name)
        link.type = link_type
        link.linkname = target
        tf.addfile(link)
        for name in regular:
            info = tarfile.TarInfo(name)
            info.size = len(_MEMBER_BODY)
            tf.addfile(info, io.BytesIO(_MEMBER_BODY))
    return buf.getvalue()


def _tar_regular_only(names: tuple[str, ...]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name in names:
            info = tarfile.TarInfo(name)
            info.size = len(_MEMBER_BODY)
            tf.addfile(info, io.BytesIO(_MEMBER_BODY))
    return buf.getvalue()


def _mar_wrapping_pt_with_link(link_name: str, target: str) -> bytes:
    """A `.mar` (zip) whose `model.pt` member is itself a torch zip holding a
    symlink member -- the rule must reach the link one container deeper via the
    nested-zip walk."""
    inner = _zip_with_link(link_name, target)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("model.pt", inner)
        zf.writestr("MAR-INF/MANIFEST.json", b"{}")
    return buf.getvalue()


def _scan(tmp_path: Path, name: str, blob: bytes) -> list[str]:
    p = tmp_path / name
    p.write_bytes(blob)
    return [f.rule_id for f in ModelFileScanner().scan_file(p)]


class TestZipSymlinkFixtureSurvivesParser:
    """The fixture shape must survive Python's own zipfile round-trip, or the
    detection would be testing a member the real parser never yields."""

    def test_s_iflnk_bit_and_target_are_readable(self):
        blob = _zip_with_link("evil_link", "../../etc/passwd", regular=())
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            info = zf.infolist()[0]
            assert stat.S_ISLNK(info.external_attr >> 16)
            assert zf.read(info.filename) == b"../../etc/passwd"


class TestTarLinks:
    def test_symlink_traversing_target_fires(self, tmp_path):
        rules = _scan(tmp_path, "evil.nemo", _tar_with_link("x", "../../etc/passwd"))
        assert "MFV-ARCHIVE-002" in rules

    def test_symlink_absolute_target_fires(self, tmp_path):
        # Absolute with no ".." must still fire.
        rules = _scan(tmp_path, "abs.nemo", _tar_with_link("x", "/etc/passwd"))
        assert "MFV-ARCHIVE-002" in rules

    def test_hardlink_absolute_target_fires(self, tmp_path):
        rules = _scan(
            tmp_path, "hard.nemo",
            _tar_with_link("x", "/etc/shadow", link_type=tarfile.LNKTYPE),
        )
        assert "MFV-ARCHIVE-002" in rules

    def test_symlink_safe_relative_target_does_not_fire(self, tmp_path):
        rules = _scan(tmp_path, "ok.nemo", _tar_with_link("x", "weights/shard1"))
        assert "MFV-ARCHIVE-002" not in rules

    def test_regular_members_only_do_not_fire(self, tmp_path):
        rules = _scan(
            tmp_path, "plain.nemo",
            _tar_regular_only(("model_config.yaml", "model_weights.ckpt")),
        )
        assert "MFV-ARCHIVE-002" not in rules


class TestZipLinks:
    def test_symlink_traversing_target_fires(self, tmp_path):
        rules = _scan(tmp_path, "evil.pt", _zip_with_link("x", "../../etc/cron.d/x"))
        assert "MFV-ARCHIVE-002" in rules

    def test_symlink_absolute_target_fires(self, tmp_path):
        rules = _scan(tmp_path, "abs.pt", _zip_with_link("x", "/etc/passwd"))
        assert "MFV-ARCHIVE-002" in rules

    def test_symlink_safe_relative_target_does_not_fire(self, tmp_path):
        rules = _scan(tmp_path, "ok.pt", _zip_with_link("x", "weights/shard1"))
        assert "MFV-ARCHIVE-002" not in rules

    def test_regular_members_only_do_not_fire(self, tmp_path):
        rules = _scan(
            tmp_path, "plain.pt", _zip_regular_only(("archive/data.pkl", "archive/version")),
        )
        assert "MFV-ARCHIVE-002" not in rules

    def test_nested_mar_link_fires_one_container_deeper(self, tmp_path):
        rules = _scan(
            tmp_path, "evil.mar", _mar_wrapping_pt_with_link("x", "../../../etc/x"),
        )
        assert "MFV-ARCHIVE-002" in rules


class TestFindingShape:
    def test_severity_category_and_cwe(self, tmp_path):
        p = tmp_path / "evil.pt"
        p.write_bytes(_zip_with_link("x", "../../etc/passwd"))
        finding = next(
            f for f in ModelFileScanner().scan_file(p) if f.rule_id == "MFV-ARCHIVE-002"
        )
        assert finding.severity.name == "MEDIUM"
        assert finding.category.value == "path_traversal"
        assert 22 in (finding.cwe_ids or [])
        assert 59 in (finding.cwe_ids or [])
