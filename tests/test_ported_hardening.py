"""Hardening ported from Rowan's former copy of this scanner.

Rowan carried these fixes in its own vendored scanner before it switched to
depending on Hayward. Each test pins one resource or fail-closed guarantee.
"""

from __future__ import annotations

import os
import pickle
import struct
import sys
import time
import zipfile
from pathlib import Path

from hayward._tensors import _check_tflite_layout
from hayward.scanner import ModelFileScanner

SEVENZ_MAGIC = b"7z\xbc\xaf\x27\x1c"


def _install_7z_stub(tmp_path: Path, monkeypatch, listing: str, payload_size: int = 1,
                     listing_exit: int = 0) -> Path:
    """Fake `7zz` that logs every call. `l` prints `listing`; `x` emits a
    payload of `payload_size` bytes, to stdout with `-so`, else into `-o<dir>`."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "calls.log"
    code = (
        "import sys, pathlib\n"
        "args = sys.argv[1:]\n"
        f"pathlib.Path({str(log)!r}).open('a').write(' '.join(args[:2]) + '\\n')\n"
        "if args and args[0] == 'l':\n"
        f"    sys.stdout.write({listing!r})\n"
        f"    sys.exit({listing_exit})\n"
        "if args and args[0] == 'x':\n"
        f"    data = b'0' * {payload_size}\n"
        "    if '-so' in args:\n"
        "        sys.stdout.buffer.write(data)\n"
        "    else:\n"
        "        outdir = next(a[2:] for a in args if a.startswith('-o'))\n"
        "        (pathlib.Path(outdir) / 'x.pkl').write_bytes(data)\n"
        "    sys.exit(0)\n"
        "sys.exit(1)\n"
    )
    if os.name == "nt":
        (bindir / "stub7zz.py").write_text(code)
        (bindir / "7zz.bat").write_text(f'@"{sys.executable}" "%~dp0stub7zz.py" %*\n')
    else:
        stub = bindir / "7zz"
        stub.write_text(f"#!{sys.executable}\n" + code)
        stub.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    return log


def _calls(log: Path) -> list[str]:
    return log.read_text().splitlines() if log.exists() else []


class TestSevenZipFailsClosed:
    def test_oversized_archive_never_invokes_extractor(self, tmp_path, monkeypatch):
        log = _install_7z_stub(tmp_path, monkeypatch, "Path = a.7z\nPath = x.pkl\nSize = 1\n")
        archive = tmp_path / "large.7z"
        archive.write_bytes(SEVENZ_MAGIC + b"x" * 32)
        scanner = ModelFileScanner()
        scanner._SEVENZ_MAX_ARCHIVE_BYTES = 16
        findings = scanner.scan_file(archive)
        assert _calls(log) == []
        assert findings[0].metadata["skipped_reason"] == "oversized_archive"

    def test_garbled_size_does_not_bypass_the_quota(self, tmp_path, monkeypatch):
        # A non-numeric Size= line counts as zero, so the claimed total says
        # nothing; the quota on what extraction actually writes still holds.
        _install_7z_stub(tmp_path, monkeypatch, "Path = a.7z\nPath = x.pkl\nSize = nope\n",
                         payload_size=5_000_000)
        archive = tmp_path / "garbled.7z"
        archive.write_bytes(SEVENZ_MAGIC + b"x")
        scanner = ModelFileScanner()
        scanner.MAX_ZIP_MEMBER_BYTES = 1_000_000
        findings = scanner.scan_file(archive)
        assert findings[0].metadata["skipped_reason"] == "extraction_quota"

    def test_failed_listing_is_not_extracted(self, tmp_path, monkeypatch):
        log = _install_7z_stub(tmp_path, monkeypatch, "", listing_exit=2)
        archive = tmp_path / "broken.7z"
        archive.write_bytes(SEVENZ_MAGIC + b"x")
        findings = ModelFileScanner().scan_file(archive)
        assert not any(c.startswith("x") for c in _calls(log))
        assert findings[0].metadata["skipped_reason"] == "listing_failed"

    def test_extraction_beyond_the_claimed_size_is_capped(self, tmp_path, monkeypatch):
        # The listing claims 1 byte; the member actually expands to 5 MB.
        _install_7z_stub(tmp_path, monkeypatch, "Path = a.7z\nPath = x.pkl\nSize = 1\n",
                         payload_size=5_000_000)
        archive = tmp_path / "liar.7z"
        archive.write_bytes(SEVENZ_MAGIC + b"x")
        scanner = ModelFileScanner()
        scanner.MAX_ZIP_MEMBER_BYTES = 1_000_000
        findings = scanner.scan_file(archive)
        assert findings[0].metadata["skipped_reason"] == "extraction_quota"


def _make_npy(header_dict_str: str, payload: bytes) -> bytes:
    header = header_dict_str.encode("latin1")
    total = 10 + len(header) + 1
    pad = (16 - total % 16) % 16
    header = header + b" " * pad + b"\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + payload


def test_oversized_npy_header_is_rejected_before_parsing(tmp_path):
    """ast.literal_eval on an unbounded header is quadratic: an 8 MB header
    measured 406s and 3.4 GB RSS before the 64_000-byte cap."""
    header = "[" * 32_000 + "1" + "]" * 32_000
    p = tmp_path / "bomb.npy"
    p.write_bytes(_make_npy(header, b""))
    t0 = time.monotonic()
    findings = ModelFileScanner().scan_file(p)
    assert time.monotonic() - t0 < 1.0
    assert not [f for f in findings if f.severity.value != "info"]


def test_keras_zip_config_oversized_is_not_clean(tmp_path):
    """A config.json that decompresses past MAX_ZIP_MEMBER_BYTES is reported
    as unverified, not read in full (a 408 KB bomb expanded to 400 MB)."""
    p = tmp_path / "model.keras"
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("config.json", b"0" * 20_000_000)
    scanner = ModelFileScanner()
    scanner.MAX_ZIP_MEMBER_BYTES = 1_000_000
    findings = scanner.scan_file(p)
    assert [f.rule_id for f in findings] == ["MFV-SKIP-003"]
    assert findings[0].metadata.get("skipped_reason") == "oversized"


def test_one_bad_keras_does_not_abort_the_directory(tmp_path):
    with zipfile.ZipFile(tmp_path / "crash.keras", "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("config.json", "[" * 200000 + "]" * 200000)

    class Evil:
        def __reduce__(self):
            return (eval, ("1",))

    (tmp_path / "evil.pkl").write_bytes(pickle.dumps(Evil()))
    by_file: dict[str, list[str]] = {}
    for f in ModelFileScanner().scan_directory(tmp_path):
        by_file.setdefault(Path(f.file_path).name, []).append(f.rule_id)
    assert "MFV-SKIP-003" in by_file["crash.keras"]
    assert by_file["evil.pkl"]


def test_tflite_aliased_subgraphs_are_bounded():
    """4096 subgraph offsets aliasing one subgraph whose tensor vector claims
    every 4-byte slot: the replay stops at the file's slot budget."""
    size = 64 * 1024
    buf = bytearray(size)
    buf[4:8] = b"TFL3"
    m_vt, m = 8, 20
    buf[m_vt:m_vt + 10] = struct.pack("<HHHHH", 10, 8, 0, 0, 4)
    buf[m:m + 4] = struct.pack("<i", m - m_vt)
    sgs_vec = 32
    buf[m + 4:m + 8] = struct.pack("<I", sgs_vec - (m + 4))
    buf[sgs_vec:sgs_vec + 4] = struct.pack("<I", 4096)
    sg_vt = sgs_vec + 4 + 4 * 4096
    sg = sg_vt + 8
    buf[sg_vt:sg_vt + 8] = struct.pack("<HHHH", 8, 8, 0, 4)
    buf[sg:sg + 4] = struct.pack("<i", sg - sg_vt)
    for i in range(4096):
        at = sgs_vec + 4 + 4 * i
        buf[at:at + 4] = struct.pack("<I", sg - at)
    tensors_vec = sg + 8
    buf[sg + 4:sg + 8] = struct.pack("<I", tensors_vec - (sg + 4))
    buf[tensors_vec:tensors_vec + 4] = struct.pack("<I", size // 4)
    struct.pack_into("<I", buf, 0, m)

    start = time.monotonic()
    problems = _check_tflite_layout(bytes(buf))
    assert time.monotonic() - start < 2
    assert problems
