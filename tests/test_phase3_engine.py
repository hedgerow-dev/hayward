"""Phase-3 scanner engine fixes.

Covers four fixes to hayward.scanner:

- HW-113: the pickle resync (after a walk-killing splice) now also finds a
  protocol-0/1 payload, whose GLOBAL opener carries no PROTO marker.
- HW-114: the embedded-pickle walk inspects BYTEARRAY8 literals, recurses to a
  bounded depth, and surfaces inner calls convicted by ARGUMENT triage (not
  only inner streams naming an already-denied callable).
- HW-115: GGUF metadata that runs past the content-scan window now emits an
  MFV-GGUF-004 coverage finding instead of silently leaving later keys
  unchecked.
- HW-117f: the raw-deflate zip fallback surfaces MFV-PICKLE-003 for an
  oversized member instead of silently skipping it, matching the strict path.
"""

import io
import struct
import zipfile

from hayward import scanner as scanner_module
from hayward.scanner import (
    _ZIP_MEMBER_OVERSIZED,
    ModelFileScanner,
    _embedded_pickle_denied_globals,
    _next_pickle_offset,
    _parse_gguf_metadata,
    _resolve_pickle_globals,
)

# ── pickle opcode builders ──────────────────────────────────────────

def _su(text: str) -> bytes:
    """SHORT_BINUNICODE."""
    raw = text.encode()
    return bytes([0x8C, len(raw)]) + raw


def _short_binbytes(payload: bytes) -> bytes:
    """SHORT_BINBYTES (payload under 256 bytes)."""
    return bytes([0x43, len(payload)]) + payload


def _bytearray8(payload: bytes) -> bytes:
    """BYTEARRAY8 opcode carrying `payload`."""
    return b"\x96" + struct.pack("<Q", len(payload)) + payload


def _os_system_pickle(command: str) -> bytes:
    """A standalone protocol-4 pickle calling os.system(command)."""
    return (
        b"\x80\x04"
        + _su("os") + _su("system") + b"\x93"
        + _su(command) + b"\x85" + b"R."
    )


# ── HW-113: proto-0/1 resync ────────────────────────────────────────

class TestProto0ResyncAfterSplice:
    """After a splice kills the main opcode walk, the resync skips forward to
    the next plausible pickle. It searched only for the proto-2..5 PROTO
    marker, so a protocol-0/1 payload (which opens on a bare GLOBAL, `c...`)
    sitting after the splice was never reached."""

    # Raw bytes that are not valid opcodes, so the walk dies here the way a
    # joblib raw-array splice kills it, forcing the resync path.
    _WALK_KILLER = b"\xff\xff\xff\xff"

    # A protocol-0 os.system('id') pickle: GLOBAL, then REDUCE over one arg.
    _PROTO0_PAYLOAD = b"cos\nsystem\n(S'id'\ntR."

    def test_resync_reaches_a_proto0_payload_after_a_splice(self):
        data = b"\x80\x04N." + self._WALK_KILLER + self._PROTO0_PAYLOAD

        globals_found, _calls, _memo = _resolve_pickle_globals(data)

        assert "os.system" in globals_found, globals_found

    def test_next_offset_returns_the_proto0_opener(self):
        data = b"\x80\x04N." + self._WALK_KILLER + self._PROTO0_PAYLOAD
        # The proto-0 opener starts right after the 4-byte splice.
        assert _next_pickle_offset(data, 4) == data.index(b"cos\n")

    def test_scan_flags_the_spliced_proto0_payload(self, tmp_path):
        p = tmp_path / "spliced.pkl"
        p.write_bytes(b"\x80\x04N." + self._WALK_KILLER + self._PROTO0_PAYLOAD)

        findings = ModelFileScanner().scan_file(p)

        assert any(f.rule_id == "MFV-PICKLE-001" for f in findings), (
            [(f.rule_id, f.message) for f in findings]
        )

    def test_spurious_proto0_match_in_raw_bytes_is_not_a_conviction(self, tmp_path):
        """A 'c<word>\\n<word>\\n' shape inside raw array bytes is not a real
        pickle. The resync may resume there, but the walk resolves only a
        harmless unknown global, never a denied one, so nothing is convicted."""
        # `carray_data\nfloat32\n` matches the proto-0 GLOBAL shape but names an
        # unknown (undenied) callable; the trailing bytes are array noise.
        raw = b"\x80\x04N." + b"\xff\xff" + b"carray_data\nfloat32\n" + bytes(range(8))
        p = tmp_path / "benign.pkl"
        p.write_bytes(raw)

        findings = ModelFileScanner().scan_file(p)

        assert not any(f.rule_id == "MFV-PICKLE-001" for f in findings), (
            [(f.rule_id, f.message) for f in findings]
        )


# ── HW-114: embedded-pickle walk ────────────────────────────────────

class TestEmbeddedPickleWalk:
    """The bytes-literal pickle walk used to inspect only BINBYTES-family
    literals, one level deep, and report only inner streams naming an
    already-denied callable."""

    def test_bytearray8_literal_is_inspected(self):
        # BYTEARRAY8, not a BINBYTES opcode, carries the inner os.system pickle.
        outer = b"\x80\x04" + _bytearray8(_os_system_pickle("id")) + b"."

        assert _embedded_pickle_denied_globals(outer) == ["os.system"]

    def test_two_level_nesting_is_reached(self):
        # outer -> bytes-literal pickle -> bytes-literal pickle, with the denied
        # global two levels down.
        inner_two = _os_system_pickle("id")
        inner_one = b"\x80\x04" + _short_binbytes(inner_two) + b"."
        outer = b"\x80\x04" + _short_binbytes(inner_one) + b"."

        assert _embedded_pickle_denied_globals(outer) == ["os.system"]

    def test_argument_convicted_unknown_call_surfaces(self):
        # An inner callable that is unknown by name but convicted by its URL
        # argument. The descriptor, not a denied name, is the evidence.
        inner = (
            b"\x80\x04"
            + _su("mymodule") + _su("myfunc") + b"\x93"
            + _su("http://evil.example/x") + b"\x85" + b"R."
        )
        outer = b"\x80\x04" + _short_binbytes(inner) + b"."

        evidence = _embedded_pickle_denied_globals(outer)

        assert len(evidence) == 1, evidence
        assert "mymodule.myfunc" in evidence[0]
        assert "URL" in evidence[0]

    def test_argument_convicted_call_reaches_the_finding(self, tmp_path):
        inner = (
            b"\x80\x04"
            + _su("mymodule") + _su("myfunc") + b"\x93"
            + _su("http://evil.example/x") + b"\x85" + b"R."
        )
        outer = b"\x80\x04" + _short_binbytes(inner) + b"."
        p = tmp_path / "nested.pkl"
        p.write_bytes(outer)

        findings = ModelFileScanner().scan_file(p)

        pickle008 = [f for f in findings if f.rule_id == "MFV-PICKLE-008"]
        assert pickle008, [(f.rule_id, f.message) for f in findings]
        assert any(
            "mymodule.myfunc" in g
            for g in pickle008[0].metadata["nested_globals"]
        )

    def test_benign_nested_pickle_is_not_flagged(self):
        # An inner stream naming only an allowed callable, invoking nothing.
        inner = b"\x80\x04" + _su("collections") + _su("OrderedDict") + b"\x93" + b"."
        outer = b"\x80\x04" + _short_binbytes(inner) + b"."

        assert _embedded_pickle_denied_globals(outer) == []


# ── HW-115: GGUF metadata scan window ───────────────────────────────

class TestGgufMetadataWindow:
    """The GGUF content pass stops at GGUF_METADATA_SCAN_BYTES. A dangerous key
    beyond that offset was never content-checked and no coverage finding was
    emitted; now MFV-GGUF-004 records the gap."""

    @staticmethod
    def _kv_string(key: str, value: str) -> bytes:
        kb = key.encode()
        vb = value.encode()
        return (
            struct.pack("<Q", len(kb)) + kb
            + struct.pack("<I", 8)                # value type 8 = STRING
            + struct.pack("<Q", len(vb)) + vb
        )

    def _gguf(self, entries: list[bytes]) -> bytes:
        # version, tensor_count, kv_count
        header = b"GGUF" + struct.pack("<IQQ", 3, 0, len(entries))
        return header + b"".join(entries)

    def test_key_past_the_window_emits_a_coverage_finding(self, tmp_path, monkeypatch):
        monkeypatch.setattr(scanner_module, "GGUF_METADATA_SCAN_BYTES", 64)
        # First entry alone pushes the offset past 64, so the dangerous second
        # key sits beyond the content-scan window.
        blob = self._gguf([
            self._kv_string("padding", "x" * 60),
            self._kv_string("tokenizer.chat_template", "{{ ''.__class__ }}"),
        ])
        p = tmp_path / "windowed.gguf"
        p.write_bytes(blob)

        findings = ModelFileScanner().scan_file(p)

        coverage = [f for f in findings if f.rule_id == "MFV-GGUF-004"]
        assert coverage, [(f.rule_id, f.message) for f in findings]
        assert coverage[0].metadata["skipped_reason"] == "gguf_metadata_window"

    def test_small_gguf_within_the_window_emits_no_coverage_finding(self, tmp_path):
        blob = self._gguf([self._kv_string("general.name", "tiny-model")])
        p = tmp_path / "small.gguf"
        p.write_bytes(blob)

        findings = ModelFileScanner().scan_file(p)

        assert not any(f.rule_id == "MFV-GGUF-004" for f in findings), (
            [(f.rule_id, f.message) for f in findings]
        )

    def test_parse_reports_the_window_truncation_flag(self):
        blob = self._gguf([
            self._kv_string("padding", "x" * 60),
            self._kv_string("general.name", "past-the-window"),
        ])
        _result, window_truncated = _parse_gguf_metadata(blob, 64)
        assert window_truncated is True

        _result2, window_truncated2 = _parse_gguf_metadata(blob, 10_000_000)
        assert window_truncated2 is False


# ── HW-117f: raw-deflate oversized member parity ────────────────────

class TestRawDeflateOversizedMember:
    """The strict zip read reports an oversized member as MFV-PICKLE-003. The
    raw-deflate fallback used to return None for both oversized and unreadable
    members, so the caller skipped a zip bomb in silence. The raw reader now
    distinguishes the two."""

    def _deflated_zip(self, payload: bytes) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("data.pkl", payload)
        return buf.getvalue()

    def test_oversized_raw_member_emits_the_zip_bomb_finding(self, tmp_path, monkeypatch):
        # os.system sits in the first bytes; the member decompresses well past
        # the lowered cap. If the truncated prefix were scanned, MFV-PICKLE-001
        # would fire; parity means MFV-PICKLE-003 fires and nothing is scanned.
        payload = _os_system_pickle("id") + b"N" * 800
        p = tmp_path / "bomb.pt"
        p.write_bytes(self._deflated_zip(payload))

        scanner = ModelFileScanner()
        monkeypatch.setattr(scanner, "MAX_ZIP_MEMBER_BYTES", 100)

        # Force the raw fallback: make the strict reader raise a caught error.
        def _raise_strict(*_args, **_kwargs):
            raise RuntimeError("forced strict-reader refusal")

        monkeypatch.setattr(scanner, "_read_zip_member_capped", _raise_strict)

        findings = scanner._scan_pytorch_zip(p)

        assert any(f.rule_id == "MFV-PICKLE-003" for f in findings), (
            [(f.rule_id, f.message) for f in findings]
        )
        # No truncated payload bytes were scanned, so os.system never resolved.
        assert not any(f.rule_id == "MFV-PICKLE-001" for f in findings), (
            [(f.rule_id, f.message) for f in findings]
        )

    def test_raw_reader_distinguishes_oversized_from_unreadable(self, tmp_path, monkeypatch):
        payload = _os_system_pickle("id") + b"N" * 800
        p = tmp_path / "bomb.pt"
        p.write_bytes(self._deflated_zip(payload))

        scanner = ModelFileScanner()
        monkeypatch.setattr(scanner, "MAX_ZIP_MEMBER_BYTES", 100)
        info = zipfile.ZipFile(p).infolist()[0]

        # Opt-in: oversized returns the sentinel, not None.
        assert scanner._read_zip_member_raw(
            p, info, report_oversized=True,
        ) is _ZIP_MEMBER_OVERSIZED
        # Default: oversized still collapses to None (unchanged for old callers).
        assert scanner._read_zip_member_raw(p, info) is None
