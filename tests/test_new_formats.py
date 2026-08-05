"""Regression tests for GitHub issue #77: MFV coverage for ONNX,
TensorFlow SavedModel, and joblib/numpy allow_pickle payloads.

ModelFileScanner previously covered pickle/SafeTensors/GGUF/Keras only.
These formats carry the same code-execution surface `modelscan` (Protect
AI) already flags and this project didn't:

- ONNX custom ops (onnxruntime-extensions' PyOp/PythonOp embed a Python
  callable that runs at inference time) and external_data path traversal.
- TensorFlow SavedModel GraphDef ops that read/write files or invoke an
  embedded Python callable (PyFunc).
- numpy .npy/.npz object-dtype arrays, which numpy.load(allow_pickle=True)
  deserializes via a real pickle stream -- the same RCE surface as .pkl.
- joblib, which wraps a plain or zlib-compressed pickle stream.

All four route through the existing hardened callable-resolution pickle
scanner (_scan_pickle) rather than duplicating that logic.
"""

from __future__ import annotations

import io
import json
import os
import pickle
import struct
import tarfile
import zipfile
import zlib

from hayward.findings import Severity
from hayward.scanner import ModelFileScanner


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _pb_field(num: int, wire_type: int, payload: bytes) -> bytes:
    return _varint((num << 3) | wire_type) + payload


def _pb_length_delimited(num: int, data: bytes) -> bytes:
    return _pb_field(num, 2, _varint(len(data)) + data)


def _pb_string_field(num: int, s: str) -> bytes:
    return _pb_length_delimited(num, s.encode("utf-8"))


class _Evil:
    def __reduce__(self):
        return (os.system, ("echo pwned",))


class _Benign:
    pass


def _make_npy(header_dict_str: str, payload: bytes) -> bytes:
    header = header_dict_str.encode("latin1")
    total = 10 + len(header) + 1
    pad = (16 - total % 16) % 16
    header = header + b" " * pad + b"\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + payload


def _build_gguf(kvs: list[tuple[bytes, bytes]]) -> bytes:
    """Build a minimal, structurally-valid GGUF file with only string-typed
    metadata KV entries (value_type=8), matching the real GGUF spec's
    header + KV section so the scanner's structure-aware parser
    (_parse_gguf_metadata) can actually walk it."""

    def gguf_string(s: bytes) -> bytes:
        return struct.pack("<Q", len(s)) + s

    header = b"GGUF" + struct.pack("<IQQ", 3, 0, len(kvs))
    body = b"".join(gguf_string(k) + struct.pack("<I", 8) + gguf_string(v) for k, v in kvs)
    return header + body


class TestTfliteLayout:
    """MFV-TFLITE-001: tensor dimension products a 32-bit loader cannot
    hold (CVE-2026-42627's shape)."""

    @staticmethod
    def _build_tflite(tensors: list[list[int]]) -> bytes:
        """A minimal valid TFLite FlatBuffer: one subgraph, N tensors with
        the given shapes. FlatBuffers uoffsets are unsigned and point
        forward, so everything is laid out in reference order and patched."""
        buf = bytearray(b"\x00\x00\x00\x00TFL3")

        def align() -> None:
            while len(buf) % 4:
                buf.append(0)

        # Model table (vtable for fields 0..2; only subgraphs=2 present)
        align()
        m_vt = len(buf)
        buf += struct.pack("<HHHHH", 10, 8, 0, 0, 4)
        align()
        m = len(buf)
        buf += struct.pack("<i", m - m_vt)
        m_sgs_at = len(buf)
        buf += b"\x00\x00\x00\x00"  # patched: subgraphs vector offset

        # Subgraphs vector (one entry, patched)
        align()
        sgs_vec = len(buf)
        buf += struct.pack("<I", 1)
        sg_entry_at = len(buf)
        buf += b"\x00\x00\x00\x00"
        buf[m_sgs_at:m_sgs_at + 4] = struct.pack("<I", sgs_vec - m_sgs_at)

        # SubGraph table (vtable for fields 0..1; only tensors=1 present)
        align()
        sg_vt = len(buf)
        buf += struct.pack("<HHHH", 8, 8, 0, 4)
        align()
        sg = len(buf)
        buf += struct.pack("<i", sg - sg_vt)
        sg_tensors_at = len(buf)
        buf += b"\x00\x00\x00\x00"  # patched: tensors vector offset
        buf[sg_entry_at:sg_entry_at + 4] = struct.pack("<I", sg - sg_entry_at)

        # Tensors vector (entries patched after each tensor table is written)
        align()
        tensors_vec = len(buf)
        buf += struct.pack("<I", len(tensors))
        entry_positions = []
        for _ in tensors:
            entry_positions.append(len(buf))
            buf += b"\x00\x00\x00\x00"
        buf[sg_tensors_at:sg_tensors_at + 4] = struct.pack("<I", tensors_vec - sg_tensors_at)

        # Tensor tables, each with its own vtable and forward shape vector
        for dims, entry_at in zip(tensors, entry_positions, strict=True):
            align()
            t_vt = len(buf)
            buf += struct.pack("<HHHHH", 10, 12, 0, 4, 8)  # fields 0..2 at 0,4,8
            align()
            t = len(buf)
            buf += struct.pack("<i", t - t_vt)
            t_shape_at = len(buf)
            buf += b"\x00\x00\x00\x00"  # patched: shape vector offset
            buf += struct.pack("<B", 1) + b"\x00\x00\x00"   # type UINT8 + pad
            buf[entry_at:entry_at + 4] = struct.pack("<I", t - entry_at)

            align()
            sv = len(buf)
            buf += struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}i", *dims)
            buf[t_shape_at:t_shape_at + 4] = struct.pack("<I", sv - t_shape_at)

        buf[0:4] = struct.pack("<I", m)
        return bytes(buf)

    def _scan(self, tmp_path, blob: bytes):
        p = tmp_path / "model.tflite"
        p.write_bytes(blob)
        return ModelFileScanner().scan_file(p)

    def test_benign_model_clean(self, tmp_path):
        blob = self._build_tflite([[1, 224, 224, 3], [1001, 512]])
        findings = self._scan(tmp_path, blob)
        assert findings == [], [(f.rule_id, f.message) for f in findings]

    def test_dimension_product_wrapping_32bit_flagged(self, tmp_path):
        blob = self._build_tflite([[65536, 65536]])
        findings = self._scan(tmp_path, blob)
        assert any(f.rule_id == "MFV-TFLITE-001" for f in findings), (
            [(f.rule_id, f.severity) for f in findings]
        )

    def test_negative_dimension_flagged(self, tmp_path):
        blob = self._build_tflite([[1, -1, 3]])
        findings = self._scan(tmp_path, blob)
        assert any(f.rule_id == "MFV-TFLITE-001" for f in findings)

    def test_non_tflite_magic_not_scanned(self, tmp_path):
        findings = self._scan(tmp_path, b"\x00" * 64)
        assert findings == []


class TestSevenZScanning:
    def test_7z_without_extractor_is_not_silent(self, tmp_path, monkeypatch):
        """No 7zz on PATH: the archive must be reported as unverifiable,
        never skipped clean (malicious1.7z in picklescan's suite read as
        zero findings before this handler existed)."""
        monkeypatch.setattr("hayward.scanner.shutil.which", lambda name: None)
        p = tmp_path / "model.7z"
        p.write_bytes(b"7z\xBC\xAF\x27\x1C" + b"\x00" * 64)

        findings = ModelFileScanner().scan_file(p)
        assert [f.rule_id for f in findings] == ["MFV-7Z-001"]
        assert "NOT a clean verdict" in findings[0].message

    def test_non_7z_content_not_scanned(self, tmp_path):
        p = tmp_path / "model.7z"
        p.write_bytes(b"PK\x03\x04" + b"\x00" * 32)
        findings = ModelFileScanner().scan_file(p)
        assert findings == []

    def test_extractor_path_scans_members(self, tmp_path, monkeypatch):
        """With a 7zz present, members are extracted and scanned. The
        extractor is faked with a python script that writes a malicious
        pickle member."""
        script = tmp_path / "fake7zz"
        payload = tmp_path / "payload.pkl"
        payload.write_bytes(b"\x80\x04cos\nsystem\nS'id'\n\x85R.")
        script.write_text(
            "#!/usr/bin/env python3\n"
            "import shutil, sys, pathlib\n"
            "out = next(a[2:] for a in sys.argv[1:] if a.startswith('-o'))\n"
            f"shutil.copy({str(payload)!r}, str(pathlib.Path(out) / 'payload.pkl'))\n"
        )
        script.chmod(0o755)
        monkeypatch.setattr(
            "hayward.scanner.shutil.which",
            lambda name: str(script) if name == "7zz" else None,
        )
        p = tmp_path / "model.7z"
        p.write_bytes(b"7z\xBC\xAF\x27\x1C" + b"\x00" * 64)

        findings = ModelFileScanner().scan_file(p)
        assert any(f.rule_id == "MFV-PICKLE-001" for f in findings), (
            [(f.rule_id, f.message[:80]) for f in findings]
        )


class TestPmmlScanning:
    """PMML files: XML with parse-time XXE protection and code-exec
    function-references in <Apply> transformations."""

    def test_doctype_triggers_xxe(self, tmp_path):
        p = tmp_path / "model.pmml"
        p.write_bytes(
            b'<?xml version="1.0"?>'
            b'<!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>'
            b'<PMML xmlns="http://www.dmg.org/PMML-4_4"><DataDictionary/></PMML>'
        )
        findings = ModelFileScanner().scan_file(p)
        assert any(f.rule_id == "MFV-PMML-001" and f.severity == Severity.HIGH
                   for f in findings), [(f.rule_id, f.severity) for f in findings]

    def test_dangerous_function_flagged(self, tmp_path):
        p = tmp_path / "model.pmml"
        p.write_bytes(
            b'<?xml version="1.0"?>'
            b'<PMML xmlns="http://www.dmg.org/PMML-4_4">'
            b'<TransformationDictionary>'
            b'<DerivedField name="x"><Apply function="java.lang.Runtime.getRuntime().exec(\'id\')"/></DerivedField>'
            b'</TransformationDictionary></PMML>'
        )
        findings = ModelFileScanner().scan_file(p)
        assert any(f.rule_id == "MFV-PMML-002" for f in findings), (
            [(f.rule_id, f.severity) for f in findings]
        )

    def test_benign_pmml_no_findings(self, tmp_path):
        p = tmp_path / "model.pmml"
        p.write_bytes(
            b'<?xml version="1.0"?>'
            b'<PMML xmlns="http://www.dmg.org/PMML-4_4">'
            b'<DataDictionary><DataField name="x" optype="continuous" dataType="double"/></DataDictionary>'
            b'</PMML>'
        )
        findings = ModelFileScanner().scan_file(p)
        assert findings == [], [(f.rule_id, f.message) for f in findings]

    def test_non_pmml_xml_not_scanned(self, tmp_path):
        p = tmp_path / "model.pmml"
        p.write_bytes(b'<?xml version="1.0"?><root>hi</root>')
        findings = ModelFileScanner().scan_file(p)
        assert findings == []

    def test_garbage_bytes_report_non_coverage_without_crashing(self, tmp_path):
        """Originally asserted `findings == []`. That was the silence-as-clean
        pattern: a .pmml this parser cannot read was never checked, and a more
        permissive PMML engine may still consume it (arXiv 2508.19774). The
        no-crash intent is unchanged; the clean verdict is not."""
        p = tmp_path / "model.pmml"
        p.write_bytes(b"\xff\xfe not xml")
        findings = ModelFileScanner().scan_file(p)
        assert [f.rule_id for f in findings] == ["MFV-SKIP-003"]


class TestTarTruncation:
    """A tar cut inside a member header ends iteration with no exception:
    the walk yields the members before the cut and silently never sees the
    rest. The walk-is-complete check flags the non-zero unread tail."""

    @staticmethod
    def _tar_three_members() -> bytes:
        import tarfile as _tf
        buf = io.BytesIO()
        with _tf.open(fileobj=buf, mode="w") as tf:
            for i in range(3):
                data = b"\x80\x04cos\nsystem\nS'id'\n\x85R." if i == 2 else b"x" * (1000 * (i + 1))
                info = _tf.TarInfo(f"m{i}.pkl")
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
        return buf.getvalue()

    def test_header_cut_flags_unread_tail_and_keeps_findings(self, tmp_path):
        blob = self._tar_three_members()
        # 4300 bytes lands inside the third member's header.
        p = tmp_path / "model.nemo"
        p.write_bytes(blob[:4300])

        findings = ModelFileScanner().scan_file(p)
        assert any(f.rule_id == "MFV-SKIP-002" for f in findings), (
            [(f.rule_id, f.message[:80]) for f in findings]
        )

    def test_complete_tar_no_skip_finding(self, tmp_path):
        blob = self._tar_three_members()
        p = tmp_path / "model.nemo"
        p.write_bytes(blob)

        findings = ModelFileScanner().scan_file(p)
        assert not any(f.rule_id == "MFV-SKIP-002" for f in findings), (
            [(f.rule_id, f.message[:80]) for f in findings]
        )
        # And the third member's payload is found on the complete walk.
        assert any(f.rule_id == "MFV-PICKLE-001" for f in findings)


class TestEmbeddedExecutables:
    """MFV-EXEC-001: a loadable binary inside a model file. Every check is
    two-stage structural, because bare magics occur by chance in weight
    blobs."""

    @staticmethod
    def _pe() -> bytes:
        stub = bytearray(b"MZ" + b"\x00" * 62)
        struct.pack_into("<I", stub, 0x3C, 0x80)
        stub += b"\x00" * (0x80 - len(stub)) + b"PE\0\0" + b"\x00" * 24
        return bytes(stub)

    @staticmethod
    def _elf() -> bytes:
        return (b"\x7fELF" + bytes([2, 1, 1]) + b"\x00" * 9
                + struct.pack("<HH", 2, 0x3E) + b"\x00" * 40)

    def test_pe_in_pickle_flagged(self, tmp_path):
        p = tmp_path / "model.pkl"
        p.write_bytes(pickle.dumps({"w": [1.0]}) + self._pe())
        findings = ModelFileScanner().scan_file(p)
        assert any(f.rule_id == "MFV-EXEC-001" for f in findings)

    def test_elf_in_pickle_flagged(self, tmp_path):
        p = tmp_path / "model.pkl"
        p.write_bytes(pickle.dumps({"w": [1.0]}) + b"\x00" * 512 + self._elf())
        findings = ModelFileScanner().scan_file(p)
        assert any(f.rule_id == "MFV-EXEC-001" for f in findings)

    def test_bare_magic_alone_not_flagged(self, tmp_path):
        """MZ and ELF magics without the second stage are weight bytes."""
        p = tmp_path / "weights.pkl"
        p.write_bytes(pickle.dumps({"w": [1.0]}) + b"MZ" + b"\x00" * 100
                      + b"\x7fELF" + b"\x00" * 100)
        findings = ModelFileScanner().scan_file(p)
        assert not any(f.rule_id == "MFV-EXEC-001" for f in findings)

    def test_exe_member_in_torch_zip_flagged(self, tmp_path):
        import io as _io
        buf = _io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("archive/data.pkl", pickle.dumps({"w": [1.0]}))
            zf.writestr("archive/payload", self._elf())
        p = tmp_path / "model.pt"
        p.write_bytes(buf.getvalue())
        findings = ModelFileScanner().scan_file(p)
        assert any(f.rule_id == "MFV-EXEC-001" for f in findings), (
            [(f.rule_id, f.severity) for f in findings]
        )


class TestGgufLayoutArithmetic:
    """MFV-GGUF-005: the container's counts, lengths, dimensions and offsets
    must fit the file without wrapping u64 -- the CVE-2025-53630 /
    CVE-2026-27940 / CVE-2026-33298 shape."""

    @staticmethod
    def _gguf(*, kv_count: int = 0, kvs: bytes = b"",
              tensors: list[tuple[bytes, list[int], int, int]] | None = None,
              version: int = 3, data_bytes: int = 0,
              kv_count_override: int | None = None,
              tensor_count_override: int | None = None) -> bytes:
        tensors = tensors or []
        out = b"GGUF" + struct.pack(
            "<IQQ", version,
            tensor_count_override if tensor_count_override is not None else len(tensors),
            kv_count_override if kv_count_override is not None else kv_count,
        )
        out += kvs
        for name, dims, type_id, offset in tensors:
            out += struct.pack("<Q", len(name)) + name
            out += struct.pack("<I", len(dims))
            out += struct.pack(f"<{len(dims)}Q", *dims)
            out += struct.pack("<I", type_id)
            out += struct.pack("<Q", offset)
        # Data section: aligned to 32 after the infos.
        out += b"\x00" * ((32 - len(out) % 32) % 32) + b"\x00" * data_bytes
        return out

    def _scan(self, tmp_path, blob: bytes):
        p = tmp_path / "model.gguf"
        p.write_bytes(blob)
        return ModelFileScanner().scan_file(p)

    def test_benign_gguf_with_tensor_no_layout_finding(self, tmp_path):
        blob = self._gguf(
            tensors=[(b"weight", [256, 256], 0, 0)],  # F32: 256*256*4 bytes
            data_bytes=256 * 256 * 4,
        )
        findings = self._scan(tmp_path, blob)
        assert not any(f.rule_id == "MFV-GGUF-005" for f in findings), (
            [(f.rule_id, f.message) for f in findings]
        )

    def test_kv_count_overflow(self, tmp_path):
        findings = self._scan(tmp_path, self._gguf(kv_count_override=10**9))
        assert any(f.rule_id == "MFV-GGUF-005" for f in findings)

    def test_tensor_count_overflow(self, tmp_path):
        findings = self._scan(tmp_path, self._gguf(tensor_count_override=10**12))
        assert any(f.rule_id == "MFV-GGUF-005" for f in findings)

    def test_dimensions_wrap_u64(self, tmp_path):
        """CVE-2026-33298's published shape: ne = [1024, 1024, 4398046511105, 1]."""
        findings = self._scan(tmp_path, self._gguf(
            tensors=[(b"t", [1024, 1024, 4398046511105, 1], 0, 0)],
        ))
        assert any(f.rule_id == "MFV-GGUF-005" for f in findings)

    def test_tensor_overruns_file(self, tmp_path):
        findings = self._scan(tmp_path, self._gguf(
            tensors=[(b"t", [1000], 0, 0)],  # needs 4000 bytes, file has ~0
        ))
        assert any(f.rule_id == "MFV-GGUF-005" for f in findings)

    def test_unknown_version(self, tmp_path):
        findings = self._scan(tmp_path, self._gguf(version=99))
        assert any(f.rule_id == "MFV-GGUF-005" for f in findings)

    def test_huge_kv_key_length(self, tmp_path):
        kvs = struct.pack("<Q", 1 << 40)  # key_len far past the file
        findings = self._scan(tmp_path, self._gguf(kv_count=1, kvs=kvs))
        assert any(f.rule_id == "MFV-GGUF-005" for f in findings)

    def test_too_many_dims(self, tmp_path):
        findings = self._scan(tmp_path, self._gguf(
            tensors=[(b"t", [1, 1, 1, 1, 1], 0, 0)],
        ))
        assert any(f.rule_id == "MFV-GGUF-005" for f in findings)


class TestSafetensorsLayoutArithmetic:
    """MFV-ST-006: header offsets must fit the data section, match shape x
    dtype, and not overlap -- the spec's own contract."""

    @staticmethod
    def _safetensors(tensors: dict, data_bytes: int) -> bytes:
        header = json.dumps(tensors).encode()
        return struct.pack("<Q", len(header)) + header + b"\x00" * data_bytes

    def _scan(self, tmp_path, blob: bytes):
        p = tmp_path / "model.safetensors"
        p.write_bytes(blob)
        return ModelFileScanner().scan_file(p)

    def test_benign_layout_clean(self, tmp_path):
        blob = self._safetensors(
            {"w": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]}},
            16,
        )
        findings = self._scan(tmp_path, blob)
        assert not any(f.rule_id == "MFV-ST-006" for f in findings), (
            [(f.rule_id, f.message) for f in findings]
        )

    def test_span_disagrees_with_shape(self, tmp_path):
        blob = self._safetensors(
            {"w": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 12]}},
            12,
        )
        findings = self._scan(tmp_path, blob)
        assert any(f.rule_id == "MFV-ST-006" for f in findings)

    def test_tensor_larger_than_4gb(self, tmp_path):
        blob = self._safetensors(
            {"w": {"dtype": "F32", "shape": [2**31], "data_offsets": [0, 2**33]}},
            16,
        )
        findings = self._scan(tmp_path, blob)
        assert any(f.rule_id == "MFV-ST-006" for f in findings)

    def test_end_past_data_section(self, tmp_path):
        blob = self._safetensors(
            {"w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 100]}},
            16,
        )
        findings = self._scan(tmp_path, blob)
        assert any(f.rule_id == "MFV-ST-006" for f in findings)

    def test_overlapping_tensors(self, tmp_path):
        blob = self._safetensors(
            {
                "a": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
                "b": {"dtype": "F32", "shape": [4], "data_offsets": [8, 24]},
            },
            24,
        )
        findings = self._scan(tmp_path, blob)
        assert any(f.rule_id == "MFV-ST-006" for f in findings)


class TestSkopsScanning:
    """skops (.skops) is a zip of schema.json + .npy members. The schema's
    __module__/__class__ pairs classify with the pickle engine's
    allow/deny/unknown split, and two loader invariants (MethodNode
    consistency, OperatorFuncNode's module) are checked as structure."""

    @staticmethod
    def _build_skops(schema: dict, extra_members: dict[str, bytes] | None = None) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("schema.json", json.dumps(schema))
            for name, blob in (extra_members or {}).items():
                zf.writestr(name, blob)
        return buf.getvalue()

    @staticmethod
    def _json_node(value: str) -> dict:
        return {"__class__": "str", "__module__": "builtins",
                "__loader__": "JsonNode", "content": value, "is_json": True}

    def _scan(self, tmp_path, blob: bytes):
        path = tmp_path / "model.skops"
        path.write_bytes(blob)
        return ModelFileScanner().scan_file(path)

    def test_benign_sklearn_model_nothing_above_info(self, tmp_path):
        """The julien-c/skops-digits shape: an sklearn estimator plus
        builtins containers. Since 2026-08-04 the classifier knows
        class-shaped sklearn refs as ordinary ML machinery, so this produces
        no findings at all."""
        schema = {
            "__class__": "LinearRegression",
            "__module__": "sklearn.linear_model._base",
            "__loader__": "ObjectNode",
            "content": {
                "__class__": "dict", "__module__": "builtins",
                "__loader__": "DictNode",
                "content": {
                    "fit_intercept": self._json_node("true"),
                    "n_features_in_": self._json_node("1"),
                },
            },
        }
        findings = self._scan(tmp_path, self._build_skops(schema))
        assert findings == [], [(f.rule_id, f.severity, f.message) for f in findings]

    def test_builtins_only_schema_is_silent(self, tmp_path):
        """builtins containers are the serialization machinery itself, not
        payload: a schema of nothing but them produces no findings."""
        schema = {
            "__class__": "dict", "__module__": "builtins",
            "__loader__": "DictNode",
            "content": {"a": self._json_node("1"),
                        "b": self._json_node("\"x\"")},
        }
        findings = self._scan(tmp_path, self._build_skops(schema))
        assert not findings, [(f.rule_id, f.message) for f in findings]

    def test_denied_type_reference_is_critical(self, tmp_path):
        schema = {
            "__class__": "system", "__module__": "posix",
            "__loader__": "ObjectNode", "content": {},
        }
        findings = self._scan(tmp_path, self._build_skops(schema))
        hits = [f for f in findings if f.rule_id == "MFV-SKOPS-001"]
        assert hits and all(f.severity == Severity.CRITICAL for f in hits), (
            [(f.rule_id, f.severity) for f in findings]
        )

    def test_function_node_reaching_os_system_is_critical(self, tmp_path):
        schema = {
            "__class__": "system", "__module__": "os",
            "__loader__": "FunctionNode",
        }
        findings = self._scan(tmp_path, self._build_skops(schema))
        assert any(f.rule_id == "MFV-SKOPS-001" for f in findings)

    def test_method_node_inconsistency_is_critical(self, tmp_path):
        """CVE-2025-54413's primitive: the outer pair says one type, the
        bound object is another. skops < 0.12.0 trusted the outer and
        called the inner."""
        schema = {
            "__class__": "LinearRegression",
            "__module__": "sklearn.linear_model._base",
            "__loader__": "MethodNode",
            "content": {
                "obj": {"__class__": "system", "__module__": "os",
                        "__loader__": "ObjectNode", "content": {}},
                "func": "fit",
            },
        }
        findings = self._scan(tmp_path, self._build_skops(schema))
        assert any(f.rule_id == "MFV-SKOPS-002"
                   and f.severity == Severity.CRITICAL for f in findings), (
            [(f.rule_id, f.severity) for f in findings]
        )

    def test_method_node_consistent_not_flagged(self, tmp_path):
        schema = {
            "__class__": "Pipeline", "__module__": "sklearn.pipeline",
            "__loader__": "MethodNode",
            "content": {
                "obj": {"__class__": "Pipeline", "__module__": "sklearn.pipeline",
                        "__loader__": "ObjectNode", "content": {}},
                "func": "fit",
            },
        }
        findings = self._scan(tmp_path, self._build_skops(schema))
        assert not any(f.rule_id == "MFV-SKOPS-002" for f in findings)

    def test_operator_func_node_outside_operator_module_is_critical(self, tmp_path):
        schema = {
            "__class__": "system", "__module__": "os",
            "__loader__": "OperatorFuncNode", "attrs": [],
        }
        findings = self._scan(tmp_path, self._build_skops(schema))
        assert any(f.rule_id == "MFV-SKOPS-002" for f in findings)

    def test_operator_func_node_legitimate_not_flagged(self, tmp_path):
        schema = {
            "__class__": "itemgetter", "__module__": "operator",
            "__loader__": "OperatorFuncNode", "attrs": [],
        }
        findings = self._scan(tmp_path, self._build_skops(schema))
        assert not any(f.rule_id == "MFV-SKOPS-002" for f in findings)

    def test_custom_class_is_info_only(self, tmp_path):
        """The scikit-learn/persistence example shape: a __main__ transformer.
        Loading it needs trusted=..., which is exactly the INFO bucket."""
        schema = {
            "__class__": "DivideColumns", "__module__": "__main__",
            "__loader__": "ObjectNode", "content": {},
        }
        findings = self._scan(tmp_path, self._build_skops(schema))
        assert any(f.rule_id == "MFV-SKOPS-003" for f in findings)
        assert not [f for f in findings if f.severity != Severity.INFO]

    def test_embedded_pickle_gets_the_pickle_engine_and_a_note(self, tmp_path):
        evil = b"\x80\x04cos\nsystem\nS'id'\n\x85R."
        findings = self._scan(
            tmp_path,
            self._build_skops(
                {"__class__": "dict", "__module__": "builtins",
                 "__loader__": "DictNode", "content": {}},
                {"payload.pkl": evil},
            ),
        )
        assert any(f.rule_id == "MFV-SKOPS-004" for f in findings)
        assert any(f.rule_id == "MFV-PICKLE-001" for f in findings), (
            [(f.rule_id, f.severity) for f in findings]
        )

    def test_non_zip_skops_is_not_a_clean_verdict(self, tmp_path):
        """A .skops that does not open as a zip was previously a silent clean
        verdict. skops is not wired into the extension-confusion check, so
        nothing else owned the case."""
        findings = self._scan(tmp_path, b"not a zip at all")
        assert [f.rule_id for f in findings] == ["MFV-SKIP-003"]

    def test_missing_schema_is_info_not_crash(self, tmp_path):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("weights.npy", b"\x93NUMPY\x01\x00")
        findings = self._scan(tmp_path, buf.getvalue())
        assert [f.rule_id for f in findings] == ["MFV-SKOPS-005"]
        assert findings[0].severity == Severity.INFO


class TestOnnxScanning:
    def test_pyop_custom_op_flagged(self, tmp_path):
        node = _pb_string_field(4, "PyOp")
        graph = _pb_length_delimited(1, node)
        model = _pb_length_delimited(7, graph)
        p = tmp_path / "evil.onnx"
        p.write_bytes(model)

        findings = ModelFileScanner().scan_file(p)
        assert any(f.rule_id == "MFV-ONNX-001" for f in findings)
        assert all(f.severity.value in ("high", "critical") for f in findings if f.rule_id == "MFV-ONNX-001")

    def test_benign_op_not_flagged(self, tmp_path):
        node = _pb_string_field(4, "Conv")
        graph = _pb_length_delimited(1, node)
        model = _pb_length_delimited(7, graph)
        p = tmp_path / "benign.onnx"
        p.write_bytes(model)

        findings = ModelFileScanner().scan_file(p)
        assert findings == []

    def test_external_data_path_traversal_flagged(self, tmp_path):
        node = _pb_string_field(3, "../../etc/passwd")
        graph = _pb_length_delimited(1, node)
        model = _pb_length_delimited(7, graph)
        p = tmp_path / "traversal.onnx"
        p.write_bytes(model)

        findings = ModelFileScanner().scan_file(p)
        assert any(f.rule_id == "MFV-ONNX-002" for f in findings)

    def test_malformed_protobuf_does_not_crash(self, tmp_path):
        p = tmp_path / "garbage.onnx"
        p.write_bytes(b"\xff\xff\xff\xff\xff\xff\xff\xff\xff")
        findings = ModelFileScanner().scan_file(p)
        assert findings == []

    @staticmethod
    def _external_data(*entries: tuple[str, str]) -> bytes:
        """A TensorProto-shaped message: repeated StringStringEntryProto
        (field 13 = external_data), each entry {1: key, 2: value}."""
        tensor = b""
        for key, value in entries:
            entry = _pb_string_field(1, key) + _pb_string_field(2, value)
            tensor += _pb_length_delimited(13, entry)
        return tensor

    def test_absolute_external_data_path_flagged(self, tmp_path):
        """MFV-ONNX-002 used to require a literal "..": "/etc/shadow" and
        "C:\\..." escape the model directory just as surely. Checked only in
        external_data location values, because ONNX node names begin with "/"
        by convention and are not paths."""
        for location in ("/etc/shadow", "C:\\Windows\\System32\\drivers\\etc\\hosts"):
            model = self._external_data(("location", location))
            p = tmp_path / "absolute.onnx"
            p.write_bytes(model)

            findings = ModelFileScanner().scan_file(p)
            assert any(f.rule_id == "MFV-ONNX-002" for f in findings), location

    def test_scoped_node_names_are_not_paths(self, tmp_path):
        """The shape that false-positived on all 16 real ONNX files in the
        quickset benign corpus: node names like "/bert/Cast" are hierarchical
        scope names, not filesystem paths."""
        graph = b"".join(_pb_string_field(4, name) for name in (
            "/bert/Cast", "/embeddings/Add", "/Cast_output_0",
        ))
        model = _pb_length_delimited(7, graph)
        p = tmp_path / "node_names.onnx"
        p.write_bytes(model)

        findings = ModelFileScanner().scan_file(p)
        assert findings == [], [(f.rule_id, f.message) for f in findings]

    def test_unexpected_external_data_key_flagged(self, tmp_path):
        """CVE-2026-34445: keys outside the four-key contract reached
        setattr() on the ExternalDataInfo object before onnx 1.21.0."""
        model = self._external_data(
            ("location", "weights.bin"),
            ("offset", "0"),
            ("__class__", "evil"),
        )
        p = tmp_path / "setattr.onnx"
        p.write_bytes(model)

        findings = ModelFileScanner().scan_file(p)
        hits = [f for f in findings if f.rule_id == "MFV-ONNX-003"]
        assert hits and all(f.severity == Severity.HIGH for f in hits), (
            [(f.rule_id, f.severity) for f in findings]
        )

    def test_legitimate_external_data_keys_not_flagged(self, tmp_path):
        model = self._external_data(
            ("location", "weights.bin"),
            ("offset", "4096"),
            ("length", "1048576"),
            ("checksum", "sha1:abc"),
        )
        p = tmp_path / "clean_external.onnx"
        p.write_bytes(model)

        findings = ModelFileScanner().scan_file(p)
        assert findings == [], [(f.rule_id, f.message) for f in findings]

    def test_metadata_props_map_is_not_anchored(self, tmp_path):
        """ModelProto.metadata_props is the same StringStringEntry shape with
        free-form keys. Without a "location" key present, the external-data
        check must not fire on it."""
        tensor = b""
        for key, value in (("author", "someone"), ("license", "MIT")):
            entry = _pb_string_field(1, key) + _pb_string_field(2, value)
            tensor += _pb_length_delimited(14, entry)
        p = tmp_path / "metadata.onnx"
        p.write_bytes(tensor)

        findings = ModelFileScanner().scan_file(p)
        assert not any(f.rule_id == "MFV-ONNX-003" for f in findings)

    def test_protobuf_junk_with_dotdot_is_not_a_path(self, tmp_path):
        """The hub sweep's FP shape: quantized weight bytes decoding as
        printable junk that happens to contain ".." and "/". A traversal
        string must be path-shaped (leading ../, or a /../ segment), not
        merely contain two dots somewhere."""
        junk = "54..9.00-30,-----,,---+,-..,---------+./1..*"
        node = _pb_string_field(3, junk)
        graph = _pb_length_delimited(1, node)
        model = _pb_length_delimited(7, graph)
        p = tmp_path / "junk.onnx"
        p.write_bytes(model)

        findings = ModelFileScanner().scan_file(p)
        assert not any(f.rule_id == "MFV-ONNX-002" for f in findings), (
            [(f.rule_id, f.message) for f in findings]
        )


class TestTfSavedModelScanning:
    def test_pyfunc_op_flagged(self, tmp_path):
        node = _pb_string_field(2, "PyFunc")
        graph = _pb_length_delimited(2, node)
        saved_model = _pb_length_delimited(2, graph)
        p = tmp_path / "saved_model.pb"
        p.write_bytes(saved_model)

        findings = ModelFileScanner().scan_file(p)
        assert any(f.rule_id == "MFV-TF-001" for f in findings)

    def test_benign_op_not_flagged(self, tmp_path):
        node = _pb_string_field(2, "MatMul")
        graph = _pb_length_delimited(2, node)
        saved_model = _pb_length_delimited(2, graph)
        p = tmp_path / "saved_model.pb"
        p.write_bytes(saved_model)

        findings = ModelFileScanner().scan_file(p)
        assert findings == []

    def test_saver_checkpoint_boilerplate_not_flagged(self, tmp_path):
        """MergeV2Checkpoints and ShardedFilename are tf.train.Saver
        save/restore-side ops present in the MetaGraphDef of ordinary
        SavedModels as checkpoint boilerplate. They must not fire."""
        for op in ("MergeV2Checkpoints", "ShardedFilename"):
            node = _pb_string_field(2, op)
            graph = _pb_length_delimited(2, node)
            saved_model = _pb_length_delimited(2, graph)
            p = tmp_path / "saved_model.pb"
            p.write_bytes(saved_model)

            findings = ModelFileScanner().scan_file(p)
            assert findings == []

    def test_readfile_op_flagged(self, tmp_path):
        node = _pb_string_field(2, "ReadFile")
        graph = _pb_length_delimited(2, node)
        saved_model = _pb_length_delimited(2, graph)
        p = tmp_path / "saved_model.pb"
        p.write_bytes(saved_model)

        findings = ModelFileScanner().scan_file(p)
        assert any(f.rule_id == "MFV-TF-001" for f in findings)

    def test_arbitrary_pb_file_not_scanned_as_savedmodel(self, tmp_path):
        """Only the canonical saved_model.pb filename is treated as a
        TF graph -- an unrelated .pb file with the same dangerous-looking
        string must not be scanned at all."""
        node = _pb_string_field(2, "PyFunc")
        graph = _pb_length_delimited(2, node)
        data = _pb_length_delimited(2, graph)
        p = tmp_path / "some_other_file.pb"
        p.write_bytes(data)

        findings = ModelFileScanner().scan_file(p)
        assert findings == []


class TestNumpyAllowPickleScanning:
    def test_object_dtype_npy_with_rce_payload_is_critical(self, tmp_path):
        payload = pickle.dumps(_Evil())
        npy = _make_npy("{'descr': '|O', 'fortran_order': False, 'shape': (1,), }", payload)
        p = tmp_path / "evil.npy"
        p.write_bytes(npy)

        findings = ModelFileScanner().scan_file(p)
        assert any(f.rule_id == "MFV-PICKLE-001" and f.severity.value == "critical" for f in findings)

    def test_numeric_dtype_npy_is_clean(self, tmp_path):
        npy = _make_npy(
            "{'descr': '<i8', 'fortran_order': False, 'shape': (3,), }",
            b"\x01\x00\x00\x00\x00\x00\x00\x00" * 3,
        )
        p = tmp_path / "benign.npy"
        p.write_bytes(npy)

        findings = ModelFileScanner().scan_file(p)
        assert findings == []

    def test_npz_archive_member_attribution(self, tmp_path):
        payload = pickle.dumps(_Evil())
        evil_npy = _make_npy("{'descr': '|O', 'fortran_order': False, 'shape': (1,), }", payload)
        benign_npy = _make_npy(
            "{'descr': '<i8', 'fortran_order': False, 'shape': (1,), }", b"\x00" * 8
        )
        p = tmp_path / "model.npz"
        with zipfile.ZipFile(p, "w") as zf:
            zf.writestr("arr_0.npy", evil_npy)
            zf.writestr("arr_1.npy", benign_npy)

        findings = ModelFileScanner().scan_file(p)
        assert any("arr_0.npy" in f.message for f in findings)

    def test_npz_traversal_member_flagged(self, tmp_path):
        p = tmp_path / "evil.npz"
        with zipfile.ZipFile(p, "w") as zf:
            # ZipInfo direct: writestr() sanitizes ".." out of plain arcnames,
            # which would test nothing.
            zf.writestr(zipfile.ZipInfo("../../etc/cron.d/payload.npy"), b"junk")
            zf.writestr("arr_0.npy", b"junk")

        findings = ModelFileScanner().scan_file(p)
        assert any(f.rule_id == "MFV-NPZ-001" for f in findings)

    def test_npz_absolute_member_flagged(self, tmp_path):
        p = tmp_path / "evil.npz"
        with zipfile.ZipFile(p, "w") as zf:
            zf.writestr(zipfile.ZipInfo("/etc/passwd.npy"), b"junk")

        findings = ModelFileScanner().scan_file(p)
        assert any(f.rule_id == "MFV-NPZ-001" for f in findings)

    def test_npz_duplicate_normalized_names_flagged(self, tmp_path):
        p = tmp_path / "evil.npz"
        with zipfile.ZipFile(p, "w") as zf:
            zf.writestr("dir\\arr_0.npy", b"one")
            zf.writestr("dir/arr_0.npy", b"two")

        findings = ModelFileScanner().scan_file(p)
        assert any(f.rule_id == "MFV-NPZ-001" for f in findings)

    def test_npz_symlink_member_flagged(self, tmp_path):
        p = tmp_path / "evil.npz"
        with zipfile.ZipFile(p, "w") as zf:
            info = zipfile.ZipInfo("link.npy")
            info.external_attr = 0o120777 << 16  # S_IFLNK
            zf.writestr(info, "/etc/passwd")

        findings = ModelFileScanner().scan_file(p)
        assert any(f.rule_id == "MFV-NPZ-001" for f in findings)

    def test_npz_flat_archive_stays_clean(self, tmp_path):
        """numpy.savez's own layout: flat .npy members, modest ratio."""
        npy = _make_npy(
            "{'descr': '<i8', 'fortran_order': False, 'shape': (3,), }",
            b"\x01\x00\x00\x00\x00\x00\x00\x00" * 3,
        )
        p = tmp_path / "clean.npz"
        with zipfile.ZipFile(p, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("arr_0.npy", npy)

        findings = ModelFileScanner().scan_file(p)
        assert findings == [], [(f.rule_id, f.message) for f in findings]


class TestJoblibScanning:
    def test_uncompressed_joblib_rce_payload_flagged(self, tmp_path):
        p = tmp_path / "model.joblib"
        p.write_bytes(pickle.dumps(_Evil()))

        findings = ModelFileScanner().scan_file(p)
        assert any(f.rule_id == "MFV-PICKLE-001" for f in findings)

    def test_zlib_compressed_joblib_rce_payload_flagged(self, tmp_path):
        p = tmp_path / "model.joblib"
        p.write_bytes(zlib.compress(pickle.dumps(_Evil())))

        findings = ModelFileScanner().scan_file(p)
        assert any(f.rule_id == "MFV-PICKLE-001" for f in findings)

    def test_benign_joblib_is_clean_or_info_only(self, tmp_path):
        p = tmp_path / "model.joblib"
        p.write_bytes(pickle.dumps(_Benign()))

        findings = ModelFileScanner().scan_file(p)
        assert all(f.severity.value not in ("high", "critical") for f in findings)

    def test_joblib_normal_compressed_payload_still_scanned(self, tmp_path):
        """A joblib file whose decompressed size is well within the cap must
        still be scanned normally (no regression from the size-cap fix)."""
        p = tmp_path / "model.joblib"
        p.write_bytes(zlib.compress(pickle.dumps(_Evil())))

        findings = ModelFileScanner().scan_file(p)
        assert any(f.rule_id == "MFV-PICKLE-001" for f in findings)
        assert not any(f.rule_id == "MFV-JOBLIB-002" for f in findings)


class TestGgufContentChecks:
    """FPs found by the quickset benign corpus (2026-08-03): real unsloth
    DeepSeek-V4 GGUF shards tripped MFV-GGUF-002 on vocabulary tokens and
    MFV-GGUF-003 on the word "scenarios." in template prose."""

    @staticmethod
    def _gguf_with_vocab(tokens: list[bytes], template: bytes = b"") -> bytes:
        """Minimal GGUF with a string-array KV (tokenizer.ggml.tokens, type 9)
        and optionally a chat_template, matching the on-disk layout the
        scanner's parser walks."""
        def gguf_string(s: bytes) -> bytes:
            return struct.pack("<Q", len(s)) + s

        def kv_string(key: bytes, value: bytes) -> bytes:
            return gguf_string(key) + struct.pack("<I", 8) + gguf_string(value)

        def kv_string_array(key: bytes, values: list[bytes]) -> bytes:
            return (
                gguf_string(key)
                + struct.pack("<I", 9)       # ARRAY
                + struct.pack("<I", 8)       # element type STRING
                + struct.pack("<Q", len(values))
                + b"".join(gguf_string(v) for v in values)
            )

        kvs = kv_string_array(b"tokenizer.ggml.tokens", tokens)
        count = 1
        if template:
            kvs += kv_string(b"tokenizer.chat_template", template)
            count += 1
        return b"GGUF" + struct.pack("<IQQ", 3, 0, count) + kvs

    def test_code_like_vocab_tokens_not_flagged(self, tmp_path):
        """A vocabulary is training-data substrings, not metadata anyone
        wrote: a code-trained model necessarily has tokens like "<class" and
        "exec(". None of that is executable in a GGUF runtime."""
        path = tmp_path / "code-vocab.gguf"
        path.write_bytes(self._gguf_with_vocab([
            b"hello", b"<class", b"exec(", b"__import__", b"world",
        ]))

        findings = ModelFileScanner().scan_file(path)
        assert not any(f.rule_id == "MFV-GGUF-002" for f in findings), (
            f"vocab tokens tripped the metadata content check: "
            f"{[(f.rule_id, f.message) for f in findings]}"
        )

    def test_template_prose_containing_signature_substring_not_flagged(self, tmp_path):
        """"os." appears in ordinary English ("scenarios."). Jinja renders
        prose outside {{ }}/{% %} verbatim, so a signature there is not
        executable and must not fire."""
        template = (
            b"{% for message in messages %}{{ message['content'] }}{% endfor %}"
            b"Test against all potential paths, edge cases, and adversarial "
            b"scenarios. Explicitly write out your deliberation process."
        )
        path = tmp_path / "prose-template.gguf"
        path.write_bytes(self._gguf_with_vocab([b"hello"], template))

        findings = ModelFileScanner().scan_file(path)
        assert not any(f.rule_id == "MFV-GGUF-003" for f in findings), (
            f"prose tripped the SSTI check: "
            f"{[(f.rule_id, f.message) for f in findings]}"
        )

    def test_signature_substring_inside_a_jinja_string_literal_not_flagged(self, tmp_path):
        """The DeepSeek-V4 shape: system-prompt prose embedded as a string
        literal inside a Jinja block. A literal is data; "os." inside it is
        not executable."""
        template = (
            b"{% if enable_thinking %}{{ 'Test against all potential paths, "
            b"edge cases, and adversarial scenarios.\\nExplicitly write out "
            b"your entire deliberation process.' }}{% endif %}"
        )
        path = tmp_path / "literal-template.gguf"
        path.write_bytes(self._gguf_with_vocab([b"hello"], template))

        findings = ModelFileScanner().scan_file(path)
        assert not any(f.rule_id == "MFV-GGUF-003" for f in findings), (
            f"a string literal tripped the SSTI check: "
            f"{[(f.rule_id, f.message) for f in findings]}"
        )


class TestGgufChatTemplateSsti:
    """DEF-32: MFV-GGUF-003 used to flag any tokenizer.chat_template
    containing the substring "{{" -- but "{{ }}" is ordinary Jinja2 variable
    substitution syntax present in every real chat-tuned GGUF model (e.g.
    Llama-3's real template renders message['role'] this way), so the old
    check false-positived on almost every legitimate model. The rule now
    searches the template body for actual code-execution/sandbox-escape
    constructs instead of mere "{{" presence."""

    def test_realistic_llama3_style_template_not_flagged(self, tmp_path):
        """A real Llama-3-style Jinja2 chat template -- ordinary role/content
        substitution, loops, and conditionals, no code-exec constructs --
        must produce zero GGUF-chat-template findings."""
        llama3_style_template = (
            "{% for message in messages %}"
            "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' "
            "+ message['content'] | trim + '<|eot_id|>' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
            "{% endif %}"
        )
        path = tmp_path / "llama3-style.gguf"
        path.write_bytes(_build_gguf([
            (b"general.name", b"llama-3-style-chat"),
            (b"tokenizer.chat_template", llama3_style_template.encode()),
        ]))

        findings = ModelFileScanner().scan_file(path)
        assert not any(f.rule_id == "MFV-GGUF-003" for f in findings), (
            f"Expected zero GGUF-chat-template findings for an ordinary Jinja2 "
            f"substitution template, got: {[(f.rule_id, f.message) for f in findings]}"
        )

    def test_template_with_globals_introspection_still_flagged(self, tmp_path):
        """A chat_template reaching for __globals__-style introspection to
        escape the Jinja2 sandbox must still fire CRITICAL."""
        path = tmp_path / "evil.gguf"
        path.write_bytes(_build_gguf([
            (b"general.name", b"evil-model"),
            (b"tokenizer.chat_template", b"{{ 7*7 }} {{ lipsum.__globals__ }}"),
        ]))

        findings = ModelFileScanner().scan_file(path)
        critical = [f for f in findings if f.rule_id == "MFV-GGUF-003"]
        assert critical, (
            f"Expected a CRITICAL MFV-GGUF-003 finding, got: {[(f.rule_id, f.message) for f in findings]}"
        )
        assert all(f.severity.value == "critical" for f in critical)

    def test_template_with_os_system_still_flagged(self, tmp_path):
        """A chat_template that shells out via os.system must still fire."""
        path = tmp_path / "evil2.gguf"
        path.write_bytes(_build_gguf([
            (b"tokenizer.chat_template", b"{{ ().__class__.__base__.__subclasses__() }} {{ os.system('id') }}"),
        ]))

        findings = ModelFileScanner().scan_file(path)
        assert any(f.rule_id == "MFV-GGUF-003" for f in findings)


class TestProtobufWalker:
    """Unit tests for the schema-less protobuf field walker shared by the
    ONNX and TF SavedModel scanners."""

    def test_extracts_nested_string(self):
        from hayward.scanner import _extract_protobuf_strings

        inner = _pb_string_field(1, "hello world")
        outer = _pb_length_delimited(2, inner)
        assert "hello world" in _extract_protobuf_strings(outer)

    def test_truncated_input_does_not_raise(self):
        from hayward.scanner import _extract_protobuf_strings

        # A length-delimited field claiming more bytes than actually present.
        truncated = _varint((1 << 3) | 2) + _varint(1000) + b"short"
        assert _extract_protobuf_strings(truncated) == []

    def test_empty_input_returns_empty(self):
        from hayward.scanner import _extract_protobuf_strings

        assert _extract_protobuf_strings(b"") == []

    def test_deprecated_group_wire_type_bails_cleanly(self):
        from hayward.scanner import _extract_protobuf_strings

        group_start = _varint((1 << 3) | 3)  # wire_type 3 = deprecated start_group
        assert _extract_protobuf_strings(group_start) == []


# -- Model-format coverage: keras_metadata.pb, .th, .mar, .nemo --
#
# Each case below is a format a real loader will deserialize but that
# scan_file previously skipped outright on the extension/filename gate.

_LAMBDA_CONFIG = (
    '{"class_name": "Sequential", "config": {"name": "sequential", "layers": '
    '[{"class_name": "Lambda", "config": {"name": "lambda", "function": '
    '["BASE64CODEOBJECT", null, null], "function_type": "lambda", '
    '"module": "__main__"}}]}}'
)

_PLAIN_CONFIG = (
    '{"class_name": "Sequential", "config": {"name": "sequential", "layers": '
    '[{"class_name": "Dense", "config": {"name": "dense", "units": 8}}]}}'
)


class TestKerasMetadataPbScanning:
    """A SavedModel export writes the Keras layer graph to
    keras_metadata.pb, not to saved_model.pb. On all four MalHug repos
    that ship a Lambda-layer payload the base64 code object is present in
    keras_metadata.pb and absent from the sibling saved_model.pb, so
    gating .pb on the saved_model.pb filename alone missed the payload
    entirely."""

    def test_lambda_layer_in_keras_metadata_is_flagged(self, tmp_path):
        p = tmp_path / "keras_metadata.pb"
        p.write_bytes(_pb_string_field(2, _LAMBDA_CONFIG))

        findings = ModelFileScanner().scan_file(p)
        assert any(f.rule_id == "MFV-KERAS-001" for f in findings)

    def test_lambda_free_keras_metadata_is_not_flagged(self, tmp_path):
        p = tmp_path / "keras_metadata.pb"
        p.write_bytes(_pb_string_field(2, _PLAIN_CONFIG))

        findings = ModelFileScanner().scan_file(p)
        assert not any(f.rule_id == "MFV-KERAS-001" for f in findings)

    def test_other_pb_filenames_still_skipped(self, tmp_path):
        """Widening the .pb gate must not turn every stray protobuf into a
        Keras scan -- only the two canonical export filenames qualify."""
        p = tmp_path / "fingerprint.pb"
        p.write_bytes(_pb_string_field(2, _LAMBDA_CONFIG))

        assert ModelFileScanner().scan_file(p) == []

    def test_one_layer_serialized_twice_counts_once(self, tmp_path):
        """keras_metadata.pb stores the graph under both `config` and
        `model_config`; the same Lambda must not be reported twice."""
        doubled = (
            '{"class_name": "Sequential", "config": {"layers": [{"class_name": '
            '"Lambda", "config": {"name": "lambda"}}]}, "model_config": '
            '{"class_name": "Sequential", "config": {"layers": [{"class_name": '
            '"Lambda", "config": {"name": "lambda"}}]}}}'
        )
        p = tmp_path / "keras_metadata.pb"
        p.write_bytes(_pb_string_field(2, doubled))

        findings = ModelFileScanner().scan_file(p)
        lambda_findings = [f for f in findings if f.rule_id == "MFV-KERAS-001"]
        assert len(lambda_findings) == 1
        assert lambda_findings[0].metadata["lambda_layers"] == ["Lambda layer 'lambda'"]


class TestTorchThExtension:
    """`.th` is a torch checkpoint suffix in real use; torch.load reads it
    like any other checkpoint. MalHug's 191fdp/test ships a 7KB
    ViT-L-14_stats.th whose inner pickle calls __builtin__.exec."""

    def test_zip_wrapped_th_checkpoint_is_scanned(self, tmp_path):
        p = tmp_path / "ViT-L-14_stats.th"
        with zipfile.ZipFile(p, "w") as zf:
            zf.writestr("archive/data.pkl", pickle.dumps(_Evil()))

        findings = ModelFileScanner().scan_file(p)
        assert any(f.rule_id == "MFV-PICKLE-001" for f in findings)

    def test_flat_pickle_th_checkpoint_is_scanned(self, tmp_path):
        p = tmp_path / "legacy.th"
        p.write_bytes(pickle.dumps(_Evil()))

        findings = ModelFileScanner().scan_file(p)
        assert any(f.rule_id == "MFV-PICKLE-001" for f in findings)


class TestTorchServeMarArchive:
    """torch-model-archiver packs the serialized model plus handler source
    into a zip. With --serialized-file model.pt the payload sits two
    containers deep, so a single-level walk reports the archive clean."""

    def test_flat_pickle_member_is_scanned(self, tmp_path):
        p = tmp_path / "model.mar"
        with zipfile.ZipFile(p, "w") as zf:
            zf.writestr("MAR-INF/MANIFEST.json", "{}")
            zf.writestr("model.pkl", pickle.dumps(_Evil()))

        findings = ModelFileScanner().scan_file(p)
        assert any(f.rule_id == "MFV-PICKLE-001" for f in findings)

    def test_nested_checkpoint_is_scanned_one_level_down(self, tmp_path):
        inner = io.BytesIO()
        with zipfile.ZipFile(inner, "w") as iz:
            iz.writestr("archive/data.pkl", pickle.dumps(_Evil()))
        p = tmp_path / "model.mar"
        with zipfile.ZipFile(p, "w") as zf:
            zf.writestr("MAR-INF/MANIFEST.json", "{}")
            zf.writestr("model.pt", inner.getvalue())

        findings = ModelFileScanner().scan_file(p)
        assert any(f.rule_id == "MFV-PICKLE-001" for f in findings)
        assert any("model.pt/archive/data.pkl" in f.message for f in findings)

    def test_benign_archive_is_quiet(self, tmp_path):
        p = tmp_path / "clean.mar"
        with zipfile.ZipFile(p, "w") as zf:
            zf.writestr("MAR-INF/MANIFEST.json", "{}")
            zf.writestr("handler.py", "def handle(x):\n    return x\n")

        assert ModelFileScanner().scan_file(p) == []


class TestNemoTarContainer:
    """`.nemo` is a tar holding model_weights.ckpt. NeMo pins
    weights_only=False on load, so PyTorch 2.6's safe-load default does
    not cover it."""

    @staticmethod
    def _write_tar(path, members):
        with tarfile.open(path, "w") as tf:
            for name, blob in members:
                info = tarfile.TarInfo(name)
                info.size = len(blob)
                tf.addfile(info, io.BytesIO(blob))

    def test_flat_pickle_member_is_scanned(self, tmp_path):
        p = tmp_path / "model.nemo"
        self._write_tar(p, [
            ("model_config.yaml", b"name: x\n"),
            ("model_weights.ckpt", pickle.dumps(_Evil())),
        ])

        findings = ModelFileScanner().scan_file(p)
        assert any(f.rule_id == "MFV-PICKLE-001" for f in findings)
        assert any("model_weights.ckpt" in f.message for f in findings)

    def test_zip_format_checkpoint_inside_tar_is_scanned(self, tmp_path):
        inner = io.BytesIO()
        with zipfile.ZipFile(inner, "w") as iz:
            iz.writestr("archive/data.pkl", pickle.dumps(_Evil()))
        p = tmp_path / "model.nemo"
        self._write_tar(p, [("model_weights.ckpt", inner.getvalue())])

        findings = ModelFileScanner().scan_file(p)
        assert any(f.rule_id == "MFV-PICKLE-001" for f in findings)

    def test_benign_checkpoint_is_quiet(self, tmp_path):
        p = tmp_path / "clean.nemo"
        self._write_tar(p, [
            ("model_config.yaml", b"name: clean\n"),
            ("weights.bin", b"\x00" * 4096),
        ])

        assert ModelFileScanner().scan_file(p) == []

    def test_truncated_tar_keeps_findings_from_members_already_scanned(self, tmp_path):
        """Exception-oriented evasion (arXiv 2508.19774): a member crafted to
        end the walk early must not erase detections from earlier members,
        which would make "payload plus malformed member" quieter than the
        payload on its own."""
        p = tmp_path / "model.nemo"
        self._write_tar(p, [
            ("model_weights.ckpt", pickle.dumps(_Evil())),
            ("padding.bin", b"\x00" * 8192),
        ])
        raw = p.read_bytes()
        # Cut inside the second member's *data* so tarfile raises once the
        # first member has already been scanned. Cutting inside its header
        # instead ends the walk with no exception at all, which is a separate
        # gap recorded in HANDOFF-model-scanning.md.
        p.write_bytes(raw[:raw.index(b"padding.bin") + 600])

        findings = ModelFileScanner().scan_file(p)
        assert any(f.rule_id == "MFV-PICKLE-001" for f in findings)
        assert any(f.rule_id == "MFV-SKIP-002" for f in findings)

    def test_non_tar_nemo_is_not_a_clean_verdict(self, tmp_path):
        """A .nemo that does not open as a tar was previously a silent clean
        verdict; the NEMO early return in scan_file bypasses the
        extension-confusion check, so nothing else owned the case."""
        p = tmp_path / "broken.nemo"
        p.write_bytes(b"not a tar archive at all")

        findings = ModelFileScanner().scan_file(p)
        assert [f.rule_id for f in findings] == ["MFV-SKIP-003"]


class TestExceptionOrientedEvasion:
    """Exception-oriented evasion (arXiv 2508.19774): a file crafted to make
    a parser raise must never come out the other side as a silent clean
    verdict, and a failure partway through a container walk must not discard
    findings already collected. Every handler fixed in this sweep reports
    MFV-SKIP-002 (container walk ended early) or MFV-SKIP-003 (content could
    not be verified) instead of swallowing."""

    def test_unparseable_pickle_stays_silent(self, tmp_path):
        """_scan_pickle: opcode walk fails and no raw danger signature is
        present. This was briefly MFV-SKIP-003 and was reverted on measured
        evidence: "0 globals + no raw hit" is the profile of every harmless
        stream (pickle.dumps(42), and the raw tensor blobs inside every
        torch checkpoint, which the two-byte opener sniff matches by chance).
        Pickle execution requires a GLOBAL the loader can reach, and the
        loader reaches it only through opcodes this same walk reads, so this
        stream cannot carry a working payload. Silence is loader parity."""
        p = tmp_path / "model.pkl"
        p.write_bytes(b"\xff\xfe not a pickle at all")

        findings = ModelFileScanner().scan_file(p)
        assert findings == []

    def test_pure_data_pickle_stays_silent(self, tmp_path):
        """pickle.dumps(42): a valid pickle with no GLOBAL opcodes. The other
        half of the same profile."""
        p = tmp_path / "data.pkl"
        p.write_bytes(pickle.dumps(42))

        findings = ModelFileScanner().scan_file(p)
        assert findings == []

    def test_scan_file_stat_failure_is_not_clean(self, tmp_path, monkeypatch):
        p = tmp_path / "model.pkl"
        p.write_bytes(pickle.dumps(_Benign()))

        def boom(self):
            raise OSError("crafted")
        monkeypatch.setattr("pathlib.Path.stat", boom)

        findings = ModelFileScanner().scan_file(p)
        assert [f.rule_id for f in findings] == ["MFV-SKIP-003"]

    def test_scan_file_read_failure_is_not_clean(self, tmp_path, monkeypatch):
        p = tmp_path / "model.pkl"
        p.write_bytes(pickle.dumps(_Benign()))

        def boom(self):
            raise OSError("crafted")
        monkeypatch.setattr("pathlib.Path.read_bytes", boom)

        findings = ModelFileScanner().scan_file(p)
        assert [f.rule_id for f in findings] == ["MFV-SKIP-003"]

    def test_keras_header_read_failure_is_not_clean(self, tmp_path, monkeypatch):
        p = tmp_path / "model.h5"
        p.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 64)

        def boom(*args, **kwargs):
            raise OSError("crafted")
        monkeypatch.setattr("builtins.open", boom)

        findings = ModelFileScanner()._scan_keras(p)
        assert [f.rule_id for f in findings] == ["MFV-SKIP-003"]

    def test_keras_body_read_failure_is_not_clean(self, tmp_path, monkeypatch):
        p = tmp_path / "model.h5"
        p.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 64)

        def boom(self):
            raise OSError("crafted")
        monkeypatch.setattr("pathlib.Path.read_bytes", boom)

        findings = ModelFileScanner()._scan_keras(p)
        assert [f.rule_id for f in findings] == ["MFV-SKIP-003"]

    def test_keras_zip_config_read_failure_is_not_clean(self, tmp_path, monkeypatch):
        p = tmp_path / "model.keras"
        with zipfile.ZipFile(p, "w") as zf:
            zf.writestr("config.json", b"{}")

        def boom(self, *args, **kwargs):
            raise OSError("crafted")
        monkeypatch.setattr("zipfile.ZipFile.read", boom)

        findings = ModelFileScanner().scan_file(p)
        assert [f.rule_id for f in findings] == ["MFV-SKIP-003"]

    def test_keras_zip_config_unparseable_is_not_clean(self, tmp_path):
        """A .keras whose config.json is not JSON previously returned [] with
        the layer graph never checked."""
        p = tmp_path / "model.keras"
        with zipfile.ZipFile(p, "w") as zf:
            zf.writestr("config.json", b"{not json at all")

        findings = ModelFileScanner().scan_file(p)
        assert [f.rule_id for f in findings] == ["MFV-SKIP-003"]

    def test_pytorch_zip_walk_failure_keeps_member_findings(self, tmp_path, monkeypatch):
        """_scan_pytorch_zip: a failure partway through the member walk
        previously fell back to a flat-pickle scan of the zip bytes,
        discarding every finding from members already scanned."""
        p = tmp_path / "model.pt"
        with zipfile.ZipFile(p, "w") as zf:
            zf.writestr("archive/data.pkl", pickle.dumps(_Evil()))
            zf.writestr("archive/extra.pkl", pickle.dumps(_Benign()))

        real = ModelFileScanner._scan_pickle
        calls = 0

        def flaky(self, file_path, data):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise zipfile.BadZipFile("crafted")
            return real(self, file_path, data)
        monkeypatch.setattr(ModelFileScanner, "_scan_pickle", flaky)

        findings = ModelFileScanner().scan_file(p)
        assert any(f.rule_id == "MFV-PICKLE-001" for f in findings)
        assert any(f.rule_id == "MFV-SKIP-002" for f in findings)

    def test_inner_zip_walk_failure_keeps_member_findings(self, tmp_path, monkeypatch):
        """_scan_zip_bytes (nested/tar route): same discard bug, reached via a
        zip-format checkpoint inside a .nemo tar."""
        inner = io.BytesIO()
        with zipfile.ZipFile(inner, "w") as iz:
            iz.writestr("archive/data.pkl", pickle.dumps(_Evil()))
            iz.writestr("archive/extra.pkl", pickle.dumps(_Benign()))
        p = tmp_path / "model.nemo"
        TestNemoTarContainer._write_tar(p, [("model_weights.ckpt", inner.getvalue())])

        real = ModelFileScanner._scan_pickle
        calls = 0

        def flaky(self, file_path, data):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("crafted")
            return real(self, file_path, data)
        monkeypatch.setattr(ModelFileScanner, "_scan_pickle", flaky)

        findings = ModelFileScanner().scan_file(p)
        assert any(f.rule_id == "MFV-PICKLE-001" for f in findings)
        assert any(f.rule_id == "MFV-SKIP-002" for f in findings)

    def test_inner_zip_open_failure_is_not_clean(self, tmp_path):
        """_scan_zip_bytes: a tar member with the zip local-file magic that is
        not actually a zip previously returned [] with nothing said."""
        p = tmp_path / "model.nemo"
        TestNemoTarContainer._write_tar(
            p, [("model_weights.ckpt", b"PK\x03\x04" + b"\xff" * 100)],
        )

        findings = ModelFileScanner().scan_file(p)
        assert any(
            f.rule_id == "MFV-SKIP-002" and "model_weights.ckpt" in f.message
            for f in findings
        )

    def test_npz_walk_failure_keeps_container_findings(self, tmp_path, monkeypatch):
        """_scan_numpy (.npz): the walk failing partway previously discarded
        the container-discipline findings collected before the failure."""
        p = tmp_path / "evil.npz"
        with zipfile.ZipFile(p, "w") as zf:
            zf.writestr("../escape.npy", b"\x93NUMPY\x01\x00")

        def boom(self, file_path, data, member=""):
            raise OSError("crafted")
        monkeypatch.setattr(ModelFileScanner, "_scan_npy_bytes", boom)

        findings = ModelFileScanner().scan_file(p)
        assert any(f.rule_id == "MFV-NPZ-001" for f in findings)
        assert any(f.rule_id == "MFV-SKIP-002" for f in findings)

    def test_onnx_recursion_limit_is_not_silent(self, tmp_path, monkeypatch):
        """_scan_onnx: RecursionError from the protobuf walk previously
        returned the (empty) findings list, silencing the whole scan."""
        p = tmp_path / "model.onnx"
        p.write_bytes(_pb_string_field(7, "graph"))

        def boom(data, max_depth=12):
            raise RecursionError("crafted")
        monkeypatch.setattr("hayward.scanner._extract_protobuf_strings", boom)

        findings = ModelFileScanner().scan_file(p)
        assert [f.rule_id for f in findings] == ["MFV-SKIP-003"]

    def test_tf_savedmodel_recursion_limit_is_not_silent(self, tmp_path, monkeypatch):
        p = tmp_path / "saved_model.pb"
        p.write_bytes(_pb_string_field(1, "node"))

        def boom(data, max_depth=12):
            raise RecursionError("crafted")
        monkeypatch.setattr("hayward.scanner._extract_protobuf_strings", boom)

        findings = ModelFileScanner().scan_file(p)
        assert [f.rule_id for f in findings] == ["MFV-SKIP-003"]
