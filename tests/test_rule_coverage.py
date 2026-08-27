"""HW-131: direct tests for rules that previously had no named test.

Every fixture is hand-built and tiny: the SafeTensors format is an 8-byte
little-endian u64 header length + JSON header + tensor bytes; GGUF is the
``GGUF`` magic + version/counts header + typed KV entries; ``.keras`` is a
zip holding ``config.json``. The same builder patterns as
test_new_formats.py, reproduced here so this file stands alone.

The meta-test at the bottom re-reads docs/rules.md and the tests/
directory at runtime, so a future rule added to the docs without a test
asserting its ID fails CI instead of silently shipping untested.
"""

from __future__ import annotations

import json
import os
import pickle
import re
import struct
import zipfile
from pathlib import Path

from hayward.findings import Severity
from hayward.scanner import ModelFileScanner

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent

_RULE_ID_RE = re.compile(r"MFV-[A-Z0-9]+-\d+")


# ── fixture builders ────────────────────────────────────────────────


def _safetensors(header_obj: dict, data_bytes: int = 0) -> bytes:
    """8-byte u64 header length + JSON header + tensor bytes."""
    header = json.dumps(header_obj).encode()
    return struct.pack("<Q", len(header)) + header + b"\x00" * data_bytes


def _gguf(kvs: list[tuple[bytes, bytes]]) -> bytes:
    """Minimal structurally-valid GGUF: magic + version 3 header + string-typed
    (value_type=8) KV entries, same shape as test_new_formats._build_gguf."""

    def gguf_string(s: bytes) -> bytes:
        return struct.pack("<Q", len(s)) + s

    header = b"GGUF" + struct.pack("<IQQ", 3, 0, len(kvs))
    body = b"".join(gguf_string(k) + struct.pack("<I", 8) + gguf_string(v) for k, v in kvs)
    return header + body


def _keras_zip(config: dict) -> bytes:
    """A .keras container: a zip whose config.json carries the layer graph."""
    import io

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("config.json", json.dumps(config))
    return buf.getvalue()


class _Evil:
    def __reduce__(self):
        return (os.system, ("echo pwned",))


def _scan(tmp_path: Path, name: str, blob: bytes):
    p = tmp_path / name
    p.write_bytes(blob)
    return ModelFileScanner().scan_file(p)


# ── MFV-ST-002 / 003 / 004 / 005 ────────────────────────────────────


class TestSafetensorsRules:
    """The four SafeTensors integrity rules, each asserted by name."""

    def test_st_002_header_size_over_limit(self, tmp_path):
        # Declared header length just past the 100 MB limit.
        blob = struct.pack("<Q", 100_000_001) + b"\x00" * 8
        findings = _scan(tmp_path, "model.safetensors", blob)
        assert [f.rule_id for f in findings] == ["MFV-ST-002"]
        assert findings[0].severity == Severity.HIGH

    def test_st_003_header_extends_past_eof(self, tmp_path):
        # Header length inside the limit but far past the actual file end.
        blob = struct.pack("<Q", 1000) + b"{}"
        findings = _scan(tmp_path, "model.safetensors", blob)
        assert [f.rule_id for f in findings] == ["MFV-ST-003"]
        assert findings[0].severity == Severity.HIGH

    def test_st_004_header_not_json(self, tmp_path):
        # Right length, undecodable-as-JSON header bytes.
        header = b"\xff\xfe{not json"
        blob = struct.pack("<Q", len(header)) + header
        findings = _scan(tmp_path, "model.safetensors", blob)
        assert [f.rule_id for f in findings] == ["MFV-ST-004"]
        assert findings[0].severity == Severity.HIGH

    def test_st_005_dangerous_metadata_key_is_critical(self, tmp_path):
        blob = _safetensors(
            {
                "__metadata__": {"exec": "1", "__import__": "os"},
                "w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
            },
            4,
        )
        findings = _scan(tmp_path, "model.safetensors", blob)
        st005 = [f for f in findings if f.rule_id == "MFV-ST-005"]
        assert st005, [(f.rule_id, f.message) for f in findings]
        assert st005[0].severity == Severity.CRITICAL
        assert sorted(st005[0].metadata["suspicious_keys"]) == ["__import__", "exec"]

    def test_st_005_benign_metadata_keys_do_not_trigger(self, tmp_path):
        # Word-boundary matching: substrings of exec/eval/import stay clean.
        blob = _safetensors(
            {
                "__metadata__": {"evaluation_metric": "auc", "import_date": "2026-01-01"},
                "w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
            },
            4,
        )
        findings = _scan(tmp_path, "model.safetensors", blob)
        assert not any(f.rule_id == "MFV-ST-005" for f in findings)


# ── MFV-GGUF-001 ────────────────────────────────────────────────────


class TestGgufMagicRule:
    """MFV-GGUF-001: magic number missing or invalid."""

    def test_bad_magic_flags_gguf_001(self, tmp_path):
        # One byte off the GGUF magic, enough length to be a real header.
        blob = b"GGUG" + b"\x00" * 12
        findings = _scan(tmp_path, "model.gguf", blob)
        assert [f.rule_id for f in findings] == ["MFV-GGUF-001"]
        assert findings[0].severity == Severity.HIGH

    def test_truncated_file_flags_gguf_001(self, tmp_path):
        findings = _scan(tmp_path, "model.gguf", b"GG")
        assert [f.rule_id for f in findings] == ["MFV-GGUF-001"]

    def test_ggml_magic_is_non_coverage_not_gguf_001(self, tmp_path):
        # GGUF's predecessor is a known format the scanner does not parse:
        # the honest verdict is MFV-SKIP-003, not a bad-magic claim.
        findings = _scan(tmp_path, "model.gguf", b"lmgg" + b"\x00" * 12)
        assert [f.rule_id for f in findings] == ["MFV-SKIP-003"]
        assert findings[0].metadata["skipped_reason"] == "ggml_format"

    def test_valid_magic_does_not_trigger(self, tmp_path):
        findings = _scan(tmp_path, "model.gguf", _gguf([(b"general.architecture", b"llama")]))
        assert not any(f.rule_id == "MFV-GGUF-001" for f in findings)


# ── MFV-KERAS-002 ───────────────────────────────────────────────────


class TestKerasUnrecognizedClassRule:
    """MFV-KERAS-002: a layer class not on the known-builtin list."""

    def test_custom_layer_class_flags_keras_002(self, tmp_path):
        config = {
            "class_name": "Sequential",
            "config": {
                "name": "sequential",
                "layers": [
                    {"class_name": "MyCustomLayer", "config": {"name": "custom_1"}},
                    {"class_name": "Dense", "config": {"name": "dense_1"}},
                ],
            },
        }
        findings = _scan(tmp_path, "model.keras", _keras_zip(config))
        keras002 = [f for f in findings if f.rule_id == "MFV-KERAS-002"]
        assert keras002, [(f.rule_id, f.message) for f in findings]
        assert keras002[0].severity == Severity.INFO
        assert keras002[0].metadata["unrecognized_classes"] == ["MyCustomLayer"]

    def test_builtin_only_graph_does_not_trigger(self, tmp_path):
        config = {
            "class_name": "Sequential",
            "config": {
                "name": "sequential",
                "layers": [
                    {"class_name": "Dense", "config": {"name": "dense_1"}},
                    {"class_name": "Dropout", "config": {"name": "dropout_1"}},
                ],
            },
        }
        findings = _scan(tmp_path, "model.keras", _keras_zip(config))
        assert not any(f.rule_id == "MFV-KERAS-002" for f in findings)


# ── MFV-CONFUSE-001 ─────────────────────────────────────────────────


class TestExtensionConfusionRule:
    """MFV-CONFUSE-001: the extension says one format, the magic bytes say
    another. Fires only when the extension-directed parse structurally
    failed AND the bytes positively match a different known format."""

    def test_pickle_bytes_in_safetensors_trigger(self, tmp_path):
        blob = pickle.dumps(_Evil(), protocol=2)
        findings = _scan(tmp_path, "model.safetensors", blob)
        confuse = [f for f in findings if f.rule_id == "MFV-CONFUSE-001"]
        assert confuse, [(f.rule_id, f.message) for f in findings]
        assert confuse[0].metadata["extension_format"] == "safetensors"
        assert confuse[0].metadata["sniffed_format"] == "pickle"
        # The real content is re-scanned as the sniffed format: the denied
        # os.system global inside the pickle is still convicted.
        assert any(f.rule_id == "MFV-PICKLE-001" for f in findings)

    def test_gguf_bytes_in_safetensors_trigger(self, tmp_path):
        blob = _gguf([(b"general.architecture", b"llama")])
        findings = _scan(tmp_path, "model.safetensors", blob)
        confuse = [f for f in findings if f.rule_id == "MFV-CONFUSE-001"]
        assert confuse, [(f.rule_id, f.message) for f in findings]
        assert confuse[0].metadata["sniffed_format"] == "gguf"

    def test_zip_pt_checkpoint_does_not_trigger(self, tmp_path):
        # A modern .pt is a zip wrapping the pickle: the extension-confusion
        # check is deliberately skipped for it (the zip path has its own
        # member-level handling), so a legitimate checkpoint stays clean.
        p = tmp_path / "model.pt"
        with zipfile.ZipFile(p, "w") as zf:
            zf.writestr("archive/data.pkl", pickle.dumps({"w": [1.0]}, protocol=2))
        findings = ModelFileScanner().scan_file(p)
        assert not any(f.rule_id == "MFV-CONFUSE-001" for f in findings)

    def test_corrupted_safetensors_matching_no_format_does_not_trigger(self, tmp_path):
        # A parse failure alone is "corrupted", not "spoofed": without a
        # positive match for some OTHER format there is no confusion finding.
        blob = struct.pack("<Q", 16) + b"\x00" * 16
        findings = _scan(tmp_path, "model.safetensors", blob)
        assert not any(f.rule_id == "MFV-CONFUSE-001" for f in findings)

    def test_valid_safetensors_does_not_trigger(self, tmp_path):
        blob = _safetensors(
            {"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}, 4
        )
        findings = _scan(tmp_path, "model.safetensors", blob)
        assert findings == []


# ── meta-test: every documented rule has a named test ───────────────


def test_every_documented_rule_has_a_named_test():
    """docs/rules.md is the contract: each rule ID it lists must appear in at
    least one test file, so a rule cannot ship (or regress) without a test
    asserting it by name."""
    rules_md = (REPO_ROOT / "docs" / "rules.md").read_text(encoding="utf-8")
    documented = set(_RULE_ID_RE.findall(rules_md))
    assert documented, "no rule IDs found in docs/rules.md -- regex or doc changed?"

    tested: set[str] = set()
    for py in sorted(TESTS_DIR.glob("test_*.py")):
        tested.update(_RULE_ID_RE.findall(py.read_text(encoding="utf-8")))

    missing = documented - tested
    assert not missing, (
        f"rules documented in docs/rules.md with no test asserting them by name: "
        f"{sorted(missing)}"
    )
