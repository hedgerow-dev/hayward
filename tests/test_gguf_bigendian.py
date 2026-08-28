"""Big-endian GGUF must not false-positive.

The scaled comparative benchmark caught Hayward flagging a real, valid public
big-endian GGUF model (taronaeo/tinyllamas-BE) at HIGH with MFV-GGUF-005. The
GGUF magic (`GGUF`) is four ASCII bytes and reads the same either way, but the
version and every count/length after it are stored big-endian on big-endian
builds (llama.cpp ships these for s390x). Read little-endian, the version
decodes to a huge byte-swapped value and the u64 counts look like wrapped
arithmetic, which the layout check reports as a parser exploit. Big-endian is
now recognised and treated as a coverage gap (INFO), not a verdict, the same
way GGUF v1 and the GGML predecessor are.
"""

from __future__ import annotations

import struct

from hayward import ModelFileScanner


def _scan(tmp_path, name, blob):
    p = tmp_path / name
    p.write_bytes(blob)
    return ModelFileScanner().scan_file(p)


def _gguf_be(version: int = 3, tensor_count: int = 4, kv_count: int = 4) -> bytes:
    # magic is byte-order independent; version + u64 counts stored big-endian.
    return b"GGUF" + struct.pack(">IQQ", version, tensor_count, kv_count)


def test_big_endian_gguf_is_coverage_not_a_false_high(tmp_path):
    findings = _scan(tmp_path, "model.gguf", _gguf_be(3, 4, 4))
    rule_ids = [f.rule_id for f in findings]
    # The bug fired MFV-GGUF-005 at HIGH; now the only finding is coverage.
    assert "MFV-GGUF-005" not in rule_ids, [(f.rule_id, f.severity) for f in findings]
    assert not any(f.severity.value in ("critical", "high") for f in findings), [
        (f.rule_id, f.severity) for f in findings
    ]
    assert any(f.metadata.get("skipped_reason") == "gguf_big_endian" for f in findings), [
        (f.rule_id, f.metadata) for f in findings
    ]


def test_big_endian_gguf_v2_also_recognised(tmp_path):
    findings = _scan(tmp_path, "model.gguf", _gguf_be(2, 8, 8))
    assert not any(f.severity.value in ("critical", "high") for f in findings), [
        (f.rule_id, f.severity) for f in findings
    ]


def test_little_endian_v3_still_parses_and_is_not_treated_as_big_endian(tmp_path):
    blob = b"GGUF" + struct.pack("<IQQ", 3, 0, 0)
    findings = _scan(tmp_path, "model.gguf", blob)
    assert not any(f.metadata.get("skipped_reason") == "gguf_big_endian" for f in findings), [
        (f.rule_id, f.metadata) for f in findings
    ]
