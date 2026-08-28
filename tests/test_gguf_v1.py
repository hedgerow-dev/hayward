"""GGUF version 1 must not false-positive.

A comparative benchmark caught Hayward flagging a real, valid public GGUF model
(klosax/tinyllamas-stories-gguf) at HIGH with MFV-GGUF-005. Root cause: GGUF v1
stores its tensor/kv counts and its string/array lengths as 32-bit fields, while
v2 and v3 use 64-bit. Reading a v1 header with the v2/v3 layout decodes the
counts into absurd values (billions of tensors in a 1 MB file), which the layout
check reports as wrapped-arithmetic overflow. v1 is now recognised and treated
as a coverage gap (INFO), not a verdict.
"""

from __future__ import annotations

import struct

from hayward import ModelFileScanner


def _scan(tmp_path, name, blob):
    p = tmp_path / name
    p.write_bytes(blob)
    return ModelFileScanner().scan_file(p)


def _gguf_v1(tensor_count: int = 0, kv_count: int = 0) -> bytes:
    # v1 header: magic, version (u32), tensor_count (u32), kv_count (u32).
    return b"GGUF" + struct.pack("<III", 1, tensor_count, kv_count)


def test_gguf_v1_is_coverage_not_a_false_high(tmp_path):
    findings = _scan(tmp_path, "model.gguf", _gguf_v1())
    rule_ids = [f.rule_id for f in findings]
    # The old bug fired MFV-GGUF-005 at HIGH; now the only finding is coverage.
    assert rule_ids == ["MFV-GGUF-004"], [(f.rule_id, f.severity) for f in findings]
    assert findings[0].metadata.get("skipped_reason") == "gguf_v1_unparsed"
    assert not any(f.rule_id == "MFV-GGUF-005" for f in findings)


def test_gguf_v1_does_not_fail_the_high_gate(tmp_path):
    # A v1 header carrying counts that would look like overflow under the u64
    # layout must still not reach the fail-on-high threshold.
    findings = _scan(tmp_path, "model.gguf", _gguf_v1(tensor_count=48, kv_count=18))
    assert not any(f.severity.value in ("critical", "high") for f in findings), (
        [(f.rule_id, f.severity) for f in findings]
    )


def test_gguf_v3_still_parses_and_is_not_treated_as_v1(tmp_path):
    # A structurally valid v3 header (u64 counts) is unaffected by the v1 path.
    blob = b"GGUF" + struct.pack("<IQQ", 3, 0, 0)
    findings = _scan(tmp_path, "model.gguf", blob)
    assert not any(
        f.metadata.get("skipped_reason") == "gguf_v1_unparsed" for f in findings
    )
