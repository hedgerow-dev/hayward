"""Legitimate GGUF variants must not be reported as exploits or tampering.

A 1.7.0 benchmark over 108 small GGUF files from popular Hub repos flagged four
benign ones. Each test below builds the same shape:

* a 5-dimension tensor (ComfyUI-GGUF video VAEs store these),
* a 0-dimension scalar tensor (an NVIDIA TTS model ships one),
* a tensor type id from a fork's own quantisation (ids 100 and 101 in a ROCm build),
* a vocabulary under an architecture's own key prefix (`<arch>.tokenizer.tokens`)
  that contains words like "subprocess".

llama.cpp refuses the first three cleanly, which is a compatibility gap, not a
parser exploit. The u64 wrap and overrun checks still run on every tensor.
"""

from __future__ import annotations

import struct

from hayward import ModelFileScanner

STRING, ARRAY = 8, 9
F32 = 0


def _str(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def _gguf(tensors, kv=()):
    """tensors: (name, dims, type_id, offset). kv: (key, string or list of strings)."""
    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv))
    for key, values in kv:
        if isinstance(values, str):
            out += _str(key) + struct.pack("<I", STRING) + _str(values)
            continue
        out += _str(key) + struct.pack("<IIQ", ARRAY, STRING, len(values))
        out += b"".join(_str(v) for v in values)
    for name, dims, type_id, offset in tensors:
        out += _str(name) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims)
        out += struct.pack("<IQ", type_id, offset)
    return out + b"\0" * 4096


def _serious(tmp_path, blob):
    p = tmp_path / "model.gguf"
    p.write_bytes(blob)
    return [(f.rule_id, f.message[:120]) for f in ModelFileScanner().scan_file(p)
            if f.severity.value in ("critical", "high", "medium")]


def test_a_five_dimension_tensor_is_not_an_exploit(tmp_path):
    assert _serious(tmp_path, _gguf([("vae.conv.weight", (3, 3, 3, 4, 2), F32, 0)])) == []


def test_a_scalar_tensor_is_not_an_exploit(tmp_path):
    assert _serious(tmp_path, _gguf([("scale", (), F32, 0)])) == []
    assert _serious(tmp_path, _gguf([("scale", (), 8, 0)])) == []  # Q8_0, a block type


def test_a_fork_type_id_is_not_an_exploit(tmp_path):
    assert _serious(tmp_path, _gguf([("blk.0.w", (32, 4), 100, 0), ("blk.1.w", (32, 4), 101, 0)])) == []


def test_a_vocab_under_an_architecture_prefix_is_data(tmp_path):
    vocab = ["hello", "subprocess", "<class", "world"]
    assert _serious(tmp_path, _gguf([], kv=[("kimodo.tokenizer.tokens", vocab)])) == []


# ---- the real exploit shapes still fire ------------------------------------------------

def test_dimensions_that_wrap_u64_still_fire(tmp_path):
    rules = [r for r, _ in _serious(tmp_path, _gguf([("w", (1 << 40, 1 << 30, 1 << 10, 2, 2), F32, 0)]))]
    assert "MFV-GGUF-005" in rules


def test_an_absurd_dimension_count_still_fires(tmp_path):
    blob = b"GGUF" + struct.pack("<IQQ", 3, 1, 0) + _str("w") + struct.pack("<I", 1_000_000) + b"\0" * 64
    assert "MFV-GGUF-005" in [r for r, _ in _serious(tmp_path, blob)]


def test_an_overrunning_tensor_still_fires(tmp_path):
    rules = [r for r, _ in _serious(tmp_path, _gguf([("w", (1 << 20, 1 << 10), F32, 0)]))]
    assert "MFV-GGUF-005" in rules


def test_a_payload_in_an_ordinary_metadata_key_still_fires(tmp_path):
    rules = [r for r, _ in _serious(tmp_path, _gguf([], kv=[("general.notes", ["import subprocess"])]))]
    assert "MFV-GGUF-002" in rules


def test_a_named_chat_template_called_tokens_is_not_treated_as_vocab(tmp_path):
    kv = [("tokenizer.chat_template.tokens", ["{{ __import__('os') }}"])]
    assert "MFV-GGUF-002" in [r for r, _ in _serious(tmp_path, _gguf([], kv=kv))]


# llama.cpp can select a named template (`tokenizer.chat_template.<name>`) and
# render it, so MFV-GGUF-003 covers those the same as the default template.
# This payload has none of MFV-GGUF-002's substrings, so only GGUF-003 sees it.
_RCE_TEMPLATE = "{{ cycler.__init__.__globals__.os.popen('id').read() }}"


def test_a_malicious_named_chat_template_fires(tmp_path):
    kv = [("tokenizer.chat_template.tool_use", _RCE_TEMPLATE)]
    found = _serious(tmp_path, _gguf([], kv=kv))
    assert [r for r, _ in found] == ["MFV-GGUF-003"]
    assert "tokenizer.chat_template.tool_use" in found[0][1]


def test_a_benign_named_chat_template_is_clean(tmp_path):
    template = "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}{% endfor %}"
    assert _serious(tmp_path, _gguf([], kv=[("tokenizer.chat_template.tool_use", template)])) == []
