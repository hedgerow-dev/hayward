"""GGUF / GGML container parsing and chat-template SSTI signatures.

Extracted verbatim from ``hayward.scanner`` (HW-147): structure-aware GGUF
metadata parsing, the container-arithmetic layout replay, and the shared
Jinja2 SSTI signature used by both the GGUF chat_template path and the
transformers JSON path.

``_unsafe_name_reason`` lives in ``hayward._tensors`` and is re-exported on
``hayward.scanner``; ``_check_gguf_layout`` reaches it lazily through
``_scanner._unsafe_name_reason`` so the circular import stays safe.
"""

from __future__ import annotations

import re
import struct
from typing import Any

import hayward.scanner as _scanner

# Code-execution/sandbox-escape constructs that distinguish a real SSTI
# payload from ordinary Jinja2 variable substitution inside a chat_template.
# Every legitimate chat-tuned model's template is full of "{{ }}" -- that's
# just how Jinja2 spells a variable reference -- so presence of "{{" alone
# is not a signal. What distinguishes a payload is Python object
# introspection, sandbox escape, or a shell-out primitive appearing inside
# the template.
_GGUF_CHAT_TEMPLATE_SSTI_SIGNATURES: tuple[str, ...] = (
    "__globals__", "__class__", "__mro__", "__subclasses__",
    "__builtins__", ".__init__.__globals__",
    "__base__", "__bases__", "__reduce__", "__getattribute__",
    "os.", "subprocess.", "popen",
    # The `|attr` filter reassembles a dunder from a string literal, e.g.
    # ``{{ ''|attr('__class__') }}``: the dunder hides inside a literal, which
    # this pass blanks, so the signatures above would miss it. `|attr` itself
    # stays in code position and is essentially never used by a real chat
    # template. `namespace`/`cycler`/`lipsum` are deliberately NOT signatures:
    # modern chat templates use `namespace()` for stateful loops, so matching it
    # would false-positive on legitimate models.
    "|attr", "attr(",
)

# The only parts of a Jinja2 template that execute: {{ expression }} and
# {% statement %}. Prose outside these delimiters (and inside {# comments #})
# is rendered verbatim, so signature-matching there measures vocabulary, not
# behavior.
_GGUF_JINJA_BLOCK_RE = re.compile(r"\{\{.*?\}\}|\{%.*?%\}", re.DOTALL)

# String literals inside a Jinja block are data, not code: {{ 'scenarios.' }}
# renders the text, it does not touch `os`. Real templates embed instructions
# this way (measured: unsloth's DeepSeek-V4 template carries its system-prompt
# prose inside a {{ '...' }} literal), so literals are blanked before the
# signature match. {{ os.system('id') }} still matches via `os.` in code
# position; {{ ''.__class__ }} still matches via `__class__`.
_GGUF_JINJA_STRING_RE = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")


def _jinja_ssti_signature(template: str) -> str | None:
    """Return the first code-execution signature that appears in *code
    position* inside a Jinja2 template, or None if the template is ordinary
    substitution.

    Code position is the inside of a ``{{ }}`` / ``{% %}`` block with string
    literals blanked out: prose outside a block renders verbatim, and a string
    literal inside a block is data, so neither can reach ``os`` or Python's
    object-introspection surface. Blank the literals, join what remains of the
    blocks, then look for a signature.

    Factored out so the GGUF metadata path (MFV-GGUF-003) and the transformers
    JSON path (MFV-HF-001) share one definition of "payload vs. ordinary
    template" and cannot drift apart. Real chat templates trip the raw
    signatures in prose and inside ``{{ '...' }}`` literals (measured:
    unsloth's DeepSeek-V4 template hits "os." on the word "scenarios."), which
    is exactly what the code-position restriction filters out.
    """
    executable = _GGUF_JINJA_STRING_RE.sub(
        "",
        " ".join(m.group(0) for m in _GGUF_JINJA_BLOCK_RE.finditer(template)),
    )
    return next(
        (sig for sig in _GGUF_CHAT_TEMPLATE_SSTI_SIGNATURES if sig in executable),
        None,
    )


# ── GGUF metadata parsing ───────────────────────────────────────────
#
# Structure-aware KV parsing per the GGUF binary spec, instead of decoding
# the raw file bytes and substring-matching across everything (which used to
# match tensor names/binary weight data as readily as actual metadata, and
# couldn't distinguish "this string is a metadata *value*" from "this byte
# sequence just happens to appear somewhere in the file"). Only string and
# string-array typed KV entries are materialized; every other type is
# skipped by its known fixed/computed size so the offset stays correct
# without needing to store values we don't check.

GGUF_MAGIC = b"GGUF"

# GGML is GGUF's predecessor and is still shipped: whisper.cpp publishes its
# models in it, often under a .gguf name. Stored little-endian, its magic
# 0x67676d6c reads as b"lmgg" on disk. Calling one of those "corrupted" is
# wrong twice over, since the file is neither corrupt nor GGUF, and a hub
# sweep found this was 10 of 17 actionable findings across 1,974 real files.
GGML_MAGICS = (b"lmgg", b"fmgg", b"tjgg")   # ggml, ggmf, ggjt

GGUF_METADATA_SCAN_BYTES = 10_000_000

# Standard GGUF `general.*` keys documented as free-text/descriptive (per the
# GGUF KV spec) -- exempt from the dangerous-substring content check below,
# since ordinary prose in a description/author/license/tags field routinely
# contains words like "subprocess" or "class" without being executable.
_GGUF_FREETEXT_KEYS = frozenset({
    "general.name", "general.description", "general.author",
    "general.organization", "general.license", "general.license.name",
    "general.url", "general.doi", "general.repo_url", "general.tags",
    "general.languages", "general.datasets", "general.finetune",
    "general.basename", "general.quantized_by", "general.size_label",
    "general.source.url", "general.source.doi", "general.source.repo_url",
    # tokenizer.ggml.tokens is the model's vocabulary: a list of arbitrary
    # substrings harvested from training text, not metadata anyone wrote. A
    # code-trained vocab necessarily contains tokens like "<class", "exec("
    # and "__import__" (measured: unsloth's DeepSeek-V4 GGUFs trip three of
    # the patterns below on vocab alone). Vocabulary is data, and no code
    # path in a GGUF runtime executes it.
    #
    # The same argument covers the other tokenizer tables. merges is the BPE
    # merge list, built from the same training text and equally certain to
    # contain arbitrary substrings. A 2,456-file hub sweep found unsloth's
    # gemma GGUFs tripping the "subprocess" pattern on merges alone, which is
    # this exact false positive recurring on a different key. token_type and
    # scores are numeric, and the chat template keeps its own dedicated check
    # (MFV-GGUF-003), which looks for execution constructs rather than
    # substrings.
    "tokenizer.ggml.tokens",
    "tokenizer.ggml.merges",
    "tokenizer.ggml.token_type",
    "tokenizer.ggml.scores",
})

_GGUF_TYPE_STRING = 8
_GGUF_TYPE_ARRAY = 9
_GGUF_FIXED_TYPE_SIZES: dict[int, int] = {
    0: 1,  # UINT8
    1: 1,  # INT8
    2: 2,  # UINT16
    3: 2,  # INT16
    4: 4,  # UINT32
    5: 4,  # INT32
    6: 4,  # FLOAT32
    7: 1,  # BOOL
    10: 8,  # UINT64
    11: 8,  # INT64
    12: 8,  # FLOAT64
}

# ggml type id -> (block_size, type_size), from ggml's public type traits.
# Needed to replay nbytes arithmetic the way llama.cpp computes it, which is
# where the integer-overflow CVEs live (CVE-2026-33298's ne = [1024, 1024,
# 4398046511105, 1] wraps the u64 product). Unknown ids simply skip the
# bounds replay; an implausibly large id is itself reported.
_GGML_TYPE_TRAITS: dict[int, tuple[int, int]] = {
    0: (1, 4),      # F32
    1: (1, 2),      # F16
    2: (32, 18),    # Q4_0
    3: (32, 20),    # Q4_1
    6: (32, 22),    # Q5_0
    7: (32, 24),    # Q5_1
    8: (32, 34),    # Q8_0
    9: (32, 40),    # Q8_1
    10: (256, 84),  # Q2_K
    11: (256, 110),  # Q3_K
    12: (256, 144),  # Q4_K
    13: (256, 176),  # Q5_K
    14: (256, 210),  # Q6_K
    15: (256, 292),  # Q8_K
    16: (256, 66),  # IQ2_XXS
    17: (256, 74),  # IQ2_XS
    18: (256, 98),  # IQ3_XXS
    19: (256, 50),  # IQ1_S
    20: (32, 18),   # IQ4_NL
    21: (256, 110),  # IQ3_S
    22: (256, 82),  # IQ2_S
    23: (256, 136),  # IQ4_XS
    24: (1, 1),     # I8
    25: (1, 2),     # I16
    26: (1, 4),     # I32
    27: (1, 8),     # I64
    28: (1, 8),     # F64
    29: (256, 56),  # IQ1_M
    30: (1, 2),     # BF16
    34: (256, 54),  # TQ1_0
    35: (256, 66),  # TQ2_0
    39: (256, 82),  # IQ2_M
}

_GGUF_MAX_DIMS = 4          # GGML_MAX_DIMS
_GGUF_MAX_PLAUSIBLE_TYPE = 64
_GGUF_MAX_KEY_BYTES = 1 << 20
_U64 = 1 << 64


def _check_gguf_layout(data: bytes) -> list[str]:
    """Replay the GGUF container's arithmetic and report every inconsistency.

    llama.cpp allocates and reads based on these counts, lengths, dimensions
    and offsets, computed in u64. A file whose numbers do not fit the file
    (or fit only by wrapping u64) is a parser exploit, not a corrupt model:
    CVE-2025-53630 (count overflow to OOB), CVE-2026-27940 (underallocate,
    then fread past the buffer) and CVE-2026-33298 (nbytes wraps, defeating
    size validation) are all this shape. Computed in Python ints, which do
    not wrap; anything exceeding u64 is reported as the wrap shape.
    """
    problems: list[str] = []
    file_size = len(data)
    if file_size < 24:
        return problems  # too small to hold a header; magic check owns this
    version, tensor_count, kv_count = struct.unpack_from("<IQQ", data, 4)
    if version not in (1, 2, 3):
        problems.append(f"GGUF version {version} (spec defines 1-3)")
    # Every KV entry costs at least 8 (length) + 4 (type) bytes; every tensor
    # info at least 8 (name length) + 4 (n_dims) + 4 (type) + 8 (offset).
    # A count the file cannot pay for is a heap-overflow setup.
    if kv_count * 12 > file_size:
        problems.append(
            f"kv_count {kv_count} cannot fit in {file_size} bytes "
            f"(min 12 bytes/entry)"
        )
    if tensor_count * 24 > file_size:
        problems.append(
            f"tensor_count {tensor_count} cannot fit in {file_size} bytes "
            f"(min 24 bytes/info)"
        )

    offset = 24
    try:
        for i in range(min(kv_count, file_size // 12 + 1)):
            (key_len,) = struct.unpack_from("<Q", data, offset)
            offset += 8
            if key_len > _GGUF_MAX_KEY_BYTES:
                problems.append(f"KV key length {key_len} bytes (buffer-overflow shape)")
                return problems
            if offset + key_len + 4 > file_size:
                problems.append(f"KV entry {i} extends past end of file")
                return problems
            offset += key_len
            (value_type,) = struct.unpack_from("<I", data, offset)
            offset += 4
            if value_type == _GGUF_TYPE_STRING:
                (vlen,) = struct.unpack_from("<Q", data, offset)
                offset += 8
                if offset + vlen > file_size:
                    problems.append(f"KV string value {i} extends past end of file")
                    return problems
                offset += vlen
            elif value_type == _GGUF_TYPE_ARRAY:
                (elem_type,) = struct.unpack_from("<I", data, offset)
                (count,) = struct.unpack_from("<Q", data, offset + 4)
                offset += 12
                if elem_type == _GGUF_TYPE_STRING:
                    truncated = False
                    for _ in range(min(count, file_size // 8 + 1)):
                        (slen,) = struct.unpack_from("<Q", data, offset)
                        offset += 8
                        if offset + slen > file_size:
                            problems.append(f"KV string array {i} extends past end of file")
                            truncated = True
                            break
                        offset += slen
                    if truncated:
                        return problems
                else:
                    elem_size = _GGUF_FIXED_TYPE_SIZES.get(elem_type)
                    if elem_size is None:
                        problems.append(f"KV array {i} has unknown element type {elem_type}")
                        return problems
                    if offset + count * elem_size > file_size:
                        problems.append(
                            f"KV array {i} ({count} x {elem_size}B) extends past end of file"
                        )
                        return problems
                    offset += count * elem_size
            else:
                size = _GGUF_FIXED_TYPE_SIZES.get(value_type)
                if size is None:
                    problems.append(f"KV entry {i} has unknown value type {value_type}")
                    return problems
                offset += size
    except (struct.error, IndexError):
        problems.append("KV section truncated before kv_count entries")
        return problems

    try:
        for i in range(min(tensor_count, file_size // 24 + 1)):
            (name_len,) = struct.unpack_from("<Q", data, offset)
            if name_len > _GGUF_MAX_KEY_BYTES:
                problems.append(f"tensor {i} name length {name_len} bytes")
                return problems
            try:
                tensor_name = data[offset + 8:offset + 8 + name_len].decode(
                    "utf-8", "replace")
            except (ValueError, IndexError):
                tensor_name = ""
            reason = _scanner._unsafe_name_reason(tensor_name)
            if reason is not None:
                problems.append(
                    f"tensor {i} name {tensor_name!r} carries a {reason}")
            if offset + 8 + name_len + 16 > file_size:
                problems.append(f"tensor info {i} extends past end of file")
                return problems
            offset += 8 + name_len
            (n_dims,) = struct.unpack_from("<I", data, offset)
            offset += 4
            if n_dims < 1 or n_dims > _GGUF_MAX_DIMS:
                problems.append(f"tensor {i} has {n_dims} dimensions (GGML_MAX_DIMS is 4)")
                return problems
            if offset + 8 * n_dims + 12 > file_size:
                problems.append(f"tensor info {i} extends past end of file")
                return problems
            dims = struct.unpack_from(f"<{n_dims}Q", data, offset)
            offset += 8 * n_dims
            (type_id,) = struct.unpack_from("<I", data, offset)
            offset += 4
            (tensor_offset,) = struct.unpack_from("<Q", data, offset)
            offset += 8
            if type_id >= _GGUF_MAX_PLAUSIBLE_TYPE:
                problems.append(f"tensor {i} has implausible ggml type id {type_id}")
                continue
            elements = 1
            wrapped = False
            for dim in dims:
                elements *= dim
                if elements >= _U64:
                    problems.append(
                        f"tensor {i} dimensions wrap u64 (CVE-2026-33298 shape)"
                    )
                    wrapped = True
                    break
            traits = _GGML_TYPE_TRAITS.get(type_id)
            if traits is None or wrapped:
                continue
            block, type_size = traits
            nbytes = (dims[0] // block) * type_size if block > 1 else elements * type_size
            if block > 1:
                for dim in dims[1:]:
                    nbytes *= dim
            if nbytes >= _U64:
                problems.append(f"tensor {i} byte size wraps u64")
                continue
            # Tensor data starts after ALL tensor infos, aligned, so the true
            # end is at least this large: a file failing even the base-less
            # check cannot hold the tensor under any alignment.
            if tensor_offset + nbytes > file_size:
                problems.append(
                    f"tensor {i} (offset {tensor_offset} + {nbytes} bytes) "
                    f"overruns the {file_size}-byte file"
                )
    except (struct.error, IndexError):
        problems.append("tensor info section truncated before tensor_count entries")
    return problems


def _read_gguf_string(data: bytes, offset: int) -> tuple[str, int]:
    (length,) = struct.unpack_from("<Q", data, offset)
    offset += 8
    raw = data[offset:offset + length]
    return raw.decode("utf-8", errors="replace"), offset + length


def _read_gguf_value(data: bytes, offset: int, value_type: int) -> tuple[Any, int]:
    """Read a GGUF value, returning (value, new_offset).

    Only STRING and ARRAY-of-STRING are materialized (all the security
    checks below need); every other type is skipped by size and returned as
    None.
    """
    if value_type == _GGUF_TYPE_STRING:
        return _read_gguf_string(data, offset)
    if value_type == _GGUF_TYPE_ARRAY:
        (elem_type,) = struct.unpack_from("<I", data, offset)
        offset += 4
        (count,) = struct.unpack_from("<Q", data, offset)
        offset += 8
        if elem_type == _GGUF_TYPE_STRING:
            values: list[str] = []
            for _ in range(count):
                s, offset = _read_gguf_string(data, offset)
                values.append(s)
            return values, offset
        elem_size = _GGUF_FIXED_TYPE_SIZES.get(elem_type)
        if elem_size is None:
            raise ValueError(f"unknown GGUF array element type {elem_type}")
        return None, offset + count * elem_size
    size = _GGUF_FIXED_TYPE_SIZES.get(value_type)
    if size is None:
        raise ValueError(f"unknown GGUF value type {value_type}")
    return None, offset + size


def _parse_gguf_metadata(data: bytes, max_offset: int) -> tuple[dict[str, Any], bool]:
    """Parse the GGUF header + metadata KV section into a dict of
    string/string-array-valued keys. Raises ValueError/struct.error on a
    malformed or truncated stream -- callers should treat that as
    "can't verify structurally" rather than a positive signal either way.

    Returns ``(result, window_truncated)``. ``window_truncated`` is True only
    when the content scan stopped at the `max_offset` window with KV entries
    still unread: metadata past the window was never content-checked, and the
    caller reports that as a coverage gap. A clean finish (all entries read) or
    a structural error (which raises) both leave it False.
    """
    if len(data) < 24 or data[:4] != GGUF_MAGIC:
        raise ValueError("not a GGUF file")
    _version, _tensor_count, kv_count = struct.unpack_from("<IQQ", data, 4)
    offset = 4 + 4 + 8 + 8
    result: dict[str, Any] = {}
    window_truncated = False
    for _ in range(kv_count):
        if offset > max_offset:
            # Ran into the content-scan window with entries still to read.
            # Signal the gap rather than silently returning a partial dict.
            window_truncated = True
            break
        key, offset = _read_gguf_string(data, offset)
        # _read_gguf_string advances by an attacker-declared length, so offset
        # can land far past the file inside a single iteration. unpack_from
        # raises OverflowError rather than struct.error once it exceeds
        # ssize_t, which is not in this function's documented failure set and
        # escaped to the caller as a crash. Re-check the bound here instead.
        if offset < 0 or offset + 4 > len(data):
            raise ValueError(f"KV entry runs past the file at offset {offset}")
        (value_type,) = struct.unpack_from("<I", data, offset)
        offset += 4
        value, offset = _read_gguf_value(data, offset, value_type)
        if value is not None:
            result[key] = value
    return result, window_truncated
