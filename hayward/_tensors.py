"""Tensor-container layout checks: SafeTensors, TFLite, and the schema-less
protobuf walker shared by ONNX and TF SavedModel.

Extracted verbatim from ``hayward.scanner`` (HW-147). Pure structural analysis
over model bytes: FlatBuffers/SafeTensors size arithmetic, the tensor-name
safety check, and a generic protobuf field walker with the ONNX external-data
and dangerous-op tables.

This module is self-contained -- it references no names that stay in
``hayward.scanner`` -- so it does not import the scanner module at all.
"""

from __future__ import annotations

import re
import struct

# ── TFLite layout arithmetic ─────────────────────────────────────
#
# TFLite is FlatBuffers: no code executes on load, but the *parser* computes
# tensor sizes from attacker-chosen dimensions, and CVE-2026-42627 (ArmNN,
# published 2026-05) is exactly that: TensorShape::GetNumElements()
# multiplies dimensions in 32-bit without overflow detection, understates
# the allocation, and BatchToSpaceNd reads past it during Optimize(). The
# check is the same arithmetic-invariant shape as the GGUF/SafeTensors
# passes: replay the dimension product in ints that do not wrap and flag
# anything a 32-bit loader cannot hold.
#
# FlatBuffers layout used here (no schema needed):
#   buffer[0:4]   u32 offset to the root table
#   buffer[4:8]   "TFL3" file identifier
#   table:        int32 soffset back to its vtable; vtable is
#                 u16 vtable_len, u16 table_len, u16 offset per field id
#   vector:       u32 offset to it, then u32 count, then elements
# Field ids (TFLite schema, stable since 2017): Model.subgraphs = 2,
# SubGraph.tensors = 1, Tensor.shape = 1, Tensor.type = 2.

_TFLITE_MAGIC = b"TFL3"
_TFLITE_MODEL_SUBGRAPHS = 2
_TFLITE_SUBGRAPH_TENSORS = 1
_TFLITE_TENSOR_SHAPE = 1

_TFLITE_MAX_DIMS_PRODUCT_32 = 1 << 32


def _fb_table_field(data: bytes, table: int, field_id: int) -> int | None:
    """Absolute offset of a table field's inline value, or None if absent
    (FlatBuffers omits default-valued fields)."""
    if table < 4 or table + 4 > len(data):
        return None
    (soffset,) = struct.unpack_from("<i", data, table)
    vtable = table - soffset
    if vtable < 0 or vtable + 4 > len(data):
        return None
    vtable_len, _table_len = struct.unpack_from("<HH", data, vtable)
    slot = 4 + 2 * field_id
    if slot + 2 > vtable_len or vtable + slot + 2 > len(data):
        return None
    (rel,) = struct.unpack_from("<H", data, vtable + slot)
    if rel == 0:
        return None
    return table + rel


def _fb_indirect(data: bytes, at: int | None) -> int | None:
    """Follow a u32 relative offset to its target's absolute offset."""
    if at is None or at + 4 > len(data):
        return None
    (rel,) = struct.unpack_from("<I", data, at)
    target = at + rel
    return target if 0 <= target <= len(data) else None


def _fb_int_vector(data: bytes, vec: int | None) -> list[int] | None:
    """Read a vector of int32 at absolute offset `vec`."""
    if vec is None or vec + 4 > len(data):
        return None
    (count,) = struct.unpack_from("<I", data, vec)
    if count > 8:  # tensor rank is small; GGML_MAX_DIMS is 4
        return None
    if vec + 4 + 4 * count > len(data):
        return None
    return list(struct.unpack_from(f"<{count}i", data, vec + 4)) if count else []


def _check_tflite_layout(data: bytes) -> list[str]:
    """Replay every tensor's dimension product in a TFLite model and report
    shapes a 32-bit loader cannot allocate."""
    problems: list[str] = []
    if len(data) < 8:
        return problems
    (root_rel,) = struct.unpack_from("<I", data, 0)
    root = _fb_indirect(data, 0) if root_rel else None
    if root is None:
        return ["root table offset points outside the file"]
    subgraphs_vec = _fb_indirect(data, _fb_table_field(data, root, _TFLITE_MODEL_SUBGRAPHS))
    if subgraphs_vec is None or subgraphs_vec + 4 > len(data):
        return problems  # no subgraphs: unusual, but nothing to replay
    (n_subgraphs,) = struct.unpack_from("<I", data, subgraphs_vec)
    if n_subgraphs > 4096:
        problems.append(f"{n_subgraphs} subgraphs (implausible)")
        return problems
    for sg in range(n_subgraphs):
        subgraph = _fb_indirect(data, subgraphs_vec + 4 + 4 * sg)
        if subgraph is None:
            continue
        tensors_vec = _fb_indirect(data, _fb_table_field(data, subgraph, _TFLITE_SUBGRAPH_TENSORS))
        if tensors_vec is None or tensors_vec + 4 > len(data):
            continue
        (n_tensors,) = struct.unpack_from("<I", data, tensors_vec)
        if n_tensors > (len(data) // 4):
            problems.append(
                f"subgraph {sg} claims {n_tensors} tensors in a {len(data)}-byte file"
            )
            return problems
        for t in range(n_tensors):
            tensor = _fb_indirect(data, tensors_vec + 4 + 4 * t)
            if tensor is None:
                continue
            shape = _fb_int_vector(data, _fb_indirect(
                data, _fb_table_field(data, tensor, _TFLITE_TENSOR_SHAPE)))
            if shape is None or not shape:
                continue
            if any(d < 0 for d in shape):
                problems.append(f"tensor {t} of subgraph {sg} has a negative dimension")
                continue
            elements = 1
            for d in shape:
                elements *= d
                if elements >= _TFLITE_MAX_DIMS_PRODUCT_32:
                    problems.append(
                        f"tensor {t} of subgraph {sg} has {elements} elements: "
                        f"a 32-bit loader's dimension product wraps "
                        f"(CVE-2026-42627 shape)"
                    )
                    break
    return problems


_SAFETENSORS_DTYPE_SIZES: dict[str, int] = {

    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1,
    "U64": 8, "U32": 4, "U16": 2, "U8": 1,
    "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1,
}

# A single tensor larger than 4GB is the integer-overflow shape in C-side
# loaders; aggregate sizes past the file are the OOB
# shape (its MFV014). Both are replayed here as arithmetic, not thresholds.
_SAFETENSORS_MAX_TENSOR_BYTES = 1 << 32


def _unsafe_name_reason(name: str) -> str | None:
    """Why a tensor or member name is not safe to use as a filename.

    Names inside a model container look inert, but plenty of real tooling
    turns them into paths: shard converters, `save_pretrained` round trips,
    and anything that materialises tensors individually. A name carrying a
    traversal segment, an absolute path or a control character is aimed at
    that behaviour, and it is not something a training run produces by
    accident. Published bypass proofs of concept ship exactly these five
    shapes for SafeTensors alone.
    """
    if not isinstance(name, str) or not name:
        return None
    if "\x00" in name:
        return "embedded NUL byte"
    if "\r" in name or "\n" in name:
        return "embedded newline, which can forge a record boundary"
    normalised = name.replace("\\", "/")
    parts = normalised.split("/")
    if ".." in parts:
        return "parent-directory traversal segment"
    if normalised.startswith("/"):
        return "absolute path"
    if len(name) > 1 and name[1] == ":" and name[0].isalpha():
        return "Windows drive-absolute path"
    return None


def _check_safetensors_layout(header: object, data_section_len: int) -> list[str]:
    """Replay a SafeTensors header's size arithmetic against the file.

    The header is attacker-chosen JSON; loaders allocate and memcpy from it.
    The spec itself requires offsets to be ordered and non-overlapping, so a
    file breaking that is malformed by its own contract, and safetensors-rs
    refuses it -- which is precisely why it must not sail through here.
    """
    problems: list[str] = []
    if not isinstance(header, dict):
        return problems
    spans: list[tuple[int, int, str]] = []
    for name, entry in header.items():
        reason = _unsafe_name_reason(name)
        if reason is not None:
            problems.append(f"tensor name {name!r} carries a {reason}")
        if name == "__metadata__" or not isinstance(entry, dict):
            continue
        offsets = entry.get("data_offsets")
        if (
            not isinstance(offsets, list) or len(offsets) != 2
            or not all(isinstance(o, int) and o >= 0 for o in offsets)
        ):
            problems.append(f"tensor {name!r} has malformed data_offsets")
            continue
        start, end = offsets
        if end < start:
            problems.append(f"tensor {name!r} has end before start in data_offsets")
            continue
        span = end - start
        shape = entry.get("shape")
        dtype = entry.get("dtype")
        dtype_size = _SAFETENSORS_DTYPE_SIZES.get(dtype) if isinstance(dtype, str) else None
        if (
            dtype_size is not None
            and isinstance(shape, list)
            and all(isinstance(d, int) and d >= 0 for d in shape)
        ):
            elements = 1
            for d in shape:
                elements *= d
            if elements * dtype_size != span:
                problems.append(
                    f"tensor {name!r}: shape x dtype = {elements * dtype_size} bytes "
                    f"but its span is {span}"
                )
        if span > _SAFETENSORS_MAX_TENSOR_BYTES:
            problems.append(
                f"tensor {name!r} is {span} bytes (>4GB, integer-overflow shape "
                f"in C loaders)"
            )
        if end > data_section_len:
            problems.append(
                f"tensor {name!r} ends at {end}, past the {data_section_len}-byte "
                f"data section"
            )
        spans.append((start, end, name))
    spans.sort()
    # spans[1:] is one shorter by construction, so the pairing is
    # deliberately ragged: strict=False.
    for (_prev_start, prev_end, prev_name), (start, _end, name) in zip(
            spans, spans[1:], strict=False):
        if start < prev_end:
            problems.append(
                f"tensors {prev_name!r} and {name!r} overlap in the data section"
            )
    return problems


# exec/eval/import are matched as whole words in __metadata__ keys: bare
# substring matching false-positived at CRITICAL on ordinary keys like
# `evaluation_metric` and `import_date`. `__import__` keeps its own
# alternative because \b treats the surrounding underscores as word
# characters and would never match it. __reduce__/__builtins__ stay
# substring checks -- the surrounding underscores already delimit them, and
# no benign key embeds either.
_ST_METADATA_DANGEROUS_KEY_RE = re.compile(r"\b(?:exec|eval|import)\b|__import__")


# ── Minimal schema-less protobuf walker (ONNX + TF SavedModel) ──────
#
# Both formats are protobuf messages (onnx.ModelProto / tensorflow.GraphDef),
# but pulling in the `onnx`/`protobuf` packages as a hard dependency just to
# read op names and path strings out of a possibly-malicious model file is
# more surface area than this needs. Protobuf's wire format is fully
# self-describing enough to walk generically without the .proto schema: each
# field is a (field_number, wire_type) varint key followed by a
# type-appropriate value, and a "wire_type 2" (length-delimited) field is
# either a string/bytes leaf or a nested submessage -- ambiguous without the
# schema, so this walks it both ways (recurse if it parses as a submessage,
# also try it as UTF-8 text) rather than guessing.

def _read_varint(data: bytes, i: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        if i >= len(data):
            raise IndexError("truncated varint")
        b = data[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, i
        shift += 7
        if shift > 70:
            raise ValueError("varint too long")


def _iter_protobuf_fields(data: bytes):
    """Yield (field_number, wire_type, value) for each top-level field.

    Tolerates truncated/malformed input by stopping early (not raising) --
    this walks attacker-controlled files, so a deliberately corrupted
    protobuf blob must degrade to "found nothing further", never a crash.
    Bails on wire types 3/4 (deprecated start/end group) rather than risk
    misinterpreting the rest of the stream.
    """
    i = 0
    n = len(data)
    while i < n:
        try:
            key, i = _read_varint(data, i)
        except (IndexError, ValueError):
            return
        wire_type = key & 0x7
        field_num = key >> 3
        if wire_type == 0:
            try:
                val, i = _read_varint(data, i)
            except (IndexError, ValueError):
                return
            yield field_num, wire_type, val
        elif wire_type == 1:
            if i + 8 > n:
                return
            yield field_num, wire_type, data[i : i + 8]
            i += 8
        elif wire_type == 2:
            try:
                length, i = _read_varint(data, i)
            except (IndexError, ValueError):
                return
            if length < 0 or i + length > n:
                return
            yield field_num, wire_type, data[i : i + length]
            i += length
        elif wire_type == 5:
            if i + 4 > n:
                return
            yield field_num, wire_type, data[i : i + 4]
            i += 4
        else:
            return


def _extract_protobuf_strings(data: bytes, max_depth: int = 12) -> list[str]:
    """Recursively collect every printable UTF-8 string found in any
    length-delimited field, at any nesting depth -- op names, custom-op
    attribute values, and external-data path references all show up as
    plain string leaves somewhere in this tree, regardless of which message
    type actually declares them (which we don't know, schema-less)."""
    strings: list[str] = []
    if max_depth <= 0:
        return strings
    for _field_num, wire_type, value in _iter_protobuf_fields(data):
        if wire_type != 2 or not isinstance(value, (bytes, bytearray)):
            continue
        sub_fields = list(_iter_protobuf_fields(value))
        if sub_fields:
            strings.extend(_extract_protobuf_strings(bytes(value), max_depth - 1))
        try:
            text = value.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if text and text.isprintable():
            strings.append(text)
    return strings


# ONNX custom ops with documented code-execution behavior (onnxruntime-extensions'
# PyOp/PythonOp embeds a Python callable, by name or source, that runs at
# inference time -- this is the real, previously-exploited ONNX RCE vector,
# not a hypothetical).
ONNX_DANGEROUS_OPS: frozenset[str] = frozenset({"PyOp", "PythonOp", "Inference"})

# TensorFlow ops that read/write the filesystem or run an embedded Python
# callable at graph-execution time -- the same class of "presence is the
# signal" evidence as MFV-KERAS-001's Lambda-layer check.
#
# Deliberately NOT in this set: ShardedFilename and MergeV2Checkpoints.
# tf.train.Saver adds both to a graph's save/restore machinery, so they
# appear in the MetaGraphDef of ordinary SavedModels as checkpoint
# boilerplate and run only when the save/restore ops are explicitly
# invoked, never at inference. Their presence carries no signal.
TF_DANGEROUS_OPS: frozenset[str] = frozenset({
    "PyFunc", "PyFuncStateless", "ReadFile", "WriteFile",
})

# The only keys ONNX's TensorProto.external_data map (StringStringEntryProto)
# can legitimately carry. Before onnx 1.21.0 (CVE-2026-34445) these keys were
# passed to setattr() on the ExternalDataInfo object, so a key outside this
# set overwrote internal properties. A file carrying one is never legitimate.
_ONNX_EXTERNAL_DATA_KEYS: frozenset[str] = frozenset(
    {"location", "offset", "length", "checksum"}
)

# Absolute path shapes that escape the model's directory just as surely as a
# ".." segment: POSIX root (but not the protocol-relative "//"), UNC, and
# drive-letter paths. Checked only against external_data location values,
# never whole-file: ONNX node names are hierarchical scope paths that begin
# with "/" by convention ("/bert/Cast"), so a leading slash anywhere else is
# the naming scheme, not a path.
_ONNX_ABSOLUTE_PATH_RE = re.compile(r"^(?:/[^/]|\\\\|[A-Za-z]:[\\/])")

# Anything that is not a plain relative filename: a scheme, a UNC path,
# or a protocol-relative reference.
_ONNX_REMOTE_LOCATION_RE = re.compile(
    r"^(?:[a-zA-Z][a-zA-Z0-9+.-]*://|//|\\\\\\\\)"
)


def _string_string_entry(value: bytes) -> tuple[str, str] | None:
    """If `value` parses as a StringStringEntryProto {key: 1, value: 2} with
    exactly those two fields and both decode as UTF-8, return (key, value),
    else None. Anything richer is some other message and not ours to read."""
    key: str | None = None
    val: str | None = None
    for field_num, wire_type, field_value in _iter_protobuf_fields(value):
        if wire_type != 2 or not isinstance(field_value, (bytes, bytearray)):
            return None
        try:
            text = bytes(field_value).decode("utf-8")
        except UnicodeDecodeError:
            return None
        if field_num == 1:
            key = text
        elif field_num == 2:
            val = text
        else:
            return None
    if key is None:
        return None
    return key, val if val is not None else ""


def _onnx_external_data_maps(data: bytes, max_depth: int = 12) -> list[dict[str, str]]:
    """Collect message-shaped regions that look like a TensorProto's repeated
    external_data map. The walk is schema-less, so the anchor is the map's own
    contents: a StringStringEntry map carrying a "location" key is, in an ONNX
    file, external_data with overwhelming probability (ModelProto.metadata_props,
    the only other such map in common use, keys on author/license/converted_by,
    not "location")."""
    maps: list[dict[str, str]] = []
    if max_depth <= 0:
        return maps
    entries: dict[str, str] = {}
    for _field_num, wire_type, value in _iter_protobuf_fields(data):
        if wire_type != 2 or not isinstance(value, (bytes, bytearray)):
            continue
        entry = _string_string_entry(bytes(value))
        if entry is not None:
            entries[entry[0]] = entry[1]
        else:
            maps.extend(_onnx_external_data_maps(bytes(value), max_depth - 1))
    if "location" in entries:
        maps.append(entries)
    return maps
