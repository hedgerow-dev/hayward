"""Security analysis of machine-learning model files.

Detects code execution, deserialisation attacks and malformed containers
without importing, deserialising or executing anything it reads.

Covers:
  - Pickle (.pkl, .pt, .pth): unsafe opcode and __reduce__ detection
  - SafeTensors: header validation and layout arithmetic
  - GGUF: metadata inspection and container arithmetic
  - Keras H5: Lambda layer and custom object detection
  - ONNX: custom-op RCE (PyOp/PythonOp) and external_data handling
  - TensorFlow SavedModel: GraphDef ops that read, write or execute
  - numpy .npy/.npz: allow_pickle object-array payloads
  - joblib: a plain or zlib-compressed pickle stream
  - Keras keras_metadata.pb -- the layer graph a SavedModel export writes
    outside saved_model.pb, where Lambda-layer payloads actually live
  - TorchServe .mar and NVIDIA NeMo .nemo -- zip/tar containers whose
    inner checkpoint is an ordinary pickle
  - skops .skops -- schema type references against the pickle classifier,
    plus the two loader invariants behind CVE-2025-54412/54413
"""

from __future__ import annotations

import ast
import bz2
import contextlib
import io
import json
import logging
import lzma
import re
import shutil
import struct
import subprocess
import tarfile
import tempfile
import zipfile
import zlib
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from defusedxml.common import EntitiesForbidden

# ── Facade re-exports (HW-147) ──────────────────────────────────────
#
# scanner.py was split into private sibling modules. The module-level helpers,
# constants and dataclasses those modules now own are re-imported here so that
# `from hayward.scanner import <name>` keeps resolving for every name the test
# suite, cli.py and __init__.py rely on. The redundant `as` aliases mark these
# as intentional re-exports (ruff F401 / mypy explicit re-export). The moved
# code reads any test-tunable constants (pickle/keras/embedded-exec limits)
# back off this module by attribute, so monkeypatching them here still steers
# the extracted functions.
from hayward._binary import _ELF_MACHINES as _ELF_MACHINES
from hayward._binary import _MACHO_CPUTYPES as _MACHO_CPUTYPES
from hayward._binary import _MACHO_MAGICS as _MACHO_MAGICS
from hayward._binary import _find_embedded_executables as _find_embedded_executables
from hayward._gguf import _GGML_TYPE_TRAITS as _GGML_TYPE_TRAITS
from hayward._gguf import _GGUF_CHAT_TEMPLATE_SSTI_SIGNATURES as _GGUF_CHAT_TEMPLATE_SSTI_SIGNATURES
from hayward._gguf import _GGUF_FIXED_TYPE_SIZES as _GGUF_FIXED_TYPE_SIZES
from hayward._gguf import _GGUF_FREETEXT_KEYS as _GGUF_FREETEXT_KEYS
from hayward._gguf import _GGUF_JINJA_BLOCK_RE as _GGUF_JINJA_BLOCK_RE
from hayward._gguf import _GGUF_JINJA_STRING_RE as _GGUF_JINJA_STRING_RE
from hayward._gguf import _GGUF_MAX_DIMS as _GGUF_MAX_DIMS
from hayward._gguf import _GGUF_MAX_KEY_BYTES as _GGUF_MAX_KEY_BYTES
from hayward._gguf import _GGUF_MAX_PLAUSIBLE_TYPE as _GGUF_MAX_PLAUSIBLE_TYPE
from hayward._gguf import _GGUF_TYPE_ARRAY as _GGUF_TYPE_ARRAY
from hayward._gguf import _GGUF_TYPE_STRING as _GGUF_TYPE_STRING
from hayward._gguf import _U64 as _U64
from hayward._gguf import GGML_MAGICS as GGML_MAGICS
from hayward._gguf import GGUF_MAGIC as GGUF_MAGIC
from hayward._gguf import GGUF_METADATA_SCAN_BYTES as GGUF_METADATA_SCAN_BYTES
from hayward._gguf import _check_gguf_layout as _check_gguf_layout
from hayward._gguf import _jinja_ssti_signature as _jinja_ssti_signature
from hayward._gguf import _parse_gguf_metadata as _parse_gguf_metadata
from hayward._gguf import _read_gguf_string as _read_gguf_string
from hayward._gguf import _read_gguf_value as _read_gguf_value
from hayward._keras import _KERAS_BUILTIN_LAYER_CLASSES as _KERAS_BUILTIN_LAYER_CLASSES
from hayward._keras import _KERAS_CONFIG_ANCHOR as _KERAS_CONFIG_ANCHOR
from hayward._keras import _KERAS_CONFIG_BACKTRACK as _KERAS_CONFIG_BACKTRACK
from hayward._keras import _KERAS_CONFIG_WINDOW as _KERAS_CONFIG_WINDOW
from hayward._keras import _KERAS_MAX_CONFIG_ANCHORS as _KERAS_MAX_CONFIG_ANCHORS
from hayward._keras import _KERAS_STREAM_CHUNK as _KERAS_STREAM_CHUNK
from hayward._keras import _balanced_brace_json_end as _balanced_brace_json_end
from hayward._keras import _extract_keras_model_config as _extract_keras_model_config
from hayward._keras import _find_keras_risky_layers as _find_keras_risky_layers
from hayward._keras import _find_keras_unrecognized_classes as _find_keras_unrecognized_classes
from hayward._keras import _read_keras_config_window as _read_keras_config_window
from hayward._lfs import _LFS_OID_RE as _LFS_OID_RE
from hayward._lfs import _LFS_PROBE_BYTES as _LFS_PROBE_BYTES
from hayward._lfs import _LFS_SIZE_RE as _LFS_SIZE_RE
from hayward._lfs import _LFS_VERSION_LINE as _LFS_VERSION_LINE
from hayward._lfs import _lfs_pointer_finding as _lfs_pointer_finding
from hayward._lfs import _parse_lfs_pointer as _parse_lfs_pointer
from hayward._pickle_engine import _BYTES_PUSH_OPCODES as _BYTES_PUSH_OPCODES
from hayward._pickle_engine import _EMBEDDED_PICKLE_MAX_DEPTH as _EMBEDDED_PICKLE_MAX_DEPTH
from hayward._pickle_engine import _FLOAT_PUSH_OPCODES as _FLOAT_PUSH_OPCODES
from hayward._pickle_engine import _INT_PUSH_OPCODES as _INT_PUSH_OPCODES
from hayward._pickle_engine import _MAX_PICKLE_RESYNCS as _MAX_PICKLE_RESYNCS
from hayward._pickle_engine import _MEMO_FETCH_OPCODES as _MEMO_FETCH_OPCODES
from hayward._pickle_engine import _MEMO_INDEX_BAND_FACTOR as _MEMO_INDEX_BAND_FACTOR
from hayward._pickle_engine import _MEMO_INDEX_BAND_FLOOR as _MEMO_INDEX_BAND_FLOOR
from hayward._pickle_engine import _MEMO_INDEX_MAX_TRACKED as _MEMO_INDEX_MAX_TRACKED
from hayward._pickle_engine import _MEMO_STORE_OPCODES as _MEMO_STORE_OPCODES
from hayward._pickle_engine import _NESTED_PICKLE_OPENERS as _NESTED_PICKLE_OPENERS
from hayward._pickle_engine import _OPAQUE_PUSH_OPCODES as _OPAQUE_PUSH_OPCODES
from hayward._pickle_engine import _PICKLE_ALLOWED_ML_CLASS_ROOTS as _PICKLE_ALLOWED_ML_CLASS_ROOTS
from hayward._pickle_engine import _PICKLE_ALLOWED_ML_EXACT as _PICKLE_ALLOWED_ML_EXACT
from hayward._pickle_engine import (
    _PICKLE_ALLOWED_STORAGE_PARENTS as _PICKLE_ALLOWED_STORAGE_PARENTS,
)
from hayward._pickle_engine import _PICKLE_ALLOWED_STRING_ARG_OK as _PICKLE_ALLOWED_STRING_ARG_OK
from hayward._pickle_engine import _PICKLE_ARG_HOSTNAME_RE as _PICKLE_ARG_HOSTNAME_RE
from hayward._pickle_engine import _PICKLE_ARG_PATH_RE as _PICKLE_ARG_PATH_RE
from hayward._pickle_engine import _PICKLE_ARG_REPR as _PICKLE_ARG_REPR
from hayward._pickle_engine import _PICKLE_ARG_SHELL_RE as _PICKLE_ARG_SHELL_RE
from hayward._pickle_engine import _PICKLE_ARG_URL_RE as _PICKLE_ARG_URL_RE
from hayward._pickle_engine import _PICKLE_CODE_OBJECT_BUILDERS as _PICKLE_CODE_OBJECT_BUILDERS
from hayward._pickle_engine import _PICKLE_DENIED_ATTR_NAMES as _PICKLE_DENIED_ATTR_NAMES
from hayward._pickle_engine import _PICKLE_DENIED_MODULES as _PICKLE_DENIED_MODULES
from hayward._pickle_engine import _PICKLE_GENERIC_ATTR_NAMES as _PICKLE_GENERIC_ATTR_NAMES
from hayward._pickle_engine import _PICKLE_MARK as _PICKLE_MARK
from hayward._pickle_engine import _PICKLE_MAX_PARTIAL_TEXTS as _PICKLE_MAX_PARTIAL_TEXTS
from hayward._pickle_engine import _PICKLE_MAX_SOURCE_PARSE_BYTES as _PICKLE_MAX_SOURCE_PARSE_BYTES
from hayward._pickle_engine import _PICKLE_ML_CLASS_NAME_RE as _PICKLE_ML_CLASS_NAME_RE
from hayward._pickle_engine import _PICKLE_OPAQUE as _PICKLE_OPAQUE
from hayward._pickle_engine import _PICKLE_PATTERN_ARG_CALLABLES as _PICKLE_PATTERN_ARG_CALLABLES
from hayward._pickle_engine import _PICKLE_PROTO0_GLOBAL_RE as _PICKLE_PROTO0_GLOBAL_RE
from hayward._pickle_engine import _PICKLE_RESYNC_MARKERS as _PICKLE_RESYNC_MARKERS
from hayward._pickle_engine import _PICKLE_STORAGE_NAME_RE as _PICKLE_STORAGE_NAME_RE
from hayward._pickle_engine import _STRING_PUSH_OPCODES as _STRING_PUSH_OPCODES
from hayward._pickle_engine import PICKLE_ALLOWED_GLOBALS as PICKLE_ALLOWED_GLOBALS
from hayward._pickle_engine import PICKLE_DANGER_SIGNATURES as PICKLE_DANGER_SIGNATURES
from hayward._pickle_engine import PICKLE_DENIED_GLOBALS as PICKLE_DENIED_GLOBALS
from hayward._pickle_engine import PickleMemoProfile as PickleMemoProfile
from hayward._pickle_engine import PickleResolvedCall as PickleResolvedCall
from hayward._pickle_engine import (
    _allowed_call_has_anomalous_string as _allowed_call_has_anomalous_string,
)
from hayward._pickle_engine import _classify_pickle_global as _classify_pickle_global
from hayward._pickle_engine import (
    _embedded_pickle_denied_globals as _embedded_pickle_denied_globals,
)
from hayward._pickle_engine import _is_denied_pickle_module as _is_denied_pickle_module
from hayward._pickle_engine import _is_ml_constructor_allowed as _is_ml_constructor_allowed
from hayward._pickle_engine import _is_pickle_literal as _is_pickle_literal
from hayward._pickle_engine import _iter_pickle_arg_values as _iter_pickle_arg_values
from hayward._pickle_engine import _iter_pickle_host_port_pairs as _iter_pickle_host_port_pairs
from hayward._pickle_engine import _iter_pickle_source_targets as _iter_pickle_source_targets
from hayward._pickle_engine import _laundered_denied_ref as _laundered_denied_ref
from hayward._pickle_engine import _nested_pickle_evidence as _nested_pickle_evidence
from hayward._pickle_engine import _nested_pickle_globals as _nested_pickle_globals
from hayward._pickle_engine import _next_pickle_offset as _next_pickle_offset
from hayward._pickle_engine import _pickle_arg_text as _pickle_arg_text
from hayward._pickle_engine import _pickle_source_denied_target as _pickle_source_denied_target
from hayward._pickle_engine import _pickle_stream_truncated as _pickle_stream_truncated
from hayward._pickle_engine import _PickleCallResultType as _PickleCallResultType
from hayward._pickle_engine import _PickleGlobalRef as _PickleGlobalRef
from hayward._pickle_engine import _PickleMarkType as _PickleMarkType
from hayward._pickle_engine import _PickleOpaqueType as _PickleOpaqueType
from hayward._pickle_engine import _PickleWalkBudgetExceeded as _PickleWalkBudgetExceeded
from hayward._pickle_engine import _pop_to_mark as _pop_to_mark
from hayward._pickle_engine import _profile_memo_indices as _profile_memo_indices
from hayward._pickle_engine import _propagate_result_invoked as _propagate_result_invoked
from hayward._pickle_engine import _resolve_pickle_globals as _resolve_pickle_globals
from hayward._pickle_engine import _triage_unknown_pickle_call as _triage_unknown_pickle_call
from hayward._pickle_engine import _walk_one_pickle as _walk_one_pickle
from hayward._tensors import _ONNX_ABSOLUTE_PATH_RE as _ONNX_ABSOLUTE_PATH_RE
from hayward._tensors import _ONNX_EXTERNAL_DATA_KEYS as _ONNX_EXTERNAL_DATA_KEYS
from hayward._tensors import _ONNX_REMOTE_LOCATION_RE as _ONNX_REMOTE_LOCATION_RE
from hayward._tensors import _SAFETENSORS_DTYPE_SIZES as _SAFETENSORS_DTYPE_SIZES
from hayward._tensors import _SAFETENSORS_MAX_TENSOR_BYTES as _SAFETENSORS_MAX_TENSOR_BYTES
from hayward._tensors import _ST_METADATA_DANGEROUS_KEY_RE as _ST_METADATA_DANGEROUS_KEY_RE
from hayward._tensors import _TFLITE_MAGIC as _TFLITE_MAGIC
from hayward._tensors import _TFLITE_MAX_DIMS_PRODUCT_32 as _TFLITE_MAX_DIMS_PRODUCT_32
from hayward._tensors import _TFLITE_MODEL_SUBGRAPHS as _TFLITE_MODEL_SUBGRAPHS
from hayward._tensors import _TFLITE_SUBGRAPH_TENSORS as _TFLITE_SUBGRAPH_TENSORS
from hayward._tensors import _TFLITE_TENSOR_SHAPE as _TFLITE_TENSOR_SHAPE
from hayward._tensors import ONNX_DANGEROUS_OPS as ONNX_DANGEROUS_OPS
from hayward._tensors import TF_DANGEROUS_OPS as TF_DANGEROUS_OPS
from hayward._tensors import _check_safetensors_layout as _check_safetensors_layout
from hayward._tensors import _check_tflite_layout as _check_tflite_layout
from hayward._tensors import _extract_protobuf_strings as _extract_protobuf_strings
from hayward._tensors import _fb_indirect as _fb_indirect
from hayward._tensors import _fb_int_vector as _fb_int_vector
from hayward._tensors import _fb_table_field as _fb_table_field
from hayward._tensors import _iter_protobuf_fields as _iter_protobuf_fields
from hayward._tensors import _onnx_external_data_maps as _onnx_external_data_maps
from hayward._tensors import _read_varint as _read_varint
from hayward._tensors import _string_string_entry as _string_string_entry
from hayward._tensors import _unsafe_name_reason as _unsafe_name_reason
from hayward.findings import Category, Finding, Severity
from hayward.signatures import find_signature_artifacts, signature_findings

logger = logging.getLogger(__name__)

# Fully-qualified `module.name` globals that grant code/command execution or
# native-code loading if referenced by a pickle's GLOBAL/STACK_GLOBAL opcode
# (regardless of whether __reduce__ actually invokes them via REDUCE/NEWOBJ --
# a reference alone is enough evidence of tampering intent for a model file).
# ── Pickle engine tunables (kept here for the test facade) ─────────
#
# The pickle VM, classifier and triage moved to hayward._pickle_engine and
# are re-exported at the top of this module. These two operand/memo budget
# caps stay here because the suite monkeypatches them on the scanner module;
# the VM reads them back off this module by attribute.
# Cap on entries in the simulated memo dict itself. The index cap above bounds
# the *profiling* set, but `memo[arg] = stack[-1]` grew a live dict alongside
# it: 500MB of `N\x94` (NONE+MEMOIZE) is 250M entries and a multi-GB OOM fully
# inside the default scan cap. Same bound and same 1000x-over-real-models
# headroom as the index cap. On overflow the memo freezes (no new entries
# recorded) and the walk continues: GLOBAL/STACK_GLOBAL detection does not
# read the memo, so a frozen memo degrades resolution of later GETs, never
# the verdict.
_PICKLE_MEMO_MAX_ENTRIES = 1_000_000
# Cap on the simulated operand stack. MARK/push bombs (`(` or `N` repeated to
# the scan cap) grow it without bound otherwise. Entries are shared sentinels
# or values already materialised from file bytes, so the bound is memory-safe;
# 4M is ~200x deeper than the largest legitimate stream measured here (a
# 20k-key dict pickled with MARK-based containers). On overflow the walk
# terminates and the caller reports the unread remainder as a coverage gap --
# a stream whose stack never fits is not the shape of a model file.
_PICKLE_STACK_MAX_DEPTH = 4_000_000


# Severity rank mirroring Finding.severity_order in findings.py (lower is
# worse). Selecting the worst verdict by index into _PICKLE_UNKNOWN_TIERS
# used to invert the result -- that dict inserts [HIGH, MEDIUM, LOW], so
# max() by insertion index returned the LOWEST severity present.
_SEVERITY_ORDER: dict[Severity, int] = {
    Severity.CRITICAL: 0,
    Severity.HIGH: 1,
    Severity.MEDIUM: 2,
    Severity.LOW: 3,
    Severity.INFO: 4,
}


# Confidence and message lead per escalated tier, in emission order. Both are
# a function of the severity alone, so they live here rather than being
# returned alongside it and carried around. The tiers span "an unrecognized
# callable is being handed a shell command" and "...handed a file path"; one
# message for both would overstate at one end.
_PICKLE_UNKNOWN_TIERS: dict[Severity, tuple[float, str]] = {
    Severity.HIGH: (0.70, "Pickle file invokes unrecognized global(s) with arguments indicating "
                          "command execution, network access, or data exfiltration"),
    Severity.MEDIUM: (0.50, "Pickle file invokes unrecognized global(s) with arguments that name a "
                            "code-execution callable for dynamic resolution"),
    Severity.LOW: (0.40, "Pickle file invokes unrecognized global(s) with filesystem path arguments"),
}


# ── Embedded executable detection ────────────────────────────────
#
# A model file that *contains* a loadable binary is essentially always
# malicious. The magics are 2-4 bytes, so matching them bare turns every
# large weight blob into a false positive ('MZ' appears by chance roughly
# once per 64K of random data). Each check below is therefore structural:
# the second stage has to parse, which random tensor bytes do not survive.

# Hard budget of candidate magic occurrences validated per format. The hit
# caps below only stop on *validated* hits; a file of near-miss magics
# (500MB of "MZMZ...") otherwise iterated once per occurrence -- minutes of
# pure-Python loop with zero hits to show. Real polyglot payloads carry one
# or a handful of binaries, so a few thousand validations per format is far
# beyond anything legitimate; past the budget that format's search stops.
# Read by _find_embedded_executables (moved to hayward._binary); kept here so
# the test suite can monkeypatch it on the scanner module. The detector itself
# is re-exported at the top of this module.
_EMBEDDED_EXEC_MAX_CANDIDATES = 4096


# ── Shared file helper ──────────────────────────────────────────────


def _read_file_magic(path: Path, count: int) -> bytes:
    """First `count` bytes, or b"" when the file cannot be read."""
    try:
        with open(path, "rb") as f:
            return f.read(count)
    except OSError:
        return b""


# HDF5's fixed 8-byte file signature (used by both legacy Keras .h5/.hdf5
# and any other HDF5-backed format) -- shared between _scan_keras and the
# format-sniffing helper below so the two don't drift on the literal.
HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


def _skip_unverified_finding(
    file_path: Path, detail: str, metadata: dict[str, Any] | None = None,
) -> Finding:
    """MFV-SKIP-003: the file could not be opened, read, or parsed, so its
    content was never verified. A parser error is a suspicious state, not a
    clean verdict (exception-oriented evasion, arXiv 2508.19774): real
    loaders are routinely more permissive than this scanner's parser, so a
    file that breaks the parse here can still execute downstream."""
    return Finding(
        rule_id="MFV-SKIP-003",
        message=f"{detail} Content could not be verified. NOT a clean verdict.",
        severity=Severity.LOW,
        category=Category.AI_ML,
        file_path=str(file_path),
        confidence=0.50,
        metadata=metadata or {},
    )


class ModelFormat(str, Enum):
    PICKLE = "pickle"
    SAFETENSORS = "safetensors"
    GGUF = "gguf"
    KERAS = "keras"
    PYTORCH_ZIP = "pytorch_zip"
    ONNX = "onnx"
    TF_SAVEDMODEL = "tf_savedmodel"
    KERAS_METADATA = "keras_metadata"
    NEMO = "nemo"
    NUMPY = "numpy"
    JOBLIB = "joblib"
    SKOPS = "skops"
    TFLITE = "tflite"
    SEVENZ = "sevenz"
    PMML = "pmml"
    UNKNOWN = "unknown"


# Directory names never worth walking: version control, virtualenvs, build
# output, and vendored dependency trees. A model file inside site-packages
# belongs to a library, not to the project being scanned.
SKIP_DIR_NAMES = frozenset({
    ".git", ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "__pycache__", "node_modules", "venv", ".venv", "env",
    "dist", "build", ".eggs", "site-packages", "dist-packages",
    "vendor", "vendored", "third_party", "third-party",
    "bower_components", ".yarn", ".pnp", "jspm_packages",
    "bundle", "bundles",
})



# joblib.dump(compress=...) accepts zlib, gzip, bz2, lzma and xz. Sniffing
# only zlib meant an lzma-compressed pickle was handed to the pickle walker
# still compressed, found no opcodes, and passed clean. A published bypass
# proof of concept exploits exactly that: payload2_lzma_rce.joblib carries
# builtins.eval("__import__('os').popen('id').read()") behind lzma.
_COMPRESSED_STREAM_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x1f\x8b", "gzip"),
    (b"BZh", "bz2"),
    (b"\xfd7zXZ\x00", "xz"),
    (b"\x5d\x00\x00", "lzma"),
)


def _sniff_compression(data: bytes) -> str | None:
    """Which codec wraps this stream, if any."""
    for magic, name in _COMPRESSED_STREAM_MAGIC:
        if data.startswith(magic):
            return name
    # zlib has no constant magic: byte 0 is 0x78 for the window sizes any
    # real compressor emits, and byte 1 encodes the check bits.
    if len(data) >= 2 and data[0] == 0x78 and data[1] in (0x01, 0x5E, 0x9C, 0xDA):
        return "zlib"
    return None


def _new_decompressor(kind: str):
    if kind == "zlib":
        return zlib.decompressobj()
    if kind == "gzip":
        return zlib.decompressobj(16 + zlib.MAX_WBITS)
    if kind == "bz2":
        return bz2.BZ2Decompressor()
    if kind == "xz":
        return lzma.LZMADecompressor(format=lzma.FORMAT_XZ)
    return lzma.LZMADecompressor(format=lzma.FORMAT_ALONE)


def _zip_blob_upto_archive_end(blob: bytes) -> bytes:
    """Trim trailing garbage past a zip archive's own EOCD.

    The raw member fallback reads up to the size cap because the declared
    member size is attacker-controlled and may be a lie; for a STORED member
    that means the read can run on into whatever follows the member in the
    outer container (its central directory, another member). A nested zip
    handed to zipfile with that garbage attached enumerates the *outer*
    archive's directory instead and never sees the inner payload.

    Re-derives the archive's true end from the structure itself: a run of
    local file headers, then the central directory, then the EOCD. Every
    length walked comes from a *local* record the attacker would have to
    forge a second time independently of the central-directory lie. Any
    inconsistency returns the blob unchanged -- never worse than before.
    """
    pos = 0
    n = len(blob)
    while pos + 30 <= n and blob[pos:pos + 4] == b"PK\x03\x04":
        name_len, extra_len = struct.unpack("<HH", blob[pos + 26:pos + 30])
        (comp_size,) = struct.unpack("<I", blob[pos + 18:pos + 22])
        pos += 30 + name_len + extra_len + comp_size
        if pos > n:
            return blob
    if pos + 4 > n or blob[pos:pos + 4] != b"PK\x01\x02":
        return blob
    while pos + 46 <= n and blob[pos:pos + 4] == b"PK\x01\x02":
        name_len, extra_len, cmt_len = struct.unpack("<HHH", blob[pos + 28:pos + 34])
        pos += 46 + name_len + extra_len + cmt_len
        if pos > n:
            return blob
    if pos + 22 <= n and blob[pos:pos + 4] == b"PK\x05\x06":
        (cmt_len,) = struct.unpack("<H", blob[pos + 20:pos + 22])
        end = pos + 22 + cmt_len
        if end <= n:
            return blob[:end]
    return blob


# Sentinel the raw zip reader returns (only when a caller opts in via
# report_oversized) to mean "this member decompresses past MAX_ZIP_MEMBER_BYTES"
# as distinct from None, which stays "unreadable / not located". The strict
# read path already reports the oversized case as MFV-PICKLE-003; this lets the
# raw fallback reach the same verdict instead of skipping the member silently.
_ZIP_MEMBER_OVERSIZED = object()


# Called once per completed file as (done, total, path). The scanner never
# writes progress itself: it hands these numbers to a caller-supplied callback
# so the cli can route them to stderr and stdout stays a clean report.
ProgressCallback = Callable[[int, int, Path], None]


def _path_excluded(path: Path, root: Path, patterns: list[str]) -> bool:
    """True if `path` matches any --exclude glob, tested as the file is walked.

    A pattern is matched (fnmatch) against three views of the path: its route
    relative to the scan root, its bare filename, and every individual path
    component. The component test is what lets a directory-name pattern such as
    `checkpoints` prune everything beneath that directory, not just a file
    literally named `checkpoints`.
    """
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path
    candidates = [relative.as_posix(), path.name, *relative.parts]
    for pattern in patterns:
        for candidate in candidates:
            if fnmatch(candidate, pattern):
                return True
    return False


@dataclass(frozen=True)
class _ScanConfig:
    """The scanner state a worker process must reproduce to scan identically.

    Only these two settings change a file's findings, so they are the only
    thing shipped to workers. A frozen dataclass is picklable, which a closure
    over the parent scanner would not be.
    """

    max_scan_bytes: int
    check_signatures: bool


# One configured scanner per worker process, built by the pool initializer and
# reused for every task that worker handles, so the format map and config are
# constructed once rather than per file.
_WORKER_SCANNER: ModelFileScanner | None = None


def _worker_init(config: _ScanConfig) -> None:
    """ProcessPoolExecutor initializer: stand up this worker's scanner once."""
    global _WORKER_SCANNER
    scanner = ModelFileScanner()
    # Instance attribute shadows the class constant for this run, exactly as
    # the cli's --max-size does in the main process.
    scanner.MAX_SCAN_BYTES = config.max_scan_bytes
    scanner.check_signatures = config.check_signatures
    _WORKER_SCANNER = scanner


def _worker_scan(path_str: str) -> list[Finding]:
    """Scan one file in a worker. Paths cross the process boundary as strings
    (a Path pickles fine too, but a str is unambiguous and cheap)."""
    if _WORKER_SCANNER is None:  # _worker_init always runs before any task
        raise RuntimeError("worker scanner was not initialized")
    return _WORKER_SCANNER.scan_file(Path(path_str))


def _scan_paths_parallel(
    files: list[Path],
    config: _ScanConfig,
    jobs: int,
    progress: ProgressCallback | None = None,
) -> list[tuple[Path, list[Finding]]]:
    """Scan `files` across a process pool and return per-file results.

    Results are gathered as futures complete (order is irrelevant, the report
    re-sorts), but every submitted file appears exactly once in the output, so
    the parallel result is the same set of findings the sequential scan
    produces.
    """
    total = len(files)
    results: list[tuple[Path, list[Finding]]] = []
    with ProcessPoolExecutor(
        max_workers=jobs, initializer=_worker_init, initargs=(config,)
    ) as executor:
        future_to_path = {
            executor.submit(_worker_scan, str(f)): f for f in files
        }
        done = 0
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            file_findings = future.result()
            done += 1
            if progress is not None:
                progress(done, total, path)
            results.append((path, file_findings))
    return results


class ModelFileScanner:
    """Scans ML model files for backdoors and unsafe content."""

    # Annotated as int (not the inferred Literal) because both are overridden
    # per instance: --max-size shadows MAX_SCAN_BYTES for a run, and tests
    # lower MAX_ZIP_MEMBER_BYTES to exercise the bomb caps.
    MAX_SCAN_BYTES: int = 500_000_000  # 500 MB; larger files are streamed or skipped
    MAX_ZIP_MEMBER_BYTES: int = 200_000_000  # 200 MB cap on decompressed pickle members

    # Extensions too generic to map to a format, but too commonly used by real
    # model files for a directory walk to skip. `.bin` is `pytorch_model.bin`,
    # the default weight filename transformers used before safetensors and
    # still the most common pickle-bearing file on HuggingFace -- and equally
    # the extension of any unrelated binary blob. `.zip` is the same problem
    # one level up: a zip holding `data.pkl` is a checkpoint whatever it is
    # called, and shipping a model as a zip is ordinary practice, but most
    # `.zip` files are not models. This set only widens `scan_directory`'s
    # rglob patterns; `scan_file` decides every unmapped extension by content
    # sniff, so it does not consult it.
    _AMBIGUOUS_EXTENSIONS: frozenset[str] = frozenset({".bin", ".zip"})

    # HuggingFace / transformers config files. These carry the auto_map,
    # trust_remote_code and chat_template RCE vectors (MFV-HF-001/002), and
    # they are always `*.json`. `.json` is not in `_format_map` because it is
    # not dispatched by magic bytes: `_scan_file_dispatch` routes a `.json`
    # only when its bytes parse as a JSON object. But a directory scan still has
    # to *find* them, or a downloaded model repo (the normal way this runs) is
    # reported clean while carrying the exact vectors this catches. The scan is
    # cheap and false-positive-safe: `_scan_json_config` fires only on those
    # three fields, so an unrelated `package.json` yields nothing.
    _CONFIG_EXTENSIONS: frozenset[str] = frozenset({".json"})

    def __init__(self):
        self._archive_depth = 0
        # Opt-in provenance pass. Off by default so an ordinary scan says
        # nothing about signatures; the CLI's --check-signatures turns it on,
        # and then every scanned file also gets an MFV-SIG-001 note when a
        # sibling signature or attestation artifact is present (see
        # hayward.signatures). Detection only: this tool does no cryptographic
        # verification and claims none.
        self.check_signatures = False
        self._format_map: dict[str, ModelFormat] = {
            ".pkl": ModelFormat.PICKLE,
            ".pickle": ModelFormat.PICKLE,
            ".pth": ModelFormat.PICKLE,
            ".pt": ModelFormat.PICKLE,
            ".ckpt": ModelFormat.PICKLE,
            # PyTorch Lite / mobile (`.ptl`): torch.jit's mobile export writes
            # a zip that wraps a pickle exactly like a `.pt` checkpoint, and
            # the on-device interpreter unpickles it on load. Same container,
            # same threat, so it takes the same PICKLE handling (which routes a
            # real zip through _scan_pytorch_zip and a flat pickle through the
            # opcode walk).
            ".ptl": ModelFormat.PICKLE,
            # `.th` is a torch checkpoint suffix in real use (OpenCLIP and
            # several eval harnesses write `*_stats.th`), and torch.load reads
            # it like any other checkpoint -- zip-wrapped or flat pickle. Both
            # paths are already handled below; only the extension was missing.
            ".th": ModelFormat.PICKLE,
            # TorchServe model archive: a zip holding the serialized model
            # plus the handler source. `torch-model-archiver` is the standard
            # way a PyTorch model reaches a serving host, and TorchServe
            # unpickles the payload on load, so the archive is exactly as
            # dangerous as the checkpoint inside it.
            ".mar": ModelFormat.PICKLE,
            # NVIDIA NeMo checkpoint: a tar archive whose `model_weights.ckpt`
            # is an ordinary pickle. NeMo pins `weights_only=False` on load to
            # stay compatible with its own config objects, so PyTorch 2.6's
            # safe-load default does not protect it.
            ".nemo": ModelFormat.NEMO,
            ".safetensors": ModelFormat.SAFETENSORS,
            ".gguf": ModelFormat.GGUF,
            ".h5": ModelFormat.KERAS,
            ".hdf5": ModelFormat.KERAS,
            ".keras": ModelFormat.KERAS,
            ".onnx": ModelFormat.ONNX,
            # TensorFlow SavedModel is a directory (saved_model.pb + variables/),
            # not a self-contained file -- the .pb extension is also used by
            # arbitrary other protobufs, so scan_file further requires the
            # canonical filename before treating a .pb as a SavedModel graph.
            ".pb": ModelFormat.TF_SAVEDMODEL,
            ".npy": ModelFormat.NUMPY,
            ".npz": ModelFormat.NUMPY,
            ".joblib": ModelFormat.JOBLIB,
            # 7z archives: py7zr (the only Python reader) is LGPL-2.1, which
            # the closed-source direction cannot absorb, and no model
            # framework saves 7z, so extraction is opportunistic via a system
            # 7zz binary and the archive is always reported as unverifiable
            # without one rather than silently skipped.
            ".7z": ModelFormat.SEVENZ,
            # TFLite executes no code on load, but its parsers compute
            # tensor sizes from attacker-chosen dimensions, and
            # CVE-2026-42627 (ArmNN, 2026-05) is exactly that overflow.
            ".tflite": ModelFormat.TFLITE,
            # skops.io's pickle-free sklearn format: a zip of schema.json +
            # .npy members. Its security contract is structural (a closed
            # loader set and two loader invariants), so it is checked as
            # structure rather than by name list.
            ".skops": ModelFormat.SKOPS,
            # PMML is XML and its threat model is XXE (file-read, SSRF)
            # plus embedded code-exec constructs in transformations,
            # neither of which need the op-registration the earlier
            # "data-only" rejection assumed.
            ".pmml": ModelFormat.PMML,
        }


    def scan_file(self, file_path: Path) -> list[Finding]:
        """Scan a single model file.

        Never raises on input problems: any failure degrades to an
        MFV-SKIP-003 coverage finding instead of propagating, because one
        crashing file aborts the whole directory scan -- crash-as-evasion
        (arXiv 2508.19774), and a directory walk is how this tool is
        actually run. KeyboardInterrupt and SystemExit derive from
        BaseException, not Exception, and propagate untouched.
        """
        try:
            findings = self._scan_file_dispatch(file_path)
        except OSError:
            findings = [_skip_unverified_finding(
                file_path,
                "The file could not be read, so it was never scanned.",
                metadata={"skipped_reason": "unreadable"},
            )]
        except Exception as exc:
            logger.debug("scan_file failed on %s: %s", file_path, exc)
            findings = [_skip_unverified_finding(
                file_path,
                f"The scan failed with an unexpected error "
                f"({type(exc).__name__}), so the file was never fully analysed.",
                metadata={"skipped_reason": "scan_error",
                          "error": type(exc).__name__},
            )]

        if self.check_signatures:
            # A provenance note, never a gate: report any sibling signature or
            # attestation. Detection cannot be allowed to break the scan, so it
            # runs inside the same firewall the dispatch does.
            try:
                artifacts = find_signature_artifacts(file_path)
                findings = findings + signature_findings(file_path, artifacts)
            except Exception as exc:                         # noqa: BLE001
                logger.debug("signature check failed on %s: %s", file_path, exc)

        return findings

    def _scan_file_dispatch(self, file_path: Path) -> list[Finding]:
        """The scan_file body, wrapped by the exception firewall above."""
        # Before any format dispatch: a Git LFS pointer is a placeholder a
        # few lines long, whatever extension it wears, and the content it
        # stands in for was never fetched. Reading it further scans nothing.
        lfs_finding = _lfs_pointer_finding(file_path)
        if lfs_finding is not None:
            return [lfs_finding]

        ext = file_path.suffix.lower()
        fmt = self._format_map.get(ext, ModelFormat.UNKNOWN)

        # An extension the map doesn't name is not a reason to stop; it falls
        # through to the content sniff further down.
        if fmt == ModelFormat.TF_SAVEDMODEL:
            # .pb is also used by arbitrary other protobufs (TF frozen graphs,
            # unrelated proto blobs), so only the two canonical filenames a
            # SavedModel export writes are treated as scannable.
            #
            # `keras_metadata.pb` is not a graph: it is a SavedMetadata
            # protobuf whose node metadata carries the Keras `model_config`
            # JSON, i.e. the same layer graph the legacy `.h5` path already
            # parses. That is where a Lambda layer's marshalled Python
            # function is actually stored -- verified on four MalHug repos
            # (m0kr4n3/model3, mastersplinter/infected_test,
            # mkiani/unsafe-saved-model, opendiffusion/sentimentcheck): the
            # base64 code object is present in `keras_metadata.pb` and absent
            # from the sibling `saved_model.pb` in every one of them.
            name = file_path.name
            if name == "keras_metadata.pb":
                fmt = ModelFormat.KERAS_METADATA
            elif name != "saved_model.pb":
                return []

        if fmt == ModelFormat.NEMO:
            # Checked before the size gate below: that cap exists to bound
            # `read_bytes()`, and tarfile reads member by member straight off
            # disk without ever materialising the whole archive. Real .nemo
            # checkpoints run to hundreds of megabytes, so gating them on
            # total file size would skip the common case.
            return self._scan_tar_pickles(file_path)

        if fmt == ModelFormat.SKOPS:
            # Same reasoning as NEMO: zipfile reads members lazily, so the
            # whole-file cap below would only skip large ensembles for no
            # memory it would ever have spent.
            return self._scan_skops(file_path)

        if fmt == ModelFormat.SEVENZ:
            return self._scan_7z(file_path)

        try:
            file_size = file_path.stat().st_size
        except OSError:
            return [_skip_unverified_finding(
                file_path,
                "The file's size could not be determined, so it was never scanned.",
                metadata={"skipped_reason": "unreadable"},
            )]

        if file_size > self.MAX_SCAN_BYTES:
            # The cap exists because `read_bytes()` below would pull the whole
            # file into memory. That cost is real for a flat file, but a ZIP
            # container does not require it: `zipfile` reads lazily off disk
            # and only the pickle member is ever decompressed, already bounded
            # by MAX_ZIP_MEMBER_BYTES.
            #
            # Gating on file size rather than on parse size was a trivially
            # exploitable evasion: pad a checkpoint past the limit and the
            # scanner stops looking. Measured on MalHug, a corpus of real
            # in-the-wild malicious HuggingFace models, this accounted for
            # **every one** of this scanner's true blind spots. The clearest case is
            # `MustEr/rager_legacy`, 522MB on disk whose `archive/data.pkl` is
            # 20,203 bytes: a 20KB parse was being skipped to avoid a cost
            # that a ZIP container never charges. The classifier here calls
            # those payloads denied once it actually reads them
            # (`runpy._run_code`, `__builtin__.eval`, `__builtin__.exec`).
            if zipfile.is_zipfile(file_path):
                return self._scan_pytorch_zip(file_path)

            # Same argument for HDF5. A Keras .h5 keeps its architecture in a
            # `model_config` attribute of a few kilobytes; the rest is weights.
            # Two MalHug models, MustEr/vgg_official and vgg16_light, are 553MB
            # files carrying a real malicious Lambda layer, and both were being
            # skipped here to avoid a cost that only the weights impose. The
            # anchor is located by streaming and only a bounded window around
            # it is ever held in memory.
            if _read_file_magic(file_path, len(HDF5_SIGNATURE)) == HDF5_SIGNATURE:
                window = _read_keras_config_window(file_path)
                config = _extract_keras_model_config(window) if window else None
                if config is not None:
                    return self._keras_findings_from_config(file_path, config)

            return [Finding(
                rule_id="MFV-SKIP-001",
                message=f"Model file exceeds {self.MAX_SCAN_BYTES // 1_000_000}MB scan limit "
                        f"({file_size // 1_000_000}MB) and is not a ZIP container, so it cannot "
                        f"be scanned without reading it entirely into memory. NOT a clean "
                        f"verdict: the file was never analysed.",
                severity=Severity.LOW,
                category=Category.AI_ML,
                file_path=str(file_path),
                confidence=0.50,
                metadata={"file_size": file_size, "skipped_reason": "oversized"},
            )]

        try:
            data = file_path.read_bytes()
        except OSError:
            return [_skip_unverified_finding(
                file_path,
                "The file could not be read, so it was never scanned.",
                metadata={"skipped_reason": "unreadable"},
            )]

        # HW-142: transformers reads its RCE-bearing configuration from
        # standalone *.json files (tokenizer_config.json, chat_template.json,
        # config.json), never from a model binary. Those carry a `.json`
        # extension, which `_format_map` does not name, so they reach here as
        # UNKNOWN and would otherwise fall to the content sniff, which has no
        # JSON-config signature and returns []. Route a `.json` file whose
        # bytes parse as a JSON *object* to the config scanner. Gated on the
        # `.json` extension deliberately: a transformers config is always
        # named `*.json`, and sniffing every '{'-led blob in a tree as a model
        # config would false-positive on the unrelated JSON that litters a
        # repo (package manifests, tokenizer vocab, dataset shards).
        if file_path.suffix.lower() == ".json":
            try:
                # json.loads takes bytes directly and auto-detects the UTF
                # encoding. The catch set mirrors the other bounded JSON parses
                # in this file: deeply nested input raises RecursionError /
                # MemoryError, malformed or non-UTF input raises the decode /
                # value errors. A `.json` we cannot parse as an object is not
                # evidence of a config, so it degrades to "not scanned" here.
                parsed = json.loads(data)
            except (UnicodeDecodeError, json.JSONDecodeError, RecursionError,
                    ValueError, MemoryError):
                parsed = None
            if isinstance(parsed, dict):
                return self._scan_json_config(file_path, parsed)

        if fmt == ModelFormat.UNKNOWN:
            # Every extension the map doesn't name lands here, ambiguous
            # (`.bin`, `.zip`) and unrecognized (`.dat`, or no suffix at all)
            # alike. A path handed to scan_file was named deliberately, and
            # the extension is the one part of a file an attacker renames for
            # free: `danger.dat` in the canary repo mcpotato/42-eicar-street
            # is a `builtins.eval` pickle that this method used to return []
            # for, while the identical bytes named `.pkl` reported CRITICAL.
            # scan_directory keeps its extension pre-filter, so deciding by
            # content here costs a directory walk nothing.
            #
            # `.bin` names the single most common PyTorch weight file
            # on HuggingFace (`pytorch_model.bin`), but it is also used for
            # arbitrary unrelated binaries, so it can't simply be mapped to
            # PICKLE -- that would run the pickle parser over every stray blob
            # and report parse failures as findings. Let the content decide,
            # the same way torch.load itself does. Anything `_sniff_format`
            # can't positively identify is skipped, since an unidentified file
            # with an unclaimed extension isn't evidence of a model at all.
            fmt = self._sniff_format(data) or ModelFormat.UNKNOWN
            if fmt == ModelFormat.UNKNOWN:
                return []

        # `.pth` is an overloaded extension: besides a PyTorch checkpoint it is
        # also Python's own path-configuration file, a plain-text list of
        # directories (plus optional `import ...` lines) that setuptools,
        # virtualenv, and editable installs drop into site-packages. Those are
        # not pickles, so opcode analysis raises and MFV-PICKLE-002's raw-byte
        # fallback matches the `import`/`__import__` text in them -- a pure
        # false positive at HIGH severity. Measured across scan-targets/: 9 of
        # 9 `.pth` files were path-configuration files and none was a
        # checkpoint. Require the real thing's magic bytes: torch.save()
        # produces either a ZIP (default since PyTorch 1.6) or a protocol-2+
        # pickle, which always opens with the PROTO opcode `\x80`. Same
        # reasoning as the `saved_model.pb` guard above -- an extension that
        # other tooling also uses is not on its own evidence of the format.
        if file_path.suffix.lower() == ".pth" and not (
            data[:1] == b"\x80" or data[:4] == b"PK\x03\x04"
        ):
            return []

        # Modern PyTorch checkpoints (.pt/.pth/.ckpt) are ZIP archives that
        # wrap the real pickle (archive/data.pkl). torch.save() has produced
        # these by default since PyTorch 1.6. Scan inside the archive; fall
        # back to flat-pickle scanning for the legacy non-zip format. This is
        # tracked separately from `fmt` because a zip-wrapped checkpoint
        # already gets its own (member-level) parse-failure handling inside
        # _scan_pytorch_zip -- the extension-confusion check below only
        # applies to the flat-pickle path.
        is_zip_pickle = fmt == ModelFormat.PICKLE and zipfile.is_zipfile(file_path)

        findings = self._run_scanner_for_format(fmt, file_path, data, is_zip_pickle=is_zip_pickle)

        # Extension-spoofing check (issue #152): a format-specific parse
        # failure alone doesn't distinguish "corrupted file" from "wrong
        # extension entirely" -- a raw pickle saved as .safetensors fails
        # SafeTensors' JSON-header parse and gets reported as merely
        # "corrupted", but several real loaders (torch.load, safetensors
        # readers, GGUF loaders) sniff magic bytes rather than trusting the
        # extension, so that file still executes when actually loaded
        # downstream. Only fires when the extension-directed scan hit an
        # identifiable structural-parse-failure signal AND the raw bytes
        # match a *different* known format -- never invents a mismatch on a
        # file that's simply corrupted (see _sniff_format/_extension_parse_failed).
        if not is_zip_pickle and self._extension_parse_failed(fmt, file_path, data, findings):
            sniffed = self._sniff_format(data)
            if sniffed is not None and sniffed != fmt:
                findings = list(findings)
                findings.append(self._confuse_finding(file_path, fmt, sniffed))
                findings.extend(self._run_scanner_for_format(sniffed, file_path, data))

        exec_hits = _find_embedded_executables(data)
        if exec_hits:
            findings.append(self._executable_finding(file_path, exec_hits))

        return findings

    def _executable_finding(
        self, file_path: Path, hits: list[str], member: str | None = None,
    ) -> Finding:
        label = f"[{member}] " if member else ""
        return Finding(
            rule_id="MFV-EXEC-001",
            message=f"{label}Model file contains a loadable binary: "
                    f"{'; '.join(hits[:3])}. No serialization format writes "
                    f"one; this is a payload appended to or embedded in the "
                    f"file (polyglot/steganography), not model data.",
            severity=Severity.HIGH,
            category=Category.AI_ML,
            file_path=str(file_path),
            confidence=0.8,
            cwe_ids=[506],
            metadata={"embedded": hits[:10], **({"zip_member": member} if member else {})},
        )

    def _run_scanner_for_format(
        self, fmt: ModelFormat, file_path: Path, data: bytes, is_zip_pickle: bool = False,
    ) -> list[Finding]:
        """Dispatch to the format-specific `_scan_*` method for `fmt`.

        Shared between the normal extension-directed dispatch in `scan_file`
        and the sniffed-format re-scan triggered by a detected extension
        mismatch, so the two paths can't drift on how each format is invoked.
        """
        if fmt == ModelFormat.PICKLE:
            # `is_zip_pickle` is only ever True when this is the primary
            # extension-directed dispatch for a .pt/.pth/.pkl/... file that
            # zipfile.is_zipfile() already confirmed is a real zip; the
            # sniffed-format re-scan path (is_zip_pickle defaults False)
            # always means "matched via the flat pickle-opcode fallback in
            # _sniff_format", so it always goes to _scan_pickle directly.
            if is_zip_pickle:
                return self._scan_pytorch_zip(file_path)
            return self._scan_pickle(file_path, data)
        elif fmt == ModelFormat.SAFETENSORS:
            return self._scan_safetensors(file_path, data)
        elif fmt == ModelFormat.GGUF:
            return self._scan_gguf(file_path, data)
        elif fmt == ModelFormat.KERAS:
            return self._scan_keras(file_path)
        elif fmt == ModelFormat.PYTORCH_ZIP:
            return self._scan_pytorch_zip(file_path)
        elif fmt == ModelFormat.ONNX:
            return self._scan_onnx(file_path, data)
        elif fmt == ModelFormat.TF_SAVEDMODEL:
            return self._scan_tf_savedmodel(file_path, data)
        elif fmt == ModelFormat.KERAS_METADATA:
            return self._scan_keras_metadata(file_path, data)
        elif fmt == ModelFormat.NUMPY:
            return self._scan_numpy(file_path, data)
        elif fmt == ModelFormat.JOBLIB:
            return self._scan_joblib(file_path, data)
        elif fmt == ModelFormat.SKOPS:
            return self._scan_skops(file_path)
        elif fmt == ModelFormat.TFLITE:
            return self._scan_tflite(file_path, data)
        elif fmt == ModelFormat.SEVENZ:
            return self._scan_7z(file_path)
        elif fmt == ModelFormat.PMML:
            return self._scan_pmml(file_path, data)

        return []

    def _extension_parse_failed(
        self, fmt: ModelFormat, file_path: Path, data: bytes, findings: list[Finding],
    ) -> bool:
        """True if the extension-directed scan of `data` could not
        structurally parse it as `fmt` -- the trigger condition for sniffing
        the bytes against the other known `ModelFormat` signatures.

        Only wired for formats with a reliable "this isn't even structurally
        valid X" signal:
          - SAFETENSORS / GGUF: the scan methods return early with a single
            "corrupted"-flavored finding (MFV-ST-00{1,2,3,4} / MFV-GGUF-00{1,4})
            when the header/magic/JSON doesn't parse -- checking the return
            for that marker is cheaper than re-deriving the same parse.
          - PICKLE (flat, non-zip only): re-run the same opcode walk
            `_scan_pickle` already attempts; an exception there is the
            "not a valid pickle stream" signal.
          - KERAS: neither the HDF5 signature nor the zip-based .keras
            container matched.
          - NUMPY: neither the .npy header nor (for .npz) the zip container
            parsed.

        Deliberately NOT wired for ONNX/TF_SAVEDMODEL or JOBLIB: their scan
        methods degrade to "found nothing" on non-matching bytes rather than
        raising or emitting a dedicated corrupted finding (ONNX/TF's
        schema-less protobuf walker just stops early; joblib always falls
        through to a plain pickle scan of whatever bytes it's given), so
        there's no signal to distinguish "wrong format entirely" from "valid
        but uninteresting file" without risking false MFV-CONFUSE-001s on
        legitimate, sparse files.
        """
        if fmt == ModelFormat.SAFETENSORS:
            return bool(findings) and findings[0].rule_id in {
                "MFV-ST-001", "MFV-ST-002", "MFV-ST-003", "MFV-ST-004",
            }
        if fmt == ModelFormat.GGUF:
            return bool(findings) and findings[0].rule_id in {"MFV-GGUF-001", "MFV-GGUF-004"}
        if fmt == ModelFormat.PICKLE:
            return not self._is_parseable_pickle(data)
        if fmt == ModelFormat.KERAS:
            return data[:8] != HDF5_SIGNATURE and not zipfile.is_zipfile(file_path)
        if fmt == ModelFormat.NUMPY:
            if file_path.suffix.lower() == ".npz":
                return not zipfile.is_zipfile(file_path)
            return self._parse_npy_header(data) is None
        return False

    def _confuse_finding(
        self, file_path: Path, extension_fmt: ModelFormat, sniffed_fmt: ModelFormat,
    ) -> Finding:
        """Build the MFV-CONFUSE-001 finding for a detected extension/content
        mismatch -- same bypass class fickling's PyTorch-polyglot detection
        exists to catch: several real loaders (torch.load, safetensors
        readers, GGUF loaders) sniff magic bytes rather than trusting the
        extension, so a file MFV would otherwise wave through as "corrupted,
        skip" can still execute when actually loaded downstream.
        """
        return Finding(
            rule_id="MFV-CONFUSE-001",
            message=(
                f"File has a '{file_path.suffix}' extension (implying {extension_fmt.value} "
                f"format) but its content matches {sniffed_fmt.value} magic bytes/structure "
                f"instead, and failed to parse as {extension_fmt.value}. Possible "
                f"extension-spoofing to evade format-specific scanning -- the real content "
                f"was re-scanned as {sniffed_fmt.value} below."
            ),
            severity=Severity.HIGH,
            category=Category.AI_ML,
            file_path=str(file_path),
            confidence=0.8,
            metadata={
                "extension_format": extension_fmt.value,
                "sniffed_format": sniffed_fmt.value,
                "extension": file_path.suffix,
            },
        )

    def _is_parseable_pickle(self, data: bytes) -> bool:
        """True if `data` walks as a complete, well-formed pickle opcode
        stream -- shared by `_extension_parse_failed` (is *this* file's
        content a valid pickle?) and `_sniff_format` (does some *other*
        file's content happen to be a valid pickle?) so both call sites
        stay on the identical definition of "parseable" as `_scan_pickle`
        itself uses, rather than each re-deriving it.
        """
        try:
            _resolve_pickle_globals(data)
        except Exception:
            return False
        return True

    def _sniff_format(self, data: bytes) -> ModelFormat | None:
        """Identify which (if any) of the known `ModelFormat` container
        signatures `data` actually matches, independent of any file
        extension -- the magic-byte/structural sniff underlying the
        extension-confusion check in `scan_file`.

        Checked most-specific-first so a format with a strong, cheap
        signature (a fixed magic number) is never shadowed by a looser
        heuristic (the SafeTensors header-size-prefix check, or the
        pickle-opcode-parseability fallback, which will happily "match"
        almost any short byte string that isn't already claimed by
        something more specific). Returns None if nothing matches -- callers
        must treat that as "genuinely unidentifiable", not a positive signal
        for any particular format.
        """
        if len(data) >= 24 and data[:4] == GGUF_MAGIC:
            return ModelFormat.GGUF

        if data[:8] == HDF5_SIGNATURE:
            return ModelFormat.KERAS

        if data[:4] == b"PK\x03\x04":
            # ZIP-based containers (PyTorch checkpoint / .keras / .npz) all
            # share the same local-file-header magic -- only the archive's
            # member names actually distinguish them.
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    names = zf.namelist()
            except (zipfile.BadZipFile, OSError):
                return None
            if any(n.rsplit("/", 1)[-1] == "data.pkl" or n.endswith((".pkl", ".pickle")) for n in names):
                return ModelFormat.PYTORCH_ZIP
            if "config.json" in names:
                return ModelFormat.KERAS
            if names and all(n.endswith(".npy") for n in names):
                return ModelFormat.NUMPY
            # Ambiguous zip container with none of the above markers --
            # most model-file zip confusion in practice is a mislabeled
            # PyTorch checkpoint, so that's the more useful default guess,
            # but re-scanning it will itself find nothing if this guess is
            # wrong (no false "denied global" style finding gets fabricated).
            return ModelFormat.PYTORCH_ZIP

        if data[:6] == self._NPY_MAGIC:
            return ModelFormat.NUMPY

        # SafeTensors' own header-size-prefix heuristic: first 8 bytes are a
        # little-endian u64 header length, followed by that many bytes of
        # valid JSON -- mirrors the validation _scan_safetensors performs,
        # just checked speculatively here instead of raising findings.
        if len(data) >= 8:
            header_size = struct.unpack("<Q", data[:8])[0]
            if 0 < header_size <= 100_000_000 and 8 + header_size <= len(data):
                try:
                    parsed = json.loads(data[8:8 + header_size].decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError, RecursionError,
                        ValueError, MemoryError):
                    parsed = None
                if isinstance(parsed, dict):
                    return ModelFormat.SAFETENSORS

        # joblib's compressed variant: a zlib stream that inflates to a
        # parseable pickle. Checked before the bare pickle-opcode fallback
        # since a compressed joblib file's raw bytes won't themselves parse
        # as pickle opcodes.
        if len(data) >= 2 and data[0] == 0x78 and data[1] in (0x01, 0x5E, 0x9C, 0xDA):
            try:
                decompressed = self._decompress_zlib_capped(data, self.MAX_ZIP_MEMBER_BYTES)
            except zlib.error:
                decompressed = None
            if decompressed is not None and self._is_parseable_pickle(decompressed):
                return ModelFormat.JOBLIB

        # Pickle-opcode-parseability as the pickle "signature" -- reuses the
        # same opcode walk _scan_pickle already performs rather than a
        # second parser; if pickletools can walk the full stream without
        # raising, that's as strong a positive signal as this format gets
        # (pickle has no fixed magic number of its own).
        if self._is_parseable_pickle(data):
            return ModelFormat.PICKLE

        # ONNX (ModelProto) / TF SavedModel (GraphDef) are both schema-less
        # protobuf at this point -- neither has a magic number, so this
        # falls back to a structural heuristic on the already-available
        # top-level field walk: GraphDef repeats its `node` field (field 1,
        # length-delimited) once per graph node, often dozens/hundreds of
        # times, while ModelProto's `graph` field (field 7) appears exactly
        # once. Best-effort, same spirit as _extract_protobuf_strings's own
        # "walk it schema-lessly since we don't have the .proto" approach.
        try:
            fields = list(_iter_protobuf_fields(data))
        except RecursionError:
            fields = []
        if fields:
            node_like = sum(1 for fn, wt, _ in fields if fn == 1 and wt == 2)
            if node_like > 5:
                return ModelFormat.TF_SAVEDMODEL
            if any(fn == 7 and wt == 2 for fn, wt, _ in fields):
                return ModelFormat.ONNX

        return None

    def discover_files(
        self, root: Path, exclude: list[str] | None = None
    ) -> list[Path]:
        """Return every scannable model file under `root`, in walk order.

        Split out from scan_directory so the caller can see the file list
        before scanning: the cli needs it to compose the content cache (serve
        cache hits, scan only misses) and to drive a single progress total
        across several targets. The discovery rules are exactly what
        scan_directory used to inline.
        """
        root_resolved = root.resolve()
        # An earlier version matched skip names against substrings of the
        # whole path and excluded only `.git` and `__pycache__`, so it walked
        # the target's installed dependencies. Match on path *components*
        # instead, so a directory whose name merely contains "bundle" is kept.
        skip_dirs = SKIP_DIR_NAMES
        files: list[Path] = []
        # Directory discovery must cover the same extension set scan_file
        # accepts, including the ambiguous ones it resolves by content sniff --
        # otherwise a pytorch_model.bin is scannable when named directly but
        # invisible to a directory scan, which is how the tool is actually run.
        for ext in (*self._format_map, *self._AMBIGUOUS_EXTENSIONS, *self._CONFIG_EXTENSIONS):
            for f in root.rglob(f"*{ext}"):
                # rglob matches directories too. Model caches routinely name a
                # directory after the file it holds (`<repo>--model.joblib/`),
                # so without this every such directory was handed to scan_file,
                # failed to read, and produced a spurious MFV-SKIP-003.
                if not f.is_file():
                    continue
                try:
                    parts = f.relative_to(root).parts
                except ValueError:
                    parts = f.parts
                if any(p in skip_dirs for p in parts):
                    continue
                # User excludes are matched against the path as walked (its
                # path relative to the scan root) and against every single
                # component, so a directory-name pattern prunes the subtree.
                if exclude and _path_excluded(f, root, exclude):
                    continue
                # Skip symlinks whose target escapes the scan root, so a
                # hostile repo can't get files outside the target read as
                # model-file findings (CWE-59/22).
                if f.is_symlink():
                    try:
                        resolved = f.resolve()
                        if resolved != root_resolved and root_resolved not in resolved.parents:
                            continue
                    except OSError:
                        continue
                files.append(f)
        return files

    def scan_directory(
        self,
        root: Path,
        *,
        jobs: int = 1,
        exclude: list[str] | None = None,
        progress: ProgressCallback | None = None,
    ) -> list[Finding]:
        """Recursively scan a directory for model files.

        `jobs` > 1 fans the discovered files out across a process pool;
        `jobs` <= 1 stays fully sequential (the default, so nothing changes for
        existing callers). `exclude` drops files matching any glob pattern.
        `progress`, if given, is called once per completed file with
        (done, total, path) so a caller can report progress without the
        scanner ever writing to a stream itself.
        """
        files = self.discover_files(root, exclude)
        results = self._scan_paths(files, jobs=jobs, progress=progress)
        findings: list[Finding] = []
        for _path, file_findings in results:
            findings.extend(file_findings)
        return findings

    def _scan_paths(
        self,
        files: list[Path],
        *,
        jobs: int = 1,
        progress: ProgressCallback | None = None,
    ) -> list[tuple[Path, list[Finding]]]:
        """Scan an explicit list of files, sequentially or across a pool.

        Returns per-file results (path, findings) so the caller can attribute
        each file's findings, which the content cache needs to store a miss
        under that file's hash. Ordering of the returned list does not affect
        the report (it re-sorts), but no file is dropped or double-counted.
        """
        total = len(files)
        if jobs is None or jobs <= 1:
            results: list[tuple[Path, list[Finding]]] = []
            for index, f in enumerate(files, 1):
                file_findings = self.scan_file(f)
                if progress is not None:
                    progress(index, total, f)
                results.append((f, file_findings))
            return results
        # Parallel path. The worker config must reach every worker process
        # (closures are not picklable), so it is carried in a small frozen
        # dataclass and rebuilt into a configured scanner once per worker by
        # the pool initializer. max_size and check_signatures are the only
        # state a worker's scan_file depends on.
        config = _ScanConfig(
            max_scan_bytes=self.MAX_SCAN_BYTES,
            check_signatures=self.check_signatures,
        )
        return _scan_paths_parallel(files, config, jobs, progress)

    # ── Pickle scanning ───────────────────────────────────────────

    def _scan_pickle(self, file_path: Path, data: bytes) -> list[Finding]:
        """Scan a pickle file by resolving every GLOBAL/STACK_GLOBAL reference
        to its `module.name` callable and classifying each against an
        allow/deny list, instead of flagging the mere presence of REDUCE-style
        opcodes (which appear in essentially every real pickle -- an
        ``OrderedDict``, a tensor rebuild, ... -- and previously gave a benign
        state_dict the same CRITICAL verdict as actual malware).
        """
        findings: list[Finding] = []

        if _pickle_stream_truncated(data):
            # Reported before anything else, and independently of what the
            # walk below does or does not resolve: a stream that never
            # terminated was not read to the end, whether or not the part
            # that did arrive happened to contain something.
            findings.append(_skip_unverified_finding(
                file_path,
                "Pickle stream ends before its STOP opcode, so the file is "
                "truncated and anything past the cut was never read.",
                {"skipped_reason": "pickle_truncated", "file_size": len(data)},
            ))

        embedded = _embedded_pickle_denied_globals(data)
        if embedded:
            findings.append(Finding(
                rule_id="MFV-PICKLE-008",
                message=f"Pickle file carries a second pickle stream as a bytes "
                        f"literal, and that inner stream references: "
                        f"{', '.join(embedded)}. Anything handed those bytes "
                        f"deserializes them, so the outer callable does not need "
                        f"to be dangerous itself.",
                severity=Severity.CRITICAL,
                category=Category.DESERIALIZATION,
                file_path=str(file_path),
                confidence=0.9,
                cwe_ids=[502],
                metadata={"nested_globals": embedded},
            ))

        try:
            globals_found, resolved_calls, memo_profile = _resolve_pickle_globals(data)
        except Exception as e:
            # Corrupted or non-standard pickle stream -- can't resolve
            # callables, so fall back to a weaker raw-byte signal rather than
            # staying silent. Kept out of the normal (parseable) path because
            # a substring can't distinguish a real dangerous call from an
            # unrelated string, which is what made the old always-on version
            # of this check false-positive-prone.
            logger.debug("Pickle opcode analysis failed for %s: %s", file_path, e)
            raw_hits = [sig.decode(errors="replace") for sig in PICKLE_DANGER_SIGNATURES if sig in data]
            if raw_hits:
                findings.append(Finding(
                    rule_id="MFV-PICKLE-002",
                    message=f"Pickle stream could not be parsed as valid opcodes, but contains "
                            f"suspicious byte patterns: {', '.join(raw_hits[:5])}. Possibly corrupted "
                            f"or an obfuscated/malformed backdoored file.",
                    severity=Severity.HIGH,
                    category=Category.DESERIALIZATION,
                    file_path=str(file_path),
                    confidence=0.5,
                    cwe_ids=[502],
                    metadata={"signatures": raw_hits, "file_size": len(data), "unparseable": True},
                ))
            else:
                # Silence is correct here, and flagging it was measured to be
                # wrong. "0 globals resolved + no raw signature" is the
                # profile of every *harmless* stream: pickle.dumps(42), and
                # the raw tensor-storage blobs inside every torch checkpoint
                # (the two-byte opener sniff matches float data constantly;
                # tiny-random-t5/distilbert in the clean cache fired this way).
                # Pickle execution requires a GLOBAL/STACK_GLOBAL the loader
                # can reach, and the loader reaches it only through opcodes
                # this same walk reads: a stream with no resolvable global
                # cannot carry a working payload, so there is no evasion to
                # report. This is loader parity, not a blind spot.
                pass
            return findings

        if memo_profile.walk_budget_exceeded:
            # The walk terminated on its stack budget, not on a STOP opcode:
            # whatever sat past that point was never read, and saying nothing
            # would claim a complete analysis of a stream that was not one.
            findings.append(_skip_unverified_finding(
                file_path,
                "The pickle walk exceeded its simulated-stack budget, so the "
                "stream past that point was never read.",
                {"skipped_reason": "pickle_walk_budget", "file_size": len(data)},
            ))

        denied = sorted({g for g in globals_found if _classify_pickle_global(g) == "denied"})
        unknown = sorted({g for g in globals_found if _classify_pickle_global(g) == "unknown"})
        allowed = sorted({g for g in globals_found if _classify_pickle_global(g) == "allowed"})

        # Pair each classified ref with the first resolved literal-argument
        # call found for it (if any) -- concrete evidence like
        # os.system('curl ... | sh') is far stronger triage signal than the
        # bare callable name, but a ref that was only ever *referenced*
        # (never reduced/called, or reduced with a non-literal/dynamic
        # argument) still falls back to reporting the callable alone,
        # exactly as before. Multiple distinct literal-arg calls to the same
        # ref are rare in practice; the first one found is kept so the
        # message doesn't balloon if it happens.
        # First call per ref wins, except that a fully-resolved call always
        # beats a partially-resolved one for the same callable -- concrete
        # arguments are strictly better evidence than "this literal appeared
        # somewhere in the argument list".
        calls_by_ref: dict[str, PickleResolvedCall] = {}
        for call in resolved_calls:
            existing = calls_by_ref.get(call.ref)
            if existing is None or (existing.partial_texts and not call.partial_texts):
                calls_by_ref[call.ref] = call

        # An allowlisted *name* does not allowlist its *arguments*.
        #
        # ShadowPickle's "Overwritten Module" variant (arXiv:2607.17503),
        # reported at 63% evasion across ten scanners and 0% detection by both
        # picklescan and ModelScan, never references a dangerous callable at
        # all. It calls `collections.OrderedDict` -- allowlisted here, by
        # picklescan, by ModelScan, and by PyTorch's own weights-only
        # unpickler -- and passes it a *string*. A trojaned `collections`
        # already resident in the victim environment (installed via a `.pth`
        # file that rebinds `sys.modules["collections"]` at interpreter start)
        # executes it. No `os`, no `posix`, no `exec` anywhere in the stream,
        # so every name-based check passes by construction.
        #
        # The generalisation is not another list: it is that the allow list
        # answers "is this callable safe to *name*", which is a different
        # question from "is this callable being handed something no legitimate
        # model would hand it". The same argument-evidence triage already used
        # on the unknown bucket runs here too. `OrderedDict` reconstructing a
        # state dict never receives a shell command, a URL or a filesystem
        # path, whatever `collections` happens to resolve to at load time.
        allowed_evidence: dict[str, tuple[Severity, str]] = {}
        for ref in allowed:
            if _is_ml_constructor_allowed(ref):
                # Config constructors take strings constantly (Pipeline step
                # names, TrainingArguments' output_dir, torch.device('cuda')).
                # The anomalous-string premise is false for the whole class,
                # measured: 24 of 215 benign models tripped it the day these
                # refs became allowed.
                continue
            call = calls_by_ref.get(ref)
            shape_reason = _allowed_call_has_anomalous_string(ref, call)
            if shape_reason is not None:
                allowed_evidence[ref] = (Severity.MEDIUM, shape_reason)
                continue
            verdict = _triage_unknown_pickle_call(call)
            if verdict is not None:
                allowed_evidence[ref] = verdict

        if allowed_evidence:
            details = "; ".join(
                f"{calls_by_ref[ref].format()} [{reason}]"
                for ref, (_sev, reason) in sorted(allowed_evidence.items())
            )
            worst = min(
                (sev for sev, _ in allowed_evidence.values()),
                key=lambda s: _SEVERITY_ORDER.get(s, len(_SEVERITY_ORDER)),
            )
            findings.append(Finding(
                rule_id="MFV-PICKLE-006",
                message=f"Pickle file invokes a known-safe callable with an argument no "
                        f"legitimate model would supply: {details}. The callable's name is on "
                        f"the allow list, which is exactly what makes this worth reporting -- "
                        f"an attacker who can influence the loading environment can repurpose a "
                        f"whitelisted constructor without naming anything dangerous.",
                severity=worst,
                category=Category.DESERIALIZATION,
                file_path=str(file_path),
                confidence=0.65,
                cwe_ids=[502],
                metadata={
                    "allowed_globals": sorted(allowed_evidence),
                    "file_size": len(data),
                    "triage": {
                        ref: {
                            "severity": sev.value,
                            "reason": reason,
                            "call": calls_by_ref[ref].format(),
                        }
                        for ref, (sev, reason) in sorted(allowed_evidence.items())
                    },
                },
            ))

        if denied:
            denied_display = [
                calls_by_ref[ref].format() if ref in calls_by_ref else ref
                for ref in denied
            ]
            findings.append(Finding(
                rule_id="MFV-PICKLE-001",
                message=f"Pickle file references unsafe callable(s) that grant code/command "
                        f"execution on load: {', '.join(denied_display)}. These opcodes can execute "
                        f"arbitrary code during deserialization.",
                severity=Severity.CRITICAL,
                category=Category.DESERIALIZATION,
                file_path=str(file_path),
                confidence=0.95,
                cwe_ids=[502],
                metadata={
                    "denied_globals": denied,
                    "file_size": len(data),
                    "denied_calls": {ref: calls_by_ref[ref].format() for ref in denied if ref in calls_by_ref},
                },
            ))

        if unknown and not denied:
            unknown_calls = {ref: calls_by_ref[ref].format() for ref in unknown if ref in calls_by_ref}

            # Re-triage the unknown bucket against that argument-level
            # evidence before falling through to the INFO catch-all. Without
            # this, every documented picklescan bypass gadget -- none of
            # which is on any deny list, by construction -- was reported at
            # INFO as "likely a legitimate custom class".
            verdicts: dict[str, tuple[Severity, str]] = {}
            for ref in unknown:
                verdict = _triage_unknown_pickle_call(calls_by_ref.get(ref))
                if verdict is not None:
                    verdicts[ref] = verdict

            # One finding per tier rather than per global, so a file with a
            # dozen escalated globals doesn't produce a dozen findings.
            # Iterating the tier table keeps emission order deterministic.
            for severity, (confidence, lead) in _PICKLE_UNKNOWN_TIERS.items():
                refs = sorted(ref for ref, (sev, _reason) in verdicts.items() if sev == severity)
                if not refs:
                    continue
                details = "; ".join(
                    f"{unknown_calls.get(ref, ref)} [{verdicts[ref][1]}]" for ref in refs
                )
                findings.append(Finding(
                    rule_id="MFV-PICKLE-005",
                    message=f"{lead}: {details}. This verdict comes from the resolved call "
                            f"arguments, not from the callable's name -- these globals are on "
                            f"neither the known-safe nor the known-dangerous list, which is "
                            f"precisely where pickle scanner bypass gadgets live.",
                    severity=severity,
                    category=Category.DESERIALIZATION,
                    file_path=str(file_path),
                    confidence=confidence,
                    cwe_ids=[502],
                    metadata={
                        "unknown_globals": refs,
                        "file_size": len(data),
                        "triage": {
                            ref: {
                                "severity": severity.value,
                                "confidence": confidence,
                                "reason": verdicts[ref][1],
                                "call": unknown_calls.get(ref),
                            }
                            for ref in refs
                        },
                    },
                ))

            # Whatever survived re-triage keeps the original INFO treatment.
            residual = [ref for ref in unknown if ref not in verdicts]
            residual_calls = {ref: expr for ref, expr in unknown_calls.items() if ref not in verdicts}

            if residual:
                findings.append(Finding(
                    rule_id="MFV-PICKLE-004",
                    message=f"Pickle file references unrecognized global(s) not on the known-safe "
                            f"list: {', '.join(residual[:10])}. Likely a legitimate custom class or "
                            f"model type, but not verifiable by static analysis -- review the source "
                            f"if this file's origin is untrusted.",
                    severity=Severity.INFO,
                    category=Category.DESERIALIZATION,
                    file_path=str(file_path),
                    confidence=0.3,
                    cwe_ids=[502],
                    metadata={
                        "unknown_globals": residual,
                        "file_size": len(data),
                        "unknown_calls": residual_calls,
                    },
                ))

        if memo_profile.out_of_band:
            # Reported independently of what the globals walk found: this is
            # evidence about how the file was *assembled*, not about what any
            # spliced code does, so it still fires when the payload uses only
            # allow-listed callables and stays silent on a file that merely
            # references something unrecognized.
            oob = memo_profile.out_of_band
            shown = ", ".join(str(i) for i in oob[:5])
            if len(oob) > 5:
                shown += ", ..."
            findings.append(Finding(
                rule_id="MFV-PICKLE-007",
                message=f"Pickle stream writes {len(oob)} memo slot(s) ({shown}) separated by a "
                        f"large gap from the rest of the {memo_profile.slots}-slot memo it uses. "
                        f"Python's picklers number memo slots consecutively, so a single pickle's "
                        f"indices always form one unbroken run -- a hole this size means the stream "
                        f"was not written by one ordinary pickle.dump. It is the signature of a "
                        f"payload spliced into an existing model file with its memo writes offset "
                        f"to avoid colliding with the host's.",
                severity=Severity.HIGH,
                category=Category.DESERIALIZATION,
                file_path=str(file_path),
                confidence=0.75,
                cwe_ids=[502],
                metadata={
                    "out_of_band_memo_indices": list(oob[:20]),
                    "out_of_band_count": len(oob),
                    "memo_slots": memo_profile.slots,
                    "max_memo_index": memo_profile.max_index,
                    "file_size": len(data),
                },
            ))

        return findings

    # ── SafeTensors scanning ──────────────────────────────────────

    def _scan_safetensors(self, file_path: Path, data: bytes) -> list[Finding]:
        """Validate SafeTensors format integrity."""
        findings: list[Finding] = []

        if len(data) < 8:
            return [Finding(
                rule_id="MFV-ST-001",
                message="SafeTensors file is too small (< 8 bytes). Corrupted or invalid.",
                severity=Severity.HIGH,
                category=Category.AI_ML,
                file_path=str(file_path),
                confidence=0.95,
            )]

        # Read header size (first 8 bytes, little-endian u64)
        header_size = struct.unpack("<Q", data[:8])[0]

        if header_size > 100_000_000:
            return [Finding(
                rule_id="MFV-ST-002",
                message=f"SafeTensors header size ({header_size}) exceeds limit. Possible DoS or malformed file.",
                severity=Severity.HIGH,
                category=Category.AI_ML,
                file_path=str(file_path),
                confidence=0.90,
            )]

        if 8 + header_size > len(data):
            return [Finding(
                rule_id="MFV-ST-003",
                message="SafeTensors header extends beyond file boundary. Corrupted or truncated.",
                severity=Severity.HIGH,
                category=Category.AI_ML,
                file_path=str(file_path),
                confidence=0.95,
            )]

        # Parse JSON header
        try:
            header = json.loads(data[8:8 + header_size].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError,
                ValueError, MemoryError):
            # Deeply nested header JSON raises RecursionError/MemoryError
            # rather than JSONDecodeError; same catch set as the skops path.
            return [Finding(
                rule_id="MFV-ST-004",
                message="SafeTensors header is not valid JSON. Corrupted or maliciously crafted.",
                severity=Severity.HIGH,
                category=Category.AI_ML,
                file_path=str(file_path),
                confidence=0.95,
            )]

        layout_problems = _check_safetensors_layout(
            header, len(data) - 8 - int(header_size))
        if layout_problems:
            findings.append(Finding(
                rule_id="MFV-ST-006",
                message=f"SafeTensors layout arithmetic is inconsistent with the file: "
                        f"{'; '.join(layout_problems[:5])}. Loaders allocate and memcpy "
                        f"from these offsets; a file whose numbers do not fit it is a "
                        f"parser exploit, not a corrupt model.",
                severity=Severity.HIGH,
                category=Category.AI_ML,
                file_path=str(file_path),
                confidence=0.8,
                cwe_ids=[787],
                metadata={"layout_problems": layout_problems[:20]},
            ))

        # Validate metadata, flagging suspicious keys
        suspicious_keys: list[str] = []
        metadata = header.get("__metadata__", {})
        if isinstance(metadata, dict):
            for key in metadata:
                if not isinstance(key, str):
                    continue
                lowered = key.lower()
                if (
                    "__reduce__" in lowered
                    or "__builtins__" in lowered
                    or _ST_METADATA_DANGEROUS_KEY_RE.search(lowered)
                ):
                    suspicious_keys.append(key)

        if suspicious_keys:
            findings.append(Finding(
                rule_id="MFV-ST-005",
                message=f"SafeTensors metadata contains suspicious keys: {', '.join(suspicious_keys)}. "
                        f"These could attempt code execution on load.",
                severity=Severity.CRITICAL,
                category=Category.AI_ML,
                file_path=str(file_path),
                confidence=0.90,
                metadata={"suspicious_keys": suspicious_keys},
            ))

        return findings

    # ── GGUF scanning ─────────────────────────────────────────────

    def _scan_gguf(self, file_path: Path, data: bytes) -> list[Finding]:
        """Inspect GGUF metadata for embedded code or suspicious content.

        Parses the actual header + KV metadata structure (magic, version,
        tensor/kv counts, typed entries) instead of decoding the raw file
        and substring-matching across everything -- the old approach matched
        tensor names and binary weight data as readily as real metadata
        values, and couldn't say *which* field a hit came from.
        """
        findings: list[Finding] = []

        if len(data) < 8 or data[:4] != GGUF_MAGIC:
            if data[:4] in GGML_MAGICS:
                # A known format we do not parse, so the honest verdict is
                # non-coverage rather than a claim about the content.
                return [_skip_unverified_finding(
                    file_path,
                    "The file is in GGML format, GGUF's predecessor, which this "
                    "scanner does not parse.",
                    metadata={"skipped_reason": "ggml_format"},
                )]
            return [Finding(
                rule_id="MFV-GGUF-001",
                message="GGUF magic number invalid or missing. Corrupted or non-GGUF file.",
                severity=Severity.HIGH,
                category=Category.AI_ML,
                file_path=str(file_path),
                confidence=0.95,
            )]

        layout_problems = _check_gguf_layout(data)
        if layout_problems:
            findings.append(Finding(
                rule_id="MFV-GGUF-005",
                message=f"GGUF container arithmetic is inconsistent with the file: "
                        f"{'; '.join(layout_problems[:5])}. llama.cpp allocates and "
                        f"reads from these counts and dimensions in u64; a file that "
                        f"only fits by wrapping is a parser exploit (CVE-2025-53630, "
                        f"CVE-2026-27940, CVE-2026-33298), not a corrupt model.",
                severity=Severity.HIGH,
                category=Category.AI_ML,
                file_path=str(file_path),
                confidence=0.8,
                cwe_ids=[787, 190],
                metadata={"layout_problems": layout_problems[:20]},
            ))

        try:
            kv, kv_window_truncated = _parse_gguf_metadata(data, GGUF_METADATA_SCAN_BYTES)
        except (ValueError, struct.error, IndexError, OverflowError) as e:
            logger.debug("GGUF metadata parse failed for %s: %s", file_path, e)
            if not any(f.rule_id == "MFV-GGUF-005" for f in findings):
                # Only report "could not parse" when the layout pass found
                # nothing structurally wrong. If GGUF-005 already fired,
                # the counting problems were already noted and this adds
                # nothing actionable.
                findings.append(Finding(
                    rule_id="MFV-GGUF-004",
                    message="GGUF magic number is valid, but the metadata KV section could not be "
                            "parsed -- possibly truncated, a non-standard variant, or corrupted. "
                            "Content could not be verified.",
                    severity=Severity.INFO,
                    category=Category.AI_ML,
                    file_path=str(file_path),
                    confidence=0.3,
                ))
            return findings

        dangerous_patterns: list[tuple[str, str]] = [
            ("__reduce__", "pickle reduce found, code execution possible"),
            ("<class", "Python class reference, potential deserialization risk"),
            ("subprocess", "subprocess reference in model metadata"),
            ("exec(", "exec() call in model metadata"),
            ("eval(", "eval() call in model metadata"),
            ("import(", "dynamic import in model metadata"),
            ("__import__", "__import__ dynamic import, code execution possible"),
            ("__builtins__", "builtins reference, sandbox escape risk"),
        ]

        hits: list[str] = []
        for key, value in kv.items():
            if key in _GGUF_FREETEXT_KEYS:
                continue
            values = value if isinstance(value, list) else [value]
            for v in values:
                if not isinstance(v, str):
                    continue
                for pattern, desc in dangerous_patterns:
                    if pattern in v:
                        hits.append(f"{desc} (metadata key '{key}')")

        if hits:
            findings.append(Finding(
                rule_id="MFV-GGUF-002",
                message=f"GGUF metadata contains suspicious content: {'; '.join(hits[:3])}. "
                        f"Model may have been tampered with.",
                severity=Severity.CRITICAL,
                category=Category.AI_ML,
                file_path=str(file_path),
                confidence=0.85,
                metadata={"hits": hits},
            ))

        # Check the actual chat_template metadata value for real SSTI
        # code-execution constructs, rather than the mere presence of "{{" --
        # every legitimate chat-tuned GGUF ships a Jinja2 template using
        # "{{ }}" for ordinary variable substitution (e.g. Llama-3's real
        # template is `{{ message['role'] }}`-shaped), so that alone is not
        # evidence of tampering and would false-positive on nearly every
        # real model. Only flag templates that actually reach for Python's
        # object-introspection/sandbox-escape surface or shell out.
        #
        # Signatures are matched only in code position: inside Jinja
        # expression/statement delimiters, with string literals blanked.
        # Nothing else executes: prose outside the delimiters is rendered
        # verbatim, and a string literal inside a block is data. Real
        # templates collide with the signatures in both places (measured:
        # unsloth's DeepSeek-V4 template trips "os." on the word
        # "scenarios.", both in prose and inside a {{ '...' }} literal).
        chat_template = kv.get("tokenizer.chat_template")
        if isinstance(chat_template, str):
            ssti_hit = _jinja_ssti_signature(chat_template)
            if ssti_hit is not None:
                findings.append(Finding(
                    rule_id="MFV-GGUF-003",
                    message=f"GGUF tokenizer.chat_template contains a code-execution construct "
                            f"({ssti_hit!r}) inside Jinja2 template syntax. "
                            f"CVE-2026-5760: Malicious model weights/metadata can inject template code for RCE.",
                    severity=Severity.CRITICAL,
                    category=Category.SSTI,
                    file_path=str(file_path),
                    confidence=0.80,
                    cwe_ids=[94, 1336],
                ))

        if kv_window_truncated:
            # The layout pass (_check_gguf_layout) walked the whole file, but
            # this content pass stopped at GGUF_METADATA_SCAN_BYTES. Keys past
            # that offset were never content-checked, so record the coverage
            # gap rather than let a dangerous key hide past the window in
            # silence. MFV-GGUF-004 is already this module's GGUF coverage rule.
            findings.append(Finding(
                rule_id="MFV-GGUF-004",
                message="GGUF metadata extends past the "
                        f"GGUF_METADATA_SCAN_BYTES ({GGUF_METADATA_SCAN_BYTES} bytes) "
                        "content-scan window, so keys beyond it were not "
                        "content-checked. Content past the window could not be "
                        "verified.",
                severity=Severity.INFO,
                category=Category.AI_ML,
                file_path=str(file_path),
                confidence=0.3,
                metadata={"skipped_reason": "gguf_metadata_window"},
            ))

        return findings

    # ── HuggingFace / transformers JSON config scanning ───────────

    def _scan_json_config(self, file_path: Path, config: dict) -> list[Finding]:
        """Inspect a transformers ``*.json`` config for its two RCE vectors.

        transformers loads these files straight from a model repo, so a
        malicious value in one executes on load the same way a malicious
        pickle does:

        1. A Jinja2 ``chat_template`` (in tokenizer_config.json /
           chat_template.json). This is the MFV-GGUF-003 threat carried in
           JSON instead of GGUF metadata: the template is rendered on load,
           so a code-execution construct in code position is SSTI -> RCE.

        2. ``auto_map`` and ``trust_remote_code`` (in config.json). auto_map
           names custom Python modules in the repo that transformers imports
           and executes when trust_remote_code is honored, the classic
           HF-hub remote-code vector.

        Conservative by construction: only the fields transformers actually
        reads are inspected, the chat_template is matched in Jinja *code
        position* (see ``_jinja_ssti_signature``) so ordinary substitution
        templates do not trip it, and everything else in the config is
        ignored. A plain config with none of these fields yields nothing.
        """
        findings: list[Finding] = []

        # ── Vector 1: chat_template SSTI ──────────────────────────
        # transformers accepts either a single template string or, in the
        # multi-template form, a list of {"name", "template"} entries. Collect
        # every template string present and match each in code position.
        templates: list[str] = []
        chat_template = config.get("chat_template")
        if isinstance(chat_template, str):
            templates.append(chat_template)
        elif isinstance(chat_template, list):
            for entry in chat_template:
                if isinstance(entry, dict) and isinstance(entry.get("template"), str):
                    templates.append(entry["template"])

        for template in templates:
            ssti_hit = _jinja_ssti_signature(template)
            if ssti_hit is not None:
                findings.append(Finding(
                    rule_id="MFV-HF-001",
                    message=f"chat_template contains a code-execution construct "
                            f"({ssti_hit!r}) inside Jinja2 template syntax. "
                            f"transformers renders this template on load, so a "
                            f"malicious template is server-side template injection "
                            f"for RCE -- the MFV-GGUF-003 / CVE-2026-5760 threat "
                            f"carried in a standalone JSON config.",
                    severity=Severity.CRITICAL,
                    category=Category.SSTI,
                    file_path=str(file_path),
                    confidence=0.80,
                    cwe_ids=[94, 1336],
                ))
                # One finding per file: the template is the vector, and a
                # second matching entry adds no new actionable information.
                break

        # ── Vector 2: auto_map / trust_remote_code remote code ────
        # auto_map maps transformers Auto* classes to dotted paths inside the
        # repo's own custom_code modules, which transformers imports and runs
        # when the model is loaded with trust_remote_code=True. Flag its
        # presence; it is not proof of malice (legitimate custom-architecture
        # models use it) but it is exactly the surface a hub-hosted repo uses
        # to reach code execution, hence HIGH at moderate confidence.
        auto_map = config.get("auto_map")
        if auto_map is not None:
            # Record the dotted targets when they are the ordinary
            # str -> str / str -> [str, str] shape, so a reviewer can see which
            # module would be imported without re-opening the file.
            targets: list[str] = []
            if isinstance(auto_map, dict):
                for value in auto_map.values():
                    if isinstance(value, str):
                        targets.append(value)
                    elif isinstance(value, list):
                        targets.extend(v for v in value if isinstance(v, str))
            findings.append(Finding(
                rule_id="MFV-HF-002",
                message="config declares auto_map, which points transformers at "
                        "custom Python modules in the model repo. Those modules "
                        "are imported and executed when the model is loaded with "
                        "trust_remote_code=True -- the HuggingFace-hub remote-code "
                        "execution vector.",
                severity=Severity.HIGH,
                category=Category.INJECTION,
                file_path=str(file_path),
                confidence=0.70,
                cwe_ids=[94],
                metadata={"auto_map_targets": targets} if targets else {},
            ))

        # A config that ships trust_remote_code: true is a model asserting that
        # its own repo code should be executed without prompting. That is the
        # remote-code vector stated in the config itself, reported separately
        # from auto_map because either can appear without the other.
        if config.get("trust_remote_code") is True:
            findings.append(Finding(
                rule_id="MFV-HF-002",
                message="config sets trust_remote_code: true, asking loaders to "
                        "import and execute the repo's own Python without "
                        "prompting. A model's config asserting that its code "
                        "should be trusted is the remote-code execution vector, "
                        "not a safe default.",
                severity=Severity.HIGH,
                category=Category.INJECTION,
                file_path=str(file_path),
                confidence=0.75,
                cwe_ids=[94],
            ))

        return findings

    # ── Keras H5 scanning ─────────────────────────────────────────

    def _scan_keras(self, file_path: Path) -> list[Finding]:
        """Detect Lambda layers and unrecognized custom layer classes in
        Keras H5 files by walking the actual parsed layer graph
        (model_config JSON), instead of substring-matching words like
        "lambda"/"function"/"custom_object" anywhere in the raw HDF5
        container -- which also holds tensor names and weight bytes that can
        spuriously contain those words (e.g. an "activation function" config
        string, or a layer literally named "function").
        """
        try:
            with open(file_path, "rb") as f:
                header = f.read(8)
        except OSError:
            return [_skip_unverified_finding(
                file_path,
                "The file could not be read, so its layer graph was never checked.",
                metadata={"skipped_reason": "unreadable"},
            )]

        # HDF5 signature check
        if header[:8] != HDF5_SIGNATURE:
            # Not HDF5, so try the zip-based .keras format
            return self._scan_keras_zip(file_path)

        try:
            data = file_path.read_bytes()
        except OSError:
            return [_skip_unverified_finding(
                file_path,
                "The file could not be read, so its layer graph was never checked.",
                metadata={"skipped_reason": "unreadable"},
            )]

        config = _extract_keras_model_config(data)
        if config is None:
            # Weights-only H5 (no model_config attribute) -- no layer graph
            # to check; the file contains only numeric tensor data.
            return []

        return self._keras_findings_from_config(file_path, config)

    def _scan_keras_zip(self, file_path: Path) -> list[Finding]:
        """Scan the new .keras format (a zip containing config.json with the
        same class_name/layer-graph shape as the legacy H5 model_config)."""
        try:
            with zipfile.ZipFile(file_path, "r") as zf:
                if "config.json" not in zf.namelist():
                    return []
                raw = zf.read("config.json")
        except zipfile.BadZipFile:
            # Not a zip at all: the extension-confusion check in scan_file
            # owns that case (it re-scans the bytes against every known magic).
            return []
        except (OSError, KeyError):
            return [_skip_unverified_finding(
                file_path,
                "The archive's config.json could not be read, so the layer "
                "graph was never checked.",
                metadata={"skipped_reason": "unreadable_member"},
            )]

        try:
            config = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return [_skip_unverified_finding(
                file_path,
                "The archive's config.json does not parse as JSON, so the "
                "layer graph was never checked.",
                metadata={"skipped_reason": "unparseable_config"},
            )]
        if not isinstance(config, dict):
            return []

        return self._keras_findings_from_config(file_path, config)

    def _scan_keras_metadata(self, file_path: Path, data: bytes) -> list[Finding]:
        """Scan a SavedModel's `keras_metadata.pb` for the same risky layer
        graph the `.h5`/`.keras` paths look for.

        The file is a `SavedMetadata` protobuf, but the Keras `model_config`
        it carries is a plain JSON string field, so the existing raw-byte
        config extractor reads it without needing protobuf or tensorflow
        installed. The root node's metadata holds the fully nested layer
        graph, which is why the first extracted object is sufficient.
        """
        config = _extract_keras_model_config(data)
        if config is None:
            return []
        return self._keras_findings_from_config(file_path, config)

    def _keras_findings_from_config(self, file_path: Path, config: dict) -> list[Finding]:
        findings: list[Finding] = []

        lambda_layers = _find_keras_risky_layers(config)
        if lambda_layers:
            findings.append(Finding(
                rule_id="MFV-KERAS-001",
                message=f"Keras model contains {len(lambda_layers)} Lambda layer(s): "
                        f"{', '.join(lambda_layers[:3])}. Lambda layers embed a "
                        f"marshalled/serialized Python function that executes on model load.",
                severity=Severity.HIGH,
                category=Category.DESERIALIZATION,
                file_path=str(file_path),
                confidence=0.85,
                cwe_ids=[502],
                metadata={"lambda_layers": lambda_layers},
            ))

        unrecognized = _find_keras_unrecognized_classes(config)
        if unrecognized:
            findings.append(Finding(
                rule_id="MFV-KERAS-002",
                message=f"Keras model references layer class(es) not on the known-builtin "
                        f"list: {', '.join(unrecognized[:10])}. Likely a legitimate custom "
                        f"layer, but not verifiable by static analysis -- executes via "
                        f"registered custom_objects at load time.",
                severity=Severity.INFO,
                category=Category.DESERIALIZATION,
                file_path=str(file_path),
                confidence=0.3,
                cwe_ids=[502],
                metadata={"unrecognized_classes": unrecognized},
            ))

        return findings

    # ── PyTorch zip scanning ──────────────────────────────────────

    def _read_zip_member_capped(self, zf: zipfile.ZipFile, name: str, max_bytes: int) -> bytes | None:
        """Read a zip member's decompressed bytes, aborting once max_bytes is exceeded.

        ZipFile.read() decompresses the full stream regardless of the
        (attacker-controlled) declared file_size in the central directory --
        a small compressed member can still expand to gigabytes. Read in
        chunks and stop as soon as the cap is exceeded instead of trusting
        that metadata, so a malicious checkpoint can't exhaust memory.
        """
        chunks: list[bytes] = []
        total = 0
        with zf.open(name) as fh:
            while True:
                chunk = fh.read(1_000_000)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    return None
                chunks.append(chunk)
        return b"".join(chunks)

    # First bytes of a pickle stream: protocol 2..5 opens with `\x80<proto>`,
    # and protocol 0/1 opens with one of a small set of printable opcodes.
    # `b"c"` (GLOBAL) over-triggers on ordinary text members, but dropping it
    # would stop sniffing protocol-0 payloads entirely -- a detection
    # regression an attacker could pick deliberately -- so it stays.
    _PICKLE_OPENERS: tuple[bytes, ...] = (
        b"\x80\x02", b"\x80\x03", b"\x80\x04", b"\x80\x05",
        b"c", b"(", b"]", b"}",
    )

    def _zip_member_may_be_pickle(
        self, zf: zipfile.ZipFile, info: zipfile.ZipInfo,
        source: Path | bytes | None,
    ) -> bool:
        """Whether a zip member is worth running the pickle analysis over.

        Name first, because `archive/data.pkl` is the convention and checking a
        string is free. But the name is attacker-chosen: picklescan's corpus
        carries the same payload as `data.txt` precisely because scanners that
        filter members by extension skip it. So anything not matched by name is
        sniffed on its opening bytes instead.

        Only the first two bytes are read, which keeps the cost negligible on
        checkpoints whose other members are large tensor blobs.

        The decision never gates on the declared file_size: it is
        attacker-controlled central-directory metadata, and inflating it used
        to keep a pickle member from ever being sniffed. The sniff reads real
        bytes, and every later read path keeps its own decompression cap.

        `source` is where the raw fallback reads: the archive's path on disk,
        or the archive's bytes for a nested archive held in memory, or None
        when neither is available (member then stays un-sniffed).
        """
        name = info.filename
        if name.endswith((".pkl", ".pickle")) or name.rsplit("/", 1)[-1] == "data.pkl":
            return True
        try:
            with zf.open(info) as handle:
                head = handle.read(2)
        except (OSError, zipfile.BadZipFile, RuntimeError,
                NotImplementedError, ValueError):
            # Unreadable through the strict path (lied central-directory
            # sizes raise BadZipFile *or* ValueError from zipfile's offset
            # arithmetic). Sniff the stored bytes, for the same reason the
            # read fallback exists below.
            if source is None:
                return False
            raw = self._read_zip_member_raw(source, info, head_only=True)
            head = raw or b""
        return head.startswith(self._PICKLE_OPENERS)

    def _read_zip_member_raw(
        self, source: Path | bytes, info: zipfile.ZipInfo, head_only: bool = False,
        report_oversized: bool = False,
    ) -> bytes | None | object:
        """Read a member's bytes straight out of the archive, bypassing
        `zipfile`'s flag-bit checks.

        `source` is the archive's path on disk, or the archive's bytes
        themselves for a nested archive held in memory (whose local-header
        offsets are relative to it, not to any file on disk).

        Used only when the strict reader refuses. Parses the *local* file
        header (whose name/extra lengths can differ from the central
        directory's, which is itself a parser-differential trick) to find where
        the data starts, then returns it: verbatim for STORED members, raw-
        inflated for DEFLATED ones.

        Returns None when the member cannot be located, is unreadable, or (by
        default) exceeds the size cap. `report_oversized` splits that last case
        out: when True, a member that decompresses past MAX_ZIP_MEMBER_BYTES
        returns the `_ZIP_MEMBER_OVERSIZED` sentinel instead of None, so the
        caller can raise MFV-PICKLE-003 for it exactly as the strict path does,
        rather than skipping a zip bomb in silence. None still means unreadable.
        Never raises: this runs on attacker-controlled bytes.
        """
        try:
            handle = (io.BytesIO(source) if isinstance(source, bytes)
                      else open(source, "rb"))
            with handle:
                handle.seek(info.header_offset)
                local = handle.read(30)
                if len(local) < 30 or local[:4] != b"PK\x03\x04":
                    return None
                name_len, extra_len = struct.unpack("<HH", local[26:30])
                handle.seek(info.header_offset + 30 + name_len + extra_len)
                # No gate on the declared compress_size: like file_size it is
                # attacker-controlled, and inflating it used to make this
                # fallback refuse plainly readable members before reading a
                # byte. The actual read is capped instead, which is the bound
                # that was always supposed to do the work. head_only reads
                # enough compressed bytes to inflate the two-byte sniff.
                want = 128 if head_only else self.MAX_ZIP_MEMBER_BYTES + 1
                blob = handle.read(want)
        except (OSError, ValueError):
            # ValueError: zipfile's offset arithmetic can hand back a
            # negative header_offset when the central directory lies.
            return None

        if info.compress_type == zipfile.ZIP_STORED:
            if head_only:
                return blob[:2]
            if len(blob) > self.MAX_ZIP_MEMBER_BYTES:
                # We read MAX+1 stored bytes and got more than the cap, so the
                # member is genuinely oversized (not merely unreadable).
                return _ZIP_MEMBER_OVERSIZED if report_oversized else None
            return blob
        try:
            decompressor = zlib.decompressobj(-15)
            if head_only:
                return decompressor.decompress(blob, 2) or None
            out = decompressor.decompress(blob, self.MAX_ZIP_MEMBER_BYTES)
        except zlib.error:
            return None
        if not decompressor.eof:
            # decompress() stopped short of the stream end for one of two
            # reasons, and they are different verdicts:
            #   - it hit max_length with compressed input still pending
            #     (unconsumed_tail non-empty): the member decompresses past the
            #     cap, i.e. oversized, the same verdict the strict read returns.
            #   - the compressed input ran out first (unconsumed_tail empty):
            #     a cut-off stream, i.e. unreadable, never scanned as whole.
            if decompressor.unconsumed_tail:
                return _ZIP_MEMBER_OVERSIZED if report_oversized else None
            return None
        return out

    _ZIP_LOCAL_MAGIC = b"PK\x03\x04"

    # Cap on how many tar members are inspected. A tar has no central
    # directory, so the only way to know what it holds is to walk it; this
    # bounds that walk against an archive padded with millions of tiny
    # entries. Real .nemo checkpoints hold a handful of files.
    MAX_TAR_MEMBERS = 1000

    def _scan_tar_pickles(self, file_path: Path) -> list[Finding]:
        """Scan the pickle streams inside a tar-based checkpoint container.

        NVIDIA NeMo's `.nemo` is a plain (usually uncompressed) tar holding
        `model_config.yaml` alongside `model_weights.ckpt`, and that ckpt is
        an ordinary pickle -- either flat or, for newer exports, itself a zip
        in torch's own format. Both shapes are handled.

        Nothing is ever extracted to disk, so tar path-traversal ("zip slip")
        is not reachable from here: members are only read into memory, under
        the same per-member size cap the zip path uses.
        """
        findings: list[Finding] = []
        try:
            archive = tarfile.open(file_path, "r:*")
        except (tarfile.TarError, OSError, EOFError):
            # A .nemo that does not open as a tar is corrupt or extension-
            # spoofed. scan_file's early return for NEMO bypasses the
            # extension-confusion check, so nothing else owns this case:
            # silence would be a clean verdict on a file never analysed.
            return [_skip_unverified_finding(
                file_path,
                "The .nemo container does not open as a tar archive.",
                metadata={"skipped_reason": "container_open_failed"},
            )]

        try:
            with archive as tf:
                hit_member_cap = False
                for index, member in enumerate(tf):
                    if index >= self.MAX_TAR_MEMBERS:
                        hit_member_cap = True
                        break
                    if not member.isfile():
                        continue
                    if member.size < 2 or member.size > self.MAX_ZIP_MEMBER_BYTES:
                        continue
                    try:
                        handle = tf.extractfile(member)
                        if handle is None:
                            continue
                        blob = handle.read(self.MAX_ZIP_MEMBER_BYTES)
                    except (OSError, tarfile.TarError):
                        continue
                    if len(blob) < 2:
                        continue

                    if blob[:4] == self._ZIP_LOCAL_MAGIC:
                        findings.extend(self._scan_zip_bytes(blob, member.name, file_path))
                        continue
                    if not blob.startswith(self._PICKLE_OPENERS):
                        exec_hits = _find_embedded_executables(blob)
                        if exec_hits:
                            findings.append(self._executable_finding(
                                file_path, exec_hits, member=member.name))
                        continue
                    for f in self._scan_pickle(file_path, blob):
                        f.message = f"[{member.name}] {f.message}"
                        f.metadata = {**(f.metadata or {}), "tar_member": member.name}
                        findings.append(f)
                # Iteration ending is not proof the file ended. A tar cut
                # inside a member *header* stops the walk with no exception
                # at all (measured: members before the cut are yielded, the
                # rest vanish silently). The walk is only complete when what
                # remains past the read offset is the format's own zero
                # padding; a non-zero tail is unexamined content.
                if hit_member_cap:
                    findings.append(Finding(
                        rule_id="MFV-SKIP-002",
                        message=f"Tar container has more than {self.MAX_TAR_MEMBERS} "
                                f"members; only the first {self.MAX_TAR_MEMBERS} were "
                                f"analysed. NOT a clean verdict for the remainder.",
                        severity=Severity.LOW,
                        category=Category.AI_ML,
                        file_path=str(file_path),
                        confidence=0.50,
                        metadata={"skipped_reason": "member_cap"},
                    ))
                else:
                    consumed = tf.offset
                    file_size = file_path.stat().st_size
                    if 0 < consumed < file_size:
                        with open(file_path, "rb") as fh:
                            fh.seek(consumed)
                            tail = fh.read(1 << 20)
                        if any(tail):
                            findings.append(Finding(
                                rule_id="MFV-SKIP-002",
                                message=f"Tar walk stopped {file_size - consumed} bytes "
                                        f"before the end of the file with no error raised; "
                                        f"the unread tail contains data. NOT a clean verdict "
                                        f"for the remainder of the archive.",
                                severity=Severity.LOW,
                                category=Category.AI_ML,
                                file_path=str(file_path),
                                confidence=0.55,
                                metadata={"skipped_reason": "unread_tail",
                                          "unread_bytes": file_size - consumed},
                            ))
        except (tarfile.TarError, OSError, EOFError) as exc:
            # The walk died partway. Returning [] here would discard the members
            # already scanned, which makes "payload in member 1, malformed
            # member 2" a better evasion than the payload on its own. Keep what
            # was found and say the rest was never looked at.
            findings.append(Finding(
                rule_id="MFV-SKIP-002",
                message=f"Tar container walk ended early ({type(exc).__name__}), so any "
                        f"members past the failure point were never analysed. NOT a clean "
                        f"verdict for the remainder of the archive.",
                severity=Severity.LOW,
                category=Category.AI_ML,
                file_path=str(file_path),
                confidence=0.50,
                metadata={"skipped_reason": "tar_walk_failed", "error": type(exc).__name__},
            ))
        return findings

    def _scan_nested_zip_member(
        self, zf: zipfile.ZipFile, info: zipfile.ZipInfo, file_path: Path,
    ) -> list[Finding]:
        """Scan pickle streams one level further down, inside a zip member
        that is itself a zip.

        TorchServe's `.mar` is the case that makes this necessary:
        `torch-model-archiver --serialized-file model.pt` stores a whole
        PyTorch checkpoint (already a zip wrapping `archive/data.pkl`) as an
        archive member, so the payload sits two containers deep and a
        single-level walk reports the file clean.

        Bounded to exactly one extra level. That matches the real packaging
        convention and leaves no unbounded recursion for a nested-zip bomb to
        drive; the per-member size cap still applies to every actual read at
        both levels. The decision never gates on the declared file_size: it
        is attacker-controlled, and inflating it used to keep a nested
        checkpoint from ever being sniffed.
        """
        try:
            with zf.open(info) as handle:
                if handle.read(4) != self._ZIP_LOCAL_MAGIC:
                    return []
            blob = self._read_zip_member_capped(zf, info.filename, self.MAX_ZIP_MEMBER_BYTES)
        except (OSError, zipfile.BadZipFile, RuntimeError,
                NotImplementedError, ValueError):
            # Same lie the .pt walk already handles one level up: lied flag
            # bits or lied sizes make the strict reader refuse a plainly
            # readable member. The only caller opens `zf` on `file_path`
            # itself, so the raw local-header read is valid here.
            blob = self._read_zip_member_raw(file_path, info)
            if blob is None or blob[:4] != self._ZIP_LOCAL_MAGIC:
                return []
            # The raw read is capped, not member-exact: a lied declared size
            # lets it run past the member into the outer container's own
            # bytes, which would confuse the nested archive's parse.
            blob = _zip_blob_upto_archive_end(blob)
        if blob is None:
            return []

        return self._scan_zip_bytes(blob, info.filename, file_path)

    def _scan_zip_bytes(self, blob: bytes, prefix: str, file_path: Path) -> list[Finding]:
        """Scan every pickle member of a zip archive held in memory.

        Shared by the nested-zip descent (a `.pt` inside a `.mar`) and the tar
        container walk (a zip-format checkpoint inside a `.nemo`); both hold
        the inner archive as bytes rather than as a file on disk.
        """
        findings: list[Finding] = []
        try:
            with zipfile.ZipFile(io.BytesIO(blob)) as inner_zf:
                for inner in inner_zf.infolist():
                    # The raw-bytes fallback gets the archive's own bytes:
                    # local-header offsets are relative to this in-memory
                    # archive, not to any file on disk.
                    if not self._zip_member_may_be_pickle(inner_zf, inner, blob):
                        continue
                    try:
                        data = self._read_zip_member_capped(
                            inner_zf, inner.filename, self.MAX_ZIP_MEMBER_BYTES,
                        )
                    except (OSError, zipfile.BadZipFile, RuntimeError,
                            NotImplementedError, ValueError):
                        # Same lied-metadata shape one level up: the strict
                        # reader refuses, the bytes are plainly there.
                        data = self._read_zip_member_raw(blob, inner)
                    if data is None:
                        continue
                    label = f"{prefix}/{inner.filename}"
                    for f in self._scan_pickle(file_path, data):
                        f.message = f"[{label}] {f.message}"
                        f.metadata = {**(f.metadata or {}), "zip_member": label}
                        findings.append(f)
        except (zipfile.BadZipFile, OSError) as exc:
            # The inner archive never opened or the walk died partway.
            # Returning [] discards the members already scanned, the same
            # evasion the tar walk's MFV-SKIP-002 exists to close.
            findings.append(Finding(
                rule_id="MFV-SKIP-002",
                message=f"[{prefix}] Zip container walk ended early "
                        f"({type(exc).__name__}), so any members past the failure "
                        f"point were never analysed. NOT a clean verdict for the "
                        f"remainder of the archive.",
                severity=Severity.LOW,
                category=Category.AI_ML,
                file_path=str(file_path),
                confidence=0.50,
                metadata={
                    "skipped_reason": "zip_walk_failed",
                    "error": type(exc).__name__,
                    "zip_member": prefix,
                },
            ))
        return findings

    def _torch_source_finding(
        self, file_path: Path, names: list[str],
    ) -> Finding | None:
        """MFV-TORCH-001: a torch zip carries executable Python source.

        Conservative by construction, to avoid flagging an ordinary data
        member: it fires only when the archive both (a) carries `*.py`
        members and (b) looks like a torch container that would actually
        execute them. Recognition markers are the two layouts that do:
        torch.package's `.data/` directory and its `extern_modules` /
        `python_version` manifest, and TorchScript's `code/` directory
        alongside a `data.pkl` / `constants.pkl` pickle. A plain state_dict
        checkpoint has no `.py` members at all, so it never reaches the
        second condition.
        """
        source_members = [n for n in names if n.endswith(".py")]
        if not source_members:
            return None

        # A torch container, not just any zip that happens to hold a .py.
        # `.data/` (with a leading path segment or at the root) and a
        # top-level `code/` are the package/TorchScript signatures; the
        # pickle sidecars confirm it is a serialized model rather than a
        # source tarball someone renamed.
        def _is_torch_layout(n: str) -> bool:
            segments = n.split("/")
            last = segments[-1]
            return (
                ".data" in segments
                or segments[0] == "code"
                or last in ("data.pkl", "constants.pkl", "extern_modules")
            )

        if not any(_is_torch_layout(n) for n in names):
            return None

        shown = ", ".join(sorted(source_members)[:5])
        return Finding(
            rule_id="MFV-TORCH-001",
            message=f"Torch archive carries executable Python source that runs "
                    f"on load: {shown}. torch.package's PackageImporter imports "
                    f"these modules and torch.jit compiles a TorchScript "
                    f"`code/` directory, so loading the model executes the "
                    f"packaged code, not just the weights.",
            severity=Severity.HIGH,
            category=Category.DESERIALIZATION,
            file_path=str(file_path),
            confidence=0.75,
            cwe_ids=[94],
            metadata={"source_members": sorted(source_members)[:20]},
        )

    def _scan_pytorch_zip(self, file_path: Path) -> list[Finding]:
        """Scan the pickle stream(s) inside a PyTorch (or other) zip checkpoint.

        torch.save() stores the real pickle as ``archive/data.pkl`` inside a ZIP
        container. We extract every pickle member and run the same opcode/byte
        analysis on each inner stream, re-attributing findings to the member so
        the report points at ``model.pt:archive/data.pkl``.
        """
        findings: list[Finding] = []
        try:
            zf = zipfile.ZipFile(file_path, "r")
        except (zipfile.BadZipFile, OSError):
            # Not actually a zip, so treat it as a legacy flat pickle. The
            # read is capped: this path is also reachable from scan_file's
            # oversized-file branch for anything zipfile.is_zipfile() merely
            # liked the look of, where an unbounded read_bytes() would pull
            # multi-GB into memory despite the cap that exists to prevent
            # exactly that. A pickle never needs more than the scan cap to
            # be analysed, so the prefix carries every verdict this path can.
            with open(file_path, "rb") as handle:
                data = handle.read(self.MAX_SCAN_BYTES)
            return self._scan_pickle(file_path, data)
        try:
            with zf:
                # Before the pickle walk: a torch zip can carry executable
                # Python *source*, not just a pickle. torch.package archives
                # ship a `.data/` layout with `*.py` modules that
                # PackageImporter imports on load, and a TorchScript archive
                # ships a `code/` directory of `.py` that torch.jit compiles
                # and runs on load. Either way the source executes when the
                # model is loaded, so it is a code-execution surface a pickle
                # scan alone would miss. Reported separately from the pickle
                # findings via MFV-TORCH-001.
                names = [info.filename for info in zf.infolist()]
                source_finding = self._torch_source_finding(file_path, names)
                if source_finding is not None:
                    findings.append(source_finding)
                for info in zf.infolist():
                    name = info.filename
                    if not self._zip_member_may_be_pickle(zf, info, file_path):
                        findings.extend(self._scan_nested_zip_member(zf, info, file_path))
                        continue
                    try:
                        inner = self._read_zip_member_capped(zf, name, self.MAX_ZIP_MEMBER_BYTES)
                    except (OSError, zipfile.BadZipFile, RuntimeError,
                            NotImplementedError, ValueError):
                        # A member the strict reader refuses is NOT evidence of
                        # safety, and skipping it silently reported the file
                        # clean. The refusal is usually a lie: setting the
                        # general-purpose flag bits for "encrypted" (0x1),
                        # "compressed patched data" (0x20) or "strong
                        # encryption" (0x40) makes Python's `zipfile` bail,
                        # while the member is plainly STORED and readable, and
                        # loaders that use their own zip reader (torch ships a
                        # miniz-based one) go straight past it. Three bytes of
                        # header edit therefore evaded this scanner completely
                        # on `.pt`, the most common PyTorch checkpoint format.
                        #
                        # So fall back to the bytes as they physically sit in
                        # the archive. If a member really is encrypted, that
                        # read yields ciphertext, the pickle walk fails on it
                        # and nothing is reported -- which is the correct
                        # outcome, and is what keeps a genuinely
                        # password-protected archive quiet.
                        # report_oversized so an oversized member read through
                        # the raw fallback reaches the same MFV-PICKLE-003
                        # verdict below as the strict path, instead of the raw
                        # None being silently skipped (a zip bomb the strict
                        # reader would have flagged but a lied flag bit routed
                        # here).
                        inner = self._read_zip_member_raw(
                            file_path, info, report_oversized=True,
                        )
                        if inner is None:
                            continue
                    if inner is None or inner is _ZIP_MEMBER_OVERSIZED:
                        findings.append(Finding(
                            rule_id="MFV-PICKLE-003",
                            message=f"[{name}] Pickle member exceeds "
                                    f"{self.MAX_ZIP_MEMBER_BYTES // 1_000_000}MB decompressed size "
                                    f"limit -- possible zip bomb. Skipping for memory safety.",
                            severity=Severity.HIGH,
                            category=Category.AI_ML,
                            file_path=str(file_path),
                            confidence=0.6,
                            metadata={"zip_member": name, "skipped_reason": "oversized_decompressed"},
                        ))
                        continue
                    exec_hits = _find_embedded_executables(inner)
                    if exec_hits:
                        findings.append(self._executable_finding(file_path, exec_hits, member=name))
                    for f in self._scan_pickle(file_path, inner):
                        f.message = f"[{name}] {f.message}"
                        f.metadata = {**(f.metadata or {}), "zip_member": name}
                        findings.append(f)
        except (zipfile.BadZipFile, OSError) as exc:
            # The walk died partway. Falling back to a flat-pickle scan of zip
            # bytes (which are not a pickle) would discard the members already
            # scanned, making "payload in member 1, malformed member 2" a
            # better evasion than the payload on its own. Keep what was found
            # and say the rest was never looked at.
            findings.append(Finding(
                rule_id="MFV-SKIP-002",
                message=f"Zip container walk ended early ({type(exc).__name__}), so any "
                        f"members past the failure point were never analysed. NOT a clean "
                        f"verdict for the remainder of the archive.",
                severity=Severity.LOW,
                category=Category.AI_ML,
                file_path=str(file_path),
                confidence=0.50,
                metadata={"skipped_reason": "zip_walk_failed", "error": type(exc).__name__},
            ))
        return findings

    # ── skops scanning ──────────────────────────────────────────────
    #
    # skops (skops.io) is Hugging Face's pickle-free sklearn persistence: a
    # zip holding `schema.json` (a tree of typed nodes) plus one `.npy`
    # member per array. Its security contract is structural, which makes it
    # checkable without any name list of our own:
    #
    # 1. Every object/function the loader will resolve and instantiate is a
    #    `__module__` + `__class__` pair in the schema, so type references
    #    classify with exactly the pickle engine's allow/deny/unknown split.
    #    A skops file reaching for `os.system` is as damning as a pickle
    #    doing it, and for the same reason.
    # 2. Two loaders carry a structural invariant the file can break:
    #    MethodNode must agree with the object it binds (breaking the
    #    agreement is CVE-2025-54413's primitive), and OperatorFuncNode must
    #    live in the `operator` module (CVE-2025-54412's). skops >= 0.12.0
    #    raises on both at load; older versions execute. A file in the broken
    #    shape is malicious per se, whatever the victim's skops version.
    # 3. The format never embeds pickle (its reason to exist is not to). A
    #    member that walks as a pickle stream defeats the guarantee and gets
    #    the pickle engine's full analysis plus a note of its own.

    # Loaders skops 0.x can emit (NODE_TYPE_MAPPING plus the ReduceNode
    # subclasses). Closed set: anything else is a protocol this scanner does
    # not know and lands in the unknown bucket rather than being trusted.
    _SKOPS_KNOWN_LOADERS: frozenset[str] = frozenset({
        "DictNode", "DefaultDictNode", "ListNode", "SetNode", "TupleNode",
        "FunctionNode", "PartialNode", "TypeNode", "SliceNode",
        "ConstructorFromReduceNode", "ObjectNode", "MethodNode", "JsonNode",
        "BytesNode", "BytearrayNode", "OperatorFuncNode", "NdArrayNode",
        "MaskedArrayNode", "RandomStateNode", "RandomGeneratorNode",
        "DTypeNode", "SparseMatrixNode", "ReduceNode", "TreeNode", "LossNode",
        "QuantileForestNode", "_DictWithDeprecatedKeysNode", "CachedNode",
    })

    # Cap on schema.json bytes handed to json.loads: attacker-chosen nesting
    # is the one place a value drives real work. Real schemas are kilobytes;
    # a gradient-boosting ensemble is the largest legitimate case seen and
    # stays under 1MB.
    _SKOPS_MAX_SCHEMA_BYTES = 20_000_000
    # Bound on nodes walked per schema, against a billion-laughs JSON tree.
    _SKOPS_MAX_SCHEMA_NODES = 200_000

    def _scan_skops(self, file_path: Path) -> list[Finding]:
        findings: list[Finding] = []
        try:
            zf = zipfile.ZipFile(file_path, "r")
        except (zipfile.BadZipFile, OSError):
            # skops.io.dump only ever writes a zip, so a non-zip .skops is
            # corrupt or extension-spoofed. skops is not wired into the
            # extension-confusion check, so nothing else owns this case.
            return [_skip_unverified_finding(
                file_path,
                "The .skops file does not open as a zip archive, which "
                "skops.io.dump always writes.",
                metadata={"skipped_reason": "container_open_failed"},
            )]
        with zf:
            schemas = [
                info for info in zf.infolist()
                if info.filename.rsplit("/", 1)[-1] == "schema.json"
            ]
            if not schemas:
                findings.append(Finding(
                    rule_id="MFV-SKOPS-005",
                    message="Archive has a .skops extension but contains no schema.json, "
                            "which skops.io.dump always writes. Not a standard skops file; "
                            "content could not be verified.",
                    severity=Severity.INFO,
                    category=Category.AI_ML,
                    file_path=str(file_path),
                    confidence=0.4,
                ))
            for info in schemas:
                if info.file_size > self._SKOPS_MAX_SCHEMA_BYTES:
                    findings.append(Finding(
                        rule_id="MFV-SKOPS-005",
                        message=f"[{info.filename}] schema.json exceeds "
                                f"{self._SKOPS_MAX_SCHEMA_BYTES // 1_000_000}MB; too large to "
                                f"verify. NOT a clean verdict.",
                        severity=Severity.LOW,
                        category=Category.AI_ML,
                        file_path=str(file_path),
                        confidence=0.4,
                    ))
                    continue
                try:
                    raw = self._read_zip_member_capped(
                        zf, info.filename, self._SKOPS_MAX_SCHEMA_BYTES)
                except (OSError, zipfile.BadZipFile, RuntimeError,
                        NotImplementedError, ValueError):
                    # One unreadable member must not abort the whole archive
                    # walk: the pickle-member pass below still has to run.
                    raw = None
                if raw is None:
                    continue
                try:
                    schema = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError, RecursionError,
                        ValueError, MemoryError):
                    findings.append(Finding(
                        rule_id="MFV-SKOPS-005",
                        message=f"[{info.filename}] schema.json does not parse as JSON; "
                                "content could not be verified.",
                        severity=Severity.INFO,
                        category=Category.AI_ML,
                        file_path=str(file_path),
                        confidence=0.4,
                    ))
                    continue
                findings.extend(self._scan_skops_schema(file_path, info.filename, schema))

            # Any member that walks as a pickle stream gets the pickle
            # engine, same as a .mar member, plus a note that the archive
            # broke the format's pickle-free guarantee.
            for info in zf.infolist():
                if info in schemas:
                    continue
                if not self._zip_member_may_be_pickle(zf, info, file_path):
                    continue
                try:
                    inner = self._read_zip_member_capped(zf, info.filename,
                                                         self.MAX_ZIP_MEMBER_BYTES)
                except (OSError, zipfile.BadZipFile, RuntimeError,
                        NotImplementedError, ValueError):
                    # Lied metadata makes the strict reader refuse a plainly
                    # readable member; the bytes on disk are still readable.
                    inner = self._read_zip_member_raw(file_path, info)
                if inner is None:
                    continue
                findings.append(Finding(
                    rule_id="MFV-SKOPS-004",
                    message=f"[{info.filename}] skops archive embeds a pickle stream. "
                            "The format's security proposition is that it never contains "
                            "one, so its presence defeats the guarantee the loader relies "
                            "on. The stream itself is analysed separately.",
                    severity=Severity.LOW,
                    category=Category.DESERIALIZATION,
                    file_path=str(file_path),
                    confidence=0.6,
                    cwe_ids=[502],
                    metadata={"zip_member": info.filename},
                ))
                for f in self._scan_pickle(file_path, inner):
                    f.message = f"[{info.filename}] {f.message}"
                    f.metadata = {**(f.metadata or {}), "zip_member": info.filename}
                    findings.append(f)
        return findings

    def _scan_skops_schema(
        self, file_path: Path, member: str, schema: object,
    ) -> list[Finding]:
        """Walk one parsed schema tree. Iterative with a node cap: both the
        depth and the breadth of this JSON are attacker-chosen."""
        denied: list[str] = []
        unknown: list[str] = []
        unknown_loaders: list[str] = []
        structural: list[str] = []

        stack = [schema]
        visited = 0
        while stack:
            node = stack.pop()
            if isinstance(node, list):
                stack.extend(node)
            elif isinstance(node, dict):
                stack.extend(node.values())
                visited += 1
                if visited > self._SKOPS_MAX_SCHEMA_NODES:
                    break
                loader = node.get("__loader__")
                if loader is None:
                    continue
                if loader not in self._SKOPS_KNOWN_LOADERS:
                    unknown_loaders.append(str(loader))
                module = node.get("__module__")
                klass = node.get("__class__")
                if loader == "OperatorFuncNode" and module != "operator":
                    structural.append(
                        f"OperatorFuncNode from module {module!r} (the loader "
                        f"resolves it in `operator`; any other module is "
                        f"CVE-2025-54412's primitive, fixed in skops "
                        f"0.12.0, executed before it)"
                    )
                if loader == "MethodNode" and isinstance(node.get("content"), dict):
                    obj = node["content"].get("obj")
                    if isinstance(obj, dict) and (
                        obj.get("__module__") != module
                        or obj.get("__class__") != klass
                    ):
                        structural.append(
                            f"MethodNode declaring {module}.{klass} but binding "
                            f"{obj.get('__module__')}.{obj.get('__class__')} "
                            f"(CVE-2025-54413's primitive: the pre-0.12.0 "
                            f"loader trusted the outer pair and called the inner)"
                        )
                if isinstance(module, str) and isinstance(klass, str):
                    ref = f"{module}.{klass}"
                    verdict = _classify_pickle_global(ref)
                    if verdict == "denied":
                        denied.append(ref)
                    elif verdict == "unknown" and module != "builtins":
                        # builtins containers are the serialization machinery
                        # itself (DictNode of builtins.dict, ...), not payload.
                        unknown.append(ref)

        findings: list[Finding] = []
        if structural:
            findings.append(Finding(
                rule_id="MFV-SKOPS-002",
                message=f"[{member}] skops loader state is inconsistent with the "
                        f"loader's own contract: {'; '.join(structural[:3])}. "
                        "skops >= 0.12.0 refuses this file; older versions resolve "
                        "and call through it.",
                severity=Severity.CRITICAL,
                category=Category.DESERIALIZATION,
                file_path=str(file_path),
                confidence=0.9,
                cwe_ids=[502],
                metadata={"inconsistencies": structural},
            ))
        if denied:
            findings.append(Finding(
                rule_id="MFV-SKOPS-001",
                message=f"[{member}] skops schema references type(s) that grant "
                        f"code/command execution on load: {', '.join(sorted(denied))}. "
                        "skops resolves every __module__/__class__ pair it is asked to "
                        "trust; these must never appear in a model.",
                severity=Severity.CRITICAL,
                category=Category.DESERIALIZATION,
                file_path=str(file_path),
                confidence=0.9,
                cwe_ids=[502],
                metadata={"denied_types": sorted(denied)},
            ))
        if unknown or unknown_loaders:
            parts = []
            if unknown:
                parts.append(
                    "type reference(s) outside the trusted set: "
                    + ", ".join(sorted(set(unknown))[:10])
                )
            if unknown_loaders:
                parts.append(
                    "unrecognized __loader__ name(s): "
                    + ", ".join(sorted(set(unknown_loaders)))
                )
            findings.append(Finding(
                rule_id="MFV-SKOPS-003",
                message=f"[{member}] skops schema carries {'; '.join(parts)}. "
                        "Likely a legitimate custom class or a newer protocol, but "
                        "not verifiable by static analysis -- loading this file "
                        "requires passing trusted= to skops.io.load, which is the "
                        "decision that needs a human.",
                severity=Severity.INFO,
                category=Category.DESERIALIZATION,
                file_path=str(file_path),
                confidence=0.4,
                cwe_ids=[502],
                metadata={
                    "unknown_types": sorted(set(unknown)),
                    "unknown_loaders": sorted(set(unknown_loaders)),
                },
            ))
        return findings


    # ── ONNX scanning ──────────────────────────────────────────────

    def _scan_onnx(self, file_path: Path, data: bytes) -> list[Finding]:
        """ONNX's real, previously-exploited RCE vector is a custom op
        (onnxruntime-extensions' PyOp/PythonOp) that embeds a Python callable
        to run at inference time; a secondary risk is a maliciously crafted
        `external_data` location string escaping the model directory. Both
        show up as plain string leaves somewhere in the model's protobuf
        tree -- walked schema-lessly (see _extract_protobuf_strings) rather
        than requiring the `onnx`/`protobuf` packages just to read op names.
        """
        findings: list[Finding] = []
        try:
            strings = _extract_protobuf_strings(data)
        except RecursionError:
            return [_skip_unverified_finding(
                file_path,
                "The protobuf walk blew the recursion limit on this file, so "
                "its op names and external_data strings were never extracted.",
                metadata={"skipped_reason": "recursion_limit"},
            )]

        dangerous_ops = sorted({s for s in strings if s in ONNX_DANGEROUS_OPS})
        if dangerous_ops:
            findings.append(Finding(
                rule_id="MFV-ONNX-001",
                message=f"ONNX model references custom op(s) with documented code-execution "
                        f"behavior: {', '.join(dangerous_ops)}. onnxruntime-extensions' PyOp/"
                        f"PythonOp embed a Python callable that runs at inference time.",
                severity=Severity.HIGH,
                category=Category.AI_ML,
                file_path=str(file_path),
                confidence=0.75,
                cwe_ids=[502, 94],
                metadata={"dangerous_ops": dangerous_ops},
            ))

        external_maps = _onnx_external_data_maps(data)
        # A location is meant to name a sibling file. A URL there makes the
        # loader fetch it, which is SSRF from inside a model: a published
        # proof of concept points one at 169.254.169.254, the cloud instance
        # metadata endpoint, to lift credentials during a scan or a load.
        remote_locations = sorted({
            location
            for entries in external_maps
            for location in [entries.get("location", "")]
            if _ONNX_REMOTE_LOCATION_RE.match(location)
        })
        if remote_locations:
            findings.append(Finding(
                rule_id="MFV-ONNX-004",
                message=f"ONNX external_data location(s) point off the filesystem: "
                        f"{', '.join(remote_locations[:5])}. The loader fetches "
                        f"what location names, so a URL here makes loading the "
                        f"model issue a request the operator never asked for.",
                severity=Severity.HIGH,
                category=Category.SSRF,
                file_path=str(file_path),
                confidence=0.85,
                cwe_ids=[918],
                metadata={"locations": remote_locations[:20]},
            ))

        bad_locations = sorted({
            location
            for entries in external_maps
            for location in [entries.get("location", "")]
            if ".." in location or _ONNX_ABSOLUTE_PATH_RE.match(location)
        })
        traversal_paths = sorted({
            s for s in strings
            if len(s) < 4096 and (
                s.startswith("../") or "/../" in s or s.startswith("..\\") or "\\..\\" in s
            )
        } | set(bad_locations))
        if traversal_paths:
            findings.append(Finding(
                rule_id="MFV-ONNX-002",
                message=f"ONNX model contains path-like string(s) with '..' traversal "
                        f"segments or absolute external_data locations: "
                        f"{', '.join(traversal_paths[:5])}. If referenced via "
                        f"external_data, this escapes the model's own directory when "
                        f"the weights are loaded.",
                severity=Severity.MEDIUM,
                category=Category.PATH_TRAVERSAL,
                file_path=str(file_path),
                confidence=0.5,
                cwe_ids=[22],
                metadata={"traversal_paths": traversal_paths[:20]},
            ))

        unexpected_keys = sorted({
            key
            for entries in external_maps
            for key in entries
            if key not in _ONNX_EXTERNAL_DATA_KEYS
        })
        if unexpected_keys:
            findings.append(Finding(
                rule_id="MFV-ONNX-003",
                message=f"ONNX external_data carries key(s) outside the format's "
                        f"four-key contract (location/offset/length/checksum): "
                        f"{', '.join(unexpected_keys[:5])}. Before onnx 1.21.0 "
                        f"(CVE-2026-34445) these keys reached setattr() on the "
                        f"ExternalDataInfo object, overwriting internal properties.",
                severity=Severity.HIGH,
                category=Category.AI_ML,
                file_path=str(file_path),
                confidence=0.75,
                cwe_ids=[502],
                metadata={"unexpected_keys": unexpected_keys[:20]},
            ))

        return findings

    # ── TensorFlow SavedModel scanning ─────────────────────────────

    def _scan_tf_savedmodel(self, file_path: Path, data: bytes) -> list[Finding]:
        """A SavedModel's GraphDef can embed ops that read/write files or
        invoke an embedded Python callable (PyFunc) at graph-execution time.
        Same schema-less protobuf walk as ONNX -- TF's GraphDef/NodeDef
        wire format is walked for op-name strings without needing the
        `tensorflow` package."""
        findings: list[Finding] = []
        try:
            strings = _extract_protobuf_strings(data)
        except RecursionError:
            return [_skip_unverified_finding(
                file_path,
                "The protobuf walk blew the recursion limit on this file, so "
                "its op names were never extracted.",
                metadata={"skipped_reason": "recursion_limit"},
            )]

        dangerous_ops = sorted({s for s in strings if s in TF_DANGEROUS_OPS})
        if dangerous_ops:
            findings.append(Finding(
                rule_id="MFV-TF-001",
                message=f"TensorFlow SavedModel graph references op(s) that read/write the "
                        f"filesystem or invoke an embedded Python callable at execution time: "
                        f"{', '.join(dangerous_ops)}.",
                severity=Severity.HIGH,
                category=Category.AI_ML,
                file_path=str(file_path),
                confidence=0.6,
                cwe_ids=[502, 94],
                metadata={"dangerous_ops": dangerous_ops},
            ))
        return findings

    # ── PMML scanning ──────────────────────────────────────────────
    #
    # PMML is XML. The earlier "data-only, no code execution" rejection
    # was correct for code execution and wrong as a threat model: XXE is a
    # file-read and SSRF primitive needing none. PMML engines additionally
    # support transformations via <Apply function="..."> whose function
    # name is an execution vector in some runtimes. Both are checked here;
    # the parser itself (defusedxml) forbids DOCTYPE and therefore XXE at
    # parse time, turning it into a positive signal.

    _PMML_NAMESPACE = "http://www.dmg.org/PMML-4_4"
    _PMML_DANGEROUS_FUNCTION_PATTERNS = (
        "exec(", "eval(", "__import__", "os.", "subprocess",
        "java.lang.Runtime", "java.lang.ProcessBuilder",
        "Runtime.getRuntime",
    )

    def _scan_pmml(self, file_path: Path, data: bytes) -> list[Finding]:
        findings: list[Finding] = []
        import defusedxml.ElementTree as ET
        try:
            root = ET.fromstring(data)
        except EntitiesForbidden as exc:
            findings.append(Finding(
                rule_id="MFV-PMML-001",
                message=f"PMML file contains an external entity declaration "
                        f"({exc.name!r} referencing {exc.sysid!r}). "
                        "XXE is a file-read and SSRF primitive that needs no "
                        "code execution.",
                severity=Severity.HIGH,
                category=Category.INJECTION,
                file_path=str(file_path),
                confidence=0.9,
                cwe_ids=[611, 918],
                metadata={"entity_name": exc.name, "system_id": exc.sysid},
            ))
            return findings
        except ET.ParseError as exc:
            # Exception-oriented evasion applies to XML as much as to pickle:
            # returning here reported a crafted PMML as clean. PMML engines
            # differ in strictness, so a document this parser rejects can still
            # be consumed downstream.
            findings.append(_skip_unverified_finding(
                file_path,
                f"The PMML document did not parse as XML ({exc}), so its "
                f"entity declarations and <Apply> functions were never checked.",
                metadata={"skipped_reason": "unparseable_xml"},
            ))
            return findings

        if not root.tag.endswith("PMML"):
            return findings

        dangerous: list[str] = []
        for elem in root.iter():
            func = elem.attrib.get("function", "")
            if func and any(pat in func for pat in self._PMML_DANGEROUS_FUNCTION_PATTERNS):
                dangerous.append(f"<{elem.tag.split('}')[-1]} function={func!r}>")

        if dangerous:
            findings.append(Finding(
                rule_id="MFV-PMML-002",
                message=f"PMML file references function name(s) associated with "
                        f"code execution: {', '.join(dangerous[:5])}. Some PMML "
                        f"engines evaluate <Apply> function attributes as code.",
                severity=Severity.HIGH,
                category=Category.AI_ML,
                file_path=str(file_path),
                confidence=0.7,
                cwe_ids=[94],
                metadata={"dangerous_functions": dangerous[:20]},
            ))
        return findings

    # ── 7z container scanning ───────────────────────────────────────

    _SEVENZ_MAGIC = b"7z\xBC\xAF\x27\x1C"
    _SEVENZ_MAX_TOTAL_BYTES = 500_000_000

    # 7z is the one container that re-enters scan_file on what it extracts, so
    # an archive containing an archive recurses. Measured with a stub extractor
    # that yielded a copy of its own input: RecursionError, which is an
    # unhandled crash rather than a verdict. Real checkpoints are never nested
    # this deep; four levels is generous.
    MAX_ARCHIVE_DEPTH = 4
    # The depth counter is instance state (initialised in __init__), not a
    # class attribute mutated through `self`: the class-attribute form makes
    # per-scan state look shared and survives a scan that dies before its
    # decrement only by accident of rebinding.

    def _scan_7z(self, file_path: Path) -> list[Finding]:
        """7z is an extraction problem, not a parse problem: py7zr is
        LGPL-2.1 (unabsorbable under the closed-source direction) and no
        model framework saves 7z, so the archive is read through a system
        7zz when one exists. Without one, the verdict is "present and
        unverifiable", never silent-clean: the same payload we flag CRITICAL
        in every .zip variant shipped as malicious1.7z in picklescan's own
        test suite and read as zero findings before this handler existed.
        """
        head = _read_file_magic(file_path, len(self._SEVENZ_MAGIC))
        if not head:
            return [_skip_unverified_finding(
                file_path,
                "The archive could not be opened, so its members were never listed.",
                metadata={"skipped_reason": "unreadable"},
            )]
        if head != self._SEVENZ_MAGIC:
            return []

        if self._archive_depth >= self.MAX_ARCHIVE_DEPTH:
            return [_skip_unverified_finding(
                file_path,
                f"Nested archives exceed the depth limit of "
                f"{self.MAX_ARCHIVE_DEPTH}, so this one was not extracted.",
                metadata={"skipped_reason": "archive_depth",
                          "depth": self._archive_depth},
            )]

        extractor = shutil.which("7zz") or shutil.which("7z")
        if extractor is None:
            return [Finding(
                rule_id="MFV-7Z-001",
                message="7z archive present, but no extractor is available to "
                        "inspect it (py7zr is LGPL-2.1 and cannot be bundled "
                        "here; a system 7zz would be used if installed). "
                        "Content could not be verified. NOT a clean verdict.",
                severity=Severity.INFO,
                category=Category.AI_ML,
                file_path=str(file_path),
                confidence=0.4,
                metadata={"skipped_reason": "no_extractor"},
            )]

        with tempfile.TemporaryDirectory() as tmp:
            try:
                listing = subprocess.run(  # noqa: S603 - fixed argv, no shell, "--" guards the path
                    # `--` stops switch parsing: a scanned file named
                    # "-o/tmp/x.7z" would otherwise reach 7-Zip as a flag.
                    [extractor, "l", "-slt", "--", str(file_path)],
                    capture_output=True, text=True, timeout=60, check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                listing = None
            if listing is not None and listing.returncode == 0:
                total = 0
                for line in listing.stdout.splitlines():
                    if not line.startswith("Size ="):
                        continue
                    # A non-numeric Size= line (attacker-chosen extractor
                    # output shape) raises ValueError; it counts as zero
                    # rather than aborting the scan.
                    with contextlib.suppress(ValueError):
                        total += int(line.split("=", 1)[1].strip() or 0)
            else:
                total = 0
            if total > self._SEVENZ_MAX_TOTAL_BYTES:
                return [Finding(
                    rule_id="MFV-7Z-001",
                    message=f"7z archive claims {total // 1_000_000}MB extracted; "
                            "over the extraction cap. Content could not be verified. "
                            "NOT a clean verdict.",
                    severity=Severity.LOW,
                    category=Category.AI_ML,
                    file_path=str(file_path),
                    confidence=0.5,
                    metadata={"skipped_reason": "oversized", "claimed_total": total},
                )]
            try:
                proc = subprocess.run(  # noqa: S603 - fixed argv, no shell, "--" guards the path
                    [extractor, "x", "-y", f"-o{tmp}", "--", str(file_path)],
                    capture_output=True, text=True, timeout=300, check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                return [_skip_unverified_finding(
                    file_path,
                    f"7z extraction failed ({type(exc).__name__}).",
                    metadata={"skipped_reason": "extract_failed"},
                )]
            if proc.returncode != 0:
                return [_skip_unverified_finding(
                    file_path,
                    "7z extraction failed.",
                    metadata={"skipped_reason": "extract_failed",
                              "stderr": proc.stderr[:200]},
                )]

            findings: list[Finding] = []
            tmp_root = Path(tmp).resolve()
            self._archive_depth += 1
            try:
                for extracted in sorted(Path(tmp).rglob("*")):
                    # A 7z member can be a symlink, and the extractor will
                    # happily create it. Following one reads a file outside the
                    # extraction directory and prints its contents into the
                    # report, which turns a scan into an arbitrary-file-read
                    # primitive for whoever supplied the archive (CWE-59/22).
                    # Only regular files that resolve inside tmp are scanned.
                    try:
                        resolved = extracted.resolve()
                    except OSError:
                        continue
                    if extracted.is_symlink() or resolved != extracted:
                        if tmp_root not in resolved.parents:
                            findings.append(_skip_unverified_finding(
                                file_path,
                                f"Archive member {extracted.name!r} is a link "
                                f"pointing outside the extraction directory, so "
                                f"it was not followed.",
                                metadata={"skipped_reason": "link_escape",
                                          "member": extracted.name},
                            ))
                            continue
                    if not resolved.is_file():
                        continue
                    member = str(extracted.relative_to(tmp))
                    for f in self.scan_file(extracted):
                        f.message = f"[{member}] {f.message}"
                        f.metadata = {**(f.metadata or {}), "7z_member": member}
                        findings.append(f)
            finally:
                self._archive_depth -= 1
            return findings

    # ── TFLite scanning ────────────────────────────────────────────

    def _scan_tflite(self, file_path: Path, data: bytes) -> list[Finding]:
        """TFLite executes no code on load, but its parsers allocate from
        attacker-chosen tensor dimensions. CVE-2026-42627 (ArmNN, 2026-05)
        is the shape: dimensions multiplied in 32-bit without overflow
        detection understate the allocation and the layer then reads past
        it. Replayed here in ints that do not wrap."""
        if len(data) < 8 or data[4:8] != _TFLITE_MAGIC:
            return []
        problems = _check_tflite_layout(data)
        if not problems:
            return []
        return [Finding(
            rule_id="MFV-TFLITE-001",
            message=f"TFLite tensor dimensions are inconsistent with a 32-bit "
                    f"loader: {'; '.join(problems[:5])}. CVE-2026-42627: a "
                    f"crafted model's dimension product wraps ArmNN's 32-bit "
                    f"arithmetic and the layer reads past the allocation.",
            severity=Severity.HIGH,
            category=Category.AI_ML,
            file_path=str(file_path),
            confidence=0.75,
            cwe_ids=[190, 125],
            metadata={"layout_problems": problems[:20]},
        )]

    # ── numpy .npy/.npz scanning ───────────────────────────────────

    _NPY_MAGIC = b"\x93NUMPY"

    # Cap on header bytes handed to ast.literal_eval. A v2 header_len is a
    # u32 (up to 4GB) and the whole file can sit under the 500MB scan cap;
    # real .npy headers are ~100 bytes, so 1MB is four orders of magnitude
    # of headroom. Over-cap is treated as an unparseable header, the same
    # verdict as any other header literal_eval rejects.
    _NPY_MAX_HEADER_BYTES = 1 << 20

    def _parse_npy_header(self, data: bytes) -> tuple[dict[str, Any], int] | None:
        """Parse a .npy file's header dict and return (header, data_offset).

        The header is a Python dict *literal* (not executable code) --
        ast.literal_eval evaluates only literal containers/constants, so
        this can't itself become a code-execution path even though it's
        reading attacker-controlled bytes.
        """
        if data[:6] != self._NPY_MAGIC or len(data) < 10:
            return None
        major = data[6]
        if major == 1:
            if len(data) < 10:
                return None
            header_len = struct.unpack("<H", data[8:10])[0]
            header_start = 10
        else:
            if len(data) < 12:
                return None
            header_len = struct.unpack("<I", data[8:12])[0]
            header_start = 12
        if header_len > self._NPY_MAX_HEADER_BYTES:
            return None
        header_end = header_start + header_len
        if header_end > len(data):
            return None
        try:
            header = ast.literal_eval(data[header_start:header_end].decode("latin1").strip())
        except (ValueError, SyntaxError, UnicodeDecodeError, MemoryError,
                RecursionError):
            # MemoryError/RecursionError: the header literal is
            # attacker-shaped (a 1MB nest of braces reaches both) and the
            # verdict for it is the same as for any unparseable header.
            return None
        if not isinstance(header, dict):
            return None
        return header, header_end

    def _scan_npy_bytes(self, file_path: Path, data: bytes, member: str = "") -> list[Finding]:
        parsed = self._parse_npy_header(data)
        if parsed is None:
            return []
        header, data_offset = parsed
        descr = str(header.get("descr", ""))
        if "O" not in descr:
            # Numeric/fixed-width dtype -- the data section is raw array
            # bytes, not a pickle stream; nothing to scan.
            return []

        findings = self._scan_pickle(file_path, data[data_offset:])
        if member:
            for f in findings:
                f.message = f"[{member}] {f.message}"
                f.metadata = {**(f.metadata or {}), "zip_member": member}
        return findings

    # Zip-slip discipline for .npz (and any model bundle treated as a zip):
    # numpy only ever writes flat "name.npy" members, so member names that
    # escape the archive root, duplicate each other after normalization, or
    # carry symlink bits are never legitimate. Ratios bound the decompression
    # side: DEFLATE's worst case is ~1000:1, and a member claiming a large
    # decompressed size from a tiny compressed blob is the classic zip bomb.
    _ZIP_BOMB_MIN_MEMBER_BYTES = 1_000_000
    _ZIP_BOMB_MAX_RATIO = 100

    def _check_npz_container(self, zf: zipfile.ZipFile, file_path: Path) -> list[Finding]:
        problems: list[str] = []
        seen_names: dict[str, str] = {}
        for info in zf.infolist():
            # orig_filename, not filename: ZipInfo.__init__ rewrites os.sep to
            # "/" while reading the central directory, so on Windows a
            # backslash member arrives already laundered into a plain path and
            # the checks below cannot see it. orig_filename is the raw name as
            # the archive stores it, on every platform.
            name = info.orig_filename
            segments = name.replace("\\", "/").split("/")
            normalized = "/".join(s for s in segments if s not in ("", "."))
            if (
                name.startswith(("/", "\\"))
                or re.match(r"^[A-Za-z]:[\\/]", name)
                or ".." in segments
            ):
                problems.append(f"member {name!r} escapes the archive root")
            first = seen_names.get(normalized)
            if first is not None and first != name:
                problems.append(
                    f"members {first!r} and {name!r} normalize to the same "
                    f"name ({normalized!r}), so which one a reader gets is "
                    f"parser-dependent"
                )
            else:
                seen_names.setdefault(normalized, name)
            # Unix mode bits in the high half of external_attr: a symlink
            # member extracts as a link, not a file.
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                problems.append(f"member {name!r} is a symlink entry")
            if (
                info.compress_type != zipfile.ZIP_STORED
                and info.compress_size > 0
                and info.file_size >= self._ZIP_BOMB_MIN_MEMBER_BYTES
                and info.file_size / info.compress_size > self._ZIP_BOMB_MAX_RATIO
            ):
                problems.append(
                    f"member {name!r} claims {info.file_size} bytes from "
                    f"{info.compress_size} compressed "
                    f"({info.file_size // max(info.compress_size, 1)}:1)"
                )
        if not problems:
            return []
        return [Finding(
            rule_id="MFV-NPZ-001",
            message=f"NPZ container violates zip-safety discipline: "
                    f"{'; '.join(problems[:5])}. Extracting this archive can "
                    f"write outside its target directory or exhaust disk/memory.",
            severity=Severity.MEDIUM,
            category=Category.PATH_TRAVERSAL,
            file_path=str(file_path),
            confidence=0.6,
            cwe_ids=[22],
            metadata={"problems": problems[:20]},
        )]

    def _scan_numpy(self, file_path: Path, data: bytes) -> list[Finding]:
        """`numpy.load(..., allow_pickle=True)` on an object-dtype array is a
        pickle load in disguise -- numpy serializes an object-dtype array's
        elements via pickle.dump, so the .npy data section past the header
        is a real pickle stream once descr indicates dtype=object."""
        if file_path.suffix.lower() == ".npz":
            findings: list[Finding] = []
            try:
                zf = zipfile.ZipFile(file_path, "r")
            except (zipfile.BadZipFile, OSError):
                # Not a zip at all: the extension-confusion check in
                # scan_file owns that case for .npz.
                return []
            try:
                with zf:
                    findings.extend(self._check_npz_container(zf, file_path))
                    for name in zf.namelist():
                        if not name.endswith(".npy"):
                            continue
                        try:
                            member_data = self._read_zip_member_capped(zf, name, self.MAX_ZIP_MEMBER_BYTES)
                        except (OSError, zipfile.BadZipFile, RuntimeError, ValueError):
                            continue
                        if member_data is None:
                            continue
                        findings.extend(self._scan_npy_bytes(file_path, member_data, member=name))
            except (zipfile.BadZipFile, OSError) as exc:
                # The walk died partway. Returning [] would discard the
                # container check and the members already scanned.
                findings.append(Finding(
                    rule_id="MFV-SKIP-002",
                    message=f"Zip container walk ended early ({type(exc).__name__}), so any "
                            f"members past the failure point were never analysed. NOT a clean "
                            f"verdict for the remainder of the archive.",
                    severity=Severity.LOW,
                    category=Category.AI_ML,
                    file_path=str(file_path),
                    confidence=0.50,
                    metadata={"skipped_reason": "zip_walk_failed", "error": type(exc).__name__},
                ))
            return findings
        return self._scan_npy_bytes(file_path, data)

    # ── joblib scanning ────────────────────────────────────────────

    def _decompress_capped(self, data: bytes, max_bytes: int,
                           codec: str = "zlib") -> bytes | None:
        """Bounded decompression for any codec joblib writes.

        Same contract as the zlib-only version it generalises: chunked, and
        None once the output would exceed the cap, so a small archive cannot
        expand into a memory-exhausting payload.
        """
        decompressor = _new_decompressor(codec)
        out = bytearray()
        step = 1 << 20
        for offset in range(0, len(data), step):
            chunk = data[offset:offset + step]
            produced = decompressor.decompress(chunk, max_bytes - len(out) + 1)
            out.extend(produced)
            if len(out) > max_bytes:
                return None
            if getattr(decompressor, "eof", False):
                break
        return bytes(out)

    def _decompress_zlib_capped(self, data: bytes, max_bytes: int) -> bytes | None:
        """Inflate a raw zlib stream in bounded chunks, aborting once
        max_bytes is exceeded.

        Mirrors ``_read_zip_member_capped``: DEFLATE's declared/compressed
        size can't be trusted as a proxy for its decompressed size (worst-case
        expansion is roughly 1000:1+), so a naive ``zlib.decompress(data)``
        call can be handed a small, well-under-the-file-size-cap payload that
        still exhausts memory on load. ``decompressobj.decompress(buf,
        max_length)`` only inflates up to ``max_length`` bytes per call and
        leaves the rest of the not-yet-processed input in
        ``unconsumed_tail`` -- feed that back in on the next call and stop as
        soon as the running total crosses the cap, instead of ever
        materializing the full inflated buffer.
        """
        decompressor = zlib.decompressobj()
        chunks: list[bytes] = []
        total = 0
        chunk_size = 1_000_000
        pending = data
        while True:
            out = decompressor.decompress(pending, chunk_size)
            total += len(out)
            if total > max_bytes:
                return None
            chunks.append(out)
            pending = decompressor.unconsumed_tail
            if not pending:
                break
        tail = decompressor.flush()
        total += len(tail)
        if total > max_bytes:
            return None
        chunks.append(tail)
        return b"".join(chunks)

    def _scan_joblib(self, file_path: Path, data: bytes) -> list[Finding]:
        """joblib.dump() without compression writes a plain pickle stream;
        with compress= set, it wraps a zlib-compressed pickle behind a zlib
        magic header. Both routes ultimately end in a real pickle stream,
        so both are handed to the same hardened callable-resolution scanner
        pickle.loads() itself uses -- this doesn't attempt joblib's other
        compressor backends (lz4, blosc) or its large-array memmap chunking,
        best-effort per issue #77's own scope.

        Decompression is bounded the same way `_read_zip_member_capped`
        bounds zip-backed formats (DEF-33): a small, well-under-the-file-size
        -cap compressed stream can still expand into a memory-exhausting
        payload, so this never calls the unbounded `zlib.decompress`.
        """
        codec = _sniff_compression(data)
        if codec is not None:
            try:
                decompressed = self._decompress_capped(
                    data, self.MAX_ZIP_MEMBER_BYTES, codec)
            except (zlib.error, OSError, EOFError, lzma.LZMAError):
                pass
            else:
                if decompressed is None:
                    return [Finding(
                        rule_id="MFV-JOBLIB-002",
                        message=f"joblib file's {codec}-compressed payload exceeds "
                                f"{self.MAX_ZIP_MEMBER_BYTES // 1_000_000}MB decompressed size "
                                f"limit -- possible decompression bomb. Skipping for memory safety.",
                        severity=Severity.HIGH,
                        category=Category.AI_ML,
                        file_path=str(file_path),
                        confidence=0.6,
                        metadata={"skipped_reason": "oversized_decompressed"},
                    )]
                return self._scan_pickle(file_path, decompressed)
        return self._scan_pickle(file_path, data)
