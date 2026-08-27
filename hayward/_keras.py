"""Keras model_config extraction (HW-147 split out of scanner.py).

Keras stores the model architecture as a literal UTF-8 JSON attribute value
embedded in the HDF5 file. Rather than a full HDF5 parser (a new dependency),
locate the JSON blob directly and parse it, precise enough to walk the actual
layer graph instead of substring-matching the raw binary container (which also
contains tensor names and weight bytes that can spuriously contain words like
"lambda"/"function").

Self-contained: this module needs nothing from scanner.py. scanner.py imports
these names back so `from hayward.scanner import _extract_keras_model_config`
(and the rest) keeps resolving for the tests and the scan methods.
"""

from __future__ import annotations

import json
from typing import Any

_KERAS_BUILTIN_LAYER_CLASSES = frozenset({
    "Sequential", "Functional", "Model",
    "InputLayer", "Input",
    "Dense", "Activation", "Dropout", "Flatten", "Reshape", "Permute",
    "RepeatVector", "Masking", "Embedding",
    "Conv1D", "Conv2D", "Conv3D", "Conv1DTranspose", "Conv2DTranspose",
    "Conv3DTranspose", "SeparableConv1D", "SeparableConv2D",
    "DepthwiseConv2D", "Cropping1D", "Cropping2D", "Cropping3D",
    "UpSampling1D", "UpSampling2D", "UpSampling3D",
    "ZeroPadding1D", "ZeroPadding2D", "ZeroPadding3D",
    "MaxPooling1D", "MaxPooling2D", "MaxPooling3D",
    "AveragePooling1D", "AveragePooling2D", "AveragePooling3D",
    "GlobalMaxPooling1D", "GlobalMaxPooling2D", "GlobalMaxPooling3D",
    "GlobalAveragePooling1D", "GlobalAveragePooling2D", "GlobalAveragePooling3D",
    "LSTM", "GRU", "SimpleRNN", "Bidirectional", "TimeDistributed",
    "ConvLSTM1D", "ConvLSTM2D", "ConvLSTM3D",
    "BatchNormalization", "LayerNormalization", "GroupNormalization",
    "SpatialDropout1D", "SpatialDropout2D", "SpatialDropout3D",
    "GaussianNoise", "GaussianDropout", "AlphaDropout",
    "Add", "Subtract", "Multiply", "Average", "Maximum", "Minimum",
    "Concatenate", "Dot",
    "Attention", "AdditiveAttention", "MultiHeadAttention",
    "LeakyReLU", "PReLU", "ELU", "ReLU", "Softmax", "ThresholdedReLU",
    "TextVectorization", "Normalization",
    "Rescaling", "Resizing", "CenterCrop",
    # Not layers at all, but standard entries in a Keras 3 model_config's
    # optimizer/loss/metrics/initializer sections, all inert at load:
    # optimizers.
    "Adam", "AdamW", "SGD", "RMSprop", "Adagrad", "Adadelta", "Adamax",
    "Nadam", "Lion", "Lamb", "Ftrl",
    # Losses and metrics.
    "BinaryCrossentropy", "CategoricalCrossentropy",
    "SparseCategoricalCrossentropy", "MeanSquaredError", "MeanAbsoluteError",
    "MeanAbsolutePercentageError", "Huber", "KLDivergence", "CosineSimilarity",
    "FocalLoss", "Hinge", "SquaredHinge", "Poisson", "LogCosh",
    "Accuracy", "BinaryAccuracy", "CategoricalAccuracy",
    "SparseCategoricalAccuracy", "AUC", "Precision", "Recall", "F1Score",
    "PrecisionAtRecall", "RecallAtPrecision", "Mean", "Sum", "R2Score",
    "TopKCategoricalAccuracy", "SparseTopKCategoricalAccuracy",
    # Initializers and the dtype policy object.
    "GlorotUniform", "GlorotNormal", "HeNormal", "HeUniform", "LecunNormal",
    "LecunUniform", "Ones", "Zeros", "RandomNormal", "RandomUniform",
    "TruncatedNormal", "Orthogonal", "Identity", "Constant", "VarianceScaling",
    "DTypePolicy",
    # Keras 3's internal tensor marker in functional graphs.
    "__keras_tensor__",
})


# Streaming search for the Keras architecture blob in a file too large to hold
# in memory. The window is sized just past _extract_keras_model_config's own
# 20MB balanced-brace scan limit, so a config it could parse from a full read
# is equally parseable from the window.
_KERAS_CONFIG_ANCHOR = b'"class_name"'
_KERAS_STREAM_CHUNK = 8 * 1024 * 1024
_KERAS_CONFIG_WINDOW = 24_000_000
_KERAS_CONFIG_BACKTRACK = 4096

# Bound on how many `"class_name"` anchors _extract_keras_model_config tries.
# The attacker controls HDF5 attribute order/content, so a benign decoy object
# placed before the real config used to hide a Lambda layer; the extractor
# now walks subsequent anchors until it finds risky layers. 64 is far beyond
# any real model's count of top-level config objects, and the total scan work
# stays inside the pre-existing 20MB budget either way.
_KERAS_MAX_CONFIG_ANCHORS = 64


def _read_keras_config_window(path) -> bytes | None:
    """Locate the `model_config` blob in an oversized HDF5 file by streaming.

    Returns a bounded window containing the anchor, positioned far enough back
    that the caller's `rfind(b"{")` still finds the opening brace, or None when
    the anchor never appears (a weights-only file has no architecture to check).
    Peak memory is one chunk plus one window, never the file.
    """
    overlap = len(_KERAS_CONFIG_ANCHOR) - 1
    try:
        with open(path, "rb") as f:
            pos = 0          # file offset of buf[0]
            tail = b""
            while True:
                chunk = f.read(_KERAS_STREAM_CHUNK)
                if not chunk:
                    return None
                buf = tail + chunk
                idx = buf.find(_KERAS_CONFIG_ANCHOR)
                if idx != -1:
                    start = max(0, pos + idx - _KERAS_CONFIG_BACKTRACK)
                    f.seek(start)
                    return f.read(_KERAS_CONFIG_WINDOW)
                keep = min(len(buf), overlap)
                pos += len(buf) - keep
                tail = buf[len(buf) - keep:] if keep else b""
    except OSError:
        return None


def _balanced_brace_json_end(data: bytes, start: int, limit: int) -> int | None:
    """Index just past the balanced `{...}` object opening at `start`, or
    None if the braces never balance within `limit`. Quote-aware: braces
    inside JSON strings do not count."""
    depth = 0
    in_string = False
    escape = False
    for i in range(start, limit):
        byte = data[i:i + 1]
        if in_string:
            if escape:
                escape = False
            elif byte == b"\\":
                escape = True
            elif byte == b'"':
                in_string = False
            continue
        if byte == b'"':
            in_string = True
        elif byte == b"{":
            depth += 1
        elif byte == b"}":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def _extract_keras_model_config(data: bytes) -> dict | None:
    """Best-effort extraction of Keras's `model_config` JSON blob from a raw
    HDF5 byte stream: locate a `"class_name"`-anchored JSON object and parse
    it with a quote-aware balanced-brace scan. Returns None if no valid JSON
    model config is found (e.g. a weights-only H5 file with no architecture
    attribute -- nothing to check).

    Iterates subsequent anchors (bounded) rather than stopping at the first:
    the attacker controls HDF5 attribute order/content, and a benign decoy
    object placed before the real config used to hide a Lambda layer. The
    first parsed config carrying risky layers wins; otherwise the first
    parseable config is returned so the unrecognized-class check still sees
    what the old single-anchor behaviour saw.
    """
    marker = b'"class_name"'
    idx = data.find(marker)
    first_parsed: dict | None = None
    attempts = 0
    # Total balanced-brace scan budget across all anchors. Equals the old
    # single-anchor limit, so one anchor behaves exactly as before and a
    # file of near-miss anchors cannot multiply the work.
    scan_budget = 20_000_000
    while idx != -1 and attempts < _KERAS_MAX_CONFIG_ANCHORS and scan_budget > 0:
        attempts += 1
        start = data.rfind(b"{", 0, idx)
        if start != -1:
            scan_limit = min(len(data), start + scan_budget)
            end = _balanced_brace_json_end(data, start, scan_limit)
            scan_budget -= (end - start) if end is not None else (scan_limit - start)
            if end is not None:
                try:
                    parsed = json.loads(data[start:end].decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError, RecursionError,
                        ValueError, MemoryError):
                    # Deeply nested JSON raises RecursionError/MemoryError
                    # rather than JSONDecodeError; same catch set as the
                    # skops schema path.
                    parsed = None
                if isinstance(parsed, dict):
                    if first_parsed is None:
                        first_parsed = parsed
                    if _find_keras_risky_layers(parsed):
                        return parsed
        idx = data.find(marker, idx + 1)
    return first_parsed


def _find_keras_risky_layers(config: Any) -> list[str]:
    """Recursively walk a parsed Keras model_config, returning human-readable
    descriptions of Lambda layers / unrecognized layer classes actually
    present in the layer graph -- not a substring match. A Lambda layer's
    config embeds a marshalled/base64-encoded Python function that executes
    on load, so its presence alone (regardless of what it does) is the
    signal; an unrecognized (non-builtin) class name is weaker evidence
    (most are legitimate custom architectures) and reported separately.
    """
    risky: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            class_name = node.get("class_name")
            if class_name == "Lambda":
                layer_name = (node.get("config") or {}).get("name", "unnamed")
                risky.append(f"Lambda layer '{layer_name}'")
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(config)
    # A config can hold the same layer graph more than once -- a SavedModel's
    # `keras_metadata.pb` stores it under both `config` and `model_config` --
    # and counting one Lambda layer twice overstates the finding. Layer names
    # are unique within a Keras model, so collapsing identical descriptions
    # only ever merges re-serialized copies of one layer, never two real ones.
    deduped: list[str] = []
    for item in risky:
        if item not in deduped:
            deduped.append(item)
    return deduped


def _find_keras_unrecognized_classes(config: Any) -> list[str]:
    """Same walk as _find_keras_risky_layers but collects layer class names
    not on the known-builtin list (excluding Lambda, handled separately) --
    weaker, informational-only evidence."""
    unrecognized: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            class_name = node.get("class_name")
            if (
                isinstance(class_name, str)
                and class_name != "Lambda"
                and class_name not in _KERAS_BUILTIN_LAYER_CLASSES
            ):
                unrecognized.add(class_name)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(config)
    return sorted(unrecognized)
