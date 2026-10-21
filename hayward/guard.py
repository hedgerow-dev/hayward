"""Load guard: scan a model file at the moment something loads it.

    import hayward.guard  # one line, top of a notebook or script

From then on `pickle.load`, `joblib.load`, `torch.load` and
`numpy.load(allow_pickle=True)` scan their input first and raise
`UnsafeModelError` instead of loading anything at or above the threshold.
`pickle.loads` is left alone: it is an in-memory IPC and cache hot path, not
a model-file load.
`install(fail_on=..., mode=...)` changes the settings, `uninstall()` restores
the original loaders, and `HAYWARD_GUARD=off` makes the import a no-op.

Coverage findings (MFV-SKIP-* and the rest of COVERAGE_RULE_IDS) block too,
unlike the CLI, where failing on them is opt-in: the guard runs the file after
scanning it, so a file it could not fully scan is not loaded. That includes
bytes the loader would unpickle that the scanner could not parse at all. Warn
mode warns and loads.

A file is scanned as the format of the loader reading it, not as its name
says, since the loaders themselves decide by content: a pickle named `.pth`
is scanned as a pickle by numpy.load, and a gzip file named `.pkl` as a
compressed joblib file by joblib.load.

Only the module attributes are patched. Code that bound a loader before the
guard was installed (`from pickle import load`), direct `pickle.Unpickler`
use, `dill` and `cloudpickle` are not covered.
"""

from __future__ import annotations

import functools
import io
import os
import pickle
import shutil
import sys
import tempfile
import threading
import warnings
from dataclasses import replace
from pathlib import Path

from hayward.cli import _THRESHOLDS, _render_text
from hayward.findings import Finding, is_coverage_gap
from hayward.scanner import ModelFileScanner

# module -> (attribute, name of the source parameter, suffix to scan as). The
# suffix tells the scanner which format the loader will read; numpy's depends
# on the first bytes, see _numpy_suffix.
_TARGETS = {
    "pickle": [("load", "file", ".pkl")],
    "joblib": [("load", "filename", ".joblib")],
    "numpy": [("load", "file", None)],
    "torch": [("load", "f", ".pt")],
}

_limit = _THRESHOLDS["high"]
_mode = "block"
_max_size: int | None = None  # None keeps the scanner's own 500MB cap
_patched: list[tuple[object, str, object]] = []
# Per process, keyed on (path, suffix, size, mtime_ns), so a training loop
# that reloads a checkpoint does not rescan it every time. numpy's suffix is
# None here: it is read from the file on a miss.
_memo: dict[tuple[str, str | None, int, int], list[Finding]] = {}
# numpy, joblib and torch call pickle.load (and joblib numpy.load) themselves.
# `_local.sources` is the stack of what the guarded calls running on this
# thread scanned: resolved paths, and the stream objects handed to loaders.
# A nested call is skipped only when it rereads one of those. joblib's
# NDArrayWrapper names a second file inside the pickle, and that load is
# scanned like any other.
_local = threading.local()


class UnsafeModelError(pickle.UnpicklingError):
    """Raised instead of loading a file the scan flagged.

    A subclass of `pickle.UnpicklingError`, so an existing `except` around a
    load treats it as a failed load.
    """

    def __init__(self, file: str, rule_ids: list[str], message: str):
        super().__init__(message)
        self.file = file
        self.rule_ids = rule_ids


def _scanner() -> ModelFileScanner:
    scanner = ModelFileScanner()
    if _max_size is not None:
        scanner.MAX_SCAN_BYTES = _max_size  # per instance, as `--max-size` does
    # The loader runs next: bytes it would unpickle that the scanner could not
    # parse are a coverage gap here, not the silence the CLI gives them.
    scanner.report_unparsed_pickles = True
    return scanner


def _numpy_suffix(head: bytes) -> str:
    # numpy.load decides by the first bytes, not the name: a zip is an .npz,
    # the .npy magic an array, and anything else goes to pickle.load.
    if head.startswith((b"PK\x03\x04", b"PK\x05\x06")):
        return ".npz"
    if head.startswith(b"\x93NUMPY"):
        return ".npy"
    return ".pkl"


def _scan_path(path: str, suffix: str | None) -> list[Finding]:
    try:
        st = os.stat(path)
        key = (path, suffix, st.st_size, st.st_mtime_ns)
        if key not in _memo:
            if suffix is None:
                with open(path, "rb") as f:
                    suffix = _numpy_suffix(f.read(6))
            _memo[key] = _scanner().scan_file(Path(path), _as_suffix=suffix)
    except OSError:
        return []  # the loader raises its own error for a missing file
    return _memo[key]


def _scan_stream(stream, suffix: str | None, label: str) -> list[Finding]:
    """Copy the rest of `stream` to a private temp file and scan that."""
    if suffix is None:
        start = stream.tell()
        suffix = _numpy_suffix(stream.read(6))
        stream.seek(start)
    fd, tmp = tempfile.mkstemp(prefix="hayward-guard-", suffix=suffix)  # mode 0600
    try:
        with os.fdopen(fd, "wb") as out:
            shutil.copyfileobj(stream, out)
        findings = _scanner().scan_file(Path(tmp))
    finally:
        os.unlink(tmp)
    return [replace(f, file_path=label) for f in findings]


def _resolve(path) -> str:
    return os.path.realpath(os.fsdecode(path))


def _rereads_scanned_source(source, scanned: list) -> bool:
    """Whether a nested load reads what an enclosing guarded call scanned.

    That is the same path, the same stream object, or a stream whose name is
    a path that was scanned. A stream is followed through the wrappers a
    library puts around it (io.BufferedReader's `raw`, the `_fp` of joblib's
    and the standard library's decompressing readers) back to the file it
    reads, which is how joblib hands an object array to pickle.load.
    """
    paths = {s for s in scanned if isinstance(s, str)}
    if isinstance(source, (str, bytes, os.PathLike)):
        return _resolve(source) in paths
    for _ in range(4):
        if source is None:
            return False
        if any(source is s for s in scanned):
            return True
        name = getattr(source, "name", None)
        if isinstance(name, (str, bytes, os.PathLike)) and _resolve(name) in paths:
            return True
        source = getattr(source, "raw", None) or getattr(source, "_fp", None)
    return False


def _check(source, suffix: str | None, takes_path: bool):
    """Scan what a loader is about to read.

    Returns what the loader should read, which is `source` itself except for
    a non-seekable stream (read into memory so the loader gets the bytes the
    scan saw), and what was scanned, for the nested-call check.
    """
    if takes_path and isinstance(source, (str, bytes, os.PathLike)):
        path = _resolve(source)
        _enforce(_scan_path(path, suffix), os.fsdecode(source))
        return source, path
    if not hasattr(source, "read"):
        return source, None  # not a stream either: the loader raises its own error
    label = str(getattr(source, "name", "<stream>"))
    seekable = getattr(source, "seekable", None)
    if seekable is None or not seekable():
        buffer = io.BytesIO()
        shutil.copyfileobj(source, buffer)
        buffer.seek(0)
        source = buffer
    start = source.tell()
    try:
        findings = _scan_stream(source, suffix, label)
    finally:
        source.seek(start)
    _enforce(findings, label)
    return source, source


def _enforce(findings: list[Finding], label: str) -> None:
    gaps = [f for f in findings if is_coverage_gap(f)]
    hits = [f for f in findings if not is_coverage_gap(f) and f.severity_order <= _limit]
    parts = []
    if hits:
        parts.append(f"hayward flagged {label}:\n" + _render_text(hits, None, False))
    if gaps:
        parts.append(f"hayward could not fully scan {label}:\n" + _render_text(gaps, None, False))
    if not parts:
        return
    message = "\n\n".join(parts)
    if _mode == "block":
        message += "\nRefused to load it."
        if gaps:
            message += ' hayward.guard.install(mode="warn") loads it anyway.'
        raise UnsafeModelError(label, sorted({f.rule_id for f in hits + gaps}), message)
    # stacklevel 4 points the warning at the caller of the patched loader.
    warnings.warn(message + "\nLoading it anyway (warn mode).", stacklevel=4)


def _wrap(module_name: str, original, param: str, suffix: str | None):
    takes_path = module_name != "pickle"

    @functools.wraps(original)
    def guarded(*args, **kwargs):
        if module_name == "numpy":
            # numpy.load(file, mmap_mode, allow_pickle, ...): without
            # allow_pickle it refuses object arrays, so there is nothing to scan.
            if not kwargs.get("allow_pickle", args[2] if len(args) > 2 else False):
                return original(*args, **kwargs)
        if not args and param not in kwargs:
            return original(*args, **kwargs)  # the loader raises its own error
        scanned = _local.__dict__.setdefault("sources", [])
        source = args[0] if args else kwargs[param]
        if scanned and _rereads_scanned_source(source, scanned):
            return original(*args, **kwargs)
        source, key = _check(source, suffix, takes_path)
        if args:
            args = (source, *args[1:])
        else:
            kwargs[param] = source
        scanned.append(key)
        try:
            return original(*args, **kwargs)
        finally:
            scanned.pop()

    return guarded


def _patch(module) -> None:
    for attr, param, suffix in _TARGETS[module.__name__]:
        original = getattr(module, attr, None)
        if original is not None:
            setattr(module, attr, _wrap(module.__name__, original, param, suffix))
            _patched.append((module, attr, original))


class _ImportHook:
    """Patches joblib, numpy or torch when it is first imported, so the guard
    never imports a heavy framework itself."""

    def find_spec(self, name, path, target=None):
        if name not in _TARGETS:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(name, path, target)
            if spec is not None:
                break
        else:
            return None
        loader = spec.loader
        if loader is None or not hasattr(loader, "exec_module"):
            return spec
        exec_module = loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            if _hook in sys.meta_path:  # still installed, e.g. on a later reload
                _patch(module)

        loader.exec_module = exec_and_patch
        return spec


_hook = _ImportHook()


def install(fail_on: str = "high", mode: str = "block", max_size: int | None = None) -> None:
    """Patch the supported loaders. Calling it again replaces the settings.

    `fail_on` uses the CLI's `--fail-on` scale. Coverage gaps block at every
    threshold, `never` included, because the loader runs the file next.
    `mode="warn"` reports through `warnings.warn` and lets the load go ahead.
    `max_size` raises the per-file scan cap in bytes, like `--max-size`, so a
    large non-zip file can be scanned instead of refused as unread.
    """
    global _limit, _mode, _max_size
    if fail_on not in _THRESHOLDS:
        raise ValueError(f"fail_on must be one of {', '.join(_THRESHOLDS)}")
    if mode not in ("block", "warn"):
        raise ValueError("mode must be 'block' or 'warn'")
    if max_size is not None and max_size <= 0:
        raise ValueError("max_size must be a positive byte count")
    uninstall()
    _limit, _mode, _max_size = _THRESHOLDS[fail_on], mode, max_size
    _memo.clear()  # a different cap can change a file's findings
    for name in _TARGETS:
        if name in sys.modules:
            _patch(sys.modules[name])
    sys.meta_path.insert(0, _hook)


def uninstall() -> None:
    """Restore the original loaders and stop patching new imports."""
    while _patched:
        module, attr, original = _patched.pop()
        setattr(module, attr, original)
    if _hook in sys.meta_path:
        sys.meta_path.remove(_hook)


if os.environ.get("HAYWARD_GUARD", "").lower() != "off":
    install()
