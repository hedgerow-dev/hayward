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
scanning it, so a file it could not fully scan is not loaded. Warn mode
warns and loads.

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

# module -> (attribute, name of the source parameter, suffix for a spooled
# copy). The suffix tells the scanner which format the loader will read.
_TARGETS = {
    "pickle": [("load", "file", ".pkl")],
    "joblib": [("load", "filename", ".joblib")],
    "numpy": [("load", "file", ".npy")],
    "torch": [("load", "f", ".pt")],
}

_limit = _THRESHOLDS["high"]
_mode = "block"
_patched: list[tuple[object, str, object]] = []
# Per process, keyed on (path, size, mtime_ns), so a training loop that
# reloads a checkpoint does not rescan it every time.
_memo: dict[tuple[str, int, int], list[Finding]] = {}
# numpy and torch call pickle.load themselves. The outer call already
# scanned the whole file, so a nested guarded call on this thread is skipped.
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


def _scan_path(path: str | os.PathLike) -> list[Finding]:
    path = os.path.abspath(os.fsdecode(path))
    try:
        st = os.stat(path)
    except OSError:
        return []  # the loader raises its own error for a missing file
    key = (path, st.st_size, st.st_mtime_ns)
    if key not in _memo:
        _memo[key] = ModelFileScanner().scan_file(Path(path))
    return _memo[key]


def _scan_stream(stream, suffix: str, label: str) -> list[Finding]:
    """Copy the rest of `stream` to a private temp file and scan that."""
    fd, tmp = tempfile.mkstemp(prefix="hayward-guard-", suffix=suffix)  # mode 0600
    try:
        with os.fdopen(fd, "wb") as out:
            shutil.copyfileobj(stream, out)
        findings = ModelFileScanner().scan_file(Path(tmp))
    finally:
        os.unlink(tmp)
    return [replace(f, file_path=label) for f in findings]


def _check(source, suffix: str, takes_path: bool):
    """Scan what a loader is about to read and return what it should read.

    That is `source` itself, except for a non-seekable stream, which is read
    into memory so the loader can be handed the bytes the scan saw.
    """
    if takes_path and isinstance(source, (str, os.PathLike)):
        _enforce(_scan_path(source), os.fspath(source))
        return source
    if not hasattr(source, "read"):
        return source  # not a stream either: the loader raises its own error
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
    return source


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


def _wrap(module_name: str, original, param: str, suffix: str):
    takes_path = module_name != "pickle"

    @functools.wraps(original)
    def guarded(*args, **kwargs):
        if getattr(_local, "busy", False):
            return original(*args, **kwargs)
        if module_name == "numpy":
            # numpy.load(file, mmap_mode, allow_pickle, ...): without
            # allow_pickle it refuses object arrays, so there is nothing to scan.
            if not kwargs.get("allow_pickle", args[2] if len(args) > 2 else False):
                return original(*args, **kwargs)
        _local.busy = True
        try:
            if args:
                args = (_check(args[0], suffix, takes_path), *args[1:])
            elif param in kwargs:
                kwargs[param] = _check(kwargs[param], suffix, takes_path)
            return original(*args, **kwargs)
        finally:
            _local.busy = False

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


def install(fail_on: str = "high", mode: str = "block") -> None:
    """Patch the supported loaders. Calling it again replaces the settings.

    `fail_on` uses the CLI's `--fail-on` scale. `mode="warn"` reports through
    `warnings.warn` and lets the load go ahead.
    """
    global _limit, _mode
    if fail_on not in _THRESHOLDS:
        raise ValueError(f"fail_on must be one of {', '.join(_THRESHOLDS)}")
    if mode not in ("block", "warn"):
        raise ValueError("mode must be 'block' or 'warn'")
    uninstall()
    _limit, _mode = _THRESHOLDS[fail_on], mode
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
