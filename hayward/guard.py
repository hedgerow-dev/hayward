"""Load guard: refuse a model's unsafe globals as the loader resolves them.

    import hayward.guard  # one line, top of a notebook or script

`pickle.load`, `joblib.load`, `torch.load` and `numpy.load(allow_pickle=True)`
are wrapped. While one of them runs on a thread, every global an unpickler
resolves on that thread is judged: the C `_pickle.Unpickler`, the
pure-Python `pickle._Unpickler` and the subclasses torch, joblib and dill
build on them all raise the `pickle.find_class` audit event before they
import anything, so the guard's audit hook sees each global exactly as the
loader will use it, whatever container, compression or header trick wrapped
the bytes. A global the CLI's pickle classifier denies raises
`UnsafeModelError` there, before it is imported; so does one whose imported
object is a denied callable under another name (`logging.os`, an alias).

Remote code is the exception to that: while the guard is installed, code
transformers runs from a model repo (trust_remote_code=True) is refused
before it runs, at any time, unless `install(remote_code=[...])` names the
repo.

Outside a guarded load the hook does nothing, so `pickle.loads`,
multiprocessing and other in-process unpickling are untouched. That also
means `pickle.loads`, an `Unpickler` used directly, dill's and cloudpickle's
own loaders, a loader bound before the guard was installed and TorchScript
(`torch.jit.load` unpickles in C++) are not checked at load time.

By default unknown globals load, as they are INFO in the CLI. That raises the
bar but is not a sandbox: a pickle built from unknown globals can reach the
guard's own state. `install(strict=True)` refuses every global that is not on
the scanner's known-safe list (what MFV-PICKLE-004 reports), which also
refuses the getattr-style globals such tampering needs.

Unless `install(prescan=False)`, each wrapped loader also runs the static
scan first. The scan adds what the load-time check cannot see, the arguments
a global is called with (an unknown resolver handed the string "os.system"),
for formats the scanner reads the way the loader does. A file is scanned as
the format of the loader reading it, not as its name says. Coverage findings
(MFV-SKIP-*) from the scan warn rather than block: the load-time check still
covers that file. A stream that cannot seek is not pre-scanned.

With `install(verified=True)` each file a loader reads by path is first looked
up by its SHA-256 in the signed verdict log (`hayward verify --sha256`). A file
the log vouches for skips the static scan; anything else is scanned as usual,
with a warning, and `verified="strict"` refuses it instead. The load-time check
runs either way.

`install(fail_on=..., mode=..., max_size=..., prescan=..., strict=...,
remote_code=..., verified=...)`
changes the settings, `uninstall()` restores the loaders and turns the hook
off (an audit hook cannot be removed), and `HAYWARD_GUARD=off` makes the
import a no-op. Warn mode warns and loads.
"""

from __future__ import annotations

import _compat_pickle
import functools
import hashlib
import os
import pickle
import shutil
import sys
import tempfile
import threading
import types
import warnings
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from hayward._pickle_engine import _classify_pickle_global, _laundered_denied_ref
from hayward.cli import _THRESHOLDS, _render_text
from hayward.findings import Category, Finding, Severity, is_coverage_gap
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
_prescan = True
_strict = False
_verified: bool | str = False  # False, True or "strict"
# The verdict log, looked up once per process: the checkpoint (or why there is no
# usable one) and the shards fetched from it. Cleared by install().
_log: dict = {}
# File sha256 by the same file identity as _memo.
_hashes: dict[tuple, str] = {}
# In strict mode these are refused whatever `fail_on` says: each reports a
# global that is not on the known-safe list, or remote code from a repo that
# install(remote_code=...) does not name.
_STRICT_RULES = frozenset({"MFV-PICKLE-001", "MFV-PICKLE-004", "MFV-PICKLE-005", "MFV-HF-002"})
# Repos (org/name) and local folder names whose remote code may run, each as
# the tuple of path parts transformers files it under, see _module_parts.
_remote_code: frozenset[tuple[str, ...]] = frozenset()
_hook_on = False  # the audit hook stays registered once added; this turns it off
_hook_added = False
_patched: list[tuple[object, str, object]] = []
# Per process, keyed on the path, suffix and the file's identity (size,
# mtime, inode, ctime: a copy with its mtime restored still changes ctime),
# so a training loop that reloads a checkpoint does not rescan it every time.
# numpy's suffix is None here: it is read from the file on a miss.
_memo: dict[tuple, list[Finding]] = {}
# numpy, joblib and torch call pickle.load (and joblib numpy.load) themselves.
# `_local.sources` is the stack of what the guarded calls running on this
# thread scanned: resolved paths, and the stream objects handed to loaders.
# A nested call is skipped only when it rereads one of those. joblib's
# NDArrayWrapper names a second file inside the pickle, and that load is
# scanned like any other. `_local.labels` is the matching stack of file names,
# for the audit hook's error message; the hook acts only while it is not
# empty, that is while a guarded load runs on this thread.
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
        key = (path, suffix, st.st_size, st.st_mtime_ns, st.st_ino, st.st_ctime_ns)
        findings = _memo.get(key)  # install() may clear the memo on another thread
        if findings is None:
            if suffix is None:
                with open(path, "rb") as f:
                    suffix = _numpy_suffix(f.read(6))
            findings = _memo[key] = _scanner().scan_file(Path(path), _as_suffix=suffix)
    except OSError:
        return []  # the loader raises its own error for a missing file
    return findings


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


def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _file_sha256(path: str) -> str:
    st = os.stat(path)
    key = (path, st.st_size, st.st_mtime_ns, st.st_ino, st.st_ctime_ns)
    digest = _hashes.get(key)
    if digest is None:
        digest = _hashes[key] = _hash_file(path)
    return digest


def _not_vouched(digest: str) -> str | None:
    """Why the verdict log does not vouch for these bytes, or None when it does: every
    current verdict for them is clean with no coverage gap. The checkpoint is fetched
    once per process, and a failure is not retried, so loads do not each wait on it."""
    from hayward import receipt, verify

    errors = (OSError, ValueError, KeyError, TypeError, AttributeError, receipt.SignatureError)
    if "checkpoint" not in _log:
        _log["shards"] = {}
        try:
            _log["checkpoint"] = verify.fetch_checkpoint()
        except ImportError:
            _log["checkpoint"] = None
            _log["error"] = 'the verdict log could not be checked: it needs pip install "hayward[sign]"'
        except errors as exc:
            _log["checkpoint"] = None
            _log["error"] = f"the verdict log could not be checked ({exc})"
    if _log["checkpoint"] is None:
        return _log["error"]
    n, statement, signed_at = _log["checkpoint"]
    if datetime.now(timezone.utc) - signed_at > verify.MAX_CHECKPOINT_AGE:
        return f"its newest checkpoint was signed over {verify.MAX_CHECKPOINT_AGE.days} days ago"
    prefix = digest[:2]
    if prefix not in _log["shards"]:
        try:
            _log["shards"][prefix] = verify.fetch_shard(statement, n, prefix)
        except errors as exc:
            _log["shards"][prefix] = f"the verdict log could not be checked ({exc})"
    shard = _log["shards"][prefix]
    if isinstance(shard, str):
        return shard
    entries = shard.get(digest) or []
    if not entries:
        return "it has no current verdict"
    if any(not isinstance(e, dict) or e.get("verdict") != "clean" or verify.has_gap(e) for e in entries):
        return "its verdict is not clean with every part read"
    return None


def _vouched(source, takes_path: bool, label: str) -> str | None:
    """With install(verified=...): the resolved path when the verdict log vouches for
    the file a loader is about to read, so the pre-scan can be skipped. Otherwise None
    after a warning, or UnsafeModelError in strict mode."""
    if takes_path and isinstance(source, (str, bytes, os.PathLike)):
        path = _resolve(source)
        try:
            digest = _file_sha256(path)
        except OSError:
            return None  # the loader raises its own error for a missing file
        reason = _not_vouched(digest)
        if reason is None:
            return path
    else:
        reason = "it is read from a stream, which cannot be looked up by hash"
    if _verified == "strict":
        raise UnsafeModelError(label, [], f"hayward refused {label}: it is not vouched for by the "
                                          f"verdict log, because {reason}.")
    # stacklevel 3 points the warning at the caller of the patched loader.
    warnings.warn(f"hayward: {label} is not vouched for by the verdict log, because {reason}; "
                  f"checking it locally instead.", stacklevel=3)
    return None


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


def _check(source, suffix: str | None, takes_path: bool, label: str):
    """Scan what a loader is about to read, and return what was scanned, for
    the nested-call check.

    A stream that cannot seek is left to the load-time check: reading it
    through to the end first would hang on a pipe the writer keeps open,
    which is how multiprocessing hands a spawned worker its task.
    """
    if takes_path and isinstance(source, (str, bytes, os.PathLike)):
        path = _resolve(source)
        _enforce(_scan_path(path, suffix), label)
        return path
    seekable = getattr(source, "seekable", None)
    if not hasattr(source, "read") or seekable is None or not seekable():
        return None
    start = source.tell()
    try:
        findings = _scan_stream(source, suffix, label)
    finally:
        source.seek(start)
    _enforce(findings, label)
    return source


def _enforce(findings: list[Finding], label: str) -> None:
    gaps = [f for f in findings if is_coverage_gap(f)]
    hits = [f for f in findings if not is_coverage_gap(f)
            and (f.severity_order <= _limit or (_strict and f.rule_id in _STRICT_RULES))]
    # stacklevel 4 points a warning at the caller of the patched loader.
    if gaps:
        # The load-time check still sees every global the loader resolves,
        # so a file the scan could not fully read is no longer refused.
        warnings.warn(
            f"hayward could not fully scan {label}; only the load-time check applies:\n"
            + _render_text(gaps, None, False), stacklevel=4)
    if not hits:
        return
    message = f"hayward flagged {label}:\n" + _render_text(hits, None, False)
    if _mode == "block":
        raise UnsafeModelError(label, sorted({f.rule_id for f in hits}), message + "\nRefused to load it.")
    warnings.warn(message + "\nLoading it anyway (warn mode).", stacklevel=4)


def _real_name(module: str, name: str) -> str | None:
    """`module.name` as the object find_class returns for it names itself.

    Imports `module`, which find_class does next anyway. Walks a dotted
    `name` the way protocol 4 does.
    """
    try:
        __import__(module)
        obj = sys.modules[module]
        for part in name.split("."):
            obj = getattr(obj, part)
    except Exception:
        return None  # find_class raises its own error
    if isinstance(obj, types.ModuleType):
        return obj.__name__
    mod, qual = getattr(obj, "__module__", None), getattr(obj, "__qualname__", None)
    return f"{mod}.{qual}" if isinstance(mod, str) and isinstance(qual, str) else None


def _refused_global(module, name) -> tuple[str, str] | None:
    """(rule id, ref) when find_class(module, name) must not hand the loader
    what it resolves to, else None."""
    if not isinstance(module, str) or not isinstance(name, str):
        return None  # find_class raises on these itself
    # Protocol 0-2 names go through pickle's Python 2 mapping (fix_imports)
    # first: `cPickle.loads` loads `pickle.loads`. Both are judged, as the
    # scanner does, since the event does not say which protocol is reading.
    pairs = [(module, name)]
    if (module, name) in _compat_pickle.NAME_MAPPING:
        pairs.append(_compat_pickle.NAME_MAPPING[(module, name)])
    elif module in _compat_pickle.IMPORT_MAPPING:
        pairs.append((_compat_pickle.IMPORT_MAPPING[module], name))
    # By name first, so a refused module is never imported.
    for mod, qual in pairs:
        if _classify_pickle_global(f"{mod}.{qual}") == "denied":
            return "MFV-PICKLE-001", f"{mod}.{qual}"
        if (laundered := _laundered_denied_ref(qual)) is not None:
            return "MFV-PICKLE-001", laundered
    if _strict:
        for mod, qual in pairs:
            if _classify_pickle_global(f"{mod}.{qual}") != "allowed":
                return "MFV-PICKLE-004", f"{mod}.{qual}"
    for mod, qual in pairs:
        real = _real_name(mod, qual)
        if real is None:
            continue
        verdict = _classify_pickle_global(real)
        if verdict == "denied":
            return "MFV-PICKLE-001", real
        if _strict and verdict != "allowed":
            return "MFV-PICKLE-004", real
    return None


def _module_parts(name: str) -> tuple[str, ...]:
    """The folders transformers files `name`'s code under: each part made a
    valid identifier the way its _sanitize_module_name does."""
    parts = []
    for part in name.split("/"):
        part = part.replace(".", "_dot_").replace("-", "_hyphen_")
        parts.append("_" + part if part[:1].isdigit() else part)
    return tuple(parts)


def _check_remote_code(code) -> None:
    """transformers runs a repo's own .py files (trust_remote_code=True) from
    its dynamic-module cache, laid out as transformers_modules/<org>/<name>/
    <commit>/ for a Hub repo and transformers_modules/<folder>/<hash>/ for a
    local one. The exec audit event fires before any of that code runs."""
    filename = getattr(code, "co_filename", "")
    if not isinstance(filename, str) or "transformers_modules" not in filename:
        return
    parts = Path(filename).parts
    if "transformers_modules" not in parts[:-1]:
        return
    if not code.co_names:
        return  # the empty package __init__.py files transformers writes
    last = max(i for i, part in enumerate(parts[:-1]) if part == "transformers_modules")
    folders = parts[last + 1:-1]
    # The last folder is the commit or content hash; what precedes it names the repo.
    if folders[:-1] in _remote_code:
        return
    repo = "/".join(folders[:-1]) or filename
    _enforce([Finding(
        rule_id="MFV-HF-002",
        message=f"transformers is about to run Python code from the model repo {repo} "
                f"({parts[-1]}), loaded with trust_remote_code=True. To allow this "
                f"repo, call hayward.guard.install(remote_code=[...]) with its id.",
        severity=Severity.HIGH,
        category=Category.INJECTION,
        file_path=filename,
        cwe_ids=[94],
        metadata={"at": "load time", "repo": repo},
    )], repo)


def _audit(event: str, args: tuple) -> None:
    if event == "exec" and _hook_on:
        _check_remote_code(args[0])
        return
    if event != "pickle.find_class" or not _hook_on:
        return
    labels = getattr(_local, "labels", None)
    if not labels:
        return  # no guarded load is running on this thread
    refused = _refused_global(*args)
    if refused is None:
        return
    rule_id, ref = refused
    if rule_id == "MFV-PICKLE-001":
        why, severity, key = ("a callable that grants code/command execution on load",
                              Severity.CRITICAL, "denied_globals")
    else:
        why, severity, key = ("which is not on the known-safe list (strict mode)",
                              Severity.INFO, "unknown_globals")
    _enforce([Finding(
        rule_id=rule_id,
        message=f"The pickle asked the loader for {ref}, {why}.",
        severity=severity,
        category=Category.DESERIALIZATION,
        file_path=labels[-1],
        cwe_ids=[502],
        metadata={key: [ref], "at": "load time"},
    )], labels[-1])


def _label(source, outer: list[str]) -> str:
    if isinstance(source, (str, bytes, os.PathLike)):
        return os.fsdecode(source)
    name = getattr(source, "name", None)
    if isinstance(name, (str, bytes, os.PathLike)):
        return os.fsdecode(name)
    return outer[-1] if outer else "<stream>"


def _refusal_in(exc: BaseException) -> UnsafeModelError | None:
    """The guard's refusal behind `exc`: numpy re-raises errors from
    pickle.load as a plain UnpicklingError."""
    for _ in range(8):
        if exc is None or isinstance(exc, UnsafeModelError):
            return exc
        exc = exc.__cause__ or exc.__context__
    return None


def _wrap(module_name: str, original, param: str, suffix: str | None):
    takes_path = module_name != "pickle"

    @functools.wraps(original)
    def guarded(*args, **kwargs):
        if module_name == "numpy":
            # numpy.load(file, mmap_mode, allow_pickle, ...): without
            # allow_pickle it refuses object arrays, so there is nothing to scan.
            if not kwargs.get("allow_pickle", args[2] if len(args) > 2 else False):
                return original(*args, **kwargs)
        elif module_name == "pickle" and \
                sys._getframe(1).f_globals.get("__name__") == "multiprocessing.spawn":
            # A spawn or forkserver child unpickles its Process object from
            # the parent's pipe this way. That is IPC, not a model load. A
            # call nested inside a guarded load is still checked by the
            # enclosing one.
            return original(*args, **kwargs)
        if not args and param not in kwargs:
            return original(*args, **kwargs)  # the loader raises its own error
        scanned = _local.__dict__.setdefault("sources", [])
        labels = _local.__dict__.setdefault("labels", [])
        source = args[0] if args else kwargs[param]
        label = _label(source, labels)
        if scanned and _rereads_scanned_source(source, scanned):
            return original(*args, **kwargs)
        key = _vouched(source, takes_path, label) if _verified else None
        if key is None and _prescan:
            key = _check(source, suffix, takes_path, label)
        scanned.append(key)
        labels.append(label)
        try:
            return original(*args, **kwargs)
        except Exception as e:
            refusal = _refusal_in(e)
            if refusal is None or refusal is e:
                raise
            raise refusal from None
        finally:
            scanned.pop()
            labels.pop()

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
            # One loader can serve many modules (zipimport, frozen apps), so
            # this runs for whatever it loads next too.
            if module.__name__ in _TARGETS and _hook in sys.meta_path:
                _patch(module)

        loader.exec_module = exec_and_patch
        return spec


_hook = _ImportHook()


def install(
    fail_on: str = "high", mode: str = "block", max_size: int | None = None,
    prescan: bool = True, strict: bool = False, remote_code: tuple[str, ...] | list[str] = (),
    verified: bool | str = False,
) -> None:
    """Turn on the load-time check and patch the supported loaders. Calling it
    again replaces the settings.

    `fail_on` uses the CLI's `--fail-on` scale; a denied global is CRITICAL.
    `mode="warn"` reports through `warnings.warn` and lets the load go ahead.
    `max_size` raises the pre-scan's per-file cap in bytes, like
    `--max-size`, so a large non-zip file is scanned rather than left to the
    load-time check alone. `prescan=False` keeps only the load-time check.
    `strict=True` also refuses every global that is not on the scanner's
    known-safe list, at load time and in the pre-scan, whatever `fail_on` is.
    `remote_code` names the repos (`org/name`) or local model folders whose own
    Python code transformers may run with trust_remote_code=True; any other
    is refused as MFV-HF-002.
    `verified=True` looks each file a loader reads by path up in the signed verdict
    log (as `hayward verify --sha256` does): a file it vouches for skips the pre-scan,
    and anything else is checked locally as usual, with a warning. `verified="strict"`
    refuses anything the log does not vouch for. The load-time check runs either way.
    It needs hayward[sign] and the network.
    """
    global _limit, _mode, _max_size, _prescan, _strict, _hook_on, _hook_added, _remote_code, _verified
    if fail_on not in _THRESHOLDS:
        raise ValueError(f"fail_on must be one of {', '.join(_THRESHOLDS)}")
    if mode not in ("block", "warn"):
        raise ValueError("mode must be 'block' or 'warn'")
    if max_size is not None and max_size <= 0:
        raise ValueError("max_size must be a positive byte count")
    if verified not in (False, True, "strict"):
        raise ValueError('verified must be False, True or "strict"')
    if isinstance(remote_code, str):
        raise ValueError("remote_code must be a list of repo ids, not one string")
    if any(not name.strip("/") for name in remote_code):
        raise ValueError("remote_code entries must be repo ids or folder names, not empty")
    uninstall()
    _limit, _mode, _max_size, _prescan = _THRESHOLDS[fail_on], mode, max_size, prescan
    _strict = bool(strict)
    _verified = verified
    _log.clear()
    # transformers 4.x filed the name as is; newer releases sanitize it.
    _remote_code = frozenset(form for name in remote_code
                             for form in (tuple(name.strip("/").split("/")), _module_parts(name.strip("/"))))
    _memo.clear()  # a different cap can change a file's findings
    for name in _TARGETS:
        if name in sys.modules:
            _patch(sys.modules[name])
    sys.meta_path.insert(0, _hook)
    if not _hook_added:
        sys.addaudithook(_audit)
        _hook_added = True
    _hook_on = True


def uninstall() -> None:
    """Restore the original loaders, stop patching new imports and turn the
    load-time check off."""
    global _hook_on
    _hook_on = False
    while _patched:
        module, attr, original = _patched.pop()
        setattr(module, attr, original)
    if _hook in sys.meta_path:
        sys.meta_path.remove(_hook)


if os.environ.get("HAYWARD_GUARD", "").lower() != "off":
    install()
