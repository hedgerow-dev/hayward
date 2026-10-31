"""The load guard: `import hayward.guard` refuses unsafe globals as a loader
resolves them, and scans the file first unless `install(prescan=False)`.

The malicious fixtures call `builtins.eval("40+2")`. That is a CRITICAL
finding, and harmless if the guard ever fails to stop it: the test then sees
42 instead of an exception and fails.

numpy, joblib and torch tests skip when the library is absent; the pickle
tests need nothing beyond the standard library.
"""

from __future__ import annotations

import hashlib
import io
import lzma
import os
import pickle
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import warnings
import zipfile
from datetime import datetime, timedelta, timezone
from urllib.error import URLError

import pytest

from hayward import guard, verify
from hayward.scanner import ModelFileScanner

# Importing hayward.guard installed it. Every test installs it explicitly.
guard.uninstall()
ORIGINAL_LOAD = pickle.load
ORIGINAL_LOADS = pickle.loads


@pytest.fixture(autouse=True)
def _uninstall_after():
    yield
    guard.uninstall()


def _short_binunicode(text: str) -> bytes:
    raw = text.encode()
    return bytes([0x8C, len(raw)]) + raw


def _call_pickle(module: str, name: str, arg: str) -> bytes:
    """A standalone protocol-4 pickle calling module.name(arg)."""
    return (
        b"\x80\x04"
        + _short_binunicode(module)
        + _short_binunicode(name)
        + b"\x93"
        + _short_binunicode(arg)
        + b"\x85"
        + b"R."
    )


EVAL = _call_pickle("builtins", "eval", "40+2")
# An unfamiliar global is INFO, below the default threshold.
UNKNOWN = _call_pickle("hayward_no_such_pkg.mod", "Thing", "x")
BENIGN = {"weights": [1.0, 2.0]}
# The same eval in protocol 0, which opens with GLOBAL rather than PROTO.
EVAL_P0 = b"cbuiltins\neval\n(S'40+2'\ntR."


def _scan_spy(monkeypatch) -> list[int]:
    """Record the permission bits of every file the guard hands the scanner."""
    modes: list[int] = []
    real = ModelFileScanner.scan_file

    def spy(self, path, **kwargs):
        modes.append(stat.S_IMODE(os.stat(path).st_mode))
        return real(self, path, **kwargs)

    monkeypatch.setattr(ModelFileScanner, "scan_file", spy)
    return modes


class _Pipe:
    """A stream pickle.load accepts but that cannot seek."""

    def __init__(self, data: bytes):
        self._buf = io.BytesIO(data)

    def read(self, n=-1):
        return self._buf.read(n)

    def readline(self):
        return self._buf.readline()

    def seekable(self):
        return False


def _load(data: bytes):
    return pickle.load(io.BytesIO(data))


class TestPickle:
    def test_load_blocks_malicious_stream(self):
        guard.install()
        with pytest.raises(guard.UnsafeModelError) as exc:
            _load(EVAL)
        assert isinstance(exc.value, pickle.UnpicklingError)
        assert exc.value.rule_ids == ["MFV-PICKLE-001"]
        assert "MFV-PICKLE-001" in str(exc.value)
        assert "builtins.eval" in str(exc.value)

    def test_load_passes_benign(self):
        guard.install()
        assert _load(pickle.dumps(BENIGN)) == BENIGN

    @pytest.mark.parametrize("strict", [False, True])
    def test_loads_outside_a_model_load_is_untouched(self, strict):
        # Not wrapped and, outside a guarded load, not checked: pickle.loads
        # is the in-memory IPC and cache path. A known limit of the scope.
        guard.install(strict=strict)
        assert pickle.loads is ORIGINAL_LOADS
        assert pickle.loads(EVAL) == 42

    def test_load_blocks_malicious_file(self, tmp_path):
        guard.install()
        p = tmp_path / "evil.pkl"
        p.write_bytes(EVAL)
        with open(p, "rb") as f, pytest.raises(guard.UnsafeModelError) as exc:
            pickle.load(f)
        assert exc.value.file == str(p)
        assert str(p) in str(exc.value)

    def test_pickle_never_treats_a_path_as_a_file(self, tmp_path):
        p = tmp_path / "evil.pkl"
        p.write_bytes(EVAL)
        guard.install()
        with pytest.raises(TypeError):
            pickle.load(p)
        with pytest.raises(TypeError):
            pickle.load(str(p))

    def test_load_rewinds_the_stream(self, tmp_path):
        guard.install()
        p = tmp_path / "two.pkl"
        p.write_bytes(pickle.dumps(BENIGN) + pickle.dumps([3, 4]))
        with open(p, "rb") as f:
            assert pickle.load(f) == BENIGN
            assert pickle.load(f) == [3, 4]

    def test_non_seekable_stream(self):
        guard.install()
        with pytest.raises(guard.UnsafeModelError):
            pickle.load(_Pipe(EVAL))
        assert pickle.load(_Pipe(pickle.dumps(BENIGN))) == BENIGN

    def test_spool_is_private_and_deleted(self, tmp_path, monkeypatch):
        spool = tmp_path / "spool"
        spool.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(spool))
        modes = _scan_spy(monkeypatch)
        guard.install()

        assert _load(pickle.dumps(BENIGN)) == BENIGN
        with pytest.raises(guard.UnsafeModelError):
            _load(EVAL)

        assert len(modes) == 2
        assert list(spool.iterdir()) == []
        if os.name == "posix":
            assert modes == [0o600, 0o600]

    def test_warn_mode_lets_the_load_run(self):
        guard.install(mode="warn")
        with pytest.warns(UserWarning, match="MFV-PICKLE-001"):
            assert _load(EVAL) == 42

    def test_threshold(self):
        guard.install()
        # Below `high`: the guard lets it through and pickle fails on the import.
        with pytest.raises(ModuleNotFoundError):
            _load(UNKNOWN)
        guard.install(fail_on="info")
        with pytest.raises(guard.UnsafeModelError):
            _load(UNKNOWN)

    def test_coverage_gap_warns_and_loads(self, monkeypatch):
        # The load-time check still covers a file the scan could not fully
        # read, so it loads with a warning rather than being refused.
        monkeypatch.setattr(ModelFileScanner, "MAX_SCAN_BYTES", 8)
        guard.install()
        with pytest.warns(UserWarning, match="MFV-SKIP-001"):
            assert _load(pickle.dumps(BENIGN)) == BENIGN

    def test_coverage_gap_keeps_the_load_time_check(self, monkeypatch):
        monkeypatch.setattr(ModelFileScanner, "MAX_SCAN_BYTES", 8)
        guard.install()
        with pytest.warns(UserWarning, match="MFV-SKIP-001"), \
                pytest.raises(guard.UnsafeModelError) as exc:
            _load(EVAL)
        assert exc.value.rule_ids == ["MFV-PICKLE-001"]

    def test_max_size_lets_a_large_file_be_scanned(self, monkeypatch, recwarn):
        monkeypatch.setattr(ModelFileScanner, "MAX_SCAN_BYTES", 8)
        guard.install(max_size=1_000_000)
        assert _load(pickle.dumps(BENIGN)) == BENIGN
        assert not [w for w in recwarn if "MFV-SKIP" in str(w.message)]

    def test_bad_options_are_rejected(self):
        with pytest.raises(ValueError):
            guard.install(fail_on="severe")
        with pytest.raises(ValueError):
            guard.install(mode="log")
        with pytest.raises(ValueError):
            guard.install(max_size=0)
        assert pickle.load is ORIGINAL_LOAD

    def test_uninstall_restores_originals(self):
        guard.install()
        assert pickle.load is not ORIGINAL_LOAD
        guard.uninstall()
        assert pickle.load is ORIGINAL_LOAD

    @pytest.mark.parametrize("protocol", range(6))
    def test_every_protocol_loads_from_a_file(self, tmp_path, protocol):
        p = tmp_path / "model.pkl"
        p.write_bytes(pickle.dumps(BENIGN, protocol=protocol))
        guard.install()
        with open(p, "rb") as f:
            assert pickle.load(f) == BENIGN

    def test_unparseable_stream_warns_and_pickle_fails_on_it(self):
        guard.install()
        with pytest.warns(UserWarning, match="MFV-SKIP-003"), \
                pytest.raises(pickle.UnpicklingError) as exc:
            _load(b"\xff\xfe")
        assert not isinstance(exc.value, guard.UnsafeModelError)

    def test_reinstall_replaces_settings_without_double_wrapping(self):
        guard.install()
        guard.install(mode="warn")
        with pytest.warns(UserWarning):
            assert _load(EVAL) == 42
        guard.uninstall()
        assert pickle.load is ORIGINAL_LOAD


def _run(code: str, *args: str, env_extra: dict[str, str] | None = None) -> str:
    env = {k: v for k, v in os.environ.items() if k != "HAYWARD_GUARD"}
    env.update(env_extra or {})
    result = subprocess.run(
        [sys.executable, "-c", code, *args],
        env=env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


class TestImport:
    def test_import_installs(self):
        out = _run("import pickle; o = pickle.load; import hayward.guard; print(pickle.load is o)")
        assert out.strip() == "False"

    def test_env_off_makes_import_a_no_op(self):
        out = _run(
            "import pickle; o = pickle.load; import hayward.guard; print(pickle.load is o)",
            env_extra={"HAYWARD_GUARD": "off"},
        )
        assert out.strip() == "True"

    def test_a_shared_loader_keeps_loading_other_modules(self, tmp_path):
        # zipimport serves every module in the zip from one loader, which the
        # guard wraps when it loads a target.
        archive = tmp_path / "libs.zip"
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("joblib/__init__.py", "def load(filename):\n    return 'loaded'\n")
            z.writestr("other.py", "VALUE = 1\n")
        code = (
            "import sys\n"
            "sys.path.insert(0, sys.argv[1])\n"
            "import hayward.guard, joblib, other\n"
            "print(joblib.load(sys.argv[1] + '/missing'), other.VALUE)\n"
        )
        assert _run(code, str(archive)).split() == ["loaded", "1"]

    @pytest.mark.parametrize("module, call", [
        ("numpy", "numpy.load(sys.argv[1], allow_pickle=True)"),
        ("joblib", "joblib.load(sys.argv[1])"),
        ("torch", "torch.load(sys.argv[1], weights_only=False)"),
    ])
    def test_framework_imported_after_the_guard(self, tmp_path, module, call):
        pytest.importorskip(module)
        p = tmp_path / "evil.pkl"
        p.write_bytes(EVAL)
        code = (
            "import sys, hayward.guard\n"
            f"assert {module!r} not in sys.modules, 'the guard imported {module}'\n"
            f"import {module}\n"
            "try:\n"
            f"    {call}\n"
            "except hayward.guard.UnsafeModelError:\n"
            "    print('blocked')\n"
        )
        assert _run(code, str(p)).strip() == "blocked"


def _evil_npy() -> bytes:
    header = b"{'descr': '|O', 'fortran_order': False, 'shape': (1,), }"
    total = 10 + len(header) + 1
    header += b" " * ((64 - total % 64) % 64) + b"\n"
    return b"\x93NUMPY\x01\x00" + len(header).to_bytes(2, "little") + header + EVAL


class TestNumpy:
    @pytest.fixture(autouse=True)
    def _np(self):
        self.np = pytest.importorskip("numpy")

    def test_allow_pickle_blocks_malicious(self, tmp_path):
        p = tmp_path / "evil.npy"
        p.write_bytes(_evil_npy())
        guard.install()
        with pytest.raises(guard.UnsafeModelError):
            self.np.load(p, allow_pickle=True)
        with pytest.raises(guard.UnsafeModelError):
            self.np.load(p, None, True)  # allow_pickle passed positionally
        with open(p, "rb") as f, pytest.raises(guard.UnsafeModelError):
            self.np.load(f, allow_pickle=True)

    def test_no_scan_without_allow_pickle(self, tmp_path, monkeypatch):
        p = tmp_path / "plain.npy"
        self.np.save(p, self.np.arange(3))
        modes = _scan_spy(monkeypatch)
        guard.install()
        assert self.np.load(p).tolist() == [0, 1, 2]
        assert modes == []

    def test_repeat_loads_are_memoised(self, tmp_path, monkeypatch):
        p = tmp_path / "obj.npy"
        self.np.save(p, self.np.array([BENIGN], dtype=object), allow_pickle=True)
        modes = _scan_spy(monkeypatch)
        guard.install()
        for _ in range(3):
            assert self.np.load(p, allow_pickle=True)[0] == BENIGN
        assert len(modes) == 1
        st = os.stat(p)
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
        self.np.load(p, allow_pickle=True)
        assert len(modes) == 2
        # Replaced in place with the size and mtime restored, as `cp -p`
        # does: a different file all the same.
        st = os.stat(p)
        q = tmp_path / "copy.npy"
        q.write_bytes(p.read_bytes())
        os.replace(q, p)
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))
        self.np.load(p, allow_pickle=True)
        assert len(modes) == 3

    def test_uninstall_restores(self):
        original = self.np.load
        guard.install()
        assert self.np.load is not original
        guard.uninstall()
        assert self.np.load is original


class TestJoblib:
    @pytest.fixture(autouse=True)
    def _joblib(self):
        self.joblib = pytest.importorskip("joblib")

    def test_blocks_malicious_path_and_file(self, tmp_path):
        p = tmp_path / "evil.joblib"
        p.write_bytes(EVAL)
        guard.install()
        with pytest.raises(guard.UnsafeModelError):
            self.joblib.load(p)
        with open(p, "rb") as f, pytest.raises(guard.UnsafeModelError):
            self.joblib.load(f)

    def test_benign_loads(self, tmp_path):
        p = tmp_path / "model.joblib"
        self.joblib.dump(BENIGN, p)
        guard.install()
        assert self.joblib.load(p) == BENIGN
        with open(p, "rb") as f:
            assert self.joblib.load(f) == BENIGN


class TestTorch:
    @pytest.fixture(autouse=True)
    def _torch(self):
        self.torch = pytest.importorskip("torch")

    def test_blocks_malicious(self, tmp_path):
        p = tmp_path / "evil.pt"
        p.write_bytes(EVAL)
        guard.install()
        with pytest.raises(guard.UnsafeModelError):
            self.torch.load(p, weights_only=False)
        with pytest.raises(guard.UnsafeModelError):
            self.torch.load(io.BytesIO(EVAL), weights_only=False)

    @pytest.mark.parametrize("zipped", [True, False])
    def test_benign_checkpoint_loads(self, tmp_path, zipped):
        p = tmp_path / "model.pt"
        self.torch.save({"w": self.torch.ones(2)}, p, _use_new_zipfile_serialization=zipped)
        guard.install()
        for weights_only in (True, False):
            assert self.torch.load(p, weights_only=weights_only)["w"].tolist() == [1.0, 1.0]
            with open(p, "rb") as f:
                assert self.torch.load(f, weights_only=weights_only)["w"].tolist() == [1.0, 1.0]

    def test_protocol0_pth_is_blocked(self, tmp_path):
        p = tmp_path / "model.pth"
        p.write_bytes(EVAL_P0)
        guard.install()
        with pytest.raises(guard.UnsafeModelError):
            self.torch.load(p, weights_only=False)

    def test_benign_legacy_tar_checkpoint_loads(self, tmp_path):
        # The pre-1.6 tar layout, built by hand: no storages, no tensors.
        import tarfile

        members = [("storages", pickle.dumps(0) + pickle.dumps([])),
                   ("tensors", pickle.dumps(0)), ("pickle", pickle.dumps(BENIGN))]
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:", format=tarfile.PAX_FORMAT) as tf:
            for name, data in members:
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
        p = tmp_path / "legacy.pt"
        p.write_bytes(buf.getvalue())
        guard.install()
        assert self.torch.load(p, weights_only=False) == BENIGN

    def test_legacy_tar_checkpoint_is_blocked(self, tmp_path):
        import tarfile

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:", format=tarfile.PAX_FORMAT) as tf:
            info = tarfile.TarInfo("storages")
            info.size = len(EVAL)
            tf.addfile(info, io.BytesIO(EVAL))
        p = tmp_path / "legacy.pt"
        p.write_bytes(buf.getvalue())
        guard.install()
        with pytest.raises(guard.UnsafeModelError):
            self.torch.load(p, weights_only=False)


class TestNestedLoads:
    """A load inside a load is skipped only when it rereads the source the
    outer call scanned."""

    @pytest.fixture(autouse=True)
    def _libs(self):
        self.np = pytest.importorskip("numpy")
        self.joblib = pytest.importorskip("joblib")

    def test_joblib_side_file_is_scanned(self, tmp_path):
        # joblib's pre-0.10 NDArrayWrapper makes joblib.load call
        # numpy.load(<file named in the pickle>, allow_pickle=True).
        from joblib.numpy_pickle_compat import NDArrayWrapper

        side = tmp_path / "side.npy"
        side.write_bytes(EVAL)
        model = tmp_path / "model.joblib"
        model.write_bytes(pickle.dumps(NDArrayWrapper("side.npy", self.np.ndarray), protocol=2))
        guard.install()
        with pytest.raises(guard.UnsafeModelError) as exc:
            self.joblib.load(model)
        assert exc.value.file == str(side)
        assert "MFV-PICKLE-001" in exc.value.rule_ids

    def test_rereading_the_scanned_file_is_not_rescanned(self, tmp_path, monkeypatch):
        # numpy.load hands a pickle file to pickle.load on the handle it opened.
        p = tmp_path / "model.pkl"
        p.write_bytes(pickle.dumps(BENIGN))
        modes = _scan_spy(monkeypatch)
        guard.install()
        assert self.np.load(p, allow_pickle=True) == BENIGN
        assert len(modes) == 1

    def test_compressed_joblib_object_arrays_are_not_rescanned(self, tmp_path, monkeypatch):
        # joblib unpickles an object array from its own decompressing wrapper
        # around the file it opened, which is still the scanned source.
        p = tmp_path / "model.joblib"
        self.joblib.dump({"o": self.np.array(["a", "b"], dtype=object)}, p, compress=("gzip", 3))
        modes = _scan_spy(monkeypatch)
        guard.install()
        assert self.joblib.load(p)["o"].tolist() == ["a", "b"]
        assert len(modes) == 1


class TestLoaderFormat:
    """A path is scanned as the format of the loader reading it, not its name."""

    @pytest.fixture(autouse=True)
    def _libs(self):
        self.np = pytest.importorskip("numpy")
        self.joblib = pytest.importorskip("joblib")

    @pytest.mark.parametrize("name", ["model.pth", "model.onnx", "graph.pb", "weights.npz"])
    def test_pickle_under_another_name_is_blocked(self, tmp_path, name):
        p = tmp_path / name
        p.write_bytes(EVAL_P0)
        guard.install()
        with pytest.raises(guard.UnsafeModelError):
            self.np.load(p, allow_pickle=True)
        with pytest.raises(guard.UnsafeModelError):
            self.joblib.load(p)

    def test_bytes_path_is_scanned(self, tmp_path):
        p = tmp_path / "evil.pkl"
        p.write_bytes(EVAL)
        guard.install()
        with pytest.raises(guard.UnsafeModelError):
            self.joblib.load(os.fsencode(p))
        with pytest.raises(guard.UnsafeModelError):
            self.np.load(os.fsencode(p), allow_pickle=True)

    def test_gzip_pickle_named_pkl_is_blocked(self, tmp_path):
        import gzip

        p = tmp_path / "model.pkl"
        p.write_bytes(gzip.compress(EVAL))
        guard.install()
        with pytest.raises(guard.UnsafeModelError):
            self.joblib.load(p)

    @pytest.mark.parametrize("name", ["junk.dat", "junk.json", "junk.pkl"])
    def test_content_the_scanner_cannot_read_warns(self, tmp_path, name):
        # Under an unclaimed name the scanner reports nothing for it; the
        # loader unpickles it anyway, so the guard says it could not read it.
        p = tmp_path / name
        p.write_bytes(b"\xff\xfe not a pickle")
        guard.install()
        for load in (lambda: self.np.load(p, allow_pickle=True), lambda: self.joblib.load(p)):
            with pytest.warns(UserWarning, match="MFV-SKIP-003"), pytest.raises(Exception) as exc:
                load()
            assert not isinstance(exc.value, guard.UnsafeModelError)

    def test_object_npy_named_npz_is_blocked(self, tmp_path):
        p = tmp_path / "arrays.npz"
        p.write_bytes(_evil_npy())
        guard.install()
        with pytest.raises(guard.UnsafeModelError):
            self.np.load(p, allow_pickle=True)


class TestBenignLoads:
    """Files the libraries write themselves load under the guard."""

    @pytest.fixture(autouse=True)
    def _libs(self):
        self.np = pytest.importorskip("numpy")
        self.joblib = pytest.importorskip("joblib")

    def _model(self):
        np = self.np
        return {"w": np.arange(6.0).reshape(2, 3), "labels": np.array(["a", "b"], dtype=object),
                "meta": BENIGN}

    @pytest.mark.parametrize("compress", [0, 3, ("zlib", 3), ("gzip", 3), ("bz2", 3), ("lzma", 3), ("xz", 3)])
    def test_joblib_dump(self, tmp_path, compress):
        p = tmp_path / "model.joblib"
        self.joblib.dump(self._model(), p, compress=compress)
        guard.install()
        assert self.joblib.load(p)["labels"].tolist() == ["a", "b"]
        with open(p, "rb") as f:
            assert self.joblib.load(f)["meta"] == BENIGN

    def test_numpy_save_and_savez(self, tmp_path):
        np = self.np
        npy = tmp_path / "obj.npy"
        np.save(npy, np.array([BENIGN], dtype=object), allow_pickle=True)
        npz = tmp_path / "arrays.npz"
        np.savez(npz, n=np.arange(3), o=np.array([BENIGN], dtype=object))
        guard.install()
        assert np.load(npy, allow_pickle=True)[0] == BENIGN
        with np.load(npz, allow_pickle=True) as z:
            assert z["n"].tolist() == [0, 1, 2]
            assert z["o"][0] == BENIGN
        with open(npz, "rb") as f, np.load(f, allow_pickle=True) as z:
            assert z["o"][0] == BENIGN

    @pytest.mark.parametrize("name", ["model.pkl", "model.npy", "model.onnx"])
    def test_files_named_for_another_format_load(self, tmp_path, name):
        # Scanned as the file's name says, each would be an extension
        # mismatch; scanned as the loader reads it, it is what it is.
        p = tmp_path / name
        self.joblib.dump(self._model(), p, compress=3)
        q = tmp_path / ("plain-" + name)
        q.write_bytes(pickle.dumps(BENIGN))
        guard.install()
        assert self.joblib.load(p)["meta"] == BENIGN
        assert self.joblib.load(q) == BENIGN
        assert self.np.load(q, allow_pickle=True) == BENIGN


# ── Load-time check ────────────────────────────────────────────────────


class _Eval:
    """Pickles as builtins.eval("40+2"), whatever container it lands in."""

    def __reduce__(self):
        return (eval, ("40+2",))


# The eval payload with no PROTO header: a bare STACK_GLOBAL stream.
EVAL_BARE = EVAL[2:]


def _tar(members) -> bytes:
    """A pre-1.6 torch tar checkpoint from (name, data, type, linkname)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:", format=tarfile.USTAR_FORMAT) as tf:
        for name, data, kind, link in members:
            info = tarfile.TarInfo(name)
            info.type, info.linkname, info.size = kind, link, len(data)
            tf.addfile(info, io.BytesIO(data) if data else None)
    return buf.getvalue()


_TAR_BASE = [
    ("storages", pickle.dumps(0, protocol=2) + pickle.dumps([], protocol=2), tarfile.REGTYPE, ""),
    ("tensors", pickle.dumps(0, protocol=2), tarfile.REGTYPE, ""),
]


def _npy(header: str, payload: bytes) -> bytes:
    total = 10 + len(header) + 1
    header += " " * ((64 - total % 64) % 64) + "\n"
    return b"\x93NUMPY\x01\x00" + len(header).to_bytes(2, "little") + header.encode("latin1") + payload


@pytest.fixture(params=[True, False], ids=["prescan", "load-time-only"])
def prescan(request):
    return request.param


class TestParserDifferentials:
    """The security reviews' bypasses: each file reads differently to the
    loader than to the scanner. The load-time check sees the global the
    loader resolves, so each is refused with or without the pre-scan."""

    def _torch_blocks(self, tmp_path, data, prescan):
        torch = pytest.importorskip("torch")
        p = tmp_path / "model.pt"
        p.write_bytes(data)
        guard.install(prescan=prescan)
        with pytest.raises(guard.UnsafeModelError) as exc:
            torch.load(p, weights_only=False)
        assert "MFV-PICKLE-001" in exc.value.rule_ids
        assert exc.value.file == str(p)

    def test_tar_symlink_member(self, tmp_path, prescan):
        data = _tar(_TAR_BASE + [("blob", EVAL_BARE, tarfile.REGTYPE, ""),
                                 ("pickle", b"", tarfile.SYMTYPE, "blob")])
        self._torch_blocks(tmp_path, data, prescan)

    def test_tar_hardlink_member(self, tmp_path, prescan):
        data = _tar(_TAR_BASE + [("blob", EVAL_BARE, tarfile.REGTYPE, ""),
                                 ("pickle", b"", tarfile.LNKTYPE, "blob")])
        self._torch_blocks(tmp_path, data, prescan)

    def test_tar_unknown_member_type(self, tmp_path, prescan):
        self._torch_blocks(tmp_path, _tar(_TAR_BASE + [("pickle", EVAL_BARE, b"Z", "")]), prescan)

    def test_tar_member_over_the_member_cap(self, tmp_path, prescan, monkeypatch):
        monkeypatch.setattr(ModelFileScanner, "MAX_ZIP_MEMBER_BYTES", 1000)
        padded = EVAL_BARE + b"\x00" * 2000  # torch stops reading at STOP
        self._torch_blocks(tmp_path, _tar(_TAR_BASE + [("pickle", padded, tarfile.REGTYPE, "")]),
                           prescan)

    def test_tar_zip_polyglot_over_the_size_cap(self, tmp_path, prescan, monkeypatch):
        monkeypatch.setattr(ModelFileScanner, "MAX_SCAN_BYTES", 15_000)
        reg = tarfile.REGTYPE
        data = _tar([("ccollections\nOrderedDict\n", b"x", reg, "")] + _TAR_BASE
                    + [("pickle", EVAL_BARE, reg, ""), ("padding", b"\x00" * 20_000, reg, "")])
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("archive/version", "3\n")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # the pre-scan's coverage warning
            self._torch_blocks(tmp_path, data + buf.getvalue(), prescan)

    @pytest.mark.parametrize("first, second", [("alone", "xz"), ("xz", "alone")])
    def test_mixed_lzma_and_xz_streams(self, tmp_path, prescan, first, second):
        joblib = pytest.importorskip("joblib")
        fmt = {"alone": lzma.FORMAT_ALONE, "xz": lzma.FORMAT_XZ}
        data = (lzma.compress(b"ccollections\nOrderedDict\n0", format=fmt[first])
                + lzma.compress(EVAL_P0, format=fmt[second]))
        p = tmp_path / "model.joblib"
        p.write_bytes(data)
        guard.install(prescan=prescan)
        with pytest.raises(guard.UnsafeModelError):
            joblib.load(p)

    @pytest.mark.parametrize("header", [
        "{'descr': 'object', 'fortran_order': False, 'shape': (1,), }",
        "{'descr': '|O', 'fortran_order': False, 'shape': (1L,), }",
    ], ids=["descr-object", "python2-long-shape"])
    def test_npy_header_numpy_reads_differently(self, tmp_path, prescan, header):
        np = pytest.importorskip("numpy")
        p = tmp_path / "evil.npy"
        p.write_bytes(_npy(header, pickle.dumps(_Eval(), protocol=2)))
        guard.install(prescan=prescan)
        # numpy re-raises pickle errors as a plain UnpicklingError; the guard
        # unwraps its own refusal from behind it.
        with pytest.raises(guard.UnsafeModelError):
            np.load(p, allow_pickle=True)

    @pytest.mark.parametrize("compress", [0, ("gzip", 3)])
    def test_joblib_payload_after_an_inline_array(self, tmp_path, prescan, compress):
        np = pytest.importorskip("numpy")
        joblib = pytest.importorskip("joblib")
        p = tmp_path / "tail.pkl"
        joblib.dump([np.zeros(16), _Eval()], p, compress=compress)
        guard.install(prescan=prescan)
        with pytest.raises(guard.UnsafeModelError):
            joblib.load(p)

    def test_round_one_cases(self, tmp_path, prescan):
        # joblib side file, gzip pickle named .pkl, protocol 0 named .pth.
        import gzip

        np = pytest.importorskip("numpy")
        joblib = pytest.importorskip("joblib")
        from joblib.numpy_pickle_compat import NDArrayWrapper

        (tmp_path / "side.npy").write_bytes(EVAL)
        (tmp_path / "model.joblib").write_bytes(
            pickle.dumps(NDArrayWrapper("side.npy", np.ndarray), protocol=2))
        (tmp_path / "model.pkl").write_bytes(gzip.compress(EVAL))
        (tmp_path / "x.pth").write_bytes(EVAL_P0)
        guard.install(prescan=prescan)
        for load in (lambda: joblib.load(tmp_path / "model.joblib"),
                     lambda: joblib.load(tmp_path / "model.pkl"),
                     lambda: joblib.load(tmp_path / "x.pth"),
                     lambda: np.load(tmp_path / "x.pth", allow_pickle=True)):
            with pytest.raises(guard.UnsafeModelError):
                load()


def _unpickle(data: bytes):
    """A global a pickle can call that unpickles more bytes itself."""
    return pickle.loads(data)


def _unpickle_on_another_thread(data: bytes):
    out = []
    t = threading.Thread(target=lambda: out.append(pickle.loads(data)))
    t.start()
    t.join()
    return out[0]


class _Calls:
    """Pickles as fn(arg)."""

    def __init__(self, fn, arg):
        self.fn, self.arg = fn, arg

    def __reduce__(self):
        return (self.fn, (self.arg,))


def _via_bound_before_install(data):
    return ORIGINAL_LOAD(io.BytesIO(data))


def _via_c_unpickler(data):
    import _pickle

    return _pickle.Unpickler(io.BytesIO(data)).load()


def _via_python_unpickler(data):
    return pickle._Unpickler(io.BytesIO(data)).load()


def _via_unpickler(data):
    return pickle.Unpickler(io.BytesIO(data)).load()


_UNPICKLERS = {
    "bound before install": _via_bound_before_install,
    "_pickle.Unpickler": _via_c_unpickler,
    "pickle._Unpickler": _via_python_unpickler,
    "pickle.Unpickler": _via_unpickler,
    "pickle.loads": _unpickle,
}

_STARTED, _RELEASE = threading.Event(), threading.Event()


def _wait(_):
    _STARTED.set()
    return _RELEASE.wait(10)


class TestScope:
    """The load-time check acts only while a guarded loader runs on the
    calling thread."""

    @pytest.mark.parametrize("strict", [False, True])
    @pytest.mark.parametrize("how", list(_UNPICKLERS))
    def test_unpickling_outside_a_guarded_load_is_untouched(self, how, strict):
        # A known limit: none of these is a guarded loader.
        guard.install(strict=strict)
        assert _UNPICKLERS[how](EVAL) == 42

    @pytest.mark.parametrize("how", list(_UNPICKLERS))
    def test_every_unpickler_inside_a_guarded_load_is_checked(self, how):
        # A global the model calls that unpickles more bytes on the same
        # thread is inside the load, whichever unpickler it uses.
        guard.install(prescan=False)
        outer = pickle.dumps(_Calls(_UNPICKLERS[how], EVAL))
        with pytest.raises(guard.UnsafeModelError) as exc:
            pickle.load(io.BytesIO(outer))
        assert exc.value.rule_ids == ["MFV-PICKLE-001"]

    def test_nested_loads_keep_the_check(self):
        guard.install(prescan=False)
        inner = pickle.dumps(_Calls(_unpickle, EVAL))
        with pytest.raises(guard.UnsafeModelError):
            pickle.load(io.BytesIO(pickle.dumps(_Calls(_unpickle, inner))))
        # The check is still on for the rest of the outer load after an
        # inner one returns.
        outer = pickle.dumps([_Calls(_unpickle, pickle.dumps(BENIGN)), _Calls(eval, "40+2")])
        with pytest.raises(guard.UnsafeModelError):
            pickle.load(io.BytesIO(outer))

    def test_other_threads_are_untouched(self):
        # A known limit of the default mode: work a pickle hands to another
        # thread is outside the load. Strict mode refuses the handoff.
        guard.install(prescan=False)
        data = pickle.dumps(_Calls(_unpickle_on_another_thread, EVAL))
        assert pickle.load(io.BytesIO(data)) == 42
        guard.install(prescan=False, strict=True)
        with pytest.raises(guard.UnsafeModelError) as exc:
            pickle.load(io.BytesIO(data))
        assert exc.value.rule_ids == ["MFV-PICKLE-004"]

    def test_concurrent_loads_on_two_threads(self):
        # One thread's guarded load does not cover another thread, and the
        # other thread's own guarded load is still checked.
        guard.install(prescan=False)
        _STARTED.clear()
        _RELEASE.clear()
        results = {}

        def slow_benign():
            results["benign"] = pickle.load(io.BytesIO(pickle.dumps(_Calls(_wait, None))))

        t = threading.Thread(target=slow_benign)
        t.start()
        assert _STARTED.wait(10)
        try:
            assert pickle.loads(EVAL) == 42
            with pytest.raises(guard.UnsafeModelError):
                _load(EVAL)
        finally:
            _RELEASE.set()
            t.join()
        assert results["benign"] is True

    @pytest.mark.parametrize("strict", [False, True])
    def test_in_process_ipc_is_untouched(self, strict):
        import socket

        guard.install(strict=strict)
        assert pickle.loads(pickle.dumps(socket.AF_INET)) == socket.AF_INET
        st = os.stat(".")
        assert pickle.loads(pickle.dumps(st)) == st

    @pytest.mark.parametrize("module", ["dill", "cloudpickle"])
    def test_dill_and_cloudpickle_own_loaders_are_untouched(self, module):
        # A known limit: these call no guarded loader by its patched name.
        lib = pytest.importorskip(module)
        guard.install(prescan=False)
        assert lib.loads(EVAL) == 42
        assert lib.loads(lib.dumps(BENIGN)) == BENIGN

    def test_dill_inside_a_guarded_load_is_checked(self):
        dill = pytest.importorskip("dill")
        guard.install(prescan=False)
        with pytest.raises(guard.UnsafeModelError):
            pickle.load(io.BytesIO(pickle.dumps(_Calls(dill.loads, EVAL))))


class TestLoadTimeCheck:

    @pytest.mark.parametrize("data, ref", [
        # A module that imports os hands out os itself.
        (b"clogging\nos\n.", "os"),
        # Protocol 4 walks a dotted name.
        (b"\x80\x04" + _short_binunicode("logging") + _short_binunicode("os.system") + b"\x93.",
         "os.system"),
        # Python 2 names that pickle maps onto denied ones.
        (b"\x80\x02ccPickle\nloads\n.", "pickle.loads"),
        (b"\x80\x02c__builtin__\neval\n.", "__builtin__.eval"),
    ], ids=["module-alias", "dotted-name", "cPickle", "__builtin__"])
    def test_aliases_are_refused(self, data, ref):
        guard.install(prescan=False)
        with pytest.raises(guard.UnsafeModelError) as exc:
            _load(data)
        assert ref in str(exc.value)

    def test_denied_module_is_refused_before_it_is_imported(self):
        if "test.support" in sys.modules:
            pytest.skip("test.support already imported")
        guard.install(prescan=False)
        with pytest.raises(guard.UnsafeModelError):
            _load(_call_pickle("test.support", "run_unittest", "x"))
        assert "test.support" not in sys.modules

    def test_unknown_global_is_allowed(self):
        guard.install(prescan=False)
        with pytest.raises(ModuleNotFoundError):
            _load(UNKNOWN)

    def test_warn_mode_and_never_let_it_run(self):
        guard.install(prescan=False, mode="warn")
        with pytest.warns(UserWarning, match="MFV-PICKLE-001"):
            assert _load(EVAL) == 42
        guard.install(prescan=False, fail_on="never")
        assert _load(EVAL) == 42

    def test_uninstall_turns_it_off(self):
        guard.install(prescan=False)
        guard.uninstall()
        assert _load(EVAL) == 42

    def test_argument_evidence_needs_the_prescan(self):
        # importlib.import_module("os"), then getattr(_, "getcwd")(): the
        # callable comes from a call result, never from find_class, so only
        # the pre-scan, reading the arguments, sees "os". A known limit.
        data = b"cbuiltins\ngetattr\n(cimportlib\nimport_module\n(Vos\ntRVgetcwd\ntR)R."
        guard.install(prescan=False)
        assert pickle.load(io.BytesIO(data)) == os.getcwd()
        guard.install()
        with pytest.raises(guard.UnsafeModelError) as exc:
            pickle.load(io.BytesIO(data))
        assert exc.value.rule_ids == ["MFV-PICKLE-005"]

    @pytest.mark.parametrize("protocol", [0, 2, 5])
    def test_strict_loads_stdlib_value_types(self, protocol):
        import datetime
        import decimal
        import pathlib
        import uuid
        meta = {"saved": datetime.datetime(2026, 10, 22, tzinfo=datetime.timezone.utc),
                "price": decimal.Decimal("1.5"), "run": uuid.UUID(int=7),
                "path": pathlib.PurePosixPath("/opt/models/a.bin")}
        guard.install(strict=True)
        assert _load(pickle.dumps(meta, protocol=protocol)) == meta

    def test_strict_refuses_an_unknown_global_before_it_runs(self):
        # Default mode lets unknown globals load, so a pickle built from them
        # can reach the guard's own state (a documented limit). Strict mode
        # refuses any global off the known-safe list before it is called.
        data = b"cbuiltins\ngetattr\n(cbuiltins\nstr\nVupper\ntR(Vx\ntR."
        guard.install(prescan=False)
        assert pickle.load(io.BytesIO(data)) == "X"
        guard.install(prescan=False, strict=True)
        with pytest.raises(guard.UnsafeModelError):
            pickle.load(io.BytesIO(data))

    @pytest.mark.parametrize("strict", [False, True])
    def test_multiprocessing_works(self, tmp_path, strict):
        # A spawned worker gets its Process object through pickle.load on a
        # pipe, and its tasks and results through pickle.loads. Neither is a
        # model load: os.getpid travels as a denied global.
        script = tmp_path / "pool.py"
        script.write_text(
            "import concurrent.futures, multiprocessing, os, sys, hayward.guard\n"
            "hayward.guard.install(strict=sys.argv[1] == 'True')\n"
            "if __name__ == '__main__':\n"
            "    ctx = multiprocessing.get_context('spawn')\n"
            "    with concurrent.futures.ProcessPoolExecutor(\n"
            "            1, mp_context=ctx, initializer=os.getpid) as ex:\n"
            "        print(ex.submit(len, 'abc').result(), ex.submit(os.getpid).result() > 0)\n"
            "    p = ctx.Process(target=os.getpid)\n"
            "    p.start()\n"
            "    p.join(30)\n"
            "    print(p.exitcode)\n"
        )
        env = {k: v for k, v in os.environ.items() if k != "HAYWARD_GUARD"}
        root = os.path.dirname(os.path.dirname(os.path.abspath(guard.__file__)))
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [root, env.get("PYTHONPATH")]))
        result = subprocess.run([sys.executable, str(script), str(strict)], env=env,
                                capture_output=True, text=True, timeout=120)
        assert result.stdout.split() == ["3", "True", "0"], result.stderr


class TestMoreBenignLoads:
    def test_savez_compressed_object_member(self, tmp_path, prescan):
        np = pytest.importorskip("numpy")
        p = tmp_path / "arrays.npz"
        np.savez_compressed(p, n=np.arange(3), o=np.array([BENIGN], dtype=object))
        guard.install(prescan=prescan)
        with np.load(p, allow_pickle=True) as z:
            assert z["o"][0] == BENIGN

    def test_sklearn_models(self, tmp_path, prescan):
        np = pytest.importorskip("numpy")
        joblib = pytest.importorskip("joblib")
        pytest.importorskip("sklearn")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(0)
        X, y = rng.normal(size=(40, 3)), rng.integers(0, 2, 40)
        models = [Pipeline([("s", StandardScaler()), ("lr", LogisticRegression())]).fit(X, y),
                  RandomForestClassifier(n_estimators=3, random_state=0).fit(X, y)]
        for i, model in enumerate(models):
            joblib.dump(model, tmp_path / f"m{i}.joblib", compress=3)
            (tmp_path / f"m{i}.pkl").write_bytes(pickle.dumps(model))
        guard.install(prescan=prescan)
        for i, model in enumerate(models):
            with open(tmp_path / f"m{i}.pkl", "rb") as f:
                loaded = [joblib.load(tmp_path / f"m{i}.joblib"), pickle.load(f)]
            for m in loaded:
                assert (m.predict(X) == model.predict(X)).all()

    def test_torch_module_and_torchscript(self, tmp_path, prescan):
        torch = pytest.importorskip("torch")
        net = torch.nn.Linear(2, 1)
        torch.save(net, tmp_path / "module.pt")
        torch.jit.save(torch.jit.script(net), tmp_path / "scripted.pt")
        guard.install(prescan=prescan)
        assert isinstance(torch.load(tmp_path / "module.pt", weights_only=False), torch.nn.Linear)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # torch: dispatching to torch.jit.load
            torch.load(tmp_path / "scripted.pt", weights_only=False)

    def test_dill_function(self, prescan):
        dill = pytest.importorskip("dill")
        guard.install(prescan=prescan)
        assert dill.loads(dill.dumps(lambda x: x + 1))(1) == 2


class TestStrict:
    """install(strict=True) refuses every global not on the known-safe list."""

    def test_known_safe_globals_load(self, prescan):
        import collections

        guard.install(strict=True, prescan=prescan)
        for protocol in range(6):
            assert _load(pickle.dumps(BENIGN, protocol=protocol)) == BENIGN
        od = collections.OrderedDict(a={1, 2})
        assert _load(pickle.dumps(od)) == od

    def test_unknown_global_is_refused_before_it_is_imported(self, prescan):
        guard.install(strict=True, prescan=prescan)
        with pytest.raises(guard.UnsafeModelError) as exc:
            _load(UNKNOWN)  # a ModuleNotFoundError would mean it got to the import
        assert exc.value.rule_ids == ["MFV-PICKLE-004"]
        assert "hayward_no_such_pkg.mod.Thing" in str(exc.value)

    def test_values_off_the_known_safe_list_are_refused(self):
        import ipaddress

        guard.install(strict=True, prescan=False)
        with pytest.raises(guard.UnsafeModelError) as exc:
            _load(pickle.dumps(ipaddress.ip_address("10.0.0.1")))
        assert "ipaddress.IPv4Address" in str(exc.value)

    def test_fail_on_does_not_relax_it(self):
        guard.install(strict=True, fail_on="never", prescan=False)
        for data in (UNKNOWN, EVAL):
            with pytest.raises(guard.UnsafeModelError):
                _load(data)

    def test_warn_mode_lets_the_load_run(self):
        guard.install(strict=True, mode="warn", prescan=False)
        with pytest.warns(UserWarning, match="MFV-PICKLE-004"), pytest.raises(ModuleNotFoundError):
            _load(UNKNOWN)

    def test_the_resolved_object_must_be_known_safe_too(self, monkeypatch):
        # A known-safe name that resolves to something else is refused under
        # the name of what it resolves to.
        import argparse
        import functools

        data = pickle.dumps(argparse.Namespace(a=1))
        monkeypatch.setattr(argparse, "Namespace", functools.partial)
        guard.install(strict=True, prescan=False)
        with pytest.raises(guard.UnsafeModelError) as exc:
            _load(data)
        assert "functools.partial" in str(exc.value)

    def test_prescan_refuses_unknown_globals(self, tmp_path):
        # Also where the load-time check cannot see: TorchScript is unpickled
        # in C++ by torch.jit.load.
        torch = pytest.importorskip("torch")
        p = tmp_path / "scripted.pt"
        torch.jit.save(torch.jit.script(torch.nn.Linear(2, 1)), p)
        guard.install(strict=True)
        with pytest.raises(guard.UnsafeModelError) as exc, warnings.catch_warnings():
            warnings.simplefilter("ignore")  # torch: dispatching to torch.jit.load
            torch.load(p, weights_only=False)
        assert exc.value.rule_ids == ["MFV-PICKLE-004"]

    def test_model_formats(self, tmp_path, prescan):
        np = pytest.importorskip("numpy")
        joblib = pytest.importorskip("joblib")
        p = tmp_path / "model.joblib"
        joblib.dump({"w": np.arange(6.0), "o": np.array(["a", "b"], dtype=object)}, p, compress=3)
        npy = tmp_path / "obj.npy"
        np.save(npy, np.array([BENIGN], dtype=object), allow_pickle=True)
        guard.install(strict=True, prescan=prescan)
        assert joblib.load(p)["o"].tolist() == ["a", "b"]
        assert np.load(npy, allow_pickle=True)[0] == BENIGN

    def test_torch_state_dict_and_sklearn(self, tmp_path, prescan):
        torch = pytest.importorskip("torch")
        joblib = pytest.importorskip("joblib")
        pytest.importorskip("sklearn")
        from sklearn.linear_model import LogisticRegression

        state = tmp_path / "state.pt"
        torch.save(torch.nn.Linear(2, 1).state_dict(), state)
        module = tmp_path / "module.pt"
        torch.save(torch.nn.Linear(2, 1), module)
        sk = tmp_path / "lr.joblib"
        joblib.dump(LogisticRegression().fit([[0.0], [1.0]], [0, 1]), sk)
        guard.install(strict=True, prescan=prescan)
        assert set(torch.load(state, weights_only=False)) == {"weight", "bias"}
        assert joblib.load(sk).predict([[1.0]]).tolist() == [1]
        # A whole nn.Module names its classes, which are not on the list.
        with pytest.raises(guard.UnsafeModelError):
            torch.load(module, weights_only=False)


# ---- remote code: transformers runs a repo's .py files from transformers_modules ----


def _run_remote_module(root, parts: list[str], source: str = "RAN = 40 + 2\n"):
    """Run a file the way transformers' get_class_in_module does: from its
    dynamic-module cache, through spec_from_file_location and exec_module."""
    import importlib.util

    folder = root.joinpath("transformers_modules", *parts)
    folder.mkdir(parents=True)
    path = folder / "modeling_toy.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("transformers_modules_test_toy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestRemoteCode:
    def test_remote_code_is_refused_by_default(self, tmp_path):
        guard.install()
        with pytest.raises(guard.UnsafeModelError) as exc:
            _run_remote_module(tmp_path, ["org", "repo", "abc123"])
        assert "MFV-HF-002" in exc.value.rule_ids
        assert "org/repo" in str(exc.value)

    def test_an_allowed_repo_runs(self, tmp_path):
        guard.install(remote_code=["org/repo"])
        assert _run_remote_module(tmp_path, ["org", "repo", "abc123"]).RAN == 42

    def test_an_allowed_repo_matches_transformers_sanitized_names(self, tmp_path):
        # transformers writes my-org/model.v2 as my_hyphen_org/model_dot_v2.
        guard.install(remote_code=["my-org/model.v2"])
        assert _run_remote_module(tmp_path, ["my_hyphen_org", "model_dot_v2", "abc123"]).RAN == 42

    def test_an_allowed_repo_matches_the_unsanitized_layout_too(self, tmp_path):
        # transformers 4.x kept the name as is.
        guard.install(remote_code=["my-org/model.v2"])
        assert _run_remote_module(tmp_path, ["my-org", "model.v2", "abc123"]).RAN == 42

    def test_a_single_string_is_rejected(self):
        with pytest.raises(ValueError):
            guard.install(remote_code="org/repo")

    def test_an_empty_name_is_rejected(self):
        with pytest.raises(ValueError):
            guard.install(remote_code=["/"])

    def test_strict_mode_refuses_whatever_fail_on_is(self, tmp_path):
        guard.install(strict=True, fail_on="never")
        with pytest.raises(guard.UnsafeModelError):
            _run_remote_module(tmp_path, ["org", "repo", "abc123"])

    def test_allowing_one_repo_does_not_allow_another(self, tmp_path):
        guard.install(remote_code=["org/repo"])
        with pytest.raises(guard.UnsafeModelError):
            _run_remote_module(tmp_path, ["org", "other", "abc123"])

    def test_a_local_folder_is_allowed_by_its_name(self, tmp_path):
        # A local model folder lands as transformers_modules/<folder name>/<hash>.
        guard.install(remote_code=["my_model"])
        assert _run_remote_module(tmp_path, ["my_model", "0f3a"]).RAN == 42

    def test_warn_mode_warns_and_runs(self, tmp_path):
        guard.install(mode="warn")
        with pytest.warns(UserWarning, match="MFV-HF-002"):
            assert _run_remote_module(tmp_path, ["org", "repo", "abc123"]).RAN == 42

    def test_threshold_above_high_lets_it_run(self, tmp_path):
        guard.install(fail_on="critical")
        assert _run_remote_module(tmp_path, ["org", "repo", "abc123"]).RAN == 42

    def test_uninstalled_guard_does_not_refuse(self, tmp_path):
        guard.install()
        guard.uninstall()
        assert _run_remote_module(tmp_path, ["org", "repo", "abc123"]).RAN == 42

    def test_other_code_is_untouched(self, tmp_path):
        guard.install()
        path = tmp_path / "ordinary.py"
        path.write_text("RAN = 42\n")
        import importlib.util
        spec = importlib.util.spec_from_file_location("ordinary_test_mod", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module.RAN == 42


# ---- install(verified=...): the verdict log ------------------------------------------------

CLEAN = [{"verdict": "clean", "coverage_gaps": []}]


@pytest.fixture
def log(monkeypatch):
    """A fake verdict log: file sha256 -> entries. Counts checkpoint fetches."""
    state = {"entries": {}, "fetches": 0, "signed": datetime.now(timezone.utc), "error": None}

    def checkpoint(*args, **kwargs):
        state["fetches"] += 1
        if state["error"] is not None:
            raise state["error"]
        return 7, {"subject": []}, state["signed"]

    def shard(statement, n, prefix, service=verify.SERVICE):
        return {d: e for d, e in state["entries"].items() if d.startswith(prefix)}

    monkeypatch.setattr(verify, "fetch_checkpoint", checkpoint)
    monkeypatch.setattr(verify, "fetch_shard", shard)
    return state


def _joblib_file(tmp_path, payload=None, raw: bytes | None = None) -> tuple[str, str]:
    joblib = pytest.importorskip("joblib")
    path = tmp_path / "model.joblib"
    if raw is not None:
        path.write_bytes(raw)
    else:
        joblib.dump(BENIGN if payload is None else payload, path)
    return str(path), hashlib.sha256(path.read_bytes()).hexdigest()


def _count_scans(monkeypatch) -> list[str]:
    seen: list[str] = []
    real = ModelFileScanner.scan_file

    def spy(self, path, **kwargs):
        seen.append(str(path))
        return real(self, path, **kwargs)

    monkeypatch.setattr(ModelFileScanner, "scan_file", spy)
    return seen


def test_a_vouched_file_loads_without_a_local_scan(tmp_path, log, monkeypatch):
    import joblib
    path, digest = _joblib_file(tmp_path)
    log["entries"][digest] = CLEAN
    scans = _count_scans(monkeypatch)
    guard.install(verified=True)
    assert joblib.load(path) == BENIGN
    assert scans == []


def test_a_file_without_a_verdict_is_scanned_locally_with_a_warning(tmp_path, log, monkeypatch):
    import joblib
    path, _ = _joblib_file(tmp_path)
    scans = _count_scans(monkeypatch)
    guard.install(verified=True)
    with pytest.warns(UserWarning, match="not vouched for"):
        assert joblib.load(path) == BENIGN
    assert scans


def test_a_malicious_file_without_a_verdict_is_still_refused(tmp_path, log):
    import joblib
    path, _ = _joblib_file(tmp_path, raw=EVAL)
    guard.install(verified=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(guard.UnsafeModelError):
            joblib.load(path)


def test_a_vouched_file_still_gets_the_load_time_check(tmp_path, log):
    # A wrong verdict skips only the pre-scan: the loader's own globals are still judged.
    import joblib
    path, digest = _joblib_file(tmp_path, raw=EVAL)
    log["entries"][digest] = CLEAN
    guard.install(verified=True)
    with pytest.raises(guard.UnsafeModelError):
        joblib.load(path)


@pytest.mark.parametrize("entries", [[], [{"verdict": "not_verified", "coverage_gaps": ["MFV-SKIP-001"]}],
                                     [{"verdict": "remote_code", "coverage_gaps": []}],
                                     [{"verdict": "clean", "coverage_gaps": ["MFV-SKIP-001"]}],
                                     [*CLEAN, {"verdict": "flagged", "coverage_gaps": []}]])
def test_strict_refuses_anything_not_vouched_for(tmp_path, log, entries):
    import joblib
    path, digest = _joblib_file(tmp_path)
    if entries:
        log["entries"][digest] = entries
    guard.install(verified="strict")
    with pytest.raises(guard.UnsafeModelError, match="not vouched for"):
        joblib.load(path)


def test_an_unreachable_log_falls_back_and_strict_refuses(tmp_path, log):
    import joblib
    path, digest = _joblib_file(tmp_path)
    log["entries"][digest] = CLEAN
    log["error"] = URLError("offline")
    guard.install(verified=True)
    with pytest.warns(UserWarning, match="could not be checked"):
        assert joblib.load(path) == BENIGN
    with pytest.warns(UserWarning, match="could not be checked"):  # the same reason, not retried
        assert joblib.load(path) == BENIGN
    assert log["fetches"] == 1
    guard.install(verified="strict")
    with pytest.raises(guard.UnsafeModelError, match="could not be checked"):
        joblib.load(path)


def test_a_missing_signing_extra_is_named(tmp_path, log, monkeypatch):
    import joblib
    path, _ = _joblib_file(tmp_path)

    def no_sigstore(*args, **kwargs):
        raise ImportError("No module named 'sigstore'")

    monkeypatch.setattr(verify, "fetch_checkpoint", no_sigstore)
    guard.install(verified="strict")
    for _ in range(2):
        with pytest.raises(guard.UnsafeModelError, match=r"hayward\[sign\]"):
            joblib.load(path)


def test_a_stale_checkpoint_vouches_for_nothing(tmp_path, log):
    import joblib
    path, digest = _joblib_file(tmp_path)
    log["entries"][digest] = CLEAN
    log["signed"] = datetime.now(timezone.utc) - timedelta(days=4)
    guard.install(verified="strict")
    with pytest.raises(guard.UnsafeModelError, match="over 3 days"):
        joblib.load(path)


def test_the_log_and_the_hash_are_looked_up_once(tmp_path, log, monkeypatch):
    import joblib
    path, digest = _joblib_file(tmp_path)
    log["entries"][digest] = CLEAN
    hashed = []
    real = guard._hash_file
    monkeypatch.setattr(guard, "_hash_file", lambda p: hashed.append(p) or real(p))
    guard.install(verified=True)
    joblib.load(path)
    joblib.load(path)
    assert log["fetches"] == 1 and len(hashed) == 1


def test_a_stream_cannot_be_looked_up(log):
    data = io.BytesIO(pickle.dumps(BENIGN))
    guard.install(verified=True)
    with pytest.warns(UserWarning, match="stream"):
        assert pickle.load(data) == BENIGN
    guard.install(verified="strict")
    with pytest.raises(guard.UnsafeModelError, match="stream"):
        pickle.load(io.BytesIO(pickle.dumps(BENIGN)))


def test_strict_lets_a_vouched_file_load_through_nested_loaders(tmp_path, log):
    # numpy.load hands an object array to pickle.load on the same open file.
    np = pytest.importorskip("numpy")
    path = tmp_path / "objects.npy"
    np.save(path, np.array([{"a": 1}], dtype=object), allow_pickle=True)
    log["entries"][hashlib.sha256(path.read_bytes()).hexdigest()] = CLEAN
    guard.install(verified="strict")
    assert np.load(path, allow_pickle=True)[0] == {"a": 1}


def test_the_log_is_not_consulted_by_default(tmp_path, log):
    import joblib
    path, _ = _joblib_file(tmp_path)
    guard.install()
    assert joblib.load(path) == BENIGN
    assert log["fetches"] == 0


def test_verified_takes_true_false_or_strict():
    with pytest.raises(ValueError):
        guard.install(verified="yes")
