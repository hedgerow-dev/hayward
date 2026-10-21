"""The load guard: `import hayward.guard` scans a file before a loader runs it.

The malicious fixtures call `builtins.eval("40+2")`. That is a CRITICAL
finding, and harmless if the guard ever fails to stop it: the test then sees
42 instead of an exception and fails.

numpy, joblib and torch tests skip when the library is absent; the pickle
tests need nothing beyond the standard library.
"""

from __future__ import annotations

import io
import os
import pickle
import stat
import subprocess
import sys
import tempfile

import pytest

from hayward import guard
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

    def test_loads_is_not_patched(self):
        # An in-memory IPC and cache hot path, not a model-file load.
        guard.install()
        assert pickle.loads is ORIGINAL_LOADS

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

    def test_coverage_gap_blocks(self, monkeypatch):
        # The guard runs the file after scanning it, so a file it could not
        # fully read is not loaded, whatever the threshold.
        monkeypatch.setattr(ModelFileScanner, "MAX_SCAN_BYTES", 8)
        guard.install(fail_on="never")
        with pytest.raises(guard.UnsafeModelError) as exc:
            _load(pickle.dumps(BENIGN))
        assert exc.value.rule_ids == ["MFV-SKIP-001"]
        assert "could not fully scan" in str(exc.value)
        assert 'install(mode="warn")' in str(exc.value)

    def test_coverage_gap_warns_and_loads_in_warn_mode(self, monkeypatch):
        monkeypatch.setattr(ModelFileScanner, "MAX_SCAN_BYTES", 8)
        guard.install(mode="warn")
        with pytest.warns(UserWarning, match="MFV-SKIP-001"):
            assert _load(pickle.dumps(BENIGN)) == BENIGN

    def test_max_size_lets_a_large_file_be_scanned(self, monkeypatch):
        # Past the cap a non-zip file is a coverage gap and refused; raising
        # the cap scans it instead, without turning blocking off.
        monkeypatch.setattr(ModelFileScanner, "MAX_SCAN_BYTES", 8)
        guard.install()
        with pytest.raises(guard.UnsafeModelError) as exc:
            _load(pickle.dumps(BENIGN))
        assert exc.value.rule_ids == ["MFV-SKIP-001"]
        guard.install(max_size=1_000_000)
        assert _load(pickle.dumps(BENIGN)) == BENIGN
        with pytest.raises(guard.UnsafeModelError) as exc:
            _load(EVAL)
        assert exc.value.rule_ids == ["MFV-PICKLE-001"]

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

    def test_unparseable_stream_is_refused(self):
        # The scanner cannot read it, and pickle runs it next: not a clean
        # verdict, whatever pickle would make of it.
        guard.install()
        with pytest.raises(guard.UnsafeModelError) as exc:
            _load(b"\xff\xfe")
        assert exc.value.rule_ids == ["MFV-SKIP-003"]

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
    def test_content_the_scanner_cannot_read_is_refused(self, tmp_path, name):
        # Under an unclaimed name the scanner reports nothing for it; the
        # loader would unpickle it anyway, so the guard does not run it.
        p = tmp_path / name
        p.write_bytes(b"\xff\xfe not a pickle")
        guard.install()
        for load in (lambda: self.np.load(p, allow_pickle=True), lambda: self.joblib.load(p)):
            with pytest.raises(guard.UnsafeModelError) as exc:
                load()
            assert exc.value.rule_ids == ["MFV-SKIP-003"]

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

    @pytest.mark.parametrize("compress", [0, 3, ("gzip", 3), ("bz2", 3), ("lzma", 3), ("xz", 3)])
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
