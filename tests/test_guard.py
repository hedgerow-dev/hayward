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


def _scan_spy(monkeypatch) -> list[int]:
    """Record the permission bits of every file the guard hands the scanner."""
    modes: list[int] = []
    real = ModelFileScanner.scan_file

    def spy(self, path):
        modes.append(stat.S_IMODE(os.stat(path).st_mode))
        return real(self, path)

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


class TestPickle:
    @pytest.mark.parametrize("wrap", [bytes, bytearray, memoryview])
    def test_loads_blocks_malicious(self, wrap):
        guard.install()
        with pytest.raises(guard.UnsafeModelError) as exc:
            pickle.loads(wrap(EVAL))
        assert isinstance(exc.value, pickle.UnpicklingError)
        assert exc.value.rule_ids == ["MFV-PICKLE-001"]
        assert "MFV-PICKLE-001" in str(exc.value)
        assert "builtins.eval" in str(exc.value)

    def test_loads_passes_benign(self):
        guard.install()
        assert pickle.loads(pickle.dumps(BENIGN)) == BENIGN

    def test_load_blocks_malicious_file(self, tmp_path):
        guard.install()
        p = tmp_path / "evil.pkl"
        p.write_bytes(EVAL)
        with open(p, "rb") as f, pytest.raises(guard.UnsafeModelError) as exc:
            pickle.load(f)
        assert exc.value.file == str(p)
        assert str(p) in str(exc.value)

    def test_pickle_never_treats_a_str_as_a_path(self, tmp_path):
        p = tmp_path / "evil.pkl"
        p.write_bytes(EVAL)
        guard.install()
        with pytest.raises(TypeError):
            pickle.loads(str(p))
        with pytest.raises(TypeError):
            pickle.load(p)

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

        assert pickle.loads(pickle.dumps(BENIGN)) == BENIGN
        with pytest.raises(guard.UnsafeModelError):
            pickle.loads(EVAL)

        assert len(modes) == 2
        assert list(spool.iterdir()) == []
        if os.name == "posix":
            assert modes == [0o600, 0o600]

    def test_warn_mode_lets_the_load_run(self):
        guard.install(mode="warn")
        with pytest.warns(UserWarning, match="MFV-PICKLE-001"):
            assert pickle.loads(EVAL) == 42

    def test_threshold(self):
        guard.install()
        # Below `high`: the guard lets it through and pickle fails on the import.
        with pytest.raises(ModuleNotFoundError):
            pickle.loads(UNKNOWN)
        guard.install(fail_on="info")
        with pytest.raises(guard.UnsafeModelError):
            pickle.loads(UNKNOWN)

    def test_coverage_gap_warns_but_never_blocks(self, monkeypatch):
        monkeypatch.setattr(ModelFileScanner, "MAX_SCAN_BYTES", 8)
        guard.install(fail_on="info")
        with pytest.warns(UserWarning, match="MFV-SKIP-001"):
            assert pickle.loads(pickle.dumps(BENIGN)) == BENIGN

    def test_bad_options_are_rejected(self):
        with pytest.raises(ValueError):
            guard.install(fail_on="severe")
        with pytest.raises(ValueError):
            guard.install(mode="log")
        assert pickle.load is ORIGINAL_LOAD

    def test_uninstall_restores_originals(self):
        guard.install()
        assert pickle.load is not ORIGINAL_LOAD
        assert pickle.loads is not ORIGINAL_LOADS
        guard.uninstall()
        assert pickle.load is ORIGINAL_LOAD
        assert pickle.loads is ORIGINAL_LOADS

    def test_reinstall_replaces_settings_without_double_wrapping(self):
        guard.install()
        guard.install(mode="warn")
        with pytest.warns(UserWarning):
            assert pickle.loads(EVAL) == 42
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

    def test_benign_checkpoint_loads(self, tmp_path):
        p = tmp_path / "model.pt"
        self.torch.save({"w": self.torch.ones(2)}, p)
        guard.install()
        assert self.torch.load(p)["w"].tolist() == [1.0, 1.0]
        with open(p, "rb") as f:
            assert self.torch.load(f)["w"].tolist() == [1.0, 1.0]
