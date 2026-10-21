"""Content the real loaders sniff must be scanned whatever the file is called.

torch.load, numpy.load, joblib.load and pickle.load decide what a file is by
its bytes, not its name. Each case here is content one of them runs that the
scanner used to pass clean because of the extension it wore. Standard library
only: the payloads are hand-built bytes.
"""

from __future__ import annotations

import bz2
import gzip
import io
import lzma
import pickle
import tarfile
import zipfile
import zlib

import pytest

from hayward.findings import Severity, is_coverage_gap
from hayward.scanner import ModelFileScanner


def _short_binunicode(text: str) -> bytes:
    raw = text.encode()
    return bytes([0x8C, len(raw)]) + raw


# builtins.eval("40+2"), protocol 4: CRITICAL, harmless if ever executed.
EVAL = (
    b"\x80\x04" + _short_binunicode("builtins") + _short_binunicode("eval")
    + b"\x93" + _short_binunicode("40+2") + b"\x85R."
)
# The same call in protocol 0, which opens with GLOBAL rather than PROTO.
EVAL_P0 = b"cbuiltins\neval\n(S'40+2'\ntR."


def _npy_object(payload: bytes) -> bytes:
    header = b"{'descr': '|O', 'fortran_order': False, 'shape': (1,), }"
    total = 10 + len(header) + 1
    header += b" " * ((64 - total % 64) % 64) + b"\n"
    return b"\x93NUMPY\x01\x00" + len(header).to_bytes(2, "little") + header + payload


def _tar(members: list[tuple[str, bytes]]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:", format=tarfile.PAX_FORMAT) as tf:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _scan(tmp_path, name: str, data: bytes):
    p = tmp_path / name
    p.write_bytes(data)
    return ModelFileScanner().scan_file(p)


def _flagged(findings) -> bool:
    return any(f.severity in (Severity.CRITICAL, Severity.HIGH) for f in findings)


def _critical(findings) -> bool:
    return any(f.rule_id == "MFV-PICKLE-001" for f in findings)


class TestProtocolZeroPth:
    def test_protocol0_pth_is_scanned(self, tmp_path):
        assert _critical(_scan(tmp_path, "model.pth", EVAL_P0))

    def test_protocol0_pth_with_leading_opcodes_and_garbage(self, tmp_path):
        # Opens with INT/POP, not GLOBAL, and ends in a byte no opcode owns.
        # pickle runs the eval before it reaches the garbage.
        assert _critical(_scan(tmp_path, "model.pth", b"I5\n0" + EVAL_P0[:-1] + b"\xff"))

    @pytest.mark.parametrize("protocol", range(6))
    def test_benign_pickles_in_pth_scan_clean(self, tmp_path, protocol):
        data = pickle.dumps({"w": [1.0, 2.0], "n": "layer"}, protocol=protocol)
        assert _scan(tmp_path, "model.pth", data) == []


class TestCompressedPickles:
    @pytest.mark.parametrize("compress", [
        gzip.compress, bz2.compress, zlib.compress,
        lambda d: lzma.compress(d, format=lzma.FORMAT_XZ),
        lambda d: lzma.compress(d, format=lzma.FORMAT_ALONE),
    ], ids=["gzip", "bz2", "zlib", "xz", "lzma"])
    @pytest.mark.parametrize("name", ["model.pkl", "model.pth", "model.pt", "model.bin"])
    def test_compressed_pickle_is_decompressed(self, tmp_path, compress, name):
        # joblib.load decompresses any of these whatever the file is called.
        assert _critical(_scan(tmp_path, name, compress(EVAL)))

    @pytest.mark.parametrize("compress", [
        gzip.compress, bz2.compress, zlib.compress,
        lambda d: lzma.compress(d, format=lzma.FORMAT_XZ),
        lambda d: lzma.compress(d, format=lzma.FORMAT_ALONE),
    ], ids=["gzip", "bz2", "zlib", "xz", "lzma"])
    def test_pickle_split_across_concatenated_streams(self, tmp_path, compress):
        # joblib reads bz2, xz and lzma through the standard library's
        # readers, which carry on into a following stream. The first half
        # resolves a harmless global, so it does not read as unparseable.
        first, second = b"ccollections\nOrderedDict\n)R0", EVAL_P0
        data = compress(first) + compress(second) + b"\x00" * 8
        assert _critical(_scan(tmp_path, "model.joblib", data))

    def test_lzma_with_a_small_dictionary(self, tmp_path):
        # joblib detects lzma on two bytes (5d 00); a 4KB dictionary makes the
        # third byte non-zero.
        data = lzma.compress(EVAL, format=lzma.FORMAT_ALONE,
                             filters=[{"id": lzma.FILTER_LZMA1, "dict_size": 4096}])
        assert data[2] != 0
        assert _critical(_scan(tmp_path, "model.joblib", data))

    def test_joblib_compat_zfile(self, tmp_path):
        # joblib < 0.10 wrote "ZF", a padded hex length, then zlib data, and
        # joblib.load still reads it.
        data = b"ZF" + hex(len(EVAL)).ljust(19).encode() + zlib.compress(EVAL)
        assert _critical(_scan(tmp_path, "model.joblib", data))
        assert _critical(_scan(tmp_path, "model.pkl", data))

    def test_lz4_is_a_coverage_gap(self, tmp_path):
        findings = _scan(tmp_path, "model.joblib", b"\x04\x22\x4d\x18" + b"\x00" * 32)
        assert any(is_coverage_gap(f) for f in findings)

    @pytest.mark.parametrize("name", ["model.joblib", "model.pkl"])
    def test_corrupt_or_lz4_stream_is_a_coverage_gap(self, tmp_path, name):
        for data in (b"\x1f\x8b\x08\x00" + b"\xff" * 32, b"\x04\x22\x4d\x18" + b"\x00" * 32):
            assert any(is_coverage_gap(f) for f in _scan(tmp_path, name, data))

    def test_benign_compressed_pickle_named_pkl_is_not_a_mismatch(self, tmp_path):
        # joblib.dump(..., "model.pkl", compress=...) is ordinary use.
        for compress in (zlib.compress, gzip.compress, bz2.compress):
            data = compress(pickle.dumps({"w": [1.0]}, protocol=4))
            assert _scan(tmp_path, "model.pkl", data) == []


class TestTarCheckpoints:
    def test_tar_with_a_zip_appended(self, tmp_path):
        # zipfile finds the trailing archive; torch.load sees no zip magic at
        # byte 0 and opens the tar.
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, "w") as zf:
            zf.writestr("archive/data.pkl", pickle.dumps({"a": 1}, protocol=2))
        data = _tar([("storages", EVAL)]) + zbuf.getvalue()
        assert _critical(_scan(tmp_path, "legacy.pt", data))

    def test_legacy_torch_tar_pt(self, tmp_path):
        data = _tar([("storages", EVAL), ("tensors", b""), ("pickle", b"")])
        assert _critical(_scan(tmp_path, "legacy.pt", data))

    def test_torch_tar_member_not_opening_with_a_pickle_opener(self, tmp_path):
        # torch unpickles `storages` whatever its first opcode is.
        data = _tar([("storages", b"I1\n0" + EVAL_P0)])
        assert _critical(_scan(tmp_path, "legacy.pt", data))

    def test_tar_header_that_is_a_pickle(self, tmp_path):
        # pickle.load and joblib.load read a tar from byte 0 as a pickle, and
        # the first member name sits there.
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:", format=tarfile.USTAR_FORMAT) as tf:
            info = tarfile.TarInfo(EVAL_P0.decode())
            info.size = 2
            tf.addfile(info, io.BytesIO(b"N."))
        assert _critical(_scan(tmp_path, "model.pkl", buf.getvalue()))


class TestZipPolyglot:
    def test_pickle_before_a_zip(self, tmp_path):
        # zipfile finds the archive at the end; torch, pickle and joblib read
        # the pickle at the start.
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, "w") as zf:
            zf.writestr("archive/data.pkl", pickle.dumps({"a": 1}, protocol=2))
        assert _critical(_scan(tmp_path, "model.pt", EVAL + zbuf.getvalue()))

        # Past the size cap a zip is still walked lazily; the prefix too.
        p = tmp_path / "big.pt"
        p.write_bytes(EVAL + zbuf.getvalue())
        scanner = ModelFileScanner()
        scanner.MAX_SCAN_BYTES = 64
        assert _critical(scanner.scan_file(p))


class TestOtherExtensions:
    @pytest.mark.parametrize("name", ["model.onnx", "graph.pb"])
    def test_pickle_under_a_protobuf_extension(self, tmp_path, name):
        assert _flagged(_scan(tmp_path, name, EVAL))

    @pytest.mark.parametrize("name", ["weights.npy", "weights.npz", "weights.pb"])
    def test_torch_zip_under_another_name(self, tmp_path, name):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("archive/data.pkl", EVAL)
        assert _critical(_scan(tmp_path, name, buf.getvalue()))
        # Past the size cap too, where the zip is walked lazily.
        scanner = ModelFileScanner()
        scanner.MAX_SCAN_BYTES = 64
        assert _critical(scanner.scan_file(tmp_path / name))

    def test_object_npy_named_npz(self, tmp_path):
        assert _critical(_scan(tmp_path, "arrays.npz", _npy_object(EVAL)))

    def test_npz_member_without_npy_suffix(self, tmp_path):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("evil", _npy_object(EVAL))
        assert _critical(_scan(tmp_path, "arrays.npz", buf.getvalue()))


class TestUnparsedPickleIsNotClean:
    """`report_unparsed_pickles` (set by a load-time caller): bytes a loader would
    unpickle that the walk cannot read are a coverage gap. Off by default, so
    the CLI keeps its measured silence on them."""

    @staticmethod
    def _strict_scan(tmp_path, name: str, data: bytes):
        p = tmp_path / name
        p.write_bytes(data)
        scanner = ModelFileScanner()
        scanner.report_unparsed_pickles = True
        return scanner.scan_file(p)

    @pytest.mark.parametrize("name", ["model.pkl", "model.pt", "model.joblib"])
    def test_unparseable_content_is_a_coverage_gap(self, tmp_path, name):
        data = b"\xff\xfe not a pickle at all"
        assert any(is_coverage_gap(f) for f in self._strict_scan(tmp_path, name, data))
        assert _scan(tmp_path, name, data) == []

    def test_unparseable_object_npy_payload_is_a_coverage_gap(self, tmp_path):
        findings = self._strict_scan(tmp_path, "arrays.npy", _npy_object(b"\xff\xfe"))
        assert any(is_coverage_gap(f) for f in findings)

    def test_unparseable_torch_data_pkl_is_a_coverage_gap(self, tmp_path):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("archive/data.pkl", b"\xff\xfe")
        findings = self._strict_scan(tmp_path, "model.pt", buf.getvalue())
        assert any(is_coverage_gap(f) for f in findings)

    @pytest.mark.parametrize("protocol", range(6))
    def test_benign_pickles_scan_clean(self, tmp_path, protocol):
        data = pickle.dumps({"w": [1.0, 2.0], "n": "layer"}, protocol=protocol)
        for name in ("model.pkl", "model.pt", "model.joblib"):
            assert self._strict_scan(tmp_path, name, data) == []
