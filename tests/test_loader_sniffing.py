"""Content the real loaders sniff must be scanned whatever the file is called.

torch.load, numpy.load, joblib.load and pickle.load decide what a file is by
its bytes, not its name. Each case here is content one of them runs that the
scanner used to pass clean because of the extension it wore. Standard library
only: the payloads are hand-built bytes.
"""

from __future__ import annotations

import bz2
import datetime
import decimal
import fractions
import gzip
import io
import lzma
import pathlib
import pickle
import tarfile
import uuid
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


# os.system as a protocol 2 pickle writes it on Linux: a plain GLOBAL opcode naming
# "posix" (os.system's real __module__), one string argument, REDUCE. The command is
# harmless; the shape is what matters.
def _posix_system_p2(command: str = "echo hayward-test") -> bytes:
    raw = command.encode()
    return (b"\x80\x02cposix\nsystem\nq\x00X" + len(raw).to_bytes(4, "little") + raw
            + b"q\x01\x85q\x02Rq\x03.")


class TestPosixSystemInLegacyTar:
    """A pickle that calls posix.system, wrapped as an old-style tar checkpoint whose
    only member is `storages`, the layout torch.load still reads. Seen in real files
    on the Hub: the tar is a single 10 KB block, and the name is usually
    pytorch_model.bin even though torch reads it as a checkpoint."""

    def test_tar_is_one_padded_block_like_the_real_files(self):
        assert len(_tar([("storages", _posix_system_p2())])) == 10240

    @pytest.mark.parametrize("name", ["pytorch_model.bin", "legacy.pt", "legacy.pth"])
    def test_posix_system_is_critical_under_each_name(self, tmp_path, name):
        findings = _scan(tmp_path, name, _tar([("storages", _posix_system_p2())]))
        critical = [f for f in findings if f.rule_id == "MFV-PICKLE-001"]
        assert critical, [(f.rule_id, f.severity) for f in findings]
        assert critical[0].severity == Severity.CRITICAL
        assert "posix.system" in critical[0].message

    def test_a_shell_pipeline_argument_changes_nothing(self, tmp_path):
        command = "OUT=$(echo aGVsbG8= | base64 -d | timeout 60 sh 2>&1)"
        findings = _scan(tmp_path, "pytorch_model.bin", _tar([("storages", _posix_system_p2(command))]))
        assert _critical(findings)


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


# builtins.eval("40+2") with no PROTO in front: it opens with none of the
# scanner's pickle openers, and torch still unpickles it under a known name.
EVAL_BARE = EVAL[2:]


def _torch_tar(extra: list[tuple[str, bytes | None, bytes, str]]) -> bytes:
    """A legacy torch tar: harmless `storages` and `tensors`, then `extra` as
    (name, data, tar type, linkname)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:", format=tarfile.USTAR_FORMAT) as tf:
        base = [("storages", pickle.dumps(0, protocol=2) + pickle.dumps([], protocol=2)),
                ("tensors", pickle.dumps(0, protocol=2))]
        for name, data in base:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        for name, data, kind, linkname in extra:
            info = tarfile.TarInfo(name)
            info.type = kind
            info.linkname = linkname
            if data is None:
                tf.addfile(info)
            else:
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class TestTorchTarMemberResolution:
    """torch.load reads `pickle` with tarfile.extractfile, which follows links
    and serves an unknown member type as a regular file."""

    @pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE])
    def test_pickle_member_that_is_a_link(self, tmp_path, kind):
        data = _torch_tar([("blob", EVAL_BARE, tarfile.REGTYPE, ""),
                           ("pickle", None, kind, "blob")])
        assert _critical(_scan(tmp_path, "legacy.pt", data))

    def test_pickle_member_with_an_unknown_type(self, tmp_path):
        data = _torch_tar([("pickle", EVAL_BARE, b"Z", "")])
        assert _critical(_scan(tmp_path, "legacy.pt", data))

    def test_dangling_link_is_a_coverage_gap(self, tmp_path):
        data = _torch_tar([("pickle", None, tarfile.SYMTYPE, "missing")])
        assert any(is_coverage_gap(f) for f in _scan(tmp_path, "legacy.pt", data))

    def test_oversized_member_is_read_up_to_the_cap_and_reported(self, tmp_path):
        p = tmp_path / "legacy.pt"
        p.write_bytes(_torch_tar([("pickle", EVAL + b"\x00" * 200, tarfile.REGTYPE, "")]))
        scanner = ModelFileScanner()
        scanner.MAX_ZIP_MEMBER_BYTES = 64
        findings = scanner.scan_file(p)
        assert _critical(findings)
        assert any(is_coverage_gap(f) for f in findings)

    @pytest.mark.parametrize("name", ["pytorch_model.bin", "model.dat"])
    def test_legacy_tar_under_an_unmapped_name(self, tmp_path, name):
        data = _torch_tar([("pickle", EVAL_BARE, tarfile.REGTYPE, "")])
        assert _critical(_scan(tmp_path, name, data))

    def test_benign_legacy_tar_is_clean(self, tmp_path):
        data = _torch_tar([("pickle", pickle.dumps({"w": 1}, protocol=2), tarfile.REGTYPE, "")])
        assert _scan(tmp_path, "legacy.pt", data) == []


class TestOversizedTar:
    """Past the size cap a file torch.load reads as a tar is walked from disk,
    even when zipfile also finds an archive at its end."""

    @pytest.mark.parametrize("with_zip", [True, False])
    def test_oversized_tar_is_walked(self, tmp_path, with_zip):
        data = _torch_tar([("pickle", EVAL_BARE, tarfile.REGTYPE, "")])
        if with_zip:
            zbuf = io.BytesIO()
            with zipfile.ZipFile(zbuf, "w") as zf:
                zf.writestr("archive/version", "3\n")
            data += zbuf.getvalue()
        p = tmp_path / "big.pt"
        p.write_bytes(data)
        scanner = ModelFileScanner()
        scanner.MAX_SCAN_BYTES = 64
        findings = scanner.scan_file(p)
        assert _critical(findings)
        # Read from byte 0 as a pickle, the file is still past the cap.
        assert any(f.rule_id == "MFV-SKIP-001" for f in findings)


class TestMixedLzmaStreams:
    """joblib reads lzma and xz through lzma.LZMAFile, which decodes every
    stream in either format."""

    @pytest.mark.parametrize("first,second", [
        (lzma.FORMAT_ALONE, lzma.FORMAT_XZ), (lzma.FORMAT_XZ, lzma.FORMAT_ALONE),
    ], ids=["lzma-then-xz", "xz-then-lzma"])
    @pytest.mark.parametrize("name", ["model.joblib", "model.pkl"])
    def test_mixed_formats(self, tmp_path, first, second, name):
        data = (lzma.compress(b"ccollections\nOrderedDict\n0", format=first)
                + lzma.compress(EVAL_P0, format=second))
        assert _critical(_scan(tmp_path, name, data))

    def test_undecodable_later_stream_is_a_coverage_gap(self, tmp_path):
        data = (lzma.compress(pickle.dumps({"w": 1}, protocol=2), format=lzma.FORMAT_XZ)
                + b"\xfd7zXZ\x00 not an xz stream")
        assert any(is_coverage_gap(f) for f in _scan(tmp_path, "model.joblib", data))

    def test_zero_padding_after_the_stream_is_clean(self, tmp_path):
        data = lzma.compress(pickle.dumps({"w": 1}, protocol=2), format=lzma.FORMAT_XZ)
        assert _scan(tmp_path, "model.joblib", data + b"\x00" * 8) == []


def _npy(header: bytes, payload: bytes) -> bytes:
    total = 10 + len(header) + 1
    header += b" " * ((64 - total % 64) % 64) + b"\n"
    return b"\x93NUMPY\x01\x00" + len(header).to_bytes(2, "little") + header + payload


class TestNpyHeaderParity:
    """The .npy header is read the way numpy's _read_array_header reads it."""

    @pytest.mark.parametrize("descr", ["object", "object_", "<f8,object", "(2,)O", "T"])
    def test_object_dtype_spellings(self, tmp_path, descr):
        header = b"{'descr': '%s', 'fortran_order': False, 'shape': (1,), }" % descr.encode()
        assert _critical(_scan(tmp_path, "arr.npy", _npy(header, EVAL)))

    def test_structured_object_field(self, tmp_path):
        header = b"{'descr': [('a', '<f8'), ('b', '|O')], 'fortran_order': False, 'shape': (1,), }"
        assert _critical(_scan(tmp_path, "arr.npy", _npy(header, EVAL)))

    def test_python2_long_in_shape(self, tmp_path):
        header = b"{'descr': '|O', 'fortran_order': False, 'shape': (1L,), }"
        assert _critical(_scan(tmp_path, "arr.npy", _npy(header, EVAL)))

    def test_unparseable_header_is_a_coverage_gap(self, tmp_path):
        header = b"{'descr': '|O', 'fortran_order': False, 'shape': (1,) oops"
        assert any(is_coverage_gap(f) for f in _scan(tmp_path, "arr.npy", _npy(header, EVAL)))

    def test_numeric_structured_array_is_clean(self, tmp_path):
        header = b"{'descr': [('Open', '<f8'), ('Obs', '<i4')], 'fortran_order': False, 'shape': (2,), }"
        assert _scan(tmp_path, "arr.npy", _npy(header, b"\x00" * 24)) == []


def _sbu(text: str) -> bytes:
    return _short_binunicode(text)


def _stack_global(module: str, name: str) -> bytes:
    return _sbu(module) + _sbu(name) + b"\x93"


def _binint(value: int) -> bytes:
    return b"J" + value.to_bytes(4, "little", signed=True)


def _numpy_dtype(typestr: str = "f8", flags: int = 0) -> bytes:
    # numpy's own pickled form: numpy.dtype(typestr, False, True), then
    # BUILD with (3, '<', None, None, None, elsize, alignment, flags).
    return (_stack_global("numpy", "dtype") + _sbu(typestr) + b"\x89\x88\x87R"
            + b"(K\x03" + _sbu("<") + b"NNN" + _binint(-1) + _binint(-1) + _binint(flags)
            + b"tb")


def _joblib_array(raw: bytes, *, dtype: bytes | None = None, count: int = 16,
                  align: bool = True) -> bytes:
    """A NumpyArrayWrapper the way joblib.dump writes one, followed by the
    array bytes it splices into the stream."""
    state = (b"}(" + _sbu("subclass") + _stack_global("numpy", "ndarray")
             + _sbu("shape") + b"K" + bytes([count]) + b"\x85"
             + _sbu("order") + _sbu("C")
             + _sbu("dtype") + (_numpy_dtype() if dtype is None else dtype))
    if align:
        state += _sbu("numpy_array_alignment_bytes") + b"K\x10"
    return (_stack_global("joblib.numpy_pickle", "NumpyArrayWrapper") + b")\x81"
            + state + b"ub" + raw)


EVAL_CALL = _stack_global("builtins", "eval") + _sbu("40+2") + b"\x85R"


def _joblib_list(*items: bytes) -> bytes:
    return b"\x80\x04](" + b"".join(items) + b"e."


class TestJoblibInlineArrays:
    """joblib splices a numeric array's raw bytes into the pickle right after
    the wrapper's BUILD and carries on with the same pickle."""

    def test_payload_after_an_aligned_array(self, tmp_path):
        data = _joblib_list(_joblib_array(b"\x03\xff\xff\xff" + b"\x00" * 128), EVAL_CALL)
        assert _critical(_scan(tmp_path, "model.joblib", data))
        assert _critical(_scan(tmp_path, "model.pkl", data))

    def test_payload_after_an_unaligned_array(self, tmp_path):
        # joblib <= 1.1 wrote no alignment padding.
        data = _joblib_list(_joblib_array(b"\x00" * 128, align=False), EVAL_CALL)
        assert _critical(_scan(tmp_path, "model.joblib", data))

    def test_payload_after_an_array_in_a_gzip_file(self, tmp_path):
        data = _joblib_list(_joblib_array(b"\x00" + b"\x00" * 128), EVAL_CALL)
        assert _critical(_scan(tmp_path, "model.joblib", gzip.compress(data)))

    def test_array_that_cannot_be_sized_is_a_coverage_gap(self, tmp_path):
        data = _joblib_list(_joblib_array(b"\x00" * 128, dtype=_sbu("f8")), EVAL_CALL)
        assert any(is_coverage_gap(f) for f in _scan(tmp_path, "model.joblib", data))

    def test_plain_pickle_reading_is_kept(self, tmp_path):
        # pickle.load does not skip array bytes: what follows the BUILD is
        # opcodes to it, so it is still walked as opcodes.
        data = _joblib_list(_joblib_array(EVAL_CALL + b"0" * 200, align=False))
        assert _critical(_scan(tmp_path, "model.pkl", data))

    def test_benign_numeric_and_object_arrays_are_clean(self, tmp_path):
        nested = b"\x80\x05" + b"](K\x01K\x02e."
        data = _joblib_list(
            _joblib_array(b"\x03\xff\xff\xff" + b"\x00" * 128),
            _joblib_array(b"\x00" * 128, align=False),
            _joblib_array(nested, dtype=_numpy_dtype("O8", flags=63)),
            _sbu("tail"),
        )
        assert _scan(tmp_path, "model.joblib", data) == []


class TestPython2Names:
    """Below protocol 3 the unpickler maps Python 2 names to Python 3 ones
    (pickle._compat_pickle) before it imports anything."""

    @pytest.mark.parametrize("data", [
        b"\x80\x02ccPickle\nloads\nC\x02N.\x85R.",
        b"ccPickle\nloads\n(S'N.'\ntR.",
        b"\x80\x02" + _stack_global("cPickle", "loads") + b"C\x02N.\x85R.",
    ], ids=["proto2-global", "proto0", "proto2-stack-global"])
    def test_cpickle_loads_is_mapped(self, tmp_path, data):
        assert _critical(_scan(tmp_path, "model.pkl", data))


class TestStdlibValueTypes:
    """Timestamps, prices, ids and config paths in ordinary model metadata are
    on the known-safe list, under every protocol."""

    VALUES = {
        "datetime": datetime.datetime(2026, 10, 22, 9, 30, tzinfo=datetime.timezone.utc),
        "timezone": datetime.timezone(datetime.timedelta(hours=2), "CEST"),
        "date": datetime.date(2026, 10, 22),
        "decimal": decimal.Decimal("1.5"),
        "uuid": uuid.UUID(int=12345),
        "fraction": fractions.Fraction(1, 3),
        "posix-path": pathlib.PurePosixPath("/opt/models/a.bin"),
        "windows-path": pathlib.PureWindowsPath("C:/models/a.bin"),
    }

    @pytest.mark.parametrize("protocol", range(6))
    @pytest.mark.parametrize("name", sorted(VALUES))
    def test_scans_clean(self, tmp_path, name, protocol):
        data = pickle.dumps(self.VALUES[name], protocol=protocol)
        assert _scan(tmp_path, "meta.pkl", data) == []

    def test_path_argument_still_counts_for_other_callables(self, tmp_path):
        data = b"clinecache\ngetline\n(V/etc/hosts\nI1\ntR."
        findings = _scan(tmp_path, "model.pkl", data)
        assert any(f.rule_id == "MFV-PICKLE-005" for f in findings)


class TestTransformersExactList:
    """transformers is allowed by exact name only: its namespace loads files
    and code by design, and real pickles use only Trainer's training classes."""

    @pytest.mark.parametrize("ref", [
        "transformers.training_args.TrainingArguments",
        "transformers.trainer_utils.IntervalStrategy",
        "transformers.trainer_utils.SchedulerType",
    ])
    def test_training_classes_are_allowed(self, ref):
        from hayward._pickle_engine import _classify_pickle_global
        assert _classify_pickle_global(ref) == "allowed"

    @pytest.mark.parametrize("ref", [
        "transformers.AutoModel",
        "transformers.pipelines.TextGenerationPipeline",
        "transformers.trainer.Trainer",
    ])
    def test_other_class_shaped_names_are_not(self, ref):
        from hayward._pickle_engine import _classify_pickle_global
        assert _classify_pickle_global(ref) != "allowed"

    def test_sklearn_shape_rule_is_unchanged(self):
        from hayward._pickle_engine import _classify_pickle_global
        assert _classify_pickle_global("sklearn.linear_model._logistic.LogisticRegression") == "allowed"
