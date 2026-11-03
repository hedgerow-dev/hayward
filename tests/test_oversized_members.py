"""A member past MAX_ZIP_MEMBER_BYTES is never skipped silently.

Each container used to drop an oversized member with no finding, so a few
hundred bytes of compressed padding hid a payload and the file scanned clean.
Now the start of the member is scanned when a loader reads it from the start
(a pickle, an object .npy), and MFV-SKIP-002 says the rest was not read.
"""

from __future__ import annotations

import io
import os
import pickle
import tarfile
import zipfile

import numpy as np
import pytest

from hayward import ModelFileScanner

CAP = 4096
PAD = b"\0" * (CAP * 4)


class _Evil:
    def __reduce__(self):
        return (os.system, ("id",))


EVIL = pickle.dumps(_Evil(), protocol=2)


def _scanner():
    scanner = ModelFileScanner()
    scanner.MAX_ZIP_MEMBER_BYTES = CAP
    return scanner


def _torch_zip(data_pkl: bytes, weights: bytes = PAD) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("archive/data.pkl", data_pkl)
        zf.writestr("archive/data/0", weights)
        zf.writestr("archive/version", "3\n")
    return buf.getvalue()


def _rules(findings):
    return {(f.rule_id, f.severity.value) for f in findings}


def test_a_mar_holding_an_oversized_checkpoint_is_not_verified(tmp_path):
    path = tmp_path / "model.mar"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("model.pt", _torch_zip(EVIL))
        zf.writestr("MAR-INF/MANIFEST.json", "{}")
    assert ("MFV-SKIP-002", "low") in _rules(_scanner().scan_file(path))


def test_a_nemo_holding_an_oversized_checkpoint_is_not_verified(tmp_path):
    path = tmp_path / "model.nemo"
    blob = _torch_zip(EVIL)
    with tarfile.open(path, "w:gz") as tf:
        info = tarfile.TarInfo("model_weights.ckpt")
        info.size = len(blob)
        tf.addfile(info, io.BytesIO(blob))
    assert ("MFV-SKIP-002", "low") in _rules(_scanner().scan_file(path))


def test_an_oversized_pickle_inside_a_nested_checkpoint_has_its_start_scanned(tmp_path):
    # The inner archive fits under the cap; its deflated data.pkl does not.
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("archive/data.pkl", EVIL + PAD)
        zf.writestr("archive/version", "3\n")
    blob = inner.getvalue()
    assert len(blob) < CAP
    path = tmp_path / "model.nemo"
    with tarfile.open(path, "w") as tf:
        info = tarfile.TarInfo("model_weights.ckpt")
        info.size = len(blob)
        tf.addfile(info, io.BytesIO(blob))
    rules = _rules(_scanner().scan_file(path))
    assert ("MFV-PICKLE-001", "critical") in rules and ("MFV-SKIP-002", "low") in rules


def test_an_oversized_object_npy_in_an_npz_is_scanned_from_its_start(tmp_path):
    arr = np.empty(1, dtype=object)
    arr[0] = _Evil()
    member = io.BytesIO()
    np.save(member, arr, allow_pickle=True)
    path = tmp_path / "data.npz"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("x.npy", member.getvalue() + PAD)
    rules = _rules(_scanner().scan_file(path))
    assert ("MFV-PICKLE-001", "critical") in rules and ("MFV-SKIP-002", "low") in rules


def test_an_oversized_numeric_npy_in_an_npz_says_nothing(tmp_path):
    member = io.BytesIO()
    np.save(member, np.zeros(CAP, dtype=np.float64))
    path = tmp_path / "data.npz"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("x.npy", member.getvalue())
    assert _scanner().scan_file(path) == []


@pytest.mark.parametrize("name", ["notes.bin", "weights.dat"])
def test_an_oversized_zip_in_a_compressed_tar_is_not_verified(tmp_path, name):
    # Seeking inside a gzip stream means inflating it again, so it is not streamed.
    path = tmp_path / "bundle.tar.gz"
    blob = _torch_zip(EVIL)
    with tarfile.open(path, "w:gz") as tf:
        info = tarfile.TarInfo(name)
        info.size = len(blob)
        tf.addfile(info, io.BytesIO(blob))
    assert ("MFV-SKIP-002", "low") in _rules(_scanner().scan_file(path))


def test_an_npz_past_the_file_cap_still_reads_its_object_arrays(tmp_path):
    arr = np.empty(1, dtype=object)
    arr[0] = _Evil()
    member = io.BytesIO()
    np.save(member, arr, allow_pickle=True)
    path = tmp_path / "data.npz"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("x.npy", member.getvalue())
        zf.writestr("pad.npy", PAD)
    scanner = _scanner()
    scanner.MAX_SCAN_BYTES = 10_000
    assert ("MFV-PICKLE-001", "critical") in _rules(scanner.scan_file(path))


def _set_encrypted_flag(data: bytes, name: bytes) -> bytes:
    """Set bit 0 of the general-purpose flags for `name` in both headers."""
    out = bytearray(data)
    for sig, flag_at, name_at in ((b"PK\x03\x04", 6, 30), (b"PK\x01\x02", 8, 46)):
        pos = out.find(sig)
        while pos != -1:
            if out[pos + name_at:pos + name_at + len(name)] == name:
                out[pos + flag_at] |= 1
            pos = out.find(sig, pos + 4)
    return bytes(out)


def test_an_oversized_nested_checkpoint_behind_a_lied_flag_is_not_verified(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("model.pt", _torch_zip(EVIL))
    path = tmp_path / "model.mar"
    path.write_bytes(_set_encrypted_flag(buf.getvalue(), b"model.pt"))
    assert ("MFV-SKIP-002", "low") in _rules(_scanner().scan_file(path))


def test_an_unreadable_member_is_not_called_oversized(tmp_path):
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("archive/data.pkl", EVIL)
    raw = bytearray(inner.getvalue())
    for sig, method_at in ((b"PK\x03\x04", 8), (b"PK\x01\x02", 10)):
        pos = raw.find(sig)
        raw[pos + method_at] = 99  # a compression method zipfile cannot read
    path = tmp_path / "model.nemo"
    with tarfile.open(path, "w") as tf:
        info = tarfile.TarInfo("model_weights.ckpt")
        info.size = len(raw)
        tf.addfile(info, io.BytesIO(bytes(raw)))
    findings = _scanner().scan_file(path)
    assert not any(f.metadata.get("skipped_reason") == "oversized_member" for f in findings)


# ---- streamed: the outer layer is uncompressed, so the nested checkpoint is read in place


def _plain_tar(path, name: str, blob: bytes):
    with tarfile.open(path, "w") as tf:  # NeMo writes .nemo with "w:", uncompressed
        info = tarfile.TarInfo(name)
        info.size = len(blob)
        tf.addfile(info, io.BytesIO(blob))


def _stored_mar(path, blob: bytes):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:  # torch-model-archiver zip-store
        zf.writestr("model.pt", blob)
        zf.writestr("MAR-INF/MANIFEST.json", "{}")


@pytest.mark.parametrize("pack", ["nemo", "mar"])
def test_an_oversized_checkpoint_in_an_uncompressed_outer_is_read_in_place(tmp_path, pack):
    path = tmp_path / f"model.{pack}"
    blob = _torch_zip(EVIL)
    assert len(blob) > CAP
    if pack == "nemo":
        _plain_tar(path, "model_weights.ckpt", blob)
    else:
        _stored_mar(path, blob)
    rules = _rules(_scanner().scan_file(path))
    assert ("MFV-PICKLE-001", "critical") in rules
    assert ("MFV-SKIP-002", "low") not in rules


@pytest.mark.parametrize("pack", ["nemo", "mar"])
def test_a_benign_oversized_checkpoint_in_an_uncompressed_outer_is_clean(tmp_path, pack):
    path = tmp_path / f"model.{pack}"
    blob = _torch_zip(pickle.dumps({"w": 1}, protocol=2))
    if pack == "nemo":
        _plain_tar(path, "model_weights.ckpt", blob)
    else:
        _stored_mar(path, blob)
    assert _scanner().scan_file(path) == []
