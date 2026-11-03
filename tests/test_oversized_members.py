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
def test_an_oversized_zip_in_a_tar_is_not_verified(tmp_path, name):
    path = tmp_path / "bundle.tar"
    blob = _torch_zip(EVIL)
    with tarfile.open(path, "w") as tf:
        info = tarfile.TarInfo(name)
        info.size = len(blob)
        tf.addfile(info, io.BytesIO(blob))
    assert ("MFV-SKIP-002", "low") in _rules(_scanner().scan_file(path))
