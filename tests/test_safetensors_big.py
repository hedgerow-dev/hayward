"""A SafeTensors file over the scan cap is checked by its header, not skipped.

Everything a SafeTensors loader acts on (the header's dtypes, shapes, offsets and
metadata) is in the first 8 + header_size bytes; the rest is raw tensor bytes that
no loader executes. Like a big zip checkpoint (only the pickle member is read) and
a big Keras file (only the config is read), a big SafeTensors file is read by its
header instead of being reported as never scanned. Files here are sparse, so the
tests stay fast; the cap is lowered for the same reason.
"""

from __future__ import annotations

import json
import struct

import pytest

from hayward import ModelFileScanner

CAP = 4096


def _write(path, header: dict, data_bytes: int):
    raw = json.dumps(header).encode()
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw)
        f.truncate(8 + len(raw) + data_bytes)  # sparse: the tensor bytes are never written


@pytest.fixture
def scan(monkeypatch):
    monkeypatch.setattr(ModelFileScanner, "MAX_SCAN_BYTES", CAP)
    return lambda path: ModelFileScanner().scan_file(path)


def test_a_big_valid_file_is_checked_not_skipped(tmp_path, scan):
    p = tmp_path / "model.safetensors"
    _write(p, {"w": {"dtype": "F32", "shape": [1024, 64], "data_offsets": [0, 1024 * 64 * 4]}}, 1024 * 64 * 4)
    assert scan(p) == []


def test_a_big_file_with_a_bad_layout_is_still_caught(tmp_path, scan):
    p = tmp_path / "model.safetensors"
    _write(p, {"w": {"dtype": "F32", "shape": [1024, 64], "data_offsets": [0, 1 << 40]}}, 1024 * 64 * 4)
    assert "MFV-ST-006" in [f.rule_id for f in scan(p)]


def test_a_big_file_with_suspicious_metadata_is_still_caught(tmp_path, scan):
    p = tmp_path / "model.safetensors"
    header = {"__metadata__": {"__reduce__": "os.system"},
              "w": {"dtype": "F32", "shape": [1024, 64], "data_offsets": [0, 1024 * 64 * 4]}}
    _write(p, header, 1024 * 64 * 4)
    assert "MFV-ST-005" in [f.rule_id for f in scan(p)]


def test_a_header_that_runs_past_the_file_is_caught(tmp_path, scan):
    p = tmp_path / "model.safetensors"
    p.write_bytes(struct.pack("<Q", 50_000) + b"{" + b" " * 100)
    with p.open("r+b") as f:
        f.truncate(CAP * 2)  # over the cap, but shorter than the header claims
    assert "MFV-ST-003" in [f.rule_id for f in scan(p)]


def test_a_local_receipt_says_a_big_file_was_read_by_its_header(tmp_path):
    from hayward import cli

    d = tmp_path / "m"
    d.mkdir()
    _write(d / "big.safetensors", {"w": {"dtype": "U8", "shape": [CAP * 2], "data_offsets": [0, CAP * 2]}}, CAP * 2)
    _write(d / "small.safetensors", {"w": {"dtype": "U8", "shape": [16], "data_offsets": [0, 16]}}, 16)
    out = tmp_path / "r.json"
    cli.main(["scan", str(d), "--max-size", str(CAP), "--receipt", str(out), "-f", "json", "-o", str(tmp_path / "x.json")])
    files = json.loads(out.read_text())["predicate"]["files"]
    assert files["big.safetensors"]["read"] == "header"
    assert "read" not in files["small.safetensors"]
