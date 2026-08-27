"""HW-134: proof that scanning makes no outbound calls.

The scanner's contract is pure static analysis of local bytes: it must
never dial out, whatever the file contains. socket.socket,
socket.create_connection, socket.getaddrinfo and socket.gethostbyname
are monkeypatched to record the attempt and raise; every scan below must
complete and the record must stay empty.

(The record is the real detector: scan_file's exception firewall would
catch the AssertionError and degrade to MFV-SKIP-003, which would still
look like "the scan completed".)
"""

from __future__ import annotations

import json
import os
import pickle
import socket
import struct
import zipfile
from pathlib import Path

import pytest

from hayward.scanner import ModelFileScanner


@pytest.fixture()
def no_network(monkeypatch):
    attempts: list[str] = []

    def deny(label: str):
        attempts.append(label)
        raise AssertionError(f"hayward attempted network access via {label}")

    class _DeniedSocket:
        def __init__(self, *args, **kwargs):
            deny("socket.socket")

    monkeypatch.setattr(socket, "socket", _DeniedSocket)
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: deny("socket.create_connection"))
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: deny("socket.getaddrinfo"))
    monkeypatch.setattr(socket, "gethostbyname", lambda *a, **k: deny("socket.gethostbyname"))
    return attempts


# ── tiny inline fixtures, one per format family ─────────────────────


class _Evil:
    def __reduce__(self):
        return (os.system, ("echo pwned",))


def _pickle_bytes() -> bytes:
    # Malicious on purpose: if this still convicts under the socket ban,
    # the engine demonstrably ran, not merely returned early.
    return pickle.dumps(_Evil(), protocol=2)


def _safetensors_bytes() -> bytes:
    header = json.dumps(
        {"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    ).encode()
    return struct.pack("<Q", len(header)) + header + b"\x00" * 4


def _gguf_bytes() -> bytes:
    def gguf_string(s: bytes) -> bytes:
        return struct.pack("<Q", len(s)) + s

    kvs = [(b"general.architecture", b"llama")]
    header = b"GGUF" + struct.pack("<IQQ", 3, 0, len(kvs))
    body = b"".join(gguf_string(k) + struct.pack("<I", 8) + gguf_string(v) for k, v in kvs)
    return header + body


def _npy_bytes() -> bytes:
    header = b"{'descr': '|i1', 'fortran_order': False, 'shape': (1,)}"
    total = 10 + len(header) + 1
    header += b" " * ((16 - total % 16) % 16) + b"\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + b"\x00"


def _write_fixture_set(root: Path) -> list[Path]:
    files = {
        "model.pkl": _pickle_bytes(),
        "model.safetensors": _safetensors_bytes(),
        "model.gguf": _gguf_bytes(),
        "model.npy": _npy_bytes(),
    }
    pt = root / "model.pt"
    with zipfile.ZipFile(pt, "w") as zf:
        zf.writestr("archive/data.pkl", pickle.dumps({"w": [1.0]}, protocol=2))
    out = [pt]
    for name, blob in files.items():
        p = root / name
        p.write_bytes(blob)
        out.append(p)
    return out


# ── the proof ───────────────────────────────────────────────────────


def test_scan_file_makes_no_network_calls(no_network, tmp_path):
    scanner = ModelFileScanner()
    all_findings = []
    for p in _write_fixture_set(tmp_path):
        all_findings.extend(scanner.scan_file(p))

    assert no_network == []
    # Sanity: the scans did real work -- the malicious pickle convicted.
    assert any(f.rule_id == "MFV-PICKLE-001" for f in all_findings)


def test_scan_directory_makes_no_network_calls(no_network, tmp_path):
    _write_fixture_set(tmp_path)

    findings = ModelFileScanner().scan_directory(tmp_path)

    assert no_network == []
    assert any(f.rule_id == "MFV-PICKLE-001" for f in findings)
