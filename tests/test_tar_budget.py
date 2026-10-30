"""Issue #10: walking a tar must not let a small file force unbounded work.

Skipping a member of a compressed tar means decompressing it, and a GNU long-name or
pax header has its declared size read into memory inside tarfile itself. The limits
are lowered here so the tests stay fast; the shapes are the real ones.
"""

from __future__ import annotations

import bz2
import gzip
import io
import lzma
import tarfile

import pytest

from hayward import ModelFileScanner

# builtins.eval("40+2"), protocol 4: CRITICAL, harmless if ever executed.
EVAL = b"\x80\x04\x8c\x08builtins\x8c\x04eval\x93\x8c\x0440+2\x85R."


def tar_bytes(members: list[tuple[str, bytes]], fmt=tarfile.PAX_FORMAT) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=fmt) as tf:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def scan(tmp_path, name: str, data: bytes):
    p = tmp_path / name
    p.write_bytes(data)
    return ModelFileScanner().scan_file(p)


def reasons(findings):
    return {f.metadata.get("skipped_reason") for f in findings if f.metadata}


def denied(findings):
    return [f for f in findings if f.rule_id == "MFV-PICKLE-001"]


ZEROS = b"\0" * 3_000_000  # compresses to a few kilobytes


def test_a_compressed_tar_stops_at_the_decompression_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(ModelFileScanner, "_TAR_MAX_DECOMPRESSED_BYTES", 1_000_000)
    data = gzip.compress(tar_bytes([("model_config.yaml", ZEROS), ("model_weights.ckpt", EVAL)]))
    findings = scan(tmp_path, "model.nemo", data)
    assert "decompression_budget" in reasons(findings)
    assert not denied(findings)  # the walk stopped before the member past the budget


def test_within_the_budget_the_payload_is_still_found(tmp_path):
    data = gzip.compress(tar_bytes([("model_config.yaml", ZEROS), ("model_weights.ckpt", EVAL)]))
    findings = scan(tmp_path, "model.nemo", data)
    assert denied(findings)
    assert "decompression_budget" not in reasons(findings)


def test_an_uncompressed_tar_has_no_decompression_budget(tmp_path, monkeypatch):
    # Skipping a member of a plain tar is a seek, not work, so the budget does not apply.
    monkeypatch.setattr(ModelFileScanner, "_TAR_MAX_DECOMPRESSED_BYTES", 1_000_000)
    findings = scan(tmp_path, "model.nemo", tar_bytes([("model_config.yaml", ZEROS), ("model_weights.ckpt", EVAL)]))
    assert denied(findings)
    assert "decompression_budget" not in reasons(findings)


@pytest.mark.parametrize("compress", [False, True])
def test_an_oversized_long_name_header_is_refused(tmp_path, monkeypatch, compress):
    monkeypatch.setattr(ModelFileScanner, "_TAR_MAX_EXTENDED_HEADER_BYTES", 4096)
    long_name = "a/" * 5000 + "x.ckpt"  # a 10 KB GNU long-name header
    data = tar_bytes([("first.txt", b"hello"), (long_name, b"data"), ("model_weights.ckpt", EVAL)],
                     fmt=tarfile.GNU_FORMAT)
    findings = scan(tmp_path, "model.nemo", gzip.compress(data) if compress else data)
    gap = [f for f in findings if f.metadata and f.metadata.get("skipped_reason") == "tar_walk_failed"]
    assert gap, [(f.rule_id, f.metadata) for f in findings]
    assert "header" in gap[0].message


def test_an_oversized_pax_header_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(ModelFileScanner, "_TAR_MAX_EXTENDED_HEADER_BYTES", 4096)
    data = tar_bytes([("first.txt", b"hello"), ("b" * 9000, b"data"), ("model_weights.ckpt", EVAL)])
    findings = scan(tmp_path, "model.nemo", gzip.compress(data))
    assert "tar_walk_failed" in reasons(findings)


def test_ordinary_long_names_still_work(tmp_path):
    data = tar_bytes([("d/" * 60 + "model_weights.ckpt", EVAL)], fmt=tarfile.GNU_FORMAT)
    findings = scan(tmp_path, "model.nemo", gzip.compress(data))
    assert denied(findings)


# ---- compressed tars are routed now that the walk has a budget -----------------------------

COMPRESS = {".tgz": gzip.compress, ".tbz2": bz2.compress, ".txz": lzma.compress,
            ".tar.gz": gzip.compress, ".tar.bz2": bz2.compress, ".tar.xz": lzma.compress}
TRAVERSAL = tar_bytes([("../../etc/evil.sh", b"echo pwned")])


@pytest.mark.parametrize("suffix", sorted(COMPRESS))
def test_each_compressed_tar_form_is_scanned(tmp_path, suffix):
    findings = scan(tmp_path, f"archive{suffix}", COMPRESS[suffix](TRAVERSAL))
    assert any(f.rule_id == "MFV-ARCHIVE-001" for f in findings), [(f.rule_id, f.message[:60]) for f in findings]


@pytest.mark.parametrize("suffix", [".tar.gz", ".tgz"])
def test_a_directory_scan_finds_compressed_tars(tmp_path, suffix):
    (tmp_path / f"bundle{suffix}").write_bytes(COMPRESS[suffix](TRAVERSAL))
    found = ModelFileScanner().discover_files(tmp_path)
    assert [p.name for p in found] == [f"bundle{suffix}"]


def test_a_pickle_renamed_tgz_is_still_caught(tmp_path):
    # Not a tar at all: falls through to the content sniff, like a renamed .tar.
    findings = scan(tmp_path, "model.tgz", EVAL)
    assert denied(findings)
