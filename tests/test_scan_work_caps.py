"""Caps on the work one scan can be made to do, each reported, never silent."""

from __future__ import annotations

import pickle

from hayward import ModelFileScanner
from hayward import scanner as scanner_module

CALL = b"cos\nsystem\n(S'id'\ntR"


def _scan(tmp_path, name, data):
    path = tmp_path / name
    path.write_bytes(data)
    return {(f.rule_id, f.metadata.get("skipped_reason")) for f in ModelFileScanner().scan_file(path)}


def test_the_pickle_walk_stops_at_its_opcode_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(scanner_module, "_PICKLE_MAX_OPS", 1000)
    found = _scan(tmp_path, "m.pkl", b"\x80\x02" + b"N0" * 5000 + CALL + b".")
    assert ("MFV-SKIP-003", "pickle_walk_budget") in found


def test_a_payload_before_the_budget_runs_out_is_still_flagged(tmp_path, monkeypatch):
    monkeypatch.setattr(scanner_module, "_PICKLE_MAX_OPS", 1000)
    found = _scan(tmp_path, "m.pkl", b"\x80\x02" + CALL + b"0" + b"N0" * 5000 + b".")
    assert ("MFV-PICKLE-001", None) in found and ("MFV-SKIP-003", "pickle_walk_budget") in found


def test_an_ordinary_pickle_stays_inside_the_budget(tmp_path):
    data = pickle.dumps({f"layer.{i}.weight": [0.0] * 8 for i in range(2000)}, protocol=2)
    assert _scan(tmp_path, "m.pkl", data) == set()


def test_a_checkpoint_stops_reading_members_past_its_byte_budget(tmp_path):
    # Many small members that each decompress to a lot: no one member is past
    # the member cap, but together they are past the per-file budget.
    import zipfile
    scanner = ModelFileScanner()
    scanner.MAX_ZIP_MEMBER_BYTES = 10_000
    scanner.MAX_ZIP_FILE_READ_BYTES = 25_000
    path = tmp_path / "model.pt"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for i in range(10):
            zf.writestr(f"archive/m{i}.pkl", b"\x80\x02N." + b"\0" * 9_000)
        zf.writestr("archive/z.pkl", b"\x80\x02" + CALL + b".")
    found = {(f.rule_id, f.metadata.get("skipped_reason")) for f in scanner.scan_file(path)}
    assert ("MFV-SKIP-002", "file_read_budget") in found


def test_the_byte_budget_also_stops_inside_a_nested_archive(tmp_path):
    import io
    import zipfile
    scanner = ModelFileScanner()
    scanner.MAX_ZIP_MEMBER_BYTES = 10_000
    scanner.MAX_ZIP_FILE_READ_BYTES = 25_000
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w", zipfile.ZIP_DEFLATED) as zf:
        for i in range(10):
            zf.writestr(f"archive/m{i}.pkl", b"\x80\x02N." + b"\0" * 9_000)
    path = tmp_path / "model.mar"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("model.pt", inner.getvalue())
    found = {(f.rule_id, f.metadata.get("skipped_reason")) for f in scanner.scan_file(path)}
    assert ("MFV-SKIP-002", "file_read_budget") in found


def test_a_nested_pickle_found_before_the_budget_runs_out_is_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(scanner_module, "_PICKLE_MAX_OPS", 1000)
    inner = b"\x80\x02" + CALL + b"."
    data = b"\x80\x02C" + bytes([len(inner)]) + inner + b"0" + b"N0" * 5000 + b"."
    found = _scan(tmp_path, "m.pkl", data)
    assert ("MFV-PICKLE-008", None) in found


def test_a_nested_pickle_past_the_budget_is_not_verified(tmp_path, monkeypatch):
    # The outer walk sees one bytes literal; the walk of the pickle inside it
    # runs out. That must be reported, not read as "nothing found inside".
    monkeypatch.setattr(scanner_module, "_PICKLE_MAX_OPS", 1000)
    inner = b"\x80\x02" + b"N0" * 2000 + CALL + b"."
    data = (b"\x80\x02cnumpy\nload\n" + b"cio\nBytesIO\n" + b"B" + len(inner).to_bytes(4, "little")
            + inner + b"\x85R\x85R.")
    found = _scan(tmp_path, "m.pkl", data)
    assert ("MFV-SKIP-003", "pickle_walk_budget") in found


def test_many_tiny_pickles_do_not_escape_the_budget(monkeypatch):
    # The truncation check starts a fresh walk per pickle; the budget spans them all.
    from hayward import _pickle_engine
    monkeypatch.setattr(scanner_module, "_PICKLE_MAX_OPS", 1000)
    walked = [0]
    real = _pickle_engine.genops

    def counting(data):
        for item in real(data):
            walked[0] += 1
            yield item

    monkeypatch.setattr(_pickle_engine, "genops", counting)
    _pickle_engine._pickle_stream_truncated(b"\x80\x02N." * 5000)
    assert walked[0] <= 1100
