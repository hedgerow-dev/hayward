"""INT, LONG, FLOAT, PUT and GET arguments are read the way the loaders read them.

pickletools parses `I` and `L` lines in base 10. On Python 3.10 to 3.13 the
loaders use base 0 (and the C one also takes a leading-zero octal and stops at a
NUL), so `I0x1` loads but made the opcode walk raise before the import that
followed. A payload behind it scanned clean. The C loader also stops a FLOAT
or a PUT/GET index at a NUL, where pickletools raised.
"""

from __future__ import annotations

import pytest

from hayward import ModelFileScanner
from hayward._pickle_engine import genops

CALL = b"cos\nsystem\n(S'id'\ntR."


@pytest.mark.parametrize("prefix", [
    b"\x80\x02I0x1\n0",   # hex INT, then POP
    b"(I0x1\n1",          # protocol 0: MARK, hex INT, POP_MARK
    b"\x80\x02L0x1L\n0",  # hex LONG
    b"\x80\x02I0o7\n0",   # octal, Python spelling
    b"\x80\x02I017\n0",   # octal, C strtol spelling
    b"\x80\x02I0b1\n0",   # binary
    b"\x80\x02I1\x00x\n0",  # C stops at the NUL
    b"\x80\x02F1.0\x00\n0",  # FLOAT, C stops at the NUL
    b"(F1\x00\n1",
    b"\x80\x02Np1\x00\n0",   # PUT, C stops at the NUL
    b"\x80\x02Np1\n0g1\x00\n0",  # GET
    b"]S'\xe9'\n0",          # STRING: a latin1 or bytes load reads non-ASCII
    b"(S'\xe9'\n1",
    b"P\\\n0",               # PERSID: no escapes, for a persistent_load hook
])
def test_payload_behind_a_loader_only_integer_is_flagged(tmp_path, prefix):
    path = tmp_path / "model.pkl"
    path.write_bytes(prefix + CALL)
    findings = ModelFileScanner().scan_file(path)
    assert any(f.severity.value in ("critical", "high") for f in findings), [
        (f.rule_id, f.severity) for f in findings
    ]


def test_walk_reads_loader_integers():
    ops = [(op.name, arg) for op, arg, _ in genops(b"I0x10\nL0x0fL\nI01\n.")]
    assert ops == [("INT", 16), ("LONG", 15), ("INT", True), ("STOP", None)]


def test_walk_still_rejects_what_no_loader_reads():
    with pytest.raises(ValueError):
        list(genops(b"Iabc\n."))


def test_walk_reads_strings_and_names_like_the_loaders():
    # Python 3.10 and 3.11 pickletools decoded these as ASCII and raised.
    data = b"S'\xe9'\ncbuiltins\n\xc3\xa9\n."
    ops = [(op.name, arg) for op, arg, _ in genops(data)]
    assert ops == [("STRING", "\xe9"), ("GLOBAL", "builtins \xe9"), ("STOP", None)]
