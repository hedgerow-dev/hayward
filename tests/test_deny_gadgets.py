"""Specific deny-list gadgets that a re-measurement surfaced as gaps.

`uuid._get_command_stdout` is a living-off-the-land gadget: uuid uses it to
shell out for the host MAC address, so handed attacker arguments it runs an
arbitrary command. It was in the unknown (INFO) tier because uuid is a
legitimate module and only the specific callable is dangerous. It is now denied
by name, without denying the whole uuid module (which would flag an ordinary
pickled uuid.UUID).
"""

from __future__ import annotations

from hayward import ModelFileScanner
from hayward.scanner import _classify_pickle_global


def _short_binunicode(text: str) -> bytes:
    raw = text.encode()
    return bytes([0x8C, len(raw)]) + raw


def _reduce_pickle(module: str, name: str, arg: str) -> bytes:
    """A protocol-4 pickle calling module.name(arg)."""
    return (
        b"\x80\x04"
        + _short_binunicode(module)
        + _short_binunicode(name)
        + b"\x93"
        + _short_binunicode(arg)
        + b"\x85"
        + b"R."
    )


def test_uuid_get_command_stdout_is_denied():
    assert _classify_pickle_global("uuid._get_command_stdout") == "denied"
    assert _classify_pickle_global("uuid._popen") == "denied"


def test_uuid_gadget_pickle_is_convicted(tmp_path):
    p = tmp_path / "model.pkl"
    p.write_bytes(_reduce_pickle("uuid", "_get_command_stdout", "id"))
    findings = ModelFileScanner().scan_file(p)
    assert any(f.rule_id == "MFV-PICKLE-001" for f in findings), (
        [(f.rule_id, f.message) for f in findings]
    )


def test_ordinary_uuid_is_not_flagged(tmp_path):
    # The whole uuid module is NOT denied: a benign pickled uuid.UUID must stay
    # clean, which is why the wildcard was rejected in favour of named gadgets.
    assert _classify_pickle_global("uuid.UUID") == "unknown"
    p = tmp_path / "id.pkl"
    p.write_bytes(_reduce_pickle("uuid", "UUID", "12345678123456781234567812345678"))
    findings = ModelFileScanner().scan_file(p)
    assert not any(f.rule_id == "MFV-PICKLE-001" for f in findings), (
        [(f.rule_id, f.message) for f in findings]
    )
