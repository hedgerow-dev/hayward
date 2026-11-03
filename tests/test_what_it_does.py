"""Pickle findings carry one line from `hayward explain` saying what the pickle would do."""

from __future__ import annotations

import collections
import pickle

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
