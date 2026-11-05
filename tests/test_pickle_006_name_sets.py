"""MFV-PICKLE-006 and sets of names.

A protocol-2 pickle stores a set of strings as builtins.set([...]). PEFT pickles
a LoraConfig inside torch.save checkpoints, and its target_modules is such a set
('q_proj', 'k_proj', ...), so every LoRA checkpoint was reported MEDIUM. Plain
names in a set or frozenset are data; free text handed to one still is not.
"""

from __future__ import annotations

import pickle

import pytest

from hayward import ModelFileScanner


def _rules(tmp_path, data):
    path = tmp_path / "model.pkl"
    path.write_bytes(data)
    return {f.rule_id for f in ModelFileScanner().scan_file(path)}


@pytest.mark.parametrize("value", [
    {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"},
    frozenset({"lm_head", "model.layers.0.self_attn.q_proj"}),
])
def test_a_set_of_module_names_is_data(tmp_path, value):
    data = pickle.dumps({"target_modules": value, "r": 8}, protocol=2)
    assert "MFV-PICKLE-006" not in _rules(tmp_path, data)


@pytest.mark.parametrize("text", ["ls -la", "import os; os.system('id')", "a\nb", "/bin/sh"])
def test_free_text_in_a_set_is_still_reported(tmp_path, text):
    data = pickle.dumps({"target_modules": {"q_proj", text}}, protocol=2)
    assert "MFV-PICKLE-006" in _rules(tmp_path, data)


def test_free_text_in_a_later_set_call_is_still_reported(tmp_path):
    # The scanner kept one call per callable, so a clean first call hid the rest.
    data = pickle.dumps([{"q_proj"}, {"ls -la"}], protocol=2)
    assert "MFV-PICKLE-006" in _rules(tmp_path, data)


def test_free_text_in_a_later_call_to_any_allowlisted_callable_is_still_reported(tmp_path):
    # [OrderedDict(), OrderedDict('ls -la')]: the clean call comes first.
    data = (b"\x80\x02]q\x00(ccollections\nOrderedDict\n)Rccollections\nOrderedDict\n"
            b"X\x06\x00\x00\x00ls -la\x85Re.")
    assert "MFV-PICKLE-006" in _rules(tmp_path, data)


def test_a_long_name_is_not_a_plain_name(tmp_path):
    data = pickle.dumps({"target_modules": {"x" + "696d706f7274206f73" * 20}}, protocol=2)
    assert "MFV-PICKLE-006" in _rules(tmp_path, data)
