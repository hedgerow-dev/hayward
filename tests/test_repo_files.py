"""The rest of a model repo: chat templates and tokenizer data files.

transformers renders `chat_template.jinja` (what recent versions write) the same way
as the `chat_template` field of tokenizer_config.json, so it gets the same check
(MFV-HF-001). Tokenizer data files (merges.txt, vocab.txt, tokenizer.model) are read
by tokenizer libraries that run no code, so the check is that they are what their
name says and not a pickle or archive wearing that name. A directory scan finds all
of them, so a receipt covers them.
"""

from __future__ import annotations

import pytest

from hayward import ModelFileScanner

EVAL = b"\x80\x04\x8c\x08builtins\x8c\x04eval\x93\x8c\x0440+2\x85R."
SSTI = "{{ cycler.__init__.__globals__.os.popen('id').read() }}"
SAFE = "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"


def scan(tmp_path, name: str, data: bytes):
    p = tmp_path / name
    p.write_bytes(data)
    return [f.rule_id for f in ModelFileScanner().scan_file(p)]


def test_a_malicious_jinja_chat_template_is_caught(tmp_path):
    assert "MFV-HF-001" in scan(tmp_path, "chat_template.jinja", SSTI.encode())


def test_an_ordinary_jinja_chat_template_is_clean(tmp_path):
    assert scan(tmp_path, "chat_template.jinja", SAFE.encode()) == []


@pytest.mark.parametrize("name", ["merges.txt", "vocab.txt", "tokenizer.model", "spiece.model"])
def test_ordinary_tokenizer_data_is_clean(tmp_path, name):
    assert scan(tmp_path, name, b"#version: 0.2\nh e\nt h\n") == []


@pytest.mark.parametrize("name", ["merges.txt", "tokenizer.model"])
def test_a_pickle_named_as_tokenizer_data_is_caught(tmp_path, name):
    assert "MFV-PICKLE-001" in scan(tmp_path, name, EVAL)


def test_a_directory_scan_finds_templates_and_tokenizer_data(tmp_path):
    for name in ("chat_template.jinja", "merges.txt", "vocab.txt", "tokenizer.model", "notes.txt", "x.model"):
        (tmp_path / name).write_text("x")
    found = sorted(p.name for p in ModelFileScanner().discover_files(tmp_path))
    assert found == ["chat_template.jinja", "merges.txt", "tokenizer.model", "vocab.txt"]
