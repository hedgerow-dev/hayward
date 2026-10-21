"""The pre-commit hook in `.pre-commit-hooks.yaml` must not drift from the
scanner.

Its `files:` regex decides which staged files pre-commit hands to `hayward
scan`, so an extension the scanner learns but the hook does not name is a
model file a commit can carry through unscanned. The expected regex is built
here from the same extension sets the directory walk uses, and the YAML must
hold exactly that string.

PyYAML is not a dependency, so the hook file is kept to flat `key: value`
lines and read with a few lines of string handling.
"""

from __future__ import annotations

import re
from pathlib import Path

from hayward import cli
from hayward.scanner import ModelFileScanner

HOOKS_FILE = Path(__file__).resolve().parent.parent / ".pre-commit-hooks.yaml"


def _hook_fields() -> dict[str, str]:
    fields = {}
    for line in HOOKS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.removeprefix("- ").partition(":")
        value = value.strip()
        if value.startswith("'") and value.endswith("'"):
            value = value[1:-1]
        fields[key.strip()] = value
    return fields


def _scanned_extensions() -> set[str]:
    # The same union scan_directory's discovery walks.
    scanner = ModelFileScanner()
    return {
        *scanner._format_map,
        *scanner._AMBIGUOUS_EXTENSIONS,
        *scanner._CONFIG_EXTENSIONS,
    }


def _expected_regex() -> str:
    names = sorted(re.escape(ext.removeprefix(".")) for ext in _scanned_extensions())
    return r"(?i)\.(" + "|".join(names) + r")$"


def test_hook_definition():
    fields = _hook_fields()
    assert fields["id"] == "hayward"
    assert fields["entry"] == "hayward scan --fail-on high"
    assert fields["language"] == "python"
    assert fields["pass_filenames"] == "true"
    assert fields["types"] == "[file]"


def test_files_regex_is_generated_from_scanner_extensions():
    expected = _expected_regex()
    assert _hook_fields()["files"] == expected, (
        "the scanner's extension list changed; set the hook's files: to "
        f"'{expected}'"
    )


def test_files_regex_matches_every_scanned_extension():
    pattern = re.compile(_hook_fields()["files"])
    for ext in _scanned_extensions():
        assert pattern.search(f"models/weights{ext}"), ext
        assert pattern.search(f"models/WEIGHTS{ext.upper()}"), ext
    for path in ("README.md", "src/app.py", "weights.pt.txt", "pyproject.toml"):
        assert not pattern.search(path), path


def _os_system_pickle() -> bytes:
    # Built from raw opcodes, never by pickling a live object.
    def s(text: str) -> bytes:
        return bytes([0x8C, len(text)]) + text.encode()
    return b"\x80\x04" + s("os") + s("system") + b"\x93" + s("id") + b"\x85R."


def test_cli_as_pre_commit_calls_it(tmp_path):
    # pre-commit passes every matching staged file as a positional argument,
    # and the regex matches files that are not models (`package.json`, a
    # random `.bin`). Those must scan clean, not error.
    (tmp_path / "package.json").write_text('{"name": "app"}')
    (tmp_path / "blob.bin").write_bytes(b"\x00not a model\xff" * 64)
    (tmp_path / "notes.zip").write_bytes(b"plain text, not a zip")
    benign = [str(tmp_path / n) for n in ("package.json", "blob.bin", "notes.zip")]
    assert cli.main(["scan", "--fail-on", "high", *benign]) == 0

    evil = tmp_path / "evil.pkl"
    evil.write_bytes(_os_system_pickle())
    assert cli.main(["scan", "--fail-on", "high", *benign, str(evil)]) == 1
