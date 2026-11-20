"""Directory discovery skips dependency trees by name, but only where the caller wants that.

`hayward scan` on a project keeps skipping venv/, build/, site-packages/ and the like.
`skip_dirs=frozenset()` (what `hayward hub` uses through --no-skip-dirs) walks everything.
"""

from pathlib import Path

from hayward.cli import main as cli_main
from hayward.scanner import SKIP_DIR_NAMES, ModelFileScanner

EVIL = b"\x80\x04\x8c\x08builtins\x8c\x04eval\x93\x8c\x0440+2\x85R."


def _tree(tmp_path: Path) -> Path:
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "model.pkl").write_bytes(EVIL)
    (tmp_path / "top.pkl").write_bytes(EVIL)
    return tmp_path


def test_default_discovery_still_skips_named_folders(tmp_path):
    names = {p.relative_to(tmp_path).as_posix() for p in ModelFileScanner().discover_files(_tree(tmp_path))}
    assert names == {"top.pkl"}
    assert "build" in SKIP_DIR_NAMES


def test_empty_skip_set_walks_every_folder(tmp_path):
    found = ModelFileScanner().discover_files(_tree(tmp_path), skip_dirs=frozenset())
    assert {p.relative_to(tmp_path).as_posix() for p in found} == {"top.pkl", "build/model.pkl"}


def test_cli_flag_scans_skipped_folders(tmp_path, capsys):
    root = _tree(tmp_path)
    cli_main(["scan", str(root), "--format", "json"])
    plain = capsys.readouterr().out
    cli_main(["scan", str(root), "--format", "json", "--no-skip-dirs"])
    walked = capsys.readouterr().out
    assert "build" not in plain.replace("\\\\", "/")
    assert "build" in walked.replace("\\\\", "/")
