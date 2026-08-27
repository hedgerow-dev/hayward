"""CLI and report-surface tests: exit codes, output formats, -o behaviour,
--fail-on / --fail-on-coverage, and colour controls.

`cli.main` takes an argv list and returns the exit code, so everything here
runs in-process; argparse usage errors surface as SystemExit(2).
"""

from __future__ import annotations

import argparse
import json

import pytest

from hayward import cli
from hayward.findings import Category, Finding, Severity
from hayward.scanner import ModelFileScanner


def _short_binunicode(text: str) -> bytes:
    raw = text.encode()
    return bytes([0x8C, len(raw)]) + raw


def _os_system_pickle(command: str) -> bytes:
    """A standalone protocol-4 pickle calling os.system(command)."""
    return (
        b"\x80\x04"
        + _short_binunicode("os")
        + _short_binunicode("system")
        + b"\x93"
        + _short_binunicode(command)
        + b"\x85"
        + b"R."
    )


@pytest.fixture
def clean_dir(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    return models


@pytest.fixture
def evil_file(tmp_path):
    target = tmp_path / "evil.pkl"
    target.write_bytes(_os_system_pickle("id"))
    return target


class TestExitCodes:
    def test_clean_scan_exits_zero(self, clean_dir):
        assert cli.main(["scan", str(clean_dir)]) == 0

    def test_findings_at_threshold_exit_one(self, evil_file):
        assert cli.main(["scan", str(evil_file), "--fail-on", "high"]) == 1
        assert cli.main(["scan", str(evil_file), "--fail-on", "critical"]) == 1

    def test_fail_on_never_exits_zero_despite_findings(self, evil_file):
        assert cli.main(["scan", str(evil_file), "--fail-on", "never"]) == 0

    def test_crash_exits_two(self, tmp_path, monkeypatch, capsys):
        def boom(self, path):
            raise RuntimeError("boom")

        monkeypatch.setattr(ModelFileScanner, "scan_file", boom)
        target = tmp_path / "model.pkl"
        target.write_bytes(b"\x80\x04.")
        assert cli.main(["scan", str(target)]) == 2
        assert "hayward" in capsys.readouterr().err

    def test_directory_scan_crash_exits_two(self, clean_dir, monkeypatch, capsys):
        def boom(self, root):
            raise ValueError("boom")

        monkeypatch.setattr(ModelFileScanner, "scan_directory", boom)
        assert cli.main(["scan", str(clean_dir)]) == 2
        assert "hayward" in capsys.readouterr().err

    def test_missing_target_exits_two(self, tmp_path, capsys):
        assert cli.main(["scan", str(tmp_path / "nope")]) == 2
        assert "no such file" in capsys.readouterr().err

    def test_usage_error_exits_two(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            cli.main(["scan"])
        assert excinfo.value.code == 2

    def test_bad_format_choice_exits_two(self, evil_file):
        with pytest.raises(SystemExit) as excinfo:
            cli.main(["scan", str(evil_file), "-f", "yaml"])
        assert excinfo.value.code == 2


class TestJsonOutput:
    def test_json_parses_with_documented_keys(self, evil_file, capsys):
        rc = cli.main(["scan", str(evil_file), "-f", "json", "--fail-on", "never"])
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["tool"] == "hayward"
        for key in ("version", "generated", "root", "counts", "findings", "coverage_gaps"):
            assert key in data
        assert data["counts"].get("critical", 0) >= 1
        finding = data["findings"][0]
        for key in ("rule_id", "message", "severity", "category",
                    "file", "confidence", "cwe_ids", "metadata"):
            assert key in finding

    def test_json_to_file(self, evil_file, tmp_path):
        out = tmp_path / "report.json"
        rc = cli.main(
            ["scan", str(evil_file), "-f", "json", "-o", str(out), "--fail-on", "never"])
        assert rc == 0
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["findings"]


class TestTextOutput:
    def test_text_to_file_is_plain_ansi_free_text_not_json(self, evil_file, tmp_path, capsys):
        out = tmp_path / "report.txt"
        rc = cli.main(
            ["scan", str(evil_file), "-f", "text", "-o", str(out), "--fail-on", "never"])
        assert rc == 0
        body = out.read_text(encoding="utf-8")
        assert "\x1b[" not in body
        assert "finding(s)" in body
        assert "MFV-PICKLE-001" in body
        with pytest.raises(json.JSONDecodeError):
            json.loads(body)
        assert f"Report written to {out}" in capsys.readouterr().out

    def test_text_to_file_clean_scan(self, clean_dir, tmp_path):
        out = tmp_path / "report.txt"
        rc = cli.main(["scan", str(clean_dir), "-f", "text", "-o", str(out)])
        assert rc == 0
        assert out.read_text(encoding="utf-8") == f"No findings in {clean_dir}"

    def test_unwritable_output_exits_two(self, evil_file, tmp_path, capsys):
        out = tmp_path / "does-not-exist" / "report.txt"
        rc = cli.main(["scan", str(evil_file), "-f", "text", "-o", str(out)])
        assert rc == 2
        assert "could not write" in capsys.readouterr().err


class TestFailOnCoverage:
    def _patch_gap(self, monkeypatch, target):
        gap = Finding(
            rule_id="MFV-SKIP-003",
            message="could not be fully read",
            severity=Severity.INFO,
            category=Category.AI_ML,
            file_path=str(target),
        )
        monkeypatch.setattr(ModelFileScanner, "scan_file", lambda self, path: [gap])

    def test_coverage_gap_alone_exits_zero(self, tmp_path, monkeypatch):
        target = tmp_path / "opaque.bin"
        target.write_bytes(b"x")
        self._patch_gap(monkeypatch, target)
        assert cli.main(["scan", str(target), "--fail-on", "never"]) == 0

    def test_fail_on_coverage_exits_one(self, tmp_path, monkeypatch):
        target = tmp_path / "opaque.bin"
        target.write_bytes(b"x")
        self._patch_gap(monkeypatch, target)
        assert cli.main(
            ["scan", str(target), "--fail-on", "never", "--fail-on-coverage"]) == 1


class TestColour:
    def test_color_always_emits_ansi(self, evil_file, capsys):
        cli.main(["scan", str(evil_file), "--color", "always", "--fail-on", "never"])
        assert "\x1b[" in capsys.readouterr().out

    def test_color_never_is_plain(self, evil_file, capsys):
        cli.main(["scan", str(evil_file), "--color", "never", "--fail-on", "never"])
        assert "\x1b[" not in capsys.readouterr().out

    def test_no_colour_flag_still_works(self, evil_file, capsys):
        cli.main(["scan", str(evil_file), "--no-colour", "--fail-on", "never"])
        assert "\x1b[" not in capsys.readouterr().out

    def test_auto_is_plain_when_not_a_tty(self, evil_file, capsys):
        cli.main(["scan", str(evil_file), "--fail-on", "never"])
        assert "\x1b[" not in capsys.readouterr().out

    def test_colour_enabled_rules(self, monkeypatch):
        class Tty:
            def isatty(self):
                return True

        class NotTty:
            def isatty(self):
                return False

        def ns(color, no_colour=False):
            return argparse.Namespace(color=color, no_colour=no_colour)

        assert cli._colour_enabled(ns("always"), NotTty()) is True
        assert cli._colour_enabled(ns("never"), Tty()) is False
        assert cli._colour_enabled(ns("auto"), Tty()) is True
        assert cli._colour_enabled(ns("auto"), NotTty()) is False
        assert cli._colour_enabled(ns("auto", no_colour=True), Tty()) is False
        monkeypatch.setenv("NO_COLOR", "1")
        assert cli._colour_enabled(ns("auto"), Tty()) is False
        assert cli._colour_enabled(ns("always"), Tty()) is True
