"""HW-126: tests for the desktop window's non-display logic and the polish
fixes (headless-Tk error, file-dialog filter, multi-file scan, full-path
display).

Tk needs a display. The two tests that build a real window skip cleanly where
there is none (headless CI), while the headless-error path and the pure helper
run everywhere.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path

import pytest

from hayward import gui


def _short_binunicode(text: str) -> bytes:
    raw = text.encode()
    return bytes([0x8C, len(raw)]) + raw


def _os_system_pickle(command: str) -> bytes:
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
def app():
    """A HaywardApp on a real Tk root, or a skip where no display exists."""
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display for Tk")
    root.withdraw()
    application = gui.HaywardApp(root)
    yield application
    root.destroy()


class TestHeadless:
    def test_main_reports_cleanly_when_tk_cannot_open(self, monkeypatch, capsys):
        def no_display(*_a, **_k):
            raise tk.TclError("no display name and no $DISPLAY environment variable")

        monkeypatch.setattr(gui.tk, "Tk", no_display)
        assert gui.main() == 1
        err = capsys.readouterr().err
        assert "hayward-gui" in err
        assert "command line" in err  # points the user at the CLI


class TestCommonRoot:
    def test_single_path_is_its_own_root(self, tmp_path):
        assert gui._common_root([tmp_path / "a.pkl"]) == tmp_path / "a.pkl"

    def test_multiple_paths_use_common_ancestor(self, tmp_path):
        a = tmp_path / "sub" / "a.pkl"
        b = tmp_path / "sub" / "b.pkl"
        assert gui._common_root([a, b]) == tmp_path / "sub"


class TestFileDialogFilter:
    def test_open_dialog_offers_the_pickle_bearing_extensions(self, app, monkeypatch):
        captured = {}

        def fake_open(**kwargs):
            captured.update(kwargs)
            return ""  # user cancels; we only care about the filter

        monkeypatch.setattr(gui.filedialog, "askopenfilename", fake_open)
        app._pick_file()
        pattern = " ".join(ext for _label, ext in captured["filetypes"])
        for wanted in ("*.pickle", "*.ckpt", "*.th", "*.hdf5"):
            assert wanted in pattern


class TestMultiFileScan:
    def test_worker_scans_every_dropped_file(self, app, tmp_path):
        a = tmp_path / "a.pkl"
        b = tmp_path / "b.pkl"
        a.write_bytes(_os_system_pickle("id"))
        b.write_bytes(_os_system_pickle("whoami"))

        app._worker([a, b])
        kind, findings = app.queue.get_nowait()
        assert kind == "done"
        flagged = {f.file_path for f in findings if f.rule_id == "MFV-PICKLE-001"}
        assert str(a) in flagged and str(b) in flagged


class TestDisplayPath:
    def test_display_path_shows_full_path_outside_root(self, app, tmp_path):
        app.target = tmp_path / "models"
        finding = type("F", (), {"file_path": "/elsewhere/model.pkl"})()
        # Outside the scan root, the full path is shown rather than a bare name.
        assert app._display_path(finding) == str(Path("/elsewhere/model.pkl"))
