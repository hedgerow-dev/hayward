"""HW-122: the code snippets in docs/usage.md must run as written.

The library example once sorted findings with a key that compared an int
against a str, raising TypeError the moment anyone followed it. This runs the
documented snippet against a real scan so a future regression fails CI instead
of a reader's first attempt.
"""

from __future__ import annotations

from hayward import ModelFileScanner, Severity, is_coverage_gap


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


def test_usage_library_example_runs(tmp_path):
    # Mirrors the "## Python" example in docs/usage.md verbatim in shape.
    model = tmp_path / "model.pt"
    model.write_bytes(_os_system_pickle("id"))

    scanner = ModelFileScanner()
    findings = scanner.scan_file(model)

    blocking = [f for f in findings if f.severity in (Severity.CRITICAL, Severity.HIGH)]
    unread = [f for f in findings if is_coverage_gap(f)]
    assert blocking  # the os.system payload is CRITICAL
    assert unread == []

    # The sort key is an int (severity_order); this line is the one that used
    # to raise TypeError when the docs sorted on a mixed type.
    for f in sorted(findings, key=lambda f: f.severity_order):
        assert isinstance(f.rule_id, str)
        assert isinstance(f.severity.value, str)
        d = f.to_dict()
        assert d["rule_id"] == f.rule_id


def test_usage_directory_example_runs(tmp_path):
    model = tmp_path / "model.pkl"
    model.write_bytes(_os_system_pickle("id"))
    findings = ModelFileScanner().scan_directory(tmp_path)
    assert any(f.rule_id == "MFV-PICKLE-001" for f in findings)
