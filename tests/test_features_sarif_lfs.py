"""HW-140 (SARIF output + JSON schema_version) and HW-141 (Git LFS pointer
detection, MFV-LFS-001).

`cli.main` takes an argv list and returns the exit code, so CLI behaviour is
exercised in-process.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hayward import __version__, cli
from hayward.findings import (
    COVERAGE_RULE_IDS,
    Category,
    Finding,
    Severity,
    is_coverage_gap,
)
from hayward.report import to_sarif
from hayward.scanner import ModelFileScanner

_LFS_OID = "ab" * 32


def _lfs_pointer(size: int = 131_450_032, oid: str = _LFS_OID, extra: str = "") -> bytes:
    return (
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{oid}\n"
        f"size {size}\n"
        f"{extra}"
    ).encode()


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


def _finding(rule_id: str, severity: Severity, path: str = "/srv/models/model.pt",
             message: str = "Something was found. More detail follows.") -> Finding:
    return Finding(
        rule_id=rule_id,
        message=message,
        severity=severity,
        category=Category.AI_ML,
        file_path=path,
    )


class TestLfsPointerDetection:
    @pytest.mark.parametrize("name", ["pytorch_model.bin", "model.pt"])
    def test_pointer_reports_lfs_finding(self, tmp_path, name):
        target = tmp_path / name
        target.write_bytes(_lfs_pointer())
        findings = ModelFileScanner().scan_file(target)
        assert len(findings) == 1
        finding = findings[0]
        assert finding.rule_id == "MFV-LFS-001"
        assert finding.severity is Severity.INFO
        assert is_coverage_gap(finding)
        assert "131450032" in finding.message
        assert _LFS_OID in finding.message
        assert finding.metadata["lfs_declared_size"] == 131_450_032
        assert finding.metadata["lfs_oid"] == _LFS_OID
        assert finding.metadata["lfs_malformed"] is False

    def test_pointer_is_registered_coverage_gap(self):
        assert "MFV-LFS-001" in COVERAGE_RULE_IDS
        finding = _finding("MFV-LFS-001", Severity.INFO)
        assert is_coverage_gap(finding)

    def test_real_small_file_does_not_trigger(self, tmp_path):
        target = tmp_path / "model.pt"
        target.write_bytes(b"\x80\x04.")
        findings = ModelFileScanner().scan_file(target)
        assert not any(f.rule_id == "MFV-LFS-001" for f in findings)

    def test_small_text_file_does_not_trigger(self, tmp_path):
        target = tmp_path / "notes.bin"
        target.write_bytes(b"version 2\nof something else entirely\n")
        assert ModelFileScanner().scan_file(target) == []

    def test_pointer_tolerates_trailing_whitespace_and_extra_keys(self, tmp_path):
        target = tmp_path / "weights.safetensors"
        body = (
            "version https://git-lfs.github.com/spec/v1 \n"
            f"oid sha256:{_LFS_OID}\n"
            "ext-0-xxhash: 0123456789abcdef\n"
            "size 42\n"
        )
        target.write_bytes(body.encode())
        findings = ModelFileScanner().scan_file(target)
        assert [f.rule_id for f in findings] == ["MFV-LFS-001"]
        assert findings[0].metadata["lfs_declared_size"] == 42

    def test_malformed_pointer_still_reported_with_note(self, tmp_path):
        # Documented behaviour: a bad oid does not fall through to normal
        # scanning. The version line still marks the file as a placeholder
        # for content that was never fetched, so it is reported as a pointer
        # with a note about the malformation.
        target = tmp_path / "model.bin"
        target.write_bytes(_lfs_pointer(oid="not-a-valid-oid"))
        findings = ModelFileScanner().scan_file(target)
        assert [f.rule_id for f in findings] == ["MFV-LFS-001"]
        finding = findings[0]
        assert "malformed" in finding.message
        assert finding.metadata["lfs_malformed"] is True
        assert "lfs_oid" not in finding.metadata
        assert finding.metadata["lfs_declared_size"] == 131_450_032
        assert is_coverage_gap(finding)

    def test_pointer_missing_size_reported_as_malformed(self, tmp_path):
        target = tmp_path / "model.gguf"
        target.write_bytes(
            b"version https://git-lfs.github.com/spec/v1\n"
            b"oid sha256:" + _LFS_OID.encode() + b"\n"
        )
        findings = ModelFileScanner().scan_file(target)
        assert [f.rule_id for f in findings] == ["MFV-LFS-001"]
        assert findings[0].metadata["lfs_malformed"] is True
        assert "lfs_declared_size" not in findings[0].metadata

    def test_pointer_fail_on_coverage_exits_one(self, tmp_path):
        target = tmp_path / "pytorch_model.bin"
        target.write_bytes(_lfs_pointer())
        assert cli.main(["scan", str(target), "--fail-on", "never"]) == 0
        assert cli.main(
            ["scan", str(target), "--fail-on", "never", "--fail-on-coverage"]) == 1

    def test_pointer_lands_in_json_coverage_gaps(self, tmp_path, capsys):
        target = tmp_path / "pytorch_model.bin"
        target.write_bytes(_lfs_pointer())
        rc = cli.main(["scan", str(target), "-f", "json", "--fail-on", "never"])
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["coverage_gaps"] == [str(target)]
        assert data["findings"][0]["rule_id"] == "MFV-LFS-001"


class TestSarifOutput:
    def test_required_210_keys(self):
        findings = [_finding("MFV-PICKLE-001", Severity.CRITICAL)]
        data = json.loads(to_sarif(findings, None, __version__))
        assert data["$schema"] == "https://json.schemastore.org/sarif-2.1.0.json"
        assert data["version"] == "2.1.0"
        run = data["runs"][0]
        driver = run["tool"]["driver"]
        assert driver["name"] == "hayward"
        assert driver["version"] == __version__
        assert driver["informationUri"] == "https://github.com/hedgerow-dev/hayward"
        assert isinstance(driver["rules"], list)
        assert isinstance(run["results"], list)

    @pytest.mark.parametrize("severity,level", [
        (Severity.CRITICAL, "error"),
        (Severity.HIGH, "error"),
        (Severity.MEDIUM, "warning"),
        (Severity.LOW, "note"),
        (Severity.INFO, "note"),
    ])
    def test_level_mapping(self, severity, level):
        data = json.loads(to_sarif([_finding("MFV-X-001", severity)], None, "0"))
        assert data["runs"][0]["results"][0]["level"] == level

    def test_rules_deduplicated(self):
        findings = [
            _finding("MFV-PICKLE-001", Severity.CRITICAL, path="/m/a.pt"),
            _finding("MFV-PICKLE-001", Severity.CRITICAL, path="/m/b.pt"),
            _finding("MFV-SKIP-003", Severity.LOW, path="/m/c.pt"),
        ]
        data = json.loads(to_sarif(findings, None, "0"))
        driver = data["runs"][0]["tool"]["driver"]
        assert [r["id"] for r in driver["rules"]] == ["MFV-PICKLE-001", "MFV-SKIP-003"]
        assert all(r["shortDescription"]["text"] for r in driver["rules"])
        results = data["runs"][0]["results"]
        assert [r["ruleIndex"] for r in results] == [0, 0, 1]

    def test_uris_relative_to_root(self):
        findings = [_finding("MFV-PICKLE-001", Severity.HIGH,
                             path="/srv/models/sub/model.pt")]
        data = json.loads(to_sarif(findings, None, "0"))
        uri = data["runs"][0]["results"][0]["locations"][0][
            "physicalLocation"]["artifactLocation"]["uri"]
        assert uri == "/srv/models/sub/model.pt"

        data = json.loads(to_sarif(findings, Path("/srv/models"), "0"))
        uri = data["runs"][0]["results"][0]["locations"][0][
            "physicalLocation"]["artifactLocation"]["uri"]
        assert uri == "sub/model.pt"

    def test_empty_findings(self):
        data = json.loads(to_sarif([], None, "0"))
        assert data["runs"][0]["tool"]["driver"]["rules"] == []
        assert data["runs"][0]["results"] == []

    def test_cli_sarif_end_to_end(self, tmp_path, capsys):
        target = tmp_path / "evil.pkl"
        target.write_bytes(_os_system_pickle("id"))
        rc = cli.main(["scan", str(target), "-f", "sarif", "--fail-on", "never"])
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["version"] == "2.1.0"
        results = data["runs"][0]["results"]
        assert results and results[0]["ruleId"] == "MFV-PICKLE-001"
        assert results[0]["level"] == "error"
        uri = results[0]["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]
        assert uri == "evil.pkl"

    def test_cli_sarif_to_file(self, tmp_path):
        target = tmp_path / "evil.pkl"
        target.write_bytes(_os_system_pickle("id"))
        out = tmp_path / "report.sarif"
        rc = cli.main(
            ["scan", str(target), "-f", "sarif", "-o", str(out), "--fail-on", "never"])
        assert rc == 0
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["runs"][0]["results"]


class TestJsonSchemaVersion:
    def test_json_output_contains_schema_version(self, tmp_path, capsys):
        target = tmp_path / "evil.pkl"
        target.write_bytes(_os_system_pickle("id"))
        rc = cli.main(["scan", str(target), "-f", "json", "--fail-on", "never"])
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["schema_version"] == 1
