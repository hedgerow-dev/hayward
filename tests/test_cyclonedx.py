"""HW-151: CycloneDX 1.6 ML-BOM output (report.to_cyclonedx).

The renderer is exercised directly against constructed findings so the BOM
structure is asserted without a full scan. render("cyclonedx", ...) is checked
too, since that is the entry point cli.py and gui.py call.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from hayward.findings import Category, Finding, Severity
from hayward.report import render, to_cyclonedx


def _finding(
    rule_id: str,
    severity: Severity,
    path: str = "/srv/models/model.pt",
    message: str = "Something was found. More detail follows.",
    cwe_ids: list[int] | None = None,
) -> Finding:
    return Finding(
        rule_id=rule_id,
        message=message,
        severity=severity,
        category=Category.AI_ML,
        file_path=path,
        cwe_ids=cwe_ids or [],
    )


class TestCycloneDxStructure:
    def test_bom_format_and_spec_version(self):
        data = json.loads(to_cyclonedx([], None, "1.2.3"))
        assert data["bomFormat"] == "CycloneDX"
        assert data["specVersion"] == "1.6"
        # A CycloneDX BOM version is an integer, defaulting to 1.
        assert data["version"] == 1

    def test_metadata_tool_component(self):
        data = json.loads(to_cyclonedx([], None, "9.9.9"))
        tools = data["metadata"]["tools"]["components"]
        assert tools == [{"type": "application", "name": "hayward", "version": "9.9.9"}]

    def test_metadata_timestamp_is_iso8601_utc(self):
        data = json.loads(to_cyclonedx([], None, "0"))
        stamp = data["metadata"]["timestamp"]
        # ISO-8601 with the Z zone designator, parseable by a strict consumer.
        assert stamp.endswith("Z")
        # fromisoformat accepts the 'Z' suffix only from 3.11; normalise it so
        # this holds on the project's baseline too.
        parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        assert parsed.tzinfo is not None


class TestCycloneDxComponents:
    def test_one_component_per_distinct_file(self):
        findings = [
            _finding("MFV-PICKLE-001", Severity.CRITICAL, path="/m/a.pt"),
            _finding("MFV-SKIP-003", Severity.LOW, path="/m/a.pt"),
            _finding("MFV-PICKLE-001", Severity.CRITICAL, path="/m/b.pt"),
        ]
        data = json.loads(to_cyclonedx(findings, None, "0"))
        components = data["components"]
        # Two findings share /m/a.pt, so two distinct files -> two components.
        assert len(components) == 2
        assert all(c["type"] == "machine-learning-model" for c in components)
        names = {c["name"] for c in components}
        assert names == {"/m/a.pt", "/m/b.pt"}
        # Every component carries a bom-ref, and they are unique.
        refs = [c["bom-ref"] for c in components]
        assert len(set(refs)) == len(refs)

    def test_component_names_are_relative_to_root(self):
        findings = [_finding("MFV-PICKLE-001", Severity.HIGH,
                             path="/srv/models/sub/model.pt")]
        data = json.loads(to_cyclonedx(findings, Path("/srv/models"), "0"))
        assert data["components"][0]["name"] == "sub/model.pt"
        assert data["components"][0]["bom-ref"] == "sub/model.pt"


class TestCycloneDxVulnerabilities:
    def test_one_vulnerability_per_finding(self):
        findings = [
            _finding("MFV-PICKLE-001", Severity.CRITICAL, path="/m/a.pt"),
            _finding("MFV-SKIP-003", Severity.LOW, path="/m/a.pt"),
            _finding("MFV-ONNX-001", Severity.HIGH, path="/m/b.pt"),
        ]
        data = json.loads(to_cyclonedx(findings, None, "0"))
        vulns = data["vulnerabilities"]
        assert len(vulns) == 3
        assert {v["id"] for v in vulns} == {
            "MFV-PICKLE-001", "MFV-SKIP-003", "MFV-ONNX-001"
        }

    def test_rating_severity_mapping(self):
        for severity in (
            Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM,
            Severity.LOW, Severity.INFO,
        ):
            data = json.loads(to_cyclonedx([_finding("MFV-X-001", severity)], None, "0"))
            ratings = data["vulnerabilities"][0]["ratings"]
            assert ratings == [{"severity": severity.value}]

    def test_cwes_carried_when_present(self):
        findings = [_finding("MFV-PICKLE-001", Severity.CRITICAL,
                             cwe_ids=[502, 94])]
        data = json.loads(to_cyclonedx(findings, None, "0"))
        assert data["vulnerabilities"][0]["cwes"] == [502, 94]

    def test_cwes_omitted_when_absent(self):
        findings = [_finding("MFV-SKIP-003", Severity.LOW, cwe_ids=[])]
        data = json.loads(to_cyclonedx(findings, None, "0"))
        assert "cwes" not in data["vulnerabilities"][0]

    def test_description_is_the_message(self):
        findings = [_finding("MFV-PICKLE-001", Severity.CRITICAL,
                             message="A dangerous reduce was found.")]
        data = json.loads(to_cyclonedx(findings, None, "0"))
        assert data["vulnerabilities"][0]["description"] == "A dangerous reduce was found."

    def test_affects_references_the_file_component(self):
        findings = [
            _finding("MFV-PICKLE-001", Severity.CRITICAL, path="/m/a.pt"),
            _finding("MFV-ONNX-001", Severity.HIGH, path="/m/b.pt"),
        ]
        data = json.loads(to_cyclonedx(findings, None, "0"))
        component_refs = {c["bom-ref"] for c in data["components"]}
        for vuln in data["vulnerabilities"]:
            affects = vuln["affects"]
            assert len(affects) == 1
            # Every affects[].ref must resolve to a real component bom-ref.
            assert affects[0]["ref"] in component_refs

    def test_coverage_gap_finding_becomes_a_vulnerability(self):
        """A coverage gap (e.g. MFV-LFS-001) is still a finding, so it gets a
        component and a vulnerability like any other."""
        findings = [
            _finding("MFV-PICKLE-001", Severity.CRITICAL, path="/m/a.pt"),
            _finding("MFV-LFS-001", Severity.INFO, path="/m/big.bin"),
        ]
        data = json.loads(to_cyclonedx(findings, None, "0"))
        ids = {v["id"] for v in data["vulnerabilities"]}
        assert "MFV-LFS-001" in ids
        assert len(data["components"]) == 2


class TestCycloneDxEmptyAndRender:
    def test_empty_findings_render_a_valid_empty_bom(self):
        data = json.loads(to_cyclonedx([], None, "0"))
        assert data["bomFormat"] == "CycloneDX"
        assert data["specVersion"] == "1.6"
        assert data["components"] == []
        assert data["vulnerabilities"] == []

    def test_render_dispatches_to_cyclonedx(self):
        findings = [_finding("MFV-PICKLE-001", Severity.CRITICAL)]
        via_render = render("cyclonedx", findings, None, "0")
        direct = to_cyclonedx(findings, None, "0")
        assert via_render == direct
        assert json.loads(via_render)["bomFormat"] == "CycloneDX"
