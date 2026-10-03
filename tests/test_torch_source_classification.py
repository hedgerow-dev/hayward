"""Independent format and operation fixtures; no model loader is invoked."""

import pickle
import zipfile

import pytest

from hayward import ModelFileScanner, Severity, torch_source


def archive(tmp_path, members, name="arbitrary.pt"):
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as zf:
        for member, data in members.items():
            zf.writestr(member, data)
    return path


def package(tmp_path, source, prefix="delivery/"):
    return archive(tmp_path, {
        prefix + ".data/extern_modules": b"torch\n",
        prefix + ".data/python_version": b"3.12\n",
        prefix + "payload.pkl": pickle.dumps({"value": 3}),
        prefix + "widget.py": source,
    })


@pytest.mark.parametrize("prefix", ["", "delivery/"])
def test_package_source_presence_is_not_a_dangerous_operation(tmp_path, prefix):
    findings = ModelFileScanner().scan_file(package(tmp_path, b"VALUE = 3\n", prefix))
    assert any(f.rule_id == "MFV-TORCH-001" and f.severity == Severity.LOW
               and f.metadata["archive_kinds"] == ["torch_package"] for f in findings)
    assert not any(f.severity in {Severity.HIGH, Severity.CRITICAL} for f in findings)


@pytest.mark.parametrize("source, operation", [
    ("import os as dispatch\ndispatch.system('echo example')", "os.system"),
    ("from os import system as launch\nlaunch('echo example')", "os.system"),
    ("import subprocess as process\nprocess.run(['example'])", "subprocess.run"),
    ("import builtins as runtime\nruntime.exec('VALUE=3')", "builtins.exec"),
    ("def invoke(value):\n    return eval(value)", "builtins.eval"),
])
def test_concrete_execution_operation_is_retained(tmp_path, source, operation):
    findings = ModelFileScanner().scan_file(package(tmp_path, source))
    hits = [f for f in findings if f.rule_id == "MFV-TORCH-002"]
    assert len(hits) == 1
    assert hits[0].severity == Severity.HIGH
    assert operation in {op["call"] for op in hits[0].metadata["operations"]}
    assert "reachability and intent are not established" in hits[0].message


@pytest.mark.parametrize("source", [
    "import os\nos = custom\nos.system(value)",
    "import os\ndef invoke(os):\n    return os.system(value)",
    "import os\nos.system = custom\nos.system(value)",
    "from .widget_helpers import eval\neval(value)",
    "def eval(value):\n    return value\neval('3')",
    "import os\nfrom custom import *\nos.system(value)",
    "import os, custom as os\nos.system(value)",
    "widget.eval()",
])
def test_shadowed_or_unresolved_calls_are_not_convicted(tmp_path, source):
    findings = ModelFileScanner().scan_file(package(tmp_path, source))
    assert not any(f.rule_id == "MFV-TORCH-002" for f in findings)


def test_scanner_does_not_execute_packaged_source(tmp_path):
    marker = tmp_path / "must-not-exist"
    source = f"exec({('open(' + repr(str(marker)) + ', chr(119)).write(chr(120))')!r})"
    findings = ModelFileScanner().scan_file(package(tmp_path, source))
    assert any(f.rule_id == "MFV-TORCH-002" for f in findings)
    assert not marker.exists()


@pytest.mark.parametrize("members", [
    {"checkpoint/data.pkl": pickle.dumps({}), "extras/help.py": b"VALUE=3"},
    {"first/data.pkl": pickle.dumps({}), "other/code/node.py": b"VALUE=3"},
    {"first/.data/extern_modules": b"torch", "other/module.py": b"VALUE=3"},
    {"code/module.py": b"VALUE=3"},
])
def test_unrelated_sources_are_not_torch_source_members(tmp_path, members):
    findings = ModelFileScanner().scan_file(archive(tmp_path, members))
    assert not any(f.rule_id.startswith("MFV-TORCH-") for f in findings)


def test_torchscript_presence_does_not_suppress_pickle_operation(tmp_path):
    # An inert pickle byte stream resolving os.system; never deserialized.
    path = archive(tmp_path, {"graph/data.pkl": b"cos\nsystem\n.",
                            "graph/code/__torch__/unit.py": b"class Unit: pass\n"})
    findings = ModelFileScanner().scan_file(path)
    assert any(f.rule_id == "MFV-TORCH-001" and f.severity == Severity.LOW for f in findings)
    assert any(f.rule_id == "MFV-PICKLE-001" and f.severity == Severity.CRITICAL for f in findings)


@pytest.mark.parametrize("source", [b"not valid python !!!", b"#" + b"a" * 100])
def test_source_failure_is_coverage_not_a_clean_verdict(tmp_path, monkeypatch, source):
    monkeypatch.setattr(torch_source, "MAX_SOURCE_BYTES", 64)
    findings = ModelFileScanner().scan_file(package(tmp_path, source))
    assert any(f.rule_id == "MFV-SKIP-002" and f.metadata.get("source_member") for f in findings)


def test_package_source_member_budget_reports_incomplete_analysis(tmp_path, monkeypatch):
    monkeypatch.setattr(torch_source, "MAX_SOURCE_MEMBERS", 1)
    path = archive(tmp_path, {".data/extern_modules": b"torch",
                            "a.py": b"VALUE=3", "b.py": b"VALUE=4"})
    findings = ModelFileScanner().scan_file(path)
    assert any(f.rule_id == "MFV-SKIP-002" and f.metadata["skipped_reason"] == "source_budget"
               for f in findings)


def test_duplicate_source_members_report_incomplete_coverage(tmp_path):
    path = package(tmp_path, b"import os\nos.system('example')")
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(path, "a") as zf:
            zf.writestr("delivery/widget.py", b"VALUE = 3")
    findings = ModelFileScanner().scan_file(path)
    assert any(f.rule_id == "MFV-SKIP-002"
               and f.metadata.get("skipped_reason") == "duplicate_source_member"
               for f in findings)
