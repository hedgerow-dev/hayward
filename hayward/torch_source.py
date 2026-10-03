"""Static, bounded classification of source members in torch containers.

TorchScript's Python-like graph representation is not a Python module import.
Neither source presence nor archive provenance is a malicious-operation proof.
"""

from __future__ import annotations

import ast
import zipfile
from collections import Counter
from pathlib import Path

from hayward.findings import Category, Finding, Severity

MAX_SOURCE_MEMBERS = 32
MAX_SOURCE_BYTES = 256 * 1024
MAX_TOTAL_SOURCE_BYTES = 2 * 1024 * 1024

EXECUTION_CALLS = frozenset({
    "builtins.eval", "builtins.exec", "os.system", "os.popen",
    "posix.system", "nt.system", "subprocess.Popen", "subprocess.run",
    "subprocess.call", "subprocess.check_call", "subprocess.check_output",
})


def classify_sources(names: list[str]) -> dict[str, list[str]]:
    """Match source to a container marker at the same archive root."""
    packages = {n[:-len(".data/extern_modules")] for n in names
                if n == ".data/extern_modules" or n.endswith("/.data/extern_modules")}
    scripts = {n[:-len("data.pkl")] for n in names
               if n == "data.pkl" or n.endswith("/data.pkl")}
    result: dict[str, list[str]] = {}
    for name in sorted(set(names)):
        if not name.endswith(".py"):
            continue
        if any(name.startswith(root) for root in packages):
            result.setdefault("torch_package", []).append(name)
        elif any(name.startswith(root + "code/") for root in scripts):
            result.setdefault("torchscript", []).append(name)
    return result


def source_presence(path: Path, sources: dict[str, list[str]]) -> Finding | None:
    if not sources:
        return None
    members = sorted({n for group in sources.values() for n in group})
    descriptions = []
    if "torchscript" in sources:
        descriptions.append("TorchScript graph source is present. It is parsed by the "
                            "JIT importer, not imported as unrestricted Python")
    if "torch_package" in sources:
        descriptions.append("Python source is present in a torch.package layout. "
                            "PackageImporter can execute these modules when importing them")
    return Finding(
        rule_id="MFV-TORCH-001", severity=Severity.LOW,
        category=Category.DESERIALIZATION, file_path=str(path), confidence=0.95,
        message="; ".join(descriptions) + ". Source presence alone does not establish "
                "a malicious operation. Treat untrusted models as programs; graph "
                "operator and runtime behavior are not fully assessed here. Members: "
                + ", ".join(members[:5]),
        metadata={"archive_kinds": sorted(sources), "source_members": members[:20],
                  "rule_class": "presence", "evidence_tier": "presence"},
    )


def _execution_calls(tree: ast.AST) -> list[dict]:
    imports: dict[str, str] = {}
    bindings: Counter = Counter()
    shadowed = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name.split(".")[0]
                bindings[local] += 1
                imports[local] = alias.name if alias.asname else local
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    # A wildcard can shadow builtins or a known module alias.
                    return []
                local = alias.asname or alias.name
                bindings[local] += 1
                if node.module and not node.level:
                    imports[local] = node.module + "." + alias.name
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            shadowed.add(node.id)
        elif isinstance(node, ast.arg):
            shadowed.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            shadowed.add(node.name)
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del)):
            value = node.value
            while isinstance(value, ast.Attribute):
                value = value.value
            if isinstance(value, ast.Name):
                shadowed.add(value.id)
    shadowed.update(name for name, count in bindings.items() if count > 1)
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        parts = []
        value = node.func
        while isinstance(value, ast.Attribute):
            parts.append(value.attr)
            value = value.value
        if not isinstance(value, ast.Name) or value.id in shadowed:
            continue
        qualified = imports.get(value.id)
        if qualified is None and value.id not in bindings and not parts and value.id in {"eval", "exec"}:
            qualified = "builtins." + value.id
        if qualified is None:
            continue
        qualified += "".join("." + part for part in reversed(parts))
        if qualified in EXECUTION_CALLS:
            calls.append({"call": qualified, "line": node.lineno})
    return calls


def package_source_findings(zf: zipfile.ZipFile, path: Path,
                            members: list[str]) -> list[Finding]:
    """Inspect Python modules without importing or executing any model code."""
    findings = []
    consumed = 0
    member_counts = Counter(info.filename for info in zf.infolist())
    for index, member in enumerate(members):
        reason = None
        if member_counts[member] != 1:
            reason = "duplicate_source_member"
        elif index >= MAX_SOURCE_MEMBERS or consumed >= MAX_TOTAL_SOURCE_BYTES:
            reason = "source_budget"
        else:
            cap = min(MAX_SOURCE_BYTES, MAX_TOTAL_SOURCE_BYTES - consumed)
            try:
                with zf.open(member) as handle:
                    source = handle.read(cap + 1)
                consumed += len(source)
                if len(source) > cap:
                    reason = "source_size"
                else:
                    tree = ast.parse(source, filename=member)
                    operations = _execution_calls(tree)
                    if operations:
                        findings.append(Finding(
                            rule_id="MFV-TORCH-002", severity=Severity.HIGH,
                            category=Category.DESERIALIZATION, file_path=str(path),
                            confidence=0.85, cwe_ids=[94],
                            message=f"Packaged Python module {member} contains explicit "
                                    "execution operations: " + ", ".join(sorted({
                                        op['call'] for op in operations
                                    })) + ". These calls can execute when the relevant "
                                    "module or model is imported/invoked. Static operation "
                                    "evidence; reachability and intent are not established.",
                            metadata={"source_member": member, "operations": operations[:20],
                                      "archive_kind": "torch_package",
                                      "evidence_tier": "static-operation"},
                        ))
            except (OSError, zipfile.BadZipFile, RuntimeError, NotImplementedError,
                    ValueError, EOFError, SyntaxError, MemoryError, RecursionError) as exc:
                reason = type(exc).__name__
        if reason:
            findings.append(Finding(
                rule_id="MFV-SKIP-002", severity=Severity.LOW, category=Category.AI_ML,
                file_path=str(path), confidence=0.5,
                message=f"Python source analysis did not complete for {member} "
                        f"({reason}). This is not a clean verdict for that source.",
                metadata={"source_member": member, "skipped_reason": reason},
            ))
            if reason == "source_budget":
                break
    return findings
