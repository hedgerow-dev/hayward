# Usage

## Install

```bash
pip install hayward
```

Python 3.10 or newer. One runtime dependency, `defusedxml`. No native
extensions, no model framework, and no network access at any point.

## Command line

```bash
hayward scan model.pt              # one file
hayward scan ./models              # a directory, recursively
hayward scan ./models -f json      # machine-readable
hayward scan ./models -f html -o report.html   # shareable
```

| Option | Default | Effect |
|---|---|---|
| `-f, --format` | `text` | `text`, `json`, `html` or `markdown` |
| `-o, --output` | stdout | Write the report to a file |
| `--fail-on` | `high` | Lowest severity that exits non-zero. `critical`, `high`, `medium`, `low`, `info`, `never` |
| `--fail-on-coverage` | off | Also exit non-zero when a file could not be fully read |
| `--no-colour` | off | Plain output regardless of terminal |

### Exit codes

| Code | Meaning |
|---|---|
| 0 | Nothing at or above the fail threshold |
| 1 | Findings at or above the fail threshold |
| 2 | The scan could not run |

`--fail-on` defaults to `high`, so INFO and LOW do not break a build. Whichever
threshold you pick, state it whenever you quote a number from it. The
difference between counting the INFO tier and not counting it is the single
largest lever on any scanner's false-positive rate, this one included.

## Reports

`html` produces one self-contained page: no external stylesheet, font or
script, so it opens from an email attachment on a machine with no network and
renders the same. `markdown` is for pasting into a ticket or a pull request.
`json` is for machines.

The desktop app exports the same three through **Export report**, choosing the
format from the extension you give the file. A report written from the window
is byte-identical to one written from the command line.

### JSON output

```json
{
  "tool": "hayward",
  "version": "1.0.0",
  "root": "/srv/models",
  "findings": [
    {
      "rule_id": "MFV-PICKLE-001",
      "message": "Pickle file references unsafe callable(s) ...",
      "severity": "critical",
      "category": "deserialization",
      "file": "/srv/models/model.pt",
      "confidence": 0.95,
      "cwe_ids": [502],
      "metadata": {}
    }
  ],
  "coverage_gaps": []
}
```

`coverage_gaps` lists files that were not fully read. See
[coverage](coverage.md).

## Continuous integration

### GitHub Actions

```yaml
- run: pip install hayward
- run: hayward scan ./models --fail-on high --fail-on-coverage
```

### GitLab CI

```yaml
model-scan:
  image: python:3.12
  script:
    - pip install hayward
    - hayward scan ./models --fail-on high -f json > hayward.json
  artifacts:
    when: always
    paths: [hayward.json]
```

### Pre-commit

```yaml
repos:
  - repo: local
    hooks:
      - id: hayward
        name: scan model files
        entry: hayward scan
        language: system
        files: '\.(pt|pth|pkl|bin|safetensors|gguf|h5|keras|onnx|npy|npz|joblib)$'
```

## Desktop

```bash
hayward-gui
```

One window. Choose a file or a folder, or drop one on it if your Tk build has
`tkdnd`. **Export report** writes HTML, Markdown or JSON, choosing the format
from the extension you give the file. Results are listed by severity, and
selecting one shows the full message and any CWE mapping. The "Show unknowns"
checkbox controls whether INFO findings are listed, the same choice the CLI's
`--fail-on` threshold makes.

Built on tkinter, which ships with Python, so the desktop app adds no
dependency. The scan runs on a worker thread, so a large directory does not
freeze the window.

## Python

```python
from pathlib import Path
from hayward import ModelFileScanner, Severity, is_coverage_gap

scanner = ModelFileScanner()

findings = scanner.scan_file(Path("model.pt"))
findings = scanner.scan_directory(Path("models"))
```

`scan_file` and `scan_directory` are the whole API. Both return a list of
`Finding`.

```python
blocking = [f for f in findings if f.severity_order <= Severity.HIGH.value]
unread   = [f for f in findings if is_coverage_gap(f)]

for f in findings:
    print(f.rule_id, f.severity.value, f.file_path, f.message)
    print(f.to_dict())
```

### Finding

| Attribute | Type | Notes |
|---|---|---|
| `rule_id` | `str` | Stable identifier, catalogued in [rules.md](rules.md) |
| `message` | `str` | What was found, including the resolved evidence |
| `severity` | `Severity` | `CRITICAL`, `HIGH`, `MEDIUM`, `LOW`, `INFO` |
| `category` | `Category` | Coarse grouping for reporting |
| `file_path` | `str` | Absolute path |
| `confidence` | `float` | The scanner's own estimate. Reported for information; it does not change severity |
| `cwe_ids` | `list[int]` | CWE mappings where one applies |
| `metadata` | `dict` | Rule-specific detail, including `skipped_reason` |

`severity_order` sorts findings, with 0 as the most severe.

Importing the package is cheap: one dependency, no model framework and no
native extensions, so calling it per file in a loop is fine.

## Suppressing findings

There is no ignore file. Filter on `rule_id` in your own code or with `jq`,
and keep the decision visible in your repository rather than inside the
scanner:

```bash
hayward scan ./models -f json \
  | jq '.findings | map(select(.rule_id != "MFV-PICKLE-004"))'
```
