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
hayward scan a.pt ./models b.pkl   # several files and directories at once
hayward scan ./models -f json      # machine-readable
hayward scan ./models -f html -o report.html   # shareable
```

`scan` takes one or more targets. Each may be a file or a directory, and their
findings are aggregated into one report. A single directory target is reported
relative to itself, as before; with several targets the report shows paths
relative to their common ancestor (or absolute paths when they share none).

| Option | Default | Effect |
|---|---|---|
| `-f, --format` | `text` | `text`, `json`, `html`, `markdown`, `sarif` or `cyclonedx` |
| `-o, --output` | stdout | Write the report to a file |
| `--fail-on` | `high` | Lowest severity that exits non-zero. `critical`, `high`, `medium`, `low`, `info`, `never` |
| `--fail-on-coverage` | off | Also exit non-zero when a file could not be fully read |
| `--exclude` | off | Glob pattern (fnmatch) of files/directories to skip. Repeatable (see [Excluding paths](#excluding-paths)) |
| `--max-size` | `500M` | Override the per-file scan cap. Plain byte count or a `k`/`m`/`g` suffix, e.g. `200M`, `1G`, `500k` |
| `--jobs` | `1` | Scan a directory's files across N worker processes (see [Parallel scanning](#parallel-scanning)) |
| `--policy` | off | JSON per-rule severity overrides, applied first (see [Policy overrides](#policy-overrides)) |
| `--cache` | off | Content-hash cache so a re-run skips unchanged files (see [Scan cache](#scan-cache)) |
| `--progress` | off | Write a running `scanned N/total` counter to stderr |
| `--verbose` | off | Log each file to stderr as it is scanned |
| `--quiet` | off | Suppress progress and informational lines on stderr |
| `--allowlist` | off | JSON allowlist of findings to suppress (see [Allowlisting](#allowlisting-findings)) |
| `--baseline` | off | A prior JSON report; `--fail-on` then applies only to *new* findings (see [Baseline mode](#baseline-mode)) |
| `--check-signatures` | off | Also report sibling signature/attestation artifacts. Detection only, not verification |
| `--color` | `auto` | `auto`, `always` or `never`. `auto` uses colour only when stdout is a terminal and `NO_COLOR` is not set |
| `--no-colour` | off | Plain output regardless of terminal (same as `--color never`) |

`--progress`, `--verbose` and `--quiet` write only to stderr, never stdout, so a
piped report (for example `-f json`) stays clean.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | Nothing at or above the fail threshold |
| 1 | Findings at or above the fail threshold |
| 2 | The scan could not run: usage error, unreadable target, or a crash inside the scanner |

`--fail-on` defaults to `high`, so INFO and LOW do not break a build. Whichever
threshold you pick, state it whenever you quote a number from it. The
difference between counting the INFO tier and not counting it is the single
largest lever on any scanner's false-positive rate, this one included.

## Reports

`html` produces one self-contained page: no external stylesheet, font or
script, so it opens from an email attachment on a machine with no network and
renders the same. `markdown` is for pasting into a ticket or a pull request.
`json` is for machines. `sarif` is SARIF 2.1.0, the format CI platforms such
as GitHub code scanning ingest. `cyclonedx` is a CycloneDX 1.6 ML-BOM: one
component per scanned file, one vulnerability per finding, for tools that
consume a bill of materials.

The desktop app exports the same formats through **Export report**, choosing
the format from the extension you give the file. A report written from the
window is byte-identical to one written from the command line.

### JSON output

```json
{
  "schema_version": 1,
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

To publish results to GitHub code scanning, use the bundled composite action,
which runs the scan and writes SARIF for `github/codeql-action/upload-sarif`.
See [the GitHub Action guide](github-action.md).

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

`hayward scan` accepts several targets, but a hook that forwarded every staged
filename would scan unrelated files. Scan your models directory instead and use
`pass_filenames: false`; the `files` pattern then only controls *when* the
hook runs (whenever a matching file is staged). Point the entry at whatever
directory holds your model files.

```yaml
repos:
  - repo: local
    hooks:
      - id: hayward
        name: scan model files
        entry: hayward scan ./models --fail-on high
        language: system
        pass_filenames: false
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
blocking = [f for f in findings if f.severity in (Severity.CRITICAL, Severity.HIGH)]
unread   = [f for f in findings if is_coverage_gap(f)]

for f in sorted(findings, key=lambda f: f.severity_order):
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

## Excluding paths

`--exclude` takes a glob pattern (Python `fnmatch`) and skips anything it
matches. It is repeatable, so pass it more than once to skip several patterns.

The pattern is matched against the path as walked: the file's route relative to
the scan root, its bare filename, and every individual path component. Matching
a component is what lets a directory-name pattern prune a whole subtree, not
just a file that happens to share the name.

```bash
hayward scan ./models --exclude '*.onnx' --exclude 'checkpoints'
```

The first pattern skips every ONNX file; the second skips any `checkpoints`
directory and everything beneath it.

## Parallel scanning

`--jobs N` fans a directory's files out across N worker processes. The default
is `1`, which is fully sequential and unchanged from before, so nothing about
an existing run differs until you ask for more workers.

```bash
hayward scan ./models --jobs 8
```

The result is identical to a sequential scan: the same files are read with the
same settings (your `--max-size` and `--check-signatures` reach every worker),
and no file is dropped or counted twice. Only the order findings are gathered
in differs, which the report normalises anyway. Pair it with `--progress` to
watch a large tree go by.

## Policy overrides

`--policy` points at a small JSON file that remaps the severity of named rules,
so a team can tune what a finding costs without touching the built-in rule set.

```json
{
  "severity_overrides": {
    "MFV-PICKLE-004": "low",
    "MFV-HF-002": "critical"
  }
}
```

Only the severity changes; nothing else about a finding is recomputed, and a
rule you do not list is untouched. An unknown severity string is rejected at
load with exit code 2, so a typo surfaces immediately rather than silently
matching nothing.

The override is applied **first**, before allowlist suppression, before the
baseline diff, and before the `--fail-on` gate, so every later stage sees the
severities you chose. A remapped `critical` you have accepted down to `low`
stops failing a `--fail-on high` build; a `low` you raise to `critical` starts
failing one.

## Scan cache

`--cache` stores each file's findings keyed by the sha256 of its bytes, so a
re-run skips any file whose content has not changed since the cache was
written.

```bash
hayward scan ./models --cache .hayward-cache.json
```

The key is the content hash, not the modification time, so an edit that
preserves the mtime (a restore, a `touch -r`, a checkout) still misses the
cache and gets re-scanned. The cache is also stamped with the hayward version
and dropped wholesale on an upgrade, so a new or changed rule can never be
hidden behind a stale clean verdict. A missing or corrupt cache file is treated
as empty, and a cache that cannot be written prints a warning without changing
the scan's exit code. The cache composes with `--jobs`: cache hits are served
in the main process and only the misses are handed to the workers.

## Baseline mode

A team adopting the scanner on an existing pile of models does not want CI to
fail on the whole backlog at once. `--baseline` compares a fresh scan against a
prior JSON report and applies `--fail-on` only to findings that are **new**
relative to it. The report still shows everything; only the exit code changes.

```bash
# Record the baseline once, from a known state.
hayward scan ./models -f json -o hayward-baseline.json

# In CI, fail only on findings introduced since that baseline.
hayward scan ./models --baseline hayward-baseline.json --fail-on high
```

A one-line summary (`N new, N unchanged, N fixed`) is written to stderr. A
finding is "the same" across runs when its rule id, its file (relative to the
scan root), and its rule-specific detail all match, so moving a file or
re-running changes nothing, while a genuinely new issue counts as new.

## Allowlisting findings

For findings you have reviewed and accept, `--allowlist` suppresses them from
the report and the exit code, with an audit trail. The allowlist is a JSON
file, reviewable in your repository, where each entry is keyed by the file's
**sha256** and the **rule id** and carries a required justification:

```json
[
  {
    "sha256": "de4e240c47bb...<64 hex>",
    "rule_id": "MFV-PICKLE-004",
    "justification": "reviewed 2026-08-27, benign custom optimiser",
    "approved_by": "you@example.com",
    "expires": "2027-01-01"
  }
]
```

```bash
hayward scan ./models --allowlist hayward-allowlist.json
```

Keying on the content hash is the point: the moment the file's bytes change, the
entry stops matching and the finding resurfaces for re-review. An `expires` date
does the same on a schedule. Every suppression is announced on stderr, so the
report is clean but the decision is never silent.

An entry suppresses **every** finding of that rule on that exact file content,
not one specific hit. If a file trips the same rule twice (two unknown globals,
say), one entry hides both, so review all of a rule's findings on a file before
you accept it. The content hash keeps this honest: any edit to the file voids
the entry.

For an ad-hoc filter without a config file, `rule_id` is in the JSON report, so
`jq` works too:

```bash
hayward scan ./models -f json \
  | jq '.findings | map(select(.rule_id != "MFV-PICKLE-004"))'
```

## Signatures

`--check-signatures` reports sibling signature and attestation artifacts next to
a model (a detached `.sig`, a Sigstore bundle, an in-toto/SLSA/DSSE
attestation), as an INFO `MFV-SIG-001` finding stating what the artifact
structurally claims. This is detection only: the scanner does no network I/O and
no cryptographic verification, so a present signature is reported, never
trusted.
