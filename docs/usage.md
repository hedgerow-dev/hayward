# Usage

## Install

```bash
pip install hayward
```

Python 3.10 or newer. One runtime dependency, `defusedxml`. No native
extensions, no model framework, and `hayward scan` makes no network calls at any point. Only
[`hayward hub`](#scan-a-hub-repo) and creating a [`hayward lock`](#pin-a-hub-repo) use the
network, and only when you run them.

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

## Scan a Hub repo

```bash
hayward hub org/name
hayward hub org/name --revision v1.0 --fail-on high --format sarif -o report.sarif
```

This fetches a Hugging Face repo's scannable files and runs `scan` on them, so you
can check a repo before you download it into your own project. It uses the
network, and nothing is loaded or run. Any other flag goes to `scan`.

- Files come from one pinned commit, the one the Hub reports for `--revision`
  (default `main`). A file with a Hub hash is checked against it after download.
- It fetches what a directory scan would find: the weights, every `.json`
  config, `.jinja` chat templates and tokenizer data (`merges.txt`, `vocab.txt`,
  `tokenizer.model`), so a hub scan covers the same files as scanning a
  download. `.py` files are not scanned: custom code is never vouched for.
- Files over `--max-file-mb` (default 50) are not fetched whole. A **zip
  checkpoint** (what `torch.save` writes: `.pt`, `.pth`, `.bin`, `.ckpt`) is read
  by Range request for its pickle member only, which is the part that can run
  code. A **`.safetensors`** file is read for its header only, which holds
  everything a loader acts on (dtypes, shapes, offsets, metadata). The run says
  so on stderr; the tensor data is not read and the partial read cannot be
  checked against the Hub's file hash. At most 64 MB is fetched per file and 16
  pickle members are read.
- Anything else over the cap is named on stderr as `NOT SCANNED`, with the
  reason: an older non-zip checkpoint, a `.gguf` or `.onnx` file, a zip that
  will not parse. A clean result does not cover those files; raise
  `--max-file-mb` to fetch them whole.
- With `--receipt`, a file read in part is recorded under the Hub's own SHA-256
  for it, marked `"read": "header"` or `"read": "pickle members"` and
  `"hash_source": "hub"`, so the receipt names the real file and verifies
  against a full download. One with no Hub hash is left out of the receipt.
- A private or gated repo needs `HF_TOKEN` in the environment. It is sent only to
  `huggingface.co`, and not on to the CDN it redirects to.
- The report names the target `hf://org/name@commit`. The temp folder is deleted.

## Pin a Hub repo

```bash
hayward lock org/name --revision main -o hayward.lock.json
hayward lock --verify hayward.lock.json ./downloaded-snapshot
```

`lock` records the commit and every file in the repo with its hash (SHA-256 for
LFS files, the git blob id for the rest). `--verify` checks a downloaded snapshot
against it, offline: a file that is missing, changed, or **not in the lock** fails,
so an extra `.py` cannot ride along.

It is written for one profile: `transformers` `from_pretrained` at a pinned
revision. It fails closed. Custom code files are listed, and anything the profile
would load from outside the repo is recorded as unresolved and the lock is marked
`needs_review` (exit code 1): an `auto_map` entry that loads code from another
repo (`other/repo--module.Class`), or an adapter's base model. A lock pins what
will be loaded; it does not say the files are safe. Scan them as well.

## Scan receipts

```bash
hayward scan ./models --receipt receipt.intoto.json
hayward receipt verify receipt.intoto.json ./models
```

`--receipt` writes an unsigned [in-toto](https://in-toto.io) statement. Its
subjects are the scanned files with their SHA-256 digests. Its predicate
(`https://hedgerow.dev/hayward/scan-receipt/v1`) records the Hayward version and
settings, the scan time, and for each file its verdict, the rules that fired and
its coverage gaps (what was not fully read). It is taken from the scanner's own
findings, before `--policy` and `--allowlist`, so your acceptance decisions stay
separate from what the scanner saw.

`receipt verify` checks a directory against it, offline: a file that is missing
or changed fails, and so does a model file the receipt does not cover. Verify
the same directory you scanned.

### Signing a receipt

With the optional extra, Hayward signs a receipt with Sigstore and verifies the
signature:

```bash
pip install "hayward[sign]"
hayward receipt sign receipt.intoto.json          # writes receipt.intoto.json.sigstore.json
hayward receipt verify receipt.intoto.json.sigstore.json ./models \
  --identity https://github.com/ORG/REPO/.github/workflows/scan.yml@refs/heads/main \
  --issuer https://token.actions.githubusercontent.com
```

In CI with a workload identity (GitHub Actions with `id-token: write`, GitLab
and others) signing is automatic. On a laptop it opens a browser to log in, and
because **signing writes that login's email to Sigstore's public transparency
log, permanently**, it asks first (`--yes` skips the question).

A signed receipt only verifies with `--identity` and `--issuer`: a valid
signature from just anyone proves nothing. The output is a standard Sigstore
bundle, so a receipt signed by the action's `attest` input (download it with
`gh attestation download`) verifies the same way, and `cosign` and
`gh attestation verify` can check Hayward's bundles. Only signing and signed
verification use the network.

You can also sign with cosign:

```bash
cosign attest-blob --type https://hedgerow.dev/hayward/scan-receipt/v1 \
  --predicate <(jq .predicate receipt.intoto.json) models/model.safetensors
```

A receipt is analysis evidence, not a safety certificate. Old receipts are never
rewritten: rescan to get a new one, and treat the newest receipt for a digest as
current.

## Check a published verdict

```bash
pip install "hayward[sign]"
hayward verify org/name                       # the commit main points at
hayward verify org/name --revision v1.0 ./downloaded-snapshot
```

The hayward-verify service (verify.hedgerow.dev) scans popular Hub repos at pinned
commits and publishes a signed scan receipt for each one it finds clean or could
not fully read. `hayward verify` resolves the revision to a commit and fetches the
receipt for it, then checks four things:

- it was signed by the service's own workflow (`hedgerow-dev/hayward-verify`)
  on its weekly schedule, from that repo and owner by their numeric GitHub ids;
  a manual run, any other workflow, or a renamed or recreated repo is refused
- it was signed under 30 days ago, by the signing certificate's time (the
  service re-signs weekly, so an older verdict means the repo is no longer
  being checked)
- it names this repo and this commit
- its file hashes match the Hub's
- with a directory, your downloaded files match it, and **every** file in the
  directory is the file at that commit (the Hub's SHA-256 for LFS files, the git
  blob id for the rest), so a changed `.py` or config the scan never covered is
  caught too. A Hugging Face cache snapshot works; files you did not download are
  fine.

It prints each file's verdict and every file the verdict does not cover (with a
directory, only the files you downloaded: download what you will load, and
verify that), marking `.py` files as CODE: custom model code runs with
`trust_remote_code=True` and Hayward does not vouch for it. Documentation,
licences and images are not listed. It shows which Hayward version made the
verdict and when, and says so when that is older than yours. It ends with the
full commit id: load that exact revision, since a branch can move after you
check it. A verdict that covers no files is not a pass.

- **No published verdict is not a pass.** The repo may not have been scanned yet,
  or may have been held back: the service never publishes a flagged repo, so
  "nothing published" and "flagged" look the same. Exit code 1.
- **Gaps fail.** A verdict means "vouched for", so a file that was not fully
  checked (`not_verified`), one that runs the repo's own code (`remote_code`),
  or one not covered fails the command. `--allow-gaps`
  accepts them, still listed. This is the opposite of `scan`, on purpose.
- Big SafeTensors files are checked by their header and shown as `(header read)`.
- `--service`, `--identity`, `--issuer`, `--trigger`, `--repository-id` and
  `--owner-id` point it at a self-hosted service.

A verdict is analysis evidence, not a safety certificate.

## Explain a pickle

```bash
hayward explain model.pt
hayward explain model.pt --all
```

`scan` says whether a file is dangerous; `explain` shows what it would do, as
readable lines, without running anything:

```text
_0 = builtins.__import__('subprocess')
    # why: on Hayward's deny list: can run code, commands, or reach files or the network
_1 = builtins.getattr(_0, 'getoutput')
    # why: on Hayward's deny list: can run code, commands, or reach files or the network
_2 = _1('echo $(id)')
    # why: calls the result of an earlier line; argument looks like a shell command
```

It reads a bare pickle, the `.pkl` members of a zip checkpoint, and the members
of an old tar checkpoint. Each callable is classified with the scanner's own
allow and deny lists, so `explain` and `scan` agree. Calls to known-safe
functions, and objects rebuilt from classes with no arguments, are counted on
one line instead of listed; `--all` lists them. Long strings, big containers and
deep nesting are shortened, and a hostile or truncated stream stops with a note
saying where. It is for reading a file you already have; it is not a verdict.

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
  "version": "1.1.0",
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
  "coverage_gaps": [],
  "verdicts": {"/srv/models/model.pt": "flagged"}
}
```

`coverage_gaps` lists files that were not fully read. See
[coverage](coverage.md).

`verdicts` gives each file with findings one verdict, separate from severity:
`flagged` (a detection at LOW or above), `remote_code` (the file makes the
loader import the repo's own Python, `MFV-HF-002`: how many legitimate models
work, but code nobody vouched for), `not_verified` (only coverage gaps, so the
file was not fully read) or `clean` (only INFO notes). A detection wins over
remote code, and both win over a coverage gap. A file with no findings is clean and is not listed. In
SARIF output each result carries its file's verdict as `properties.verdict`.

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

Add Hayward to `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/hedgerow-dev/hayward
    rev: v1.2.6
    hooks:
      - id: hayward
```

The hook scans each staged file whose extension Hayward reads and blocks the
commit on any finding at `high` or above. To change the threshold, override
`args`, for example `args: [--fail-on, critical]`.

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

## Coding agents (MCP)

```bash
pip install "hayward[mcp]"
claude mcp add hayward -- hayward mcp
```

`hayward mcp` runs a stdio MCP server with one tool, `scan_path(path,
fail_on="high")`. It returns JSON with the findings and a `verdict`:

- `blocked`: a finding at or above `fail_on`, the same gate as `--fail-on`.
- `incomplete`: nothing reached the gate, but a file was not fully read.
- `clean`: neither.

A missing path comes back as a tool error. The tool is read-only, never loads
the model and makes no network calls, so an agent can be told to scan any
model file before loading it. The extra adds the official MCP Python SDK
(MIT); the core install is unchanged.

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

## Load guard

```python
import hayward.guard  # one line, top of a notebook or script
```

From then on four loaders are guarded:

- `pickle.load`
- `joblib.load`
- `numpy.load`, only when `allow_pickle=True`
- `torch.load`

Each guarded call does two checks. First it scans the file the way that
loader will read it, and refuses it on a finding at or above `high`. Then,
while the loader runs, every global the unpickler resolves on that thread is
checked before it is imported, so a file the scan read differently from the
loader is still caught when it names a denied callable.

A refused load raises `hayward.guard.UnsafeModelError`, a subclass of
`pickle.UnpicklingError`, so an existing `except` around a load still
catches it. The message is the CLI's text for the findings; `.file` and
`.rule_ids` carry the same detail.

```python
import hayward.guard

hayward.guard.install(strict=True)             # refuse anything not known-safe
hayward.guard.install(fail_on="critical")      # same scale as --fail-on
hayward.guard.install(mode="warn")             # warnings.warn, then load anyway
hayward.guard.install(max_size=2_000_000_000)  # raise the scan's 500MB cap, in bytes
hayward.guard.install(prescan=False)           # keep only the load-time check
hayward.guard.uninstall()                      # restore the original loaders
```

Each `install` call replaces the previous settings. A denied callable is
CRITICAL, so outside strict mode `fail_on="never"` lets it load, like warn
mode without the warning. `HAYWARD_GUARD=off` makes the import a no-op, for
CI that already scans.

### Remote code

With `trust_remote_code=True`, transformers runs the model repo's own `.py`
files. The guard refuses that code before it runs, as `MFV-HF-002`, for any
`from_pretrained` call, unless you name the repo:

```python
hayward.guard.install(remote_code=["org/name"])  # a Hub repo
hayward.guard.install(remote_code=["my_model"])  # a local model folder, by its name
```

Naming a repo allows its code at any revision, so pin `revision=` as well, or
check the snapshot with `hayward lock --verify`. Like other findings, it
follows `fail_on` and `mode`: it is HIGH, so `fail_on="critical"` lets it run
and warn mode warns. Strict mode refuses it whatever `fail_on` is. A one-part
name such as `my_model` matches a local folder of that name wherever it is.

### Default mode raises the bar; it is not a sandbox

By default the guard refuses denied callables and anything the scan rates
at or above `fail_on`. A global on neither the deny list nor the known-safe
list loads, as it is INFO in the CLI. That is enough to stop known malicious
files and the parser tricks the scanner has been tested against. It is not
enough for a file from someone you do not trust: a pickle built only from
unrecognized globals can still run code, including by reaching the guard's
own state from inside the load. The scan catches some of those from their
arguments, not all of them.

### Strict mode

`install(strict=True)` refuses every global that is not on the scanner's
known-safe list, the list behind `MFV-PICKLE-004`. It applies at load time
and to the scan's `MFV-PICKLE-004` and `MFV-PICKLE-005` findings, whatever
`fail_on` is. This is the setting for files you do not trust.

It refuses some ordinary files. What loads: built-in containers, numpy
arrays, torch tensors and state dicts, standard-library values such as
`datetime`, `decimal`, `uuid` and `pathlib`, class-shaped objects from
scikit-learn and SciPy, and transformers' `TrainingArguments`. What it
refuses includes pandas objects, numpy random generator state, other
transformers objects, a whole pickled `nn.Module`, TorchScript archives and
your own classes. Strict mode trusts the known-safe list, including its
name-shape rule for scikit-learn and SciPy.

### What it does not cover

- Unpickling that does not go through a guarded loader is untouched in both
  modes: `pickle.loads`, an `Unpickler` used directly, and dill's and
  cloudpickle's own load functions. This keeps in-process IPC, caches and
  multiprocessing working. They are checked only when they run inside a
  guarded load on the same thread.
- A loader bound before the guard was installed (`from pickle import load`,
  or a library that kept its own reference) or called by another name, such
  as `torch.serialization.load`.
- TorchScript. `torch.load` hands a TorchScript archive to `torch.jit.load`,
  which unpickles in C++ where the load-time check cannot see. Only the scan
  covers it.
- `torch.load(weights_only=True)` uses torch's own restricted unpickler; the
  scan still runs, the load-time check does not.
- In default mode, work a pickle hands to another thread.
- Remote code that is not run by transformers from its `transformers_modules`
  cache, such as diffusers' custom pipelines, and remote code transformers
  already imported before the guard was installed.
- A stream that cannot seek is not scanned first; the load-time check still
  applies. A file the scan could not fully read (a coverage finding,
  `MFV-SKIP-*`) warns and loads, with the load-time check still on.

The scan reads the whole file before the loader does, so on a large file it
costs more than the load. A path is remembered for the life of the process
by path, size and mtime, so reloading the same checkpoint does not rescan
it.

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
structurally claims. This is detection only: scanning does no network I/O and
no cryptographic verification, so a present signature is reported, never
trusted.
