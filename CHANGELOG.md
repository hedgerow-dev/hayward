# Changelog

Notable changes, newest first. Versions follow [semantic versioning](https://semver.org).
Rule identifiers are part of the public interface and will not be renamed
within a major version.

## Unreleased

### Added

- `hayward verify --sha256 HASH` checks one file hash against the service's
  signed verdict log: a checkpoint under 3 days old, the shard it names, and
  every current verdict for those bytes. It passes only if all are clean with
  no coverage gaps (`--allow-gaps` as for a repo). No entry is not a pass.

### Fixed

- Two false alarms found by scanning popular Hub repos. `MFV-HF-001` and
  `MFV-GGUF-003` no longer read `selectattr(` and `rejectattr(`, ordinary Jinja
  filters, as the `attr` filter. `MFV-ST-006` no longer flags a single tensor
  over 4 GB: real models ship far bigger ones (DeepSeek, 98 GB). A tensor must
  still match its shape and dtype and fit its file.

## 1.11.1 (2026-10-31)

### Changed

- `hayward hub` fetches up to 8 files at a time. A big repo is hundreds of
  shards, each a header read, so it was spending most of its time waiting on
  one request after another. Results are reported in the Hub's file order, and
  a hash mismatch in any file, or Ctrl-C, stops it without waiting for the
  files still queued.

### Fixed

- `hayward hub` refuses files whose names differ only by case or Unicode form
  (`Model.pkl` and `model.pkl`). On macOS and Windows they are one file, so one
  could replace the other before the scan and its own bytes would never be read.

## 1.11.0 (2026-10-30)

### Added

- A SafeTensors file over the 500 MB scan cap is checked by its header (dtypes,
  shapes, offsets, metadata: everything a loader acts on) instead of reported
  as never scanned (`MFV-SKIP-001`). Most real models are multi-GB SafeTensors
  files. Its tensor bytes are not searched for embedded binaries.
- `hayward hub` reads a big `.safetensors` file's header by Range request, so a
  15 GB model is checked in seconds instead of being NOT SCANNED.
- `hayward verify org/name [DIR]` checks a Hub repo against the signed verdict
  the hayward-verify service publishes for its commit: signed by the service's
  own workflow on its schedule, under 30 days old, naming this repo and commit,
  with hashes that match the Hub's, and, with DIR, every file matching that
  commit. Gaps fail unless `--allow-gaps`. No published verdict is not a pass.
  The signer is pinned by its repo's and owner's numeric GitHub ids, not only
  by name, and a verdict's age comes from its signing certificate, not from the
  receipt. Needs `hayward[sign]`.
- Receipts from `hayward hub` name their source (`predicate.source`: the repo
  and commit).
- `hayward hub` fetches every file a directory scan would find: every `.json`
  config (not only config.json and tokenizer_config.json), `.jinja` chat
  templates and tokenizer data, so a hub scan covers the same files as a
  scanned download.
- `chat_template.jinja` (what recent transformers writes) is checked for
  template injection (`MFV-HF-001`), and directory scans find it and the
  tokenizer data files (`merges.txt`, `vocab.txt`, `tokenizer.model`), which are
  checked to be what their name says.
- `hayward verify` with a directory counts gaps only among the files you
  downloaded.

### Changed

- A file whose only detection is `MFV-HF-002` (it makes transformers import
  the repo's own Python) gets the verdict `remote_code` instead of `flagged`.
  Many legitimate models work this way; the code is still not vouched for, and
  `hayward verify` treats it as a gap. Severity and the `--fail-on` gate are
  unchanged.

### Fixed

- `hayward hub --receipt` recorded a big checkpoint read in part under the hash
  of the partial local copy, so the receipt could never verify against the real
  file. A file read in part is now recorded under the Hub's own SHA-256, marked
  `read` and `hash_source: hub`, or left out when the Hub gives no hash.

## 1.10.0 (2026-10-30)

### Added

- Signing receipts outside GitHub (#38, phase 2). With the optional
  `hayward[sign]` extra (sigstore-python), `hayward receipt sign` writes a
  Sigstore bundle over the exact receipt bytes, and `hayward receipt verify`
  checks a bundle with a required `--identity` and `--issuer` before checking
  the files. In CI signing uses the workload identity; on a laptop it asks
  first, since the login's email goes into Sigstore's public log. Bundles from
  the action's `attest` input verify the same way.

## 1.9.1 (2026-10-30)

### Fixed

- The GitHub Action's `attest: true` now signs the receipt when the scan fails
  the build, as documented. In 1.9.0 the signing steps were skipped after a
  failing scan, because a composite action adds `success()` to any condition
  without a status function. Found by running the action on a real runner.

## 1.9.0 (2026-10-30)

### Added

- Scan receipts (#38). `hayward scan --receipt FILE` writes an unsigned in-toto
  statement: each scanned file's SHA-256, plus the Hayward version, settings,
  and each file's verdict, rules and coverage gaps. It is taken before
  `--policy` and `--allowlist`. `hayward receipt verify FILE DIR` checks a
  directory against it offline. The GitHub Action gains `receipt` and
  `attest` inputs; `attest: true` signs the receipt with `actions/attest`
  (keyless Sigstore). A receipt is analysis evidence, not a safety
  certificate.

## 1.8.0 (2026-10-30)

### Added

- A verdict per file, separate from severity (#14): `flagged`, `not_verified`
  or `clean`. JSON reports carry it as `verdicts`, and each SARIF result as
  `properties.verdict`. `--fail-on` and `--fail-on-coverage` are unchanged.

## 1.7.2 (2026-10-30)

### Fixed

- `MFV-GGUF-003` now checks named chat templates
  (`tokenizer.chat_template.<name>`, for example `.tool_use`) for Jinja2
  code-execution constructs, not just `tokenizer.chat_template`. llama.cpp can
  select and render a named template, so a payload there was missed before.
  The finding names the key it came from.
- GGUF metadata is content-checked up to 64 MB, up from 10 MB. Big
  vocabularies (Gemma 4, Qwen 3.8) run 10 to 15 MB, so every key after them was
  left unchecked with an `MFV-GGUF-004` coverage note. In a sample of 108
  popular GGUF files this went from 10 partly scanned files to none.

## 1.7.1 (2026-10-30)

### Fixed

- Four kinds of legitimate GGUF file no longer raise a false alarm, found by
  scanning a sample of popular Hub GGUF repos. `MFV-GGUF-005` accepts tensors
  with 0 to 8 dimensions (ComfyUI-GGUF video VAEs use 5, a scalar has 0) and
  tensor type ids from forks' own quantisations. The u64 wrap and overrun
  checks still run on every tensor. `MFV-GGUF-002` treats a vocabulary kept
  under an architecture's own prefix (`<arch>.tokenizer.tokens`) as data, like
  `tokenizer.ggml.tokens`.

## 1.7.0 (2026-10-30)

### Added

- The load guard refuses remote code. With `trust_remote_code=True`,
  transformers runs a model repo's own Python files; once `hayward.guard` is
  imported, that code is refused before it runs (`MFV-HF-002`) unless
  `install(remote_code=["org/name"])` names the repo. This changes what
  `import hayward.guard` does for anyone loading remote code today.

## 1.6.0 (2026-10-30)

### Fixed

- Walking a tar can no longer be made to pin the CPU (#10). A compressed tar now
  has a 500 MB budget on decompressed bytes, checked before each member is read
  or skipped, and a GNU long-name or pax header over 1 MB is refused before it is
  read into memory. Either stops the walk with an `MFV-SKIP-002` coverage
  finding, keeping what was already found.

### Added

- Compressed tars are scanned: `.tgz`, `.tbz2`, `.txz`, `.tar.gz`, `.tar.bz2`
  and `.tar.xz`, now that the walk is bounded. The pre-commit hook's file
  pattern includes them, and `hayward hub` fetches them.

### Changed

- Constructors of plain values (`int`, `float`, `bool`, `str`, `bytes`,
  `range`, in both the `builtins` and `__builtin__` spellings) are on the
  known-safe list, like `list`, `dict` and `set` already were. They no longer
  add an INFO "unrecognized global" to ordinary files. `getattr`, `type` and
  `eval` stay off the list.
- The package's own comments and docstrings describe attacks in words instead
  of quoting payloads, so other scanners stop flagging Hayward itself (#17). A
  test keeps it that way.

## 1.5.0 (2026-10-30)

### Added

- `hayward hub org/name` fetches a Hugging Face repo's scannable files at a pinned
  commit and scans them, so a repo can be checked before it is downloaded into a
  project. It uses the network; `hayward scan` still makes no network calls. A
  big zip checkpoint is read by Range request for its pickle member only, so a multi-gigabyte `.pt` or `.bin` is checked by fetching a few
  kilobytes; the run says the tensor data was not read. Other files over
  `--max-file-mb` are listed as NOT SCANNED with the reason, never as clean. File
  names from the API are checked, Hub hashes are verified, and `HF_TOKEN` goes
  only to `huggingface.co`.

- `hayward lock org/name` pins a repo's commit and every file's hash for the
  `transformers` `from_pretrained` profile, and `hayward lock --verify` checks a
  downloaded snapshot against it offline (missing, changed and extra files fail).
  It fails closed: code loaded from another repo, or an adapter's base model, is
  recorded as unresolved and the lock is marked `needs_review`.

- `hayward explain FILE` shows what a pickle would do, as readable lines, one per
  call, with a "why" line that uses the scanner's own allow and deny lists, so
  `explain` and `scan` agree. Nothing is run. Known-safe calls and plain object
  rebuilds are counted on one line unless `--all` is given. Reads bare pickles,
  zip checkpoints and old tar checkpoints; names from the file are shown escaped.

### Fixed

- `distutils.spawn.spawn` (and `setuptools._distutils.spawn`, where `distutils`
  resolves on Python 3.12+) is now denied. It runs a program, and a public
  scanner-bypass proof of concept called it with a list argument that no argument
  check reads as a command, so it came out as INFO. The reference corpus had the
  same file labelled benign; it is now labelled malicious.

## 1.4.3 (2026-10-29)

### Fixed

- Binary data in a pickle no longer comes out HIGH (`MFV-PICKLE-006`). A
  protocol 2 pickle stores a bytes literal as `_codecs.encode(text, 'latin1')`,
  and for binary data that text is random characters that matched the
  command-shape check by chance (a backtick, a `;` beside a space). Resnet and
  zipformer checkpoints were affected. The text is now turned back into bytes
  first, so the printable-ASCII gate that already applies to other bytes
  arguments applies here too. Readable command text still flags. As with
  protocol 3 bytes, a payload with a control character added is not matched.

### Changed

- `MFV-GGUF-001` (a `.gguf` file with no GGUF magic) is now INFO and counts as a
  coverage gap: **not a clean verdict**, no longer a HIGH that fails
  `--fail-on high`. A pickle, zip or other dangerous format wearing a `.gguf`
  name is still `MFV-CONFUSE-001` at HIGH and is scanned as what it is.
  Unverified bytes are reported as unverified, not as a threat.

## 1.4.2 (2026-10-28)

### Changed

- PyPI summary, keywords and the README opening now name the formats Hayward
  scans (PyTorch, pickle, joblib, GGUF, SafeTensors, ONNX, Keras), so the
  package is easier to find. No scanner or rule changes.

## 1.4.1 (2026-10-23)

### Added

- `server.json` for the official MCP Registry, listing `hayward mcp` as
  `io.github.hedgerow-dev/hayward`. Clients start it with
  `uvx --with "mcp>=2.3,<3" hayward mcp`, so the core install keeps its one
  dependency.

### Changed

- The PyPI page carries the current README: the load guard and MCP demo
  sections.

## 1.4.0 (2026-10-22)

### Added

- **Load guard.** `import hayward.guard` guards `pickle.load`, `joblib.load`,
  `numpy.load(allow_pickle=True)` and `torch.load`: each file is scanned
  first, and while the loader runs every global it resolves on that thread is
  checked before import. A refusal raises `UnsafeModelError`. Other
  unpickling in the process, such as `pickle.loads` and multiprocessing, is
  untouched. The default mode raises the bar but is not a sandbox;
  `install(strict=True)` refuses every global not on the known-safe list and
  is the setting for untrusted files. `mode="warn"`, `max_size`,
  `prescan=False`, `uninstall()` and `HAYWARD_GUARD=off` are supported. The
  usage guide lists what it does not cover.
- The `dev` extra includes numpy and joblib again, so CI runs the guard's
  tests for those loaders.

### Changed

- The known-safe list now includes standard-library value types that model
  metadata carries: `datetime` dates, times and timezones, `decimal.Decimal`,
  `uuid.UUID`, `fractions.Fraction` and `pathlib` path classes, plus the
  Python 2 spellings protocols 0 and 1 write for plain instances. Path
  classes no longer count a filesystem path argument as evidence.
- transformers objects are on the known-safe list by exact name only:
  `TrainingArguments` and the four `trainer_utils` enums Trainer pickles.
  Other class-shaped transformers names are no longer trusted by shape.

## 1.2.7 (2026-10-22)

### Fixed (security)

- **Legacy tar members read through links.** A `pickle`, `storages`,
  `tensors` or `sys_info` member that is a symlink, a hard link or of an
  unknown type is read the way torch.load reads it. One that resolves to
  nothing readable, or is larger than the member cap, reports a coverage
  gap; an oversized one still has its start scanned.
- **Oversized tar checkpoints.** Past the 500MB cap, a file that torch.load
  reads as a tar is walked as a tar, also when a zip is appended to it.
- **Mixed lzma and xz streams.** After the first stream, each one is decoded
  in either format, as `lzma.LZMAFile` does. A later stream that does not
  decode reports `MFV-SKIP-003` instead of ending the read silently.
- **`.npy` headers read as numpy reads them.** `descr` spellings such as
  `'object'` and structured fields count as object dtypes, Python 2 `L`
  integers in the header are accepted, and a header that does not parse
  reports `MFV-SKIP-003`.
- **joblib arrays inside the pickle.** The scanner skips the array data joblib
  writes after each array, as joblib.load does, and scans the rest of the
  pickle. An array it cannot size reports `MFV-SKIP-003`.
- **Python 2 names.** Below protocol 3, globals are classified under the
  Python 3 names pickle maps them to (`cPickle.loads` as `pickle.loads`).

## 1.2.6 (2026-10-21)

Two new ways to run the scanner, and scanner fixes for files the loaders
read differently from their names. No rule changes.

### Added

- **pre-commit hook.** `.pre-commit-hooks.yaml` with hook id `hayward`. Its
  file pattern is generated from the scanner's own extension list, and a test
  fails if the two drift.
- **MCP server for coding agents.** `hayward mcp` runs a stdio server with one
  read-only tool, `scan_path`, returning `blocked`, `clean` or `incomplete`.
  Needs the new optional extra: `pip install "hayward[mcp]"`. The core
  install still has one dependency.

### Changed

- The README states the coverage guarantee precisely: bytes that match no
  model format from the first byte report nothing.

### Fixed (security)

The loaders decide what a file is by its bytes. The scanner sometimes decided
by its name, and passed content clean that a loader would run.

- **Pickle-family files are scanned whatever they hold.** A protocol 0 or 1
  pickle named `.pth` is scanned (path-configuration files still are not). A
  gzip, bz2, xz or lzma pickle under any pickle extension is decompressed and
  scanned, as zlib already was. lzma is detected on the same two bytes joblib
  checks, and joblib's pre-0.10 `ZF` layout is read. An lz4 joblib file, or a
  compressed stream that will not decompress, reports `MFV-SKIP-003`.
  Concatenated compressed streams are read through to the last one, as
  joblib's bz2, xz and lzma readers do. A
  compressed pickle under a pickle extension (`joblib.dump` to `model.pkl`
  with `compress=`) is scanned as joblib and no longer reports
  `MFV-CONFUSE-001`.
- **Legacy tar checkpoints.** A `.pt` (or other pickle extension) that is an
  uncompressed tar is walked as torch.load reads it, with `storages`,
  `tensors`, `pickle` and `sys_info` scanned whatever byte they open with,
  including when a zip is appended to the tar.
- **Bytes in front of a zip.** A pickle followed by a zip archive is scanned
  as a pickle too, since the loaders read it from byte 0. This holds past the
  500MB cap.
- **NumPy by content.** An object `.npy` named `.npz` is scanned, and so is
  an `.npz` member that lacks the `.npy` suffix. A zip under a NumPy name that
  is not all arrays, such as a torch checkpoint, reports `MFV-CONFUSE-001`
  and is scanned as one.
- **Pickles under protobuf names.** A pickle, compressed pickle, zip or `.npy`
  named `.onnx`, or any `.pb` other than `saved_model.pb`, is scanned. A zip
  named `.pb` is scanned past the size cap too.
## 1.2.5 (2026-10-03)

- Calibrate `MFV-TORCH-001` as a LOW source-presence signal. Distinguish
  TorchScript graph representations from Python modules in torch.package;
  unrelated `.py` files beside a checkpoint no longer establish either format.
- Add `MFV-TORCH-002` for statically resolved execution operations in packaged
  Python. Import aliases are supported; shadowed and unresolved calls are not
  convicted. Source is parsed, never imported or executed. Reachability and
  malicious intent are not claimed.
- Bound Python source inspection by per-member, total-byte, and member-count
  budgets. Unreadable, invalid, or over-budget source produces `MFV-SKIP-002`.
  Existing pickle and archive-structure findings remain independent.

## 1.2.4 (2026-10-02)

Resource and fail-closed hardening, ported from the model scanner that Rowan
carried before it switched to depending on Hayward. No rule changes.

### Fixed

- **7z archives are bounded by what extraction actually writes.** A listing's
  sizes are claims; extraction is now watched and stopped once a member passes
  `MAX_ZIP_MEMBER_BYTES`, the total passes 500MB, or it writes more than 1,000
  files (`MFV-SKIP-003`, `skipped_reason: extraction_quota`). The tree is
  measured every 50ms and after exit, so a fast disk can overshoot by one
  interval's writes. Extractor stderr no longer goes to an unread pipe, which
  let a noisy archive stall the scan for the full 300-second timeout.
- **7z archives fail closed on a bad listing.** An archive over 500MB is never
  handed to the extractor, and a listing that fails, overflows 2MB, names more
  than 1,000 members or claims an oversized member is reported as not scanned
  instead of being extracted. A non-numeric `Size =` line still counts as zero;
  the quota above now bounds it.
- **`.keras` `config.json` reads are capped.** The member was decompressed in
  full: a 408KB archive expanding to 400MB drove memory to 1.4GB. Oversized
  configs are reported as `MFV-SKIP-003` (`skipped_reason: oversized`).
- **TFLite layout checks are bounded.** Aliased subgraph offsets could replay
  one inflated tensor table thousands of times: a 64KB file took 14 seconds. The
  replay now stops at a file-wide budget and reports `MFV-TFLITE-001`.

## 1.2.3 (2026-08-28)

A scaled comparative benchmark (128 real public models, seven formats) surfaced
two issues; both are fixed here.

### Fixed

- Big-endian GGUF no longer false-positives. The `GGUF` magic is byte-order
  independent, but the version and every count after it are stored big-endian on
  big-endian builds (llama.cpp ships these for s390x). Read little-endian, the
  u64 counts looked like wrapped arithmetic and reported `MFV-GGUF-005` at HIGH
  on a valid public model. Big-endian is now recognised and reported as a
  coverage gap (`MFV-GGUF-004`, INFO), not a verdict, the same as v1 and GGML.

### Added

- Plain `.tar` archives are now scanned with the same tar walker that backs
  `.nemo`, so a traversing member name (`MFV-ARCHIVE-001`) or a traversing
  symlink (`MFV-ARCHIVE-002`) is reported instead of passing silently. A `.tar`
  that is not actually a tar (a renamed pickle, say) still falls through to the
  usual content sniff, so the rename trick stays caught. Compressed tars
  (`.tgz`, `.tar.gz`) are deliberately not routed yet: that needs a
  decompression budget on the member walk first.

## 1.2.2 (2026-08-28)

### Fixed

- GGUF version 1 no longer false-positives. v1 stores its header counts and
  lengths as 32-bit fields, while v2 and v3 use 64-bit, so reading a valid v1
  model with the v2/v3 layout decoded the counts into absurd values and
  reported `MFV-GGUF-005` at HIGH. v1 is now recognised and reported as a
  coverage gap (`MFV-GGUF-004`, INFO), not a verdict. Surfaced by a comparative
  benchmark against a real public model.

## 1.2.1 (2026-08-27)

### Fixed

- Shorten the GitHub Action's `action.yml` description to under the 125
  characters the Marketplace requires. No scanner change; identical to 1.2.0.

## 1.2.0 (2026-08-27)

Archive-safety detections and a re-measurement. This release adds two rules for
crafted containers and closes a deny-list gadget a reproducible measurement of
1.1.0 surfaced.

### Added

- `MFV-ARCHIVE-001` (MEDIUM, CWE-22): flags an archive member whose name is
  unsafe as an extraction path, across the containers Hayward walks but never
  extracts. A member named `../../x`, an absolute path, a Windows drive path, a
  UNC path, or one carrying a NUL/newline escapes the target directory when a
  loader or unpacker writes it to disk (zip slip). The check now runs on the
  torch zip (`.pt`), the nested zip inside a TorchServe `.mar`, and the tar
  based NVIDIA `.nemo`; numpy's `.npz` was already covered by `MFV-NPZ-001`. A
  normal nested member such as `archive/data.pkl` does not fire.
- `MFV-ARCHIVE-002` (MEDIUM, CWE-22/59): flags an archive member that is a
  symlink or hard link whose target escapes the extraction directory. Distinct
  from `MFV-ARCHIVE-001`, which flags an unsafe member name: here the member is
  the link. On extraction the loader creates the link, then a later member
  written through it lands wherever the link points (the classic tar/zip
  symlink attack, the CVE-2007-4559 family). Covers the tar-based `.nemo` (tar
  sym/hard links) and the zip containers `.pt`/`.mar` (a member with the
  `S_IFLNK` unix mode, its body the link target). A link to a safe in-tree
  target such as `weights/shard1` does not fire.

### Fixed

- Deny `uuid._get_command_stdout` and `uuid._popen`, living-off-the-land
  gadgets that shell out through subprocess (GHSA-g38g-8gr9-h9xp). They were
  surfaced only at INFO before. The `uuid` module itself stays allowed, so an
  ordinary pickled `uuid.UUID` is not flagged.

## 1.1.0 (2026-08-27)

A hardening and feature release built from the in-depth review of 2026-08-23.
The scanner's entire input surface is adversarial, so the largest part of this
work makes the walk uncrashable, unhangable and un-OOM-able on hostile files,
since a crash is itself an evasion: one crafted file first in a directory used
to abort the whole scan.

### Removed (public API)

- `ModelFileFinding` is no longer exported from `hayward`. It was an unused
  dataclass that was never constructed. Removing a public name is why this is a
  minor version bump rather than a patch. `Finding` is unchanged and remains the
  single record type a scan returns.
- `Finding.start_line` is gone. It was always `0` and never serialized.

### Added

- **SARIF 2.1.0 output** (`-f sarif`), the format GitHub code scanning and other
  CI platforms ingest, plus a `schema_version` field in the JSON report.
- **Git LFS pointer detection** (`MFV-LFS-001`, INFO coverage): a pointer file
  is a placeholder for content stored elsewhere, so the real bytes were never
  fetched and nothing was scanned. Reported as a coverage gap, never a clean
  verdict.
- `--color {auto,always,never}` and `NO_COLOR` support. `--no-colour` stays as
  an alias.
- A `py.typed` marker, so the package's type hints are now visible to type
  checkers (the `Typing :: Typed` classifier is no longer a false claim).
- **HuggingFace / transformers JSON config detection.** `MFV-HF-001` flags a
  code-execution construct in a `.json` config's Jinja `chat_template` (the
  `MFV-GGUF-003` threat in a standalone file), and `MFV-HF-002` flags `auto_map`
  and `trust_remote_code: true`, the HuggingFace-hub remote-code vector.
- **Executable source inside torch archives** (`MFV-TORCH-001`): a torch zip
  (`.pt` / `.ptl` / TorchScript / torch.package) that carries `code/*.py` or a
  `.data/` package layout runs that Python on load. `.ptl` mobile checkpoints
  are now scanned like any other torch zip.
- **CycloneDX 1.6 ML-BOM output** (`-f cyclonedx`): one component per scanned
  file, one vulnerability per finding, for tools that consume a bill of
  materials.
- **`--allowlist`**: suppress reviewed findings through an auditable JSON file,
  each entry keyed by the file's sha256 and rule id and requiring a
  justification, so a changed file resurfaces the finding and every suppression
  is announced on stderr.
- **`--baseline`**: compare a scan against a prior JSON report and apply
  `--fail-on` only to findings that are new, for adopting the scanner on an
  existing tree without failing on the whole backlog.
- **`--check-signatures`** (`MFV-SIG-001`, INFO): report sibling signature and
  attestation artifacts (a detached `.sig`, a Sigstore bundle, an
  in-toto/SLSA/DSSE attestation). Detection and structural reporting only, never
  cryptographic verification.
- **`--policy`**: remap per-rule severities through a small JSON file, without
  editing the curated rule set.
- **`--cache`**: a content-hash scan cache so a re-run skips unchanged files,
  invalidated by the package version and by the flags that change a verdict.
- **Directory-scan ergonomics**: several targets at once, `--exclude` globs,
  `--jobs N` for parallel scanning, `--max-size` to tune the per-file cap, and
  `--progress` / `--verbose` / `--quiet` (all on stderr, so a piped report stays
  clean).
- **A composite GitHub Action** that runs the scan and writes SARIF for
  `github/codeql-action/upload-sarif`, and a reproducible Docker image plus a
  committed browser-demo build pipeline.

### Fixed

- **Scanner self-DoS hardening.** The simulated pickle memo and operand stack
  are bounded; the walk degrades to an `MFV-SKIP-003` coverage finding instead
  of running out of memory on a crafted stream. Embedded-pickle opcode
  iteration is lazy, the embedded-executable search has a per-format occurrence
  budget, and a `.npy` v2 header length is capped before `ast.literal_eval`.
- **Per-file exception firewall.** An unexpected failure inside `scan_file` now
  degrades to `MFV-SKIP-003` for that one file instead of aborting the whole
  directory scan. `KeyboardInterrupt` and `SystemExit` still propagate.
- **`MFV-PICKLE-006` severity inversion.** A file mixing evidence tiers now
  reports the most severe, not the least.
- **Keras decoy evasion.** Config extraction walks past a benign decoy object to
  find a Lambda layer hidden behind it, rather than stopping at the first
  `"class_name"` anchor.
- **Zip declared-size gate evasion.** A pickle member is sniffed and read from
  its real bytes even when the central directory lies about its size; the
  decompression-bomb caps on the read path are unchanged.
- A crash inside the scanner now exits `2`, distinct from `1` (findings at or
  above the threshold), and `-f text -o FILE` writes plain text to the file
  rather than JSON.
- Assorted correctness fixes to the pickle walk (junk `STACK_GLOBAL` operands,
  unhashable container elements, opener-tuple hygiene) and word-boundary
  matching for SafeTensors `__metadata__` keys, so ordinary keys like
  `evaluation_metric` no longer false-positive at CRITICAL.
- **Evasion gaps closed.** The post-splice resync now also finds protocol-0/1
  payloads, not just protocol 2 to 5; the embedded-pickle walk covers
  `BYTEARRAY8` literals, recurses two levels, and propagates argument-level
  evidence, not just denied names; and GGUF metadata past the content-scan
  window now reports a coverage gap (`MFV-GGUF-004`) instead of silence. The raw
  zip-member fallback emits the zip-bomb finding at parity with the strict path
  rather than silently truncating.

### Internal

- The single large `scanner.py` was split into focused modules behind a facade:
  `_pickle_engine`, `_gguf`, `_tensors`, `_keras`, `_binary` and `_lfs`.
  `hayward.scanner` re-exports every name, so the public and library API is
  unchanged; this is a code move with no behavior change.

## 1.0.1 (2026-08-08)

### Fixed

- `scan_file` no longer returns an empty result for a path whose extension is
  not on the supported list. A file named directly is identified by its
  content, so a malicious pickle renamed `danger.dat`, or carrying no
  extension at all, is now reported exactly as the same bytes named
  `danger.pkl` were. `scan_directory` still discovers candidates by
  extension, and [coverage](docs/coverage.md) states that limit.

- A pickle stream that ends before its `STOP` opcode now reports
  `MFV-SKIP-003`. It previously produced nothing at all, so a payload placed
  past the cut was invisible and a partially downloaded checkpoint was
  indistinguishable from one that had been read and found clean.

- Duplicate member names in an `.npz` are compared on the raw stored name.
  `zipfile` rewrites a backslash to a forward slash while reading the central
  directory, so on Windows two members differing only by separator arrived
  identical and `MFV-NPZ-001` never fired. The same archive was flagged
  correctly on Linux and macOS.

## 1.0.0 (2026-08-06)

First release. The scanner was developed inside a larger static-analysis tool
and is published here as a standalone package, with its own command line,
desktop window and test suite.

### Formats

Pickle and every container that carries one (PyTorch zip and legacy layouts,
joblib behind zlib, gzip, bz2, lzma or xz, NumPy `.npy` and `.npz`, TorchServe
`.mar`, NVIDIA NeMo `.nemo`, skops), plus SafeTensors, GGUF, Keras H5 and
`.keras`, ONNX, TensorFlow SavedModel, TFLite and PMML. Format is decided by
magic bytes, and a file whose extension disagrees with its content is
reported.

### Detection

- Pickle streams are walked as opcodes with `pickletools`. Nothing is
  imported, deserialised or executed.
- An unknown callable is judged by the arguments it was resolved with, not by
  its name: a URL, a shell command, a `(host, port)` pair or a literal naming
  a denied function each promote it to an actionable finding. This is what
  catches gadgets that are on no list.
- A second pickle stream carried as a bytes literal is scanned too
  (`MFV-PICKLE-008`): `numpy.load(BytesIO(<pickle>))` is the pattern, where
  the outer callable is on no deny list and the payload exists only once the
  inner bytes are read.
- Dotted `GLOBAL` names are resolved segment by segment, the way
  `Unpickler.find_class` resolves them. `GLOBAL "torch" "serialization.os.system"`
  reaches `os.system` on load, but a deny list keyed on the joined string sees
  only `torch.serialization.os.system` and files it as unknown. The shape
  appears in PickleBall's published corpus.
- Container arithmetic is replayed against the file for GGUF, SafeTensors and
  TFLite, covering the integer-overflow class behind CVE-2025-53630,
  CVE-2026-27940, CVE-2026-33298 and CVE-2026-42627.
- Tensor names are validated as paths wherever tooling would write them to
  one, covering traversal segments, absolute and drive-absolute paths, and
  embedded NUL or newline. Part of `MFV-ST-006` and `MFV-GGUF-005`.
- ONNX `external_data` is checked for keys beyond the four the format
  defines (CVE-2026-34445) and for locations that are absolute, traversing,
  or off the filesystem entirely. A published proof of concept points a
  location at `169.254.169.254`, so loading the model reaches the cloud
  metadata endpoint (`MFV-ONNX-004`).
- PMML is checked for external entity declarations, where the read primitive
  needs no code execution at all.
- Embedded PE, ELF and Mach-O images are reported. No serialisation format
  writes one.

### Coverage reporting

Every failure to read a file produces a finding rather than silence.
`MFV-SKIP-001`, `MFV-SKIP-002`, `MFV-SKIP-003`, `MFV-7Z-001` and
`MFV-GGUF-004` all mean analysis did not complete, and
`hayward.is_coverage_gap()` identifies them.

This closes the exception-oriented evasion described in
[arXiv 2508.19774](https://arxiv.org/abs/2508.19774), where a payload is
placed behind a deliberate parse error so the scanner reports nothing while
the target framework loads the file anyway. Both the raising and the silent
variants are handled: a container walk that ends early is detected by checking
that it consumed the whole file, since iteration ending is not proof the file
ended.

### Interfaces

- `hayward scan` with text and JSON output, a configurable `--fail-on`
  threshold, and `--fail-on-coverage` for builds that should not pass on an
  unread file.
- `hayward-gui`, a single tkinter window. No added dependency.
- `ModelFileScanner.scan_file` and `.scan_directory` as the library API.

### Measured position

Measured against public and in-the-wild corpora (hash-pinned HuggingFace
models, picklescan's test data, PickleCloak, MalHug): clean on the benign
models and a strong detection rate on the malicious ones, with the misses
(such as a `.7z` archive with no extractor installed) documented.

All self-measured against corpora that do not ship in this repository. Not
independently reproducible; see docs/accuracy.md, and run the harness
(`scripts/eval_corpus.py`) against your own corpus.
