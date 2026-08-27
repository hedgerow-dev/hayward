# Changelog

Notable changes, newest first. Versions follow [semantic versioning](https://semver.org).
Rule identifiers are part of the public interface and will not be renamed
within a major version.

## Unreleased

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
