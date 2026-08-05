# Changelog

Notable changes, newest first. Versions follow [semantic versioning](https://semver.org).
Rule identifiers are part of the public interface and will not be renamed
within a major version.

## 1.0.0 (2026-08-05)

First release. The scanner was developed inside a larger static-analysis tool
and is published here as a standalone package, with its own command line,
desktop window and test suite.

### Formats

Pickle and every container that carries one (PyTorch zip and legacy layouts,
joblib, NumPy `.npy` and `.npz`, TorchServe `.mar`, NVIDIA NeMo `.nemo`,
skops), plus SafeTensors, GGUF, Keras H5 and `.keras`, ONNX, TensorFlow
SavedModel, TFLite and PMML. Format is decided by magic bytes, and a file
whose extension disagrees with its content is reported.

### Detection

- Pickle streams are walked as opcodes with `pickletools`. Nothing is
  imported, deserialised or executed.
- An unknown callable is judged by the arguments it was resolved with, not by
  its name: a URL, a shell command, a `(host, port)` pair or a literal naming
  a denied function each promote it to an actionable finding. This is what
  catches gadgets that are on no list.
- Container arithmetic is replayed against the file for GGUF, SafeTensors and
  TFLite, covering the integer-overflow class behind CVE-2025-53630,
  CVE-2026-27940, CVE-2026-33298 and CVE-2026-42627.
- ONNX `external_data` is checked against the four-key contract
  (CVE-2026-34445) and for absolute or traversing locations.
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

Against 215 hash-pinned models from the HuggingFace Hub: nothing above INFO,
5 files in the INFO tier. Against picklescan's test corpus: 34 of 35 malicious
files, the miss being a `.7z` archive with no extractor installed. Against
PickleCloak: 49 of 57 exploits and 91 of 97 gadget chains. Against MalHug:
87 of 87 read and detected, 86 naming the sink the corpus records.

All self-measured against corpora that do not ship in this repository, using
a harness that is not published yet. Not reproducible by a reader today; see
docs/accuracy.md.
