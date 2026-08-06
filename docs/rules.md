# Rule catalogue

Every rule Hayward can emit, with its severity, CWE mapping and trigger
condition. Rule identifiers are stable: they are what you put in a
suppression, a ticket or a dashboard, so they do not get renamed.

Hayward dispatches on magic bytes, walks the container, and simulates the
pickle virtual machine without importing or executing anything it reads.

Some tools write tensors to disk under the names the file gives them, so
Hayward checks tensor names as filesystem paths. A traversal segment or an
embedded newline in a tensor name is not something a training run produces.

Severity is fixed per rule except `MFV-PICKLE-005` and `MFV-PICKLE-006`, which
derive theirs from the strength of the evidence found.

## Pickle and pickle-bearing formats

| Rule | Severity | CWE | Fires when |
|------|----------|-----|------------|
| `MFV-PICKLE-001` | CRITICAL | 502 | References a callable that grants code or command execution on load |
| `MFV-PICKLE-002` | HIGH | 502 | Stream will not parse as opcodes but carries suspicious byte patterns |
| `MFV-PICKLE-003` | HIGH | - | Member exceeds the decompressed-size limit, possible zip bomb |
| `MFV-PICKLE-004` | INFO | 502 | References a global on neither the allow nor the deny list |
| `MFV-PICKLE-005` | HIGH/MEDIUM/LOW | 502 | An unknown callable's **resolved arguments** convict it: a URL, a shell-shaped string, a `(host, port)` pair, or a literal naming a denied global. Severity follows the strength of the signal, never the callable's name |
| `MFV-PICKLE-006` | HIGH/MEDIUM | 502 | An allow-listed callable is handed an argument no legitimate model would supply |
| `MFV-PICKLE-007` | HIGH | 502 | Memo slots sit in a band far from the rest of the stream, the signature of a spliced-in pickle |
| `MFV-PICKLE-008` | CRITICAL | 502 | A **second pickle stream is carried as a bytes literal** and references a denied callable. `numpy.load(BytesIO(<pickle>))` is the shape: the outer callable is on no deny list and the payload exists only once the inner bytes are read |
| `MFV-JOBLIB-002` | HIGH | - | joblib's compressed payload exceeds the decompressed-size limit |

## SafeTensors

| Rule | Severity | CWE | Fires when |
|------|----------|-----|------------|
| `MFV-ST-001` | HIGH | - | File is under 8 bytes |
| `MFV-ST-002` | HIGH | - | Declared header size exceeds the limit |
| `MFV-ST-003` | HIGH | - | Header extends past the end of the file |
| `MFV-ST-004` | HIGH | - | Header is not valid JSON |
| `MFV-ST-005` | CRITICAL | - | Metadata carries keys that could attempt execution on load |
| `MFV-ST-006` | HIGH | 787 | Offsets, shapes and dtypes do not agree with the file (loaders allocate and `memcpy` from these numbers), **or a tensor name carries a path**: traversal segment, absolute or drive-absolute path, embedded NUL or newline |

## GGUF

| Rule | Severity | CWE | Fires when |
|------|----------|-----|------------|
| `MFV-GGUF-001` | HIGH | - | Magic number missing or invalid |
| `MFV-GGUF-002` | CRITICAL | - | Metadata carries suspicious content |
| `MFV-GGUF-003` | CRITICAL | 94, 1336 | `tokenizer.chat_template` contains a code-execution construct in Jinja2 syntax |
| `MFV-GGUF-004` | INFO | - | Magic is valid but the KV section will not parse. **Not a clean verdict**, see coverage limits |
| `MFV-GGUF-005` | HIGH | 787, 190 | Container arithmetic wraps or overruns the file (the `CVE-2025-53630` / `CVE-2026-27940` / `CVE-2026-33298` class), **or a tensor name carries a path** |

## Other formats

| Rule | Severity | CWE | Fires when |
|------|----------|-----|------------|
| `MFV-KERAS-001` | HIGH | 502 | Lambda layer embedding a marshalled Python function |
| `MFV-KERAS-002` | INFO | 502 | Layer class not on the known-builtin list |
| `MFV-ONNX-001` | HIGH | 502, 94 | Custom op with documented code-execution behaviour (PyOp/PythonOp) |
| `MFV-ONNX-002` | MEDIUM | 22 | Path string with `..` traversal, or an absolute `external_data` location |
| `MFV-ONNX-003` | HIGH | 502 | `external_data` key beyond the four the format defines. `CVE-2026-34445` |
| `MFV-ONNX-004` | HIGH | 918 | `external_data` location points off the filesystem (a URL, UNC or protocol-relative reference). The loader fetches what `location` names, so loading the model issues a request. A published proof of concept points one at `169.254.169.254`, the cloud metadata endpoint |
| `MFV-TF-001` | HIGH | 502, 94 | SavedModel graph op that touches the filesystem or invokes embedded Python |
| `MFV-TFLITE-001` | HIGH | 190, 125 | Tensor dimensions inconsistent with a 32-bit loader. `CVE-2026-42627` |
| `MFV-NPZ-001` | MEDIUM | 22 | NPZ member violates zip-safety discipline (traversal, absolute path, duplicate name) |
| `MFV-PMML-001` | HIGH | 611, 918 | External entity declaration. XXE needs no code execution to read files or reach the network |
| `MFV-PMML-002` | HIGH | 94 | Function name associated with code execution in `<Apply>` |
| `MFV-SKOPS-001` | CRITICAL | 502 | skops schema references an execution-granting type |
| `MFV-SKOPS-002` | CRITICAL | 502 | skops loader state contradicts the loader's own contract |
| `MFV-SKOPS-003` | INFO | 502 | skops schema carries something unverifiable from the schema alone |
| `MFV-SKOPS-004` | LOW | 502 | skops archive embeds a pickle stream, defeating the format's premise |
| `MFV-SKOPS-005` | INFO | - | `.skops` extension with no `schema.json` |

## Structural and coverage rules

| Rule | Severity | CWE | Fires when |
|------|----------|-----|------------|
| `MFV-EXEC-001` | HIGH | 506 | A loadable binary (PE, ELF, Mach-O) is embedded. No serialisation format writes one |
| `MFV-CONFUSE-001` | HIGH | - | Extension implies one format, magic bytes say another |
| `MFV-SKIP-001` | LOW | - | File exceeds the 500MB cap and cannot be streamed (ZIP and Keras HDF5 can) |
| `MFV-SKIP-002` | LOW | - | A tar walk ended early, so members past the failure were never analysed |
| `MFV-SKIP-003` | LOW | - | Content could not be verified |
| `MFV-7Z-001` | INFO | - | A `.7z` archive is present and no extractor is available |

## Coverage limits, stated plainly

**`MFV-SKIP-001/002/003`, `MFV-7Z-001` and `MFV-GGUF-004` are not clean
verdicts.** Each means some or all of the file was never analysed. Treat them as
"unknown", never as "safe", and do not count them as detections when scoring.

Known blind spots as of 2026-08-05:

- **Files over 500MB are read only when the format allows it without loading
  the whole file.** ZIP containers are streamed member by member, and HDF5 is
  streamed to its `model_config` attribute, so oversized `.pt` and Keras `.h5`
  models are both scanned. Any other format over the cap reports
  `MFV-SKIP-001` and is not read.
- **`.7z` archives need a system `7zz` or `7z` on PATH.** py7zr is LGPL-2.1 and
  cannot be bundled. With an extractor present the archive is unpacked to a
  temporary directory and every member scanned; without one, `MFV-7Z-001`
  reports the gap. One malicious sample in picklescan's own suite is
  missed on a machine with no extractor.
- **Nested archives are scanned to a depth of 4.** Beyond that `MFV-SKIP-003`
  reports the gap rather than recursing.
- **`MFV-PICKLE-004` is the unknown-global bucket**, structurally the same tier
  as picklescan's `suspicious` and ModelAudit's `warning`. It is INFO by design.
  Any published false-positive figure must state whether INFO was counted.

