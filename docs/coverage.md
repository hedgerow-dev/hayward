# Coverage

Most scanners answer one question: did I find something. Hayward also answers
a second: did I actually read the file. The second is where the failures hide.

**A clean report should mean the file was examined and found clean, not that
the parser gave up.**

## The attack this defends against

Published research ([arXiv 2508.19774](https://arxiv.org/abs/2508.19774),
"The Art of Hide and Seek") describes placing a payload behind a deliberate
parse error. The scanner's exception handler fires, the scanner reports
nothing, and the target framework, which is more permissive, loads the file
anyway.

The malformed file is *quieter* than the plain one. The paper reports 22
pickle-bearing model-loading paths across five frameworks, 19 of them missed
by existing scanners, and nine instances of the technique, seven of which
bypassed every scanner tested.

Hayward was vulnerable to this. Eighteen of its forty exception handlers
swallowed errors into a clean verdict, and one discarded findings it had
already made, so a payload plus a malformed member scanned cleaner than the
payload alone. Fixing that meant one rule applied everywhere: **no parse
attempt may lower suspicion or erase prior evidence.**

There is a quieter variant still. A tar truncated inside a member header ends
iteration with no exception at all, so no handler can catch it. Hayward
detects that by checking the walk consumed the whole file, because iteration
ending is not proof the file ended.

## The coverage rules

| Rule | Means |
|---|---|
| `MFV-SKIP-001` | Over the size cap and not in a streamable format |
| `MFV-SKIP-002` | A container walk ended early; members past that point were never seen |
| `MFV-SKIP-003` | Content could not be verified |
| `MFV-7Z-001` | A `.7z` archive with no extractor available |
| `MFV-GGUF-004` | Valid GGUF magic, unparseable metadata section |

**None of these is a clean verdict, and none is a detection.** Treat them as
unknown, and count them in a column of their own when scoring.

```python
from hayward import is_coverage_gap

unread = [f for f in findings if is_coverage_gap(f)]
```

```bash
hayward scan ./models --fail-on-coverage   # exit 1 on any unread file
hayward scan ./models -f json | jq '.coverage_gaps'
```

## Known limits

Stated rather than hidden, as of version 1.0.0.

**Files over 500 MB** are read only when the format allows it without loading
the whole file into memory. ZIP containers are streamed member by member, and
HDF5 is streamed to its `model_config` attribute, so oversized PyTorch and
Keras models are both scanned. Anything else over the cap reports
`MFV-SKIP-001`, including a named file whose extension is not recognised:
padding a payload past the cap and renaming it is an evasion, so the gap is
reported rather than passed over.

This limit used to be absolute, and it cost real detections: two 553 MB Keras
models in the MalHug corpus carry genuine malicious Lambda layers and were
never read. Both are now found by streaming to the config, holding a bounded
window in memory rather than the whole file.

**Directory scans find candidates by extension.** A file handed to
`scan_file` directly, on the command line or through the API, is identified by
its content whatever it is called, including with no extension at all. A
directory walk cannot do the same without reading every file in the tree, so
`scan_directory` globs the 24 supported extensions plus `.bin` and `.zip`. A
malicious pickle named `payload.dat` is reported when named, and not found by
a scan of the directory holding it. Unpack an archive of unknown provenance
and scan the members by name if this matters to you.

**`.7z` archives** need a system `7zz` or `7z` on `PATH`. py7zr is LGPL-2.1
and is not bundled. With an extractor present the archive is unpacked to a
temporary directory and every member scanned. Without one, `MFV-7Z-001`
reports the gap, and one malicious sample in picklescan's test corpus is
missed.

**Nested archives** are followed to a depth of four. Deeper nesting reports
`MFV-SKIP-003` rather than recursing.

**Truncation detection can misfire on opaque bytes.** A pickle stream that
ends before its `STOP` opcode reports `MFV-SKIP-003`. Deciding that means
walking opcodes, and opcode decoding succeeds on far more byte sequences than
are actually pickles: roughly one buffer in fifty that happens to open with a
`PROTO` marker decodes all the way to the end and is reported as truncated.

Reaching that on a real file needs two coincidences at once, a segment
boundary landing on those two bytes and the content past it decoding cleanly.
It has not been observed: **zero of 286 real Hub models and zero of 395 files
across the picklescan, MalHug and PickleCloak corpora produce this finding.**
The same measurement says the rule is unexercised by those corpora, so the
only evidence it works is the test suite.

The cost is a `LOW` coverage finding rather than a detection, so it does not
fail a build at the default threshold. Demanding more evidence before
reporting was tried and it lost genuine truncations. For a rule whose entire
job is to speak up when the file was not read, a false alarm is the cheaper
error.

**`MFV-PICKLE-004` is the unknown bucket**, structurally the same tier as
picklescan's `suspicious` and ModelAudit's `warning`. It is INFO by design,
and it is not a coverage gap: the file was read, and something in it was not
recognised.

## Why this is worth reporting

We ran five scanners over the same 254 files, 215 of them real models from
the HuggingFace Hub. Two returned no verdict on more than half the corpus
while reporting no findings for those files: one on 160 of the 254, the other
on 142. A third returned no verdict on 37. Only two of the five returned a
verdict on every file.

None of that is visible unless coverage is reported as a number. That is the
argument for making it one.
