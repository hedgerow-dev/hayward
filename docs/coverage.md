# Coverage

Most scanners answer one question: did I find something. Hayward answers two,
because the second is where the failures hide.

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
`MFV-SKIP-001`.

This limit used to be absolute, and it cost real detections: two 553 MB Keras
models in the MalHug corpus carry genuine malicious Lambda layers and were
never read. Both are now found in about ten milliseconds, at roughly 67 MB of
peak memory rather than 553 MB.

**`.7z` archives** need a system `7zz` or `7z` on `PATH`. py7zr is LGPL-2.1
and is not bundled. With an extractor present the archive is unpacked to a
temporary directory and every member scanned. Without one, `MFV-7Z-001`
reports the gap, and one malicious sample in picklescan's test corpus is
missed.

**Nested archives** are followed to a depth of four. Deeper nesting reports
`MFV-SKIP-003` rather than recursing.

**`MFV-PICKLE-004` is the unknown bucket**, structurally the same tier as
picklescan's `suspicious` and ModelAudit's `warning`. It is INFO by design,
and it is not a coverage gap: the file was read, and something in it was not
recognised.

## Why this is worth reporting

Measured across five scanners on the same 215 real models, three returned no
verdict at all on more than half the corpus while reporting no findings for
those files. One silently read zero files out of 130. Another emitted "could
not parse" on 95 files while printing a clean summary.

None of that is visible unless coverage is reported as a number. That is the
argument for making it one.
