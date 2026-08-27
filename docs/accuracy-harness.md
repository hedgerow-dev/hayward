# Accuracy harness

`scripts/eval_corpus.py` scores Hayward against a corpus **you** supply. Give
it a directory of sample files and a manifest that labels each one malicious or
benign, and it runs the scanner over every sample and reports what it caught,
what it missed, and what it flagged that it should not have.

It measures only the corpus you hand it. **No corpus ships in this repository**,
and the harness invents no numbers of its own. The open decision about whether
this project can publish a reproducible corpus is at the bottom of this page.

## Running it

```bash
python scripts/eval_corpus.py CORPUS_DIR MANIFEST --threshold high
```

`CORPUS_DIR` holds the sample files. `MANIFEST` labels them. Sample paths in the
manifest are resolved relative to `CORPUS_DIR` unless they are absolute. The
report prints to stdout, or to a file with `-o report.txt`.

Check the wiring with no corpus at all:

```bash
python scripts/eval_corpus.py --selftest
```

That runs the built-in checks of the classification and metric arithmetic. It
imports nothing from the `hayward` package, so it works on a bare checkout
before anything is built.

## The manifest

Two formats, chosen by file extension.

**JSON** (`.json`). A list of records:

```json
[
  {"path": "malicious/nested_reduce.pkl", "expected": "malicious",
   "rule": "MFV-PICKLE-REDUCE-001", "sink": "os.system"},
  {"path": "benign/resnet50.safetensors", "expected": "benign"}
]
```

`path` and `expected` are required. `expected` is `malicious` or `benign`.
`rule` and `sink` are optional: they say which rule id, or which sink token, a
correct detection ought to name, so the harness can report not just that a
malicious file was caught but that it was caught for the right reason.

JSON also accepts an object with a `"samples"` list, or a flat mapping of
`path -> "benign"/"malicious"` for the simple case with no per-sample reasons.

**CSV** (`.csv`). A header row, then one sample per line:

```csv
path,expected,rule,sink
malicious/nested_reduce.pkl,malicious,MFV-PICKLE-REDUCE-001,os.system
benign/resnet50.safetensors,benign,,
```

## What the report contains

* **Counts.** True positives (malicious, detected), false negatives (malicious,
  missed), true negatives (benign, clean), false positives (benign, flagged).
* **Rates.** Precision, recall, and false-positive rate, computed from those
  counts. They describe the supplied corpus at the supplied threshold and
  nothing else.
* **Per-rule hit counts.** How many findings each rule id produced, with
  coverage rules marked.
* **Detection reason.** Where the manifest named a rule or sink, how many
  detections named it correctly.
* **Misses.** Every malicious sample that scored below the threshold, with its
  highest real severity, so a near-miss is legible.
* **False positives.** Every benign sample flagged at or above the threshold,
  with the rules that fired.
* **Coverage gaps.** Samples the scanner could not fully read (see below).

## The threshold is the biggest lever

`--threshold` sets the lowest severity that counts as a detection. It defaults
to `high`. The report states the threshold and whether it counts the INFO tier,
because that one choice moves the numbers more than anything else in the tool.

INFO is Hayward's "unfamiliar, not dangerous" bucket. A malicious sample that
fires only INFO is a **miss** under `--threshold high` and a **detection** under
`--threshold info`. A benign sample that fires only INFO is a clean **true
negative** under `high` and a **false positive** under `info`. Both readings are
honest. They answer different questions, so a detection or false-positive figure
means nothing unless the threshold is quoted with it. This is the same point
`docs/accuracy.md` makes about every scanner's numbers, including everyone
else's.

## Coverage gaps are not verdicts

A file the scanner could not fully read (a Git LFS pointer, a 7z archive with no
extractor on the machine, a file that broke the parser) is neither a detection
nor a clean verdict. The harness puts those samples in a **coverage bucket** of
their own and keeps them out of the true/false positive and negative counts.

So recall here is "of the malicious files it actually read, how many did it
catch", not "of all malicious files". The coverage counts sit beside it for the
rest. This mirrors how Hayward treats coverage everywhere else; see
`docs/coverage.md`. If a malicious sample lands in the coverage bucket rather
than in true positives, the fix is usually to install the missing extractor or
fetch the LFS content, not to change the threshold.

## The blocking decision (maintainer)

**The flagship accuracy figures Hayward has quoted are self-measured against
corpora that do not ship in this repository.** `docs/accuracy.md` says so
plainly: the numbers came from a separate, unpublished harness, and until that
harness and its corpora are public, a reader cannot reproduce a single figure.

This harness is the runnable half of the answer. The missing half is a
decision only the maintainer can make: **does this project own an accuracy
corpus it is allowed to redistribute, and will it publish one?** The samples in
question are third-party model files (real models from the Hub for the benign
side) and malicious pickles (some derived from other projects' corpora). Whether
they can be redistributed is a licensing question, not a technical one, and it
gates any reproducible number this project publishes.

Three options, so the decision is concrete:

1. **Publish an owned corpus.** Assemble a benign and malicious set this project
   holds the rights to redistribute, ship it (or a hash-pinned fetch of it), and
   point this harness at it. Highest effort, but the only path to numbers a
   reader can reproduce from this repo alone.

2. **Point at third-party corpora with a fetch script.** Do not redistribute
   anything. Ship a manifest plus a script that fetches the public corpora named
   in `docs/accuracy.md` (PickleBench, PickleBall, SafePickle, MalHug,
   PickleCloak) into `CORPUS_DIR`, respecting each corpus's own license. Lower
   effort, but reproducibility depends on those third parties staying reachable
   and on their licenses permitting the use.

3. **Keep the numbers qualitative.** Ship the harness (this) so anyone can score
   their own files, and stop quoting specific figures the reader cannot check.
   Least effort, and the most defensible under the project's "no unverifiable
   numbers" rule, at the cost of a headline detection figure.

Until this is decided, the harness runs against a corpus if one is provided, and
this repository publishes no accuracy figure of its own.
