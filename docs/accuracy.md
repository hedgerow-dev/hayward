# Accuracy

**Every number on this page is self-measured.** One section, the reproducible
1.1.0 detection below, can be re-run by anyone against a public corpus. The rest
cannot yet.

Those other corpora live in a separate harness, quickset, which scores five
scanners against the same files. **That harness is not public yet**, so those
figures cannot currently be reproduced by a reader. Until it is, treat them as
a vendor claim, which is exactly the standard this page applies to everyone
else's numbers.

What ships here that you *can* check today: the test suite, which builds its
own fixtures and needs no network, and the scanner itself.

The 1.1.0 detection figures in the next section **can** be re-run against a
public corpus. The figures after that were measured 2026-08-06 at version 1.0.0,
have not been re-run since, and describe that release rather than the current
one.

## Reproducible detection (1.1.0)

Unlike the rest of this page, this you can re-run. Measured 2026-08-27 against
`hayward 1.1.0` installed from PyPI, over the public picklescan test fixtures at
a pinned commit (`f15d54da3dec9aa28a87ede82f87882bb80f1023`), with labels taken
from picklescan's own test assertions.

- **Malicious: 80 of 83 detected at HIGH or above** (all 80 as CRITICAL), and
  **0 of 83 returned a clean verdict.**
- The three not raised to HIGH each still produced a finding. `malicious1.7z` is
  a coverage gap (no bundled `7z` extractor). Two are stdlib
  living-off-the-land gadgets surfaced at INFO rather than convicted:
  `sys_module_override_sploit.pkl` (attribute-override obfuscation) and one
  `uuid._get_command_stdout` variant whose five siblings *are* caught at
  CRITICAL. Both are genuine gaps, tracked for a later release.
- **Benign: 14 of 14 picklescan benign fixtures clean**, including a real 244 KB
  legacy-pickle torch `state_dict` and six numpy arrays.

The benign result is a smoke check, not a false-positive rate: 14 files is far
too few to state one. A real rate needs the larger corpus this page's other
numbers hint at.

Reproduce it:

```bash
pip install hayward==1.1.0
git clone --depth 1 https://github.com/mmaitre314/picklescan
hayward scan picklescan/tests/data --fail-on high
```

## False positives

215 real models from the HuggingFace Hub, pinned by SHA-256, across 14 formats
and 64 publishers.

| Threshold | False positives |
|---|---|
| Above INFO | 0 of 215 |
| Including INFO | 5 of 215 |

Both are true. They differ only in whether the unknown tier counts, and that
is the single largest lever on any scanner's false-positive rate. Whenever you
quote one of these, say which.

The five INFO findings are two sklearn Cython loss classes, an attribute
lookup on an object rebuilt by an allow-listed callable, a Keras
functional-API marker, and a
user-defined transformer in `__main__` from scikit-learn's own documentation.
That last one is the scanner being correct: a class defined in `__main__` is
exactly what static analysis cannot verify.

## Detection

| Corpus | Result |
|---|---|
| picklescan `tests/data`, 35 malicious | 34 of 35 |
| PickleCloak exploits | 49 of 57 |
| PickleCloak gadget chains | 91 of 97 |
| MalHug, 87 fetchable in-the-wild models | 87 read, 87 detected, 86 naming the correct sink |

## How to read those numbers

**PickleCloak has no benign half.** A scanner that flags every file scores
100% on it. Two of the scanners measured alongside Hayward do exactly that,
and they also flag more than 40% of real models. A detection figure from an
all-malicious corpus is meaningless without a false-positive figure from a
real one.

**The picklescan corpus tests a list Hayward inherited.** Matching it closely
mostly measures that the list was inherited. Strip every picklescan-derived
entry from the deny data and Hayward drops to 25 of 35 there, below
picklescan's own 34. On PickleCloak the same stripped build stays ahead, at
42 of 57 against picklescan's 27, which is where judging callables by their
arguments shows up. See [how it works](how-it-works.md).

**The one miss on picklescan's corpus is `malicious1.7z`**, on a machine with
no `7z` extractor installed. It is reported as a coverage gap rather than
passing silently. See [coverage](coverage.md).

**215 benign models is a small sample.** A zero-event result over 215 trials
gives a 95% upper bound near 1.4%, not zero. The corpus is small enough that
one unlucky format family would move it.

**51 of the 215 carry HuggingFace's own `caution` label.** All 51 were
disassembled at the opcode level and contain no code-execution path; the label
reflects imports outside HuggingFace's allowlist, not a malware finding. That
audit was also ours.

**No one else publishes the last figure.** Naming the sink the corpus
records, in 86 of 87 MalHug detections, has nothing to be compared against,
so treat it as unopposed rather than as a win.

## Prior art

Hayward is not the first tool here.

[picklescan](https://github.com/mmaitre314/picklescan) is wired into the
HuggingFace Hub and its deny-list data is the foundation this builds on, with
attribution in [NOTICE](../NOTICE).
[ModelScan](https://github.com/protectai/modelscan) is the open-source tier of
Protect AI's Guardian, which scans the Hub at a scale nothing here approaches.
[fickling](https://github.com/trailofbits/fickling) is a better tool than this
one for interactive analysis and says plainly that it is not built to gate a
build. [ModelAudit](https://github.com/promptfoo/promptfoo) registers more
formats than Hayward does and arrived at an allowlist-first design
independently, for the same reason.

Published corpora worth knowing about, several of which are used above:
PickleBench (ShadowPickle), PickleBall, SafePickle, MalHug and PickleCloak.

What Hayward adds is argument-evidence promotion and coverage as a reported
quantity. Not breadth, and not a longer list.

## Reproducing this

**The 1.1.0 detection section above is reproducible today** against the public
picklescan corpus, with the exact commands it lists. The full false-positive
and comparative figures are **not** yet: the quickset harness is not published,
and saying so is more useful than a command that does not work.

When it is published, the shape is: the benign corpus is fetched from the Hub
and hash-pinned on first use, and malicious payloads are generated at run time
and never stored, so nothing malicious ships in either repository.

What you can run now:

```bash
pip install -e ".[dev]" && pytest      # the full suite, no network
hayward scan ./your-own-models         # your files, your numbers
```

The second is worth more than our table. If Hayward is noisy on your models,
that is a bug report we want.
