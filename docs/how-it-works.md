# How it works

## Nothing is executed

Pickle streams are walked as opcodes with `pickletools`, simulating the stack
and the memo (pickle's object cache). No `pickle.load`, no `pickle.loads`, no `__import__`, no
`importlib`, no `eval`. Containers are read member by member. XML goes through
`defusedxml`, so entity expansion is refused rather than performed.

Format is decided by magic bytes. A file whose extension and content disagree
is reported as such, because that mismatch is a known evasion rather than an
accident.

## The deny list is not the interesting part

Every pickle scanner starts with a list of dangerous callables: `os.system`,
`subprocess.Popen`, `builtins.eval`. Hayward has one too, and part of it is
inherited from picklescan with attribution in [NOTICE](../NOTICE).

Deny lists lose. The attacker needs one function nobody listed, and automated
gadget mining keeps finding them in the standard library. Meanwhile every
callable that is on neither the allow list nor the deny list lands in an
"unknown" tier that nobody triages, which is where the real bypasses live.

## Argument-evidence promotion

So instead of asking what a callable is called, Hayward asks what it was
handed.

The opcode walk resolves each call's arguments to concrete values wherever the
stream makes that possible. An unknown callable is then promoted out of the
unknown tier when its arguments convict it:

| Evidence in the resolved arguments | Result |
|---|---|
| A URL | HIGH |
| A shell-shaped command string | HIGH |
| A `(host, port)` pair | HIGH |
| A literal naming a function that *is* denied | HIGH |
| A denied attribute name | MEDIUM |
| A filesystem path | LOW |

`torch.utils.collect_env.run` is on no deny list. Handed `curl ... | sh`, it
does not need to be.

The severity follows the strength of the evidence, never the callable's name.
That is what generalises to gadgets nobody has catalogued, and it is why the
unknown tier stays small without the deny list doing all of the work.

The same logic runs in reverse. An allow-listed callable handed an argument no
legitimate model would supply is also promoted, because being on the allow
list is not a licence to accept anything.

## Structural checks

Code execution is not the only way a model file hurts you. Native loaders
allocate and copy from numbers in the file's header, and those numbers are
attacker-controlled.

Hayward replays the container arithmetic and reports when it does not agree
with the file:

- **GGUF**: tensor and metadata counts, key lengths, dimensions multiplied
  without wrapping, offsets inside the file. This is the class behind
  CVE-2025-53630, CVE-2026-27940 and CVE-2026-33298, where an integer overflow
  produces an undersized allocation that a later read overruns.
- **SafeTensors**: every tensor's declared span against its shape and dtype,
  against the data section, and against its neighbours.
- **TFLite**: tensor dimensions that a 32-bit loader cannot allocate
  (CVE-2026-42627).

A file can be entirely free of executable content and still corrupt the heap
of the thing that opens it.

## Format-specific checks

- **ONNX**: custom ops with documented execution behaviour (`PyOp`,
  `PythonOp`), `external_data` entries carrying keys beyond the four the
  format defines (CVE-2026-34445), and locations that are absolute or
  traverse upward.
- **Keras**: Lambda layers, which embed a marshalled Python function, and
  layer classes that are not built in.
- **TensorFlow**: GraphDef ops that read or write the filesystem or invoke
  embedded Python.
- **PMML**: external entity declarations, where the file-read and SSRF
  primitive needs no code execution at all, and functions associated with
  evaluation.
- **skops**: schema type references checked against the same classifier the
  pickle path uses, plus the two loader invariants behind CVE-2025-54412 and
  CVE-2025-54413.
- **Embedded executables**: PE, ELF and Mach-O magic anywhere in a model file.
  No serialisation format writes one.

## Reading what it could not read

Every failure to parse produces a finding rather than silence. That is a
design position rather than an implementation detail, and it has its own page:
[coverage](coverage.md).
