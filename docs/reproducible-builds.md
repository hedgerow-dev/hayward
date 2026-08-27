# Reproducible builds

Hayward ships three artifacts from one source tree: the wheel on PyPI, a
container image, and the static in-browser demo. All three are reproducible.
The version comes from a single place, `__version__` in
`hayward/__init__.py`, which hatchling reads at build time (`pyproject.toml`
sets `[tool.hatch.version] path = "hayward/__init__.py"`). There is no second
copy to keep in sync.

## The wheel

The package builds with hatchling through the standard build frontend:

```bash
pip install build
python -m build            # writes dist/hayward-<version>-py3-none-any.whl and .tar.gz
```

The wheel is pure Python (`py3-none-any`) with one runtime dependency,
`defusedxml`. hatchling builds are deterministic and honour
`SOURCE_DATE_EPOCH`, so set it if you need bit-identical wheels across
machines and dates:

```bash
SOURCE_DATE_EPOCH=$(git log -1 --format=%ct) python -m build
```

## The container image

The `Dockerfile` at the repo root installs the package from the local source
and runs the CLI as an unprivileged user.

```bash
docker build -t hayward:local .
```

Scan a file by mounting it in read-only. The image's entrypoint is `hayward`,
so you pass the subcommand and arguments directly:

```bash
docker run --rm -v "$PWD:/work:ro" hayward:local scan /work/model.pt
docker run --rm -v "$PWD/models:/work:ro" hayward:local scan /work -f json
```

Reproducibility notes:

- The base is pinned by tag (`python:3.12-slim`). A released image should also
  pin by digest so a moved tag cannot change the base under a fixed release.
  Get the digest with `docker buildx imagetools inspect python:3.12-slim` and
  append it: `FROM python:3.12-slim@sha256:<digest>`.
- `.dockerignore` trims the build context to what hatchling needs, so tests,
  caches, virtualenvs, and prior build output never enter the image or perturb
  its layers.
- The image runs as uid 10001, never root, and needs no write access to its
  own install.

## The in-browser demo

The published demo is a static page. It loads Pyodide from a pinned CDN, uses
micropip to install Hayward's own wheel plus defusedxml, and runs the real
scanner in the browser. No server-side Python is involved, which is why a plain
static host (GitHub Pages, a static Hugging Face Space) can serve it.

Build it with the committed pipeline:

```bash
pip install build
python scripts/build_browser_demo.py            # builds into dist/browser-demo
```

That builds the wheel, copies it into the output directory, and generates
`index.html` wired to install and call it. Serve the directory over HTTP
(micropip cannot fetch the wheel over `file://`):

```bash
python -m http.server --directory dist/browser-demo 8000
```

Reproducibility notes:

- Two versions are pinned as constants at the top of
  `scripts/build_browser_demo.py`: `PYODIDE_VERSION` and `DEFUSEDXML_PIN`.
  Bump them deliberately; they define the demo.
- The generated `index.html` embeds no timestamps, hostnames, or other
  run-specific data, so it is byte-for-byte identical given the same source and
  the same pinned versions. Combined with a reproducible wheel, the whole
  output directory reproduces exactly.
- The page calls the real public API. It writes the picked file into an
  in-browser filesystem and runs
  `ModelFileScanner().scan_file(path)`, reading `rule_id`, `severity`,
  `message`, and `category` off each returned `Finding`. Nothing is uploaded.
