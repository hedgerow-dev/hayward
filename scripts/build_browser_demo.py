#!/usr/bin/env python3
"""Build the static, in-browser Hayward demo.

The published demo is a static page: it loads Pyodide from a pinned CDN, uses
micropip to install Hayward's own wheel plus defusedxml, and runs the real
scanner in the browser. There is no server-side Python, which is why a plain
static host (a GitHub Pages site, a static Hugging Face Space) can serve it.

What this script does, in order:
  1. Reads the version from hayward/__init__.py (the single source of truth).
  2. Builds the wheel with `python -m build --wheel` (needs the `build`
     package: `pip install build`).
  3. Copies that wheel into the output directory.
  4. Generates index.html next to it, wired to install and call the wheel.

Reproducibility: the output is byte-for-byte identical given the same source
tree and the same pinned versions below (PYODIDE_VERSION, DEFUSEDXML_PIN). The
generated HTML embeds no timestamps, hostnames, or other run-specific data.
The wheel itself is reproducible because hatchling builds are deterministic and
SOURCE_DATE_EPOCH-aware; set SOURCE_DATE_EPOCH if you need bit-identical wheels
across machines.

Run it:
  python scripts/build_browser_demo.py                 # builds into dist/browser-demo
  python scripts/build_browser_demo.py --out site      # choose the output dir
  python scripts/build_browser_demo.py --skip-build     # reuse the wheel in dist/

Then serve the output over HTTP (micropip cannot fetch the wheel over file://):
  python -m http.server --directory dist/browser-demo 8000
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

# --- Pinned versions. Bump these deliberately; they define the demo. ---------
# Pyodide runtime, loaded from the jsDelivr CDN. Ships CPython 3.12.
PYODIDE_VERSION = "0.26.4"
# defusedxml is Hayward's only runtime dependency. Pinned so the browser
# resolves the same wheel every time rather than "whatever is latest on PyPI".
DEFUSEDXML_PIN = "0.7.1"

REPO_ROOT = Path(__file__).resolve().parent.parent
INIT_PY = REPO_ROOT / "hayward" / "__init__.py"


def read_version() -> str:
    """Parse __version__ out of hayward/__init__.py without importing it."""
    text = INIT_PY.read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if match is None:
        sys.exit(f"could not find __version__ in {INIT_PY}")
    return match.group(1)


def build_wheel() -> None:
    """Build the wheel into dist/ with the standard build frontend."""
    print("building wheel with `python -m build --wheel` ...")
    try:
        subprocess.run(
            [sys.executable, "-m", "build", "--wheel", str(REPO_ROOT)],
            check=True,
        )
    except FileNotFoundError:
        sys.exit("the `build` package is required: pip install build")
    except subprocess.CalledProcessError as exc:
        sys.exit(f"wheel build failed (exit {exc.returncode})")


def find_wheel(version: str) -> Path:
    """Locate the built wheel for this version in dist/."""
    wheel = REPO_ROOT / "dist" / f"hayward-{version}-py3-none-any.whl"
    if not wheel.is_file():
        sys.exit(
            f"expected wheel not found: {wheel}\n"
            f"run without --skip-build, or build it first with `python -m build --wheel`."
        )
    return wheel


def render_index_html(wheel_name: str, version: str) -> str:
    """Return the static demo page, wired to the given wheel filename.

    The page installs the wheel and defusedxml with micropip, then calls the
    real public API: `ModelFileScanner().scan_file(path)`, which returns
    Finding objects carrying rule_id, severity, message, category, confidence.
    """
    template = _INDEX_TEMPLATE
    return (
        template.replace("__PYODIDE_VERSION__", PYODIDE_VERSION)
        .replace("__DEFUSEDXML_PIN__", DEFUSEDXML_PIN)
        .replace("__WHEEL_NAME__", wheel_name)
        .replace("__HAYWARD_VERSION__", version)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the static in-browser Hayward demo.")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "dist" / "browser-demo",
        help="output directory (default: dist/browser-demo)",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="reuse the existing wheel in dist/ instead of rebuilding it",
    )
    args = parser.parse_args(argv)

    version = read_version()
    print(f"hayward version: {version}")

    if not args.skip_build:
        build_wheel()
    wheel = find_wheel(version)

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    shutil.copy2(wheel, out / wheel.name)
    (out / "index.html").write_text(
        render_index_html(wheel.name, version), encoding="utf-8"
    )

    print(f"wrote {out / 'index.html'}")
    print(f"wrote {out / wheel.name}")
    print(
        "\nserve it over HTTP (file:// will not work for micropip):\n"
        f"  python -m http.server --directory {out} 8000"
    )
    return 0


# The demo page. Placeholder tokens (__PYODIDE_VERSION__, __DEFUSEDXML_PIN__,
# __WHEEL_NAME__, __HAYWARD_VERSION__) are substituted by render_index_html so
# the CSS and JS braces below need no escaping.
_INDEX_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Hayward - in-browser model file scanner</title>
<style>
  :root { color-scheme: light dark; }
  body {
    font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    max-width: 760px; margin: 0 auto; padding: 2rem 1.25rem; line-height: 1.5;
  }
  h1 { font-size: 1.5rem; margin-bottom: 0.25rem; }
  .sub { opacity: 0.7; margin-top: 0; }
  .card {
    border: 1px solid rgba(128,128,128,0.35); border-radius: 10px;
    padding: 1rem 1.25rem; margin: 1.25rem 0;
  }
  #status { font-variant-numeric: tabular-nums; opacity: 0.8; }
  input[type=file] { display: block; margin: 0.5rem 0; }
  button {
    font: inherit; padding: 0.5rem 1rem; border-radius: 8px;
    border: 1px solid rgba(128,128,128,0.5); background: transparent; cursor: pointer;
  }
  button:disabled { opacity: 0.5; cursor: default; }
  table { border-collapse: collapse; width: 100%; margin-top: 0.75rem; }
  th, td { text-align: left; padding: 0.4rem 0.6rem; border-bottom: 1px solid rgba(128,128,128,0.25); vertical-align: top; }
  th { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.03em; opacity: 0.7; }
  .sev { font-weight: 600; white-space: nowrap; }
  .sev-critical { color: #c026d3; }
  .sev-high { color: #dc2626; }
  .sev-medium { color: #d97706; }
  .sev-low { color: #2563eb; }
  .sev-info { opacity: 0.7; }
  code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.9em; }
  .clean { color: #16a34a; font-weight: 600; }
  footer { margin-top: 2rem; font-size: 0.85rem; opacity: 0.7; }
</style>
</head>
<body>
  <h1>Hayward</h1>
  <p class="sub">Scans machine-learning model files for code execution and unsafe content, without loading them. Running in your browser, version __HAYWARD_VERSION__.</p>

  <div class="card">
    <p id="status">Loading the scanner (Pyodide + wheel). This takes a moment on first load.</p>
    <input type="file" id="file" disabled />
    <button id="scan" disabled>Scan file</button>
  </div>

  <div id="results"></div>

  <footer>
    Nothing you select is uploaded. The file is read into an in-browser
    filesystem and scanned locally by the same Hayward package published on
    PyPI. Pyodide __PYODIDE_VERSION__, defusedxml __DEFUSEDXML_PIN__.
  </footer>

  <script src="https://cdn.jsdelivr.net/pyodide/v__PYODIDE_VERSION__/full/pyodide.js"></script>
  <script>
  // Python glue: defines run_scan(path) -> JSON string of findings. It calls
  // the real public API and reads the fields the Finding dataclass exposes.
  const PY_GLUE = `
import json
from pathlib import Path
from hayward import ModelFileScanner

_scanner = ModelFileScanner()

def run_scan(path):
    findings = _scanner.scan_file(Path(path))
    return json.dumps([
        {
            "rule_id": f.rule_id,
            "severity": f.severity.value,
            "message": f.message,
            "category": f.category.value,
            "confidence": f.confidence,
        }
        for f in findings
    ])
`;

  const statusEl = document.getElementById("status");
  const fileEl = document.getElementById("file");
  const scanBtn = document.getElementById("scan");
  const resultsEl = document.getElementById("results");

  let pyodide = null;
  let runScan = null;

  async function init() {
    pyodide = await loadPyodide();
    statusEl.textContent = "Installing Hayward and defusedxml with micropip ...";
    await pyodide.loadPackage("micropip");
    // Install the pinned dependency and the wheel served next to this page.
    await pyodide.runPythonAsync(
      'import micropip\\n' +
      'await micropip.install(["defusedxml==__DEFUSEDXML_PIN__", "./__WHEEL_NAME__"])'
    );
    await pyodide.runPythonAsync(PY_GLUE);
    runScan = pyodide.globals.get("run_scan");
    statusEl.textContent = "Ready. Pick a model file to scan.";
    fileEl.disabled = false;
    scanBtn.disabled = false;
  }

  function severityRank(sev) {
    return { critical: 0, high: 1, medium: 2, low: 3, info: 4 }[sev] ?? 5;
  }

  function render(findings) {
    if (findings.length === 0) {
      resultsEl.innerHTML = '<div class="card"><span class="clean">No findings.</span> Nothing at or above a reportable tier was detected.</div>';
      return;
    }
    findings.sort((a, b) => severityRank(a.severity) - severityRank(b.severity));
    let rows = "";
    for (const f of findings) {
      const sev = f.severity;
      rows +=
        '<tr>' +
        '<td class="sev sev-' + sev + '">' + sev + '</td>' +
        '<td><code>' + escapeHtml(f.rule_id) + '</code></td>' +
        '<td>' + escapeHtml(f.message) + '</td>' +
        '</tr>';
    }
    resultsEl.innerHTML =
      '<div class="card"><table><thead><tr>' +
      '<th>Severity</th><th>Rule</th><th>Message</th>' +
      '</tr></thead><tbody>' + rows + '</tbody></table></div>';
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  async function onScan() {
    const file = fileEl.files && fileEl.files[0];
    if (!file) return;
    scanBtn.disabled = true;
    resultsEl.innerHTML = "";
    statusEl.textContent = 'Scanning "' + file.name + '" ...';
    try {
      // Keep the original extension: the scanner dispatches on it.
      const name = file.name.split(/[\\\\/]/).pop() || "model.bin";
      const path = "/tmp/" + name;
      const bytes = new Uint8Array(await file.arrayBuffer());
      pyodide.FS.writeFile(path, bytes);
      const json = runScan(path);
      render(JSON.parse(json));
      statusEl.textContent = "Done. Pick another file to scan again.";
    } catch (err) {
      statusEl.textContent = "Scan failed: " + err;
    } finally {
      scanBtn.disabled = false;
    }
  }

  scanBtn.addEventListener("click", onScan);
  init().catch((err) => {
    statusEl.textContent = "Failed to load the scanner: " + err;
  });
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
