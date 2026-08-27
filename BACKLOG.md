# Hayward backlog

Generated from the in-depth review of 2026-08-23 (engine, architecture, tests,
feature-gap analysis). Items are grouped by workstream; IDs are stable so they
can be referenced in commits and PRs.

Priority legend: **P0** = critical (scanner DoS / verified bug), **P1** = major,
**P2** = minor, **P3** = feature/enhancement.

---

## WS-1 — Scanner self-DoS hardening (P0)

A scanner whose entire input surface is adversarial must not be crashable,
hangable, or OOM-able. Crashes are also evasion: one crashing file first in a
directory aborts the whole scan.

### HW-101 — Cap the simulated pickle memo dict and stack
- **Where:** `hayward/scanner.py:1044-1063` (`_walk_one_pickle`)
- **Problem:** `memo_indices` is capped at 1M but the `memo` dict itself
  (`memo[arg] = stack[-1]`) and the operand stack grow without bound. 500 MB of
  `N\x94` (NONE+MEMOIZE) = 250M dict entries → multi-GB OOM, fully inside the
  default 500 MB scan cap.
- **Acceptance:** bounded memory on crafted streams; walk degrades gracefully
  (skip finding or frozen memo) instead of OOM; no behaviour change on
  well-formed files; regression test with monkeypatched/small caps.

### HW-102 — Lazy opcode iteration in `_embedded_pickle_denied_globals`
- **Where:** `hayward/scanner.py:1666`
- **Problem:** `ops = list(pickletools.genops(...))` materialises the entire
  opcode list — 500 MB of 1-byte opcodes → ~500M tuples → tens of GB. Runs on
  every pickle file before the main walk.
- **Acceptance:** streaming iteration, same detection semantics, memory O(1) in
  opcode count.

### HW-103 — Occurrence budget in `_find_embedded_executables`
- **Where:** `hayward/scanner.py:2027-2064`
- **Problem:** PE/ELF/Mach-O candidate loops iterate once per magic occurrence
  regardless of the validated-hit cap (cap stops on *validated* hits only).
  500 MB of `MZMZ…` ≈ a minute of pure-Python loop. Verified.
- **Acceptance:** hard iteration/occurrence budget per format; on budget
  exhaustion stop searching (optionally emit a coverage finding); timing test.

### HW-104 — Per-file exception firewall + complete catch sets
- **Where:** `scan_file` entry (`hayward/scanner.py:3170+`); json catch sets at
  `scanner.py:3982, 2681, 3527`; `.npy` header at `scanner.py:5442-5451`;
  7z listing parse at `scanner.py:5316-5320`.
- **Problem:** (a) any unhandled exception aborts the entire directory scan —
  crash-as-evasion; (b) `json.loads` on a deeply nested header raises
  `RecursionError`/`MemoryError`, caught nowhere (the skops path at 4836
  already catches them — apply consistently); (c) `.npy` v2 `header_len` is a
  u32 (≤4 GB) handed to `ast.literal_eval` uncapped; (d) non-numeric `Size=`
  line in a 7z listing raises `ValueError`.
- **Acceptance:** `scan_file` never raises for non-OSError input problems —
  degrade to `MFV-SKIP-003`; literal_eval input capped (~1 MB; real headers are
  ~100 bytes); regression tests for each vector.

---

## WS-2 — Correctness / evasion bugs (P0–P1)

### HW-110 — Severity inversion in MFV-PICKLE-006 (P0, verified)
- **Where:** `scanner.py:3764-3768`
- **Problem:** `worst = max(..., key=index into _PICKLE_UNKNOWN_TIERS)` indexes
  in insertion order `[HIGH, MEDIUM, LOW]`, so `max` returns the **lowest**
  severity present. Verified: worst of {HIGH, LOW} → LOW.
- **Acceptance:** use `severity_order`; regression test mixing tiers.

### HW-111 — Keras config extraction tries only the first anchor (P0, verified evasion)
- **Where:** `_extract_keras_model_config`, `scanner.py:2638-2684`
- **Problem:** only the first `"class_name"` anchor is examined; a benign decoy
  object before the real config hides a Lambda layer. Attacker controls HDF5
  attribute order/content.
- **Acceptance:** iterate anchors (bounded, e.g. ≤64) until risky layers found
  or anchors exhausted; decoy regression test.

### HW-112 — Zip declared-size gate evasion (P1)
- **Where:** `scanner.py:4375` (`_zip_member_may_be_pickle`), `4579`
  (`_scan_nested_zip_member`), raw fallback `4416-4418`
- **Problem:** gating on attacker-controlled `info.file_size`/`compress_size`
  before reading a byte. Inflate the central-directory size → pickle member is
  never sniffed, read, or reported.
- **Acceptance:** sniff/read head bytes regardless of declared size (read path
  stays capped); regression test with lying central directory.

### HW-113 — Resync is blind to protocol 0/1 pickles (P1)
- **Where:** `scanner.py:624-626, 785-793`
- **Problem:** after a walk-killing splice (joblib raw arrays), recovery only
  finds `\x80\x02..\x80\x05` markers. A proto-0/1 payload after the splice is
  never found.
- **Acceptance:** heuristic proto-0 resync (GLOBAL `c<mod>\n<name>\n` patterns)
  or, if infeasible without false positives, an explicit coverage finding for
  the unread tail; documented either way.

### HW-114 — Embedded-pickle walk: one level, bytes-only, denied-only (P1)
- **Where:** `_embedded_pickle_denied_globals`, `scanner.py:1650-1676`
- **Problem:** (a) `BYTEARRAY8` literals not inspected; (b) inner stream walk
  doesn't recurse — two-level nesting is invisible; (c) only *denied* globals
  propagate, unknown-bucket argument evidence is dropped.
- **Acceptance:** BYTEARRAY8 covered; ≥2 nesting levels (bounded); unknown-tier
  evidence propagates; regression tests.

### HW-115 — GGUF metadata content checks stop at 10 MB (P1)
- **Where:** `scanner.py:1908, 2501-2528` (`GGUF_METADATA_SCAN_BYTES`)
- **Problem:** a malicious `tokenizer.chat_template` or dangerous-substring key
  placed after 10 MB of earlier KV entries is silently never checked (no
  coverage finding either). Layout pass walks the whole file; content pass
  doesn't.
- **Acceptance:** content checks cover the full metadata section with bounded
  per-string inspection budgets; or a coverage finding past the window.

### HW-116 — Unbounded `read_bytes()` fallback in `_scan_pytorch_zip` (P1)
- **Where:** `scanner.py:4662-4665`, reachable from the >500 MB path at
  `3196-3197`
- **Problem:** if `zipfile.ZipFile` raises, fallback reads the whole
  (multi-GB) file into memory despite the cap that exists to prevent exactly
  that.
- **Acceptance:** capped/streaming read; oversized path handled like other
  oversized files.

### HW-117 — Minor correctness batch (P2)
- STACK_GLOBAL with non-string operands builds junk refs → bogus INFO unknowns
  (`scanner.py:1110-1121`): push opaque, skip recording.
- FROZENSET/DICT raise `TypeError` on unhashable elements while SETITEM/
  SETITEMS suppress it (`scanner.py:1006-1034`): make consistent (attacker can
  kill the walk at a chosen point).
- safetensors `__metadata__` keys matched by bare substrings (`"exec"`,
  `"eval"`, `"import"`) at CRITICAL (`scanner.py:4017`): word-boundary
  matching (`evaluation_metric`, `import_date` currently false-positive).
- `_PICKLE_OPENERS` contains duplicate `b"("`/`b"\x28"` and single-byte `b"c"`
  over-triggers (`scanner.py:4353-4356`); `_NESTED_PICKLE_OPENERS` includes
  FRAME (`\x95`, never an opener) and omits proto-0/1 openers (`scanner.py:1628`).
- `_archive_depth` is a class attribute mutated through instances
  (`scanner.py:5258`) — make it instance state.
- Raw deflate fallback silently returns truncated 200 MB prefix instead of the
  zip-bomb finding the strict path emits (`scanner.py:4428`).

---

## WS-3 — CLI / report / packaging / docs fixes (P1–P2)

### HW-105 — Crashes must exit 2, not 1 (P1)
- **Where:** `hayward/cli.py:121`
- **Problem:** unhandled non-OSError exception → Python exits 1,
  indistinguishable from "findings at/above threshold". SECURITY.md declares
  crashes security-relevant.
- **Acceptance:** `main()` wraps scanning in an exception firewall → stderr
  message + exit 2; exit-code table in docs updated.

### HW-120 — `-f text -o FILE` silently writes JSON (P1)
- **Where:** `cli.py:127-131`
- **Acceptance:** write plain (ANSI-free) text to file, matching what the
  terminal would show.

### HW-121 — False `Typing :: Typed` claim (P1)
- **Where:** `pyproject.toml:30`; missing `hayward/py.typed`
- **Acceptance:** add `py.typed` marker, verify it lands in the wheel.

### HW-122 — Docs examples that break when followed (P1)
- **Where:** `docs/usage.md:147` (library example compares `int <= str` →
  TypeError, verified), `docs/usage.md:103-112` (pre-commit hook passes every
  staged filename but `scan` accepts one positional → argparse exit 2 with ≥2
  files, verified)
- **Acceptance:** both snippets run as written.

### HW-123 — Version has two sources of truth (P2)
- **Where:** `pyproject.toml:7` + `hayward/__init__.py:25`
- **Acceptance:** single source (hatchling dynamic version or
  importlib.metadata).

### HW-124 — Dead code sweep (P2)
- `ModelFileFinding` dataclass (`scanner.py:2987-2994`) never constructed but
  exported (`__init__.py:23`) — remove both.
- `SAFETENSORS_MAGIC` (`scanner.py:2749`) unused — remove.
- `Finding.start_line` always 0, never serialized (`findings.py`) — remove.
- Redundant `engine="mfv"` at ~50 call sites — drop, use the `"hayward"`
  default.
- Note in CHANGELOG (public-API removal → minor version bump).

### HW-125 — Colour/NO_COLOR ergonomics (P2)
- **Where:** `cli.py:128`
- **Acceptance:** honour `NO_COLOR` env; add `--color {auto,always,never}`
  (keep `--no-colour` as alias).

### HW-126 — GUI polish batch (P2)
- Friendly error when Tk can't initialise (headless Linux) — `gui.py:388-392`.
- File-dialog filter missing `.pickle`, `.ckpt`, `.th`, `.hdf5` — `gui.py:235`.
- Multi-file drop scans only the first file — `gui.py:250-252`.
- `_display_path` falls back to bare `path.name` — show full path (`gui.py:333`).

### HW-127 — Release pipeline hardening (P2)
- Pin actions to commit SHAs (currently mutable `@v4`/`@v5` tags).
- `publish.yml`: smoke-test the built wheel (install + run one scan) before
  upload; add PyPI attestation/provenance step.

---

## WS-4 — Test gaps (P1–P2)

### HW-130 — Regression tests for every WS-1/WS-2 fix (P0 follow-through)
One named test per fix above (DoS caps, severity inversion, Keras decoy, zip
gate, crash firewall, npy cap, etc.).

### HW-131 — Untested rules (P1)
Direct tests for `MFV-ST-002/003/004/005` (ST-005 is CRITICAL),
`MFV-GGUF-001`, `MFV-KERAS-002`, `MFV-CONFUSE-001` (the headline
magic-byte-vs-extension feature currently has no direct test).

### HW-132 — CLI/report surface tests (P1)
Exit codes 0/1/2 (including crash→2), text/json/html/markdown output,
`--fail-on`, `--fail-on-coverage`, `-o` behaviour. Zero tests exist for
`cli.py`/`report.py` today.

### HW-133 — Coverage measurement (P2)
pytest-cov in dev deps + CI job reporting coverage (no threshold initially).

### HW-134 — No-outbound-calls proof (P2)
Socket-blocking test: monkeypatch `socket.socket` to raise, run a directory
scan over all fixture types, assert no attempt.

---

## WS-5 — Features (P3, prioritised)

### High impact
- **HW-140 — SARIF output** + `schema_version` in JSON. The CI-distribution
  channel; GitHub code scanning ingest format.
- **HW-141 — Git LFS pointer detection** → coverage finding. Trivial to build,
  purest expression of the "did we actually read the file" thesis.
- **HW-142 — HF repo JSON vectors**: `tokenizer_config.json` `chat_template`
  (Jinja2, executed by transformers) and `config.json` `auto_map` /
  `trust_remote_code` (the classic HF-hub RCE vector). The GGUF equivalent
  (MFV-GGUF-003) already exists — this is extension, not invention.
- **HW-143 — Config file with auditable allowlisting** (hash-based,
  justification-required). Biggest enterprise-adoption blocker.
- **HW-144 — Publish the accuracy corpus/harness.** Every accuracy figure is
  currently self-declared unverifiable. *Blocked: needs corpus ownership
  decision.*

### Medium
- **HW-145 — GitHub Action** (paired with HW-140 SARIF upload).
- **HW-146 — Parallel directory scanning + `--progress`/`--verbose`/`--quiet`.**
  Also: multiple positional targets, `--exclude`, tunable size cap.
- **HW-147 — Split `scanner.py` (5,685 lines) into modules**: pickle engine /
  classifier / binary formats / protobuf walker / containers / keras. Dedupe
  the 4× container-walk boilerplate; add a finding factory. Do after WS-1/WS-2
  stabilise.
- **HW-148 — Reproducible flagship builds**: Docker image; commit the
  WASM/pyodide build pipeline (browser demo currently unreproducible from
  source).
- **HW-149 — Format breadth**: torch.package/TorchScript Python-source members,
  MLflow model dirs, CoreML `.mlpackage`, JAX msgpack, shelve, `.ptl`.
- **HW-150 — Baseline/diff mode for brownfield CI** (first-mover opportunity).

### Low / watch
- **HW-151 — CycloneDX ML-BOM emission** (standard exists, demand doesn't yet).
- **HW-152 — Sigstore/SLSA model-signature verification** (ecosystem nascent).
- **HW-153 — Plugin architecture, per-rule severity overrides, scan caching.**

---

## Execution plan

| Phase | Items | Owner |
|---|---|---|
| 1 | HW-101…104, 110…112, 116, 117 (+ own regression tests) | subagent A (scanner.py only) |
| 1 | HW-105, 120…127 | subagent B (cli/report/gui/docs/CI) |
| 2 | HW-131, 133, 134 | subagent C |
| 2 | HW-132 | subagent D |
| 2 | HW-140, 141 | subagent E (if phase 1 lands clean) |
| 3 | HW-113, 114, 115, 124, CHANGELOG, version bump | maintainer / follow-up |
| later | WS-5 remainder | roadmap |
