#!/usr/bin/env python3
"""Fetch the reference accuracy corpus from the Hugging Face Hub.

    python scripts/fetch_corpus.py corpus/manifest.json CORPUS_DIR [--only benign]

The corpus is not stored in this repository. The manifest names each file by
Hub repo, pinned commit and SHA-256, and this script downloads exactly those
bytes into CORPUS_DIR, ready for `scripts/eval_corpus.py CORPUS_DIR MANIFEST`.
A file whose hash differs from the manifest is deleted and reported, never scanned.

Nothing is loaded or executed: the files are only written to disk. Malicious
entries are deliberate proof-of-concept files from public research repos; treat
CORPUS_DIR as untrusted data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HUB = "https://huggingface.co"
USER_AGENT = "hayward-corpus-fetch/1.0 (https://github.com/hedgerow-dev/hayward)"
PAUSE_SECONDS = 0.3
TIMEOUT_SECONDS = 60


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, dest: Path) -> None:
    if not url.startswith(HUB + "/"):  # the manifest is data: only ever fetch from the Hub over https
        raise ValueError(f"refusing a non-Hub URL: {url}")
    for attempt in range(6):
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310 - https Hub prefix checked above
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp, dest.open("wb") as out:  # noqa: S310 - same
                while block := resp.read(1 << 20):
                    out.write(block)
            return
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == 5:
                raise
            wait = exc.headers.get("Retry-After", "")
            time.sleep(int(wait) if wait.isdigit() else 5 * 2 ** attempt)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("manifest", type=Path)
    p.add_argument("dest", type=Path)
    p.add_argument("--only", choices=("benign", "malicious"), help="fetch just one label")
    args = p.parse_args()
    entries = json.loads(args.manifest.read_text())
    if args.only:
        entries = [e for e in entries if e["expected"] == args.only]
    root = args.dest.resolve()
    ok = skipped = 0
    failed: list[str] = []
    for e in entries:
        target = (root / e["path"]).resolve()
        if root not in target.parents:  # the manifest is data; never write outside DEST
            failed.append(f"{e['path']}: path escapes the corpus directory")
            continue
        if target.exists() and sha256_of(target) == e["sha256"]:
            skipped += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"{HUB}/{urllib.parse.quote(e['repo'], safe='/')}/resolve/{e['revision']}/{urllib.parse.quote(e['hub_path'])}"
        try:
            time.sleep(PAUSE_SECONDS)
            download(url, target)
        except (OSError, urllib.error.URLError) as exc:
            target.unlink(missing_ok=True)
            failed.append(f"{e['path']}: {exc}")
            continue
        if sha256_of(target) != e["sha256"]:
            target.unlink(missing_ok=True)
            failed.append(f"{e['path']}: sha256 differs from the manifest")
            continue
        ok += 1
    print(f"fetched {ok}, already present {skipped}, failed {len(failed)}")
    for line in failed:
        print("  FAILED", line, file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
