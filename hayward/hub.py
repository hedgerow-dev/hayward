"""Scan a Hugging Face Hub repo without loading it: ``hayward hub org/name``.

This is the one command that uses the network. Plain ``hayward scan`` never does.
It lists the repo's files at a pinned commit, downloads the ones Hayward can scan
into a temp folder, then runs the ordinary ``scan`` on that folder, so every
``scan`` flag, report format and exit code works unchanged. Nothing is loaded or
executed, and the temp folder is deleted afterwards.

A file over the size cap is not fetched and is named on stderr as NOT SCANNED, so a
clean result is never read as covering it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from . import __version__
from .cli import main as scan_main
from .scanner import ModelFileScanner

HUB = "https://huggingface.co"
TIMEOUT_SECONDS = 30
RETRIES_429 = 4
# "org/name": ASCII letters, digits, "_", "." and "-", no empty or dot-only parts.
REPO_RE = re.compile(r"\w[\w.-]{0,95}/\w[\w.-]{0,95}", re.ASCII)
# A branch, tag or commit. No slash, so it cannot change the shape of the URL.
REVISION_RE = re.compile(r"\w[\w.-]{0,99}", re.ASCII)
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
# Of the .json files only these carry something the scanner reads (auto_map,
# trust_remote_code, chat templates), so the rest are not worth downloading.
CONFIG_NAMES = frozenset({"config.json", "tokenizer_config.json"})


class HubError(Exception):
    """Something went wrong talking to the Hub; reported as exit code 2."""


class TooLarge(Exception):
    """A download passed the per-file cap."""


def _request(url: str) -> urllib.request.Request:
    """A request that can only go to the Hub over https.

    HF_TOKEN, if set, is for private or gated repos. It is added as an *unredirected*
    header, because /resolve/ redirects to a CDN host and a plain header would be
    copied there too.
    """
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https" or parts.netloc != "huggingface.co":
        raise ValueError(f"refusing a URL that is not on {HUB}: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": f"hayward-hub/{__version__}"})  # noqa: S310 - https Hub host checked above
    token = os.environ.get("HF_TOKEN")
    if token:
        req.add_unredirected_header("Authorization", f"Bearer {token}")
    return req


def _open(url: str):
    for attempt in range(RETRIES_429 + 1):
        try:
            return urllib.request.urlopen(_request(url), timeout=TIMEOUT_SECONDS)  # noqa: S310 - https Hub host checked in _request
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == RETRIES_429:
                raise
            wait = exc.headers.get("Retry-After", "")
            time.sleep(int(wait) if wait.isdigit() else 5 * 2 ** attempt)
    raise AssertionError("unreachable")


def _api_get(url: str) -> dict:
    with _open(url) as resp:
        return json.load(resp)


def _download(url: str, dest: Path, max_bytes: int) -> None:
    with _open(url) as resp, dest.open("wb") as out:
        length = resp.headers.get("Content-Length", "")
        if length.isdigit() and int(length) > max_bytes:
            raise TooLarge(int(length))
        written = 0
        while block := resp.read(1 << 20):
            written += len(block)
            if written > max_bytes:
                raise TooLarge(written)
            out.write(block)


def _scannable_suffixes() -> frozenset[str]:
    scanner = ModelFileScanner()
    return frozenset((*scanner._format_map, *scanner._AMBIGUOUS_EXTENSIONS))


def _wanted(name: str, suffixes: frozenset[str]) -> bool:
    base = name.rsplit("/", 1)[-1]
    return base in CONFIG_NAMES or Path(base).suffix.lower() in suffixes


def _safe_relative(name: str) -> bool:
    """A file name from the API is data. Refuse anything that could leave the folder."""
    if not name or name.startswith("/") or "\\" in name or "\x00" in name:
        return False
    return all(part not in ("", ".", "..") for part in name.split("/"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(repo: str, revision: str, dest: Path, max_bytes: int) -> dict:
    """Download a repo's scannable files into ``dest``.

    Returns ``{"sha", "fetched", "too_large", "refused"}``. Every file comes from the
    commit the API reported, so a push during the scan cannot swap one in.
    """
    quoted = urllib.parse.quote(repo, safe="/")
    info = _api_get(f"{HUB}/api/models/{quoted}/revision/{urllib.parse.quote(revision)}?blobs=true")
    sha = str(info.get("sha", ""))
    if not COMMIT_RE.fullmatch(sha):
        raise HubError("the Hub did not return a commit id for this repo")
    suffixes = _scannable_suffixes()
    result: dict = {"sha": sha, "fetched": [], "too_large": [], "refused": []}
    root = dest.resolve()
    for sibling in info.get("siblings", []):
        name = str(sibling.get("rfilename", ""))
        if not _wanted(name, suffixes):
            continue
        if not _safe_relative(name):
            result["refused"].append(name)
            continue
        size = sibling.get("size")
        if isinstance(size, int) and size > max_bytes:
            result["too_large"].append((name, size))
            continue
        target = (root / name).resolve()
        if root not in target.parents:
            result["refused"].append(name)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"{HUB}/{quoted}/resolve/{sha}/{urllib.parse.quote(name)}"
        try:
            _download(url, target, max_bytes)
        except TooLarge as exc:
            target.unlink(missing_ok=True)
            result["too_large"].append((name, exc.args[0]))
            continue
        expected = (sibling.get("lfs") or {}).get("sha256")
        if expected and _sha256(target) != expected:
            raise HubError(f"sha256 of {name} differs from the Hub's own record; not scanning")
        result["fetched"].append(name)
    return result


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="hayward hub",
        description="Fetch a Hugging Face Hub repo's model files and scan them. Uses the network; "
                    "nothing is loaded. Any other flag is passed to `hayward scan`.",
    )
    parser.add_argument("repo", help="the repo id, e.g. org/name")
    parser.add_argument("--revision", default="main", help="branch, tag or commit (default main)")
    parser.add_argument("--max-file-mb", type=float, default=50,
                        help="skip files bigger than this and say so (default 50)")
    args, scan_flags = parser.parse_known_args(argv)
    if not REPO_RE.fullmatch(args.repo):
        print(f"hayward hub: {args.repo!r} is not a repo id like org/name", file=sys.stderr)
        return 2
    if not REVISION_RE.fullmatch(args.revision):
        print(f"hayward hub: {args.revision!r} is not a usable revision (a branch without "
              "slashes, a tag or a commit id)", file=sys.stderr)
        return 2
    quiet = "--quiet" in scan_flags or "-q" in scan_flags
    with tempfile.TemporaryDirectory(prefix="hayward-hub-") as tmp:
        try:
            got = fetch(args.repo, args.revision, Path(tmp), int(args.max_file_mb * 1024 * 1024))
        except (HubError, OSError, ValueError, urllib.error.URLError) as exc:
            print(f"hayward hub: {exc}", file=sys.stderr)
            return 2
        if not quiet:
            print(f"hayward: {args.repo} at {got['sha'][:12]}: fetched {len(got['fetched'])} file(s)",
                  file=sys.stderr)
        for name in got["refused"]:
            print(f"hayward hub: refusing the file name {name!r}", file=sys.stderr)
        code = scan_main(["scan", tmp, *scan_flags], label=f"hf://{args.repo}@{got['sha'][:12]}")
    for name, size in got["too_large"]:
        print(f"hayward hub: NOT SCANNED (over the cap, {size / 1_000_000:.0f} MB): {name}", file=sys.stderr)
    if got["too_large"]:
        print("hayward hub: a clean result does not cover those files; raise --max-file-mb to fetch them",
              file=sys.stderr)
    return code
