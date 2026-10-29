"""Scan a Hugging Face Hub repo without loading it: ``hayward hub org/name``.

It uses the network (so does creating a ``hayward lock``). Plain ``hayward scan`` never does.
It lists the repo's files at a pinned commit, downloads the ones Hayward can scan
into a temp folder, then runs the ordinary ``scan`` on that folder, so every
``scan`` flag, report format and exit code works unchanged. Nothing is loaded or
executed, and the temp folder is deleted afterwards.

A file over the size cap is not fetched whole. A zip checkpoint (what ``torch.save``
writes) is instead read by Range request for its pickle member only, which is the part
that can run code; the tensor data is not read, and the run says so. Anything else over
the cap is named on stderr as NOT SCANNED, so a clean result is never read as covering it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import zlib
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

# Range reads: only the zip checkpoints torch.save writes.
RANGE_SUFFIXES = frozenset({".pt", ".pth", ".bin", ".ckpt", ".ptl", ".th"})
RANGE_BUDGET = 64 * 1024 * 1024  # bytes fetched, and bytes of pickle held, per file
ZIP_MAGIC = b"PK\x03\x04"
PICKLE_MEMBER_SUFFIXES = (".pkl", ".pickle")
MAX_PICKLE_MEMBERS = 16  # torch.save writes one; each member costs a few requests
# What a malformed zip can raise while it is parsed or read.
ZIP_ERRORS = (zipfile.BadZipFile, ValueError, EOFError, RuntimeError, NotImplementedError,
              struct.error, zlib.error)


class HubError(Exception):
    """Something went wrong talking to the Hub; reported as exit code 2."""


class TooLarge(Exception):
    """A download passed the per-file cap."""


class RangeSkip(Exception):
    """A big file could not be read by range; the message says why, in plain words."""


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


def _open(url: str, byte_range: tuple[int, int] | None = None):
    for attempt in range(RETRIES_429 + 1):
        req = _request(url)
        if byte_range is not None:
            # A plain header, so it follows the redirect to the CDN; the token does not.
            req.add_header("Range", f"bytes={byte_range[0]}-{byte_range[1]}")
        try:
            return urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS)  # noqa: S310 - https Hub host checked in _request
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


def _fetch_range(url: str, start: int, stop: int, size: int) -> bytes:
    """Bytes ``start`` to ``stop`` (exclusive) of a ``size``-byte file, or RangeSkip.

    Only the exact range asked for is accepted: a 206 whose Content-Range names those
    bytes of a file of that size, over https. Anything else could be a whole file, or
    a different file, handed back as if it were the slice.
    """
    try:
        with _open(url, (start, stop - 1)) as resp:
            if not resp.geturl().startswith("https://"):
                raise RangeSkip("the download was redirected off https")
            if resp.status != 206:
                raise RangeSkip("the server does not support range requests for this file")
            if resp.headers.get("Content-Range") != f"bytes {start}-{stop - 1}/{size}":
                raise RangeSkip("the server returned a different range than the one asked for")
            data = resp.read(stop - start + 1)
    except (OSError, urllib.error.URLError) as exc:
        raise RangeSkip(f"the range request failed ({type(exc).__name__})") from None
    if len(data) != stop - start:
        raise RangeSkip("the server returned a short range")
    return data


class RangeFile:
    """A read-only, seekable view of a remote file, enough for ``zipfile``.

    Every byte requested counts against ``budget``, checked before the request is sent.
    """

    def __init__(self, url: str, size: int, budget: int):
        self.url, self.size, self.budget = url, size, budget
        self.pos = 0
        self.fetched = 0

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        base = {os.SEEK_SET: 0, os.SEEK_CUR: self.pos, os.SEEK_END: self.size}[whence]
        if base + offset < 0:
            raise OSError("seek before the start of the file")
        self.pos = base + offset
        return self.pos

    def read(self, n: int = -1) -> bytes:
        stop = self.size if n is None or n < 0 else min(self.pos + n, self.size)
        if stop <= self.pos:
            return b""
        if stop - self.pos > self.budget - self.fetched:
            raise RangeSkip("the pickle is bigger than the range-read budget")
        data = _fetch_range(self.url, self.pos, stop, self.size)
        self.fetched += len(data)
        self.pos = stop
        return data


def _range_scan_zip(url: str, size: int, dest: Path) -> list[str]:
    """Rebuild a big zip checkpoint at ``dest`` holding only its pickle members.

    Returns the member names. Raises RangeSkip for anything that is not a plain zip
    checkpoint, or that goes over the budget; the caller reports the file as NOT SCANNED.
    """
    remote = RangeFile(url, size, RANGE_BUDGET)
    # torch.load treats a file that does not open with the zip magic as the legacy
    # layout (a tar or a bare pickle), even when a zip sits at its end.
    if remote.read(4) != ZIP_MAGIC:
        raise RangeSkip("it is not a zip checkpoint (an older layout), so only a full read could scan it")
    try:
        with zipfile.ZipFile(remote) as zf, zipfile.ZipFile(dest, "w") as out:
            members = [i for i in zf.infolist() if i.filename.endswith(PICKLE_MEMBER_SUFFIXES)]
            if not members:
                raise RangeSkip("it has no pickle member to read")
            if len(members) > MAX_PICKLE_MEMBERS:
                raise RangeSkip(f"it has more than {MAX_PICKLE_MEMBERS} pickle members")
            if any(not _safe_relative(i.filename) for i in members):
                raise RangeSkip("a pickle member name is not a plain relative path")
            if sum(i.compress_size for i in members) > RANGE_BUDGET - remote.fetched:
                raise RangeSkip("the pickle is bigger than the range-read budget")
            held = 0  # decompressed bytes kept, capped by the same budget
            for info in members:
                with zf.open(info) as member:
                    data = member.read(RANGE_BUDGET - held + 1)
                held += len(data)
                if held > RANGE_BUDGET:
                    raise RangeSkip("the pickle is bigger than the range-read budget")
                out.writestr(info.filename, data)
    except ZIP_ERRORS:
        raise RangeSkip("its zip structure could not be read") from None
    return [i.filename for i in members]


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

    Returns ``{"sha", "fetched", "partial", "too_large", "refused"}``. ``partial`` holds big
    zip checkpoints of which only the pickle member was read; ``too_large`` holds
    ``(name, size, reason)`` for files not scanned at all. Every file comes from the commit
    the API reported, so a push during the scan cannot swap one in.
    """
    quoted = urllib.parse.quote(repo, safe="/")
    info = _api_get(f"{HUB}/api/models/{quoted}/revision/{urllib.parse.quote(revision)}?blobs=true")
    sha = str(info.get("sha", ""))
    if not COMMIT_RE.fullmatch(sha):
        raise HubError("the Hub did not return a commit id for this repo")
    suffixes = _scannable_suffixes()
    result: dict = {"sha": sha, "fetched": [], "partial": [], "too_large": [], "refused": []}
    root = dest.resolve()
    for sibling in info.get("siblings", []):
        name = str(sibling.get("rfilename", ""))
        if not _wanted(name, suffixes):
            continue
        if not _safe_relative(name):
            result["refused"].append(name)
            continue
        size = sibling.get("size")
        target = (root / name).resolve()
        if root not in target.parents:
            result["refused"].append(name)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"{HUB}/{quoted}/resolve/{sha}/{urllib.parse.quote(name)}"
        if isinstance(size, int) and size > max_bytes:
            reason = "over the cap"
            if Path(name).suffix.lower() in RANGE_SUFFIXES:
                try:
                    members = _range_scan_zip(url, size, target)
                    result["partial"].append((name, size, members))
                    continue
                except RangeSkip as exc:
                    target.unlink(missing_ok=True)
                    reason = f"over the cap; {exc}"
            result["too_large"].append((name, size, reason))
            continue
        try:
            _download(url, target, max_bytes)
        except TooLarge as exc:
            target.unlink(missing_ok=True)
            result["too_large"].append((name, exc.args[0], "over the cap"))
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
    if got["partial"]:
        names = ", ".join(name for name, _, _ in got["partial"])
        print(f"hayward hub: read only the pickle member of {len(got['partial'])} big checkpoint(s), "
              f"not the tensor data: {names}", file=sys.stderr)
    for name, size, reason in got["too_large"]:
        print(f"hayward hub: NOT SCANNED ({size / 1_000_000:.0f} MB, {reason}): {name}", file=sys.stderr)
    if got["too_large"]:
        print("hayward hub: a clean result does not cover those files; raise --max-file-mb to fetch them whole",
              file=sys.stderr)
    return code
