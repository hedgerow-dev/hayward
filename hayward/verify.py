"""``hayward verify org/name [DIR]``: check a Hub repo against its published, signed verdict.

    hayward verify org/name [--revision REV] [DIR] [--allow-gaps]
    hayward verify --sha256 HASH [--allow-gaps]

The hayward-verify service scans popular Hub repos at pinned commits and publishes a
signed scan receipt for each one it finds clean or could not fully read; it never
publishes a flagged repo. This command resolves the revision to a commit, fetches the
receipt for that commit, checks that it was signed by the service's own workflow on its
schedule, that it is under 30 days old, that it names this repo and commit, and that its
file hashes match the Hub's. With DIR, a downloaded snapshot, every file there must be
the file at that commit. A file the verdict does not fully cover fails it, unless
--allow-gaps.

With --sha256, one file hash is looked up in the published verdict log instead: the
newest signed checkpoint (under 3 days old) names a digest for each of 256 shards, and
the shard holding the hash lists every current verdict for those bytes. It passes only
if each one is clean with no coverage gaps.

No published verdict is not a pass: the repo may not have been scanned yet, or may
have been held back. It needs ``hayward[sign]`` and uses the network. A verdict is
analysis evidence, not a safety certificate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import __version__, hub, receipt

SERVICE = "https://verify.hedgerow.dev"
IDENTITY = "https://github.com/hedgerow-dev/hayward-verify/.github/workflows/scan.yml@refs/heads/main"
ISSUER = "https://token.actions.githubusercontent.com"
# Only the scheduled run signs: a manual re-run, or any other workflow run on main,
# carries a different trigger in its certificate and is refused.
TRIGGER = "schedule"
# The signing repo and its owner by numeric id, which survive no rename or recreation.
REPOSITORY_ID = "1440229852"
OWNER_ID = "289219779"
MAX_AGE_DAYS = 30  # the service re-signs weekly; older means it stopped or the repo was dropped
CHECKPOINT_TYPE = "https://hedgerow.dev/hayward/checkpoint/v1"
MAX_CHECKPOINT_AGE = timedelta(days=3)  # the log is signed every 6 hours; a receipt counts while a fresh checkpoint lists it
SHA256_RE = re.compile(r"[0-9a-f]{64}")
MAX_BUNDLE_BYTES = 16 * 1024 * 1024
TIMEOUT_SECONDS = 30


def bundle_url(service: str, repo: str, sha: str) -> str:
    return f"{service.rstrip('/')}/v1/hf/{repo}/{sha}.sigstore.json"


def _fetch(url: str) -> bytes | None:
    """The bundle at ``url``, or None when nothing is published there."""
    req = urllib.request.Request(url, headers={"User-Agent": f"hayward-verify/{__version__}"})  # noqa: S310 - https checked by the caller
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:  # noqa: S310 - same
            if urllib.parse.urlsplit(resp.geturl()).scheme != "https":
                raise ValueError("the service redirected off https")
            data = resp.read(MAX_BUNDLE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    if len(data) > MAX_BUNDLE_BYTES:
        raise ValueError("the published bundle is too large")
    return data


def check(statement: dict, repo: str, sha: str, hub_hashes: dict[str, str | None]) -> list[str]:
    """Problems binding the receipt to this repo and commit; an empty list means it fits.

    ``hub_hashes`` maps each file in the commit to the Hub's sha256 for it, or None for a
    file the Hub keeps in git (it gives no sha256 for those).
    """
    source = statement["predicate"].get("source") or {}
    expected = {"host": "huggingface.co", "repo": repo, "revision": sha}
    if source != expected:
        return [f"the verdict is for {source.get('repo')}@{str(source.get('revision'))[:12]}, "
                f"not {repo}@{sha[:12]}"]
    problems = []
    for subject in statement["subject"]:
        name = subject["name"]
        if name not in hub_hashes:
            problems.append(f"not in the repo at this commit: {name}")
        elif hub_hashes[name] is not None and subject["digest"].get("sha256") != hub_hashes[name]:
            problems.append(f"differs from the Hub's sha256: {name}")
    return problems


# Files no loader reads: documentation, licences and images. Everything else a repo
# holds is reported when the verdict does not cover it.
_DOC_SUFFIXES = (".md", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp")


def _never_loaded(name: str) -> bool:
    base = name.rsplit("/", 1)[-1]
    return (base.lower().endswith(_DOC_SUFFIXES) or base == ".gitattributes"
            or base.upper().startswith(("LICENSE", "LICENCE", "NOTICE")))


def _age_days(scanned: object) -> int | None:
    try:
        when = datetime.fromisoformat(str(scanned))
    except ValueError:
        return None
    if when.tzinfo is None:
        return None
    return (datetime.now(timezone.utc) - when).days


def _older(version: str, than: str) -> bool:
    """True when ``version`` is a plain x.y.z release older than ``than``."""
    def parts(v: str) -> tuple[int, ...] | None:
        bits = v.split(".")
        return tuple(int(b) for b in bits) if all(b.isdigit() for b in bits) else None
    a, b = parts(version), parts(than)
    return a is not None and b is not None and a < b


def _snapshot_names(root: Path) -> set[str]:
    """Every file under ``root`` as a repo-relative name, following symlinks."""
    names = set()
    for path in root.rglob("*"):
        rel = path.relative_to(root).as_posix()
        # .cache: huggingface_hub's own bookkeeping in a --local-dir download
        if rel.split("/", 1)[0] not in (".git", ".cache") and path.is_file():
            names.add(rel)
    return names


def check_snapshot(root: Path, hub_files: dict[str, dict]) -> list[str]:
    """Every file under ``root`` must be the file at this commit: the Hub's sha256 for
    LFS files, the git blob id for the rest. Files you did not download are fine.

    Symlinks are followed, so a Hugging Face cache snapshot (links into ../blobs) works.
    """
    from .lock import _git_blob

    problems = []
    for rel in sorted(_snapshot_names(root)):
        path = root / rel
        entry = hub_files.get(rel)
        if entry is None:
            problems.append(f"not in the repo at this commit: {rel}")
            continue
        lfs = (entry.get("lfs") or {}).get("sha256")
        blob = entry.get("blobId")
        if lfs:
            same = hub._sha256(path) == lfs
        else:
            same = isinstance(blob, str) and _git_blob(path) == blob
        if not same:
            problems.append(f"differs from the commit: {rel}")
    return problems


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="hayward verify", description=(__doc__ or "").splitlines()[0])
    parser.add_argument("repo", nargs="?", help="the repo id, e.g. org/name")
    parser.add_argument("directory", nargs="?", type=Path, help="a downloaded snapshot to check too")
    parser.add_argument("--revision", default="main", help="branch, tag or commit (default main)")
    parser.add_argument("--sha256", help="look up one file's SHA-256 in the verdict log instead of a repo")
    parser.add_argument("--allow-gaps", action="store_true",
                        help="pass even when a file was not fully checked or not covered by the verdict")
    parser.add_argument("--service", default=SERVICE, help="verdict service URL, for a self-hosted one")
    parser.add_argument("--identity", default=IDENTITY, help="the signer the service's verdicts must carry")
    parser.add_argument("--issuer", default=ISSUER, help="the OIDC issuer of that signer")
    parser.add_argument("--trigger", default=TRIGGER, help="the GitHub event the signing run must come from")
    parser.add_argument("--repository-id", default=REPOSITORY_ID, help="the signing repo's numeric GitHub id")
    parser.add_argument("--owner-id", default=OWNER_ID, help="the signing repo owner's numeric GitHub id")
    args = parser.parse_args(argv)
    if (args.repo is None) == (args.sha256 is None):
        print("hayward verify: give a repo id or --sha256 HASH, not both", file=sys.stderr)
        return 2
    if args.sha256 is not None and not SHA256_RE.fullmatch(args.sha256):
        print("hayward verify: --sha256 takes 64 lowercase hex characters", file=sys.stderr)
        return 2
    if args.sha256 is not None and args.revision != "main":
        print("hayward verify: --revision is for a repo, not --sha256", file=sys.stderr)
        return 2
    if args.sha256 is None and (not hub.REPO_RE.fullmatch(args.repo) or not hub.REVISION_RE.fullmatch(args.revision)):
        print(f"hayward verify: {args.repo!r} at {args.revision!r} is not a usable repo id and revision",
              file=sys.stderr)
        return 2
    if urllib.parse.urlsplit(args.service).scheme != "https":
        print("hayward verify: the service must be https", file=sys.stderr)
        return 2
    try:
        return _run_sha256(args) if args.sha256 is not None else _run(args)
    except ImportError:
        print('hayward verify: checking signatures needs: pip install "hayward[sign]"', file=sys.stderr)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, receipt.SignatureError, hub.HubError,
            urllib.error.URLError) as exc:
        print(f"hayward verify: {exc}", file=sys.stderr)
    return 2


def _run(args: argparse.Namespace) -> int:
    quoted = urllib.parse.quote(args.repo, safe="/")
    info = hub._api_get(f"{hub.HUB}/api/models/{quoted}/revision/{urllib.parse.quote(args.revision)}?blobs=true")
    sha = str(info.get("sha", ""))
    if not hub.COMMIT_RE.fullmatch(sha):
        raise hub.HubError("the Hub did not return a commit id for this repo")
    hub_files = {str(s.get("rfilename", "")): s for s in info.get("siblings", [])}
    hub_hashes = {name: (s.get("lfs") or {}).get("sha256") for name, s in hub_files.items()}
    raw = _fetch(bundle_url(args.service, args.repo, sha))
    if raw is None:
        print(f"{args.repo}@{sha[:12]}: no published verdict. It may not have been scanned yet, or may "
              f"have been held back. This is not a pass.")
        return 1
    statement = json.loads(receipt._sigstore_open(raw, args.identity, args.issuer, args.trigger,
                                                  args.repository_id, args.owner_id))
    receipt._require_receipt(statement)
    problems = check(statement, args.repo, sha, hub_hashes)
    # Age from the signing certificate, which Sigstore vouches for, not from the
    # receipt's own `scanned` field.
    age = _age_days(receipt._signed_at(raw).isoformat())
    if age is None or age > MAX_AGE_DAYS:
        problems.append(f"the verdict is {'of unknown age' if age is None else f'{age} days old'}; "
                        f"over {MAX_AGE_DAYS} days does not count. Wait for the next rescan.")
    if not statement["subject"]:
        problems.append("the verdict covers no files")
    if args.directory is not None and not problems:
        # Every file in the snapshot must be the file at this commit, so a changed
        # .py or config the scan never covered is caught too. The verdict is already
        # bound to the commit, so a covered file you did not download is fine.
        problems += check_snapshot(args.directory, hub_files)

    files = statement["predicate"]["files"]
    # With a snapshot, only the files you downloaded (what you will load) can be gaps;
    # without one, every file in the repo can.
    in_scope = _snapshot_names(args.directory) if args.directory is not None else set(hub_files)
    uncovered = sorted(n for n in hub_files if n in in_scope and n not in files and not _never_loaded(n))
    scanner = statement["predicate"].get("scanner") or {}
    print(f"{args.repo} at commit {sha}, signed by {args.identity}")
    print(f"  scanned {statement['predicate'].get('scanned', '?')} with hayward {scanner.get('version', '?')}")
    if _older(str(scanner.get("version", "")), __version__):
        print(f"  note: that is older than this hayward ({__version__}); detections added since were not run")
    for name, entry in sorted(files.items()):
        note = f" ({entry['read']} read)" if entry.get("read") else ""
        if entry.get("verdict") == "remote_code":
            note += " (runs code from the repo with trust_remote_code; not vouched for)"
        print(f"  {entry.get('verdict', '?'):<13} {name}{note}")
    for name in uncovered:
        kind = "not covered, CODE" if name.lower().endswith(".py") else "not covered"
        print(f"  {kind:<13} {name}")
    left_out = [n for n in hub_files if n not in in_scope and n not in files and not _never_loaded(n)]
    if left_out:
        print(f"  ({len(left_out)} other file(s) in the repo were not downloaded and are not checked; "
              f"load only what you verified)")
    for problem in problems:
        print(f"  problem: {problem}")

    verdicts = [entry.get("verdict") for name, entry in files.items() if name in in_scope]
    gaps = verdicts.count("not_verified") + verdicts.count("remote_code") + len(uncovered)
    if problems or "flagged" in verdicts:
        print(f"{args.repo}: does NOT match its published verdict")
        return 1
    if gaps and not args.allow_gaps:
        print(f"{args.repo}: {gaps} file(s) not fully checked, so this is not a pass "
              f"(--allow-gaps to accept them)")
        return 1
    print(f"{args.repo}: verdict verified ({verdicts.count('clean')} clean, {gaps} not fully checked). "
          f"Load this exact commit: revision={sha}")
    return 0


def _printable(value: object) -> str:
    """``value`` with control characters replaced, so a file name cannot forge output lines."""
    return "".join(c if c.isprintable() else "?" for c in str(value))


def fetch_checkpoint(service: str = SERVICE, identity: str = IDENTITY, issuer: str = ISSUER,
                     trigger: str = TRIGGER, repository_id: str = REPOSITORY_ID,
                     owner_id: str = OWNER_ID) -> tuple[int, dict, datetime]:
    """The verdict log's current checkpoint: its run number, its signed statement and
    when it was signed (from the certificate). Raises ValueError for anything that does
    not check out, receipt.SignatureError for a wrong signer."""
    base = service.rstrip("/")
    head = _fetch(f"{base}/v1/head.json")
    if head is None:
        raise ValueError("the service publishes no verdict log")
    n = json.loads(head).get("checkpoint")
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise ValueError("the verdict log's head names no checkpoint")
    raw = _fetch(f"{base}/v1/checkpoints/{n}.sigstore.json")
    if raw is None:
        raise ValueError(f"checkpoint {n} is not published")
    statement = json.loads(receipt._sigstore_open(raw, identity, issuer, trigger, repository_id, owner_id))
    if (not isinstance(statement, dict) or statement.get("_type") != receipt.STATEMENT_TYPE
            or statement.get("predicateType") != CHECKPOINT_TYPE):
        raise ValueError("not a Hayward verdict log checkpoint")
    run = statement["predicate"].get("run")
    if not isinstance(run, int) or isinstance(run, bool) or run != n:
        raise ValueError(f"the head names checkpoint {n}, but it is signed as run {run}")
    # Age from the signing certificate, as for receipts.
    return n, statement, receipt._signed_at(raw)


def fetch_shard(statement: dict, n: int, prefix: str, service: str = SERVICE) -> dict:
    """The shard of checkpoint ``n`` for hashes starting with ``prefix``, checked against
    the digest the signed checkpoint names for it."""
    digests = {s["name"]: s["digest"].get("sha256") for s in statement["subject"]}
    digest = digests.get(prefix)
    if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
        raise ValueError(f"checkpoint {n} names no shard {prefix}")
    data = _fetch(f"{service.rstrip('/')}/v1/shards/{digest}.json")
    if data is None:
        raise ValueError(f"shard {prefix} of checkpoint {n} is not published")
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"shard {prefix} does not match the checkpoint")
    shard = json.loads(data)
    if not isinstance(shard, dict):
        raise ValueError(f"shard {prefix} is not a hash index")
    return shard


def has_gap(entry: dict) -> bool:
    """A publishable verdict that does not fully clear the file."""
    verdict = entry.get("verdict")
    return verdict in ("not_verified", "remote_code") or (verdict == "clean" and bool(entry.get("coverage_gaps")))


def _run_sha256(args: argparse.Namespace) -> int:
    n, statement, signed_at = fetch_checkpoint(args.service, args.identity, args.issuer, args.trigger,
                                               args.repository_id, args.owner_id)
    if datetime.now(timezone.utc) - signed_at > MAX_CHECKPOINT_AGE:
        print(f"checkpoint {n} was signed over {MAX_CHECKPOINT_AGE.days} days ago, so it does not "
              f"count. The service may have stopped. This is not a pass.")
        return 1
    entries = fetch_shard(statement, n, args.sha256[:2], args.service).get(args.sha256) or []

    print(f"sha256 {args.sha256}, checkpoint {n} signed by {args.identity}")
    if not entries:
        print(f"{args.sha256[:12]}: no current verdict. It may not have been scanned, or may have been "
              f"withdrawn or held back. This is not a pass.")
        return 1
    failed = gaps = 0
    for entry in entries:
        verdict = entry.get("verdict")
        note = f" ({_printable(entry['read'])} read)" if entry.get("read") else ""
        line = (f"  {_printable(verdict)}{note}  {_printable(entry.get('repo'))}@"
                f"{_printable(entry.get('commit'))[:12]}  {_printable(entry.get('name'))}"
                f"  hayward {_printable(entry.get('scanner', '?'))}")
        if entry.get("hash_source") == "hub":
            line += "  (hash from the Hub)"
        if verdict == "remote_code":
            line += "  (runs code from the repo with trust_remote_code; not vouched for)"
        if entry.get("coverage_gaps"):
            line += f"  gaps: {', '.join(map(_printable, entry['coverage_gaps']))}"
        print(line)
        if has_gap(entry):
            gaps += 1
        elif verdict != "clean":
            failed += 1  # flagged is never published, so anything else is wrong
    if failed:
        print(f"{args.sha256[:12]}: an entry is not a publishable verdict; this is NOT a pass")
        return 1
    if gaps and not args.allow_gaps:
        print(f"{args.sha256[:12]}: {gaps} of {len(entries)} verdict(s) not fully checked, so this is not a pass "
              f"(--allow-gaps to accept them)")
        return 1
    print(f"{args.sha256[:12]}: verdict verified ({len(entries) - gaps} clean, {gaps} not fully checked). "
          f"Analysis evidence, not a safety certificate.")
    return 0
