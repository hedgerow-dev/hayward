"""``hayward lock``: pin a Hub repo's files for one loading profile, and check a snapshot.

    hayward lock org/name [--revision REV] [-o hayward.lock.json]
    hayward lock --verify hayward.lock.json path/to/snapshot

The lock names the commit and every file in the repo with its hash (SHA-256 for LFS
files, the git blob id for the rest), so a downloaded snapshot can be checked without
trusting where it came from. It is written for one profile, ``transformers``
``from_pretrained`` at a pinned revision, and it fails closed: anything that profile
would load from outside this repo is pinned too or recorded as unresolved, and an
unresolved lock is marked "needs_review". Custom code from one other repo is pinned at
a commit, loaded with ``code_revision``; an adapter's base model cannot be, because
transformers loads it at the revision passed for the adapter.

A lock pins what will be loaded. It says nothing about whether the files are safe:
scan them with ``hayward hub`` or ``hayward scan``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import urllib.error
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

from . import hub

FORMAT = "hayward-lock/1"
PROFILE = "transformers.from_pretrained"
CODE_SUFFIXES = frozenset({".py", ".so", ".pyd", ".dll", ".dylib", ".pyc"})
CONFIG_MAX_BYTES = 1024 * 1024
# Any *config.json can carry an auto_map: the model and tokenizer configs, and also the
# processor, image-processor and feature-extractor configs the Auto classes read.
CONFIG_SUFFIX = "config.json"
MAX_CONFIGS = 50
ADAPTER_FILE = "adapter_config.json"


def _git_blob(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha1(b"blob %d\0" % size)  # noqa: S324 - git's content id, not a security hash
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(repo: str, sha: str, name: str, tmp: Path) -> dict | None:
    dest = tmp / Path(name).name
    url = f"{hub.HUB}/{urllib.parse.quote(repo, safe='/')}/resolve/{sha}/{urllib.parse.quote(name)}"
    try:
        hub._download(url, dest, CONFIG_MAX_BYTES)
        data = json.loads(dest.read_text(encoding="utf-8"))
    except (OSError, ValueError, hub.TooLarge, urllib.error.URLError):
        return None
    return data if isinstance(data, dict) else None


def _auto_map_refs(auto_map) -> list[str]:
    """Every class an ``auto_map`` names: ``module.Class`` in this repo, or
    ``other/repo--module.Class`` in another."""
    values = auto_map.values() if isinstance(auto_map, dict) else []
    return [item for value in values for item in (value if isinstance(value, list) else [value])
            if isinstance(item, str)]


def _pin_remote_code(refs: list[str], local_code: bool) -> tuple[dict | None, str]:
    """Pin the other repo that ``refs`` load code from, or say why it cannot be pinned.

    transformers loads every such ref at one ``code_revision``, and loads this repo's own
    auto_map code at that same revision once it is given, so only code from a single
    other repo, with none from this one, can be pinned."""
    repos = sorted({ref.split("--", 1)[0] for ref in refs})
    if len(repos) > 1:
        return None, "code comes from more than one other repo, and one code_revision cannot pin them all"
    if local_code:
        return None, "this repo's own code is loaded too, and code_revision would load it at the other repo's commit"
    other = repos[0]
    if not hub.REPO_RE.fullmatch(other):
        return None, "not a repo id like org/name"
    try:
        info = hub._api_get(f"{hub.HUB}/api/models/{urllib.parse.quote(other, safe='/')}?blobs=true")
    except (hub.HubError, OSError, ValueError, urllib.error.URLError):
        info = {}
    sha = str(info.get("sha", ""))
    if not hub.COMMIT_RE.fullmatch(sha):
        return None, "the Hub did not return a commit for that repo"
    code_files = []
    for sibling in sorted(info.get("siblings", []), key=lambda s: str(s.get("rfilename", ""))):
        name = str(sibling.get("rfilename", ""))
        if Path(name).suffix.lower() not in CODE_SUFFIXES:
            continue
        if not hub._safe_relative(name):
            return None, "that repo has a code file name that is not a plain relative path"
        lfs_sha = (sibling.get("lfs") or {}).get("sha256")
        code_files.append({"path": name, **({"sha256": lfs_sha} if lfs_sha else {"git_blob": sibling.get("blobId")})})
    present = {f["path"] for f in code_files}
    for ref in refs:
        module = ref.split("--", 1)[1].split(".", 1)[0] + ".py"
        if module not in present:
            return None, f"{module} is not in that repo"
    return {"kind": "remote_code_repo", "repo": other, "revision": sha, "code_files": code_files}, ""


def create(repo: str, revision: str) -> dict:
    quoted = urllib.parse.quote(repo, safe="/")
    info = hub._api_get(f"{hub.HUB}/api/models/{quoted}/revision/{urllib.parse.quote(revision)}?blobs=true")
    sha = str(info.get("sha", ""))
    if not hub.COMMIT_RE.fullmatch(sha):
        raise hub.HubError("the Hub did not return a commit id for this repo")
    files, code_files, unresolved = [], [], []
    names = set()
    for sibling in sorted(info.get("siblings", []), key=lambda s: str(s.get("rfilename", ""))):
        name = str(sibling.get("rfilename", ""))
        if not hub._safe_relative(name):
            unresolved.append({"kind": "unsafe_name", "ref": name,
                               "why": "a file name that is not a plain relative path"})
            continue
        entry: dict = {"path": name, "size": sibling.get("size")}
        lfs_sha = (sibling.get("lfs") or {}).get("sha256")
        if lfs_sha:
            entry["sha256"] = lfs_sha
        else:
            entry["git_blob"] = sibling.get("blobId")
        files.append(entry)
        names.add(name)
        if Path(name).suffix.lower() in CODE_SUFFIXES:
            code_files.append(name)
    requires_remote_code = local_code = False
    remote_refs: list[str] = []
    with tempfile.TemporaryDirectory(prefix="hayward-lock-") as tmp:
        configs = sorted(n for n in names if n.rsplit("/", 1)[-1].endswith(CONFIG_SUFFIX))
        if len(configs) > MAX_CONFIGS:
            unresolved.append({"kind": "unreadable", "ref": f"{len(configs)} config files",
                               "why": f"more than {MAX_CONFIGS} config files; not all were checked"})
        for name in configs[:MAX_CONFIGS]:
            config = _read_json(repo, sha, name, Path(tmp))
            if config is None:
                unresolved.append({"kind": "unreadable", "ref": name,
                                   "why": "could not be read as JSON, so what it loads is unknown"})
                continue
            if config.get("auto_map"):
                requires_remote_code = True
                for ref in _auto_map_refs(config["auto_map"]):
                    if "--" in ref and ref.split("--", 1)[0] != repo:
                        remote_refs.append(ref)
                    else:
                        local_code = True
        if ADAPTER_FILE in names:
            adapter = _read_json(repo, sha, ADAPTER_FILE, Path(tmp))
            base = (adapter or {}).get("base_model_name_or_path")
            if adapter is None:
                unresolved.append({"kind": "unreadable", "ref": ADAPTER_FILE,
                                   "why": "could not be read as JSON, so what it loads is unknown"})
            elif base:
                unresolved.append({"kind": "base_model", "ref": str(base),
                                   "why": "transformers loads an adapter's base model at the revision passed "
                                          "for the adapter, so one revision cannot pin both"})
    dependencies, load_with = [], {"revision": sha}
    if remote_refs:
        dependency, why = _pin_remote_code(remote_refs, local_code)
        if dependency:
            dependencies.append(dependency)
            load_with["code_revision"] = dependency["revision"]
        else:
            unresolved.extend({"kind": "remote_code_repo", "ref": ref, "why": why} for ref in remote_refs)
    return {
        "format": FORMAT, "profile": PROFILE, "repo": repo, "revision": sha,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "needs_review" if unresolved else "complete",
        "requires_trust_remote_code": requires_remote_code,
        "load_with": load_with, "dependencies": dependencies,
        "code_files": code_files, "unresolved": unresolved, "files": files,
    }


def verify(lock: dict, root: Path) -> list[str]:
    """Problems with ``root`` as a snapshot of ``lock``; an empty list means it matches."""
    problems = []
    if lock.get("format") != FORMAT:
        return [f"not a {FORMAT} lock"]
    if lock.get("status") != "complete":
        problems.append("the lock needs review: it has unresolved dependencies")
    locked = {f["path"]: f for f in lock.get("files", [])}
    for name, entry in locked.items():
        path = root / name
        if not path.is_file():
            problems.append(f"missing: {name}")
        elif "sha256" in entry and _sha256(path) != entry["sha256"]:
            problems.append(f"changed: {name}")
        elif "git_blob" in entry and _git_blob(path) != entry["git_blob"]:
            problems.append(f"changed: {name}")
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel.split("/", 1)[0] == ".git":
            continue
        if rel not in locked:
            problems.append(f"not in the lock: {rel}")
    return problems


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="hayward lock", description=(__doc__ or "").splitlines()[0])
    parser.add_argument("repo", nargs="?", help="the repo id to lock, e.g. org/name")
    parser.add_argument("--revision", default="main", help="branch, tag or commit (default main)")
    parser.add_argument("-o", "--output", type=Path, default=Path("hayward.lock.json"))
    parser.add_argument("--verify", nargs=2, metavar=("LOCK", "DIR"), help="check a snapshot against a lock")
    args = parser.parse_args(argv)

    if args.verify:
        lock_path, root = Path(args.verify[0]), Path(args.verify[1])
        try:
            data = json.loads(lock_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"hayward lock: {exc}", file=sys.stderr)
            return 2
        problems = verify(data, root)
        for problem in problems:
            print(f"  {problem}")
        print(f"{root}: {'matches' if not problems else 'does NOT match'} {data.get('repo')}@"
              f"{str(data.get('revision'))[:12]}")
        return 1 if problems else 0

    if not args.repo or not hub.REPO_RE.fullmatch(args.repo):
        print(f"hayward lock: {args.repo!r} is not a repo id like org/name", file=sys.stderr)
        return 2
    if not hub.REVISION_RE.fullmatch(args.revision):
        print(f"hayward lock: {args.revision!r} is not a usable revision", file=sys.stderr)
        return 2
    try:
        data = create(args.repo, args.revision)
    except (hub.HubError, OSError, ValueError, urllib.error.URLError) as exc:
        print(f"hayward lock: {exc}", file=sys.stderr)
        return 2
    args.output.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    print(f"hayward: locked {args.repo} at {data['revision'][:12]}: {len(data['files'])} file(s) "
          f"-> {args.output}", file=sys.stderr)
    if "code_revision" in data["load_with"]:
        print(f"hayward lock: load with revision={data['revision']}, "
              f"code_revision={data['load_with']['code_revision']}", file=sys.stderr)
    for item in data["unresolved"]:
        print(f"hayward lock: unresolved {item['kind']}: {item['ref']} ({item['why']})", file=sys.stderr)
    if data["status"] != "complete":
        print("hayward lock: needs review: the profile would load something this lock does not pin",
              file=sys.stderr)
        return 1
    return 0
