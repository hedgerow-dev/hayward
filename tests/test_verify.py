"""`hayward verify org/name`: check a Hub repo against its published, signed verdict.

The network is faked: the Hub API, the service fetch and the Sigstore check are
replaced, so these run offline. The signature check itself is tested in
test_receipt.py and was run live against real bundles.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from datetime import datetime, timedelta, timezone

import pytest

from hayward import cli, hub, receipt, verify

SHA = "e" * 40
CLEAN = pickle.dumps({"w": [1, 2]}, protocol=4)
SHARD = "f" * 64  # the Hub's LFS sha256 for a big file


def statement(files: dict, *, repo="org/name", revision=SHA, digests=None, days_old=1):
    digests = digests or {}
    scanned = (datetime.now(timezone.utc) - timedelta(days=days_old)).isoformat(timespec="seconds")
    return {
        "_type": receipt.STATEMENT_TYPE,
        "predicateType": receipt.PREDICATE_TYPE,
        "subject": [{"name": n, "digest": {"sha256": digests.get(n, SHARD)}} for n in files],
        "predicate": {"scanner": {"name": "hayward", "version": "test", "settings": {}}, "scanned": scanned,
                      "source": {"host": "huggingface.co", "repo": repo, "revision": revision},
                      "note": receipt.NOTE, "files": files},
    }


@pytest.fixture
def service(monkeypatch):
    """Publish a statement, list the Hub's files, and record what was fetched."""
    state = {"published": {}, "siblings": [], "fetched": [], "signer": None}

    def fetch(url):
        state["fetched"].append(url)
        return state["published"].get(url)

    def api_get(url):
        assert url.startswith("https://huggingface.co/api/models/org/name/revision/")
        return {"sha": SHA, "siblings": state["siblings"]}

    def open_bundle(raw, identity, issuer, trigger=None, repository_id=None, owner_id=None):
        state["signer"] = (identity, issuer, trigger, repository_id, owner_id)
        return json.loads(raw)["payload"].encode()

    def signed_at(raw):
        # The fake bundle carries its signing time; a real one has it in the certificate.
        return datetime.fromisoformat(json.loads(raw)["signed_at"])

    monkeypatch.setattr(verify, "_fetch", fetch)
    monkeypatch.setattr(hub, "_api_get", api_get)
    monkeypatch.setattr(receipt, "_sigstore_open", open_bundle)
    monkeypatch.setattr(receipt, "_signed_at", signed_at)

    def publish(st, siblings, signed_days_ago=None):
        url = verify.bundle_url(verify.SERVICE, "org/name", SHA)
        signed = st["predicate"].get("scanned") if signed_days_ago is None else (
            datetime.now(timezone.utc) - timedelta(days=signed_days_ago)).isoformat()
        state["published"][url] = json.dumps({"mediaType": "x", "payload": json.dumps(st),
                                              "signed_at": signed}).encode()
        state["siblings"] = siblings

    state["publish"] = publish
    return state


SIBLINGS = [{"rfilename": "model.safetensors", "lfs": {"sha256": SHARD}},
            {"rfilename": "config.json"}, {"rfilename": "README.md"}]
FILES = {"model.safetensors": {"verdict": "clean", "rules": [], "coverage_gaps": [],
                               "read": "header", "hash_source": "hub"},
         "config.json": {"verdict": "clean", "rules": [], "coverage_gaps": []}}


def test_a_published_clean_verdict_verifies(service, capsys):
    service["publish"](statement(FILES), SIBLINGS)
    assert cli.main(["verify", "org/name"]) == 0
    out = capsys.readouterr().out
    assert "clean" in out and "model.safetensors" in out
    # only the service's own workflow, its scheduled run, from its repo and owner by id
    assert service["signer"] == (verify.IDENTITY, verify.ISSUER, "schedule", verify.REPOSITORY_ID, verify.OWNER_ID)


def test_no_published_verdict_is_not_a_pass(service, capsys):
    service["siblings"] = SIBLINGS
    assert cli.main(["verify", "org/name"]) == 1
    assert "no published verdict" in capsys.readouterr().out


def test_a_verdict_for_another_repo_is_refused(service, capsys):
    service["publish"](statement(FILES, repo="other/repo"), SIBLINGS)
    assert cli.main(["verify", "org/name"]) == 1
    assert "other/repo" in capsys.readouterr().out


def test_a_verdict_for_another_commit_is_refused(service, capsys):
    service["publish"](statement(FILES, revision="0" * 40), SIBLINGS)
    assert cli.main(["verify", "org/name"]) == 1


def test_a_hash_that_differs_from_the_hub_is_refused(service, capsys):
    service["publish"](statement(FILES, digests={"model.safetensors": "1" * 64}), SIBLINGS)
    assert cli.main(["verify", "org/name"]) == 1
    assert "differs from the Hub" in capsys.readouterr().out


def test_a_model_file_the_verdict_does_not_cover_is_named(service, capsys):
    siblings = [*SIBLINGS, {"rfilename": "extra.bin", "lfs": {"sha256": "2" * 64}}]
    service["publish"](statement(FILES), siblings)
    assert cli.main(["verify", "org/name"]) == 1  # strict: a gap is not a pass
    out = capsys.readouterr().out
    assert "not covered" in out and "extra.bin" in out and "--allow-gaps" in out
    assert cli.main(["verify", "org/name", "--allow-gaps"]) == 0


def test_a_not_verified_file_fails_unless_gaps_are_allowed(service):
    files = dict(FILES, **{"config.json": {"verdict": "not_verified", "rules": [], "coverage_gaps": ["MFV-SKIP-003"]}})
    service["publish"](statement(files), SIBLINGS)
    assert cli.main(["verify", "org/name"]) == 1
    assert cli.main(["verify", "org/name", "--allow-gaps"]) == 0


def test_a_verdict_over_30_days_old_does_not_count(service, capsys):
    service["publish"](statement(FILES, days_old=31), SIBLINGS)
    assert cli.main(["verify", "org/name"]) == 1
    assert "31 days old" in capsys.readouterr().out


def test_age_comes_from_the_signature_not_the_receipt(service, capsys):
    # A receipt can claim any scan date; the signing certificate's time is what counts.
    service["publish"](statement(FILES, days_old=0), SIBLINGS, signed_days_ago=40)
    assert cli.main(["verify", "org/name"]) == 1
    assert "40 days old" in capsys.readouterr().out


def _snapshot(tmp_path):
    """A downloaded snapshot, the Hub's file list for it, and a verdict covering the model."""
    from hayward.lock import _git_blob

    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "model.pkl").write_bytes(CLEAN)
    (snap / "config.json").write_text("{}")
    (snap / "generation_config.json").write_text("{}")
    (snap / "modeling_x.py").write_text("print('custom model code')")
    (snap / "README.md").write_text("hi")
    sha256 = hashlib.sha256(CLEAN).hexdigest()
    siblings = [{"rfilename": "model.pkl", "lfs": {"sha256": sha256}}]
    for name in ("config.json", "generation_config.json", "modeling_x.py", "README.md"):
        siblings.append({"rfilename": name, "blobId": _git_blob(snap / name)})
    files = {"model.pkl": {"verdict": "clean", "rules": [], "coverage_gaps": []},
             "config.json": {"verdict": "clean", "rules": [], "coverage_gaps": []}}
    digests = {"model.pkl": sha256, "config.json": hashlib.sha256(b"{}").hexdigest()}
    return snap, statement(files, digests=digests), siblings


def test_a_downloaded_snapshot_is_checked_against_the_verdict(service, tmp_path):
    snap, st, siblings = _snapshot(tmp_path)
    service["publish"](st, siblings)
    assert cli.main(["verify", "org/name", str(snap), "--allow-gaps"]) == 0
    (snap / "model.pkl").write_bytes(CLEAN + b"x")
    assert cli.main(["verify", "org/name", str(snap)]) == 1


def test_changed_code_the_scan_never_covered_is_still_caught(service, tmp_path, capsys):
    snap, st, siblings = _snapshot(tmp_path)
    service["publish"](st, siblings)
    (snap / "modeling_x.py").write_text("import os; os.system('id')")
    assert cli.main(["verify", "org/name", str(snap)]) == 1
    assert "differs from the commit: modeling_x.py" in capsys.readouterr().out


def test_a_file_added_to_the_snapshot_is_caught(service, tmp_path, capsys):
    snap, st, siblings = _snapshot(tmp_path)
    service["publish"](st, siblings)
    (snap / "extra_code.py").write_text("x = 1")
    assert cli.main(["verify", "org/name", str(snap)]) == 1
    assert "not in the repo at this commit: extra_code.py" in capsys.readouterr().out


def test_uncovered_code_is_named_and_docs_are_not_counted(service, tmp_path, capsys):
    _, st, siblings = _snapshot(tmp_path)
    service["publish"](st, siblings)
    assert cli.main(["verify", "org/name", "--allow-gaps"]) == 0
    out = capsys.readouterr().out
    assert "not covered, CODE modeling_x.py" in out
    assert "generation_config.json" in out
    assert "README.md" not in out
    assert "2 not fully checked" in out
    assert f"revision={SHA}" in out


def test_the_service_must_be_https(service):
    assert cli.main(["verify", "org/name", "--service", "http://example.com"]) == 2


def test_a_bad_repo_id_is_refused_before_any_request(service):
    assert cli.main(["verify", "../etc"]) == 2
    assert service["fetched"] == []


def test_a_verdict_that_covers_no_files_fails(service, capsys):
    service["publish"](statement({}), SIBLINGS)
    assert cli.main(["verify", "org/name"]) == 1
    assert "covers no files" in capsys.readouterr().out


def test_a_verdict_from_an_older_hayward_says_so(service, capsys):
    st = statement(FILES)
    st["predicate"]["scanner"]["version"] = "0.1.0"
    service["publish"](st, SIBLINGS)
    assert cli.main(["verify", "org/name"]) == 0
    assert "older than this hayward" in capsys.readouterr().out


def test_with_a_snapshot_only_downloaded_files_can_be_gaps(service, tmp_path, capsys):
    snap, st, siblings = _snapshot(tmp_path)
    service["publish"](st, siblings)
    # The repo also holds a TF weight file the verdict does not cover; it was not downloaded.
    siblings.append({"rfilename": "tf_model.h5", "lfs": {"sha256": "3" * 64}})
    (snap / "modeling_x.py").unlink()
    (snap / "generation_config.json").unlink()
    assert cli.main(["verify", "org/name", str(snap)]) == 0
    assert "tf_model.h5" not in capsys.readouterr().out
    assert cli.main(["verify", "org/name"]) == 1  # without a snapshot, the whole repo counts


def test_a_covered_file_you_did_not_download_is_fine(service, tmp_path, capsys):
    snap, st, siblings = _snapshot(tmp_path)
    service["publish"](st, siblings)
    for name in ("config.json", "modeling_x.py", "generation_config.json"):
        (snap / name).unlink()  # downloaded only the weights
    assert cli.main(["verify", "org/name", str(snap)]) == 0
    assert "not downloaded" in capsys.readouterr().out


def test_remote_code_is_shown_and_needs_allow_gaps(service, capsys):
    files = dict(FILES, **{"config.json": {"verdict": "remote_code", "rules": ["MFV-HF-002"], "coverage_gaps": []}})
    service["publish"](statement(files), SIBLINGS)
    assert cli.main(["verify", "org/name"]) == 1
    out = capsys.readouterr().out
    assert "remote_code" in out and "runs code" in out
    assert cli.main(["verify", "org/name", "--allow-gaps"]) == 0


# `hayward verify --sha256 HASH`: one file hash against the verdict log.

HASH = "ab" + "1" * 62


def _entry(**over):
    entry = {"receipt": "9" * 64, "repo": "org/name", "commit": SHA, "name": "model.safetensors",
             "scanned": "2026-10-31T00:00:00+00:00", "scanner": {"name": "hayward", "version": "1.11.1"},
             "verdict": "clean", "rules": [], "coverage_gaps": []}
    entry.update(over)
    return entry


@pytest.fixture
def log(service):
    """Publish a verdict log: head, a checkpoint and the shards it names."""
    base = verify.SERVICE

    def publish(entries, *, run=7, head=None, predicate_type=verify.CHECKPOINT_TYPE, signed_hours_ago=1,
                shard_bytes=None):
        shard = json.dumps({HASH: entries} if entries else {}).encode()
        shards = {f"{i:02x}": b"{}" for i in range(256)}
        shards[HASH[:2]] = shard
        st = {"_type": receipt.STATEMENT_TYPE, "predicateType": predicate_type,
              "subject": [{"name": k, "digest": {"sha256": hashlib.sha256(v).hexdigest()}}
                          for k, v in shards.items()],
              "predicate": {"run": run, "previous": None, "signed": "2026-10-31T00:00:00Z",
                            "scanner": {"name": "hayward", "version": "1.11.1"}}}
        signed = (datetime.now(timezone.utc) - timedelta(hours=signed_hours_ago)).isoformat()
        pub = service["published"]
        pub[f"{base}/v1/head.json"] = json.dumps({"checkpoint": run if head is None else head}).encode()
        pub[f"{base}/v1/checkpoints/{run if head is None else head}.sigstore.json"] = json.dumps(
            {"mediaType": "x", "payload": json.dumps(st), "signed_at": signed}).encode()
        for v in shards.values():
            pub[f"{base}/v1/shards/{hashlib.sha256(v).hexdigest()}.json"] = v
        if shard_bytes is not None:
            pub[f"{base}/v1/shards/{hashlib.sha256(shard).hexdigest()}.json"] = shard_bytes

    return publish


def test_a_clean_hash_in_the_current_checkpoint_passes(service, log, capsys):
    log([_entry()])
    assert cli.main(["verify", "--sha256", HASH]) == 0
    out = capsys.readouterr().out
    assert "clean" in out and f"org/name@{SHA[:12]}" in out and "model.safetensors" in out and "1.11.1" in out
    assert service["signer"] == (verify.IDENTITY, verify.ISSUER, "schedule", verify.REPOSITORY_ID, verify.OWNER_ID)


def test_a_hash_with_no_entry_is_not_a_pass(log, capsys):
    log([])
    assert cli.main(["verify", "--sha256", HASH]) == 1
    out = capsys.readouterr().out
    assert "no current verdict" in out and "not a pass" in out


def test_a_tampered_shard_is_refused(log, capsys):
    log([_entry()], shard_bytes=json.dumps({HASH: [_entry()], "x": []}).encode())
    assert cli.main(["verify", "--sha256", HASH]) != 0
    assert "does not match the checkpoint" in capsys.readouterr().err


def test_a_checkpoint_of_another_type_is_refused(log):
    log([_entry()], predicate_type=receipt.PREDICATE_TYPE)
    assert cli.main(["verify", "--sha256", HASH]) != 0


def test_a_checkpoint_for_another_run_than_head_is_refused(log, capsys):
    log([_entry()], run=7, head=8)
    assert cli.main(["verify", "--sha256", HASH]) != 0
    assert "run 7" in capsys.readouterr().err


def test_a_checkpoint_over_3_days_old_does_not_count(log, capsys):
    log([_entry()], signed_hours_ago=3 * 24 + 1)
    assert cli.main(["verify", "--sha256", HASH]) == 1
    assert "3 days" in capsys.readouterr().out


def test_a_gap_fails_unless_gaps_are_allowed(log, capsys):
    log([_entry(verdict="not_verified", coverage_gaps=["MFV-SKIP-001"])])
    assert cli.main(["verify", "--sha256", HASH]) == 1
    assert "--allow-gaps" in capsys.readouterr().out
    assert cli.main(["verify", "--sha256", HASH, "--allow-gaps"]) == 0


def test_a_clean_entry_with_coverage_gaps_is_a_gap(log):
    log([_entry(coverage_gaps=["MFV-SKIP-003"])])
    assert cli.main(["verify", "--sha256", HASH]) == 1
    assert cli.main(["verify", "--sha256", HASH, "--allow-gaps"]) == 0


def test_every_entry_must_be_clean(log):
    log([_entry(), _entry(repo="other/repo", verdict="remote_code")])
    assert cli.main(["verify", "--sha256", HASH]) == 1


def test_a_flagged_entry_never_passes(log):
    log([_entry(verdict="flagged")])
    assert cli.main(["verify", "--sha256", HASH, "--allow-gaps"]) == 1


def test_control_characters_in_an_entry_cannot_forge_output(log, capsys):
    log([_entry(verdict="not_verified", name="x.bin\nverdict verified")])
    assert cli.main(["verify", "--sha256", HASH]) == 1
    assert "x.bin?verdict verified" in capsys.readouterr().out


def test_a_header_read_entry_says_so(log, capsys):
    log([_entry(read="header", hash_source="hub")])
    assert cli.main(["verify", "--sha256", HASH]) == 0
    out = capsys.readouterr().out
    assert "clean (header read)" in out and "hash from the Hub" in out


@pytest.mark.parametrize("argv", [[], ["org/name", "--sha256", HASH], ["--sha256", HASH.upper()],
                                  ["--sha256", HASH[:-1]], ["--sha256", HASH, "./snap"],
                                  ["--sha256", HASH, "--revision", "v1"]])
def test_exactly_one_target_and_a_well_formed_hash(service, argv):
    assert cli.main(["verify", *argv]) == 2
    assert service["fetched"] == []
