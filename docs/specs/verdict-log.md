# Hayward verdict log: spec

Status: draft for Kenneth, 2026-10-31. Spec only, nothing built.

## 1. Problem

- Hayward can tell you a model file is safe to load, but only if you run it first.
- hayward-verify already publishes signed verdicts per repo commit, but only for the current top 100, and each deploy wipes the last.
- Nobody can ask "has anyone vouched for these exact bytes?" by file hash.
- Nothing proves a verdict was not quietly edited or removed, or tells a client a verdict was withdrawn.
- A loader hook and Hub-wide scanning (both out of scope) need all four fixed first.

## 2. What exists today

Checked by reading the code, not the brief.

- **Receipts** (`hayward/receipt.py`): an in-toto Statement per scan. Subjects are file names and SHA-256s. Per file: `verdict` (`clean`, `not_verified`, `remote_code`, `flagged`), `rules`, `coverage_gaps`, and `read: header` for big SafeTensors. Docstring rule already: receipts are never rewritten; the newest one for a digest is the current word.
- **Signing**: `hayward receipt sign` wraps the receipt in a Sigstore DSSE bundle (`_sigstore_sign`). The bundle carries its Rekor inclusion proof, so it verifies offline.
- **Client** (`hayward/verify.py`): `hayward verify org/name` fetches `verify.hedgerow.dev/v1/hf/{repo}/{commit}.sigstore.json`, pins the signer (scheduled `scan.yml` on `main`, repo id `1440229852`, owner id `289219779`), takes age from the certificate (30 days max), and checks hashes against the Hub. Fails on gaps unless `--allow-gaps`. Notes when the scanner is older than the client.
- **Remote reads** (`hayward/hub.py`): big zip checkpoints read for their pickle members only, big SafeTensors for the header only. Those files get `hash_source: hub`: the SHA-256 is the Hub's claim, not one Hayward computed.
- **Service** (`~/Developer/projects/hayward-verify`, `verify_service.py`, `scan.yml`): every 6 hours, top 100 by downloads. Three jobs with one credential each. Flagged repos are withheld into an age-encrypted, padded report; logs carry totals only. Each run re-signs all ~100 bundles and deploys a **fresh Firebase Hosting version**, so anything not re-signed disappears.
- **Pilot** (`scripts/hub_pilot_inventory.py`, `scripts/hub_pilot_scan.py`): exist only as **untracked files in the main checkout**, not in git. Inventory counts unique SHA-256s across top models; scan runs `hayward hub` on pickle-type repos.
- **Decisions already made** (project memory, 2026-10-30): publish only repos with no flagged file; flagged goes to private review and a report to Hugging Face; a signed "withheld" marker after a 30-day window is planned, not built.
- No `CONTEXT.md` or `docs/adr/` exists yet.

## 3. The design

### Recommended: a signed checkpoint per run, over static files

Keep receipts as they are. Add one signed file per run that says "these are the current verdicts", and index the rest by hash.

Files on the existing Firebase site (all static):

| Path | Signed? | What it is |
|---|---|---|
| `/v1/receipts/{sha256}.json` | No, content-addressed | One receipt, named by the SHA-256 of its bytes. Never changed or deleted. |
| `/v1/shards/{sha256}.json` | No, content-addressed | One of 256 shards (by first two hex chars of a file hash). Maps file hash to a list of `{receipt, repo, commit, name, verdict, read, coverage_gaps, scanner}`. |
| `/v1/checkpoints/{n}.sigstore.json` | **Yes** | Run number `n`, time, scanner version, SHA-256 of checkpoint `n-1`, the 256 shard digests, and a list of withheld repo commits (after their window). |
| `/v1/head.json` | No | The latest `n`. A hint only. |
| `/v1/hf/{repo}/{commit}.json` | No | Pointer: repo commit to receipt digest. |
| `/v1/hf/{repo}/{commit}.sigstore.json` | Yes | Today's per-repo bundle, kept for 1.11 clients until they age out. |

Lookup of one file hash: fetch `head.json`, the checkpoint (check signature, signer, age), the one shard it names, the receipt it names. Every step after the checkpoint is checked by digest, so a lying shard or pointer can only cause a miss, never a pass.

Why this shape:

- **One Rekor entry per run**, not one per repo. Today's design re-signs 100 repos every 6 hours; at the measured ~4,700 new repos a day (knowledge note `hayward-verify-every-new-upload.md`, 2026-10-31) per-repo signing is the stated Sigstore volume risk.
- **Freshness and withdrawal for free.** "Current" means "in a checkpoint signed in the last N days". Dropping a receipt from the next checkpoint withdraws it, without editing anything.
- **Deletion shows.** Checkpoints chain by digest and each is in Rekor with a time.
- **No server.** Same hosting, same three-job workflow, standard library only in deploy.

### Strongest rejected alternative: run our own transparency log

A Rekor or Tessera instance of our own, one entry per file verdict. It gives inclusion proofs per verdict and a real Merkle log. Rejected because it is a running service with storage, keys and uptime to own, and the public Rekor already gives append-only timestamps for the one thing we need to anchor: the checkpoint. Revisit only if a third party needs per-verdict inclusion proofs.

Also rejected: **keep per-repo bundles and add a hash index.** Smallest change, but it has no withdrawal, no deletion detection, and keeps the Rekor volume problem.

## 4. Answers to the eight questions

### 4.1 Unit of record

- **Both, with one source of truth.** The receipt is per repo commit (it binds names, hashes and commit). The shards index it per file hash.
- **Repo question** (`org/name@commit`): pointer gives the receipt; the client checks the receipt digest appears in the current checkpoint's shard for any of its subject hashes, then runs today's `verify.check` against the Hub file list.
- **Loader question** (one file): the hook has the Hub's LFS SHA-256 before download, or hashes the bytes after. Small non-LFS files (configs, `.py`) have only a git blob id on the Hub, so the hook must hash them after download, before load.
- **Same bytes, many repos**: a file verdict depends only on bytes, scanner version and settings, so one hash can carry entries from many repos. A hash is "clean" only if every current entry for it is.

### 4.2 What is published

- `clean`: "No known attack found in the parts read." Never "safe".
- `not_verified`: "Not fully read; not vouched for." Reason from `coverage_gaps`.
- `remote_code`: "Runs code from the repo (`trust_remote_code`); that code is not vouched for."
- `flagged`: **not published.** Keep the rule. No argument found for changing it.
- **"We looked and will not vouch"**: after the 30-day window, the checkpoint lists the repo commit as `withheld`. Repo and commit only: no file names, hashes or rule ids. By-hash lookups for the flagged file just miss, and a miss is never a pass.

### 4.3 Trust

- **Signer**: the scheduled `scan.yml` run, pinned exactly as today (identity, issuer, `schedule` trigger, repo id, owner id).
- **Rotation**: identities are keyless, so rotation means a new workflow path or repo. The client carries a list of accepted signers; a new one ships in a Hayward release while the old one stays for one overlap period, then is removed. Sigstore root rotation is handled by sigstore-python's TUF updates.
- **Stale vs current**: a checkpoint older than N days does not count. Today N is 30 for receipts; with a checkpoint every 6 hours, a much shorter N is possible (decision 3 below). The receipt's own age stops mattering: a receipt is current while a fresh checkpoint lists it.
- **New rules**: each checkpoint records its scanner version. When a release adds rules, the service rescans everything current before listing it under the new version. A file the new version flags is dropped from the next checkpoint (withdrawn) and enters private review. Old receipts stay in the archive, not current.

### 4.4 Storage and query

- **Where**: the existing Firebase site `verify.hedgerow.dev`. No database, no server.
- **Append**: the deploy job must **carry the previous version's files forward**. Firebase reuses files by hash (`populateFiles`), so only new files upload. Today it replaces everything, which is the first bug to fix.
- **Cheap lookup**: four small GETs, three of them content-addressed and cacheable forever (`Cache-Control: immutable`).
- **Offline**: `hayward` can download all 256 shards for a checkpoint once, then answer any hash locally until the checkpoint ages out.
- **Unchecked**: whether Firebase Hosting caps file count per version. Check before the archive grows (milestone step 1).

### 4.5 Append-only proof

- Every checkpoint is signed and logged in Rekor, with an inclusion proof inside the bundle.
- Each checkpoint names the previous one's digest. A deleted or edited checkpoint breaks the chain; a forged replacement needs a new signature, which Rekor timestamps later and publicly.
- Receipts and shards are content-addressed, so an edit changes the name.
- **Limit**: Rekor proves what we signed and when; it does not stop us signing two different chains. A small monitor script (anyone can run it, including a second scheduled job) walks the chain and keeps a copy. Do not depend on searching Rekor by identity; treat that API as not guaranteed.

### 4.6 Coverage honesty

- Every shard entry carries `read` and `coverage_gaps`. A file with either is never shown as plain `clean`; the client prints `clean (header read)` or `not_verified`.
- `hash_source: hub` is copied into the entry. For a range-read file the hash is the Hub's claim; a loader that downloaded the file checks its own hash, so a mismatch fails.
- Files the scan did not cover are absent, and absence is never a pass. The client's existing "not covered, CODE" output stays.

### 4.7 Abuse and disputes

- **Owner disputes a withheld repo**: private vulnerability reporting on hayward-verify (already on). Kenneth reviews. A false positive is fixed in Hayward, released, rescanned, published. No manual overrides of verdicts.
- **Someone reports a clean file as malicious**: same channel. If credible, drop it from the next checkpoint (within 6 hours), then fix the rule.
- **Gaming the queue**: there is no public submission queue. The service picks repos from the Hub API itself, and the sign job re-checks that list (it already does for top-N). Inflating downloads only gets a repo scanned, which is fine. Flooding new uploads: cap repos per owner per run (needed once new uploads are scanned).
- **Swap after verdict**: not possible; verdicts bind to hashes and commits.
- **Evading the scanner**: the real risk. Answer is wording ("analysis evidence, not a safety certificate") plus rescans and withdrawal.

### 4.8 Cost

Backed facts:

- **Hosting**: Firebase Spark plan gives 10 GB storage and 10 GB/month transfer at no cost; over the storage cap, deploys stop. Blaze bills $0.026/GB storage and $0.15/GB transfer past those ([Firebase Hosting quotas and pricing](https://firebase.google.com/docs/hosting/usage-quotas-pricing)).
- **Signing**: Rekor's public instance is free, with a 99.5% availability objective ([sigstore/rekor README](https://github.com/sigstore/rekor)). Sigstore publishes no rate limit figure that I found.
- **Compute**: GitHub-hosted standard runners are free on public repos.
- **Measured**: scans took about 15 to 25 s per repo on the Mac mini research run (project memory, 2026-10-30); about 4,700 new model repos a day (knowledge note, 2026-10-31).

Pilot (top 100, every 6 hours): **$0**. Storage grows only when a repo gets a new commit.

Whole Hub, new uploads only (**estimate**): 4,700 repos x 15 to 25 s is about 20 to 33 runner-hours a day, still $0 on a public repo but needs sharding to fit job limits. Storage depends on receipt size, which is **not measured**; measure it in step 2 below. The full existing Hub backlog is not counted; the inventory script can count it.

## 5. Decisions for Kenneth

1. **Move signing to one checkpoint per run?** Per-repo bundles stay for 1.11 clients until they expire. **Recommend yes**: it fixes Rekor volume, freshness and withdrawal in one step.
2. **Withheld marker scope?** Repo commit only (no files, hashes or rules), or also list file hashes. **Recommend repo commit only**: hashes would point at the flagged file in public.
3. **How fresh must a checkpoint be?** **Recommend 3 days**: survives a missed run or two of the 6-hourly job, withdraws a bad verdict within days instead of 30.

## 6. First milestone (one week)

1. Deploy carries forward the previous version's files -> verify: test with a fake Hosting API; after two runs a receipt from the first still resolves. Check Firebase's per-version file limit.
2. Sign job writes content-addressed receipts and 256 shards -> verify: unit test that every subject hash maps to its receipt digest; log the measured receipt and shard sizes.
3. Sign job signs one checkpoint (new predicate type `https://hedgerow.dev/hayward/checkpoint/v1`) naming the previous one -> verify: test the chain; a live run puts one checkpoint in Rekor.
4. Client `hayward verify --sha256 HASH` reads head, checkpoint, shard, receipt -> verify: tests reject a tampered shard, a wrong signer, a stale checkpoint, and a `read: header` entry shown as plain clean.
5. Monitor script walks the checkpoint chain -> verify: test flags a missing or edited checkpoint.

Per-repo bundles and `hayward verify org/name` keep working unchanged throughout.
