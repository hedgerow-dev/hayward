# GitHub Action

`action.yml` at the repo root packages `hayward scan` as a composite GitHub
Action. It scans your model files, writes a SARIF report, and sets the build's
exit status from the findings. Pair it with `github/codeql-action/upload-sarif`
to send the results to GitHub code scanning, where they appear on the Security
tab and inline on pull requests.

## Usage

Add a workflow like this to a consuming repository:

```yaml
name: Model scan
on:
  push:
    paths:
      - "models/**"
  pull_request:

permissions:
  contents: read
  security-events: write

jobs:
  hayward:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Scan model files
        id: scan
        uses: hedgerow-dev/hayward@v1
        with:
          path: models
          fail-on: high
          output: hayward.sarif

      - name: Upload SARIF
        if: always()
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: ${{ steps.scan.outputs.sarif }}
```

`security-events: write` is required for the upload step to publish results.
`if: always()` runs the upload even when the scan step fails the build (see
Exit status), so a failing scan still records its findings on the Security tab.

## Inputs

| Input              | Required | Default          | Description                                                                                             |
| ------------------ | -------- | ---------------- | ------------------------------------------------------------------------------------------------------- |
| `path`             | yes      |                  | File or directory of model files to scan.                                                               |
| `fail-on`          | no       | `high`           | Lowest severity that fails the build: `critical`, `high`, `medium`, `low`, `info`, or `never`.          |
| `fail-on-coverage` | no       | `false`          | Set to `true` to also fail when a file could not be fully read.                                         |
| `lock`             | no       | (empty)          | A `hayward lock` file. The path must match it exactly, or the build fails before the scan. See below.   |
| `receipt`          | no       | (empty)          | Path for a scan receipt (hayward 1.9.0 or later). See below.                                            |
| `attest`           | no       | `false`          | Set to `true` to sign the receipt with GitHub artifact attestations. See below.                         |
| `output`           | no       | `hayward.sarif`  | Path for the SARIF report.                                                                              |
| `version`          | no       | (empty)          | Version of the `hayward` package to install from PyPI. See Version below.                               |

### Verify against a lock

With `lock`, the action first runs `hayward lock --verify` on `path`: every file
in the lock must be present with the same hash, and nothing else may be there,
so an extra `.py` or a swapped weight file fails the build before anything is
scanned. Make the lock with `hayward lock org/name -o hayward.lock.json` (hayward
1.5.0 or later) and commit it next to the workflow. A lock that needs review
never verifies. A lock pins what is loaded; the scan that follows still checks
what those files do.

```yaml
      - uses: hedgerow-dev/hayward@v1
        with:
          path: models/my-model
          lock: models/my-model.lock.json
```

### Signed scan receipts

With `receipt`, the scan also writes an in-toto statement of each file's hash
and what the scanner found (see [usage](usage.md#scan-receipts)). With
`attest: true` as well, the action signs it with
[`actions/attest`](https://github.com/actions/attest): keyless Sigstore, stored
with the repository's attestations. Signing runs even when the scan fails the
build, since a receipt for a flagged file is still evidence. The job needs two
permissions:

```yaml
    permissions:
      contents: read
      id-token: write
      attestations: write
    steps:
      - uses: actions/checkout@v4
      - uses: hedgerow-dev/hayward@v1
        with:
          path: models
          receipt: hayward.receipt.json
          attest: "true"
```

Check a signed receipt later with
`gh attestation verify models/model.safetensors --repo OWNER/REPO --predicate-type https://hedgerow.dev/hayward/scan-receipt/v1`.

## Outputs

| Output    | Description                                        |
| --------- | -------------------------------------------------- |
| `sarif`   | Path to the SARIF report from the scan.            |
| `receipt` | Path to the scan receipt, when one was requested.  |

## Exit status

The scan step follows Hayward's exit codes: `0` when nothing reaches the
`fail-on` threshold, `1` when a finding does (or when `fail-on-coverage` is
`true` and a file could not be fully read), and `2` when the scan itself could
not run. Exit `1` fails the job, which is what gates a pull request. Keep the
upload step on `if: always()` so the SARIF is published either way.

## Version

The action installs the `hayward` package from PyPI. An empty `version`
installs the latest release; set `version` to pin one, for example `1.1.0`.
Pinning is recommended for a CI gate, so a result cannot change under you when a
new release ships.

The action is pinned to a major version by its tag: `hedgerow-dev/hayward@v1`
tracks the latest `v1.x` of the action, or pin a full tag such as
`hedgerow-dev/hayward@v1.1.0` for an exact, reproducible action.
