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
| `output`           | no       | `hayward.sarif`  | Path for the SARIF report.                                                                              |
| `version`          | no       | (empty)          | Version of the `hayward` package to install from PyPI. See Version below.                               |

## Outputs

| Output  | Description                              |
| ------- | ---------------------------------------- |
| `sarif` | Path to the SARIF report from the scan.  |

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
