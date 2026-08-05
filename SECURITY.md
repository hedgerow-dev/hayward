# Security policy

## Reporting a vulnerability

Email hello@hedgerow.dev. Please do not open a public issue for a security
report.

Include what you have: a file that reproduces it, the version, and what you
expected. A minimal reproduction is worth more than a description.

## What counts

**Input that crashes the scanner is a security issue.** Hayward reads hostile
files by design. It does not execute them, but it is a parser, and under a CI
gate an unhandled exception is indistinguishable from a scan that never ran.
An attacker who can crash the scanner has bypassed it.

**A file that scans clean and should not** is a security issue. Include the
file or a script that builds it.

**A file that scans clean because Hayward could not read it** is a bug in the
coverage reporting rather than a bypass, and still worth reporting. Every
parse failure is supposed to produce a finding. If one does not, that is the
defect.

## What does not count

Findings on files that are genuinely unusual are not vulnerabilities. The INFO
tier exists for content the scanner cannot verify, and it is meant to be
populated. If you think a rule is too noisy on real models, open an issue with
the model rather than a security report.

## Handling

We will acknowledge within a few working days, agree a disclosure date with
you, and credit you in the changelog unless you would rather we did not.
