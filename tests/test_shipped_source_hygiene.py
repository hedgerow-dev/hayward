"""Issue #17: the shipped package must not carry literal attack payloads in its text.

Vulnetix flagged hayward as malware for docstrings and comments that quoted the attacks
it detects. Describe attacks in words under hayward/; keep literal payloads in tests/
and docs/, which are not in the wheel.
"""

import re
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent / "hayward"
PATTERNS = [
    re.compile(r"(curl|wget)[^\n]*\|\s*(ba)?sh"),   # download piped into a shell
    re.compile(r"/etc/(passwd|shadow)"),            # reading a system credential file
    re.compile(r"rm -rf /"),
    re.compile(r"bash -i|/dev/tcp/|\bnc -e"),         # reverse shells
]


def test_no_literal_attack_payloads_in_the_package():
    hits = []
    for path in sorted(PACKAGE.rglob("*.py")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if any(p.search(line) for p in PATTERNS):
                hits.append(f"{path.name}:{number}: {line.strip()[:80]}")
    assert hits == [], "\n".join(hits)
