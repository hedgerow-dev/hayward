"""Correctness properties of the scan cache that are easy to regress.

Two of these guard fixes made when the cache was wired into the CLI:

- Hashing an unreadable file must not turn one bad file into a hard failure.
  Without the cache, the scanner's per-file firewall degrades an unreadable
  file to a coverage gap and the directory walk continues (HW-104). The cache
  hashes a file before scanning it, so it must fall back to a direct scan when
  the hash cannot be taken, or it would reintroduce crash-as-evasion.

- The cache keys on file bytes, but --check-signatures adds findings that
  depend on sibling files, not on the model's bytes. The CLI folds the config
  into the cache's version tag, so a run with different flags must miss a cache
  written under another setting rather than trust it.
"""

from __future__ import annotations

import json
from pathlib import Path

from hayward import cli
from hayward.cache import ScanCache
from hayward.findings import Category, Finding, Severity


def _short_binunicode(text: str) -> bytes:
    raw = text.encode()
    return bytes([0x8C, len(raw)]) + raw


def _os_system_pickle(command: str) -> bytes:
    return (
        b"\x80\x04"
        + _short_binunicode("os")
        + _short_binunicode("system")
        + b"\x93"
        + _short_binunicode(command)
        + b"\x85"
        + b"R."
    )


class TestCacheFirewall:
    def test_unhashable_file_falls_back_to_a_direct_scan(self):
        """A path that cannot be hashed (here, one that does not exist) must
        not raise out of get_or_scan: it degrades to calling the scan
        callable, whose own firewall owns the unreadable case."""
        sentinel = Finding(
            rule_id="MFV-SKIP-003", message="degraded", severity=Severity.LOW,
            category=Category.AI_ML, file_path="/nope",
        )
        cache = ScanCache()
        result = cache.get_or_scan(Path("/does/not/exist"), lambda _p: [sentinel])
        assert result == [sentinel]
        # It was not cached (nothing to key it under), so a second call scans
        # again rather than serving a stale entry.
        calls = {"n": 0}

        def counting(_p):
            calls["n"] += 1
            return [sentinel]

        cache.get_or_scan(Path("/does/not/exist"), counting)
        assert calls["n"] == 1

    def test_unreadable_file_under_cache_does_not_abort_the_scan(self, tmp_path):
        """A directory scan with --cache over a good file and an unreadable one
        still exits on the good file's findings, not with a crash (exit 2)."""
        models = tmp_path / "models"
        models.mkdir()
        (models / "evil.pkl").write_bytes(_os_system_pickle("id"))
        # A directory named like a model file cannot be read as bytes: opening
        # it raises OSError, which is the unreadable case the firewall owns.
        (models / "trap.pkl").mkdir()
        cache_file = tmp_path / "cache.json"
        rc = cli.main(["scan", str(models), "--cache", str(cache_file),
                       "--fail-on", "high"])
        assert rc == 1  # the malicious file still fails the build, no crash


class TestCacheKeyedByPath:
    def test_byte_identical_files_at_different_paths_are_both_reported(self, tmp_path, capsys):
        """Two byte-identical malicious files must each be reported under their
        own path. A cache keyed on content alone would serve the first file's
        result (and path) for the second, hiding it."""
        models = tmp_path / "models"
        models.mkdir()
        (models / "a.pkl").write_bytes(_os_system_pickle("id"))
        (models / "b.pkl").write_bytes(_os_system_pickle("id"))  # identical bytes
        cache = tmp_path / "cache.json"

        # First run populates the cache; second run is served from it.
        cli.main(["scan", str(models), "--cache", str(cache), "-f", "json",
                  "--fail-on", "never"])
        capsys.readouterr()
        cli.main(["scan", str(models), "--cache", str(cache), "-f", "json",
                  "--fail-on", "never"])
        data = json.loads(capsys.readouterr().out)
        flagged = {f["file"] for f in data["findings"] if f["rule_id"] == "MFV-PICKLE-001"}
        assert flagged == {str(models / "a.pkl"), str(models / "b.pkl")}


class TestAllowlistFirewall:
    def test_unreadable_file_under_allowlist_does_not_abort(self, tmp_path):
        """--allowlist hashes each finding's file; an unreadable one must leave
        its finding standing, not abort the run with exit 2."""
        models = tmp_path / "models"
        models.mkdir()
        (models / "evil.pkl").write_bytes(_os_system_pickle("id"))
        (models / "trap.pkl").mkdir()  # a directory named like a file: OSError on read
        allow = tmp_path / "allow.json"
        allow.write_text(json.dumps([]))  # empty allowlist, but the apply path still hashes
        rc = cli.main(["scan", str(models), "--allowlist", str(allow),
                       "--fail-on", "high"])
        assert rc == 1  # the malicious file still fails; no crash to exit 2


class TestCacheConfigNamespacing:
    def test_signatures_run_is_not_served_to_a_plain_run(self, tmp_path, capsys):
        model = tmp_path / "model.pkl"
        model.write_bytes(_os_system_pickle("id"))
        (tmp_path / "model.pkl.sigstore.json").write_text(json.dumps({
            "mediaType": "application/vnd.dev.sigstore.bundle+json;version=0.3",
            "verificationMaterial": {"certificate": {"rawBytes": "AA=="}},
        }))
        cache_file = tmp_path / "cache.json"

        # Run 1: with --check-signatures, so MFV-SIG-001 is produced and cached.
        cli.main(["scan", str(model), "--check-signatures", "--cache", str(cache_file),
                  "-f", "json", "--fail-on", "never"])
        first = json.loads(capsys.readouterr().out)
        assert any(f["rule_id"] == "MFV-SIG-001" for f in first["findings"])

        # Run 2: same cache file, but WITHOUT --check-signatures. The file bytes
        # are unchanged, so a bytes-only cache would wrongly replay MFV-SIG-001.
        # The config-namespaced tag makes it miss and rescan clean of it.
        cli.main(["scan", str(model), "--cache", str(cache_file),
                  "-f", "json", "--fail-on", "never"])
        second = json.loads(capsys.readouterr().out)
        assert not any(f["rule_id"] == "MFV-SIG-001" for f in second["findings"])
