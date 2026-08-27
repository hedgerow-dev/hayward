"""Tests for hayward.cache: content-hash scan cache (HW-153 part B).

The scanner is never imported. scan_callable is a fake, and a spy that raises
if called proves a hit never re-scans.
"""

from __future__ import annotations

import pytest

from hayward.cache import ScanCache
from hayward.findings import Category, Finding, Severity


def _finding(file_path: str, rule_id: str = "MFV-PICKLE-001") -> Finding:
    return Finding(
        rule_id=rule_id,
        message="os.system call in pickle",
        severity=Severity.CRITICAL,
        category=Category.DESERIALIZATION,
        file_path=file_path,
        confidence=0.9,
        cwe_ids=[502],
        metadata={"ref": "os.system"},
    )


def _write(tmp_path, name: str, data: bytes):
    path = tmp_path / name
    path.write_bytes(data)
    return path


def test_miss_scans_and_hit_does_not(tmp_path):
    path = _write(tmp_path, "model.pkl", b"\x80\x04.")
    cache = ScanCache(version_tag="1.0.0")

    calls = []

    def scan(p):
        calls.append(p)
        return [_finding(str(p))]

    first = cache.get_or_scan(path, scan)
    assert len(calls) == 1
    assert [f.rule_id for f in first] == ["MFV-PICKLE-001"]

    def spy(_p):
        raise AssertionError("hit must not call scan_callable")

    second = cache.get_or_scan(path, spy)
    assert len(calls) == 1  # unchanged: second run was a hit
    assert [f.rule_id for f in second] == ["MFV-PICKLE-001"]


def test_hit_round_trips_the_finding_fields(tmp_path):
    path = _write(tmp_path, "model.pkl", b"\x80\x04.")
    cache = ScanCache(version_tag="1.0.0")

    original = _finding(str(path))
    cache.get_or_scan(path, lambda _p: [original])

    (restored,) = cache.get_or_scan(path, lambda _p: pytest.fail("should hit"))
    assert restored.rule_id == original.rule_id
    assert restored.message == original.message
    assert restored.severity == Severity.CRITICAL
    assert restored.category == Category.DESERIALIZATION
    assert restored.file_path == str(path)
    assert restored.confidence == 0.9
    assert restored.cwe_ids == [502]
    assert restored.metadata == {"ref": "os.system"}


def test_changed_file_content_misses(tmp_path):
    path = _write(tmp_path, "model.pkl", b"\x80\x04.")
    cache = ScanCache(version_tag="1.0.0")

    calls = []

    def scan(p):
        calls.append(p.read_bytes())
        return [_finding(str(p))]

    cache.get_or_scan(path, scan)
    assert len(calls) == 1

    # Same path, different bytes: the content hash changes, so it misses even
    # though nothing about the name or location moved.
    path.write_bytes(b"\x80\x04N.")
    cache.get_or_scan(path, scan)
    assert len(calls) == 2


def test_different_version_tag_invalidates(tmp_path):
    path = _write(tmp_path, "model.pkl", b"\x80\x04.")
    cache_path = tmp_path / "cache.json"

    old = ScanCache(version_tag="1.0.0")
    old.get_or_scan(path, lambda _p: [_finding(str(path))])
    old.save(cache_path)

    # A newer scanner: the cache file on disk was written under 1.0.0, so
    # loading it under 1.1.0 must drop every entry and force a re-scan.
    reloaded = ScanCache.load(cache_path, version_tag="1.1.0")
    assert reloaded.entries == {}

    calls = []
    reloaded.get_or_scan(path, lambda p: calls.append(p) or [_finding(str(p))])
    assert len(calls) == 1


def test_same_version_tag_hits_after_load(tmp_path):
    path = _write(tmp_path, "model.pkl", b"\x80\x04.")
    cache_path = tmp_path / "cache.json"

    first = ScanCache(version_tag="1.0.0")
    first.get_or_scan(path, lambda _p: [_finding(str(path))])
    first.save(cache_path)

    reloaded = ScanCache.load(cache_path, version_tag="1.0.0")
    (restored,) = reloaded.get_or_scan(path, lambda _p: pytest.fail("should hit"))
    assert restored.rule_id == "MFV-PICKLE-001"


def test_save_load_round_trip_on_tmp_path(tmp_path):
    path = _write(tmp_path, "model.pkl", b"\x80\x04.")
    cache_path = tmp_path / "cache.json"

    cache = ScanCache(version_tag="1.0.0")
    cache.get_or_scan(path, lambda _p: [_finding(str(path))])
    cache.save(cache_path)

    reloaded = ScanCache.load(cache_path, version_tag="1.0.0")
    assert reloaded.version_tag == "1.0.0"
    assert reloaded.entries == cache.entries


def test_missing_cache_file_loads_empty(tmp_path):
    reloaded = ScanCache.load(tmp_path / "does-not-exist.json", version_tag="1.0.0")
    assert reloaded.entries == {}
    assert reloaded.version_tag == "1.0.0"


def test_corrupt_cache_file_loads_empty(tmp_path):
    cache_path = tmp_path / "cache.json"
    cache_path.write_text("{not json", encoding="utf-8")
    reloaded = ScanCache.load(cache_path, version_tag="1.0.0")
    assert reloaded.entries == {}


def test_empty_findings_are_cached_as_a_hit(tmp_path):
    """A clean file (no findings) must still be a cache hit, not re-scanned
    forever because an empty list looks falsy."""
    path = _write(tmp_path, "clean.pkl", b"\x80\x04.")
    cache = ScanCache(version_tag="1.0.0")

    calls = []
    cache.get_or_scan(path, lambda p: calls.append(p) or [])
    cache.get_or_scan(path, lambda _p: pytest.fail("clean file must hit"))
    assert len(calls) == 1


def test_default_version_tag_is_the_package_version():
    from hayward import __version__

    assert ScanCache().version_tag == __version__
