"""Tests for MFV-PICKLE-007: a disproportionate hole in a pickle stream's
memo index run.

Splicing a payload into an existing PyTorch `data.pkl` (ShadowPickle,
arXiv:2607.17503) leaves a structural artefact that has nothing to do with
what the payload does: to keep the spliced-in memo writes from clobbering
the host model's, the injector rewrites every `BINPUT i` in the payload to
`LONG_BINPUT i + <large constant>`. The resulting stream stores into a few
slots numbered in the hundreds of millions alongside the host's own few
hundred, with nothing in between.

Python's picklers number memo slots consecutively (`idx = len(self.memo)` in
CPython's `Pickler.memoize`), so one pickle's indices always form a single
unbroken run and that shape has no benign producer. The detector keys on the
*hole*, never on a particular offset, which is what these tests pin down:

  (a) a spliced stream is flagged whatever offset the splicer picked;
  (b) a spliced stream is flagged even when the payload's callables are
      entirely allow-listed, because the evidence is about how the file was
      assembled rather than about what it executes;
  (c) real pickles, at every protocol and including the large ones whose
      memo legitimately runs to thousands of slots and therefore uses
      LONG_BINPUT normally, are not flagged;
  (d) neither is the one benign shape that does reach a large absolute index,
      a `Pickler` shared across many `dump()` calls, which is why the check
      is on the span-to-slots ratio and not on the index itself.
"""

from __future__ import annotations

import io
import pickle
import re
import struct

import pytest

from hayward.findings import Severity
from hayward.scanner import (
    ModelFileScanner,
    _profile_memo_indices,
    _resolve_pickle_globals,
)


class _Payload:
    """Stand-in for a spliced payload. `str` rather than something dangerous
    on purpose: MFV-PICKLE-007 must fire on the splicing artefact alone, with
    no help from the denied-globals list."""

    def __reduce__(self):
        return (str, ("payload",))


def _splice_block(payload: bytes, offset: int) -> bytes:
    """Apply the ShadowPickle transform to a payload pickle: strip PROTO and
    STOP so the block can sit mid-stream, offset every memo write, and wrap
    the whole thing in MARK ... POP_MARK so its stack effects are discarded
    and the host still deserializes.

    The `BINPUT i` -> `LONG_BINPUT i + offset` rewrite is the released
    injector's, generalized to an arbitrary offset so the tests can prove the
    detector isn't keyed to one tool's constant.
    """
    body = payload
    if body[:1] == b"\x80":
        body = body[2:]
    if body[-1:] == b".":
        body = body[:-1]
    body = re.sub(
        rb"q(.)",
        lambda m: b"r" + struct.pack("<I", m.group(1)[0] + offset),
        body,
        flags=re.DOTALL,
    )
    return b"(" + body + b"1"


def _spliced_stream(host_obj, offset: int) -> bytes:
    """A protocol-2 host pickle with a memo-offset payload block spliced in
    just after PROTO, the way an injector splices at an opcode boundary."""
    host = pickle.dumps(host_obj, protocol=2)
    block = _splice_block(pickle.dumps(_Payload(), protocol=2), offset)
    return host[:2] + block + host[2:]


def _benign_host() -> dict:
    """Big enough to memoize a few hundred slots, so the host's own memo is
    a realistic denominator rather than a handful of entries."""
    return {f"layer.{i}.weight": [float(i)] * 3 for i in range(300)}


class TestOutOfBandMemoIndices:
    def test_spliced_stream_is_flagged(self, tmp_path):
        """The published transform, at the offset the paper reports."""
        data = _spliced_stream(_benign_host(), 0x11100000)
        path = tmp_path / "model.pkl"
        path.write_bytes(data)

        findings = ModelFileScanner().scan_file(path)
        oob = [f for f in findings if f.rule_id == "MFV-PICKLE-007"]
        assert oob, f"expected MFV-PICKLE-007, got {[f.rule_id for f in findings]}"
        assert oob[0].severity == Severity.HIGH
        assert oob[0].metadata["out_of_band_count"] >= 1
        assert oob[0].metadata["max_memo_index"] >= 0x11100000

    @pytest.mark.parametrize("offset", [0x11100000, 0x40000000, 10_000_000, 100_000])
    def test_detection_is_not_tied_to_one_offset(self, offset):
        """A splicer choosing any other large offset trips the same check --
        the discriminator is the disproportion, not the constant."""
        _globals, _calls, profile = _resolve_pickle_globals(
            _spliced_stream(_benign_host(), offset)
        )
        assert profile.out_of_band, f"offset {offset} went undetected"
        assert min(profile.out_of_band) >= offset

    def test_indices_are_judged_per_pickle_not_across_the_file(self):
        """torch's legacy format concatenates several pickles in one file, and
        a real ``Unpickler`` starts each `load()` with an empty memo. So each
        pickle is profiled on its own rather than the file's indices being
        unioned.

        This is the case that makes the difference: a splice into a small
        pickle followed by a large state_dict. Judged per pickle the small
        one's 7 slots make the hole enormous; unioned, the state_dict's slot
        count raises the threshold above the hole and the splice disappears.
        """
        spliced = _spliced_stream({f"w{i}": i for i in range(3)}, 100_000)
        big = pickle.dumps({f"k{i}": (i, str(i)) for i in range(5000)}, protocol=2)

        _globals, _calls, profile = _resolve_pickle_globals(spliced + big)
        assert profile.out_of_band
        assert min(profile.out_of_band) >= 100_000

        # The union of both pickles' indices is what accumulating would have
        # profiled, and it does not flag.
        _g2, _c2, alone = _resolve_pickle_globals(big)
        union = set(range(profile.slots)) | set(profile.out_of_band) | set(range(alone.slots))
        assert _profile_memo_indices(union).out_of_band == ()

    def test_fires_without_any_dangerous_global(self):
        """`str` is allow-listed, so nothing else in the scan reacts to this
        file. The splicing evidence has to stand on its own, which is the
        point of a payload-independent signal."""
        data = _spliced_stream(_benign_host(), 0x11100000)
        path_globals, _calls, profile = _resolve_pickle_globals(data)
        assert profile.out_of_band
        assert not any("system" in g or "subprocess" in g for g in path_globals)


class TestNoFalsePositives:
    @pytest.mark.parametrize("protocol", range(0, pickle.HIGHEST_PROTOCOL + 1))
    def test_ordinary_pickles_are_clean(self, protocol):
        """Every protocol, including the proto 0/1 `PUT`/`BINPUT` spellings
        and proto 4+'s implicit `MEMOIZE` counter."""
        data = pickle.dumps(_benign_host(), protocol=protocol)
        _globals, _calls, profile = _resolve_pickle_globals(data)
        assert profile.out_of_band == ()
        assert profile.max_index == profile.slots - 1

    def test_large_memo_using_long_binput_is_clean(self):
        """A model big enough that its own memo passes 256 slots emits
        LONG_BINPUT perfectly legitimately. "Uses LONG_BINPUT" must not be
        the signal; only an index out of proportion to the memo is."""
        data = pickle.dumps({f"k{i}": (i, str(i)) for i in range(5000)}, protocol=2)
        assert b"r" in data  # LONG_BINPUT is genuinely present
        _globals, _calls, profile = _resolve_pickle_globals(data)
        assert profile.slots > 5000
        assert profile.out_of_band == ()

    def test_shared_memo_across_dumps_is_clean(self):
        """A `Pickler` reused across many `dump()` calls carries its memo
        forward, so a later sub-stream owns only a handful of slots while
        numbering them in the thousands.

        This is a real false positive an absolute-index threshold produces:
        at 2000 dumps a sub-stream here writes 11 slots at indices past 4092,
        which clears any fixed ceiling low enough to be useful. Its run is
        unbroken, so the span-to-slots ratio stays at 1.0 and it must stay
        quiet.
        """
        buf = io.BytesIO()
        pickler = pickle.Pickler(buf, protocol=2)
        for i in range(2000):
            pickler.dump({f"k{i}_{j}": [j] for j in range(5)})

        _globals, _calls, profile = _resolve_pickle_globals(buf.getvalue())
        assert profile.out_of_band == ()

    def test_recursive_structures_are_clean(self):
        """Recursion drives the pickler down its POP_MARK/GET paths, which is
        the least ordinary opcode shape a real pickler emits."""
        node = {}
        node["self"] = node
        node["kids"] = [node, (node,)]
        for protocol in range(0, pickle.HIGHEST_PROTOCOL + 1):
            _globals, _calls, profile = _resolve_pickle_globals(
                pickle.dumps(node, protocol=protocol)
            )
            assert profile.out_of_band == (), f"protocol {protocol}"


class TestMemoProfile:
    def test_stream_with_no_memo_writes(self):
        """A pickle small enough that nothing gets memoized at all -- the
        profile must be empty rather than dividing by zero."""
        profile = _profile_memo_indices(set())
        assert profile.slots == 0
        assert profile.max_index == -1
        assert profile.out_of_band == ()

    def test_threshold_scales_with_slot_count(self):
        """A small stream's hole is judged against the absolute floor; a large
        one's against its own slot count, so a big model gets proportionally
        more headroom rather than a fixed ceiling. The same index 5000 is
        out of band in the first stream and unremarkable in the second."""
        assert _profile_memo_indices(set(range(10)) | {5000}).out_of_band == (5000,)
        assert _profile_memo_indices(set(range(10_000)) | {5000}).out_of_band == ()
