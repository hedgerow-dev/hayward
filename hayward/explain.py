"""``hayward explain FILE``: what a pickle-based model file would do, as readable lines.

Nothing is loaded or run. The opcodes are read with ``pickletools`` and turned into
pseudo-Python, one line per call:

    _0 = builtins.__import__('subprocess')
        # why: on Hayward's deny list: can run code, commands, or reach files or the network
    _1 = builtins.getattr(_0, 'getoutput')
    _2 = _1('echo $(id)')
        # why: calls the result of an earlier line; argument looks like a shell command

Each callable is classified by the scanner's own allow and deny lists, so ``explain``
and ``scan`` cannot disagree about what is dangerous. Calls to known-safe functions
(tensor rebuilds, mostly) are hidden and counted unless ``--all`` is given.

Hostile input is expected: every limit below stops the walk with a note, never a crash,
and nothing derived from the file is ever evaluated.
"""

from __future__ import annotations

import io
import pickletools
import tarfile
import zipfile
from collections import Counter
from pathlib import Path

from ._pickle_engine import _PICKLE_ARG_SHELL_RE, _PICKLE_ARG_URL_RE, _classify_pickle_global

MAX_OPS = 2_000_000            # opcodes walked per pickle
MAX_STACK = 1_000_000          # values held on the stack
MAX_PICKLES = 8                # pickles read back to back from one stream
MAX_MEMBERS = 32               # pickle members read from one archive
MAX_FILE_BYTES = 512 * 1024 * 1024
MAX_STATEMENTS = 300           # lines printed per pickle
MAX_TEXT = 120                 # characters of a string shown
MAX_ITEMS = 8                  # items of a container shown
MAX_DEPTH = 6                  # nesting shown
MAX_LINE = 400                 # characters of one printed line

_MARK = object()


class Const:
    def __init__(self, value):
        self.value = value


class Glob:
    def __init__(self, ref: str):
        self.ref = ref
        self.level = _classify_pickle_global(ref)


class Call:
    def __init__(self, func, args: list, kwargs=None, rebuild: bool = False):
        self.func, self.args, self.kwargs = func, args, kwargs
        self.rebuild = rebuild  # NEWOBJ with no arguments: an object being reconstructed
        self.var: int | None = None


class Build:
    def __init__(self, obj, state):
        self.obj, self.state = obj, state
        self.var: int | None = None


class Seq:
    def __init__(self, kind: str, items: list):
        self.kind, self.items = kind, items


class Dct:
    def __init__(self, pairs: list):
        self.pairs = pairs


class Odd:
    """Something the walk cannot represent: shown as-is, never guessed at."""

    def __init__(self, text: str):
        self.text = text


_CONST_OPS = {"NONE", "NEWTRUE", "NEWFALSE", "INT", "BININT", "BININT1", "BININT2", "LONG",
              "LONG1", "LONG4", "FLOAT", "BINFLOAT", "STRING", "BINSTRING", "SHORT_BINSTRING",
              "BINBYTES", "SHORT_BINBYTES", "BINBYTES8", "BYTEARRAY8", "UNICODE", "BINUNICODE",
              "SHORT_BINUNICODE", "BINUNICODE8"}
_FIXED_VALUES = {"NONE": None, "NEWTRUE": True, "NEWFALSE": False}


class _Walk:
    """One pickle's opcodes, turned into nodes. Calls are recorded, never made."""

    def __init__(self):
        self.stack: list = []
        self.memo: dict = {}
        self.calls: list = []      # Call and Build nodes, in the order the file makes them
        self.result = None
        self.note = ""

    def pop(self):
        if self.stack and self.stack[-1] is not _MARK:
            return self.stack.pop()
        return Odd("<stack underflow>")

    def pop_mark(self) -> list:
        items: list = []
        while self.stack:
            value = self.stack.pop()
            if value is _MARK:
                items.reverse()
                return items
            items.append(value)
        items.reverse()
        return [Odd("<missing MARK>"), *items]

    def push(self, node) -> None:
        if len(self.stack) >= MAX_STACK:
            raise _Stop(f"the stack passed {MAX_STACK:,} values")
        self.stack.append(node)

    def record(self, node):
        self.calls.append(node)
        self.push(node)

    def run(self, stream: io.BytesIO) -> bool:
        """Walk to STOP. True if the pickle ended normally."""
        ops = 0
        try:
            for op, arg, _ in pickletools.genops(stream):
                ops += 1
                if ops > MAX_OPS:
                    raise _Stop(f"more than {MAX_OPS:,} opcodes")
                if self.step(op.name, arg):
                    return True
        except _Stop as exc:
            self.note = f"stopped: {exc}"
        except (ValueError, EOFError, UnicodeDecodeError) as exc:
            self.note = f"stopped at byte {stream.tell()}: {exc}"
        if not self.note:
            self.note = f"stopped at byte {stream.tell()}: the stream ends without STOP"
        return False

    def step(self, name: str, arg) -> bool:  # noqa: C901 - one branch per opcode family
        if name in _CONST_OPS:
            self.push(Const(_FIXED_VALUES[name] if name in _FIXED_VALUES else arg))
        elif name in ("PROTO", "FRAME"):
            pass
        elif name == "STOP":
            self.result = self.pop()
            return True
        elif name == "MARK":
            self.push(_MARK)
        elif name == "POP":
            if self.stack:
                self.stack.pop()
        elif name == "POP_MARK":
            self.pop_mark()
        elif name == "DUP":
            self.push(self.stack[-1] if self.stack else Odd("<stack underflow>"))
        elif name in ("PUT", "BINPUT", "LONG_BINPUT"):
            self.memo[arg] = self.stack[-1] if self.stack else Odd("<empty>")
        elif name == "MEMOIZE":
            self.memo[len(self.memo)] = self.stack[-1] if self.stack else Odd("<empty>")
        elif name in ("GET", "BINGET", "LONG_BINGET"):
            self.push(self.memo.get(arg, Odd(f"<missing memo {arg}>")))
        elif name == "EMPTY_TUPLE":
            self.push(Seq("tuple", []))
        elif name == "TUPLE":
            self.push(Seq("tuple", self.pop_mark()))
        elif name in ("TUPLE1", "TUPLE2", "TUPLE3"):
            n = int(name[-1])
            items = [self.pop() for _ in range(n)]
            items.reverse()
            self.push(Seq("tuple", items))
        elif name == "EMPTY_LIST":
            self.push(Seq("list", []))
        elif name == "LIST":
            self.push(Seq("list", self.pop_mark()))
        elif name in ("EMPTY_DICT",):
            self.push(Dct([]))
        elif name == "DICT":
            items = self.pop_mark()
            # A hostile stream can hold an odd count; the leftover key is dropped, not raised on.
            self.push(Dct(list(zip(items[::2], items[1::2], strict=False))))
        elif name in ("EMPTY_SET",):
            self.push(Seq("set", []))
        elif name == "FROZENSET":
            self.push(Seq("frozenset", self.pop_mark()))
        elif name == "APPEND":
            value = self.pop()
            self._extend([value])
        elif name in ("APPENDS", "ADDITEMS"):
            self._extend(self.pop_mark())
        elif name == "SETITEM":
            value, key = self.pop(), self.pop()
            self._set_items([key, value])
        elif name == "SETITEMS":
            self._set_items(self.pop_mark())
        elif name == "GLOBAL":
            module, _, attr = str(arg).partition(" ")
            self.push(Glob(f"{module}.{attr}"))
        elif name == "STACK_GLOBAL":
            attr, module = self.pop(), self.pop()
            if isinstance(module, Const) and isinstance(attr, Const):
                self.push(Glob(f"{module.value}.{attr.value}"))
            else:
                self.push(Odd("<global named by a computed value>"))
        elif name == "INST":
            module, _, attr = str(arg).partition(" ")
            self.record(Call(Glob(f"{module}.{attr}"), self.pop_mark()))
        elif name == "OBJ":
            items = self.pop_mark() or [Odd("<missing class>")]
            self.record(Call(items[0], items[1:]))
        elif name == "REDUCE":
            args, func = self.pop(), self.pop()
            self.record(Call(func, args.items if isinstance(args, Seq) else [args]))
        elif name == "NEWOBJ":
            args, cls = self.pop(), self.pop()
            items = args.items if isinstance(args, Seq) else [args]
            self.record(Call(cls, items, rebuild=not items))
        elif name == "NEWOBJ_EX":
            kwargs, args, cls = self.pop(), self.pop(), self.pop()
            items = args.items if isinstance(args, Seq) else [args]
            empty_kwargs = isinstance(kwargs, Dct) and not kwargs.pairs
            self.record(Call(cls, items, kwargs, rebuild=not items and empty_kwargs))
        elif name == "BUILD":
            state, obj = self.pop(), self.pop()
            self.record(Build(obj, state))
        elif name == "PERSID":
            self.push(Odd("<persistent id>"))
        elif name == "BINPERSID":
            self.pop()
            self.push(Odd("<persistent id>"))
        else:  # EXT1/2/4, NEXT_BUFFER, READONLY_BUFFER, and anything newer
            self.push(Odd(f"<unsupported: {name}>"))
        return False

    def _extend(self, values: list) -> None:
        target = self.stack[-1] if self.stack else None
        if isinstance(target, Seq):
            target.items.extend(values)

    def _set_items(self, flat: list) -> None:
        target = self.stack[-1] if self.stack else None
        if isinstance(target, Dct):
            target.pairs.extend(zip(flat[::2], flat[1::2], strict=False))


class _Stop(Exception):
    pass


# ---- rendering ---------------------------------------------------------------------------

def _shown(node, show_all: bool) -> bool:
    if isinstance(node, Call):
        if isinstance(node.func, Glob):
            if node.rebuild and node.func.level == "unknown":
                return show_all  # a saved model rebuilding its own layers: summarised, not listed
            return show_all or node.func.level != "allowed"
        return True  # calling a value the file computed is always worth a line
    if isinstance(node, Build):
        # A BUILD is worth a line when it sets the state of something already shown.
        if isinstance(node.obj, (Call, Build)):
            return show_all or node.obj.var is not None
        return show_all
    return False


def _text(value) -> str:
    if isinstance(value, str):
        if len(value) > MAX_TEXT:
            return f"{value[:MAX_TEXT]!r}...({len(value)} chars)"
        return repr(value)
    if isinstance(value, (bytes, bytearray)):
        try:
            printable = len(value) <= MAX_TEXT and value.decode("ascii").isprintable()
        except UnicodeDecodeError:
            printable = False
        return repr(bytes(value)) if printable else f"<{len(value)} bytes>"
    return repr(value)


def _expr(node, depth: int = 0) -> str:
    if depth > MAX_DEPTH:
        return "..."
    if isinstance(node, Const):
        return _text(node.value)
    if isinstance(node, Glob):
        return node.ref
    if isinstance(node, (Call, Build)):
        if node.var is not None:
            return f"_{node.var}"
        if isinstance(node, Call):
            return f"{_expr(node.func, depth + 1)}(...)"
        return "BUILD(...)"
    if isinstance(node, Seq):
        return _container(node.kind, [_expr(i, depth + 1) for i in node.items[:MAX_ITEMS]],
                          len(node.items))
    if isinstance(node, Dct):
        shown = [f"{_expr(k, depth + 1)}: {_expr(v, depth + 1)}" for k, v in node.pairs[:MAX_ITEMS]]
        more = f", ...({len(node.pairs)} items)" if len(node.pairs) > MAX_ITEMS else ""
        return "{" + ", ".join(shown) + more + "}"
    if isinstance(node, Odd):
        return node.text
    return "<?>"


def _container(kind: str, shown: list[str], total: int) -> str:
    more = f", ...({total} items)" if total > len(shown) else ""
    body = ", ".join(shown) + more
    if kind == "tuple":
        return f"({body},)" if total == 1 else f"({body})"
    if kind == "list":
        return f"[{body}]"
    return f"{kind}({{{body}}})" if body else f"{kind}()"


def _statement(node) -> str:
    if isinstance(node, Call):
        args = [_expr(a, 1) for a in node.args[:MAX_ITEMS]]
        if len(node.args) > MAX_ITEMS:
            args.append(f"...({len(node.args)} arguments)")
        if node.kwargs is not None and not (isinstance(node.kwargs, Dct) and not node.kwargs.pairs):
            args.append(f"**{_expr(node.kwargs, 1)}")
        line = f"_{node.var} = {_expr(node.func, 1)}({', '.join(args)})"
    else:
        line = f"_{node.var} = BUILD({_expr(node.obj, 1)}, {_expr(node.state, 1)})"
    return line if len(line) <= MAX_LINE else line[:MAX_LINE] + "..."


def _strings(node, depth: int = 0) -> list[str]:
    if depth > 2:
        return []
    if isinstance(node, Const):
        value = node.value
        if isinstance(value, str):
            return [value]
        if isinstance(value, (bytes, bytearray)):
            try:
                text = value.decode("ascii")
            except UnicodeDecodeError:
                return []
            return [text] if text.isprintable() else []
    if isinstance(node, Seq):
        return [t for item in node.items[:MAX_ITEMS] for t in _strings(item, depth + 1)]
    return []


def _why(node) -> str:
    if isinstance(node, Build):
        return "sets the state of the object on an earlier line (runs its __setstate__)"
    if isinstance(node.func, Glob):
        reason = {
            "denied": "on Hayward's deny list: can run code, commands, or reach files or the network",
            "unknown": "not on Hayward's allow list: check what this is before loading",
            "allowed": "known-safe",
        }[node.func.level]
    elif isinstance(node.func, (Call, Build)) and node.func.var is not None:
        reason = "calls the result of an earlier line"
    else:
        reason = "calls a value built inside the file"
    hints = []
    for text in (t for a in node.args for t in _strings(a)):
        if _PICKLE_ARG_URL_RE.search(text):
            hints.append("argument is a URL")
        elif _PICKLE_ARG_SHELL_RE.search(text) or (text.startswith("/") and " " in text.strip()):
            hints.append("argument looks like a shell command")
    return "; ".join([reason, *dict.fromkeys(hints)])


def _render(walk: _Walk, label: str, show_all: bool) -> list[str]:
    lines = [f"# {label}"]
    hidden: Counter[str] = Counter()
    rebuilt: Counter[str] = Counter()
    shown = 0
    for node in walk.calls:
        if _shown(node, show_all):
            node.var = shown
            shown += 1
        elif isinstance(node, Call) and isinstance(node.func, Glob):
            (rebuilt if node.rebuild else hidden)[node.func.ref] += 1
    printed = 0
    for node in walk.calls:
        if node.var is None:
            continue
        if printed == MAX_STATEMENTS:
            lines.append(f"# ...{shown - printed} more lines not shown")
            break
        lines.append(_statement(node))
        lines.append(f"    # why: {_why(node)}")
        printed += 1
    if hidden:
        top = ", ".join(f"{ref} x{n}" for ref, n in hidden.most_common(3))
        lines.append(f"# {sum(hidden.values())} calls to known-safe functions hidden ({top}); --all shows them")
    if rebuilt:
        top = ", ".join(ref for ref, _ in rebuilt.most_common(3))
        lines.append(f"# {sum(rebuilt.values())} objects rebuilt from {len(rebuilt)} classes not on the allow "
                     f"list, with no arguments ({top}, ...); --all shows them")
    if not walk.calls:
        lines.append("# no calls: this pickle only builds plain data")
    elif not shown and not hidden:
        lines.append("# no calls to flag")
    if walk.note:
        lines.append(f"# {walk.note}")
    return lines


# ---- containers --------------------------------------------------------------------------

def _blobs(data: bytes) -> list[tuple[str, bytes]]:
    buf = io.BytesIO(data)
    if zipfile.is_zipfile(buf):
        try:
            with zipfile.ZipFile(buf) as zf:
                members = [i for i in zf.infolist() if i.filename.endswith((".pkl", ".pickle"))]
                out = []
                for info in sorted(members, key=lambda i: i.filename)[:MAX_MEMBERS]:
                    if info.file_size > MAX_FILE_BYTES:
                        out.append((f"{info.filename} (too large to read: {info.file_size:,} bytes)", b""))
                    else:
                        out.append((info.filename, zf.read(info)))
                return out or [("zip archive with no pickle member", b"")]
        except (zipfile.BadZipFile, ValueError, RuntimeError, OSError, EOFError) as exc:
            return [(f"zip archive that could not be read ({type(exc).__name__})", b"")]
    if data[:1] != b"\x80" and tarfile.is_tarfile(io.BytesIO(data)):
        try:
            with tarfile.open(fileobj=io.BytesIO(data)) as tf:
                out = []
                for member in tf.getmembers()[:MAX_MEMBERS]:
                    handle = tf.extractfile(member) if member.isfile() else None
                    if handle is not None:
                        out.append((member.name, handle.read(MAX_FILE_BYTES)))
                return out
        except (tarfile.TarError, OSError, EOFError) as exc:
            return [(f"tar archive that could not be read ({type(exc).__name__})", b"")]
    return [("pickle", data)]


def explain_bytes(data: bytes, show_all: bool = False) -> str:
    """Readable lines for every pickle in ``data`` (a bare pickle, a zip or a tar)."""
    lines: list[str] = []
    for label, blob in _blobs(data):
        if not blob:
            lines.append(f"# {label}")
            continue
        stream = io.BytesIO(blob)
        for index in range(MAX_PICKLES):
            start = stream.tell()
            if start >= len(blob):
                break
            walk = _Walk()
            finished = walk.run(stream)
            where = f"{label}" if start == 0 else f"{label}, pickle at byte {start}"
            if start > 0 and not walk.calls and not finished:
                lines.append(f"# {label}: {len(blob) - start:,} bytes after the last pickle are not a pickle")
                break
            lines.extend(_render(walk, where, show_all))
            if not finished:
                break
            if index == MAX_PICKLES - 1 and stream.tell() < len(blob):
                lines.append(f"# stopped: more than {MAX_PICKLES} pickles in one stream")
    return "\n".join(lines)


def explain_file(path: Path, show_all: bool = False) -> str:
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        return f"# {path.name}: {size:,} bytes, larger than the {MAX_FILE_BYTES:,} byte limit for explain"
    return explain_bytes(path.read_bytes(), show_all)
