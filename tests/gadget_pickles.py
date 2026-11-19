"""Build pickle bytes for published scanner-evasion gadgets, opcode by opcode.

Nothing here unpickles anything: these are byte strings for the scanner to read.
Every command is a harmless `echo`. Shapes are taken from gadget files found on the
Hugging Face Hub in November 2026 (operator.methodcaller over importlib.import_module,
code.InteractiveInterpreter.runsource, mailcap.findmatch, logging.config.dictConfig, ...).
"""

from __future__ import annotations

import struct


class Call:
    """A callable applied to arguments: GLOBAL + args tuple + REDUCE."""

    def __init__(self, target: str | Call, *args):
        self.target, self.args = target, args


def _emit(value) -> bytes:
    if isinstance(value, Call):
        head = _global(value.target) if isinstance(value.target, str) else _emit(value.target)
        return head + _tuple(value.args) + b"R"
    if isinstance(value, bool):
        return b"\x88" if value else b"\x89"
    if isinstance(value, int):
        return b"J" + struct.pack("<i", value)
    if isinstance(value, str):
        raw = value.encode()
        return b"X" + struct.pack("<I", len(raw)) + raw
    if isinstance(value, bytes):
        return b"C" + bytes([len(value)]) + value
    if isinstance(value, tuple):
        return _tuple(value)
    if isinstance(value, list):
        return b"]" + b"(" + b"".join(_emit(v) for v in value) + b"e"
    if isinstance(value, dict):
        return b"}" + b"(" + b"".join(_emit(k) + _emit(v) for k, v in value.items()) + b"u"
    raise TypeError(type(value))


def _global(dotted: str) -> bytes:
    module, name = dotted.rsplit(".", 1)
    return b"c" + module.encode() + b"\n" + name.encode() + b"\n"


def _tuple(items) -> bytes:
    return b"(" + b"".join(_emit(v) for v in items) + b"t"


def build(value) -> bytes:
    return b"\x80\x02" + _emit(value) + b"."


OS = Call("importlib.import_module", "os")

GADGETS: dict[str, bytes] = {
    "methodcaller_system": build(Call(Call("operator.methodcaller", "system", "echo hi"), OS)),
    "methodcaller_system_bytes": build(Call(Call("operator.methodcaller", "system", b"echo hi"), OS)),
    "operator_call": build(Call("operator.call", Call("operator.methodcaller", "system", "echo hi"), OS)),
    "getitem_vars": build(Call(Call("operator.getitem", Call("builtins.vars", OS), "system"), "echo hi")),
    "getattr_static": build(Call(Call("inspect.getattr_static", OS, "system"), "echo hi")),
    "runsource": build(Call(Call("operator.methodcaller", "runsource", "print(1)"),
                            Call("code.InteractiveInterpreter"))),
    "trace_run": build(Call(Call("operator.methodcaller", "run", "print(1)"), Call("trace.Trace"))),
    "mailcap": build(Call("mailcap.findmatch",
                          {"text/plain": [{"view": "cat %s", "test": "echo hi"}]}, "text/plain")),
    "dictconfig_popen": build(Call("logging.config.dictConfig",
                                   {"version": 1, "handlers": {"h": {"()": "subprocess.Popen", "args": ["echo", "hi"]}},
                                    "root": {"handlers": ["h"]}})),
    "dictconfig_ext": build(Call("logging.config.dictConfig",
                                 {"version": 1, "handlers": {"h": {"()": "ext://os.system", "command": "echo hi"}},
                                  "root": {"handlers": ["h"]}})),
    "atexit": build(Call("atexit.register", Call("operator.methodcaller", "system", "echo hi"), OS)),
    "codeop": build(Call(Call("types.FunctionType", Call("codeop.compile_command", "print(1)"), {}))),
    "copyreg_extension": build(Call("copyreg.add_extension", "os", "system", 31337)),
    "torch_hub": build(Call("torch.hub.load", "someone/repo", "model")),
    "import_os_only": build(OS),
}
