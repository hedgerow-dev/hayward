"""`hayward explain`: what a pickle would do, as readable lines. Nothing is ever run.

Every pickle here is hand-built bytes. Commands are harmless; one test proves a
command is not executed by checking that its marker file never appears.
"""

from __future__ import annotations

import io
import pickle
import tarfile
import zipfile

from hayward import explain
from hayward.cli import main as cli_main


def s(text: str) -> bytes:  # BINUNICODE
    raw = text.encode()
    return b"X" + len(raw).to_bytes(4, "little") + raw


def call1(module: str, name: str, arg: str) -> bytes:  # module.name(arg), protocol 2
    return b"c" + module.encode() + b"\n" + name.encode() + b"\n" + s(arg) + b"\x85R"


def p2(body: bytes) -> bytes:
    return b"\x80\x02" + body + b"."


def test_a_denied_call_is_shown_with_why(tmp_path):
    out = explain.explain_bytes(p2(call1("posix", "system", "echo $(id)")))
    assert "_0 = posix.system('echo $(id)')" in out
    assert "deny list" in out
    assert "shell command" in out


def test_the_import_getattr_call_chain_shows_the_final_command():
    body = (b"cbuiltins\ngetattr\n" + call1("builtins", "__import__", "subprocess")
            + s("getoutput") + b"\x86R" + s("echo $(id)") + b"\x85R")
    out = explain.explain_bytes(p2(body))
    assert "_0 = builtins.__import__('subprocess')" in out
    assert "_1 = builtins.getattr(_0, 'getoutput')" in out
    assert "_2 = _1('echo $(id)')" in out
    assert "earlier line" in out


def test_known_safe_calls_are_hidden_and_counted():
    rebuild = b"ctorch._utils\n_rebuild_tensor_v2\n" + s("x") + b"\x85R"
    body = b"(" + rebuild * 5 + b"l"
    out = explain.explain_bytes(p2(body))
    assert "_rebuild_tensor_v2" in out and "5" in out and "hidden" in out
    assert "_0 =" not in out
    full = explain.explain_bytes(p2(body), show_all=True)
    assert "_4 = torch._utils._rebuild_tensor_v2('x')" in full


def test_a_value_used_twice_is_named_once():
    body = call1("mypkg", "make", "a") + b"q\x00" + b"cbuiltins\nprint\n" + b"h\x00h\x00\x86R"
    out = explain.explain_bytes(p2(body))
    assert "_0 = mypkg.make('a')" in out
    assert "builtins.print(_0, _0)" in out


def test_an_unknown_callable_is_marked_unknown():
    out = explain.explain_bytes(p2(call1("mypkg", "Thing", "x")))
    assert "_0 = mypkg.Thing('x')" in out and "allow list" in out


def test_long_strings_and_bytes_are_shortened_with_their_length():
    out = explain.explain_bytes(p2(call1("posix", "system", "A" * 5000)))
    assert "5000" in out and "A" * 500 not in out
    out = explain.explain_bytes(p2(b"cmypkg\nf\n" + b"C\x05\x00\x01\x02\x03\x04" + b"\x85R"))
    assert "<5 bytes>" in out


def test_a_zip_checkpoint_is_read_member_by_member():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("archive/data.pkl", p2(call1("posix", "system", "echo hi")))
        z.writestr("archive/data/0", b"\0" * 64)
    out = explain.explain_bytes(buf.getvalue())
    assert "archive/data.pkl" in out and "posix.system('echo hi')" in out


def test_an_old_tar_checkpoint_is_read():
    buf = io.BytesIO()
    payload = p2(call1("posix", "system", "echo hi"))
    with tarfile.open(fileobj=buf, mode="w") as t:
        info = tarfile.TarInfo("storages")
        info.size = len(payload)
        t.addfile(info, io.BytesIO(payload))
    out = explain.explain_bytes(buf.getvalue())
    assert "storages" in out and "posix.system('echo hi')" in out


def test_a_truncated_stream_stops_cleanly():
    data = p2(call1("posix", "system", "echo hi"))
    out = explain.explain_bytes(data[:-6])
    assert "posix.system" in out or "stopped" in out
    assert "stopped" in out


def test_an_unsupported_opcode_is_named_not_guessed():
    out = explain.explain_bytes(b"\x80\x02\x82\x01\x85cmypkg\nf\n\x85R" + b".")  # EXT1
    assert "<unsupported: EXT1>" in out


def test_a_huge_container_is_summarised():
    body = b"cmypkg\nf\n(" + b"K\x01" * 10000 + b"t\x85R"
    out = explain.explain_bytes(p2(body))
    assert "10000 items" in out and len(out) < 2000


def test_deep_nesting_is_bounded():
    body = b"cmypkg\nf\n" + b"(" * 5000 + b")" + b"t" * 5000 + b"\x85R"
    out = explain.explain_bytes(p2(body))
    assert "mypkg.f(" in out and len(out) < 5000


def test_nothing_is_ever_run(tmp_path):
    marker = tmp_path / "MARKER"
    explain.explain_bytes(p2(call1("posix", "system", f"touch {marker}")))
    assert not marker.exists()


def test_an_ordinary_state_dict_has_no_flagged_calls():
    out = explain.explain_bytes(pickle.dumps({"w": [1.0, 2.0], "b": "x"}, protocol=4))
    assert "no calls" in out


def test_the_cli_explains_a_file_and_exits_zero(tmp_path, capsys):
    f = tmp_path / "model.pkl"
    f.write_bytes(p2(call1("posix", "system", "echo hi")))
    assert cli_main(["explain", str(f)]) == 0
    assert "posix.system('echo hi')" in capsys.readouterr().out


def test_the_cli_reports_a_missing_file(tmp_path, capsys):
    assert cli_main(["explain", str(tmp_path / "nope.pkl")]) == 2


def test_object_rebuilds_are_summarised_but_a_denied_class_is_not():
    rebuild = b"cmypkg.layers\nConv\n)\x81"   # NEWOBJ with no arguments
    body = b"(" + rebuild * 4 + b"cposix\nsystem\n" + s("echo hi") + b"\x85\x81" + b"l"
    out = explain.explain_bytes(p2(body))
    assert "4 objects rebuilt" in out and "mypkg.layers.Conv" in out
    assert "_0 = posix.system('echo hi')" in out
    assert "_1 =" not in out
