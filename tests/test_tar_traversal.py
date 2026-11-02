"""Plain `.tar` archives get the same traversal/symlink checks as `.nemo`.

The scaled comparative benchmark caught Hayward returning no finding at all on a
`.tar` carrying a `../` member name and on a `.tar` carrying a traversing
symlink. The tar scanner that backs `.nemo` already reports both
(MFV-ARCHIVE-001 for an unsafe member name, MFV-ARCHIVE-002 for a link whose
target escapes), but `.tar` was never routed to it. It now is.
"""

from __future__ import annotations

import io
import os
import pickle
import tarfile

import pytest

from hayward import ModelFileScanner


def _tar(path, build):
    with tarfile.open(path, "w") as tf:
        build(tf)
    return ModelFileScanner().scan_file(path)


def test_plain_tar_traversal_member_name_flagged(tmp_path):
    def build(tf):
        data = b"x" * 32
        info = tarfile.TarInfo("../../evil.py")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))

    findings = _tar(tmp_path / "model.tar", build)
    assert any(f.rule_id == "MFV-ARCHIVE-001" for f in findings), [
        (f.rule_id, f.severity) for f in findings
    ]


def test_plain_tar_traversal_symlink_flagged(tmp_path):
    def build(tf):
        info = tarfile.TarInfo("innocent.txt")
        info.type = tarfile.SYMTYPE
        info.linkname = "../../../../etc/passwd"
        tf.addfile(info)

    findings = _tar(tmp_path / "model.tar", build)
    assert any(f.rule_id == "MFV-ARCHIVE-002" for f in findings), [
        (f.rule_id, f.severity) for f in findings
    ]


def test_benign_tar_has_no_high_finding(tmp_path):
    def build(tf):
        data = b"\x00" * 128
        info = tarfile.TarInfo("weights/model.bin")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))

    findings = _tar(tmp_path / "model.tar", build)
    assert not any(f.severity.value in ("critical", "high") for f in findings), [
        (f.rule_id, f.severity) for f in findings
    ]


class _Evil:
    def __reduce__(self):
        return (os.system, ("id",))


def test_pickle_renamed_to_tar_is_still_content_sniffed(tmp_path):
    # Routing `.tar` to the tar scanner must not skip the content sniff every
    # other unclaimed extension gets. A dangerous pickle renamed `.tar` (which
    # does not open as a tar) must still be caught, not dismissed as an
    # unopenable container (MFV-SKIP-003). This is the danger.dat rename trick.
    path = tmp_path / "payload.tar"
    path.write_bytes(pickle.dumps(_Evil()))  # dumps only serialises the reduce
    findings = ModelFileScanner().scan_file(path)
    assert any(f.severity.value in ("critical", "high") for f in findings), [
        (f.rule_id, f.severity) for f in findings
    ]
    assert not any(f.rule_id == "MFV-SKIP-003" for f in findings), [
        (f.rule_id, f.severity) for f in findings
    ]


def _member(tf, name, data):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tf.addfile(info, io.BytesIO(data))


def test_text_member_that_opens_like_a_pickle_is_not_flagged(tmp_path):
    # Seen on the Hub: a web app's source tree as `repo.tar`. Minified JS opens
    # with `(function`, which is a MARK opcode followed by no opcode at all, and
    # its later `exec(` text matched the raw-byte fallback at HIGH.
    js = b"(function(global, factory) { var m = /x/.exec(s); })(this);\n"
    findings = _tar(tmp_path / "repo.tar", lambda tf: _member(tf, "static/lib/docx.umd.min.js", js))
    assert not any(f.severity.value in ("critical", "high") for f in findings), [
        (f.rule_id, f.severity) for f in findings
    ]


def test_protocol_0_pickle_member_is_still_flagged(tmp_path):
    payload = pickle.dumps(_Evil(), protocol=0)
    findings = _tar(tmp_path / "repo.tar", lambda tf: _member(tf, "notes.txt", payload))
    assert any(f.severity.value in ("critical", "high") for f in findings), [
        (f.rule_id, f.severity) for f in findings
    ]


def test_pickle_member_that_breaks_after_its_call_is_still_flagged(tmp_path):
    # pickle.load runs the REDUCE before it reaches the bad byte, so a stream
    # that fails to parse only after an import still counts.
    payload = b"cos\nsystem\n(S'id'\ntR\xff exec("
    findings = _tar(tmp_path / "repo.tar", lambda tf: _member(tf, "notes.txt", payload))
    assert any(f.severity.value in ("critical", "high") for f in findings), [
        (f.rule_id, f.severity) for f in findings
    ]


@pytest.mark.parametrize("prefix", [b"(I0x1\n1", b"(I017\n1", b"(I1\x00x\n1", b"(F1\x00\n1", b"(S'\xe9'\n1"])
def test_protocol_0_member_behind_a_loader_only_number_is_still_flagged(tmp_path, prefix):
    # These load, so the walk must read past them rather than call the member text.
    payload = prefix + b"cos\nsystem\n(S'id'\ntR."
    findings = _tar(tmp_path / "repo.tar", lambda tf: _member(tf, "notes.txt", payload))
    assert any(f.severity.value in ("critical", "high") for f in findings), [
        (f.rule_id, f.severity) for f in findings
    ]
