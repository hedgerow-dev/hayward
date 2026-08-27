"""Embedded PE/ELF/Mach-O executable detection.

Extracted verbatim from ``hayward.scanner`` (HW-147). The tunable
``_EMBEDDED_EXEC_MAX_CANDIDATES`` stays defined on ``hayward.scanner`` and is
read from there so the test suite's ``monkeypatch.setattr(scanner_module,
"_EMBEDDED_EXEC_MAX_CANDIDATES", ...)`` continues to steer this function.
"""

from __future__ import annotations

import struct

import hayward.scanner as _scanner

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
    candidates = 0
    while pos != -1 and len(hits) < 10 and candidates < _scanner._EMBEDDED_EXEC_MAX_CANDIDATES:
        candidates += 1
        # PE: e_lfanew at 0x3C points at a "PE\0\0" signature.
        if pos + 0x40 <= len(data):
            (lfanew,) = struct.unpack_from("<I", data, pos + 0x3C)
            if 0 < lfanew < 1 << 20 and data[pos + lfanew:pos + lfanew + 4] == b"PE\0\0":
                hits.append(f"PE/Windows executable at offset {pos}")
        pos = data.find(b"MZ", pos + 1)

    pos = data.find(b"\x7fELF")
    candidates = 0
    while pos != -1 and len(hits) < 20 and candidates < _scanner._EMBEDDED_EXEC_MAX_CANDIDATES:
        candidates += 1
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

    candidates = 0
    for magic in _MACHO_MAGICS:
        pos = data.find(magic)
        while pos != -1 and len(hits) < 30 and candidates < _scanner._EMBEDDED_EXEC_MAX_CANDIDATES:
            candidates += 1
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
