"""Hand-built, in-memory Visor tar (vmtar) archive builder for compatibility tests.

A vmtar archive splits metadata and payloads:

* All member headers are packed back-to-back at the start of the archive, exactly
  like a regular tar. Visor members therefore do *not* reserve inline data blocks
  after their header; the next header starts at ``header_offset + 512``.
* The payloads live in a separate, page-aligned data region at the end of the
  archive. A visor header stores the absolute, little-endian offset of its payload
  in bytes ``496:500`` (``offset_data``). Regular tar members keep their payload
  inline, directly after their header.

The builder deliberately constructs every header by hand (including the tar
checksum) instead of using :mod:`tarfile`, so the tests control the exact byte
layout and can record the header offset, payload offset and expected bytes of
every member. Nothing is written to the host file system.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import Enum

BLOCKSIZE = 512
PAGESIZE = 4096

# Type flags from the tar specification / GNU extensions.
REGTYPE = b"0"
DIRTYPE = b"5"
PAX_XHDTYPE = b"x"
GNUTYPE_LONGNAME = b"L"


class MemberKind(Enum):
    """The structural role a member plays inside the hand-built archive."""

    VISOR = "visor"
    VISOR_DIR = "visor-dir"
    VISOR_EMPTY = "visor-empty"
    VISOR_PAX = "visor-pax"
    VISOR_GNULONG = "visor-gnulong"
    VISOR_SHARED = "visor-shared"
    VISOR_TAIL = "visor-tail"
    NORMAL = "normal"


@dataclass
class TarMember:
    """A single logical archive member with its recorded on-disk layout.

    ``header_offset`` points at the member's own (real) header. For PAX/GNU long
    name members the extension header sits immediately before it; its position is
    recorded in ``extension_offset``. ``content_offset`` is the absolute payload
    offset for visor members (``0`` for zero-length members) and the inline data
    offset for regular members.
    """

    name: str
    kind: MemberKind
    payload: bytes = b""
    expected: bytes = b""
    header_offset: int = -1
    extension_offset: int | None = None
    content_offset: int | None = None
    is_visor: bool = True

    def __post_init__(self) -> None:
        self.is_visor = self.kind is not MemberKind.NORMAL
        if not self.expected:
            self.expected = self.payload


@dataclass
class ArchiveLayout:
    """The resulting archive bytes together with a fully resolved layout map."""

    data: bytes
    members: list[TarMember] = field(default_factory=list)
    data_region_start: int = 0
    size: int = 0

    def by_name(self, name: str) -> TarMember:
        return next(member for member in self.members if member.name == name)

    def diagram(self) -> str:
        """Render an ASCII map of header and payload offsets."""
        lines = [
            f"archive size: {self.size} bytes",
            f"header region : 0 .. {self.data_region_start - 1}",
            f"data region   : {self.data_region_start} .. {self.size - 1} (page aligned, {PAGESIZE} bytes)",
            "",
            "header region (512 byte blocks, packed; no inline visor data):",
        ]
        for member in self.members:
            tag = member.kind.value
            if member.extension_offset is not None:
                lines.append(f"  {member.extension_offset:6d}  [{tag} extension header] -> {member.name}")
            lines.append(f"  {member.header_offset:6d}  [{tag:14}] {member.name}")
        lines.extend(["", "data region (absolute offsets referenced by visor headers):"])
        for member in self.members:
            if member.kind in (MemberKind.VISOR_DIR, MemberKind.VISOR_EMPTY):
                lines.append(f"  {0:6d}  [{member.kind.value:14}] {member.name} (no payload)")
            else:
                length = len(member.payload)
                end = (member.content_offset or 0) + length
                note = "inline" if member.kind is MemberKind.NORMAL else f"{length} bytes"
                lines.append(
                    f"  {member.content_offset:6d}  [{member.kind.value:14}] {member.name} " f"({note}, ends at {end})"
                )
        return "\n".join(lines) + "\n"


def _checksum(header: bytearray) -> bytes:
    """Return the 8-byte octal tar checksum field (six digits, NUL, space)."""
    return f"{sum(header):06o}\x00 ".encode()


def _set_visor_offset(archive: bytearray, header_offset: int, data_offset: int) -> None:
    """Overwrite a visor header's payload offset and keep its checksum valid.

    The tar checksum covers the whole header, including bytes ``496:500``. A naive
    offset edit invalidates the checksum and makes the reader reject the header,
    which would hide an offset bug behind a parse error. Recomputing it forces the
    reader to accept the header and expose the wrong payload location.
    """
    struct.pack_into("<I", archive, header_offset + 496, data_offset)
    header = archive[header_offset : header_offset + BLOCKSIZE]
    placeholder = bytearray(header)
    placeholder[148:156] = b" " * 8
    archive[header_offset + 148 : header_offset + 156] = _checksum(placeholder)


def _make_header(
    name: str,
    size: int,
    typeflag: bytes,
    *,
    visor: bool,
    data_offset: int = 0,
) -> bytes:
    header = bytearray(BLOCKSIZE)
    header[0:100] = name.encode()[:100].ljust(100, b"\x00")
    header[100:108] = b"0000644\x00"
    header[108:116] = b"0000000\x00"
    header[116:124] = b"0000000\x00"
    header[124:136] = f"{size:011o}\x00".encode()
    header[136:148] = b"00000000000\x00"
    header[148:156] = b" " * 8
    header[156:157] = typeflag
    header[157:257] = b"\x00" * 100
    if visor:
        # [257:264] holds the visor marker; ustar magic/version stay empty.
        header[257:264] = b"visor  "
        header[496:500] = struct.pack("<I", data_offset)
        header[500:504] = b"\x00" * 4
        header[504:508] = struct.pack("<I", 0)
        header[508:512] = struct.pack("<I", 0)
    else:
        header[257:263] = b"ustar\x00"
        header[263:265] = b"00"
        header[265:297] = b"root".ljust(32, b"\x00")
        header[297:329] = b"root".ljust(32, b"\x00")
    header[148:156] = _checksum(header)
    return bytes(header)


def _pax_record(keyword: str, value: str) -> bytes:
    """Encode a single ``"%d %s=%s\\n"`` PAX record with a correct length.

    The decimal length counts the whole record, including the length field, the
    separating space and the trailing newline. It is grown until the encoded
    record's own length matches the declared length.
    """
    body = f" {keyword}={value}\n"
    length = len(body) + 1
    while True:
        record = f"{length}{body}".encode()
        if len(record) == length:
            return record
        length = len(record)


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def default_members() -> list[TarMember]:
    """Return the canonical mixed-layout member set used by the tests."""
    pax_name = "pax/" + "segment/" * 12 + "visor-pax-member-with-a-deliberately-long-name.bin"
    gnu_name = "gnu/" + "segment/" * 12 + "visor-gnu-longname-member-with-a-deliberately-long-name.bin"
    return [
        TarMember("etc/", MemberKind.VISOR_DIR, b""),
        TarMember("alpha.bin", MemberKind.VISOR, b"A" * 300),
        TarMember(pax_name, MemberKind.VISOR_PAX, b"P" * 700),
        TarMember(gnu_name, MemberKind.VISOR_GNULONG, b"G" * 900),
        TarMember("empty.bin", MemberKind.VISOR_EMPTY, b""),
        TarMember("normal.txt", MemberKind.NORMAL, b"N" * 123),
        TarMember("shared/one.bin", MemberKind.VISOR_SHARED, b"S" * 500),
        TarMember("tail.bin", MemberKind.VISOR_TAIL, b"T" * 600),
    ]


def build_archive(members: list[TarMember] | None = None) -> ArchiveLayout:
    """Lay out a vmtar archive and resolve every member's offsets."""
    members = [TarMember(m.name, m.kind, m.payload, m.expected) for m in (members or default_members())]

    # --- Pass 1: pack the header region and remember each block's position. ---
    # Each entry is (member, extension_header_bytes_or_none, real_header_bytes,
    # inline_payload_bytes). Real headers carry a placeholder offset of 0 which
    # is patched once the data region offsets are known.
    header_chunks: list[tuple[TarMember, bytes, bytes]] = []
    cursor = 0
    for member in members:
        kind = member.kind
        if kind is MemberKind.VISOR_DIR:
            member.header_offset = cursor
            header_chunks.append((member, b"", _make_header(member.name, 0, DIRTYPE, visor=True)))
            cursor += BLOCKSIZE
        elif kind is MemberKind.NORMAL:
            member.header_offset = cursor
            member.content_offset = cursor + BLOCKSIZE
            inline = member.payload.ljust(_round_up(len(member.payload), BLOCKSIZE), b"\x00")
            header = _make_header(member.name, len(member.payload), REGTYPE, visor=False)
            header_chunks.append((member, b"", header + inline))
            cursor += BLOCKSIZE + len(inline)
        elif kind is MemberKind.VISOR_PAX:
            member.extension_offset = cursor
            record = _pax_record("path", member.name)
            extension = _make_header("pax_global_header", len(record), PAX_XHDTYPE, visor=False)
            extension += record.ljust(_round_up(len(record), BLOCKSIZE), b"\x00")
            cursor += len(extension)
            member.header_offset = cursor
            real = _make_header("pax_placeholder", len(member.payload), REGTYPE, visor=True)
            header_chunks.append((member, extension, real))
            cursor += BLOCKSIZE
        elif kind is MemberKind.VISOR_GNULONG:
            member.extension_offset = cursor
            long_name = member.name.encode() + b"\x00"
            extension = _make_header("././@LongLink", len(long_name), GNUTYPE_LONGNAME, visor=False)
            extension += long_name.ljust(_round_up(len(long_name), BLOCKSIZE), b"\x00")
            cursor += len(extension)
            member.header_offset = cursor
            real = _make_header(member.name[:100], len(member.payload), REGTYPE, visor=True)
            header_chunks.append((member, extension, real))
            cursor += BLOCKSIZE
        else:
            member.header_offset = cursor
            header_chunks.append(
                (member, b"", _make_header(member.name[:100], len(member.payload), REGTYPE, visor=True))
            )
            cursor += BLOCKSIZE

    header_end = cursor
    # Two zero blocks terminate the header region; padding then reaches the first page.
    data_region_start = _round_up(header_end + 2 * BLOCKSIZE, PAGESIZE)

    # --- Pass 2: place visor payloads in the separate, page-aligned data region. ---
    region = bytearray()
    position = data_region_start
    for member in members:
        if member.kind in (MemberKind.VISOR_DIR, MemberKind.VISOR_EMPTY):
            member.content_offset = 0
            continue
        if member.kind is MemberKind.NORMAL:
            continue
        member.content_offset = position
        region.extend(member.payload)
        position += len(member.payload)
        if member.kind is not MemberKind.VISOR_TAIL:
            padding = _round_up(len(member.payload), PAGESIZE) - len(member.payload)
            region.extend(b"\x00" * padding)
            position += padding

    # --- Pass 3: patch payload offsets into the real headers and assemble. ---
    archive = bytearray()
    for member, extension, real_block in header_chunks:
        archive.extend(extension)
        header = bytearray(real_block[:BLOCKSIZE])
        struct.pack_into("<I", header, 496, member.content_offset or 0)
        header[148:156] = b" " * 8
        header[148:156] = _checksum(header)
        archive.extend(header)
        archive.extend(real_block[BLOCKSIZE:])
    archive.extend(b"\x00" * (2 * BLOCKSIZE))
    archive.extend(b"\x00" * (data_region_start - len(archive)))
    archive.extend(region)

    return ArchiveLayout(data=bytes(archive), members=members, data_region_start=data_region_start, size=len(archive))
