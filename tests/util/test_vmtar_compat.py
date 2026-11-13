"""Compatibility tests for the Visor tar (vmtar) reader.

These tests build vmtar archives entirely in memory with a hand-rolled builder
(:mod:`tests.util.vmtar_layout`) so that every member's header offset, payload
offset and expected bytes are known up front. They focus on the interaction
between two independent file positions that a vmtar reader must keep apart:

* ``TarFile.offset``      - where the *next header* is scanned from. Visor headers
                            are packed together, so this must advance by one 512
                            byte block and never by the member's size.
* ``TarInfo.offset_data`` - where the *payload* lives, in a separate page-aligned
                            data region at the end of the archive.

Two regressions are explicitly pinned down:

1. "skip the next header by size" - treating a visor member like a normal tar
   member when scanning, which desynchronises header parsing.
2. "lose the visor offset after extension header processing" - letting PAX/GNU
   long name handling recalculate the next-header position from inline data.

Nothing is extracted to the host file system.
"""

from __future__ import annotations

import gzip
import lzma
import tarfile
from io import BytesIO

import pytest

from dissect.hypervisor.util import vmtar
from tests._util import absolute_path
from tests.util import vmtar_layout as layout_mod
from tests.util.vmtar_layout import BLOCKSIZE, MemberKind, _set_visor_offset, build_archive


@pytest.fixture(scope="module")
def layout() -> layout_mod.ArchiveLayout:
    return build_archive()


@pytest.fixture(params=["plain", "gzip", "lzma"])
def wrapped_archive(request: pytest.FixtureRequest, layout: layout_mod.ArchiveLayout) -> bytes:
    if request.param == "plain":
        return layout.data
    if request.param == "gzip":
        return gzip.compress(layout.data)
    return lzma.compress(layout.data)


def _open(data: bytes) -> vmtar.VisorTarFile:
    return vmtar.open(fileobj=BytesIO(data))


def test_layout_is_fully_recorded(layout: layout_mod.ArchiveLayout) -> None:
    """The builder records a block-aligned header and a page-aligned data region."""
    assert layout.data_region_start % layout_mod.PAGESIZE == 0
    for member in layout.members:
        assert member.header_offset % BLOCKSIZE == 0
        if member.extension_offset is not None:
            assert member.extension_offset % BLOCKSIZE == 0
            assert member.extension_offset < member.header_offset
        if member.kind is MemberKind.NORMAL:
            assert member.content_offset == member.header_offset + BLOCKSIZE


def test_member_enumeration_matches_recorded_layout(layout: layout_mod.ArchiveLayout, wrapped_archive: bytes) -> None:
    """Scanning reaches every member in order, regardless of the outer layer."""
    tar = _open(wrapped_archive)
    scanned = list(tar)
    expected_names = [member.name.rstrip("/") for member in layout.members]
    assert [member.name for member in scanned] == expected_names

    by_name = {member.name: member for member in scanned}
    for spec in layout.members:
        member = by_name[spec.name.rstrip("/")]
        assert member.is_visor == spec.is_visor
        if spec.extension_offset is not None:
            # A PAX/GNU long name member reports the extension header as its offset.
            assert member.offset == spec.extension_offset
        else:
            assert member.offset == spec.header_offset


def test_payload_offsets_and_bytes_match_layout(layout: layout_mod.ArchiveLayout, wrapped_archive: bytes) -> None:
    """Reading content uses the separate data offset, never inline header bytes."""
    tar = _open(wrapped_archive)
    for spec in layout.members:
        member = tar.getmember(spec.name.rstrip("/"))
        if spec.kind in (MemberKind.VISOR_DIR, MemberKind.VISOR_EMPTY):
            assert member.size == 0
            continue
        assert member.isreg()
        assert member.offset_data == spec.content_offset
        assert tar.extractfile(member).read() == spec.expected


def test_scanning_and_reading_do_not_interfere(layout: layout_mod.ArchiveLayout, wrapped_archive: bytes) -> None:
    """Reading a member mid-scan must not move the next-header scan position."""
    tar = _open(wrapped_archive)
    specs = layout.members
    seen: list[str] = []
    for index, member in enumerate(tar):
        seen.append(member.name)
        if member.isreg() and member.size:
            spec = layout.by_name(member.name)
            assert tar.extractfile(member).read() == spec.expected
        # After the read, the scan cursor points at the following header block.
        if index + 1 < len(specs):
            following = specs[index + 1]
            next_header = following.extension_offset or following.header_offset
            assert tar.offset == next_header
    assert seen == [member.name.rstrip("/") for member in specs]

    # Re-reading after a full scan returns identical, stable bytes.
    for spec in layout.members:
        if spec.kind in (MemberKind.VISOR_DIR, MemberKind.VISOR_EMPTY):
            continue
        member = tar.getmember(spec.name.rstrip("/"))
        assert tar.extractfile(member).read() == spec.expected


def test_zero_length_members_have_no_data_region_footprint(layout: layout_mod.ArchiveLayout) -> None:
    empty = layout.by_name("empty.bin")
    assert empty.kind is MemberKind.VISOR_EMPTY
    assert empty.payload == b"" and empty.content_offset == 0

    tar = _open(layout.data)
    member = tar.getmember("empty.bin")
    assert member.size == 0
    assert tar.extractfile(member) is None or tar.extractfile(member).read() == b""


def test_shared_data_region_is_referenced_by_absolute_offset(layout: layout_mod.ArchiveLayout) -> None:
    """All visor payloads share one region and are reached via absolute offsets."""
    visor_payloads = [
        member
        for member in layout.members
        if member.kind not in (MemberKind.NORMAL, MemberKind.VISOR_DIR, MemberKind.VISOR_EMPTY)
    ]
    assert visor_payloads
    for member in visor_payloads:
        assert member.content_offset is not None
        assert layout.data_region_start <= member.content_offset < layout.size

    tar = _open(layout.data)
    for member in visor_payloads:
        info = tar.getmember(member.name.rstrip("/"))
        assert info.offset_data == member.content_offset


def test_tail_member_content_ends_exactly_at_eof(layout: layout_mod.ArchiveLayout) -> None:
    """A payload placed at the end of the file reads fully without trailing blocks."""
    tail = layout.by_name("tail.bin")
    assert tail.kind is MemberKind.VISOR_TAIL
    assert tail.content_offset + len(tail.payload) == layout.size

    for data in (layout.data, gzip.compress(layout.data), lzma.compress(layout.data)):
        tar = _open(data)
        info = tar.getmember("tail.bin")
        assert info.offset_data == tail.content_offset
        assert tar.extractfile(info).read() == tail.expected


def test_corrupted_offset_cannot_fake_success(layout: layout_mod.ArchiveLayout) -> None:
    """A valid header pointing at another member's payload is detected by content."""
    alpha = layout.by_name("alpha.bin")
    tail = layout.by_name("tail.bin")
    corrupted = bytearray(layout.data)
    _set_visor_offset(corrupted, alpha.header_offset, tail.content_offset)

    # The header checksum stays valid, so the archive still enumerates fully.
    tar = _open(bytes(corrupted))
    assert [member.name for member in tar.getmembers()] == [member.name.rstrip("/") for member in layout.members]
    read_back = tar.extractfile(tar.getmember("alpha.bin")).read()
    assert read_back != alpha.expected
    # The reader faithfully seeks to the corrupted offset and reads alpha's declared
    # size from tail's payload, so the wrong source is observable rather than faked.
    assert read_back == tail.expected[: len(alpha.payload)]
    assert read_back[:1] == b"T"


def test_non_tar_input_is_rejected() -> None:
    with pytest.raises(tarfile.ReadError):
        _open(b"definitely not a tar archive" * 32)
    with pytest.raises(tarfile.ReadError):
        _open(b"")


# ---------------------------------------------------------------------------
# Regression guards: deliberately broken readers must fail the assertions above.
# ---------------------------------------------------------------------------


class _SizeSkipTarInfo(vmtar.VisorTarInfo):
    """Broken reader: advance the scan by member size instead of one header block."""

    def _proc_member(self, tar_file: tarfile.TarFile) -> tarfile.TarInfo:
        self.offset_data = tar_file.fileobj.tell()
        offset = self.offset_data
        if self.isreg() or self.type not in tarfile.SUPPORTED_TYPES:
            offset += self._block(self.size)
        tar_file.offset = offset
        self._apply_pax_info(tar_file.pax_headers, tar_file.encoding, tar_file.errors)
        return self


class _SizeSkipTarFile(vmtar.VisorTarFile):
    def __init__(self, *args, **kwargs) -> None:
        super(vmtar.VisorTarFile, self).__init__(*args, **kwargs, tarinfo=_SizeSkipTarInfo)


class _PaxLoseOffsetTarInfo(vmtar.VisorTarInfo):
    """Broken reader: extension header handling recalculates the scan offset inline."""

    def _proc_pax(self, tar_file: tarfile.TarFile) -> tarfile.TarInfo:
        member = super()._proc_pax(tar_file)
        offset = member.offset_data
        if member.isreg() or member.type not in tarfile.SUPPORTED_TYPES:
            offset += member._block(member.size)
        tar_file.offset = offset
        return member


class _PaxLoseOffsetTarFile(vmtar.VisorTarFile):
    def __init__(self, *args, **kwargs) -> None:
        super(vmtar.VisorTarFile, self).__init__(*args, **kwargs, tarinfo=_PaxLoseOffsetTarInfo)


def test_regression_skip_next_header_by_size_is_caught(layout: layout_mod.ArchiveLayout) -> None:
    expected = [member.name.rstrip("/") for member in layout.members]
    broken = _SizeSkipTarFile.open(fileobj=BytesIO(layout.data))
    with pytest.raises(AssertionError):
        assert [member.name for member in broken.getmembers()] == expected


def test_regression_lost_visor_offset_after_extension_header_is_caught(
    layout: layout_mod.ArchiveLayout,
) -> None:
    pax_spec = next(member for member in layout.members if member.kind is MemberKind.VISOR_PAX)
    broken = _PaxLoseOffsetTarFile.open(fileobj=BytesIO(layout.data))
    members = {member.name: member for member in broken.getmembers()}
    pax_member = members[pax_spec.name.rstrip("/")]

    # The broken scan either misplaces the payload offset or drops later members.
    enumeration_ok = [member.name for member in broken.getmembers()] == [
        member.name.rstrip("/") for member in layout.members
    ]
    offset_ok = pax_member.offset_data == pax_spec.content_offset
    content_ok = broken.extractfile(pax_member).read() == pax_spec.expected
    assert not (enumeration_ok and offset_ok and content_ok)


def test_archive_layout_diagram_is_stable(layout: layout_mod.ArchiveLayout) -> None:
    """The retained layout map documents every recorded header and payload offset."""
    diagram = layout.diagram()
    for member in layout.members:
        assert member.name.rstrip("/") in diagram
        assert str(member.header_offset) in diagram
    assert str(layout.data_region_start) in diagram


def test_checked_in_layout_diagram_matches_builder(layout: layout_mod.ArchiveLayout) -> None:
    """The documented sample map cannot drift from the archive the tests build."""
    diagram_path = absolute_path("_data/util/vmtar/compat_layout.md")
    document = diagram_path.read_text()
    assert layout.diagram() in document
