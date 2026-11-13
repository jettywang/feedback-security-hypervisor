from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from dissect.hypervisor.disk.c_vhd import SECTOR_SIZE, c_vhd
from dissect.hypervisor.disk.vhd import DISK_TYPE_DIFFERENCING, DISK_TYPE_DYNAMIC, DISK_TYPE_FIXED, VHD, checksum
from dissect.hypervisor.disk.vhd_diagnose import (
    PLATFORM_CODE_W2RU,
    diagnose,
    format_diagnostics,
    resolve_parent_path,
    verify_parent,
)
from dissect.hypervisor.exceptions import InvalidHeaderError, InvalidVirtualDisk

BLOCK_SIZE = 4096
BLOCK_COUNT = 8
SECTORS_PER_BLOCK = BLOCK_SIZE // SECTOR_SIZE
TABLE_OFFSET = 1536
DATA_OFFSET = 2048
PARENT_MARKER = 0xA5
CHILD_MARKER = 0xCC
PARTIAL_MARKER = 0xDD
POISON_MARKER = 0xEE
DYNAMIC_MARKER = 0x11
DYNAMIC_MARKER_2 = 0x22
PARENT_UID = b"\x99" * 16
CHILD_UID = b"\x02" * 16
PARENT_TIMESTAMP = 0x12345678
PARENT_NAME = "parent.vhd"


def make_footer(disk_type: int, data_offset: int, size: int, unique_id: bytes, timestamp: int) -> bytes:
    footer = c_vhd.footer()
    footer.cookie = b"conectix"
    footer.features = 0x2
    footer.version = 0x00010000
    footer.data_offset = data_offset
    footer.timestamp = timestamp
    footer.creator_application = 0
    footer.creator_version = 0
    footer.creator_host_os = 0
    footer.original_size = size
    footer.current_size = size
    footer.disk_geometry = 0
    footer.disk_type = disk_type
    footer.unique_id = unique_id
    footer.saved_state = b"\x00"
    raw = bytes(footer) + b"\x00"
    footer.checksum = checksum(raw, 64)
    return bytes(footer) + b"\x00"


def make_dynamic_header(
    table_entries: int,
    parent_uid: bytes = b"\x00" * 16,
    parent_timestamp: int = 0,
    parent_name: str = "",
    locators: list[c_vhd.parent_locator] | None = None,
) -> bytes:
    header = c_vhd.dynamic_header()
    header.cookie = b"cxsparse"
    header.data_offset = 0xFFFFFFFFFFFFFFFF
    header.table_offset = TABLE_OFFSET
    header.header_version = 0x00010000
    header.max_table_entries = table_entries
    header.block_size = BLOCK_SIZE
    header.parent_unique_id = parent_uid
    header.parent_timestamp = parent_timestamp
    header.parent_unicode_name = parent_name.encode("utf-16-le").ljust(512, b"\x00")[:512]
    header.parent_locators = (locators or []) + [c_vhd.parent_locator() for _ in range(8 - len(locators or []))]
    raw = bytes(header)
    header.checksum = checksum(raw, 36)
    return bytes(header)


def make_block(bitmap_mask: int, marker: int) -> bytes:
    bitmap = bytearray(SECTOR_SIZE)
    bitmap[0] = bitmap_mask
    return bytes(bitmap) + bytes([marker]) * BLOCK_SIZE


def build_parent(path: Path) -> None:
    size = BLOCK_SIZE * BLOCK_COUNT
    body = bytes([PARENT_MARKER]) * size
    path.write_bytes(body + make_footer(DISK_TYPE_FIXED, 0xFFFFFFFFFFFFFFFF, size, PARENT_UID, PARENT_TIMESTAMP))


def build_fixed_child(path: Path) -> None:
    size = BLOCK_SIZE * BLOCK_COUNT
    body = bytes([DYNAMIC_MARKER]) * size
    path.write_bytes(body + make_footer(DISK_TYPE_FIXED, 0xFFFFFFFFFFFFFFFF, size, CHILD_UID, 0))


def build_dynamic_child(path: Path) -> None:
    # Block 0 and 2 fully present; blocks 1 and 3 are sparse zeros.
    entries = [0xFFFFFFFF] * BLOCK_COUNT
    blocks: list[bytes] = []
    cursor = DATA_OFFSET // SECTOR_SIZE
    for index, (marker, mask) in enumerate(
        [(DYNAMIC_MARKER, 0xFF), (None, None), (DYNAMIC_MARKER_2, 0xFF), (None, None)]
    ):
        if marker is None:
            continue
        block = make_block(mask, marker)
        entries[index] = cursor
        blocks.append(block)
        # Each on-disk block is one sector bitmap followed by the block's data sectors.
        cursor += SECTORS_PER_BLOCK + 1

    image = bytearray(cursor * SECTOR_SIZE)
    image[512:1536] = make_dynamic_header(BLOCK_COUNT)
    struct.pack_into(f">{BLOCK_COUNT}I", image, TABLE_OFFSET, *entries)
    offset = DATA_OFFSET
    for block in blocks:
        image[offset : offset + len(block)] = block
        offset += len(block)
    image += make_footer(DISK_TYPE_DYNAMIC, 512, BLOCK_SIZE * BLOCK_COUNT, CHILD_UID, 0)
    path.write_bytes(bytes(image))


def build_differencing_child(path: Path, parent_path: str = PARENT_NAME) -> None:
    # Block 0 fully child-owned. Block 1 is absent from the child BAT.
    # Block 2 is half child-owned and half poison; block 3 is absent too.
    entries = [0xFFFFFFFF] * BLOCK_COUNT
    block0 = make_block(0xFF, CHILD_MARKER)
    partial = make_block(0xF0, PARTIAL_MARKER)
    # Poison the four sectors whose bitmap bits are clear.
    partial = partial[: SECTOR_SIZE + 4 * SECTOR_SIZE] + bytes([POISON_MARKER]) * (4 * SECTOR_SIZE)
    entries[0] = DATA_OFFSET // SECTOR_SIZE
    entries[2] = (DATA_OFFSET // SECTOR_SIZE) + 1 + SECTORS_PER_BLOCK

    image = bytearray(DATA_OFFSET)
    image[512:1536] = make_dynamic_header(BLOCK_COUNT)
    struct.pack_into(f">{BLOCK_COUNT}I", image, TABLE_OFFSET, *entries)
    offset = DATA_OFFSET
    for block in (block0, partial):
        image += b"\x00" * (offset - len(image))
        image += block
        offset += len(block)

    locator_blob = parent_path.encode("utf-16-le")
    locator_data_offset = len(image)
    image += locator_blob
    locator = c_vhd.parent_locator(
        platform_code=PLATFORM_CODE_W2RU,
        platform_data_space=SECTOR_SIZE,
        platform_data_length=len(locator_blob),
        reserved=0,
        platform_data_offset=locator_data_offset,
    )
    image[512:1536] = make_dynamic_header(BLOCK_COUNT, PARENT_UID, PARENT_TIMESTAMP, parent_path, [locator])
    image += make_footer(DISK_TYPE_DIFFERENCING, 512, BLOCK_SIZE * BLOCK_COUNT, CHILD_UID, 0)
    path.write_bytes(bytes(image))


@pytest.fixture
def vhd_set(tmp_path: Path) -> dict[str, Path]:
    parent = tmp_path / PARENT_NAME
    fixed = tmp_path / "fixed.vhd"
    dynamic = tmp_path / "dynamic.vhd"
    differencing = tmp_path / "child.vhd"

    build_parent(parent)
    build_fixed_child(fixed)
    build_dynamic_child(dynamic)
    build_differencing_child(differencing)
    return {"parent": parent, "fixed": fixed, "dynamic": dynamic, "differencing": differencing}


def test_fixed_disk_requests_are_trusted(vhd_set: dict[str, Path]) -> None:
    with vhd_set["fixed"].open("rb") as fh:
        vhd = VHD(fh)
        assert vhd.disk_type == DISK_TYPE_FIXED
        assert vhd.read(SECTOR_SIZE) == bytes([DYNAMIC_MARKER]) * SECTOR_SIZE

    report = diagnose(vhd_set["fixed"])
    assert report.reader_verdict == "supported"
    assert report.content_trust == "trusted"
    assert report.blocks is None


def test_dynamic_disk_distinguishes_present_and_sparse_blocks(vhd_set: dict[str, Path]) -> None:
    with vhd_set["dynamic"].open("rb") as fh:
        vhd = VHD(fh)
        assert vhd.disk_type == DISK_TYPE_DYNAMIC
        assert vhd.read(SECTOR_SIZE) == bytes([DYNAMIC_MARKER]) * SECTOR_SIZE

        vhd.seek(BLOCK_SIZE)
        assert vhd.read(SECTOR_SIZE) == b"\x00" * SECTOR_SIZE

        vhd.seek(2 * BLOCK_SIZE)
        assert vhd.read(SECTOR_SIZE) == bytes([DYNAMIC_MARKER_2]) * SECTOR_SIZE

    report = diagnose(vhd_set["dynamic"])
    assert [block.state for block in report.blocks] == [
        "allocated",
        "zero",
        "allocated",
        "zero",
        "zero",
        "zero",
        "zero",
        "zero",
    ]
    assert report.block_summary == {"allocated": 2, "zero": 6}
    assert report.content_trust == "trusted"


def test_differencing_disk_is_refused_instead_of_silently_flattened(vhd_set: dict[str, Path]) -> None:
    with vhd_set["differencing"].open("rb") as fh:
        with pytest.raises(InvalidVirtualDisk, match="Differencing VHD files are not supported"):
            VHD(fh)

    report = diagnose(vhd_set["differencing"])
    assert report.disk_type == DISK_TYPE_DIFFERENCING
    assert report.reader_verdict == "refused: parent chain not implemented"
    assert report.content_trust == "untrusted"
    assert report.parent_unique_id == PARENT_UID
    assert report.parent_timestamp == PARENT_TIMESTAMP
    assert report.parent_reference == PARENT_NAME
    assert [block.state for block in report.blocks] == [
        "allocated",
        "parent",
        "partial_parent",
        "parent",
        "parent",
        "parent",
        "parent",
        "parent",
    ]
    assert any("7 block(s) require parent data" in warning for warning in report.warnings)
    assert report.block_summary == {"allocated": 1, "partial_parent": 1, "parent": 6}


def test_diagnosis_does_not_confuse_relative_locator_with_parent_content(vhd_set: dict[str, Path]) -> None:
    report = diagnose(vhd_set["differencing"])
    assert len(report.parent_locators or []) == 1
    locator = report.parent_locators[0]
    assert locator.kind == "parent_unicode_relative"
    assert locator.value == PARENT_NAME

    resolved = resolve_parent_path(report)
    assert resolved == vhd_set["parent"]
    verification = verify_parent(report, vhd_set["parent"])
    assert verification.ok
    assert verification.checks == {
        "parent_parseable": True,
        "parent_unique_id": True,
        "parent_timestamp": True,
        "parent_size": True,
        "parent_not_differencing": True,
    }


def test_wrong_parent_identity_is_rejected_even_when_filename_matches(vhd_set: dict[str, Path]) -> None:
    wrong = vhd_set["fixed"]
    report = diagnose(vhd_set["differencing"])
    verification = verify_parent(report, wrong)
    assert not verification.ok
    assert verification.checks["parent_unique_id"] is False
    assert any("Parent unique ID mismatch" in note for note in verification.notes or [])


def test_parent_size_mismatch_is_rejected(tmp_path: Path, vhd_set: dict[str, Path]) -> None:
    wrong = tmp_path / "wrong-size.vhd"
    wrong.write_bytes(
        bytes([PARENT_MARKER]) * BLOCK_SIZE
        + make_footer(DISK_TYPE_FIXED, 0xFFFFFFFFFFFFFFFF, BLOCK_SIZE, PARENT_UID, PARENT_TIMESTAMP)
    )

    report = diagnose(vhd_set["differencing"])
    verification = verify_parent(report, wrong)
    assert not verification.ok
    assert verification.checks["parent_unique_id"] is True
    assert verification.checks["parent_size"] is False
    assert any("Virtual size mismatch" in note for note in verification.notes or [])


def test_bitmap_blind_flat_read_would_return_poison_not_parent(vhd_set: dict[str, Path]) -> None:
    # A bitmap-blind reader (the historical behavior) skips the bitmap and reads child block bytes
    # for every sector, including sectors whose bitmap bit is clear. Make that misleading result explicit.
    report = diagnose(vhd_set["differencing"])
    partial = next(block for block in report.blocks if block.state == "partial_parent")
    assert partial.bat_entry is not None

    bitmap_sectors = 1
    data_offset = (partial.bat_entry + bitmap_sectors) * SECTOR_SIZE
    with vhd_set["differencing"].open("rb") as fh:
        fh.seek(data_offset)
        child_block = fh.read(BLOCK_SIZE)

    # The first four sectors are genuinely child-owned; the last four are poison in the child file.
    assert child_block[: 4 * SECTOR_SIZE] == bytes([PARTIAL_MARKER]) * (4 * SECTOR_SIZE)
    assert child_block[4 * SECTOR_SIZE :] == bytes([POISON_MARKER]) * (4 * SECTOR_SIZE)

    # Correct overlay would keep the four child sectors and inherit PARENT_MARKER for the clear bits.
    with vhd_set["parent"].open("rb") as fh:
        parent_block = fh.read(BLOCK_SIZE)
    expected = bytes([PARTIAL_MARKER]) * (4 * SECTOR_SIZE) + bytes([PARENT_MARKER]) * (4 * SECTOR_SIZE)
    assert expected != child_block
    assert expected[4 * SECTOR_SIZE :] == parent_block[4 * SECTOR_SIZE :]


def test_dynamic_disk_with_partial_bitmap_is_not_reported_trusted(tmp_path: Path) -> None:
    path = tmp_path / "mislabeled.vhd"
    entries = [DATA_OFFSET // SECTOR_SIZE] + [0xFFFFFFFF] * (BLOCK_COUNT - 1)
    block = make_block(0xF0, PARTIAL_MARKER)
    image = bytearray(DATA_OFFSET + len(block))
    image[512:1536] = make_dynamic_header(BLOCK_COUNT)
    struct.pack_into(f">{BLOCK_COUNT}I", image, TABLE_OFFSET, *entries)
    image[DATA_OFFSET : DATA_OFFSET + len(block)] = block
    image += make_footer(DISK_TYPE_DYNAMIC, 512, BLOCK_SIZE * BLOCK_COUNT, CHILD_UID, 0)
    path.write_bytes(bytes(image))

    report = diagnose(path)
    assert report.blocks[0].state == "ambiguous"
    assert report.content_trust == "untrusted"

    with path.open("rb") as fh:
        vhd = VHD(fh)
        with pytest.raises(InvalidVirtualDisk, match="unset sector bitmap bit"):
            vhd.read(SECTOR_SIZE)


def test_footer_type_and_data_offset_conflict_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "conflict.vhd"
    footer_bytes = make_footer(DISK_TYPE_FIXED, 512, BLOCK_SIZE, CHILD_UID, 0)
    path.write_bytes(b"\x00" * BLOCK_SIZE + footer_bytes)

    report = diagnose(path)
    assert report.reader_verdict == "refused: footer disk type conflicts with data_offset"
    with path.open("rb") as fh:
        with pytest.raises(InvalidVirtualDisk, match="declares a fixed disk"):
            VHD(fh)


def test_footer_checksum_failure_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "bad-checksum.vhd"
    build_fixed_child(path)
    raw = bytearray(path.read_bytes())
    # Footer is the final 512 bytes; its checksum field starts at relative offset 64.
    footer_offset = len(raw) - 512
    raw[footer_offset + 64] ^= 0x01
    path.write_bytes(bytes(raw))

    report = diagnose(path)
    assert report.footer_checksum_valid is False
    assert report.reader_verdict == "refused: invalid footer checksum"
    with path.open("rb") as fh:
        with pytest.raises(InvalidHeaderError):
            VHD(fh)


def test_truncated_fixed_disk_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "truncated-fixed.vhd"
    footer_bytes = make_footer(DISK_TYPE_FIXED, 0xFFFFFFFFFFFFFFFF, BLOCK_SIZE * 2, CHILD_UID, 0)
    path.write_bytes(b"\x00" * BLOCK_SIZE + footer_bytes)

    report = diagnose(path)
    assert report.content_trust == "untrusted"
    assert any("Fixed image is truncated" in warning for warning in report.warnings or [])
    with path.open("rb") as fh:
        with pytest.raises(InvalidVirtualDisk, match="fixed disk is truncated"):
            VHD(fh)


def test_bat_entry_zero_is_rejected_not_treated_as_sparse(tmp_path: Path) -> None:
    path = tmp_path / "bat-zero.vhd"
    entries = [0] + [0xFFFFFFFF] * (BLOCK_COUNT - 1)
    image = bytearray(DATA_OFFSET)
    image[512:1536] = make_dynamic_header(BLOCK_COUNT)
    struct.pack_into(f">{BLOCK_COUNT}I", image, TABLE_OFFSET, *entries)
    image += make_footer(DISK_TYPE_DYNAMIC, 512, BLOCK_SIZE * BLOCK_COUNT, CHILD_UID, 0)
    path.write_bytes(bytes(image))

    report = diagnose(path)
    assert report.blocks[0].state == "corrupt"
    with path.open("rb") as fh:
        vhd = VHD(fh)
        with pytest.raises(InvalidVirtualDisk, match="Invalid BAT entry 0"):
            vhd.read(SECTOR_SIZE)


def test_truncated_allocated_block_is_flagged(tmp_path: Path) -> None:
    path = tmp_path / "truncated-block.vhd"
    entries = [DATA_OFFSET // SECTOR_SIZE] + [0xFFFFFFFF] * (BLOCK_COUNT - 1)
    image = bytearray(DATA_OFFSET + SECTOR_SIZE)
    image[512:1536] = make_dynamic_header(BLOCK_COUNT)
    struct.pack_into(f">{BLOCK_COUNT}I", image, TABLE_OFFSET, *entries)
    # Only a bitmap sector is present; the 4096-byte data block is missing.
    image[DATA_OFFSET:] = b"\xff" * SECTOR_SIZE
    image += make_footer(DISK_TYPE_DYNAMIC, 512, BLOCK_SIZE * BLOCK_COUNT, CHILD_UID, 0)
    path.write_bytes(bytes(image))

    report = diagnose(path)
    assert report.blocks[0].state == "corrupt"
    assert "Block data extends past end of file" in report.blocks[0].detail


def test_unreadable_parent_locator_is_reported_not_silently_ignored(tmp_path: Path) -> None:
    path = tmp_path / "bad-locator.vhd"
    entries = [0xFFFFFFFF] * BLOCK_COUNT
    image = bytearray(DATA_OFFSET)
    locator = c_vhd.parent_locator(
        platform_code=PLATFORM_CODE_W2RU,
        platform_data_space=SECTOR_SIZE,
        platform_data_length=20,
        reserved=0,
        platform_data_offset=10**9,
    )
    image[512:1536] = make_dynamic_header(BLOCK_COUNT, PARENT_UID, PARENT_TIMESTAMP, PARENT_NAME, [locator])
    struct.pack_into(f">{BLOCK_COUNT}I", image, TABLE_OFFSET, *entries)
    image += make_footer(DISK_TYPE_DIFFERENCING, 512, BLOCK_SIZE * BLOCK_COUNT, CHILD_UID, 0)
    path.write_bytes(bytes(image))

    report = diagnose(path)
    assert report.parent_locators[0].value is None
    assert any("Parent locator 0" in warning for warning in report.warnings or [])


def test_block_limit_keeps_large_image_diagnostic_bounded(vhd_set: dict[str, Path]) -> None:
    report = diagnose(vhd_set["dynamic"], block_limit=1)
    assert [block.block for block in report.blocks or []] == [0]
    assert "Block assessment limited to 1 of 8 blocks." in (report.warnings or [])


def test_diagnostic_json_and_text_are_runnable(
    vhd_set: dict[str, Path], capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from dissect.hypervisor.tools import vhd_diagnose

    monkeypatch.setattr(
        "sys.argv",
        ["vhd-diagnose", str(vhd_set["differencing"]), "--json"],
    )
    assert vhd_diagnose.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["disk_type_name"] == "differencing"
    assert payload["content_trust"] == "untrusted"
    assert payload["block_summary"] == {"allocated": 1, "partial_parent": 1, "parent": 6}
    assert payload["parent_verification"]["ok"] is True

    text = format_diagnostics(diagnose(vhd_set["differencing"]))
    assert "Support matrix" in text
    assert "partial_parent" in text
    assert "refused: parent chain not implemented" in text
