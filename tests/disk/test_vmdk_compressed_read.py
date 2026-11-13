from __future__ import annotations

import io
import os
import struct
import zlib

import pytest

from dissect.hypervisor.disk.c_vmdk import SECTOR_SIZE, c_vmdk
from dissect.hypervisor.disk.vmdk import VMDK, VMDK_MAGIC, SparseDisk

# Small forensic geometry so a single image stays cheap but still exercises
# multiple grain tables and every interesting grain kind.
GRAIN_SECTORS = 8
GRAIN_BYTES = GRAIN_SECTORS * SECTOR_SIZE
GRAIN_TABLE_ENTRIES = 2
# Adjacent logical grains deliberately have a different compressed record length.
# A = allocated compressed grain, Z = explicit zero grain (GTE == 1),
# P = grain not present in the child, served from the parent disk (GTE == 0).
GRAIN_KINDS = ("A", "A", "Z", "A", "P", "A", "P", "A")
NUM_GRAINS = len(GRAIN_KINDS)
CAPACITY_SECTORS = NUM_GRAINS * GRAIN_SECTORS
DESCRIPTOR_SECTOR = 1
GRAIN_DIRECTORY_SECTOR = 2
GRAIN_TABLE_START = 3
NUM_GRAIN_TABLES = (NUM_GRAINS + GRAIN_TABLE_ENTRIES - 1) // GRAIN_TABLE_ENTRIES
DATA_START_SECTOR = GRAIN_TABLE_START + NUM_GRAIN_TABLES


def _grain_payload(grain: int, salt: int) -> bytes:
    """Return a unique, low-entropy grain so each compressed record has a distinct length."""
    marker = (grain * 31 + salt) & 0xFF
    data = bytearray([marker] * GRAIN_BYTES)
    seed = (0x9E3779B9 + grain * 2_654_435_761 + salt) & 0xFFFFFFFF
    for offset in range((grain + 1) * 48):
        seed = (seed * 1_103_515_245 + 12345) & 0xFFFFFFFF
        data[offset % GRAIN_BYTES] ^= (seed >> 16) & 0xFF
    for offset in range(0, GRAIN_BYTES, 251):
        data[offset] = grain & 0xFF
    return bytes(data)


def _pad_to_sector(data: bytearray) -> None:
    remainder = len(data) % SECTOR_SIZE
    if remainder:
        data.extend(b"\x00" * (SECTOR_SIZE - remainder))


def _sparse_header(flags: int, capacity: int, grain_size: int, grain_table_entries: int, overhead: int) -> bytes:
    header = c_vmdk.VMDKSparseExtentHeader()
    header.magic = VMDK_MAGIC
    header.version = c_vmdk.SPARSE_VERSION_INCOMPAT_FLAGS
    header.flags = flags
    header.capacity = capacity
    header.grain_size = grain_size
    header.descriptor_offset = DESCRIPTOR_SECTOR
    header.descriptor_size = 1
    header.num_grain_table_entries = grain_table_entries
    header.secondary_grain_directory_offset = 0
    header.primary_grain_directory_offset = GRAIN_DIRECTORY_SECTOR
    header.overhead = overhead
    header.is_dirty = 0
    header.single_end_line_char = b"\n"
    header.non_end_line_char = b" "
    header.double_end_line_chars = b"\r\n"
    header.compress_algorithm = c_vmdk.SPARSE_COMPRESSALGORITHM_DEFLATE if flags & c_vmdk.SPARSEFLAG_COMPRESSED else 0
    return header.dumps()


def _descriptor(capacity: int) -> bytes:
    descriptor = (
        "# Disk DescriptorFile\n"
        "version=1\n"
        "CID=aaaaaaaa\n"
        "parentCID=ffffffff\n"
        'createType="streamOptimized"\n\n'
        "# Extent Description\n"
        f'RW {capacity} SPARSE "forensic.vmdk"\n'
    ).encode()
    return descriptor.ljust(SECTOR_SIZE, b"\x00")[:SECTOR_SIZE]


def _build_parent_image() -> tuple[bytes, dict[int, bytes]]:
    """Build an uncompressed parent that backs every logical grain."""
    payloads = {grain: _grain_payload(grain, salt=0x77) for grain in range(NUM_GRAINS)}
    image = bytearray(b"\x00" * SECTOR_SIZE)
    image.extend(_descriptor(CAPACITY_SECTORS))
    image.extend(
        b"".join(struct.pack("<I", GRAIN_TABLE_START + table) for table in range(NUM_GRAIN_TABLES)).ljust(
            SECTOR_SIZE, b"\x00"
        )
    )

    grain_table_entries = []
    for grain in range(NUM_GRAINS):
        grain_table_entries.append(DATA_START_SECTOR + grain * GRAIN_SECTORS)
    for table in range(NUM_GRAIN_TABLES):
        entries = grain_table_entries[table * GRAIN_TABLE_ENTRIES : (table + 1) * GRAIN_TABLE_ENTRIES]
        image.extend(b"".join(struct.pack("<I", entry) for entry in entries).ljust(SECTOR_SIZE, b"\x00"))

    for grain in range(NUM_GRAINS):
        image.extend(payloads[grain])

    image[0 : len(_sparse_header(0, CAPACITY_SECTORS, GRAIN_SECTORS, GRAIN_TABLE_ENTRIES, DATA_START_SECTOR))] = (
        _sparse_header(0, CAPACITY_SECTORS, GRAIN_SECTORS, GRAIN_TABLE_ENTRIES, DATA_START_SECTOR)
    )
    return bytes(image), payloads


def _build_child_image(
    payloads: dict[int, bytes],
    embedded_lba: bool,
    contiguous_slots: bool,
) -> tuple[bytes, list[int], dict[int, int]]:
    """Build a compressed child image and return the bytes, GTEs and physical record sectors."""
    flags = c_vmdk.SPARSEFLAG_COMPRESSED
    if embedded_lba:
        flags |= c_vmdk.SPARSEFLAG_EMBEDDED_LBA

    records: dict[int, bytes] = {}
    for grain, data in payloads.items():
        compressed = zlib.compress(data)
        if embedded_lba:
            header = struct.pack("<QI", grain * GRAIN_SECTORS, len(compressed))
        else:
            header = struct.pack("<I", len(compressed))
        records[grain] = header + compressed

    grain_table_entries: list[int] = []
    data_area = bytearray()
    physical_sectors: dict[int, int] = {}
    packed_sector = DATA_START_SECTOR

    for grain, kind in enumerate(GRAIN_KINDS):
        if kind == "Z":
            grain_table_entries.append(1)
            continue
        if kind == "P":
            grain_table_entries.append(0)
            continue

        record = records[grain]
        if contiguous_slots:
            # Every logical grain reserves its own grain_size-spaced slot, including the gaps left
            # by zero/missing grains. Two records that are physically adjacent therefore still sit
            # grain_size sectors apart, which is exactly what makes get_runs consolidate them.
            assert len(record) <= GRAIN_BYTES
            sector = DATA_START_SECTOR + grain * GRAIN_SECTORS
            start = (sector - DATA_START_SECTOR) * SECTOR_SIZE
            end = start + len(record)
            if len(data_area) < end:
                data_area.extend(b"\x00" * (end - len(data_area)))
            data_area[start:end] = record
        else:
            # Realistic packed layout: records are stored back to back and sector padded, so their
            # GTEs differ by the (varying) compressed length rather than by grain_size.
            sector = packed_sector
            start = (sector - DATA_START_SECTOR) * SECTOR_SIZE
            if len(data_area) < start:
                data_area.extend(b"\x00" * (start - len(data_area)))
            data_area.extend(record)
            _pad_to_sector(data_area)
            packed_sector = DATA_START_SECTOR + len(data_area) // SECTOR_SIZE

        physical_sectors[grain] = sector
        grain_table_entries.append(sector)

    image = bytearray(b"\x00" * SECTOR_SIZE)
    image.extend(_descriptor(CAPACITY_SECTORS))
    image.extend(
        b"".join(struct.pack("<I", GRAIN_TABLE_START + table) for table in range(NUM_GRAIN_TABLES)).ljust(
            SECTOR_SIZE, b"\x00"
        )
    )
    for table in range(NUM_GRAIN_TABLES):
        entries = grain_table_entries[table * GRAIN_TABLE_ENTRIES : (table + 1) * GRAIN_TABLE_ENTRIES]
        image.extend(b"".join(struct.pack("<I", entry) for entry in entries).ljust(SECTOR_SIZE, b"\x00"))
    image.extend(data_area)

    header = _sparse_header(flags, CAPACITY_SECTORS, GRAIN_SECTORS, GRAIN_TABLE_ENTRIES, DATA_START_SECTOR)
    image[0 : len(header)] = header
    return bytes(image), grain_table_entries, physical_sectors


def _expected_logical_image(
    child_payloads: dict[int, bytes], parent_payloads: dict[int, bytes], with_parent: bool
) -> bytearray:
    """Build the expected bytes independently of get_runs and the reader under test."""
    expected = bytearray()
    for grain, kind in enumerate(GRAIN_KINDS):
        if kind == "A":
            expected.extend(child_payloads[grain])
        elif kind == "Z":
            expected.extend(b"\x00" * GRAIN_BYTES)
        elif with_parent:
            expected.extend(parent_payloads[grain])
        else:
            # A missing grain with no parent reads back as zeros.
            expected.extend(b"\x00" * GRAIN_BYTES)
    return expected


def _assert_bytes_exact(
    disk: SparseDisk,
    expected: bytes,
    start_sector: int,
    sector_count: int,
    grain_table_entries: list[int],
    label: str,
) -> None:
    try:
        actual = disk.read_sectors(start_sector, sector_count)
    except Exception as error:
        _persist_image(disk, label)
        raise AssertionError(
            f"{label}: read_sectors({start_sector}, {sector_count}) raised {type(error).__name__}: {error}"
        ) from error

    wanted = expected[start_sector * SECTOR_SIZE : (start_sector + sector_count) * SECTOR_SIZE]
    if actual == wanted:
        return

    for offset, (got_byte, want_byte) in enumerate(zip(actual, wanted)):
        if got_byte != want_byte:
            absolute_sector = start_sector + offset // SECTOR_SIZE
            grain = absolute_sector // GRAIN_SECTORS
            physical_sector = disk._lookup_grain(grain)
            _persist_image(disk, label)
            raise AssertionError(
                f"{label}: first mismatch at logical sector {absolute_sector} (byte "
                f"{absolute_sector * SECTOR_SIZE + (offset % SECTOR_SIZE)}), grain {grain} kind "
                f"{GRAIN_KINDS[grain]}, GTE/physical sector {physical_sector} "
                f"(file offset {physical_sector * SECTOR_SIZE:#x}), GTEs {grain_table_entries}, "
                f"got {got_byte:#04x} want {want_byte:#04x}"
            )
    raise AssertionError(f"{label}: length mismatch, got {len(actual)} bytes want {len(wanted)}")


def _persist_image(disk: SparseDisk, label: str) -> None:
    """Keep a minimal trigger image on disk so a failure can be inspected offline."""
    target = os.environ.get("VMDK_REGRESSION_DUMP_DIR")
    if not target:
        return
    os.makedirs(target, exist_ok=True)
    disk.fh.seek(0, io.SEEK_END)
    size = disk.fh.tell()
    disk.fh.seek(0)
    path = os.path.join(target, f"{label}.vmdk")
    with open(path, "wb") as handle:
        handle.write(disk.fh.read(size))


def _open_child(image: bytes, with_parent: bool) -> SparseDisk:
    child = SparseDisk(io.BytesIO(image))
    if with_parent:
        parent_image, _ = _build_parent_image()
        child.parent = SparseDisk(io.BytesIO(parent_image))
    return child


@pytest.mark.parametrize("embedded_lba", [False, True], ids=["cmp-header", "embedded-lba"])
@pytest.mark.parametrize("contiguous_slots", [False, True], ids=["packed", "contiguous-slots"])
@pytest.mark.parametrize("with_parent", [False, True], ids=["no-parent", "with-parent"])
def test_compressed_grain_reads_match_independent_logical_image(
    embedded_lba: bool, contiguous_slots: bool, with_parent: bool
) -> None:
    """Large reads, small reads and seeks over compressed grains must all match the logical image."""
    child_payloads = {grain: _grain_payload(grain, salt=0x11) for grain, kind in enumerate(GRAIN_KINDS) if kind == "A"}
    _, parent_payloads = _build_parent_image()
    image, grain_table_entries, physical_sectors = _build_child_image(child_payloads, embedded_lba, contiguous_slots)

    # The sample must keep adjacent allocated grains at different compressed lengths.
    allocated = [grain for grain, kind in enumerate(GRAIN_KINDS) if kind == "A"]
    record_lengths = []
    for grain in allocated:
        data = child_payloads[grain]
        header_len = c_vmdk.SparseGrainLBAHeaderOnDisk.size if embedded_lba else 4
        record_lengths.append(header_len + len(zlib.compress(data)))
    assert any(left != right for left, right in zip(record_lengths, record_lengths[1:]))

    disk = _open_child(image, with_parent)
    expected = bytes(_expected_logical_image(child_payloads, parent_payloads, with_parent))
    label = f"compressed-{ 'embedded' if embedded_lba else 'header' }-{ 'contig' if contiguous_slots else 'packed' }-{ 'parent' if with_parent else 'standalone' }"

    # Explicit zero grains and missing grains must be represented as expected.
    assert disk._lookup_grain(2) == 1
    assert disk._lookup_grain(4) == 0
    assert disk._lookup_grain(6) == 0

    # One large read spanning every grain, including the grain-table boundaries.
    _assert_bytes_exact(disk, expected, 0, CAPACITY_SECTORS, grain_table_entries, f"{label}-large")

    # Many small reads whose size is unrelated to the grain/sector geometry.
    assembled = bytearray()
    sector = 0
    step = 3
    while sector < CAPACITY_SECTORS:
        count = min(step, CAPACITY_SECTORS - sector)
        assembled.extend(disk.read_sectors(sector, count))
        sector += count
    assert bytes(assembled) == expected

    # Reads after seeks to grain boundaries and awkward mid-grain offsets.
    for start_sector in (0, 1, GRAIN_SECTORS - 1, GRAIN_SECTORS, 3 * GRAIN_SECTORS + 2, 5 * GRAIN_SECTORS):
        count = min(GRAIN_SECTORS, CAPACITY_SECTORS - start_sector)
        _assert_bytes_exact(disk, expected, start_sector, count, grain_table_entries, f"{label}-seek")

    # Every physical record sector must be a real allocated GTE, never a synthetic jump target.
    for grain, sector in physical_sectors.items():
        assert disk._lookup_grain(grain) == sector


def test_compressed_grain_consolidation_advances_to_correct_records() -> None:
    """A consolidated compressed run must read each grain from its own physical record."""
    child_payloads = {grain: _grain_payload(grain, salt=0x11) for grain, kind in enumerate(GRAIN_KINDS) if kind == "A"}
    _, parent_payloads = _build_parent_image()
    image, grain_table_entries, physical_sectors = _build_child_image(
        child_payloads, embedded_lba=False, contiguous_slots=True
    )
    disk = _open_child(image, with_parent=True)
    expected = bytes(_expected_logical_image(child_payloads, parent_payloads, True))

    # Grains 0 and 1 are adjacent allocated grains with grain_size-spaced GTEs, so get_runs is
    # forced to consolidate them into a single run even though their compressed lengths differ.
    assert physical_sectors[1] - physical_sectors[0] == GRAIN_SECTORS
    consolidated = disk.get_runs(0, 2 * GRAIN_SECTORS)
    assert consolidated == [(physical_sectors[0], 0, 2 * GRAIN_SECTORS, None)]

    # Reading the consolidated run must still return the two distinct grain payloads in order.
    _assert_bytes_exact(
        disk,
        expected,
        0,
        2 * GRAIN_SECTORS,
        grain_table_entries,
        "consolidated-run",
    )

    # A read starting mid-grain inside the consolidated run keeps the correct per-grain offset.
    _assert_bytes_exact(
        disk,
        expected,
        2,
        GRAIN_SECTORS + 3,
        grain_table_entries,
        "consolidated-run-midgrain",
    )


def test_vmdk_stream_seek_and_read_consistency() -> None:
    """The public VMDK stream must stay consistent across buffered seeks and sequential reads."""
    child_payloads = {grain: _grain_payload(grain, salt=0x11) for grain, kind in enumerate(GRAIN_KINDS) if kind == "A"}
    _, parent_payloads = _build_parent_image()
    image, _, _ = _build_child_image(child_payloads, embedded_lba=True, contiguous_slots=False)
    stream = VMDK(io.BytesIO(image))
    expected = bytes(_expected_logical_image(child_payloads, parent_payloads, False))

    assert stream.read(GRAIN_BYTES) == expected[:GRAIN_BYTES]

    stream.seek(3 * GRAIN_BYTES + 17)
    position = 3 * GRAIN_BYTES + 17
    chunk = stream.read(2 * GRAIN_BYTES)
    assert chunk == expected[position : position + 2 * GRAIN_BYTES]

    stream.seek(0)
    sequential = bytearray()
    while len(sequential) < len(expected):
        sequential.extend(stream.read(1234))
    assert bytes(sequential) == expected
