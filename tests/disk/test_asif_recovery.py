from __future__ import annotations

import hashlib
import io
import plistlib
from dataclasses import dataclass

import pytest

from dissect.hypervisor.disk.asif import ASIF
from dissect.hypervisor.disk.c_asif import c_asif
from dissect.hypervisor.exceptions import ASIFRecoveryError

# Small, fully controlled geometry. A 32 KiB chunk keeps generated images tiny while still satisfying
# the loader's real header/table/directory arithmetic (4 KiB chunks leave no usable table entries).
BLOCK_SIZE = 512
CHUNK_SIZE = 32768
VIRTUAL_SECTORS = 192  # 96 KiB virtual disk, three data chunks
MAX_SECTORS = 512  # 256 KiB maximum size, enough to address the reserved metadata chunk

# Fixed physical layout of the generated image (chunk numbers). Metadata and table B precede the
# newest directory so truncating the directory tail leaves the metadata block intact.
DIR_A_CHUNK = 1
TABLE_A_CHUNK = 2
DATA_A_CHUNK = 3
DATA_B_CHUNK = 4
METADATA_CHUNK = 5
TABLE_B_CHUNK = 6
DIR_B_CHUNK = 7

DATA_A_PATTERN = 0xA0
DATA_B_PATTERN = 0xB0
MISSING_CHUNK = 99


@dataclass(frozen=True)
class Geometry:
    num_directory_entries: int
    num_table_entries: int
    num_reserved_table_entries: int
    size_per_table: int
    metadata_slot: int


@dataclass(frozen=True)
class BuiltImage:
    """A generated ASIF image plus the facts tests need to assert against."""

    data: bytes
    geometry: Geometry
    data_a_chunk: int
    data_b_chunk: int
    metadata_chunk: int
    directory_b_offset: int
    table_b_offset: int


def compute_geometry() -> Geometry:
    """Mirror the loader's own layout arithmetic for the small test geometry."""
    blocks_per_chunk = CHUNK_SIZE // BLOCK_SIZE
    num_reserved_table_entries = (4 * CHUNK_SIZE) // blocks_per_chunk
    max_table_entries = CHUNK_SIZE >> 3
    num_table_entries = max_table_entries - (
        max_table_entries % (num_reserved_table_entries + 1)
    )
    num_reserved_directory_entries = (num_reserved_table_entries + num_table_entries) // (
        num_reserved_table_entries + 1
    )
    num_usable_entries = num_table_entries - num_reserved_directory_entries
    size_per_table = num_usable_entries * CHUNK_SIZE
    max_size = MAX_SECTORS * BLOCK_SIZE
    num_directory_entries = (size_per_table + max_size - 1) // size_per_table

    # The reserved-area metadata chunk is addressed through the same table entry formula as data.
    meta_relative_block = (METADATA_CHUNK * CHUNK_SIZE) // BLOCK_SIZE
    meta_slot = (
        meta_relative_block // blocks_per_chunk
        + (meta_relative_block // blocks_per_chunk) * num_reserved_table_entries
    ) // num_reserved_table_entries
    return Geometry(
        num_directory_entries=num_directory_entries,
        num_table_entries=num_table_entries,
        num_reserved_table_entries=num_reserved_table_entries,
        size_per_table=size_per_table,
        metadata_slot=meta_slot,
    )


def _build_table(geometry: Geometry, metadata_chunk: int, data_mapping: dict[int, int]) -> bytes:
    entries = [0] * geometry.num_table_entries
    for data_slot, chunk in data_mapping.items():
        index = (
            data_slot + data_slot * geometry.num_reserved_table_entries
        ) // geometry.num_reserved_table_entries
        entries[index] = chunk
    entries[geometry.metadata_slot] = metadata_chunk
    return b"".join(entry.to_bytes(8, "big") for entry in entries)


def _build_directory(version: int, table_chunk: int, num_directory_entries: int) -> bytes:
    entries = [0] * num_directory_entries
    entries[0] = table_chunk
    return version.to_bytes(8, "big") + b"".join(entry.to_bytes(8, "big") for entry in entries)


def build_image() -> BuiltImage:
    """Build a valid in-memory ASIF image with two complete, distinct directory versions."""
    geometry = compute_geometry()
    chunks: dict[int, bytes] = {}

    chunks[DIR_A_CHUNK] = _build_directory(1, TABLE_A_CHUNK, geometry.num_directory_entries)
    chunks[DIR_B_CHUNK] = _build_directory(2, TABLE_B_CHUNK, geometry.num_directory_entries)

    chunks[TABLE_A_CHUNK] = _build_table(
        geometry, METADATA_CHUNK, {0: DATA_A_CHUNK, 1: DATA_A_CHUNK, 2: DATA_A_CHUNK}
    )
    chunks[TABLE_B_CHUNK] = _build_table(
        geometry, METADATA_CHUNK, {0: DATA_B_CHUNK, 1: DATA_B_CHUNK, 2: DATA_B_CHUNK}
    )

    chunks[DATA_A_CHUNK] = bytes([DATA_A_PATTERN]) * CHUNK_SIZE
    chunks[DATA_B_CHUNK] = bytes([DATA_B_PATTERN]) * CHUNK_SIZE

    plist = plistlib.dumps(
        {
            "internal metadata": {"stable uuid": "00000000-0000-0000-0000-000000000001"},
            "user metadata": {"case": "recovery-fixture"},
        }
    )
    meta_header = c_asif.asif_meta_header(
        header_signature=c_asif.ASIF_META_HEADER_SIGNATURE,
        header_version=1,
        header_size=c_asif.asif_meta_header.size,
        data_size=len(plist),
        unk_14=0,
    ).dumps()
    metadata = meta_header + plist
    chunks[METADATA_CHUNK] = metadata + b"\x00" * (CHUNK_SIZE - len(metadata))

    total_size = (max(chunks) + 1) * CHUNK_SIZE
    image = bytearray(total_size)
    for chunk, payload in chunks.items():
        start = chunk * CHUNK_SIZE
        image[start : start + len(payload)] = payload

    header = c_asif.asif_header(
        header_signature=c_asif.ASIF_HEADER_SIGNATURE,
        header_version=1,
        header_size=c_asif.asif_header.size,
        header_flags=0,
        directory_offsets=[DIR_A_CHUNK * CHUNK_SIZE, DIR_B_CHUNK * CHUNK_SIZE],
        guid=b"\x01" * 16,
        sector_count=VIRTUAL_SECTORS,
        max_sector_count=MAX_SECTORS,
        chunk_size=CHUNK_SIZE,
        block_size=BLOCK_SIZE,
        total_segments=1,
        metadata_chunk=METADATA_CHUNK,
        unk_50=b"\x00" * 16,
        read_only_flags=0,
        metadata_flags=0,
        metadata_read_only_flags=0,
    ).dumps()
    image[: len(header)] = header

    return BuiltImage(
        data=bytes(image),
        geometry=geometry,
        data_a_chunk=DATA_A_CHUNK,
        data_b_chunk=DATA_B_CHUNK,
        metadata_chunk=METADATA_CHUNK,
        directory_b_offset=DIR_B_CHUNK * CHUNK_SIZE,
        table_b_offset=TABLE_B_CHUNK * CHUNK_SIZE,
    )


def truncate_newest_directory(image: BuiltImage) -> bytes:
    """Keep the newest directory's version field but cut off its table entry array."""
    return image.data[: image.directory_b_offset + 8]


def retarget_newest_table_to_missing_chunk(image: BuiltImage) -> bytes:
    """Rewrite the newest directory's table pointer at a chunk that does not exist."""
    corrupted = bytearray(image.data)
    pointer = image.directory_b_offset + 8
    corrupted[pointer : pointer + 8] = MISSING_CHUNK.to_bytes(8, "big")
    return bytes(corrupted)


def corrupt_metadata_only(image: BuiltImage) -> bytes:
    """Corrupt only the reserved-area plist payload, leaving directories and data intact."""
    corrupted = bytearray(image.data)
    plist_start = image.metadata_chunk * CHUNK_SIZE + c_asif.asif_meta_header.size
    corrupted[plist_start : plist_start + 16] = b"\xff" * 16
    return bytes(corrupted)


def test_healthy_image_reports_both_versions_and_metadata() -> None:
    image = build_image()
    report = ASIF.inspect_recovery(io.BytesIO(image.data))

    assert report.active_version == 2
    assert report.recommended_version == 2
    assert [status.version for status in report.directories] == [2, 1]
    assert all(status.readable and status.complete for status in report.directories)
    assert report.data.readable
    assert report.metadata.readable
    assert report.metadata.internal_metadata_readable
    assert report.metadata.user_metadata_readable


def test_truncated_newest_directory_fails_cleanly_and_old_directory_recovers() -> None:
    image = build_image()
    raw = truncate_newest_directory(image)
    report = ASIF.inspect_recovery(io.BytesIO(raw))

    # The newest directory header (version) is readable, but its table entry array is truncated.
    newest, oldest = report.directories
    assert newest.version == 2
    assert newest.readable is False
    assert isinstance(newest.error, str)
    assert oldest.version == 1
    assert oldest.readable is True
    assert oldest.complete is True

    # The nominal active version is still 2; the recommended recovery version is explicitly the old one.
    assert report.active_version == 2
    assert report.recommended_version == 1
    assert report.data.readable
    assert report.metadata.readable  # metadata block survived the truncation

    asif = ASIF(io.BytesIO(raw))
    with asif.open() as stream:
        assert stream.directory_version == 1
        assert stream.is_active_directory is False
        assert stream.read(CHUNK_SIZE) == bytes([DATA_A_PATTERN]) * CHUNK_SIZE

    # Asking for the truncated version is an explicit, typed failure rather than a silent merge.
    with pytest.raises(ASIFRecoveryError):
        asif.open(directory=2)
    # Passing the Directory object directly is rejected the same way.
    with pytest.raises(ASIFRecoveryError):
        asif.open(directory=asif.nominal_active_directory)


def test_missing_table_block_is_reported_with_affected_range() -> None:
    image = build_image()
    raw = retarget_newest_table_to_missing_chunk(image)
    report = ASIF.inspect_recovery(io.BytesIO(raw))

    newest, oldest = report.directories
    assert newest.version == 2
    assert newest.readable is True  # directory header and entries are intact
    assert newest.missing_tables == 1
    assert MISSING_CHUNK in newest.unreadable_chunks
    assert newest.affected_virtual_range == (0, image.geometry.size_per_table)
    assert newest.complete is False
    assert oldest.complete is True
    assert report.recommended_version == 1

    asif = ASIF(io.BytesIO(raw))
    with asif.open(directory=1) as stream:
        assert stream.directory_version == 1
        assert stream.read(CHUNK_SIZE) == bytes([DATA_A_PATTERN]) * CHUNK_SIZE
    # Reading through the damaged newest directory surfaces the underlying short read.
    with asif.open(directory=2) as stream, pytest.raises(EOFError):
        stream.read(CHUNK_SIZE)


def test_recovered_bytes_come_from_the_selected_directory_copy() -> None:
    image = build_image()
    raw = truncate_newest_directory(image)
    asif = ASIF(io.BytesIO(raw))

    with asif.open(directory=1) as old_stream:
        old_bytes = old_stream.read(CHUNK_SIZE * 3)

    # The same virtual offsets resolve to the older copy's data A chunks, never data B of version 2.
    assert old_bytes == bytes([DATA_A_PATTERN]) * CHUNK_SIZE * 3
    assert bytes([DATA_B_PATTERN]) not in old_bytes

    # On a healthy image, explicitly selecting each version returns that version's own bytes.
    healthy = ASIF(io.BytesIO(image.data))
    with healthy.open(directory=2) as new_stream:
        assert new_stream.read(CHUNK_SIZE) == bytes([DATA_B_PATTERN]) * CHUNK_SIZE
    with healthy.open(directory=1) as old_stream:
        assert old_stream.read(CHUNK_SIZE) == bytes([DATA_A_PATTERN]) * CHUNK_SIZE


def test_corrupt_metadata_does_not_block_data_recovery() -> None:
    image = build_image()
    raw = corrupt_metadata_only(image)
    report = ASIF.inspect_recovery(io.BytesIO(raw))

    # Directories and data are independently reported as fully readable.
    assert all(status.readable and status.complete for status in report.directories)
    assert report.data.readable
    assert report.recommended_version == 2

    # Only the reserved-area metadata is damaged.
    assert report.metadata.present is True
    assert report.metadata.header_readable is True
    assert report.metadata.plist_readable is False
    assert report.metadata.internal_metadata_readable is False
    assert report.metadata.user_metadata_readable is False
    assert isinstance(report.metadata.error, str)

    asif = ASIF(io.BytesIO(raw))
    assert asif.user_metadata == {}
    assert asif.internal_metadata == {}
    with asif.open() as stream:
        assert stream.directory_version == 2
        assert stream.read(CHUNK_SIZE) == bytes([DATA_B_PATTERN]) * CHUNK_SIZE


def test_recovery_is_read_only_and_leaves_image_unchanged() -> None:
    image = build_image()
    for mutator in (lambda image: image.data, truncate_newest_directory, corrupt_metadata_only):
        raw = mutator(image)
        digest_before = hashlib.sha256(raw).digest()
        fh = io.BytesIO(raw)

        asif = ASIF(fh)
        report = asif.recovery_report
        for status in report.directories:
            if status.readable:
                with asif.open(directory=status.version) as stream:
                    stream.read(CHUNK_SIZE)
        if asif.header.metadata_chunk:
            with asif.open(reserved=True) as stream:
                stream.seek(asif.header.metadata_chunk * CHUNK_SIZE)
                stream.read(CHUNK_SIZE)

        assert hashlib.sha256(fh.getvalue()).digest() == digest_before


def test_unknown_directory_version_is_rejected() -> None:
    image = build_image()
    asif = ASIF(io.BytesIO(image.data))
    with pytest.raises(ASIFRecoveryError):
        asif.open(directory=99)
