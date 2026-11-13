from __future__ import annotations

import io
import struct
from functools import lru_cache
from typing import BinaryIO

from dissect.util.stream import AlignedStream

from dissect.hypervisor.disk.c_vhd import SECTOR_SIZE, c_vhd
from dissect.hypervisor.exceptions import InvalidHeaderError, InvalidSignature, InvalidVirtualDisk

FOOTER_COOKIE = b"conectix"
DYNAMIC_HEADER_COOKIE = b"cxsparse"

DISK_TYPE_FIXED = 2
DISK_TYPE_DYNAMIC = 3
DISK_TYPE_DIFFERENCING = 4

DISK_TYPE_NAMES = {
    DISK_TYPE_FIXED: "fixed",
    DISK_TYPE_DYNAMIC: "dynamic",
    DISK_TYPE_DIFFERENCING: "differencing",
}

FOOTER_SIZE = 512
LEGACY_FOOTER_SIZE = 511
FOOTER_CHECKSUM_OFFSET = 64
DYNAMIC_HEADER_SIZE = 1024
DYNAMIC_HEADER_CHECKSUM_OFFSET = 36


def checksum(data: bytes | bytearray, offset: int) -> int:
    """Calculate the one's complement checksum used by VHD footers and dynamic headers."""
    data = bytearray(data)
    data[offset : offset + 4] = b"\x00" * 4
    return ~sum(data) & 0xFFFFFFFF


def _find_footer(fh: BinaryIO) -> tuple[bytes, c_vhd.footer, int]:
    """Locate the final footer without validating its checksum."""
    file_size = fh.seek(0, io.SEEK_END)

    for size in (FOOTER_SIZE, LEGACY_FOOTER_SIZE):
        if file_size < size:
            continue

        fh.seek(-size, io.SEEK_END)
        raw = fh.read(size)
        if len(raw) == size and raw[:8] == FOOTER_COOKIE:
            return raw, c_vhd.footer(raw), size

    raise InvalidSignature(f"Invalid VHD footer cookie (expected {FOOTER_COOKIE!r})")


def _read_footer_raw(fh: BinaryIO) -> tuple[bytes, c_vhd.footer, int]:
    """Read and validate a footer, returning the raw footer, parsed footer and footer size.

    A footer can be 511 bytes in very old VHD files; normal VHD files use 512 bytes.
    """
    raw, footer, size = _find_footer(fh)
    expected = checksum(raw, FOOTER_CHECKSUM_OFFSET)
    if footer.checksum != expected:
        raise InvalidHeaderError(f"Invalid VHD footer checksum: {footer.checksum:#x} != {expected:#x}")

    return raw, footer, size


def read_footer(fh: BinaryIO) -> c_vhd.footer:
    """Read the VHD footer at the end of a file."""
    return _read_footer_raw(fh)[1]


class VHD(AlignedStream):
    # Note: split VHD files are currently unsupported.

    def __init__(self, fh: BinaryIO):
        self.fh = fh

        raw_footer, footer, footer_size = _read_footer_raw(fh)
        self.footer = footer
        self.disk_type = footer.disk_type

        if footer.disk_type == DISK_TYPE_FIXED:
            if footer.data_offset != 0xFFFFFFFFFFFFFFFF:
                raise InvalidVirtualDisk(
                    "VHD footer declares a fixed disk, but data_offset points to a dynamic header "
                    f"({footer.data_offset:#x})"
                )

            file_size = fh.seek(0, io.SEEK_END)
            data_size = file_size - footer_size
            if footer.current_size > data_size:
                raise InvalidVirtualDisk(
                    f"VHD fixed disk is truncated: virtual size {footer.current_size} > data size {data_size}"
                )

            # Fixed size, data starts at 0
            self.disk = FixedDisk(fh, footer)
        elif footer.disk_type == DISK_TYPE_DYNAMIC:
            if footer.data_offset in (0, 0xFFFFFFFFFFFFFFFF):
                raise InvalidVirtualDisk(
                    "VHD footer declares a dynamic disk, but data_offset does not point to a dynamic header "
                    f"({footer.data_offset:#x})"
                )

            self.disk = DynamicDisk(fh, footer)
        elif footer.disk_type == DISK_TYPE_DIFFERENCING:
            raise InvalidVirtualDisk(
                "Differencing VHD files are not supported by this reader: their BAT and sector bitmap "
                "reference parent data, which this reader cannot resolve. Refusing to open the file instead "
                "of returning zeroes or child data for inherited sectors. Use the vhd-diagnose tool to "
                "inspect the image and its parent locator without opening it as a flat disk."
            )
        else:
            raise InvalidVirtualDisk(f"Unsupported VHD disk type: {footer.disk_type}")

        super().__init__(self.disk.size)

    def _read(self, offset: int, length: int) -> bytes:
        sector = offset // SECTOR_SIZE
        count = (length + SECTOR_SIZE - 1) // SECTOR_SIZE

        return self.disk.read_sectors(sector, count)


class Disk:
    def __init__(self, fh: BinaryIO, footer: c_vhd.footer | None = None):
        self.fh = fh
        self.footer = footer if footer else read_footer(fh)
        self.size = self.footer.current_size

    def read_sectors(self, sector: int, count: int) -> bytes:
        raise NotImplementedError


class FixedDisk(Disk):
    def read_sectors(self, sector: int, count: int) -> bytes:
        self.fh.seek(sector * SECTOR_SIZE)
        data = self.fh.read(count * SECTOR_SIZE)
        if len(data) != count * SECTOR_SIZE:
            raise InvalidVirtualDisk(
                f"Short read from fixed VHD at sector {sector}: expected {count * SECTOR_SIZE} bytes, got {len(data)}"
            )
        return data


class DynamicDisk(Disk):
    """Dynamic VHD disk.

    Every allocated block is preceded by a sector bitmap. For dynamic (non-differencing) disks all bits in the
    bitmap must be set; the bits are still skipped here because the bitmap is part of the block's on-disk layout.
    """

    def __init__(self, fh: BinaryIO, footer: c_vhd.footer | None = None):
        super().__init__(fh, footer)

        if self.footer.disk_type != DISK_TYPE_DYNAMIC:
            raise InvalidVirtualDisk(
                f"DynamicDisk only supports disk type {DISK_TYPE_DYNAMIC}, got {self.footer.disk_type}"
            )

        self.header = read_dynamic_header(fh, self.footer)
        self.bat = BlockAllocationTable(fh, self.header.table_offset, self.header.max_table_entries)

        self._sectors_per_block = self.header.block_size // SECTOR_SIZE
        # Sector bitmaps are padded to SECTOR_SIZE boundaries
        # Save bitmap size in sectors
        self._sector_bitmap_size = (((self._sectors_per_block + 7) // 8) + SECTOR_SIZE - 1) // SECTOR_SIZE

        if self.size > self.header.max_table_entries * self.header.block_size:
            raise InvalidVirtualDisk(
                f"VHD virtual size {self.size} exceeds BAT capacity "
                f"({self.header.max_table_entries} blocks of {self.header.block_size} bytes)"
            )

        file_size = fh.seek(0, io.SEEK_END)
        if self.header.table_offset + self.header.max_table_entries * 4 > file_size:
            raise InvalidVirtualDisk("VHD BAT extends beyond the end of the file")

        self._validated_bitmap_blocks: set[int] = set()

    def _validate_sector_bitmap(self, block: int, sector_offset: int) -> None:
        """Validate that an allocated block is fully present in this file.

        A dynamic (non-differencing) VHD must have every sector bitmap bit set for allocated blocks. Zero bits mean
        the sector belongs to a parent disk, which this reader cannot resolve.
        """
        if block in self._validated_bitmap_blocks:
            return

        self.fh.seek(sector_offset * SECTOR_SIZE)
        bitmap = self.fh.read(self._sector_bitmap_size * SECTOR_SIZE)
        if len(bitmap) != self._sector_bitmap_size * SECTOR_SIZE:
            raise InvalidVirtualDisk(f"Short read of sector bitmap for dynamic VHD block {block}")

        for index in range(self._sectors_per_block):
            if not bitmap[index >> 3] & (0x80 >> (index & 7)):
                raise InvalidVirtualDisk(
                    f"Dynamic VHD block {block} has an unset sector bitmap bit for sector {index} in the block; "
                    "this indicates parent-inherited data (differencing semantics) on a disk whose footer declares "
                    "disk type 3"
                )

        self._validated_bitmap_blocks.add(block)

    def read_sectors(self, sector: int, count: int) -> bytes:
        result = []
        while count > 0:
            block, offset = divmod(sector, self._sectors_per_block)
            block_remaining = self._sectors_per_block - offset

            read_count = min(count, block_remaining)
            if read_count <= 0:
                raise InvalidVirtualDisk(
                    f"Dynamic VHD block {block} has {self._sectors_per_block} sectors, which is too small for "
                    f"sector offset {offset}"
                )

            sector_offset = self.bat[block]
            if sector_offset is None:
                # Sparse
                result.append((b"\x00" * SECTOR_SIZE) * read_count)
                sector += read_count
                count -= read_count
                continue
            if sector_offset == 0:
                raise InvalidVirtualDisk(f"Invalid BAT entry 0 for dynamic VHD block {block}")

            self._validate_sector_bitmap(block, sector_offset)
            self.fh.seek((sector_offset + self._sector_bitmap_size + offset) * SECTOR_SIZE)
            data = self.fh.read(read_count * SECTOR_SIZE)
            if len(data) != read_count * SECTOR_SIZE:
                raise InvalidVirtualDisk(
                    f"Short read from dynamic VHD block {block} at sector {sector}: "
                    f"expected {read_count * SECTOR_SIZE} bytes, got {len(data)}"
                )
            result.append(data)

            sector += read_count
            count -= read_count

        return b"".join(result)


def read_dynamic_header(fh: BinaryIO, footer: c_vhd.footer | None = None) -> c_vhd.dynamic_header:
    """Read and validate the dynamic disk header pointed to by the footer."""
    if footer is None:
        footer = read_footer(fh)

    fh.seek(footer.data_offset)
    raw = fh.read(DYNAMIC_HEADER_SIZE)
    if len(raw) != DYNAMIC_HEADER_SIZE or raw[:8] != DYNAMIC_HEADER_COOKIE:
        raise InvalidSignature(f"Invalid VHD dynamic header cookie (expected {DYNAMIC_HEADER_COOKIE!r})")

    header = c_vhd.dynamic_header(raw)
    if header.checksum != checksum(raw, DYNAMIC_HEADER_CHECKSUM_OFFSET):
        raise InvalidHeaderError(
            "Invalid VHD dynamic header checksum: "
            f"{header.checksum:#x} != {checksum(raw, DYNAMIC_HEADER_CHECKSUM_OFFSET):#x}"
        )

    if header.block_size == 0 or header.block_size % SECTOR_SIZE:
        raise InvalidVirtualDisk(f"Invalid VHD block size: {header.block_size}")
    if header.block_size & (header.block_size - 1):
        raise InvalidVirtualDisk(f"VHD block size must be a power of two: {header.block_size}")
    if header.max_table_entries == 0:
        raise InvalidVirtualDisk("Invalid VHD BAT: max_table_entries is zero")

    return header


class BlockAllocationTable:
    """Implementation of the BAT.

    Entries are uint32 sector offsets to blocks in the file.
    """

    ENTRY = struct.Struct(">I")

    def __init__(self, fh: BinaryIO, offset: int, max_entries: int):
        self.fh = fh
        self.offset = offset
        self.max_entries = max_entries

        self.get = lru_cache(4096)(self.get)

    def get(self, block: int) -> int | None:
        # This could be improved by caching the entire BAT (or chunks if too large)
        if block + 1 > self.max_entries:
            raise ValueError(f"Invalid block {block} (max block is {self.max_entries - 1})")

        self.fh.seek(self.offset + block * 4)
        raw_entry = self.fh.read(4)
        if len(raw_entry) != 4:
            raise InvalidVirtualDisk(f"Short read of BAT entry {block}")
        sector_offset = self.ENTRY.unpack(raw_entry)[0]
        if sector_offset == 0xFFFFFFFF:
            sector_offset = None
        return sector_offset

    def __getitem__(self, block: int) -> int | None:
        return self.get(block)


# Re-exported for backwards-compatible discovery from the original VHD module.
from dissect.hypervisor.disk.vhd_diagnose import (  # noqa: E402,F401
    BlockState,
    ParentLocatorInfo,
    ParentVerification,
    VHDDiagnostics,
    diagnose,
    format_diagnostics,
    resolve_parent_path,
    verify_parent,
)
