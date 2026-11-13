from __future__ import annotations

import io
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from dissect.hypervisor.disk.c_vhd import SECTOR_SIZE, c_vhd
from dissect.hypervisor.exceptions import Error, InvalidVirtualDisk
from dissect.hypervisor.disk.vhd import (
    DISK_TYPE_DIFFERENCING,
    DISK_TYPE_DYNAMIC,
    DISK_TYPE_FIXED,
    DISK_TYPE_NAMES,
    DYNAMIC_HEADER_CHECKSUM_OFFSET,
    DYNAMIC_HEADER_COOKIE,
    DYNAMIC_HEADER_SIZE,
    FOOTER_CHECKSUM_OFFSET,
    _find_footer,
    checksum,
)


@dataclass(frozen=True)
class BlockState:
    """Forensic assessment of one virtual disk block."""

    block: int
    bat_entry: int | None
    state: str
    detail: str

    @property
    def trusted(self) -> bool:
        """True for data fully determined by this file, including documented sparse zero blocks."""
        return self.state in ("zero", "allocated")

    @property
    def requires_parent(self) -> bool:
        return self.state in ("parent", "partial_parent")

    @property
    def trusted_without_parent(self) -> bool:
        return self.state == "allocated"


@dataclass(frozen=True)
class ParentLocatorInfo:
    """Decoded VHD parent locator entry."""

    index: int
    platform_code: int
    platform_data_space: int
    platform_data_length: int
    platform_data_offset: int
    kind: str
    value: str | None
    raw: bytes | None


PLATFORM_CODE_W2RU = 0x57325255  # parent Unicode relative path
PLATFORM_CODE_W2KU = 0x57324B55  # parent Unicode absolute path (Windows)
PLATFORM_CODE_WI2R = 0x57493252  # parent Windows relative path
PLATFORM_CODE_WI2K = 0x5749324B  # parent Windows absolute path
PLATFORM_CODE_MAC = 0x4D414320  # parent Macintosh path
PLATFORM_CODE_MACX = 0x4D414358  # parent Macintosh absolute path

LOCATOR_KINDS = {
    PLATFORM_CODE_W2RU: ("parent_unicode_relative", "utf-16-le"),
    PLATFORM_CODE_W2KU: ("parent_unicode_absolute", "utf-16-le"),
    PLATFORM_CODE_WI2R: ("parent_windows_relative", "utf-16-le"),
    PLATFORM_CODE_WI2K: ("parent_windows_absolute", "utf-16-le"),
    PLATFORM_CODE_MAC: ("parent_macintosh_relative", "mac_roman"),
    PLATFORM_CODE_MACX: ("parent_macintosh_absolute", "mac_roman"),
}


def _decode_locator(fh: BinaryIO, index: int, locator: c_vhd.parent_locator) -> ParentLocatorInfo:
    kind, encoding = LOCATOR_KINDS.get(locator.platform_code, (f"unknown_0x{locator.platform_code:08x}", None))

    raw = None
    value = None
    file_size = fh.seek(0, io.SEEK_END)
    end = locator.platform_data_offset + locator.platform_data_length
    if encoding and locator.platform_data_length and locator.platform_data_offset and end <= file_size:
        fh.seek(locator.platform_data_offset)
        raw = fh.read(locator.platform_data_length)
        if len(raw) == locator.platform_data_length:
            value = raw.decode(encoding, errors="replace").rstrip("\x00")
        else:
            raw = None

    return ParentLocatorInfo(
        index=index,
        platform_code=locator.platform_code,
        platform_data_space=locator.platform_data_space,
        platform_data_length=locator.platform_data_length,
        platform_data_offset=locator.platform_data_offset,
        kind=kind,
        value=value,
        raw=raw,
    )


@dataclass
class VHDDiagnostics:
    """Structural diagnostics for a VHD file, without opening it as a disk stream."""

    path: Path | None
    file_size: int
    footer: c_vhd.footer
    footer_size: int
    footer_offset: int
    footer_checksum_valid: bool
    disk_type: int
    disk_type_name: str
    virtual_size: int
    header: c_vhd.dynamic_header | None = None
    header_offset: int | None = None
    header_checksum_valid: bool | None = None
    parent_unique_id: bytes | None = None
    parent_timestamp: int | None = None
    parent_name: str | None = None
    parent_locators: list[ParentLocatorInfo] | None = None
    bat: list[int] | None = None
    blocks: list[BlockState] | None = None
    warnings: list[str] | None = None

    def __post_init__(self) -> None:
        if self.warnings is None:
            self.warnings = []

    @property
    def reader_verdict(self) -> str:
        if not self.footer_checksum_valid:
            return "refused: invalid footer checksum"
        if self.disk_type == DISK_TYPE_FIXED:
            if self.footer.data_offset != 0xFFFFFFFFFFFFFFFF:
                return "refused: footer disk type conflicts with data_offset"
            return "supported"
        if self.disk_type == DISK_TYPE_DYNAMIC:
            if self.footer.data_offset in (0, 0xFFFFFFFFFFFFFFFF):
                return "refused: footer disk type conflicts with data_offset"
            return "supported"
        if self.disk_type == DISK_TYPE_DIFFERENCING:
            return "refused: parent chain not implemented"
        return "refused: unsupported disk type"

    @property
    def content_trust(self) -> str:
        """Whether content read through the current reader can be trusted as the disk contents."""
        if self.reader_verdict != "supported":
            return "untrusted"
        if self.disk_type == DISK_TYPE_FIXED:
            data_size = self.file_size - self.footer_size
            return "trusted" if self.virtual_size <= data_size else "untrusted"
        if self.blocks is None:
            return "unassessed"
        if any(block.state in ("corrupt", "ambiguous") for block in self.blocks):
            return "untrusted"
        return "trusted"

    @property
    def parent_reference(self) -> str | None:
        """Best parent locator value, preferring relative paths (portable across evidence copies)."""
        if not self.parent_locators:
            return None
        for locator in self.parent_locators:
            if locator.kind == "parent_unicode_relative" and locator.value:
                return locator.value
        for locator in self.parent_locators:
            if locator.value:
                return locator.value
        return None

    @property
    def block_summary(self) -> dict[str, int]:
        """Count block states in the assessed portion of the image."""
        summary: dict[str, int] = {}
        for block in self.blocks or []:
            summary[block.state] = summary.get(block.state, 0) + 1
        return summary


def _parse_parent_name(raw: bytes) -> str:
    """Decode the UTF-16 parent name field, tolerating malformed/padded data."""
    try:
        name = raw.decode("utf-16-le")
    except UnicodeDecodeError:
        return raw.decode("utf-16-le", errors="replace")
    return name.rstrip("\x00")


def diagnose(path: Path, block_limit: int | None = None) -> VHDDiagnostics:
    """Parse a VHD file structurally and report what the current reader can and cannot trust.

    This deliberately does not call :class:`VHD`: the point is to describe the on-disk structure
    (footer disk type, parent locator, BAT and sector bitmaps) independently of the reader's verdict.

    Args:
        path: Path to the VHD file.
        block_limit: Optional maximum number of blocks to assess. Use this on large images to keep the
            diagnostic bounded; the block support matrix will be marked partial.
    """
    with path.open("rb") as fh:
        raw_footer, footer, footer_size = _find_footer(fh)
        file_size = path.stat().st_size
        footer_offset = file_size - footer_size

        footer_valid = footer.checksum == checksum(raw_footer, FOOTER_CHECKSUM_OFFSET)

        diagnostics = VHDDiagnostics(
            path=path,
            file_size=file_size,
            footer=footer,
            footer_size=footer_size,
            footer_offset=footer_offset,
            footer_checksum_valid=footer_valid,
            disk_type=footer.disk_type,
            disk_type_name=DISK_TYPE_NAMES.get(footer.disk_type, f"unknown ({footer.disk_type})"),
            virtual_size=footer.current_size,
        )

        if not footer_valid:
            diagnostics.warnings.append(
                f"Footer checksum {footer.checksum:#x} does not match computed "
                f"{checksum(raw_footer, FOOTER_CHECKSUM_OFFSET):#x}; footer fields may be corrupt."
            )

        if footer.disk_type == DISK_TYPE_FIXED:
            if footer.data_offset != 0xFFFFFFFFFFFFFFFF:
                diagnostics.warnings.append(
                    "Footer says fixed, but data_offset points to a dynamic header; reader refuses the file."
                )
            data_size = file_size - footer_size
            if footer.current_size > data_size:
                diagnostics.warnings.append(
                    f"Fixed image is truncated: virtual size {footer.current_size}, data area {data_size}."
                )
            return diagnostics

        if footer.disk_type not in (DISK_TYPE_DYNAMIC, DISK_TYPE_DIFFERENCING):
            diagnostics.warnings.append(f"Unknown disk type {footer.disk_type}; no BAT/parent semantics defined.")
            return diagnostics

        if footer.data_offset in (0, 0xFFFFFFFFFFFFFFFF):
            diagnostics.warnings.append(
                f"Footer disk type is {diagnostics.disk_type_name}, but data_offset is {footer.data_offset:#x}; "
                "no dynamic header to parse."
            )
            return diagnostics

        fh.seek(footer.data_offset)
        raw_header = fh.read(DYNAMIC_HEADER_SIZE)
        if len(raw_header) != DYNAMIC_HEADER_SIZE or raw_header[:8] != DYNAMIC_HEADER_COOKIE:
            diagnostics.warnings.append("Dynamic header cookie missing or header truncated; no BAT to assess.")
            return diagnostics

        header = c_vhd.dynamic_header(raw_header)
        diagnostics.header = header
        diagnostics.header_offset = footer.data_offset
        diagnostics.header_checksum_valid = header.checksum == checksum(raw_header, DYNAMIC_HEADER_CHECKSUM_OFFSET)
        diagnostics.parent_unique_id = bytes(header.parent_unique_id)
        diagnostics.parent_timestamp = header.parent_timestamp
        diagnostics.parent_name = _parse_parent_name(bytes(header.parent_unicode_name))

        diagnostics.parent_locators = [
            _decode_locator(fh, index, locator)
            for index, locator in enumerate(header.parent_locators)
            if locator.platform_code or locator.platform_data_length or locator.platform_data_offset
        ]
        for locator in diagnostics.parent_locators:
            if locator.platform_data_space and locator.platform_data_length > locator.platform_data_space:
                diagnostics.warnings.append(
                    f"Parent locator {locator.index} claims data length {locator.platform_data_length} but only "
                    f"{locator.platform_data_space} bytes of space."
                )

        if not diagnostics.header_checksum_valid:
            diagnostics.warnings.append("Dynamic header checksum is invalid; BAT offsets should not be trusted.")

        if footer.disk_type == DISK_TYPE_DIFFERENCING:
            if diagnostics.parent_unique_id == b"\x00" * 16 and not diagnostics.parent_name:
                diagnostics.warnings.append(
                    "Footer says differencing, but the dynamic header carries no parent identity."
                )
            if not diagnostics.parent_locators:
                diagnostics.warnings.append(
                    "Parent locator table is empty: the parent cannot be located from this file alone, even if "
                    "parent_unique_id is present."
                )
            for locator in diagnostics.parent_locators:
                if locator.value is None:
                    diagnostics.warnings.append(
                        f"Parent locator {locator.index} ({locator.kind}) could not be decoded "
                        f"(offset {locator.platform_data_offset:#x}, length {locator.platform_data_length})."
                    )

        try:
            bat, blocks, bat_warnings = _diagnose_blocks(
                fh, header, footer, diagnostics.header_checksum_valid, block_limit
            )
        except (Error, ValueError, struct.error) as exc:
            diagnostics.warnings.append(f"Could not assess BAT/sector bitmap: {exc}")
            return diagnostics

        diagnostics.bat = bat
        diagnostics.blocks = blocks
        diagnostics.warnings.extend(bat_warnings)
        return diagnostics


def _diagnose_blocks(
    fh: BinaryIO,
    header: c_vhd.dynamic_header,
    footer: c_vhd.footer,
    header_checksum_valid: bool,
    block_limit: int | None = None,
) -> tuple[list[int], list[BlockState], list[str]]:
    warnings: list[str] = []
    sectors_per_block = header.block_size // SECTOR_SIZE
    if header.block_size == 0 or header.block_size % SECTOR_SIZE:
        raise InvalidVirtualDisk(f"invalid block size {header.block_size}")
    if header.block_size & (header.block_size - 1):
        raise InvalidVirtualDisk(f"block size must be a power of two: {header.block_size}")
    if header.max_table_entries == 0:
        raise InvalidVirtualDisk("max_table_entries is zero")

    bitmap_sectors = (((sectors_per_block + 7) // 8) + SECTOR_SIZE - 1) // SECTOR_SIZE
    file_size = fh.seek(0, io.SEEK_END)

    expected_entries = (footer.current_size + header.block_size - 1) // header.block_size
    if expected_entries > header.max_table_entries:
        warnings.append(
            f"Virtual size needs {expected_entries} BAT entries, but BAT declares only "
            f"{header.max_table_entries}; tail reads would fail."
        )

    bat_end = header.table_offset + header.max_table_entries * 4
    if bat_end > file_size:
        raise InvalidVirtualDisk("BAT extends beyond end of file")

    fh.seek(header.table_offset)
    bat = list(struct.unpack(f">{header.max_table_entries}I", fh.read(header.max_table_entries * 4)))

    differing = footer.disk_type == DISK_TYPE_DIFFERENCING
    blocks: list[BlockState] = []

    entries = min(expected_entries, header.max_table_entries)
    if block_limit is not None:
        entries = min(entries, block_limit)

    for index in range(entries):
        entry = bat[index]
        if entry == 0xFFFFFFFF:
            state = "parent" if differing else "zero"
            detail = (
                "BAT entry 0xFFFFFFFF: block is not stored in this file."
                if differing
                else "BAT entry 0xFFFFFFFF: dynamic disk treats this block as zero-filled."
            )
            blocks.append(BlockState(index, None, state, detail))
            continue

        if entry == 0:
            blocks.append(
                BlockState(
                    index,
                    0,
                    "corrupt",
                    "BAT entry 0 is not a valid data block offset; page zero would be read as disk data.",
                )
            )
            continue

        if entry < 1 or entry >= file_size // SECTOR_SIZE:
            blocks.append(
                BlockState(
                    index,
                    entry,
                    "corrupt",
                    f"BAT entry {entry:#x} points before the dynamic header or outside the file.",
                )
            )
            continue

        block_end = (entry + bitmap_sectors) * SECTOR_SIZE + sectors_per_block * SECTOR_SIZE
        if block_end > file_size:
            blocks.append(
                BlockState(
                    index,
                    entry,
                    "corrupt",
                    f"Block data extends past end of file (needs {block_end}, file is {file_size}).",
                )
            )
            continue

        if not header_checksum_valid:
            blocks.append(BlockState(index, entry, "ambiguous", "Header checksum invalid; BAT offsets are unverified."))
            continue

        fh.seek(entry * SECTOR_SIZE)
        bitmap = fh.read(bitmap_sectors * SECTOR_SIZE)
        if len(bitmap) != bitmap_sectors * SECTOR_SIZE:
            blocks.append(BlockState(index, entry, "corrupt", "Sector bitmap is truncated."))
            continue

        present_count = sum(1 for sector in range(sectors_per_block) if bitmap[sector >> 3] & (0x80 >> (sector & 7)))

        if present_count == sectors_per_block:
            blocks.append(
                BlockState(
                    index,
                    entry,
                    "allocated",
                    "All sector bitmap bits set: every sector in the block is stored in this file.",
                )
            )
        elif differing:
            if present_count == 0:
                blocks.append(
                    BlockState(
                        index,
                        entry,
                        "parent",
                        "All bitmap bits clear: block content must come from the parent disk.",
                    )
                )
            else:
                blocks.append(
                    BlockState(
                        index,
                        entry,
                        "partial_parent",
                        f"{present_count}/{sectors_per_block} sectors stored here; the remaining sectors require "
                        "the parent disk.",
                    )
                )
        else:
            blocks.append(
                BlockState(
                    index,
                    entry,
                    "ambiguous",
                    f"Footer says dynamic, but {sectors_per_block - present_count}/{sectors_per_block} bitmap bits "
                    "are clear; the current reader returns child block bytes for those sectors rather than "
                    "parent data.",
                )
            )

    if differing:
        stored = sum(1 for block in blocks if block.state in ("parent", "partial_parent"))
        if stored:
            warnings.append(
                f"{stored} block(s) require parent data; the current reader cannot resolve them. "
                "Opening the file as a flat disk would silently return wrong bytes."
            )

    if block_limit is not None and entries < min(expected_entries, header.max_table_entries):
        warnings.append(
            f"Block assessment limited to {entries} of {min(expected_entries, header.max_table_entries)} blocks."
        )

    return bat, blocks, warnings


@dataclass
class ParentVerification:
    """Result of checking a candidate parent against a child's recorded parent identity."""

    child: Path | None
    candidate: Path
    reference: str | None
    resolved: Path | None
    checks: dict[str, bool]
    parent_unique_id: bytes | None = None
    parent_timestamp: int | None = None
    parent_size: int | None = None
    notes: list[str] | None = None

    def __post_init__(self) -> None:
        if self.notes is None:
            self.notes = []

    @property
    def ok(self) -> bool:
        return bool(self.checks) and all(self.checks.values())


def resolve_parent_path(diagnostics: VHDDiagnostics) -> Path | None:
    """Resolve the recorded parent locator to an existing path, preferring relative paths."""
    reference = diagnostics.parent_reference
    if reference is None or diagnostics.path is None:
        return None

    # Both Windows and Unix separators can occur in locators.
    normalized = reference.replace("\\", "/")
    base = diagnostics.path.parent
    candidates = [base / normalized]

    if normalized.startswith("/"):
        candidates.append(Path(normalized))

    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate
        except OSError:
            continue

    return None


def verify_parent(diagnostics: VHDDiagnostics, candidate: Path) -> ParentVerification:
    """Verify a candidate parent file against the child's recorded parent identity.

    A matching filename is not enough: VHD differencing children record the parent's unique ID and timestamp.
    Opening the wrong file as parent silently corrupts inherited sectors, so both must match.
    """
    checks: dict[str, bool] = {}
    notes = []

    if not candidate.exists():
        return ParentVerification(
            child=diagnostics.path,
            candidate=candidate,
            reference=diagnostics.parent_reference,
            resolved=None,
            checks={"candidate_exists": False},
            notes=["Candidate parent does not exist."],
        )

    try:
        parent_diag = diagnose(candidate, block_limit=0)
    except (Error, OSError, ValueError) as exc:
        return ParentVerification(
            child=diagnostics.path,
            candidate=candidate,
            reference=diagnostics.parent_reference,
            resolved=candidate,
            checks={"parent_parseable": False},
            notes=[f"Candidate parent is not a parseable VHD: {exc}"],
        )

    checks["parent_parseable"] = True
    parent_id = bytes(parent_diag.footer.unique_id)
    parent_timestamp = parent_diag.footer.timestamp

    if diagnostics.parent_unique_id is None:
        notes.append("Child records no parent_unique_id.")
        checks["parent_unique_id"] = False
    else:
        checks["parent_unique_id"] = parent_id == diagnostics.parent_unique_id
        if not checks["parent_unique_id"]:
            notes.append(
                f"Parent unique ID mismatch: child expects {diagnostics.parent_unique_id.hex()}, "
                f"candidate has {parent_id.hex()}."
            )

    if diagnostics.parent_timestamp is None:
        notes.append("Child records no parent_timestamp.")
        checks["parent_timestamp"] = False
    else:
        checks["parent_timestamp"] = parent_timestamp == diagnostics.parent_timestamp
        if not checks["parent_timestamp"]:
            notes.append(
                f"Parent timestamp mismatch: child expects {diagnostics.parent_timestamp}, "
                f"candidate has {parent_timestamp}."
            )

    checks["parent_size"] = parent_diag.virtual_size == diagnostics.virtual_size
    if not checks["parent_size"]:
        notes.append(
            f"Virtual size mismatch: child is {diagnostics.virtual_size} bytes, candidate is "
            f"{parent_diag.virtual_size}."
        )

    checks["parent_not_differencing"] = parent_diag.disk_type != DISK_TYPE_DIFFERENCING
    if not checks["parent_not_differencing"]:
        notes.append("Candidate is itself a differencing disk; a complete chain is required.")

    return ParentVerification(
        child=diagnostics.path,
        candidate=candidate,
        reference=diagnostics.parent_reference,
        resolved=candidate,
        checks=checks,
        parent_unique_id=parent_id,
        parent_timestamp=parent_timestamp,
        parent_size=parent_diag.virtual_size,
        notes=notes,
    )


def format_diagnostics(diagnostics: VHDDiagnostics, verification: ParentVerification | None = None) -> str:
    """Render diagnostics as a human-readable report with an explicit support matrix."""
    lines = [
        f"VHD: {diagnostics.path}",
        f"File size: {diagnostics.file_size} bytes",
        f"Footer: offset {diagnostics.footer_offset}, size {diagnostics.footer_size}, "
        f"checksum {'OK' if diagnostics.footer_checksum_valid else 'INVALID'}",
        f"Footer disk type: {diagnostics.disk_type} ({diagnostics.disk_type_name})",
        f"Virtual size: {diagnostics.virtual_size} bytes",
        f"Current reader: {diagnostics.reader_verdict}",
        f"Content trust: {diagnostics.content_trust}",
    ]

    if diagnostics.header is not None:
        header = diagnostics.header
        lines += [
            f"Dynamic header: offset {diagnostics.header_offset}, "
            f"checksum {'OK' if diagnostics.header_checksum_valid else 'INVALID'}",
            f"BAT: offset {header.table_offset}, entries {header.max_table_entries}, "
            f"block size {header.block_size}",
            f"Parent unique ID: {diagnostics.parent_unique_id.hex() if diagnostics.parent_unique_id else '-'}",
            f"Parent timestamp: {diagnostics.parent_timestamp}",
            f"Parent name: {diagnostics.parent_name or '-'}",
            f"Parent locator: {diagnostics.parent_reference or '-'}",
        ]
        for locator in diagnostics.parent_locators or []:
            lines.append(
                f"  locator[{locator.index}] {locator.kind} = {locator.value!r} "
                f"(code {locator.platform_code:#x}, data offset {locator.platform_data_offset:#x}, "
                f"length {locator.platform_data_length})"
            )

    lines.append("")
    if diagnostics.blocks is None:
        lines.append("Support matrix: not assessed (no readable BAT).")
    else:
        lines.append("Support matrix (per block):")
        lines.append(f"Block summary: {diagnostics.block_summary}")
        lines.append("  block  bat_entry   state          meaning")
        for block in diagnostics.blocks:
            entry = "-" if block.bat_entry is None else f"{block.bat_entry:#x}"
            lines.append(f"  {block.block:<6} {entry:<11} {block.state:<14} {block.detail}")

    if diagnostics.warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"  - {warning}" for warning in diagnostics.warnings)

    if verification is not None:
        lines.append("")
        lines.append(f"Parent verification: {verification.candidate}")
        for name, ok in verification.checks.items():
            lines.append(f"  {'OK' if ok else 'FAIL'}  {name}")
        for note in verification.notes or []:
            lines.append(f"  note: {note}")
        if not verification.ok:
            lines.append("  verifier verdict: parent identity NOT established; inherited sectors remain unverified.")

    return "\n".join(lines)
