from __future__ import annotations

import plistlib
from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import Any, BinaryIO
from uuid import UUID

from dissect.util.stream import AlignedStream

from dissect.hypervisor.disk.c_asif import c_asif
from dissect.hypervisor.exceptions import ASIFRecoveryError, InvalidSignature


@dataclass(frozen=True)
class DirectoryStatus:
    """Readability assessment of a single on-disk directory copy.

    Attributes:
        version: Directory version number, or ``None`` if even the version could not be read.
        offset: Physical offset of the directory in the image.
        readable: Whether the directory header and its full table entry array can be read.
        is_active: Whether this is the highest-version (active) directory.
        error: Error description when the directory is not readable, otherwise ``None``.
        num_tables: Number of table entries the directory refers to.
        readable_tables: Number of those tables whose entry array can be read.
        missing_tables: Number of tables whose chunk is missing or unreadable.
        unreadable_chunks: Physical chunk numbers that could not be read (table and data chunks).
        affected_virtual_range: ``(start, end)`` virtual byte span affected by missing blocks, or ``None``.
        complete: Whether every table and data chunk referenced by this directory is readable.
    """

    version: int | None
    offset: int
    readable: bool
    is_active: bool
    error: str | None
    num_tables: int
    readable_tables: int
    missing_tables: int
    unreadable_chunks: tuple[int, ...]
    affected_virtual_range: tuple[int, int] | None
    complete: bool


@dataclass(frozen=True)
class DataStatus:
    """Readability assessment of the data blocks reachable through the readable directories.

    Attributes:
        readable: Whether at least one directory can serve its complete data set.
        complete_directories: Number of directories whose referenced data is fully readable.
        unreadable_chunks: Sorted physical chunk numbers missing across all readable directories.
        affected_virtual_range: ``(start, end)`` virtual byte span of the missing data, or ``None``.
        detail: Human readable summary of the assessment.
    """

    readable: bool
    complete_directories: int
    unreadable_chunks: tuple[int, ...]
    affected_virtual_range: tuple[int, int] | None
    detail: str


@dataclass(frozen=True)
class MetadataStatus:
    """Readability assessment of the reserved-area metadata block and its plist.

    Attributes:
        present: Whether the header references a metadata chunk.
        header_readable: Whether the metadata chunk and its ``meta`` header can be read.
        plist_readable: Whether the embedded plist can be parsed.
        internal_metadata_readable: Whether the ``internal metadata`` dictionary was parsed.
        user_metadata_readable: Whether the ``user metadata`` dictionary was parsed.
        error: Error description of the first metadata failure, or ``None``.
    """

    present: bool
    header_readable: bool
    plist_readable: bool
    internal_metadata_readable: bool
    user_metadata_readable: bool
    error: str | None

    @property
    def readable(self) -> bool:
        """Whether the full metadata block, including the plist, was read and parsed."""
        return self.present and self.header_readable and self.plist_readable


@dataclass(frozen=True)
class RecoveryReport:
    """Result of a read-only recovery inspection of an ASIF image.

    The report assesses directory, data and metadata readability independently. It never merges
    directory versions; each :class:`DirectoryStatus` describes exactly one on-disk copy.

    Attributes:
        directories: Readability of each directory copy, ordered as the loader orders them.
        data: Aggregate readability of the data blocks.
        metadata: Readability of the reserved-area metadata plist.
        active_version: Version of the active (highest-version) directory, even if unreadable.
        recommended_version: Highest version whose directory and data are fully readable, or ``None``.
    """

    directories: tuple[DirectoryStatus, ...]
    data: DataStatus
    metadata: MetadataStatus
    active_version: int | None
    recommended_version: int | None


class ASIF:
    """Apple Sparse Image Format (ASIF) disk image.

    ASIF disk images are a virtual disk format introduced in macOS Tahoe. They can be used in Apple's Virtualization
    framework, as well as through Disk Utility.

    An ASIF file is pretty straight forward. There's a small header which, among some other details, contains two
    directory offsets. Each directory contains a list of tables, which in turn contain a list of data entries. Each data
    entry points to a chunk of data in the ASIF file. The chunk size is defined in the header and is typically 1 MiB.
    The chunk size is always a multiple of the block size, which is also defined in the header (typically 512 bytes).
    Each directory has a version number, and the directory with the highest version number is the active directory. This
    allows for atomic updates of the directory/table data.

    The maximum virtual disk size seems to be just under 4 PiB, with a small portion at the end reserved for metadata.
    The actual size of the virtual disk is defined in the header, as well as the maximum size the disk can grow to.

    The offset to the metadata block is typically ``(4 PiB - 1 chunk)``, meaning it's within the reserved area.
    The metadata block contains a small header and a plist. The plist should contain an ``internal metadata`` and
    ``user metadata`` dictionary. Besides a "stable uuid", it's unclear what the metadata is used for or how to set it.

    Construction is fault tolerant for forensic recovery: a truncated directory or a corrupted metadata plist no longer
    prevent the image from opening. Use :meth:`inspect_recovery` (or :attr:`recovery_report`) to assess which directory
    versions, data blocks and metadata are actually readable, and :meth:`open` with an explicit ``directory`` version to
    read from one chosen, unmodified directory copy. A metadata failure never blocks data access.

    Args:
        fh: File-like object containing the ASIF image.

    Resources:
        - Reversing ``diskimagescontroller``
        - https://developer.apple.com/documentation/virtualization/vzdiskimagestoragedeviceattachment/
    """

    def __init__(self, fh: BinaryIO):
        self.fh = fh

        self.header = c_asif.asif_header(fh)
        if self.header.header_signature != c_asif.ASIF_HEADER_SIGNATURE:
            raise InvalidSignature(
                f"Not a valid ASIF image (expected {c_asif.ASIF_HEADER_SIGNATURE:#x}, "
                f"got {self.header.header_signature:#x})"
            )

        self.guid = UUID(bytes=self.header.guid)
        self.block_size = self.header.block_size
        self.chunk_size = self.header.chunk_size
        self.size = self.header.sector_count * self.block_size
        self.max_size = self.header.max_sector_count * self.block_size

        # The following math is taken from the assembly with some creative variable naming
        # It's possible that some of this can be simplified or the names improved
        self._blocks_per_chunk = self.chunk_size // self.block_size

        # This check doesn't really make sense, but keep it in for now
        reserved_size = 4 * self.chunk_size
        self._num_reserved_table_entries = (
            1 if reserved_size < self._blocks_per_chunk else reserved_size // self._blocks_per_chunk
        )

        self._max_table_entries = self.chunk_size >> 3
        self._num_table_entries = self._max_table_entries - (
            self._max_table_entries % (self._num_reserved_table_entries + 1)
        )
        self._num_reserved_directory_entries = (self._num_reserved_table_entries + self._num_table_entries) // (
            self._num_reserved_table_entries + 1
        )
        self._num_usable_entries = self._num_table_entries - self._num_reserved_directory_entries
        # This is the size in bytes of data covered by a single table
        self._size_per_table = self._num_usable_entries * self.chunk_size

        max_size = self.block_size * self.header.max_sector_count
        self._num_directory_entries = (self._size_per_table + max_size - 1) // self._size_per_table

        self._aligned_table_size = (
            (self.block_size + 8 * self._num_table_entries - 1) // self.block_size * self.block_size
        )

        # Keep every directory copy, including unreadable ones. nominal_active_directory is the
        # highest-version copy regardless of readability (the format's notion of "active"), while
        # self.directories orders readable copies (highest version first) ahead of unreadable ones.
        all_directories = [Directory(self, offset) for offset in self.header.directory_offsets]
        self.nominal_active_directory = max(
            all_directories,
            key=lambda d: d.version if d.version is not None else -1,
        )
        self.directories = sorted(
            all_directories,
            key=lambda d: (d.readable, d.version if d.version is not None else -1),
            reverse=True,
        )
        self.readable_directories = [directory for directory in self.directories if directory.readable]
        # The default stream uses the highest version whose table entry array is actually usable, so a
        # truncated newest directory never produces a stream that fails on its first read.
        self.active_directory = next(
            (directory for directory in self.readable_directories if self._directory_usable(directory)),
            None,
        )

        # Metadata loading is best effort. A bad signature or an unparsable plist must not take down
        # access to the directory and data blocks, which are assessed separately.
        self.metadata_header = None
        self.metadata: dict[str, Any] = {}
        self.metadata_error: str | None = None
        self._metadata_header_readable = False
        self._metadata_plist_readable = False
        if self.header.metadata_chunk:
            # The metadata block is read through a directory's table mapping. Try the nominal active
            # directory first and fall back to other usable directories when its table mapping cannot
            # resolve the reserved chunk. This only changes which copy maps the reserved area; it never
            # merges directory versions for regular data access.
            metadata_errors: list[str] = []
            for metadata_directory in self._metadata_directories():
                try:
                    with self.open(directory=metadata_directory, reserved=True) as disk:
                        metadata_offset = self.header.metadata_chunk * self.chunk_size
                        disk.seek(metadata_offset)
                        self.metadata_header = c_asif.asif_meta_header(disk)

                        if self.metadata_header.header_signature != c_asif.ASIF_META_HEADER_SIGNATURE:
                            raise InvalidSignature(
                                f"Invalid a ASIF metadata header (expected {c_asif.ASIF_META_HEADER_SIGNATURE:#x}, "
                                f"got {self.metadata_header.header_signature:#x})"
                            )
                        self._metadata_header_readable = True

                        disk.seek(metadata_offset + self.metadata_header.header_size)
                        self.metadata = plistlib.loads(
                            disk.read(self.metadata_header.data_size).strip(b"\x00")
                        )
                        self._metadata_plist_readable = True
                        metadata_errors.clear()
                        break
                except (
                    OSError,
                    EOFError,
                    ValueError,
                    plistlib.InvalidFileException,
                    InvalidSignature,
                    ASIFRecoveryError,
                ) as exc:
                    # A bad plist or signature is a property of the metadata block itself, not the
                    # directory that maps it, so another directory cannot help: stop retrying.
                    if isinstance(exc, (plistlib.InvalidFileException, InvalidSignature, ValueError)):
                        metadata_errors = [f"{type(exc).__name__}: {exc}"]
                        break
                    metadata_errors.append(
                        f"directory v{metadata_directory.version}: {type(exc).__name__}: {exc}"
                    )
            if metadata_errors:
                self.metadata_error = "; ".join(metadata_errors)

    def _directory_usable(self, directory: Directory) -> bool:
        """Return whether a directory's version and full table entry array can be read."""
        if not directory.readable:
            return False
        try:
            # Accessing entries populates the cached property and surfaces a truncated array.
            _ = directory.entries
        except ASIFRecoveryError:
            return False
        return True

    def _metadata_directories(self) -> list[Directory]:
        """Candidate directories that can map the reserved metadata block, highest version first.

        The nominal active directory comes first; only copies whose table entry array can be read are
        included. Duplicates are removed while preserving order.
        """
        candidates = [self.nominal_active_directory, *self.readable_directories]
        result: list[Directory] = []
        for candidate in candidates:
            if candidate is not None and self._directory_usable(candidate) and candidate not in result:
                result.append(candidate)
        return result

    @property
    def internal_metadata(self) -> dict[str, Any]:
        """Get internal metadata from the ASIF image.

        Returns an empty dictionary when the metadata block is missing, corrupted or unparsable.
        Consult :attr:`recovery_report` for the separate metadata readability assessment.

        Returns:
            A dictionary containing the internal metadata.
        """
        return self.metadata.get("internal metadata", {})

    @property
    def user_metadata(self) -> dict[str, Any]:
        """Get user metadata from the ASIF image.

        Returns an empty dictionary when the metadata block is missing, corrupted or unparsable.
        Consult :attr:`recovery_report` for the separate metadata readability assessment.

        Returns:
            A dictionary containing the user metadata.
        """
        return self.metadata.get("user metadata", {})

    @property
    def directory_versions(self) -> tuple[int | None, ...]:
        """Version numbers of all directory copies, in loader order (highest readable version first)."""
        return tuple(directory.version for directory in self.directories)

    @classmethod
    def inspect_recovery(cls, fh: BinaryIO) -> RecoveryReport:
        """Open an image read-only and return its full recovery assessment.

        This is the explicit read-only recovery entry point. It never writes to ``fh``, never merges
        the two directory copies and never presents a fallback directory as the active one. Directory,
        data and metadata readability are reported independently.

        Args:
            fh: File-like object containing the ASIF image.

        Returns:
            A :class:`RecoveryReport` describing each directory version, the data blocks and the
            reserved-area metadata.
        """
        return cls(fh).recovery_report

    @cached_property
    def recovery_report(self) -> RecoveryReport:
        """Recovery assessment of this image, computed once and cached.

        Returns:
            The :class:`RecoveryReport` for this image.
        """
        directory_statuses = []
        unreadable_chunks: set[int] = set()
        affected_start: int | None = None
        affected_end: int | None = None
        complete_directories = 0

        nominal_active = self.nominal_active_directory
        for directory in self.directories:
            status = self._probe_directory(directory, directory is nominal_active)
            directory_statuses.append(status)
            if status.complete:
                complete_directories += 1
            for chunk in status.unreadable_chunks:
                unreadable_chunks.add(chunk)
            if status.affected_virtual_range is not None:
                start, end = status.affected_virtual_range
                affected_start = start if affected_start is None else min(affected_start, start)
                affected_end = end if affected_end is None else max(affected_end, end)

        recommended = next(
            (status.version for status in directory_statuses if status.complete and status.version is not None),
            None,
        )
        data_readable = complete_directories > 0
        if data_readable:
            detail = (
                f"{complete_directories} directory version(s) have fully readable data"
                if unreadable_chunks
                else "All referenced data chunks are readable"
            )
        else:
            detail = "No directory version can serve its complete data set"

        data_status = DataStatus(
            readable=data_readable,
            complete_directories=complete_directories,
            unreadable_chunks=tuple(sorted(unreadable_chunks)),
            affected_virtual_range=(
                None if affected_start is None or affected_end is None else (affected_start, affected_end)
            ),
            detail=detail,
        )

        metadata_status = MetadataStatus(
            present=bool(self.header.metadata_chunk),
            header_readable=self._metadata_header_readable,
            plist_readable=self._metadata_plist_readable,
            internal_metadata_readable=self._metadata_plist_readable
            and "internal metadata" in self.metadata,
            user_metadata_readable=self._metadata_plist_readable and "user metadata" in self.metadata,
            error=self.metadata_error,
        )

        return RecoveryReport(
            directories=tuple(directory_statuses),
            data=data_status,
            metadata=metadata_status,
            active_version=nominal_active.version if nominal_active is not None else None,
            recommended_version=recommended,
        )

    def _probe_directory(self, directory: Directory, is_active: bool) -> DirectoryStatus:
        """Probe one directory copy and every table and data chunk it references."""
        if not directory.readable:
            return DirectoryStatus(
                version=directory.version,
                offset=directory.offset,
                readable=False,
                is_active=is_active,
                error=directory.error,
                num_tables=0,
                readable_tables=0,
                missing_tables=0,
                unreadable_chunks=(),
                affected_virtual_range=None,
                complete=False,
            )

        num_tables = 0
        readable_tables = 0
        missing_tables = 0
        unreadable_chunks: set[int] = set()
        affected_start: int | None = None
        affected_end: int | None = None

        def mark_range(start: int, end: int) -> None:
            nonlocal affected_start, affected_end
            affected_start = start if affected_start is None else min(affected_start, start)
            affected_end = end if affected_end is None else max(affected_end, end)

        try:
            table_entries = directory.entries
        except ASIFRecoveryError:
            # The version was readable but the table entry array is truncated.
            return DirectoryStatus(
                version=directory.version,
                offset=directory.offset,
                readable=False,
                is_active=is_active,
                error=directory.entries_error,
                num_tables=0,
                readable_tables=0,
                missing_tables=0,
                unreadable_chunks=(),
                affected_virtual_range=None,
                complete=False,
            )

        for table_index, table_chunk in enumerate(table_entries):
            if table_chunk == 0:
                continue
            num_tables += 1
            table = directory.table(table_index)
            try:
                data_entries = table.entries
            except (ASIFRecoveryError, EOFError, OSError):
                missing_tables += 1
                unreadable_chunks.add(table_chunk)
                start = table.virtual_offset
                mark_range(start, start + self._size_per_table)
                continue
            readable_tables += 1

            for data_slot in range(self._num_usable_entries):
                entry = data_entries[self._data_entry_index(data_slot)]
                chunk = entry & 0x7FFFFFFFFFFFFF
                if chunk == 0:
                    continue
                if not self._chunk_readable(chunk):
                    unreadable_chunks.add(chunk)
                    start = table.virtual_offset + data_slot * self.chunk_size
                    mark_range(start, start + self.chunk_size)

        affected_range = (
            None if affected_start is None or affected_end is None else (affected_start, affected_end)
        )
        complete = missing_tables == 0 and not unreadable_chunks
        return DirectoryStatus(
            version=directory.version,
            offset=directory.offset,
            readable=True,
            is_active=is_active,
            error=None,
            num_tables=num_tables,
            readable_tables=readable_tables,
            missing_tables=missing_tables,
            unreadable_chunks=tuple(sorted(unreadable_chunks)),
            affected_virtual_range=affected_range,
            complete=complete,
        )

    def _data_entry_index(self, data_slot: int) -> int:
        """Translate a zero-based data slot to its index in a table entry array.

        Mirrors the arithmetic used by :class:`DataStream` so probing and actual reads agree on which
        table entries address data.
        """
        return (
            data_slot
            + data_slot * self._num_reserved_table_entries
        ) // self._num_reserved_table_entries

    def _chunk_readable(self, chunk: int) -> bool:
        """Return whether a full chunk can be read from the underlying image."""
        raw_offset = chunk * self.chunk_size
        try:
            self.fh.seek(raw_offset)
            data = self.fh.read(self.chunk_size)
        except OSError:
            return False
        return len(data) == self.chunk_size

    def _select_directory(self, directory: int | Directory | None) -> Directory:
        if directory is None:
            if self.active_directory is None:
                raise ASIFRecoveryError("No readable directory version is available in this image")
            return self.active_directory
        if isinstance(directory, Directory):
            selected = directory
        else:
            matches = [candidate for candidate in self.directories if candidate.version == directory]
            if not matches:
                available = ", ".join(
                    str(candidate.version) if candidate.version is not None else "<unreadable>"
                    for candidate in self.directories
                )
                raise ASIFRecoveryError(f"Directory version {directory} not found (available: {available})")
            selected = matches[0]
        if not self._directory_usable(selected):
            reason = selected.entries_error if selected.readable else selected.error
            raise ASIFRecoveryError(
                f"Directory version {selected.version} at offset {selected.offset:#x} is not readable: {reason}"
            )
        return selected

    def open(self, directory: int | Directory | None = None, reserved: bool = False) -> DataStream:
        """Open a read-only stream to the ASIF image data.

        The stream reads from exactly one directory copy; versions are never merged and a fallback
        directory is never silently substituted.

        Args:
            directory: Explicit directory version number or :class:`Directory` to read from. When
                ``None`` the highest readable directory is used and :attr:`DataStream.is_active_directory`
                indicates whether it is also the nominal active (highest-version) directory.
            reserved: Whether to allow reading into the reserved area of the ASIF image.

        Returns:
            A stream-like object that can be used to read the image data.
        """
        selected = self._select_directory(directory)
        return DataStream(self, selected, reserved)


class Directory:
    """ASIF Directory.

    A directory has a version (``uint64``) followed by a list of table entries (``uint64[]``).
    The version number is used to determine the active directory, with the highest version being the active one.
    Each table entry is a chunk number and points to a table in the ASIF image.

    A truncated or otherwise unreadable directory no longer raises during construction. :attr:`readable`
    reflects whether the directory version can be read; a truncated table entry array is reported by
    the :class:`DirectoryStatus` assessment and raises on explicit access via :attr:`entries`.

    Args:
        asif: The ASIF image this directory belongs to.
        offset: Offset of the directory in the ASIF image.
    """

    def __init__(self, asif: ASIF, offset: int):
        self.asif = asif
        self.offset = offset

        self.version: int | None = None
        self.readable = True
        self.error: str | None = None
        self.entries_error: str | None = None

        try:
            self.asif.fh.seek(offset)
            self.version = c_asif.uint64(self.asif.fh)
        except (EOFError, OSError) as exc:
            self.readable = False
            self.error = f"{type(exc).__name__}: {exc}"

        self.table = lru_cache(128)(self.table)

    def __repr__(self) -> str:
        version = self.version if self.version is not None else "unreadable"
        return f"<Directory offset={self.offset:#x} version={version} readable={self.readable}>"

    @cached_property
    def entries(self) -> list[int]:
        """List of table entries in the directory.

        Raises:
            ASIFRecoveryError: If the table entry array cannot be read (for example a truncated directory).
        """
        try:
            # Seek over the version
            self.asif.fh.seek(self.offset + 8)
            return c_asif.uint64[self.asif._num_directory_entries](self.asif.fh)
        except (EOFError, OSError) as exc:
            self.entries_error = f"{type(exc).__name__}: {exc}"
            raise ASIFRecoveryError(
                f"Directory version {self.version} at offset {self.offset:#x} cannot be read: "
                f"{self.entries_error}"
            ) from exc

    def table(self, index: int) -> Table:
        """Get a table from the directory.

        Args:
            index: Index of the table in the directory.
        """
        if index >= self.asif._num_directory_entries:
            raise IndexError("Table index out of range")
        return Table(self, index)


class Table:
    """ASIF Table.

    A table contains a list of data entries (``uint64[]``). Each data entry is a chunk number and points to a chunk of
    data in the ASIF image. Each table covers a fixed amount of data in the virtual disk.

    Data entries have 55 bits usable for the chunk number and 9 bits reserved for flags.

    .. rubric :: Encoding
    .. code-block:: c

        0b00000000 01111111 11111111 11111111 11111111 11111111 11111111 11111111  (chunk number)
        0b00111111 10000000 00000000 00000000 00000000 00000000 00000000 00000000  (reserved)
        0b01000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000  (entry dirty)
        0b10000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000  (content dirty)

    Args:
        directory: The directory this table belongs to.
        index: Index of the table in the directory.
    """

    def __init__(self, directory: Directory, index: int):
        self.asif = directory.asif
        self.directory = directory
        self.index = index

        self.offset = self.directory.entries[index] * self.asif.chunk_size
        self.virtual_offset = index * self.asif._size_per_table

    def __repr__(self) -> str:
        return f"<Table index={self.index} offset={self.offset:#x} virtual_offset={self.virtual_offset:#x}>"

    @cached_property
    def entries(self) -> list[int]:
        """List of data entries in the table."""
        self.asif.fh.seek(self.offset)
        return c_asif.uint64[self.asif._num_table_entries](self.asif.fh)


class DataStream(AlignedStream):
    """Stream to read data from an ASIF image using one explicitly selected directory.

    Args:
        asif: The ASIF image to read from.
        directory: The single directory copy this stream reads from.
        reserved: Whether to allow reading into the reserved area of the ASIF image.
    """

    def __init__(self, asif: ASIF, directory: Directory, reserved: bool = False):
        super().__init__(asif.max_size if reserved else asif.size, align=asif.chunk_size)
        self.asif = asif
        self.reserved = reserved
        self.directory = directory
        self.is_active_directory = directory is asif.nominal_active_directory

    @property
    def directory_version(self) -> int | None:
        """Version of the directory copy this stream is actually reading from."""
        return self.directory.version

    def _read(self, offset: int, length: int) -> bytes:
        result = []
        while length:
            table = self.directory.table(offset // self.asif._size_per_table)
            relative_block_index = (offset // self.asif.block_size) - (table.virtual_offset // self.asif.block_size)
            data_idx = (
                relative_block_index // self.asif._blocks_per_chunk
                + relative_block_index // self.asif._blocks_per_chunk * self.asif._num_reserved_table_entries
            ) // self.asif._num_reserved_table_entries

            # 0x8000000000000000 = content dirty bit
            # 0x4000000000000000 = entry dirty bit
            # 0x3F80000000000000 = reserved bits
            chunk = table.entries[data_idx] & 0x7FFFFFFFFFFFFF
            raw_offset = chunk * self.asif.chunk_size

            read_length = min(length, self.asif.chunk_size)
            if chunk == 0:
                result.append(b"\x00" * read_length)
            else:
                self.asif.fh.seek(raw_offset)
                result.append(self.asif.fh.read(read_length))

            offset += read_length
            length -= read_length

        return b"".join(result)
