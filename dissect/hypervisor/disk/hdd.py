from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO
from uuid import UUID

from defusedxml import ElementTree
from dissect.util.stream import AlignedStream

from dissect.hypervisor.disk.c_hdd import SECTOR_SIZE, c_hdd
from dissect.hypervisor.exceptions import InvalidHeaderError

if TYPE_CHECKING:
    from collections.abc import Iterator
    from xml.etree.ElementTree import Element

DEFAULT_TOP_GUID = UUID("{5fbaabe3-6958-40ff-92a7-860e329aab41}")
NULL_GUID = UUID("00000000-0000-0000-0000-000000000000")


class HDD:
    """Parallels HDD virtual disk implementation.

    Args:
        path: The path to the .hdd directory or .hdd file in a .hdd directory.
    """

    def __init__(self, path: Path):
        if path.is_file():
            path = path.parent
        self.path = path

        descriptor_path = path.joinpath("DiskDescriptor.xml")
        if not descriptor_path.exists():
            raise ValueError(f"Invalid Parallels HDD path: {path} (missing DiskDescriptor.xml)")

        self.descriptor = Descriptor(descriptor_path)

    def _candidate_image_paths(self, path: Path) -> Iterator[Path]:
        """Yield candidate paths for an image file, in order of preference.

        Args:
            path: The image path as referenced by the descriptor.
        """
        root = self.path
        filename = path.name

        if path.is_absolute():
            yield path

            # If the absolute path does not exist, we're probably dealing with a HDD
            # that's been copied or moved (e.g., uploaded or copied as evidence)
            # Try a couple of common patterns to see if we can locate the file
            #
            # Example variables:
            # root = /some/path/example.pvm/example.hdd/
            # path = /other/path/absolute.pvm/absolute.hdd/absolute.ext

            # File is in same HDD directory
            # candidate_path = /some/path/example.pvm/example.hdd/absolute.ext
            yield root / filename
            # File is in a separate HDD directory in parent (VM) directory
            # candidate_path = /some/path/example.pvm/absolute.hdd/absolute.ext
            yield root.parent / path.parent.name / filename
            # File is in .pvm directory in parent of parent directory (linked clones)
            # candidate_path = /some/path/absolute.pvm/absolute.hdd/absolute.ext
            yield root.parent.parent / path.parent.parent.name / path.parent.name / filename
        else:
            # If the path is relative, it's always relative to the HDD root
            yield root / path

    def _resolve_image_path(self, path: Path) -> Path | None:
        """Resolve an image path to an existing file, without opening it.

        Args:
            path: The image path as referenced by the descriptor.
        """
        for candidate in self._candidate_image_paths(path):
            if candidate.exists():
                return candidate
        return None

    def _open_image(self, path: Path) -> BinaryIO:
        """Helper method for opening image files relative to this HDD.

        Args:
            path: The path to the image file to open.
        """
        resolved_path = self._resolve_image_path(path)
        if resolved_path is None:
            # No candidate exists, open the original path to raise a useful error
            resolved_path = path if path.is_absolute() else self.path / path
        return resolved_path.open("rb")

    def open(self, guid: str | UUID | None = None) -> BinaryIO:
        """Open a stream for this HDD, optionally for a specific snapshot.

        If no snapshot GUID is provided, the "top" snapshot will be used.

        Args:
            guid: The snapshot GUID to open.
        """
        if guid and not isinstance(guid, UUID):
            guid = UUID(guid)

        if guid is None:
            guid = self.descriptor.snapshots.top_guid or DEFAULT_TOP_GUID

        chain = self.descriptor.get_snapshot_chain(guid)

        streams = []
        for storage in self.descriptor.storage_data.storages:
            stream = None
            for guid in chain[::-1]:
                image = storage.find_image(guid)
                fh = self._open_image(Path(image.file))

                if image.type == "Compressed":
                    fh = HDS(fh, parent=stream)
                elif image.type != "Plain":
                    raise ValueError(f"Unsupported image type: {image.type}")

                stream = fh

            streams.append((storage, stream))

        return StorageStream(streams)

    def verify(self, guid: str | UUID | None = None) -> IntegrityReport:
        """Verify the integrity of this HDD before opening it, without reading image data.

        Checks the ancestor chain of the target snapshot GUID, the image coverage of every
        storage and the logical sector ranges of the storages. Only the descriptor and file
        existence are inspected; no image content is read and nothing is modified.

        Args:
            guid: The snapshot GUID to verify. Defaults to the "top" snapshot.

        Returns:
            An :class:`IntegrityReport` with the resolved chain and a list of found issues.
        """
        if guid and not isinstance(guid, UUID):
            guid = UUID(guid)

        if guid is None:
            guid = self.descriptor.snapshots.top_guid or DEFAULT_TOP_GUID

        issues: list[IntegrityIssue] = []

        # Check for duplicate snapshot GUIDs in the descriptor
        shots_by_guid: dict[UUID, Shot] = {}
        for shot in self.descriptor.snapshots.shots:
            if shot.guid in shots_by_guid:
                issues.append(
                    IntegrityIssue(
                        "duplicate-guid",
                        f"Duplicate snapshot GUID in descriptor: {shot.guid}",
                        guid=shot.guid,
                    )
                )
            else:
                shots_by_guid[shot.guid] = shot

        # Walk the ancestor chain of the target GUID, detecting cycles and missing parents
        chain: list[UUID] = []
        if guid not in shots_by_guid:
            issues.append(
                IntegrityIssue(
                    "unknown-snapshot",
                    f"Snapshot GUID not found in descriptor: {guid}",
                    guid=guid,
                )
            )
        else:
            current = guid
            while True:
                if current in chain:
                    issues.append(
                        IntegrityIssue(
                            "parent-cycle",
                            f"Cycle in snapshot parent chain at GUID: {current}",
                            guid=current,
                        )
                    )
                    break
                chain.append(current)

                parent = shots_by_guid[current].parent
                if parent == NULL_GUID:
                    break
                if parent not in shots_by_guid:
                    issues.append(
                        IntegrityIssue(
                            "missing-parent",
                            f"Snapshot {current} references missing parent snapshot: {parent}",
                            guid=parent,
                        )
                    )
                    break
                current = parent

        # Check that every storage has an image for every snapshot in the chain
        # and that every referenced image file is present on disk
        for storage in self.descriptor.storage_data.storages:
            image_guids = {image.guid for image in storage.images}
            for chain_guid in chain:
                if chain_guid not in image_guids:
                    issues.append(
                        IntegrityIssue(
                            "missing-image",
                            f"Storage {storage.start}-{storage.end} has no image for snapshot {chain_guid}",
                            guid=chain_guid,
                            storage=storage,
                        )
                    )

            for image in storage.images:
                if self._resolve_image_path(Path(image.file)) is None:
                    issues.append(
                        IntegrityIssue(
                            "missing-file",
                            f"Image file not found on disk: {image.file}",
                            guid=image.guid,
                            storage=storage,
                            file=image.file,
                        )
                    )

        # Check the logical sector ranges of the storages for gaps and overlaps
        covered = 0
        sorted_storages = sorted(
            self.descriptor.storage_data.storages, key=lambda storage: (storage.start, storage.end)
        )
        for storage in sorted_storages:
            if storage.end <= storage.start:
                issues.append(
                    IntegrityIssue(
                        "invalid-range",
                        f"Storage has an invalid sector range: {storage.start}-{storage.end}",
                        storage=storage,
                    )
                )
                continue
            if storage.start < covered:
                issues.append(
                    IntegrityIssue(
                        "storage-overlap",
                        f"Storage {storage.start}-{storage.end} overlaps with already covered range up to {covered}",
                        storage=storage,
                    )
                )
            elif storage.start > covered:
                issues.append(
                    IntegrityIssue(
                        "storage-gap",
                        f"Gap in storage coverage between sectors {covered} and {storage.start}",
                        storage=storage,
                    )
                )
            covered = max(covered, storage.end)

        return IntegrityReport(guid, chain, issues)


class Descriptor:
    """Helper class for working with ``DiskDescriptor.xml``.

    References:
        - https://github.com/qemu/qemu/blob/master/docs/interop/prl-xml.txt

    Args:
        path: The path to ``DiskDescriptor.xml``.
    """

    def __init__(self, path: Path):
        self.path = path

        self.xml: Element = ElementTree.fromstring(path.read_text())
        self.storage_data = StorageData.from_xml(self.xml.find("StorageData"))
        self.snapshots = Snapshots.from_xml(self.xml.find("Snapshots"))

    def get_snapshot_chain(self, guid: UUID) -> list[UUID]:
        """Return the snapshot chain for a given snapshot GUID.

        Args:
            guid: The snapshot GUID to return a chain for.
        """
        shot = self.snapshots.find_shot(guid)

        chain = [shot.guid]
        while shot.parent != NULL_GUID:
            shot = self.snapshots.find_shot(shot.parent)
            chain.append(shot.guid)

        return chain


@dataclass
class XMLEntry:
    @classmethod
    def from_xml(cls, element: Element) -> XMLEntry:
        if element.tag != cls.__name__:
            raise ValueError(f"Invalid {cls.__name__} XML element")
        return cls._from_xml(element)

    @classmethod
    def _from_xml(cls, element: Element) -> XMLEntry:
        raise NotImplementedError


@dataclass
class StorageData(XMLEntry):
    storages: list[Storage]

    @classmethod
    def _from_xml(cls, element: Element) -> StorageData:
        return cls(list(map(Storage.from_xml, element.iterfind("Storage"))))


@dataclass
class Storage(XMLEntry):
    start: int
    end: int
    images: list[Image]

    @classmethod
    def _from_xml(cls, element: Element) -> Storage:
        start = int(element.find("Start").text)
        end = int(element.find("End").text)
        images = list(map(Image.from_xml, element.iterfind("Image")))

        return cls(start, end, images)

    def find_image(self, guid: UUID) -> Image:
        """Find a specific image GUID.

        Args:
            guid: The image GUID to find.

        Raises:
            KeyError: If the GUID could not be found.
        """
        for image in self.images:
            if image.guid == guid:
                return image

        raise KeyError(f"Image GUID not found: {guid}")


@dataclass
class Image(XMLEntry):
    guid: UUID
    type: str
    file: str

    @classmethod
    def _from_xml(cls, element: Element) -> Image:
        return cls(
            UUID(element.find("GUID").text),
            element.find("Type").text,
            element.find("File").text,
        )


@dataclass
class Snapshots(XMLEntry):
    top_guid: UUID | None
    shots: list[Shot]

    @classmethod
    def _from_xml(cls, element: Element) -> Snapshots:
        top_guid = element.find("TopGUID")
        if top_guid is not None:
            top_guid = UUID(top_guid.text)
        shots = list(map(Shot.from_xml, element.iterfind("Shot")))

        return cls(top_guid, shots)

    def find_shot(self, guid: UUID) -> Shot:
        """Find a specific snapshot GUID.

        Args:
            guid: The snapshot GUID to find.

        Raises:
            KeyError: If the GUID could not be found.
        """
        for shot in self.shots:
            if shot.guid == guid:
                return shot

        raise KeyError(f"Shot GUID not found: {guid}")


@dataclass
class Shot(XMLEntry):
    guid: UUID
    parent: UUID

    @classmethod
    def _from_xml(cls, element: Element) -> Shot:
        return cls(
            UUID(element.find("GUID").text),
            UUID(element.find("ParentGUID").text),
        )


@dataclass
class IntegrityIssue:
    """A single integrity issue found by :meth:`HDD.verify`.

    Args:
        category: The category of the issue, e.g. ``parent-cycle``, ``duplicate-guid``,
            ``missing-parent``, ``unknown-snapshot``, ``missing-image``, ``missing-file``,
            ``invalid-range``, ``storage-gap`` or ``storage-overlap``.
        message: A human readable description of the issue.
        guid: The snapshot or image GUID related to the issue, if any.
        storage: The :class:`Storage` related to the issue, if any.
        file: The image file related to the issue, if any.
    """

    category: str
    message: str
    guid: UUID | None = None
    storage: Storage | None = None
    file: str | None = None


@dataclass
class IntegrityReport:
    """Result of :meth:`HDD.verify`.

    Args:
        guid: The snapshot GUID that was verified.
        chain: The resolved ancestor chain of the target GUID, target first.
        issues: The list of integrity issues that were found.
    """

    guid: UUID
    chain: list[UUID]
    issues: list[IntegrityIssue]

    @property
    def ok(self) -> bool:
        """Return whether no integrity issues were found."""
        return not self.issues

    def by_category(self, category: str) -> list[IntegrityIssue]:
        """Return all issues of a specific category.

        Args:
            category: The issue category to filter on.
        """
        return [issue for issue in self.issues if issue.category == category]


class StorageStream(AlignedStream):
    """Stream implementation for HDD streams.

    HDD files can exist of one or multiple streams, starting at consecutive offsets.
    This class stitches all streams together into a single stream.

    Args:
        streams: A list of :class:`Storage` and file-like object tuples.
    """

    def __init__(self, streams: list[tuple[Storage, BinaryIO]]):
        self.streams = sorted(streams, key=lambda entry: entry[0].start)
        self._lookup = []

        size = 0
        for storage, _ in self.streams:
            self._lookup.append(storage.start)
            size = storage.end

        super().__init__(size * SECTOR_SIZE)

    def _read(self, offset: int, length: int) -> bytes:
        sector = offset // SECTOR_SIZE
        count = (length + SECTOR_SIZE - 1) // SECTOR_SIZE

        result = []
        stream_idx = bisect_right(self._lookup, sector) - 1

        while count > 0 and stream_idx < len(self.streams):
            storage, stream = self.streams[stream_idx]
            sectors_remaining = storage.end - sector
            read_sectors = min(sectors_remaining, count)

            stream.seek((sector - storage.start) * SECTOR_SIZE)
            result.append(stream.read(read_sectors * SECTOR_SIZE))

            sector += read_sectors
            count -= read_sectors
            stream_idx += 1

        return b"".join(result)


class HDS(AlignedStream):
    """Parallels HDS implementation.

    HDS is the format for Parallels sparse disk files.

    Args:
        fh: The file-like object to the HDS file.
        parent: Optional file-like object for the parent HDS file.
    """

    def __init__(self, fh: BinaryIO, parent: BinaryIO | None = None):
        self.fh = fh
        self.parent = parent

        self.header = c_hdd.pvd_header(fh)
        if self.header.m_Sig not in (c_hdd.SIGNATURE_STRUCTURED_DISK_V1, c_hdd.SIGNATURE_STRUCTURED_DISK_V2):
            raise InvalidHeaderError(f"Invalid HDS header signature: {self.header.m_Sig}")

        if self.header.m_Sig == c_hdd.SIGNATURE_STRUCTURED_DISK_V1:
            size = self.header.m_SizeInSectors_v1
            self._bat_step = self.header.m_Sectors
            self._bat_multiplier = 1
        else:
            size = self.header.m_SizeInSectors_v2
            self._bat_step = 1
            self._bat_multiplier = self.header.m_Sectors

        self.cluster_size = self.header.m_Sectors * SECTOR_SIZE

        self.data_offset = self.header.m_FirstBlockOffset
        self.in_use = self.header.m_DiskInUse == c_hdd.SIGNATURE_DISK_IN_USE

        super().__init__(size * SECTOR_SIZE)

    @cached_property
    def bat(self) -> list[int]:
        """Return the block allocation table (BAT)."""
        self.fh.seek(len(c_hdd.pvd_header))
        return c_hdd.uint32[self.header.m_Size](self.fh)

    def _read(self, offset: int, length: int) -> bytes:
        result = []

        for read_offset, read_size in self._iter_runs(offset, length):
            # Sentinel value for sparse runs
            if read_offset is None:
                if self.parent:
                    self.parent.seek(offset)
                    result.append(self.parent.read(read_size))
                else:
                    result.append(b"\x00" * read_size)
            else:
                self.fh.seek(read_offset)
                result.append(self.fh.read(read_size))

            offset += read_size
            length -= read_size

        return b"".join(result)

    def _iter_runs(self, offset: int, length: int) -> Iterator[tuple[int, int]]:
        """Iterate optimized read runs for a given offset and read length.

        Args:
            offset: The offset in bytes to generate runs for.
            length: The length in bytes to generate runs for.
        """
        bat = self.bat

        run_offset = None
        run_size = 0

        while offset < self.size and length > 0:
            cluster_idx, offset_in_cluster = divmod(offset, self.cluster_size)
            read_size = min(self.cluster_size - offset_in_cluster, length)

            bat_entry = bat[cluster_idx]
            # BAT entry of 0 means either a sparse or a parent read
            # Use 0 to denote a sparse run for now to make calculations easier
            read_offset = 0 if bat_entry == 0 else bat_entry * self._bat_multiplier * SECTOR_SIZE + offset_in_cluster

            if run_offset is None:
                # First iteration
                run_offset = read_offset
                run_size = read_size
            elif (read_offset == run_offset + run_size) or (run_offset, read_offset) == (0, 0):
                # Consecutive (sparse) clusters
                run_size += read_size
            else:
                # New run
                # Replace 0 with None as sparse sentinel value
                yield (run_offset or None, run_size)

                # Reset run
                run_offset = read_offset
                run_size = read_size

            offset += read_size
            length -= read_size

        if run_offset is not None:
            # Flush remaining run
            # Replace 0 with None as sparse sentinel value
            yield (run_offset or None, run_size)
