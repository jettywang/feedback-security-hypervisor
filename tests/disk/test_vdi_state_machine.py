from __future__ import annotations

import io
import struct
from dataclasses import dataclass

from dissect.hypervisor.disk.c_vdi import (
    VDI_IMAGE_BLOCK_FREE,
    VDI_IMAGE_BLOCK_ZERO,
    VDI_IMAGE_SIGNATURE,
    c_vdi,
)
from dissect.hypervisor.disk.vdi import VDI

# This regression deliberately uses an independent logical byte model. The
# constructor below writes the block table and block contents directly; expected
# bytes come from a plain byte array, never from the production VDI mapping.


@dataclass(frozen=True)
class BlockSpec:
    """One logical block in a synthetic VDI."""

    kind: str
    content: bytes = b""
    slot: int | None = None


@dataclass(frozen=True)
class SyntheticVDI:
    block_size: int
    specs: tuple[BlockSpec, ...]
    tail_length: int
    raw: bytes
    parent_raw: bytes
    off_blocks: int
    off_data: int

    @property
    def size(self) -> int:
        return (len(self.specs) - 1) * self.block_size + self.tail_length

    @property
    def table(self) -> tuple[int, ...]:
        return tuple(
            VDI_IMAGE_BLOCK_FREE
            if spec.kind == "free"
            else VDI_IMAGE_BLOCK_ZERO
            if spec.kind == "zero"
            else spec.slot
            for spec in self.specs
        )

    def logical(self, use_parent: bool = True) -> bytes:
        logical = []
        for spec in self.specs:
            if spec.kind == "free":
                logical.append(spec.content if use_parent else b"\x00" * self.block_size)
            elif spec.kind == "zero":
                logical.append(b"\x00" * self.block_size)
            else:
                logical.append(spec.content.ljust(self.block_size, b"\xee"))
        return b"".join(logical)[: self.size]


def build_vdi(block_size: int, specs: tuple[BlockSpec, ...], tail_length: int) -> SyntheticVDI:
    """Write a VDI header, block table and physical block contents directly."""
    slots = [spec.slot for spec in specs if spec.kind == "alloc"]
    if len(slots) != len(set(slots)) or None in slots:
        raise ValueError("allocated blocks need unique physical slots")

    off_blocks = 512
    off_data = 1024
    n_slots = max(slots) + 1 if slots else 0
    size = (len(specs) - 1) * block_size + tail_length

    preheader = c_vdi.VDIPREHEADER(
        szFileInfo=b"state machine regression",
        u32Signature=VDI_IMAGE_SIGNATURE,
        u32Version=0x10001,
    )
    header = c_vdi.VDIHEADER1PLUS(
        cbHeader=c_vdi.VDIHEADER1PLUS.size,
        u32Type=c_vdi.VDI_IMAGE_TYPE.NORMAL,
        fFlags=c_vdi.VDI_IMAGE_FLAGS(0),
        szComment=b"state machine regression",
        offBlocks=off_blocks,
        offData=off_data,
        cbDisk=size,
        cbBlock=block_size,
        cbBlockExtra=0,
        cBlocks=len(specs),
        cBlocksAllocated=len(slots),
    )

    raw = bytearray(off_data + n_slots * block_size)
    raw[: preheader.size] = preheader.dumps()
    raw[preheader.size : preheader.size + header.size] = header.dumps()

    for index, spec in enumerate(specs):
        value = (
            VDI_IMAGE_BLOCK_FREE
            if spec.kind == "free"
            else VDI_IMAGE_BLOCK_ZERO
            if spec.kind == "zero"
            else spec.slot
        )
        struct.pack_into("<i", raw, off_blocks + 4 * index, value)
        if spec.kind == "alloc":
            start = off_data + spec.slot * block_size
            raw[start : start + block_size] = spec.content.ljust(block_size, b"\xee")

    parent_raw = bytearray(size)
    for index, spec in enumerate(specs):
        if spec.kind == "free":
            start = index * block_size
            parent_raw[start : start + block_size] = spec.content.ljust(block_size, b"\x00")

    return SyntheticVDI(
        block_size=block_size,
        specs=specs,
        tail_length=tail_length,
        raw=bytes(raw),
        parent_raw=bytes(parent_raw),
        off_blocks=off_blocks,
        off_data=off_data,
    )


class RecordingIO(io.BytesIO):
    """BytesIO that records the byte ranges read from it."""

    def __init__(self, data: bytes):
        super().__init__(data)
        self.events: list[tuple[int, int]] = []

    def read(self, size: int = -1) -> bytes:
        start = self.tell()
        data = super().read(size)
        self.events.append((start, len(data)))
        return data


class ByteCursor:
    """Independent, deliberately simple logical-disk read model."""

    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    @property
    def size(self) -> int:
        return len(self.data)

    def seek(self, pos: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            new_pos = pos
        elif whence == io.SEEK_CUR:
            new_pos = max(0, self.pos + pos)
        elif whence == io.SEEK_END:
            new_pos = max(0, self.size + pos)
        else:
            raise ValueError(f"invalid whence {whence}")
        self.pos = new_pos
        return self.pos

    def read(self, length: int = -1) -> bytes:
        if length == -1:
            length = max(0, self.size - self.pos)
        result = self.data[self.pos : self.pos + length]
        self.pos += len(result)
        return result


@dataclass(frozen=True)
class Operation:
    view: str
    action: str
    value: int = 0
    whence: int = io.SEEK_SET

    def __str__(self) -> str:
        if self.action == "seek":
            return f"{self.view}.seek({self.value}, whence={self.whence})"
        return f"{self.view}.read({self.value})"


class _ControlReader:
    """Control reader that follows a deliberately wrong table interpretation."""

    mode = "table"

    def __init__(self, sample: SyntheticVDI):
        self.sample = sample
        self.block_size = sample.block_size
        self.size = sample.size
        self.pos = 0
        self.fh = RecordingIO(sample.raw)

    def seek(self, pos: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            new_pos = pos
        elif whence == io.SEEK_CUR:
            new_pos = max(0, self.pos + pos)
        elif whence == io.SEEK_END:
            new_pos = max(0, self.size + pos)
        else:
            raise ValueError(f"invalid whence {whence}")
        self.pos = new_pos
        return self.pos

    def _read_block(self, block_idx: int, offset: int, length: int) -> bytes:
        entry = self.sample.table[block_idx]
        if entry == VDI_IMAGE_BLOCK_FREE:
            return b"\x00" * length
        if entry == VDI_IMAGE_BLOCK_ZERO:
            return b"\x00" * length

        slot = block_idx if self.mode == "contiguous" else entry
        self.fh.seek(self.sample.off_data + slot * self.block_size + offset)
        return self.fh.read(length)

    def read(self, length: int = -1) -> bytes:
        if length == -1:
            length = max(0, self.size - self.pos)
        else:
            length = min(length, max(0, self.size - self.pos))

        result = []
        while length > 0:
            block_idx, offset = divmod(self.pos, self.block_size)
            read_len = min(length, self.block_size - offset)
            result.append(self._read_block(block_idx, offset, read_len))
            self.pos += read_len
            length -= read_len
        return b"".join(result)

    def clear_trace(self) -> None:
        self.fh.events.clear()

    def describe_trace(self) -> tuple[str, ...]:
        described = []
        for start, length in self.fh.events:
            if length == 0:
                continue
            slot, offset = divmod(start - self.sample.off_data, self.block_size)
            described.append(f"physical slot {slot} +{offset} ({length} bytes)")
        return tuple(described)


class FreeZeroMergedReader(_ControlReader):
    """Negative control: incorrectly folds FREE blocks into ZERO handling."""

    mode = "merged"


class PhysicalContiguityReader(_ControlReader):
    """Negative control: incorrectly assumes physical slots follow logical order."""

    mode = "contiguous"


@dataclass
class OpenedView:
    stream: object
    cursor: ByteCursor
    clear_trace: object
    describe_trace: object


@dataclass
class OperationFailure:
    label: str
    operations: tuple[Operation, ...]
    failing: Operation
    use_parent: bool
    logical_range: tuple[int, int]
    requested_length: int
    expected: bytes | int
    actual: bytes | int
    expected_sources: tuple[str, ...]
    actual_sources: tuple[str, ...]

    def __str__(self) -> str:
        requested = "EOF" if self.requested_length == -1 else str(self.requested_length)
        expected = self.expected if isinstance(self.expected, int) else _short_bytes(self.expected)
        actual = self.actual if isinstance(self.actual, int) else _short_bytes(self.actual)
        lines = [
            f"{self.label}: VDI state-machine mismatch",
            "shortest operation sequence:",
            *(f"  {index + 1}. {operation}" for index, operation in enumerate(self.operations)),
            f"failing operation: {self.failing}",
            f"logical range: [{self.logical_range[0]}, {self.logical_range[1]}) (requested {requested})",
            f"expected: {expected}",
            f"actual:   {actual}",
            f"expected source blocks: {self.expected_sources}",
            f"actual source blocks:   {self.actual_sources}",
        ]
        return "\n".join(lines)

    __repr__ = __str__


def _short_bytes(data: bytes, limit: int = 48) -> str:
    if len(data) <= limit:
        return repr(data)
    return f"{data[:limit]!r}... ({len(data)} bytes)"


def _describe_expected_sources(
    sample: SyntheticVDI, start: int, end: int, use_parent: bool
) -> tuple[str, ...]:
    if start >= end:
        return ()

    described = []
    first_block = start // sample.block_size
    last_block = (end - 1) // sample.block_size
    for block_idx in range(first_block, last_block + 1):
        block_start = block_idx * sample.block_size
        slice_start = max(start, block_start)
        slice_end = min(end, block_start + sample.block_size)
        offset = slice_start - block_start
        length = slice_end - slice_start
        entry = sample.table[block_idx]
        if entry == VDI_IMAGE_BLOCK_FREE:
            if use_parent:
                described.append(
                    f"logical block {block_idx}: FREE -> parent offset "
                    f"{block_start + offset}..{block_start + offset + length}"
                )
            else:
                described.append(f"logical block {block_idx}: FREE without parent -> zeroes")
        elif entry == VDI_IMAGE_BLOCK_ZERO:
            described.append(f"logical block {block_idx}: ZERO -> zeroes")
        else:
            file_start = sample.off_data + entry * sample.block_size + offset
            described.append(
                f"logical block {block_idx}: physical slot {entry} "
                f"(file offset {file_start}..{file_start + length})"
            )
    return tuple(described)


def _open_views(sample: SyntheticVDI, names: set[str], reader: str, use_parent: bool) -> dict[str, OpenedView]:
    if reader != "vdi":
        controls = {
            name: FreeZeroMergedReader(sample)
            if reader == "merged"
            else PhysicalContiguityReader(sample)
            if reader == "contiguous"
            else None
            for name in names
        }
        if any(control is None for control in controls.values()):
            raise ValueError(f"unknown reader {reader}")
        return {
            name: OpenedView(
                control,
                ByteCursor(sample.logical(use_parent)),
                control.clear_trace,
                control.describe_trace,
            )
            for name, control in controls.items()
        }

    # One disk instance, two independently opened streams.
    child = RecordingIO(sample.raw)
    parent = RecordingIO(sample.parent_raw) if use_parent else None
    vdi = VDI(child, parent=parent)

    def clear_trace() -> None:
        child.events.clear()
        if parent is not None:
            parent.events.clear()

    def describe_trace() -> tuple[str, ...]:
        described = []
        for start, length in child.events:
            if length == 0:
                continue
            if start >= sample.off_data:
                slot, offset = divmod(start - sample.off_data, sample.block_size)
                described.append(f"physical slot {slot} +{offset} ({length} bytes)")
            else:
                described.append(f"VDI file offset {start} ({length} bytes)")
        if parent is not None:
            for start, length in parent.events:
                if length:
                    block_idx, offset = divmod(start, sample.block_size)
                    described.append(
                        f"parent offset {start} (logical block {block_idx} +{offset}, {length} bytes)"
                    )
        return tuple(described)

    clear_trace()
    return {
        name: OpenedView(vdi.open(), ByteCursor(sample.logical(use_parent)), clear_trace, describe_trace)
        for name in names
    }


def run_state_machine(
    sample: SyntheticVDI,
    script: tuple[Operation, ...],
    reader: str = "vdi",
    use_parent: bool = True,
) -> OperationFailure | None:
    """Run a script against one reader and an independent byte model."""
    views = _open_views(sample, {op.view for op in script}, reader, use_parent)

    for index, operation in enumerate(script):
        view = views[operation.view]
        view.clear_trace()
        cursor = view.cursor

        if operation.action == "seek":
            expected_position = cursor.seek(operation.value, operation.whence)
            actual_position = view.stream.seek(operation.value, operation.whence)
            if actual_position != expected_position:
                return OperationFailure(
                    label=reader,
                    operations=tuple(script[: index + 1]),
                    failing=operation,
                    use_parent=use_parent,
                    logical_range=(expected_position, expected_position),
                    requested_length=0,
                    expected=expected_position,
                    actual=actual_position,
                    expected_sources=(),
                    actual_sources=view.describe_trace(),
                )
            continue

        if operation.action != "read":
            raise ValueError(f"unknown operation {operation.action}")

        start = cursor.pos
        expected = cursor.read(operation.value)
        actual = view.stream.read(operation.value)
        if actual != expected:
            end = start + len(expected)
            return OperationFailure(
                label=reader,
                operations=tuple(script[: index + 1]),
                failing=operation,
                use_parent=use_parent,
                logical_range=(start, end),
                requested_length=operation.value,
                expected=expected,
                actual=actual,
                expected_sources=_describe_expected_sources(sample, start, end, use_parent),
                actual_sources=view.describe_trace(),
            )

    return None


def find_failure(
    sample: SyntheticVDI,
    script: tuple[Operation, ...],
    reader: str = "vdi",
    use_parent: bool = True,
) -> OperationFailure | None:
    """Find a failure, then greedily shrink its operation sequence."""
    failure = run_state_machine(sample, script, reader, use_parent)
    if failure is None:
        return None

    best = failure
    changed = True
    while changed:
        changed = False
        for index in range(len(best.operations)):
            candidate = best.operations[:index] + best.operations[index + 1 :]
            if not candidate:
                continue
            smaller = run_state_machine(sample, candidate, reader, use_parent)
            if smaller is not None:
                best = smaller
                changed = True
                break
    return best


BLOCK_SIZE = 16
TAIL_LENGTH = 5
SAMPLE = build_vdi(
    BLOCK_SIZE,
    (
        BlockSpec("alloc", b"abcdefghijklmnop", slot=2),
        BlockSpec("free", b"PPPPPPPPPPPPPPPP"),
        BlockSpec("alloc", b"2222222222222222", slot=0),
        BlockSpec("zero"),
        BlockSpec("alloc", b"tail!", slot=1),
    ),
    TAIL_LENGTH,
)
EXPECTED_LOGICAL = bytes(
    [
        *b"abcdefghijklmnop",
        *b"PPPPPPPPPPPPPPPP",
        *b"2222222222222222",
        *(0 for _ in range(BLOCK_SIZE)),
        *b"tail!",
    ]
)
EXPECTED_ZERO_FILLED_PARENT = bytes(
    [
        *b"abcdefghijklmnop",
        *(0 for _ in range(BLOCK_SIZE)),
        *b"2222222222222222",
        *(0 for _ in range(BLOCK_SIZE)),
        *b"tail!",
    ]
)


SINGLE_VIEW_SCRIPT = (
    Operation("disk", "seek", 5),
    Operation("disk", "read", 3),
    Operation("disk", "read", 20),
    Operation("disk", "seek", BLOCK_SIZE),
    Operation("disk", "read", 33),
    Operation("disk", "seek", 3 * BLOCK_SIZE),
    Operation("disk", "read", 7),
    Operation("disk", "read", 22),
    Operation("disk", "seek", SAMPLE.size),
    Operation("disk", "read", 8),
    Operation("disk", "read", -1),
    Operation("disk", "seek", 0),
    Operation("disk", "read", SAMPLE.size),
    Operation("disk", "read", -1),
    Operation("disk", "seek", 4 * BLOCK_SIZE),
    Operation("disk", "read", BLOCK_SIZE),
    Operation("disk", "seek", SAMPLE.size - 2),
    Operation("disk", "read", 10),
    Operation("disk", "seek", 2, io.SEEK_END),
    Operation("disk", "read", 10),
    Operation("disk", "seek", 20),
    Operation("disk", "read", 0),
    Operation("disk", "read", 1),
)


TWO_VIEW_SCRIPT = (
    Operation("left", "seek", 5),
    Operation("right", "seek", 3 * BLOCK_SIZE),
    Operation("left", "read", 20),
    Operation("right", "read", 7),
    Operation("left", "read", 3),
    Operation("right", "read", 25),
    Operation("left", "seek", BLOCK_SIZE),
    Operation("right", "seek", 0),
    Operation("left", "read", 33),
    Operation("right", "read", SAMPLE.size),
    Operation("left", "seek", 0),
    Operation("right", "seek", SAMPLE.size - 2),
    Operation("left", "read", 20),
    Operation("right", "read", 2),
    Operation("left", "seek", 4 * BLOCK_SIZE),
    Operation("right", "seek", 5),
    Operation("left", "read", BLOCK_SIZE),
    Operation("right", "read", 11),
)


def test_vdi_state_machine_layout_is_written_directly() -> None:
    """The fixture is a real VDI whose block table and bytes are stated explicitly."""
    assert SAMPLE.table == (2, VDI_IMAGE_BLOCK_FREE, 0, VDI_IMAGE_BLOCK_ZERO, 1)
    assert SAMPLE.logical() == EXPECTED_LOGICAL
    assert SAMPLE.logical(use_parent=False) == EXPECTED_ZERO_FILLED_PARENT

    with io.BytesIO(SAMPLE.raw) as fh:
        vdi = VDI(fh, parent=io.BytesIO(SAMPLE.parent_raw))
        assert vdi.block_size == BLOCK_SIZE
        assert vdi.number_of_blocks == 5
        assert vdi.size == len(EXPECTED_LOGICAL)
        assert vdi.data_offset == SAMPLE.off_data


def test_vdi_state_machine_free_zero_and_scattered_blocks() -> None:
    """Seeks, mixed read lengths, EOF and repositioning all follow the byte model."""
    failure = find_failure(SAMPLE, SINGLE_VIEW_SCRIPT)
    assert failure is None, failure


def test_vdi_state_machine_missing_parent() -> None:
    """A FREE block without a parent reads as zeroes, including inside one block."""
    failure = find_failure(SAMPLE, SINGLE_VIEW_SCRIPT, use_parent=False)
    assert failure is None, failure


def test_vdi_state_machine_two_views_do_not_share_position() -> None:
    """Two streams from one disk keep separate cursors and alignment buffers."""
    failure = find_failure(SAMPLE, TWO_VIEW_SCRIPT)
    assert failure is None, failure


def test_vdi_state_machine_detects_free_zero_merge() -> None:
    """A reader that folds FREE into ZERO is caught, with repro details retained."""
    failure = find_failure(SAMPLE, SINGLE_VIEW_SCRIPT, reader="merged")
    assert failure is not None
    assert failure.failing.action == "read"
    assert failure.operations[-1] == failure.failing
    assert len(failure.operations) <= len(SINGLE_VIEW_SCRIPT)
    assert failure.expected_sources
    assert failure.actual_sources
    assert isinstance(failure.expected, bytes) and b"PPPP" in failure.expected


def test_vdi_state_machine_detects_physical_contiguity_assumption() -> None:
    """A reader that assumes table order equals physical order is caught."""
    failure = find_failure(SAMPLE, SINGLE_VIEW_SCRIPT, reader="contiguous")
    assert failure is not None
    assert "logical block 0: physical slot 2" in "\n".join(failure.expected_sources)
    assert failure.actual_sources
