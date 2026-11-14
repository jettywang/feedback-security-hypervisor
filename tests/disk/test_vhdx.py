from __future__ import annotations

import bisect
import gzip
from collections.abc import Callable
from types import SimpleNamespace
from uuid import UUID

import pytest

from dissect.hypervisor.disk.vhdx import VHDX, _iter_partial_runs, c_vhdx
from dissect.hypervisor.exceptions import MissingParentDisk
from tests._util import absolute_path


def test_vhdx_fixed() -> None:
    with gzip.open(absolute_path("_data/disk/vhdx/fixed.vhdx.gz"), "rb") as fh:
        v = VHDX(fh)

        assert v.size == 0xA00000
        assert v.block_size == 0x200000
        assert v.has_parent == 0
        assert v.sector_size == 0x200
        assert v.bat.chunk_ratio == 0x800
        assert v.id == UUID("4a49d245-db0a-4634-9818-9f93db5ba6c1")

        assert v.read(512) == bytes.fromhex(
            "33c08ed0bc007c8ec08ed8be007cbf0006b90002fcf3a450681c06cbfbb90400"
            "bdbe07807e00007c0b0f850e0183c510e2f1cd1888560055c6461105c6461000"
            "b441bbaa55cd135d720f81fb55aa7509f7c101007403fe46106660807e100074"
            "2666680000000066ff760868000068007c680100681000b4428a56008bf4cd13"
            "9f83c4109eeb14b80102bb007c8a56008a76018a4e028a6e03cd136661731cfe"
            "4e11750c807e00800f848a00b280eb845532e48a5600cd135deb9e813efe7d55"
            "aa756eff7600e88d007517fab0d1e664e88300b0dfe660e87c00b0ffe664e875"
            "00fbb800bbcd1a6623c0753b6681fb54435041753281f90201722c666807bb00"
            "006668000200006668080000006653665366556668000000006668007c000066"
            "6168000007cd1a5a32f6ea007c0000cd18a0b707eb08a0b607eb03a0b50732e4"
            "0500078bf0ac3c007409bb0700b40ecd10ebf2f4ebfd2bc9e464eb002402e0f8"
            "2402c3496e76616c696420706172746974696f6e207461626c65004572726f72"
            "206c6f6164696e67206f7065726174696e672073797374656d004d697373696e"
            "67206f7065726174696e672073797374656d000000637b9a8a3dad8c00000002"
            "030007e525008000000000380000000000000000000000000000000000000000"
            "00000000000000000000000000000000000000000000000000000000000055aa"
        )

        v.seek(0x200000)
        assert v.read(512) == b"\xff" * 512

        for block in range(v.bat._pb_count):
            block_entry = v.bat.pb(block)
            assert block_entry.state == c_vhdx.PAYLOAD_BLOCK_FULLY_PRESENT


def test_vhdx_dynamic() -> None:
    with gzip.open(absolute_path("_data/disk/vhdx/dynamic.vhdx.gz"), "rb") as fh:
        v = VHDX(fh)

        assert v.size == 0xA00000
        assert v.block_size == 0x2000000
        assert v.has_parent == 0
        assert v.sector_size == 0x200
        assert v.bat.chunk_ratio == 0x80
        assert v.id == UUID("788015f0-5e93-4bd2-a5de-b0cd8459db11")

        assert v.read(512) == bytes.fromhex(
            "33c08ed0bc007c8ec08ed8be007cbf0006b90002fcf3a450681c06cbfbb90400"
            "bdbe07807e00007c0b0f850e0183c510e2f1cd1888560055c6461105c6461000"
            "b441bbaa55cd135d720f81fb55aa7509f7c101007403fe46106660807e100074"
            "2666680000000066ff760868000068007c680100681000b4428a56008bf4cd13"
            "9f83c4109eeb14b80102bb007c8a56008a76018a4e028a6e03cd136661731cfe"
            "4e11750c807e00800f848a00b280eb845532e48a5600cd135deb9e813efe7d55"
            "aa756eff7600e88d007517fab0d1e664e88300b0dfe660e87c00b0ffe664e875"
            "00fbb800bbcd1a6623c0753b6681fb54435041753281f90201722c666807bb00"
            "006668000200006668080000006653665366556668000000006668007c000066"
            "6168000007cd1a5a32f6ea007c0000cd18a0b707eb08a0b607eb03a0b50732e4"
            "0500078bf0ac3c007409bb0700b40ecd10ebf2f4ebfd2bc9e464eb002402e0f8"
            "2402c3496e76616c696420706172746974696f6e207461626c65004572726f72"
            "206c6f6164696e67206f7065726174696e672073797374656d004d697373696e"
            "67206f7065726174696e672073797374656d000000637b9af93dad8c00000002"
            "030007e525008000000000380000000000000000000000000000000000000000"
            "00000000000000000000000000000000000000000000000000000000000055aa"
        )

        v.seek(0x200000)
        assert v.read(512) == b"\xff" * 512


def test_vhdx_differencing() -> None:
    with (
        pytest.raises(IOError, match="Failed to open parent disk with locator"),
        gzip.open(absolute_path("_data/disk/vhdx/differencing.avhdx.gz"), "rb") as fh,
    ):
        VHDX(fh)


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        ((b"\xff", 0, 8), [(1, 8)]),
        ((b"\xff", 4, 4), [(1, 4)]),
        ((b"\x00", 0, 8), [(0, 8)]),
        ((b"\x00", 4, 4), [(0, 4)]),
        ((b"\xff\x00", 0, 8), [(1, 8)]),
        ((b"\xff\x00", 4, 8), [(1, 4), (0, 4)]),
        ((b"\x00\x00", 0, 12), [(0, 12)]),
        ((b"\x00\xff", 4, 8), [(0, 4), (1, 4)]),
        ((b"\xf0\xf0", 0, 16), [(0, 4), (1, 4), (0, 4), (1, 4)]),
        ((b"\x0f\x0f", 0, 16), [(1, 4), (0, 4), (1, 4), (0, 4)]),
        ((b"\x00", 0, 6), [(0, 6)]),
        ((b"\x00", 1, 6), [(0, 6)]),
        # Starting at a non-zero bit and crossing bitmap byte boundaries
        ((b"\xf0", 3, 4), [(0, 1), (1, 3)]),
        ((b"\xf0\x00", 3, 13), [(0, 1), (1, 4), (0, 8)]),
        ((b"\xf0\x00", 4, 12), [(1, 4), (0, 8)]),
        ((b"\x0f\x0f", 4, 8), [(0, 4), (1, 4)]),
        ((b"\x24\x49", 3, 10), [(0, 2), (1, 1), (0, 2), (1, 1), (0, 2), (1, 1), (0, 1)]),
        # A run that ends within the first bitmap byte while starting past bit 0
        ((b"\x00", 5, 3), [(0, 3)]),
        ((b"\xff", 5, 3), [(1, 3)]),
        ((b"", 0, 0), []),
    ],
)
def test_vhdx_partial_runs(test_input: tuple[bytes, int, int], expected: list[tuple[int, int]]) -> None:
    assert list(_iter_partial_runs(*test_input)) == expected


# ---------------------------------------------------------------------------
# In-memory differencing disk fixtures
#
# A real differencing VHDX needs an on-disk parent file. To exercise the
# partially-present block bitmap logic deterministically (including bitmaps that
# do not start at bit 0), we build VHDX objects directly around a sparse
# in-memory file and a minimal fake block allocation table. Parent and child
# sectors get deliberately different, recognizable content.
# ---------------------------------------------------------------------------

SYNTH_SECTOR_SIZE = 512
SYNTH_BLOCK_SIZE = 1 * 1024 * 1024
SYNTH_SECTORS_PER_BLOCK = SYNTH_BLOCK_SIZE // SYNTH_SECTOR_SIZE  # 2048
SYNTH_CHUNK_RATIO = 4096
SYNTH_DISK_BLOCKS = SYNTH_CHUNK_RATIO + 1
SYNTH_MB = 1024 * 1024


class _SparseFile:
    """In-memory file handle; only written extents exist, holes read as zeroes."""

    def __init__(self) -> None:
        self._extents: dict[int, bytes] = {}
        self._sorted: list[int] = []
        self.pos = 0

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = offset
        return self.pos

    def write(self, data: bytes, offset: int | None = None) -> int:
        if offset is None:
            offset = self.pos
        self._extents[offset] = bytes(data)
        self._sorted = sorted(self._extents)
        self.pos = offset + len(data)
        return len(data)

    def read(self, n: int) -> bytes:
        out = bytearray(n)
        pos = self.pos
        end = pos + n
        first = max(bisect.bisect_right(self._sorted, pos) - 1, 0)
        for idx in range(first, len(self._sorted)):
            extent_offset = self._sorted[idx]
            if extent_offset >= end:
                break
            extent = self._extents[extent_offset]
            lo = max(pos, extent_offset)
            hi = min(end, extent_offset + len(extent))
            if hi > lo:
                out[lo - pos : hi - pos] = extent[lo - extent_offset : hi - extent_offset]
        self.pos = end
        return bytes(out)


def _bat_entry(state: int, offset_mb: int) -> SimpleNamespace:
    return SimpleNamespace(state=state, file_offset_mb=offset_mb)


def _parent_sector(sector: int) -> bytes:
    return (0x10 + (sector & 0x7F)).to_bytes(1, "little") * SYNTH_SECTOR_SIZE


def _child_sector(sector: int) -> bytes:
    return (0x80 + (sector & 0x7F)).to_bytes(1, "little") * SYNTH_SECTOR_SIZE


class _FakeBAT:
    def __init__(self, payload: dict[int, SimpleNamespace], bitmaps: dict[int, SimpleNamespace]) -> None:
        self._payload = payload
        self._bitmaps = bitmaps

    def pb(self, block: int) -> SimpleNamespace:
        return self._payload.get(block, _bat_entry(c_vhdx.PAYLOAD_BLOCK_NOT_PRESENT, 0))

    def sb(self, block: int) -> SimpleNamespace:
        return self._bitmaps[block // SYNTH_CHUNK_RATIO]


def _build_synthetic_vhdx(
    layout: dict[int, list[bool] | None], sector_writer: Callable[[int], bytes]
) -> VHDX:
    vhdx = VHDX.__new__(VHDX)
    vhdx.sector_size = SYNTH_SECTOR_SIZE
    vhdx.block_size = SYNTH_BLOCK_SIZE
    vhdx._sectors_per_block = SYNTH_SECTORS_PER_BLOCK
    vhdx._chunk_ratio = SYNTH_CHUNK_RATIO
    vhdx.size = SYNTH_DISK_BLOCKS * SYNTH_BLOCK_SIZE
    vhdx.parent = None

    fh = _SparseFile()
    vhdx.fh = fh
    payload: dict[int, SimpleNamespace] = {}
    bitmaps: dict[int, SimpleNamespace] = {}
    alloc_mb = 1

    for block, pattern in layout.items():
        if pattern is None:
            payload.setdefault(block, _bat_entry(c_vhdx.PAYLOAD_BLOCK_NOT_PRESENT, 0))
            continue
        if all(pattern):
            offset_mb = alloc_mb
            alloc_mb += 1
            data = b"".join(
                sector_writer(block * SYNTH_SECTORS_PER_BLOCK + i)
                for i in range(SYNTH_SECTORS_PER_BLOCK)
            )
            fh.write(data, offset_mb * SYNTH_MB)
            payload[block] = _bat_entry(c_vhdx.PAYLOAD_BLOCK_FULLY_PRESENT, offset_mb)
            continue
        if not any(pattern):
            payload[block] = _bat_entry(c_vhdx.PAYLOAD_BLOCK_NOT_PRESENT, 0)
            continue

        offset_mb = alloc_mb
        alloc_mb += 1
        base = offset_mb * SYNTH_MB
        for idx, present in enumerate(pattern):
            if present:
                fh.write(
                    sector_writer(block * SYNTH_SECTORS_PER_BLOCK + idx),
                    base + idx * SYNTH_SECTOR_SIZE,
                )
        payload[block] = _bat_entry(c_vhdx.PAYLOAD_BLOCK_PARTIALLY_PRESENT, offset_mb)

        chunk = block // SYNTH_CHUNK_RATIO
        if chunk not in bitmaps:
            bitmap = bytearray(SYNTH_CHUNK_RATIO * SYNTH_SECTORS_PER_BLOCK // 8)
            for other_block, other_pattern in layout.items():
                if other_block // SYNTH_CHUNK_RATIO != chunk or not other_pattern:
                    continue
                for idx, present in enumerate(other_pattern):
                    if present:
                        bit = (other_block % SYNTH_CHUNK_RATIO) * SYNTH_SECTORS_PER_BLOCK + idx
                        bitmap[bit // 8] |= 1 << (bit % 8)
            bitmap_mb = alloc_mb
            alloc_mb += 1
            fh.write(bytes(bitmap), bitmap_mb * SYNTH_MB)
            bitmaps[chunk] = _bat_entry(c_vhdx.SB_BLOCK_PRESENT, bitmap_mb)

    vhdx.bat = _FakeBAT(payload, bitmaps)
    return vhdx


@pytest.fixture(scope="module")
def synthetic_disks() -> tuple[VHDX, VHDX, dict[int, list[bool] | None]]:
    pattern_a = [idx % 3 == 2 for idx in range(SYNTH_SECTORS_PER_BLOCK)]
    pattern_b = [idx % 5 in (0, 3) for idx in range(SYNTH_SECTORS_PER_BLOCK)]
    pattern_x = [(idx % 7) in (1, 4) for idx in range(SYNTH_SECTORS_PER_BLOCK)]

    child_layout = {
        0: pattern_a,  # partial, first bitmap bit unset (sector must come from parent)
        1: [True] * SYNTH_SECTORS_PER_BLOCK,  # fully present
        2: pattern_b,  # partial, mixed parent/child sectors
        3: None,  # not present -> entirely parent backed
        SYNTH_CHUNK_RATIO - 1: None,  # last block of chunk 0, parent backed
        SYNTH_CHUNK_RATIO: pattern_x,  # partial block in chunk 1
    }
    parent_layout = {block: [True] * SYNTH_SECTORS_PER_BLOCK for block in range(SYNTH_DISK_BLOCKS)}

    parent = _build_synthetic_vhdx(parent_layout, _parent_sector)
    child = _build_synthetic_vhdx(child_layout, _child_sector)
    child.parent = parent
    return child, parent, child_layout


def _expected_synthetic_sector(layout: dict[int, list[bool] | None], sector: int) -> bytes:
    block, sector_in_block = divmod(sector, SYNTH_SECTORS_PER_BLOCK)
    pattern = layout.get(block)
    if pattern is not None and pattern[sector_in_block]:
        return _child_sector(sector)
    return _parent_sector(sector)


def _read_single_sectors(vhdx: VHDX, sector: int, count: int) -> bytes:
    return b"".join(vhdx.read_sectors(sector + idx, 1) for idx in range(count))


@pytest.mark.parametrize(
    ("sector", "count"),
    [
        (3, 10),  # starts in the middle of a bitmap byte, mixed parent/child runs
        (6, 300),  # crosses a bitmap byte boundary
        (5, 2000),  # crosses a payload data block boundary
        (SYNTH_SECTORS_PER_BLOCK - 3, 10),  # partial -> fully present block
        (2 * SYNTH_SECTORS_PER_BLOCK + 1, SYNTH_SECTORS_PER_BLOCK + 5),  # partial -> not present
        (3 * SYNTH_SECTORS_PER_BLOCK - 4, 12),  # partial -> parent-backed block
        (SYNTH_CHUNK_RATIO * SYNTH_SECTORS_PER_BLOCK + 2, 50),  # partial block in chunk 1
        (SYNTH_CHUNK_RATIO * SYNTH_SECTORS_PER_BLOCK - 20, 60),  # crosses a chunk boundary
        (0, 4 * SYNTH_SECTORS_PER_BLOCK + 8),  # spans every populated chunk 0 block
    ],
)
def test_vhdx_partial_block_boundaries(
    synthetic_disks: tuple[VHDX, VHDX, dict[int, list[bool] | None]], sector: int, count: int
) -> None:
    child, _parent, layout = synthetic_disks

    bulk = child.read_sectors(sector, count)
    singles = _read_single_sectors(child, sector, count)
    expected = b"".join(_expected_synthetic_sector(layout, sector + idx) for idx in range(count))

    assert bulk == expected
    assert singles == expected


def test_vhdx_partial_block_fuzz(synthetic_disks: tuple[VHDX, VHDX, dict[int, list[bool] | None]]) -> None:
    child, _parent, layout = synthetic_disks
    rng = pytest.importorskip("random").Random(1234)

    for _ in range(300):
        sector = rng.randrange(0, 4 * SYNTH_SECTORS_PER_BLOCK + 100)
        count = rng.randrange(1, 3000)
        bulk = child.read_sectors(sector, count)
        singles = _read_single_sectors(child, sector, count)
        expected = b"".join(_expected_synthetic_sector(layout, sector + idx) for idx in range(count))
        assert bulk == singles == expected

    for _ in range(100):
        sector = rng.randrange(
            SYNTH_CHUNK_RATIO * SYNTH_SECTORS_PER_BLOCK - 500,
            SYNTH_CHUNK_RATIO * SYNTH_SECTORS_PER_BLOCK + 1500,
        )
        count = rng.randrange(1, 600)
        bulk = child.read_sectors(sector, count)
        singles = _read_single_sectors(child, sector, count)
        expected = b"".join(_expected_synthetic_sector(layout, sector + idx) for idx in range(count))
        assert bulk == singles == expected


def test_vhdx_partial_block_requires_parent(
    synthetic_disks: tuple[VHDX, VHDX, dict[int, list[bool] | None]]
) -> None:
    _child, _parent, layout = synthetic_disks
    orphan = _build_synthetic_vhdx(layout, _child_sector)
    assert orphan.parent is None

    # Sector 0 is parent backed (bit 0 of the partial block bitmap is unset).
    with pytest.raises(MissingParentDisk):
        orphan.read_sectors(0, 8)

    # Child-backed sectors in the same partial block remain readable without a parent.
    assert orphan.read_sectors(2, 1) == _child_sector(2)
    # A fully present child block is self-contained.
    assert orphan.read_sectors(SYNTH_SECTORS_PER_BLOCK, 2) == _child_sector(
        SYNTH_SECTORS_PER_BLOCK
    ) + _child_sector(SYNTH_SECTORS_PER_BLOCK + 1)
    # A genuinely unmapped block reads as zeroes, never adjacent child payload.
    assert orphan.read_sectors(3 * SYNTH_SECTORS_PER_BLOCK, 1) == b"\x00" * SYNTH_SECTOR_SIZE
