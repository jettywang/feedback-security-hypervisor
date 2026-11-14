from __future__ import annotations

import gzip
import hashlib
import struct
from io import BytesIO
from pathlib import Path
from typing import BinaryIO
from unittest.mock import patch

import pytest

from dissect.hypervisor.disk.c_qcow2 import QCOW2_MAGIC, c_qcow2
from dissect.hypervisor.disk.qcow2 import (
    QCow2,
    QCow2Mapping,
    QCow2MapSource,
    QCow2Stream,
)
from dissect.hypervisor.exceptions import Error
from tests._util import absolute_path

def mock_open_gz(self: Path, *args, **kwargs) -> BinaryIO:
    return gzip.open(self if self.suffix.lower() == ".gz" else self.with_suffix(self.suffix + ".gz"))


def _be64(value: int) -> bytes:
    return struct.pack(">Q", value)


def build_extl2_qcow2(backing: BytesIO | None = None) -> tuple[BytesIO, dict[str, int]]:
    """Build a minimal in-memory extended L2 (subcluster) QCOW2 image.

    The 64KiB clusters of the 1MiB disk are laid out as:
      * cluster 0: fully unallocated (plain) subclusters
      * cluster 1: fully allocated normal subclusters
      * cluster 2: mixed allocated / zero-plain / unallocated-plain / allocated subclusters
      * cluster 3: fully explicit zero (plain) subclusters
      * ...
      * cluster 8: allocated normal cluster pointing at an external-style data region
      * cluster 9: fully allocated normal cluster
    """
    cluster_bits = 16
    cluster_size = 1 << cluster_bits
    sub_size = cluster_size // c_qcow2.QCOW_EXTL2_SUBCLUSTERS_PER_CLUSTER
    nb_clusters = 16

    header_off = 0
    l1_off = cluster_size
    l2_off = 2 * cluster_size
    data_a_off = 3 * cluster_size
    data_b_off = 5 * cluster_size
    total = 6 * cluster_size

    backing_name = b"backing.qcow2"
    backing_file_offset = 0
    backing_file_size = 0
    if backing is not None:
        backing_file_offset = total
        backing_file_size = len(backing_name)

    header = struct.pack(
        ">IIQIIQIIQ QIIQ QQQII B7s",
        QCOW2_MAGIC,
        3,
        backing_file_offset,
        backing_file_size,
        cluster_bits,
        nb_clusters * cluster_size,
        0,
        1,
        l1_off,
        0,
        0,
        0,
        0,
        c_qcow2.QCOW2_INCOMPAT_EXTL2,
        0,
        0,
        4,
        112,
        0,
        b"\x00" * 7,
    )

    buf = bytearray(total + (backing_file_size if backing is not None else 0))
    buf[header_off : header_off + len(header)] = header
    if backing is not None:
        buf[backing_file_offset : backing_file_offset + backing_file_size] = backing_name
    buf[l1_off : l1_off + 8] = _be64(l2_off)

    def set_l2_entry(
        idx: int, cluster_offset: int, alloc_mask: int = 0, zero_mask: int = 0, copied: bool = True
    ) -> None:
        off = l2_off + idx * 16
        entry = cluster_offset | (c_qcow2.QCOW_OFLAG_COPIED if copied else 0)
        buf[off : off + 8] = _be64(entry)
        buf[off + 8 : off + 16] = _be64(alloc_mask | (zero_mask << 32))

    full = (1 << 32) - 1
    set_l2_entry(0, 0, copied=False)
    set_l2_entry(1, data_a_off, alloc_mask=full)
    set_l2_entry(
        2,
        data_b_off,
        alloc_mask=sum(1 << i for i in [*range(0, 8), *range(24, 32)]),
        zero_mask=sum(1 << i for i in range(8, 16)),
    )
    set_l2_entry(3, 0, zero_mask=full, copied=False)
    set_l2_entry(8, data_a_off, alloc_mask=full)
    set_l2_entry(9, data_b_off, alloc_mask=full)

    locations = {
        "cluster_size": cluster_size,
        "subcluster_size": sub_size,
        "data_a": data_a_off,
        "data_b": data_b_off,
    }
    return BytesIO(bytes(buf)), locations


def assert_runs_cover(runs: list[QCow2Mapping], offset: int, length: int) -> None:
    """Assert mappings are contiguous, non-empty and exactly cover the requested guest range."""
    assert runs, "expected at least one mapping"
    cursor = offset
    for run in runs:
        assert run.guest_length > 0
        assert run.guest_offset == cursor
        cursor += run.guest_length
    assert cursor == offset + length


def assert_mapping_matches_read(stream: QCow2Stream, runs: list[QCow2Mapping]) -> None:
    """Assert every provable mapping's evidence bytes equal what actually reading the image returns."""
    qcow2 = stream.qcow2
    for run in runs:
        stream.seek(run.guest_offset)
        expected = stream.read(run.guest_length)

        if not run.provable:
            assert run.source == QCow2MapSource.UNKNOWN
            assert run.evidence is None
            continue

        if run.source in (QCow2MapSource.ZERO, QCow2MapSource.UNALLOCATED_ZERO):
            assert expected == b"\x00" * run.guest_length
            assert run.evidence is None
            continue

        run.evidence.seek(run.evidence_offset)
        evidence_bytes = run.evidence.read(run.evidence_length)
        if run.source == QCow2MapSource.COMPRESSED:
            # The evidence is the whole compressed cluster; partial guest ranges still decompress the full cluster
            decompressed = stream._decompress(evidence_bytes)
            offset_in_cluster = run.guest_offset & (qcow2.cluster_size - 1)
            evidence_bytes = decompressed[offset_in_cluster : offset_in_cluster + run.guest_length]

        assert evidence_bytes == expected


def mapping_at(runs: list[QCow2Mapping], offset: int) -> QCow2Mapping:
    for run in runs:
        if run.guest_offset <= offset < run.guest_offset + run.guest_length:
            return run
    raise KeyError(f"No mapping covers guest offset {offset:#x}")


@pytest.mark.parametrize(
    "path",
    [
        "_data/disk/qcow2/basic.qcow2.gz",
        "_data/disk/qcow2/basic-zstd.qcow2.gz",
    ],
)
def test_basic(path: str) -> None:
    with gzip.open(absolute_path(path), "rb") as fh:
        qcow2 = QCow2(fh)

        assert qcow2.backing_file is None
        assert qcow2.data_file is qcow2.fh
        assert qcow2.size == 536870912

        with qcow2.open() as stream:
            for i in range(255):
                assert stream.read(1024 * 1024).strip(bytes([i])) == b"", f"Mismatch at offset {i * 1024 * 1024:#x}"


def test_data_file() -> None:
    path = absolute_path("_data/disk/qcow2/data-file.qcow2.gz")

    # Test with file handle
    with gzip.open(path, "rb") as fh:
        with pytest.raises(Error, match=r"data-file required but not provided \(image_data_file = 'data-file.bin'\)"):
            QCow2(fh)

        with gzip.open(path.with_name("data-file.bin.gz"), "rb") as fh_bin:
            qcow2 = QCow2(fh, data_file=fh_bin)

            assert qcow2.backing_file is None
            assert qcow2.data_file is fh_bin

            with qcow2.open() as stream:
                for i in range(255):
                    assert stream.read(1024 * 1024).strip(bytes([i])) == b"", f"Mismatch at offset {i * 1024 * 1024:#x}"

        # Test with allow_no_data_file
        qcow2 = QCow2(fh, allow_no_data_file=True)
        assert qcow2.data_file is None
        with pytest.raises(Error, match=r"data-file required but not provided \(image_data_file = 'data-file.bin'\)"):
            qcow2.open()

    # Test with Path
    with patch.object(Path, "open", mock_open_gz), patch.object(Path, "exists", return_value=True):
        qcow2 = QCow2(path)

        assert qcow2.backing_file is None
        assert qcow2.data_file is not qcow2.fh

        with qcow2.open() as stream:
            for i in range(255):
                assert stream.read(1024 * 1024).strip(bytes([i])) == b"", f"Mismatch at offset {i * 1024 * 1024:#x}"

    # Test with absolute path
    with patch.object(Path, "open", lambda *args: None), patch.object(Path, "exists", return_value=False):
        qcow2.image_data_file = "/absolute/path/to/nothing.qcow2"
        with pytest.raises(
            Error,
            match=r"data-file '(?:[A-Z]:\\+)?[/\\]+absolute[/\\]+path[/\\]+to[/\\]+nothing\.qcow2' not found \(image_data_file = '/absolute/path/to/nothing\.qcow2'\)",  # noqa: E501
        ):
            qcow2._open_data_file(None)


def test_backing_file() -> None:
    file1 = absolute_path("_data/disk/qcow2/backing-chain-1.qcow2.gz")
    file2 = absolute_path("_data/disk/qcow2/backing-chain-2.qcow2.gz")
    file3 = absolute_path("_data/disk/qcow2/backing-chain-3.qcow2.gz")

    # Test with file handle
    with gzip.open(file1, "rb") as fh1, gzip.open(file2, "rb") as fh2, gzip.open(file3, "rb") as fh3:
        with pytest.raises(
            Error, match=r"backing-file required but not provided \(auto_backing_file = 'backing-chain-2.qcow2'\)"
        ):
            QCow2(fh1)

        with pytest.raises(
            Error, match=r"backing-file required but not provided \(auto_backing_file = 'backing-chain-3.qcow2'\)"
        ):
            QCow2(fh1, backing_file=fh2)

        backing2 = QCow2(fh2, backing_file=fh3)
        assert isinstance(backing2.backing_file, QCow2Stream)

        qcow2 = QCow2(fh1, backing_file=backing2.open())
        assert isinstance(qcow2.backing_file, QCow2Stream)

        # Test with allow_no_backing_file
        qcow2 = QCow2(fh1, allow_no_backing_file=True)
        assert qcow2.backing_file is None
        with pytest.raises(
            Error, match=r"backing-file required but not provided \(auto_backing_file = 'backing-chain-2.qcow2'\)"
        ):
            qcow2.open()

    # Test with Path
    with patch.object(Path, "open", mock_open_gz), patch.object(Path, "exists", return_value=True):
        qcow2 = QCow2(file1)

        assert isinstance(qcow2.backing_file, QCow2Stream)
        assert qcow2.backing_file.qcow2.fh.name == str(file2)

        assert isinstance(qcow2.backing_file.qcow2.backing_file, QCow2Stream)
        assert qcow2.backing_file.qcow2.backing_file.qcow2.fh.name == str(file3)

        # Test reading through the backing chain
        with QCow2(file3).open() as stream:
            assert stream.read(1024 * 1024).strip(b"\x00") == b"Something here"
            assert stream.read(1024 * 1024).strip(b"\x00") == b"Something here too"
            assert stream.read(1024 * 1024).strip(b"\x00") == b"Something here three"
            assert stream.read(1024 * 1024).strip(b"\x00") == b""
            assert stream.read(1024 * 1024).strip(b"\x00") == b""

        with QCow2(file2).open() as stream:
            assert stream.read(1024 * 1024).strip(b"\x00") == b"Nothing here"
            assert stream.read(1024 * 1024).strip(b"\x00") == b"Something here too"
            assert stream.read(1024 * 1024).strip(b"\x00") == b"Something here three"
            assert stream.read(1024 * 1024).strip(b"\x00") == b"Something here four"
            assert stream.read(1024 * 1024).strip(b"\x00") == b""

        with QCow2(file1).open() as stream:
            assert stream.read(1024 * 1024).strip(b"\x00") == b"Nothing here"
            assert stream.read(1024 * 1024).strip(b"\x00") == b"Nothing here two"
            assert stream.read(1024 * 1024).strip(b"\x00") == b"Something here three"
            assert stream.read(1024 * 1024).strip(b"\x00") == b"Something here four"
            assert stream.read(1024 * 1024).strip(b"\x00") == b"Something here five"

    # Test with absolute path
    with patch.object(Path, "open", lambda *args: None), patch.object(Path, "exists", return_value=False):
        qcow2.auto_backing_file = "/absolute/path/to/nothing.qcow2"
        with pytest.raises(
            Error,
            match=r"backing-file '(?:[A-Z]:\\+)?[/\\]+absolute[/\\]+path[/\\]+to[/\\]+nothing\.qcow2' not found \(auto_backing_file = '/absolute/path/to/nothing\.qcow2'\)",  # noqa: E501
        ):
            qcow2._open_backing_file(None)


def test_snapshot() -> None:
    with gzip.open(absolute_path("_data/disk/qcow2/snapshot.qcow2.gz"), "rb") as fh:
        qcow2 = QCow2(fh)

        assert qcow2.backing_file is None
        assert qcow2.data_file is qcow2.fh
        assert qcow2.size == 536870912

        with qcow2.open() as stream:
            assert stream.read(4 * 1024 * 1024).strip(b"\x00") == b""

        assert len(qcow2.snapshots) == 2
        assert qcow2.snapshots[0].id == "1"
        assert qcow2.snapshots[0].name == "you can't see me"
        assert qcow2.snapshots[1].id == "2"
        assert qcow2.snapshots[1].name == "confused"

        with qcow2.snapshots[1].open() as stream:
            assert hashlib.sha1(stream.read(813857)).hexdigest() == "c97f53aece77ea49099d15e5f53af3af5f62fb54"

        with qcow2.snapshots[0].open() as stream:
            assert hashlib.sha1(stream.read(2261577)).hexdigest() == "2c7a6b5f6b5c4739f6d24c11e86c764bdf86096f"


def test_map_runs_extended_l2_subclusters() -> None:
    fh, locations = build_extl2_qcow2()
    qcow2 = QCow2(fh)
    assert qcow2.has_subclusters

    length = 5 * locations["cluster_size"]
    runs = list(qcow2.map_runs(0, length))
    assert_runs_cover(runs, 0, length)

    cluster_size = locations["cluster_size"]
    sub_size = locations["subcluster_size"]

    # Cluster 0: fully unallocated, reads back as zero without a backing file
    unallocated = mapping_at(runs, 0)
    assert unallocated.source == QCow2MapSource.UNALLOCATED_ZERO
    assert unallocated.layer == 0
    assert unallocated.granularity == sub_size
    assert unallocated.provable

    # Cluster 1: fully allocated, evidence lives in the image at data region A
    allocated = mapping_at(runs, cluster_size)
    assert allocated.source == QCow2MapSource.DATA
    assert allocated.evidence is qcow2.data_file
    assert allocated.evidence_range == (locations["data_a"], cluster_size)
    assert allocated.granularity == sub_size

    # Cluster 2: subclusters split into allocated / explicit zero / unallocated / allocated runs
    mixed = 2 * cluster_size
    assert mapping_at(runs, mixed + 0 * sub_size).source == QCow2MapSource.DATA
    zero_run = mapping_at(runs, mixed + 8 * sub_size)
    assert zero_run.source == QCow2MapSource.ZERO
    assert zero_run.guest_length == 8 * sub_size
    assert zero_run.evidence is None
    hole_run = mapping_at(runs, mixed + 16 * sub_size)
    assert hole_run.source == QCow2MapSource.UNALLOCATED_ZERO
    assert hole_run.guest_length == 8 * sub_size
    assert mapping_at(runs, mixed + 24 * sub_size).source == QCow2MapSource.DATA

    # Cluster 3: explicit zero subclusters are distinguished from a plain hole
    explicit_zero = mapping_at(runs, 3 * cluster_size)
    assert explicit_zero.source == QCow2MapSource.ZERO
    assert explicit_zero.guest_length == cluster_size

    with qcow2.open() as stream:
        assert_mapping_matches_read(stream, runs)


def test_map_runs_extended_l2_backing_subclusters() -> None:
    backing = BytesIO(b"\x11" * (16 * 65536))
    fh, locations = build_extl2_qcow2(backing=backing)
    qcow2 = QCow2(fh, backing_file=backing, allow_no_backing_file=True)
    assert qcow2.has_backing_file

    runs = list(qcow2.map_runs(0, 5 * locations["cluster_size"]))

    # Unallocated subclusters now resolve into the raw backing file instead of reading as zero
    unallocated = mapping_at(runs, 0)
    assert unallocated.source == QCow2MapSource.BACKING
    assert unallocated.layer == 1
    assert unallocated.evidence is backing
    assert unallocated.evidence_range == (0, locations["cluster_size"])

    hole_run = mapping_at(runs, 2 * locations["cluster_size"] + 16 * locations["subcluster_size"])
    assert hole_run.source == QCow2MapSource.BACKING

    # Explicit zero subclusters still override the backing file with zeros
    zero_run = mapping_at(runs, 2 * locations["cluster_size"] + 8 * locations["subcluster_size"])
    assert zero_run.source == QCow2MapSource.ZERO

    with qcow2.open() as stream:
        assert_mapping_matches_read(stream, runs)


def test_map_runs_external_data_file() -> None:
    qcow_path = absolute_path("_data/disk/qcow2/data-file.qcow2.gz")
    bin_path = absolute_path("_data/disk/qcow2/data-file.bin.gz")

    with gzip.open(qcow_path, "rb") as fh, gzip.open(bin_path, "rb") as fh_bin:
        qcow2 = QCow2(fh, data_file=fh_bin)
        assert qcow2.has_data_file

        length = 5 * 1024 * 1024
        runs = list(qcow2.map_runs(0, length))
        assert_runs_cover(runs, 0, length)

        assert all(run.source == QCow2MapSource.DATA for run in runs)
        assert all(run.evidence is fh_bin for run in runs)
        # Uncompressed data is mapped 1:1 between guest offsets and the external data file
        assert all(run.evidence_range == (run.guest_offset, run.guest_length) for run in runs)

        with qcow2.open() as stream:
            assert_mapping_matches_read(stream, runs)


@pytest.mark.parametrize(
    "path",
    [
        "_data/disk/qcow2/basic.qcow2.gz",
        "_data/disk/qcow2/basic-zstd.qcow2.gz",
    ],
)
def test_map_runs_compressed_granularity(path: str) -> None:
    with gzip.open(absolute_path(path), "rb") as fh:
        qcow2 = QCow2(fh)
        runs = list(qcow2.map_runs(0, 8 * 1024 * 1024))
        compressed = [run for run in runs if run.source == QCow2MapSource.COMPRESSED]
        assert compressed

        for run in compressed:
            # Compressed clusters are always mapped as a whole guest cluster
            assert run.granularity == qcow2.cluster_size
            assert run.guest_length == qcow2.cluster_size
            assert run.evidence is qcow2.fh
            assert run.evidence_length <= qcow2.cluster_size
            assert run.compression_type == qcow2.compression_type

        with qcow2.open() as stream:
            assert_mapping_matches_read(stream, runs)

        # A sub-range of a compressed cluster still reports the full compressed cluster as evidence
        offset = compressed[0].guest_offset + qcow2.cluster_size // 2
        partial = list(qcow2.map_runs(offset, 1024))
        assert partial[0].source == QCow2MapSource.COMPRESSED
        assert partial[0].evidence_range == compressed[0].evidence_range


def test_map_runs_backing_chain_layers() -> None:
    file1 = absolute_path("_data/disk/qcow2/backing-chain-1.qcow2.gz")
    file2 = absolute_path("_data/disk/qcow2/backing-chain-2.qcow2.gz")
    file3 = absolute_path("_data/disk/qcow2/backing-chain-3.qcow2.gz")

    with gzip.open(file1, "rb") as fh1, gzip.open(file2, "rb") as fh2, gzip.open(file3, "rb") as fh3:
        backing2 = QCow2(fh2, backing_file=fh3)
        qcow2 = QCow2(fh1, backing_file=backing2.open())

        mib = 1024 * 1024
        runs = list(qcow2.map_runs(0, 5 * mib))
        assert_runs_cover(runs, 0, 5 * mib)

        # The same logical ranges are defined by different layers of the backing chain
        assert mapping_at(runs, 0 * mib).layer == 1
        assert mapping_at(runs, 1 * mib).layer == 0
        assert mapping_at(runs, 2 * mib).layer == 2
        assert mapping_at(runs, 3 * mib).layer == 1
        assert mapping_at(runs, 4 * mib).layer == 0

        # Every data mapping names the actual image file the bytes come from
        assert mapping_at(runs, 0 * mib).evidence_name.endswith("backing-chain-2.qcow2.gz")
        assert mapping_at(runs, 2 * mib).evidence_name.endswith("backing-chain-3.qcow2.gz")
        assert mapping_at(runs, 4 * mib).evidence_name.endswith("backing-chain-1.qcow2.gz")

        with qcow2.open() as stream:
            assert_mapping_matches_read(stream, runs)


def test_map_runs_missing_backing_keeps_known_mappings() -> None:
    file1 = absolute_path("_data/disk/qcow2/backing-chain-1.qcow2.gz")
    file2 = absolute_path("_data/disk/qcow2/backing-chain-2.qcow2.gz")

    # Top image without any backing layer: local mappings survive, backing ranges are unprovable
    with gzip.open(file1, "rb") as fh1:
        qcow2 = QCow2(fh1, allow_no_backing_file=True)
        runs = list(qcow2.map_runs(0, 5 * 1024 * 1024))

        data_runs = [run for run in runs if run.source == QCow2MapSource.DATA]
        unknown_runs = [run for run in runs if run.source == QCow2MapSource.UNKNOWN]
        assert data_runs and unknown_runs
        assert all(run.provable for run in data_runs)
        assert all(not run.provable and run.evidence is None for run in unknown_runs)

        # Reading requires the backing file, but the mappings themselves stay available
        with pytest.raises(Error, match="backing-file required but not provided"):
            qcow2.open()
        assert_mapping_matches_read(QCow2Stream(qcow2), runs)

    # A missing layer in the middle of the chain keeps the mappings resolvable through available layers
    with gzip.open(file1, "rb") as fh1, gzip.open(file2, "rb") as fh2:
        # chain-2 is opened without its chain-3 backing layer, then chain-1 references its stream directly
        partial_backing = QCow2(fh2, allow_no_backing_file=True)
        chained = QCow2(fh1, backing_file=QCow2Stream(partial_backing))
        partial_runs = list(chained.map_runs(0, 5 * 1024 * 1024))

        # Local data of both available layers stays provable ...
        assert any(run.source == QCow2MapSource.DATA and run.layer == 0 for run in partial_runs)
        assert any(run.source == QCow2MapSource.DATA and run.layer == 1 for run in partial_runs)
        # ... while ranges that would need the missing chain-3 are unprovable
        unknown_runs = [run for run in partial_runs if run.source == QCow2MapSource.UNKNOWN]
        assert unknown_runs
        assert all(not run.provable and run.evidence is None for run in unknown_runs)
        assert {run.layer for run in unknown_runs} == {1}


def test_map_runs_snapshots_same_range_different_sources() -> None:
    with gzip.open(absolute_path("_data/disk/qcow2/snapshot.qcow2.gz"), "rb") as fh:
        qcow2 = QCow2(fh)

        guest_offset = 0
        current = mapping_at(list(qcow2.map_runs(0, qcow2.cluster_size)), guest_offset)
        snap0 = mapping_at(list(qcow2.snapshots[0].open().map_runs(0, qcow2.cluster_size)), guest_offset)
        snap1 = mapping_at(list(qcow2.snapshots[1].open().map_runs(0, qcow2.cluster_size)), guest_offset)

        # The identical logical range points at different host ranges in each snapshot's L1 table
        assert current.source == snap0.source == snap1.source == QCow2MapSource.DATA
        assert current.evidence is snap0.evidence is snap1.evidence is qcow2.fh
        assert len({current.evidence_offset, snap0.evidence_offset, snap1.evidence_offset}) == 3

        with qcow2.open() as current_stream, qcow2.snapshots[0].open() as snap0_stream, qcow2.snapshots[
            1
        ].open() as snap1_stream:
            assert_mapping_matches_read(current_stream, [current])
            assert_mapping_matches_read(snap0_stream, [snap0])
            assert_mapping_matches_read(snap1_stream, [snap1])

            # And the bytes actually returned by the three views differ
            current_stream.seek(guest_offset)
            snap0_stream.seek(guest_offset)
            snap1_stream.seek(guest_offset)
            values = {
                current_stream.read(qcow2.cluster_size),
                snap0_stream.read(qcow2.cluster_size),
                snap1_stream.read(qcow2.cluster_size),
            }
            assert len(values) == 3
