from __future__ import annotations

import gzip
from pathlib import Path
from typing import BinaryIO
from unittest.mock import patch
from uuid import UUID

import pytest

from dissect.hypervisor.disk.hdd import HDD
from tests._util import absolute_path

Path_open = Path.open

BASE_GUID = UUID("11111111-1111-1111-1111-111111111111")
MID_GUID = UUID("22222222-2222-2222-2222-222222222222")
TOP_GUID = UUID("33333333-3333-3333-3333-333333333333")
NULL_GUID = UUID("00000000-0000-0000-0000-000000000000")


@pytest.fixture
def plain_hdd() -> Path:
    return absolute_path("_data/disk/hdd/plain.hdd")


@pytest.fixture
def expanding_hdd() -> Path:
    return absolute_path("_data/disk/hdd/expanding.hdd")


@pytest.fixture
def split_hdd() -> Path:
    return absolute_path("_data/disk/hdd/split.hdd")


def mock_open_gz(self: Path, *args, **kwargs) -> BinaryIO:
    if self.suffix.lower() != ".hds":
        return Path_open(self, *args, **kwargs)

    return gzip.open(self.with_suffix(self.suffix + ".gz"))


def test_plain_hdd(plain_hdd: Path) -> None:
    hdd = HDD(plain_hdd)
    storages = hdd.descriptor.storage_data.storages

    assert len(storages) == 1
    assert storages[0].start == 0
    assert storages[0].end == 204800
    assert len(storages[0].images) == 1
    assert storages[0].images[0].type == "Plain"

    with patch.object(Path, "open", mock_open_gz):
        stream = hdd.open()

        for i in range(100):
            assert stream.read(1024 * 1024).strip(bytes([i])) == b"", f"Mismatch at offset {i * 1024 * 1024:#x}"


def test_expanding_hdd(expanding_hdd: Path) -> None:
    hdd = HDD(expanding_hdd)
    storages = hdd.descriptor.storage_data.storages

    assert len(storages) == 1
    assert storages[0].start == 0
    assert storages[0].end == 204800
    assert len(storages[0].images) == 1
    assert storages[0].images[0].type == "Compressed"

    with patch.object(Path, "open", mock_open_gz):
        stream = hdd.open()

        for i in range(100):
            assert stream.read(1024 * 1024).strip(bytes([i])) == b"", f"Mismatch at offset {i * 1024 * 1024:#x}"


def test_split_hdd(split_hdd: Path) -> None:
    hdd = HDD(split_hdd)
    storages = hdd.descriptor.storage_data.storages

    assert len(storages) == 6

    split_sizes = [3989504, 3989504, 3989504, 3989504, 3989504, 1024000]

    start = 0

    for storage, split_size in zip(storages, split_sizes, strict=False):
        assert storage.start == start
        assert storage.end == start + split_size
        assert len(storage.images) == 1
        assert storage.images[0].type == "Compressed"

        start = storage.end

    with patch.object(Path, "open", mock_open_gz):
        stream = hdd.open()

        assert stream.read(1024 * 1024).strip(b"\x01") == b""

        offset = 0
        for i, split_size in enumerate(split_sizes):
            offset += split_size * 512
            stream.seek(offset - 512)

            buf = stream.read(1024)
            if i < 5:
                assert buf == bytes([i + 1] * 512) + bytes([i + 2] * 512)
            else:
                assert buf == bytes([i + 1] * 512)


def test_file_use_parent(plain_hdd: Path) -> None:
    hdd = HDD(plain_hdd.joinpath("plain.hdd"))
    storages = hdd.descriptor.storage_data.storages

    assert len(storages) == 1


def _make_hdd(
    tmp_path: Path,
    storages: list[tuple[int, int, list[tuple[UUID, str, str]]]],
    shots: list[tuple[UUID, UUID]],
    top_guid: UUID | None = None,
    missing_files: tuple[str, ...] = (),
) -> Path:
    """Create a synthetic .hdd directory with a descriptor and (empty) image files."""
    hdd_dir = tmp_path / "synthetic.hdd"
    hdd_dir.mkdir()

    storages_xml = ""
    for start, end, images in storages:
        images_xml = "".join(
            f"<Image><GUID>{guid}</GUID><Type>{type_}</Type><File>{file}</File></Image>"
            for guid, type_, file in images
        )
        storages_xml += (
            f"<Storage><Start>{start}</Start><End>{end}</End><Blocksize>2048</Blocksize>{images_xml}</Storage>"
        )

    shots_xml = "".join(f"<Shot><GUID>{guid}</GUID><ParentGUID>{parent}</ParentGUID></Shot>" for guid, parent in shots)
    top_xml = f"<TopGUID>{top_guid}</TopGUID>" if top_guid else ""

    hdd_dir.joinpath("DiskDescriptor.xml").write_text(
        "<?xml version='1.0' encoding='UTF-8'?>"
        '<Parallels_disk_image Version="1.0">'
        f"<StorageData>{storages_xml}</StorageData>"
        f"<Snapshots>{top_xml}{shots_xml}</Snapshots>"
        "</Parallels_disk_image>"
    )

    for _, _, images in storages:
        for _, _, file in images:
            if file not in missing_files:
                hdd_dir.joinpath(file).touch()

    return hdd_dir


def _chain_storages() -> list[tuple[int, int, list[tuple[UUID, str, str]]]]:
    """Two storages, each with images for a full BASE -> MID -> TOP chain."""
    return [
        (
            0,
            100,
            [
                (BASE_GUID, "Compressed", f"synthetic.0.{BASE_GUID}.hds"),
                (MID_GUID, "Compressed", f"synthetic.0.{MID_GUID}.hds"),
                (TOP_GUID, "Compressed", f"synthetic.0.{TOP_GUID}.hds"),
            ],
        ),
        (
            100,
            200,
            [
                (BASE_GUID, "Compressed", f"synthetic.1.{BASE_GUID}.hds"),
                (MID_GUID, "Compressed", f"synthetic.1.{MID_GUID}.hds"),
                (TOP_GUID, "Compressed", f"synthetic.1.{TOP_GUID}.hds"),
            ],
        ),
    ]


def _chain_shots() -> list[tuple[UUID, UUID]]:
    return [
        (BASE_GUID, NULL_GUID),
        (MID_GUID, BASE_GUID),
        (TOP_GUID, MID_GUID),
    ]


def test_verify_ok(tmp_path: Path) -> None:
    hdd = HDD(_make_hdd(tmp_path, _chain_storages(), _chain_shots(), top_guid=TOP_GUID))

    report = hdd.verify()

    assert report.ok
    assert report.issues == []
    assert report.guid == TOP_GUID
    assert report.chain == [TOP_GUID, MID_GUID, BASE_GUID]


def test_verify_missing_fragment(tmp_path: Path) -> None:
    """The descriptor describes a complete chain, but one fragment file is not delivered."""
    missing_file = f"synthetic.1.{MID_GUID}.hds"
    hdd = HDD(
        _make_hdd(tmp_path, _chain_storages(), _chain_shots(), top_guid=TOP_GUID, missing_files=(missing_file,))
    )

    report = hdd.verify()

    assert not report.ok
    assert report.chain == [TOP_GUID, MID_GUID, BASE_GUID]

    issues = report.by_category("missing-file")
    assert len(issues) == 1
    assert issues[0].file == missing_file
    assert issues[0].guid == MID_GUID
    assert issues[0].storage.start == 100
    assert issues[0].storage.end == 200


def test_verify_parent_cycle(tmp_path: Path) -> None:
    shots = [
        (BASE_GUID, NULL_GUID),
        (MID_GUID, TOP_GUID),
        (TOP_GUID, MID_GUID),
    ]
    hdd = HDD(_make_hdd(tmp_path, _chain_storages(), shots, top_guid=TOP_GUID))

    report = hdd.verify()

    assert not report.ok
    issues = report.by_category("parent-cycle")
    assert len(issues) == 1
    assert issues[0].guid == TOP_GUID
    assert report.chain == [TOP_GUID, MID_GUID]


def test_verify_duplicate_guid(tmp_path: Path) -> None:
    shots = _chain_shots() + [(MID_GUID, BASE_GUID)]
    hdd = HDD(_make_hdd(tmp_path, _chain_storages(), shots, top_guid=TOP_GUID))

    report = hdd.verify()

    assert not report.ok
    issues = report.by_category("duplicate-guid")
    assert len(issues) == 1
    assert issues[0].guid == MID_GUID


def test_verify_missing_parent(tmp_path: Path) -> None:
    unknown_guid = UUID("44444444-4444-4444-4444-444444444444")
    shots = [
        (BASE_GUID, NULL_GUID),
        (MID_GUID, unknown_guid),
        (TOP_GUID, MID_GUID),
    ]
    hdd = HDD(_make_hdd(tmp_path, _chain_storages(), shots, top_guid=TOP_GUID))

    report = hdd.verify()

    assert not report.ok
    issues = report.by_category("missing-parent")
    assert len(issues) == 1
    assert issues[0].guid == unknown_guid
    assert report.chain == [TOP_GUID, MID_GUID]


def test_verify_unknown_snapshot(tmp_path: Path) -> None:
    unknown_guid = UUID("44444444-4444-4444-4444-444444444444")
    hdd = HDD(_make_hdd(tmp_path, _chain_storages(), _chain_shots(), top_guid=TOP_GUID))

    report = hdd.verify(unknown_guid)

    assert not report.ok
    issues = report.by_category("unknown-snapshot")
    assert len(issues) == 1
    assert issues[0].guid == unknown_guid
    assert report.chain == []


def test_verify_missing_intermediate_generation(tmp_path: Path) -> None:
    """One storage is missing the image for the intermediate snapshot generation."""
    storages = _chain_storages()
    storages[1][2].pop(1)
    hdd = HDD(_make_hdd(tmp_path, storages, _chain_shots(), top_guid=TOP_GUID))

    report = hdd.verify()

    assert not report.ok
    issues = report.by_category("missing-image")
    assert len(issues) == 1
    assert issues[0].guid == MID_GUID
    assert issues[0].storage.start == 100
    assert issues[0].storage.end == 200


def test_verify_storage_gap(tmp_path: Path) -> None:
    storages = _chain_storages()
    storages[1] = (150, 250, storages[1][2])
    hdd = HDD(_make_hdd(tmp_path, storages, _chain_shots(), top_guid=TOP_GUID))

    report = hdd.verify()

    assert not report.ok
    issues = report.by_category("storage-gap")
    assert len(issues) == 1
    assert issues[0].storage.start == 150
    assert issues[0].storage.end == 250


def test_verify_storage_overlap(tmp_path: Path) -> None:
    storages = _chain_storages()
    storages[1] = (50, 250, storages[1][2])
    hdd = HDD(_make_hdd(tmp_path, storages, _chain_shots(), top_guid=TOP_GUID))

    report = hdd.verify()

    assert not report.ok
    issues = report.by_category("storage-overlap")
    assert len(issues) == 1
    assert issues[0].storage.start == 50
    assert issues[0].storage.end == 250
