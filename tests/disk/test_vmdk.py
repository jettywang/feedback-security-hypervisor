from __future__ import annotations

import gzip
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from dissect.hypervisor.disk.c_vmdk import c_vmdk
from dissect.hypervisor.disk.vmdk import VMDK, DiskDescriptor, ExtentDescriptor, ParentChainError, open_parent
from tests._util import absolute_path


def test_vmdk_sesparse() -> None:
    with gzip.open(absolute_path("_data/disk/vmdk/sesparse.vmdk.gz"), "rb") as fh:
        vmdk = VMDK(fh)

        disk = vmdk.disks[0]

        assert disk.is_sesparse
        assert disk._grain_directory_size == 0x20000
        assert disk._grain_table_size == 0x1000
        assert disk._grain_entry_type == c_vmdk.uint64
        assert disk._grain_directory[0] == 0x1000000000000000

        header = disk.header
        assert header.magic == c_vmdk.SESPARSE_CONST_HEADER_MAGIC
        assert header.version == 0x200000001

        assert vmdk.read(0x1000000) == b"a" * 0x1000000


@pytest.mark.parametrize(
    ("extent_description", "expected_extents"),
    [
        (
            'RW 123456789 SPARSE "disk.vmdk"',
            [
                ExtentDescriptor(
                    raw='RW 123456789 SPARSE "disk.vmdk"',
                    access_mode="RW",
                    sectors=123456789,
                    type="SPARSE",
                    filename='"disk.vmdk"',
                    start_sector=None,
                    partition_uuid=None,
                    device_identifier=None,
                ),
            ],
        ),
        (
            'RW 123456789 FLAT "disk-flat.vmdk" 0',
            [
                ExtentDescriptor(
                    raw='RW 123456789 FLAT "disk-flat.vmdk" 0',
                    access_mode="RW",
                    sectors=123456789,
                    type="FLAT",
                    filename='"disk-flat.vmdk"',
                    start_sector=0,
                    partition_uuid=None,
                    device_identifier=None,
                )
            ],
        ),
        (
            "RDONLY 0 ZERO",
            [
                ExtentDescriptor(
                    raw="RDONLY 0 ZERO",
                    access_mode="RDONLY",
                    sectors=0,
                    type="ZERO",
                ),
            ],
        ),
        (
            'NOACCESS 123456789 SPARSE "disk-sparse.vmdk" 123 partition-uuid device-id',
            [
                ExtentDescriptor(
                    raw='NOACCESS 123456789 SPARSE "disk-sparse.vmdk" 123 partition-uuid device-id',
                    access_mode="NOACCESS",
                    sectors=123456789,
                    type="SPARSE",
                    filename='"disk-sparse.vmdk"',
                    start_sector=123,
                    partition_uuid="partition-uuid",
                    device_identifier="device-id",
                ),
            ],
        ),
        ("RW 1234567890", []),
        ('RDONLY "file.vmdk"', []),
        ("NOACCESS", []),
        (
            'RW 1234567890 SPARSE "disk with spaces.vmdk"',
            [
                ExtentDescriptor(
                    raw='RW 1234567890 SPARSE "disk with spaces.vmdk"',
                    access_mode="RW",
                    sectors=1234567890,
                    type="SPARSE",
                    filename='"disk with spaces.vmdk"',
                    start_sector=None,
                    partition_uuid=None,
                    device_identifier=None,
                )
            ],
        ),
        (
            'RW 1234567890 SPARSE "disk with spaces.vmdk" 123',
            [
                ExtentDescriptor(
                    raw='RW 1234567890 SPARSE "disk with spaces.vmdk" 123',
                    access_mode="RW",
                    sectors=1234567890,
                    type="SPARSE",
                    filename='"disk with spaces.vmdk"',
                    start_sector=123,
                    partition_uuid=None,
                    device_identifier=None,
                )
            ],
        ),
        (
            'RW 1234567890 SPARSE "disk with spaces.vmdk" 123 part-uuid',
            [
                ExtentDescriptor(
                    raw='RW 1234567890 SPARSE "disk with spaces.vmdk" 123 part-uuid',
                    access_mode="RW",
                    sectors=1234567890,
                    type="SPARSE",
                    filename='"disk with spaces.vmdk"',
                    start_sector=123,
                    partition_uuid="part-uuid",
                    device_identifier=None,
                )
            ],
        ),
        (
            'RW 1234567890 SPARSE "disk with spaces.vmdk" 123 part-uuid device-id',
            [
                ExtentDescriptor(
                    raw='RW 1234567890 SPARSE "disk with spaces.vmdk" 123 part-uuid device-id',
                    access_mode="RW",
                    sectors=1234567890,
                    type="SPARSE",
                    filename='"disk with spaces.vmdk"',
                    start_sector=123,
                    partition_uuid="part-uuid",
                    device_identifier="device-id",
                )
            ],
        ),
        (
            r'RW 16777216 SPARSE "this is an example "\' diskëäô:)\\\'`\foo.vmdk" 123',
            [
                ExtentDescriptor(
                    raw=r'RW 16777216 SPARSE "this is an example "\' diskëäô:)\\\'`\foo.vmdk" 123',
                    access_mode="RW",
                    sectors=16777216,
                    type="SPARSE",
                    filename=r'"this is an example "\' diskëäô:)\\\'`\foo.vmdk"',
                    start_sector=123,
                    partition_uuid=None,
                    device_identifier=None,
                )
            ],
        ),
        (
            r'RW 13371337 SPARSE "🦊 🦊 🦊.vmdk"',
            [
                ExtentDescriptor(
                    raw=r'RW 13371337 SPARSE "🦊 🦊 🦊.vmdk"',
                    access_mode="RW",
                    sectors=13371337,
                    type="SPARSE",
                    filename='"🦊 🦊 🦊.vmdk"',
                )
            ],
        ),
    ],
    ids=(
        "sparse",
        "flat",
        "zero",
        "sparse-ids",
        "bad-1",
        "bad-2",
        "bad-3",
        "spaces-four-parts",
        "spaces-five-parts",
        "spaces-six-parts",
        "spaces-seven-parts",
        "specials-five-parts",
        "emoji-four-parts",
    ),
)
def test_vmdk_extent_description(extent_description: str, expected_extents: list[ExtentDescriptor]) -> None:
    """Test if we correctly parse VMDK sparse and flat extent descriptions.

    Resources:
        - https://github.com/libyal/libvmdk/blob/main/documentation/VMWare%20Virtual%20Disk%20Format%20(VMDK).asciidoc#22-extent-descriptions
    """
    descriptor = DiskDescriptor.parse(extent_description)
    assert descriptor.extents == expected_extents


def test_open_parent_all_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test open_parent handles absolute and relative filename_hint paths."""

    # Mock Path.exists to simulate file existence
    def mock_exists(path: Path) -> bool:
        return str(path) in {
            "/a/b/c/d.vmdk",  # Case: absolute path
            "base/relative/hint.vmdk",  # Case: full relative path
            "base/hint.vmdk",  # Case: basename in same dir
            "../sibling/hint.vmdk",  # Case: fallback to sibling
        }

    with monkeypatch.context() as m:
        m.setattr("pathlib.Path.exists", mock_exists)

        # Mock VMDK to avoid real file I/O
        mock_vmdk = MagicMock()
        m.setattr("dissect.hypervisor.disk.vmdk.VMDK", lambda *args, **kwargs: mock_vmdk)
        mock_vmdk.path = "mocked-path"

        # Case: Absolute path — should use /a/b/c/d.vmdk directly
        vmdk = open_parent(Path("base"), "/a/b/c/d.vmdk")
        assert str(vmdk.path) == "mocked-path"

        # Case: Full relative path — try base/relative/hint.vmdk
        vmdk = open_parent(Path("base"), "relative/hint.vmdk")
        assert str(vmdk.path) == "mocked-path"

        # Case: Basename only — fall back to base/hint.vmdk
        vmdk = open_parent(Path("base"), "hint.vmdk")
        assert str(vmdk.path) == "mocked-path"

        # Case: Fallback to sibling — try ../sibling/hint.vmdk
        vmdk = open_parent(Path("base"), "sibling/hint.vmdk")
        assert str(vmdk.path) == "mocked-path"


def _write_vmdk_sample(
    directory: Path,
    name: str,
    cid: str,
    parent_cid: str = "ffffffff",
    parent_hint: str | None = None,
) -> Path:
    """Write a minimal monolithic-flat VMDK sample (descriptor + 1 sector flat extent)."""
    flat_name = f"{name.removesuffix('.vmdk')}-flat.vmdk"
    lines = [
        "# Disk DescriptorFile",
        "version=1",
        f"CID={cid}",
        f"parentCID={parent_cid}",
        'createType="monolithicFlat"',
    ]
    if parent_hint is not None:
        lines.append(f'parentFileNameHint="{parent_hint}"')
    lines += [
        "",
        "# Extent description",
        f'RW 1 FLAT "{flat_name}" 0',
        "",
        "# The Disk Data Base",
        "#DDB",
        'ddb.virtualHWVersion = "4"',
    ]
    path = directory.joinpath(name)
    path.write_text("\n".join(lines) + "\n")
    directory.joinpath(flat_name).write_bytes(b"\x00" * 512)
    return path


@pytest.fixture
def tracked_open(monkeypatch: pytest.MonkeyPatch) -> list:
    """Track all file handles opened through Path.open."""
    opened = []
    real_open = Path.open

    def tracking_open(self: Path, *args, **kwargs):  # noqa: ANN001, ANN202
        fh = real_open(self, *args, **kwargs)
        opened.append(fh)
        return fh

    monkeypatch.setattr(Path, "open", tracking_open)
    return opened


def test_vmdk_parent_chain_migrated_candidates(tmp_path: Path) -> None:
    """A renamed chain is resolved from caller-supplied candidates by CID, not by file name."""
    _write_vmdk_sample(tmp_path, "original-name.vmdk", cid="99999999")  # same-named but different disk
    _write_vmdk_sample(tmp_path, "renamed-parent.vmdk", cid="11111111")
    child = _write_vmdk_sample(
        tmp_path, "renamed-child.vmdk", cid="22222222", parent_cid="11111111", parent_hint="original-name.vmdk"
    )

    candidates = list(tmp_path.iterdir())
    vmdk = VMDK(child, parent_candidates=candidates)

    assert vmdk.parent is not None
    assert vmdk.parent.descriptor.attr["CID"] == "11111111"
    assert vmdk.parent.descriptor.attr["parentCID"] == "ffffffff"
    vmdk.close()


def test_vmdk_parent_chain_same_name_wrong_disk(tmp_path: Path, tracked_open: list) -> None:
    """A same-named but different parent disk is rejected by CID and all handles are released."""
    _write_vmdk_sample(tmp_path, "disk.vmdk", cid="99999999")
    child = _write_vmdk_sample(
        tmp_path, "disk-000001.vmdk", cid="22222222", parent_cid="11111111", parent_hint="disk.vmdk"
    )

    with pytest.raises(ParentChainError, match=r"child CID 22222222.*has CID 99999999, expected 11111111"):
        VMDK(child)

    assert len(tracked_open) > 0
    assert all(fh.closed for fh in tracked_open)


def test_vmdk_parent_chain_two_level_cycle(tmp_path: Path, tracked_open: list) -> None:
    """A two-level cyclic chain is rejected and all handles are released."""
    _write_vmdk_sample(tmp_path, "a.vmdk", cid="aaaaaaaa", parent_cid="bbbbbbbb", parent_hint="b.vmdk")
    _write_vmdk_sample(tmp_path, "b.vmdk", cid="bbbbbbbb", parent_cid="aaaaaaaa", parent_hint="a.vmdk")

    with pytest.raises(ParentChainError, match=r"Cyclic.*child CID bbbbbbbb.*parent CID aaaaaaaa"):
        VMDK(tmp_path / "a.vmdk")

    assert len(tracked_open) > 0
    assert all(fh.closed for fh in tracked_open)


def test_vmdk_parent_chain_ambiguous_candidates(tmp_path: Path) -> None:
    """Multiple candidates matching the required parent CID are rejected as ambiguous."""
    first = _write_vmdk_sample(tmp_path, "first.vmdk", cid="11111111")
    second = _write_vmdk_sample(tmp_path, "second.vmdk", cid="11111111")
    child = _write_vmdk_sample(
        tmp_path, "child.vmdk", cid="22222222", parent_cid="11111111", parent_hint="parent.vmdk"
    )

    with pytest.raises(ParentChainError, match=r"Ambiguous parent for child CID 22222222"):
        VMDK(child, parent_candidates=[first, second])
