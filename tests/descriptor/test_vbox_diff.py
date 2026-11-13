from __future__ import annotations

from io import StringIO
from uuid import UUID

import pytest

from dissect.hypervisor.descriptor.vbox import VBox
from dissect.hypervisor.descriptor.vbox_diff import DiskChangeType, diff_disk_attachments, explain_disk_attachments
from tests._util import absolute_path

NS = 'xmlns="http://www.virtualbox.org/"'

ROOT = UUID("22222222-0000-0000-0000-000000000000")
BRANCH_A = UUID("33333333-0000-0000-0000-000000000000")
BRANCH_B = UUID("44444444-0000-0000-0000-000000000000")

BASE_A = UUID("aaaaaaaa-0000-0000-0000-000000000001")
LAYER_A2 = UUID("aaaaaaaa-0000-0000-0000-000000000002")
LAYER_A3 = UUID("aaaaaaaa-0000-0000-0000-000000000003")
LAYER_A4 = UUID("aaaaaaaa-0000-0000-0000-000000000004")
BASE_B = UUID("bbbbbbbb-0000-0000-0000-000000000001")
LAYER_B2 = UUID("bbbbbbbb-0000-0000-0000-000000000002")


@pytest.fixture()
def forked() -> VBox:
    with absolute_path("_data/descriptor/vbox/forked-snapshots.vbox").open() as fh:
        return VBox(fh)


def _descriptor(media_xml: str, image_uuid: UUID | None, port: int = 0, controller: str = "SATA") -> VBox:
    image_node = f'<Image uuid="{{{image_uuid}}}"/>' if image_uuid is not None else ""
    return VBox(
        StringIO(
            f"""<VirtualBox {NS} version="1.19-linux">
  <Machine uuid="{{55555555-0000-0000-0000-000000000000}}" name="t"
    currentSnapshot="{{66666666-0000-0000-0000-000000000000}}">
    <MediaRegistry><HardDisks>{media_xml}</HardDisks></MediaRegistry>
    <Snapshot uuid="{{66666666-0000-0000-0000-000000000000}}" name="s" timeStamp="2026-02-01T08:00:00Z">
      <Hardware><StorageControllers><StorageController name="{controller}" type="AHCI" PortCount="4">
        <AttachedDevice type="HardDisk" port="{port}" device="0">{image_node}</AttachedDevice>
      </StorageController></StorageControllers></Hardware>
    </Snapshot>
    <Hardware><StorageControllers><StorageController name="{controller}" type="AHCI" PortCount="4">
      <AttachedDevice type="HardDisk" port="{port}" device="0">{image_node}</AttachedDevice>
    </StorageController></StorageControllers></Hardware>
  </Machine></VirtualBox>"""
        )
    )


def test_media_chain_helpers(forked: VBox) -> None:
    leaf = forked.media[LAYER_A3]
    assert [disk.uuid for disk in leaf.chain] == [BASE_A, LAYER_A2, LAYER_A3]
    assert [disk.uuid for disk in leaf.ancestors] == [BASE_A, LAYER_A2]
    assert forked.media[BASE_A].is_ancestor_of(leaf)
    assert not leaf.is_ancestor_of(forked.media[BASE_A])


def test_attachments_include_slot_and_unresolved_media() -> None:
    dangling = _descriptor("", UUID("99999999-0000-0000-0000-000000000099"))
    attachment = dangling.hardware.attachments[0]

    assert attachment.slot == ("SATA", 0, 0)
    assert attachment.image_uuid == UUID("99999999-0000-0000-0000-000000000099")
    assert attachment.medium is None
    # Resolved-only API keeps filtering unresolved attachments.
    assert dangling.hardware.disks == []
    # The raw XML node remains available for traceability.
    assert attachment.element.tag.endswith("AttachedDevice")
    assert attachment.image_element is not None


def test_forked_tree_layout(forked: VBox) -> None:
    assert len(forked.snapshots) == 3
    assert forked.snapshots[BRANCH_A].parent.uuid == ROOT
    assert forked.snapshots[BRANCH_B].parent.uuid == ROOT

    root_slots = {attachment.slot: attachment.image_uuid for attachment in forked.snapshots[ROOT].hardware.attachments}
    assert root_slots == {("SATA", 0, 0): BASE_A, ("SATA", 1, 0): BASE_B}


def test_layer_added_and_removed(forked: VBox) -> None:
    root = forked.snapshots[ROOT]
    branch_a = forked.snapshots[BRANCH_A]

    added = diff_disk_attachments(root, branch_a)
    assert [change.change_type for change in added] == [DiskChangeType.LAYER_ADDED]
    change = added[0]
    assert change.left_medium.uuid == BASE_A
    assert change.right_medium.uuid == LAYER_A3
    assert change.common_ancestor.uuid == BASE_A
    assert "2 differencing layer(s) added" in change.notes

    removed = diff_disk_attachments(branch_a, root)
    assert [change.change_type for change in removed] == [DiskChangeType.LAYER_REMOVED]
    assert removed[0].common_ancestor.uuid == BASE_A
    assert "2 differencing layer(s) removed" in removed[0].notes


def test_disk_swap_with_shared_ancestor(forked: VBox) -> None:
    branch_a = forked.snapshots[BRANCH_A]
    branch_b = forked.snapshots[BRANCH_B]

    changes = diff_disk_attachments(branch_a, branch_b)
    swap = next(change for change in changes if change.change_type == DiskChangeType.DISK_SWAPPED)

    assert swap.left_medium.uuid == LAYER_A3
    assert swap.right_medium.uuid == LAYER_A4
    # The forked differencing disks share their parent layer, but are distinct leaf images.
    assert swap.common_ancestor.uuid == LAYER_A2
    assert swap.slot == ("SATA", 0, 0)


def test_move_to_another_slot_with_layer_change(forked: VBox) -> None:
    branch_a = forked.snapshots[BRANCH_A]
    branch_b = forked.snapshots[BRANCH_B]

    changes = diff_disk_attachments(branch_a, branch_b)
    move = next(change for change in changes if change.change_type == DiskChangeType.MOVED_SLOT)

    assert move.left.slot == ("SATA", 1, 0)
    assert move.right.slot == ("SATA", 2, 0)
    assert move.left_medium.uuid == BASE_B
    assert move.right_medium.uuid == LAYER_B2
    assert "moved and extended" in move.notes

    # The same slot must not also be reported as a separate detach/attach pair.
    assert all(change.change_type != DiskChangeType.MEDIUM_DETACHED for change in changes)


def test_exact_uuid_move_without_layer_change() -> None:
    disk = '<HardDisk uuid="{11111111-0000-0000-0000-000000000001}" location="disk.vdi" format="VDI"/>'
    left = _descriptor(disk, UUID("11111111-0000-0000-0000-000000000001"), port=0)
    right = _descriptor(disk, UUID("11111111-0000-0000-0000-000000000001"), port=1)

    changes = diff_disk_attachments(left.hardware, right.hardware)
    assert [change.change_type for change in changes] == [DiskChangeType.MOVED_SLOT]
    assert changes[0].notes is None


def test_attach_and_detach() -> None:
    disk = '<HardDisk uuid="{11111111-0000-0000-0000-000000000001}" location="disk.vdi" format="VDI"/>'
    empty = _descriptor("", None)
    with_disk = _descriptor(disk, UUID("11111111-0000-0000-0000-000000000001"))

    attached = diff_disk_attachments(empty.hardware, with_disk.hardware)
    assert [change.change_type for change in attached] == [DiskChangeType.MEDIUM_ATTACHED]
    assert attached[0].left.image_uuid is None
    detached = diff_disk_attachments(with_disk.hardware, empty.hardware)
    assert [change.change_type for change in detached] == [DiskChangeType.MEDIUM_DETACHED]
    assert detached[0].right.image_uuid is None


def test_path_only_change_is_distinct() -> None:
    uuid = UUID("77777777-0000-0000-0000-000000000001")
    left = _descriptor(f'<HardDisk uuid="{{{uuid}}}" location="old/path.vdi" format="VDI"/>', uuid)
    right = _descriptor(f'<HardDisk uuid="{{{uuid}}}" location="new/path.vdi" format="VDI"/>', uuid)

    changes = diff_disk_attachments(left.hardware, right.hardware)
    assert [change.change_type for change in changes] == [DiskChangeType.PATH_CHANGED]
    assert changes[0].left_medium.location == "old/path.vdi"
    assert changes[0].right_medium.location == "new/path.vdi"


def test_missing_parent_keeps_unknown_relationship() -> None:
    left = _descriptor("", UUID("99999999-0000-0000-0000-000000000099"))
    right = _descriptor(
        '<HardDisk uuid="{11111111-0000-0000-0000-000000000001}" location="present.vdi" format="VDI"/>',
        UUID("11111111-0000-0000-0000-000000000001"),
    )

    changes = diff_disk_attachments(left.hardware, right.hardware)
    assert [change.change_type for change in changes] == [DiskChangeType.UNKNOWN_RELATIONSHIP]
    change = changes[0]
    assert change.left_medium is None
    assert change.right_medium.uuid == UUID("11111111-0000-0000-0000-000000000001")
    assert "missing from its media registry" in change.notes


def test_every_change_is_traceable_to_both_nodes(forked: VBox) -> None:
    changes = diff_disk_attachments(forked.snapshots[BRANCH_A], forked.snapshots[BRANCH_B])
    assert changes
    for change in changes:
        assert change.left is not None
        assert change.right is not None
        assert change.left.element.tag.endswith("AttachedDevice")
        assert change.right.element.tag.endswith("AttachedDevice")
        # Each side points back at its owning hardware node.
        assert change.left.hardware.element.tag.endswith("Hardware")
        assert change.right.hardware.element.tag.endswith("Hardware")
        assert change.explanation


def test_vbox_convenience_entry_point(forked: VBox) -> None:
    changes = forked.diff_snapshots(ROOT, BRANCH_A)
    assert changes
    assert all(change.explanation for change in changes)
    report = explain_disk_attachments(forked.snapshots[ROOT], forked.snapshots[BRANCH_A])
    assert "layer_added" in report


def test_no_changes_returns_empty() -> None:
    disk = '<HardDisk uuid="{11111111-0000-0000-0000-000000000001}" location="disk.vdi" format="VDI"/>'
    descriptor = _descriptor(disk, UUID("11111111-0000-0000-0000-000000000001"))
    assert diff_disk_attachments(descriptor.hardware, descriptor.hardware) == []
    assert explain_disk_attachments(descriptor.hardware, descriptor.hardware) == "No disk attachment differences."
