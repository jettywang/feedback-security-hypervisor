from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING
from uuid import UUID

from dissect.hypervisor.descriptor.vbox import Attachment, HardDisk, Hardware

if TYPE_CHECKING:
    from dissect.hypervisor.descriptor.vbox import Machine, Snapshot

Slot = tuple[str | None, int, int]


class DiskChangeType(Enum):
    """The kind of disk difference observed between two hardware states."""

    DISK_SWAPPED = "disk_swapped"
    LAYER_ADDED = "layer_added"
    LAYER_REMOVED = "layer_removed"
    MOVED_SLOT = "moved_slot"
    MEDIUM_ATTACHED = "medium_attached"
    MEDIUM_DETACHED = "medium_detached"
    PATH_CHANGED = "path_changed"
    UNKNOWN_RELATIONSHIP = "unknown_relationship"


@dataclass(frozen=True)
class DiskChange:
    """A single, traceable disk difference between two hardware states.

    Every change keeps references to the controller attachment on each side (when present) and to
    the resolved media. Attachments whose medium UUID is missing from a media registry are kept as
    unresolved relationships and never reconstructed from a file path.

    Args:
        change_type: The classified kind of difference.
        left: The attachment on the left side, or ``None`` when the slot only exists on the right.
        right: The attachment on the right side, or ``None`` when the slot only exists on the left.
        left_medium: The resolved left medium, or ``None`` when absent or not in the registry.
        right_medium: The resolved right medium, or ``None`` when absent or not in the registry.
        common_ancestor: The shared ancestor medium when the chains overlap, otherwise ``None``.
        notes: Additional human-readable detail, or ``None``.
    """

    change_type: DiskChangeType
    left: Attachment | None
    right: Attachment | None
    left_medium: HardDisk | None
    right_medium: HardDisk | None
    common_ancestor: HardDisk | None = None
    notes: str | None = None

    @property
    def slot(self) -> Slot:
        """The controller slot the change is reported on, preferring the left attachment."""
        attachment = self.left if self.left is not None else self.right
        return attachment.slot

    @property
    def slot_name(self) -> str:
        """A human-readable representation of the reported controller slot."""
        attachment = self.left if self.left is not None else self.right
        return attachment.slot_name

    @property
    def explanation(self) -> str:
        """A human-readable explanation of the difference."""
        if self.change_type == DiskChangeType.MOVED_SLOT:
            return (
                f"{_describe_medium(self.left, self.left_medium)} moved from {self.left.slot_name} "
                f"to {self.right.slot_name}"
            )
        if self.change_type == DiskChangeType.MEDIUM_ATTACHED:
            return f"{_describe_medium(self.right, self.right_medium)} attached at {self.slot_name}"
        if self.change_type == DiskChangeType.MEDIUM_DETACHED:
            return f"{_describe_medium(self.left, self.left_medium)} detached from {self.slot_name}"
        if self.change_type == DiskChangeType.PATH_CHANGED:
            return (
                f"same medium {self.left_medium.uuid} but its file path changed from "
                f"{self.left_medium.location!r} to {self.right_medium.location!r} at {self.slot_name}"
            )
        if self.change_type == DiskChangeType.LAYER_ADDED:
            return (
                f"{_describe_medium(self.left, self.left_medium)} extended with "
                f"{self.notes or 'a new differencing layer'}: now {_describe_medium(self.right, self.right_medium)} "
                f"at {self.slot_name}"
            )
        if self.change_type == DiskChangeType.LAYER_REMOVED:
            return (
                f"{_describe_medium(self.left, self.left_medium)} reverted by "
                f"{self.notes or 'removing differencing layer(s)'}: now "
                f"{_describe_medium(self.right, self.right_medium)} at {self.slot_name}"
            )
        if self.change_type == DiskChangeType.DISK_SWAPPED:
            suffix = f"; shared ancestor {self.common_ancestor.uuid}" if self.common_ancestor else ""
            return (
                f"disk replaced at {self.slot_name}: "
                f"{_describe_medium(self.left, self.left_medium)} -> "
                f"{_describe_medium(self.right, self.right_medium)}{suffix}"
            )
        return (
            f"relationship cannot be determined at {self.slot_name}: "
            f"{_describe_medium(self.left, self.left_medium)} vs "
            f"{_describe_medium(self.right, self.right_medium)} ({self.notes or 'a parent medium is missing'})"
        )


def diff_disk_attachments(
    left: Hardware | Snapshot | Machine,
    right: Hardware | Snapshot | Machine,
) -> list[DiskChange]:
    """Compare the hard disk attachments of two hardware states.

    The states may be two snapshots, a snapshot and the current machine, or two hardware states from
    different descriptors. Comparison is based on controller slots and on the media UUID parent
    chains recorded in the media registries; file contents are never read.

    Args:
        left: The left hardware state, snapshot, or machine.
        right: The right hardware state, snapshot, or machine.

    Returns:
        The ordered list of classified disk differences.
    """
    left_hardware = _as_hardware(left)
    right_hardware = _as_hardware(right)

    left_by_slot = {attachment.slot: attachment for attachment in left_hardware.attachments}
    right_by_slot = {attachment.slot: attachment for attachment in right_hardware.attachments}

    changes: list[DiskChange] = []
    moved_right_uuids: set[UUID] = set()

    ordered_slots = [attachment.slot for attachment in left_hardware.attachments]
    ordered_slots.extend(
        slot for slot in (attachment.slot for attachment in right_hardware.attachments) if slot not in left_by_slot
    )

    for slot in ordered_slots:
        left_attachment = left_by_slot.get(slot)
        right_attachment = right_by_slot.get(slot)

        if left_attachment is not None and right_attachment is not None:
            change = _classify_shared_slot(left_attachment, right_attachment)
            if change is not None:
                changes.append(change)
        elif left_attachment is not None:
            moved_to = _find_moved_attachment(right_hardware, left_attachment)
            if moved_to is not None:
                moved_right_uuids.add(moved_to.image_uuid)
                changes.append(
                    DiskChange(
                        DiskChangeType.MOVED_SLOT,
                        left_attachment,
                        moved_to,
                        left_attachment.medium,
                        moved_to.medium,
                        notes=_move_notes(left_attachment, moved_to),
                    )
                )
            else:
                changes.append(
                    DiskChange(
                        DiskChangeType.MEDIUM_DETACHED,
                        left_attachment,
                        None,
                        left_attachment.medium,
                        None,
                        notes=_unresolved_note(left_attachment),
                    )
                )
        elif right_attachment is not None and right_attachment.image_uuid not in moved_right_uuids:
            changes.append(
                DiskChange(
                    DiskChangeType.MEDIUM_ATTACHED,
                    None,
                    right_attachment,
                    None,
                    right_attachment.medium,
                    notes=_unresolved_note(right_attachment),
                )
            )

    return changes


def explain_disk_attachments(
    left: Hardware | Snapshot | Machine,
    right: Hardware | Snapshot | Machine,
) -> str:
    """Render the disk differences between two states as a human-readable report."""
    changes = diff_disk_attachments(left, right)
    if not changes:
        return "No disk attachment differences."

    lines = [f"{index}. [{change.change_type.value}] {change.explanation}" for index, change in enumerate(changes, 1)]
    return "\n".join(lines)


def _as_hardware(state: Hardware | Snapshot | Machine) -> Hardware:
    if isinstance(state, Hardware):
        return state
    return state.hardware


def _attachment_with_uuid(hardware: Hardware, image_uuid: UUID | None) -> Attachment | None:
    if image_uuid is None:
        return None
    for attachment in hardware.attachments:
        if attachment.image_uuid == image_uuid:
            return attachment
    return None


def _find_moved_attachment(hardware: Hardware, left: Attachment) -> Attachment | None:
    """Find the right-side slot a left attachment moved to.

    A move matches on the exact medium UUID first. When the medium instead grew or dropped a
    differencing layer while being relocated, the match falls back to a shared media UUID chain.
    Unresolved media (missing from a registry) never match by file path.
    """
    exact = _attachment_with_uuid(hardware, left.image_uuid)
    if exact is not None:
        return exact

    left_medium = left.medium
    if left_medium is None:
        return None

    for attachment in hardware.attachments:
        right_medium = attachment.medium
        if right_medium is None:
            continue
        if left_medium.is_ancestor_of(right_medium) or right_medium.is_ancestor_of(left_medium):
            return attachment
    return None


def _move_notes(left: Attachment, right: Attachment) -> str | None:
    left_medium = left.medium
    right_medium = right.medium
    if left_medium is None or right_medium is None or left.image_uuid == right.image_uuid:
        return None
    if left_medium.is_ancestor_of(right_medium):
        return f"moved and extended with {len(right_medium.chain) - len(left_medium.chain)} differencing layer(s)"
    if right_medium.is_ancestor_of(left_medium):
        return f"moved and reverted by {len(left_medium.chain) - len(right_medium.chain)} differencing layer(s)"
    return None


def _unresolved_note(attachment: Attachment) -> str | None:
    if attachment.image_uuid is not None and attachment.medium is None:
        return f"medium {attachment.image_uuid} is not present in the media registry"
    return None


def _classify_shared_slot(left: Attachment, right: Attachment) -> DiskChange | None:
    left_medium = left.medium
    right_medium = right.medium

    if left.image_uuid is None and right.image_uuid is None:
        return None
    if left.image_uuid is None:
        return DiskChange(DiskChangeType.MEDIUM_ATTACHED, left, right, None, right_medium)
    if right.image_uuid is None:
        return DiskChange(DiskChangeType.MEDIUM_DETACHED, left, right, left_medium, None)
    if left.image_uuid == right.image_uuid:
        if left_medium is not None and right_medium is not None and left_medium.location != right_medium.location:
            return DiskChange(DiskChangeType.PATH_CHANGED, left, right, left_medium, right_medium)
        return None
    if left_medium is None or right_medium is None:
        return DiskChange(
            DiskChangeType.UNKNOWN_RELATIONSHIP,
            left,
            right,
            left_medium,
            right_medium,
            notes="one side references a medium missing from its media registry",
        )

    return _classify_media_chain(left, right, left_medium, right_medium)


def _classify_media_chain(
    left: Attachment,
    right: Attachment,
    left_medium: HardDisk,
    right_medium: HardDisk,
) -> DiskChange:
    left_chain = left_medium.chain
    right_chain = right_medium.chain
    left_uuids = [disk.uuid for disk in left_chain]
    right_uuids = [disk.uuid for disk in right_chain]

    if left_medium.uuid in right_uuids:
        layers = len(right_chain) - len(left_chain)
        return DiskChange(
            DiskChangeType.LAYER_ADDED,
            left,
            right,
            left_medium,
            right_medium,
            common_ancestor=left_medium,
            notes=f"{layers} differencing layer(s) added on top of {left_medium.uuid}",
        )
    if right_medium.uuid in left_uuids:
        layers = len(left_chain) - len(right_chain)
        return DiskChange(
            DiskChangeType.LAYER_REMOVED,
            left,
            right,
            left_medium,
            right_medium,
            common_ancestor=right_medium,
            notes=f"{layers} differencing layer(s) removed back to {right_medium.uuid}",
        )

    shared = [uuid for uuid in right_uuids if uuid in frozenset(left_uuids)]
    common_ancestor = left_medium.vbox.media.get(shared[-1]) if shared else None

    notes = f"forked chains share base medium {shared[-1]}" if shared else "chains share no common medium"
    return DiskChange(
        DiskChangeType.DISK_SWAPPED,
        left,
        right,
        left_medium,
        right_medium,
        common_ancestor=common_ancestor,
        notes=notes,
    )


def _describe_medium(attachment: Attachment | None, medium: HardDisk | None) -> str:
    if attachment is None:
        return "no attachment"
    if attachment.image_uuid is None:
        return "empty slot"
    if medium is None:
        return f"unresolved medium {attachment.image_uuid} (not in media registry)"
    return f"medium {medium.uuid} ({medium.location})"
