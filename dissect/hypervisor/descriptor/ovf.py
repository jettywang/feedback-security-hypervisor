from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, TextIO

from defusedxml import ElementTree

if TYPE_CHECKING:
    from collections.abc import Iterator
    from xml.etree.ElementTree import Element


@dataclass
class DiskMapping:
    """Relation between a disk drive of a ``VirtualSystem`` and its backing evidence file.

    Attributes:
        system: ``ovf:id`` of the owning ``VirtualSystem``.
        instance_id: ``rasd:InstanceID`` of the disk drive item.
        controller: Name of the parent storage controller the drive is attached to.
        address: ``rasd:AddressOnParent``, i.e. the position on the controller.
        host_resource: Raw ``rasd:HostResource`` value of the disk drive item.
        disk_id: Referenced ``ovf:diskId`` from the ``DiskSection``, if any.
        file_id: Referenced ``ovf:File`` id from the ``References`` section, if any.
        href: Final evidence file. ``None`` for empty disks without a backing file
            and for references that could not be resolved. Hrefs may point to external
            locations (e.g. URLs); they are returned as-is and never opened.
        error: Human readable reason why ``href`` could not be determined, if any.
    """

    system: str | None = None
    instance_id: str | None = None
    controller: str | None = None
    address: str | None = None
    host_resource: str | None = None
    disk_id: str | None = None
    file_id: str | None = None
    href: str | None = None
    error: str | None = None


class OVF:
    NS: Final[dict[str, str]] = {
        "ovf": "http://schemas.dmtf.org/ovf/envelope/1",
        "rasd": "http://schemas.dmtf.org/wbem/wscim/1/cim-schema/2/CIM_ResourceAllocationSettingData",
    }

    FILE_XPATH = "ovf:References/ovf:File"
    DISK_XPATH = "ovf:DiskSection/ovf:Disk"
    SYSTEM_XPATH = ".//ovf:VirtualSystem"
    ITEM_XPATH = "ovf:VirtualHardwareSection/ovf:Item"
    DISK_DRIVE_XPATH = 'ovf:VirtualSystem/ovf:VirtualHardwareSection/ovf:Item/[rasd:ResourceType="17"]'

    RESOURCE_TYPE_DISK_DRIVE: Final = "17"
    RESOURCE_TYPE_CONTROLLER: Final = "20"

    def __init__(self, fh: TextIO):
        self.fh = fh
        self.xml: Element = ElementTree.fromstring(fh.read())

        self.references: dict[str, str] = {}
        self.duplicate_file_ids: set[str] = set()
        self.anonymous_files: list[str | None] = []
        for file_ in self.xml.findall(self.FILE_XPATH, self.NS):
            file_id = file_.get("{{{ovf}}}id".format(**self.NS))
            href = file_.get("{{{ovf}}}href".format(**self.NS))
            if file_id is None:
                self.anonymous_files.append(href)
            elif file_id in self.references:
                self.duplicate_file_ids.add(file_id)
            else:
                self.references[file_id] = href

        self._disks: dict[str, str | None] = {}
        self.duplicate_disk_ids: set[str] = set()
        self.anonymous_disks: list[str | None] = []
        for disk in self.xml.findall(self.DISK_XPATH, self.NS):
            disk_id = disk.get("{{{ovf}}}diskId".format(**self.NS))
            file_ref = disk.get("{{{ovf}}}fileRef".format(**self.NS))
            if disk_id is None:
                self.anonymous_disks.append(file_ref)
            elif disk_id in self._disks:
                self.duplicate_disk_ids.add(disk_id)
            else:
                self._disks[disk_id] = file_ref

    def _rasd_text(self, item: Element, tag: str) -> str | None:
        child = item.find("{{{rasd}}}{tag}".format(tag=tag, **self.NS))
        if child is not None and child.text:
            return child.text.strip()
        return None

    def _resolve_file(self, file_id: str) -> tuple[str | None, str | None]:
        if file_id in self.duplicate_file_ids:
            return None, f"ambiguous file id {file_id!r}: defined multiple times"
        if file_id not in self.references:
            return None, f"unknown file id {file_id!r}"
        return self.references[file_id], None

    def _resolve_disk(self, disk_id: str) -> tuple[str | None, str | None, str | None]:
        if disk_id in self.duplicate_disk_ids:
            return None, None, f"ambiguous disk id {disk_id!r}: defined multiple times"
        if disk_id not in self._disks:
            return None, None, f"unknown disk id {disk_id!r}"
        file_id = self._disks[disk_id]
        if file_id is None:
            # Empty disk without a backing file.
            return None, None, None
        href, error = self._resolve_file(file_id)
        return file_id, href, error

    def _map_item(self, system_id: str | None, controllers: dict[str, str], item: Element) -> DiskMapping:
        mapping = DiskMapping(
            system=system_id,
            instance_id=self._rasd_text(item, "InstanceID"),
            address=self._rasd_text(item, "AddressOnParent"),
        )
        parent = self._rasd_text(item, "Parent")
        mapping.controller = controllers.get(parent)

        host_resource = self._rasd_text(item, "HostResource")
        mapping.host_resource = host_resource
        if not host_resource:
            mapping.error = "missing HostResource"
            return mapping

        xpath = host_resource.removeprefix("ovf:")
        if xpath.startswith("/disk/"):
            mapping.disk_id = xpath.split("/")[-1]
            mapping.file_id, mapping.href, mapping.error = self._resolve_disk(mapping.disk_id)
        elif xpath.startswith("/file/"):
            mapping.file_id = xpath.split("/")[-1]
            mapping.href, mapping.error = self._resolve_file(mapping.file_id)
        elif "://" in host_resource:
            # External reference; record it but never dereference it.
            mapping.error = f"external reference {host_resource!r} not resolved"
        else:
            mapping.error = f"unsupported HostResource {host_resource!r}"
        return mapping

    def systems(self) -> Iterator[tuple[str | None, list[DiskMapping]]]:
        """Yield ``(system_id, disk_mappings)`` for every ``VirtualSystem``.

        ``VirtualSystem`` elements are found both directly under the ``Envelope``
        and nested inside a ``VirtualSystemCollection``.
        """
        for system in self.xml.findall(self.SYSTEM_XPATH, self.NS):
            system_id = system.get("{{{ovf}}}id".format(**self.NS))
            controllers = {}
            drives = []
            for item in system.findall(self.ITEM_XPATH, self.NS):
                resource_type = self._rasd_text(item, "ResourceType")
                if resource_type == self.RESOURCE_TYPE_CONTROLLER:
                    name = self._rasd_text(item, "Caption") or self._rasd_text(item, "ElementName")
                    controllers[self._rasd_text(item, "InstanceID")] = name
                elif resource_type == self.RESOURCE_TYPE_DISK_DRIVE:
                    drives.append(item)
            yield system_id, [self._map_item(system_id, controllers, item) for item in drives]

    def disks(self) -> Iterator[str]:
        for _, mappings in self.systems():
            for mapping in mappings:
                if mapping.href is not None:
                    yield mapping.href
