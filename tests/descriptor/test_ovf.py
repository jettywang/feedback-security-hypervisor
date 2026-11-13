from __future__ import annotations

from io import StringIO

from dissect.hypervisor.descriptor.ovf import OVF

# Handcrafted to include some test cases
TEST_OVF = """<?xml version="1.0"?>
<Envelope ovf:version="1.0" xml:lang="en-US" xmlns="http://schemas.dmtf.org/ovf/envelope/1" xmlns:ovf="http://schemas.dmtf.org/ovf/envelope/1" xmlns:rasd="http://schemas.dmtf.org/wbem/wscim/1/cim-schema/2/CIM_ResourceAllocationSettingData" xmlns:vssd="http://schemas.dmtf.org/wbem/wscim/1/cim-schema/2/CIM_VirtualSystemSettingData" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:vbox="http://www.virtualbox.org/ovf/machine">
  <References>
    <File ovf:id="file1" ovf:href="disk1.vmdk"/>
    <File ovf:id="file2" ovf:href="disk2.vmdk"/>
    <File ovf:id="file3" ovf:href="disk3.vmdk"/>
  </References>
  <DiskSection>
    <Info>List of the virtual disks used in the package</Info>
    <Disk ovf:capacity="1024" ovf:diskId="vmdisk1" ovf:fileRef="file1" ovf:format="http://www.vmware.com/interfaces/specifications/vmdk.html#streamOptimized" vbox:uuid="d3f38cfc-d512-425c-9ca8-5c046492a9a8"/>
    <Disk ovf:capacity="1024" ovf:diskId="vmdisk2" ovf:fileRef="file2" ovf:format="http://www.vmware.com/interfaces/specifications/vmdk.html#streamOptimized" vbox:uuid="663d3136-5211-4d29-aacf-b4ef38be5fa2"/>
  </DiskSection>
  <NetworkSection>
    <Info>Logical networks used in the package</Info>
    <Network ovf:name="NAT">
      <Description>Logical network used by this appliance.</Description>
    </Network>
  </NetworkSection>
  <VirtualSystem ovf:id="Test OVF">
    <Info>A virtual machine</Info>
    <OperatingSystemSection ovf:id="100">
      <Info>The kind of installed guest operating system</Info>
      <Description>Linux26_64</Description>
      <vbox:OSType ovf:required="false">Linux26_64</vbox:OSType>
    </OperatingSystemSection>
    <VirtualHardwareSection>
      <Info>Virtual hardware requirements for a virtual machine</Info>
      <System>
        <vssd:ElementName>Virtual Hardware Family</vssd:ElementName>
        <vssd:InstanceID>0</vssd:InstanceID>
        <vssd:VirtualSystemIdentifier>Test OVF</vssd:VirtualSystemIdentifier>
        <vssd:VirtualSystemType>virtualbox-2.2</vssd:VirtualSystemType>
      </System>
      <Item>
        <rasd:Caption>1 virtual CPU</rasd:Caption>
        <rasd:Description>Number of virtual CPUs</rasd:Description>
        <rasd:ElementName>1 virtual CPU</rasd:ElementName>
        <rasd:InstanceID>1</rasd:InstanceID>
        <rasd:ResourceType>3</rasd:ResourceType>
        <rasd:VirtualQuantity>1</rasd:VirtualQuantity>
      </Item>
      <Item>
        <rasd:AllocationUnits>MegaBytes</rasd:AllocationUnits>
        <rasd:Caption>1536 MB of memory</rasd:Caption>
        <rasd:Description>Memory Size</rasd:Description>
        <rasd:ElementName>1536 MB of memory</rasd:ElementName>
        <rasd:InstanceID>2</rasd:InstanceID>
        <rasd:ResourceType>4</rasd:ResourceType>
        <rasd:VirtualQuantity>1536</rasd:VirtualQuantity>
      </Item>
      <Item>
        <rasd:Address>0</rasd:Address>
        <rasd:Caption>sataController0</rasd:Caption>
        <rasd:Description>SATA Controller</rasd:Description>
        <rasd:ElementName>sataController0</rasd:ElementName>
        <rasd:InstanceID>3</rasd:InstanceID>
        <rasd:ResourceSubType>AHCI</rasd:ResourceSubType>
        <rasd:ResourceType>20</rasd:ResourceType>
      </Item>
      <Item>
        <rasd:Address>0</rasd:Address>
        <rasd:Caption>usb</rasd:Caption>
        <rasd:Description>USB Controller</rasd:Description>
        <rasd:ElementName>usb</rasd:ElementName>
        <rasd:InstanceID>4</rasd:InstanceID>
        <rasd:ResourceType>23</rasd:ResourceType>
      </Item>
      <Item>
        <rasd:AddressOnParent>3</rasd:AddressOnParent>
        <rasd:AutomaticAllocation>false</rasd:AutomaticAllocation>
        <rasd:Caption>sound</rasd:Caption>
        <rasd:Description>Sound Card</rasd:Description>
        <rasd:ElementName>sound</rasd:ElementName>
        <rasd:InstanceID>5</rasd:InstanceID>
        <rasd:ResourceSubType>ensoniq1371</rasd:ResourceSubType>
        <rasd:ResourceType>35</rasd:ResourceType>
      </Item>
      <Item>
        <rasd:AddressOnParent>0</rasd:AddressOnParent>
        <rasd:Caption>disk1</rasd:Caption>
        <rasd:Description>Disk Image</rasd:Description>
        <rasd:ElementName>disk1</rasd:ElementName>
        <rasd:HostResource>/disk/vmdisk1</rasd:HostResource>
        <rasd:InstanceID>6</rasd:InstanceID>
        <rasd:Parent>3</rasd:Parent>
        <rasd:ResourceType>17</rasd:ResourceType>
      </Item>
      <Item>
        <rasd:AddressOnParent>1</rasd:AddressOnParent>
        <rasd:Caption>disk2</rasd:Caption>
        <rasd:Description>Disk Image</rasd:Description>
        <rasd:ElementName>disk2</rasd:ElementName>
        <rasd:HostResource>ovf:/disk/vmdisk2</rasd:HostResource>
        <rasd:InstanceID>8</rasd:InstanceID>
        <rasd:Parent>3</rasd:Parent>
        <rasd:ResourceType>17</rasd:ResourceType>
      </Item>
      <Item>
        <rasd:AddressOnParent>2</rasd:AddressOnParent>
        <rasd:Caption>disk3</rasd:Caption>
        <rasd:Description>Disk Image</rasd:Description>
        <rasd:ElementName>disk3</rasd:ElementName>
        <rasd:HostResource>ovf:/file/file3</rasd:HostResource>
        <rasd:InstanceID>9</rasd:InstanceID>
        <rasd:Parent>3</rasd:Parent>
        <rasd:ResourceType>17</rasd:ResourceType>
      </Item>
      <Item>
        <rasd:AutomaticAllocation>true</rasd:AutomaticAllocation>
        <rasd:Caption>Ethernet adapter on 'NAT'</rasd:Caption>
        <rasd:Connection>NAT</rasd:Connection>
        <rasd:ElementName>Ethernet adapter on 'NAT'</rasd:ElementName>
        <rasd:InstanceID>7</rasd:InstanceID>
        <rasd:ResourceType>10</rasd:ResourceType>
      </Item>
    </VirtualHardwareSection>
    <vbox:Machine ovf:required="false" version="1.19-linux" uuid="{68383d3a-9a23-4789-a580-4f0a79437fd6}" name="Test OVF" OSType="Linux26_64" snapshotFolder="Snapshots" lastStateChange="2023-09-20T12:15:55Z">
      <ovf:Info>Complete VirtualBox machine configuration in VirtualBox format</ovf:Info>
      <Hardware>
        <CPU>
          <PAE enabled="true"/>
          <LongMode enabled="true"/>
          <X2APIC enabled="true"/>
          <HardwareVirtExLargePages enabled="true"/>
        </CPU>
        <Memory RAMSize="1536"/>
        <HID Pointing="USBTablet"/>
        <Display controller="VMSVGA" VRAMSize="32"/>
        <BIOS>
          <IOAPIC enabled="true"/>
          <SmbiosUuidLittleEndian enabled="true"/>
        </BIOS>
        <USB>
          <Controllers>
            <Controller name="OHCI" type="OHCI"/>
            <Controller name="EHCI" type="EHCI"/>
          </Controllers>
        </USB>
        <Network>
          <Adapter slot="0" enabled="true" MACAddress="FFFFFFFFFFFF" type="virtio">
            <NAT/>
          </Adapter>
        </Network>
        <AudioAdapter driver="ALSA" enabled="true" enabledOut="true"/>
        <RTC localOrUTC="UTC"/>
        <Clipboard/>
        <StorageControllers>
          <StorageController name="SATA" type="AHCI" PortCount="4" useHostIOCache="true" Bootable="true" IDE0MasterEmulationPort="0" IDE0SlaveEmulationPort="1" IDE1MasterEmulationPort="2" IDE1SlaveEmulationPort="3">
            <AttachedDevice type="HardDisk" hotpluggable="false" port="0" device="0">
              <Image uuid="{d3f38cfc-d512-425c-9ca8-5c046492a9a8}"/>
            </AttachedDevice>
          </StorageController>
        </StorageControllers>
      </Hardware>
    </vbox:Machine>
  </VirtualSystem>
</Envelope>
"""  # noqa


def test_ovf() -> None:
    ovf = OVF(StringIO(TEST_OVF))
    assert list(ovf.disks()) == ["disk1.vmdk", "disk2.vmdk", "disk3.vmdk"]


# Handcrafted VirtualSystemCollection with two systems sharing a disk,
# plus edge cases: direct file reference, empty disk, duplicate and missing ids.
TEST_OVF_COLLECTION = """<?xml version="1.0"?>
<Envelope ovf:version="1.0" xml:lang="en-US" xmlns="http://schemas.dmtf.org/ovf/envelope/1" xmlns:ovf="http://schemas.dmtf.org/ovf/envelope/1" xmlns:rasd="http://schemas.dmtf.org/wbem/wscim/1/cim-schema/2/CIM_ResourceAllocationSettingData">
  <References>
    <File ovf:id="file1" ovf:href="vm1-boot.vmdk"/>
    <File ovf:id="file2" ovf:href="shared-data.vmdk"/>
    <File ovf:id="file3" ovf:href="vm2-boot.vmdk"/>
    <File ovf:id="file4" ovf:href="tools.iso"/>
    <File ovf:id="file5" ovf:href="https://example.com/remote.vmdk"/>
    <File ovf:id="filedup" ovf:href="dup-a.vmdk"/>
    <File ovf:id="filedup" ovf:href="dup-b.vmdk"/>
    <File ovf:href="orphan.vmdk"/>
  </References>
  <DiskSection>
    <Info>List of the virtual disks used in the package</Info>
    <Disk ovf:capacity="1024" ovf:diskId="vmdisk1" ovf:fileRef="file1"/>
    <Disk ovf:capacity="1024" ovf:diskId="vmdisk2" ovf:fileRef="file2"/>
    <Disk ovf:capacity="1024" ovf:diskId="vmdisk3" ovf:fileRef="file3"/>
    <Disk ovf:capacity="1024" ovf:diskId="vmdisk4"/>
    <Disk ovf:capacity="1024" ovf:diskId="vmdiskdup" ovf:fileRef="filedup"/>
    <Disk ovf:capacity="1024" ovf:diskId="vmdiskdup" ovf:fileRef="file1"/>
    <Disk ovf:capacity="1024" ovf:diskId="vmdisk-missing" ovf:fileRef="file-missing"/>
  </DiskSection>
  <VirtualSystemCollection ovf:id="collection">
    <Info>Two appliances sharing a data disk</Info>
    <VirtualSystem ovf:id="vm1">
      <Info>First virtual machine</Info>
      <VirtualHardwareSection>
        <Info>Virtual hardware requirements for a virtual machine</Info>
        <Item>
          <rasd:Caption>scsiController0</rasd:Caption>
          <rasd:Description>SCSI Controller</rasd:Description>
          <rasd:ElementName>scsiController0</rasd:ElementName>
          <rasd:InstanceID>3</rasd:InstanceID>
          <rasd:ResourceType>20</rasd:ResourceType>
        </Item>
        <Item>
          <rasd:AddressOnParent>0</rasd:AddressOnParent>
          <rasd:Caption>disk1</rasd:Caption>
          <rasd:HostResource>ovf:/disk/vmdisk1</rasd:HostResource>
          <rasd:InstanceID>4</rasd:InstanceID>
          <rasd:Parent>3</rasd:Parent>
          <rasd:ResourceType>17</rasd:ResourceType>
        </Item>
        <Item>
          <rasd:AddressOnParent>1</rasd:AddressOnParent>
          <rasd:Caption>disk2</rasd:Caption>
          <rasd:HostResource>ovf:/disk/vmdisk2</rasd:HostResource>
          <rasd:InstanceID>5</rasd:InstanceID>
          <rasd:Parent>3</rasd:Parent>
          <rasd:ResourceType>17</rasd:ResourceType>
        </Item>
        <Item>
          <rasd:AddressOnParent>2</rasd:AddressOnParent>
          <rasd:Caption>disk3</rasd:Caption>
          <rasd:HostResource>ovf:/disk/vmdisk4</rasd:HostResource>
          <rasd:InstanceID>6</rasd:InstanceID>
          <rasd:Parent>3</rasd:Parent>
          <rasd:ResourceType>17</rasd:ResourceType>
        </Item>
        <Item>
          <rasd:AddressOnParent>3</rasd:AddressOnParent>
          <rasd:Caption>disk4</rasd:Caption>
          <rasd:HostResource>ovf:/disk/vmdiskdup</rasd:HostResource>
          <rasd:InstanceID>7</rasd:InstanceID>
          <rasd:Parent>3</rasd:Parent>
          <rasd:ResourceType>17</rasd:ResourceType>
        </Item>
        <Item>
          <rasd:AddressOnParent>4</rasd:AddressOnParent>
          <rasd:Caption>cdrom1</rasd:Caption>
          <rasd:HostResource>ovf:/file/file4</rasd:HostResource>
          <rasd:InstanceID>8</rasd:InstanceID>
          <rasd:Parent>3</rasd:Parent>
          <rasd:ResourceType>17</rasd:ResourceType>
        </Item>
      </VirtualHardwareSection>
    </VirtualSystem>
    <VirtualSystem ovf:id="vm2">
      <Info>Second virtual machine</Info>
      <VirtualHardwareSection>
        <Info>Virtual hardware requirements for a virtual machine</Info>
        <Item>
          <rasd:Caption>sataController0</rasd:Caption>
          <rasd:Description>SATA Controller</rasd:Description>
          <rasd:ElementName>sataController0</rasd:ElementName>
          <rasd:InstanceID>3</rasd:InstanceID>
          <rasd:ResourceType>20</rasd:ResourceType>
        </Item>
        <Item>
          <rasd:AddressOnParent>0</rasd:AddressOnParent>
          <rasd:Caption>disk1</rasd:Caption>
          <rasd:HostResource>ovf:/disk/vmdisk3</rasd:HostResource>
          <rasd:InstanceID>4</rasd:InstanceID>
          <rasd:Parent>3</rasd:Parent>
          <rasd:ResourceType>17</rasd:ResourceType>
        </Item>
        <Item>
          <rasd:AddressOnParent>1</rasd:AddressOnParent>
          <rasd:Caption>disk2</rasd:Caption>
          <rasd:HostResource>ovf:/disk/vmdisk2</rasd:HostResource>
          <rasd:InstanceID>5</rasd:InstanceID>
          <rasd:Parent>3</rasd:Parent>
          <rasd:ResourceType>17</rasd:ResourceType>
        </Item>
        <Item>
          <rasd:AddressOnParent>2</rasd:AddressOnParent>
          <rasd:Caption>disk3</rasd:Caption>
          <rasd:HostResource>ovf:/disk/vmdisk-missing</rasd:HostResource>
          <rasd:InstanceID>6</rasd:InstanceID>
          <rasd:Parent>3</rasd:Parent>
          <rasd:ResourceType>17</rasd:ResourceType>
        </Item>
        <Item>
          <rasd:AddressOnParent>3</rasd:AddressOnParent>
          <rasd:Caption>disk4</rasd:Caption>
          <rasd:HostResource>ovf:/file/file5</rasd:HostResource>
          <rasd:InstanceID>7</rasd:InstanceID>
          <rasd:Parent>3</rasd:Parent>
          <rasd:ResourceType>17</rasd:ResourceType>
        </Item>
      </VirtualHardwareSection>
    </VirtualSystem>
  </VirtualSystemCollection>
</Envelope>
"""  # noqa


def test_ovf_virtual_system_collection() -> None:
    ovf = OVF(StringIO(TEST_OVF_COLLECTION))
    systems = dict(ovf.systems())
    assert list(systems) == ["vm1", "vm2"]

    vm1 = systems["vm1"]
    assert [(m.controller, m.address, m.disk_id, m.file_id, m.href) for m in vm1] == [
        ("scsiController0", "0", "vmdisk1", "file1", "vm1-boot.vmdk"),
        ("scsiController0", "1", "vmdisk2", "file2", "shared-data.vmdk"),
        # Empty disk without a backing file.
        ("scsiController0", "2", "vmdisk4", None, None),
        # Ambiguous: "vmdiskdup" is defined twice in the DiskSection.
        ("scsiController0", "3", "vmdiskdup", None, None),
        # Direct file reference, no DiskSection entry involved.
        ("scsiController0", "4", None, "file4", "tools.iso"),
    ]
    assert vm1[2].error is None
    assert "ambiguous disk id 'vmdiskdup'" in vm1[3].error

    vm2 = systems["vm2"]
    assert [(m.controller, m.address, m.disk_id, m.file_id, m.href) for m in vm2] == [
        ("sataController0", "0", "vmdisk3", "file3", "vm2-boot.vmdk"),
        # The shared disk maps to the same evidence file as in vm1.
        ("sataController0", "1", "vmdisk2", "file2", "shared-data.vmdk"),
        # The referenced file id "file-missing" is not defined.
        ("sataController0", "2", "vmdisk-missing", "file-missing", None),
        # External references are reported but never dereferenced.
        ("sataController0", "3", None, "file5", "https://example.com/remote.vmdk"),
    ]
    assert "unknown file id 'file-missing'" in vm2[2].error
    assert vm2[3].error is None

    # Duplicate and anonymous ids are reported on the OVF level.
    assert ovf.duplicate_file_ids == {"filedup"}
    assert ovf.duplicate_disk_ids == {"vmdiskdup"}
    assert ovf.anonymous_files == ["orphan.vmdk"]

    # The shared disk shows up once per system, empty and unresolvable disks are skipped.
    assert list(ovf.disks()) == [
        "vm1-boot.vmdk",
        "shared-data.vmdk",
        "tools.iso",
        "vm2-boot.vmdk",
        "shared-data.vmdk",
        "https://example.com/remote.vmdk",
    ]
