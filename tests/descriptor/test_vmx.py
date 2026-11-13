from __future__ import annotations

import base64
import hashlib
import hmac
from urllib.parse import quote

import pytest

try:
    from Crypto.Cipher import AES

    HAS_AES = True
except ImportError:
    HAS_AES = False

from dissect.hypervisor.descriptor.vmx import (
    HAS_PYCRYPTODOME,
    HAS_PYSTANDALONE,
    HMAC_MAP,
    VMX,
)
from tests._util import absolute_path


def test_vmx() -> None:
    data_scsi = """
scsi0.virtualDev = "lsisas1068"
scsi0.present = "TRUE"
scsi0:0.fileName = "Virtual Disk-cl1.vmdk"
scsi0:0.present = "TRUE"
scsi0:0.redo = ""
scsi0.pciSlotNumber = "160"
scsi0.sasWWID = "50 05 05 68 05 82 7f 70"
    """
    vmx_scsi = VMX.parse(data_scsi)

    assert len(vmx_scsi.attr) == 7
    assert vmx_scsi.disks() == ["Virtual Disk-cl1.vmdk"]

    data_nvme = """
nvme0.pcislotnumber = "160"
nvme0.present = "TRUE"
nvme0:0.filename = "Virtual Disk-000003.vmdk"
nvme0:0.present = "TRUE"
nvme0:0.redo = ""
    """
    vmx_nvme = VMX.parse(data_nvme)

    assert len(vmx_nvme.attr) == 5
    assert vmx_nvme.disks() == ["Virtual Disk-000003.vmdk"]

    data_ide = """
ide1:0.deviceType = "cdrom-image"
ide1:0.fileName = "file.iso"
ide1:0.present = "TRUE"
ide0:0.fileName = "Virtual Disk 2.vmdk"
ide0:0.present = "TRUE"
    """
    vmx_ide = VMX.parse(data_ide)

    assert len(vmx_ide.attr) == 5
    assert vmx_ide.disks() == ["Virtual Disk 2.vmdk"]

    data_sata = """
sata0.present = "TRUE"
sata0:0.fileName = "Virtual Disk 3.vmdk"
sata0:0.present = "TRUE"
    """
    vmx_sata = VMX.parse(data_sata)

    assert len(vmx_sata.attr) == 3
    assert vmx_sata.disks() == ["Virtual Disk 3.vmdk"]

    data_casing = """
sata0:0.FILENAME = "Virtual Disk 3.vmdk"
    """
    vmx_casing = VMX.parse(data_casing)

    assert len(vmx_casing.attr) == 1
    assert vmx_casing.disks() == ["Virtual Disk 3.vmdk"]


@pytest.mark.skipif((not HAS_PYCRYPTODOME and not HAS_PYSTANDALONE), reason="No crypto module available")
def test_vmx_encrypted() -> None:
    vmx = VMX.parse(absolute_path("_data/descriptor/vmx/encrypted.vmx").read_text())

    assert vmx.encrypted

    vmx.unlock_with_phrase("password")

    assert "datafilekey" in vmx.attr


PHRASE = "correct horse battery staple"
FINAL_KEY = bytes(range(32))
WRONG_KEY = bytes(range(32, 64))
CONFIG = "\n".join(
    [
        '.encoding = "UTF-8"',
        'displayName = "Synthetic encrypted VM"',
        'nvme0:0.filename = "disk-under-test.vmdk"',
        'nvme0:0.present = "TRUE"',
        'datafilekey = "type=key:cipher=AES-256:key=c2VjcmV0LWtleQ=="',
        'cryptoState = "encrypted"',
    ]
)
DISK = "disk-under-test.vmdk"
SECRET = "c2VjcmV0LWtleQ=="


def _seal(key: bytes, plaintext: bytes, iv: bytes, mac: str = "HMAC-SHA-1") -> bytes:
    digest = HMAC_MAP[mac][0]
    padding = 16 - len(plaintext) % 16
    encrypted = AES.new(key, AES.MODE_CBC, iv).encrypt(plaintext + bytes([padding]) * padding)
    return iv + encrypted + hmac.digest(key, plaintext, digest)


def _wrapping_key(passphrase: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha1", passphrase.encode(), salt, 1200, 32)


def _pair(
    passphrase: str,
    final_key: bytes,
    salt: bytes,
    phrase_id: str,
    *,
    corrupt_mac: bool = False,
) -> str:
    wrapped_key = f"type=key:cipher=AES-256:key={base64.b64encode(final_key).decode()}".encode()
    sealed = _seal(_wrapping_key(passphrase, salt), wrapped_key, b"\x01" * 16)
    if corrupt_mac:
        sealed = sealed[:-1] + bytes([sealed[-1] ^ 0xFF])

    crypto_dict = (
        f"pass2key=PBKDF2-HMAC-SHA-1:cipher=AES-256:rounds=1200:salt={base64.b64encode(salt).decode()}"
    )
    locator = f"phrase/{quote(phrase_id, safe='')}/{quote(crypto_dict, safe='')}"
    return f"pair/({locator},HMAC-SHA-1,{quote(base64.b64encode(sealed).decode(), safe='')})"


def _vmx_text(pairs: list[str], encrypted_config: bytes | None = None) -> str:
    payload = encrypted_config if encrypted_config is not None else _seal(FINAL_KEY, CONFIG.encode(), b"\x02" * 16)
    return "\n".join(
        [
            '.encoding = "UTF-8"',
            'displayName = "Synthetic encrypted VM"',
            f'encryption.keySafe = "vmware:key/list/({",".join(pairs)})"',
            f'encryption.data = "{base64.b64encode(payload).decode()}"',
        ]
    )


def _assert_locked(vmx: VMX, original: dict[str, str]) -> None:
    assert vmx.attr == original
    assert not vmx.unlocked
    assert vmx.disks() == []
    assert vmx.attr.get("datafilekey") is None


@pytest.mark.skipif(not HAS_AES, reason="PyCryptodome is required for the synthetic fixtures")
def test_vmx_unlock_failure_then_success_and_repeat() -> None:
    vmx = VMX.parse(_vmx_text([_pair(PHRASE, FINAL_KEY, b"test-salt-000000", "test")]))
    original = dict(vmx.attr)

    assert vmx.encrypted
    assert not vmx.unlocked
    assert vmx.disks() == []

    with pytest.raises(ValueError) as exc_info:
        vmx.unlock_with_phrase("wrong phrase")

    message = str(exc_info.value)
    assert PHRASE not in message
    assert "wrong phrase" not in message
    assert SECRET not in message
    assert "disk-under-test" not in message
    _assert_locked(vmx, original)

    vmx.unlock_with_phrase(PHRASE)

    assert vmx.unlocked
    assert vmx.disks() == [DISK]
    assert vmx.attr["datafilekey"] == f"type=key:cipher=AES-256:key={SECRET}"

    unlocked = dict(vmx.attr)
    vmx.unlock_with_phrase(PHRASE)

    assert vmx.attr == unlocked
    assert vmx.unlocked
    assert vmx.disks() == [DISK]

    with pytest.raises(ValueError):
        vmx.unlock_with_phrase("another wrong phrase")

    assert vmx.attr == unlocked
    assert vmx.unlocked
    assert vmx.disks() == [DISK]


@pytest.mark.skipif(not HAS_AES, reason="PyCryptodome is required for the synthetic fixtures")
def test_vmx_unlock_multiple_candidates() -> None:
    # The first candidate unwraps to a valid-looking key, but the second one has
    # a damaged wrapping MAC. Neither may abort the scan before the good pair.
    wrong_key = _pair(PHRASE, WRONG_KEY, b"bad-salt-0000000", "bad-key")
    corrupt_mac = _pair(PHRASE, FINAL_KEY, b"corrupt-salt-000", "corrupt", corrupt_mac=True)
    good = _pair(PHRASE, FINAL_KEY, b"good-salt-000000", "good")

    vmx = VMX.parse(_vmx_text([wrong_key, corrupt_mac, good]))
    original = dict(vmx.attr)

    with pytest.raises(ValueError):
        vmx.unlock_with_phrase("wrong phrase")
    _assert_locked(vmx, original)

    vmx.unlock_with_phrase(PHRASE)

    assert vmx.unlocked
    assert vmx.disks() == [DISK]


@pytest.mark.skipif(not HAS_AES, reason="PyCryptodome is required for the synthetic fixtures")
def test_vmx_unlock_rejects_corrupt_payload() -> None:
    corrupt_payload = _seal(FINAL_KEY, CONFIG.encode(), b"\x03" * 16)[:-1] + b"\xff"
    vmx = VMX.parse(_vmx_text([_pair(PHRASE, FINAL_KEY, b"test-salt-000000", "test")], corrupt_payload))
    original = dict(vmx.attr)

    with pytest.raises(ValueError) as exc_info:
        vmx.unlock_with_phrase(PHRASE)

    message = str(exc_info.value)
    assert PHRASE not in message
    assert SECRET not in message
    _assert_locked(vmx, original)


@pytest.mark.skipif(not HAS_AES, reason="PyCryptodome is required for the synthetic fixtures")
def test_vmx_unlock_rejects_unparseable_configuration() -> None:
    # Authenticated but not a configuration: it must not be published, and a
    # valid later candidate must still be able to unlock the VMX.
    garbage = b"this is not a VMX line\nand this line has no separator either"
    decoy = _pair(PHRASE, WRONG_KEY, b"decoy-salt-000000", "decoy")
    good = _pair(PHRASE, FINAL_KEY, b"good-salt-000000", "good")

    vmx = VMX.parse(_vmx_text([decoy], _seal(WRONG_KEY, garbage, b"\x04" * 16)))
    original = dict(vmx.attr)

    with pytest.raises(ValueError) as exc_info:
        vmx.unlock_with_phrase(PHRASE)
    _assert_locked(vmx, original)
    assert "this is not a VMX line" not in str(exc_info.value)

    vmx = VMX.parse(_vmx_text([decoy, good], _seal(FINAL_KEY, CONFIG.encode(), b"\x05" * 16)))
    vmx.unlock_with_phrase(PHRASE)

    assert vmx.unlocked
    assert vmx.disks() == [DISK]
