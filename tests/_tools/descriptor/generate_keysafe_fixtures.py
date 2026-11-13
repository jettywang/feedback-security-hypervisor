"""Deterministically generate KeySafe contract-test fixtures.

This generator is intentionally independent of
``dissect.hypervisor.descriptor.vmx``: it never imports the parser under test,
so it cannot regenerate the expected answers from the implementation. Every
passphrase, salt, IV, wrapped key and expected encryption key is hard-coded in
this file as the authoritative sample specification. Only the sealing math
(PBKDF2, AES-CBC + PKCS#7, HMAC) and the percent-encoding are performed here.

All key material is test-only and derived from fixed byte patterns. The script
never reads any real virtual machine or user credential.

Run from the repository root:

    .venv/bin/python tests/_tools/descriptor/generate_keysafe_fixtures.py
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from pathlib import Path
from urllib.parse import quote

from Crypto.Cipher import AES

OUT_PATH = Path(__file__).resolve().parents[2] / "_data" / "descriptor" / "vmx" / "keysafe_contract_fixtures.json"

# --- Authoritative, hand-specified test material (NOT parser derived) -------
PHRASE_ID = "id,with/slash(and)paren"
PASSPHRASE = "correct horse battery staple"
WRONG_PASSPHRASE = "correct horse battery staple!"
PASS2KEY = "PBKDF2-HMAC-SHA-1"
CIPHER = "AES-256"
ROUNDS = 1000
HMAC_TYPE = "HMAC-SHA-1"
HMAC_SIZE = 20
SALT = bytes(range(1, 17))
IV = bytes(reversed(range(16)))
# The final VM encryption key the KeySafe is expected to unseal to.
EXPECTED_KEY = bytes((index * 7 + 3) % 256 for index in range(32))
# A second wrapped key living in a separate pair (multi-locator case).
SECOND_PASSPHRASE = "second test passphrase"
SECOND_ID = "second-id"
SECOND_SALT = bytes(range(33, 49))
SECOND_IV = bytes((index + 5) % 256 for index in range(16))
SECOND_EXPECTED_KEY = bytes((index * 11 + 1) % 256 for index in range(32))


def b64(value: bytes) -> str:
    return base64.b64encode(value).decode()


def encode_crypto_dict(fields: dict[str, str]) -> str:
    # VMware double-encodes values embedded in a crypto dict so the inner '='
    # of base64 survives the outer crypto-dict split. A single full percent
    # encoding has the same effect; we encode every field value here.
    return quote(":".join(f"{key}={quote(value, safe='')}" for key, value in fields.items()), safe="")


def build_phrase(phrase_id: str, salt: bytes, *, rounds: int = ROUNDS) -> str:
    crypto_dict = encode_crypto_dict(
        {
            "pass2key": PASS2KEY,
            "cipher": CIPHER,
            "rounds": str(rounds),
            "salt": b64(salt),
        }
    )
    return f"phrase/{quote(phrase_id, safe='')}/{crypto_dict}"


def derive_key(passphrase: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha1", passphrase.encode(), salt, ROUNDS, 32)


def seal(payload: bytes, key: bytes, iv: bytes) -> bytes:
    """Reproduce the parser's wire format: IV || AES-CBC(PKCS#7) || HMAC."""
    pad_length = 16 - (len(payload) % 16)
    padded = payload + bytes([pad_length]) * pad_length
    ciphertext = AES.new(key, AES.MODE_CBC, iv=iv).encrypt(padded)
    digest = hmac.digest(key, payload, "sha1")
    return iv + ciphertext + digest


def build_pair(passphrase: str, phrase_id: str, salt: bytes, iv: bytes, expected_key: bytes) -> str:
    wrapping_key = derive_key(passphrase, salt)
    payload = f"type=key:cipher=AES-256:key={b64(expected_key)}".encode()
    sealed = seal(payload, wrapping_key, iv)
    phrase = build_phrase(phrase_id, salt)
    return f"pair/({phrase},{quote(HMAC_TYPE, safe='')},{quote(b64(sealed), safe='')})"


def main() -> None:
    first_pair = build_pair(PASSPHRASE, PHRASE_ID, SALT, IV, EXPECTED_KEY)
    second_pair = build_pair(SECOND_PASSPHRASE, SECOND_ID, SECOND_SALT, SECOND_IV, SECOND_EXPECTED_KEY)

    # A deliberately corrupted payload: flip one byte inside the ciphertext so
    # the structure parses but the HMAC verification must fail.
    first_sealed = seal(
        f"type=key:cipher=AES-256:key={b64(EXPECTED_KEY)}".encode(),
        derive_key(PASSPHRASE, SALT),
        IV,
    )
    tampered = bytearray(first_sealed)
    tampered[16] ^= 0x01
    encoded_phrase = build_phrase(PHRASE_ID, SALT)
    encoded_data = quote(b64(bytes(tampered)), safe="")
    tampered_pair = f"pair/({encoded_phrase},{quote(HMAC_TYPE, safe='')},{encoded_data})"

    # VMX-level encryption data sealed with the key the first pair unseals to.
    vmx_payload = b'displayName = "Sealed VM"\ndataFileKey = "test-only-key"\n'
    vmx_iv = bytes((index * 3 + 9) % 256 for index in range(16))
    vmx_blob = seal(vmx_payload, EXPECTED_KEY, vmx_iv)

    fixtures = {
        "note": "Test-only key material generated independently of the vmx parser.",
        "phrase_id": PHRASE_ID,
        "passphrase": PASSPHRASE,
        "wrong_passphrase": WRONG_PASSPHRASE,
        "second_passphrase": SECOND_PASSPHRASE,
        "second_id": SECOND_ID,
        "salt_b64": b64(SALT),
        "iv_b64": b64(IV),
        "expected_key_b64": b64(EXPECTED_KEY),
        "second_expected_key_b64": b64(SECOND_EXPECTED_KEY),
        "keysafe_single": f"vmware:key/list/({first_pair})",
        "keysafe_multi": f"vmware:key/list/({first_pair},{second_pair})",
        "keysafe_tampered_mac": f"vmware:key/list/({tampered_pair})",
        "encryption_data_b64": b64(vmx_blob),
        "expected_vmx_dictionary": {
            "displayname": "Sealed VM",
            "datafilekey": "test-only-key",
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(fixtures, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
