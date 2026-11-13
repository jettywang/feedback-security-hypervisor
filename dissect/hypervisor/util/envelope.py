# References:
# - /bin/crypto-util
# - /usr/lib/vmware/tpm/bin/keypersist
# - /lib/libvmlibs.so

from __future__ import annotations

import contextlib
import hashlib
import io
import os
import tempfile
from base64 import b64decode
from typing import BinaryIO, Iterator, NamedTuple
from urllib.parse import unquote
from uuid import UUID

try:
    import _pystandalone

    HAS_PYSTANDALONE = True
except ImportError:
    HAS_PYSTANDALONE = False

try:
    from Crypto.Cipher import AES

    HAS_PYCRYPTODOME = True
except ImportError:
    HAS_PYCRYPTODOME = False

from dissect.cstruct import cstruct
from dissect.util.stream import RangeStream

c_def = """
struct EnvelopeFileHeader {
    char    magic[21];
    char    _pad[483];
    uint32  size;
    uint32  version;
};

struct DataTransformAeadFooter {
    char    magic[23];
    char    _pad[9];
    char    data[4056];
    uint32  size;
    uint32  version;
};

struct DataTransformCryptoFooter {
    char    magic[25];
    char    _pad[479];
    uint32  padding;
    uint32  version;
};

enum AttributeType : uint8 {
    Invalid = 0x0,
    // These are guesses based on size and used attributes
    UInt8 = 0x1,
    UInt16 = 0x2,
    UInt32 = 0x3,
    UInt64 = 0x4,
    Int8 = 0x5,
    Int16 = 0x6,
    Int32 = 0x7,
    Int64 = 0x8,
    Float = 0x9,
    Double = 0xA,
    // These are certain
    String = 0xB,
    Bytes = 0xC
};
"""
c_envelope = cstruct().load(c_def)

FILE_HEADER_MAGIC = b"DataTransformEnvelope"
FOOTER_AEAD_MAGIC = b"DataTransformAeadFooter"
FOOTER_CRYPTO_MAGIC = b"DataTransformCryptoFooter"

PBKDF2_SALT = b"This is obfuscation, not encryption. If you want encryption, use TPM."

ENVELOPE_BLOCK_SIZE = 4096
ENVELOPE_ATTRIBUTE_TYPE_MAP = {
    c_envelope.AttributeType.Invalid: None,
    c_envelope.AttributeType.UInt8: c_envelope.uint8,
    c_envelope.AttributeType.UInt16: c_envelope.uint16,
    c_envelope.AttributeType.UInt32: c_envelope.uint32,
    c_envelope.AttributeType.UInt64: c_envelope.uint64,
    c_envelope.AttributeType.Int8: c_envelope.int8,
    c_envelope.AttributeType.Int16: c_envelope.int16,
    c_envelope.AttributeType.Int32: c_envelope.int32,
    c_envelope.AttributeType.Int64: c_envelope.int64,
    c_envelope.AttributeType.Float: c_envelope.float,
    c_envelope.AttributeType.Double: c_envelope.double,
    c_envelope.AttributeType.String: None,
    c_envelope.AttributeType.Bytes: None,
}

DECRYPT_CHUNK_SIZE = 1024 * 1024 * 4
CRYPTO_FOOTER_SIZE = c_envelope.DataTransformCryptoFooter.size
_SPOOL_SUFFIX = ".ve-decrypted"


class EnvelopeAttribute(NamedTuple):
    type: int
    flag: int
    value: bytes


class Envelope:
    """Implements an encryption envelope as used within ESXi."""

    def __init__(self, fh: BinaryIO, verify: bool = True):
        self.fh = fh
        self.verify = verify

        header_buf = io.BytesIO(self.fh.read(ENVELOPE_BLOCK_SIZE))
        self.header = c_envelope.EnvelopeFileHeader(header_buf)

        if self.header.magic != FILE_HEADER_MAGIC:
            raise ValueError("Invalid envelope file")

        if self.header.version != 2:
            raise ValueError("Unsupported envelope type")

        self.version = self.header.version
        self.attributes = _read_envelope_attributes(header_buf)
        for req in ("vmware.keyInfo", "vmware.cipherName", "vmware.keyHash"):
            if req not in self.attributes:
                raise ValueError(f"Missing required {req} attribute")

        self.key_info = self.attributes["vmware.keyInfo"].value
        self.cipher_name = self.attributes["vmware.cipherName"].value
        self.key_hash = self.attributes["vmware.keyHash"].value
        self.iv = self.attributes.get("vmware.iv", EnvelopeAttribute(None, None, None)).value
        self.digest = None

        if self.cipher_name == "AES-256-GCM":
            self.fh.seek(-ENVELOPE_BLOCK_SIZE, io.SEEK_END)
            aead_footer = c_envelope.DataTransformAeadFooter(self.fh)
            if aead_footer.version != 1:
                raise ValueError("Invalid AEAD footer")
            self.digest = aead_footer.data[: aead_footer.size]
        else:
            raise NotImplementedError(f"Unsupported cipher: {self.cipher_name}")

        self.fh.seek(0, io.SEEK_END)
        size = self.fh.tell()
        self.size = size - (2 * ENVELOPE_BLOCK_SIZE)
        self.data = RangeStream(self.fh, ENVELOPE_BLOCK_SIZE, self.size)

    def decrypt(self, key: bytes, aad: bytes | None = None) -> bytes:
        """Decrypt the data in this envelope.

        This buffers the entire decrypted payload in memory. For large envelopes
        prefer :meth:`decrypt_stream`, which keeps memory bounded regardless of the
        envelope size and only exposes plaintext after authentication succeeds.

        Arguments:
            key: decryption key to use
            aad: optional associated data to include for AEAD ciphers
        """
        cipher = self._new_cipher(key, aad)

        self.data.seek(0)
        offset = 0
        decrypted = bytearray(self.size)
        while True:
            chunk = self.data.read(DECRYPT_CHUNK_SIZE)
            if not chunk:
                break

            chunk_size = len(chunk)
            decrypted[offset : offset + chunk_size] = cipher.decrypt(chunk)
            offset += chunk_size

        footer = c_envelope.DataTransformCryptoFooter(bytes(decrypted[-CRYPTO_FOOTER_SIZE:]))
        decrypted = decrypted[: -ENVELOPE_BLOCK_SIZE - footer.padding]

        if self.verify:
            cipher.verify(self.digest)

        return bytes(decrypted)

    @contextlib.contextmanager
    def decrypt_stream(
        self, key: bytes, aad: bytes | None = None, *, tmp_dir: str | None = None
    ) -> Iterator[BinaryIO]:
        """Stream-decrypt this envelope using bounded memory.

        The envelope is decrypted in fixed-size chunks into a temporary file, so
        memory usage stays bounded by :data:`DECRYPT_CHUNK_SIZE` instead of growing
        with the envelope size. Authentication is completed before any plaintext is
        handed out:

        1. the GCM tag (``self.digest``), the packed file header AAD and the optional
           ``aad`` are verified while streaming;
        2. the encrypted :class:`DataTransformCryptoFooter` is parsed and validated;
        3. only then is the temporary file truncated to the verified payload length
           and yielded as a readable, seekable binary stream positioned at the start.

        On any failure (short read, failed GCM/AAD verification, invalid footer or
        padding, or an error while writing the temporary file) the temporary file is
        closed and deleted and the exception propagates, so unauthenticated plaintext
        never leaves this context manager. The temporary file is likewise removed when
        the ``with`` block exits.

        Arguments:
            key: decryption key to use
            aad: optional associated data to include for AEAD ciphers
            tmp_dir: optional directory for the temporary file (defaults to the
                operating system's temporary directory)

        Yields:
            A seekable binary stream containing the authenticated plaintext.
        """
        cipher = self._new_cipher(key, aad)

        spool, spool_path = _open_spool(tmp_dir)
        try:
            # Keep only the final CRYPTO_FOOTER_SIZE bytes, which hold the crypto
            # footer (the last block is padded with random data). This rolling window
            # never grows beyond CRYPTO_FOOTER_SIZE, regardless of envelope size.
            tail = bytearray()
            total = 0

            self.data.seek(0)
            while True:
                chunk = self.data.read(DECRYPT_CHUNK_SIZE)
                if not chunk:
                    break

                plain = cipher.decrypt(chunk)
                spool.write(plain)
                total += len(plain)

                tail.extend(plain)
                if len(tail) > CRYPTO_FOOTER_SIZE:
                    del tail[: len(tail) - CRYPTO_FOOTER_SIZE]

            # A truncated envelope normally fails the GCM tag, but detect a short
            # read explicitly so callers get an unambiguous error.
            if total != self.size:
                raise ValueError(
                    f"Short read in envelope data: expected {self.size} bytes, got {total} bytes"
                )

            # Verify the GCM tag before validating or exposing any plaintext.
            if self.verify:
                cipher.verify(self.digest)

            footer = c_envelope.DataTransformCryptoFooter(bytes(tail))
            if footer.magic != FOOTER_CRYPTO_MAGIC:
                raise ValueError("Invalid crypto footer")

            overhead = ENVELOPE_BLOCK_SIZE + footer.padding
            if overhead > total:
                raise ValueError(f"Invalid crypto footer padding: {footer.padding}")

            # Authentication has succeeded; drop the padding and final block so the
            # yielded stream only contains the verified payload.
            spool.truncate(total - overhead)
            spool.flush()
            spool.seek(0)
            yield spool
        finally:
            spool.close()
            with contextlib.suppress(FileNotFoundError):
                os.unlink(spool_path)

    def _new_cipher(self, key: bytes, aad: bytes | None = None):
        """Build an initialized AEAD cipher and feed it the envelope AAD."""
        if not HAS_PYSTANDALONE and not HAS_PYCRYPTODOME:
            raise RuntimeError("No crypto module available")

        key_hash = hashlib.sha256(self.cipher_name.encode() + key).digest()
        if key_hash != self.key_hash:
            raise ValueError("Key hash doesn't match")

        if self.cipher_name != "AES-256-GCM":
            raise NotImplementedError(f"Unsupported cipher: {self.cipher_name}")

        if not self.iv:
            raise ValueError("Missing IV")

        if HAS_PYSTANDALONE:
            cipher = _pystandalone.aes_256_gcm(key, self.iv)
        else:
            cipher = AES.new(key, AES.MODE_GCM, nonce=self.iv)

        # The file header is included in the AAD, as well as any optional variable data
        cipher.update(_pack_envelope_header(self))
        if aad:
            cipher.update(aad)

        return cipher


class KeyStore:
    """Implements a file based keystore as used in ESXi."""

    def __init__(self, store: dict[str, str]):
        self.store = store

        self.mode = self.store.get("mode", None)
        if not self.mode:
            raise ValueError("Keystore has no mode")

        self._id = None
        self._key = None

        if self.mode == "NONE":
            data = self.store["ConfigEncData"]
            obj = {}
            for opt in data.split(":"):
                name, _, value = opt.partition("=")
                obj[name.strip()] = unquote(value.strip())

            self._id = str(UUID(bytes=b64decode(obj["keyId"])))

            data1 = b64decode(obj["data1"])
            data2 = b64decode(obj["data2"])

            self._key = hashlib.pbkdf2_hmac("sha256", data1 + PBKDF2_SALT, data2, 100000)
        else:
            raise NotImplementedError("Only NONE is implemented")

    @property
    def id(self) -> str:
        return self._id

    @property
    def key(self) -> bytes:
        return self._key

    @classmethod
    def from_text(cls, text: str) -> KeyStore:
        """Parse a key store from a string.

        Arguments:
            text: string to parse a key store from
        """
        store = {}

        for line in text.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            name, _, value = line.partition("=")

            name = name.strip()
            value = value.strip(' "')

            if name.startswith("."):
                store[name] = value
            else:
                node = store
                parts = name.strip().split(".")
                for part in parts[:-1]:
                    if part not in node:
                        node[part] = {}
                    node = node[part]

                node[parts[-1]] = value.strip(' "')

        return cls(store)


def _open_spool(tmp_dir: str | None = None) -> tuple[BinaryIO, str]:
    """Create a named temporary file used to spool decrypted data.

    The file is created in the most restrictive mode (``0o600``) and is unlinked by
    :meth:`Envelope.decrypt_stream` in its ``finally`` block.
    """
    fd, path = tempfile.mkstemp(suffix=_SPOOL_SUFFIX, dir=tmp_dir)
    return os.fdopen(fd, "w+b"), path


def _read_envelope_attributes(buf: BinaryIO) -> dict[str, EnvelopeAttribute]:
    attributes = {}
    while True:
        try:
            attr_type = c_envelope.AttributeType(buf)
            if attr_type == c_envelope.AttributeType.Invalid:
                break

            flag = c_envelope.uint8(buf)
            buf.read(2)

            name = c_envelope.char[None](buf).decode()
            if attr_type in ENVELOPE_ATTRIBUTE_TYPE_MAP:
                if attr_type == c_envelope.AttributeType.String:
                    value = c_envelope.char[None](buf).decode()
                elif attr_type == c_envelope.AttributeType.Bytes:
                    value = buf.read(c_envelope.uint64(buf))
                else:
                    value = ENVELOPE_ATTRIBUTE_TYPE_MAP[attr_type](buf)
            else:
                raise NotImplementedError(f"Unknown attribute type: {attr_type}")

            attributes[name] = EnvelopeAttribute(attr_type, flag, value)
        except EOFError:
            break

    return attributes


def _pack_envelope_header(envelope: Envelope, block_size: int = ENVELOPE_BLOCK_SIZE) -> bytes:
    stream = io.BytesIO()

    stream.write(b"\x00" * 512)

    _pack_attributes(stream, envelope.attributes)

    size = stream.tell()
    remainder = size % block_size
    if remainder:
        stream.write(b"\x00" * (block_size - remainder))
    size = stream.tell()

    stream.seek(0)
    c_envelope.EnvelopeFileHeader(
        magic=b"DataTransformEnvelope",
        size=size - len(c_envelope.EnvelopeFileHeader),
        version=envelope.version,
    ).write(stream)

    return stream.getvalue()


def _pack_attributes(stream: BinaryIO, attributes: dict[str, EnvelopeAttribute]) -> None:
    for name, attribute in attributes.items():
        c_envelope.AttributeType.write(stream, attribute.type)
        c_envelope.uint8.write(stream, attribute.flag)
        c_envelope.uint16.write(stream, 0)
        c_envelope.char[None].write(stream, name.encode())

        if attribute.type == c_envelope.AttributeType.String:
            c_envelope.char[None].write(stream, attribute.value.encode())
        elif attribute.type == c_envelope.AttributeType.Bytes:
            c_envelope.uint64.write(stream, len(attribute.value))
            stream.write(attribute.value)
        else:
            ENVELOPE_ATTRIBUTE_TYPE_MAP[attribute.type].write(stream, attribute.value)

    # Null terminate attributes
    stream.write(b"\x00" * 4)
