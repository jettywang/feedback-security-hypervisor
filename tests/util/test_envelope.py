from __future__ import annotations

import hashlib
import io
import os
import tempfile
import tracemalloc
from types import SimpleNamespace
from typing import BinaryIO

import pytest

from dissect.hypervisor.util import envelope as envelope_module
from dissect.hypervisor.util.envelope import (
    DECRYPT_CHUNK_SIZE,
    ENVELOPE_BLOCK_SIZE,
    HAS_PYCRYPTODOME,
    HAS_PYSTANDALONE,
    Envelope,
    EnvelopeAttribute,
    KeyStore,
    c_envelope,
)
from tests._util import absolute_path

CRYPTO_REQUIRED = pytest.mark.skipif(
    (not HAS_PYCRYPTODOME and not HAS_PYSTANDALONE), reason="No crypto module available"
)


def test_envelope_keystore() -> None:
    store = KeyStore.from_text(absolute_path("_data/util/envelope/encryption.info").read_text())

    assert store.store[".encoding"] == "UTF-8"
    assert store.store["includeKeyCache"] == "FALSE"
    assert store.store["mode"] == "NONE"
    assert store.store["ConfigEncData"] == (
        "keyId=fmLOxWrvTX6Di8rjLu/SUQ%3d%3d:"
        "data1=Uz8MFZfqTbWN2jIHbFPhag%3d%3d:"
        "data2=IDcZE41aR+uxDv7Iz8zJSA%3d%3d:"
        "version=1"
    )

    assert store.mode == "NONE"
    assert store._id == "7e62cec5-6aef-4d7e-838b-cae32eefd251"
    assert store._key == bytes.fromhex("ae29634dca8627013f7c7cf2d05b4d5cc444d42cd4e8acbaa4fb815dda3b3066")


def test_envelope() -> None:
    with absolute_path("_data/util/envelope/local.tgz.ve").open("rb") as fh:
        ev = Envelope(fh)

    assert ev.key_info == "7e62cec5-6aef-4d7e-838b-cae32eefd251"
    assert ev.cipher_name == "AES-256-GCM"
    assert ev.key_hash == bytes.fromhex("7eaad9a3be03cbf5e092adc8ce04724f9b2a49785fba2533c56b7e26977b8c86")
    assert ev.iv == bytes.fromhex("574d565f0000000000000000")
    assert ev.digest == bytes.fromhex("ee54bdd51ac29a73a3bcaac47b3b6d93")
    assert ev.size == 102400


@CRYPTO_REQUIRED
def test_envelope_decrypt() -> None:
    with absolute_path("_data/util/envelope/local.tgz.ve").open("rb") as fh:
        ev = Envelope(fh)
        store = KeyStore.from_text(absolute_path("_data/util/envelope/encryption.info").read_text())

        decrypted = ev.decrypt(store.key, aad=b"ESXConfiguration")

    assert len(decrypted) == 94293
    assert hashlib.sha256(decrypted).hexdigest() == "fe131620351b9fd5fc4aef219bf3211340f3742464c038e1695e7b6667f86952"


# ---------------------------------------------------------------------------
# Helpers for building synthetic envelopes
# ---------------------------------------------------------------------------

TEST_IV = bytes.fromhex("574d565f0000000000000000")
TEST_KEY = bytes.fromhex("ae29634dca8627013f7c7cf2d05b4d5cc444d42cd4e8acbaa4fb815dda3b3066")
TEST_CIPHER = "AES-256-GCM"


def _packed_header(iv: bytes) -> bytes:
    """Reproduce the 4096-byte AAD header that Envelope packs from attributes."""
    namespace = SimpleNamespace(
        version=2,
        attributes={
            "vmware.keyInfo": EnvelopeAttribute(c_envelope.AttributeType.String, 0, "test-key"),
            "vmware.cipherName": EnvelopeAttribute(
                c_envelope.AttributeType.String, 0, TEST_CIPHER
            ),
            "vmware.keyHash": EnvelopeAttribute(
                c_envelope.AttributeType.Bytes,
                0,
                hashlib.sha256(TEST_CIPHER.encode() + TEST_KEY).digest(),
            ),
            "vmware.iv": EnvelopeAttribute(c_envelope.AttributeType.Bytes, 0, iv),
        },
    )
    header = envelope_module._pack_envelope_header(namespace)
    assert len(header) == ENVELOPE_BLOCK_SIZE
    return header


def build_envelope(
    payload: bytes,
    *,
    key: bytes = TEST_KEY,
    iv: bytes = TEST_IV,
    aad: bytes | None = None,
    padding_override: int | None = None,
) -> bytes:
    """Build a valid on-disk AES-256-GCM envelope for ``payload``."""
    header = _packed_header(iv)

    from Crypto.Cipher import AES

    cipher = AES.new(key, AES.MODE_GCM, nonce=iv)
    cipher.update(header)
    if aad:
        cipher.update(aad)

    natural_padding = (-len(payload)) % ENVELOPE_BLOCK_SIZE
    footer_padding = natural_padding if padding_override is None else padding_override

    footer_stream = io.BytesIO()
    c_envelope.DataTransformCryptoFooter(
        magic=b"DataTransformCryptoFooter",
        _pad=b"\x00" * 479,
        padding=footer_padding,
        version=2,
    ).write(footer_stream)
    crypto_footer = footer_stream.getvalue()
    assert len(crypto_footer) == c_envelope.DataTransformCryptoFooter.size

    # Match the real on-disk layout: payload, the natural block-alignment padding, the
    # remainder of the final 4096-byte block, and the 512-byte crypto footer last.
    # ``padding_override`` only changes the value claimed by the footer, letting us craft
    # a GCM-valid envelope whose declared padding is impossible.
    plain = (
        payload
        + os.urandom(natural_padding)
        + os.urandom(ENVELOPE_BLOCK_SIZE - c_envelope.DataTransformCryptoFooter.size)
        + crypto_footer
    )

    ciphertext, digest = cipher.encrypt_and_digest(plain)
    assert len(ciphertext) == len(plain)

    aead_stream = io.BytesIO()
    c_envelope.DataTransformAeadFooter(
        magic=b"DataTransformAeadFooter",
        _pad=b"\x00" * 9,
        data=digest + b"\x00" * (4056 - len(digest)),
        size=len(digest),
        version=1,
    ).write(aead_stream)
    aead_footer = aead_stream.getvalue()
    assert len(aead_footer) == ENVELOPE_BLOCK_SIZE

    return header + ciphertext + aead_footer


class _ShortReadStream:
    """File wrapper whose readable data ends at ``cut`` once armed, while seeking
    still reports the full (declared) length, simulating a truncated body / short read.

    The wrapper stays fully readable while the :class:`Envelope` is constructed (so the
    header and trailing AEAD footer parse correctly); call ``arm()`` before the data
    stream is consumed to make the middle of the file read as a short read.
    """

    def __init__(self, buf: bytes, cut: int):
        self._buf = buf
        self._cut = cut
        self._pos = 0
        self._armed = False

    def arm(self) -> None:
        self._armed = True

    def seek(self, pos: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = pos
        elif whence == io.SEEK_CUR:
            self._pos += pos
        elif whence == io.SEEK_END:
            self._pos = len(self._buf) + pos
        return self._pos

    def tell(self) -> int:
        return self._pos

    def read(self, n: int = -1) -> bytes:
        limit = self._cut if self._armed else len(self._buf)
        if self._pos >= limit:
            return b""
        if n is None or n < 0:
            n = limit - self._pos
        chunk = self._buf[self._pos : min(self._pos + n, limit)]
        self._pos += len(chunk)
        return chunk


class _RecordingSpool:
    """Wraps a real temp file and records the largest single write, proving writes are
    chunk-bounded rather than emitted as one envelope-sized buffer."""

    def __init__(self, wrapped: BinaryIO, path: str):
        self._wrapped = wrapped
        self.path = path
        self.max_write = 0
        self.write_count = 0
        self.bytes_written = 0
        self.closed = False

    def write(self, data: bytes) -> int:
        self.max_write = max(self.max_write, len(data))
        self.write_count += 1
        self.bytes_written += len(data)
        return self._wrapped.write(data)

    def close(self) -> None:
        self._wrapped.close()
        self.closed = True

    def __getattr__(self, name: str):
        return getattr(self._wrapped, name)


class _FailingSpool(_RecordingSpool):
    """Raises after ``fail_after`` writes to simulate an output write failure."""

    def __init__(self, wrapped: BinaryIO, path: str, fail_after: int):
        super().__init__(wrapped, path)
        self._fail_after = fail_after

    def write(self, data: bytes) -> int:
        if self.write_count >= self._fail_after:
            raise OSError("simulated spool write failure")
        return _RecordingSpool.write(self, data)


def _install_spool(monkeypatch, factory):
    created: list = []

    def fake_open_spool(tmp_dir=None):
        fd, path = tempfile.mkstemp(suffix=".ve-decrypted", dir=tmp_dir)
        spool = factory(os.fdopen(fd, "w+b"), path)
        created.append(spool)
        return spool, path

    monkeypatch.setattr(envelope_module, "_open_spool", fake_open_spool)
    return created


# ---------------------------------------------------------------------------
# decrypt_stream behaviour
# ---------------------------------------------------------------------------


@CRYPTO_REQUIRED
def test_decrypt_stream_roundtrip() -> None:
    payload = os.urandom(20_000)
    blob = build_envelope(payload, aad=b"ESXConfiguration")

    ev = Envelope(io.BytesIO(blob))
    with ev.decrypt_stream(TEST_KEY, aad=b"ESXConfiguration") as out:
        assert out.seekable()
        result = out.read()

    assert result == payload

    # A fresh cipher (new Envelope) must reproduce the same payload via the legacy API.
    legacy = Envelope(io.BytesIO(blob)).decrypt(TEST_KEY, aad=b"ESXConfiguration")
    assert legacy == payload
    assert result == legacy


@CRYPTO_REQUIRED
def test_decrypt_stream_real_envelope() -> None:
    with absolute_path("_data/util/envelope/local.tgz.ve").open("rb") as fh:
        ev = Envelope(fh)
        store = KeyStore.from_text(absolute_path("_data/util/envelope/encryption.info").read_text())
        with ev.decrypt_stream(store.key, aad=b"ESXConfiguration") as out:
            result = out.read()

    assert len(result) == 94293
    assert hashlib.sha256(result).hexdigest() == "fe131620351b9fd5fc4aef219bf3211340f3742464c038e1695e7b6667f86952"


@CRYPTO_REQUIRED
def test_decrypt_stream_short_read(tmp_path) -> None:
    payload = os.urandom(20_000)
    blob = build_envelope(payload)
    cut = ENVELOPE_BLOCK_SIZE + 5_000

    source = _ShortReadStream(blob, cut)
    ev = Envelope(source)
    source.arm()

    spool_dir = tmp_path / "spool"
    spool_dir.mkdir()
    with pytest.raises(ValueError, match="Short read"):
        with ev.decrypt_stream(TEST_KEY, tmp_dir=str(spool_dir)):
            pytest.fail("plaintext must not be yielded on a short read")

    # The temporary resource must be reclaimed on failure.
    assert list(spool_dir.iterdir()) == []


@CRYPTO_REQUIRED
def test_decrypt_stream_bad_aad(tmp_path) -> None:
    payload = os.urandom(20_000)
    blob = build_envelope(payload, aad=b"correct-aad")

    ev = Envelope(io.BytesIO(blob))

    spool_dir = tmp_path / "spool"
    spool_dir.mkdir()
    with pytest.raises(ValueError):  # GCM tag mismatch raised by cipher.verify
        with ev.decrypt_stream(TEST_KEY, aad=b"wrong-aad", tmp_dir=str(spool_dir)):
            pytest.fail("plaintext must not be yielded on AAD/GCM failure")

    assert list(spool_dir.iterdir()) == []


@CRYPTO_REQUIRED
def test_decrypt_stream_bad_padding(tmp_path) -> None:
    # A well-formed GCM envelope whose crypto footer claims an impossible padding
    # length: authentication passes, but the footer validation must reject it before
    # any plaintext is yielded.
    payload = os.urandom(3_000)
    blob = build_envelope(payload, padding_override=100_000)

    ev = Envelope(io.BytesIO(blob))

    spool_dir = tmp_path / "spool"
    spool_dir.mkdir()
    with pytest.raises(ValueError, match="Invalid crypto footer padding"):
        with ev.decrypt_stream(TEST_KEY, tmp_dir=str(spool_dir)):
            pytest.fail("plaintext must not be yielded on invalid padding")

    assert list(spool_dir.iterdir()) == []


@CRYPTO_REQUIRED
def test_decrypt_stream_write_failure(monkeypatch, tmp_path) -> None:
    payload = os.urandom(6_000)
    blob = build_envelope(payload)
    ev = Envelope(io.BytesIO(blob))

    created = _install_spool(monkeypatch, lambda fh, path: _FailingSpool(fh, path, fail_after=0))

    with pytest.raises(OSError, match="simulated spool write failure"):
        with ev.decrypt_stream(TEST_KEY, tmp_dir=str(tmp_path)):
            pytest.fail("plaintext must not be yielded on an output write failure")

    assert len(created) == 1
    assert created[0].closed is True
    assert not os.path.exists(created[0].path)


@CRYPTO_REQUIRED
def test_decrypt_stream_cleans_up_on_success(monkeypatch, tmp_path) -> None:
    blob = build_envelope(os.urandom(20_000))
    ev = Envelope(io.BytesIO(blob))

    created = _install_spool(monkeypatch, _RecordingSpool)

    with ev.decrypt_stream(TEST_KEY, tmp_dir=str(tmp_path)) as out:
        path = created[0].path
        assert os.path.exists(path)
        out.read()

    # Spool is closed and the named temporary file is removed once the block exits.
    assert created[0].closed is True
    assert not os.path.exists(path)


class _MemoryProbeSpool(_RecordingSpool):
    """Records live Python allocation (current, not high-water peak) after every write,
    while the stream is being produced."""

    def __init__(self, wrapped: BinaryIO, path: str):
        super().__init__(wrapped, path)
        self.max_live = 0

    def write(self, data: bytes) -> int:
        result = _RecordingSpool.write(self, data)
        current, _ = tracemalloc.get_traced_memory()
        self.max_live = max(self.max_live, current)
        return result


def _write_envelope(tmp_path, name: str, payload: bytes) -> str:
    path = tmp_path / name
    path.write_bytes(build_envelope(payload))
    return str(path)


@CRYPTO_REQUIRED
def test_decrypt_stream_bounded_memory(monkeypatch, tmp_path) -> None:
    # Realistic on-disk sources so the full envelope is not held in a Python byte string.
    small_path = _write_envelope(tmp_path, "small.ve", os.urandom(10_000))
    # ~20 MiB payload: well over DECRYPT_CHUNK_SIZE so it spans many chunks.
    large_path = _write_envelope(tmp_path, "large.ve", os.urandom(5 * DECRYPT_CHUNK_SIZE + 12345))

    def run(path: str) -> _MemoryProbeSpool:
        created = _install_spool(monkeypatch, _MemoryProbeSpool)
        with open(path, "rb") as fh:
            ev = Envelope(fh)
            tracemalloc.start()
            try:
                with ev.decrypt_stream(TEST_KEY, tmp_dir=str(tmp_path)) as out:
                    out.read()
            finally:
                tracemalloc.stop()
        return created[0]

    small_spool = run(small_path)
    large_spool = run(large_path)

    # Write-count / size proof: the largest single spool write is always a fixed chunk
    # and never scales with the envelope size, even though the large envelope is 1000x
    # bigger and spans many more writes.
    assert large_spool.bytes_written > 5 * DECRYPT_CHUNK_SIZE
    assert large_spool.bytes_written > small_spool.bytes_written * 100
    assert large_spool.write_count > small_spool.write_count
    assert large_spool.max_write <= DECRYPT_CHUNK_SIZE
    assert small_spool.max_write <= DECRYPT_CHUNK_SIZE

    # Live-memory proof: peak resident Python memory while streaming the ~20 MiB
    # envelope stays within a small constant number of chunks (bounded), far below the
    # envelope size (i.e. it does not grow linearly with the envelope).
    assert large_spool.max_live < 3 * DECRYPT_CHUNK_SIZE
    assert large_spool.max_live < large_spool.bytes_written
    # Bounded means the large run's live memory is comparable to the small run, not
    # proportional to the 1000x size difference.
    assert large_spool.max_live < small_spool.max_live + 2 * DECRYPT_CHUNK_SIZE

    # Contrast with the legacy API, which allocates one full-size output buffer and
    # therefore grows linearly with the envelope size.
    with open(large_path, "rb") as fh:
        ev_large = Envelope(fh)
        tracemalloc.start()
        try:
            ev_large.decrypt(TEST_KEY)
            _, legacy_peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
    assert legacy_peak > ev_large.size
    assert legacy_peak > large_spool.max_live

