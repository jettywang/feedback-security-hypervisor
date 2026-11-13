"""Minimal, dependency-free builder for synthetic Hyper-V "HyperVStorage" files.

The on-disk layout mimics what ``dissect.hypervisor.descriptor.hyperv.HyperVFile``
parses (version 0x400):

    0x0000  storage header #1   (0x1000 slot)
    0x1000  storage header #2   (0x1000 slot)
    0x2000  object table        (fixed OBJECT_TABLE_OFFSET)
    0x3000+ key tables / file objects
    ...     replay log (offset referenced by the active header)

Everything is checksummed with CRC32 over the structure with its checksum
field zeroed, exactly like real VMCX/VMRS files (verified against shipped
samples). The builder deliberately exposes low-level "damage" hooks so a
forensic test can create torn headers, half-written key tables and dangling
cross-generation parent pointers.

This module only writes bytes; it never imports dissect, so the same generator
can be used on a box without the library.
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field

PAGE = 0x1000
SIG_HEADER = 0x01282014
SIG_REPLAY = 0x01110003
SIG_OBJECT = 0x01110001
SIG_KEYTABLE = 0x0002

# ObjectEntryType
OT_OBJECT_TABLE = 1
OT_KEY_TABLE = 2
OT_FILE = 3
OT_FREE = 4
OT_REPLAY_LOG = 6

# KeyDataType
KT_FREE = 1
KT_INT = 3
KT_UINT = 4
KT_STRING = 6
KT_BOOL = 8
KT_NODE = 9

FLAG_FILE_OBJECT = 0x0100  # high byte of the uint16 type field


def crc32(buf: bytes | bytearray) -> int:
    return zlib.crc32(bytes(buf)) & 0xFFFFFFFF


@dataclass
class KeyEntry:
    key: str
    ktype: int
    value: object = None
    parent_table_idx: int = 0
    parent_offset: int = 0
    insertion_sequence: int = 0
    # resolved while serialising
    offset: int = 0
    raw: bytes = b""


@dataclass
class KeyTableSpec:
    index: int
    sequence_number: int
    entries: list[KeyEntry] = field(default_factory=list)
    # when set, the stored header checksum is deliberately wrong
    corrupt_header_checksum: bool = False
    # when set, the table payload is truncated to this many bytes (torn write)
    truncate_to: int | None = None
    # serialised bytes, assigned during build()
    _blob: bytes = b""

    def node(self, key, parent_table_idx=0, parent_offset=0, insertion_sequence=0):
        e = KeyEntry(key, KT_NODE, None, parent_table_idx, parent_offset, insertion_sequence)
        self.entries.append(e)
        return e

    def uint(self, key, value, parent_table_idx=0, parent_offset=0, insertion_sequence=0):
        self.entries.append(
            KeyEntry(key, KT_UINT, int(value), parent_table_idx, parent_offset, insertion_sequence)
        )

    def string(self, key, value, parent_table_idx=0, parent_offset=0, insertion_sequence=0):
        self.entries.append(
            KeyEntry(key, KT_STRING, value, parent_table_idx, parent_offset, insertion_sequence)
        )


def _encode_entry(e: KeyEntry) -> None:
    name = e.key.encode("utf-8") + b"\x00"
    header_size = 21
    # data_offset is relative to the entry *body* (the bytes after the 21-byte
    # header): the key name starts at 0, so the value begins right after the name.
    # Verified against real VMCX/VMRS entries (library slices raw[:data_offset-1]).
    data_offset = len(name)  # value starts right after the name

    if e.ktype == KT_NODE:
        # 8 unknown bytes + 4 byte insertion sequence
        value_bytes = b"\x00" * 8 + struct.pack("<I", e.insertion_sequence)
    elif e.ktype == KT_UINT:
        value_bytes = struct.pack("<Q", e.value)
    elif e.ktype == KT_INT:
        value_bytes = struct.pack("<q", e.value)
    elif e.ktype == KT_BOOL:
        value_bytes = struct.pack("<I", 1 if e.value else 0)
    elif e.ktype == KT_STRING:
        payload = e.value.encode("utf-16-le")
        value_bytes = struct.pack("<I", len(payload)) + payload
    else:
        raise ValueError(f"unsupported type {e.ktype}")

    body = name + value_bytes
    total = header_size + len(body)
    # entry header: uint16 type, uint32 size, uint16 ptable, uint32 poffset,
    # uint32 checksum, uint32 insertion, uint8 data_offset
    hdr_no_cksum = struct.pack(
        "<HIHIIIB",
        e.ktype,
        total,
        e.parent_table_idx,
        e.parent_offset,
        0,  # checksum placeholder
        e.insertion_sequence,
        data_offset,
    )
    checksum = crc32(hdr_no_cksum + body)
    hdr = struct.pack(
        "<HIHIIIB",
        e.ktype,
        total,
        e.parent_table_idx,
        e.parent_offset,
        checksum,
        e.insertion_sequence,
        data_offset,
    )
    e.raw = hdr + body


class HiveBuilder:
    """Builds a version-0x400 HyperVStorage image in a bytearray."""

    def __init__(self, alignment: int = PAGE):
        self.alignment = alignment
        self.buf = bytearray()
        # table index -> list of (offset, size, KeyTableSpec)
        self.table_records: list[tuple[int, int, KeyTableSpec]] = []
        self.replay_offset = 0
        self.replay_num_entries = 0
        self.replay_max_entries = 145
        # damage hooks
        self.torn_second_header = False
        self.bad_second_header_checksum = False
        self.header2_extra: dict = {}

    def _align(self, n: int) -> int:
        if n % self.alignment == 0:
            return n
        return n + self.alignment - (n % self.alignment)

    def _place(self, data: bytes) -> int:
        offset = self._align(len(self.buf))
        if len(self.buf) < offset:
            self.buf += b"\x00" * (offset - len(self.buf))
        self.buf += data
        return offset

    def add_key_table(self, spec: KeyTableSpec) -> int:
        for e in spec.entries:
            _encode_entry(e)
        # assign intra-table offsets (first entry starts right after the 10-byte table header)
        cursor = 10
        for e in spec.entries:
            e.offset = cursor
            cursor += len(e.raw)
        table_header_no_cksum = struct.pack(
            "<HHHI", SIG_KEYTABLE, spec.index, spec.sequence_number, 0
        )
        payload = b"".join(e.raw for e in spec.entries)
        # Real Hyper-V files only CRC32 the 10-byte key table *header* here; entry
        # integrity is carried by each entry's own checksum field instead.
        checksum = 0 if spec.corrupt_header_checksum else crc32(table_header_no_cksum)
        header = struct.pack("<HHHI", SIG_KEYTABLE, spec.index, spec.sequence_number, checksum)
        blob = header + payload
        if spec.truncate_to is not None:
            blob = blob[: spec.truncate_to]
        # store the serialised blob; fixed page offsets are assigned later in build()
        spec._blob = blob
        self.table_records.append((0, self.alignment, spec))
        return 0

    def set_replay_log(self, num_entries: int, max_entries: int = 145):
        self.replay_num_entries = num_entries
        self.replay_max_entries = max_entries

    def build(self, seq1: int, seq2: int) -> bytes:
        n_tables = len(self.table_records)
        # deterministic page map: headers 0x0/0x1000, object table 0x2000,
        # key tables from 0x3000, replay log in the page after the last table.
        key_table_base = 0x3000
        replay_offset = key_table_base + n_tables * self.alignment
        replay_size = self.alignment
        total_size = replay_offset + replay_size
        self.buf = bytearray(b"\x00" * total_size)

        # ---- place key tables at fixed pages ----
        resolved_records = []
        for i, (_, size, spec) in enumerate(self.table_records):
            offset = key_table_base + i * self.alignment
            blob = spec._blob
            self.buf[offset : offset + len(blob)] = blob
            resolved_records.append((offset, size, spec))
        self.table_records = resolved_records

        # ---- build replay log ----
        replay = bytearray()
        rl_body = struct.pack(
            "<IIBIIIIIIB",
            SIG_REPLAY,
            0,  # checksum
            self.replay_num_entries,
            0,  # unknown1
            self.replay_max_entries,
            0, 0, 0, 0,  # unknown2..5
            0,  # unknown6
        )
        rl_checksum = crc32(rl_body)
        rl_header = struct.pack(
            "<IIBIIIIIIB",
            SIG_REPLAY,
            rl_checksum,
            self.replay_num_entries,
            0,
            self.replay_max_entries,
            0, 0, 0, 0,
            0,
        )
        replay += rl_header
        for i in range(self.replay_num_entries):
            entry_no_cksum = struct.pack("<QIIIII", 0, 0, 0, 0, 0, 0)
            entry = struct.pack(
                "<QIIIII",
                0,  # offset (irrelevant to collection logic)
                0,  # size
                0, 0,
                crc32(entry_no_cksum),
                0,  # data_checksum
            )
            replay += entry
        self.buf[replay_offset : replay_offset + len(replay)] = replay

        # ---- object table at fixed 0x2000 ----
        obj_entries = b""
        n_obj = n_tables + 1  # key tables + one replay log entry
        for offset, size, spec in self.table_records:
            ote = struct.pack("<BIQIB", OT_KEY_TABLE, 0, offset, size, 1)
            # checksum over 0x12 bytes with the uint32 checksum field zeroed
            body = bytearray(ote)
            body[1:5] = b"\x00\x00\x00\x00"
            ote = struct.pack("<BIQIB", OT_KEY_TABLE, crc32(body), offset, size, 1)
            obj_entries += ote
        # replay log object entry
        ote = bytearray(struct.pack("<BIQIB", OT_REPLAY_LOG, 0, replay_offset, replay_size, 1))
        body = bytearray(ote)
        body[1:5] = b"\x00\x00\x00\x00"
        ote = struct.pack("<BIQIB", OT_REPLAY_LOG, crc32(body), replay_offset, replay_size, 1)
        obj_entries += ote

        obj_table_blob = struct.pack("<II", SIG_OBJECT, n_obj) + obj_entries
        self._place_at(0x2000, obj_table_blob)

        # ---- two storage headers ----
        def storage_header(seq, torn=False, bad_checksum=False):
            h = struct.pack(
                "<IIH I Q I QQ I",
                SIG_HEADER if not torn else 0xDEADBEEF,
                0,  # checksum placeholder
                seq,
                0x400,
                0,  # unknown2
                self.alignment,
                replay_offset,
                replay_size,
                0x1000,  # header_size
            )
            checksum = 0 if bad_checksum else crc32(h)
            return struct.pack(
                "<IIH I Q I QQ I",
                SIG_HEADER if not torn else 0xDEADBEEF,
                checksum,
                seq,
                0x400,
                0,
                self.alignment,
                replay_offset,
                replay_size,
                0x1000,
            )

        h1 = storage_header(seq1)
        h2 = storage_header(
            seq2,
            torn=self.torn_second_header,
            bad_checksum=self.bad_second_header_checksum,
        )
        self._place_at(0x0000, h1)
        self._place_at(0x1000, h2)

        return bytes(self.buf)

    def _place_at(self, offset: int, data: bytes) -> int:
        end = offset + len(data)
        if len(self.buf) < end:
            self.buf += b"\x00" * (end - len(self.buf))
        self.buf[offset:end] = data
        return offset
