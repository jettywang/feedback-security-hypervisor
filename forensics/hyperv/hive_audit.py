"""Independent, evidence-first auditor for synthetic/real HyperVStorage files.

Unlike ``HyperVFile`` (which *trusts* the highest sequence number and never
replays the log), this auditor:

  * validates every CRC32 instead of assuming the newest structure is intact;
  * reports which header/table would be *selected* vs which is actually *valid*;
  * resolves every key entry's parent pointer and flags dangling cross-
    generation references;
  * distinguishes a clean file from a dirty one that still carries replay
    entries (collected, not replayed).

Run directly for a human-readable report; import :func:`audit_file` to use the
machine-readable result in tests.
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field

SIG_HEADER = 0x01282014
SIG_REPLAY = 0x01110003
SIG_OBJECT = 0x01110001
SIG_KEYTABLE = 0x0002
KT_FREE = 1
KT_NODE = 9
OBJECT_TABLE_OFFSET = 0x2000
HEADER_OFFSETS = (0x0000, 0x1000)


def crc32(buf: bytes | bytearray) -> int:
    return zlib.crc32(bytes(buf)) & 0xFFFFFFFF


@dataclass
class Verdict:
    path: str
    headers: list[dict] = field(default_factory=list)
    selected_header_seq: int | None = None
    selected_header_valid: bool | None = None
    key_tables: list[dict] = field(default_factory=list)
    selected_tables: dict[int, dict] = field(default_factory=dict)
    dangling_parents: list[dict] = field(default_factory=list)
    replay: dict = field(default_factory=dict)
    consistent: bool | None = None
    reasons: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"== {self.path} =="]
        for h in self.headers:
            lines.append(
                f"  header@0x{h['offset']:x} seq={h['seq']:>3} sig=0x{h['sig']:08x} "
                f"crc_ok={h['crc_ok']} replay_off=0x{h['replay_offset']:x}"
            )
        lines.append(
            f"  library would select header seq={self.selected_header_seq} "
            f"(crc valid: {self.selected_header_valid})"
        )
        for t in self.key_tables:
            lines.append(
                f"  keytable idx={t['index']} seq={t['seq']:>3} off=0x{t['offset']:x} "
                f"header_crc_ok={t['header_crc_ok']} entries={t['entry_count']} torn={t['torn']}"
            )
        for idx, t in sorted(self.selected_tables.items()):
            lines.append(
                f"  -> selected for idx={idx}: seq={t['seq']} crc_ok={t['crc_ok']} "
                f"torn={t['torn']}"
            )
        for d in self.dangling_parents:
            lines.append(
                f"  DANGLING parent: key={d['key']!r} in table idx={d['table_index']} "
                f"points -> table idx={d['parent_table']} off=0x{d['parent_offset']:x}"
            )
        lines.append(
            f"  replay log: num_entries={self.replay.get('num_entries')} "
            f"max={self.replay.get('max_entries')} crc_ok={self.replay.get('crc_ok')} "
            f"dirty={self.replay.get('dirty')}"
        )
        lines.append(f"  CONSISTENT GENERATION PROVEN: {self.consistent}")
        for r in self.reasons:
            lines.append(f"     - {r}")
        return "\n".join(lines)


def _read_header(buf, offset):
    raw = bytes(buf[offset : offset + 46])
    sig, stored, seq, version, unk2, align, rl_off, rl_size, hsize = struct.unpack(
        "<IIH I Q I QQ I", raw
    )
    zeroed = bytearray(raw)
    zeroed[4:8] = b"\x00\x00\x00\x00"
    return {
        "offset": offset,
        "sig": sig,
        "seq": seq,
        "version": version,
        "alignment": align,
        "replay_offset": rl_off,
        "replay_size": rl_size,
        "stored_crc": stored,
        "crc_ok": (sig == SIG_HEADER) and crc32(zeroed) == stored,
    }


def _read_key_table(buf, offset, size):
    raw = bytes(buf[offset : offset + size])
    available = len(raw)
    torn = available < 10
    info = {
        "offset": offset,
        "index": None,
        "seq": None,
        "crc_ok": False,
        "header_crc_ok": False,
        "torn": False,
        "entry_count": 0,
        "entries": [],
    }
    if available < 10:
        info["torn"] = True
        return info
    sig, index, seq, stored_crc = struct.unpack("<HHHI", raw[:10])
    info["index"] = index
    info["seq"] = seq

    # Walk entries defensively. The object table reports a full page size and a
    # healthy table may fill the whole page with no end marker, so length alone
    # cannot distinguish a tear. A half-written table is recognised by structural
    # impossibilities that valid tables never contain:
    #   * a partial entry header at the end of the page;
    #   * a zeroed header slot where an entry was expected;
    #   * an entry header whose declared size runs past the page;
    #   * an impossible data_offset, or a present header whose key/value body is
    #     entirely zero padding (the S4-style mid-entry truncation).
    cursor = 10
    entries = []
    while cursor < available:
        if cursor + 21 > available:
            if any(raw[cursor:available]):
                info["torn"] = True  # partial entry header
            break
        header_bytes = raw[cursor : cursor + 21]
        etype, esize, ptable, poff, ecksum, ins, doff = struct.unpack(
            "<HIHIIIB", header_bytes
        )
        if esize == 0:
            break  # optional end-of-entries marker
        if all(b == 0 for b in header_bytes):
            info["torn"] = True  # zeroed slot where a serialised entry was expected
            break
        if cursor + esize > available:
            info["torn"] = True  # declared entry runs past the table page
            break
        etype_low = etype & 0xFF
        entry_body = raw[cursor + 21 : cursor + esize]
        # Free entries (type 1) legitimately carry data_offset=0 and an empty body;
        # only allocated, typed entries must have a plausible offset and real bytes.
        if etype_low != KT_FREE:
            if doff == 0 or (cursor + 21 + doff - 1) > (cursor + esize):
                info["torn"] = True  # data_offset impossible for a fully written entry
                break
            if not any(entry_body):
                info["torn"] = True  # header present but key/value body is zero padding
                break
        # doff is relative to the entry body (bytes after the 21-byte header):
        # the key name occupies body[0 : doff-1], value bytes start at body[doff].
        name_start = cursor + 21
        name_end = cursor + 21 + doff - 1
        key = bytes(raw[name_start:name_end]).decode("utf-8", "replace")
        entries.append(
            {
                "key": key,
                "type": etype & 0xFF,
                "table_index": index,
                "parent_table": ptable,
                "parent_offset": poff,
                "entry_offset": cursor,
            }
        )
        cursor += esize
    info["entries"] = entries
    info["entry_count"] = len(entries)

    # The key table checksum is a plain CRC32 over the 10-byte header only
    # (verified against shipped VMCX/VMRS). It stays valid even when the entry
    # area is torn, so a forensics verdict must combine crc_ok with the torn flag.
    # Entry bodies carry their own checksum field, but real files use a
    # VmDataStore-specific routine there, so consistency is not gated on it.
    header_region = bytearray(raw[:10])
    header_region[6:10] = b"\x00\x00\x00\x00"
    info["header_crc_ok"] = (sig == SIG_KEYTABLE) and crc32(header_region) == stored_crc
    info["crc_ok"] = info["header_crc_ok"] and not info["torn"]
    return info


def audit_file(path: str) -> Verdict:
    buf = open(path, "rb").read()
    v = Verdict(path=path)

    # ---- headers ----
    for off in HEADER_OFFSETS:
        v.headers.append(_read_header(buf, off))
    valid_headers = [h for h in v.headers if h["crc_ok"]]
    # mimic HyperVFile line 55: pure max sequence, ties -> header2
    chosen = max(v.headers, key=lambda h: h["seq"])
    v.selected_header_seq = chosen["seq"]
    v.selected_header_valid = chosen["crc_ok"]

    # ---- replay log from the *selected* header (exactly what the library follows) ----
    rl_off = chosen["replay_offset"]
    rl_raw = bytes(buf[rl_off : rl_off + 34])
    rsig, rcrc, num, u1, mx, u2, u3, u4, u5, u6 = struct.unpack("<IIBIIIIIIB", rl_raw)
    rz = bytearray(rl_raw)
    rz[4:8] = b"\x00\x00\x00\x00"
    v.replay = {
        "offset": rl_off,
        "sig": rsig,
        "num_entries": num,
        "max_entries": mx,
        "crc_ok": (rsig == SIG_REPLAY) and crc32(rz) == rcrc,
        "dirty": num > 0,
    }

    # ---- object table -> key tables ----
    osig, num_entries = struct.unpack("<II", bytes(buf[0x2000:0x2008]))
    cursor = 0x2008
    by_index: dict[int, list[dict]] = {}
    for _ in range(num_entries):
        if cursor + 18 > len(buf):
            v.reasons.append("object table declares more entries than the file holds")
            v.consistent = False
            break
        etype, ecksum, eoff, esize, alloc = struct.unpack(
            "<BIQIB", bytes(buf[cursor : cursor + 18])
        )
        cursor += 18
        if alloc == 0 or etype != 2:  # only KeyTable entries matter here
            continue
        t = _read_key_table(buf, eoff, esize)
        v.key_tables.append(t)
        if t["index"] is not None:
            by_index.setdefault(t["index"], []).append(t)

    # mimic HyperVFile line 87/100: highest sequence per index, no validity gate
    for index, tables in by_index.items():
        winner = max(tables, key=lambda t: (t["seq"] or -1))
        v.selected_tables[index] = winner

    # ---- resolve parent pointers across the *selected* generation ----
    selected_entries: dict[tuple[int, int], dict] = {}
    for index, t in v.selected_tables.items():
        for e in t["entries"]:
            selected_entries[(index, e["entry_offset"])] = e
    for index, t in v.selected_tables.items():
        for e in t["entries"]:
            if e["parent_table"] == 0:
                continue  # root node has no parent
            ref = (e["parent_table"], e["parent_offset"])
            if ref not in selected_entries:
                v.dangling_parents.append(
                    {
                        "key": e["key"],
                        "table_index": index,
                        "parent_table": e["parent_table"],
                        "parent_offset": e["parent_offset"],
                    }
                )

    # ---- verdict: explicit tri-state True / False / None(unknown) ----
    hard_failure = False
    if not valid_headers:
        hard_failure = True
        v.reasons.append("no storage header passes its CRC32 check")
    elif not chosen["crc_ok"]:
        hard_failure = True
        v.reasons.append(
            f"library selects header seq={chosen['seq']} but its CRC32 is invalid "
            "(torn/newest header); a valid older header exists"
        )
    bad_tables = [t for t in v.selected_tables.values() if not t["crc_ok"] or t["torn"]]
    if bad_tables:
        hard_failure = True
        for t in bad_tables:
            v.reasons.append(
                f"selected key table idx={t['index']} seq={t['seq']} is "
                f"crc_ok={t['crc_ok']} torn={t['torn']}"
            )
    if v.dangling_parents:
        hard_failure = True
        v.reasons.append(
            f"{len(v.dangling_parents)} entry/entries reference a parent absent from the "
            "selected generation (cross-generation dangling pointer)"
        )

    if hard_failure:
        v.consistent = False
    elif v.replay["dirty"]:
        # structures are intact, but the image is pre-replay: without applying the
        # log we cannot prove the resulting state, so do NOT call it the latest config.
        v.consistent = None
        v.reasons.append(
            f"replay log holds {v.replay['num_entries']} outstanding entr(ies): the on-disk "
            "tables are a pre-replay image; HyperVFile only collects the log, it never replays"
        )
        v.reasons.append("consistency cannot be proven without applying the replay log")
    else:
        v.consistent = True
        v.reasons.append(
            "all selected structures pass CRC32, every parent resolves, and the replay log is empty"
        )
    return v


if __name__ == "__main__":
    import sys

    for p in sys.argv[1:]:
        print(audit_file(p).summary())
        print()
