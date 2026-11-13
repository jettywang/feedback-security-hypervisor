"""Reproduce the HyperVFile generation-selection failure modes on minimal samples.

Scenarios (each is a self-contained, checksummed synthetic file):

  S1 same key, two generations, different values
  S2 child in the newer table points at a parent that only exists in the older
  S3 the newest storage header is torn (bad signature/CRC); older header is valid
  S4 the newest key table is torn (truncated); older generation is intact
  S5 dirty replay log (outstanding entries) - collection vs replay distinction

For every sample we record:
  * what ``HyperVFile`` actually selects/exposes (or which exception it raises);
  * what the independent CRC-aware auditor proves.
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from forensics.hyperv.hive_audit import audit_file
from forensics.hyperv.hive_build import HiveBuilder, KeyTableSpec

OUT = Path(__file__).parent / "samples"


def _library_behaviour(path: Path) -> dict:
    from dissect.hypervisor.descriptor.hyperv import HyperVFile

    result = {"loaded": False, "error": None, "selected_header_seq": None,
              "selected_table_seq": {}, "values": {}, "replay_entries": None}
    try:
        hf = HyperVFile(open(path, "rb"))
        result["loaded"] = True
        result["selected_header_seq"] = hf.header.sequence_number
        for idx, tables in hf.key_tables.items():
            result["selected_table_seq"][idx] = tables[0].sequence_number
        result["replay_entries"] = hf.replay_logs[0].header.num_entries
        try:
            result["values"] = hf.as_dict()
        except Exception as exc:  # dangling parents blow up here / during linking
            result["error"] = f"{type(exc).__name__}: {exc}"
            result["values"] = "<unavailable: tree linking/value read failed>"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["trace"] = traceback.format_exc(limit=2)
    return result


def build_s1() -> Path:
    """Same key 'answer' exists in table idx=1 seq=1 (value 111) and seq=2 (value 222)."""
    b = HiveBuilder()
    old = KeyTableSpec(index=1, sequence_number=1)
    old.node("root")                                   # offset 10
    old.uint("answer", 111, parent_table_idx=1, parent_offset=10)
    new = KeyTableSpec(index=1, sequence_number=2)
    new.node("root")
    new.uint("answer", 222, parent_table_idx=1, parent_offset=10)
    b.add_key_table(old)
    b.add_key_table(new)
    p = OUT / "s1_same_key_two_gens.bin"
    p.write_bytes(b.build(seq1=2, seq2=1))
    return p


def build_s2() -> Path:
    """Newer table's child points to a parent node that only exists in the older table."""
    b = HiveBuilder()
    old = KeyTableSpec(index=1, sequence_number=1)
    old_parent = old.node("root")                      # offset 10
    # node 'parent' lives only in the OLD generation, at a known offset
    old_parent_node = old.node("parent", parent_table_idx=1, parent_offset=10)
    new = KeyTableSpec(index=1, sequence_number=2)
    new.node("root")
    # child references the OLD parent by (table idx 1, old parent offset); that offset
    # is not a node in the new table layout
    new.uint(
        "child",
        999,
        parent_table_idx=1,
        parent_offset=old_parent_node.offset,
    )
    b.add_key_table(old)
    b.add_key_table(new)
    p = OUT / "s2_cross_generation_parent.bin"
    p.write_bytes(b.build(seq1=2, seq2=1))
    return p


def build_s3() -> Path:
    """Newest storage header (higher sequence) is torn; older header is intact."""
    b = HiveBuilder()
    t = KeyTableSpec(index=1, sequence_number=1)
    t.node("root")
    t.uint("answer", 777, parent_table_idx=1, parent_offset=10)
    b.add_key_table(t)
    b.torn_second_header = True  # bad signature + zero/garbled CRC on the high-seq header
    p = OUT / "s3_torn_newest_header.bin"
    # header2 gets the HIGHER sequence but is torn
    p.write_bytes(b.build(seq1=40, seq2=41))
    return p


def build_s4() -> Path:
    """Newest key table is half-written (truncated, CRC cannot validate)."""
    b = HiveBuilder()
    old = KeyTableSpec(index=1, sequence_number=1)
    old.node("root")
    old.uint("answer", 111, parent_table_idx=1, parent_offset=10)
    torn = KeyTableSpec(index=1, sequence_number=2, truncate_to=18)  # only header + partial entry
    torn.node("root")
    torn.uint("answer", 222, parent_table_idx=1, parent_offset=10)
    b.add_key_table(old)
    b.add_key_table(torn)
    p = OUT / "s4_torn_newest_keytable.bin"
    p.write_bytes(b.build(seq1=2, seq2=1))
    return p


def build_s5() -> Path:
    """Clean structures but a dirty replay log with outstanding entries."""
    b = HiveBuilder()
    t = KeyTableSpec(index=1, sequence_number=3)
    t.node("root")
    t.uint("answer", 111, parent_table_idx=1, parent_offset=10)
    b.add_key_table(t)
    b.set_replay_log(num_entries=3, max_entries=145)
    p = OUT / "s5_dirty_replay_log.bin"
    p.write_bytes(b.build(seq1=2, seq2=1))
    return p


SCENARIOS = [
    ("S1 same key, two generations", build_s1),
    ("S2 cross-generation parent", build_s2),
    ("S3 torn newest header", build_s3),
    ("S4 torn newest key table", build_s4),
    ("S5 dirty replay log", build_s5),
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for title, builder in SCENARIOS:
        path = builder()
        print("=" * 78)
        print(title, "->", path.name)
        print("-" * 78)
        lib = _library_behaviour(path)
        verdict = audit_file(str(path))
        print("[HyperVFile library behaviour]")
        if lib["error"] and not lib["loaded"]:
            # A hard load failure is the expected evidence for S2/S3, not a script error.
            print("  load FAILED (expected for this scenario):", lib["error"])
        else:
            print("  loaded:", lib["loaded"])
            print("  selected header seq:", lib["selected_header_seq"])
            print("  selected key-table seq per index:", lib["selected_table_seq"])
            print("  replay log num_entries (collected, never replayed):", lib["replay_entries"])
            print("  exposed as_dict:", lib["values"])
            if lib["error"]:
                print("  tree/value error:", lib["error"])
        print("[independent CRC-aware auditor]")
        print(verdict.summary())
        print()
    print("=" * 78)
    print(f"generated {len(SCENARIOS)} samples under {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
