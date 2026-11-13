"""Forensic regression tests for HyperVFile generation selection.

These tests do NOT assert that the highest sequence number is the "latest valid
config". They pin down what HyperVFile actually does when structures are torn or
cross-generational, and require the independent CRC-aware auditor to refuse the
"consistent" label whenever it cannot be proven.
"""
from __future__ import annotations

import pytest

from forensics.hyperv.hive_audit import audit_file
from forensics.hyperv.hive_build import HiveBuilder, KeyTableSpec
from forensics.hyperv.reproduce import (
    build_s1,
    build_s2,
    build_s3,
    build_s4,
    build_s5,
    _library_behaviour,
)


def test_s1_highest_sequence_wins_for_same_key():
    path = build_s1()
    lib = _library_behaviour(path)
    verdict = audit_file(str(path))

    # HyperVFile picks seq=2 purely by sequence number and exposes value 222.
    assert lib["loaded"] is True
    assert lib["selected_table_seq"] == {1: 2}
    assert lib["values"] == {"root": {"answer": 222}}

    # Both generations are individually intact and the log is empty -> provable.
    assert verdict.consistent is True
    assert verdict.selected_tables[1]["seq"] == 2


def test_s2_cross_generation_parent_is_dangling():
    path = build_s2()
    lib = _library_behaviour(path)
    verdict = audit_file(str(path))

    # The newer generation references a parent that only exists in the older one;
    # HyperVFile cannot even link the tree and raises during construction.
    assert lib["loaded"] is False
    assert "KeyError" in lib["error"]

    assert verdict.consistent is False
    assert len(verdict.dangling_parents) == 1
    assert verdict.dangling_parents[0]["key"] == "child"


def test_s3_torn_newest_header_is_not_treated_as_valid():
    path = build_s3()
    lib = _library_behaviour(path)
    verdict = audit_file(str(path))

    # Library prefers the higher sequence number, which is torn -> hard failure.
    assert lib["loaded"] is False
    assert "InvalidSignature" in lib["error"]
    assert verdict.selected_header_seq == 41
    assert verdict.selected_header_valid is False
    assert verdict.consistent is False


def test_s4_torn_newest_key_table_exposes_garbage_but_auditor_refuses():
    path = build_s4()
    lib = _library_behaviour(path)
    verdict = audit_file(str(path))

    # HyperVFile still "loads" and selects the torn high-sequence table, silently
    # exposing a zero-filled bogus root key instead of falling back to seq=1.
    assert lib["loaded"] is True
    assert lib["selected_table_seq"] == {1: 2}
    assert lib["values"] == {"\x00" * 16: {}}

    winner = verdict.selected_tables[1]
    assert winner["torn"] is True
    assert winner["header_crc_ok"] is True  # header CRC alone does not catch this
    assert verdict.consistent is False


def test_s5_dirty_replay_log_is_unproven_not_latest():
    path = build_s5()
    lib = _library_behaviour(path)
    verdict = audit_file(str(path))

    # HyperVFile collects the three outstanding log entries but never replays them.
    assert lib["loaded"] is True
    assert lib["replay_entries"] == 3
    assert lib["values"] == {"root": {"answer": 111}}

    # Intact structures, but the image is pre-replay: state cannot be proven,
    # so the auditor must return None rather than label it the latest config.
    assert verdict.replay["dirty"] is True
    assert verdict.consistent is None


def test_real_shipped_samples_are_provably_consistent():
    from tests._util import absolute_path

    for name in ("test.vmcx", "test.VMRS"):
        path = absolute_path(f"_data/descriptor/hyperv/{name}")
        verdict = audit_file(str(path))
        assert verdict.consistent is True, name
        assert verdict.replay["dirty"] is False
        assert all(t["header_crc_ok"] for t in verdict.selected_tables.values())
