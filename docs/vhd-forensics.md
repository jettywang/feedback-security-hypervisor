# Legacy VHD forensic support matrix

This note describes what the VHD reader can prove and what it cannot prove when it encounters a
fixed, dynamic or differencing VHD. It does **not** implement a parent chain.

## Hard rule

A file that opens without an exception is not automatically correct disk content. Legacy VHD
differencing images can look like dynamic images unless the footer disk type, parent locator,
BAT and sector bitmap are all interpreted.

## On-disk structures that decide inheritance

| Structure | Meaning in a fixed VHD | Meaning in a dynamic VHD | Meaning in a differencing VHD |
|---|---|---|---|
| Footer `disk_type` | `2` | `3` | `4`; identifies the image as a child |
| Footer `data_offset` | `0xFFFFFFFFFFFFFFFF`; data starts at file offset 0 | Points to the dynamic header | Points to the dynamic header |
| Dynamic header | Absent | Holds BAT and block size | Also holds parent UUID, timestamp, Unicode name and up to eight parent locators |
| Parent locator | Not applicable | Normally absent | Platform-specific path to the parent; never sufficient by itself |
| BAT entry | Not applicable | `0xFFFFFFFF` means sparse zero | `0xFFFFFFFF` means the whole block must come from the parent |
| Sector bitmap | Not applicable | Every bit for an allocated block must be 1 | 1 = sector stored in the child; 0 = sector inherited from parent |
| Footer/header checksum | Must validate | Must validate | Must validate before trusting BAT/bitmap-derived conclusions |

## Request result matrix

| Footer type and on-disk condition | Correct source | Current reader result | Trust verdict |
|---|---|---|---|
| Fixed VHD, request inside `current_size`, valid checksum and not truncated | Local file bytes | Reads local file bytes | Trusted |
| Fixed VHD, `data_offset` is not `0xFFFFFFFFFFFFFFFF` | Ambiguous footer | Refuses | Correct fail-closed behavior |
| Dynamic VHD, BAT allocated and all bitmap bits set | Local block bytes | Reads local block bytes | Trusted |
| Dynamic VHD, BAT `0xFFFFFFFF` | Defined zero block | Returns zeros | Trusted |
| Dynamic VHD, BAT entry 0 | Invalid offset | Refuses | Correct fail-closed behavior |
| Dynamic VHD, allocated block has a clear bitmap bit | Differencing semantics on a type-3 footer | Refuses when that block is read | Correct fail-closed behavior |
| Differencing VHD, BAT allocated and all bitmap bits set | Child block | Reader refuses the whole image at open time | Fail-closed; child bytes can be assessed with the diagnostic, but not flattened without chain support |
| Differencing VHD, BAT `0xFFFFFFFF` | Parent block | Reader refuses the whole image at open time | Fail-closed; a bitmap-blind reader would incorrectly return zeros |
| Differencing VHD, partial sector bitmap | Child for 1-bits, parent for 0-bits | Reader refuses the whole image at open time | Fail-closed; a bitmap-blind reader would incorrectly return child bytes, including poison bytes in cleared sectors |
| Differencing VHD with missing/unreadable parent locator | Parent cannot be located | Reader refuses the whole image | Correct; inherited content cannot be established |
| Candidate parent has a different UUID, timestamp or size | Wrong chain candidate | Diagnostic reports failed identity checks | Candidate cannot be used as evidence of inherited content |
| Invalid footer or dynamic header checksum | Metadata may be corrupt | Reader refuses | Correct fail-closed behavior |

## Why the old behavior was misleading

The previous dispatch logic used only `footer.data_offset`. A differencing VHD has a dynamic
header pointer, so it was constructed as `DynamicDisk` even though `footer.disk_type == 4`.

That old path had two inherited-content errors:

1. Sparse BAT entries (`0xFFFFFFFF`) were zero-filled. In a differencing image, those blocks must
   be read from the parent.
2. Sector bitmaps were used only to skip to the block data area. The reader never checked the
   bits, so clear bits were silently treated as child data instead of parent data.

The current reader fails closed for `disk_type == 4` and for inconsistent dynamic bitmaps instead
of presenting those incorrect bytes as forensic evidence.

## Runnable diagnostic

The diagnostic never opens the VHD as a flat disk and never follows a parent chain:

```sh
python -m dissect.hypervisor.tools.vhd_diagnose image.vhd
python -m dissect.hypervisor.tools.vhd_diagnose image.vhd --json
python -m dissect.hypervisor.tools.vhd_diagnose child.vhd --parent parent.vhd
```

The text output contains:

- footer and dynamic header checksum status;
- footer disk type and `data_offset` consistency;
- parent UUID, timestamp, Unicode name and decoded parent locator;
- BAT entry classification;
- a per-block support matrix (`allocated`, `zero`, `parent`, `partial_parent`, `ambiguous`,
  `corrupt`);
- explicit warnings whenever parent data is required but cannot be resolved;
- parent identity checks when a candidate parent is supplied.

## Minimal byte-marker control set

The test suite generates a parent, fixed child, dynamic child and differencing child in
`temporary_path`:

- parent marker: `0xA5`;
- dynamic child markers: `0x11` and `0x22`;
- differencing child marker: `0xCC`;
- partially inherited child sectors also contain `0xEE` poison bytes in the child block.

The differencing fixture makes the failure observable: a bitmap-blind read returns `0xEE`, while
the correct overlay would inherit `0xA5` from the parent for the clear bitmap bits.
