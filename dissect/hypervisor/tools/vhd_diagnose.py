from __future__ import annotations

import argparse
import json
import struct
import sys
from dataclasses import asdict
from pathlib import Path

from dissect.hypervisor.disk.vhd import DISK_TYPE_DIFFERENCING
from dissect.hypervisor.disk.vhd_diagnose import diagnose, format_diagnostics, resolve_parent_path, verify_parent
from dissect.hypervisor.exceptions import Error


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="vhd-diagnose",
        description=(
            "Structurally inspect a VHD file and report which sectors the current reader can trust. "
            "This never opens the file as a flat disk and never resolves a parent chain."
        ),
    )
    parser.add_argument("vhd", type=Path, help="VHD file to inspect")
    parser.add_argument("-p", "--parent", type=Path, help="candidate parent file to verify against the child")
    parser.add_argument("--block-limit", type=int, default=None, help="maximum number of blocks to assess")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    if not args.vhd.is_file():
        parser.error(f"not a file: {args.vhd}")

    try:
        diagnostics = diagnose(args.vhd, block_limit=args.block_limit)
    except (Error, OSError, ValueError, struct.error) as exc:
        parser.exit(2, f"vhd-diagnose: {args.vhd}: {exc}\n")

    verification = None
    if args.parent is not None:
        verification = verify_parent(diagnostics, args.parent)
    elif diagnostics.disk_type == DISK_TYPE_DIFFERENCING:
        resolved = resolve_parent_path(diagnostics)
        if resolved is not None:
            verification = verify_parent(diagnostics, resolved)

    if args.json:
        report = {
            "path": str(diagnostics.path),
            "file_size": diagnostics.file_size,
            "footer_offset": diagnostics.footer_offset,
            "footer_size": diagnostics.footer_size,
            "footer_checksum_valid": diagnostics.footer_checksum_valid,
            "disk_type": diagnostics.disk_type,
            "disk_type_name": diagnostics.disk_type_name,
            "virtual_size": diagnostics.virtual_size,
            "reader_verdict": diagnostics.reader_verdict,
            "content_trust": diagnostics.content_trust,
            "parent_unique_id": diagnostics.parent_unique_id.hex() if diagnostics.parent_unique_id else None,
            "parent_timestamp": diagnostics.parent_timestamp,
            "parent_name": diagnostics.parent_name,
            "parent_reference": diagnostics.parent_reference,
            "parent_locators": [asdict(locator) for locator in diagnostics.parent_locators or []],
            "bat": diagnostics.bat,
            "block_summary": diagnostics.block_summary,
            "blocks": [asdict(block) for block in diagnostics.blocks or []],
            "warnings": diagnostics.warnings,
            "parent_verification": (
                {
                    "candidate": str(verification.candidate),
                    "resolved": str(verification.resolved) if verification.resolved else None,
                    "ok": verification.ok,
                    "checks": verification.checks,
                    "notes": verification.notes,
                }
                if verification
                else None
            ),
        }
        print(json.dumps(report, indent=2, default=str))
    else:
        print(format_diagnostics(diagnostics, verification))

    return 0


if __name__ == "__main__":
    sys.exit(main())
