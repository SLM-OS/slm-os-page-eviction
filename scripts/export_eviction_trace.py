#!/usr/bin/env python3
"""Export per-scenario eviction access traces for the SLM-OS hardware harness.

SLM-OS's `bench eviction-e2e` (#979) replays these traces through the real
weight/workspace pools + real eviction policies on hardware, so the per-policy
fault rates can be compared directly against this simulator's results.

The simulator's residency identity is the tuple (model_id, layer_idx,
pool_type) — see core.py `content_key`. `AccessRequest` already carries that
tuple plus `access_pattern`, so we can export the request sequence verbatim;
no simulator instrumentation is needed. The harness keys residency on the same
tuple and stamps `access_pattern` so the in-tree (parity-tested) policies see
the same feature inputs the simulator fed them.

Blob format (little-endian):
    magic    : 4 bytes  b"EVT1"
    version  : u32       = 1
    seed     : u32       (the workload seed used)
    n_scen   : u32
    directory: n_scen * { name[16], n_accesses u32, record_offset u32 }
    records  : 8 bytes each, grouped by scenario in directory order:
               model_id u8, pool_type u8, access_pattern u8, _pad u8,
               layer_idx i32

PoolType:      Weight=0, Workspace=1
AccessPattern: SEQUENTIAL=0, RANDOM=1, STRIDED=2, BURST=3
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

from src.simulator.workload import WorkloadGenerator

MAGIC = b"EVT1"
VERSION = 1


def export(seed: int, out_path: Path) -> None:
    names = WorkloadGenerator.all_scenario_names()
    wg = WorkloadGenerator(seed=seed)

    # Encode each scenario's records first; assemble the blob after, once
    # offsets are known.
    scen_records: list[tuple[str, bytes]] = []
    for name in names:
        requests = wg.generate_scenario(name)
        buf = bytearray()
        for r in requests:
            buf += struct.pack(
                "<BBBxi",
                int(r.model_id) & 0xFF,
                int(r.pool_type) & 0xFF,
                int(r.access_pattern) & 0xFF,
                int(r.layer_idx),
            )
        scen_records.append((name, bytes(buf)))

    n_scen = len(scen_records)
    header_len = 4 + 4 + 4 + 4               # magic + version + seed + n_scen
    dir_entry_len = 16 + 4 + 4               # name + n_accesses + offset
    records_start = header_len + n_scen * dir_entry_len

    header = bytearray()
    header += MAGIC
    header += struct.pack("<III", VERSION, seed, n_scen)

    directory = bytearray()
    records = bytearray()
    offset = records_start
    for name, rec in scen_records:
        n_acc = len(rec) // 8
        name_b = name.encode("ascii")[:16].ljust(16, b"\x00")
        directory += name_b + struct.pack("<II", n_acc, offset)
        records += rec
        offset += len(rec)

    blob = bytes(header) + bytes(directory) + bytes(records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(blob)

    total_acc = sum(len(r) // 8 for _, r in scen_records)
    print(f"Wrote {out_path} ({len(blob)} bytes, {n_scen} scenarios, "
          f"{total_acc} accesses, seed={seed})")
    for name, rec in scen_records:
        print(f"  {name:18s} {len(rec) // 8:6d} accesses")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    export(args.seed, args.out)


if __name__ == "__main__":
    main()
