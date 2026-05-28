"""Aggregate Phase 2.1 pilot logs into per-variant tables for both MSE
lesion matrix and NN decode EM/F1.

Reads {variant}_seed{N}.log files in --logs_dir.
"""
from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def parse_lesion_block(text):
    m = re.search(r"LESION MATRIX[^\n]*\n[^\n]*\n((?:\s*\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+.*\n){1,8})", text)
    if not m:
        return None
    out = {}
    for line in m.group(1).splitlines():
        s = line.strip()
        if not s or s.startswith("[eval]"):
            continue
        parts = s.split()
        if len(parts) < 7:
            continue
        try:
            vals = [float(x) for x in parts[1:7]]
        except ValueError:
            continue
        out[parts[0]] = {
            "full": vals[0], "zero_a": vals[1], "zero_b": vals[2], "zero_both": vals[3],
            "delta_a": vals[4], "delta_b": vals[5],
        }
    return out


def parse_decode_block(text):
    """Block looks like:
    [decode] NEAREST-NEIGHBOR EM/F1 (n_batches=25, bs=64)
        bucket      full EM/F1          zero_a          zero_b       zero_both
            AB      45.2/ 52.1      31.0/ 38.4      20.8/ 25.6       1.2/  3.8  (n=...)
    """
    m = re.search(r"NEAREST-NEIGHBOR EM/F1[^\n]*\n[^\n]*\n((?:\s*\S+.*\n){1,8})", text)
    if not m:
        return None
    out = {}
    for line in m.group(1).splitlines():
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) < 9:
            continue
        bucket = parts[0]
        try:
            cells = [
                (float(parts[1].split("/")[0]), float(parts[2])),  # full em / f1
                (float(parts[3].split("/")[0]), float(parts[4])),
                (float(parts[5].split("/")[0]), float(parts[6])),
                (float(parts[7].split("/")[0]), float(parts[8])),
            ]
        except (ValueError, IndexError):
            continue
        names = ("full","zero_a","zero_b","zero_both")
        out[bucket] = {n: {"em": cells[i][0], "f1": cells[i][1]} for i, n in enumerate(names)}
    return out


def parse_final_loss(text):
    losses = re.findall(r"\[train\] step=\s*\d+\s+.*?loss=([0-9.]+)", text)
    return float(losses[-1]) if losses else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs_dir", default=os.path.expanduser("~/elf_mas/runs/phase2_1_pilot_synth"))
    p.add_argument("--variants", nargs="+", default=["full", "identical_ctx", "context_shuffled", "frozen_agents"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = p.parse_args()

    table_lesion = defaultdict(lambda: defaultdict(list))
    table_decode = defaultdict(lambda: defaultdict(list))
    final_losses = defaultdict(list)
    for variant in args.variants:
        for seed in args.seeds:
            log = Path(args.logs_dir) / f"{variant}_seed{seed}.log"
            if not log.exists():
                print(f"[warn] missing {log}")
                continue
            text = log.read_text()
            les = parse_lesion_block(text)
            dec = parse_decode_block(text)
            fl = parse_final_loss(text)
            if fl is not None: final_losses[variant].append(fl)
            if les:
                for b, row in les.items():
                    table_lesion[variant][b].append(row)
            if dec:
                for b, row in dec.items():
                    table_decode[variant][b].append(row)

    print(f"\n=== PHASE 2.1 PILOT — {args.logs_dir} ===")
    for variant in args.variants:
        ls = final_losses.get(variant, [])
        if ls:
            mu, sd = mean(ls), (stdev(ls) if len(ls) > 1 else 0.0)
            print(f"\n[{variant}]  final loss (n={len(ls)} seeds): {mu:.4f} ± {sd:.4f}")
        else:
            print(f"\n[{variant}]  (no completed runs)")
            continue

        if table_lesion[variant]:
            print(f"  -- MSE lesion (Δa, Δb) --")
            print(f"  {'bucket':>10s}  {'Δa':>14s}  {'Δb':>14s}")
            for b in ("AB","A-only","B-only","neither"):
                rows = table_lesion[variant].get(b, [])
                if not rows: continue
                das = [r["delta_a"] for r in rows]
                dbs = [r["delta_b"] for r in rows]
                da_str = f"{mean(das):.4f}±{stdev(das) if len(das)>1 else 0:.4f}"
                db_str = f"{mean(dbs):.4f}±{stdev(dbs) if len(dbs)>1 else 0:.4f}"
                print(f"  {b:>10s}  {da_str:>14s}  {db_str:>14s}")

        if table_decode[variant]:
            print(f"  -- EM/F1 decode (full / zero_a / zero_b / zero_both) --")
            print(f"  {'bucket':>10s}  {'full EM/F1':>14s}  {'zero_a':>14s}  {'zero_b':>14s}  {'zero_both':>14s}")
            for b in ("AB","A-only","B-only","neither"):
                rows = table_decode[variant].get(b, [])
                if not rows: continue
                def fmt(name):
                    ems = [r[name]["em"] for r in rows]
                    f1s = [r[name]["f1"] for r in rows]
                    return f"{mean(ems):5.1f}/{mean(f1s):5.1f}"
                print(f"  {b:>10s}  {fmt('full'):>14s}  {fmt('zero_a'):>14s}  {fmt('zero_b'):>14s}  {fmt('zero_both'):>14s}")


if __name__ == "__main__":
    main()
