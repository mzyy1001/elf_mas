"""Aggregate Phase 1.5 sweep logs into a single per-variant table.

Reads logs at $HOME/elf_mas/runs/phase1_5/{variant}_seed{N}.log,
parses the lesion matrices that look like:

    [eval] LESION MATRIX (mean MSE; n_batches=25, bs=64)
        bucket        full      zero_a      zero_b   zero_both  Δa=full-zero_a  Δb=full-zero_b
            AB      0.0824      0.1825   ...

and reports per-variant means + std across seeds.
"""
from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def parse_lesion_block(text: str):
    """Return a dict bucket -> {full, zero_a, zero_b, zero_both, delta_a, delta_b}."""
    m = re.search(r"LESION MATRIX[^\n]*\n[^\n]*\n((?:\s*\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+.*\n){1,8})", text)
    if not m:
        return None
    out = {}
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("[eval]"):
            continue
        parts = line.split()
        # Expected: bucket full zero_a zero_b zero_both delta_a delta_b (n=...)
        if len(parts) < 7:
            continue
        bucket = parts[0]
        try:
            vals = [float(x) for x in parts[1:7]]
        except ValueError:
            continue
        out[bucket] = {
            "full": vals[0], "zero_a": vals[1], "zero_b": vals[2], "zero_both": vals[3],
            "delta_a": vals[4], "delta_b": vals[5],
        }
    return out


def parse_final_loss(text: str):
    """Last '[train] step=...  loss=X.YYYY' line before 'done'."""
    losses = re.findall(r"\[train\] step=\s*\d+\s+.*?loss=([0-9.]+)", text)
    return float(losses[-1]) if losses else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs_dir", default=os.path.expanduser("~/elf_mas/runs/phase1_5"))
    p.add_argument("--variants", nargs="+", default=["full", "identical_ctx", "frozen_agents"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = p.parse_args()

    table = defaultdict(lambda: defaultdict(list))  # variant -> bucket -> list of dicts
    final_losses = defaultdict(list)
    for variant in args.variants:
        for seed in args.seeds:
            log_path = Path(args.logs_dir) / f"{variant}_seed{seed}.log"
            if not log_path.exists():
                print(f"[warn] missing {log_path}")
                continue
            text = log_path.read_text()
            lesion = parse_lesion_block(text)
            fl = parse_final_loss(text)
            if fl is not None:
                final_losses[variant].append(fl)
            if not lesion:
                print(f"[warn] no lesion matrix in {log_path}")
                continue
            for bucket, row in lesion.items():
                table[variant][bucket].append(row)

    print(f"\n=== PHASE 1.5 SWEEP — {args.logs_dir} ===")
    for variant in args.variants:
        ls = final_losses.get(variant, [])
        if ls:
            mu = mean(ls)
            sd = stdev(ls) if len(ls) > 1 else 0.0
            print(f"\n[{variant}]  final loss (n={len(ls)} seeds): {mu:.4f} ± {sd:.4f}")
        else:
            print(f"\n[{variant}]  (no completed runs)")
            continue
        print(f"{'bucket':>10s}  {'full':>10s}  {'zero_a':>10s}  {'zero_b':>10s}  {'zero_both':>10s}  {'Δa':>14s}  {'Δb':>14s}")
        for bucket in ("AB", "A-only", "B-only", "neither"):
            rows = table[variant].get(bucket, [])
            if not rows:
                continue
            def mu_sd(key):
                xs = [r[key] for r in rows]
                if len(xs) == 0: return "—"
                m = mean(xs)
                s = stdev(xs) if len(xs) > 1 else 0.0
                return f"{m:.4f}±{s:.4f}"
            print(f"{bucket:>10s}  "
                  f"{mu_sd('full'):>13s}  {mu_sd('zero_a'):>13s}  {mu_sd('zero_b'):>13s}  "
                  f"{mu_sd('zero_both'):>13s}  {mu_sd('delta_a'):>15s}  {mu_sd('delta_b'):>15s}")


if __name__ == "__main__":
    main()
