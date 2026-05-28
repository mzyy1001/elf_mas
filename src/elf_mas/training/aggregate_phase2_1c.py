"""Aggregate Phase 2.1c (extractive-decode) logs into a per-variant table.

Parses the `EXTRACTIVE NN DECODE` output block. Reports per-bucket × variant:
  - recall (% items where gold is in extracted candidates)
  - full EM/F1
  - zero_a, zero_b, zero_both EM/F1
  - EM|recall (full)
plus final loss and MSE lesion asymmetry (parsed from the LESION MATRIX block).
"""
from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


_EM_F1_RE = re.compile(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)")


def parse_extractive_block(text):
    m = re.search(r"EXTRACTIVE NN DECODE[^\n]*\n[^\n]*\n((?:\s*\S+.*\n){1,8})", text)
    if not m:
        return None
    out = {}
    for line in m.group(1).splitlines():
        s = line.strip()
        if not s or s.startswith("[") or "—" in s:
            continue
        parts = s.split()
        if len(parts) < 5:
            continue
        bucket = parts[0]
        if bucket not in ("AB", "A-only", "B-only", "neither"):
            continue
        try:
            n = int(parts[1])
            recall = float(parts[2].rstrip("%"))
            csize = float(parts[3])
            # Extract all EM/F1 pairs from the rest of the line (handles spaces around '/')
            rest = " ".join(parts[4:])
            pairs = _EM_F1_RE.findall(rest)
            if len(pairs) < 4:
                continue
            cells = [(float(em), float(f1)) for em, f1 in pairs[:4]]
            # last bare float in line is EM|rec
            em_cond = float(parts[-1])
        except (ValueError, IndexError):
            continue
        names = ("full", "zero_a", "zero_b", "zero_both")
        out[bucket] = {
            "n": n, "recall": recall, "cand_size": csize, "em_cond_recall": em_cond,
            **{names[i]: {"em": cells[i][0], "f1": cells[i][1]} for i in range(4)},
        }
    return out


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
        out[parts[0]] = {"delta_a": vals[4], "delta_b": vals[5], "full_mse": vals[0]}
    return out


def parse_final_loss(text):
    losses = re.findall(r"\[train\] step=\s*\d+\s+.*?loss=([0-9.]+)", text)
    return float(losses[-1]) if losses else None


def _mean_sd(xs):
    if not xs:
        return float("nan"), 0.0
    if len(xs) == 1:
        return xs[0], 0.0
    return mean(xs), stdev(xs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs_dir", default=os.path.expanduser("~/elf_mas/runs/phase2_1c_pilot_musique"))
    p.add_argument("--variants", nargs="+", default=["full", "identical_ctx", "context_shuffled", "frozen_agents"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = p.parse_args()

    tbl = defaultdict(lambda: defaultdict(list))  # variant -> bucket -> list of dicts (extractive)
    tbl_les = defaultdict(lambda: defaultdict(list))
    losses = defaultdict(list)
    for variant in args.variants:
        for seed in args.seeds:
            log = Path(args.logs_dir) / f"{variant}_seed{seed}.log"
            if not log.exists():
                print(f"[warn] missing {log}")
                continue
            text = log.read_text()
            ex = parse_extractive_block(text)
            les = parse_lesion_block(text)
            fl = parse_final_loss(text)
            if fl is not None: losses[variant].append(fl)
            if ex:
                for b, row in ex.items():
                    tbl[variant][b].append(row)
            if les:
                for b, row in les.items():
                    tbl_les[variant][b].append(row)

    print(f"\n=== PHASE 2.1c PILOT (extractive NN) — {args.logs_dir} ===")
    for variant in args.variants:
        ls = losses.get(variant, [])
        mu, sd = _mean_sd(ls)
        if ls:
            print(f"\n[{variant}]  final MSE (n={len(ls)} seeds): {mu:.4f} ± {sd:.4f}")
        else:
            print(f"\n[{variant}]  (no completed runs)")
            continue

        # Extractive table
        if tbl[variant]:
            print(f"  -- extractive NN decode --")
            print(f"  {'bucket':>10s}  {'recall':>8s}  {'cand_sz':>8s}  "
                  f"{'full EM/F1':>14s}  {'zero_a':>14s}  {'zero_b':>14s}  {'zero_both':>14s}  {'EM|rec full':>13s}")
            for b in ("AB","A-only","B-only","neither"):
                rows = tbl[variant].get(b, [])
                if not rows: continue
                rec = mean([r["recall"] for r in rows])
                cs = mean([r["cand_size"] for r in rows])
                em_cr = mean([r["em_cond_recall"] for r in rows])
                cells = []
                for a in ("full","zero_a","zero_b","zero_both"):
                    ems = [r[a]["em"] for r in rows]
                    f1s = [r[a]["f1"] for r in rows]
                    cells.append(f"{mean(ems):5.1f}/{mean(f1s):5.1f}")
                print(f"  {b:>10s}  {rec:>7.1f}%  {cs:>7.1f}   "
                      + "  ".join(f"{c:>14s}" for c in cells) + f"  {em_cr:>12.1f}")

        # MSE lesion (Δa, Δb)
        if tbl_les[variant]:
            print(f"  -- MSE lesion --")
            print(f"  {'bucket':>10s}  {'Δa':>14s}  {'Δb':>14s}  {'ratio':>8s}")
            for b in ("AB","A-only","B-only","neither"):
                rows = tbl_les[variant].get(b, [])
                if not rows: continue
                da = mean([r["delta_a"] for r in rows])
                db = mean([r["delta_b"] for r in rows])
                ratio = (da / db) if db > 0 else float("inf")
                print(f"  {b:>10s}  {da:>14.4f}  {db:>14.4f}  {ratio:>7.2f}x")


if __name__ == "__main__":
    main()
