#!/usr/bin/env bash
# Phase 2.1c pilot — per-question extractive NN decode on MuSiQue.
# 4 variants × 3 seeds = 12 runs.

set -euo pipefail

OUT=${OUT:-$HOME/elf_mas/runs/phase2_1c_pilot_musique}
DATASET=${DATASET:-$HOME/elf_mas/data/musique_2hop_v1}
NUM_STEPS=${NUM_STEPS:-2000}
BATCH=${BATCH:-64}
LR=${LR:-3e-4}
EVAL_BATCHES=${EVAL_BATCHES:-25}
DECODE_BATCHES=${DECODE_BATCHES:-25}

mkdir -p "$OUT"
echo "Phase 2.1c pilot -> $OUT (dataset=$DATASET)"
echo "started: $(date)"

for variant in full identical_ctx context_shuffled frozen_agents; do
  for seed in 0 1 2; do
    name="${variant}_seed${seed}"
    log="$OUT/${name}.log"
    if [ -f "$log" ] && grep -q "EXTRACTIVE NN DECODE" "$log"; then
      echo "[skip] $name (already done)"
      continue
    fi
    echo "[run]  $name -> $log"
    python -m elf_mas.training.train_coupled \
      --num_steps "$NUM_STEPS" --batch_size "$BATCH" --lr "$LR" \
      --variant "$variant" --seed "$seed" \
      --dataset_dir "$DATASET" \
      --eval --extractive_decode \
      --eval_max_batches "$EVAL_BATCHES" --decode_max_batches "$DECODE_BATCHES" \
      > "$log" 2>&1
    echo "       done: $(date)"
  done
done

echo "ALL DONE: $(date)"
