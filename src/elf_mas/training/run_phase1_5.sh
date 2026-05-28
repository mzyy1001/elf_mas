#!/usr/bin/env bash
# Phase 1.5 sweep: 3 variants × 3 seeds = 9 runs.
# Run from chen, with `source ~/elf_mas/setup_env.sh && export CUDA_VISIBLE_DEVICES=5` first.

set -euo pipefail

OUT=${OUT:-$HOME/elf_mas/runs/phase1_5}
NUM_STEPS=${NUM_STEPS:-2000}
BATCH=${BATCH:-64}
LR=${LR:-3e-4}
EVAL_BATCHES=${EVAL_BATCHES:-25}

mkdir -p "$OUT"
echo "Phase 1.5 sweep -> $OUT"
echo "num_steps=$NUM_STEPS batch=$BATCH lr=$LR eval_batches=$EVAL_BATCHES"
echo "started: $(date)"

for variant in full identical_ctx frozen_agents; do
  for seed in 0 1 2; do
    name="${variant}_seed${seed}"
    log="$OUT/${name}.log"
    if [ -f "$log" ] && grep -q "LESION MATRIX" "$log"; then
      echo "[skip] $name (already done)"
      continue
    fi
    echo "[run]  $name -> $log"
    python -m elf_mas.training.train_coupled \
      --num_steps "$NUM_STEPS" --batch_size "$BATCH" --lr "$LR" \
      --variant "$variant" --seed "$seed" \
      --eval --eval_max_batches "$EVAL_BATCHES" > "$log" 2>&1
    echo "       done: $(date)"
  done
done

echo "ALL DONE: $(date)"
