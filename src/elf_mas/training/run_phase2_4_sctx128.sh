#!/usr/bin/env bash
# Phase 2.4 — S_ctx=128 full sweep on MuSiQue.
# 5 variants × 3 seeds = 15 runs. Same training budget as Phase 2.1c.

set -euo pipefail

OUT=${OUT:-$HOME/elf_mas/runs/phase2_4_sctx128}
DATASET=${DATASET:-$HOME/elf_mas/data/musique_2hop_v128}
NUM_STEPS=${NUM_STEPS:-2000}
BATCH=${BATCH:-64}
LR=${LR:-3e-4}
EVAL_BATCHES=${EVAL_BATCHES:-25}
DECODE_BATCHES=${DECODE_BATCHES:-25}
S_CTX=${S_CTX:-128}

mkdir -p "$OUT"
echo "Phase 2.4 S_ctx=$S_CTX -> $OUT (dataset=$DATASET)"
echo "started: $(date)"

for variant in full single_model identical_ctx context_shuffled frozen_agents; do
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
      --S_ctx "$S_CTX" \
      --eval --extractive_decode \
      --eval_max_batches "$EVAL_BATCHES" --decode_max_batches "$DECODE_BATCHES" \
      > "$log" 2>&1
    echo "       done: $(date)"
  done
done

echo "ALL DONE: $(date)"
