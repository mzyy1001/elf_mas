#!/usr/bin/env bash
# Phase 2.1 pilot — validate NN decoder + B7 context-shuffled on synth_2fact_v1
# before scaling to MuSiQue. 4 variants × 3 seeds = 12 runs.
#
# Usage: source ~/elf_mas/setup_env.sh && export CUDA_VISIBLE_DEVICES=5 && bash run_phase2_1_pilot.sh

set -euo pipefail

OUT=${OUT:-$HOME/elf_mas/runs/phase2_1_pilot_synth}
DATASET=${DATASET:-$HOME/elf_mas/data/synth_2fact_v1}
NUM_STEPS=${NUM_STEPS:-2000}
BATCH=${BATCH:-64}
LR=${LR:-3e-4}
EVAL_BATCHES=${EVAL_BATCHES:-25}
DECODE_BATCHES=${DECODE_BATCHES:-25}

mkdir -p "$OUT"
echo "Phase 2.1 pilot -> $OUT (dataset=$DATASET)"
echo "started: $(date)"

for variant in full identical_ctx context_shuffled frozen_agents; do
  for seed in 0 1 2; do
    name="${variant}_seed${seed}"
    log="$OUT/${name}.log"
    if [ -f "$log" ] && grep -q "NEAREST-NEIGHBOR EM/F1" "$log"; then
      echo "[skip] $name (already done)"
      continue
    fi
    echo "[run]  $name -> $log"
    python -m elf_mas.training.train_coupled \
      --num_steps "$NUM_STEPS" --batch_size "$BATCH" --lr "$LR" \
      --variant "$variant" --seed "$seed" \
      --dataset_dir "$DATASET" \
      --eval --decode \
      --eval_max_batches "$EVAL_BATCHES" --decode_max_batches "$DECODE_BATCHES" \
      > "$log" 2>&1
    echo "       done: $(date)"
  done
done

echo "ALL DONE: $(date)"
