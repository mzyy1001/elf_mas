#!/usr/bin/env bash
# Cola+MAS variant sweep: B0/B6/B7/B2/B4 × N_seeds, 5000 steps each.
# Run sequentially on chen GPU 5 (or pass CUDA_VISIBLE_DEVICES=N).
#
# Usage:  bash scripts/cola_mas_variant_sweep.sh [num_steps] [n_seeds]
#         CUDA_VISIBLE_DEVICES=5 bash scripts/cola_mas_variant_sweep.sh 5000 3
set -euo pipefail

NUM_STEPS="${1:-5000}"
N_SEEDS="${2:-3}"
DATASET="${DATASET:-/home/chenhongrui/elf_mas/data/synth_2fact_v1}"
RUNS_ROOT="${RUNS_ROOT:-/home/chenhongrui/elf_mas/runs/cola_mas_sweep}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1e-4}"
S_CTX="${S_CTX:-64}"
S_Q="${S_Q:-16}"
# Phase 3 breakthrough recipe (all three required — ablating any one regressed to 0% EM):
# LoRA in-block MAS + uniform-t + CE aux. See claude_memory/project_phase3_breakthrough.md.
LORA_LAYERS="${LORA_LAYERS:-8,12,16,20}"
LAMBDA_CE="${LAMBDA_CE:-0.3}"
T_DIST="${T_DIST:-uniform}"

mkdir -p "$RUNS_ROOT"
ts=$(date +%Y%m%d_%H%M%S)
echo "[sweep] starting at $ts; num_steps=$NUM_STEPS n_seeds=$N_SEEDS dataset=$DATASET"

for variant in full identical_ctx context_shuffled frozen_agents single_model; do
  for seed in $(seq 0 $((N_SEEDS - 1))); do
    label="${variant}_seed${seed}"
    out="${RUNS_ROOT}/${label}"
    if [[ -f "${out}/lora_state.pt" ]]; then
      echo "[sweep] SKIP ${label} (already trained)"; continue
    fi
    mkdir -p "$out"
    echo "[sweep] === training ${label} ==="
    PYTHONPATH=src python -m elf_mas_pt.training.train_cola \
      --dataset_dir "$DATASET" \
      --variant "$variant" --batch_size "$BATCH_SIZE" --num_steps "$NUM_STEPS" \
      --lr "$LR" --seed "$seed" --S_q "$S_Q" --S_ctx "$S_CTX" \
      --lora_mode --lora_layers "$LORA_LAYERS" --lambda_ce "$LAMBDA_CE" --t_distribution "$T_DIST" \
      --log_every 200 --prefix_cond --out_dir "$out" 2>&1 | tee "${out}/stdout.log"
    echo "[sweep] === evaluating ${label} ==="
    PYTHONPATH=src python -m elf_mas_pt.eval.eval_cola_mas \
      --ckpt_dir "$out" --dataset_dir "$DATASET" --split eval \
      --n_items 200 --batch_size "$BATCH_SIZE" --T_inf 16 \
      --S_q "$S_Q" --S_ctx "$S_CTX" --variant "$variant" --lora_mode --prefix_cond 2>&1 | tee "${out}/eval_stdout.log"
  done
done
echo "[sweep] DONE"
