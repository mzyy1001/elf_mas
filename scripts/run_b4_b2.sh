#!/usr/bin/env bash
# Run the two architectural baselines that were missing from the LoRA sweep,
# B4 single_model FIRST (the key parameter-matched control), then B2 frozen_agents.
# Train (30k) + eval (200 items) each, into runs/cola_mas_sweep/ alongside B0/B6/B7.
set -euo pipefail

DATASET="/home/chenhongrui/elf_mas/data/synth_2fact_v1"
RUNS_ROOT="/home/chenhongrui/elf_mas/runs/cola_mas_sweep"
NUM_STEPS="${1:-30000}"

for variant in single_model frozen_agents; do
  out="${RUNS_ROOT}/${variant}_seed0"
  if [[ -f "${out}/lora_state.pt" ]]; then
    echo "[b4b2] SKIP ${variant} (already trained)"; continue
  fi
  mkdir -p "$out"
  echo "[b4b2] === training ${variant} (${NUM_STEPS} steps) ==="
  PYTHONPATH=src python -m elf_mas_pt.training.train_cola \
    --dataset_dir "$DATASET" \
    --variant "$variant" --batch_size 8 --num_steps "$NUM_STEPS" \
    --lr 1e-4 --seed 0 --S_q 16 --S_ctx 64 \
    --lora_mode --lora_layers 8,12,16,20 --lambda_ce 0.3 --t_distribution uniform \
    --log_every 200 --prefix_cond --out_dir "$out" 2>&1 | tee "${out}/stdout.log"
  echo "[b4b2] === evaluating ${variant} ==="
  PYTHONPATH=src python -m elf_mas_pt.eval.eval_cola_mas \
    --ckpt_dir "$out" --dataset_dir "$DATASET" --split eval \
    --n_items 200 --batch_size 8 --T_inf 16 --S_q 16 --S_ctx 64 \
    --variant "$variant" --lora_mode 2>&1 | tee "${out}/eval_stdout.log"
done
echo "[b4b2] DONE"
