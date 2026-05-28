#!/usr/bin/env bash
# CE-loss lambda sweep over LoRA-MAS, synth_2fact_v1.
set -euo pipefail

DATASET="/home/chenhongrui/elf_mas/data/synth_2fact_v1"
RUNS_ROOT="/home/chenhongrui/elf_mas/runs/cola_mas_ce_sweep"

for lam in 0.1 0.3 1.0; do
  out="${RUNS_ROOT}/synth_lora_lce${lam}_seed0_10k"
  if [[ -f "${out}/lora_state.pt" ]]; then
    echo "[sweep] SKIP λ=$lam (already trained)"; continue
  fi
  mkdir -p "$out"
  echo "[sweep] === training λ_ce = $lam ==="
  PYTHONPATH=src python -m elf_mas_pt.training.train_cola \
    --dataset_dir "$DATASET" \
    --variant full --batch_size 8 --num_steps 10000 --lr 1e-4 --seed 0 \
    --S_q 16 --S_ctx 64 --log_every 200 --prefix_cond \
    --lora_mode --lora_layers 8,12,16,20 --lora_inner 512 --lora_heads 8 \
    --lambda_ce "$lam" \
    --out_dir "$out" 2>&1 | tee "${out}/stdout.log"
  echo "[sweep] === eval λ_ce = $lam ==="
  PYTHONPATH=src python -m elf_mas_pt.eval.eval_cola_mas \
    --ckpt_dir "$out" --dataset_dir "$DATASET" --split eval \
    --n_items 200 --batch_size 8 --T_inf 16 \
    --variant full --lora_mode 2>&1 | tee "${out}/eval_stdout.log"
done
echo "[sweep] DONE"
