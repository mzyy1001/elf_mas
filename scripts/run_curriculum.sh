#!/usr/bin/env bash
# Randomized-KG curriculum probe: can genuine hop-1 (agent-A) usage be induced on
# the shortcut-free task by warming up on single-hop retrieval first?
#   Phase 1: train on A-only + B-only only (single-hop retrieval warm-up).
#   Phase 2: resume, train on the full mix (composition).
#   Then evaluate the lesion matrix (does Da become > 0?).
set -euo pipefail

DS="${DS:-/home/chenhongrui/elf_mas/data/synth_2fact_randomkg_v1}"
ROOT="${ROOT:-/home/chenhongrui/elf_mas/runs/cola_mas_randomkg}"
RECIPE="--lora_mode --lora_layers 8,12,16,20 --lambda_ce 0.3 --t_distribution uniform \
        --batch_size 8 --lr 1e-4 --seed 0 --S_q 16 --S_ctx 64 --log_every 200 --prefix_cond --variant full"

echo "[curr] === Phase 1: single-hop warm-up (A-only + B-only), 15k steps ==="
PYTHONPATH=src python -m elf_mas_pt.training.train_cola --dataset_dir "$DS" $RECIPE \
    --buckets "A-only,B-only" --num_steps 15000 --out_dir "$ROOT/curr_phase1"

echo "[curr] === Phase 2: resume on full mix, 30k steps ==="
PYTHONPATH=src python -m elf_mas_pt.training.train_cola --dataset_dir "$DS" $RECIPE \
    --num_steps 30000 --resume_from "$ROOT/curr_phase1/lora_state.pt" --out_dir "$ROOT/curr_phase2"

echo "[curr] === Lesion eval on curriculum model ==="
PYTHONPATH=src python -m elf_mas_pt.eval.eval_cola_mas --ckpt_dir "$ROOT/curr_phase2" \
    --dataset_dir "$DS" --split eval --n_items 200 --batch_size 8 --T_inf 16 \
    --S_q 16 --S_ctx 64 --variant full --lora_mode --prefix_cond

echo "[curr] CURRICULUM_DONE"
