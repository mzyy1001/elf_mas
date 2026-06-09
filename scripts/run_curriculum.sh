#!/usr/bin/env bash
# Randomized-KG curriculum probe: induce genuine hop-1 (agent-A) usage on the
# shortcut-free task by warming up on single-hop retrieval first, then composing.
#   Phase 1: train on A-only + B-only only (single-hop retrieval warm-up).
#   Phase 2: resume, train on the full mix (composition).
#   Then evaluate the lesion matrix (does Da become > 0?).
# Parametrized via env vars: LAYERS, P1_STEPS, P2_STEPS, TAG.
set -euo pipefail

DS="${DS:-/home/chenhongrui/elf_mas/data/synth_2fact_randomkg_v1}"
ROOT="${ROOT:-/home/chenhongrui/elf_mas/runs/cola_mas_randomkg}"
LAYERS="${LAYERS:-8,12,16,20}"
P1_STEPS="${P1_STEPS:-15000}"
P2_STEPS="${P2_STEPS:-60000}"
TAG="${TAG:-curr}"
RECIPE="--lora_mode --lora_layers $LAYERS --lambda_ce 0.3 --t_distribution uniform \
        --batch_size 8 --lr 1e-4 --seed 0 --S_q 16 --S_ctx 64 --log_every 500 --prefix_cond --variant full"

echo "[$TAG] === Phase 1: single-hop warm-up (A-only+B-only), $P1_STEPS steps, layers=$LAYERS ==="
PYTHONPATH=src python -m elf_mas_pt.training.train_cola --dataset_dir "$DS" $RECIPE \
    --buckets "A-only,B-only" --num_steps "$P1_STEPS" --out_dir "$ROOT/${TAG}_phase1"

echo "[$TAG] === Phase 2: resume on full mix, $P2_STEPS steps ==="
PYTHONPATH=src python -m elf_mas_pt.training.train_cola --dataset_dir "$DS" $RECIPE \
    --num_steps "$P2_STEPS" --resume_from "$ROOT/${TAG}_phase1/lora_state.pt" --out_dir "$ROOT/${TAG}_phase2"

echo "[$TAG] === Lesion eval ==="
PYTHONPATH=src python -m elf_mas_pt.eval.eval_cola_mas --ckpt_dir "$ROOT/${TAG}_phase2" \
    --dataset_dir "$DS" --split eval --n_items 200 --batch_size 8 --T_inf 16 \
    --S_q 16 --S_ctx 64 --variant full --lora_mode --prefix_cond

echo "[$TAG] CURRICULUM_DONE"
