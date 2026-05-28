# Phase 3 — Cola+MAS lesion result (2026-05-24)

## Headline

First demonstration of role-specific MAS specialization on a frozen Cola-DLM
(1.8B) backbone. AB-bucket per-token EM: **58.5% (full) → 0.0% (A_only) → 15.9% (B_only) → 0.0% (Neither)**
on `synth_2fact_v1` dev (n=200).

| Bucket  | AB    | A_only | B_only | Neither | Δa     | Δb     |
|---------|-------|--------|--------|---------|--------|--------|
| AB      | 58.5% |   0.0% |  15.9% |    0.0% | +42.7pp| +58.5pp|
| B-only  | 35.8% |   0.0% |   5.7% |    0.0% | +30.2pp| +35.8pp|
| A-only  |  0.0% |   0.0% |   0.0% |    0.0% |   —    |   —    |
| neither | 64.7% |  29.4% |   0.0% |    0.0% | +64.7pp| +35.3pp|

Δa = drop when ctx_a is removed (B_only ablation), Δb = drop when ctx_b is removed (A_only ablation).
Sampling: T_inf=16 Euler steps, prefix-conditioned Cola DiT with KV-cache priming.

## What worked

Three changes had to land together — ablating any one regressed eval EM to 0%:

1. **LoRA in-block MAS** (`mas_in_block.py`). Per-agent cross-attn injected
   inside Cola DiT at layers `[8, 12, 16, 20]`. inner_dim=512, 8 heads, zero-init
   out_proj, gate bias=-2 (sigmoid≈0.12). Cola DiT + VAE fully frozen.
   ~25M trainable params total. Earlier tail-MAS heads (8–27M) appended to v_0
   could not push past Cola's "format-shaped continuation" attractor.

2. **Uniform t-sampling** (`--t_distribution uniform`). The ELF default was
   logit-normal centered at p_mean=-1.5, meaning training t was mostly ~0.18.
   Inference starts from t=1 (pure noise) — entirely OOD. Switching to
   uniform[0.05, 0.95] gives equal coverage of high-noise cases where the
   model must do real work to recover the target latent.

3. **CE auxiliary loss** (`--lambda_ce 0.3`). Token-CE on x-prediction
   `z_clean_pred = z_t - t * v_pred` decoded through the frozen Cola VAE.
   Velocity MSE alone produced type-correct latents (the answer's category)
   but not token-correct ones (the specific answer); CE pressures decode-
   correctness directly. λ=0.3 was the sweet spot of the {0.1, 0.3, 1.0} sweep.

Plus prefix conditioning (Q-only SQuAD-template; ctx_a/ctx_b enter only through
MAS heads), 30k steps × batch 8 × lr 1e-4. ~100 min on chen GPU 5 (A100 80GB).

## The decode-strip artifact (read before re-evaluating any prior checkpoint)

Every Phase 3 eval before this one reported 0% text-EM. That number was a
**metric artifact**, not a model failure. The pad-region latents (positions 1–15
of the 16-position answer block) decode to non-pad content tokens like `" Tor"`,
`" all"`, `" on"`. The Cola tokenizer joins these onto the answer with no space,
so `"Belgium" + "Tor"` becomes `"BelgiumTor"`, which SQuAD-style normalize fails.

Mitigation: use `per_token_em` / `per_token_f1` (in `eval_cola_mas.py`): tokenize
gold, take the first K predicted token IDs, compare ID lists directly. This is
how the 58.5% number above was computed. Text-EM on the same checkpoint is 0%.

Oracle VAE roundtrip is 100% EM on synth and 99.5% on MuSiQue with this metric —
the VAE is reliable; it's just our text-decode-then-normalize pipeline that
collapses the signal.

## Reproduction

```bash
# Training (~100 min on chen GPU 5)
PYTHONPATH=src python -m elf_mas_pt.training.train_cola \
    --dataset_dir /home/chenhongrui/elf_mas/data/synth_2fact_v1 \
    --variant full --batch_size 8 --num_steps 30000 --lr 1e-4 --seed 0 \
    --S_q 16 --S_ctx 64 --log_every 500 --prefix_cond \
    --lora_mode --lora_layers 8,12,16,20 --lora_inner 512 --lora_heads 8 \
    --lambda_ce 0.3 --t_distribution uniform \
    --out_dir runs/cola_mas_pilot/b0_lora_uniform_lce0.3_30k

# Eval (~2 min for 200 items × 4 ablations)
PYTHONPATH=src python -m elf_mas_pt.eval.eval_cola_mas \
    --ckpt_dir runs/cola_mas_pilot/b0_lora_uniform_lce0.3_30k \
    --dataset_dir /home/chenhongrui/elf_mas/data/synth_2fact_v1 \
    --split eval --n_items 200 --batch_size 8 --T_inf 16 \
    --variant full --lora_mode
```

## Qualitative examples (AB-bucket, full ablation)

| Q | gold | gold token IDs | pred (first K+2) | match |
|---|---|---|---|---|
| What country does Hank live in? | Belgium | [22404, 90339] | [22404, 90339, 49075, 9822] | ✓ |
| What country does Adam live in? | Ireland | [40, 87566] | [40, 87566, 689, 3723] | ✓ |
| What country does Yael live in? | Finland | [9312, 1974] | [9312, 1974, 689, 264] | ✓ |
| What is the capital of Norway? | Reykjavik | [697, 73640, 62559, 1609] | [697, 66, 62559, 1609, ...] | ✗ (token 1 differs) |

## What's next (Phase 3.5 onward)

1. **Variant sweep**: train identical_ctx (B6), context_shuffled (B7),
   frozen_agents (B2), and matched single-head (B4) using the SAME recipe.
   Expectation: B0 keeps the asymmetry; B6/B7/B2 collapse it; B4 underperforms.
2. **MuSiQue with the same recipe**: oracle VAE on MuSiQue is 99.5%, recipe is
   substrate-agnostic. Expect lower absolute EM (MuSiQue answers are more
   diverse than synth's closed country set) but the asymmetry pattern should
   still emerge.
3. **Why A-only and B-only buckets underperform**: the answer is a single
   word in one specific sentence within ctx_a or ctx_b. Unlike AB-bucket's
   regular two-hop pattern ("X lives in Y" + "Y is in Z"), 1-hop retrieval
   from a paragraph is genuinely harder for this architecture. Likely
   addressable by longer training or larger LoRA inner_dim.

Checkpoint: `runs/cola_mas_pilot/b0_lora_uniform_lce0.3_30k/lora_state.pt`
(includes the 4 wrappers' mas_a/mas_b state dicts; layer indices + hyperparams
are stored in the same file for portable loading).
