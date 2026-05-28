---
name: project-phase3-breakthrough
description: "Phase 3 (Cola+MAS) reached lesion asymmetry on 2026-05-24 — AB-bucket 58.5% per-token EM with clean drops to A_only=0, B_only=15.9, Neither=0. The recipe that worked, and the artifact that hid earlier signal."
metadata: 
  node_type: memory
  type: project
  originSessionId: a4fa26be-d662-41d5-a043-71652dcaf996
---

Phase 3 (Cola+MAS) reached its first clear demonstration of role-specific MAS specialization on `synth_2fact_v1` on 2026-05-24.

**Headline result** (synth_2fact_v1 dev, 200 items, per-token-ID EM):

| Bucket | AB | A_only | B_only | Neither |
|---|---|---|---|---|
| AB (2-hop) | **58.5%** | 0.0% | 15.9% | 0.0% |
| B-only | 35.8% | 0.0% | 5.7% | 0.0% |
| neither | 64.7% | 29.4% | 0.0% | 0.0% |

AB-bucket: Δa=42.7pp, Δb=58.5pp, ratio≈0.73. MAS-B (city→country fact) is more critical than MAS-A (person→city fact), consistent with the question form ("What country does X live in?" — final step needs country fact).

**Why:** the user wants empirical evidence of trained-end-to-end role-specific specialization on a frozen LLM substrate. Phase 1 (synth) and Phase 2 (MuSiQue on ELF) both showed asymmetry; Phase 3 was meant to scale to a "bigger diffusion" (Cola-DLM, 1.8B params). After many iterations, the working setup is now demonstrated.

**How to apply:** if extending to MuSiQue / other tasks or running the variant sweep (B6 identical_ctx / B7 context_shuffled / B2 frozen_agents / B4 single_model), reuse the EXACT recipe below — deviations are likely to reintroduce 0% EM.

**The decisive recipe** (all three together; ablating any one regressed to 0% EM in earlier runs):

1. **LoRA in-block MAS** — `MASBlockWrapper` patches Cola DiT at layers `[8,12,16,20]`. Per-agent cross-attn (inner_dim=512, 8 heads) injected on the answer-block forward only. Cola DiT + VAE fully frozen. ~25M trainable params.
2. **Uniform t-sampling** (`--t_distribution uniform`) — fixes the train/eval distribution gap. Logit-normal (p_mean=-1.5) sampled t mostly near 0.18, but inference starts from pure noise (t=1) which was OOD for the model.
3. **CE auxiliary loss** (`--lambda_ce 0.3`) — token-CE on x-prediction `z_clean_pred = z_t - t*v_pred` decoded through the frozen VAE. Pressures decode-correct latents, not just type-correct.

Plus prefix conditioning (Q-only SQuAD-template prompt; ctx_a/ctx_b enter only through MAS heads) and 30k steps × batch 8 × lr 1e-4. ~100 min on chen GPU 5 (A100 80GB).

**The decode-strip artifact** — important to know for any future eval. Text-level `tokenizer.decode(toks)` joins the answer token with the next pad-region junk token (e.g., `"Belgium"+"Tor"→"BelgiumTor"`), so SQuAD-style normalize_answer produces 0% EM even when position 0 is exactly right. **Use `per_token_em` / `per_token_f1` (per-token-ID first-K match) as the primary metric**; text-level normalize is misleading.

Files: `src/elf_mas_pt/model/mas_in_block.py` (LoRA wrappers), `src/elf_mas_pt/training/train_cola.py` (trainer with `--lora_mode`, `--lambda_ce`, `--t_distribution`), `src/elf_mas_pt/eval/eval_cola_mas.py` (eval with per-token metrics).

Checkpoint: `/home/chenhongrui/elf_mas/runs/cola_mas_pilot/b0_lora_uniform_lce0.3_30k`.

Related: [[project-elf-mas]], [[project-phase1-results]].
