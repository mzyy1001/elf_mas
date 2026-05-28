# Phase 1 results — Coupled Multi-Agent Language Flows on synth_2fact_v1

**Date:** 2026-05-16 / 2026-05-17
**Setup:** chen A100 GPU 5, ELF-B-de-en frozen backbone (105M params), `CoupledMASHeads` trainable (10.06M params), `synth_2fact_v1` 50K train + 5K eval (`notes/synthetic_task_v1.md`). AdamW lr 3e-4, batch 64, 2000 steps, weight decay 1e-3, gate logit bias = −2. ~65 s per run incl. JIT.

## 1. Headline result — specialization is private-context-driven

Per-variant lesion matrix (mean ± std across 3 seeds; eval = 25 batches × 64 items):

### `full` (B0, ours): private `ctx_A` to Agent A only, `ctx_B` to Agent B only

| bucket | full | zero_a | zero_b | zero_both | Δa | Δb |
|---|---|---|---|---|---|---|
| AB     | 0.092±0.008 | 0.183±0.003 | 0.290±0.028 | 0.548±0.002 | **0.092±0.011** | **0.199±0.021** |
| A-only | 0.335±0.004 | 0.465±0.017 | 0.364±0.015 | 0.585±0.001 | **0.130±0.016** | 0.029±0.014 |
| B-only | 0.218±0.005 | 0.274±0.014 | 0.401±0.020 | 0.569±0.002 | 0.056±0.015 | **0.183±0.016** |
| neither | 0.142±0.023 | 0.322±0.002 | 0.241±0.023 | 0.622±0.001 | 0.180±0.024 | 0.098±0.008 |

### `identical_ctx` (B6 ablation): both agents see `concat(ctx_A, ctx_B)`

| bucket | full | zero_a | zero_b | zero_both | Δa | Δb |
|---|---|---|---|---|---|---|
| AB     | 0.043±0.001 | 0.196±0.045 | 0.206±0.059 | 0.548±0.002 | 0.153±0.044 | 0.164±0.060 |
| A-only | 0.322±0.008 | 0.415±0.014 | 0.387±0.021 | 0.585±0.001 | 0.093±0.021 | 0.065±0.013 |
| B-only | 0.224±0.006 | 0.323±0.026 | 0.351±0.030 | 0.569±0.002 | 0.099±0.021 | 0.127±0.035 |
| neither | 0.120±0.020 | 0.298±0.028 | 0.233±0.038 | 0.622±0.001 | 0.177±0.047 | 0.113±0.019 |

### Asymmetry-ratio comparison

| Bucket | `full` Δa/Δb | `identical_ctx` Δa/Δb | Δratio (full−id) |
|---|---|---|---|
| A-only | **0.130 / 0.029 = 4.50×** | 0.093 / 0.065 = 1.43× | **3.07×** |
| B-only | 0.056 / 0.183 = 1/3.28× → **3.28× for Δb** | 0.099 / 0.127 = 1/1.28× | **2.00×** |

**Interpretation.** With private contexts, ablating an agent hurts ~4× more when its private context contained the answer than when the other agent's did. With shared contexts, this asymmetry shrinks to ~1.4×. The specialization signature is **caused by the private-context inductive bias**, not by the architecture alone.

### `frozen_agents` (B2 ablation): residual head params frozen at zero, only gates train

All lesion deltas = 0 (no residual to ablate). Final loss = 0.569 ± 0.006 (= unconditional v_0 flow-matching loss). Confirms our coupled-MAS heads contribute the **~60% loss reduction** from 0.569 → 0.222 in the `full` variant.

## 2. Loss across variants (n=3 seeds each, 2000 steps)

| Variant | Final loss |
|---|---|
| `full` (B0) | **0.222 ± 0.014** |
| `identical_ctx` (B6) | 0.209 ± 0.010 |
| `frozen_agents` (B2) | 0.569 ± 0.006 |

Important: `identical_ctx` has slightly *lower* loss than `full`. Agents with more info do better on average. **The specialization claim survives because the *role asymmetry* under lesion collapses in B6**, not because B6 is worse on the answer-prediction objective. This is the right kind of result: B6 is a "stronger" model that nonetheless fails the role-specificity test.

## 3. What this means for the v0.3 thesis

- ✅ **Training-induced specialization emerges.** The mechanism (coupled velocity fields with private contexts, trained end-to-end) produces causal, evidence-grounded role specialization within 2000 steps.
- ✅ **Private-context-per-agent is the load-bearing ingredient.** The B6 control collapses the asymmetry, ruling out "any MAS architecture would show this."
- ✅ **Multi-seed stability.** σ of Δa, Δb is small (0.01–0.05) relative to the asymmetry signal (0.03–0.20).
- ⚠️ **`neither` bucket still confabulates.** Both agents register large lesion drops there. Needs explicit "unknown" supervision or longer training before we claim the model abstains correctly.
- ⚠️ **Loss-level metric only.** No nearest-neighbor accuracy decode yet. Phase 1.6 work.

## 4. What's NOT yet tested (deferred)

| Variant | Tests | Status |
|---|---|---|
| B1 (stop-grad between flow steps) | End-to-end backprop through trajectory matters | Needs an unrolled denoising trainer (current is single-step); deferred to Phase 1.7 |
| B3 (sequential latent-passing CIPHER/LatentMAS-style) | Our coupled-flow structure vs sequential passing | Needs a different architecture; Phase 1.8 |
| B4 (matched single ELF on full ctx) | Multi-agent decomposition matters at all | Roughly approximated by frozen_agents (= v_0 alone); proper B4 needs to train a single ELF with question+ctx_a+ctx_b condition |
| B5 (text-channel MAS) | Differentiable channel matters | Out of scope for v1 |
| B7 (context-shuffled) | Specialization is not trivial from input segregation | Cheap to add — Phase 1.6 |

## 5. Repro

On chen, GPU 5:
```bash
source ~/elf_mas/setup_env.sh && export CUDA_VISIBLE_DEVICES=5
cd ~/elf_mas/src
bash elf_mas/training/run_phase1_5.sh       # 9 runs, ~16 min
python -m elf_mas.training.aggregate_phase1_5
```

Raw per-run logs at `~/elf_mas/runs/phase1_5/{variant}_seed{N}.log`. Aggregated table at `~/elf_mas/runs/phase1_5/aggregated.txt`.

## 6. Next experiments (in order)

1. **B7 context-shuffled** — agents get private contexts but assigned to wrong agents. Should also collapse the asymmetry. Cheap (data swap).
2. **Nearest-neighbor decode** — per-bucket accuracy, not just MSE. Adds substance to the specialization claim.
3. **Longer training + `unknown` supervision** — clean up the `neither` bucket.
4. **B4 proper** — train a matched-FLOPs single ELF on full condition. Need to fork ELF's train.py or add a "no-agents" variant to our trainer.
5. **B1 stop-grad** — requires an unrolled K-step rollout trainer.
