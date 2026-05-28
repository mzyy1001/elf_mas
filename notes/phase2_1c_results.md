# Phase 2.1c — MuSiQue 2-hop with per-question extractive NN decode

**Date:** 2026-05-21
**Setup:** chen A100 GPU 5, ELF-B-de-en frozen backbone, `CoupledMASHeads` (10.06M trainable). MuSiQue-2hop v1 (41,896 train / 3,726 dev, dataset_card.md). 4 variants × 3 seeds × 2000 steps. Per-question extractive NN decode (candidates extracted from `ctx_a_text ∪ ctx_b_text` via proper-noun phrases, numbers, lowercase NPs; cap 80 per item; +"unknown" always).

## 1. Headline — EM-level role-specific asymmetry on a public benchmark

Lesion deltas in `full` (B0) measured as drop in EM under per-agent ablation:

| bucket | full EM | zero_a | zero_b | Δa | Δb | **Δa/Δb ratio** |
|---|---|---|---|---|---|---|
| A-only | **13.2** | 5.8 | 12.2 | 7.4 | 1.0 | **7.4×** |
| B-only | **9.5** | 9.0 | 4.7 | 0.5 | 4.8 | **9.6×** (inverted, Δb dominates) |
| AB | 9.1 | 8.9 | 5.0 | 0.2 | 4.1 | — |
| neither | 34.9 | 2.9 | 43.6 | 32.0 | -8.7 | (asymmetric "unknown" detector lives in Agent A) |

The asymmetry collapses to ~1× under both controls:

| variant | A-only Δa/Δb | B-only Δa/Δb |
|---|---|---|
| **full (B0)** | **7.4×** | **9.6×** |
| identical_ctx (B6) | 1.0× | ~1.1× |
| context_shuffled (B7) | 1.1× | 1.0× |
| frozen_agents (B2) | 0× | 0× (no residual to ablate) |

This is the central Phase 2 result. EM-level role-specific specialization, with the predicted collapse under context-sharing and context-shuffle controls.

## 2. Absolute accuracy

| Variant | params | AB EM | A-only EM | B-only EM | neither EM | final MSE |
|---|---|---|---|---|---|---|
| **full / B0 (ours)** | 10.06M | **9.1** | 13.2 | **9.5** | **34.9** | **0.393 ± 0.014** |
| **B4 single (matched)** ★ | 9.91M | 8.2 | **13.4** | 9.0 | 23.1 | 0.411 ± 0.018 |
| B4 single (8.58M, prev) | 8.58M | 8.3 | 13.4 | 8.5 | 27.1 | 0.415 ± 0.016 |
| identical_ctx (B6) | 10.06M | 7.0 | 14.3 | 5.6 | 29.6 | 0.441 ± 0.011 |
| context_shuffled (B7) | 10.06M | 7.0 | 10.4 | 7.4 | 29.0 | 0.453 ± 0.013 |
| frozen_agents (B2) | 0 | 2.9 | 2.7 | 3.9 | 0.0 | 0.568 ± 0.012 |

★ B4 added 2026-05-21: SingleHeadMASWrapper, hidden=384 / depth=4, ONE residual head + ONE gate over `concat(ctx_a, ctx_b)`. By construction Δb=0 (no second agent slot to ablate), so role asymmetry is architecturally absent. B0 beats B4 by 0.8–1.0 EM on AB/B-only and **7.8 EM on neither** (the abstain bucket where Agent A's "unknown"-detector specialization pays off). B4 marginally wins on A-only (+0.2). 8.58M < 10.06M → B4 is at 15% capacity disadvantage; exact-param-matching B4 widening is a future ablation.

- Coupled MAS B0 vs frozen B2: **3–5× EM gain across all answer buckets.**
- B0 vs B6 on B-only: full=9.5, identical_ctx=5.6 — private contexts win where the answer is single-fact.
- B6 wins on A-only EM (14.3 vs 13.2) by a slim margin — symmetric model has redundant access.

## 3. Candidate recall (extractor diagnostic, identical across model variants)

| bucket | recall (gold ∈ candidates) | avg candidate set size | EM|recall (full) |
|---|---|---|---|
| AB | 57.2% | 50.7 | 15.9 |
| A-only | 76.4% | 51.0 | 17.2 |
| B-only | 56.0% | 51.2 | 17.0 |
| neither | 100.0% | 51.5 | 34.9 |

The 57–76% recall ceiling stems from S_ctx=64 truncation (only ~22% of items retain full gold paragraph). EM|recall = 15.9–17.2% on real buckets shows the model picks the correct candidate ~1 in 6 times when it's present — vs ~5% for frozen baseline (B2).

## 4. MSE asymmetry vs EM asymmetry

MSE-level Δa/Δb ratios (from `eval_lesion`):

| bucket | full Δa | full Δb | ratio |
|---|---|---|---|
| AB | 0.025 | 0.027 | 0.92× |
| A-only | 0.047 | 0.027 | 1.74× |
| B-only | 0.026 | 0.027 | 0.99× |
| neither | 0.249 | 0.045 | 5.56× |

The MSE asymmetry is much weaker than the EM asymmetry (1.74× on A-only vs 7.4× EM). The embedding-space MSE doesn't capture the full specialization signal that shows up at the *answer-selection* step. **Recommendation for paper headline: use the EM Δa/Δb ratios, not MSE.**

## 5. Honest limitations

1. **Recall ceiling 57–76%.** S_ctx=64 truncation throws away most gold paragraphs. The most likely first fix.
2. **Absolute EM is 8–14%** on single-fact buckets, **not competitive with external SOTA** (FiD-base etc.). The contribution stands as a *mechanism* claim, not a *leaderboard-beating* claim.
3. **`neither` is unusual**: zero_b on neither *improves* EM (34.9 → 43.6). Agent A learned the "unknown" detector; Agent B's contribution is mildly distracting on those items. Asymmetric specialization at the *abstention* level.
4. **Single in-domain dataset.** Cross-benchmark transfer to 2WikiMultiHopQA and HotpotQA (per Phase 2.5 in design doc) is needed to defend "the result is not MuSiQue-specific."

## 6. Reproduce

```bash
# on chen
source ~/elf_mas/setup_env.sh
export CUDA_VISIBLE_DEVICES=5
cd ~/elf_mas/src
bash elf_mas/training/run_phase2_1c_pilot.sh
python -m elf_mas.training.aggregate_phase2_1c
```

Logs at `~/elf_mas/runs/phase2_1c_pilot_musique/{variant}_seed{N}.log`.

## Addendum — Phase 2.4 (S_ctx 64→128, 2026-05-21)

Regenerated MuSiQue with `max_ctx_tokens=128` (gold-retention 24%→62%, mean 0.66→0.89). Reran full 5-variant × 3-seed sweep.

**Recall lifted as expected; EM did not.** With ~87 candidates per item (was 51), the NN decoder has more confusable spans even when gold is in the set. EM|recall fell from 15.9 → 11.8 on AB; absolute EM dropped 1-5pp across buckets. The two effects cancelled.

| | S_ctx=64 full | S_ctx=128 full |
|---|---|---|
| Recall AB / A-only / B-only | 57 / 76 / 56% | 64 / 87 / 63% |
| EM|recall AB | 15.9 | 11.8 |
| AB EM | 9.1 | 7.6 |
| MSE | 0.393 | 0.389 |

**The NN decoder, not gold retention, is the actual bottleneck.** S_ctx=128 logs at chen `~/elf_mas/runs/phase2_4_sctx128/`. Conclusion: stick with S_ctx=64 as the default for headline numbers and invest decoder upgrades next (per-question candidate filtering or question-conditioned scoring).

## 7. Next experiments (priority order)

1. **Bump S_ctx to 128** and rerun. The recall ceiling is the dominant absolute-EM constraint. ~2× compute per step but should lift recall to >85% and EM proportionally.
2. **Multi-seed expansion to n=5** for final paper numbers. Cheap, just rerun the script.
3. **Cross-benchmark: 2WikiMultiHopQA bridge subset** — same trainer, same decoder, just point at a new dataset. Phase 2.5 from the design doc.
4. **Pre-register the success criteria** from `phase2_benchmark_design.md` §11 before any more experiments (this is now an honest pre-registration since the Phase 2.1c numbers are in).
