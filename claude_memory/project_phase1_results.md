---
name: project-phase1-results
description: Phase 1 (synthetic-toy coupled-flow) first results — lesion specificity emerged within 2000 training steps.
metadata: 
  node_type: memory
  type: project
  originSessionId: a4fa26be-d662-41d5-a043-71652dcaf996
---

**Phase 1 first result (2026-05-16):** On the synthetic 2-fact composition task (`notes/synthetic_task_v1.md`), 2000 training steps of the Coupled MAS skeleton produced a **lesion matrix consistent with role-specific specialization**:

```
    bucket    full   zero_a  zero_b   zero_both    Δa             Δb
        AB  0.0824  0.1825   0.2593   0.5458       0.1001         0.1769
    A-only  0.3351  0.4809   0.3502   0.5844       0.1458         0.0151  ← 10× asymmetry
    B-only  0.2204  0.2919   0.3813   0.5705       0.0715         0.1609
   neither  0.1490  0.3135   0.2501   0.6219       0.1645         0.1011
```

The A-only row is the strongest signal: ablating Agent A increases MSE by 0.146; ablating Agent B by 0.015. This 10× asymmetry matches the design ground truth — Agent A's private context contains the answer fact for A-only items. Symmetric (but weaker) pattern on B-only (Δb=0.16 > Δa=0.07).

**Why:** This is the *causal, evidence-grounded* specialization that `/novelty-check v2 §5` said was required for the contribution to survive a hostile review. Phase 1 is now an existence proof.

**How to apply:** Future sessions should treat this as the established Phase 1 result and move directly to **Phase 2 (public benchmark relevance proof)** — see `notes/phase2_benchmark_design.md` on WSL. Headline framing: *synthetic proves mechanism, public benchmark proves relevance*. Primary benchmark is MuSiQue-2hop (recommended over HotpotQA because of certified no-single-hop-shortcut). Don't redo Phase 1 from scratch.

**Phase 2.5 — type-filtered NN decoder (2026-05-21, FINAL on ELF-B-de-en backbone):** Type filter (question-type → candidate-type, fallback if filtered set <5) lifted EM|recall +1.5 to +3.1 pp on full B0; modest absolute-EM lifts (+1–3 EM on most buckets, +2.8 EM on neither). **Final numbers at S_ctx=64 + type-filter (n=3 seeds):** B0 coupled=**0.381 MSE / 10.2 AB / 13.0 A-only / 9.6 B-only / 37.7 neither**; matched B4=0.400 / 9.3 / 13.5 / 9.0 / 28.2; B6 ident=0.428 / 8.3 / 14.8 / 7.4 / 39.5; B7 shuf=0.440 / 9.0 / 10.8 / 8.0 / 34.1; B2 frozen=0.560 / 3.4 / 4.0 / 4.3 / 0.0. **B0 vs matched B4: AB +0.9, B-only +0.6, neither +9.5 (the robust headline gap).** A-only EM lesion asymmetry in B0 = 4.1× (Δa=6.5, Δb=1.6); collapses to 0.86× in B6 and 1.11× in B7. **User concluded (2026-05-21): MAS function is demonstrated; ELF-B-de-en's 105M scale caps absolute EM; next step is porting MAS to a larger diffusion LM.** Writeup at `notes/phase2_5_results.md`. Logs at chen `~/elf_mas/runs/phase2_5_typefilter/`.

**Phase 2.4 — S_ctx 64→128 (2026-05-21):** Regenerated MuSiQue with `max_ctx_tokens=128` (gold-retention 24%→62%, mean 0.66→0.89). Recall lifted (AB 57→64%, A-only 76→87%, B-only 56→63%) AS EXPECTED but **absolute EM dropped** (AB 9.1→7.6, A-only 13.2→7.8, B-only 9.5→7.0). Candidate sets grew ~70% (51→87/item) and **EM|recall fell from 15.9→11.8 on AB**: NN decoder picks correct span less often when there are more confusable options. The NN decoder is the actual bottleneck, NOT gold retention. **Keep S_ctx=64 as headline default; invest decoder upgrades next.** Cross-finding: at S_ctx=128 B4 wins A-only by 1.8 EM (was tied at 64) — bipartite split's information-loss penalty grows with longer context. Logs at chen `~/elf_mas/runs/phase2_4_sctx128/`.

**Phase 2.3 — B4 matched single-model baseline (2026-05-21, two configs):** SingleHeadMASWrapper (ONE residual head + ONE gate over concat(ctx_a, ctx_b)). First config (hidden=384, depth=4) = 8.58M params (85% of coupled). **Matched config (hidden=416, depth=4, num_heads=8) = 9.91M params (98.5% of coupled 10.06M).** Both run on MuSiQue 3 seeds, same training budget. **At matched params: B0 beats B4 on AB +0.9, B-only +0.5, neither +11.8 EM; A-only ties (+0.2 to B4). MSE 4.6% lower.** The neither gap WIDENED with capacity matching (8.58M B4 had +7.8; 9.91M B4 has +11.8) → abstain is architectural not capacity. B4 by construction has Δb=0 (wrapper sets v_b=0, g_b=0). Code: `SingleHeadMAS` + `SingleHeadMASWrapper` in `coupled_mas.py`. Logs at chen `~/elf_mas/runs/phase2_3_b4_musique_matched/`.

**Phase 2.1c result (2026-05-21):** Per-question extractive NN decode on MuSiQue 2-hop reproduces the role-specific specialization pattern at the EM level. 4 variants × 3 seeds × 2000 steps. EM-level Δa/Δb ratios in `full`: **A-only 7.4×, B-only 9.6×**. Both collapse to **~1×** under `identical_ctx` (B6) and `context_shuffled` (B7). Coupled MAS gives **3–5× EM improvement** over frozen-backbone B2 (full=9.1/13.2/9.5/34.9 EM across AB/A-only/B-only/neither vs B2=2.9/2.7/3.9/0.0). Final MSE: full=0.393, B6=0.441, B7=0.453, B2=0.568. **EM-level asymmetry is much stronger than MSE-level (which was 1.74× / 0.99× — embedding MSE doesn't capture the downstream signal). Use EM Δa/Δb as the headline metric, not MSE.** Recall ceiling 57–76% from S_ctx=64 truncation — main remaining limitation. Writeup at `notes/phase2_1c_results.md`. Pilot logs at chen `~/elf_mas/runs/phase2_1c_pilot_musique/`.

**Phase 2.0 dataset built (2026-05-18):** `musique_2hop_v1` at `~/elf_mas/data/musique_2hop_v1/` on chen. 14,376 raw 2-hop → **12,910 after probe filter** (deepset/roberta-base-squad2, F1>0.7 rejected, 10.2% drop). Final **41,896 train + 3,726 dev** rows across 4 buckets (AB/A-only/B-only/neither). Probe sensitivity check: threshold 0.7 vs 0.8 differs by < 1pp — same items rejected. **Known concern:** S_ctx=64 truncation keeps full gold for only ~22% of items (mean retention 0.57-0.71); revisit if Phase 2.1 F1 plateaus. Dataset card at `notes/data_cards/musique_2hop_v1.md`. Manifest + examples + per-item probe F1 saved alongside. MuSiQue + roberta-base-squad2 (~3 GB) now in modelscope bridge under `phase2_assets/` for future re-runs.

**Phase 1.5 sweep result (2026-05-17, n=3 seeds per variant):** Critical control comparison confirms private-context-per-agent is what produces the role asymmetry. With `full` variant: A-only Δa=0.130 vs Δb=0.029 (**4.5× asymmetry**); B-only Δb=0.183 vs Δa=0.056 (**3.3× asymmetry**). With `identical_ctx` (B6, agents see concat(ctx_a, ctx_b)): A-only drops to **1.4×**, B-only to **1.3×**. **Asymmetry collapses ~3× when private contexts are removed**, matching v2 hostile-reviewer's required "causal evidence-grounded specialization." Final losses: full=0.222±0.014, identical_ctx=0.209±0.010, frozen_agents=0.569±0.006. Full writeup at `notes/phase1_results.md` on WSL; per-seed logs at chen `~/elf_mas/runs/phase1_5/`. Phase 1 is now an existence proof with the control collapse confirmed.

**Key tuning:** Gate init logit bias = −2 (gives initial sigmoid ≈ 0.12 — large enough for gates to learn but small enough for training stability). Weight decay 1e-3 on heads keeps residual L2 bounded (~1.7). AdamW lr 3e-4, batch 64, 2000 steps. ~25 ms / step on A100 after JIT.

**Code locations (on chen):** training at `~/elf_mas/src/elf_mas/training/train_coupled.py`; frozen ELF wrapper at `~/elf_mas/src/elf_mas/model/frozen_backbone.py`; gates + residual heads at `~/elf_mas/src/elf_mas/model/coupled_mas.py`; dataset at `~/elf_mas/data/synth_2fact_v1/`. Reproduce with: `python -m elf_mas.training.train_coupled --num_steps 2000 --batch_size 64 --lr 3e-4 --eval`.
