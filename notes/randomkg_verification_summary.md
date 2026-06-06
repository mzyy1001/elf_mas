# Randomized-KG control: verification + what it reveals

**Date:** 2026-06-05 · dataset `data/synth_2fact_randomkg_v1` (train 20k / eval 3k) ·
B0 trained 30k with the standard recipe → `runs/cola_mas_randomkg/full_seed0`.

## Why we built it
On the original `synth_2fact_v1` the KG is deterministic (person→fixed city→fixed country),
so the answer is recoverable from the **person name in the question prefix**. A tuned-lens
probe hit ~100% with **no context** — a memorization shortcut that confounds any
mechanistic claim. The randomized-KG variant assigns person→city and city→country
**per example** (stated only in the contexts), so the answer is not determined by the name.

## Verification of the four checks
1. **No-context tuned lens ≈ chance — PASS.** city 0.008–0.030 (chance 0.020), country
   0.038–0.077 (chance 0.050) at all layers, vs **1.00 flat** on the original. Shortcut removed.
2. **No-context / Neither model ≈ 0 — PASS.** AB-bucket `rm_both` = 0.000; the bare-context-off
   model cannot answer.
3. **B0 still learns from context — PARTIAL.** It learns from **ctx_B but not ctx_A**:

   | AB bucket | full | rmA (keep B) | rmB (keep A) | rm_both |
   |---|---|---|---|---|
   | per-token EM | **0.179** | 0.179 (Δa **0.000**) | 0.000 (Δb +0.179) | 0.000 |

   Removing agent A does nothing (Δa=0); removing agent B collapses it. EM fell from the
   original **0.585 → 0.179**.
4. **Tuned lens (context on) — legitimate now, clear result.** Fig `notes/figures/tunedlens_randomkg.png`:
   - **hop-2 country:** at chance through layers 0–7, then **turns on exactly at the first MAS
     layer (L8: 0.16)** and rises across the MAS layers to ~0.30 at L23 (chance 0.05).
   - **hop-1 bridge city:** stays at **chance everywhere** (~0.05–0.07, chance 0.02); the gold
     city never appears in the probe's top-3 at any layer.
   - no-context control flat at chance (re-confirms check 1).

## What this reveals (the important part)
**Removing the person-name shortcut shows that this MAS model/recipe does NOT learn genuine
two-hop composition.**
- It never represents the **bridge city (hop-1)** in a linearly-decodable form, and the lesion
  shows agent A (the first-hop context) is unused (Δa=0).
- It develops only a **partial, prior-biased country signal** (hop-2): probe ~0.30, model EM 0.18,
  with a strong frequency bias (e.g. "Belgium" dominates the top-3 across examples).
- The country signal does appear **at/after the MAS injection layers** and accumulates with depth
  — so the contexts are being read, but not composed via a bridge entity.

**Implication for the project's headline.** The original strong composition signature
(AB EM 0.585, Δa +0.427) was **substantially enabled by the deterministic-KG / person-name
shortcut**. On the shortcut-free task the same recipe degrades to a non-compositional, ctx_B-only
strategy. This is a sobering but important honesty correction: the earlier "role-specific
composition" result should be qualified — it was not robust to removing a memorization shortcut.

## Caveats / what is NOT proven
- **Single seed, single recipe.** This shows hop-1 did not emerge *with the current recipe*
  (LoRA layers {8,12,16,20}, 30k steps, this scale). It does **not** prove the task is
  unlearnable — more steps, capacity, a curriculum (train single-hops first), or different
  injection layers might enable genuine composition. Worth testing before any strong claim.
- Probe accuracy ≠ model use (decodability is a diagnostic). But here probe AND lesion AND EM
  all agree (hop-1 absent), which is consistent and mutually reinforcing.
- EM 0.179 is above the 0.000 floor, so the model did learn *something* from ctx_B.

## Files
- generator `src/elf_mas/data/synth_2fact_randomkg.py`; dataset `data/synth_2fact_randomkg_v1/`.
- B0 `runs/cola_mas_randomkg/full_seed0/`; lesion `eval_n200_T16.json`.
- tuned lens `runs/cola_mas_randomkg/tuned_lens_{full,none}.json`; fig `notes/figures/tunedlens_randomkg.png`.
