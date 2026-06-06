# Tuned-lens probe — B0 (full MAS) on `synth_2fact_v1`  (TRAINED DIAGNOSTIC PROBE)

**Date:** 2026-06-05 · checkpoint `runs/cola_mas_sweep/full_seed0`.

## Method
For each DiT layer 0–23, a **tiny linear probe `Linear(2048→C)`** (no nonlinearity) is
trained on the **train split** and scored on the **held-out eval split**. Two AB targets:
- **hop-1 = bridge city** (parsed `"{person} lives in {city}"` from ctx_A); 40 classes, chance 0.025.
- **hop-2 = answer country** (`r["answer"]`); 18 classes, chance 0.056.

Feature = mean-pooled answer-block hidden at **step 0 (t=1, pure noise + context)**. ~2000 train /
600 eval AB items. **This is a trained diagnostic probe, NOT the model's native output.**

Control: re-run with `--context_mode none` (MAS off → contexts never injected) to test whether the
decodability comes from the contexts or from the question prefix.

## Result: a confound, not a depth story
Figure: `notes/figures/tunedlens_full_vs_nocontext.png`.

| Layer | full city | full country | **no-ctx city** | **no-ctx country** |
|---|---|---|---|---|
| 0–4 | 0.16→0.90 | 0.20→0.93 | (same, rising) | (same) |
| 5–7 | ~1.00 | ~1.00 | **1.00** | **1.00** |
| 8 (MAS) | 0.94 | 0.97 | **1.00** | **1.00** |
| 12/16/20 (MAS) | ~0.95→0.91 | ~0.95→0.90 | **1.00** | **1.00** |
| 23 | 0.87 | 0.89 | **1.00** | **1.00** |

- **Both city and country are ~100% decodable by layer 5** — *before* the first MAS layer (8).
- **The no-context control is 100% (flat) at every layer.** With zero context injected, the probe still
  recovers the answer perfectly.

**Conclusion: this is person-name prefix leakage, not context composition.** The synthetic KG is
deterministic (each person → fixed city → fixed country) and eval reuses the same KG, so the person
name in the question prefix *determines* the answer. A trained linear probe simply memorizes
person→city→country and reads it off the early person-name representation. Both "hops" decode at the
same (very early) depth because both are functions of the same person token.

## What it can and cannot show
- **Cannot** answer "does hop-1 become decodable before hop-2" on this dataset — the prefix shortcut
  defeats it; city and country are equally, trivially decodable from the name.
- **Decodability ≠ causal use.** The *model* does need the contexts (B2 no-context EM = 0.000; lesion
  and swap show context dependence). The probe finds a shortcut the model itself does not rely on —
  the classic "probing accuracy ≠ what the model uses," amplified here by the deterministic KG.
- **One genuine signal:** in the full run, accuracy *dips* after the MAS layers (country: 1.00 → 0.89
  by L23; `full − no-context` gap grows from 0 at L7 to −0.10/−0.12 by L20–23). Injecting the contexts
  *perturbs* the clean person-name manifold — i.e. the MAS adds real context-derived computation that
  doesn't align with the memorizable shortcut. This is consistent with the model composing from
  contexts, but it is a weak/indirect signal, not a localization of the hops.

## Consolidated lesson on which probes to trust (Phase 3)
- **gates** → misleading (anti-correlate with importance);
- **vanilla logit-lens** → blind below L22 (basis mismatch);
- **tuned lens (this dataset)** → confounded by person-name prefix leakage (deterministic KG);
- **per-layer knockout + per-agent lesion + A/B swap** → the trustworthy causal probes.

## Recommended fix to actually localize composition
Build a **randomized-KG dataset variant**: assign each person's city (and the city's country)
*per item*, stated only in ctx_A/ctx_B, so the answer is **not** determined by the person name. Then
the prefix shortcut is gone, and a tuned lens (or even logit-lens) can legitimately test whether the
bridge city becomes decodable at a shallower layer than the country, and whether that aligns with the
MAS layers. This is a small change to `src/elf_mas/data/synth_2fact.py` (randomize `person_city` /
`city_country` per example instead of from a fixed KG).
