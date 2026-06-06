# Layer-wise logit-lens probe — B0 (full MAS) on `synth_2fact_v1`

**Date:** 2026-06-05 · checkpoint `runs/cola_mas_sweep/full_seed0` · 16 examples (10 AB, 2 each A-only/B-only/neither) · fixed seed 0 · T_inf=16.

## Method (and its approximation)
During normal B0 sampling we hook **all 24 DiT blocks** and capture the answer-block
hidden state after each. To "decode" a layer we apply the model's **real output head** —
`txt_out_ada` (adaLN modulated by the timestep embedding) + `txt_out` (Linear 2048→16) —
to that intermediate hidden, giving a per-layer velocity `v_L`; then the same x-prediction
shortcut `z_clean = z_t − t·v_L` and a frozen-VAE decode to tokens.

- **Correctness check (passed):** applying the head to layer 23's hidden reproduces the
  model's actual velocity exactly — `max|Δv| = 0.0`. So the head is genuine and correctly wired.
- **Approximation (critical):** layers < 23 were **not trained** to be read by this head.
  Decoded intermediate text is a **diagnostic probe, not the model's output**.

## Result: the logit-lens is *blind* below the final two layers
Per-token EM is **0.000 at every step for all layers 0–21**, including every MAS layer
{8, 12, 16, 20}. Signal appears only at:
- **Layer 23:** decodable throughout (≈0.62 at step 0, 0.44–0.62 across steps).
- **Layer 22:** nothing until the last two steps (0.12 at step 14, 0.50 at step 15).

Intermediate decodes are gibberish, e.g. (example 0, gold `Belgium`, final step):
`L8 → " to they the, drive't drive…"`, `L16 → " over. wrong. land eat…"`,
`L20 → " yes to know America…"`, `L23 → "Belgium…"` (correct first token, then
pad-region junk — the known decode-strip artifact).

Figures (`notes/figures/`):
- `logitlens_layer_step.png` — layer×step heatmap: all dark except the bottom two rows.
- `logitlens_em_by_layer.png` — EM vs layer: flat 0 until a jump at 22→23.
- `logitlens_layer_example.png` — layer×example: only rows 22–23 vary.

## What this shows
1. **In this DiT, the answer only becomes *linearly head-decodable* in the last 1–2 layers.**
   The representation is reorganized into the readout basis right at the end; intermediate
   states (where the MAS actually injects) are not in that basis.
2. **The answer commits early in *denoising time*** — layer 23 already implies the correct
   answer at step 0 (from pure noise + context), then merely maintains/refines it. So
   "formation" is depth-late, time-early.

## What this CANNOT prove (and the honest caveat)
- **It does not contradict the knockout finding** (layers 8/12 causally critical, 16/20 refine).
  Logit-lens measures *linear decodability by the final head*, not *causal contribution*.
  The early MAS layers do essential work — knockout proves it — but in a representation the
  final head cannot read until layers 22–23 transform it.
- **Vanilla logit-lens is the wrong tool to trace per-layer formation here.** Combined with the
  earlier probes we now have a consistent meta-lesson on *which signals to trust*:
  - gate values → **misleading** (anti-correlate with importance);
  - logit-lens → **uninformative below L22** (basis mismatch);
  - **per-layer knockout + per-agent lesion → the trustworthy causal probes.**

## Recommended follow-up to actually trace formation
A **tuned lens**: train a small per-layer affine map (hidden_L → answer latent / token),
~minutes, frozen backbone. It corrects the basis mismatch and *can* reveal where the answer
first appears across depth. It introduces a *learned* readout (must be labelled as such), but
it's the standard fix when vanilla logit-lens fails. This would directly test "does Agent A
write the first-hop entity early and Agent B complete the second hop later," which the vanilla
logit-lens here cannot.
