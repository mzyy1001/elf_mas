# Role-Conditioned Specialization in Coupled Multi-Agent Language Flows: A Control-Variant Study on a Frozen 1.8B Diffusion LM

*Internal experiment report — Phase 3, synth_2fact_v1, seed 0. Compiled 2026-06-03.*

## Abstract

We ask whether end-to-end training of gated per-agent velocity residuals injected into a frozen Cola-DLM (1.8B) backbone induces *role-specific* specialization, rather than merely adding capacity. On a synthetic two-fact composition task, the coupled two-agent model (B0/`full`) reaches 0.585 per-token EM on the two-hop (AB) bucket with a clean lesion signature (removing either agent degrades performance; removing agent B is the more damaging, Δb=+0.585 vs Δa=+0.427). Two data-side controls degrade monotonically — removing private-context distinctness (`identical_ctx`, 0.256) and breaking the context–question correspondence (`context_shuffled`, 0.061) — which **is consistent with** specialization being driven by correctly-paired private contexts rather than raw capacity. The two architectural baselines that would isolate capacity from coupling — single_model (B4) and frozen_agents (B2) — are currently **pending** (re-running after a fixed implementation bug; see §6). All numbers are single-seed; error bars are not yet available.

## 1. Experimental Setup

**Backbone.** Cola-DLM: DiT (1829.9M params) + VAE (501.9M params), both frozen. MAS is injected as in-block LoRA-style per-agent cross-attention at DiT layers {8, 12, 16, 20} (inner dim 512, 8 heads, zero-init output, gate bias −2), ≈25.3M trainable params (dual configuration).

**Training recipe (fixed across variants).** Uniform-t sampling, token-CE auxiliary loss (λ=0.3) via the frozen VAE decoder, prefix conditioning (question-only prompt; private contexts enter only through MAS heads), 30k steps, batch 8, lr 1e-4, seed 0.

**Task / data.** `synth_2fact_v1`, a two-fact composition task with four example buckets (AB / A-only / B-only / neither). Headline numbers are reported on the **AB (two-hop) bucket** of the eval split (n=82 of 200 items).

**Metric.** Per-token-ID exact match (`per_token_em`): tokenize the gold answer, compare the first-K predicted token IDs. Text-level EM is *not* used as the primary metric because a decode-strip artifact (the answer token is joined to trailing pad-region tokens, e.g. `"Belgium"+"Tor"→"BelgiumTor"`) drives text-EM to 0% even when position 0 is correct.

**Evaluation.** Lesion matrix via gate/context ablation at sample time (T_inf=16 Euler steps, prefix-conditioned with KV-cache priming), n=200 items. Ablations: `AB` (both agents active), `A_only` (keep A, remove B), `B_only` (keep B, remove A), `Neither` (both removed).

**Variants.**

| ID | Name | Type | Description | Status |
|----|------|------|-------------|--------|
| B0 | `full` | — | Two role-conditioned agents with private contexts ctx_a, ctx_b | **valid** |
| B6 | `identical_ctx` | data-side | Both agents receive the same (shared) context | **valid** |
| B7 | `context_shuffled` | data-side | Contexts permuted across examples (breaks context↔question pairing) | **valid** |
| B4 | `single_model` | architecture-side | One parameter-matched head over concat(ctx_a, ctx_b) | **pending** |
| B2 | `frozen_agents` | architecture-side | Dual architecture, agent transforms frozen, gates-only training | **pending** |

Data-side variants alter only the context fed to the model; architecture-side variants alter the MAS module wiring / which parameters train.

## 2. Results

Main results — AB (two-hop) bucket, per-token EM (fraction in [0,1]); single seed (seed 0). Δa = EM(full) − EM(remove A); Δb = EM(full) − EM(remove B).

| Variant | Full EM | rmA (B_only) | rmB (A_only) | rm_both (Neither) | Δa | Δb | Interpretation |
|---------|--------:|-------------:|-------------:|------------------:|------:|------:|----------------|
| **B0 full** | **0.585** | 0.159 | 0.000 | 0.000 | +0.427 | +0.585 | Both agents causally load-bearing; B more critical than A |
| **B6 identical_ctx** | 0.256 | 0.049 | 0.012 | 0.000 | +0.207 | +0.244 | EM more than halves; lesion asymmetry compresses toward symmetry |
| **B7 context_shuffled** | 0.061 | 0.012 | 0.012 | 0.000 | +0.049 | +0.049 | Near-collapse; lesions small and symmetric (no coherent role structure) |
| **B4 single_model** | pending | pending | pending | pending | pending | pending | Param-matched capacity control (running, ≈step 12.4k/30k) |
| **B2 frozen_agents** | pending | pending | pending | pending | pending | pending | No-trained-agent floor (queued) |

Notes: `rmA`/`B_only` = keep only agent B; `rmB`/`A_only` = keep only agent A. The B0 row reproduces the prior standalone 30k run (0.585 EM) to the reported precision, indicating the fixed sweep recipe is faithful.

## 3. Ablation / Control Analysis

**Confirmed evidence (B0 vs B6 vs B7).** Full performance and lesion asymmetry degrade monotonically as the role-relevant structure of the input is removed:

- **B0 (full).** EM 0.585 with a strong, asymmetric lesion signature (Δb=+0.585 > Δa=+0.427). Removing agent B collapses the AB-bucket EM to 0.000; removing agent A leaves 0.159. This **is consistent with** each agent carrying distinct, necessary information, and **suggests** agent B (the fact required for the final hop) is the more critical role — matching the task's question form.
- **B6 (identical_ctx).** When both agents see the same context, EM falls to 0.256 (a >2× drop) and the lesion asymmetry compresses (Δa=+0.207, Δb=+0.244, closer to symmetric). This **supports** the interpretation that *distinct private contexts*, not the two-head architecture alone, are responsible for the asymmetric specialization in B0.
- **B7 (context_shuffled).** Breaking the context↔question correspondence collapses EM to 0.061 with near-zero, symmetric lesions (Δa=Δb=+0.049). This **is consistent with** the model relying on correctly-paired evidence; under shuffling there is no exploitable role structure.

Taken together, the B0 ≫ B6 ≫ B7 ordering **supports** the claim that the observed specialization is driven by correctly-paired, role-distinct private contexts. The evidence is graded (B6 retains partial signal) rather than binary.

**Pending evidence (B4, B2).** These are required to separate *coupling/role-specialization* from *raw capacity* and *trained-agent necessity*:
- **B4 (single_model)** is parameter-matched (single head, inner dim 848 ≈ 2×512; measured 25.51M vs 25.31M trainable, within ~1%) and sees the same information via concat(ctx_a, ctx_b). If B0 > B4 at matched parameters, that would **support** a benefit from coupled role-conditioning beyond capacity. **Not yet available.**
- **B2 (frozen_agents)** freezes the agent transforms (gates-only training) and is expected to approximate a no-MAS floor; it isolates whether trained agent residuals (not just gates) are necessary. **Not yet available.**

## 4. Validity and Bug Note

**Bug (silent variant fallback).** In the prior sweep, `frozen_agents` (B2) and `single_model` (B4) were trained with `--lora_mode`, but the LoRA training branch constructed the model solely via the dual-head patch and never branched on `--variant`. As a result both silently trained as `full`. This was confirmed directly: the B2, B4, and B0 checkpoints were **byte-identical** (md5 `4c64d210…`) with identical final loss (1.7658). The `--variant` flag affected only the data path, which acts on `identical_ctx`/`context_shuffled` alone.

**Discarded results.** The earlier `frozen_agents` and `single_model` sweep outputs are invalid and have been discarded (directories removed); they must not be cited. The valid variants (B0, B6, B7) are unaffected because their behavior is realized on the data side.

**Parser / axis-orientation correction.** An initial reading of the eval JSON transposed its two axes (treating the outer key as the ablation when it is the data bucket, and vice-versa), which produced an incorrect `rmA` value of 0.358 for B0. The corrected orientation yields `rmA` = 0.159 for B0, which matches the independently-recorded prior 30k run. All numbers in §2 use the corrected orientation.

**Guard added.** A runtime assertion (`assert_lora_variant_realized`) now fails loudly if a requested variant did not actually reshape the model or its trainable parameters: `single_model` must build single-mode wrappers and be parameter-matched (0.85–1.15× the dual reference); `frozen_agents` must have <10% of MAS parameters trainable; `full`/`identical_ctx`/`context_shuffled` must train 100%. The eval path additionally warns when `--variant` disagrees with the checkpoint's recorded training variant. This guard was unit-tested to reject the exact silent-fallback configuration and to accept all valid configurations.

## 5. Conclusion

**Strongest supported conclusion.** On `synth_2fact_v1`, end-to-end training of gated per-agent residuals on a frozen 1.8B diffusion LM produces an asymmetric, causally-grounded lesion signature (B0: Δb=+0.585, Δa=+0.427 at 0.585 EM), and this signature degrades monotonically when private-context distinctness (B6) or context–question correspondence (B7) is removed. This **supports** the hypothesis that the specialization is driven by role-distinct, correctly-paired private contexts rather than by the two-head architecture per se.

**Not yet proven.** We have not yet shown that coupling outperforms a parameter-matched single model (B4) or that trained agent transforms (vs. gates alone) are necessary (B2); both are pending. All results are single-seed, so the magnitude of effects is not yet bounded by variance.

**Next steps.** (i) Complete the corrected B4 and B2 runs and populate the pending rows; (ii) add seeds 1–2 for all five variants to obtain error bars (five GPUs are currently free, permitting parallel execution); (iii) report the matched B0-vs-B4 comparison as the headline capacity control.
