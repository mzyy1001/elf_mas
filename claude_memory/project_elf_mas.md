---
name: project-elf-mas
description: "Hongrui's main research project — building a multi-agent system on ELF where agents exchange continuous latents instead of tokens."
metadata: 
  node_type: memory
  type: project
  originSessionId: a4fa26be-d662-41d5-a043-71652dcaf996
---

Hongrui's ML research project (started 2026-05-14): build a multi-agent system using ELF (Embedded Language Flows, arXiv 2605.10938) as the per-agent backbone. **Direction PIVOTED 2026-05-14 after `/novelty-check` (see `notes/novelty_check_v1.md`).**

**Original pitch (now dead, novelty 3/10):** "MAS where agents exchange continuous embeddings instead of tokens, end-to-end trained." Killed by CIPHER (ICLR 2024), LatentMAS (ICML 2026 Spotlight), Interlat (Nov 2025).

**New direction — Coupled Multi-Agent Language Flows (PLAN.md v0.2):** Train multiple role-conditioned velocity-residual agents over a single shared ELF latent answer trajectory `z_t`, with **end-to-end backprop through every flow step**. Flow equation: `v(z_t, t) = v_0(z_t, t) + Σ_i g_i(z_t, t) · v_i(z_t, t, ctx_i)`. Backbone `v_0` is pretrained ELF (frozen or LoRA); each agent contributes a velocity residual `v_i` from its private context `ctx_i`; gates `g_i` are state- and time-dependent.

**Contribution is the TRAINING claim, not the communication claim.** Headline result aimed at: *end-to-end gradient-coupled multi-agent denoising induces emergent role-wise / phase-wise specialization that frozen / stop-gradient variants cannot.* Specialization operationalized via (a) gate-activation profiles differing across agents, (b) role-specific lesion drops.

**Six required baselines** isolating each component of the training claim: B1 stop-gradient between agents, B2 frozen agents (gates-only training), B3 sequential latent-passing (CIPHER/LatentMAS-style), B4 parameter-matched single ELF, B5 text-channel MAS, B6 identical-context agents (no role conditioning).

**Task ordering: synthetic toy first, NOT HotpotQA first.** Phase 1 = synthetic compositional task (parse-then-execute / translate-then-aggregate / arithmetic-chain) where specialization is observable by construction. Phase 2 = translation purely as ELF reproduction sanity (NOT a "does MAS work" experiment). Phase 3 = HotpotQA / 2WikiMQA distractor with oracle retrieval.

**Stance on prior work:** DIAL (Foerster 2016) is *related work, not a killer* — tiny MARL policies, not language flows. CIPHER / LatentMAS / Interlat pass last-layer hidden states sequentially, never share a denoising trajectory. The flow-matching structure (shared `t` coordinate + velocity decomposition) is *load-bearing*, not a backbone swap.

**How to apply:** Bias toward (a) the velocity-residual + gated architecture from PLAN.md §4; (b) gate-activation probing and single-agent lesions as the specialization metric, NOT downstream accuracy as the headline; (c) building the synthetic toy BEFORE HotpotQA — gates must separate on a controlled task before scaling; (d) training adapters + gates only in v1, backbone frozen; (e) any draft must include the prior-work paragraph in PLAN.md §11. See [[reference-tooling]] for repo paths.
