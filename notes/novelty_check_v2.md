# Novelty Check Report — ELF-MAS v0.2 (Coupled Multi-Agent Language Flows)

**Date:** 2026-05-14
**Reviewer:** Claude (Phases A/B/D) + Codex `gpt-5.5` xhigh reasoning (Phase C)
**Pitch under review:** PLAN.md v0.2 (`Coupled Multi-Agent Language Flows`)
**Trace:** `.aris/traces/novelty-check/2026-05-14_run02/`
**Previous run:** `notes/novelty_check_v1.md` (v0.1, scored 3/10 — pivoted)

---

## Verdict

| | v0.1 | **v0.2** |
|---|---|---|
| **Novelty score** | 3/10 | **6/10** |
| **Recommendation** | PIVOT | **PROCEED WITH CAUTION** |

The reframing succeeded. The contribution is no longer "MAS via latent comm" (which was 3/10 because of CIPHER/LatentMAS/Interlat). The defensible v0.2 contribution is the **conjunction**:

> Multi-agent + **private contexts per agent** + shared denoising trajectory in a flow-matching language model + **end-to-end backprop through all flow steps** + **causal evidence-grounded role specialization** as the empirical finding.

No piece alone is novel; the conjunction is.

## What survived

| Claim | Verdict | Reason |
|---|---|---|
| C1 (additive gated velocity in language flow) | **partially survives** | MoE-FM (YAN, 2604.15009) already decomposes velocity into specialized vector fields. Our angle: experts are *role-conditioned with private contexts*, not routed-to-same-input. Must be framed accordingly. |
| C2 (multi-denoiser shared trajectory) | **survives if multi-agent with private ctx** | YAN / LLaDA-MoE share trajectory but are single-model MoE on shared input. |
| C3 (E2E backprop through all flow steps) | **standard practice** | Not a contribution. Use as background. |
| C4 (emergent role/phase specialization) | **conditionally survives** | Phase-wise routing is in 2604.01622; causal-lesion validation of expert identity is in 2604.14434. Our delta: *context-isolated* agents specialize on a language-flow trajectory under joint training. Must be evidence-grounded, not just gate-plot. |
| C5 (stop-gradient ablation as central test) | **survives if operationally tight** | Codex flags "between agents" as ambiguous in a simultaneous additive field — must define operationally. |

## Closest prior work (all verified by WebFetch)

| arXiv | Title (year) | Domain / setting | Overlap | Key delta from v0.2 |
|---|---|---|---|---|
| [2604.15009](https://arxiv.org/abs/2604.15009) | YAN: MoE Flow Matching for LM (Apr 2026) | flow-matching language, single model | additive velocity decomposition into "locally specialized vector fields" | single-model MoE on input, no private contexts, no MAS, no specialization headline |
| [2604.01622](https://arxiv.org/abs/2604.01622) | Expert-Choice Routing for DLMs (Apr 2026) | diffusion language, single model | timestep-dependent expert capacity = phase-wise routing | single-model MoE, no MAS, no private contexts |
| [2509.24389](https://arxiv.org/abs/2509.24389) | LLaDA-MoE: Sparse MoE DLM (Sep 2025) | diffusion language, single model | MoE inside DLM | single-model MoE, no MAS framing |
| [2604.14434](https://arxiv.org/abs/2604.14434) | Geometric Routing: Causal Expert Control in MoE (Apr 2026) | standard MoE LLM | causal lesion-style validation of expert identity | not flow / not multi-agent / not private-context |
| [2206.01714](https://arxiv.org/abs/2206.01714) | Composable Diffusion (Liu/Du, ECCV 2022) | image | additive score composition, conceptual precedent | image, trained-then-composed, no MAS |
| [2601.07152](https://arxiv.org/abs/2601.07152) | Agents of Diffusion (2026) | DLM as RL-tool | "multi-agent" name | AR LLM agents prompt the DLM with text, RL training, no shared trajectory — different setting |
| [2601.21251](https://arxiv.org/abs/2601.21251) | Skill MoE Policy (SMP, ICLR 2026) | diffusion policy / robotics | phase-consistent sticky gates, specialization | control not language, no MAS communication |
| [2505.22323](https://arxiv.org/abs/2505.22323) | Advancing Expert Specialization for Better MoE (NeurIPS 2025) | standard MoE LLM | expert specialization improvements | AR LLM, no flow, no MAS |

Withdrawn: arXiv 2510.18515 "Socialized Learning..." — authors withdrew due to invalid code; **do not cite.**

Still in play from v1: CIPHER, LatentMAS, Interlat, Coconut, Mixture-of-Thoughts, DIAL (related-work only, not killer), MoDE / Switch-DiT / Race-DiT, MAC-Flow, "Not All Denoising Steps Are Equal."

## The hostile-reviewer killer (Codex's quote, verbatim)

> "This is a context-conditioned MoE diffusion language model on top of ELF; the claimed agent specialization is the standard router/expert specialization already studied in MoE and diffusion-policy work, and the stop-gradient ablation only shows that unrolled end-to-end training matters."

This sentence still kills v0.2 **if** experiments only show gate plots and end-to-end-training-helps. To survive it, experimental design must produce:

1. Lesions that drop accuracy in a role-specific, evidence-private way (Agent B's lesion hurts only on items where its private passage was needed — and that pattern survives in stop-gradient and frozen variants only as a much weaker effect).
2. Controls that rule out "private context makes specialization inevitable" — identical-context and context-shuffled baselines.
3. An operationally tight definition of "stop-gradient between agents."

## Required changes to PLAN.md before implementation (from Codex)

1. **Add identical-context control (already B6 in §7) AND a context-shuffled control** — agents get private context vectors but assigned to wrong agents. This kills the trivial "private context → specialization" reading.
2. **Pre-register success criteria.** A run is a *win* iff: (a) accuracy ≥ matched single ELF, (b) `g_i(t)` profiles differ across agents with p < 0.01 across seeds, (c) role-specific lesion drops are statistically significant, AND (d) all three of (a)–(c) disappear under stop-gradient.
3. **Operationally define stop-gradient.** "Stop-gradient between agents at flow-step boundaries" = truncate gradient through `z_t` across flow steps; forward pass identical to ours. This isolates the value of *unrolled E2E training through the trajectory*.
4. **Reframe the C4 evidence bar.** Headline figure is NOT a gate-activation heatmap. Headline figure is **role-specific lesion drops × evidence-private subsets** under ours vs. stop-gradient vs. frozen.

## Suggested positioning (paragraph for paper intro)

> "Mixture-of-experts diffusion language models (YAN, LLaDA-MoE, Expert-Choice DLM Routing) already decompose flow-matching velocity into specialized vector fields and route across timesteps. Causal validation of expert identity has also been studied in standard MoE (Geometric Routing). What is missing is a *multi-agent* setting where experts see **disjoint private context** rather than the same input, jointly steer **one** shared language-flow trajectory, and are trained **end-to-end** through every flow step. We show that this setting produces causal, evidence-grounded role specialization — agents specialize on the information their private context grants — and that this pattern collapses under stop-gradient and frozen ablations on a controlled compositional task before transferring to multi-hop QA."

## Show-stopper risks unique to v0.2

- **R_v2.1** Lesion drops may not be evidence-private — agents may pool features generically. Then we have specialization (gate variance) without role-grounding (lesion specificity). Mitigation: pick the synthetic task so that private context is provably necessary for a known subset of items.
- **R_v2.2** Single-model MoE-FM (e.g., YAN) at matched compute may beat our MAS variant. Mitigation: argue + measure that the WIN is on compositional/multi-evidence items where private-context-conditioning is what helps, not on language modeling perplexity.
- **R_v2.3** `2604.01622` already captures the "different experts at different timesteps in DLMs" point. Our phase-wise specialization claim must be the *agent-role × phase* joint distribution, not just phase variation per expert.

## Suggested next actions

1. **Patch PLAN.md** to add: context-shuffled control (B7), pre-registration of success criteria, operational stop-gradient definition, and the four new prior-work entries (2604.01622, 2604.15009, 2604.14434, 2509.24389). Done by Claude in this session.
2. **Phase 1 design doc (`notes/synthetic_task_v1.md`)** must commit to a task where private-context necessity for a known subset is verifiable by construction. Top candidate: a 2-fact composition task where fact A is in Agent A's private context and fact B is in Agent B's, neither alone is sufficient, and the answer requires combining them. Lesion specificity becomes trivial to measure.
3. **Re-run novelty-check is NOT required before implementation** — v0.2 cleared the bar. Re-run after Phase 1 toy results, before HotpotQA, to check whether new MoE-DLM or latent-MAS work appears that necessitates a re-pivot.
