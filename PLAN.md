# Coupled Multi-Agent Language Flows

**Status:** draft v0.3 — 2026-05-14 (post-novelty-check v2 patches)
**Working dir:** `/home/mzyy1001/elf_mas`
**Compute:** local RTX 5090 (32 GB) + remote NVIDIA-GPU server
**Backbone paper:** *ELF: Embedded Language Flows* (Hu et al., 2026, arXiv 2605.10938)
**Backbone code:** `baselines/ELF/` (clone of `lillian039/ELF`, JAX/TPU)
**Novelty assessments:** `notes/novelty_check_v1.md` (v0.1, 3/10 — pivoted); `notes/novelty_check_v2.md` (v0.2, 6/10 — PROCEED WITH CAUTION)

---

## 1. One-line thesis

**Train multiple role-conditioned velocity-residual agents over a single shared ELF latent answer trajectory, with end-to-end backpropagation through the entire denoising process — and show that this joint training induces useful phase-wise / role-wise specialization on compositional tasks.**

The contribution is the **training claim**, not the communication claim. The architecture (coupled velocity fields) is the mechanism; the result we expect to defend is that *gradient-coupled multi-agent denoising produces specialization that frozen or stop-gradient variants cannot.*

## 2. What changed from v0.1

`/novelty-check` on the v0.1 pitch returned a novelty score of 3/10. The framing "MAS that communicates via latent embeddings, end-to-end trained" had already been claimed by CIPHER (ICLR 2024), LatentMAS (ICML 2026 Spotlight), and Interlat (Nov 2025). DIAL (Foerster 2016) is **related work, not a killer** — it is classical MARL with tiny policies, not differentiable language flows. The pivot below preserves the differentiable-MAS intuition but reorganizes the contribution around what is actually novel: *coupled continuous-time flow evolution under joint backprop*, with the empirical finding being *emergent specialization across the denoising trajectory.*

## 3. Why ELF flows are the right substrate

ELF denoises in a continuous T5 embedding space and discretizes only at the final time step (Algorithms 1 & 2, paper §3.1–3.2). For a multi-agent system this gives:

1. A **shared timestep coordinate `t ∈ [0,1]`** that is principled and continuous — agents can specialize by phase, not just by location in a pipeline.
2. A **differentiable trajectory `z_t`** whose update direction is a velocity field `v(z_t, t)` — agents can be defined as additive contributors to that velocity, not as token producers.
3. **No argmax** between an agent's output and the next agent's input — gradients propagate cleanly across all agents through all flow steps.
4. **Pretrained checkpoints** (`embedded-language-flows/ELF-B-de-en`, `ELF-B-xsum`, `ELF-B-owt`) provide a strong shared backbone, so we are training adapters + gates + (optionally) per-agent residual heads, not a 105M model from scratch.

## 4. Architecture: Coupled Multi-Agent Language Flows

### 4.1 The flow equation

Replace ELF's single velocity field `v_θ(z_t, t)` with a **gated sum of role-conditioned velocity residuals over a shared trajectory**:

```
v(z_t, t) = v_0(z_t, t) + Σ_i g_i(z_t, t) · v_i(z_t, t, ctx_i)
```

- `z_t`: shared answer latent in T5 embedding space, evolving under flow matching.
- `v_0`: the frozen (or lightly LoRA-tuned) ELF backbone — provides "default" language-flow dynamics.
- `v_i`: the **velocity residual** produced by agent *i*, conditioned on its private context `ctx_i`.
- `g_i(z_t, t)`: a **learned gate** in [0,1], time- and state-dependent. The gate is the locus of emergent specialization — we will inspect `g_i(z_t, t)` to claim phase-wise role separation.
- Discretization happens only once, at `t = 1`.

### 4.2 Agents and their private contexts

Roles are defined exclusively by **what private context each agent sees**, never by supervised role labels.

| Agent | `ctx_i` |
|---|---|
| A — Planner / Decomposer | the question (and a running "plan latent" if we use one) |
| B — Evidence / Retriever | retrieved passages (oracle in v1, learned later) |
| C — Reasoner / Synthesizer | the question + the current shared latent `z_t` (no private docs) |

All three agents read the shared `z_t` and timestep `t`; only their private context differs. None of them emits text. Gradients from the final CE/MSE loss at `t = 1` flow back through every flow step into every `v_i` and every `g_i`.

### 4.3 What is trainable

Three regimes, in order of increasing risk:

1. **Adapters + gates only** (default for v1). LoRA on each agent's velocity head, full training on each `g_i`. Backbone `v_0` frozen.
2. **Adapters + gates + light backbone tuning.** LoRA on `v_0` too.
3. **Full joint fine-tune.** Everything trainable. Only attempt after (1) clearly works.

## 5. Core research questions

- **RQ1 (training, central).** Does end-to-end backprop through a coupled denoising trajectory produce *measurable specialization* across `v_i` / `g_i` that a stop-gradient or frozen-agent variant cannot?
- **RQ2 (architecture).** Does coupled velocity-residual MAS outperform a sequential latent-passing MAS (CIPHER-style / LatentMAS-style) at matched trainable parameters on compositional tasks?
- **RQ3 (interpretability).** Does specialization concentrate along the **denoising-phase axis** (different agents dominate different `t`), the **content axis** (different agents handle different evidence types), or both?
- **RQ4 (utility).** Does coupled-flow MAS outperform a parameter-matched single ELF on multi-hop QA?

Specialization is operationalized as: (a) gate-activation profiles `g_i(t)` differing significantly across agents over the trajectory, AND (b) lesioning a single agent (set `v_i = 0`) causing role-specific degradation that lesioning others does not cause.

## 6. Tasks

Reordered from v0.1. Synthetic toy is now the **first** experiment, before HotpotQA.

| Tier | Task | Purpose | Goes / no-goes |
|------|------|---------|----------------|
| 0 | **Synthetic compositional task** (e.g. *parse-then-execute* on tiny formal language; *translate-then-summarize* pipeline on a synthetic bilingual corpus; *step-1-output → step-2-input* arithmetic chains) | Test whether gates separate at all under joint training, in a setting where specialization is observable by construction | Gates must separate `g_i(t)` distinctly across agents; lesion drops must be role-specific |
| 1 | **WMT14 De-En** | ELF reproduction gate — confirm the pretrained ELF-B-de-en runs and our coupled architecture doesn't break the base model. Not the headline experiment. | BLEU ≥ 26.4 ± 1 with `v_i = 0` (single-agent equivalent) |
| 2 | **MuSiQue-2hop (recommended primary) → 2WikiMultiHopQA bridge → HotpotQA distractor** | Public-benchmark relevance proof. Full design at `notes/phase2_benchmark_design.md`. MuSiQue chosen primary because its 2-hop items are externally certified to require composition (no single-paragraph shortcut, by construction); 4 buckets built via native sub-question decomposition. | Asymmetry collapse holds on MuSiQue dev (p<0.01, 5 seeds); cross-transfer to 2Wiki bridge subset |
| 3 | **HotpotQA fullwiki / learned retrieval** (stretch) | Removes the oracle to test whether the latent channel survives noisy evidence | optional, if (2) clearly works |

## 7. Baselines

Each baseline isolates one component of the training claim.

| # | Baseline | Isolates |
|---|----------|----------|
| B1 | **Stop-gradient between agents** (forward identical, no backprop through `z_t` across flow steps) | End-to-end backprop matters |
| B2 | **Frozen agents** (only `g_i` and a small final head are trained) | Joint training of the residuals matters |
| B3 | **Sequential latent-passing MAS** (CIPHER / LatentMAS-style: one agent's last hidden state → next agent, no shared trajectory) | Coupled velocity fields over a shared trajectory matters |
| B4 | **Single ELF, matched FLOPs / matched params** | Multi-agent decomposition matters at all |
| B5 | **Text-channel MAS** (same private contexts, but agents emit text that the next agent re-encodes via T5) | Differentiable channel matters |
| B6 | **Identical-context agents** (every agent gets the full context — no role conditioning) | Role-specific private context matters |
| B7 | **Context-shuffled agents** (private contexts exist but are routed to the wrong agents) | Specialization is driven by training, not by trivial input segregation |

The "headline plot" is **not** a gate-activation heatmap. It is **role-specific lesion drops on evidence-private subsets** — under ours vs. B1 (stop-gradient) vs. B2 (frozen). Gate plots are a secondary panel.

### 7.1 Pre-registered success criteria

A Phase 1 run counts as a *win* iff all four hold:
1. Accuracy of "ours" ≥ accuracy of B4 (matched single ELF) on the synthetic task.
2. Gate-activation profiles `g_i(t)` differ across agents with p < 0.01 across ≥ 3 seeds.
3. Role-specific lesion drops on evidence-private subsets are statistically significant: removing Agent X's residual hurts items requiring Agent X's private context substantially more than items requiring other agents' private context.
4. **All three of (1)–(3) collapse** under B1 (stop-gradient) and B2 (frozen) — i.e. the joint-training claim is what produces them.

### 7.2 Operational definition of "stop-gradient between agents" (B1)

In a simultaneous additive velocity field, "between agents" is ambiguous. Operationally:
- Forward pass is identical to "ours" — the same `v = v_0 + Σ_i g_i · v_i` is applied at every flow step.
- Backward pass: gradient is truncated through `z_t` across flow-step boundaries (each flow step's loss only backprops into the gates and residuals computed at that step, not into earlier steps).
- This isolates the contribution of **unrolled end-to-end training through the trajectory**, holding capacity and the additive-residual structure constant.

## 8. Phases

### Phase 0 — Environment + feasibility (this week)
1. ✅ Clone ELF, ARIS, install codex, register MCP.
2. ✅ Codex authed.
3. ✅ Novelty check; pivot recorded here and in memory.
4. **Decide JAX-on-GPU vs PyTorch port.** Test `pip install -U "jax[cuda12]"` on remote server + RTX 5090. If Blackwell support is missing locally, run JAX on remote.
5. Download `embedded-language-flows/ELF-B-de-en` and run `eval.py` end-to-end. Target: BLEU within ±1 of 26.4. **Go/no-go.**

### Phase 1 — Synthetic toy (weeks 1–2)
1. Design a *parse-then-execute* synthetic task (e.g. tiny λ-calculus reduction, or "translate German digits → English words → sum") with controllable difficulty knobs (depth, vocabulary size).
2. Implement the 2-agent coupled flow on top of a small ELF (or PyTorch reimplementation of ELF — see Risks).
3. Train with **just B1 (ours), B1 (stop-grad), B2 (frozen-residuals)**. The toy is cheap so we can run 3+ seeds.
4. **Success metric:** specialization is measurable — gates separate, lesions are role-specific. If this fails, the whole project is in trouble — see §10.

### Phase 1.5 — Translation sanity (week 3, parallel with Phase 1)
1. Use the pretrained ELF-B-de-en checkpoint as the shared `v_0`.
2. Wire the 2-agent coupled flow into the WMT14 pipeline with `v_i = 0` initialization so initial behavior matches the base model.
3. **Success metric:** BLEU ≥ 26.4 ± 1 with no MAS contribution; BLEU stays ≥ base when MAS is turned on. *This is purely a "we did not break ELF" check.*

### Phase 2 — Public-benchmark relevance proof (weeks 4–8, see `notes/phase2_benchmark_design.md`)

Headline pivot: *synthetic proves mechanism (Phase 1 done); public benchmark proves relevance*.

1. **Phase 2.0 — MuSiQue-2hop dataset prep.** Filter (single-hop probe with `deepset/roberta-base-squad2`, answer-length, paragraph-Jaccard), build 4 buckets via sub-question decomposition (AB / A-only from `(question1, paragraph_support_1)` / B-only / neither from cross-item).
2. **Phase 2.1 — decoder + B7 context-shuffled.** Add nearest-neighbor and autoregressive decode (current Phase 1 trainer only measures MSE). Add `--variant context_shuffled`. Run B0/B6/B7 × 5 seeds.
3. **Phase 2.2 — stop-gradient B1.** Build unrolled K-step rollout trainer; add B1 ablation. This is the load-bearing "joint training through trajectory" test.
4. **Phase 2.3 — matched single ELF B4.** Train v_0 alone on `[ctx_A‖ctx_B‖Q]` at matched FLOPs.
5. **Phase 2.4 — external QA baseline R.** FiD-base or LongT5 fine-tuned on MuSiQue, for the absolute EM/F1 reference point.
6. **Phase 2.5 — cross-benchmark transfer.** Re-run B0/B6/B7 on 2WikiMultiHopQA bridge subset → HotpotQA distractor.
7. **Phase 2.6 — `/kill-argument` adversarial review** then write-up via `/paper-plan`.

Pre-registered success criteria (file: `runs/phase2/preregister.json`):
1. B0 EM/F1 > B2 (frozen agents) at p < 0.01 over 5 seeds.
2. B0 per-bucket asymmetry ≥ 2× on A-only and B-only.
3. Asymmetry collapses to ≤ 1.5× under both B6 (identical_ctx) and B7 (context_shuffled), DiD p < 0.01.
4. Collapse direction reproduces on 2WikiMHQA bridge subset (p < 0.05).

If 1–3 hold but 4 fails → MuSiQue-only result, back to cross-benchmark analysis. If 1 fails but 2–3 hold → specialization-only framing. If 2–3 fail → abandon benchmark contribution.

### Phase 3 — Ablations, analysis, write-up (weeks 8–10)
- Latent trajectory length (4 / 8 / 16 / 32 steps), number of agents, context overlap, gate temperature, noise injection on `z_t`.
- Failure analysis on hard hops.
- ARIS `/paper-plan` → `/paper-write` if results pass `/result-to-claim`.

## 9. Risks

| # | Risk | Mitigation |
|---|------|-----------|
| R1 | ELF JAX code doesn't run on Blackwell (RTX 5090) | Use remote server for JAX; reserve 5090 for PyTorch port if needed; PyTorch reimplementation of ELF inference path is feasible (≈ 1–2 weeks) |
| R2 | Pretrained ELF checkpoints don't reload outside TPU | Phase 0.5 tests this explicitly; we have HF-hosted weights that should be platform-agnostic |
| R3 | **Specialization does not emerge on the synthetic toy** (gates collapse to one agent, or all gates flat) | Top-1 gating with temperature annealing; load-balancing loss; per-agent role-pretraining on auxiliary tasks before unfreezing the joint loss; explicit `t`-conditioned gate priors |
| R4 | Single ELF (matched params, no MAS) beats coupled-flow MAS | This is the most likely outcome on translation (low compositionality). It is *expected* — coupled-flow's win is supposed to be on compositional tasks (synthetic toy, multi-hop QA), not translation. We name this in the paper if needed. |
| R5 | Text-channel MAS still wins because pretrained text is a strong interface | Train a stronger latent channel via auxiliary contrastive loss between `z_t` and natural-language hop descriptions; or accept the negative result on text-channel and reframe contribution around RQ1 (training claim) and RQ3 (specialization claim) |
| R6 | ELF at 105M–342M is too small for multi-hop reasoning | Distill from a larger AR LLM into ELF before MAS; fall back to easier compositional benchmarks (CommonsenseQA, StrategyQA); scale to ELF-L (652M) on remote server only |
| R7 | Backpropagating through 16-step flow trajectories with multiple agents is memory-heavy | Gradient checkpointing per flow step; fewer flow steps (8) during training; mixed precision; offload to remote server when 5090 OOMs |
| R8 | Reframed pitch still collides with something we haven't found | ✅ v0.2 cleared `/novelty-check` v2 (6/10). Re-run after Phase 1 toy results, before HotpotQA. |
| R9 | Lesions show specialization (gate variance) but NOT role-grounding (lesion specificity) — i.e. agents pool features generically | Choose Phase 1 synthetic task so that private context is *provably necessary* for a known item subset (e.g. 2-fact composition: fact A only in ctx_A, fact B only in ctx_B, neither alone is sufficient). Lesion specificity becomes verifiable by construction. |
| R10 | Reviewer killer: *"this is context-conditioned MoE-DLM on top of ELF; specialization is just standard MoE routing and stop-gradient only shows unrolled training matters"* | Survive only if experiments show **causal evidence-grounded** lesion patterns AND those patterns collapse under stop-gradient. Gate plots alone do NOT survive this attack. |

## 10. Compute & storage budget

- **Local RTX 5090 (32 GB):** Phase 1 (synthetic toy, ELF-B size, batch ≤ 64). Estimated 5–10 GPU-days for the full toy phase including baselines + ablations.
- **Remote server:** Phase 2 translation sanity (single GPU), Phase 3 HotpotQA at ELF-M scale (larger memory budget needed for 3 agents × 16 flow steps). Estimated 25–50 GPU-days.
- **Storage:** ~80 GB (T5 caches + ELF checkpoints + WMT14 + HotpotQA + checkpoints across baselines). 703 GB free locally — fine.
- **External APIs:** Codex for ARIS review skills (cheap; bursty usage during write-up).

## 11. Distinction from prior work (paragraph for any paper draft)

Mixture-of-experts diffusion language models already decompose flow-matching velocity into specialized vector fields and route across timesteps. Causal validation of expert identity has also been studied in standard MoE. **What is missing is a multi-agent setting where experts see *disjoint private context* rather than the same input, jointly steer *one* shared language-flow trajectory, and are trained *end-to-end* through every flow step.** We show that this setting produces causal, evidence-grounded role specialization — agents specialize on the information their private context grants — and that this pattern collapses under stop-gradient and frozen ablations.

**Single-model MoE in diffusion/flow-matching language models (close neighbors, but no MAS, no private contexts):**
1. **YAN: MoE-FM for LM** ([2604.15009](https://arxiv.org/abs/2604.15009), Apr 2026) — decomposes flow into "locally specialized vector fields" for fast NAR inference. Single-model MoE on shared input. No multi-agent framing. Speed headline, not specialization.
2. **Expert-Choice Routing for DLMs** ([2604.01622](https://arxiv.org/abs/2604.01622), Apr 2026) — timestep-dependent expert capacity in DLMs. Single-model.
3. **LLaDA-MoE** ([2509.24389](https://arxiv.org/abs/2509.24389), Sep 2025) — sparse MoE inside a DLM. Single-model.

**Expert specialization & causal expert control (standard MoE, AR LLMs — methodological precedents):**
4. **Geometric Routing** ([2604.14434](https://arxiv.org/abs/2604.14434), Apr 2026) — causally meaningful expert identities + intervention/lesion-style validation. AR LLM standard MoE. Closest *methodological* precedent for our C4 evidence bar.
5. **Advancing Expert Specialization for Better MoE** ([2505.22323](https://arxiv.org/abs/2505.22323), NeurIPS 2025) — routing improvements for MoE specialization. AR LLM.
6. **DeepSeekMoE** (Jan 2024) — fine-grained expert specialization in AR MoE LLMs.

**Multi-agent latent communication (related, distinct setting — sequential hidden-state passing, not coupled flow):**
7. **CIPHER** ([2310.06272](https://arxiv.org/abs/2310.06272), ICLR 2024), **LatentMAS** ([2511.20639](https://arxiv.org/abs/2511.20639), ICML 2026), **Interlat** ([2511.09149](https://arxiv.org/abs/2511.09149), Nov 2025), **Mixture of Thoughts** ([2509.21164](https://arxiv.org/abs/2509.21164), Sep 2025) — pass last-layer hidden states between agents sequentially. None shares a denoising trajectory; none defines agents as additive contributors to a velocity field.
8. **Coconut** ([2412.06769](https://arxiv.org/abs/2412.06769), Dec 2024) — single-agent continuous CoT in an AR LLM.

**Compositional / phase-wise diffusion (conceptual precedents, image / control domain):**
9. **Composable Diffusion** ([2206.01714](https://arxiv.org/abs/2206.01714), ECCV 2022) — additive score composition in image diffusion, trained-then-composed.
10. **MoDE** ([2412.12953](https://arxiv.org/abs/2412.12953)), **Switch-DiT** ([2403.09176](https://arxiv.org/abs/2403.09176)), **Race-DiT** ([2503.16057](https://arxiv.org/abs/2503.16057)) — image-domain MoE denoisers with timestep routing.
11. **Skill MoE Policy (SMP, ICLR 2026)** ([2601.21251](https://arxiv.org/abs/2601.21251)) — diffusion-policy MoE with phase-consistent gates, robot control.
12. **"Not All Denoising Steps Are Equal"** ([2604.02340](https://arxiv.org/abs/2604.02340), 2026) — MDLM scheduling for compute efficiency.
13. **MAC-Flow** ([2511.05005](https://arxiv.org/abs/2511.05005), Nov 2025) — flow matching for MARL coordination policies, control not language.

**Differentiable MAS (precedent, wrong setting — cite and move on):**
14. **DIAL** ([1605.06676](https://arxiv.org/abs/1605.06676), Foerster 2016) — classical MARL with tiny policies and coordination tasks. Conceptual precedent for end-to-end differentiable inter-agent learning. Not a setting-matched killer.

**Diffusion + multi-agent (different setting):**
15. **Agents of Diffusion** ([2601.07152](https://arxiv.org/abs/2601.07152), 2026) — AR-LLM agents prompt a DLM via natural-language feedback, RL training; no shared trajectory.

Our contribution is the **training claim**: that gradient-coupled multi-agent denoising over a single shared ELF trajectory, where agents have *disjoint private contexts*, produces **causal evidence-grounded role specialization** measurable via role-specific lesion drops on evidence-private subsets — and that this pattern collapses under stop-gradient and frozen ablations on a controlled compositional task before transferring to multi-hop QA.

## 12. Immediate next actions

1. **(user)** Approve this v0.2 framing or push back. Specifically: are the six baselines and the synthetic-task-first ordering acceptable?
2. **Re-run `/novelty-check`** on v0.2 before any 2-week implementation block (R8). The reframing is narrower so this should pass, but the diff is worth one more sweep.
3. Phase 0.4: JAX-on-GPU feasibility test on the remote server.
4. Phase 0.5: pull `embedded-language-flows/ELF-B-de-en` and run `eval.py`. Go/no-go for the whole project.
5. Phase 1 design doc: pick the synthetic task and its difficulty knobs (separate doc at `notes/synthetic_task_v1.md`). Hongrui should choose between parse-then-execute, translate-then-aggregate, and arithmetic-chain.

---

## Appendix A — Repo layout

```
elf_mas/
├── PLAN.md                  ← this file (v0.2)
├── aris-source/             ← cloned ARIS
├── baselines/
│   └── ELF/                 ← lillian039/ELF baseline
├── notes/
│   └── novelty_check_v1.md  ← prior-art audit + pivot rationale
├── .aris/
│   └── traces/              ← codex review traces
└── src/                     ← MAS code (TBC)
```

## Appendix B — Set-up checklist

- [x] Codex CLI installed (`codex-cli 0.130.0`)
- [x] Codex authed (`~/.codex/auth.json`, model `gpt-5.5`)
- [x] Codex registered as Claude Code MCP (user scope, MCP health ✓)
- [x] ARIS skills present at `~/.claude/skills/` (78 skills)
- [x] ELF baseline cloned at `baselines/ELF/`
- [x] Novelty check v1 complete, project pivoted
- [x] JAX-on-GPU verified on chen A100 (8× A100-80GB, CUDA 12.2 driver, JAX 0.4.38 with bundled CUDA 12.9 wheels via PATH prepend for ptxas)
- [x] ELF-B-de-en reproduction within ±1 BLEU of paper — **BLEU 26.53** (paper 26.4 test, README 26.7 val), 64-step ODE CFG=2, 3m13s on 1× A100
- [ ] Synthetic task chosen and `notes/synthetic_task_v1.md` written
