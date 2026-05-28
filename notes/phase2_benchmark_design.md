# Phase 2 — Public Benchmark Validation of Coupled Multi-Agent Language Flows

**Status:** draft v0.1 — 2026-05-17
**Predecessor:** Phase 1 (synthetic `synth_2fact_v1`, results at `notes/phase1_results.md`). Phase 1 is the **mechanism proof**; Phase 2 is the **relevance proof** on a recognized public benchmark.
**Headline framing:** *Synthetic proves mechanism; public benchmark proves relevance.*

The goal is to take the same Coupled MAS architecture that produced the 3.3–4.5× role-specific lesion asymmetry on `synth_2fact_v1`, and show that:

1. it reproduces on a public multi-hop QA dataset whose multi-hop structure is **externally validated**;
2. the same controls (identical_ctx, context-shuffled, stop-gradient) still collapse the asymmetry;
3. the gains carry over to **standard QA metrics (EM/F1)**, not just our internal MSE objective;
4. the result holds across **dataset boundaries** (cross-benchmark transfer to HotpotQA last).

---

## 1. Dataset choice — recommendation

I compared the three multi-hop QA benchmarks against the requirement *"answer must require both pieces of evidence, and evidence must be cleanly bipartite"*.

| Benchmark | Size | Hops | Single-hop shortcut? | Evidence bipartite? | Verdict |
|---|---|---|---|---|---|
| **MuSiQue-Ans** (Trivedi et al. 2022) | 25K | 2–4 (2hop subset = 15K) | **Provably no** — items constructed to defeat single-step solvers (single-hop reasoners drop to ≤30 F1) | Yes — each hop has a dedicated supporting paragraph + sub-question | **Recommended primary** |
| **2WikiMultiHopQA** (Ho et al. 2020) | 192K | 2 (compositional/bridge subset = ~80K) | Partial — comparison questions often have surface heuristics; bridge questions clean | Yes — `evidence` triples, supporting facts grouped by entity | Strong secondary; use bridge subset only |
| **HotpotQA** (Yang et al. 2018) | 113K | 2 (~76K distractor; bridge subset ~70K) | **Known issue** — Min et al. 2019: ~24% answerable from a single paragraph | Yes via `supporting_facts` | External-validation only; use after MuSiQue + 2Wiki pass |

### Recommendation

**Primary benchmark: MuSiQue-Ans, 2-hop subset (`musique-ans-2hop`, ~15K items).**

Reasons in order:
1. **No single-hop shortcut by construction.** MuSiQue was built specifically by composing pairs of single-hop questions and *verifying* that a single-hop QA model trained on SQuAD cannot solve them. This directly attacks the "the model just exploited one paragraph" failure mode that would tank our specialization claim.
2. **Native sub-question decomposition.** Each 2-hop item carries `question_decomposition` with `question1`, `question2`, `answer1`, `answer2`, `paragraph_support_1`, `paragraph_support_2`. We can build **all four buckets (AB / A-only / B-only / neither)** from a single dataset by using sub-questions and partial contexts (§ 3).
3. **Smaller and cleaner.** 15K filtered items is enough for our adapter-only training. Avoids the long-tail noise of HotpotQA.

**Secondary benchmark: 2WikiMultiHopQA, bridge + compositional subsets (~80K items).** Used to test that the result is not MuSiQue-specific.

**External-validation benchmark: HotpotQA, oracle-supporting-facts setting first, distractor setting after.** Done only if MuSiQue + 2Wiki both produce the predicted asymmetry collapse, to claim the result generalises beyond datasets that are unusually clean.

---

## 2. Splitting evidence into private contexts

### 2.1 The canonical mapping for MuSiQue-2hop

Per MuSiQue's schema:
- `question` — the original 2-hop question Q
- `answer` — the final answer Y
- `question_decomposition` — list of two single-hop steps, each with `(question_i, answer_i, paragraph_support_i)`

The mapping is:
- `Q` (the original question) → goes to **all** agents and to `v_0` (frozen ELF backbone).
- `ctx_A` ← `paragraph_support_1` (the paragraph that resolves hop 1).
- `ctx_B` ← `paragraph_support_2` (the paragraph that resolves hop 2).
- `Y` ← final answer.

Distractor paragraphs from the same item (MuSiQue items ship with 20 candidate paragraphs of which only 2 are supporting) are **split between `ctx_A` and `ctx_B` evenly at random**, so that each private context contains 1 supporting + ~9 distractor sentences. This stops the model from solving items by "paragraph contains keyword X = answer".

### 2.2 Building the four buckets

Same bucket structure as `synth_2fact_v1`, so the Phase 1 lesion-drop matrix carries over directly.

| Bucket | What it is on MuSiQue | How constructed |
|---|---|---|
| **AB** | Original 2-hop item | Standard split per §2.1 — both supporting paragraphs split across A and B; answer is the final 2-hop answer. |
| **A-only** | Sub-question 1 alone | Use `(question1, answer1, paragraph_support_1)` as a single-hop item. `ctx_A` = supporting paragraph + distractors; `ctx_B` = distractor-only paragraphs from the SAME item. |
| **B-only** | Sub-question 2 alone | Symmetric. |
| **neither** | Cross-item un-answerable | Take `question1` from item *i* but the contexts from item *j* (no overlap). Answer = `"unknown"`. |

This gives **AB : A-only : B-only : neither ≈ 40 : 25 : 25 : 10** matching the synthetic task's proportions, except now each bucket is grounded in real Wikipedia evidence.

### 2.3 For 2WikiMultiHopQA (bridge / compositional subsets only)

Use the per-item `evidences` field: each evidence is `[entity, relation, value]` plus a sentence pointer. Group supporting sentences by **Wikipedia title** — the natural bipartite split for bridge questions ("In what country is the city where X was born?": one article is about X, the other about the city). For compositional questions, the two titles are usually the two reasoning hops.

A-only / B-only buckets via decomposition aren't natively available on 2Wiki, so for that benchmark we:
- Only build AB items + neither items from cross-item splits.
- Drop A-only / B-only — the headline asymmetry-collapse comparison happens on AB items only (full vs identical_ctx).

This is weaker than MuSiQue's coverage, hence MuSiQue is primary.

### 2.4 For HotpotQA (distractor setting)

Same as 2Wiki: group supporting sentences by `title`, two titles per item ⇒ natural `ctx_A`/`ctx_B`.

---

## 3. Filtering rules

Apply in order. Discard items failing any check.

1. **Answer length ≤ 5 T5-small tokens.** Matches our `S_answer = 16` budget and our nearest-neighbor / autoregressive decode.
2. **Both supporting paragraphs are non-trivially distinct** — Jaccard(token-set(p1), token-set(p2)) < 0.5. Avoids items where the two "supporting" paragraphs actually overlap heavily.
3. **Single-hop probe fails.** Run a pretrained single-paragraph QA model (e.g., `deepset/roberta-base-squad2`) on `(Q, ctx_A only)` and `(Q, ctx_B only)`. If either gives F1 > 0.7 on the full answer, the item is single-hop-shortcuttable — drop it.
4. **Sub-question answers are not substrings of the final answer.** Some MuSiQue items have `answer1 == answer` due to bridge compositions; these can leak. Drop where the overlap is high.
5. **Bucket constructibility.** For A-only / B-only / neither items, ensure the constructed bucket actually meets its definition (e.g. `neither` item's question really has no answer in either context — verify with the same single-paragraph QA probe).

After filtering, expected sizes (MuSiQue-2hop): ~10K AB + (10K A-only synthesized + 10K B-only synthesized + 4K neither synthesized from cross-item) ≈ **30K total**.

---

## 4. Train / dev / test protocol

### 4.1 Splits

Use MuSiQue's native train / dev split. Apply filtering (§3) to each split independently. Test only on official dev (MuSiQue's test labels are held out). Cross-benchmark eval on 2Wiki dev and HotpotQA dev.

| Split | Source | Approx after filtering | Use |
|---|---|---|---|
| train | MuSiQue train | ~25K filtered (mix of all 4 buckets) | model training |
| dev (in-domain) | MuSiQue dev | ~3K | hyper-tuning, lesion matrix headline |
| test-cross-1 | 2WikiMHQA dev (bridge+comp) | ~10K AB-only items | cross-benchmark transfer |
| test-cross-2 | HotpotQA dev distractor | ~5K AB-only items | external validation |

### 4.2 Training protocol

Mirror Phase 1.5 + scale up:
- **Backbone:** frozen ELF-B-de-en (105M). Phase 2.1 ablation: also try ELF-B-xsum (summarization-pretrained — closer to QA than translation).
- **Trainable:** `CoupledMASHeads` (10M params + room to scale to 30M).
- **Optimiser:** AdamW lr 3e-4, weight decay 1e-3.
- **Schedule:** logit-normal t-sampler (same as Phase 1).
- **Batch size:** 64, **steps:** 20K (≈ 50 passes over filtered train, ≈ 8 min on A100 after JIT).
- **Seeds:** 3 per variant for headline; 5 per variant for the final result.

### 4.3 Pre-registration

Before training, write `runs/phase2/preregister.json` with: dataset version hash, filtering hash, train/dev split, hyperparameters, success criteria (§ 5.2). Refuse to revise after seeing results.

---

## 5. Baselines

The Phase 1.5 ablation suite, scaled up to the benchmark:

| Id | Variant | Tests | Phase 1.5 result if applicable |
|---|---|---|---|
| **B0** | Ours: coupled MAS, private contexts, joint training | The full pitch | A-only 4.5× asymmetry |
| **B1** | Stop-gradient through `z_t` across flow steps | Whether end-to-end backprop through trajectory matters | NOT YET BUILT (single-step trainer) — Phase 2.2 |
| **B2** | Frozen residual heads, only gates train | Whether MAS adds value above v_0 alone | loss 0.569 (= frozen v_0) ✓ |
| **B4** | Single ELF, full condition `[ctx_A || ctx_B || Q]`, matched FLOPs | Whether multi-agent decomposition matters at all | NOT YET BUILT — Phase 2.3 |
| **B5** | Text-channel MAS — agents emit tokens that the next agent re-encodes | Whether differentiable channel matters | Out of scope v0.1 — Phase 2.5 |
| **B6** | Identical contexts — both agents see `concat(ctx_A, ctx_B)` | Whether private-context-per-agent is the load-bearing ingredient | A-only ratio 4.5× → 1.4× ✓ |
| **B7** | Context-shuffled — agents see private contexts but **mis-aligned with question structure** | Whether the asymmetry is just from data segregation | NOT YET BUILT — Phase 2.1 |
| **R**  | Strong external QA baseline — FiD-base or LongT5 fine-tuned on MuSiQue | Whether our absolute EM/F1 is competitive at all | NOT BUILT — Phase 2.4 |

**The headline plot is still B0 vs B6 vs B7** on the role-asymmetry signal (§ 6), with R giving the EM/F1 context bar.

---

## 6. Lesion and context-shuffle tests

Identical to Phase 1.5 in spirit but with EM/F1 instead of MSE.

### 6.1 The lesion matrix

For each bucket × ablation, decode an answer (§ 7) and compute mean EM, F1.

| | full EM/F1 | zero_a EM/F1 | zero_b EM/F1 | zero_both EM/F1 | Δa F1 | Δb F1 |
|---|---|---|---|---|---|---|
| AB | … | … | … | … | … | … |
| A-only | … | … | … | … | **expected high** | expected ~0 |
| B-only | … | … | … | … | expected ~0 | **expected high** |
| neither | … | … | … | … | low | low |

### 6.2 The headline asymmetry-collapse comparison

The same table for `full` and `identical_ctx` and `context_shuffled` variants side-by-side. The Phase 1.5 result was a **3.3–4.5× → 1.3–1.4× collapse** in the Δa/Δb asymmetry ratio. Phase 2's success criterion: **the same direction-of-collapse holds on MuSiQue-2hop**, p < 0.01 over 5 seeds.

### 6.3 Statistical tests

For each (variant, bucket) cell, report mean ± std over seeds. For the asymmetry ratio:
- Paired t-test on per-item Δa − Δb: H0 = no asymmetry.
- For the **collapse** claim: difference-in-differences across (full, identical_ctx) on the same items, paired t-test.

These give the p-values the v0.3 `/novelty-check v2` success criteria required (PLAN.md §7.1).

---

## 7. Decoding answers (Phase 1's open problem, now mandatory)

For Phase 1 we used MSE-level lesion drops because EM/F1 needs an answer string. Phase 2 must have a real decoder. Two options, in increasing order of fidelity:

1. **Nearest-neighbor candidate decode (MVP).** Pre-tokenize all answer candidates from the benchmark vocabulary (Wikipedia entity names, dates, "unknown"). Encode each via frozen T5. For each eval item, predict `v_pred` at `t = 1` via ODE rollout (K=16 Euler steps starting from noise). Pool over valid answer positions → vector. Cosine-sim against candidate embeddings → argmax. Cheap; works only when answer set is fixed.
2. **Autoregressive decode via ELF's decoder branch (preferred for final result).** ELF has a `decoder_step_active=True` mode that unembeds `x_pred` → vocab logits. Run the ODE rollout to `t=1`, switch the model into decode mode, get logits, then greedy / beam sample tokens. Closer to ELF's intended generation pipeline, supports arbitrary answers.

Both decoders are built in Phase 2.1 alongside B7. Compare on a fixed dev slice; pick whichever gives saner answers as the headline decoder.

---

## 8. Expected failure modes and mitigations

| # | Failure mode | Mitigation |
|---|---|---|
| F1 | **Filtering too aggressive.** After all 5 rules, < 5K training items survive. | Relax rule 3 to F1 > 0.8 (catches fewer single-hop shortcuts but more items survive). Or supplement train with 2WikiMHQA bridge items. |
| F2 | **ELF-B-de-en mis-fires on English-only QA.** Translation backbone may produce gibberish answers. | Phase 2.1 ablation: try ELF-B-xsum (summarization) and ELF-B-owt (unconditional language) as alternative backbones. Pick the one with the best frozen-baseline F1 on dev. |
| F3 | **Multi-token answers break NN decode.** Many MuSiQue answers are person/place names (2–5 tokens). | Switch to autoregressive decode (§ 7.2). Verify token-pool decode and autoregressive decode rank the same top-1 on a dev slice. |
| F4 | **Asymmetry signal is real but small (Δ ~ 0.02–0.05 F1).** | Larger train set (full MuSiQue + 2Wiki concatenated), more seeds (5+), report effect size with CI rather than just point estimates. |
| F5 | **B4 (matched single ELF) catches up to B0.** If a single ELF on `[ctx_A‖ctx_B‖Q]` gets the same EM/F1, the MAS contribution is null. | This would be a negative result — and a real one. The fall-back claim is then *specialization* alone (B0 vs B6 asymmetry collapse), not absolute accuracy gain. PLAN.md v0.3 §1 already states the contribution is the training-claim, not raw accuracy. |
| F6 | **MuSiQue contexts too long for `S_ctx = 64`.** Wikipedia paragraphs are typically 60–120 tokens. | Increase `S_ctx` to 128 or 192. With S_question=24 and S_answer=16, our frozen ELF input is ctx_A(128)+ctx_B(128)+question(24)+answer(16)=296 tokens. ELF was trained at max_length=128 — we'd need to either truncate ruthlessly, retrain ELF at longer context, or split ELF's input differently (encode each ctx separately and only pass pooled summaries to v_0). |
| F7 | **`F6` is real and forces ELF max_length retraining.** | This is the most likely *real* blocker. Mitigation paths in priority order: (a) summarize/truncate paragraphs to ≤ 50 tokens via a small T5 summarizer (cheap); (b) use a pre-trained encoder-only model with native long context (LongT5, Linformer) **for the encoding step only** — keep ELF-B as v_0 in T5-embedding space; (c) retrain ELF at max_length=256 from the de-en checkpoint (~$200 GPU-day cost). |
| F8 | **Distractor paragraphs leak the answer.** MuSiQue's distractors are not random — they're paragraphs that mention the same entities as the supporting ones, just not informative for the answer. The agent might still find a shortcut. | Run filtering rule 3 (single-hop probe) on every item to verify no shortcut. Keep only items where the probe fails on both half-contexts. |
| F9 | **B7 context-shuffled accidentally degenerates to B6.** If we shuffle by swapping ctx_A ↔ ctx_B per item, the model still sees both private contexts, just labelled with the "wrong" gate. The asymmetry may persist by symmetry. | Stronger B7: shuffle ctx_A and ctx_B *across items* (so Agent A gets item-i's ctx_A but Agent B gets item-j's ctx_B). This breaks evidence-question alignment entirely and is the proper "is alignment necessary?" control. |

---

## 9. Why this addresses the "context-conditioned MoE" reviewer attack

From `/novelty-check v2 §5` (the hostile-reviewer killer sentence):

> *"This is a context-conditioned MoE diffusion language model on top of ELF; the claimed agent specialization is the standard router/expert specialization already studied in MoE and diffusion-policy work, and the stop-gradient ablation only shows that unrolled end-to-end training matters."*

Phase 2's defence is structural, not rhetorical:

1. **MuSiQue items are externally certified to require composition.** Unlike our synthetic task where we *constructed* the composition requirement, MuSiQue items have *passed an external test* (single-hop QA models can't solve them). This eliminates the "the task is trivial and any specialization is artificial" angle.
2. **The asymmetry-collapse control (B6 identical_ctx) reproduces on real evidence.** A bare context-conditioned MoE wouldn't show the same collapse — it routes *over* the same input, so removing the private-context split shouldn't change anything. Our model does change, because it's not just routing — it's training role-specialized residuals over a shared trajectory whose evolution depends on per-agent input.
3. **B7 (context-shuffled) and B4 (matched single ELF) attack two more reviewer arguments:** "the asymmetry is just data segregation" (B7) and "you didn't beat a strong single model" (B4). Phase 2 commits to running both.
4. **EM/F1 grounds the claim in a metric the field accepts.** MSE on T5 embeddings is our internal signal; an external reviewer will dismiss it. Phase 2's headline metric is F1 on a benchmark with a known leaderboard.
5. **Cross-benchmark transfer (HotpotQA last) tests for benchmark-specific overfit.** If the asymmetry collapse holds on MuSiQue but vanishes on 2Wiki and HotpotQA, the result was MuSiQue-specific. If it transfers, the claim is robust.

The pitch becomes:

> *We train role-conditioned velocity-residual agents over one shared ELF latent trajectory with disjoint private contexts. On the MuSiQue 2-hop benchmark — which is externally certified to require multi-hop composition — our coupled flow produces a 3–5× role-specific lesion asymmetry that **collapses to 1×** under context-sharing (B6), context-shuffling (B7), and frozen-residuals (B2), while joint end-to-end training through the trajectory (B0 vs B1) is shown to be load-bearing. The same effect transfers to 2WikiMultiHopQA and HotpotQA. EM/F1 is competitive with strong external baselines.*

That's the paper.

---

## 10. Execution order

Phase 2 broken into shippable chunks. Each leaves a clean intermediate result.

1. **Phase 2.0 — dataset preparation.** Build MuSiQue-2hop filtered, 4-bucket synth-style. Verify with the same sanity-check script as `synth_2fact_v1`. ~2 days incl. download + filter probe model.
2. **Phase 2.1 — decoder + B7.** Implement nearest-neighbor and autoregressive decoders. Add `--variant context_shuffled` to the trainer. Run B0/B6/B7 × 3 seeds on MuSiQue dev. ~3 days.
3. **Phase 2.2 — stop-gradient B1.** Build an unrolled K=4 step rollout trainer; add B1 ablation. ~5 days.
4. **Phase 2.3 — matched single ELF B4.** Add a "no-agents" config to the trainer (just train v_0's adapters on `[ctx_A‖ctx_B‖Q]`). ~2 days.
5. **Phase 2.4 — external baseline R.** Run FiD-base or LongT5 fine-tuned on MuSiQue. ~3 days.
6. **Phase 2.5 — cross-benchmark.** Re-run B0/B6/B7 on 2Wiki then HotpotQA. ~3 days.
7. **Phase 2.6 — `/kill-argument` adversarial review + writeup.** Run on the final result table; iterate on any surviving rebuttal. ~3 days.

Total ~21 calendar days of single-grad-student effort, ~80–120 GPU-hours on chen's A100s.

---

## 11. Pre-registered success criteria

Before any Phase 2 training, write to `runs/phase2/preregister.json`:

A Phase 2 result is a **win** iff *all four* hold on MuSiQue dev (after filtering, n ≥ 2000 items per bucket):

1. **Absolute EM and F1 of B0 are > B2 (frozen agents) by a statistically significant margin** (paired t-test, p < 0.01 over 5 seeds). I.e. the coupled MAS does something useful.
2. **Per-bucket role asymmetry on B0 is at least 2× in A-only and 2× in B-only buckets,** where asymmetry := Δa/Δb (or Δb/Δa), measured on F1.
3. **The asymmetry collapses to ≤ 1.5× under both B6 (identical_ctx) and B7 (context_shuffled).** Difference-in-differences paired t-test p < 0.01 between B0 and {B6, B7}.
4. **The collapse direction reproduces on 2WikiMultiHopQA bridge subset** (smaller absolute asymmetry is OK, just same sign of B0 − B6 effect, p < 0.05).

If 1–3 hold but 4 fails, the result is **MuSiQue-only** and goes back for cross-benchmark analysis before paper draft. If 1 fails (no absolute gain), pivot the framing to "specialization alone, not accuracy" — but only if 2 and 3 still hold; otherwise abandon the benchmark contribution and write a *position paper* on the synthetic-task evidence alone.

---

## 12. Open questions to resolve before kicking off Phase 2.0

These need a decision before any code:

1. **Backbone choice — ELF-B-de-en, ELF-B-xsum, or ELF-B-owt?** Hot take: ELF-B-xsum (summarization-pretrained) is probably best aligned with QA. Need a 3-way frozen-baseline comparison on MuSiQue dev to confirm. ~1 GPU-day.
2. **Filter strictness — F1 > 0.7 (strict) vs F1 > 0.8 (loose) for the single-hop-shortcut rule?** Strict gives smaller but cleaner train; loose gives more items but accepts some shortcut leakage. Run a sensitivity check: keep an item-level shortcut score and report both.
3. **Decoder fidelity — NN candidate decode vs autoregressive decode for the headline metric?** NN is faster and simpler but capped at the candidate vocabulary; autoregressive is more general but harder to debug. Probably ship both, pick the higher-F1 one as headline.
4. **`S_ctx` budget — 64 (current) vs 128 vs 192?** Increasing this is the most likely real cost. Decision waits for F7 mitigation choice.
5. **5 seeds at full scale, or 3?** Statistical power vs compute time. With 20K steps × 5 variants × 5 seeds = 125 runs × ~8 min each ≈ 17 GPU-hours — easily fits in one chen day. Recommend 5.

Once these are decided, Phase 2.0 implementation is mechanical.
