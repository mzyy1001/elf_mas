# Synthetic Task v1 — 2-Fact Composition QA

**Date:** 2026-05-16
**Purpose:** Phase 1 pilot task for Coupled Multi-Agent Language Flows. Designed so that **role-specific lesion drops are provable by construction** (PLAN.md §10 R9, novelty-check v2 §3).
**Output:** HF `datasets.Dataset` saved to disk, T5-small-tokenized, ELF-compatible schema.

---

## 1. Task definition

A QA task over a tiny synthetic knowledge graph. Each item consists of:

- a **question** `Q` in natural English,
- a private context `ctx_A` given only to **Agent A** (Evidence-A),
- a private context `ctx_B` given only to **Agent B** (Evidence-B),
- a **gold answer** `Y` (short string).

Both contexts are written in English. Distractors (irrelevant facts) are present in both `ctx_A` and `ctx_B`.

## 2. Item buckets (the load-bearing design)

Every item is generated in **exactly one** of four buckets. The bucket determines which agent's private context is strictly necessary to answer `Q`:

| Bucket | Needs `ctx_A`? | Needs `ctx_B`? | Generation pattern |
|---|---|---|---|
| **AB** | ✅ | ✅ | `Q` requires composing one fact from `ctx_A` with one fact from `ctx_B` |
| **A-only** | ✅ | ❌ | `Q` is answerable from `ctx_A` alone; `ctx_B` is distractor-only |
| **B-only** | ❌ | ✅ | symmetric |
| **neither** | ❌ | ❌ | `Q` is unanswerable from both contexts (test of confabulation suppression) |

**Bucket proportions in train+eval:** `AB:A-only:B-only:neither = 40:25:25:10`.

**Why this is the right design.** Bucket membership is the **ground truth for which agent should matter**, by construction. Hostile reviewer (novelty-check v2 §5) said: gate plots alone don't survive — we need *causal evidence-grounded lesion patterns*. With this schema, when we ablate Agent A by zeroing its velocity residual `v_A`, the **predicted lesion drop pattern** is:

| Bucket | Lesion Agent A → expected drop | Lesion Agent B → expected drop |
|---|---|---|
| AB | **large** (joint fact needed) | **large** |
| A-only | **large** | none |
| B-only | none | **large** |
| neither | none | none |

If our MAS shows this pattern (per-bucket interaction with per-agent lesion), we have *causal* role-specific specialization. If it doesn't, the headline claim collapses, and we will know exactly why — that is the value of this design.

## 3. Vocabulary and KG schema

Tiny closed-world KG to keep ELF-B (105M, frozen T5-small encoder) within its competence:

- **100 people** (random first names from a fixed pool, e.g. `Alice`, `Bob`, `Charlie`, …)
- **50 cities** (`Berlin`, `Tokyo`, `Cairo`, …)
- **20 countries** (`Germany`, `Japan`, `Egypt`, …)
- **40 occupations** (`engineer`, `chef`, `pilot`, …)
- **20 hobbies** (`chess`, `painting`, …)

**Relations used (5 total):**

| Rel | Subject type | Object type | English template |
|---|---|---|---|
| `lives_in` | person | city | `"X lives in Y."` |
| `capital_of` | country | city | `"The capital of Y is X."` |
| `works_as` | person | occupation | `"X works as a Y."` |
| `hobby_of` | person | hobby | `"X enjoys Y."` |
| `city_in` | city | country | `"Y is in X."` |

## 4. Example items per bucket

**AB example (composition of `lives_in` + `city_in`):**
- `ctx_A`: `"Alice lives in Berlin. Bob works as an engineer. Charlie enjoys chess."`
- `ctx_B`: `"Berlin is in Germany. Cairo is in Egypt. Tokyo is in Japan."`
- `Q`: `"What country is the city where Alice lives in?"`
- `Y`: `"Germany"`
- Reasoning: requires `lives_in(Alice, Berlin)` (only in A) + `city_in(Berlin, Germany)` (only in B).

**A-only example:**
- `ctx_A`: `"Bob lives in Tokyo. Alice works as a pilot. Charlie enjoys painting."`
- `ctx_B`: `"Cairo is in Egypt. Diana enjoys chess. Berlin is in Germany."` (distractors)
- `Q`: `"What does Alice do for work?"`
- `Y`: `"pilot"`

**B-only example:** symmetric.

**neither example:**
- `ctx_A`: `"Alice lives in Berlin. Bob works as an engineer."`
- `ctx_B`: `"Tokyo is in Japan. Cairo is in Egypt."`
- `Q`: `"What hobby does Charlie enjoy?"` (no fact about Charlie's hobby in either context)
- `Y`: `"unknown"` (special unanswerable token)

## 5. Distractors

Each context has **1 fact relevant to the question** (if applicable) plus **2-3 distractor facts of the same schema**, drawn from the same vocabulary but disjoint entities/relations from the question. Distractors prevent trivial copying and force the agent to actually use its private context conditionally.

## 6. Lengths

T5-small tokenizer-based budget per item:

| Field | Target tokens | Notes |
|---|---|---|
| `ctx_A` | 30-60 | 3-4 facts, each ~10-15 toks |
| `ctx_B` | 30-60 | same |
| `Q` | 8-15 | one short interrogative |
| `Y` | 1-4 | short noun phrase |

ELF-B-de-en is trained at `max_length=128, max_input_length=64`. We will mirror that constraint: combined `ctx_A + ctx_B + Q` ≤ 128 tokens, answer ≤ 16 tokens.

## 7. Output dataset schema

Saved with `datasets.Dataset.save_to_disk()`. Each example has:

```python
{
  "bucket":      str,            # one of "AB" / "A-only" / "B-only" / "neither"
  "ctx_a_text":  str,            # raw English
  "ctx_b_text":  str,            # raw English
  "question":    str,            # raw English
  "answer":      str,            # raw English
  "ctx_a_input_ids": List[int],  # T5-small tokens
  "ctx_b_input_ids": List[int],
  "question_input_ids": List[int],
  "input_ids":   List[int],      # answer tokens (matches ELF's `input_ids` convention for the target)
  # For single-ELF baselines that concatenate everything as one condition:
  "condition_input_ids": List[int],  # = ctx_a + " " + ctx_b + " " + question, tokenized
}
```

This schema is **compatible with ELF's conditional eval path** (single-baseline uses `condition_input_ids` + `input_ids`) AND lets the MAS variants read `ctx_a_input_ids` and `ctx_b_input_ids` separately for the velocity-residual conditioning.

## 8. Dataset sizes

| Split | Total items | AB | A-only | B-only | neither |
|---|---|---|---|---|---|
| train | 50,000 | 20,000 | 12,500 | 12,500 | 5,000 |
| eval  | 5,000  | 2,000 | 1,250 | 1,250 | 500 |

**Why these sizes:** Phase 1 is a *can the mechanism work* check, not a SOTA push. 50K is enough for an ELF-B-adapter-only finetune to converge in 1-3 hours on a single A100. Eval is large enough that bucket-conditional accuracies have <1% sampling noise.

## 9. Generation determinism

`numpy.random.default_rng(seed=42)` for the fact graph + sampling. Generation is fully deterministic given the seed. The script regenerates the dataset reproducibly.

## 10. Files

- `src/elf_mas/data/synth_2fact.py` — generator: produces a HuggingFace `Dataset` with the schema in §7, saves to disk.
- `data/synth_2fact_v1/train` and `data/synth_2fact_v1/eval` — saved Arrow files on chen.

Both will live on chen under `~/elf_mas/data/synth_2fact_v1/`.

## 11. Sanity checks (run after first generation)

1. **Bucket counts** match §8 proportions.
2. **Per-bucket length distributions** — no truncation at 128.
3. **Held-out** 100 items hand-inspected for clean English and unambiguous answers.
4. **Frozen-ELF baseline forward pass** — run ELF-B-de-en's encoder on `condition_input_ids` for a handful of examples; verify shapes match what ELF expects.

The Phase 1 model code starts only after these four pass.

## 12. Out-of-scope for v1 (intentionally)

- Larger KG / longer reasoning chains. We can extend later if v1 gates separate cleanly.
- Multi-hop > 2 (3-fact composition is an ablation, not the headline).
- Non-English. ELF-B-de-en was pretrained for translation, but the task is *not* a translation task; we treat the encoder as a generic semantic encoder over English.
- Realistic noise (typos, paraphrases). All facts use the templates in §3 verbatim.

## 13. Risks for v1 specifically

| # | Risk | Mitigation |
|---|------|-----------|
| T1 | ELF-B-de-en's translation pretraining mis-fires on English-only QA | Acceptable: this task is for the *MAS mechanism*, not absolute accuracy. The single-ELF baseline (B4 in PLAN.md §7) suffers equally so the comparison is fair. |
| T2 | T5-small encoder can't distinguish near-duplicate entity names (e.g. `Alice` vs `Alicia`) | Use entity names that tokenize cleanly into 1-3 unique tokens. Check after generation. |
| T3 | Bucket leakage — distractor facts accidentally answer the question | Generator enforces disjoint entities between query-relevant facts and distractors per item. Sanity-check #3 catches the rest. |
| T4 | The model collapses to "always answer from `ctx_A`" because of training-data bias | Per-bucket counts in §8 give A-only and B-only equal mass (12,500 each), so any agent shortcut hurts validation. |
