"""2-Fact Composition QA generator — RANDOMIZED-KG variant.

Difference vs synth_2fact.py: there is NO global knowledge graph. Every example
draws its own random person->city and city->country (and personal attributes)
assignments, stated ONLY in the contexts. So the ANSWER IS NOT DETERMINED BY THE
PERSON NAME (or any prefix token) — it can only be recovered by reading ctx_A
(+ ctx_B for AB). This removes the person-name prefix shortcut that let a linear
probe decode the answer at layer ~5 with no context.

Same bucket structure (AB / A-only / B-only / neither), same proportions, same
templates, same distractor design (relevant fact unique within its context;
distractors are random facts about distinct other entities). Schema identical to
synth_2fact.py so it drops into the existing Cola training/eval pipeline.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
from datasets import Dataset, Features, Sequence, Value
from transformers import AutoTokenizer

# -------------------------- vocabulary (identical to synth_2fact.py) --------------------------

FIRST_NAMES = [
    "Alice", "Bob", "Charlie", "Diana", "Ethan", "Fiona", "George", "Hannah",
    "Ivan", "Julia", "Kevin", "Laura", "Marcus", "Nina", "Oscar", "Paula",
    "Quincy", "Rachel", "Sam", "Tina", "Uma", "Victor", "Wendy", "Xavier",
    "Yara", "Zoe", "Adam", "Bella", "Cody", "Daisy", "Eric", "Felix",
    "Gina", "Henry", "Iris", "Jack", "Karen", "Leo", "Mia", "Noah",
    "Olivia", "Peter", "Quinn", "Rose", "Sara", "Tom", "Ulysses", "Vera",
    "Will", "Yvonne", "Zack", "Anna", "Brian", "Clara", "Derek", "Emma",
    "Frank", "Grace", "Hugo", "Isla", "James", "Katie", "Liam", "Maya",
    "Nate", "Olga", "Paul", "Rita", "Steve", "Tara", "Ursula", "Vince",
    "Wade", "Yuki", "Zara", "Amelia", "Ben", "Carmen", "Dean", "Elena",
    "Finn", "Gloria", "Hank", "Ivy", "Joel", "Kate", "Lucas", "Megan",
    "Nora", "Owen", "Penny", "Ralph", "Sophie", "Theo", "Una", "Vivian",
    "Wes", "Yael", "Zoe2", "Arthur",
]
assert len(FIRST_NAMES) == 100

CITIES = [
    "Berlin", "Tokyo", "Cairo", "Paris", "Madrid", "Rome", "Vienna", "Athens",
    "Lisbon", "Oslo", "Helsinki", "Warsaw", "Prague", "Budapest", "Dublin", "Brussels",
    "Amsterdam", "Stockholm", "Copenhagen", "Reykjavik", "Sofia", "Bucharest", "Riga", "Tallinn",
    "Vilnius", "Zagreb", "Sarajevo", "Belgrade", "Skopje", "Tirana", "Valletta", "Nicosia",
    "Cairo2", "Lagos", "Nairobi", "Accra", "Dakar", "Algiers", "Rabat", "Tunis",
    "Lima", "Bogota", "Quito", "Santiago", "Caracas", "Asuncion", "Montevideo", "LaPaz",
    "Bangkok", "Manila",
]
assert len(CITIES) == 50

COUNTRIES = [
    "Germany", "Japan", "Egypt", "France", "Spain", "Italy", "Austria", "Greece",
    "Portugal", "Norway", "Finland", "Poland", "Czechia", "Hungary", "Ireland", "Belgium",
    "Netherlands", "Sweden", "Denmark", "Iceland",
]
assert len(COUNTRIES) == 20

OCCUPATIONS = [
    "engineer", "chef", "pilot", "doctor", "teacher", "lawyer", "musician", "writer",
    "farmer", "scientist", "artist", "soldier", "sailor", "banker", "baker", "barber",
    "tailor", "carpenter", "plumber", "electrician", "designer", "architect", "nurse", "dentist",
    "pharmacist", "accountant", "auditor", "consultant", "actor", "singer", "dancer", "athlete",
    "coach", "journalist", "editor", "photographer", "translator", "librarian", "researcher", "analyst",
]
assert len(OCCUPATIONS) == 40

HOBBIES = [
    "chess", "painting", "reading", "hiking", "cycling", "swimming", "gardening", "cooking",
    "fishing", "running", "yoga", "knitting", "pottery", "skiing", "surfing", "climbing",
    "dancing", "singing", "drawing", "photography",
]
assert len(HOBBIES) == 20


# -------------------------- fact rendering (identical) --------------------------

def render_lives_in(p, c):     return f"{p} lives in {c}."
def render_works_as(p, j):     return f"{p} works as {'an' if j[0] in 'aeiou' else 'a'} {j}."
def render_hobby_of(p, h):     return f"{p} enjoys {h}."
def render_city_in(c, co):     return f"{c} is in {co}."
def render_capital_of(co, c):  return f"The capital of {co} is {c}."


# -------------------------- randomized distractors (no global KG) --------------------------

def _personal_distractors(rng, n_extra: int, exclude_persons: set) -> List[str]:
    """n_extra random personal facts about DISTINCT persons not in exclude_persons.
    Distinct persons => no contradictory facts about the same person."""
    facts, used = [], set(exclude_persons)
    while len(facts) < n_extra:
        cands = [p for p in FIRST_NAMES if p not in used]
        if not cands:
            break
        p = str(rng.choice(cands)); used.add(p)
        kind = int(rng.integers(0, 3))
        if kind == 0:
            facts.append(render_lives_in(p, str(rng.choice(CITIES))))
        elif kind == 1:
            facts.append(render_works_as(p, str(rng.choice(OCCUPATIONS))))
        else:
            facts.append(render_hobby_of(p, str(rng.choice(HOBBIES))))
    return facts


def _geo_distractors(rng, n_extra: int, exclude_cities: set, exclude_countries: set) -> List[str]:
    """n_extra random geo facts using DISTINCT cities not in exclude_cities, and
    countries not in exclude_countries. Distinct cities => no contradictory
    'city is in / is capital of' facts about the same city."""
    facts, used_cities = [], set(exclude_cities)
    while len(facts) < n_extra:
        cands_c = [c for c in CITIES if c not in used_cities]
        cands_co = [x for x in COUNTRIES if x not in exclude_countries] or COUNTRIES
        if not cands_c:
            break
        c = str(rng.choice(cands_c)); used_cities.add(c)
        co = str(rng.choice(cands_co))
        if int(rng.integers(0, 2)) == 0:
            facts.append(render_city_in(c, co))
        else:
            facts.append(render_capital_of(co, c))
    return facts


def _shuffle(rng, facts: List[str]) -> List[str]:
    facts = list(facts); rng.shuffle(facts); return facts


# -------------------------- per-item generation (per-example random) --------------------------

def gen_AB_item(rng) -> dict:
    """lives_in(P,C) in ctx_A + city_in(C,Country) in ctx_B, all RANDOM per item.
    Q: 'What country does P live in?'  A: Country. Answer not determined by P."""
    p = str(rng.choice(FIRST_NAMES))
    c = str(rng.choice(CITIES))
    co = str(rng.choice(COUNTRIES))
    ctx_a = _shuffle(rng, [render_lives_in(p, c)]
                     + _personal_distractors(rng, int(rng.integers(2, 4)), {p}))
    ctx_b = _shuffle(rng, [render_city_in(c, co)]
                     + _geo_distractors(rng, int(rng.integers(2, 4)), {c}, {co}))
    return {"bucket": "AB", "ctx_a_text": " ".join(ctx_a), "ctx_b_text": " ".join(ctx_b),
            "question": f"What country does {p} live in?", "answer": co}


def gen_A_only_item(rng) -> dict:
    p = str(rng.choice(FIRST_NAMES))
    relation = int(rng.integers(0, 3))
    if relation == 0:
        ans = str(rng.choice(CITIES)); fact_a = render_lives_in(p, ans)
        question = f"In what city does {p} live?"
    elif relation == 1:
        ans = str(rng.choice(OCCUPATIONS)); fact_a = render_works_as(p, ans)
        question = f"What does {p} do for work?"
    else:
        ans = str(rng.choice(HOBBIES)); fact_a = render_hobby_of(p, ans)
        question = f"What does {p} enjoy doing?"
    ctx_a = _shuffle(rng, [fact_a] + _personal_distractors(rng, int(rng.integers(2, 4)), {p}))
    ctx_b = _shuffle(rng, _geo_distractors(rng, int(rng.integers(3, 5)), set(), set()))
    return {"bucket": "A-only", "ctx_a_text": " ".join(ctx_a), "ctx_b_text": " ".join(ctx_b),
            "question": question, "answer": ans}


def gen_B_only_item(rng) -> dict:
    if int(rng.integers(0, 2)) == 0:
        c = str(rng.choice(CITIES)); co = str(rng.choice(COUNTRIES))
        fact_b = render_city_in(c, co); question = f"Which country is {c} in?"; ans = co
    else:
        co = str(rng.choice(COUNTRIES)); c = str(rng.choice(CITIES))
        fact_b = render_capital_of(co, c); question = f"What is the capital of {co}?"; ans = c
    ctx_b = _shuffle(rng, [fact_b] + _geo_distractors(rng, int(rng.integers(2, 4)), {c}, {co}))
    ctx_a = _shuffle(rng, _personal_distractors(rng, int(rng.integers(3, 5)), set()))
    return {"bucket": "B-only", "ctx_a_text": " ".join(ctx_a), "ctx_b_text": " ".join(ctx_b),
            "question": question, "answer": ans}


def gen_neither_item(rng) -> dict:
    target = str(rng.choice(FIRST_NAMES))
    relation = int(rng.integers(0, 3))
    question = {0: f"In what city does {target} live?",
                1: f"What does {target} do for work?",
                2: f"What does {target} enjoy doing?"}[relation]
    ctx_a = _shuffle(rng, _personal_distractors(rng, int(rng.integers(3, 5)), {target}))
    ctx_b = _shuffle(rng, _geo_distractors(rng, int(rng.integers(3, 5)), set(), set()))
    return {"bucket": "neither", "ctx_a_text": " ".join(ctx_a), "ctx_b_text": " ".join(ctx_b),
            "question": question, "answer": "unknown"}


GEN_FNS = {"AB": gen_AB_item, "A-only": gen_A_only_item, "B-only": gen_B_only_item, "neither": gen_neither_item}
PROPORTIONS = {"AB": 0.40, "A-only": 0.25, "B-only": 0.25, "neither": 0.10}


def generate_split(rng, n: int) -> List[dict]:
    counts = {b: round(n * p) for b, p in PROPORTIONS.items()}
    counts["AB"] += n - sum(counts.values())
    items: List[dict] = []
    for bucket, k in counts.items():
        for _ in range(k):
            items.append(GEN_FNS[bucket](rng))
    rng.shuffle(items)
    return items


# -------------------------- tokenization + saving (identical schema) --------------------------

def tokenize_items(items, tokenizer, max_ctx, max_q, max_ans, max_combined):
    cols = {k: [] for k in ["bucket", "ctx_a_text", "ctx_b_text", "question", "answer",
                            "ctx_a_input_ids", "ctx_b_input_ids", "question_input_ids",
                            "input_ids", "condition_input_ids"]}
    for it in items:
        ca = tokenizer(it["ctx_a_text"], add_special_tokens=False, truncation=True, max_length=max_ctx)["input_ids"]
        cb = tokenizer(it["ctx_b_text"], add_special_tokens=False, truncation=True, max_length=max_ctx)["input_ids"]
        q = tokenizer(it["question"], add_special_tokens=False, truncation=True, max_length=max_q)["input_ids"]
        a = tokenizer(it["answer"], add_special_tokens=False, truncation=True, max_length=max_ans)["input_ids"]
        cond = tokenizer(f"{it['ctx_a_text']} {it['ctx_b_text']} {it['question']}",
                         add_special_tokens=False, truncation=True, max_length=max_combined)["input_ids"]
        for k, v in [("bucket", it["bucket"]), ("ctx_a_text", it["ctx_a_text"]), ("ctx_b_text", it["ctx_b_text"]),
                     ("question", it["question"]), ("answer", it["answer"]), ("ctx_a_input_ids", ca),
                     ("ctx_b_input_ids", cb), ("question_input_ids", q), ("input_ids", a),
                     ("condition_input_ids", cond)]:
            cols[k].append(v)
    return cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=43)        # different default seed from original
    parser.add_argument("--n_train", type=int, default=12_000)
    parser.add_argument("--n_eval", type=int, default=3_000)
    parser.add_argument("--tokenizer", type=str, default="t5-small")
    parser.add_argument("--max_ctx", type=int, default=64)
    parser.add_argument("--max_q", type=int, default=24)
    parser.add_argument("--max_ans", type=int, default=8)
    parser.add_argument("--max_combined", type=int, default=128)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rng_train = np.random.default_rng(args.seed + 1)
    rng_eval = np.random.default_rng(args.seed + 2)
    print(f"[gen] RANDOMIZED-KG: {args.n_train} train + {args.n_eval} eval items")
    train_items = generate_split(rng_train, args.n_train)
    eval_items = generate_split(rng_eval, args.n_eval)
    print(f"[gen] train buckets: {Counter(it['bucket'] for it in train_items)}")
    print(f"[gen] eval  buckets: {Counter(it['bucket'] for it in eval_items)}")

    print(f"[gen] loading tokenizer {args.tokenizer!r}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    train_cols = tokenize_items(train_items, tok, args.max_ctx, args.max_q, args.max_ans, args.max_combined)
    eval_cols = tokenize_items(eval_items, tok, args.max_ctx, args.max_q, args.max_ans, args.max_combined)

    features = Features({
        "bucket": Value("string"), "ctx_a_text": Value("string"), "ctx_b_text": Value("string"),
        "question": Value("string"), "answer": Value("string"),
        "ctx_a_input_ids": Sequence(Value("int32")), "ctx_b_input_ids": Sequence(Value("int32")),
        "question_input_ids": Sequence(Value("int32")), "input_ids": Sequence(Value("int32")),
        "condition_input_ids": Sequence(Value("int32")),
    })
    Dataset.from_dict(train_cols, features=features).save_to_disk(str(out_dir / "train"))
    Dataset.from_dict(eval_cols, features=features).save_to_disk(str(out_dir / "eval"))
    print(f"[gen] saved {out_dir}/train ({len(train_items)}) and /eval ({len(eval_items)})")

    manifest = {
        "variant": "randomized_kg", "seed": args.seed,
        "n_train": len(train_items), "n_eval": len(eval_items), "tokenizer": args.tokenizer,
        "bucket_proportions": PROPORTIONS,
        "note": "Per-example random person->city and city->country; answer NOT determined by person name.",
        "kg_sizes": {"n_people": len(FIRST_NAMES), "n_cities": len(CITIES), "n_countries": len(COUNTRIES),
                     "n_occupations": len(OCCUPATIONS), "n_hobbies": len(HOBBIES)},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[gen] manifest -> {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
