"""2-Fact Composition QA generator for Coupled MAS Language Flows.

See notes/synthetic_task_v1.md for the design (buckets, schema, sizes).
Generates a HuggingFace `Dataset` saved with `save_to_disk` so it is
directly loadable by ELF's data path (`load_from_disk`).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from datasets import Dataset, Features, Sequence, Value
from transformers import AutoTokenizer

# -------------------------- vocabulary --------------------------

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
]  # 100 names
assert len(FIRST_NAMES) == 100, f"got {len(FIRST_NAMES)} names"

CITIES = [
    "Berlin", "Tokyo", "Cairo", "Paris", "Madrid", "Rome", "Vienna", "Athens",
    "Lisbon", "Oslo", "Helsinki", "Warsaw", "Prague", "Budapest", "Dublin", "Brussels",
    "Amsterdam", "Stockholm", "Copenhagen", "Reykjavik", "Sofia", "Bucharest", "Riga", "Tallinn",
    "Vilnius", "Zagreb", "Sarajevo", "Belgrade", "Skopje", "Tirana", "Valletta", "Nicosia",
    "Cairo2", "Lagos", "Nairobi", "Accra", "Dakar", "Algiers", "Rabat", "Tunis",
    "Lima", "Bogota", "Quito", "Santiago", "Caracas", "Asuncion", "Montevideo", "LaPaz",
    "Bangkok", "Manila",
]  # 50 cities
assert len(CITIES) == 50

COUNTRIES = [
    "Germany", "Japan", "Egypt", "France", "Spain", "Italy", "Austria", "Greece",
    "Portugal", "Norway", "Finland", "Poland", "Czechia", "Hungary", "Ireland", "Belgium",
    "Netherlands", "Sweden", "Denmark", "Iceland",
]  # 20 countries
assert len(COUNTRIES) == 20

OCCUPATIONS = [
    "engineer", "chef", "pilot", "doctor", "teacher", "lawyer", "musician", "writer",
    "farmer", "scientist", "artist", "soldier", "sailor", "banker", "baker", "barber",
    "tailor", "carpenter", "plumber", "electrician", "designer", "architect", "nurse", "dentist",
    "pharmacist", "accountant", "auditor", "consultant", "actor", "singer", "dancer", "athlete",
    "coach", "journalist", "editor", "photographer", "translator", "librarian", "researcher", "analyst",
]  # 40 occupations
assert len(OCCUPATIONS) == 40

HOBBIES = [
    "chess", "painting", "reading", "hiking", "cycling", "swimming", "gardening", "cooking",
    "fishing", "running", "yoga", "knitting", "pottery", "skiing", "surfing", "climbing",
    "dancing", "singing", "drawing", "photography",
]  # 20 hobbies
assert len(HOBBIES) == 20


# -------------------------- KG construction --------------------------

@dataclass
class KG:
    """Per-entity fact lookups. Each person has exactly one of each personal fact;
    each city is in exactly one country; each country has exactly one capital city."""
    person_city: Dict[str, str]            # person -> city they live in
    person_job: Dict[str, str]             # person -> occupation
    person_hobby: Dict[str, str]           # person -> hobby
    city_country: Dict[str, str]           # city -> country it is in
    country_capital: Dict[str, str]        # country -> capital city


def build_kg(rng: np.random.Generator) -> KG:
    """Build a deterministic KG from the fixed vocabulary."""
    # Every city is in exactly one country (random assignment, possibly repeated countries)
    city_country = {c: str(rng.choice(COUNTRIES)) for c in CITIES}
    # Each country has exactly one capital — pick one of its assigned cities
    country_to_cities: Dict[str, List[str]] = {co: [] for co in COUNTRIES}
    for city, co in city_country.items():
        country_to_cities[co].append(city)
    country_capital = {}
    for co in COUNTRIES:
        opts = country_to_cities[co]
        if not opts:  # shouldn't happen given counts but safe
            opts = [str(rng.choice(CITIES))]
        country_capital[co] = str(rng.choice(opts))
    # Per-person facts: random city / job / hobby
    person_city = {p: str(rng.choice(CITIES)) for p in FIRST_NAMES}
    person_job = {p: str(rng.choice(OCCUPATIONS)) for p in FIRST_NAMES}
    person_hobby = {p: str(rng.choice(HOBBIES)) for p in FIRST_NAMES}
    return KG(person_city, person_job, person_hobby, city_country, country_capital)


# -------------------------- fact rendering --------------------------

def render_lives_in(p: str, c: str) -> str:
    return f"{p} lives in {c}."


def render_works_as(p: str, j: str) -> str:
    art = "an" if j[0] in "aeiou" else "a"
    return f"{p} works as {art} {j}."


def render_hobby_of(p: str, h: str) -> str:
    return f"{p} enjoys {h}."


def render_city_in(c: str, co: str) -> str:
    return f"{c} is in {co}."


def render_capital_of(co: str, c: str) -> str:
    return f"The capital of {co} is {c}."


# -------------------------- per-item generation --------------------------

def _sample_personal_distractor_fact(rng: np.random.Generator, kg: KG, exclude_persons: set) -> str:
    """A random personal fact (lives_in/works_as/hobby_of) about someone NOT in exclude_persons."""
    candidates = [p for p in FIRST_NAMES if p not in exclude_persons]
    p = str(rng.choice(candidates))
    kind = int(rng.integers(0, 3))
    if kind == 0:
        return render_lives_in(p, kg.person_city[p])
    elif kind == 1:
        return render_works_as(p, kg.person_job[p])
    else:
        return render_hobby_of(p, kg.person_hobby[p])


def _sample_geo_distractor_fact(rng: np.random.Generator, kg: KG, exclude_cities: set, exclude_countries: set) -> str:
    """A random geo fact (city_in or capital_of) NOT involving excluded entities."""
    kind = int(rng.integers(0, 2))
    if kind == 0:
        cands = [c for c in CITIES if c not in exclude_cities and kg.city_country[c] not in exclude_countries]
        if not cands:
            cands = CITIES
        c = str(rng.choice(cands))
        return render_city_in(c, kg.city_country[c])
    else:
        cands = [co for co in COUNTRIES if co not in exclude_countries]
        if not cands:
            cands = COUNTRIES
        co = str(rng.choice(cands))
        return render_capital_of(co, kg.country_capital[co])


def _pad_with_distractors(rng: np.random.Generator, kg: KG, base: List[str],
                          sampler, exclude_args, n_extra: int) -> List[str]:
    """Append n_extra distractor facts to `base`, using the provided sampler."""
    facts = list(base)
    seen = set(facts)
    tries = 0
    while len([f for f in facts if f not in base]) < n_extra and tries < 50:
        f = sampler(rng, kg, *exclude_args)
        if f not in seen:
            facts.append(f)
            seen.add(f)
        tries += 1
    rng.shuffle(facts)
    return facts


def gen_AB_item(rng: np.random.Generator, kg: KG) -> dict:
    """Composition: lives_in(P, C) [in ctx_A] + city_in(C, Country) [in ctx_B].
    Question: 'What country does <P> live in?'  Answer: Country."""
    # Person -> city -> country chain that is fully resolvable
    while True:
        p = str(rng.choice(FIRST_NAMES))
        c = kg.person_city[p]
        co = kg.city_country[c]
        if co is not None:
            break
    fact_a = render_lives_in(p, c)
    fact_b = render_city_in(c, co)

    ctx_a_facts = _pad_with_distractors(
        rng, kg, [fact_a],
        _sample_personal_distractor_fact,
        ({p},),  # exclude P
        n_extra=int(rng.integers(2, 4)),
    )
    ctx_b_facts = _pad_with_distractors(
        rng, kg, [fact_b],
        _sample_geo_distractor_fact,
        ({c}, {co}),  # exclude C and Country
        n_extra=int(rng.integers(2, 4)),
    )
    return {
        "bucket": "AB",
        "ctx_a_text": " ".join(ctx_a_facts),
        "ctx_b_text": " ".join(ctx_b_facts),
        "question": f"What country does {p} live in?",
        "answer": co,
    }


def gen_A_only_item(rng: np.random.Generator, kg: KG) -> dict:
    """Single-fact query answerable from ctx_A. Question template depends on relation."""
    p = str(rng.choice(FIRST_NAMES))
    relation = int(rng.integers(0, 3))  # 0=lives_in (asks city), 1=works_as, 2=hobby
    if relation == 0:
        fact_a = render_lives_in(p, kg.person_city[p])
        question = f"In what city does {p} live?"
        answer = kg.person_city[p]
    elif relation == 1:
        fact_a = render_works_as(p, kg.person_job[p])
        question = f"What does {p} do for work?"
        answer = kg.person_job[p]
    else:
        fact_a = render_hobby_of(p, kg.person_hobby[p])
        question = f"What does {p} enjoy doing?"
        answer = kg.person_hobby[p]
    ctx_a_facts = _pad_with_distractors(
        rng, kg, [fact_a],
        _sample_personal_distractor_fact,
        ({p},),
        n_extra=int(rng.integers(2, 4)),
    )
    # ctx_B: pure distractors, geo facts. Make sure none mention P (P isn't in geo facts anyway).
    ctx_b_facts = _pad_with_distractors(
        rng, kg, [],
        _sample_geo_distractor_fact,
        (set(), set()),
        n_extra=int(rng.integers(3, 5)),
    )
    return {
        "bucket": "A-only",
        "ctx_a_text": " ".join(ctx_a_facts),
        "ctx_b_text": " ".join(ctx_b_facts),
        "question": question,
        "answer": answer,
    }


def gen_B_only_item(rng: np.random.Generator, kg: KG) -> dict:
    """Single geo fact answerable from ctx_B."""
    relation = int(rng.integers(0, 2))  # 0=city_in (asks country), 1=capital_of (asks capital)
    if relation == 0:
        c = str(rng.choice(CITIES))
        co = kg.city_country[c]
        fact_b = render_city_in(c, co)
        question = f"Which country is {c} in?"
        answer = co
    else:
        co = str(rng.choice(COUNTRIES))
        c = kg.country_capital[co]
        fact_b = render_capital_of(co, c)
        question = f"What is the capital of {co}?"
        answer = c
    ctx_b_facts = _pad_with_distractors(
        rng, kg, [fact_b],
        _sample_geo_distractor_fact,
        ({c}, {co}),
        n_extra=int(rng.integers(2, 4)),
    )
    # ctx_A: personal distractor facts only
    ctx_a_facts = _pad_with_distractors(
        rng, kg, [],
        _sample_personal_distractor_fact,
        (set(),),
        n_extra=int(rng.integers(3, 5)),
    )
    return {
        "bucket": "B-only",
        "ctx_a_text": " ".join(ctx_a_facts),
        "ctx_b_text": " ".join(ctx_b_facts),
        "question": question,
        "answer": answer,
    }


def gen_neither_item(rng: np.random.Generator, kg: KG) -> dict:
    """Question about an entity absent from both contexts. Answer = 'unknown'."""
    target = str(rng.choice(FIRST_NAMES))
    relation = int(rng.integers(0, 3))
    if relation == 0:
        question = f"In what city does {target} live?"
    elif relation == 1:
        question = f"What does {target} do for work?"
    else:
        question = f"What does {target} enjoy doing?"
    # Distractor contexts that do NOT mention `target`.
    ctx_a_facts = _pad_with_distractors(
        rng, kg, [],
        _sample_personal_distractor_fact,
        ({target},),
        n_extra=int(rng.integers(3, 5)),
    )
    ctx_b_facts = _pad_with_distractors(
        rng, kg, [],
        _sample_geo_distractor_fact,
        (set(), set()),
        n_extra=int(rng.integers(3, 5)),
    )
    return {
        "bucket": "neither",
        "ctx_a_text": " ".join(ctx_a_facts),
        "ctx_b_text": " ".join(ctx_b_facts),
        "question": question,
        "answer": "unknown",
    }


GEN_FNS = {
    "AB": gen_AB_item,
    "A-only": gen_A_only_item,
    "B-only": gen_B_only_item,
    "neither": gen_neither_item,
}

PROPORTIONS = {"AB": 0.40, "A-only": 0.25, "B-only": 0.25, "neither": 0.10}


def generate_split(rng: np.random.Generator, kg: KG, n: int) -> List[dict]:
    counts = {b: round(n * p) for b, p in PROPORTIONS.items()}
    # Adjust rounding to hit exact n
    diff = n - sum(counts.values())
    counts["AB"] += diff
    items: List[dict] = []
    for bucket, k in counts.items():
        fn = GEN_FNS[bucket]
        for _ in range(k):
            items.append(fn(rng, kg))
    rng.shuffle(items)
    return items


# -------------------------- tokenization + saving --------------------------

def tokenize_items(items: List[dict], tokenizer, max_ctx: int, max_q: int,
                   max_ans: int, max_combined: int) -> Dict[str, list]:
    """Add tokenized fields to each item. Returns columns-of-lists."""
    cols: Dict[str, list] = {
        "bucket": [], "ctx_a_text": [], "ctx_b_text": [], "question": [], "answer": [],
        "ctx_a_input_ids": [], "ctx_b_input_ids": [], "question_input_ids": [],
        "input_ids": [], "condition_input_ids": [],
    }
    for it in items:
        ctx_a_ids = tokenizer(it["ctx_a_text"], add_special_tokens=False, truncation=True, max_length=max_ctx)["input_ids"]
        ctx_b_ids = tokenizer(it["ctx_b_text"], add_special_tokens=False, truncation=True, max_length=max_ctx)["input_ids"]
        q_ids = tokenizer(it["question"], add_special_tokens=False, truncation=True, max_length=max_q)["input_ids"]
        ans_ids = tokenizer(it["answer"], add_special_tokens=False, truncation=True, max_length=max_ans)["input_ids"]
        combined_text = f"{it['ctx_a_text']} {it['ctx_b_text']} {it['question']}"
        cond_ids = tokenizer(combined_text, add_special_tokens=False, truncation=True, max_length=max_combined)["input_ids"]
        cols["bucket"].append(it["bucket"])
        cols["ctx_a_text"].append(it["ctx_a_text"])
        cols["ctx_b_text"].append(it["ctx_b_text"])
        cols["question"].append(it["question"])
        cols["answer"].append(it["answer"])
        cols["ctx_a_input_ids"].append(ctx_a_ids)
        cols["ctx_b_input_ids"].append(ctx_b_ids)
        cols["question_input_ids"].append(q_ids)
        cols["input_ids"].append(ans_ids)
        cols["condition_input_ids"].append(cond_ids)
    return cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_train", type=int, default=50_000)
    parser.add_argument("--n_eval", type=int, default=5_000)
    parser.add_argument("--tokenizer", type=str, default="t5-small",
                        help="HF model id or local path for the T5 tokenizer")
    parser.add_argument("--max_ctx", type=int, default=64)
    parser.add_argument("--max_q", type=int, default=24)
    parser.add_argument("--max_ans", type=int, default=8)
    parser.add_argument("--max_combined", type=int, default=128)
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output dir; will write {out_dir}/train and {out_dir}/eval")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    kg = build_kg(rng)
    # Use distinct rng streams for splits so the eval split is independent
    rng_train = np.random.default_rng(args.seed + 1)
    rng_eval = np.random.default_rng(args.seed + 2)

    print(f"[gen] generating {args.n_train} train + {args.n_eval} eval items")
    train_items = generate_split(rng_train, kg, args.n_train)
    eval_items = generate_split(rng_eval, kg, args.n_eval)
    print(f"[gen] train buckets: {Counter(it['bucket'] for it in train_items)}")
    print(f"[gen] eval  buckets: {Counter(it['bucket'] for it in eval_items)}")

    print(f"[gen] loading tokenizer {args.tokenizer!r}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    print(f"[gen] tokenizing train ...")
    train_cols = tokenize_items(train_items, tok, args.max_ctx, args.max_q, args.max_ans, args.max_combined)
    print(f"[gen] tokenizing eval ...")
    eval_cols = tokenize_items(eval_items, tok, args.max_ctx, args.max_q, args.max_ans, args.max_combined)

    features = Features({
        "bucket": Value("string"),
        "ctx_a_text": Value("string"),
        "ctx_b_text": Value("string"),
        "question": Value("string"),
        "answer": Value("string"),
        "ctx_a_input_ids": Sequence(Value("int32")),
        "ctx_b_input_ids": Sequence(Value("int32")),
        "question_input_ids": Sequence(Value("int32")),
        "input_ids": Sequence(Value("int32")),
        "condition_input_ids": Sequence(Value("int32")),
    })
    ds_train = Dataset.from_dict(train_cols, features=features)
    ds_eval = Dataset.from_dict(eval_cols, features=features)

    train_path = out_dir / "train"
    eval_path = out_dir / "eval"
    ds_train.save_to_disk(str(train_path))
    ds_eval.save_to_disk(str(eval_path))

    print(f"[gen] saved train: {train_path} ({len(ds_train)} items)")
    print(f"[gen] saved eval:  {eval_path} ({len(ds_eval)} items)")

    # length distribution sanity
    def lendist(col):
        ls = [len(x) for x in col]
        return {"min": min(ls), "median": int(np.median(ls)), "p95": int(np.percentile(ls, 95)), "max": max(ls)}
    print(f"[gen] train condition_input_ids lengths: {lendist(train_cols['condition_input_ids'])}")
    print(f"[gen] train input_ids (answer) lengths:  {lendist(train_cols['input_ids'])}")
    print(f"[gen] train ctx_a_input_ids lengths:     {lendist(train_cols['ctx_a_input_ids'])}")
    print(f"[gen] train ctx_b_input_ids lengths:     {lendist(train_cols['ctx_b_input_ids'])}")

    # Save manifest for downstream consumers
    manifest = {
        "seed": args.seed,
        "n_train": len(ds_train),
        "n_eval": len(ds_eval),
        "tokenizer": args.tokenizer,
        "max_ctx": args.max_ctx, "max_q": args.max_q, "max_ans": args.max_ans, "max_combined": args.max_combined,
        "bucket_proportions": PROPORTIONS,
        "kg_sizes": {
            "n_people": len(FIRST_NAMES), "n_cities": len(CITIES), "n_countries": len(COUNTRIES),
            "n_occupations": len(OCCUPATIONS), "n_hobbies": len(HOBBIES),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[gen] manifest written to {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
