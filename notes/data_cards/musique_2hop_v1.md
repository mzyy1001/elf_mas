# MuSiQue-2hop v1 — Phase 2.0 processed dataset

- **Source:** `dgslibisey/MuSiQue` (downloaded 2026-05-18)
- **Filter:** 2-hop items only; SQuAD2 shortcut probe (deepset/roberta-base-squad2) with F1 > **0.7** rejected
- **Schema:** ELF-compatible (ctx_a_input_ids, ctx_b_input_ids, question_input_ids, input_ids, condition_input_ids); matches `synth_2fact_v1`
- **Budgets:** S_ctx=64, S_question=24, S_answer=8, S_combined=160 (T5-small)
- **Seed:** 42

## Reproduce
```bash
python -m elf_mas.data.musique_2hop \
    --musique_dir <musique_jsonl_dir> \
    --probe_dir   <roberta-base-squad2 dir> \
    --t5_path     <t5-small dir> \
    --out_dir     /home/chenhongrui/elf_mas/data/musique_2hop_v1 \
    --probe_thresh 0.7 --probe_batch 128
```

## Per-split summary

| Split | Raw N | After 2-hop | Probe dropped @0.7 | @0.8 | @0.9 | After probe | Final rows |
|---|---|---|---|---|---|---|---|
| train | 19938 | 14376 | 1466 | 1390 | 1359 | 12910 | 41896 |
| dev | 2417 | 1252 | 109 | 103 | 103 | 1143 | 3726 |

## Bucket counts

| Split | AB | A-only | B-only | neither |
|---|---|---|---|---|
| train | 12910 | 12910 | 12910 | 3166 |
| dev | 1143 | 1143 | 1143 | 297 |

## Truncation / evidence retention

| Split | mean gold_retained ctx_A | mean gold_retained ctx_B | items with full gold kept (frac) |
|---|---|---|---|
| train | 0.707 | 0.573 | 0.239 |
| dev | 0.660 | 0.601 | 0.210 |

`gold_retained` = (gold tokens kept in ctx_A or ctx_B after truncation) / (gold tokens before).
`full gold kept` = fraction of items where the supporting paragraph survived truncation without loss.

## Tokenized length distributions

### train
| field | min | median | p95 | max |
|---|---|---|---|---|
| condition_input_ids | 106 | 138 | 150 | 160 |
| ctx_a_input_ids | 39 | 64 | 64 | 64 |
| ctx_b_input_ids | 38 | 64 | 64 | 64 |
| question_input_ids | 3 | 11 | 22 | 24 |
| input_ids | 1 | 3 | 8 | 8 |

### dev
| field | min | median | p95 | max |
|---|---|---|---|---|
| condition_input_ids | 117 | 139 | 151 | 160 |
| ctx_a_input_ids | 50 | 64 | 64 | 64 |
| ctx_b_input_ids | 46 | 64 | 64 | 64 |
| question_input_ids | 3 | 12 | 23 | 24 |
| input_ids | 1 | 3 | 8 | 8 |

## Example items per bucket (one each)

### A-only
- **question:** who was the ruler of england in 1616
- **answer:** James I
- **ctx_a:** Kingdom of Denmark -- Christian IV (1588 -- 1648) Duchy of Schleswig -- Christian IV (1588 -- 1648) and John Adolphus (1590 -- 1616) in condominial rule Christian IV (1588 -- 1648) and Frederick III (...
- **ctx_b:** Partial Bible translations into languages of the English people can be traced back to the late 7th century, including translations into Old and Middle English. More than 450 translations into English ...

### neither
- **question:** who was the one who got away katy perry
- **answer:** unknown
- **ctx_a:** When Philippine de Rothschild was ten years old, she witnessed the Gestapo arrest her mother, who later died at Ravensbrück concentration camp, the only known member of the Rothschild family to die du...
- **ctx_b:** BCS: 50 Years is a review volume edited by Leon Cooper, a 1972 Nobel Laureate in Physics, and Dmitri Feldman of Brown University, first published in 2010. The Kaskapau Formation in northern Alberta re...

### AB
- **question:** In what episode of The Office does Dwight save Pam's husband from Roy?
- **answer:** ``The Negotiation ''
- **ctx_a:** The will they or won't they ''tension between Jim and Pam is a strong storyline in the early episodes of The Office, encompassing much of Seasons 1 to 3. In the opener of Season 4, the two characters ...
- **ctx_b:** The Negotiation ''(originally titled Labor Negotiation'') is the nineteenth episode of the third season of the American comedy television series The Office, and the show's forty - seventh episode over...

### B-only
- **question:** Woodrow Wilson >> child
- **answer:** Jessie Woodrow Wilson
- **ctx_a:** The 1827 State of the Union Address was written by John Quincy Adams, the sixth President of the United States. It was given on Tuesday, December 4, 1827, to the United States House of Representatives...
- **ctx_b:** Jessie Woodrow Wilson Sayre (August 28, 1887 – January 15, 1933) was a daughter of US President Woodrow Wilson and Ellen Louise Axson. She was a political activist, and "She worked vigorously for wome...
