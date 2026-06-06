"""In-context baseline: frozen Cola, NO MAS, NO training.

Sends a one-shot SQuAD-style prompt that INCLUDES both contexts:
    <demo> Context: {ctx_a} {ctx_b}\nQuestion: {q}\nAnswer:
and generates the answer block with the pure frozen backbone (lora_wrappers=[]).
Uses the SAME 200-item selection (seed 0, rng.choice) as eval_cola_mas, so the
per-bucket numbers are directly comparable to the B0/B4/... lesion matrix.
"""
from __future__ import annotations
import argparse, json, time
from collections import Counter, defaultdict
import numpy as np
import torch
from datasets import load_from_disk

from elf_mas_pt.model.frozen_cola import FrozenColaBackbone, FrozenColaSpec
from elf_mas_pt.eval.eval_cola_mas import (
    sample_mas_block, decode_block_to_texts, decode_block_to_token_ids,
    per_token_em, per_token_f1, em as text_em, f1 as text_f1,
)
from elf_mas_pt.training.train_cola import (
    encode_texts_to_padded_latents, encode_prompts_per_sample_block_aligned, _SQUAD_DEMO,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", default="/home/chenhongrui/elf_mas/data/synth_2fact_v1")
    p.add_argument("--split", default="eval")
    p.add_argument("--cola_base", default="/home/chenhongrui/elf_mas/hf_cache/elf-mas-bridge/phase3_assets/cola_dlm")
    p.add_argument("--cola_src", default="/home/chenhongrui/elf_mas/cola_dlm_src")
    p.add_argument("--n_items", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--T_inf", type=int, default=16)
    p.add_argument("--S_q", type=int, default=16)
    p.add_argument("--S_ctx", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="")
    args = p.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    print("[ic] loading Cola backbone", flush=True)
    spec = FrozenColaSpec(
        dit_path=f"{args.cola_base}/cola_dlm/cola_dit",
        vae_path=f"{args.cola_base}/cola_dlm/cola_vae",
        tokenizer_path=f"{args.cola_base}/tokenizer.json",
        cola_src_path=args.cola_src,
    )
    backbone = FrozenColaBackbone.load(spec, device)

    ds = load_from_disk(f"{args.dataset_dir}/{args.split}")
    rng = np.random.default_rng(args.seed)  # SAME selection as eval_cola_mas
    idx = rng.choice(len(ds), size=min(args.n_items, len(ds)), replace=False).tolist()
    items = list(ds.select(idx))
    print(f"[ic] {len(items)} items, buckets: {dict(Counter(r['bucket'] for r in items))}", flush=True)

    results = defaultdict(list)
    examples = []
    n_batches = (len(items) + args.batch_size - 1) // args.batch_size
    t0 = time.time()
    for b in range(n_batches):
        rows = items[b * args.batch_size:(b + 1) * args.batch_size]
        q = [r["question"] for r in rows]
        a = [r["ctx_a_text"] for r in rows]
        c = [r["ctx_b_text"] for r in rows]
        # encoded q/ctx are unused by the no-MAS path but kept for a valid call signature
        q_lat, q_mask = encode_texts_to_padded_latents(backbone, q, args.S_q)
        a_lat, a_mask = encode_texts_to_padded_latents(backbone, a, args.S_ctx)
        b_lat, b_mask = encode_texts_to_padded_latents(backbone, c, args.S_ctx)
        # IN-CONTEXT prompt: both contexts go into Cola's own prefix
        prompts = [_SQUAD_DEMO + f"Context: {a[i]} {c[i]}\nQuestion: {q[i]}\nAnswer:"
                   for i in range(len(rows))]
        prompt_lat_list, prompt_lens = encode_prompts_per_sample_block_aligned(backbone, prompts)

        z_pred = sample_mas_block(
            backbone, None,
            q_lat=q_lat, q_mask=q_mask,
            ctx_a_lat=a_lat, ctx_a_mask=a_mask,
            ctx_b_lat=b_lat, ctx_b_mask=b_mask,
            T_inf=args.T_inf, ablation="AB", prefix_cond=True,
            prompt_lat_list=prompt_lat_list, prompt_lens=prompt_lens,
            lora_wrappers=[],  # <-- pure frozen backbone, no MAS
        )
        preds_text = decode_block_to_texts(backbone, z_pred)
        preds_tok = decode_block_to_token_ids(backbone, z_pred)
        for r, pt, ptok in zip(rows, preds_text, preds_tok):
            gold = r["answer"]
            results[r["bucket"]].append((
                text_em(pt, gold), text_f1(pt, gold),
                per_token_em(ptok, gold, backbone.tokenizer),
                per_token_f1(ptok, gold, backbone.tokenizer),
            ))
            if len(examples) < 10:
                K = len(backbone.tokenizer.encode(gold).ids)
                examples.append({"bucket": r["bucket"], "q": r["question"], "gold": gold,
                                 "pred": pt, "pred_tok_firstK": ptok[:K]})
        if b % 5 == 0 or b == n_batches - 1:
            print(f"[ic] batch {b+1}/{n_batches} ({time.time()-t0:.0f}s)", flush=True)

    summary = {}
    for bk in sorted(results):
        s = results[bk]
        summary[bk] = {
            "em": float(np.mean([x[0] for x in s])),
            "f1": float(np.mean([x[1] for x in s])),
            "em_tok": float(np.mean([x[2] for x in s])),
            "f1_tok": float(np.mean([x[3] for x in s])),
            "n": len(s),
        }
    overall_tok = float(np.mean([x[2] for s in results.values() for x in s]))
    print("\n[ic] IN-CONTEXT (frozen Cola, both ctx in prompt, no MAS) per-token EM:")
    print("%-9s %6s %8s %8s" % ("bucket", "n", "em_tok", "f1_tok"))
    for bk in ["AB", "A-only", "B-only", "neither"]:
        if bk in summary:
            d = summary[bk]
            print("%-9s %6d %8.3f %8.3f" % (bk, d["n"], d["em_tok"], d["f1_tok"]))
    print("overall em_tok = %.3f" % overall_tok)
    print("\n[ic] examples:")
    for e in examples[:8]:
        print(f"  [{e['bucket']:7s}] Q={e['q']!r} gold={e['gold']!r} pred={e['pred']!r} firstK={e['pred_tok_firstK']}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "overall_em_tok": overall_tok,
                       "examples": examples, "args": vars(args)}, f, indent=2)
        print(f"[ic] saved to {args.out}")


if __name__ == "__main__":
    main()
