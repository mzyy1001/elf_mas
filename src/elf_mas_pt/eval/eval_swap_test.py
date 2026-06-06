"""A/B swap test on B0 (full MAS), no retraining.

Eval-time intervention: feed ctx_B into agent A's head and ctx_A into agent B's
head (swap the contexts, keep the trained weights). If the two agents learned
non-interchangeable roles (cooperative division of labor), swapping should drop
EM sharply. Same 200-item selection (seed 0) as the lesion matrix.
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
import numpy as np
import torch
from datasets import load_from_disk

from elf_mas_pt.model.frozen_cola import FrozenColaBackbone, FrozenColaSpec
from elf_mas_pt.model.mas_in_block import patch_cola_dit_with_mas
from elf_mas_pt.eval.eval_cola_mas import (
    sample_mas_block, decode_block_to_token_ids, per_token_em, per_token_f1,
)
from elf_mas_pt.training.train_cola import build_batch_inputs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default="/home/chenhongrui/elf_mas/runs/cola_mas_sweep/full_seed0")
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
    spec = FrozenColaSpec(
        dit_path=f"{args.cola_base}/cola_dlm/cola_dit",
        vae_path=f"{args.cola_base}/cola_dlm/cola_vae",
        tokenizer_path=f"{args.cola_base}/tokenizer.json",
        cola_src_path=args.cola_src,
    )
    backbone = FrozenColaBackbone.load(spec, device)

    ckpt = torch.load(f"{args.ckpt_dir}/lora_state.pt", map_location=device)
    layers = ckpt["lora_layer_indices"]
    txt_dim = backbone.dit.config.txt_dim if hasattr(backbone.dit.config, "txt_dim") else 2048
    wrappers = patch_cola_dit_with_mas(
        backbone.dit, txt_dim=txt_dim, ctx_lat_dim=backbone.latent_dim,
        layer_indices=layers, inner_dim=ckpt["lora_inner"], num_heads=ckpt["lora_heads"],
        block_size=backbone.block_size, mode="dual",
    )
    for w, wc in zip(wrappers, ckpt["wrappers"]):
        w.mas_a.load_state_dict(wc["mas_a"]); w.mas_b.load_state_dict(wc["mas_b"]); w.eval()
    for pa in backbone.dit.parameters():
        pa.requires_grad_(False)

    ds = load_from_disk(f"{args.dataset_dir}/{args.split}")
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(ds), size=min(args.n_items, len(ds)), replace=False).tolist()
    items = list(ds.select(idx))
    print(f"[swap] {len(items)} items; buckets={dict(__import__('collections').Counter(r['bucket'] for r in items))}", flush=True)

    sample_rng = np.random.default_rng(args.seed + 1)
    scores = {"normal": defaultdict(list), "swapped": defaultdict(list)}
    n_batches = (len(items) + args.batch_size - 1) // args.batch_size
    for b in range(n_batches):
        rows = items[b * args.batch_size:(b + 1) * args.batch_size]
        batch = build_batch_inputs(backbone, rows, variant="full", S_ctx=args.S_ctx, S_q=args.S_q, rng=sample_rng)
        for mode in ("normal", "swapped"):
            if mode == "normal":
                ca, ma, cb, mb = batch["ctx_a_lat"], batch["ctx_a_mask"], batch["ctx_b_lat"], batch["ctx_b_mask"]
            else:  # feed ctx_B into agent A's head and ctx_A into agent B's head
                ca, ma, cb, mb = batch["ctx_b_lat"], batch["ctx_b_mask"], batch["ctx_a_lat"], batch["ctx_a_mask"]
            z = sample_mas_block(
                backbone, None,
                q_lat=batch["q_lat"], q_mask=batch["q_mask"],
                ctx_a_lat=ca, ctx_a_mask=ma, ctx_b_lat=cb, ctx_b_mask=mb,
                T_inf=args.T_inf, ablation="AB", prefix_cond=True,
                prompt_lat_list=batch.get("prompt_lat_list"), prompt_lens=batch.get("prompt_lens"),
                lora_wrappers=wrappers,
            )
            toks = decode_block_to_token_ids(backbone, z)
            for r, tk in zip(rows, toks):
                scores[mode][r["bucket"]].append(per_token_em(tk, r["answer"], backbone.tokenizer))
        if b % 5 == 0 or b == n_batches - 1:
            print(f"[swap] batch {b+1}/{n_batches}", flush=True)

    print("\n[swap] per-token EM, normal vs swapped (B0 full, ablation=AB):")
    print("%-9s %6s %8s %8s %8s" % ("bucket", "n", "normal", "swapped", "delta"))
    summary = {}
    for bk in ["AB", "A-only", "B-only", "neither"]:
        nv = scores["normal"].get(bk, []); sv = scores["swapped"].get(bk, [])
        if not nv:
            continue
        nm, sm = float(np.mean(nv)), float(np.mean(sv))
        print("%-9s %6d %8.3f %8.3f %+8.3f" % (bk, len(nv), nm, sm, sm - nm))
        summary[bk] = {"n": len(nv), "normal": nm, "swapped": sm, "delta": sm - nm}
    if args.out:
        json.dump({"summary": summary, "args": vars(args)}, open(args.out, "w"), indent=2)
        print(f"[swap] saved {args.out}")


if __name__ == "__main__":
    main()
