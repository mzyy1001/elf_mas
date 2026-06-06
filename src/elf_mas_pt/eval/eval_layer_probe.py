"""Per-layer MAS contribution probe for B0 (full) on the AB bucket.

Enables the trained MAS on chosen layer subsets only (real decode path, no
logit-lens approximation) and measures per-token EM. Reveals the causal role of
each patched level {8,12,16,20}:
  - 'none' (sanity ~0), 'all' (sanity ~0.585)
  - single-layer knockout: all-but-one  -> that layer's marginal importance
  - cumulative bottom-up: [8], [8,12], [8,12,16], [8,12,16,20]
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict, Counter
import numpy as np
import torch
from datasets import load_from_disk

from elf_mas_pt.model.frozen_cola import FrozenColaBackbone, FrozenColaSpec
from elf_mas_pt.model.mas_in_block import patch_cola_dit_with_mas, clear_context_all
from elf_mas_pt.training.train_cola import (
    build_batch_inputs, prime_prefix_kv_cache, compute_v0_block_with_cache, _disable_dit_kv_cache,
)
from elf_mas_pt.eval.eval_cola_mas import decode_block_to_token_ids, per_token_em


@torch.no_grad()
def sample_subset(backbone, wrappers, layers, enabled, batch, T_inf, T=1000.0):
    B = batch["q_lat"].shape[0]; bs = backbone.block_size; D = backbone.latent_dim
    device = backbone.device
    z = torch.randn(B, bs, D, device=device, dtype=torch.float32)
    clear_context_all(wrappers)
    prime_prefix_kv_cache(backbone, batch["prompt_lat_list"], batch["prompt_lens"])
    for w, layer in zip(wrappers, layers):
        if layer in enabled:
            w.set_mas_context(batch["ctx_a_lat"], batch["ctx_b_lat"],
                              batch["ctx_a_mask"], batch["ctx_b_mask"], "AB")
    dt = 1.0 / T_inf
    for i in range(T_inf):
        t_frac = 1.0 - i * dt
        t_internal = torch.full((B,), t_frac * T, device=device, dtype=torch.float32)
        v0 = compute_v0_block_with_cache(backbone, z, t_internal, batch["prompt_lens"]).float()
        z = z - dt * v0
    clear_context_all(wrappers); _disable_dit_kv_cache(backbone)
    return z


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default="/home/chenhongrui/elf_mas/runs/cola_mas_sweep/full_seed0")
    p.add_argument("--dataset_dir", default="/home/chenhongrui/elf_mas/data/synth_2fact_v1")
    p.add_argument("--split", default="eval")
    p.add_argument("--cola_base", default="/home/chenhongrui/elf_mas/hf_cache/elf-mas-bridge/phase3_assets/cola_dlm")
    p.add_argument("--cola_src", default="/home/chenhongrui/elf_mas/cola_dlm_src")
    p.add_argument("--bucket", default="AB")
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
        block_size=backbone.block_size, mode="dual")
    for w, wc in zip(wrappers, ckpt["wrappers"]):
        w.mas_a.load_state_dict(wc["mas_a"]); w.mas_b.load_state_dict(wc["mas_b"]); w.eval()
    for pa in backbone.dit.parameters():
        pa.requires_grad_(False)

    ds = load_from_disk(f"{args.dataset_dir}/{args.split}")
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(ds), size=min(args.n_items, len(ds)), replace=False).tolist()
    items = [r for r in list(ds.select(idx)) if r["bucket"] == args.bucket]
    print(f"[layer] {len(items)} '{args.bucket}'-bucket items; layers={layers}", flush=True)

    L = list(layers)
    configs = {
        "none": [],
        "all": L,
        "knockout_8": [x for x in L if x != 8],
        "knockout_12": [x for x in L if x != 12],
        "knockout_16": [x for x in L if x != 16],
        "knockout_20": [x for x in L if x != 20],
        "cum_[8]": [8],
        "cum_[8,12]": [8, 12],
        "cum_[8,12,16]": [8, 12, 16],
        "cum_[12,16,20]": [12, 16, 20],
        "cum_[16,20]": [16, 20],
        "cum_[20]": [20],
    }
    sample_rng = np.random.default_rng(args.seed + 1)
    # Pre-build batches once (same inputs across configs)
    batches = []
    nb = (len(items) + args.batch_size - 1) // args.batch_size
    for b in range(nb):
        rows = items[b * args.batch_size:(b + 1) * args.batch_size]
        batches.append((rows, build_batch_inputs(backbone, rows, variant="full",
                                                 S_ctx=args.S_ctx, S_q=args.S_q, rng=sample_rng)))

    results = {}
    for name, enabled in configs.items():
        ems = []
        for rows, batch in batches:
            z = sample_subset(backbone, wrappers, layers, set(enabled), batch, args.T_inf)
            toks = decode_block_to_token_ids(backbone, z)
            for r, tk in zip(rows, toks):
                ems.append(per_token_em(tk, r["answer"], backbone.tokenizer))
        results[name] = float(np.mean(ems))
        print(f"[layer] {name:18s} enabled={enabled}  EM={results[name]:.3f}", flush=True)

    print("\n[layer] single-layer marginal importance (all=%.3f minus knockout):" % results["all"])
    for layer in L:
        print("  layer %2d : drop = %+.3f" % (layer, results["all"] - results[f"knockout_{layer}"]))
    if args.out:
        json.dump({"results": results, "layers": L, "args": vars(args)}, open(args.out, "w"), indent=2)
        print(f"[layer] saved {args.out}")


if __name__ == "__main__":
    main()
