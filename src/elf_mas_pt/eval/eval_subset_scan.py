"""All-subset MAS-injection scan (analyses 1-3) for a trained B0 checkpoint.

For every subset of the MAS injection layers {8,12,16,20}, enable ONLY those
layers' MAS heads and run the 4 lesion ablations (AB/A_only/B_only/Neither) over
the eval set, scoring per-bucket per-token EM and F1.

Derives:
  (1) per-subset AB full/rmA/rmB/rm_both + Da/Db (+ per-bucket EM/F1)
  (2) prefix curve  {}->{8}->{8,12}->{8,12,16}->{8,12,16,20}
      suffix curve  {}->{20}->{16,20}->{12,16,20}->{8,12,16,20}
  (3) pairwise synergy(i,j)=EM({i,j})-EM({i})-EM({j})+EM({})

Default checkpoint/dataset = fixed-KG B0 (strong but SHORTCUT-CONTAMINATED).
"""
from __future__ import annotations
import argparse, json, itertools
from collections import defaultdict
import numpy as np
import torch
from datasets import load_from_disk

from elf_mas_pt.model.frozen_cola import FrozenColaBackbone, FrozenColaSpec
from elf_mas_pt.model.mas_in_block import patch_cola_dit_with_mas, clear_context_all
from elf_mas_pt.training.train_cola import (
    build_batch_inputs, prime_prefix_kv_cache, compute_v0_block_with_cache, _disable_dit_kv_cache,
)
from elf_mas_pt.eval.eval_cola_mas import decode_block_to_token_ids, per_token_em, per_token_f1

T = 1000.0


@torch.no_grad()
def sample_subset(backbone, wrappers, layers, enabled, ablation, batch, T_inf):
    B = batch["q_lat"].shape[0]; bs = backbone.block_size; D = backbone.latent_dim
    device = backbone.device
    z = torch.randn(B, bs, D, device=device, dtype=torch.float32)
    clear_context_all(wrappers)
    prime_prefix_kv_cache(backbone, batch["prompt_lat_list"], batch["prompt_lens"])
    for w, layer in zip(wrappers, layers):
        if layer in enabled:
            w.set_mas_context(batch["ctx_a_lat"], batch["ctx_b_lat"],
                              batch["ctx_a_mask"], batch["ctx_b_mask"], ablation)
    dt = 1.0 / T_inf
    for i in range(T_inf):
        t = torch.full((B,), (1.0 - i * dt) * T, device=device, dtype=torch.float32)
        v0 = compute_v0_block_with_cache(backbone, z, t, batch["prompt_lens"]).float()
        z = z - dt * v0
    clear_context_all(wrappers); _disable_dit_kv_cache(backbone)
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="/home/chenhongrui/elf_mas/runs/cola_mas_sweep/full_seed0")
    ap.add_argument("--dataset_dir", default="/home/chenhongrui/elf_mas/data/synth_2fact_v1")
    ap.add_argument("--tag", default="fixedkg")
    ap.add_argument("--cola_base", default="/home/chenhongrui/elf_mas/hf_cache/elf-mas-bridge/phase3_assets/cola_dlm")
    ap.add_argument("--cola_src", default="/home/chenhongrui/elf_mas/cola_dlm_src")
    ap.add_argument("--n_items", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--T_inf", type=int, default=16)
    ap.add_argument("--S_q", type=int, default=16)
    ap.add_argument("--S_ctx", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/home/chenhongrui/elf_mas/runs/cola_mas_sweep/subset_scan.json")
    args = ap.parse_args()

    device = torch.device("cuda"); torch.manual_seed(args.seed)
    spec = FrozenColaSpec(dit_path=f"{args.cola_base}/cola_dlm/cola_dit",
                          vae_path=f"{args.cola_base}/cola_dlm/cola_vae",
                          tokenizer_path=f"{args.cola_base}/tokenizer.json", cola_src_path=args.cola_src)
    backbone = FrozenColaBackbone.load(spec, device); dit = backbone.dit
    ckpt = torch.load(f"{args.ckpt_dir}/lora_state.pt", map_location=device)
    layers = ckpt["lora_layer_indices"]
    txt_dim = dit.config.txt_dim if hasattr(dit.config, "txt_dim") else 2048
    wrappers = patch_cola_dit_with_mas(dit, txt_dim=txt_dim, ctx_lat_dim=backbone.latent_dim,
        layer_indices=layers, inner_dim=ckpt["lora_inner"], num_heads=ckpt["lora_heads"],
        block_size=backbone.block_size, mode="dual")
    for w, wc in zip(wrappers, ckpt["wrappers"]):
        w.mas_a.load_state_dict(wc["mas_a"]); w.mas_b.load_state_dict(wc["mas_b"]); w.eval()
    for p in dit.parameters():
        p.requires_grad_(False)

    ds = load_from_disk(f"{args.dataset_dir}/eval")
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(ds), size=min(args.n_items, len(ds)), replace=False).tolist()
    items = list(ds.select(idx))
    print(f"[scan] {len(items)} items; layers={layers}", flush=True)

    # all 16 subsets
    subsets = []
    for r in range(len(layers) + 1):
        for combo in itertools.combinations(layers, r):
            subsets.append(list(combo))
    ablations = ["AB", "A_only", "B_only", "Neither"]
    buckets = ["AB", "A-only", "B-only", "neither"]

    sample_rng = np.random.default_rng(args.seed + 1)
    nb = (len(items) + args.batch_size - 1) // args.batch_size
    batches = [(items[b*args.batch_size:(b+1)*args.batch_size],
                build_batch_inputs(backbone, items[b*args.batch_size:(b+1)*args.batch_size],
                                   variant="full", S_ctx=args.S_ctx, S_q=args.S_q, rng=sample_rng))
               for b in range(nb)]

    results = {}  # subset_key -> ablation -> bucket -> {em_tok,f1_tok,n}
    for si, enabled in enumerate(subsets):
        key = "+".join(str(x) for x in enabled) if enabled else "none"
        results[key] = {}
        for abl in ablations:
            acc = defaultdict(lambda: [0.0, 0.0, 0])  # bucket -> [em_sum,f1_sum,n]
            for rows, batch in batches:
                z = sample_subset(backbone, wrappers, layers, set(enabled), abl, batch, args.T_inf)
                toks = decode_block_to_token_ids(backbone, z)
                for r, tk in zip(rows, toks):
                    e = per_token_em(tk, r["answer"], backbone.tokenizer)
                    f = per_token_f1(tk, r["answer"], backbone.tokenizer)
                    a = acc[r["bucket"]]; a[0] += e; a[1] += f; a[2] += 1
            results[key][abl] = {bk: {"em_tok": acc[bk][0]/acc[bk][2], "f1_tok": acc[bk][1]/acc[bk][2],
                                      "n": acc[bk][2]} for bk in acc}
        ab = results[key]
        full = ab["AB"]["AB"]["em_tok"]; rmA = ab["B_only"]["AB"]["em_tok"]
        rmB = ab["A_only"]["AB"]["em_tok"]; both = ab["Neither"]["AB"]["em_tok"]
        print(f"[scan] {si+1}/16 subset={key:12s} AB full={full:.3f} rmA={rmA:.3f} rmB={rmB:.3f} "
              f"rm_both={both:.3f} Da={full-rmA:+.3f} Db={full-rmB:+.3f}", flush=True)

    # helper: AB-bucket full (ablation=AB) EM for a subset
    def emAB(enabled):
        key = "+".join(str(x) for x in enabled) if enabled else "none"
        return results[key]["AB"]["AB"]["em_tok"]

    prefix = [[], [8], [8,12], [8,12,16], [8,12,16,20]]
    suffix = [[], [20], [16,20], [12,16,20], [8,12,16,20]]
    prefix_curve = [{"subset": s, "em": emAB(s)} for s in prefix]
    suffix_curve = [{"subset": s, "em": emAB(s)} for s in suffix]

    em0 = emAB([])
    synergy = {}
    for i, j in itertools.combinations(layers, 2):
        synergy[f"{i},{j}"] = emAB([i, j]) - emAB([i]) - emAB([j]) + em0

    out = {"tag": args.tag, "ckpt_dir": args.ckpt_dir, "dataset_dir": args.dataset_dir,
           "layers": layers, "n_items": len(items), "results": results,
           "prefix_curve": prefix_curve, "suffix_curve": suffix_curve,
           "synergy": synergy, "em_empty": em0,
           "note": "Subset = which MAS layers are ENABLED. Lesion ablation applied within enabled layers."}
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n[scan] prefix curve EM: {[round(c['em'],3) for c in prefix_curve]}")
    print(f"[scan] suffix curve EM: {[round(c['em'],3) for c in suffix_curve]}")
    print(f"[scan] synergy: " + ", ".join(f"{k}:{v:+.3f}" for k, v in synergy.items()))
    print(f"[scan] saved {args.out}")


if __name__ == "__main__":
    main()
