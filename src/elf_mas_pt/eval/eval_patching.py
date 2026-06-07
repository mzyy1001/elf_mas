"""Activation/residual patching (analysis 5) for a trained B0 checkpoint.

Per-step counterfactual patch: at each denoising step, the ablated run's shared
hidden state AFTER DiT block L is overwritten with the value it WOULD have had with
both agents active (a full-A forward on the SAME z_t). Downstream layers then
proceed with the patched hidden. If this restores the answer, layer L's
representation (with A's accumulated writes through depth L) is causally sufficient
to rescue the ablated computation.

Both forwards each step share the ablated run's current z_t, so hidden states are
aligned. Configs: rmA+patch@{8,12,16}, rmB+patch@{12,20}, plus full / rmA / rmB
baselines. AB bucket only. Default = fixed-KG B0 (SHORTCUT-CONTAMINATED).
"""
from __future__ import annotations
import argparse, json
import numpy as np
import torch
from datasets import load_from_disk

from elf_mas_pt.model.frozen_cola import FrozenColaBackbone, FrozenColaSpec
from elf_mas_pt.model.mas_in_block import patch_cola_dit_with_mas, clear_context_all
from elf_mas_pt.training.train_cola import (
    build_batch_inputs, prime_prefix_kv_cache, compute_v0_block_with_cache, _disable_dit_kv_cache,
)
from elf_mas_pt.eval.eval_cola_mas import decode_block_to_token_ids, per_token_em
T = 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="/home/chenhongrui/elf_mas/runs/cola_mas_sweep/full_seed0")
    ap.add_argument("--dataset_dir", default="/home/chenhongrui/elf_mas/data/synth_2fact_v1")
    ap.add_argument("--cola_base", default="/home/chenhongrui/elf_mas/hf_cache/elf-mas-bridge/phase3_assets/cola_dlm")
    ap.add_argument("--cola_src", default="/home/chenhongrui/elf_mas/cola_dlm_src")
    ap.add_argument("--n_items", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--T_inf", type=int, default=16)
    ap.add_argument("--S_q", type=int, default=16)
    ap.add_argument("--S_ctx", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/home/chenhongrui/elf_mas/runs/cola_mas_sweep/patching.json")
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

    # one hook per block: capture or replace its output depending on shared state
    hk = {"mode": None, "layer": None, "buf": None}   # mode in {None,'capture','replace'}
    def mk(i):
        def h(m, inp, out):
            if hk["layer"] == i and hk["mode"] == "capture":
                hk["buf"] = out.detach().clone()
            elif hk["layer"] == i and hk["mode"] == "replace" and hk["buf"] is not None:
                return hk["buf"]
        return h
    for i, blk in enumerate(dit.blocks):
        blk.register_forward_hook(mk(i))

    bs = backbone.block_size; D = backbone.latent_dim
    ABL = {"full": "AB", "rmA": "B_only", "rmB": "A_only"}

    @torch.no_grad()
    def set_ctx(batch, ablation):
        for w in wrappers:
            w.set_mas_context(batch["ctx_a_lat"], batch["ctx_b_lat"], batch["ctx_a_mask"], batch["ctx_b_mask"], ablation)

    @torch.no_grad()
    def sample(batch, ablation, patch_layer=None):
        """Sample answer block. If patch_layer set: each step, run a full-A forward to
        capture block-`patch_layer` output, then run the ablated forward replacing it."""
        N = batch["q_lat"].shape[0]
        torch.manual_seed(4321)
        z = torch.randn(N, bs, D, device=device, dtype=torch.float32)
        clear_context_all(wrappers)
        prime_prefix_kv_cache(backbone, batch["prompt_lat_list"], batch["prompt_lens"])
        dt = 1.0 / args.T_inf
        for i in range(args.T_inf):
            t = torch.full((N,), (1.0 - i * dt) * T, device=device, dtype=torch.float32)
            if patch_layer is not None:
                # (a) capture full-A hidden after patch_layer on the SAME z
                set_ctx(batch, "AB"); hk["mode"] = "capture"; hk["layer"] = patch_layer; hk["buf"] = None
                _ = compute_v0_block_with_cache(backbone, z, t, batch["prompt_lens"])
                # (b) ablated forward, replacing block-patch_layer output with captured full-A
                set_ctx(batch, ablation); hk["mode"] = "replace"
                v0 = compute_v0_block_with_cache(backbone, z, t, batch["prompt_lens"]).float()
                hk["mode"] = None; hk["layer"] = None; hk["buf"] = None
            else:
                set_ctx(batch, ablation)
                v0 = compute_v0_block_with_cache(backbone, z, t, batch["prompt_lens"]).float()
            z = z - dt * v0
        clear_context_all(wrappers); _disable_dit_kv_cache(backbone)
        return z

    ds = load_from_disk(f"{args.dataset_dir}/eval")
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(ds), size=min(args.n_items, len(ds)), replace=False).tolist()
    items = [r for r in list(ds.select(idx)) if r["bucket"] == "AB"]
    print(f"[patch] {len(items)} AB items; layers={layers}", flush=True)

    configs = [("full", "full", None), ("rmA", "rmA", None), ("rmB", "rmB", None),
               ("rmA+patch@8", "rmA", 8), ("rmA+patch@12", "rmA", 12), ("rmA+patch@16", "rmA", 16),
               ("rmB+patch@12", "rmB", 12), ("rmB+patch@20", "rmB", 20)]
    sample_rng = np.random.default_rng(args.seed + 1)
    nb = (len(items) + args.batch_size - 1) // args.batch_size
    batches = [(items[b*args.batch_size:(b+1)*args.batch_size],
                build_batch_inputs(backbone, items[b*args.batch_size:(b+1)*args.batch_size],
                                   variant="full", S_ctx=args.S_ctx, S_q=args.S_q, rng=sample_rng))
               for b in range(nb)]

    em = {}
    for name, ablation, pl in configs:
        s = n = 0
        for rows, batch in batches:
            z = sample(batch, ABL[ablation], patch_layer=pl)
            for r, tk in zip(rows, decode_block_to_token_ids(backbone, z)):
                s += per_token_em(tk, r["answer"], backbone.tokenizer); n += 1
        em[name] = s / n
        print(f"[patch] {name:16s} AB EM={em[name]:.3f}", flush=True)

    full, rmA, rmB = em["full"], em["rmA"], em["rmB"]
    def restore(name, base):
        denom = full - base
        return (em[name] - base) / denom if abs(denom) > 1e-6 else float("nan")
    restoration = {
        "rmA+patch@8": restore("rmA+patch@8", rmA), "rmA+patch@12": restore("rmA+patch@12", rmA),
        "rmA+patch@16": restore("rmA+patch@16", rmA), "rmB+patch@12": restore("rmB+patch@12", rmB),
        "rmB+patch@20": restore("rmB+patch@20", rmB),
    }
    out = {"layers": layers, "n_items": len(items), "em": em, "restoration_frac": restoration,
           "note": "restoration_frac = (EM_patched - EM_ablated)/(EM_full - EM_ablated)."}
    json.dump(out, open(args.out, "w"), indent=2)
    print("\n[patch] restoration fraction (1.0=fully rescued, 0=no rescue):")
    for k, v in restoration.items():
        print(f"  {k:16s}: {v:+.2f}")
    print(f"[patch] saved {args.out}")


if __name__ == "__main__":
    main()
