"""Layer-to-layer mediation probe (analysis 4) for a trained B0 checkpoint.

Captures, at MAS layers {8,12,16,20}, agent A/B internals during ONE forward at a
FIXED denoising state (step 0, t=1, fixed noise z) so all conditions share z_t and
differences are attributable to the context manipulation:
  q_A, q_B  = output of mas_a.q_in / mas_b.q_in  (the cross-attn QUERY, a fn of the
              SHARED hidden state h at that layer)
  gate_A,gate_B = sigmoid gate values
  ||resid_A||, ||resid_B|| = gated-residual norms

Conditions: full / rmA (B_only) / rmB (A_only) / ctxA_shuffled / ctxB_shuffled / swap.

Key question: does removing/shuffling A change agent B's QUERY at later layers
(12/16/20)? q_B reads the shared hidden state, so a change means A's writes
propagate to B through the residual stream (A->B mediation). q_B at layer 8 should
be unchanged by A (A injects at 8 only AFTER B reads the frozen block-8 output).
Reports per-layer cosine similarity and linear CKA of q_B(full) vs q_B(condition).
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
T = 1000.0


def linear_cka(X, Y):
    X = X - X.mean(0, keepdim=True); Y = Y - Y.mean(0, keepdim=True)
    xy = (X.T @ Y).pow(2).sum()
    xx = (X.T @ X).pow(2).sum().sqrt(); yy = (Y.T @ Y).pow(2).sum().sqrt()
    return float(xy / (xx * yy + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="/home/chenhongrui/elf_mas/runs/cola_mas_sweep/full_seed0")
    ap.add_argument("--dataset_dir", default="/home/chenhongrui/elf_mas/data/synth_2fact_v1")
    ap.add_argument("--cola_base", default="/home/chenhongrui/elf_mas/hf_cache/elf-mas-bridge/phase3_assets/cola_dlm")
    ap.add_argument("--cola_src", default="/home/chenhongrui/elf_mas/cola_dlm_src")
    ap.add_argument("--n_items", type=int, default=160)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--S_q", type=int, default=16)
    ap.add_argument("--S_ctx", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/home/chenhongrui/elf_mas/runs/cola_mas_sweep/mediation.json")
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
    wl = {layer: w for w, layer in zip(wrappers, layers)}

    # hooks: q_in output, gate output, residual (module output) for mas_a/mas_b at each MAS layer
    cap = {"on": False, "data": {}}
    def mk_q(layer, ag):
        def h(m, i, o):
            if cap["on"]: cap["data"][("q", layer, ag)] = o.detach()
        return h
    def mk_gate(layer, ag):
        def h(m, i, o):
            if cap["on"]: cap["data"][("gate", layer, ag)] = torch.sigmoid(o.detach())
        return h
    def mk_res(layer, ag):
        def h(m, i, o):
            if cap["on"]: cap["data"][("res", layer, ag)] = o.detach()
        return h
    for layer, w in wl.items():
        w.mas_a.q_in.register_forward_hook(mk_q(layer, "A")); w.mas_b.q_in.register_forward_hook(mk_q(layer, "B"))
        w.mas_a.gate.register_forward_hook(mk_gate(layer, "A")); w.mas_b.gate.register_forward_hook(mk_gate(layer, "B"))
        w.mas_a.register_forward_hook(mk_res(layer, "A")); w.mas_b.register_forward_hook(mk_res(layer, "B"))

    ds = load_from_disk(f"{args.dataset_dir}/eval")
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(ds), size=min(args.n_items, len(ds)), replace=False).tolist()
    items = [r for r in list(ds.select(idx)) if r["bucket"] == "AB"]   # AB only for hop mediation
    print(f"[med] {len(items)} AB items; layers={layers}", flush=True)

    conditions = ["full", "rmA", "rmB", "ctxA_shuf", "ctxB_shuf", "swap"]
    abl = {"full": "AB", "rmA": "B_only", "rmB": "A_only", "ctxA_shuf": "AB", "ctxB_shuf": "AB", "swap": "AB"}
    # accumulate per (cond, kind, layer, agent)
    store = {c: {} for c in conditions}
    sample_rng = np.random.default_rng(args.seed + 1)
    nb = (len(items) + args.batch_size - 1) // args.batch_size
    bs = backbone.block_size; D = backbone.latent_dim

    @torch.no_grad()
    def run(rows, batch, cond):
        N = len(rows)
        torch.manual_seed(1234)              # SAME noise across conditions
        z = torch.randn(N, bs, D, device=device, dtype=torch.float32)
        ca, ma, cb, mb = batch["ctx_a_lat"], batch["ctx_a_mask"], batch["ctx_b_lat"], batch["ctx_b_mask"]
        if cond == "ctxA_shuf":
            perm = torch.randperm(N); ca, ma = ca[perm], ma[perm]
        elif cond == "ctxB_shuf":
            perm = torch.randperm(N); cb, mb = cb[perm], mb[perm]
        elif cond == "swap":
            ca, ma, cb, mb = cb, mb, ca, ma
        clear_context_all(wrappers)
        prime_prefix_kv_cache(backbone, batch["prompt_lat_list"], batch["prompt_lens"])
        for w in wrappers:
            w.set_mas_context(ca, cb, ma, mb, abl[cond])
        cap["on"] = True; cap["data"] = {}
        t = torch.full((N,), 1.0 * T, device=device, dtype=torch.float32)
        _ = compute_v0_block_with_cache(backbone, z, t, batch["prompt_lens"])
        cap["on"] = False
        clear_context_all(wrappers); _disable_dit_kv_cache(backbone)
        out = {}
        for (kind, layer, ag), v in cap["data"].items():
            vv = v.reshape(N, bs, -1) if v.dim() == 3 else v.reshape(N, -1)
            if kind == "q":
                out[(kind, layer, ag)] = vv.mean(1).float().cpu()              # (N, inner)
            elif kind == "gate":
                out[(kind, layer, ag)] = vv.float().mean(1).cpu()              # (N,)
            else:  # residual norm per sample
                out[(kind, layer, ag)] = vv.float().flatten(1).norm(dim=1).cpu()  # (N,)
        return out

    for b in range(nb):
        rows = items[b*args.batch_size:(b+1)*args.batch_size]
        batch = build_batch_inputs(backbone, rows, variant="full", S_ctx=args.S_ctx, S_q=args.S_q, rng=sample_rng)
        for c in conditions:
            o = run(rows, batch, c)
            for k, v in o.items():
                store[c].setdefault(k, []).append(v)
        print(f"[med] batch {b+1}/{nb}", flush=True)
    for c in conditions:
        for k in store[c]:
            store[c][k] = torch.cat(store[c][k], 0)

    # metrics: q_B(full) vs q_B(cond) per layer; gate & resid norm means
    report = {"layers": layers, "conditions": conditions, "qB_cosine_vs_full": {}, "qB_cka_vs_full": {},
              "gate_mean": {}, "resid_norm_mean": {}}
    for layer in layers:
        kqf = ("q", layer, "B")
        if kqf in store["full"]:
            Xf = store["full"][kqf]
            for c in conditions:
                if kqf in store[c]:
                    Xc = store[c][kqf]
                    cos = torch.nn.functional.cosine_similarity(Xf, Xc, dim=1).mean().item()
                    report["qB_cosine_vs_full"].setdefault(c, {})[str(layer)] = cos
                    report["qB_cka_vs_full"].setdefault(c, {})[str(layer)] = linear_cka(Xf, Xc)
        for ag in ["A", "B"]:
            for c in conditions:
                kg = ("gate", layer, ag); kr = ("res", layer, ag)
                if kg in store[c]:
                    report["gate_mean"].setdefault(c, {})[f"{layer}_{ag}"] = float(store[c][kg].mean())
                if kr in store[c]:
                    report["resid_norm_mean"].setdefault(c, {})[f"{layer}_{ag}"] = float(store[c][kr].mean())
    json.dump(report, open(args.out, "w"), indent=2)
    print("\n[med] q_B cosine-sim to full (1.0=identical; lower=A changed B's query):")
    for c in ["rmA", "ctxA_shuf", "ctxB_shuf", "swap"]:
        row = report["qB_cosine_vs_full"].get(c, {})
        print(f"  {c:10s}: " + " ".join(f"L{l}={row.get(str(l),float('nan')):.3f}" for l in layers))
    print(f"[med] saved {args.out}")


if __name__ == "__main__":
    main()
