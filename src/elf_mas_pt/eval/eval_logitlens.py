"""Layer-wise logit-lens probe for the trained B0 (full MAS) Cola checkpoint.

For every DiT layer 0-23 we capture the answer-block hidden state during normal
B0 sampling (fixed seed, fixed T_inf) and apply the model's REAL output head
(txt_out_ada adaLN(timestep) + txt_out Linear 2048->16) to get a per-layer
predicted velocity, then x-pred z_clean = z_t - t*v and VAE-decode it.

APPROXIMATION (important): the output head is the genuine final head, but
intermediate layers were NOT trained to be read by it, so decoded intermediate
text is a DIAGNOSTIC PROBE, not the model's real output. Built-in check: the
layer-23 logit-lens velocity must equal the model's actual velocity (sanity).

Saves numeric grids (layer x step x example) + qualitative decoded text to JSON;
plotting is done separately (matplotlib not on chen).
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
from elf_mas_pt.eval.eval_cola_mas import decode_block_to_token_ids, per_token_em, per_token_f1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default="/home/chenhongrui/elf_mas/runs/cola_mas_sweep/full_seed0")
    p.add_argument("--dataset_dir", default="/home/chenhongrui/elf_mas/data/synth_2fact_v1")
    p.add_argument("--split", default="eval")
    p.add_argument("--cola_base", default="/home/chenhongrui/elf_mas/hf_cache/elf-mas-bridge/phase3_assets/cola_dlm")
    p.add_argument("--cola_src", default="/home/chenhongrui/elf_mas/cola_dlm_src")
    p.add_argument("--n_ab", type=int, default=10)
    p.add_argument("--n_other", type=int, default=2)   # per other bucket
    p.add_argument("--T_inf", type=int, default=16)
    p.add_argument("--S_q", type=int, default=16)
    p.add_argument("--S_ctx", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="/home/chenhongrui/elf_mas/runs/cola_mas_sweep/logitlens.json")
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
    dit = backbone.dit

    ckpt = torch.load(f"{args.ckpt_dir}/lora_state.pt", map_location=device)
    layers = ckpt["lora_layer_indices"]; mas_layers = set(layers)
    txt_dim = dit.config.txt_dim if hasattr(dit.config, "txt_dim") else 2048
    wrappers = patch_cola_dit_with_mas(
        dit, txt_dim=txt_dim, ctx_lat_dim=backbone.latent_dim, layer_indices=layers,
        inner_dim=ckpt["lora_inner"], num_heads=ckpt["lora_heads"], block_size=backbone.block_size, mode="dual")
    for w, wc in zip(wrappers, ckpt["wrappers"]):
        w.mas_a.load_state_dict(wc["mas_a"]); w.mas_b.load_state_dict(wc["mas_b"]); w.eval()
    for pa in dit.parameters():
        pa.requires_grad_(False)
    n_blocks = len(dit.blocks)

    # ----- hooks: capture emb_in output + each block output during sampling steps -----
    state = {"capture": False, "emb": None, "hid": [None] * n_blocks}
    def emb_hook(m, inp, out):
        if state["capture"]:
            state["emb"] = out.detach()
    dit.emb_in.register_forward_hook(emb_hook)
    def mk_block_hook(i):
        def hook(m, inp, out):
            if state["capture"]:
                state["hid"][i] = out.detach()
        return hook
    for i, blk in enumerate(dit.blocks):
        blk.register_forward_hook(mk_block_hook(i))

    def apply_head(h_flat, emb_flat):
        # h_flat (N,2048), emb_flat (N,emb_dim) -> velocity (N,16) using the REAL head
        h2 = dit.txt_out_ada(h_flat, emb=emb_flat, layer="out", mode="in",
                             hid_shape=None, norm_layer=dit.txt_out_norm)
        return dit.txt_out.proj(h2)

    # ----- pick examples -----
    ds = load_from_disk(f"{args.dataset_dir}/{args.split}")
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(ds), size=min(200, len(ds)), replace=False).tolist()
    pool = list(ds.select(idx))
    picked = []
    for bk, k in [("AB", args.n_ab), ("A-only", args.n_other), ("B-only", args.n_other), ("neither", args.n_other)]:
        picked += [r for r in pool if r["bucket"] == bk][:k]
    rows = picked
    N = len(rows)
    print(f"[ll] {N} examples: {[(r['bucket']) for r in rows]}", flush=True)

    sample_rng = np.random.default_rng(args.seed + 1)
    batch = build_batch_inputs(backbone, rows, variant="full", S_ctx=args.S_ctx, S_q=args.S_q, rng=sample_rng)
    set_ctx = lambda: [w.set_mas_context(batch["ctx_a_lat"], batch["ctx_b_lat"],
                                         batch["ctx_a_mask"], batch["ctx_b_mask"], "AB") for w in wrappers]

    bs = backbone.block_size; D = backbone.latent_dim; T = 1000.0
    gold_first = [backbone.tokenizer.encode(r["answer"]).ids[:1] for r in rows]

    torch.manual_seed(args.seed)
    z = torch.randn(N, bs, D, device=device, dtype=torch.float32)
    clear_context_all(wrappers)
    prime_prefix_kv_cache(backbone, batch["prompt_lat_list"], batch["prompt_lens"])
    set_ctx()

    em = np.zeros((n_blocks, args.T_inf, N)); ftm = np.zeros_like(em); f1 = np.zeros_like(em)
    qual = {}             # (example, layer, step) -> text   (subset)
    qual_examples = list(range(min(6, N)))
    qual_layers = [0, 4, 8, 12, 16, 20, 23]
    qual_steps = [0, args.T_inf // 2, args.T_inf - 1]
    sanity = 0.0

    dt = 1.0 / args.T_inf
    with torch.no_grad():
        for step in range(args.T_inf):
            t_frac = 1.0 - step * dt
            t_internal = torch.full((N,), t_frac * T, device=device, dtype=torch.float32)
            z_in = z
            state["capture"] = True
            v0 = compute_v0_block_with_cache(backbone, z_in, t_internal, batch["prompt_lens"]).float()
            state["capture"] = False
            emb_flat = state["emb"]
            # logit-lens per layer
            for L in range(n_blocks):
                h = state["hid"][L]                              # (N*bs, 2048)
                vL = apply_head(h, emb_flat).float().reshape(N, bs, D)
                if L == n_blocks - 1:
                    sanity = max(sanity, float((vL - v0).abs().max()))
                x_pred = z_in - t_frac * vL
                toks = decode_block_to_token_ids(backbone, x_pred)
                for j, tk in enumerate(toks):
                    em[L, step, j] = per_token_em(tk, rows[j]["answer"], backbone.tokenizer)
                    f1[L, step, j] = per_token_f1(tk, rows[j]["answer"], backbone.tokenizer)
                    ftm[L, step, j] = 1.0 if (len(tk) > 0 and gold_first[j] and tk[0] == gold_first[j][0]) else 0.0
                    if j in qual_examples and L in qual_layers and step in qual_steps:
                        from elf_mas_pt.eval.eval_cola_mas import decode_block_to_texts
                        # decode text once per (L,step) batch is cheaper, but keep simple:
                        qual[f"{j}|{L}|{step}"] = decode_block_to_texts(backbone, x_pred[j:j+1])[0][:40]
            z = z - dt * v0
            print(f"[ll] step {step+1}/{args.T_inf} done", flush=True)
    clear_context_all(wrappers); _disable_dit_kv_cache(backbone)

    out = {
        "layers_all": list(range(n_blocks)),
        "mas_layers": sorted(mas_layers),
        "T_inf": args.T_inf,
        "examples": [{"bucket": r["bucket"], "question": r["question"], "answer": r["answer"]} for r in rows],
        "qual_examples": qual_examples, "qual_layers": qual_layers, "qual_steps": qual_steps,
        "em_grid": em.mean(axis=2).tolist(),         # (layer, step) mean over examples
        "ftm_grid": ftm.mean(axis=2).tolist(),
        "f1_grid": f1.mean(axis=2).tolist(),
        "em_layer_step_example": em.tolist(),         # full (layer, step, example)
        "qual_text": qual,
        "sanity_layer23_vs_model_max_abs_v_diff": sanity,
        "note": "logit-lens via REAL output head applied to intermediate hidden; APPROXIMATE for L<23.",
    }
    json.dump(out, open(args.out, "w"))
    print(f"\n[ll] sanity (layer23 head == model velocity) max|Δv| = {sanity:.3e}  (should be ~0)")
    print(f"[ll] saved {args.out}")
    # quick console peek: per-layer best-over-steps mean EM
    best = em.mean(axis=2).max(axis=1)
    print("[ll] per-layer best-over-steps mean per-token EM:")
    for L in range(n_blocks):
        tag = " <-MAS" if L in mas_layers else ""
        print("  L%2d: %.3f%s" % (L, best[L], tag))


if __name__ == "__main__":
    main()
