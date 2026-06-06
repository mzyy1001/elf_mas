"""Probe trained gate activations of B0 (full MAS) on the AB bucket.

For each patched layer {8,12,16,20} and each agent {A,B}, record the sigmoid gate
value g = sigmoid(gate([q_pool || kv_pool])) over all AB-bucket items x all T_inf
denoising steps, with ablation='AB' (both agents active). Same 200-item selection
(seed 0) as the lesion matrix; AB bucket = 82 items.
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
import numpy as np
import torch
from datasets import load_from_disk

from elf_mas_pt.model.frozen_cola import FrozenColaBackbone, FrozenColaSpec
from elf_mas_pt.model.mas_in_block import patch_cola_dit_with_mas
from elf_mas_pt.eval.eval_cola_mas import sample_mas_block
from elf_mas_pt.training.train_cola import build_batch_inputs


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
    assert ckpt.get("mode", "dual") == "dual", "gate probe expects a dual-mode (full) checkpoint"
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

    # Hook each gate Linear; its output is the pre-sigmoid logit (B,1).
    gate_acc = defaultdict(list)
    init_bias = {}
    def make_hook(layer, agent):
        def hook(module, inp, out):
            gate_acc[(layer, agent)].extend(torch.sigmoid(out.detach().float()).flatten().cpu().tolist())
        return hook
    for w, layer in zip(wrappers, layers):
        w.mas_a.gate.register_forward_hook(make_hook(layer, "A"))
        w.mas_b.gate.register_forward_hook(make_hook(layer, "B"))
        init_bias[(layer, "A")] = float(w.mas_a.gate.bias.detach().float().item())
        init_bias[(layer, "B")] = float(w.mas_b.gate.bias.detach().float().item())

    ds = load_from_disk(f"{args.dataset_dir}/{args.split}")
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(ds), size=min(args.n_items, len(ds)), replace=False).tolist()
    items = [r for r in list(ds.select(idx)) if r["bucket"] == args.bucket]
    print(f"[gate] {len(items)} '{args.bucket}'-bucket items; layers={layers}; T_inf={args.T_inf}", flush=True)

    sample_rng = np.random.default_rng(args.seed + 1)
    n_batches = (len(items) + args.batch_size - 1) // args.batch_size
    for b in range(n_batches):
        rows = items[b * args.batch_size:(b + 1) * args.batch_size]
        batch = build_batch_inputs(backbone, rows, variant="full", S_ctx=args.S_ctx, S_q=args.S_q, rng=sample_rng)
        _ = sample_mas_block(
            backbone, None,
            q_lat=batch["q_lat"], q_mask=batch["q_mask"],
            ctx_a_lat=batch["ctx_a_lat"], ctx_a_mask=batch["ctx_a_mask"],
            ctx_b_lat=batch["ctx_b_lat"], ctx_b_mask=batch["ctx_b_mask"],
            T_inf=args.T_inf, ablation="AB", prefix_cond=True,
            prompt_lat_list=batch.get("prompt_lat_list"), prompt_lens=batch.get("prompt_lens"),
            lora_wrappers=wrappers,
        )

    print(f"\n[gate] mean sigmoid gate on '{args.bucket}' bucket (init was sigmoid(-2)=0.119):")
    print("%-6s %8s %8s %8s %8s" % ("layer", "gate_A", "std_A", "gate_B", "std_B"))
    summary = {}
    for layer in layers:
        a = np.array(gate_acc[(layer, "A")]); bb = np.array(gate_acc[(layer, "B")])
        print("%-6d %8.3f %8.3f %8.3f %8.3f" % (layer, a.mean(), a.std(), bb.mean(), bb.std()))
        summary[str(layer)] = {"gate_A_mean": float(a.mean()), "gate_A_std": float(a.std()),
                               "gate_B_mean": float(bb.mean()), "gate_B_std": float(bb.std()),
                               "n_calls": int(a.size)}
    allA = np.concatenate([np.array(gate_acc[(l, "A")]) for l in layers])
    allB = np.concatenate([np.array(gate_acc[(l, "B")]) for l in layers])
    print("%-6s %8.3f %8s %8.3f" % ("ALL", allA.mean(), "", allB.mean()))
    print(f"\n[gate] trained gate bias per (layer,agent): {init_bias}")
    if args.out:
        json.dump({"summary": summary, "overall": {"A": float(allA.mean()), "B": float(allB.mean())},
                   "trained_bias": {f"{k[0]}_{k[1]}": v for k, v in init_bias.items()},
                   "args": vars(args)}, open(args.out, "w"), indent=2)
        print(f"[gate] saved {args.out}")


if __name__ == "__main__":
    main()
