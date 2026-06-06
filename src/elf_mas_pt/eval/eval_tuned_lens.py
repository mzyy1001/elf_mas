"""Minimal TUNED LENS (trained diagnostic probe) for B0 (full MAS) Cola checkpoint.

For each DiT layer 0-23 we train a tiny per-layer LINEAR readout
    Linear(2048 -> C)
from the mean-pooled answer-block hidden state to a discrete target, on the
TRAIN split, and evaluate accuracy on the held-out EVAL split. Two targets on
AB (two-hop) items:
  * hop-2 = final answer COUNTRY  (r["answer"])
  * hop-1 = bridge CITY           (parsed: "{person} lives in {city}" in ctx_A)

This is a TRAINED DIAGNOSTIC PROBE, NOT the model's native output. It measures
whether each kind of information is *linearly decodable* from a layer's
representation, correcting the basis mismatch that made the vanilla logit-lens
blind below the final layers.

Feature = mean over the 16 answer-block positions of the layer's hidden state,
captured at denoising step 0 (t=1, pure noise + context) so decodability
reflects context processing, not z_t leakage.
"""
from __future__ import annotations
import argparse, json, re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk

from elf_mas_pt.model.frozen_cola import FrozenColaBackbone, FrozenColaSpec
from elf_mas_pt.model.mas_in_block import patch_cola_dit_with_mas, clear_context_all
from elf_mas_pt.training.train_cola import (
    build_batch_inputs, prime_prefix_kv_cache, compute_v0_block_with_cache, _disable_dit_kv_cache,
)

T = 1000.0


def parse_ab(r):
    m = re.search(r"What country does (\w+) live in", r["question"])
    if not m:
        return None, None
    p = m.group(1)
    mc = re.search(re.escape(p) + r" lives in ([A-Za-z0-9]+)", r["ctx_a_text"])
    if not mc:
        return None, None
    return mc.group(1), r["answer"]          # (hop1 city, hop2 country)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="/home/chenhongrui/elf_mas/runs/cola_mas_sweep/full_seed0")
    ap.add_argument("--dataset_dir", default="/home/chenhongrui/elf_mas/data/synth_2fact_v1")
    ap.add_argument("--cola_base", default="/home/chenhongrui/elf_mas/hf_cache/elf-mas-bridge/phase3_assets/cola_dlm")
    ap.add_argument("--cola_src", default="/home/chenhongrui/elf_mas/cola_dlm_src")
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--n_eval", type=int, default=600)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--S_q", type=int, default=16)
    ap.add_argument("--S_ctx", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--context_mode", default="full", choices=["full", "none", "shuffled"],
                    help="full=normal MAS; none=contexts NOT injected (control for person-name "
                         "prefix leakage); shuffled=ctx_a<->ctx_b swapped")
    ap.add_argument("--out", default="/home/chenhongrui/elf_mas/runs/cola_mas_sweep/tuned_lens.json")
    args = ap.parse_args()

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
    nL = len(dit.blocks); bs = backbone.block_size; D = backbone.latent_dim

    # ---- hooks: capture each block output during feature-extraction forward ----
    state = {"capture": False, "hid": [None] * nL}
    def mk(i):
        def hook(m, inp, out):
            if state["capture"]:
                state["hid"][i] = out.detach()
        return hook
    for i, blk in enumerate(dit.blocks):
        blk.register_forward_hook(mk(i))

    # ---- collect AB items with parseable hop-1/hop-2 from train + eval ----
    def collect(split, n):
        ds = load_from_disk(f"{args.dataset_dir}/{split}")
        rows = []
        for r in ds:
            if r["bucket"] != "AB":
                continue
            city, country = parse_ab(r)
            if city is None:
                continue
            rr = dict(r); rr["_city"] = city; rr["_country"] = country
            rows.append(rr)
            if len(rows) >= n:
                break
        return rows
    train_rows = collect("train", args.n_train)
    eval_rows = collect("eval", args.n_eval)
    print(f"[tl] train AB={len(train_rows)}  eval AB={len(eval_rows)}", flush=True)
    print(f"[tl] parse sanity (first 3 train): "
          + "; ".join(f"{r['_city']}->{r['_country']}" for r in train_rows[:3]), flush=True)

    @torch.no_grad()
    def feats_for(rows):
        sample_rng = np.random.default_rng(args.seed + 7)
        per_layer = [[] for _ in range(nL)]
        nb = (len(rows) + args.batch_size - 1) // args.batch_size
        for b in range(nb):
            rb = rows[b * args.batch_size:(b + 1) * args.batch_size]
            batch = build_batch_inputs(backbone, rb, variant="full", S_ctx=args.S_ctx, S_q=args.S_q, rng=sample_rng)
            N = len(rb)
            z = torch.randn(N, bs, D, device=device, dtype=torch.float32)
            clear_context_all(wrappers)
            prime_prefix_kv_cache(backbone, batch["prompt_lat_list"], batch["prompt_lens"])
            if args.context_mode == "none":
                pass  # leave MAS off: contexts never injected (prefix-only)
            else:
                if args.context_mode == "shuffled":
                    ca, ma, cb, mb = batch["ctx_b_lat"], batch["ctx_b_mask"], batch["ctx_a_lat"], batch["ctx_a_mask"]
                else:
                    ca, ma, cb, mb = batch["ctx_a_lat"], batch["ctx_a_mask"], batch["ctx_b_lat"], batch["ctx_b_mask"]
                for w in wrappers:
                    w.set_mas_context(ca, cb, ma, mb, "AB")
            t_internal = torch.full((N,), 1.0 * T, device=device, dtype=torch.float32)
            state["capture"] = True
            _ = compute_v0_block_with_cache(backbone, z, t_internal, batch["prompt_lens"])
            state["capture"] = False
            for L in range(nL):
                h = state["hid"][L].reshape(N, bs, -1).float().mean(dim=1)  # (N,2048)
                per_layer[L].append(h.cpu())
            clear_context_all(wrappers); _disable_dit_kv_cache(backbone)
            if b % 20 == 0:
                print(f"[tl] feats batch {b+1}/{nb}", flush=True)
        return [torch.cat(x, 0) for x in per_layer]

    Xtr = feats_for(train_rows)
    Xev = feats_for(eval_rows)

    # ---- label vocabularies from TRAIN ----
    def vocab(rows, key):
        v = {}
        for r in rows:
            v.setdefault(r[key], len(v))
        return v
    city_v = vocab(train_rows, "_city"); country_v = vocab(train_rows, "_country")
    def labels(rows, key, v):
        return torch.tensor([v.get(r[key], -1) for r in rows], dtype=torch.long)
    ytr_city = labels(train_rows, "_city", city_v); yev_city = labels(eval_rows, "_city", city_v)
    ytr_co = labels(train_rows, "_country", country_v); yev_co = labels(eval_rows, "_country", country_v)
    print(f"[tl] |city classes|={len(city_v)} (chance {1/len(city_v):.3f})  "
          f"|country classes|={len(country_v)} (chance {1/len(country_v):.3f})", flush=True)

    def train_probe(Xtr_L, ytr, Xev_L, yev, C):
        keep_tr = ytr >= 0; keep_ev = yev >= 0
        xt = Xtr_L[keep_tr].to(device); yt = ytr[keep_tr].to(device)
        xe = Xev_L[keep_ev].to(device); ye = yev[keep_ev].to(device)
        mu = xt.mean(0, keepdim=True); sd = xt.std(0, keepdim=True) + 1e-6
        xt = (xt - mu) / sd; xe = (xe - mu) / sd
        probe = nn.Linear(xt.shape[1], C).to(device)
        opt = torch.optim.Adam(probe.parameters(), lr=1e-2, weight_decay=1e-3)
        for _ in range(args.epochs):
            opt.zero_grad(); F.cross_entropy(probe(xt), yt).backward(); opt.step()
        with torch.no_grad():
            acc = (probe(xe).argmax(1) == ye).float().mean().item()
        return acc, probe, (mu, sd)

    city_acc, country_acc = [], []
    probes = {}
    for L in range(nL):
        ac, pc, sc = train_probe(Xtr[L], ytr_city, Xev[L], yev_city, len(city_v))
        ao, po, so = train_probe(Xtr[L], ytr_co, Xev[L], yev_co, len(country_v))
        city_acc.append(ac); country_acc.append(ao)
        probes[L] = (pc, sc, po, so)
        tag = " <-MAS" if L in mas_layers else ""
        print(f"[tl] L{L:2d}  hop1-city acc={ac:.3f}  hop2-country acc={ao:.3f}{tag}", flush=True)

    # ---- qualitative: top-3 city & country predictions by depth for a few eval AB items ----
    inv_city = {v: k for k, v in city_v.items()}; inv_co = {v: k for k, v in country_v.items()}
    qual = []
    for j in range(min(4, len(eval_rows))):
        r = eval_rows[j]; entry = {"gold_city": r["_city"], "gold_country": r["_country"],
                                   "question": r["question"], "by_layer": {}}
        for L in [8, 12, 16, 20, 23]:
            pc, (muc, sdc), po, (muo, sdo) = probes[L]
            with torch.no_grad():
                xc = ((Xev[L][j:j+1].to(device) - muc) / sdc); top_c = pc(xc).topk(3, dim=1).indices[0].tolist()
                xo = ((Xev[L][j:j+1].to(device) - muo) / sdo); top_o = po(xo).topk(3, dim=1).indices[0].tolist()
            entry["by_layer"][str(L)] = {"city_top3": [inv_city[i] for i in top_c],
                                         "country_top3": [inv_co[i] for i in top_o]}
        qual.append(entry)

    out = {
        "layers_all": list(range(nL)), "mas_layers": sorted(mas_layers),
        "city_acc": city_acc, "country_acc": country_acc,
        "chance_city": 1 / len(city_v), "chance_country": 1 / len(country_v),
        "n_city_classes": len(city_v), "n_country_classes": len(country_v),
        "n_train": len(train_rows), "n_eval": len(eval_rows),
        "qualitative": qual,
        "note": "TRAINED DIAGNOSTIC PROBE (linear, per-layer). NOT the model's native output. "
                "Feature=mean-pooled answer-block hidden at step0/t=1. hop1=bridge city, hop2=answer country.",
    }
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n[tl] saved {args.out}")


if __name__ == "__main__":
    main()
