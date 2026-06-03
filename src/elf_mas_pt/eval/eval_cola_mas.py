"""Block-by-block sampling + lesion eval for Cola+MAS.

Generates a single answer block (16 latent positions) via rectified-flow Euler
integration on the frozen Cola DiT augmented by the trained MAS heads:

  v(z_t, t) = v_0(z_t, t) + g_a(z_t, t, ctx_a) * v_a(z_t, t, ctx_a)
                         + g_b(z_t, t, ctx_b) * v_b(z_t, t, ctx_b)

Lesion ablations override the gates at sample time:
  AB        -> both gates active
  A_only    -> g_b := 0
  B_only    -> g_a := 0
  Neither   -> g_a := 0, g_b := 0  (= unconditional Cola)

Decode: VAE.decode(z_final) -> token logits -> argmax -> strip pad/eos -> text.
Scoring: SQuAD-style normalize_answer + EM/F1.
"""
from __future__ import annotations

import argparse
import json
import re
import string
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from elf_mas_pt.model.frozen_cola import FrozenColaBackbone, FrozenColaSpec
from elf_mas_pt.model.mas_heads import CoupledMASHeads, SingleHeadMASWrapper
from elf_mas_pt.model.mas_in_block import (
    MASBlockWrapper, patch_cola_dit_with_mas, set_context_all, clear_context_all,
    collect_mas_params,
)
from elf_mas_pt.training.train_cola import (
    build_batch_inputs, compute_v0_unconditional,
    prime_prefix_kv_cache, compute_v0_block_with_cache, _disable_dit_kv_cache,
)


# ----------------------- scoring -----------------------

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", flags=re.UNICODE)
_PUNCT = set(string.punctuation)


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = _ARTICLES_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def em(pred: str, gold: str) -> float:
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0


def f1(pred: str, gold: str) -> float:
    p_toks = normalize_answer(pred).split()
    g_toks = normalize_answer(gold).split()
    if not p_toks or not g_toks:
        return float(p_toks == g_toks)
    common = Counter(p_toks) & Counter(g_toks)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    prec = overlap / len(p_toks)
    rec = overlap / len(g_toks)
    return 2 * prec * rec / (prec + rec)


# ----------------------- sampling -----------------------

@torch.no_grad()
def sample_mas_block(
    backbone: FrozenColaBackbone,
    mas_model: Optional[torch.nn.Module],
    q_lat: Tensor, q_mask: Tensor,
    ctx_a_lat: Tensor, ctx_a_mask: Tensor,
    ctx_b_lat: Tensor, ctx_b_mask: Tensor,
    T_inf: int = 16,
    ablation: str = "AB",
    T: float = 1000.0,
    prefix_cond: bool = False,
    prompt_lat_list: Optional[List[Tensor]] = None,
    prompt_lens: Optional[List[int]] = None,
    lora_wrappers: Optional[List[MASBlockWrapper]] = None,
) -> Tensor:
    """Sample one answer block (B, block_size, latent_dim) via rectified-flow Euler.

    z_t = (1 - t)*z_target + t*noise  with t in [0,1]
    Euler step: z_{t-dt} = z_t - dt * v_pred
    """
    B = q_lat.shape[0]
    bs = backbone.block_size
    D = backbone.latent_dim
    device = backbone.device
    bb_dtype = q_lat.dtype

    z = torch.randn(B, bs, D, device=device, dtype=torch.float32)

    # Per-agent cross-attn context: [question || ctx_*]
    ctx_a_full = torch.cat([q_lat, ctx_a_lat], dim=1).float()
    ctx_a_full_mask = torch.cat([q_mask, ctx_a_mask], dim=1)
    ctx_b_full = torch.cat([q_lat, ctx_b_lat], dim=1).float()
    ctx_b_full_mask = torch.cat([q_mask, ctx_b_mask], dim=1)

    # x_mask: we generate the whole 16-position block (no answer-length info at inference)
    x_mask = torch.ones(B, bs, dtype=torch.bool, device=device)

    # Prime prefix KV cache ONCE per item-batch (reused across all T_inf steps)
    if prefix_cond:
        assert prompt_lat_list is not None and prompt_lens is not None, \
            "prefix_cond requires prompt_lat_list and prompt_lens"
        # MAS must be OFF during prefix prime — the prime forward processes prefix tokens,
        # not the answer block, so its (B*L_pre) layout doesn't match (B*bs) MAS reshape.
        if lora_wrappers is not None:
            clear_context_all(lora_wrappers)
        prime_prefix_kv_cache(backbone, prompt_lat_list, prompt_lens)
    # Activate LoRA-MAS context for the block-forward portion (with ablation)
    if lora_wrappers is not None:
        set_context_all(
            lora_wrappers,
            ctx_a=ctx_a_lat, ctx_b=ctx_b_lat,
            ctx_a_mask=ctx_a_mask, ctx_b_mask=ctx_b_mask,
            ablation=ablation,
        )
    dt = 1.0 / T_inf
    for i in range(T_inf):
        t_frac = 1.0 - i * dt  # in (dt, 1]
        t_internal = torch.full((B,), t_frac * T, device=device, dtype=torch.float32)

        # Frozen v_0 from Cola DiT
        if prefix_cond:
            v_0 = compute_v0_block_with_cache(
                backbone, z.to(bb_dtype), t_internal.to(bb_dtype),
                prompt_lens=prompt_lens,
            ).float()
        else:
            v_0 = compute_v0_unconditional(backbone, z.to(bb_dtype), t_internal.to(bb_dtype)).float().float()

        if lora_wrappers is not None:
            # In LoRA mode, MAS injection is INSIDE Cola, so v_0 already includes it.
            v = v_0
        else:
            # Tail-head MAS path
            t_frac_t = torch.full((B,), t_frac, device=device, dtype=torch.float32)
            v_a, v_b, g_a, g_b = mas_model(
                z, t_frac_t, ctx_a_full, ctx_b_full,
                x_mask=x_mask, ctx_a_mask=ctx_a_full_mask, ctx_b_mask=ctx_b_full_mask,
            )
            if ablation == "A_only":
                g_b = torch.zeros_like(g_b)
            elif ablation == "B_only":
                g_a = torch.zeros_like(g_a)
            elif ablation == "Neither":
                g_a = torch.zeros_like(g_a)
                g_b = torch.zeros_like(g_b)
            elif ablation != "AB":
                raise ValueError(f"Unknown ablation {ablation!r}")
            v = v_0 + g_a[:, None, None] * v_a + g_b[:, None, None] * v_b
        z = z - dt * v
    if lora_wrappers is not None:
        clear_context_all(lora_wrappers)
    if prefix_cond:
        _disable_dit_kv_cache(backbone)
    return z  # (B, bs, D) float32


# ----------------------- decode -----------------------

@torch.no_grad()
def decode_block_to_texts(backbone: FrozenColaBackbone, z_block: Tensor) -> List[str]:
    """z_block: (B, block_size, latent_dim) float32. Returns list of decoded strings."""
    B, bs, D = z_block.shape
    device = z_block.device
    bb_dtype = next(backbone.vae.parameters()).dtype
    z_flat = z_block.reshape(B * bs, D).to(bb_dtype)
    txt_shape = torch.full((B, 1), bs, dtype=torch.long, device=device)
    logits = backbone.decode_latent(z_flat, txt_shape=txt_shape, txt_q_shape=txt_shape)
    # logits is (1, B*bs, V) per VAE convention; squeeze leading dim
    if logits.dim() == 3:
        logits = logits.squeeze(0)
    pred_ids = logits.argmax(dim=-1).cpu().tolist()  # length B*bs

    pad_id = backbone.spec.pad_token_id
    eos_id = backbone.spec.eos_token_id
    out: List[str] = []
    for i in range(B):
        toks = pred_ids[i * bs:(i + 1) * bs]
        # Strip pad anywhere, and truncate at first eos
        if eos_id in toks:
            toks = toks[:toks.index(eos_id)]
        toks = [t for t in toks if t != pad_id]
        text = backbone.tokenizer.decode(toks)
        out.append(text)
    return out


@torch.no_grad()
def decode_block_to_token_ids(backbone: FrozenColaBackbone, z_block: Tensor) -> List[List[int]]:
    """Like `decode_block_to_texts` but returns raw token IDs per sample (length bs).
    Used for per-token-ID exact-match scoring against gold tokenization, which
    sidesteps the tokenizer-decode merging issue ('Belgium'+'Tor'->'BelgiumTor')."""
    B, bs, D = z_block.shape
    device = z_block.device
    bb_dtype = next(backbone.vae.parameters()).dtype
    z_flat = z_block.reshape(B * bs, D).to(bb_dtype)
    txt_shape = torch.full((B, 1), bs, dtype=torch.long, device=device)
    logits = backbone.decode_latent(z_flat, txt_shape=txt_shape, txt_q_shape=txt_shape)
    if logits.dim() == 3:
        logits = logits.squeeze(0)
    pred_ids = logits.argmax(dim=-1).cpu().tolist()
    return [pred_ids[i * bs:(i + 1) * bs] for i in range(B)]


def per_token_em(pred_ids: List[int], gold_text: str, tokenizer) -> float:
    """1.0 iff pred_ids[:len(gold_toks)] == gold_toks (exact match on first K)."""
    gold_toks = tokenizer.encode(gold_text).ids
    if not gold_toks:
        return 0.0
    K = len(gold_toks)
    return 1.0 if pred_ids[:K] == gold_toks else 0.0


def per_token_f1(pred_ids: List[int], gold_text: str, tokenizer) -> float:
    """F1 over the per-token-ID set in first K positions of pred vs gold tokens."""
    gold_toks = tokenizer.encode(gold_text).ids
    if not gold_toks:
        return 0.0
    K = len(gold_toks)
    pred_head = pred_ids[:K]
    from collections import Counter
    common = Counter(pred_head) & Counter(gold_toks)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    prec = overlap / max(len(pred_head), 1)
    rec = overlap / max(len(gold_toks), 1)
    return 2 * prec * rec / (prec + rec)


# ----------------------- main eval loop -----------------------

def lesion_eval(
    backbone: FrozenColaBackbone,
    mas_model: Optional[torch.nn.Module],
    dataset,
    n_items: int,
    batch_size: int = 8,
    T_inf: int = 16,
    S_q: int = 16,
    S_ctx: int = 64,
    ablations: Optional[List[str]] = None,
    seed: int = 0,
    verbose: bool = True,
    prefix_cond: bool = False,
    lora_wrappers: Optional[List] = None,
) -> Dict:
    """Run lesion matrix: per-bucket × per-ablation EM/F1 over n_items."""
    if ablations is None:
        ablations = ["AB", "A_only", "B_only", "Neither"]

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset), size=min(n_items, len(dataset)), replace=False).tolist()
    items = list(dataset.select(idx))
    if verbose:
        bucket_counts = Counter(r["bucket"] for r in items)
        print(f"[eval] {len(items)} items, buckets: {dict(bucket_counts)}", flush=True)

    # Pre-encode contexts (same for all ablations — context shuffling lives in training, not eval)
    sample_rng = np.random.default_rng(seed + 1)
    results = defaultdict(list)  # (bucket, ablation) -> list of (em, f1, em_tok, f1_tok)
    examples: List[Dict] = []

    n_batches = (len(items) + batch_size - 1) // batch_size
    t0 = time.time()
    for b in range(n_batches):
        rows = items[b * batch_size:(b + 1) * batch_size]
        batch = build_batch_inputs(backbone, rows, variant="full",
                                   S_ctx=S_ctx, S_q=S_q, rng=sample_rng)
        for abl in ablations:
            z_pred = sample_mas_block(
                backbone, mas_model,
                q_lat=batch["q_lat"], q_mask=batch["q_mask"],
                ctx_a_lat=batch["ctx_a_lat"], ctx_a_mask=batch["ctx_a_mask"],
                ctx_b_lat=batch["ctx_b_lat"], ctx_b_mask=batch["ctx_b_mask"],
                T_inf=T_inf, ablation=abl, prefix_cond=prefix_cond,
                prompt_lat_list=batch.get("prompt_lat_list"),
                prompt_lens=batch.get("prompt_lens"),
                lora_wrappers=lora_wrappers,
            )
            preds_text = decode_block_to_texts(backbone, z_pred)
            preds_tok = decode_block_to_token_ids(backbone, z_pred)
            for r, p_text, p_tok in zip(rows, preds_text, preds_tok):
                gold = r["answer"]
                results[(r["bucket"], abl)].append((
                    em(p_text, gold),
                    f1(p_text, gold),
                    per_token_em(p_tok, gold, backbone.tokenizer),
                    per_token_f1(p_tok, gold, backbone.tokenizer),
                ))
                if abl == "AB" and len(examples) < 6:
                    examples.append({"bucket": r["bucket"], "question": r["question"],
                                     "gold": gold, "pred": p_text,
                                     "pred_tok_first_K": p_tok[:len(backbone.tokenizer.encode(gold).ids)]})
        if verbose and (b % 5 == 0 or b == n_batches - 1):
            print(f"[eval] batch {b+1}/{n_batches} ({time.time()-t0:.0f}s)", flush=True)

    # Aggregate (text + per-token)
    summary = {}
    buckets = sorted({k[0] for k in results.keys()})
    for bk in buckets:
        summary[bk] = {}
        for abl in ablations:
            scores = results.get((bk, abl), [])
            if not scores:
                summary[bk][abl] = {"em": None, "f1": None,
                                    "em_tok": None, "f1_tok": None, "n": 0}
                continue
            ems = np.mean([s[0] for s in scores])
            f1s = np.mean([s[1] for s in scores])
            ems_t = np.mean([s[2] for s in scores])
            f1s_t = np.mean([s[3] for s in scores])
            summary[bk][abl] = {"em": float(ems), "f1": float(f1s),
                                "em_tok": float(ems_t), "f1_tok": float(f1s_t),
                                "n": len(scores)}
    return {"summary": summary, "examples": examples,
            "n_items": len(items), "T_inf": T_inf}


def format_matrix(summary: Dict, ablations: List[str], metric: str = "em_tok") -> str:
    """Pretty-print the per-bucket × per-ablation matrix for the chosen metric.

    metric in {'em', 'f1', 'em_tok', 'f1_tok'}: which entry to display in each cell."""
    buckets = sorted(summary.keys())
    lines = []
    head = f"{'bucket':<12s}" + "".join(f"{a:>14s}" for a in ablations)
    lines.append(head)
    lines.append("-" * len(head))
    for bk in buckets:
        cells = [f"{bk:<12s}"]
        for a in ablations:
            d = summary[bk].get(a, {})
            if d.get(metric) is None:
                cells.append(f"{'--':>14s}")
            else:
                cells.append(f"{d[metric]*100:>13.1f}%")
        lines.append("".join(cells))
    return "\n".join(lines)


def asymmetry(summary: Dict, metric: str = "em_tok") -> Dict[str, float]:
    """Δa = EM(AB) − EM(B_only) ; Δb = EM(AB) − EM(A_only). For A-only bucket,
    Δa should be large (need ctx_a) and Δb small (ctx_b is irrelevant).
    Asymmetry ratio = Δa/Δb (or its inverse for B-only). >>1 means specialization."""
    out = {}
    for bk, row in summary.items():
        em_ab = row.get("AB", {}).get(metric)
        em_a = row.get("A_only", {}).get(metric)
        em_b = row.get("B_only", {}).get(metric)
        if None in (em_ab, em_a, em_b):
            continue
        d_a = em_ab - em_b  # drop when ctx_a removed (B-only)
        d_b = em_ab - em_a  # drop when ctx_b removed (A-only)
        out[bk] = {"d_a": float(d_a), "d_b": float(d_b),
                   "ratio_a_over_b": float(d_a / max(d_b, 1e-6)) if d_b > 0 else None,
                   "ratio_b_over_a": float(d_b / max(d_a, 1e-6)) if d_a > 0 else None}
    return out


# ----------------------- CLI -----------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True,
                   help="Run dir containing mas_state_dict.pt and train_log.json")
    p.add_argument("--dataset_dir", default="/home/chenhongrui/elf_mas/data/synth_2fact_v1")
    p.add_argument("--split", default="eval",
                   help="Which split under dataset_dir (eval or dev)")
    p.add_argument("--cola_base", default="/home/chenhongrui/elf_mas/hf_cache/elf-mas-bridge/phase3_assets/cola_dlm")
    p.add_argument("--cola_src", default="/home/chenhongrui/elf_mas/cola_dlm_src")
    p.add_argument("--n_items", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--T_inf", type=int, default=16)
    p.add_argument("--S_q", type=int, default=16)
    p.add_argument("--S_ctx", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--variant", default="full",
                   choices=["full", "identical_ctx", "context_shuffled",
                            "frozen_agents", "single_model"],
                   help="Which MAS architecture the checkpoint was trained with")
    p.add_argument("--prefix_cond", action="store_true",
                   help="Use prefix-conditioned v_0 (must match training setup)")
    p.add_argument("--head_hidden", type=int, default=384)
    p.add_argument("--head_depth", type=int, default=2)
    p.add_argument("--head_heads", type=int, default=6)
    p.add_argument("--gate_hidden", type=int, default=256)
    p.add_argument("--lora_mode", action="store_true",
                   help="Load in-block LoRA MAS checkpoint (lora_state.pt).")
    args = p.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    print(f"[eval] loading Cola backbone")
    spec = FrozenColaSpec(
        dit_path=f"{args.cola_base}/cola_dlm/cola_dit",
        vae_path=f"{args.cola_base}/cola_dlm/cola_vae",
        tokenizer_path=f"{args.cola_base}/tokenizer.json",
        cola_src_path=args.cola_src,
    )
    backbone = FrozenColaBackbone.load(spec, device)

    mas_model = None
    lora_wrappers = None
    if args.lora_mode:
        ckpt_path = Path(args.ckpt_dir) / "lora_state.pt"
        print(f"[eval] loading LoRA state from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        txt_dim = backbone.dit.config.txt_dim if hasattr(backbone.dit.config, "txt_dim") else 2048
        mode = ckpt.get("mode", "dual")  # old checkpoints predate `mode`; they are dual
        lora_wrappers = patch_cola_dit_with_mas(
            backbone.dit,
            txt_dim=txt_dim,
            ctx_lat_dim=backbone.latent_dim,
            layer_indices=ckpt["lora_layer_indices"],
            inner_dim=ckpt["lora_inner"],
            num_heads=ckpt["lora_heads"],
            block_size=backbone.block_size,
            mode=mode,
            single_inner=ckpt.get("single_inner", 848),
            single_heads=ckpt["lora_heads"],
        )
        for w, wckpt in zip(lora_wrappers, ckpt["wrappers"]):
            if mode == "single":
                w.mas_s.load_state_dict(wckpt["mas_s"])
            else:
                w.mas_a.load_state_dict(wckpt["mas_a"])
                w.mas_b.load_state_dict(wckpt["mas_b"])
            w.eval()
        for p_ in backbone.dit.parameters():
            p_.requires_grad_(False)
        n_lora = sum(p.numel() for p in collect_mas_params(lora_wrappers))
        ck_variant = ckpt.get("variant")
        if ck_variant is not None and ck_variant != args.variant:
            print(f"[eval][WARN] --variant={args.variant!r} but checkpoint was trained as "
                  f"{ck_variant!r}; using the checkpoint's architecture (mode={mode}).")
        print(f"[eval] LoRA-mode mode={mode} variant={ck_variant or args.variant} "
              f"patched layers {ckpt['lora_layer_indices']}  loaded {n_lora:,} params")
    else:
        if args.variant == "single_model":
            mas_model = SingleHeadMASWrapper(
                latent_dim=backbone.latent_dim,
                head_hidden_size=int(args.head_hidden * 1.35),
                head_depth=args.head_depth * 2,
                head_num_heads=args.head_heads + 2,
                gate_hidden_size=args.gate_hidden,
            ).to(device)
        else:
            mas_model = CoupledMASHeads(
                latent_dim=backbone.latent_dim,
                head_hidden_size=args.head_hidden,
                head_depth=args.head_depth,
                head_num_heads=args.head_heads,
                gate_hidden_size=args.gate_hidden,
            ).to(device)
        ckpt = Path(args.ckpt_dir) / "mas_state_dict.pt"
        print(f"[eval] loading MAS state dict from {ckpt}")
        sd = torch.load(ckpt, map_location=device)
        mas_model.load_state_dict(sd)
        mas_model.eval()

    print(f"[eval] loading dataset {args.dataset_dir}/{args.split}")
    from datasets import load_from_disk
    try:
        ds = load_from_disk(f"{args.dataset_dir}/{args.split}")
    except FileNotFoundError:
        alt = "dev" if args.split == "eval" else "eval"
        print(f"[eval] split {args.split!r} missing, trying {alt!r}")
        ds = load_from_disk(f"{args.dataset_dir}/{alt}")
    print(f"[eval] {len(ds)} items in split")

    out = lesion_eval(
        backbone=backbone, mas_model=mas_model, dataset=ds,
        n_items=args.n_items, batch_size=args.batch_size, T_inf=args.T_inf,
        S_q=args.S_q, S_ctx=args.S_ctx, seed=args.seed,
        prefix_cond=args.prefix_cond or args.lora_mode,
        lora_wrappers=lora_wrappers,
    )
    abls = ["AB", "A_only", "B_only", "Neither"]
    print("\n[eval] per-bucket per-token EM (%):")
    print(format_matrix(out["summary"], abls, metric="em_tok"))
    print("\n[eval] per-bucket per-token F1 (%):")
    print(format_matrix(out["summary"], abls, metric="f1_tok"))
    print("\n[eval] per-bucket text-EM (%):")
    print(format_matrix(out["summary"], abls, metric="em"))
    asy = asymmetry(out["summary"], metric="em_tok")
    print("\n[eval] asymmetry (per bucket, per-token EM):")
    for bk, d in asy.items():
        print(f"  {bk:<12s} d_a={d['d_a']*100:+5.1f}  d_b={d['d_b']*100:+5.1f}  "
              f"a/b={d.get('ratio_a_over_b')}")
    print("\n[eval] qualitative examples (AB):")
    for ex in out["examples"]:
        print(f"  [{ex['bucket']:<8s}] Q: {ex['question'][:80]!r}\n"
              f"            gold: {ex['gold']!r}\n"
              f"            pred: {ex['pred']!r}")

    save = Path(args.ckpt_dir) / f"eval_n{args.n_items}_T{args.T_inf}.json"
    with open(save, "w") as f:
        json.dump({"args": vars(args), "summary": out["summary"],
                   "asymmetry": asy, "examples": out["examples"]}, f, indent=2)
    print(f"\n[eval] saved to {save}")


if __name__ == "__main__":
    main()
