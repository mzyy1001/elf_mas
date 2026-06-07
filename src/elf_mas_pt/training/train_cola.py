"""Train the Coupled-MAS heads on the frozen Cola backbone (PyTorch).

Mirrors `src/elf_mas/training/train_coupled.py` (JAX) for the Cola substrate:

  - Reuses `synth_2fact_v1` or `musique_2hop_v1` (HF Datasets on disk) — we read
    the RAW TEXT fields (`ctx_a_text`, `ctx_b_text`, `question`, `answer`) and
    re-tokenize with the OLMo 2 tokenizer + VAE-encode per batch. The token IDs
    pre-stored in those datasets are T5-style and discarded here.
  - Loss is rectified-flow velocity MSE: target = noise - z_target, where
    z_target = VAE.encode(answer) (padded to one Cola block of 16 latent
    positions). v_0 is the frozen Cola DiT velocity (unconditional, no KV cache);
    the MAS heads cross-attend to VAE-encoded [question || ctx_*] and add
    `g_a * v_a + g_b * v_b` to v_0.
  - Lesion semantics + variants match the JAX trainer: full / identical_ctx /
    context_shuffled / frozen_agents / single_model.

Run a smoke pass with `--smoke` (5 steps, batch 4) before any longer training.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from elf_mas_pt.model.frozen_cola import FrozenColaBackbone, FrozenColaSpec  # type: ignore
from elf_mas_pt.model.mas_heads import CoupledMASHeads, SingleHeadMASWrapper, combine_velocities
from elf_mas_pt.model.mas_in_block import (
    MASBlockWrapper, patch_cola_dit_with_mas, collect_mas_params,
    set_context_all, clear_context_all, injection_param_count,
)


# Which variants are realizable in the in-block LoRA path, and how.
#   data-side  : differ only in build_batch_inputs (context selection)
#   arch-side  : differ in the MAS module wiring / which params train
_LORA_DATA_VARIANTS = {"full", "identical_ctx", "context_shuffled"}
_LORA_ARCH_VARIANTS = {"frozen_agents", "single_model"}
_LORA_SUPPORTED = _LORA_DATA_VARIANTS | _LORA_ARCH_VARIANTS


def assert_lora_variant_realized(variant, wrappers, n_train, n_total,
                                 txt_dim, ctx_lat_dim, layer_indices,
                                 inner_dim, num_heads):
    """Fail loudly if a variant did not actually change the model/training.

    Guards against the silent-fallback bug where --variant frozen_agents /
    single_model trained a vanilla `full` model because the lora branch ignored
    --variant."""
    if variant not in _LORA_SUPPORTED:
        raise ValueError(f"variant {variant!r} not supported in --lora_mode "
                         f"(supported: {sorted(_LORA_SUPPORTED)})")
    modes = {w.mode for w in wrappers}
    if variant == "single_model":
        assert modes == {"single"}, (
            f"single_model must build single-mode wrappers; got modes={modes}. "
            "The lora path silently fell back to the dual `full` architecture.")
        ref_dual = 2 * len(layer_indices) * injection_param_count(
            txt_dim, ctx_lat_dim, inner_dim, num_heads)
        ratio = n_train / max(ref_dual, 1)
        assert 0.85 <= ratio <= 1.15, (
            f"single_model is not parameter-matched to dual B0: "
            f"{n_train:,} vs {ref_dual:,} (ratio {ratio:.3f}); tune --single_inner.")
    elif variant == "frozen_agents":
        assert modes == {"dual"}, f"frozen_agents must be dual-mode; got {modes}"
        assert 0 < n_train < n_total, (
            f"frozen_agents must freeze the agent transforms (train gates only); "
            f"got n_train={n_train:,} n_total={n_total:,} — nothing was frozen.")
        assert n_train < 0.1 * n_total, (
            f"frozen_agents should train only gates (<10% of params); "
            f"got {n_train:,}/{n_total:,}.")
    else:  # full / identical_ctx / context_shuffled — pure dual, train everything
        assert modes == {"dual"}, f"{variant} must be dual-mode; got {modes}"
        assert n_train == n_total, (
            f"{variant} should train all MAS params; got {n_train:,}/{n_total:,}.")


# ---------------------- batching ----------------------

def encode_texts_to_padded_latents(
    backbone: FrozenColaBackbone, texts: List[str], max_len: int,
) -> Tuple[Tensor, Tensor]:
    """Tokenize + VAE-encode a list of texts, then pad to a common length.

    Returns:
      latents: (B, max_len, latent_dim) — bf16 on device, zero-padded
      mask:    (B, max_len) bool — True at valid positions
    """
    ids_list = [backbone.tokenize(t) for t in texts]
    per_sample = backbone.encode_text(ids_list)  # list of (L_i, latent_dim)
    B = len(per_sample)
    D = backbone.latent_dim
    device = backbone.device
    dtype = per_sample[0].dtype
    out = torch.zeros(B, max_len, D, device=device, dtype=dtype)
    mask = torch.zeros(B, max_len, dtype=torch.bool, device=device)
    for i, lat in enumerate(per_sample):
        n = min(lat.shape[0], max_len)
        out[i, :n] = lat[:n]
        mask[i, :n] = True
    return out, mask


# ---- Cola SQuAD-style prompt (matches inference.py apply_prompt_template) ----

_SQUAD_DEMO = (
    "Context: The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) "
    "were the people who in the 10th and 11th centuries gave their name to Normandy, "
    "a region in France. They were descended from Norse raiders and pirates from "
    "Denmark, Iceland and Norway.\n"
    "Question: In what country is Normandy located?\n"
    "Answer: France\n\n"
)


def build_question_prompt(question: str) -> str:
    """Cola-style one-shot QA prompt with NO actual context. The demo establishes
    the format (Question/Answer); the model has no info to answer from — that's
    by design: contexts are provided through MAS heads, not through Cola's prefix."""
    return _SQUAD_DEMO + f"Question: {question}\nAnswer:"


def encode_prompts_per_sample_block_aligned(
    backbone: FrozenColaBackbone, prompts: List[str],
) -> Tuple[List[Tensor], List[int]]:
    """Tokenize + VAE-encode a list of prompts WITH per-sample block-align padding.

    Returns:
      latents_list: per-sample (L_i, D) tensors where L_i is block-aligned (multiple of block_size)
      lengths:      per-sample L_i ints
    """
    block_size = backbone.block_size
    ids_list_padded: List[List[int]] = []
    for prompt in prompts:
        ids = backbone.tokenize(prompt)
        padded, _ = backbone.pad_to_block(ids, block_size=block_size)
        ids_list_padded.append(padded)
    per_sample = backbone.encode_text(ids_list_padded)  # list of (L_i, D)
    lengths = [lat.shape[0] for lat in per_sample]
    return per_sample, lengths


def encode_answer_to_block(
    backbone: FrozenColaBackbone, texts: List[str], block_size: int = 16,
) -> Tuple[Tensor, Tensor]:
    """Encode answers and pad to ONE Cola block (default 16 latent positions).
    Loss mask covers only the real-answer positions; pad positions don't contribute.

    Returns:
      z_target: (B, block_size, latent_dim)
      mask:     (B, block_size) bool — True at real-answer positions
    """
    return encode_texts_to_padded_latents(backbone, texts, max_len=block_size)


def encode_answer_to_block_with_ids(
    backbone: FrozenColaBackbone, texts: List[str], block_size: int = 16,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Like `encode_answer_to_block` but also returns gold token IDs (B, block_size)
    for use as CE-loss targets. Pad positions are filled with `pad_token_id`.
    Returns: (z_target, ans_mask, gold_ids)."""
    z_target, ans_mask = encode_answer_to_block(backbone, texts, block_size)
    device = backbone.device
    pad_id = backbone.spec.pad_token_id
    B = len(texts)
    gold_ids = torch.full((B, block_size), pad_id, dtype=torch.long, device=device)
    for i, t in enumerate(texts):
        ids = backbone.tokenize(t)
        n = min(len(ids), block_size)
        gold_ids[i, :n] = torch.tensor(ids[:n], dtype=torch.long, device=device)
    return z_target, ans_mask, gold_ids


def make_context_shuffled_indices(B: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Two independent derangements over the batch — Agent A gets ctx from
    a different item than Agent B, and neither equals the current item."""
    def derangement(k):
        for _ in range(8):
            p = rng.permutation(k)
            if k <= 1 or np.all(p != np.arange(k)):
                return p
        return (np.arange(k) + 1) % k
    return derangement(B), derangement(B)


def build_batch_inputs(
    backbone: FrozenColaBackbone,
    rows: List[Dict],
    variant: str,
    S_ctx: int,
    S_q: int,
    rng: np.random.Generator,
) -> Dict[str, Tensor]:
    """Encode + pad the texts for one batch under the chosen variant."""
    q_texts = [r["question"] for r in rows]
    a_texts = [r["ctx_a_text"] for r in rows]
    b_texts = [r["ctx_b_text"] for r in rows]
    ans_texts = [r["answer"] for r in rows]
    buckets = [r["bucket"] for r in rows]

    if variant == "identical_ctx":
        a_texts_used = b_texts_used = [a + " " + b for a, b in zip(a_texts, b_texts)]
    elif variant == "context_shuffled":
        B = len(rows)
        pa, pb = make_context_shuffled_indices(B, rng)
        a_texts_used = [a_texts[pa[i]] for i in range(B)]
        b_texts_used = [b_texts[pb[i]] for i in range(B)]
    else:
        a_texts_used = a_texts
        b_texts_used = b_texts

    q_lat, q_mask = encode_texts_to_padded_latents(backbone, q_texts, max_len=S_q)
    a_lat, a_mask = encode_texts_to_padded_latents(backbone, a_texts_used, max_len=S_ctx)
    b_lat, b_mask = encode_texts_to_padded_latents(backbone, b_texts_used, max_len=S_ctx)
    z_target, ans_mask, gold_ids = encode_answer_to_block_with_ids(
        backbone, ans_texts, block_size=backbone.block_size,
    )

    # Cola DiT prefix: one-shot SQuAD template + question (NO contexts) — per-sample
    # block-aligned variable-length latents
    prompts = [build_question_prompt(q) for q in q_texts]
    prompt_lat_list, prompt_lens = encode_prompts_per_sample_block_aligned(backbone, prompts)
    return {
        "q_lat": q_lat, "q_mask": q_mask,
        "ctx_a_lat": a_lat, "ctx_a_mask": a_mask,
        "ctx_b_lat": b_lat, "ctx_b_mask": b_mask,
        "z_target": z_target, "ans_mask": ans_mask, "gold_ids": gold_ids,
        "buckets": buckets,
        "prompt_lat_list": prompt_lat_list,
        "prompt_lens": prompt_lens,
    }


# ---------------------- noise schedule ----------------------

def sample_t(B: int, p_mean: float = -1.5, p_std: float = 0.8, eps: float = 0.05,
             T: float = 1000.0, device=torch.device("cuda"),
             distribution: str = "logit_normal") -> Tensor:
    """t-sample for the rectified-flow path.

    distribution:
      'logit_normal' (default, ELF-style): biased toward low t (sigmoid of p_mean).
      'uniform': uniform in [eps, 1-eps]. Higher coverage of high-noise cases
                 (where sampling-from-pure-noise actually starts), at the cost of
                 less optimization on the dense low-noise regime."""
    if distribution == "logit_normal":
        z = torch.randn(B, device=device) * p_std + p_mean
        t01 = torch.sigmoid(z).clamp(eps, 1.0 - eps)
    elif distribution == "uniform":
        t01 = torch.rand(B, device=device) * (1.0 - 2.0 * eps) + eps
    else:
        raise ValueError(f"Unknown t-distribution {distribution!r}")
    return t01 * T


# ---------------------- velocity from DiT (unconditional, no KV cache) ----------------------

def compute_v0_unconditional(
    backbone: FrozenColaBackbone, z_block: Tensor, t_per_sample: Tensor,
) -> Tensor:
    """Apply Cola's DiT to z_block in stateless / no-cache mode and return v_0.

    Args:
      z_block: (B, block_size, latent_dim)
      t_per_sample: (B,)
    Returns:
      v_0: (B, block_size, latent_dim)
    """
    B, bs, D = z_block.shape
    z_flat = z_block.reshape(B * bs, D)
    device = z_block.device
    txt_shape = torch.full((B, 1), bs, dtype=torch.long, device=device)
    txt_q_shape = torch.full((B, 1), bs, dtype=torch.long, device=device)
    # Cola's DiT expects per-token timestep tile to match flat length
    t_per_token = t_per_sample.repeat_interleave(bs)
    backbone.enable_kv_cache(False)
    out = backbone.dit(
        txt=z_flat,
        txt_shape=txt_shape,
        txt_q_shape=txt_q_shape,
        timestep=t_per_token.to(z_block.dtype),
        update_kv=False,
        use_kv_cache=False,
    )
    return out.txt_sample.reshape(B, bs, D)


def _reset_dit_kv_cache(backbone: FrozenColaBackbone):
    """Clear + re-enable empty KV cache on all DiT blocks."""
    for blk in backbone.dit.blocks:
        blk.set_kv_cache(False)
        blk.set_kv_cache(True)


def _disable_dit_kv_cache(backbone: FrozenColaBackbone):
    for blk in backbone.dit.blocks:
        blk.set_kv_cache(False)


def prime_prefix_kv_cache(
    backbone: FrozenColaBackbone,
    prompt_lat_list: List[Tensor],
    prompt_lens: List[int],
) -> Tuple[Tensor, Tensor]:
    """Reset DiT KV cache and prime it with per-sample variable-length prefixes
    at timestep=0. Returns (txt_shape_prefix, dummy_ts) for reuse by callers."""
    bb_dtype = next(backbone.dit.parameters()).dtype
    device = backbone.device
    _reset_dit_kv_cache(backbone)
    # Concatenate per-sample latents into one flat tensor
    flat = torch.cat([lat.to(bb_dtype) for lat in prompt_lat_list], dim=0)  # (sum_L, D)
    txt_shape_prefix = torch.tensor([[L] for L in prompt_lens], device=device, dtype=torch.long)
    ts_prefix = torch.zeros(flat.shape[0], device=device, dtype=bb_dtype)
    _ = backbone.dit(
        txt=flat,
        txt_shape=txt_shape_prefix,
        txt_q_shape=txt_shape_prefix,
        timestep=ts_prefix,
        update_kv=True,
        use_kv_cache=True,
    )
    return txt_shape_prefix, ts_prefix


def compute_v0_block_with_cache(
    backbone: FrozenColaBackbone,
    z_block: Tensor,             # (B, bs, D)
    t_per_sample: Tensor,        # (B,)
    prompt_lens: List[int],      # per-sample prefix lengths
) -> Tensor:
    """Forward the answer block through Cola DiT assuming KV cache is already
    primed with per-sample prefixes via `prime_prefix_kv_cache`. Returns velocity
    at block positions (B, bs, D)."""
    B, bs, D = z_block.shape
    device = z_block.device
    bb_dtype = next(backbone.dit.parameters()).dtype

    z_flat = z_block.reshape(B * bs, D).to(bb_dtype)
    txt_shape_cum = torch.tensor([[L + bs] for L in prompt_lens], device=device, dtype=torch.long)
    txt_q_shape = torch.full((B, 1), bs, device=device, dtype=torch.long)
    ts_block = t_per_sample.to(bb_dtype)[:, None].expand(-1, bs).reshape(-1)
    out = backbone.dit(
        txt=z_flat,
        txt_shape=txt_shape_cum,
        txt_q_shape=txt_q_shape,
        timestep=ts_block,
        update_kv=False,
        use_kv_cache=True,
    )
    return out.txt_sample.reshape(B, bs, D)


def compute_v0_with_prefix(
    backbone: FrozenColaBackbone,
    z_block: Tensor,
    t_per_sample: Tensor,
    prompt_lat_list: List[Tensor],
    prompt_lens: List[int],
) -> Tensor:
    """Cola-style per-sample prefix conditioning via KV cache. One-shot helper:
    primes prefix, computes block velocity, clears cache. (For repeated calls
    with the same prefix — e.g., across diffusion steps — call
    `prime_prefix_kv_cache` once and then `compute_v0_block_with_cache` repeatedly.)"""
    prime_prefix_kv_cache(backbone, prompt_lat_list, prompt_lens)
    v_block = compute_v0_block_with_cache(backbone, z_block, t_per_sample, prompt_lens)
    _disable_dit_kv_cache(backbone)
    return v_block


# ---------------------- training step ----------------------

@dataclass
class TrainConfig:
    S_q: int = 16     # question latent length budget
    S_ctx: int = 64   # per-agent context latent length budget
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-3
    p_mean: float = -1.5
    p_std: float = 0.8
    t_eps: float = 0.05
    T: float = 1000.0
    t_distribution: str = "logit_normal"  # or 'uniform'


def loss_step_lora(backbone, wrappers, batch, tcfg, lambda_ce: float = 0.0
                   ) -> Tuple[Tensor, Dict[str, float]]:
    """Loss step for in-block LoRA-style MAS.

    If lambda_ce > 0, adds an auxiliary cross-entropy loss on the x-prediction
    decoded through the frozen Cola VAE: forces decode-correct latents directly.

    Always uses prefix conditioning."""
    z_target = batch["z_target"]
    ans_mask = batch["ans_mask"]
    B = z_target.shape[0]
    device = z_target.device
    bb_dtype = z_target.dtype

    t = sample_t(B, tcfg.p_mean, tcfg.p_std, tcfg.t_eps, T=tcfg.T, device=device,
                 distribution=tcfg.t_distribution)
    noise = torch.randn_like(z_target)
    t_frac = (t / tcfg.T).clamp(0.0, 1.0)
    z_t = ((1.0 - t_frac)[:, None, None] * z_target.float()
           + t_frac[:, None, None] * noise.float())
    v_target = noise.float() - z_target.float()

    # 1) Prime prefix: MAS off, only Cola sees the prefix
    clear_context_all(wrappers)
    with torch.no_grad():
        prime_prefix_kv_cache(backbone, batch["prompt_lat_list"], batch["prompt_lens"])

    # 2) Block forward with MAS injection active
    set_context_all(
        wrappers,
        ctx_a=batch["ctx_a_lat"], ctx_b=batch["ctx_b_lat"],
        ctx_a_mask=batch["ctx_a_mask"], ctx_b_mask=batch["ctx_b_mask"],
        ablation="AB",
    )
    v_pred = compute_v0_block_with_cache(
        backbone, z_t.to(bb_dtype), t.to(bb_dtype),
        prompt_lens=batch["prompt_lens"],
    ).float()
    clear_context_all(wrappers)
    _disable_dit_kv_cache(backbone)

    sqerr = (v_pred - v_target).pow(2).mean(dim=-1)
    valid = ans_mask.float()
    loss_velocity = (sqerr * valid).sum() / valid.sum().clamp(min=1.0)

    stats = {
        "loss_velocity": float(loss_velocity.detach()),
        "v_pred_l2": float(v_pred.pow(2).mean().sqrt().detach()),
        "v_target_l2": float(v_target.pow(2).mean().sqrt().detach()),
    }

    if lambda_ce > 0.0:
        # x-prediction: z_clean_pred = z_t - t_frac * v_pred
        # (since z_t = (1-t) z_clean + t noise, and v_target = noise - z_clean,
        #  so z_clean = z_t - t (noise - z_clean) → not closed-form in z_t alone;
        #  the standard x-prediction shortcut from v at fractional-t is
        #  z_clean = z_t - t * v.)
        z_clean_pred = z_t - t_frac[:, None, None] * v_pred
        # Decode through frozen VAE with gradients ON (VAE weights frozen, but
        # gradients flow back through to z_clean_pred and hence MAS params).
        bs = z_clean_pred.shape[1]
        D = z_clean_pred.shape[2]
        z_flat = z_clean_pred.reshape(B * bs, D).to(bb_dtype)
        txt_shape = torch.full((B, 1), bs, dtype=torch.long, device=device)
        logits = backbone.vae.decode(
            z=z_flat,
            txt_shape=txt_shape,
            txt_q_shape=txt_shape,
        )
        if logits.dim() == 3:
            logits = logits.squeeze(0)  # (B*bs, vocab)
        gold_ids_flat = batch["gold_ids"].reshape(-1)
        # Per-token CE (masked by ans_mask)
        ce_per = F.cross_entropy(logits.float(), gold_ids_flat,
                                 reduction="none")  # (B*bs,)
        valid_flat = ans_mask.reshape(-1).float()
        loss_ce = (ce_per * valid_flat).sum() / valid_flat.sum().clamp(min=1.0)
        stats["loss_ce"] = float(loss_ce.detach())
        # Per-position argmax accuracy for monitoring
        with torch.no_grad():
            pred_ids = logits.argmax(dim=-1)
            correct = (pred_ids == gold_ids_flat).float() * valid_flat
            stats["tok_acc"] = float(correct.sum() / valid_flat.sum().clamp(min=1.0))
        loss = loss_velocity + lambda_ce * loss_ce
    else:
        loss = loss_velocity

    stats["loss"] = float(loss.detach())
    return loss, stats


def loss_step(backbone, mas_model, batch, tcfg, prefix_cond: bool = False
              ) -> Tuple[Tensor, Dict[str, float]]:
    """Single training step. Returns (loss, stats)."""
    z_target = batch["z_target"]
    ans_mask = batch["ans_mask"]
    B = z_target.shape[0]
    device = z_target.device

    # Sample t and build noisy z_t (rectified flow path)
    bb_dtype = z_target.dtype
    t = sample_t(B, tcfg.p_mean, tcfg.p_std, tcfg.t_eps, T=tcfg.T, device=device)
    noise = torch.randn_like(z_target)
    t_frac = (t / tcfg.T).clamp(0.0, 1.0)
    z_t = ((1.0 - t_frac)[:, None, None] * z_target.float()
           + t_frac[:, None, None] * noise.float())
    v_target = noise.float() - z_target.float()  # rectified flow velocity target

    # Frozen v_0 from Cola DiT
    with torch.no_grad():
        if prefix_cond:
            v_0 = compute_v0_with_prefix(
                backbone, z_t.to(bb_dtype), t.to(bb_dtype),
                prompt_lat_list=batch["prompt_lat_list"],
                prompt_lens=batch["prompt_lens"],
            )
        else:
            v_0 = compute_v0_unconditional(backbone, z_t.to(bb_dtype), t.to(bb_dtype))

    # Build cross-attn context for each agent: [question_latent || ctx_*_latent]
    ctx_a_full = torch.cat([batch["q_lat"], batch["ctx_a_lat"]], dim=1)
    ctx_a_full_mask = torch.cat([batch["q_mask"], batch["ctx_a_mask"]], dim=1)
    ctx_b_full = torch.cat([batch["q_lat"], batch["ctx_b_lat"]], dim=1)
    ctx_b_full_mask = torch.cat([batch["q_mask"], batch["ctx_b_mask"]], dim=1)

    # MAS forward (trainable)
    v_a, v_b, g_a, g_b = mas_model(
        z_t, t_frac, ctx_a_full.float(), ctx_b_full.float(),
        x_mask=ans_mask, ctx_a_mask=ctx_a_full_mask, ctx_b_mask=ctx_b_full_mask,
    )
    v_pred = combine_velocities(v_0.float(), v_a, v_b, g_a, g_b)
    sqerr = (v_pred - v_target).pow(2).mean(dim=-1)  # (B, block_size)
    valid = ans_mask.float()
    loss = (sqerr * valid).sum() / valid.sum().clamp(min=1.0)

    stats = {
        "loss": float(loss.detach()),
        "g_a_mean": float(g_a.mean().detach()),
        "g_b_mean": float(g_b.mean().detach()),
        "v_a_l2": float(v_a.pow(2).mean().sqrt().detach()),
        "v_b_l2": float(v_b.pow(2).mean().sqrt().detach()),
        "v_0_l2": float(v_0.float().pow(2).mean().sqrt().detach()),
    }
    return loss, stats


# ---------------------- main ----------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", default="/home/chenhongrui/elf_mas/data/synth_2fact_v1")
    p.add_argument("--cola_base", default="/home/chenhongrui/elf_mas/hf_cache/elf-mas-bridge/phase3_assets/cola_dlm")
    p.add_argument("--cola_src", default="/home/chenhongrui/elf_mas/cola_dlm_src")
    p.add_argument("--variant", default="full",
                   choices=["full", "identical_ctx", "context_shuffled", "frozen_agents", "single_model"])
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--S_q", type=int, default=16)
    p.add_argument("--S_ctx", type=int, default=64)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--out_dir", default=None,
                   help="If set, save MAS state_dict and stats to this directory at the end.")
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--prefix_cond", action="store_true",
                   help="Feed [question||ctx_a||ctx_b] as prefix to Cola DiT (recommended).")
    p.add_argument("--head_hidden", type=int, default=384)
    p.add_argument("--head_depth", type=int, default=2)
    p.add_argument("--head_heads", type=int, default=6)
    p.add_argument("--gate_hidden", type=int, default=256)
    p.add_argument("--lora_mode", action="store_true",
                   help="Use in-block MAS LoRA-style injection (per-layer cross-attn + gate) "
                        "instead of the tail MAS heads.")
    p.add_argument("--lora_layers", default="12,18",
                   help="Comma-separated DiT block indices to patch with MAS injection.")
    p.add_argument("--lora_inner", type=int, default=512)
    p.add_argument("--lora_heads", type=int, default=8)
    p.add_argument("--single_inner", type=int, default=848,
                   help="Inner dim of the single_model (B4) head; default 848 "
                        "param-matches the dual B0 (2x512 heads) within ~1%.")
    p.add_argument("--lambda_ce", type=float, default=0.0,
                   help="Weight for token-CE auxiliary loss via frozen VAE decoder. "
                        "0 disables CE; values like 0.1/0.3/1.0 work as multipliers on velocity MSE.")
    p.add_argument("--t_distribution", default="logit_normal",
                   choices=["logit_normal", "uniform"])
    p.add_argument("--buckets", default="",
                   help="Comma-separated bucket filter for training (e.g. 'A-only,B-only' for a "
                        "single-hop curriculum warm-up). Empty = all buckets.")
    p.add_argument("--resume_from", default="",
                   help="Path to a lora_state.pt to initialize MAS wrappers from (curriculum "
                        "phase-2 resume). Only valid with --lora_mode.")
    args = p.parse_args()

    if args.smoke:
        args.num_steps = 5
        args.batch_size = 4

    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    print(f"[train] loading Cola backbone")
    spec = FrozenColaSpec(
        dit_path=f"{args.cola_base}/cola_dlm/cola_dit",
        vae_path=f"{args.cola_base}/cola_dlm/cola_vae",
        tokenizer_path=f"{args.cola_base}/tokenizer.json",
        cola_src_path=args.cola_src,
    )
    backbone = FrozenColaBackbone.load(spec, device)
    print(f"[train] block_size={backbone.block_size} latent_dim={backbone.latent_dim}")

    print(f"[train] loading dataset {args.dataset_dir}")
    from datasets import load_from_disk
    ds = load_from_disk(f"{args.dataset_dir}/train")
    print(f"[train] {len(ds)} train items")
    if args.buckets:
        keep = set(b.strip() for b in args.buckets.split(","))
        ds = ds.filter(lambda r: r["bucket"] in keep)
        print(f"[train] bucket filter {sorted(keep)} -> {len(ds)} items")

    # Build MAS model
    mas_wrappers = []
    if args.lora_mode:
        if args.variant not in _LORA_SUPPORTED:
            raise ValueError(f"variant {args.variant!r} not supported in --lora_mode "
                             f"(supported: {sorted(_LORA_SUPPORTED)})")
        layer_indices = [int(x) for x in args.lora_layers.split(",")]
        # Inspect Cola DiT to get txt_dim
        txt_dim = backbone.dit.config.txt_dim if hasattr(backbone.dit.config, "txt_dim") else 2048
        # B4 single_model uses a single concat-context head; all others are dual.
        mas_mode = "single" if args.variant == "single_model" else "dual"
        mas_wrappers = patch_cola_dit_with_mas(
            backbone.dit,
            txt_dim=txt_dim,
            ctx_lat_dim=backbone.latent_dim,
            layer_indices=layer_indices,
            inner_dim=args.lora_inner,
            num_heads=args.lora_heads,
            block_size=backbone.block_size,
            mode=mas_mode,
            single_inner=args.single_inner,
            single_heads=args.lora_heads,
        )
        # Curriculum resume: initialize MAS wrappers from a prior checkpoint.
        if args.resume_from:
            rck = torch.load(args.resume_from, map_location=device)
            assert rck.get("mode", "dual") == mas_mode, "resume mode mismatch"
            for w, wc in zip(mas_wrappers, rck["wrappers"]):
                if mas_mode == "single":
                    w.mas_s.load_state_dict(wc["mas_s"])
                else:
                    w.mas_a.load_state_dict(wc["mas_a"]); w.mas_b.load_state_dict(wc["mas_b"])
            print(f"[train] resumed MAS wrappers from {args.resume_from}")
        # Ensure Cola DiT base params remain frozen
        for p_ in backbone.dit.parameters():
            p_.requires_grad_(False)
        # Re-enable grad on MAS params only
        mas_params = collect_mas_params(mas_wrappers)
        for p_ in mas_params:
            p_.requires_grad_(True)
        n_total = sum(p.numel() for p in mas_params)
        # B2 frozen_agents: freeze the agent transforms, train ONLY the gates.
        if args.variant == "frozen_agents":
            for w in mas_wrappers:
                injs = [w.mas_s] if w.mode == "single" else [w.mas_a, w.mas_b]
                for inj in injs:
                    for name, p_ in inj.named_parameters():
                        if not name.startswith("gate"):
                            p_.requires_grad_(False)
        train_params = [p for p in mas_params if p.requires_grad]
        n_params = sum(p.numel() for p in train_params)
        # GUARD: ensure the requested variant actually reshaped the model/training.
        assert_lora_variant_realized(
            args.variant, mas_wrappers, n_params, n_total,
            txt_dim, backbone.latent_dim, layer_indices,
            args.lora_inner, args.lora_heads)
        print(f"[train] LoRA-mode variant={args.variant!r} mode={mas_mode} "
              f"patched layers {layer_indices}  trainable params: {n_params:,}/{n_total:,}")
        mas_model = None
        optim = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=1e-3)
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
        n_params = sum(p.numel() for p in mas_model.parameters())
        print(f"[train] variant={args.variant!r}  trainable params: {n_params:,}")
        freeze_residual = (args.variant == "frozen_agents")
        if freeze_residual:
            for name, p_ in mas_model.named_parameters():
                if name.startswith("agent_a") or name.startswith("agent_b") or name.startswith("head"):
                    p_.requires_grad_(False)
        optim = torch.optim.AdamW(
            [p_ for p_ in mas_model.parameters() if p_.requires_grad],
            lr=args.lr, weight_decay=1e-3,
        )

    tcfg = TrainConfig(batch_size=args.batch_size, lr=args.lr, S_q=args.S_q,
                        S_ctx=args.S_ctx, t_distribution=args.t_distribution)

    rng = np.random.default_rng(args.seed)
    shuffle_rng = np.random.default_rng(args.seed + 17)

    n = len(ds)
    t0 = time.time()
    log_history: List[Dict[str, float]] = []
    for step in range(args.num_steps):
        idx = rng.integers(0, n, size=args.batch_size)
        rows = list(ds.select(idx.tolist()))
        batch = build_batch_inputs(backbone, rows, args.variant, tcfg.S_ctx, tcfg.S_q, shuffle_rng)
        if args.lora_mode:
            loss, stats = loss_step_lora(backbone, mas_wrappers, batch, tcfg,
                                         lambda_ce=args.lambda_ce)
        else:
            loss, stats = loss_step(backbone, mas_model, batch, tcfg, prefix_cond=args.prefix_cond)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if step < 3 or step % args.log_every == 0 or step == args.num_steps - 1:
            elapsed = time.time() - t0
            sstr = " ".join(f"{k}={v:.4f}" for k, v in stats.items())
            print(f"[train] step={step:5d}  {sstr}  ({elapsed:.1f}s)", flush=True)
            log_history.append({"step": step, "elapsed_s": elapsed, **stats})
    total = time.time() - t0
    print(f"[train] done; total {total:.1f}s for {args.num_steps} steps", flush=True)

    if args.out_dir:
        import json
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        if args.lora_mode:
            mas_mode = mas_wrappers[0].mode if mas_wrappers else "dual"
            if mas_mode == "single":
                wrapper_states = [{"mas_s": w.mas_s.state_dict()} for w in mas_wrappers]
            else:
                wrapper_states = [
                    {"mas_a": w.mas_a.state_dict(), "mas_b": w.mas_b.state_dict()}
                    for w in mas_wrappers
                ]
            ckpt = {
                "lora_layer_indices": [int(x) for x in args.lora_layers.split(",")],
                "lora_inner": args.lora_inner,
                "lora_heads": args.lora_heads,
                "mode": mas_mode,
                "single_inner": args.single_inner,
                "variant": args.variant,
                "wrappers": wrapper_states,
            }
            torch.save(ckpt, out / "lora_state.pt")
        else:
            torch.save(mas_model.state_dict(), out / "mas_state_dict.pt")
        with open(out / "train_log.json", "w") as f:
            json.dump({
                "args": vars(args),
                "total_s": total,
                "history": log_history,
            }, f, indent=2)
        print(f"[train] saved checkpoint + log to {out}", flush=True)


# Tiny shim so we don't import a non-existent name from frozen_cola
FrozenBackboneSpec_to_cola_spec = lambda x: x


if __name__ == "__main__":
    main()
