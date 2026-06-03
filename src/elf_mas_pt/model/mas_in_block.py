"""Per-layer MAS injection inside Cola DiT — the 'LoRA' variant chosen by user.

Instead of a single MAS head appended to v_0, this wires per-agent trainable
cross-attention blocks INTO selected Cola DiT layers. The frozen Cola block
runs first; then for each agent we apply:
    h <- h + g_x(h) * cross_attn_x(h, ctx_x)
The gates and cross-attn modules are trained end-to-end; Cola's weights stay
frozen. Lesion ablations zero out gates per agent.

Design notes:
  - We don't modify Cola's nn.Linear in place; we WRAP the block. This keeps
    the patch reversible and avoids breaking Cola's KV cache + RoPE plumbing.
  - The cross-attn op is small: hidden_size = `inner_dim` (default 512) << Cola's
    txt_dim (2048), with a project-down / project-up pair to keep param count
    manageable. Roughly 2.5M params per agent per layer at default sizes.
  - Cola DiT works on FLAT tensors (sum_L, txt_dim) with per-sample lengths in
    txt_shape. During the answer-block forward (the only place we activate MAS),
    every sample has txt_q_shape == block_size, so we can reshape flat
    (B*bs, txt_dim) -> (B, bs, txt_dim) cleanly.
  - The wrapper carries per-batch state set BEFORE the forward (ctx_a, ctx_b,
    ablation). Reset / set via `set_mas_context(...)` and `clear_mas_context()`.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MASInjection(nn.Module):
    """Per-agent cross-attn residual: h <- h + g(h) * out_proj(cross_attn(h_d, ctx_d)).

    Inner projection drops txt_dim (2048) -> inner_dim (512), cross-attn lives at
    inner_dim, output is up-projected back to txt_dim. Zero-init out_proj so
    untrained injection is a no-op.
    """
    def __init__(self, txt_dim: int, ctx_lat_dim: int,
                 inner_dim: int = 512, num_heads: int = 8,
                 init_gate_bias: float = -2.0):
        super().__init__()
        self.txt_dim = txt_dim
        self.inner_dim = inner_dim
        # Project Cola hidden -> inner_dim for cross-attn Q
        self.q_norm = nn.LayerNorm(txt_dim)
        self.q_in = nn.Linear(txt_dim, inner_dim)
        # Project private context (latent dim, e.g. 16) -> inner_dim for K/V
        self.ctx_in = nn.Linear(ctx_lat_dim, inner_dim)
        # Cross-attn at inner_dim
        self.cross_attn = nn.MultiheadAttention(inner_dim, num_heads, batch_first=True)
        # Up-project back to txt_dim, zero-init for residual safety
        self.out_proj = nn.Linear(inner_dim, txt_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        # Gate (scalar per sample): pooled-h + ctx pool -> sigmoid
        self.gate = nn.Linear(inner_dim * 2, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, init_gate_bias)

    def forward(self, h: Tensor, ctx_lat: Tensor, ctx_mask: Optional[Tensor]) -> Tensor:
        """
        h: (B, bs, txt_dim) — Cola hidden state at this layer's output
        ctx_lat: (B, S_ctx, ctx_lat_dim) — VAE-encoded private context (padded)
        ctx_mask: (B, S_ctx) bool — True at valid context positions
        Returns: residual (B, bs, txt_dim) — already gated, ready to add to h.
        """
        q = self.q_in(self.q_norm(h))                 # (B, bs, inner)
        kv = self.ctx_in(ctx_lat.to(q.dtype))         # (B, S_ctx, inner)
        kpm = (~ctx_mask.bool()) if ctx_mask is not None else None  # True = PAD
        attn_out, _ = self.cross_attn(q, kv, kv, key_padding_mask=kpm, need_weights=False)
        # Gate uses pooled q + pooled kv (keep all in kv.dtype to avoid bf16/fp32 mix)
        if ctx_mask is not None:
            m = ctx_mask.to(kv.dtype).unsqueeze(-1)
            kv_pool = (kv * m).sum(1) / m.sum(1).clamp(min=1.0)
        else:
            kv_pool = kv.mean(dim=1)
        q_pool = q.mean(dim=1)
        g = torch.sigmoid(self.gate(torch.cat([q_pool, kv_pool], dim=-1)))  # (B, 1)
        # Residual back to txt_dim
        delta = self.out_proj(attn_out)               # (B, bs, txt_dim)
        return g[:, None, :] * delta                  # broadcast over bs


class MASBlockWrapper(nn.Module):
    """Wraps a single Cola DiT block; after the block's frozen forward, applies
    per-agent MAS injection IFF `self.apply_mas == True`.

    The wrapper carries lazily-set per-batch state (`ctx_a`, `ctx_b`, ablation).
    The original ColaDiTBlock's forward signature is preserved.
    """
    def __init__(self, original_block: nn.Module, txt_dim: int,
                 ctx_lat_dim: int, inner_dim: int = 512, num_heads: int = 8,
                 block_size: int = 16, mode: str = "dual",
                 single_inner: int = 848, single_heads: int = 8):
        super().__init__()
        assert mode in {"dual", "single"}, f"unknown wrapper mode {mode!r}"
        self.original = original_block
        self.block_size = block_size
        self.mode = mode
        if mode == "dual":
            # Two per-agent residual heads (variants B0 full / B6 / B7 / B2).
            self.mas_a = MASInjection(txt_dim, ctx_lat_dim, inner_dim, num_heads)
            self.mas_b = MASInjection(txt_dim, ctx_lat_dim, inner_dim, num_heads)
        else:
            # B4 single_model: ONE param-matched head over concat(ctx_a, ctx_b).
            self.mas_s = MASInjection(txt_dim, ctx_lat_dim, single_inner, single_heads)
        # Per-batch state (set by trainer / sampler)
        self._ctx_a: Optional[Tensor] = None
        self._ctx_b: Optional[Tensor] = None
        self._ctx_a_mask: Optional[Tensor] = None
        self._ctx_b_mask: Optional[Tensor] = None
        self._apply_mas: bool = False
        self._ablation: str = "AB"   # AB / A_only / B_only / Neither
        self._B: int = 0

    def set_mas_context(self, ctx_a: Tensor, ctx_b: Tensor,
                        ctx_a_mask: Tensor, ctx_b_mask: Tensor,
                        ablation: str = "AB"):
        """Stash per-batch private contexts + ablation. Call before the DiT pass
        that processes the answer block."""
        assert ablation in {"AB", "A_only", "B_only", "Neither"}
        self._ctx_a, self._ctx_b = ctx_a, ctx_b
        self._ctx_a_mask, self._ctx_b_mask = ctx_a_mask, ctx_b_mask
        self._ablation = ablation
        self._apply_mas = True
        self._B = ctx_a.shape[0]

    def clear_mas_context(self):
        """Disable MAS injection (e.g., during the prefix prime pass)."""
        self._apply_mas = False

    def forward(self, txt, **kwargs):
        out = self.original(txt, **kwargs)
        if not self._apply_mas or self._ctx_a is None:
            return out
        B = self._B
        bs = self.block_size
        if out.shape[0] != B * bs:
            # Not the block-forward pass (e.g., prefix prime ran with apply_mas=False
            # but a stale call slipped through). Skip injection.
            return out
        h = out.reshape(B, bs, -1)
        if self.mode == "single":
            # Single head over concat(ctx_a, ctx_b); lesions mask the A- or B-half.
            if self._ablation == "Neither":
                return out  # residual is identically zero
            ctx = torch.cat([self._ctx_a, self._ctx_b], dim=1)
            mask = torch.cat([self._ctx_a_mask, self._ctx_b_mask], dim=1)
            Sa = self._ctx_a_mask.shape[1]
            if self._ablation == "A_only":      # keep A's context, drop B's
                mask = mask.clone(); mask[:, Sa:] = False
            elif self._ablation == "B_only":    # keep B's context, drop A's
                mask = mask.clone(); mask[:, :Sa] = False
            h = h + self.mas_s(h, ctx, mask)
            return h.reshape(B * bs, -1)
        # dual mode (B0 / B6 / B7 / B2)
        residual = torch.zeros_like(h)
        if self._ablation in {"AB", "A_only"}:
            residual = residual + self.mas_a(h, self._ctx_a, self._ctx_a_mask)
        if self._ablation in {"AB", "B_only"}:
            residual = residual + self.mas_b(h, self._ctx_b, self._ctx_b_mask)
        h = h + residual
        return h.reshape(B * bs, -1)

    # Pass through KV cache toggles to the original block
    def set_kv_cache(self, flag):
        if hasattr(self.original, "set_kv_cache"):
            self.original.set_kv_cache(flag)


def patch_cola_dit_with_mas(
    dit: nn.Module,
    txt_dim: int,
    ctx_lat_dim: int,
    layer_indices: List[int],
    inner_dim: int = 512,
    num_heads: int = 8,
    block_size: int = 16,
    mode: str = "dual",
    single_inner: int = 848,
    single_heads: int = 8,
) -> List[MASBlockWrapper]:
    """Replace selected DiT blocks (`dit.blocks[i]`) with MASBlockWrapper.
    Returns the list of wrappers for downstream context-setting.

    mode="dual"   -> two per-agent heads (B0/B6/B7/B2).
    mode="single" -> one param-matched head over concat(ctx_a,ctx_b) (B4)."""
    wrappers: List[MASBlockWrapper] = []
    for idx in layer_indices:
        orig = dit.blocks[idx]
        p0 = next(orig.parameters())
        wrapper = MASBlockWrapper(orig, txt_dim=txt_dim, ctx_lat_dim=ctx_lat_dim,
                                  inner_dim=inner_dim, num_heads=num_heads,
                                  block_size=block_size, mode=mode,
                                  single_inner=single_inner, single_heads=single_heads)
        # Move ONLY the new MAS submodules to Cola's device + dtype.
        # The original block is already there; moving the whole wrapper would
        # re-cast the frozen Cola weights (no-op but wasteful).
        if mode == "single":
            wrapper.mas_s = wrapper.mas_s.to(device=p0.device, dtype=p0.dtype)
        else:
            wrapper.mas_a = wrapper.mas_a.to(device=p0.device, dtype=p0.dtype)
            wrapper.mas_b = wrapper.mas_b.to(device=p0.device, dtype=p0.dtype)
        dit.blocks[idx] = wrapper
        wrappers.append(wrapper)
    return wrappers


def collect_mas_params(wrappers: List[MASBlockWrapper]) -> List[nn.Parameter]:
    """All MAS params (cross-attn + gates + projections) across wrappers."""
    params = []
    for w in wrappers:
        if w.mode == "single":
            params.extend(w.mas_s.parameters())
        else:
            params.extend(w.mas_a.parameters())
            params.extend(w.mas_b.parameters())
    return params


def injection_param_count(txt_dim: int, ctx_lat_dim: int,
                          inner_dim: int, num_heads: int = 8) -> int:
    """Param count of one MASInjection — used to size/verify the param-matched
    single_model (B4) head against the dual (B0) reference."""
    m = MASInjection(txt_dim, ctx_lat_dim, inner_dim, num_heads)
    return sum(p.numel() for p in m.parameters())


def set_context_all(wrappers: List[MASBlockWrapper],
                    ctx_a: Tensor, ctx_b: Tensor,
                    ctx_a_mask: Tensor, ctx_b_mask: Tensor,
                    ablation: str = "AB"):
    """Bulk-set per-batch state on every wrapper."""
    for w in wrappers:
        w.set_mas_context(ctx_a, ctx_b, ctx_a_mask, ctx_b_mask, ablation)


def clear_context_all(wrappers: List[MASBlockWrapper]):
    for w in wrappers:
        w.clear_mas_context()
