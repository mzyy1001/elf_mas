"""PyTorch port of the Coupled-MAS heads, sized for Cola's 16-dim latent.

Mirrors `src/elf_mas/model/coupled_mas.py` (Flax) one-to-one:

  - TimestepEmbedder        : sinusoidal time + 2-layer MLP -> (B, H)
  - VelocityResidualHead    : cross-attn transformer over (z_t, t, ctx_emb)
                              with zero-init out_proj so untrained head contributes 0
  - StateTimeGate           : MLP over pooled(z), pooled(ctx), t_emb -> sigmoid scalar
                              with bias-init -2 so initial gate ~ 0.12
  - CoupledMASHeads         : (v_a, v_b, g_a, g_b) — same 4-tuple API as JAX
  - SingleHeadMASWrapper    : B4 matched-baseline — single head + gate over concat ctx

Default sizes tuned so the coupled wrapper sits ~5M trainable params; the single
wrapper widens (depth=4, hidden=448) to roughly match.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _attn_mask_for_keys(key_mask: Optional[Tensor]) -> Optional[Tensor]:
    """Convert a (B, S_k) bool mask to (B, 1, 1, S_k) additive mask (0 / -inf)."""
    if key_mask is None:
        return None
    return torch.zeros_like(key_mask, dtype=torch.float32).masked_fill_(
        ~key_mask.bool(), float("-inf")
    )[:, None, None, :]


# ----------------------- helpers -----------------------

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, freq_dim: Optional[int] = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.freq_dim = freq_dim or hidden_size
        self.mlp1 = nn.Linear(self.freq_dim, hidden_size)
        self.mlp2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, t: Tensor) -> Tensor:
        # t: (B,) float
        half = self.freq_dim // 2
        device = t.device
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=device, dtype=torch.float32) / half
        )
        args = t[:, None].float() * freqs[None, :] * 1000.0
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if emb.shape[-1] < self.freq_dim:
            emb = F.pad(emb, (0, self.freq_dim - emb.shape[-1]))
        emb = F.silu(self.mlp1(emb))
        emb = self.mlp2(emb)
        return emb  # (B, hidden_size)


# ----------------------- velocity residual head -----------------------

class _AttnBlock(nn.Module):
    """Self-attn + cross-attn + MLP block — pre-norm, residual everywhere."""
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm3 = nn.LayerNorm(hidden_size)
        hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(hidden_size, hidden), nn.GELU(), nn.Linear(hidden, hidden_size))

    def forward(self, h: Tensor, ctx_h: Tensor,
                self_kpm: Optional[Tensor] = None,
                cross_kpm: Optional[Tensor] = None) -> Tensor:
        # self_kpm / cross_kpm are key_padding_masks (B, S) with True == PAD
        x = self.norm1(h)
        sa, _ = self.self_attn(x, x, x, key_padding_mask=self_kpm, need_weights=False)
        h = h + sa
        x = self.norm2(h)
        ca, _ = self.cross_attn(x, ctx_h, ctx_h, key_padding_mask=cross_kpm, need_weights=False)
        h = h + ca
        x = self.norm3(h)
        h = h + self.mlp(x)
        return h


class VelocityResidualHead(nn.Module):
    """Small cross-attn transformer producing a velocity residual.

    Inputs:
      x        : (B, S, D_latent)         — current noisy block
      t        : (B,) float in [0, 1]
      ctx_emb  : (B, S_ctx, D_latent)     — agent's private VAE-encoded context
      x_mask   : (B, S) bool             — True for valid positions
      ctx_mask : (B, S_ctx) bool
    Returns:
      v        : (B, S, D_latent)         — velocity residual, zero-init out_proj
    """
    def __init__(self, latent_dim: int = 16, hidden_size: int = 384, depth: int = 2,
                 num_heads: int = 6, mlp_ratio: float = 2.0):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.in_proj = nn.Linear(latent_dim, hidden_size)
        self.ctx_proj = nn.Linear(latent_dim, hidden_size)
        self.t_emb = TimestepEmbedder(hidden_size)
        self.blocks = nn.ModuleList([_AttnBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)])
        self.out_proj = nn.Linear(hidden_size, latent_dim)
        # Zero-init out_proj so an untrained head contributes 0
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: Tensor, t: Tensor, ctx_emb: Tensor,
                x_mask: Optional[Tensor] = None, ctx_mask: Optional[Tensor] = None) -> Tensor:
        h = self.in_proj(x)
        ctx_h = self.ctx_proj(ctx_emb)
        h = h + self.t_emb(t)[:, None, :]
        # PyTorch MHA `key_padding_mask`: True == PAD (will be ignored)
        self_kpm = (~x_mask.bool()) if x_mask is not None else None
        cross_kpm = (~ctx_mask.bool()) if ctx_mask is not None else None
        for block in self.blocks:
            h = block(h, ctx_h, self_kpm=self_kpm, cross_kpm=cross_kpm)
        return self.out_proj(h)


# ----------------------- state-and-time gate -----------------------

class StateTimeGate(nn.Module):
    """g(z_t, t, ctx) -> scalar in [0,1]. Bias-init -2 so initial sigmoid ~ 0.12."""
    def __init__(self, latent_dim: int = 16, hidden_size: int = 256, init_logit_bias: float = -2.0):
        super().__init__()
        self.z_in = nn.Linear(latent_dim, hidden_size)
        self.ctx_in = nn.Linear(latent_dim, hidden_size)
        self.t_emb = TimestepEmbedder(hidden_size)
        self.mlp_hidden = nn.Linear(hidden_size * 3, hidden_size)
        self.out = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.constant_(self.out.bias, init_logit_bias)

    @staticmethod
    def _masked_mean(z: Tensor, m: Optional[Tensor]) -> Tensor:
        if m is None:
            return z.mean(dim=1)
        mm = m.float().unsqueeze(-1)
        return (z * mm).sum(dim=1) / mm.sum(dim=1).clamp(min=1.0)

    def forward(self, x: Tensor, t: Tensor, ctx_emb: Tensor,
                x_mask: Optional[Tensor] = None, ctx_mask: Optional[Tensor] = None) -> Tensor:
        z_pool = self._masked_mean(x, x_mask)
        ctx_pool = self._masked_mean(ctx_emb, ctx_mask)
        h = torch.cat([self.z_in(z_pool), self.ctx_in(ctx_pool), self.t_emb(t)], dim=-1)
        h = F.gelu(self.mlp_hidden(h))
        return torch.sigmoid(self.out(h)[:, 0])  # (B,)


# ----------------------- coupled MAS wrapper -----------------------

class CoupledMASHeads(nn.Module):
    """Returns (v_a, v_b, g_a, g_b). Mirrors the JAX module."""
    def __init__(self, latent_dim: int = 16,
                 head_hidden_size: int = 384, head_depth: int = 2, head_num_heads: int = 6,
                 gate_hidden_size: int = 256):
        super().__init__()
        kwargs = dict(latent_dim=latent_dim, hidden_size=head_hidden_size,
                      depth=head_depth, num_heads=head_num_heads)
        self.agent_a = VelocityResidualHead(**kwargs)
        self.agent_b = VelocityResidualHead(**kwargs)
        self.gate_a = StateTimeGate(latent_dim=latent_dim, hidden_size=gate_hidden_size)
        self.gate_b = StateTimeGate(latent_dim=latent_dim, hidden_size=gate_hidden_size)

    def forward(self, x: Tensor, t: Tensor, ctx_a_emb: Tensor, ctx_b_emb: Tensor,
                x_mask: Optional[Tensor] = None,
                ctx_a_mask: Optional[Tensor] = None,
                ctx_b_mask: Optional[Tensor] = None
                ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        v_a = self.agent_a(x, t, ctx_a_emb, x_mask=x_mask, ctx_mask=ctx_a_mask)
        v_b = self.agent_b(x, t, ctx_b_emb, x_mask=x_mask, ctx_mask=ctx_b_mask)
        g_a = self.gate_a(x, t, ctx_a_emb, x_mask=x_mask, ctx_mask=ctx_a_mask)
        g_b = self.gate_b(x, t, ctx_b_emb, x_mask=x_mask, ctx_mask=ctx_b_mask)
        return v_a, v_b, g_a, g_b


class SingleHeadMASWrapper(nn.Module):
    """B4 matched-single-model: ONE head + ONE gate over concat(ctx_a, ctx_b).
    Returns (v, 0, g, 0) to match the coupled 4-tuple — `zero_b` is a no-op cell."""
    def __init__(self, latent_dim: int = 16,
                 head_hidden_size: int = 416, head_depth: int = 4, head_num_heads: int = 8,
                 gate_hidden_size: int = 256):
        super().__init__()
        self.head = VelocityResidualHead(
            latent_dim=latent_dim, hidden_size=head_hidden_size,
            depth=head_depth, num_heads=head_num_heads,
        )
        self.gate = StateTimeGate(latent_dim=latent_dim, hidden_size=gate_hidden_size)

    def forward(self, x: Tensor, t: Tensor, ctx_a_emb: Tensor, ctx_b_emb: Tensor,
                x_mask: Optional[Tensor] = None,
                ctx_a_mask: Optional[Tensor] = None,
                ctx_b_mask: Optional[Tensor] = None
                ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        ctx_full = torch.cat([ctx_a_emb, ctx_b_emb], dim=1)
        ctx_full_mask = None
        if ctx_a_mask is not None and ctx_b_mask is not None:
            ctx_full_mask = torch.cat([ctx_a_mask, ctx_b_mask], dim=1)
        v = self.head(x, t, ctx_full, x_mask=x_mask, ctx_mask=ctx_full_mask)
        g = self.gate(x, t, ctx_full, x_mask=x_mask, ctx_mask=ctx_full_mask)
        return v, torch.zeros_like(v), g, torch.zeros_like(g)


def combine_velocities(v_0: Tensor, v_a: Tensor, v_b: Tensor, g_a: Tensor, g_b: Tensor) -> Tensor:
    """v = v_0 + g_a * v_a + g_b * v_b."""
    return v_0 + g_a[:, None, None] * v_a + g_b[:, None, None] * v_b


# ----------------------- smoke test -----------------------

def smoke():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, S, D = 4, 16, 16
    S_ctx = 32
    x = torch.randn(B, S, D, device=device)
    t = torch.linspace(0.05, 0.95, B, device=device)
    ctx_a = torch.randn(B, S_ctx, D, device=device)
    ctx_b = torch.randn(B, S_ctx, D, device=device)
    x_mask = torch.ones(B, S, dtype=torch.bool, device=device)
    ctx_a_mask = torch.ones(B, S_ctx, dtype=torch.bool, device=device)
    ctx_b_mask = torch.ones(B, S_ctx, dtype=torch.bool, device=device)

    for name, model_cls in [("coupled", CoupledMASHeads), ("single", SingleHeadMASWrapper)]:
        m = model_cls().to(device)
        n_params = sum(p.numel() for p in m.parameters())
        v_a, v_b, g_a, g_b = m(x, t, ctx_a, ctx_b, x_mask=x_mask, ctx_a_mask=ctx_a_mask, ctx_b_mask=ctx_b_mask)
        v_0 = torch.zeros_like(x)
        v = combine_velocities(v_0, v_a, v_b, g_a, g_b)
        target = torch.randn_like(x)
        loss = (v - target).pow(2).mean()
        loss.backward()
        grad_norm = sum(p.grad.pow(2).sum().item() if p.grad is not None else 0 for p in m.parameters()) ** 0.5
        print(f"[{name:8s}] params={n_params:>10,}  "
              f"v_a {tuple(v_a.shape)} max|v_a|={v_a.abs().max():.6f}  "
              f"g_a {tuple(g_a.shape)} mean={g_a.mean():.4f} "
              f"loss={loss.item():.4f}  grad_norm={grad_norm:.4f}")


if __name__ == "__main__":
    smoke()
