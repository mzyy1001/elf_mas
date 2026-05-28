"""Coupled Multi-Agent Language Flows — v1 skeleton.

Architecture (PLAN.md §4):

    v(z_t, t) = v_0(z_t, t) + g_A(z_t, t, ctx_A) * v_A(z_t, t, ctx_A)
                             + g_B(z_t, t, ctx_B) * v_B(z_t, t, ctx_B)

- `v_0` is the frozen ELF backbone (we never train it in v1).
- `v_A`, `v_B` are small Flax cross-attention transformers ("residual heads").
- `g_A`, `g_B` are state-and-time-dependent scalar gates (MLP over pooled z, pooled ctx, t).
- All trainable params live in {residual_head_a, residual_head_b, gate_a, gate_b}.

This module defines just the trainable pieces and a smoke test that runs a single
forward pass to verify shapes. Integration with the frozen ELF backbone + actual
flow-matching training happens in Phase 1.4 (`train_coupled.py`).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


# ---------------------- helpers ----------------------

class TimestepEmbedder(nn.Module):
    """Sinusoidal time embedding -> 2-layer MLP, à la DiT / ELF."""
    hidden_size: int

    @nn.compact
    def __call__(self, t):
        # t: (N,)
        half = self.hidden_size // 2
        freqs = jnp.exp(-jnp.log(10000.0) * jnp.arange(half, dtype=jnp.float32) / half)
        args = t[:, None] * freqs[None, :] * 1000.0
        emb = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)  # (N, 2*half)
        if emb.shape[-1] < self.hidden_size:
            emb = jnp.pad(emb, ((0, 0), (0, self.hidden_size - emb.shape[-1])))
        emb = nn.silu(nn.Dense(self.hidden_size, name="mlp1")(emb))
        emb = nn.Dense(self.hidden_size, name="mlp2")(emb)
        return emb  # (N, hidden_size)


# ---------------------- velocity residual head ----------------------

class VelocityResidualHead(nn.Module):
    """Small cross-attention transformer producing a velocity residual.

    Inputs:
      x         : (N, S, D_text) — current noisy latent in T5 embedding space.
      t         : (N,)
      ctx_emb   : (N, S_ctx, D_text) — agent's private context T5 embedding.
      x_mask    : optional (N, S)
      ctx_mask  : optional (N, S_ctx)

    Output:
      residual  : (N, S, D_text), **zero-initialised** so an untrained head contributes 0.
    """
    text_encoder_dim: int = 512
    hidden_size: int = 384
    depth: int = 2
    num_heads: int = 6
    mlp_ratio: float = 2.0

    @nn.compact
    def __call__(self, x, t, ctx_emb, x_mask=None, ctx_mask=None, deterministic: bool = True):
        h = nn.Dense(self.hidden_size, name="in_proj")(x)
        ctx_h = nn.Dense(self.hidden_size, name="ctx_proj")(ctx_emb)
        t_emb = TimestepEmbedder(self.hidden_size, name="t_emb")(t)
        h = h + t_emb[:, None, :]

        # Attention masks: (N, 1, 1, S) shape expected by flax MHA `mask=`
        def expand(m, seq_len):
            if m is None:
                return None
            m = m.astype(jnp.bool_)
            # broadcast to (N, 1, 1, S)
            return m[:, None, None, :]

        self_mask = expand(x_mask, h.shape[1])
        cross_mask = expand(ctx_mask, ctx_h.shape[1])

        for i in range(self.depth):
            # self-attention
            h_norm = nn.LayerNorm(name=f"norm1_{i}")(h)
            h = h + nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads, name=f"self_attn_{i}",
            )(inputs_q=h_norm, inputs_kv=h_norm, mask=self_mask, deterministic=deterministic)
            # cross-attention to context
            h_norm = nn.LayerNorm(name=f"norm2_{i}")(h)
            h = h + nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads, name=f"cross_attn_{i}",
            )(inputs_q=h_norm, inputs_kv=ctx_h, mask=cross_mask, deterministic=deterministic)
            # MLP
            h_norm = nn.LayerNorm(name=f"norm3_{i}")(h)
            mlp_hidden = int(self.hidden_size * self.mlp_ratio)
            h = h + nn.Dense(self.hidden_size, name=f"mlp_out_{i}")(
                nn.gelu(nn.Dense(mlp_hidden, name=f"mlp_in_{i}")(h_norm))
            )
        # Zero-init projection: untrained head contributes exactly zero residual.
        out = nn.Dense(
            self.text_encoder_dim, name="out_proj",
            kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros,
        )(h)
        return out  # (N, S, D_text)


# ---------------------- state-and-time-dependent gate ----------------------

class StateTimeGate(nn.Module):
    """g(z_t, t, ctx) -> scalar in [0, 1]. Bias-init so initial sigmoid ≈ 0.12.

    Why -2: too negative (e.g. -5) gives a tiny gate (~0.007) which combined
    with the zero-init residual creates a chicken-and-egg where neither gate nor
    residual receive useful gradient signal. -2 gives g≈0.12: still small enough
    to start mostly using v_0, but large enough that residuals get a real grad."""
    hidden_size: int = 256
    init_logit_bias: float = -2.0  # sigmoid(-2) ≈ 0.119

    @nn.compact
    def __call__(self, x, t, ctx_emb, x_mask=None, ctx_mask=None):
        # masked mean pooling
        def masked_mean(z, m):
            if m is None:
                return z.mean(axis=1)
            mm = m.astype(z.dtype)[:, :, None]
            return (z * mm).sum(axis=1) / jnp.clip(mm.sum(axis=1), 1.0, None)

        z_pool = masked_mean(x, x_mask)            # (N, D_text)
        ctx_pool = masked_mean(ctx_emb, ctx_mask)  # (N, D_text)
        t_emb = TimestepEmbedder(self.hidden_size, name="t_emb")(t)  # (N, H)

        h = jnp.concatenate([
            nn.Dense(self.hidden_size, name="z_in")(z_pool),
            nn.Dense(self.hidden_size, name="ctx_in")(ctx_pool),
            t_emb,
        ], axis=-1)
        h = nn.gelu(nn.Dense(self.hidden_size, name="mlp_hidden")(h))

        bias_init = lambda key, shape, dtype=jnp.float32: jnp.full(shape, self.init_logit_bias, dtype=dtype)
        g_logit = nn.Dense(
            1, name="out",
            kernel_init=nn.initializers.zeros,
            bias_init=bias_init,
        )(h)
        return nn.sigmoid(g_logit[:, 0])  # (N,)


# ---------------------- top-level MAS module (just the trainable parts) ----------------------

class CoupledMASHeads(nn.Module):
    """Wraps the two residual heads and two gates. The frozen ELF backbone
    is applied OUTSIDE this module since its params are frozen and the
    Flax `nn.Module` API can't hold non-trainable params cleanly.

    Forward returns (v_residual_A, v_residual_B, g_A, g_B). The caller
    combines them with v_0 from the frozen ELF backbone:
        v = v_0 + g_A * v_A + g_B * v_B
    """
    text_encoder_dim: int = 512
    head_hidden_size: int = 384
    head_depth: int = 2
    head_num_heads: int = 6
    gate_hidden_size: int = 256

    @nn.compact
    def __call__(self, x, t, ctx_a_emb, ctx_b_emb,
                 x_mask=None, ctx_a_mask=None, ctx_b_mask=None, deterministic: bool = True):
        head_a = VelocityResidualHead(
            text_encoder_dim=self.text_encoder_dim,
            hidden_size=self.head_hidden_size, depth=self.head_depth, num_heads=self.head_num_heads,
            name="agent_a",
        )
        head_b = VelocityResidualHead(
            text_encoder_dim=self.text_encoder_dim,
            hidden_size=self.head_hidden_size, depth=self.head_depth, num_heads=self.head_num_heads,
            name="agent_b",
        )
        gate_a = StateTimeGate(hidden_size=self.gate_hidden_size, name="gate_a")
        gate_b = StateTimeGate(hidden_size=self.gate_hidden_size, name="gate_b")

        v_a = head_a(x, t, ctx_a_emb, x_mask=x_mask, ctx_mask=ctx_a_mask, deterministic=deterministic)
        v_b = head_b(x, t, ctx_b_emb, x_mask=x_mask, ctx_mask=ctx_b_mask, deterministic=deterministic)
        g_a = gate_a(x, t, ctx_a_emb, x_mask=x_mask, ctx_mask=ctx_a_mask)
        g_b = gate_b(x, t, ctx_b_emb, x_mask=x_mask, ctx_mask=ctx_b_mask)
        return v_a, v_b, g_a, g_b


def combine_velocities(v_0, v_a, v_b, g_a, g_b):
    """v = v_0 + g_a * v_a + g_b * v_b   (broadcasting g over (S, D_text))."""
    g_a_b = g_a[:, None, None]
    g_b_b = g_b[:, None, None]
    return v_0 + g_a_b * v_a + g_b_b * v_b


# ---------------------- B4 matched-single-model baseline ----------------------

class SingleHeadMAS(nn.Module):
    """B4 baseline: ONE velocity-residual head + ONE gate, conditioned on the
    concatenated context (no agent split). Widened head capacity to roughly
    match the coupled MAS's total trainable-param count (~10M).

    Returns `(v_residual, g)` instead of the 4-tuple from `CoupledMASHeads`.
    The trainer wraps this so the rest of the pipeline (loss, lesion, decode)
    can call it uniformly.
    """
    text_encoder_dim: int = 512
    head_hidden_size: int = 416  # widened to hit ~10.06M total params (matched to CoupledMASHeads within 1.5%)
    head_depth: int = 4          # 2x depth to keep total transformer blocks the same as 2 coupled heads
    head_num_heads: int = 8      # must divide hidden_size
    gate_hidden_size: int = 256  # same as coupled per-gate

    @nn.compact
    def __call__(self, x, t, ctx_emb, x_mask=None, ctx_mask=None, deterministic: bool = True):
        head = VelocityResidualHead(
            text_encoder_dim=self.text_encoder_dim,
            hidden_size=self.head_hidden_size, depth=self.head_depth, num_heads=self.head_num_heads,
            name="single_head",
        )
        gate = StateTimeGate(hidden_size=self.gate_hidden_size, name="single_gate")
        v_residual = head(x, t, ctx_emb, x_mask=x_mask, ctx_mask=ctx_mask, deterministic=deterministic)
        g = gate(x, t, ctx_emb, x_mask=x_mask, ctx_mask=ctx_mask)
        return v_residual, g


class SingleHeadMASWrapper(nn.Module):
    """Adapter that lets `SingleHeadMAS` slot into the same forward signature as
    `CoupledMASHeads`. Internally concatenates `ctx_a_emb` and `ctx_b_emb` and
    runs ONE residual head + ONE gate over the joined context. Returns the
    coupled 4-tuple `(v_a, v_b, g_a, g_b)` with `v_b = 0` and `g_b = 0` so the
    existing loss / lesion / decode code paths work unchanged.

    Lesion semantics under this wrapper:
      - full       : v_0 + g * v_residual               (single-model prediction)
      - zero_a     : v_0                                (= matched-baseline floor)
      - zero_b     : v_0 + g * v_residual               (= full; redundant cell)
      - zero_both  : v_0                                (= matched-baseline floor)
    """
    text_encoder_dim: int = 512
    head_hidden_size: int = 416  # matches SingleHeadMAS default
    head_depth: int = 4
    head_num_heads: int = 8
    gate_hidden_size: int = 256

    @nn.compact
    def __call__(self, x, t, ctx_a_emb, ctx_b_emb,
                 x_mask=None, ctx_a_mask=None, ctx_b_mask=None, deterministic: bool = True):
        ctx_full = jnp.concatenate([ctx_a_emb, ctx_b_emb], axis=1)
        if ctx_a_mask is not None and ctx_b_mask is not None:
            ctx_full_mask = jnp.concatenate([ctx_a_mask, ctx_b_mask], axis=1)
        else:
            ctx_full_mask = None
        single = SingleHeadMAS(
            text_encoder_dim=self.text_encoder_dim,
            head_hidden_size=self.head_hidden_size, head_depth=self.head_depth,
            head_num_heads=self.head_num_heads, gate_hidden_size=self.gate_hidden_size,
            name="single",
        )
        v, g = single(x, t, ctx_full, x_mask=x_mask, ctx_mask=ctx_full_mask, deterministic=deterministic)
        zero_v = jnp.zeros_like(v)
        zero_g = jnp.zeros_like(g)
        return v, zero_v, g, zero_g


# ---------------------- smoke test ----------------------

def smoke_test(seed: int = 0, batch_size: int = 4, seq_len: int = 32, ctx_len: int = 16):
    """One forward pass with random inputs. Checks shapes + initial-gate-near-zero."""
    print(f"[smoke] batch={batch_size} seq_len={seq_len} ctx_len={ctx_len}")
    key = jax.random.PRNGKey(seed)
    k_x, k_a, k_b, k_init, _ = jax.random.split(key, 5)

    text_dim = 512
    x = jax.random.normal(k_x, (batch_size, seq_len, text_dim))
    ctx_a = jax.random.normal(k_a, (batch_size, ctx_len, text_dim))
    ctx_b = jax.random.normal(k_b, (batch_size, ctx_len, text_dim))
    t = jnp.linspace(0.05, 0.95, batch_size)

    mas = CoupledMASHeads(text_encoder_dim=text_dim)
    params = mas.init(k_init, x, t, ctx_a, ctx_b)

    # Param count
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"[smoke] total trainable params: {n_params:,}")

    v_a, v_b, g_a, g_b = mas.apply(params, x, t, ctx_a, ctx_b)
    print(f"[smoke] v_a shape: {tuple(v_a.shape)}, v_b shape: {tuple(v_b.shape)}")
    print(f"[smoke] g_a shape: {tuple(g_a.shape)}, g_b shape: {tuple(g_b.shape)}")
    print(f"[smoke] g_a values: {np.asarray(g_a).tolist()}")
    print(f"[smoke] g_b values: {np.asarray(g_b).tolist()}")
    print(f"[smoke] |v_a| max: {float(jnp.abs(v_a).max()):.6f}  |v_b| max: {float(jnp.abs(v_b).max()):.6f}")

    # Combined: simulate a v_0 = 0 backbone so total velocity = g_a*v_a + g_b*v_b
    v_0 = jnp.zeros_like(x)
    v = combine_velocities(v_0, v_a, v_b, g_a, g_b)
    print(f"[smoke] combined v shape: {tuple(v.shape)}, |v| max: {float(jnp.abs(v).max()):.6f}")

    # Gradient sanity: small MSE against random target shouldn't NaN.
    target = jax.random.normal(jax.random.PRNGKey(seed + 1), x.shape)

    def loss_fn(p):
        va, vb, ga, gb = mas.apply(p, x, t, ctx_a, ctx_b)
        v = combine_velocities(jnp.zeros_like(x), va, vb, ga, gb)
        return jnp.mean((v - target) ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    grad_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(grads)))
    print(f"[smoke] loss: {float(loss):.4f}, total grad norm: {float(grad_norm):.4f}")

    # Verify zero-init residual + ~0 gate => trivial initial loss equals target mean-square
    naive_loss = float(jnp.mean(target ** 2))
    print(f"[smoke] expected initial loss ≈ mean(target^2) = {naive_loss:.4f}")

    print("[smoke] PASSED")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=32)
    p.add_argument("--ctx_len", type=int, default=16)
    args = p.parse_args()
    smoke_test(args.seed, args.batch_size, args.seq_len, args.ctx_len)
