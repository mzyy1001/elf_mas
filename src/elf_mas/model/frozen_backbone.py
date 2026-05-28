"""Wrap the frozen ELF-B-de-en backbone for use as `v_0` in the coupled MAS.

Loads:
  - the JAX T5-small encoder + its pretrained params (encoder_checkpoint pkl);
  - the ELF-B Flax model + its EMA params from an orbax checkpoint;

and exposes a single callable `compute_v0(...)` that takes
the noisy target embedding `x_t`, the condition (= question) embedding, and time `t`,
and returns the backbone's velocity / x-prediction `v_0` (restricted to target positions).

Heavy I/O lives here; the actual training step imports `FrozenELFBackbone` and never
re-loads the params per-step. `compute_v0` is jit-compiled.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import numpy as np

# ELF backbone code lives in baselines/ELF/src — we add it to sys.path lazily.
_ELF_SRC_DEFAULT = "/home/chenhongrui/elf_mas/ELF/src"


def _ensure_elf_on_path(elf_src: str | None = None):
    p = elf_src or _ELF_SRC_DEFAULT
    if p not in sys.path:
        sys.path.insert(0, p)


@dataclass
class FrozenBackboneSpec:
    """Configuration needed to rebuild ELF + T5 encoder."""
    t5_path: str               # local path or HF id
    encoder_ckpt: str          # path to encoder pkl
    elf_ckpt: str              # path to orbax checkpoint dir (e.g. .../checkpoint_0)
    model: str = "ELF-B"       # ELF_B / ELF_M / ELF_L
    max_length: int = 128
    text_encoder_dim: int = 512
    num_time_tokens: int = 4
    num_self_cond_cfg_tokens: int = 4
    num_model_mode_tokens: int = 4
    bottleneck_dim: int = 128
    self_cond_prob: float = 0.5  # determines whether x has 1x or 2x text_dim
    vocab_size: int = 32100       # T5-small's tokenizer.vocab_size (excludes the 28 special tokens)
    latent_mean: float = 0.0
    latent_std: float = 0.2
    elf_src: str | None = None


class FrozenELFBackbone:
    """Loads everything once, then is callable.

    Use:
        backbone = FrozenELFBackbone.load(spec)
        cond_emb = backbone.encode_text(question_ids, question_mask)
        v0       = backbone.compute_v0(x_t, t, cond_emb, cond_mask, target_mask)
    """

    def __init__(self, spec: FrozenBackboneSpec, encoder_apply_fn, encoder_params, elf_model, elf_params):
        self.spec = spec
        self._encoder_apply_fn = encoder_apply_fn
        self._encoder_params = encoder_params
        self._elf_model = elf_model
        self._elf_params = elf_params

    # ---------- loading ----------

    @classmethod
    def load(cls, spec: FrozenBackboneSpec) -> "FrozenELFBackbone":
        _ensure_elf_on_path(spec.elf_src)
        from modules.t5_encoder import get_encoder  # type: ignore
        from modules.model import ELF_models        # type: ignore
        from utils.checkpoint_utils import load_encoder_checkpoint, load_checkpoint  # type: ignore

        print(f"[backbone] loading T5 encoder config {spec.t5_path!r}", flush=True)
        encoder_config, encoder_model, _ = get_encoder(spec.t5_path, jnp.float32)
        if encoder_config.d_model != spec.text_encoder_dim:
            raise ValueError(f"text_encoder_dim mismatch: spec={spec.text_encoder_dim} cfg={encoder_config.d_model}")
        print(f"[backbone] loading encoder params {spec.encoder_ckpt!r}", flush=True)
        encoder_params = load_encoder_checkpoint(spec.encoder_ckpt)

        print(f"[backbone] building ELF_{spec.model[-1]} model", flush=True)
        model_cls = ELF_models[spec.model]
        elf_model = model_cls(
            text_encoder_dim=spec.text_encoder_dim,
            max_length=spec.max_length,
            attn_drop=0.0, proj_drop=0.0,
            num_time_tokens=spec.num_time_tokens,
            num_self_cond_cfg_tokens=spec.num_self_cond_cfg_tokens,
            num_model_mode_tokens=spec.num_model_mode_tokens,
            bottleneck_dim=spec.bottleneck_dim,
            vocab_size=spec.vocab_size,
        )

        # Init param tree shape with dummy inputs, then load checkpoint into the same tree.
        in_dim = 2 * spec.text_encoder_dim if spec.self_cond_prob > 0 else spec.text_encoder_dim
        dummy_x = jnp.ones((1, spec.max_length, in_dim))
        dummy_t = jnp.ones((1,))
        dummy_sc_cfg = jnp.ones((1,)) if spec.num_self_cond_cfg_tokens > 0 else None
        print(f"[backbone] init ELF param tree (dummy_x={dummy_x.shape})", flush=True)
        init_rng = jax.random.PRNGKey(0)
        # ELF init on CPU to avoid OOM on the eval GPU during init
        with jax.default_device(jax.devices("cpu")[0]):
            init_params = elf_model.init(
                init_rng,
                x=dummy_x, t=dummy_t,
                self_cond_cfg_scale=dummy_sc_cfg,
                deterministic=True,
            )
        # Build a TrainState template so we can use ELF's load_checkpoint as-is.
        from utils.train_utils import TrainState  # type: ignore
        import optax, copy
        with jax.default_device(jax.devices("cpu")[0]):
            optimizer = optax.adamw(learning_rate=1e-4)
            state = TrainState.create(
                apply_fn=elf_model.apply,
                params=init_params["params"],
                tx=optimizer,
                dropout_rng=jax.random.PRNGKey(1),
                ema_params1=copy.deepcopy(init_params["params"]),
            )
        print(f"[backbone] loading ELF checkpoint {spec.elf_ckpt!r}", flush=True)
        state, step = load_checkpoint(spec.elf_ckpt, state)
        # EMA params are what eval uses — they're the better-trained set.
        elf_params = {"params": state.ema_params1}
        print(f"[backbone] loaded checkpoint at step {step} (using ema_params1)", flush=True)
        return cls(spec, encoder_model.apply, encoder_params, elf_model, elf_params)

    # ---------- public API ----------

    @property
    def text_encoder_dim(self) -> int:
        return self.spec.text_encoder_dim

    def encode_text(self, input_ids, attention_mask):
        """Returns normalized encoder hidden states (B, S, D_text)."""
        latents = self._encoder_apply_fn(
            {"params": self._encoder_params}, input_ids=input_ids,
            attention_mask=attention_mask, deterministic=True,
        )
        if hasattr(latents, "last_hidden_state"):
            latents = latents.last_hidden_state
        elif isinstance(latents, dict):
            latents = latents["last_hidden_state"]
        return (latents - self.spec.latent_mean) / self.spec.latent_std

    def compute_v0(self, x_t, t, cond_emb, cond_mask, target_mask):
        """Apply the frozen ELF backbone with [cond_emb || x_t] as input.

        ELF was trained at fixed max_length (RoPE pre-computes for that length),
        so we pad the combined [cond, target] sequence to spec.max_length here.

        x_t       : (B, S_tgt, D_text)
        t         : (B,)
        cond_emb  : (B, S_cond, D_text)
        cond_mask : (B, S_cond)  1 for valid cond tokens
        target_mask : (B, S_tgt) 1 for valid target tokens
        Returns v_0 : (B, S_tgt, D_text)  — sliced to target positions only.
        """
        return _frozen_v0_jit(
            self._elf_model, self._elf_params,
            x_t, t, cond_emb, cond_mask, target_mask,
            self.spec.num_self_cond_cfg_tokens > 0, self.spec.self_cond_prob > 0,
            self.spec.max_length,
        )


# JITed pure function so we don't recompile on every call.
def _frozen_v0_impl(elf_model, elf_params, x_t, t, cond_emb, cond_mask, target_mask,
                    use_sc_cfg: bool, self_cond_double: bool, max_length: int):
    B, S_cond, D = cond_emb.shape
    S_tgt = x_t.shape[1]
    # If model was trained with self_cond, expects 2x dim at input. We feed zeros for the
    # x_pred channel (no self-conditioning at inference v1 — clean test of mechanism).
    x_in_target = x_t
    if self_cond_double:
        x_in_target = jnp.concatenate([x_t, jnp.zeros_like(x_t)], axis=-1)
    cond_in = cond_emb
    if self_cond_double:
        cond_in = jnp.concatenate([cond_emb, jnp.zeros_like(cond_emb)], axis=-1)
    x_combined = jnp.concatenate([cond_in, x_in_target], axis=1)  # (B, S_cond+S_tgt, ...)
    full_mask = jnp.concatenate([cond_mask, target_mask], axis=1).astype(jnp.float32)  # (B, S)

    # ELF was trained at fixed max_length; pad to that length so RoPE shapes match.
    S_used = x_combined.shape[1]
    pad_amount = max_length - S_used
    if pad_amount > 0:
        D_in = x_combined.shape[-1]
        x_combined = jnp.concatenate([
            x_combined,
            jnp.zeros((B, pad_amount, D_in), dtype=x_combined.dtype),
        ], axis=1)
        full_mask = jnp.concatenate([
            full_mask,
            jnp.zeros((B, pad_amount), dtype=full_mask.dtype),
        ], axis=1)

    sc_cfg = jnp.ones((B,)) if use_sc_cfg else None
    out, _decoder_logits = elf_model.apply(
        elf_params, x=x_combined, t=t,
        attention_mask=full_mask,
        deterministic=True,
        self_cond_cfg_scale=sc_cfg,
    )
    # `out` is (B, max_length, D_text). Slice to valid target positions.
    return out[:, S_cond:S_cond + S_tgt, :]


_frozen_v0_jit = jax.jit(_frozen_v0_impl, static_argnames=("elf_model", "use_sc_cfg", "self_cond_double", "max_length"))


# ---------- smoke test ----------

def smoke(spec: FrozenBackboneSpec, batch_size: int = 2):
    backbone = FrozenELFBackbone.load(spec)
    rng = jax.random.PRNGKey(0)
    # Tiny dummy batch
    S_cond, S_tgt = 16, 8
    ids = jax.random.randint(rng, (batch_size, S_cond), 0, spec.vocab_size)
    mask = jnp.ones((batch_size, S_cond), dtype=jnp.int32)
    cond_emb = backbone.encode_text(ids, mask)
    print(f"[backbone-smoke] cond_emb: {tuple(cond_emb.shape)}")
    x_t = jax.random.normal(rng, (batch_size, S_tgt, spec.text_encoder_dim))
    t = jnp.array([0.4, 0.7])[:batch_size]
    target_mask = jnp.ones((batch_size, S_tgt), dtype=jnp.int32)
    v0 = backbone.compute_v0(x_t, t, cond_emb, mask, target_mask)
    print(f"[backbone-smoke] v0: {tuple(v0.shape)}  mean={float(v0.mean()):.4f} std={float(v0.std()):.4f}")
    print("[backbone-smoke] PASSED")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--t5_path", default="/home/chenhongrui/elf_mas/hf_cache/elf-mas-bridge/t5-small")
    p.add_argument("--encoder_ckpt", default="/home/chenhongrui/elf_mas/hf_cache/elf-mas-bridge/t5_small_encoder_jax/t5_small_encoder_jax.pkl")
    p.add_argument("--elf_ckpt", default="/home/chenhongrui/elf_mas/hf_cache/elf-mas-bridge/ELF-B-de-en/checkpoint_0")
    p.add_argument("--batch_size", type=int, default=2)
    args = p.parse_args()
    spec = FrozenBackboneSpec(
        t5_path=args.t5_path,
        encoder_ckpt=args.encoder_ckpt,
        elf_ckpt=args.elf_ckpt,
    )
    smoke(spec, batch_size=args.batch_size)
