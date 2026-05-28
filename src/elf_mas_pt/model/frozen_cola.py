"""Frozen Cola-DLM backbone wrapper (PyTorch).

Mirrors the role `frozen_backbone.py` played for ELF: load Cola's DiT + VAE
once, never train them, expose three methods to the rest of the MAS code:

  - `encode_text(input_ids_list)` -> list of per-sample VAE latents
       (used to encode question, ctx_A, ctx_B, answer)
  - `decode_latent(z, txt_shape, txt_q_shape)` -> token IDs
       (used at eval time to produce text from a denoised latent)
  - `compute_v0(z_t, t, prefix_latent)` -> velocity at one block
       (the heart of the flow-matching step; KV cache handles prefix conditioning)

Architecture details that drove the design (Phase 3.1):
  - DiT.forward returns `ColaDiTOutput(txt_sample=...)` where `txt_sample` IS
    the per-block velocity field `v_psi(z_t^(b), t; z_0^(<b))`. No cross-attn
    in DiT blocks, conditioning enters only via (timestep -> AdaLN) and
    (prefix -> KV cache).
  - DiT works in "NA / flatten-concat" form: `txt` has shape `(L_q_total, c)`
    with per-sample lengths in `txt_shape` / `txt_q_shape`.
  - VAE encode takes a `list[Tensor]` and returns variable-length latents.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch import Tensor


@dataclass
class FrozenColaSpec:
    """Where to find each piece of Cola."""
    dit_path: str
    vae_path: str
    tokenizer_path: str
    cola_src_path: str = "/home/chenhongrui/elf_mas/cola_dlm_src"
    dtype: torch.dtype = torch.bfloat16
    pad_token_id: int = 100277
    eos_token_id: int = 100257


class FrozenColaBackbone:
    """Holds frozen Cola DiT + VAE + tokenizer; provides encode/decode/velocity."""

    def __init__(self, spec: FrozenColaSpec, dit, vae, tokenizer, device: torch.device):
        self.spec = spec
        self.dit = dit
        self.vae = vae
        self.tokenizer = tokenizer
        self.device = device
        self.block_size = int(dit.block_size)
        self.latent_dim = int(getattr(dit.config, "txt_in_channels", 16))
        # Freeze
        for p in self.dit.parameters():
            p.requires_grad_(False)
        for p in self.vae.parameters():
            p.requires_grad_(False)
        self.dit.eval()
        self.vae.eval()

    # ---------- loading ----------

    @classmethod
    def load(cls, spec: FrozenColaSpec, device: torch.device) -> "FrozenColaBackbone":
        if spec.cola_src_path not in sys.path:
            sys.path.insert(0, spec.cola_src_path)
        from cola_dlm import ColaDiTModel, ColaTextVAEModel  # type: ignore
        from tokenizers import Tokenizer

        print(f"[cola] loading DiT from {spec.dit_path}", flush=True)
        dit = ColaDiTModel.from_pretrained(spec.dit_path, torch_dtype=spec.dtype).to(device).eval()
        print(f"[cola] loading VAE from {spec.vae_path}", flush=True)
        vae = ColaTextVAEModel.from_pretrained(spec.vae_path, torch_dtype=spec.dtype).to(device).eval()
        print(f"[cola] loading tokenizer from {spec.tokenizer_path}", flush=True)
        tokenizer = Tokenizer.from_file(spec.tokenizer_path)
        n_dit = sum(p.numel() for p in dit.parameters())
        n_vae = sum(p.numel() for p in vae.parameters())
        print(f"[cola] DiT {n_dit/1e6:.1f}M, VAE {n_vae/1e6:.1f}M params (frozen)", flush=True)
        return cls(spec, dit, vae, tokenizer, device)

    # ---------- tokenization helpers ----------

    def tokenize(self, text: str) -> List[int]:
        return self.tokenizer.encode(text).ids

    def pad_to_block(self, ids: List[int], block_size: Optional[int] = None) -> Tuple[List[int], int]:
        """Pad `ids` with pad_token_id so length is a multiple of block_size. Returns
        (padded_ids, original_len)."""
        bs = block_size or self.block_size
        n = len(ids)
        rem = n % bs
        if rem == 0:
            return ids, n
        pad = bs - rem
        return ids + [self.spec.pad_token_id] * pad, n

    # ---------- VAE encode ----------

    @torch.no_grad()
    def encode_text(self, input_ids_list: List[List[int]],
                    block_align: Optional[int] = None) -> List[Tensor]:
        """Encode a batch of token-id sequences to per-sample VAE latents.

        `block_align` (default = VAE patch_size * block_size = 1) is the
        modulus the per-sample length must be padded to before encoding.
        Returns a list of (L_i, latent_dim) tensors on `self.device`.
        """
        bs = block_align or 1  # VAE patch_size * block_size = 1
        padded = []
        for ids in input_ids_list:
            ids, _ = self.pad_to_block(ids, block_size=bs)
            padded.append(torch.tensor(ids, dtype=torch.long, device=self.device))
        out = self.vae.encode(padded)
        return list(out.latents_list)

    # ---------- VAE decode ----------

    @torch.no_grad()
    def decode_latent(self, z: Tensor, txt_shape: Tensor, txt_q_shape: Tensor) -> Tensor:
        """Decode latent z to token logits via VAE.

        z: (L_q_total, latent_dim)
        txt_shape: (B, 1) per-sample K lengths
        txt_q_shape: (B, 1) per-sample Q lengths
        Returns: token logits (L_q_total, vocab_size)
        """
        return self.vae.decode(z, txt_shape=txt_shape, txt_q_shape=txt_q_shape)

    # ---------- DiT velocity (frozen v_0) ----------

    @torch.no_grad()
    def compute_v0_block(
        self,
        z_block: Tensor,            # (L_block_total, latent_dim) -- one block per sample, flat
        timestep: float | Tensor,
        txt_shape: Tensor,          # (B, 1) full K length including prefix + committed blocks + current
        txt_q_shape: Tensor,        # (B, 1) usually block_size during block gen
        update_kv: bool = False,
        use_kv_cache: bool = True,
    ) -> Tensor:
        """Run Cola DiT for one block; return the velocity (txt_sample)."""
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=z_block.device, dtype=z_block.dtype)
        out = self.dit(
            txt=z_block,
            txt_shape=txt_shape,
            txt_q_shape=txt_q_shape,
            timestep=timestep,
            update_kv=update_kv,
            use_kv_cache=use_kv_cache,
        )
        return out.txt_sample

    def clear_kv_cache(self):
        """Reset the DiT's KV cache so the next forward starts from a clean prefix."""
        for block in self.dit.blocks:
            block.set_kv_cache(False)
            block.set_kv_cache(True)  # re-enable empty cache

    def enable_kv_cache(self, on: bool = True):
        for block in self.dit.blocks:
            block.set_kv_cache(on)


# ---------- smoke test ----------

def smoke(spec: FrozenColaSpec, device_str: str = "cuda"):
    device = torch.device(device_str)
    backbone = FrozenColaBackbone.load(spec, device)
    # Encode three short texts
    texts = [
        "Berlin is in Germany.",
        "Albert Einstein was born in 1879.",
        "unknown",
    ]
    ids_list = [backbone.tokenize(t) for t in texts]
    print(f"[smoke] token lens: {[len(x) for x in ids_list]}")
    latents = backbone.encode_text(ids_list)
    for i, lat in enumerate(latents):
        print(f"[smoke] sample {i}: latent shape {tuple(lat.shape)} dtype {lat.dtype}")
    # Round-trip decode: flatten, build txt_shape, decode
    L_per = [lat.shape[0] for lat in latents]
    z_flat = torch.cat(latents, dim=0)
    txt_shape = torch.tensor([[L] for L in L_per], dtype=torch.long, device=device)
    logits = backbone.decode_latent(z_flat, txt_shape=txt_shape, txt_q_shape=txt_shape)
    print(f"[smoke] decode logits shape {tuple(logits.shape)}")
    # Argmax decode — logits is (1, L_total, V); flatten to a list of ints
    pred_ids = logits.argmax(dim=-1).squeeze(0).cpu().tolist()
    cursor = 0
    for i, n in enumerate(L_per):
        toks = pred_ids[cursor:cursor + n]
        cursor += n
        recovered = backbone.tokenizer.decode(toks)
        print(f"[smoke] sample {i} orig: {texts[i]!r}")
        print(f"[smoke] sample {i} round-trip: {recovered!r}")
    print("[smoke] DONE")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="/home/chenhongrui/elf_mas/hf_cache/elf-mas-bridge/phase3_assets/cola_dlm")
    args = p.parse_args()
    spec = FrozenColaSpec(
        dit_path=f"{args.base}/cola_dlm/cola_dit",
        vae_path=f"{args.base}/cola_dlm/cola_vae",
        tokenizer_path=f"{args.base}/tokenizer.json",
    )
    smoke(spec)
