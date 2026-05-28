"""Train the Coupled MAS heads on top of the frozen ELF-B-de-en backbone.

v1 training step (PLAN.md §4 + notes/synthetic_task_v1.md):
  - encode question, ctx_A, ctx_B, target answer via frozen T5
  - sample t ~ logit-normal, build noisy x_t = (1 - t) * noise + t * x_target
  - v_0 = frozen ELF backbone applied with [question || x_t] (question as condition)
  - v_a, v_b, g_a, g_b = trainable CoupledMASHeads
  - v_pred = v_0 + g_a * v_a + g_b * v_b
  - loss = mean( ||v_pred - x_target||^2 ) over valid answer positions only

The flow-matching specifics match ELF's rectified-flow x-prediction (the predicted
quantity IS x_target; ELF's `add_noise` and our trainer agree on that convention).

Run a smoke pass with `--smoke` (5 steps, batch 8) before any longer training.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from datasets import load_from_disk

from elf_mas.model.coupled_mas import CoupledMASHeads, SingleHeadMASWrapper, combine_velocities
from elf_mas.model.frozen_backbone import FrozenBackboneSpec, FrozenELFBackbone


# ---------------------- batching ----------------------

def _pad_ids(ids_list, max_len, pad_id=0):
    out = np.zeros((len(ids_list), max_len), dtype=np.int32)
    mask = np.zeros((len(ids_list), max_len), dtype=np.int32)
    for i, ids in enumerate(ids_list):
        n = min(len(ids), max_len)
        out[i, :n] = ids[:n]
        mask[i, :n] = 1
    return out, mask


def collate_batch(rows, S_question, S_ctx, S_answer, variant: str = "full", rng: np.random.Generator | None = None):
    """Pad a list of rows from the HF dataset into fixed-length tensors.

    variant=
      'full'             — standard: ctx_a / ctx_b separate (default; matches Phase 1.4)
      'identical_ctx'    — B6: both agents see concat(ctx_a, ctx_b) (truncated to S_ctx).
                            Kills the private-context-per-agent inductive bias.
      'context_shuffled' — B7: ctx_a comes from row[i + shift_a], ctx_b from row[i + shift_b].
                            Q and gold answer stay aligned with row[i]. Both ctx are private,
                            but mis-aligned with the question/answer — tests whether private-
                            context segregation alone is enough to drive specialization.
    """
    q_ids, q_mask = _pad_ids([r["question_input_ids"] for r in rows], S_question)
    if variant == "identical_ctx":
        joined = [r["ctx_a_input_ids"] + r["ctx_b_input_ids"] for r in rows]
        ca_ids, ca_mask = _pad_ids(joined, S_ctx)
        cb_ids, cb_mask = ca_ids.copy(), ca_mask.copy()
    elif variant == "context_shuffled":
        n = len(rows)
        if rng is None:
            rng = np.random.default_rng()
        # Two independent random permutations, both derangements when possible:
        # row i gets ctx_a from rows[perm_a[i]] and ctx_b from rows[perm_b[i]]; neither equals i.
        def derangement(k):
            for _ in range(8):
                p = rng.permutation(k)
                if k <= 1 or all(p != np.arange(k)):
                    return p
            # fallback: simple shift
            return (np.arange(k) + 1) % k
        perm_a = derangement(n)
        perm_b = derangement(n)
        ca_ids, ca_mask = _pad_ids([rows[perm_a[i]]["ctx_a_input_ids"] for i in range(n)], S_ctx)
        cb_ids, cb_mask = _pad_ids([rows[perm_b[i]]["ctx_b_input_ids"] for i in range(n)], S_ctx)
    else:  # "full"
        ca_ids, ca_mask = _pad_ids([r["ctx_a_input_ids"] for r in rows], S_ctx)
        cb_ids, cb_mask = _pad_ids([r["ctx_b_input_ids"] for r in rows], S_ctx)
    ans_ids, ans_mask = _pad_ids([r["input_ids"] for r in rows], S_answer)
    buckets = [r["bucket"] for r in rows]
    return {
        "question_ids": q_ids, "question_mask": q_mask,
        "ctx_a_ids": ca_ids, "ctx_a_mask": ca_mask,
        "ctx_b_ids": cb_ids, "ctx_b_mask": cb_mask,
        "answer_ids": ans_ids, "answer_mask": ans_mask,
        "buckets": buckets,
        # Raw text — pass through unchanged regardless of variant, so candidate
        # extraction uses each item's true contexts even when token IDs are shuffled.
        "ctx_a_text": [r["ctx_a_text"] for r in rows],
        "ctx_b_text": [r["ctx_b_text"] for r in rows],
        "question_text": [r["question"] for r in rows],
        "answer_text": [r["answer"] for r in rows],
    }


def iterate_batches(ds, batch_size, S_question, S_ctx, S_answer, shuffle=True, seed=0, variant="full"):
    n = len(ds)
    order_rng = np.random.default_rng(seed)
    shuffle_rng = np.random.default_rng(seed + 1)  # for B7 cross-item shuffle
    order = order_rng.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n - batch_size + 1, batch_size):
        idx = order[i:i + batch_size]
        rows = ds.select(idx.tolist())
        yield collate_batch(list(rows), S_question, S_ctx, S_answer, variant=variant, rng=shuffle_rng)


# ---------------------- noise schedule (logit-normal) ----------------------

def sample_t(key, batch_size, p_mean=-1.5, p_std=0.8, eps=0.05):
    """ELF's logit-normal t-schedule (denoiser branch)."""
    z = jax.random.normal(key, (batch_size,)) * p_std + p_mean
    t = jax.nn.sigmoid(z)
    t = jnp.clip(t, eps, 1.0 - eps)
    return t


# ---------------------- training step ----------------------

@dataclass
class TrainConfig:
    S_question: int = 16
    S_ctx: int = 64
    S_answer: int = 16
    batch_size: int = 8
    lr: float = 1e-4
    p_mean: float = -1.5
    p_std: float = 0.8
    t_eps: float = 0.05
    weight_decay: float = 1e-3  # light decay on head weights to dampen runaway |v_a|, |v_b|


def _zero_residual_head_grads(grads):
    """B2 ablation: zero out residual-head grads, keep gates trainable.
    Treats any param subtree under 'agent_a' or 'agent_b' as frozen."""
    def visit(path, value):
        if any(p == "agent_a" or p == "agent_b" for p in (k.key if hasattr(k, "key") else str(k) for k in path)):
            return jnp.zeros_like(value)
        return value
    return jax.tree_util.tree_map_with_path(visit, grads)


def make_train_step(backbone: FrozenELFBackbone, mas_model: CoupledMASHeads, tcfg: TrainConfig, freeze_residuals: bool = False):
    """Build a pure-fn train_step. backbone and mas_model are static; params + opt_state move.
    If freeze_residuals=True (B2 ablation), residual-head gradients are zeroed every step."""

    def loss_fn(params, batch, t, noise):
        # Encode all four inputs via frozen T5
        q_emb = backbone.encode_text(batch["question_ids"], batch["question_mask"])
        ca_emb = backbone.encode_text(batch["ctx_a_ids"], batch["ctx_a_mask"])
        cb_emb = backbone.encode_text(batch["ctx_b_ids"], batch["ctx_b_mask"])
        ans_emb = backbone.encode_text(batch["answer_ids"], batch["answer_mask"])  # x_target

        t_b = t[:, None, None]  # (B, 1, 1)
        x_t = (1.0 - t_b) * noise + t_b * ans_emb  # rectified-flow path

        # Frozen v_0 — question is condition, target = noisy answer
        v_0 = backbone.compute_v0(
            x_t, t, q_emb,
            batch["question_mask"].astype(jnp.float32),
            batch["answer_mask"].astype(jnp.float32),
        )

        # Trainable residuals + gates
        v_a, v_b, g_a, g_b = mas_model.apply(
            params, x_t, t, ca_emb, cb_emb,
            x_mask=batch["answer_mask"], ctx_a_mask=batch["ctx_a_mask"], ctx_b_mask=batch["ctx_b_mask"],
        )
        v_pred = combine_velocities(v_0, v_a, v_b, g_a, g_b)

        # x-prediction loss: predict the clean answer embedding directly
        sqerr = (v_pred - ans_emb) ** 2  # (B, S_ans, D)
        valid = batch["answer_mask"].astype(jnp.float32)  # (B, S_ans)
        masked_sqerr = sqerr.mean(axis=-1) * valid          # (B, S_ans)
        loss = masked_sqerr.sum() / jnp.clip(valid.sum(), 1.0, None)
        # Auxiliary stats
        stats = {
            "loss": loss,
            "g_a_mean": g_a.mean(),
            "g_b_mean": g_b.mean(),
            "g_a_std": g_a.std(),
            "g_b_std": g_b.std(),
            "v_a_l2": jnp.sqrt(jnp.mean(v_a ** 2)),
            "v_b_l2": jnp.sqrt(jnp.mean(v_b ** 2)),
            "v_0_l2": jnp.sqrt(jnp.mean(v_0 ** 2)),
        }
        return loss, stats

    @jax.jit
    def step_fn(params, opt_state, batch, rng):
        rng_t, rng_n = jax.random.split(rng)
        t = sample_t(rng_t, tcfg.batch_size, tcfg.p_mean, tcfg.p_std, tcfg.t_eps)
        noise = jax.random.normal(rng_n, (tcfg.batch_size, tcfg.S_answer, backbone.text_encoder_dim))

        (loss, stats), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch, t, noise)
        if freeze_residuals:
            grads = _zero_residual_head_grads(grads)
        # AdamW update
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, stats

    optimizer = optax.adamw(learning_rate=tcfg.lr, weight_decay=tcfg.weight_decay)
    return step_fn, optimizer


def init_params(mas_model, backbone, tcfg, seed=0):
    key = jax.random.PRNGKey(seed)
    bs = tcfg.batch_size
    x = jnp.zeros((bs, tcfg.S_answer, backbone.text_encoder_dim))
    t = jnp.full((bs,), 0.5)
    ctx_a = jnp.zeros((bs, tcfg.S_ctx, backbone.text_encoder_dim))
    ctx_b = jnp.zeros((bs, tcfg.S_ctx, backbone.text_encoder_dim))
    return mas_model.init(key, x, t, ctx_a, ctx_b)


# ---------------------- main ----------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", default="/home/chenhongrui/elf_mas/data/synth_2fact_v1")
    p.add_argument("--t5_path", default="/home/chenhongrui/elf_mas/hf_cache/elf-mas-bridge/t5-small")
    p.add_argument("--encoder_ckpt", default="/home/chenhongrui/elf_mas/hf_cache/elf-mas-bridge/t5_small_encoder_jax/t5_small_encoder_jax.pkl")
    p.add_argument("--elf_ckpt", default="/home/chenhongrui/elf_mas/hf_cache/elf-mas-bridge/ELF-B-de-en/checkpoint_0")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--smoke", action="store_true", help="5 steps + verbose, no checkpointing")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval", action="store_true", help="run lesion eval at end of training")
    p.add_argument("--eval_max_batches", type=int, default=50, help="batches of eval set to use for lesion matrix")
    p.add_argument("--S_ctx", type=int, default=64, help="Max tokens per agent context")
    p.add_argument("--S_question", type=int, default=24, help="Max tokens for question")
    p.add_argument("--S_answer", type=int, default=16, help="Max tokens for answer")
    p.add_argument("--variant", default="full",
                   choices=["full", "identical_ctx", "context_shuffled", "frozen_agents", "single_model"],
                   help="Ablation: full=B0 (default), identical_ctx=B6, context_shuffled=B7, frozen_agents=B2, single_model=B4")
    p.add_argument("--decode", action="store_true",
                   help="Run nearest-neighbor decoder after lesion eval; reports EM/F1 per bucket × ablation")
    p.add_argument("--elf_decode", action="store_true",
                   help="Run ELF native decoder (ODE rollout + decoder_step_active=True); reports EM/F1")
    p.add_argument("--extractive_decode", action="store_true",
                   help="Run per-question NN decode with candidates extracted from ctx_A ∪ ctx_B (entities/numbers + 'unknown'). Reports EM/F1/recall/EM|recall.")
    p.add_argument("--decode_filter_by_type", action="store_true",
                   help="Filter candidates by question-type matching before NN scoring (numbers for 'when', proper nouns for 'who'/'where', etc.)")
    p.add_argument("--decode_question_weight", type=float, default=0.0,
                   help="Weight λ for question-conditioned scoring: score = (1-λ)*cos(v_pred, cand) + λ*cos(q_emb, cand). 0 = pure v_pred (default)")
    p.add_argument("--oracle_decode", action="store_true",
                   help="Add the gold answer to per-item candidates as an oracle sanity check (clearly separated from real setting)")
    p.add_argument("--decode_K", type=int, default=16, help="ODE steps for ELF decode")
    p.add_argument("--decode_max_batches", type=int, default=50)
    args = p.parse_args()

    if args.smoke:
        args.num_steps = 5

    tcfg = TrainConfig(batch_size=args.batch_size, lr=args.lr,
                       S_ctx=args.S_ctx, S_question=args.S_question, S_answer=args.S_answer)

    print(f"[train] loading frozen backbone")
    spec = FrozenBackboneSpec(t5_path=args.t5_path, encoder_ckpt=args.encoder_ckpt, elf_ckpt=args.elf_ckpt)
    backbone = FrozenELFBackbone.load(spec)

    print(f"[train] loading dataset {args.dataset_dir}/train")
    ds_train = load_from_disk(f"{args.dataset_dir}/train")
    print(f"[train] {len(ds_train)} train items")

    if args.variant == "single_model":
        mas_model = SingleHeadMASWrapper(text_encoder_dim=backbone.text_encoder_dim)
        print(f"[train] B4 matched-single-model: SingleHeadMASWrapper")
    else:
        mas_model = CoupledMASHeads(text_encoder_dim=backbone.text_encoder_dim)
    params = init_params(mas_model, backbone, tcfg, seed=args.seed)
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"[train] trainable params: {n_params:,}")

    freeze_residuals = (args.variant == "frozen_agents")
    print(f"[train] variant={args.variant!r} freeze_residuals={freeze_residuals}")
    step_fn, optimizer = make_train_step(backbone, mas_model, tcfg, freeze_residuals=freeze_residuals)
    opt_state = optimizer.init(params)

    rng = jax.random.PRNGKey(args.seed + 100)
    t0 = time.time()
    for step, batch in enumerate(iterate_batches(
            ds_train, tcfg.batch_size, tcfg.S_question, tcfg.S_ctx, tcfg.S_answer,
            shuffle=True, seed=args.seed, variant=args.variant)):
        if step >= args.num_steps:
            break
        # Move np arrays into jax arrays (no .device_put needed; jit will handle it)
        batch_jax = {k: jnp.asarray(v) for k, v in batch.items() if isinstance(v, np.ndarray)}
        rng, step_rng = jax.random.split(rng)
        params, opt_state, stats = step_fn(params, opt_state, batch_jax, step_rng)
        if step < 3 or step % 50 == 0 or step == args.num_steps - 1:
            elapsed = time.time() - t0
            stat_str = " ".join(f"{k}={float(v):.4f}" for k, v in stats.items())
            print(f"[train] step={step:5d}  {stat_str}  ({elapsed:.1f}s elapsed)")
    print(f"[train] done; total {time.time() - t0:.1f}s for {args.num_steps} steps")

    # ============================================
    # Lesion eval: bucket × ablation loss matrix
    # ============================================
    if args.eval:
        eval_lesion(backbone, mas_model, params, tcfg, args)
    if args.decode:
        eval_decode(backbone, mas_model, params, tcfg, args)
    if args.elf_decode:
        eval_elf_decode(backbone, mas_model, params, tcfg, args)
    if args.extractive_decode:
        eval_extractive_decode(backbone, mas_model, params, tcfg, args, oracle=args.oracle_decode)


def eval_lesion(backbone, mas_model, params, tcfg, args):
    """Compute per-bucket loss under 4 ablation variants:
      "full"       : v = v_0 + g_a v_a + g_b v_b
      "zero_a"     : v = v_0           + g_b v_b
      "zero_b"     : v = v_0 + g_a v_a
      "zero_both"  : v = v_0                          (= just frozen backbone)
    For each bucket × variant, report mean MSE on eval set.
    """
    eval_split = "dev" if (Path(args.dataset_dir) / "dev").exists() else "eval"
    print(f"[eval] loading {args.dataset_dir}/{eval_split}")
    ds_eval = load_from_disk(f"{args.dataset_dir}/{eval_split}")
    print(f"[eval] {len(ds_eval)} eval items")

    def loss_with_ablation(p, batch, t, noise, mask_a, mask_b):
        q_emb = backbone.encode_text(batch["question_ids"], batch["question_mask"])
        ca_emb = backbone.encode_text(batch["ctx_a_ids"], batch["ctx_a_mask"])
        cb_emb = backbone.encode_text(batch["ctx_b_ids"], batch["ctx_b_mask"])
        ans_emb = backbone.encode_text(batch["answer_ids"], batch["answer_mask"])
        t_b = t[:, None, None]
        x_t = (1.0 - t_b) * noise + t_b * ans_emb
        v_0 = backbone.compute_v0(
            x_t, t, q_emb,
            batch["question_mask"].astype(jnp.float32),
            batch["answer_mask"].astype(jnp.float32),
        )
        v_a, v_b, g_a, g_b = mas_model.apply(
            p, x_t, t, ca_emb, cb_emb,
            x_mask=batch["answer_mask"], ctx_a_mask=batch["ctx_a_mask"], ctx_b_mask=batch["ctx_b_mask"],
        )
        # ablation masks scalar 0/1; broadcast to gate
        v_pred = v_0 + (mask_a * g_a)[:, None, None] * v_a + (mask_b * g_b)[:, None, None] * v_b
        sqerr = (v_pred - ans_emb) ** 2
        valid = batch["answer_mask"].astype(jnp.float32)
        per_item = (sqerr.mean(axis=-1) * valid).sum(axis=1) / jnp.clip(valid.sum(axis=1), 1.0, None)
        return per_item  # (B,) per-item losses

    eval_step = jax.jit(loss_with_ablation, static_argnums=())

    rng = jax.random.PRNGKey(args.seed + 1000)
    bucket_losses = {
        bucket: {"full": [], "zero_a": [], "zero_b": [], "zero_both": []}
        for bucket in ("AB", "A-only", "B-only", "neither")
    }
    n_batches = 0
    for batch in iterate_batches(
            ds_eval, tcfg.batch_size, tcfg.S_question, tcfg.S_ctx, tcfg.S_answer,
            shuffle=False, seed=args.seed, variant=args.variant):
        rng, rng_t, rng_n = jax.random.split(rng, 3)
        t = sample_t(rng_t, tcfg.batch_size, tcfg.p_mean, tcfg.p_std, tcfg.t_eps)
        noise = jax.random.normal(rng_n, (tcfg.batch_size, tcfg.S_answer, backbone.text_encoder_dim))
        batch_jax = {k: jnp.asarray(v) for k, v in batch.items() if isinstance(v, np.ndarray)}
        for name, (mask_a, mask_b) in [("full", (1.0, 1.0)), ("zero_a", (0.0, 1.0)),
                                       ("zero_b", (1.0, 0.0)), ("zero_both", (0.0, 0.0))]:
            per_item = eval_step(params, batch_jax, t, noise, mask_a, mask_b)
            per_item = np.asarray(per_item)
            for i, bucket in enumerate(batch["buckets"]):
                bucket_losses[bucket][name].append(float(per_item[i]))
        n_batches += 1
        if n_batches >= args.eval_max_batches:
            break

    print(f"\n[eval] LESION MATRIX (mean MSE; n_batches={n_batches}, bs={tcfg.batch_size})")
    print(f"{'bucket':>10s}  " + "  ".join(f"{n:>10s}" for n in ("full", "zero_a", "zero_b", "zero_both", "Δa=full-zero_a", "Δb=full-zero_b")))
    for bucket in ("AB", "A-only", "B-only", "neither"):
        full_m = np.mean(bucket_losses[bucket]["full"]) if bucket_losses[bucket]["full"] else float("nan")
        za_m = np.mean(bucket_losses[bucket]["zero_a"]) if bucket_losses[bucket]["zero_a"] else float("nan")
        zb_m = np.mean(bucket_losses[bucket]["zero_b"]) if bucket_losses[bucket]["zero_b"] else float("nan")
        zboth_m = np.mean(bucket_losses[bucket]["zero_both"]) if bucket_losses[bucket]["zero_both"] else float("nan")
        n_items = len(bucket_losses[bucket]["full"])
        print(f"{bucket:>10s}  {full_m:10.4f}  {za_m:10.4f}  {zb_m:10.4f}  {zboth_m:10.4f}  {za_m-full_m:14.4f}  {zb_m-full_m:14.4f}    (n={n_items})")
    print("[eval] Δa large on AB & A-only, small on B-only & neither  ==> agent-A is role-specific.")
    print("[eval] Δb large on AB & B-only, small on A-only & neither  ==> agent-B is role-specific.")


# ---------------------- nearest-neighbor decoder (EM/F1) ----------------------

_TOKEN_RE = __import__("re").compile(r"\w+", __import__("re").UNICODE)


def _tok(s):
    return _TOKEN_RE.findall(s.lower())


def _squad_f1(pred, gold):
    pt, gt = _tok(pred), _tok(gold)
    if not pt and not gt: return 1.0
    if not pt or not gt: return 0.0
    from collections import Counter
    common = Counter(pt) & Counter(gt)
    n_same = sum(common.values())
    if n_same == 0: return 0.0
    p = n_same / len(pt); r = n_same / len(gt)
    return 2 * p * r / (p + r)


def eval_decode(backbone, mas_model, params, tcfg, args):
    """Nearest-neighbor decode (PLAN.md §7 Phase 2.1 option 1).

    1. Build candidate set = unique answers in train (+ 'unknown').
    2. Encode each candidate via frozen T5, mean-pool -> matrix C in R^(K, D_text).
    3. For each eval item, run a 1-step forward at t≈0 with x_t = noise:
         v_pred = v_0(noise, t=0, question) + g_a * v_a + g_b * v_b
       pool v_pred over valid answer positions -> vector p.
    4. Cosine-sim(p, C), argmax -> predicted answer string.
    5. Report mean EM and F1 per bucket × ablation.
    """
    from datasets import load_from_disk
    print(f"[decode] building candidate set from {args.dataset_dir}/train")
    ds_train = load_from_disk(f"{args.dataset_dir}/train")
    answers = sorted({a for a in ds_train["answer"]} | {"unknown"})
    print(f"[decode] {len(answers)} unique candidate answers")

    # Tokenize + encode candidates in batches
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.t5_path, legacy=False)
    cand_ids, cand_mask = _pad_ids([tok(a, add_special_tokens=False)["input_ids"] for a in answers], tcfg.S_answer)

    @jax.jit
    def encode_cands(ids, mask):
        emb = backbone.encode_text(ids, mask)  # (K, S_ans, D)
        m = mask.astype(emb.dtype)[:, :, None]
        return (emb * m).sum(axis=1) / jnp.clip(m.sum(axis=1), 1.0, None)  # (K, D)

    # Encode candidates in chunks to avoid OOM on large vocab
    K = len(answers)
    chunk = 512
    C = []
    for i in range(0, K, chunk):
        C.append(encode_cands(jnp.asarray(cand_ids[i:i + chunk]), jnp.asarray(cand_mask[i:i + chunk])))
    C = jnp.concatenate(C, axis=0)  # (K, D)
    C_norm = C / jnp.clip(jnp.linalg.norm(C, axis=-1, keepdims=True), 1e-9, None)

    @jax.jit
    def predict_pooled(p, batch, t, noise, mask_a, mask_b):
        q_emb = backbone.encode_text(batch["question_ids"], batch["question_mask"])
        ca_emb = backbone.encode_text(batch["ctx_a_ids"], batch["ctx_a_mask"])
        cb_emb = backbone.encode_text(batch["ctx_b_ids"], batch["ctx_b_mask"])
        # x_t = noise (t=0 baseline)
        v_0 = backbone.compute_v0(
            noise, t, q_emb,
            batch["question_mask"].astype(jnp.float32),
            batch["answer_mask"].astype(jnp.float32),
        )
        v_a, v_b, g_a, g_b = mas_model.apply(
            p, noise, t, ca_emb, cb_emb,
            x_mask=batch["answer_mask"], ctx_a_mask=batch["ctx_a_mask"], ctx_b_mask=batch["ctx_b_mask"],
        )
        v_pred = v_0 + (mask_a * g_a)[:, None, None] * v_a + (mask_b * g_b)[:, None, None] * v_b
        # mean-pool over valid answer positions
        m = batch["answer_mask"].astype(v_pred.dtype)[:, :, None]
        pooled = (v_pred * m).sum(axis=1) / jnp.clip(m.sum(axis=1), 1.0, None)  # (B, D)
        # cosine sim against C
        pooled_n = pooled / jnp.clip(jnp.linalg.norm(pooled, axis=-1, keepdims=True), 1e-9, None)
        sims = pooled_n @ C_norm.T  # (B, K)
        return jnp.argmax(sims, axis=-1)  # (B,)

    print(f"[decode] loading {args.dataset_dir}/dev")
    ds_eval = load_from_disk(f"{args.dataset_dir}/dev" if (Path(args.dataset_dir) / "dev").exists() else f"{args.dataset_dir}/eval")
    print(f"[decode] {len(ds_eval)} eval items")

    rng = jax.random.PRNGKey(args.seed + 2000)
    bucket_em = {b: {n: [] for n in ("full","zero_a","zero_b","zero_both")} for b in ("AB","A-only","B-only","neither")}
    bucket_f1 = {b: {n: [] for n in ("full","zero_a","zero_b","zero_both")} for b in ("AB","A-only","B-only","neither")}
    answer_arr = np.array(answers)
    n_batches = 0
    for batch in iterate_batches(ds_eval, tcfg.batch_size, tcfg.S_question, tcfg.S_ctx, tcfg.S_answer,
                                 shuffle=False, seed=args.seed, variant=args.variant):
        rng, rng_n = jax.random.split(rng)
        # use a fixed t=0.02 (not exactly 0 to avoid degenerate cases), x_t = pure noise
        t = jnp.full((tcfg.batch_size,), 0.02)
        noise = jax.random.normal(rng_n, (tcfg.batch_size, tcfg.S_answer, backbone.text_encoder_dim))
        batch_jax = {k: jnp.asarray(v) for k, v in batch.items() if isinstance(v, np.ndarray)}
        gold = batch["buckets"], [batch["answer_ids"][i] for i in range(tcfg.batch_size)]
        # decode gold strings
        gold_strs = [tok.decode(batch["answer_ids"][i][:int(batch["answer_mask"][i].sum())], skip_special_tokens=True).strip()
                     for i in range(tcfg.batch_size)]
        for name, (mask_a, mask_b) in [("full", (1.0, 1.0)), ("zero_a", (0.0, 1.0)),
                                       ("zero_b", (1.0, 0.0)), ("zero_both", (0.0, 0.0))]:
            pred_idx = np.asarray(predict_pooled(params, batch_jax, t, noise, mask_a, mask_b))
            preds = answer_arr[pred_idx]
            for i, bucket in enumerate(batch["buckets"]):
                em = 1.0 if preds[i].strip().lower() == gold_strs[i].strip().lower() else 0.0
                f1 = _squad_f1(preds[i], gold_strs[i])
                bucket_em[bucket][name].append(em)
                bucket_f1[bucket][name].append(f1)
        n_batches += 1
        if n_batches >= args.decode_max_batches:
            break

    print(f"\n[decode] NEAREST-NEIGHBOR EM/F1 (n_batches={n_batches}, bs={tcfg.batch_size})")
    print(f"{'bucket':>10s}  {'full EM/F1':>14s}  {'zero_a':>14s}  {'zero_b':>14s}  {'zero_both':>14s}")
    for bucket in ("AB","A-only","B-only","neither"):
        cells = []
        for name in ("full","zero_a","zero_b","zero_both"):
            ems = bucket_em[bucket][name]
            f1s = bucket_f1[bucket][name]
            if ems:
                cells.append(f"{100*np.mean(ems):5.1f}/{100*np.mean(f1s):5.1f}")
            else:
                cells.append("    —")
        n_items = len(bucket_em[bucket]["full"])
        print(f"{bucket:>10s}  " + "  ".join(f"{c:>14s}" for c in cells) + f"  (n={n_items})")


# ---------------------- ELF native decode (ODE rollout + decoder_step_active) ----------------------

def _build_decode_step(backbone, mas_model, S_q, S_ans, max_length, num_K, self_cond_double, num_sc_cfg):
    """Build a jit-compiled ODE rollout + ELF decode step that returns token IDs.

    Mirrors ELF's `_dlm_decode_batch` (generation_utils.py) at the final step but
    drives the trajectory with our coupled velocity instead of v_0 alone.
    """
    elf_model = backbone._elf_model
    elf_params = backbone._elf_params

    @jax.jit
    def decode_step(params, batch, noise, mask_a, mask_b):
        q_emb = backbone.encode_text(batch["question_ids"], batch["question_mask"])
        ca_emb = backbone.encode_text(batch["ctx_a_ids"], batch["ctx_a_mask"])
        cb_emb = backbone.encode_text(batch["ctx_b_ids"], batch["ctx_b_mask"])

        z = noise  # (B, S_ans, D)
        t_grid = jnp.linspace(0.05, 0.95, num_K + 1)

        for i in range(num_K):
            t_now = t_grid[i]
            t_b = jnp.full((z.shape[0],), t_now)
            v_0 = backbone.compute_v0(
                z, t_b, q_emb,
                batch["question_mask"].astype(jnp.float32),
                batch["answer_mask"].astype(jnp.float32),
            )
            v_a, v_b, g_a, g_b = mas_model.apply(
                params, z, t_b, ca_emb, cb_emb,
                x_mask=batch["answer_mask"], ctx_a_mask=batch["ctx_a_mask"], ctx_b_mask=batch["ctx_b_mask"],
            )
            x_pred = v_0 + (mask_a * g_a)[:, None, None] * v_a + (mask_b * g_b)[:, None, None] * v_b
            dt = t_grid[i + 1] - t_grid[i]
            z = z + dt * (x_pred - z) / jnp.clip(1.0 - t_now, 1e-6, None)

        z_double = jnp.concatenate([z, jnp.zeros_like(z)], axis=-1) if self_cond_double else z
        q_double = jnp.concatenate([q_emb, jnp.zeros_like(q_emb)], axis=-1) if self_cond_double else q_emb
        x_combined = jnp.concatenate([q_double, z_double], axis=1)
        full_mask = jnp.concatenate([
            batch["question_mask"].astype(jnp.float32),
            batch["answer_mask"].astype(jnp.float32),
        ], axis=1)
        S_used = x_combined.shape[1]
        if S_used < max_length:
            pad_amt = max_length - S_used
            D_in = x_combined.shape[-1]
            x_combined = jnp.concatenate([x_combined, jnp.zeros((x_combined.shape[0], pad_amt, D_in))], axis=1)
            full_mask = jnp.concatenate([full_mask, jnp.zeros((x_combined.shape[0], pad_amt))], axis=1)

        t_final = jnp.ones((z.shape[0],))
        sc_cfg = jnp.ones((z.shape[0],)) if num_sc_cfg > 0 else None
        _, dec_logits = elf_model.apply(
            elf_params, x_combined, t_final,
            attention_mask=full_mask, deterministic=True,
            self_cond_cfg_scale=sc_cfg, decoder_step_active=jnp.array(True),
        )
        return jnp.argmax(dec_logits[:, S_q:S_q + S_ans, :], axis=-1)

    return decode_step


def eval_elf_decode(backbone, mas_model, params, tcfg, args):
    """ELF native decode (ODE rollout in latent space + decoder_step_active=True at t=1).
    Reports per-bucket × ablation EM and F1."""
    from datasets import load_from_disk
    from transformers import AutoTokenizer
    eval_split = "dev" if (Path(args.dataset_dir) / "dev").exists() else "eval"
    print(f"[elf-decode] loading {args.dataset_dir}/{eval_split}")
    ds_eval = load_from_disk(f"{args.dataset_dir}/{eval_split}")
    print(f"[elf-decode] {len(ds_eval)} eval items, K={args.decode_K} ODE steps")

    tok = AutoTokenizer.from_pretrained(args.t5_path, legacy=False)
    decode_step = _build_decode_step(
        backbone, mas_model, tcfg.S_question, tcfg.S_answer,
        backbone.spec.max_length, args.decode_K,
        backbone.spec.self_cond_prob > 0, backbone.spec.num_self_cond_cfg_tokens,
    )

    rng = jax.random.PRNGKey(args.seed + 3000)
    bucket_em = {b: {n: [] for n in ("full","zero_a","zero_b","zero_both")} for b in ("AB","A-only","B-only","neither")}
    bucket_f1 = {b: {n: [] for n in ("full","zero_a","zero_b","zero_both")} for b in ("AB","A-only","B-only","neither")}
    n_batches = 0
    for batch in iterate_batches(ds_eval, tcfg.batch_size, tcfg.S_question, tcfg.S_ctx, tcfg.S_answer,
                                 shuffle=False, seed=args.seed, variant=args.variant):
        rng, rng_n = jax.random.split(rng)
        noise = jax.random.normal(rng_n, (tcfg.batch_size, tcfg.S_answer, backbone.text_encoder_dim))
        batch_jax = {k: jnp.asarray(v) for k, v in batch.items() if isinstance(v, np.ndarray)}
        gold_strs = [tok.decode(batch["answer_ids"][i][:int(batch["answer_mask"][i].sum())],
                                skip_special_tokens=True).strip()
                     for i in range(tcfg.batch_size)]
        for name, (mask_a, mask_b) in [("full", (1.0, 1.0)), ("zero_a", (0.0, 1.0)),
                                       ("zero_b", (1.0, 0.0)), ("zero_both", (0.0, 0.0))]:
            pred_ids = np.asarray(decode_step(params, batch_jax, noise, mask_a, mask_b))
            preds = [tok.decode(row, skip_special_tokens=True).strip() for row in pred_ids]
            for i, bucket in enumerate(batch["buckets"]):
                em = 1.0 if preds[i].strip().lower() == gold_strs[i].strip().lower() else 0.0
                f1 = _squad_f1(preds[i], gold_strs[i])
                bucket_em[bucket][name].append(em)
                bucket_f1[bucket][name].append(f1)
        n_batches += 1
        if n_batches >= args.decode_max_batches:
            break

    print(f"\n[elf-decode] ELF-NATIVE DECODE EM/F1 (n_batches={n_batches}, bs={tcfg.batch_size}, K={args.decode_K})")
    print(f"{'bucket':>10s}  {'full EM/F1':>14s}  {'zero_a':>14s}  {'zero_b':>14s}  {'zero_both':>14s}")
    for bucket in ("AB","A-only","B-only","neither"):
        cells = []
        for name in ("full","zero_a","zero_b","zero_both"):
            ems = bucket_em[bucket][name]
            f1s = bucket_f1[bucket][name]
            if ems:
                cells.append(f"{100*np.mean(ems):5.1f}/{100*np.mean(f1s):5.1f}")
            else:
                cells.append("    —")
        n_items = len(bucket_em[bucket]["full"])
        print(f"{bucket:>10s}  " + "  ".join(f"{c:>14s}" for c in cells) + f"  (n={n_items})")


# ---------------------- per-question extractive NN decode ----------------------

import re as _re

_NUM_RE = _re.compile(r"\b\d+(?:[,\.]\d+)*\b")
# Proper noun phrases: 1-7 consecutive capitalized words (allows apostrophe, hyphen, "of", "von", "de", unicode)
# Includes a single-word lowercase connector (of/von/de/la/the) so names like "Gotthold von X" or "Princess of Wales"
# don't get cut in the middle.
_PROPER_RE = _re.compile(
    r"\b(?:[A-Z][\w'\-]*"
    r"(?:\s+(?:of|von|de|der|la|le|du|the)\s+[A-Z][\w'\-]*|"
    r"\s+[A-Z][\w'\-]*){0,6})\b",
    _re.UNICODE,
)
# Lowercase noun-like phrases (1-3 lowercase words) — occupations, hobbies, etc.
_LC_NP_RE = _re.compile(r"\b[a-z]{3,}(?:\s+[a-z]{3,}){0,2}\b")
# Token n-grams of any case (length 1-3) — recall safety net
_NGRAM_TOKEN_RE = _re.compile(r"\b[\w'\-]+(?:\s+[\w'\-]+){0,2}\b")

_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "in", "on", "at", "to", "for", "with",
    "is", "are", "was", "were", "be", "been", "being", "as", "by", "from", "that",
    "this", "these", "those", "it", "its", "his", "her", "their", "what", "which",
    "who", "whom", "whose", "when", "where", "why", "how",
}


def _normalize_answer(s: str) -> str:
    """SQuAD-style normalization: lowercase, strip punctuation, strip articles, collapse whitespace."""
    s = s.lower().strip()
    s = _re.sub(r"\b(a|an|the)\b", " ", s)
    s = _re.sub(r"[^\w\s]", " ", s)
    s = _re.sub(r"\s+", " ", s).strip()
    return s


def _em_norm(pred: str, gold: str) -> float:
    return 1.0 if _normalize_answer(pred) == _normalize_answer(gold) else 0.0


# ---------------------- decoder type heuristics ----------------------

_Q_DATE_RE = _re.compile(r"\b(when|year|date|century|decade|founded|abolished|created|born|died|launched|established|started|ended|finished|invented|published|released|opened|closed|formed|dissolved|elected|granted|signed|introduced)\b", _re.IGNORECASE)
_Q_NUM_RE = _re.compile(r"\b(how many|how much|number of|count|population|height|length|distance|amount|percentage|percent|figures)\b", _re.IGNORECASE)
_Q_PERSON_RE = _re.compile(r"\b(who|whom|whose)\b|>>\s*(author|director|composer|writer|artist|founder|president|leader|spouse|husband|wife|father|mother|son|daughter|sibling|brother|sister|child|children|owner|inventor|discoverer|painter|musician|actor|actress|king|queen|prince|princess|chief|head|sponsor|performer|singer|architect|designer|developer|publisher|editor|narrator|host|coach|manager|chairman|ceo|founder)", _re.IGNORECASE)
_Q_LOC_RE = _re.compile(r"\b(where|country|city|state|province|county|continent|nation|capital|region|territory|district|town|village|island|river|mountain|lake)\b|>>\s*(country|city|location|place|capital|located|birthplace|deathplace|origin|hometown|residence|administrative|headquarters|country of citizenship|place of)", _re.IGNORECASE)


def question_types(q: str) -> set:
    """Return a set of acceptable answer types for question `q`."""
    types = set()
    if _Q_DATE_RE.search(q):
        types.add("date_or_num")
    if _Q_NUM_RE.search(q):
        types.add("date_or_num")
    if _Q_PERSON_RE.search(q):
        types.add("proper_phrase")
    if _Q_LOC_RE.search(q):
        types.add("proper_phrase")
    if not types:
        types = {"any"}
    types.add("unknown_token")  # always allow the abstain candidate
    return types


def candidate_type(c: str) -> str:
    """Classify a candidate into one of {'date_or_num', 'proper_phrase', 'proper_word', 'other', 'unknown_token'}."""
    if c.strip().lower() == "unknown":
        return "unknown_token"
    if _re.search(r"\b(19|20)\d{2}\b", c) or _re.fullmatch(r"\d[\d,\.\-]*", c.strip()):
        return "date_or_num"
    if any(ch.isupper() for ch in c) and " " in c:
        return "proper_phrase"
    if c and c[0].isupper():
        return "proper_word"
    return "other"


def filter_candidates_by_type(cands: List[str], q: str, min_keep: int = 5) -> List[str]:
    """Filter candidates to those whose type matches the question.
    Falls back to the original list if the filtered set is too small."""
    qt = question_types(q)
    if "any" in qt:
        return cands
    # Accept date_or_num if question asks for date/num; proper_phrase if asks for who/where.
    # Always allow unknown_token.
    keep = [c for c in cands if candidate_type(c) in qt]
    if len(keep) < min_keep:
        return cands  # fallback to avoid hurting recall
    return keep


def extract_candidates(text: str, max_per_text: int = 80) -> List[str]:
    """Return a list of unique span candidates from `text`.

    Coverage:
      - numbers / years (NUM_RE)
      - proper noun phrases of length 1..7 with allowed connectors (of/von/de/...)
      - lowercase noun-like phrases of length 1..3
      - sub-spans of long proper-noun phrases (so "John Lennon" is in addition to "John Winston Ono Lennon")
    """
    if not text:
        return []
    cands = set()
    for m in _NUM_RE.findall(text):
        cands.add(m)
    proper_hits = []
    for m in _PROPER_RE.findall(text):
        c = m.strip()
        if not c or len(c.split()) > 7:
            continue
        cands.add(c)
        proper_hits.append(c)
    # Sub-spans of multi-word proper nouns (recall booster): include each suffix/prefix of 2+ tokens
    for c in proper_hits:
        toks = c.split()
        if len(toks) >= 3:
            for L in range(2, len(toks)):
                # prefix
                cands.add(" ".join(toks[:L]))
                # suffix
                cands.add(" ".join(toks[len(toks) - L:]))
    for m in _LC_NP_RE.findall(text):
        c = m.strip()
        if c and c not in _STOPWORDS and len(c) >= 3 and len(c.split()) <= 3:
            cands.add(c)
    # If we still have very few, fall back to general token n-grams (lower precision, higher recall)
    if len(cands) < 20:
        for m in _NGRAM_TOKEN_RE.findall(text):
            c = m.strip()
            if c and c.lower() not in _STOPWORDS and 2 <= len(c) and len(c.split()) <= 3:
                cands.add(c)
            if len(cands) >= max_per_text:
                break
    # Cap to keep batch encoding tractable
    if len(cands) > max_per_text:
        # prefer longer candidates (more informative)
        cands = set(sorted(cands, key=lambda x: (-len(x.split()), x))[:max_per_text])
    return sorted(cands)


def _build_predict_pooled(backbone, mas_model, S_ans):
    """Jit-compiled forward that returns pooled v_pred (1-step at t=0.02, noise input)."""
    @jax.jit
    def fn(params, batch, t, noise, mask_a, mask_b):
        q_emb = backbone.encode_text(batch["question_ids"], batch["question_mask"])
        ca_emb = backbone.encode_text(batch["ctx_a_ids"], batch["ctx_a_mask"])
        cb_emb = backbone.encode_text(batch["ctx_b_ids"], batch["ctx_b_mask"])
        v_0 = backbone.compute_v0(
            noise, t, q_emb,
            batch["question_mask"].astype(jnp.float32),
            batch["answer_mask"].astype(jnp.float32),
        )
        v_a, v_b, g_a, g_b = mas_model.apply(
            params, noise, t, ca_emb, cb_emb,
            x_mask=batch["answer_mask"], ctx_a_mask=batch["ctx_a_mask"], ctx_b_mask=batch["ctx_b_mask"],
        )
        v_pred = v_0 + (mask_a * g_a)[:, None, None] * v_a + (mask_b * g_b)[:, None, None] * v_b
        m = batch["answer_mask"].astype(v_pred.dtype)[:, :, None]
        return (v_pred * m).sum(axis=1) / jnp.clip(m.sum(axis=1), 1.0, None)  # (B, D)

    return fn


def _build_cand_encoder(backbone):
    """Mean-pool encoder for a chunk of candidate tokenizations."""
    @jax.jit
    def fn(ids, mask):
        emb = backbone.encode_text(ids, mask)
        m = mask.astype(emb.dtype)[:, :, None]
        return (emb * m).sum(axis=1) / jnp.clip(m.sum(axis=1), 1.0, None)  # (K, D)

    return fn


def eval_extractive_decode(backbone, mas_model, params, tcfg, args, oracle: bool = False):
    """Per-question NN decode against candidates extracted from ctx_A ∪ ctx_B.

    Reports per-bucket × ablation: EM, F1, gold-candidate-recall, EM|gold-in-candidates.
    If oracle=True, the gold answer is added to candidates (sanity-check upper bound).
    """
    from datasets import load_from_disk
    from transformers import AutoTokenizer
    eval_split = "dev" if (Path(args.dataset_dir) / "dev").exists() else "eval"
    label = "ORACLE" if oracle else "EXTRACTIVE"
    print(f"[{label.lower()}-decode] loading {args.dataset_dir}/{eval_split}")
    ds_eval = load_from_disk(f"{args.dataset_dir}/{eval_split}")
    print(f"[{label.lower()}-decode] {len(ds_eval)} eval items")

    tok = AutoTokenizer.from_pretrained(args.t5_path, legacy=False)
    predict_pooled = _build_predict_pooled(backbone, mas_model, tcfg.S_answer)
    encode_cands = _build_cand_encoder(backbone)

    rng = jax.random.PRNGKey(args.seed + 4000)
    buckets = ("AB", "A-only", "B-only", "neither")
    ablations = ("full", "zero_a", "zero_b", "zero_both")
    bucket_em = {b: {a: [] for a in ablations} for b in buckets}
    bucket_f1 = {b: {a: [] for a in ablations} for b in buckets}
    bucket_recall = {b: [] for b in buckets}  # gold-in-candidates per item
    bucket_em_cond_recall = {b: {a: [] for a in ablations} for b in buckets}
    bucket_cand_sizes = {b: [] for b in buckets}
    filter_by_type = getattr(args, "decode_filter_by_type", False)
    q_weight = float(getattr(args, "decode_question_weight", 0.0))

    examples_correct = []
    examples_wrong = []

    n_batches = 0
    for batch in iterate_batches(ds_eval, tcfg.batch_size, tcfg.S_question, tcfg.S_ctx, tcfg.S_answer,
                                 shuffle=False, seed=args.seed, variant=args.variant):
        # Build per-item candidates from raw text
        ctx_a_text = batch.get("ctx_a_text") if "ctx_a_text" in batch else None
        ctx_b_text = batch.get("ctx_b_text") if "ctx_b_text" in batch else None
        # If the collator didn't carry the text, fetch from dataset
        if ctx_a_text is None:
            raise RuntimeError("collate_batch did not include text fields; reload may be needed.")

        per_item_cands: List[List[str]] = []
        per_item_gold: List[str] = []
        per_item_question: List[str] = []
        for i in range(tcfg.batch_size):
            cands = set(extract_candidates(ctx_a_text[i])) | set(extract_candidates(ctx_b_text[i]))
            cands.add("unknown")
            gold = batch["answer_text"][i] if "answer_text" in batch else ""
            q = batch["question_text"][i] if "question_text" in batch else ""
            if oracle and gold:
                cands.add(gold)
            cand_list = sorted(cands)
            if filter_by_type:
                cand_list = filter_candidates_by_type(cand_list, q)
            per_item_cands.append(cand_list)
            per_item_gold.append(gold)
            per_item_question.append(q)
            bucket_cand_sizes[batch["buckets"][i]].append(len(cand_list))

        # Encode all candidates from the batch in one shot for efficiency
        flat_cands: List[str] = []
        offsets: List[Tuple[int, int]] = []  # (start, end) in flat_cands per item
        for clist in per_item_cands:
            s = len(flat_cands)
            flat_cands.extend(clist)
            offsets.append((s, len(flat_cands)))

        cand_token_ids = [tok(c, add_special_tokens=False, truncation=True, max_length=tcfg.S_answer)["input_ids"]
                          for c in flat_cands]
        cand_ids_np, cand_mask_np = _pad_ids(cand_token_ids, tcfg.S_answer)
        # Encode in chunks to avoid OOM on very large batches
        cand_pools = []
        chunk_sz = 512
        for k in range(0, len(flat_cands), chunk_sz):
            pool_chunk = encode_cands(jnp.asarray(cand_ids_np[k:k + chunk_sz]),
                                      jnp.asarray(cand_mask_np[k:k + chunk_sz]))
            cand_pools.append(np.asarray(pool_chunk))
        if cand_pools:
            cand_pool = np.concatenate(cand_pools, axis=0)  # (sum_n_cands, D)
        else:
            cand_pool = np.zeros((0, backbone.text_encoder_dim))

        # Compute v_pred pooled per ablation, then NN per item
        rng, rng_n = jax.random.split(rng)
        t = jnp.full((tcfg.batch_size,), 0.02)
        noise = jax.random.normal(rng_n, (tcfg.batch_size, tcfg.S_answer, backbone.text_encoder_dim))
        batch_jax = {k: jnp.asarray(v) for k, v in batch.items() if isinstance(v, np.ndarray)}

        # Pre-encode the question embeddings (mean-pool) for question-conditioned scoring.
        if q_weight > 0:
            q_pool_jax = encode_cands(jnp.asarray(batch["question_ids"]), jnp.asarray(batch["question_mask"]))
            q_pool_np = np.asarray(q_pool_jax)
            q_pool_norm = q_pool_np / np.clip(np.linalg.norm(q_pool_np, axis=-1, keepdims=True), 1e-9, None)
        else:
            q_pool_norm = None

        for name, (mask_a, mask_b) in [("full", (1.0, 1.0)), ("zero_a", (0.0, 1.0)),
                                       ("zero_b", (1.0, 0.0)), ("zero_both", (0.0, 0.0))]:
            pred_pool = np.asarray(predict_pooled(params, batch_jax, t, noise, mask_a, mask_b))  # (B, D)
            pred_norm = pred_pool / np.clip(np.linalg.norm(pred_pool, axis=-1, keepdims=True), 1e-9, None)
            for i in range(tcfg.batch_size):
                s, e = offsets[i]
                if e <= s:
                    continue
                C = cand_pool[s:e]
                C_norm = C / np.clip(np.linalg.norm(C, axis=-1, keepdims=True), 1e-9, None)
                sims_pred = C_norm @ pred_norm[i]  # (n_cands,)
                if q_pool_norm is not None:
                    sims_q = C_norm @ q_pool_norm[i]
                    sims = (1.0 - q_weight) * sims_pred + q_weight * sims_q
                else:
                    sims = sims_pred
                pred_idx = int(np.argmax(sims))
                pred = per_item_cands[i][pred_idx]
                gold = per_item_gold[i]
                bucket = batch["buckets"][i]

                em = _em_norm(pred, gold)
                f1 = _squad_f1(pred, gold)
                bucket_em[bucket][name].append(em)
                bucket_f1[bucket][name].append(f1)
                # gold-in-candidates check
                gold_in = any(_normalize_answer(c) == _normalize_answer(gold) for c in per_item_cands[i])
                if name == "full":
                    bucket_recall[bucket].append(1.0 if gold_in else 0.0)
                if gold_in:
                    bucket_em_cond_recall[bucket][name].append(em)
                # Collect a few examples for the "full" ablation
                if name == "full" and len(examples_correct) < 4 and em > 0.5:
                    examples_correct.append({"q": batch["question_text"][i], "gold": gold, "pred": pred,
                                              "n_cands": len(per_item_cands[i]), "bucket": bucket})
                if name == "full" and len(examples_wrong) < 4 and em < 0.5 and gold_in:
                    examples_wrong.append({"q": batch["question_text"][i], "gold": gold, "pred": pred,
                                           "n_cands": len(per_item_cands[i]), "bucket": bucket})

        n_batches += 1
        if n_batches >= args.decode_max_batches:
            break

    # Report
    print(f"\n[{label.lower()}-decode] EXTRACTIVE NN DECODE (n_batches={n_batches}, bs={tcfg.batch_size}, oracle={oracle})")
    print(f"{'bucket':>10s}  {'n':>6s}  {'recall':>8s}  {'cand_size':>11s}  "
          f"{'full EM/F1':>14s}  {'zero_a':>14s}  {'zero_b':>14s}  {'zero_both':>14s}  {'EM|rec full':>13s}")
    for bucket in buckets:
        if not bucket_em[bucket]["full"]:
            continue
        n_items = len(bucket_em[bucket]["full"])
        rec = 100 * np.mean(bucket_recall[bucket]) if bucket_recall[bucket] else 0.0
        csize = np.mean(bucket_cand_sizes[bucket]) if bucket_cand_sizes[bucket] else 0.0
        cells = []
        for a in ablations:
            ems = bucket_em[bucket][a]
            f1s = bucket_f1[bucket][a]
            cells.append(f"{100*np.mean(ems):5.1f}/{100*np.mean(f1s):5.1f}" if ems else "    —")
        em_cr = bucket_em_cond_recall[bucket]["full"]
        em_cr_s = f"{100*np.mean(em_cr):5.1f}" if em_cr else "    —"
        print(f"{bucket:>10s}  {n_items:>6d}  {rec:>7.1f}%  {csize:>10.1f}   "
              + "  ".join(f"{c:>14s}" for c in cells) + f"  {em_cr_s:>13s}")
    if examples_correct:
        print(f"\n  [{label.lower()}-decode] Correct examples (full):")
        for ex in examples_correct:
            print(f"    [{ex['bucket']}] Q: {ex['q'][:80]}  gold={ex['gold']!r}  pred={ex['pred']!r}  n_cands={ex['n_cands']}")
    if examples_wrong:
        print(f"\n  [{label.lower()}-decode] Wrong examples (full, gold WAS in candidates):")
        for ex in examples_wrong:
            print(f"    [{ex['bucket']}] Q: {ex['q'][:80]}  gold={ex['gold']!r}  pred={ex['pred']!r}  n_cands={ex['n_cands']}")


if __name__ == "__main__":
    main()
