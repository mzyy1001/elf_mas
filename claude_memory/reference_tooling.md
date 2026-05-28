---
name: reference-tooling
description: "Tooling state for the ELF-MAS project — repo paths, codex/ARIS install status, env files."
metadata: 
  node_type: memory
  type: reference
  originSessionId: a4fa26be-d662-41d5-a043-71652dcaf996
---

State of dev tooling for `/home/mzyy1001/elf_mas` (as of 2026-05-14):

- **ELF baseline code:** `/home/mzyy1001/elf_mas/baselines/ELF/` — clone of `lillian039/ELF`, JAX/TPU. Entry: `src/train.py`, `src/eval.py`. Configs in `src/configs/training_configs/*.yml`. Pretrained checkpoints on HuggingFace under `embedded-language-flows/{ELF-B-owt, ELF-B-de-en, ELF-B-xsum, ELF-M-owt, ELF-L-owt}`.
- **ARIS source:** `/home/mzyy1001/elf_mas/aris-source/` — clone of `wanshuiyin/Auto-claude-code-research-in-sleep`. The skills were *already* copied to `~/.claude/skills/` before 2026-05-14 (existing dir from May 13); the May 14 `cp -r` was effectively an update to the latest commit (`745f856`). The aris-source clone is kept around so future updates can use `tools/smart_update.sh`.
- **Codex CLI:** installed at the user-owned fnm Node prefix (`codex-cli 0.130.0`). Registered as a Claude Code MCP server in **user scope** (`claude mcp add codex -s user -- codex mcp-server`). **Authed 2026-05-14 09:15** (`~/.codex/auth.json`, model `gpt-5.5`); MCP health check shows `codex: ✓ Connected`.
- **`.env.example`:** copied to `/home/mzyy1001/elf_mas/.env.example`. Filled-in `.env` needs `GEMINI_API_KEY` and optional others for ARIS sub-skills to work.

**How to apply:** Codex review skills are live — feel free to invoke `/research-review`, `/auto-review-loop`, `/kill-argument`, `/novelty-check`, etc. without re-checking auth (unless they fail with auth errors). For any ELF run, use the repo at `baselines/ELF/` and check `src/configs/training_configs/` for the YAML that matches the task.

**chen server setup (chen-side ELF runtime, validated 2026-05-15):**
- Conda env: `/data/chenhongrui/miniconda3/envs/elf_mas` (Python 3.10).
- Activation script: `~/elf_mas/setup_env.sh` — sources conda, sets `PATH` to prepend `nvidia/cuda_nvcc/bin` (for ptxas 12.9), pins `CUDA_VISIBLE_DEVICES` to a free GPU (2-7; 0/1 are in use by other tenants), sets `PYTHONNOUSERSITE=1` (critical — user-site numpy 2.x breaks env), appends `LD_LIBRARY_PATH` with `/data/anaconda3/envs/elmrec/lib/python3.12/site-packages/nvidia/cudnn/lib` (borrowed cuDNN 8 for torch 2.3.0+cu121; JAX still uses bundled cuDNN 9 since env site-packages wins).
- Key pinned versions in env: jax==0.4.38, jaxlib==0.4.38, flax==0.10.2, optax==0.2.5, orbax-checkpoint==0.5.23, transformers==4.41.2 (needs tokenizers<0.20), wandb upgraded to 0.27.0 (0.16.6 protobuf-broke), numpy<2.0.0, torch 2.3.0+cu121 (--no-deps via aliyun mirror).
- Repo + assets:
  - ELF code: `~/elf_mas/ELF/` (cloned from lillian039/ELF).
  - HF assets: `~/elf_mas/hf_cache/elf-mas-bridge/` (downloaded via modelscope from `mzyy1001/elf-mas-bridge` — see [[reference-network]]).
- Working eval invocation (BLEU 26.53 on WMT14 De-En val, 64-step ODE CFG=2, 3m13s on 1× A100): `cd ~/elf_mas/ELF/src && source ~/elf_mas/setup_env.sh && export CUDA_VISIBLE_DEVICES=2 && export HF_HUB_OFFLINE=1 && python eval.py --config configs/training_configs/train_de-en_ELF-B.yml --checkpoint_path ~/elf_mas/hf_cache/elf-mas-bridge/ELF-B-de-en/checkpoint_0 --config_override eval_data_path=~/elf_mas/hf_cache/elf-mas-bridge/wmt14_de-en_validation_t5 --config_override encoder_checkpoint=~/elf_mas/hf_cache/elf-mas-bridge/t5_small_encoder_jax/t5_small_encoder_jax.pkl --config_override encoder_model_name=~/elf_mas/hf_cache/elf-mas-bridge/t5-small --config_override use_wandb=false`.
