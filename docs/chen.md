# chen compute server — connection info

> **Security note.** This file contains **non-secret** connection info only
> (host / port / user). The **private SSH key is NOT stored in this repository**
> and must never be committed. Obtain the key out-of-band (secure copy / secret
> store). If this repo is public, be aware that this advertises the server's SSH
> endpoint.

## SSH

```sshconfig
# ~/.ssh/config
Host chen
    HostName 122.225.39.134      # public IP (internal is 10.20.46.86)
    Port 2222
    User chenhongrui
    IdentityFile ~/.ssh/id_server_chen   # private key — keep local, NOT in git
```

With that block in place: `ssh chen`.

- Host: `122.225.39.134`  ·  Port: `2222`  ·  User: `chenhongrui`
- Auth: SSH key `~/.ssh/id_server_chen` (private key kept locally only).
- Shared, multi-tenant box (8× A100-80GB). Check `nvidia-smi` for a free GPU before
  launching; it is often busy with other tenants.

## Runtime / paths (chen-side)

- Activate env: `cd ~/elf_mas && source setup_env.sh` (conda env `elf_mas`, sets
  `CUDA_VISIBLE_DEVICES`, CUDA toolchain, `PYTHONNOUSERSITE=1`, cuDNN paths).
- Repo: `~/elf_mas/` · data: `~/elf_mas/data/` · runs/checkpoints: `~/elf_mas/runs/`.
- HF assets (Cola-DLM, T5, etc.): `~/elf_mas/hf_cache/elf-mas-bridge/`.
- Network: HuggingFace / Google are GFW-blocked on chen; move HF assets via the
  `modelscope.cn` bridge repo `mzyy1001/elf-mas-bridge` (see project notes).
