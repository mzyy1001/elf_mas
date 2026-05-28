---
name: reference-network
description: chen server network constraints — international ingress is throttled; HF blocked; modelscope is the working bridge.
metadata: 
  node_type: memory
  type: reference
  originSessionId: a4fa26be-d662-41d5-a043-71652dcaf996
---

chen server (Hangzhou ChinaNet AS4134) has highly asymmetric internet — fine to most Chinese clouds, but **HuggingFace and Google are GFW-blocked**, and **incoming international bandwidth is throttled to ~20 KB/s** regardless of origin (verified both from WSL home broadband and from Imperial College `lab` server — same wall).

**What works at fast speeds from chen:**
- aliyun mirrors (`mirrors.aliyun.com` for pip, OSS `oss-cn-hangzhou`): MB/s.
- **modelscope.cn** (Alibaba's HF mirror, https://modelscope.cn): chen→modelscope downloads at 7-10 MB/s, WSL→modelscope uploads at 15.9 MB/s.
- GitHub HTTPS git operations: slow but functional (~100 KB/s for clones).
- jsdelivr.net, Cloudflare CDN, modelscope, gitee, weiyun, quark, baidu pan, etc.: all reachable.

**What's blocked:**
- huggingface.co (DNS poisoned + TCP blocked, 0 B/s).
- hf-mirror.com (also 0 B/s — surprising; that mirror is normally reachable).
- google.com, google APIs.

**The bridge pattern that works for moving HF assets to chen:**
1. Download from HF on WSL (or any machine with HF access).
2. `huggingface-cli download <hf-repo> --local-dir <local-dir>`.
3. Use modelscope's `HubApi.upload_folder` to push to a private modelscope repo. Uses parallel HTTP API uploads (NOT git+LFS — `push_model` hangs on git-LFS over the international link; `upload_folder` is much more reliable).
4. On chen: `modelscope.snapshot_download(repo_id, local_dir=..., allow_patterns=[...])` to pull at LAN speed.

**Reference modelscope bridge repo (Hongrui's):** `mzyy1001/elf-mas-bridge` (private model repo). Contents currently hosted there: ELF-B-de-en checkpoint, T5 small encoder pkl, WMT14 De-En validation t5, t5-small tokenizer files. Auth via SDK token (long-lived; rotate when concerned).

**How to apply:** When the project needs any HF asset on chen, follow the bridge pattern above. NEVER try direct scp WSL→chen for files > 5 MB — wastes hours at 20 KB/s. NEVER expect chen to reach huggingface.co directly. When proposing a workflow that needs HF on chen, default to "download on WSL → push to modelscope bridge → pull on chen." If the assets are already in `mzyy1001/elf-mas-bridge`, the snapshot_download with `allow_patterns` can target just the needed subset.
