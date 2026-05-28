---
name: project-compute
description: GPU resources available for the ELF-MAS project as of 2026-05-14 — local RTX 5090 + remote NVIDIA server.
metadata: 
  node_type: memory
  type: project
  originSessionId: a4fa26be-d662-41d5-a043-71652dcaf996
---

Compute available for ELF-MAS (as of 2026-05-14):

- **Local:** 1× NVIDIA RTX 5090 (Blackwell architecture, 32 GB VRAM). Sits on `/home/mzyy1001` (WSL2, Ubuntu, 703 GB free).
- **Remote:** an NVIDIA-GPU server (model/SSH config not yet shared with Claude). Likely older arch (sm_80/sm_89), so probably **better JAX/TPU-port compatibility** than the local Blackwell GPU.
- **No TPU access.** ELF's reference code is JAX/TPU-first, so JAX-on-CUDA must be validated; PyTorch port may be needed.

**Why this matters:** Blackwell (sm_120) has bleeding-edge CUDA support — JAX 0.4.x stable wheels may not work on the 5090 yet. Prefer running the original ELF JAX code on the remote server first; reserve the 5090 for PyTorch experiments or for inference if JAX-on-Blackwell is unstable.

**How to apply:** When suggesting a run command, default to the remote server for any JAX run unless Hongrui confirms JAX works on the 5090. When suggesting PyTorch experiments, default to the 5090 (Blackwell is fine for recent PyTorch nightlies). Multi-seed sweeps and ELF-M (342M) belong on the remote server.
