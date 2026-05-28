---
name: feedback-framing
description: "Hongrui's preferences for how to frame the ELF-MAS contribution — training claim central, DIAL/MARL not a killer."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: a4fa26be-d662-41d5-a043-71652dcaf996
---

When framing the contribution of the ELF-MAS / Coupled Multi-Agent Language Flows project (and analogous research projects), Hongrui prefers:

1. **The headline contribution is the TRAINING claim**, not the architecture and not the communication channel. For ELF-MAS: "end-to-end backprop through coupled denoising trajectories induces emergent specialization." Architecture (velocity residuals + gates) is the *mechanism*; the empirical finding (specialization emerges) is the *result*.
2. **Headline experiment shows the training claim**, not generic accuracy gains. E.g. coupled-flow MAS vs. stop-gradient variant on a specialization metric — not "latent beats text" on a leaderboard.
3. **Do not treat classical MARL papers (DIAL, CommNet, etc.) as novelty-killers** for modern language-agent work. They are related work on the *general* differentiable-MAS idea but operate on tiny policies and coordination tasks, not language flows or pretrained backbones. Cite, do not surrender.
4. **Synthetic / controlled task before downstream benchmark.** When a research claim depends on something emerging (specialization, structure, separation), pilot on a setting where emergence is *observable by construction* — only then scale to HotpotQA / GSM8K / etc.

**Why:** confirmed 2026-05-14 when Hongrui rewrote the post-novelty-check pivot. He accepted the "Coupled Multi-Agent Language Flows" name and the velocity-residual mechanism, but explicitly pushed back on (a) treating DIAL as a killer (it isn't — wrong setting), and (b) making the headline about "latent beats text" — the win condition should be about training-induced specialization.

**How to apply:** When proposing new directions, ablations, or paper narratives for this project — and for any future research projects where the user is rebuilding after a novelty hit — default the headline framing to the training/learning claim, not the architectural or comm-channel claim. When listing prior work that "kills" a pitch, distinguish *setting-matched kills* from *general-precedent related work* — only the former should pivot the project. When the contribution depends on emergent behavior, propose a controlled pilot first.
