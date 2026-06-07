# DiT layer roles & MAS-injection interactions — causal probe suite

**Date:** 2026-06-07 · model **B0 fixed-KG** `runs/cola_mas_sweep/full_seed0` · dataset
`synth_2fact_v1` · AB bucket, per-token EM. **The model analysed is SHORTCUT-CONTAMINATED**
(person name determines the answer); see "Safe vs confounded" below.

Outputs: `notes/lyr_subset_results.csv`, `notes/figures/layer_{prefix_suffix,synergy,mediation,patching}.png`,
JSONs `notes/lyr_{subset_scan,mediation,patching}.json`.

## 1. All-subset scan (AB full EM, both agents, only listed layers enabled)
| subset | EM | subset | EM |
|---|---|---|---|
| {} | 0.000 | {8,12} | **0.207** |
| {8} | 0.012 | {8,12,16} | 0.317 |
| {12}/{16}/{20} | 0.000 | {8,12,20} | **0.500** |
| {8,16} | 0.037 | {8,16,20} | 0.171 |
| {8,20} | 0.049 | {12,16,20} | 0.256 |
| {12,16} | 0.037 | {16,20} | 0.012 |
| {12,20} | 0.098 | **{8,12,16,20}** | **0.622** |

## 2. Prefix / suffix curves (`layer_prefix_suffix.png`)
- prefix {}→{8}→{8,12}→{8,12,16}→all: **0.0, 0.012, 0.207, 0.317, 0.622**
- suffix {}→{20}→{16,20}→{12,16,20}→all: **0.0, 0.0, 0.012, 0.256, 0.622**

Late layers **alone are useless** ({20}=0, {16,20}=0.012); they only work once early layers are
added. **Front-loaded** — early layers build a substrate later layers refine.

## 3. Pairwise synergy (`layer_synergy.png`)
`synergy(8,12)=+0.195` (dominant), `(12,20)=+0.098`, all others ≤+0.037, `(16,20)=+0.012`.
→ the synergistic core is the **early pair (8,12)**; late pair (16,20) has ~no standalone synergy.

## 4. Mediation: does manipulating A change agent B's QUERY? (`layer_mediation.png`)
cosine(q_B full vs condition):

| cond | L8 | L12 | L16 | L20 |
|---|---|---|---|---|
| rmA | 1.000 | 0.958 | 0.938 | 0.990 |
| ctxA_shuf | 1.000 | 0.990 | 0.994 | 0.998 |
| **ctxB_shuf** (control) | 1.000 | 1.000 | 0.998 | 0.999 |
| swap | 1.000 | 0.986 | 0.989 | 0.996 |

- **L8 identical under every condition** (sanity: B reads the pre-injection block-8 output).
- **Removing A changes B's query at L12/L16** (cos 0.958/0.938) → A's writes propagate to B
  through the shared hidden state (A→B mediation), localized to mid layers.
- **ctxB-shuffle control ≈1.0** → B's query is insensitive to ctx_B content (enters via B's K/V,
  not its query). Clean.
- The representational change is **small (~4–6%)** — but patching (below) shows it is causally
  potent. So query/gate magnitudes again **understate** causal importance; patching is the
  trustworthy probe.

## 5. Activation patching (causal sufficiency; `layer_patching.png`)
full=0.646, rmA=0.171, rmB=0.000. Restoration fraction = (patched−ablated)/(full−ablated):

| patch | EM | restoration |
|---|---|---|
| rmA + h_after_L8 | 0.500 | **0.69** |
| rmA + h_after_L12 | 0.537 | 0.77 |
| rmA + h_after_L16 | 0.561 | 0.82 |
| rmB + h_after_L12 | 0.293 | 0.45 |
| **rmB + h_after_L20** | 0.646 | **1.00** |

- **Agent A's contribution is causally sufficient early:** patching the hidden after **layer 8**
  alone rescues 69% of the rmA loss (rising to 82% by L16). A does its essential work by ~L8 and
  writes it into the shared stream.
- **Agent B's contribution finalizes late:** patching after **layer 20** *fully* rescues rmB
  (1.00), while L12 only partially (0.45). B's second-hop write completes by the last MAS layer.

→ A causal **early-A → late-B pipeline mediated by the shared hidden state**.

## Answers to the seven questions
- **Necessary layers** (knockout = all − {all\L}): **L12 (+0.451) > L8 (+0.366) > L20 (+0.305) ≫ L16 (+0.122)**. 8/12/20 necessary; 16 largely dispensable.
- **Sufficient subsets:** **{8,12,20}=0.500 (80% of full) without layer 16**; {8,12}=0.207 is the strongest pair. No single layer is sufficient (all ≈0).
- **Synergy pairs:** **(8,12)** strongly (+0.195), (12,20) moderate; (16,20)≈0.
- **Do later layers depend on earlier?** **Yes** — suffix-alone curve is ~0; late layers require the early substrate; patching A@L8 rescues most of the answer.
- **A→B via shared hidden state?** **Yes, causally** — q_B changes at L12/L16 when A is removed, and patching A's early hidden rescues rmA. The representational shift is small but causally large.
- **16/20 refine vs redundant?** **L16 is largely redundant** (knockout +0.12, synergy ~0, {8,12,20} skips it). **L20 is a strong late refiner that depends on early layers** (knockout +0.305; {8,12}→{8,12,20} adds +0.293; useless alone) and is where agent B's write becomes sufficient (patch@20=full rescue).
- **Safe vs confounded:**
  - **Safe (mechanics of this model):** a real, well-localized causal pipeline — A writes early (sufficient by ~L8), B finalizes at L20, mediated by the residual stream; front-loaded; 16 redundant; synergy in (8,12).
  - **Confounded:** this is the **shortcut-contaminated** fixed-KG model. The randomized-KG control showed the *same recipe* does **not** use agent A at all (Δa=0) once the person-name shortcut is removed. So the clean A→B pipeline here is **at least partly shortcut-enabled** — the model learned to route ctx_A's content because it correlated with the memorizable answer. These mechanics describe *how this model routes information*; they are **not** evidence of genuine shortcut-free composition. A randomized-KG model that first learns nonzero A-dependent composition is required before this layer analysis is meaningful for the shortcut-free claim.
