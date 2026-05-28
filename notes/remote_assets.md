# Remote Asset Locations

Big artifacts (model weights, datasets, training checkpoints) are not committed
to git. They live on the chen server at the paths below. To reproduce Phase 3,
either re-run the data prep scripts and re-download Cola from the modelscope
bridge, or `rsync` from these locations.

## On chen (`ssh chen`)

### Cola-DLM backbone (frozen)

```
/home/chenhongrui/elf_mas/hf_cache/elf-mas-bridge/phase3_assets/cola_dlm/
├── cola_dlm/cola_dit/        # 1.8B-param DiT, bfloat16
├── cola_dlm/cola_vae/        # 500M-param VAE
└── tokenizer.json            # OLMo-2 tokenizer
```

Mirrored from `modelscope.cn/mzyy1001/elf-mas-bridge` (the WSL→chen transit
repo). HuggingFace is blocked on chen; modelscope is the only working transit.

### ELF backbone (Phase 1/2 baseline)

```
/home/chenhongrui/elf_mas/baselines/ELF/                # source
/home/chenhongrui/elf_mas/hf_transfer/                  # ELF-B-de-en weights
/home/chenhongrui/elf_mas/hf_transfer_t5/               # T5 encoder weights
```

### Datasets

```
/home/chenhongrui/elf_mas/data/synth_2fact_v1/          # Phase 1/3 synth task
/home/chenhongrui/elf_mas/data/musique_2hop_v1/         # Phase 2 benchmark
/home/chenhongrui/elf_mas/cola_tasks/squad.jsonl        # SQuAD repro probe
/home/chenhongrui/elf_mas/cola_tasks/lambada.jsonl      # LAMBADA repro probe
```

### Phase 3 training runs (LoRA + CE checkpoints)

```
/home/chenhongrui/elf_mas/runs/cola_mas_pilot/
├── b0_lora_uniform_lce0.3_30k/     # ← Phase 3 breakthrough (58.5% AB-bucket EM)
├── b0_synth_pfx_seed0_5k/          # earlier 5k prefix-cond run
├── b0_synth_pfx_big_seed0_20k/     # 20k bigger-heads run (superseded)
└── b0_lora_seed0_10k/              # 10k LoRA-no-CE run (superseded)
```

The 30k uniform-t run is the one to use. Its `lora_state.pt` contains all
4 wrappers' MAS state dicts plus the layer indices + hyperparams.

### Phase 3 CE λ-sweep

```
/home/chenhongrui/elf_mas/runs/cola_mas_ce_sweep/
├── synth_lora_lce0.1_seed0_10k/
├── synth_lora_lce0.3_seed0_10k/
└── synth_lora_lce1.0_seed0_10k/
```

These use the OLD logit-normal t-distribution and 10k steps — superseded by the
30k uniform-t run, but kept for the record of what didn't work and why.

## On local WSL (`/home/mzyy1001/elf_mas/`)

Source + notes only (this repo). Plus a `phase3_assets/` mirror of the Cola
weights for local sanity checks — not committed, can be re-downloaded from
modelscope.
