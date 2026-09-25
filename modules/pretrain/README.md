# `modules/pretrain` — Pretraining Module

This module trains the AMPLIFY protein language model with masked language modeling (MLM) on the clustered, scored datasets produced by [`modules/data`](../data/README.md). It is configured through YAML files (validated by Pydantic) with optional CLI dotlist overrides via OmegaConf.

```
Prepare sources → Build epoch mixture → Collate & mask → Train loop → Checkpoints
```

A run is assembled from components (model, tokenizer, optimizer, scheduler, collator, dataloaders, metric logger), each configured by its own top-level YAML section and swappable without touching the trainer.

## Table of Contents

- [Quick Start](#quick-start)
- [Directory Layout](#directory-layout)
- [Output Disk Layout](#output-disk-layout)
- [Dataset](#dataset)
- [DataLoader](#dataloader)
- [Collator](#collator)
- [Model & Tokenizer](#model-tokenizer)
- [Optimizer & Scheduler](#optimizer-scheduler)
- [Trainer](#trainer)
- [Metric Logger](#metric-logger)
- [Running a Pretraining Run](#running-a-pretraining-run)
- [torch.compile](#torchcompile)
- [Config System](#config-system)

---

## Quick Start

Before starting, run `uv sync --extra pretrain` from the `flair-plm` root. Packed
pretraining works with native PyTorch variable-length attention; FlashAttention is
an optional accelerator (see the [main README](../../README.md)).

To quickly test the pretraining pipeline, use `configs/config.yaml` + `configs/smoke_gpu.yaml`: a small model (4 layers, hidden size 256) trained on 5k real sequences per source (see `configs/smoke_data/`).

### Single GPU

To run the pretraining smoke test on a single GPU:

```bash
accelerate launch --mixed_precision bf16 \
    modules/pretrain/src/run_pretrain.py \
    modules/pretrain/configs/config.yaml \
    modules/pretrain/configs/smoke_gpu.yaml
```

### Multi-GPU

To run the pretraining smoke test on a 4 GPUs:

```bash
accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 \
    modules/pretrain/src/run_pretrain.py \
    modules/pretrain/configs/config.yaml \
    modules/pretrain/configs/smoke_gpu.yaml
```

Note that `dataloader.max_tokens` is **per process**, so the effective batch scales with `num_processes` and step counts divide accordingly. Phase 1 dataset preparation is guarded by `local_main_process_first()`, so rank 0 builds the cache and the rest reuse it.

> Example run on 4x A100s on Mila cluster (packed sequences): final train loss 2.83 after 54 steps (3 epochs).

### Artifacts

Written under `trainer.output_dir` (`smoke_runs/gpu` by default):

```
smoke_runs/gpu/
├── metrics.jsonl                     ← one JSON record per logging flush
├── checkpoint_epoch_0_step_50/       ← resumable Accelerate state
│   ├── model.safetensors, optimizer.bin, scheduler.bin
│   ├── random_states_0.pkl, custom_checkpoint_0.pkl
│   └── trainer_state.pt              ← epoch/step counters
└── hf_checkpoints/checkpoint_50/     ← portable HF checkpoint
    ├── config.json, model.safetensors
    ├── modeling_amplify.py, configuration_amplify.py, tokenizer.py
    └── tokenizer.json, tokenizer_config.json
```

**To verify that the portable checkpoint** loads without this repo on `sys.path`:

```python
from transformers import AutoModelForMaskedLM
model = AutoModelForMaskedLM.from_pretrained(
    "smoke_runs/gpu/hf_checkpoints/checkpoint_50", trust_remote_code=True
)
```

### Real pretraining runs

Once the smoke test passes, launch a full-size run. `configs/config.yaml` **is** the 350M config (standalone, runnable as-is, and ships with working Hugging Face Hub defaults for `bfd`/`mgnify`/`uniref`); for 120M, merge `120M.yaml` on top (see [Running a Pretraining Run](#running-a-pretraining-run) for CLI overrides, resuming, and merging configs). To point at local Parquet data instead of the Hub for a given cluster, override `dataset.sources.*` via CLI dotlist (see [Dataset](#dataset) for the full override options — removing a source, switching its type, or adding a new one):

```bash
uv run accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 \
    modules/pretrain/src/run_pretrain.py \
    modules/pretrain/configs/config.yaml \
    modules/pretrain/configs/120M.yaml \
    dataset.sources.bfd.type=parquet \
    dataset.sources.bfd.path=data_subsets/bfd_assembled/*.parquet \
    dataset.sources.bfd.cluster_column=cluster_rep_at_30 \
    dataset.sources.bfd.sampling_fraction=1.0 \
    dataset.sources.mgnify.type=parquet \
    dataset.sources.mgnify.path=data_subsets/mgnify_assembled/*.parquet \
    dataset.sources.mgnify.cluster_column=cluster_rep_at_30 \
    dataset.sources.mgnify.sampling_fraction=1.0 \
    dataset.sources.uniref.type=parquet \
    dataset.sources.uniref.path=data_subsets/uniref_assembled/*.parquet \
    dataset.sources.uniref.cluster_column=cluster_rep_at_30 \
    dataset.sources.uniref.sampling_fraction=1.0
```

Note: On multi-GPU, the first run against a new/uncached dataset can take longer than PyTorch's
default NCCL collective timeout. If you see a timeout error, build the cache first with `dataset.prepare_only=true` (below), or raise `trainer.dist_timeout_minutes` (default `90`) for that first run. Subsequent runs will reuse the cache and are fast.

### Building the dataset cache first (CPU-only)

On the full datasets, the one-time cache build (split, flatten, cluster index) can take hours. Doing it inside a real training run wastes GPU hours, and on multi-GPU it can trip the NCCL watchdog: the build is wrapped in `local_main_process_first()`, so the other ranks sit at a NCCL barrier while rank 0 works, and they abort if it takes too long.

`dataset.prepare_only=true` builds the caches and exits before the model is created, so it needs no GPU and no distributed setup. Run it as a single plain process:

```bash
python modules/pretrain/src/run_pretrain.py \
    modules/pretrain/configs/config.yaml \
    modules/pretrain/configs/120M.yaml \
    dataset.sources.uniref.type=parquet \
    dataset.sources.uniref.path=data_subsets/uniref_assembled/*.parquet \
    dataset.sources.uniref.cluster_column=cluster_rep_at_30 \
    dataset.prepare_only=true
```

It finishes with `Data sources prepared. Exiting (dataset.prepare_only).`, and the real run then starts warm (`Cluster index cache hit` per source).

### What triggers a cache rebuild

Each source is cached independently, so changing the source mix (adding, removing, or re-weighting sources) never rebuilds another source's cache. Within a source, the three cache layers are keyed as follows:

| Cache | Rebuilt when you change | Not affected by |
|---|---|---|
| Base load (parquet → Arrow) | The source files (name, size, or modification time; re-syncing files counts), or the Hub revision | Anything below |
| Train/val split (the expensive pass) | The source files, `split_column`, `val_holdout_modulus` | `num_proc`, `split_writer_batch_size`, `cluster_column`\*, `score_column`\*, `sampling_fraction`, curriculum |
| Cluster index | Anything that rebuilds the split, plus `cluster_column` or `score_column` | `sampling_fraction`, curriculum |

\* Only if the new column was already carried in the split cache (see [Reusing one cache across cluster thresholds](#reusing-one-cache-across-cluster-thresholds)).

Per-threshold row counts (`valid_count-*.npy`) are cached next to the cluster index, so a new curriculum threshold only adds a quick scan.

Keep `HF_HOME`/`HF_DATASETS_CACHE` identical between runs, on storage visible from every node.

### Staging data to node-local disk (network filesystems)

Training reads rows in random order. If `HF_DATASETS_CACHE` is on a network filesystem that doesn't page-cache those reads (e.g. Mila's `/network/projects`, BeeGFS with `tuneFileCacheType=buffered`), every read is a network round trip. Large sources like BFD then risk starving the GPUs. Set `dataset.local_stage_dir` to a node-local directory to copy each source's sequence-only train/val splits there once per job before training starts:

```bash
    dataset.local_stage_dir=$SLURM_TMPDIR/flair_plm_stage \
    trainer.dist_timeout_minutes=90
```

Copies are named by dataset fingerprint and reused by the other ranks on the node. The copy runs inside `local_main_process_first()`, so raise `trainer.dist_timeout_minutes` to cover it. Leave it unset when the cache is already on local disk.

### Reusing one cache across cluster thresholds

`cluster_column` is both the grouping key and, on its own, the train/val split key — so changing it would rerun the expensive split+flatten pass *and* move the holdout, leaving runs at different thresholds with non-comparable validation sets. Two per-source settings decouple them so one split+flatten pass serves every threshold:

* `split_column` — the column used for the train/val holdout, defaulting to `cluster_column`. Pin it to a fixed, coarse clustering (`cluster_rep_at_30`; `cluster_rep_at_80` for OAS).
* `extra_cluster_columns` — additional cluster columns retained in the cache. List every threshold you plan to train on.

With both set, switching `cluster_column` rebuilds only the cluster index and reuses the split+flatten output:

```yaml
uniref:
  cluster_column: cluster_rep_at_30       # override per experiment
  split_column: cluster_rep_at_30         # fixed: holdout never moves
  extra_cluster_columns: [cluster_rep_at_30, cluster_rep_at_50, cluster_rep_at_70]
```

Costs and caveats:

* **Set these on the first build.** The split cache keeps only the columns listed at build time, and the column list is not part of its cache key. The defaults in `config.yaml` already list every cluster column each source provides, so any `cluster_column` works out of the box. If you build with a shorter list and add a column later, the old cache is reused and fails with `KeyError: Field "..." does not exist in schema`; delete that source's split cache to rebuild it.
* Disk grows with the number of carried columns: ~1.35x for 4, ~1.59x for 7.
* Pinning `split_column` assumes clusterings are **nested** (a coarse cluster is never split across train and val at a finer threshold). This holds by construction for UniRef; verify it for independently produced clusterings.

### Launching on a SLURM cluster (e.g. Mila cluster)

A minimal single-node, multi-GPU example on how to submit the job with `sbatch` instead of an interactive `salloc` (to be adapted depending on needs and compute):

```bash
#!/bin/bash
#SBATCH --job-name=<your_job_name>
#SBATCH --output=logs/%x_%j_output.txt
#SBATCH --error=logs/%x_%j_error.txt
#SBATCH --time=0-03:00
#SBATCH --partition=short-unkillable
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G

set -euo pipefail
cd <path_to_flair-plm>

# Activate a pre-built venv.
source <path_to_venv>/bin/activate

# Launch pretraining (merging base config and 120M-specific config).
uv run accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 \
    modules/pretrain/src/run_pretrain.py \
    modules/pretrain/configs/config.yaml \
    modules/pretrain/configs/120M.yaml \
    "$@"
```

To submit the job after saving the script above as a shell script (additionally forwarding any config overrides as extra args if needed):

```bash
sbatch <your_script>.sh trainer.max_steps=2000
```

---

## Directory Layout

```
modules/pretrain/
├── README.md                        ← this file
│
├── configs/
│   ├── config.yaml                  ← default config, every section explicit; this
│   │                                  is the 350M config (standalone, runnable as-is)
│   ├── 120M.yaml                    ← 120M-specific overlay on config.yaml (118.3M params), 4x A100
│   ├── smoke_cpu.yaml               ← CI-only CPU config (synthetic data)
│   ├── smoke_gpu.yaml               ← CUDA smoke overlay on config.yaml (bf16, fused, packed)
│   └── smoke_data/                  ← 5k-row real parquet slices used by smoke_gpu.yaml
│       ├── bfd_smoke.parquet, mgnify_smoke.parquet, uniref_smoke.parquet
│
└── src/
    ├── config.py                    ← PretrainConfig: top-level Pydantic schema composing
    │                                  every component config into one object
    ├── run_pretrain.py              ← CLI entry point: builds components, wires the Trainer
    │
    ├── dataset/
    │   ├── dataset.py               ← DatasetConfig, source preparation, cluster indexing,
    │   │                              score curriculum, per-epoch mixture building
    │   ├── dataloader.py            ← DataLoaderConfig, TokenBatchSampler, build_split_dataloader
    │   └── collator.py              ← CollatorConfig, DataCollator (MLM masking + padding/packing)
    │
    ├── model/
    │   ├── configuration_amplify.py ← AMPLIFYConfig (HF PretrainedConfig)
    │   ├── modeling_amplify.py      ← AMPLIFY encoder + MLM/classification heads
    │   └── tokenizer.py             ← TokenizerConfig + ProteinTokenizer (character-level)
    │
    ├── optimizer/optimizer.py       ← OptimizerConfig + decay/no-decay param grouping
    ├── scheduler/scheduler.py       ← SchedulerConfig (HF get_scheduler wrapper)
    ├── trainer/trainer.py           ← TrainerConfig, WandbConfig, Trainer (train loop, checkpoints)
    └── metric/logger.py             ← TrainingLogger: on-device metric accumulation + reduction
```

Shared config loading helpers live in `modules/core/utils/config_loader.py`.

---

## Output Disk Layout

A training run writes everything under `trainer.output_dir`, including W&B's
local log data by default, so every run is a single self-contained folder:

```
<output_dir>/
├── metrics.jsonl                                  ← append-only metric records (mirrors W&B)
├── wandb/                                         ← local W&B run data (if wandb.enabled)
├── checkpoint_epoch_<E>_step_<S>/                 ← resumable training state
│   ├── model.safetensors                          ← model weights
│   ├── optimizer.bin, scheduler.bin               ← optimizer/scheduler state
│   ├── random_states_*.pkl                        ← RNG state for exact resumption
│   └── trainer_state.pt                           ← epoch, global_step, batches_completed
└── hf_checkpoints/                                ← portable, self-contained HF checkpoints
    └── checkpoint_<S>/
        ├── config.json                            ← includes auto_map
        ├── model.safetensors
        ├── modeling_amplify.py                    ← copied source, so the checkpoint
        ├── configuration_amplify.py                 loads without this repo on the path
        ├── tokenizer.py
        └── tokenizer.json, tokenizer_config.json
```

Resumable checkpoints are saved **directly** in `output_dir`; portable ones go under `hf_checkpoints/`. Use `checkpoint_epoch_*` to resume an interrupted run (`trainer.resume_from_checkpoint`), and `hf_checkpoints/` for downstream evaluation, fine-tuning, or Hub upload:

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained(path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
```

### Checkpoint saving, decoupling, and rotation

The two checkpoint types fire strictly on their own independent schedules and never trigger each other:

- **Resumable state** (`checkpoint_epoch_*_step_*`) saves every `trainer.save_steps` optimizer steps.
- **HF checkpoint** (`hf_checkpoints/checkpoint_<S>/`) saves every `trainer.hf_save_steps` optimizer steps.
- Neither is saved automatically at the end of every epoch anymore. Instead, both are guaranteed to be saved exactly once at the very end of training (after the last epoch/`max_steps`), unless a step-based save already landed on that exact final step.
- Set `trainer.save_total_limit` / `trainer.hf_save_total_limit` (both `None`/unlimited by default) to cap how many checkpoints of each type are kept on disk; the oldest (by step) are deleted automatically as new ones are saved. This keeps storage bounded on long runs.

### Uploading a checkpoint to the Hugging Face Hub

Once a checkpoint under `hf_checkpoints/` looks good, push it with `modules/core`'s shared upload helper (remember to set `HF_TOKEN` in your env. first):

```python
from modules.core.utils.hf_upload import upload_to_hf

upload_to_hf(
    folder_path="runs/120M/hf_checkpoints/checkpoint_8000",
    repo_id="org-name/amplify-120m",
    commit_message="AMPLIFY 120M, step 8000",
)
```

This is the same helper `modules/data`'s Upload step uses; see [`hf_upload.py`](../core/utils/hf_upload.py) for all options (private repos, revisions, path filters).

### W&B run resumption

Setting the same `wandb.run_name` on a re-launched job (e.g. after a SLURM preemption, via `trainer.resume_from_checkpoint`) does **not** by itself resume the same W&B run. W&B only resumes runs matched by a stable `id`. To make this work transparently, `wandb.id` defaults to a slug of `wandb.run_name`, combined with `wandb.resume: "allow"` (creates the run if the id is new, resumes it if it already exists). Set `wandb.id` explicitly if you need distinct W&B runs under the same `run_name`, or a different resume policy via `wandb.resume` (`"must"`, `"never"`, ...).

---

## Dataset

**Purpose:** Turn one or more assembled datasets (from `modules/data`) into a per-epoch training mixture. Runs in **two phases**:

1. **Prepare (once, cached)** — for each source: load the Parquet/Hub dataset, split clusters into train/val, and build a `cluster → row indices` index sorted by score. No score filtering happens here; that is done per epoch in phase 2. Cached to disk so repeated runs and multi-GPU ranks don't rebuild it (see [What triggers a cache rebuild](#what-triggers-a-cache-rebuild)).
2. **Build epoch mixture (once per epoch)** — pick one representative row per cluster (varying with the epoch seed), apply the score curriculum threshold for that epoch, subsample each source by its `sampling_fraction`, and concatenate the sources.

The `score` column comes from `modules/data`'s score step (see [its README](../data/README.md#step-4--score)): a per-sequence **RED** (Residue Embedding Diversity) score from an existing pretrained pLM, where higher means the sequence's embeddings are more informationally distinct (can be seen as a quality/diversity signal).

**Cluster-level deduplication** is the core idea: each epoch samples one random representative per cluster (uniformly, independent of prior epochs), so redundant homologs don't dominate any single epoch. Over enough epochs the model sees the full *inter*-cluster diversity, and, depending on cluster size, increasingly more of the *intra*-cluster diversity too (since each row's chance of being picked in a given epoch is `1/cluster_size` and the same representative can recur).

**Score curriculum:** the minimum-score threshold ramps from `start_score` to `end_score` over `ramp_epochs`, then holds. Training starts on nearly all data and progressively narrows to the highest-quality sequences.

**Key config fields (`dataset`):**

| Field | Default | Description |
|---|---|---|
| `sources` | Hub defaults for `bfd`/`mgnify`/`uniref` | Mapping of name → data source (see below); at least one required. Overridable per source via CLI dotlist without editing `config.yaml` — remove a source (`sampling_fraction=0`), switch its `type` (`hub`↔`parquet`), or add a new source name entirely (see [Real pretraining runs](#real-pretraining-runs)). An empty/all-zero-fraction `sources` raises a clear `ValueError` at startup. |
| `score_column` | `red_score` | Column holding quality scores (`RED` from the data pipeline) |
| `score_curriculum` | `type: linear, start_score: 0.0, end_score: 0.0, ramp_epochs: 1` | `type` (`linear`/`cosine`/`exponential`), `start_score`, `end_score`, `ramp_epochs` |
| `val_min_score` | `0.0` | Minimum score for a row to be eligible for validation |
| `val_holdout_modulus` | `200` | A cluster is held out for validation when `hash(cluster) % modulus == 0` |
| `seed` | `25` | Base seed; combined with the epoch number to vary sampling |
| `num_proc` | `null` | Processes for the one-time split pass. Not part of the cache key, so it can differ between runs |
| `cluster_index_cache_dir` | `null` | Where to cache cluster→row-index maps (defaults to the HF datasets cache) |
| `prepare_only` | `false` | Build the caches and exit without training (see [Building the dataset cache first](#building-the-dataset-cache-first-cpu-only)) |
| `local_stage_dir` | `null` | Node-local dir to copy sequence-only splits to before training (see [Staging data to node-local disk](#staging-data-to-node-local-disk-network-filesystems)) |

**Source types**: every entry in `sources` uses one flat schema (`DataSourceConfig`), discriminated by its `type` field. This is to allow a CLI dotlist override switch a source's `type` cleanly (see below).

| `type` | Relevant fields | Description |
|---|---|---|
| `hub` | `repo_id`, `subset`, `split`, `revision` | Load from the Hugging Face Hub |
| `parquet` | `path` | Load from a local file, directory, or glob |

All source types also require `cluster_column`: the column holding cluster IDs for that source (e.g. `cluster_rep_at_30` from the data pipeline). It's set per source rather than once for the whole dataset because different sources may be clustered at different identity thresholds and store cluster IDs under different column names.

Each source also sets `split_column` and `extra_cluster_columns` so one cache serves several thresholds — see [Reusing one cache across cluster thresholds](#reusing-one-cache-across-cluster-thresholds).

Both accept `sampling_fraction` (default `1.0`), which specifies the fraction of a dataset's clusters to retain each epoch, allowing to adjust how much each source contributes to the final mix. A source with `sampling_fraction: 0.0` is skipped entirely during Phase 1 (its path/repo is never touched), so **removing a source is just a CLI override**, no file edit needed:

```bash
dataset.sources.bfd.sampling_fraction=0
```

**Switching a source's `type`** (e.g. `config.yaml`'s Hub default → local Parquet, or vice versa) is also a pure CLI override. Set the new `type` and its required field, the previous type's now-unused fields (e.g. a stale `repo_id` after switching to `parquet`) are simply ignored, not rejected:

```bash
dataset.sources.uniref.type=parquet \
dataset.sources.uniref.path=data_subsets/uniref_assembled/*.parquet
```

**Adding a brand-new source name** not present in `config.yaml` also works purely via CLI, as dotlist overrides can build an entirely new nested `dataset.sources.<name>.*` block from scratch:

```bash
dataset.sources.custom.type=parquet \
dataset.sources.custom.path=data_subsets/custom_assembled/*.parquet \
dataset.sources.custom.cluster_column=cluster_rep_at_30
```

**Input:** assembled Parquet from `modules/data` (or any HF dataset with sequence/cluster/score columns)
**Output:** in-memory `datasets.Dataset` per split, rebuilt per epoch for train

---

## DataLoader

**Purpose:** Batch the epoch dataset and hand it to the trainer. Supports two mutually exclusive batching modes:

- **Token-budget batching** (`max_tokens`) — a `TokenBatchSampler` groups sequences so each batch holds roughly a fixed number of *tokens* rather than a fixed number of sequences. This keeps GPU memory stable despite highly variable protein lengths (recommended mode).
- **Fixed batching** (`batch_size`) — a constant number of sequences per batch.

**Key config fields (`dataloader`):**

| Field | Default | Description |
|---|---|---|
| `max_tokens` | `32768` | Token budget per batch (mutually exclusive with `batch_size`) |
| `batch_size` | `null` | Sequences per batch |
| `dataloader_num_workers` | `8` | PyTorch worker processes |
| `persistent_workers` | `true` | Keep workers alive between epochs |
| `pin_memory` | `true` | Copy tensors into pinned memory for faster host→device transfer |
| `prefetch_factor` | `4` | Batches prefetched per worker |
| `in_order` | `true` | `false` yields batches as workers finish; faster if some batches are slow, but order is non-deterministic |
| `drop_last` | `true` | Drop a trailing uneven batch (keeps ranks balanced in distributed runs) |

**Input:** prepared sources + epoch number
**Output:** a `DataLoader` yielding collated batches

---

## Collator

**Purpose:** Tokenize a list of sequences, apply MLM masking, and assemble the model-ready batch. Two output layouts:

- **Padded** (`packing: false`) — a standard `(B, L)` batch with an attention mask. Works on CPU and GPU.
- **Packed** (`packing: true`) — all sequences concatenated into a flat `(1, T)` buffer with `cu_seqlens` offsets, consumed by variable-length attention. It uses optional FlashAttention 4 on Hopper and newer GPUs when installed, otherwise PyTorch's native kernel. **GPU only**. Eliminates padding waste.

**Masking schedules** (`masking_type`) control the per-batch masking rate:

| Type | Behavior |
|---|---|
| `fixed` | Always `mlm_probability` (default 0.15) |
| `beta` | Sampled per batch from a Beta distribution shaped by `masking_k` |
| `cosine` | Cosine-shaped schedule; averages well below `mlm_probability` |

**Key config fields (`collator`):**

| Field | Default | Description |
|---|---|---|
| `mlm` | `true` | Enable masked language modeling |
| `mlm_probability` | `0.15` | Base masking probability |
| `masking_type` | `fixed` | `fixed`, `beta`, or `cosine` |
| `masking_k` | `10` | Concentration parameter for the beta schedule |
| `max_length` | `512` | Truncation length |
| `packing` | `true` | Emit a packed variable-length attention batch instead of a padded one |
| `pad_to_multiple_of` | `null` | Round padded length up to a multiple of this value |
| `random_truncate` | `true` | Sample a random window when truncating, instead of always taking the prefix |
| `exclude_special_tokens_from_masking` | `true` | Never mask BOS/EOS/PAD |

**Input:** list of dataset rows
**Output:** dict of tensors (`input_ids`, `labels`, `attention_mask`, `position_ids`, plus `cu_seqlens`/`max_seqlen` when packed) — keys match the model's forward signature, so `model(**batch)` works directly

---

## Model & Tokenizer

**Purpose:** The AMPLIFY encoder — a BERT-style bidirectional transformer with modern components: **RMSNorm**, **rotary position embeddings (RoPE)**, and a **SwiGLU** feed-forward block. It is a standard HF `PreTrainedModel`, split across `configuration_amplify.py` and `modeling_amplify.py` following HF conventions, with MLM, sequence-classification, and token-classification heads.

The tokenizer is **character-level**: one token per amino acid, plus special tokens. Ambiguous residues (`X`, `B`, `O`, `U`, `Z`, `J`) are removed by default.

**Key config fields (`model`):**

| Field | Default | Description |
|---|---|---|
| `hidden_size` | `960` | Hidden dimension |
| `num_hidden_layers` | `32` | Encoder blocks |
| `num_attention_heads` | `15` | Attention heads |
| `intermediate_size` | `2560` | SwiGLU inner dimension |
| `max_position_embeddings` | `2048` | Maximum context length supported by RoPE |
| `rope_theta` | `10000.0` | RoPE base frequency |
| `vocab_size` | `32` | Must match the tokenizer vocabulary |
| `norm_eps` | `1e-5` | RMSNorm epsilon |
| `embedding_init_range` / `decoder_init_range` | `0.02` | Uniform init bounds |

The defaults above give a **~354 M parameter** model. We shrink `hidden_size` / `num_hidden_layers` for the smoke tests mentionned above.

**Key config fields (`tokenizer`):** `vocab` (ordered list; index = token ID), `vocab_size`, the special tokens (`pad`/`unk`/`mask`/`bos`/`eos`), `ambiguous_tokens`, and `remove_ambiguous`.

The model has two attention paths, selected by the collator's `packing` flag: **SDPA** for padded batches and **variable-length attention** for packed batches (GPU only). Packed attention automatically uses optional FlashAttention 4 (FA4) on Hopper and newer GPUs (compute capability 9.0+, such as H100 and B200) when its `flash_attn.cute` kernel is installed; older hardware falls back to native PyTorch. This keeps FlashAttention out of the required dependencies while enabling FA4 on the hardware that benefits from it. All packed kernels require BF16 or FP16 inputs.

---

## Optimizer & Scheduler

**Purpose:** Build the optimizer with correct parameter grouping and the learning-rate schedule.

The optimizer splits parameters into **decay** and **no-decay** groups: 1-D parameters (biases, norm weights) and embeddings are excluded from weight decay, which is standard practice for transformer pretraining.

**Key config fields (`optimizer`):**

| Field | Default | Description |
|---|---|---|
| `type` | `AdamW` | One of `AdamW`, `Adam`, `Adafactor`, `Lamb` |
| `lr` | `1e-3` | Peak learning rate |
| `betas` | `[0.9, 0.95]` | Adam beta parameters |
| `eps` | `1e-8` | Numerical stability epsilon (Adafactor expects a tuple) |
| `weight_decay` | `0.01` | Applied to the decay group only |
| `fused` | `true` | Fused kernel — **CUDA only**, set `false` on CPU |

**Key config fields (`scheduler`):**

| Field | Default | Description |
|---|---|---|
| `type` (alias of `lr_scheduler_type`) | `cosine_with_min_lr` | Any HF `get_scheduler` type |
| `num_warmup_steps` | `500` | Linear warmup steps |
| `num_training_steps` | `8400` | Total steps the schedule decays over |
| `kwargs` (alias of `scheduler_specific_kwargs`) | `{min_lr_rate: 0.1}` | Scheduler-specific arguments |

`SchedulerConfig` sets `populate_by_name=True`, so either the short YAML alias or the full field name is accepted.

---

## Trainer

**Purpose:** Own the training lifecycle: the epoch loop, gradient accumulation, clipping, scheduler stepping, evaluation, checkpointing, and resumption. Built on Accelerate, so single-GPU, multi-GPU (DDP/FSDP), and CPU runs use the same code path.

**Notes:**

- **Per-epoch mixtures** — the trainer requests a fresh dataloader from a provider callable each epoch, which is what drives the curriculum and cluster resampling.
- **Resumption** — `resume_from_checkpoint` restores model, optimizer, scheduler, RNG, and the metric logger's counters, then skips already-consumed batches within the epoch.
- **SIGTERM handling** — saves a checkpoint and exits cleanly, for preemptible SLURM jobs.
- **Two checkpoint types, fully decoupled** — resumable state and portable HF checkpoints save on independent schedules (`save_steps`/`hf_save_steps`) and never trigger each other; neither saves automatically every epoch, only on their step schedule plus one guaranteed save at the very end of training.
- **Checkpoint rotation** — `save_total_limit`/`hf_save_total_limit` cap how many checkpoints of each type are kept, deleting the oldest as new ones are saved.

**Key config fields (`trainer`):**

| Field | Default | Description |
|---|---|---|
| `num_epochs` | `10` | Number of epochs (ignored when `max_steps` is set) |
| `max_steps` | `null` | Stop after N optimizer steps; overrides `num_epochs` |
| `max_grad_norm` | `1.0` | Gradient clipping norm |
| `gradient_accumulation_steps` | `8` | Micro-batches per optimizer step; must match the `Accelerator` |
| `output_dir` | `runs/350M` | Root for all run artifacts |
| `resume_from_checkpoint` | `null` | Checkpoint directory to resume from |
| `logging_steps` | `100` | Log metrics every N optimizer steps |
| `eval_steps` | `500` | Run validation every N steps |
| `save_steps` | `500` | Save resumable state every N steps |
| `hf_save_steps` | `2000` | Save a portable HF checkpoint every N steps |
| `save_total_limit` | `null` | Max resumable checkpoints kept; oldest deleted beyond this. `null` = unlimited |
| `hf_save_total_limit` | `null` | Max HF checkpoints kept; oldest deleted beyond this. `null` = unlimited |
| `tf32` | `true` | Enable TF32 matmuls |
| `compile*` | `false` | `torch.compile` settings — see [torch.compile](#torchcompile) |

---

## Metric Logger

**Purpose:** Accumulate training metrics **on-device** across all ranks and reduce them with a single all-reduce at flush time rather than syncing every step, keeping metric tracking off the critical path. Metrics go to `accelerator.log` (forwarded to W&B when enabled) and to `metrics.jsonl`; the counters are checkpointed, so they survive resumption.

**Metrics tracked:** `train/loss`, `train/perplexity`, `train/accuracy`, `train/learning_rate`, `train/grad_norm`, `train/weight_norm`, `train/tokens`, `train/masked_tokens`, `train/samples`, `train/epoch`, `train/global_step`, `train/tokens_per_second`, `train/steps_per_second`, and `eval/<split>/{loss,perplexity,accuracy}`.

**Key config fields (`wandb`):**

| Field | Default | Description |
|---|---|---|
| `enabled` | `false` | Enable W&B logging via Accelerate trackers |
| `project` | `flair-pretrain` | W&B project name |
| `entity` | `null` | W&B team/user |
| `run_name` | `"AMPLIFY_350M"` | Display name for the run |
| `tags` | `["350M", "pretrain"]` | Tags attached to the run |
| `mode` | `offline` | `online`, `offline`, or `disabled`. Defaults to `offline` for nodes without internet access; sync later with `wandb sync <dir>` |
| `dir` | `null` | Local directory for run data; defaults to `<trainer.output_dir>/wandb` if unset |
| `id` | `null` | Explicit W&B run id; defaults to a slug of `run_name` for deterministic resume |
| `resume` | `allow` | Resume mode passed to `wandb.init` alongside `id` (`allow`, `must`, `never`, ...) |

---

## Running a Pretraining Run

The entry point is `modules/pretrain/src/run_pretrain.py`. It accepts one or more YAML config files followed by optional `key=value` dotlist overrides; multiple files merge left-to-right (later wins), the same convention as `modules/data` and `modules/evaluation`.

### Config only

```bash
uv run modules/pretrain/src/run_pretrain.py modules/pretrain/configs/config.yaml
```

### Config + CLI overrides

```bash
uv run modules/pretrain/src/run_pretrain.py \
    modules/pretrain/configs/config.yaml \
    trainer.output_dir=/scratch/test_run \
    optimizer.lr=5e-4
```

### Base config merged with an experiment config

Merge a small overlay on top of the base config. `120M.yaml`/`smoke_gpu.yaml` are both
overlays on `config.yaml`, pass them together, e.g.:

```bash
uv run modules/pretrain/src/run_pretrain.py \
    modules/pretrain/configs/config.yaml \
    modules/pretrain/configs/120M.yaml
```

### Multi-GPU with Accelerate

Launch through `accelerate` rather than `python` directly; the module picks up the distributed environment automatically:

```bash
uv run accelerate launch --multi_gpu --num_processes 4 \
    modules/pretrain/src/run_pretrain.py \
    modules/pretrain/configs/config.yaml
```

### Resuming an interrupted run

```bash
uv run modules/pretrain/src/run_pretrain.py \
    modules/pretrain/configs/config.yaml \
    trainer.resume_from_checkpoint=./checkpoints/checkpoint_epoch_3_step_5000
```

### New config from scratch

Copy `modules/pretrain/configs/config.yaml` as a starting point (or any other yaml config), as it contains every section with working defaults, then edit the sections you need.

---

## torch.compile

Off by default. Enable with `trainer.compile=true`; Accelerate applies it **after**
the DDP wrap, so gradient all-reduce still overlaps with backward compute.

```bash
uv run accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 \
    modules/pretrain/src/run_pretrain.py \
    modules/pretrain/configs/config.yaml modules/pretrain/configs/120M.yaml \
    trainer.compile=true trainer.compile_dynamic=true trainer.max_steps=50
```

Gains come from fusing the elementwise glue (RMSNorm, SwiGLU, rotary, residual
adds), not attention. The packed path also uses plain PyTorch for SwiGLU and
rotary so those operations can be fused with the surrounding elementwise math.

| Field | Default | Description |
|---|---|---|
| `compile` | `false` | Compile the model with the inductor backend |
| `compile_mode` | `default` | `default`, `max-autotune`, or `max-autotune-no-cudagraphs`. Avoid `reduce-overhead` (CUDA graphs need static shapes) |
| `compile_dynamic` | `null` | `true` compiles one shape-agnostic graph; `null` lets Dynamo detect dynamism; `false` forces static shapes |
| `compile_fullgraph` | `false` | Error on graph breaks — diagnostic only |

**Always set `compile_dynamic: true` when `dataloader.max_tokens` is set.** Token-budget
batching gives every batch a different token count and `max_seqlen` (a Python int
Dynamo guards on), so static shapes recompile continuously and run *slower* than eager.
Verify on the first run: a quiet log means the shapes settled:

```bash
TORCH_LOGS=recompiles uv run ... trainer.compile=true trainer.max_steps=50
```

---

## Config System

All config classes are Pydantic `BaseModel` subclasses composed into a single `PretrainConfig` (`src/config.py`):

```
PretrainConfig
├── model:      AMPLIFYModelConfig
├── tokenizer:  TokenizerConfig
├── optimizer:  OptimizerConfig
├── scheduler:  SchedulerConfig
├── collator:   CollatorConfig
├── dataset:    DatasetConfig
│   └── sources: dict[str, LocalDataSource | HubDataSource]
│   └── score_curriculum: CurriculumConfig
├── dataloader: DataLoaderConfig
├── trainer:    TrainerConfig
└── wandb:      WandbConfig
```

Every config class sets `extra="forbid"`, so a typo in a config key fails loudly at load time rather than being silently ignored:

```
$ ... collator.mlm_probabilty=0.5
ValidationError: collator.mlm_probabilty
  Extra inputs are not permitted [type=extra_forbidden]
```

The **Default** column in the component tables above shows the value shipped in
`configs/config.yaml`, which is not always the schema default (e.g. `trainer.num_epochs`
is `10` in the config, `1` in the schema).

---
