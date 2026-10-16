# `modules/pretrain` — Pretraining Module

This module pretrains AMPLIFY, a BERT-style protein language model, with masked
language modeling (MLM).

It sits between the two other modules of this repo:

```
modules/data            →  modules/pretrain        →  modules/evaluate
clustered, scored          MLM training, outputs       downstream benchmarks
protein datasets           HF checkpoints              on those checkpoints
```

A run goes through five main stages:

```
Prepare sources   →  Plan the epoch  →  Batch, tokenize  →  Train loop  →  Checkpoints
(once, cached)       (every epoch)       and mask                            and metrics
```

## Table of Contents

- [Key Concepts](#key-concepts)
- [Input Data](#input-data)
- [Quick Start](#quick-start)
  - [Single GPU](#single-gpu)
  - [Multi-GPU](#multi-gpu)
  - [What to expect](#what-to-expect)
  - [Artifacts](#artifacts)
- [Running and Configuring](#running-and-configuring)
  - [Config files](#config-files)
  - [Resuming an interrupted run](#resuming-an-interrupted-run)
  - [Batch size and steps](#batch-size-and-steps)
  - [Common recipes](#common-recipes)
  - [Config schema](#config-schema)
- [What Happens During a Run](#what-happens-during-a-run)
- [Code Layout](#code-layout)
- [Dataset](#dataset)
- [DataLoader](#dataloader)
- [Collator](#collator)
- [Model & Tokenizer](#model--tokenizer)
- [Optimizer & Scheduler](#optimizer--scheduler)
- [Trainer](#trainer)
- [Metric Logger](#metric-logger)
- [Scaling Up to Full Runs](#scaling-up-to-full-runs)
  - [Real pretraining runs](#real-pretraining-runs)
  - [Launching on a SLURM cluster (e.g. Mila cluster)](#launching-on-a-slurm-cluster-eg-mila-cluster)
  - [Building the dataset cache first (CPU-only)](#building-the-dataset-cache-first-cpu-only)
  - [Staging data to node-local disk (network filesystems)](#staging-data-to-node-local-disk-network-filesystems)
  - [Random-read advice (`madvise_random`)](#random-read-advice-madvise_random)
  - [What triggers a cache rebuild](#what-triggers-a-cache-rebuild)
  - [Reusing one cache across cluster thresholds](#reusing-one-cache-across-cluster-thresholds)
  - [Copying a built cache to another cluster](#copying-a-built-cache-to-another-cluster)
  - [torch.compile](#torchcompile)
- [Outputs](#outputs)
  - [Output disk layout](#output-disk-layout)
  - [Checkpoint saving, decoupling, and rotation](#checkpoint-saving-decoupling-and-rotation)
  - [Uploading a checkpoint to the Hugging Face Hub](#uploading-a-checkpoint-to-the-hugging-face-hub)
  - [W&B run resumption](#wb-run-resumption)
- [Using a Trained Model with Hugging Face](#using-a-trained-model-with-hugging-face)
- [Troubleshooting](#troubleshooting)
- [Testing](#testing)

---

## Key Features

- **One sequence per cluster per epoch.** `modules/data` groups similar sequences into
  clusters at several identity thresholds (columns like `cluster_rep_at_30`, meaning
  ≥30% sequence identity). Protein databases are highly redundant, so training on every
  row would over-weight large families. Each epoch picks one random sequence from every
  cluster. The model sees every family each epoch, and different members of a family across
  epochs.
- **Score curriculum.** Each sequence has a quality score (RED, from `modules/data`).
  A minimum-score threshold can be applied over epochs, so training starts on all the data
  and narrows to the best sequences towards the end.
- **Validation holdout.** Whole clusters are held out for validation. This avoids
  leakage: two similar proteins from the same cluster can never end up one in train
  and the other in val, which would inflate validation scores. The val set also stays
  the same across runs and epochs.
- **Masked language modeling (MLM).** A fraction of the residues in each sequence (set by
  the collator config) is replaced by `<mask>`. The loss is computed on those
  positions only.
- **Token-budget batches.** Protein lengths vary a lot, so batches hold a fixed number
  of tokens (`max_tokens`) rather than a fixed number of sequences, which keeps GPU
  memory stable.
- **Packing.** By default, the sequences of a batch are concatenated into one row with
  no padding, and attention is restricted to each sequence (variable-length attention).

---

## Input Data

Each data source is a table (local Parquet files or a HuggingFace Hub dataset),
normally the assembled output of [`modules/data`](../data/README.md). Ready-to-use
assembled datasets are on the Hub (see `flair-bio/uniref`, `flair-bio/bfd`,
`flair-bio/mgnify`, `flair-bio/oas`) and are also the defaults in `configs/config.yaml`.

| Column | Required | Description |
|---|---|---|
| `sequence` | Yes | The amino-acid sequence |
| `cluster_column` (e.g. `cluster_rep_at_30`) | Yes | Cluster ID used to pick one sequence per cluster each epoch |
| `split_column` | Yes (defaults to `cluster_column`) | Cluster ID used for the train/val holdout |
| `score_column` (default `red_score`) | Yes | Quality score used by the curriculum and `val_min_score` |
| `extra_cluster_columns` | Only if configured | Other thresholds kept in the cache, so `cluster_column` can be switched without a full rebuild |
| `sequence_length` | No | Computed from `sequence` if absent |

Other columns are ignored. A missing required column fails at load time with a
`KeyError` naming it.

---

## Quick Start

**Requirements:** one or more CUDA GPUs with bf16 support (e.g. A100, H100, B200), and
HuggingFace Hub access to download the dataset. Install from the `flair-plm` root:

```bash
uv sync --extra pretrain
```

FlashAttention 4 is an **optional** speed-up on Hopper and newer GPUs (see the
[main README](../../README.md)); PyTorch's native kernels are used otherwise.

The smoke test merges `configs/smoke_gpu.yaml` on top of `configs/config.yaml`: a
small model (4 layers, hidden size 256) trained for 150 steps on OAS from the Hub
(~1.9M sequences). A single command downloads OAS, builds its cache and trains. No
separate cache-building step is needed at this size, and later runs reuse the cache.
Run commands from the repo root with the venv activated (or prefix them with `uv run`).

### Single GPU

To run the pretraining smoke test on a single GPU:

```bash
accelerate launch --mixed_precision bf16 \
    modules/pretrain/src/run_pretrain.py \
    modules/pretrain/configs/config.yaml \
    modules/pretrain/configs/smoke_gpu.yaml
```

### Multi-GPU

To run the pretraining smoke test on 4 GPUs:

```bash
accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 \
    modules/pretrain/src/run_pretrain.py \
    modules/pretrain/configs/config.yaml \
    modules/pretrain/configs/smoke_gpu.yaml
```

### Artifacts

Written under `trainer.output_dir` (`smoke_runs/gpu` by default):

```
smoke_runs/gpu/
├── metrics.jsonl                     ← one JSON record per logging flush
├── checkpoint_epoch_0_step_50/       ← resumable Accelerate state
│   ├── model.safetensors, optimizer.bin, scheduler.bin
│   ├── random_states_0.pkl, sampler*.bin
│   ├── custom_checkpoint_0.pkl       ← metric logger counters
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

## Running and Configuring

### Config files

| File | Use |
|---|---|
| `configs/config.yaml` | Base config: every section, with the 350M model and the four Hub datasets. Runnable on its own |
| `configs/120M.yaml` | Overlay for the 120M model (only the fields that differ) |
| `configs/smoke_gpu.yaml` | Overlay for the quick smoke test to validate the setup |

`run_pretrain.py` takes one or more YAML files, merged left to right (later files
win), followed by optional `key=value` overrides applied last. Same convention as
`modules/data` and `modules/evaluate`. To write a new experiment, create a small
overlay with only the fields you change, rather than copying `config.yaml`.

```bash
accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 \
    modules/pretrain/src/run_pretrain.py \
    modules/pretrain/configs/config.yaml \
    modules/pretrain/configs/120M.yaml \
    trainer.output_dir=runs/my_run optimizer.lr=5e-4
```

### Resuming an interrupted run

Add `trainer.resume_from_checkpoint=<output_dir>/checkpoint_epoch_<E>_step_<S>` to
the same command. Model, optimizer, scheduler, RNG and metric counters are restored,
and already-consumed batches of the current epoch are also skipped.

### Batch size and steps

Note that `dataloader.max_tokens` is **per GPU**: each epoch is split into batches of at
most `max_tokens` tokens, and each GPU processes its own batch. Tokens per optimizer step are:

```
max_tokens × gradient_accumulation_steps × num_processes
```

Both default configs give ~1M tokens per step on 4 GPUs (350M: 32768 × 8 × 4;
120M: 65536 × 4 × 4). When changing the GPU count, adjust `gradient_accumulation_steps`
to keep this constant. `trainer.max_steps` sets the run length (`num_epochs` is
ignored when it is set), and `scheduler.num_training_steps` sets the LR decay length;
past it, the LR stays at its final value.

### Common recipes

Append these overrides to the launch command above, or put them in an overlay YAML.

| Goal | Overrides |
|---|---|
| Train on one source only (e.g. UniRef) | `dataset.sources.bfd.sampling_fraction=0 dataset.sources.mgnify.sampling_fraction=0 dataset.sources.oas.sampling_fraction=0` |
| Use local parquet instead of the Hub | `dataset.sources.uniref.type=parquet dataset.sources.uniref.path='/path/to/uniref_assembled/*.parquet'` |
| Use the 120M model | Add `modules/pretrain/configs/120M.yaml` after `config.yaml` |
| Short test run | `trainer.max_steps=500 trainer.eval_steps=100 trainer.save_steps=100` |
| Debug without `torch.compile` | `trainer.compile=false` |
| Log to W&B online | `wandb.enabled=true wandb.mode=online wandb.run_name=my_run` |
| Fit in less GPU memory, same tokens per step | `dataloader.max_tokens=16384 trainer.gradient_accumulation_steps=16` (default 32768 × 8) |

### Config schema

All config classes are Pydantic models composed into a single `PretrainConfig` (`src/config.py`):

```
PretrainConfig
├── model:      AMPLIFYModelConfig
├── tokenizer:  TokenizerConfig
├── optimizer:  OptimizerConfig
├── scheduler:  SchedulerConfig
├── collator:   CollatorConfig
├── dataset:    DatasetConfig
│   ├── sources: dict[str, DataSourceConfig]
│   └── score_curriculum: CurriculumConfig
├── dataloader: DataLoaderConfig
├── trainer:    TrainerConfig
├── ddp:        DDPConfig
└── wandb:      WandbConfig
```

Every config class sets `extra="forbid"`, so a typo in a config key fails loudly at load time rather than being silently ignored:

```
$ ... collator.mlm_probabilty=0.5
ValidationError: collator.mlm_probabilty
  Extra inputs are not permitted [type=extra_forbidden]
```

The **Default** columns below show the values in `configs/config.yaml`, which can
differ from the schema defaults (e.g. `trainer.compile` is `true` in the config,
`false` in the schema).

---

## What Happens During a Run

`run_pretrain.py` wires the components together, then hands over to the `Trainer`:

1. **Load the config**: merge the YAML files and overrides, validated into `PretrainConfig`.
2. **Set up the runtime**: create the Accelerate `Accelerator` (GPUs, mixed precision,
   DDP, `torch.compile`) and seed the RNGs. With `dataset.prepare_only=true`, it builds
   the data caches and exits here.
3. **Build the components**: model, tokenizer, optimizer, scheduler, collator.
4. **Prepare the sources (phase 1)**: load each source, split it into train/val by
   cluster, and build its cluster index. Cached to disk: on each node the first rank
   builds it while the others wait, then reuse it.
5. **Build the validation dataloader**: once, since the val set never changes.
6. **Train**: at the start of each epoch, the trainer asks for a new train dataloader
   (phase 2: one sequence per cluster, curriculum threshold, mixing the sources),
   then loops over its batches, evaluating, logging and checkpointing on the
   configured step intervals.

For each training step, one batch goes through:

```
TokenBatchSampler  →  dataset rows    →  DataCollator            →  model         →  loss, backward,
(row ids under        (sequences)        (tokenize, truncate,       (AMPLIFY MLM)     optimizer step,
 max_tokens)                              mask, pack)                                 metrics
```

---

## Code Layout

```
modules/pretrain/
├── README.md                        ← this file
│
├── configs/
│   ├── config.yaml                  ← default config, every section explicit; this
│   │                                  is the 350M config (standalone, runnable as-is)
│   ├── 120M.yaml                    ← 120M-specific overlay on config.yaml (118.3M params), 4x A100
│   └── smoke_gpu.yaml               ← CUDA smoke overlay on config.yaml (small model, OAS from the Hub)
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

## Dataset

**Purpose:** Turn one or more assembled datasets (from `modules/data`) into a per-epoch training mixture. Runs in **two phases**:

1. **Prepare (once, cached)** — for each source: load the Parquet/Hub dataset, split clusters into train/val, and build a `cluster → row indices` index sorted by score. No score filtering happens here: that is done per epoch in phase 2. Cached to disk so repeated runs and multi-GPU ranks don't rebuild it (see [What triggers a cache rebuild](#what-triggers-a-cache-rebuild)).
2. **Build epoch mixture (once per epoch, NumPy only)** — keep rows meeting this epoch's score curriculum threshold, pick one random representative per cluster (varying with the epoch seed), subsample each source's clusters by its `sampling_fraction` (a fresh random subset each epoch), then concatenate and shuffle. The result is an `EpochPlan` of row indices into a `ConcatDataset` built once over all sources, so no dataset is rewritten per epoch.

The `score` column comes from `modules/data`'s score step (see [its README](../data/README.md#step-4--score)): a per-sequence **RED** (Residue Embedding Diversity) score from an existing pretrained pLM, where higher means the sequence's embeddings are more informationally distinct (can be seen as a quality/diversity signal).

See [Key Concepts](#key-concepts) for the ideas behind clusters and the curriculum.
Within a cluster, each epoch picks uniformly among the rows passing that epoch's score
threshold, independently of earlier epochs, so a row in a cluster of size *n* is picked
with probability 1/*n* per epoch. The curriculum threshold ramps from `start_score` to
`end_score` over `ramp_epochs` (`linear`, `cosine` or `exponential`), then holds.

**Key config fields (`dataset`):**

| Field | Default | Description |
|---|---|---|
| `sources` | Hub defaults for `bfd`/`mgnify`/`uniref`/`oas` | Mapping of name → data source (see below); at least one required. Overridable per source via CLI dotlist without editing `config.yaml` — remove a source (`sampling_fraction=0`), switch its `type` (`hub`↔`parquet`), or add a new source name entirely (see [Real pretraining runs](#real-pretraining-runs)). An empty/all-zero-fraction `sources` raises a clear `ValueError` at startup. |
| `score_column` | `red_score` | Column holding quality scores (`RED` from the data pipeline) |
| `score_curriculum` | `type: linear, start_score: 0.0, end_score: 0.0, ramp_epochs: 1` | `type` (`linear`/`cosine`/`exponential`), `start_score`, `end_score`, `ramp_epochs` |
| `val_min_score` | `0.0` | Minimum score for a row to be eligible for validation |
| `val_holdout_modulus` | `200` | A cluster is held out for validation when `hash(cluster) % modulus == 0` |
| `seed` | `25` | Base seed; combined with the epoch number to vary sampling |
| `num_proc` | `12` | Processes for the one-time cache build: the initial parquet/hub load into Arrow (parallel over files) and the split pass. Not part of the cache key, so it can differ between runs |
| `split_writer_batch_size` | `100000` | Rows per Arrow batch written by the split pass. Not part of the cache key |
| `cluster_index_cache_dir` | `null` | Where to cache cluster→row-index maps (defaults to the HF datasets cache) |
| `prepare_only` | `false` | Build the caches and exit without training (see [Building the dataset cache first](#building-the-dataset-cache-first-cpu-only)) |
| `export_dir` | `null` | With `prepare_only`, write each source as a portable folder loadable with `type: prepared` (see [Copying a built cache to another cluster](#copying-a-built-cache-to-another-cluster)) |
| `local_stage_dir` | `null` | Node-local dir to copy sequence-only splits to before training (see [Staging data to node-local disk](#staging-data-to-node-local-disk-network-filesystems)) |

**Source types**: every entry in `sources` uses one flat schema (`DataSourceConfig`), discriminated by its `type` field. This is to let a CLI dotlist override switch a source's `type` cleanly (see below).

| `type` | Relevant fields | Description |
|---|---|---|
| `hub` | `repo_id`, `subset`, `split`, `revision` | Load from the Hugging Face Hub |
| `parquet` | `path` | Load from a local file, directory, or glob |
| `prepared` | `path` | Load a folder written by `export_dir` (see [Copying a built cache to another cluster](#copying-a-built-cache-to-another-cluster)) |

All source types also require `cluster_column`: the column holding cluster IDs for that source (e.g. `cluster_rep_at_30` from the data pipeline). It's set per source rather than once for the whole dataset because different sources may be clustered at different identity thresholds and store cluster IDs under different column names.

Each source also sets `split_column` and `extra_cluster_columns` so one cache serves several thresholds — see [Reusing one cache across cluster thresholds](#reusing-one-cache-across-cluster-thresholds).

`madvise_random` (default `true`) turns off kernel readahead on that source's memory-mapped splits. Set it to `false` for sources small enough to stay in page cache — see [Random-read advice](#random-read-advice-madvise_random).

All source types accept `sampling_fraction` (default `1.0`), which specifies the fraction of a dataset's clusters to draw **per epoch**, allowing to adjust how much each source contributes to the final mix. The whole source is still prepared and memory-mapped once; a different random subset of clusters is drawn each epoch (seeded by `seed` + epoch), so over several epochs most clusters are eventually seen. Note that a run ending within one epoch (typical with `max_steps`) only sees that single subset. A source with `sampling_fraction: 0.0` is skipped entirely during Phase 1 (its path/repo is never touched), so **removing a source is just a CLI override**, no file edit needed:

```bash
dataset.sources.bfd.sampling_fraction=0
```

**Switching a source's `type`** (e.g. `config.yaml`'s Hub default → local Parquet, or vice versa) is also a pure CLI override. Set the new `type` and its required field, the previous type's now-unused fields (e.g. a stale `repo_id` after switching to `parquet`) are simply ignored, not rejected:

```bash
dataset.sources.uniref.type=parquet \
dataset.sources.uniref.path=/path/to/uniref_assembled/*.parquet
```

**Adding a brand-new source name** not present in `config.yaml` also works purely via CLI, as dotlist overrides can build an entirely new nested `dataset.sources.<name>.*` block from scratch:

```bash
dataset.sources.custom.type=parquet \
dataset.sources.custom.path=/path/to/custom_assembled/*.parquet \
dataset.sources.custom.cluster_column=cluster_rep_at_30
```

**Input:** assembled Parquet from `modules/data` (or any HF dataset with sequence/cluster/score columns)
**Output:** a persistent `ConcatDataset` over the prepared sources, plus an `EpochPlan` (row ids and lengths) per epoch for train and a fixed one for val

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

**Masking schedules** (`masking_type`) control the masking rate, sampled per sequence. Selected positions are always replaced by `<mask>`:

| Type | Behavior |
|---|---|
| `fixed` | Always `mlm_probability` (default 0.15) |
| `beta` | Sampled per sequence from a Beta distribution centred on `mlm_probability`, shaped by `masking_k` |
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

Model sizes of the shipped configs:

| Config | `hidden_size` | `num_hidden_layers` | `num_attention_heads` | `intermediate_size` | Parameters |
|---|---|---|---|---|---|
| `smoke_gpu.yaml` | 256 | 4 | 8 | 512 | 2.6M |
| `120M.yaml` | 640 | 24 | 10 | 1712 | 118.3M |
| `config.yaml` (350M) | 960 | 32 | 15 | 2560 | 354.0M |

**Key config fields (`tokenizer`):** `vocab` (ordered list; index = token ID), `vocab_size`, the special tokens (`pad`/`unk`/`mask`/`bos`/`eos`), `ambiguous_tokens`, and `remove_ambiguous`.

The model has two attention paths, selected by the collator's `packing` flag: **SDPA** for padded batches and **variable-length attention** for packed batches (GPU only). Packed attention automatically uses optional FlashAttention 4 (FA4) on Hopper and newer GPUs (such as H100 and B200) when its `flash_attn.cute` kernel is installed; older hardware falls back to native PyTorch. This keeps FlashAttention out of the required dependencies while enabling FA4 on the hardware that benefits from it. All packed kernels require BF16 or FP16 inputs.

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
| `num_warmup_steps` | `1000` | Linear warmup steps |
| `num_training_steps` | `90000` | Total steps the schedule decays over |
| `kwargs` (alias of `scheduler_specific_kwargs`) | `{min_lr_rate: 0.1}` | Scheduler-specific arguments |

`SchedulerConfig` sets `populate_by_name=True`, so either the short YAML alias or the full field name is accepted.

---

## Trainer

**Purpose:** Own the training lifecycle: the epoch loop, gradient accumulation, clipping, scheduler stepping, evaluation, checkpointing, and resumption. Built on Accelerate, so single-GPU, multi-GPU (DDP), and CPU runs use the same code path.

**Notes:**

- **Per-epoch mixtures** — the trainer requests a fresh dataloader from a provider callable each epoch, which is what drives the curriculum and cluster resampling.
- **Resumption** — `resume_from_checkpoint` restores model, optimizer, scheduler, RNG, and the metric logger's counters, then skips already-consumed batches within the epoch.
- **SIGTERM handling** — saves a checkpoint and exits cleanly, for preemptible SLURM jobs.
- **Two checkpoint types, fully decoupled** — resumable state and portable HF checkpoints save on independent schedules (`save_steps`/`hf_save_steps`) and never trigger each other; neither saves automatically every epoch, only on their step schedule plus one guaranteed save at the very end of training.
- **Checkpoint rotation** — `save_total_limit`/`hf_save_total_limit` cap how many checkpoints of each type are kept, deleting the oldest as new ones are saved.

**Key config fields (`trainer`):**

| Field | Default | Description |
|---|---|---|
| `num_epochs` | `1` | Number of epochs (ignored when `max_steps` is set) |
| `max_steps` | `100000` | Stop after N optimizer steps; overrides `num_epochs` |
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
| `expandable_segments` | `true` | Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation with variable batch shapes |
| `deterministic` | `false` | `torch.use_deterministic_algorithms`; for debugging only (slow) |
| `dist_timeout_minutes` | `90` | Distributed collective timeout; raise for first-time cache builds or staging |
| `compile*` | `true` | `torch.compile` settings — see [torch.compile](#torchcompile) |

**Key config fields (`ddp`):** only used for multi-GPU runs.

| Field | Default | Description |
|---|---|---|
| `broadcast_buffers` | `true` | Sync module buffers at the start of each forward |
| `bucket_cap_mb` | `100` | Gradient bucket size; no effect on a single node |
| `gradient_as_bucket_view` | `false` | Let gradients alias the all-reduce buckets to save memory |
| `static_graph` | `false` | Tell DDP the graph is identical every step |
| `comm_hook` | `none` | `bf16`/`fp16` compress gradients before all-reduce (less traffic, may affect loss) |

---

## Metric Logger

**Purpose:** Accumulate training metrics **on-device** across all ranks and reduce them with a single all-reduce at flush time rather than syncing every step, keeping metric tracking off the critical path. Metrics go to `accelerator.log` (forwarded to W&B when enabled) and to `metrics.jsonl`; the counters are checkpointed, so they survive resumption.

**Metrics tracked:** `train/loss`, `train/perplexity`, `train/accuracy`, `train/learning_rate`, `train/grad_norm`, `train/weight_norm`, `train/tokens`, `train/masked_tokens`, `train/samples`, `train/epoch`, `train/global_step`, `train/tokens_per_second`, `train/steps_per_second`, `eval/<split>/{loss,perplexity,accuracy}`, and dataloader wait times per step (`dataloader/iter_{p50,p90,p99,max}_ms`, `dataloader/slowest_rank`, `dataloader/iter_step_pct`) to spot input-pipeline stalls.

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

## Scaling Up to Full Runs

The smoke test builds its cache inside the training command. For the full datasets,
the steps below avoid wasting GPU time and network I/O.

### Real pretraining runs

Once the smoke test passes, launch a full-size run. `configs/config.yaml` **is** the 350M config (standalone, runnable as-is, and ships with working Hugging Face Hub defaults for `bfd`/`mgnify`/`uniref`/`oas`); for 120M, merge `120M.yaml` on top (see [Running and Configuring](#running-and-configuring) for CLI overrides, resuming, and merging configs). To point at local Parquet data instead of the Hub for a given cluster, override `dataset.sources.*` via CLI dotlist (see [Dataset](#dataset) for the full override options — removing a source, switching its type, or adding a new one):

```bash
uv run accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 \
    modules/pretrain/src/run_pretrain.py \
    modules/pretrain/configs/config.yaml \
    modules/pretrain/configs/120M.yaml \
    dataset.sources.bfd.type=parquet \
    dataset.sources.bfd.path=/path/to/bfd_assembled/*.parquet \
    dataset.sources.bfd.cluster_column=cluster_rep_at_30 \
    dataset.sources.bfd.sampling_fraction=1.0 \
    dataset.sources.mgnify.type=parquet \
    dataset.sources.mgnify.path=/path/to/mgnify_assembled/*.parquet \
    dataset.sources.mgnify.cluster_column=cluster_rep_at_30 \
    dataset.sources.mgnify.sampling_fraction=1.0 \
    dataset.sources.uniref.type=parquet \
    dataset.sources.uniref.path=/path/to/uniref_assembled/*.parquet \
    dataset.sources.uniref.cluster_column=cluster_rep_at_30 \
    dataset.sources.uniref.sampling_fraction=1.0
```

Note: On multi-GPU, the first run against a new/uncached dataset can take longer than PyTorch's
default NCCL collective timeout. If you see a timeout error, build the cache first with `dataset.prepare_only=true` (below), or raise `trainer.dist_timeout_minutes` (default `90`) for that first run. Subsequent runs will reuse the cache and are fast.

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
#SBATCH --gres=gpu:a100l:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G

set -euo pipefail
cd <path_to_flair-plm>

# Activate a pre-built venv (then call `accelerate` directly, not via `uv run`).
source <path_to_venv>/bin/activate

# accelerate starts one process per GPU inside this single Slurm task, so split
# the CPUs between them instead of letting each use all of them.
NUM_PROCESSES=4
export OMP_NUM_THREADS=$(( SLURM_CPUS_PER_TASK / NUM_PROCESSES ))
export TOKENIZERS_PARALLELISM=false

# Must match the cache-build job: the prepared dataset is looked up here.
export HF_HOME=<path_to_shared_hf_cache>
export HF_DATASETS_CACHE="$HF_HOME/datasets"

# Launch pretraining (merging base config and 120M-specific config).
accelerate launch --multi_gpu --num_processes "$NUM_PROCESSES" --mixed_precision bf16 \
    modules/pretrain/src/run_pretrain.py \
    modules/pretrain/configs/config.yaml \
    modules/pretrain/configs/120M.yaml \
    "$@"
```

Create `logs/` before submitting (`mkdir -p logs`): Slurm opens the output files before the script runs.

To submit the job after saving the script above as a shell script (additionally forwarding any config overrides as extra args if needed):

```bash
sbatch <your_script>.sh trainer.max_steps=2000
```

### Building the dataset cache first (CPU-only)

On the full datasets, the one-time cache build (split, flatten, cluster index) can take hours. Doing it inside a real training run wastes GPU hours, and on multi-GPU it can trip the NCCL watchdog: the build is wrapped in `local_main_process_first()`, so the other ranks sit at a NCCL barrier while rank 0 works, and they abort if it takes too long.

This is only needed for large sources; small ones like OAS can build inside the training run, as in the smoke test.

`dataset.prepare_only=true` builds the caches and exits before the model is created, so it needs no GPU and no distributed setup. Run it as a single plain process:

```bash
python modules/pretrain/src/run_pretrain.py \
    modules/pretrain/configs/config.yaml \
    modules/pretrain/configs/120M.yaml \
    dataset.sources.uniref.type=parquet \
    dataset.sources.uniref.path=/path/to/uniref_assembled/*.parquet \
    dataset.sources.uniref.cluster_column=cluster_rep_at_30 \
    dataset.prepare_only=true
```

It finishes with `Data sources prepared. Exiting (dataset.prepare_only).`, and the real run then starts warm (`Cluster index cache hit` per source).

### Staging data to node-local disk (network filesystems)

Training reads rows in random order. If `HF_DATASETS_CACHE` is on a network filesystem that doesn't page-cache those reads (e.g. Mila's `/network/projects`, BeeGFS with `tuneFileCacheType=buffered`), every read is a network round trip. Large sources like BFD then risk starving the GPUs. Set `dataset.local_stage_dir` to a node-local directory to copy each source's sequence-only train/val splits there once per job before training starts:

```bash
    dataset.local_stage_dir=$SLURM_TMPDIR/flair_plm_stage \
    trainer.dist_timeout_minutes=90
```

Copies are named by dataset fingerprint and reused by the other ranks on the node. The copy runs inside `local_main_process_first()`, so raise `trainer.dist_timeout_minutes` to cover it. Leave it unset when the cache is already on local disk.

### Random-read advice (`madvise_random`)

The dataset shards are memory-mapped Arrow files. A random ~140 B row read causes the OS kernel to optimistically read ahead up to ~256 KiB of the file into the page cache. The random nature of the reads means that most result in this page fault. Setting `sources.<name>.madvise_random` (default `true`) calls the Linux `madvise(MADV_RANDOM)` on the dataset file mappings so that each page fault reads a _much_ smaller single page than the default 256 KiB.

* Enable it for datasets too big for the node's memory page cache (BFD, Mgnify). This reduces network scratch read pressure by ~36x.
* Disable it for sources that can all cumulatively fit.

This setting only applies to training. `prepare_only` skips it since building reads sequentially. The readahead is useful prefetch in that read pattern.

```bash
dataset.sources.bfd.madvise_random=true
```

### What triggers a cache rebuild

Each source is cached independently, so changing the source mix (adding, removing, or re-weighting sources) never rebuilds another source's cache. Within a source, the three cache layers are keyed as follows:

| Cache | Rebuilt when you change | Not affected by |
|---|---|---|
| Base load (parquet → Arrow) | The source files (name, size, or modification time; re-syncing files counts), or the Hub revision | Anything below |
| Train/val split (the expensive pass) | The source files, `split_column`, `val_holdout_modulus` | `num_proc`, `split_writer_batch_size`, `madvise_random`, `cluster_column`\*, `score_column`\*, `sampling_fraction`, curriculum |
| Cluster index | Anything that rebuilds the split, plus `cluster_column` or `score_column` | `sampling_fraction`, curriculum |

\* Only if the new column was already carried in the split cache (see [Reusing one cache across cluster thresholds](#reusing-one-cache-across-cluster-thresholds)).

Per-threshold row counts (`valid_count-*.npy`) are cached next to the cluster index, so a new curriculum threshold only adds a quick scan.

Keep `HF_HOME`/`HF_DATASETS_CACHE` identical between runs, on storage visible from every node.

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
* Pinning `split_column` assumes the clusterings are **nested**: every finer cluster (e.g. at 70%) lies entirely within one coarser cluster (e.g. at 30%), so it can never straddle train and val. `modules/data` guarantees this for all sources through cascaded clustering; check it if you bring clusterings produced another way.

### Copying a built cache to another cluster

The HF cache above is tied to the parquet files' absolute paths and modification times, so it can't be reused on another filesystem (e.g. Mila → Compute Canada). Instead, export each source once into a self-contained folder by adding `dataset.export_dir` to a `prepare_only` run:

```bash
    dataset.prepare_only=true \
    dataset.export_dir=/path/to/flair_cache
```

This writes `flair_cache/<source>/` with `train/`, `val/` (every carried column, so any exported cluster threshold still works), `cluster_index/` and `meta.json`. Copy that folder anywhere with `rsync -a`, any number of times, and point the source at it:

```bash
    dataset.sources.uniref.type=prepared \
    dataset.sources.uniref.path=/path/to/flair_cache/uniref \
    dataset.sources.uniref.cluster_column=cluster_rep_at_70
```

No parquet files or HF cache are needed, and the cluster index is reused (`Cluster index cache hit`). A `cluster_column` not used at export time builds its index inside the folder. `split_column` and `val_holdout_modulus` are fixed at export: configuring a different value for a `prepared` source raises an error rather than being silently ignored, and changing them means exporting again from parquet. `local_stage_dir` works as usual.

The export logs, and records under `hf_cache` in `meta.json`, the HF cache folders it was built from: the split Arrow files, the cluster index folders and, for Hub sources, the downloaded files. Once every run uses `type: prepared` (check with one `prepare_only` run that the indexes hit), you can delete exactly those folders to free the space. Don't delete the whole `HF_DATASETS_CACHE`, which other sources share. After that, using `type: parquet`/`hub` again, or re-exporting, rebuilds from scratch.

### Using torch.compile

On in `config.yaml` (`trainer.compile=true`, `compile_dynamic=true`); the schema default is off.
Disable with `trainer.compile=false` (e.g. for CPU or quick debugging). Accelerate applies it
**after** the DDP wrap, so gradient all-reduce still overlaps with backward compute.

Gains come from fusing the elementwise glue (RMSNorm, SwiGLU, rotary, residual
adds), not attention. The packed path also uses plain PyTorch for SwiGLU and
rotary so those operations can be fused with the surrounding elementwise math.

| Field | Default | Description |
|---|---|---|
| `compile` | `true` | Compile the model with the inductor backend |
| `compile_mode` | `default` | `default`, `max-autotune`, or `max-autotune-no-cudagraphs`. Avoid `reduce-overhead` (CUDA graphs need static shapes) |
| `compile_dynamic` | `true` | `true` compiles one shape-agnostic graph; `null` lets Dynamo detect dynamism; `false` forces static shapes |
| `compile_fullgraph` | `false` | Error on graph breaks — diagnostic only |

**Always set `compile_dynamic: true` when `dataloader.max_tokens` is set.** Token-budget
batching gives every batch a different token count and `max_seqlen` (a Python int
Dynamo guards on), so static shapes recompile continuously and run *slower* than eager.
Verify on the first run: a quiet log means the shapes settled:

```bash
TORCH_LOGS=recompiles uv run ... trainer.compile=true trainer.max_steps=50
```

---

## Outputs

### Output disk layout

A training run writes everything under `trainer.output_dir`, including W&B's
local log data by default, so every run is a single self-contained folder:

```
<output_dir>/
├── metrics.jsonl                                  ← append-only metric records (mirrors W&B)
├── wandb/                                         ← local W&B run data (if wandb.enabled)
├── checkpoint_epoch_<E>_step_<S>/                 ← resumable training state
│   ├── model.safetensors                          ← model weights
│   ├── optimizer.bin, scheduler.bin               ← optimizer/scheduler state
│   ├── random_states_*.pkl, sampler*.bin          ← RNG and sampler state for exact resumption
│   ├── custom_checkpoint_0.pkl                    ← metric logger counters
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

Resumable checkpoints are saved **directly** in `output_dir`; portable ones go under `hf_checkpoints/`. Use `checkpoint_epoch_*` to resume an interrupted run (`trainer.resume_from_checkpoint`), and `hf_checkpoints/` for downstream evaluation, fine-tuning, or Hub upload (see [Using a Trained Model with Hugging Face](#using-a-trained-model-with-hugging-face)).

### Checkpoint saving, decoupling, and rotation

The two checkpoint types fire strictly on their own independent schedules and never trigger each other:

- **Resumable state** (`checkpoint_epoch_*_step_*`) saves every `trainer.save_steps` optimizer steps.
- **HF checkpoint** (`hf_checkpoints/checkpoint_<S>/`) saves every `trainer.hf_save_steps` optimizer steps.
- Neither is saved automatically at the end of every epoch. Instead, both are guaranteed to be saved exactly once at the very end of training (after the last epoch/`max_steps`), unless a step-based save already landed on that exact final step.
- Set `trainer.save_total_limit` / `trainer.hf_save_total_limit` (both `None`/unlimited by default) to cap how many checkpoints of each type are kept on disk; the oldest (by step) are deleted automatically as new ones are saved. This keeps storage bounded on long runs.

### Uploading a checkpoint to the HuggingFace Hub

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

## Using a Trained Model with HuggingFace

Every checkpoint in `hf_checkpoints/` is self-contained: `config.json` has an `auto_map`,
and the AMPLIFY source files are copied next to the weights. Any project with
`transformers` can load it through the Auto classes, from a local folder or a Hub
repo, without this codebase. Note that `trust_remote_code=True` is required because
AMPLIFY is not a built-in `transformers` architecture.

| Auto class | AMPLIFY class | Use |
|---|---|---|
| `AutoConfig` | `AMPLIFYConfig` | Model hyperparameters |
| `AutoTokenizer` | `ProteinTokenizer` | Amino-acid tokenizer (adds `<bos>`/`<eos>`) |
| `AutoModel` | `AMPLIFYModel` | Encoder only: per-residue embeddings |
| `AutoModelForMaskedLM` | `AMPLIFYForMaskedLM` | Pretraining head: masked-residue logits |
| `AutoModelForSequenceClassification` | `AMPLIFYForSequenceClassification` | Mean-pooled head for per-protein labels |
| `AutoModelForTokenClassification` | `AMPLIFYForTokenClassification` | Per-residue labels |

**Embeddings:**

```python
import torch
from transformers import AutoModel, AutoTokenizer

path = "runs/120M/hf_checkpoints/checkpoint_8000"  # or a Hub repo id
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModel.from_pretrained(path, trust_remote_code=True).eval()

batch = tokenizer(["MKTAYIAKQR", "MSV"], padding=True, return_tensors="pt")
with torch.no_grad():
    hidden = model(**batch).last_hidden_state                   # (B, S, hidden_size)
mask = batch["attention_mask"].unsqueeze(-1)
protein_emb = (hidden * mask).sum(1) / mask.sum(1)              # (B, hidden_size)
```

**Fine-tuning:** load a task head on the pretrained encoder:

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    path, trust_remote_code=True, num_labels=3
)
```

The new `classifier` weights start untrained, and the unused `lm_head` is dropped;
`transformers` lists both in its load report, which is expected. To share a
checkpoint, see [Uploading a checkpoint to the Hugging Face Hub](#uploading-a-checkpoint-to-the-hugging-face-hub).

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| NCCL / distributed timeout on the first multi-GPU run | Rank 0 is still building the cache. Build it first with `dataset.prepare_only=true`, or raise `trainer.dist_timeout_minutes` |
| `KeyError: Cluster column '...' not found` | The source lacks that column; check `cluster_column`/`split_column`/`score_column` against [Input Data](#input-data) |
| `KeyError: Field "..." does not exist in schema` after changing `cluster_column` | That column wasn't carried in the split cache; see [Reusing one cache across cluster thresholds](#reusing-one-cache-across-cluster-thresholds) |
| `RuntimeError: Fused optimizer requested ... CUDA is not available` | Running without a GPU; pretraining requires CUDA |
| Training much slower than expected with `torch.compile` | Shapes keep recompiling; keep `compile_dynamic: true` and check with `TORCH_LOGS=recompiles` |
| `dataloader/iter_step_pct` high (GPUs waiting for data) | Data on a network filesystem; set `dataset.local_stage_dir` (see [Staging data](#staging-data-to-node-local-disk-network-filesystems)) or raise `dataloader_num_workers` |
| CUDA out of memory | Lower `dataloader.max_tokens` and raise `gradient_accumulation_steps` to keep tokens per step (see [Batch size and steps](#batch-size-and-steps)) |

---

## Testing

```bash
uv run pytest tests/pretrain
```

Unit and integration tests for every component. GPU-only tests are skipped on CPU.
