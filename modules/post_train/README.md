# Post-training with latent alignment

Teach AMPLIFY to capture structural information **from protein sequences alone**.
Training pairs sequences with precomputed per-residue structure embeddings,
updates the pretrained sequence model, and exports a reusable AMPLIFY checkpoint.
The exported model needs neither structures nor structure embeddings at inference.

The method is called **`latent_alignment`** throughout the configuration and
method-specific code. It aligns continuous embedding vectors, not protein
coordinates. Structural alignment describes the broader goal; `latent_alignment`
identifies the method implemented here.

This module handles post-training, not structure acquisition, embedding
extraction or benchmark-specific fine-tuning. Downstream evaluation remains in
[`modules/evaluate`](../evaluate).

## Quick start

Run from the repository root. You need a complete local native AMPLIFY
masked-language-model checkpoint, a prepared embedding dataset, and a
BF16-capable CUDA GPU.

```bash
uv run --extra post_train python -m modules.post_train.src.run_post_train \
  modules/post_train/configs/latent_alignment.yaml \
  student.checkpoint=/absolute/path/to/native-amplify \
  student.trust_remote_code=true \
  dataset.path=/absolute/path/to/miae-l-embeddings \
  output_dir=/absolute/path/to/new-run
```

Replace the paths before running. **Use a new output directory for each run.**
Checkpoint-owned Python is executed during loading; review it before enabling
`student.trust_remote_code`.

The runner uses the first visible CUDA GPU, respecting scheduler allocations and
`CUDA_VISIBLE_DEVICES`. It fails if no GPU is visible rather than silently using
CPU. Explicit overrides are available: `training.device=cuda:0`, or
`training.device=cpu` for small checks.

### Use a pre-downloaded Hugging Face dataset

Download the dataset before launching training, keeping `dataset.json` and its
relative embedding paths together. Point `dataset.path` at the downloaded
directory, not the manifest file:

```bash
dataset.path=/absolute/path/to/miae-l-embeddings
```

Training finds `dataset.json` inside that directory without contacting the Hub.
Use an absolute path, `./...` or `~/...` to make local intent explicit; a missing
local path fails rather than falling back to a Hub download. Leave
`dataset.revision` unset for local data. For reproducibility, pin and record the
Hub commit when downloading; training also records the manifest hash and verifies
embedding file checksums.

### Alternatively, load directly from the Hub

The same setting also accepts a Hub dataset ID:

```yaml
dataset:
  path: flair-bio/miae-l-embeddings
```

An existing local directory takes precedence; otherwise, a Hub ID downloads or
reuses a cached snapshot. `dataset.revision` is optional: omit it for the default
branch, or specify a branch, tag or commit hash. A full commit hash is recommended
for reproducibility. The resolved snapshot commit is recorded in provenance even
when the requested revision is a branch or omitted.

Cache placement uses standard Hugging Face settings such as `HF_HOME` and
`HF_HUB_CACHE`; there is no dataset-specific cache setting. Authenticate separately
for private datasets. Inputs are read-only. For either source, the manifest
defaults to `dataset.json`; override `dataset.manifest_file` for a nested manifest.

## What training does

Each optimizer update combines two objectives:

1. **Masked language modeling (MLM):** predict selected residues from corrupted
   sequences using the pretrained MLM decoder. By default, 15% of residues are
   selected; of those, 80% become MASK, 10% become a uniformly sampled vocabulary
   token, and 10% remain unchanged.
2. **Latent alignment:** match each clean-sequence residue representation to its
   corresponding frozen structure embedding. Separate trainable linear
   projections map both representations into a shared, L2-normalized space.
   Symmetric InfoNCE pulls matching pairs together and contrasts them against
   every other residue in the same protein pool, including same-protein residues.

```text
loss = MLM loss + alignment.loss_weight * latent alignment loss
```

The default alignment weight is `0.5`. Sequence features are captured just before
the encoder's final normalization. The entire native sequence model, including
its MLM decoder, is updated alongside the new projections and temperature.
Structure embeddings remain fixed; **no teacher encoder is loaded**.

Training makes **one pass** over the training split, then exports the final
checkpoint. The update count is `number_of_training_proteins / training.batch_size`.
There is no early stopping, checkpoint selection, learning-rate schedule or
optimizer resume.

## Configure a run

Edit [`configs/latent_alignment.yaml`](configs/latent_alignment.yaml), or append
`key=value` overrides to the command. YAML files merge left to right, followed by
terminal overrides. Relative paths resolve from your shell's working directory.

The supplied YAML uses:

| Setting | Value | Meaning |
| --- | --- | --- |
| `training.batch_size` | 256 | Proteins contributing to one optimizer update |
| `alignment.pool_proteins` | 64 | Proteins sharing contrastive negatives; four independent pools per update |
| `training.microbatch_size` | 32 | Proteins processed in each model forward |
| `training.anchor_chunk_size` | 1024 | Residue anchors processed per contrastive similarity block |
| `alignment.projection_dim` | 512 | Shared sequence/structure embedding dimension |
| `alignment.loss_weight` | 0.5 | Alignment contribution to the total loss |
| `training.encoder_lr` | `3e-6` | Learning rate for the pretrained encoder and MLM decoder |
| `training.head_lr` | `1e-3` | Learning rate for projections and temperature |
| `training.validation_interval` | 25 | Optimizer updates between diagnostic validations |

The training count must be divisible by the optimizer batch size. The batch size
must be a multiple of the pool size, and the microbatch size must divide the pool
size. Proteins are never silently dropped or repeated.

**Microbatch size and anchor chunk size control different memory costs.**
Microbatches limit model-forward memory; anchor chunks limit similarity-matrix
memory. Every anchor still compares against all candidate residues in its pool.
Gradient caching replays microbatches to accumulate full-pool gradients without
keeping all encoder graphs in memory; smaller microbatches do not shrink the
negative pool.

Other validated settings cover MLM corruption, temperature, AdamW betas/decay,
gradient clipping and the random seed. CUDA execution uses BF16 autocast with
FP32 parameters, gradients and contrastive softmax. Deterministic execution is
enabled, but bitwise equality across hardware or software versions is not promised.
Override the top-level seed with `seed=123`; it controls initialization,
training randomness and diagnostic validation sampling/masking.

## Monitor training and validation

The terminal shows a progress bar with update count, elapsed time, ETA, training
losses and the latest validation MLM loss. Set `training.progress_bar=false` to
disable it. Full numeric records are always saved to `metrics.jsonl`.

Validation runs **before training, every `training.validation_interval` updates,
and after the final update**. It uses fixed corruption masks on a deterministic
sample of up to 32 validation proteins per source.
This measures **MLM only**, not latent alignment or downstream performance.
Validation does not choose a checkpoint or determine when training ends.

### Enable Weights & Biases

W&B is disabled by default. After authenticating separately, append:

```bash
wandb.enabled=true \
wandb.mode=online \
wandb.entity=YOUR_USERNAME_OR_TEAM \
wandb.project=amplify \
wandb.run_name=amplify-latent-alignment
```

`wandb.entity` is a W&B username or team, not necessarily an organization slug.
Use `wandb.mode=offline` to log locally without syncing online.
Local W&B files default to `<output_dir>/wandb`.
Training metrics appear under `train/*` and diagnostic metrics under
`validation/*`. The logger records settings and metrics, not model weights,
source code, protein sequences or embedding targets.

## Required inputs

### Student checkpoint

Supply a complete native AMPLIFY checkpoint: weights, model configuration,
tokenizer assets and local Python referenced by `auto_map`. Both single-file
and split-file model layouts are supported and preserved on export.
The supported interface has `amplify.norm` and one pretrained `lm_head`;
the loader does not convert architectures or replace missing weights.

Vocabulary size and PAD/MASK/BOS/EOS IDs come from checkpoint assets.
Tokenization must preserve one token per residue plus BOS/EOS, agree with the
model, and support every dataset residue. Positional capacity must cover the
longest actual sequence plus those two tokens. Inputs are never cropped.

### Embedding dataset

```text
dataset.json
embeddings/
  protein-a.parquet
  protein-b.parquet
  ...
```

The manifest contains `metadata` and `records`. Records identify the protein,
chain, source, train/validation split, uppercase sequence, ordered residue IDs,
relative embedding path and file SHA256.

Only `parquet_per_protein_v1` is supported. Each file contains one row per
residue, row-identity columns and float32 embedding features.
`metadata.embedding_schema.feature_columns` defines their order and dimension.
Every file must agree with the manifest, have finite nonzero feature rows, and
pass checksum and residue-mapping checks when loaded.

Teacher details in `metadata.structural_teacher` are optional provenance, not
training settings. No teacher identity or teacher checkpoint hash is prescribed.
Supplied dimension declarations must agree with the feature columns; model and
revision declarations must agree when present in both manifest and file metadata.
Original dataset metadata is preserved in the run's provenance.

Training IDs and sequences must be unique, and train/validation splits must be
nonempty and sequence-disjoint. Input validation does not establish biological
independence: audit homology and downstream benchmark overlap separately.

## Outputs

```text
<output_dir>/
  amplify/             # Standalone native model, tokenizer and code
  alignment_heads.pt   # Training-only projections and temperature
  config.json          # Resolved settings and selected device
  provenance.json      # Dataset metadata and input hashes
  metrics.jsonl        # Training and validation records
  summary.json         # Successful completion and run summary
  wandb/               # Present when W&B logging is enabled
```

Pass **`<output_dir>/amplify`** to downstream evaluation as the model checkpoint.
It does not need the original student directory, structure embeddings or
alignment heads. Load checkpoint-owned code in a fresh process to avoid global
model/kernel registration collisions.

Only a completed run writes `summary.json`. If interrupted, retain the incomplete
output for diagnosis and restart with a fresh directory.
