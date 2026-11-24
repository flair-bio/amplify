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
Hub commit when downloading.

### Alternatively, load directly from the Hub

The same setting also accepts a Hub dataset ID:

```yaml
dataset:
  path: flair-bio/miae-l-embeddings
```

An existing local directory takes precedence; otherwise, a Hub ID downloads or
reuses a cached snapshot. `dataset.revision` is optional: omit it for the default
branch, or specify a branch, tag or commit hash. A full commit hash is recommended
for reproducibility. The resolved snapshot commit is recorded in `run.json` even
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
loss = mlm.loss_weight * MLM loss
     + alignment.loss_weight * latent alignment loss
```

The default MLM and alignment weights are `1.0` and `0.5`, respectively.
Sequence features are captured just before the encoder's final normalization.
The entire native sequence model, including its MLM decoder, is updated
alongside the new projections and temperature. Structure embeddings remain
fixed; **no teacher encoder is loaded**.

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
| `alignment.pool_proteins` | 256 | Proteins sharing contrastive negatives; one pool per update |
| `training.microbatch_size` | 32 | Proteins processed in each model forward |
| `training.anchor_chunk_size` | 1024 | Residue anchors processed per contrastive similarity block |
| `alignment.projection_dim` | 512 | Shared sequence/structure embedding dimension |
| `mlm.loss_weight` | 1.0 | MLM contribution to the total loss |
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
negative pool. Larger pools increase contrastive computation and memory use.

Other validated settings cover MLM corruption, temperature, AdamW betas/decay,
gradient clipping and the random seed. CUDA execution uses BF16 autocast with
FP32 parameters, gradients and contrastive softmax. Python, NumPy and PyTorch
random-number generators are seeded; cuDNN deterministic mode is enabled, but
PyTorch's global deterministic-algorithms mode is not forced. Exact repeatability
across runs, hardware or software versions is not guaranteed.
Override the top-level seed with `seed=123`; it controls initialization,
training randomness and diagnostic validation masking.

## Monitor training and validation

The terminal shows startup milestones, a progress bar with update count, elapsed
time, ETA, training losses and the latest validation MLM loss, and a protein
counter during validation. Final training and validation metrics are saved in
`run.json`; per-update curves are sent to W&B when enabled.

Validation runs **before training, every `training.validation_interval` updates,
and after the final update**. It evaluates every validation protein, with fixed
corruption masks and deterministic source/ID ordering.
This measures the **unweighted MLM loss** only, not latent alignment or
downstream performance.
Validation does not choose a checkpoint or determine when training ends.

### Enable Weights & Biases

W&B is disabled by default. After authenticating separately, append:

```bash
wandb.enabled=true \
wandb.entity=YOUR_USERNAME_OR_TEAM \
wandb.project=amplify \
wandb.run_name=amplify-latent-alignment
```

`wandb.entity` is a W&B username or team, not necessarily an organization slug.
When enabled, W&B uses its standard SDK settings.
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
and relative embedding path.

Only `parquet_per_protein_v1` is supported. Each file contains one row per
residue, row-identity columns and float32 embedding features.
`metadata.embedding_schema.feature_columns` defines their order and dimension.
Files are checked for schema, metadata, finite nonzero values and residue
identity when loaded. Input file checksums are not computed or verified.

Teacher details in `metadata.structural_teacher` are optional provenance, not
training settings. No teacher identity is required.
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
  run.json             # Resolved settings, provenance and final metrics
```

Pass **`<output_dir>/amplify`** to downstream evaluation as the model checkpoint.
It does not need the original student directory or structure embeddings. Load
checkpoint-owned code in a fresh process to avoid global model/kernel
registration collisions.

`run.json` is written before training with the resolved settings and provenance,
then updated with final metrics and a completed status after export. If
interrupted, retain the incomplete output for diagnosis and restart with a fresh
directory. It records input paths and metadata but does not contain file hashes.
