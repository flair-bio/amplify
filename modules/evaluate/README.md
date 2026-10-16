# `modules/evaluate` — Evaluation Module

This module measures how well a pretrained protein language model performs on
downstream tasks, such as predicting where a protein localizes in the cell or
which of its residues touch in 3D. Alongside each score it can report bootstrap
confidence intervals and test the model against controls (random weights,
shuffled sequences or labels, simple baselines), which shows how much of the
score actually comes from pretraining.

It is the last module of this repo:

```
modules/data         →  modules/pretrain      →  modules/evaluate
clustered, scored       MLM training,            downstream tasks, controls,
protein datasets        HF checkpoints           confidence intervals
```

Every run goes through four steps:

```
Prepare            →  Tune               →  Predict            →  Score
(load model + data,   (probe or fine-tune;  (test split, plus     (metrics, CIs,
 build dataloaders)    skipped if as-is)     each control)         control tests)
```

A run is configured by stacking YAML files (a base config, then one file each for
the model, task, and dataset) plus optional `key=value` overrides; see
[Running and Configuring](#running-and-configuring).

Deeper guides on controls, datasets, dataset sourcing, and ProteinGym are in
[`docs/`](docs/README.md).

## Table of Contents

- [Quick Start](#quick-start)
  - [Sequence classification](#sequence-classification)
  - [Contact prediction](#contact-prediction)
- [Key Concepts](#key-concepts)
  - [Glossary](#glossary)
  - [Supported Tasks](#supported-tasks)
  - [Datasets](#datasets)
- [Running and Configuring](#running-and-configuring)
  - [Config files](#config-files)
  - [Running many combinations](#running-many-combinations)
  - [Common recipes](#common-recipes)
  - [New dataset, task, or model overlay](#new-dataset-task-or-model-overlay)
  - [Config schema](#config-schema)
- [Step 1 — Prepare](#step-1--prepare)
- [Step 2 — Tune](#step-2--tune)
- [Step 3 — Predict](#step-3--predict)
- [Step 4 — Score](#step-4--score)
- [Outputs](#outputs)
  - [Output disk layout](#output-disk-layout)
  - [Reading results](#reading-results)
- [Advanced Workflows](#advanced-workflows)
  - [Frozen-trunk probe overlays (`configs/tuning/`)](#frozen-trunk-probe-overlays-configstuning)
  - [Layerwise linear probes](#layerwise-linear-probes)
  - [Cross-validation](#cross-validation)
  - [Categorical-Jacobian contact prediction](#categorical-jacobian-contact-prediction)
- [Developer Guide](#developer-guide)
  - [Code Layout](#code-layout)
  - [Pipeline Architecture \& Distributed Execution](#pipeline-architecture--distributed-execution)
  - [Adding a Task Type](#adding-a-task-type)
- [Troubleshooting](#troubleshooting)
- [Testing](#testing)

---

## Quick Start

Install from the `flair-plm` root, and run all commands below from there:

```bash
uv sync --extra evaluate
```

The smoke tests add [`configs/smoke_test.yaml`](configs/smoke_test.yaml) to a
model, task, and dataset overlay. It shrinks batch size, sequence length, and
bootstrap resamples, and sets `steps.mode=evaluate_as_is`, so Tune is skipped
and only Prepare → Predict → Score run. A CUDA GPU is only needed for packed
batches (`steps.prepare.packed=true`, off by default).

### Sequence classification

```bash
uv run accelerate launch modules/evaluate/src/run_evaluate.py \
    modules/evaluate/configs/config.yaml \
    modules/evaluate/configs/models/amplify_120m.yaml \
    modules/evaluate/configs/tasks/sequence_classification.yaml \
    modules/evaluate/configs/datasets/localization_prediction.yaml \
    modules/evaluate/configs/smoke_test.yaml
```

If the run prints `=== Evaluation Pipeline Completed Successfully ===`, your
environment and the Prepare, Predict, and Score steps work. To exercise training
too, append a [probe overlay](#frozen-trunk-probe-overlays-configstuning) such as
`modules/evaluate/configs/tuning/torch_linear.yaml`.

### Contact prediction

The same smoke overlay works for contact prediction. `local.yaml` runs a single
local process, and `n_permutations=5` keeps the trained-vs-control statistics
cheap enough for a laptop:

```bash
uv run accelerate launch \
    --config_file modules/evaluate/configs/accelerate/local.yaml \
    modules/evaluate/src/run_evaluate.py \
    modules/evaluate/configs/config.yaml \
    modules/evaluate/configs/models/amplify_120m.yaml \
    modules/evaluate/configs/tasks/contact_prediction.yaml \
    modules/evaluate/configs/datasets/contact_prediction_binary.yaml \
    modules/evaluate/configs/smoke_test.yaml \
    steps.score.n_permutations=5
```

Smoke-test results are written under `./eval/smoke_test/`; see [Outputs](#outputs)
for the layout and how to read them.

---

## Key Concepts

### Glossary

| Term | Meaning |
|---|---|
| Trunk | The pretrained protein language model being evaluated |
| Head (probe) | A small task-specific model trained on the trunk's outputs (embeddings). Chosen by `steps.tune.head.head_type`: neural (`mlp`, `torch_linear`) or classical (`knn`, `sklearn_linear`, `random_forest`, `xgboost`) |
| Embedding cache | Trunk outputs computed once and reused, so heads on a frozen trunk train without rerunning the trunk. Classical heads require it |
| Mode | `steps.mode`: `tune_probe` trains a head on a frozen trunk, `finetune` trains head and trunk, `evaluate_as_is` trains nothing (zero-shot tasks) |
| Zero-shot | Scored straight from the pretrained model, with no task-specific training |
| Overlay | A YAML file merged over `configs/config.yaml`; when files set the same key, the later file wins |
| Workspace | Output directory for one (model, dataset) pair under `workspace.base_path` |
| Variant | One run configuration inside a workspace, e.g. `mlp-frozen_trunk` or `torch_linear-frozen_trunk-layer_6`; see [Output disk layout](#output-disk-layout) |
| Control | A baseline the trained model is compared against (random weights, shuffled labels, class prior, ...). Beating it rules out that shortcut as the source of the score; see [Controls](docs/controls.md) |
| Separation | Contact prediction: how far apart two residues are in the sequence. Bins: short `[6, 12)`, medium `[12, 24)`, long `≥ 24` |
| P@L | Contact prediction: precision over the top-L scored pairs, L = sequence length; P@L/5 uses the top L/5 |

### Supported Tasks

| Task type | Predicts |
|---|---|
| `sequence_classification` | One class per protein |
| `sequence_regression` | One number per protein |
| `sequence_multilabel_classification` | A set of labels per protein |
| `token_classification` | One class per residue |
| `contact_prediction` | Which residue pairs touch |
| `categorical_jacobian` | Contacts, zero-shot |
| `pseudo_perplexity` | Nothing; scores masked residues, zero-shot |
| `calibration` | Homolog amino-acid profiles at conserved positions, zero-shot |

`categorical_jacobian`, `pseudo_perplexity` and `calibration` run only with
`steps.mode=evaluate_as_is`. Target formats, why each task matters, and default
metrics per dataset are in the [dataset catalog](docs/datasets.md#evaluation-tasks).

### Datasets

The [dataset catalog](docs/datasets.md) lists each shipped overlay, what its
labels represent, and which task uses it; [Dataset Sourcing](docs/dataset-sourcing.md)
covers building and publishing them. ProteinGym uses per-assay variant tables and
its own runners (`run_proteingym_zero_shot.py`, `run_proteingym_supervised.py`)
instead of `run_evaluate.py`; see the [ProteinGym guide](docs/proteingym.md).

---

## Running and Configuring

### Config files

| File | Use |
|---|---|
| `configs/config.yaml` | Base config: every section, with every field documented. Runnable baseline |
| `configs/smoke_test.yaml` | Overlay for a quick, bounded smoke test |
| `configs/models/*.yaml` | One overlay per evaluated model (name, repo) |
| `configs/tasks/*.yaml` | Task type, behavior, and default controls |
| `configs/datasets/*.yaml` | One overlay per dataset; see the [catalog](docs/datasets.md) |
| `configs/tuning/*.yaml` | One probe/HPO overlay per `steps.tune.head.head_type` |
| `configs/accelerate/*.yaml` | `accelerate launch` configs (e.g. single-process local runs) |

The entry point is `modules/evaluate/src/run_evaluate.py`. It takes one or more YAML files, merged left-to-right (later files win), followed by optional `key=value` overrides:

```bash
uv run accelerate launch modules/evaluate/src/run_evaluate.py \
    modules/evaluate/configs/config.yaml \
    modules/evaluate/configs/models/amplify_120m.yaml \
    modules/evaluate/configs/tasks/token_classification.yaml \
    modules/evaluate/configs/datasets/ssp_q3.yaml \
    steps.mode=finetune \
    steps.prepare.embedding_cache.enabled=false \
    steps.score.n_samples=1000
```

### Running many combinations

There is no sweep driver: each invocation handles one (model, dataset, head) combination.
Run combinations against one `workspace.base_path` so every result lands in a
`scores_long.parquet` that [Reading results](#reading-results) can combine. Locally,
loop over overlays:

```bash
for model in amplify_120m amplify_350m esm2_35m; do
  uv run accelerate launch modules/evaluate/src/run_evaluate.py \
      modules/evaluate/configs/config.yaml \
      modules/evaluate/configs/models/${model}.yaml \
      modules/evaluate/configs/tasks/sequence_classification.yaml \
      modules/evaluate/configs/datasets/localization_prediction.yaml \
      steps.score.bootstrap_enabled=true
done
```

On a cluster, a Slurm array job runs the combinations in parallel, one per task:

```bash
#!/bin/bash
#SBATCH --job-name=evaluate
#SBATCH --account=def-yourgroup
#SBATCH --time=0-06:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-8          # len(MODELS) * len(DATASETS) - 1

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODELS=(amplify_120m amplify_350m esm2_35m)
DATASETS=(localization_prediction metal_ion_binding solubility_prediction)
model=${MODELS[$((SLURM_ARRAY_TASK_ID / ${#DATASETS[@]}))]}
dataset=${DATASETS[$((SLURM_ARRAY_TASK_ID % ${#DATASETS[@]}))]}

uv run accelerate launch modules/evaluate/src/run_evaluate.py \
    modules/evaluate/configs/config.yaml \
    modules/evaluate/configs/models/${model}.yaml \
    modules/evaluate/configs/tasks/sequence_classification.yaml \
    modules/evaluate/configs/datasets/${dataset}.yaml \
    workspace.base_path=/path/to/eval
```

Give each array task its own (model, dataset) pair. Per-variant outputs under
`runs/<variant>/` never clash, but two files are deliberately shared by every
variant of a pair: `scores/scores_long.parquet`, which Score updates by
read-merge-write without a lock, and the `tmp/embeddings/` cache. Concurrent
tasks on the same pair (e.g. different heads) can therefore drop each other's
`scores_long` rows; run those sequentially within one task.

### Common recipes

Append these overrides to any launch command, or put them in an overlay YAML.

| Goal | Overrides |
|---|---|
| Skip tuning and score the checkpoint as-is | `steps.mode=evaluate_as_is steps.prepare.embedding_cache.enabled=false`. For supervised tasks the head comes from the checkpoint; a pretrained-only checkpoint gets a randomly initialized head, so use this for fine-tuned checkpoints or zero-shot tasks |
| Fine-tune the trunk and native MLP head | `steps.mode=finetune steps.prepare.embedding_cache.enabled=false`, or append `configs/tuning/mlp_finetuning.yaml` |
| Bootstrap confidence intervals and control tests | `steps.score.bootstrap_enabled=true` |
| Use a generated split instead of the default | `steps.prepare.split.name=stratified` (see [Named benchmark splits](#named-benchmark-splits)) |
| K-fold cross-validated probe scores | `cross_validation.enabled=true` (see [Cross-validation](#cross-validation)) |
| Classical probe on cached embeddings | Add `modules/evaluate/configs/tuning/random_forest.yaml` after the dataset overlay |
| Disable control conditions | `steps.predict.controls=[]` |
| Write results elsewhere | `workspace.base_path=/path/to/eval` |

### New dataset, task, or model overlay

Copy an existing file under `configs/datasets/`, `configs/tasks/`, or
`configs/models/`. Dataset overlays set the source, target type, label format,
and class count. Task overlays set `task_type` and task behavior. Task-specific
builder parameters belong in `workspace.model_kwargs`.

### Config schema

All config classes are Pydantic models composed into a single `EvaluationConfig` (`src/config.py`):

```
EvaluationConfig
├── workspace: EvaluationWorkspaceConfig
├── wandb:     WandbConfig
├── cross_validation: CrossValidationConfig
└── steps:     StepsConfig
    ├── layerwise: LayerwiseConfig
    ├── prepare:   PrepareConfig
    ├── tune:      TuneConfig
    │   └── hyperparameter_search: HyperparameterSearchConfig
    ├── predict:   PredictConfig
    └── score:     ScoreConfig
```

`EvaluationConfig` and `StepsConfig` set `extra="forbid"`, so a misspelled
top-level or `steps.*` key fails at load time rather than being silently ignored.

The **Default** columns in the step sections below are the Pydantic schema
defaults. The checked-in [`configs/config.yaml`](configs/config.yaml) is a
runnable project baseline and intentionally overrides some of them, including
sampling, tuning epochs and batch size, HPO trial count, metric aggregation,
and whether bootstrap scoring is enabled.

---

## Step 1 — Prepare

**Purpose:** Load the tokenizer, model, and dataset; validate task labels; and
build the train, validation, and test dataloaders. The default dataset columns
are `sequence`, `targets`, and `id`; if the configured ID column is absent,
Prepare uses positional identifiers.
For positional labels, tokenization preserves ambiguous residues by disabling
`remove_ambiguous`; this setting is saved with the tokenizer.

**Key config fields (`steps.prepare`):**

| Field | Schema default | Description |
|---|---|---|
| `batch_size` | `16` | Examples per dataloader batch, unless a packed token budget is set |
| `dataloader_num_workers` | `8` | Workers per rank and loader; `0` loads batches in the main process |
| `tokenize_num_proc` | `8` | Positive process count for up-front tokenization, independent of dataloader workers |
| `max_length` | `512` | Maximum tokenized sequence length |
| `split.name` | `default` | Select `train_<name>`, `validation_<name>`, and `test_<name>` for a named split method |
| `validation_fraction` | `0.1` | Share of train held out when validation is missing or explicitly unset |
| `length_grouped_sampling` | `false` | Group similarly sized sequences to reduce padding |
| `embedding_cache.enabled` | `false` | Cache frozen-trunk embeddings for supported probe modes |
| `packed` | `false` | Pack multiple sequences into one forward pass; requires CUDA and a compatible AMPLIFY checkpoint |
| `max_tokens_per_batch` | `null` | Optional packed token budget; single-process only and supersedes `batch_size` |

Sequence and token probe task YAMLs enable embedding caching. For `finetune`
or `evaluate_as_is`, disable it or load the finetuning/smoke overlay **after**
the task YAML. Sequence features support disk reuse; token features stay in
RAM with no cap (`max_cache_gb` limits disk writes only). Disable token caching
if host memory is insufficient.

Frozen-trunk extraction uses deterministic length order when grouping is enabled,
or dataset order otherwise, keeping trained and control predictions aligned
across extraction batch sizes. Grouping is part of the cache key; cached feature
training still shuffles batches.

Match worker counts to the job's CPU allocation. Persistent dataloader pools
can coexist across loaders; tokenization runs first on the main rank to populate
the shared cache.

Pseudo-perplexity prepares `test` under all three roles without random cropping,
but scores only the test loader and performs no fitting. Preparation is still
repeated per role; to use another split, change all three source mappings.

### Named benchmark splits

The default split uses the dataset's `train`, `validation`, and `test` names.
Set `steps.prepare.split.name=stratified` (or another generated method) to load
the corresponding suffixed split names. For benchmarks with unrelated split
names, set `steps.prepare.split.train`, `.validation`, and `.test` explicitly.
Setting validation to `null` holds out `validation_fraction` of the training
split.

Packed mode requires a CUDA device and a compatible AMPLIFY checkpoint whose
forward signature accepts `cu_seqlens` and `max_seqlen`; sequence-classification
heads must also accept `num_sequences`. Pairwise tasks such as
`contact_prediction` are not supported. The collator emits flattened token
inputs with sequence-boundary metadata, and prediction/embedding utilities
unpack outputs to the usual per-example rows.

By default, packed `batch_size` still counts examples. Set
`steps.prepare.max_tokens_per_batch` to batch by tokens instead; this requires
`steps.prepare.packed=true`, supports one process only, and includes padding to
`pad_to_multiple_of` in the budget. For example:

```bash
uv run accelerate launch --num_processes 1 \
    modules/evaluate/src/run_evaluate.py \
    modules/evaluate/configs/config.yaml \
    modules/evaluate/configs/models/amplify_120m.yaml \
    modules/evaluate/configs/tasks/sequence_classification.yaml \
    modules/evaluate/configs/datasets/localization_prediction.yaml \
    steps.prepare.packed=true \
    steps.prepare.max_tokens_per_batch=4096
```

Packed mode supports frozen-trunk embedding caching. Sequence-level tasks keep
one pooled embedding per source example; token tasks keep one row per valid
token. Packed and padded embeddings use distinct cache keys. ProteinGym uses
separate runners and is unaffected.

**Input:** `workspace.dataset_repo_id` (or `dataset_name`) and
`workspace.model_repo_id` (or `model_name`).
**Output:** in-memory `PreparedArtifacts` containing the tokenizer, model, and
dataloaders.

---

## Step 2 — Tune

**Purpose:** Train on the `train` split according to `steps.mode` (see [Glossary](#glossary)); skipped for `evaluate_as_is`. Probe tuning can use cached pooled embeddings and hyperparameter search over `hyperparameter_search.search_space`.

**Key config fields (`steps.tune`):**

| Field | Default | Description |
|---|---|---|
| `head.head_type` | `mlp` | `mlp` (existing native model-head path), `torch_linear` (one `torch.nn.Linear` layer), `sklearn_linear` (sklearn logistic/ridge), `knn`, `random_forest`, or `xgboost` |
| `head.search_space` | `{}` | Classical-estimator dimensions; candidates are selected on the validation split, and the winning estimator is used for test prediction after fitting on `train` only |
| `metric_aggregation` | `weighted` | Averaging used for validation metric selection; automatically follows `steps.score.metric_aggregation` unless set explicitly; supports `micro`, `macro`, `weighted`, or `samples` |
| `learning_rate` | `1e-4` | Used when `hyperparameter_search.enabled=false` |
| `weight_decay` | `1e-4` | Used when `hyperparameter_search.enabled=false` |
| `max_epochs` | `10` | Maximum training epochs; used when `hyperparameter_search.enabled=false` |
| `warmup_ratio` | `0.06` | Fraction of total training steps used for LR warmup |
| `max_grad_norm` | `1.0` | Gradient clipping norm |
| `accumulation_steps` | `1` | Batches accumulated per optimizer step |
| `batch_size` | `128` | Cached-embedding probe batch size; set to `null` to use the PrepareStep loader size |
| `early_stopping_patience` | `5` | Stop early after this many epochs with no validation improvement |
| `optimizer` | `adamw` | One of `adamw`, `adam`, `sgd`, `rmsprop`, `adagrad`, `adafactor` |
| `optimizer_kwargs` | `{}` | Extra optimizer constructor kwargs, e.g. `{momentum: 0.9}` for `sgd` |
| `lr_scheduler_type` | `linear` | Any name accepted by `transformers.get_scheduler` |
| `scheduler_kwargs` | `{}` | Extra scheduler-specific kwargs, e.g. `{num_cycles: 3}` |
| `save_best_model` | `true` | Persist the best model + tokenizer to `workspace.model_path` after tuning |
| `variance_seeds` | `[]` | Optional extra seeds for logging validation-metric mean/std; diagnostic only and supported for cached-embedding probes |
| `hyperparameter_search.enabled` | `false` | Enable hyperparameter search instead of a single fixed-config run |
| `hyperparameter_search.n_trials` | `5` | Number of trials |
| `hyperparameter_search.direction` | `null` | `maximize` or `minimize` the target metric; inferred from the metric when omitted |
| `hyperparameter_search.metric` | `null` | Task-specific default metric unless overridden |
| `hyperparameter_search.search_pattern` | `bohb` | `bohb` (Optuna TPE sampler with median pruning), `random`, or `grid` |
| `hyperparameter_search.search_space.*` | see [`configs/config.yaml`](configs/config.yaml) | Candidate values for `learning_rate`, `max_epochs`, and `weight_decay`; probe batch size is configured separately with `steps.tune.batch_size` |

For classical `head_type` values, put fixed estimator settings in
`head.hyperparameters` and swept dimensions in `head.search_space`. Every
candidate trains only on `train` and is compared on `validation`; after
selection, the winning estimator remains fit on `train` for the held-out
`test` prediction, matching the torch probe paths. A validation split is
therefore required when a classical search space is configured. With
`hyperparameter_search.enabled=false`, every combination is evaluated as a
grid. With it enabled, `n_trials` and `search_pattern` select grid, random, or
Optuna BOHB-style trials over those estimator parameters. Classical estimators
fit atomically, so BOHB cannot prune within a fit.

`torch_linear` is trained through the PyTorch optimization loop and has no
hidden projection: its input is precisely the selected cached embedding
vector. For sequence-level classification/regression it requires
`steps.prepare.embedding_cache.enabled=true`; for token classification it can
also use a compatible native `nn.Linear` model head without the cache. Classical
probes always require `steps.mode=tune_probe` and
`steps.prepare.embedding_cache.enabled=true`. Set `embedding_cache.layers: [-1]`
for the final layer, or list any valid model hidden-state indices to concatenate
pooled representations from several layers.

Cached validation features stay on the device across epochs, and validation
inference skips loss computation.

**Input:** `PreparedArtifacts` from the Prepare step (or a resumed checkpoint)
**Output:** in-memory updated `PreparedArtifacts.model`; `<workspace>/runs/<variant>/model/` on disk when `save_best_model=true`

Contact checkpoints save the trunk and tokenizer alongside `contact_head.pt`
and `contact_config.json`. Reload them with
`get_task("contact_prediction").load_finetuned_model(path)`.

---

## Step 3 — Predict

**Purpose:** Run inference over the `test` split for the trained model and for each configured control condition. Control predictions are rebuilt every run from the current head; only frozen-trunk embeddings are reused. No metrics are computed here — that's [Score](#step-4--score)'s job.

**Control conditions (`controls`):** each removes one possible source of the trained model's performance, so [Score](#step-4--score) can test whether the trained model still beats it.

| Control | What changes |
|---|---|
| `untrained` | Pretrained checkpoint with a fresh head |
| `random_head` | Trained trunk, re-initialized head |
| `random_trunk` | Re-initialized trunk, trained head (probes retrained in layerwise mode) |
| `random_both` | Trunk and head re-initialized |
| `scrambled_sequences` | Residues shuffled within each sequence |
| `scrambled_labels` | Trained predictions scored against permuted labels |
| `majority_class` / `train_mean` | Train class prior / train label mean |
| `separation_prior` | Train contact rate per residue separation |

Per-task presets, invalid combinations, and how to read the results are in
[Controls](docs/controls.md).

**Key config fields (`steps.predict`):**

| Field | Default | Description |
|---|---|---|
| `collect_probabilities` | `false` | Collect task-appropriate per-class/per-label probabilities (softmax for single-label, sigmoid for multilabel); auto-enabled when an effective metric requires probabilities, such as `roc_auc` or `average_precision` |
| `predictions_filename` | `predictions.parquet` | Output filename under `<workspace>/runs/<variant>/preds/` |
| `persist_to_disk` | `true` | Write predictions and the prediction manifest; disable for fast in-process evaluation |
| `store_sequences` | `false` | Decode and include source sequences in the main predictions table |
| `controls` | `[untrained]` | Any of the controls above; task configs set per-task presets |

**Input:** `PreparedArtifacts` (trained model + test dataloader) from Prepare/Tune
**Output:** `<workspace>/runs/<variant>/preds/predictions.parquet`; settings are also recorded in `<workspace>/runs/<variant>/run_manifest.json`'s "predict" section

---

## Step 4 — Score

**Purpose:** Compute task-specific metrics from the Predict step's output and, optionally, bootstrap confidence intervals for those metrics plus a statistical comparison between the trained model and each configured control condition, to quantify how much of its performance is attributable to learning from the data rather than model/data artifacts.

Bootstrap and permutation tests reuse predictions, not model inference.
Multilabel AP/MCC use optimized array calculations, multiclass ROC-AUC retains
one-vs-one semantics without explicit class-pair loops, and contact resampling
reuses per-protein scores.

**Key config fields (`steps.score`):**

| Field | Default | Description |
|---|---|---|
| `metrics` | `null` | List of metrics to compute (task-specific defaults apply when unset), e.g. `accuracy`, `f1`, `precision`, `recall`, `average_precision`, `roc_auc`, `mcc`, `mse`, `r2`, `pearsonr`, `spearmanr`, `contact_precision_at_l_long`, `contact_auc_long`, `pseudo_perplexity`, `mean_log_prob`, `kl`, `kl_t`, `kl_nt`, `calibration`, `calibration_t`, `calibration_nt`, `accuracy_t`, `accuracy_nt`, `memorization_nt` |
| `metric_aggregation` | `weighted` | scikit-learn `average` for classification metrics: `micro` pools all classes/labels, `macro` weights each class equally (exposes rare-class failures), `weighted` weights classes by support, `samples` averages per example (multilabel only). Ignored by regression metrics and single-label MCC |
| `bootstrap_enabled` | `false` | Enable bootstrap confidence intervals + control comparisons |
| `n_samples` | `1000` | Bootstrap resamples; the code warns when below 1000 because confidence-interval tails are poorly estimated |
| `n_permutations` | `1000` | Trained-vs-control paired permutations; the smallest p-value is `1 / (n_permutations + 1)` before correction |
| `output_filename` | `bootstrap_results.parquet` | Per-resample bootstrap metric values (bootstrap mode only) |
| `confidence_interval` | `0.95` | Confidence level for bootstrap intervals |
| `long_filename` | `scores_long.parquet` | Tidy/long-format sidecar written on every run, with or without bootstrap — concatenable across (model, dataset) runs for plotting |

**Input:** `<workspace>/runs/<variant>/preds/predictions.parquet`
**Output:** `<workspace>/scores/scores_long.parquet` (always, shared across run variants), plus `<workspace>/runs/<variant>/scores/bootstrap_results.parquet` when `bootstrap_enabled=true`; scores/confidence intervals are also recorded in `<workspace>/runs/<variant>/run_manifest.json` under the `score`/`bootstrap` sections

---

## Outputs

### Output disk layout

`get_evaluation_workspace()` (in [`src/dataset/workspace.py`](src/dataset/workspace.py)) resolves a workspace directory keyed by model + dataset, so results for different (model, dataset) pairs never collide:

```
<base_path>/
└── <model_repo_or_name>/
    └── <dataset_repo_or_name>/
        ├── tmp/
        │   └── embeddings/            ← reusable frozen-trunk embedding cache
        ├── scores/
        │   ├── scores_long.parquet    ← shared comparison table across variants
        │   └── layerwise_comparison.parquet ← layerwise runs only
        └── runs/
            └── <variant>/
                ├── stats/              ← per-variant statistics, when produced
                ├── model/              ← saved model/head artifacts when enabled
                ├── run_manifest.json   ← Predict/Score settings and results
                ├── preds/
                │   ├── predictions.parquet
                │   └── control_*_predictions.parquet
                └── scores/
                    └── bootstrap_results.parquet ← written when bootstrap is enabled
```

`model_repo_id`/`dataset_repo_id` are preferred over `model_name`/`dataset_name` for the on-disk folder name when set, so results stay unambiguous across Hub repos that share a short mnemonic.

For training runs, `<variant>` is `<head_type>-frozen_trunk` for `tune_probe`
or `<head_type>-tuned_trunk` for `finetune`; `evaluate_as_is` uses `pretrained`.
A non-default `steps.prepare.split.name` is appended to the variant. Seeds and
other hyperparameters are not part of this path, so runs with the same model,
dataset, mode, head, and split can overwrite one another; use a separate
`workspace.base_path` when those settings need separate artifacts. The shared
`scores_long.parquet` records the resulting model, dataset, head, and split.

### Reading results

`<workspace>/scores/scores_long.parquet` is the table for comparing runs.
Every row carries the run identity (`model_name`, `dataset_name`, `task_type`,
`split`, `model_repo_id`, `dataset_repo_id`, `head_type`, `layer_index`,
`variant`, `seed`) plus a `metric`. Cross-validation fold (`-fold_<k>`) and
pooled (`-oof`) runs also set `cv_id`, `fold_id`, and `split_seed` (see
[Cross-validation](#cross-validation)):

| Row | `control` | Filled columns |
|---|---|---|
| Trained model | null | `value` (full test set); `ci_mean`/`ci_lower`/`ci_upper` when bootstrap is enabled |
| Trained vs. control (bootstrap only) | control name | `value` (trained bootstrap mean), `control_mean`, `diff_mean`/`diff_lower`/`diff_upper`, `p_value`, `p_value_adjusted` |

- `diff_*` is always trained − control from paired bootstrap resamples, so for
  lower-is-better metrics (`mse`, `mae`, `pseudo_perplexity`, `kl*`) a negative diff
  favors the trained model.
- `p_value` is a one-sided paired permutation test of "trained beats control";
  `p_value_adjusted` applies Benjamini–Hochberg across controls within each
  metric, not across metrics.
- The trained model beats a control when the diff interval excludes 0 in the
  favorable direction and `p_value_adjusted` is small. Which controls matter
  for which claim: [Interpreting results](docs/controls.md#interpreting-results).

Combine every run under one `workspace.base_path` (repo IDs have `/` replaced
by `_`, so workspaces are always two levels deep):

```python
from pathlib import Path
import polars as pl

scores = pl.concat(
    [pl.read_parquet(p) for p in Path("eval").glob("*/*/scores/scores_long.parquet")],
    how="diagonal_relaxed",
)
scores.filter(pl.col("control").is_null()).pivot(
    on="metric", index=["model_name", "dataset_name", "variant", "split"], values="value"
)
```

`layerwise_comparison.parquet` compares each probed layer with the deepest one
(`diff_*` = layer − reference), using two-sided permutation p-values adjusted
across layers per metric.

---

## Advanced Workflows

These build on the steps above; each is a few extra overlays or overrides on a
normal run.

### Frozen-trunk probe overlays (`configs/tuning/`)

Each probe overlay (`knn.yaml`, `sklearn_linear.yaml`, `random_forest.yaml`,
`xgboost.yaml`, `mlp.yaml`, `torch_linear.yaml`) is self-contained: it sets
`steps.mode=tune_probe` and the embedding cache itself, so it is just one more
file after the dataset overlay:

```bash
uv run accelerate launch modules/evaluate/src/run_evaluate.py \
    modules/evaluate/configs/smoke_test.yaml \
    modules/evaluate/configs/models/amplify_120m.yaml \
    modules/evaluate/configs/tasks/sequence_classification.yaml \
    modules/evaluate/configs/datasets/metal_ion_binding.yaml \
    modules/evaluate/configs/tuning/random_forest.yaml
```

The exception is `mlp_finetuning.yaml`, which trains the trunk too and so does
not use the embedding cache.

### Layerwise linear probes

Answers "which layer carries the signal?" by training an independent linear
probe on every requested layer in one run, through the same pipeline (controls
and bootstrap included). The embedding cache is recommended for sequence tasks:
each layer is extracted once per split and reused across epochs and runs.

```bash
uv run accelerate launch modules/evaluate/src/run_evaluate.py \
    modules/evaluate/configs/config.yaml \
    modules/evaluate/configs/models/amplify_120m.yaml \
    modules/evaluate/configs/tasks/sequence_classification.yaml \
    modules/evaluate/configs/datasets/localization_prediction.yaml \
    steps.layerwise.enabled=true \
    steps.layerwise.layers=all \
    steps.tune.head.head_type=torch_linear \
    steps.prepare.embedding_cache.enabled=true
```

**Layers.** `steps.layerwise.layers` is `all` or a list such as `"[0,6,-1]"`.

- `0` is the token-embedding output; mean-pooled, it is an amino-acid
  composition baseline, so `scrambled_sequences` matches it by construction.
- `-1` is the final block's output after the trunk's final norm.
- Intermediate indices are the un-normalized residual stream.

**Features.** Defaults keep depths comparable:

- Sequence tasks L2-normalize each residue, then mean-pool over residues
  (boundary and padding tokens excluded, unknown residues kept). Set
  `steps.layerwise.normalize_before_pooling=false` to keep raw magnitudes, or
  `steps.layerwise.pooling=cls` when the tokenizer has a CLS/BOS token.
- Token tasks use unpooled, label-aligned residues without L2 normalization,
  so norm differences between `-1` and the residual stream survive
  standardization.
- Features are standardized with train-split statistics
  (`steps.layerwise.standardize_features=false` disables this); regression
  targets are standardized too.
- Extraction runs in float32; reduced precision is opt-in.

**Training.** All heads share `steps.tune` optimizer settings and
`learning_rate`; each layer keeps its own best validation epoch, saved in its
`probe_head.pt`. If every layer picks the last epoch, raise
`steps.tune.max_epochs`. HPO and `variance_seeds` are not supported.

**Memory.** With the cache, sequence embeddings go to disk when eligible and
token embeddings stay in memory (the size estimate is logged before
extraction); `embedding_cache.dtype` and `cache_policy` control precision and
reuse. Without the cache, the trunk runs once per batch per epoch for all heads,
so memory scales with batch size, sequence length, and depth, not dataset size.

**Support.** Sequence classification, multilabel classification, scalar
regression, and token classification, with padded or compatible AMPLIFY packed
batches (`packed=true` needs CUDA; token-budget batches are single-process).

**Controls and outputs.**

- `untrained` and `random_head` are identical here and share one run.
  `random_trunk` retrains probes on a random trunk, and `random_both` applies
  untrained heads to it; see [Controls](docs/controls.md). Controls that change
  the trunk or input always run on tokens, even with the cache.
- Each layer is its own `<variant>-layer_<index>` run variant.
- With bootstrap enabled, `layerwise_comparison.parquet` compares every layer
  with the deepest one (see [Reading results](#reading-results)). Residues of
  one protein are resampled together, and its cost grows with the number of
  layers.

### Cross-validation

Set `cross_validation.enabled=true` to wrap Prepare → Tune → Predict →
Score in an outer K-fold loop (`src/runners/cross_validation.py`). Folds are
planned on raw data (`src/dataset/folds.py`) from `pool_roles`, which defaults
to `[train, validation]`; the official test split is never pooled. Both source
splits must exist. Set `cross_validation.pool_roles=[train]` to use training
data only (including datasets without a source validation split). Each fold
trains a fresh probe on its
train role, selects on an inner validation holdout (one fold of an inner
`round(1 / inner_validation_fraction)`-fold split with the same strategy, so
groups and strata are respected), and predicts its held-out outer fold.

```bash
uv run python -m modules.evaluate.src.run_evaluate \
    modules/evaluate/configs/config.yaml <model/task/dataset overlays> \
    cross_validation.enabled=true \
    cross_validation.strategy=stratified_group_kfold \
    cross_validation.group_column=sequence_cluster_id
```

CV settings live at the top level because they orchestrate the entire pipeline,
rather than configuring an individual step. Replace the former
`steps.cross_validation.*` overrides with `cross_validation.*`.

**Restrictions:** `steps.mode=tune_probe`, scalar `sequence_classification` /
`sequence_regression` tasks, layerwise disabled, `bootstrap_enabled=false`,
`persist_to_disk=true`, no `variance_seeds`, and a single process. The pool
must have finite scalar labels, non-null groups (group strategies), no
duplicate sequences across groups, and enough examples per class for
stratification.

Fold membership depends only on the data, split config, and `split_seed`
(not the training `seed`), so heads and models are compared on identical folds.
The assignments' fingerprint gives the `cv_id`.

**Outputs.** Each fold plan has a `cv_id`; each full configuration, including
the training seed and hyperparameters, also gets a distinct `config_hash` in
its output path. This lets runs share fold assignments without overwriting
each other's results:

| Path under `<workspace>/runs/` | Contents |
| --- | --- |
| `<variant>-cv_<cv_id>-run_<config_hash>/cv_manifest.json` | Status, config, per-fold counts and scores, fold summary (mean/std), OOF and control OOF scores |
| `<variant>-cv_<cv_id>-run_<config_hash>/assignments.parquet` | One row per (pool row, fold) with `fold_id` and `role` |
| `<variant>-cv_<cv_id>-run_<config_hash>/scores/scores_long.parquet` | Long-format fold + OOF rows with `cv_id`, `fold_id` (null for OOF), `split_seed` |
| `<variant>-cv_<cv_id>-run_<config_hash>-fold_<k>/` | Ordinary run for fold `k` |
| `<variant>-cv_<cv_id>-run_<config_hash>-oof/` | Pooled out-of-fold predictions and scores |

A completed experiment is not overwritten. Rerunning an interrupted one
reuses its finished folds when the fold plan and full config are unchanged
(`fingerprint`/`config_hash` in the manifest). Different configs get separate
paths, while configs with the same fold assignments retain the same `cv_id`.
Older output directories without the config hash are left untouched and are
not automatically resumed.

**Interpretation:** the fold std is descriptive spread, not a confidence
interval (folds share training data). Pooled OOF scores generally differ from
the mean of fold scores, especially for rank or threshold metrics.

### Categorical-Jacobian contact prediction

A zero-shot contact predictor: nothing is trained. The idea is that if mutating
residue *i* changes what the model predicts at residue *j*, the two are likely
in contact. Concretely, for each residue the model scores all 20 canonical
substitutions and records the change in its 20 canonical output logits at every
position, relative to the wild type. The `(L, 20, L, 20)` Jacobian is centered
on all four axes, symmetrized, Frobenius-reduced to `(L, L)`, and Average
Product Corrected (diagonal excluded). Scores are rankings, not calibrated
probabilities.

This bounded example keeps sequence length and batch small for a local check:

```bash
uv run accelerate launch modules/evaluate/src/run_evaluate.py \
    modules/evaluate/configs/config.yaml \
    modules/evaluate/configs/models/amplify_120m.yaml \
    modules/evaluate/configs/tasks/categorical_jacobian.yaml \
    modules/evaluate/configs/datasets/contact_prediction_binary.yaml \
    steps.prepare.max_length=32 \
    steps.prepare.batch_size=64 \
    steps.prepare.dataloader_num_workers=0 \
    steps.prepare.dataloader_persistent_workers=false \
    steps.prepare.dataloader_pin_memory=false \
    steps.prepare.tokenize_num_proc=1 \
    steps.prepare.length_grouped_sampling=false \
    steps.prepare.pad_to_multiple_of=null \
    steps.predict.controls=[] \
    steps.score.bootstrap_enabled=false \
    workspace.base_path=./eval/smoke_test
```

For a full run, drop the bounded overrides:

```bash
uv run accelerate launch modules/evaluate/src/run_evaluate.py \
    modules/evaluate/configs/config.yaml \
    modules/evaluate/configs/models/amplify_120m.yaml \
    modules/evaluate/configs/tasks/categorical_jacobian.yaml \
    modules/evaluate/configs/datasets/contact_prediction_binary.yaml
```

Sequences are processed one at a time, with `20 * L` mutant inputs batched by
`workspace.model_kwargs.mutant_batch_size`. The Jacobian is held in float32,
falls back to float16, and the run fails before allocation if it still exceeds
`workspace.model_kwargs.max_jacobian_bytes`. On limited hardware, lower
`steps.prepare.max_length` or `mutant_batch_size`.

---

## Developer Guide

### Code Layout

```
modules/evaluate/
├── README.md                        ← this file
├── docs/                            ← controls, dataset catalog, dataset sourcing, ProteinGym guides
├── configs/                         ← base config + overlays; see Config files above
├── scripts/dataset_sourcing/        ← dataset sourcing entry points (docs/dataset-sourcing.md)
└── src/
    ├── config.py                    ← EvaluationConfig: top-level Pydantic schema
    ├── run_evaluate.py              ← CLI entry point: runs the four steps in order
    ├── run_proteingym_*.py          ← ProteinGym zero-shot and supervised runners
    ├── runners/                     ← orchestration: shared step pipeline, cross-validation
    ├── dataset/                     ← workspace path layout, dataset loading, collation, CV folds
    ├── dataset_sourcing/            ← sourcing pipeline used by scripts/dataset_sourcing
    ├── proteingym/                  ← ProteinGym sourcing, baselines, scoring
    ├── schemas/artifacts.py         ← typed handoffs between steps
    ├── model/                       ← probe heads, contact head, categorical Jacobian, layerwise probes
    ├── steps/                       ← prepare.py, tune.py, predict.py, score.py
    ├── tasks/                       ← one TaskHandler per task_type (see Adding a Task Type)
    ├── metrics/                     ← metric callables, separation-binned contact metrics
    └── utils/                       ← controls, bootstrap/permutation stats, embedding cache,
                                       HPO, config validation, I/O, tracking
```

Shared config loading helpers live in `modules/core/utils/config_loader.py`.

### Pipeline Architecture & Distributed Execution

The four steps are separate objects (`PrepareStep`/`TuneStep`/`PredictStep`/`ScoreStep`) chained by `run_evaluate.py`:

- **Prepare → Tune → Predict** share an in-memory `PreparedArtifacts` (`schemas/artifacts.py`): model, tokenizer, and train/validation/test dataloaders. When the embedding cache is on, `embedding_backend` also carries the frozen trunk and token dataloader, because `random_trunk` and `scrambled_sequences` controls must re-embed the test split; only `ControlBuilder` (`utils/controls.py`) reaches into it.
- **Score** runs on rank 0 only, as a plain CPU step. `run_evaluate.py` passes it the in-memory `PredictOutput`; without one, it rebuilds predictions from `run_manifest.json` and the parquet files it lists, so scoring can be rerun standalone.

#### Distributed execution (`accelerate`) patterns

Under `accelerate launch`, several processes (ranks) run the same code, usually one per GPU. These patterns recur across `tune.py`, `predict.py`, `utils/controls.py`, and `utils/hpo.py`:

1. **Every rank must reach every collective call.** `accelerator.gather_for_metrics()`, `broadcast()`, and `broadcast_object_list()` are collective operations: every process must call them, in the same order, or the ranks that do call them hang forever waiting for the ones that don't. So a branch that looks like it should be gated on `if accelerator.is_main_process:` (e.g. "only build this control's model") often *can't* be, if anything inside it issues a collective call. `ControlBuilder.build`'s docstring shows a branch that is deliberately *not* wrapped in error handling for this reason.
2. **Decide once on rank 0, then broadcast the decision.** For a choice that must be identical on every rank (e.g. "does a cached file exist," "should this HPO trial be pruned"), only rank 0 evaluates the condition; the boolean is wrapped in a tensor and sent to every rank via `broadcast(..., from_process=0)` before anyone branches on it. Letting each rank decide independently can split them, because filesystem state, floating-point rounding, or the Optuna `trial` object (which exists only on rank 0) can differ.
3. **Freeze/unfreeze parameters *before* `accelerator.prepare()`, never after.** Changing `requires_grad` on a model already wrapped for DDP leaves its gradient-reduction buckets built around the wrong parameter set. `configure_trainable_params()` always runs before `accelerator.prepare(model)` in `TuneStep`.
4. **Classical probes fit identically on every rank.** Each rank holds the full gathered training embeddings and fits the estimator with the same seed, so no live sklearn object has to be broadcast; only rank 0 saves it.

Before adding a control condition, training-loop branch, or cross-rank decision, search for `broadcast(` and `is_main_process` in `utils/controls.py` and `steps/tune.py`; the pattern you need likely exists already.

### Adding a Task Type

Every step (`prepare`/`tune`/`predict`/`score`) is task-agnostic: it looks up a
`TaskHandler` for `workspace.task_type` via `modules.evaluate.src.tasks.get_task`
and delegates all task-specific behavior to it. Adding a new task type means
adding one handler module, not editing the steps.

1. Create `modules/evaluate/src/tasks/<your_task>.py` with a class that
   subclasses `TaskHandler`, sets `name` to the `task_type` string, and is
   decorated with `@register_task`. Override only what differs from the
   defaults in `TaskHandler` (see its docstrings): typically
   `auto_model_class`/`build_model`, `label_dtype`/`collate_labels`,
   `extract_predictions`/`split_batch_rows`, and `metrics`/`default_metric_names`.
   `name`, `auto_model_class` (unless `build_model` is overridden), and at
   least one metric/`default_metric_names` entry are required; a handler
   missing one fails at import time, not at first use.
   Tasks that expand one source example into several tokenized rows (e.g.
   one row per masked residue) override `expand_tokenized_example`; tasks
   whose dataset overlay has no real label column set
   `requires_source_labels = False` so `tokenize_dataset` doesn't require one.
2. Import the new module from `modules/evaluate/src/tasks/__init__.py` so it
   self-registers.
3. Add a task config under `modules/evaluate/configs/tasks/`, and declare the
   accepted dataset target types and label formats on the handler.
   If the head size comes from `workspace.num_labels` (classification-style
   heads), also override `validate_num_labels()` to check it against the labels
   actually present in the dataset (see `sequence_classification.py`/
   `token_classification.py`). This catches a wrong `num_labels` before the
   head is built, instead of as a cryptic index-out-of-bounds error in training.
4. Add focused tests for task-specific behavior. `tests/evaluate/test_tasks.py`
   is parametrized over every registered task, so the new handler is also
   exercised by the shared contract suite automatically.

`contact_prediction` is the non-standard example: it overrides the model
builder, pairwise label collation, ragged prediction-row handling, and contact
metrics. Its labels are sparse positive `[i, j]` residue pairs, with implicit
negative labels for valid pairs at separation six or greater. It uses a
symmetric low-rank pairwise head and stores ragged per-sequence pairs, so
separation-binned precision-at-L is computed without materializing dense pair
features. See
[`contact_prediction.py`](src/tasks/contact_prediction.py) and
[`contact_model.py`](src/model/contact_model.py).

`categorical_jacobian` reuses the contact label and metric contract but swaps
the trainable head for the frozen MLM; see
[Categorical-Jacobian contact prediction](#categorical-jacobian-contact-prediction).

No changes to `prepare.py`/`collator.py`/`dataloader.py`/`predict.py`/`score.py`
are needed unless the task requires something genuinely new (e.g. a non-HF
model class, or pairwise/2D labels); in that case, override the relevant
`TaskHandler` hook rather than adding a branch in the step.

#### Minimal example

The smallest possible handler, a sequence-level classification task scored
only by accuracy, is roughly:

```python
# modules/evaluate/src/tasks/my_task.py
from transformers import AutoModelForSequenceClassification

from modules.evaluate.src.metrics import metrics
from modules.evaluate.src.tasks.base import TaskHandler
from modules.evaluate.src.tasks.registry import register_task


@register_task
class MyTask(TaskHandler):
    name = "my_task"
    auto_model_class = AutoModelForSequenceClassification
    metrics = {"accuracy": metrics.accuracy}
    default_metric_names = ("accuracy",)
```

Then add `from modules.evaluate.src.tasks import my_task  # noqa: F401` to
`modules/evaluate/src/tasks/__init__.py`. See
[`sequence_classification.py`](src/tasks/sequence_classification.py) for a
slightly fuller real example, and
[`token_classification.py`](src/tasks/token_classification.py) for one that
overrides the ragged-label hooks (`is_ragged`, `aligns_labels`,
`align_labels`, `collate_labels`, `split_batch_rows`, `flatten_for_metrics`,
`scramble_labels`).

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Results from before the positional-label or multilabel-control fixes | Rerun affected positional evaluations. Regenerate older softmax-based multilabel controls and their score comparisons; embedding caches remain reusable and trained-model scores are unaffected by the control-probability fix. |
| `One of the subprocesses has abruptly died during map operation` | A tokenization worker exited, often because the job exceeded its CPU RAM allocation. Check Slurm/cgroup OOM counters; `steps.prepare.tokenize_num_proc=1` disables map multiprocessing for diagnosis (Accelerate's `--num_processes` does not). Pseudo-perplexity expands each protein into one masked row per residue, so it uses one source protein per map batch to bound RAM; the expanded on-disk dataset can still be large. |
| `ValueError: Configured <role> source split '...' was not found` | The dataset doesn't publish that split; check `steps.prepare.split` against the listed available splits (see [Named benchmark splits](#named-benchmark-splits)) |
| `ValueError: Packed evaluation requires CUDA.` | `steps.prepare.packed=true` on a machine without a GPU; set it to `false` |
| `head_type='...' requires steps.prepare.embedding_cache.enabled=True` | Classical probes fit on cached pooled embeddings; add `steps.prepare.embedding_cache.enabled=true` or use a `configs/tuning/` overlay |
| Multi-GPU run hangs | A rank skipped a collective call; see [Distributed execution patterns](#distributed-execution-accelerate-patterns) |

---

## Testing

```bash
uv run pytest -n auto tests/evaluate
```

`tests/evaluate/test_tasks.py` runs a shared contract suite over every registered task.
