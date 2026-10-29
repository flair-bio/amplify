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

Evaluation follows five stage APIs:

```
Prepare → Tune → Controls → Predict → Score
```

Tune fits the primary predictor and is skipped for `evaluate_as_is`.
Controls prepares independently fitted controls (including zero-shot controls
when requested); inference-only ablations remain in Predict.

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
  - [Predictor overlays (`configs/predictors/`)](#predictor-overlays-configspredictors)
  - [Layerwise linear probes](#layerwise-linear-probes)
  - [Zero-shot contact prediction](#categorical-jacobian-contact-prediction)
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
and only Prepare → Predict → Score run. It resets the head to the native MLP
path and disables caching, HPO, and refitting, including when layered after an
alternative probe. A CUDA GPU is only needed for packed
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
too, append a [probe overlay](#predictor-overlays-configspredictors) such as
`modules/evaluate/configs/predictors/torch_linear.yaml`.

### Contact prediction

The replacement CASP14-16 benchmark consumes the schema-6.0 Hugging Face dataset
only. `tasks/contact_prediction.py` owns its data/prediction/scoring orchestration;
the existing step classes delegate to that handler and persist its artifacts.
The data contract/alignment also live in that task module, shared extraction in
`model/inference.py`, and ranking/aggregation in `metrics/contact_metrics.py`.
The frozen-embedding, attention, and zero-shot categorical-Jacobian predictors
run through the normal CLI and prediction entry point. CASP controls remain a
subsequent stage. The old Biomap contact overlay and embedding head are removed.
The dataset selects the shared `contact_prediction` task; an explicit predictor
overlay selects extraction, fitting and inference without changing that task.

```bash
uv run python modules/evaluate/src/run_evaluate.py \
    modules/evaluate/configs/config.yaml \
    modules/evaluate/configs/models/amplify_120m.yaml \
    modules/evaluate/configs/datasets/contact_prediction_casp.yaml \
    modules/evaluate/configs/predictors/contact_embedding.yaml \
    "workspace.dataset_revision=<immutable-40-character-HF-commit>" \
    workspace.run_id=casp-embedding \
    seed=0
```

For the attention readout, replace `contact_embedding.yaml` with
`contact_attention.yaml` and use a distinct `workspace.run_id`.

The final residue embeddings are frozen, unpooled and unnormalized. Two
independent bias-free 512-dimensional projections produce averaged directional
logits plus a zero-initialized scalar bias. Fitting uses all known unordered
training pairs at original separation >=6, equal pair weighting, AdamW
(lr=0.001, matrix weight decay=10, bias decay=0), and one effective 20-protein
batch per epoch. Validation uses mean structure P@L over the fixed 20 validation
structures, strict improvement and patience 10; the best head is restored.
There is no HPO, class weighting, negative subsampling, scheduler, clipping,
feature standardization or train+validation refit. `max_epochs` is a safety
limit: reaching it before patience is a failed fit.

The attention preset uses all layer/head probability maps, sum-symmetrizes each
map, applies APC, and fits one L1 logistic regression (`C=0.15`,
`solver=liblinear`) on all known unordered training pairs at original separation
>=6. It does not standardize features or select settings on validation data.
Training attention maps are transient, and test maps are never persisted.
Select it with `configs/predictors/contact_attention.yaml` instead of the
embedding preset.

### Categorical-Jacobian contact prediction

The zero-shot categorical Jacobian shares the CASP dataset, alignment, unit
scoring, and full-chain inference path. Select it with
`configs/predictors/contact_jacobian.yaml`. It computes canonical-amino-acid
mutant-minus-wild-type log-softmax deltas, keeps the Jacobian in FP32, centers
all four categorical/residue axes, symmetrizes directional effects, applies
the Frobenius reduction and APC, and removes self-pairs. It does not fit a
head or persist test Jacobians. This preset requires a masked-LM checkpoint
with one tokenizer token per canonical amino acid.

`tokenize_contact_chain` supplies uncropped tokens and original-coordinate
position IDs. Native AMPLIFY and ESM rotary computations preserve gaps and
grow positional caches without rescaling. `PreparedArtifacts.contact_dataset`
passes these chains to the standard `PredictStep.run()`, which delegates
scientific processing to the task handler and uses ordinary prediction
persistence. Pass a local `workspace.dataset_repo_id` directory and omit the revision for
offline Parquet loading. No source coordinate files or source-manifest joins
are used. The runtime dataset fingerprint covers the scoring rows and fixed
development splits.

Inputs are cleaned once per chain; global/EU rows share full-chain inference,
and exact original sequences share predictions without merging their identities.
Reference-free chains skip inference but retain unscored rows. Test features
are transient; persisted predictions contain candidate scores, labels and
original pair indices, not embeddings or attention maps.
Development embeddings are cached across seeds; attention maps are not cached.
Embedding-cache keys include the backbone weights/configuration, tokens,
original positions and precision. Use
`steps.prepare.embedding_cache.cache_policy=off` to disable the embedding cache.
Run one seed per invocation; final repetitions use seeds 0 through 4.

Score writes `unit_scores.parquet` and `scores.parquet`, reporting both original
separations (>=6 and >=24), AUC (mean precision at ten length cutoffs), P@L,
P@L/2, P@L/5 and ROC-AUC. EUs, chains and complexes are averaged equally within
each edition, with metric-specific contributing counts. Development validation
averages structures equally. Generic flat bootstrap and controls are rejected
until their CASP-specific integration is implemented.

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
| `contact_prediction` | Which residue pairs touch (fixed CASP protocol) |
| `pseudo_perplexity` | Nothing; scores masked residues, zero-shot |
| `calibration` | Homolog amino-acid profiles at conserved positions, zero-shot |

`pseudo_perplexity` and `calibration` run only with
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

The base [`configs/config.yaml`](configs/config.yaml) defines the complete default
protocol: an MLP probe trained over cached embeddings from a frozen trunk, with
100-trial hyperparameter search and fresh train+validation refitting. Supply the
base, model, and dataset configs; no MLP tuning overlay is needed. Full trunk
fine-tuning and other probe types remain explicit alternatives in `configs/predictors/`.
Composition is base → model/task → dataset → optional alternative → smoke
settings → CLI overrides; later values win.

Task overlays explicitly select the reported metric and the validation
selection metric/direction: Macro-F1 for classification, R² for regression
(ProtParams), pooled Spearman for the current stability dataset, long-range
contact AUC for contacts, calibration agreement, and pseudo-perplexity.
Only the selected metric is reported by default; override `steps.score.metrics`
to request diagnostic metrics. Stability's intended grouped Macro-Spearman
still requires verified parent-design metadata and scorer support. Calibration
agreement reports `100 × mean(exp(-KL))` over per-position KL values; it does
not exponentiate the mean KL. The target/non-target breakdowns use the same
scale and aggregation.
Classification F1 includes every class configured by `workspace.num_labels`,
even when absent from a split or resample (zero F1 for an absent class).
This includes background class 0, both binary classes, all three Q3 states,
and all eight Q8 states. Multilabel F1 averages over the configured indicator
columns, including zero-only columns. Validation selection, final scores,
controls, bootstrap, and permutations share this policy; token positions
labelled `-100` remain excluded. Precision and recall use the same vocabulary.

The base MLP protocol and neural tuning alternatives enable 100 TPE trials,
with learning-rate bounds
`[1e-5, 1e-3]`, epoch choices `[1, 2, 5, 10, 20, 40, 100]`, cosine decay,
and no early stopping. The cached MLP searches widths `[128, 256, 512]`,
hidden depths `[1, 2]` (each hidden layer is Linear-GELU), and
batches `[8, 16, 32, 64, 128, 256]`; token batches count valid residues, not
proteins. TPE samples weight decay continuously on a log scale from `1e-6`
through `1e-1` (`weight_decay_sampling: log`). Set this to `categorical`
for discrete candidates, including zero decay; log bounds must be finite and
positive. Random/grid searches still use the listed candidates, as they do
for learning rate. The default MLP and cached tuning alternatives enable fresh
train+validation final refitting after selection; live/native fine-tuning and layerwise refit remain
implementation work.
To use fixed training settings instead, set
`steps.tune.hyperparameter_search.enabled=false`;
`steps.tune.learning_rate`, `weight_decay`, `max_epochs`, and `batch_size` then
apply. During HPO, searched values replace their fixed counterparts, not cap them.
Classical-estimator overlays retain their estimator-specific spaces.
Calibration and pseudo-perplexity task overlays disable caching, HPO, and
refitting. CASP predictor overlays fix their feature extraction or fitting
protocols and reject generic HPO and controls.

`datasets/` selects the source, label contract and default scientific task.
`tasks/` owns task data handling and scoring. `predictors/` selects how predictions
are produced across preparation, optional fitting and inference, including future
zero-shot contact predictors; it is not restricted to tuning.

| File | Use |
|---|---|
| `configs/config.yaml` | Complete default cached-MLP benchmark protocol |
| `configs/smoke_test.yaml` | Overlay for a quick, bounded smoke test |
| `configs/models/*.yaml` | One overlay per evaluated model (name, repo) |
| `configs/tasks/*.yaml` | Task type, behavior, and default controls |
| `configs/datasets/*.yaml` | One overlay per dataset; see the [catalog](docs/datasets.md) |
| `configs/predictors/*.yaml` | Explicit prediction approaches, including CASP embeddings and alternative probes |
| `configs/accelerate/*.yaml` | `accelerate launch` configs (e.g. single-process local runs) |

Dataset overlays declare `workspace.default_task_type`, for example
`sequence_multilabel_classification` in `datasets/GO_bp.yaml`. When no task
YAML is supplied, `run_evaluate.py` automatically inserts the corresponding
`configs/tasks/<default_task_type>.yaml` immediately before the dataset overlay.
This loads the task's training mode, embedding cache settings, and control
conditions as well as its task type. Dataset settings, later overlays, and CLI
overrides can still customize those defaults:

```bash
uv run accelerate launch modules/evaluate/src/run_evaluate.py \
    modules/evaluate/configs/config.yaml \
    modules/evaluate/configs/models/amplify_120m.yaml \
    modules/evaluate/configs/datasets/GO_bp.yaml
```

An explicit task YAML takes precedence over the dataset default, in either
order. The CASP dataset defaults to `contact_prediction` and requires
an explicit predictor overlay such as `contact_embedding.yaml`,
`contact_attention.yaml`, or `contact_jacobian.yaml`. Ordinary tasks retain
their default MLP behavior without a predictor overlay. `pseudo_perplexity` works with any compatible sequence
dataset; `uniprot_ur100p_test` selects the AMPLIFY paper's UniProt test set.
Without an explicit task YAML, a CLI `workspace.task_type=...` selects a
different task overlay. The baseline Slurm launcher accepts dataset-only rows
plus an optional task column for extra methods; ProteinGym rows select their
separate runner explicitly.

The entry point is `modules/evaluate/src/run_evaluate.py`. Explicit YAML files
are merged left-to-right (later files win), followed by optional `key=value`
overrides:

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

Sequence task presets request shared `majority_class` or `train_mean`
baselines. Generate them before model evaluation using the same dataset,
split settings, and `workspace.base_path` (no model overlay required):

```bash
uv run python -m modules.evaluate.src.run_evaluate --baselines \
    modules/evaluate/configs/config.yaml \
    modules/evaluate/configs/datasets/solubility_prediction.yaml
```

Model scoring loads these artifacts without fitting them. Use
`baselines.names=[]` to disable shared baselines. See [Shared sequence
baselines](docs/controls.md#shared-sequence-baselines).

CUDA evaluation enables TensorFloat-32 by default with top-level `tf32: true`,
matching pretraining's `trainer.tf32` convention. Append `tf32=false` to any
standard or ProteinGym evaluation command to disable it. This controls CUDA
FP32 matrix multiplication and cuDNN; tensor dtypes and autocast settings are
configured separately. The setting applies to inference and probe training.
TF32 can change numerical scores while speeding up evaluation; see [PyTorch's precision documentation](https://docs.pytorch.org/docs/2.14/generated/torch.set_float32_matmul_precision.html).
Embedding cache keys include backend precision, so changing this setting
recomputes embeddings instead of reusing outputs from another precision.

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
DATASETS=(fold_prediction localization_prediction solubility_prediction)
model=${MODELS[$((SLURM_ARRAY_TASK_ID / ${#DATASETS[@]}))]}
dataset=${DATASETS[$((SLURM_ARRAY_TASK_ID % ${#DATASETS[@]}))]}

uv run accelerate launch modules/evaluate/src/run_evaluate.py \
    modules/evaluate/configs/config.yaml \
    modules/evaluate/configs/models/${model}.yaml \
    modules/evaluate/configs/tasks/sequence_classification.yaml \
    modules/evaluate/configs/datasets/${dataset}.yaml \
    workspace.base_path=/path/to/eval
```

Give each array task its own model/dataset/run combination. Score tables are
run-local, so different runs do not overwrite each other's rows. Embedding
caches remain shared across a model's runs; avoid concurrent cache refreshes
for the same model/dataset pair.

### Common recipes

Append these overrides to any launch command, or put them in an overlay YAML.

| Goal | Overrides |
|---|---|
| Skip tuning and score the checkpoint as-is | `steps.mode=evaluate_as_is steps.prepare.embedding_cache.enabled=false steps.tune.hyperparameter_search.enabled=false steps.tune.refit_on_train_validation=false`. For supervised tasks the head comes from the checkpoint; a pretrained-only checkpoint gets a randomly initialized head, so use this for fine-tuned checkpoints or zero-shot tasks |
| Fine-tune the trunk and native MLP head | Append `configs/predictors/mlp_finetuning.yaml`; it disables caching/refitting and clears cached-MLP search dimensions |
| Bootstrap confidence intervals and control tests | `steps.score.bootstrap_enabled=true` |
| Disable TF32 for CUDA FP32 operations | `tf32=false` (default: `tf32=true`) |
| Use a generated split instead of the default | `steps.prepare.split.name=stratified` (see [Named benchmark splits](#named-benchmark-splits)) |
| Classical probe on cached embeddings | Add `modules/evaluate/configs/predictors/random_forest.yaml` after the dataset overlay |
| Disable control conditions | `steps.predict.controls=[]` |
| Disable shared sequence baselines | `baselines.names=[]` |
| Write results elsewhere | `workspace.base_path=/path/to/eval` |
| Group model versions into one benchmark comparison | `workspace.base_path=/path/to/client-benchmark-v1` |
| Keep another evaluation of the same model/probe | `workspace.run_id=run02` |
| Label a specific checkpoint version | `workspace.model_name=amplify-350m-muon workspace.model_repo_id=/path/to/checkpoint` |
| Override the benchmark output root | `workspace.output_dir=/path/to/client-benchmark-v1` (task/dataset are appended; applies to models and baselines) |

### New dataset, task, or model overlay

Copy an existing file under `configs/datasets/`, `configs/tasks/`, or
`configs/models/`. Dataset overlays set the source, target type, label format,
class count, and `workspace.default_task_type`. Task overlays set the effective
`task_type` and task behavior. Task-specific
builder parameters belong in `workspace.model_kwargs`.

### Config schema

All config classes are Pydantic models composed into a single `EvaluationConfig` (`src/contracts/config.py`):

```
EvaluationConfig
├── tf32:      bool (default: true)
├── workspace: EvaluationWorkspaceConfig
├── wandb:     WandbConfig
├── external_test_sets: list[ExternalTestSetConfig]
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

The base protocol enables embedding caching. For `finetune` or `evaluate_as_is`,
use the finetuning/smoke overlay **after** the task YAML, or explicitly disable
caching and refitting (and disable HPO or clear cached-MLP search dimensions).
Zero-shot task YAMLs already handle these exceptions. Sequence features support
disk reuse; token features stay in
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

| Field | Schema default | Description |
|---|---|---|
| `head.head_type` | `mlp` | `mlp` (cached MLP by default; native head without caching), `torch_linear` (one `torch.nn.Linear` layer), `sklearn_linear` (sklearn logistic/ridge), `knn`, `random_forest`, or `xgboost` |
| `head.search_space` | `{}` | Classical-estimator dimensions; candidates fit on train and are selected on validation, before optional final refitting |
| `refit_on_train_validation` | `false` | Base YAML sets `true`: rebuild the selected probe on train+validation without early stopping or validation selection. Requires nonempty development splits; live/native and layerwise refit are unsupported |
| `metric_aggregation` | `weighted` | Averaging used for validation metric selection; automatically follows `steps.score.metric_aggregation` unless set explicitly; supports `micro`, `macro`, `weighted`, or `samples` |
| `learning_rate` | `1e-4` | Used when `hyperparameter_search.enabled=false` |
| `weight_decay` | `1e-4` | Used when `hyperparameter_search.enabled=false`; applies to matrices and `ZeroCenteredRMSNorm` gain offsets. Other one-dimensional parameters are excluded |
| `max_epochs` | `10` | Maximum training epochs; used when `hyperparameter_search.enabled=false` |
| `warmup_ratio` | `0.06` | Fraction of total training steps used for LR warmup |
| `max_grad_norm` | `1.0` | Gradient clipping norm |
| `accumulation_steps` | `1` | Batches accumulated per optimizer step |
| `batch_size` | `128` | Cached-embedding probe batch size; set to `null` to use the PrepareStep loader size |
| `standardize_features` | `true` | Standardize ordinary cached MLP/Torch linear inputs with training-set statistics; layerwise probing uses its own setting |
| `early_stopping_patience` | `5` | Stop early after this many epochs with no validation improvement |
| `optimizer` | `adamw` | One of `adamw`, `adam`, `sgd`, `rmsprop`, `adagrad`, `adafactor` |
| `optimizer_kwargs` | `{}` | Extra optimizer constructor kwargs, e.g. `{momentum: 0.9}` for `sgd` |
| `lr_scheduler_type` | `linear` | Any name accepted by `transformers.get_scheduler` |
| `scheduler_kwargs` | `{}` | Extra scheduler-specific kwargs, e.g. `{num_cycles: 3}` |
| `save_best_model` | `true` | Persist the best model + tokenizer to `workspace.model_path` after tuning |
| `variance_seeds` | `[]` | Optional extra seeds for logging validation-metric mean/std; diagnostic only and supported for cached-embedding probes |
| `hyperparameter_search.enabled` | `false` | Enable hyperparameter search instead of a single fixed-config run |
| `hyperparameter_search.n_trials` | `5` | Total trial budget for BOHB/random; grid evaluates every combination |
| `hyperparameter_search.direction` | `null` | `maximize` or `minimize` the target metric; inferred from the metric when omitted |
| `hyperparameter_search.metric` | `null` | Task-specific default metric unless overridden |
| `hyperparameter_search.search_pattern` | `bohb` | `bohb` (Optuna TPE sampler with median pruning), `random`, or `grid` |
| `hyperparameter_search.search_space.*` | see [`configs/config.yaml`](configs/config.yaml) | Candidate values for `learning_rate`, `max_epochs`, and `weight_decay`; optional cached-MLP dimensions are listed below |
| `hyperparameter_search.search_space.weight_decay_sampling` | `log` | TPE: continuous log-scale sampling between min/max `weight_decay` bounds; `categorical` permits discrete candidates including zero. Random/grid always use the listed candidates |
| `hyperparameter_search.search_space.hidden_size` | `null` | Optional hidden widths for cached MLP probes; `null` keeps the fixed `head.hidden_size` |
| `hyperparameter_search.search_space.hidden_depth` | `null` | Optional hidden-layer counts for cached MLP probes; `null` keeps `head.hidden_depth` (default 1). Native heads and layerwise probes do not support this search |
| `hyperparameter_search.search_space.batch_size` | `null` | Optional cached MLP training batch sizes; `null` keeps `steps.tune.batch_size` or the prepared loader size |

The base `configs/config.yaml` searches hidden widths `[128, 256, 512]`,
hidden depths `[1, 2]`, and probe batch sizes `[8, 16, 32, 64, 128, 256]`
jointly with learning rate, weight decay, and epochs in 100 trials total.
BOHB and random search cover each width, depth, and batch size in the initial
trials when the budget permits; grid search evaluates every combination.
Width, depth, and batch-size search require cached MLP
probes. Probe batches contain sequences for sequence tasks and individual
residues for token tasks. Each trial uses its batch size for training,
validation, and the optimizer schedule; the selected batch is saved with the
winning hyperparameters.
Candidates reuse training-only feature standardization and are selected by the
validation metric. With `refit_on_train_validation: true`, a fresh head with the
selected architecture is trained on combined train+validation data for the
selected full epoch budget, using its selected batch size and a new optimizer
and scheduler. Feature statistics are recomputed on that development union;
test data is never used for fitting. Early stopping and pruning are disabled
during refit. With HPO disabled, the configured fixed architecture and budget
are fit directly on development data. Saved weights and preprocessing belong
to the final refit; `final_fit_split` records its training scope in checkpoint
metadata. With refit disabled, the selected trial's weights are retained.
With no architecture search configured,
`head.hidden_size` and `head.hidden_depth` set the fixed architecture.
Seed-variance diagnostics still run on the original train/validation splits
before refitting. Those loaders are retained for controls; `majority_class`
and `train_mean` continue to use train-only labels. Cached `random_trunk`
independently selects and refits its own probe with the same search budget
and training protocol on random-trunk features, including its own preprocessing.
Its checkpoint is saved under `<variant>-random_trunk` when saving is enabled.

For classical `head_type` values, put fixed estimator settings in
`head.hyperparameters` and swept dimensions in `head.search_space`. Every
candidate trains only on `train` and is compared on `validation`. Cached
classical overlays enable final refitting: a fresh estimator with the selected
settings, including its scaler where applicable, is fit on train+validation.
A validation split is required for search or refit; with refit disabled,
the winning training-only estimator is retained. With
`hyperparameter_search.enabled=false`, every combination is evaluated as a
grid. With it enabled, `n_trials` and `search_pattern` select grid, random, or
Optuna BOHB-style trials over those estimator parameters. Classical estimators
fit atomically, so BOHB cannot prune within a fit.

`torch_linear` is trained through the PyTorch optimization loop and has no
hidden projection: it uses the selected cached embedding
vector. For sequence-level classification/regression it requires
`steps.prepare.embedding_cache.enabled=true`; for token classification it can
also use a compatible native `nn.Linear` model head without the cache.
For native token probing, explicitly disable `refit_on_train_validation` when
disabling the cache; final refitting currently supports cached probes only.
Classical probes always require `steps.mode=tune_probe` and
`steps.prepare.embedding_cache.enabled=true`. Set `embedding_cache.layers: [-1]`
for the final layer, or list any valid model hidden-state indices to concatenate
pooled representations from several layers.

Ordinary cached `mlp` and `torch_linear` probes standardize each feature with
the training-set mean and population standard deviation. Sequence tasks use
pooled embeddings; token tasks use valid training residues, excluding ignored
labels. Constant features use a standard-deviation floor of `1e-6`. Validation
and HPO trials reuse the training transformation; final refitting recomputes it
on train+validation. Random-trunk probes fit their own transformation on random
features. Test data never contributes statistics. Set
`steps.tune.standardize_features=false` to use raw features.

The fitted statistics are saved in `probe_head.pt`, and
`probe_head_config.json` records `standardize_features` for reconstruction.
Raw embedding caches remain reusable across this setting. Standardization is
fixed during HPO and does not change regression targets or
`normalize_before_pooling` (per-residue L2 normalization). Layerwise probing
uses `steps.layerwise.standardize_features`; native uncached heads and
fine-tuning do not use this cached-probe setting. Classical and ProteinGym
supervised probes retain their existing preprocessing.

Cached validation features stay on the device across epochs, and validation
inference skips loss computation.

**Input:** `PreparedArtifacts` from the Prepare step (or a resumed checkpoint)
**Output:** in-memory updated `PreparedArtifacts.model`; `<run>/model/` on disk when `save_best_model=true`

Contact checkpoints save the trunk and tokenizer alongside `contact_head.pt`
and `contact_config.json`. Reload them with
`get_task("contact_prediction").load_finetuned_model(path)`.

---

## Step 3 — Predict

INFO logs report start/completion and elapsed time for trained prediction,
model controls, control embedding extraction/cache I/O, and persistence.
Long prediction/extraction loops report completed batches at most every 10
seconds (after a batch finishes), on the main process only. Layerwise logs name
conditions sharing a forward pass; validation loops remain quiet.

**Purpose:** Run inference over the `test` split for the trained model and for each configured control condition. Control predictions are rebuilt every run from the current head; only frozen-trunk embeddings are reused. No metrics are computed here — that's [Score](#step-4--score)'s job.

**Control conditions (`controls`):** each removes one possible source of the trained model's performance, so [Score](#step-4--score) can test whether the trained model still beats it.

| Control | What changes |
|---|---|
| `untrained` | Pretrained checkpoint with a fresh head |
| `random_head` | Trained trunk, re-initialized head |
| `random_trunk` | Random trunk with independently tuned/refitted cached probe; retrained layerwise probes; trunk-replacement ablation in live/native runs |
| `random_both` | Trunk and head re-initialized |
| `scrambled_sequences` | Residues shuffled within each sequence |
| `scrambled_labels` | Trained predictions scored against permuted labels |
| `majority_class` | Train class prior; inside Predict for token tasks only |
| `separation_prior` | Train contact rate per residue separation |

Per-task presets, invalid combinations, and how to read the results are in
[Controls](docs/controls.md).

**Key config fields (`steps.predict`):**

| Field | Default | Description |
|---|---|---|
| `collect_probabilities` | `false` | Collect task-appropriate per-class/per-label probabilities (softmax for single-label, sigmoid for multilabel); auto-enabled when an effective metric requires probabilities, such as `roc_auc` or `average_precision` |
| `predictions_filename` | `predictions.parquet` | Output filename directly under the run directory |
| `persist_to_disk` | `true` | Write predictions and the prediction manifest; disable for fast in-process evaluation |
| `store_sequences` | `false` | Decode and include source sequences in the main predictions table |
| `controls` | `[random_trunk]` | Per-model controls; token/contact presets add their existing priors. Sequence priors use top-level `baselines.names` and the separate `--baselines` route |
| `embedding_batch_size` | `null` | Cached-feature rows per prediction batch, including model-based controls; `null` inherits the prepared loader size. Does not change live trunk batches. |

Cached token probes batch **residues**, not proteins. The token-classification
task overlay sets `steps.tune.batch_size=1024` for cached training/validation and
`steps.predict.embedding_batch_size=16384` for cached prediction and control
heads. Protein batches remain controlled by `steps.prepare.batch_size` (live
inference) and `steps.prepare.embedding_batch_size` (feature extraction).
Changing the training batch size changes optimization and the learning-rate
schedule's update count; compare validation quality when tuning it. Set
`steps.tune.batch_size=null steps.predict.embedding_batch_size=null` to restore
the previous inherited batch sizes. These settings do not change uncached
training or inference. Layerwise controls that require live trunk features
continue to use protein batches.

For cached sequence classification, multilabel classification, and regression,
the task overlays set prediction batches to 1,024 examples. Cached probe tuning
uses the searched batch size by default. With HPO disabled, it uses
`steps.tune.batch_size`, falling back to the prepared loader batch size when
null. Embedding extraction has its own `steps.prepare.embedding_batch_size` setting.

**Input:** `PreparedArtifacts` (trained model + test dataloader) from Prepare/Tune
**Output:** `<run>/predictions.parquet` and `<run>/controls/<control>/predictions.parquet`; settings and control paths are recorded in `<run>/run_manifest.json`'s "predict" section.

---

## Step 4 — Score

All tasks implement `score(predictions, config, seed=...)` without filesystem
access. Tasks select metrics, aggregation and supported uncertainty methods;
`metrics/scoring.py` supplies flat-task scoring and paired statistical analysis.
`ScoreStep` loads baseline artifacts, invokes that common contract, then publishes
the returned flat or structured results. It has no contact-specific dispatch.

**Purpose:** Compute task-specific metrics from the Predict step's output and, optionally, bootstrap confidence intervals for those metrics plus a statistical comparison between the trained model and each configured control condition, to quantify how much of its performance is attributable to learning from the data rather than model/data artifacts.

Bootstrap and permutation tests reuse predictions, not model inference.
Multilabel AP/MCC use optimized array calculations, multiclass ROC-AUC retains
one-vs-one semantics without explicit class-pair loops, and contact resampling
reuses per-protein scores.
Token-classification resampling with `micro`, `macro`, or `weighted` aggregation
precomputes per-protein confusion matrices for accuracy, F1, precision, recall,
and MCC. Bootstrap samples sum counts with protein multiplicities; paired
permutations swap whole protein matrices. Metrics are computed from pooled
counts, not averaged protein scores. This preserves protein grouping and seeded
draws while avoiding repeated scans of all residue predictions. Other tasks,
custom metrics, and unsupported aggregation modes retain the existing path.

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

**Input:** `<run>/predictions.parquet`
**Exact scores:** `<run>/condition_scores.parquet` contains one row per
condition/metric with full-test `score` and evaluation metadata. `trained`
identifies the evaluated model (the frozen-trunk probe in `tune_probe` mode);
other conditions use their control/baseline names. This table is written even
without bootstrap, preserves undefined scores as NaN, and is replaced on
rescoring. Unlike comparison-row bootstrap means, these are exact test scores.

**Output:** `<run>/scores_long.parquet` (always), plus `<run>/bootstrap_results.parquet` when `bootstrap_enabled=true`; scores/confidence intervals are also recorded in `<run>/run_manifest.json` under the `score`/`bootstrap` sections. Concatenate run-local score tables when comparing models or repeated runs.
`bootstrap.control_comparisons` records each control's bootstrap mean, the
trained-minus-control mean difference and confidence interval, and raw/adjusted
permutation p-values (an empty object when there are no comparisons). These
intervals describe the paired differences, not standalone control-score intervals.

---

## Outputs

### Output disk layout

`get_evaluation_workspace()` (in [`src/dataset/workspace.py`](src/dataset/workspace.py))
groups outputs by **benchmark root -> task -> dataset -> baselines/models -> run**:

```
<base_path>/<task_type>/<dataset_name>/
├── baselines/<baseline>/fit-<12-char-hash>/
│   ├── fit_manifest.json
│   └── test-<12-char-hash>/
│       ├── manifest.json
│       ├── predictions.parquet
│       └── log
└── models/<model_name>/
    ├── workspace_manifest.json
    ├── tmp/embeddings/                 ← reusable across model runs
    └── <variant>-<run_id>/seed_<seed>/
        ├── model/
        ├── predictions.parquet
        ├── controls/random_trunk/
        │   ├── model/
        │   └── predictions.parquet
        ├── bootstrap_results.parquet
        ├── scores_long.parquet
        ├── condition_scores.parquet      ← exact scores for model, controls, baselines
        ├── layerwise_comparison.parquet  ← when layer comparisons apply
        ├── run_manifest.json
        └── log
```

`workspace.base_path` names the benchmark root, for example
`eval_results/controls_verification`. No extra `experiments/` or `runs/` level
is added. `workspace.run_id` is required for model runs; the base YAML sets
`run01`. It is appended to the variant, for example `mlp-frozen_trunk-run01`.
Override it to retain another run. Use the same ID across compared models,
and short, distinct
`model_name` labels for different checkpoints, for example
`amplify-350m-reference`, `amplify-350m-muon`, and `amplify-350m-mask30`.
Repository IDs and local paths are input locations, not directory names.
Run/model/dataset labels must be safe single-directory labels (letters,
digits, dots, underscores, hyphens; starting with a letter or digit).

For training runs, `<variant>` is `<head_type>-frozen_trunk` for `tune_probe`
or `<head_type>-tuned_trunk` for `finetune`; `evaluate_as_is` uses `pretrained`.
A non-default `steps.prepare.split.name` is appended to the variant.

Ordinary pipeline runs refuse existing variant outputs, including interrupted
evaluations. Choose a new `workspace.run_id` to retain both experiments, or
explicitly pass `--overwrite` to replace only that variant/run seed and its derived
layer outputs. Nested controls and run-local comparison tables are replaced too;
other seeds, variants, models, caches, and shared baselines are preserved. Matching
concurrent jobs fail rather than overwriting an active run. There is no
ordinary-run resume mode.

Ordinary CLI runs isolate artifacts under `seed_<seed>`. The experiment directory
also records seed-independent settings; sibling seeds cannot use different
settings or, for CASP embeddings, different dataset/backbone content. Historical
unseeded outputs are not migrated or removed: choose a new run ID.

Even with `--overwrite`, a model/dataset label already associated with a
different input source cannot be reused in the same workspace: choose a new
label or benchmark root. Treat checkpoint versions as immutable; replacing file
contents at an unchanged source path does not automatically create a new
version label.

`workspace.output_dir` overrides the **benchmark root** for both model and
baseline jobs. It does not change model loading. Keep the root stable to
reuse baseline artifacts across run IDs. Baseline directories use short hashes
for navigation; full identities are verified in manifests, and collisions
raise an error rather than reusing incompatible results. Separate fit/test
identities allow one fit to serve internal and external tests.

Application logs are written directly to `log` inside each ordinary model run
and baseline test directory; compatible baseline reruns append to their log.
Model logs start after output reservation; baseline logs start after data
resolution. Shell timing output and progress bars remain in the terminal.
Directories are created only when needed. Fitted random-trunk probes are not
separate sibling runs: their models and predictions live together in
`controls/random_trunk/`. Layerwise outputs remain separately named layer runs.
Existing output directories are not moved or removed automatically. These
guards apply to pipeline execution; individual step APIs remain usable for
standalone scoring and low-level workflows.

### External test sets

`external_test_sets` lists test-only datasets scored with the model fitted on
the workspace dataset. They are never used for training, tuning, or selection,
and must share the main dataset's label contract. After the main Predict and
Score, each set is tokenized with the same settings, embedded by the same
frozen trunk when the cache is on, and run through the same fitted probe,
standardization, and controls. Matching the number of classes is not enough:
the class-index-to-label mapping must also match. The current BioMap
localization card documents integer labels but not their class names, so
verify its mapping against setHARD before interpreting that external score.
`localization_prediction.yaml` configures `flair-bio/sethard` this way:

```yaml
external_test_sets:
  - name: sethard
    dataset_repo_id: flair-bio/sethard
    split: test
```

Outputs use the external dataset's own workspace and the same variant:
`<task>/<external.name>/models/<model>/<variant>-<run_id>/`.
Its comparison table lives directly inside that run.
All requested output targets are checked before training or replacement. Sets are
skipped with a warning under `evaluate_as_is` (no fitted probe).

### Reading results

`<workspace>/scores/scores_long.parquet` is the table for comparing runs.
Every row carries the run identity (`model_name`, `dataset_name`, `task_type`,
`split`, `model_repo_id`, `dataset_repo_id`, `head_type`, `layer_index`,
`variant`, `seed`) plus a `metric`:

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

### Predictor overlays (`configs/predictors/`)

Cached MLP probing is already the base default. Each alternative probe overlay
(`knn.yaml`, `sklearn_linear.yaml`, `random_forest.yaml`,
`xgboost.yaml`, `torch_linear.yaml`) is self-contained: it sets
`steps.mode=tune_probe` and the embedding cache itself, so it is just one more
file after the dataset overlay:

```bash
uv run accelerate launch modules/evaluate/src/run_evaluate.py \
    modules/evaluate/configs/smoke_test.yaml \
    modules/evaluate/configs/models/amplify_120m.yaml \
    modules/evaluate/configs/tasks/sequence_classification.yaml \
    modules/evaluate/configs/datasets/localization_prediction.yaml \
    modules/evaluate/configs/predictors/random_forest.yaml
```

The exception is `mlp_finetuning.yaml`, which trains the trunk too and so does
not use the embedding cache.

These are composable presets over existing `steps.prepare`, `steps.tune` and
`steps.predict` settings, not a separate selector or registry. Later overlays
and CLI overrides remain authoritative, subject to normal task/protocol validation.
Existing metadata and output variants identify the head and trunk condition;
contacts use `bilinear512-frozen_trunk` under the shared `contact_prediction` task.
The former `configs/tuning/` paths and
`contact_prediction_embedding` task identity are removed, without aliases.
Calibration, pseudo-perplexity and the separate ProteinGym runners are unchanged.

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
    steps.tune.hyperparameter_search.enabled=false \
    steps.tune.refit_on_train_validation=false \
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
- Extraction uses float32 tensors by default; float16/bfloat16 autocast is opt-in.
  CUDA FP32 computation follows the top-level `tf32` setting.

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

### Contact predictor presets

The CASP contact predictors are composable overlays on the shared
`contact_prediction` task. Use `contact_embedding.yaml` for the fitted
bilinear512 head, `contact_attention.yaml` for the fitted attention readout,
or `contact_jacobian.yaml` for zero-shot masked-LM mutation effects. The
obsolete standalone `categorical_jacobian` task has been removed.

---

## Developer Guide

### Code Layout

```
modules/evaluate/
├── README.md                        ← this file
├── docs/                            ← controls, dataset catalog, dataset sourcing, ProteinGym guides
├── configs/                         ← base config + overlays; see Config files above
├── scripts/dataset_sourcing/        ← sourcing entry points; reusable helpers in shared/ (docs/dataset-sourcing.md)
└── src/
    ├── run_evaluate.py              ← CLI entry point: runs the stage lifecycle
    ├── run_proteingym_*.py          ← ProteinGym zero-shot and supervised runners
    ├── contracts/                   ← configuration/loading/validation and typed handoffs
    ├── runners/                     ← shared step pipeline
    ├── dataset/                     ← loading/alignment, feature datasets/cache, baseline artifacts, workspaces
    ├── proteingym/                  ← ProteinGym sourcing, baselines, scoring
    ├── model/                       ← architectures, extraction, inference, weight ablations
    ├── trainers/                    ← selection/HPO/refit and Torch/classical/contact/layerwise fitting
    ├── steps/                       ← stage lifecycle, cache access and publication
    ├── tasks/                       ← scientific preparation/fitting/inference/scoring and capabilities
    ├── metrics/                     ← scoring, hierarchical aggregation, bootstrap/comparisons, tables
    └── utils/                       ← storage, I/O, seeds, tracking, amino acids;
                                       narrow ProteinGym pooling compatibility export
```

Shared config loading helpers live in `modules/core/utils/config_loader.py`.

### Pipeline Architecture & Distributed Execution

The runner chains `PrepareStep`, optional `TuneStep`, `ControlsStep`,
`PredictStep` and `ScoreStep`:

- **Prepare → Tune → Predict** share an in-memory `PreparedArtifacts` (`contracts/artifacts.py`). Execution-specific state lives in `embedding`, `layerwise`, or `contact` payloads; incompatible contact/loader combinations fail at construction. The embedding backend retains the frozen trunk and token loaders. Fitting services return `FitResult`; the stage owns publication. Independently fitted random-trunk probes reuse those services without nested stages.
- **Score** runs on rank 0 only, as a plain CPU step. `run_evaluate.py` passes it the in-memory `PredictOutput`; without one, it rebuilds predictions from `run_manifest.json` and the parquet files it lists, so scoring can be rerun standalone.
- **Controls** owns independent cached, layerwise and task-specific control
  preparation/fitting. Tune fits only the primary predictor. The control stage
  shares existing cache/trainer/publication APIs without nested training runs;
  Predict retains shared extraction for inference-only ablations.
- **Tasks** coordinate scientific preparation, trainer selection, observations
  and scoring. Contact declares development feature requests; Prepare
  materializes/caches them before Tune invokes task fitting. Tasks never receive
  workspace objects or manage cache paths, distributed lifecycle or publication.
  Fixed CASP/predictor validation belongs to `ContactPredictionTask.validate_config`.
  Independent control preparation runs even in `evaluate_as_is`.
  The run manifest's `score` section includes primary and control metric summaries
  with CASP settings and paths to aggregate/unit Parquet files. Contact bootstrap
  is unavailable until complex-level resampling is implemented.
- **Results** use `PipelineResult.evaluations`: records carrying the workspace,
  prediction metadata, and optional score. The original layer-keyed
  `predictions` and `scores` properties remain compatibility views.

Cache and result files publish through unique same-directory temporary files
and atomic replacement. Independent contact-feature/baseline jobs coordinate
with atomic directory creation, not POSIX advisory locks or rank barriers.
Waiting times out after five minutes. A killed owner leaves `<lock>.d/owner`;
confirm that host/PID has terminated before manually removing that specific
lock directory. Workspace reservation locks are unchanged. These guarantees
depend on the filesystem's atomic mkdir/rename semantics; local concurrency
tests do not certify NFS/GPFS deployments, and publication is not power-loss
durability.

Contact backbone compatibility is isolated in `model/adapters.py`. Current
position-aware backbones use original `position_ids`; old ESM rotary and legacy
AMPLIFY checkpoints still require contained table adaptation. Tables are restored
after success or exceptions. Jacobian inference composes supported public encoders
and FP32 MLM heads without replacing `forward`.

#### Distributed execution (`accelerate`) patterns

Under `accelerate launch`, several processes (ranks) run the same code, usually one per GPU. These patterns recur across stages and `trainers/selection.py`, `trainers/torch.py`, and `trainers/selection_helpers.py`:

1. **Every rank must reach every collective call.** `accelerator.gather_for_metrics()`, `broadcast()`, and `broadcast_object_list()` are collective operations: every process must call them, in the same order, or the ranks that do call them hang forever waiting for the ones that don't. So a branch that looks like it should be gated on `if accelerator.is_main_process:` (e.g. "only build this control's model") often *can't* be, if anything inside it issues a collective call. `ControlBuilder.build`'s docstring shows a branch that is deliberately *not* wrapped in error handling for this reason.
2. **Decide once on rank 0, then broadcast the decision.** For a choice that must be identical on every rank (e.g. "does a cached file exist," "should this HPO trial be pruned"), only rank 0 evaluates the condition; the boolean is wrapped in a tensor and sent to every rank via `broadcast(..., from_process=0)` before anyone branches on it. Letting each rank decide independently can split them, because filesystem state, floating-point rounding, or the Optuna `trial` object (which exists only on rank 0) can differ.
3. **Freeze/unfreeze parameters *before* `accelerator.prepare()`, never after.** Changing `requires_grad` on a model already wrapped for DDP leaves its gradient-reduction buckets built around the wrong parameter set. `configure_trainable_params()` always runs before `accelerator.prepare(model)` in `TuneStep`.
4. **Classical probes fit identically on every rank.** Each rank holds the full gathered training embeddings and fits the estimator with the same seed, so no live sklearn object has to be broadcast; only rank 0 saves it.

Before adding a control condition, training-loop branch, or cross-rank decision,
inspect collectives in the existing stage and trainer implementation.

### Adding a Task Type

Every step (`prepare`/`tune`/`predict`/`score`) is task-agnostic: it looks up a
`TaskHandler` for `workspace.task_type` via `modules.evaluate.src.tasks.get_task`
and delegates all task-specific behavior to it. Adding a new task type means
adding one handler module, not editing the steps.

1. Create `modules/evaluate/src/tasks/<your_task>.py` with a class that
   subclasses `FlatMetricsTask` (or `StructuredTask` for structured scoring), sets
   `name` to the `task_type` string, and is
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

`contact_prediction` supplies the structured scoring contract: original-coordinate
units, fixed CASP protocols and hierarchical aggregation. Dataset preparation
and candidate pairs live in `dataset/contact.py`; fitted and zero-shot predictors
share the observation/scoring semantics. See
[`contact_prediction.py`](src/tasks/contact_prediction.py) and
[`contact_model.py`](src/model/contact_model.py).

The `categorical_jacobian` predictor reuses the contact label and metric
contract but replaces a fitted head with the frozen MLM; see
[Categorical-Jacobian contact prediction](#categorical-jacobian-contact-prediction).

Declare supported heads, controls, modes and data behavior on the task. Reuse
existing services where possible. A genuinely new execution strategy may require
explicit stage selection; do not introduce task lifecycle hooks or callbacks
into stages from trainers.

#### Minimal example

The smallest possible handler, a sequence-level classification task scored
only by accuracy, is roughly:

```python
# modules/evaluate/src/tasks/my_task.py
from transformers import AutoModelForSequenceClassification

from modules.evaluate.src.metrics import metrics
from modules.evaluate.src.tasks.base import FlatMetricsTask
from modules.evaluate.src.tasks.registry import register_task


@register_task
class MyTask(FlatMetricsTask):
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
| `head_type='...' requires steps.prepare.embedding_cache.enabled=True` | Classical probes fit on cached pooled embeddings; add `steps.prepare.embedding_cache.enabled=true` or use a `configs/predictors/` overlay |
| Multi-GPU run hangs | A rank skipped a collective call; see [Distributed execution patterns](#distributed-execution-accelerate-patterns) |

---

## Testing

```bash
uv run pytest -n auto tests/evaluate
```

`tests/evaluate/test_tasks.py` runs a shared contract suite over every registered task.
