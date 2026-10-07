# `modules/evaluate` — Evaluation Module

This module evaluates a pretrained protein language model on a downstream task. The pipeline supports evaluating a task head as-is, training a lightweight probe on a frozen trunk, or fine-tuning the trunk and native MLP head. Results are scored against held-out data with optional bootstrap confidence intervals and comparisons against randomized, scrambled, and baseline control conditions. The four-step pipeline is configured through YAML files (validated by Pydantic) with optional CLI dotlist overrides via OmegaConf.

```
Prepare → Tune → Predict → Score
```

The top-level `steps.mode` selects the execution path: `tune_probe` trains a probe while keeping the trunk frozen, `finetune` trains the trunk and native MLP head, and `evaluate_as_is` skips tuning and evaluates the task model loaded from the checkpoint or Hub.

The built-in task handlers are `sequence_classification`,
`sequence_multilabel_classification`, `sequence_regression`,
`token_classification`, `contact_prediction`, `categorical_jacobian`, and
`pseudo_perplexity`. Sequence-level classification and regression use standard
Transformers sequence heads; token classification uses a token head, and
multilabel classification uses independent binary outputs per sequence.
`contact_prediction` uses a custom symmetric low-rank pairwise head and stores
ragged per-sequence pairs so that separation-binned precision-at-L metrics can
be computed without materializing dense pair features. `categorical_jacobian`
scores residue pairs from the categorical Jacobian of a frozen masked-language
model trunk. `categorical_jacobian` and `pseudo_perplexity` are zero-shot tasks
and therefore run only in `evaluate_as_is` mode.

## Table of Contents

- [Quick Start — Smoke Test](#quick-start--smoke-test)
- [Supported Tasks](#supported-tasks)
- [Supported Datasets](#supported-datasets)
- [ProteinGym Evaluation](#proteingym-evaluation)
- [Directory Layout](#directory-layout)
- [Pipeline Architecture \& Distributed Execution](#pipeline-architecture--distributed-execution)
- [Workspace Disk Layout](#workspace-disk-layout)
- [Dataset Sourcing](src/dataset_sourcing/README.md)
- [Step 1 — Prepare](#step-1--prepare)
- [Step 2 — Tune](#step-2--tune)
- [Step 3 — Predict](#step-3--predict)
- [Step 4 — Score](#step-4--score)
- [Running the Pipeline](#running-the-pipeline)
- [Config System](#config-system)

---

## Quick Start — Smoke Test

Before running a full evaluation, validate the bounded `evaluate_as_is` path with the bundled [`configs/smoke_test.yaml`](configs/smoke_test.yaml). It runs Prepare → Predict → Score with a small batch size and a short bootstrap; it skips Tune.

```bash
uv run accelerate launch modules/evaluate/src/run_evaluate.py \
    modules/evaluate/configs/smoke_test.yaml \
    modules/evaluate/configs/models/amplify_120m.yaml \
    modules/evaluate/configs/tasks/sequence_classification.yaml \
    modules/evaluate/configs/datasets/localization_prediction.yaml
```

By default the output is written under `./eval/<model>/<dataset>/`. To send it elsewhere, override `workspace.base_path`:

```bash
uv run accelerate launch modules/evaluate/src/run_evaluate.py \
    modules/evaluate/configs/smoke_test.yaml \
    modules/evaluate/configs/models/amplify_120m.yaml \
    modules/evaluate/configs/tasks/sequence_classification.yaml \
    modules/evaluate/configs/datasets/localization_prediction.yaml \
    workspace.base_path=/tmp/eval_test
```

If the run completes and prints `=== Evaluation Pipeline Completed Successfully ===`, your environment and the Prepare, Predict, and Score steps are working. Use a tuning mode to validate training as well.

### Layerwise linear probes

For independent probes at every hidden-state depth, enable the online
layerwise mode. This uses the same Prepare, Tune, Predict, and Score pipeline,
including configured controls and bootstrap comparisons:

```bash
uv run accelerate launch modules/evaluate/src/run_evaluate.py \
    modules/evaluate/configs/config.yaml \
    modules/evaluate/configs/models/amplify_120m.yaml \
    modules/evaluate/configs/tasks/sequence_classification.yaml \
    modules/evaluate/configs/datasets/localization_prediction.yaml \
    steps.layerwise.enabled=true \
    steps.layerwise.layers=all \
    steps.tune.head.head_type=torch_linear \
    steps.prepare.embedding_cache.enabled=false
```

`steps.layerwise.layers` accepts `all` or a list such as `[0,6,-1]` (quote the
whole override in a shell). Index 0 is the token embedding output; with mean
pooling it is an amino-acid composition baseline, so `scrambled_sequences`
matches the trained probe there by construction. Index -1 is the final block's
output after the trunk's final norm; intermediate indices are the
un-normalized residual stream. Use `steps.layerwise.pooling=cls` for CLS-pooled sequence
probes when the tokenizer has a CLS/BOS token; token classification instead
uses unpooled, label-aligned residues. Token probes are not L2-normalized, so
per-residue norm differences between index -1 (after the final norm) and the
residual stream remain after standardization. Sequence classification, multilabel
classification, scalar regression, and token classification work with padded or compatible AMPLIFY
packed batches (`packed=true` requires CUDA; variable-token-budget batches
remain single-process). HPO and variance-seed sweeps are not supported in
this mode; `steps.tune` supplies the optimizer and epoch settings.

To keep depths comparable, every head sees features standardized with
train-split statistics (disable with `steps.layerwise.standardize_features=false`),
and regression heads predict standardized targets.
All heads share `steps.tune.learning_rate`; each layer keeps its own best
validation epoch, which is logged and saved in each `probe_head.pt`. If every
layer selects the last epoch, raise `steps.tune.max_epochs`.

`untrained` and `random_head` are identical with a frozen trunk and share one
run. `random_trunk` trains probes on a randomly initialized trunk with the
same procedure, which separates what pretraining adds from what the
architecture provides; `random_both` applies untrained heads to that trunk.
Random-trunk controls always run the trunk on tokens, even with the cache.
With `steps.score.bootstrap_enabled=true`, `layerwise_comparison.parquet`
compares every layer with the deepest probed layer using paired bootstrap
difference intervals and two-sided permutation p-values, Benjamini-Hochberg
adjusted across layers. Rows sharing a source ID (cached token residues) are
resampled and permuted together, so residues of one protein are not treated as
independent. Its cost grows with the number of layers.

With the cache disabled, the frozen trunk runs once per batch **per epoch**
for all requested heads together. Activations are discarded after that batch:
memory scales with batch size, sequence length, and requested depth, not
dataset size. Enabling the embedding cache is recommended for sequence tasks:
it extracts the selected layers once per split and reuses them across epochs
and runs:

```bash
steps.layerwise.enabled=true \
steps.layerwise.layers=all \
steps.tune.head.head_type=torch_linear \
steps.prepare.embedding_cache.enabled=true
```

Add these overrides to the launch command above. Layerwise settings determine
which layers are extracted, their pooling, and trunk autocast precision;
`steps.prepare.embedding_cache.dtype` sets the stored embedding precision,
and `cache_policy` controls disk reuse. Sequence-level embeddings are disk
cached when eligible; token-level embeddings are kept in memory for the run
(about tokens x hidden size x layers x bytes per value; the estimate is logged
before extraction).
Cached features train the independent heads without repeating trunk passes.
Controls that alter the trunk or input still require token-based passes.
Checkpoints, predictions,
manifests, and scores are isolated by `layer_<index>` run variants; the
shared `scores_long.parquet` also records those variants for plotting.

Layerwise and cached sequence probes default to L2-normalizing token
representations before residue-only mean pooling, limiting sensitivity to raw
activation magnitude. Boundary and padding tokens are excluded; unknown
residues are retained. Use `steps.layerwise.normalize_before_pooling=false`
to retain raw representation magnitudes. Frozen feature extraction defaults
to float32; reduced precision is opt-in. Online and cached pooling use the
same settings.

### Contact-prediction smoke test

A contact-prediction smoke run uses the same pipeline with a bounded
separation-prior control and five paired permutations, which keeps the
pairwise statistical comparison practical on a local machine:

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

## Supported Tasks

| Task type | Labels | Default metrics |
|---|---|---|
| `sequence_classification` | One class per sequence | F1, accuracy, MCC, ROC-AUC |
| `sequence_multilabel_classification` | Fixed-width binary indicator vector per sequence | F1, average precision |
| `sequence_regression` | One continuous value per sequence | MSE, MAE, R2, Pearson, Spearman |
| `token_classification` | One class per valid residue/token | F1, accuracy, precision, recall, MCC |
| `contact_prediction` | Sparse positive residue pairs; implicit negatives at valid separation | P@L, P@L/2, P@L/5 plus separation-binned contact AUC metrics |
| `categorical_jacobian` | Jacobian-derived residue-pair contact scores; sparse positive labels remain the evaluation target | P@L, P@L/2, P@L/5 plus separation-binned contact AUC metrics |
| `pseudo_perplexity` | None required; each residue's own token id is the per-mask target derived at prepare time | Pseudo-perplexity, mean log-probability |

## Supported Datasets

The dataset catalog lists each shipped overlay, what its labels represent, and
which task uses its target contract: [Evaluation Dataset Catalog](configs/datasets/README.md).

## ProteinGym Evaluation

ProteinGym uses per-assay variant tables rather than the standard
`sequence`/`targets` dataset contract. Track archives, source configuration,
and generated artifact layouts are documented in the
[ProteinGym dataset-sourcing guide](src/dataset_sourcing/README.md#proteingym-dataset-sourcing).
Four tracks are supported: DMS or clinical, substitutions or indels. The
separate runners support zero-shot scoring and supervised frozen-embedding
evaluation.

Zero-shot scoring uses masked-marginal scoring for substitutions and
pseudo-likelihood scoring for indels; the runner selects the latter from the
sourced `is_indel` flag. Because it does not train, zero-shot scoring does not
require train/validation/test splits:

```bash
uv run modules/evaluate/src/run_proteingym_zero_shot.py \
    modules/evaluate/configs/proteingym_zero_shot.yaml \
    model_id=facebook/esm2_t12_35M_UR50D
```

The zero-shot runner can also score reference baselines with
`baselines: [random, blosum62, random_init_trunk]`. `blosum62` is a simple
substitution-only matrix baseline, not ProteinGym's official MSA-derived
Site-Independent scorer. Baseline outputs are written under
`output_dir/baselines/<name>/`.

Supervised evaluation fits a probe per assay using pooled embeddings. Choose
`probe_type=linear|mlp|torch_linear`; `probe_search_space` performs a grid
search on an inner holdout of each training fold before refitting the winner on
all available training rows. Splits can use an official fold column
(`fold_random_5`, `fold_modulo_5`, or `fold_contiguous_5`),
`split=official_average` to average applicable official schemes equally,
`split=seeded` to use the sourced train/validation/test split, or
`split=custom_kfold` for a reproducible generated k-fold split:

```bash
uv run modules/evaluate/src/run_proteingym_supervised.py \
    modules/evaluate/configs/proteingym_supervised.yaml \
    model_id=facebook/esm2_t12_35M_UR50D
```

Both runners write `<dms_id>_scored.parquet` (each variant's `model_score`),
`assay_scores.parquet` (per-assay Spearman for continuous DMS scores or AUC for
binary clinical labels), and `summary.json` under `output_dir`. The summary
aggregates by taking the mean over UniProt IDs, then over function categories.
Set `compute_uncertainty: true` to add bootstrap standard errors;
`n_bootstrap` controls the resamples for either runner and defaults to `10000`.

---

## Directory Layout

```
modules/evaluate/
├── README.md                        ← this file
│
├── configs/                         ← yaml configs (base + composable overlays)
│   ├── config.yaml                  ← base/default config with all fields documented
│   ├── smoke_test.yaml              ← fast config for smoke-testing
│   ├── proteingym_supervised.yaml   ← ProteinGym per-assay probe scoring
│   ├── proteingym_zero_shot.yaml    ← ProteinGym zero-shot scoring
│   ├── accelerate/                  ← accelerate launch configs (e.g. single-process local runs)
│   ├── dataset_sourcing/
│   ├── datasets/                    ← dataset overlays and catalog (`README.md`)
│   ├── tasks/                       ← task behavior and defaults
│   │   ├── categorical_jacobian.yaml
│   │   ├── contact_prediction.yaml
│   │   ├── pseudo_perplexity.yaml
│   │   ├── sequence_classification.yaml
│   │   ├── sequence_multilabel_classification.yaml
│   │   ├── sequence_regression.yaml
│   │   └── token_classification.yaml
│   ├── models/                      ← one overlay per evaluated model (name, repo)
│   │   ├── amplify_120m.yaml
│   │   ├── amplify_350m.yaml
│   │   ├── esm2_35m.yaml
│   │   └── esm_150m.yaml
│   └── tuning/                      ← one hyperparameter-search overlay per steps.tune.head.head_type
│       ├── knn.yaml
│       ├── sklearn_linear.yaml
│       ├── random_forest.yaml
│       ├── xgboost.yaml
│       ├── mlp.yaml
│       ├── mlp_finetuning.yaml
│       └── torch_linear.yaml
│
├── jobs/
├── scripts/                          ← one-off utilities (not part of the pipeline)
│   └── dataset_sourcing/
│
└── src/
    ├── config.py                    ← EvaluationConfig: top-level Pydantic schema that
    │                                  composes StepsConfig + EvaluationWorkspaceConfig + WandbConfig
    ├── run_evaluate.py               ← CLI entry point: loads config and runs the four steps in order
    ├── run_proteingym_supervised.py  ← ProteinGym per-assay probe scoring entry point
    ├── run_proteingym_zero_shot.py   ← ProteinGym zero-shot scoring entry point
    │
    ├── dataset/
    │   ├── workspace.py              ← EvaluationWorkspaceConfig + EvaluationWorkspace (path layout)
    │   ├── dataloader.py             ← dataset loading, tokenization, DataLoader construction
    │   └── collator.py               ← EvaluationCollator (padding/attention-mask handling)
    │
    ├── dataset_sourcing/
    ├── proteingym/                   ← ProteinGym sourcing, baselines, and scoring support
    │
    ├── schemas/
    │   └── artifacts.py              ← shared dataclasses passed between steps

    ├── model/
    │   ├── contact_model.py          ← symmetric low-rank pairwise contact head
    │   ├── categorical_jacobian.py   ← zero-shot categorical-Jacobian contact model
    │   └── heads.py                  ← torch and sklearn-compatible probe heads
    │
    ├── steps/                        ← one file per pipeline step
    │   ├── prepare.py                ← Step 1: load tokenizer/dataset/model, build dataloaders
    │   ├── tune.py                   ← Step 2: fine-tune trunk/head (+ optional HPO)
    │   ├── predict.py                ← Step 3: run inference for the trained model + controls
    │   └── score.py                  ← Step 4: compute metrics, bootstrap CIs, control comparisons
    │
    ├── tasks/                        ← one TaskHandler per task_type; see "Adding a task type" below
    │   ├── base.py                   ← TaskHandler ABC: model, label, metric, and control hooks
    │   ├── contact_prediction.py     ← ragged pairwise contact labels and metrics
    │   ├── categorical_jacobian.py   ← zero-shot Jacobian contact task handler
    │   ├── registry.py                ← @register_task / get_task(task_type)
    │   ├── sequence_classification.py
    │   ├── sequence_multilabel_classification.py
    │   ├── sequence_regression.py
    │   ├── token_classification.py
    │   └── pseudo_perplexity.py
    │
    ├── metrics/
    │   ├── contact_metrics.py        ← separation-binned contact precision and AUC
    │   └── metrics.py                ← shared sklearn/scipy metric callables used by task handlers
    │
    └── utils/                        ← seeding, HPO helpers, I/O, W&B tracking, model-inference loop,
                                         frozen-trunk embedding cache, control-condition building
                                         (ControlBuilder: untrained/random-reinit/scrambled/baseline
                                         conditions PredictStep compares the trained model against)
```

Shared config loading helpers live in `modules/core/utils/config_loader.py`.

## Pipeline Architecture & Distributed Execution

<details>
<summary>Show pipeline handoffs and distributed-execution details</summary>

The four steps are separate Python objects (`PrepareStep`/`TuneStep`/`PredictStep`/`ScoreStep`) chained by `run_evaluate.py`. Two typed handoffs carry state between them:

- **In-process, in-memory:** `PreparedArtifacts` (`schemas/artifacts.py`) is what `PrepareStep` returns and `TuneStep`/`PredictStep` consume directly within the same process/run. Its surface is deliberately small and uniform: `model`, `tokenizer`, `train_dataloader`, `val_dataloader`, `test_dataloader`. `TuneStep` and `ScoreStep` never need anything beyond this.
  - The one exception is `embedding_backend` (`utils/embed.py::EmbeddingBackend`), populated only when `steps.prepare.embedding_cache.enabled=True`. Rather than expose the cache's internals (the frozen trunk, the original token dataloader, pooling/layers/dtype/hidden size) as separate optional fields callers have to know about, they're bundled behind this one object. Only `ControlBuilder` (`utils/controls.py`) reaches into it, because a handful of control conditions (`random_trunk`, `scrambled_sequences`) must re-embed the test split under a perturbed trunk or scrambled input — something a pooled-`inputs_embeds`-only `test_dataloader` can no longer support on its own. `PredictStep`'s own logic only asks *whether* it's populated (to decide if sequences need separate decoding); it never touches the trunk/pooling details directly.
- **Cross-process, on-disk:** `ScoreStep` never receives an in-memory `PredictOutput`. It reconstructs everything it needs — predictions, control predictions, metadata — purely from `run_manifest.json` plus the parquet files it points to (see `ScoreStep._load_predict_output`). This is what lets Score run as a single, plain (rank-0-only) CPU process regardless of how many GPUs Prepare/Tune/Predict used, without needing to know anything about `Accelerator` state.

### Distributed execution (`accelerate`) patterns

Most steps run under `accelerate launch` (potentially many processes/GPUs), and the same handful of patterns recur throughout `tune.py`/`predict.py`/`utils/controls.py`/`utils/hpo.py`. They're documented once here instead of only as scattered inline comments:

1. **Every rank must reach every collective call.** `accelerator.gather_for_metrics()`, `broadcast()`, and `broadcast_object_list()` are collective operations: every process in the run must call them, in the same order, or the ranks that do call them hang forever waiting for ones that don't. This means control-flow that looks like it should be `if accelerator.is_main_process: ...`-gated (e.g. "only build this control's model," "only check if this file exists") often *can't* be, if anything inside that branch issues a collective call — see `ControlBuilder.build`'s docstring for a concrete example of a branch that's deliberately *not* wrapped in error handling for this reason.
2. **Decide once on rank 0, then broadcast the decision.** For a choice that must be identical across every rank (e.g. "does a cached file exist on disk," "has this HPO trial's validation metric stopped improving," "should this trial be pruned"), only rank 0 evaluates the real condition (e.g. by touching the filesystem, or holding the only real Optuna `trial` object); the boolean result is wrapped in a tensor and sent to every rank via `broadcast(..., from_process=0)` before any rank branches on it. Never let each rank evaluate the condition independently — filesystem state, floating-point rounding, or an Optuna trial only existing on rank 0 can make ranks disagree.
3. **Freeze/unfreeze parameters *before* `accelerator.prepare()`, never after.** Changing `requires_grad` on a model already wrapped for DDP leaves its gradient-reduction buckets built around the wrong parameter set. `configure_trainable_params()` always runs before `accelerator.prepare(model)` in `TuneStep`.
4. **CPU-only, deterministic scoring runs on rank 0 only.** Classical probe fitting is performed independently on each rank after the identical training embeddings have been gathered; this avoids broadcasting live sklearn estimators while keeping every rank's model state consistent for subsequent prediction.

If you're adding a new control condition, training loop branch, or cross-rank decision, search for `broadcast(` and `is_main_process` in `utils/controls.py`/`steps/tune.py` first — the pattern you need almost certainly already exists there.

</details>

## Adding a task type

<details>
<summary>Show task-authoring guide</summary>

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
   least one metric/`default_metric_names` entry are required -- a handler
   missing one of these fails immediately at import time, not at first use.
   Tasks that expand one source example into several tokenized rows (e.g.
   one row per masked residue) override `expand_tokenized_example`; tasks
   whose dataset overlay has no real label column set
   `requires_source_labels = False` so `tokenize_dataset` doesn't require one.
2. Import the new module from `modules/evaluate/src/tasks/__init__.py` so it
   self-registers.
3. Add a task config under `modules/evaluate/configs/tasks/`.
   Declare accepted dataset target types and label formats on the handler.
   If your task's head size is derived directly from `workspace.num_labels`
   (i.e. a classification-style head), also override
   `validate_num_labels()` to check the resolved `num_labels` against label
   values actually present in the dataset (see `sequence_classification.py`/
   `token_classification.py`) -- this catches a `num_labels` count that
   doesn't match the data before a wrong-sized head is built, rather than
   surfacing as a cryptic index-out-of-bounds error during training.
4. Add focused tests for task-specific behavior. `tests/evaluate/test_tasks.py`
    is parametrized over every registered task, so the new handler is also
    exercised by the shared contract suite automatically.

`contact_prediction` is the non-standard example: it overrides the model
builder, pairwise label collation, ragged prediction-row handling, and contact
metrics. Its labels are sparse positive `[i, j]` residue pairs, with implicit
negative labels for valid pairs at separation six or greater. See
[`contact_prediction.py`](src/tasks/contact_prediction.py) and
[`contact_model.py`](src/model/contact_model.py).

`categorical_jacobian` reuses the contact-prediction label and metric contract,
but replaces the trainable pairwise head with a frozen masked-language-model
trunk. It is a zero-shot rank scorer: no probe is tuned, no model parameters
are updated, and the returned pair scores are not calibrated probabilities or
binary-class logits.

For each real residue position, the model evaluates all 20 canonical amino-acid
substitutions and records the change in the MLM's raw output logits for the 20
canonical output channels relative to the wild-type sequence. The resulting
`(L, 20, L, 20)` tensor is centered across all four categorical-Jacobian axes,
directionally symmetrized before the nonlinear Frobenius reduction, reduced to
an `(L, L)` map, and Average Product Corrected with the diagonal excluded from
the APC statistics. Leading/trailing special tokens are excluded from the
residue coordinates and validated against the tokenizer layout.

The implementation processes one sequence at a time and batches its `20 * L`
mutant forward inputs with `workspace.model_kwargs.mutant_batch_size`. The
resident Jacobian is allocated in float32 when possible, falls back to float16
when needed, and raises before allocation if it still exceeds
`workspace.model_kwargs.max_jacobian_bytes`. Reduce `steps.prepare.max_length`
or `mutant_batch_size` when running on limited hardware; the latter affects
forward-pass memory, while the Jacobian cap controls the resident sensitivity
tensor.

Use [`categorical_jacobian.yaml`](configs/tasks/categorical_jacobian.yaml) with
[`contact_prediction_binary.yaml`](configs/datasets/contact_prediction_binary.yaml).
Tuning and head-related controls are not applicable. The model returns no
training loss because APC/Frobenius scores are not valid inputs to binary
cross-entropy.

No changes to `prepare.py`/`collator.py`/`dataloader.py`/`predict.py`/`score.py`
are needed unless the task requires something genuinely new (e.g. a non-HF
model class, or pairwise/2D labels) -- in which case override the relevant
`TaskHandler` hook rather than adding a branch in the step.

### Minimal example

The smallest possible handler -- a sequence-level classification task using
only accuracy -- is roughly:

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

</details>


---

## Workspace Disk Layout

`get_evaluation_workspace()` (in [`src/dataset/workspace.py`](src/dataset/workspace.py)) resolves a workspace directory keyed by model + dataset, so results for different (model, dataset) pairs never collide:

```
<base_path>/
└── <model_repo_or_name>/
    └── <dataset_repo_or_name>/
        ├── tmp/
        │   └── embeddings/            ← reusable frozen-trunk embedding cache
        ├── scores/
        │   └── scores_long.parquet    ← shared comparison table across variants
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

---

## Step 1 — Prepare

**Purpose:** Load the tokenizer, model, and dataset; validate task labels; and
build the train, validation, and test dataloaders. The default dataset columns
are `sequence`, `targets`, and `id`; if the configured ID column is absent,
Prepare uses positional identifiers.

**Key config fields (`steps.prepare`):**

| Field | Schema default | Description |
|---|---|---|
| `batch_size` | `16` | Examples per dataloader batch, unless a packed token budget is set |
| `max_length` | `512` | Maximum tokenized sequence length |
| `split.name` | `default` | Select `train_<name>`, `validation_<name>`, and `test_<name>` for a named split method |
| `validation_fraction` | `0.1` | Share of train held out when validation is missing or explicitly unset |
| `length_grouped_sampling` | `false` | Group similarly sized sequences to reduce padding |
| `embedding_cache.enabled` | `false` | Cache frozen-trunk embeddings for supported probe modes |
| `packed` | `false` | Pack multiple sequences into one forward pass; requires CUDA and a compatible AMPLIFY checkpoint |
| `max_tokens_per_batch` | `null` | Optional packed token budget; single-process only and supersedes `batch_size` |

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

**Purpose:** Train the configured model path on the `train` split. `tune_probe` freezes the pretrained trunk and trains a probe head, `finetune` trains the trunk and native MLP head, and `evaluate_as_is` skips this step entirely. Probe tuning can optionally use cached pooled embeddings and can perform hyperparameter search (Optuna TPE with median pruning, random, or grid) over `hyperparameter_search.search_space`.

**Key config fields (`steps.tune`):**

| Field | Default | Description |
|---|---|---|
| `mode` | `tune_probe` | `tune_probe` trains a frozen-trunk probe, `finetune` trains the trunk plus native MLP head, or `evaluate_as_is` skips tuning |
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
| `hyperparameter_search.search_pattern` | `bohb` | One of `bohb` (Bayesian optimization + Hyperband), `random`, `grid` |
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

**Input:** `PreparedArtifacts` from the Prepare step (or a resumed checkpoint)
**Output:** in-memory updated `PreparedArtifacts.model`; `<workspace>/runs/<variant>/model/` on disk when `save_best_model=true`

---

## Step 3 — Predict

**Purpose:** Run inference over the `test` split for the trained model and, for any configured control condition not already cached for this (model, dataset) pair, run inference for those too. No metrics are computed here — that's [Score](#step-4--score)'s job.

**Control conditions (`controls`):** each isolates a different possible explanation for the trained model's performance, so [Score](#step-4--score) can compare the trained model against it.

| Control | What it measures |
|---|---|
| `untrained` | The pretrained trunk with a freshly-initialized head, i.e. the model as it was before fine-tuning — how much fine-tuning improved on the base model |
| `random_trunk` | Trained model with its trunk randomly re-initialized (head kept) — how much the fine-tuned trunk (vs. the head alone) drives performance |
| `random_head` | Trained model with its head randomly re-initialized (trunk kept) — how much the fine-tuned head (vs. the trunk alone) drives performance |
| `random_both` | Trained model with both trunk and head randomly re-initialized — a fully-untrained baseline for comparison |
| `scrambled_labels` | No inference needed — derived directly from the trained model's own predictions/labels with labels shuffled, to test whether metrics exceed chance-level label alignment |
| `scrambled_sequences` | Trained model's predictions on inputs whose token order has been shuffled — tests reliance on sequence order/content vs. residue composition alone |
| `majority_class` | No model inference — predicts the most frequent class in the train split and exposes the training class prior as probabilities; a baseline for single-label classification tasks |
| `train_mean` | No model inference — predicts the train split's label mean for every example; the regression counterpart to `majority_class`, for `sequence_regression` tasks |
| `separation_prior` | No model inference — for `contact_prediction` and `categorical_jacobian`, predicts the empirical contact rate for each residue-separation bin |

The task configs ship with these control presets; override them with
`steps.predict.controls` when needed:

| Task type | Controls in the shipped task config |
|---|---|
| `sequence_classification` | `untrained`, `random_trunk`, `random_head`, `random_both`, `scrambled_labels`, `scrambled_sequences`, `majority_class` |
| `sequence_multilabel_classification` | `untrained`, `random_trunk`, `random_head`, `random_both`, `scrambled_labels`, `scrambled_sequences` |
| `sequence_regression` | `untrained`, `random_trunk`, `random_head`, `random_both`, `scrambled_labels`, `scrambled_sequences`, `train_mean` |
| `token_classification` | `untrained`, `random_trunk`, `random_head`, `random_both`, `scrambled_labels` |
| `contact_prediction` | `untrained`, `scrambled_labels`, `separation_prior` |
| `categorical_jacobian` | `scrambled_labels`, `separation_prior` |
| `pseudo_perplexity` | `random_both` |

Compatibility notes:

- `scrambled_sequences` is rejected for ragged tasks such as `token_classification`, `contact_prediction`, and `categorical_jacobian`, because position-aligned labels are not shuffled with the input tokens.
- `majority_class` is not defined for `sequence_regression` or `sequence_multilabel_classification`; use `train_mean` for regression. `train_mean` is only valid for `sequence_regression`.
- `separation_prior` is only valid for `contact_prediction` and `categorical_jacobian`.
- Classical probe heads do not support `untrained`, `random_head`, or `random_both`, which need a re-initializable head.

**Key config fields (`steps.predict`):**

| Field | Default | Description |
|---|---|---|
| `collect_probabilities` | `false` | Collect task-appropriate per-class/per-label probabilities (softmax for single-label, sigmoid for multilabel); auto-enabled when an effective metric requires probabilities, such as `roc_auc` or `average_precision` |
| `predictions_filename` | `predictions.parquet` | Output filename under `<workspace>/runs/<variant>/preds/` |
| `persist_to_disk` | `true` | Write predictions and the prediction manifest; disable for fast in-process evaluation |
| `store_sequences` | `false` | Decode and include source sequences in the main predictions table |
| `controls` | `[untrained]` | Subset of `untrained`, `random_trunk`, `random_head`, `random_both`, `scrambled_labels`, `scrambled_sequences`, `majority_class`, `train_mean`, `separation_prior` |

**Input:** `PreparedArtifacts` (trained model + test dataloader) from Prepare/Tune
**Output:** `<workspace>/runs/<variant>/preds/predictions.parquet`; settings are also recorded in `<workspace>/runs/<variant>/run_manifest.json`'s "predict" section

---

## Step 4 — Score

**Purpose:** Compute task-specific metrics from the Predict step's output and, optionally, bootstrap confidence intervals for those metrics plus a statistical comparison between the trained model and each configured control condition, to quantify how much of its performance is attributable to learning from the data rather than model/data artifacts.

**Key config fields (`steps.score`):**

| Field | Default | Description |
|---|---|---|
| `metrics` | `null` | List of metrics to compute (task-specific defaults apply when unset), e.g. `accuracy`, `f1`, `precision`, `recall`, `average_precision`, `roc_auc`, `mcc`, `mse`, `r2`, `pearsonr`, `spearmanr`, `contact_precision_at_l_long`, `contact_auc_long`, `pseudo_perplexity`, `mean_log_prob` |
| `metric_aggregation` | `weighted` | One of `micro`, `macro`, `weighted`, `samples` |
| `bootstrap_enabled` | `false` | Enable bootstrap confidence intervals + control comparisons |
| `n_samples` | `1000` | Bootstrap resamples; the code warns when below 1000 because confidence-interval tails are poorly estimated |
| `n_permutations` | `1000` | Trained-vs-control paired randomization-test permutations; larger values improve p-value resolution |
| `output_filename` | `bootstrap_results.parquet` | Per-resample bootstrap metric values (bootstrap mode only) |
| `confidence_interval` | `0.95` | Confidence level for bootstrap intervals |
| `long_filename` | `scores_long.parquet` | Tidy/long-format sidecar written on every run, with or without bootstrap — concatenable across (model, dataset) runs for plotting |

**Input:** `<workspace>/runs/<variant>/preds/predictions.parquet`
**Output:** `<workspace>/scores/scores_long.parquet` (always, shared across run variants), plus `<workspace>/runs/<variant>/scores/bootstrap_results.parquet` when `bootstrap_enabled=true`; scores/confidence intervals are also recorded in `<workspace>/runs/<variant>/run_manifest.json` under the `score`/`bootstrap` sections

> **Note:** `n_samples` controls bootstrap interval resolution; fewer than about
> 1000 resamples makes the interval tails unstable. `n_permutations` controls
> paired-test p-value resolution: the smallest unadjusted nonzero value is
> `1 / (n_permutations + 1)`, before multiple-comparison correction.

---

## Running the Pipeline

The entry point is `modules/evaluate/src/run_evaluate.py`. It accepts one or more YAML config files followed by optional `key=value` dotlist overrides. When multiple YAML files are given, they are merged left-to-right (later files win on shared keys) — this is how base, model, task, and dataset overlays are composed together.

### Base config + model + task + dataset overlays

```bash
uv run accelerate launch modules/evaluate/src/run_evaluate.py \
    modules/evaluate/configs/config.yaml \
    modules/evaluate/configs/models/amplify_120m.yaml \
    modules/evaluate/configs/tasks/sequence_classification.yaml \
    modules/evaluate/configs/datasets/localization_prediction.yaml
```

There is currently no multi-(model, dataset, head_type) sweep driver: each invocation above
handles exactly one combination, and a Slurm array or shell loop over combinations is left to
the caller (see `modules/data/mila_cluster_jobs/` for that pattern in a different module).

### Config + CLI overrides

```bash
uv run accelerate launch modules/evaluate/src/run_evaluate.py \
    modules/evaluate/configs/config.yaml \
    modules/evaluate/configs/models/amplify_120m.yaml \
    modules/evaluate/configs/tasks/token_classification.yaml \
    modules/evaluate/configs/datasets/ssp_q3.yaml \
    steps.mode=finetune \
    steps.score.bootstrap_enabled=true \
    steps.score.n_samples=1000
```

### Frozen-trunk probe overlays (`configs/tuning/`)

Each frozen-trunk probe overlay (`knn.yaml`, `sklearn_linear.yaml`, `random_forest.yaml`, `xgboost.yaml`, `mlp.yaml`, `torch_linear.yaml`) is self-contained, including its own `steps.mode=tune_probe` and embedding-cache settings, so it is just one more file after the model/dataset configs:

```bash
uv run accelerate launch modules/evaluate/src/run_evaluate.py \
    modules/evaluate/configs/smoke_test.yaml \
    modules/evaluate/configs/models/amplify_120m.yaml \
    modules/evaluate/configs/tasks/sequence_classification.yaml \
    modules/evaluate/configs/datasets/metal_ion_binding.yaml \
    modules/evaluate/configs/tuning/random_forest.yaml
```

`mlp_finetuning.yaml` (unfrozen-trunk full fine-tuning) is the one exception — it doesn't use the embedding cache.

### Evaluating a model as-is

Set `steps.mode=evaluate_as_is` to skip tuning and run Prepare → Predict → Score
with the task model loaded from the configured model checkpoint or Hub repo:

```bash
uv run accelerate launch modules/evaluate/src/run_evaluate.py \
    modules/evaluate/configs/config.yaml \
    modules/evaluate/configs/models/amplify_120m.yaml \
    modules/evaluate/configs/tasks/sequence_classification.yaml \
    modules/evaluate/configs/datasets/localization_prediction.yaml \
    steps.mode=evaluate_as_is \
    steps.score.bootstrap_enabled=true
```

### Categorical-Jacobian contact prediction

The categorical-Jacobian overlay evaluates a pretrained masked-language model
without tuning a probe or fine-tuning the trunk. This bounded example keeps
the sequence length and mutation batch small for local validation:

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

For a full run, use the same model and dataset overlays without the bounded
overrides:

```bash
uv run accelerate launch modules/evaluate/src/run_evaluate.py \
    modules/evaluate/configs/config.yaml \
    modules/evaluate/configs/models/amplify_120m.yaml \
    modules/evaluate/configs/tasks/categorical_jacobian.yaml \
    modules/evaluate/configs/datasets/contact_prediction_binary.yaml
```

This task requires `steps.mode: evaluate_as_is`. `separation_prior` and
`scrambled_labels` are supported controls; controls can be enabled or disabled
through `steps.predict.controls`.

### New dataset, task, or model overlay

Copy an existing file under `configs/datasets/`, `configs/tasks/`, or
`configs/models/`. Dataset overlays set the source, target type, label format,
and class count. Task overlays set `task_type` and task behavior. Task-specific
builder parameters belong in `workspace.model_kwargs`.

---

## Config System

All config classes are Pydantic `BaseModel` subclasses composed into a single `EvaluationConfig` (`src/config.py`):

```
EvaluationConfig
├── workspace: EvaluationWorkspaceConfig
├── wandb: WandbConfig
└── steps: StepsConfig
    ├── prepare: PrepareConfig
    ├── tune:    TuneConfig
    │   └── hyperparameter_search: HyperparameterSearchConfig
    ├── predict: PredictConfig
    └── score:   ScoreConfig
```

The defaults in the tables below are the Pydantic schema defaults. The checked-in
[`configs/config.yaml`](configs/config.yaml) is a runnable project baseline and
intentionally overrides some of them, including sampling, tuning epochs and
batch size, HPO trial count, metric aggregation, and whether bootstrap scoring
is enabled.

---
