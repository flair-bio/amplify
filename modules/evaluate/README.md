# `modules/evaluate` — Evaluation Module

This module evaluates a pretrained protein language model on a downstream task. The pipeline supports evaluating a task head as-is, training a lightweight probe on a frozen trunk, or fine-tuning the trunk and native MLP head. Results are scored against held-out data with optional bootstrap confidence intervals and comparisons against randomized, scrambled, and baseline control conditions. The four-step pipeline is configured through YAML files (validated by Pydantic) with optional CLI dotlist overrides via OmegaConf.

```
Prepare → Tune → Predict → Score
```

The top-level `steps.mode` selects the execution path: `tune_probe` trains a probe while keeping the trunk frozen, `finetune` trains the trunk and native MLP head, and `evaluate_as_is` skips tuning and evaluates the task model loaded from the checkpoint or Hub.

The built-in task handlers are `sequence_classification`, `sequence_regression`,
`token_classification`, `contact_prediction`, `categorical_jacobian`, and
`pseudo_perplexity`. The first three use the standard Transformers
sequence/token model heads. `contact_prediction` uses a custom symmetric
low-rank pairwise head and stores ragged per-sequence pairs so that
separation-binned precision-at-L metrics can be computed without
materializing dense pair features. `categorical_jacobian` and
`pseudo_perplexity` are zero-shot tasks scored from a frozen
masked-language-model trunk and therefore run only in `evaluate_as_is` mode.

## Table of Contents

- [Quick Start — Smoke Test](#quick-start--smoke-test)
- [Supported Tasks](#supported-tasks)
- [Directory Layout](#directory-layout)
- [Pipeline Architecture \& Distributed Execution](#pipeline-architecture--distributed-execution)
- [Workspace Disk Layout](#workspace-disk-layout)
- [Dataset Sourcing](#dataset-sourcing)
- [Step 1 — Prepare](#step-1--prepare)
- [Step 2 — Tune](#step-2--tune)
- [Step 3 — Predict](#step-3--predict)
- [Step 4 — Score](#step-4--score)
- [Running the Pipeline](#running-the-pipeline)
- [Config System](#config-system)

---

## Quick Start — Smoke Test

Before running a full evaluation, validate the pipeline end-to-end with the bundled [`configs/smoke_test.yaml`](configs/smoke_test.yaml), which runs every step (Prepare → Tune → Predict → Score) against a small batch size and a single epoch:

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

If the run completes and prints `=== Evaluation Pipeline Completed Successfully ===`, your environment and all four pipeline steps are working.

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
| `sequence_regression` | One continuous value per sequence | MSE, MAE, R2, Pearson, Spearman |
| `token_classification` | One class per valid residue/token | F1, accuracy, precision, recall, MCC |
| `contact_prediction` | Sparse positive residue pairs; implicit negatives at valid separation | P@L, P@L/2, P@L/5 plus separation-binned contact AUC metrics |
| `categorical_jacobian` | Jacobian-derived residue-pair contact scores; sparse positive labels remain the evaluation target | P@L, P@L/2, P@L/5 plus separation-binned contact AUC metrics |
| `pseudo_perplexity` | None required; each residue's own token id is the per-mask target derived at prepare time | Pseudo-perplexity, mean log-probability |

## Supported Datasets

The evaluation module currently includes overlays for the following datasets.
Most datasets are expected to expose `sequence` and `targets` columns with
`train`, `validation`, and `test` splits. `pseudo_perplexity` only needs
`sequence`; masked rows and target token ids are derived during preparation.
Dataset overlays declare the target type, label format, and Hub repository.
Task overlays select how compatible targets are evaluated.

| Dataset | Hub repository | Compatible tasks | Labels |
|---|---|---|---|
| `localization_prediction` | `swhitfield/biomap-research-localization_prediction` | Sequence classification, pseudo-perplexity | 10 classes per sequence |
| `metal_ion_binding` | `swhitfield/biomap-research-metal_ion_binding` | Sequence classification, pseudo-perplexity | 2 classes per sequence |
| `optimal_temperature` | `swhitfield/biomap-research-optimal_temperature` | Sequence regression, pseudo-perplexity | One continuous value per sequence |
| `ssp_q3` | `swhitfield/biomap-research-ssp_q3` | Token classification, pseudo-perplexity | 3 classes per residue/token |
| `contact_prediction_binary` | `swhitfield/biomap-research-contact_prediction_binary` | Contact prediction, categorical Jacobian, pseudo-perplexity | Sparse positive residue pairs with implicit negatives at valid separation |

The sourcing workflow also supports the TAPE ProteinNet contact-prediction
dataset. [`source_tape_proteinnet.py`](scripts/dataset_sourcing/source_tape_proteinnet.py)
converts its LMDB splits into the same `sequence`/`targets` contract, applying
the downstream sequence-separation filtering used by `contact_prediction`.
ProteinNet currently has a sourcing configuration but no dedicated overlay in
`configs/datasets/`; use the generated dataset repository or local artifacts
with a contact-prediction workspace configuration.

---

## Directory Layout

```
modules/evaluate/
├── README.md                        ← this file
│
├── configs/                         ← yaml configs (base + composable overlays)
│   ├── config.yaml                  ← base/default config with all fields documented
│   ├── smoke_test.yaml              ← fast config for smoke-testing
│   ├── accelerate/                  ← accelerate launch configs (e.g. single-process local runs)
│   ├── dataset_sourcing/            ← configs for the dataset-sourcing scripts
│   ├── datasets/                    ← dataset source and target metadata
│   │   ├── contact_prediction_binary.yaml
│   │   ├── localization_prediction.yaml
│   │   ├── metal_ion_binding.yaml
│   │   ├── optimal_temperature.yaml
│   │   └── ssp_q3.yaml
│   ├── tasks/                       ← task behavior and defaults
│   │   ├── categorical_jacobian.yaml
│   │   ├── contact_prediction.yaml
│   │   ├── pseudo_perplexity.yaml
│   │   ├── sequence_classification.yaml
│   │   ├── sequence_regression.yaml
│   │   └── token_classification.yaml
│   ├── models/                      ← one overlay per evaluated model (name, repo)
│   │   ├── amplify_120m.yaml
│   │   ├── amplify_350m.yaml
│       └── ...
│   │   └── esm2_35m.yaml
│   └── tuning/                      ← one hyperparameter-search overlay per steps.tune.head.head_type
│       ├── knn.yaml
│       ├── sklearn_linear.yaml
│       ├── random_forest.yaml
│       ├── xgboost.yaml
│       ├── mlp.yaml
│       ├── mlp_finetuning.yaml
│       └── torch_linear.yaml
│
├── scripts/                          ← one-off utilities (not part of the pipeline)
│   └── dataset_sourcing/
│       ├── source_biomap_datasets.py ← sources/preprocesses/uploads Biomap datasets to HF
│       ├── source_proteingym.py      ← sources ProteinGym datasets to HF
│       └── source_tape_proteinnet.py ← converts ProteinNet archive/LMDB data and uploads it
│
└── src/
    ├── config.py                    ← EvaluationConfig: top-level Pydantic schema that
    │                                  composes StepsConfig + EvaluationWorkspaceConfig + WandbConfig
    ├── run_evaluate.py               ← CLI entry point: loads config and runs the four steps in order
    │
    ├── dataset/
    │   ├── workspace.py              ← EvaluationWorkspaceConfig + EvaluationWorkspace (path layout)
    │   ├── dataloader.py             ← dataset loading, tokenization, DataLoader construction
    │   └── collator.py               ← EvaluationCollator (padding/attention-mask handling)
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
    │   ├── sequence_regression.py
    │   └── token_classification.py
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

## Adding a task type

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

`<variant>` is the run's evaluation condition: head type + trunk mode (or `pretrained` for `evaluate_as_is`), suffixed with `steps.prepare.split.name` when it isn't `default`. So `mlp-frozen_trunk` and `mlp-frozen_trunk-stratified` keep separate checkpoints, predictions, and manifests, while the shared `scores_long.parquet` carries a `split` column and is upserted per (variant, split).

---

## Dataset Sourcing

**Purpose:** before a dataset can be evaluated, it needs to exist in the shape the [Prepare](#step-1--prepare) step expects (a `sequence`/`targets` HF dataset with `train`/`validation`/`test` splits). `modules/evaluate/src/dataset_sourcing/` is a generic download → preprocess → upload pipeline for turning an upstream HF dataset repo into that shape; `modules/evaluate/scripts/dataset_sourcing/source_biomap_datasets.py` is the concrete one-off script that plugs in the Biomap-research source.

```
modules/evaluate/src/dataset_sourcing/
├── config.py      ← DatasetSourcingConfig: Pydantic schema for a sourcing run + upload repo-id resolution
├── spec.py        ← DatasetSourceSpec: per-source repo-id template + column renames
├── pipeline.py     ← resolve_datasets()/run_for_dataset()/run_cli(): orchestration + CLI entrypoint
├── hf_io.py        ← download_dataset()/upload_dataset_artifact(): HF Hub download/upload calls
├── preprocess.py   ← load_splits()/minimal_preprocess(): locate/load raw splits, then rename/filter/truncate
└── outputs.py      ← write_hf_split_files()/write_stats()/write_seqkit_stats()/write_dataset_card()
```

Each dataset goes through:

```
Download (HF Hub) → Preprocess (rename/split/truncate) → Stats + dataset card → Upload (HF Hub)
```

Raw downloads land under `<work_dir>/downloads/<dataset>/`, and processed artifacts (split parquet files, `stats.json`, seqkit stats, `README.md`) are written to `<work_dir>/outputs/<dataset>/`, which is also what gets uploaded.

### Adding a new source

Subclass `DatasetSourcingConfig` with the source's known dataset names and default `work_dir`/`repo_prefix`, build a `DatasetSourceSpec` with the upstream `repo_id_template` and any `column_rename`, then call `run_cli(sys.argv[1:], model_cls=..., spec=...)` — see [`scripts/dataset_sourcing/source_biomap_datasets.py`](scripts/dataset_sourcing/source_biomap_datasets.py) for a complete example.

### Example — sourcing a Biomap-research dataset

[`configs/dataset_sourcing/source_biomap_datasets.yaml`](configs/dataset_sourcing/source_biomap_datasets.yaml) selects one dataset (`optimal_temperature`), skips upload (no `repo_owner`/`repo_id` set), and only builds local artifacts under `work_dir`:

```bash
uv run python modules/evaluate/scripts/dataset_sourcing/source_biomap_datasets.py \
    modules/evaluate/configs/dataset_sourcing/source_biomap_datasets.yaml
```

To source every known Biomap-research dataset and upload each to its own repo under an owner, override `all_datasets` and `repo_owner` on the CLI:

```bash
uv run python modules/evaluate/scripts/dataset_sourcing/source_biomap_datasets.py \
    modules/evaluate/configs/dataset_sourcing/source_biomap_datasets.yaml \
    all_datasets=true \
    repo_owner=<your-hf-username-or-org>
```

This uploads each dataset to `<repo_owner>/biomap-research-<dataset_name>` (per `repo_name_prefix` in the config), one HF dataset repo per dataset (`upload_layout: per_dataset_repo`).

Biomap preprocessing preserves the downloaded splits and, when `create_split_subsets: true` is set in the sourcing YAML, publishes three additional deterministic split methods. If the download lacks validation data, validation is held out from training along MMseqs cluster boundaries when MMseqs is available, with seeded random sampling as a fallback.

| Method | Hugging Face splits | Behavior |
|---|---|---|
| Downloaded (default) | `train`, `validation`, `test` | Preserves source assignments, except synthesized validation when absent |
| Random | `train_random`, `validation_random`, `test_random` | Seeded uniform row partition |
| Stratified | `train_stratified`, `validation_stratified`, `test_stratified` | Balances categorical labels, quantile-binned continuous targets, or dominant token labels |
| Hold-cluster-out | `train_cluster`, `validation_cluster`, `test_cluster` | Assigns each complete MMseqs sequence cluster to exactly one subset; no within-cluster stratification |

The downloaded method is the default evaluation input. Pooled cluster generation requires MMseqs. Selecting another method only means setting `steps.prepare.split.name`, which resolves the three splits to `train_<name>`/`validation_<name>`/`test_<name>`:

```bash
uv run accelerate launch modules/evaluate/src/run_evaluate.py \
    modules/evaluate/configs/config.yaml \
    modules/evaluate/configs/models/amplify_120m.yaml \
    modules/evaluate/configs/tasks/sequence_classification.yaml \
    modules/evaluate/configs/datasets/metal_ion_binding.yaml \
    steps.prepare.split.name=stratified
```

If the dataset doesn't publish those splits, Prepare fails with the list of splits it did find. Datasets with benchmark-specific split names set the roles explicitly — see [Named benchmark splits](#named-benchmark-splits).

### Example — sourcing TAPE ProteinNet

[`scripts/dataset_sourcing/source_tape_proteinnet.py`](scripts/dataset_sourcing/source_tape_proteinnet.py)
converts the TAPE ProteinNet LMDB splits into the same
`sequence`/`targets` dataset contract used by the contact-prediction task.
The converter derives contacts from valid residue coordinates, writes split
artifacts and statistics, and can publish the result as a Hugging Face dataset.

The checked-in config reads pre-staged LMDB split directories under
`downloaded_datasets/`. Invoke the Python script with the config and overrides
directly; `uv` installs the `lmdb` dependency for this source:

```bash
uv run --with lmdb modules/evaluate/scripts/dataset_sourcing/source_tape_proteinnet.py \
    modules/evaluate/configs/dataset_sourcing/source_tape_proteinnet.yaml \
    source_folder=/path/to/proteinnet \
    repo_owner=<your-hf-username-or-org>
```

The converter still contains an explicit archive fallback for compatibility,
but the checked-in workflow and generated dataset provenance use the TAPE LMDB
source.

### ProteinGym scoring

ProteinGym doesn't fit the `sequence`/`targets` shape above: it is organized as
per-assay variant tables with its own UniProt-ID/function-category aggregation.
It is therefore sourced and scored outside the Prepare→Tune→Predict→Score
pipeline. Four tracks are supported -- DMS or clinical, substitutions or indels
-- plus zero-shot (masked-marginal / pseudo-likelihood) and supervised
frozen-embedding probe scoring.

```
modules/evaluate/scripts/dataset_sourcing/source_proteingym.py ← download+unzip a track's assay archive and its reference CSV, write normalized parquet
modules/evaluate/src/proteingym/masked_marginal.py    ← substitution masked-marginal scoring + indel pseudo-likelihood scoring
modules/evaluate/src/proteingym/supervised.py          ← pooled-embedding extraction + per-assay CV (official or custom folds)
modules/evaluate/src/proteingym/aggregate.py           ← per-assay Spearman/AUC + UniProt-ID → function-category aggregation
modules/evaluate/src/run_proteingym_zero_shot.py       ← zero-shot CLI entry point
modules/evaluate/src/run_proteingym_supervised.py      ← supervised CLI entry point
```

Sourcing follows the method in the [ProteinGym GitHub README](https://github.com/OATML-Markslab/ProteinGym):
the reference CSV comes from the GitHub repo, and each track's assays come from one zip
hosted at `<base_url>/ProteinGym_<version>/<track>.zip` (e.g.
`https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_substitutions.zip`).
The zip is always downloaded and extracted in full; `assays`/`max_assays` only limit which
extracted assay CSVs get converted to parquet. Select a track with `track: dms_substitutions |
dms_indels | clinical_substitutions | clinical_indels` in the config.

As with the other sourcing scripts, this only writes local artifacts by
default. Set an HF owner to upload the normalized track after conversion:

```bash
uv run python modules/evaluate/scripts/dataset_sourcing/source_proteingym.py \
    modules/evaluate/configs/dataset_sourcing/source_proteingym.yaml \
    track=dms_substitutions \
    repo_owner=<your-hf-username-or-org>
```

This uploads to `<repo_owner>/proteingym-<track>`. Use
`upload_layout=single_repo repo_id=<owner>/<repo>` to place the artifact in a
single repository instead.

Each track is written to its own `work_dir/<track>/outputs` subfolder by
default, so sourcing multiple tracks does not overwrite another track's
`reference.parquet`, `stats.json`, or `README.md`. Set `work_dir` explicitly to
opt back into a shared directory across tracks.

Sourcing always writes one parquet file per assay under `data/`, preserving
ProteinGym's official fold columns (`fold_random_5`, `fold_modulo_5`, and
`fold_contiguous_5`) when present. To also write a combined artifact, use
`combine_assays=true`; it is written as `data/combined.parquet` while the
per-assay files remain available. With `create_split_subsets=true`, sourcing
writes reproducible per-assay `train`/`validation`/`test` labels using a seeded
80/10/10 split and writes `combined_train_random.parquet`,
`combined_validation_random.parquet`, `combined_test_random.parquet` plus the
corresponding `stratified` artifacts. These reuse the same partition helpers as
the other dataset sources. MMseqs cluster subsets are intentionally omitted:
within one assay every row is a single/double mutant of the same wild-type
sequence, so sequence-identity clustering does not provide a meaningful
partition. Use the official `fold_contiguous_5` or `fold_modulo_5` columns for
a position-based generalization split instead.
Supervised evaluation can use the persisted split with `split=seeded`, an
official ProteinGym fold column, or a reproducible `split=custom_kfold`.
`split=official_average` evaluates each applicable official scheme and averages
their assay metrics equally. The supervised probe is selected with
`probe_type=linear|mlp|torch_linear`; `probe_search_space` performs a grid
search on an inner holdout of each training fold before refitting the winner on
all available training rows. Zero-shot evaluation does not train and therefore
does not require a train/validation/test split.

Assays with fewer than three variants are retained in their per-assay and
combined artifacts but do not receive synthetic split labels or appear in the
pooled split views, because three nonempty splits cannot be formed from fewer
than three observations.

Then score zero-shot with a masked language model (indel assays automatically use
pseudo-likelihood scoring instead of masked-marginal, based on the sourced
`is_indel` flag):

```bash
uv run modules/evaluate/src/run_proteingym_zero_shot.py \
    modules/evaluate/configs/proteingym_zero_shot.yaml \
    model_id=facebook/esm2_t12_35M_UR50D
```

Or fit a supervised probe per assay using either the official ProteinGym CV
folds (`split: fold_random_5 | fold_modulo_5 | fold_contiguous_5`), the
equal-weight official average (`split: official_average`), the persisted
train/test split (`split: seeded`), or a freshly generated k-fold split
(`split: custom_kfold`):

```bash
uv run modules/evaluate/src/run_proteingym_supervised.py \
    modules/evaluate/configs/proteingym_supervised.yaml \
    model_id=facebook/esm2_t12_35M_UR50D
```

Both entry points write `<dms_id>_scored.parquet` (each variant's `model_score`),
`assay_scores.parquet` (per-assay Spearman for continuous DMS scores, or AUC for
binary clinical labels), and `summary.json` (the ProteinGym-style aggregate: mean
over UniProt IDs, then over function categories) under `output_dir`.

The zero-shot config can also run reference scorers with `baselines: [random,
blosum62, random_init_trunk]`; each baseline is written below
`output_dir/baselines/<name>/`. `blosum62` is substitution-only and is a simple
matrix baseline, not ProteinGym's official MSA-derived Site-Independent scorer.
Set `compute_uncertainty: true` to add the bootstrap standard error to
`summary.json`; `n_bootstrap` controls the number of resamples for both entry
points and defaults to `10000`.



## Step 1 — Prepare

**Purpose:** Validate the workspace, load the tokenizer, load and tokenize the dataset once, derive the task-specific label count, build the collator/DataLoaders, and instantiate the model wrapper matching `task_type` (`AutoModelForSequenceClassification` for `sequence_classification`/`sequence_regression`, `AutoModelForTokenClassification` for `token_classification`, the custom trunk-plus-pairwise head for `contact_prediction`, or the frozen masked-language-model Jacobian wrapper for `categorical_jacobian`).

### Named benchmark splits

`steps.prepare.split.name` selects which published split method to evaluate on. `default` loads `train`/`validation`/`test`; any other name loads `train_<name>`/`validation_<name>`/`test_<name>`, and Prepare raises if those splits aren't in the dataset. The name is written to prediction metadata and `scores_long.parquet`, so repeated runs for different generalization conditions remain separate.

```yaml
steps:
    prepare:
        split:
            name: stratified # loads train_stratified / validation_stratified / test_stratified
```

Set a role explicitly when a benchmark's split names don't follow that convention; explicit roles win over the derived name.

```yaml
steps:
    prepare:
        split:
            name: low_to_high_mutation
            train: train_low_mutation
            validation: validation_low_mutation
            test: test_high_mutation
```

Set `validation: null` to synthesize validation data from the selected training split using `validation_fraction`. This configuration selects precomputed dataset splits; benchmark-specific assignment logic, such as sequence-identity clustering or mutation counting, should be applied when publishing the dataset splits.

**Key config fields (`steps.prepare`):**

| Field | Default | Description |
|---|---|---|
| `dataloader_pin_memory` | `true` | Pin host memory for faster host→GPU transfer |
| `dataloader_persistent_workers` | `true` | Keep dataloader workers alive between epochs |
| `batch_size` | `16` | Examples per batch |
| `packed` | `false` | Pack AMPLIFY inputs for CUDA variable-length attention; pairwise tasks are unsupported |
| `max_tokens_per_batch` | `null` | Optional packed token budget; single-process only; supersedes `batch_size` and length grouping |
| `embedding_batch_size` | `128` | Examples per batch during frozen-trunk embedding extraction |
| `dataloader_num_workers` | `null` | Defaults to `min(os.cpu_count(), num_gpus * 4)` when unset |
| `max_length` | `512` | Maximum sequence length after truncation |
| `pad_to_multiple_of` | `8` | Pads batches to a multiple of this value |
| `sequence_column` | `sequence` | Dataset column containing the input sequence |
| `label_column` | `targets` | Dataset column containing the label(s) — biomap-research eval datasets use `targets`, not `label` |
| `id_column` | `id` | Per-example identifier column; falls back to a positional index if absent |
| `tokenize_num_proc` | `null` | Defaults to `min(os.cpu_count(), num_gpus * 4)` when unset |
| `split` | `{name: default, train: train, validation: validation, test: test}` | Split method to evaluate on; a non-`default` `name` resolves the roles to `<role>_<name>` unless set explicitly |
| `validation_fraction` | `0.1` | Share of `train` held out as validation when `split.validation` is `null` or missing from the dataset |
| `random_truncate_by_split` | `{train: true, validation: false, test: false}` | Randomly crop only the configured splits; evaluation splits remain deterministic by default |
| `length_grouped_sampling` | `false` | Batch similarly-tokenized-length examples together to reduce per-batch padding |
| `embedding_cache.layers` | `[-1]` | Hidden-state indices to use for a frozen-embedding probe; each is pooled with `embedding_cache.pooling` and concatenated, e.g. `[-1, -2, -3]` |
| `embedding_cache.autocast_dtype` | `float16` | Precision used during the frozen-trunk forward pass; `float32` disables autocast |
| `embedding_cache.normalize_before_pooling` | `false` | L2-normalize token representations before pooling |

### Packed evaluation

Set `steps.prepare.packed=true` to concatenate the sequences in each batch and
run AMPLIFY's CUDA variable-length attention. Packed evaluation requires:

- CUDA; CPU evaluation does not support packed batches.
- An AMPLIFY checkpoint re-uploaded after the packed model forward signatures
    were added. The checkpoint must accept `cu_seqlens` and `max_seqlen`; sequence
    classification heads must also accept `num_sequences`.
- A non-pairwise task. `contact_prediction` and other pairwise tasks are not
    supported because their inputs and outputs are residue-pair structured.

The collator emits flattened `(1, T)` `input_ids`, per-token `position_ids`,
`cu_seqlens`, `max_seqlen`, and (for sequence classification) `num_sequences`.
The model uses `cu_seqlens` to prevent attention from crossing sequence
boundaries. Prediction and embedding utilities unpack model outputs back to
one row per source example before existing task metrics run, so packed mode
does not change the prediction file schema.

By default, `batch_size` still controls how many examples are packed together.
Set `steps.prepare.max_tokens_per_batch` to use a token budget instead; this
option is single-process only, supersedes `batch_size`, and requires
`steps.prepare.packed=true`. The budget includes the collator's padding to
`pad_to_multiple_of`.

For example, a single-GPU packed evaluation can be launched with:

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

    The same settings are available as the composable
    [`configs/packed.yaml`](configs/packed.yaml) overlay:

    ```bash
    uv run accelerate launch --num_processes 1 \
        modules/evaluate/src/run_evaluate.py \
        modules/evaluate/configs/config.yaml \
        modules/evaluate/configs/models/amplify_120m.yaml \
        modules/evaluate/configs/tasks/sequence_classification.yaml \
        modules/evaluate/configs/datasets/localization_prediction.yaml \
        modules/evaluate/configs/packed.yaml
    ```

Packed mode is also compatible with frozen-trunk embedding caching. Cached
embeddings retain one pooled row per source example, while token-level tasks
retain one row per valid token; packed and padded cache entries use distinct
cache keys. ProteinGym uses separate runners and is unchanged by this option.

**Input:** `workspace.dataset_repo_id` (or `dataset_name`), `workspace.model_repo_id` (or `model_name`)
**Output:** in-memory `PreparedArtifacts` (tokenizer, model, dataloaders) — not persisted to disk

---

## Step 2 — Tune

**Purpose:** Train the configured model path on the `train` split. `tune_probe` freezes the pretrained trunk and trains a probe head, `finetune` trains the trunk and native MLP head, and `evaluate_as_is` skips this step entirely. Probe tuning can optionally use cached pooled embeddings and can perform hyperparameter search (Optuna TPE with median pruning, random, or grid) over `hyperparameter_search.search_space`.

**Key config fields (`steps.tune`):**

| Field | Default | Description |
|---|---|---|
| `mode` | `tune_probe` | `tune_probe` trains a frozen-trunk probe, `finetune` trains the trunk plus native MLP head, or `evaluate_as_is` skips tuning |
| `head.head_type` | `mlp` | `mlp` (existing native model-head path), `torch_linear` (one `torch.nn.Linear` layer), `sklearn_linear` (sklearn logistic/ridge), `knn`, `random_forest`, or `xgboost` |
| `head.search_space` | `{}` | Classical-estimator dimensions; candidates are selected on the validation split, and the winning estimator is used for test prediction after fitting on `train` only |
| `metric_aggregation` | `macro` | Averaging used for validation metric selection; automatically follows `steps.score.metric_aggregation` unless set explicitly |
| `learning_rate` | `1e-4` | Used when `hyperparameter_search.enabled=false` |
| `weight_decay` | `1e-4` | Used when `hyperparameter_search.enabled=false` |
| `max_epochs` | `3` | Maximum training epochs; used when `hyperparameter_search.enabled=false` |
| `warmup_ratio` | `0.06` | Fraction of total training steps used for LR warmup |
| `max_grad_norm` | `1.0` | Gradient clipping norm |
| `accumulation_steps` | `1` | Batches accumulated per optimizer step |
| `batch_size` | `null` | Cached-embedding probe batch size; set to `null` to use the PrepareStep loader size |
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
| `majority_class` | No model inference — predicts the most frequent class in the train split and exposes the training class prior as probabilities; a simple non-model baseline for classification tasks |
| `train_mean` | No model inference — predicts the train split's label mean for every example; the regression counterpart to `majority_class`, for `sequence_regression` tasks |
| `separation_prior` | No model inference — for `contact_prediction` and `categorical_jacobian`, predicts the empirical contact rate for each residue-separation bin |

**Key config fields (`steps.predict`):**

| Field | Default | Description |
|---|---|---|
| `collect_probabilities` | `false` | Collect per-example per-class softmax probabilities; auto-enabled when effective metrics include `roc_auc` |
| `predictions_filename` | `predictions.parquet` | Output filename under `<workspace>/runs/<variant>/preds/` |
| `persist_to_disk` | `true` | Write predictions and the prediction manifest; disable for fast in-process evaluation |
| `controls` | `[untrained]` | Subset of `untrained`, `random_trunk`, `random_head`, `random_both`, `scrambled_labels`, `scrambled_sequences`, `majority_class`, `train_mean`, `separation_prior` |

**Input:** `PreparedArtifacts` (trained model + test dataloader) from Prepare/Tune
**Output:** `<workspace>/runs/<variant>/preds/predictions.parquet`; settings are also recorded in `<workspace>/runs/<variant>/run_manifest.json`'s "predict" section

---

## Step 4 — Score

**Purpose:** Compute task-specific metrics from the Predict step's output and, optionally, bootstrap confidence intervals for those metrics plus a statistical comparison between the trained model and each configured control condition, to quantify how much of its performance is attributable to learning from the data rather than model/data artifacts.

**Key config fields (`steps.score`):**

| Field | Default | Description |
|---|---|---|
| `metrics` | `null` | List of metrics to compute (task-specific defaults apply when unset), e.g. `accuracy`, `f1`, `precision`, `recall`, `roc_auc`, `mcc`, `mse`, `r2`, `pearsonr`, `spearmanr`, `contact_precision_at_l_long`, `contact_auc_long`, `pseudo_perplexity`, `mean_log_prob` |
| `metric_aggregation` | `macro` | One of `micro`, `macro`, `weighted` |
| `bootstrap_enabled` | `true` | Enable bootstrap confidence intervals + control comparisons |
| `n_samples` | `1000` | Bootstrap resamples — the code warns if below the recommended minimum of 1000 for statistically meaningful CIs/p-values |
| `n_permutations` | `1000` | Trained-vs-control paired randomization-test permutations; larger values improve p-value resolution |
| `output_filename` | `bootstrap_results.parquet` | Per-resample bootstrap metric values (bootstrap mode only) |
| `confidence_interval` | `0.95` | Confidence level for bootstrap intervals |
| `long_filename` | `scores_long.parquet` | Tidy/long-format sidecar written on every run, with or without bootstrap — concatenable across (model, dataset) runs for plotting |

**Input:** `<workspace>/runs/<variant>/preds/predictions.parquet`
**Output:** `<workspace>/scores/scores_long.parquet` (always, shared across run variants), plus `<workspace>/runs/<variant>/scores/bootstrap_results.parquet` when `bootstrap_enabled=true`; scores/confidence intervals are also recorded in `<workspace>/runs/<variant>/run_manifest.json` under the `score`/`bootstrap` sections

> **Note:** with `n_samples` below ~1000, bootstrap confidence intervals and trained-vs-control p-values are too coarse to support a claim of statistical significance (the smallest achievable nonzero p-value is `1/n_samples`).

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
intentionally overrides some of them, such as sampling, metric aggregation,
and whether bootstrap scoring is enabled.

---
