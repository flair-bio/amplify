# Evaluation

Evaluate protein language models on downstream tasks, compare them with
controls, and report scores with optional confidence intervals.

**All evaluations use one entry point:** `modules/evaluate/src/run_evaluate.py`.
YAML files select the model, dataset, task, and prediction approach. Ordinary
tasks, contact prediction, shared baselines, and ProteinGym use this same CLI,
but have different runner functions where their scientific protocols differ.

- [Quick start](#quick-start)
- [Running and configuring](#running-and-configuring)
- [Tasks and datasets](#tasks-and-datasets)
- [Controls and scoring](#controls-and-scoring)
- [Other workflows](#other-workflows)
- [Outputs](#outputs)
- [How the code works](#developer-guide)
- [Troubleshooting](#troubleshooting)
- [Testing](#testing)

## Quick Start

Run all commands from the repository root. Install the evaluation dependencies:

```bash
uv sync --extra evaluate
```

### Sequence classification

This example trains an MLP probe on frozen AMPLIFY embeddings to predict one
of two solubility labels per protein:

```bash
uv run python -m modules.evaluate.src.run_evaluate \
    modules/evaluate/configs/config.yaml \
    modules/evaluate/configs/models/amplify_120m.yaml \
    modules/evaluate/configs/datasets/solubility_prediction.yaml \
    workspace.run_id=quickstart \
    steps.tune.hyperparameter_search.enabled=false \
    steps.tune.max_epochs=2 \
    'steps.predict.controls=[]' \
    steps.score.bootstrap_enabled=false
```

These are reduced training settings for learning the workflow, not the full
benchmark protocol. The command still processes the dataset and may download
the checkpoint and data on its first run. A compatible GPU speeds up backbone
inference; ordinary padded evaluation can also run on CPU.

The base preset normally uses **100 hyperparameter-search trials, fresh
train+validation refitting, a random-trunk control, and bootstrap scoring**.
The control selects and refits its own probe, adding work. To use the full
preset, omit the four `steps.*` overrides and choose a different run ID.

For an inference-only environment check, append
[`configs/smoke_test.yaml`](configs/smoke_test.yaml) after the dataset YAML and
before any CLI overrides. It disables fitting and shortens sequences, but
does not limit the number of examples. A pretrained checkpoint may have a
newly initialized classification head, so its smoke-test accuracy is not a
meaningful downstream benchmark.

## Running and Configuring

### Glossary

| Term | Meaning |
|---|---|
| Trunk / backbone | The protein language model that produces representations |
| Head / probe | A smaller task-specific model operating on those representations |
| Embedding cache | Frozen representations reused instead of running the backbone every training epoch |
| Task | Label, prediction, and scoring semantics, such as sequence classification |
| Runner | A function coordinating preparation, fitting, prediction, and publication |
| Control | A model or data ablation, such as random backbone weights or shuffled labels |
| Shared baseline | A separately generated reference method, such as amino-acid composition |
| Workspace / variant | The output location and label for a model/dataset/run configuration |

### Config files

Ordinary runs compose files in order: **base, model, dataset, optional predictor
overlay**, followed by `key=value` overrides. Later settings win.

| Location | Responsibility |
|---|---|
| [`configs/config.yaml`](configs/config.yaml) | Default cached-MLP evaluation protocol |
| [`configs/models/`](configs/models/) | Checkpoint source and output label |
| [`configs/datasets/`](configs/datasets/) | Dataset source, label contract, default task, and benchmark-specific settings |
| [`configs/tasks/`](configs/tasks/) | Task defaults, including metrics and supported evaluation mode |
| [`configs/predictors/`](configs/predictors/) | Alternative probes, fine-tuning, and contact predictors |
| [`configs/baselines/`](configs/baselines/) | Methods for the separate `--baselines` route |
| [`configs/accelerate/`](configs/accelerate/) | Launch settings for Accelerate |

A dataset's `workspace.default_task_type` automatically loads the matching
task YAML immediately before that dataset overlay. You usually do not need
to supply a task file. An explicit task YAML uses normal left-to-right merging;
without one, `workspace.task_type=...` can select another task overlay, subject
to dataset compatibility.

Input sources and output labels are separate:

```yaml
workspace:
  model_name: amplify-120m
  model_repo_id: flair-bio/amplify-120m
  dataset_name: solubility_prediction
  dataset_repo_id: flair-bio/evaluation-suite
  dataset_config_name: solubility_prediction
  run_id: experiment01
```

`dataset_config_name` selects the named dataset inside the suite.
`dataset_name` labels its output directory. Model IDs may also be local
checkpoint directories.

The configuration is validated with Pydantic in
[`config.py`](src/config.py). Unsupported combinations fail
early. The descriptions here refer to shipped presets; schema fallback
defaults can differ from `configs/config.yaml`.

### Common recipes

Add these files or overrides after the dataset overlay:

| Goal | Setting |
|---|---|
| Use the default frozen-backbone MLP probe | No additional predictor overlay |
| Use a linear or classical probe | Append `configs/predictors/torch_linear.yaml`, `sklearn_linear.yaml`, `knn.yaml`, `random_forest.yaml`, or `xgboost.yaml` |
| Fine-tune the backbone and native head | Append `configs/predictors/mlp_finetuning.yaml` |
| Use fixed training settings | `steps.tune.hyperparameter_search.enabled=false steps.tune.max_epochs=20` |
| Change the HPO budget | `steps.tune.hyperparameter_search.n_trials=10` |
| Disable controls | `'steps.predict.controls=[]'` |
| Disable bootstrap and permutation comparisons | `steps.score.bootstrap_enabled=false` |
| Keep another run | `workspace.run_id=experiment02` |
| Change the seed | `seed=42` |
| Move the benchmark output root | `workspace.base_path=/path/to/eval` |
| Store caches elsewhere | `workspace.tmp_path=/path/to/shared-cache` |
| Disable CUDA TF32 | `tf32=false` |

Quote list-valued overrides so shells such as zsh do not interpret brackets.
Use predictor overlays rather than changing only `head_type` or `steps.mode`:
they also adjust cache, refit, and search settings to compatible values.
Classical probes require cached embeddings. Fine-tuning requires live backbone
computation, so its overlay disables the embedding cache.

### Running many combinations

One invocation handles one model/dataset configuration and one target.
Repeat invocations for other models, seeds, or targets:

```bash
for model in amplify_120m esm2_35m; do
    uv run python -m modules.evaluate.src.run_evaluate \
        modules/evaluate/configs/config.yaml \
        "modules/evaluate/configs/models/${model}.yaml" \
        modules/evaluate/configs/datasets/solubility_prediction.yaml \
        workspace.run_id=comparison01
done
```

These use the full preset. Run sequentially unless you have resources for
multiple evaluations; avoid simultaneous cache refreshes for the same inputs.

## Tasks and Datasets

### Supported tasks

| Task | Output |
|---|---|
| `sequence_classification` | One class per protein |
| `sequence_regression` | One numeric target per protein |
| `sequence_multilabel_classification` | Multiple labels per protein |
| `token_classification` | One class per residue |
| `contact_prediction` | Residue-pair contact scores using the fixed CASP protocol |
| `pseudo_perplexity` | A zero-shot masked-residue score |
| `calibration` | Zero-shot agreement with homolog amino-acid profiles |

Pseudo-perplexity and calibration require `evaluate_as_is`; their task overlays
configure this automatically. ProteinGym uses assay-specific workflows rather
than an ordinary dataset task.

Ordinary supervised datasets normally provide `sequence`, `targets`, and
optionally `id`. Column names are configurable under `steps.prepare`.
Classification needs `workspace.num_labels` and compatible integer labels.
The [dataset catalog](docs/datasets.md) documents each dataset's label contract.

To use a downloaded suite, keep the dataset overlay and override
`workspace.dataset_repo_id=/path/to/evaluation-suite` and
`workspace.dataset_revision=null`. For a standalone Hub repository, select its
ID and set `workspace.dataset_config_name=null` if it has no named config.
Local loading supports the suite layout or split Parquet artifacts. Dataset
sourcing, standardization, and publication are maintained in
[amplify-downstream-data-prep](https://github.com/flair-bio/amplify-downstream-data-prep/tree/main);
this module consumes the prepared artifacts.

### Named benchmark splits

By default, the roles are `train`, `validation`, and `test`. If validation is
absent, a seeded fraction of training is held out.

`steps.prepare.split.name=stratified` selects `train_stratified`,
`validation_stratified`, and `test_stratified`. For other naming conventions,
set `steps.prepare.split.train`, `.validation`, and `.test` explicitly.
Only select splits actually provided by the dataset.

### Test subsets

Use named column-equality filters to score several parts of one test split:

```yaml
steps:
  predict:
    test_subsets:
      cohort_a: {cohort: a}
      cohort_b: {cohort: b}
```

The column must exist in the source test split. Filters within one subset are
combined with AND; missing columns and empty selections are errors.
The primary predictor and independently fitted controls are fitted once and
reused. Outputs go under `tests/<name>/`; their metadata is recorded in the
root `run_manifest.json`.

Set `steps.predict.test_subsets=null` to disable all inherited subsets.
For a single inherited entry, use e.g.
`steps.predict.test_subsets.casp14=null`. Contact filters must retain every
scoring unit belonging to each selected chain.

### External test sets

`external_test_sets` scores separate datasets with the same fitted predictor
and controls, without using them for training or model selection:

```yaml
external_test_sets:
  - name: external_cohort
    dataset_repo_id: owner/dataset
    split: test
```

External data must share the main dataset's label meanings and class-index
mapping, not just its number of classes. Outputs use the external dataset
name under the same benchmark root. All destinations are reserved before
fitting. External sets are skipped in `evaluate_as_is` mode.

## Controls and Scoring

### Controls

`steps.predict.controls` requests model/data ablations. The ordinary preset
requests `random_trunk`: an independently selected and refitted probe on
random-backbone features in cached-probe runs. Other supported controls include
fresh heads, fully random models, and shuffled sequences or labels.

Support and behavior depend on the task and prediction approach. In particular,
live/native random-trunk ablations differ from independently fitted cached
probes. `scrambled_labels` is produced during scoring, not model inference.
See the [controls guide](docs/controls.md) for definitions and interpretation.

### Shared baselines

Generate model-independent baselines separately, without a model overlay:

```bash
uv run python -m modules.evaluate.src.run_evaluate --baselines \
    modules/evaluate/configs/config.yaml \
    modules/evaluate/configs/datasets/solubility_prediction.yaml \
    modules/evaluate/configs/baselines/sequence_classification.yaml \
    'baselines.names=[majority_class]'
```

They publish their own predictions and scores. Model runs do not automatically
load or compare these shared artifacts. Match dataset, split, and preprocessing
settings when comparing results. Some baselines, such as MMseqs2 neighbors,
require additional external tools; see [shared baselines](docs/controls.md#shared-sequence-baselines).

<a id="step-4--score"></a>

### Scoring

Task and dataset overlays choose the reported metrics and the validation
selection metric separately. For example, solubility selects by validation
accuracy and reports test accuracy and macro F1.

For ordinary flat tasks, the base preset enables bootstrap confidence intervals
and paired trained-versus-control comparisons. Key settings are:

```yaml
steps:
  score:
    bootstrap_enabled: true
    n_samples: 1000
    n_permutations: 1000
    confidence_interval: 0.95
```

These statistics reuse predictions; they do not rerun inference.
The task determines valid metrics and resampling units. Contact prediction
uses its own hierarchical scoring and rejects generic bootstrap settings.

## Other Workflows

### Contact prediction

The CASP14-16 dataset uses the shared `contact_prediction` task with one of
three predictor overlays:

| Overlay in `configs/predictors/` | Approach |
|---|---|
| `contact_embedding.yaml` | Frozen residue embeddings with a fitted bilinear512 head |
| `contact_attention.yaml` | Attention features with a fixed L1 logistic readout |
| `contact_jacobian.yaml` | Zero-shot categorical-Jacobian scoring from a masked LM |

```bash
uv run python -m modules.evaluate.src.run_evaluate \
    modules/evaluate/configs/config.yaml \
    modules/evaluate/configs/models/amplify_120m.yaml \
    modules/evaluate/configs/datasets/contact_prediction_casp.yaml \
    modules/evaluate/configs/predictors/contact_embedding.yaml \
    'workspace.dataset_revision=<immutable-40-character-HF-commit>' \
    workspace.run_id=casp-embedding \
    seed=0
```

Contact runs require **one process**, schema-6.0 data, and an immutable
40-character Hub commit SHA, or a local Parquet artifact with revision unset.
The presets enforce fixed scientific protocols; do not layer generic HPO,
refitting, or smoke settings over them.

The dataset selects `casp14`, `casp15`, and `casp16` test subsets. Fitted heads
and their random-trunk controls reuse one development fit across these subsets;
the Jacobian predictor remains zero-shot. Inference uses full chains and
original residue coordinates, not generic cropped sequence batches.

Reports include per-unit `unit_scores.parquet` and hierarchical
`scores.parquet`, rather than the ordinary flat-task table.
Protocol implementations are in [contact fitting](src/trainers/contact.py),
[contact task semantics](src/tasks/contact_prediction.py), and
[contact metrics](src/metrics/contact_metrics.py).

### Layerwise linear probes

To compare representations at different depths, append
`configs/predictors/torch_linear.yaml` and then these overrides to an ordinary
sequence or token task command:

```text
steps.layerwise.enabled=true
'steps.layerwise.layers=[0,6,-1]'
steps.tune.hyperparameter_search.enabled=false
steps.tune.refit_on_train_validation=false
```

Choose indices supported by the model: `0` is the embedding output and `-1`
is the final hidden state. Each layer has an independent linear head and its
own output variant. With bootstrap enabled, `layerwise_comparison.parquet`
compares layers with the deepest selected layer. All-layer caches can be large.

### ProteinGym

Use a ProteinGym preset directly, **without** the ordinary base config:

```bash
uv run python -m modules.evaluate.src.run_evaluate \
    modules/evaluate/configs/proteingym_zero_shot.yaml
```

Use `proteingym_supervised.yaml` for supervised assay evaluation. Both require
one process and use the same runtime and artifact infrastructure while retaining
assay-specific data, mutation scoring, fold selection, and aggregation.
See the [ProteinGym guide](docs/proteingym.md) for inputs and protocols.

## Outputs

### Output disk layout

An ordinary model run uses:

```text
<benchmark_root>/<task_type>/<dataset_name>/<model_name>/
└── <variant>-<run_id>/seed_<seed>/
    ├── model/
    ├── predictions.parquet
    ├── scores.parquet
    ├── run_manifest.json
    ├── controls/<control>/
    │   ├── model/
    │   └── predictions.parquet
    └── tests/<subset>/          # when named test subsets are configured
        ├── predictions.parquet
        └── scores.parquet
```

Only applicable files are written. Named subsets replace the root test
prediction/score files; their checkpoint and manifest remain shared at the run
root. Subset control predictions are also stored inside the subset directory;
fitted control checkpoints remain shared. Inference-only runs need not save a
model. Contact adds structured score tables; layerwise probes use separate
layer variants.

`workspace.base_path` defaults to `eval`; `workspace.output_dir`, when set,
overrides that benchmark root. The default probe variant is
`mlp-frozen_trunk`; fine-tuning uses `mlp-tuned_trunk`.
The quick-start run is therefore under
`eval/sequence_classification/solubility_prediction/amplify-120m/mlp-frozen_trunk-quickstart/seed_710019/`.

Existing model runs are rejected, including interrupted ones. Use a new
`workspace.run_id` to keep another experiment, or `--overwrite` to replace the
selected run/seed. There is no resume mode. Different input sources require
distinct model/dataset labels or a new benchmark root.

Embedding caches live separately under
`<benchmark_root>/tmp/<task>/<dataset>/<model>/embeddings/`, shared across
compatible runs. `workspace.tmp_path` changes the cache root. Do not remove
caches or locks while jobs are active. `max_cache_gb` limits disk writes, not RAM.

Shared baselines live at
`<benchmark_root>/<task>/<dataset>/baselines/<method>/fit-<hash>/test-<hash>/`,
with their own predictions, scores, and manifest.

### Reading results

For ordinary flat tasks, `predictions.parquet` holds identifiers, predictions,
labels, and optional probabilities/sequences. `scores.parquet` is a tidy
metric table with model/dataset/variant/seed identity.

Combine model score tables for sequence classification:

```python
from pathlib import Path

import polars as pl

paths = Path("eval/sequence_classification").glob("*/*/*/seed_*/scores.parquet")
scores = pl.concat([pl.read_parquet(path) for path in paths], how="diagonal_relaxed")
print(
    scores.filter(pl.col("control").is_null()).select(
        "model_name", "dataset_name", "variant", "seed", "metric", "value"
    )
)
```

This selects root model reports; subset reports live one level deeper under
`tests/<name>/`. A null `control` identifies standalone scores. Non-null
control rows contain paired comparisons, including `diff_*`, `p_value`, and
`p_value_adjusted`. Differences are trained minus control: positive favors the
trained model for higher-is-better metrics, negative for losses such as MSE.
Do not concatenate contact or ProteinGym tables into this flat-task schema.

## Developer Guide

### Code Layout

```text
modules/evaluate/
├── configs/             # runnable presets and overlays
├── docs/                # controls, dataset catalog, ProteinGym guide
└── src/
    ├── run_evaluate.py  # sole CLI
    ├── config/          # schemas, composition, validation
    ├── runners/         # runtime setup and flat/contact/baseline orchestration
    ├── dataset/         # loading, tokenization, subsets, feature/data preparation
    ├── model/           # construction, architectures, extraction, prediction
    ├── trainers/        # fitting, selection, training, final refitting
    ├── controls/        # independent control construction/fitting and inference
    ├── tasks/           # label/prediction semantics and scientific scoring
    ├── metrics/         # metrics, aggregation, bootstrap, comparisons
    ├── artifacts/       # records, paths, caches, checkpoints, publication
    ├── proteingym/      # assay workflows and benchmark-specific protocols
    └── utils/           # seeds, tracking, amino-acid constants
```

### Execution flow

The ordinary cached sequence-classification path is:

```text
run_evaluate.main()
  parse_cli() -> load_run_config()
  initialize_runtime()
  run_evaluation()                           runners/pipeline.py
    select runner; validate contact protocol if applicable
    reserve_outputs()
      _prepare_test_subsets()
      run_flat_evaluation()                  runners/flat_evaluation.py
        get_flat_task()
        build_predictor() + load_tokenizer()
        prepare_training_data()
        build_probe()
        fit_predictor() -> save_predictor()
        prepare_control_predictors()
        evaluate_test_set()                  once per test view
          predict_test_set()
          persist_predictions()
          task.score()
          publish_scores()
```

Output reservation happens before dataset/model work. Contact validation
precedes reservation. Fitting, when required, happens once for the primary
predictor and each independent control; test views reuse the fitted resources.
Reservation records success or failure, and the runtime closes tracking when
the run exits.

<a id="step-1--prepare"></a>

### Preparation and fitting

For cached probing, data flows from sequences to tokens, then through the frozen
backbone to embeddings. Sequence tasks pool residue representations; token
tasks retain label-aligned residue features.

`prepare_training_data()` returns separate development and test records; it
does not construct or own the predictor. The runner retains the original model
in `predictor.trunk` and replaces `predictor.model` with the probe.

`fit_predictor()` uses development data, selects hyperparameters on validation,
and optionally refits a fresh probe on train+validation. Feature standardization
uses training statistics during selection and development-union statistics
during refitting. Test labels never contribute to model selection or fitting.
Prediction produces records; task scoring computes results without filesystem
I/O; artifact functions publish them.

There are **no step or trainer wrapper classes**. `steps.prepare`,
`steps.tune`, `steps.predict`, and `steps.score` are configuration sections.
Task objects define scientific semantics, not workflow execution. Models and
workspaces own state; ordinary functions coordinate the work.

### Main records

| Record | Defined in | Carries |
|---|---|---|
| `Runtime` | `runners/runtime.py` | Device and application-owned Accelerator |
| `Predictor` | `model/factory.py` | Live model, tokenizer, optional frozen trunk |
| `TrainingData`, `TestData` | `dataset/types.py` | Separate loaders and feature/provenance metadata |
| `ControlPredictors` | `model/controls.py` | Independent control models and seeds |
| `ControlInputs` | `controls/contact.py` | Contact-control construction and serialization resources |
| `FitResult`, `PredictOutput`, `ScoreOutput` | `artifacts/results.py` | Fitting provenance, predictions, and scores |
| `EvaluationResult`, `PipelineResult` | `artifacts/results.py` | Per-view and complete evaluation results |
| `AssayResult` | `proteingym/runner.py` | Assay predictions with an optional protocol-specific score override |

The records pass data explicitly; constructing one does not perform I/O.
`PipelineResult.evaluations` contains workspace/prediction/score records.
Named subsets and external datasets are available through `test_subsets`
and `external`. The CLI publishes outputs; Python callers can inspect the
returned results.

### Distributed execution (`accelerate`) patterns

Ordinary flat evaluation uses one application-owned `Accelerator`. To launch
with Accelerate, replace `uv run python -m` with
`uv run accelerate launch --module` and supply the appropriate launch options
or a config from `configs/accelerate/`. Contact and ProteinGym require one process.

All ranks must participate in collectives, control preparation, and checkpoint
state gathering. Only the main rank publishes outputs and scores gathered flat
predictions. Configure trainable parameters before wrapping models with
`accelerator.prepare()`. Follow the existing trainer and runner patterns when
changing distributed code.

### New dataset, task, or model overlay

Copy an existing YAML in the matching config directory. Dataset overlays
declare the source, default task, target type, label format, and class count
where applicable. Prepare new source data in
[amplify-downstream-data-prep](https://github.com/flair-bio/amplify-downstream-data-prep/tree/main).
Adding a dataset or checkpoint normally does not require another Python task.

### Adding a Task Type

For a new ordinary task, add a narrow `FlatTask` subclass in `tasks/`, declare
its label contract and metrics, and add its class name to `_TASKS` in
[`tasks/registry.py`](src/tasks/registry.py). Add a task YAML and extend
[`model/factory.py`](src/model/factory.py) if model construction differs.
Use [sequence classification](src/tasks/sequence_classification.py) as a small
example and [token classification](src/tasks/token_classification.py) for
label-alignment hooks.

A genuinely different execution protocol belongs in runner functions, not in
task methods that construct models or call trainers. Exercise the scientific
behavior using the existing task and pipeline tests.

## Troubleshooting

| Symptom | Check |
|---|---|
| Run already exists | Choose another `workspace.run_id`, or deliberately pass `--overwrite` |
| Missing split, invalid labels, or class-count error | Match the dataset overlay, label column, `num_labels`, and split configuration |
| Tokenization worker exits | Check host RAM; try `steps.prepare.tokenize_num_proc=1` |
| Host RAM runs out | Token/all-layer features can be large; the embedding disk-cache budget is not a RAM cap |
| Invalid mode/head/cache combination | Use the matching predictor overlay rather than changing only one field |
| Packed evaluation fails on CPU | Leave `steps.prepare.packed=false`; packing requires CUDA and a compatible AMPLIFY checkpoint |
| Multi-GPU evaluation hangs | Check that every rank reaches the same collectives |
| Classification accuracy is poor with `evaluate_as_is` | A pretrained-only checkpoint may have a randomly initialized task head; train a probe or load a task-fitted checkpoint |

## Testing

Install development dependencies and run the existing evaluation suite:

```bash
uv sync --extra evaluate --extra pretrain --extra dev
uv run pytest tests/evaluate
```

For focused changes, select the relevant files, such as
`test_pipeline_tasks.py`, `test_subsets.py`, or `test_contact_embedding.py`.
Task contracts are covered by `test_tasks.py`.
