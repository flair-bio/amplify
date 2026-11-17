# Evaluation module refactor implementation plan

## Contents

- [1. Purpose and final decisions](#1-purpose-and-final-decisions)
- [2. Scope and non-goals](#2-scope-and-non-goals)
- [3. Current architecture and the problems to address](#3-current-architecture-and-the-problems-to-address)
- [4. Architecture rules](#4-architecture-rules)
- [5. Complete target source tree](#5-complete-target-source-tree)
- [6. Per-file implementation specification](#6-per-file-implementation-specification)
- [7. One-entry-point configuration and dispatch](#7-one-entry-point-configuration-and-dispatch)
- [8. Data and result contracts](#8-data-and-result-contracts)
- [9. Multi-target removal specification](#9-multi-target-removal-specification)
- [10. Scientific and execution invariants](#10-scientific-and-execution-invariants)
- [11. Artifact compatibility](#11-artifact-compatibility)
- [12. Migration map](#12-migration-map)
- [13. Implementation phases and exit criteria](#13-implementation-phases-and-exit-criteria)
- [14. Validation strategy](#14-validation-strategy)
- [15. Final verification checklist](#15-final-verification-checklist)
- [16. Guidance for keeping this refactor simple](#16-guidance-for-keeping-this-refactor-simple)

## 1. Purpose and final decisions

This document is an implementation plan, not an instruction to rewrite the
module in one change. The goal is to make the existing evaluation capabilities
easier to understand and extend without introducing a generic workflow
framework or changing scientific protocols.

Implementation and migration of the existing tests are complete. The current
directory layout and service ownership are documented in
`modules/evaluate/README.md`. The evaluation suite passes locally: 1,715 passed
and 3 skipped (two require CUDA; one task case does not apply to
pseudo-perplexity). Ruff, formatting, mypy, and the repository hooks also pass.
Real distributed/GPU parity has not been validated on this machine.

The final design must satisfy these decisions:

1. **One entry point: `modules/evaluate/src/run_evaluate.py`.** Ordinary
   evaluation, baseline-only evaluation, ProteinGym supervised evaluation, and
   ProteinGym zero-shot evaluation all start there.
2. **Remove multi-target evaluation.** Remove its configuration, dispatch,
   model runner, baseline runner, aggregation, tests, and documentation.
   Ordinary single-target regression remains supported.
3. **Use `artifacts/`, not `storage/`.** This package owns output paths,
   checkpoints, prediction files, reports, manifests, caches, locks, and atomic
   writes.
4. **Keep a focused `proteingym/` package.** Retain benchmark-specific assay,
   fold, mutation-scoring, and aggregation logic there. Share application
   infrastructure and suitable numerical/model primitives with other runs.
5. **Remove `steps/` as an implementation layer.** Preparation, fitting,
   controls, prediction, scoring, and publication remain operations, but not
   classes whose only state is configuration.
6. **Use functions for orchestration.** Do not add a workflow superclass,
   stage registry, DAG executor, dependency-injection container, or universal
   predictor wrapper.
7. **Separate data, predictors, and results.** Replace the mutable
   `PreparedArtifacts` handoff with smaller, explicit contracts.
8. **Preserve scientific behavior and output compatibility**, except for the
   explicitly requested removal of multi-target execution and consolidation
   of command-line entry points.

The following is the intended direction:

```text
run_evaluate.py
    |
    +-- shared config loading and runtime setup
    |
    +-- ordinary evaluation workflow
    |       +-- flat/token/embedding evaluation
    |       +-- contact evaluation
    |       +-- baseline-only evaluation
    |
    +-- ProteinGym supervised workflow
    |
    +-- ProteinGym zero-shot workflow
            |
            +-- shared model, fitting, tracking, and artifact primitives
```

ProteinGym is part of the same application, not a second application.
Keeping its scientific rules together does not require separate CLI scripts.

## 2. Scope and non-goals

### In scope

- Establish clear ownership for orchestration, scientific semantics, data,
  fitting, inference, and persistence.
- Remove stage classes and cross-stage calls.
- Narrow the task interface.
- Separate training inputs from test inputs.
- Preserve fit-once/evaluate-many behavior for subsets and external tests.
- Integrate ProteinGym with the shared CLI and run infrastructure.
- Remove multi-target support end to end.
- Update configurations, tests, launch examples, and developer documentation.
- Preserve the current supported model families and predictor backends.

### Not in scope

- Changing optimizer defaults, metrics, fitting protocols, or benchmark rules.
- Adding multi-output regression as a replacement for multi-target execution.
- Introducing assay parallelism or distributed ProteinGym support.
- Replacing OmegaConf, Pydantic, Accelerate, PyTorch, or sklearn.
- Redesigning dataset sourcing or changing sourced dataset schemas.
- Introducing a new package manager, test runner, or formatting tool.
- Renaming every existing function simply for naming consistency.
- Deleting existing run directories, historical multi-target outputs, or
  user-generated caches.
- Making all result tables identical across scientifically different tasks.

## 3. Current architecture and the problems to address

The main sources of complexity are:

- `tasks/base.py` mixes scientific semantics with model loading, fitting,
  feature requests, independent control construction, and scoring.
- `steps/prepare.py` combines dataset loading, tokenization, model construction,
  feature caching, layerwise handling, and test-set replacement.
- `PreparedArtifacts` represents several execution states through a model
  field and many optional data/control payloads.
- Tuning mutates that shared payload instead of returning a distinct fitted
  predictor result.
- Control preparation imports and calls methods on other step classes.
- `steps/tune.py` includes backend-specific checkpoint publication.
- Prediction and scoring combine scientific operations with file publication.
- Contact prediction has a substantially independent protocol but is wrapped
  in interfaces designed primarily for flat-row evaluations.
- The ProteinGym CLI scripts contain runtime setup and scientific workflow
  logic in addition to argument parsing.
- Multi-target behavior crosses the CLI, configuration loader, baseline
  runner, model runner, aggregation helpers, and dataset presets.

These are responsibility problems, not merely file-length problems. Moving
the current classes into differently named directories will not solve them.

## 4. Architecture rules

### 4.1 Functions versus classes

Use the following defaults:

| Requirement | Preferred implementation |
| --- | --- |
| Parse inputs and call operations in order | Function |
| Compute a transformation or metric | Function |
| Describe internal values | Dataclass or existing appropriate typed record |
| Validate external config or serialized metadata | Pydantic model |
| Own trainable parameters and buffers | `torch.nn.Module` |
| Own a resource or meaningful run lifetime | Small class or context manager |
| Provide cohesive alternative task semantics | Small task interface and implementations |

Do not retain a class solely because its constructor stores `config`.
Do not convert existing useful model/cache/workspace objects into function
collections just to remove classes.

### 4.2 Dependency boundaries

The intended imports flow from orchestration toward capabilities:

```text
CLI
  -> config loading
  -> runtime
  -> runners / ProteinGym run functions
       -> dataset preparation
       -> tasks
       -> model construction and inference
       -> trainers
       -> controls
       -> metrics
       -> artifacts
```

Specific rules:

- Lower-level modules never import `runners/` or `run_evaluate.py`.
- Task semantics never reserve outputs, initialize Accelerate, or publish
  files.
- Dataset records never own fitted predictors or fitted controls.
- Metric calculations do not require a workspace or tracking session.
- Artifact modules do not decide benchmark metrics or fit models.
- Common modules do not import ProteinGym implementations to provide generic
  behavior. The application dispatch/schema-selection boundary is the
  deliberate exception: it must know which benchmark run/config to select.
- `proteingym/` may use common capabilities; ordinary evaluation must not
  depend on the ProteinGym package.
- Keep shared config schema modules import-light. Validate task registry
  membership after schema construction rather than importing the full task
  graph from every field definition.
- Use `TYPE_CHECKING` for annotation-only imports where appropriate. Do not
  replace a real dependency cycle with pervasive dynamic imports.
- Lightweight package `__init__.py` files must not eagerly import every
  model/backend or initialize logging.

### 4.3 A cohesive exception is preferable to a universal abstraction

Flat evaluation, contact prediction, and assay-based evaluation need not
implement the same preparation or fitting interface.

Share operations when their input/output contracts and scientific behavior
actually match. A short amount of explicit workflow code is preferable to a
generic runner with many backend flags and exceptional branches.

### 4.4 Structure inside files

Use the following default order:

1. Module docstring.
2. Future imports, standard-library imports, third-party imports, project
   imports.
3. Constants and small supporting type declarations.
4. Public functions/classes, grouped by responsibility.
5. Private helpers near the public operation they support.
6. The execution guard in `run_evaluate.py` only.

There is no rule requiring all functions before all classes. A model class
and its factory can share a file; metric functions do not need a class.
Keep public orchestration easy to read from top to bottom. Prefer explicit
keyword arguments at boundaries with several similar settings.

## 5. Complete target source tree

Each package has an `__init__.py`. These should contain only package
documentation and a deliberately small set of stable exports, if any.
They are shown below so the final layout is unambiguous.

This planning artifact lives in `modules/evaluate/docs/REFACTOR_PLAN.md`,
outside the source package.

```text
modules/evaluate/src/
├── __init__.py
├── run_evaluate.py
│
├── config/
│   ├── __init__.py
│   ├── shared.py
│   ├── evaluation.py
│   ├── load.py
│   └── validation.py
│
├── runners/
│   ├── __init__.py
│   ├── runtime.py
│   ├── evaluate.py
│   ├── flat.py
│   ├── contact.py
│   └── baselines.py
│
├── tasks/
│   ├── __init__.py
│   ├── base.py
│   ├── registry.py
│   ├── sequence_classification.py
│   ├── sequence_multilabel_classification.py
│   ├── sequence_regression.py
│   ├── token_classification.py
│   ├── pseudo_perplexity.py
│   ├── calibration.py
│   └── contact_prediction.py
│
├── dataset/
│   ├── __init__.py
│   ├── types.py
│   ├── loading.py
│   ├── preparation.py
│   ├── tokenization.py
│   ├── dataloader.py
│   ├── collator.py
│   ├── embeddings.py
│   ├── subsets.py
│   ├── baseline_data.py
│   ├── contact.py
│   └── contact_features.py
│
├── model/
│   ├── __init__.py
│   ├── factory.py
│   ├── heads.py
│   ├── classical.py
│   ├── inference.py
│   ├── embeddings.py
│   ├── layerwise.py
│   ├── adapters.py
│   ├── controls.py
│   ├── baselines.py
│   ├── contact_model.py
│   ├── contact_attention.py
│   └── contact_jacobian.py
│
├── trainers/
│   ├── __init__.py
│   ├── fit.py
│   ├── selection.py
│   ├── refit.py
│   ├── torch.py
│   ├── classical.py
│   ├── layerwise.py
│   ├── contact.py
│   └── optimization.py
│
├── controls/
│   ├── __init__.py
│   ├── contact.py
│   ├── fitting.py
│   └── inference.py
│
├── metrics/
│   ├── __init__.py
│   ├── metrics.py
│   ├── scoring.py
│   ├── bootstrap.py
│   ├── comparisons.py
│   ├── calibration_metrics.py
│   ├── contact_metrics.py
│   └── tables.py
│
├── artifacts/
│   ├── __init__.py
│   ├── results.py
│   ├── workspace.py
│   ├── cache.py
│   ├── checkpoints.py
│   ├── predictions.py
│   ├── reports.py
│   ├── manifests.py
│   ├── baselines.py
│   └── files.py
│
├── proteingym/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── runner.py
│   ├── supervised.py
│   ├── zero_shot.py
│   ├── masked_marginal.py
│   ├── aggregate.py
│   └── baselines.py
│
└── utils/
    ├── __init__.py
    ├── amino_acids.py
    ├── seed.py
    └── tracking.py
```

There is deliberately no:

- `steps/`;
- legacy `runners/pipeline.py`;
- `contracts/`;
- `storage/`;
- multi-target module;
- `run_proteingym_supervised.py`;
- `run_proteingym_zero_shot.py`;
- second `main()` or executable guard in a source module.

The tree is a responsibility map, not a requirement to fill small files with
wrappers. Extract actual implementations into the listed modules. If an
extraction would create a file containing only a forwarding function,
reconsider that boundary before adding the file.

## 6. Per-file implementation specification

Names in this section are proposed public APIs. Preserve existing function
names where they already express the responsibility well. Private helpers
should remain with the implementation that uses them.

### 6.1 Entry point, runtime, and result records

#### `run_evaluate.py`

Public API:

- `main() -> None`: the only CLI entry point.

Private helpers:

- `_parse_args(argv)`: parse config paths, dotlist overrides, `--baselines`,
  and `--overwrite`.
- `_dispatch_run(config, runtime, *, baseline_only, overwrite)`: use explicit
  branches to call ordinary evaluation or one of the ProteinGym run functions.

Responsibilities:

- Parse arguments.
- Load the appropriate typed config.
- Enter the shared runtime context.
- Dispatch to a run function.
- Convert usage/config errors into clear CLI errors and a nonzero exit.

Do not put model creation, assay loops, tokenization, metric calculation,
checkpoint serialization, or logging configuration at import time here.
Keep the `if __name__ == "__main__": main()` guard only in this file.

#### `runners/runtime.py`

Types:

- `Runtime`: small dataclass containing resolved device, process identity,
  optional Accelerator, and tracking handle.

Functions:

- `configure_logging()`.
- `configure_precision(tf32)`.
- `initialize_runtime(config, *, baseline_only=False)`: context manager that
  configures the runtime and always closes tracking resources.
- `validate_process_support(config, runtime)`: reject unsupported process
  counts before creating outputs or loading expensive models.

Reuse `utils.seed.seed_everything()`. Do not reimplement seeding.
Construct one Accelerator where the selected workflow needs it; pass the
instance down rather than constructing fallback instances in low-level
operations.

Keep branch-specific defaults during integration. In particular, do not
silently replace the current ProteinGym seed/dtype defaults with ordinary
evaluation defaults merely because setup now lives in a shared module.

Baseline-only runs should not load a model or require a GPU merely because
the shared entry point supports model evaluation.

#### `artifacts/results.py`

Types:

- `FitResult`: fitted model plus selection, final-fit, seed, and protocol
  metadata.
- `EvaluationMetadata`, `ContactEvaluationMetadata`: serialized identifying
  context.
- `PredictOutput`: observation table, requested conditions, control tables,
  metadata, and optional publication references.
- `ScoreOutput`, `StructuredScoreOutput`, `ContactScoreOutput`: existing
  scientific result shapes, without write methods.
- `EvaluationResult`, `PipelineResult`: top-level result records.
- `ConfidenceInterval`, `ControlComparison`: statistical result records.
- `ArtifactReferences`: optional small path record if needed to avoid adding
  unrelated path fields to every scientific result.

Keep the existing useful layer-keyed `PipelineResult.predictions` and
`PipelineResult.scores` views during migration. Preserve external/subset
result access.

Result records may describe output paths; constructing them must not write
files. If `EvaluationResult.workspace` remains for Python API compatibility,
keep its type import annotation-only rather than making results depend on
artifact implementation at runtime.

### 6.2 `config/`

#### `shared.py`

Types:

- `RuntimeConfig`: shared seed/precision/device settings where applicable.
- `WandbConfig`.
- `DatasetWorkspaceConfig`.
- `EvaluationWorkspaceConfig`.
- `EmbeddingCacheConfig`.

Functions:

- `validate_directory_label()`.
- `dataset_source_identity()`.

Keep source identity, output identity, and task semantics distinct. A
ProteinGym workspace must not need to register a flat task to obtain paths.
Do not introduce a misleading universal label schema for all benchmarks.

Shared config types should not initialize filesystem objects or download data.
Pure path derivation is acceptable; output reservation is not.

#### `evaluation.py`

Types:

- `DatasetSplitConfig`, `ExternalTestSetConfig`.
- `PrepareConfig`, `HeadConfig`, `TuneConfig`.
- `HyperparameterSearchConfig`, `HyperparameterSearchSpaceConfig`.
- `PredictConfig`, `ScoreConfig`, `LayerwiseConfig`, `BaselinesConfig`.
- `StepsConfig`: retained initially as the existing config namespace, not as
  an instruction to retain step classes.
- `EvaluationConfig`, `BaselineRunConfig`.

Remove `MultiTargetConfig` and all `multi_target` fields.
Keep the ordinary `steps.*` YAML structure during this refactor to avoid
unnecessary churn across experiment overlays.

Put ordinary variant-name derivation here or in a nearby pure helper. It
must preserve head/mode/split output identities.

#### `load.py`

Functions:

- `load_run_config(paths, overrides=None, *, baseline_only=False)`:
  application-level loader.
- `compose_evaluation_config(paths, overrides=None)`: retain ordinary task
  overlay composition behavior.
- `load_evaluation_config(paths, overrides=None)`.
- `load_baseline_config(paths, overrides=None)`.

The application loader selects a schema using the top-level `workflow`
discriminator described in section 7. It must not attempt to validate a
ProteinGym config against ordinary task defaults first.
For an ordinary baseline-only request, select `BaselineRunConfig` before
requiring model-only settings. Reject a ProteinGym baseline-only flag rather
than accidentally applying the ordinary baseline projection.

Preserve:

- multiple YAML paths;
- left-to-right YAML merge order;
- dotlist overrides applied after merging;
- ordinary dataset-selected default task insertion;
- existing task-overlay precedence;
- explicitly selected baseline-only settings.

Check for removed multi-target configuration before projecting an ordinary
config into `BaselineRunConfig`. Do not silently drop it during projection.

Reuse `modules.core.utils.config_loader` and OmegaConf rather than creating
a second merge implementation.

#### `validation.py`

Functions:

- `validate_evaluation_config(config)`.
- `validate_baseline_config(config)`.
- `validate_runtime_constraints(config, runtime)`.
- `reject_removed_options(raw_config)`.

Resolve capability compatibility, probability collection, selection metric
direction, packing restrictions, cache restrictions, and refit restrictions.
Delegate contact scientific protocol rules to the contact domain functions.
Delegate ProteinGym scientific validation to its config/domain modules.

Local data/model checks still belong where actual inputs become available;
do not assume config validation can validate real label contents.

### 6.3 `runners/`

#### `evaluate.py`

Functions:

- `run_evaluation(config, runtime, *, dataset=None, overwrite=False)`.
- `select_evaluation_workflow(config)`: choose flat or contact evaluation
  from the resolved ordinary task.
- `_reserve_output_targets(...)`: coordinate reservation of the main and
  external outputs before any fit begins.
- `_complete_run(...)` and `_fail_run(...)`, if needed for coordinated
  manifest transitions.

Ordinary output reservation remains rank-aware. Broadcast reservation
failures so other ranks do not proceed into collectives after rank zero fails.
Reuse artifact workspace operations instead of duplicating locking.

This is not a registry of stage objects. It is an application coordinator.

#### `flat.py`

Functions:

- `run_flat_evaluation(config, workspace, runtime, *, dataset=None)`.
- `evaluate_test_set(predictor, controls, test_data, task, config, workspace,
  runtime)`.
- `_evaluate_layerwise_test_set(...)`: keep per-layer evaluation and
  comparisons explicit.

Order:

1. Resolve the task once with the configured class vocabulary.
2. Load/validate the raw inputs and requested test selections.
3. Build the predictor and prepare any required training inputs.
4. Fit, or prepare the evaluate-as-is predictor without fitting.
5. Prepare independent control predictors once.
6. Prepare each test input view without mutating training state.
7. Predict and gather observations.
8. Score and publish on the appropriate process.
9. Return `PipelineResult`.

Named test subsets and external tests reuse the fitted predictor,
standardization state, selected settings, and fitted controls.
Never rebuild/refit the primary predictor inside the test-set loop.

#### `contact.py`

Functions:

- `run_contact_evaluation(config, workspace, runtime, *, dataset=None)`.
- `evaluate_contact_test_set(...)`.
- `_prepare_contact_controls(...)` only if a workflow-specific coordination
  helper is needed.

Call the contact data, feature, fitting, prediction, and scoring services
directly. Do not route through `FlatTask` or create empty token loaders to
satisfy a generic signature.

Keep protocol validation, development fingerprints, fixed fitting rules,
zero-shot behavior, independent random-trunk controls, and per-edition test
publication explicit.

#### `baselines.py`

Functions:

- `run_baselines(config, *, dataset=None)`.
- `_run_baselines_for_test(...)`.
- `_run_contact_baselines(...)`.
- `_baseline_run_config(...)` and `_baseline_score_config(...)` where useful.

Use model baseline builders for predictions, metric functions for scoring,
and artifact functions for persistence/reuse.

Remove `run_multi_target_baselines()` and
`_publish_multi_target_summary()`. Do not replace them with a hidden loop in
another module.

### 6.4 `tasks/`

#### `base.py`

Types:

- `TaskCapabilities`: immutable declaration of stable supported features.
- `FlatTask`: small interface/default implementation for flat-row task
  semantics.

Methods to retain or consolidate:

- `resolve_num_labels()`.
- `validate_labels()` / the existing categorical label checks.
- `validate_dataset_contract()`.
- `align_labels()`.
- `expand_tokenized_example()`.
- `collate_labels()`.
- `extract_predictions()`.
- `predicted_probabilities()`.
- `split_batch_rows()` with contextual information where necessary.
- `default_metrics()`, `validate_metric_names()`.
- `prepare_scoring()`.
- `compute_metrics()`, optional grouped/per-example metric hooks.
- `bootstrap_group_id()`, `flatten_for_metrics()`.
- `scramble_labels()` where the task defines the valid permutation unit.

These methods need not all be abstract. Keep simple defaults for actual
shared semantics, and override only real differences.

Remove from the interface:

- Hugging Face model loading and checkpoint loading;
- generic predictor fitting;
- feature-cache requests;
- independent control model construction;
- publication settings;
- global pipeline/workspace validation.

Do not preserve `TaskHandler -> FlatMetricsTask -> StructuredTask` merely
for API continuity. Contact uses its own domain functions.

#### `registry.py`

Functions:

- `get_flat_task(task_type, num_labels=None)`.
- `available_flat_tasks()`.

A small explicit name-to-class mapping is sufficient. Task instances hold
their own classification vocabulary; do not mutate a shared singleton.
Preserve support for the current ordinary task names.

#### `sequence_classification.py`

- `SequenceClassificationTask`: scalar categorical targets, prediction
  decoding, vocabulary checks, supported metrics and controls.

#### `sequence_multilabel_classification.py`

- `SequenceMultilabelClassificationTask`: multi-hot labels, sigmoid
  probabilities, thresholds, multilabel metrics, and input validation.

#### `sequence_regression.py`

- `SequenceRegressionTask`: scalar continuous targets, prediction decoding,
  regression metrics.
- Keep a private grouped-scoring implementation if needed to preserve
  equal-weight cluster scoring and task-valid label scrambling.

Validation groups and test metric groups remain independent settings.

#### `token_classification.py`

- `TokenClassificationTask`: residue alignment, ignore-index masking,
  variable-length prediction rows, valid scrambling behavior, and metrics.

#### `pseudo_perplexity.py`

- `PseudoPerplexityTask`: masked-example expansion, target token extraction,
  log-probability interpretation, and supported metrics.

It remains evaluation-only; do not introduce dummy fitting.

#### `calibration.py`

- `CalibrationTask`: masked-site/profile preparation, decoding, cohort
  metadata, and scoring preparation.

Keep fold-based temperature cross-fitting and cohort definitions. Do not
turn calibration fitting during scoring into ordinary probe tuning.

#### `contact_prediction.py`

Functions:

- `validate_contact_protocol(config)`.
- `validate_contact_runtime(config, runtime)`.
- `build_contact_metadata(...)`.
- `predict_contact_units(dataset, predict_chain, metadata)`.
- `score_contact_predictions(predictions, config, *, seed)`.

These functions assemble scientific observations/results and enforce the
contact protocol. Ranking and aggregation mathematics stay in
`metrics/contact_metrics.py`.

Move data alignment to `dataset/contact.py`, feature requirements to
`dataset/contact_features.py`, fitting to `trainers/contact.py`, and control
model construction to `model/controls.py` or existing contact model helpers.
Do not recreate the old large handler as a renamed class.

### 6.5 `dataset/`

#### `types.py`

Types:

- `CachedFeatures`: cached-representation metadata and its extraction backend.
- `LayerwiseInputs`: original token loaders for layerwise probing.
- `TrainingData`: training, optional validation, and named final-refit inputs.
- `TestData`: one named evaluation input view with metadata and grouping.
- `FeatureRequest`, `FeatureSet`: scientific feature requirements and
  materialized values/fingerprint where the existing request mechanism helps.

Design requirements:

- Keep cache metadata explicit on data records. Do not add single-loader
  wrapper classes that provide no additional behavior or provenance.
- Keep concrete embedding-dataset inspection within fitting and inference
  services rather than distributing it through task semantics.
- No fitted model or independent-control model fields.
- Retain original tokens/provenance where inference controls need them.
- Do not require fake training/validation inputs for evaluation-only tasks.
- Contact datasets remain their own types rather than nullable members of
  an all-purpose input container.
- Prefer immutable identity/provenance records. Tensors and loaders are
  inherently stateful; frozen dataclasses do not make those deeply immutable.

#### `loading.py`

Functions:

- `load_raw_evaluation_dataset()`, `load_evaluation_dataset()`.
- `resolve_evaluation_data()`, `resolve_external_data()`.
- `select_dataset_splits()`, `ensure_validation_split()`.

Move source/split resolution out of the current large dataloader module.
Keep split synthesis deterministic and preserve dataset revision handling.

#### `preparation.py`

Functions:

- `prepare_training_data(...) -> tuple[TrainingData, TestData]`.
- `prepare_test_data(...) -> TestData`.
- `prepare_feature_inputs(...)`: extracted loaders and cache metadata.

Coordinate loading outputs, tokenization, collators, loaders, and feature
materialization. Accept a tokenizer/extractor explicitly where needed;
do not create and retain a fitted predictor inside the returned data object.

External tests and named subsets both use `prepare_test_data()` with
different resolved sources/selections. They must not swap test fields on a
training-and-model container.

#### `tokenization.py`

Functions:

- `tokenize_dataset()`.
- `collect_original_positions()`.

Preserve task-controlled expansion and alignment, cropping behavior,
stable IDs, split-specific random truncation, and original residue positions.

#### `dataloader.py`

Types:

- `DataLoaderConfig`.
- `LengthGroupedSampler`, `TokenBudgetBatchSampler`.

Functions:

- `build_dataloaders()`, `token_lengths()`.
- `metric_groups_by_id()`, `metric_groups_for_ids()`.
- `attach_metric_groups()`.

Keep batching and row/group alignment here. Dataset downloading and
tokenization no longer belong in this module.

#### `collator.py`

Types:

- `EvaluationCollator`, `EvaluationCollatorConfig`.

Functions:

- `unpack_packed()`, `input_id_rows()`.

Preserve packed/unpacked label behavior and the required model-forward
arguments. Do not add evaluation orchestration.

#### `embeddings.py`

Types:

- `EmbeddingDataset`.
- A reduced `EmbeddingBackend` or `FeatureSource` record describing source
  tokens and extraction identity, not owning a trained model.

Functions:

- `build_embedding_dataloaders()`.
- `embedding_collate_fn()`.
- `find_embedding_dataset()`: retain only where needed during migration,
  then remove if explicit input types make it redundant.

Model-side extraction stays in `model/embeddings.py`; cache I/O stays in
`artifacts/cache.py`.

#### `subsets.py`

Functions:

- `select_test_subsets()`.
- `resolve_test_sets()`: resolve main/subset/external descriptors without
  fitting or running inference.

Validate missing columns, empty subsets, and contact-chain completeness
before expensive fitting. Output names are labels, not a reason to alter
development-set fitting identity.

#### `baseline_data.py`

Functions:

- `baseline_context()`, `contact_baseline_context()`.
- `external_baseline_context()`, `preprocessing_identity()`.
- `sequence_rows()`, `unlabeled_sequence_rows()`, `site_profile_rows()`.
- `artifact_identity()` or a more domain-specific identity helper.
- `align_baseline()`, `align_token_baseline()`, `mmseqs_coverage()`.

Keep scientific input/preprocessing identities and alignment here.
Move file loading, directory selection, and persisted-artifact validation
to `artifacts/baselines.py` where those operations concern storage.

#### `contact.py`

Types:

- `ContactAlignment`, `ContactUnit`, `ContactChain`, `ContactDataset`.
- `ContactTokens`, `CandidatePairs`.

Functions:

- `load_contact_dataset()`, `prepare_contact_dataset()`.
- `align_contact_sequence()`, `tokenize_contact_chain()`.
- `candidate_pairs()`.
- `separation_prior_scores()`, `separation_prior_prediction_rows()`.

Keep canonical source-to-clean-to-tokenized residue alignment and candidate
pair eligibility in one domain module.

#### `contact_features.py`

Functions:

- `contact_development_feature_identity()`.
- `contact_development_attention_fingerprint()`.
- `request_contact_features(...)`.
- `prepare_contact_features(...)`.

Describe/extract required scientific features using explicit model/data
inputs. Delegate caching to `artifacts/cache.py`.
Keep test selection out of the development-feature identity.

### 6.6 `model/`

#### `factory.py`

Functions:

- `build_predictor(...)`.
- `build_backbone(...)`.
- `build_masked_lm(...)`.
- `load_tokenizer(...)`.

Centralize ordinary task-to-Hugging-Face-wrapper selection and supported
construction choices. ProteinGym uses backbone/MLM builders directly.

Pass regression/multilabel loss settings explicitly. Preserve
`trust_remote_code`, model kwargs, dtype, native-head checks, and model-family
behavior.

Do not silently standardize tokenizer settings across tasks: ProteinGym
coordinate preservation is an explicit requirement, not a global default
change.

#### `heads.py`

- `EmbeddingProbeHead`.
- `build_probe_layers()`.
- `validate_native_linear_head()`.

Own torch head architecture and its learned/fitted normalization buffers.
Do not include sklearn search orchestration.

#### `classical.py`

Functions:

- `build_estimator()`.
- `predict_classical()`.
- Backend-specific prediction/probability conversion helpers.

Keep estimator construction shared where semantics match.
Move search-space generation and selection to `trainers/selection.py`.
Do not equate ProteinGym's `linear` alias with ordinary `sklearn_linear`
without preserving the actual estimator/loss choices.

#### `inference.py`

Functions:

- `predict_tokens()`.
- `predict_features()`.
- `gather_prediction_rows()`.
- `gather_training_labels()`.
- `pad_for_gather()`, `unique_id_indices()` where still needed.

Prediction returns observations, not files. Preserve row IDs, probability
alignment, duplicate-padding removal, original residue positions, and
grouping metadata.

#### `embeddings.py`

Functions:

- `extract_embeddings()`.
- `residue_attention_mask()`, `pool_hidden_states()`.
- `pool_hidden_state_layers()`, `select_token_hidden_state_layers()`.
- `validate_embedding_cache_compat()`, `autocast_context()`.

Reuse these primitives from ProteinGym only with equivalent masking,
normalization, layer selection, truncation, and pooling settings.
Matching tensor shapes is not evidence that two extraction protocols match.

#### `layerwise.py`

- `LayerwiseProbeModel`.
- `resolve_layers()`.
- `predict_layerwise()`.
- Existing efficient shared-feature prediction helpers as private functions.

Keep multi-layer batching and shared-trunk efficiencies.
Control preparation belongs in `controls/`; this module may still execute
several explicitly supplied conditions together for efficient inference.

#### `adapters.py`

- `masked_lm_logits()`.
- `contact_positions()`.

Keep model-family compatibility details isolated. Avoid a new adapter class
hierarchy unless actual backend behavior requires one.

#### `controls.py`

- `ControlPredictors`: independent control models and seeds shared across test
  views.
- `build_random_trunk()`.
- `build_control_model()`.
- `contact_random_trunk_seed()`.
- `make_sequence_scrambler()`.

Own construction/input-transformation mechanics, not control fitting,
artifact reuse, or comparison statistics.

#### `baselines.py`

- `build_sequence_baseline()`.
- `build_prediction_baseline()`.
- `build_amino_acid_frequency_baseline()`.
- `build_calibration_baseline()`, `calibration_baseline_logits()`.
- `amino_acid_composition()` and MMseqs helper operations.

Preserve baseline definitions and explicit unsupported combinations.
ProteinGym-specific random/BLOSUM scoring stays in its focused package.

#### `contact_model.py`

- `BilinearHead`, `ContactPredictionModel`.
- `build_contact_model()`, `load_contact_trunk()`, `load_contact_model()`.

Keep predictor implementation and native model load/save behavior here.
The artifact coordinator decides where/when checkpoint files are written.

#### `contact_attention.py`

- `ContactAttentionModel`.
- Private attention extraction, symmetrization, and APC feature helpers.

#### `contact_jacobian.py`

- `ContactJacobianModel`.
- `build_contact_jacobian_model()`.
- `jacobian_to_contacts()`.

Do not change numerical protocols while moving these modules.

### 6.7 `trainers/`

#### `fit.py`

- `fit_predictor(...) -> FitResult`.

Select ordinary torch/classical/layerwise fitting and return the selected
model and fit metadata. The contact workflow can call contact fitting
functions directly.

The input data is not the return value. Do not mutate a shared
`prepared.model` field.

#### `selection.py`

- `select_hyperparameters()`.
- `evaluate_candidate()`.
- `sample_candidates()`, `create_optuna_study()`.
- `resolve_search_direction()`.
- `classical_search_space()`, `has_classical_search_space()`.

Keep supported search patterns, candidate bounds, pruning, architecture
search, and task-defined validation scoring.
Return a selection result with selected settings and score rather than
storing those values on a long-lived selector object.

Extract genuinely matching selection mechanics for ProteinGym without
changing its inner-holdout or candidate-selection protocol.

#### `refit.py`

- `refit_predictor()`.
- `estimate_seed_variance()`.
- Helpers to assemble permitted final-refit inputs.

Keep final refit distinct from selection. Extra final-refit splits must not
be inspected during candidate selection. Preserve the final-fit split label.

#### `torch.py`

- `TorchTrainer`, if it owns a run's model/optimizer/training state.
- `train()` / `fit()`, `evaluate_validation()` methods.
- Private batch iteration, early stopping, checkpoint restoration helpers.

If the refactored implementation only stores configuration, use module
functions instead of retaining a class unnecessarily.
Keep collective operations explicit and preserve gradient accumulation.

#### `classical.py`

- `fit_classical_probe()`.
- Input conversion and fitted preprocessing helpers.

Own fitting one estimator, not the whole experiment.

#### `layerwise.py`

- `fit_layerwise_probes()`.
- `fit_layerwise_random_trunk_control()`.

Preserve independent layer heads, per-layer normalization, selection
metadata, and currently unsupported combinations.

#### `contact.py`

- `BilinearProtocol`.
- Existing typed fitting-settings records.
- `fit_contact_head()`.
- `fit_contact_attention_head()`.

Retain the fixed scientific protocols. A zero-shot Jacobian predictor does
not need a fake optimizer loop; its fit metadata may explicitly indicate
that fitting was not required.

#### `optimization.py`

- `build_optimizer()`, `split_decay_parameters()`.
- `build_scheduler()`.
- `checkpoint_state()`, `load_checkpoint_state()`.
- `cleanup_between_trials()`, `configure_trainable_params()`.

Keep these helpers independent of publication and workflow order.

### 6.8 `controls/`

#### `fitting.py`

Functions:

- `prepare_control_predictors()`.
- `fit_control_predictor()`.

Handle ordinary cached random-trunk controls and layerwise controls through
explicit supported branches.
Use ordinary data/fitting/artifact functions, never neighboring step classes.

Construct independent feature identities for controls that change the trunk.
Preserve fitted controls across all test sets.

#### `contact.py`

- `ControlInputs`: an independently constructed contact model, seed, protocol
  metadata, and serialization model.
- `build_contact_controls()`.
- `contact_control_settings()`.

Keep the contact-specific record beside the functions that construct and
consume it. Shared `ControlPredictors` lives in `model/controls.py`.

#### `inference.py`

- `predict_control_conditions()`.
- Helpers for random-head, random-both, untrained, and scrambled-input
  inference where supported.

Do not fit inside inference.
Label scrambling remains a scoring operation; it should not trigger a model
forward pass or produce a misleading fitted predictor.

### 6.9 `metrics/`

#### `metrics.py`

Retain scalar metrics and validation exceptions:

- `accuracy()`, `f1()`, `precision()`, `recall()`, `mcc()`.
- `roc_auc()`, `average_precision()`, multilabel variants.
- `mse()`, `mae()`, `r2()`, `pearson()`, `spearman()`.
- `pseudo_perplexity()`, `mean_log_prob()`.
- `UndefinedMetricError`.

Preserve vocabulary-aware averaging and undefined-metric handling.

#### `scoring.py`

- `score_predictions()`: ordinary task point estimates, conditions,
  uncertainty, and paired comparisons.
- `compare_layer_scores()`.
- Private bootstrap coordination helpers.

Do not write files, update manifests, or initialize tracking here.
Contact and ProteinGym use their domain-specific scoring/aggregation
functions rather than pretending all observations are flat examples.

#### `bootstrap.py`

Retain:

- `as_indexable_array()`, `id_groups()`, `bootstrap_indices()`.
- `token_confusion_matrices()`, `confusion_scores()`.
- `bootstrap_confusion_scores()`.
- `mean_per_example_scores()`, `full_scores()`, `ci_stats()`.
- `collect_control_arrays()`.

Preserve resampling units, paired alignment, supported fast paths, and seeds.

#### `comparisons.py`

- `permutation_p_values()`.
- `benjamini_hochberg()`.
- `summarize_controls()`.

#### `calibration_metrics.py`

- `forward_kl()`, `fit_temperature()`, `crossfit_temperatures()`.
- `cohort_kl()`, `cohort_score()`, `cohort_accuracy()`.
- `mean_temperature()`.

#### `contact_metrics.py`

- `contact_prediction_row()`.
- `score_contact_prediction()`, `score_contact_unit()`.
- `aggregate_contact_scores()`.
- `precision_at_l()`, `contact_range_auc()`.

#### `tables.py`

- `build_long_df()`.
- `upsert_long_df(existing_frame, new_frame, ...)`: retain table-composition
  semantics, but accept frames instead of a path. Artifact publication loads
  any existing table, calls this pure operation, and writes the result.

Remove `aggregate_task_scores()` after confirming its remaining callers are
only the removed multi-target feature. Do not remove contact or ProteinGym
aggregation, which is unrelated to multi-target execution.

### 6.10 `artifacts/`

#### `workspace.py`

- `EvaluationWorkspace`.
- `get_evaluation_workspace()`.
- Existing `control_workspace()`, `test_workspace()`, `reserve_run()`,
  `initialize_run()`, and experiment identity checks.
- `reserve_run_outputs(...)`: common reservation/lifecycle context manager
  where behavior genuinely matches across workflows.

Retain the workspace class: it represents a coherent path/resource identity.
Preserve path layout, locks, overwrite checks, named test namespaces, shared
checkpoint paths, and separate temporary cache roots.

Do not require every workflow to use the ordinary task registry to create
paths. Keep workflow-level all-rank coordination outside low-level file locks.

#### `cache.py`

- `EmbeddingCache`.
- `embedding_cache_key()`, `tensor_cache_key()`.
- `weights_fingerprint()`, `estimate_embedding_bytes()`.
- `validate_tensor_cache()`, `load_tensor_cache()`.
- Cache materialization helper accepting an extraction callable.

Preserve invalidation identities, cache policies, dtype/shape validation,
cache size budget behavior, locking, and atomic writes.

#### `checkpoints.py`

- `save_predictor(...)`.
- `load_predictor(...)`.
- Private torch-head, layerwise-head, sklearn, and native-model serializers.

Accept explicit serialization inputs and backend metadata.
Preserve existing native model serialization methods rather than replacing
every format with a generic pickle.

Collective state collection may require every rank even though only rank
zero writes files. Make that calling contract explicit; do not place a
collective call inside an unconditional main-process-only block.

#### `predictions.py`

- `persist_predictions(...)`.
- `load_predictions(...)`.

Publish primary/control frames, preserve filenames and manifest sections,
and return publication references or updated result records.
Keep row alignment and domain metadata intact.

#### `reports.py`

- `publish_scores(...)`.
- `publish_structured_scores(...)`.
- `publish_layer_comparison(...)`.
- `publish_proteingym_summary(...)`.

Reuse low-level writing primitives, not a universal score-table schema.
Preserve the existing flat tidy tables, contact unit/hierarchy tables, and
ProteinGym assay/function/overall reports.
The ProteinGym publisher accepts already computed tables/metadata; it must
not import benchmark fitting or aggregation implementations.

Tracking calls can be coordinated here or by the workflow after scoring;
the numerical scoring functions themselves remain side-effect-free.

#### `manifests.py`

- `update_run_manifest()`.
- `validate_artifact_identity()`.
- `record_run_status()` if useful to avoid repeated status payload assembly.

Preserve atomic updates, namespaced test metadata, config serialization,
input fingerprints, and complete/failed status semantics.
Do not mark a run complete before publication finishes.

#### `baselines.py`

- `baseline_artifact_directory()`.
- `load_baseline_artifact()`.
- `save_baseline_artifact()`.
- `validate_baseline_artifact()`.
- Existing legacy-sidecar upgrade/cleanup helpers where still supported.

Separate persisted artifact handling from baseline prediction and scoring.
Never reuse a baseline solely because a file exists; validate its scientific
input/preprocessing identity first.

#### `files.py`

- `write_json()`, `write_parquet()`, `write_settings_json()`.
- `atomic_output()`, `cache_lock()`.
- `downcast_prediction_dtypes()`.

These are low-level filesystem/serialization operations. Do not add task
dispatch or benchmark protocol logic.

### 6.11 `proteingym/`

#### Why this package remains

The benchmark has a cohesive set of scientific conventions. Keeping those
together improves discoverability and avoids pretending benchmark-only
behavior is already a reusable general framework.

It must not own a parallel application stack: no CLI, duplicate logging
setup, duplicate seed/precision setup, generic file writers, or independent
workspace implementation.

#### `config.py`

Types:

- `ProteinGymWorkspaceConfig`.
- `ProteinGymRunConfig`.
- `ProteinGymSupervisedConfig`.
- `ProteinGymZeroShotConfig`.

Compose shared runtime/workspace/tracking components where compatible.
Keep benchmark-specific assay, fold, probe, track, and baseline settings here.

Preserve supported legacy direct-path configuration intentionally during
migration; do not silently reinterpret its fallback behavior as the strict
workspace-backed source behavior.

Move I/O and data download out of validators. Resolve scientific options
without forcing clinical labels into an ordinary regression schema.

#### `data.py`

- `resolve_suite()`.
- `load_assay_reference()`.
- `load_assay()`.

Preserve selected-subfolder downloads, pinned revisions, local suite-root
validation, reference/data layout checks, assay selection, and source errors.
Move source resolution here from `proteingym/workspace.py` and the runner.

#### `runner.py`

- `AssayResult`: predictions with an optional protocol-specific reported score.
- `run_assay_loop(...)`.

Iterate assays, call a supplied evaluator, collect per-assay observations,
invoke benchmark aggregation, and coordinate publication through
`artifacts/`.

The evaluator receives explicit assay metadata/data and may return an array or
`AssayResult` when a protocol-specific metric override is required. Keep this
small record beside the assay loop rather than in a separate `types.py`.
Models/tokenizers are created once outside this loop. Keep the existing
sequential execution model.

#### `supervised.py`

- `run_supervised(config, runtime, *, overwrite=False)`.
- `evaluate_supervised_assay(...)`.
- `resolve_folds()`.
- `run_cv()`, `run_seeded_split()`.
- `pool_embeddings()` only while the shared extractor cannot express the
  exact existing behavior.
- Private probe/scaling helpers until proven safe to replace with shared
  training primitives.

Move orchestration out of `run_proteingym_supervised.py` into this module.
Use the shared artifact lifecycle and runtime.

Share estimator construction, tensor operations, and training primitives
where contracts match. Preserve per-assay/per-fold independence, inner
holdout selection, fold-local preprocessing, and refitting.

#### `zero_shot.py`

- `run_zero_shot(config, runtime, *, overwrite=False)`.
- `predict_zero_shot_assay(...)`.
- `evaluate_reference_condition(...)`.

Move orchestration out of `run_proteingym_zero_shot.py`.
Use shared model/tokenizer construction with explicit coordinate-preserving
tokenizer behavior. Dispatch substitutions versus indels from source flags.

#### `masked_marginal.py`

- `parse_mutant()`.
- `score_assay_masked_marginal()`.
- `score_assay_indel()`.
- Existing private window/residue-index/log-likelihood helpers.

Keep this domain algorithm separate from pseudo-perplexity task semantics.

#### `aggregate.py`

- `binarize_labels()`.
- `compute_assay_metric()`, `compute_all_assay_metrics()`.
- Existing assay AUC/MCC/NDCG/top-recall calculations.
- `aggregate_proteingym_scores()`.
- `aggregate_proteingym_metrics()`.
- `bootstrap_standard_error_by_category()`.

Retain track-specific hierarchy, label direction, assay weighting,
clinical pooling, and uncertainty.

#### `baselines.py`

- `score_assay_random()`.
- `score_assay_blosum62()`.

The random-initialized-trunk condition uses the shared model construction
mechanics where equivalent, but retains its benchmark condition name,
initialization behavior, and output identity.

#### Remove `proteingym/workspace.py`

Its responsibilities move as follows:

- Suite resolution -> `proteingym/data.py`.
- Run reservation, paths, and manifest operations -> `artifacts/`.
- Single-process checks -> shared runtime validation.
- Benchmark orchestration -> `run_supervised()` / `run_zero_shot()`.

### 6.12 `utils/`

#### `amino_acids.py`

- Existing amino-acid constants.
- `resolve_aa_token_ids()`.

#### `seed.py`

- `seed_everything()`.

#### `tracking.py`

- `log_wandb()`.
- Small tracking integration helpers if needed.

Do not create a new catch-all helpers module. The current generic I/O and
filesystem helpers move to `artifacts/`.

## 7. One-entry-point configuration and dispatch

### 7.1 Workflow discriminator

Use a top-level discriminator with these values:

```yaml
workflow: evaluation
```

```yaml
workflow: proteingym_supervised
```

```yaml
workflow: proteingym_zero_shot
```

For ordinary runs, `evaluation` is the default when the key is absent.
The resolved ordinary `workspace.task_type` selects flat versus contact
execution; there is no need to specify both `workflow: contact` and
`task_type: contact_prediction`.

ProteinGym presets must specify their workflow explicitly. Do not guess
supervised versus zero-shot execution from the filename or probe settings.

Each config branch is a Pydantic type with an appropriate literal workflow
field. Reuse shared components rather than creating a single giant schema
whose fields are mostly irrelevant to any given run.

### 7.2 Supported CLI routes

```bash
# Ordinary model evaluation; existing overlay composition remains valid.
uv run python modules/evaluate/src/run_evaluate.py \
  modules/evaluate/configs/config.yaml \
  modules/evaluate/configs/models/amplify_120m.yaml \
  modules/evaluate/configs/datasets/localization_prediction.yaml

# Ordinary dataset baselines.
uv run python modules/evaluate/src/run_evaluate.py \
  --baselines \
  modules/evaluate/configs/config.yaml \
  <dataset-overlay.yaml> \
  <baseline-overlay.yaml>

# ProteinGym zero-shot; create or update this preset during migration.
uv run python modules/evaluate/src/run_evaluate.py \
  modules/evaluate/configs/proteingym_zero_shot.yaml

# ProteinGym supervised.
uv run python modules/evaluate/src/run_evaluate.py \
  modules/evaluate/configs/proteingym_supervised.yaml
```

The ProteinGym preset paths appear in existing documentation; verify which
presets are actually checked in and add/update them as needed. Do not assume
an example path proves a current file exists.

All routes accept multiple config paths and dotlist overrides consistently.
Retain `--overwrite` for model/ProteinGym runs. Preserve the current rejection
of `--baselines --overwrite` for ordinary validated/reusable baseline artifacts.

For the first implementation:

- `--baselines` selects ordinary baseline-only execution.
- ProteinGym reference conditions remain configured by its existing
  `baselines` option.
- Reject `--baselines` with a ProteinGym workflow with an actionable message
  rather than silently changing the meaning of the flag.

### 7.3 Dispatch shape

The following is architectural pseudocode, not a drop-in implementation:

```python
def main() -> None:
    args = _parse_args()
    config = load_run_config(
        args.paths, args.overrides, baseline_only=args.baselines
    )
    with initialize_runtime(config, baseline_only=args.baselines) as runtime:
        if config.workflow == "proteingym_supervised":
            run_supervised(config, runtime, overwrite=args.overwrite)
        elif config.workflow == "proteingym_zero_shot":
            run_zero_shot(config, runtime, overwrite=args.overwrite)
        elif args.baselines:
            run_baselines(config)
        else:
            run_evaluation(config, runtime, overwrite=args.overwrite)
```

Validate invalid flag/workflow combinations before dispatch. Ordinary
baseline loading must still use its reduced config schema rather than
requiring model-only settings.

There must be no executable ProteinGym wrapper scripts in the final tree.
Programmatic `run_supervised()` and `run_zero_shot()` APIs are functions,
not additional entry points.

## 8. Data and result contracts

### 8.1 Replace the universal prepared payload

Current conceptual flow:

```text
PreparedArtifacts
  -> same object with fitted model
  -> same object with controls
  -> copied object with different test inputs
```

Target conceptual flow:

```text
TrainingData ------------------> fitting ------------------> FitResult
Built predictor -------------------^                            |
                                                               v
TestData + ready predictor + control predictors ------------> PredictOutput
                                                               |
                                                               v
                                                    ScoreOutput / structured output
                                                               |
                                                               v
                                                    Artifact publication
```

The workflow owns these values and their lifetimes.

### 8.2 Evaluate-as-is is explicit

Do not force evaluation-only tasks through a fitting service merely to obtain
a `FitResult`.

- Build the appropriate predictor.
- Apply required runtime wrapping exactly once.
- Record that fitting did not run.
- Prepare supported controls.
- Evaluate.

For ordinary native models with newly initialized task heads, preserve the
existing construction/seed behavior; do not relabel an untrained head as a
previously fitted predictor.

### 8.3 Contact is a separate input family

Use `ContactDataset` and contact models directly.
Do not populate `train_dataloader=None`, `contact=...`, and many unrelated
fields just to satisfy an ordinary prepared-payload constructor.

The bilinear, attention, and Jacobian predictors retain their distinct
feature and fitting requirements.

### 8.4 Publication is a separate boundary

Prediction/scoring functions return scientific results.
Artifact functions publish those results and return paths or updated result
records.

Results need not be deeply immutable, but callers must not depend on
undocumented mutation to know whether a model has been fitted or a control
has been prepared.

### 8.5 ProteinGym assay result contract

The assay result must distinguish:

- per-variant predictions;
- a metric computed directly from those predictions;
- an explicitly reported metric obtained by averaging separate CV scheme
  scores.

For `official_average`, preserve both the representative prediction output
and the official reported metric. Never replace mean-of-metrics with
metric-of-mean-predictions during integration.

## 9. Multi-target removal specification

This is a deliberate feature removal, not a relocation.

### 9.1 Source removal

- Remove `runners/multi_target.py`.
- Remove `MultiTargetConfig`.
- Remove `EvaluationConfig.multi_target`.
- Remove `BaselineRunConfig.multi_target`.
- Remove multi-target selection from `run_evaluate.py`.
- Remove multi-target projection in the baseline config loader.
- Remove `run_multi_target_baselines()`.
- Remove `_publish_multi_target_summary()`.
- Remove multi-target-only resume and summary output logic.
- Remove `metrics.tables.aggregate_task_scores()` if no non-multi-target
  caller remains.
- Remove imports and aliases that supported only those paths.

### 9.2 Configuration behavior

Reject any non-null legacy `multi_target` section before evaluation, with a
message such as:

```text
multi_target evaluation has been removed. Configure one label_column and
run one scalar-target evaluation per invocation.
```

For migration compatibility, an explicit `multi_target: null` can be removed
from the raw config before typed validation. Non-null values must never be
silently ignored or interpreted as one of their target columns.

This compatibility check is not a retained config field or feature.

### 9.3 UniProt physicochemical property preset

The current `configs/datasets/uniprot_protparams.yaml` selects many target
columns through `multi_target`.

Convert this to an ordinary single-property preset:

- Remove the `multi_target` block.
- Set `steps.prepare.label_column: length` as an explicit, documented
  initial scalar target.
- Preserve regression metrics/selection settings unless a separate
  scientific change is requested.
- Document overrides for other individual property columns.
- Require/document distinct `workspace.run_id` values when changing the
  target so outputs cannot be confused or overwritten.
- Ensure input and baseline identities include the selected label column.

Do not remove the underlying sourced property columns or the dataset itself.
Do not implement an internal shell loop, sweep runner, or renamed batch
target feature as part of this refactor.

### 9.4 Tests and documentation

- Remove `tests/evaluate/test_multi_target.py`.
- Remove multi-target-only cases/imports from `test_baselines.py`.
- Replace config-loader preservation tests with rejection/null-compatibility
  tests.
- Remove active multi-target usage from the dataset catalog and examples.
- Retain tests for ordinary scalar regression and selected label columns.
- Preserve multilabel classification, layerwise evaluation, external test
  sets, multiple named subsets, and ProteinGym multi-assay evaluation.
  Those are not the removed multi-target feature.
- Leave historical multi-target output directories untouched.

## 10. Scientific and execution invariants

Treat these as acceptance requirements, not optional cleanup.

### Ordinary evaluation

- Preserve sequence classification, multilabel classification, scalar
  regression, token classification, pseudo-perplexity, and calibration.
- Preserve configured class vocabularies, including absent test classes.
- Preserve token ignore-index behavior and original residue positions.
- Preserve the distinction between validation groups and test metric groups.
- Preserve metric averaging, undefined-metric behavior, and probability
  requirements.
- Preserve supported frozen-probe and finetuning behavior.

### Fitting and selection

- Candidate selection never reads held-out test targets or test metrics.
- Feature standardization for selection uses training data only.
- Final refit recomputes fitted preprocessing on the permitted refit inputs.
- Additional final-refit splits remain excluded from selection.
- Preserve head-width/depth/batch-size search semantics.
- Preserve seed-variance reporting independently of the selected final fit.
- Preserve selected checkpoint restoration and selection metric direction.

### Controls

- Independent random-trunk predictors are fitted independently.
- Cached control features use the control trunk's identity.
- Inference-only ablations do not trigger fitting.
- Label scrambling uses the task's valid permutation/scoring units.
- Unsupported combinations still fail early.
- Controls and primary observations remain aligned by stable IDs.

### Data/cache identity

- Cache identities reflect weights, tokenizer/preprocessing, layer/pooling,
  dtype/precision, cropping, and source inputs as currently required.
- Cache sharing across compatible probe variants remains possible.
- Changing source labels/preprocessing invalidates relevant baseline reuse.
- Test subset choice does not invalidate development-only fitting features.
- Test-specific fingerprints still distinguish evaluation outputs.
- Cache-off and cache-budget fallback behavior remain supported.

### Contact prediction

- Preserve schema/protocol/revision validation and chain completeness.
- Preserve alignment across source residues, cleaned sequence, and tokens.
- Preserve the fixed development/test separation.
- Preserve the bilinear protocol's pair weighting, optimizer,
  regularization, patience, checkpoint selection, and failure on reaching an
  epoch safety limit before the protocol finishes.
- Preserve the attention protocol's feature transformation and fixed fit.
- Preserve zero-shot Jacobian semantics.
- Do not introduce flat-row bootstrap for a structured contact protocol
  whose resampling units differ.
- Preserve fit-once reuse across CASP test subsets.

### ProteinGym

- Preserve DMS versus clinical target interpretation.
- Preserve clinical prediction/label direction.
- Preserve substitution versus indel dispatch.
- Preserve mutation-coordinate/tokenization checks.
- Preserve the current deterministic long-sequence/cropping behavior.
- Official folds must actually exist; do not replace missing official folds
  with generated random folds.
- Preserve the applicable scheme set for substitutions versus indels.
- Preserve inner selection boundaries and per-fold independent fitting.
- Preserve official-average reporting.
- Preserve protein/function/gene/pooled aggregation distinctions.
- Preserve random and BLOSUM62 baseline definitions and unsupported cases.
- Preserve uncertainty methods and effective seed behavior.
- Keep single-process execution until separately designed and validated.

### Distributed/runtime behavior

- Configure trainable parameters before creating wrappers whose gradient
  state depends on those parameters.
- Prepare each model/loader only as required; avoid double wrapping.
- Keep collective operations on all required ranks.
- Publish files only from the appropriate process.
- Communicate rank-zero reservation failures before other ranks continue.
- Keep task-specific restrictions on packing/token budgets/process counts.
- Always finish tracking resources on both success and failure.
- Do not mask an original exception with an artifact-cleanup exception.

## 11. Artifact compatibility

Renaming the Python package to `artifacts/` does not rename existing on-disk
directories or change file formats.

Preserve, where currently applicable:

- benchmark/task/dataset/model/run/seed identity;
- the fitted checkpoint directory;
- named `tests/<name>/` output locations;
- control checkpoint/prediction locations;
- `predictions.parquet`;
- `scores.parquet`;
- `run_manifest.json`;
- contact unit-score and summary reports;
- ProteinGym legacy direct-path outputs;
- ProteinGym workspace-backed consolidated outputs.

Check compatibility with existing plotting/analysis consumers before
changing columns or metadata keys. If a necessary schema change is found,
make it explicit and versioned rather than silently changing the format.

Move legacy artifact-upgrade code only when still needed by supported
read/reuse paths. Do not delete compatibility code just because its file is
being reorganized.

## 12. Migration map

| Current location | Target owner |
| --- | --- |
| `contracts/config.py` | `config/shared.py`, `config/evaluation.py` |
| `contracts/config_loader.py` | `config/load.py` |
| `contracts/validation.py` | `config/validation.py` plus domain validation |
| `contracts/artifacts.py` | `artifacts/results.py`, `dataset/types.py`, control records |
| `runners/pipeline.py` | `runners/evaluate.py`, `flat.py`, `contact.py` |
| `runners/baselines.py` | `runners/baselines.py`, `artifacts/baselines.py`, reports |
| `runners/multi_target.py` | Delete |
| `steps/prepare.py` | Dataset preparation, model factory, feature/cache services |
| `steps/tune.py` | `trainers/fit.py`, `artifacts/checkpoints.py` |
| `steps/controls.py` | `controls/`, model control construction |
| `steps/predict.py` | Workflow coordination, model inference, artifact prediction writers |
| `steps/score.py` | Pure scoring plus `artifacts/reports.py` |
| `dataset/workspace.py` | `artifacts/workspace.py` |
| `dataset/embedding_cache.py` | `artifacts/cache.py` |
| Loading/tokenization in `dataset/dataloader.py` | `dataset/loading.py`, `tokenization.py` |
| Classical construction/inference in `model/heads.py` | `model/classical.py` |
| Search logic in `model/heads.py` | `trainers/selection.py` |
| `trainers/selection_helpers.py` | Selection/refit/optimization owners |
| Task model-loading methods | `model/factory.py` |
| Task fitting methods | Fitting services or explicit domain workflow |
| Contact task feature identity/extraction | `dataset/contact_features.py` |
| Contact task preparation/alignment | `dataset/contact.py`, contact workflow |
| `run_proteingym_supervised.py` | `proteingym/supervised.py`; remove script |
| `run_proteingym_zero_shot.py` | `proteingym/zero_shot.py`; remove script |
| `proteingym/workspace.py` | Shared artifacts/runtime plus `proteingym/data.py` |
| `utils/io.py` | Artifact files/manifests/predictions/reports |
| `utils/storage.py` | `artifacts/files.py` |

## 13. Implementation phases and exit criteria

Implement in reviewable increments. Each phase should leave the supported
execution paths working. Temporary import shims may be used for internal
migration, but must be deleted before the final phase.

Do not preserve extra executable wrappers: route the ProteinGym functions
through the single CLI before removing their old scripts.

### Phase 0: Characterize existing behavior

Tasks:

- Inventory public imports, CLI examples, configuration presets, test
  fixtures, and output consumers.
- Record representative existing outputs using tiny local test models/data.
- Identify which tests assert scientific behavior versus old class/mocking
  structure.
- Establish current test outcomes before attributing failures to the
  refactor.
- Confirm actual checked-in ProteinGym config presets and launch paths.

Exit criteria:

- A representative behavior/output matrix exists in tests or test fixtures.
- Known environmental limitations are recorded without changing protocols.
- No production behavior has changed.

### Phase 1: Remove multi-target support

Depends on Phase 0.

Tasks:

- Apply section 9 to ordinary and baseline execution.
- Migrate the UniProt property overlay to one explicit scalar target.
- Add clear rejection of non-null removed options.
- Preserve explicit-null config compatibility if selected.
- Remove obsolete target aggregation and resume code.

Exit criteria:

- Scalar regression and scalar baselines still work.
- Removed multi-target configs fail before outputs are reserved.
- No active source/config/docs path advertises multi-target execution.
- Multilabel, layerwise, subset, external-test, and multi-assay behavior is
  unchanged.

### Phase 2: Establish `artifacts/` and result ownership

Depends on Phase 0; can be developed independently of Phase 1 where files
do not conflict.

Tasks:

- Move workspace/cache/file primitives with minimal behavioral edits.
- Split manifest, prediction, checkpoint, report, and baseline publication.
- Extract pure result types to `artifacts/results.py`.
- Keep explicit all-rank versus writer-only serialization contracts.
- Update targeted imports and artifact tests.

Exit criteria:

- Workspace paths, locks, overwrite rules, identities, and formats are
  unchanged.
- Scoring can run without creating files.
- Fitting can return a result without saving a checkpoint.
- No new `storage/` package is introduced.

### Phase 3: Separate configuration and runtime

Depends on Phase 2 for the shared artifact lifecycle integration.

Tasks:

- Extract config components without changing ordinary overlay precedence.
- Add the workflow discriminator and application-level loader.
- Preserve baseline config projection and fail on removed options.
- Add `runners/runtime.py`; remove import-time application logging setup.
- Centralize precision/seeding/tracking setup without changing effective
  values.
- Keep unsupported process-count checks before model loading.

Exit criteria:

- Existing ordinary YAML stacks load equivalently.
- ProteinGym configs choose their own schema without flat-task insertion.
- Unknown workflows/flags fail clearly.
- Tracking closes on exceptions.
- Baseline-only execution does not load model resources.

### Phase 4: Introduce explicit data contracts

Depends on Phase 2.

Tasks:

- Add training/test and token/feature records.
- Split loading/tokenization from loader construction.
- Extract training/test preparation from `PrepareStep`.
- Preserve alignment/group/provenance and retained tokens for controls.
- Prepare external/subset data without copying a model-and-training payload.

Exit criteria:

- Data preparation returns no fitted model or fitted control.
- Ordinary raw-token and cached-feature cases work.
- Evaluation-only tasks do not need fake train inputs.
- External/subset preparation cannot alter training inputs or fitted state.

### Phase 5: Narrow tasks and extract model construction

Depends on Phases 3 and 4.

Tasks:

- Introduce the smaller `FlatTask` interface.
- Move Hugging Face and predictor construction to `model/factory.py`.
- Preserve task vocabulary and label/prediction/scoring semantics.
- Move contact protocol functions out of the broad task hierarchy.
- Keep contact feature identities/extraction separate from cache I/O.

Exit criteria:

- Tasks contain no workflow, checkpoint-publication, or independent fitting
  service.
- Flat task types remain selectable with the existing names.
- Native loss/head selection remains equivalent across supported families.
- Contact no longer inherits unused flat behavior.

### Phase 6: Extract fitting and control services

Depends on Phases 4 and 5.

Tasks:

- Make `fit_predictor()` consume explicit inputs and return `FitResult`.
- Split selection, refit, and optimizer mechanics.
- Retain stateful trainer classes only where they own a meaningful lifetime.
- Move independent control fitting and inference into their own functions.
- Preserve efficient shared-feature layerwise inference.
- Remove calls from one former stage into another former stage.

Exit criteria:

- Fitting does not mutate an all-purpose prepared payload.
- Search/refit/standardization/seed metadata remain correct.
- Independent controls fit once and are reused.
- Unsupported control/backend combinations still reject early.

### Phase 7: Replace ordinary step orchestration

Depends on Phases 3 through 6.

Tasks:

- Implement `runners/evaluate.py`, `flat.py`, `contact.py`, `baselines.py`.
- Call preparation, fitting, prediction, scoring, and publication functions.
- Preserve output reservation and distributed failure propagation.
- Preserve fit-once subset/external test evaluation.
- Remove old ordinary runner and step call sites.

Exit criteria:

- Ordinary tasks, contact predictors, baseline-only execution, layerwise
  probing, subsets, and external tests all use the function-based workflows.
- Tests no longer need to monkeypatch stage constructors.
- The old shared payload is no longer required.

### Phase 8: Integrate ProteinGym into the single CLI

Depends on Phases 2 and 3. Reuse Phase 6 primitives only after equivalence is
demonstrated; do not make benchmark migration depend on rewriting every
trainer.

Tasks:

- Define explicit `AssayResult` beside the assay loop in `proteingym/runner.py`.
- Extract suite/assay resolution to `proteingym/data.py`.
- Move script orchestration into `run_supervised()` and `run_zero_shot()`.
- Use shared runtime/model/artifact facilities.
- Keep the assay loop, folds, mutation algorithms, and aggregation local.
- Route both workflows from `run_evaluate.py`.
- Update all shipped presets and launcher/documentation examples.
- Delete both `run_proteingym_*.py` scripts.
- Remove `proteingym/workspace.py` after its responsibilities are migrated.

Exit criteria:

- Both ProteinGym modes run through `run_evaluate.py`.
- The package owns no CLI or duplicate application runtime.
- All supported tracks and split schemes preserve scientific results.
- Shared extraction/fitting is used only where behavior matches.
- No extra workflow forwarding module exists solely to call a ProteinGym
  run function.

### Phase 9: Remove obsolete modules and update documentation

Depends on all applicable previous phases.

Tasks:

- Delete temporary internal import shims.
- Remove `steps/`, `contracts/`, and legacy pipeline/multi-target runners after
  callers are migrated; keep functional orchestration in `runners/`.
- Remove obsolete I/O/storage helper modules.
- Update README, controls/dataset/ProteinGym guides, and launch examples.
- Update test imports and architecture-oriented fixtures.
- Explain the new file ownership and how to add a task/predictor/benchmark.

Exit criteria:

- Only `run_evaluate.py` is an executable source entry point.
- No production import references deleted packages.
- No multi-target implementation remains.
- No misleading old CLI commands remain in active documentation.
- The source tree follows the target responsibilities.

## 14. Validation strategy

Use existing tools and tests. Do not add a new lint/test framework for this
refactor. Run the smallest relevant groups during each phase, then the
evaluation suite at integration milestones.

### 14.1 Existing test groups to preserve and adapt

| Area | Existing tests |
| --- | --- |
| Config and composition | `test_config.py`, `test_config_loader.py` |
| Tasks and offline pipeline | `test_tasks.py`, `test_pipeline_tasks.py` |
| Data and batching | `test_dataloader.py`, `test_collator.py`, `test_prepare.py` |
| Cached representations | `test_embed.py`, `test_embedding_loaders.py`, `test_storage.py` |
| Fitting/search/refit | `test_tune.py`, `test_refit.py`, `test_probe_standardization.py`, `test_mlp_width_search.py`, `test_mlp_batch_search.py` |
| Inference/controls | `test_predict.py`, `test_controls.py` |
| Metrics/publication | `test_metrics.py`, `test_score.py`, `test_io.py` |
| Subsets/external tests | `test_subsets.py`, `test_pipeline_external.py` |
| Baselines | `test_baselines.py` |
| Workspace/runtime precision | `test_workspace.py`, `test_precision.py` |
| Calibration | `test_calibration.py` |
| Contact | `test_contact_data.py`, `test_contact_prediction_task.py`, `test_contact_embedding.py`, `test_contact_attention.py`, `test_contact_jacobian.py`, `test_casp_contact_metrics.py` |
| ProteinGym | `test_proteingym.py`, `test_proteingym_workspace.py` |

Move or rename test files only when it improves ownership. Do not rewrite
all tests simply to match new filenames.

### 14.2 Focused new/changed behavior tests

Add targeted cases for:

- One CLI dispatching ordinary, contact, ProteinGym supervised, and
  ProteinGym zero-shot configurations.
- Config overlay precedence and workflow selection.
- Missing/unknown workflow errors and invalid flag combinations.
- Non-null multi-target rejection before output creation.
- Optional explicit-null compatibility.
- Single-property UniProt config and label-column identity.
- Independent test preparation preserving fitted predictor/control identity.
- Fit-once behavior across multiple named/external tests.
- Pure scoring producing no filesystem side effects.
- Tracking/resource cleanup on failure.
- Explicit assay metric overrides preserving official-average scores.
- Shared runtime preserving precision, seeding, and process restrictions.
- Importing library modules not configuring application logging or starting
  execution.

Prefer behavior assertions to tests that merely assert a new class/function
exists. Reuse the current tiny local model/data fixtures rather than
downloading real checkpoints for normal unit tests.

### 14.3 Commands

Examples from the repository root:

```bash
# Configuration/CLI/removal work.
uv run pytest \
  tests/evaluate/test_config.py \
  tests/evaluate/test_config_loader.py \
  tests/evaluate/test_baselines.py

# Data, controls, and workflow extraction.
uv run pytest \
  tests/evaluate/test_pipeline_tasks.py \
  tests/evaluate/test_pipeline_external.py \
  tests/evaluate/test_subsets.py \
  tests/evaluate/test_controls.py \
  tests/evaluate/test_refit.py

# ProteinGym integration.
uv run pytest \
  tests/evaluate/test_proteingym.py \
  tests/evaluate/test_proteingym_workspace.py \
  tests/evaluate/test_precision.py

# Evaluation integration milestone.
uv run pytest tests/evaluate

# Existing repository checks on changed Python/config files.
uv run pre-commit run --files <changed-files>
```

The repository hooks include Ruff checks/import sorting, Ruff formatting,
and mypy plus generic file checks. Fix issues caused by the refactor; do not
expand the work into unrelated repository cleanup.

Use the existing environment first. Restore/install dependencies only if
the selected command reports missing tools/packages or dependency manifests
have actually changed.

### 14.4 Numerical parity

For representative cases, compare pre/post:

- prediction IDs, shapes, ordering, and metadata;
- primary and control predictions;
- selected settings and final-fit split metadata;
- metric values and uncertainty summaries;
- contact unit/hierarchy scores;
- ProteinGym per-assay and aggregate metrics;
- output path identities and report schemas.

Use exact comparisons for structural contracts and deterministic integer
results. Use justified floating-point tolerances for numerical comparisons;
do not hide a protocol change behind a loose tolerance.

GPU/packed/distributed smoke checks should be run on suitable hardware when
available. State clearly which hardware-dependent paths were not exercised.
Do not claim CPU tests establish distributed correctness.

## 15. Final verification checklist

Checked items describe implementation and local integration validation.
Hardware-dependent CUDA checks remain skipped; distributed/GPU parity is not
certified.

- [x] `run_evaluate.py` is the sole source entry point.
- [x] No executable ProteinGym wrapper scripts remain.
- [x] Ordinary and ProteinGym routes share application setup.
- [x] ProteinGym scientific rules remain cohesive in `proteingym/`.
- [x] Filesystem/cache/output ownership lives in `artifacts/`.
- [x] No new `storage/` package exists.
- [x] `steps/` and `contracts/` are removed; `runners/` contains functional
  orchestration rather than stage classes.
- [x] There is no multi-target runner, config model, baseline path, or summary
  aggregator.
- [x] Removed multi-target configs fail clearly rather than being ignored.
- [x] The property dataset preset is single-target.
- [x] Multilabel, layerwise, subsets, external tests, and assay collections
  remain supported.
- [x] No universal mutable prepared payload carries data, model, and controls.
- [x] Task handlers own semantics rather than execution infrastructure.
- [x] Fitting returns explicit fit results.
- [x] Scoring works without a workspace or output writes.
- [x] Checkpoint/publication collective requirements are explicit.
- [x] Existing scientific protocols and report identities are retained.
- [x] Library imports have no application setup side effects.
- [x] Examples and presets use the single entry point.
- [x] Source formatting, linting, typing and existing generic hooks pass.
- [x] Legacy tests are migrated and the complete local evaluation suite passes.

## 16. Guidance for keeping this refactor simple

- Extract working implementations before redesigning their interfaces.
- Separate behavioral changes from mechanical file moves where possible.
- Do not add a shared helper until at least two concrete callers have
  genuinely matching semantics.
- Do not make a new abstract base class for a finite set of explicit branches.
- Keep related scientific rules together even when some functions could be
  placed in generically named files.
- Keep configuration compatibility where it is cheap and intentional; do not
  preserve removed execution features.
- Prefer an honest domain-specific function over a misleading generic one.
- Avoid replacing stage classes with equally unnecessary service classes.
- Do not use the refactor to change benchmark methodology.
- Finish removing transitional layers rather than leaving both architectures
  permanently active.
