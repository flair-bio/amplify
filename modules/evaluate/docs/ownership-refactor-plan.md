# Evaluation ownership refactor plan

## Purpose and constraints

This is the follow-up to [the first refactor plan](refactor-plan.md). It defines
the agreed target ownership, migration sequence, and acceptance criteria.
The structure below records the implemented ownership boundaries.

The first refactor extracted services and preserved behavior, but substantial
evaluation behavior remained in `utils/`, contact extraction had a separate
`features/` package, and trainers received callbacks into `TuneStep`.
This pass corrects dependency direction rather than merely renaming directories.

## Implementation status

The scientific-coordination follow-up is complete. Tasks now own scientific
preparation requirements, fitting protocols (including calls to pure trainers),
observation construction, independent-control specifications and scoring policy.
Steps own cache access, workspace identity checks, distributed lifecycle and
publication. Tasks must not receive workspace objects, publish files or manage
cache locations. This supersedes earlier restrictions on tasks invoking fitting.

The follow-up removes exported contact feature preparation and contact-specific
stage methods. Feature requests describe extraction identity, shape and dtype;
Prepare materializes them before Tune. Flat tasks share their ordinary fitting
implementation; structured tasks implement explicit preparation, prediction and
scoring contracts rather than optional lifecycle hooks. No new registry or
per-task forwarding classes are introduced.

Scoring now has a common task entry point for flat and structured predictions.
Tasks return in-memory results and select supported uncertainty protocols;
`metrics/scoring.py` owns flat scoring, bootstrap and paired layer comparisons.
`ScoreStep` owns loading and publication, dispatching only on result shape rather
than task type. Control arrays are prepared once for point scores and bootstrap,
preserving the seeded label-permutation/resampling stream.

The controls/scoring follow-up passed 1,551 evaluation tests with one skip.
Configured hooks and standalone mypy passed for 90 runtime/sourcing files.

Contact protocol guards now share predictor capability checks. Fixed bilinear
settings have one immutable definition in the trainer, used by fitting and the
compatibility check rather than duplicated numerical checks in the task.
The trainer accepts only the epoch safety limit as a fitting option and records
the effective settings in `contact_training.json`; incompatible generic
overrides remain explicit errors.

Earlier scientific-coordination verification: 1,547 evaluation tests passed with one skip. Configured
hooks passed, standalone mypy checked 89 runtime/sourcing files, and dependency
checks found no stage/runner imports from tasks, trainers, models, datasets or
contracts. All 12 scratch command configurations and Bash syntax were validated.
ProteinGym remains unchanged. No commits or pull requests were made.

The preceding ownership pass was implemented without commits or pull requests.
`steps/contact.py` was removed: preparation/cache coordination is in
`steps/prepare.py`, fitting/publication in `steps/tune.py`, and independent
contact controls in `steps/controls.py`. Obsolete feature/schema packages,
utility implementations, the forwarding-only control trainer, and unreachable
separation-prior code are removed. ProteinGym evaluation is unchanged.

Verification: the complete evaluation suite passed with 1,545 tests and one
skip. The final contact/pipeline check passed 42 tests. Configured lint,
formatting, whitespace, YAML and mypy hooks passed; standalone mypy checked
90 runtime/sourcing files. Dependency checks found no stage/runner imports
from trainers, tasks, contracts, models or datasets.

The scratch verification commands were updated with standalone baseline runs;
all 12 command configurations and Bash syntax were validated without launching
full benchmark workloads. Full local checkpoint/data evaluations remain the
user's workflow verification.

## Constraints

- Work only on `oseaj/eval-update-contact-prediction-refactor`.
- Preserve existing uncommitted changes. Do not commit or open pull requests.
- Keep only executable `run_*` modules and package initialization at `src/` root.
- Retain model lifecycle stages in `steps/`; model-free baselines have a separate runner.
- Do not add top-level `features/`, `representations/`, `controls/`, or
  `baselines/` packages.
- Keep task-specific scientific behavior under its appropriate owner without
  creating a privileged top-level contact package.
- Leave ProteinGym evaluation implementations, protocols, and commands unchanged.
  Preserve shared APIs they consume through narrow compatibility exports only
  where required.
- Keep sourcing entry points under `scripts/dataset_sourcing/` and reusable
  sourcing code in its `shared/` package. Never restore sourcing under `src/`.
- Do not restore evaluation cross-validation.
- Preserve YAML keys, overlay precedence, artifact paths/formats, alignment,
  split isolation, seed streams, predictor presets, and distributed behavior.
- Keep comments succinct. Avoid new registries, frameworks, forwarding layers,
  one-class-per-file conventions, or unrelated cleanup.

## Target structure and file responsibilities

Package initializers are omitted. Existing unaffected files remain in place.
Split files by substantial ownership boundaries, not to meet a file-count goal.

```text
modules/evaluate/
├── configs/
├── docs/
├── scripts/
│   └── dataset_sourcing/
│       ├── source_*.py
│       ├── run_hhblits.py
│       └── shared/
│           ├── config.py
│           ├── downloads.py
│           ├── hf_io.py
│           ├── outputs.py
│           ├── pipeline.py
│           ├── preprocess.py
│           ├── spec.py
│           └── stats.py
└── src/
    ├── run_evaluate.py
    ├── run_proteingym_supervised.py
    ├── run_proteingym_zero_shot.py
    ├── contracts/
    │   ├── config.py
    │   ├── config_loader.py
    │   ├── validation.py
    │   └── artifacts.py
    ├── runners/
    │   ├── baselines.py
    │   └── pipeline.py
    ├── steps/
    │   ├── prepare.py
    │   ├── tune.py
    │   ├── predict.py
    │   ├── score.py
    │   └── controls.py
    ├── tasks/
    │   ├── base.py
    │   ├── registry.py
    │   ├── sequence_classification.py
    │   ├── sequence_multilabel_classification.py
    │   ├── sequence_regression.py
    │   ├── token_classification.py
    │   ├── pseudo_perplexity.py
    │   ├── calibration.py
    │   └── contact_prediction.py
    ├── dataset/
    │   ├── workspace.py
    │   ├── dataloader.py
    │   ├── collator.py
    │   ├── contact.py
    │   ├── embeddings.py
    │   ├── embedding_cache.py
    │   └── baseline_data.py
    ├── model/
    │   ├── baselines.py
    │   ├── heads.py
    │   ├── embeddings.py
    │   ├── inference.py
    │   ├── controls.py
    │   ├── layerwise.py
    │   ├── contact_model.py
    │   ├── contact_attention.py
    │   ├── contact_jacobian.py
    │   └── adapters.py
    ├── trainers/
    │   ├── selection.py
    │   ├── selection_helpers.py
    │   ├── torch.py
    │   ├── classical.py
    │   ├── layerwise.py
    │   └── contact.py
    ├── metrics/
    │   ├── metrics.py
    │   ├── contact_metrics.py
    │   ├── bootstrap.py
    │   ├── comparisons.py
    │   ├── scoring.py
    │   └── tables.py
    ├── utils/
    │   ├── storage.py
    │   ├── io.py
    │   ├── seed.py
    │   ├── tracking.py
    │   ├── amino_acids.py
    │   └── embed.py                  # ProteinGym pooling compatibility only
    └── proteingym/                   # Unchanged
```

### Contracts

`contracts/` groups configuration definitions/loading and typed stage boundaries.
It is not a general service package.

| File | Owns | Principal types/functions |
| --- | --- | --- |
| `config.py` | Evaluation/stage configuration models, defaults, generic cross-stage checks | Existing `EvaluationConfig`, `StepsConfig`, stage configs, `BaselinesConfig`, `EmbeddingCacheConfig`; generic config validation |
| `config_loader.py` | Evaluation-specific overlay composition and parsing | Existing `compose_evaluation_config`, `load_evaluation_config` |
| `validation.py` | Generic stage/head compatibility checks | Reads task capabilities; scientific protocols stay under tasks |
| `artifacts.py` | Typed in-memory handoffs, metadata, fitting results, evaluation result records | Prepared payloads, `PredictOutput`, score outputs, fitting result types, evaluation identity |

Reuse `modules.core.utils.config_loader`; do not duplicate YAML loading.
Configuration models must not import steps or trainer implementations. Retain
generic validation beside models unless its actual size justifies another file.
Task-specific validation runs explicitly after parsing through task lookup.
Keep metadata in `artifacts.py` unless it becomes independently substantial.

### Runner and stages

| File | Owns | Principal API |
| --- | --- | --- |
| `runners/pipeline.py` | Preflight, run reservation, execution order, external evaluation, final result assembly | `run_pipeline` |
| `steps/prepare.py` | Materialize task inputs and declared feature requests, cache policy, external inputs | `PrepareStep.run`, `prepare_external_test`, `prepare_features` |
| `steps/tune.py` | Invoke task fitting and publish the returned model/settings | `TuneStep.run` |
| `steps/predict.py` | Coordinate primary/control inference, align observations, publish predictions | `PredictStep.run`, `persist_predictions` |
| `steps/score.py` | Reload or consume predictions, invoke scorers/comparisons, publish results | `ScoreStep.run`, prediction reload |
| `runners/baselines.py` | Standalone train-only baseline orchestration and publication | `run_baselines`; calculations belong to `model/baselines.py`, loading to `dataset/baseline_data.py` |
| `steps/controls.py` | Materialize/fit independent cached, layerwise and task controls; publish outputs | `ControlsStep.run`; `ControlBuilder` remains an inference helper used by Predict |

Preserve useful existing APIs rather than adding wrappers to rename them.
Steps may use small private helpers. They must not own optimizer loops,
hyperparameter-search algorithms, task semantics, or nested evaluation runs.
Baseline algorithms are small enough to live with their stage initially;
artifact identity/loading/alignment belongs in the dataset package.

Do not introduce a contact-specific stage module. The steps consume explicit
flat/structured task contracts, not concrete contact handlers. Contact input
construction, trainer choice, observation construction and CASP restrictions
belong to `ContactPredictionTask`. Cache access and publication stay in steps.

### Task semantics

| File | Owns |
| --- | --- |
| `tasks/base.py` | Explicit semantic contract: labels, alignment, prediction interpretation, metric/scorer capabilities, supported controls, protocol validation |
| `tasks/registry.py` | Existing task registration and lookup |
| `sequence_classification.py` | Scalar categorical targets, class probabilities, classification scoring policy |
| `sequence_multilabel_classification.py` | Binary indicator vectors, sigmoid interpretation, multilabel scoring policy |
| `sequence_regression.py` | Continuous targets, scalar predictions, regression scoring policy |
| `token_classification.py` | Residue alignment, ignored labels, token prediction/scoring semantics |
| `pseudo_perplexity.py` | Masked-residue example expansion and log-probability interpretation |
| `calibration.py` | Profile alignment, masking/window semantics, calibration interpretation |
| `contact_prediction.py` | Fixed CASP/predictor constraints, structured prediction representation, hierarchical scoring policy |

Preserve existing semantic methods such as label alignment, example expansion,
prediction extraction, and metric calculation where appropriate. Tasks declare
supported controls and own task-specific label/input transformations.

Tasks may coordinate scientific computations and call pure trainers, models and
metrics. They do not receive workspace objects, coordinate caches, publish files,
or manage distributed lifecycle. Avoid reproducing generic random-model construction in every task.
Replace lifecycle hooks returning `None` to request a generic fallback with
explicitly selected implementations. A task may legitimately require structured
scoring rather than flat metrics; callers must honor that capability.

### Dataset and cached inputs

| File | Owns | Principal API |
| --- | --- | --- |
| `workspace.py` | Identity, paths, output reservation | Existing workspace types and reservation methods |
| `dataloader.py` | Dataset loading, source/split resolution, tokenization and loaders | Evaluation/external resolution functions and split loaders |
| `dataloader.py` | Generic tokenization and dataloader construction | `tokenize_dataset`, `build_dataloaders` |
| `collator.py` | Padding, packing, batch assembly | Existing collators |
| `contact.py` | Contact records/datasets, fixed-split loading, validation, original-coordinate alignment and candidate pairs | Existing contact dataset types, preparation/alignment/candidate functions |
| `embeddings.py` | Feature datasets, collation, loader construction, retained token-loader context | `EmbeddingDataset`, `EmbeddingBackend`, `find_embedding_dataset`, `embedding_collate_fn`, `build_embedding_dataloaders` |
| `embedding_cache.py` | Explicit representation identity, size policy, cache validation/read/write | `EmbeddingCache`, cache keys, weight fingerprints, byte estimation |
| `baseline_data.py` | Reusable baseline identities, manifests, content validation and observation alignment | Baseline context/identity/path helpers, load and align functions |

The cache receives explicit scientific/extraction identity, not a concrete task
handler. Storage primitives do not know what an embedding or contact is.
Move data definitions out of `tasks/contact_prediction.py` before making model
and trainer dependencies point at them, preventing task/model import cycles.

### Models and model execution

| File | Owns | Principal API |
| --- | --- | --- |
| `heads.py` | Torch head architectures, estimator construction and serialization | Existing head/model builders |
| `embeddings.py` | Generic hidden-state extraction, residue pooling, layer selection, extraction precision | `extract_embeddings`, pooling/layer functions |
| `inference.py` | Batched model execution, collective gathering, row assembly using task semantics | Existing live/cached inference functions and gathering helpers |
| `controls.py` | Shared random-trunk/head construction and model ablation mechanics | Existing random model/head builders |
| `layerwise.py` | Layerwise model, efficient multi-depth inference and control feature reuse | Existing layerwise model and substantive inference functions |
| `contact_model.py` | Bilinear architecture, full-chain residue extraction and model serialization | Existing contact predictor |
| `contact_attention.py` | Attention extraction/transformation, logistic prediction and serialization | Existing attention predictor |
| `contact_jacobian.py` | Categorical mutation effects and contact-score prediction | Existing Jacobian predictor |
| `adapters.py` | Backbone position/precision compatibility | Existing positional context and MLM composition |

Model extraction returns tensors or prediction rows. It must not choose cache
paths or publish workspace outputs. Generic inference may consume task semantic
interfaces but must not import/check concrete contact handlers.

### Fitting

| File | Owns | Principal API |
| --- | --- | --- |
| `selection.py` | Shared search, best-candidate selection, optional refit and canonical seed selection | A substantive fitting entry point, e.g. `fit_predictor`, returning typed results |
| `selection_helpers.py` | Optimizer/scheduler construction, checkpoints and search mechanics | Shared fitting helpers, no stage callbacks |
| `torch.py` | Torch training, validation, early stopping, selected state | Existing Torch trainer |
| `classical.py` | Estimator fitting/sweeps, train-only preprocessing, final refit | Existing classical trainer |
| `layerwise.py` | Independent layer-head optimization and stopped-head synchronization | Existing layerwise trainer |
| `contact.py` | Fixed bilinear and attention fitting protocols | Existing contact fitting functions |

Primary and control fits use the same fitting services. Remove callback
dependencies from trainers to private `TuneStep` methods. Trainers return fitted
models, selected settings/history and seed/feature metadata; steps serialize
those outputs. Trainer state selection is distinct from workspace publication.
Control preparation invokes these services directly from `steps/controls.py`;
there is no forwarding-only control trainer.

### Scoring mathematics

| File | Owns |
| --- | --- |
| `metrics.py` | Existing ordinary task metric calculations |
| `contact_metrics.py` | Pair/unit metrics and equal-weight hierarchical aggregation |
| `bootstrap.py` | Resampling and confidence intervals |
| `comparisons.py` | Paired controls, permutation statistics and cross-layer comparisons |
| `tables.py` | Tidy/long result-table construction |

These modules return values/tables. They do not write stage manifests or choose
output paths. `ScoreStep` owns persistence.

### Small utilities

| File | Owns |
| --- | --- |
| `storage.py` | Atomic publication and bounded filesystem exclusion |
| `io.py` | Serialization and file logging |
| `seed.py` | RNG initialization and deterministic seed derivation |
| `tracking.py` | Optional experiment tracking |
| `amino_acids.py` | Shared alphabet and token mapping primitives |

No concrete task imports, feature extraction engines, baseline estimators,
control orchestration, training, or scientific protocol decisions belong here.
Move evaluation-specific prediction downcasting/manifest policy out of generic
I/O when separating its existing contents.

## Interactions and execution contract

```text
run_evaluate.main
  -> contracts.config_loader
       -> core YAML loader
       -> config models / generic validation
  -> task protocol validation
  -> runners.pipeline.run_pipeline
       -> preflight and output reservation
       -> PrepareStep
            -> dataset resolution/interpretation
            -> model construction/extraction
            -> dataset embedding cache/loaders
       -> TuneStep, only when primary fitting is required
            -> selection and fitting services
            -> publish fitted model/settings
       -> ControlsStep.run
            -> task support/transformations
            -> model ablation construction
            -> fitting services only when required
       -> PredictStep
            -> model inference
            -> task prediction interpretation
            -> aligned primary/control artifacts
       -> ScoreStep
            -> task scoring policy
            -> metric/aggregation/comparison functions
            -> publish score artifacts
       -> evaluation result records
```

Control preparation is an explicit lifecycle operation shared by tuned and
zero-shot evaluations, not a hidden second `TuneStep.run`. Resolve its concrete
implementation once using existing task/predictor capabilities. Do not add a
generic workflow/plugin registry or make every task implement empty methods.
Reuse prepared feature inputs where possible; model/control objects and seeds
must remain independent.

```text
run_evaluate --baselines
  -> Baseline stage
       -> dataset resolution
       -> fit train-only statistics
       -> baseline artifacts

model scoring
  -> load/validate reusable baseline artifacts
  -> align observations
  -> metric/comparison functions
```

Dependency rules:

1. Runners call stages; stages invoke data/model/training/scoring services.
2. Services do not call stages or runners, including through injected callbacks.
3. Contracts do not import concrete stage/trainer implementations.
4. Tasks coordinate scientific fitting/inference/scoring; steps own caching,
   distributed lifecycle and publication.
5. Generic infrastructure never imports concrete tasks.
6. Structured versus flat prediction/scoring are explicit supported contracts,
   not an inheritance promise subsequently rejected at runtime.
7. Layerwise inference can be specialized, but its results enter the same
   prediction/scoring handoff as ordinary results.

## Concrete task and TuneStep boundaries

The following sketches illustrate ownership, not final signatures or a mandate
to introduce forwarding helpers. Preserve existing semantic APIs where they fit.

### A task coordinates scientific computations, not infrastructure

A token classification handler exposes behavior such as:

```python
class TokenClassificationTask(FlatMetricsTask):
    name = "token_classification"
    label_dtype = torch.long

    def validate_config(self, config):
        # Reject invalid scientific combinations.
        ...

    def align_labels(self, labels, residue_mapping):
        # Map residue labels to tokens, ignoring special/padded positions.
        ...

    def extract_predictions(self, logits, *, labels):
        return logits.argmax(dim=-1)

    def prediction_rows(self, predictions, labels):
        # Preserve protein boundaries and remove ignored positions.
        ...

    def compute_metrics(self, predictions, labels, metric_names):
        ...

    def scramble_labels(self, labels, rng):
        # Permute valid labels within each protein.
        ...
```

Declare supported controls explicitly using the currently supported set; the
sketch does not narrow existing behavior. Generic model/head randomization is
shared model behavior, not reimplemented in each handler.

| Consumer | Task behavior it uses |
| --- | --- |
| Preparation | Label requirements, alignment, example expansion |
| Training | Output interpretation and validation metric semantics |
| Prediction | Observation construction, original coordinates and row boundaries |
| Scoring | Metric/scoring policy and task-specific label scrambling |
| Preflight | Protocol constraints and supported controls |

Contact coordinates data preparation, predictor fitting, structured observations
and hierarchical scoring. Data mechanics live in `dataset/contact.py`,
optimization in trainers, and extraction in models. Fixed CASP/predictor
configuration checks live in `ContactPredictionTask.validate_config`; there is
no separate contact config module.

Do not make flat metric computation mandatory on the common task base.
Flat and structured scorers have explicit contracts appropriate to their inputs;
only tasks supporting flat metrics expose that contract. A boolean capability
alone is insufficient if an unsupported inherited method is still required.
Avoid a base-class method that raises or returns `None` for callers to discover
which execution path to try.

Remove workspace-aware orchestration hooks from task handlers:
`prepare_artifacts`, `tune_artifacts`, `prepare_controls`, and workspace-aware
prediction/publication hooks. Replace their real behavior with explicitly
scientific contracts that do not manage workspace/cache/publication.
Semantic scoring functions may return scores/tables but never publish them.

### TuneStep invokes fitting and publishes its result

```python
class TuneStep:
    def __init__(self, config: TuneConfig):
        self.config = config

    def run(self, workspace, prepared, accelerator, seed) -> PreparedArtifacts:
        # Validate this stage's required inputs.
        result = task.fit(
            prepared=prepared,
            config=self.config,
            inputs=workspace.config,
            accelerator=accelerator,
            seed=seed,
        )
        # Publish model/settings through the existing distributed policy.
        prepared.model = result.model
        return prepared
```

`FitResult` is a typed handoff, not an additional wrapper around
`PreparedArtifacts`. It carries the fitted model and applicable selection,
history, refit, feature and seed metadata without forcing identical history
schemas on unrelated algorithms. The runner explicitly hands the fitted model
to prediction.

The stage invokes the task's fitting contract. The shared ordinary task
implementation uses `trainers/selection.py` for layerwise, classical or Torch
fitting; contact invokes its fixed-protocol trainers using prepared features.
Selection services consume task semantics without recursively calling task fitting.
Resolve the scientific protocol once per fitting request. A few
algorithm-selection branches are acceptable and do not require a plugin registry.

Each fitting implementation returns the fitting contract and owns its applicable
optimization/search/refit behavior. None runs preparation, prediction, scoring,
or a secondary pipeline. Trainers may consume task semantic functions, but never
callbacks into private TuneStep methods. Validation and publication can stay
inline or as substantive stage methods; do not extract trivial forwarding helpers.
Preserve collective participation and model-serialization requirements rather
than assuming every persistence operation can blindly be rank-zero-only.

### Controls remain independent of primary fitting

The runner retains this shared lifecycle:

```text
prepared = PrepareStep.run(...)
if primary fitting is required:
    fitted = TuneStep.run(...)
    hand fitted model to prepared prediction inputs
prepared = ControlsStep.run(...)
predictions = PredictStep.run(...)
scores = ScoreStep.run(...)
```

Control preparation combines task support/transformations, shared model
construction, and fitting services when needed. It executes for zero-shot runs
too. Independently fitted controls call the same trainer services directly,
never `TuneStep.run`, and do not refit the primary predictor.

## Migration map

| Current owner | Destination |
| --- | --- |
| Root `config.py`, stage config classes, generic config validation | `contracts/config.py` |
| `utils/config_loader.py` | `contracts/config_loader.py` |
| `schemas/artifacts.py`, fitting result contracts | `contracts/artifacts.py` |
| Contact data/alignment definitions in task module | `dataset/contact.py` |
| `utils/embed.py` extraction/pooling | `model/embeddings.py` |
| `utils/embed.py` cache/identity | `dataset/embedding_cache.py` |
| `utils/embed.py` datasets/loaders/backend | `dataset/embeddings.py` |
| `features/contact.py` | Split extraction, cache identity, and preparation coordination among their owners |
| `utils/inference.py` | `model/inference.py` |
| `utils/layerwise.py` | Model/inference behavior to `model/layerwise.py`; comparisons to metrics |
| `utils/controls.py` | Task semantics, model ablations, trainer fitting, prediction coordination |
| `utils/hpo.py`, remaining TuneStep search/refit algorithms | `trainers/selection.py` |
| Baseline execution and calculations | `runners/baselines.py` and `model/baselines.py` |
| Baseline identity/loading/alignment from `utils/baselines.py` | `dataset/baseline_data.py` |
| `utils/bootstrap.py`, `utils/long_format.py` | Bootstrap/comparisons and metric tables |
| `utils/stats.py` | Sourcing-only SeqKit statistics in `scripts/dataset_sourcing/shared/stats.py` |
| Former `tasks/contact_config.py` | `ContactPredictionTask.validate_config` |

The separation-prior implementation and prediction branch were removed:
configuration rejected them for every supported task, leaving only legacy tests
as consumers. No new contact baseline was enabled.

## Implementation sequence

### 0. Record baseline and compatibility consumers

- Inventory existing dirty files, stage APIs, ProteinGym imports, direct callers
  and disk reload consumers.
- Capture relevant tests/checks before changing ownership.
- Record the current local verification commands in the scratch command file;
  preserve their YAML and output behavior.

Acceptance: a concrete import/API migration inventory and reproducible baseline.

### 1. Establish contracts without execution dependencies

- Move configuration/loading and artifact contracts to `contracts/`.
- Move stage configuration models without importing stages back into contracts.
- Preserve core YAML loader reuse and task-overlay precedence.
- Run task protocol validation at the existing preflight boundary and preserve
  configuration-loading rejection behavior expected by current consumers.
- Update direct callers atomically; retain compatibility only for actual users.

Acceptance: no contract-to-stage imports or new import cycles; config and disk
round-trip tests pass.

### 2. Separate contact data from task orchestration

- Extract contact data types, fixed-split validation, alignment and pairs to
  `dataset/contact.py`.
- Keep task policy and semantic interpretation in its handler.
- Update models/trainers to import data types from the dataset owner.

Acceptance: original coordinates, missing labels, fixed splits, candidate pairs
and all predictor outputs remain unchanged.

### 3. Split embeddings and remove the contact-only feature package

- Split generic extraction, cache contracts, and feature datasets/loaders.
- Remove cache/persistence concerns from model extraction.
- Distribute `features/contact.py` by responsibility rather than moving it whole.
- Remove `features/` once all consumers are migrated.

Acceptance: cached/live parity, fingerprints, refresh/reuse, failed publication,
independent writers, train-only features and layerwise extraction reuse pass.

### 4. Correct training dependency direction

- Move substantive HPO/refit/seed-selection behavior into trainer ownership.
- Remove private TuneStep callbacks from trainer constructors.
- Return typed fitting results; publish models/settings at the stage boundary.
- Keep the same canonical model, refit budget and independently fit control.

Acceptance: TuneStep coordinates rather than optimizes; no nested stage runs or
trainer-to-stage dependencies; HPO/refit/standardization regressions pass.

### 5. Relocate controls and baselines

- Put control support and task-specific transformations in task handlers.
- Put shared ablation construction in `model/controls.py`.
- Prepare independently trained controls in `steps/controls.py` and reuse
  the fitting services directly; do not retain a forwarding-only control trainer.
- Keep explicit control preparation available without primary fitting.
- Move the baseline stage and artifact contracts to their agreed owners.
- Audit/remove the unused separation-prior path without expanding CASP support.

Acceptance: controls remain row-aligned and seed-stable; baseline statistics use
training labels only; legacy accepted configs warn/normalize as before.

### 6. Consolidate inference and scoring

- Move inference engines and statistical algorithms out of utilities.
- Use explicit semantic capabilities instead of contact-specific step branches
  or fallback hooks.
- Resolve existing supported implementations at a single boundary; do not
  introduce a new plugin framework.
- Preserve standalone stage calls, persisted reload, layerwise comparison and
  external evaluation.

Acceptance: one shared lifecycle, no `run_contacts`/task-owned pipeline, no
deliberately unsupported inherited scoring operations.

### 7. Remove obsolete modules and validate the final tree

- Remove emptied old packages/modules and temporary migration aliases.
- Keep aliases only for genuine unchanged ProteinGym/public consumers.
- Update README, architecture documentation and relevant import examples.
- Verify root contains only run entry points and package initialization.
- Review remaining utils imports for domain leakage.

Acceptance: no dead duplicate implementations, unjustified forwarding files,
stale imports, or unexpected behavior/config/artifact changes.

## Verification gates

After each migration run the smallest affected tests and configured hooks.
Before completion:

1. Run `.venv/bin/python -m pytest -q tests/evaluate`.
2. Run configured pre-commit hooks on all changed files, including Ruff/import
   sorting, formatting and mypy. Preserve unrelated work when hooks edit files.
3. Run `git diff --check` and inspect changed paths. Verify ProteinGym evaluation
   files remain unchanged.
4. Exercise tiny real AMPLIFY/ESM fixtures for sequence classification,
   regression, multilabel, token, pseudo-perplexity and calibration.
5. Exercise cached/live, classical, layerwise, HPO/refit and independent
   random-trunk control paths, including zero-shot controls.
6. Exercise bilinear, attention and Jacobian contact prediction with original
   coordinate gaps and standalone scoring reload.
7. Preserve shared-filesystem timeout/failure semantics and test independent
   writers locally; do not claim NFS/GPFS guarantees from local tests.
8. Run CUDA precision checks where available and distinguish unavailable checks.
9. Parse the existing scratch verification commands without running them, then
   use user-authorized real workloads as integration evidence. A parsed command
   is not evidence that a full benchmark completed.

Document any unavoidable compatibility/private-backbone limitations honestly.
Success means clearer ownership and one coherent execution flow, not simply
fewer lines in `utils/` or a directory tree matching this document.

## Suggested commit message

```text
refactor(evaluate): unify task protocols and pipeline ownership

- Integrate contact prediction through task-owned preparation, fitting,
  prediction, and scoring contracts instead of parallel contact pipelines
- Extract training orchestration from models and TuneStep into trainers
- Add an explicit ControlsStep for independently fitted controls
- Separate model-free baseline runs from model evaluation and controls
- Move baseline predictors into model/ and consolidate dataset resolution
- Introduce typed artifact contracts and clarify dependency boundaries
- Move bootstrap and layer comparisons out of ScoreStep into metrics
- Publish contact metrics and control summaries in run_manifest.json
- Centralize fixed contact fitting settings and simplify protocol guards
- Replace advisory file locks with bounded atomic cache coordination
- Move dataset sourcing scripts, helpers, and configs under scripts/
- Remove evaluation cross-validation and obsolete implementation paths
- Update tests and documentation while preserving ProteinGym evaluation

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
```
