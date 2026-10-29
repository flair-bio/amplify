# Evaluation refactor implementation plan

This records the first refactor. The follow-up ownership and current module
locations are documented in [the ownership refactor](ownership-refactor-plan.md).

## Scope and constraints

Continue on `oseaj/eval-update-contact-prediction-refactor`, preserving the existing
uncommitted work. Implement one phase at a time; each phase must leave a usable
evaluation engine.

- Do not create commits or pull requests.
- Leave ProteinGym evaluation scripts, fold behavior, and benchmark protocols
  unchanged. Preserve shared APIs they consume through narrow compatibility
  wrappers when necessary.
- Keep dataset-sourcing entry points in `modules/evaluate/scripts/dataset_sourcing`
  and reusable helpers in its `shared/` package, never under `src/`.
- Do not reintroduce evaluation-engine cross-validation.
- Preserve CASP splits, alignment, candidate pairs, hierarchical weighting,
  predictor presets, seed streams, and missing-label behavior.
- Preserve persisted artifact paths and formats unless a migration is explicitly
  documented and tested.
- Do not add a generic plugin framework, new benchmark suite, or distributed
  contact execution as part of this refactor.

## Current state

Implementation is complete through phases 0-8; combined verification is recorded
below. All changes remain uncommitted on the requested branch.

| Phase | Implemented ownership/contract |
| --- | --- |
| 0 | Separate contact control preparation, including zero-shot Jacobian controls and predictor metadata |
| 1 | Contact fitting in `trainers/contact.py`; development extraction/fingerprints in `features/contact.py`; typed fitting settings |
| 2 | Typed lifecycle hooks, optional tokenizer, incompatible prepared-state validation, explicit flat/structured scoring capability |
| 3 | CASP validation in `ContactPredictionTask.validate_config`; predictor task compatibility in `HeadConfig` |
| 4 | Model-only control conditions; inline heuristics in `predict.baselines`, reusable sequence baselines unchanged; legacy normalization with warnings |
| 5 | Layerwise optimization/synchronization and cached-control fitting delegated to trainers; no recursive `TuneStep.run` |
| 6 | Shared result records with prediction metadata; compatibility layer-keyed properties for existing consumers |
| 7 | Explicit encoder/FP32-head composition; isolated positional compatibility with state restoration |
| 8 | Staged Polars aggregation, atomic publication, bounded independent-job directory locks |

Deliberate limits: old backbone APIs still require isolated rotary table
compatibility. Workspace reservation locks remain unchanged. Bulk dataset
schema checks remain explicit where Arrow kernels would not preserve diagnostics.
`TuneStep` retains HPO/refit/persistence coordination and feature extraction;
the services own optimization, rather than introducing another orchestration
framework. ProteinGym evaluation is unchanged; evaluation CV remains removed.

## Target ownership

| Component | Responsibility |
| --- | --- |
| Pipeline and steps | Execution order, process coordination, invoking typed services |
| Task handlers | Task semantics, data interpretation, prediction representation, scoring policy |
| Models | Architecture, forward/extraction behavior, model serialization |
| Trainers | Optimization, search, early stopping, selection, refit, training results |
| Control fitting service | Independent control construction and fitting, including zero-shot controls |
| Feature extraction | Domain-specific extraction using shared cache/storage primitives |
| Scorers | Flat metrics or structured aggregation, with explicit input contracts |
| Configuration | Local validation plus task/predictor compatibility checks |
| Storage utilities | Atomic publication and explicit coordination policy |

Prefer small functions and existing services over new class hierarchies. Keep the
shared pipeline independent of concrete contact model classes.

## Phase 0: Establish the baseline and close lifecycle gaps

**Dependencies:** None. Complete before further structural changes.

**Work**

- Record the current changed-file inventory and existing test/check failures.
  Do not revert unrelated work or fix unrelated baseline failures.
- Exercise the shared pipeline for bilinear, attention, and categorical-Jacobian
  predictors, with and without `random_trunk`.
- Specifically investigate zero-shot controls: the shared pipeline skips tuning
  in `evaluate_as_is`, while contact control construction currently happens in
  `tune_artifacts`. Add a regression demonstrating the intended Jacobian control
  behavior before fixing the lifecycle.
- Check that zero-shot metadata preserves the selected predictor identity, and
  that accelerator preparation does not alter predictor device/precision behavior.
- Separate control preparation from primary-model tuning so inference-only runs
  can construct their controls without invoking a primary trainer.
- Verify persisted prediction reload and standalone scoring for all three
  predictors, not just in-memory pipeline results.

**Acceptance**

- Zero-shot evaluation never enters the primary optimization path.
- Requested controls are available in both tuned and zero-shot evaluations.
- Predictor identity, seeds, control artifacts, and score reload survive round trips.
- Unsupported process counts and workspace-seed mismatches fail before output
  reservation or model loading.

## Phase 1: Extract contact fitting and feature services

**Dependencies:** Phase 0.

**Primary locations:** `model/contact_model.py`, `model/contact_attention.py`,
`tasks/contact_prediction.py`, `utils/embed.py`, `trainers/`.

**Work**

- Move `fit_contact_head` and `fit_contact_attention_head` into contact trainer
  services. Move `fit_random_trunk_contact_control` into a control-fitting
  service that delegates to those trainers when fitting is required.
- Move `contact_development_features` and contact-specific cache fingerprint
  construction out of generic embedding utilities into a contact feature module.
- Return typed fitting results: selected settings, training history, feature
  identity, and control seed metadata. Keep artifact publication separate from
  optimization.
- Reduce contact lifecycle hooks to delegation; do not make the task handler the
  new owner of optimization, model-specific persistence, or cache coordination.
- Update direct callers and tests. Retain wrappers only for actual compatibility
  consumers; avoid permanent duplicate implementations.

**Acceptance**

- Model modules no longer implement fitting loops, estimator fitting, or trainer
  imports.
- Generic embedding utilities no longer import contact dataset/task types.
- Tiny deterministic fixtures preserve fitted predictions, selected settings,
  train/validation isolation, checkpoints, and random-trunk seed streams.

## Phase 2: Strengthen execution and artifact contracts

**Dependencies:** Phase 1.

**Primary locations:** `tasks/base.py`, `schemas/artifacts.py`, step modules,
`tasks/contact_prediction.py`.

**Work**

- Replace lifecycle-hook `Any` parameters with existing config, workspace,
  artifact, metadata, and result types; use type-only imports to avoid cycles.
- Introduce a small typed execution context only if it removes repeated
  cross-step plumbing. Do not pass the entire evaluation config everywhere.
- Use explicit payload variants or constructor validation to reject incompatible
  contact, cached-feature, and live-token prepared states.
- Ensure optional tokenizer/loaders reflect real requirements, rather than
  annotating values as mandatory when contact fixtures or executions omit them.
- Separate flat metric computation from structured prediction scoring. Contact
  should not advertise an applicable flat-metric capability and then always
  raise in `compute_metrics`.
- Audit all callers and subclasses before narrowing interfaces. Preserve legacy
  contact metric helpers where genuine consumers still need them.
- Keep existing layerwise and ordinary output keys during this phase; normalize
  result identity separately in Phase 6.

**Acceptance**

- The changed contracts type-check without new broad `Any` escapes.
- Invalid prepared states fail at construction or at one explicit boundary.
- Callers use a declared scoring capability rather than a task name or a method
  that is deliberately unsupported.
- All supported execution modes retain standalone step and disk-reload behavior.

## Phase 3: Localize configuration validation

**Dependencies:** Phase 2.

**Primary locations:** `utils/config_validation.py`, `config.py`, task/predictor
configuration modules.

**Work**

- Keep generic cross-step checks in the central validator.
- Move CASP task constraints to the contact task boundary, and bilinear,
  attention, and Jacobian constraints to their predictor configuration owners.
- Reuse the existing task registry for task validation; avoid introducing a
  second registry unless predictor dispatch actually needs one.
- Distinguish invalid combinations from protocol defaults. Do not silently
  accept or rewrite unsupported scientific settings.
- Preserve current YAML inputs and useful validation messages during extraction.

**Acceptance**

- Adding a task-specific compatibility rule does not require another hardcoded
  task-name branch in the central validator.
- Parameterized config tests cover valid presets, rejected overrides, evaluation
  modes, control support, and dataset compatibility.

## Phase 4: Complete baseline/control separation

**Dependencies:** Phases 0 and 3.

**Primary locations:** baseline runner/utilities, `utils/controls.py`,
prediction/scoring configs, control documentation.

**Work**

- Route model-independent baselines through baseline configuration and artifact
  production; reserve inference controls for model/input/label ablations.
- Audit `separation_prior` support before migration. Preserve currently supported
  behavior; do not enable a new CASP baseline merely by relocating its helper.
- Normalize legacy baseline names in `predict.controls` at one config boundary,
  with an explicit deprecation warning and duplicate-request validation.
- Preserve paired row alignment, probability columns, training-only statistics,
  reusable artifact identity, and baseline provenance.

**Acceptance**

- `ControlBuilder` receives only actual control conditions.
- New configurations contain no dataset heuristics in `predict.controls`.
- Legacy accepted inputs still work through the documented migration path.
- Baselines never use validation/test labels to fit training statistics.

## Phase 5: Finish reducing TuneStep

**Dependencies:** Phases 1 and 2.

**Primary locations:** `steps/tune.py`, `trainers/classical.py`,
`trainers/torch.py`, layerwise fitting utilities.

**Work**

- Move the remaining layerwise optimization and stopped-head synchronization
  into a dedicated trainer.
- Move cached-control fitting out of the step and reuse trainer services without
  recursive `TuneStep.run` calls.
- Give trainer results one selection/refit/persistence handoff. Keep variance
  reporting separate from choosing the canonical returned model.
- Remove forwarding helpers only when they no longer serve callers; avoid
  replacing one large step with a single equally broad service.

**Acceptance**

- `TuneStep` selects and invokes services rather than implementing epoch loops,
  layerwise synchronization, or secondary training runs.
- Search selection, refit epoch budgets, feature standardization, model saving,
  and canonical-seed selection are unchanged.
- Existing ordinary, cached, classical, and layerwise regressions pass.

## Phase 6: Normalize layerwise result handling

**Dependencies:** Phases 2 and 5.

**Work**

- Represent evaluation identity explicitly, including optional layer index,
  without making every downstream consumer understand `int | None` keys.
- Make shared prediction/scoring orchestration iterate result records rather
  than duplicating ordinary and layerwise lifecycle code.
- Keep depth-specific inference and paired cross-layer comparison inside their
  specialized services; shared feature extraction must remain efficient.
- Preserve existing artifact paths and provide a narrow adapter for current
  `PipelineResult` consumers if changing its public shape is necessary.

**Acceptance**

- Ordinary and layerwise results follow one orchestration contract.
- Layer metadata, per-layer persistence, paired comparisons, and control row
  alignment remain unchanged.

## Phase 7: Isolate and replace fragile model adaptation

**Dependencies:** Phases 0 and 1.

**Primary locations:** `model/contact_model.py`, `model/contact_jacobian.py`,
model-family adapters.

**Work**

- Prefer explicit `position_ids` and supported backbone/head calls.
- Replace temporary `head.forward` assignment with explicit precision-controlled
  composition where the supported backbone API permits it.
- Isolate unavoidable model-family compatibility code behind a narrow adapter;
  validate capabilities and fail clearly for unsupported implementations.
- Remove private RoPE table mutation where public APIs support equivalent
  original-coordinate positions. Do not substitute cleaned contiguous positions
  if doing so changes contact predictions.
- If an API cannot support exact behavior, document the blocker rather than
  claiming the private dependency has been eliminated.

**Acceptance**

- Supported paths do not replace methods on shared model instances.
- Tests cover original-coordinate gaps, long sequences, repeated inference,
  exceptions, precision, and unchanged model state after prediction.
- CPU tests pass; CUDA precision checks run where hardware is available and are
  reported as unverified when unavailable.

## Phase 8: Improve aggregation and cache coordination

**Dependencies:** Phases 1 and 2.

**Work**

- Replace hierarchical Python grouping in contact aggregation with staged Polars
  expressions: units to chains, chains to complexes, then benchmark summaries.
- Preserve per-metric null handling, contributing counts, identity validation,
  and equal weighting at every hierarchy. Never replace hierarchy with pooled
  pair-level averages.
- Use Arrow kernels for bulk pre-tokenization checks where semantics match;
  retain targeted row diagnostics and task-specific alignment code.
- Centralize cache publication and coordination. Use atomic publication with
  validation; distinguish ranks within a run from independent concurrent jobs.
- Do not replace cross-job locks with Accelerate barriers: barriers coordinate
  participating ranks, not independent processes on shared storage.
- Preserve existing workspace reservation protection unless its replacement
  provides equivalent exclusion and failure recovery.

**Acceptance**

- Parity tests cover null metrics, missing labels, unequal unit counts, duplicates,
  mixed identities, and complex metadata errors.
- A representative benchmark demonstrates the intended performance improvement.
- Concurrent writers, failed extraction, refresh/reuse, and partial files are
  tested. NFS/GPFS guarantees are documented, not inferred from local tests.

## Validation and delivery gates

For each phase, run the smallest affected existing tests first, plus focused
regressions for behavior being fixed. Use local tiny checkpoints and datasets;
normal tests must not require Hub access or a GPU.

Before accepting the combined refactor:

1. Run `pytest -q tests/evaluate`.
2. Run the repository's checks with `make check`, or use
   `pre-commit run --files <changed-files> --show-diff-on-failure` while iterating.
   The configured hooks include Ruff/import sorting, formatting, and mypy.
   Hooks can edit files: inspect their diff and preserve unrelated user changes.
3. Run `git diff --check` and inspect the changed-file list.
4. Run broader repository tests when shared compatibility consumers are affected.
   CI uses `uv run pytest tests/ -n auto`; limit local parallelism to the resources
   actually available on the shared machine.
5. Compare against Phase 0 failures. Resolve newly introduced failures; report
   unrelated pre-existing failures and unavailable hardware checks explicitly.

Do not install or update dependencies speculatively. Use the existing locked
environment and hook versions; restore missing tools only after a validation
command demonstrates that they are unavailable.

Each phase handoff should state the changed ownership/contracts, compatibility
decisions, validation outcomes, and remaining blockers. Keep implementation
changes uncommitted and do not open a PR.

## Verification and run commands

Combined evaluation suite: **1,527 passed, 1 skipped**. Eight focused final
regressions also passed after tightening invalid config inputs and baseline
manifest reloads. The repository's configured hooks passed on all changed
evaluation files, including pinned Ruff/import sorting, formatting, and mypy;
`git diff --check` passed. Existing multiprocessing deprecation and undefined
metric warnings remain. The staged aggregation benchmark preserved parity on
18,000 rows and ran approximately 2.28 times faster than the Python hierarchy.

The offline execution matrix uses tiny real AMPLIFY/ESM checkpoints, real
tokenization, fitting/inference, persisted predictions, and standalone scoring.
It covers sequence classification/regression, multilabel, token classification,
pseudo-perplexity, calibration, all three contact predictors, random-trunk
controls, cached/live probes, fine-tuning, classical probes, and layerwise runs.
CPU and CUDA checks exercise public MLM composition and FP32 heads. Full
benchmark datasets and multi-node/shared-filesystem deployments are separate
validation workloads, not implied by tiny-model success.

Run the integration checks from the repository root:

```bash
.venv/bin/python -m pytest -q \
  tests/evaluate/test_pipeline_tasks.py \
  tests/evaluate/test_contact_embedding.py \
  tests/evaluate/test_contact_jacobian.py \
  tests/evaluate/test_predict.py \
  tests/evaluate/test_storage.py
```

For actual datasets, use the existing task/model overlays. These combinations
have been configuration-validated; executing them requires their checkpoints,
datasets, and sufficient compute:

| Evaluation | Dataset overlay | Extra overlay |
| --- | --- | --- |
| Sequence classification | `localization_prediction.yaml` | None |
| Regression | `stability_prediction.yaml` | None |
| Multilabel | `GO_mf.yaml` | None |
| Token classification | `ssp_q3.yaml` | None |
| Pseudo-perplexity | `uniprot_protparams.yaml` | `tasks/pseudo_perplexity.yaml` before dataset |
| Calibration | `human_calibration.yaml` | None |
| Contact | `contact_prediction_casp.yaml` | One of `predictors/contact_embedding.yaml`, `contact_attention.yaml`, `contact_jacobian.yaml` |

For example, generate reusable sequence baselines before the model run:

```bash
uv run python -m modules.evaluate.src.run_evaluate --baselines \
  modules/evaluate/configs/config.yaml \
  modules/evaluate/configs/datasets/localization_prediction.yaml

uv run python -m modules.evaluate.src.run_evaluate \
  modules/evaluate/configs/config.yaml \
  modules/evaluate/configs/models/esm2_35m.yaml \
  modules/evaluate/configs/datasets/localization_prediction.yaml \
  workspace.run_id=refactor-check
```

The default cached-MLP protocol performs a substantial HPO sweep. For a plumbing
smoke run only, append `steps.tune.hyperparameter_search.enabled=false`,
`steps.tune.refit_on_train_validation=false`, and `steps.tune.max_epochs=1`.
Do not use that epoch override for the fixed bilinear contact protocol.

Contact uses the same entry point, with optional independent controls:

```bash
uv run python -m modules.evaluate.src.run_evaluate \
  modules/evaluate/configs/config.yaml \
  modules/evaluate/configs/models/esm2_35m.yaml \
  modules/evaluate/configs/datasets/contact_prediction_casp.yaml \
  modules/evaluate/configs/predictors/contact_jacobian.yaml \
  'steps.predict.controls=[random_trunk]' \
  workspace.run_id=refactor-check
```

Replace the final predictor overlay for bilinear or attention fitting. Use a new
run ID to preserve previous results; `--overwrite` explicitly replaces only the
selected run. No evaluation CV option remains. ProteinGym retains its separate,
unchanged commands.
