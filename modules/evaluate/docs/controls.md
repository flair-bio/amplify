# Controls

A test score on its own doesn't say where the performance comes from: the
pretrained trunk, the tuned head, amino-acid composition, or just the label
distribution. Each control removes one of these. Score then tests whether the
trained model still beats it.

- [How controls flow through a run](#how-controls-flow-through-a-run)
- [Catalog](#catalog)
- [Defaults and compatibility](#defaults-and-compatibility)
- [Interpreting results](#interpreting-results)
- [Statistics](#statistics)
- [Adding a control](#adding-a-control)

## How controls flow through a run

```
steps.predict.controls → Predict: build each control's predictions
                       → Score: paired bootstrap + permutation test vs. trained
                       → scores_long.parquet rows with control=<name>
```

- Predict writes model controls next to `predictions.parquet`.
  `scrambled_labels` needs no inference and has no file.
- Sequence `majority_class`, `train_mean`, `composition`, and `mmseqs_knn` run
  separately with `run_evaluate.py --baselines`. Score loads their shared
  artifacts; model evaluations never generate them.
- Token `majority_class` uses the same shared `--baselines` workflow via
  top-level `baselines.names=[majority_class]`. Its prior counts all labeled
  residues in the original training split, excluding `-100`, regardless of
  model, training crop, seed, or final refit.
- Pseudo-perplexity's `amino_acid_frequency` baseline is also generated with
  `--baselines`; it estimates an IID residue prior from the test sequences
  without labels because the benchmark is test-only.
- CASP supports the per-model `random_trunk` control and the model-agnostic
  `separation_prior` baseline. The latter is generated once with `--baselines`
  and reused by every predictor. Bilinear and attention controls fit their own
  heads; Jacobian controls are zero-shot. Their hierarchical scores do not
  require generic bootstrap.
- Model-control predictions are rebuilt from the current head; only
  frozen-trunk embeddings are reused through the embedding cache.
- Comparisons only exist when `steps.score.bootstrap_enabled=true`. The
  base config enables it; the schema default is `false`. Without it,
  controls are predicted but never scored.

### Shared sequence baselines

Baseline defaults live in `configs/tasks/*.yaml` under `baselines.names`, not
in separate baseline config files. Override that list with `baselines.names=[...]`
when selecting a subset.

Generate dataset-level artifacts once, without a model overlay or accelerator
launch:

```bash
uv run python -m modules.evaluate.src.run_evaluate --baselines \
  modules/evaluate/configs/config.yaml \
  modules/evaluate/configs/datasets/solubility_prediction.yaml
```

Then run the normal model evaluation with the same dataset/split settings and
`workspace.base_path`. Task presets select
`baselines.names=[majority_class, composition, mmseqs_knn]` for sequence
classification and `[train_mean, composition, mmseqs_knn]` for sequence
regression.
`baselines.names=[]` disables shared comparisons independently of model controls.
The baseline route ignores model/tuning settings. Label-based sequence
baselines use train-only labels and sequences, never validation or final-refit
data. External test predictions use the same source training fit. The
pseudo-perplexity frequency baseline is the exception: it estimates its
unsupervised amino-acid frequencies from the evaluated sequences, since this
benchmark has no training split. It uses the test prefix allowed by
`steps.prepare.max_length` (reserving one leading and one trailing special
token), add-one smoothing over the 20 canonical residues plus one pooled
ambiguous-residue bucket, and scores each residue with its log frequency.

Artifacts live under
`<benchmark-root>/<task_type>/<dataset_name>/baselines/<name>/fit-<hash>/test-<hash>/`.
Identity includes actual sequence/ID/label content, source split mappings, and
the task/class contract. Completed artifacts are reused; inconsistent or
corrupt files fail explicitly. Scoring checks IDs, labels, and observation
counts, retaining repeated observations. The run manifest records artifact
references and baseline point scores even without bootstrap comparisons.
Constant-prediction Spearman is NaN; undefined paired comparisons are omitted.

### Shared CASP contact baseline

The sequence-separation prior also follows the shared artifact workflow. Generate
it once with the contact dataset configuration (no model or predictor overlay):

```bash
uv run python -m modules.evaluate.src.run_evaluate --baselines \
  modules/evaluate/configs/config.yaml \
  modules/evaluate/configs/datasets/contact_prediction_casp.yaml \
  workspace.edition=14
```

Ordinary contact evaluations use the task preset's
`baselines.names=[separation_prior]`; Score loads the stored candidate-pair
predictions and reports their hierarchical CASP metrics beside the model.
Artifact identity includes the contact dataset/schema fingerprint, CASP edition,
and prior protocol. Generate separately for each edition when the workspace is
edition-specific.

Missing artifacts are an error with instructions to run `--baselines`, not an
automatic baseline job. Duplicate requests fail.

### Shared token baseline

Run `--baselines` with the token task and dataset configuration before model
evaluation, just as for sequence tasks. The compact artifact stores the
training prior and original test labels once per source protein; it does not
repeat class probabilities for every residue. Score broadcasts that prior
onto the exact live or cached prediction rows after checking protein IDs and
that evaluated labels form a valid cropped window (with `-100` positions
removed). Truncation does not refit the prior or require another artifact.
Source protein IDs must be unique; datasets without IDs use source row indices.
External tests reuse the main training prior.

This intentionally replaces the old per-run prior computed from cropped
training dataloaders. Token task presets no longer use
`steps.predict.baselines`; disable shared comparisons with `baselines.names=[]`.

## Catalog

![Control conditions and how Score compares them](../../../figures/eval_controls.png)

*Left: what each control replaces or scrambles. Right: illustrative scores.
`majority_class` is often well above chance on imbalanced tasks; controls that
are not retrained land near chance; how much signal `scrambled_sequences`
keeps depends on the task.*

`random_trunk` trains a separate probe in cached and layerwise runs.
Other model controls replace weights or inputs without retraining.
For CASP, the task specifies an independent randomized predictor and fitting
protocol. The control stage materializes its feature requests and publishes
the result; bilinear/attention controls train independently, while the
categorical Jacobian remains zero-shot.

| Control | What changes | Retrained? | Question it answers |
|---|---|---|---|
| `untrained` | Pretrained checkpoint with a freshly initialized head | No | Did tuning help at all? |
| `random_head` | Trained trunk, re-initialized head | No | Does the tuned head carry the signal? |
| `random_trunk` | Re-initialized trunk with its own fitted probe | Yes for cached/layerwise probes; no for live/native runs | What does pretraining add beyond the architecture? |
| `random_both` | Trunk and head both re-initialized | No | Fully random-weight baseline |
| `scrambled_sequences` | Trained model, residues shuffled within each sequence | — | Does the model use residue order, or only composition and length? |
| `scrambled_labels` | Trained predictions scored against permuted labels | — | Is performance above chance for this label distribution? |
| `majority_class` | Most frequent train class, with the train class prior as probabilities | — | Classification floor; ROC-AUC is 0.5 by construction |
| `train_mean` | Train label mean for every example | — | Regression floor |
| `composition` | Logistic regression or ridge regression on normalized 20-AA composition | Yes | Can residue composition alone predict the target? |
| `mmseqs_knn` | Label/value from the top-1 MMseqs2 match in the train split | No | Does a close training homolog transfer the label? |
| `separation_prior` | Inverse original sequence separation for each residue pair | No | How much contact score comes from sequence proximity alone? |
| `amino_acid_frequency` | Smoothed IID residue probabilities fit on evaluated sequences | No | How well does a global amino-acid prior score residues? |

Details that change the interpretation:

- **`untrained` vs. `random_head`:** with a frozen trunk (`tune_probe`) both
  mean pretrained trunk + random head, so they are redundant. They differ only
  in `finetune`, where `random_head` keeps the fine-tuned trunk.
- **`random_trunk` with ordinary cached probes:** Tune independently selects a
  fresh MLP, linear probe, or classical estimator on random-trunk train features
  and validation labels using the same search space and budget. When final
  refitting is enabled, a fresh probe and preprocessing are fit on random-trunk
  train+validation features. Test features never enter selection or fitting.
  Predict regenerates the same seeded random trunk and uses that fitted probe,
  not the pretrained-feature probe's weights or scaler. Pass the requested
  controls to `ControlsStep.run` after tuning and before prediction; a missing fitted probe is an
  error. Checkpoints/settings use a separate `<variant>-random_trunk` workspace.
  This adds a second search/refit when HPO is enabled.
- **`random_trunk` in layerwise mode:** probes are retrained on a random trunk
  with the same layerwise procedure (not the ordinary cached HPO/refit path).
  `random_both` applies untrained heads to that random trunk. See
  [Layerwise linear probes](../README.md#layerwise-linear-probes).
- **`random_trunk` in live/native runs:** retains the fitted head and replaces
  only trunk weights at inference. This is a weight-replacement ablation, not
  a separately trained architecture baseline. Live/native control retraining
  and refitting remain deferred.
- **`scrambled_sequences`:** special tokens, padding, and each sequence's first
  residue stay in place, so length and composition are preserved.
- **`scrambled_labels` permutes at the task's natural unit:**
  - sequence tasks: across examples;
  - `token_classification`: within each sequence;
  - contact tasks: across scored pairs within each protein.
- **Randomness:** each control re-seeds from the run seed, so its result
  doesn't depend on the order in which controls are built.

Prediction/cache coordination is in [`steps/controls.py`](../src/steps/controls.py);
weight ablations are in [`model/controls.py`](../src/model/controls.py).
Label and input scrambling belong to task handlers in [`src/tasks/`](../src/tasks/).

## Defaults and compatibility

Each task config ships a preset; override it with `steps.predict.controls`
(`[]` disables controls):

| Task type | Controls in the shipped task config |
|---|---|
| `sequence_classification` | `random_trunk`; shared `majority_class`, `composition`, `mmseqs_knn` |
| `sequence_multilabel_classification` | `random_trunk` |
| `sequence_regression` | `random_trunk`; shared `train_mean`, `composition`, `mmseqs_knn` |
| `token_classification` | `random_trunk`; shared `majority_class` baseline |
| `contact_prediction` | `random_trunk`; shared `separation_prior` |
| `pseudo_perplexity` | `random_trunk`; shared `amino_acid_frequency` |
| `calibration` | `random_trunk` |

`random_trunk` is the per-model default. Other controls remain available by
explicit override where supported. Shared sequence, token and CASP contact
baselines run once per dataset/split/protocol with `--baselines`.
MMseqs k-NN requires the `mmseqs` executable on `PATH`. Composition is fitted
with scikit-learn logistic regression (classification) or ridge regression
(regression). MMseqs uses sensitivity 7.5 and up to 300 prefilter candidates,
with masking and composition correction enabled. It transfers the target from
the highest-bit-score training hit with E-value <= 0.001; ties use lower E-value,
then training row order. This is approximate significant-hit transfer, not an
exhaustive nearest-neighbor search.

Queries without a significant hit use the training mean (regression) or training
class-frequency probabilities and their majority class (classification).
Predictions retain every test row and include a boolean `fallback` column.
Artifact manifests and per-model score baseline references report `coverage`:
`matched`, `fallback`, `total`, and `fraction_matched`. Full-test metrics therefore
measure **MMseqs top-hit transfer with training-prior fallback**, not matched-only
performance. The versioned MMseqs protocol uses new artifacts; existing constant
and composition artifacts remain reusable. Both sequence methods are deterministic
and use no validation/test labels.

Config validation ([`contracts/validation.py`](../src/contracts/validation.py))
rejects these combinations before any work starts:

- `scrambled_sequences` on ragged tasks (`token_classification`, contact
  tasks): position-aligned labels would not move with the shuffled tokens.
- `majority_class` on regression or multilabel tasks; `train_mean` outside
  `sequence_regression`; generic controls on the CASP contact path.
- `untrained`, `random_head`, `random_both` with classical heads (`knn`,
  `sklearn_linear`, `random_forest`, `xgboost`): a fitted estimator has no
  random re-initialization.

## Interpreting results

Each control answers a different question, so read them together:

| Trained model beats... | What it shows | If it doesn't |
|---|---|---|
| `scrambled_labels`, `majority_class`, `train_mean` | Predictions beat the label prior | The model has learned nothing usable for this task |
| `scrambled_sequences` | The model uses residue order | The task may be solvable from composition and length alone |
| `random_trunk` (cached/layerwise) | Pretrained features beat a separately fitted random-feature probe | The comparison does not establish a benefit from pretraining |
| `untrained`, `random_head`, `random_trunk` (live/native), `random_both` | The trained weights are what produce the score | Something is wrong with tuning or loading; check the run |

The last row is mainly a sanity check: a head never trained on random features,
or a random head, usually lands near chance, so beating these is expected.
Use the first three rows to support claims about what a model has learned.

On imbalanced data, `majority_class` is often the hardest floor to beat in
accuracy or weighted F1. MCC and ROC-AUC score it at chance by construction
(0 and 0.5), so check it against a metric where it isn't automatically at chance.

## Statistics

Per (control, metric), Score reports:

- `diff_mean`, `diff_lower`, `diff_upper`: trained − control over paired
  bootstrap resamples (both conditions use the same resampled rows). For
  lower-is-better metrics (`mse`, `mae`, `pseudo_perplexity`, `kl*`), negative favors
  the trained model.
- `p_value`: one-sided paired permutation test of "trained beats control";
  each permutation swaps each example's prediction between the two conditions.
  The smallest possible value is `1 / (n_permutations + 1)`.
- `p_value_adjusted`: Benjamini–Hochberg across controls within each metric.
  Metrics are not pooled, because accuracy, F1, and MCC on the same predictions
  are highly correlated.
- Rows that share a source ID (per-residue rows from one protein, or the
  calibration positions of one protein) are resampled and swapped together.

Column layout and a loading snippet: [Reading results](../README.md#reading-results).
Settings: [Step 4 — Score](../README.md#step-4--score). Implementation:
[`metrics/bootstrap.py`](../src/metrics/bootstrap.py) and
[`metrics/comparisons.py`](../src/metrics/comparisons.py).

## Adding a control

1. Declare the name in `ControlCondition` in
   [`contracts/config.py`](../src/contracts/config.py), with weight-ablation
   constants in [`model/controls.py`](../src/model/controls.py) where applicable.
2. Coordinate predictions in [`steps/controls.py`](../src/steps/controls.py),
   using model inference and task transformations. Keep model-independent
   heuristic calculations in
   [`dataset/baseline_data.py`](../src/dataset/baseline_data.py);
   the standalone runner is [`runners/baselines.py`](../src/runners/baselines.py).
3. Declare task support in its handler. Generic head/stage compatibility checks
   remain in [`contracts/validation.py`](../src/contracts/validation.py).
4. Support or explicitly reject it in layerwise mode
   ([`model/layerwise.py`](../src/model/layerwise.py)).
5. Add it to the relevant task presets in `configs/tasks/`.

Score picks up any control present in Predict's output; no scoring changes are
needed. For multi-GPU code paths, follow the
[distributed execution patterns](../README.md#distributed-execution-accelerate-patterns).
