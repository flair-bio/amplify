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
- Shared `majority_class` and `train_mean` are generated and scored by a separate
  baseline run (`--suite <suite> --baselines`). Model evaluation never requires
  these artifacts. Token majority is also a shared train-only baseline.
  CASP contact controls are not yet integrated with the
  contact-specific inference and scoring path.
- Model-control predictions are rebuilt from the current head; only
  frozen-trunk embeddings are reused through the embedding cache.
- Comparisons only exist when `steps.score.bootstrap_enabled=true`. The
  base config enables it; the schema default is `false`. Without it,
  controls receive point scores but no paired uncertainty/significance results.

### Shared sequence baselines

Generate dataset-level artifacts once, without a model overlay or accelerator
launch. `--baselines` selects every job of the suite whose task declares shared
baselines:

```bash
uv run python -m modules.evaluate.src.run_evaluate \
  --suite sequence_classification --baselines \
  --output-root /path/to/eval
```

Run model evaluation and baseline evaluation in either order with the same
dataset/split settings and `runtime.output_root` (`--output-root`). Task presets
select `baselines.names=[majority_class]` for sequence/token classification and
`[train_mean]` for sequence regression. These names select generation defaults,
not implicit comparisons. Model runs save provenance even with `baselines.names=[]`.
The baseline route ignores model/tuning settings and uses train-only labels,
never validation or final-refit-only data. External test predictions use the
same source training prior.

Artifacts live under
`<output-root>/<task>/<dataset>/baselines/<name>/protocol-<hash>/fit-<hash>/test-<hash>/`.
Identity includes actual sequence/ID/label content, source split mappings, and
the task/class contract. Completed artifacts are reused; inconsistent or
corrupt files fail explicitly. Scoring checks IDs, labels, and observation
counts, retaining repeated sequence observations. Token comparisons require unique
source IDs and match original evaluated residue positions/masks, including cropping
and ignored labels (`-100`). Baseline predictions cover all valid test residues;
comparison scores are recomputed on the model's evaluated population, not copied
from full-population standalone scores. Standalone baseline score reports live in
`scores/<metric-protocol-hash>.json` beside their predictions.
Constant-prediction Spearman is NaN; undefined paired comparisons are omitted.

### Disk-only rescoring, comparison and recovery

Generate missing baselines for **every** token dataset in the suite, not just
datasets whose model jobs failed:

```bash
uv run python -m modules.evaluate.src.run_evaluate \
  --suite token_classification --baselines --output-root /path/to/eval

# Rescore saved predictions without loading a model or running inference.
uv run python -m modules.evaluate.src.run_score \
  --manifest /path/to/model/run_manifest.json

# Explicitly select conditions, regardless of the run's original baselines.names.
uv run python -m modules.evaluate.src.run_score \
  --manifest /path/to/model/run_manifest.json \
  --baseline majority_class --control random_trunk
```

Comparison outputs are separate under the model run's `comparisons/<protocol-hash>/`.
Missing/incompatible baselines fail only this explicit stage; standalone model
scores remain intact. Repeat `--baseline`/`--control` to select more conditions;
unselected saved controls are excluded. Both stages accept `--score-config FILE`
(a `ScoreConfig` JSON object) to change metrics or enable bootstrap/significance.

For older failed runs, first rescore their saved predictions. Add
`--recover-context` when comparing if provenance or token coordinates are missing.
Recovery loads the **producing run's saved evaluation config**, original data and
pinned tokenizer to reconstruct its exact crop/mask population; it does not train
or perform model inference, nor rewrite prediction files. Changed fingerprints,
ambiguous IDs or unmatched populations are rejected. Never infer residue coordinates
from matching labels alone. Recovery requires the original datasets and configuration
to remain available. Standalone rescoring needs neither baselines nor token coordinates.

Available shared sequence/token baselines (multilabel classification is unsupported):

| Name | Tasks | Protocol (version 1, identity-bound) |
| --- | --- | --- |
| `majority_class` | sequence/token classification | Most frequent valid training label; ignored residues excluded (default for classification) |
| `train_mean` | regression | Training-label mean (default for regression) |
| `composition` | classification, regression | Normalized frequencies over 20 canonical amino acids; ambiguous residues excluded from the denominator. Classification: `LogisticRegression(C=1.0, lbfgs, max_iter=1000)`; regression: `Ridge(alpha=1.0)`. Preset `baselines/composition.yaml`. |
| `mmseqs_knn` | classification, regression | MMseqs2 search against the training split (sensitivity 7.5, e-value 1e-3, max 300 hits, k=5). Classification votes are bit-weighted, with the training-label prior used when there is no hit; regression is a bit-weighted mean, with the training mean as fallback. Requires `mmseqs` on `PATH`. Preset `baselines/mmseqs_knn.yaml`. |

Each protocol dict is part of the artifact identity. Changing a setting
(e.g., `C`) creates a new `protocol-<hash>` directory. Artifacts from before
this identity change live at the old paths and are **not** adopted: run
`--baselines` again to create the protocol-bound artifacts.

The AA-frequency and contact separation-prior estimators are wired into the
baseline-only route (`--baselines`, see below). The contact protocol is
`separation_prior`; the PPL protocol is `amino_acid_frequency`.

### Structural baselines (approved conservative defaults)

| Name | Task | Protocol |
| --- | --- | --- |
| `separation_prior` | `contact_prediction` | Residue-pair contact rate per original separation, fit on the **train** split's labelled CASP chains. Canonical pair IDs (clean sequence, original `i`, `j`) are de-duplicated; only pairs at separation ≥ 6 count (`MIN_SEPARATION`); Laplace smoothing over the eligible separations; unseen separations fall back to the smoothed global rate. Scored on the test split with the CASP metrics. |
| `amino_acid_frequency` | `pseudo_perplexity` | Multinomial over the 20 canonical amino acids, Laplace(1), ambiguous/non-canonical residues excluded. Fit on a **declared public train-only reference corpus** (`baselines.amino_acid_reference` in `configs/tasks/pseudo_perplexity.yaml`: `flair-bio/contact_prediction_casp`, split `train`, pinned SHA `258a928…`). Never the PPL test split. PPL = exp(-mean log p) per residue. Baseline-only; not row-aligned to model predictions. |

Limitations: the reference corpus is small (the CASP train structures, not a
large UniRef-scale sample), so the amino-acid frequency prior is a weak
reference. Structural baselines exist only as `--baselines` artifacts; the
contact separation prior is never a model control. Both are unavailable (with
an explicit error) when their inputs cannot be resolved; nothing falls back to
test data.

## Catalog

![Control conditions and how Score compares them](../../../figures/eval_controls.png)

*Left: what each control replaces or scrambles. Right: illustrative scores.
`majority_class` is often well above chance on imbalanced tasks; controls that
are not retrained land near chance; how much signal `scrambled_sequences`
keeps depends on the task.*

`random_trunk` trains a separate probe in cached and layerwise runs.
Other model controls replace weights or inputs without retraining.

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
  controls to `TuneStep.run` before prediction; a missing fitted probe is an
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

Source: [`utils/controls.py`](../src/utils/controls.py); label scrambling is
`scramble_labels` on each task handler in [`src/tasks/`](../src/tasks/).

## Defaults and compatibility

Each task config ships a preset; override it with `steps.predict.controls`
(`[]` disables controls):

| Task type | Controls in the shipped task config |
|---|---|
| `sequence_classification` | `random_trunk`; shared `majority_class` |
| `sequence_multilabel_classification` | `random_trunk` |
| `sequence_regression` | `random_trunk`; shared `train_mean` |
| `token_classification` | `random_trunk`, `majority_class` |
| `contact_prediction` | None (CASP controls are not yet integrated) |
| `pseudo_perplexity` | `random_trunk` |
| `calibration` | `random_trunk` |

`random_trunk` is the per-model default. Other controls remain available by
explicit override where supported. Shared sequence baselines run once per
dataset/split/protocol with `--baselines`; token majority computes per run.
Composition and MMseqs k-NN are available as opt-in sequence presets (see
above). Multilabel priors and token baselines are not wired. The contact
separation prior and pseudo-perplexity amino-acid frequency are baseline-only
(see Structural baselines).

Config validation ([`utils/config_validation.py`](../src/utils/config_validation.py))
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
| `scrambled_labels`, `majority_class`, `train_mean`, `separation_prior` | Predictions beat the label or separation prior | The model has learned nothing usable for this task |
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
[`utils/bootstrap.py`](../src/utils/bootstrap.py).

## Adding a control

1. Add the name to the constants and `ControlCondition` in
   [`utils/controls.py`](../src/utils/controls.py), and to the relevant
   `RANDOMIZED_*`/`NO_INFERENCE_CONTROLS` sets.
2. Build its predictions in `ControlBuilder.build` (non-model baselines) or
   `_build_control_model` / `_run_embedding_control_inference` (model
   variants, with and without the embedding cache). Return rows aligned to the
   trained model's `id`s.
3. Reject unsupported task types or heads in
   [`utils/config_validation.py`](../src/utils/config_validation.py).
4. Support or explicitly reject it in layerwise mode
   ([`utils/layerwise.py`](../src/utils/layerwise.py)).
5. Add it to the relevant task presets in `configs/tasks/`.

Score picks up any control present in Predict's output; no scoring changes are
needed. For multi-GPU code paths, follow the
[distributed execution patterns](../README.md#distributed-execution-accelerate-patterns).
