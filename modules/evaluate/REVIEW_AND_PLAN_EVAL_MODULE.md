# `modules/evaluate` — Review & Analysis Plan

_Reviewed at commit `416471a` (Eval embeddings, #217). Scope: everything under `modules/evaluate/` (≈8 k lines of `src/`, 4.2 k lines of tests, configs, README). `dataset_sourcing/` is only touched where it affects evaluation validity._

_Revision 3 (final). Revision 2 added the stale control-cache bug (§2.4/§6), cache-entry atomicity (§3.2), corrected the ProteinGym description (§5.2), qualified the predict-time-control, pruning-broadcast and "three modes" claims, and marked performance numbers as estimates. Revision 3 tightens the proposed fixes so they do not introduce new problems: single-file cache entries (not ids-first), a transform identity in the embedding cache key, separate (not "combined") seed/test variance reporting, a symmetric/low-memory contact head, consistent mode names, the `C`×`alpha` duplicate-candidate defect, id type preservation, a total cache quota, and the seed-dependent dataset fingerprint. Section 7.0 lists which proposals are optional, to avoid replacing one kind of over-engineering with another._

_Status note: this review predates the current working-tree fixes. The Phase 0 items for unequal-shape gathers, ragged scrambled-sequence controls, stale control caching, cache atomicity, trunk/head validation, RF parallelism, classical `C`/`alpha` filtering, cache fingerprints, and the lockfile have since been addressed. The remaining unchecked items are proposals or follow-up improvements, not assumed defects._

This document answers four questions:

1. Where is the code over-engineered for a research codebase?
2. Is it efficient/scalable — in particular hyper-parameter tuning (early stopping, pruning) for every probe type, and embedding caching?
3. Are the control conditions scientifically valid per task type, and what controls are missing?
4. Does the current design make it easy to add ProteinGym and contact prediction?

Each section ends with concrete, file-level suggested changes. Section 7 collects them into a prioritized plan.

---

## 0. TL;DR

| # | Finding | Severity | Section |
|---|---------|----------|---------|
| 1 | **Multi-GPU correctness bug (statically confirmed, not reproduced on 2 GPUs):** `accelerator.gather_for_metrics` is called on tensors whose shape differs across ranks (`(B, S)` token predictions/labels, `(B, S, H)` token embeddings, `input_ids`). accelerate's `gather` requires identical shapes (`pad_across_processes` exists for exactly this); token_classification, `store_sequences`, `scrambled_sequences`, and ragged embedding extraction are all likely wrong/hanging under `num_processes > 1`. Untested because the test-suite is single-process. | **High** | 3.3 |
| 2 | **Stale control cache (silent wrong results):** `random_trunk`/`scrambled_sequences` (and, in cache mode, `untrained`/`random_head`) control predictions are cached under a key built from `cache_identity()`, which for `EmbeddingProbeHead` is only *shapes* (`heads.py:200`) and for `ClassicalHeadModule` only *constructor params* (`heads.py:443`) — never the fitted weights, the embedding layers/pooling, or the test-set fingerprint. Re-training the same variant (different HPO outcome, different `embedding_cache.layers`, new dataset push) silently reuses controls produced by an older head. | **High** | 2.4, 6 |
| 3 | **`seeds` is a hyper-parameter search dimension** (TPE even "learns" the best seed). Selecting the seed on validation is optimistic selection; seeds should be *averaged over*, not *searched over*. There is no run-to-run variance reported anywhere. | **High (science)** | 3.1, 4.4 |
| 4 | **`random_trunk`/`random_head`/`random_both`/`untrained` are test-time ablations, not substitutes for the trained random-feature baselines the literature uses.** Weights are re-initialised at predict time under a head never fitted to those features, so they measure distribution mismatch, are hard to interpret, and overlap (exactly `untrained` ≡ `random_head` in embedding-cache mode). The informative control — *train the same probe on a randomly-initialised trunk* — does not exist. | **High (science)** | 4 |
| 5 | **Probe training on cached embeddings is DDP'd.** Every rank trains the same 1024→10 linear layer with all-reduce per step, plus `broadcast` for early-stop/prune decisions, `broadcast_object_list` of sklearn estimators, `DistributedSampler` for control inference… This is the source of most of the complexity in `tune.py`/`controls.py` and is very likely *slower* than single-process. Use multi-GPU for embedding extraction only; run HPO trials/seeds in parallel across ranks instead. | **High (efficiency + simplicity)** | 2.1, 3.1 |
| 6 | **Token-level embeddings are never cached and are fully materialised in RAM on every rank.** 100 k seqs × 500 tok × 1280 d × fp16 ≈ 128 GB *per rank*. Contact prediction will need per-token embeddings; this needs a disk-backed sharded store (or no caching + per-epoch frozen forward). | **High (scalability)** | 3.2 |
| 7 | **Bootstrap/permutation cost scales as `n_samples × n_controls × n_metrics × sklearn-call-over-all-tokens`**; for token tasks this is plausibly hours of single-threaded CPU (estimate — benchmark). First step: NumPy confusion-matrix metrics instead of sklearn per call; second step if needed: per-example sufficient statistics so a resample is a `bincount`. | Medium-High | 3.3 |
| 8 | `num_epochs` is a categorical HP *and* drives the LR schedule → Hyperband pruning compares trials at the same epoch under different schedules (biased against long schedules). `early_stopping_patience` is `null` by default, TPE's `n_startup_trials=10` with `n_trials=15` makes BOHB ≈ random search. | Medium | 3.1 |
| 9 | Classical heads: no `n_jobs`, no XGBoost GPU/early-stopping, kNN refits per `k`, RF re-grows per `n_estimators`, duplicate `C`×`alpha` candidates, Optuna over ≤12-point categorical grids. | Medium | 3.1 |
| 10 | Embedding cache: `test` never cached (re-embedded for every head-type variant + every re-embedding control), `seed` always in the key (identical val/test embeddings recomputed per seed — and the HF dataset fingerprint is seed-dependent too), random-trunk embeddings never cached, ids gathered with `gather_object` per batch and stored as strings, all-gather to every rank instead of shard-and-write, two-file entry that is not atomic as a unit, per-entry (not total) size budget. | Medium | 3.2 |
| 11 | `scrambled_sequences` is invalid for token-level tasks (labels stay position-aligned to shuffled residues), and crashes for token_classification + embedding cache (`extract_embeddings(is_ragged=True, tokenizer=…)` raises). | Medium (science) | 4 |
| 12 | The `TaskHandler` abstraction covers model class/labels/metrics but **not** the inference procedure, pairwise labels, per-sequence metric aggregation, CV folds, or "no training" tasks. Contact prediction (L×L labels, P@L by separation, per-sequence averaging) needs step-level changes; ProteinGym zero-shot (MLM masked-marginal scoring, official per-assay metrics + UniProt-ID/function-category aggregation) is better served by a separate entry point than by generalising the pipeline. | Medium-High | 5 |
| 13 | Over-engineering: a large share of `tune.py`/`embed.py`/`controls.py` is rank-lockstep plumbing, a two-layer dataloader stack over in-memory tensors, a hashed parquet cache for control *predictions*, `ClassicalHeadModule` faking an `nn.Module`, a re-implementation of HF's `LengthGroupedSampler`, a 4-field hard-coded search-space schema, and 150 lines of cross-field config validation that exists because the mode (frozen-probe vs fine-tune vs evaluate-as-is) is implicit. | Medium | 2 |

Overall: the module is thorough, well-documented and carefully reasoned (reproducibility, cache keys, distributed pitfalls), but it is built as a **production-style general-purpose pipeline** rather than a **research harness**. The recommendation is not to throw it away but to (a) make the *mode* explicit, (b) move multi-process concerns out of probe training, (c) restructure controls around "trained baselines vs test-time perturbations", and (d) generalise the task contract only as far as the next two tasks actually need (§7.0).

All speed-up factors, timings and line-count deltas in this document are **estimates from reading the code, not measurements**; Phase 0 includes a small benchmark to confirm the ones that drive prioritisation.

---

## 1. What the module does today (orientation)

```
run_evaluate.py
  ├─ PrepareStep   tokenize HF dataset → DataLoaders → HF Auto* model
  │                └─ [embedding_cache] frozen-trunk forward once per split → EmbeddingDataset loaders + EmbeddingProbeHead
  ├─ TuneStep      torch loop (mlp / torch_linear, optionally tune_trunk) with optional Optuna/grid/random HPO
  │                └─ classical heads (sklearn_linear / knn / rf / xgb) via sweep_classical_head on rank 0
  ├─ PredictStep   run_inference on test + ControlBuilder for each control → predictions.parquet + run_manifest.json
  └─ ScoreStep     metrics, bootstrap CIs, paired permutation test vs each control, BH correction → scores_long.parquet
```

Task-specific behaviour lives in `tasks/{sequence_classification,sequence_regression,token_classification}.py` via `TaskHandler` hooks (`build_model`, `collate_labels`, `align_labels`, `extract_predictions`, `split_batch_rows`, `flatten_for_metrics`, `compute_metrics`, `scramble_labels`).

Key file sizes: `tune.py` 955, `embed.py` 844, `controls.py` 687, `heads.py` 471, `dataloader.py` 437, `score.py` 413, `bootstrap.py` 350, `prepare.py` 364.

---

## 2. Over-engineering

The guiding question for each item: *does this complexity buy a research result, or does it buy generality/robustness we do not need?*

### 2.1 Distributed (accelerate) machinery wrapped around probe training — **largest source of complexity**

Evidence:
- `steps/tune.py:671-730` — per-epoch `broadcast(should_prune)` and `broadcast(should_stop)` so ranks stop "in lockstep".
- `steps/tune.py:374-379` — `broadcast_object_list([best_estimator, …])` pickles a fitted sklearn estimator (RF can be hundreds of MB) to every rank.
- `steps/tune.py:892-945` — every rank creates its own Optuna study; only rank 0's trial is real; hparams broadcast per trial.
- `utils/controls.py:432-530` — `DistributedSampler` + id-based de-duplication so that a head over *already fully-replicated* embeddings runs sharded.
- `utils/embed.py:455-582` — all-gather of embeddings/labels/ids to *every* rank every batch.
- README "Distributed execution patterns" section exists to document all of this.

Why it is over-engineering: with a **frozen trunk**, the only GPU-heavy work is the one-off embedding extraction. Everything after that (probe training, HPO, controls, scoring) is tiny; DDP over a linear layer adds an all-reduce per step and buys nothing. For **trunk fine-tuning** DDP is warranted; there the early-stop broadcast is redundant (every rank already holds the gathered validation metric) while the prune broadcast remains necessary as long as rank 0 alone owns the Optuna trial.

Suggested change:
- Make the execution regime explicit in config (see §2.5 for the full mode list), not inferred from flag combinations; the two that matter here:
  - `mode: tune_probe` — (optionally multi-GPU) `extract_embeddings` → **rank 0 only** (or, better, *each rank independently*) runs probe training/HPO/controls/scoring on in-memory/GPU-resident tensors with **no collectives at all**. Use the other ranks for *embarrassingly parallel* trials/seeds (see 3.1).
  - `mode: finetune` — DDP training loop as now. The *early-stop* broadcast is redundant (the validation metric is already identical on every rank after `gather_for_metrics`, so each rank derives the same decision). The *prune* broadcast is **not** redundant under the current design: only rank 0 owns the real Optuna trial, so `should_prune()` must still be communicated — keep that one `broadcast` per epoch plus the one `broadcast_object_list(hparams)` per trial, unless HPO orchestration is moved out of the training processes altogether (e.g. a driver that launches one `accelerate` job per trial — simpler, but only worth it if multi-GPU fine-tuning with HPO is a real use case, see §8).
- Delete: `_run_local_head_inference`, `DistributedSampler` in controls, estimator broadcast, rank-lockstep comments. Estimated −300 lines (to be confirmed when done).

### 2.2 Two-layer dataloader stack over an in-memory tensor

`EmbeddingDataset` (`embed.py:611`) → `embedding_collate_fn` (`embed.py:679`) → `accelerator.prepare(DataLoader)` → later `find_embedding_dataset` (`embed.py:633`, walks wrapper `.dataset` attributes) → `build_tensor_embedding_dataloader` (`embed.py:645`, rebuilds a `TensorDataset` loader) → `TuneStep._prepare_cached_dataloaders` (`tune.py:235`) → `_train` then has to handle both `tuple` and `dict` batch shapes (`tune.py:618-625`), as does `run_inference` (`inference.py:85-92`).

Suggested change: represent cached embeddings as a plain `EmbeddingSplit(X: Tensor[N,H] (on device), y: Tensor[N], ids: list)` and iterate with `torch.randperm(N)[i:i+bs]` slices. No `Dataset`, no collate, no `DataLoader`, no wrapper-walking. Also solves the per-batch CPU→GPU copy (3.1). Estimated −150 lines, large speed-up.

### 2.3 `ClassicalHeadModule` pretending to be an `nn.Module`

`heads.py:383-470`: sklearn estimator wrapped in `nn.Module` with a `_device_anchor` parameter, `log(predict_proba)` as "logits", one-hot "logits" when no `predict_proba`, `.to(inputs_embeds.device)` so it can flow through `run_inference` + `gather_for_metrics`. Plus `cache_identity()` for control cache hashing, `save`/`load` via joblib.

Suggested change: in `tune_probe` mode the classical path is `est.fit(X_tr, y_tr); est.predict(X_te)` producing the predictions DataFrame directly. No torch wrapper needed. Keep joblib save. Estimated −90 lines.

### 2.4 Control-prediction parquet cache — over-built *and* unsafe

`controls.py:195-332, 532-566` + `cache_identity()` on two head classes + `weights_fingerprint`. ~170 lines of caching, plus a `broadcast` for the `use_cache` decision (2.1) and a column-set sanity check. On a cache hit it does avoid the re-embedding pass; the problems are:

1. **Invalidation is wrong (high severity).** The key is `(control, seed, collect_probabilities, is_scrambled, model_identity)`. `model_identity` is `cache_identity()` when available: `EmbeddingProbeHead` → `embedding_size:probe_hidden_size:num_labels` (`heads.py:200-202`, *shapes only*); `ClassicalHeadModule` → `head_type` + `get_params()` (`heads.py:443-447`, *constructor params only*). Neither includes the fitted weights/trees, the embedding layers/pooling/normalisation that produced the features, or a test-set fingerprint. The cache lives under the variant directory, so **re-running the same variant** after (a) a different HPO outcome (affects `random_trunk`/`scrambled_sequences`, which run the *trained* head), (b) `embedding_cache.layers: [-1]` → `[-2]` (same `embedding_size`), (c) a different `hyperparameters`/`search_space` that happens to select the same estimator params, or (d) a new push of the dataset with the same ids, silently reuses control predictions from the *previous* head — the trained-vs-control comparison is then wrong without any warning. The column-set check cannot catch this. Only the plain-HF path (`type(model).__name__ + weights_fingerprint`) is keyed on weights.
2. **Poor reuse.** Predictions are head-specific, so the cache cannot be shared across head types, whereas the expensive part (re-embedding test under a re-initialised trunk / scrambled input) *is* shareable.

Suggested change: drop the control-prediction cache; cache **embeddings** instead (test split, random-trunk embeddings per seed, scrambled-sequence embeddings per seed) through the same `EmbeddingCache`. The existing content key already distinguishes a re-initialised trunk (via `model_fingerprint`) and the feature settings, but **not** an input perturbation: the dataset fingerprint describes the *unperturbed* tokenised dataset, so scrambled-sequence embeddings of the test split would collide with the clean ones (outright, once `seed` is dropped from the key for deterministic splits). Add an explicit `input_transform` field to `embedding_cache_key` (`None` | `"scramble_sequences:v1:seed=<s>"`), i.e. control name + algorithm version + scramble seed. Controls then become cheap to rebuild every run, cannot go stale w.r.t. the head, and the code shrinks (estimated −170 lines). If the prediction cache is kept in the interim, Phase 0 must at minimum fold a hash of the fitted head state (state_dict for torch heads, `joblib`-serialised estimator or `coef_`/trees digest for sklearn) plus the embedding cache key into `_control_cache_key`.

### 2.5 Implicit "mode" → 150 lines of cross-field validation

`utils/config_validation.py` enforces: `embedding_cache ⇒ ¬tune_trunk ∧ tune_head`; classical head ⇒ embedding_cache ∧ ¬tune_trunk ∧ no `untrained/random_head/random_both`; `torch_linear ⇒ ¬tune_trunk ∧ (embedding_cache ∨ token_classification)`; `resume_from_checkpoint ⇒ ¬embedding_cache`; direction/metric consistency; `collect_probabilities` auto-enable; … The knobs `embedding_cache.enabled × tune_trunk × tune_head × head_type × controls` are not independent. The valid combinations collapse into three *regimes* (not an exhaustive enumeration of every legal flag tuple — e.g. token_classification can train the frozen native linear head without the cache, which is still "frozen probe" with features computed on the fly; and `tune_trunk=tune_head=False` on a repo that already ships a task head is a legitimate "evaluate as-is" run):

| mode | what is trained | inputs | head types |
|---|---|---|---|
| `tune_probe` | head only | cached embeddings (or on-the-fly frozen features for native token heads) | linear / mlp / sklearn_linear / knn / rf / xgb |
| `finetune` | trunk + head | tokens | native HF head (mlp) |
| `evaluate_as_is` | nothing | tokens | an existing task head shipped with the checkpoint (no random head allowed) |

Two smells the explicit mode would remove: (i) the current "pretrained" variant on a base model evaluates a **randomly-initialised head** (the smoke-test default) — meaningful only as a control; (ii) `tune_trunk=True, tune_head=False` is accepted by the schema and then silently overridden to `tune_head=True` with a warning (`hpo.py:197-203`) — either reject it at validation or support it.

Suggested change: `steps.mode: tune_probe | finetune | evaluate_as_is`, with head/controls/caching derived. "Headless zero-shot scoring" (ProteinGym) is *not* a fourth pipeline mode — it is a separate entry point with its own scorer contract (§5.2), which keeps the pipeline's mode set small. Most of `config_validation.py` and the `TORCH_EMBEDDING_HEAD_TYPES`/`CLASSICAL_HEAD_TYPES` special-casing disappears; the README matrix becomes three rows.

### 2.6 Smaller items

| Item | Where | Suggestion |
|---|---|---|
| Re-implementation of HF `LengthGroupedSampler` (+ mega-batch logic, longest-first swap) | `dataloader.py:83-158` | `from transformers.trainer_pt_utils import LengthGroupedSampler` (accepts `lengths=` and `generator=`; confirm the import still exists in the pinned transformers 5.9). −80 lines. |
| `HyperparameterSearchSpaceConfig` hard-codes `learning_rate/num_epochs/weight_decay/seeds` | `tune.py:60-68` | Free-form `dict[str, list | {low,high,log}]` like classical `head.search_space`; then `hidden_size`, `dropout`, `batch_size` can be searched and the two search-space mechanisms unify. |
| `learning_rate` list means "categorical" under grid/random but "log-uniform between min and max" under bohb | `tune.py:912-917` | Make the semantics explicit in the schema (`{low, high, log: true}` vs list). |
| `store_sequences` decodes sequences by running the token dataloader back through `tokenizer.batch_decode` (with a gather) | `predict.py:242-268` | The sequences are a column of the HF dataset; join by `id`. −30 lines. |
| `_gather_train_labels` iterates the train DataLoader with collectives to compute a class prior | `controls.py:104-118` | Read the label column from the HF dataset (or the cached labels tensor). |
| 6-entry optimizer registry, scheduler kwargs, `gradient_accumulation_steps`, `warmup_ratio` for a linear probe | `hpo.py:21-31`, `tune.py:104-152` | Harmless but YAGNI; keep AdamW + (linear|cosine|constant). |
| `weights_fingerprint` is a per-tensor sum/abs-sum heuristic | `embed.py:292-309` | See §3.2.10: for cache *invalidation* use a real digest (full streaming hash of the state dict, or the checkpoint's immutable commit hash + config). |
| `resume_from_checkpoint` + `save_best_model` + `load_finetuned_model` | `run_evaluate.py`, `tune.py:434-484` | Only meaningful for `finetune`; make it mode-specific. |
| Extremely long explanatory comments (10–20 lines) on nearly every block | everywhere | Part of why `tune.py` is 955 lines. Trim to 1–2 lines once the distributed plumbing is gone. |
| README says classical winner is *refit on train+validation* (`README.md:321, 345`); code fits on train only (`heads.py:299-303`) | docs | Fix the README (train-only is the right choice for comparability). |
| `ControlComparison` docstring describes `p_value` as a bootstrap tail fraction (it is a permutation p-value) and says BH is applied "jointly across every (control, metric)" (it is per metric, across controls — `bootstrap.py:275-277`) | `schemas/artifacts.py:198-208` | Update. |
| `config.yaml:40` says the embedding cache is "unsupported for task_type=token_classification"; the validator allows it and `torch_linear.yaml` relies on it. `config.yaml:62` says `early_stopping_patience` "also drives optuna pruning" — pruning runs whenever `prune_enabled`, independent of patience (`tune.py:555`) | configs | Fix comments. |
| `save_best_model` for cached probes is a dead end: classical heads write `head.joblib` (`tune.py:388-400`; `ClassicalHeadModule.load` exists at `heads.py:463` but no pipeline path calls it) and torch probes write `probe_head.pt` (`tune.py:449-461`), while resume only recognises an HF `config.json` (`run_evaluate.py:121`) and is anyway rejected with the cache on | `tune.py`, `run_evaluate.py` | Either make resume mode-aware for both probe families or stop writing these files. |

---

## 3. Efficiency & scalability

### 3.1 Hyper-parameter tuning (early stopping & pruning, per probe type)

#### Current behaviour per head type

| head_type | training | HPO | early stopping | pruning |
|---|---|---|---|---|
| `mlp` (cached) / `torch_linear` | epoch loop over DataLoader batches of 128 (`tune.py:604-644`), DDP | grid / random / TPE+Hyperband over `lr × wd × num_epochs × seeds` | only if `early_stopping_patience` set (default `null`) | `trial.report` per epoch + Hyperband (bohb only) |
| `mlp` (finetune trunk) | same loop with trunk unfrozen | same | same | same |
| `sklearn_linear / knn / rf / xgb` | one `fit` per candidate (`heads.py:282-380`), rank 0 | grid / random / TPE over categorical grid | n/a | n/a (atomic fits) |

#### Problems

1. **Per-step overhead very likely dominates probe training.** Cached-embedding probes iterate a Python `DataLoader` (collate → pin → H2D copy) in batches of 128; for 100 k examples that is ~800 steps/epoch × up to 40 epochs × 15 trials (each BOHB trial samples *one* seed, `tune.py:924`; multiply by the seed count only for an exhaustive grid or for the post-HPO seed loop proposed below) of mostly Python/launch overhead, plus a DDP all-reduce per step. With `X` resident on the GPU and batch ≥1024 (or full batch for linear probes), an epoch over 100 k × 1024 should be on the order of milliseconds. **Hypothesis: ≥10× speed-up** for the probe path — cheap to verify with a 20-line benchmark (Phase 0) and, if confirmed, the single biggest HPO efficiency win.
2. **`seeds` as a search dimension** (`tune.py:924`, `_generate_candidates` `tune.py:781-784`). Selecting the best seed on validation is optimistic selection, and TPE will waste budget "learning" the best seed. With cached embeddings, seeds are cheap; they should be an *outer* loop: select hparams (1 seed), then retrain the selected config with `k` seeds and report mean ± std / CI across seeds (or use all seeds per trial and optimise the mean). See also §4.4.
3. **`num_epochs` as a categorical HP + LR schedule tied to it** (`tune.py:581-594`). Hyperband compares `val` at epoch *e* across trials; a `num_epochs=40` trial at epoch 5 is at a high LR on a linear decay while a `num_epochs=5` trial has fully annealed — pruning is biased. Replace with `max_epochs` + `early_stopping_patience` (default e.g. 5); the best epoch *is* the tuned epoch count. This also removes the `last_epoch_score` vs `best_epoch_score` branch (`tune.py:741`).
4. **Optuna configuration.** `TPESampler(seed)` has `n_startup_trials=10` by default; with `n_trials=15` (mlp.yaml) the first 10 trials are random — BOHB ≈ random search. `HyperbandPruner()` with `max_resource="auto"` infers resource from the first completed trial, which under (3) varies. Suggest: `n_startup_trials = max(5, n_trials // 3)`, `multivariate=True`, `HyperbandPruner(min_resource=3, max_resource=max_epochs)` (or `MedianPruner(n_warmup_steps=3)` which is simpler and works well at this scale), and a `--storage` option so trials can run in parallel processes.
5. **No parallelism across trials.** Trials run strictly sequentially even though every rank holds a full copy of the embeddings. With the DDP removed (2.1), rank `r` of `W` can run trials `r, r+W, …` (grid/random; each rank writes its trial results to a per-rank file, rank 0 reduces and picks the best after `wait_for_everyone()`) or share an Optuna journal-file storage (TPE) — near-linear speed-up in the number of GPUs, *instead of* the current slow-down. Optional (§7.0).
6. **Classical heads:**
   - `RandomForest*` default `n_jobs=None` → single-threaded on 100 k × 1024 (`heads.py:77-85`). Set `n_jobs=-1` by default.
   - `XGB*`: no `device="cuda"` (`hist` is already the default in xgboost ≥ 2.0), and `n_estimators` is swept as a grid instead of fitting once with `early_stopping_rounds` on the validation split (which also *prunes* automatically and reports the best iteration). Same for RF: `warm_start=True` lets one fit grow 100→300→500 trees and evaluate at each size.
   - kNN: refit + re-predict per `k` (`heads.py:327-331`); call `kneighbors(X_val, n_neighbors=max(ks))` once and slice for each `k`.
   - `sklearn_linear` over `C`/`alpha`: `LogisticRegression(warm_start=True)` along the regularisation path, or `joblib.Parallel` across candidates; for Ridge the `alpha` path is closed-form given one SVD of `X_tr` (sklearn's `RidgeCV` does this for LOO-CV; for a held-out split it is a ~15-line helper, or just accept the 4 refits — they are cheap).
   - Optuna+TPE over ≤12 categorical points (`sklearn_linear.yaml: n_trials: 4`) is pointless: use grid when `|grid| ≤ ~30`, random/TPE only for larger or continuous spaces.
   - **Duplicate candidates for `sklearn_linear`:** `sklearn_linear.yaml` lists both `C` (classification) and `alpha` (regression); `ParameterGrid` is built over *both* (`heads.py:313-316`, 3 × 4 = 12 candidates) and the irrelevant one is only discarded inside `build_estimator` (`heads.py:240-256`) — so classification fits each effective `C` four times and regression each `alpha` three times, and with `n_trials: 4` random/TPE may draw the same effective model repeatedly. Filter the search space by task *before* building the grid/TPE space.
   - Standardisation (`StandardScaler` in the Pipeline) is recomputed per candidate; scale once.
7. **Validation cost per epoch** (`_evaluate_validation`, `tune.py:743-771`): runs `run_inference` (gather → CPU → Python lists → sklearn). For cached embeddings compute the metric on-device/NumPy without list conversion. Minor but it multiplies with epochs × trials.
8. **`checkpoint_state` copies trainable params to CPU on every improvement** (`tune.py:703-706`) — fine for heads, 1.4 GB per copy for a 350 M fine-tune; acceptable but consider keeping the best state on GPU for heads and only offloading for `finetune`.

#### Suggested design (tune_probe mode)

```python
# pseudo-code
X_tr, y_tr, X_va, y_va, X_te = load_or_extract(...)          # tensors on device (fp16/bf16 → fp32 on the fly)
def train_probe(hp, seed) -> (probe, best_val, best_epoch):
    torch.manual_seed(seed); probe = make_head(hp)
    for epoch in range(hp.max_epochs):
        for idx in torch.randperm(N, device=dev).split(hp.batch_size): step(X_tr[idx], y_tr[idx])
        val = metric(probe(X_va), y_va)                        # on device
        trial.report(val, epoch); if trial.should_prune(): raise TrialPruned
        if not improved(val): patience -= 1; if patience == 0: break
    return best
study = optuna.create_study(...)                                # optional: storage=journal_file so ranks run disjoint trials
best = study.best_params
results = [train_probe(best, s) for s in seeds]                # outer seed loop → mean ± std
```

Same skeleton for classical heads (`fit` instead of the epoch loop, `early_stopping_rounds` for XGB). Everything is single-process; multi-GPU = more trials/seeds in parallel.

### 3.2 Embedding caching

What is good: single safetensors file per split (no inode explosion), atomic tensor-file replacement, content-keyed (`embedding_cache_key`) including weights + dataset fingerprints, preallocated output buffer, fast path for last layer, budget check.

Problems, by impact:

1. **Token-level (ragged) embeddings are never cached and live in RAM on every rank** (`embed.py:61-64, 765-769`; docstring rationale "100–1000× larger"). For `ssp_q3`-scale data this is several GB; for realistic token-level datasets or contact prediction it is tens–hundreds of GB, replicated per rank (all-gathered every batch, `embed.py:498-501`). This is the scalability ceiling. **Proposal:** a `TokenEmbeddingStore` = one `np.memmap`/safetensors file of shape `(total_tokens, H)` in fp16/bf16 + an `offsets` int64 array (`[N+1]`), written **per-rank shard** during extraction then concatenated (or kept as `K` shard files + a global index), read with `mmap` during probe training (random access by sequence, batched by token ranges). Same store serves token classification (rows = tokens), contact prediction (rows = tokens; pairwise features built on the fly per sequence), and sequence tasks (rows = sequences). Optionally reduce `H` with a fixed random projection / PCA (kept in the cache key) when storage is the bottleneck — note it as an option, not a default.
2. **Gather-to-every-rank** (`embed.py:498-531`, `gather_for_metrics` per batch, `gather_object` for ids at `embed.py:553-554`). Network cost `O(N·H·W)`, memory `O(N·H)` per rank, and `gather_object` of Python id lists per batch is slow. **Proposal:** each rank extracts its shard and writes `split_{key}.rank{r}.safetensors` together with the **explicit global row index of every row** (add an integer `row_idx` column to the tokenised dataset and pass it through the collator, rather than assuming anything about sampler order), `wait_for_everyone()`, rank 0 merges by `row_idx` (or readers open all shards). Ids are then looked up from the dataset by `row_idx` on rank 0, never sent over NCCL. Also fixes the unequal-shape gather bug (3.3) for free.
3. **`test` split is never cached** (`CACHEABLE_SPLITS`, `embed.py:64`) — rationale "only read once per run". But a sweep over head types (`knn`, `sklearn_linear`, `mlp`, `rf`, `xgb` overlays — all share the `embeddings_path`, which is deliberately variant-independent) re-embeds test every run, and `random_trunk` / `scrambled_sequences` re-embed test every run for every variant. Cache test too; cache perturbed-trunk embeddings under their own `model_fingerprint`, and perturbed-*input* embeddings under an explicit `input_transform` key field (2.4). Note that enabling test caching exposes an **id type loss**: `EmbeddingCache.write` stores every id as `str` (`embed.py:377`) while fresh token-loader ids keep their dataset type, so integer-id datasets would fail the `join(on="id")` in `controls.py:324`; preserve the type (ids as an int64 tensor when integral, or a typed JSON payload) or normalise to `str` everywhere.
4. **`seed` is always in the cache key** (`embed.py:278`). It only affects the crop when `random_truncate=True`; for `validation`/`test` (deterministic crops) two pipeline seeds recompute identical embeddings. Use `seed if random_truncate else None` — **but that alone will not yield reuse**: the key also contains the HF `Dataset._fingerprint`, and the tokenisation `map` function closes over `seed` (`dataloader.py:327`), so `datasets` hashes a seed-dependent function and the fingerprint differs per seed even for deterministic splits. Move `seed` out of the closure into `fn_kwargs` (it is hashed either way, but as a kwarg it can be `None` for deterministic splits) so the fingerprint is seed-independent when `random_truncate` is false; verify with two seeds that val/test hit the cache.
5. **`max_cache_gb: 2.0` is a per-entry limit** (`fits_budget` per split, `embed.py:351-357`) that silently degrades to no caching (warning only); 3 layers × 1280 × 500 k examples × fp16 = 3.8 GB. Raise it (e.g. 20 GB), make the miss *loud* (error unless `allow_uncached=true`) — silently recomputing every run is the worse failure mode on a cluster — and, once test/control/seed/layer entries are cached too, add a simple **total quota** for the cache directory (size check + oldest-atime eviction, a 20-line helper; no LRU machinery).
6. **Cache location** `<base_path>/<model>/<dataset>/tmp/embeddings` ties the cache to the results tree. Add `embedding_cache.dir` (e.g. `$SCRATCH/plm_embeddings/`) shared across `base_path`s/users; the key already makes files self-identifying.
7. **Training crops are baked into the cache** (random truncation per seed) — fine, but means one train embedding per seed. For long-sequence datasets consider caching *untruncated* per-token embeddings once and pooling/cropping at read time (only possible with item 1).
8. **Per-layer concatenation is baked in** (`layers: [-1,-2]` → one concatenated tensor). Caching per layer and concatenating at read time lets one extraction serve `[-1]`, `[-2]`, `[-1,-2]` probes (layer-wise probing is a very common analysis). Cheap change if the store is per-layer files sharing an index.
9. **Cache entries are not atomic as a unit** (`embed.py:359-378`): the safetensors file is written via tmp + `replace` (good), but the `*.ids.json` sidecar is then written non-atomically, and `exists()` checks only the tensor file. A crash/kill between the two leaves an entry that `exists()` accepts and `read()` fails on (`FileNotFoundError`/truncated JSON) on every later run until someone deletes it by hand. Fix: make the entry **one file** — put ids in the safetensors `metadata` (strings) or as an int64 tensor, written via tmp + `replace` as now. Two-file orderings (ids-first + `exists()` checks both) are *not* sufficient: under `cache_policy: refresh` an older tensor already exists, so a crash after replacing the ids leaves new ids paired with the old tensor and both files present. Avoid generation counters/manifests — the single-file representation is the simplest correct option.
10. `weights_fingerprint` (`embed.py:292-309`) is a *heuristic* (per-tensor sum and abs-sum), not a content hash: any permutation of values within a tensor collides. It is adequate for distinguishing "pretrained vs re-initialised vs re-trained" trunks in practice, but the docstring's "changes whenever any parameter value changes" overstates it, and once the cache is relied on for *correctness* (controls, §2.4) a collision-prone key is the wrong tool. Use a real digest: a full streaming `sha256` over the state-dict bytes (≈1–3 s for a 350 M model, once per run) or, for Hub checkpoints, the immutable commit hash (`model.config._commit_hash`) + config. A strided sample is still a heuristic.
11. `EmbeddingCache.read` (`embed.py:380-390`) is fine for pooled embeddings; the token store (item 1) should use `np.memmap`/safetensors slicing rather than whole-tensor reads.

### 3.3 Other efficiency / correctness issues

1. **Unequal-shape `gather_for_metrics` (multi-GPU bug; statically confirmed, not yet reproduced).** `utils/inference.py:113` gathers `(preds, labels[, probs][, input_ids])`; for token_classification these are `(B, S_rank)` with `S_rank` = that rank's padded batch length (dynamic padding in `EvaluationCollator`, `pad_to_multiple_of=8`). `utils/embed.py:501` gathers `(B, S_rank, H)` token embeddings; `predict.py:254` gathers `input_ids`. accelerate's `gather` (`_gpu_gather_one` → `all_gather_into_tensor`) requires identical shapes across ranks; `Accelerator.pad_across_processes` exists precisely for this. Affected under `num_processes > 1`: token_classification predict/validation/HPO, ragged embedding extraction, `store_sequences=True`, `scrambled_sequences` (gathers `input_ids`), `_decode_token_dataloader`. **Action:** reproduce with 2 GPUs on `ssp_q3` — or cheaper, add a CI job running the smoke test under `accelerate launch --num_processes 2 --cpu` (gloo's `all_gather` into `empty_like` buffers should fail loudly on shape mismatch, which makes the bug visible without GPUs); stop-gap: `pad_across_processes` on each tensor **separately with its own pad value** (`labels` → `-100`, `preds` → `-100` or any sentinel stripped by `split_batch_rows`, `input_ids` → `pad_token_id`, probabilities/embeddings → `0` with a length mask) before gathering — a single shared `pad_index` would corrupt one of them; or, better, eliminate the gathers via shard-and-write (3.2.2) and per-rank prediction files merged by row index.
2. **Bootstrap & permutation cost** (`score.py:318-347`, `bootstrap.py:133-170`): per resample, `full_scores` calls sklearn once *per metric* (each call re-validates and re-encodes the whole label vector), for the trained model and every control; then `n_permutations × 2 × n_controls` more. Back-of-envelope for token_classification with ~1 M tokens: `1000 resamples × (1 + 4 controls) × 5 metrics × ~0.3 s/sklearn call` is on the order of hours, plus a comparable permutation cost, single-threaded (estimate — benchmark on `ssp_q3`). Ragged resampling additionally rebuilds Python lists via `flatten_for_metrics` every time (`bootstrap.py:58-59`). **Proposal, in increasing order of effort:** (a) replace per-metric sklearn calls with one NumPy confusion matrix per resample (`np.bincount(y*C + ŷ, minlength=C²)`) from which accuracy/precision/recall/F1 (micro/macro/weighted)/MCC are closed-form — a small, local change in `metrics.py`/`bootstrap.py` that should already give a large speed-up; (b) only if still too slow (very large token counts, contact prediction): per-example sufficient statistics (per-example confusion contribution / per-sequence metric) so a resample is `bincount(idx) @ stats`. ROC-AUC and Spearman need raw rows in either case. (b) also makes contact-prediction bootstrap natural: resample sequences, average precomputed per-sequence P@L.
3. **`run_inference` materialises everything on every rank as Python lists** (`inference.py:127-150`). Fine for 10 k examples; for token-level outputs keep tensors and write parquet from NumPy.
4. `seed_everything` sets `torch.backends.cudnn.deterministic = True` globally — irrelevant for transformers, and only *may* exclude faster algorithms if conv layers are ever used. Harmless today; make it opt-in for tidiness.
5. `Accelerator()` is constructed repeatedly (`run_evaluate.py`, every step, `TuneStep` with `gradient_accumulation_steps`). Works because of the shared `AcceleratorState`, but construct once and pass it.

---

## 4. Controls — scientific validity

### 4.1 What the current controls actually measure

| control | implementation | sequence cls/reg (finetune) | sequence cls/reg (frozen probe) | token cls | verdict |
|---|---|---|---|---|---|
| `untrained` | `from_pretrained` again: pretrained trunk + whatever head `from_pretrained` gives — a **random head** for a base checkpoint, but a repo's *existing* task head if it ships one (`controls.py:591-603`) | random linear read-out of good features: usually near chance, not guaranteed | **exactly** ≡ `random_head` in embedding-cache mode (both build a fresh `EmbeddingProbeHead`, `controls.py:357, 413-419`) | as the finetune column | Uninformative for base checkpoints; exactly redundant in cache mode. Not the "untrained LM" baseline of the literature (which trains the probe). |
| `random_trunk` | re-init trunk, **keep trained head**, predict (`controls.py:604-609`, `controls.py:367-370`) | trained head applied to features it never saw: a distribution-mismatch ablation whose value depends on retained head bias/priors, not a defined baseline | same | same | Not informative as a *baseline*. The meaningful version is **random-init trunk + *trained* probe** (TAPE/ESM "random LM" baseline), which isolates the contribution of *pre-training* from architecture + probe capacity. Missing. |
| `random_head` | `deepcopy` trained model, re-init head in place via `_init_weights` | close to `untrained` but not identical: AMPLIFY's `_init_weights` does not touch `nn.Linear.bias` (`modeling_amplify.py:445-455`) so the trained class-prior bias survives — "random head" ≠ random | exactly ≡ `untrained` | as the finetune column | Largely redundant with `untrained`; the retained bias makes it less interpretable, not more. |
| `random_both` | both re-init | test-time ablation with no defined expectation | — | same | Redundant; `majority_class`/`scrambled_labels` give a *defined* chance reference. |
| `scrambled_labels` | permute test labels, keep predictions (`tasks/base.py:219`; within-sequence for token tasks `token_classification.py:94-104`) | chance given label marginals | ✓ | ✓ (within-sequence composition preserved — good) | Valid *chance* control; but the paired permutation test already gives the same null. Keep as a sanity check. |
| `scrambled_sequences` | shuffle residues (first residue fixed, `controls.py:676`) keep labels | ✓ composition-vs-order | ✓ | **invalid**: labels stay position-aligned, so residue *i* now carries label of residue *j*; measures label misalignment, not order dependence | Fix for token tasks: apply the *same* permutation to labels (tests context vs residue identity), or disable. Also **crashes** for token tasks with embedding cache (`embed.py:440-445` raises when `tokenizer` is passed with `is_ragged=True`; `controls.py:409-410` passes it). "First amino acid fixed" is arbitrary (only meaningful if it is Met) — document or remove. |
| `majority_class` | train prior (`controls.py:120-165`) | ✓ | ✓ | ✓ (token-level prior) | Valid, essential. ROC-AUC 0.5 by construction. |
| `train_mean` | train label mean | ✓ (regression) | ✓ | n/a | Valid. Consider `train_median` (MAE-optimal) alongside. |

In short: four of the eight controls are legitimate *test-time ablations* of the trained model, but they have no defined expected value, overlap heavily, and are not substitutes for trained baselines. The questions reviewers most often ask — "how much is pre-training vs architecture?", "how much beats a trivial sequence-only baseline?", "is this homology leakage?" — are not answered by any current control.

### 4.2 Proposed control taxonomy

Separate **(A) baselines that are trained** — same probe, same HPO budget, different *features* — from **(B) test-time perturbations of the trained model**, and **(C) chance references**:

| id | type | features / procedure | question answered | tasks |
|---|---|---|---|---|
| `random_init_trunk` **(new, highest priority)** | A | re-initialise trunk with `seed` (cache embeddings under its fingerprint), train the *same* probe with the *same* HPO | contribution of pre-training vs architecture prior + probe capacity | all (incl. contact, ProteinGym-supervised) |
| `onehot_probe` / `composition_probe` **(new)** | A | sequence tasks: 20-dim AA composition (+ length, optionally 2-mers); token tasks: one-hot of a ±k window (e.g. k=7, 21×20 dims); contact: pairwise one-hot + |i−j| | does the PLM beat a trivial sequence-only representation? (classic SSP baseline ≈ 60 % Q3 from windows) | all supervised |
| `nearest_train_neighbor` **(new, optional)** | A | label of the most similar train sequence (MMseqs2 / Biopython alignment identity); report identity to nearest neighbour per test example | homology leakage / memorisation; also enables *stratified* reporting by identity bins | sequence-level, token-level |
| `majority_class` / `train_mean` (`train_median`) | C | keep | chance given label marginals | cls / reg |
| `scrambled_labels` | C | keep (within-sequence for token tasks) | chance given label marginals, paired | all |
| `scrambled_sequences` | B | keep for sequence tasks; for token tasks permute labels jointly → `shuffled_context`; for contact apply same permutation to both axes | order/context dependence | sequence / token / contact |
| `untrained`, `random_head`, `random_both` | — | **remove** (subsumed by `random_init_trunk` + `majority_class`) | — | — |
| `random_trunk` | — | **rename/replace** by `random_init_trunk` (with training) | — | — |

Task-specific additions for the roadmap tasks (§5): ProteinGym zero-shot — `random_scores` (Spearman ≈ 0), a `blosum62` substitution-matrix baseline (note: ProteinGym's official "Site-Independent" baseline is an MSA-derived model, a different and stronger reference), `random_init_trunk` masked-marginal scores; contact prediction — `separation_prior` (predict contact probability from |i−j| frequency on train), `random_init_trunk` probe, `shuffled_context` (permute residues *and* both map axes), and ESM-style `attention_lr` (logistic regression on symmetrised+APC attention maps) as a reference *method* rather than a control.

### 4.3 Statistics

- The paired permutation test + per-metric BH (`bootstrap.py:84-171, 262-350`) is sound. Two notes: (i) for model-based controls labels are identical so swapping `(pred, label)` tuples is just swapping predictions — fine; (ii) "p-value" on `scrambled_labels` is circular (the control *is* a permutation null) — drop the test for that control or present it as the null.
- Bootstrap resampling unit: examples for sequence tasks ✓; for token tasks it resamples **sequences** (object arrays of per-sequence lists) ✓ — the right unit (tokens within a protein are not independent). For contacts the unit is **sequences**. For ProteinGym, do not invent a bootstrap: report the official scorer's numbers (its leaderboard uncertainty resamples *grouped performance rows within function categories*, not raw variants, and is expressed relative to a reference model); any within-assay variant bootstrap is an *additional*, clearly separated diagnostic (§5.2).
- Multi-class ROC-AUC uses `ovo` with `average=metric_aggregation`; `macro` default in `config.yaml` is fine; document that `weighted` differs.

### 4.4 Variance across training runs (missing)

Bootstrap CIs cover **test-sampling** variance only. Run-to-run (seed) variance is often larger for probes and fine-tuning, and currently the seed is *selected* (3.1.2). Proposal: after HPO, retrain the selected config with ≥3 seeds and report the two sources **separately**: per-seed metrics (each with its own test-set bootstrap CI) plus the across-seed `mean ± std`, all in `scores_long.parquet` with a `seed` column (currently deliberately excluded, `long_format.py:19-25`). Do *not* bootstrap the seed-averaged predictions and call it a combined CI — that is the test-sampling CI of an ensemble and contains no training variance. A genuinely combined interval would need a hierarchical bootstrap (resample seeds, then examples); it is not needed for a research report and is left out deliberately.

---

## 5. Extensibility: ProteinGym and contact prediction

### 5.1 What the abstraction assumes today

The steps hard-code these assumptions (outside `TaskHandler`):

| assumption | where | ProteinGym zero-shot | ProteinGym supervised | contact prediction |
|---|---|---|---|---|
| dataset = HF `DatasetDict` with `train/validation/test`, columns `sequence`, `targets` | `dataloader.py:238-268`, `prepare.py:158-174` | ✗ 217 substitution DMS assays (ProteinGym v1.x), no train; per-assay CSVs with WT + mutant sequences / mutation strings | ✗ per-assay 5-fold CV splits (random / modulo / contiguous) | ✓ (biomap `contact_prediction_binary` exists) |
| one HF `Auto*` model with `.logits`; prediction = `extract_predictions(logits)` | `inference.py:97-101`, `tasks/base.py:160` | ✗ needs `AutoModelForMaskedLM` and a *scoring procedure* (masked-marginal: one masked copy of the WT per unique mutated site, processed in batches, then `log p(mut) − log p(wt)` looked up per variant; or pseudo-log-likelihood) | ✓-ish | ✗ pairwise head over token embeddings (or attention maps); needs `(L,L)` logits + pairwise loss with separation mask |
| labels: scalar per example or list per token; `collate_labels(labels, B, S)` | `tasks/base.py:152` | scalar DMS score ✓ | ✓ | ✗ needs `(B, S, S)` with `-100` off-crop/off-sep; `align_labels` must crop 2-D |
| metrics on a *flat* vector (`flatten_for_metrics`) | `tasks/base.py:172-217` | ✗ official zero-shot metrics per assay — Spearman, AUC, MCC, NDCG, top-K recall — then **aggregated by averaging within UniProt ID, then within function category (Activity/Binding/Expression/Organismal Fitness/Stability), then across categories**; not a plain mean over assays | ✗ Spearman + MSE per assay, computed per CV scheme (random / modulo / contiguous) and averaged across schemes, then the same UniProt-ID → category aggregation | ✗ P@L, P@L/2, P@L/5 at short/medium/long separation **per sequence** then mean |
| a `Tune` step is required unless both `tune_*` are false, in which case a random head is used | `run_evaluate.py:116-141` | ✗ zero-shot = no head | ✓ | ✓ |
| predictions stored as scalar/list columns in parquet | `predict.py:163-175` | ✓ | ✓ | ✗ `L×L` floats per example (1 k seqs × 512² × 4 B ≈ 1 GB); store sparse top-k per separation bin or per-sequence metrics |
| `collect_probabilities` ⇒ `logits.softmax` on `(B, C)` | `inference.py:103-106` | n/a | n/a | ✗ `(B, S, S)` |
| embeddings cache: pooled `(N,H)` or ragged per token in RAM | §3.2 | n/a (zero-shot needs MLM logits, not embeddings) | pooled ✓ | ✗ needs per-token store (and optionally attentions) |
| controls: re-init at predict time | §4 | ✗ need `random_init_trunk` scoring + substitution-matrix baseline | ✓ after §4 | ✗ need separation prior, pairwise `shuffled_context` |
| `EvaluationMetadata.split == "test"` single result per run | `schemas/artifacts.py` | ✗ per-assay results (hundreds) + aggregate | ✗ per assay × fold | ✓ |

Verdict: **the `TaskHandler` hooks cover label shape, model class and metric names, but neither new task fits without editing `prepare/predict/score`** — exactly what the README promises is unnecessary. Contact prediction is the closer of the two (tokenisation, collator hooks, ragged bootstrap, workspace, tidy output reusable; model/head, pairwise labels, metrics, prediction storage, token embedding store new). ProteinGym zero-shot is essentially a different pipeline (no tune, custom scorer, per-assay grouped metrics with the official aggregation).

### 5.2 Proposed task contract (incremental, keeps the registry)

**Guard against over-generalising.** Add hooks only when the *second* concrete task needs them; do not pre-build a universal contract. The minimal, lowest-risk path per task:

- **ProteinGym zero-shot: do not force it through Prepare→Tune→Predict→Score.** A separate entry point (`run_proteingym_zero_shot.py`) that reuses trunk/tokenizer loading, the embedding/autocast utilities and the workspace, runs the masked-marginal scorer, writes per-assay score CSVs in the official layout, and **calls/ports the official ProteinGym scoring script** for metrics, the UniProt-ID → function-category aggregation and its uncertainty estimate, is simpler and more defensible than re-implementing that aggregation inside `ScoreStep`. Controls (`random_init_trunk` scores, `blosum62` baseline) are just two more scorers writing the same CSV layout. Any within-assay variant bootstrap is an *extra* diagnostic reported separately from the official numbers, never in their place.
- **Contact prediction and ProteinGym supervised** fit the existing 4-step pipeline *after* the `TaskHandler` gains the hooks below — these are the ones that have to exist anyway for any task with structured labels, grouped metrics or per-sequence aggregation.

Extend `TaskHandler` only with what the *next concrete task* needs; every hook below is **optional** (default = today's behaviour) and is introduced together with its first consumer, not up front:

```python
class TaskHandler:
    # existing: name, build_model, collate_labels, align_labels, metrics, scramble_labels ...
    example_unit: str = "example"           # resampling unit for bootstrap/permutation: example | sequence   (contact: sequence)
    def predict(self, model, batch) -> dict[str, Tensor]    # default: logits→argmax(+softmax); contact: per-sequence upper-tri scores
    def features_for_probe(self, token_states, batch) -> Tensor  # default: pooled / per-token; contact: pairwise (chunked)
    def per_example_stats(self, outputs, labels) -> np.ndarray   # OPTIONAL, only if §3.3.2(a) is too slow; contact: per-sequence P@L
    def default_controls(self) -> list[str]
```

Not proposed any more: `requires_training` / a `zero_shot` pipeline mode (zero-shot lives in its own entry point), `group_column`/assay aggregation inside `ScoreStep` (the official ProteinGym scorer does it), `baseline_features` as a generic hook (a one-hot/composition baseline is a small standalone feature function used by the `onehot_probe` control). A `load_splits(repo_id)` override on the handler is enough for CV folds if ProteinGym-supervised is in scope; a separate `DatasetAdapter` class is not needed until a third layout appears.

Concretely for the two roadmap tasks:

- **Contact prediction** (`tasks/contact_prediction.py`): per-token embeddings (token store or per-epoch frozen forward, §7.0) → a pairwise head that is **symmetric by construction and never materialises an `(L, L, kH)` feature tensor** — at `L=512, H=1280`, fp16, a `[h_i, h_j, h_i⊙h_j, |h_i−h_j|]` tensor is `512² × 4·1280 × 2 B ≈ 2.5 GiB per sequence` before activations, and the `[h_i, h_j]` concatenation is order-sensitive. Use instead: (i) a **low-rank bilinear** head `z_ij = (W_q h_i)ᵀ (W_k h_j)` with `W_q = W_k` or explicit `(z_ij + z_ji)/2` (O(L·r) feature memory plus only the `L×L` score matrix itself), or (ii) symmetric pair features `h_i + h_j, |h_i − h_j|, h_i ⊙ h_j` projected to a small `d` *before* pairing (project to `d ≈ 64–128`, then pair) and evaluated in row chunks; (iii) during training, sample a subset of pairs per sequence (all positives + a fixed ratio of negatives at `sep ≥ min_sep`) instead of the dense map — v0's `StructureRegressor(use_low_rank=...)` already points this way. Loss = CE/BCE with `pos_weight` and `min_sep` mask (v0 `contact_map_loss`); `collate_labels` builds `(B,S,S)` (bool/int8, not float); `predict` returns upper-triangular probabilities for `sep ≥ 6` computed in chunks (store per-sequence sparse arrays or just the top-`L` per separation bin, not dense maps); per-sequence P@L/2/5 × {short, medium, long} + AUC; bootstrap over sequences. Also an *unsupervised* reference path: logistic regression on symmetrised + APC attention maps (needs `output_attentions`, a second "feature kind" in the extractor). Random truncation must crop both map axes; prefer no random truncation for contacts (or `max_length` large enough).
- **ProteinGym**: zero-shot — separate entry point (above): model = `AutoModelForMaskedLM`; scoring = masked-marginal (**one masked copy of the WT per unique mutated site**, batched; multi-mutants summed over sites; `log p(mut) − log p(wt)` per variant; truncation/window handling for WTs longer than `max_length` as in the reference implementations); metrics and aggregation via the official ProteinGym scorer (Spearman, AUC, MCC, NDCG, top-K recall; UniProt-ID then function-category averaging). Supervised — `tasks/proteingym_supervised.py` on top of `tune_probe` with per-assay CV folds (a `load_splits` override), features = pooled mutant embedding (and/or WT-delta), official metrics (Spearman, MSE) per assay and per CV scheme (random / modulo / contiguous), averaged across schemes, then the official UniProt-ID → category aggregation via the official supervised scorer.

### 5.3 Things already in place that help

`@register_task` + import-time checks; `is_ragged`/object-array bootstrap; `align_labels`/`collate_labels` hooks; per-task `scramble_labels`; `EvaluationMetadata.split` placeholder; `scores_long.parquet` tidy output (add `seed`, and `fold` if ProteinGym-supervised is in scope).

---

## 6. Bugs & inconsistencies found along the way

1. Unequal-shape gathers under multi-GPU (§3.3.1) — **verify with 2 GPUs on `ssp_q3`**.
2. `scrambled_sequences` + `token_classification` + `embedding_cache` raises (`controls.py:395-411` → `embed.py:440-445`).
3. AMPLIFY `_init_weights` leaves `Linear.bias` (and norms) untouched → `random_head`/`random_trunk` controls keep trained biases/norm scales (`modeling_amplify.py:445-461`). For ESM the HF `_init_weights` zeroes biases. Either zero biases explicitly in `_reinit_submodule` or (preferred) construct the control model fresh from config (`AutoModel.from_config`) rather than `apply(_init_weights)` on a copy.
4. README vs code: classical refit on train+val (README) vs train-only (code) — code is right, README wrong.
5. `EmbeddingBackend.autocast_dtype` default `"float32"` vs config default `"float16"` (`embed.py:607`) — only matters for direct construction; harmless.
6. `HyperparameterSearchConfig.direction` default `"maximize"` + `model_fields_set` sniffing to detect "explicitly set" (`tune.py:507-513`, `config_validation.py:38-55`) — fragile; use `direction: Literal[...] | None = None` and resolve from the metric.
7. `TuneConfig.metric_aggregation` default `"weighted"` vs `config.yaml` `"macro"` for score — auto-synced, but the defaults in code disagree; pick one.
8. `config.yaml` controls comment omits `train_mean`; `PredictConfig.controls` default `["untrained"]` is not valid for classical heads (validator catches it, but the default should be mode-aware).
9. `_train` returns `last_epoch_score` when no patience is set but `best_epoch_state` is never restored — with `num_epochs` searched this is "intended", but once `max_epochs` + patience is the norm, always return/restore best.
10. `ensure_validation_split` carves validation from train by a *random* split (`dataloader.py:181-208`) while `dataset_sourcing` has a cluster-aware (MMseqs2) validation split (`preprocess.py:54-210`). Random validation splits risk homology leakage → optimistic model selection; prefer the clustered split (or warn loudly that the dataset lacks one).
11. `smoke_test.yaml` exercises the "pretrained" (random-head) variant as the main result; fine for a smoke test, but see §2.5.
12. **Stale control-prediction cache** (§2.4): `cache_identity()` ignores fitted weights / feature settings / test-set content, so re-running a variant can silently reuse another head's controls. High severity; fix or remove in Phase 0.
13. **Non-atomic cache entry**: safetensors written atomically, `*.ids.json` not, `exists()` checks only the former (`embed.py:359-378`). A kill between the two writes leaves a permanently broken entry.
14. `tune_trunk=True, tune_head=False` is accepted and silently overridden (`hpo.py:197-203`); reject at validation instead.
15. `random_truncate` is a *fixed* per-example crop (seeded by `(seed, index)`, `dataloader.py:320-329`), not per-epoch augmentation. Correct for cached embeddings (the crop must be fixed), but in `finetune` mode the name suggests augmentation it does not provide — rename (`train_crop: random_fixed | start`) or document.
16. `weights_fingerprint` is a collision-prone heuristic, not a content hash (§3.2.10) — replace with a real digest before relying on it for control/embedding cache correctness.
17. Stale config comments: `config.yaml:40` (token caching "unsupported"), `config.yaml:62` (patience "drives pruning").
18. `sklearn_linear` search space mixes `C` and `alpha` → duplicate candidates (§3.1.6).
19. Cache ids stored as `str` (`embed.py:377`) — latent join failure for integer-id datasets once test embeddings are cached (§3.2.3).
20. Tokenisation mapper closes over `seed` → seed-dependent dataset fingerprint even for deterministic splits (§3.2.4).
21. **Stale `uv.lock`** (last updated with #208 on Jul 29; `pyproject.toml` changed on Aug 19): the lock's `evaluate` extra lists only `accelerate/peft/scikit-learn/scipy` — `optuna` and `xgboost` are missing. `uv run` re-resolves silently, but `uv sync --locked`/`--frozen` (CI, reproducible cluster envs) installs without them, so the *default* `search_pattern: bohb` and `head_type: xgboost` raise `ImportError` there. Run `uv lock`, commit, and add `uv lock --check` to CI.

---

## 7. Prioritized plan

Effort: S ≤ ½ day, M ≈ 1–2 days, L ≈ 3–5 days (estimates). Each phase leaves the suite green.

### 7.0 What is *not* recommended (avoid trading one over-engineering for another)

The proposals below are deliberately tiered. Mandatory for correctness/science: Phase 0, the seed/epoch changes in Phase 1, the control restructuring in Phase 3. Everything else is conditional:

| proposal | do it only if… | simpler fallback |
|---|---|---|
| `TokenEmbeddingStore` (memmap + offsets, sharded) | probe epochs × HPO trials × seeds make repeated frozen forwards the bottleneck for token/contact tasks | no token cache: run the frozen trunk forward inside the epoch loop under `no_grad` (one full trunk pass over the dataset per epoch; still DDP-able) — zero new infrastructure |
| per-example sufficient statistics for bootstrap | NumPy confusion-matrix metrics (§3.3.2a) are still too slow | §3.3.2a alone |
| Optuna shared storage / parallel trials across ranks | HPO wall-clock on one GPU is actually a problem after Phase 1 | sequential trials on rank 0; other ranks idle or run other seeds |
| `DatasetAdapter` abstraction | a third dataset layout appears | per-task `load_splits` override on the handler (enough for ProteinGym-supervised folds) |
| `nearest_train_neighbor` control | homology leakage is a question you need answered per run | one-off analysis script using `modules/data` clustering |
| `features_for_probe` / `predict` hooks | contact prediction is being implemented | keep pooled / per-token and logits→argmax as today |
| `per_example_stats` hook, `group_column`, `requires_training`, a `zero_shot` pipeline mode | — (not recommended; see §5.2) | NumPy confusion matrix; official ProteinGym scorer; separate zero-shot entry point |
| total cache quota / eviction | test/control/seed/layer entries are all cached | 20-line size check + oldest-atime delete, no LRU library |
| free-form search-space schema | someone needs to sweep `hidden_size`/`dropout` | keep the 4 fields, drop `seeds`, rename `num_epochs`→`max_epochs` |

### Phase 0 — Verify & quick fixes (S–M)
- [x] **Stale control cache:** control prediction caching was removed; controls are recomputed from current features/models (§2.4). *(S)*
- [x] **Cache-entry atomicity:** cache data and metadata use one atomic safetensors entry (§3.2.9, §3.2.3). *(S)*
- [x] Fix unequal-shape gathers with per-tensor padding before collection (`inference.py`, `embed.py`, `predict.py`); verified with the two-process CPU `ssp_q3` smoke run. *(S)*
- [ ] Micro-benchmarks that gate later phases: (a) probe epoch time DataLoader vs GPU-resident slicing, (b) bootstrap time on `ssp_q3` with sklearn vs NumPy confusion matrix. *(S)*
- [x] Fix `scrambled_sequences` for ragged + cache by raising a clear config error. *(S)*
- [x] Reject `tune_trunk=True, tune_head=False` at validation instead of silently overriding. *(S)*
- [x] Set RF `n_jobs=-1` and filter `C`/`alpha` by task before building the classical grid. *(S)*
- [x] Make cache identity seed-aware only for random truncation, include the input transform and test split, and use a real weight digest. *(S–M)*
- [x] Optuna: `n_startup_trials` scaled to the trial budget, `MedianPruner(n_warmup_steps=3)` instead of Hyperband (whose resource ladder is ill-defined while `num_epochs` is a search dimension). *(S)*
- [x] README/doc fixes (refit claim, `ControlComparison` docstring, `train_mean`, `config.yaml` comments, and weight-fingerprint wording). *(S)*
- [x] Regenerate `uv.lock` with the Optuna/XGBoost evaluation dependencies. *(S)*

### Phase 1 — Make HPO efficient and scientifically clean (M)
- [x] `tune_probe`: GPU-resident tensors via `TuneStep._cached_embedding_split`/`_cached_batches` (index-slice batching, on-device validation metric, no `DataLoader`/collate in the training or validation loop for cached embeddings). `EmbeddingDataset`/`find_embedding_dataset` were kept (as the storage container `PrepareStep`/`embed.py` populate once), but `build_tensor_embedding_dataloader`/`_prepare_cached_dataloaders` were dead/removed — confirmed 2026-08-26 (`build_tensor_embedding_dataloader` had zero callers left in `src/`, only its own test; deleted both, all 372 `tests/evaluate` pass). *(M)*
- [ ] Replace `num_epochs` search dim with `max_epochs` + `early_stopping_patience` (default 5); always restore best epoch. *(S)*
- [x] Move `seeds` out of the search space → outer loop after selection (2026-08-26): `TuneConfig.variance_seeds` (default `[]`, opt-in) retrains the already-selected hyperparameters with extra seeds and logs validation-metric mean/std (`TuneStep._report_torch_seed_variance` for mlp/torch_linear, reusing `sweep_classical_head` with a fixed single candidate for classical heads) — diagnostic only, does not alter the canonical (steps.seed) model/estimator returned by the step. Deliberately did NOT add a `seed` column loop to `scores_long.parquet`/write per-seed artifacts: that would require Predict+Score to run per seed too (out of TuneStep's scope) — logging mean/std via `logger.info`/`log_wandb` is the minimal fix for "no run-to-run variance reported anywhere" without restructuring the pipeline. Also deliberately scoped to `cached_probe` only (frozen trunk over cached embeddings): reinitializing a full trunk+head per seed for `finetune` mode would be expensive and isn't where "seeds are cheap" applies. Tests: `TestVarianceSeeds` in `tests/evaluate/test_tune.py`. *(S–M)*
- [ ] Classical heads: warm-start/`kneighbors` reuse, XGB `early_stopping_rounds`, RF `warm_start`, scale once, grid for small spaces. *(M)*
- [ ] Free-form search-space schema shared by torch and classical heads. *(S)*

### Phase 2 — De-DDP the probe path, simplify (L)
- [x] Introduce `steps.mode: tune_probe | finetune | evaluate_as_is`; derive head/cache/controls (`config.py`, `utils/config_validation.py`). Note: `config_validation.py` did not shrink to ~40 lines as estimated (still 149) — it's mode-*derived* validation now (task/head_type × mode), not the old implicit-inference style, so the length is legitimate per-combination guardrails rather than leftover inference logic. *(M)*
- [ ] Extraction: per-rank shard files + merge by an explicit `row_idx` column carried through the dataset/collator; ids resolved on rank 0; per-layer files. *(M)*
- [x] `tune_probe`: everything after extraction runs without collectives — confirmed 2026-08-26: no `accelerator.prepare(model)`, no `DistributedSampler`, no prune/early-stop broadcast, no estimator broadcast when `_cached_embedding_split` returns non-`None` (`cached_probe`/`distributed_training` flags in `TuneStep`). Cross-rank trial/seed parallelism itself remains unimplemented (optional). *(M, parallel part optional)*
- [x] Remove `ClassicalHeadModule`, `_run_local_head_inference`, estimator broadcast, control-prediction parquet cache (cache embeddings instead), `_decode_token_dataloader` — all confirmed gone as of 2026-08-26 (classical heads are plain fitted sklearn estimators; local control inference is `utils/inference.run_local_embedding_inference`; no `cache_identity`/parquet cache remains in `utils/controls.py`; `predict.py` still owns `_decode_token_dataloader` for the main prediction path, which is correct — it's not control-specific). HF `LengthGroupedSampler` import was NOT done — `dataset/dataloader.py` still has its own hand-rolled `LengthGroupedSampler` class; low priority, not re-attempted. *(M)*
- [x] `finetune`: keep DDP loop; drop the redundant early-stop broadcast (gathered metric is identical on all ranks) — confirmed: `should_stop` is computed and checked with no `broadcast()` call at all now, for either mode; keep the prune broadcast while rank 0 owns the Optuna trial — confirmed: `should_prune` is still `broadcast`ed, but only `if not cached_probe` (i.e. only in `finetune`/distributed training). *(S)*

### Phase 3 — Controls (M)
- [ ] Implement `random_init_trunk` (fresh `from_config` model, cached embeddings under its fingerprint, same probe + HPO) and `onehot_probe`/`composition_probe` baselines; remove `untrained`/`random_head`/`random_both`; replace `random_trunk`. *(M)*
- [ ] Token-task `shuffled_context` (joint permutation); `train_median`; optional `nearest_train_neighbor` via MMseqs2 (reuse `modules/data` clustering utilities) with identity-binned reporting. *(M)*
- [ ] Drop the permutation p-value for `scrambled_labels` (it *is* the null). *(S)*

### Phase 4 — Fast scoring (S–M)
- [ ] NumPy confusion-matrix metrics per resample instead of per-metric sklearn calls (§3.3.2a); drop the permutation test for `scrambled_labels`. *(S)*
- [ ] Only if the Phase 0 benchmark says it is still too slow: per-example sufficient statistics (`per_example_stats`/`aggregate`) and `bincount` resampling. *(M)*
- [ ] Resampling unit per task (`example | sequence`). *(S)*

### Phase 5 — Contact prediction (L)
- [ ] Decide token-embedding strategy from the Phase 0 numbers: per-epoch frozen forward (no new code) vs `TokenEmbeddingStore` (memmap + offsets, per-rank shards, per-layer, optional attention maps). *(M if store)*
- [ ] `tasks/contact_prediction.py` per §5.2 (symmetric low-rank/bilinear pair head, chunked or sampled pair evaluation — never a dense `(L,L,kH)` feature tensor; `(B,S,S)` int8 labels; separation-binned P@L metrics; sparse prediction storage; `separation_prior` control; attention-LR reference). Port `contact_map_loss`/`compute_precisions` from `modules_v0/evaluation/src/apperitif/utils/metrics.py`. *(L)*

### Phase 6 — ProteinGym (M–L)
- [ ] Zero-shot: separate entry point reusing model loading + workspace; masked-marginal scorer (one masked WT copy per unique site, batched); write per-assay CSVs in the official layout; run/port the official ProteinGym scorer for metrics, UniProt-ID/function-category aggregation and its uncertainty; `random_init_trunk` and `blosum62` baselines as extra scorers. *(M)*
- [ ] Supervised (if in scope): per-assay CV folds (random / modulo / contiguous) via a `load_splits` override, on top of `tune_probe`; official supervised scorer (Spearman/MSE per scheme, averaged, then aggregated). *(M–L)*

Rough expected net effect on code size after Phases 1–3 (estimates): `tune.py` 955 → ~450, `embed.py` 844 → ~500 (incl. a store, if built), `controls.py` 687 → ~350, `heads.py` 471 → ~300, `config_validation.py` 154 → ~40; tests shrink correspondingly (the distributed-lockstep tests go away; add a 2-process CPU smoke job to CI and a 2-GPU smoke job under `jobs/`).

---

## 8. Open questions for the team

1. Is multi-node/multi-GPU **fine-tuning** (`tune_trunk=True`) a real use case, or is evaluation in practice frozen-probe + single-GPU fine-tuning? (Determines how much DDP plumbing to keep in `finetune` mode.)
2. Expected dataset scale for token-level tasks and contacts (number of sequences × length) — sizes the token store and whether dimensionality reduction should be offered.
3. Which ProteinGym tracks are in scope (substitution zero-shot, indel, supervised CV, clinical)? Each has its own data layout.
4. Should `nearest_train_neighbor`/identity-stratified reporting be part of every run (it needs MMseqs2 on the eval node) or a separate analysis script?
5. Is the `scores_long.parquet` schema consumed by anything external yet? (Adding `seed` — and `fold`, if ProteinGym-supervised is in scope — is otherwise free.)
