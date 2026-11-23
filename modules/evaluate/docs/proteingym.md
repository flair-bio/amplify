# ProteinGym

[ProteinGym](https://proteingym.org/) (Notin et al., NeurIPS 2023; Marks lab)
is the standard benchmark for **variant-effect prediction**: given a protein
and a mutation, predict how much the mutation changes the protein's function.
It collects many independent experiments, each measuring one protein's
mutants, and scores models against them with a common protocol and public
leaderboard.

Why it is in this evaluation suite:

- **It tests something the other benchmarks don't.** The other datasets ask
  what an embedding says about a *protein*. ProteinGym asks whether the model
  can tell apart near-identical sequences that differ by one or a few residues.
  That is the core question in protein engineering and in interpreting
  genetic variants.
- **Zero-shot is meaningful.** A masked language model's likelihood ratio
  between the mutant and wild-type residue is a well-established fitness
  predictor, so models can be compared without any training. Supervised
  probes are also supported.
- **Protocol-aware evaluation.** The runners implement ProteinGym-style
  scoring and aggregation. Compare with published leaderboard models only
  when the dataset release, splits, scoring method, and context limits match.

Four tracks are supported:

| Track | Each assay is | Variant label | Per-assay metric |
|---|---|---|---|
| DMS substitutions | A deep mutational scan: thousands of single (sometimes multiple) substitutions of one protein, with a measured readout such as activity, binding, stability, or growth | Continuous `dms_score` | Spearman |
| DMS indels | The same, for insertions and deletions | Continuous `dms_score` | Spearman |
| Clinical substitutions | Human disease-gene variants curated as pathogenic or benign | Binary | AUC |
| Clinical indels | The same, for insertions and deletions | Binary | AUC |

Each row is one variant (`mutated_sequence`) of an assay's wild-type sequence.
DMS scores are only comparable within an assay. That is why ProteinGym uses
per-assay variant tables rather than the standard `sequence`/`targets`
dataset contract. Both dedicated runners use the shared `run_evaluate.py`
entry point. Prepared tracks are listed in the
[dataset catalog](datasets.md#separate-dataset-workflows).

- [Prepared inputs](#prepared-inputs)
- [Zero-shot scoring](#zero-shot-scoring)
- [Supervised evaluation](#supervised-evaluation)
- [Execution flow](#execution-flow)
- [Outputs](#outputs)
- [Metrics and aggregation](#metrics-and-aggregation)

## Prepared inputs

Dataset sourcing, normalization, and publication are maintained in
[amplify-downstream-data-prep](https://github.com/flair-bio/amplify-downstream-data-prep/tree/main).
This module only loads prepared artifacts and runs evaluations.

The workspace presets expect:

```text
<suite root>/
  proteingym-dms_substitutions/
    reference.parquet
    data/
      <dms_id>.parquet
      ...
```

`reference.parquet` identifies each assay with `dms_id`, `uniprot_id`, and
`coarse_selection_type`; zero-shot scoring also uses its wild-type
`target_seq`. The `is_indel` and `is_clinical` flags select track-specific
scoring and label handling.

Each assay file contains `mutated_sequence` and either continuous `dms_score`
or binary `dms_score_bin` labels. Substitution zero-shot scoring uses mutation
codes in `mutant`, such as `A42V` or `A42V:G80D`. Supervised evaluation also
requires the fold column or persisted `split` labels selected by its config.
Combined tables are not read by these runners.

The shipped presets use `flair-bio/evaluation-suite` and the
`proteingym-dms_substitutions` subfolder. To select another prepared track,
override **both** `workspace.dataset_name` (the output label) and
`workspace.dataset_config_name` (the input subfolder), for example:

```text
workspace.dataset_name=proteingym-dms_indels
workspace.dataset_config_name=proteingym-dms_indels
```

`workspace.dataset_repo_id` may instead point to a local **suite root**;
`workspace.dataset_revision` must then be null. For Hub inputs, set
`workspace.dataset_revision` to a commit SHA to pin the snapshot; only the
selected subfolder is downloaded. Missing suite folders or unavailable Hub
snapshots fail rather than silently switching sources. Individual missing
assay files are warned about and skipped; check logs before interpreting
benchmark coverage. `max_assays=N` limits evaluation to the first N reference
rows, not a random sample.

## Zero-shot scoring

Zero-shot scoring uses masked-marginal scoring for substitutions and
pseudo-likelihood scoring for indels; the runner selects the latter from the
prepared `is_indel` flag. Because it does not train, zero-shot scoring does not
require train/validation/test splits:

```bash
uv run python -m modules.evaluate.src.run_evaluate \
    modules/evaluate/configs/proteingym_zero_shot.yaml \
    workspace.model_repo_id=flair-bio/amplify-120m
```

All evaluation commands use `run_evaluate.py`. The ProteinGym presets declare
`workflow: proteingym_zero_shot` or `workflow: proteingym_supervised`, selecting
assay-specific functions with the shared runtime and artifact services rather
than the ordinary flat-data workflow. Use these presets directly, without
the ordinary base, task, or model YAML overlays.

For example, launch on a single GPU with a local suite and checkpoint:

```bash
accelerate launch \
    --config_file modules/evaluate/configs/accelerate/local.yaml \
    --num_processes 1 \
    --num_cpu_threads_per_process 4 \
    modules/evaluate/src/run_evaluate.py \
    modules/evaluate/configs/proteingym_zero_shot.yaml \
    workspace.model_name=amplify-120m_checkpoint_100000 \
    workspace.model_repo_id=/checkpoints/checkpoint_100000 \
    workspace.dataset_repo_id=/datasets/evaluation-suite \
    workspace.dataset_revision=null \
    workspace.output_dir=/results \
    workspace.run_id=run01 \
    seed=710019
```

Both runners require one process; this launch does not shard assays across GPUs.

Both ProteinGym runners enable TF32 for CUDA FP32 operations by default
(`tf32: true`), matching the standard evaluation runner. Append `tf32=false` to
disable it. This controls internal CUDA computation independently of `dtype`;
TF32 retains FP32 tensors but can change numerical scores.

The zero-shot runner disables ambiguous-residue removal (for example, `U`) so
each source residue retains its token and coordinate. Tokenizers that cannot
represent one token per residue raise an error rather than silently shifting
mutation positions.

For a substitution `A42V`, the score is
`log P(V | position 42 masked) - log P(A | position 42 masked)`.
Each unique mutated position is evaluated once per assay, reusing its
distribution across variants. Multi-mutant scores sum these differences,
each calculated in the wild-type context. Higher scores mean greater
model preference relative to wild type, not a calibrated experimental fitness.
For long proteins, substitutions use a window centered on the mutated
position within the `max_length` token budget, including special tokens.

Indels use the mutant's summed masked pseudo-log-likelihood minus the
wild type's: each retained residue is scored with its own token masked.
For indels, sequences longer than `max_length` use the leftmost residues that
fit alongside the tokenizer's added special tokens. The crop is deterministic;
BOS/CLS and EOS remain context and are never scored as residues. This limits
indel scoring to the retained prefix, rather than a full-sequence sliding-window
pseudo-likelihood.

The zero-shot runner can also score reference baselines with
`baselines: [random, blosum62, random_init_trunk]`. `blosum62` is a simple
substitution-only matrix baseline, not ProteinGym's official MSA-derived
Site-Independent scorer; it returns undefined scores for indels.
`random_init_trunk` initializes the entire MLM from its architecture config,
including the prediction head. In workspace runs, all conditions share
`predictions.parquet` and `scores.parquet`, distinguished by `control`.
These scorers use `baselines`, not the ordinary CLI `--baselines` flag.

## Supervised evaluation

Supervised evaluation fits a separate probe for each assay and training fold,
using embeddings from a backbone that is not fine-tuned. The final hidden
states are mean-pooled over non-padding tokens (including boundary special
tokens). Tokenization enables truncation at `max_length`; embeddings are
computed once per assay and reused across folds, without a persistent cache.
Choose `probe_type=linear|mlp|torch_linear`; `probe_search_space` performs a grid
search on an inner holdout of each training fold before refitting the winner on
all available training rows. Splits can use an official fold column
(`fold_random_5`, `fold_modulo_5`, or `fold_contiguous_5`),
`split=official_average` to average applicable official schemes equally,
`split=seeded` to use persisted train/validation/test labels, or
`split=custom_kfold` for a reproducible generated k-fold split:

```bash
uv run python -m modules.evaluate.src.run_evaluate \
    modules/evaluate/configs/proteingym_supervised.yaml \
    workspace.model_repo_id=flair-bio/amplify-120m
```

The preset defaults to `probe_type=linear` (Ridge for continuous targets,
LogisticRegression for binary targets), `split=fold_random_5`, and no
hyperparameter search. Each held-out fold is predicted by a fresh probe
fitted only on the other folds. Standardization uses the outer training rows,
not the held-out fold.

| Split | Behavior |
|---|---|
| `fold_random_5`, `fold_modulo_5`, `fold_contiguous_5` | Use the corresponding prepared fold column; return out-of-fold predictions |
| `official_average` | Run all three official schemes for substitutions, or only random folds for indels; report the mean of the scheme metrics |
| `seeded` | Fit on persisted `train` rows and predict only `test` rows; persisted `validation` rows are not used |
| `custom_kfold` | Generate shuffled folds using `seed` and `custom_kfold_splits` (default 5) |

For `official_average`, saved predictions are averaged across schemes, but
the reported assay score is the **mean of the separate scheme scores**, not
the metric of those averaged predictions.

Official fold columns must be present in each assay; the evaluator does not
generate or download them. Missing columns raise an error. Verify the prepared
artifact's schema and provenance before selecting a scheme. Generated random
folds or persisted random splits are not equivalent to official position-based
splits.

**Current implementation limitation:** `pool_embeddings()` does not disable
gradients before converting embeddings to NumPy. With a normal trainable
backbone this raises a `RuntimeError`; embedding extraction needs a
`torch.no_grad()` context. This is separate from the dataset migration.

## Execution flow

```text
run_evaluate.py::main()
  -> load_run_config()
  -> initialize_runtime()
  -> run_zero_shot() or run_supervised()
       -> reserve outputs and resolve the prepared suite
       -> load tokenizer and model
       -> run_assay_loop()
            -> load one assay
            -> zero-shot scores OR embeddings and fold-local probe fitting
            -> calculate assay metrics
       -> aggregate metrics and publish results
```

[`proteingym/runner.py`](../src/proteingym/runner.py) owns the shared assay
loop. [`zero_shot.py`](../src/proteingym/zero_shot.py) and
[`supervised.py`](../src/proteingym/supervised.py) supply the scoring behavior.
These workflows bypass the ordinary task registry and
[`runners/pipeline.py`](../src/runners/pipeline.py).

## Outputs

Both workspace runners produce three files per seed, matching the regular
evaluator's artifact names:

- `predictions.parquet`: variants from all assays and requested conditions,
  with source columns retained, `dms_id`, `control`, `prediction`, and `label`.
  Existing IDs are preserved; absent IDs use `<dms_id>:<row_index>`.
  Clinical binary labels are normalized so higher values mean benign.
- `scores.parquet`: long-form `metric`/`value` rows with `control` and
  `scope` (`assay`, `function`, or `overall`). Assay rows carry their assay,
  protein, function, and variant count. DMS overall rows use hierarchical
  aggregation, not a pooled variant correlation; clinical indels use pooled
  AUC.
- `run_manifest.json`: configuration, resolved inputs, run status, aggregate
  summaries (including bootstrap errors when enabled), and artifact paths.

The flattened hierarchy matches the regular evaluator:

```text
<workspace.output_dir>/proteingym/<dataset_name>/<model_name>/
  <variant>-<run_id>/seed_<seed>/
```

Zero-shot uses variant `zero_shot`; supervised uses
`<probe_type>-frozen_trunk-<split>`. Logs go to the console, not a separate
run file. A run lock prevents concurrent writes.
Existing run outputs are rejected before model loading; pass `--overwrite`
to replace that run while preserving sibling seeds and variants, or choose
a new `workspace.run_id`. Sibling seeds must use the same configuration and
resolved inputs. There is no per-assay resume.

Workspace identity and sibling-seed experiment metadata live inside each
seed manifest; no workspace/experiment JSON sidecars are created.
`score.scores` contains the primary aggregate, including per-function means.
`score.condition_scores` contains `primary` and each requested zero-shot
baseline/control, with `scores`, `predictions_path`, and `scores_path`.
All conditions share the two Parquet files, distinguished by `control`
(`primary`, `random`, `blosum62`, or `random_init_trunk`); there are no per-assay
files or baseline subdirectories in workspace runs. These existing ProteinGym
scorers are not the ordinary pipeline's shared dataset baselines. No shared
baseline artifacts are changed.

`workspace.tmp_path` is a scratch **root**, with coordination locks under
`<tmp_path>/proteingym/<dataset_name>/<model_name>/locks/<output-root-hash>/`.
When omitted, the root defaults to `<workspace.output_dir>/tmp`.
The shared workspace also defines `embeddings/` under that model/dataset
scratch path; the current ProteinGym supervised runner computes pooled
embeddings in memory rather than using a persistent embedding cache.
Zero-shot does not extract embeddings. Checkpoint loading remains separate
from all output/cache paths and uses `workspace.model_repo_id`.
Neither runner saves fitted probe checkpoints.

Inspect a workspace run's primary result:

```python
import json
from pathlib import Path
import polars as pl

manifest = json.loads(Path("/results/path/to/seed_710019/run_manifest.json").read_text())
print(manifest["score"]["scores"])
scores = pl.read_parquet(manifest["score"]["scores_path"])
print(scores.filter((pl.col("scope") == "assay") & (pl.col("control") == "primary")))
```

Existing flat configs containing `model_id`, `data_dir`, `output_dir`, and
optional `data_repo_id` remain supported with their original direct paths
and overwrite behavior, including per-assay `<dms_id>_scored.parquet`,
`summary.json`, and `assay_scores.parquet`,
without workspace manifests or locking. Do not mix
flat input/output overrides with workspace settings. To use the shipped
preset in legacy mode, explicitly set `workspace=null` and provide all three
flat paths. A standalone artifact repo can still be read through legacy
`data_repo_id`; the workspace presets instead expect a suite with a named
subfolder.

## Metrics and aggregation

The primary per-assay metric is Spearman when `dms_score` is present, otherwise
ROC-AUC from `dms_score_bin`. Clinical labels are flipped so larger scores
correspond to benign variants. Zero-shot DMS runs additionally report NDCG
and top-decile recall for continuous labels, and AUC and median-threshold MCC
when binary labels are available. Supervised runs report only the primary
metric; they do not currently publish MSE.

Aggregation follows the track's benchmark protocol:

| Track | Summary aggregation |
|---|---|
| DMS substitutions and indels | Mean over assays within each protein/function pair, then over proteins within each function, then equally over functions |
| Clinical substitutions | Equal mean of per-gene AUCs; genes without both classes among scored variants have undefined AUC and are excluded from the mean |
| Clinical indels | One AUC over all scored variants pooled across genes, including genes with only one label class |

See the official [DMS evaluation code](https://github.com/OATML-Markslab/ProteinGym/blob/main/proteingym/performance_DMS_benchmarks.py)
and [clinical evaluation code](https://github.com/OATML-Markslab/ProteinGym/blob/main/proteingym/performance_clinical_benchmarks.py).
Clinical substitutions report `aggregation: per_gene` and the number of genes
with a finite metric in `num_genes`; clinical indels report `aggregation: pooled`.
Undefined metrics remain in `scores.parquet` assay rows (or legacy
`assay_scores.parquet`) for inspection but do not
contribute to averages. If no metric is defined, the aggregate is NaN.

Set `compute_uncertainty: true` to add bootstrap standard errors for DMS and
clinical substitution means; clinical indels do not report a bootstrap error.
`n_bootstrap` controls primary-condition resamples for either runner and
defaults to `10000`; zero-shot baseline conditions do not currently receive
the uncertainty settings. The clinical substitution bootstrap resamples genes.
DMS bootstrap resamples proteins within each function category, not variant
rows. DMS bootstrap errors measure uncertainty in this model's aggregate,
whereas the official leaderboard
reports uncertainty relative to the best model. Metrics retain full precision
rather than reproducing the leaderboard's intermediate rounding.
