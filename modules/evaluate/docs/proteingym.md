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
- **Comparable numbers.** The scoring and aggregation follow ProteinGym's own
  protocol, so results can be placed next to published leaderboard models.

Four tracks are supported:

| Track | Each assay is | Variant label | Per-assay metric |
|---|---|---|---|
| DMS substitutions | A deep mutational scan: thousands of single (sometimes multiple) substitutions of one protein, with a measured readout such as activity, binding, stability, or growth | Continuous `DMS_score` | Spearman |
| DMS indels | The same, for insertions and deletions | Continuous `DMS_score` | Spearman |
| Clinical substitutions | Human disease-gene variants curated as pathogenic or benign | Binary | AUC |
| Clinical indels | The same, for insertions and deletions | Binary | AUC |

Each row is one variant (`mutated_sequence`) of an assay's wild-type sequence.
DMS scores are only comparable within an assay. That is why ProteinGym uses
per-assay variant tables rather than the standard `sequence`/`targets`
dataset contract, and has its own sourcing script and two dedicated runners
instead of `run_evaluate.py`. Hub artifacts are listed in the
[dataset catalog](datasets.md#separate-dataset-workflows).

- [Sourcing](#sourcing)
- [Zero-shot scoring](#zero-shot-scoring)
- [Supervised evaluation](#supervised-evaluation)
- [Outputs](#outputs)

## Sourcing

Sourcing follows the method in the
[ProteinGym GitHub README](https://github.com/OATML-Markslab/ProteinGym): the
reference CSV comes from the GitHub repo (the `main` branch by default; set
`reference_base_url` to a commit to pin it), and each track's assays come from one
zip hosted at `<base_url>/ProteinGym_<version>/<track>.zip` (for example,
`https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_substitutions.zip`).
The full archive for the selected track is downloaded and extracted, and the
source converts every assay listed in that track's reference file. The sourcing
script has no assay-subset option. Select a track with
`track: dms_substitutions | dms_indels | clinical_substitutions |
clinical_indels` in [`source_proteingym.yaml`](../configs/dataset_sourcing/source_proteingym.yaml).

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
a different path to opt back into a shared directory across tracks. The
checked-in YAML sets `overwrite_output: true`, so rerunning it replaces
existing output artifacts.

Sourcing always writes one parquet file per assay under `data/`, preserving
ProteinGym's official fold columns (`fold_random_5`, `fold_modulo_5`, and
`fold_contiguous_5`) when present. The checked-in YAML enables
`combine_assays: true`, which writes `data/combined.parquet` while keeping the
per-assay files available. With `create_split_subsets: true`, sourcing writes
reproducible per-assay `train`/`validation`/`test` labels using a seeded 80/10/10
split, plus canonical and pooled random/stratified combined split files. The
pooled stratified view balances each assay's share of every split; raw DMS
scores are not comparable across assays, so they are not used as strata. The
seeded splits are uniform random over variants, so the same mutated position
appears in several splits. These reuse the same partition helpers
as the other dataset sources. MMseqs cluster subsets are intentionally omitted:
within one assay every row is a single/double mutant of the same wild-type
sequence, so sequence-identity clustering does not provide a meaningful
partition. Use the official `fold_contiguous_5` or `fold_modulo_5` columns for
a position-based generalization split instead. Official folds are retained in
the artifacts for evaluation.

Assays with fewer than three variants are retained in their per-assay and
combined artifacts but do not receive synthetic split labels or appear in the
pooled split views, because three nonempty splits cannot be formed from fewer
than three observations.

## Zero-shot scoring

Zero-shot scoring uses masked-marginal scoring for substitutions and
pseudo-likelihood scoring for indels; the runner selects the latter from the
sourced `is_indel` flag. Because it does not train, zero-shot scoring does not
require train/validation/test splits:

```bash
uv run modules/evaluate/src/run_proteingym_zero_shot.py \
    modules/evaluate/configs/proteingym_zero_shot.yaml \
    model_id=flair-bio/amplify-120m
```

Both ProteinGym runners enable TF32 for CUDA FP32 operations by default
(`tf32: true`), matching the standard evaluation runner. Append `tf32=false` to
disable it. This controls internal CUDA computation independently of `dtype`;
TF32 retains FP32 tensors but can change numerical scores.

The zero-shot runner disables ambiguous-residue removal (for example, `U`) so
each source residue retains its token and coordinate. Tokenizers that cannot
represent one token per residue raise an error rather than silently shifting
mutation positions.

For indels, sequences longer than `max_length` use the leftmost residues that
fit alongside the tokenizer's added special tokens. The crop is deterministic;
BOS/CLS and EOS remain context and are never scored as residues. This limits
indel scoring to the retained prefix, rather than a full-sequence sliding-window
pseudo-likelihood.

The zero-shot runner can also score reference baselines with
`baselines: [random, blosum62, random_init_trunk]`. `blosum62` is a simple
substitution-only matrix baseline, not ProteinGym's official MSA-derived
Site-Independent scorer. Baseline outputs are written under
`output_dir/baselines/<name>/`.

## Supervised evaluation

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
    model_id=flair-bio/amplify-120m
```

## Outputs

Both runners write `<dms_id>_scored.parquet` (each variant's `model_score`),
`assay_scores.parquet` (per-assay Spearman for continuous DMS scores or AUC for
binary clinical labels), and `summary.json` under `output_dir`.

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
Undefined metrics remain in `assay_scores.parquet` for inspection but do not
contribute to averages. If no metric is defined, the aggregate is NaN.

Set `compute_uncertainty: true` to add bootstrap standard errors for DMS and
clinical substitution means; clinical indels do not report a bootstrap error.
`n_bootstrap` controls the resamples for either runner and defaults to `10000`.
The clinical substitution bootstrap resamples genes. DMS bootstrap errors
measure uncertainty in this model's aggregate, whereas the official leaderboard
reports uncertainty relative to the best model. Metrics retain full precision
rather than reproducing the leaderboard's intermediate rounding.
