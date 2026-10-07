# Dataset Sourcing

This guide covers dataset sourcing for the evaluation module: the shared
Hugging Face repository pipeline, source-specific scripts, output artifacts,
and how to make sourced datasets available to evaluation. The main module
workflow is documented in the [evaluation README](../../README.md).

## Table of Contents

- [Pipeline Overview](#pipeline-overview)
- [Shared Configuration](#shared-configuration)
- [Shipped YAML Defaults](#shipped-yaml-defaults)
- [Adding a New Source](#adding-a-new-source)
- [Example: Biomap-Research](#example--sourcing-a-biomap-research-dataset)
- [Example: TAPE ProteinNet](#example--sourcing-tape-proteinnet)
- [UniProt Gene Ontology](#uniprot-gene-ontology-annotations)
- [UniProt Residue Annotations](#uniprot-residue-annotations)
- [InterPro Labels](#interpro-labels)
- [UniProt ProtParam](#uniprot-protparam-regression-properties)
- [ProteinGym Dataset Sourcing](#proteingym-dataset-sourcing)

## Pipeline Overview

**Purpose:** before a dataset can be evaluated, it needs to exist in the shape
the [Prepare](../../README.md#step-1--prepare) step expects (a
`sequence`/`targets` Hugging Face dataset with `train`/`validation`/`test`
splits). [`pipeline.py`](pipeline.py) provides a shared download → preprocess
→ upload flow specifically for sources that already publish datasets as
Hugging Face Hub repositories; the Biomap-research source is its concrete
example. Sources that start from raw APIs or files and need custom target
construction use their own `run()` flow, while reusing shared sourcing helpers
where appropriate.

```
modules/evaluate/src/dataset_sourcing/
├── downloads.py   ← shared UniProt query construction and source-file download helpers
├── config.py      ← DatasetSourcingConfig: Pydantic schema for a sourcing run + upload repo-id resolution
├── spec.py        ← DatasetSourceSpec: per-source repo-id template + column renames
├── pipeline.py    ← HF-repository sources: resolve_datasets()/run_for_dataset()/run_cli()
├── hf_io.py       ← download_dataset()/upload_dataset_artifact(): HF Hub download/upload calls
├── preprocess.py  ← load_splits()/minimal_preprocess(): locate/load raw splits, then rename/filter/truncate
└── outputs.py     ← write_hf_split_files()/write_stats()/write_seqkit_stats()/write_dataset_card()
```

For a source handled by this HF-repository pipeline, each dataset goes through:

```
Download (HF Hub) → Preprocess (rename/split/truncate) → Stats + dataset card → Upload (HF Hub)
```

Raw downloads land under `<work_dir>/downloads/<dataset>/`, and processed
artifacts (split parquet files, `stats.json` with row counts, columns, and
SeqKit metrics, and `README.md`) are written to
`<work_dir>/outputs/<dataset>/`, which is also what gets uploaded. Pinned source
downloads are kept in a separate `<dataset>/<source_revision>/` subdirectory so
previously downloaded snapshots cannot mix.

This output layout describes `preprocess: minimal`, which all checked-in
HF-repository configs use. With `preprocess: none`, the downloaded source
snapshot is uploaded as-is with a generated card and stats.

## Shared Configuration

**Sequence length:** `max_sequence_length` is optional; `null` retains all
sequence lengths. It is supported by the shared HF-repository pipeline, TAPE
ProteinNet, and the UniProt-backed GO, residue-annotation, InterPro, and
ProtParam sources. When set, the cutoff is applied before generated split
assignment and MMseqs2 clustering. ProteinGym does not expose this setting;
its partitions are based on assay variants.

**UniProt scope:** the GO, residue-annotation, ProtParam, and InterPro configs
select `organism_id` (`null` for all organisms; `9606` for human) and
`review_status`: `reviewed` selects Swiss-Prot, `unreviewed` selects TrEMBL,
and `all` selects both. The shipped defaults are human and reviewed.

**Split subsets:** where supported, `create_split_subsets: true` adds pooled
random and stratified splits without changing the canonical splits. Sequence
datasets also get hold-cluster-out splits when enough MMseqs clusters are
available. ProteinGym instead creates seeded per-assay splits and pooled
random/stratified views, with cluster partitioning disabled.

## Shipped YAML Defaults

The checked-in YAMLs below are the defaults used by the source commands in this
guide. All leave `repo_owner` and `repo_id` unset, so they create local artifacts
without uploading. Set an upload target explicitly when publishing is intended.

| Config | Default selection | `work_dir` | Notable settings |
|---|---|---|---|
| [`source_biomap_datasets.yaml`](../../configs/dataset_sourcing/source_biomap_datasets.yaml) | All 12 Biomap-research datasets | `eval_datasets/biomap_research_sourcing` | `create_split_subsets: true`; no sequence-length cutoff |
| [`source_interpro.yaml`](../../configs/dataset_sourcing/source_interpro.yaml) | All 7 InterPro groups | `eval_datasets/interpro_sourcing` | Human reviewed UniProtKB; `min_label_count: 2`; split subsets enabled |
| [`source_proteingym.yaml`](../../configs/dataset_sourcing/source_proteingym.yaml) | `dms_substitutions` | `eval_datasets/proteingym_sourcing` | `combine_assays: true`; seeded 80/10/10 splits; `overwrite_output: true` |
| [`source_tape_proteinnet.yaml`](../../configs/dataset_sourcing/source_tape_proteinnet.yaml) | Fixed `proteinnet_contact_prediction` dataset | `eval_datasets/tape_proteinnet_sourcing` | Reads staged LMDB splits from `downloaded_datasets`; split subsets enabled |
| [`source_uniprot_annotations.yaml`](../../configs/dataset_sourcing/source_uniprot_annotations.yaml) | All 12 residue-annotation groups | `eval_datasets/uniprot_annotations_sourcing` | Human reviewed UniProtKB; split subsets enabled |
| [`source_uniprot_go.yaml`](../../configs/dataset_sourcing/source_uniprot_go.yaml) | All 3 GO aspects | `eval_datasets/uniprot_go_sourcing` | Human reviewed UniProtKB; derived label-count threshold; split subsets enabled; `overwrite_output: true` |
| [`source_uniprot_protparams.yaml`](../../configs/dataset_sourcing/source_uniprot_protparams.yaml) | Fixed `uniprot_protparams` dataset | `eval_datasets/uniprot_protparams_sourcing` | Human reviewed UniProtKB; 80/10/10 canonical splits and split subsets enabled |

The ProteinGym source appends the selected track to its default `work_dir` at
runtime, so the checked-in `dms_substitutions` config writes under
`eval_datasets/proteingym_sourcing/dms_substitutions/`. The GO and ProteinGym
YAMLs set `overwrite_output: true`, so rerunning them replaces the corresponding
existing output artifacts.

## Adding a New Source

If the source already provides an HF dataset repository with splits that need
only the generic renaming/preprocessing, subclass `DatasetSourcingConfig`,
define a `DatasetSourceSpec`, and call
`run_cli(sys.argv[1:], model_cls=..., spec=...)`. This path downloads the HF
repository and is demonstrated by
[`source_biomap_datasets.py`](../../scripts/dataset_sourcing/source_biomap_datasets.py).

If the source is a raw API or file, or requires source-specific parsing,
target/vocabulary construction, or split logic, give its script a
source-specific `run(config)` flow instead. Such scripts can still reuse
helpers from `dataset_sourcing` for config loading, dataset selection,
preprocessing, output writing, stats, and upload;
[`source_uniprot_go.py`](../../scripts/dataset_sourcing/source_uniprot_go.py)
and [`source_interpro.py`](../../scripts/dataset_sourcing/source_interpro.py)
are examples.

## Example — sourcing a Biomap-research dataset

[`source_biomap_datasets.yaml`](../../configs/dataset_sourcing/source_biomap_datasets.yaml)
sets `all_datasets: true`, so the default command processes all 12 datasets
listed in that file. With no `repo_owner` or `repo_id`, it only builds local
artifacts under `eval_datasets/biomap_research_sourcing/outputs/`:

```bash
uv run python modules/evaluate/scripts/dataset_sourcing/source_biomap_datasets.py \
    modules/evaluate/configs/dataset_sourcing/source_biomap_datasets.yaml
```

To upload the same 12 datasets to one repository per dataset, set
`repo_owner` (the `all_datasets=true` override is explicit but already matches
the checked-in YAML):

```bash
uv run python modules/evaluate/scripts/dataset_sourcing/source_biomap_datasets.py \
    modules/evaluate/configs/dataset_sourcing/source_biomap_datasets.yaml \
    all_datasets=true \
    repo_owner=<your-hf-username-or-org>
```

This uploads each dataset to `<repo_owner>/biomap-research-<dataset_name>` (per
`repo_name_prefix` in the config), one HF dataset repo per dataset
(`upload_layout: per_dataset_repo`).

Biomap preprocessing preserves the downloaded splits and, when
`create_split_subsets: true` is set in the sourcing YAML, publishes three
additional deterministic split methods. If the download lacks validation
data, validation is held out from training along MMseqs cluster boundaries
when MMseqs is available, with seeded random sampling as a fallback.

For a reproducible source snapshot, set
`source_revision=<upstream-commit-sha>` when sourcing. This pins the downloaded
Hugging Face dataset and is recorded in its generated card; without it, the
default branch remains unpinned. The generated card links to the upstream card
for label definitions, license, and citation, and distinguishes the canonical
rows from alternate partitions.

| Method | Hugging Face splits | Behavior |
|---|---|---|
| Downloaded (default) | `train`, `validation`, `test` | Preserves source assignments, except synthesized validation when absent |
| Random | `train_random`, `validation_random`, `test_random` | Seeded uniform row partition |
| Stratified | `train_stratified`, `validation_stratified`, `test_stratified` | Balances categorical labels, quantile-binned continuous targets, or dominant token labels |
| Hold-cluster-out | `train_cluster`, `validation_cluster`, `test_cluster` | Assigns each complete MMseqs sequence cluster to exactly one subset; no within-cluster stratification |

The downloaded method is the default evaluation input. Pooled cluster
generation requires MMseqs. Selecting another method means setting
`steps.prepare.split.name`, which resolves the three splits to
`train_<name>`/`validation_<name>`/`test_<name>`:

```bash
uv run accelerate launch modules/evaluate/src/run_evaluate.py \
    modules/evaluate/configs/config.yaml \
    modules/evaluate/configs/models/amplify_120m.yaml \
    modules/evaluate/configs/tasks/sequence_classification.yaml \
    modules/evaluate/configs/datasets/metal_ion_binding.yaml \
    steps.prepare.split.name=stratified
```

If the dataset doesn't publish those splits, Prepare fails with the list of
splits it did find. Datasets with benchmark-specific split names set the roles
explicitly; see [Named benchmark splits](../../README.md#named-benchmark-splits).

## Example — sourcing TAPE ProteinNet

[`source_tape_proteinnet.py`](../../scripts/dataset_sourcing/source_tape_proteinnet.py)
converts the TAPE ProteinNet LMDB splits into the same
`sequence`/`targets` dataset contract used by the contact-prediction task. The
converter derives contacts from valid residue coordinates, writes split
artifacts and statistics, and can publish the result as a Hugging Face dataset.

The checked-in config reads pre-staged LMDB split directories under
`downloaded_datasets/` and creates split subsets. This script always processes
the fixed `proteinnet_contact_prediction` dataset; its `all_datasets` and
`datasets` config fields do not change that selection. Invoke the Python script
with the config and overrides directly; `uv` installs the `lmdb` dependency
for this source:

```bash
uv run --with lmdb modules/evaluate/scripts/dataset_sourcing/source_tape_proteinnet.py \
    modules/evaluate/configs/dataset_sourcing/source_tape_proteinnet.yaml \
    source_folder=/path/to/proteinnet \
    repo_owner=<your-hf-username-or-org>
```

The converter still contains an explicit archive fallback for compatibility,
but the checked-in workflow and generated dataset provenance use the TAPE LMDB
source. Set `max_sequence_length` to filter longer sequences before any optional
random, stratified, or MMseqs2 cluster split subsets are generated.

## UniProt Gene Ontology annotations

[`source_uniprot_go.py`](../../scripts/dataset_sourcing/source_uniprot_go.py)
builds protein-level annotation datasets from UniProtKB's three Gene Ontology
(GO) aspects. Each aspect is a separate prediction task:

| Config selection | Dataset name | Meaning |
|---|---|---|
| `molecular_function` | `GO_mf` | The molecular activity or function of a protein, such as binding or catalysis. |
| `biological_process` | `GO_bp` | The broader biological process in which a protein participates. |
| `cellular_component` | `GO_cc` | The cellular location or macromolecular complex associated with a protein. |

Each output contains protein `sequence` values and multi-hot `targets` vectors;
`label_vocabulary.json` maps vector positions to GO identifiers and term names.
These labels reflect UniProtKB annotations, not experimental evidence of
causality. The source has no upstream train/validation/test splits, so it
assigns whole MMseqs2 sequence clusters to the 80/10/10 splits.

The checked-in config selects all three aspects for human reviewed UniProtKB
proteins and writes local artifacts under
`eval_datasets/uniprot_go_sourcing/outputs/`. Source those selections locally
with:

```bash
uv run modules/evaluate/scripts/dataset_sourcing/source_uniprot_go.py \
    modules/evaluate/configs/dataset_sourcing/source_uniprot_go.yaml
```

Source all three aspects with:

```bash
uv run modules/evaluate/scripts/dataset_sourcing/source_uniprot_go.py \
    modules/evaluate/configs/dataset_sourcing/source_uniprot_go.yaml \
    all_datasets=true
```

To select specific aspects, set `all_datasets=false` and list them, for example
to source molecular function and biological process:

```bash
uv run modules/evaluate/scripts/dataset_sourcing/source_uniprot_go.py \
    modules/evaluate/configs/dataset_sourcing/source_uniprot_go.yaml \
    all_datasets=false \
    'datasets=[molecular_function,biological_process]'
```

The default query uses reviewed human UniProtKB proteins (`organism_id: 9606`).
GO terms annotated to fewer than the configured `min_label_count` proteins are
omitted; by default this threshold is derived from the split fractions to
target at least ten positive examples per term in the smallest split. Add
`repo_owner=<your-hf-username-or-org>` to upload each selected aspect as its
own Hugging Face dataset.

## UniProt residue annotations

[`source_uniprot_annotations.py`](../../scripts/dataset_sourcing/source_uniprot_annotations.py)
downloads UniProtKB feature annotations for the configured organism and review
status (human/reviewed by default) and prepares residue-level classification
datasets for secondary structure, post-translational modifications,
glycosylation, phosphorylation, lipidation, membrane passage, topology,
peptides, functional sites, domains, regions, and structures. Each artifact
contains integer per-residue `targets` and a `label_vocabulary.json`.
Overlapping annotations are joined into deterministic composite labels. A
feature block is labeled only when its own `/evidence` qualifier contains
`ECO:0000269` or `ECO:0000305`; secondary-structure features also accept
PDB-derived `ECO:0007829|PDB:<structure ID>` evidence. Missing, malformed, and
other evidence is excluded. Proteins with only evidence-rejected candidate
features are excluded from the negative pool, while genuinely feature-free
proteins may be sampled as negatives. Background residues on retained proteins
are an assumed unannotated class, not experimentally verified absence.

The checked-in config builds all supported groups locally. Add
`repo_owner=<your-hf-username-or-org>` to upload each dataset, or select one
group with `all_datasets=false datasets=[uniprot_secondary_structure]`:

```bash
uv run modules/evaluate/scripts/dataset_sourcing/source_uniprot_annotations.py \
    modules/evaluate/configs/dataset_sourcing/source_uniprot_annotations.yaml
```

Set `max_sequence_length=512` to apply the sequence cutoff before cluster-based
split assignment. Dataset contents reflect the configured UniProt query,
selected annotation group, and MMseqs2 split pipeline.

Every source writes a `README.md` dataset card beside its processed data. The
card records the effective query, review and organism filters, split settings,
label vocabulary, and row counts. Re-run the relevant source after changing a
YAML filter so local and uploaded cards reflect the new source snapshot.

## InterPro labels

[`source_interpro.py`](../../scripts/dataset_sourcing/source_interpro.py)
uses the official InterPro `entry.list` download for entry names/types and
UniProtKB's `xref_interpro` export for protein sequences and entry memberships
matching the configured organism and review status. It builds the seven
configured protein-level label groups: active site, binding site, conserved
site, domain, family, homologous superfamily, and repeat. Each dataset stores
a multi-hot `targets` vector and its ordered `label_vocabulary.json`.

The checked-in config builds all seven groups locally for reviewed human
UniProtKB proteins, with a minimum of two proteins per retained label and
additional random, stratified, and cluster split subsets. Add
`repo_owner=<your-hf-username-or-org>` to upload them, or select a category
with `all_datasets=false datasets=[interpro_domain]`:

```bash
uv run modules/evaluate/scripts/dataset_sourcing/source_interpro.py \
    modules/evaluate/configs/dataset_sourcing/source_interpro.yaml
```

The protein-entry mapping comes from UniProt's export rather than downloading
InterPro's full multi-gigabyte `protein2ipr.dat.gz`; the InterPro label metadata
is sourced from the EBI release `entry.list`. Splits use the current MMseqs2
cluster pipeline.

## UniProt ProtParam regression properties

[`source_uniprot_protparams.py`](../../scripts/dataset_sourcing/source_uniprot_protparams.py)
builds one dataset, `uniprot_protparams`, with `id`, `sequence`, `split`, and
31 named numeric property columns from the configured UniProtKB subset (human
and reviewed by default). There is no `targets` column: select one property
per scalar regression run with `steps.prepare.label_column`. These properties
are derived from sequence or reported by UniProt, not independent experimental
measurements. No filter is applied based on UniProt-reported protein length.
If `max_sequence_length` is set, longer sequences are filtered before MMseqs2
clustering so split assignment reflects the retained dataset; the output name
then includes the cutoff.

| Feature | Meaning |
|---|---|
| `length` | UniProt-reported sequence length in amino acids. |
| `mass` | UniProt-reported molecular mass in daltons. |
| `percent_<aa>` | Percentage composition for each canonical amino acid: `a`, `c`, `d`, `e`, `f`, `g`, `h`, `i`, `k`, `l`, `m`, `n`, `p`, `q`, `r`, `s`, `t`, `v`, `w`, and `y`. |
| `fraction_helix_aas`, `fraction_turn_aas`, `fraction_sheet_aas` | ProtParam estimates of the sequence fractions assigned to alpha helix, turn, and beta sheet; these are predictions, not experimentally observed structures. |
| `instability_index` | ProtParam sequence-based instability index; values above 40 conventionally indicate a potentially unstable protein. |
| `gravy` | Grand average of hydropathicity across the sequence; positive values indicate greater average hydrophobicity. |
| `isoelectric_point` | Theoretical pH at which the protein has zero net charge. |
| `charge_at_ph4_7`, `charge_at_ph7_2`, `charge_at_ph8` | ProtParam estimates of net charge at pH 4.7, 7.2, and 8.0, respectively. |

ProtParam calculations substitute ambiguous residues as `U→C`, `O→L`,
`B→N`, `Z→Q`, `J→L`, and `X→G`; the original sequence remains in the
dataset. Biopython is needed only to run this source script, so the command
below supplies it for this run without adding it to the project dependencies:

```bash
uv run --with biopython modules/evaluate/scripts/dataset_sourcing/source_uniprot_protparams.py \
    modules/evaluate/configs/dataset_sourcing/source_uniprot_protparams.yaml
```

For example, run a frozen-trunk scalar regression evaluation on `gravy` from
the local dataset:

```bash
uv run accelerate launch modules/evaluate/src/run_evaluate.py \
    modules/evaluate/configs/config.yaml \
    modules/evaluate/configs/models/esm2_35m.yaml \
    modules/evaluate/configs/tasks/sequence_regression.yaml \
    modules/evaluate/configs/datasets/uniprot_protparams.yaml \
    modules/evaluate/configs/tuning/torch_linear.yaml \
    workspace.base_path=./eval/protparams/gravy
```

For another property, change both `steps.prepare.label_column` (for
example, `mass`) and the output path (for example,
`workspace.base_path=./eval/protparams/mass`) so runs do not overwrite each
other. The dataset overlay points to the local artifact; override
`workspace.dataset_repo_id` if it was uploaded to the Hub. These are separate
single-output regression runs.

## ProteinGym dataset sourcing

ProteinGym is organized as per-assay variant tables rather than the standard
`sequence`/`targets` dataset shape. This section covers downloading and
preparing those artifacts; scoring runners, split selection, and outputs are
documented in the [ProteinGym evaluation guide](../../README.md#proteingym-evaluation).

Sourcing follows the method in the
[ProteinGym GitHub README](https://github.com/OATML-Markslab/ProteinGym): the
reference CSV comes from the GitHub repo, and each track's assays come from one
zip hosted at `<base_url>/ProteinGym_<version>/<track>.zip` (for example,
`https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_substitutions.zip`).
The full archive for the selected track is downloaded and extracted, and the
source converts every assay listed in that track's reference file. The sourcing
script has no assay-subset option. Select a track with
`track: dms_substitutions | dms_indels | clinical_substitutions |
clinical_indels` in the config.

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
a different path to opt back into a shared directory across tracks.

Sourcing always writes one parquet file per assay under `data/`, preserving
ProteinGym's official fold columns (`fold_random_5`, `fold_modulo_5`, and
`fold_contiguous_5`) when present. The checked-in YAML enables
`combine_assays: true`, which writes `data/combined.parquet` while keeping the
per-assay files available. With `create_split_subsets: true`, sourcing writes
reproducible per-assay `train`/`validation`/`test` labels using a seeded 80/10/10
split, plus canonical and pooled random/stratified combined split files. These
reuse the same partition helpers
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
