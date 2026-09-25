# `modules/data` — Data Pipeline Module

This module turns raw public protein sequence databases into clean, scored, and clustered training datasets ready for protein language model pretraining. The seven-step pipeline is configured through YAML files (validated by Pydantic) with optional CLI dotlist overrides via OmegaConf.

```
Download → Preprocess → Cluster → Score → Assemble → Upload → Stats
```

Each step can be independently enabled/disabled via `steps.<step>.enabled` in the config, making it possible to run individual stages or skip ones that are already done.

## Table of Contents

- [Quick Start — Toy Smoke Test (yeast)](#quick-start--toy-smoke-test-yeast)
- [Directory Layout](#directory-layout)
- [Dataset Disk Layout](#dataset-disk-layout)
- [Step 1 — Download](#step-1--download)
- [Step 2 — Preprocess](#step-2--preprocess)
- [Step 3 — Cluster](#step-3--cluster)
- [Step 4 — Score](#step-4--score)
- [Step 5 — Assemble](#step-5--assemble)
- [Step 6 — Upload](#step-6--upload)
- [Step 7 — Stats](#step-7--stats)
- [Running on a Slurm-Managed Cluster](#running-on-a-slurm-managed-cluster)
- [Config System](#config-system)

---

## Quick Start — Toy Smoke Test (yeast)

Before running a full dataset, you can validate that the pipeline works end-to-end on your
machine using the bundled [`configs/config.yaml`](configs/config.yaml) — the fully-documented
base config, which defaults to a single small yeast reference proteome (`dataset.name: test_yeast`).
Every step is disabled by default, so enable the ones you want to smoke-test
via CLI overrides:

```bash
uv run modules/data/src/run_data.py modules/data/configs/config.yaml \
    steps.download.enabled=true \
    steps.preprocess.enabled=true \
    steps.cluster.enabled=true \
    steps.assemble.enabled=true
```

By default the output is written under `./datasets/test_yeast/`. To send it elsewhere,
override `dataset.base_path` by appending `dataset.base_path=/path/to/output` to the command line or editing the config. The pipeline will create the directory if it doesn't exist.


**What a successful run produces** (under `<base_path>/test_yeast/`):

```
test_yeast/
├── download/UP000002311_559292.fasta.gz   ← raw yeast proteome
├── train/
│   └── test_yeast_assembled.parquet       ← final assembled parquet (sequences + clusters)
├── tmp/
│   ├── test_yeast_all.fasta.gz            ← normalized FASTA (needed for the cluster step)
│   ├── test_yeast_fasta_splits/           ← per-chunk gzipped FASTA
│   ├── test_yeast_parquet_shards/         ← base sequence shards
│   ├── test_yeast_cluster_assignments/_shards ← base sequence shards
│   └── test_yeast_cluster_assignments.parquet
└── stats/
    ├── test_yeast_stats.parquet           ← seqkit summary statistics (from preprocess)
    ├── test_yeast_stats_summary.json      ← Stats step: length/score/cluster summary over the assembled parquet
    └── test_yeast_length_dist.png, ...    ← Stats step: distribution figures
```

If the run completes and `train/test_yeast_assembled.parquet` exists,
your environment and the four core pipeline steps (Download → Preprocess → Cluster → Assemble) are working.


> **Running a real dataset:** `configs/config.yaml` is the base config — it defines every field
> with the yeast smoke-test defaults. `configs/{uniref,mgnify,bfd,oas}.yaml` are overlays that
> only set the fields that differ for that dataset (source URLs, thread counts, cluster
> thresholds, repo IDs, etc.). Always pass the base config first, then the overlay — later files
> win on shared keys (OmegaConf merge):
>
> ```bash
> uv run modules/data/src/run_data.py \
>     modules/data/configs/config.yaml \
>     modules/data/configs/bfd.yaml \
>     steps.download.enabled=true
> ```
>
> An overlay alone (without `config.yaml`) will fail validation — it's missing required base
> fields like `dataset.base_path`.

---

## Directory Layout

```
modules/data/
├── README.md                        ← this file
│
├── configs/                         ← yaml configs (one per dataset or use case)
│   ├── config.yaml                  ← base config with every field documented (defaults to the yeast smoke test); layer a
│   │                                  dataset overlay on top to run a production dataset (see "Config System" below)
│   ├── uniref.yaml                  ← UniRef overlay: fields that differ from config.yaml (source_urls, thread counts, etc.)
│   ├── mgnify.yaml                  ← MGnify overlay
│   ├── bfd.yaml                     ← BFD overlay
│   └── oas.yaml                     ← OAS (paired antibody) overlay
│
└── src/
    ├── config.py                    ← DataPipelineConfig: top-level Pydantic schema that
    │                                  composes StepsConfig + DatasetConfig into one object
    ├── run_data.py                  ← CLI entry point: loads config and runs enabled steps
    │
    ├── dataset/
    │   └── dataset.py               ← DatasetConfig + Dataset class
    │
    ├── steps/                       ← one file per pipeline step
    │   ├── download.py              ← Step 1: fetch source files via aria2c
    │   ├── preprocess.py            ← Step 2: decompress → FASTA → Parquet shards + stats
    │   ├── cluster.py               ← Step 3: cascaded MMseqs2 sequence clustering
    │   ├── score.py                 ← Step 4: pLM-based per-sequence quality scoring
    │   ├── assemble.py              ← Step 5: join base sequences, clusters, and scores
    │   ├── upload.py                ← Step 6: push assembled parquets to Hugging Face Hub
    │   └── stats.py                 ← Step 7: compute summary statistics over the assembled parquet
    │
    └── utils/                       ← shared helpers
        ├── io_utils.py               ← download/process helpers, path resolution
        ├── conversion_utils.py        ← FASTA ↔ Parquet conversion
        ├── cluster_utils.py           ← MMseqs cascade helpers (thresholds, tmp dirs, joins)
        └── __init__.py               ← package marker
```

Shared config loading helpers live in `modules/core/utils/config_loader.py`.

---

## Dataset Disk Layout

After a full pipeline run, the dataset directory looks like this:

```
<base_path>/
└── <name>/
    ├── download/                            ← raw files fetched by the download step
    ├── tmp/                                 ← intermediate artifacts (safe to delete after assembly)
    │   ├── <name>_all.fasta.gz              ← monolithic FASTA (needed for the cluster step)
    │   ├── <name>_fasta_splits/             ← per-chunk gzipped FASTA files (seqkit_split_size seqs each)
    │   ├── <name>_parquet_shards/           ← per-chunk Parquet files (base sequences)
    │   ├── <name>_cluster_assignments.parquet  ← MMseqs2 cluster membership table
    │   ├── <name>_scores/                   ← per-sequence pLM scores (one file per base shard, same filename)
    │   └── <name>_assemble_work/            ← assemble step scratch space, kept (not auto-deleted) for resumability
    │       ├── _cluster_buckets/            ← cluster table pre-split into num_join_buckets hash buckets
    │       └── _assemble_parts/             ← per-shard, per-bucket join parts prior to finalization
    ├── train/                               ← final training-ready artifacts (assembled dataset only)
    │   └── <name>_assembled.parquet         ← final joined dataset (base_source is a single file)
    │       (or <name>_assembled/ directory of shards when base_source is a directory of shards)
    └── stats/
        ├── <name>_stats.parquet             ← seqkit summary statistics (from preprocess)
        ├── <name>_stats_summary.json        ← Stats step: length/score/cluster summary over the assembled parquet
        └── <name>_length_dist.png, ...      ← Stats step: distribution figures
```

---

## Step 1 — Download

**Purpose:** Fetch all raw source files declared in `dataset.source_urls` using `aria2c` for parallel, multi-connection downloading. Already-present files are skipped unless `overwrite: true`.

**Key config fields (`steps.download`):**

| Field | Default | Description |
|---|---|---|
| `enabled` | `true` | Skip this step entirely when `false` |
| `overwrite` | `false` | Re-download files that already exist |
| `split` | `16` | Number of aria2c split segments per file |
| `max_connections_per_server` | `16` | Parallel connections to a single server |
| `min_split_size` | `"1M"` | Minimum size per aria2c split segment |
| `max_concurrent_downloads` | `1` | Number of files downloaded simultaneously |

**Input:** `dataset.source_urls` (list of HTTP/FTP URLs in the dataset config)
**Output:** raw files in `<dataset>/download/`

---

## Step 2 — Preprocess

**Purpose:** Convert raw downloaded files into a normalized FASTA and a sharded Parquet dataset, then compute sequence statistics. The step is dataset-aware: `uniref`/`test_yeast`, `mgnify`, and `bfd` each have a dedicated extraction pipeline (`pv | pigz -dc | seqkit (or tar) | awk | tee | pigz | split`) to handle their source format efficiently. `bfd` additionally extracts only the `*_a3m.ffdata` member from the source tarball (skipping the `cs219`/`hhm` profile members, which aren't sequence data) and strips alignment gaps/consensus/database-match rows from the A3M records. The monolithic FASTA is split into chunks of `seqkit_split_size` sequences, which are then converted to Parquet in parallel.

`oas` is handled differently: it reads paired heavy/light antibody CSVs directly with pandas (dropping rows where ANARCI flags either chain as `"Shorter"`), writes a FASTA record per pair (`heavy_seq` + `XXXXX` linker + `light_seq`, for stats purposes only), and — because the source already provides structured columns — writes the base Parquet shard **directly from the CSV** (one shard per source file), bypassing the generic FASTA→Parquet converter entirely (`convert_fasta_to_parquet` is forced to `false`). Set `extra_columns` to carry additional source CSV columns straight through into the Parquet shard. `test_mode` for `oas` limits processing to the first 2 source CSV files (rather than a sequence-count cutoff).

**Unique sequence IDs:** while streaming, every FASTA header is rewritten inline by an `awk` stage to `>{DATASET}_{hash} {original_header}`, where `hash` is an 8-hex-character Knuth multiplicative hash of the per-stream sequence counter (a bijection on integers < 2^32, so no collisions up to 4.29B sequences). The original header is preserved as the description so `source_id` can still be recovered during Parquet conversion, without a second pass over the data.

**Sub-stages (all independently skippable via config):**

1. **FASTA generation** — decompress source, linearize sequences, rewrite headers with unique IDs, write `<name>_all.fasta.gz` and per-chunk split files
2. **Parquet conversion** — convert each FASTA chunk to a Parquet shard in parallel using `polars-bio`
3. **SeqKit stats** — compute length distribution statistics across all FASTA chunks

**Key config fields (`steps.preprocess`):**

| Field | Default | Description |
|---|---|---|
| `enabled` | `true` | Skip this step entirely when `false` |
| `test_mode` | `true` | Limit extraction to the first 5 M sequences for local testing |
| `generate_fasta` | `true` | Run FASTA extraction and chunking sub-stage |
| `overwrite_fasta` | `false` | Force re-generation of existing monolithic FASTA and split chunk files |
| `overwrite_parquet` | `false` | Force re-generation of existing Parquet shards |
| `fasta_id_column` | `sequence_id` | Column name for the unique hashed sequence ID in the output Parquet |
| `fasta_source_id_column` | `original_id` | Column name for the original (pre-hash) sequence identifier |
| `fasta_description_column` | `description` | Column name for the remaining FASTA header text |
| `fasta_sequence_column` | `sequence` | Column name for the sequence string |
| `seqkit_split_size` | `50_000_000` | Sequences per FASTA chunk (and Parquet shard) |
| `fasta_extract_threads` | `8` | `pigz` threads for decompressing raw source files |
| `fasta_write_threads` | `2` | `seqkit`/`pigz` threads for linearizing and writing the monolithic FASTA |
| `fasta_split_threads` | `2` | `pigz` threads for compressing each split chunk |
| `fasta_split_compression_level` | `1` | gzip level for temporary split chunks (1 = fastest, final output stays at the default level) |
| `extra_columns` | `[]` | Extra source columns to carry into the base Parquet (`oas` only); requires `convert_fasta_to_parquet=false` |
| `stats_threads` | `os.cpu_count()` | Threads used by the SeqKit stats sub-stage |
| `convert_fasta_to_parquet` | `true` | Run FASTA → Parquet conversion sub-stage |
| `parquet_conversion_workers` | `2` | Number of shards converted in parallel |

**Input:** raw files in `<dataset>/download/`
**Output:** `<dataset>/tmp/<name>_all.fasta.gz`, `<dataset>/tmp/<name>_fasta_splits/`, `<dataset>/tmp/<name>_parquet_shards/`, `<dataset>/stats/`

---

## Step 3 — Cluster

**Purpose:** Reduce redundancy by grouping similar sequences using MMseqs2 (`easy-linclust` by default, or `easy-cluster`). The step performs a **cascaded** clustering: starting from the highest identity threshold and working down, each round clusters the representative sequences from the previous round. Assignments are propagated between rounds via memory-bounded, hash-bucketed joins (`polars` lazy scans over on-disk buckets) so the join never needs the full population in RAM. The result is a single Parquet table with one cluster assignment column per threshold level.

By default, cascade intermediates live in a temporary directory that is cleaned up automatically. Set `output_dir` (or `resume: true`, which requires it) to persist them under a stable path and resume an interrupted run without redoing completed rounds.

**Key config fields (`steps.cluster`):**

| Field | Default | Description |
|---|---|---|
| `enabled` | `true` | Skip this step entirely when `false` |
| `identity_thresholds` | `[0.9, 0.8, ..., 0.3]` | Sequence identity thresholds per cascade round |
| `coverage_threshold` | `0.8` | Minimum fraction of the shorter sequence that must align |
| `num_threads` | `1` | CPU threads per MMseqs2 call |
| `resume` | `false` | Reuse parquet outputs from already-completed rounds (requires `output_dir`) |
| `input_fasta` | `null` | Override the default input FASTA path |
| `output_dir` | `null` | Persist cascade workspace/intermediates here instead of a temp dir |
| `output_path` | `null` | Override the default final cluster-assignments Parquet path |

**Nested MMseqs settings (`steps.cluster.mmseqs`):**

| Field | Default | Description |
|---|---|---|
| `command.executable` | `mmseqs` | MMseqs2 binary to invoke |
| `command.workflow` | `easy-linclust` | MMseqs workflow: `easy-linclust` or `easy-cluster` |
| `command.coverage_mode` | `1` | MMseqs `--cov-mode` value |
| `command.parquet_compression` | `zstd` | Compression codec for intermediate cascade Parquet files |
| `command.split_memory_limit` | `null` | Pass `--split-memory-limit` to MMseqs2 (e.g. `"64G"`) |
| `command.clust_hash` | `false` | Enable MMseqs hash-based pre-dedup before k-mer matching |
| `command.linclust_version` | `2` | MMseqs linclust algorithm version (`easy-linclust` only) |
| `command.alignment_mode` | `null` | Pass `--alignment-mode` (e.g. `3` for exact residue identity) |
| `command.seq_id_mode` | `null` | Pass `--seq-id-mode` (e.g. `0` to divide identities by aligned columns) |
| `command.sensitivity` | `null` | Pass `-s` prefilter sensitivity (e.g. `7.5`) |
| `command.cluster_reassign` | `null` | Pass `--cluster-reassign 1` to revalidate members after cascading |
| `command.max_seqs` | `null` | Pass `--max-seqs` candidates retained per sequence |
| `command.kmer_per_seq` | `null` | Pass `--kmer-per-seq` seeds (Linclust sensitivity) |
| `command.num_join_buckets` | `32` | Bucket count for the memory-bounded cascade joins |
| `columns.*` | — | Column name templates used across the cascade (rarely need overriding) |
| `outputs.*` | — | Filename/directory templates for MMseqs intermediate files (rarely need overriding) |

**Input:** `<dataset>/tmp/<name>_all.fasta.gz`
**Output:** `<dataset>/tmp/<name>_cluster_assignments.parquet`

---

## Step 4 — Score

**Purpose:** Compute a per-sequence score using an existing protein language model. The scoring function is **RED** (Residue Embedding Diversity), which measures how distinct a sequence's contextual embeddings are — sequences with higher RED scores are more informationally diverse. The step runs inference directly through an HF `AutoModel` (no 🤗 Accelerate) using `torch.multiprocessing`: one worker process is spawned per locally visible GPU (falls back to a single CPU worker if none), each writing its assigned shards straight to their final Parquet file (atomic tmp-then-rename), so there is no rank-partial-merge step to reason about.

For very large datasets, set `shard_index` / `shard_count` to split the Parquet input across multiple jobs running in parallel (e.g. across nodes). If left unset, they are auto-derived from `SLURM_ARRAY_TASK_ID`/`SLURM_ARRAY_TASK_MIN`/`SLURM_ARRAY_TASK_MAX`, so a SLURM job array works out of the box without extra config. Within a job, shards are balanced by row count across the local GPU workers.

**Memory note:** each GPU worker loads its shards one at a time (via a `DataLoader` over a HF `Dataset.from_parquet`), so peak memory per worker is roughly the size of its single largest assigned shard — use `shard_count` to further split the work if a shard is too large for a node.

**Key config fields (`steps.score`):**

| Field | Default | Description |
|---|---|---|
| `enabled` | `true` | Skip this step entirely when `false` |
| `inference_model` | `flair-bio/amplify-350m` | HF model identifier for embedding inference |
| `score_column` | `RED` | Name of the output score column in the Parquet (RED is the only scoring metric supported) |
| `sequence_name_col` | `sequence_id` | Column holding each sequence's unique identifier in the input Parquet |
| `sequence_column` | `sequence` | Column holding the sequence string in the input Parquet |
| `packed` | `true` | Pack multiple sequences per forward pass |
| `mixed_precision` | `bf16` | Autocast precision: `bf16`, `fp16`, or `no` (fp32). Must resolve to half-precision when `packed=true` |
| `batch_size` | `16` | Sequences (or packs) per batch |
| `max_length` | `2048` | Maximum token length per packed batch |
| `pad_to_multiple_of` | `8` | Padding alignment for unpacked batches |
| `dataloader_workers` | `4` | PyTorch `DataLoader` worker processes |
| `output_base_path` | `null` | Override the default output root (defaults to the dataset's `tmp_path`); score shards are always written to a `<name>_scores/` subdirectory underneath |
| `shard_index` | `null` | 0-based index of this worker's slice (for distributed scoring); auto-derived from SLURM array env vars if unset |
| `shard_count` | `null` | Total number of parallel workers; auto-derived from SLURM array env vars if unset |
| `resume` | `true` | Skip shards whose output already exists |
| `override_source` | `null` | Override the default Parquet input path |

**Input:** `<dataset>/tmp/<name>_parquet_shards/` (or `override_source`)
**Output:** `<dataset>/tmp/<name>_scores/` — one Parquet file per base shard, same filename as its base shard (e.g. `chunk_000000.parquet`)

### Packed (GPU) vs unpacked (CPU/GPU)

The `packed` and `mixed_precision` fields together decide which hardware the step can run on:

| Mode | `packed` | `mixed_precision` | Hardware |
|---|---|---|---|
| **Packed** | `true` | `bf16` or `fp16` | **GPU only** |
| **Unpacked** | `false` | `no` (fp32) | CPU **or** GPU |

- **Packed** routes through PyTorch's native variable-length attention, which is GPU-only and half-precision-only. If `packed=true` resolves to fp32, the step fails fast with an actionable error.
- **Unpacked** uses standard padded batching and works anywhere.
- The device(s) are selected automatically (one worker process per visible CUDA GPU, else a single CPU worker): there is no device flag. On a GPU machine, prepend `CUDA_VISIBLE_DEVICES=""` to force the CPU path.

**Note:** in YAML and OmegaConf CLI overrides, the bare token `no` is coerced to the boolean `False`. Always quote it so it stays the string `"no"` — on the CLI use the quote-preserving form `'steps.score.mixed_precision="no"'`, and in YAML write `mixed_precision: "no"`.

#### Example use case: packed scoring on a single GPU

```bash
uv run modules/data/src/run_data.py \
    modules/data/configs/config.yaml \
    steps.score.enabled=true \
    steps.score.packed=true \
    steps.score.mixed_precision=bf16
```

#### Example use case: unpacked scoring on a CPU-only machine

```bash
uv run modules/data/src/run_data.py \
    modules/data/configs/config.yaml \
    steps.score.enabled=true \
    steps.score.packed=false \
    steps.score.mixed_precision="no"
```

---

## Step 5 — Assemble

**Purpose:** Produce the final training Parquet file (or shards) by joining the base sequence shards with the cluster assignment table and the score table on the shared sequence ID. The cluster source is required; the score source is optional (if missing, it's skipped and those columns are absent). The output mode is inferred automatically from `base_source`:

- **`base_source` is a single file** — generates a single merged `.parquet` file.
- **`base_source` is a directory of shards** — writes one output shard per input shard with a mirrored filename. The cluster table is bucketed by `hash(id) % num_join_buckets` once, then each bucket is joined against every shard in turn (`ProcessPoolExecutor`, `spawn` context), so a large cluster table is read once total instead of once per shard, and each join is bounded to one bucket's worth of data. Score shards are matched to base shards by filename and joined in afterward.

Both modes are resumable: with `skip_existing: true` (the default), already-produced outputs are skipped on rerun; set `overwrite: true` to force full reprocessing regardless. Every output is written atomically, so a crash never leaves a partial/corrupt file that a later run would mistake for complete.

> Tune `max_workers`/`num_join_buckets` at `sbatch` submission time (CLI overrides) rather than in the dataset config — the right values depend on the cluster table size and node resources.

**Key config fields (`steps.assemble`):**

| Field | Default | Description |
|---|---|---|
| `enabled` | `true` | Skip this step entirely when `false` |
| `base_id_column` | `sequence_id` | Sequence ID column name in the base shards |
| `cluster_id_column` | `sequence_id` | Sequence ID column name in the cluster assignments table |
| `score_id_column` | `sequence_id` | Sequence ID column name in the score table |
| `dataset_id` | `""` | Data source identifier stamped into the output (e.g. `"UniRef100"`, `"BFD"`, `"MGnify"`) |
| `skip_existing` | `true` | Skip outputs (shards or the single file) that already exist on rerun |
| `overwrite` | `false` | Force full reprocessing regardless of existing outputs |
| `max_workers` | `null` | Process pool size for sharded assembly (defaults to a runtime-determined size) |
| `num_join_buckets` | `32` | Number of hash buckets to split the cluster table into before joining (bounds memory per join) |
| `base_source` | `null` | Override default base Parquet path (directory of shards or a single file) |
| `output_dir` | `null` | Override default output directory (sharded mode) |
| `output_file` | `null` | Override default output file path (single-file mode) |
| `cluster_source` | `null` | Override default cluster assignments path |
| `score_source` | `null` | Override default scores path |

**Input:** `<dataset>/tmp/<name>_parquet_shards/`, `<name>_cluster_assignments.parquet`, `<name>_scores/`
**Output:** `<dataset>/train/<name>_assembled.parquet` (single-file mode) or `<dataset>/train/<name>_assembled/` (sharded mode, one output shard per input shard, mirrored filenames)

**Example assembled output** (`pd.read_parquet("chunk_000003.parquet").head()`):

```
                                sequence_description  \
0  Large ribosomal subunit protein eL27B OS=Sacch...
1  Transposon Ty1-DR6 Gag polyprotein OS=Saccharo...
2  26S proteasome complex subunit SEM1 OS=Sacchar...

                                            sequence  sequence_length  \
0  MAKFLKAGKVAVVVRGRYAGKKVVIVKPHDEGSKSHPFGHALVAGI...              136
1  MESQQLSQHSPISHGSACASVTSKEVHTNQDPLDVSASKTEECEKA...              440
2  MSTDVAAAQAQSKIDLTKKKNEEINKKSLEEDDEFEDFPIDTWANG...               89

  cluster_rep_at_90 cluster_rep_at_60 cluster_rep_at_30       RED
0            P0C2H6            P0C2H6            P0C2H6  0.297607
1            Q12193            O13535            O13535  0.036224
2            O94742            O94742            O94742  0.077820
```

Alongside the sequence identifier columns, the assembled dataset notably includes
`sequence_description`, `sequence`, and `sequence_length`, one `cluster_rep_at_<t>` column
per clustering threshold (present only if the cluster step ran), and the score column
(`RED`, present only if the score step ran).

---

## Step 6 — Upload

**Purpose:** Push the assembled Parquet dataset to a Hugging Face Hub dataset repository. Requires the `HF_TOKEN` (or `HUGGINGFACE_HUB_TOKEN`) environment variable. Upload is disabled by default and requires `repo_id` to be explicitly set.

**Key config fields (`steps.upload`):**

| Field | Default | Description |
|---|---|---|
| `enabled` | `false` | Must be explicitly set to `true` |
| `repo_id` | `null` | Target HF dataset repo, e.g. `"org/my-dataset"` |
| `private` | `true` | Create the repo as private if it does not exist |
| `split` | `train` | Path prefix inside the repo (HF dataset split name) |
| `revision` | `null` | Target branch or tag (defaults to `main`) |
| `source_override` | `null` | Override the default assembled parquet path |

**Input:** `<dataset>/train/<name>_assembled.parquet` (or shards)
**Output:** HF dataset at `https://huggingface.co/datasets/<repo_id>`

---

## Step 7 — Stats

**Purpose:** Compute summary statistics (row count, sequence length distribution, score distribution/quantiles/mode, summed length, ambiguous-residue counts, cluster-threshold reduction) over the assembled dataset, write them out as a JSON file, and render distribution figures (length/score histograms, cluster-reduction bar chart, length/score box plots) for quick QA and slides. The source is resolved as either a single Parquet file or a directory of shards; a directory of shards is tried first, falling back to a single `.parquet` file if no shard directory is found.

Each expensive piece (numeric aggregation, ambiguous-residue scan, per-threshold cluster counts, score mode) runs as its own narrowly-scoped Polars pass rather than one combined query, keeping peak memory bounded on very large datasets. Cluster counts use `approx_n_unique()` (HyperLogLog, near-constant memory) instead of an exact count, since exact cardinality on hundreds of millions of cluster IDs can OOM.

This step can also run standalone, without a `Dataset` (e.g. against an arbitrary Parquet path outside the pipeline's directory layout) — in that case `override_source` and `override_output` must both be set.

**Key config fields (`steps.stats`):**

| Field | Default | Description |
|---|---|---|
| `enabled` | `true` | Skip this step entirely when `false` |
| `override_source` | `null` | Override default source (a Parquet file or directory of shards); required if run without a `Dataset` |
| `override_output` | `null` | Override default output path (a file or directory); required if run without a `Dataset` |
| `length_col` | `sequence_length` | Column used for sequence length statistics |
| `score_col` | `red_score` | Column used for score statistics |
| `sequence_col` | `sequence` | Column scanned for ambiguous-residue counting |
| `quantiles` | `[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]` | Quantiles computed over `length_col`/`score_col` (0.25/0.5/0.75 are always included, for the box plots) |
| `count_ambiguous` | `true` | Count ambiguous residues (`ambiguous_chars`); requires a full scan of `sequence_col` |
| `ambiguous_chars` | `XBZJ` | Characters considered ambiguous residues |
| `count_cluster_thresholds` | `true` | Count distinct clusters per identity threshold, from `cluster_rep_at_*` columns written by the cluster step |
| `cluster_col_prefix` | `cluster_rep_at_` | Prefix used to auto-detect cluster threshold columns |
| `make_plots` | `true` | Render distribution figures alongside the JSON summary |
| `plot_num_bins` | `50` | Number of histogram bins for the distribution plots |
| `plot_dpi` | `150` | DPI for saved figures |

**Input:** `<dataset>/train/<name>_assembled/` (or `<name>_assembled.parquet`)
**Output:** `<dataset>/stats/<name>_stats_summary.json`, plus `<name>_length_dist.png` / `<name>_length_dist_log.png`, `<name>_score_dist.png` / `<name>_score_dist_log.png` (linear- and log-scale y-axis, since a dominant peak can hide smaller bins on a linear axis), `<name>_cluster_reduction.png`, `<name>_length_boxplot.png`, `<name>_score_boxplot.png`

---

## Running on a Slurm-Managed Cluster

These are minimal `sbatch` templates to get you started. The options below (resources, `$SLURM_TMPDIR` usage, etc.) are tuned for the Digital Research Alliance of Canada (DRAC) clusters (Rorqual, Fir, Nibi, etc.) — freely adjust `--account`, `--time`, paths, and resources to fit your own cluster and allocation. Copy whichever templates you need into your own scripts directory; they're intentionally bare-bones so you can shape them around how you actually run the pipeline.

**Suggested resources:**
- **Download / Preprocess / Cluster** — single node, `--cpus-per-task=192`, `--mem=700G` (these steps are CPU/IO-bound: decompression, `seqkit`, MMseqs2, Parquet conversion).
- **Score** — single node, `--gpus=h100:4`, `--cpus-per-task=48`, `--mem=0` (GPU-bound inference via a per-GPU `torch.multiprocessing` worker pool; on Fir, a GPU node has exactly 4x H100 80GB, 48 cores, and ~1125G memory, so this requests the whole node).

> **Note on Cluster:** MMseqs2 does heavy random I/O against its working directory. Point `steps.cluster.output_dir` at `$SLURM_TMPDIR` (node-local NVMe) instead of your Lustre/project filesystem, then copy the result back to permanent storage once the job finishes.

### 1–2. Download / Preprocess

```bash
#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --account=def-yourgroup
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=700G

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

uv run modules/data/src/run_data.py \
    modules/data/configs/config.yaml \
    modules/data/configs/uniref.yaml \
    steps.download.enabled=true \
    steps.preprocess.enabled=true
```

### 3. Cluster (local NVMe)

```bash
#!/bin/bash
#SBATCH --job-name=cluster
#SBATCH --account=def-yourgroup
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=700G

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

uv run modules/data/src/run_data.py \
    modules/data/configs/config.yaml \
    modules/data/configs/uniref.yaml \
    steps.cluster.enabled=true \
    steps.cluster.num_threads="$SLURM_CPUS_ON_NODE" \
    steps.cluster.resume=true \
    steps.cluster.output_dir="$SLURM_TMPDIR/cascade"   # node-local NVMe, not Lustre

# Copy the persisted cascade workspace back to permanent storage before it's wiped.
rsync -a "$SLURM_TMPDIR/cascade/" /path/to/permanent/storage/cascade/
```

### 4. Score (multi-GPU, one worker process per GPU)

```bash
#!/bin/bash
#SBATCH --job-name=score
#SBATCH --account=def-yourgroup
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --gpus=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
export HF_HUB_OFFLINE=1   # compute nodes have no internet; pre-download models on a login node first

uv run modules/data/src/run_data.py \
    modules/data/configs/config.yaml \
    modules/data/configs/uniref.yaml \
    steps.score.enabled=true \
    steps.score.mixed_precision=bf16
```

No launcher is needed — the step spawns one worker process per GPU visible on the node.

For datasets too large for one job, submit this as a `--array` job and pass `steps.score.shard_index`/`steps.score.shard_count` (or rely on the `SLURM_ARRAY_TASK_*` auto-detection described in [Step 4 — Score](#step-4--score)).

### 5–7. Assemble / Upload / Stats

```bash
#!/bin/bash
#SBATCH --job-name=assemble
#SBATCH --account=def-yourgroup
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

uv run modules/data/src/run_data.py \
    modules/data/configs/uniref.yaml \
    steps.assemble.enabled=true \
    steps.upload.enabled=true \
    steps.stats.enabled=true
```

---

## Config System

All config classes are Pydantic `BaseModel` subclasses composed into a single `DataPipelineConfig` (`src/config.py`):

```
DataPipelineConfig
├── dataset: DatasetConfig
└── steps: StepsConfig
    ├── download:   DownloadConfig
    ├── preprocess: PreprocessConfig
    ├── cluster:    ClusterConfig
    │   └── mmseqs: MmseqsClusteringConfig
    │       ├── columns:  MmseqsColumnConfig
    │       ├── outputs:  MmseqsOutputConfig
    │       └── command:  MmseqsCommandConfig
    ├── score:      ScoreConfig
    ├── assemble:   AssembleConfig
    ├── upload:     UploadConfig
    └── stats:      StatsConfig
```

---
