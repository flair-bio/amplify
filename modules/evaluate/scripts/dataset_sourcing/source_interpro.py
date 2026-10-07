"""Source protein-level InterPro annotation datasets."""

from __future__ import annotations

import json
import logging
import re
import shutil
import sys
from pathlib import Path

import polars as pl
from pydantic import ConfigDict, Field

from modules.core.utils.hf_download import resolve_hf_token
from modules.evaluate.src.dataset_sourcing.config import UniProtDatasetSourcingConfig
from modules.evaluate.src.dataset_sourcing.downloads import (
    UNIPROT_STREAM_URL,
    build_uniprot_query,
    download_uniprot_tsv,
    download_url,
)
from modules.evaluate.src.dataset_sourcing.hf_io import upload_dataset_artifact
from modules.evaluate.src.dataset_sourcing.outputs import (
    generated_utc_date,
    write_dataset_stats,
    write_hf_split_files,
)
from modules.evaluate.src.dataset_sourcing.pipeline import (
    load_runtime_config_from_argv,
    resolve_datasets,
)
from modules.evaluate.src.dataset_sourcing.preprocess import (
    create_full_dataset_splits,
    create_split_subsets,
    minimal_preprocess,
)

logger = logging.getLogger(__name__)

INTERPRO_DOWNLOAD_URL = "https://ftp.ebi.ac.uk/pub/databases/interpro/current_release"
INTERPRO_ENTRY_LIST_URL = f"{INTERPRO_DOWNLOAD_URL}/entry.list"

# Dataset name -> ENTRY_TYPE in the official InterPro entry.list.
INTERPRO_CATEGORIES = {
    "interpro_active_site": "Active_site",
    "interpro_binding_site": "Binding_site",
    "interpro_conserved_site": "Conserved_site",
    "interpro_domain": "Domain",
    "interpro_family": "Family",
    "interpro_homologous_superfamily": "Homologous_superfamily",
    "interpro_repeat": "Repeat",
}

_INTERPRO_ID_RE = re.compile(r"IPR\d+")


class SourceInterProConfig(UniProtDatasetSourcingConfig):
    """Config for sourcing InterPro protein annotations."""

    model_config = ConfigDict(extra="forbid")

    all_datasets: bool = True
    work_dir: Path = Path("eval_datasets/interpro_sourcing")
    repo_name_prefix: str = ""
    repo_prefix: str = "interpro"
    commit_message: str = "Add sourced InterPro annotation datasets"
    organism_id: int | None = Field(9606, ge=1)
    interpro_download_url: str = INTERPRO_DOWNLOAD_URL
    min_label_count: int = Field(2, ge=1)
    split_fractions: dict[str, float] = Field(
        default_factory=lambda: {"train": 0.8, "validation": 0.1, "test": 0.1}
    )


def parse_interpro_accessions(cell: str | None) -> list[str]:
    """Extract unique InterPro accessions from a UniProt xref field."""
    if not cell:
        return []
    return sorted(set(_INTERPRO_ID_RE.findall(cell)))


def download_sources(
    config: SourceInterProConfig, downloads_dir: Path
) -> tuple[Path, Path]:
    entry_list_url = f"{config.interpro_download_url.rstrip('/')}/entry.list"
    uniprot_path = download_uniprot_tsv(
        config,
        downloads_dir,
        "uniprot_interpro.tsv.gz",
        ("accession", "sequence", "xref_interpro"),
    )
    entry_list_path = download_url(
        entry_list_url,
        downloads_dir / "interpro_entry_list.tsv",
        force=config.force_download,
    )
    return uniprot_path, entry_list_path


def read_interpro_entry_list(path: Path) -> pl.DataFrame:
    """Read InterPro's entry.list metadata table."""
    entries = pl.read_csv(path, separator="\t", infer_schema=False)
    required = {"ENTRY_AC", "ENTRY_TYPE", "ENTRY_NAME"}
    missing = sorted(required - set(entries.columns))
    if missing:
        raise ValueError(
            f"InterPro entry.list is missing columns {missing}; found {entries.columns}"
        )
    return entries.select("ENTRY_AC", "ENTRY_TYPE", "ENTRY_NAME").unique(
        subset="ENTRY_AC", keep="first", maintain_order=True
    )


def frame_from_interpro_tables(
    uniprot: pl.DataFrame,
    entries: pl.DataFrame,
    dataset_name: str,
    min_label_count: int = 2,
) -> tuple[pl.DataFrame, list[dict[str, str]]]:
    """Build protein-level multi-hot targets for one InterPro entry type."""
    if dataset_name not in INTERPRO_CATEGORIES:
        raise ValueError(
            f"Unknown InterPro dataset {dataset_name!r}. "
            f"Choose from: {sorted(INTERPRO_CATEGORIES)}"
        )
    if min_label_count < 1:
        raise ValueError("min_label_count must be at least 1")

    column_map = {name.lower(): name for name in uniprot.columns}
    accession_col = column_map.get("entry")
    sequence_col = column_map.get("sequence")
    interpro_col = column_map.get("interpro")
    if accession_col is None or sequence_col is None or interpro_col is None:
        raise ValueError(
            "UniProt TSV must contain Entry, Sequence, and InterPro columns; "
            f"found {uniprot.columns}"
        )

    category = INTERPRO_CATEGORIES[dataset_name]
    type_by_accession = dict(entries.select("ENTRY_AC", "ENTRY_TYPE").iter_rows())
    name_by_accession = dict(entries.select("ENTRY_AC", "ENTRY_NAME").iter_rows())

    # Count each protein once per entry; these counts define the retained vocabulary.
    protein_rows: list[tuple[str, str, set[str]]] = []
    label_counts: dict[str, int] = {}
    for accession, sequence, interpro_cell in uniprot.select(
        accession_col, sequence_col, interpro_col
    ).iter_rows():
        if not isinstance(sequence, str) or not sequence.strip():
            continue
        matching_accessions = {
            interpro_accession
            for interpro_accession in parse_interpro_accessions(interpro_cell)
            if type_by_accession.get(interpro_accession) == category
        }
        protein_rows.append((str(accession), sequence.strip(), matching_accessions))
        for interpro_accession in matching_accessions:
            label_counts[interpro_accession] = (
                label_counts.get(interpro_accession, 0) + 1
            )

    vocabulary = sorted(
        accession
        for accession, count in label_counts.items()
        if count >= min_label_count
    )
    if not vocabulary:
        raise ValueError(
            f"No {category} labels were observed at least {min_label_count} times."
        )

    vocabulary_index = {accession: index for index, accession in enumerate(vocabulary)}
    # Target positions follow sorted vocabulary; omit proteins with no retained labels.
    records: list[dict[str, object]] = []
    for accession, sequence, matching_accessions in protein_rows:
        targets = [0] * len(vocabulary)
        for interpro_accession in matching_accessions:
            index = vocabulary_index.get(interpro_accession)
            if index is not None:
                targets[index] = 1
        if any(targets):
            records.append({"id": accession, "sequence": sequence, "targets": targets})

    labels = [
        {
            "accession": accession,
            "name": name_by_accession[accession],
            "entry_type": category,
        }
        for accession in vocabulary
    ]
    return pl.DataFrame(records), labels


def write_interpro_card(
    output_dir: Path,
    config: SourceInterProConfig,
    dataset_name: str,
    source_name: str,
    labels: list[dict[str, str]],
    split_frames: dict[str, pl.DataFrame],
) -> Path:
    rows = "\n".join(
        f"- `{split}`: {frame.height} rows"
        for split, frame in sorted(split_frames.items())
    )
    entry_type = INTERPRO_CATEGORIES[source_name].replace("_", " ").lower()
    organism_filter = config.organism_id if config.organism_id is not None else "all"
    card = f"""# {dataset_name}

Protein-level multi-label dataset sourced from InterPro entry metadata and
UniProtKB protein-to-InterPro cross-references.

## Intended use

Protein {entry_type} annotation prediction from sequence. This evaluates
recovery of curated InterPro classifications, not an experimental assay result.

## Source and labels

- InterPro metadata: `{config.interpro_download_url.rstrip("/")}/entry.list`
- UniProt REST API: `{UNIPROT_STREAM_URL}`
- Card generated (UTC): `{generated_utc_date()}`
- Organism Taxonomy ID: `{organism_filter}` (`all` = all organisms).
- Review status: `{config.review_status}` (`reviewed` = Swiss-Prot, `unreviewed` = TrEMBL, `all` = both).
- UniProt query: `{build_uniprot_query(config.organism_id, config.review_status)}`
- UniProt TSV fields: `accession,sequence,xref_interpro`
- InterPro entry type: `{INTERPRO_CATEGORIES[source_name]}`
- Keep entries observed in at least {config.min_label_count} proteins and
    present in the training split.
- `targets` is a multi-hot vector ordered by InterPro accession. Accession,
  display name, and type are recorded in `label_vocabulary.json`.
- Proteins without a retained entry before splitting are dropped; after
    training-only vocabulary filtering, other splits may contain all-zero targets.

## Splits

Whole MMseqs2 `easy-linclust` clusters are assigned to splits targeting
`{config.split_fractions}`, with minimum identity
`{config.validation_cluster_identity_threshold}`, minimum coverage
`{config.validation_cluster_coverage_threshold}`, and
`{config.validation_cluster_num_threads}` thread(s). Seed: `{config.seed}`.
When `create_split_subsets` is enabled, pooled random, stratified, and
hold-cluster-out subsets are also included. Hold-cluster-out subsets require
enough MMseqs clusters to populate all three roles.
The optional maximum sequence length is `{config.max_sequence_length}` and is
applied before vocabulary construction and MMseqs2 clustering.

## Split sizes

{rows}

Vocabulary size: {len(labels)}.

## Dataset statistics

[`stats.json`](stats.json) at the dataset root contains row counts by split,
columns, and SeqKit sequence-length metrics.
"""
    path = output_dir / "README.md"
    path.write_text(card, encoding="utf-8")
    return path


def run(config: SourceInterProConfig) -> list[Path]:
    selected_datasets = resolve_datasets(
        config.datasets, config.all_datasets, list(INTERPRO_CATEGORIES)
    )

    token = resolve_hf_token(config.token)
    downloads_dir = config.work_dir / "downloads"
    uniprot_path, entry_list_path = download_sources(config, downloads_dir)
    uniprot = pl.read_csv(uniprot_path, separator="\t")
    if config.max_sequence_length is not None:
        # Apply the length limit before label counts and clustering.
        sequence_col = next(col for col in uniprot.columns if col.lower() == "sequence")
        uniprot = uniprot.filter(
            pl.col(sequence_col).str.len_chars() <= config.max_sequence_length
        )
    entries = read_interpro_entry_list(entry_list_path)
    output_paths: list[Path] = []

    for source_name in selected_datasets:
        dataset_name = source_name
        if config.max_sequence_length is not None:
            dataset_name = f"{dataset_name}_{config.max_sequence_length}_cutoff"

        frame, labels = frame_from_interpro_tables(
            uniprot, entries, source_name, min_label_count=config.min_label_count
        )
        split_frames = create_full_dataset_splits(
            frame,
            split_fractions=config.split_fractions,
            seed=config.seed,
            identity_threshold=config.validation_cluster_identity_threshold,
            coverage_threshold=config.validation_cluster_coverage_threshold,
            num_threads=config.validation_cluster_num_threads,
        )
        # Held-out labels must not define the training task vocabulary.
        train_targets = split_frames["train"]["targets"].to_list()
        retained = [
            index
            for index in range(len(labels))
            if any(target[index] for target in train_targets)
        ]
        if not retained:
            raise ValueError("No InterPro labels remain in the training split.")
        labels = [labels[index] for index in retained]
        split_frames = {
            split: data.with_columns(
                pl.col("targets").map_elements(
                    lambda target, indices=retained: [
                        target[index] for index in indices
                    ],
                    return_dtype=pl.List(pl.Int64),
                )
            )
            for split, data in split_frames.items()
        }
        split_frames = minimal_preprocess(
            dataset_name, split_frames, config.max_sequence_length, column_rename={}
        )
        if config.create_split_subsets:
            split_frames = create_split_subsets(
                split_frames,
                seed=config.seed,
                identity_threshold=config.validation_cluster_identity_threshold,
                coverage_threshold=config.validation_cluster_coverage_threshold,
                num_threads=config.validation_cluster_num_threads,
            )

        output_dir = config.work_dir / "outputs" / dataset_name
        if output_dir.exists() and config.overwrite_output:
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_hf_split_files(output_dir, split_frames)
        write_dataset_stats(
            output_dir, dataset_name, split_frames, threads=config.stats_threads
        )
        vocabulary_path = output_dir / "label_vocabulary.json"
        vocabulary_path.write_text(
            json.dumps(
                {
                    "dataset_name": dataset_name,
                    "vocabulary_version": 1,
                    "labels": labels,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        write_interpro_card(
            output_dir, config, dataset_name, source_name, labels, split_frames
        )
        if config.has_upload_target():
            upload_dataset_artifact(output_dir, dataset_name, config, token)
        output_paths.append(output_dir)

    return output_paths


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        stream=sys.stdout,
        force=True,
    )
    config = load_runtime_config_from_argv(sys.argv[1:], model_cls=SourceInterProConfig)
    run(config)


if __name__ == "__main__":
    main()
