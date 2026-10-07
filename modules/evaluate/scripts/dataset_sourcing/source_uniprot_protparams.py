"""Source ProtParam regression targets from a configurable UniProt subset."""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

import polars as pl
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from pydantic import ConfigDict, Field

from modules.core.utils.hf_download import resolve_hf_token
from modules.evaluate.src.dataset_sourcing.config import (
    UniProtDatasetSourcingConfig,
    UniProtReviewStatus,
)
from modules.evaluate.src.dataset_sourcing.downloads import (
    UNIPROT_STREAM_URL,
    build_uniprot_query,
    download_uniprot_tsv,
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
)

logger = logging.getLogger(__name__)

DATASET_NAME = "uniprot_protparams"
PROTPARAM_COLUMNS = (
    "length",
    "mass",
    *(f"percent_{amino_acid}" for amino_acid in "acdefghiklmnpqrstvwy"),
    "fraction_helix_aas",
    "fraction_turn_aas",
    "fraction_sheet_aas",
    "instability_index",
    "gravy",
    "isoelectric_point",
    "charge_at_ph4_7",
    "charge_at_ph7_2",
    "charge_at_ph8",
)


class SourceUniprotProtParamsConfig(UniProtDatasetSourcingConfig):
    """Config for sourcing ProtParam properties from UniProt proteins."""

    model_config = ConfigDict(extra="forbid")

    all_datasets: bool = True
    work_dir: Path = Path("eval_datasets/uniprot_protparams_sourcing")
    repo_name_prefix: str = ""
    repo_prefix: str = "uniprot"
    commit_message: str = "Add sourced UniProt ProtParam regression dataset"
    organism_id: int | None = Field(9606, ge=1)
    review_status: UniProtReviewStatus = "reviewed"
    split_fractions: dict[str, float] = Field(
        default_factory=lambda: {"train": 0.8, "validation": 0.1, "test": 0.1}
    )


def _analysis_sequence(sequence: str) -> str:
    """Normalize ambiguous residues for analysis; leave the source unchanged."""
    return (
        sequence.replace("U", "C")
        .replace("O", "L")
        .replace("B", "N")
        .replace("Z", "Q")
        .replace("J", "L")
        .replace("X", "G")
    )


def frame_from_uniprot_table(
    raw: pl.DataFrame,
    review_status: UniProtReviewStatus = "reviewed",
) -> pl.DataFrame:
    """Calculate ProtParam properties for rows matching the review-status filter."""
    column_map = {name.lower(): name for name in raw.columns}
    required = {"entry", "reviewed", "length", "mass", "sequence"}
    missing = sorted(required - set(column_map))
    if missing:
        raise ValueError(f"UniProt TSV is missing required columns {missing}")

    records: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for row in raw.iter_rows(named=True):
        accession = str(row[column_map["entry"]])
        sequence = row[column_map["sequence"]]
        row_review_status = row[column_map["reviewed"]]
        if (
            row_review_status not in {"reviewed", "unreviewed"}
            or (review_status != "all" and row_review_status != review_status)
            or not isinstance(sequence, str)
            or not sequence.strip()
        ):
            continue
        try:
            length = int(row[column_map["length"]])
            mass = float(row[column_map["mass"]])
        except (TypeError, ValueError):
            continue
        if accession in seen_ids:
            continue

        sequence = sequence.strip().rstrip("*-")
        if not sequence:
            continue
        try:
            protein = ProteinAnalysis(_analysis_sequence(sequence))
            helix, turn, sheet = protein.secondary_structure_fraction()
            record: dict[str, object] = {
                "id": accession,
                "sequence": sequence,
                "length": length,
                "mass": mass,
                "fraction_helix_aas": helix,
                "fraction_turn_aas": turn,
                "fraction_sheet_aas": sheet,
                "instability_index": protein.instability_index(),
                "gravy": protein.gravy(),
                "isoelectric_point": protein.isoelectric_point(),
                "charge_at_ph4_7": protein.charge_at_pH(4.7),
                "charge_at_ph7_2": protein.charge_at_pH(7.2),
                "charge_at_ph8": protein.charge_at_pH(8.0),
            }
            record.update(
                {
                    f"percent_{amino_acid.lower()}": percent
                    for amino_acid, percent in protein.amino_acids_percent.items()
                }
            )
        except (KeyError, ValueError):
            continue
        records.append(record)
        seen_ids.add(accession)

    if not records:
        raise ValueError(
            f"No {review_status} UniProt proteins with valid source data found"
        )
    return pl.from_dicts(records)


def _dataset_name(config: SourceUniprotProtParamsConfig) -> str:
    if config.max_sequence_length is None:
        return DATASET_NAME
    return f"{DATASET_NAME}_{config.max_sequence_length}_cutoff"


def write_protparams_card(
    output_dir: Path,
    config: SourceUniprotProtParamsConfig,
    dataset_name: str,
    split_frames: dict[str, pl.DataFrame],
) -> Path:
    split_rows = "\n".join(
        f"- `{split}`: {frame.height} rows"
        for split, frame in sorted(split_frames.items())
    )
    organism_filter = config.organism_id if config.organism_id is not None else "all"
    length_filter = (
        f"Sequences longer than `{config.max_sequence_length}` residues are removed "
        "before MMseqs2 cluster assignment."
        if config.max_sequence_length is not None
        else "No maximum sequence-length filter was applied."
    )
    card = f"""# {dataset_name}

Protein-level sequence property dataset for UniProtKB entries. `length` and
`mass` are UniProt-reported; the other properties are calculated with
Biopython's `ProteinAnalysis`. Select one property per scalar regression run
with `steps.prepare.label_column`.

## Source and preprocessing

- UniProt REST API: `{UNIPROT_STREAM_URL}`
- Card generated (UTC): `{generated_utc_date()}`
- UniProt query: `{build_uniprot_query(config.organism_id, config.review_status)}`
- Organism Taxonomy ID: `{organism_filter}`.
- Review status: `{config.review_status}` (`reviewed` = Swiss-Prot, `unreviewed` = TrEMBL, `all` = both).
- No filter is applied based on UniProt-reported protein length.
- Ambiguous residues are normalized for analysis as U->C, O->L, B->N, Z->Q,
  J->L, and X->G; the original sequence is retained.
- Property columns: {", ".join(f"`{column}`" for column in PROTPARAM_COLUMNS)}.
- Properties are sequence-derived estimates, not experimental measurements.
- {length_filter}

## Splits

MMseqs2 cluster holdout using split fractions `{config.split_fractions}`, minimum
identity `{config.validation_cluster_identity_threshold}`, and minimum coverage
`{config.validation_cluster_coverage_threshold}`. Seed: `{config.seed}`.
When `create_split_subsets` is enabled, pooled random, length-stratified, and
hold-cluster-out subsets are also included. Hold-cluster-out subsets require
enough MMseqs clusters to populate all three roles.

{split_rows}

[`stats.json`](stats.json) contains row counts, columns, and sequence-length
metrics.
"""
    path = output_dir / "README.md"
    path.write_text(card, encoding="utf-8")
    return path


def run(config: SourceUniprotProtParamsConfig) -> list[Path]:
    resolve_datasets(config.datasets, config.all_datasets, [DATASET_NAME])

    token = resolve_hf_token(config.token)
    downloads_dir = config.work_dir / "downloads"
    source_path = download_uniprot_tsv(
        config,
        downloads_dir,
        "uniprot_protparams.tsv.gz",
        ("accession", "reviewed", "length", "mass", "sequence"),
    )
    raw = pl.read_csv(source_path, separator="\t", infer_schema=False)
    frame = frame_from_uniprot_table(raw, config.review_status)
    if config.max_sequence_length is not None:
        frame = frame.filter(
            pl.col("sequence").str.len_chars() <= config.max_sequence_length
        )
        if frame.height == 0:
            raise ValueError(
                "No UniProt ProtParam proteins remain after applying "
                f"max_sequence_length={config.max_sequence_length}"
            )

    # Keep whole sequence clusters together to reduce homology leakage across splits.
    split_frames = create_full_dataset_splits(
        frame,
        split_fractions=config.split_fractions,
        seed=config.seed,
        identity_threshold=config.validation_cluster_identity_threshold,
        coverage_threshold=config.validation_cluster_coverage_threshold,
        num_threads=config.validation_cluster_num_threads,
    )
    dataset_name = _dataset_name(config)
    output_splits = {
        split: split_frame.select("id", "sequence", *PROTPARAM_COLUMNS).with_columns(
            pl.lit(split).alias("split")
        )
        for split, split_frame in split_frames.items()
    }
    if config.create_split_subsets:
        output_splits = create_split_subsets(
            output_splits,
            seed=config.seed,
            identity_threshold=config.validation_cluster_identity_threshold,
            coverage_threshold=config.validation_cluster_coverage_threshold,
            num_threads=config.validation_cluster_num_threads,
            stratification_column="length",
        )
    output_dir = config.work_dir / "outputs" / dataset_name
    if output_dir.exists() and config.overwrite_output:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_hf_split_files(output_dir, output_splits)
    write_dataset_stats(
        output_dir, dataset_name, output_splits, threads=config.stats_threads
    )
    write_protparams_card(output_dir, config, dataset_name, output_splits)
    if config.has_upload_target():
        upload_dataset_artifact(output_dir, dataset_name, config, token)
    return [output_dir]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        stream=sys.stdout,
        force=True,
    )
    config = load_runtime_config_from_argv(
        sys.argv[1:], model_cls=SourceUniprotProtParamsConfig
    )
    run(config)


if __name__ == "__main__":
    main()
