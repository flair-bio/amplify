"""Source the calibration dataset from a precomputed homolog-profile export.

The export directory holds ``result.json`` plus ``balanced/`` and
``high_nt_095/`` subsets, each with ``sequences.csv`` (``id, sequence``),
``consensus_positions.csv`` (``sequence_name, position`` (0-based),
``wt_aa, A..Y``) and ``position_support.csv`` (``category``,
``consensus_aas``, ``consensus_proportion``, ...).

The output is a test-only split with one row per protein whose ``targets``
column lists ``[position, f_A, ..., f_Y, is_t, bin, fold, in_balanced,
in_high_nt, consensus_mask]`` for each site.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import polars as pl

from modules.core.utils.hf_download import resolve_hf_token
from modules.evaluate.scripts.dataset_sourcing.shared.config import (
    DatasetSourcingConfig,
)
from modules.evaluate.scripts.dataset_sourcing.shared.hf_io import (
    upload_dataset_artifact,
)
from modules.evaluate.scripts.dataset_sourcing.shared.outputs import (
    generated_utc_date,
    write_dataset_stats,
    write_hf_split_files,
)
from modules.evaluate.scripts.dataset_sourcing.shared.pipeline import (
    prepare_output_dir,
    run_script,
)
from modules.evaluate.src.utils.amino_acids import CANONICAL_AMINO_ACIDS

logger = logging.getLogger(__name__)

AA_COLUMNS = list(CANONICAL_AMINO_ACIDS)
CONSERVATION_EDGES = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SUBSETS = {"balanced": "in_balanced", "high_nt_095": "in_high_nt"}
NUM_FOLDS = 5
FOLD_SEED = 0
SITE_FIELDS = ["is_t", "bin", "fold", "in_balanced", "in_high_nt", "consensus_mask"]


class SourceCalibrationConfig(DatasetSourcingConfig):
    """Config for sourcing the calibration dataset."""

    dataset_name: str = "omicmsa_calibration"
    export_dir: Path
    # Optional ``groups.csv`` (dataset, sequence_name, group, fold) to reuse
    # an existing fold assignment instead of computing one.
    groups_path: Path | None = None


def read_export(export_dir: Path) -> tuple[str, pl.DataFrame, pl.DataFrame]:
    """Return the dataset name, sequences, and sites of both subsets.

    A site present in both subsets is kept once, flagged as both.
    """
    receipt = json.loads((export_dir / "result.json").read_text())
    if receipt["selection"]["edges"] != CONSERVATION_EDGES:
        raise ValueError(f"Expected conservation bin edges {CONSERVATION_EDGES}.")

    sequence_frames, site_frames = [], []
    for subset in SUBSETS:
        directory = export_dir / subset
        sequence_frames.append(pl.read_csv(directory / "sequences.csv"))
        support = pl.read_csv(
            directory / "position_support.csv",
            columns=[
                "sequence_name",
                "position",
                "category",
                "consensus_aas",
                "consensus_proportion",
            ],
        )
        site_frames.append(
            pl.read_csv(directory / "consensus_positions.csv")
            .join(support, on=["sequence_name", "position"], validate="1:1")
            .with_columns(
                **{flag: pl.lit(name == subset) for name, flag in SUBSETS.items()}
            )
        )
    sequences = pl.concat(sequence_frames).unique().sort("id")
    flags = list(SUBSETS.values())
    sites = (
        pl.concat(site_frames)
        .group_by("sequence_name", "position", maintain_order=True)
        .agg(pl.exclude(flags).first(), pl.col(flags).any())
    )

    if sequences["id"].is_duplicated().any():
        raise ValueError("A protein has different sequences in the two subsets.")
    if sequences["sequence"].str.contains(f"[^{CANONICAL_AMINO_ACIDS}]").any():
        raise ValueError("Sequences must contain only the 20 canonical amino acids.")
    if not sites["category"].is_in(["T", "NT"]).all():
        raise ValueError("Site categories must be T or NT.")
    residues = sites.join(
        sequences, left_on="sequence_name", right_on="id", how="left"
    ).select(pl.col("sequence").str.slice(pl.col("position"), 1) == pl.col("wt_aa"))
    if not residues.to_series().fill_null(False).all():
        raise ValueError("Every site's wt_aa must match its sequence.")
    return receipt["dataset"], sequences, sites


def assign_folds(sequences: pl.DataFrame) -> pl.DataFrame:
    """Deal proteins into folds, keeping identical sequences together.

    Each group is named by its smallest id; sorted names are shuffled once and
    assigned round-robin.
    """
    groups = sequences.with_columns(group=pl.col("id").min().over("sequence"))
    names = np.array(sorted(groups["group"].unique()))
    np.random.default_rng(FOLD_SEED).shuffle(names)
    fold_of = {name: i % NUM_FOLDS for i, name in enumerate(names)}
    return groups.select("id", fold=pl.col("group").replace_strict(fold_of))


def read_folds(groups_path: Path, dataset: str) -> pl.DataFrame:
    return (
        pl.read_csv(groups_path)
        .filter(pl.col("dataset") == dataset)
        .select(id="sequence_name", fold="fold")
    )


def build_calibration_frame(
    sequences: pl.DataFrame, sites: pl.DataFrame, folds: pl.DataFrame
) -> pl.DataFrame:
    """Group sites per protein, adding their bin, T/NT flag and fold."""
    bins = np.searchsorted(
        CONSERVATION_EDGES, sites["consensus_proportion"].to_numpy(), side="right"
    )
    aa_bits = {aa: 1 << i for i, aa in enumerate(AA_COLUMNS)}
    sites = sites.with_columns(
        is_t=pl.col("category") == "T",
        bin=pl.Series(np.minimum(bins - 1, len(CONSERVATION_EDGES) - 2)),
        consensus_mask=pl.col("consensus_aas")
        .str.split("|")
        .list.eval(pl.element().replace_strict(aa_bits))
        .list.sum(),
    ).join(folds, left_on="sequence_name", right_on="id", how="left")
    if sites["fold"].is_null().any():
        raise ValueError("Every protein needs a fold.")
    return (
        sites.sort("sequence_name", "position")
        .group_by("sequence_name", maintain_order=True)
        .agg(
            pl.concat_list(
                pl.col("position", *AA_COLUMNS, *SITE_FIELDS).cast(pl.Float64)
            ).alias("targets")
        )
        .join(sequences, left_on="sequence_name", right_on="id")
        .select(id="sequence_name", sequence="sequence", targets="targets")
    )


def write_card(
    output_dir: Path, config: SourceCalibrationConfig, frame: pl.DataFrame
) -> None:
    sites = frame["targets"].explode()
    field = {name: 1 + len(AA_COLUMNS) + i for i, name in enumerate(SITE_FIELDS)}
    balanced = sites.list.get(field["in_balanced"]) == 1
    high_nt = sites.list.get(field["in_high_nt"]) == 1
    is_t = sites.list.get(field["is_t"]) == 1
    card = f"""---
pretty_name: {config.dataset_name}
---

# {config.dataset_name}

Generated: {generated_utc_date()}

Zero-shot calibration benchmark: masked-residue predictions are compared to
homolog amino-acid profiles. Evaluated with `task_type: calibration`.

- `test`: {frame.height} proteins, {sites.len()} sites.
- Balanced cohort: {(balanced & is_t).sum()} T and {(balanced & ~is_t).sum()} NT
  sites over conservation bins `{CONSERVATION_EDGES}`.
- High-NT cohort: {high_nt.sum()} NT sites with consensus >= 0.95.
- `targets`: per protein, a list of `[position, f_A, ..., f_Y, is_t, bin, fold,
  in_balanced, in_high_nt, consensus_mask]`: 0-based residue position, homolog
  frequencies in `{CANONICAL_AMINO_ACIDS}` order, whether the residue is the
  consensus (T), the conservation bin, the cross-fitting fold, cohort
  membership, and a bitmask of the (possibly tied) consensus residues.

[`stats.json`](stats.json) contains row counts and SeqKit metrics.
"""
    (output_dir / "README.md").write_text(card, encoding="utf-8")


def run(config: SourceCalibrationConfig) -> Path:
    dataset, sequences, sites = read_export(config.export_dir)
    folds = (
        assign_folds(sequences)
        if config.groups_path is None
        else read_folds(config.groups_path, dataset)
    )
    frame = build_calibration_frame(sequences, sites, folds)
    logger.info("Built %d proteins with %d sites.", frame.height, sites.height)

    split_frames = {"test": frame}
    output_dir = prepare_output_dir(config, config.dataset_name)
    write_hf_split_files(output_dir, split_frames)
    write_dataset_stats(output_dir, split_frames, threads=config.stats_threads)
    write_card(output_dir, config, frame)
    upload_dataset_artifact(
        output_dir, config.dataset_name, config, resolve_hf_token(config.token)
    )
    return output_dir


def main() -> None:
    run_script(SourceCalibrationConfig, run)


if __name__ == "__main__":
    main()
