"""Source the calibration dataset from homolog alignments or precomputed profiles.

Two inputs are supported:

- ``a3m_dir``: HHblits ``<id>.a3m`` alignments (see ``run_hhblits.py``). The
  query (first sequence) is the protein; per-column amino-acid frequencies are
  counted over the query and its homologs.
- ``proteome_path`` + ``consensus_path``: precomputed profile CSVs
  (``id, sequence`` and ``sequence_name, position (0-based), wt_aa, A..Y``).

The output is a test-only split with one row per protein whose ``targets``
column lists ``[position, f_A, ..., f_Y]`` for each conserved position.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl
from pydantic import Field, model_validator

from modules.core.utils.hf_download import resolve_hf_token
from modules.evaluate.scripts.dataset_sourcing.shared.config import DatasetSourcingConfig
from modules.evaluate.scripts.dataset_sourcing.shared.hf_io import upload_dataset_artifact
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


class SourceCalibrationConfig(DatasetSourcingConfig):
    """Config for sourcing the calibration dataset."""

    dataset_name: str = "human_calibration"
    a3m_dir: Path | None = None
    proteome_path: Path | None = None
    consensus_path: Path | None = None
    min_consensus_freq: float = Field(0.70, gt=0.0, le=1.0)
    # a3m input only (the CSVs carry no homolog counts).
    min_homologs: int = Field(10, ge=0)

    @model_validator(mode="after")
    def _one_input(self) -> SourceCalibrationConfig:
        csv_set = [p is not None for p in (self.proteome_path, self.consensus_path)]
        if (self.a3m_dir is not None and any(csv_set)) or (
            self.a3m_dir is None and not all(csv_set)
        ):
            raise ValueError(
                "Set either a3m_dir, or both proteome_path and consensus_path."
            )
        return self


# Byte -> amino-acid index; anything else (gaps, X, ...) maps to NUM_AA and is ignored.
_AA_LOOKUP = np.full(256, len(AA_COLUMNS), dtype=np.int64)
_AA_LOOKUP[np.frombuffer(CANONICAL_AMINO_ACIDS.encode(), dtype=np.uint8)] = np.arange(
    len(AA_COLUMNS)
)


def read_a3m(path: Path) -> list[str]:
    """Return the aligned sequences of an a3m file, insertions (lowercase) removed."""
    sequences: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            sequences.append("")
        elif sequences:
            sequences[-1] += "".join(c for c in line.strip() if not c.islower())
    return sequences


def a3m_profile(
    sequences: list[str], min_consensus_freq: float
) -> tuple[str, pl.DataFrame]:
    """Return the query sequence and its conserved columns."""
    if len({len(s) for s in sequences}) != 1:
        raise ValueError("a3m match columns have unequal lengths.")
    msa = np.frombuffer("".join(sequences).upper().encode(), dtype=np.uint8)
    msa = msa.reshape(len(sequences), -1)
    msa = msa[:, msa[0] != ord("-")]
    num_aa = len(AA_COLUMNS)
    length = msa.shape[1]
    flat = _AA_LOOKUP[msa] + (num_aa + 1) * np.arange(length)
    counts = np.bincount(flat.ravel(), minlength=(num_aa + 1) * length)
    counts = counts.reshape(length, num_aa + 1)[:, :num_aa]
    totals = counts.sum(axis=1, keepdims=True)
    freqs = counts / np.maximum(totals, 1)
    keep = (totals[:, 0] > 0) & (freqs.max(axis=1) >= min_consensus_freq)

    query = msa[0].tobytes().decode()
    positions = np.flatnonzero(keep)
    profile = pl.DataFrame(
        {
            "position": positions,
            "wt_aa": [query[i] for i in positions],
            **{aa: freqs[keep, i] for i, aa in enumerate(AA_COLUMNS)},
        },
        schema_overrides={"position": pl.Int64},
    )
    return query, profile


def read_a3m_profiles(
    a3m_dir: Path, min_consensus_freq: float, min_homologs: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build proteome and consensus frames from ``<id>.a3m`` files."""
    ids, queries, profiles = [], [], []
    skipped = 0
    for path in sorted(a3m_dir.glob("*.a3m")):
        sequences = read_a3m(path)
        if len(sequences) - 1 < min_homologs:
            skipped += 1
            continue
        protein_id = path.name.removesuffix(".a3m").removesuffix(".align")
        query, profile = a3m_profile(sequences, min_consensus_freq)
        ids.append(protein_id)
        queries.append(query)
        profiles.append(profile.with_columns(sequence_name=pl.lit(protein_id)))
    logger.info(
        "Read %d alignments; skipped %d with fewer than %d homologs.",
        len(ids),
        skipped,
        min_homologs,
    )
    proteome = pl.DataFrame({"id": ids, "sequence": queries})
    return proteome, pl.concat(profiles)


def build_calibration_frame(
    proteome: pl.DataFrame, consensus: pl.DataFrame, min_consensus_freq: float
) -> pl.DataFrame:
    """Join profiles to sequences and group conserved positions per protein.

    Proteins with any position whose ``wt_aa`` disagrees with the sequence (or
    falls outside it) are dropped whole: their profile was built on another
    isoform, so none of their positions can be trusted.
    """
    joined = consensus.join(
        proteome, left_on="sequence_name", right_on="id", how="inner"
    ).with_columns(
        residue=pl.col("sequence").str.slice(pl.col("position"), 1),
    )
    mismatched = joined.filter(pl.col("residue") != pl.col("wt_aa"))[
        "sequence_name"
    ].unique()
    if mismatched.len():
        logger.warning(
            "Dropping %d proteins whose wt_aa disagrees with the sequence.",
            mismatched.len(),
        )

    kept = joined.filter(
        ~pl.col("sequence_name").is_in(mismatched.implode()),
        pl.col("wt_aa").is_in(AA_COLUMNS),
        pl.max_horizontal(AA_COLUMNS) >= min_consensus_freq,
    )
    return (
        kept.sort("sequence_name", "position")
        .group_by("sequence_name", maintain_order=True)
        .agg(
            pl.col("sequence").first(),
            pl.concat_list(pl.col("position").cast(pl.Float64), *AA_COLUMNS).alias(
                "targets"
            ),
        )
        .rename({"sequence_name": "id"})
    )


def write_card(
    output_dir: Path, config: SourceCalibrationConfig, frame: pl.DataFrame
) -> None:
    num_positions = frame["targets"].list.len().sum()
    if config.a3m_dir is not None:
        source = (
            "HHblits alignments (query counted in the profile, proteins with "
            f"fewer than {config.min_homologs} homologs dropped)"
        )
    else:
        source = "precomputed HHblits profiles"
    card = f"""---
pretty_name: {config.dataset_name}
---

# {config.dataset_name}

Generated: {generated_utc_date()}

Zero-shot calibration benchmark: proteins with conserved alignment columns
(consensus frequency >= {config.min_consensus_freq}) from {source}.
Evaluated with `task_type: calibration`.

- `test`: {frame.height} proteins, {num_positions} conserved positions.
- `targets`: per protein, a list of `[position, f_A, ..., f_Y]` (0-based
  residue position, then homolog frequencies in `{CANONICAL_AMINO_ACIDS}` order).

[`stats.json`](stats.json) contains row counts and SeqKit metrics.
"""
    (output_dir / "README.md").write_text(card, encoding="utf-8")


def run(config: SourceCalibrationConfig) -> Path:
    if config.a3m_dir is not None:
        proteome, consensus = read_a3m_profiles(
            config.a3m_dir, config.min_consensus_freq, config.min_homologs
        )
    else:
        proteome = pl.read_csv(config.proteome_path, columns=["id", "sequence"])
        consensus = pl.read_csv(
            config.consensus_path,
            columns=["sequence_name", "position", "wt_aa", *AA_COLUMNS],
        )
    frame = build_calibration_frame(proteome, consensus, config.min_consensus_freq)
    logger.info(
        "Built %d proteins with %d conserved positions.",
        frame.height,
        frame["targets"].list.len().sum(),
    )

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
