"""Write processed splits, stats, seqkit metrics, and dataset cards to disk."""

from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path
from typing import Any

import polars as pl

from modules.evaluate.src.utils.stats import compute_seqkit_stats
from modules.evaluate.src.dataset_sourcing.config import DatasetSourcingConfig
from modules.evaluate.src.dataset_sourcing.spec import DatasetSourceSpec


def write_hf_split_files(
    output_dir: Path,
    split_frames: dict[str, pl.DataFrame],
) -> Path:
    """Write processed split frames using Hugging Face shard filenames."""
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    for split_name, frame in split_frames.items():
        file_name = f"{split_name}-00000-of-00001.parquet"
        frame.write_parquet(data_dir / file_name)

    return data_dir


def build_seqkit_split_fastas(
    output_dir: Path,
    split_frames: dict[str, pl.DataFrame],
) -> tuple[Path, Path]:
    """Create temporary split FASTA files and a combined FASTA for seqkit stats."""
    split_fasta_dir = output_dir / "tmp" / "seqkit_fasta_splits"
    combined_fasta = output_dir / "tmp" / "seqkit_all.fasta.gz"

    # Clear stale FASTA shards from a prior run so seqkit only sees the current splits.
    if split_fasta_dir.exists():
        shutil.rmtree(split_fasta_dir)
    split_fasta_dir.mkdir(parents=True, exist_ok=True)
    combined_fasta.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(combined_fasta, "wt", encoding="utf-8") as combined_handle:
        for split_name, frame in split_frames.items():
            split_path = split_fasta_dir / f"{split_name}.fasta.gz"
            with gzip.open(split_path, "wt", encoding="utf-8") as split_handle:
                for seq_id, sequence in frame.select(["id", "sequence"]).iter_rows():
                    header = f">{seq_id}\n"
                    seq_line = f"{sequence}\n"
                    split_handle.write(header)
                    split_handle.write(seq_line)
                    combined_handle.write(header)
                    combined_handle.write(seq_line)

    return split_fasta_dir, combined_fasta


def write_seqkit_stats(
    output_dir: Path,
    dataset_name: str,
    split_frames: dict[str, pl.DataFrame],
    threads: int,
) -> Path:
    """Compute and persist seqkit stats for the provided split frames."""
    split_fasta_dir, combined_fasta = build_seqkit_split_fastas(
        output_dir=output_dir,
        split_frames=split_frames,
    )

    stats_dir = output_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    compute_seqkit_stats(
        split_fasta_dir=split_fasta_dir,
        fasta_output=combined_fasta,
        stats_path=stats_dir,
        dataset_name=dataset_name,
        threads=threads,
    )

    return stats_dir / f"{dataset_name}_stats.parquet"


def write_stats(output_dir: Path, split_frames: dict[str, pl.DataFrame]) -> Path:
    """Write lightweight row-count and column metadata for generated splits."""
    rows_by_split = {
        split_name: frame.height for split_name, frame in split_frames.items()
    }
    rows_total = sum(rows_by_split.values())
    columns = []
    if split_frames:
        first_split = next(iter(split_frames.values()))
        columns = first_split.columns

    stats: dict[str, Any] = {
        "rows_total": rows_total,
        "rows_by_split": rows_by_split,
        "columns": columns,
    }

    stats_path = output_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats_path


def write_dataset_card(
    output_dir: Path,
    dataset_name: str,
    config: DatasetSourcingConfig,
    split_frames: dict[str, pl.DataFrame] | None,
    validation_was_synthesized: bool,
    spec: DatasetSourceSpec,
) -> Path:
    """Render and write a README.md dataset card describing preparation steps."""
    seed = config.seed
    rows_by_split: dict[str, int] = {}
    columns: list[str] = []
    if split_frames:
        rows_by_split = {
            split_name: frame.height for split_name, frame in split_frames.items()
        }
        columns = next(iter(split_frames.values())).columns

    split_lines = "\n".join(
        [f"- `{name}`: {count} rows" for name, count in sorted(rows_by_split.items())]
    )
    if not split_lines:
        split_lines = "- Splits are stored as parquet shards under `data/`."

    length_line = (
        f"- Applied max sequence length filter: {config.max_sequence_length}."
        if config.max_sequence_length is not None
        else "- No max sequence length filter was applied."
    )
    rename_line = (
        "- Renamed source columns: "
        + ", ".join(
            f"`{source}` -> `{target}`"
            for source, target in sorted(spec.column_rename.items())
        )
        + "."
        if spec.column_rename
        else "- No source columns were renamed."
    )
    columns_line = (
        "- Columns: " + ", ".join(f"`{column}`" for column in columns) + "."
        if columns
        else "- Columns: see the parquet shards under `data/`."
    )
    validation_line = (
        # Note when the card describes a synthesized (vs. upstream-provided) validation split.
        "- Validation handling: upstream did not provide a validation split, so "
        f"`validation` was sampled from `train` using seed `{seed}` and matched to "
        "the `test` split size. MMseqs2 cluster boundaries were used when the "
        "`mmseqs` binary was available, with seeded random row sampling as the fallback."
        if validation_was_synthesized
        else f"- Validation handling: upstream-provided validation split was preserved; configured seed was `{seed}`."
    )
    has_cluster_splits = bool(split_frames) and any(
        name.endswith("_cluster") for name in (split_frames or {})
    )
    split_subset_line = ""
    if has_cluster_splits:
        split_subset_line = (
            "- Split subsets: canonical `train`/`validation`/`test` preserve the "
            "downloaded source splits; `_random`, `_stratified`, and `_cluster` "
            f"suffixes select pooled alternative partitions generated with seed `{seed}`. "
            "Each pooled partition reuses the canonical split sizes.\n"
        )
    # Thresholds only affected the output if some MMseqs clustering actually ran.
    clustering_section = ""
    if has_cluster_splits or validation_was_synthesized:
        clustering_section = (
            "## Sequence clustering\n\n"
            "Configured settings used for cluster-based split assignment:\n\n"
            f"- Minimum sequence identity (`--min-seq-id`): `{config.validation_cluster_identity_threshold}`\n"
            f"- Minimum coverage (`-c`): `{config.validation_cluster_coverage_threshold}`\n"
            f"- Threads: `{config.validation_cluster_num_threads}`\n"
            f"- Seed (cluster ordering / tie-breaking): `{seed}`\n\n"
        )
    source_line = spec.card_source_line.format(
        repo_id=spec.source_repo_id(dataset_name)
    )

    card = (
        f"# {dataset_name}\n\n"
        f"{source_line}\n\n"
        "## Data files\n\n"
        "Parquet files are stored under `data/` using Hugging Face split naming conventions\n"
        "(`train-*`, `validation-*`, `test-*`).\n\n"
        "## Preparation\n\n"
        f"- Preprocess mode: `{config.preprocess}`.\n"
        f"- Seed: `{seed}`.\n"
        f"{length_line}\n"
        f"{rename_line}\n"
        f"{columns_line}\n"
        f"{validation_line}\n"
        f"{split_subset_line}\n"
        f"{clustering_section}"
        "## Split sizes\n\n"
        f"{split_lines}\n\n"
        "## Loading\n\n"
        "```python\n"
        "from datasets import load_dataset\n\n"
        'ds = load_dataset("<owner>/<repo>")\n'
        "print(ds)\n"
        "```\n"
    )

    card_path = output_dir / "README.md"
    card_path.write_text(card, encoding="utf-8")
    return card_path
