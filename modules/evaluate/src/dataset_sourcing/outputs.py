"""Write processed splits, stats, seqkit metrics, and dataset cards to disk."""

from __future__ import annotations

import gzip
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import polars as pl

from modules.data.src.steps.cluster import MmseqsClusteringConfig
from modules.evaluate.src.dataset_sourcing.config import (
    DatasetSourcingConfig,
    join_repo_path,
)
from modules.evaluate.src.dataset_sourcing.preprocess import ValidationSplitMethod
from modules.evaluate.src.dataset_sourcing.spec import DatasetSourceSpec
from modules.evaluate.src.utils.stats import compute_seqkit_stats


def generated_utc_date() -> str:
    return datetime.now(UTC).date().isoformat()


def write_hf_split_files(
    output_dir: Path,
    split_frames: dict[str, pl.DataFrame],
) -> Path:
    """Write processed split frames using Hugging Face shard filenames."""
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    expected_files = {
        f"{split_name}-00000-of-00001.parquet" for split_name in split_frames
    }
    for path in data_dir.glob("*-00000-of-00001.parquet"):
        if path.name not in expected_files:
            path.unlink()

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
    """Append SeqKit metrics to the dataset's stats JSON."""
    split_fasta_dir, combined_fasta = build_seqkit_split_fastas(
        output_dir=output_dir,
        split_frames=split_frames,
    )

    seqkit_stats = compute_seqkit_stats(
        split_fasta_dir=split_fasta_dir,
        fasta_output=combined_fasta,
        threads=threads,
    )

    stats_path = output_dir / "stats.json"
    stats: dict[str, Any] = (
        cast(dict[str, Any], json.loads(stats_path.read_text(encoding="utf-8")))
        if stats_path.exists()
        else {}
    )
    stats["seqkit_stats"] = seqkit_stats.to_dicts()
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    legacy_stats_path = output_dir / "stats" / f"{dataset_name}_stats.parquet"
    legacy_stats_path.unlink(missing_ok=True)
    if legacy_stats_path.parent.exists() and not any(
        legacy_stats_path.parent.iterdir()
    ):
        legacy_stats_path.parent.rmdir()

    return stats_path


def write_dataset_stats(
    output_dir: Path,
    dataset_name: str,
    split_frames: dict[str, pl.DataFrame],
    threads: int,
    seqkit_split_frames: dict[str, pl.DataFrame] | None = None,
) -> Path:
    """Write row statistics and SeqKit metrics for a sourced dataset."""
    write_stats(output_dir, split_frames)
    return write_seqkit_stats(
        output_dir=output_dir,
        dataset_name=dataset_name,
        split_frames=split_frames
        if seqkit_split_frames is None
        else seqkit_split_frames,
        threads=threads,
    )


def stats_json_is_complete(stats_path: Path) -> bool:
    """Whether stats JSON contains both base metadata and SeqKit metrics."""
    try:
        stats: Any = json.loads(stats_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(stats, dict):
        return False
    stats_dict = cast(dict[str, Any], stats)
    return all(
        key in stats_dict for key in ("rows_total", "rows_by_split", "columns")
    ) and isinstance(stats_dict.get("seqkit_stats"), list)


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
    validation_method: ValidationSplitMethod,
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

    if config.preprocess != "minimal":
        length_line = (
            "- Max sequence length filter was not applied (`preprocess=none`)."
        )
    elif config.max_sequence_length is not None:
        length_line = (
            f"- Applied max sequence length filter: {config.max_sequence_length}."
        )
    else:
        length_line = "- No max sequence length filter was applied."
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
    if config.preprocess != "minimal":
        validation_line = (
            "- Validation handling: raw source files were preserved; "
            "no split was synthesized."
        )
    elif validation_method == "source":
        validation_line = (
            "- Validation handling: downloaded source split was preserved."
        )
    elif validation_method == "mmseqs2":
        validation_line = (
            "- Validation handling: upstream did not provide a validation split, "
            f"so `validation` was sampled from `train` using seed `{seed}` and "
            "approximately the `test` split size using MMseqs2 cluster boundaries."
        )
    elif validation_method == "random":
        validation_line = (
            "- Validation handling: upstream did not provide a validation split, "
            f"so `validation` was sampled from `train` using seeded random row "
            f"sampling with seed `{seed}` and matched to the `test` split size."
        )
    else:
        validation_line = (
            "- Validation handling: upstream did not provide a validation split, "
            "and the empty `test` split resulted in an empty synthesized `validation` split."
        )
    has_cluster_splits = bool(split_frames) and any(
        name.endswith("_cluster") for name in (split_frames or {})
    )
    alternative_splits = sorted(
        name for name in rows_by_split if name not in {"train", "validation", "test"}
    )
    if config.preprocess != "minimal":
        split_subset_line = "- Split subsets: not generated because `preprocess=none` preserves raw files.\n"
    elif config.create_split_subsets and alternative_splits:
        split_subset_line = (
            "- Split subsets: canonical `train`/`validation`/`test` retain source "
            "assignments except any synthesized validation; alternate views are "
            + ", ".join(f"`{name}`" for name in alternative_splits)
            + f", generated with seed `{seed}` from pooled canonical rows. They "
            f"repartition the same {sum(rows_by_split.get(role, 0) for role in ('train', 'validation', 'test'))} "
            "canonical rows, not additional examples.\n"
        )
    elif config.create_split_subsets:
        split_subset_line = (
            "- Split subsets: requested, but no alternate views were written.\n"
        )
    else:
        split_subset_line = (
            "- Split subsets: disabled by `create_split_subsets=false`.\n"
        )
    # Thresholds only affected the output if some MMseqs clustering actually ran.
    clustering_section = ""
    if config.preprocess == "minimal" and (
        has_cluster_splits or validation_method == "mmseqs2"
    ):
        command = MmseqsClusteringConfig().command
        clustering_section = (
            "## Sequence clustering\n\n"
            "Configured settings for MMseqs2 cluster-based assignment, used when "
            "that path runs:\n\n"
            f"- MMseqs workflow: `{command.workflow}`\n"
            f"- Minimum sequence identity (`--min-seq-id`): `{config.validation_cluster_identity_threshold}`\n"
            f"- Minimum coverage (`-c`): `{config.validation_cluster_coverage_threshold}`\n"
            f"- Coverage mode (`--cov-mode`): `{command.coverage_mode}`\n"
            f"- Linclust version (`--linclust-version`): `{command.linclust_version}`\n"
            f"- Threads: `{config.validation_cluster_num_threads}`\n"
            f"- Seed (cluster ordering / tie-breaking): `{seed}`\n\n"
        )
    source_line = spec.card_source_line.format(
        repo_id=spec.source_repo_id(dataset_name)
    )
    upstream_card_url = (
        f"https://huggingface.co/datasets/{spec.source_repo_id(dataset_name)}"
        f"/blob/{config.source_revision or 'main'}/README.md"
    )
    source_revision_line = (
        f"`{config.source_revision}`"
        if config.source_revision
        else "unpinned (default branch; pin `source_revision` to a commit to reproduce the input)"
    )
    intended_use = spec.card_descriptions.get(
        dataset_name,
        f"Protein sequence benchmark examples for the `{dataset_name}` task.",
    )
    data_files_line = (
        "Processed parquet files are stored under `data/` using Hugging Face split "
        "naming conventions (`train-*`, `validation-*`, `test-*`)."
        if config.preprocess == "minimal"
        else "Raw source files are retained without preprocessing."
    )
    if config.has_upload_target():
        repo_id = config.resolve_upload_repo_id(dataset_name)
        data_dir = (
            f', data_dir="{join_repo_path(config.repo_prefix, dataset_name)}"'
            if config.upload_layout == "single_repo"
            else ""
        )
        loading_line = f'ds = load_dataset("{repo_id}"{data_dir})'
    elif config.preprocess == "none":
        revision = (
            f', revision="{config.source_revision}"' if config.source_revision else ""
        )
        loading_line = (
            f'ds = load_dataset("{spec.source_repo_id(dataset_name)}"{revision})'
        )
    else:
        loading_line = (
            'ds = load_dataset("parquet", data_files={'
            '"train": "data/train-*.parquet", '
            '"validation": "data/validation-*.parquet", '
            '"test": "data/test-*.parquet"})'
        )

    card = (
        f"# {dataset_name}\n\n"
        f"{source_line}\n\n"
        "## Intended use\n\n"
        f"{intended_use}\n\n"
        "## Provenance\n\n"
        f"- Card generated (UTC): `{generated_utc_date()}`\n\n"
        f"- Upstream revision: {source_revision_line}.\n"
        f"- Upstream dataset card: {upstream_card_url} (label definitions, license, citation).\n\n"
        "## Data files\n\n"
        f"{data_files_line}\n\n"
        "Dataset statistics: [`stats.json`](stats.json) at the dataset root. "
        "It includes row counts by split, columns, and SeqKit sequence-length metrics.\n\n"
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
        f"{loading_line}\n"
        "print(ds)\n"
        "```\n"
    )

    card_path = output_dir / "README.md"
    card_path.write_text(card, encoding="utf-8")
    return card_path
