"""HF-repository sourcing orchestration and CLI entrypoint."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from pydantic import BaseModel

from modules.core.utils.config_loader import load_and_parse
from modules.core.utils.hf_download import resolve_hf_token
from modules.evaluate.src.dataset_sourcing.config import DatasetSourcingConfig
from modules.evaluate.src.dataset_sourcing.hf_io import (
    download_dataset,
    upload_dataset_artifact,
)
from modules.evaluate.src.dataset_sourcing.outputs import (
    stats_json_is_complete,
    write_dataset_card,
    write_dataset_stats,
    write_hf_split_files,
)
from modules.evaluate.src.dataset_sourcing.preprocess import (
    build_seqkit_frames_from_raw_splits,
    create_split_subsets,
    load_splits,
    load_splits_with_method,
    minimal_preprocess,
)
from modules.evaluate.src.dataset_sourcing.spec import DatasetSourceSpec

logger = logging.getLogger(__name__)


def resolve_datasets(
    selected: list[str] | None, use_all: bool, known_datasets: list[str]
) -> list[str]:
    """Resolve and validate the final dataset list requested by the user."""
    if use_all:
        return known_datasets

    if not selected:
        raise ValueError(
            "No datasets selected. Use --datasets <name ...> or --all-datasets."
        )

    # Preserve caller-provided order while dropping accidental duplicates.
    unique: list[str] = []
    for name in selected:
        if name not in unique:
            unique.append(name)

    unknown = sorted(set(unique) - set(known_datasets))
    if unknown:
        raise ValueError(
            f"Unknown dataset(s): {unknown}. Known options: {known_datasets}"
        )

    return unique


def _split_subset_files_match_config(data_dir: Path, enabled: bool) -> bool:
    roles = ("train", "validation", "test")
    required_methods = ("random", "stratified")
    all_methods = (*required_methods, "cluster")
    existing = {path.name for path in data_dir.glob("*.parquet")}

    if enabled:
        required = {
            f"{role}_{method}-00000-of-00001.parquet"
            for role in roles
            for method in required_methods
        }
        return required.issubset(existing)

    subset_files = {
        f"{role}_{method}-00000-of-00001.parquet"
        for role in roles
        for method in all_methods
    }
    return not existing.intersection(subset_files)


def load_runtime_config_from_argv[ConfigT: BaseModel](
    argv: list[str], model_cls: type[ConfigT]
) -> ConfigT:
    """Load config with the standard CLI pattern shared across sourcing scripts.

    Usage:
        <script>.py <config.yaml> [<override.yaml> ...] [key=value ...]
    """
    # OmegaConf-style dotlist overrides (key=value) vs. plain YAML config paths.
    config_paths = [a for a in argv if "=" not in a]
    overrides = [a for a in argv if "=" in a]

    if not config_paths:
        print(
            "Usage: <script>.py <config.yaml> [<override.yaml> ...] [key=value ...]",
            file=sys.stderr,
        )
        raise SystemExit(1)

    logger.info("Loading config from: %s", config_paths)
    return load_and_parse(
        path=config_paths, model_cls=model_cls, overrides=overrides or None
    )


def run_for_dataset(
    dataset_name: str,
    config: DatasetSourcingConfig,
    token: str | None,
    downloads_dir: Path,
    outputs_dir: Path,
    spec: DatasetSourceSpec,
) -> None:
    """Download an HF dataset repo and run generic preprocess/output/upload steps."""
    data_dir = download_dataset(
        dataset_name=dataset_name,
        downloads_dir=downloads_dir,
        token=token,
        force_download=config.force_download,
        spec=spec,
        source_revision=config.source_revision,
    )
    dataset_output_dir = outputs_dir / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    if config.preprocess == "minimal":
        stats_path = dataset_output_dir / "stats.json"
        readme_path = dataset_output_dir / "README.md"
        data_dir_out = dataset_output_dir / "data"
        # Skip reprocessing if a prior run already produced a complete set of outputs.
        if (
            data_dir_out.exists()
            and any(data_dir_out.glob("*.parquet"))
            and stats_json_is_complete(stats_path)
            and readme_path.exists()
            and _split_subset_files_match_config(
                data_dir_out, enabled=config.create_split_subsets
            )
            and not config.overwrite_output
            and not config.force_download
            and config.source_revision is None
        ):
            logger.info(
                "Processed output already exists for %s and --overwrite-output is off. "
                "Reusing %s",
                dataset_name,
                dataset_output_dir,
            )
        else:
            split_frames, validation_method = load_splits_with_method(
                data_dir,
                seed=config.seed,
                column_rename=spec.column_rename,
                validation_cluster_identity_threshold=config.validation_cluster_identity_threshold,
                validation_cluster_coverage_threshold=config.validation_cluster_coverage_threshold,
                validation_cluster_num_threads=config.validation_cluster_num_threads,
                max_sequence_length=config.max_sequence_length,
            )
            hf_splits = minimal_preprocess(
                dataset_name=dataset_name,
                split_frames=split_frames,
                max_sequence_length=config.max_sequence_length,
                column_rename=spec.column_rename,
            )
            if config.create_split_subsets:
                hf_splits = create_split_subsets(
                    hf_splits,
                    seed=config.seed,
                    identity_threshold=config.validation_cluster_identity_threshold,
                    coverage_threshold=config.validation_cluster_coverage_threshold,
                    num_threads=config.validation_cluster_num_threads,
                )
            write_hf_split_files(dataset_output_dir, hf_splits)
            write_dataset_stats(
                output_dir=dataset_output_dir,
                dataset_name=dataset_name,
                split_frames=hf_splits,
                threads=config.stats_threads,
            )
            write_dataset_card(
                output_dir=dataset_output_dir,
                dataset_name=dataset_name,
                config=config,
                split_frames=hf_splits,
                validation_method=validation_method,
                spec=spec,
            )
            logger.info(
                "Wrote processed dataset (%s rows) and complete stats JSON to %s",
                sum(frame.height for frame in hf_splits.values()),
                stats_path,
            )
        source_for_upload = dataset_output_dir
    else:
        # preprocess="none": upload the raw downloaded snapshot as-is, annotated with stats/card.
        source_for_upload = data_dir.parent
        raw_split_frames = load_splits(data_dir, seed=config.seed)
        seqkit_frames = build_seqkit_frames_from_raw_splits(
            dataset_name=dataset_name,
            split_frames=raw_split_frames,
        )
        write_dataset_stats(
            output_dir=source_for_upload,
            dataset_name=dataset_name,
            split_frames=seqkit_frames,
            threads=config.stats_threads,
        )
        write_dataset_card(
            output_dir=source_for_upload,
            dataset_name=dataset_name,
            config=config,
            split_frames=None,
            validation_method="source",
            spec=spec,
        )

    if config.has_upload_target():
        upload_dataset_artifact(
            source_path=source_for_upload,
            dataset_name=dataset_name,
            config=config,
            token=token,
        )
    else:
        logger.info(
            "No upload target configured. Set repo_owner (per_dataset_repo) or repo_id (single_repo). "
            "Skipping upload for %s.",
            dataset_name,
        )


def run_cli[ConfigT: BaseModel](
    argv: list[str], model_cls: type[ConfigT], spec: DatasetSourceSpec
) -> None:
    """CLI for HF-repository sources handled by :func:`run_for_dataset`.

    Raw API/file sources with custom parsing or target construction should use
    source-specific orchestration and reuse shared helpers as appropriate.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = load_runtime_config_from_argv(argv, model_cls=model_cls)

    selected = resolve_datasets(
        config.datasets, config.all_datasets, spec.known_datasets
    )
    token = resolve_hf_token(config.token)
    work_dir = config.work_dir.expanduser().resolve()
    downloads_dir = work_dir / "downloads"
    outputs_dir = work_dir / "outputs"

    downloads_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Starting dataset sourcing for %s dataset(s): %s", len(selected), selected
    )
    logger.info("Working directory: %s", work_dir)
    logger.info("Preprocess mode: %s", config.preprocess)

    for dataset_name in selected:
        logger.info("===== Processing dataset: %s =====", dataset_name)
        run_for_dataset(
            dataset_name=dataset_name,
            config=config,
            token=token,
            downloads_dir=downloads_dir,
            outputs_dir=outputs_dir,
            spec=spec,
        )

    logger.info("Dataset sourcing pipeline completed successfully.")
