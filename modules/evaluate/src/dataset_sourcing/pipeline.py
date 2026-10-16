"""HF-repository sourcing orchestration and CLI entrypoint."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from pydantic import BaseModel

from modules.core.utils.config_loader import load_runtime_config_from_argv
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
    minimal_preprocess,
    read_source_splits,
)
from modules.evaluate.src.dataset_sourcing.spec import DatasetSourceSpec

logger = logging.getLogger(__name__)

# Hidden, so it is never uploaded; records the config that produced the outputs.
_CONFIG_FINGERPRINT_FILE = ".sourcing_config.json"
# Settings that do not change the generated files, so changing them keeps the cache.
_FINGERPRINT_EXCLUDE = {
    "token",
    "datasets",
    "all_datasets",
    "work_dir",
    "stats_threads",
    "force_download",
    "overwrite_output",
    "revision",
    "commit_message",
    "private",
}


def resolve_datasets(
    selected: list[str] | None, use_all: bool, known_datasets: list[str]
) -> list[str]:
    """Resolve the selection, with all-datasets taking precedence.

    Explicit selections are validated, deduplicated, and kept in caller order.
    """
    if use_all:
        return list(known_datasets)

    if not selected:
        raise ValueError(
            "No datasets selected. Use --datasets <name ...> or --all-datasets."
        )

    unique = list(dict.fromkeys(selected))

    unknown = sorted(set(unique) - set(known_datasets))
    if unknown:
        raise ValueError(
            f"Unknown dataset(s): {unknown}. Known options: {known_datasets}"
        )

    return unique


def _config_fingerprint(
    config: BaseModel, spec: DatasetSourceSpec, data_dir: Path
) -> str:
    """Fingerprint output-affecting config, spec, and source parquet contents."""
    source_files: dict[str, str] = {}
    for path in sorted(data_dir.rglob("*.parquet")):
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        source_files[path.relative_to(data_dir).as_posix()] = digest
    return json.dumps(
        {
            "config": config.model_dump(mode="json", exclude=_FINGERPRINT_EXCLUDE),
            "spec": {
                key: value
                for key, value in asdict(spec).items()
                if key != "known_datasets"
            },
            "source_files": source_files,
        },
        sort_keys=True,
        indent=2,
    )


def prepare_output_dir(config: DatasetSourcingConfig, dataset_name: str) -> Path:
    """Create the output directory, deleting its prior contents only when requested."""
    output_dir = config.work_dir / "outputs" / dataset_name
    if output_dir.exists() and config.overwrite_output:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def run_script[ConfigT: BaseModel](
    model_cls: type[ConfigT], run: Callable[[ConfigT], object]
) -> None:
    """Shared ``main()`` for sourcing scripts: set up logging, load config from argv, run."""
    _configure_logging()
    run(load_runtime_config_from_argv(sys.argv[1:], model_cls))


def run_for_dataset(
    dataset_name: str,
    config: DatasetSourcingConfig,
    token: str | None,
    downloads_dir: Path,
    outputs_dir: Path,
    spec: DatasetSourceSpec,
) -> None:
    """Run one HF-repository source through download, preprocessing, and output.

    Minimal-preprocess artifacts are reused only when their config and source
    parquet fingerprint still match. ``preprocess=none`` preserves the raw
    snapshot while adding a generated card and stats.
    """
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
        fingerprint_path = dataset_output_dir / _CONFIG_FINGERPRINT_FILE
        fingerprint = _config_fingerprint(config, spec, data_dir)
        # Reuse outputs only when a prior run with this exact config finished.
        if (
            data_dir_out.exists()
            and any(data_dir_out.glob("*.parquet"))
            and stats_json_is_complete(stats_path)
            and readme_path.exists()
            and fingerprint_path.exists()
            and fingerprint_path.read_text(encoding="utf-8") == fingerprint
            and not config.overwrite_output
            and not config.force_download
        ):
            logger.info(
                "Processed output already exists for %s with the same config. "
                "Reusing %s",
                dataset_name,
                dataset_output_dir,
            )
        else:
            fingerprint_path.unlink(missing_ok=True)
            split_frames, validation_method = load_splits(
                data_dir, config, column_rename=spec.column_rename
            )
            hf_splits = minimal_preprocess(
                dataset_name=dataset_name,
                split_frames=split_frames,
                column_rename=spec.column_rename,
            )
            if config.create_split_subsets:
                hf_splits = create_split_subsets(hf_splits, config.seed, config)
            write_hf_split_files(dataset_output_dir, hf_splits)
            write_dataset_stats(
                output_dir=dataset_output_dir,
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
            fingerprint_path.write_text(fingerprint, encoding="utf-8")
            logger.info(
                "Wrote processed dataset (%s rows) and complete stats JSON to %s",
                sum(frame.height for frame in hf_splits.values()),
                stats_path,
            )
        source_for_upload = dataset_output_dir
    else:
        # preprocess="none": upload the raw downloaded snapshot as-is, annotated with stats/card.
        source_for_upload = data_dir.parent
        seqkit_frames = build_seqkit_frames_from_raw_splits(
            dataset_name=dataset_name,
            split_frames=read_source_splits(data_dir),
        )
        write_dataset_stats(
            output_dir=source_for_upload,
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

    upload_dataset_artifact(
        source_path=source_for_upload,
        dataset_name=dataset_name,
        config=config,
        token=token,
    )


def run_cli[ConfigT: BaseModel](
    argv: list[str], model_cls: type[ConfigT], spec: DatasetSourceSpec
) -> None:
    """CLI for HF-repository sources handled by :func:`run_for_dataset`.

    Raw API/file sources with custom parsing or target construction should use
    source-specific orchestration and reuse shared helpers as appropriate.
    """
    _configure_logging()

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
