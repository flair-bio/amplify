"""One-off script to source ProteinGym assays: DMS or clinical, substitutions
or indels.

Downloads follow the method documented in the ProteinGym README (one
zip per track, hosted on the Marks lab server), e.g.::

    curl -o DMS_ProteinGym_substitutions.zip \
        https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_substitutions.zip

Unlike the biomap/tape-proteinnet sources, ProteinGym has no train/val/test
splits: each track is a reference CSV (from the GitHub repo) describing its
assays plus one zip of per-assay CSVs (wild-type sequence + every scored
variant). The complete selected track is converted to parquet after the
archive is downloaded and extracted in full.

Column names are resolved case-insensitively against a few known aliases so
minor schema variants across ProteinGym releases don't break sourcing.
"""

from __future__ import annotations

import logging
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Internal domain-specific utility imports
from modules.core.utils.hf_download import resolve_hf_token
from modules.evaluate.src.dataset_sourcing.hf_io import upload_dataset_artifact
from modules.evaluate.src.dataset_sourcing.outputs import (
    write_seqkit_stats,
    write_stats,
)
from modules.evaluate.src.dataset_sourcing.pipeline import (
    load_runtime_config_from_argv,
)
from modules.evaluate.src.dataset_sourcing.preprocess import (
    create_split_subsets,
    random_partition_indices,
)

# Setup module logger
logger = logging.getLogger(__name__)

# Default URLs and dataset version constants
_DEFAULT_BASE_URL = "https://marks.hms.harvard.edu/proteingym"
_DEFAULT_VERSION = "v1.3"
_DEFAULT_REFERENCE_BASE_URL = (
    "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/reference_files"
)

# Supported track categories
ProteinGymTrack = Literal[
    "dms_substitutions", "dms_indels", "clinical_substitutions", "clinical_indels"
]


class _TrackSpec(NamedTuple):
    """Metadata container for a specific ProteinGym track."""

    zip_filename: str
    reference_filename: str
    is_indel: bool
    is_clinical: bool


# Registry mapping each supported track to its expected remote filenames and metadata flags.
# Filenames are confirmed against the upstream repo's resources table (all four tracks).
_TRACKS: dict[ProteinGymTrack, _TrackSpec] = {
    "dms_substitutions": _TrackSpec(
        "DMS_ProteinGym_substitutions.zip", "DMS_substitutions.csv", False, False
    ),
    "dms_indels": _TrackSpec(
        "DMS_ProteinGym_indels.zip", "DMS_indels.csv", True, False
    ),
    "clinical_substitutions": _TrackSpec(
        "clinical_ProteinGym_substitutions.zip",
        "clinical_substitutions.csv",
        False,
        True,
    ),
    "clinical_indels": _TrackSpec(
        "clinical_ProteinGym_indels.zip", "clinical_indels.csv", True, True
    ),
}

# Column mappings: standardizes variable naming across different dataset release versions.
# uniprot_id/coarse_selection_type are mandatory for DMS tracks but absent from the
# clinical reference files, which have no UniProt grouping or functional-category split.
_REFERENCE_REQUIRED_COLUMN_ALIASES = {
    "dms_id": ["DMS_id"],
    "dms_filename": ["DMS_filename"],
    "target_seq": ["target_seq", "wt_seq", "sequence"],
}
_REFERENCE_OPTIONAL_COLUMN_ALIASES = {
    "uniprot_id": ["UniProt_ID"],
    "coarse_selection_type": ["coarse_selection_type"],
}

# Mandatory columns required in every individual assay file
_ASSAY_REQUIRED_COLUMN_ALIASES = {
    "mutated_sequence": ["mutated_sequence"],
}

# Optional assay columns preserved if present in source files:
# - mutant: mutation mutation codes (e.g., A12G)
# - dms_score: continuous experimental score
# - dms_score_bin / label: binary classification target
# - fold_*: official CV fold splits provided by ProteinGym
_ASSAY_OPTIONAL_COLUMN_ALIASES = {
    "mutant": ["mutant"],
    "dms_score": ["DMS_score"],
    "dms_score_bin": ["DMS_score_bin", "DMS_bin_score", "label"],
    "fold_random_5": ["fold_random_5"],
    "fold_modulo_5": ["fold_modulo_5"],
    "fold_contiguous_5": ["fold_contiguous_5"],
}


class ProteinGymSourcingConfig(BaseModel):
    """Configuration schema and validation for ProteinGym pipeline execution."""

    model_config = ConfigDict(extra="forbid")

    # Source locations
    base_url: str = _DEFAULT_BASE_URL
    version: str = _DEFAULT_VERSION
    reference_base_url: str = _DEFAULT_REFERENCE_BASE_URL
    track: ProteinGymTrack = "dms_substitutions"

    # Workspace & process controls
    work_dir: Path = Path("eval_datasets/proteingym_sourcing")
    combine_assays: bool = False
    seed: int = 1957723
    create_split_subsets: bool = True

    # Train/Validation partitioning parameters
    train_fraction: float = Field(default=0.8, gt=0, lt=1)
    validation_fraction: float = Field(default=0.1, gt=0, lt=1)

    # Sequence clustering thresholds for dataset splitting (MMseqs2/CD-HIT style)
    validation_cluster_identity_threshold: float = 0.3
    validation_cluster_coverage_threshold: float = 0.8
    validation_cluster_num_threads: int = 1

    # Hugging Face export parameters
    upload_layout: Literal["per_dataset_repo", "single_repo"] = "per_dataset_repo"
    repo_owner: str | None = None
    repo_name_prefix: str = "proteingym-"
    repo_id: str | None = None
    revision: str | None = None
    commit_message: str = "Add sourced ProteinGym dataset"
    private: bool = True
    token: str | None = None

    # Overwrite flags and resource allocation
    force_download: bool = False
    overwrite_output: bool = False
    stats_threads: int = Field(default=8, gt=0)

    @model_validator(mode="after")
    def _namespace_default_work_dir_by_track(self) -> "ProteinGymSourcingConfig":
        """Give each track its own subfolder under the shared default work_dir.

        Otherwise sourcing a second track with an unmodified config silently
        overwrites the first track's root-level artifacts (reference.parquet,
        stats.json, README.md). Explicit user-provided work_dir values are
        left untouched -- namespacing only applies to the class default.
        """
        if self.work_dir == type(self).model_fields["work_dir"].default:
            self.work_dir = self.work_dir / self.track
        return self

    @model_validator(mode="after")
    def _validate_split_fractions(self) -> "ProteinGymSourcingConfig":
        """Ensure split ratios leave remaining capacity (>0%) for the test set."""
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("train_fraction + validation_fraction must be less than 1")
        return self

    def resolve_upload_repo_id(self, dataset_name: str) -> str:
        """Resolve the target Hugging Face repository ID depending on chosen layout strategy."""
        if self.upload_layout == "single_repo":
            if not self.repo_id:
                raise ValueError("repo_id is required for single_repo uploads")
            return self.repo_id

        owner = self.repo_owner or (
            self.repo_id.split("/", 1)[0] if self.repo_id else None
        )
        if not owner:
            raise ValueError("repo_owner is required for per_dataset_repo uploads")
        return f"{owner}/{self.repo_name_prefix}{dataset_name}".strip("/")

    def has_upload_target(self) -> bool:
        """Check if remote repository configuration settings are valid for upload."""
        return (
            bool(self.repo_id)
            if self.upload_layout == "single_repo"
            else bool(self.repo_owner or self.repo_id)
        )


def _resolve_column(frame: pl.DataFrame, aliases: list[str], context: str) -> str:
    """Case-insensitively map expected column alias list to an actual column name in DataFrame.

    Raises ValueError if mandatory column is missing.
    """
    lower_map = {name.lower(): name for name in frame.columns}
    for alias in aliases:
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]
    raise ValueError(
        f"None of the expected {context} columns {aliases} were found; "
        f"available columns: {frame.columns}"
    )


def _resolve_optional_column(frame: pl.DataFrame, aliases: list[str]) -> str | None:
    """Case-insensitively resolve an optional column alias list. Returns None if absent."""
    lower_map = {name.lower(): name for name in frame.columns}
    for alias in aliases:
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]
    return None


def _download_file(url: str, destination: Path, force_download: bool) -> Path:
    """Download a remote resource safely using a temporary file to avoid partial writes."""
    if destination.exists() and not force_download:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s to %s", url, destination)

    # Write to a temporary .tmp file first, then replace target path atomically
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")
    urllib.request.urlretrieve(url, temporary_path)
    temporary_path.replace(destination)
    return destination


def _download_reference(
    config: ProteinGymSourcingConfig, downloads_dir: Path
) -> pl.DataFrame:
    """Download the track's reference CSV metadata file and canonicalize column names."""
    track = _TRACKS[config.track]
    reference_url = f"{config.reference_base_url}/{track.reference_filename}"
    reference_path = _download_file(
        reference_url, downloads_dir / track.reference_filename, config.force_download
    )

    # Read reference mapping and apply standardized column aliases
    raw = pl.read_csv(reference_path, infer_schema_length=10000)
    renamed = {
        canonical: _resolve_column(raw, aliases, canonical)
        for canonical, aliases in _REFERENCE_REQUIRED_COLUMN_ALIASES.items()
    }
    frame = raw.rename({source: canonical for canonical, source in renamed.items()})

    for canonical, aliases in _REFERENCE_OPTIONAL_COLUMN_ALIASES.items():
        source = _resolve_optional_column(raw, aliases)
        if source is not None:
            frame = frame.rename({source: canonical})

    if "uniprot_id" not in frame.columns:
        # Clinical reference files have one row per gene, so the DMS id doubles
        # as the grouping key the DMS aggregation hierarchy expects.
        frame = frame.with_columns(pl.col("dms_id").alias("uniprot_id"))
    if "coarse_selection_type" not in frame.columns:
        if not track.is_clinical:
            raise ValueError(
                "None of the expected coarse_selection_type columns "
                f"{_REFERENCE_OPTIONAL_COLUMN_ALIASES['coarse_selection_type']} were "
                f"found; available columns: {raw.columns}"
            )
        frame = frame.with_columns(pl.lit("Clinical").alias("coarse_selection_type"))
    return frame


def _download_and_extract_assays(
    config: ProteinGymSourcingConfig, downloads_dir: Path
) -> Path:
    """Download and un-zip the full track assay archive to local directory."""
    track = _TRACKS[config.track]
    zip_url = f"{config.base_url}/ProteinGym_{config.version}/{track.zip_filename}"
    zip_path = _download_file(
        zip_url, downloads_dir / track.zip_filename, config.force_download
    )

    extract_dir = downloads_dir / "assays" / config.track
    marker = extract_dir / ".extracted"

    # Use a hidden marker file to avoid unnecessary re-extraction
    if marker.exists() and not config.force_download:
        return extract_dir

    extract_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Extracting %s to %s", zip_path, extract_dir)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)

    marker.touch()
    return extract_dir


def _load_assay(extract_dir: Path, filename: str, dms_id: str) -> pl.DataFrame:
    """Locate, parse, and normalize individual assay CSV files into standard DataFrames."""
    # Find matching filename recursively in case extraction flattened/nested subfolders
    matches = sorted(extract_dir.rglob(filename))
    if not matches:
        raise FileNotFoundError(
            f"Assay CSV '{filename}' for {dms_id} not found under {extract_dir}."
        )
    raw = pl.read_csv(matches[0], infer_schema_length=10000)

    # Match mandatory columns
    canonical_to_source = {
        canonical: _resolve_column(raw, aliases, canonical)
        for canonical, aliases in _ASSAY_REQUIRED_COLUMN_ALIASES.items()
    }

    # Match optional columns if they exist
    for canonical, aliases in _ASSAY_OPTIONAL_COLUMN_ALIASES.items():
        source = _resolve_optional_column(raw, aliases)
        if source is not None:
            canonical_to_source[canonical] = source

    # Extract relevant columns, standardize names, and attach metadata
    frame = raw.rename(
        {source: canonical for canonical, source in canonical_to_source.items()}
    )
    frame = frame.select(list(canonical_to_source.keys())).with_columns(
        pl.lit(dms_id).alias("dms_id")
    )

    # Generate unique global variant identifier (`dms_id:variant_notation`)
    variant_key = (
        pl.coalesce([pl.col("mutant"), pl.col("mutated_sequence")])
        if "mutant" in frame.columns
        else pl.col("mutated_sequence")
    )
    frame = frame.with_columns(
        (pl.col("dms_id") + ":" + variant_key.cast(pl.Utf8)).alias("id")
    )

    # Verify uniqueness of constructed variant identifiers
    if frame["id"].n_unique() != frame.height:
        raise ValueError(f"Assay {dms_id} contains duplicate assay/mutant identifiers")
    return frame


def _add_seeded_splits(
    frame: pl.DataFrame,
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> pl.DataFrame:
    """Partition an assay into seeded splits when it has enough variants."""
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be less than 1")
    if frame.height < 3:
        return frame

    # Compute explicit item allocations, guaranteeing at least 1 sample per split split role
    train_size = max(1, int(frame.height * train_fraction))
    validation_size = max(1, int(frame.height * validation_fraction))
    test_size = frame.height - train_size - validation_size
    if test_size < 1:
        test_size = 1
        validation_size = max(1, frame.height - train_size - test_size)
    split_sizes = {
        "train": train_size,
        "validation": validation_size,
        "test": test_size,
    }

    # Generate partitioned indices using configured seed
    partition = random_partition_indices(frame.height, split_sizes, seed)

    # Assign split labels directly to DataFrame column
    split = np.empty(frame.height, dtype=object)
    for role, indices in partition.items():
        split[indices] = role
    return frame.with_columns(pl.Series("split", split))


def write_proteingym_card(
    output_dir: Path,
    config: ProteinGymSourcingConfig,
    track: _TrackSpec,
    combined: pl.DataFrame,
    reference: pl.DataFrame,
    split_names: list[str],
) -> Path:
    """Write a dataset card for the normalized ProteinGym track artifact."""
    archive_url = f"{config.base_url}/ProteinGym_{config.version}/{track.zip_filename}"
    reference_url = f"{config.reference_base_url}/{track.reference_filename}"
    assay_type = "clinical" if track.is_clinical else "DMS"
    variant_type = "indel" if track.is_indel else "substitution"
    score_columns = [
        column
        for column in ("dms_score", "dms_score_bin")
        if column in combined.columns
    ]
    optional_columns = [
        column
        for column in (
            "mutant",
            "fold_random_5",
            "fold_modulo_5",
            "fold_contiguous_5",
        )
        if column in combined.columns
    ]
    data_files = ["- `data/<dms_id>.parquet`: one normalized parquet file per assay."]
    if config.combine_assays or config.create_split_subsets:
        data_files.append(
            "- `data/combined.parquet`: all normalized assays concatenated."
        )
    if split_names:
        data_files.append(
            "- `data/combined_<split>.parquet`: canonical seeded and pooled "
            "random/stratified split views."
        )
    data_files.extend(
        [
            "- `reference.parquet`: assay metadata including UniProt IDs, "
            "selection type, and wild-type sequences.",
            "- `stats.json` and `stats/<track>_stats.parquet`: row-count and "
            "sequence-length diagnostics.",
        ]
    )
    split_section = (
        "When enabled, each assay receives deterministic `train`, `validation`, "
        f"and `test` labels using seed `{config.seed}` with fractions "
        f"`{config.train_fraction}`, `{config.validation_fraction}`, and the "
        "remaining rows. The generated pooled `*_random` and `*_stratified` "
        "views are alternative evaluation partitions and may overlap in rows; "
        "they are not additional independent data. MMseqs clustering is not "
        "used because variants within an assay share one wild-type sequence."
        if config.create_split_subsets
        else "No train/validation/test labels are generated. ProteinGym assays "
        "are retained as complete per-assay datasets."
    )
    if config.create_split_subsets:
        split_section += (
            " Assays with fewer than three variants are retained without split "
            "labels and are excluded from pooled split views."
        )
    score_section = (
        ", ".join(f"`{column}`" for column in score_columns)
        if score_columns
        else "no normalized score column was found"
    )
    optional_section = (
        ", ".join(f"`{column}`" for column in optional_columns)
        if optional_columns
        else "no optional mutation or official-fold columns were found"
    )
    card = (
        f"# ProteinGym {config.track}\n\n"
        f"ProteinGym `{config.version}` {assay_type} {variant_type} assays, "
        "normalized for use by the evaluation workflows in this repository. "
        "Each row represents one scored variant from one assay.\n\n"
        "## Provenance\n\n"
        f"- Reference metadata: [{track.reference_filename}]({reference_url})\n"
        f"- Assay archive: [{track.zip_filename}]({archive_url})\n"
        "- Project: [OATML-Markslab/ProteinGym]"
        "(https://github.com/OATML-Markslab/ProteinGym)\n"
        f"- Assays in this track: `{reference.height}`\n"
        f"- Variants in the combined artifact: `{combined.height}`\n\n"
        "## Data Files\n\n" + "\n".join(data_files) + "\n\n"
        "## Schema\n\n"
        "Every assay file contains `id`, `dms_id`, and `mutated_sequence`. "
        f"Normalized score columns present: {score_section}. Optional columns "
        f"present: {optional_section}.\n\n"
        "## Splits\n\n" + split_section + "\n\n"
        "## Loading\n\n"
        "These are ordinary parquet files rather than a Hugging Face DatasetDict "
        "with standard `train`/`validation`/`test` shards. Load an assay or a "
        "named combined view explicitly:\n\n"
        "```python\n"
        "import polars as pl\n\n"
        'assay = pl.read_parquet("data/<dms_id>.parquet")\n'
        'reference = pl.read_parquet("reference.parquet")\n'
        "```\n\n"
        "## Limitations\n\n"
        "ProteinGym scores and labels are assay-specific and should not be "
        "interpreted as directly comparable measurements across assays. The "
        "synthetic split labels are provided for repository evaluation and do "
        "not replace ProteinGym's official fold columns when those are present.\n"
    )
    card_path = output_dir / "README.md"
    card_path.write_text(card, encoding="utf-8")
    return card_path


def run_sourcing(config: ProteinGymSourcingConfig) -> Path:
    """Main execution pipeline: download raw data, process assays to parquet, and produce split subsets."""
    work_dir = config.work_dir.expanduser().resolve()
    downloads_dir = work_dir / "downloads"
    outputs_dir = work_dir / "outputs"

    # Cleanup output directory if overwrite flag is enabled
    if outputs_dir.exists() and config.overwrite_output:
        shutil.rmtree(outputs_dir)
    output_dir = outputs_dir / "data"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    track = _TRACKS[config.track]

    # Step 1: Fetch metadata index
    logger.info("Downloading ProteinGym '%s' reference file", config.track)
    reference = _download_reference(config, downloads_dir)
    logger.info("Processing all %d assays", reference.height)

    # Step 2: Fetch and un-zip assay files
    extract_dir = _download_and_extract_assays(config, downloads_dir)

    # Step 3: Parse each assay CSV and write per-assay Parquet files
    assay_frames: dict[str, pl.DataFrame] = {}
    for row in reference.iter_rows(named=True):
        dms_id, filename = row["dms_id"], row["dms_filename"]
        assay = _load_assay(extract_dir, filename, dms_id)
        if config.create_split_subsets:
            if assay.height < 3:
                logger.warning(
                    "Skipping seeded splits for %s: it contains only %d variants",
                    dms_id,
                    assay.height,
                )
            assay = _add_seeded_splits(
                assay,
                config.train_fraction,
                config.validation_fraction,
                config.seed,
            )
        assay.write_parquet(output_dir / f"{dms_id}.parquet")
        assay_frames[dms_id] = assay
        logger.info("Wrote %s (%d variants)", dms_id, assay.height)

    # Step 4: Concatenate individual assays into unified parquet files
    subset_frames: dict[str, pl.DataFrame] = {}
    combined = (
        pl.concat(list(assay_frames.values()), how="diagonal_relaxed")
        if assay_frames
        else pl.DataFrame()
    )

    if assay_frames and (config.combine_assays or config.create_split_subsets):
        combined.write_parquet(output_dir / "combined.parquet")

        # Step 5: Perform sequence clustering & create final split datasets if required
        if config.create_split_subsets:
            target_column = next(
                (
                    name
                    for name in ("dms_score", "dms_score_bin")
                    if name in combined.columns
                ),
                None,
            )
            if target_column is None:
                raise ValueError(
                    "ProteinGym split subsets require dms_score or dms_score_bin"
                )

            # Map canonical names expected by sequence clustering utilities
            split_input = combined.with_columns(
                pl.col("mutated_sequence").alias("sequence"),
                pl.col(target_column).alias("targets"),
            )
            canonical = {
                name: split_input.filter(pl.col("split") == name)
                for name in ("train", "validation", "test")
            }

            # Form cluster-aware subsets across splits
            subset_frames = create_split_subsets(
                canonical,
                seed=config.seed,
                identity_threshold=config.validation_cluster_identity_threshold,
                coverage_threshold=config.validation_cluster_coverage_threshold,
                num_threads=config.validation_cluster_num_threads,
                include_cluster=False,
            )

            # Write out individual combined split files (e.g. combined_train.parquet)
            for split_name, split_frame in subset_frames.items():
                split_frame.drop(["sequence", "targets"]).write_parquet(
                    output_dir / f"combined_{split_name}.parquet"
                )

    # Step 6: Process and export reference metadata
    reference_out = reference.select(
        ["dms_id", "uniprot_id", "coarse_selection_type", "target_seq"]
    ).with_columns(
        pl.lit(track.is_indel).alias("is_indel"),
        pl.lit(track.is_clinical).alias("is_clinical"),
    )
    reference_out.write_parquet(outputs_dir / "reference.parquet")

    # Step 7: Generate dataset statistics and FASTA/SeqKit sequence diagnostics
    stats_frames = (
        {
            split_name: split_frame.drop(["sequence", "targets"])
            for split_name, split_frame in subset_frames.items()
        }
        if subset_frames
        else {"all": combined}
    )
    write_stats(outputs_dir, stats_frames)

    seqkit_frames = {
        split_name: frame.select(
            [
                "id",
                pl.col("mutated_sequence").alias("sequence"),
            ]
        )
        for split_name, frame in stats_frames.items()
    }
    write_seqkit_stats(
        output_dir=outputs_dir,
        dataset_name=config.track,
        split_frames=seqkit_frames,
        threads=config.stats_threads,
    )

    write_proteingym_card(
        output_dir=outputs_dir,
        config=config,
        track=track,
        combined=combined,
        reference=reference,
        split_names=list(subset_frames),
    )

    # Step 8: Upload artifacts to Hugging Face Hub (if configured)
    if config.has_upload_target():
        upload_dataset_artifact(
            outputs_dir,
            f"{config.track}",
            config,
            resolve_hf_token(config.token),
        )

    logger.info("ProteinGym sourcing complete: %s", work_dir / "outputs")
    return work_dir / "outputs"


def main() -> None:
    """CLI entrypoint: initializes logging, parses CLI arguments, and runs sourcing."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Parse command line overrides directly into the Pydantic configuration class
    config = load_runtime_config_from_argv(
        sys.argv[1:], model_cls=ProteinGymSourcingConfig
    )
    run_sourcing(config)


if __name__ == "__main__":
    main()
