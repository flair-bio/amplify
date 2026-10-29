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

import hashlib
import logging
import shutil
import zipfile
from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np
import polars as pl
from pydantic import Field, model_validator

from modules.core.utils.hf_download import resolve_hf_token
from modules.evaluate.scripts.dataset_sourcing.shared.config import HubUploadConfig
from modules.evaluate.scripts.dataset_sourcing.shared.downloads import download_url
from modules.evaluate.scripts.dataset_sourcing.shared.hf_io import (
    upload_dataset_artifact,
)
from modules.evaluate.scripts.dataset_sourcing.shared.outputs import (
    generated_utc_date,
    write_dataset_stats,
)
from modules.evaluate.scripts.dataset_sourcing.shared.pipeline import run_script
from modules.evaluate.scripts.dataset_sourcing.shared.preprocess import (
    create_split_subsets,
    random_partition_indices,
)

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://marks.hms.harvard.edu/proteingym"
_DEFAULT_VERSION = "v1.3"
_DEFAULT_REFERENCE_BASE_URL = (
    "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/reference_files"
)
# One variant per train/validation/test role.
_MIN_ASSAY_VARIANTS_FOR_SPLITS = 3

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

_ASSAY_REQUIRED_COLUMN_ALIASES = {
    "mutated_sequence": ["mutated_sequence"],
}

# Optional assay columns preserved if present in source files:
# - mutant: mutation codes (e.g., A12G)
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


class ProteinGymSourcingConfig(HubUploadConfig):
    """Configuration schema and validation for ProteinGym pipeline execution."""

    base_url: str = _DEFAULT_BASE_URL
    version: str = _DEFAULT_VERSION
    # Defaults to the upstream main branch; pin to a commit to match `version` exactly.
    reference_base_url: str = _DEFAULT_REFERENCE_BASE_URL
    track: ProteinGymTrack = "dms_substitutions"

    work_dir: Path = Path("eval_datasets/proteingym_sourcing")
    combine_assays: bool = False
    seed: int = 1957723
    create_split_subsets: bool = True

    train_fraction: float = Field(default=0.8, gt=0, lt=1)
    validation_fraction: float = Field(default=0.1, gt=0, lt=1)

    repo_name_prefix: str = "proteingym-"
    commit_message: str = "Add sourced ProteinGym dataset"

    force_download: bool = False
    overwrite_output: bool = False
    stats_threads: int = Field(default=8, gt=0)

    @model_validator(mode="after")
    def _namespace_default_work_dir_by_track(self) -> ProteinGymSourcingConfig:
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
    def _validate_split_fractions(self) -> ProteinGymSourcingConfig:
        """Ensure split ratios leave remaining capacity (>0%) for the test set."""
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("train_fraction + validation_fraction must be less than 1")
        return self


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


def _download_reference(
    config: ProteinGymSourcingConfig, downloads_dir: Path
) -> pl.DataFrame:
    """Download the track's reference CSV metadata file and canonicalize column names."""
    track = _TRACKS[config.track]
    reference_url = f"{config.reference_base_url}/{track.reference_filename}"
    reference_path = download_url(
        reference_url,
        downloads_dir / config.version / track.reference_filename,
        force=config.force_download,
        cache_identity=reference_url,
    )

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
    zip_path = download_url(
        zip_url,
        downloads_dir / config.version / track.zip_filename,
        force=config.force_download,
        cache_identity=zip_url,
    )

    extract_dir = downloads_dir / config.version / "assays" / config.track
    # The marker records which archive URL was extracted.
    marker = extract_dir / ".extracted"
    if (
        marker.exists()
        and marker.read_text(encoding="utf-8") == zip_url
        and not config.force_download
    ):
        return extract_dir

    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True)
    logger.info("Extracting %s to %s", zip_path, extract_dir)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)

    marker.write_text(zip_url, encoding="utf-8")
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

    canonical_to_source = {
        canonical: _resolve_column(raw, aliases, canonical)
        for canonical, aliases in _ASSAY_REQUIRED_COLUMN_ALIASES.items()
    }
    for canonical, aliases in _ASSAY_OPTIONAL_COLUMN_ALIASES.items():
        source = _resolve_optional_column(raw, aliases)
        if source is not None:
            canonical_to_source[canonical] = source

    frame = raw.rename(
        {source: canonical for canonical, source in canonical_to_source.items()}
    )
    frame = frame.select(list(canonical_to_source.keys())).with_columns(
        pl.lit(dms_id).alias("dms_id")
    )

    # Indels have no mutant code, so fall back to the full mutated sequence.
    variant_key = (
        pl.coalesce([pl.col("mutant"), pl.col("mutated_sequence")])
        if "mutant" in frame.columns
        else pl.col("mutated_sequence")
    )
    frame = frame.with_columns(
        (pl.col("dms_id") + ":" + variant_key.cast(pl.Utf8)).alias("id")
    )

    if frame["id"].n_unique() != frame.height:
        raise ValueError(f"Assay {dms_id} contains duplicate assay/mutant identifiers")
    return frame


def _add_seeded_splits(
    frame: pl.DataFrame,
    train_fraction: float,
    validation_fraction: float,
    seed: int,
    dms_id: str = "",
) -> pl.DataFrame:
    """Partition an assay into seeded splits when it has enough variants."""
    if frame.height < _MIN_ASSAY_VARIANTS_FOR_SPLITS:
        return frame

    # Reserve at least one validation and test row before training gets the rest;
    # test absorbs the rounding remainder of 1 - train_fraction - validation_fraction.
    validation_size = max(1, int(frame.height * validation_fraction))
    test_size = max(
        1, frame.height - int(frame.height * train_fraction) - validation_size
    )
    split_sizes = {
        "train": frame.height - validation_size - test_size,
        "validation": validation_size,
        "test": test_size,
    }

    # Use a stable per-assay seed so partitions are reproducible and independent.
    assay_seed = (
        int.from_bytes(hashlib.sha256(f"{seed}:{dms_id}".encode()).digest()[:8], "big")
        if dms_id
        else seed
    )
    partition = random_partition_indices(frame.height, split_sizes, assay_seed)

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
            (
                "- `reference.parquet`: assay metadata including UniProt IDs, "
                "selection type, and wild-type sequences."
            ),
            (
                "- [`stats.json`](stats.json) at the dataset root: row counts, "
                "columns, and SeqKit sequence-length diagnostics."
            ),
        ]
    )
    split_section = (
        "When enabled, each assay receives deterministic `train`, `validation`, "
        f"and `test` labels using seed `{config.seed}` with fractions "
        f"`{config.train_fraction}`, `{config.validation_fraction}`, and the "
        "remaining rows. These are uniform random variant splits: variants at the "
        "same mutated position appear in several splits, so they overestimate "
        "generalization. For supervised benchmarking prefer ProteinGym's official "
        "`fold_contiguous_5`/`fold_modulo_5` columns when present. The pooled "
        "`*_random` and `*_stratified` views repartition the same rows; "
        "`*_stratified` balances each assay's share of every split (scores are "
        "not comparable across assays, so they are not used for stratification). "
        "MMseqs clustering is not used because variants within an assay share one "
        "wild-type sequence."
        if config.create_split_subsets
        else "No train/validation/test labels are generated. ProteinGym assays "
        "are retained as complete per-assay datasets."
    )
    if config.create_split_subsets:
        split_section += (
            f" Assays with fewer than {_MIN_ASSAY_VARIANTS_FOR_SPLITS} variants are "
            "retained without split labels and are excluded from pooled split views."
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
        f"- Reference metadata: [{track.reference_filename}]({reference_url}) "
        "(unpinned unless `reference_base_url` names a commit or tag)\n"
        f"- Assay archive: [{track.zip_filename}]({archive_url})\n"
        "- Project: [OATML-Markslab/ProteinGym]"
        "(https://github.com/OATML-Markslab/ProteinGym)\n"
        f"- Assays in this track: `{reference.height}`\n"
        f"- Variants in the combined artifact: `{combined.height}`\n"
        f"- Card generated (UTC): `{generated_utc_date()}`\n\n"
        "## Data Files\n\n" + "\n".join(data_files) + "\n\n"
        "## Intended use\n\n"
        "Variant-effect prediction and assay-level functional assessment. Scores "
        "support ranking or classification within an assay.\n\n"
        "## Schema\n\n"
        "Every assay file contains `id`, `dms_id`, and `mutated_sequence`. "
        f"Normalized score columns present: {score_section}. Optional columns "
        f"present: {optional_section}.\n\n"
        "## Processing\n\n"
        f"- `combine_assays`: `{str(config.combine_assays).lower()}`.\n"
        f"- `create_split_subsets`: `{str(config.create_split_subsets).lower()}`.\n\n"
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

    if outputs_dir.exists() and config.overwrite_output:
        shutil.rmtree(outputs_dir)
    output_dir = outputs_dir / "data"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    track = _TRACKS[config.track]

    logger.info("Downloading ProteinGym '%s' reference file", config.track)
    reference = _download_reference(config, downloads_dir)
    logger.info("Processing all %d assays", reference.height)
    extract_dir = _download_and_extract_assays(config, downloads_dir)

    assay_frames: dict[str, pl.DataFrame] = {}
    # Split within assays because variant scores and labels are assay-specific.
    for row in reference.iter_rows(named=True):
        dms_id, filename = row["dms_id"], row["dms_filename"]
        assay = _load_assay(extract_dir, filename, dms_id)
        if config.create_split_subsets:
            if assay.height < _MIN_ASSAY_VARIANTS_FOR_SPLITS:
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
                dms_id,
            )
        assay.write_parquet(output_dir / f"{dms_id}.parquet")
        assay_frames[dms_id] = assay
        logger.info("Wrote %s (%d variants)", dms_id, assay.height)

    subset_frames: dict[str, pl.DataFrame] = {}
    combined = (
        pl.concat(list(assay_frames.values()), how="diagonal_relaxed")
        if assay_frames
        else pl.DataFrame()
    )

    if assay_frames and (config.combine_assays or config.create_split_subsets):
        combined.write_parquet(output_dir / "combined.parquet")

        if config.create_split_subsets:
            canonical = {
                name: combined.filter(pl.col("split") == name)
                for name in ("train", "validation", "test")
            }
            # Variants share a wild type, so pooled alternatives do not need clustering.
            subset_frames = create_split_subsets(
                canonical, config.seed, stratification_column="dms_id"
            )
            for split_name, split_frame in subset_frames.items():
                split_frame.write_parquet(output_dir / f"combined_{split_name}.parquet")

    reference_out = reference.select(
        ["dms_id", "uniprot_id", "coarse_selection_type", "target_seq"]
    ).with_columns(
        pl.lit(track.is_indel).alias("is_indel"),
        pl.lit(track.is_clinical).alias("is_clinical"),
    )
    reference_out.write_parquet(outputs_dir / "reference.parquet")

    stats_frames = subset_frames or {"all": combined}
    seqkit_frames = {
        split_name: frame.select(
            [
                "id",
                pl.col("mutated_sequence").alias("sequence"),
            ]
        )
        for split_name, frame in stats_frames.items()
    }
    write_dataset_stats(
        output_dir=outputs_dir,
        split_frames=stats_frames,
        threads=config.stats_threads,
        seqkit_split_frames=seqkit_frames,
    )

    write_proteingym_card(
        output_dir=outputs_dir,
        config=config,
        track=track,
        combined=combined,
        reference=reference,
        split_names=list(subset_frames),
    )

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
    run_script(ProteinGymSourcingConfig, run_sourcing)


if __name__ == "__main__":
    main()
