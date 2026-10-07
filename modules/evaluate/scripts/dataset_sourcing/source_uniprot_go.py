"""Source a GO-annotated protein dataset directly from the UniProt REST API.

Unlike the biomap-research/TAPE sources, these datasets have no upstream
train/validation/test split at all: :func:`preprocess.create_full_dataset_splits`
builds the canonical splits from a single MMseqs2-clustered pull, so no split
contains near-duplicates of another split's sequences.
"""

from __future__ import annotations

import json
import logging
import math
import re
import shutil
import sys
from pathlib import Path

import polars as pl
from pydantic import ConfigDict, Field, model_validator

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
    minimal_preprocess,
)

logger = logging.getLogger(__name__)

# GO aspect -> UniProtKB TSV field name (each cell looks like
# "protein binding [GO:0005515]; enzyme binding [GO:0019899]").
GO_ASPECT_FIELDS = {
    "molecular_function": "go_f",
    "biological_process": "go_p",
    "cellular_component": "go_c",
}

# GO aspect -> the human-readable column header UniProt's TSV export actually
# uses for that field (not the field code itself).
GO_ASPECT_HEADERS = {
    "molecular_function": "Gene Ontology (molecular function)",
    "biological_process": "Gene Ontology (biological process)",
    "cellular_component": "Gene Ontology (cellular component)",
}

# GO aspect -> base dataset name.
GO_ASPECT_BASE_NAMES = {
    "molecular_function": "GO_mf",
    "biological_process": "GO_bp",
    "cellular_component": "GO_cc",
}

_GO_ID_RE = re.compile(r"GO:\d+")
_GO_ANNOTATION_RE = re.compile(r"([^;]*?)\s*\[(GO:\d+)\]")

# Minimum expected positive examples of a label in the smallest split for a
# probe to have a reasonable chance of learning it (and for that split's
# metric on it to be a stable estimate, not 1-2 lucky/unlucky examples).
MIN_POSITIVE_EXAMPLES_PER_SPLIT = 10


class SourceUniprotGoConfig(UniProtDatasetSourcingConfig):
    """Config for sourcing one or more GO-aspect datasets from UniProtKB."""

    model_config = ConfigDict(extra="forbid")

    all_datasets: bool = False
    datasets: list[str] | None = Field(default_factory=lambda: ["molecular_function"])

    work_dir: Path = Path("datasets/uniprot_go_sourcing")
    repo_name_prefix: str = "uniprot-"
    repo_prefix: str = "uniprot"
    commit_message: str = "Add sourced UniProt GO annotation dataset"

    # None: auto-derived from the selected aspect + max_sequence_length below
    # (e.g. 'GO_mf' with no cutoff, 'GO_mf_512_cutoff' with max_sequence_length=512).
    # Only valid when one aspect is selected; set explicitly to override.
    dataset_name: str | None = None
    review_status: UniProtReviewStatus = "reviewed"
    # None: auto-derived from split_fractions below, so the smallest split
    # gets ~MIN_POSITIVE_EXAMPLES_PER_SPLIT positives of every kept label
    # (assuming labels are distributed roughly proportionally to split size).
    # Set explicitly to override.
    min_label_count: int | None = Field(
        None,
        description=(
            "Drop GO terms observed in fewer than this many proteins before "
            "building the multi-hot vocabulary, so the label set excludes "
            "terms too rare to learn from a probe."
        ),
    )
    # Default split proportions are 0.8/0.1/0.1.
    split_fractions: dict[str, float] = Field(
        default_factory=lambda: {"train": 0.8, "validation": 0.1, "test": 0.1}
    )

    @model_validator(mode="after")
    def _default_min_label_count(self) -> SourceUniprotGoConfig:
        if self.min_label_count is None:
            smallest_split_fraction = min(self.split_fractions.values())
            self.min_label_count = math.ceil(
                MIN_POSITIVE_EXAMPLES_PER_SPLIT / smallest_split_fraction
            )
        return self


def _dataset_name_for_aspect(
    config: SourceUniprotGoConfig, go_aspect: str, selected_count: int
) -> str:
    if config.dataset_name is not None:
        if selected_count != 1:
            raise ValueError(
                "dataset_name is only valid when one GO aspect is selected"
            )
        return config.dataset_name

    base = GO_ASPECT_BASE_NAMES[go_aspect]
    if config.max_sequence_length is not None:
        base = f"{base}_{config.max_sequence_length}_cutoff"
    return base


def _requested_uniprot_fields() -> list[str]:
    return [
        "accession",
        "reviewed",
        "id",
        "protein_name",
        "gene_names",
        "organism_name",
        "length",
        "sequence",
        *GO_ASPECT_FIELDS.values(),
    ]


def parse_go_ids(cell: str | None) -> list[str]:
    """Extract every ``GO:0000000``-style id from a UniProt GO TSV cell."""
    if not cell:
        return []
    return sorted(set(_GO_ID_RE.findall(cell)))


def build_go_vocabulary(
    go_ids_per_row: list[list[str]], min_label_count: int
) -> list[str]:
    """Return the sorted, deterministic vocabulary of GO ids observed at
    least ``min_label_count`` times, dropping terms too rare to learn from."""
    counts: dict[str, int] = {}
    for go_ids in go_ids_per_row:
        for go_id in go_ids:
            counts[go_id] = counts.get(go_id, 0) + 1
    return sorted(go_id for go_id, count in counts.items() if count >= min_label_count)


def build_go_label_names(
    annotation_cells: list[str | None], vocabulary: list[str]
) -> dict[str, str]:
    """Map retained GO ids to names supplied in UniProt annotation cells."""
    retained_ids = set(vocabulary)
    names: dict[str, str] = {}
    for cell in annotation_cells:
        if not cell:
            continue
        for name, go_id in _GO_ANNOTATION_RE.findall(cell):
            name = name.strip()
            if go_id in retained_ids and name:
                names.setdefault(go_id, name)
    return {go_id: names.get(go_id, go_id) for go_id in vocabulary}


def build_multihot_targets(
    go_ids_per_row: list[list[str]], vocabulary: list[str]
) -> list[list[int]]:
    # A fixed vocabulary order gives each vector position one consistent GO term.
    vocab_index = {go_id: i for i, go_id in enumerate(vocabulary)}
    targets: list[list[int]] = []
    for go_ids in go_ids_per_row:
        vector = [0] * len(vocabulary)
        for go_id in go_ids:
            index = vocab_index.get(go_id)
            if index is not None:
                vector[index] = 1
        targets.append(vector)
    return targets


def frame_from_uniprot_table(
    raw: pl.DataFrame, go_aspect: str, min_label_count: int
) -> tuple[pl.DataFrame, list[str]]:
    """Parse one GO aspect into an id/sequence/targets frame and vocabulary."""
    go_column = GO_ASPECT_HEADERS[go_aspect]
    column_map = {name.lower(): name for name in raw.columns}
    accession_col = column_map.get("entry", raw.columns[0])
    sequence_col = column_map.get("sequence")
    go_col = column_map.get(go_column.lower())
    if sequence_col is None or go_col is None:
        raise ValueError(
            f"UniProt TSV is missing expected columns 'Sequence'/'{go_column}'. "
            f"Found: {raw.columns}"
        )

    go_ids_per_row = [parse_go_ids(cell) for cell in raw[go_col].to_list()]
    vocabulary = build_go_vocabulary(go_ids_per_row, min_label_count)
    targets = build_multihot_targets(go_ids_per_row, vocabulary)

    frame = raw.select(
        pl.col(accession_col).alias("id"),
        pl.col(sequence_col).alias("sequence"),
    ).with_columns(pl.Series("targets", targets, dtype=pl.List(pl.Int64)))

    frame = frame.filter(
        pl.col("sequence").is_not_null()
        & (pl.col("sequence").str.len_chars() > 0)
        & (pl.col("targets").list.sum() > 0)
    )
    return frame, vocabulary


def frame_from_uniprot_tsv(
    tsv_path: Path, go_aspect: str, min_label_count: int
) -> tuple[pl.DataFrame, list[str]]:
    """Parse one GO aspect from a downloaded UniProt TSV."""
    return frame_from_uniprot_table(
        pl.read_csv(tsv_path, separator="\t"), go_aspect, min_label_count
    )


def write_go_vocabulary(
    output_dir: Path,
    go_aspect: str,
    vocabulary: list[str],
    label_names: dict[str, str],
) -> Path:
    path = output_dir / "label_vocabulary.json"
    path.write_text(
        json.dumps(
            {
                "go_aspect": go_aspect,
                "vocabulary_version": 1,
                "labels": vocabulary,
                "label_names": label_names,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def write_go_card(
    output_dir: Path,
    config: SourceUniprotGoConfig,
    dataset_name: str,
    go_aspect: str,
    split_frames: dict[str, pl.DataFrame],
    vocabulary: list[str],
) -> Path:
    rows = "\n".join(
        f"- `{split}`: {frame.height} rows"
        for split, frame in sorted(split_frames.items())
    )
    requested_fields = ", ".join(_requested_uniprot_fields())
    organism_filter = config.organism_id if config.organism_id is not None else "all"
    length_filter = (
        f"Sequences longer than `{config.max_sequence_length}` residues are removed before vocabulary construction and MMseqs2 clustering."
        if config.max_sequence_length is not None
        else "No maximum sequence-length filter was applied."
    )
    card = f"""# {dataset_name}

Sourced from the UniProt REST API (`{UNIPROT_STREAM_URL}`) and prepared for
Hugging Face datasets usage.

## Intended use

Protein-level Gene Ontology annotation prediction for the `{go_aspect}` aspect.
Labels measure recovery of UniProt annotations, not experimental causality.

## Source

- Selected GO aspect: `{go_aspect}`
- Card generated (UTC): `{generated_utc_date()}`
- Organism Taxonomy ID: `{organism_filter}` (`all` = all organisms).
- Review status: `{config.review_status}` (`reviewed` = Swiss-Prot, `unreviewed` = TrEMBL, `all` = both).
UniProtKB TSV fields requested: `{requested_fields}`.

Query:

`{build_uniprot_query(config.organism_id, config.review_status)}`

## Preprocessing

- Read the selected aspect's Gene Ontology TSV field and extract distinct GO
  identifiers from each protein's annotation cell.
- Keep GO identifiers found in at least {config.min_label_count} query-result
    proteins, then discard identifiers absent from the training split. Sort the
    retained identifiers lexicographically; this order defines
    the target-vector positions in `label_vocabulary.json`. Human-readable term
    names are recorded there in `label_names`, keyed by GO identifier.
- Keep rows with a non-empty sequence and at least one retained GO identifier.
- Apply the optional sequence-length filter before splitting these rows by
    sequence clusters. Sequences are whitespace-trimmed. {length_filter}
- Output columns are `id`, `sequence`, `targets`, and `split`. IDs are generated
  as `<dataset>_<split>_<row-index>` after split assignment.

## Labels

`targets` is a multi-hot vector over {len(vocabulary)} GO ids (aspect:
`{go_aspect}`), each observed in at least {config.min_label_count}
proteins in the query result and present in training. The ordered vocabulary is in
`label_vocabulary.json`.

## Splitting and MMseqs2

MMseqs2 clusters the eligible sequences in one `easy-linclust` pass. The
effective clustering settings are:

- Minimum sequence identity (`--min-seq-id`): `{config.validation_cluster_identity_threshold}`
- Minimum coverage (`-c`): `{config.validation_cluster_coverage_threshold}`
- Coverage mode (`--cov-mode`): `1`
- Threads (`--threads`): `{config.validation_cluster_num_threads}`
- Linclust algorithm version (`--linclust-version`): `2`
- Other MMseqs2 options are omitted and use MMseqs2 defaults.

Whole clusters are assigned to exactly one split. Target row counts are
calculated from fractions `{config.split_fractions}` (rounded to the nearest
integer, with rounding drift assigned to the largest split). Clusters are
considered largest-first and greedily assigned to the split with the greatest
relative size deficit; the configured seed `{config.seed}` breaks ties between
equal-sized clusters. No upstream split assignment is available.

When `create_split_subsets` is enabled, pooled random, stratified, and
hold-cluster-out subsets are also included. Hold-cluster-out subsets require
enough MMseqs clusters to populate all three roles.

Split row counts after preprocessing:

{rows}

## Dataset statistics

[`stats.json`](stats.json) at the dataset root contains row counts by split,
columns, and SeqKit sequence-length metrics.
"""
    path = output_dir / "README.md"
    path.write_text(card, encoding="utf-8")
    return path


def run(config: SourceUniprotGoConfig) -> list[Path]:
    go_aspects = resolve_datasets(
        config.datasets, config.all_datasets, list(GO_ASPECT_FIELDS)
    )
    assert config.min_label_count is not None
    if config.dataset_name is not None and len(go_aspects) != 1:
        raise ValueError("dataset_name is only valid when one GO aspect is selected")

    token = resolve_hf_token(config.token)
    downloads_dir = config.work_dir / "downloads"
    tsv_path = download_uniprot_tsv(
        config,
        downloads_dir,
        "uniprot_go.tsv.gz",
        _requested_uniprot_fields(),
    )
    raw = pl.read_csv(tsv_path, separator="\t")
    if config.max_sequence_length is not None:
        sequence_col = next(col for col in raw.columns if col.lower() == "sequence")
        raw = raw.filter(
            pl.col(sequence_col).str.len_chars() <= config.max_sequence_length
        )
    output_dirs: list[Path] = []

    for go_aspect in go_aspects:
        dataset_name = _dataset_name_for_aspect(config, go_aspect, len(go_aspects))
        frame, vocabulary = frame_from_uniprot_table(
            raw, go_aspect, config.min_label_count
        )
        go_column = GO_ASPECT_HEADERS[go_aspect]
        column_map = {name.lower(): name for name in raw.columns}
        label_names = build_go_label_names(
            raw[column_map[go_column.lower()]].to_list(), vocabulary
        )
        split_frames = create_full_dataset_splits(
            frame,
            split_fractions=config.split_fractions,
            seed=config.seed,
            identity_threshold=config.validation_cluster_identity_threshold,
            coverage_threshold=config.validation_cluster_coverage_threshold,
            num_threads=config.validation_cluster_num_threads,
        )
        # Held-out labels must not expand the training task vocabulary.
        train_targets = split_frames["train"]["targets"].to_list()
        retained = [
            index
            for index in range(len(vocabulary))
            if any(target[index] for target in train_targets)
        ]
        if not retained:
            raise ValueError("No GO terms remain in the training split.")
        vocabulary = [vocabulary[index] for index in retained]
        label_names = {label: label_names[label] for label in vocabulary}
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
            dataset_name,
            split_frames,
            config.max_sequence_length,
            column_rename={},
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
        write_go_vocabulary(output_dir, go_aspect, vocabulary, label_names)
        write_go_card(
            output_dir, config, dataset_name, go_aspect, split_frames, vocabulary
        )
        if config.has_upload_target():
            upload_dataset_artifact(output_dir, dataset_name, config, token)
        output_dirs.append(output_dir)
    return output_dirs


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        stream=sys.stdout,
        force=True,
    )
    config = load_runtime_config_from_argv(
        sys.argv[1:], model_cls=SourceUniprotGoConfig
    )
    run(config)


if __name__ == "__main__":
    main()
