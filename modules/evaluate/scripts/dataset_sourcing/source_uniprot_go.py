"""Source a GO-annotated protein dataset directly from the UniProt REST API.

Unlike the biomap-research/TAPE sources, these datasets have no upstream
train/validation/test split at all: :func:`preprocess.create_full_dataset_splits`
builds the canonical splits from a single MMseqs2-clustered pull, reducing
homology shared across splits.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import NamedTuple

import polars as pl
from pydantic import ConfigDict, Field

from modules.core.utils.hf_download import resolve_hf_token
from modules.evaluate.scripts.dataset_sourcing.shared.config import (
    UniProtLabelSupportConfig,
    UniProtReviewStatus,
)
from modules.evaluate.scripts.dataset_sourcing.shared.downloads import (
    UNIPROT_STREAM_URL,
    build_uniprot_query,
    download_uniprot_tsv,
    download_url,
    read_uniprot_tsv,
)
from modules.evaluate.scripts.dataset_sourcing.shared.hf_io import upload_dataset_artifact
from modules.evaluate.scripts.dataset_sourcing.shared.outputs import (
    generated_utc_date,
    mmseqs_settings_lines,
    write_dataset_stats,
    write_hf_split_files,
)
from modules.evaluate.scripts.dataset_sourcing.shared.pipeline import (
    prepare_output_dir,
    resolve_datasets,
    run_script,
)
from modules.evaluate.scripts.dataset_sourcing.shared.preprocess import (
    merge_label_sets_by_sequence,
    split_by_clusters,
)

logger = logging.getLogger(__name__)

GO_BASIC_OBO_URL = "https://purl.obolibrary.org/obo/go/go-basic.obo"
# Aspect roots annotate every protein in their aspect, so they carry no signal.
GO_ROOT_TERMS = frozenset({"GO:0003674", "GO:0008150", "GO:0005575"})

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


class GoOntology(NamedTuple):
    """``is_a``/``part_of`` parents, term names, and alt-id aliases from go-basic."""

    parents: dict[str, set[str]]
    names: dict[str, str]
    aliases: dict[str, str]
    version: str


class SourceUniprotGoConfig(UniProtLabelSupportConfig):
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


def read_go_ontology(path: Path) -> GoOntology:
    """Parse ``[Term]`` stanzas of go-basic.obo, whose relations never cross aspects."""
    parents: dict[str, set[str]] = {}
    names: dict[str, str] = {}
    aliases: dict[str, str] = {}
    term_id: str | None = None
    in_term = False
    version = "unknown"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("["):
            term_id = None
            in_term = line == "[Term]"
            continue
        key, _, value = line.partition(": ")
        if key == "data-version":
            version = value
        if not in_term:
            continue
        if key == "id":
            term_id = value
            parents[term_id] = set()
        elif term_id is None:
            continue
        elif key == "name":
            names[term_id] = value
        elif key == "alt_id":
            aliases[value] = term_id
        elif key == "is_a":
            parents[term_id].add(value.split(" ", 1)[0])
        elif key == "relationship" and value.startswith("part_of "):
            parents[term_id].add(value.split(" ")[1])
    return GoOntology(parents, names, aliases, version)


def propagate_go_ids(go_ids: set[str], ontology: GoOntology) -> set[str]:
    """Add every ancestor (true-path rule), dropping the uninformative aspect roots."""
    closure: set[str] = set()
    stack = [ontology.aliases.get(go_id, go_id) for go_id in go_ids]
    while stack:
        go_id = stack.pop()
        if go_id not in closure:
            closure.add(go_id)
            stack.extend(ontology.parents.get(go_id, ()))
    return closure - GO_ROOT_TERMS


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


def build_multihot_targets(
    go_ids_per_row: list[list[str]], vocabulary: list[str]
) -> list[list[int]]:
    """Encode each row against one fixed vocabulary, ignoring absent GO terms."""
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
    raw: pl.DataFrame,
    go_aspect: str,
    min_label_count: int,
    ontology: GoOntology,
) -> tuple[pl.DataFrame, list[str]]:
    """Parse one GO aspect into an id/sequence/targets frame and vocabulary.

    Annotations are propagated to all ancestors before counting support.
    """
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
    # Merge identical sequences first so duplicate entries cannot inflate label counts.
    rows = merge_label_sets_by_sequence(
        (
            str(accession),
            sequence.strip(),
            propagate_go_ids(set(parse_go_ids(cell)), ontology),
        )
        for accession, sequence, cell in raw.select(
            accession_col, sequence_col, go_col
        ).iter_rows()
        if isinstance(sequence, str) and sequence.strip()
    )
    go_ids_per_row = [sorted(go_ids) for _, _, go_ids in rows]
    vocabulary = build_go_vocabulary(go_ids_per_row, min_label_count)
    targets = build_multihot_targets(go_ids_per_row, vocabulary)

    frame = pl.DataFrame(
        {
            "id": [accession for accession, _, _ in rows],
            "sequence": [sequence for _, sequence, _ in rows],
        },
        schema={"id": pl.Utf8, "sequence": pl.Utf8},
    ).with_columns(pl.Series("targets", targets, dtype=pl.List(pl.Int8)))

    frame = frame.filter(pl.col("targets").list.sum() > 0)
    return frame, vocabulary


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
                "vocabulary_version": 2,
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
    ontology_version: str,
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
All UniProt GO evidence codes are included, including electronic (IEA)
annotations, many of which are themselves inferred from sequence signatures
(e.g. InterPro2GO); scores therefore partly reflect agreement with
sequence-based annotation pipelines.

## Source

- Selected GO aspect: `{go_aspect}`
- Card generated (UTC): `{generated_utc_date()}`
- Organism Taxonomy ID: `{organism_filter}` (`all` = all organisms).
- Review status: `{config.review_status}` (`reviewed` = Swiss-Prot, `unreviewed` = TrEMBL, `all` = both).
- GO ontology: `{GO_BASIC_OBO_URL}` (`data-version: {ontology_version}`).
UniProtKB TSV fields requested: `{requested_fields}`.

Query:

`{build_uniprot_query(config.organism_id, config.review_status)}`

## Preprocessing

- Read the selected aspect's Gene Ontology TSV field and extract distinct GO
  identifiers from each protein's annotation cell.
- Propagate each annotation to all of its `is_a`/`part_of` ancestors in
    go-basic (true-path rule), mapping alternate ids to primary ids. The
    aspect root term is dropped because every protein would carry it.
- Consider GO identifiers found in at least {config.min_label_count} query-result
    proteins after propagation. After splitting, keep terms with at least
    {config.min_train_label_count} positives in training,
    {config.min_validation_label_count} in validation, and
    {config.min_test_label_count} in test. Sort the
    retained identifiers lexicographically; this order defines
    the target-vector positions in `label_vocabulary.json`. Human-readable term
    names from go-basic are recorded there in `label_names`.
- Keep rows with a non-empty sequence and at least one retained GO identifier.
- Apply the optional sequence-length filter before splitting these rows by
    sequence clusters. Sequences are whitespace-trimmed. {length_filter}
- Output columns are `id`, `sequence`, `targets`, and `split`. `id` is the
  UniProtKB accession of the first entry carrying each unique sequence.

## Labels

`targets` is a multi-hot vector over {len(vocabulary)} GO ids (aspect:
`{go_aspect}`). Each term meets the configured source and canonical split
support cutoffs. Because validation and test support select the terms, scores
are conditional on that label selection. The ordered vocabulary is in `label_vocabulary.json`.

## Splitting and MMseqs2

MMseqs2 clusters the eligible sequences once. The effective clustering
settings are:

{mmseqs_settings_lines(config)}

Whole clusters are assigned to exactly one split. Target row counts are
calculated from fractions `{config.split_fractions}` (rounded to the nearest
integer, with rounding drift assigned to the largest split). Clusters are
considered largest-first and greedily assigned to the split with the greatest
relative size deficit; the configured seed `{config.seed}` breaks ties between
equal-sized clusters. No upstream split assignment is available.

When `create_split_subsets` is enabled, pooled random, stratified (by each
protein's globally rarest positive term), and hold-cluster-out subsets are
also included. Hold-cluster-out subsets require enough MMseqs clusters to
populate all three roles.

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
    """Build selected GO aspects, filter support after splitting, and optionally upload."""
    go_aspects = resolve_datasets(
        config.datasets, config.all_datasets, list(GO_ASPECT_FIELDS)
    )
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
    raw = read_uniprot_tsv(tsv_path, config.max_sequence_length)
    ontology = read_go_ontology(
        download_url(
            GO_BASIC_OBO_URL,
            downloads_dir / "go-basic.obo",
            force=config.force_download,
        )
    )
    output_dirs: list[Path] = []

    for go_aspect in go_aspects:
        dataset_name = _dataset_name_for_aspect(config, go_aspect, len(go_aspects))
        frame, vocabulary = frame_from_uniprot_table(
            raw, go_aspect, config.min_label_count, ontology
        )
        split_frames, retained = split_by_clusters(
            frame,
            config,
            dataset_name,
            min_support_by_split={
                "train": config.min_train_label_count,
                "validation": config.min_validation_label_count,
                "test": config.min_test_label_count,
            },
            multilabel=True,
        )
        assert retained is not None
        vocabulary = [vocabulary[index] for index in retained]
        label_names = {label: ontology.names.get(label, label) for label in vocabulary}

        output_dir = prepare_output_dir(config, dataset_name)
        write_hf_split_files(output_dir, split_frames)
        write_dataset_stats(output_dir, split_frames, threads=config.stats_threads)
        write_go_vocabulary(output_dir, go_aspect, vocabulary, label_names)
        write_go_card(
            output_dir,
            config,
            dataset_name,
            go_aspect,
            split_frames,
            vocabulary,
            ontology.version,
        )
        upload_dataset_artifact(output_dir, dataset_name, config, token)
        output_dirs.append(output_dir)
    return output_dirs


def main() -> None:
    run_script(SourceUniprotGoConfig, run)


if __name__ == "__main__":
    main()
