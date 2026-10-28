"""Source residue-level annotation datasets from UniProtKB."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import polars as pl
from pydantic import ConfigDict, Field

from modules.core.utils.hf_download import resolve_hf_token
from modules.evaluate.scripts.dataset_sourcing.shared.config import UniProtDatasetSourcingConfig
from modules.evaluate.scripts.dataset_sourcing.shared.downloads import (
    UNIPROT_STREAM_URL,
    build_uniprot_query,
    download_uniprot_tsv,
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
from modules.evaluate.scripts.dataset_sourcing.shared.preprocess import split_by_clusters

logger = logging.getLogger(__name__)

BACKGROUND_LABEL = "-"
AMBIGUOUS_LABEL = "?"
IGNORE_INDEX = -100
# Disulfide coordinates name the two bonded cysteines, not a residue span.
ENDPOINT_FEATURE_TYPES = frozenset({"DISULFID"})

# UniProt TSV display header -> REST API field code.
FEATURE_FIELDS = {
    "Helix": "ft_helix",
    "Beta strand": "ft_strand",
    "Turn": "ft_turn",
    "Modified residue": "ft_mod_res",
    "Lipidation": "ft_lipid",
    "Disulfide bond": "ft_disulfid",
    "Glycosylation": "ft_carbohyd",
    "Transmembrane": "ft_transmem",
    "Intramembrane": "ft_intramem",
    "Topological domain": "ft_topo_dom",
    "Propeptide": "ft_propep",
    "Signal peptide": "ft_signal",
    "Transit peptide": "ft_transit",
    "Active site": "ft_act_site",
    "Binding site": "ft_binding",
    "DNA binding": "ft_dna_bind",
    "Domain [FT]": "ft_domain",
    "Region": "ft_region",
    "Repeat": "ft_repeat",
    "Coiled coil": "ft_coiled",
    "Zinc finger": "ft_zn_fing",
}

# UniProt TSV display header -> feature-type keyword that prefixes each entry in that column's cells.
FEATURE_TYPES = {
    "Helix": "HELIX",
    "Beta strand": "STRAND",
    "Turn": "TURN",
    "Modified residue": "MOD_RES",
    "Lipidation": "LIPID",
    "Disulfide bond": "DISULFID",
    "Glycosylation": "CARBOHYD",
    "Transmembrane": "TRANSMEM",
    "Intramembrane": "INTRAMEM",
    "Topological domain": "TOPO_DOM",
    "Propeptide": "PROPEP",
    "Signal peptide": "SIGNAL",
    "Transit peptide": "TRANSIT",
    "Active site": "ACT_SITE",
    "Binding site": "BINDING",
    "DNA binding": "DNA_BIND",
    "Domain [FT]": "DOMAIN",
    "Region": "REGION",
    "Repeat": "REPEAT",
    "Coiled coil": "COILED",
    "Zinc finger": "ZN_FING",
}

# Dataset name -> feature headers combined into its per-residue target.
ANNOTATION_GROUPS: dict[str, tuple[str, ...]] = {
    "uniprot_secondary_structure": ("Helix", "Beta strand", "Turn"),
    "uniprot_post_translational_modification": (
        "Modified residue",
        "Lipidation",
        "Disulfide bond",
        "Glycosylation",
    ),
    "uniprot_glycosylation": ("Glycosylation",),
    "uniprot_phosphorylation": ("Modified residue",),
    "uniprot_lipidation": ("Lipidation",),
    "uniprot_membrane_pass": ("Transmembrane", "Intramembrane"),
    "uniprot_topology": ("Topological domain",),
    "uniprot_peptide": ("Propeptide", "Signal peptide", "Transit peptide"),
    "uniprot_functional_sites": ("Active site", "Binding site", "DNA binding"),
    "uniprot_domains": ("Domain [FT]",),
    "uniprot_regions": ("Region",),
    "uniprot_structures": ("Repeat", "Coiled coil", "Zinc finger"),
}

# Groups whose classes come from each feature's /note text instead of its
# feature-type keyword.
NOTE_LABEL_DATASETS = frozenset(
    {
        "uniprot_glycosylation",
        "uniprot_phosphorylation",
        "uniprot_lipidation",
        "uniprot_topology",
    }
)

# Positions may be fuzzy (<5, >9) or unknown (?).
_COORDINATE = r"(?:[<>?]?\d+|\?)"
# One match per feature: "TYPE start[..end]" at the cell start or after "; ".
_FEATURE_RE = re.compile(
    rf"(?:^|;\s*)([A-Z_]+)\s+({_COORDINATE})(?:\.\.({_COORDINATE}))?"
)
_NOTE_RE = re.compile(r'/note="([^"]*)"')
_PARENTHETICAL_RE = re.compile(r"\s*\([^)]*\)")
_EVIDENCE_RE = re.compile(r'/evidence\s*=\s*(?:"([^"]*)"|([^;]*))')
# Lookarounds reject matches inside a longer token (e.g. ECO:00002691).
_ALLOWED_EVIDENCE_RE = re.compile(r"(?<![A-Z0-9])ECO:(?:0000269|0000305)(?![A-Z0-9])")
_PDB_STRUCTURE_EVIDENCE_RE = re.compile(
    r"(?<![A-Z0-9])ECO:0007829\|PDB:[A-Z0-9]+(?![A-Z0-9])"
)


class SourceUniprotAnnotationsConfig(UniProtDatasetSourcingConfig):
    """Config for sourcing UniProt feature annotations as residue labels."""

    model_config = ConfigDict(extra="forbid")

    all_datasets: bool = True
    work_dir: Path = Path("eval_datasets/uniprot_annotations_sourcing")
    repo_name_prefix: str = ""
    repo_prefix: str = "uniprot"
    commit_message: str = "Add sourced UniProt residue annotation datasets"
    organism_id: int | None = Field(9606, ge=1)


def _requested_uniprot_fields() -> list[str]:
    used_features = {
        feature for group in ANNOTATION_GROUPS.values() for feature in group
    }
    return [
        "accession",
        "sequence",
        "length",
        *(field for name, field in FEATURE_FIELDS.items() if name in used_features),
    ]


def _has_allowed_evidence(
    cell: str,
    block_start: int,
    block_end: int,
    note_spans: list[tuple[int, int]],
    feature_header: str,
) -> bool:
    """Check only this feature block; `/note` text is not evidence.

    PDB-derived evidence is accepted only for secondary-structure features.
    """
    for match in _EVIDENCE_RE.finditer(cell, block_start, block_end):
        if any(start <= match.start() < end for start, end in note_spans):
            continue
        evidence = match.group(1) or match.group(2) or ""
        if _ALLOWED_EVIDENCE_RE.search(evidence) or (
            feature_header in ANNOTATION_GROUPS["uniprot_secondary_structure"]
            and _PDB_STRUCTURE_EVIDENCE_RE.search(evidence)
        ):
            return True
    return False


def _note_label(note: str) -> str | None:
    """Label from a feature note with parentheticals removed.

    ``N-linked (GlcNAc...) asparagine`` becomes ``N-linked asparagine``. Notes
    opening with a qualifier such as ``(Microbial infection)`` return ``None``
    so the generic feature code is used instead.
    """
    if note.startswith("("):
        return None
    label = _PARENTHETICAL_RE.sub("", note.split(";", 1)[0]).strip()
    return label or None


def build_residue_labels(
    feature_cells: dict[str, str | None],
    sequence_length: int,
    use_note: bool = False,
    phosphorylation_only: bool = False,
) -> tuple[list[str], bool]:
    """Expand UniProt feature coordinates into one deterministic label per residue.

    Residues covered by conflicting features, or by a relevant feature whose
    evidence is not accepted or whose position is uncertain (``?``), are
    ``AMBIGUOUS_LABEL`` (ignored in training). Also returns whether any
    relevant feature was rejected for its evidence or position.
    """
    residue_labels: list[set[str]] = [set() for _ in range(sequence_length)]
    has_evidence_rejected_feature = False

    for feature_header, cell in feature_cells.items():
        if not cell:
            continue
        expected_type = FEATURE_TYPES.get(feature_header)
        if expected_type is None:
            raise ValueError(f"Unsupported UniProt feature header: {feature_header}")
        note_spans = [match.span() for match in _NOTE_RE.finditer(cell)]
        matches = [
            match
            for match in _FEATURE_RE.finditer(cell)
            if not any(start <= match.start() < end for start, end in note_spans)
        ]
        for match_index, match in enumerate(matches):
            if match.group(1) != expected_type:
                continue
            start_text = match.group(2)
            end_text = match.group(3) or start_text
            start_match = re.search(r"\d+", start_text)
            end_match = re.search(r"\d+", end_text)
            uncertain = "?" in start_text or "?" in end_text
            if start_match is None or end_match is None:
                # Unknown position: the feature exists but cannot be placed.
                has_evidence_rejected_feature |= uncertain
                continue
            start = int(start_match.group())
            end = int(end_match.group())
            if start < 1 or end < start:
                continue

            block_end = (
                matches[match_index + 1].start()
                if match_index + 1 < len(matches)
                else len(cell)
            )
            feature_label = match.group(1)
            if use_note:
                note_match = _NOTE_RE.search(cell[match.end() : block_end])
                if note_match:
                    feature_label = _note_label(note_match.group(1)) or feature_label
            feature_label = feature_label.replace(" ", "_")
            if phosphorylation_only and "phospho" not in feature_label.casefold():
                continue
            # Evidence is scoped to this feature block, not neighboring annotations.
            if uncertain or not _has_allowed_evidence(
                cell, match.start(), block_end, note_spans, feature_header
            ):
                has_evidence_rejected_feature = True
                feature_label = AMBIGUOUS_LABEL

            # UniProt coordinates are 1-based inclusive; convert to 0-based indices.
            positions = (
                (start - 1, end - 1)
                if expected_type in ENDPOINT_FEATURE_TYPES
                else range(start - 1, end)
            )
            for residue_index in positions:
                if residue_index < sequence_length:
                    residue_labels[residue_index].add(feature_label)

    return (
        [
            next(iter(labels))
            if len(labels) == 1
            else (AMBIGUOUS_LABEL if labels else BACKGROUND_LABEL)
            for labels in residue_labels
        ],
        has_evidence_rejected_feature,
    )


def _has_annotated_label(labels: list[str]) -> bool:
    return any(label not in (BACKGROUND_LABEL, AMBIGUOUS_LABEL) for label in labels)


def merge_residue_labels(first: list[str], second: list[str]) -> list[str]:
    """Combine labels from two entries sharing a sequence; disagreements are ambiguous."""
    return [
        a
        if a == b or b == BACKGROUND_LABEL
        else b
        if a == BACKGROUND_LABEL
        else AMBIGUOUS_LABEL
        for a, b in zip(first, second, strict=True)
    ]


def _frame_from_uniprot_table(
    raw: pl.DataFrame, dataset_name: str
) -> tuple[pl.DataFrame, list[str]]:
    """Merge identical sequences and encode residue labels plus ignore positions."""
    if dataset_name not in ANNOTATION_GROUPS:
        raise ValueError(
            f"Unknown UniProt annotation dataset {dataset_name!r}. "
            f"Choose from: {sorted(ANNOTATION_GROUPS)}"
        )

    column_map = {name.lower(): name for name in raw.columns}
    accession_col = column_map.get("entry", raw.columns[0])
    sequence_col = column_map.get("sequence")
    feature_headers = ANNOTATION_GROUPS[dataset_name]
    missing = [header for header in feature_headers if header.lower() not in column_map]
    if sequence_col is None or missing:
        raise ValueError(
            f"UniProt TSV is missing required columns for {dataset_name}: "
            f"sequence={sequence_col is not None}, features={missing}. "
            f"Found: {raw.columns}"
        )

    use_note = dataset_name in NOTE_LABEL_DATASETS
    phosphorylation_only = dataset_name == "uniprot_phosphorylation"
    # Identical sequences collapse to one row (first accession) with merged labels.
    parsed_by_sequence: dict[str, tuple[str, str, list[str], bool]] = {}
    for row in raw.iter_rows(named=True):
        sequence = row[sequence_col]
        if not isinstance(sequence, str) or not sequence.strip():
            continue
        feature_cells = {
            header: row[column_map[header.lower()]] for header in feature_headers
        }
        sequence = sequence.strip()
        labels, has_evidence_rejected_feature = build_residue_labels(
            feature_cells,
            len(sequence),
            use_note=use_note,
            phosphorylation_only=phosphorylation_only,
        )
        existing = parsed_by_sequence.get(sequence)
        if existing is None:
            parsed_by_sequence[sequence] = (
                str(row[accession_col]),
                sequence,
                labels,
                has_evidence_rejected_feature,
            )
        else:
            parsed_by_sequence[sequence] = (
                existing[0],
                sequence,
                merge_residue_labels(existing[2], labels),
                existing[3] or has_evidence_rejected_feature,
            )
    parsed_rows = list(parsed_by_sequence.values())

    # Vocabulary comes from deduplicated rows so dropped duplicates cannot add labels.
    label_set = {BACKGROUND_LABEL}
    for _, _, labels, _ in parsed_rows:
        label_set.update(label for label in labels if label != AMBIGUOUS_LABEL)

    vocabulary = sorted(label_set)
    vocabulary_index = {label: index for index, label in enumerate(vocabulary)}
    records: list[dict[str, object]] = []
    for accession, sequence, labels, has_evidence_rejected_feature in parsed_rows:
        has_annotation = _has_annotated_label(labels)
        records.append(
            {
                "id": accession,
                "sequence": sequence,
                "targets": [
                    IGNORE_INDEX
                    if label == AMBIGUOUS_LABEL
                    else vocabulary_index[label]
                    for label in labels
                ],
                "_has_annotation": has_annotation,
                "_has_evidence_rejected_feature": (
                    has_evidence_rejected_feature
                    or (not has_annotation and AMBIGUOUS_LABEL in labels)
                ),
            }
        )

    if not records:
        raise ValueError(f"No UniProt sequence rows found for {dataset_name}.")
    return pl.DataFrame(records), vocabulary


def write_label_vocabulary(
    output_dir: Path, dataset_name: str, vocabulary: list[str]
) -> Path:
    path = output_dir / "label_vocabulary.json"
    path.write_text(
        json.dumps(
            {
                "dataset_name": dataset_name,
                "vocabulary_version": 1,
                "labels": vocabulary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def write_annotation_card(
    output_dir: Path,
    config: SourceUniprotAnnotationsConfig,
    dataset_name: str,
    source_group: str,
    split_frames: dict[str, pl.DataFrame],
    vocabulary: list[str],
) -> Path:
    """Record evidence rules, label semantics, filters, and split provenance."""
    rows = "\n".join(
        f"- `{split}`: {frame.height} rows"
        for split, frame in sorted(split_frames.items())
    )
    fields = ANNOTATION_GROUPS[source_group]
    api_fields = ", ".join(FEATURE_FIELDS[field] for field in fields)
    label_source = (
        "UniProt `/note` values"
        if source_group in NOTE_LABEL_DATASETS
        else "UniProt feature codes"
    )
    task_name = {
        "uniprot_secondary_structure": "secondary-structure",
        "uniprot_post_translational_modification": "post-translational modification",
        "uniprot_glycosylation": "glycosylation",
        "uniprot_phosphorylation": "phosphorylation",
        "uniprot_lipidation": "lipidation",
        "uniprot_membrane_pass": "membrane passage",
        "uniprot_topology": "membrane topology",
        "uniprot_peptide": "peptide processing",
        "uniprot_functional_sites": "functional-site",
        "uniprot_domains": "domain annotation",
        "uniprot_regions": "region annotation",
        "uniprot_structures": "structural-motif",
    }[source_group]
    phosphorylation_note = (
        " Only labels containing `phospho` are retained."
        if source_group == "uniprot_phosphorylation"
        else ""
    )
    length_line = (
        f"Maximum sequence length `{config.max_sequence_length}` is applied before "
        "vocabulary construction and MMseqs2 clustering."
        if config.max_sequence_length is not None
        else "No maximum sequence-length filter was applied."
    )
    subset_line = (
        " Pooled random, stratified, and hold-cluster-out subsets are also included; "
        "hold-cluster-out subsets require enough clusters."
        if config.create_split_subsets
        else ""
    )
    organism_filter = config.organism_id if config.organism_id is not None else "all"
    card = f"""# {dataset_name}

Residue-level annotation dataset sourced from the UniProtKB REST API.

## Intended use

Residue-level {task_name} prediction from protein sequence. Scores indicate
recovery of curated UniProt features, not independent experimental evidence.

## Source and labels

- Source API: `{UNIPROT_STREAM_URL}`
- Card generated (UTC): `{generated_utc_date()}`
- Organism Taxonomy ID: `{organism_filter}` (`all` = all organisms).
- Review status: `{config.review_status}` (`reviewed` = Swiss-Prot, `unreviewed` = TrEMBL, `all` = both).
- Query: `{build_uniprot_query(config.organism_id, config.review_status)}`
- Requested UniProt fields: `{", ".join(_requested_uniprot_fields())}`
- UniProt feature headers: {", ".join(f"`{field}`" for field in fields)}
- UniProt feature fields: `{api_fields}`
- Per-residue class source: {label_source}.{phosphorylation_note}
- Feature blocks are labeled only when their own `/evidence` qualifier contains
    `ECO:0000269` or `ECO:0000305`; secondary-structure features also accept
    `ECO:0007829|PDB:<structure ID>` (PDB-derived structural assignments).
    Missing, malformed, and other evidence is excluded; `/note` text is never
    used as evidence.
- Background residues use `-`. Conflicting annotations and positions covered
    by features with rejected evidence or uncertain coordinates use the ignored
    target `-100`. Other integer targets index the sorted `label_vocabulary.json`.
- Disulfide bonds label only their two bonded cysteines, not the span between them.
- Note-derived labels drop parenthetical text (e.g. `N-linked (GlcNAc...) asparagine`
    becomes `N-linked asparagine`); notes starting with a qualifier such as
    `(Microbial infection)` fall back to the feature code.
- The ordered label vocabulary contains {len(vocabulary)} labels.
- Rows without a non-empty sequence are dropped; all target vectors have the
  same length as their sequence.

## Preprocessing and splits

Identical sequences are collapsed to one entry before vocabulary construction,
negative sampling, and splitting; their residue labels are merged, with
disagreeing residues ignored (`-100`). Features with uncertain (`?`) positions
are treated like evidence-rejected features. Proteins with at least one qualified annotated residue are retained. Genuinely
feature-free proteins may be sampled as unannotated negatives to balance the
positive pool; proteins with candidate features rejected only for missing,
malformed, or non-allowed evidence or uncertain positions are excluded from that negative pool.
Background residues on retained proteins use `-` as an unannotated background
class, not experimentally verified absence. `id` is the UniProtKB accession of
the entry kept for each unique sequence. Splits are assigned by whole MMseqs2
clusters targeting `{config.split_fractions}`. {length_line}{subset_line}

Clustering settings:

{mmseqs_settings_lines(config)}

## Split sizes

{rows}

## Dataset statistics

[`stats.json`](stats.json) at the dataset root contains row counts by split,
columns, and SeqKit sequence-length metrics.
"""
    path = output_dir / "README.md"
    path.write_text(card, encoding="utf-8")
    return path


def balance_negative_examples(frame: pl.DataFrame, seed: int) -> pl.DataFrame:
    """Keep at most as many unannotated rows as there are positive rows."""
    positives = frame.filter(pl.col("_has_annotation"))
    negatives = frame.filter(~pl.col("_has_annotation"))
    negative_count = min(positives.height, negatives.height)
    if negative_count == 0:
        return positives
    return pl.concat(
        [positives, negatives.sample(n=negative_count, seed=seed)], how="vertical"
    )


def run(config: SourceUniprotAnnotationsConfig) -> list[Path]:
    """Build selected residue-annotation datasets and optionally publish them."""
    selected_datasets = resolve_datasets(
        config.datasets, config.all_datasets, list(ANNOTATION_GROUPS)
    )

    token = resolve_hf_token(config.token)
    downloads_dir = config.work_dir / "downloads"
    raw_path = download_uniprot_tsv(
        config,
        downloads_dir,
        "uniprot_annotations.tsv.gz",
        _requested_uniprot_fields(),
    )
    raw = read_uniprot_tsv(raw_path, config.max_sequence_length)
    output_paths: list[Path] = []

    for source_group in selected_datasets:
        dataset_name = source_group
        if config.max_sequence_length is not None:
            dataset_name = f"{dataset_name}_{config.max_sequence_length}_cutoff"

        frame, vocabulary = _frame_from_uniprot_table(raw, source_group)
        # Rejected evidence is unknown, not a valid negative annotation.
        frame = frame.filter(
            pl.col("_has_annotation") | ~pl.col("_has_evidence_rejected_feature")
        )
        if frame.filter(pl.col("_has_annotation")).is_empty():
            raise ValueError(
                f"No proteins with positive UniProt annotations were found for "
                f"{source_group}."
            )
        frame = balance_negative_examples(frame, seed=config.seed).drop(
            ["_has_annotation", "_has_evidence_rejected_feature"]
        )
        split_frames, _ = split_by_clusters(frame, config, dataset_name)

        output_dir = prepare_output_dir(config, dataset_name)
        write_hf_split_files(output_dir, split_frames)
        write_dataset_stats(output_dir, split_frames, threads=config.stats_threads)
        write_label_vocabulary(output_dir, dataset_name, vocabulary)
        write_annotation_card(
            output_dir,
            config,
            dataset_name,
            source_group,
            split_frames,
            vocabulary,
        )
        upload_dataset_artifact(output_dir, dataset_name, config, token)
        output_paths.append(output_dir)

    return output_paths


def main() -> None:
    run_script(SourceUniprotAnnotationsConfig, run)


if __name__ == "__main__":
    main()
