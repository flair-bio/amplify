"""Source residue-level annotation datasets from UniProtKB."""

from __future__ import annotations

import json
import logging
import re
import shutil
import sys
from pathlib import Path

import polars as pl
from pydantic import ConfigDict, Field

from modules.core.utils.hf_download import resolve_hf_token
from modules.evaluate.src.dataset_sourcing.config import UniProtDatasetSourcingConfig
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

BACKGROUND_LABEL = "-"
AMBIGUOUS_LABEL = "?"

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

NOTE_LABEL_DATASETS = frozenset(
    {
        "uniprot_glycosylation",
        "uniprot_phosphorylation",
        "uniprot_lipidation",
        "uniprot_topology",
    }
)

_FEATURE_RE = re.compile(r"(?:^|;\s*)([A-Z_]+)\s+(<?\d+)(?:\.\.(>?\d+))?")
_NOTE_RE = re.compile(r'/note="([^"]*)"')
_EVIDENCE_RE = re.compile(r'/evidence\s*=\s*(?:"([^"]*)"|([^;]*))')
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
    split_fractions: dict[str, float] = Field(
        default_factory=lambda: {"train": 0.8, "validation": 0.1, "test": 0.1}
    )


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


def _build_residue_labels(
    feature_cells: dict[str, str | None],
    sequence_length: int,
    use_note: bool = False,
    phosphorylation_only: bool = False,
) -> tuple[list[str], bool]:
    """Expand UniProt feature coordinates into one deterministic label per residue."""
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
            start_match = re.search(r"\d+", match.group(2))
            if start_match is None:
                continue
            start = int(start_match.group())
            end_text = match.group(3)
            if end_text:
                end_match = re.search(r"\d+", end_text)
                if end_match is None:
                    continue
                end = int(end_match.group())
            else:
                end = start
            if start < 1 or end < start:
                continue

            block_end = (
                matches[match_index + 1].start()
                if match_index + 1 < len(matches)
                else len(cell)
            )
            # Evidence is scoped to this feature block, not neighboring annotations.
            if not _has_allowed_evidence(
                cell, match.start(), block_end, note_spans, feature_header
            ):
                has_evidence_rejected_feature = True
                continue
            feature_label = match.group(1)
            if use_note:
                note_match = _NOTE_RE.search(cell[match.end() : block_end])
                if note_match:
                    note = note_match.group(1)
                    if "(" not in note and ")" not in note:
                        feature_label = note.split(";", 1)[0]
            feature_label = feature_label.replace(" ", "_")
            if phosphorylation_only and "phospho" not in feature_label.casefold():
                continue

            # UniProt coordinates are 1-based inclusive; convert to a 0-based range.
            for residue_index in range(start - 1, min(end, sequence_length)):
                residue_labels[residue_index].add(feature_label)

    # Conflicting overlaps use ignore-index; residues without features are background.
    return (
        [
            next(iter(labels))
            if len(labels) == 1
            else (AMBIGUOUS_LABEL if labels else BACKGROUND_LABEL)
            for labels in residue_labels
        ],
        has_evidence_rejected_feature,
    )


def build_residue_labels(
    feature_cells: dict[str, str | None],
    sequence_length: int,
    use_note: bool = False,
    phosphorylation_only: bool = False,
) -> list[str]:
    """Expand qualified UniProt feature coordinates into residue labels."""
    labels, _ = _build_residue_labels(
        feature_cells,
        sequence_length,
        use_note=use_note,
        phosphorylation_only=phosphorylation_only,
    )
    return labels


def _frame_from_uniprot_table(
    raw: pl.DataFrame, dataset_name: str
) -> tuple[pl.DataFrame, list[str]]:
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
    label_set = {BACKGROUND_LABEL}
    parsed_rows: list[tuple[str, str, list[str], bool]] = []
    for row in raw.iter_rows(named=True):
        sequence = row[sequence_col]
        if not isinstance(sequence, str) or not sequence.strip():
            continue
        feature_cells = {
            header: row[column_map[header.lower()]] for header in feature_headers
        }
        sequence = sequence.strip()
        labels, has_evidence_rejected_feature = _build_residue_labels(
            feature_cells,
            len(sequence),
            use_note=use_note,
            phosphorylation_only=phosphorylation_only,
        )
        label_set.update(label for label in labels if label != AMBIGUOUS_LABEL)
        parsed_rows.append(
            (
                str(row[accession_col]),
                sequence,
                labels,
                has_evidence_rejected_feature,
            )
        )

    vocabulary = sorted(label_set)
    vocabulary_index = {label: index for index, label in enumerate(vocabulary)}
    records: list[dict[str, object]] = []
    for accession, sequence, labels, has_evidence_rejected_feature in parsed_rows:
        has_annotation = any(
            label not in (BACKGROUND_LABEL, AMBIGUOUS_LABEL) for label in labels
        )
        records.append(
            {
                "id": accession,
                "sequence": sequence,
                "targets": [
                    -100 if label == AMBIGUOUS_LABEL else vocabulary_index[label]
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


def frame_from_uniprot_tsv(
    tsv_path: Path, dataset_name: str
) -> tuple[pl.DataFrame, list[str]]:
    """Parse one annotation group from a downloaded UniProt TSV."""
    return _frame_from_uniprot_table(
        pl.read_csv(tsv_path, separator="\t"), dataset_name
    )


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
            Missing, malformed, and other evidence is excluded; `/note` text is never used as evidence.
- Background residues use `-`. Overlapping, ambiguous residues use the ignored
    target `-100` rather than a compound class. Other integer targets index the
    sorted `label_vocabulary.json`.
- The ordered label vocabulary contains {len(vocabulary)} labels.
- Rows without a non-empty sequence are dropped; all target vectors have the
  same length as their sequence.

## Preprocessing and splits

Proteins with at least one qualified annotated residue are retained. Genuinely
feature-free proteins may be sampled as unannotated negatives to balance the
positive pool; proteins with candidate features rejected only for missing,
malformed, or non-allowed evidence are excluded from that negative pool.
Background residues on retained proteins use `-` as an unannotated background
class, not experimentally verified absence. Splits are assigned by whole
MMseqs2 `easy-linclust` clusters with minimum identity
`{config.validation_cluster_identity_threshold}`, minimum coverage
`{config.validation_cluster_coverage_threshold}`, and
`{config.validation_cluster_num_threads}` thread(s), targeting
`{config.split_fractions}`. {length_line}{subset_line}

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
    raw = pl.read_csv(raw_path, separator="\t")
    if config.max_sequence_length is not None:
        sequence_col = next(col for col in raw.columns if col.lower() == "sequence")
        raw = raw.filter(
            pl.col(sequence_col).str.len_chars() <= config.max_sequence_length
        )
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
        split_frames = create_full_dataset_splits(
            frame,
            split_fractions=config.split_fractions,
            seed=config.seed,
            identity_threshold=config.validation_cluster_identity_threshold,
            coverage_threshold=config.validation_cluster_coverage_threshold,
            num_threads=config.validation_cluster_num_threads,
        )
        split_frames = minimal_preprocess(
            dataset_name, split_frames, config.max_sequence_length, column_rename={}
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
        write_label_vocabulary(output_dir, dataset_name, vocabulary)
        write_annotation_card(
            output_dir,
            config,
            dataset_name,
            source_group,
            split_frames,
            vocabulary,
        )
        if config.has_upload_target():
            upload_dataset_artifact(output_dir, dataset_name, config, token)
        output_paths.append(output_dir)

    return output_paths


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        stream=sys.stdout,
        force=True,
    )
    config = load_runtime_config_from_argv(
        sys.argv[1:], model_cls=SourceUniprotAnnotationsConfig
    )
    run(config)


if __name__ == "__main__":
    main()
