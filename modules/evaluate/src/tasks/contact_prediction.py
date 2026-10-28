"""HF-only CASP contact units, original-coordinate alignment and evaluation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import polars as pl
import torch
from datasets import DatasetDict
from transformers import PreTrainedTokenizerBase

from modules.evaluate.src.metrics import contact_metrics
from modules.evaluate.src.model.contact_model import (
    build_contact_model,
    load_contact_model,
)
from modules.evaluate.src.tasks.base import MetricFn, TaskHandler
from modules.evaluate.src.tasks.registry import register_task

if TYPE_CHECKING:
    from modules.evaluate.src.schemas.artifacts import (
        ContactScoreOutput,
        EvaluationMetadata,
        PredictOutput,
    )

SCHEMA_VERSION = "6.0"
PROTOCOL_VERSION = "casp-contact-v1"
SPLITS = ("train", "validation", "test")
CANONICAL = frozenset("ACDEFGHIKLMNPQRSTVWY")
AMBIGUOUS = frozenset("XBOUZJ")
REQUIRED_COLUMNS = frozenset(
    {
        "id",
        "record_id",
        "sequence",
        "split",
        "source",
        "edition",
        "complex_id",
        "target",
        "unit_id",
        "kind",
        "positions",
        "labels_available",
        "positive_pairs",
        "valid_positions",
    }
)


@dataclass(frozen=True)
class ContactAlignment:
    sequence: str
    clean_sequence: str
    retained_positions: tuple[int, ...]
    original_to_retained: tuple[int, ...]


@dataclass(frozen=True)
class ContactUnit:
    id: str
    unit_id: str
    kind: str
    positions: tuple[int, ...] | None
    labels_available: bool
    positive_pairs: tuple[tuple[int, int], ...]
    valid_positions: tuple[int, ...]


@dataclass(frozen=True)
class ContactChain:
    record_id: str
    split: str
    source: str
    edition: int | None
    complex_id: str | None
    target: str | None
    alignment: ContactAlignment
    units: tuple[ContactUnit, ...]


@dataclass(frozen=True)
class ContactDataset:
    splits: dict[str, tuple[ContactChain, ...]]
    fingerprint: str
    source: str
    revision: str | None
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True)
class ContactTokens:
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    position_ids: tuple[int, ...]
    residue_token_indices: tuple[int, ...]


def align_contact_sequence(sequence: str) -> ContactAlignment:
    """Derive both runtime mappings; deleted original positions map to -1."""
    if not isinstance(sequence, str) or not sequence:
        raise ValueError("Expected a nonempty uppercase protein sequence")
    if set(sequence) - CANONICAL - AMBIGUOUS:
        raise ValueError("Expected an uppercase protein sequence")
    retained = tuple(i for i, aa in enumerate(sequence) if aa in CANONICAL)
    inverse = [-1] * len(sequence)
    for clean, original in enumerate(retained):
        inverse[original] = clean
    return ContactAlignment(
        sequence, "".join(sequence[i] for i in retained), retained, tuple(inverse)
    )


def tokenize_contact_chain(
    chain: ContactChain, tokenizer: PreTrainedTokenizerBase
) -> ContactTokens:
    """Tokenize the whole cleaned chain, preserving original positional gaps."""
    alignment = chain.alignment
    if not alignment.retained_positions:
        raise ValueError(f"{chain.record_id}: no retained residues to tokenize")
    encoded = tokenizer(
        alignment.clean_sequence,
        add_special_tokens=True,
        truncation=False,
        padding=False,
        return_special_tokens_mask=True,
    )
    ids = tuple(encoded["input_ids"])
    special = encoded["special_tokens_mask"]
    residues = tuple(i for i, masked in enumerate(special) if not masked)
    expected_ids = tokenizer.convert_tokens_to_ids(list(alignment.clean_sequence))
    if (
        len(residues) != len(alignment.retained_positions)
        or not residues
        or residues != tuple(range(residues[0], residues[-1] + 1))
        or [ids[i] for i in residues] != expected_ids
        or tokenizer.unk_token_id in expected_ids
    ):
        raise ValueError(
            f"{chain.record_id}: contact inputs require one unchanged token per residue"
        )
    positions = list(range(len(ids)))
    for token, original in zip(residues, alignment.retained_positions):
        positions[token] = residues[0] + original
    # EOS and any other suffix tokens follow the full original sequence.
    for token in range(residues[-1] + 1, len(ids)):
        positions[token] += len(alignment.sequence) - len(alignment.clean_sequence)
    return ContactTokens(ids, (1,) * len(ids), tuple(positions), residues)


def _positions(value: Any, length: int, field: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{field} must be an integer list")
    if any(
        isinstance(i, bool) or not isinstance(i, Integral) or not 0 <= int(i) < length
        for i in value
    ):
        raise ValueError(f"{field} contains a noninteger or out-of-bounds index")
    result = tuple(int(i) for i in value)
    if any(a >= b for a, b in pairwise(result)):
        raise ValueError(f"{field} must be strictly increasing")
    return result


def _unit(row: dict[str, Any]) -> ContactUnit:
    length = len(row["sequence"])
    kind = row["kind"]
    if kind not in ("global", "intra_eu"):
        raise ValueError("kind must be global or intra_eu")
    if (kind == "global") != (row["positions"] is None):
        raise ValueError("Only global rows must have null positions")
    if kind == "global" and row["unit_id"] != "chain":
        raise ValueError("Global unit_id must be chain")
    if row["id"] != f"{row['record_id']}/{kind}/{row['unit_id']}":
        raise ValueError("Scoring id must equal record_id/kind/unit_id")
    members = (
        None
        if row["positions"] is None
        else _positions(row["positions"], length, "positions")
    )
    valid = _positions(row["valid_positions"], length, "valid_positions")
    if members is not None and not set(valid).issubset(members):
        raise ValueError("valid_positions must be restricted to EU members")
    if not isinstance(row["labels_available"], bool):
        raise TypeError("labels_available must be boolean")
    raw_pairs = row["positive_pairs"]
    if not isinstance(raw_pairs, (list, tuple)):
        raise TypeError("positive_pairs must be a list of [i, j] pairs")
    pairs = []
    known = set(valid)
    for pair in raw_pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(
                "Each positive_pairs entry must contain exactly two indices"
            )
        i, j = _positions(pair, length, "positive_pairs entry")
        if i not in known or j not in known:
            raise ValueError("Positive contacts require two valid positions")
        pairs.append((i, j))
    if len(set(pairs)) != len(pairs):
        raise ValueError("Duplicate positive pair within a scoring row")
    if not row["labels_available"] and (pairs or valid):
        raise ValueError("Unavailable labels require empty pairs and valid_positions")
    return ContactUnit(
        row["id"],
        row["unit_id"],
        kind,
        members,
        row["labels_available"],
        tuple(pairs),
        valid,
    )


def prepare_contact_dataset(
    dataset: DatasetDict,
    *,
    source: str = "in-memory",
    revision: str | None = None,
    edition: int | None = None,
) -> ContactDataset:
    """Validate all splits before optionally selecting a test edition."""
    if not isinstance(dataset, DatasetDict) or set(dataset) != set(SPLITS):
        raise ValueError(
            "Contact data requires fixed train, validation and test splits"
        )
    if edition is not None and (
        isinstance(edition, bool)
        or not isinstance(edition, Integral)
        or edition not in (14, 15, 16)
    ):
        raise ValueError("edition must be 14, 15 or 16")
    splits: dict[str, tuple[ContactChain, ...]] = {}
    seen_rows: set[str] = set()
    seen_chains: set[str] = set()
    digest = hashlib.sha256(SCHEMA_VERSION.encode())
    metadata = ("sequence", "source", "edition", "complex_id", "target")
    for split in SPLITS:
        missing = REQUIRED_COLUMNS - set(dataset[split].column_names)
        if missing:
            raise ValueError(f"{split}: missing contact columns {sorted(missing)}")
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in dataset[split]:
            for name in ("id", "record_id", "unit_id", "source"):
                if not isinstance(row[name], str) or not row[name]:
                    raise ValueError(f"{name} must be a nonempty string")
            if row["split"] != split:
                raise ValueError(
                    f"{row['id']}: split column differs from dataset split"
                )
            if row["id"] in seen_rows:
                raise ValueError(f"Duplicate scoring id: {row['id']}")
            seen_rows.add(row["id"])
            if split == "test":
                if (
                    row["source"] != "casp"
                    or isinstance(row["edition"], bool)
                    or row["edition"] not in (14, 15, 16)
                    or not isinstance(row["edition"], Integral)
                    or any(
                        not isinstance(row[name], str) or not row[name]
                        for name in ("complex_id", "target")
                    )
                ):
                    raise ValueError(f"{row['id']}: invalid CASP metadata")
            elif row["source"] != "trrosetta" or any(
                row[name] is not None for name in ("edition", "complex_id", "target")
            ):
                raise ValueError(f"{row['id']}: invalid development metadata")
            grouped.setdefault(row["record_id"], []).append(row)
        chains = []
        for record_id, rows in grouped.items():
            if record_id in seen_chains:
                raise ValueError(f"Chain appears in multiple splits: {record_id}")
            seen_chains.add(record_id)
            first = rows[0]
            alignment = align_contact_sequence(first["sequence"])
            if any(row[key] != first[key] for row in rows for key in metadata):
                raise ValueError(
                    f"{record_id}: inconsistent sequence or chain metadata"
                )
            units = tuple(_unit(row) for row in rows)
            if sum(unit.kind == "global" for unit in units) != 1:
                raise ValueError(f"{record_id}: expected exactly one global row")
            if split != "test" and len(units) != 1:
                raise ValueError("Development structures must have one global row only")
            if split == "test" and edition is not None and first["edition"] != edition:
                continue
            for row in rows:
                content = {key: row[key] for key in sorted(REQUIRED_COLUMNS)}
                digest.update(json.dumps(content, separators=(",", ":")).encode())
                digest.update(b"\n")
            chains.append(
                ContactChain(
                    record_id,
                    split,
                    first["source"],
                    first["edition"],
                    first["complex_id"],
                    first["target"],
                    alignment,
                    units,
                )
            )
        splits[split] = tuple(chains)
    return ContactDataset(splits, digest.hexdigest(), source, revision)


@dataclass(frozen=True)
class CandidatePairs:
    i: np.ndarray
    j: np.ndarray
    original_i: np.ndarray
    original_j: np.ndarray
    labels: np.ndarray
    length: int


def candidate_pairs(
    unit: ContactUnit, alignment: ContactAlignment, minsep: int = 6
) -> CandidatePairs:
    """Shared training/scoring candidates, without combining different EU maps."""
    if isinstance(minsep, bool) or not isinstance(minsep, int) or minsep < 1:
        raise ValueError("minsep must be a positive integer")
    retained = np.asarray(alignment.retained_positions, dtype=np.int64)
    length = (
        len(retained)
        if unit.positions is None
        else int(np.isin(retained, unit.positions).sum())
    )
    inverse = np.asarray(alignment.original_to_retained, dtype=np.int64)
    valid = np.asarray(unit.valid_positions, dtype=np.int64)
    valid = valid[inverse[valid] >= 0]
    a, b = np.triu_indices(len(valid), 1)
    original_i, original_j = valid[a], valid[b]
    separated = original_j - original_i >= minsep
    original_i, original_j = original_i[separated], original_j[separated]
    codes = original_i * len(alignment.sequence) + original_j
    positives = np.asarray(unit.positive_pairs, dtype=np.int64).reshape(-1, 2)
    positive_codes = positives[:, 0] * len(alignment.sequence) + positives[:, 1]
    return CandidatePairs(
        inverse[original_i],
        inverse[original_j],
        original_i,
        original_j,
        np.isin(codes, positive_codes).astype(np.int8),
        length,
    )


# Loss/metric ignore index for padding and (unscored) short-range pairs.
IGNORE_INDEX = -100


@register_task
class ContactPredictionTask(TaskHandler):
    """Fixed-split CASP contact prediction, alignment and scoring."""

    name = "contact_prediction"
    is_ragged = True
    pairwise = True
    aligns_labels = True
    label_dtype = torch.int8
    accepted_target_types = frozenset({"categorical"})
    accepted_label_formats = frozenset({"contact_pairs"})

    #: Tokenized-residue separation below which a pair is never scored (it's
    #: a sequence-adjacency contact, not a structural one).
    MIN_SEP: ClassVar[int] = 6

    metrics: ClassVar[dict[str, MetricFn]] = {
        **contact_metrics.PRECISION_METRICS,
        **contact_metrics.RANGE_AUC_METRICS,
    }
    default_metric_names = (
        *contact_metrics.PRECISION_METRICS,
        *contact_metrics.RANGE_AUC_METRICS,
    )

    def default_metrics(self) -> list[str]:
        if self.name == "contact_prediction":
            return list(contact_metrics.CASP_METRICS)
        return super().default_metrics()

    def validate_metric_names(self, metric_names: Sequence[str]) -> None:
        if self.name != "contact_prediction":
            return super().validate_metric_names(metric_names)
        unknown = set(metric_names) - set(contact_metrics.CASP_METRICS)
        if unknown:
            raise KeyError(f"Unknown CASP contact metrics: {sorted(unknown)}")

    def compute_metrics(
        self,
        predictions: Sequence,
        labels: Sequence,
        metric_names: Sequence[str],
        average: str = "weighted",
        probabilities: Sequence | None = None,
    ) -> dict[str, float]:
        if self.name == "contact_prediction":
            raise ValueError(
                "Use score_units: CASP aggregation requires unit, chain and complex identities"
            )
        return super().compute_metrics(
            predictions, labels, metric_names, average, probabilities
        )

    def prepare_dataset(
        self,
        path_or_repo: str | Path,
        *,
        revision: str | None = None,
        edition: int | None = None,
        cache_dir: str | Path | None = None,
    ) -> ContactDataset:
        """Load fixed HF scoring rows, without generic cropping or split synthesis."""
        from modules.evaluate.src.dataset.dataloader import load_contact_dataset

        return load_contact_dataset(
            path_or_repo, revision=revision, edition=edition, cache_dir=cache_dir
        )

    def predict_units(
        self,
        dataset: ContactDataset,
        predictor: Callable[[ContactChain], np.ndarray | torch.Tensor],
        metadata: EvaluationMetadata,
        *,
        split: str = "test",
    ) -> PredictOutput:
        """Coordinate full-chain extraction and preserve every unit's identity."""
        from modules.evaluate.src.schemas.artifacts import (
            ContactEvaluationMetadata,
            PredictOutput,
        )
        from modules.evaluate.src.utils.inference import run_contact_inference

        if metadata.seed is None:
            raise ValueError("Contact predictions require an explicit run seed")
        contact_metadata = ContactEvaluationMetadata.model_validate(
            {
                **metadata.model_dump(),
                "dataset_revision": dataset.revision,
                "dataset_fingerprint": dataset.fingerprint,
                "schema_version": SCHEMA_VERSION,
                "protocol_version": PROTOCOL_VERSION,
            }
        )
        rows_by_chain = {}
        for chain, scores in run_contact_inference(dataset, predictor, split=split):
            rows_by_chain[chain.record_id] = [
                dict(
                    id=unit.id,
                    record_id=chain.record_id,
                    unit_id=unit.unit_id,
                    split=chain.split,
                    source=chain.source,
                    edition=chain.edition,
                    complex_id=chain.complex_id,
                    target=chain.target,
                    kind=unit.kind,
                    labels_available=unit.labels_available,
                    **contact_metrics.contact_prediction_row(
                        scores, unit, chain.alignment
                    ),
                )
                for unit in chain.units
            ]
        rows = [
            row
            for chain in dataset.splits[split]
            for row in rows_by_chain[chain.record_id]
        ]
        if not rows:
            raise ValueError(f"No contact scoring rows in split {split}")
        frame = pl.DataFrame(rows, infer_schema_length=None).with_columns(
            pl.col("original_i", "original_j").cast(pl.List(pl.Int32)),
            pl.col("labels").cast(pl.List(pl.Int8)),
            pl.col("scores").cast(pl.List(pl.Float64)),
        )
        return PredictOutput(metadata=contact_metadata, predictions=frame)

    def score_units(self, predictions: PredictOutput) -> ContactScoreOutput:
        """Score all CASP settings, then aggregate EUs, chains and complexes."""
        from modules.evaluate.src.schemas.artifacts import (
            ContactEvaluationMetadata,
            ContactScoreOutput,
        )

        metadata = predictions.metadata
        if (
            not isinstance(metadata, ContactEvaluationMetadata)
            or metadata.schema_version != SCHEMA_VERSION
            or metadata.protocol_version != PROTOCOL_VERSION
            or not metadata.dataset_fingerprint
            or metadata.seed is None
        ):
            raise ValueError(
                "Contact scoring requires schema-6.0 CASP prediction metadata"
            )
        if set(predictions.requested_controls) != set(predictions.controls):
            raise ValueError("Every requested CASP control requires prediction rows")
        if set(predictions.controls) - {"random_trunk"}:
            raise ValueError("CASP contact prediction only supports random_trunk")
        if predictions.predictions.is_empty():
            raise ValueError("Contact scoring requires prediction rows")
        payload_columns = {
            "original_i",
            "original_j",
            "labels",
            "scores",
            "length",
            "status",
        }
        identity = {
            key: value
            for key, value in metadata.model_dump().items()
            if key not in {"split", "dataset_name", "dataset_repo_id"}
        }

        def score_rows(
            frame: pl.DataFrame, control: str | None = None
        ) -> tuple[pl.DataFrame, pl.DataFrame]:
            rows = []
            seen = set()
            for row in frame.to_dicts():
                if row["id"] in seen:
                    raise ValueError(f"Duplicate scoring id: {row['id']}")
                seen.add(row["id"])
                for minsep in contact_metrics.CASP_SEPARATIONS:
                    scored = dict(
                        {
                            key: value
                            for key, value in row.items()
                            if key not in payload_columns
                        },
                        **identity,
                        minsep=minsep,
                        **contact_metrics.score_contact_prediction(row, minsep),
                    )
                    if control is not None:
                        scored["control"] = control
                    rows.append(scored)
            summaries = contact_metrics.aggregate_contact_scores(rows)
            if control is not None:
                for summary in summaries:
                    summary["control"] = control
            return (
                pl.DataFrame(rows, infer_schema_length=None),
                pl.DataFrame(summaries, infer_schema_length=None),
            )

        unit_scores, summaries = score_rows(predictions.predictions)
        control_scores = {}
        control_summaries = {}
        primary_by_id = {row["id"]: row for row in predictions.predictions.to_dicts()}
        for name, frame in predictions.controls.items():
            if set(frame["id"].to_list()) != set(primary_by_id):
                raise ValueError(
                    f"{name} control rows do not match primary prediction IDs"
                )
            for row in frame.to_dicts():
                primary = primary_by_id[row["id"]]
                for key in ("original_i", "original_j", "labels", "length", "status"):
                    if row[key] != primary[key]:
                        raise ValueError(
                            f"{name} control candidates differ from primary predictions"
                        )
            control_scores[name], control_summaries[name] = score_rows(frame, name)
        return ContactScoreOutput(
            metadata,
            unit_scores,
            summaries,
            control_unit_scores=control_scores,
            control_summaries=control_summaries,
        )

    def model_kwargs(self) -> dict[str, Any]:
        return {}

    def build_model(
        self,
        model_id: str,
        num_labels: int,
        model_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        if model_kwargs:
            raise ValueError(
                "Contact predictor settings are fixed by the CASP protocol"
            )
        return build_contact_model(model_id)

    def load_finetuned_model(self, model_path: str) -> Any:
        return load_contact_model(model_path)

    def resolve_num_labels(self, num_labels: int | None, dataset_name: str) -> int:
        # Not a class count (binary per-pair score); nothing derives from it.
        return 2

    # -- data ----------------------------------------------------------------

    def align_labels(
        self,
        label: Any,
        *,
        num_residues: int,
        start: int,
        kept: int,
        offset: int,
        input_len: int,
        example_index: int,
    ) -> Any:
        """Crop+shift raw ``[i, j]`` contact pairs into tokenized coordinates.

        The aligned dict retains the cropped sequence length and resolved
        residue positions, excluding special tokens from candidate pairs.
        """
        end = start + kept
        if isinstance(label, dict):
            pairs = label["pairs"]
            source_valid_positions = label["valid_positions"] or []
        else:
            pairs = label
            source_valid_positions = range(num_residues)
        pairs = [
            [offset + (i - start), offset + (j - start)]
            for i, j in pairs
            if start <= i < end and start <= j < end
        ]
        valid_positions = [
            offset + (position - start)
            for position in source_valid_positions
            if start <= position < end
        ]
        return {
            "pairs": pairs,
            "valid_positions": valid_positions,
            "res_start": offset,
            "res_end": offset + kept,
            "length": kept,
        }

    def collate_labels(
        self, labels: list[Any], batch_size: int, seq_len: int
    ) -> torch.Tensor:
        idx = torch.arange(seq_len)
        base_valid = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() >= self.MIN_SEP
        collated = torch.full(
            (batch_size, seq_len, seq_len), IGNORE_INDEX, dtype=self.label_dtype
        )
        for b, entry in enumerate(labels):
            res_start, res_end = entry["res_start"], entry["res_end"]
            resolved = torch.zeros(seq_len, dtype=torch.bool)
            resolved[entry.get("valid_positions", range(res_start, res_end))] = True
            valid = base_valid.clone()
            # Exclude padding *and* special-token positions (before res_start,
            # at/after res_end): only real residues can be scored, positive
            # or implicit-negative.
            valid[res_end:, :] = False
            valid[:, res_end:] = False
            valid[:res_start, :] = False
            valid[:, :res_start] = False
            valid &= resolved[:, None] & resolved[None, :]
            collated[b][valid] = 0
            for i, j in entry["pairs"]:
                if valid[i, j]:
                    collated[b, i, j] = 1
                    collated[b, j, i] = 1
        return collated

    def extract_predictions(
        self, logits: torch.Tensor, *, labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        return logits.sigmoid()

    def split_batch_rows(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        *,
        residue_lengths: torch.Tensor | None = None,
    ) -> list[tuple[Any, Any]]:
        rows: list[tuple[Any, Any]] = []
        for row_index, (pred_mat, label_mat) in enumerate(zip(predictions, labels)):
            valid = label_mat != IGNORE_INDEX
            # Every scored pair appears twice (symmetric); count each once.
            upper = valid & torch.triu(torch.ones_like(valid), diagonal=1)
            i_idx, j_idx = upper.nonzero(as_tuple=True)
            scores = [float(score) for score in pred_mat[i_idx, j_idx].tolist()]
            seps = [int(sep) for sep in (j_idx - i_idx).tolist()]
            values = [int(value) for value in label_mat[i_idx, j_idx].tolist()]
            if residue_lengths is not None:
                length = int(residue_lengths[row_index])
            else:
                scored_cols = valid.any(dim=0).nonzero(as_tuple=True)[0]
                length = (
                    int(scored_cols.max() - scored_cols.min() + 1)
                    if scored_cols.numel()
                    else 0
                )
            pred_row = {"scores": scores, "separations": seps}
            label_row = {"length": length, "values": values}
            rows.append((pred_row, label_row))
        return rows

    def flatten_for_metrics(self, values: list[Any]) -> list[Any]:
        # Precision-at-L needs each sequence's own top-scored candidates and
        # length, so this task's metrics consume the per-sequence rows
        # directly instead of a pooled flat sequence.
        return values

    def compute_per_example_metrics(
        self,
        predictions: Sequence,
        labels: Sequence,
        metric_names: Sequence[str],
        average: str = "weighted",
        probabilities: Sequence | None = None,
    ) -> np.ndarray:
        """Compute contact scores once per protein for paired resampling.

        Rows follow the input order and columns follow metric_names. Each
        existing metric is a mean over proteins with defined contributions,
        so resampling these scores preserves its aggregation and NaN handling.
        """
        self.validate_metric_names(metric_names)
        return np.asarray(
            [
                [
                    self.metrics[name]([prediction], [label], average)
                    for name in metric_names
                ]
                for prediction, label in zip(predictions, labels)
            ],
            dtype=np.float64,
        ).reshape(len(predictions), len(metric_names))

    def scramble_labels(self, labels: list[Any], rng: np.random.Generator) -> list[Any]:
        """Permute each row's per-pair labels while retaining its length."""
        scrambled: list[Any] = []
        for row in labels:
            if not isinstance(row, dict):
                length, values = row[0], row[1:]
                permuted_idx = rng.permutation(len(values))
                scrambled.append([length] + [values[j] for j in permuted_idx])
                continue
            length, values = row["length"], row["values"]
            permuted_idx = rng.permutation(len(values))
            scrambled.append(
                {"length": length, "values": [values[j] for j in permuted_idx]}
            )
        return scrambled
