"""Fixed-split contact data and original-coordinate alignment."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from itertools import pairwise
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
from datasets import DatasetDict
from transformers import PreTrainedTokenizerBase

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


def load_contact_dataset(
    path_or_repo: str | Path,
    *,
    config_name: str | None = None,
    revision: str | None = None,
    edition: int | None = None,
    cache_dir: str | Path | None = None,
) -> ContactDataset:
    """Load fixed HF scoring splits without synthesis or coordinate dependencies."""
    from modules.evaluate.src.dataset.dataloader import load_raw_evaluation_dataset

    path = Path(path_or_repo)
    if not path.is_dir():
        if isinstance(path_or_repo, Path) or path.is_absolute():
            raise FileNotFoundError(path)
        if revision is None or re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None:
            raise ValueError(
                "Hub contact data requires a pinned 40-character commit SHA"
            )
    raw = load_raw_evaluation_dataset(
        path_or_repo,
        config_name=config_name,
        revision=revision,
        cache_dir=cache_dir,
    )
    return prepare_contact_dataset(
        raw, source=str(path_or_repo), revision=revision, edition=edition
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

    @property
    def development_fingerprint(self) -> str:
        """Bind fitting to development sequences, units and labels, never tests."""
        from dataclasses import asdict

        payload = {
            "schema_version": self.schema_version,
            "splits": {
                split: [asdict(chain) for chain in self.splits[split]]
                for split in ("train", "validation")
            },
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


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


def separation_prior_scores(chain: ContactChain) -> np.ndarray:
    """Rank pairs by inverse separation in original, possibly discontinuous coordinates."""
    positions = np.asarray(chain.alignment.retained_positions, dtype=np.float64)
    return 1.0 / (1.0 + np.abs(positions[:, None] - positions[None, :]))


def separation_prior_prediction_rows(
    dataset: ContactDataset, split: str = "test"
) -> list[dict[str, Any]]:
    """Produce canonical contact prediction rows without loading a model."""
    from modules.evaluate.src.metrics.contact_metrics import contact_prediction_row

    rows = []
    for chain in dataset.splits[split]:
        scores = separation_prior_scores(chain)
        for unit in chain.units:
            rows.append(
                {
                    "id": unit.id,
                    "record_id": chain.record_id,
                    "unit_id": unit.unit_id,
                    "split": chain.split,
                    "source": chain.source,
                    "edition": chain.edition,
                    "complex_id": chain.complex_id,
                    "target": chain.target,
                    "kind": unit.kind,
                    "labels_available": unit.labels_available,
                    **contact_prediction_row(scores, unit, chain.alignment),
                }
            )
    return rows


# Loss/metric ignore index for padding and (unscored) short-range pairs.
IGNORE_INDEX = -100
