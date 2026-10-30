"""Baseline-only structural priors: contact separation and amino-acid frequency.

Both baselines are fitted from training data only and are published as
run-once artifacts keyed by protocol, fit identity, and test identity. They
are never row-aligned to model runs: the contact prior is scored through the
same CASP predict/score path as models, and the pseudo-perplexity baseline is
a tokenizer-independent residue-level unigram reference.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import polars as pl

from modules.evaluate.src.predictors import priors
from modules.evaluate.src.predictors.composition import CANONICAL_AMINO_ACIDS
from modules.evaluate.src.schemas.artifacts import EvaluationMetadata
from modules.evaluate.src.tasks.contact_prediction import (
    ContactAlignment,
    ContactDataset,
    ContactPredictionTask,
    candidate_pairs,
)
from modules.evaluate.src.utils.baselines import (
    BASELINE_PROTOCOLS,
    AminoAcidReference,
    identity_hash,
)

CONTACT_BASELINE = "separation_prior"
AMINO_ACID_BASELINE = "amino_acid_frequency"
CONTACT_MODEL_NAME = "baseline:separation_prior"


def _normalise(payload: Any) -> Any:
    """JSON round trip so tuples/numpy scalars compare equal to stored manifests."""
    return json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _directory(root: Path, identity: Mapping[str, Any]) -> Path:
    name = identity["baseline"]
    return (
        Path(root)
        / name
        / f"protocol-{identity_hash(identity['protocol'])[:12]}"
        / f"fit-{identity_hash(identity['fit'])[:12]}"
        / f"test-{identity_hash(identity['test'])[:12]}"
    )


def _verify(directory: Path, identity: Mapping[str, Any]) -> Path:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("identity") != _normalise(identity):
        raise ValueError(
            f"Incompatible baseline manifest at {manifest_path}; "
            "delete the directory to regenerate it."
        )
    for filename, digest in manifest.get("files", {}).items():
        if _sha256(directory / filename) != digest:
            raise ValueError(f"Corrupt baseline artifact file: {directory / filename}")
    return directory


def _publish(
    root: Path,
    identity: Mapping[str, Any],
    write: Callable[[Path], dict[str, Any]],
) -> Path:
    """Publish a baseline once; never overwrite or adopt an incomplete artifact."""
    normalised = _normalise(identity)
    directory = _directory(root, normalised)
    if (directory / "manifest.json").exists():
        return _verify(directory, normalised)
    if directory.exists():
        raise FileExistsError(
            f"Incomplete baseline directory without manifest: {directory}; "
            "it is not adopted. Remove it and rerun --baselines."
        )
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = directory.with_name(f".{directory.name}.{uuid4().hex}.tmp")
    staging.mkdir()
    try:
        extra = write(staging)
        files = {
            path.name: _sha256(path)
            for path in sorted(staging.iterdir())
            if path.is_file()
        }
        manifest = {
            "baseline": normalised["baseline"],
            "identity": normalised,
            "files": files,
            **_normalise(extra),
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2))
        if directory.exists():
            raise FileExistsError(f"Baseline appeared during run: {directory}")
        staging.rename(directory)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return directory


# --------------------------------------------------------------- contact prior


def _training_pair_table(dataset: ContactDataset) -> dict[tuple[str, int, int], bool]:
    """Canonical training pairs keyed by (clean sequence, original i, original j).

    Overlapping units or duplicate chains of one protein collapse to one ID; a
    pair is a contact if any labelled copy marks it so. Only labelled training
    units contribute, and pairs closer than MIN_SEPARATION are never scored.
    """
    table: dict[tuple[str, int, int], bool] = {}
    for chain in dataset.splits["train"]:
        alignment = chain.alignment
        for unit in chain.units:
            if not unit.labels_available:
                continue
            candidates = candidate_pairs(unit, alignment, minsep=priors.MIN_SEPARATION)
            for i, j, label in zip(
                candidates.original_i.tolist(),
                candidates.original_j.tolist(),
                candidates.labels.tolist(),
                strict=True,
            ):
                key = (alignment.clean_sequence, int(i), int(j))
                table[key] = table.get(key, False) or bool(label)
    return table


def separation_training_pairs(dataset: ContactDataset) -> list[tuple[int, bool]]:
    """``(separation, is_contact)`` for each canonical training pair, deduplicated."""
    return [
        (j - i, is_contact)
        for (_, i, j), is_contact in sorted(_training_pair_table(dataset).items())
    ]


def separation_matrix(
    prior: priors.SeparationPrior, alignment: ContactAlignment
) -> np.ndarray:
    """Square retained-order matrix of prior probabilities.

    Separations use original residue coordinates, so non-canonical gaps widen
    the separation exactly as they do for the contact labels.
    """
    retained = np.asarray(alignment.retained_positions, dtype=np.int64)
    separation = np.abs(retained[:, None] - retained[None, :])
    matrix = np.zeros(separation.shape, dtype=np.float32)
    scored = separation >= priors.MIN_SEPARATION
    values = {
        int(value): prior.probability(int(value))
        for value in np.unique(separation[scored])
    }
    matrix[scored] = [values[int(value)] for value in separation[scored]]
    return matrix


def _contact_fit_identity(
    dataset: ContactDataset, dataset_identity: Mapping[str, Any]
) -> dict[str, Any]:
    table = _training_pair_table(dataset)
    return {
        "dataset": dict(dataset_identity),
        "fit_split": "train",
        "pairs": identity_hash(
            [[seq, i, j, int(label)] for (seq, i, j), label in sorted(table.items())]
        ),
        "pair_count": len(table),
    }


def _contact_test_identity(
    dataset: ContactDataset, dataset_identity: Mapping[str, Any]
) -> dict[str, Any]:
    units = [
        [unit.id, unit.labels_available, list(unit.positive_pairs)]
        for chain in dataset.splits["test"]
        for unit in chain.units
    ]
    return {
        "dataset": dict(dataset_identity),
        "split": "test",
        "fingerprint": identity_hash(units),
    }


def _compute_contact_scores(
    dataset: ContactDataset, prior: priors.SeparationPrior, seed: int
) -> Any:
    task = ContactPredictionTask()
    metadata = EvaluationMetadata(
        model_name=CONTACT_MODEL_NAME,
        dataset_name=dataset.source,
        task_type="contact_prediction",
        split="test",
        dataset_repo_id=dataset.source,
        seed=seed,
    )

    def predictor(chain: Any) -> np.ndarray:
        return separation_matrix(prior, chain.alignment)

    predictions = task.predict_units(dataset, predictor, metadata, split="test")
    from modules.evaluate.src.predictors.contact import score_contact_predictions

    return score_contact_predictions(predictions, bootstrap_enabled=False, metrics=None)


def run_contact_separation_baseline(
    *,
    dataset: ContactDataset,
    dataset_identity: Mapping[str, Any],
    root: Path,
    seed: int,
) -> Path:
    """Fit the separation prior on train, score the test split, publish once."""
    protocol = BASELINE_PROTOCOLS[CONTACT_BASELINE]
    identity = {
        "baseline": CONTACT_BASELINE,
        "protocol": protocol,
        "fit": _contact_fit_identity(dataset, dataset_identity),
        "test": _contact_test_identity(dataset, dataset_identity),
    }

    def write(staging: Path) -> dict[str, Any]:
        pairs = separation_training_pairs(dataset)
        prior = priors.fit_separation_prior(pairs)
        scores = _compute_contact_scores(dataset, prior, seed)
        scores.unit_scores.write_parquet(staging / "unit_scores.parquet")
        scores.summaries.write_parquet(staging / "summaries.parquet")
        return {"training_pairs": len(pairs), "rows": scores.unit_scores.height}

    return _publish(Path(root), identity, write)


# ------------------------------------------------------- amino-acid frequency


def load_amino_acid_reference(
    reference: AminoAcidReference,
    *,
    loader: Callable[..., Iterable[Mapping[str, Any]]] | None = None,
) -> list[str]:
    """Read the pinned train split of a declared corpus (network at execution only)."""
    if loader is None:
        loader = _hub_train_loader
    rows = loader(reference.repo_id, split=reference.split, revision=reference.revision)
    return [str(row[reference.sequence_column]) for row in rows]


def _hub_train_loader(
    repo_id: str, *, split: str, revision: str
) -> Iterable[Mapping[str, Any]]:
    from datasets import load_dataset

    return load_dataset(repo_id, split=split, revision=revision)


def _compute_amino_acid_rows(
    frequencies: np.ndarray, test_rows: Sequence[tuple[str, str]]
) -> tuple[pl.DataFrame, dict[str, Any]]:
    ids: list[str] = []
    labels: list[str] = []
    predictions: list[float] = []
    excluded = 0
    for example_id, sequence in test_rows:
        for position, residue in enumerate(sequence.upper()):
            if residue not in CANONICAL_AMINO_ACIDS:
                excluded += 1
                continue
            ids.append(f"{example_id}:{position}")
            labels.append(residue)
            predictions.append(priors.amino_acid_log_probability(frequencies, residue))
    if not predictions:
        raise ValueError(
            "amino_acid_frequency has no canonical test residues to score."
        )
    perplexity = float(np.exp(-np.mean(predictions)))
    frame = pl.DataFrame({"id": ids, "label": labels, "prediction": predictions})
    return frame, {
        "pseudo_perplexity": perplexity,
        "residues": len(predictions),
        "excluded_residues": excluded,
    }


def run_amino_acid_frequency_baseline(
    *,
    reference: AminoAcidReference | None,
    reference_sequences: Sequence[str],
    test_rows: Sequence[tuple[str, str]],
    dataset_identity: Mapping[str, Any],
    root: Path,
) -> Path:
    """Fit residue frequencies on the declared reference train split; score test."""
    if reference is None:
        raise ValueError(
            "amino_acid_frequency requires baselines.amino_acid_reference "
            "(repo_id plus a 40-hex commit SHA of a train-only corpus)."
        )
    frequencies = priors.fit_amino_acid_frequencies(reference_sequences)
    identity = {
        "baseline": AMINO_ACID_BASELINE,
        "protocol": BASELINE_PROTOCOLS[AMINO_ACID_BASELINE],
        "fit": {
            "reference": reference.model_dump(),
            "frequencies": identity_hash(frequencies.tolist()),
            "fit_policy": "reference_train_only",
        },
        "test": {
            "dataset": dict(dataset_identity),
            "split": "test",
            "fingerprint": identity_hash([[i, s] for i, s in test_rows]),
        },
    }

    def write(staging: Path) -> dict[str, Any]:
        frame, metrics = _compute_amino_acid_rows(frequencies, test_rows)
        frame.write_parquet(staging / "predictions.parquet")
        return {"rows": frame.height, "metrics": metrics}

    return _publish(Path(root), identity, write)
