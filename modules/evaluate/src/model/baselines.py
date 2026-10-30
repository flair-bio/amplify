"""Model-independent sequence and residue-frequency baselines."""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch

from modules.evaluate.src.contracts.config import BaselineName
from modules.evaluate.src.tasks import get_task

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AMBIGUOUS_AMINO_ACIDS = frozenset("XBOUZJ")


def build_sequence_baseline(
    name: BaselineName,
    task_type: str,
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    num_labels: int | None,
    *,
    seed: int = 710019,
) -> pl.DataFrame:
    """Build model-independent, train-only sequence prediction rows."""
    labels = [row["label"] for row in train_rows]
    if task_type == "token_classification":
        if name != "majority_class" or not num_labels:
            raise ValueError("Token baselines require majority_class and num_labels.")
        valid_labels = [
            label
            for sequence_labels in labels
            for label in sequence_labels
            if label != -100
        ]
        if not valid_labels:
            raise ValueError("The train split contains no valid residue labels.")
        counts = np.bincount(valid_labels, minlength=num_labels)
        majority = int(counts.argmax())
        probabilities = (counts / counts.sum()).tolist()
        # Store one prior per protein, not a probability vector for every residue.
        return pl.DataFrame(
            {
                "id": [row["id"] for row in test_rows],
                "label": [row["label"] for row in test_rows],
                "prediction": [majority] * len(test_rows),
                "probability": [probabilities] * len(test_rows),
            }
        )
    if task_type == "sequence_classification":
        if name not in {"majority_class", "composition", "mmseqs_knn"}:
            raise ValueError(f"baseline={name!r} is not valid for classification.")
        if not num_labels:
            raise ValueError(f"{name} requires a positive num_labels value.")
        if name == "majority_class":
            frequencies: dict[int, int] = defaultdict(int)
            for label in labels:
                frequencies[label] += 1
            counts = [frequencies[index] for index in range(num_labels)]
            prediction = max(range(len(counts)), key=counts.__getitem__)
            probabilities = [count / len(labels) for count in counts]
            baseline_predictions = [prediction] * len(test_rows)
            probability_rows = [probabilities] * len(test_rows)
        elif name == "composition" and len(set(labels)) == 1:
            prediction = int(labels[0])
            baseline_predictions = [prediction] * len(test_rows)
            probability_rows = [
                [float(class_index == prediction) for class_index in range(num_labels)]
                for _ in test_rows
            ]
        elif name == "composition":
            from sklearn.linear_model import LogisticRegression

            features = np.stack(
                [amino_acid_composition(row["sequence"]) for row in train_rows]
            )
            test_features = np.stack(
                [amino_acid_composition(row["sequence"]) for row in test_rows]
            )
            estimator = LogisticRegression(
                max_iter=1000, random_state=seed, solver="lbfgs"
            )
            estimator.fit(features, np.asarray(labels, dtype=np.int64))
            baseline_predictions = estimator.predict(test_features).astype(int).tolist()
            probability_rows = _class_probabilities(
                estimator.predict_proba(test_features),
                estimator.classes_,
                num_labels,
            )
        elif name == "mmseqs_knn":
            neighbors = _mmseqs_nearest_neighbors(train_rows, test_rows)
            class_counts = np.bincount(labels, minlength=num_labels)
            majority = int(class_counts.argmax())
            prior = (class_counts / class_counts.sum()).tolist()
            baseline_predictions = [
                majority if index is None else int(train_rows[index]["label"])
                for index in neighbors
            ]
            probability_rows = [
                (
                    prior
                    if index is None
                    else [
                        float(class_index == prediction)
                        for class_index in range(num_labels)
                    ]
                )
                for index, prediction in zip(
                    neighbors, baseline_predictions, strict=True
                )
            ]
        else:
            raise ValueError(f"Unknown classification baseline: {name}.")
        frame = pl.DataFrame(
            {
                "id": [row["id"] for row in test_rows],
                "label": [row["label"] for row in test_rows],
                "prediction": baseline_predictions,
                "probability": probability_rows,
            }
        )
        if name == "mmseqs_knn":
            frame = frame.with_columns(
                pl.Series("fallback", [index is None for index in neighbors])
            )
        return frame
    if task_type != "sequence_regression":
        raise ValueError(f"Sequence baseline is not supported for {task_type}.")
    if name not in {"train_mean", "composition", "mmseqs_knn"}:
        raise ValueError(f"baseline={name!r} is not valid for regression.")
    if name == "train_mean":
        train_mean = math.fsum(labels) / len(labels)
        regression_predictions = [train_mean] * len(test_rows)
    elif name == "composition":
        from sklearn.linear_model import Ridge

        features = np.stack(
            [amino_acid_composition(row["sequence"]) for row in train_rows]
        )
        test_features = np.stack(
            [amino_acid_composition(row["sequence"]) for row in test_rows]
        )
        estimator = Ridge(alpha=1.0)
        estimator.fit(features, np.asarray(labels, dtype=float))
        regression_predictions = estimator.predict(test_features).tolist()
    else:
        neighbors = _mmseqs_nearest_neighbors(train_rows, test_rows)
        train_mean = math.fsum(labels) / len(labels)
        regression_predictions = [
            train_mean if index is None else float(train_rows[index]["label"])
            for index in neighbors
        ]
    frame = pl.DataFrame(
        {
            "id": [row["id"] for row in test_rows],
            "label": [row["label"] for row in test_rows],
            "prediction": regression_predictions,
        }
    )
    if name == "mmseqs_knn":
        frame = frame.with_columns(
            pl.Series("fallback", [index is None for index in neighbors])
        )
    return frame


def amino_acid_composition(sequence: str) -> np.ndarray:
    """Return canonical amino-acid frequencies, ignoring ambiguous symbols."""
    if not isinstance(sequence, str) or not sequence:
        raise ValueError("Composition baselines require nonempty protein sequences.")
    sequence = sequence.upper()
    counts = np.asarray([sequence.count(amino_acid) for amino_acid in AMINO_ACIDS])
    total = int(counts.sum())
    if not total:
        raise ValueError("Composition baseline found no canonical amino acids.")
    return counts.astype(np.float64) / total


def _class_probabilities(
    probabilities: np.ndarray, classes: np.ndarray, num_labels: int
) -> list[list[float]]:
    full = np.zeros((len(probabilities), num_labels), dtype=np.float64)
    full[:, classes.astype(int)] = probabilities
    return full.tolist()


def _mmseqs_sequence(sequence: str) -> str:
    return "".join(
        amino_acid if amino_acid in AMINO_ACIDS else "X"
        for amino_acid in sequence.upper()
    )


def _mmseqs_nearest_neighbors(
    train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]
) -> list[int | None]:
    """Return significant top training hits, or None when no hit passes."""
    executable = shutil.which("mmseqs")
    if executable is None:
        raise FileNotFoundError(
            "The mmseqs_knn baseline requires the MMseqs2 'mmseqs' executable on PATH."
        )
    if not train_rows or not test_rows:
        raise ValueError("MMseqs k-NN requires nonempty train and test sequences.")
    for row in [*train_rows, *test_rows]:
        sequence = row["sequence"]
        if (
            not isinstance(sequence, str)
            or not sequence
            or any(
                amino_acid.upper() not in AMINO_ACIDS
                and amino_acid.upper() not in AMBIGUOUS_AMINO_ACIDS
                for amino_acid in sequence
            )
            or any(character.isspace() for character in sequence)
        ):
            raise ValueError("MMseqs k-NN requires valid protein sequences.")

    with tempfile.TemporaryDirectory(prefix="amplify_mmseqs_knn_") as temporary:
        temp_path = Path(temporary)
        query_path = temp_path / "query.fasta"
        target_path = temp_path / "train.fasta"
        result_path = temp_path / "matches.tsv"
        tmp_path = temp_path / "tmp"
        query_path.write_text(
            "".join(
                f">test_{i}\n{_mmseqs_sequence(row['sequence'])}\n"
                for i, row in enumerate(test_rows)
            )
        )
        target_path.write_text(
            "".join(
                f">train_{i}\n{_mmseqs_sequence(row['sequence'])}\n"
                for i, row in enumerate(train_rows)
            )
        )
        result = subprocess.run(
            [
                executable,
                "easy-search",
                str(query_path),
                str(target_path),
                str(result_path),
                str(tmp_path),
                "--format-output",
                "query,target,bits,evalue",
                "--max-seqs",
                "300",
                "-s",
                "7.5",
                "--mask",
                "1",
                "--comp-bias-corr",
                "1",
                "--threads",
                "1",
                "-e",
                "0.001",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise RuntimeError(
                "MMseqs2 nearest-neighbor search failed: "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )
        if not result_path.exists():
            raise RuntimeError("MMseqs2 completed without producing its result table.")
        matches: dict[int, tuple[float, float, int]] = {}
        for line in result_path.read_text().splitlines():
            query, target, bits, evalue = line.split("\t")
            query_index = int(query.removeprefix("test_"))
            target_index = int(target.removeprefix("train_"))
            if not 0 <= query_index < len(test_rows) or not 0 <= target_index < len(
                train_rows
            ):
                raise ValueError("MMseqs2 returned an unknown query or training ID.")
            bit_score, expectation = float(bits), float(evalue)
            if (
                not math.isfinite(bit_score)
                or not math.isfinite(expectation)
                or expectation < 0
            ):
                raise ValueError("MMseqs2 returned an invalid alignment score.")
            if expectation > 0.001:
                continue
            # Highest bit score, then lowest E-value, then training row order.
            rank = (-bit_score, expectation, target_index)
            if query_index not in matches or rank < matches[query_index]:
                matches[query_index] = rank
        return [
            matches[index][2] if index in matches else None
            for index in range(len(test_rows))
        ]


def build_amino_acid_frequency_baseline(
    rows: list[dict[str, Any]], *, max_residues: int
) -> pl.DataFrame:
    """Score each test residue under a smoothed, corpus-wide IID amino-acid prior."""
    if max_residues < 1:
        raise ValueError("max_residues must be positive.")
    if not rows:
        raise ValueError("Amino-acid-frequency baseline requires test sequences.")
    sequences = [row["sequence"][:max_residues].upper() for row in rows]
    unknown_index = len(AMINO_ACIDS)
    frequency_symbols = f"{AMINO_ACIDS}X"
    residue_indices = {
        amino_acid: index for index, amino_acid in enumerate(AMINO_ACIDS)
    }
    counts = np.ones(len(frequency_symbols), dtype=np.float64)
    for sequence in sequences:
        for amino_acid in sequence:
            if (
                amino_acid not in AMINO_ACIDS
                and amino_acid not in AMBIGUOUS_AMINO_ACIDS
            ):
                raise ValueError(
                    f"Unsupported residue {amino_acid!r} in amino-acid-frequency data."
                )
            counts[residue_indices.get(amino_acid, unknown_index)] += 1
    frequencies = counts / counts.sum()
    output_ids: list[str] = []
    output_scores: list[float] = []
    for index, (row, sequence) in enumerate(zip(rows, sequences, strict=True)):
        example_id = row["id"]
        for position, amino_acid in enumerate(sequence):
            output_ids.append(f"{example_id}:{index}:{position}")
            output_scores.append(
                float(
                    math.log(
                        frequencies[residue_indices.get(amino_acid, unknown_index)]
                    )
                )
            )
    return pl.DataFrame({"id": output_ids, "prediction": output_scores})


def build_prediction_baseline(
    name: BaselineName,
    task_type: str,
    dataset_name: str,
    num_labels: int | None,
    predictions: pl.DataFrame,
    train_labels: torch.Tensor,
) -> pl.DataFrame:
    """Build sequence-level or token-level constant baseline rows."""
    if name == "majority_class":
        if task_type == "sequence_regression":
            raise ValueError(
                "baseline='majority_class' is only defined for classification "
                "tasks; use the train-mean regression baseline instead."
            )
        class_count = get_task(task_type).resolve_num_labels(num_labels, dataset_name)
        counts = torch.bincount(train_labels.to(torch.long), minlength=class_count)
        majority_class = int(counts.argmax().item())
        class_probabilities = (counts / counts.sum()).tolist()
        baseline_predictions = [
            (
                [majority_class] * len(prediction)
                if isinstance(prediction, list)
                else majority_class
            )
            for prediction in predictions["prediction"].to_list()
        ]
        data: dict[str, list[Any]] = {
            "id": predictions["id"].to_list(),
            "prediction": baseline_predictions,
        }
        if "probability" in predictions.columns:
            data["probability"] = [
                (
                    [class_probabilities] * len(prediction)
                    if isinstance(prediction, list)
                    else class_probabilities
                )
                for prediction in baseline_predictions
            ]
        return pl.DataFrame(data)
    if task_type != "sequence_regression":
        raise ValueError(
            "baseline='train_mean' is only defined for sequence_regression tasks; "
            "use a majority_class baseline for classification tasks."
        )
    train_mean = train_labels.to(torch.float32).mean().item()
    return pl.DataFrame(
        {
            "id": predictions["id"].to_list(),
            "prediction": [train_mean] * predictions.height,
        }
    )
