"""MMseqs2 nearest-neighbour sequence baselines (protocol version 1).

Each test sequence is searched against the training sequences with
``mmseqs easy-search`` using fixed settings. Neighbours are ordered by bit
score (descending), E-value (ascending), then training index (ascending);
duplicate training targets keep their best hit; the top ``k`` hits are kept.

* Classification: each neighbour votes for its class with weight equal to its
  bit score. Probabilities are normalized votes; the prediction is the argmax
  with ties to the lowest class index.
* Regression: bit-score-weighted mean of neighbour targets.
* No neighbour: classification uses the training class frequencies and
  regression uses the training mean. Both are train-only.

Search is single-threaded so output is reproducible. The binary is resolved
from ``PATH`` and missing binaries are an explicit error.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

MMSEQS_KNN_PROTOCOL: dict[str, Any] = {
    "version": 1,
    "k": 5,
    "sensitivity": 7.5,
    "evalue": 1e-3,
    "max_seqs": 300,
    "threads": 1,
    "neighbor_order": "bits descending, evalue ascending, training index ascending",
    "deduplicate": "keep the best hit per training target",
    "weighting": "bit score",
    "no_hit_classification": "training class frequencies",
    "no_hit_regression": "training mean",
}


@dataclass(frozen=True)
class Hit:
    target: int
    bits: float
    evalue: float


def parse_search_hits(text: str) -> dict[int, list[Hit]]:
    """Parse ``query,target,bits,evalue`` rows written by ``easy-search``."""
    hits: dict[int, list[Hit]] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        query, target, bits, evalue = line.split("\t")
        hits.setdefault(int(query[1:]), []).append(
            Hit(target=int(target[1:]), bits=float(bits), evalue=float(evalue))
        )
    return hits


def select_neighbors(
    hits: Mapping[int, Sequence[Hit]], *, num_queries: int, k: int
) -> dict[int, list[Hit]]:
    """Return the deterministic top-``k`` unique-target neighbours per query."""
    selected: dict[int, list[Hit]] = {}
    for query in range(num_queries):
        ordered = sorted(
            hits.get(query, ()), key=lambda hit: (-hit.bits, hit.evalue, hit.target)
        )
        seen: set[int] = set()
        unique: list[Hit] = []
        for hit in ordered:
            if hit.target in seen:
                continue
            seen.add(hit.target)
            unique.append(hit)
        selected[query] = unique[:k]
    return selected


def knn_classify(
    hits: Mapping[int, Sequence[Hit]],
    *,
    train_labels: Sequence[int],
    num_queries: int,
    num_labels: int,
    k: int = MMSEQS_KNN_PROTOCOL["k"],
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(predictions, probabilities)`` by bit-weighted neighbour votes."""
    neighbors = select_neighbors(hits, num_queries=num_queries, k=k)
    counts = Counter(train_labels)
    prior = np.array([counts[index] for index in range(num_labels)], dtype=float)
    prior = prior / prior.sum()
    probabilities = np.zeros((num_queries, num_labels), dtype=np.float64)
    for query in range(num_queries):
        if not neighbors[query]:
            probabilities[query] = prior
            continue
        votes = np.zeros(num_labels, dtype=np.float64)
        for hit in neighbors[query]:
            votes[int(train_labels[hit.target])] += hit.bits
        probabilities[query] = votes / votes.sum()
    return probabilities.argmax(axis=1), probabilities


def knn_regress(
    hits: Mapping[int, Sequence[Hit]],
    *,
    train_labels: Sequence[float],
    num_queries: int,
    k: int = MMSEQS_KNN_PROTOCOL["k"],
) -> np.ndarray:
    """Return bit-weighted neighbour means, falling back to the training mean."""
    neighbors = select_neighbors(hits, num_queries=num_queries, k=k)
    fallback = float(np.mean(np.asarray(train_labels, dtype=float)))
    predictions = np.empty(num_queries, dtype=np.float64)
    for query in range(num_queries):
        if not neighbors[query]:
            predictions[query] = fallback
            continue
        weights = np.array([hit.bits for hit in neighbors[query]], dtype=float)
        values = np.array([float(train_labels[hit.target]) for hit in neighbors[query]])
        predictions[query] = float(np.sum(weights * values) / weights.sum())
    return predictions


def run_search(
    queries: Sequence[str],
    targets: Sequence[str],
    *,
    work_root: Path | None = None,
) -> dict[int, list[Hit]]:
    """Search each query against all targets; keys are query row indices."""
    binary = shutil.which("mmseqs")
    if binary is None:
        raise RuntimeError(
            "The mmseqs_knn baseline requires the 'mmseqs' binary on PATH."
        )
    protocol = MMSEQS_KNN_PROTOCOL
    with tempfile.TemporaryDirectory(dir=work_root) as directory:
        root = Path(directory)
        query_path = root / "queries.fasta"
        target_path = root / "targets.fasta"
        output_path = root / "hits.tsv"
        query_path.write_text(
            "".join(f">q{i}\n{seq}\n" for i, seq in enumerate(queries))
        )
        target_path.write_text(
            "".join(f">t{i}\n{seq}\n" for i, seq in enumerate(targets))
        )
        argv = [
            binary,
            "easy-search",
            str(query_path),
            str(target_path),
            str(output_path),
            str(root / "tmp"),
            "-s",
            str(protocol["sensitivity"]),
            "-e",
            str(protocol["evalue"]),
            "--max-seqs",
            str(protocol["max_seqs"]),
            "--threads",
            str(protocol["threads"]),
            "--format-output",
            "query,target,bits,evalue",
        ]
        try:
            subprocess.run(
                argv, check=True, capture_output=True, text=True, shell=False
            )
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"mmseqs easy-search failed: {error.stderr.strip()}"
            ) from error
        return parse_search_hits(
            output_path.read_text() if output_path.exists() else ""
        )
