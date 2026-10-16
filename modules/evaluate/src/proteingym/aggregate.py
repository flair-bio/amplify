"""Per-assay metrics (Spearman/NDCG/Top_recall for continuous DMS scores,
ROC-AUC/MCC for binary labels) and the official ProteinGym hierarchical
aggregation: mean across assays within a UniProt ID, then mean across
UniProt IDs within a function category, then mean across categories (each
weighted equally)."""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.metrics import matthews_corrcoef, roc_auc_score

from modules.evaluate.src.metrics.metrics import spearman as _spearman

# ProteinGym's raw clinical label column ("DMS_score_bin"/"label") stores
# "Pathogenic"/"Benign" strings, not 0/1 -- both the DMS and clinical tracks
# route through this, so it needs to accept whichever form the source gives.
_STRING_LABEL_MAP = {"pathogenic": 1, "benign": 0}


def binarize_labels(values: object) -> np.ndarray:
    """Coerce a ``dms_score_bin`` column to a numeric 0/1 array.

    Already-numeric input is returned unchanged; string labels (e.g.
    ProteinGym's clinical "Pathogenic"/"Benign") are mapped to 1/0.
    """
    array = np.asarray(values)
    if array.dtype.kind in "fiub":
        return array.astype(float)
    return np.array(
        [_STRING_LABEL_MAP[str(v).strip().lower()] for v in array], dtype=float
    )


def compute_assay_spearman(model_scores: np.ndarray, dms_scores: np.ndarray) -> float:
    """Spearman rank correlation for one DMS assay, ignoring unscoreable variants."""
    valid = ~np.isnan(model_scores)
    if valid.sum() < 2:
        return float("nan")
    return _spearman(model_scores[valid], dms_scores[valid], average="")


def compute_assay_auc(
    model_scores: np.ndarray, labels: np.ndarray, flip_labels: bool = False
) -> float:
    """ROC-AUC for one binary-labeled (e.g. clinical pathogenic/benign) assay.

    ``flip_labels`` inverts a pathogenic=1/benign=0 label so that, as with DMS
    fitness scores, higher is "better" (benign) -- matching the direction of
    zero-shot model scores. Ported from the official
    ``performance_clinical_benchmarks.py``'s ``y_true = 1 - y_true`` step.
    """
    labels = binarize_labels(labels)
    valid = ~np.isnan(model_scores)
    model_scores, labels = model_scores[valid], labels[valid]
    if flip_labels:
        labels = 1 - labels
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, model_scores))


def compute_assay_ndcg(
    model_scores: np.ndarray, dms_scores: np.ndarray, top: float = 10.0
) -> float:
    """NDCG@top-decile against min-max-normalized DMS scores, ported from the
    official ``calc_ndcg`` (gains of 0 are dropped from both the observed and
    ideal rankings before discounting).
    """
    valid = ~np.isnan(model_scores)
    if valid.sum() < 2:
        return float("nan")
    model_scores, dms_scores = model_scores[valid], dms_scores[valid]
    k = int(np.floor(len(dms_scores) * (top / 100)))
    if k < 1:
        return float("nan")
    score_range = dms_scores.max() - dms_scores.min()
    if score_range == 0:
        return float("nan")
    gains = (dms_scores - dms_scores.min()) / score_range

    def _discounted_gain(ranks: np.ndarray) -> float:
        ranks_k, gains_k = ranks[ranks <= k], gains[ranks <= k]
        ranks_fil, gains_fil = ranks_k[gains_k != 0], gains_k[gains_k != 0]
        if len(ranks_fil) == 0:
            return 0.0
        return float(np.sum(gains_fil / np.log2(ranks_fil + 1)))

    ranks = np.argsort(np.argsort(-model_scores)) + 1
    ideal_ranks = np.argsort(np.argsort(-gains)) + 1
    dcg = _discounted_gain(ranks)
    idcg = _discounted_gain(ideal_ranks)
    return dcg / idcg if idcg > 0 else 0.0


def compute_assay_top_recall(
    model_scores: np.ndarray,
    dms_scores: np.ndarray,
    top_true: float = 10.0,
    top_model: float = 10.0,
) -> float:
    """Fraction of the top-decile-true variants also in the top-decile-model
    variants, ported from the official ``calc_toprecall``."""
    valid = ~np.isnan(model_scores)
    if valid.sum() < 2:
        return float("nan")
    model_scores, dms_scores = model_scores[valid], dms_scores[valid]
    top_true_mask = dms_scores >= np.percentile(dms_scores, 100 - top_true)
    top_model_mask = model_scores >= np.percentile(model_scores, 100 - top_model)
    if top_true_mask.sum() == 0:
        return 0.0
    return float((top_true_mask & top_model_mask).sum() / top_true_mask.sum())


def compute_assay_mcc(
    model_scores: np.ndarray, labels: np.ndarray, flip_labels: bool = False
) -> float:
    """Matthews correlation coefficient between the true binary label and the
    model score binarized at its own median, ported from the official MCC
    column (``performance_DMS_benchmarks.py``'s median-cutoff binarization).
    """
    labels = binarize_labels(labels)
    valid = ~np.isnan(model_scores)
    if valid.sum() < 2:
        return float("nan")
    model_scores, labels = model_scores[valid], labels[valid]
    if flip_labels:
        labels = 1 - labels
    if len(np.unique(labels)) < 2:
        return float("nan")
    predicted = (model_scores >= np.median(model_scores)).astype(int)
    if len(np.unique(predicted)) < 2:
        return float("nan")
    return float(matthews_corrcoef(labels, predicted))


def compute_assay_metric(
    model_scores: np.ndarray,
    dms_scores: np.ndarray | None = None,
    dms_score_bin: np.ndarray | None = None,
    flip_binary_labels: bool = False,
) -> tuple[str, float]:
    """Pick Spearman (continuous score available) or AUC (binary-only, e.g. clinical)."""
    if dms_scores is not None:
        return "spearman", compute_assay_spearman(model_scores, dms_scores)
    if dms_score_bin is not None:
        return "auc", compute_assay_auc(
            model_scores, dms_score_bin, flip_labels=flip_binary_labels
        )
    raise ValueError(
        "Assay has neither a continuous score nor a binary label to score against."
    )


def compute_all_assay_metrics(
    model_scores: np.ndarray,
    dms_scores: np.ndarray | None = None,
    dms_score_bin: np.ndarray | None = None,
    flip_binary_labels: bool = False,
) -> dict[str, float]:
    """Every metric applicable given the columns available, matching how the
    official DMS zero-shot benchmark reports Spearman/NDCG/Top_recall (from
    the continuous score) and AUC/MCC (from the binary label) as independent
    columns rather than picking one -- both sets are computed together when
    an assay has both a continuous score and a binary label.
    """
    metrics: dict[str, float] = {}
    if dms_scores is not None:
        metrics["spearman"] = compute_assay_spearman(model_scores, dms_scores)
        metrics["ndcg"] = compute_assay_ndcg(model_scores, dms_scores)
        metrics["top_recall"] = compute_assay_top_recall(model_scores, dms_scores)
    if dms_score_bin is not None:
        metrics["auc"] = compute_assay_auc(
            model_scores, dms_score_bin, flip_labels=flip_binary_labels
        )
        metrics["mcc"] = compute_assay_mcc(
            model_scores, dms_score_bin, flip_labels=flip_binary_labels
        )
    return metrics


def bootstrap_standard_error_by_category(
    assay_results: pl.DataFrame,
    score_column: str = "score",
    n_resamples: int = 10000,
    seed: int = 0,
) -> float:
    """Non-parametric bootstrap SE of the overall aggregate.

    Resamples (with replacement) the per-protein scores within each function
    category, averages within category then across categories, and takes the
    standard deviation across resamples -- the same procedure as the official
    leaderboard's ``compute_bootstrap_standard_error_functional_categories``.
    One deliberate difference: the leaderboard estimates the SE of a *model's
    score relative to the top-ranked model* (it is comparing many models at
    once); a single-model run has no other models to be relative to, so this
    reports the SE of the model's own aggregate score directly.
    """
    per_protein = (
        assay_results.group_by(["coarse_selection_type", "uniprot_id"])
        .agg(pl.col(score_column).mean())
        .sort(["coarse_selection_type", "uniprot_id"])
    )
    category_arrays = [
        per_protein.filter(pl.col("coarse_selection_type") == category)[
            score_column
        ].to_numpy()
        for category in sorted(per_protein["coarse_selection_type"].unique().to_list())
    ]
    rng = np.random.default_rng(seed)
    # Draw all resamples for a category at once (n_resamples x len(arr) index
    # matrix) instead of looping n_resamples times per category in Python.
    category_means = [
        np.nanmean(arr[rng.integers(0, len(arr), size=(n_resamples, len(arr)))], axis=1)
        for arr in category_arrays
    ]
    samples = np.mean(np.stack(category_means), axis=0)
    return float(np.std(samples, ddof=1))


def aggregate_proteingym_scores(
    assay_results: pl.DataFrame,
    score_column: str = "score",
    compute_uncertainty: bool = False,
    n_bootstrap: int = 10000,
    bootstrap_seed: int = 0,
) -> dict[str, object]:
    """Aggregate per-assay metrics the way the ProteinGym leaderboard does.

    ``assay_results`` must have columns ``dms_id``, ``uniprot_id``,
    ``coarse_selection_type``, ``score_column`` (Spearman or AUC values --
    whichever ``compute_assay_metric`` returned for these assays).
    """
    per_protein = assay_results.group_by(["coarse_selection_type", "uniprot_id"]).agg(
        pl.col(score_column).mean()
    )
    per_category = per_protein.group_by("coarse_selection_type").agg(
        pl.col(score_column).mean()
    )
    overall = float(per_category[score_column].mean())
    summary: dict[str, object] = {
        "overall": overall,
        "per_category": dict(
            zip(per_category["coarse_selection_type"], per_category[score_column])
        ),
    }
    if compute_uncertainty:
        summary["bootstrap_standard_error"] = bootstrap_standard_error_by_category(
            assay_results, score_column, n_bootstrap, bootstrap_seed
        )
    return summary


def aggregate_proteingym_metrics(
    assay_results: pl.DataFrame,
    metric_columns: list[str],
    compute_uncertainty: bool = False,
    n_bootstrap: int = 10000,
    bootstrap_seed: int = 0,
) -> dict[str, dict[str, object]]:
    """Aggregate each metric column independently, matching how the official
    benchmark reports Spearman/AUC/MCC/NDCG/Top_recall as separate tables
    rather than one combined score. Columns absent or entirely null (e.g.
    ``mcc``/``ndcg``/``top_recall`` for clinical-only assays) are skipped.
    """
    return {
        column: aggregate_proteingym_scores(
            assay_results, column, compute_uncertainty, n_bootstrap, bootstrap_seed
        )
        for column in metric_columns
        if column in assay_results.columns
        and assay_results[column].drop_nulls().len() > 0
    }
