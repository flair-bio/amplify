"""Classical estimator construction and direct inference."""

from __future__ import annotations

import logging
from time import monotonic
from typing import Any

import numpy as np
import torch
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from modules.evaluate.src.config import HeadType

logger = logging.getLogger(__name__)
_ESTIMATOR_REGISTRY = {
    "sklearn_linear": (LogisticRegression, Ridge, {"max_iter": 1000}),
    "knn": (KNeighborsClassifier, KNeighborsRegressor, {"n_neighbors": 1}),
    "random_forest": (
        RandomForestClassifier,
        RandomForestRegressor,
        {"n_estimators": 100, "n_jobs": -1},
    ),
}
_SCALED_HEAD_TYPES = frozenset({"sklearn_linear", "knn"})
_CLASSIFICATION_ONLY_PARAMETERS = frozenset({"C"})
_REGRESSION_ONLY_PARAMETERS = frozenset({"alpha"})


def _xgboost_estimator_classes() -> tuple[type[BaseEstimator], type[BaseEstimator]]:
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except ImportError as exc:
        raise ImportError(
            "head_type='xgboost' requires the 'xgboost' package. Install it "
            "via `uv pip install -e '.[evaluate]'` or `pip install xgboost`."
        ) from exc
    return XGBClassifier, XGBRegressor


def build_estimator(
    head_type: HeadType,
    task_type: str,
    hyperparameters: dict[str, Any],
    seed: int | None = None,
) -> BaseEstimator:
    """Instantiate the sklearn-compatible estimator for *head_type*/*task_type*."""
    is_regression = task_type == "sequence_regression"
    if head_type == "xgboost":
        classifier_cls, regressor_cls = _xgboost_estimator_classes()
        defaults: dict[str, Any] = {}
    else:
        classifier_cls, regressor_cls, defaults = _ESTIMATOR_REGISTRY[head_type]
    estimator_cls = regressor_cls if is_regression else classifier_cls
    if head_type == "sklearn_linear":
        defaults = {
            **defaults,
            ("alpha" if is_regression else "C"): 1.0,
        }
        allowed = (
            _REGRESSION_ONLY_PARAMETERS
            if is_regression
            else _CLASSIFICATION_ONLY_PARAMETERS
        )
        hyperparameters = {
            key: value
            for key, value in hyperparameters.items()
            if key
            not in (_CLASSIFICATION_ONLY_PARAMETERS | _REGRESSION_ONLY_PARAMETERS)
            or key in allowed
        }
    estimator = estimator_cls(**{**defaults, **hyperparameters})
    if (
        seed is not None
        and "random_state" not in hyperparameters
        and "random_state" in estimator.get_params(deep=False)
    ):
        estimator.set_params(random_state=seed)
    if head_type in _SCALED_HEAD_TYPES:
        # Pipeline delegates classes_/predict_proba to its final step.
        estimator = Pipeline([("scaler", StandardScaler()), ("estimator", estimator)])
    return estimator


def _classical_logits(
    estimator: BaseEstimator,
    inputs_embeds: torch.Tensor,
    task_type: str,
    num_labels: int,
) -> torch.Tensor:
    """Convert fitted-estimator outputs to the logits expected by task handlers."""
    features = inputs_embeds.detach().cpu().numpy()
    probabilities = (
        estimator.predict_proba(features)
        if hasattr(estimator, "predict_proba")
        else None
    )
    if probabilities is not None:
        classes = np.asarray(estimator.classes_, dtype=np.int64)
        full_probabilities = np.full(
            (probabilities.shape[0], num_labels), 1e-12, dtype=np.float64
        )
        full_probabilities[:, classes] = np.clip(probabilities, 1e-12, 1.0)
        return torch.from_numpy(np.log(full_probabilities)).float()
    if task_type == "sequence_regression":
        return torch.as_tensor(
            estimator.predict(features), dtype=torch.float32
        ).unsqueeze(-1)
    return nn.functional.one_hot(
        torch.as_tensor(np.asarray(estimator.predict(features), dtype=np.int64)),
        num_classes=num_labels,
    ).float()


def run_classical_inference(
    estimator: BaseEstimator,
    dataloader: Any,
    task_type: str,
    num_labels: int,
    collect_probabilities: bool = False,
    progress_label: str | None = None,
) -> tuple[list[Any], list[Any], list[Any], list[Any] | None]:
    """Run a fitted estimator over a cached-embedding dataloader."""
    from modules.evaluate.src.tasks import get_task

    task = get_task(task_type)
    all_ids: list[Any] = []
    all_predictions: list[Any] = []
    all_labels: list[Any] = []
    all_probabilities: list[Any] | None = [] if collect_probabilities else None
    started = last_log = monotonic()
    for batch_idx, batch in enumerate(dataloader, 1):
        batch = dict(batch)
        ids = batch.pop("id", None)
        labels = batch.pop("labels")
        logits = _classical_logits(
            estimator, batch["inputs_embeds"], task_type, num_labels
        )
        predictions = task.extract_predictions(logits, labels=labels)
        for prediction, label in task.split_batch_rows(predictions, labels):
            all_predictions.append(prediction)
            all_labels.append(label)
        if ids is None:
            all_ids.extend(range(len(all_ids), len(all_ids) + len(predictions)))
        else:
            all_ids.extend(ids.tolist() if torch.is_tensor(ids) else ids)
        if all_probabilities is not None:
            all_probabilities.extend(logits.softmax(dim=-1).tolist())
        if progress_label:
            now = monotonic()
            if now - last_log >= 10:
                logger.info(
                    "%s: %d batches completed (%.1fs elapsed)",
                    progress_label,
                    batch_idx,
                    now - started,
                )
                last_log = now
    return all_ids, all_predictions, all_labels, all_probabilities


_DEFAULT_SEARCH_SPACES: dict[str, dict[str, list[Any]]] = {
    "knn": {"n_neighbors": [1, 3, 5]}
}


def _filter_classical_search_space(
    head_type: HeadType, task_type: str, search_space: dict[str, list[Any]]
) -> dict[str, list[Any]]:
    """Remove estimator parameters that cannot apply to this task."""
    if head_type != "sklearn_linear":
        return search_space
    allowed = (
        _REGRESSION_ONLY_PARAMETERS
        if task_type == "sequence_regression"
        else _CLASSIFICATION_ONLY_PARAMETERS
    )
    return {
        key: values
        for key, values in search_space.items()
        if key not in _CLASSIFICATION_ONLY_PARAMETERS | _REGRESSION_ONLY_PARAMETERS
        or key in allowed
    }


def classical_search_space(
    head_type: HeadType,
    task_type: str,
    search_space: dict[str, list[Any]],
    default_hyperparameters: dict[str, Any],
) -> dict[str, list[Any]]:
    """Resolve the task-compatible search candidates for a classical head."""
    filtered = _filter_classical_search_space(head_type, task_type, search_space)
    if filtered:
        return filtered
    if not default_hyperparameters:
        return _DEFAULT_SEARCH_SPACES.get(head_type, {})
    return {}


def has_classical_search_space(
    head_type: HeadType,
    search_space: dict[str, list[Any]],
    default_hyperparameters: dict[str, Any],
) -> bool:
    """Whether this head configuration selects among multiple candidates."""
    return bool(
        search_space
        or (_DEFAULT_SEARCH_SPACES.get(head_type, {}) and (not default_hyperparameters))
    )
