"""
Pluggable prediction heads for the evaluation probe.

``EmbeddingProbeHead`` is a linear or two-layer torch probe over precomputed
embeddings. ``"torch_linear"`` is a single :class:`torch.nn.Linear` layer trained by
backpropagation through
:class:`~modules.evaluate.src.steps.tune.TuneStep`'s existing epoch/optimizer
loop. With ``steps.prepare.embedding_cache.enabled=True``, it uses the
standalone :class:`EmbeddingProbeHead`. Without the cache, it trains the
model's native head only when that head is itself a plain ``nn.Linear``.
This supports AMPLIFY sequence heads and all shipped token heads while
rejecting composite heads such as ESM's sequence-classification head.
``"mlp"`` remains the
existing native model-head path, and can also train a two-layer probe on
frozen cached embeddings. Every classical
head type (``"sklearn_linear"``, ``"knn"``, ``"random_forest"``,
``"xgboost"``) is a
scikit-learn-compatible estimator, fit once (optionally swept over a
hyperparameter grid, scored on the validation split) directly on cached
pooled embeddings -- these have no gradient, so they bypass TuneStep's
training loop entirely. They therefore require
``steps.prepare.embedding_cache.enabled=True``: a classical estimator needs a
fixed-size feature vector per example, not a token sequence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

if TYPE_CHECKING:
    import optuna

from modules.evaluate.src.utils.hpo import create_optuna_study, sample_candidates

logger = logging.getLogger(__name__)


@dataclass
class _HeadOutput:
    """Minimal model-output contract shared by every probe head."""

    loss: torch.Tensor | None
    logits: torch.Tensor


HeadType = Literal[
    "mlp",
    "torch_linear",
    "sklearn_linear",
    "knn",
    "random_forest",
    "xgboost",
]

# Head types with no gradient: fit once (or swept) on cached embeddings
# instead of trained through TuneStep's epoch loop.
CLASSICAL_HEAD_TYPES = frozenset({"sklearn_linear", "knn", "random_forest", "xgboost"})
TORCH_EMBEDDING_HEAD_TYPES = frozenset({"torch_linear"})

# head_type -> (classifier estimator class, regressor estimator class, default kwargs).
_ESTIMATOR_REGISTRY: dict[
    str, tuple[type[BaseEstimator], type[BaseEstimator], dict[str, Any]]
] = {
    "sklearn_linear": (LogisticRegression, Ridge, {"max_iter": 1000}),
    "knn": (KNeighborsClassifier, KNeighborsRegressor, {"n_neighbors": 1}),
    "random_forest": (
        RandomForestClassifier,
        RandomForestRegressor,
        {"n_estimators": 100, "n_jobs": -1},
    ),
}

# KNN evaluates the built-in candidate values; the other heads use their
# single default configuration unless a search space is supplied explicitly.
_DEFAULT_SEARCH_SPACES: dict[str, dict[str, list[Any]]] = {
    "knn": {"n_neighbors": [1, 3, 5]},
}

# Head types whose estimator is scale/distance-sensitive (linear coefficient
# magnitude, Euclidean nearest-neighbor distance); tree ensembles are
# scale-invariant and deliberately excluded.
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


class HeadConfig(BaseModel):
    """Selects and parameterizes the prediction head trained by TuneStep."""

    model_config = ConfigDict(extra="forbid")

    head_type: HeadType = Field(
        "mlp",
        description=(
            "'torch_linear': one torch.nn.Linear layer trained by gradient "
            "descent. With embedding_cache.enabled=True it is constructed "
            "over cached embeddings; without the cache, the loaded native "
            "task head must itself be a single nn.Linear. 'mlp': existing "
            "native model-head "
            "path. 'sklearn_linear'/'knn'/'random_forest'/'xgboost': "
            "classical scikit-learn-compatible estimators fit once on cached "
            "pooled embeddings -- require "
            "steps.prepare.embedding_cache.enabled=True in tune_probe mode."
        ),
    )
    hidden_size: int = Field(
        512,
        gt=0,
        description=(
            "Hidden layer width for an MLP probe over cached embeddings; "
            "ignored for other head types."
        ),
    )
    hyperparameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Constructor kwargs for the chosen estimator (ignored for 'mlp').",
    )
    search_space: dict[str, list[Any]] = Field(
        default_factory=dict,
        description=(
            "Hyperparameter grid swept via sklearn.model_selection.ParameterGrid "
            "(every combination is fit and scored on the validation split; "
            "ignored for 'mlp'). Empty means no sweep: fit once with "
            "'hyperparameters'."
        ),
    )


def build_probe_layers(
    input_dim: int, hidden_size: int | None, output_dim: int
) -> tuple[nn.Module, nn.Module]:
    """(projection, classifier) pair: identity+linear, or Linear-GELU-Linear
    when ``hidden_size`` is set. Shared shape for every torch probe head,
    including the standalone one in ``proteingym/supervised.py``.
    """
    if hidden_size is None:
        return nn.Identity(), nn.Linear(input_dim, output_dim)
    return (
        nn.Sequential(nn.Linear(input_dim, hidden_size), nn.GELU()),
        nn.Linear(hidden_size, output_dim),
    )


class EmbeddingProbeHead(nn.Module):
    """A linear or two-layer probe over precomputed embeddings.

    Replaces the frozen trunk and native task head after the embedding cache
    has materialized pooled features. It accepts the same ``forward`` output
    contract as the native model wrappers, so the training and inference
    paths do not need a separate probe-specific implementation.
    """

    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        task_type: str,
        probe_hidden_size: int | None = None,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.task_type = task_type
        self.embedding_size = hidden_size
        self.probe_hidden_size = probe_hidden_size
        self.projection, self.classifier = build_probe_layers(
            hidden_size, probe_hidden_size, num_labels
        )

    def cache_identity(self) -> str:
        """Return the shape identity used by control-output caching."""
        return f"embedding_probe:{self.embedding_size}:{self.probe_hidden_size}:{self.num_labels}"

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        inputs_embeds = inputs_embeds.to(
            device=self.classifier.weight.device, dtype=self.classifier.weight.dtype
        )
        logits = self.classifier(self.projection(inputs_embeds))
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.task_type == "sequence_regression":
                loss = nn.functional.mse_loss(
                    logits.squeeze(-1), labels.to(logits.dtype)
                )
            else:
                loss = nn.functional.cross_entropy(logits, labels)
        return _HeadOutput(loss=loss, logits=logits)


def validate_native_linear_head(model: nn.Module) -> None:
    """Require a plain native classifier for an uncached ``torch_linear`` run."""
    native_head = getattr(model, "classifier", None)
    if not isinstance(native_head, nn.Linear):
        raise ValueError(
            "head_type='torch_linear' without embedding_cache requires the "
            "loaded model's native classifier to be a single torch.nn.Linear. "
            "Enable steps.prepare.embedding_cache.enabled=True to use a "
            "standalone linear probe with this model."
        )


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
        if key not in (_CLASSIFICATION_ONLY_PARAMETERS | _REGRESSION_ONLY_PARAMETERS)
        or key in allowed
    }


def has_classical_search_space(
    head_type: HeadType,
    search_space: dict[str, list[Any]],
    default_hyperparameters: dict[str, Any],
) -> bool:
    """Whether this head configuration selects among multiple candidates."""
    return bool(
        search_space
        or (_DEFAULT_SEARCH_SPACES.get(head_type, {}) and not default_hyperparameters)
    )


def sweep_classical_head(
    head_type: HeadType,
    task_type: str,
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
    search_space: dict[str, list[Any]],
    default_hyperparameters: dict[str, Any],
    metric_name: str,
    direction: Literal["maximize", "minimize"],
    search_pattern: Literal["grid", "random", "bohb"] = "bohb",
    n_trials: int | None = None,
    seed: int = 710019,
    average: str = "weighted",
) -> tuple[BaseEstimator, dict[str, Any], float]:
    """Fit and validate classical probe candidates, returning the best one.

    Candidates are selected exclusively on ``val_X``/``val_y``, and the winner
    is returned as fit on the training split alone -- matching the
    mlp/torch_linear paths, so head types stay comparable rather than
    differing in how much data they saw.

    ``grid`` evaluates every ``ParameterGrid`` combination. ``random`` samples
    up to ``n_trials`` combinations without replacement. ``bohb`` uses the
    shared Optuna TPE configuration; atomic sklearn fits cannot report
    intermediate results, so no pruning happens within a trial.
    """
    from modules.evaluate.src.tasks import get_task

    effective_search_space = _filter_classical_search_space(
        head_type,
        task_type,
        search_space,
    ) or (
        _DEFAULT_SEARCH_SPACES.get(head_type, {}) if not default_hyperparameters else {}
    )
    candidates: list[dict[str, Any]] = (
        list(ParameterGrid(effective_search_space))
        if effective_search_space
        else [default_hyperparameters]
    )
    candidates = [{**default_hyperparameters, **candidate} for candidate in candidates]

    if search_pattern == "random" and len(candidates) > 1:
        candidates = sample_candidates(candidates, n_trials or len(candidates), seed)

    best_estimator: BaseEstimator | None = None
    best_hparams: dict[str, Any] | None = None
    best_score = float("-inf") if direction == "maximize" else float("inf")
    is_better = (
        (lambda s: s > best_score)
        if direction == "maximize"
        else (lambda s: s < best_score)
    )

    def evaluate(hparams: dict[str, Any]) -> float:
        nonlocal best_estimator, best_hparams, best_score
        estimator = build_estimator(head_type, task_type, hparams, seed=seed)
        estimator.fit(train_X, train_y)
        predictions = estimator.predict(val_X)
        task = get_task(task_type)
        probabilities = (
            estimator.predict_proba(val_X)
            if hasattr(estimator, "predict_proba")
            else None
        )
        score = task.compute_metrics(
            predictions.tolist(),
            val_y.tolist(),
            [metric_name],
            average=average,
            probabilities=probabilities.tolist() if probabilities is not None else None,
        )[metric_name]
        logger.info(
            "head_type=%s hparams=%s val_%s=%.4f",
            head_type,
            hparams,
            metric_name,
            score,
        )
        if is_better(score):
            best_estimator, best_hparams, best_score = estimator, hparams, score
        return score

    if search_pattern == "bohb" and effective_search_space:

        def objective(trial: "optuna.Trial") -> float:
            hparams = {
                **default_hyperparameters,
                **{
                    name: trial.suggest_categorical(name, values)
                    for name, values in effective_search_space.items()
                },
            }
            return evaluate(hparams)

        trials = n_trials or len(candidates)
        study = create_optuna_study(direction, seed, trials)
        study.optimize(objective, n_trials=trials)
    else:
        for hparams in candidates:
            evaluate(hparams)

    assert best_hparams is not None and best_estimator is not None
    return best_estimator, best_hparams, best_score


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
) -> tuple[list[Any], list[Any], list[Any], list[Any] | None]:
    """Run a fitted estimator over a cached-embedding dataloader."""
    from modules.evaluate.src.tasks import get_task

    task = get_task(task_type)
    all_ids: list[Any] = []
    all_predictions: list[Any] = []
    all_labels: list[Any] = []
    all_probabilities: list[Any] | None = [] if collect_probabilities else None
    for batch in dataloader:
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
    return all_ids, all_predictions, all_labels, all_probabilities
