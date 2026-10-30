"""Typed contracts shared by the generic evaluation lifecycle.

Tasks own semantics: preparing observations, choosing the predictor, and scoring.
Predictors fit on development views and predict evaluation views. The runner only
sequences these calls and holds no task-specific branches.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from modules.evaluate.src.config import EvaluationConfig
    from modules.evaluate.src.dataset.workspace import EvaluationWorkspace


@dataclass(frozen=True)
class RunContext:
    """Immutable inputs of one view (main or external) of one evaluation run.

    ``split`` labels the evaluation view: ``"test"`` for the main test split, or
    the external test set's name. ``workspace`` is the view's output workspace.
    """

    config: EvaluationConfig
    workspace: EvaluationWorkspace
    accelerator: Any
    dataset: Any = None
    split: str = "test"


@dataclass(frozen=True)
class FitSummary:
    """What a predictor reports after fitting. Opaque to the runner."""

    predictor: str
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FitOutcome:
    """A fit summary plus the fitted state that later views are built from."""

    summary: FitSummary
    fitted: Any


@dataclass(frozen=True)
class PredictionBundle:
    """Predictions for one branch of one evaluation view.

    A predictor may yield several branches for one view (for example one per
    layer, or per control condition). ``branch`` names it and ``workspace`` is the
    output location owned by that branch, so scoring and publication never reach
    back into the view's workspace.
    """

    predictor: str
    split: str
    branch: str
    workspace: EvaluationWorkspace
    payload: Any


@dataclass(frozen=True)
class ScoringResult:
    """Task-owned scores for one prediction bundle."""

    predictor: str
    split: str
    branch: str
    payload: Any


@runtime_checkable
class Predictor(Protocol):
    """Fits on development data and predicts evaluation views. Never scores."""

    name: str
    #: False when the fitted state has nothing to transfer to external views
    #: (e.g. evaluate-as-is models with no fitted probe); the runner then skips them.
    transfers_to_external: bool

    def fit(self, context: RunContext, prepared: Any) -> FitOutcome: ...

    def predict(
        self, context: RunContext, prepared: Any
    ) -> tuple[PredictionBundle, ...]: ...


@runtime_checkable
class TaskDefinition(Protocol):
    """Owns semantics: observation preparation, its predictor, and scoring."""

    name: str
    requires_single_process: bool

    def prepare(self, context: RunContext) -> Any: ...

    def prepare_external(self, context: RunContext, fitted: Any, external: Any) -> Any:
        """Prepare one external view.

        ``context`` is the MAIN view's context (its workspace carries the baseline
        and run configuration), not the external view's; the runner passes the
        external workspace only when it predicts and scores that view.
        """
        ...

    def predictor(self, context: RunContext) -> Predictor: ...

    def score(self, context: RunContext, bundle: PredictionBundle) -> ScoringResult: ...

    def finalize_view(
        self,
        context: RunContext,
        predictions: tuple[PredictionBundle, ...],
        scores: tuple[ScoringResult, ...],
    ) -> None:
        """Task-owned comparison across the branches of one view (main process only).

        Called once per view after every branch is scored, for example to compare
        layers on the same rows. Tasks with a single branch implement it as a no-op.
        """
        ...
