"""Zero-shot masked-LM evaluation (``steps.mode: evaluate_as_is``) on the generic runner.

Calibration and pseudo-perplexity each declare their observations, masking and
metrics in their own TaskHandler (``tasks/calibration.py``,
``tasks/pseudo_perplexity.py``). This lifecycle owner runs whichever of those
task types is configured: it prepares the pretrained masked LM through the
shared probe preparation, predicts with :class:`PretrainedPredictor`, and scores
through the task's registered metrics. It never fits a head.
"""

from __future__ import annotations

from typing import Any

from modules.evaluate.src.config import EvaluationConfig
from modules.evaluate.src.evaluation.contracts import (
    PredictionBundle,
    Predictor,
    RunContext,
    ScoringResult,
)
from modules.evaluate.src.predictors.pretrained import PretrainedPredictor
from modules.evaluate.src.predictors.probe_preparation import prepare_probe_artifacts
from modules.evaluate.src.scoring.run import score_bundle

#: Task types whose only supported mode is evaluate-as-is on a pretrained MLM.
MASKED_LM_TASK_TYPES = frozenset({"calibration", "pseudo_perplexity"})


class MaskedLMTask:
    """Pretrained masked-LM evaluation for the configured zero-shot task type."""

    name = "masked_lm"
    requires_single_process = False

    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config

    def prepare(self, context: RunContext) -> Any:
        return prepare_probe_artifacts(
            self.config.steps.prepare,
            context.workspace,
            seed=self.config.seed,
            accelerator=context.accelerator,
            layerwise_config=self.config.steps.layerwise,
            dataset=context.dataset,
            baseline_names=self.config.baselines.names or None,
        )

    def prepare_external(self, context: RunContext, fitted: Any, external: Any) -> Any:
        # Never reached: the pretrained predictor transfers nothing to external
        # views, so the runner skips them before preparing.
        raise RuntimeError("masked_lm tasks have no external views to prepare.")

    def predictor(self, context: RunContext) -> Predictor:
        return PretrainedPredictor(
            predict=self.config.steps.predict, seed=self.config.seed
        )

    def finalize_view(
        self,
        context: RunContext,
        predictions: tuple[PredictionBundle, ...],
        scores: tuple[ScoringResult, ...],
    ) -> None:
        """Nothing to compare: the pretrained model has one branch."""

    def score(self, context: RunContext, bundle: PredictionBundle) -> ScoringResult:
        return score_bundle(self.config.steps.score, bundle, seed=self.config.seed)
