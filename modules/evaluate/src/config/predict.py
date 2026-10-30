"""Configuration for trained, layerwise, and control prediction."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from modules.evaluate.src.utils.controls import ControlCondition


class PredictConfig(BaseModel):
    """Config. for the predict step in evaluation."""

    model_config = ConfigDict(extra="forbid")

    embedding_batch_size: int | None = Field(
        None,
        gt=0,
        description=(
            "Cached-feature rows per prediction batch, including controls. "
            "Token caches count residues, not proteins. Null inherits the "
            "prepared loader size; live trunk inference is unchanged."
        ),
    )
    collect_probabilities: bool = Field(
        False,
        description=(
            "Whether to collect per-example per-class probabilities "
            "for the trained model and any model-based controls. Required "
            "if ScoreConfig.metrics includes 'roc_auc'. When using the "
            "top-level EvaluationConfig, this is auto-enabled when needed."
        ),
    )
    predictions_filename: str = "predictions.parquet"
    persist_to_disk: bool = Field(
        True,
        description=(
            "Write predictions and the predict manifest to disk. Disable for "
            "fast in-process evaluation when ScoreStep receives this step's "
            "PredictOutput directly."
        ),
    )
    store_sequences: bool = Field(
        False,
        description=(
            "Whether to decode and persist input sequences alongside primary "
            "predictions. Kept off by default to reduce prediction-time "
            "memory use and parquet size. This only affects the main "
            "predictions table; the scrambled_sequences control still stores "
            "its transformed sequences regardless."
        ),
    )

    # Control conditions to compute predictions for, so ScoreStep can compare
    # the trained model against them:
    # - "untrained": the pretrained trunk with a freshly-initialized head,
    #   i.e. the model as it was before fine-tuning.
    # - "random_trunk": independently tuned/refitted probe on random features
    #   for cached probes; a trunk-replacement ablation for live/native runs.
    # - "random_head"/"random_both": the trained model with its head/both
    #   randomly re-initialized, without retraining.
    # - "scrambled_labels": no inference needed -- ScoreStep derives this
    #   directly from the trained model's own predictions/labels.
    # - "scrambled_sequences": the trained model's predictions on inputs
    #   whose token order has been shuffled (sequence-level tasks only).
    # - "majority_class": a constant class predictor selected from the
    #   training split's class frequencies (a simple non-model baseline).
    # - "train_mean": a constant regression predictor equal to the
    #   training split's label mean (the regression counterpart to
    #   "majority_class").
    controls: list[ControlCondition] = Field(default_factory=lambda: ["random_trunk"])
