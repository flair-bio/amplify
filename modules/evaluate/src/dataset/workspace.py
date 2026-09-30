from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from modules.evaluate.src.tasks import TaskType, available_tasks

__all__ = [
    "EvaluationWorkspace",
    "EvaluationWorkspaceConfig",
    "TaskType",
    "get_evaluation_workspace",
]


class EvaluationWorkspaceConfig(BaseModel):
    """Config for an EvaluationWorkspace, organizing files pertaining to a specific dataset and model."""

    model_config = ConfigDict(extra="forbid")

    dataset_name: str  # dataset name that serves as a mnemonic for the dataset being evaluated, e.g. "enzyme_catalytic_efficiency"
    dataset_repo_id: str | None = (
        None  # path to HuggingFace dataset repo e.g. biomap-research/enzyme_catalytic_efficiency
    )
    model_name: str  # model name that serves as a mnemonic for the model being evaluated, e.g. "amplify-120m"
    model_repo_id: str | None = (
        None  # path to HuggingFace model repo e.g. flair-bio/amplify-120m
    )
    task_type: TaskType
    target_type: Literal["categorical", "continuous"]
    label_format: Literal["scalar", "per_token", "contact_pairs"]
    num_labels: int | None = (
        None  # number of classes for *_classification tasks; unused (forced to 1) for sequence_regression
    )
    model_kwargs: dict[str, Any] = Field(
        default_factory=dict
    )  # extra task.build_model() kwargs, e.g. {"rank": 64} for contact_prediction
    base_path: str = "./eval"  # base path where the EvaluationWorkspace is stored
    tmp_path: str | None = None  # temporary path for intermediate files

    @field_validator("task_type")
    @classmethod
    def _validate_task_type(cls, value: str) -> str:
        if value not in available_tasks():
            raise ValueError(
                f"Unknown task_type '{value}'. Available: {available_tasks()}."
            )
        return value

    @model_validator(mode="after")
    def _validate_target_metadata(self) -> "EvaluationWorkspaceConfig":
        if self.target_type == "continuous":
            if self.label_format != "scalar" or self.num_labels is not None:
                raise ValueError(
                    "target_type='continuous' requires label_format='scalar' and "
                    "num_labels=null."
                )
        elif self.label_format in {"scalar", "per_token"}:
            if self.num_labels is None or self.num_labels <= 1:
                raise ValueError(
                    "Categorical scalar and per-token targets require num_labels > 1."
                )
        return self


class EvaluationWorkspace:
    """EvaluationWorkspace variant for evaluation pipelines. Includes additional paths for validation, test, predictions, and scores."""

    def __init__(
        self, config: EvaluationWorkspaceConfig, variant: str | None = None
    ) -> None:

        self.config = config
        self.variant = variant

        self.dataset_name = config.dataset_name
        self.dataset_repo_id = config.dataset_repo_id
        self.model_name = config.model_name
        self.model_repo_id = config.model_repo_id
        self.task_type = config.task_type
        self.num_labels = config.num_labels
        self.model_kwargs = config.model_kwargs
        self.base_path = config.base_path

        # If possible, enforce a model name that includes the repo it came from, for saving
        dataset_save_name = (
            self.dataset_repo_id.replace("/", "_")
            if self.dataset_repo_id
            else self.dataset_name.replace("/", "_")
        )
        model_save_name = (
            self.model_repo_id.replace("/", "_")
            if self.model_repo_id
            else self.model_name.replace("/", "_")
        )

        base = Path(config.base_path)
        workspace_root = base / model_save_name / dataset_save_name
        self.tmp_path = (
            Path(config.tmp_path) if config.tmp_path else workspace_root / "tmp"
        )
        # Disposable: safe to delete between runs, PrepareStep recomputes on miss.
        # Deliberately NOT nested under `variant`: a frozen trunk's cached
        # embeddings don't depend on head_type/mode, so every variant
        # sweep for this model+dataset can reuse the same cache.
        self.embeddings_path = self.tmp_path / "embeddings"

        # Per-condition outputs (trained model/head, its predictions, and its
        # own detailed scores) are nested under a `variant`-named
        # subdirectory so sweeping head_type/mode for the same
        # model+dataset doesn't overwrite a previous variant's results.
        # Direct workspace construction (used by isolated tests) keeps the
        # flat layout; pipeline runs always provide a variant.
        run_root = (
            workspace_root / "runs" / self.variant.replace("/", "_")
            if self.variant
            else workspace_root
        )
        self.model_path = run_root / "model"
        self.preds_path = run_root / "preds"
        self.stats_path = run_root / "stats"
        self.scores_path = run_root / "scores"

        # Consolidated Predict/Score settings + results sidecar (replaces
        # separate predict_config.json/scores.json/bootstrap_summary.json
        # files). Model checkpoint settings are deliberately NOT included
        # here: they stay beside the checkpoint in `model_path` (see
        # TuneStep._save_model) so it remains self-describing if copied out
        # on its own.
        self.run_manifest_path = run_root / "run_manifest.json"

        # Shared across every variant, unlike `scores_path` above: this is
        # where ScoreStep upserts `scores_long.parquet`, so comparing every
        # head_type/mode condition tried for this model+dataset never
        # requires globbing/concatenating multiple files by hand.
        self.comparison_path = workspace_root / "scores"

        self._all_paths = [
            self.stats_path,
            self.tmp_path,
            self.preds_path,
            self.scores_path,
            self.model_path,
            self.embeddings_path,
            self.comparison_path,
        ]

    def setup_directories(self) -> None:
        """Create all required directories on disk."""
        for path in self._all_paths:
            path.mkdir(parents=True, exist_ok=True)


def get_evaluation_workspace(
    config: EvaluationWorkspaceConfig, variant: str | None = None
) -> EvaluationWorkspace:
    return EvaluationWorkspace(config, variant=variant)
