"""Stage boundary for predictor fitting."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import joblib
import torch
from accelerate import Accelerator

from modules.evaluate.src.contracts.artifacts import FitResult, PreparedArtifacts
from modules.evaluate.src.contracts.config import (
    TuneConfig,
)
from modules.evaluate.src.dataset.workspace import (
    EvaluationWorkspace,
    get_evaluation_workspace,
)
from modules.evaluate.src.model.heads import EmbeddingProbeHead
from modules.evaluate.src.model.layerwise import LayerwiseProbeModel
from modules.evaluate.src.tasks import get_task
from modules.evaluate.src.utils.io import write_settings_json

logger = logging.getLogger(__name__)


class TuneStep:
    """Validate fit inputs, select the fitting service, and hand off its result."""

    def __init__(
        self,
        config: TuneConfig,
        mode: Literal["tune_probe", "finetune"] = "tune_probe",
    ) -> None:
        self.config = config
        self.mode = mode

    def run(
        self,
        workspace: EvaluationWorkspace,
        prepared: PreparedArtifacts,
        seed: int = 710019,
        accelerator: Accelerator | None = None,
    ) -> PreparedArtifacts:
        """Fit the primary predictor and return the updated prepared payload."""
        if accelerator is None:
            accelerator = Accelerator(
                gradient_accumulation_steps=self.config.accumulation_steps
            )

        task = get_task(workspace.task_type, workspace.num_labels)
        if prepared.input_identity:
            workspace.validate_experiment("inputs", prepared.input_identity)
        result = task.fit(
            prepared,
            self.config,
            workspace.config,
            accelerator,
            seed=seed,
            mode=self.mode,
        )
        prepared.model = result.model
        self.publish_fit(workspace, prepared, result, accelerator)
        return prepared

    def publish_fit(
        self,
        workspace: EvaluationWorkspace,
        prepared: PreparedArtifacts,
        result: FitResult,
        accelerator: Accelerator,
    ) -> None:
        """Publish selected fitting artifacts while preserving collective policy."""
        from modules.evaluate.src.utils.io import write_json

        if accelerator.is_main_process:
            for filename, settings in result.statistics.items():
                write_json(workspace.stats_path / filename, settings)
        if result.serialization == "local":
            if self.config.save_best_model and accelerator.is_main_process:
                result.model.save_pretrained(workspace.model_path)
                if prepared.tokenizer is not None:
                    prepared.tokenizer.save_pretrained(workspace.model_path)
                if result.checkpoint_settings is not None:
                    write_json(
                        workspace.model_path / "tune_config.json",
                        result.checkpoint_settings,
                    )
            accelerator.wait_for_everyone()
            return
        model = result.model
        unwrapped_model = (
            accelerator.unwrap_model(model)
            if isinstance(model, torch.nn.Module)
            else model
        )
        if isinstance(unwrapped_model, LayerwiseProbeModel):
            if self.config.save_best_model and accelerator.is_main_process:
                for position, layer in enumerate(unwrapped_model.layers):
                    layer_workspace = get_evaluation_workspace(
                        workspace.config,
                        variant=f"{workspace.variant}-layer_{layer}",
                        seed=workspace.seed,
                    )
                    layer_workspace.model_path.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "state_dict": unwrapped_model.heads[
                                str(layer)
                            ].state_dict(),
                            "feature_mean": unwrapped_model.feature_mean[
                                position
                            ].cpu(),
                            "feature_std": unwrapped_model.feature_std[position].cpu(),
                            "target_mean": unwrapped_model.target_mean.cpu(),
                            "target_std": unwrapped_model.target_std.cpu(),
                            "selection": result.layer_selections.get(layer),
                            "layer": layer,
                            "model_id": workspace.model_repo_id or workspace.model_name,
                            "task_type": workspace.task_type,
                            "seed": result.seed,
                            "num_labels": unwrapped_model.num_labels,
                            "layerwise_config": unwrapped_model.config.model_dump(
                                mode="json"
                            ),
                            "tune_config": self.config.model_dump(mode="json"),
                        },
                        layer_workspace.model_path / "probe_head.pt",
                    )
            accelerator.wait_for_everyone()
            return

        if isinstance(model, torch.nn.Module):
            if not self.config.save_best_model:
                return
            workspace.model_path.mkdir(parents=True, exist_ok=True)
            state_dict = accelerator.get_state_dict(model)
            if isinstance(unwrapped_model, EmbeddingProbeHead):
                if accelerator.is_main_process:
                    torch.save(state_dict, workspace.model_path / "probe_head.pt")
                    self._write_settings(
                        workspace.model_path / "probe_head_config.json",
                        result,
                        embedding_size=unwrapped_model.embedding_size,
                        probe_hidden_size=unwrapped_model.probe_hidden_size,
                        probe_hidden_depth=unwrapped_model.probe_hidden_depth,
                        standardize_features=unwrapped_model.standardize_features,
                        num_labels=unwrapped_model.num_labels,
                        task_type=unwrapped_model.task_type,
                    )
            else:
                if prepared.tokenizer is None:
                    raise ValueError(
                        "Saving a pretrained model requires its tokenizer."
                    )
                unwrapped_model.save_pretrained(
                    workspace.model_path,
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save,
                    state_dict=state_dict,
                )
                if accelerator.is_main_process:
                    prepared.tokenizer.save_pretrained(workspace.model_path)
                    self._write_settings(
                        workspace.model_path / "tune_config.json",
                        result,
                    )
            accelerator.wait_for_everyone()
            return

        if self.config.save_best_model:
            workspace.model_path.mkdir(parents=True, exist_ok=True)
            if accelerator.is_main_process:
                joblib.dump(model, workspace.model_path / "head.joblib")
                self._write_settings(
                    workspace.model_path / "tune_config.json",
                    result,
                )
        # Classical fits are independently materialized on all ranks and used
        # a collective barrier before returning, whether or not saving is on.
        accelerator.wait_for_everyone()

    def _write_settings(
        self,
        path: Path,
        result: FitResult,
        **extra: Any,
    ) -> None:
        write_settings_json(
            path,
            result.seed,
            self.config.model_dump(mode="json"),
            best_hyperparameters=result.selected_hyperparameters,
            final_fit_split=result.final_fit_split,
            selection_metric=result.selection_metric,
            selection_score=result.selection_score,
            seed_variance=result.seed_variance,
            **extra,
        )
