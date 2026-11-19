"""Backend-aware predictor checkpoint publication."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import torch
from accelerate import Accelerator

from modules.evaluate.src.artifacts.files import write_settings_json
from modules.evaluate.src.artifacts.results import FitResult
from modules.evaluate.src.artifacts.workspace import (
    EvaluationWorkspace,
)
from modules.evaluate.src.config import TuneConfig
from modules.evaluate.src.model.factory import Predictor
from modules.evaluate.src.model.heads import EmbeddingProbeHead
from modules.evaluate.src.model.layerwise import LayerwiseProbeModel


def save_predictor(
    config: TuneConfig,
    workspace: EvaluationWorkspace,
    predictor: Predictor,
    result: FitResult,
    accelerator: Accelerator,
) -> None:
    """Publish selected fitting artifacts while preserving collective policy."""
    from modules.evaluate.src.artifacts.files import write_json
    from modules.evaluate.src.artifacts.manifests import update_run_manifest

    if accelerator.is_main_process:
        if result.statistics:
            update_run_manifest(workspace.run_manifest_path, "fit", result.statistics)
    if result.serialization == "local":
        if config.save_best_model and accelerator.is_main_process:
            result.model.save_pretrained(workspace.model_path)
            if predictor.tokenizer is not None:
                predictor.tokenizer.save_pretrained(workspace.model_path)
            if result.checkpoint_settings is not None:
                write_json(
                    workspace.model_path / "tune_config.json",
                    result.checkpoint_settings,
                )
        accelerator.wait_for_everyone()
        return
    model = result.model
    unwrapped_model = (
        accelerator.unwrap_model(model) if isinstance(model, torch.nn.Module) else model
    )
    if isinstance(unwrapped_model, LayerwiseProbeModel):
        if config.save_best_model and accelerator.is_main_process:
            for position, layer in enumerate(unwrapped_model.layers):
                layer_workspace = EvaluationWorkspace(
                    workspace.config,
                    variant=f"{workspace.variant}-layer_{layer}",
                    seed=workspace.seed,
                )
                layer_workspace.model_path.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "state_dict": unwrapped_model.heads[str(layer)].state_dict(),
                        "feature_mean": unwrapped_model.feature_mean[position].cpu(),
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
                        "tune_config": config.model_dump(mode="json"),
                    },
                    layer_workspace.model_path / "probe_head.pt",
                )
        accelerator.wait_for_everyone()
        return
    if isinstance(model, torch.nn.Module):
        if not config.save_best_model:
            return
        workspace.model_path.mkdir(parents=True, exist_ok=True)
        state_dict = accelerator.get_state_dict(model)
        if isinstance(unwrapped_model, EmbeddingProbeHead):
            if accelerator.is_main_process:
                torch.save(state_dict, workspace.model_path / "probe_head.pt")
                _write_settings(
                    config,
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
            if predictor.tokenizer is None:
                raise ValueError("Saving a pretrained model requires its tokenizer.")
            unwrapped_model.save_pretrained(
                workspace.model_path,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=state_dict,
            )
            if accelerator.is_main_process:
                predictor.tokenizer.save_pretrained(workspace.model_path)
                _write_settings(
                    config, workspace.model_path / "tune_config.json", result
                )
        accelerator.wait_for_everyone()
        return
    if config.save_best_model:
        workspace.model_path.mkdir(parents=True, exist_ok=True)
        if accelerator.is_main_process:
            joblib.dump(model, workspace.model_path / "head.joblib")
            _write_settings(config, workspace.model_path / "tune_config.json", result)
    accelerator.wait_for_everyone()


def _write_settings(
    config: TuneConfig, path: Path, result: FitResult, **extra: Any
) -> None:
    write_settings_json(
        path,
        result.seed,
        config.model_dump(mode="json"),
        best_hyperparameters=result.selected_hyperparameters,
        final_fit_split=result.final_fit_split,
        selection_metric=result.selection_metric,
        selection_score=result.selection_score,
        seed_variance=result.seed_variance,
        **extra,
    )
