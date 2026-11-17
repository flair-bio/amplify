"""Prepare independently fitted control predictors without mutating primary inputs."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

from accelerate import Accelerator

from modules.evaluate.src.artifacts.checkpoints import save_predictor
from modules.evaluate.src.artifacts.workspace import EvaluationWorkspace
from modules.evaluate.src.config.shared import EmbeddingCacheConfig
from modules.evaluate.src.dataset.embeddings import build_embedding_dataloaders
from modules.evaluate.src.dataset.types import TrainingData
from modules.evaluate.src.model.controls import ControlPredictors, build_random_trunk
from modules.evaluate.src.model.factory import Predictor
from modules.evaluate.src.model.heads import EmbeddingProbeHead
from modules.evaluate.src.model.layerwise import LayerwiseProbeModel
from modules.evaluate.src.tasks import get_task
from modules.evaluate.src.trainers.fitting import fit_predictor
from modules.evaluate.src.trainers.layerwise import fit_layerwise_random_trunk

if TYPE_CHECKING:
    from collections.abc import Sequence

    from modules.evaluate.src.config.evaluation import TuneConfig
logger = logging.getLogger(__name__)


def prepare_control_predictors(
    config: TuneConfig,
    workspace: EvaluationWorkspace,
    predictor: Predictor,
    training: TrainingData,
    seed: int = 710019,
    accelerator: Accelerator | None = None,
    controls: Sequence[str] = (),
    *,
    fitted: bool = True,
) -> ControlPredictors:
    """Materialize requested controls; inference-only ablations stay in Predict."""
    control_predictors = ControlPredictors()
    if not controls:
        return control_predictors
    if accelerator is None:
        raise ValueError("Pass the application-owned accelerator explicitly.")
    if training.layerwise is not None:
        if {"random_trunk", "random_both"} & set(controls):
            inner = accelerator.unwrap_model(predictor.model)
            if not isinstance(inner, LayerwiseProbeModel):
                raise TypeError("Layerwise controls require a layerwise predictor.")
            control_predictors.random_trunk_model = fit_layerwise_random_trunk(
                config,
                workspace.config,
                predictor,
                training,
                inner,
                accelerator,
                seed,
                train_heads="random_trunk" in controls,
            )
        return control_predictors
    if fitted and training.embedding is not None and ("random_trunk" in controls):
        _fit_cached_random_trunk(
            config,
            workspace,
            predictor,
            training,
            control_predictors,
            seed,
            accelerator,
        )
    return control_predictors


def _fit_cached_random_trunk(
    config: TuneConfig,
    workspace: EvaluationWorkspace,
    predictor: Predictor,
    training: TrainingData,
    control_predictors: ControlPredictors,
    seed: int,
    accelerator: Accelerator,
) -> None:
    """Build cached random-trunk features and fit an independent control."""
    backend = training.embedding.backend if training.embedding is not None else None
    if backend is None:
        return
    if backend.token_train_dataloader is None or backend.cache is None:
        raise ValueError(
            "Cached random_trunk tuning requires retained train tokens and an embedding cache backend."
        )
    if (
        backend.model_id is None
        or backend.dataset_id is None
        or backend.max_length is None
    ):
        raise ValueError(
            "Cached random_trunk tuning requires model/dataset cache metadata."
        )
    task = get_task(workspace.task_type, workspace.num_labels)
    trunk = build_random_trunk(accelerator.unwrap_model(predictor.trunk), seed)
    splits = {
        "train": (backend.token_train_dataloader, backend.train_random_truncate),
        "validation": (
            backend.token_val_dataloader,
            backend.validation_random_truncate,
        ),
        **{
            name: (loader, training.refit_random_truncate_by_split.get(name, False))
            for name, loader in training.refit_token_dataloaders.items()
        },
    }
    loaders = build_embedding_dataloaders(
        model=trunk,
        splits=splits,
        accelerator=accelerator,
        cache=backend.cache,
        cache_config=EmbeddingCacheConfig(
            enabled=True,
            pooling=backend.pooling,
            layers=backend.layers,
            dtype=backend.dtype,
            autocast_dtype=backend.autocast_dtype,
            normalize_before_pooling=backend.normalize_before_pooling,
            cache_policy=backend.cache_policy,
        ),
        model_id=backend.model_id,
        dataset_id=backend.dataset_id,
        max_length=backend.max_length,
        seed=seed,
        batch_size=config.batch_size
        or getattr(training.train_dataloader, "batch_size", None)
        or getattr(
            getattr(training.train_dataloader, "batch_sampler", None),
            "batch_size",
            None,
        )
        or 128,
        extraction_batch_size=backend.extraction_batch_size,
        embedding_size=backend.hidden_size,
        label_dtype=task.label_dtype,
        is_ragged=task.is_ragged,
        length_grouped_sampling=backend.length_grouped_sampling,
    )
    control_refit = {}
    for name in training.refit_token_dataloaders:
        loader = loaders[name]
        if loader is None:
            raise ValueError(
                f"Could not build random-trunk embeddings for final-refit split '{name}'."
            )
        control_refit[name] = loader
    control_predictor = replace(
        predictor,
        model=EmbeddingProbeHead(
            backend.hidden_size,
            task.resolve_num_labels(workspace.num_labels, workspace.dataset_name),
            workspace.task_type,
            probe_hidden_size=config.head.hidden_size,
            probe_hidden_depth=config.head.hidden_depth,
        ).to(accelerator.device),
        trunk=trunk,
    )
    control_training = replace(
        training,
        train_dataloader=loaders["train"],
        val_dataloader=loaders["validation"],
        refit_dataloaders=control_refit,
        embedding=None,
    )
    control_workspace = workspace.control_workspace("random_trunk")
    control_config = config.model_copy(update={"variance_seeds": []})
    control_result = fit_predictor(
        workspace=control_workspace.config,
        predictor=control_predictor,
        training=control_training,
        config=control_config,
        accelerator=accelerator,
        seed=seed,
        mode="tune_probe",
    )
    control_predictor.model = control_result.model
    save_predictor(
        config, control_workspace, control_predictor, control_result, accelerator
    )
    control_predictors.random_trunk_model = control_result.model
    control_predictors.random_trunk_seed = control_result.seed
