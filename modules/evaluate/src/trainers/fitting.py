"""Function-based dispatch for predictor selection and fitting."""

from __future__ import annotations

import logging
from typing import Any, Literal

import torch
from accelerate import Accelerator
from torch import nn

from modules.evaluate.src.artifacts.results import FitResult
from modules.evaluate.src.config.evaluation import TuneConfig
from modules.evaluate.src.config.shared import (
    CLASSICAL_HEAD_TYPES,
    EvaluationWorkspaceConfig,
)
from modules.evaluate.src.dataset.embeddings import find_embedding_dataset
from modules.evaluate.src.dataset.types import TrainingData
from modules.evaluate.src.model.factory import Predictor
from modules.evaluate.src.model.heads import (
    EmbeddingProbeHead,
    validate_native_linear_head,
)
from modules.evaluate.src.model.layerwise import LayerwiseProbeModel
from modules.evaluate.src.tasks.token_classification import IGNORE_INDEX
from modules.evaluate.src.trainers.classical import _run_classical_head
from modules.evaluate.src.trainers.layerwise import _run_layerwise
from modules.evaluate.src.trainers.refit import _final_fit_split, _finalize_torch_probe
from modules.evaluate.src.trainers.selection import _search_hyperparameters
from modules.evaluate.src.trainers.selection_helpers import (
    _log_seed_variance,
    checkpoint_state,
    configure_trainable_params,
    load_checkpoint_state,
)
from modules.evaluate.src.trainers.torch import (
    evaluate_torch_validation,
    resolve_selection_metric,
    train_torch,
)

logger = logging.getLogger(__name__)


def _fit_predictor(
    config: TuneConfig,
    train_trunk: bool,
    _fit_result: FitResult,
    workspace: EvaluationWorkspaceConfig,
    predictor: Predictor,
    training: TrainingData,
    seed: int = 710019,
    accelerator: Accelerator | None = None,
) -> Predictor:
    """Fit the primary probe, and the trunk in finetune mode."""
    logger.info(
        "Running predictor fitting (%s)...", "finetune" if train_trunk else "tune_probe"
    )
    if accelerator is None:
        raise ValueError("Pass the application-owned accelerator explicitly.")
    if training.train_dataloader is None:
        raise ValueError(
            "Predictor fitting requires a train_dataloader, but the Prepare step did not produce one (no 'train' split in the dataset)."
        )
    search = config.hyperparameter_search
    if (
        search.enabled
        and (
            search.search_space.hidden_size is not None
            or search.search_space.hidden_depth is not None
            or search.search_space.batch_size is not None
        )
        and (
            train_trunk
            or config.head.head_type != "mlp"
            or (not isinstance(predictor.model, EmbeddingProbeHead))
            or (find_embedding_dataset(training.train_dataloader) is None)
        )
    ):
        dimension = (
            "hidden_size"
            if search.search_space.hidden_size is not None
            else "hidden_depth"
            if search.search_space.hidden_depth is not None
            else "batch_size"
        )
        raise ValueError(
            f"{dimension} search requires a cached MLP probe in tune_probe mode."
        )
    if config.refit_on_train_validation:
        if (
            train_trunk
            or isinstance(predictor.model, LayerwiseProbeModel)
            or find_embedding_dataset(training.train_dataloader) is None
            or (
                config.head.head_type not in CLASSICAL_HEAD_TYPES
                and (not isinstance(predictor.model, EmbeddingProbeHead))
            )
        ):
            raise ValueError(
                "refit_on_train_validation requires an ordinary cached probe in tune_probe mode; live/native and layerwise refit are not supported."
            )
        for name, loader in (
            ("train", training.train_dataloader),
            ("validation", training.val_dataloader),
        ):
            dataset = find_embedding_dataset(loader)
            if dataset is None or not len(dataset):
                raise ValueError(
                    f"refit_on_train_validation requires nonempty train and validation embedding datasets; missing or empty {name} split."
                )
        missing_refit_splits = set(training.refit_token_dataloaders) - set(
            training.refit_dataloaders
        )
        if missing_refit_splits:
            raise ValueError(
                f"Final-refit splits were not cached as embeddings: {sorted(missing_refit_splits)}."
            )
        for name, loader in training.refit_dataloaders.items():
            dataset = find_embedding_dataset(loader)
            if dataset is None or not len(dataset):
                raise ValueError(
                    f"refit_on_train_validation requires nonempty final-refit embedding datasets; missing or empty {name} split."
                )
    if isinstance(predictor.model, LayerwiseProbeModel):
        return _run_layerwise(
            config, _fit_result, workspace, predictor, training, accelerator, seed
        )
    if config.head.head_type in CLASSICAL_HEAD_TYPES:
        _run_classical_head(
            config, _fit_result, workspace, predictor, training, seed, accelerator
        )
        return predictor
    if not isinstance(predictor.model, nn.Module):
        raise TypeError("Torch probe tuning requires a torch.nn.Module model.")
    if config.head.head_type == "mlp" and isinstance(
        predictor.model, EmbeddingProbeHead
    ):
        predictor.model = EmbeddingProbeHead(
            predictor.model.embedding_size,
            predictor.model.num_labels,
            predictor.model.task_type,
            probe_hidden_size=config.head.hidden_size,
            probe_hidden_depth=config.head.hidden_depth,
        ).to(accelerator.device)
    model = predictor.model
    embedding_dataset = find_embedding_dataset(training.train_dataloader)
    cached_probe = embedding_dataset is not None
    if embedding_dataset is not None and isinstance(model, EmbeddingProbeHead):
        features = embedding_dataset.embeddings
        row_mask = None
        if config.standardize_features:
            if workspace.task_type == "token_classification":
                row_mask = torch.as_tensor(embedding_dataset.labels) != IGNORE_INDEX
                num_rows = int(row_mask.sum().item())
            else:
                num_rows = features.shape[0]
            logger.info(
                "Standardizing cached probe features using %d training rows.", num_rows
            )
        model.fit_standardization(
            features,
            enabled=config.standardize_features,
            row_mask=row_mask,
            show_progress=accelerator.is_main_process,
        )
    if not cached_probe and (not train_trunk):
        logger.warning(
            "Embedding cache is unavailable; tuning head_type='%s' without it will recompute frozen-trunk representations during training.",
            config.head.head_type,
        )
    if config.head.head_type == "torch_linear" and (not cached_probe):
        validate_native_linear_head(model)
    distributed_training = not cached_probe
    configure_trainable_params(model, train_trunk)
    if distributed_training:
        model = accelerator.prepare(model)
        predictor.model = model
    if config.hyperparameter_search.enabled:
        best_hparams = _search_hyperparameters(
            config,
            train_trunk,
            _fit_result,
            workspace,
            predictor,
            training,
            accelerator,
            seed,
        )
    else:
        best_hparams = {
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "max_epochs": config.max_epochs,
        }
        if not config.refit_on_train_validation:
            train_torch(
                config,
                train_trunk,
                workspace,
                predictor,
                training,
                accelerator,
                best_hparams,
            )
    _fit_result.selected_hyperparameters = dict(best_hparams)
    if config.variance_seeds and cached_probe:
        _report_torch_seed_variance(
            config,
            train_trunk,
            _fit_result,
            workspace,
            predictor,
            training,
            accelerator,
            best_hparams,
        )
    _finalize_torch_probe(
        config,
        train_trunk,
        workspace,
        predictor,
        training,
        accelerator,
        best_hparams,
        seed,
    )
    return predictor


def _report_torch_seed_variance(
    config: TuneConfig,
    train_trunk: bool,
    _fit_result: FitResult,
    workspace: EvaluationWorkspaceConfig,
    predictor: Predictor,
    training: TrainingData,
    accelerator: Accelerator,
    best_hparams: dict[str, Any],
) -> None:
    """Log validation-metric mean/std across ``variance_seeds``, so a
    single training seed's score isn't mistaken for run-to-run variance.
    Diagnostic only: restores the canonical model before returning.
    """
    if training.val_dataloader is None:
        return
    model = accelerator.unwrap_model(predictor.model)
    canonical_state = checkpoint_state(model)
    metric_name = resolve_selection_metric(config, workspace)
    scores = []
    for extra_seed in config.variance_seeds:
        torch.manual_seed(extra_seed)
        for module in model.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        train_torch(
            config,
            train_trunk,
            workspace,
            predictor,
            training,
            accelerator,
            best_hparams,
        )
        scores.append(
            evaluate_torch_validation(
                config,
                workspace,
                predictor,
                training,
                accelerator,
                metric_name,
                batch_size=best_hparams.get("batch_size"),
            )
        )
    load_checkpoint_state(model, canonical_state)
    _fit_result.seed_variance[metric_name] = scores
    _log_seed_variance(metric_name, scores)


def fit_predictor(
    *,
    workspace: EvaluationWorkspaceConfig,
    predictor: Predictor,
    training: TrainingData,
    config: TuneConfig,
    accelerator: Accelerator,
    seed: int,
    mode: Literal["tune_probe", "finetune"] = "tune_probe",
) -> FitResult:
    """Select and fit a predictor; return its model and selection provenance."""
    result = FitResult(model=predictor.model, seed=seed)
    _fit_predictor(
        config,
        mode == "finetune",
        result,
        workspace,
        predictor,
        training,
        seed=seed,
        accelerator=accelerator,
    )
    result.model = predictor.model
    result.final_fit_split = _final_fit_split(config, predictor, training)
    return result
