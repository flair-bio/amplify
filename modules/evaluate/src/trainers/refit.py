"""Final refitting on explicitly permitted development splits."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

import torch
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader

from modules.evaluate.src.config.evaluation import TuneConfig
from modules.evaluate.src.config.shared import EvaluationWorkspaceConfig
from modules.evaluate.src.dataset.embeddings import (
    EmbeddingDataset,
    find_embedding_dataset,
)
from modules.evaluate.src.dataset.types import TrainingData
from modules.evaluate.src.model.factory import Predictor
from modules.evaluate.src.model.heads import EmbeddingProbeHead
from modules.evaluate.src.tasks.token_classification import IGNORE_INDEX
from modules.evaluate.src.trainers.torch import _cached_batch_size, train_torch

logger = logging.getLogger(__name__)


def _refit_cached_probe(
    config: TuneConfig,
    train_trunk: bool,
    workspace: EvaluationWorkspaceConfig,
    predictor: Predictor,
    training: TrainingData,
    accelerator: Accelerator,
    hparams: dict[str, Any],
    seed: int,
) -> None:
    """Fit a fresh selected head on development data, retaining split loaders."""
    train = find_embedding_dataset(training.train_dataloader)
    validation = find_embedding_dataset(training.val_dataloader)
    selected = accelerator.unwrap_model(predictor.model)
    assert train is not None and validation is not None
    assert isinstance(selected, EmbeddingProbeHead)
    development_splits = [train, validation]
    for name, loader in training.refit_dataloaders.items():
        extra = find_embedding_dataset(loader)
        if extra is None:
            raise ValueError(f"Final-refit split '{name}' has no cached embeddings.")
        development_splits.append(extra)
    development = EmbeddingDataset(
        torch.cat([split.embeddings for split in development_splits]),
        [identifier for split in development_splits for identifier in split.ids],
        torch.cat([torch.as_tensor(split.labels) for split in development_splits]),
    )
    torch.manual_seed(seed)
    fresh = EmbeddingProbeHead(
        selected.embedding_size,
        selected.num_labels,
        selected.task_type,
        probe_hidden_size=selected.probe_hidden_size,
        probe_hidden_depth=selected.probe_hidden_depth,
    ).to(selected.classifier.weight)
    row_mask = None
    if workspace.task_type == "token_classification":
        row_mask = (
            torch.as_tensor(development.labels, device=development.embeddings.device)
            != IGNORE_INDEX
        )
    fresh.fit_standardization(
        development.embeddings,
        enabled=config.standardize_features,
        row_mask=row_mask,
        show_progress=accelerator.is_main_process,
    )
    loader = DataLoader(
        development,
        batch_size=hparams.get("batch_size")
        or _cached_batch_size(config, training.train_dataloader),
    )
    refit_predictor = replace(predictor, model=fresh)
    refit_training = replace(training, train_dataloader=loader, val_dataloader=None)
    logger.info(
        "Refitting fresh cached probe on %d rows from %s for %d epochs.",
        len(development),
        _final_fit_split(config, predictor, training),
        hparams["max_epochs"],
    )
    torch.manual_seed(seed)
    train_torch(
        config.model_copy(update={"early_stopping_patience": None}),
        False,
        workspace,
        refit_predictor,
        refit_training,
        accelerator,
        hparams,
    )
    predictor.model = fresh


def _finalize_torch_probe(
    config: TuneConfig,
    train_trunk: bool,
    workspace: EvaluationWorkspaceConfig,
    predictor: Predictor,
    training: TrainingData,
    accelerator: Accelerator,
    best_hparams: dict[str, Any],
    seed: int,
) -> None:
    """Apply the shared selected-model refit, evaluation, and persistence handoff."""
    if config.refit_on_train_validation:
        _refit_cached_probe(
            config,
            train_trunk,
            workspace,
            predictor,
            training,
            accelerator,
            best_hparams,
            seed,
        )
    model = predictor.model
    if not isinstance(model, nn.Module):
        raise TypeError("Torch probe finalization requires a torch model.")
    model.eval()


def _final_fit_split(
    config: TuneConfig, predictor: Predictor, training: TrainingData
) -> str:
    if not config.refit_on_train_validation:
        return "train"
    return "+".join(["train", "validation", *training.refit_dataloaders.keys()])
