"""Final refits and the random-trunk control for cached probes (extracted from TuneStep).

Both functions receive a ``TorchProbeTrainer`` (and, for the random-trunk
control, a callable that runs the same probe protocol on the control head) so
they never depend on the legacy pipeline.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection
from dataclasses import replace
from typing import Any

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from modules.evaluate.src.artifacts.feature_cache import EmbeddingCacheConfig
from modules.evaluate.src.dataset.workspace import EvaluationWorkspace
from modules.evaluate.src.model.heads import EmbeddingProbeHead
from modules.evaluate.src.predictors.embeddings import (
    EmbeddingDataset,
    build_embedding_dataloaders,
    find_embedding_dataset,
)
from modules.evaluate.src.schemas.artifacts import PreparedArtifacts
from modules.evaluate.src.tasks import get_task
from modules.evaluate.src.tasks.token_classification import IGNORE_INDEX
from modules.evaluate.src.training.torch_trainer import TorchProbeTrainer
from modules.evaluate.src.utils.controls import build_random_trunk

logger = logging.getLogger(__name__)

RandomTrunkRunner = Callable[
    [EvaluationWorkspace, PreparedArtifacts, Accelerator, int], Any
]


def refit_cached_probe(
    trainer: TorchProbeTrainer,
    workspace: EvaluationWorkspace,
    prepared: PreparedArtifacts,
    accelerator: Accelerator,
    hparams: dict[str, Any],
    seed: int,
) -> None:
    """Fit a fresh selected head on development data, retaining split loaders."""
    train = find_embedding_dataset(prepared.train_dataloader)
    validation = find_embedding_dataset(prepared.val_dataloader)
    selected = accelerator.unwrap_model(prepared.model)
    assert train is not None and validation is not None
    assert isinstance(selected, EmbeddingProbeHead)
    development_splits = [train, validation]
    for name, loader in prepared.refit_dataloaders.items():
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
    if get_task(workspace.task_type).masks_ignore_index:
        row_mask = (
            torch.as_tensor(development.labels, device=development.embeddings.device)
            != IGNORE_INDEX
        )
    fresh.fit_standardization(
        development.embeddings,
        enabled=trainer.config.standardize_features,
        row_mask=row_mask,
        show_progress=accelerator.is_main_process,
    )
    loader = DataLoader(
        development,
        batch_size=hparams.get("batch_size")
        or trainer.cached_batch_size(prepared.train_dataloader),
    )
    refit_prepared = replace(
        prepared,
        model=fresh,
        train_dataloader=loader,
        val_dataloader=None,
        test_dataloader=None,
    )
    logger.info(
        "Refitting fresh cached probe on %d rows from %s for %d epochs.",
        len(development),
        trainer.final_fit_split(prepared),
        hparams["max_epochs"],
    )
    refit_trainer = TorchProbeTrainer(
        trainer.config.model_copy(update={"early_stopping_patience": None})
    )
    torch.manual_seed(seed)
    refit_trainer.train(workspace, refit_prepared, accelerator, hparams)
    prepared.model = fresh


def fit_cached_random_trunk(
    trainer: TorchProbeTrainer,
    workspace: EvaluationWorkspace,
    prepared: PreparedArtifacts,
    accelerator: Accelerator,
    seed: int,
    controls: Collection[str],
    run_baseline: RandomTrunkRunner,
) -> None:
    """Select and refit a separate probe with the same budget on random features.

    ``run_baseline`` runs the configured tuning protocol on the control head
    (with variance seeds disabled) so the control is selected under the same
    budget as the real probe.
    """
    backend = prepared.embedding_backend
    if "random_trunk" not in controls or backend is None:
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
    trunk = build_random_trunk(accelerator.unwrap_model(backend.trunk), seed)
    logger.info(
        "Selecting a separate random-trunk probe with the configured tuning protocol."
    )
    random_trunk_splits = {
        "train": (
            backend.token_train_dataloader,
            backend.train_random_truncate,
        ),
        "validation": (
            backend.token_val_dataloader,
            backend.validation_random_truncate,
        ),
        **{
            name: (
                loader,
                prepared.refit_random_truncate_by_split.get(name, False),
            )
            for name, loader in prepared.refit_token_dataloaders.items()
        },
    }
    loaders = build_embedding_dataloaders(
        model=trunk,
        splits=random_trunk_splits,
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
        batch_size=trainer.cached_batch_size(prepared.train_dataloader),
        extraction_batch_size=backend.extraction_batch_size,
        embedding_size=backend.hidden_size,
        label_dtype=get_task(workspace.task_type).label_dtype,
        is_ragged=get_task(workspace.task_type).is_ragged,
        length_grouped_sampling=backend.length_grouped_sampling,
    )
    torch.manual_seed(seed)
    random_refit_dataloaders = {}
    for name in prepared.refit_token_dataloaders:
        loader = loaders[name]
        if loader is None:
            raise ValueError(
                f"Could not build random-trunk embeddings for final-refit split "
                f"'{name}'."
            )
        random_refit_dataloaders[name] = loader
    baseline = replace(
        prepared,
        model=EmbeddingProbeHead(
            backend.hidden_size,
            get_task(workspace.task_type).resolve_num_labels(
                workspace.num_labels, workspace.dataset_name
            ),
            workspace.task_type,
        ).to(accelerator.device),
        train_dataloader=loaders["train"],
        val_dataloader=loaders["validation"],
        test_dataloader=None,
        refit_dataloaders=random_refit_dataloaders,
        embedding_backend=None,
        random_trunk_model=None,
        random_trunk_seed=None,
    )
    baseline_workspace = workspace.control_workspace("random_trunk")
    baseline_workspace.setup_directories()
    run_baseline(baseline_workspace, baseline, accelerator, seed)
    prepared.random_trunk_model = baseline.model
    prepared.random_trunk_seed = seed
