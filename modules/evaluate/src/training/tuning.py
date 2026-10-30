"""Torch probe tuning protocol: the single implementation behind ``TuneStep``.

``fit_torch_probe`` runs the cached-probe / live-head protocol that used to live
inline in ``steps/tune.py``: optional architecture/hyperparameter search (or one
configured training run), optional train/validation seed variance, optional
final refit on development data, model persistence, and the mandatory
random-trunk control. Every heavy operation delegates to the extracted
``TorchProbeTrainer``, ``search_hyperparameters``, and ``training.refit`` modules.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection
from typing import Any

import torch
from accelerate import Accelerator
from torch import nn

from modules.evaluate.src.dataset.workspace import EvaluationWorkspace
from modules.evaluate.src.model.heads import (
    CLASSICAL_HEAD_TYPES,
    EmbeddingProbeHead,
    validate_native_linear_head,
)
from modules.evaluate.src.model.layerwise import LayerwiseProbeModel
from modules.evaluate.src.predictors.embeddings import find_embedding_dataset
from modules.evaluate.src.schemas.artifacts import PreparedArtifacts
from modules.evaluate.src.tasks import get_task
from modules.evaluate.src.tasks.token_classification import IGNORE_INDEX
from modules.evaluate.src.training.classical import run_classical_head
from modules.evaluate.src.training.layerwise import run_layerwise
from modules.evaluate.src.training.refit import (
    fit_cached_random_trunk,
    refit_cached_probe,
)
from modules.evaluate.src.training.search_loop import search_hyperparameters
from modules.evaluate.src.training.torch_trainer import TorchProbeTrainer
from modules.evaluate.src.utils.hpo import configure_trainable_params

logger = logging.getLogger(__name__)

# (workspace, prepared, accelerator, seed) -> None; runs the same tuning protocol on the control head.
ControlRunner = Callable[
    [EvaluationWorkspace, PreparedArtifacts, Accelerator, int], Any
]


def fit_torch_probe(
    trainer: TorchProbeTrainer,
    workspace: EvaluationWorkspace,
    prepared: PreparedArtifacts,
    accelerator: Accelerator,
    seed: int,
    controls: Collection[str],
    run_control: ControlRunner,
) -> PreparedArtifacts:
    """Fit the torch probe (or trunk in finetune mode) and the mandatory random-trunk control."""
    config = trainer.config
    if not isinstance(prepared.model, nn.Module):
        raise TypeError("Torch probe tuning requires a torch.nn.Module model.")
    # PrepareStep creates a linear probe for cached representations.
    # Rebuild the configured MLP architecture; cached embeddings remain unchanged.
    if config.head.head_type == "mlp" and isinstance(
        prepared.model, EmbeddingProbeHead
    ):
        prepared.model = EmbeddingProbeHead(
            prepared.model.embedding_size,
            prepared.model.num_labels,
            prepared.model.task_type,
            probe_hidden_size=config.head.hidden_size,
            probe_hidden_depth=config.head.hidden_depth,
        ).to(accelerator.device)

    model = prepared.model
    embedding_dataset = find_embedding_dataset(prepared.train_dataloader)
    cached_probe = embedding_dataset is not None
    if embedding_dataset is not None and isinstance(model, EmbeddingProbeHead):
        features = embedding_dataset.embeddings
        row_mask = None
        if config.standardize_features:
            if get_task(workspace.task_type).masks_ignore_index:
                row_mask = torch.as_tensor(embedding_dataset.labels) != IGNORE_INDEX
                num_rows = int(row_mask.sum().item())
            else:
                num_rows = features.shape[0]
            logger.info(
                "Standardizing cached probe features using %d training rows.",
                num_rows,
            )
        model.fit_standardization(
            features,
            enabled=config.standardize_features,
            row_mask=row_mask,
            show_progress=accelerator.is_main_process,
        )
    if not cached_probe and not trainer.train_trunk:
        logger.warning(
            "Embedding cache is unavailable; tuning head_type='%s' "
            "without it will recompute frozen-trunk representations "
            "during training.",
            config.head.head_type,
        )
    if config.head.head_type == "torch_linear" and not cached_probe:
        validate_native_linear_head(model)
    distributed_training = not cached_probe

    # Set requires_grad on the unwrapped model before accelerator.prepare()
    # creates DDP/FSDP wrappers. Those wrappers build gradient-reduction
    # state from the current trainable parameters; changing the flags
    # afterward can leave that state inconsistent and hang backward() in
    # multi-GPU training.
    configure_trainable_params(model, trainer.train_trunk)
    if distributed_training:
        model = accelerator.prepare(model)
        prepared.model = model

    if config.hyperparameter_search.enabled:
        # Hparam search for HF-defined heads or trunk + mlp/linear model
        best_hparams = search_hyperparameters(
            trainer, workspace, prepared, accelerator, seed
        )
    else:
        # If we're not tuning, we just use configured hparams for a single training run
        best_hparams = {
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "max_epochs": config.max_epochs,
        }
        if not config.refit_on_train_validation:
            trainer.train(workspace, prepared, accelerator, best_hparams)

    # Architecture search can replace the head.
    if config.variance_seeds and cached_probe:
        trainer.report_seed_variance(workspace, prepared, accelerator, best_hparams)
    if config.refit_on_train_validation:
        refit_cached_probe(
            trainer, workspace, prepared, accelerator, best_hparams, seed
        )

    model = prepared.model
    model.eval()
    if config.save_best_model:
        trainer.save_model(workspace, prepared, model, accelerator, best_hparams, seed)
    fit_cached_random_trunk(
        trainer, workspace, prepared, accelerator, seed, controls, run_control
    )
    return prepared


def fit_probe_protocol(
    trainer: TorchProbeTrainer,
    workspace: EvaluationWorkspace,
    prepared: PreparedArtifacts,
    *,
    seed: int = 710019,
    accelerator: Accelerator | None = None,
    controls: Collection[str] = (),
) -> PreparedArtifacts:
    """Validate and run the configured probe protocol for a prepared task.

    This is the ordinary fitting entry point: validation, layerwise and classical
    dispatch, and the torch protocol. The random-trunk control re-enters this
    function with its own non-finetune trainer and no controls, so no step object
    is constructed recursively.
    """
    config = trainer.config
    logger.info(
        "Running probe protocol (%s)...",
        "finetune" if trainer.train_trunk else "tune_probe",
    )
    if accelerator is None:
        accelerator = Accelerator(gradient_accumulation_steps=config.accumulation_steps)

    # Guard in case the Prepare step produced no train dataloader (e.g. no 'train' split).
    if prepared.train_dataloader is None:
        raise ValueError(
            "Tuning requires a train_dataloader, but the Prepare step "
            "did not produce one (no 'train' split in the dataset)."
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
            trainer.train_trunk
            or config.head.head_type != "mlp"
            or not isinstance(prepared.model, EmbeddingProbeHead)
            or find_embedding_dataset(prepared.train_dataloader) is None
        )
    ):
        dimension = (
            "hidden_size"
            if search.search_space.hidden_size is not None
            else (
                "hidden_depth"
                if search.search_space.hidden_depth is not None
                else "batch_size"
            )
        )
        raise ValueError(
            f"{dimension} search requires a cached MLP probe in tune_probe mode."
        )
    if config.refit_on_train_validation:
        if (
            trainer.train_trunk
            or isinstance(prepared.model, LayerwiseProbeModel)
            or find_embedding_dataset(prepared.train_dataloader) is None
            or (
                config.head.head_type not in CLASSICAL_HEAD_TYPES
                and not isinstance(prepared.model, EmbeddingProbeHead)
            )
        ):
            raise ValueError(
                "refit_on_train_validation requires an ordinary cached probe "
                "in tune_probe mode; live/native and layerwise refit are not supported."
            )
        for name, loader in (
            ("train", prepared.train_dataloader),
            ("validation", prepared.val_dataloader),
        ):
            dataset = find_embedding_dataset(loader)
            if dataset is None or not len(dataset):
                raise ValueError(
                    "refit_on_train_validation requires nonempty train and "
                    f"validation embedding datasets; missing or empty {name} split."
                )
        missing_refit_splits = set(prepared.refit_token_dataloaders) - set(
            prepared.refit_dataloaders
        )
        if missing_refit_splits:
            raise ValueError(
                "Final-refit splits were not cached as embeddings: "
                f"{sorted(missing_refit_splits)}."
            )
        for name, loader in prepared.refit_dataloaders.items():
            dataset = find_embedding_dataset(loader)
            if dataset is None or not len(dataset):
                raise ValueError(
                    "refit_on_train_validation requires nonempty final-refit "
                    f"embedding datasets; missing or empty {name} split."
                )
    if isinstance(prepared.model, LayerwiseProbeModel):
        return run_layerwise(trainer, workspace, prepared, accelerator, seed, controls)

    # Classical heads bypass the epoch/optimizer loop and fit on cached pooled embeddings.
    if config.head.head_type in CLASSICAL_HEAD_TYPES:
        run_classical_head(trainer, workspace, prepared, seed, accelerator)
        fit_cached_random_trunk(
            trainer,
            workspace,
            prepared,
            accelerator,
            seed,
            controls,
            _control_runner(trainer),
        )
        return prepared

    return fit_torch_probe(
        trainer,
        workspace,
        prepared,
        accelerator,
        seed,
        controls,
        _control_runner(trainer),
    )


def _control_runner(trainer: TorchProbeTrainer) -> ControlRunner:
    """Run the same protocol on the random-trunk control: no variance seeds, no controls."""
    control_trainer = TorchProbeTrainer(
        trainer.config.model_copy(update={"variance_seeds": []}),
        train_trunk=False,
    )

    def run_control(
        workspace: EvaluationWorkspace,
        prepared: PreparedArtifacts,
        accelerator: Accelerator,
        seed: int,
    ) -> None:
        fit_probe_protocol(
            control_trainer,
            workspace,
            prepared,
            seed=seed,
            accelerator=accelerator,
            controls=(),
        )

    return run_control
