"""Predict inference-only model and input control conditions."""

from __future__ import annotations

import logging
from pathlib import Path
from time import monotonic
from typing import Any

import polars as pl
import torch
from accelerate import Accelerator
from sklearn.base import BaseEstimator
from torch.utils.data import DataLoader, DistributedSampler

from modules.evaluate.src.artifacts.workspace import EvaluationWorkspace
from modules.evaluate.src.config.evaluation import PredictConfig
from modules.evaluate.src.config.shared import BASELINE_NAMES
from modules.evaluate.src.dataset.embeddings import (
    EmbeddingDataset,
    embedding_collate_fn,
    find_embedding_dataset,
)
from modules.evaluate.src.dataset.types import TestData
from modules.evaluate.src.model.classical import run_classical_inference
from modules.evaluate.src.model.controls import (
    RANDOM_TRUNK_CONTROL,
    RANDOMIZED_HEAD_CONTROLS,
    RANDOMIZED_TRUNK_CONTROLS,
    SCRAMBLED_SEQUENCES_CONTROL,
    UNTRAINED_CONTROL,
    ControlPredictors,
    build_control_model,
    build_random_trunk,
    make_sequence_scrambler,
)
from modules.evaluate.src.model.embeddings import extract_embeddings
from modules.evaluate.src.model.factory import Predictor, build_predictor
from modules.evaluate.src.model.heads import EmbeddingProbeHead
from modules.evaluate.src.model.inference import run_inference, run_local_head_inference
from modules.evaluate.src.tasks.registry import get_flat_task

logger = logging.getLogger(__name__)


def predict_control(
    config: PredictConfig,
    control: str,
    workspace: EvaluationWorkspace,
    predictor: Predictor,
    test: TestData,
    control_predictors: ControlPredictors,
    predictions_df: pl.DataFrame,
    accelerator: Accelerator,
    seed: int,
) -> tuple[pl.DataFrame, Path | None]:
    """Dispatch to the right control-building strategy.

    The returned path is always ``None``. Control predictions depend on
    the fitted head and are deliberately not cached; only trunk
    embeddings are reusable across runs.
    """
    if control in BASELINE_NAMES:
        raise ValueError(
            f"'{control}' is a model-independent baseline; build it with the baseline service, not control inference."
        )
    started = monotonic()
    if accelerator.is_main_process:
        logger.info("Control '%s' prediction started", control)
    result = _get_or_build_control(
        config,
        control,
        workspace,
        predictor,
        test,
        control_predictors,
        predictions_df,
        accelerator,
        seed,
    )
    if accelerator.is_main_process:
        logger.info(
            "Control '%s' prediction finished in %.1fs (%d rows)",
            control,
            monotonic() - started,
            result[0].height,
        )
    return result


def _get_or_build_control(
    config: PredictConfig,
    control: str,
    workspace: EvaluationWorkspace,
    predictor: Predictor,
    test: TestData,
    control_predictors: ControlPredictors,
    predictions_df: pl.DataFrame,
    accelerator: Accelerator,
    seed: int,
) -> tuple[pl.DataFrame, Path | None]:
    """Build a model-based control from the current head and embeddings."""
    is_scrambled_sequences = control == SCRAMBLED_SEQUENCES_CONTROL
    using_embeddings = test.embedding is not None
    if using_embeddings:
        ids, predictions, probabilities, sequences = _run_embedding_control_inference(
            config,
            control,
            workspace,
            predictor,
            test,
            control_predictors,
            accelerator,
            seed,
        )
    else:
        model = _build_control_model(
            config,
            control,
            workspace,
            predictor,
            test,
            control_predictors,
            accelerator,
            seed,
        )
        test_dataloader = test.test_dataloader
        test_dataset = getattr(test_dataloader, "dataset", None)
        progress_total_batches = (
            len(test_dataloader) if test_dataloader is not None else None
        )
        batch_transform = (
            make_sequence_scrambler(predictor.tokenizer, seed)
            if is_scrambled_sequences
            else None
        )
        ids, predictions, _, probabilities, sequences = run_inference(
            model,
            test_dataloader,
            workspace.task_type,
            accelerator,
            collect_probabilities=config.collect_probabilities,
            batch_transform=batch_transform,
            tokenizer=predictor.tokenizer if is_scrambled_sequences else None,
            progress_label=f"Control '{control}' prediction",
            progress_total_batches=progress_total_batches,
            progress_total_rows=len(test_dataset) if test_dataset is not None else None,
            progress_unit=get_flat_task(workspace.task_type).prediction_progress_unit,
        )
    control_data: dict[str, list[Any]] = {"id": ids, "prediction": predictions}
    if probabilities is not None:
        control_data["probability"] = probabilities
    if sequences is not None:
        control_data["sequence"] = sequences
    control_df = pl.DataFrame(control_data)
    if (
        predictions_df["id"].n_unique() < predictions_df.height
        or control_df["id"].n_unique() < control_df.height
    ):
        control_ids = control_df["id"]
        if (
            control_ids.len() != predictions_df.height
            or (control_ids != predictions_df["id"]).any()
        ):
            raise RuntimeError(
                f"Control '{control}' produced {control_ids.len()} rows in a different order/count than the {predictions_df.height} test rows; controls with repeated ids must iterate the test set in the same order as the trained model's predictions."
            )
        aligned = predictions_df.select("id").with_columns(control_df.drop("id"))
        return (aligned, None)
    aligned = predictions_df.select("id").join(
        control_df, on="id", how="left", maintain_order="left", validate="1:1"
    )
    if aligned["prediction"].null_count() > 0:
        raise RuntimeError(
            f"Control '{control}' predictions could not be aligned to all test example ids; check that the control and test dataloaders use the same ids."
        )
    return (aligned, None)


def _run_embedding_control_inference(
    config: PredictConfig,
    control: str,
    workspace: EvaluationWorkspace,
    predictor: Predictor,
    test: TestData,
    control_predictors: ControlPredictors,
    accelerator: Accelerator,
    seed: int,
) -> tuple[list[Any], list[Any], list[Any] | None, list[str] | None]:
    """Rebuild a control's predictions when
    steps.prepare.embedding_cache is enabled: test.test_dataloader
    then yields pooled embeddings (not tokens), and predictor.model isn't
    a full HF model with a `.base_model`/`._init_weights` to
    reinitialize the way :meth:`_build_control_model` does.

    The trunk is always frozen/pretrained for embedding-cache runs (a
    cross-step validator in EvaluationConfig requires tune_probe mode),
    so "untrained"'s trunk is identical to the trained model's trunk --
    only the head differs.
    """
    torch.manual_seed(seed)
    assert test.embedding is not None
    is_ragged = get_flat_task(workspace.task_type).is_ragged
    is_classical_head = isinstance(predictor.model, BaseEstimator)
    trained_head = (
        predictor.model
        if is_classical_head
        else accelerator.unwrap_model(predictor.model)
    )
    if control == RANDOM_TRUNK_CONTROL:
        if control_predictors.random_trunk_model is None:
            raise ValueError(
                "Cached random_trunk requires a fitted probe; call "
                "prepare_control_predictors with the requested controls before prediction."
            )
        if control_predictors.random_trunk_seed != seed:
            raise ValueError(
                "Cached random_trunk prediction seed must match its tuning seed."
            )
        trained_head = control_predictors.random_trunk_model
    needs_fresh_head = control in RANDOMIZED_HEAD_CONTROLS
    if needs_fresh_head and (not isinstance(trained_head, EmbeddingProbeHead)):
        raise ValueError(
            f"control='{control}' requires a randomly re-initializable head, which isn't defined for a classical (non-'mlp') head_type. Use 'random_trunk', 'scrambled_sequences', or 'scrambled_labels' instead (this should have been caught by EvaluationConfig validation)."
        )
    if control in RANDOMIZED_TRUNK_CONTROLS:
        trained_trunk = accelerator.unwrap_model(predictor.trunk)
        trunk = build_random_trunk(trained_trunk, seed)
    else:
        trunk = predictor.trunk
    is_scrambled_sequences = control == SCRAMBLED_SEQUENCES_CONTROL
    embeddings, ids, labels = _get_or_build_control_embeddings(
        config,
        control,
        workspace,
        predictor,
        test,
        control_predictors,
        accelerator,
        seed,
        trunk,
        is_scrambled_sequences,
    )
    sequences = (
        _decode_scrambled_sequences(
            config, workspace, predictor, test, control_predictors, accelerator, seed
        )
        if is_scrambled_sequences
        else None
    )
    if needs_fresh_head:
        head_model = EmbeddingProbeHead(
            trained_head.embedding_size,
            trained_head.num_labels,
            workspace.task_type,
            probe_hidden_size=trained_head.probe_hidden_size,
            probe_hidden_depth=trained_head.probe_hidden_depth,
            standardize_features=trained_head.standardize_features,
        ).to(accelerator.device)
        if trained_head.standardize_features:
            head_model.feature_mean.copy_(trained_head.feature_mean)
            head_model.feature_std.copy_(trained_head.feature_std)
    else:
        head_model = trained_head
    label_dtype = (
        torch.float if workspace.task_type == "sequence_regression" else torch.long
    )
    dataset = EmbeddingDataset(embeddings, ids, labels.to(label_dtype))
    if test.test_dataloader is None:
        raise ValueError("A test dataloader is required for local inference.")
    batch_size = config.embedding_batch_size or test.test_dataloader.batch_size or 16
    if is_classical_head:
        num_labels = get_flat_task(workspace.task_type).resolve_num_labels(
            workspace.num_labels, workspace.dataset_name
        )
        batches = (
            {
                "inputs_embeds": dataset.embeddings[start : start + batch_size],
                "labels": dataset.labels[start : start + batch_size],
                "id": dataset.ids[start : start + batch_size],
            }
            for start in range(0, len(dataset), batch_size)
        )
        control_ids, predictions, _, probabilities = run_classical_inference(
            head_model,
            batches,
            workspace.task_type,
            num_labels,
            collect_probabilities=config.collect_probabilities,
            progress_label=f"Control '{control}' prediction"
            if accelerator.is_main_process
            else None,
        )
        return (control_ids, predictions, probabilities, sequences)
    if is_ragged:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: embedding_collate_fn(batch, label_dtype),
        )
        control_ids, predictions, probabilities = run_local_head_inference(
            head_model,
            dataloader,
            workspace.task_type,
            config.collect_probabilities,
            accelerator,
            gather_across_processes=False,
            progress_label=f"Control '{control}' prediction",
        )
        predictions = [[prediction] for prediction in predictions]
        return (control_ids, predictions, probabilities, sequences)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=DistributedSampler(
            dataset,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            shuffle=False,
        ),
        collate_fn=lambda batch: embedding_collate_fn(batch, label_dtype),
    )
    control_ids, predictions, probabilities = run_local_head_inference(
        head_model,
        dataloader,
        workspace.task_type,
        config.collect_probabilities,
        accelerator,
        progress_label=f"Control '{control}' prediction",
    )
    return (control_ids, predictions, probabilities, sequences)


def _get_or_build_control_embeddings(
    config: PredictConfig,
    control: str,
    workspace: EvaluationWorkspace,
    predictor: Predictor,
    test: TestData,
    control_predictors: ControlPredictors,
    accelerator: Accelerator,
    seed: int,
    trunk: Any,
    is_scrambled_sequences: bool,
) -> tuple[torch.Tensor, list[Any], torch.Tensor]:
    assert test.embedding is not None
    backend = test.embedding.backend
    is_ragged = get_flat_task(workspace.task_type).is_ragged
    input_transform = (
        f"scramble_sequences:v1:seed={seed}" if is_scrambled_sequences else None
    )
    cache_key = backend.test_embedding_key()
    if control in RANDOMIZED_TRUNK_CONTROLS or is_scrambled_sequences:
        cache_key = backend.test_embedding_key(
            trunk=trunk, seed=seed, input_transform=input_transform
        )
    cache = backend.cache
    if (
        cache is not None
        and cache_key is not None
        and (backend.cache_policy == "reuse")
        and cache.exists("test", cache_key)
    ):
        started = monotonic()
        if accelerator.is_main_process:
            logger.info("Loading cached '%s' control embeddings from disk...", control)
        embeddings, ids, labels = cache.read("test", cache_key)
        if accelerator.is_main_process:
            logger.info(
                "Loaded control '%s' embeddings in %.1fs (%d rows)",
                control,
                monotonic() - started,
                len(ids),
            )
        return (embeddings, ids, labels)
    cached_dataset = find_embedding_dataset(test.test_dataloader)
    if input_transform is None and control not in RANDOMIZED_TRUNK_CONTROLS:
        if cached_dataset is None:
            raise ValueError(
                "Embedding controls require cached test embeddings or a token-based test dataloader."
            )
        embeddings = cached_dataset.embeddings
        ids = cached_dataset.ids
        labels = torch.as_tensor(cached_dataset.labels)
    else:
        batch_transform = (
            make_sequence_scrambler(predictor.tokenizer, seed)
            if is_scrambled_sequences
            else None
        )
        started = monotonic()
        if accelerator.is_main_process:
            logger.info("Control '%s' embedding extraction started", control)
        embeddings, ids, labels, _ = extract_embeddings(
            trunk,
            backend.token_test_dataloader,
            accelerator,
            backend.pooling,
            backend.dtype,
            layers=backend.layers or [-1],
            autocast_dtype={
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }[backend.autocast_dtype],
            normalize_before_pooling=backend.normalize_before_pooling,
            batch_transform=batch_transform,
            is_ragged=is_ragged,
            gather_across_processes=False,
            progress_label=f"Control '{control}' embedding extraction",
        )
        if accelerator.is_main_process:
            logger.info(
                "Control '%s' embedding extraction finished in %.1fs (%d rows)",
                control,
                monotonic() - started,
                len(ids),
            )
        if (
            cache is not None
            and cache_key is not None
            and (backend.cache_policy != "off")
            and (not is_ragged)
            and cache.fits_budget(
                embeddings.shape[0], embeddings.shape[-1], backend.dtype
            )
            and accelerator.is_main_process
        ):
            started = monotonic()
            logger.info("Persisting control '%s' embeddings started", control)
            cache.write("test", cache_key, embeddings, ids, labels)
            logger.info(
                "Persisting control '%s' embeddings finished in %.1fs",
                control,
                monotonic() - started,
            )
        accelerator.wait_for_everyone()
    return (embeddings, ids, labels)


def _decode_scrambled_sequences(
    config: PredictConfig,
    workspace: EvaluationWorkspace,
    predictor: Predictor,
    test: TestData,
    control_predictors: ControlPredictors,
    accelerator: Accelerator,
    seed: int,
) -> list[str]:
    assert test.embedding is not None
    if predictor.tokenizer is None:
        raise ValueError("Decoding scrambled sequences requires a tokenizer.")
    transform = make_sequence_scrambler(predictor.tokenizer, seed)
    all_sequences: list[str] = []
    for batch in test.embedding.backend.token_test_dataloader:
        batch = dict(batch)
        batch.pop("id", None)
        batch.pop("labels", None)
        transformed = transform(batch)
        all_sequences.extend(
            predictor.tokenizer.batch_decode(
                transformed["input_ids"].cpu().tolist(), skip_special_tokens=True
            )
        )
    return all_sequences


def _build_control_model(
    config: PredictConfig,
    control: str,
    workspace: EvaluationWorkspace,
    predictor: Predictor,
    test: TestData,
    control_predictors: ControlPredictors,
    accelerator: Accelerator,
    seed: int,
) -> Any:
    pretrained_model = None
    if control == UNTRAINED_CONTROL:
        task = get_flat_task(workspace.task_type)
        torch.manual_seed(seed)
        pretrained_model = build_predictor(
            workspace.model_repo_id or workspace.model_name,
            workspace.task_type,
            task.resolve_num_labels(workspace.num_labels, workspace.dataset_name),
            workspace.model_kwargs,
        )
    return build_control_model(
        control, accelerator.unwrap_model(predictor.model), seed, pretrained_model
    )
