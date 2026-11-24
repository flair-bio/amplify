"""Assemble task observations and control predictions without publication."""

from __future__ import annotations

import json
import logging
from time import monotonic
from typing import Any, cast

import polars as pl
import torch
from accelerate import Accelerator
from sklearn.base import BaseEstimator

from modules.evaluate.src.artifacts.results import EvaluationMetadata, PredictOutput
from modules.evaluate.src.artifacts.workspace import (
    EvaluationWorkspace,
)
from modules.evaluate.src.config import (
    BASELINE_NAMES,
    CLASSICAL_HEAD_TYPES,
    BaselineName,
    PredictConfig,
)
from modules.evaluate.src.controls.inference import predict_control
from modules.evaluate.src.dataset.dataloader import attach_metric_groups
from modules.evaluate.src.dataset.embeddings import find_embedding_dataset
from modules.evaluate.src.dataset.types import TestData, TrainingData
from modules.evaluate.src.model.baselines import build_prediction_baseline
from modules.evaluate.src.model.classical_estimators import run_classical_inference
from modules.evaluate.src.model.controls import NO_INFERENCE_CONTROLS, ControlPredictors
from modules.evaluate.src.model.factory import Predictor
from modules.evaluate.src.model.inference import (
    gather_training_labels,
    run_inference,
    run_local_embedding_inference,
)
from modules.evaluate.src.model.layerwise import (
    LayerwiseProbeModel,
    run_layerwise_control_conditions,
)
from modules.evaluate.src.tasks.registry import get_flat_task

logger = logging.getLogger(__name__)


def predict_layerwise(
    config: PredictConfig,
    workspace: EvaluationWorkspace,
    predictor: Predictor,
    training: TrainingData,
    test: TestData,
    control_predictors: ControlPredictors,
    accelerator: Accelerator,
    seed: int,
) -> dict[int, PredictOutput]:
    """Build per-layer prediction artifacts from live or cached features.

    Infer trained and model-based controls together where features can be
    shared, then construct constant controls from the resulting rows.
    Each layer gets its own workspace and metadata. Control IDs must match
    trained prediction order so downstream paired comparisons are valid.
    """
    if test.test_dataloader is None:
        raise ValueError("Layerwise prediction requires a test dataloader.")
    inner = accelerator.unwrap_model(predictor.model)
    if not isinstance(inner, LayerwiseProbeModel):
        raise TypeError("Layerwise prediction requires a LayerwiseProbeModel.")
    using_embeddings = test.embedding is not None
    conditions = run_layerwise_control_conditions(
        predictor,
        test,
        control_predictors,
        workspace.task_type,
        accelerator,
        config,
        seed,
    )
    sequence_by_id = None
    if using_embeddings and config.store_sequences:
        assert test.embedding is not None
        sequence_by_id = _decode_token_dataloader(
            test.embedding.backend.token_test_dataloader,
            predictor.tokenizer,
            accelerator,
        )
    outputs: dict[int, PredictOutput] = {}
    train_labels = (
        gather_training_labels(training, accelerator)
        if set(config.baselines) & BASELINE_NAMES
        else None
    )
    for layer, (ids, predictions, labels, probabilities, sequences) in conditions[
        "trained"
    ].items():
        layer_workspace = EvaluationWorkspace(
            workspace.config,
            variant=f"{workspace.variant}-layer_{layer}",
            seed=workspace.seed,
        )
        if workspace.test_subset is not None:
            layer_workspace = layer_workspace.test_workspace(workspace.test_subset)
        metadata = EvaluationMetadata(
            model_name=workspace.model_name,
            model_repo_id=workspace.model_repo_id,
            dataset_name=workspace.dataset_name,
            dataset_repo_id=workspace.dataset_repo_id,
            dataset_config_name=workspace.config.dataset_config_name,
            dataset_revision=workspace.config.dataset_revision,
            group_column=test.test_group_column,
            task_type=workspace.task_type,
            split=test.split,
            test_subset=workspace.test_subset,
            seed=seed,
            head_type="torch_linear",
            layer_index=layer,
            variant=layer_workspace.variant,
        )
        data: dict[str, list[Any]] = {
            "layer_index": [layer] * len(ids),
            "id": ids,
            "prediction": predictions,
            "label": labels,
        }
        if probabilities is not None:
            data["probability"] = probabilities
        if sequences is not None:
            data["sequence"] = sequences
        predictions_df = pl.DataFrame(data)
        predictions_df = _add_token_positions(
            predictions_df, test.test_original_positions
        )
        predictions_df = attach_metric_groups(
            predictions_df, test.test_metric_groups, test.test_group_column
        )
        if sequence_by_id is not None:
            predictions_df = predictions_df.join(sequence_by_id, on="id", how="left")
        controls: dict[str, pl.DataFrame] = {}
        for control in config.conditions:
            if control in NO_INFERENCE_CONTROLS:
                continue
            if control in BASELINE_NAMES:
                assert train_labels is not None
                controls[control] = build_prediction_baseline(
                    cast(BaselineName, control),
                    workspace.task_type,
                    workspace.dataset_name,
                    workspace.num_labels,
                    predictions_df,
                    train_labels,
                )
                controls[control] = controls[control].with_columns(
                    pl.lit(layer).alias("layer_index")
                )
                continue
            control_ids, control_predictions, _, control_probs, control_sequences = (
                conditions[control][layer]
            )
            if control_ids != ids:
                raise RuntimeError(
                    f"Layer {layer} control '{control}' changed example order."
                )
            control_data: dict[str, list[Any]] = {
                "layer_index": [layer] * len(ids),
                "id": ids,
                "prediction": control_predictions,
            }
            if control_probs is not None:
                control_data["probability"] = control_probs
            if control_sequences is not None:
                control_data["sequence"] = control_sequences
            controls[control] = pl.DataFrame(control_data)
        outputs[layer] = PredictOutput(
            metadata=metadata,
            predictions=predictions_df,
            controls=controls,
            requested_controls=config.conditions,
            config=config.model_dump(mode="json"),
            baseline_context=test.baseline_context,
        )
    return outputs


def predict_test_set(
    config: PredictConfig,
    workspace: EvaluationWorkspace,
    predictor: Predictor,
    training: TrainingData,
    test: TestData,
    control_predictors: ControlPredictors,
    seed: int = 710019,
    head_type: str | None = None,
    variant: str | None = None,
    accelerator: Accelerator | None = None,
) -> PredictOutput:
    metadata = EvaluationMetadata(
        model_name=workspace.model_name,
        model_repo_id=workspace.model_repo_id,
        dataset_name=workspace.dataset_name,
        dataset_repo_id=workspace.dataset_repo_id,
        dataset_config_name=workspace.config.dataset_config_name,
        dataset_revision=workspace.config.dataset_revision,
        group_column=test.test_group_column,
        task_type=workspace.task_type,
        split=test.split,
        test_subset=workspace.test_subset,
        seed=seed,
        head_type=head_type,
        variant=variant,
    )
    if test.test_dataloader is None:
        raise ValueError(
            "Prediction requires a test dataloader; no test split was prepared."
        )
    assert test.test_dataloader is not None
    if accelerator is None:
        raise ValueError("Pass the application-owned accelerator explicitly.")
    started = monotonic()
    test_dataset = getattr(test.test_dataloader, "dataset", None)
    progress_total_batches = len(test.test_dataloader)
    progress_total_rows = len(test_dataset) if test_dataset is not None else None
    progress_unit = get_flat_task(workspace.task_type).prediction_progress_unit
    if accelerator.is_main_process:
        logger.info("Trained prediction started")
        if progress_total_rows is not None:
            logger.info(
                "Prediction workload: %d %s across %d batches.",
                progress_total_rows,
                progress_unit,
                progress_total_batches,
            )
    progress_label = "Trained prediction" if accelerator.is_main_process else None
    using_embeddings = test.embedding is not None
    if head_type in CLASSICAL_HEAD_TYPES or isinstance(predictor.model, BaseEstimator):
        num_labels = get_flat_task(workspace.task_type).resolve_num_labels(
            workspace.num_labels, workspace.dataset_name
        )
        embedding_dataset = find_embedding_dataset(test.test_dataloader)
        if embedding_dataset is not None:
            batch_size = (
                config.embedding_batch_size or test.test_dataloader.batch_size or 16
            )
            dataloader = (
                {
                    "inputs_embeds": embedding_dataset.embeddings[
                        start : start + batch_size
                    ],
                    "labels": torch.as_tensor(embedding_dataset.labels)[
                        start : start + batch_size
                    ],
                    "id": embedding_dataset.ids[start : start + batch_size],
                }
                for start in range(0, len(embedding_dataset), batch_size)
            )
        else:
            dataloader = test.test_dataloader
        ids, predictions, labels, probabilities = run_classical_inference(
            predictor.model,
            dataloader,
            workspace.task_type,
            num_labels,
            collect_probabilities=config.collect_probabilities,
            progress_label=progress_label,
        )
        sequences = None
    elif using_embeddings:
        embedding_dataset = find_embedding_dataset(test.test_dataloader)
        if embedding_dataset is None:
            raise ValueError("Cached predictions require cached test embeddings.")
        ids, predictions, labels, probabilities = run_local_embedding_inference(
            predictor.model,
            embedding_dataset.embeddings,
            embedding_dataset.ids,
            torch.as_tensor(embedding_dataset.labels),
            workspace.task_type,
            accelerator.device,
            config.embedding_batch_size or test.test_dataloader.batch_size or 16,
            collect_probabilities=config.collect_probabilities,
            progress_label=progress_label,
        )
        sequences = None
    else:
        ids, predictions, labels, probabilities, sequences = run_inference(
            predictor.model,
            test.test_dataloader,
            workspace.task_type,
            accelerator,
            collect_probabilities=config.collect_probabilities,
            tokenizer=None
            if using_embeddings
            else predictor.tokenizer
            if config.store_sequences
            else None,
            progress_label=progress_label,
            progress_total_batches=progress_total_batches,
            progress_total_rows=progress_total_rows,
            progress_unit=progress_unit,
        )
    prediction_data: dict[str, list[Any]] = {
        "id": ids,
        "prediction": predictions,
        "label": labels,
    }
    if probabilities is not None:
        prediction_data["probability"] = probabilities
    if sequences is not None:
        prediction_data["sequence"] = sequences
    predictions_df = pl.DataFrame(prediction_data)
    predictions_df = _add_token_positions(predictions_df, test.test_original_positions)
    predictions_df = attach_metric_groups(
        predictions_df, test.test_metric_groups, test.test_group_column
    )
    if using_embeddings and config.store_sequences:
        assert test.embedding is not None
        sequence_by_id = _decode_token_dataloader(
            test.embedding.backend.token_test_dataloader,
            predictor.tokenizer,
            accelerator,
        )
        predictions_df = predictions_df.join(sequence_by_id, on="id", how="left")
    if accelerator.is_main_process:
        logger.info(
            "Trained prediction finished in %.1fs (%d rows)",
            monotonic() - started,
            predictions_df.height,
        )
    controls: dict[str, pl.DataFrame] = {}
    train_labels = None
    for control in config.conditions:
        if control in NO_INFERENCE_CONTROLS:
            continue
        if control in BASELINE_NAMES:
            if train_labels is None:
                train_labels = gather_training_labels(training, accelerator)
            control_df = build_prediction_baseline(
                cast(BaselineName, control),
                workspace.task_type,
                workspace.dataset_name,
                workspace.num_labels,
                predictions_df,
                train_labels,
            )
        else:
            control_df = predict_control(
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
        controls[control] = control_df
    output = PredictOutput(
        metadata=metadata,
        predictions=predictions_df,
        controls=controls,
        requested_controls=config.conditions,
        config=config.model_dump(mode="json"),
        baseline_context=test.baseline_context,
    )
    return output


def _decode_token_dataloader(
    token_dataloader: Any, tokenizer: Any, accelerator: Accelerator
) -> pl.DataFrame:
    """Decode every example's ``input_ids`` from *token_dataloader* to a
    ``{"id", "sequence"}`` dataframe, for joining onto embedding-mode
    predictions by ``id`` (a pooled-embedding test_dataloader has no
    ``input_ids`` to decode from directly)."""
    all_ids: list[Any] = []
    all_sequences: list[str] = []
    from modules.evaluate.src.dataset.collator import input_id_rows

    for batch in token_dataloader:
        batch = dict(batch)
        ids = batch.pop("id", None)
        all_sequences.extend(
            tokenizer.batch_decode(
                input_id_rows(batch, tokenizer.pad_token_id or 0).cpu().tolist(),
                skip_special_tokens=True,
            )
        )
        if ids is not None:
            all_ids.extend(ids.tolist() if torch.is_tensor(ids) else ids)
    if not all_ids:
        all_ids = list(range(len(all_sequences)))
    return pl.DataFrame({"id": all_ids, "sequence": all_sequences})


def _add_token_positions(
    predictions: pl.DataFrame, positions_by_id: dict[str, list[int]]
) -> pl.DataFrame:
    if "label" not in predictions.columns or not any(
        (isinstance(value, list) for value in predictions["label"].to_list())
    ):
        return predictions
    if not positions_by_id:
        return predictions
    grouped_rows: dict[str, list[tuple[int, Any]]] = {}
    identifiers: dict[str, Any] = {}
    for index, (identifier, labels) in enumerate(
        predictions.select("id", "label").iter_rows()
    ):
        key = json.dumps(identifier, allow_nan=False)
        if key not in positions_by_id:
            raise ValueError(
                "Token predictions contain an ID absent from prepared residue-position provenance."
            )
        identifiers[key] = identifier
        grouped_rows.setdefault(key, []).append((index, labels))
    positions: list[list[int] | None] = [None] * predictions.height
    for key, rows in grouped_rows.items():
        row_positions = positions_by_id[key]
        row_labels = [
            value
            for _, labels in rows
            for value in (labels if isinstance(labels, list) else [labels])
        ]
        if len(row_positions) != len(row_labels):
            raise ValueError(
                f"Token prediction labels for ID {identifiers[key]!r} ({len(row_labels)}) differ from prepared original residue positions ({len(row_positions)})."
            )
        cursor = 0
        for index, labels in rows:
            count = len(labels) if isinstance(labels, list) else 1
            positions[index] = row_positions[cursor : cursor + count]
            cursor += count
    missing_ids = set(positions_by_id) - set(grouped_rows)
    if any((positions_by_id[key] for key in missing_ids)):
        raise ValueError(
            "Token predictions are missing IDs with valid prepared residue positions."
        )
    return predictions.with_columns(
        pl.Series(
            "original_positions",
            [row or [] for row in positions],
            dtype=pl.List(pl.Int32),
        )
    )
