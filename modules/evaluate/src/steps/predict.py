"""
Run inference for the best (trained) model and, for any configured control
conditions that haven't already been computed for this model + dataset
combination, run inference for those too. No metrics are computed here --
that (plus bootstrap confidence intervals and trained-vs-control comparisons)
is :class:`~modules.evaluate.src.steps.score.ScoreStep`'s job.
"""

from __future__ import annotations

import logging
from pathlib import Path
from time import monotonic
from typing import Any, cast

import polars as pl
import torch
from accelerate import Accelerator
from sklearn.base import BaseEstimator

from modules.evaluate.src.contracts.artifacts import (
    EvaluationMetadata,
    PredictOutput,
    PreparedArtifacts,
)
from modules.evaluate.src.contracts.config import (
    BASELINE_NAMES,
    CLASSICAL_HEAD_TYPES,
    BaselineName,
    PredictConfig,
)
from modules.evaluate.src.dataset.embeddings import find_embedding_dataset
from modules.evaluate.src.dataset.workspace import (
    EvaluationWorkspace,
    get_evaluation_workspace,
)
from modules.evaluate.src.model.baselines import build_prediction_baseline
from modules.evaluate.src.model.controls import NO_INFERENCE_CONTROLS
from modules.evaluate.src.model.heads import run_classical_inference
from modules.evaluate.src.model.inference import (
    gather_training_labels,
    run_inference,
    run_local_embedding_inference,
)
from modules.evaluate.src.model.layerwise import (
    LayerwiseProbeModel,
    run_layerwise_control_conditions,
)
from modules.evaluate.src.steps.controls import ControlBuilder
from modules.evaluate.src.tasks import get_task
from modules.evaluate.src.tasks.base import StructuredTask
from modules.evaluate.src.utils.io import (
    downcast_prediction_dtypes,
    update_run_manifest,
    write_parquet,
)

logger = logging.getLogger(__name__)


def persist_predictions(
    workspace: EvaluationWorkspace, output: PredictOutput, seed: int
) -> None:
    """Write predictions and uncached controls, then record in the run
    manifest everything ScoreStep needs to rerun as a disk-only process."""
    started = monotonic()
    logger.info("Persisting predictions started (%s)", workspace.preds_path)
    config = output.config or {}
    output.predictions_path = write_parquet(
        downcast_prediction_dtypes(output.predictions),
        workspace.preds_path
        / config.get("predictions_filename", "predictions.parquet"),
    )
    for name, frame in output.controls.items():
        if name not in output.control_paths:
            output.control_paths[name] = write_parquet(
                downcast_prediction_dtypes(frame),
                workspace.control_workspace(name).preds_path / "predictions.parquet",
            )
    update_run_manifest(
        workspace.run_manifest_path,
        "predict",
        {
            "seed": seed,
            "config": config,
            "metadata": output.metadata.model_dump(mode="json"),
            "control_paths": {k: str(v) for k, v in output.control_paths.items()},
            "baseline_context": output.baseline_context,
        },
    )
    logger.info(
        "Persisting predictions finished in %.1fs (%d rows, %d controls)",
        monotonic() - started,
        output.predictions.height,
        len(output.controls),
    )


class PredictStep:
    def __init__(self, config: PredictConfig) -> None:
        self.config = config

    def run_layerwise(
        self,
        workspace: EvaluationWorkspace,
        prepared: PreparedArtifacts,
        accelerator: Accelerator,
        seed: int,
    ) -> dict[int, PredictOutput]:
        """Build per-layer prediction artifacts from live or cached features.

        Infer trained and model-based controls together where features can be
        shared, then construct constant controls from the resulting rows.
        Each layer gets its own workspace and metadata. Control IDs must match
        trained prediction order so downstream paired comparisons are valid.
        """
        if prepared.test_dataloader is None:
            raise ValueError("Layerwise prediction requires a test dataloader.")
        inner = accelerator.unwrap_model(prepared.model)
        if not isinstance(inner, LayerwiseProbeModel):
            raise TypeError("Layerwise prediction requires a LayerwiseProbeModel.")
        using_embeddings = prepared.embedding is not None
        conditions = run_layerwise_control_conditions(
            prepared, workspace.task_type, accelerator, self.config, seed
        )

        sequence_by_id = None
        if using_embeddings and self.config.store_sequences:
            assert prepared.embedding is not None
            sequence_by_id = self._decode_token_dataloader(
                prepared.embedding.backend.token_test_dataloader,
                prepared.tokenizer,
                accelerator,
            )

        outputs: dict[int, PredictOutput] = {}
        train_labels = (
            gather_training_labels(prepared, accelerator)
            if set(self.config.baselines) & BASELINE_NAMES
            else None
        )
        for layer, (ids, predictions, labels, probabilities, sequences) in conditions[
            "trained"
        ].items():
            layer_workspace = get_evaluation_workspace(
                workspace.config,
                variant=f"{workspace.variant}-layer_{layer}",
                seed=workspace.seed,
            )
            layer_workspace.setup_directories()
            metadata = EvaluationMetadata(
                model_name=workspace.model_name,
                model_repo_id=workspace.model_repo_id,
                dataset_name=workspace.dataset_name,
                dataset_repo_id=workspace.dataset_repo_id,
                dataset_config_name=workspace.config.dataset_config_name,
                dataset_revision=workspace.config.dataset_revision,
                task_type=workspace.task_type,
                split=prepared.split,
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
            if sequence_by_id is not None:
                # Feature batches contain no tokens; recover sequences by
                # source ID, including repeated IDs in residue-level caches.
                predictions_df = predictions_df.join(
                    sequence_by_id,
                    on="id",
                    how="left",
                )
            controls: dict[str, pl.DataFrame] = {}
            for control in self.config.conditions:
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
                (
                    control_ids,
                    control_predictions,
                    _,
                    control_probs,
                    control_sequences,
                ) = conditions[control][layer]
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
                requested_controls=self.config.conditions,
                config=self.config.model_dump(mode="json"),
                baseline_context=prepared.baseline_context,
            )
            if self.config.persist_to_disk and accelerator.is_main_process:
                persist_predictions(layer_workspace, outputs[layer], seed)
        return outputs

    def run(
        self,
        workspace: EvaluationWorkspace,
        prepared: PreparedArtifacts,
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
            task_type=workspace.task_type,
            split=prepared.split,
            seed=seed,
            head_type=head_type,
            variant=variant,
        )
        task = get_task(workspace.task_type, workspace.num_labels)
        if isinstance(task, StructuredTask):
            if accelerator is not None and accelerator.num_processes != 1:
                raise ValueError("Structured prediction requires a single process")
            output = task.predict(prepared, metadata, self.config)
            output.config = self.config.model_dump(mode="json")
            if self.config.persist_to_disk and (
                accelerator is None or accelerator.is_main_process
            ):
                persist_predictions(workspace, output, seed)
            return output
        if prepared.test_dataloader is None:
            raise ValueError(
                "PredictStep requires a test_dataloader, but the Prepare "
                "step did not produce one (no 'test' split in the dataset)."
            )
        assert prepared.test_dataloader is not None
        # The pipeline passes the same instance used by Prepare/Tune, so
        # gather_for_metrics() and is_main_process use one process state.
        if accelerator is None:
            accelerator = Accelerator()

        started = monotonic()
        test_dataset = getattr(prepared.test_dataloader, "dataset", None)
        progress_total_batches = len(prepared.test_dataloader)
        progress_total_rows = len(test_dataset) if test_dataset is not None else None
        progress_unit = get_task(workspace.task_type).prediction_progress_unit
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
        # steps.prepare.embedding_cache.enabled=True: test_dataloader yields
        # pooled inputs_embeds, not tokens, so sequences can't be decoded
        # from it directly -- decode separately from the retained
        # token-based dataloader and join back onto predictions by id below.
        using_embeddings = prepared.embedding is not None

        # Classical heads are fitted estimators over cached embeddings, not
        # torch modules, so they use their direct prediction path.
        if head_type in CLASSICAL_HEAD_TYPES or isinstance(
            prepared.model, BaseEstimator
        ):
            num_labels = get_task(workspace.task_type).resolve_num_labels(
                workspace.num_labels, workspace.dataset_name
            )
            embedding_dataset = find_embedding_dataset(prepared.test_dataloader)
            if embedding_dataset is not None:
                batch_size = (
                    self.config.embedding_batch_size
                    or prepared.test_dataloader.batch_size
                    or 16
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
                dataloader = prepared.test_dataloader
            ids, predictions, labels, probabilities = run_classical_inference(
                prepared.model,
                dataloader,
                workspace.task_type,
                num_labels,
                collect_probabilities=self.config.collect_probabilities,
                progress_label=progress_label,
            )
            sequences = None
        elif using_embeddings:
            embedding_dataset = find_embedding_dataset(prepared.test_dataloader)
            if embedding_dataset is None:
                raise ValueError("Cached predictions require cached test embeddings.")
            ids, predictions, labels, probabilities = run_local_embedding_inference(
                prepared.model,
                embedding_dataset.embeddings,
                embedding_dataset.ids,
                torch.as_tensor(embedding_dataset.labels),
                workspace.task_type,
                accelerator.device,
                self.config.embedding_batch_size
                or prepared.test_dataloader.batch_size
                or 16,
                collect_probabilities=self.config.collect_probabilities,
                progress_label=progress_label,
            )
            sequences = None
        else:
            ids, predictions, labels, probabilities, sequences = run_inference(
                prepared.model,
                prepared.test_dataloader,
                workspace.task_type,
                accelerator,
                collect_probabilities=self.config.collect_probabilities,
                tokenizer=(
                    None
                    if using_embeddings
                    else (prepared.tokenizer if self.config.store_sequences else None)
                ),
                progress_label=progress_label,
                progress_total_batches=progress_total_batches,
                progress_total_rows=progress_total_rows,
                progress_unit=progress_unit,
            )

        # Keep track of conditions for reproducibility and downstream steps like ScoreStep
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

        if using_embeddings and self.config.store_sequences:
            assert prepared.embedding is not None
            sequence_by_id = self._decode_token_dataloader(
                prepared.embedding.backend.token_test_dataloader,
                prepared.tokenizer,
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
        control_paths: dict[str, Path] = {}
        train_labels = None
        control_builder = ControlBuilder(
            self.config.collect_probabilities,
            embedding_batch_size=self.config.embedding_batch_size,
        )
        for control in self.config.conditions:
            if control in NO_INFERENCE_CONTROLS:
                continue
            if control in BASELINE_NAMES:
                if train_labels is None:
                    train_labels = gather_training_labels(prepared, accelerator)
                control_df = build_prediction_baseline(
                    cast(BaselineName, control),
                    workspace.task_type,
                    workspace.dataset_name,
                    workspace.num_labels,
                    predictions_df,
                    train_labels,
                )
                control_path = None
            else:
                control_df, control_path = control_builder.build(
                    control, workspace, prepared, predictions_df, accelerator, seed
                )
            controls[control] = control_df
            if control_path is not None:
                control_paths[control] = control_path

        # Every rank returns the populated predictions (ScoreStep validates
        # them on every rank); only file writes are gated on the main process.
        output = PredictOutput(
            metadata=metadata,
            predictions=predictions_df,
            controls=controls,
            control_paths=control_paths,
            requested_controls=self.config.conditions,
            config=self.config.model_dump(mode="json"),
            baseline_context=prepared.baseline_context,
        )
        if self.config.persist_to_disk and accelerator.is_main_process:
            persist_predictions(workspace, output, seed)
        return output

    def _decode_token_dataloader(
        self, token_dataloader: Any, tokenizer: Any, accelerator: Accelerator
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
