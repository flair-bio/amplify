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
from typing import Any

import polars as pl
import torch
from accelerate import Accelerator
from pydantic import BaseModel, ConfigDict, Field
from sklearn.base import BaseEstimator

from modules.evaluate.src.dataset.workspace import (
    EvaluationWorkspace,
    get_evaluation_workspace,
)
from modules.evaluate.src.model.heads import (
    CLASSICAL_HEAD_TYPES,
    run_classical_inference,
)
from modules.evaluate.src.model.layerwise import LayerwiseProbeModel
from modules.evaluate.src.schemas.artifacts import (
    EvaluationMetadata,
    PredictOutput,
    PreparedArtifacts,
)
from modules.evaluate.src.tasks import get_task
from modules.evaluate.src.utils.controls import (
    NO_INFERENCE_CONTROLS,
    ControlBuilder,
    ControlCondition,
)
from modules.evaluate.src.utils.embed import find_embedding_dataset
from modules.evaluate.src.utils.inference import (
    run_inference,
    run_local_embedding_inference,
)
from modules.evaluate.src.utils.io import (
    downcast_prediction_dtypes,
    update_run_manifest,
    write_parquet,
)
from modules.evaluate.src.utils.layerwise import (
    run_layerwise_control_conditions,
)

logger = logging.getLogger(__name__)


class PredictConfig(BaseModel):
    """Config. for the predict step in evaluation."""

    model_config = ConfigDict(extra="forbid")

    collect_probabilities: bool = Field(
        False,
        description=(
            "Whether to collect per-example per-class probabilities "
            "for the trained model and any model-based controls. Required "
            "if ScoreConfig.metrics includes 'roc_auc'. When using the "
            "top-level EvaluationConfig, this is auto-enabled when needed."
        ),
    )
    predictions_filename: str = "predictions.parquet"
    persist_to_disk: bool = Field(
        True,
        description=(
            "Write predictions and the predict manifest to disk. Disable for "
            "fast in-process evaluation when ScoreStep receives this step's "
            "PredictOutput directly."
        ),
    )
    store_sequences: bool = Field(
        False,
        description=(
            "Whether to decode and persist input sequences alongside primary "
            "predictions. Kept off by default to reduce prediction-time "
            "memory use and parquet size. This only affects the main "
            "predictions table; the scrambled_sequences control still stores "
            "its transformed sequences regardless."
        ),
    )

    # Control conditions to compute predictions for, so ScoreStep can compare
    # the trained model against them:
    # - "untrained": the pretrained trunk with a freshly-initialized head,
    #   i.e. the model as it was before fine-tuning.
    # - "random_trunk"/"random_head"/"random_both": the trained model with
    #   its trunk/head/both randomly re-initialized (isolates how much of
    #   the trained trunk vs. head drives performance).
    # - "scrambled_labels": no inference needed -- ScoreStep derives this
    #   directly from the trained model's own predictions/labels.
    # - "scrambled_sequences": the trained model's predictions on inputs
    #   whose token order has been shuffled (sequence-level tasks only).
    # - "majority_class": a constant class predictor selected from the
    #   training split's class frequencies (a simple non-model baseline).
    # - "train_mean": a constant regression predictor equal to the
    #   training split's label mean (the regression counterpart to
    #   "majority_class").
    controls: list[ControlCondition] = Field(default_factory=lambda: ["untrained"])


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
        using_embeddings = prepared.embedding_backend is not None
        conditions = run_layerwise_control_conditions(
            prepared, workspace.task_type, accelerator, self.config, seed
        )

        sequence_by_id = None
        if using_embeddings and self.config.store_sequences:
            assert prepared.embedding_backend is not None
            sequence_by_id = self._decode_token_dataloader(
                prepared.embedding_backend.token_test_dataloader,
                prepared.tokenizer,
                accelerator,
            )

        outputs: dict[int, PredictOutput] = {}
        for layer, (ids, predictions, labels, probabilities, sequences) in conditions[
            "trained"
        ].items():
            layer_workspace = get_evaluation_workspace(
                workspace.config, variant=f"{workspace.variant}-layer_{layer}"
            )
            layer_workspace.setup_directories()
            metadata = EvaluationMetadata(
                model_name=workspace.model_name,
                model_repo_id=workspace.model_repo_id,
                dataset_name=workspace.dataset_name,
                dataset_repo_id=workspace.dataset_repo_id,
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
            for control in self.config.controls:
                if control in NO_INFERENCE_CONTROLS:
                    continue
                if control in {"majority_class", "train_mean"}:
                    controls[control], _ = ControlBuilder(
                        self.config.collect_probabilities
                    ).build(
                        control,
                        layer_workspace,
                        prepared,
                        predictions_df,
                        accelerator,
                        seed,
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

            paths: dict[str, Path] = {}
            predictions_path = None
            config_dict = self.config.model_dump(mode="json")
            if self.config.persist_to_disk and accelerator.is_main_process:
                predictions_path = write_parquet(
                    downcast_prediction_dtypes(predictions_df),
                    layer_workspace.preds_path / self.config.predictions_filename,
                )
                for control_name, frame in controls.items():
                    paths[control_name] = write_parquet(
                        downcast_prediction_dtypes(frame),
                        layer_workspace.preds_path
                        / f"control_{control_name}_predictions.parquet",
                    )
                update_run_manifest(
                    layer_workspace.run_manifest_path,
                    "predict",
                    {
                        "seed": seed,
                        "config": config_dict,
                        "metadata": metadata.model_dump(mode="json"),
                        "control_paths": {
                            name: str(path) for name, path in paths.items()
                        },
                    },
                )
            outputs[layer] = PredictOutput(
                metadata=metadata,
                predictions=predictions_df,
                predictions_path=predictions_path,
                controls=controls,
                control_paths=paths,
                requested_controls=list(self.config.controls),
                config=config_dict,
            )
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
        logger.info("Running Predict step...")

        # Guard in case the user misconfigures the pipeline and tries to run PredictStep without a test dataloader (e.g. no 'test' split in the dataset).
        if prepared.test_dataloader is None:
            raise ValueError(
                "PredictStep requires a test_dataloader, but the Prepare "
                "step did not produce one (no 'test' split in the dataset)."
            )

        # The pipeline passes the same instance used by Prepare/Tune, so
        # gather_for_metrics() and is_main_process use one process state.
        if accelerator is None:
            accelerator = Accelerator()

        # steps.prepare.embedding_cache.enabled=True: test_dataloader yields
        # pooled inputs_embeds, not tokens, so sequences can't be decoded
        # from it directly -- decode separately from the retained
        # token-based dataloader and join back onto predictions by id below.
        using_embeddings = prepared.embedding_backend is not None

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
                batch_size = prepared.test_dataloader.batch_size or 16
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
                prepared.test_dataloader.batch_size or 16,
                collect_probabilities=self.config.collect_probabilities,
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
            )

        # Keep track of conditions for reproducibility and downstream steps like ScoreStep
        metadata = EvaluationMetadata(
            model_name=workspace.model_name,
            model_repo_id=workspace.model_repo_id,
            dataset_name=workspace.dataset_name,
            dataset_repo_id=workspace.dataset_repo_id,
            task_type=workspace.task_type,
            split=prepared.split,
            seed=seed,
            head_type=head_type,
            variant=variant,
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

        if using_embeddings and self.config.store_sequences:
            assert prepared.embedding_backend is not None
            sequence_by_id = self._decode_token_dataloader(
                prepared.embedding_backend.token_test_dataloader,
                prepared.tokenizer,
                accelerator,
            )
            predictions_df = predictions_df.join(sequence_by_id, on="id", how="left")

        controls: dict[str, pl.DataFrame] = {}
        control_paths: dict[str, Path] = {}
        control_builder = ControlBuilder(self.config.collect_probabilities)
        for control in self.config.controls:
            if control in NO_INFERENCE_CONTROLS:
                continue
            control_df, control_path = control_builder.build(
                control, workspace, prepared, predictions_df, accelerator, seed
            )
            controls[control] = control_df
            if control_path is None:
                # Non-model baselines aren't cached, so they have no path yet.
                control_path = (
                    workspace.preds_path / f"control_{control}_predictions.parquet"
                )
                if self.config.persist_to_disk and accelerator.is_main_process:
                    write_parquet(downcast_prediction_dtypes(control_df), control_path)
            control_paths[control] = control_path

        # Every rank must return the populated predictions_df (not just the
        # main process): downstream steps like ScoreStep run on every rank
        # too and validate that predictions is non-empty, so returning an
        # empty DataFrame on non-main ranks would fail that check and hang
        # the run under multi-process accelerate launches. Only the actual
        # file writes below are gated on is_main_process, to avoid
        # concurrent writes from multiple ranks.
        predictions_path: Path | None = None
        config_dict = self.config.model_dump(mode="json")
        if self.config.persist_to_disk and accelerator.is_main_process:
            predictions_path = write_parquet(
                downcast_prediction_dtypes(predictions_df),
                workspace.preds_path / self.config.predictions_filename,
            )

            # Record everything ScoreStep needs to run afterwards as a
            # plain, disk-only, rank-0 process (metadata, control cache
            # paths, the settings used) in the run manifest, since parquet
            # has no convenient place to embed this metadata itself.
            update_run_manifest(
                workspace.run_manifest_path,
                "predict",
                {
                    "seed": seed,
                    "config": config_dict,
                    "metadata": metadata.model_dump(mode="json"),
                    "control_paths": {
                        control: str(path) for control, path in control_paths.items()
                    },
                },
            )

        return PredictOutput(
            metadata=metadata,
            predictions=predictions_df,
            predictions_path=predictions_path,
            controls=controls,
            control_paths=control_paths,
            requested_controls=list(self.config.controls),
            config=config_dict,
        )

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
