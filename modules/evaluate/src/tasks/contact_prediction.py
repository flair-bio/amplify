"""CASP contact capabilities, coordinate-aware predictions and scoring."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import torch

from modules.evaluate.src.artifacts.results import (
    ContactEvaluationMetadata,
    ContactScoreOutput,
    PredictOutput,
)
from modules.evaluate.src.dataset.contact import (
    PROTOCOL_VERSION,
    SCHEMA_VERSION,
    ContactChain,
    ContactDataset,
)
from modules.evaluate.src.metrics import contact_metrics
from modules.evaluate.src.trainers.contact import BILINEAR_PROTOCOL

if TYPE_CHECKING:
    from modules.evaluate.src.artifacts.results import EvaluationMetadata
    from modules.evaluate.src.config import EvaluationConfig, ScoreConfig
class_labels = None
uses_predictor_identity = True
supports_packing = False
supports_layerwise = False
fixed_protocol = True
evaluation_modes = frozenset({"evaluate_as_is", "tune_probe", "finetune"})
supported_head_types = frozenset(
    {"bilinear512", "attention_l1", "categorical_jacobian"}
)
supported_controls = frozenset({"random_trunk"})
dataset_baselines = frozenset({"separation_prior"})
prediction_baselines: frozenset[str] = frozenset()
label_dtype = torch.int8
is_ragged = True
pairwise = True
aligns_labels = True
prediction_progress_unit = "examples"
requires_source_labels = True
tokenization_batch_size = 1000
accepted_target_types = frozenset({"categorical"})
accepted_label_formats = frozenset({"contact_pairs"})
metrics = {**contact_metrics.PRECISION_METRICS, **contact_metrics.RANGE_AUC_METRICS}
metrics_requiring_probabilities: frozenset[str] = frozenset()
default_metric_names = (
    *contact_metrics.PRECISION_METRICS,
    *contact_metrics.RANGE_AUC_METRICS,
)
name = "contact_prediction"
MIN_SEP = 6


def validate_head(head_type: str) -> None:
    if head_type not in supported_head_types:
        raise ValueError(
            f"head_type='{head_type}' is not supported for task_type='{name}'."
        )


def validate_dataset_config(
    target_type: str, label_format: str, dataset_name: str
) -> None:
    if not requires_source_labels:
        return
    if target_type not in accepted_target_types:
        raise ValueError(
            f"task_type='{name}' is incompatible with dataset '{dataset_name}': expected target_type in {sorted(accepted_target_types)}, got '{target_type}'."
        )
    if label_format not in accepted_label_formats:
        raise ValueError(
            f"task_type='{name}' is incompatible with dataset '{dataset_name}': expected label_format in {sorted(accepted_label_formats)}, got '{label_format}'."
        )


def validate_pipeline(
    config: EvaluationConfig, *, workspace_seed: int | None, num_processes: int
) -> None:
    if num_processes != 1:
        raise ValueError("Contact embeddings require a single-process fixed-split run")
    if workspace_seed != config.seed:
        raise ValueError("Contact workspace must be created with the configured seed")


def validate_config(config: EvaluationConfig) -> None:
    """Guard unsupported execution capabilities and ineffective overrides."""
    steps = config.steps
    tune = steps.tune
    validate_dataset_config(
        config.workspace.target_type,
        config.workspace.label_format,
        config.workspace.dataset_name,
    )
    if config.external_test_sets or steps.layerwise.enabled or steps.predict.baselines:
        raise ValueError(
            "Contact prediction requires fixed CASP splits, without generic external tests/layers/inline baselines"
        )
    if steps.prepare.packed or steps.prepare.split.model_dump() != {
        "name": "default",
        "train": "train",
        "validation": "validation",
        "test": "test",
        "refit_additional_splits": [],
    }:
        raise ValueError(
            "Contact prediction requires unchanged fixed splits and unpadded full chains"
        )
    unsupported_controls = set[str](steps.predict.controls) - supported_controls
    if unsupported_controls:
        raise ValueError(
            f"Contact prediction supports only the random_trunk control; got {sorted(unsupported_controls)}"
        )
    if steps.predict.collect_probabilities:
        raise ValueError("CASP contact prediction does not collect probabilities")
    if steps.score.bootstrap_enabled:
        raise ValueError("Contact bootstrap requires complex-level resampling")
    if steps.score.metrics is not None and set(steps.score.metrics) != set(
        contact_metrics.CASP_METRICS
    ):
        raise ValueError("Contact prediction reports fixed CASP metrics")
    predictor = tune.head.head_type
    validate_head(predictor)
    expected_mode = (
        "evaluate_as_is" if predictor == "categorical_jacobian" else "tune_probe"
    )
    if steps.mode != expected_mode:
        raise ValueError(
            f"Contact predictor '{predictor}' requires steps.mode='{expected_mode}'"
        )
    if steps.prepare.embedding_cache.enabled:
        raise ValueError(
            "Contact predictors do not use the generic pooled embedding cache"
        )
    if (
        tune.hyperparameter_search.enabled
        or tune.refit_on_train_validation
        or tune.variance_seeds
        or tune.head.hyperparameters
        or tune.head.search_space
        or config.workspace.model_kwargs
    ):
        raise ValueError(
            "Contact predictors use fixed protocols without HPO, refit, seed sweeps or head/model overrides"
        )
    if predictor != "categorical_jacobian" and tune.standardize_features:
        raise ValueError("Contact predictors do not standardize features")
    if predictor == "bilinear512":
        supplied = tune.model_dump()
        conflicts = [
            name
            for name, value in asdict(BILINEAR_PROTOCOL).items()
            if supplied[name] != value
        ]
        if conflicts:
            raise ValueError(
                f"The bilinear512 protocol fixes these settings: {', '.join(conflicts)}. Only max_epochs is a fitting override."
            )
        if tune.optimizer_kwargs or tune.scheduler_kwargs:
            raise ValueError(
                "The bilinear512 protocol has no optimizer/scheduler overrides"
            )
        if tune.max_epochs <= BILINEAR_PROTOCOL.early_stopping_patience:
            raise ValueError("Contact epoch limit must allow validation patience 10")


def resolve_num_labels(num_labels: int | None, dataset_name: str) -> int:
    return 2


def prediction_metadata(values: Mapping[str, Any]) -> EvaluationMetadata:
    return ContactEvaluationMetadata.model_validate(values)


def score(
    predictions: PredictOutput, config: ScoreConfig, *, seed: int = 710019
) -> ContactScoreOutput:
    if config.bootstrap_enabled:
        raise ValueError("CASP bootstrap must resample complexes, not flat rows")
    if config.metrics is not None and set(config.metrics) != set(
        contact_metrics.CASP_METRICS
    ):
        raise ValueError("Contact scoring reports the fixed CASP metrics")
    return score_units(predictions)


def predict_units(
    dataset: ContactDataset,
    predictor: Callable[[ContactChain], np.ndarray | torch.Tensor],
    metadata: EvaluationMetadata,
    *,
    split: str = "test",
) -> PredictOutput:
    """Coordinate full-chain extraction and preserve every unit's identity."""
    from modules.evaluate.src.model.inference import run_contact_inference

    if metadata.seed is None:
        raise ValueError("Contact predictions require an explicit run seed")
    contact_metadata = ContactEvaluationMetadata.model_validate(
        {
            **metadata.model_dump(),
            "dataset_revision": dataset.revision,
            "dataset_fingerprint": dataset.fingerprint,
            "schema_version": SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
        }
    )
    rows_by_chain = {}
    for chain, scores in run_contact_inference(dataset, predictor, split=split):
        rows_by_chain[chain.record_id] = [
            dict(
                id=unit.id,
                record_id=chain.record_id,
                unit_id=unit.unit_id,
                split=chain.split,
                source=chain.source,
                edition=chain.edition,
                complex_id=chain.complex_id,
                target=chain.target,
                kind=unit.kind,
                labels_available=unit.labels_available,
                **contact_metrics.contact_prediction_row(scores, unit, chain.alignment),
            )
            for unit in chain.units
        ]
    rows = [
        row for chain in dataset.splits[split] for row in rows_by_chain[chain.record_id]
    ]
    if not rows:
        raise ValueError(f"No contact scoring rows in split {split}")
    frame = pl.DataFrame(rows, infer_schema_length=None).with_columns(
        pl.col("original_i", "original_j").cast(pl.List(pl.Int32)),
        pl.col("labels").cast(pl.List(pl.Int8)),
        pl.col("scores").cast(pl.List(pl.Float64)),
    )
    return PredictOutput(metadata=contact_metadata, predictions=frame)


def score_units(predictions: PredictOutput) -> ContactScoreOutput:
    """Score all CASP settings, then aggregate EUs, chains and complexes."""
    metadata = predictions.metadata
    if (
        not isinstance(metadata, ContactEvaluationMetadata)
        or metadata.schema_version != SCHEMA_VERSION
        or metadata.protocol_version != PROTOCOL_VERSION
        or (not metadata.dataset_fingerprint)
        or (metadata.seed is None)
    ):
        raise ValueError("Contact scoring requires schema-6.0 CASP prediction metadata")
    if set(predictions.requested_controls) != set(predictions.controls):
        raise ValueError("Every requested CASP control requires prediction rows")
    if set(predictions.controls) - (supported_controls | dataset_baselines):
        raise ValueError(
            "CASP contact prediction supports only random_trunk and the configured dataset baselines"
        )
    if predictions.predictions.is_empty():
        raise ValueError("Contact scoring requires prediction rows")
    payload_columns = {
        "original_i",
        "original_j",
        "labels",
        "scores",
        "length",
        "status",
    }
    identity = {
        key: value
        for key, value in metadata.model_dump().items()
        if key not in {"split", "dataset_name", "dataset_repo_id"}
    }

    def score_rows(
        frame: pl.DataFrame, control: str | None = None
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        rows = []
        seen = set()
        for row in frame.to_dicts():
            if row["id"] in seen:
                raise ValueError(f"Duplicate scoring id: {row['id']}")
            seen.add(row["id"])
            for minsep in contact_metrics.CASP_SEPARATIONS:
                scored = dict(
                    {
                        key: value
                        for key, value in row.items()
                        if key not in payload_columns
                    },
                    **identity,
                    minsep=minsep,
                    **contact_metrics.score_contact_prediction(row, minsep),
                )
                if control is not None:
                    scored["control"] = control
                rows.append(scored)
        summaries = contact_metrics.aggregate_contact_scores(rows)
        if control is not None:
            for summary in summaries:
                summary["control"] = control
        return (
            pl.DataFrame(rows, infer_schema_length=None),
            pl.DataFrame(summaries, infer_schema_length=None),
        )

    unit_scores, summaries = score_rows(predictions.predictions)
    control_scores = {}
    control_summaries = {}
    primary_by_id = {row["id"]: row for row in predictions.predictions.to_dicts()}
    for name, frame in predictions.controls.items():
        if set(frame["id"].to_list()) != set(primary_by_id):
            raise ValueError(f"{name} control rows do not match primary prediction IDs")
        for row in frame.to_dicts():
            primary = primary_by_id[row["id"]]
            for key in ("original_i", "original_j", "labels", "length", "status"):
                if row[key] != primary[key]:
                    raise ValueError(
                        f"{name} control candidates differ from primary predictions"
                    )
        control_scores[name], control_summaries[name] = score_rows(frame, name)
    return ContactScoreOutput(
        metadata,
        unit_scores,
        summaries,
        control_unit_scores=control_scores,
        control_summaries=control_summaries,
    )


def default_metrics() -> list[str]:
    return list(contact_metrics.CASP_METRICS)


def validate_metric_names(metric_names: Sequence[str]) -> None:
    unknown = set(metric_names) - set(contact_metrics.CASP_METRICS)
    if unknown:
        raise KeyError(f"Unknown CASP contact metrics: {sorted(unknown)}")
