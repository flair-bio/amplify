"""Recover provenance from original data and pinned tokenization, never inference."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from modules.evaluate.src.artifacts.predictions import record_token_positions
from modules.evaluate.src.config import EvaluationConfig
from modules.evaluate.src.dataset.resolution import (
    resolve_evaluation_data,
    resolve_external_data,
)
from modules.evaluate.src.schemas.artifacts import PredictOutput
from modules.evaluate.src.utils.baselines import (
    baseline_context,
    external_baseline_context,
)


def token_population(
    dataset: Any, label_column: str, id_column: str
) -> list[dict[str, Any]]:
    return [
        {
            "id": row[id_column],
            "label": [value for value in row[label_column] if value != -100],
            "evaluated_positions": row["evaluated_positions"],
        }
        for row in dataset
    ]


def recover_prediction_context(
    output: PredictOutput, evaluation: dict[str, Any]
) -> PredictOutput:
    """Reconstruct only from the producing run's saved configuration and data.

    Source fingerprints are checked against any previously recorded context.
    Legacy token artifacts without positions require an exact population match.
    """
    config = EvaluationConfig.model_validate(evaluation)
    prepare, workspace = config.steps.prepare, config.workspace
    if output.metadata.task_type != workspace.task_type:
        raise ValueError("Saved evaluation configuration has a different task.")
    data = resolve_evaluation_data(
        workspace.dataset_repo_id or workspace.dataset_name,
        split=prepare.split,
        seed=config.seed,
        validation_fraction=prepare.validation_fraction,
        revision=workspace.dataset_revision,
        subset=workspace.dataset_subset,
    )
    context = baseline_context(
        workspace,
        data,
        [],
        sequence_column=prepare.sequence_column,
        label_column=prepare.label_column,
        id_column=prepare.id_column,
        split_roles=prepare.split.model_dump(),
    )
    if output.metadata.split != prepare.split.name:
        external = next(
            (
                item
                for item in config.external_test_sets
                if item.name == output.metadata.split
            ),
            None,
        )
        if external is None:
            raise ValueError(
                "Saved split is not present in the producing evaluation configuration."
            )
        data = resolve_external_data(
            external.dataset_repo_id,
            external.split,
            revision=external.dataset_revision,
            subset=external.dataset_subset,
        )
        context = external_baseline_context(
            context,
            workspace,
            data,
            dataset_id=external.dataset_repo_id,
            split=external.split,
            sequence_column=prepare.sequence_column,
            label_column=prepare.label_column,
            id_column=prepare.id_column,
            dataset_revision=external.dataset_revision,
            dataset_subset=external.dataset_subset,
        )
    if output.baseline_context is not None and any(
        output.baseline_context[key] != context[key] for key in ("fit", "test")
    ):
        raise ValueError(
            "Original data no longer matches saved dataset/split/label fingerprints."
        )
    if workspace.task_type == "token_classification":
        from transformers import AutoTokenizer

        from modules.evaluate.src.dataset.dataloader import tokenize_dataset

        tokenizer = AutoTokenizer.from_pretrained(
            workspace.model_repo_id or workspace.model_name,
            trust_remote_code=True,
            **(
                {"revision": workspace.model_revision}
                if workspace.model_revision
                else {}
            ),
        )
        tokenized = tokenize_dataset(
            data,
            tokenizer,
            workspace.task_type,
            prepare.sequence_column,
            prepare.label_column,
            prepare.max_length,
            id_column=prepare.id_column,
            num_proc=prepare.tokenize_num_proc,
            random_truncate_by_split=prepare.random_truncate_by_split,
            seed=config.seed,
        )
        context["population"] = token_population(
            tokenized["test"], prepare.label_column, prepare.id_column
        )
    frame = record_token_positions(
        output.predictions.drop("evaluated_positions", strict=False), context
    )
    if (
        workspace.task_type == "token_classification"
        and "evaluated_positions" not in frame.columns
    ):
        raise ValueError(
            "Cannot recover exact token crop/mask coordinates from the original population."
        )
    if "evaluated_positions" in output.predictions.columns and (
        output.predictions["evaluated_positions"].to_list()
        != frame["evaluated_positions"].to_list()
    ):
        raise ValueError(
            "Saved residue positions do not match the producing tokenization configuration."
        )
    controls = dict(output.controls)
    if "evaluated_positions" in frame.columns:
        for name, control in controls.items():
            if control["id"].to_list() != frame["id"].to_list() or (
                control["prediction"].list.len().to_list()
                != frame["prediction"].list.len().to_list()
            ):
                raise ValueError(
                    f"Cannot recover exact token alignment for control '{name}'."
                )
            controls[name] = control.with_columns(frame["evaluated_positions"])
    return replace(
        output, predictions=frame, controls=controls, baseline_context=context
    )
