"""Sequential ProtParams probes with resumption and automatic aggregation."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable
from contextlib import ExitStack
from typing import Any

from accelerate.utils import broadcast_object_list

from modules.evaluate.src.contracts.config import EvaluationConfig
from modules.evaluate.src.dataset.workspace import (
    EvaluationWorkspace,
    get_evaluation_workspace,
)
from modules.evaluate.src.metrics.tables import aggregate_protparams
from modules.evaluate.src.runners.pipeline import run_pipeline
from modules.evaluate.src.utils.io import update_run_manifest, write_json, write_parquet
from modules.evaluate.src.utils.protparams import PROTPARAM_COLUMNS
from modules.evaluate.src.utils.seed import seed_everything

logger = logging.getLogger(__name__)


def _resume_property(config: EvaluationConfig, workspace: EvaluationWorkspace) -> bool:
    if not workspace.run_manifest_path.exists():
        return False
    manifest = json.loads(workspace.run_manifest_path.read_text())
    evaluation = manifest.get("evaluation", {})
    if evaluation.get("config") != config.model_dump(mode="json"):
        raise ValueError(
            f"ProtParams outputs have incompatible settings at {workspace.run_root}; "
            "choose a new workspace.run_id or use --overwrite."
        )
    if evaluation.get("status") != "complete":
        return False
    value = manifest.get("score", {}).get("scores", {}).get("r2")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(
            f"Completed ProtParams run has invalid R2: {workspace.run_root}"
        )
    return True


def run_protparams(
    config: EvaluationConfig, accelerator: Any, *, overwrite: bool = False
) -> dict[str, Any]:
    """Run all properties through the ordinary pipeline, then publish their mean."""
    if (
        config.workspace.task_type != "sequence_regression"
        or config.workspace.dataset_name != "uniprot_protparams"
        or config.steps.layerwise.enabled
        or config.steps.predict.test_subsets
        or config.external_test_sets
        or config.steps.mode == "evaluate_as_is"
    ):
        raise ValueError(
            "ProtParams all-property mode requires ordinary regression probes "
            "on uniprot_protparams, without layerwise or additional test sets."
        )

    def on_main(action: Callable[[], Any]) -> Any:
        result = error = None
        exception = None
        if accelerator.is_main_process:
            try:
                result = action()
            except Exception as caught:
                exception, error = caught, str(caught)
        messages = [result, error]
        if accelerator.num_processes > 1:
            messages = broadcast_object_list(messages, from_process=0)
        if messages[1] is not None:
            if exception is not None:
                raise exception
            raise RuntimeError(messages[1])
        return messages[0]

    properties = []
    for prop in PROTPARAM_COLUMNS:
        payload = config.model_dump(mode="json")
        payload["workspace"]["run_id"] += f"-{prop}"
        payload["steps"]["prepare"]["label_column"] = prop
        child = EvaluationConfig.model_validate(payload)
        workspace = get_evaluation_workspace(
            child.workspace, variant=child.variant, seed=child.seed
        )
        properties.append((child, workspace))
    summary_workspace = get_evaluation_workspace(
        config.workspace.model_copy(
            update={"run_id": f"{config.workspace.run_id}-summary"}
        ),
        variant=config.variant,
        seed=config.seed,
    )
    with ExitStack() as stack:
        targets = on_main(
            lambda: stack.enter_context(summary_workspace.reserve_run(overwrite=True))
        )
        completed = on_main(
            lambda: [
                False if overwrite else _resume_property(child, workspace)
                for child, workspace in properties
            ]
        )
        on_main(
            lambda: summary_workspace.initialize_run(
                targets, overwrite=True, config=config.model_dump(mode="json")
            )
        )
        for (child, workspace), done in zip(properties, completed, strict=True):
            if done:
                logger.info(
                    "Reusing completed ProtParams property: %s",
                    child.steps.prepare.label_column,
                )
                continue
            logger.info(
                "Evaluating ProtParams property: %s", child.steps.prepare.label_column
            )
            seed_everything(child.seed)
            run_pipeline(
                child,
                workspace,
                accelerator,
                overwrite=overwrite or workspace.run_manifest_path.exists(),
            )
            accelerator.wait_for_everyone()
            accelerator.free_memory()

        def publish() -> dict[str, Any]:
            table, summary = aggregate_protparams(
                summary_workspace.workspace_root,
                f"{config.variant}-{config.workspace.run_id}",
                config.seed,
            )
            write_parquet(table, summary_workspace.run_root / "property_scores.parquet")
            write_json(summary_workspace.run_root / "summary.json", summary)
            update_run_manifest(summary_workspace.run_manifest_path, "score", summary)
            update_run_manifest(
                summary_workspace.run_manifest_path,
                "evaluation",
                {"status": "complete", "config": config.model_dump(mode="json")},
            )
            logger.info("ProtParams mean property R2: %.6g", summary["value"])
            return summary

        return on_main(publish)
