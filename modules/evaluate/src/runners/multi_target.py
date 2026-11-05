"""Run scalar evaluation over multiple target columns and aggregate their scores."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable
from contextlib import ExitStack
from typing import Any

from accelerate.utils import broadcast_object_list

from modules.evaluate.src.contracts.config import (
    EvaluationConfig,
    MultiTargetConfig,
)
from modules.evaluate.src.dataset.workspace import (
    EvaluationWorkspace,
    get_evaluation_workspace,
)
from modules.evaluate.src.metrics.tables import aggregate_task_scores
from modules.evaluate.src.runners.pipeline import run_pipeline
from modules.evaluate.src.utils.io import update_run_manifest, write_json, write_parquet
from modules.evaluate.src.utils.seed import seed_everything

logger = logging.getLogger(__name__)


def _resume_target(
    config: EvaluationConfig,
    workspace: EvaluationWorkspace,
    metric: str,
) -> bool:
    if not workspace.run_manifest_path.exists():
        return False
    manifest = json.loads(workspace.run_manifest_path.read_text())
    evaluation = manifest.get("evaluation", {})
    if evaluation.get("config") != config.model_dump(mode="json"):
        raise ValueError(
            f"Multi-target outputs have incompatible settings at {workspace.run_root}; "
            "choose a new workspace.run_id or use --overwrite."
        )
    if evaluation.get("status") != "complete":
        return False
    value = manifest.get("score", {}).get("scores", {}).get(metric)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(
            f"Completed multi-target run has invalid {metric}: {workspace.run_root}"
        )
    return True


def run_multi_target_evaluation(
    config: EvaluationConfig,
    accelerator: Any,
    *,
    multi_target: MultiTargetConfig,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Evaluate scalar targets independently, then aggregate one score per target."""
    if config.multi_target != multi_target:
        raise ValueError("The multi-target argument must match the evaluation config.")
    if (
        config.workspace.task_type != "sequence_regression"
        or config.steps.layerwise.enabled
        or config.steps.predict.test_subsets
        or config.external_test_sets
        or config.steps.mode == "evaluate_as_is"
    ):
        raise ValueError(
            "Multi-target mode requires unique scalar target columns and ordinary "
            "sequence-regression probes, without layerwise or additional test sets."
        )
    target_columns = tuple(multi_target.target_columns)
    metric = multi_target.metric

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

    targets = []
    for column in target_columns:
        payload = config.model_dump(mode="json")
        payload["multi_target"] = None
        payload["workspace"]["run_id"] += f"-{column}"
        payload["steps"]["prepare"]["label_column"] = column
        child = EvaluationConfig.model_validate(payload)
        workspace = get_evaluation_workspace(
            child.workspace, variant=child.variant, seed=child.seed
        )
        targets.append((child, workspace))
    summary_workspace = get_evaluation_workspace(
        config.workspace.model_copy(
            update={"run_id": f"{config.workspace.run_id}-summary"}
        ),
        variant=config.variant,
        seed=config.seed,
    )
    with ExitStack() as stack:
        run_targets = on_main(
            lambda: stack.enter_context(summary_workspace.reserve_run(overwrite=True))
        )
        completed = on_main(
            lambda: [
                False if overwrite else _resume_target(child, workspace, metric)
                for child, workspace in targets
            ]
        )
        on_main(
            lambda: summary_workspace.initialize_run(
                run_targets, overwrite=True, config=config.model_dump(mode="json")
            )
        )
        for (child, workspace), done in zip(targets, completed, strict=True):
            if done:
                logger.info(
                    "Reusing completed multi-target target: %s",
                    child.steps.prepare.label_column,
                )
                continue
            logger.info(
                "Evaluating multi-target target: %s",
                child.steps.prepare.label_column,
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
            table, summary = aggregate_task_scores(
                summary_workspace.workspace_root,
                f"{config.variant}-{config.workspace.run_id}",
                config.seed,
                config.workspace.dataset_name,
                target_columns,
                metric,
                target_key=multi_target.target_key,
                count_key=multi_target.count_key,
                aggregation_name=multi_target.aggregation_name,
            )
            write_parquet(
                table, summary_workspace.run_root / multi_target.score_table_filename
            )
            write_json(summary_workspace.run_root / "summary.json", summary)
            update_run_manifest(summary_workspace.run_manifest_path, "score", summary)
            update_run_manifest(
                summary_workspace.run_manifest_path,
                "evaluation",
                {"status": "complete", "config": config.model_dump(mode="json")},
            )
            logger.info(
                "Multi-target mean %s over %d targets: %.6g",
                metric,
                len(target_columns),
                summary["value"],
            )
            return summary

        return on_main(publish)
