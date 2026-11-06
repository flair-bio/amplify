"""Core behavior of multi-target score aggregation."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import polars as pl
import pytest

from modules.evaluate.src.contracts.config import EvaluationConfig
from modules.evaluate.src.dataset.workspace import get_evaluation_workspace
from modules.evaluate.src.metrics.tables import aggregate_task_scores
from modules.evaluate.src.runners import multi_target as runner


def test_aggregate_task_scores_averages_target_scores_equally(tmp_path):
    run_name = "mlp-frozen_trunk-smoke"
    targets = ("target_a", "target_b")
    values = {"target_a": 0.2, "target_b": 0.8}
    for target, score in values.items():
        run_id = f"smoke-{target}"
        manifest = {
            "evaluation": {
                "status": "complete",
                "config": {
                    "seed": 7,
                    "workspace": {
                        "dataset_name": "toy_dataset",
                        "task_type": "sequence_regression",
                        "model_name": "toy_model",
                        "run_id": run_id,
                    },
                    "steps": {"prepare": {"label_column": target}},
                },
            },
            "score": {
                "metadata": {
                    "seed": 7,
                    "dataset_name": "toy_dataset",
                    "task_type": "sequence_regression",
                    "model_name": "toy_model",
                    "variant": "mlp-frozen_trunk",
                },
                "scores": {"r2": score},
            },
        }
        manifest_path = (
            tmp_path / f"{run_name}-{target}" / "seed_7" / "run_manifest.json"
        )
        manifest_path.parent.mkdir(parents=True)
        manifest_path.write_text(json.dumps(manifest))

    scores, summary = aggregate_task_scores(
        tmp_path,
        run_name,
        7,
        "toy_dataset",
        targets,
        "r2",
    )

    assert scores["target"].to_list() == [
        "target_a",
        "target_b",
        "unweighted_target_mean",
    ]
    assert scores["r2"].to_list() == [0.2, 0.8, 0.5]
    assert scores["manifest_path"][-1] is None
    assert summary["value"] == pytest.approx(0.5)
    assert summary["n_targets"] == 2


def test_multi_target_model_publishes_scores_and_manifest_and_resumes(
    tmp_path, monkeypatch
):
    config = EvaluationConfig.model_validate(
        {
            "seed": 7,
            "workspace": {
                "run_id": "test",
                "dataset_name": "toy_dataset",
                "model_name": "toy_model",
                "task_type": "sequence_regression",
                "target_type": "continuous",
                "label_format": "scalar",
                "output_dir": str(tmp_path),
            },
            "multi_target": {"target_columns": ["a", "b"], "metric": "r2"},
        }
    )
    multi_target = config.multi_target
    assert multi_target is not None
    accelerator = SimpleNamespace(is_main_process=True, num_processes=1)
    for target in multi_target.target_columns:
        payload = config.model_dump(mode="json")
        payload["multi_target"] = None
        payload["workspace"]["run_id"] += f"-{target}"
        payload["steps"]["prepare"]["label_column"] = target
        child = EvaluationConfig.model_validate(payload)
        workspace = get_evaluation_workspace(child.workspace, child.variant, child.seed)
        workspace.run_root.mkdir(parents=True)
        workspace.run_manifest_path.write_text(
            json.dumps(
                {
                    "evaluation": {
                        "status": "complete",
                        "config": child.model_dump(mode="json"),
                    },
                    "score": {"scores": {"r2": 0.5}},
                }
            )
        )
    pipeline = Mock()
    monkeypatch.setattr(runner, "run_pipeline", pipeline)
    scores = pl.DataFrame(
        {
            "target": ["a", "b", "unweighted_target_mean"],
            "r2": [0.2, 0.8, 0.5],
            "manifest_path": ["a.json", "b.json", None],
        }
    )
    monkeypatch.setattr(
        runner,
        "aggregate_task_scores",
        lambda *args, **kwargs: (scores, {"value": 0.5}),
    )
    summary = runner.run_multi_target_evaluation(
        config, accelerator, multi_target=multi_target
    )
    workspace = get_evaluation_workspace(
        config.workspace.model_copy(update={"run_id": "test-summary"}),
        variant=config.variant,
        seed=config.seed,
    )
    assert {path.name for path in workspace.run_root.iterdir()} == {
        "scores.parquet",
        "run_manifest.json",
    }
    table = pl.read_parquet(workspace.run_root / "scores.parquet")
    assert table.equals(scores)
    manifest = json.loads(workspace.run_manifest_path.read_text())
    assert manifest["evaluation"]["status"] == "complete"
    assert manifest["score"] == summary

    assert (
        runner.run_multi_target_evaluation(
            config, accelerator, multi_target=multi_target
        )
        == summary
    )
    pipeline.assert_not_called()
    assert {path.name for path in workspace.run_root.iterdir()} == {
        "scores.parquet",
        "run_manifest.json",
    }
