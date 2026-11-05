"""Core behavior of multi-target score aggregation."""

import json

import pytest

from modules.evaluate.src.metrics.tables import aggregate_task_scores


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
        manifest_path = tmp_path / f"{run_name}-{target}" / "seed_7" / "run_manifest.json"
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

    assert scores["r2"].to_list() == [0.2, 0.8]
    assert summary["value"] == pytest.approx(0.5)
    assert summary["n_targets"] == 2
