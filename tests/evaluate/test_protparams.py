"""Core ProtParams aggregation and resumable suite behavior."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import polars as pl
import pytest
from pydantic import ValidationError

from modules.evaluate.src.contracts.config_loader import load_evaluation_config
from modules.evaluate.src.metrics.tables import aggregate_task_scores
from modules.evaluate.src.runners import multi_target
from modules.evaluate.src.runners.multi_target import run_multi_target_evaluation
from modules.evaluate.src.utils.io import update_run_manifest, write_json

from .test_config_loader import CONFIGS


@pytest.fixture
def suite(tmp_path, monkeypatch):
    config = load_evaluation_config(
        [CONFIGS / "config.yaml", CONFIGS / "datasets/uniprot_protparams.yaml"],
        overrides=["workspace.model_name=toy", f"workspace.output_dir={tmp_path}"],
    )
    target_columns = config.multi_target.target_columns
    calls = []

    def run_pipeline(child, workspace, accelerator, *, overwrite):
        calls.append((child, workspace, overwrite))
        with workspace.reserve_run(overwrite=overwrite) as targets:
            workspace.initialize_run(
                targets, overwrite=overwrite, config=child.model_dump(mode="json")
            )
            prop = child.steps.prepare.label_column
            update_run_manifest(
                workspace.run_manifest_path,
                "score",
                {
                    "metadata": {
                        "model_name": child.workspace.model_name,
                        "dataset_name": child.workspace.dataset_name,
                        "task_type": child.workspace.task_type,
                        "split": child.steps.prepare.split.name,
                        "variant": child.variant,
                        "seed": child.seed,
                    },
                    "scores": {"r2": target_columns.index(prop) - 20.0},
                },
            )
            update_run_manifest(
                workspace.run_manifest_path,
                "evaluation",
                {"status": "complete", "config": child.model_dump(mode="json")},
            )

    monkeypatch.setattr(multi_target, "run_pipeline", run_pipeline)
    accelerator = SimpleNamespace(
        is_main_process=True,
        num_processes=1,
        wait_for_everyone=MagicMock(),
        free_memory=MagicMock(),
    )
    return config, accelerator, calls


def test_suite_isolates_properties_averages_equally_and_resumes(suite):
    config, accelerator, calls = suite
    target_columns = config.multi_target.target_columns
    summary = run_multi_target_evaluation(
        config, accelerator, multi_target=config.multi_target
    )
    assert summary["value"] == pytest.approx(-5.0)
    assert [child.steps.prepare.label_column for child, _, _ in calls] == target_columns
    assert len({workspace.model_path for _, workspace, _ in calls}) == len(target_columns)
    assert config.steps.prepare.label_column == "targets"
    assert config.workspace.run_id == "run01"
    output = calls[0][1].workspace_root / f"{config.variant}-run01-summary/seed_{config.seed}"
    assert pl.read_parquet(output / "property_scores.parquet").height == len(target_columns)
    assert json.loads((output / "summary.json").read_text())["value"] == pytest.approx(-5.0)
    assert (
        run_multi_target_evaluation(
            config, accelerator, multi_target=config.multi_target
        )
        == summary
    )
    assert len(calls) == len(target_columns)

    _, workspace, _ = calls[0]
    manifest = json.loads(workspace.run_manifest_path.read_text())
    manifest["evaluation"]["status"] = "running"
    write_json(workspace.run_manifest_path, manifest)
    run_multi_target_evaluation(config, accelerator, multi_target=config.multi_target)
    assert len(calls) == len(target_columns) + 1
    assert calls[-1][1].run_root == workspace.run_root
    assert calls[-1][2]


@pytest.mark.parametrize("change", ["missing", "nan", "different_protocol"])
def test_summary_rejects_missing_invalid_or_mixed_runs(suite, change):
    config, accelerator, calls = suite
    run_multi_target_evaluation(config, accelerator, multi_target=config.multi_target)
    _, workspace, _ = calls[-1]
    path = workspace.run_manifest_path
    if change == "missing":
        path.unlink()
    else:
        manifest = json.loads(path.read_text())
        if change == "nan":
            manifest["score"]["scores"]["r2"] = float("nan")
        else:
            manifest["evaluation"]["config"]["steps"]["prepare"]["max_length"] += 1
        write_json(path, manifest)
    with pytest.raises((FileNotFoundError, ValueError)):
        aggregate_task_scores(
            workspace.workspace_root,
            f"{config.variant}-run01",
            config.seed,
            config.workspace.dataset_name,
            tuple(config.multi_target.target_columns),
            config.multi_target.metric,
            target_key=config.multi_target.target_key,
            count_key=config.multi_target.count_key,
            aggregation_name=config.multi_target.aggregation_name,
        )
    if change == "different_protocol":
        with pytest.raises(ValueError, match="incompatible settings"):
            run_multi_target_evaluation(
                config, accelerator, multi_target=config.multi_target
            )
        assert len(calls) == len(config.multi_target.target_columns)


def test_suite_overwrite_failure_removes_stale_summary(suite, monkeypatch):
    config, accelerator, calls = suite
    target_count = len(config.multi_target.target_columns)
    run_multi_target_evaluation(config, accelerator, multi_target=config.multi_target)
    run_multi_target_evaluation(
        config, accelerator, multi_target=config.multi_target, overwrite=True
    )
    assert len(calls) == 2 * target_count
    assert all(overwrite for _, _, overwrite in calls[target_count:])

    def fail(*args, **kwargs):
        raise RuntimeError("interrupted property")

    monkeypatch.setattr(multi_target, "run_pipeline", fail)
    with pytest.raises(RuntimeError, match="interrupted property"):
        run_multi_target_evaluation(
            config, accelerator, multi_target=config.multi_target, overwrite=True
        )
    output = calls[0][1].workspace_root / f"{config.variant}-run01-summary/seed_{config.seed}"
    assert not (output / "summary.json").exists()
    assert not (output / "property_scores.parquet").exists()


def test_entry_point_dispatches_configured_multi_target(suite, monkeypatch):
    import sys
    from modules.evaluate.src import run_evaluate

    config, accelerator, _ = suite
    aggregate_runner, single = MagicMock(), MagicMock()
    monkeypatch.setattr(sys, "argv", ["run_evaluate", "config.yaml"])
    monkeypatch.setattr(run_evaluate, "load_evaluation_config", lambda **kwargs: config)
    monkeypatch.setattr(run_evaluate, "Accelerator", lambda **kwargs: accelerator)
    monkeypatch.setattr(
        run_evaluate, "run_multi_target_evaluation", aggregate_runner
    )
    monkeypatch.setattr(run_evaluate, "run_pipeline", single)
    run_evaluate.main()
    aggregate_runner.assert_called_once()
    single.assert_not_called()


def test_legacy_single_property_switch_is_rejected(suite):
    config, _, _ = suite
    payload = config.model_dump()
    payload["protparams_all_properties"] = False
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        type(config).model_validate(payload)
