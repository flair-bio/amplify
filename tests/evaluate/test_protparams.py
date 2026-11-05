"""Core ProtParams aggregation and resumable suite behavior."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import polars as pl
import pytest
from pydantic import ValidationError

from modules.evaluate.src.contracts.config_loader import load_evaluation_config
from modules.evaluate.src.metrics.tables import aggregate_protparams
from modules.evaluate.src.runners import protparams
from modules.evaluate.src.utils.io import update_run_manifest, write_json
from modules.evaluate.src.utils.protparams import PROTPARAM_COLUMNS

from .test_config_loader import CONFIGS


@pytest.fixture
def suite(tmp_path, monkeypatch):
    config = load_evaluation_config(
        [CONFIGS / "config.yaml", CONFIGS / "datasets/uniprot_protparams.yaml"],
        overrides=["workspace.model_name=toy", f"workspace.output_dir={tmp_path}"],
    )
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
                    "scores": {"r2": PROTPARAM_COLUMNS.index(prop) - 20.0},
                },
            )
            update_run_manifest(
                workspace.run_manifest_path,
                "evaluation",
                {"status": "complete", "config": child.model_dump(mode="json")},
            )

    monkeypatch.setattr(protparams, "run_pipeline", run_pipeline)
    accelerator = SimpleNamespace(
        is_main_process=True,
        num_processes=1,
        wait_for_everyone=MagicMock(),
        free_memory=MagicMock(),
    )
    return config, accelerator, calls


def test_suite_isolates_properties_averages_equally_and_resumes(suite):
    config, accelerator, calls = suite
    summary = protparams.run_protparams(config, accelerator)
    assert summary["value"] == pytest.approx(-5.0)
    assert [child.steps.prepare.label_column for child, _, _ in calls] == list(PROTPARAM_COLUMNS)
    assert len({workspace.model_path for _, workspace, _ in calls}) == len(PROTPARAM_COLUMNS)
    assert config.steps.prepare.label_column == "gravy"
    assert config.workspace.run_id == "run01"
    output = calls[0][1].workspace_root / f"{config.variant}-run01-summary/seed_{config.seed}"
    assert pl.read_parquet(output / "property_scores.parquet").height == len(PROTPARAM_COLUMNS)
    assert json.loads((output / "summary.json").read_text())["value"] == pytest.approx(-5.0)
    assert protparams.run_protparams(config, accelerator) == summary
    assert len(calls) == len(PROTPARAM_COLUMNS)

    _, workspace, _ = calls[0]
    manifest = json.loads(workspace.run_manifest_path.read_text())
    manifest["evaluation"]["status"] = "running"
    write_json(workspace.run_manifest_path, manifest)
    protparams.run_protparams(config, accelerator)
    assert len(calls) == len(PROTPARAM_COLUMNS) + 1
    assert calls[-1][1].run_root == workspace.run_root
    assert calls[-1][2]


@pytest.mark.parametrize("change", ["missing", "nan", "different_protocol"])
def test_summary_rejects_missing_invalid_or_mixed_runs(suite, change):
    config, accelerator, calls = suite
    protparams.run_protparams(config, accelerator)
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
        aggregate_protparams(
            workspace.workspace_root, f"{config.variant}-run01", config.seed
        )
    if change == "different_protocol":
        with pytest.raises(ValueError, match="incompatible settings"):
            protparams.run_protparams(config, accelerator)
        assert len(calls) == len(PROTPARAM_COLUMNS)


def test_suite_overwrite_failure_removes_stale_summary(suite, monkeypatch):
    config, accelerator, calls = suite
    protparams.run_protparams(config, accelerator)
    protparams.run_protparams(config, accelerator, overwrite=True)
    assert len(calls) == 2 * len(PROTPARAM_COLUMNS)
    assert all(overwrite for _, _, overwrite in calls[len(PROTPARAM_COLUMNS):])

    def fail(*args, **kwargs):
        raise RuntimeError("interrupted property")

    monkeypatch.setattr(protparams, "run_pipeline", fail)
    with pytest.raises(RuntimeError, match="interrupted property"):
        protparams.run_protparams(config, accelerator, overwrite=True)
    output = calls[0][1].workspace_root / f"{config.variant}-run01-summary/seed_{config.seed}"
    assert not (output / "summary.json").exists()
    assert not (output / "property_scores.parquet").exists()


def test_entry_point_always_dispatches_protparams_suite(suite, monkeypatch):
    import sys
    from modules.evaluate.src import run_evaluate

    config, accelerator, _ = suite
    suite_runner, single = MagicMock(), MagicMock()
    monkeypatch.setattr(sys, "argv", ["run_evaluate", "config.yaml"])
    monkeypatch.setattr(run_evaluate, "load_evaluation_config", lambda **kwargs: config)
    monkeypatch.setattr(run_evaluate, "Accelerator", lambda **kwargs: accelerator)
    monkeypatch.setattr(run_evaluate, "run_protparams", suite_runner)
    monkeypatch.setattr(run_evaluate, "run_pipeline", single)
    run_evaluate.main()
    suite_runner.assert_called_once()
    single.assert_not_called()


def test_legacy_single_property_switch_is_rejected(suite):
    config, _, _ = suite
    payload = config.model_dump()
    payload["protparams_all_properties"] = False
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        type(config).model_validate(payload)
