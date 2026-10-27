"""Sequence baselines run independently and are consumed without refitting."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import polars as pl
import pytest
from datasets import Dataset, DatasetDict

from modules.evaluate.src.config import LayerwiseConfig
from modules.evaluate.src.dataset import resolution
from modules.evaluate.src.dataset.dataloader import ExternalTestSetConfig
from modules.evaluate.src.dataset.workspace import (
    DatasetWorkspaceConfig,
    get_evaluation_workspace,
)
from modules.evaluate.src.runners import baselines as runner
from modules.evaluate.src.runners.baselines import BaselineRunConfig, run_baselines
from modules.evaluate.src.schemas.artifacts import EvaluationMetadata, PredictOutput
from modules.evaluate.src.steps.predict import (
    PredictConfig,
    PredictStep,
    persist_predictions,
)
from modules.evaluate.src.steps.score import ScoreConfig, ScoreStep
from modules.evaluate.src.utils.baselines import (
    align_baseline,
    artifact_directory,
    baseline_context,
    load_baseline_artifact,
)

from .conftest import build_raw_dataset_dict
from .test_prepare import _make_workspace, _run_prepare


def _data(regression=False):
    train = [-3.0, 1.0, 8.0] if regression else [1, 1, 0]
    test = [0.1, 2.2, -1.3] if regression else [0, 1, 0]
    return DatasetDict(
        {
            "train": Dataset.from_dict(
                {"sequence": ["AAA", "CCC", "DDD"], "targets": train}
            ),
            "validation": Dataset.from_dict(
                {"sequence": ["WWW"], "targets": [100.0] if regression else [0]}
            ),
            "test": Dataset.from_dict(
                {
                    "sequence": ["ACE", "DFG", "HI"],
                    "targets": test,
                    "id": ["repeat", "repeat", "other"],
                }
            ),
        }
    )


def _config(tmp_path, regression=False):
    return BaselineRunConfig.model_validate(
        {
            "workspace": {
                "dataset_name": "toy",
                "task_type": (
                    "sequence_regression" if regression else "sequence_classification"
                ),
                "target_type": "continuous" if regression else "categorical",
                "label_format": "scalar",
                "num_labels": None if regression else 3,
                "base_path": str(tmp_path),
            },
            "baselines": {"names": ["train_mean" if regression else "majority_class"]},
        }
    )


def _context(config, data):
    return baseline_context(
        config.workspace,
        data,
        config.baselines.names,
        sequence_column=config.prepare.sequence_column,
        label_column=config.prepare.label_column,
        id_column=config.prepare.id_column,
        split_roles=config.prepare.split.model_dump(),
    )


@pytest.mark.parametrize("regression", [False, True])
def test_standalone_baseline_reuses_predictions_and_never_fits_validation(
    tmp_path, monkeypatch, regression, caplog
):
    caplog.set_level("INFO")
    config, data = _config(tmp_path, regression), _data(regression)
    outputs = run_baselines(config, dataset=data)
    context = _context(config, data)
    name = config.baselines.names[0]
    baseline, path = load_baseline_artifact(context, name)
    assert outputs["test"][name] == path
    assert path.is_relative_to(config.workspace.dataset_root / "baselines" / name)
    assert path.parent.name.startswith("test-")
    assert path.parent.parent.name.startswith("fit-")
    assert (path.parent / "log").read_text()
    assert (path.parent.parent / "fit_manifest.json").exists()
    assert baseline["id"].to_list() == ["repeat", "repeat", "other"]
    if regression:
        assert baseline["prediction"].to_list() == [2.0] * 3
    else:
        assert baseline["prediction"].to_list() == [1] * 3
        assert baseline["probability"][0].to_list() == pytest.approx([1 / 3, 2 / 3, 0])
    builder = Mock(side_effect=AssertionError("must not refit"))
    monkeypatch.setattr(runner, "build_sequence_baseline", builder)
    assert run_baselines(config, dataset=data) == outputs
    builder.assert_not_called()


@pytest.mark.parametrize("role", ["train", "test"])
def test_short_hash_collisions_reject_different_full_identities(
    tmp_path, monkeypatch, role
):
    from modules.evaluate.src.utils import baselines

    original_hash = baselines.identity_hash
    monkeypatch.setattr(
        baselines, "identity_hash", lambda value: "0" * 12 + original_hash(value)[12:]
    )
    config, data = _config(tmp_path), _data()
    path = run_baselines(config, dataset=data)["test"]["majority_class"]
    before = path.read_bytes()
    data[role] = data[role].map(
        lambda row, index: {"targets": 2 if index == 0 else row["targets"]},
        with_indices=True,
    )
    with pytest.raises(ValueError, match="Incompatible baseline"):
        run_baselines(config, dataset=data)
    assert path.read_bytes() == before


@pytest.mark.parametrize("change", ["train", "test", "split", "task_classes"])
def test_identity_changes_with_data_and_protocol(tmp_path, change):
    config, data = _config(tmp_path), _data()
    context = _context(config, data)
    if change in {"train", "test"}:
        data[change] = data[change].map(
            lambda row, index: {"targets": 2 if index == 0 else row["targets"]},
            with_indices=True,
        )
    elif change == "split":
        config.prepare.split.train = "another_source"
    else:
        config.workspace.num_labels = 4
    changed = _context(config, data)
    assert artifact_directory(context, "majority_class") != artifact_directory(
        changed, "majority_class"
    )


def test_external_baseline_keeps_source_training_prior(tmp_path, monkeypatch):
    config, data = _config(tmp_path), _data()
    config = BaselineRunConfig.model_validate(
        {
            **config.model_dump(),
            "external_test_sets": [
                {"name": "heldout", "dataset_repo_id": "org/heldout"}
            ],
        }
    )
    external = DatasetDict(
        {
            "holdout": Dataset.from_dict(
                {"sequence": ["WWW"], "targets": [0], "id": ["external"]}
            )
        }
    )
    config.external_test_sets[0].split = "holdout"
    monkeypatch.setattr(resolution, "load_raw_evaluation_dataset", lambda _: external)
    paths = run_baselines(config, dataset=data)
    frame = pl.read_parquet(paths["external/heldout"]["majority_class"])
    assert frame["prediction"].to_list() == [1]
    assert frame["id"].to_list() == ["external"]
    manifest = json.loads(
        paths["external/heldout"]["majority_class"]
        .with_name("manifest.json")
        .read_text()
    )
    assert manifest["identity"]["test"]["dataset"] == "org/heldout"
    assert manifest["identity"]["fit"] == _context(config, data)["fit"]


@pytest.mark.parametrize("regression", [False, True])
def test_score_loads_shared_predictions_and_standalone_manifest(
    tmp_path, monkeypatch, regression
):
    config, data = _config(tmp_path, regression), _data(regression)
    run_baselines(config, dataset=data)
    context = _context(config, data)
    name = config.baselines.names[0]
    for model in ("first", "second"):
        workspace = _make_workspace(tmp_path, config.workspace.task_type, num_labels=3)
        workspace.config.base_path = str(tmp_path)
        workspace = get_evaluation_workspace(
            workspace.config.model_copy(
                update={"model_name": model, "model_repo_id": None}
            ),
            variant="linear",
        )
        rows = data["test"]
        frame = pl.DataFrame(
            {
                "id": rows["id"],
                "label": rows["targets"],
                "prediction": [0.0, 1.0, 2.0] if regression else [0, 1, 2],
            }
        ).reverse()
        output = PredictOutput(
            metadata=EvaluationMetadata(
                model_name=model,
                dataset_name="toy",
                task_type=config.workspace.task_type,
                variant="linear",
            ),
            predictions=frame,
            baseline_context=context,
            config=PredictConfig(controls=[]).model_dump(mode="json"),
        )
        persist_predictions(workspace, output, seed=1)
        result = ScoreStep(
            ScoreConfig(
                metrics=["mse", "spearmanr"] if regression else ["f1"],
                bootstrap_enabled=True,
                n_samples=5,
                n_permutations=5,
            )
        ).run(workspace)
        assert name in result.control_comparisons
        assert not (
            workspace.preds_path / f"control_{name}_predictions.parquet"
        ).exists()
        score_manifest = json.loads(workspace.run_manifest_path.read_text())["score"]
        conditions = pl.read_parquet(workspace.run_root / "condition_scores.parquet")
        assert set(conditions["condition"]) == {"trained", name}
        assert conditions.filter(pl.col("condition") == "trained")[
            "score"
        ].to_list() == list(result.scores.values())
        assert score_manifest["baselines"][name]["identity"]["fit"]["fit_policy"] == (
            "train_only"
        )
        if regression:
            assert np.isnan(score_manifest["baselines"][name]["scores"]["spearmanr"])
            baseline_scores = conditions.filter(
                (pl.col("condition") == name) & (pl.col("metric") == "spearmanr")
            )
            assert np.isnan(baseline_scores["score"][0])
            assert "spearmanr" not in result.control_comparisons[name]


def test_missing_and_corrupt_artifacts_fail_without_building(tmp_path, monkeypatch):
    config, data = _config(tmp_path), _data()
    context = _context(config, data)
    with pytest.raises(FileNotFoundError, match="--baselines"):
        load_baseline_artifact(context, "majority_class")
    path = run_baselines(config, dataset=data)["test"]["majority_class"]
    path.write_bytes(b"incomplete parquet")
    monkeypatch.setattr(
        runner,
        "build_sequence_baseline",
        Mock(side_effect=AssertionError("must not silently rebuild corruption")),
    )
    with pytest.raises(ValueError, match="Corrupt"):
        run_baselines(config, dataset=data)


def test_concurrent_baseline_jobs_publish_one_complete_artifact(tmp_path, monkeypatch):
    config, data = _config(tmp_path), _data()
    builder = Mock(wraps=runner.build_sequence_baseline)
    monkeypatch.setattr(runner, "build_sequence_baseline", builder)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(run_baselines, config, data) for _ in range(2)]
        assert jobs[0].result() == jobs[1].result()
    builder.assert_called_once()
    baseline, _ = load_baseline_artifact(_context(config, data), "majority_class")
    assert baseline.height == len(data["test"])


def test_manifest_identity_mismatch_fails_explicitly(tmp_path):
    config, data = _config(tmp_path), _data()
    path = run_baselines(config, dataset=data)["test"]["majority_class"]
    manifest_path = path.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["identity"]["fit"]["fit_policy"] = "different_policy"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Incompatible baseline manifest"):
        load_baseline_artifact(_context(config, data), "majority_class")


def test_alignment_preserves_repeats_and_rejects_different_targets(tmp_path):
    config, data = _config(tmp_path), _data()
    run_baselines(config, dataset=data)
    baseline, _ = load_baseline_artifact(_context(config, data), "majority_class")
    predictions = baseline.select("id", "label").reverse()
    aligned = align_baseline(baseline, predictions, regression=False)
    assert aligned["id"].to_list() == ["other", "repeat", "repeat"]
    with pytest.raises(ValueError, match="IDs/labels"):
        align_baseline(
            baseline,
            predictions.with_columns(pl.lit(2).alias("label")),
            regression=False,
        )
    with pytest.raises(ValueError, match="observation counts"):
        align_baseline(baseline, predictions.head(1), regression=False)


@pytest.mark.parametrize("targets", [[None, 1, 0], [float("nan"), 1, 0], [4, 1, 0]])
def test_invalid_training_targets_rejected(tmp_path, targets):
    config, data = _config(tmp_path), _data()
    data["train"] = Dataset.from_dict(
        {"sequence": ["AAA", "CCC", "DDD"], "targets": targets}
    )
    with pytest.raises(ValueError, match="targets|class indices"):
        run_baselines(config, dataset=data)


def test_baseline_cli_uses_data_subset_without_model_validation(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2] / "modules/evaluate/configs"
    config = runner.load_baseline_config(
        [root / "config.yaml", root / "datasets/solubility_prediction.yaml"],
        ["steps.mode=not_a_model_mode", "workspace.model_name=null"],
    )
    assert config.baselines.names == ["majority_class"]
    assert not hasattr(config.workspace, "model_name")
    from modules.evaluate.src import run_evaluate

    monkeypatch.setattr(
        "sys.argv", ["run_evaluate.py", "--baselines", str(root / "config.yaml")]
    )
    monkeypatch.setattr(runner, "load_baseline_config", lambda *args: config)
    run = Mock(return_value={})
    monkeypatch.setattr(runner, "run_baselines", run)
    monkeypatch.setattr(
        run_evaluate,
        "Accelerator",
        Mock(side_effect=AssertionError("baseline CLI must not create an accelerator")),
    )
    run_evaluate.main()
    run.assert_called_once_with(config)


@pytest.mark.parametrize(
    "overrides,match",
    [
        (["baselines.names=[train_mean]"], "sequence_regression"),
        (["baselines.names=[majority_class,majority_class]"], "duplicates"),
        (["cross_validation.enabled=true"], "non-CV"),
    ],
)
def test_baseline_route_rejects_unsupported_requests(overrides, match):
    root = Path(__file__).resolve().parents[2] / "modules/evaluate/configs"
    with pytest.raises(ValueError, match=match):
        runner.load_baseline_config(
            [root / "config.yaml", root / "datasets/solubility_prediction.yaml"],
            overrides,
        )


@pytest.mark.parametrize(
    "task_type,name",
    [
        ("sequence_classification", "majority_class"),
        ("sequence_regression", "train_mean"),
    ],
)
@pytest.mark.parametrize("layerwise", [False, True])
def test_prepare_predict_carries_context_without_building_baselines(
    monkeypatch, tmp_path, model_family, task_type, name, layerwise
):
    raw = build_raw_dataset_dict(task_type)
    prepared = _run_prepare(
        monkeypatch,
        tmp_path,
        task_type,
        model_family,
        baseline_names=[name],
        dataset=raw,
        layerwise_config=LayerwiseConfig(enabled=True, layers=[0, 2])
        if layerwise
        else None,
    )
    workspace = _make_workspace(tmp_path, task_type)
    predict = PredictStep(PredictConfig(controls=[], persist_to_disk=False))
    if layerwise:
        from accelerate import Accelerator

        outputs = predict.run_layerwise(
            workspace, prepared, Accelerator(cpu=True), seed=710019
        ).values()
    else:
        outputs = [predict.run(workspace, prepared)]
    for output in outputs:
        assert output.baseline_context == prepared.baseline_context
        assert output.baseline_context["names"] == [name]
        assert output.controls == {}
    assert not (Path(workspace.base_path) / "baselines").exists()
    baseline_config = BaselineRunConfig(
        workspace=DatasetWorkspaceConfig.model_validate(
            {
                key: getattr(workspace.config, key)
                for key in DatasetWorkspaceConfig.model_fields
            }
        ),
        baselines={"names": [name]},
    )
    run_baselines(baseline_config, dataset=raw)
    for output in outputs:
        score = ScoreStep(
            ScoreConfig(
                metrics=["mse"] if task_type == "sequence_regression" else ["f1"],
                bootstrap_enabled=True,
                n_samples=5,
                n_permutations=5,
            )
        ).run(workspace, predict_result=output)
        assert name in score.control_comparisons


def test_external_prepare_and_score_use_source_fit_identity(
    monkeypatch, tmp_path, model_family
):
    raw = build_raw_dataset_dict("sequence_classification")
    external_data = DatasetDict({"test": _data()["test"]})
    external = ExternalTestSetConfig(name="separate", dataset_repo_id="org/separate")
    source, prepared = _run_prepare(
        monkeypatch,
        tmp_path,
        "sequence_classification",
        model_family,
        dataset=raw,
        baseline_names=["majority_class"],
        external=external,
        external_dataset=external_data,
    )
    workspace = _make_workspace(tmp_path, "sequence_classification")
    baseline_config = BaselineRunConfig(
        workspace=DatasetWorkspaceConfig.model_validate(
            {
                key: getattr(workspace.config, key)
                for key in DatasetWorkspaceConfig.model_fields
            }
        ),
        baselines={"names": ["majority_class"]},
        external_test_sets=[external],
    )
    paths = run_baselines(baseline_config, dataset=raw)
    assert prepared.baseline_context["fit"] == source.baseline_context["fit"]
    workspace = get_evaluation_workspace(
        workspace.config.model_copy(
            update={
                "dataset_name": external.name,
                "dataset_repo_id": external.dataset_repo_id,
            }
        )
    )
    output = PredictStep(PredictConfig(controls=[], persist_to_disk=False)).run(
        workspace, prepared
    )
    ScoreStep(ScoreConfig(metrics=["f1"])).run(workspace, predict_result=output)
    manifest = json.loads(workspace.run_manifest_path.read_text())
    assert manifest["score"]["baselines"]["majority_class"]["path"] == str(
        paths["external/separate"]["majority_class"]
    )
