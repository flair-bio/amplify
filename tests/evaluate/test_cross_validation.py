"""Offline fold-planning and real-step CV integration tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import torch
from accelerate import Accelerator
from datasets import Dataset, DatasetDict

from modules.evaluate.src.config import EvaluationConfig
from modules.evaluate.src.dataset.dataloader import DatasetSplitConfig
from modules.evaluate.src.dataset.folds import (
    CrossValidationConfig,
    make_fold_plan,
)
from modules.evaluate.src.runners import cross_validation as runner
from modules.evaluate.src.runners import pipeline
from modules.evaluate.src.schemas.artifacts import (
    EvaluationMetadata,
    PredictOutput,
    ScoreOutput,
)
from modules.evaluate.src.steps import prepare as prepare_module
from modules.evaluate.src.steps.score import ScoreStep
from modules.evaluate.src.utils.long_format import upsert_long_df

from .conftest import build_tiny_model


def raw_dataset(regression=False):
    # Unique, valid amino-acid sequences with both labels represented in
    # every biological group; lengths remain small for real model tests.
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    sequences = [
        "MK" + alphabet[index // 20] + alphabet[index % 20] + "A"
        for index in range(100)
    ]
    targets = [index % 2 for index in range(100)]
    if regression:
        targets = [index / 10.0 for index in range(100)]
    return DatasetDict(
        train=Dataset.from_dict(
            {
                "sequence": sequences[:80],
                "targets": targets[:80],
                "id": ["duplicate-source-id"] * 80,
                "cluster": [index // 4 for index in range(80)],
            }
        ),
        validation=Dataset.from_dict(
            {
                "sequence": sequences[80:],
                "targets": targets[80:],
                "cluster": [index // 4 for index in range(80, 100)],
            }
        ),
        test=Dataset.from_dict({"sequence": ["WWWW"], "targets": [1]}),
    )


def plan(dataset=None, regression=False, **kwargs):
    return make_fold_plan(
        raw_dataset(regression=regression) if dataset is None else dataset,
        CrossValidationConfig(n_splits=3, inner_validation_fraction=0.25, **kwargs),
        DatasetSplitConfig(),
        task_type="sequence_regression" if regression else "sequence_classification",
        sequence_column="sequence",
        label_column="targets",
        id_column="id",
    )


def evaluation_config(tmp_path, task="sequence_classification", head="sklearn_linear"):
    classification = task == "sequence_classification"
    return EvaluationConfig.model_validate(
        {
            "seed": 17,
            "workspace": {
                "dataset_name": "toy",
                "model_name": "tiny",
                "task_type": task,
                "target_type": "categorical" if classification else "continuous",
                "label_format": "scalar",
                "num_labels": 2 if classification else None,
                "base_path": str(tmp_path),
            },
            "cross_validation": {
                "enabled": True,
                "n_splits": 3,
                "strategy": "stratified_kfold" if classification else "kfold",
                "inner_validation_fraction": 0.25,
            },
            "steps": {
                "prepare": {
                    "embedding_cache": {"enabled": True, "cache_policy": "off"},
                    "max_length": 16,
                    "batch_size": 32,
                    "embedding_batch_size": 64,
                    "dataloader_num_workers": 0,
                    "tokenize_num_proc": 1,
                    "dataloader_pin_memory": False,
                    "dataloader_persistent_workers": False,
                },
                "tune": {
                    "head": {
                        "head_type": head,
                        "hyperparameters": (
                            {"C": 1.0} if classification else {"alpha": 1.0}
                        )
                        if head == "sklearn_linear"
                        else {},
                        "search_space": (
                            {"C": [0.1, 1.0]}
                            if classification
                            else {"alpha": [0.1, 1.0]}
                        )
                        if head == "sklearn_linear"
                        else {},
                    },
                    "save_best_model": True,
                    "max_epochs": 1,
                },
                "predict": {
                    "controls": [
                        "majority_class" if classification else "train_mean",
                        "scrambled_labels",
                    ]
                },
                "score": {
                    "metrics": ["accuracy", "f1", "roc_auc"]
                    if classification
                    else ["mse", "mae"]
                },
            },
        }
    )


@pytest.mark.parametrize(
    "strategy", ["kfold", "stratified_kfold", "group_kfold", "stratified_group_kfold"]
)
def test_partition_invariants_and_reproducibility(strategy):
    kwargs = {"strategy": strategy}
    if "group" in strategy:
        kwargs["group_column"] = "cluster"
    first, second = plan(**kwargs), plan(**kwargs)
    assert first.fingerprint == second.fingerprint
    assert first.assignments.equals(second.assignments)
    assert len(first.pool) == 100
    assert sorted(index for fold in first.folds for index in fold.test) == list(
        range(100)
    )
    for fold in first.folds:
        roles = [set(fold.train), set(fold.validation), set(fold.test)]
        assert set.union(*roles) == set(range(100))
        assert (
            not roles[0] & roles[1]
            and not roles[0] & roles[2]
            and not roles[1] & roles[2]
        )
        assert set(fold.dataset(first.pool)) == {"train", "validation", "test"}
        if "group" in strategy:
            groups = [set(first.pool.select(sorted(role))["cluster"]) for role in roles]
            assert (
                not groups[0] & groups[1]
                and not groups[0] & groups[2]
                and not groups[1] & groups[2]
            )
    assert "test" not in first.assignments["source_split"].to_list()
    assert first.pool["id"][0] == "train:0"
    assert first.assignments["source_id"][0] == '"duplicate-source-id"'
    assert first.fingerprint != plan(split_seed=12, **kwargs).fingerprint


def test_default_pool_and_train_only_without_synthesized_validation():
    result = plan()
    assert len(result.pool) == 100
    assert len(set(result.pool["id"])) == 100
    assert {"train", "validation"} == set(result.assignments["source_split"])
    dataset = raw_dataset()
    del dataset["validation"]
    assert len(plan(dataset, pool_roles=["train"]).pool) == 80
    with pytest.raises(ValueError, match="must exist"):
        plan(dataset)


def test_named_source_mapping():
    dataset = raw_dataset()
    dataset["train_cluster"] = dataset.pop("train")
    dataset["validation_cluster"] = dataset.pop("validation")
    result = make_fold_plan(
        dataset,
        CrossValidationConfig(n_splits=3, inner_validation_fraction=0.25),
        DatasetSplitConfig(name="cluster"),
        task_type="sequence_classification",
        sequence_column="sequence",
        label_column="targets",
        id_column="id",
    )
    assert result.pool["id"][0] == "train_cluster:0"
    with pytest.raises(ValueError, match="exclude the official test"):
        make_fold_plan(
            dataset,
            CrossValidationConfig(),
            DatasetSplitConfig(train="test"),
            task_type="sequence_classification",
            sequence_column="sequence",
            label_column="targets",
            id_column="id",
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pool_roles": []},
        {"pool_roles": ["train", "train"]},
        {"group_column": "cluster"},
        {"strategy": "group_kfold"},
        {"n_splits": 1},
        {"inner_validation_fraction": 1.0},
    ],
)
def test_invalid_protocol_config(kwargs):
    with pytest.raises(ValueError):
        CrossValidationConfig(**kwargs)


def test_duplicates_must_share_biological_group():
    dataset = raw_dataset()
    rows = dataset["train"].to_dict()
    rows["sequence"][1] = rows["sequence"][0]
    dataset["train"] = Dataset.from_dict(rows)
    with pytest.raises(ValueError, match="Duplicate sequences"):
        plan(dataset)
    assert plan(dataset, strategy="group_kfold", group_column="cluster")
    rows["cluster"][1] = 999
    dataset["train"] = Dataset.from_dict(rows)
    with pytest.raises(ValueError, match="Duplicate sequences"):
        plan(dataset, strategy="group_kfold", group_column="cluster")


def test_invalid_labels_groups_and_rare_classes():
    dataset = raw_dataset()
    rows = dataset["train"].to_dict()
    rows["targets"][0] = float("nan")
    dataset["train"] = Dataset.from_dict(rows)
    with pytest.raises(ValueError, match="finite scalar"):
        plan(dataset, pool_roles=["train"])
    rows["targets"] = [0] * 79 + [1]
    dataset["train"] = Dataset.from_dict(rows)
    with pytest.raises(ValueError, match="examples per class"):
        plan(dataset, pool_roles=["train"])
    rows["targets"] = [index % 2 for index in range(80)]
    rows["cluster"] = [None] * 80
    dataset["train"] = Dataset.from_dict(rows)
    with pytest.raises(ValueError, match="non-null"):
        plan(dataset, strategy="group_kfold", group_column="cluster")


def test_regression_numeric_strings_produce_the_same_folds_as_numbers():
    numeric = raw_dataset(regression=True)
    strings = DatasetDict(
        {
            split: Dataset.from_dict(
                {**data.to_dict(), "targets": [str(value) for value in data["targets"]]}
            )
            for split, data in numeric.items()
        }
    )
    expected = plan(numeric, regression=True, strategy="kfold")
    actual = plan(strings, regression=True, strategy="kfold")

    assert actual.folds == expected.folds
    assert actual.assignments.equals(expected.assignments)
    assert actual.fingerprint == expected.fingerprint
    assert sorted(index for fold in actual.folds for index in fold.test) == list(
        range(100)
    )


@pytest.mark.parametrize(
    "label",
    [
        None,
        True,
        False,
        "",
        "invalid",
        "nan",
        "inf",
        "-inf",
        "1e309",
        float("nan"),
        float("inf"),
        float("-inf"),
        [1.0],
    ],
)
def test_regression_rejects_invalid_or_nonfinite_labels(label):
    dataset = raw_dataset(regression=True)
    rows = dataset["train"].to_dict()
    rows["targets"] = [label] * len(rows["targets"])
    dataset["train"] = Dataset.from_dict(rows)

    with pytest.raises(ValueError, match="finite scalars"):
        plan(dataset, regression=True, strategy="kfold", pool_roles=["train"])


def test_classification_still_rejects_numeric_string_labels():
    dataset = raw_dataset()
    rows = dataset["train"].to_dict()
    rows["targets"] = [str(value) for value in rows["targets"]]
    dataset["train"] = Dataset.from_dict(rows)

    with pytest.raises(ValueError, match="finite scalars"):
        plan(dataset, pool_roles=["train"])


def test_missing_inner_class_fails_before_training():
    rows = raw_dataset()["train"].select(range(24)).to_dict()
    rows["targets"] = [(index // 4) % 2 for index in range(24)]
    dataset = DatasetDict(train=Dataset.from_dict(rows))
    with pytest.raises(ValueError, match="missing classes"):
        plan(
            dataset,
            strategy="group_kfold",
            group_column="cluster",
            pool_roles=["train"],
        )


@pytest.mark.parametrize(
    ("section", "update", "match"),
    [
        ("steps", {"mode": "evaluate_as_is"}, "requires tune_probe"),
        ("score", {"bootstrap_enabled": True}, "point estimates"),
        ("predict", {"persist_to_disk": False}, "persist_to_disk"),
        ("tune", {"variance_seeds": [5]}, "variance_seeds"),
        ("layerwise", {"enabled": True}, "without layerwise"),
    ],
)
def test_unsupported_combinations(tmp_path, section, update, match):
    payload = evaluation_config(tmp_path).model_dump()
    if section == "steps":
        payload["steps"].update(update)
    else:
        payload["steps"][section].update(update)
    with pytest.raises(ValueError, match=match):
        EvaluationConfig.model_validate(payload)


def test_coverage_and_descriptive_summary():
    for ids in [["a"], ["a", "a"], ["a", "c"]]:
        with pytest.raises(ValueError, match="exactly once"):
            runner.validate_prediction_coverage(
                pl.DataFrame({"id": ids}), ["a", "b"], context="OOF"
            )
    runner.validate_prediction_coverage(
        pl.DataFrame({"id": ["b", "a"]}), ["a", "b"], context="OOF"
    )
    summary = runner.summarize_fold_scores(
        [{"f1": 0.5}, {"f1": 1.0}, {"f1": float("nan")}]
    )
    assert summary["f1"]["valid_folds"] == 2
    assert summary["f1"]["values"] == [0.5, 1.0, None]
    assert summary["f1"]["mean"] == 0.75
    assert summary["f1"]["std"] == pytest.approx(np.std([0.5, 1.0], ddof=1))


def test_distributed_rejected_before_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner, "load_raw_evaluation_dataset", lambda _: pytest.fail("must not load")
    )
    with pytest.raises(ValueError, match="single process"):
        runner.run_cross_validation(
            evaluation_config(tmp_path), SimpleNamespace(num_processes=2)
        )


@pytest.mark.parametrize("task", ["sequence_classification", "sequence_regression"])
@pytest.mark.parametrize("head", ["sklearn_linear", "torch_linear"])
def test_real_steps_end_to_end(tmp_path, monkeypatch, model_family, task, head):
    config = evaluation_config(tmp_path / "eval", task, head)
    raw = raw_dataset(regression=task == "sequence_regression")
    monkeypatch.setattr(runner, "load_raw_evaluation_dataset", lambda _: raw)
    monkeypatch.setattr(
        prepare_module,
        "load_evaluation_dataset",
        lambda *args, **kwargs: pytest.fail("supplied fold must bypass loader"),
    )
    monkeypatch.setattr(
        prepare_module.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: model_family.tokenizer,
    )
    initial_model = build_tiny_model(
        model_family, task, 2 if task == "sequence_classification" else 1
    )
    state = initial_model.state_dict()
    models = []

    class OfflineModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            model = build_tiny_model(
                model_family, task, 2 if task == "sequence_classification" else 1
            )
            model.load_state_dict(state)
            models.append(model)
            return model

    monkeypatch.setattr(
        type(prepare_module.get_task(task)), "auto_model_class", OfflineModel
    )
    accelerator = Accelerator(cpu=True)
    # Bound tiny CPU kernels, restoring the shared process setting afterwards.
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        manifest_path = runner.run_cross_validation(config, accelerator)
    finally:
        torch.set_num_threads(previous_threads)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "complete"
    assert len(manifest["folds"]) == 3
    assert len(models) == 3 and len({id(model) for model in models}) == 3
    for model in models:
        for name, tensor in model.state_dict().items():
            assert torch.equal(tensor, state[name]), (
                "frozen trunk/initial model changed"
            )
    predictions = pl.read_parquet(manifest["oof_predictions"])
    assert predictions.height == predictions["id"].n_unique() == 100
    assert set(predictions["fold_id"]) == {0, 1, 2}
    if task == "sequence_classification":
        assert all(len(row) == 2 for row in predictions["probability"])
    assert set(manifest["control_oof_scores"]) == set(config.steps.predict.controls)
    assert all(
        fold["train_count"] + fold["validation_count"] + fold["test_count"] == 100
        for fold in manifest["folds"]
    )
    score_table = pl.read_parquet(
        manifest_path.parent / "scores" / "scores_long.parquet"
    )
    assert score_table["variant"].n_unique() == 4
    assert score_table["cv_id"].unique().to_list() == [manifest["cv_id"]]
    assert score_table["fold_id"].null_count() == len(config.steps.score.metrics)
    fold_workspace = runner.get_evaluation_workspace(
        config.workspace,
        variant=(
            f"{config.variant}-cv_{manifest['cv_id']}-run_{manifest['config_hash']}"
            "-fold_0"
        ),
    )
    loaded = ScoreStep(config.steps.score).load_predict_output(fold_workspace)
    assert loaded.metadata.fold_id == 0
    assert loaded.metadata.split_seed == config.cross_validation.split_seed
    assert loaded.metadata.split == "default"
    assert raw["test"]["sequence"] == ["WWWW"]
    with pytest.raises(FileExistsError):
        runner.run_cross_validation(config, accelerator)


def test_failed_fold_never_completes_experiment(tmp_path, monkeypatch):
    config = evaluation_config(tmp_path)
    monkeypatch.setattr(runner, "load_raw_evaluation_dataset", lambda _: raw_dataset())
    monkeypatch.setattr(
        pipeline.PrepareStep, "run", lambda *args, **kwargs: SimpleNamespace()
    )

    def fail(*args, **kwargs):
        raise RuntimeError("training failed")

    monkeypatch.setattr(pipeline.TuneStep, "run", fail)
    with pytest.raises(RuntimeError, match="training failed"):
        runner.run_cross_validation(config, Accelerator(cpu=True))
    manifests = list(tmp_path.glob("**/cv_manifest.json"))
    assert len(manifests) == 1
    payload = json.loads(manifests[0].read_text())
    assert payload["status"] == "running"
    assert payload["folds"] == []
    assert "oof_scores" not in payload


@pytest.mark.parametrize("changed_setting", ["seed", "hyperparameters"])
def test_rerun_resumes_completed_folds(tmp_path, monkeypatch, changed_setting):
    config = evaluation_config(tmp_path, task="sequence_regression")
    monkeypatch.setattr(
        runner, "load_raw_evaluation_dataset", lambda _: raw_dataset(regression=True)
    )
    calls: list[int] = []
    fail_on: list[int] = [1]

    def fake_pipeline(config, workspace, accelerator, dataset, cross_validation):
        calls.append(cross_validation.fold_id)
        if cross_validation.fold_id in fail_on:
            raise RuntimeError("interrupted")
        test = dataset["test"]
        output = PredictOutput(
            metadata=EvaluationMetadata(
                model_name="tiny",
                dataset_name="toy",
                task_type="sequence_regression",
                variant=workspace.variant,
                **cross_validation.model_dump(),
            ),
            predictions=pl.DataFrame(
                {"id": test["id"], "prediction": test["targets"], "label": test["targets"]}
            ),
            requested_controls=[],
            config={"predictions_filename": "predictions.parquet", "controls": []},
        )
        runner.persist_predictions(workspace, output, config.seed)
        return pipeline.PipelineResult(
            predictions={None: output},
            scores={None: ScoreOutput(scores={"mse": 0.0}, metadata=output.metadata)},
        )

    monkeypatch.setattr(pipeline, "run_pipeline", fake_pipeline)
    accelerator = Accelerator(cpu=True)
    with pytest.raises(RuntimeError, match="interrupted"):
        runner.run_cross_validation(config, accelerator)
    assert calls == [0, 1]

    fail_on.clear()
    calls.clear()
    original_manifest_path = runner.run_cross_validation(config, accelerator)
    manifest = json.loads(original_manifest_path.read_text())
    assert calls == [1, 2]
    assert manifest["status"] == "complete"
    assert [fold["fold_id"] for fold in manifest["folds"]] == [0, 1, 2]

    original_artifacts = {
        path: path.read_bytes()
        for path in original_manifest_path.parent.parent.rglob("*")
        if path.is_file()
    }
    payload = config.model_dump()
    if changed_setting == "seed":
        payload["seed"] = 18
    else:
        payload["steps"]["tune"]["head"]["hyperparameters"]["alpha"] = 2.0
    changed_config = EvaluationConfig.model_validate(payload)
    assert changed_config.variant == config.variant
    calls.clear()
    changed_path = runner.run_cross_validation(changed_config, accelerator)
    assert calls == [0, 1, 2]
    changed_manifest = json.loads(changed_path.read_text())
    assert changed_path != original_manifest_path
    assert changed_manifest["cv_id"] == manifest["cv_id"]
    assert changed_manifest["config_hash"] != manifest["config_hash"]
    assert changed_manifest["status"] == "complete"
    for path, contents in original_artifacts.items():
        assert path.read_bytes() == contents
    assert pl.read_parquet(changed_manifest["assignments"]).equals(
        pl.read_parquet(manifest["assignments"])
    )
    assert {fold["variant"] for fold in changed_manifest["folds"]}.isdisjoint(
        fold["variant"] for fold in manifest["folds"]
    )
    comparison_path = (
        runner.get_evaluation_workspace(config.workspace).comparison_path
        / config.steps.score.long_filename
    )
    comparison = pl.read_parquet(comparison_path)
    assert comparison["variant"].n_unique() == 2
    assert comparison["cv_id"].unique().to_list() == [manifest["cv_id"]]
    calls.clear()
    with pytest.raises(FileExistsError):
        runner.run_cross_validation(config, accelerator)
    with pytest.raises(FileExistsError):
        runner.run_cross_validation(changed_config, accelerator)
    assert calls == []


def test_cli_dispatches_cv_without_single_split_preparation(tmp_path, monkeypatch):
    from modules.evaluate.src import run_evaluate

    config = evaluation_config(tmp_path)
    monkeypatch.setattr("sys.argv", ["run_evaluate.py", "example.yaml"])
    monkeypatch.setattr(run_evaluate, "load_and_parse", lambda **kwargs: config)
    monkeypatch.setattr(
        run_evaluate, "Accelerator", lambda **kwargs: Accelerator(cpu=True)
    )
    calls = []
    monkeypatch.setattr(
        runner, "run_cross_validation", lambda cfg, acc: calls.append(cfg)
    )
    monkeypatch.setattr(
        run_evaluate,
        "run_pipeline",
        lambda *args, **kwargs: pytest.fail("single-split path"),
    )
    run_evaluate.main()
    assert calls == [config]


@pytest.mark.parametrize("mode", ["tune_probe", "finetune", "evaluate_as_is"])
def test_shared_pipeline_preserves_single_run_modes(tmp_path, monkeypatch, mode):
    payload = evaluation_config(tmp_path).model_dump()
    payload["cross_validation"] = {"enabled": False}
    payload["steps"]["mode"] = mode
    payload["steps"]["prepare"]["embedding_cache"]["enabled"] = False
    payload["steps"]["tune"]["head"] = {"head_type": "mlp"}
    config = EvaluationConfig.model_validate(payload)
    workspace = runner.get_evaluation_workspace(
        config.workspace, variant=config.variant
    )
    calls = []
    prepared = SimpleNamespace(model=object())
    predictions = PredictOutput(
        metadata=EvaluationMetadata(
            model_name="tiny", dataset_name="toy", task_type="sequence_classification"
        ),
        predictions=pl.DataFrame({"id": ["a"], "prediction": [0], "label": [0]}),
    )

    def prepare(*args, **kwargs):
        calls.append("prepare")
        return prepared

    def tune(*args, **kwargs):
        calls.append("tune")
        return prepared

    def predict(*args, **kwargs):
        calls.append("predict")
        assert kwargs["head_type"] == (None if mode == "evaluate_as_is" else "mlp")
        assert kwargs["variant"] == config.variant
        return predictions

    def score(*args, **kwargs):
        calls.append("score")
        assert kwargs["predict_result"] is predictions
        return ScoreOutput(scores={"accuracy": 1.0})

    def wrap(model):
        calls.append("wrap")
        return model

    monkeypatch.setattr(pipeline.PrepareStep, "run", prepare)
    monkeypatch.setattr(pipeline.TuneStep, "run", tune)
    monkeypatch.setattr(pipeline.PredictStep, "run", predict)
    monkeypatch.setattr(pipeline.ScoreStep, "run", score)
    accelerator = SimpleNamespace(
        is_main_process=True, prepare=wrap, wait_for_everyone=lambda: None
    )
    result = pipeline.run_pipeline(config, workspace, accelerator)
    assert calls == [
        "prepare",
        "wrap" if mode == "evaluate_as_is" else "tune",
        "predict",
        "score",
    ]
    assert result.predictions[None] is predictions
    assert result.scores[None].scores == {"accuracy": 1.0}


def test_old_metadata_table_accepts_cv_columns(tmp_path):
    path = tmp_path / "scores.parquet"
    old = pl.DataFrame({"variant": ["ordinary"], "split": ["default"], "value": [0.2]})
    old.write_parquet(path)
    new = pl.DataFrame(
        {
            "variant": ["cv-fold_0"],
            "split": ["default"],
            "value": [0.3],
            "cv_id": ["abc"],
            "fold_id": [0],
        }
    )
    merged = upsert_long_df(path, new, "cv-fold_0", "default")
    assert merged.height == 2
    assert merged["fold_id"].to_list() == [None, 0]
