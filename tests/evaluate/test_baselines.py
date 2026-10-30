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

from modules.evaluate.src.contracts.config import LayerwiseConfig
from modules.evaluate.src.dataset import dataloader
from modules.evaluate.src.contracts.config import ExternalTestSetConfig
from modules.evaluate.src.contracts.config import DatasetWorkspaceConfig
from modules.evaluate.src.dataset.workspace import get_evaluation_workspace
from modules.evaluate.src.runners import baselines as runner
from modules.evaluate.src.contracts.config_loader import load_baseline_config
from modules.evaluate.src.contracts.config import BaselineRunConfig
from modules.evaluate.src.runners.baselines import run_baselines
from modules.evaluate.src.contracts.artifacts import EvaluationMetadata, PredictOutput
from modules.evaluate.src.contracts.config import PredictConfig
from modules.evaluate.src.steps.predict import PredictStep, persist_predictions
from modules.evaluate.src.contracts.config import ScoreConfig
from modules.evaluate.src.steps.score import ScoreStep
from modules.evaluate.src.dataset.baseline_data import (
    align_baseline,
    align_token_baseline,
    artifact_directory,
    baseline_context,
    load_baseline_artifact,
    artifact_directory,
)
from modules.evaluate.src.model.baselines import (
    build_sequence_baseline,
    _mmseqs_nearest_neighbors,
)

from .conftest import build_raw_dataset_dict
from .test_prepare import _make_workspace, _run_prepare


def test_legacy_baseline_configuration_is_normalized_once():
    with pytest.warns(DeprecationWarning, match="predict.baselines"):
        config = PredictConfig(controls=["scrambled_labels", "majority_class"])
    assert config.controls == ["scrambled_labels"]
    assert config.baselines == ["majority_class"]
    assert config.conditions == ["scrambled_labels", "majority_class"]
    restored = PredictConfig.model_validate(config.model_dump())
    assert restored == config


def test_duplicate_legacy_and_explicit_baseline_is_rejected():
    with (
        pytest.warns(DeprecationWarning),
        pytest.raises(ValueError, match="duplicates"),
    ):
        PredictConfig(controls=["majority_class"], baselines=["majority_class"])


@pytest.mark.parametrize(
    "payload", [{"controls": None}, {"baselines": 1}, {"controls": [["random_trunk"]]}]
)
def test_invalid_condition_types_keep_schema_validation_errors(payload):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        PredictConfig.model_validate(payload)


def test_inline_baseline_requests_survive_prediction_manifest_reload(tmp_path):
    workspace = _make_workspace(tmp_path, "sequence_classification")
    workspace.setup_directories()
    config = PredictConfig(controls=[], baselines=["majority_class"])
    baseline = pl.DataFrame({"id": ["protein"], "prediction": [1]})
    output = PredictOutput(
        metadata=EvaluationMetadata(
            model_name=workspace.model_name,
            dataset_name=workspace.dataset_name,
            task_type=workspace.task_type,
            seed=workspace.seed,
        ),
        predictions=baseline.with_columns(pl.lit(1).alias("label")),
        controls={"majority_class": baseline},
        requested_controls=config.conditions,
        config=config.model_dump(mode="json"),
    )
    persist_predictions(workspace, output, workspace.seed)
    restored = ScoreStep.load_predict_output(workspace)
    assert restored.requested_controls == ["majority_class"]
    assert restored.controls["majority_class"].to_dicts() == baseline.to_dicts()


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
        max_length=config.prepare.max_length,
    )


@pytest.mark.parametrize("regression", [False, True])
@pytest.mark.parametrize("baseline", ["constant", "composition", "mmseqs_knn"])
def test_standalone_baseline_reuses_predictions_and_never_fits_validation(
    tmp_path, monkeypatch, regression, caplog, baseline
):
    caplog.set_level("INFO")
    config, data = _config(tmp_path, regression), _data(regression)
    if baseline != "constant":
        config.baselines.names = [baseline]
    monkeypatch.setattr(
        "modules.evaluate.src.model.baselines._mmseqs_nearest_neighbors",
        lambda train, test: [0, 1, 2],
    )
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
    if name == "train_mean":
        assert baseline["prediction"].to_list() == [2.0] * 3
    elif name == "majority_class":
        assert baseline["prediction"].to_list() == [1] * 3
        assert baseline["probability"][0].to_list() == pytest.approx([1 / 3, 2 / 3, 0])
    assert ("probability" in baseline.columns) is not regression
    builder = Mock(side_effect=AssertionError("must not refit"))
    monkeypatch.setattr(runner, "build_sequence_baseline", builder)
    assert run_baselines(config, dataset=data) == outputs
    builder.assert_not_called()


@pytest.mark.parametrize("role", ["train", "test"])
def test_short_hash_collisions_reject_different_full_identities(
    tmp_path, monkeypatch, role
):
    from modules.evaluate.src.dataset import baseline_data as baselines

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


@pytest.mark.parametrize(
    "change", ["train", "test", "split", "task_classes", "config", "revision"]
)
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
    elif change == "config":
        config.workspace.dataset_config_name = "another_dataset"
    elif change == "revision":
        config.workspace.dataset_revision = "a" * 40
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
    monkeypatch.setattr(dataloader, "load_raw_evaluation_dataset", lambda _: external)
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


def test_suite_baseline_loads_main_and_external_configurations(tmp_path, monkeypatch):
    config = _config(tmp_path)
    config.workspace.dataset_repo_id = "org/suite"
    config.workspace.dataset_config_name = "main"
    config.workspace.dataset_revision = "a" * 40
    config.external_test_sets = [
        ExternalTestSetConfig(
            name="heldout",
            dataset_repo_id="org/suite",
            dataset_config_name="heldout",
            dataset_revision="a" * 40,
        )
    ]
    loader = Mock(return_value=_data())
    monkeypatch.setattr(dataloader, "load_dataset", loader)
    outputs = run_baselines(config)
    assert [call.kwargs for call in loader.call_args_list] == [
        {"name": "main", "revision": "a" * 40},
        {"name": "heldout", "revision": "a" * 40},
    ]
    assert "external/heldout" in outputs
    main_manifest = json.loads(
        outputs["test"]["majority_class"].with_name("manifest.json").read_text()
    )
    external_manifest = json.loads(
        outputs["external/heldout"]["majority_class"]
        .with_name("manifest.json")
        .read_text()
    )
    assert main_manifest["identity"]["test"] != external_manifest["identity"]["test"]
    assert main_manifest["identity"]["fit"] == external_manifest["identity"]["fit"]


@pytest.mark.parametrize("regression", [False, True])
@pytest.mark.parametrize("baseline", ["constant", "composition", "mmseqs_knn"])
def test_score_loads_shared_predictions_and_standalone_manifest(
    tmp_path, monkeypatch, regression, baseline
):
    config, data = _config(tmp_path, regression), _data(regression)
    if baseline != "constant":
        config.baselines.names = [baseline]
    monkeypatch.setattr(
        "modules.evaluate.src.model.baselines._mmseqs_nearest_neighbors",
        lambda train, test: [0, 1, 2],
    )
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
        if name == "train_mean":
            assert np.isnan(score_manifest["baselines"][name]["scores"]["spearmanr"])
            baseline_scores = conditions.filter(
                (pl.col("condition") == name) & (pl.col("metric") == "spearmanr")
            )
            assert np.isnan(baseline_scores["score"][0])
            assert "spearmanr" not in result.control_comparisons[name]


@pytest.mark.parametrize("name", ["majority_class", "composition", "mmseqs_knn"])
def test_classification_artifacts_require_probabilities(tmp_path, monkeypatch, name):
    config, data = _config(tmp_path), _data()
    config.baselines.names = [name]
    monkeypatch.setattr(
        "modules.evaluate.src.model.baselines._mmseqs_nearest_neighbors",
        lambda train, test: [0, 1, 2],
    )
    run_baselines(config, dataset=data)
    context = _context(config, data)
    frame, _ = load_baseline_artifact(context, name)
    monkeypatch.setattr(pl, "read_parquet", lambda _: frame.drop("probability"))
    with pytest.raises(ValueError, match="Invalid baseline prediction schema/count"):
        load_baseline_artifact(context, name)


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


def test_shared_token_prior_uses_all_training_residues_and_is_reused(
    tmp_path, monkeypatch
):
    data = DatasetDict(
        {
            "train": Dataset.from_dict(
                {"sequence": ["AAAAAA"], "targets": [[0, -100, 1, 1, 1, 1]]}
            ),
            "validation": Dataset.from_dict(
                {"sequence": ["AAAAAA"], "targets": [[0] * 6]}
            ),
            "test": Dataset.from_dict(
                {
                    "id": ["protein"],
                    "sequence": ["AAAAAA"],
                    "targets": [[0, -100, 1, 1, 0, 1]],
                }
            ),
        }
    )
    config = BaselineRunConfig.model_validate(
        {
            "workspace": {
                "dataset_name": "tokens",
                "task_type": "token_classification",
                "target_type": "categorical",
                "label_format": "per_token",
                "num_labels": 2,
                "output_dir": tmp_path,
            },
            "prepare": {"max_length": 4},
            "baselines": {"names": ["majority_class"]},
        }
    )
    paths = run_baselines(config, dataset=data)
    context = _context(config, data)
    baseline, path = load_baseline_artifact(context, "majority_class")
    assert baseline["prediction"].to_list() == [1]
    assert baseline["probability"][0].to_list() == pytest.approx([0.2, 0.8])
    assert context["fit"]["fit_policy"] == "all_labeled_training_residues"
    changed = config.model_copy(deep=True)
    changed.prepare.max_length = 6
    changed.seed = 123
    assert _context(changed, data) == context
    monkeypatch.setattr(
        runner,
        "build_sequence_baseline",
        Mock(side_effect=AssertionError("shared token prior must not be refitted")),
    )
    assert run_baselines(changed, dataset=data) == paths
    assert paths["test"]["majority_class"] == path

    ragged = pl.DataFrame(
        {"id": ["protein"], "label": [[1, 1, 0]], "prediction": [[0, 0, 0]]}
    )
    aligned = align_token_baseline(baseline, ragged)
    assert aligned["prediction"].to_list() == [[1, 1, 1]]
    assert "probability" not in aligned.columns
    with_probabilities = ragged.with_columns(
        pl.Series("probability", [[[0.5, 0.5]] * 3])
    )
    assert align_token_baseline(baseline, with_probabilities)[
        "probability"
    ].to_list() == [[[0.2, 0.8]] * 3]
    cached = pl.DataFrame(
        {
            "id": ["protein"] * 3,
            "label": [1, 1, 0],
            "prediction": [0, 0, 0],
        }
    )
    assert align_token_baseline(baseline, cached)["prediction"].to_list() == [1] * 3
    with pytest.raises(ValueError, match="IDs"):
        align_token_baseline(baseline, cached.with_columns(pl.lit("other").alias("id")))
    with pytest.raises(ValueError, match="residue window"):
        align_token_baseline(baseline, cached.with_columns(pl.lit(0).alias("label")))


@pytest.mark.parametrize(
    "labels,match",
    [
        ([0], "one label per residue"),
        ([0, 2], "outside num_labels"),
        ([-100, -100], "no valid residue labels"),
    ],
)
def test_shared_token_prior_rejects_invalid_training_labels(tmp_path, labels, match):
    config = _config(tmp_path)
    config.workspace.task_type = "token_classification"
    config.workspace.num_labels = 2
    data = DatasetDict(
        {
            "train": Dataset.from_dict({"sequence": ["AA"], "targets": [labels]}),
            "validation": Dataset.from_dict({"sequence": ["AA"], "targets": [[0, 1]]}),
            "test": Dataset.from_dict({"sequence": ["AA"], "targets": [[0, 1]]}),
        }
    )
    with pytest.raises(ValueError, match=match):
        run_baselines(config, dataset=data)


def test_composition_baselines_fit_normalized_residue_frequencies():
    train_classification = [
        {"id": "a", "sequence": "AAAA", "label": 0},
        {"id": "b", "sequence": "AAAC", "label": 0},
        {"id": "c", "sequence": "CCCC", "label": 1},
        {"id": "d", "sequence": "CCCA", "label": 1},
    ]
    test_classification = [
        {"id": "left", "sequence": "AAAA", "label": 0},
        {"id": "right", "sequence": "CCCC", "label": 1},
    ]
    classification = build_sequence_baseline(
        "composition",
        "sequence_classification",
        train_classification,
        test_classification,
        2,
    )
    assert classification["prediction"].to_list() == [0, 1]
    assert classification["probability"].to_list()[0][0] > 0.5

    train_regression = [
        {"id": "a", "sequence": "AAAA", "label": 1.0},
        {"id": "b", "sequence": "AAAC", "label": 1.0},
        {"id": "c", "sequence": "CCCC", "label": 3.0},
        {"id": "d", "sequence": "CCCA", "label": 3.0},
    ]
    regression = build_sequence_baseline(
        "composition",
        "sequence_regression",
        train_regression,
        test_classification,
        None,
    )
    assert regression["prediction"][0] < regression["prediction"][1]


def test_mmseqs_knn_uses_top_training_hit_and_reports_missing_tool(monkeypatch):
    train = [
        {"id": "first", "sequence": "AAAA", "label": 0},
        {"id": "second", "sequence": "CCCC", "label": 1},
    ]
    test = [{"id": "query", "sequence": "CCCC", "label": 1}]

    def fake_search(command, *, check, capture_output, text):
        assert command[1] == "easy-search"
        assert command[command.index("--threads") + 1] == "1"
        Path(command[4]).write_text("test_0\ttrain_1\n")
        return type("Result", (), {"returncode": 0, "stderr": "", "stdout": ""})()

    monkeypatch.setattr(
        "modules.evaluate.src.model.baselines.shutil.which", lambda _: "/usr/bin/mmseqs"
    )
    monkeypatch.setattr(
        "modules.evaluate.src.model.baselines.subprocess.run", fake_search
    )
    assert _mmseqs_nearest_neighbors(train, test) == [1]

    monkeypatch.setattr(
        "modules.evaluate.src.model.baselines.shutil.which", lambda _: None
    )
    with pytest.raises(FileNotFoundError, match="requires the MMseqs2"):
        _mmseqs_nearest_neighbors(train, test)


def test_amino_acid_frequency_baseline_runs_on_test_only_data(tmp_path):
    data = DatasetDict(
        {
            "test": Dataset.from_dict(
                {"sequence": ["AXC", "AAAC"], "id": ["first", "second"]}
            )
        }
    )
    config = BaselineRunConfig.model_validate(
        {
            "workspace": {
                "dataset_name": "uniprot_test",
                "task_type": "pseudo_perplexity",
                "target_type": "continuous",
                "label_format": "scalar",
                "base_path": str(tmp_path),
            },
            "prepare": {"max_length": 4},
            "baselines": {"names": ["amino_acid_frequency"]},
        }
    )
    path = run_baselines(config, dataset=data)["test"]["amino_acid_frequency"]
    baseline = pl.read_parquet(path)
    context = _context(config, data)
    changed_context = baseline_context(
        config.workspace,
        data,
        config.baselines.names,
        sequence_column=config.prepare.sequence_column,
        label_column=config.prepare.label_column,
        id_column=config.prepare.id_column,
        split_roles=config.prepare.split.model_dump(),
        max_length=5,
    )
    assert artifact_directory(context, "amino_acid_frequency") != artifact_directory(
        changed_context, "amino_acid_frequency"
    )
    assert baseline["id"].to_list() == [
        "first:0:0",
        "first:0:1",
        "second:1:0",
        "second:1:1",
    ]
    assert baseline["prediction"][0] > baseline["prediction"][1]
    predictions = pl.DataFrame(
        {
            "id": ["second:1:1", "first:0:0", "second:1:0", "first:0:1"],
            "label": [1, 0, 0, 1],
            "prediction": [0.0] * 4,
        }
    )
    aligned = align_baseline(
        baseline,
        predictions,
        regression=False,
        match_labels=False,
    )
    assert aligned["id"].to_list() == predictions["id"].to_list()
    assert all(value < 0 for value in aligned["prediction"].to_list())


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
    config = load_baseline_config(
        [root / "config.yaml", root / "datasets/solubility_prediction.yaml"],
        ["steps.mode=not_a_model_mode", "workspace.model_name=null"],
    )
    assert config.baselines.names == [
        "majority_class",
        "composition",
        "mmseqs_knn",
    ]
    assert not hasattr(config.workspace, "model_name")
    from modules.evaluate.src import run_evaluate

    monkeypatch.setattr(
        "sys.argv", ["run_evaluate.py", "--baselines", str(root / "config.yaml")]
    )
    monkeypatch.setattr(run_evaluate, "load_baseline_config", lambda *args: config)
    run = Mock(return_value={})
    monkeypatch.setattr(run_evaluate, "run_baselines", run)
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
    ],
)
def test_baseline_route_rejects_unsupported_requests(overrides, match):
    root = Path(__file__).resolve().parents[2] / "modules/evaluate/configs"
    with pytest.raises(ValueError, match=match):
        load_baseline_config(
            [root / "config.yaml", root / "datasets/solubility_prediction.yaml"],
            overrides,
        )


@pytest.mark.parametrize(
    "task_type,name",
    [
        ("sequence_classification", "majority_class"),
        ("sequence_regression", "train_mean"),
        ("token_classification", "majority_class"),
    ],
)
@pytest.mark.parametrize("layerwise", [False, True])
@pytest.mark.parametrize("cached", [False, True])
def test_prepare_predict_carries_context_without_building_baselines(
    monkeypatch, tmp_path, model_family, task_type, name, layerwise, cached
):
    raw = build_raw_dataset_dict(task_type)
    prepared = _run_prepare(
        monkeypatch,
        tmp_path,
        task_type,
        model_family,
        baseline_names=[name],
        dataset=raw,
        embedding_cache={"enabled": cached},
        layerwise_config=(
            LayerwiseConfig(enabled=True, layers=[0, 2]) if layerwise else None
        ),
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
