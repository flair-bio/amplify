"""Test-only selection, shared fitting, and isolated subset artifacts."""

import json
from types import SimpleNamespace

import polars as pl
import pytest
import torch
from datasets import Dataset, DatasetDict
from pydantic import ValidationError

from modules.evaluate.src.artifacts.results import (
    EvaluationMetadata,
    FitResult,
    PredictOutput,
)
from modules.evaluate.src.artifacts.workspace import EvaluationWorkspace
from modules.evaluate.src.config import EvaluationConfig, PredictConfig
from modules.evaluate.src.utils.config_utils import (
    load_baseline_config,
    load_evaluation_config,
)
from modules.evaluate.src.dataset.contact import prepare_contact_dataset
from modules.evaluate.src.dataset.subsets import select_test_subsets
from modules.evaluate.src.dataset.types import TestData as HeldOutData
from modules.evaluate.src.dataset.types import TrainingData
from modules.evaluate.src.model.contact_features import (
    contact_development_attention_fingerprint,
    contact_fitting_identity,
)
from modules.evaluate.src.model.controls import ControlPredictors
from modules.evaluate.src.model.factory import Predictor
from modules.evaluate.src.runners import flat_evaluation as flat
from modules.evaluate.src.runners import pipeline
from modules.evaluate.src.runners.runtime import Runtime

from .test_contact_data import hf_data, row
from .test_pipeline_external import _stub_services


def test_subset_config_selects_only_one_edition_for_models_and_baselines():
    paths = [
        "modules/evaluate/configs/config.yaml",
        "modules/evaluate/configs/datasets/contact_prediction_casp.yaml",
        "modules/evaluate/configs/predictors/contact_jacobian.yaml",
    ]
    baseline_paths = [
        *paths,
        "modules/evaluate/configs/baselines/contact_prediction.yaml",
    ]
    overrides = [
        "steps.predict.test_subsets.casp14=null",
        "steps.predict.test_subsets.casp16=null",
    ]
    assert load_evaluation_config(paths, overrides).steps.predict.test_subsets == {
        "casp15": {"edition": 15}
    }
    assert load_baseline_config(baseline_paths, overrides).test_subsets == {
        "casp15": {"edition": 15}
    }
    assert (
        load_evaluation_config(
            paths, ["steps.predict.test_subsets=null"]
        ).steps.predict.test_subsets
        == {}
    )
    with pytest.raises(ValidationError, match="Extra inputs"):
        config = load_evaluation_config(paths)
        config.workspace.model_validate(
            {**config.workspace.model_dump(), "edition": 14}
        )


@pytest.mark.parametrize(
    "subsets",
    [
        {"../escape": {"edition": 14}},
        {"empty": {}},
        {"bad": {"": 14}},
        {"bad": {"edition": [14, 15]}},
        {"bad": {"edition": float("nan")}},
    ],
)
def test_subset_config_rejects_unsafe_or_unsupported_filters(subsets):
    with pytest.raises(ValidationError):
        PredictConfig(test_subsets=subsets)


def test_selection_preserves_development_and_all_chain_units():
    eu = row(
        "c14", kind="intra_eu", unit_id="D1", positions=[0, 6], valid_positions=[0, 6]
    )
    data = hf_data([row("c14"), eu, row("c15", edition=15), row("c16", edition=16)])
    selected = select_test_subsets(
        data, {"casp15": {"edition": 15}}, group_column="record_id"
    )
    assert list(selected) == ["casp15"]
    subset = selected["casp15"]
    assert subset["train"] is data["train"]
    assert subset["validation"] is data["validation"]
    assert list(subset["test"]["record_id"]) == ["c15"]
    all_units = select_test_subsets(
        data, {"casp14": {"edition": 14}}, group_column="record_id"
    )
    assert len(all_units["casp14"]["test"]) == 2
    with pytest.raises(ValueError, match="all scoring units"):
        select_test_subsets(
            data, {"global": {"kind": "global"}}, group_column="record_id"
        )
    with pytest.raises(ValueError, match="missing columns"):
        select_test_subsets(data, {"missing": {"typo": 14}})
    with pytest.raises(ValueError, match="empty"):
        select_test_subsets(data, {"empty": {"edition": 17}})


def test_generic_baselines_use_matching_subset_artifacts(tmp_path):
    import polars as pl

    from modules.evaluate.src.config import BaselineRunConfig
    from modules.evaluate.src.runners.baselines import run_baselines

    data = DatasetDict(
        {
            "train": Dataset.from_dict(
                {"id": ["train"], "sequence": ["ACDE"], "targets": [1]}
            ),
            "validation": Dataset.from_dict(
                {"id": ["val"], "sequence": ["ACDE"], "targets": [0]}
            ),
            "test": Dataset.from_dict(
                {
                    "id": ["a", "b"],
                    "sequence": ["ACDE", "FGHI"],
                    "targets": [1, 0],
                    "cohort": ["a", "b"],
                }
            ),
        }
    )
    config = BaselineRunConfig.model_validate(
        {
            "workspace": {
                "dataset_name": "toy",
                "dataset_repo_id": "toy",
                "task_type": "sequence_classification",
                "target_type": "categorical",
                "label_format": "scalar",
                "num_labels": 2,
                "base_path": str(tmp_path),
            },
            "baselines": {"names": ["majority_class"]},
            "test_subsets": {"a": {"cohort": "a"}, "b": {"cohort": "b"}},
        }
    )
    outputs = run_baselines(config, dataset=data)
    fit_directories = []
    for name, subset in select_test_subsets(data, config.test_subsets).items():
        path = outputs[f"tests/{name}"]["majority_class"]
        frame = pl.read_parquet(path)
        assert frame["id"].to_list() == [name]
        fit_directories.append(path.parents[1])
    assert fit_directories[0] == fit_directories[1]


def test_fitting_identity_changes_with_development_labels_not_test_selection(
    esm2_tokenizer,
):
    from modules.evaluate.src.dataset.contact import tokenize_contact_chain
    from modules.evaluate.src.model.contact_attention import ContactAttentionModel

    from .test_contact_embedding import small_trunk

    raw = hf_data([row("c14"), row("c15", edition=15)])
    first = prepare_contact_dataset(raw, edition=14)
    second = prepare_contact_dataset(raw, edition=15)
    assert first.fingerprint != second.fingerprint
    assert first.development_fingerprint == second.development_fingerprint
    predictor = Predictor(ContactAttentionModel(small_trunk()), esm2_tokenizer)
    identity = contact_fitting_identity(predictor, first, {})
    assert contact_fitting_identity(predictor, second, {}) == identity
    changed_raw = hf_data([row("c14"), row("c15", edition=15)])
    changed_raw["train"] = Dataset.from_list([row("train", "train", positive_pairs=[])])
    changed = prepare_contact_dataset(changed_raw, edition=14)
    assert contact_fitting_identity(predictor, changed, {}) != identity
    tokens = [
        tokenize_contact_chain(chain, esm2_tokenizer) for chain in first.splits["train"]
    ]
    metadata = dict(
        weights="same", backbone_config={}, tf32=False, matmul_precision="highest"
    )
    assert contact_development_attention_fingerprint(
        first, tokens, **metadata
    ) == contact_development_attention_fingerprint(second, tokens, **metadata)
    assert contact_development_attention_fingerprint(
        first, tokens, **metadata
    ) != contact_development_attention_fingerprint(changed, tokens, **metadata)


@pytest.mark.parametrize("mode", ["tune_probe", "evaluate_as_is"])
@pytest.mark.parametrize("names", [["a", "b"], ["b"]])
def test_pipeline_fits_once_and_predicts_only_selected_subsets(
    monkeypatch, tmp_path, mode, names
):
    config = EvaluationConfig.model_validate(
        {
            "workspace": {
                "run_id": "subsets",
                "dataset_name": "toy",
                "dataset_repo_id": "toy",
                "dataset_config_name": "subset-view",
                "dataset_revision": "a" * 40,
                "model_name": "m",
                "task_type": "sequence_classification",
                "target_type": "categorical",
                "label_format": "scalar",
                "num_labels": 2,
                "base_path": str(tmp_path),
            },
            "steps": {
                "mode": mode,
                "predict": {"test_subsets": {name: {"cohort": name} for name in names}},
            },
        }
    )
    data = DatasetDict(
        {
            "train": Dataset.from_list([{"id": "train"}]),
            "validation": Dataset.from_list([{"id": "validation"}]),
            "test": Dataset.from_list(
                [{"id": name, "cohort": name} for name in ("a", "b", "c")]
            ),
        }
    )
    calls = []
    model = _stub_services(monkeypatch, [])
    control = object()

    def prepare(config, workspace, model, tokenizer, *, dataset, **kwargs):
        return (TrainingData(None, None), HeldOutData(dataset["test"]))

    def fit(**kwargs):
        calls.append("fit")
        return FitResult(model=kwargs["predictor"].model, seed=kwargs["seed"])

    def controls(*args, **kwargs):
        calls.append("control")
        return ControlPredictors(random_trunk_model=control)

    def predict(config, workspace, predictor, training, test, controls, **kwargs):
        assert predictor.model is model
        assert controls.random_trunk_model is control
        assert list(test.test_dataloader["id"]) == [workspace.test_subset]
        assert workspace.model_path == root.model_path
        calls.append(workspace.test_subset)
        return PredictOutput(
            metadata=EvaluationMetadata(
                model_name="m",
                dataset_name="toy",
                task_type="sequence_classification",
                test_subset=workspace.test_subset,
            ),
            predictions=pl.DataFrame(),
        )

    def prepare_test(
        config, workspace, trunk, tokenizer, test, source, *, dataset, **kwargs
    ):
        assert set(dataset) == {"test"}
        assert list(dataset["test"]["id"]) == [source.name]
        assert source.dataset_repo_id == workspace.dataset_repo_id
        assert source.dataset_config_name == workspace.config.dataset_config_name
        assert source.dataset_revision == workspace.config.dataset_revision
        assert source.split == config.split.test
        return HeldOutData(dataset["test"])

    monkeypatch.setattr(flat, "prepare_training_data", prepare)
    monkeypatch.setattr(flat, "prepare_test_data", prepare_test)
    monkeypatch.setattr(flat, "fit_predictor", fit)
    monkeypatch.setattr(flat, "prepare_control_predictors", controls)
    monkeypatch.setattr(flat, "predict_test_set", predict)
    root = EvaluationWorkspace(config.workspace, config.variant, seed=config.seed)
    accelerator = SimpleNamespace(
        is_main_process=True,
        num_processes=1,
        prepare=lambda value: value,
        wait_for_everyone=lambda: None,
        device=torch.device("cpu"),
    )
    result = pipeline.run_evaluation(
        config, Runtime(torch.device("cpu"), accelerator), dataset=data
    )
    assert calls == (["fit"] if mode == "tune_probe" else []) + ["control", *names]
    assert set(result.test_subsets) == set(names)
    for name in names:
        workspace = root.test_workspace(name)
        manifest = json.loads(workspace.run_manifest_path.read_text())
        assert manifest["tests"][name]["evaluation"]["filters"] == {"cohort": name}
        assert manifest["tests"][name]["evaluation"]["status"] == "complete"
    assert not (root.run_root / "tests" / "c").exists()
