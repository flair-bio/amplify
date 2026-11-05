"""Test-only selection, shared fitting, and isolated subset artifacts."""

import json
from dataclasses import replace
from types import SimpleNamespace

import polars as pl
import pytest
from datasets import Dataset, DatasetDict
from pydantic import ValidationError

from modules.evaluate.src.contracts.artifacts import (
    EvaluationMetadata,
    PredictOutput,
    PreparedArtifacts,
    ScoreOutput,
)
from modules.evaluate.src.contracts.config import (
    EvaluationConfig,
    ExternalTestSetConfig,
    PredictConfig,
)
from modules.evaluate.src.contracts.config_loader import (
    load_baseline_config,
    load_evaluation_config,
)
from modules.evaluate.src.dataset.contact import prepare_contact_dataset
from modules.evaluate.src.dataset.subsets import select_test_subsets
from modules.evaluate.src.dataset.workspace import get_evaluation_workspace
from modules.evaluate.src.runners import pipeline
from modules.evaluate.src.steps.prepare import PrepareStep
from modules.evaluate.src.tasks import get_task
from modules.evaluate.src.tasks.contact_prediction import (
    contact_development_attention_fingerprint,
)

from .test_contact_data import hf_data, row


def test_subset_config_selects_only_one_edition_for_models_and_baselines():
    paths = [
        "modules/evaluate/configs/config.yaml",
        "modules/evaluate/configs/datasets/contact_prediction_casp.yaml",
        "modules/evaluate/configs/predictors/contact_jacobian.yaml",
    ]
    baseline_paths = [*paths, "modules/evaluate/configs/baselines/contact_prediction.yaml"]
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


def test_external_contact_selection_validates_source_and_chain_units(
    monkeypatch, tmp_path
):
    from modules.evaluate.src.dataset import dataloader
    from .test_contact_embedding import config_for

    config = config_for(tmp_path, tmp_path / "model")
    workspace = get_evaluation_workspace(config.workspace, config.variant, seed=config.seed)
    eu = row(kind="intra_eu", unit_id="D1", positions=[0, 6], valid_positions=[0, 6])
    data = hf_data([row(), eu, row("c15", edition=15)])
    calls = []

    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return data

    monkeypatch.setattr(dataloader, "load_raw_evaluation_dataset", load)
    step = PrepareStep(config.steps.prepare)
    external = ExternalTestSetConfig(
        name="heldout",
        dataset_repo_id="org/contacts",
        dataset_config_name="casp",
        test_subsets={"casp15": {"edition": 15}},
    )
    with pytest.raises(ValueError, match="commit SHA"):
        step.resolve_test_subsets(
            workspace, external.test_subsets, external=external, seed=config.seed
        )
    assert calls == []
    external.dataset_revision = "a" * 40
    subsets = step.resolve_test_subsets(
        workspace, external.test_subsets, external=external, seed=config.seed
    )
    assert set(subsets["casp15"]) == {"test"}
    assert list(subsets["casp15"]["test"]["record_id"]) == ["c15"]
    assert calls[0][1] == {"config_name": "casp", "revision": "a" * 40}
    for filters, error in (
        ({"kind": "global"}, "all scoring units"),
        ({"edition": 17}, "empty"),
    ):
        with pytest.raises(ValueError, match=error):
            step.resolve_test_subsets(
                workspace, {"invalid": filters}, external=external, seed=config.seed
            )


def test_generic_baselines_use_matching_subset_artifacts(tmp_path):
    from modules.evaluate.src.contracts.config import BaselineRunConfig
    from modules.evaluate.src.runners.baselines import run_baselines
    import polars as pl

    data = DatasetDict({
        "train": Dataset.from_dict({"id": ["train"], "sequence": ["ACDE"], "targets": [1]}),
        "validation": Dataset.from_dict({"id": ["val"], "sequence": ["ACDE"], "targets": [0]}),
        "test": Dataset.from_dict({
            "id": ["a", "b"], "sequence": ["ACDE", "FGHI"],
            "targets": [1, 0], "cohort": ["a", "b"],
        }),
    })
    config = BaselineRunConfig.model_validate({
        "workspace": {
            "dataset_name": "toy", "dataset_repo_id": "toy",
            "task_type": "sequence_classification", "target_type": "categorical",
            "label_format": "scalar", "num_labels": 2, "base_path": str(tmp_path),
        },
        "baselines": {"names": ["majority_class"]},
        "test_subsets": {"a": {"cohort": "a"}, "b": {"cohort": "b"}},
    })
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
    from modules.evaluate.src.contracts.artifacts import ContactArtifacts
    from modules.evaluate.src.dataset.contact import tokenize_contact_chain
    from modules.evaluate.src.model.contact_attention import ContactAttentionModel
    from .test_contact_embedding import small_trunk

    raw = hf_data([row("c14"), row("c15", edition=15)])
    first = prepare_contact_dataset(raw, edition=14)
    second = prepare_contact_dataset(raw, edition=15)
    assert first.fingerprint != second.fingerprint
    assert first.development_fingerprint == second.development_fingerprint
    prepared = PreparedArtifacts(
        model=ContactAttentionModel(small_trunk()),
        tokenizer=esm2_tokenizer,
        train_dataloader=None,
        val_dataloader=None,
        test_dataloader=None,
        contact=ContactArtifacts(first),
    )
    task = get_task("contact_prediction")
    identity = task.fitting_identity(prepared)
    assert (
        task.fitting_identity(replace(prepared, contact=ContactArtifacts(second)))
        == identity
    )
    changed_raw = hf_data([row("c14"), row("c15", edition=15)])
    changed_raw["train"] = Dataset.from_list([row("train", "train", positive_pairs=[])])
    changed = prepare_contact_dataset(changed_raw, edition=14)
    assert (
        task.fitting_identity(replace(prepared, contact=ContactArtifacts(changed)))
        != identity
    )
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
    model, control = object(), object()

    class Prepare:
        def __init__(self, _config):
            pass

        def resolve_test_subsets(self, workspace, subsets, *, dataset, **kwargs):
            return select_test_subsets(dataset, subsets)

        def run(self, workspace, *, dataset, **kwargs):
            return PreparedArtifacts(
                model=model,
                tokenizer=None,
                train_dataloader=None,
                val_dataloader=None,
                test_dataloader=dataset["test"],
            )

        def prepare_test_subset(self, workspace, base, name, dataset, **kwargs):
            return replace(base, test_dataloader=dataset["test"])

    class Tune:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, workspace, base, **kwargs):
            calls.append("fit")
            return base

    class Controls(Tune):
        def run(self, workspace, base, **kwargs):
            calls.append("control")
            base.controls.random_trunk_model = control
            return base

    class Predict(Tune):
        def run(self, workspace, base, **kwargs):
            assert base.model is model
            assert base.controls.random_trunk_model is control
            assert list(base.test_dataloader["id"]) == [workspace.test_subset]
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

    class Score(Tune):
        def run(self, *args, **kwargs):
            return ScoreOutput(scores={})

    for name, cls in (
        ("PrepareStep", Prepare),
        ("TuneStep", Tune),
        ("ControlsStep", Controls),
        ("PredictStep", Predict),
        ("ScoreStep", Score),
    ):
        monkeypatch.setattr(pipeline, name, cls)
    root = get_evaluation_workspace(config.workspace, config.variant)
    accelerator = SimpleNamespace(
        is_main_process=True,
        num_processes=1,
        prepare=lambda value: value,
        wait_for_everyone=lambda: None,
    )
    result = pipeline.run_pipeline(config, root, accelerator, dataset=data)
    assert calls == (["fit"] if mode == "tune_probe" else []) + ["control", *names]
    assert set(result.test_subsets) == set(names)
    for name in names:
        workspace = root.test_workspace(name)
        manifest = json.loads(workspace.run_manifest_path.read_text())
        assert manifest["tests"][name]["evaluation"]["filters"] == {
            "cohort": name
        }
        assert manifest["tests"][name]["evaluation"]["status"] == "complete"
    assert not (root.run_root / "tests" / "c").exists()
