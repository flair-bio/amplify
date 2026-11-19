"""Offline training/test preparation using tiny models and real tokenizers."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from accelerate import Accelerator as HFAccelerator
from accelerate.state import AcceleratorState
from datasets import Dataset, DatasetDict

from modules.evaluate.src.artifacts.workspace import EvaluationWorkspace
from modules.evaluate.src.config import (
    DatasetSplitConfig,
    ExternalTestSetConfig,
    PrepareConfig,
)
from modules.evaluate.src.config import EvaluationWorkspaceConfig
from modules.evaluate.src.dataset import loading as resolution_module
from modules.evaluate.src.dataset import preparation as prepare_module
from modules.evaluate.src.dataset.embeddings import find_embedding_dataset
from modules.evaluate.src.model.factory import Predictor, build_probe

from .conftest import ModelFamily, build_raw_dataset_dict, build_tiny_model
from .helpers import EvaluationInputs


def test_metric_groups_follow_original_prediction_ids():
    import polars as pl

    from modules.evaluate.src.dataset.dataloader import (
        attach_metric_groups,
        metric_groups_by_id,
    )

    dataset = DatasetDict(
        test=Dataset.from_dict(
            {
                "id": ["original-a", "original-b"],
                "cluster_id": [10, 20],
            }
        )
    )
    clusters = metric_groups_by_id(dataset["test"], "cluster_id")
    predictions = pl.DataFrame(
        {
            "id": ["original-b", "original-a"],
            "prediction": [0.0, 1.0],
            "label": [1.0, 0.0],
        }
    )
    annotated = attach_metric_groups(predictions, clusters, "cluster_id")
    assert annotated["cluster_id"].to_list() == [20, 10]
    assert annotated["id"].to_list() == predictions["id"].to_list()
    with pytest.raises(ValueError, match="test IDs"):
        attach_metric_groups(predictions.head(1), clusters, "cluster_id")


@pytest.mark.parametrize(
    "validation_group_column,group_column",
    [("validation_family", None), (None, "cluster_id")],
)
def test_prepare_retains_independent_metric_groups(
    monkeypatch, tmp_path, model_family, validation_group_column, group_column
):
    raw = build_raw_dataset_dict("sequence_regression")
    for split, column in (
        ("validation", validation_group_column),
        ("test", group_column),
    ):
        if column is not None:
            raw[split] = (
                raw[split]
                .add_column(
                    "id", [f"{split}-{index}" for index in range(len(raw[split]))]
                )
                .add_column(column, [10, 10, 20, 20])
            )
    prepared = _run_prepare(
        monkeypatch,
        tmp_path,
        "sequence_regression",
        model_family,
        dataset=raw,
        group_column=group_column,
        validation_group_column=validation_group_column,
    )
    if validation_group_column is None:
        assert prepared.training.validation_metric_groups is None
    else:
        assert prepared.training.validation_metric_groups == {
            f'"validation-{index}"': group
            for index, group in enumerate([10, 10, 20, 20])
        }
    assert prepared.test.test_group_column == group_column
    assert prepared.test.test_metric_groups == (
        {f'"test-{index}"': group for index, group in enumerate([10, 10, 20, 20])}
        if group_column is not None
        else {}
    )


def _make_workspace(tmp_path, task_type, num_labels=2):
    target_type = "continuous" if task_type == "sequence_regression" else "categorical"
    label_format = (
        "per_token"
        if task_type == "token_classification"
        else "multilabel"
        if task_type == "sequence_multilabel_classification"
        else "scalar"
    )
    dataset_num_labels = None if task_type == "sequence_regression" else num_labels
    config = EvaluationWorkspaceConfig(
        run_id="test",
        dataset_name="toy_dataset",
        dataset_repo_id="biomap-research/toy_dataset",
        model_name="toy-model",
        model_repo_id="flair-bio/toy-model",
        task_type=task_type,
        target_type=target_type,
        label_format=label_format,
        num_labels=dataset_num_labels,
        base_path=str(tmp_path / "eval"),
    )
    return EvaluationWorkspace(config)


def _run_prepare(
    monkeypatch,
    tmp_path,
    task_type,
    model_family: ModelFamily,
    num_labels=2,
    layerwise_config=None,
    external=None,
    external_dataset=None,
    dataset=None,
    baseline_names=None,
    group_column=None,
    validation_group_column=None,
    **config_kwargs,
):
    AcceleratorState._reset_state(reset_partial_state=True)
    accelerator = HFAccelerator(cpu=True)
    monkeypatch.setattr(
        resolution_module,
        "load_evaluation_dataset",
        lambda repo_id, **kwargs: build_raw_dataset_dict(task_type),
    )
    effective_num_labels = 1 if task_type == "sequence_regression" else num_labels
    config = PrepareConfig(
        batch_size=2,
        dataloader_num_workers=0,
        dataloader_persistent_workers=False,
        dataloader_pin_memory=False,
        max_length=64,
        random_truncate_by_split={"train": False, "validation": False, "test": False},
        **config_kwargs,
    )
    workspace = _make_workspace(tmp_path, task_type, num_labels=num_labels)
    workspace.setup_directories()
    model = build_tiny_model(model_family, task_type, effective_num_labels)
    training, test = prepare_module.prepare_training_data(
        config,
        workspace,
        model,
        model_family.tokenizer,
        accelerator=accelerator,
        layerwise_config=layerwise_config,
        dataset=dataset,
        baseline_names=baseline_names,
        group_column=group_column,
        validation_group_column=validation_group_column,
    )
    predictor = Predictor(
        build_probe(
            model,
            model_family.tokenizer,
            task_type,
            effective_num_labels,
            embedding_size=training.embedding.backend.hidden_size
            if training.embedding
            else None,
            layerwise_config=layerwise_config,
        ),
        model_family.tokenizer,
        model if training.embedding is not None else None,
    )
    from modules.evaluate.src.model.controls import ControlPredictors

    prepared = EvaluationInputs(predictor, training, test, ControlPredictors())
    if external is None:
        return prepared
    if external_dataset is None:
        external_dataset = DatasetDict(
            {
                "test": Dataset.from_dict(
                    {"sequence": ["ACDEFG", "MKVLAAGHW"], "targets": [1, 0]}
                )
            }
        )
    monkeypatch.setattr(
        resolution_module,
        "load_raw_evaluation_dataset",
        lambda repo_id, **kwargs: external_dataset,
    )
    external_test = prepare_module.prepare_test_data(
        config,
        workspace,
        predictor.trunk,
        predictor.tokenizer,
        test,
        external,
        accelerator=HFAccelerator(cpu=True),
        seed=0,
        layerwise_config=layerwise_config,
    )
    return (prepared, replace(prepared, test=external_test))


@pytest.mark.parametrize("cached", [False, True])
def test_prepare_selected_test_data_reuses_model_and_controls(
    monkeypatch, tmp_path, model_family, cached
):
    prepared = _run_prepare(
        monkeypatch,
        tmp_path,
        "sequence_classification",
        model_family,
        embedding_cache={"enabled": cached, "cache_policy": "off"},
    )
    workspace = _make_workspace(tmp_path, "sequence_classification")
    subset = DatasetDict(
        {
            "test": Dataset.from_dict(
                {"sequence": ["ACDEFG"], "targets": [1], "id": ["selected"]}
            )
        }
    )
    monkeypatch.setattr(
        resolution_module,
        "load_raw_evaluation_dataset",
        lambda *args, **kwargs: pytest.fail("subset reloaded the dataset"),
    )
    config = PrepareConfig(
        batch_size=2,
        tokenize_num_proc=1,
        embedding_cache={"enabled": cached, "cache_policy": "off"},
    )
    source = ExternalTestSetConfig(
        name="selected",
        dataset_repo_id=workspace.dataset_repo_id or workspace.dataset_name,
        dataset_config_name=workspace.config.dataset_config_name,
        dataset_revision=workspace.config.dataset_revision,
        split=config.split.test,
    )
    selected_test = prepare_module.prepare_test_data(
        config,
        workspace,
        prepared.predictor.trunk,
        prepared.predictor.tokenizer,
        prepared.test,
        source,
        accelerator=HFAccelerator(cpu=True),
        seed=0,
        dataset=subset,
    )
    selected = replace(prepared, test=selected_test)
    assert selected.predictor.model is prepared.predictor.model
    assert selected.controls is prepared.controls
    assert selected.training.train_dataloader is prepared.training.train_dataloader
    assert selected.training.val_dataloader is prepared.training.val_dataloader
    assert len(selected.test.test_dataloader.dataset) == 1
    assert len(prepared.test.test_dataloader.dataset) == 4
    if cached:
        assert selected.predictor.trunk is prepared.predictor.trunk
        assert (
            selected.test.embedding.backend.test_cache_key
            != prepared.test.embedding.backend.test_cache_key
        )
        assert len(selected.test.embedding.backend.token_test_dataloader.dataset) == 1


def test_prepare_caches_configured_final_refit_split_separately(
    monkeypatch, tmp_path, model_family
):
    raw = build_raw_dataset_dict("sequence_classification")
    raw["tuning_excluded"] = Dataset.from_dict(
        {"sequence": ["ACDEFG"], "targets": [1], "id": ["refit-only"]}
    )
    prepared = _run_prepare(
        monkeypatch,
        tmp_path,
        "sequence_classification",
        model_family,
        split=DatasetSplitConfig(refit_additional_splits=["tuning_excluded"]),
        embedding_cache={"enabled": True, "cache_policy": "off"},
        dataset=raw,
    )
    train = find_embedding_dataset(prepared.training.train_dataloader)
    excluded = find_embedding_dataset(
        prepared.training.refit_dataloaders["tuning_excluded"]
    )
    assert train is not None and excluded is not None
    assert len(train) == len(raw["train"])
    assert excluded.ids == ["refit-only"]
    assert prepared.training.refit_token_dataloaders["tuning_excluded"] is not None


class TestPrepareConfig:
    def test_defaults(self):
        cfg = PrepareConfig()
        assert cfg.batch_size == 8
        assert cfg.label_column == "targets"
        assert cfg.sequence_column == "sequence"
        assert cfg.dataloader_num_workers == 0
        assert cfg.tokenize_num_proc == 4

    def test_worker_counts_can_be_tuned_independently(self):
        cfg = PrepareConfig(dataloader_num_workers=0, tokenize_num_proc=2)
        assert cfg.dataloader_num_workers == 0
        assert cfg.tokenize_num_proc == 2

    @pytest.mark.parametrize(
        "overrides",
        [
            {"dataloader_num_workers": -1},
            {"dataloader_num_workers": None},
            {"tokenize_num_proc": 0},
            {"tokenize_num_proc": -1},
            {"tokenize_num_proc": None},
        ],
    )
    def test_invalid_worker_counts(self, overrides):
        with pytest.raises(ValueError):
            PrepareConfig(**overrides)

    def test_extra_fields_forbidden(self):
        with pytest.raises(Exception):
            PrepareConfig(not_a_real_field=1)


class TestPreparation:
    def test_sequence_classification_end_to_end(
        self, monkeypatch, tmp_path, model_family
    ):
        artifacts = _run_prepare(
            monkeypatch, tmp_path, "sequence_classification", model_family
        )
        assert isinstance(artifacts, EvaluationInputs)
        assert isinstance(artifacts.predictor.model, model_family.sequence_model_cls)
        assert artifacts.training.train_dataloader is not None
        assert artifacts.training.val_dataloader is not None
        assert artifacts.test.test_dataloader is not None
        batch = next(iter(artifacts.training.train_dataloader))
        assert batch["input_ids"].dtype == torch.long
        assert batch["labels"].dtype == torch.long
        output = artifacts.predictor.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        assert output.loss is not None
        assert output.logits.shape == (batch["input_ids"].shape[0], 2)

    def test_sequence_regression_uses_float_labels_and_problem_type(
        self, monkeypatch, tmp_path, model_family
    ):
        artifacts = _run_prepare(
            monkeypatch, tmp_path, "sequence_regression", model_family
        )
        assert artifacts.predictor.model.config.problem_type == "regression"
        assert artifacts.predictor.model.num_labels == 1
        batch = next(iter(artifacts.training.train_dataloader))
        assert batch["labels"].dtype == torch.float
        output = artifacts.predictor.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        assert output.logits.shape == (batch["input_ids"].shape[0], 1)
        assert output.loss is not None

    def test_token_classification_end_to_end(self, monkeypatch, tmp_path, model_family):
        artifacts = _run_prepare(
            monkeypatch, tmp_path, "token_classification", model_family, num_labels=2
        )
        assert isinstance(artifacts.predictor.model, model_family.token_model_cls)
        batch = next(iter(artifacts.training.train_dataloader))
        assert batch["labels"].shape == batch["input_ids"].shape
        output = artifacts.predictor.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        assert output.loss is not None
        assert output.logits.shape == (*batch["input_ids"].shape, 2)

    def test_external_test_set_replaces_only_test_loader(
        self, monkeypatch, tmp_path, model_family
    ):
        external = ExternalTestSetConfig(name="ext", dataset_repo_id="org/ext")
        prepared, swapped = _run_prepare(
            monkeypatch,
            tmp_path,
            "sequence_classification",
            model_family,
            external=external,
        )
        assert swapped.predictor.model is prepared.predictor.model
        assert swapped.training.train_dataloader is prepared.training.train_dataloader
        assert len(prepared.test.test_dataloader.dataset) == 4
        assert len(swapped.test.test_dataloader.dataset) == 2
        batch = next(iter(swapped.test.test_dataloader))
        assert batch["labels"].tolist() == [1, 0]

    def test_external_test_set_embeds_with_cached_trunk(
        self, monkeypatch, tmp_path, model_family
    ):
        external = ExternalTestSetConfig(name="ext", dataset_repo_id="org/ext")
        prepared, swapped = _run_prepare(
            monkeypatch,
            tmp_path,
            "sequence_classification",
            model_family,
            external=external,
            embedding_cache={"enabled": True, "cache_policy": "off"},
        )
        backend = swapped.test.embedding.backend
        assert backend.dataset_id == "org/ext"
        assert backend.test_cache_key != prepared.test.embedding.backend.test_cache_key
        assert len(backend.token_test_dataloader.dataset) == 2
        assert len(swapped.test.test_dataloader.dataset) == 2
        assert len(prepared.test.test_dataloader.dataset) == 4

    @pytest.mark.parametrize(
        "external_dataset",
        [
            DatasetDict(
                {"train": Dataset.from_dict({"sequence": ["ACDEFG"], "targets": [1]})}
            ),
            DatasetDict(
                {"test": Dataset.from_dict({"sequence": ["ACDEFG"], "targets": [2]})}
            ),
        ],
        ids=["missing-test-split", "out-of-range-label"],
    )
    def test_external_test_set_rejects_invalid_input(
        self, monkeypatch, tmp_path, model_family, external_dataset
    ):
        external = ExternalTestSetConfig(name="ext", dataset_repo_id="org/ext")
        with pytest.raises(ValueError):
            _run_prepare(
                monkeypatch,
                tmp_path,
                "sequence_classification",
                model_family,
                external=external,
                external_dataset=external_dataset,
            )

    def test_setup_directories_called(self, monkeypatch, tmp_path, model_family):
        _run_prepare(monkeypatch, tmp_path, "sequence_classification", model_family)
        workspace = _make_workspace(tmp_path, "sequence_classification")
        assert workspace.stats_path.parent.parent.exists()
