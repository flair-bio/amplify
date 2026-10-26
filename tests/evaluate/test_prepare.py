"""Tests for modules.evaluate.src.steps.prepare.PrepareStep.

Parametrized across both model families evaluated in this repo (see eval/):
``flair-bio/amplify-120m`` (custom ``ProteinTokenizer`` + real AMPLIFY head
classes) and ``facebook/esm2_t12_35M_UR50D`` (real HF ``EsmTokenizer`` + real
Esm head classes), both built offline with tiny dimensions via the
``model_family`` fixture (tests/evaluate/conftest.py).

Only the network-bound calls inside ``PrepareStep.run`` (``AutoTokenizer.
from_pretrained``, ``load_evaluation_dataset``, and the Hub download in
``AutoModel*.from_pretrained``) are monkeypatched to these offline, real
equivalents. Everything else (tokenization, collation, DataLoader
construction, ``accelerate.prepare``) runs unmocked, and a genuine forward
pass is run through the real model with the real collated batch.
"""

from __future__ import annotations

import pytest
import torch
from datasets import Dataset, DatasetDict
from accelerate import Accelerator as HFAccelerator
from accelerate.state import AcceleratorState

from modules.evaluate.src.dataset.dataloader import (
    DatasetSplitConfig,
    ExternalTestSetConfig,
)
from modules.evaluate.src.dataset.workspace import (
    EvaluationWorkspaceConfig,
    get_evaluation_workspace,
)
from modules.evaluate.src.steps import prepare as prepare_module
from modules.evaluate.src.schemas.artifacts import PreparedArtifacts
from modules.evaluate.src.steps.prepare import PrepareConfig, PrepareStep
from modules.evaluate.src.utils.embed import find_embedding_dataset

from .conftest import ModelFamily, build_raw_dataset_dict, build_tiny_model


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
    return get_evaluation_workspace(config)


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
    **config_kwargs,
):
    AcceleratorState._reset_state(reset_partial_state=True)
    monkeypatch.setattr(
        prepare_module.AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda model_id, trust_remote_code=True: model_family.tokenizer),
    )
    # Keep this test path CPU-only so it is stable across environments where
    # CUDA may be visible but not fully compatible with the local PyTorch build.
    monkeypatch.setattr(
        prepare_module,
        "Accelerator",
        lambda *args, **kwargs: HFAccelerator(cpu=True),
    )
    monkeypatch.setattr(
        prepare_module,
        "load_evaluation_dataset",
        lambda repo_id, **kwargs: build_raw_dataset_dict(task_type),
    )

    effective_num_labels = 1 if task_type == "sequence_regression" else num_labels

    class _FakeModelCls:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            return build_tiny_model(model_family, task_type, effective_num_labels)

    # get_task() returns a fresh instance per call, so patch the class
    # attribute (shared by every instance) rather than one instance's.
    monkeypatch.setattr(
        type(prepare_module.get_task(task_type)), "auto_model_class", _FakeModelCls
    )

    config = PrepareConfig(
        batch_size=2,
        dataloader_num_workers=0,
        dataloader_persistent_workers=False,
        dataloader_pin_memory=False,
        max_length=64,
        # Keep truncation deterministic across splits/families for simpler
        # assertions.
        random_truncate_by_split={
            "train": False,
            "validation": False,
            "test": False,
        },
        **config_kwargs,
    )
    workspace = _make_workspace(tmp_path, task_type, num_labels=num_labels)
    step = PrepareStep(config)
    prepared = step.run(
        workspace, layerwise_config=layerwise_config, dataset=dataset
    )
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
        prepare_module,
        "load_raw_evaluation_dataset",
        lambda repo_id: external_dataset,
    )
    return prepared, step.prepare_external_test(
        workspace,
        prepared,
        external,
        accelerator=HFAccelerator(cpu=True),
        seed=0,
        layerwise_config=layerwise_config,
    )


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

    train = find_embedding_dataset(prepared.train_dataloader)
    excluded = find_embedding_dataset(
        prepared.refit_dataloaders["tuning_excluded"]
    )
    assert train is not None and excluded is not None
    assert len(train) == len(raw["train"])
    assert excluded.ids == ["refit-only"]
    assert prepared.refit_token_dataloaders["tuning_excluded"] is not None


# ---------------------------------------------------------------------------
# PrepareConfig validation
# ---------------------------------------------------------------------------


class TestPrepareConfig:
    def test_defaults(self):
        cfg = PrepareConfig()
        assert cfg.batch_size == 16
        assert cfg.label_column == "targets"
        assert cfg.sequence_column == "sequence"
        assert cfg.dataloader_num_workers == 8
        assert cfg.tokenize_num_proc == 8

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


# ---------------------------------------------------------------------------
# PrepareStep.run — end to end (with network boundaries mocked)
# ---------------------------------------------------------------------------


class TestPrepareStepRun:
    def test_sequence_classification_end_to_end(
        self, monkeypatch, tmp_path, model_family
    ):
        artifacts = _run_prepare(
            monkeypatch, tmp_path, "sequence_classification", model_family
        )

        assert isinstance(artifacts, PreparedArtifacts)
        assert isinstance(artifacts.model, model_family.sequence_model_cls)
        assert artifacts.train_dataloader is not None
        assert artifacts.val_dataloader is not None
        assert artifacts.test_dataloader is not None

        batch = next(iter(artifacts.train_dataloader))
        assert batch["input_ids"].dtype == torch.long
        assert batch["labels"].dtype == torch.long

        # Full real forward pass through the real (tiny) model with the real batch.
        output = artifacts.model(
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
        assert artifacts.model.config.problem_type == "regression"
        assert artifacts.model.num_labels == 1

        batch = next(iter(artifacts.train_dataloader))
        assert batch["labels"].dtype == torch.float

        output = artifacts.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        assert output.logits.shape == (batch["input_ids"].shape[0], 1)
        assert output.loss is not None

    def test_token_classification_end_to_end(self, monkeypatch, tmp_path, model_family):
        artifacts = _run_prepare(
            monkeypatch,
            tmp_path,
            "token_classification",
            model_family,
            num_labels=2,
        )
        assert isinstance(artifacts.model, model_family.token_model_cls)

        batch = next(iter(artifacts.train_dataloader))
        assert batch["labels"].shape == batch["input_ids"].shape

        output = artifacts.model(
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
        assert swapped.model is prepared.model
        assert swapped.train_dataloader is prepared.train_dataloader
        assert len(prepared.test_dataloader.dataset) == 4
        assert len(swapped.test_dataloader.dataset) == 2
        batch = next(iter(swapped.test_dataloader))
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
        backend = swapped.embedding_backend
        assert backend.trunk is prepared.embedding_backend.trunk
        assert backend.dataset_id == "org/ext"
        assert backend.test_cache_key != prepared.embedding_backend.test_cache_key
        assert len(backend.token_test_dataloader.dataset) == 2
        assert len(swapped.test_dataloader.dataset) == 2
        assert len(prepared.test_dataloader.dataset) == 4

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
