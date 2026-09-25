"""Tests for modules.evaluate.src.steps.predict.PredictStep.

Builds ``PreparedArtifacts`` directly from the tiny, offline ``model_family``
fixtures (tests/evaluate/conftest.py) instead of going through ``PrepareStep``
(which needs Hub access mocked separately in tests/evaluate/test_prepare.py),
so these tests exercise a genuine forward pass (and, for model-based
controls, control-model construction) without any network calls. The
"untrained" control is intentionally not exercised in most tests since it
calls ``from_pretrained`` against a real Hub repo id.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from accelerate import Accelerator
import pytest
from datasets import DatasetDict

from modules.evaluate.src.dataset.collator import (
    EvaluationCollator,
    EvaluationCollatorConfig,
)
from modules.evaluate.src.dataset.dataloader import DataLoaderConfig, build_dataloaders
from modules.evaluate.src.dataset.dataloader import tokenize_dataset
from modules.evaluate.src.dataset.workspace import (
    EvaluationWorkspaceConfig,
    get_evaluation_workspace,
)
from modules.evaluate.src.schemas.artifacts import PreparedArtifacts
from modules.evaluate.src.steps.predict import PredictConfig, PredictStep
from modules.evaluate.src.utils.controls import ControlBuilder
from modules.evaluate.src.utils.inference import run_inference

from .conftest import ModelFamily, build_raw_dataset_dict, build_tiny_model


def test_packed_token_inference_keeps_example_rows():
    class Model(torch.nn.Module):
        def forward(
            self, input_ids, position_ids, cu_seqlens, max_seqlen, num_sequences
        ):
            return SimpleNamespace(
                logits=torch.stack((input_ids.float(), -input_ids.float()), dim=-1)
            )

    class LocalAccelerator:
        num_processes = 1

        @staticmethod
        def gather_for_metrics(values, **kwargs):
            return values

    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0]]),
        "position_ids": torch.tensor([[0, 1, 0, 1, 2, 0, 0, 0]]),
        "cu_seqlens": torch.tensor([0, 2, 5, 8], dtype=torch.int32),
        "max_seqlen": 3,
        "num_sequences": 2,
        "labels": torch.tensor([[-100, 0, -100, 1, 0, -100, -100, -100]]),
        "id": ["first", "second"],
    }
    ids, predictions, labels, _, _ = run_inference(
        Model(), [batch], "token_classification", LocalAccelerator()
    )
    assert ids == ["first", "second"]
    assert predictions == [[0], [0, 0]]
    assert labels == [[0], [1, 0]]


def _build_prepared_artifacts(
    model_family: ModelFamily,
    task_type: str,
    num_labels: int = 2,
    batch_size: int = 2,
) -> PreparedArtifacts:
    tokenizer = model_family.tokenizer
    dataset = tokenize_dataset(
        build_raw_dataset_dict(task_type, num_labels=num_labels),
        tokenizer=tokenizer,
        task_type=task_type,
        sequence_column="sequence",
        label_column="targets",
        max_length=64,
        num_proc=1,
        random_truncate=False,
    )
    collator = EvaluationCollator(
        tokenizer=tokenizer,
        task_type=task_type,
        config=EvaluationCollatorConfig(label_column="targets"),
    )
    train_dl, val_dl, test_dl = build_dataloaders(
        dataset=dataset,
        collator=collator,
        config=DataLoaderConfig(
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        ),
    )
    effective_num_labels = 1 if task_type == "sequence_regression" else num_labels
    model = build_tiny_model(model_family, task_type, effective_num_labels)
    return PreparedArtifacts(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        test_dataloader=test_dl,
    )


def _make_workspace(tmp_path, task_type, num_labels=2):
    target_type = "continuous" if task_type == "sequence_regression" else "categorical"
    label_format = "per_token" if task_type == "token_classification" else "scalar"
    dataset_num_labels = None if task_type == "sequence_regression" else num_labels
    config = EvaluationWorkspaceConfig(
        dataset_name="toy_dataset",
        model_name="toy-model",
        task_type=task_type,
        target_type=target_type,
        label_format=label_format,
        num_labels=dataset_num_labels,
        base_path=str(tmp_path / "eval"),
    )
    workspace = get_evaluation_workspace(config)
    workspace.setup_directories()
    return workspace


# ---------------------------------------------------------------------------
# PredictConfig validation
# ---------------------------------------------------------------------------


class TestPredictConfig:
    def test_defaults(self):
        cfg = PredictConfig()
        assert cfg.collect_probabilities is False
        assert cfg.store_sequences is False
        assert cfg.controls == ["untrained"]
        assert cfg.predictions_filename == "predictions.parquet"

    def test_extra_fields_forbidden(self):
        with pytest.raises(Exception):
            PredictConfig(not_a_real_field=1)


# ---------------------------------------------------------------------------
# PredictStep.run — end to end (offline, tiny real models)
# ---------------------------------------------------------------------------


class TestPredictStepRun:
    def test_writes_predictions_parquet(self, tmp_path, model_family):
        prepared = _build_prepared_artifacts(model_family, "sequence_classification")
        prepared.split = "low_to_high_mutation"
        workspace = _make_workspace(tmp_path, "sequence_classification")

        result = PredictStep(PredictConfig(controls=[])).run(workspace, prepared)

        assert result.predictions is not None
        assert result.predictions_path is not None
        assert result.predictions_path.exists()
        assert result.predictions.columns == ["id", "prediction", "label"]
        # No 'id' column on the toy dataset: falls back to a positional index.
        assert result.predictions["id"].to_list() == list(range(4))
        assert result.metadata.model_name == "toy-model"
        assert result.metadata.dataset_name == "toy_dataset"
        assert result.metadata.task_type == "sequence_classification"
        assert result.metadata.split == "low_to_high_mutation"

    def test_store_sequences_adds_sequence_column(self, tmp_path, model_family):
        prepared = _build_prepared_artifacts(model_family, "sequence_classification")
        workspace = _make_workspace(tmp_path, "sequence_classification")

        result = PredictStep(PredictConfig(store_sequences=True, controls=[])).run(
            workspace, prepared
        )

        assert "sequence" in result.predictions.columns

    def test_collect_probabilities(self, tmp_path, model_family):
        prepared = _build_prepared_artifacts(model_family, "sequence_classification")
        workspace = _make_workspace(tmp_path, "sequence_classification")

        result = PredictStep(
            PredictConfig(collect_probabilities=True, controls=[])
        ).run(workspace, prepared)

        assert "probability" in result.predictions.columns

    def test_uses_dataset_id_column_when_present(self, tmp_path, model_family):
        tokenizer = model_family.tokenizer
        task_type = "sequence_classification"
        raw = build_raw_dataset_dict(task_type)
        raw = DatasetDict(
            {
                split: ds.add_column("id", [f"{split}-{i}" for i in range(ds.num_rows)])
                for split, ds in raw.items()
            }
        )
        dataset = tokenize_dataset(
            raw,
            tokenizer=tokenizer,
            task_type=task_type,
            sequence_column="sequence",
            label_column="targets",
            max_length=64,
            num_proc=1,
            random_truncate=False,
        )
        collator = EvaluationCollator(
            tokenizer=tokenizer,
            task_type=task_type,
            config=EvaluationCollatorConfig(label_column="targets"),
        )
        train_dl, val_dl, test_dl = build_dataloaders(
            dataset=dataset,
            collator=collator,
            config=DataLoaderConfig(
                batch_size=2, num_workers=0, pin_memory=False, persistent_workers=False
            ),
        )
        model = build_tiny_model(model_family, task_type, 2)
        prepared = PreparedArtifacts(
            model=model,
            tokenizer=tokenizer,
            train_dataloader=train_dl,
            val_dataloader=val_dl,
            test_dataloader=test_dl,
        )
        workspace = _make_workspace(tmp_path, task_type)

        result = PredictStep(PredictConfig(controls=[])).run(workspace, prepared)

        assert set(result.predictions["id"].to_list()) == {
            "test-0",
            "test-1",
            "test-2",
            "test-3",
        }

    def test_raises_without_test_dataloader(self, tmp_path, model_family):
        prepared = _build_prepared_artifacts(model_family, "sequence_classification")
        prepared.test_dataloader = None
        workspace = _make_workspace(tmp_path, "sequence_classification")

        with pytest.raises(ValueError, match="test_dataloader"):
            PredictStep(PredictConfig()).run(workspace, prepared)

    @pytest.mark.parametrize(
        "control", ["random_trunk", "random_head", "random_both", "scrambled_sequences"]
    )
    def test_model_based_controls_are_computed_without_prediction_cache(
        self, tmp_path, model_family, control
    ):
        prepared = _build_prepared_artifacts(model_family, "sequence_classification")
        workspace = _make_workspace(tmp_path, "sequence_classification")

        result = PredictStep(PredictConfig(controls=[control])).run(workspace, prepared)

        assert control in result.controls
        expected_columns = (
            ["id", "prediction", "sequence"]
            if control == "scrambled_sequences"
            else ["id", "prediction"]
        )
        assert result.controls[control].columns == expected_columns
        assert control in result.control_paths
        assert result.control_paths[control].exists()
        assert result.requested_controls == [control]

        assert not list(
            workspace.preds_path.glob(
                f"control_{control}_seed710019_*_predictions.parquet"
            )
        )

    def test_random_head_reinitializes_bias(self, tmp_path, model_family):
        prepared = _build_prepared_artifacts(model_family, "sequence_classification")
        workspace = _make_workspace(tmp_path, "sequence_classification")
        bias_name, trained_bias = next(
            (name, parameter)
            for name, parameter in prepared.model.classifier.named_parameters()
            if name.endswith("bias")
        )
        with torch.no_grad():
            trained_bias.fill_(123.0)

        control_model = ControlBuilder._build_control_model(
            "random_head", workspace, prepared, Accelerator(), seed=1
        )

        assert not torch.equal(
            dict(control_model.classifier.named_parameters())[bias_name], trained_bias
        )

    def test_scrambled_labels_needs_no_inference(self, tmp_path, model_family):
        prepared = _build_prepared_artifacts(model_family, "sequence_classification")
        workspace = _make_workspace(tmp_path, "sequence_classification")

        result = PredictStep(PredictConfig(controls=["scrambled_labels"])).run(
            workspace, prepared
        )

        assert "scrambled_labels" not in result.controls
        assert result.requested_controls == ["scrambled_labels"]
        assert not list(workspace.preds_path.glob("control_scrambled_labels_*"))

    def test_majority_class_control_uses_training_split_prior(
        self, tmp_path, model_family
    ):
        prepared = _build_prepared_artifacts(model_family, "sequence_classification")
        workspace = _make_workspace(tmp_path, "sequence_classification")

        result = PredictStep(
            PredictConfig(controls=["majority_class"], collect_probabilities=True)
        ).run(workspace, prepared)

        control_df = result.controls["majority_class"]
        assert control_df.columns == ["id", "prediction", "probability"]
        assert control_df["prediction"].n_unique() == 1
        assert all(sum(row) == pytest.approx(1.0) for row in control_df["probability"])
        assert result.control_paths["majority_class"].exists()

    def test_train_mean_control_uses_training_split_mean(self, tmp_path, model_family):
        prepared = _build_prepared_artifacts(model_family, "sequence_regression")
        workspace = _make_workspace(tmp_path, "sequence_regression")

        result = PredictStep(PredictConfig(controls=["train_mean"])).run(
            workspace, prepared
        )

        control_df = result.controls["train_mean"]
        assert control_df.columns == ["id", "prediction"]
        assert control_df["prediction"].n_unique() == 1
        assert result.control_paths["train_mean"].exists()

    def test_train_mean_control_rejects_classification(self, tmp_path, model_family):
        prepared = _build_prepared_artifacts(model_family, "sequence_classification")
        workspace = _make_workspace(tmp_path, "sequence_classification")

        with pytest.raises(ValueError, match="train_mean"):
            PredictStep(PredictConfig(controls=["train_mean"])).run(workspace, prepared)
