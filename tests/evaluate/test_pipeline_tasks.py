"""Offline execution and disk-scoring round trips for the shared task pipeline."""

import numpy as np
import pytest
from accelerate import Accelerator
from datasets import Dataset, DatasetDict
from transformers import EsmForMaskedLM

from modules.evaluate.src.contracts.config import EvaluationConfig
from modules.evaluate.src.dataset.workspace import get_evaluation_workspace
from modules.evaluate.src.runners.pipeline import run_pipeline
from modules.evaluate.src.steps.score import ScoreStep
from modules.evaluate.src.utils.amino_acids import CANONICAL_AMINO_ACIDS
from modules.pretrain.src.model.modeling_amplify import (
    AMPLIFYForMaskedLM,
    register_amplify_auto_classes,
)

from .conftest import build_raw_dataset_dict


@pytest.mark.parametrize(
    "task,mode,head,cached,metric",
    [
        ("sequence_classification", "tune_probe", "mlp", True, "accuracy"),
        ("sequence_classification", "finetune", "mlp", False, "accuracy"),
        ("sequence_regression", "tune_probe", "sklearn_linear", True, "mse"),
        (
            "sequence_multilabel_classification",
            "tune_probe",
            "torch_linear",
            True,
            "f1",
        ),
        ("token_classification", "tune_probe", "torch_linear", False, "accuracy"),
        ("pseudo_perplexity", "evaluate_as_is", "mlp", False, "pseudo_perplexity"),
        ("calibration", "evaluate_as_is", "mlp", False, "calibration"),
    ],
)
def test_task_pipeline_roundtrip(
    tmp_path, model_family, task, mode, head, cached, metric
):
    auto_map = register_amplify_auto_classes()
    kwargs = dict(model_family.config_kwargs)
    config_cls = kwargs.pop("config_cls")
    model_cls = (
        AMPLIFYForMaskedLM if model_family.name == "amplify-120m" else EsmForMaskedLM
    )
    model_path = tmp_path / "model"
    model_config = config_cls(**kwargs)
    if model_family.name == "amplify-120m":
        model_config.auto_map = auto_map
    model_cls(model_config).save_pretrained(model_path)
    model_family.tokenizer.save_pretrained(model_path)
    if task == "calibration":
        profile = [0.0] * 20
        profile[CANONICAL_AMINO_ACIDS.index("C")] = 1.0
        data = Dataset.from_dict(
            {"id": ["protein"], "sequence": ["ACDE"], "targets": [[[1.0, *profile]]]}
        )
        dataset = DatasetDict({"test": data})
    elif task == "pseudo_perplexity":
        data = Dataset.from_dict({"id": ["protein"], "sequence": ["ACDE"]})
        dataset = DatasetDict({"test": data})
    else:
        dataset = build_raw_dataset_dict(task)
    label_format = {
        "token_classification": "per_token",
        "sequence_multilabel_classification": "multilabel",
        "calibration": "consensus_profile",
    }.get(task, "scalar")
    config = EvaluationConfig.model_validate(
        {
            "seed": 13,
            "workspace": {
                "run_id": "offline",
                "dataset_name": "tiny",
                "dataset_repo_id": str(tmp_path / "data"),
                "model_name": "tiny",
                "model_repo_id": str(model_path),
                "task_type": task,
                "target_type": "continuous"
                if task == "sequence_regression"
                else "categorical",
                "label_format": label_format,
                "num_labels": None
                if task in {"sequence_regression", "calibration"}
                else 2,
                "output_dir": str(tmp_path / "output"),
            },
            "steps": {
                "mode": mode,
                "prepare": {
                    "batch_size": 2,
                    "max_length": 64,
                    "tokenize_num_proc": 1,
                    "dataloader_num_workers": 0,
                    "dataloader_persistent_workers": False,
                    "dataloader_pin_memory": False,
                    "random_truncate_by_split": {
                        "train": False,
                        "validation": False,
                        "test": False,
                    },
                    "embedding_cache": {"enabled": cached, "cache_policy": "off"},
                },
                "tune": {
                    "head": {"head_type": head, "hidden_size": 8},
                    "max_epochs": 1,
                    "hyperparameter_search": {"enabled": False},
                },
                "predict": {"controls": ["random_trunk"]},
                "score": {"metrics": [metric], "bootstrap_enabled": False},
            },
        }
    )
    workspace = get_evaluation_workspace(
        config.workspace, config.variant, seed=config.seed
    )
    result = run_pipeline(config, workspace, Accelerator(cpu=True), dataset=dataset)
    prediction = result.predictions[None]
    assert prediction.predictions.height > 0
    assert set(prediction.controls) == {"random_trunk"}
    assert prediction.metadata.head_type == (None if mode == "evaluate_as_is" else head)
    restored = ScoreStep(config.steps.score).run(workspace)
    assert restored.scores == pytest.approx(result.scores[None].scores, nan_ok=True)
    assert np.isfinite(restored.scores[metric])
