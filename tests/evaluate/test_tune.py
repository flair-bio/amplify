import json
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from accelerate import Accelerator
from torch import nn

from modules.evaluate.src.schemas.artifacts import PreparedArtifacts
from modules.evaluate.src.steps.tune import (
    HyperparameterSearchConfig,
    HyperparameterSearchSpaceConfig,
    TuneConfig,
    TuneStep,
)
from modules.evaluate.src.utils.embed import EmbeddingCacheConfig
from modules.evaluate.src.model.heads import HeadConfig
from modules.evaluate.src.utils.hpo import (
    checkpoint_state,
    resolve_search_direction,
    sample_candidates,
)
from modules.evaluate.src.utils.embed import EmbeddingDataset

from .test_prepare import _run_prepare


def test_checkpoint_state_is_independent_cpu_copy() -> None:
    model = nn.Linear(2, 1)

    checkpoint = checkpoint_state(model)

    assert all(tensor.device.type == "cpu" for tensor in checkpoint.values())
    with torch.no_grad():
        model.weight.add_(1)
    assert not torch.equal(checkpoint["weight"], model.weight.cpu())


def test_random_candidates_are_unique_and_bounded_by_grid() -> None:
    search_space = HyperparameterSearchSpaceConfig(
        learning_rate=[1e-4, 1e-3],
        weight_decay=[0.0],
        max_epochs=[1, 2],
    )
    step = TuneStep(
        TuneConfig(
            hyperparameter_search=HyperparameterSearchConfig(search_space=search_space)
        )
    )

    candidates = step._generate_candidates("random", n_trials=10, seed=42)

    assert len(candidates) == 4
    assert len({tuple(candidate.items()) for candidate in candidates}) == 4


def test_removed_tuning_aliases_are_rejected() -> None:
    with pytest.raises(Exception, match="num_epochs"):
        TuneConfig(num_epochs=1)
    with pytest.raises(Exception, match="seeds"):
        HyperparameterSearchSpaceConfig(seeds=[42])


def test_sample_candidates_is_deterministic_and_without_replacement() -> None:
    candidates = [{"value": value} for value in range(5)]

    first = sample_candidates(candidates, n_trials=3, seed=42)
    second = sample_candidates(candidates, n_trials=3, seed=42)

    assert first == second
    assert len(first) == len({candidate["value"] for candidate in first}) == 3


def test_gather_embeddings_unwraps_accelerate_dataset_wrappers() -> None:
    embeddings = torch.randn(3, 4)
    labels = torch.tensor([0, 1, 0])
    embedding_dataset = EmbeddingDataset(embeddings, ["a", "b", "c"], labels)

    class DatasetWrapper:
        def __init__(self, dataset):
            self.dataset = dataset

    class LoaderWrapper:
        def __init__(self, dataset):
            self.dataset = dataset

    wrapped_loader = LoaderWrapper(DatasetWrapper(embedding_dataset))
    accelerator = MagicMock()

    actual_embeddings, actual_labels = TuneStep._gather_embeddings(
        wrapped_loader, accelerator
    )

    np.testing.assert_array_equal(actual_embeddings, embeddings.numpy())
    np.testing.assert_array_equal(actual_labels, labels.numpy())
    accelerator.gather_for_metrics.assert_not_called()


def test_resolve_direction_flips_to_minimize_for_lower_is_better_metric() -> None:
    # "mse" (the sequence_regression default metric) is lower-is-better, so
    # an unset direction must resolve to minimize.
    step = TuneStep(TuneConfig())

    assert step._resolve_direction("mse") == "minimize"
    assert step._resolve_direction("f1") == "maximize"


def test_resolve_direction_respects_explicit_direction() -> None:
    # An explicitly-set direction (even one that seems mismatched) must be
    # honored rather than overridden.
    step = TuneStep(
        TuneConfig(
            hyperparameter_search=HyperparameterSearchConfig(direction="maximize")
        )
    )

    assert step._resolve_direction("mse") == "maximize"


def test_resolve_search_direction_uses_metric_default_unless_explicit() -> None:
    assert resolve_search_direction("mse", None) == "minimize"
    assert resolve_search_direction("mse", "maximize") == "maximize"


def test_save_model_persists_model_and_tokenizer_on_main_process(tmp_path) -> None:
    step = TuneStep(TuneConfig())
    workspace = MagicMock(model_path=tmp_path / "model")

    model = MagicMock()
    tokenizer = MagicMock()
    prepared = PreparedArtifacts(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=None,
        val_dataloader=None,
        test_dataloader=None,
    )

    accelerator = MagicMock(is_main_process=True)
    accelerator.unwrap_model.return_value = model

    best_hparams = {"learning_rate": 1e-4, "weight_decay": 1e-4, "max_epochs": 10}
    step._save_model(workspace, prepared, model, accelerator, best_hparams, seed=710019)

    assert workspace.model_path.is_dir()
    accelerator.unwrap_model.assert_called_once_with(model)
    model.save_pretrained.assert_called_once_with(
        workspace.model_path,
        is_main_process=True,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict.return_value,
    )
    tokenizer.save_pretrained.assert_called_once_with(workspace.model_path)
    accelerator.wait_for_everyone.assert_called_once()

    tune_config_path = workspace.model_path / "tune_config.json"
    assert tune_config_path.exists()
    on_disk = json.loads(tune_config_path.read_text())
    assert on_disk["seed"] == 710019
    assert on_disk["best_hyperparameters"] == best_hparams
    assert on_disk["config"]["head"]["head_type"] == "mlp"


def test_save_model_skips_tokenizer_save_on_non_main_process(tmp_path) -> None:
    step = TuneStep(TuneConfig())
    workspace = MagicMock(model_path=tmp_path / "model")

    model = MagicMock()
    tokenizer = MagicMock()
    prepared = PreparedArtifacts(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=None,
        val_dataloader=None,
        test_dataloader=None,
    )

    accelerator = MagicMock(is_main_process=False)
    accelerator.unwrap_model.return_value = model

    best_hparams = {"learning_rate": 1e-4, "weight_decay": 1e-4, "max_epochs": 10}
    step._save_model(workspace, prepared, model, accelerator, best_hparams, seed=710019)

    model.save_pretrained.assert_called_once_with(
        workspace.model_path,
        is_main_process=False,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict.return_value,
    )
    tokenizer.save_pretrained.assert_not_called()
    accelerator.wait_for_everyone.assert_called_once()
    assert not (workspace.model_path / "tune_config.json").exists()


class TestVarianceSeeds:
    """variance_seeds is a diagnostic: it must not change the returned model,
    only log validation-metric mean/std across extra training seeds."""

    def test_torch_probe_variance_diagnostic_restores_canonical_weights(
        self, monkeypatch, tmp_path, model_family, caplog
    ):
        artifacts = _run_prepare(
            monkeypatch,
            tmp_path,
            "sequence_classification",
            model_family,
            embedding_cache=EmbeddingCacheConfig(enabled=True),
        )
        workspace = MagicMock(
            task_type="sequence_classification", model_path=tmp_path / "model"
        )
        step = TuneStep(
            TuneConfig(
                head=HeadConfig(head_type="mlp"),
                max_epochs=1,
                early_stopping_patience=None,
                save_best_model=False,
                variance_seeds=[1, 2],
            )
        )
        trained = step.run(workspace, artifacts, seed=42)
        canonical_state = {
            name: tensor.clone() for name, tensor in trained.model.state_dict().items()
        }

        accelerator = Accelerator()
        with caplog.at_level("INFO"):
            step._report_torch_seed_variance(
                workspace,
                trained,
                accelerator,
                {"learning_rate": 1e-3, "weight_decay": 0.0, "max_epochs": 1},
            )

        assert "Seed variance (2 extra seeds" in caplog.text
        for name, tensor in trained.model.state_dict().items():
            assert torch.equal(tensor, canonical_state[name])

    def test_classical_head_logs_variance(self, monkeypatch, tmp_path, model_family, caplog):
        artifacts = _run_prepare(
            monkeypatch,
            tmp_path,
            "sequence_classification",
            model_family,
            embedding_cache=EmbeddingCacheConfig(enabled=True),
        )
        workspace = MagicMock(
            task_type="sequence_classification", model_path=tmp_path / "model"
        )
        step = TuneStep(
            TuneConfig(
                head=HeadConfig(head_type="sklearn_linear"),
                save_best_model=False,
                variance_seeds=[1, 2],
            )
        )

        with caplog.at_level("INFO"):
            step.run(workspace, artifacts, seed=42)

        assert "Seed variance (2 extra seeds" in caplog.text


class TestBohbAcrossTaskTypesAndSettings:
    """Exercise one representative gradient-path BOHB run end to end."""

    def test_bohb_trains_successfully(
        self,
        monkeypatch,
        tmp_path,
        model_family,
    ):
        pytest.importorskip("optuna")
        task_type = "sequence_classification"
        embedding_cache_enabled = True
        head_type = "mlp"
        artifacts = _run_prepare(
            monkeypatch,
            tmp_path,
            task_type,
            model_family,
            embedding_cache=EmbeddingCacheConfig(enabled=embedding_cache_enabled),
        )
        workspace = MagicMock(task_type=task_type, model_path=tmp_path / "model")
        step = TuneStep(
            TuneConfig(
                head=HeadConfig(head_type=head_type),
                max_epochs=1,
                accumulation_steps=2,
                save_best_model=False,
                hyperparameter_search=HyperparameterSearchConfig(
                    enabled=True,
                    n_trials=2,
                    search_space=HyperparameterSearchSpaceConfig(
                        learning_rate=[1e-3, 1e-2],
                        weight_decay=[0.0],
                        max_epochs=[1],
                    ),
                ),
            )
        )

        result = step.run(workspace, artifacts, seed=42)

        assert result.model is not None


# Search spaces small enough to run quickly under bohb's real Optuna study,
# but with >1 candidate value so bohb actually has something to search over
# (sweep_classical_head's bohb branch is a no-op without one -- see heads.py).
_CLASSICAL_HEAD_SEARCH_SPACES: dict[str, dict[str, list]] = {
    "knn": {"n_neighbors": [1, 3]},
}


class TestBohbAcrossClassicalHeadTypes:
    """Exercise one representative classical-head BOHB run."""

    def test_bohb_trains_successfully(
        self, monkeypatch, tmp_path, model_family
    ):
        pytest.importorskip("optuna")
        task_type = "sequence_classification"
        head_type = "knn"

        artifacts = _run_prepare(
            monkeypatch,
            tmp_path,
            task_type,
            model_family,
            embedding_cache=EmbeddingCacheConfig(enabled=True),
        )
        workspace = MagicMock(task_type=task_type, model_path=tmp_path / "model")
        step = TuneStep(
            TuneConfig(
                head=HeadConfig(
                    head_type=head_type,
                    search_space=_CLASSICAL_HEAD_SEARCH_SPACES[head_type],
                ),
                save_best_model=False,
                hyperparameter_search=HyperparameterSearchConfig(
                    enabled=True, n_trials=2, search_pattern="bohb"
                ),
            )
        )

        result = step.run(workspace, artifacts, seed=42)

        assert hasattr(result.model, "predict")
