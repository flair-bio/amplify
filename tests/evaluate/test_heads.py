"""Tests for modules.evaluate.src.model.heads (pluggable, gradient-free heads)
and their wiring into TuneStep/EvaluationConfig.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from modules.evaluate.src.config import EvaluationConfig
from modules.evaluate.src.dataset.collator import (
    EvaluationCollator,
    EvaluationCollatorConfig,
)
from modules.evaluate.src.dataset.dataloader import (
    DataLoaderConfig,
    build_dataloaders,
    tokenize_dataset,
)
from modules.evaluate.src.dataset.workspace import (
    EvaluationWorkspaceConfig,
    get_evaluation_workspace,
)
from modules.evaluate.src.model.heads import (
    HeadConfig,
    _filter_classical_search_space,
    build_estimator,
    run_classical_inference,
    sweep_classical_head,
)
from modules.evaluate.src.schemas.artifacts import PreparedArtifacts
from modules.evaluate.src.utils.embed import (
    EmbeddingBackend,
    EmbeddingDataset,
    EmbeddingProbeHead,
    embedding_collate_fn,
    extract_embeddings,
)
from modules.evaluate.src.steps.predict import PredictConfig, PredictStep
from modules.evaluate.src.steps.tune import (
    HyperparameterSearchConfig,
    HyperparameterSearchSpaceConfig,
    TuneConfig,
    TuneStep,
)

from .conftest import ModelFamily, build_raw_dataset_dict, build_tiny_model

# ---------------------------------------------------------------------------
# build_estimator
# ---------------------------------------------------------------------------


class TestBuildEstimator:
    def test_classification_uses_classifier_class(self):
        estimator = build_estimator("random_forest", "sequence_classification", {})
        assert estimator.__class__.__name__ == "RandomForestClassifier"

    def test_regression_uses_regressor_class(self):
        estimator = build_estimator("random_forest", "sequence_regression", {})
        assert estimator.__class__.__name__ == "RandomForestRegressor"

    def test_hyperparameters_override_defaults(self):
        estimator = build_estimator("knn", "sequence_classification", {"n_neighbors": 3})
        assert estimator.named_steps["estimator"].n_neighbors == 3

    def test_sklearn_linear_uses_logistic_regression(self):
        estimator = build_estimator("sklearn_linear", "sequence_classification", {})
        assert estimator.named_steps["estimator"].__class__.__name__ == "LogisticRegression"

    def test_knn_and_sklearn_linear_are_wrapped_in_a_scaler_pipeline(self):
        # Distance/coefficient-based heads must be scale-normalized so probe
        # comparisons aren't confounded by raw embedding norm differences.
        for head_type in ("knn", "sklearn_linear"):
            estimator = build_estimator(head_type, "sequence_classification", {})
            assert estimator.__class__.__name__ == "Pipeline"
            assert estimator.named_steps["scaler"].__class__.__name__ == "StandardScaler"

    def test_random_forest_is_not_wrapped_in_a_scaler_pipeline(self):
        # Tree ensembles split on per-feature thresholds and are scale-invariant.
        estimator = build_estimator("random_forest", "sequence_classification", {})
        assert estimator.__class__.__name__ == "RandomForestClassifier"
        assert estimator.n_jobs == -1

    @pytest.mark.parametrize(
        ("task_type", "expected"),
        [
            ("sequence_classification", {"C": [0.1, 1.0]}),
            ("sequence_regression", {"alpha": [0.1, 1.0]}),
        ],
    )
    def test_sklearn_linear_search_space_keeps_only_task_parameters(
        self, task_type, expected
    ):
        search_space = {
            "C": [0.1, 1.0],
            "alpha": [0.1, 1.0],
        }

        assert (
            _filter_classical_search_space(
                "sklearn_linear", task_type, search_space
            )
            == expected
        )

    def test_xgboost_without_package_raises_informative_error(self, monkeypatch):
        import modules.evaluate.src.model.heads as heads_module

        def _raise_import_error():
            raise ImportError("no xgboost")

        monkeypatch.setattr(heads_module, "_xgboost_estimator_classes", _raise_import_error)
        with pytest.raises(ImportError, match="xgboost"):
            build_estimator("xgboost", "sequence_classification", {})


# ---------------------------------------------------------------------------
# Classical estimator behavior
# ---------------------------------------------------------------------------


class TestClassicalEstimator:
    def _toy_data(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, 4))
        y = (X[:, 0] > 0).astype(int)
        return X, y

    def test_fit_predict_round_trip(self):
        X, y = self._toy_data()
        estimator = build_estimator("sklearn_linear", "sequence_classification", {})
        estimator.fit(X, y)

        predictions = estimator.predict(X)
        assert predictions.shape == (40,)
        # A near-separable toy dataset should be fit near-perfectly.
        assert (predictions == y).mean() > 0.9

    def test_predict_proba_available_for_classifiers(self):
        X, y = self._toy_data()
        estimator = build_estimator("sklearn_linear", "sequence_classification", {})
        estimator.fit(X, y)
        probabilities = estimator.predict_proba(X)
        assert probabilities.shape == (40, 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)


# ---------------------------------------------------------------------------
# sweep_classical_head
# ---------------------------------------------------------------------------


class TestSweepClassicalHead:
    def test_picks_best_scoring_hyperparameters(self):
        rng = np.random.default_rng(1)
        train_X = rng.normal(size=(60, 2))
        train_y = (train_X[:, 0] > 0).astype(int)
        val_X = rng.normal(size=(20, 2))
        val_y = (val_X[:, 0] > 0).astype(int)

        best_head, best_hparams, best_score = sweep_classical_head(
            "knn",
            "sequence_classification",
            train_X,
            train_y,
            val_X,
            val_y,
            search_space={"n_neighbors": [1, 3, 5]},
            default_hyperparameters={},
            metric_name="accuracy",
            direction="maximize",
        )

        assert best_hparams["n_neighbors"] in (1, 3, 5)
        assert 0.0 <= best_score <= 1.0
        assert hasattr(best_head, "predict")
        # Train-only fit, matching the mlp/torch_linear paths.
        assert best_head.named_steps["estimator"].n_samples_fit_ == len(train_X)

    def test_merges_fixed_and_swept_hyperparameters(self):
        rng = np.random.default_rng(7)
        train_X = rng.normal(size=(40, 2))
        train_y = (train_X[:, 0] > 0).astype(int)
        val_X = rng.normal(size=(16, 2))
        val_y = (val_X[:, 0] > 0).astype(int)

        _, best_hparams, _ = sweep_classical_head(
            "sklearn_linear",
            "sequence_classification",
            train_X,
            train_y,
            val_X,
            val_y,
            search_space={"C": [0.1, 1.0]},
            default_hyperparameters={"max_iter": 25},
            metric_name="accuracy",
            direction="maximize",
        )

        assert best_hparams["max_iter"] == 25
        assert best_hparams["C"] in (0.1, 1.0)

    def test_probability_metric_uses_estimator_probabilities(self):
        rng = np.random.default_rng(8)
        train_X = rng.normal(size=(60, 2))
        train_y = (train_X[:, 0] > 0).astype(int)
        val_X = rng.normal(size=(20, 2))
        val_y = (val_X[:, 0] > 0).astype(int)

        _, _, score = sweep_classical_head(
            "sklearn_linear",
            "sequence_classification",
            train_X,
            train_y,
            val_X,
            val_y,
            search_space={"C": [0.1, 1.0]},
            default_hyperparameters={},
            metric_name="roc_auc",
            direction="maximize",
        )

        assert 0.0 <= score <= 1.0

    def test_no_search_space_fits_default_hyperparameters_once(self):
        rng = np.random.default_rng(2)
        train_X = rng.normal(size=(30, 2))
        train_y = (train_X[:, 0] > 0).astype(int)

        best_head, best_hparams, _ = sweep_classical_head(
            "knn",
            "sequence_classification",
            train_X,
            train_y,
            train_X,
            train_y,
            search_space={},
            default_hyperparameters={"n_neighbors": 7},
            metric_name="accuracy",
            direction="maximize",
        )

        assert best_hparams == {"n_neighbors": 7}
        assert best_head.named_steps["estimator"].n_neighbors == 7

    def test_supports_bohb_search_pattern(self):
        pytest.importorskip("optuna")
        rng = np.random.default_rng(3)
        train_X = rng.normal(size=(60, 2))
        train_y = (train_X[:, 0] > 0).astype(int)
        val_X = rng.normal(size=(20, 2))
        val_y = (val_X[:, 0] > 0).astype(int)

        best_head, best_hparams, best_score = sweep_classical_head(
            "knn",
            "sequence_classification",
            train_X,
            train_y,
            val_X,
            val_y,
            search_space={"n_neighbors": [1, 3, 5]},
            default_hyperparameters={},
            metric_name="accuracy",
            direction="maximize",
            search_pattern="bohb",
            n_trials=2,
            seed=42,
        )

        assert best_hparams["n_neighbors"] in (1, 3, 5)
        assert 0.0 <= best_score <= 1.0
        assert hasattr(best_head, "predict")


# ---------------------------------------------------------------------------
# Direct classical inference
# ---------------------------------------------------------------------------


def _embedding_dataloader(num_examples: int, hidden_size: int, num_labels: int, seed: int):
    rng = np.random.default_rng(seed)
    embeddings = torch.as_tensor(rng.normal(size=(num_examples, hidden_size)), dtype=torch.float16)
    labels = torch.as_tensor(rng.integers(0, num_labels, size=num_examples), dtype=torch.long)
    ids = list(range(num_examples))
    dataset = EmbeddingDataset(embeddings, ids, labels)
    return DataLoader(
        dataset,
        batch_size=8,
        collate_fn=lambda batch: embedding_collate_fn(batch, torch.long),
    )


class TestDirectClassicalInference:
    def test_classification_predictions_and_probabilities(self):
        dataloader = _embedding_dataloader(20, hidden_size=4, num_labels=2, seed=3)
        embeddings = dataloader.dataset.embeddings.float().numpy()
        labels = dataloader.dataset.labels.numpy()
        estimator = build_estimator("sklearn_linear", "sequence_classification", {})
        estimator.fit(embeddings, labels)

        ids, predictions, returned_labels, probabilities = run_classical_inference(
            estimator, dataloader, "sequence_classification", 2, True
        )

        assert ids == list(range(20))
        assert len(predictions) == len(returned_labels) == len(probabilities) == 20
        assert np.allclose(np.asarray(probabilities).sum(axis=1), 1.0)

    def test_regression_returns_scalar_predictions(self):
        rng = np.random.default_rng(4)
        embeddings = torch.as_tensor(rng.normal(size=(10, 4)), dtype=torch.float32)
        labels = embeddings[:, 0] * 2.0
        dataloader = DataLoader(
            EmbeddingDataset(embeddings, list(range(10)), labels),
            batch_size=5,
            collate_fn=lambda batch: embedding_collate_fn(batch, torch.float),
        )
        estimator = build_estimator("sklearn_linear", "sequence_regression", {})
        estimator.fit(embeddings.numpy(), labels.numpy())

        _, predictions, returned_labels, probabilities = run_classical_inference(
            estimator, dataloader, "sequence_regression", 1
        )

        assert len(predictions) == len(returned_labels) == 10
        assert probabilities is None


# ---------------------------------------------------------------------------
# TuneStep._run_classical_head integration
# ---------------------------------------------------------------------------


class TestTuneStepClassicalHead:
    def test_fits_sweeps_and_saves_a_classical_head(self, tmp_path):
        train_dataloader = _embedding_dataloader(40, hidden_size=4, num_labels=2, seed=10)
        val_dataloader = _embedding_dataloader(16, hidden_size=4, num_labels=2, seed=11)

        model = MagicMock(num_labels=2)
        prepared = PreparedArtifacts(
            model=model,
            tokenizer=MagicMock(),
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=None,
        )
        workspace = MagicMock(
            model_path=tmp_path / "model", task_type="sequence_classification"
        )

        step = TuneStep(
            TuneConfig(
                head=HeadConfig(head_type="knn", search_space={"n_neighbors": [1, 3]})
            )
        )
        result = step.run(workspace, prepared, seed=42)

        assert hasattr(result.model, "predict")
        assert (workspace.model_path / "head.joblib").exists()
        assert (workspace.model_path / "tune_config.json").exists()

    def test_requires_embedding_dataloader(self, tmp_path):
        token_batch_dataloader = [
            {"input_ids": torch.zeros(2, 3, dtype=torch.long), "labels": torch.zeros(2, dtype=torch.long)}
        ]
        model = MagicMock(num_labels=2)
        prepared = PreparedArtifacts(
            model=model,
            tokenizer=MagicMock(),
            train_dataloader=token_batch_dataloader,
            val_dataloader=None,
            test_dataloader=None,
        )
        workspace = MagicMock(
            model_path=tmp_path / "model", task_type="sequence_classification"
        )
        step = TuneStep(TuneConfig(head=HeadConfig(head_type="sklearn_linear")))

        with pytest.raises(ValueError, match="embedding_cache"):
            step.run(workspace, prepared, seed=42)

    def test_search_requires_validation_dataloader(self, tmp_path):
        prepared = PreparedArtifacts(
            model=MagicMock(num_labels=2),
            tokenizer=MagicMock(),
            train_dataloader=_embedding_dataloader(16, 4, 2, seed=13),
            val_dataloader=None,
            test_dataloader=None,
        )
        workspace = MagicMock(
            model_path=tmp_path / "model", task_type="sequence_classification"
        )
        step = TuneStep(
            TuneConfig(head=HeadConfig(head_type="knn", search_space={"n_neighbors": [1, 3]}))
        )

        with pytest.raises(ValueError, match="validation split"):
            step.run(workspace, prepared, seed=42)

    def test_fixed_estimator_allows_no_validation_dataloader(self, tmp_path):
        prepared = PreparedArtifacts(
            model=MagicMock(num_labels=2),
            tokenizer=MagicMock(),
            train_dataloader=_embedding_dataloader(16, 4, 2, seed=14),
            val_dataloader=None,
            test_dataloader=None,
        )
        workspace = MagicMock(
            model_path=tmp_path / "model", task_type="sequence_classification"
        )
        step = TuneStep(
            TuneConfig(
                head=HeadConfig(
                    head_type="sklearn_linear", hyperparameters={"C": 1.0}
                ),
                save_best_model=False,
            )
        )

        result = step.run(workspace, prepared, seed=42)

        assert hasattr(result.model, "predict")

    def test_torch_linear_trains_through_the_gradient_path(self, tmp_path):
        train_dataloader = Accelerator().prepare(
            _embedding_dataloader(16, hidden_size=4, num_labels=2, seed=12)
        )
        prepared = PreparedArtifacts(
            model=EmbeddingProbeHead(4, 2, "sequence_classification"),
            tokenizer=MagicMock(),
            train_dataloader=train_dataloader,
            val_dataloader=None,
            test_dataloader=None,
        )
        workspace = MagicMock(
            model_path=tmp_path / "model", task_type="sequence_classification"
        )
        step = TuneStep(
            TuneConfig(
                head=HeadConfig(head_type="torch_linear"),
                max_epochs=1,
                save_best_model=False,
            )
        )

        result = step.run(workspace, prepared, seed=42)

        assert isinstance(result.model, EmbeddingProbeHead)
        assert isinstance(result.model.classifier, torch.nn.Linear)

    def test_torch_linear_trains_over_cached_token_embeddings(self, tmp_path):
        """torch_linear also works for token_classification via the
        embedding cache: rows are per-token instead of per-example, but
        EmbeddingProbeHead/TuneStep's gradient path is agnostic to that."""
        train_dataloader = Accelerator().prepare(
            _embedding_dataloader(16, hidden_size=4, num_labels=2, seed=15)
        )
        prepared = PreparedArtifacts(
            model=EmbeddingProbeHead(4, 2, "token_classification"),
            tokenizer=MagicMock(),
            train_dataloader=train_dataloader,
            val_dataloader=None,
            test_dataloader=None,
        )
        workspace = MagicMock(
            model_path=tmp_path / "model", task_type="token_classification"
        )
        step = TuneStep(
            TuneConfig(
                head=HeadConfig(head_type="torch_linear"),
                max_epochs=1,
                save_best_model=False,
            )
        )

        result = step.run(workspace, prepared, seed=42)

        assert isinstance(result.model, EmbeddingProbeHead)
        assert isinstance(result.model.classifier, torch.nn.Linear)

    def test_torch_linear_trains_the_native_token_head_without_cache(
        self, tmp_path, model_family, caplog
    ):
        """token_classification doesn't require embedding_cache: its native
        head is already a plain nn.Linear, so torch_linear should be able to
        train that head directly with the trunk frozen, exactly like
        head_type='mlp' (without the cache; see
        test_torch_linear_trains_over_cached_token_embeddings for the
        cached-embedding alternative)."""
        tokenizer = model_family.tokenizer
        dataset = tokenize_dataset(
            build_raw_dataset_dict("token_classification", num_labels=2),
            tokenizer=tokenizer,
            task_type="token_classification",
            sequence_column="sequence",
            label_column="targets",
            max_length=64,
            num_proc=1,
            random_truncate=False,
        )
        collator = EvaluationCollator(
            tokenizer=tokenizer,
            task_type="token_classification",
            config=EvaluationCollatorConfig(label_column="targets"),
        )
        train_dl, _, _ = build_dataloaders(
            dataset=dataset,
            collator=collator,
            config=DataLoaderConfig(
                batch_size=2, num_workers=0, pin_memory=False, persistent_workers=False
            ),
        )
        model = build_tiny_model(model_family, "token_classification", 2)
        prepared = PreparedArtifacts(
            model=model,
            tokenizer=tokenizer,
            train_dataloader=Accelerator().prepare(train_dl),
            val_dataloader=None,
            test_dataloader=None,
        )
        workspace = MagicMock(
            model_path=tmp_path / "model", task_type="token_classification"
        )
        step = TuneStep(
            TuneConfig(
                head=HeadConfig(head_type="torch_linear"),
                max_epochs=1,
                save_best_model=False,
            )
        )
        trunk = model.base_model
        trunk_params_before = [p.detach().cpu().clone() for p in trunk.parameters()]

        result = step.run(workspace, prepared, seed=42)

        assert "Embedding cache is unavailable" in caplog.text
        assert result.model is model
        assert isinstance(result.model.classifier, torch.nn.Linear)
        assert not any(p.requires_grad for p in trunk.parameters())
        assert all(
            torch.equal(before, after.detach().cpu())
            for before, after in zip(trunk_params_before, trunk.parameters())
        )
        assert result.model.classifier.weight.requires_grad

    def test_bohb_search_broadcasts_hyperparameters_as_object(self, tmp_path):
        # Exercises _search_hyperparameters' default ("bohb") branch, whose
        # objective() broadcasts the sampled hparams dict via
        # broadcast_object_list rather than hand-packing a float32 tensor.
        train_dataloader = Accelerator().prepare(
            _embedding_dataloader(16, hidden_size=4, num_labels=2, seed=15)
        )
        val_dataloader = Accelerator().prepare(
            _embedding_dataloader(8, hidden_size=4, num_labels=2, seed=16)
        )
        prepared = PreparedArtifacts(
            model=EmbeddingProbeHead(4, 2, "sequence_classification"),
            tokenizer=MagicMock(),
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=None,
        )
        workspace = MagicMock(
            model_path=tmp_path / "model", task_type="sequence_classification"
        )
        step = TuneStep(
            TuneConfig(
                head=HeadConfig(head_type="torch_linear"),
                max_epochs=1,
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

        result = step.run(workspace, prepared, seed=42)

        assert isinstance(result.model, EmbeddingProbeHead)


# ---------------------------------------------------------------------------
# EvaluationConfig cross-step validation for classical heads
# ---------------------------------------------------------------------------


class TestClassicalHeadCrossStepValidation:
    def _base_kwargs(self, **tune_overrides):
        return dict(
            seed=1,
            workspace=dict(
                dataset_name="d",
                model_name="m",
                task_type="sequence_classification",
                target_type="categorical",
                label_format="scalar",
                num_labels=2,
            ),
            steps=dict(
                mode="tune_probe",
                prepare=dict(embedding_cache=dict(enabled=True)),
                tune=dict(head=dict(head_type="sklearn_linear"), **tune_overrides),
                # classical heads can't support a randomly re-initialized
                # head, so the default controls=["untrained"] must be
                # overridden to a control that only needs the trunk/inputs.
                predict=dict(controls=["random_trunk"]),
            ),
        )

    def test_requires_embedding_cache_enabled(self):
        kwargs = self._base_kwargs()
        kwargs["steps"]["prepare"] = dict(embedding_cache=dict(enabled=False))
        with pytest.raises(Exception, match="embedding_cache"):
            EvaluationConfig(**kwargs)

    def test_rejects_classical_head_in_finetune_mode(self):
        kwargs = self._base_kwargs()
        kwargs["steps"]["mode"] = "finetune"
        with pytest.raises(Exception, match="mode"):
            EvaluationConfig(**kwargs)

    def test_rejects_controls_needing_a_randomly_reinitializable_head(self):
        kwargs = self._base_kwargs()
        kwargs["steps"]["predict"] = dict(controls=["untrained", "random_head"])
        with pytest.raises(Exception, match="untrained"):
            EvaluationConfig(**kwargs)

    def test_allows_valid_combination(self):
        config = EvaluationConfig(**self._base_kwargs())
        assert config.steps.tune.head.head_type == "sklearn_linear"

    def test_torch_linear_without_cache_is_valid_before_model_loading(self):
        kwargs = self._base_kwargs()
        kwargs["steps"]["prepare"] = dict(embedding_cache=dict(enabled=False))
        kwargs["steps"]["tune"]["head"] = dict(head_type="torch_linear")
        config = EvaluationConfig(**kwargs)
        assert config.steps.tune.head.head_type == "torch_linear"


# ---------------------------------------------------------------------------
# PredictStep support for embedding-cache runs (any head type)
# ---------------------------------------------------------------------------


def _build_embedding_prepared_artifacts(
    model_family: ModelFamily,
    task_type: str,
    head_type: str = "mlp",
    num_labels: int = 2,
    batch_size: int = 2,
    layers: list[int] | None = None,
) -> PreparedArtifacts:
    """Builds a PreparedArtifacts equivalent to what PrepareStep produces
    with embedding_cache.enabled=True, but directly from tiny offline
    model_family fixtures (mirrors tests/evaluate/test_predict.py's
    _build_prepared_artifacts, extended with the embedding-mode fields)."""
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
            batch_size=batch_size, num_workers=0, pin_memory=False, persistent_workers=False
        ),
    )
    effective_num_labels = 1 if task_type == "sequence_regression" else num_labels
    trunk = build_tiny_model(model_family, task_type, effective_num_labels)
    accelerator = Accelerator()
    pooling, dtype = "mean", "float32"
    layers = layers or [-1]
    hidden_size = trunk.config.hidden_size * len(layers)
    label_dtype = torch.float if task_type == "sequence_regression" else torch.long

    def _embed(dataloader):
        embeddings, ids, labels, _ = extract_embeddings(
            trunk, dataloader, accelerator, pooling, dtype, layers=layers
        )
        embedding_dataset = EmbeddingDataset(embeddings, ids, labels.to(label_dtype))
        return embeddings, labels, DataLoader(
            embedding_dataset,
            batch_size=batch_size,
            collate_fn=lambda batch: embedding_collate_fn(batch, label_dtype),
        )

    train_embeddings, train_labels, train_emb_dl = _embed(train_dl)
    _, _, val_emb_dl = _embed(val_dl)
    _, _, test_emb_dl = _embed(test_dl)

    if head_type == "mlp":
        head = EmbeddingProbeHead(hidden_size, effective_num_labels, task_type)
    else:
        estimator = build_estimator(head_type, task_type, {})
        estimator.fit(train_embeddings.numpy(), train_labels.numpy())
        head = estimator

    return PreparedArtifacts(
        model=head,
        tokenizer=tokenizer,
        train_dataloader=train_emb_dl,
        val_dataloader=val_emb_dl,
        test_dataloader=test_emb_dl,
        embedding_backend=EmbeddingBackend(
            trunk=trunk,
            token_test_dataloader=test_dl,
            pooling=pooling,
            layers=layers,
            dtype=dtype,
            hidden_size=hidden_size,
        ),
    )


def _make_predict_workspace(tmp_path, task_type, num_labels=2):
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


class TestPredictStepWithEmbeddingCache:
    def test_store_sequences_joins_decoded_sequences_by_id(self, tmp_path, model_family):
        prepared = _build_embedding_prepared_artifacts(model_family, "sequence_classification")
        workspace = _make_predict_workspace(tmp_path, "sequence_classification")

        result = (
            PredictStep(PredictConfig(store_sequences=True, controls=[])).run(workspace, prepared)
        )

        assert "sequence" in result.predictions.columns
        assert result.predictions["sequence"].null_count() == 0
