"""Tests for top-level evaluate config validation behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from modules.core.utils.config_loader import load_and_parse
from modules.evaluate.src.config import EvaluationConfig
from modules.evaluate.src.steps.prepare import PrepareConfig


def _base_config_dict() -> dict:
    return {
        "seed": 710019,
        "workspace": {
            "dataset_name": "toy_dataset",
            "model_name": "toy_model",
            "task_type": "sequence_classification",
            "target_type": "categorical",
            "label_format": "scalar",
            "num_labels": 2,
        },
    }


class TestEvaluationConfig:
    def test_token_budget_requires_packing(self):
        payload = _base_config_dict()
        payload["steps"] = {"prepare": {"max_tokens_per_batch": 128}}
        with pytest.raises(ValueError, match="max_tokens_per_batch requires"):
            EvaluationConfig.model_validate(payload)

    @pytest.mark.parametrize(
        "task_type", ["contact_prediction", "categorical_jacobian"]
    )
    def test_pairwise_tasks_reject_packing(self, task_type):
        payload = _base_config_dict()
        payload["workspace"]["task_type"] = task_type
        payload["steps"] = {"mode": "evaluate_as_is", "prepare": {"packed": True}}
        with pytest.raises(ValueError, match="Packed evaluation does not support"):
            EvaluationConfig.model_validate(payload)

    def test_packing_with_embedding_cache_is_valid(self):
        payload = _base_config_dict()
        payload["steps"] = {
            "prepare": {"packed": True, "embedding_cache": {"enabled": True}}
        }
        assert EvaluationConfig.model_validate(payload).steps.prepare.packed

    @pytest.mark.parametrize(
        ("mode", "expected_mode", "expected_variant"),
        [
            ("tune_probe", "tune_probe", "mlp-frozen_trunk"),
            ("finetune", "finetune", "mlp-tuned_trunk"),
            ("evaluate_as_is", "evaluate_as_is", "pretrained"),
        ],
    )
    def test_explicit_mode_sets_variant(self, mode, expected_mode, expected_variant):
        payload = _base_config_dict()
        payload["steps"] = {"mode": mode}

        config = EvaluationConfig.model_validate(payload)

        assert config.steps.mode == expected_mode
        assert config.variant == expected_variant

    def test_non_default_split_gets_its_own_variant(self):
        payload = _base_config_dict()
        payload["steps"] = {"prepare": {"split": {"name": "stratified"}}}

        assert (
            EvaluationConfig.model_validate(payload).variant
            == "mlp-frozen_trunk-stratified"
        )

    def test_seed_defaults_when_omitted(self):
        payload = _base_config_dict()
        payload.pop("seed")

        cfg = EvaluationConfig.model_validate(payload)

        assert cfg.seed == 710019

    def test_accumulation_steps_matches_apperitif_option(self):
        payload = _base_config_dict()
        payload["steps"] = {"tune": {"accumulation_steps": 2}}

        cfg = EvaluationConfig.model_validate(payload)

        assert cfg.steps.tune.accumulation_steps == 2

    def test_roc_auc_auto_enables_probability_collection(self):
        payload = _base_config_dict()
        payload["steps"] = {
            "predict": {"collect_probabilities": False},
            "score": {"metrics": ["accuracy", "roc_auc"]},
        }

        cfg = EvaluationConfig.model_validate(payload)
        assert cfg.steps.predict.collect_probabilities is True

    def test_default_metrics_for_sequence_classification_enable_probabilities(self):
        payload = _base_config_dict()
        payload["steps"] = {
            "predict": {"collect_probabilities": False},
            "score": {"metrics": None},
        }

        cfg = EvaluationConfig.model_validate(payload)
        assert cfg.steps.predict.collect_probabilities is True

    def test_embedding_cache_layers_default_to_none(self):
        payload = _base_config_dict()
        config = EvaluationConfig.model_validate(payload)

        assert config.steps.prepare.embedding_cache.layers is None

    def test_explicit_maximize_allowed_for_higher_is_better_metric(self):
        payload = _base_config_dict()
        payload["steps"] = {
            "tune": {
                "hyperparameter_search": {
                    "enabled": True,
                    "direction": "maximize",
                    "metric": "f1",
                }
            }
        }

        cfg = EvaluationConfig.model_validate(payload)
        assert cfg.steps.tune.hyperparameter_search.direction == "maximize"

    @pytest.mark.parametrize(
        ("task_type", "target_type", "label_format", "num_labels"),
        [
            ("sequence_classification", "categorical", "scalar", 2),
            ("sequence_regression", "continuous", "scalar", None),
            ("token_classification", "categorical", "per_token", 3),
            ("contact_prediction", "categorical", "contact_pairs", None),
            ("categorical_jacobian", "categorical", "contact_pairs", None),
            ("pseudo_perplexity", "categorical", "scalar", 2),
        ],
    )
    def test_accepts_compatible_dataset_contract(
        self, task_type, target_type, label_format, num_labels
    ):
        payload = _base_config_dict()
        payload["workspace"].update(
            task_type=task_type,
            target_type=target_type,
            label_format=label_format,
            num_labels=num_labels,
        )
        if task_type in {"categorical_jacobian", "pseudo_perplexity"}:
            payload["steps"] = {"mode": "evaluate_as_is"}

        EvaluationConfig.model_validate(payload)

    @pytest.mark.parametrize(
        ("task_type", "target_type", "label_format"),
        [
            ("sequence_classification", "continuous", "scalar"),
            ("sequence_regression", "categorical", "scalar"),
            ("token_classification", "categorical", "scalar"),
            ("contact_prediction", "categorical", "per_token"),
        ],
    )
    def test_rejects_incompatible_dataset_contract(
        self, task_type, target_type, label_format
    ):
        payload = _base_config_dict()
        payload["workspace"].update(
            task_type=task_type,
            target_type=target_type,
            label_format=label_format,
            num_labels=2 if target_type == "categorical" else None,
        )

        with pytest.raises(ValueError, match="incompatible with dataset"):
            EvaluationConfig.model_validate(payload)


class TestPrepareConfig:
    def test_split_truncation_fields_can_be_set_explicitly(self):
        cfg = PrepareConfig(
            random_truncate_by_split={
                "train": True,
                "validation": False,
                "test": False,
            }
        )
        assert cfg.random_truncate_by_split == {
            "train": True,
            "validation": False,
            "test": False,
        }


class TestShippedConfigOverlays:
    _CONFIG_ROOT = Path(__file__).parents[2] / "modules" / "evaluate" / "configs"
    _BASE_PATHS = [
        _CONFIG_ROOT / "config.yaml",
        _CONFIG_ROOT / "models" / "esm2_35m.yaml",
        _CONFIG_ROOT / "tasks" / "sequence_classification.yaml",
        _CONFIG_ROOT / "datasets" / "metal_ion_binding.yaml",
    ]

    @pytest.mark.parametrize(
        ("task_name", "dataset_name"),
        [
            ("sequence_classification", "metal_ion_binding"),
            ("sequence_classification", "localization_prediction"),
            ("sequence_regression", "optimal_temperature"),
            ("token_classification", "ssp_q3"),
            ("contact_prediction", "contact_prediction_binary"),
            ("categorical_jacobian", "contact_prediction_binary"),
            ("pseudo_perplexity", "metal_ion_binding"),
            ("pseudo_perplexity", "localization_prediction"),
            ("pseudo_perplexity", "optimal_temperature"),
            ("pseudo_perplexity", "ssp_q3"),
            ("pseudo_perplexity", "contact_prediction_binary"),
        ],
    )
    def test_shipped_task_dataset_combinations_parse(self, task_name, dataset_name):
        config = load_and_parse(
            path=[
                self._CONFIG_ROOT / "config.yaml",
                self._CONFIG_ROOT / "models" / "esm2_35m.yaml",
                self._CONFIG_ROOT / "tasks" / f"{task_name}.yaml",
                self._CONFIG_ROOT / "datasets" / f"{dataset_name}.yaml",
            ],
            model_cls=EvaluationConfig,
        )

        assert config.workspace.task_type == task_name
        assert config.workspace.dataset_name == dataset_name

    def test_shipped_incompatible_combination_is_rejected(self):
        with pytest.raises(ValueError, match="incompatible with dataset"):
            load_and_parse(
                path=[
                    self._CONFIG_ROOT / "config.yaml",
                    self._CONFIG_ROOT / "models" / "esm2_35m.yaml",
                    self._CONFIG_ROOT / "tasks" / "sequence_regression.yaml",
                    self._CONFIG_ROOT / "datasets" / "metal_ion_binding.yaml",
                ],
                model_cls=EvaluationConfig,
            )

    def test_smoke_test_overlay_parses_with_default_embedding_layers(self):
        config = load_and_parse(
            path=[*self._BASE_PATHS, self._CONFIG_ROOT / "smoke_test.yaml"],
            model_cls=EvaluationConfig,
        )

        assert config.steps.prepare.embedding_cache.enabled is False
        assert config.steps.prepare.embedding_cache.layers is None
        assert "majority_class" not in config.steps.predict.controls

    def test_classical_probe_overlay_parses(self, tmp_path):
        controls_override = tmp_path / "controls.yaml"
        controls_override.write_text(
            "steps:\n  predict:\n    controls: [random_trunk]\n"
        )
        config = load_and_parse(
            path=[
                *self._BASE_PATHS,
                self._CONFIG_ROOT / "tuning" / "sklearn_linear.yaml",
                controls_override,
            ],
            model_cls=EvaluationConfig,
        )

        assert config.steps.prepare.embedding_cache.enabled is True
        assert config.steps.tune.head.head_type == "sklearn_linear"

    @pytest.mark.parametrize(
        ("probing_config", "expected_head_type"),
        [
            ("knn.yaml", "knn"),
            ("sklearn_linear.yaml", "sklearn_linear"),
            ("random_forest.yaml", "random_forest"),
            ("xgboost.yaml", "xgboost"),
            ("mlp_finetuning.yaml", "mlp"),
            ("torch_linear.yaml", "torch_linear"),
            ("mlp.yaml", "mlp"),
        ],
    )
    def test_probing_overlays_parse(self, tmp_path, probing_config, expected_head_type):
        controls_override = tmp_path / "controls.yaml"
        controls_override.write_text(
            "steps:\n  predict:\n    controls: [random_trunk]\n"
        )
        config = load_and_parse(
            path=[
                *self._BASE_PATHS,
                self._CONFIG_ROOT / "tuning" / probing_config,
                controls_override,
            ],
            model_cls=EvaluationConfig,
        )

        assert config.steps.tune.head.head_type == expected_head_type

    @pytest.mark.parametrize(
        "classical_probing_config",
        ["knn.yaml", "sklearn_linear.yaml", "random_forest.yaml", "xgboost.yaml"],
    )
    def test_smoke_test_overlay_combines_with_classical_probe_overlays(
        self, tmp_path, classical_probing_config
    ):
        controls_override = tmp_path / "controls.yaml"
        controls_override.write_text(
            "steps:\n  predict:\n    controls: [random_trunk]\n"
        )
        config = load_and_parse(
            path=[
                *self._BASE_PATHS,
                self._CONFIG_ROOT / "smoke_test.yaml",
                self._CONFIG_ROOT / "tuning" / classical_probing_config,
                controls_override,
            ],
            model_cls=EvaluationConfig,
        )

        assert "untrained" not in config.steps.predict.controls
