"""Tests for top-level evaluate config validation behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from modules.core.utils.config_loader import load_and_parse
from modules.evaluate.src.config import EvaluationConfig, LayerwiseConfig
from modules.evaluate.src.steps.prepare import PrepareConfig
from modules.evaluate.src.utils.embed import EmbeddingCacheConfig


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


class TestExternalTestSets:
    @staticmethod
    def _payload(**extra):
        payload = _base_config_dict()
        payload["external_test_sets"] = [
            {"name": "sethard", "dataset_repo_id": "flair-bio/sethard"}
        ]
        payload.update(extra)
        return payload

    def test_defaults_to_empty_and_split_to_test(self):
        assert (
            EvaluationConfig.model_validate(_base_config_dict()).external_test_sets
            == []
        )
        config = EvaluationConfig.model_validate(self._payload())
        assert config.external_test_sets[0].split == "test"

    @pytest.mark.parametrize(
        "extra",
        [
            {"external_test_sets": [{"name": "a", "dataset_repo_id": "x/a"}] * 2},
            {"external_test_sets": [{"name": "a/b", "dataset_repo_id": "x/a"}]},
        ],
    )
    def test_invalid_combinations_rejected(self, extra):
        with pytest.raises(ValueError):
            EvaluationConfig.model_validate(self._payload(**extra))


class TestEvaluationConfig:
    def test_cross_validation_defaults_to_disabled_at_top_level(self):
        config = EvaluationConfig.model_validate(_base_config_dict())
        assert not config.cross_validation.enabled
        assert config.cross_validation.pool_roles == ["train", "validation"]
        assert "cross_validation" in config.model_dump()
        assert "cross_validation" not in config.steps.model_dump()

    def test_cross_validation_rejects_former_steps_location(self):
        payload = _base_config_dict()
        payload["steps"] = {"cross_validation": {"enabled": True}}
        with pytest.raises(ValueError, match=r"steps\.cross_validation"):
            EvaluationConfig.model_validate(payload)

    def test_cross_validation_top_level_yaml_and_overrides(self):
        root = Path(__file__).resolve().parents[2] / "modules/evaluate/configs"
        config = load_and_parse(
            path=[
                root / "config.yaml",
                root / "tasks/sequence_classification.yaml",
                root / "datasets/localization_prediction.yaml",
            ],
            model_cls=EvaluationConfig,
            overrides=[
                "cross_validation.enabled=true",
                "cross_validation.n_splits=3",
                "steps.score.bootstrap_enabled=false",
            ],
        )
        assert config.cross_validation.enabled
        assert config.cross_validation.n_splits == 3
        assert config.cross_validation.pool_roles == ["train", "validation"]
        assert "cross_validation" not in config.steps.model_dump()

    def test_feature_extraction_defaults_to_float32_dtype(self):
        assert LayerwiseConfig().autocast_dtype == "float32"
        assert EmbeddingCacheConfig().autocast_dtype == "float32"
        assert EmbeddingCacheConfig().dtype == "float32"

    @pytest.mark.parametrize(
        "head_type", ["mlp", "sklearn_linear", "knn", "random_forest", "xgboost"]
    )
    def test_layerwise_cache_rejects_non_layerwise_heads(self, head_type):
        payload = _base_config_dict()
        payload["steps"] = {
            "layerwise": {"enabled": True},
            "prepare": {"embedding_cache": {"enabled": True}},
            "tune": {"head": {"head_type": head_type}},
        }
        with pytest.raises(ValueError, match="head_type='torch_linear'"):
            EvaluationConfig.model_validate(payload)

    def test_layerwise_accepts_sequence_multilabel(self):
        payload = _base_config_dict()
        payload["workspace"].update(
            task_type="sequence_multilabel_classification", label_format="multilabel"
        )
        payload["steps"] = {
            "layerwise": {"enabled": True},
            "tune": {"head": {"head_type": "torch_linear"}},
        }
        assert EvaluationConfig.model_validate(payload).steps.layerwise.enabled

    def test_layerwise_variant_is_distinct(self):
        payload = _base_config_dict()
        payload["steps"] = {
            "layerwise": {"enabled": True, "layers": "all"},
            "tune": {"head": {"head_type": "torch_linear"}},
        }
        assert (
            EvaluationConfig.model_validate(payload).variant
            == "layerwise-torch_linear-frozen_trunk"
        )

    def test_token_budget_requires_packing(self):
        payload = _base_config_dict()
        payload["steps"] = {"prepare": {"max_tokens_per_batch": 128}}
        with pytest.raises(ValueError, match="max_tokens_per_batch requires"):
            EvaluationConfig.model_validate(payload)

    @pytest.mark.parametrize(
        "task_type", ["contact_prediction", "categorical_jacobian", "calibration"]
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

    def test_accumulation_steps_uses_configured_value(self):
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
            ("calibration", "categorical", "consensus_profile", None),
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
        if task_type in {"categorical_jacobian", "pseudo_perplexity", "calibration"}:
            payload["steps"] = {"mode": "evaluate_as_is"}

        EvaluationConfig.model_validate(payload)

    @pytest.mark.parametrize(
        ("task_type", "target_type", "label_format"),
        [
            ("sequence_classification", "continuous", "scalar"),
            ("sequence_regression", "categorical", "scalar"),
            ("token_classification", "categorical", "scalar"),
            ("contact_prediction", "categorical", "per_token"),
            ("calibration", "categorical", "scalar"),
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
        _CONFIG_ROOT / "datasets" / "localization_prediction.yaml",
    ]

    @pytest.mark.parametrize(
        ("task_name", "dataset_name"),
        [
            ("sequence_classification", "localization_prediction"),
            ("sequence_regression", "stability_prediction"),
            ("sequence_multilabel_classification", "GO_mf"),
            ("token_classification", "ssp_q3"),
        ],
    )
    def test_frozen_probe_task_defaults_enable_cache(self, task_name, dataset_name):
        paths = [
            self._CONFIG_ROOT / "config.yaml",
            self._CONFIG_ROOT / "tasks" / f"{task_name}.yaml",
            self._CONFIG_ROOT / "datasets" / f"{dataset_name}.yaml",
        ]
        config = load_and_parse(paths, EvaluationConfig)
        assert config.steps.mode == "tune_probe"
        assert config.steps.prepare.embedding_cache.enabled
        assert config.steps.prepare.embedding_cache.dtype == "float32"
        assert config.steps.prepare.embedding_cache.autocast_dtype == "float32"

        finetune = load_and_parse(
            [*paths, self._CONFIG_ROOT / "tuning" / "mlp_finetuning.yaml"],
            EvaluationConfig,
        )
        assert finetune.steps.mode == "finetune"
        assert not finetune.steps.prepare.embedding_cache.enabled

        smoke = load_and_parse(
            [*paths, self._CONFIG_ROOT / "smoke_test.yaml"], EvaluationConfig
        )
        assert smoke.steps.mode == "evaluate_as_is"
        assert not smoke.steps.prepare.embedding_cache.enabled

    def test_pseudo_perplexity_defaults_to_deterministic_test_data(self):
        config = load_and_parse(
            [
                self._CONFIG_ROOT / "config.yaml",
                self._CONFIG_ROOT / "tasks" / "pseudo_perplexity.yaml",
                self._CONFIG_ROOT / "datasets" / "uniprot_protparams.yaml",
            ],
            EvaluationConfig,
        )
        assert config.steps.mode == "evaluate_as_is"
        assert not config.steps.prepare.embedding_cache.enabled
        split = config.steps.prepare.split
        assert (split.train, split.validation, split.test) == ("test",) * 3
        assert not any(config.steps.prepare.random_truncate_by_split.values())

    @pytest.mark.parametrize(
        ("task_name", "dataset_name"),
        [
            ("sequence_classification", "localization_prediction"),
            ("sequence_regression", "stability_prediction"),
            ("token_classification", "ssp_q3"),
            ("contact_prediction", "contact_prediction_binary"),
            ("categorical_jacobian", "contact_prediction_binary"),
            ("pseudo_perplexity", "localization_prediction"),
            ("pseudo_perplexity", "ssp_q3"),
            ("pseudo_perplexity", "contact_prediction_binary"),
            ("calibration", "human_calibration"),
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

    def test_calibration_requires_evaluate_as_is(self):
        payload = _base_config_dict()
        payload["workspace"].update(
            task_type="calibration",
            target_type="categorical",
            label_format="consensus_profile",
            num_labels=None,
        )
        payload["steps"] = {"mode": "finetune"}
        with pytest.raises(ValueError, match="requires steps.mode='evaluate_as_is'"):
            EvaluationConfig.model_validate(payload)

    def test_shipped_incompatible_combination_is_rejected(self):
        with pytest.raises(ValueError, match="incompatible with dataset"):
            load_and_parse(
                path=[
                    self._CONFIG_ROOT / "config.yaml",
                    self._CONFIG_ROOT / "models" / "esm2_35m.yaml",
                    self._CONFIG_ROOT / "tasks" / "sequence_regression.yaml",
                    self._CONFIG_ROOT / "datasets" / "ssp_q3.yaml",
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

    def test_go_multilabel_dataset_overlay_parses(self):
        config = load_and_parse(
            path=[
                self._CONFIG_ROOT / "config.yaml",
                self._CONFIG_ROOT / "models" / "esm2_35m.yaml",
                self._CONFIG_ROOT / "tasks" / "sequence_multilabel_classification.yaml",
                self._CONFIG_ROOT / "datasets" / "GO_mf.yaml",
            ],
            model_cls=EvaluationConfig,
        )

        assert config.workspace.task_type == "sequence_multilabel_classification"
        assert "majority_class" not in config.steps.predict.controls

    @pytest.mark.parametrize(
        "dataset_name",
        [
            "GO_bp",
            "GO_cc",
            "GO_mf",
            "interpro_active_site",
            "interpro_binding_site",
            "interpro_conserved_site",
            "interpro_domain",
            "interpro_family",
            "interpro_homologous_superfamily",
            "interpro_repeat",
        ],
    )
    def test_multilabel_overlay_declares_target_contract(self, dataset_name):
        config = load_and_parse(
            path=[
                self._CONFIG_ROOT / "config.yaml",
                self._CONFIG_ROOT / "models" / "esm2_35m.yaml",
                self._CONFIG_ROOT / "tasks" / "sequence_multilabel_classification.yaml",
                self._CONFIG_ROOT / "datasets" / f"{dataset_name}.yaml",
            ],
            model_cls=EvaluationConfig,
        )

        assert config.workspace.target_type == "categorical"
        assert config.workspace.label_format == "multilabel"

    @pytest.mark.parametrize(
        ("dataset_name", "task_type", "target_type", "label_format"),
        [
            ("ssp_q8", "token_classification", "categorical", "per_token"),
            (
                "fold_prediction",
                "sequence_classification",
                "categorical",
                "scalar",
            ),
            (
                "temperature_stability",
                "sequence_classification",
                "categorical",
                "scalar",
            ),
            (
                "stability_prediction",
                "sequence_regression",
                "continuous",
                "scalar",
            ),
            (
                "solubility_prediction",
                "sequence_classification",
                "categorical",
                "scalar",
            ),
            (
                "contact_prediction_binary",
                "contact_prediction",
                "categorical",
                "contact_pairs",
            ),
        ],
    )
    def test_biomap_overlay_matches_hosted_target_contract(
        self, dataset_name, task_type, target_type, label_format
    ):
        config = load_and_parse(
            path=[
                self._CONFIG_ROOT / "config.yaml",
                self._CONFIG_ROOT / "models" / "esm2_35m.yaml",
                self._CONFIG_ROOT / "tasks" / f"{task_type}.yaml",
                self._CONFIG_ROOT / "datasets" / f"{dataset_name}.yaml",
            ],
            model_cls=EvaluationConfig,
        )

        assert config.workspace.target_type == target_type
        assert config.workspace.label_format == label_format
