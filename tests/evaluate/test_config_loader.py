"""Task overlays are explicit; dataset overlays describe data only."""

from pathlib import Path

import pytest

from modules.core.utils.config_loader import load_and_parse
from modules.evaluate.src.config import EvaluationConfig
from modules.evaluate.src.steps.tune import HyperparameterSearchConfig
from modules.evaluate.src.utils.config_loader import load_evaluation_config

CONFIGS = Path(__file__).resolve().parents[2] / "modules/evaluate/configs"

# Representative dataset -> task pairings (what a suite supplies explicitly).
TASK_FOR_DATASET = {
    "fold_prediction": "sequence_classification",
    "localization_prediction": "sequence_classification",
    "GO_bp": "sequence_multilabel_classification",
    "stability_prediction": "sequence_regression",
    "uniprot_protparams": "sequence_regression",
    "ssp_q3": "token_classification",
    "uniprot_secondary_structure": "token_classification",
    "contact_prediction_casp": "contact_prediction",
    "human_calibration": "calibration",
    "uniprot_ur100p_test": "pseudo_perplexity",
}


def overlays(dataset: str, *extra: Path) -> list[Path]:
    """Base config, the explicit task overlay, the dataset, then extra overlays."""
    task = TASK_FOR_DATASET[dataset]
    return [
        CONFIGS / "config.yaml",
        CONFIGS / "tasks" / f"{task}.yaml",
        CONFIGS / "datasets" / f"{dataset}.yaml",
        *extra,
    ]


@pytest.mark.parametrize("predictor", [None, "torch_linear", "mlp_finetuning"])
def test_neural_overlays_select_continuous_log_weight_decay(predictor):
    extra = [CONFIGS / "predictors" / f"{predictor}.yaml"] if predictor else []
    config = load_evaluation_config(overlays("fold_prediction", *extra))
    search = config.steps.tune.hyperparameter_search
    assert search.enabled
    assert search.search_pattern == "tpe"
    assert search.engine_pattern == "bohb"
    assert search.search_space.weight_decay_sampling == "log"
    assert min(search.search_space.weight_decay) == 1e-6
    assert max(search.search_space.weight_decay) == 1e-1


def test_token_classification_uses_production_mlp_distributions():
    config = load_evaluation_config(overlays("ssp_q3"))
    search = config.steps.tune.hyperparameter_search
    assert config.workspace.task_type == "token_classification"
    assert search.n_trials == 100
    space = search.search_space
    assert space.batch_size == [8, 16, 32, 64, 128, 256]
    assert space.max_epochs == [1, 2, 5, 10, 20, 40, 100]
    assert space.hidden_size == [128, 256, 512]
    assert space.hidden_depth == [1, 2]
    assert space.learning_rate == [1e-5, 1e-3]


def test_legacy_bohb_search_pattern_is_accepted_as_tpe():
    legacy = HyperparameterSearchConfig(enabled=True, search_pattern="bohb")
    assert legacy.search_pattern == "tpe"
    assert legacy.engine_pattern == "bohb"
    current = HyperparameterSearchConfig(enabled=True, search_pattern="tpe")
    assert current.engine_pattern == "bohb"
    assert HyperparameterSearchConfig(search_pattern="random").engine_pattern == "random"


@pytest.mark.parametrize(
    "predictor",
    [
        "torch_linear",
        "sklearn_linear",
        "knn",
        "random_forest",
        "xgboost",
        "mlp_finetuning",
    ],
)
def test_predictor_presets_compose_existing_steps_and_allow_overrides(predictor):
    paths = overlays("fold_prediction", CONFIGS / "predictors" / f"{predictor}.yaml")
    config = load_evaluation_config(paths)
    assert config.steps.tune.head.head_type == (
        "mlp" if predictor == "mlp_finetuning" else predictor
    )
    assert config.steps.mode == (
        "finetune" if predictor == "mlp_finetuning" else "tune_probe"
    )
    customized = load_evaluation_config(paths, overrides=["steps.tune.max_epochs=25"])
    assert customized.steps.tune.max_epochs == 25


@pytest.mark.parametrize(
    "head,error",
    [
        ("mlp", "supported contact predictor preset"),
        ("torch_linear", "supported contact predictor preset"),
        ("sklearn_linear", "supported contact predictor preset"),
    ],
)
def test_contact_rejects_unsupported_heads(head, error):
    with pytest.raises(ValueError, match=error):
        load_evaluation_config(
            overlays("contact_prediction_casp"),
            overrides=[f"steps.tune.head.head_type={head}"],
        )


@pytest.mark.parametrize(
    "predictor", ["contact_embedding", "contact_attention", "contact_jacobian"]
)
def test_contact_predictors_cannot_be_used_for_other_tasks(predictor):
    with pytest.raises(ValueError, match="requires task_type='contact_prediction'"):
        load_evaluation_config(
            overlays("fold_prediction", CONFIGS / "predictors" / f"{predictor}.yaml")
        )


def test_contact_predictor_overlay_keeps_task_default_and_cli_precedence():
    config = load_evaluation_config(
        overlays("contact_prediction_casp", CONFIGS / "predictors/contact_embedding.yaml"),
        overrides=["steps.tune.max_epochs=25"],
    )
    assert config.workspace.task_type == "contact_prediction"
    assert config.steps.tune.head.head_type == "bilinear512"
    assert config.steps.tune.max_epochs == 25


def test_contact_attention_preset_selects_fixed_predictor():
    config = load_evaluation_config(
        overlays("contact_prediction_casp", CONFIGS / "predictors/contact_attention.yaml")
    )
    assert config.workspace.task_type == "contact_prediction"
    assert config.steps.tune.head.head_type == "attention_l1"
    assert not config.steps.prepare.embedding_cache.enabled
    assert not config.steps.tune.hyperparameter_search.enabled


def test_contact_jacobian_preset_selects_zero_shot_predictor():
    config = load_evaluation_config(
        overlays("contact_prediction_casp", CONFIGS / "predictors/contact_jacobian.yaml")
    )
    assert config.workspace.task_type == "contact_prediction"
    assert config.steps.tune.head.head_type == "categorical_jacobian"
    assert config.steps.mode == "evaluate_as_is"
    assert not config.steps.prepare.embedding_cache.enabled
    assert not config.steps.tune.hyperparameter_search.enabled
    assert config.steps.predict.controls == []


def test_classification_protocol_needs_model_task_and_dataset_only():
    config = load_evaluation_config(
        overlays("localization_prediction", CONFIGS / "models/amplify_120m.yaml")
    )
    assert config.steps.mode == "tune_probe"
    assert config.steps.prepare.embedding_cache.enabled
    assert config.steps.tune.head.head_type == "mlp"
    assert config.steps.tune.refit_on_train_validation
    assert config.steps.tune.standardize_features
    assert config.steps.tune.early_stopping_patience is None
    assert config.steps.tune.lr_scheduler_type == "cosine"
    search = config.steps.tune.hyperparameter_search
    assert search.enabled
    assert search.n_trials == 100
    assert search.search_space.hidden_size == [128, 256, 512]
    assert search.search_space.hidden_depth == [1, 2]
    assert search.search_space.batch_size == [8, 16, 32, 64, 128, 256]
    assert search.metric == "f1"


@pytest.mark.parametrize(
    "task,dataset",
    [
        ("calibration", "human_calibration"),
        ("pseudo_perplexity", "uniprot_ur100p_test"),
    ],
)
def test_zero_shot_tasks_disable_default_training(task, dataset):
    config = load_evaluation_config(
        [
            CONFIGS / "config.yaml",
            CONFIGS / "tasks" / f"{task}.yaml",
            CONFIGS / "datasets" / f"{dataset}.yaml",
        ]
    )
    assert config.steps.mode == "evaluate_as_is"
    assert not config.steps.prepare.embedding_cache.enabled
    assert not config.steps.tune.hyperparameter_search.enabled
    assert not config.steps.tune.refit_on_train_validation


def test_contact_probe_does_not_inherit_cached_mlp_search():
    config = load_evaluation_config(
        overlays("contact_prediction_casp", CONFIGS / "predictors/contact_embedding.yaml")
    )
    assert config.steps.tune.head.head_type == "bilinear512"
    assert config.steps.mode == "tune_probe"
    assert not config.steps.prepare.embedding_cache.enabled
    assert not config.steps.tune.refit_on_train_validation
    search = config.steps.tune.hyperparameter_search
    assert not search.enabled
    assert not config.steps.tune.standardize_features
    assert config.steps.tune.early_stopping_patience == 10
    assert config.steps.tune.learning_rate == 0.001
    assert config.steps.tune.weight_decay == 10


@pytest.mark.parametrize(
    "dataset",
    sorted((CONFIGS / "datasets").glob("*.yaml")),
    ids=lambda p: p.stem,
)
def test_dataset_overlays_do_not_select_a_task(dataset):
    with pytest.raises(ValueError, match="task_type"):
        load_evaluation_config([CONFIGS / "config.yaml", dataset])


@pytest.mark.parametrize("dataset", sorted(TASK_FOR_DATASET))
def test_explicit_task_and_dataset_compose_to_validated_config(dataset):
    extra = (
        [CONFIGS / "predictors/contact_embedding.yaml"]
        if TASK_FOR_DATASET[dataset] == "contact_prediction"
        else []
    )
    paths = overlays(dataset, *extra)
    config = load_evaluation_config(paths)
    assert config.workspace.task_type == TASK_FOR_DATASET[dataset]
    assert config.model_dump() == load_and_parse(paths, EvaluationConfig).model_dump()


@pytest.mark.parametrize(
    "task,dataset,controls",
    [
        ("sequence_classification", "fold_prediction", ["random_trunk"]),
        ("sequence_multilabel_classification", "GO_bp", ["random_trunk"]),
        ("sequence_regression", "stability_prediction", ["random_trunk"]),
        ("token_classification", "ssp_q3", ["random_trunk"]),
        ("contact_prediction", "contact_prediction_casp", []),
        ("pseudo_perplexity", "uniprot_ur100p_test", ["random_trunk"]),
        ("calibration", "human_calibration", ["random_trunk"]),
    ],
)
def test_task_control_defaults(task, dataset, controls):
    config = load_evaluation_config(
        [
            CONFIGS / "config.yaml",
            CONFIGS / "tasks" / f"{task}.yaml",
            CONFIGS / "datasets" / f"{dataset}.yaml",
            *(
                [CONFIGS / "predictors/contact_embedding.yaml"]
                if task == "contact_prediction"
                else []
            ),
        ]
    )
    assert config.steps.predict.controls == controls
    expected_baselines = {
        "sequence_classification": ["majority_class"],
        "sequence_regression": ["train_mean"],
    }.get(task, [])
    assert config.baselines.names == expected_baselines


def test_protparams_uses_sequence_regression_r2_defaults():
    config = load_evaluation_config(
        [
            CONFIGS / "config.yaml",
            CONFIGS / "models/amplify_120m.yaml",
            CONFIGS / "tasks/sequence_regression.yaml",
            CONFIGS / "datasets/uniprot_protparams.yaml",
            CONFIGS / "predictors/torch_linear.yaml",
        ]
    )
    assert config.steps.prepare.label_column == "gravy"
    assert config.steps.tune.hyperparameter_search.metric == "r2"
    assert config.steps.tune.hyperparameter_search.direction == "maximize"
    assert config.steps.score.metrics == ["r2"]


@pytest.mark.parametrize("task_first", [True, False])
def test_explicit_task_and_dataset_compose_in_either_order(task_first):
    overlays_ = [
        CONFIGS / "tasks/pseudo_perplexity.yaml",
        CONFIGS / "datasets/uniprot_protparams.yaml",
    ]
    if not task_first:
        overlays_.reverse()
    paths = [CONFIGS / "config.yaml", *overlays_]
    config = load_evaluation_config(paths)
    assert config.workspace.task_type == "pseudo_perplexity"
    assert config.model_dump() == load_and_parse(paths, EvaluationConfig).model_dump()


def test_later_overlays_and_cli_customize_explicit_task(tmp_path):
    dataset = tmp_path / "dataset.yaml"
    dataset.write_text(
        "workspace:\n"
        "  dataset_name: custom\n"
        "  target_type: categorical\n"
        "  label_format: scalar\n"
        "  num_labels: 2\n"
        "steps:\n"
        "  predict:\n"
        "    controls: [scrambled_labels]\n"
    )
    paths = [
        CONFIGS / "config.yaml",
        CONFIGS / "tasks/sequence_classification.yaml",
        dataset,
        CONFIGS / "predictors/mlp_finetuning.yaml",
    ]
    config = load_evaluation_config(
        paths,
        overrides=["steps.predict.controls=[]", "steps.tune.max_epochs=3"],
    )
    assert config.steps.mode == "finetune"
    assert not config.steps.prepare.embedding_cache.enabled
    assert config.steps.predict.controls == []
    assert config.steps.tune.max_epochs == 3
    # The dataset has higher priority than the task overlay it follows.
    config = load_evaluation_config(paths[:3])
    assert config.steps.predict.controls == ["scrambled_labels"]


def test_unknown_task_override_fails_with_available_tasks():
    with pytest.raises(ValueError, match="Unknown task_type"):
        load_evaluation_config(
            overlays("GO_bp"),
            overrides=["workspace.task_type=unknown_task"],
        )
