"""Dataset defaults preserve complete task behavior and explicit overrides."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from modules.core.utils.config_loader import load_and_parse
from modules.evaluate.src.config import EvaluationConfig
from modules.evaluate.src.utils.config_loader import load_evaluation_config

CONFIGS = Path(__file__).resolve().parents[2] / "modules/evaluate/configs"


@pytest.mark.parametrize("tuning", [None, "torch_linear", "mlp_finetuning"])
def test_neural_overlays_select_continuous_log_weight_decay(tuning):
    paths = [CONFIGS / "config.yaml", CONFIGS / "datasets/fold_prediction.yaml"]
    if tuning is not None:
        paths.append(CONFIGS / "tuning" / f"{tuning}.yaml")
    config = load_evaluation_config(paths)
    search = config.steps.tune.hyperparameter_search
    assert search.enabled
    assert search.search_pattern == "bohb"
    assert search.search_space.weight_decay_sampling == "log"
    assert min(search.search_space.weight_decay) == 1e-6
    assert max(search.search_space.weight_decay) == 1e-1


def test_default_protocol_needs_only_base_model_and_dataset():
    config = load_evaluation_config(
        [
            CONFIGS / "config.yaml",
            CONFIGS / "models/amplify_120m.yaml",
            CONFIGS / "datasets/localization_prediction.yaml",
        ]
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


def test_token_classification_uses_residue_scale_hpo():
    config = load_evaluation_config(
        [
            CONFIGS / "config.yaml",
            CONFIGS / "tasks/token_classification.yaml",
            CONFIGS / "datasets/uniprot_secondary_structure.yaml",
        ]
    )
    search = config.steps.tune.hyperparameter_search
    assert config.workspace.task_type == "token_classification"
    assert search.n_trials == 100
    assert search.search_space.batch_size == [1024, 2048, 4096]
    assert search.search_space.max_epochs == [1, 2, 5, 10]


@pytest.mark.parametrize(
    "task,dataset",
    [
        ("calibration", "human_calibration"),
        ("pseudo_perplexity", "uniprot_protparams"),
        ("categorical_jacobian", "contact_prediction_binary"),
    ],
)
def test_zero_shot_tasks_disable_default_training(task, dataset):
    config = load_evaluation_config(
        [CONFIGS / "config.yaml", CONFIGS / "datasets" / f"{dataset}.yaml"],
        overrides=[f"workspace.task_type={task}"],
    )
    assert config.steps.mode == "evaluate_as_is"
    assert not config.steps.prepare.embedding_cache.enabled
    assert not config.steps.tune.hyperparameter_search.enabled
    assert not config.steps.tune.refit_on_train_validation


def test_contact_probe_does_not_inherit_cached_mlp_search():
    config = load_evaluation_config(
        [CONFIGS / "config.yaml", CONFIGS / "datasets/contact_prediction_binary.yaml"]
    )
    assert config.steps.mode == "tune_probe"
    assert not config.steps.prepare.embedding_cache.enabled
    assert not config.steps.tune.refit_on_train_validation
    search = config.steps.tune.hyperparameter_search
    assert search.enabled
    assert search.metric == "contact_auc_long"
    assert search.search_space.hidden_size is None
    assert search.search_space.hidden_depth is None
    assert search.search_space.batch_size is None


@pytest.mark.parametrize(
    "dataset",
    sorted((CONFIGS / "datasets").glob("*.yaml")),
    ids=lambda p: p.stem,
)
def test_dataset_default_matches_explicit_task_config(dataset):
    task = OmegaConf.load(dataset).workspace.default_task_type
    expected = load_and_parse(
        [CONFIGS / "config.yaml", CONFIGS / "tasks" / f"{task}.yaml", dataset],
        EvaluationConfig,
    )
    actual = load_evaluation_config([CONFIGS / "config.yaml", dataset])
    assert actual.model_dump() == expected.model_dump()


@pytest.mark.parametrize(
    "task,dataset,controls",
    [
        (
            "sequence_classification",
            "fold_prediction",
            ["random_trunk", "majority_class"],
        ),
        ("sequence_multilabel_classification", "GO_bp", ["random_trunk"]),
        ("sequence_regression", "stability_prediction", ["random_trunk", "train_mean"]),
        ("token_classification", "ssp_q3", ["random_trunk", "majority_class"]),
        (
            "contact_prediction",
            "contact_prediction_binary",
            ["random_trunk", "separation_prior"],
        ),
        (
            "categorical_jacobian",
            "contact_prediction_binary",
            ["random_trunk", "separation_prior"],
        ),
        ("pseudo_perplexity", "uniprot_protparams", ["random_trunk"]),
        ("calibration", "human_calibration", ["random_trunk"]),
    ],
)
def test_task_control_defaults(task, dataset, controls):
    config = load_evaluation_config(
        [
            CONFIGS / "config.yaml",
            CONFIGS / "tasks" / f"{task}.yaml",
            CONFIGS / "datasets" / f"{dataset}.yaml",
        ]
    )
    assert config.steps.predict.controls == controls


def test_protparams_uses_sequence_regression_r2_defaults():
    config = load_evaluation_config(
        [
            CONFIGS / "config.yaml",
            CONFIGS / "models/amplify_120m.yaml",
            CONFIGS / "tasks/sequence_regression.yaml",
            CONFIGS / "datasets/uniprot_protparams.yaml",
            CONFIGS / "tuning/torch_linear.yaml",
        ]
    )
    assert config.steps.prepare.label_column == "gravy"
    assert config.steps.tune.hyperparameter_search.metric == "r2"
    assert config.steps.tune.hyperparameter_search.direction == "maximize"
    assert config.steps.score.metrics == ["r2"]


@pytest.mark.parametrize("task_first", [True, False])
@pytest.mark.parametrize(
    "dataset,task",
    [
        ("contact_prediction_binary", "categorical_jacobian"),
        ("uniprot_protparams", "pseudo_perplexity"),
    ],
)
def test_explicit_method_overrides_dataset_default_in_either_order(
    dataset, task, task_first
):
    overlays = [
        CONFIGS / "tasks" / f"{task}.yaml",
        CONFIGS / "datasets" / f"{dataset}.yaml",
    ]
    if not task_first:
        overlays.reverse()
    paths = [CONFIGS / "config.yaml", *overlays]
    config = load_evaluation_config(paths)
    assert config.workspace.task_type == task
    assert config.model_dump() == load_and_parse(paths, EvaluationConfig).model_dump()


def test_later_overlays_and_cli_can_customize_implicit_task(tmp_path):
    dataset = tmp_path / "dataset.yaml"
    dataset.write_text(
        "workspace:\n"
        "  default_task_type: sequence_classification\n"
        "  dataset_name: custom\n"
        "  target_type: categorical\n"
        "  label_format: scalar\n"
        "  num_labels: 2\n"
        "steps:\n"
        "  predict:\n"
        "    controls: [majority_class]\n"
    )
    config = load_evaluation_config(
        [CONFIGS / "config.yaml", dataset, CONFIGS / "tuning/mlp_finetuning.yaml"],
        overrides=["steps.predict.controls=[]", "steps.tune.max_epochs=3"],
    )
    assert config.steps.mode == "finetune"
    assert not config.steps.prepare.embedding_cache.enabled
    assert config.steps.predict.controls == []
    assert config.steps.tune.max_epochs == 3
    # The dataset itself also has higher priority than the implicit task.
    config = load_evaluation_config([CONFIGS / "config.yaml", dataset])
    assert config.steps.predict.controls == ["majority_class"]


def test_cli_task_selection_loads_method_behavior():
    config = load_evaluation_config(
        [CONFIGS / "config.yaml", CONFIGS / "datasets/contact_prediction_binary.yaml"],
        overrides=["workspace.task_type=categorical_jacobian"],
    )
    assert config.workspace.task_type == "categorical_jacobian"
    assert config.steps.mode == "evaluate_as_is"
    assert not config.steps.prepare.embedding_cache.enabled
    assert config.steps.predict.controls == ["random_trunk", "separation_prior"]


def test_unknown_default_fails_before_loading_task_file():
    with pytest.raises(ValueError, match="Unknown task type"):
        load_evaluation_config(
            [CONFIGS / "config.yaml", CONFIGS / "datasets/GO_bp.yaml"],
            overrides=["workspace.default_task_type=unknown_task"],
        )
