"""Dataset defaults preserve complete task behavior and explicit overrides."""

from pathlib import Path

from omegaconf import OmegaConf
import pytest

from modules.core.utils.config_loader import load_and_parse
from modules.evaluate.src.config import EvaluationConfig
from modules.evaluate.src.utils.config_loader import load_evaluation_config

CONFIGS = Path(__file__).resolve().parents[2] / "modules/evaluate/configs"


@pytest.mark.parametrize("tuning", ["mlp", "torch_linear", "mlp_finetuning"])
def test_neural_overlays_select_continuous_log_weight_decay(tuning):
    config = load_evaluation_config(
        [
            CONFIGS / "config.yaml",
            CONFIGS / "datasets/fold_prediction.yaml",
            CONFIGS / "tuning" / f"{tuning}.yaml",
        ]
    )
    search = config.steps.tune.hyperparameter_search
    assert search.enabled
    assert search.search_pattern == "bohb"
    assert search.search_space.weight_decay_sampling == "log"
    assert min(search.search_space.weight_decay) == 1e-6
    assert max(search.search_space.weight_decay) == 1e-1


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
    assert config.steps.predict.controls == ["scrambled_labels", "separation_prior"]


def test_unknown_default_fails_before_loading_task_file():
    with pytest.raises(ValueError, match="Unknown task type"):
        load_evaluation_config(
            [CONFIGS / "config.yaml", CONFIGS / "datasets/GO_bp.yaml"],
            overrides=["workspace.default_task_type=unknown_task"],
        )
