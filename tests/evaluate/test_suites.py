"""Suite expansion is explicit, validated and offline."""

from pathlib import Path

import pytest

from modules.evaluate.src.config import suites
from modules.evaluate.src.config.suites import (
    SuiteError,
    expand_suite,
    model_label,
    resolve_model,
    resolve_model_collection,
    select_jobs,
)

TASK_SUITES = [
    "sequence_regression",
    "sequence_classification",
    "sequence_multilabel_classification",
    "token_classification",
    "contact_prediction",
    "pseudo_perplexity",
    "calibration",
]


def _write(tmp_path: Path, name: str, text: str) -> str:
    path = tmp_path / f"{name}.yaml"
    path.write_text(text)
    return str(path)


@pytest.mark.parametrize("name", TASK_SUITES)
def test_each_task_suite_expands_with_explicit_task(name: str) -> None:
    jobs = expand_suite(name)
    assert jobs
    assert {job.task for job in jobs} == {name}
    assert all(job.suite == name for job in jobs)


def test_all_suite_matches_plan_inventory() -> None:
    jobs = expand_suite("all")
    datasets = {job.dataset for job in jobs}
    assert len(datasets) == 35
    assert len(jobs) == 37
    assert len({job.evaluation_id for job in jobs}) == len(jobs)


def test_contact_suite_expands_three_explicit_predictors() -> None:
    jobs = expand_suite("contact_prediction")
    assert [job.predictor for job in jobs] == [
        "contact_embedding",
        "contact_attention",
        "contact_jacobian",
    ]
    assert {job.dataset for job in jobs} == {"contact_prediction_casp"}


def test_sequence_suites_use_task_default_predictor() -> None:
    jobs = expand_suite("sequence_regression")
    assert {job.predictor for job in jobs} == {None}
    assert "uniprot_protparams" in {job.dataset for job in jobs}


def test_select_predictor_filters_before_loading_and_errors_on_no_match() -> None:
    jobs = expand_suite("contact_prediction")
    selected = select_jobs(jobs, predictor="contact_attention")
    assert [job.predictor for job in selected] == ["contact_attention"]
    with pytest.raises(SuiteError, match="no jobs match"):
        select_jobs(expand_suite("sequence_regression"), predictor="contact_embedding")


def test_select_dataset_filters_and_rejects_unknown_selection() -> None:
    jobs = expand_suite("sequence_regression")
    assert [job.dataset for job in select_jobs(jobs, dataset="meltome_human")] == [
        "meltome_human"
    ]
    with pytest.raises(SuiteError):
        select_jobs(jobs, dataset="not_a_dataset")


def test_unknown_task_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, "bad_task", "suite: bad\ntask: nope\nevaluations: []\n")
    with pytest.raises(SuiteError, match="unknown task"):
        expand_suite(path)


def test_unknown_dataset_is_rejected(tmp_path: Path) -> None:
    text = "suite: bad\ntask: calibration\nevaluations:\n  - dataset: missing_dataset\n"
    with pytest.raises(SuiteError, match="unknown dataset"):
        expand_suite(_write(tmp_path, "bad_dataset", text))


def test_unknown_predictor_is_rejected(tmp_path: Path) -> None:
    text = (
        "suite: bad\ntask: calibration\npredictors: [not_a_predictor]\n"
        "evaluations:\n  - dataset: human_calibration\n"
    )
    with pytest.raises(SuiteError, match="unknown predictor"):
        expand_suite(_write(tmp_path, "bad_predictor", text))


def test_aggregate_suite_may_not_declare_evaluations(tmp_path: Path) -> None:
    text = "suite: mixed\nincludes: [calibration]\ntask: calibration\n"
    with pytest.raises(SuiteError, match="aggregate"):
        expand_suite(_write(tmp_path, "mixed", text))


def test_duplicate_evaluation_is_rejected(tmp_path: Path) -> None:
    text = (
        "suite: dup\ntask: calibration\nevaluations:\n"
        "  - dataset: human_calibration\n  - dataset: human_calibration\n"
    )
    with pytest.raises(SuiteError, match="duplicate evaluation"):
        expand_suite(_write(tmp_path, "dup", text))


def test_include_cycle_is_rejected(tmp_path: Path) -> None:
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text(f"suite: a\nincludes: ['{b}']\n")
    b.write_text(f"suite: b\nincludes: ['{a}']\n")
    with pytest.raises(SuiteError, match="include cycle"):
        expand_suite(str(a))


def test_config_paths_follow_overlay_order() -> None:
    job = expand_suite("contact_prediction")[0]
    model = Path("model.yaml")
    names = [p.name for p in job.config_paths(model)]
    assert names == [
        "config.yaml",
        "model.yaml",
        "contact_prediction.yaml",
        "contact_prediction_casp.yaml",
        "contact_embedding.yaml",
    ]


def test_model_name_and_yaml_paths_resolve() -> None:
    named = resolve_model("amplify_120m")
    assert named.source == "config"
    assert named.config_path is not None
    assert named.config_path.name == "amplify_120m.yaml"
    assert named.label == "amplify_120m"
    explicit = resolve_model(str(named.config_path))
    assert explicit.config_path == named.config_path
    assert explicit.overrides == ()


def test_checkpoint_directory_resolves_to_model_overrides(tmp_path: Path) -> None:
    checkpoint = tmp_path / "amplify_ckpt_10000"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    ref = resolve_model(str(checkpoint))
    assert ref.source == "checkpoint"
    assert ref.config_path is None
    assert ref.label == "amplify_ckpt_10000"
    assert f"model.repo_id={checkpoint.resolve()}" in ref.overrides
    assert "model.name=amplify_ckpt_10000" in ref.overrides


def test_checkpoint_directory_requires_config_json(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    with pytest.raises(SuiteError, match="config.json"):
        resolve_model(str(checkpoint))


def test_missing_local_model_path_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(SuiteError, match="not a checkpoint directory"):
        resolve_model(str(tmp_path / "nope"))


def test_unsafe_checkpoint_name_requires_model_name(tmp_path: Path) -> None:
    checkpoint = tmp_path / "bad name!"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    with pytest.raises(SuiteError, match="--model-name"):
        resolve_model(str(checkpoint))
    ref = resolve_model(str(checkpoint), label="my-model")
    assert ref.label == "my-model"


def test_remote_identifier_gets_deterministic_or_explicit_label() -> None:
    ref = resolve_model("flair-bio/amplify-120m")
    assert ref.source == "remote"
    assert ref.label == "flair-bio--amplify-120m"
    assert "model.repo_id=flair-bio/amplify-120m" in ref.overrides
    assert resolve_model("flair-bio/amplify-120m", label="amp").label == "amp"


def test_missing_model_config_is_rejected() -> None:
    with pytest.raises(SuiteError, match="not found"):
        resolve_model("no_such_model")


def test_production_model_collection_resolves_in_order() -> None:
    names = [ref.config_path.name for ref in resolve_model_collection("production")]
    assert names == [
        "amplify_120m.yaml",
        "amplify_350m.yaml",
        "esm_150m.yaml",
        "esm2_35m.yaml",
    ]


def test_collection_rejects_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "dups.yaml"
    path.write_text("models: [amplify_120m, amplify_120m]\n")
    with pytest.raises(SuiteError, match="duplicate"):
        resolve_model_collection(str(path))


def test_collection_rejects_duplicate_labels_from_different_files(
    tmp_path: Path,
) -> None:
    first, second = tmp_path / "a" / "x.yaml", tmp_path / "b" / "x.yaml"
    for path in (first, second):
        path.parent.mkdir(exist_ok=True)
        path.write_text("workspace: {}\n")
    collection = tmp_path / "labels.yaml"
    collection.write_text(f"models: [{first}, {second}]\n")
    with pytest.raises(SuiteError, match="duplicate labels"):
        resolve_model_collection(str(collection))


def test_model_label_rejects_unsafe_names() -> None:
    with pytest.raises(SuiteError, match="unsafe"):
        model_label(Path("bad name!.yaml"))


def test_suite_directory_is_under_configs() -> None:
    assert suites.SUITE_DIR.is_dir()
    assert (suites.SUITE_DIR / "all.yaml").is_file()
