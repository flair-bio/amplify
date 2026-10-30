"""Suite CLI: explicit forms, offline dry-run and rejected legacy inputs."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from modules.evaluate.src import run_evaluate


def _run(monkeypatch: pytest.MonkeyPatch, *argv: str) -> None:
    monkeypatch.setattr("sys.argv", ["run_evaluate.py", *argv])
    run_evaluate.main()


@pytest.fixture
def no_model_work(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if a dry run reaches accelerate, the runner or model loading."""

    def boom(*args, **kwargs):
        raise AssertionError("dry-run must not start model or data work")

    monkeypatch.setattr(run_evaluate, "Accelerator", boom)
    monkeypatch.setattr(run_evaluate, "run_evaluation", boom)


def test_dry_run_single_model_composes_without_model_work(
    monkeypatch: pytest.MonkeyPatch, capsys, no_model_work
) -> None:
    _run(
        monkeypatch,
        "--suite",
        "calibration",
        "--model",
        "amplify_120m",
        "--dry-run",
        "--seed",
        "7",
    )
    out = capsys.readouterr().out
    assert "calibration/human_calibration/default" in out
    assert "model=amplify_120m" in out
    assert "seed=7" in out
    assert "1 run(s) validated" in out


def test_key_value_overrides_may_follow_config_options(
    monkeypatch: pytest.MonkeyPatch, capsys, tmp_path, no_model_work
) -> None:
    """Overrides on both sides of a ``--config`` option are all accepted."""
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text("workspace:\n  run_id: from_overlay\n")
    _run(
        monkeypatch,
        "--suite",
        "sequence_classification",
        "--baselines",
        "workspace.run_id=before_config",
        "--config",
        str(overlay),
        "--dry-run",
        "steps.score.n_samples=1000",
    )
    assert "run(s) validated" in capsys.readouterr().out


def test_dry_run_collection_expands_every_model(
    monkeypatch: pytest.MonkeyPatch, capsys, no_model_work
) -> None:
    _run(
        monkeypatch,
        "--suite",
        "calibration",
        "--model-collection",
        "production",
        "--dry-run",
    )
    out = capsys.readouterr().out
    assert "4 run(s) validated" in out
    for label in ("amplify_120m", "amplify_350m", "esm_150m", "esm2_35m"):
        assert f"model={label}" in out


def test_dry_run_predictor_selector_filters_contact_jobs(
    monkeypatch: pytest.MonkeyPatch, capsys, no_model_work
) -> None:
    _run(
        monkeypatch,
        "--suite",
        "contact_prediction",
        "--model",
        "esm2_35m",
        "--predictor",
        "contact_attention",
        "--dry-run",
        f"workspace.dataset_revision={'a' * 40}",
    )
    out = capsys.readouterr().out
    assert "1 run(s) validated" in out
    assert "contact_attention" in out
    assert "contact_embedding" not in out


def test_contact_seed_defaults_to_the_base_config_seed(
    monkeypatch: pytest.MonkeyPatch, capsys, no_model_work
) -> None:
    _run(
        monkeypatch,
        "--suite",
        "contact_prediction",
        "--model",
        "esm2_35m",
        "--predictor",
        "contact_attention",
        "--dry-run",
        f"workspace.dataset_revision={'a' * 40}",
    )
    assert "seed=710019" in capsys.readouterr().out


def test_contact_seed_flag_overrides_the_default(
    monkeypatch: pytest.MonkeyPatch, capsys, no_model_work
) -> None:
    _run(
        monkeypatch,
        "--suite",
        "contact_prediction",
        "--model",
        "esm2_35m",
        "--predictor",
        "contact_attention",
        "--seed",
        "42",
        "--dry-run",
        f"workspace.dataset_revision={'a' * 40}",
    )
    out = capsys.readouterr().out
    assert "seed=42" in out
    assert "seed=710019" not in out


def test_predictor_selector_without_match_errors(
    monkeypatch: pytest.MonkeyPatch, no_model_work
) -> None:
    with pytest.raises(SystemExit, match="no jobs match"):
        _run(
            monkeypatch,
            "--suite",
            "calibration",
            "--model",
            "amplify_120m",
            "--predictor",
            "contact_embedding",
            "--dry-run",
        )


def test_dry_run_baselines_reports_shared_run(
    monkeypatch: pytest.MonkeyPatch, capsys, no_model_work
) -> None:
    _run(monkeypatch, "--suite", "sequence_classification", "--baselines", "--dry-run")
    out = capsys.readouterr().out
    assert "4 run(s) validated" in out
    assert "model=-" in out


def test_baselines_over_all_suites_is_rejected_with_unsupported_suites(
    monkeypatch: pytest.MonkeyPatch, no_model_work
) -> None:
    with pytest.raises(SystemExit) as error:
        _run(monkeypatch, "--suite", "all", "--baselines", "--dry-run")
    message = str(error.value)
    assert "no shared baseline is implemented" in message
    for unsupported in ("calibration", "sequence_multilabel_classification"):
        assert unsupported in message
    for supported in (
        "sequence_regression",
        "contact_prediction",
        "token_classification",
    ):
        assert supported not in message.split("suite(s)", 1)[1].split(".", 1)[0]


def test_baselines_for_suite_without_shared_baselines_is_rejected(
    monkeypatch: pytest.MonkeyPatch, no_model_work
) -> None:
    with pytest.raises(SystemExit, match="no shared baseline is implemented"):
        _run(monkeypatch, "--suite", "calibration", "--baselines", "--dry-run")


def test_token_baseline_suite_covers_every_token_dataset(
    monkeypatch, capsys, no_model_work
):
    import yaml

    from modules.evaluate.src.config.suites import CONFIG_ROOT, expand_suite

    expected = {
        path.stem
        for path in (CONFIG_ROOT / "datasets").glob("*.yaml")
        if yaml.safe_load(path.read_text()).get("dataset", {}).get("label_format")
        == "per_token"
    }
    jobs = expand_suite("token_classification")
    assert {job.dataset for job in jobs} == expected
    _run(monkeypatch, "--suite", "token_classification", "--baselines", "--dry-run")
    assert f"{len(expected)} run(s) validated" in capsys.readouterr().out


def test_dry_run_baselines_supports_contact_and_pseudo_perplexity(
    monkeypatch: pytest.MonkeyPatch, capsys, no_model_work
) -> None:
    pinned = "0123456789abcdef0123456789abcdef01234567"
    _run(
        monkeypatch,
        "--suite",
        "contact_prediction",
        "--baselines",
        "--dry-run",
        f"workspace.dataset_revision={pinned}",
    )
    assert "model=-" in capsys.readouterr().out


def test_baseline_jobs_inject_structural_names_only(monkeypatch) -> None:
    from modules.evaluate.src.config.suites import EvaluationJob

    contact = EvaluationJob("contact_prediction", "contact_prediction", "casp", "x")
    sequence = EvaluationJob("sequence_regression", "sequence_regression", "d", None)
    injected = run_evaluate._baseline_job(contact)
    assert injected.predictor is None
    assert injected.overrides == ("baselines.names=[separation_prior]",)
    untouched = run_evaluate._baseline_job(sequence)
    assert untouched.overrides == ()


def test_legacy_positional_yaml_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit, match="no longer accepted"):
        _run(
            monkeypatch,
            "--suite",
            "calibration",
            "--model",
            "amplify_120m",
            "modules/evaluate/configs/config.yaml",
        )


def test_checkpoint_directory_is_accepted_in_dry_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path, no_model_work
) -> None:
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    _run(
        monkeypatch,
        "--suite",
        "calibration",
        "--model",
        str(checkpoint),
        "--dry-run",
    )


def test_data_root_missing_dataset_is_an_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path, no_model_work
) -> None:
    with pytest.raises(SystemExit, match="never falls back to the Hub"):
        _run(
            monkeypatch,
            "--suite",
            "calibration",
            "--model",
            "amplify_120m",
            "--data-root",
            str(tmp_path),
            "--dry-run",
        )


def test_data_root_must_be_a_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path, no_model_work
) -> None:
    with pytest.raises(SystemExit, match="not a directory"):
        _run(
            monkeypatch,
            "--suite",
            "calibration",
            "--model",
            "amplify_120m",
            "--data-root",
            str(tmp_path / "missing"),
            "--dry-run",
        )


def test_overwrite_with_baselines_is_rejected(
    monkeypatch: pytest.MonkeyPatch, no_model_work
) -> None:
    with pytest.raises(SystemExit, match="applies to model runs only"):
        _run(
            monkeypatch,
            "--suite",
            "calibration",
            "--baselines",
            "--overwrite",
            "--dry-run",
        )


def test_model_name_requires_single_model(
    monkeypatch: pytest.MonkeyPatch, no_model_work
) -> None:
    with pytest.raises(SystemExit, match="only valid with --model"):
        _run(
            monkeypatch,
            "--suite",
            "calibration",
            "--model-collection",
            "production",
            "--model-name",
            "x",
            "--dry-run",
        )


def test_model_or_collection_or_baselines_is_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(SystemExit):
        _run(monkeypatch, "--suite", "calibration")


def test_legacy_cross_validation_override_rejected_at_dry_run(
    monkeypatch: pytest.MonkeyPatch, no_model_work
) -> None:
    with pytest.raises(SystemExit, match="cross_validation"):
        _run(
            monkeypatch,
            "--suite",
            "calibration",
            "--model",
            "amplify_120m",
            "--dry-run",
            "cross_validation.enabled=true",
        )


def test_unclaimed_config_raises_before_running_evaluation(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """No engine claims the job: fail loudly instead of silently doing nothing."""
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "workspace": {
                    "run_id": "test",
                    "base_path": str(tmp_path),
                    "model_name": "offline_model",
                    "dataset_name": "offline_dataset",
                    "task_type": "sequence_classification",
                    "target_type": "categorical",
                    "label_format": "scalar",
                    "num_labels": 2,
                }
            }
        )
    )
    monkeypatch.setattr(
        run_evaluate,
        "Accelerator",
        lambda **kwargs: SimpleNamespace(is_main_process=True),
    )
    monkeypatch.setattr(run_evaluate, "seed_everything", lambda seed: None)
    monkeypatch.setattr(run_evaluate, "resolve_engine", lambda config: None)
    run = Mock()
    monkeypatch.setattr(run_evaluate, "run_evaluation", run)

    precision = torch.get_float32_matmul_precision()
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        with pytest.raises(ValueError) as error:
            run_evaluate._run_model_evaluation(
                [str(config_path)], ["steps.mode=tune_probe"], False
            )
    finally:
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32

    message = str(error.value)
    assert "task_type='sequence_classification'" in message
    assert "steps.mode='tune_probe'" in message
    run.assert_not_called()
