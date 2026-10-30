"""Offline CLI integration for the three contact predictors on the generic engine."""

import json
import sys

import pytest
from accelerate import Accelerator

from modules.evaluate.src import run_evaluate
from modules.evaluate.src.dataset.dataloader import ExternalTestSetConfig
from modules.evaluate.src.dataset.workspace import get_evaluation_workspace

from .test_contact_embedding import (
    config_for,
    development_dataset,
    install_casp_fixture,
    run_engine,
)

CASES = [
    # predictor, expected head, whether a head is fitted and stats written
    ("contact_embedding", "bilinear512", True),
    ("contact_attention", "attention_l1", True),
    ("contact_jacobian", "categorical_jacobian", False),
]


@pytest.mark.parametrize(
    "predictor, head_type, fitted",
    CASES,
    ids=[case[0] for case in CASES],
)
def test_contact_predictor_runs_through_generic_engine(
    predictor, head_type, fitted, model_family, monkeypatch, tmp_path
):
    model_path = install_casp_fixture(
        tmp_path,
        model_family,
        development_dataset(positives=True),
        masked_lm=predictor == "contact_jacobian",
    )
    # Jacobian has no fitted head, so the random-trunk control does not apply.
    controls = ("random_trunk",) if fitted else ()
    config = config_for(tmp_path, model_path, predictor=predictor, controls=controls)
    results = []
    real_run_evaluation = run_evaluate.run_evaluation

    def capture(*args, **kwargs):
        outcome = real_run_evaluation(*args, **kwargs)
        results.append(outcome)
        return outcome

    monkeypatch.setattr(run_evaluate, "load_evaluation_config", lambda **kwargs: config)
    monkeypatch.setattr(
        run_evaluate, "Accelerator", lambda **kwargs: Accelerator(cpu=True)
    )
    assert not hasattr(run_evaluate, "run_pipeline"), "legacy pipeline must be retired"
    monkeypatch.setattr(run_evaluate, "run_evaluation", capture)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_evaluate.py",
            "--suite",
            "contact_prediction",
            "--predictor",
            predictor,
            "--model",
            "esm2_35m",
        ],
    )
    run_evaluate.main()

    workspace = get_evaluation_workspace(
        config.workspace, config.variant, seed=config.seed
    )
    manifest = json.loads(workspace.run_manifest_path.read_text())
    assert manifest["evaluation"]["status"] == "complete"
    assert manifest["predict"]["metadata"]["head_type"] == head_type
    frame = results[0].main.predictions[0].payload.predictions
    assert frame.get_column("record_id").n_unique() == 20
    assert (workspace.scores_path / "unit_scores.parquet").is_file()
    training = workspace.stats_path / "contact_training.json"
    assert training.is_file() is fitted


def test_contact_predictor_skips_external_test_sets_with_warning(
    model_family, tmp_path, caplog
):
    """Contact models do not transfer; configured external sets are skipped, not scored."""
    model_path = install_casp_fixture(
        tmp_path, model_family, development_dataset(positives=True)
    )
    config = config_for(tmp_path, model_path, predictor="contact_embedding")
    config = config.model_copy(
        update={
            "external_test_sets": [
                ExternalTestSetConfig(name="heldout", dataset_repo_id="org/heldout")
            ]
        }
    )
    workspace = get_evaluation_workspace(
        config.workspace, config.variant, seed=config.seed
    )
    with caplog.at_level("WARNING"):
        outcome = run_engine(config, workspace, Accelerator(cpu=True))
    assert outcome.external == {}
    assert "Skipping external_test_sets" in caplog.text
    assert not (workspace.run_root / "external").exists()
