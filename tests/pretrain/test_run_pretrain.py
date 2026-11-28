"""Unit tests for the pretrain entrypoint's runtime wiring.

`setup_runtime` decides Accelerate/distributed behaviour for every run. These
flags are silent when wrong -- training still proceeds, just incorrectly -- so
they are pinned here rather than left to the end-to-end smoke test.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf

import modules.pretrain.src.run_pretrain as run_pretrain
from modules.core.utils.config_loader import load_config
from modules.pretrain.src.config import PretrainConfig
from modules.pretrain.src.loss.loss import FocalLoss, LossConfig

CONFIGS_DIR = Path(__file__).parents[2] / "modules" / "pretrain" / "configs"
MAIN_CONFIG = CONFIGS_DIR / "config.yaml"


@pytest.fixture
def config() -> PretrainConfig:
    return PretrainConfig(
        **OmegaConf.to_container(load_config(MAIN_CONFIG), resolve=True)
    )


@pytest.fixture(autouse=True)
def _restore_global_torch_flags():
    """`setup_runtime` mutates process-wide TF32 state."""
    precision = torch.get_float32_matmul_precision()
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    yield
    torch.set_float32_matmul_precision(precision)
    torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
    torch.backends.cudnn.allow_tf32 = cudnn_tf32


def _run_main(config: PretrainConfig, prepared_model=None):
    """Run `main()` with every component stubbed; return (model, Trainer kwargs).

    `prepared_model`, if given, is what `accelerator.prepare` returns for the
    model, to tell the prepared model apart from the raw one.
    """
    model = MagicMock()
    model.config = MagicMock(spec=[])
    accelerator = MagicMock()

    def prepare(*objs):
        if len(objs) > 1 and prepared_model is not None:
            objs = (prepared_model, *objs[1:])
        return objs if len(objs) > 1 else objs[0]

    accelerator.prepare.side_effect = prepare
    with (
        patch.object(run_pretrain, "load_and_parse", return_value=config),
        patch.object(run_pretrain, "setup_runtime", return_value=accelerator),
        patch.object(run_pretrain, "setup_tracking"),
        patch.object(run_pretrain, "get_amplify_masked_lm", return_value=model),
        patch.object(run_pretrain, "get_tokenizer"),
        patch.object(run_pretrain, "get_optimizer"),
        patch.object(run_pretrain, "get_scheduler"),
        patch.object(run_pretrain, "get_collator"),
        patch.object(run_pretrain, "prepare_sources", return_value=[]),
        patch.object(run_pretrain, "ConcatDataset"),
        patch.object(run_pretrain, "build_split_dataloader"),
        patch.object(run_pretrain, "Trainer") as trainer_cls,
        patch.object(run_pretrain.sys, "argv", ["run_pretrain.py", "cfg.yaml"]),
    ):
        run_pretrain.main()
    return model, trainer_cls.call_args.kwargs


def _run(config: PretrainConfig):
    """Call setup_runtime with Accelerate stubbed, returning its kwargs."""
    with (
        patch.object(run_pretrain, "Accelerator") as accelerator_cls,
        patch.object(run_pretrain, "set_seed") as set_seed,
    ):
        accelerator_cls.return_value = MagicMock()
        run_pretrain.setup_runtime(config)
    return accelerator_cls.call_args.kwargs, set_seed


class TestSetupTracking:
    @pytest.mark.parametrize("run_id", [None, "resume-run-123"])
    @pytest.mark.parametrize("custom_dir", [False, True])
    def test_run_id_is_forwarded_without_deriving_it_from_name(
        self, config, tmp_path, run_id, custom_dir
    ):
        config.wandb.enabled = True
        config.wandb.run_name = "Repeated display name / experiment"
        config.wandb.id = run_id
        config.wandb.resume = "must" if run_id else "allow"
        config.trainer.output_dir = tmp_path / "run"
        if custom_dir:
            config.wandb.dir = tmp_path / "tracking"
        accelerator = MagicMock(is_main_process=True)

        run_pretrain.setup_tracking(config, accelerator)

        accelerator.init_trackers.assert_called_once()
        kwargs = accelerator.init_trackers.call_args.kwargs
        tracking = kwargs["init_kwargs"]["wandb"]
        assert tracking["id"] == run_id
        assert tracking["name"] == config.wandb.run_name
        assert tracking["resume"] == config.wandb.resume
        directory = config.wandb.dir or config.trainer.output_dir / "wandb"
        assert tracking["dir"] == str(directory)
        assert directory.is_dir()

    def test_disabled_tracking_does_not_create_directory(self, config, tmp_path):
        config.wandb.enabled = False
        config.trainer.output_dir = tmp_path / "run"
        accelerator = MagicMock()
        run_pretrain.setup_tracking(config, accelerator)
        accelerator.init_trackers.assert_not_called()
        assert not config.trainer.output_dir.exists()

    def test_non_main_process_leaves_directory_creation_to_main(self, config, tmp_path):
        config.wandb.enabled = True
        config.trainer.output_dir = tmp_path / "run"
        accelerator = MagicMock(is_main_process=False)
        run_pretrain.setup_tracking(config, accelerator)
        assert not config.trainer.output_dir.exists()
        accelerator.init_trackers.assert_called_once()


class TestSetupRuntime:
    def test_scheduler_is_not_stepped_by_accelerate(self, config):
        """Trainer steps the scheduler itself; Accelerate must not also step it.

        Otherwise the LR decays once per rank per step on multi-GPU.
        """
        kwargs, _ = _run(config)
        assert kwargs["step_scheduler_with_optimizer"] is False

    def test_even_batches_is_disabled(self, config):
        # The dataloader already drops the tail, so Accelerate must not re-pad.
        kwargs, _ = _run(config)
        assert kwargs["dataloader_config"].even_batches is False

    def test_gradient_accumulation_is_forwarded(self, config):
        config.trainer.gradient_accumulation_steps = 4
        kwargs, _ = _run(config)
        assert kwargs["gradient_accumulation_steps"] == 4

    def test_distributed_timeout_is_forwarded(self, config):
        config.trainer.dist_timeout_minutes = 42
        kwargs, _ = _run(config)
        handler = kwargs["kwargs_handlers"][0]
        assert handler.timeout.total_seconds() == 42 * 60

    def test_rngs_are_seeded_from_the_dataset_seed(self, config):
        config.dataset.seed = 4242
        _, set_seed = _run(config)
        set_seed.assert_called_once_with(4242, deterministic=False)


class TestDeterministic:
    """`deterministic` is opt-in debugging aid; it must stay off by default."""

    def test_defaults_to_disabled(self, config):
        assert config.trainer.deterministic is False

    def test_forwarded_to_set_seed(self, config):
        config.trainer.deterministic = True
        _, set_seed = _run(config)
        assert set_seed.call_args.kwargs["deterministic"] is True

    def test_sets_cublas_workspace_when_enabled(self, config, monkeypatch):
        # Deterministic cuBLAS GEMMs raise unless this is set before the handle.
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
        config.trainer.deterministic = True
        _run(config)
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"

    def test_leaves_cublas_workspace_alone_when_disabled(self, config, monkeypatch):
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
        config.trainer.deterministic = False
        _run(config)
        assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ

    def test_warns_when_enabled(self, config, caplog):
        config.trainer.deterministic = True
        with caplog.at_level(logging.WARNING):
            _run(config)
        assert "not bitwise reproducible" in caplog.text

    def test_does_not_warn_when_disabled(self, config, caplog):
        config.trainer.deterministic = False
        with caplog.at_level(logging.WARNING):
            _run(config)
        assert "deterministic" not in caplog.text


class TestCompileWiring:
    def test_accelerate_does_not_compile(self, config):
        """Accelerate's dynamo plugin would wrap the model in an
        OptimizedModule; `main` compiles it in place instead."""
        config.trainer.compile = True
        kwargs, _ = _run(config)
        assert kwargs.get("dynamo_plugin") is None

    def test_compiles_the_prepared_model_in_place(self, config):
        config.trainer.compile = True
        config.trainer.compile_mode = "max-autotune"
        config.trainer.compile_fullgraph = False
        config.trainer.compile_dynamic = False
        prepared = MagicMock()

        model, kwargs = _run_main(config, prepared_model=prepared)

        prepared.compile.assert_called_once_with(
            mode="max-autotune", fullgraph=False, dynamic=False
        )
        model.compile.assert_not_called()  # not before the DDP wrap
        assert kwargs["model"] is prepared  # Trainer gets the module itself

    def test_does_not_compile_when_disabled(self, config):
        config.trainer.compile = False
        prepared = MagicMock()
        _run_main(config, prepared_model=prepared)
        prepared.compile.assert_not_called()


class TestTrackingWiring:
    def test_wandb_is_registered_only_when_enabled(self, config):
        config.wandb.enabled = True
        assert _run(config)[0]["log_with"] == "wandb"

        config.wandb.enabled = False
        assert _run(config)[0]["log_with"] is None


class TestTF32:
    @pytest.mark.parametrize(
        "tf32, expected_precision", [(True, "high"), (False, "highest")]
    )
    def test_tf32_flag_drives_matmul_precision(self, config, tf32, expected_precision):
        config.trainer.tf32 = tf32
        _run(config)

        assert torch.get_float32_matmul_precision() == expected_precision
        assert torch.backends.cuda.matmul.allow_tf32 is tf32
        assert torch.backends.cudnn.allow_tf32 is tf32


class TestPrepareOnly:
    """`dataset.prepare_only` builds caches then exits, for warming a slow
    first-time build on CPU-only hardware."""

    def test_cross_entropy_keeps_model_loss(self, config):
        config.loss = LossConfig()
        model, kwargs = _run_main(config)

        assert kwargs["loss_fn"] is None
        assert kwargs["metric_logger"].log_cross_entropy is False
        assert model.config.pretraining_loss == {"type": "cross_entropy", "gamma": 0.5}

    def test_focal_is_passed_to_trainer_and_recorded(self, config):
        config.loss = LossConfig(type="focal", gamma=0.25)
        model, kwargs = _run_main(config)

        assert isinstance(kwargs["loss_fn"], FocalLoss)
        assert kwargs["loss_fn"].gamma == 0.25
        assert kwargs["metric_logger"].log_cross_entropy is True
        assert model.config.pretraining_loss == {"type": "focal", "gamma": 0.25}


def test_distributed_training_requires_warm_caches_before_runtime(config, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")
    with (
        patch.object(run_pretrain, "load_and_parse", return_value=config),
        patch.object(run_pretrain.sys, "argv", ["run_pretrain.py", "cfg.yaml"]),
        patch.object(
            run_pretrain, "prepare_sources", side_effect=RuntimeError("cache missing")
        ) as prepare_sources,
        patch.object(run_pretrain, "setup_runtime") as setup_runtime,
        pytest.raises(RuntimeError, match="cache missing"),
    ):
        run_pretrain.main()

    prepare_sources.assert_called_once_with(config.dataset, require_prepared=True)
    setup_runtime.assert_not_called()
