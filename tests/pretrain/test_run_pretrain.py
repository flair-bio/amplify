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

from modules.core.utils.config_loader import load_config
from modules.pretrain.src.config import PretrainConfig
import modules.pretrain.src.run_pretrain as run_pretrain

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


def _run(config: PretrainConfig):
    """Call setup_runtime with Accelerate stubbed, returning its kwargs."""
    with (
        patch.object(run_pretrain, "Accelerator") as accelerator_cls,
        patch.object(run_pretrain, "set_seed") as set_seed,
    ):
        accelerator_cls.return_value = MagicMock()
        run_pretrain.setup_runtime(config)
    return accelerator_cls.call_args.kwargs, set_seed


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
    def test_no_dynamo_plugin_when_compile_disabled(self, config):
        config.trainer.compile = False
        kwargs, _ = _run(config)
        assert kwargs["dynamo_plugin"] is None

    def test_dynamo_plugin_mirrors_compile_settings(self, config):
        config.trainer.compile = True
        config.trainer.compile_mode = "max-autotune"
        config.trainer.compile_fullgraph = False
        config.trainer.compile_dynamic = False

        kwargs, _ = _run(config)
        plugin = kwargs["dynamo_plugin"]

        assert plugin is not None
        assert plugin.backend.value.lower() == "inductor"
        assert plugin.mode == "max-autotune"
        assert plugin.fullgraph is False
        assert plugin.dynamic is False


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
    def test_tf32_flag_drives_matmul_precision(
        self, config, tf32, expected_precision
    ):
        config.trainer.tf32 = tf32
        _run(config)

        assert torch.get_float32_matmul_precision() == expected_precision
        assert torch.backends.cuda.matmul.allow_tf32 is tf32
        assert torch.backends.cudnn.allow_tf32 is tf32


class TestPrepareOnly:
    """`dataset.prepare_only` builds caches then exits, for warming a slow
    first-time build on CPU-only hardware."""

    @staticmethod
    def _main(config: PretrainConfig):
        """Run main() with the config fixed and everything external stubbed."""
        with (
            patch.object(run_pretrain, "load_and_parse", return_value=config),
            patch.object(run_pretrain, "setup_runtime") as setup_runtime,
            patch.object(run_pretrain, "setup_tracking") as setup_tracking,
            patch.object(run_pretrain, "prepare_sources") as prepare_sources,
            patch.object(run_pretrain, "Trainer") as trainer_cls,
            patch.object(run_pretrain.sys, "argv", ["run_pretrain.py", "cfg.yaml"]),
        ):
            setup_runtime.return_value = MagicMock()
            prepare_sources.return_value = []
            run_pretrain.main()
        return prepare_sources, trainer_cls, setup_tracking

    def test_prepares_sources_then_exits_without_training(self, config):
        config.dataset.prepare_only = True

        prepare_sources, trainer_cls, _ = self._main(config)

        prepare_sources.assert_called_once_with(config.dataset)
        trainer_cls.assert_not_called()

    def test_skips_model_build_so_no_gpu_is_needed(self, config):
        config.dataset.prepare_only = True

        with patch.object(run_pretrain, "get_amplify_masked_lm") as build_model:
            self._main(config)

        build_model.assert_not_called()

    def test_skips_tracking_so_no_empty_run_is_logged(self, config):
        config.dataset.prepare_only = True

        _, _, setup_tracking = self._main(config)

        setup_tracking.assert_not_called()

    def test_disabled_by_default(self, config):
        assert config.dataset.prepare_only is False
