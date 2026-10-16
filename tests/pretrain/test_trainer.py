"""Basic unit tests for the pretrain Trainer."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from modules.pretrain.src.trainer import Trainer, TrainerConfig


@pytest.fixture
def accelerator():
    """A minimal Accelerator stub sufficient for Trainer's constructor/steps."""
    acc = MagicMock()
    acc.is_main_process = True
    acc.device = torch.device("cpu")
    acc.sync_gradients = True
    acc.gradient_accumulation_steps = 1
    acc.clip_grad_norm_.return_value = torch.tensor(1.0)
    return acc


@pytest.fixture
def trainer(accelerator, tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = MagicMock()
    config = TrainerConfig(output_dir=tmp_path / "checkpoints")

    return Trainer(
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader_provider=lambda epoch: [],
        config=config,
    )


class TestTrainerConfig:
    def test_num_epochs_must_be_positive(self):
        with pytest.raises(ValueError):
            TrainerConfig(num_epochs=0)


class TestTrainerInit:
    def test_initial_state_is_zeroed(self, trainer, tmp_path):
        assert trainer.global_step == 0
        assert trainer.current_epoch == 0
        assert trainer.batches_completed_in_epoch == 0
        assert trainer.output_dir == tmp_path / "checkpoints"


class TestOptimizerStep:
    def test_optimizer_step_clips_grad_and_steps_scheduler_when_syncing(self, trainer):
        list(trainer.model.parameters())[0].grad = torch.ones(2, 2)

        grad_norm = trainer.optimizer_step()

        trainer.accelerator.clip_grad_norm_.assert_called_once()
        trainer.scheduler.step.assert_called_once()
        assert grad_norm == torch.tensor(1.0)

    def test_optimizer_step_skips_clip_and_scheduler_without_sync(self, trainer):
        trainer.accelerator.sync_gradients = False

        grad_norm = trainer.optimizer_step()

        trainer.accelerator.clip_grad_norm_.assert_not_called()
        assert grad_norm is None
        trainer.scheduler.step.assert_not_called()


class TestEvaluate:
    def test_evaluate_is_a_noop_without_val_dataloader(self, trainer):
        assert trainer.val_dataloader is None

        trainer.evaluate()  # Should return early without raising.


class TestEpochMixtureBarrier:
    """The per-epoch dataloader rebuild must be serialized across ranks,
    mirroring the guard already used around `prepare_sources`.

    Without this, N ranks racing to build a first-time epoch mixture (e.g. a
    cold `valid_counts` scan) redundantly in parallel can desync badly enough
    to trip the NCCL watchdog on whichever rank finishes first -- reproduced
    on job 10847700 (bfd, first-ever threshold, 4 ranks).
    """

    def test_provider_is_called_inside_local_main_process_first(
        self, accelerator, tmp_path
    ):
        calls = []
        accelerator.local_main_process_first.return_value.__enter__ = MagicMock(
            side_effect=lambda: calls.append("enter")
        )
        accelerator.local_main_process_first.return_value.__exit__ = MagicMock(
            side_effect=lambda *a: calls.append("exit")
        )

        def provider(epoch):
            calls.append("provider")
            return []

        trainer = Trainer(
            accelerator=accelerator,
            model=torch.nn.Linear(2, 2),
            optimizer=MagicMock(),
            scheduler=MagicMock(),
            train_dataloader_provider=provider,
            config=TrainerConfig(num_epochs=1, output_dir=tmp_path / "ckpt"),
        )
        trainer._train_one_epoch = MagicMock()
        trainer.evaluate = MagicMock()
        trainer.save_checkpoint = MagicMock()

        trainer.train()

        assert calls == ["enter", "provider", "exit"], (
            "the provider must run strictly between the barrier's enter/exit, "
            "not before or after it"
        )


class TestGradientAccumulation:
    """`accelerator.accumulate` is driven by the Accelerator's own setting, so a
    config that disagrees with it would silently train at the wrong effective
    batch size. These pin the wiring and the mismatch guard."""

    def test_defaults_to_one(self):
        assert TrainerConfig().gradient_accumulation_steps == 1

    def test_must_be_positive(self):
        with pytest.raises(ValueError):
            TrainerConfig(gradient_accumulation_steps=0)

    def test_rejects_accelerator_mismatch(self, accelerator, tmp_path):
        accelerator.gradient_accumulation_steps = 1
        with pytest.raises(ValueError, match="gradient_accumulation_steps"):
            Trainer(
                accelerator=accelerator,
                model=torch.nn.Linear(2, 2),
                optimizer=MagicMock(),
                scheduler=MagicMock(),
                train_dataloader_provider=lambda epoch: [],
                config=TrainerConfig(
                    output_dir=tmp_path / "ckpt", gradient_accumulation_steps=4
                ),
            )

    def test_accepts_matching_accelerator(self, accelerator, tmp_path):
        accelerator.gradient_accumulation_steps = 4
        trainer = Trainer(
            accelerator=accelerator,
            model=torch.nn.Linear(2, 2),
            optimizer=MagicMock(),
            scheduler=MagicMock(),
            train_dataloader_provider=lambda epoch: [],
            config=TrainerConfig(
                output_dir=tmp_path / "ckpt", gradient_accumulation_steps=4
            ),
        )
        assert trainer.gradient_accumulation_steps == 4

    @pytest.mark.parametrize(
        "batches,accum,expected_optimizer_steps",
        [(8, 1, 8), (8, 2, 4), (8, 4, 2), (7, 2, 4), (1, 4, 1)],
    )
    def test_steps_per_epoch_is_in_optimizer_steps(
        self, accelerator, tmp_path, batches, accum, expected_optimizer_steps
    ):
        """`metric_logger.num_steps` counts optimizer steps, so the per-epoch
        denominator must too or `train/epoch` is off by the accumulation factor."""
        accelerator.gradient_accumulation_steps = accum
        trainer = Trainer(
            accelerator=accelerator,
            model=torch.nn.Linear(2, 2),
            optimizer=MagicMock(),
            scheduler=MagicMock(),
            train_dataloader_provider=lambda epoch: [None] * batches,
            config=TrainerConfig(
                num_epochs=1,
                output_dir=tmp_path / "ckpt",
                gradient_accumulation_steps=accum,
            ),
        )
        trainer._train_one_epoch = MagicMock()
        trainer.evaluate = MagicMock()
        trainer.save_checkpoint = MagicMock()
        trainer.train()

        assert trainer.metric_logger.steps_per_epoch == expected_optimizer_steps


class TestResumeGuardsAgainstShorterEpoch:
    """A config change can shorten the epoch, putting the saved
    `batches_completed` past its end."""

    def _make_trainer(self, accelerator, tmp_path, batches):
        trainer = Trainer(
            accelerator=accelerator,
            model=torch.nn.Linear(2, 2),
            optimizer=MagicMock(),
            scheduler=MagicMock(),
            train_dataloader_provider=lambda epoch: [None] * batches,
            config=TrainerConfig(num_epochs=1, output_dir=tmp_path / "ckpt"),
        )
        trainer._train_one_epoch = MagicMock()
        trainer.evaluate = MagicMock()
        trainer.save_checkpoint = MagicMock()
        return trainer

    def test_skips_normally_when_saved_position_is_in_range(
        self, accelerator, tmp_path
    ):
        trainer = self._make_trainer(accelerator, tmp_path, batches=10)
        trainer.batches_completed_in_epoch = 4

        trainer.train()

        accelerator.skip_first_batches.assert_called_once()
        assert accelerator.skip_first_batches.call_args[0][1] == 4

    @pytest.mark.parametrize("saved", [10, 25])
    def test_restarts_epoch_when_saved_position_is_out_of_range(
        self, accelerator, tmp_path, caplog, saved
    ):
        """Skipping every batch would silently train on nothing."""
        trainer = self._make_trainer(accelerator, tmp_path, batches=10)
        trainer.batches_completed_in_epoch = saved

        with caplog.at_level(logging.WARNING):
            trainer.train()

        accelerator.skip_first_batches.assert_not_called()
        assert "Restarting this epoch" in caplog.text
        # The full dataloader reaches the epoch loop, not an emptied one.
        assert trainer._train_one_epoch.call_args[0][0] == [None] * 10


class TestCompileConfig:
    def test_compile_defaults_to_off(self):
        assert TrainerConfig().compile is False
        assert TrainerConfig().compile_dynamic is None

    def test_rejects_unknown_compile_mode(self):
        with pytest.raises(ValueError):
            TrainerConfig(compile_mode="turbo")

    @pytest.mark.parametrize("mode", ["default", "max-autotune"])
    def test_accepts_supported_modes(self, mode):
        assert TrainerConfig(compile_mode=mode).compile_mode == mode


class TestCompiledCheckpointKeys:
    """`torch.compile` wraps the model so every state_dict key gains an
    `_orig_mod.` prefix. If that leaks into the HF checkpoint, the weights no
    longer load into a plain AMPLIFYForMaskedLM."""

    def _save(self, trainer, accelerator, state_dict, tmp_path):
        accelerator.get_state_dict.return_value = state_dict
        trainer.tokenizer = None
        trainer.save_hf_checkpoint(path=tmp_path / "hf")
        return accelerator.unwrap_model.return_value.save_pretrained.call_args.kwargs

    def test_orig_mod_prefix_is_stripped(self, trainer, accelerator, tmp_path):
        kwargs = self._save(
            trainer,
            accelerator,
            {"_orig_mod.encoder.weight": torch.zeros(1), "_orig_mod.bias": torch.zeros(1)},
            tmp_path,
        )
        assert set(kwargs["state_dict"]) == {"encoder.weight", "bias"}

    def test_uncompiled_keys_are_untouched(self, trainer, accelerator, tmp_path):
        kwargs = self._save(
            trainer, accelerator, {"encoder.weight": torch.zeros(1)}, tmp_path
        )
        assert set(kwargs["state_dict"]) == {"encoder.weight"}

    def test_unwrap_asks_accelerate_to_drop_the_compile_wrapper(
        self, trainer, accelerator, tmp_path
    ):
        self._save(trainer, accelerator, {"w": torch.zeros(1)}, tmp_path)
        assert accelerator.unwrap_model.call_args.kwargs["keep_torch_compile"] is False


class TestRotateCheckpoints:
    @staticmethod
    def _make(base_dir: Path, names: list[str]) -> None:
        for name in names:
            (base_dir / name).mkdir(parents=True)

    def _remaining(self, base_dir: Path) -> set[str]:
        return {p.name for p in base_dir.iterdir() if p.is_dir()}

    def test_keeps_the_most_recent_by_step_number(self, trainer, tmp_path):
        base = tmp_path / "ckpts"
        # Ordered numerically, 9000 is older than 10000 but sorts later as text.
        self._make(base, [f"checkpoint_epoch_1_step_{s}" for s in (9000, 10000, 11000)])

        trainer._rotate_checkpoints(base, "checkpoint_epoch_*_step_*", 2)

        assert self._remaining(base) == {
            "checkpoint_epoch_1_step_10000",
            "checkpoint_epoch_1_step_11000",
        }

    def test_supports_the_hf_checkpoint_naming_scheme(self, trainer, tmp_path):
        base = tmp_path / "hf"
        self._make(base, [f"checkpoint_{s}" for s in (100, 200, 300)])

        trainer._rotate_checkpoints(base, "checkpoint_*", 1)

        assert self._remaining(base) == {"checkpoint_300"}

    def test_no_limit_keeps_everything(self, trainer, tmp_path):
        base = tmp_path / "ckpts"
        names = [f"checkpoint_epoch_1_step_{s}" for s in (1, 2, 3)]
        self._make(base, names)

        trainer._rotate_checkpoints(base, "checkpoint_epoch_*_step_*", None)

        assert self._remaining(base) == set(names)

    def test_fewer_checkpoints_than_limit_is_a_no_op(self, trainer, tmp_path):
        base = tmp_path / "ckpts"
        names = [f"checkpoint_epoch_1_step_{s}" for s in (1, 2)]
        self._make(base, names)

        trainer._rotate_checkpoints(base, "checkpoint_epoch_*_step_*", 5)

        assert self._remaining(base) == set(names)

    def test_missing_directory_is_a_no_op(self, trainer, tmp_path):
        trainer._rotate_checkpoints(tmp_path / "absent", "checkpoint_*", 1)

    def test_non_main_process_never_deletes(self, trainer, accelerator, tmp_path):
        trainer.is_main = False
        base = tmp_path / "ckpts"
        names = [f"checkpoint_epoch_1_step_{s}" for s in (1, 2, 3)]
        self._make(base, names)

        trainer._rotate_checkpoints(base, "checkpoint_epoch_*_step_*", 1)

        assert self._remaining(base) == set(names)


class TestLoadCheckpoint:
    def test_restores_epoch_and_step_counters(self, trainer, tmp_path):
        path = tmp_path / "checkpoint_epoch_4_step_800"
        path.mkdir()
        torch.save(
            {"epoch": 4, "global_step": 800, "batches_completed": 37},
            path / "trainer_state.pt",
        )

        trainer.load_checkpoint(path)

        assert trainer.current_epoch == 4
        assert trainer.global_step == 800
        assert trainer.batches_completed_in_epoch == 37
        trainer.accelerator.load_state.assert_called_once_with(str(path))

    def test_missing_metadata_leaves_counters_at_zero(self, trainer, tmp_path):
        path = tmp_path / "checkpoint_epoch_1_step_10"
        path.mkdir()

        trainer.load_checkpoint(path)

        assert trainer.current_epoch == 0
        assert trainer.global_step == 0

    def test_missing_directory_raises(self, trainer, tmp_path):
        with pytest.raises(FileNotFoundError):
            trainer.load_checkpoint(tmp_path / "absent")


class TestDataloaderStall:
    def test_attributes_a_slow_fetch_to_its_own_step(self, accelerator, tmp_path):
        slow_ms = 20.0

        def dataloader():
            for delay_ms in [0.0, slow_ms, 0.0]:
                time.sleep(delay_ms / 1000)
                yield {}

        trainer = Trainer(
            accelerator=accelerator,
            model=torch.nn.Linear(2, 2),
            optimizer=MagicMock(),
            scheduler=MagicMock(),
            train_dataloader_provider=lambda epoch: [],
            config=TrainerConfig(output_dir=tmp_path / "ckpt"),
            metric_logger=MagicMock(),
        )
        trainer.training_step = MagicMock(return_value=(torch.tensor(0.0), None))
        trainer._train_one_epoch(
            dataloader(), epoch=0, total_batches=None, pbar=MagicMock(disable=True)
        )

        calls = trainer.metric_logger.add_dataloader_stall.call_args_list
        stalls = [c.args[0] for c in calls]
        assert len(stalls) == 3
        assert stalls[1] >= slow_ms
        assert stalls[0] < slow_ms and stalls[2] < slow_ms
