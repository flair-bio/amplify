"""Unit tests for TrainingLogger, which computes every logged train/eval metric."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from accelerate.utils import DistributedType

from modules.pretrain.src.metric.logger import TrainingLogger

VOCAB = 5


@pytest.fixture
def accelerator():
    """Single-process stand-in: `reduce` is the identity and `log` is recorded."""
    acc = MagicMock()
    acc.device = torch.device("cpu")
    acc.num_processes = 1
    acc.process_index = 0
    acc.is_main_process = True
    acc.distributed_type = DistributedType.NO
    acc.reduce.side_effect = lambda tensor, reduction: tensor
    return acc


def make_batch(labels, predictions):
    """Packed batch whose logits argmax to `predictions` at every position."""
    labels = torch.tensor([labels])
    logits = torch.nn.functional.one_hot(torch.tensor([predictions]), VOCAB).float()
    batch = {
        "labels": labels,
        "num_sequences": torch.tensor(1),
        "num_tokens": torch.tensor(labels.numel()),
    }
    return batch, SimpleNamespace(logits=logits)


def flush(logger, accelerator, tmp_path):
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.5)
    logger.log(
        grad_norm=torch.tensor(2.0),
        weight_sq_sum=torch.tensor(9.0),
        optimizer=optimizer,
        accelerator=accelerator,
        json_path=tmp_path / "metrics.jsonl",
    )
    return accelerator.log.call_args.args[0]


class TestTrainMetrics:
    def test_loss_and_accuracy_are_weighted_by_masked_tokens(
        self, accelerator, tmp_path
    ):
        logger = TrainingLogger()
        # Batch 1: 1 masked token, predicted correctly, loss 1.0.
        batch, out = make_batch(labels=[-100, 2, -100], predictions=[0, 2, 0])
        logger.add_train_batch(batch, out, torch.tensor(1.0))
        # Batch 2: 3 masked tokens, 1 correct, loss 3.0.
        batch, out = make_batch(labels=[1, 2, 3, -100], predictions=[1, 0, 0, 3])
        logger.add_train_batch(batch, out, torch.tensor(3.0))

        metrics = flush(logger, accelerator, tmp_path)

        # Token-weighted (1*1 + 3*3) / 4, not the mean of batch losses (2.0).
        assert metrics["train/loss"] == pytest.approx(2.5)
        assert metrics["train/perplexity"] == pytest.approx(math.exp(2.5))
        # Unmasked positions never count, even when predicted "correctly".
        assert metrics["train/accuracy"] == pytest.approx(2 / 4)
        assert metrics["train/tokens"] == 7
        assert metrics["train/masked_tokens"] == 4
        assert metrics["train/samples"] == 2
        assert metrics["train/learning_rate"] == 0.5
        assert metrics["train/grad_norm"] == 2.0
        assert metrics["train/weight_norm"] == pytest.approx(3.0)

    def test_window_metrics_reset_while_totals_accumulate(self, accelerator, tmp_path):
        logger = TrainingLogger()
        batch, out = make_batch(labels=[1, 2], predictions=[1, 2])
        logger.add_train_batch(batch, out, torch.tensor(1.0))
        flush(logger, accelerator, tmp_path)

        batch, out = make_batch(labels=[1, 2, 3], predictions=[0, 0, 0])
        logger.add_train_batch(batch, out, torch.tensor(4.0))
        metrics = flush(logger, accelerator, tmp_path)

        assert metrics["train/loss"] == pytest.approx(4.0)
        assert metrics["train/accuracy"] == 0.0
        assert metrics["train/tokens"] == 5
        assert metrics["train/samples"] == 2

    def test_counters_survive_a_checkpoint_roundtrip(self, accelerator, tmp_path):
        logger = TrainingLogger()
        batch, out = make_batch(labels=[1, 2], predictions=[1, 2])
        logger.add_train_batch(batch, out, torch.tensor(1.0))
        logger.num_steps = 7
        flush(logger, accelerator, tmp_path)

        resumed = TrainingLogger()
        resumed.load_state_dict(logger.state_dict())

        assert resumed.state_dict() == logger.state_dict()


class TestEvalMetrics:
    def test_eval_only_flush_reports_eval_without_fake_train_metrics(
        self, accelerator, tmp_path
    ):
        logger = TrainingLogger()
        batch, out = make_batch(labels=[1, 2, -100], predictions=[1, 0, 0])
        logger.add_eval_batch("val", batch, out, torch.tensor(2.0))

        metrics = flush(logger, accelerator, tmp_path)

        assert metrics["eval/val/loss"] == pytest.approx(2.0)
        assert metrics["eval/val/accuracy"] == pytest.approx(0.5)
        assert metrics["eval/val/perplexity"] == pytest.approx(math.exp(2.0))
        assert "train/loss" not in metrics
        assert "train/accuracy" not in metrics


class TestDataloaderMetrics:
    def test_stall_statistics_and_reset(self, accelerator, tmp_path):
        logger = TrainingLogger()
        for stall_ms in [1.0, 2.0, 3.0, 4.0, 100.0]:
            logger.add_dataloader_stall(stall_ms)

        metrics = flush(logger, accelerator, tmp_path)

        assert metrics["dataloader/iter_p50_ms"] == pytest.approx(3.0)
        assert metrics["dataloader/iter_max_ms"] == pytest.approx(100.0)
        assert metrics["dataloader/slowest_rank"] == 0
        assert 0 < metrics["dataloader/iter_step_pct"]

        # The next window starts empty, so no stale stalls are reported.
        metrics = flush(logger, accelerator, tmp_path)
        assert not any(k.startswith("dataloader/") for k in metrics)


def test_every_flush_appends_one_json_line(accelerator, tmp_path):
    logger = TrainingLogger()
    batch, out = make_batch(labels=[1], predictions=[1])
    logger.add_train_batch(batch, out, torch.tensor(1.0))
    first = flush(logger, accelerator, tmp_path)
    flush(logger, accelerator, tmp_path)

    lines = (tmp_path / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == pytest.approx(first)
