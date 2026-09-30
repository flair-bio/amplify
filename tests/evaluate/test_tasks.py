"""Contract tests covering every registered TaskHandler uniformly.

New task types are added to ``modules/evaluate/src/tasks/`` (see the
"Adding a task type" section of ``modules/evaluate/README.md``); this file
parametrizes over the live registry so a new handler is exercised here
automatically, without a bespoke test module.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from modules.evaluate.src.tasks import TaskHandler, available_tasks, get_task


@pytest.fixture(params=available_tasks())
def task(request) -> TaskHandler:
    return get_task(request.param)


def test_resolve_num_labels(task: TaskHandler):
    num_labels = task.resolve_num_labels(num_labels=4, dataset_name="toy")
    assert isinstance(num_labels, int) and num_labels >= 1


def test_collate_labels_shape(task: TaskHandler):
    batch_size, seq_len = 3, 5
    if task.pairwise:
        labels = [
            {"pairs": [[0, 1]], "res_start": 0, "res_end": seq_len},
            {"pairs": [], "res_start": 0, "res_end": seq_len},
            {"pairs": [[1, 3], [0, 2]], "res_start": 0, "res_end": seq_len},
        ]
    elif task.name == "pseudo_perplexity":
        labels = [[-100, 1, -100, -100, -100] for _ in range(batch_size)]
    elif task.is_ragged:
        labels = [[0] * seq_len for _ in range(batch_size)]
    else:
        labels = [0] * batch_size

    collated = task.collate_labels(labels, batch_size, seq_len)

    assert isinstance(collated, torch.Tensor)
    assert collated.dtype == task.label_dtype
    if task.pairwise:
        assert collated.shape == (batch_size, seq_len, seq_len)
    elif task.name == "pseudo_perplexity":
        assert collated.shape == (batch_size, seq_len)
    elif task.is_ragged:
        assert collated.shape == (batch_size, seq_len)
    else:
        assert collated.shape == (batch_size,)


def test_extract_predictions_reduces_class_dimension(task: TaskHandler):
    if task.pairwise:
        # Per-pair binary score, not a per-class logit: shape is unchanged.
        logits = torch.randn(2, 4, 4)
        preds = task.extract_predictions(logits)
        assert preds.shape == logits.shape
        return
    if task.name == "pseudo_perplexity":
        logits = torch.randn(2, 4, 3)
        labels = torch.full((2, 4), -100, dtype=torch.long)
        labels[0, 1] = 1
        labels[1, 2] = 2
        preds = task.extract_predictions(logits, labels=labels)
        assert preds.shape == (2,)
        return
    # sequence_regression's head emits a single scalar logit (num_labels=1);
    # every other task type emits per-class logits.
    num_classes = 1 if task.name == "sequence_regression" else 3
    logits = (
        torch.randn(2, 4, num_classes) if task.is_ragged else torch.randn(2, num_classes)
    )
    preds = task.extract_predictions(logits)
    assert preds.shape == logits.shape[:-1]


def test_split_batch_rows_is_row_aligned(task: TaskHandler):
    if task.pairwise:
        preds = torch.rand(2, 3, 3)
        labels = torch.full((2, 3, 3), -100, dtype=torch.long)
        labels[0, 0, 1] = 1
        labels[0, 1, 0] = 1
        labels[1, 0, 2] = 0
        labels[1, 2, 0] = 0
    elif task.name == "pseudo_perplexity":
        preds = torch.tensor([-1.5, -2.0])
        labels = torch.full((2, 3), -100, dtype=torch.long)
        labels[0, 1] = 1
        labels[1, 2] = 2
    elif task.is_ragged:
        preds = torch.tensor([[1, 2, -100], [3, 4, 5]])
        labels = torch.tensor([[1, 2, -100], [3, 4, 5]])
    else:
        preds = torch.tensor([1, 2])
        labels = torch.tensor([1, 2])

    rows = task.split_batch_rows(preds, labels)

    assert len(rows) == 2
    for pred, label in rows:
        if task.name != "pseudo_perplexity":
            assert type(pred) is type(label)
        if task.pairwise:
            assert len(pred) == len(label)


def test_default_metrics_are_computable(task: TaskHandler):
    metric_names = task.default_metrics()
    assert metric_names

    if task.pairwise:
        predictions = [
            [(-1.0, -1), (0.9, 10), (0.2, 30)],
            [(-1.0, -1), (0.4, 8)],
        ]
        labels = [[6, 1, 0], [8, 0]]
        scores = task.compute_metrics(
            predictions=predictions,
            labels=labels,
            metric_names=metric_names,
            probabilities=None,
        )
        assert set(scores) == set(metric_names)
        return

    n = 6
    rng = np.random.default_rng(0)
    if task.name == "sequence_regression":
        predictions = rng.normal(size=n).tolist()
        labels = rng.normal(size=n).tolist()
    else:
        predictions = (rng.integers(0, 2, size=n)).tolist()
        labels = (rng.integers(0, 2, size=n)).tolist()
        labels[0], labels[1] = 0, 1  # ensure both classes are present for roc_auc

    probabilities = None
    if task.metrics_requiring_probabilities:
        probs = rng.random((n, 2))
        probabilities = (probs / probs.sum(axis=1, keepdims=True)).tolist()

    scores = task.compute_metrics(
        predictions=predictions,
        labels=labels,
        metric_names=metric_names,
        probabilities=probabilities,
    )
    assert set(scores) == set(metric_names)


def test_scramble_labels_preserves_shape(task: TaskHandler):
    rng = np.random.default_rng(0)
    if task.pairwise:
        labels = [[6, 1, 0], [8, 0], [5]]
    elif task.is_ragged:
        labels = [[0, 1], [2, 3, 4], [5]]
    else:
        labels = [0, 1, 2, 3]

    scrambled = task.scramble_labels(labels, rng)

    assert len(scrambled) == len(labels)
    if task.is_ragged:
        assert [len(seq) for seq in scrambled] == [len(seq) for seq in labels]


def test_unknown_metric_name_raises(task: TaskHandler):
    with pytest.raises(KeyError):
        task.validate_metric_names(["not_a_real_metric"])


def test_label_consuming_tasks_declare_a_dataset_contract(task: TaskHandler):
    if not task.requires_source_labels:
        pytest.skip(f"{task.name} derives its targets from the sequence.")

    assert task.accepted_target_types
    assert task.accepted_label_formats


def test_missing_dataset_contract_warns_at_class_definition():
    with pytest.warns(UserWarning, match="accepted_target_types"):

        class _MissingContractTask(TaskHandler):
            name = "missing_contract"
            auto_model_class = object
            metrics = {"accuracy": lambda values, labels, average: 1.0}
            default_metric_names = ("accuracy",)
