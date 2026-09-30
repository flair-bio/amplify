"""
Shared model-inference primitive: run a model over a dataloader and gather
per-example ids/predictions/labels[/probabilities/sequences]. Used by
:class:`~modules.evaluate.src.steps.predict.PredictStep` for the trained
model's own predictions, by
:class:`~modules.evaluate.src.utils.controls.ControlBuilder` for
model-based control conditions, and by
:class:`~modules.evaluate.src.steps.tune.TuneStep` for validation-time
scoring -- kept in its own module so none of those need to import from
one another just to share this loop.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
from accelerate import Accelerator

from modules.evaluate.src.dataset.collator import input_id_rows, unpack_packed
from modules.evaluate.src.tasks import TaskType, get_task


def pad_for_gather(
    tensor: torch.Tensor,
    accelerator: Accelerator,
    pad_index: int | float = 0,
    dims: tuple[int, ...] = (1,),
) -> torch.Tensor:
    """Pad variable-length sequence tensors before a cross-process gather."""
    if getattr(accelerator, "num_processes", 1) == 1 or tensor.ndim < 2:
        return tensor
    for dim in dims:
        tensor = accelerator.pad_across_processes(tensor, dim=dim, pad_index=pad_index)
    return tensor


def run_inference(
    model: Any,
    test_dataloader: Any,
    task_type: TaskType,
    accelerator: Accelerator,
    collect_probabilities: bool = False,
    max_batches: int | None = None,
    batch_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    tokenizer: Any | None = None,
) -> tuple[list[Any], list[Any], list[Any], list[Any] | None, list[str] | None]:
    """Run the model over the test set, returning per-example ids/predictions/labels.

    Each returned list is aligned 1:1 by position: ``ids[i]`` corresponds
    to ``predictions[i]`` and ``labels[i]``. For ``token_classification``,
    ``predictions[i]``/``labels[i]`` are themselves lists of per-token
    values (ignore-index ``-100`` positions already dropped); use
    :meth:`~modules.evaluate.src.tasks.base.TaskHandler.flatten_for_metrics`
    to get
    the single flat sequence metrics expect. ``ids`` come from the dataset's
    ``id`` column when present (passed through by :class:`EvaluationCollator`);
    otherwise they fall back to a positional index over the final example
    order.

    When ``collect_probabilities`` is set for ``sequence_classification``,
    also returns per-example per-class softmax probabilities for ``roc_auc``;
    returns ``None`` otherwise to avoid unnecessary communication and host
    memory use.

    ``max_batches``, when set, stops after that many batches.

    ``batch_transform``, when set, is applied to each batch (after popping
    ``id``/``labels``, before the forward pass) -- used to build control
    conditions such as scrambled-sequence inputs without duplicating the
    inference loop.

    ``tokenizer``, when set, is used to decode each (post-``batch_transform``)
    example's ``input_ids`` back into a sequence string, returned as the
    fifth element; this is how callers recover the *scrambled* sequence text
    for the ``scrambled_sequences`` control. ``None`` otherwise.
    """
    model.eval()
    task = get_task(task_type)

    all_ids: list[Any] = []
    all_predictions: list[Any] = []
    all_labels: list[Any] = []
    all_probabilities: list[Any] | None = [] if collect_probabilities else None
    all_sequences: list[str] | None = [] if tokenizer is not None else None
    has_ids = False

    # Batches are moved to CPU right after gathering (but not converted to
    # Python lists) so GPU memory doesn't grow with the number of batches
    # accumulated across the whole test set; `.tolist()`/decoding is still
    # deferred to a single pass below.
    pred_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    prob_batches: list[torch.Tensor] | None = [] if collect_probabilities else None
    input_ids_batches: list[torch.Tensor] | None = [] if tokenizer is not None else None
    id_batches: list[Any] = []

    with torch.inference_mode():
        for batch_idx, batch in enumerate(test_dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            if isinstance(batch, (tuple, list)):
                inputs_embeds, labels = batch
                batch = {
                    "inputs_embeds": inputs_embeds,
                    "labels": labels,
                }
            else:
                batch = dict(batch)
            ids = batch.pop("id", None)
            labels = batch.pop("labels")
            if batch_transform is not None:
                batch = batch_transform(batch)
            outputs = model(**batch)
            logits = outputs.logits
            if "cu_seqlens" in batch and logits.ndim == 3:
                cu_seqlens = batch["cu_seqlens"]
                num_sequences = batch["num_sequences"]
                logits = unpack_packed(logits, cu_seqlens, num_sequences)
                labels = unpack_packed(labels, cu_seqlens, num_sequences, -100)

            gather_dims = (1, 2) if task.pairwise else (1,)
            preds = pad_for_gather(
                task.extract_predictions(logits, labels=labels),
                accelerator,
                -100,
                gather_dims,
            )
            labels = pad_for_gather(labels, accelerator, -100, gather_dims)

            gather_targets: tuple[Any, ...] = (preds, labels)
            if prob_batches is not None:
                probabilities = pad_for_gather(logits.softmax(dim=-1), accelerator)
                gather_targets = (preds, labels, probabilities)
            if input_ids_batches is not None:
                pad_token_id = getattr(tokenizer, "pad_token_id", None) or 0
                input_ids = pad_for_gather(
                    input_id_rows(batch, pad_token_id), accelerator, pad_token_id
                )
                gather_targets = gather_targets + (input_ids,)

            # Gathers across processes (and drops padding introduced by
            # the distributed sampler) so metrics reflect the full test
            # set exactly once, regardless of world size.
            gathered = accelerator.gather_for_metrics(gather_targets)
            preds, labels, *rest = gathered
            pred_batches.append(preds.cpu())
            label_batches.append(labels.cpu())
            if prob_batches is not None:
                probs, *rest = rest
                prob_batches.append(probs.cpu())
            if input_ids_batches is not None:
                (input_ids_gathered,) = rest
                input_ids_batches.append(input_ids_gathered.cpu())
            if ids is not None:
                has_ids = True
                ids = accelerator.gather_for_metrics(ids, use_gather_object=True)
                # Object gathers do not remove sampler padding, unlike tensor
                # gathers. The gathered prediction tensor is authoritative.
                ids = ids[: preds.shape[0]]
            id_batches.append(ids)

    # Single host-sync pass over the whole split: every `.cpu()`/`.tolist()`
    # call (and sequence decoding) happens here, once the full split has
    # already been gathered, instead of once per batch inside the loop above.
    for batch_idx, (preds, labels) in enumerate(zip(pred_batches, label_batches)):
        if prob_batches is not None:
            assert all_probabilities is not None
            all_probabilities.extend(prob_batches[batch_idx].cpu().tolist())
        decoded = None
        if input_ids_batches is not None:
            assert (
                tokenizer is not None
            )  # input_ids_batches is only set when tokenizer is
            decoded = tokenizer.batch_decode(
                input_ids_batches[batch_idx].cpu().tolist(), skip_special_tokens=True
            )
        ids = id_batches[batch_idx]
        for i, (pred_row, label_row) in enumerate(task.split_batch_rows(preds, labels)):
            all_predictions.append(pred_row)
            all_labels.append(label_row)
            if ids is not None:
                all_ids.append(ids[i])
            if decoded is not None:
                assert all_sequences is not None
                all_sequences.append(decoded[i])

    if not has_ids:
        # No 'id' column on the dataset: fall back to a positional index
        # over the final (gathered) example order.
        all_ids = list(range(len(all_predictions)))

    return all_ids, all_predictions, all_labels, all_probabilities, all_sequences


def run_local_embedding_inference(
    model: Any,
    embeddings: torch.Tensor,
    ids: list[Any],
    labels: torch.Tensor,
    task_type: TaskType,
    device: torch.device,
    batch_size: int,
    collect_probabilities: bool = False,
) -> tuple[list[Any], list[Any], list[Any], list[Any] | None]:
    """Run a probe over a complete, locally replicated embedding split."""
    model.eval()
    task = get_task(task_type)
    all_predictions: list[Any] = []
    all_labels: list[Any] = []
    all_probabilities: list[Any] | None = [] if collect_probabilities else None

    with torch.inference_mode():
        for start in range(0, embeddings.shape[0], batch_size):
            end = start + batch_size
            logits = model(inputs_embeds=embeddings[start:end].to(device=device)).logits
            predictions = task.extract_predictions(logits, labels=labels[start:end])
            for prediction, label in task.split_batch_rows(
                predictions, labels[start:end]
            ):
                all_predictions.append(prediction)
                all_labels.append(label)
            if all_probabilities is not None:
                all_probabilities.extend(logits.softmax(dim=-1).cpu().tolist())

    return ids, all_predictions, all_labels, all_probabilities
