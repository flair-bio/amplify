"""
Shared model-inference primitive: run a model over a dataloader and gather
per-example ids/predictions/labels[/probabilities/sequences]. Used by
:class:`~modules.evaluate.src.steps.predict.PredictStep` for the trained
model's own predictions, by
:class:`~modules.evaluate.src.steps.controls.ControlBuilder` for
model-based control conditions, and by
:class:`~modules.evaluate.src.steps.tune.TuneStep` for validation-time
scoring -- kept in its own module so none of those need to import from
one another just to share this loop.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable, Iterator
from time import monotonic
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from modules.evaluate.src.dataset.collator import input_id_rows, unpack_packed
from modules.evaluate.src.tasks import TaskType, get_task

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from modules.evaluate.src.contracts.artifacts import PreparedArtifacts
    from modules.evaluate.src.dataset.contact import ContactChain, ContactDataset


def run_contact_inference(
    dataset: ContactDataset,
    predictor: Callable[[ContactChain], np.ndarray | torch.Tensor],
    *,
    split: str = "test",
) -> Iterator[tuple[ContactChain, np.ndarray | None]]:
    """Run each exact full-chain sequence once and share scores across units."""
    grouped: dict[str, list[ContactChain]] = defaultdict(list)
    for chain in dataset.splits[split]:
        grouped[chain.alignment.sequence].append(chain)
    for shared in grouped.values():
        needed = next(
            (
                chain
                for chain in shared
                if any(unit.labels_available for unit in chain.units)
            ),
            None,
        )
        scores = None
        if needed is not None and needed.alignment.retained_positions:
            with torch.inference_mode():
                predicted = predictor(needed)
            if isinstance(predicted, torch.Tensor):
                predicted = predicted.detach().float().cpu().numpy()
            scores = np.asarray(predicted)
            del predicted
            length = len(needed.alignment.retained_positions)
            if scores.shape != (length, length):
                raise ValueError(
                    "Predictor must return a full retained-chain score matrix"
                )
        for chain in shared:
            yield chain, scores
        del scores


def pad_for_gather(
    tensor: torch.Tensor,
    accelerator: Accelerator,
    pad_index: float = 0,
    dims: tuple[int, ...] = (1,),
) -> torch.Tensor:
    """Pad variable-length sequence tensors before a cross-process gather."""
    if getattr(accelerator, "num_processes", 1) == 1 or tensor.ndim < 2:
        return tensor
    for dim in dims:
        tensor = accelerator.pad_across_processes(tensor, dim=dim, pad_index=pad_index)
    return tensor


def gather_training_labels(
    prepared: PreparedArtifacts, accelerator: Accelerator
) -> torch.Tensor:
    """Gather training targets for a model-independent prediction baseline."""
    if prepared.train_dataloader is None:
        raise ValueError("A train_dataloader is required for a train baseline.")

    from modules.evaluate.src.dataset.embeddings import find_embedding_dataset

    embedding_dataset = find_embedding_dataset(prepared.train_dataloader)
    if embedding_dataset is not None:
        return torch.as_tensor(embedding_dataset.labels).reshape(-1)

    label_rows: list[torch.Tensor] = []
    for batch in prepared.train_dataloader:
        labels = accelerator.gather_for_metrics(
            pad_for_gather(batch["labels"], accelerator, -100)
        )
        label_rows.append(labels.detach().cpu().reshape(-1))
    labels = torch.cat(label_rows)
    labels = labels[labels >= 0]
    if labels.numel() == 0:
        raise ValueError("The train split contains no valid labels.")
    return labels


def unique_id_indices(ids: list[Any], seen: set[Any]) -> list[int]:
    """Return first-occurrence row indices, updating the split's seen IDs."""
    indices = []
    for index, identifier in enumerate(ids):
        if identifier not in seen:
            seen.add(identifier)
            indices.append(index)
    return indices


def run_inference(
    model: Any,
    test_dataloader: Any,
    task_type: TaskType,
    accelerator: Accelerator,
    collect_probabilities: bool = False,
    max_batches: int | None = None,
    batch_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    tokenizer: Any | None = None,
    progress_label: str | None = None,
    progress_total_batches: int | None = None,
    progress_total_rows: int | None = None,
    progress_unit: str = "examples",
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

    ``progress_label`` enables throttled INFO batch logs for prediction callers;
    shared validation callers remain silent by default.
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
    residue_length_batches: list[torch.Tensor] | None = None
    id_batches: list[Any] = []
    processed_rows = 0

    started = last_log = monotonic()
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
            residue_lengths = batch.pop("residue_lengths", None)
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
                probabilities = pad_for_gather(
                    task.predicted_probabilities(logits), accelerator
                )
                gather_targets = (preds, labels, probabilities)
            if input_ids_batches is not None:
                pad_token_id = getattr(tokenizer, "pad_token_id", None) or 0
                input_ids = pad_for_gather(
                    input_id_rows(batch, pad_token_id), accelerator, pad_token_id
                )
                gather_targets = gather_targets + (input_ids,)
            if residue_lengths is not None:
                gather_targets = gather_targets + (residue_lengths,)

            # Gathers across processes (and drops padding introduced by
            # the distributed sampler) so metrics reflect the full test
            # set exactly once, regardless of world size.
            gathered = accelerator.gather_for_metrics(gather_targets)
            preds, labels, *rest = gathered
            processed_rows += preds.shape[0]
            rest_index = 0
            pred_batches.append(preds.cpu())
            label_batches.append(labels.cpu())
            if prob_batches is not None:
                probs = rest[rest_index]
                rest_index += 1
                prob_batches.append(probs.cpu())
            if input_ids_batches is not None:
                input_ids_gathered = rest[rest_index]
                rest_index += 1
                input_ids_batches.append(input_ids_gathered.cpu())
            if residue_lengths is not None:
                if residue_length_batches is None:
                    residue_length_batches = []
                residue_length_batches.append(rest[rest_index].cpu())
            if ids is not None:
                has_ids = True
                ids = accelerator.gather_for_metrics(ids, use_gather_object=True)
                # Object gathers do not remove sampler padding, unlike tensor
                # gathers. The gathered prediction tensor is authoritative.
                ids = ids[: preds.shape[0]]
            id_batches.append(ids)
            if progress_label and accelerator.is_main_process:
                now = monotonic()
                if now - last_log >= 10:
                    elapsed = now - started
                    if progress_total_batches is None or progress_total_rows is None:
                        logger.info(
                            "%s: %d batches completed (%.1fs elapsed)",
                            progress_label,
                            batch_idx + 1,
                            elapsed,
                        )
                    else:
                        rows_per_second = processed_rows / max(elapsed, 1e-9)
                        remaining_rows = max(0, progress_total_rows - processed_rows)
                        eta = remaining_rows / max(rows_per_second, 1e-9)
                        logger.info(
                            "%s: batch %d/%d | %d/%d %s (%.1f%%) | "
                            "%.1f %s/s | elapsed=%.1fs | ETA=%.1fs",
                            progress_label,
                            batch_idx + 1,
                            progress_total_batches,
                            processed_rows,
                            progress_total_rows,
                            progress_unit,
                            100 * processed_rows / max(progress_total_rows, 1),
                            rows_per_second,
                            progress_unit,
                            elapsed,
                            eta,
                        )
                    last_log = now

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
        batch_metadata = (
            {"residue_lengths": residue_length_batches[batch_idx]}
            if residue_length_batches is not None
            else None
        )
        rows = task.split_batch_rows_with_context(
            preds, labels, batch_metadata=batch_metadata
        )
        for i, (pred_row, label_row) in enumerate(rows):
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
    elif getattr(accelerator, "num_processes", 1) > 1:
        indices = unique_id_indices(all_ids, set())
        all_ids = [all_ids[index] for index in indices]
        all_predictions = [all_predictions[index] for index in indices]
        all_labels = [all_labels[index] for index in indices]
        if all_probabilities is not None:
            all_probabilities = [all_probabilities[index] for index in indices]
        if all_sequences is not None:
            all_sequences = [all_sequences[index] for index in indices]

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
    progress_label: str | None = None,
) -> tuple[list[Any], list[Any], list[Any], list[Any] | None]:
    """Run a probe over a complete, locally replicated embedding split."""
    model.eval()
    task = get_task(task_type)
    all_predictions: list[Any] = []
    all_labels: list[Any] = []
    all_probabilities: list[Any] | None = [] if collect_probabilities else None

    started = last_log = monotonic()
    with torch.inference_mode():
        for batch_idx, start in enumerate(range(0, embeddings.shape[0], batch_size), 1):
            end = start + batch_size
            logits = model(inputs_embeds=embeddings[start:end].to(device=device)).logits
            predictions = task.extract_predictions(logits, labels=labels[start:end])
            for prediction, label in task.split_batch_rows(
                predictions, labels[start:end]
            ):
                all_predictions.append(prediction)
                all_labels.append(label)
            if all_probabilities is not None:
                all_probabilities.extend(
                    task.predicted_probabilities(logits).cpu().tolist()
                )
            if progress_label:
                now = monotonic()
                if now - last_log >= 10:
                    logger.info(
                        "%s: %d batches completed (%.1fs elapsed)",
                        progress_label,
                        batch_idx,
                        now - started,
                    )
                    last_log = now

    return ids, all_predictions, all_labels, all_probabilities


def run_local_head_inference(
    head_model: Any,
    dataloader: DataLoader,
    task_type: str,
    collect_probabilities: bool,
    accelerator: Accelerator,
    gather_across_processes: bool = True,
    progress_label: str | None = None,
) -> tuple[list[Any], list[Any], list[Any] | None]:
    """Run *head_model* over a distributed shard of gathered embeddings.

    The embeddings are already present on every process, but the head
    should still run only on each process's sampler shard. Gather the
    resulting predictions back so callers receive one complete control
    result rather than one duplicate result per GPU.

    This dataloader is deliberately not accelerator-prepared, so
    ``gather_for_metrics`` cannot strip the DistributedSampler's padding;
    duplicates are removed by ``id`` instead of by positional truncation,
    which would depend on the sampler's interleaving order.

    ``gather_across_processes=False`` (ragged/token-level tasks) skips
    the dedup-by-id gather entirely: many rows there legitimately share
    the same sequence id (one per valid token), so id-based dedup would
    collapse them. *dataloader* must then already be the full,
    unsharded dataset (identical on every rank).
    """
    head_model.eval()
    task = get_task(task_type)
    expected_examples = len(dataloader.dataset)
    seen: dict[Any, int] = {}
    all_ids: list[Any] = []
    all_predictions: list[Any] = []
    all_probabilities: list[Any] | None = [] if collect_probabilities else None
    started = last_log = monotonic()
    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader, 1):
            batch = dict(batch)
            ids = batch.pop("id", None)
            labels = batch.pop("labels", None)
            if torch.is_tensor(labels):
                labels = labels.to(accelerator.device)
            # This dataloader is deliberately not accelerator-prepared (see
            # docstring), so batches aren't auto-moved to the device the
            # way accelerator.prepare(dataloader) would -- move them here
            # so the head runs on-device and produces dense CUDA
            # predictions/probabilities for the gather_for_metrics() below.
            batch = {
                key: (value.to(accelerator.device) if torch.is_tensor(value) else value)
                for key, value in batch.items()
            }
            logits = head_model(**batch).logits
            preds = task.extract_predictions(logits, labels=labels)
            probabilities_batch = (
                task.predicted_probabilities(logits)
                if all_probabilities is not None
                else None
            )
            if gather_across_processes:
                gather_targets: tuple[Any, ...] = (
                    pad_for_gather(preds, accelerator, -100),
                )
                if probabilities_batch is not None:
                    gather_targets = (
                        pad_for_gather(preds, accelerator, -100),
                        pad_for_gather(probabilities_batch, accelerator),
                    )
                gathered = accelerator.gather_for_metrics(gather_targets)
                preds, *rest = gathered
                batch_probabilities = (
                    rest[0].cpu().tolist() if probabilities_batch is not None else None
                )
            else:
                batch_probabilities = (
                    probabilities_batch.cpu().tolist()
                    if probabilities_batch is not None
                    else None
                )
            batch_predictions = preds.cpu().tolist()
            if ids is None:
                batch_ids: list[Any] = list(
                    range(len(all_predictions), len(all_predictions) + len(preds))
                )
            elif gather_across_processes:
                batch_ids = list(
                    accelerator.gather_for_metrics(ids, use_gather_object=True)
                )
            else:
                batch_ids = list(ids)
            if gather_across_processes:
                for position, example_id in enumerate(batch_ids[: len(preds)]):
                    if example_id in seen:
                        continue
                    seen[example_id] = len(all_predictions)
                    all_ids.append(example_id)
                    all_predictions.append(batch_predictions[position])
                    if all_probabilities is not None:
                        assert batch_probabilities is not None
                        all_probabilities.append(batch_probabilities[position])
            else:
                # No sampler padding to dedup here (see docstring): keep
                # every row, including ones that share an id.
                all_ids.extend(batch_ids)
                all_predictions.extend(batch_predictions)
                if all_probabilities is not None:
                    assert batch_probabilities is not None
                    all_probabilities.extend(batch_probabilities)
            if progress_label and accelerator.is_main_process:
                now = monotonic()
                if now - last_log >= 10:
                    logger.info(
                        "%s: %d batches completed (%.1fs elapsed)",
                        progress_label,
                        batch_idx,
                        now - started,
                    )
                    last_log = now
    if len(all_predictions) != expected_examples:
        raise RuntimeError(
            f"Local head inference produced {len(all_predictions)} unique "
            f"predictions for {expected_examples} examples."
        )
    return all_ids, all_predictions, all_probabilities
