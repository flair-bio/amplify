"""Gather per-layer predictions and controls from live or cached probe features."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Mapping, NamedTuple

import torch
from accelerate import Accelerator
from torch import nn

from modules.evaluate.src.dataset.collator import input_id_rows, unpack_packed
from modules.evaluate.src.model.heads import EmbeddingProbeHead
from modules.evaluate.src.model.layerwise import LayerwiseProbeModel
from modules.evaluate.src.tasks import get_task
from modules.evaluate.src.tasks.token_classification import IGNORE_INDEX
from modules.evaluate.src.utils.controls import (
    NO_INFERENCE_CONTROLS,
    RANDOMIZED_TRUNK_CONTROLS,
    ControlBuilder,
)
from modules.evaluate.src.utils.inference import pad_for_gather, unique_id_indices

if TYPE_CHECKING:
    from modules.evaluate.src.schemas.artifacts import PreparedArtifacts
    from modules.evaluate.src.steps.predict import PredictConfig

# With a frozen trunk these two controls are identical, so they share one run.
FRESH_HEAD_CONTROLS = frozenset({"untrained", "random_head"})
CONSTANT_CONTROLS = frozenset({"majority_class", "train_mean"})


class LayerRows(NamedTuple):
    """Aligned per-layer prediction rows; the last two fields are optional."""

    ids: list[Any]
    predictions: list[Any]
    labels: list[Any]
    probabilities: list[Any] | None
    sequences: list[str] | None


ConditionRows = dict[str, dict[int, LayerRows]]


def fresh_heads(
    inner: LayerwiseProbeModel, seed: int, device: torch.device
) -> nn.ModuleDict:
    """Seeded untrained heads, built under a forked RNG to keep global state intact."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        heads = nn.ModuleDict(
            {
                str(layer): EmbeddingProbeHead(
                    inner.embedding_size, inner.num_labels, inner.task_type
                )
                for layer in inner.layers
            }
        )
    return heads.to(device)


def run_layerwise_control_conditions(
    prepared: PreparedArtifacts,
    task_type: str,
    accelerator: Accelerator,
    config: PredictConfig,
    seed: int,
) -> ConditionRows:
    """Collect predictions for trained probes and configured controls per layer.

    Head-only controls share the trained probes' features. Scrambled inputs
    and random-trunk controls use original token batches because cached
    features cannot reflect those changes. ``random_trunk`` uses probes that
    TuneStep trained on the random trunk; ``random_both`` uses untrained heads
    on it. Constant and label-only controls are handled downstream. Every
    inferred condition must return the same layer positions.
    """
    controls: set[str] = set(config.controls)
    unsupported = controls - (
        FRESH_HEAD_CONTROLS
        | RANDOMIZED_TRUNK_CONTROLS
        | CONSTANT_CONTROLS
        | NO_INFERENCE_CONTROLS
        | {"scrambled_sequences"}
    )
    if unsupported:
        raise ValueError(f"Unsupported layerwise controls: {sorted(unsupported)}")
    if controls & RANDOMIZED_TRUNK_CONTROLS and prepared.layerwise_random_model is None:
        raise ValueError(
            "random_trunk/random_both need the random-trunk probes from "
            "TuneStep; pass the predict controls to TuneStep.run."
        )
    inner = accelerator.unwrap_model(prepared.model)
    using_embeddings = prepared.embedding_backend is not None
    token_dataloader = (
        prepared.layerwise_token_dataloaders["test"]
        if prepared.layerwise_token_dataloaders is not None
        else prepared.test_dataloader
    )
    fresh_head_controls = controls & FRESH_HEAD_CONTROLS
    conditions = run_layerwise_conditions(
        prepared.model,
        prepared.test_dataloader,
        task_type,
        accelerator,
        collect_probabilities=config.collect_probabilities,
        tokenizer=(
            prepared.tokenizer
            if config.store_sequences and not using_embeddings
            else None
        ),
        extra_heads=(
            {"fresh": fresh_heads(inner, seed, accelerator.device)}
            if fresh_head_controls
            else None
        ),
    )
    if fresh_head_controls:
        fresh_rows = conditions.pop("fresh")
        for control in fresh_head_controls:
            conditions[control] = fresh_rows
    if "scrambled_sequences" in controls:
        conditions["scrambled_sequences"] = run_layerwise_conditions(
            inner,
            token_dataloader,
            task_type,
            accelerator,
            collect_probabilities=config.collect_probabilities,
            batch_transform=ControlBuilder.make_sequence_scrambler(
                prepared.tokenizer, seed
            ),
            tokenizer=prepared.tokenizer,
        )["trained"]
    if controls & RANDOMIZED_TRUNK_CONTROLS:
        assert prepared.layerwise_random_model is not None
        random_inner = accelerator.unwrap_model(prepared.layerwise_random_model)
        random_conditions = run_layerwise_conditions(
            prepared.layerwise_random_model,
            token_dataloader,
            task_type,
            accelerator,
            collect_probabilities=config.collect_probabilities,
            extra_heads=(
                {"random_both": fresh_heads(random_inner, seed, accelerator.device)}
                if "random_both" in controls
                else None
            ),
            flatten_ragged=using_embeddings,
        )
        if "random_trunk" in controls:
            conditions["random_trunk"] = random_conditions["trained"]
        if "random_both" in controls:
            conditions["random_both"] = random_conditions["random_both"]

    expected_layers = set(inner.layers)
    for condition, layer_outputs in conditions.items():
        if set(layer_outputs) != expected_layers:
            raise RuntimeError(
                f"Layerwise condition '{condition}' returned layers "
                f"{sorted(layer_outputs)}, expected {sorted(expected_layers)}."
            )
    return conditions


def run_layerwise_conditions(
    model: Any,
    dataloader: Any,
    task_type: str,
    accelerator: Accelerator,
    collect_probabilities: bool = False,
    batch_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    tokenizer: Any | None = None,
    extra_heads: Mapping[str, nn.ModuleDict] | None = None,
    flatten_ragged: bool = False,
) -> ConditionRows:
    """Gather aligned rows keyed by condition name and hidden-state position.

    Each batch extracts features once for trained and optional control heads.
    Probabilities and sequences are None unless requested. Token predictions
    exclude ignored labels. ``flatten_ragged`` emits singleton residue rows so
    live controls match token caches, rather than one row per source sequence.

    Distributed sampler duplicates are removed by source ID, except for cached
    residue rows whose repeated IDs are intentional. Without IDs, row numbers
    are assigned as results are appended.
    """
    model.eval()
    task = get_task(task_type)
    inner = accelerator.unwrap_model(model)
    layer_ids = inner.layers
    head_sets = {"trained": inner.heads, **(extra_heads or {})}
    rows: ConditionRows = {
        name: {
            layer: LayerRows(
                [],
                [],
                [],
                [] if collect_probabilities else None,
                [] if tokenizer else None,
            )
            for layer in layer_ids
        }
        for name in head_sets
    }
    flatten = flatten_ragged and task.is_ragged
    example_offset = 0
    seen_ids: set[Any] = set()
    with torch.inference_mode():
        for raw_batch in dataloader:
            batch = dict(raw_batch)
            ids = batch.pop("id", None)
            labels = batch.pop("labels")
            if batch_transform is not None:
                batch = batch_transform(batch)
            outputs = inner.forward_conditions(head_sets, **batch)
            if "cu_seqlens" in batch and task.is_ragged:
                # Expand packed targets before gathering so they align with the
                # per-sequence rows emitted by each token-classification head.
                labels = unpack_packed(
                    labels, batch["cu_seqlens"], batch["num_sequences"], IGNORE_INDEX
                )
            targets: dict[str, torch.Tensor] = {
                "labels": pad_for_gather(labels, accelerator, IGNORE_INDEX)
            }
            for name, layer_outputs in outputs.items():
                for layer, output in layer_outputs.items():
                    targets[f"pred_{name}_{layer}"] = pad_for_gather(
                        task.extract_predictions(output.logits, labels=labels),
                        accelerator,
                        IGNORE_INDEX,
                    )
                    if collect_probabilities:
                        targets[f"prob_{name}_{layer}"] = pad_for_gather(
                            task.predicted_probabilities(output.logits), accelerator
                        )
            if tokenizer is not None:
                pad_id = tokenizer.pad_token_id or 0
                targets["input_ids"] = pad_for_gather(
                    input_id_rows(batch, pad_id), accelerator, pad_id
                )
            # Gather labels and every condition together to keep row ordering
            # identical across layers and distributed workers.
            gathered = accelerator.gather_for_metrics(targets)
            if ids is not None:
                ids = accelerator.gather_for_metrics(ids, use_gather_object=True)
                ids = ids[: gathered["labels"].shape[0]]
                # Cached ragged inputs repeat a source ID for each residue; those
                # rows are predictions, not distributed sampler padding.
                if getattr(accelerator, "num_processes", 1) > 1 and not (
                    task.is_ragged and "inputs_embeds" in batch
                ):
                    indices = unique_id_indices(ids, seen_ids)
                    ids = [ids[index] for index in indices]
                    gathered = {key: value[indices] for key, value in gathered.items()}
            elif flatten:
                ids = list(
                    range(example_offset, example_offset + gathered["labels"].shape[0])
                )
            if flatten:
                example_offset += gathered["labels"].shape[0]
            # One device-to-host copy per batch rather than one per layer and condition.
            gathered = {key: value.cpu() for key, value in gathered.items()}
            sequences = (
                tokenizer.batch_decode(
                    gathered["input_ids"].tolist(), skip_special_tokens=True
                )
                if tokenizer is not None
                else None
            )
            for name in head_sets:
                for layer in layer_ids:
                    layer_rows = rows[name][layer]
                    for index, (pred, label) in enumerate(
                        task.split_batch_rows(
                            gathered[f"pred_{name}_{layer}"], gathered["labels"]
                        )
                    ):
                        row_id = ids[index] if ids is not None else len(layer_rows.ids)
                        if flatten:
                            layer_rows.ids.extend([row_id] * len(pred))
                            layer_rows.predictions.extend([value] for value in pred)
                            layer_rows.labels.extend([value] for value in label)
                        else:
                            layer_rows.ids.append(row_id)
                            layer_rows.predictions.append(pred)
                            layer_rows.labels.append(label)
                    if layer_rows.probabilities is not None:
                        layer_rows.probabilities.extend(
                            gathered[f"prob_{name}_{layer}"].tolist()
                        )
                    if layer_rows.sequences is not None and sequences is not None:
                        layer_rows.sequences.extend(sequences)
    return rows
