"""Task-aware tokenization and residue alignment."""

from __future__ import annotations

import logging
import random
from typing import Any

from datasets import DatasetDict  # type: ignore[attr-defined]
from transformers import PreTrainedTokenizerBase

from modules.evaluate.src.tasks import TaskType
from modules.evaluate.src.tasks.registry import get_flat_task

logger = logging.getLogger(__name__)


def tokenize_dataset(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    task_type: TaskType,
    sequence_column: str,
    label_column: str,
    max_length: int,
    id_column: str = "id",
    num_proc: int = 4,
    random_truncate: bool = True,
    random_truncate_by_split: dict[str, bool] | None = None,
    seed: int = 710019,
) -> DatasetDict:
    """Tokenize every split of *dataset* once, ahead of building DataLoaders.

    Uses ``DatasetDict.map(..., batched=True, num_proc=...)`` so tokenization
    happens once (in parallel across ``num_proc`` processes) instead of being
    repeated on every ``EvaluationCollator`` call. For
    ``token_classification``, per-residue labels are aligned here too (``-100``
    at BOS/EOS), so collation only pads.
    Tasks that expand each source example use smaller map batches to bound
    peak memory (one protein per batch for pseudo-perplexity).

    Truncation behavior is split-aware:
    - ``random_truncate`` is the fallback applied everywhere when
        ``random_truncate_by_split`` is not provided.
    - ``random_truncate_by_split`` can override truncation mode per split
        (for example random on train, deterministic on validation and test).

    When random truncation is used, each crop window is deterministic per
    example from ``seed`` + dataset index, so results are stable across
    process counts and worker scheduling.

    Leading/trailing special tokens (BOS/EOS or CLS/EOS, whichever the
    tokenizer defines) are re-attached around the cropped residue span via
    tokenizer special token ids instead of ``build_inputs_with_special_tokens``
    (which some custom tokenizers do not implement).
    """
    if task_type == "contact_prediction" or any(
        "positive_pairs" in data.column_names for data in dataset.values()
    ):
        raise ValueError(
            "Contact scoring rows require tokenize_contact_chain and the contact workflow; "
            "generic tokenization would crop inputs and lose unit metadata"
        )
    task = get_flat_task(task_type)
    if task.aligns_labels and getattr(tokenizer, "remove_ambiguous", False):
        logger.info(
            "Disabling ambiguous-residue removal for positional task '%s' "
            "to preserve source label coordinates.",
            task_type,
        )
        tokenizer.remove_ambiguous = False
        tokenizer.init_kwargs["remove_ambiguous"] = False

    # Preserve the tokenizer's own special-token convention without relying on
    # build_inputs_with_special_tokens(), which is not implemented uniformly by
    # custom protein tokenizers.
    leading_special_ids = [
        token_id
        for token_id in (tokenizer.bos_token_id, tokenizer.cls_token_id)
        if token_id is not None
    ][:1]
    trailing_special_ids = [
        token_id
        for token_id in (tokenizer.eos_token_id, tokenizer.sep_token_id)
        if token_id is not None
    ][:1]
    offset = len(leading_special_ids)
    num_special_tokens = offset + len(trailing_special_ids)
    if max_length <= num_special_tokens:
        raise ValueError(
            "max_length must exceed the tokenizer's number of leading and "
            f"trailing special tokens ({num_special_tokens})."
        )
    max_residues = max_length - num_special_tokens

    def _tokenize_batch_manual(
        examples: dict[str, list[Any]],
        indices: list[int],
        random_truncate: bool,
        seed: int,
    ) -> dict[str, list[Any]]:
        # Tokenize without truncation here so we can control the crop window and
        # keep the residue/label alignment deterministic across worker counts.
        residue_ids_batch = tokenizer(
            examples[sequence_column],
            add_special_tokens=False,
            truncation=False,
            # Manual truncation is applied below; suppress tokenizer-level
            # max-length warnings that would otherwise be noisy/misleading.
            verbose=False,
        )["input_ids"]

        input_ids_batch: list[list[int]] = []
        attention_mask_batch: list[list[int]] = []
        labels_batch: list[Any] = []
        original_positions_batch: list[list[int]] = []
        ids_batch: list[Any] = []
        emit_ids = False

        for i, residue_ids in enumerate(residue_ids_batch):
            num_residues = len(residue_ids)
            start = 0
            if num_residues > max_residues:
                if random_truncate:
                    # Deterministic per-example RNG: same (seed, dataset index)
                    # always yields the same truncation window, regardless of
                    # num_proc or batch composition. A string seed is used
                    # since random.seed() hashes str/bytes via SHA-512
                    # internally, unaffected by PYTHONHASHSEED (unlike a raw
                    # tuple, which random.Random() doesn't accept anyway).
                    rng = random.Random(f"{seed}:{indices[i]}")
                    start = rng.randint(0, num_residues - max_residues)
                residue_ids = residue_ids[start : start + max_residues]

            input_ids = leading_special_ids + residue_ids + trailing_special_ids
            attention_mask = [1] * len(input_ids)
            original_positions = (
                [-1] * offset
                + list(range(start, start + len(residue_ids)))
                + [-1] * len(trailing_special_ids)
            )
            label = examples[label_column][i] if label_column in examples else None
            if task.pairwise and "valid_positions" in examples:
                label = {
                    "pairs": label,
                    "valid_positions": examples["valid_positions"][i],
                }
            if task.aligns_labels:
                label = task.align_labels(
                    label,
                    num_residues=num_residues,
                    start=start,
                    kept=len(residue_ids),
                    offset=offset,
                    input_len=len(input_ids),
                    example_index=indices[i],
                )

            original_id = examples[id_column][i] if id_column in examples else None
            if original_id is None and task_type == "token_classification":
                original_id = indices[i]
            for row in task.expand_tokenized_example(
                input_ids=input_ids,
                attention_mask=attention_mask,
                label=label,
                tokenizer=tokenizer,
                res_start=offset,
                res_end=offset + len(residue_ids),
                example_id=original_id if original_id is not None else indices[i],
                example_index=indices[i],
            ):
                input_ids_batch.append(row["input_ids"])
                attention_mask_batch.append(row["attention_mask"])
                labels_batch.append(row["label"])
                if task.aligns_labels:
                    original_positions_batch.append(original_positions)
                row_id = row.get("id", original_id)
                if row_id is not None:
                    emit_ids = True
                    ids_batch.append(row_id)
                elif emit_ids:
                    ids_batch.append(None)

        output: dict[str, list[Any]] = {
            "input_ids": input_ids_batch,
            "attention_mask": attention_mask_batch,
            label_column: labels_batch,
        }
        if task.aligns_labels:
            output["original_positions"] = original_positions_batch
        if emit_ids:
            output[id_column] = ids_batch

        return output

    random_truncate_by_split = random_truncate_by_split or {}
    mapped: dict[str, Any] = {}
    for split_name, split_dataset in dataset.items():
        if (
            task.requires_source_labels
            and label_column not in split_dataset.column_names
        ):
            raise ValueError(
                f"Dataset split '{split_name}' is missing required label "
                f"column '{label_column}' for task_type='{task_type}'."
            )
        split_random_truncate = random_truncate_by_split.get(
            split_name, random_truncate
        )
        mapped[split_name] = split_dataset.map(
            _tokenize_batch_manual,
            batched=True,
            batch_size=task.tokenization_batch_size,
            num_proc=num_proc,
            with_indices=True,
            remove_columns=split_dataset.column_names,
            fn_kwargs={
                "random_truncate": split_random_truncate,
                "seed": seed if split_random_truncate else 0,
            },
        )
    return DatasetDict(mapped)
