import math
from typing import Any, Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

from modules.pretrain.src.model.tokenizer import ProteinTokenizer


class CollatorConfig(BaseModel):
    """Configuration for :class:`DataCollator`.

    Attributes:
        max_length: Maximum sequence length after truncation.
        packing: Pack batch into a flat ``(1, T)`` buffer with ``cu_seqlens``.
        mlm_probability: Base masking probability.
        masking_type: ``"fixed"``, ``"beta"``, or ``"cosine"`` masking sampling strategy.
        masking_k: Concentration parameter for the beta distribution schedule.
        pad_to_multiple_of: Pad sequence length to a multiple of this value.
        random_truncate: Sample a random window when truncating long sequences.
        exclude_special_tokens_from_masking: Never mask special tokens (BOS, EOS, etc.).
        mlm: Enable masked language modeling
    """

    model_config = ConfigDict(extra="forbid")

    max_length: int = Field(
        512, description="Maximum sequence length after truncation."
    )
    packing: bool = Field(
        False,
        description="Pack batch into a flat (1, T) buffer with cu_seqlens.",
    )
    mlm_probability: float = Field(0.15, description="Base masking probability.")
    masking_type: Literal["fixed", "beta", "cosine"] = Field(
        "fixed",
        description=(
            'Masking schedule type: "fixed" always uses mlm_probability; '
            '"beta" samples from a Beta distribution centred on mlm_probability; '
            '"cosine" sweeps from 0 to mlm_probability following a cosine curve '
            "(i.e. mlm_probability is the maximum masking rate)."
        ),
    )
    masking_k: int = Field(
        10, description="Concentration parameter for beta distribution schedule."
    )
    pad_to_multiple_of: int | None = Field(
        None, description="Pad sequence length to a multiple of this value."
    )
    random_truncate: bool = Field(
        True, description="Sample a random window when truncating long sequences."
    )
    exclude_special_tokens_from_masking: bool = Field(
        True, description="Never mask special tokens (BOS, EOS, etc.)."
    )
    mlm: bool = Field(True, description="Enable masked language modeling.")


class DataCollator:
    """MLM data collator with packing and padding modes.

    Supports fixed masking probability or dynamic sampling via beta/cosine
    schedules, plus optional packing for PyTorch variable-length attention.

    Args:
        tokenizer: Protein tokenizer.
        collator_config: Collator configuration.
    """

    def __init__(
        self,
        tokenizer: ProteinTokenizer,
        collator_config: CollatorConfig,
    ) -> None:
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_token_id = tokenizer.mask_token_id
        self.max_length = collator_config.max_length
        self.packing = collator_config.packing
        self.mlm_probability = collator_config.mlm_probability
        self.masking_type = collator_config.masking_type
        self.random_truncate = collator_config.random_truncate
        self.exclude_special_tokens_from_masking = (
            collator_config.exclude_special_tokens_from_masking
        )
        self.mlm = collator_config.mlm
        self.pad_to_multiple_of = collator_config.pad_to_multiple_of

        # Pre-compute Beta distribution parameters for dynamic masking sampling strategy.
        self.alpha = collator_config.mlm_probability * collator_config.masking_k
        self.beta_param = (
            1 - collator_config.mlm_probability
        ) * collator_config.masking_k

        self._tokenizer = tokenizer
        self._special_ids = np.array(sorted(tokenizer.all_special_ids), dtype=np.int64)

    def _sample_mlm_probability(self) -> float:
        """Sample a masking probability from the configured distribution ("fixed", "beta", or "cosine").

        This is a per-sequence dynamic sampling strategy, not a time-based step schedule.
        """
        if self.masking_type == "beta":
            return float(np.random.beta(self.alpha, self.beta_param))
        elif self.masking_type == "cosine":
            t = np.random.random()
            return float(self.mlm_probability * (1.0 - math.cos(t * math.pi / 2)))
        return self.mlm_probability

    def _special_mask_for(self, ids: np.ndarray) -> np.ndarray:
        """Return a boolean mask of positions that should never be MLM-masked."""
        if self.exclude_special_tokens_from_masking:
            return np.isin(ids, self._special_ids)
        return ids == self.pad_token_id

    def _ensure_min_one_mask(self, masked: np.ndarray, special: np.ndarray) -> None:
        """If no position is masked, randomly mask one non-special position (in-place)."""
        if not masked.any():
            candidates = np.where(~special)[0]
            if len(candidates) > 0:
                masked[candidates[np.random.randint(len(candidates))]] = True

    def _apply_mlm_masking(self, flat_ids: np.ndarray, lengths: np.ndarray) -> tuple:
        """Apply MLM masking to a flat 1D concatenation of sequences.

        Returns ``(masked_ids, labels)`` where *labels* has ``-100`` at unmasked positions.
        """
        num_sequences = len(lengths)
        special_mask = self._special_mask_for(flat_ids)
        prob = np.repeat(
            [self._sample_mlm_probability() for _ in range(num_sequences)],
            lengths.astype(int),
        )
        prob[special_mask] = 0.0
        masked_indices = np.random.random(len(flat_ids)) < prob

        offset = 0
        for n in lengths:
            n = int(n)
            self._ensure_min_one_mask(
                masked_indices[offset : offset + n], special_mask[offset : offset + n]
            )
            offset += n

        labels = flat_ids.copy()
        labels[~masked_indices] = -100
        masked_ids = flat_ids.copy()
        masked_ids[masked_indices] = self.mask_token_id
        return masked_ids, labels

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        # Tokenize all sequences in one call (Rust-backed for speed).
        encoded = self._tokenizer(
            [s["sequence"] for s in samples],
            truncation=True,
            max_length=self.max_length,
            random_truncate=self.random_truncate,
        )
        all_ids = [np.array(ids, dtype=np.int64) for ids in encoded["input_ids"]]
        lengths = np.array([len(ids) for ids in all_ids])

        if self.packing:
            return self._collate_packed(all_ids, lengths)
        return self._collate_padded(all_ids, lengths)

    def _collate_packed(
        self, all_ids: list[np.ndarray], lengths: np.ndarray
    ) -> dict[str, Any]:
        """Flat ``(1, T)`` buffer with cumulative sequence offsets."""
        num_sequences = len(all_ids)
        total_tokens = int(lengths.sum())
        buf_size = total_tokens
        if self.pad_to_multiple_of is not None:
            buf_size = (
                math.ceil(total_tokens / self.pad_to_multiple_of)
                * self.pad_to_multiple_of
            )

        flat_ids = np.concatenate(all_ids)

        if self.mlm:
            flat_ids, labels = self._apply_mlm_masking(flat_ids, lengths)
        else:
            labels = np.full(total_tokens, -100, dtype=np.int64)

        # Pad output buffers to buf_size (a multiple of pad_to_multiple_of).
        input_ids = np.full(buf_size, self.pad_token_id, dtype=np.int64)
        input_ids[:total_tokens] = flat_ids
        labels_buf = np.full(buf_size, -100, dtype=np.int64)
        labels_buf[:total_tokens] = labels
        position_ids = np.zeros(buf_size, dtype=np.int64)
        position_ids[:total_tokens] = np.concatenate(
            [np.arange(n, dtype=np.int64) for n in lengths]
        )

        # Last entry treats the padding region as a dummy sequence, so its
        # length must be included in max_seqlen to avoid OOB reads.
        cu_seqlens = np.zeros(num_sequences + 2, dtype=np.int32)
        cu_seqlens[1 : num_sequences + 1] = np.cumsum(lengths)
        cu_seqlens[num_sequences + 1] = buf_size
        padding_segment_len = buf_size - total_tokens
        max_seqlen = max(int(lengths.max()), padding_segment_len)
        return {
            "input_ids": torch.from_numpy(input_ids).unsqueeze(0),
            "labels": torch.from_numpy(labels_buf).unsqueeze(0),
            "position_ids": torch.from_numpy(position_ids).unsqueeze(0),
            "cu_seqlens": torch.from_numpy(cu_seqlens),
            "max_seqlen": max_seqlen,
            "num_sequences": num_sequences,
            "num_tokens": total_tokens,
        }

    def _collate_padded(
        self, all_ids: list[np.ndarray], lengths: np.ndarray
    ) -> dict[str, Any]:
        """Padded ``(B, S)`` tensors with ``attention_mask``."""
        num_sequences = len(all_ids)
        seq_len = int(lengths.max())
        if self.pad_to_multiple_of is not None:
            seq_len = (
                math.ceil(seq_len / self.pad_to_multiple_of) * self.pad_to_multiple_of
            )

        input_ids = np.full((num_sequences, seq_len), self.pad_token_id, dtype=np.int64)
        for i, (ids, n) in enumerate(zip(all_ids, lengths)):
            input_ids[i, :n] = ids

        if self.mlm:
            # Flatten real tokens, apply masking, scatter back to 2D.
            real_ids = np.concatenate(
                [input_ids[i, : int(n)] for i, n in enumerate(lengths)]
            )
            masked_flat, labels_flat = self._apply_mlm_masking(real_ids, lengths)
            labels = np.full((num_sequences, seq_len), -100, dtype=np.int64)
            offset = 0
            for i, n in enumerate(lengths):
                n = int(n)
                input_ids[i, :n] = masked_flat[offset : offset + n]
                labels[i, :n] = labels_flat[offset : offset + n]
                offset += n
        else:
            labels = np.full((num_sequences, seq_len), -100, dtype=np.int64)

        cols = np.arange(seq_len)
        attention_mask = cols[None, :] < lengths[:, None]
        position_ids = np.where(attention_mask, cols[None, :], 0)

        return {
            "input_ids": torch.from_numpy(input_ids),
            "labels": torch.from_numpy(labels),
            "attention_mask": torch.from_numpy(np.ascontiguousarray(attention_mask)),
            "position_ids": torch.from_numpy(np.ascontiguousarray(position_ids)),
        }


def get_collator(
    tokenizer: ProteinTokenizer,
    collator_config: CollatorConfig,
) -> DataCollator:
    """Instantiate the DataCollator from config."""
    return DataCollator(tokenizer=tokenizer, collator_config=collator_config)
