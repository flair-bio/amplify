"""Zero-shot calibration of MLM predictions against homolog amino-acid profiles.

Labels list ``[position, f_A, ..., f_Y]`` per conserved position; each position
is masked inside a residue window centred on it.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
from transformers import AutoModelForMaskedLM

from modules.evaluate.src.metrics import metrics
from modules.evaluate.src.model.categorical_jacobian import resolve_aa_token_ids
from modules.evaluate.src.tasks.base import MetricFn, TaskHandler
from modules.evaluate.src.tasks.registry import register_task

NUM_AA = 20
WINDOW_RESIDUES = 512
KL_EPS = 1e-10


@register_task
class CalibrationTask(TaskHandler):
    """Mask each conserved residue and compare the MLM to its homolog profile."""

    name = "calibration"
    auto_model_class = AutoModelForMaskedLM
    label_dtype = torch.float
    aligns_labels = True
    tokenization_batch_size = 1
    accepted_target_types = frozenset({"categorical"})
    accepted_label_formats = frozenset({"consensus_profile"})

    metrics: ClassVar[dict[str, MetricFn]] = {
        "kl": metrics.calibration_kl,
        "kl_t": metrics.calibration_kl_t,
        "kl_nt": metrics.calibration_kl_nt,
        "calibration": metrics.calibration_score,
        "calibration_t": metrics.calibration_score_t,
        "calibration_nt": metrics.calibration_score_nt,
        "accuracy_t": metrics.calibration_accuracy_t,
        "accuracy_nt": metrics.calibration_accuracy_nt,
        "memorization_nt": metrics.calibration_memorization_nt,
    }
    default_metric_names = (
        "kl",
        "kl_t",
        "kl_nt",
        "calibration",
        "calibration_t",
        "calibration_nt",
        "accuracy_t",
        "accuracy_nt",
        "memorization_nt",
    )

    def build_model(
        self,
        model_id: str,
        num_labels: int,
        model_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        return self.auto_model_class.from_pretrained(
            model_id,
            trust_remote_code=True,
            **{**self.model_kwargs(), **(model_kwargs or {})},
        )

    def resolve_num_labels(self, num_labels: int | None, dataset_name: str) -> int:
        return 1

    def align_labels(
        self,
        label: Any,
        *,
        num_residues: int,
        start: int,
        kept: int,
        offset: int,
        input_len: int,
        example_index: int,
    ) -> list[list[float]]:
        # Re-express positions relative to the kept residues, keeping the
        # original position for row ids: [local_pos, pos, f_A..f_Y].
        return [
            [entry[0] - start, *entry]
            for entry in label
            if start <= entry[0] < start + kept
        ]

    def expand_tokenized_example(
        self,
        *,
        input_ids: list[int],
        attention_mask: list[int],
        label: Any,
        tokenizer: Any,
        res_start: int,
        res_end: int,
        example_id: Any,
    ) -> list[dict[str, Any]]:
        mask_token_id = tokenizer.mask_token_id
        if mask_token_id is None:
            raise ValueError("calibration requires a tokenizer mask token.")
        aa_ids = resolve_aa_token_ids(tokenizer)
        aa_index = {token_id: i for i, token_id in enumerate(aa_ids)}

        leading = input_ids[:res_start]
        residues = input_ids[res_start:res_end]
        trailing = input_ids[res_end:]
        max_start = max(0, len(residues) - WINDOW_RESIDUES)

        rows: list[dict[str, Any]] = []
        for local_pos, pos, *freqs in label:
            local_pos = int(local_pos)
            start = min(max(0, local_pos - WINDOW_RESIDUES // 2), max_start)
            window = residues[start : start + WINDOW_RESIDUES]
            target = local_pos - start
            wt_idx = aa_index[window[target]]
            window[target] = mask_token_id
            window_ids = leading + window + trailing
            rows.append(
                {
                    "input_ids": window_ids,
                    "attention_mask": [1] * len(window_ids),
                    # [mask index, wt index, 20 AA token ids, 20 homolog freqs]
                    "label": [len(leading) + target, wt_idx, *aa_ids, *freqs],
                    "id": f"{example_id}:{int(pos)}",
                }
            )
        return rows

    def extract_predictions(
        self, logits: torch.Tensor, *, labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return ``[KL(p_homologs || p_model), predicted AA index]`` per row."""
        if labels is None:
            raise ValueError("calibration requires labels to locate masked tokens.")

        rows = torch.arange(labels.shape[0], device=logits.device)
        mask_idx = labels[:, 0].long()
        aa_ids = labels[:, 2 : 2 + NUM_AA].long()
        aa_logits = logits[rows, mask_idx].float().gather(1, aa_ids)
        model_probs = aa_logits.softmax(dim=-1)

        homolog = labels[:, 2 + NUM_AA :].float()
        homolog = homolog / homolog.sum(dim=-1, keepdim=True)
        p = homolog.clamp(KL_EPS, 1.0)
        q = model_probs.clamp(KL_EPS, 1.0)
        kl = (p * (p / q).log()).sum(dim=-1)
        return torch.stack([kl, model_probs.argmax(dim=-1).float()], dim=-1)

    def split_batch_rows(
        self, predictions: torch.Tensor, labels: torch.Tensor
    ) -> list[tuple[Any, Any]]:
        """Pair ``[kl, pred_idx]`` with ``[wt_idx, consensus_idx]``."""
        wt_idx = labels[:, 1].long()
        consensus_idx = labels[:, 2 + NUM_AA :].argmax(dim=-1)
        compact_labels = torch.stack([wt_idx, consensus_idx], dim=-1)
        return list(zip(predictions.cpu().tolist(), compact_labels.cpu().tolist()))

    def bootstrap_group_id(self, row_id: Any) -> Any:
        """Resample whole proteins: positions of one protein are correlated."""
        return row_id.rsplit(":", 1)[0]
