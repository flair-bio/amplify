"""Zero-shot calibration of MLM predictions against homolog amino-acid profiles.

Labels list ``[position, f_A, ..., f_Y, is_t, bin, fold, in_balanced,
in_high_nt, consensus_mask]`` per site; each site is masked inside a residue
window centred on it. Scoring fits one temperature per condition by
cross-fitting over the folds.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from functools import partial
from typing import Any, ClassVar

import numpy as np
import polars as pl
import torch
from transformers import AutoModelForMaskedLM

from modules.evaluate.src.contracts.artifacts import PredictOutput, ScoreOutput
from modules.evaluate.src.contracts.config import ScoreConfig
from modules.evaluate.src.metrics import calibration_metrics
from modules.evaluate.src.metrics.calibration_metrics import COHORTS
from modules.evaluate.src.tasks.base import FlatMetricsTask, MetricFn
from modules.evaluate.src.tasks.registry import register_task
from modules.evaluate.src.utils.amino_acids import resolve_aa_token_ids

logger = logging.getLogger(__name__)

NUM_AA = 20
WINDOW_RESIDUES = 512
# Site fields after the 20 profile frequencies.
IS_T, BIN, FOLD, IN_BALANCED, IN_HIGH_NT, CONSENSUS_MASK = range(NUM_AA, NUM_AA + 6)


@register_task
class CalibrationTask(FlatMetricsTask):
    """Mask each site and compare the MLM to its homolog profile."""

    evaluation_modes = frozenset({"evaluate_as_is"})
    supports_packing = False

    name = "calibration"
    auto_model_class = AutoModelForMaskedLM
    label_dtype = torch.float
    aligns_labels = True
    tokenization_batch_size = 1
    prediction_progress_unit = "masked positions"
    accepted_target_types = frozenset({"categorical"})
    accepted_label_formats = frozenset({"consensus_profile"})

    metrics: ClassVar[dict[str, MetricFn]] = {
        **{
            f"calibration_{c}": partial(calibration_metrics.cohort_score, cohort=c)
            for c in COHORTS
        },
        **{
            f"calibration_fit_{c}": partial(
                calibration_metrics.cohort_score, cohort=c, fitted=True
            )
            for c in COHORTS
        },
        **{
            f"kl_{c}": partial(calibration_metrics.cohort_kl, cohort=c) for c in COHORTS
        },
        **{
            f"accuracy_{c}": partial(calibration_metrics.cohort_accuracy, cohort=c)
            for c in COHORTS
        },
        "temperature": calibration_metrics.mean_temperature,
    }
    default_metric_names = tuple(f"calibration_{c}" for c in COHORTS)
    # Scores are computed against the true profiles before metrics run, so
    # shuffling labels afterwards would only shuffle cohort membership.
    supported_controls = FlatMetricsTask.supported_controls - {"scrambled_labels"}

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
        # original position for row ids: [local_pos, pos, f_A..f_Y, site fields].
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
        example_index: int,
    ) -> list[dict[str, Any]]:
        mask_token_id = tokenizer.mask_token_id
        if mask_token_id is None:
            raise ValueError("calibration requires a tokenizer mask token.")
        aa_ids = resolve_aa_token_ids(tokenizer)

        leading = input_ids[:res_start]
        residues = input_ids[res_start:res_end]
        trailing = input_ids[res_end:]
        max_start = max(0, len(residues) - WINDOW_RESIDUES)

        rows: list[dict[str, Any]] = []
        for local_pos, pos, *site in label:
            local_pos = int(local_pos)
            start = min(max(0, local_pos - WINDOW_RESIDUES // 2), max_start)
            window = residues[start : start + WINDOW_RESIDUES]
            target = local_pos - start
            window[target] = mask_token_id
            window_ids = leading + window + trailing
            rows.append(
                {
                    "input_ids": window_ids,
                    "attention_mask": [1] * len(window_ids),
                    # [mask index, 20 AA token ids, 20 homolog freqs, site fields]
                    "label": [len(leading) + target, *aa_ids, *site],
                    "id": f"{example_id}:{int(pos)}",
                }
            )
        return rows

    def extract_predictions(
        self, logits: torch.Tensor, *, labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the 20 raw amino-acid logits at each masked position."""
        if labels is None:
            raise ValueError("calibration requires labels to locate masked tokens.")

        rows = torch.arange(labels.shape[0], device=logits.device)
        mask_idx = labels[:, 0].long()
        aa_ids = labels[:, 1 : 1 + NUM_AA].long()
        return logits[rows, mask_idx].float().gather(1, aa_ids)

    def split_batch_rows(
        self, predictions: torch.Tensor, labels: torch.Tensor
    ) -> list[tuple[Any, Any]]:
        """Pair the AA logits with ``[f_A..f_Y, site fields]``."""
        return list(
            zip(predictions.cpu().tolist(), labels[:, 1 + NUM_AA :].cpu().tolist())
        )

    def bootstrap_group_id(self, row_id: Any) -> Any:
        """Resample whole proteins: positions of one protein are correlated."""
        return row_id.rsplit(":", 1)[0]

    def score(
        self,
        predictions: PredictOutput,
        config: ScoreConfig,
        *,
        seed: int = 710019,
    ) -> ScoreOutput:
        """Cross-fit a temperature per condition, then score the KL rows.

        Predictions become ``[kl, kl_fit, temperature, pred_idx]`` and labels
        ``[is_t, in_balanced, in_high_nt, fold, consensus_mask]``.
        """
        labels = np.asarray(predictions.predictions["label"].to_list(), dtype=float)
        profiles = labels[:, :NUM_AA] / labels[:, :NUM_AA].sum(1, keepdims=True)
        strata = 2 * labels[:, BIN] + (1 - labels[:, IS_T])
        folds = labels[:, FOLD]
        fit_rows = labels[:, IN_BALANCED] == 1

        def kl_rows(frame: pl.DataFrame, condition: str) -> pl.DataFrame:
            logits = np.asarray(frame["prediction"].to_list(), dtype=float)
            logger.info("Cross-fitting the %s temperature.", condition)
            temperatures = calibration_metrics.crossfit_temperatures(
                logits, profiles, strata, folds, fit_rows
            )
            rows = np.column_stack(
                [
                    calibration_metrics.forward_kl(logits, profiles, 1.0),
                    calibration_metrics.forward_kl(logits, profiles, temperatures),
                    temperatures,
                    logits.argmax(1),
                ]
            )
            return frame.with_columns(prediction=pl.Series(rows.tolist()))

        compact = labels[:, [IS_T, IN_BALANCED, IN_HIGH_NT, FOLD, CONSENSUS_MASK]]
        scored = replace(
            predictions,
            predictions=kl_rows(predictions.predictions, "trained").with_columns(
                label=pl.Series(compact.tolist())
            ),
            controls={
                name: kl_rows(frame, name)
                for name, frame in predictions.controls.items()
            },
        )
        return super().score(scored, config, seed=seed)
