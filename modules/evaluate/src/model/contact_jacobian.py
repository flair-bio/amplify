"""Zero-shot CASP contact prediction from categorical mutation Jacobians."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForMaskedLM, PreTrainedTokenizerBase

from modules.evaluate.src.model.contact_model import contact_positions
from modules.evaluate.src.utils.amino_acids import resolve_aa_token_ids

if TYPE_CHECKING:
    from modules.evaluate.src.tasks.contact_prediction import ContactChain

logger = logging.getLogger(__name__)

MUTANT_BATCH_SIZE = 256
JACOBIAN_DTYPE = torch.float32
SCORE_BLOCK_SIZE = 16


def jacobian_to_contacts(jacobian: torch.Tensor) -> torch.Tensor:
    """Reduce a ``(L,20,L,20)`` Jacobian in place to an ``(L,L)`` score map."""
    if jacobian.ndim != 4 or jacobian.shape[1] != 20 or jacobian.shape[3] != 20:
        raise ValueError("Expected a Jacobian shaped (length, 20, length, 20)")
    length = jacobian.shape[0]
    if length != jacobian.shape[2]:
        raise ValueError("Jacobian residue axes must have equal lengths")
    jacobian = jacobian.float()
    for start in range(0, length, SCORE_BLOCK_SIZE):
        if not torch.isfinite(jacobian[start : start + SCORE_BLOCK_SIZE]).all():
            raise ValueError("Jacobian values must be finite")

    for axis in range(4):
        jacobian.sub_(jacobian.mean(dim=axis, keepdim=True))

    scores = torch.empty((length, length), dtype=jacobian.dtype, device=jacobian.device)
    for start in range(0, length, SCORE_BLOCK_SIZE):
        stop = min(start + SCORE_BLOCK_SIZE, length)
        forward = jacobian[start:stop]
        reverse = jacobian[:, :, start:stop, :].permute(2, 3, 0, 1)
        scores[start:stop] = torch.linalg.vector_norm(
            0.5 * (forward + reverse), dim=(1, 3)
        )
    scores = 0.5 * (scores + scores.T)
    scores.fill_diagonal_(0)

    row_sum = scores.sum(dim=-1, keepdim=True)
    col_sum = scores.sum(dim=-2, keepdim=True)
    total = scores.sum()
    correction = torch.where(
        total > 0,
        row_sum * col_sum / total.clamp_min(torch.finfo(scores.dtype).tiny),
        torch.zeros_like(scores),
    )
    scores = scores - correction
    scores = 0.5 * (scores + scores.T)
    scores.fill_diagonal_(0)
    return scores


def _canonical_log_probabilities(
    logits: torch.Tensor,
    residue_ids: torch.Tensor,
    aa_token_ids: torch.Tensor,
) -> torch.Tensor:
    return F.log_softmax(logits[:, residue_ids].float(), dim=-1)[:, :, aa_token_ids]


@contextmanager
def _jacobian_precision(model: nn.Module, device: torch.device) -> Iterator[None]:
    """Autocast trunk inference on CUDA while keeping the MLM head in FP32."""
    head = getattr(model, "lm_head", None)
    if head is None:
        head = getattr(model, "decoder", None)
    native_forward = getattr(head, "forward", None)
    use_cuda = device.type == "cuda"
    if use_cuda and head is not None and native_forward is not None:

        def full_precision_head(values: torch.Tensor) -> Any:
            with torch.autocast("cuda", enabled=False):
                return native_forward(values.float())

        head.forward = full_precision_head
    try:
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_cuda):
            yield
    finally:
        if use_cuda and head is not None and native_forward is not None:
            head.forward = native_forward


class ContactJacobianModel(nn.Module):
    """Frozen masked-LM wrapper producing full-chain contact score matrices."""

    def __init__(
        self,
        trunk: nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        *,
        mutant_batch_size: int = MUTANT_BATCH_SIZE,
    ) -> None:
        super().__init__()
        if mutant_batch_size < 1:
            raise ValueError("mutant_batch_size must be greater than zero")
        self.trunk = trunk.float().eval().requires_grad_(False)
        self.aa_token_ids: torch.Tensor
        self.register_buffer(
            "aa_token_ids",
            torch.tensor(resolve_aa_token_ids(tokenizer), dtype=torch.long),
        )
        self.mutant_batch_size = mutant_batch_size

    @torch.no_grad()
    def predict(
        self, chain: ContactChain, tokenizer: PreTrainedTokenizerBase
    ) -> torch.Tensor:
        from modules.evaluate.src.tasks.contact_prediction import tokenize_contact_chain

        tokens = tokenize_contact_chain(chain, tokenizer)
        device = next(self.trunk.parameters()).device
        input_ids = torch.tensor([tokens.input_ids], dtype=torch.long, device=device)
        position_ids = torch.tensor(
            [tokens.position_ids], dtype=torch.long, device=device
        )
        residue_ids = torch.tensor(
            tokens.residue_token_indices, dtype=torch.long, device=device
        )
        forward_parameters = inspect.signature(self.trunk.forward).parameters
        # Full-chain mutation batches are unpadded; provide a mask only when required.
        inputs: dict[str, torch.Tensor] = {"input_ids": input_ids}
        attention_parameter = forward_parameters.get("attention_mask")
        if (
            attention_parameter is not None
            and attention_parameter.default is inspect.Parameter.empty
        ):
            inputs["attention_mask"] = torch.ones_like(input_ids)
        if "position_ids" in forward_parameters:
            inputs["position_ids"] = position_ids

        length = len(residue_ids)
        aa_count = int(self.aa_token_ids.numel())
        try:
            jacobian = torch.empty(
                (length * aa_count, length, aa_count),
                device=device,
                dtype=JACOBIAN_DTYPE,
            )
        except torch.cuda.OutOfMemoryError as error:
            required_gib = length * length * aa_count * aa_count * 4 / 2**30
            raise RuntimeError(
                f"Could not allocate the FP32 categorical Jacobian for "
                f"{chain.record_id} ({length} residues; approximately "
                f"{required_gib:.2f} GiB)."
            ) from error

        def log_probabilities(batch: dict[str, torch.Tensor]) -> torch.Tensor:
            output = self.trunk(**batch)
            logits = output.logits if hasattr(output, "logits") else output[0]
            return _canonical_log_probabilities(logits, residue_ids, self.aa_token_ids)

        self.trunk.eval()
        with (
            torch.inference_mode(),
            contact_positions(self.trunk, position_ids),
            _jacobian_precision(self.trunk, device),
        ):
            baseline = log_probabilities(inputs)
            batch_size = self.mutant_batch_size
            start = 0
            while start < length * aa_count:
                stop = min(start + batch_size, length * aa_count)
                chunk_size = stop - start
                mutants: dict[str, torch.Tensor] | None = None
                delta: torch.Tensor | None = None
                try:
                    mutants = {
                        key: value.expand(chunk_size, -1).clone()
                        for key, value in inputs.items()
                    }
                    mutations = torch.arange(start, stop, device=device)
                    mutant_positions = residue_ids[mutations // aa_count]
                    mutants["input_ids"][
                        torch.arange(chunk_size, device=device), mutant_positions
                    ] = self.aa_token_ids[mutations % aa_count]
                    delta = log_probabilities(mutants) - baseline
                    jacobian[start:stop] = delta
                    start = stop
                except torch.cuda.OutOfMemoryError as error:
                    del mutants
                    del delta
                    if batch_size == 1:
                        raise RuntimeError(
                            f"Jacobian inference ran out of CUDA memory for "
                            f"{length} residues even with one mutation per batch; "
                            f"the FP32 Jacobian alone requires "
                            f"{length * length * aa_count * aa_count * 4 / 2**30:.2f} GiB."
                        ) from error
                    batch_size = max(1, batch_size // 2)
                    logger.warning(
                        "%s: CUDA OOM; retrying Jacobian inference with "
                        "mutation batch size %d",
                        chain.record_id,
                        batch_size,
                    )
                    torch.cuda.empty_cache()

        logger.info(
            "Computed categorical Jacobian | record=%s | residues=%d | "
            "mutations=%d | final_batch_size=%d | precision=float32",
            chain.record_id,
            length,
            length * aa_count,
            batch_size,
        )
        return jacobian_to_contacts(jacobian.view(length, aa_count, length, aa_count))


def build_contact_jacobian_model(
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    *,
    mutant_batch_size: int = MUTANT_BATCH_SIZE,
) -> ContactJacobianModel:
    trunk = AutoModelForMaskedLM.from_pretrained(model_id, trust_remote_code=True)
    return ContactJacobianModel(trunk, tokenizer, mutant_batch_size=mutant_batch_size)
