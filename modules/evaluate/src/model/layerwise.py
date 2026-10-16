"""Independent linear probes over cached features or a shared frozen trunk."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Mapping

import torch
from accelerate import Accelerator
from torch import nn

from modules.evaluate.src.dataset.collator import unpack_packed
from modules.evaluate.src.model.heads import EmbeddingProbeHead
from modules.evaluate.src.tasks.token_classification import IGNORE_INDEX
from modules.evaluate.src.utils.embed import (
    AUTOCAST_DTYPES,
    autocast_context,
    pool_hidden_states,
    residue_attention_mask,
)

if TYPE_CHECKING:
    from modules.evaluate.src.config import LayerwiseConfig

# Floor for standard deviations so constant features or targets don't divide by zero.
MIN_STANDARD_DEVIATION = 1e-6


def resolve_layers(layers: str | list[int], count: int) -> list[int]:
    """Resolve hidden-state tuple positions, preserving the requested order.

    Position 0 is the embedding output, not the first transformer block.
    Negative indices count from the end; ``all`` selects every position.
    Reject empty, out-of-range, or duplicate positions, including aliases such
    as -1 and the final positive index.
    """
    resolved = (
        list(range(count))
        if isinstance(layers, str)
        else [index % count if -count <= index < count else index for index in layers]
    )
    if not resolved or any(index < 0 or index >= count for index in resolved):
        raise ValueError(
            f"Requested layers {layers} outside the {count} hidden states."
        )
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"Requested layers {layers} refer to the same hidden state.")
    return resolved


class LayerwiseProbeModel(nn.Module):
    """Compare depths using independent heads over fixed representations.

    Live batches share one trunk pass across all layers and head conditions;
    cached batches bypass the trunk. Only heads train, and the trunk stays in
    eval mode. Features (optionally) and regression targets are standardized
    with train-split statistics so every depth trains at a comparable scale.
    """

    def __init__(
        self,
        task_model: Any,
        num_labels: int,
        task_type: str,
        config: LayerwiseConfig,
        tokenizer: Any,
    ) -> None:
        super().__init__()
        self.trunk: nn.Module = task_model.base_model
        # Every probe reads from the same frozen representation at its depth.
        self.trunk.requires_grad_(False)
        self.trunk.eval()
        self.num_hidden_states = task_model.config.num_hidden_layers + 1
        self.layers = resolve_layers(config.layers, self.num_hidden_states)
        self.embedding_size = task_model.config.hidden_size
        self.heads = nn.ModuleDict(
            {
                str(layer): EmbeddingProbeHead(
                    self.embedding_size, num_labels, task_type
                )
                for layer in self.layers
            }
        )
        self.num_labels = num_labels
        self.task_type = task_type
        self.config = config
        self.tokenizer = tokenizer
        # Identity until fit_standardization runs.
        self.register_buffer(
            "feature_mean", torch.zeros(len(self.layers), self.embedding_size)
        )
        self.register_buffer(
            "feature_std", torch.ones(len(self.layers), self.embedding_size)
        )
        self.register_buffer("target_mean", torch.zeros(()))
        self.register_buffer("target_std", torch.ones(()))

    @torch.no_grad()
    def fit_standardization(self, dataloader: Any, accelerator: Accelerator) -> None:
        """Set per-layer feature and regression-target statistics from *dataloader*.

        Token features count labeled residues only, so padding and special
        tokens don't affect the statistics. Feature statistics are skipped when
        ``config.standardize_features`` is False.
        """
        self.eval()
        token_level = self.task_type == "token_classification"
        regression = self.task_type == "sequence_regression"
        standardize = self.config.standardize_features
        if not standardize and not regression:
            return
        count = torch.zeros(1, dtype=torch.float64)
        sums = torch.zeros(len(self.layers), self.embedding_size, dtype=torch.float64)
        squares = torch.zeros_like(sums)
        target_sums = torch.zeros(3, dtype=torch.float64)
        for raw_batch in dataloader:
            labels = raw_batch["labels"]
            if standardize:
                batch = {
                    key: value
                    for key, value in raw_batch.items()
                    if key not in ("id", "labels")
                }
                features_by_layer, _, _ = self._extract_features(**batch)
                for position, layer in enumerate(self.layers):
                    rows = features_by_layer[layer]
                    if token_level:
                        rows = rows[labels != IGNORE_INDEX]
                    rows = rows.reshape(-1, self.embedding_size).float()
                    # Per-batch float32 sums; float64 totals stay on CPU (MPS lacks float64).
                    sums[position] += rows.sum(dim=0).cpu().double()
                    squares[position] += rows.square().sum(dim=0).cpu().double()
                # Every layer contributes the same number of rows.
                count += rows.shape[0]
            if regression:
                targets = labels.cpu().double()
                target_sums += torch.tensor(
                    [
                        targets.numel(),
                        targets.sum().item(),
                        targets.square().sum().item(),
                    ],
                    dtype=torch.float64,
                )
        totals = torch.cat([count, sums.flatten(), squares.flatten(), target_sums])
        if accelerator.num_processes > 1:
            totals = accelerator.reduce(totals.to(accelerator.device), "sum").cpu()
        count, totals = totals[0], totals[1:]
        sums, squares, target_sums = totals.split([sums.numel(), sums.numel(), 3])
        mean, std = _mean_std(sums, squares, count)
        if standardize:
            self.feature_mean.copy_(mean.view_as(self.feature_mean))
            self.feature_std.copy_(std.view_as(self.feature_std))
        if regression:
            target_mean, target_std = _mean_std(
                target_sums[1], target_sums[2], target_sums[0]
            )
            self.target_mean.copy_(target_mean)
            self.target_std.copy_(target_std)

    def train(self, mode: bool = True) -> LayerwiseProbeModel:
        """Train the probe heads without putting the frozen trunk in train mode."""
        super().train(mode)
        self.trunk.eval()
        return self

    def forward(self, **batch: Any) -> dict[int, Any]:
        """Return trained-head outputs keyed by resolved hidden-state position."""
        return self.forward_conditions({"trained": self.heads}, **batch)["trained"]

    def _trunk_states(self, batch: dict[str, Any]) -> dict[int, torch.Tensor]:
        """Run the trunk once and return the requested hidden states.

        Uses the same hidden-state tuple as embedding-cache extraction, so live
        and cached probes read identical features.
        """
        hidden_states = self.trunk(**batch, output_hidden_states=True).hidden_states
        if len(hidden_states) <= max(self.layers):
            raise ValueError("Trunk returned fewer hidden states than configured.")
        return {layer: hidden_states[layer] for layer in self.layers}

    def _pool_packed(
        self, states: dict[int, torch.Tensor], batch: dict[str, Any]
    ) -> dict[int, torch.Tensor]:
        """Pool packed ``(1, T, width)`` states per sequence in float32, like padded pooling."""
        offsets = batch["cu_seqlens"].long()
        num_sequences = batch["num_sequences"]
        starts = offsets[:num_sequences]
        lengths = offsets[1 : num_sequences + 1] - starts
        total = int(offsets[num_sequences])
        input_ids = batch["input_ids"][0, :total]
        valid = residue_attention_mask(
            input_ids, torch.ones_like(input_ids), self.tokenizer
        )
        segments = torch.repeat_interleave(
            torch.arange(num_sequences, device=lengths.device), lengths
        )[valid]
        counts = torch.bincount(segments, minlength=num_sequences).clamp(min=1)
        pooled: dict[int, torch.Tensor] = {}
        for layer, state in states.items():
            flat = state[0, :total].float()
            if self.config.normalize_before_pooling:
                flat = nn.functional.normalize(flat, p=2, dim=-1)
            if self.config.pooling == "cls":
                pooled[layer] = flat[starts]
                continue
            sums = flat.new_zeros((num_sequences, flat.shape[-1])).index_add_(
                0, segments, flat[valid]
            )
            pooled[layer] = sums / counts[:, None]
        return pooled

    def _extract_features(
        self, **batch: Any
    ) -> tuple[
        dict[int, torch.Tensor],
        torch.Tensor | None,
        Callable[[torch.Tensor], torch.Tensor] | None,
    ]:
        """Return per-layer features, labels, and an optional logit unpacker.

        Cached inputs are ``(rows, hidden_size * num_layers)`` in ``self.layers``
        order. Live sequence features are pooled to ``(batch, width)``; live
        token features keep their sequence dimension. Packed token features stay
        flat, so the unpacker pads logits back to sequence rows and token labels
        are unpacked here.
        """
        unpack: Callable[[torch.Tensor], torch.Tensor] | None = None
        batch.pop("id", None)
        labels = batch.pop("labels", None)
        embeddings = batch.pop("inputs_embeds", None)
        if embeddings is not None:
            # Cached features concatenate layers along the width dimension.
            if embeddings.ndim != 2 or embeddings.shape[
                -1
            ] != self.embedding_size * len(self.layers):
                raise ValueError("Cached layerwise embeddings have the wrong width.")
            features_by_layer = dict(
                zip(self.layers, embeddings.split(self.embedding_size, dim=-1))
            )
        else:
            autocast = autocast_context(
                next(self.trunk.parameters()).device,
                AUTOCAST_DTYPES[self.config.autocast_dtype],
            )
            with torch.no_grad(), autocast:
                states = self._trunk_states(batch)
            token_level = self.task_type == "token_classification"
            if "cu_seqlens" in batch:
                offsets = batch["cu_seqlens"]
                num_sequences = batch["num_sequences"]
                if token_level:
                    features_by_layer = states
                    unpack = lambda logits: unpack_packed(  # noqa: E731
                        logits, offsets, num_sequences
                    )
                    if labels is not None:
                        labels = unpack_packed(
                            labels, offsets, num_sequences, IGNORE_INDEX
                        )
                else:
                    features_by_layer = self._pool_packed(states, batch)
            elif token_level:
                features_by_layer = states
            else:
                mask = residue_attention_mask(
                    batch["input_ids"], batch["attention_mask"], self.tokenizer
                )
                features_by_layer = {
                    layer: pool_hidden_states(
                        state,
                        mask,
                        self.config.pooling,
                        normalize_before_pooling=self.config.normalize_before_pooling,
                    )
                    for layer, state in states.items()
                }
        return features_by_layer, labels, unpack

    def forward_conditions(
        self, head_sets: Mapping[str, nn.ModuleDict], **batch: Any
    ) -> dict[str, dict[int, Any]]:
        """Apply named head sets to the same features without rerunning the trunk.

        Each head set must contain string keys for every resolved layer.
        Results are keyed by condition, then integer layer position; optional
        labels produce a loss as well as logits for each head.
        """
        features_by_layer, labels, unpack = self._extract_features(**batch)
        outputs: dict[str, dict[int, Any]] = {name: {} for name in head_sets}
        for position, layer in enumerate(self.layers):
            features = (
                features_by_layer[layer] - self.feature_mean[position]
            ) / self.feature_std[position]
            for name, heads in head_sets.items():
                outputs[name][layer] = self._apply_head(
                    heads[str(layer)], features, labels, unpack
                )
        return outputs

    def _apply_head(
        self,
        head: nn.Module,
        features: torch.Tensor,
        labels: torch.Tensor | None,
        unpack: Callable[[torch.Tensor], torch.Tensor] | None,
    ) -> Any:
        """Return one head's output; token and regression losses are computed here."""
        if self.task_type == "token_classification":
            output = head(inputs_embeds=features)
            logits = output.logits if unpack is None else unpack(output.logits)
            # The head's loss is row-wise; padded token rows need ignore_index.
            loss = (
                None
                if labels is None
                else nn.functional.cross_entropy(
                    logits.reshape(-1, self.num_labels),
                    labels.reshape(-1),
                    ignore_index=IGNORE_INDEX,
                )
            )
            return type(output)(loss=loss, logits=logits)
        if self.task_type == "sequence_regression":
            output = head(inputs_embeds=features)
            scaled = output.logits
            loss = (
                None
                if labels is None
                else nn.functional.mse_loss(
                    scaled.squeeze(-1),
                    ((labels - self.target_mean) / self.target_std).to(scaled.dtype),
                )
            )
            return type(output)(
                loss=loss, logits=scaled * self.target_std + self.target_mean
            )
        return head(inputs_embeds=features, labels=labels)


def _mean_std(
    sums: torch.Tensor, squares: torch.Tensor, count: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Population mean and floored standard deviation from running sums."""
    total = count.clamp(min=1)
    mean = sums / total
    variance = (squares / total - mean.square()).clamp(min=0)
    return mean.float(), variance.sqrt().clamp(min=MIN_STANDARD_DEVIATION).float()
