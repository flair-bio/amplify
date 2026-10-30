"""Frozen-trunk embedding preparation for cached probes.

With ``embedding_cache`` enabled, every split's frozen-trunk features are
computed exactly once (and disk-cached for sequence-level splits), and the
probe trains on pooled embeddings instead of token batches. This module owns
that transformation: cache settings, cache keys, split embedding and external
test-set embedding. Token DataLoaders come from ``dataset.preparation``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from accelerate import Accelerator
from torch import nn

from modules.evaluate.src.artifacts.feature_cache import (
    EmbeddingCache,
    EmbeddingCacheConfig,
    embedding_cache_key,
    weights_fingerprint,
)
from modules.evaluate.src.dataset.preparation import PrepareConfig
from modules.evaluate.src.dataset.workspace import EvaluationWorkspace
from modules.evaluate.src.model.heads import EmbeddingProbeHead
from modules.evaluate.src.model.layerwise import resolve_layers
from modules.evaluate.src.predictors.embeddings import (
    EmbeddingBackend,
    build_embedding_dataloaders,
    validate_embedding_cache_compat,
)
from modules.evaluate.src.tasks import get_task


def layerwise_cache_config(
    config: PrepareConfig, layerwise_config: Any, model: nn.Module
) -> EmbeddingCacheConfig | None:
    """Cache settings matching live layerwise probe representations."""
    if layerwise_config is None:
        return None
    return config.embedding_cache.model_copy(
        update={
            "layers": resolve_layers(
                layerwise_config.layers, model.config.num_hidden_layers + 1
            ),
            "pooling": layerwise_config.pooling,
            "autocast_dtype": layerwise_config.autocast_dtype,
            "normalize_before_pooling": layerwise_config.normalize_before_pooling,
        }
    )


def embedding_test_cache_key(
    config: PrepareConfig,
    cache_config: EmbeddingCacheConfig,
    *,
    model_id: str,
    dataset_id: str,
    token_test_dataloader: Any,
    model_fingerprint: str,
    seed: int,
) -> str:
    """Identity of the test split's embeddings under *cache_config*."""
    return embedding_cache_key(
        model_id=model_id,
        dataset_id=dataset_id,
        split="test",
        max_length=config.max_length,
        random_truncate=config.random_truncate_by_split.get("test", False),
        seed=seed,
        pooling=cache_config.pooling,
        dtype=cache_config.dtype,
        layers=cache_config.resolved_layers,
        model_fingerprint=model_fingerprint,
        dataset_fingerprint=getattr(
            token_test_dataloader.dataset, "_fingerprint", None
        ),
        autocast_dtype=cache_config.autocast_dtype,
        normalize_before_pooling=cache_config.normalize_before_pooling,
        packed=config.packed,
        length_grouped_sampling=config.length_grouped_sampling,
    )


def apply_embedding_cache(
    config: PrepareConfig,
    workspace: EvaluationWorkspace,
    model: nn.Module,
    tokenizer: Any,
    train_dataloader: Any,
    val_dataloader: Any,
    test_dataloader: Any,
    refit_token_dataloaders: dict[str, Any],
    raw_test_dataloader: Any,
    accelerator: Accelerator,
    seed: int,
    num_labels: int,
    cache_config: EmbeddingCacheConfig | None = None,
) -> tuple[nn.Module, Any, Any, Any, dict[str, Any]]:
    """Replace token loaders and the full trunk with pooled-embedding loaders
    and a cached-embedding probe head.

    Only valid for a frozen trunk (``tune_probe``): a config validator rejects
    ``embedding_cache.enabled`` in other modes, since a trained trunk's output
    is not fixed and caching it would freeze stale features into every trial.
    """
    cache_config = cache_config or config.embedding_cache
    task = get_task(workspace.task_type)
    is_ragged = task.is_ragged
    validate_embedding_cache_compat(is_ragged, tokenizer, cache_config.pooling)
    hidden_size = model.config.hidden_size
    layers = cache_config.resolved_layers
    embedding_size = hidden_size * len(layers)
    model_id = workspace.model_repo_id or workspace.model_name
    dataset_id = workspace.dataset_repo_id or workspace.dataset_name
    cache = EmbeddingCache(workspace.embeddings_path, cache_config.max_cache_gb)
    # The original token-based test dataloader is kept for controls that need
    # real tokens or a re-embeddable trunk ("random_trunk", "scrambled_sequences",
    # "store_sequences"), which a pooled-embedding test loader cannot provide.
    token_test_dataloader = raw_test_dataloader
    model_fingerprint = weights_fingerprint(model)
    test_random_truncate = config.random_truncate_by_split.get("test", False)
    test_cache_key = embedding_test_cache_key(
        config,
        cache_config,
        model_id=model_id,
        dataset_id=dataset_id,
        token_test_dataloader=token_test_dataloader,
        model_fingerprint=model_fingerprint,
        seed=seed,
    )

    label_dtype = task.label_dtype

    embedding_dataloaders = build_embedding_dataloaders(
        model=model,
        splits={
            "train": (
                train_dataloader,
                config.random_truncate_by_split.get("train", True),
            ),
            "validation": (
                val_dataloader,
                config.random_truncate_by_split.get("validation", False),
            ),
            "test": (
                test_dataloader,
                config.random_truncate_by_split.get("test", False),
            ),
            **{
                split: (
                    dataloader,
                    config.random_truncate_by_split.get(split, False),
                )
                for split, dataloader in refit_token_dataloaders.items()
            },
        },
        accelerator=accelerator,
        cache=cache,
        cache_config=cache_config,
        model_id=model_id,
        dataset_id=dataset_id,
        max_length=config.max_length,
        seed=seed,
        batch_size=config.batch_size,
        extraction_batch_size=(None if config.packed else config.embedding_batch_size),
        embedding_size=embedding_size,
        label_dtype=label_dtype,
        is_ragged=is_ragged,
        model_fingerprint=model_fingerprint,
        length_grouped_sampling=config.length_grouped_sampling,
    )

    probe_model = EmbeddingProbeHead(embedding_size, num_labels, workspace.task_type)
    probe_model = probe_model.to(accelerator.device)
    refit_embedding_dataloaders = {}
    for split in refit_token_dataloaders:
        dataloader = embedding_dataloaders[split]
        if dataloader is None:
            raise ValueError(
                f"Could not build embeddings for final-refit split '{split}'."
            )
        refit_embedding_dataloaders[split] = dataloader
    embedding_extras = {
        "embedding_backend": EmbeddingBackend(
            trunk=model,
            token_test_dataloader=token_test_dataloader,
            token_train_dataloader=train_dataloader,
            token_val_dataloader=val_dataloader,
            train_random_truncate=config.random_truncate_by_split.get("train", True),
            validation_random_truncate=config.random_truncate_by_split.get(
                "validation", False
            ),
            extraction_batch_size=None
            if config.packed
            else config.embedding_batch_size,
            pooling=cache_config.pooling,
            layers=layers,
            dtype=cache_config.dtype,
            hidden_size=embedding_size,
            autocast_dtype=cache_config.autocast_dtype,
            normalize_before_pooling=cache_config.normalize_before_pooling,
            test_cache_key=test_cache_key,
            cache=cache,
            cache_policy=cache_config.cache_policy,
            model_id=model_id,
            dataset_id=dataset_id,
            max_length=config.max_length,
            test_random_truncate=test_random_truncate,
            length_grouped_sampling=config.length_grouped_sampling,
            test_dataset_fingerprint=getattr(
                token_test_dataloader.dataset, "_fingerprint", None
            ),
        ),
        "refit_dataloaders": refit_embedding_dataloaders,
    }
    return (
        probe_model,
        embedding_dataloaders["train"],
        embedding_dataloaders["validation"],
        embedding_dataloaders["test"],
        embedding_extras,
    )


def embed_external_test(
    config: PrepareConfig,
    backend: EmbeddingBackend,
    *,
    token_test_dataloader: Any,
    raw_test_dataloader: Any,
    external_id: str,
    task_is_ragged: bool,
    label_dtype: Any,
    cache_config: EmbeddingCacheConfig,
    accelerator: Accelerator,
    seed: int,
) -> tuple[Any, EmbeddingBackend]:
    """Embed an external test set with the fitted run's frozen trunk.

    Returns the pooled-embedding test loader and the backend that records the
    external set's cache identity. The trunk, pooling and layers are reused
    unchanged; only the cache key names the external dataset.
    """
    if backend.cache is None or backend.model_id is None:
        raise ValueError(
            "Embedding-cache external evaluation requires the cache and "
            "model ID recorded by the original Prepare step."
        )
    trunk = backend.trunk
    model_fingerprint = weights_fingerprint(trunk)
    embedding_dataloader = build_embedding_dataloaders(
        model=trunk,
        splits={
            "test": (
                token_test_dataloader,
                config.random_truncate_by_split.get("test", False),
            )
        },
        accelerator=accelerator,
        cache=backend.cache,
        cache_config=cache_config,
        model_id=backend.model_id,
        dataset_id=external_id,
        max_length=config.max_length,
        seed=seed,
        batch_size=config.batch_size,
        extraction_batch_size=(None if config.packed else config.embedding_batch_size),
        embedding_size=backend.hidden_size,
        label_dtype=label_dtype,
        is_ragged=task_is_ragged,
        model_fingerprint=model_fingerprint,
        length_grouped_sampling=config.length_grouped_sampling,
    )["test"]
    external_backend = replace(
        backend,
        token_test_dataloader=raw_test_dataloader,
        dataset_id=external_id,
        test_cache_key=embedding_test_cache_key(
            config,
            cache_config,
            model_id=backend.model_id,
            dataset_id=external_id,
            token_test_dataloader=raw_test_dataloader,
            model_fingerprint=model_fingerprint,
            seed=seed,
        ),
        test_dataset_fingerprint=getattr(
            raw_test_dataloader.dataset, "_fingerprint", None
        ),
    )
    return embedding_dataloader, external_backend
