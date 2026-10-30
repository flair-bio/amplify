"""Frozen-probe preparation: dataset, tokenization, model and embedding cache for one run.

This is the production owner of the Prepare stage for cached and live probes.
It was previously implemented inside ``steps/prepare.py``; the step now only
delegates here. Data loading and tokenization live in
``dataset.preparation``; frozen-trunk extraction and cache handling live in
``predictors.embedding_preparation``.
"""

from __future__ import annotations

import inspect
from dataclasses import replace
from typing import TYPE_CHECKING

from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import DatasetDict
from transformers import AutoTokenizer

from modules.evaluate.src.dataset.dataloader import ExternalTestSetConfig
from modules.evaluate.src.dataset.preparation import (
    PrepareConfig,
    build_refit_token_dataloaders,
    build_token_dataloaders,
    tokenize_split_dataset,
)
from modules.evaluate.src.dataset.resolution import (
    resolve_evaluation_data,
    resolve_external_data,
)
from modules.evaluate.src.dataset.workspace import EvaluationWorkspace
from modules.evaluate.src.model.layerwise import LayerwiseProbeModel
from modules.evaluate.src.predictors.embedding_preparation import (
    apply_embedding_cache,
    embed_external_test,
    layerwise_cache_config,
)
from modules.evaluate.src.predictors.embeddings import (
    validate_embedding_cache_compat,
)
from modules.evaluate.src.schemas.artifacts import PreparedArtifacts
from modules.evaluate.src.tasks import get_task
from modules.evaluate.src.utils.baselines import (
    BaselineName,
    baseline_context,
    external_baseline_context,
)

if TYPE_CHECKING:
    from modules.evaluate.src.config import LayerwiseConfig

# accelerate's logger adapter logs INFO/DEBUG only on the main process by
# default, preventing duplicate lines when running under distributed launch.
logger = get_logger(__name__)


def prepare_probe_artifacts(
    config: PrepareConfig,
    workspace: EvaluationWorkspace,
    *,
    seed: int = 710019,
    accelerator: Accelerator | None = None,
    layerwise_config: LayerwiseConfig | None = None,
    dataset: DatasetDict | None = None,
    baseline_names: list[BaselineName] | None = None,
) -> PreparedArtifacts:
    """Load task data and models, optionally replacing tokens with features.

    Layerwise mode attaches independent heads to the frozen trunk. When
    caching is enabled, its layer and pooling settings govern extraction,
    and heads are attached afterwards. Original token batches remain
    available for controls that change the input or trunk. A supplied
    raw *dataset* (e.g. a CV fold's roles) replaces loading and
    validation-split synthesis.
    """
    # The pipeline passes one shared instance; keep the fallback for direct
    # callers in tests and small standalone uses.
    if accelerator is None:
        accelerator = Accelerator()
    if config.packed and accelerator.device.type != "cuda":
        raise ValueError("Packed evaluation requires CUDA.")
    if config.max_tokens_per_batch is not None and accelerator.num_processes > 1:
        raise ValueError("max_tokens_per_batch supports a single process only.")

    logger.info("Running Prepare step...")

    logger.info("Setting up workspace directories...")
    workspace.setup_directories()

    model_id = workspace.model_repo_id or workspace.model_name
    dataset_id = workspace.dataset_repo_id or workspace.dataset_name

    # Load the tokenizer
    logger.info("Loading tokenizer for '%s'...", model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, **workspace.model_load_kwargs
    )

    # Load the evaluation dataset from HF hub or local cache
    logger.info("Resolving dataset '%s'...", dataset_id)
    dataset = resolve_evaluation_data(
        dataset_id,
        seed=seed,
        validation_fraction=config.validation_fraction,
        split=config.split,
        dataset=dataset,
        revision=workspace.config.dataset_revision,
        subset=workspace.config.dataset_subset,
    )
    shared_context = (
        baseline_context(
            workspace.config,
            dataset,
            baseline_names or [],
            sequence_column=config.sequence_column,
            label_column=config.label_column,
            id_column=config.id_column,
            split_roles=config.split.model_dump(),
        )
        if workspace.task_type
        in {"sequence_classification", "sequence_regression", "token_classification"}
        else None
    )

    task = get_task(workspace.task_type)
    num_labels = task.resolve_num_labels(workspace.num_labels, workspace.dataset_name)
    # Config-level validation can't see real label values, only the
    # declared contract; check the resolved num_labels against the
    # dataset's actual (pre-tokenization) label values here, before a
    # mismatch surfaces later as a cryptic index-out-of-bounds error.
    task.validate_num_labels(
        dataset, config.label_column, num_labels, workspace.dataset_name
    )

    # Let rank 0 populate the shared tokenization cache before other ranks use it
    logger.info("Tokenizing dataset...")
    dataset = tokenize_split_dataset(
        config,
        dataset,
        tokenizer,
        workspace.task_type,
        seed,
        accelerator,
    )
    if shared_context is not None and workspace.task_type == "token_classification":
        from modules.evaluate.src.artifacts.recovery import token_population

        shared_context["population"] = token_population(
            dataset["test"], config.label_column, config.id_column
        )

    logger.info("Building collator and dataloaders...")
    train_dataloader, val_dataloader, test_dataloader = build_token_dataloaders(
        config, dataset, tokenizer, workspace.task_type, seed
    )
    refit_token_dataloaders = build_refit_token_dataloaders(
        config, dataset, tokenizer, workspace.task_type, seed, accelerator
    )

    logger.info("Loading model '%s' for task '%s'...", model_id, workspace.task_type)

    # Get the task-specific model type (e.g. XXXforSequenceClassification) and build the model
    model = task.build_model(
        model_id,
        num_labels,
        {**workspace.model_kwargs, **workspace.model_load_kwargs},
    )
    if config.packed:
        parameters = inspect.signature(model.forward).parameters
        required = {"cu_seqlens", "max_seqlen"}
        if workspace.task_type in (
            "sequence_classification",
            "sequence_regression",
        ):
            required.add("num_sequences")
        if not required <= parameters.keys():
            raise ValueError(
                "Packed evaluation requires an AMPLIFY checkpoint re-uploaded "
                "with the current modeling_amplify.py."
            )
    # Keep the model unwrapped for tuning (so requires_grad can still be
    # configured before accelerator.prepare), but place it on the same
    # device that accelerator-prepared dataloaders will yield tensors on.
    model = model.to(accelerator.device)
    active_layerwise_config = (
        layerwise_config
        if layerwise_config is not None and layerwise_config.enabled
        else None
    )
    if active_layerwise_config is not None:
        validate_embedding_cache_compat(
            task.is_ragged, tokenizer, active_layerwise_config.pooling
        )
        if not config.embedding_cache.enabled:
            model = LayerwiseProbeModel(
                model,
                num_labels,
                workspace.task_type,
                active_layerwise_config,
                tokenizer=tokenizer,
            ).to(accelerator.device)

    # Leave the model raw/unwrapped here: tuning needs to set requires_grad on
    # it before wrapping, and accelerator.prepare() must only be called once on
    # the model. Random-trunk and input controls need original tokens even when
    # probes train on cached features, so preserve the token-based loaders here.
    raw_test_dataloader = test_dataloader
    train_dataloader, val_dataloader, test_dataloader = accelerator.prepare(
        train_dataloader, val_dataloader, test_dataloader
    )
    token_dataloaders = (
        {
            "train": train_dataloader,
            "validation": val_dataloader,
            "test": test_dataloader,
        }
        if active_layerwise_config is not None and config.embedding_cache.enabled
        else None
    )

    # Without caching, tuning/prediction receive token batches and run the model.
    # With caching, they receive cached features and a probe; the raw token
    # test loader remains available for controls that need to rerun the trunk.
    if config.embedding_cache.enabled:
        # Replace token batches with cached features; retain the extraction
        # backend so prediction can choose cached or token-based controls.
        (
            probe_model,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            embedding_extras,
        ) = apply_embedding_cache(
            config,
            workspace,
            model,
            tokenizer,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            refit_token_dataloaders,
            raw_test_dataloader,
            accelerator,
            seed,
            num_labels,
            # Match live probe representations and include those settings
            # in cache identity, rather than using generic cache defaults.
            cache_config=layerwise_cache_config(config, active_layerwise_config, model),
        )
        # Cache extraction needs the original task model; attach the
        # per-layer trainable heads only after its features are prepared.
        model = (
            LayerwiseProbeModel(
                model,
                num_labels,
                workspace.task_type,
                active_layerwise_config,
                tokenizer=tokenizer,
            ).to(accelerator.device)
            if active_layerwise_config is not None
            else probe_model
        )
    else:
        embedding_extras = {}

    return PreparedArtifacts(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        layerwise_token_dataloaders=token_dataloaders,
        refit_token_dataloaders=refit_token_dataloaders,
        refit_random_truncate_by_split={
            split: config.random_truncate_by_split.get(split, False)
            for split in refit_token_dataloaders
        },
        split=config.split.name,
        baseline_context=shared_context,
        **embedding_extras,
    )


def prepare_external_probe_artifacts(
    config: PrepareConfig,
    workspace: EvaluationWorkspace,
    prepared: PreparedArtifacts,
    external: ExternalTestSetConfig,
    *,
    accelerator: Accelerator,
    seed: int,
    layerwise_config: LayerwiseConfig | None = None,
) -> PreparedArtifacts:
    """Return *prepared* with its test loaders swapped for an external set.

    The fitted model, heads, and tokenizer are reused unchanged. Cached
    runs embed the external set once with the frozen trunk and the same
    pooling/layer settings, under a cache key naming the external dataset.
    """
    logger.info(
        "Preparing external test set '%s' (%s:%s)...",
        external.name,
        external.dataset_repo_id,
        external.split,
    )
    dataset = resolve_external_data(
        external.dataset_repo_id,
        external.split,
        revision=external.dataset_revision,
        subset=external.dataset_subset,
    )
    shared_context = (
        external_baseline_context(
            prepared.baseline_context,
            workspace.config,
            dataset,
            dataset_id=external.dataset_repo_id,
            split=external.split,
            dataset_revision=external.dataset_revision,
            dataset_subset=external.dataset_subset,
            sequence_column=config.sequence_column,
            label_column=config.label_column,
            id_column=config.id_column,
        )
        if prepared.baseline_context
        else None
    )

    task = get_task(workspace.task_type)
    num_labels = task.resolve_num_labels(workspace.num_labels, workspace.dataset_name)
    task.validate_num_labels(dataset, config.label_column, num_labels, external.name)
    dataset = tokenize_split_dataset(
        config,
        dataset,
        prepared.tokenizer,
        workspace.task_type,
        seed,
        accelerator,
    )
    if shared_context is not None and workspace.task_type == "token_classification":
        from modules.evaluate.src.artifacts.recovery import token_population

        shared_context["population"] = token_population(
            dataset["test"], config.label_column, config.id_column
        )
    _, _, raw_test_dataloader = build_token_dataloaders(
        config, dataset, prepared.tokenizer, workspace.task_type, seed
    )
    if raw_test_dataloader is None:
        raise ValueError(
            f"External test set '{external.name}' did not produce a test dataloader."
        )
    token_test_dataloader = accelerator.prepare(raw_test_dataloader)

    backend = prepared.embedding_backend
    token_dataloaders = prepared.layerwise_token_dataloaders
    if token_dataloaders is not None:
        token_dataloaders = {**token_dataloaders, "test": token_test_dataloader}
    if backend is None:
        return replace(
            prepared,
            test_dataloader=token_test_dataloader,
            layerwise_token_dataloaders=token_dataloaders,
            baseline_context=shared_context,
        )

    cache_config = (
        layerwise_cache_config(
            config,
            layerwise_config if layerwise_config and layerwise_config.enabled else None,
            backend.trunk,
        )
        or config.embedding_cache
    )
    embedding_dataloader, external_backend = embed_external_test(
        config,
        backend,
        token_test_dataloader=token_test_dataloader,
        raw_test_dataloader=raw_test_dataloader,
        external_id=external.dataset_repo_id,
        task_is_ragged=task.is_ragged,
        label_dtype=task.label_dtype,
        cache_config=cache_config,
        accelerator=accelerator,
        seed=seed,
    )
    return replace(
        prepared,
        test_dataloader=embedding_dataloader,
        embedding_backend=external_backend,
        layerwise_token_dataloaders=token_dataloaders,
        baseline_context=shared_context,
    )
