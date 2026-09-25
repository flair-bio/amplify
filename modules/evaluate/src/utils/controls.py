"""
Build predictions for every control condition :class:`~modules.evaluate.src.
steps.predict.PredictStep` was configured to compare the trained model
against: randomly re-initialized model variants, non-model baselines
(majority-class/train-mean), and corrupted-input (scrambled-sequence)
inference. Kept separate from ``predict.py`` since control-building is an
orthogonal axis to task type (it applies uniformly across
sequence_classification/sequence_regression/token_classification), unlike
:class:`~modules.evaluate.src.tasks.base.TaskHandler`, which owns what
varies *by* task type (e.g. its ``scramble_labels``).
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Callable, Literal

import polars as pl
import torch
from accelerate import Accelerator
from sklearn.base import BaseEstimator
from torch.utils.data import DataLoader, DistributedSampler

from modules.evaluate.src.dataset.workspace import EvaluationWorkspace
from modules.evaluate.src.model.heads import (
    EmbeddingProbeHead,
    run_classical_inference,
)
from modules.evaluate.src.schemas.artifacts import PreparedArtifacts
from modules.evaluate.src.utils.embed import (
    EmbeddingDataset,
    embedding_collate_fn,
    extract_embeddings,
    find_embedding_dataset,
)
from modules.evaluate.src.utils.inference import pad_for_gather, run_inference
from modules.evaluate.src.tasks import get_task

logger = logging.getLogger(__name__)


UNTRAINED_CONTROL = "untrained"
RANDOM_TRUNK_CONTROL = "random_trunk"
RANDOM_HEAD_CONTROL = "random_head"
RANDOM_BOTH_CONTROL = "random_both"
SCRAMBLED_LABELS_CONTROL = "scrambled_labels"
SCRAMBLED_SEQUENCES_CONTROL = "scrambled_sequences"
MAJORITY_CLASS_CONTROL = "majority_class"
TRAIN_MEAN_CONTROL = "train_mean"
SEPARATION_PRIOR_CONTROL = "separation_prior"

ControlCondition = Literal[
    "untrained",
    "random_trunk",
    "random_head",
    "random_both",
    "scrambled_labels",
    "scrambled_sequences",
    "majority_class",
    "train_mean",
    "separation_prior",
]


RANDOMIZED_MODEL_CONTROLS = frozenset(
    {
        UNTRAINED_CONTROL,
        RANDOM_TRUNK_CONTROL,
        RANDOM_HEAD_CONTROL,
        RANDOM_BOTH_CONTROL,
    }
)
RANDOMIZED_TRUNK_CONTROLS = frozenset({RANDOM_TRUNK_CONTROL, RANDOM_BOTH_CONTROL})
RANDOMIZED_HEAD_CONTROLS = frozenset(
    {
        UNTRAINED_CONTROL,
        RANDOM_HEAD_CONTROL,
        RANDOM_BOTH_CONTROL,
    }
)
NO_INFERENCE_CONTROLS = frozenset({SCRAMBLED_LABELS_CONTROL})


class ControlBuilder:
    """Builds predictions for every model-based/non-model-baseline control
    condition. Expensive trunk features are reused through the embedding cache;
    predictions are always produced by the current head in memory."""

    def __init__(self, collect_probabilities: bool) -> None:
        self.collect_probabilities = collect_probabilities

    def build(
        self,
        control: str,
        workspace: EvaluationWorkspace,
        prepared: PreparedArtifacts,
        predictions_df: pl.DataFrame,
        accelerator: Accelerator,
        seed: int,
    ) -> tuple[pl.DataFrame, Path | None]:
        """Dispatch to the right control-building strategy.

        The returned path is always ``None``. Control predictions depend on
        the fitted head and are deliberately not cached; only trunk
        embeddings are reusable across runs.
        """
        if control == MAJORITY_CLASS_CONTROL:
            return (
                self._build_majority_class_control(
                    workspace, prepared, predictions_df, accelerator
                ),
                None,
            )
        if control == TRAIN_MEAN_CONTROL:
            return (
                self._build_train_mean_control(
                    workspace, prepared, predictions_df, accelerator
                ),
                None,
            )
        if control == SEPARATION_PRIOR_CONTROL:
            return (
                self._build_separation_prior_control(
                    workspace, prepared, predictions_df
                ),
                None,
            )
        return self._get_or_build_control(
            control, workspace, prepared, predictions_df, accelerator, seed
        )

    @staticmethod
    def _gather_train_labels(
        prepared: PreparedArtifacts, accelerator: Accelerator
    ) -> torch.Tensor:
        if prepared.train_dataloader is None:
            raise ValueError("A train_dataloader is required for a train baseline.")

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

    @staticmethod
    def _build_majority_class_control(
        workspace: EvaluationWorkspace,
        prepared: PreparedArtifacts,
        predictions_df: pl.DataFrame,
        accelerator: Accelerator,
    ) -> pl.DataFrame:
        """Build a non-model baseline from training-split class frequencies.

        The control deliberately learns only the most common training class,
        never the test labels. Its probability rows preserve the empirical
        training prior, which is constant across examples -- so its ROC-AUC is
        0.5 by construction, i.e. the chance baseline rather than a measured one.
        """
        if workspace.task_type == "sequence_regression":
            raise ValueError(
                "control='majority_class' is only defined for classification "
                "tasks; use a train-mean regression baseline instead."
            )
        labels = ControlBuilder._gather_train_labels(prepared, accelerator)

        model = accelerator.unwrap_model(prepared.model)
        num_labels = int(getattr(model, "num_labels"))
        counts = torch.bincount(labels.to(torch.long), minlength=num_labels)
        majority_class = int(counts.argmax().item())
        class_probabilities = (counts / counts.sum()).tolist()

        predictions = predictions_df["prediction"].to_list()
        baseline_predictions = [
            (
                [majority_class] * len(prediction)
                if isinstance(prediction, list)
                else majority_class
            )
            for prediction in predictions
        ]
        data: dict[str, list[Any]] = {
            "id": predictions_df["id"].to_list(),
            "prediction": baseline_predictions,
        }
        if "probability" in predictions_df.columns:
            data["probability"] = [
                (
                    [class_probabilities] * len(prediction)
                    if isinstance(prediction, list)
                    else class_probabilities
                )
                for prediction in baseline_predictions
            ]
        return pl.DataFrame(data)

    @staticmethod
    def _build_train_mean_control(
        workspace: EvaluationWorkspace,
        prepared: PreparedArtifacts,
        predictions_df: pl.DataFrame,
        accelerator: Accelerator,
    ) -> pl.DataFrame:
        """Build a non-model baseline that predicts the train split's mean label.

        The regression counterpart to ``majority_class``: a constant
        predictor that learns only the training label mean, never the test
        labels themselves.
        """
        if workspace.task_type != "sequence_regression":
            raise ValueError(
                "control='train_mean' is only defined for sequence_regression "
                "tasks; use a majority_class baseline for classification tasks."
            )
        labels = ControlBuilder._gather_train_labels(prepared, accelerator)
        train_mean = labels.to(torch.float32).mean().item()

        return pl.DataFrame(
            {
                "id": predictions_df["id"].to_list(),
                "prediction": [train_mean] * predictions_df.height,
            }
        )

    @staticmethod
    def _build_separation_prior_control(
        workspace: EvaluationWorkspace,
        prepared: PreparedArtifacts,
        predictions_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """Non-model contact_prediction baseline: predicted contact
        probability is a pure function of tokenized residue separation,
        estimated from the train split's empirical contact frequency per
        separation. Reuses the trained model's own prediction rows only for
        their ``separation`` values (not their scores), so no forward pass
        is needed here, same as ``majority_class``/``train_mean``.
        """
        if workspace.task_type not in {"contact_prediction", "categorical_jacobian"}:
            raise ValueError(
                "control='separation_prior' is only defined for "
                "task_type in {'contact_prediction', 'categorical_jacobian'}."
            )
        if prepared.train_dataloader is None:
            raise ValueError(
                "A train_dataloader is required for the separation_prior control."
            )
        max_sep = 1
        for batch in prepared.train_dataloader:
            labels = batch["labels"]
            max_sep = max(max_sep, labels.shape[-1] - 1)
        prior = ControlBuilder._compute_separation_prior(
            prepared.train_dataloader, max_sep
        )

        predictions = predictions_df["prediction"].to_list()
        control_predictions: list[Any] = []
        for row in predictions:
            if isinstance(row, dict):
                separations = [int(sep) for sep in row["separations"]]
                control_predictions.append(
                    {
                        "scores": [prior[min(sep, max_sep)] for sep in separations],
                        "separations": separations,
                    }
                )
                continue
            new_row = [row[0]]
            for _score, sep in row[1:]:
                new_row.append((prior[min(int(sep), max_sep)], sep))
            control_predictions.append(new_row)
        return pl.DataFrame(
            {
                "id": predictions_df["id"].to_list(),
                "prediction": control_predictions,
            }
        )

    @staticmethod
    def _compute_separation_prior(train_dataloader: Any, max_sep: int) -> list[float]:
        """Empirical P(contact | separation) from every scored train pair,
        bucketed by tokenized residue separation (capped at ``max_sep``)."""
        counts = torch.zeros(max_sep + 1, dtype=torch.float64)
        positives = torch.zeros(max_sep + 1, dtype=torch.float64)
        for batch in train_dataloader:
            labels = batch["labels"].detach().cpu()
            batch_size, seq_len, _ = labels.shape
            idx = torch.arange(seq_len)
            sep = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs().clamp(max=max_sep)
            sep = sep.unsqueeze(0).expand(batch_size, -1, -1)
            valid = labels != -100
            counts.scatter_add_(
                0, sep[valid].long(), torch.ones(int(valid.sum()), dtype=torch.float64)
            )
            positives.scatter_add_(0, sep[valid].long(), labels[valid].double())
        return (positives / counts.clamp(min=1)).tolist()

    def _get_or_build_control(
        self,
        control: str,
        workspace: EvaluationWorkspace,
        prepared: PreparedArtifacts,
        predictions_df: pl.DataFrame,
        accelerator: Accelerator,
        seed: int,
    ) -> tuple[pl.DataFrame, Path | None]:
        """Build a model-based control from the current head and embeddings."""
        is_scrambled_sequences = control == SCRAMBLED_SEQUENCES_CONTROL
        using_embeddings = prepared.embedding_backend is not None
        if using_embeddings:
            ids, predictions, probabilities, sequences = (
                self._run_embedding_control_inference(
                    control, workspace, prepared, accelerator, seed
                )
            )
        else:
            model = self._build_control_model(
                control, workspace, prepared, accelerator, seed
            )
            batch_transform = (
                self._make_sequence_scrambler(prepared.tokenizer, seed)
                if is_scrambled_sequences
                else None
            )
            ids, predictions, _, probabilities, sequences = run_inference(
                model,
                prepared.test_dataloader,
                workspace.task_type,
                accelerator,
                collect_probabilities=self.collect_probabilities,
                batch_transform=batch_transform,
                tokenizer=prepared.tokenizer if is_scrambled_sequences else None,
            )
        control_data: dict[str, list[Any]] = {"id": ids, "prediction": predictions}
        if probabilities is not None:
            control_data["probability"] = probabilities
        if sequences is not None:
            control_data["sequence"] = sequences

        if using_embeddings and get_task(workspace.task_type).is_ragged:
            # Ragged embedding-cache rows repeat the same id once per valid
            # token (see _run_embedding_control_inference), so join(on="id")
            # would cross-join every id's duplicates instead of aligning row
            # by row. Both sides iterate the same underlying dataset in the
            # same fixed order, so align positionally instead, with an
            # explicit order check since a silent misalignment here would
            # corrupt every downstream metric.
            control_ids = pl.Series("id", ids)
            if (
                control_ids.len() != predictions_df.height
                or (control_ids != predictions_df["id"]).any()
            ):
                raise RuntimeError(
                    f"Control '{control}' produced {control_ids.len()} rows in "
                    f"a different order/count than the {predictions_df.height} "
                    "test rows; ragged controls must iterate the test set in "
                    "the same order as the trained model's predictions."
                )
            aligned = predictions_df.select("id").with_columns(
                pl.DataFrame(control_data).drop("id")
            )
            return aligned, None

        aligned = predictions_df.select("id").join(
            pl.DataFrame(control_data), on="id", how="left"
        )
        if aligned["prediction"].null_count() > 0:
            raise RuntimeError(
                f"Control '{control}' predictions could not be aligned to all "
                "test example ids; check that the control and test dataloaders "
                "use the same ids."
            )

        return aligned, None

    def _run_embedding_control_inference(
        self,
        control: str,
        workspace: EvaluationWorkspace,
        prepared: PreparedArtifacts,
        accelerator: Accelerator,
        seed: int,
    ) -> tuple[list[Any], list[Any], list[Any] | None, list[str] | None]:
        """Rebuild a control's predictions when
        steps.prepare.embedding_cache is enabled: prepared.test_dataloader
        then yields pooled embeddings (not tokens), and prepared.model isn't
        a full HF model with a `.base_model`/`._init_weights` to
        reinitialize the way :meth:`_build_control_model` does.

        The trunk is always frozen/pretrained for embedding-cache runs (a
        cross-step validator in EvaluationConfig requires tune_probe mode),
        so "untrained"'s trunk is identical to the trained model's trunk --
        only the head differs.
        """
        torch.manual_seed(seed)
        assert prepared.embedding_backend is not None
        backend = prepared.embedding_backend
        is_ragged = get_task(workspace.task_type).is_ragged
        # A fitted sklearn/xgboost estimator is never DDP-wrapped (classical
        # heads skip accelerator.prepare() entirely), and isn't a
        # torch.nn.Module, so accelerator.unwrap_model() would raise.
        is_classical_head = isinstance(prepared.model, BaseEstimator)
        trained_head = (
            prepared.model
            if is_classical_head
            else accelerator.unwrap_model(prepared.model)
        )
        needs_fresh_head = control in RANDOMIZED_HEAD_CONTROLS
        if needs_fresh_head and not isinstance(trained_head, EmbeddingProbeHead):
            raise ValueError(
                f"control='{control}' requires a randomly re-initializable "
                "head, which isn't defined for a classical (non-'mlp') "
                "head_type. Use 'random_trunk', 'scrambled_sequences', or "
                "'scrambled_labels' instead (this should have been caught "
                "by EvaluationConfig validation)."
            )

        if control in RANDOMIZED_TRUNK_CONTROLS:
            trained_trunk = accelerator.unwrap_model(backend.trunk)
            trunk = type(trained_trunk)(copy.deepcopy(trained_trunk.config)).to(
                next(trained_trunk.parameters()).device
            )
            trunk.eval()
        else:
            # "untrained"/"random_head": trunk unchanged, only the head
            # differs. "scrambled_sequences": trunk unchanged, only the
            # (scrambled) input differs.
            trunk = backend.trunk

        is_scrambled_sequences = control == SCRAMBLED_SEQUENCES_CONTROL
        embeddings, ids, labels = self._get_or_build_control_embeddings(
            control,
            workspace,
            prepared,
            accelerator,
            seed,
            trunk,
            is_scrambled_sequences,
        )
        sequences = (
            self._decode_scrambled_sequences(prepared, accelerator, seed)
            if is_scrambled_sequences
            else None
        )

        if needs_fresh_head:
            head_model = EmbeddingProbeHead(
                trained_head.embedding_size,
                trained_head.num_labels,
                workspace.task_type,
                probe_hidden_size=trained_head.probe_hidden_size,
            ).to(accelerator.device)
        else:
            head_model = trained_head

        label_dtype = (
            torch.float if workspace.task_type == "sequence_regression" else torch.long
        )
        dataset = EmbeddingDataset(embeddings, ids, labels.to(label_dtype))
        if prepared.test_dataloader is None:
            raise ValueError("A test dataloader is required for local inference.")
        batch_size = prepared.test_dataloader.batch_size or 16

        if is_classical_head:
            # The estimator isn't distributed, so every rank runs the same
            # full (unsharded) data through it locally instead of the
            # DistributedSampler + gather_for_metrics path below.
            num_labels = get_task(workspace.task_type).resolve_num_labels(
                workspace.num_labels, workspace.dataset_name
            )
            batches = (
                {
                    "inputs_embeds": dataset.embeddings[start : start + batch_size],
                    "labels": dataset.labels[start : start + batch_size],
                    "id": dataset.ids[start : start + batch_size],
                }
                for start in range(0, len(dataset), batch_size)
            )
            control_ids, predictions, _, probabilities = run_classical_inference(
                head_model,
                batches,
                workspace.task_type,
                num_labels,
                collect_probabilities=self.collect_probabilities,
            )
            return control_ids, predictions, probabilities, sequences

        if is_ragged:
            # Ragged/token-level tasks have many rows sharing the same
            # sequence id (one row per valid token), so the id-based dedup
            # _run_local_head_inference uses to strip DistributedSampler
            # padding would collapse them all down to one row per id. Every
            # rank already has the full (unsharded) dataset here, so just
            # run it locally with no sampler/gather instead.
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=lambda batch: embedding_collate_fn(batch, label_dtype),
            )
            control_ids, predictions, probabilities = self._run_local_head_inference(
                head_model,
                dataloader,
                workspace.task_type,
                self.collect_probabilities,
                accelerator,
                gather_across_processes=False,
            )
            # Match run_local_embedding_inference's per-row shape for ragged
            # tasks: each row is one valid token wrapped in a single-element
            # list (not a bare scalar), which is what ScoreStep's
            # flatten_for_metrics (itertools.chain.from_iterable) expects.
            predictions = [[prediction] for prediction in predictions]
            return control_ids, predictions, probabilities, sequences

        # Not accelerator.prepare()'d: extract_embeddings() above already
        # gathered the full test set identically onto every rank, so a
        # second accelerator.gather_for_metrics() (as run_inference() would
        # do on a distributed-sampled dataloader) would duplicate every
        # prediction once per rank instead of returning it once.
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=DistributedSampler(
                dataset,
                num_replicas=accelerator.num_processes,
                rank=accelerator.process_index,
                shuffle=False,
            ),
            collate_fn=lambda batch: embedding_collate_fn(batch, label_dtype),
        )
        control_ids, predictions, probabilities = self._run_local_head_inference(
            head_model,
            dataloader,
            workspace.task_type,
            self.collect_probabilities,
            accelerator,
        )
        return control_ids, predictions, probabilities, sequences

    def _get_or_build_control_embeddings(
        self,
        control: str,
        workspace: EvaluationWorkspace,
        prepared: PreparedArtifacts,
        accelerator: Accelerator,
        seed: int,
        trunk: Any,
        is_scrambled_sequences: bool,
    ) -> tuple[torch.Tensor, list[Any], torch.Tensor]:
        assert prepared.embedding_backend is not None
        backend = prepared.embedding_backend
        is_ragged = get_task(workspace.task_type).is_ragged
        input_transform = (
            f"scramble_sequences:v1:seed={seed}" if is_scrambled_sequences else None
        )
        cache_key = backend.test_embedding_key()
        if control in RANDOMIZED_TRUNK_CONTROLS or is_scrambled_sequences:
            cache_key = backend.test_embedding_key(
                trunk=trunk,
                seed=seed,
                input_transform=input_transform,
            )

        cache = backend.cache
        if (
            cache is not None
            and cache_key is not None
            and backend.cache_policy == "reuse"
            and cache.exists("test", cache_key)
        ):
            if accelerator.is_main_process:
                logger.info(
                    "Loading cached '%s' control embeddings from disk...", control
                )
            embeddings, ids, labels = cache.read("test", cache_key)
            return embeddings, ids, labels

        cached_dataset = find_embedding_dataset(prepared.test_dataloader)
        if input_transform is None and control not in RANDOMIZED_TRUNK_CONTROLS:
            if cached_dataset is None:
                raise ValueError(
                    "Embedding controls require cached test embeddings or a "
                    "token-based test dataloader."
                )
            embeddings = cached_dataset.embeddings
            ids = cached_dataset.ids
            labels = torch.as_tensor(cached_dataset.labels)
        else:
            batch_transform = (
                self._make_sequence_scrambler(prepared.tokenizer, seed)
                if is_scrambled_sequences
                else None
            )
            embeddings, ids, labels, _ = extract_embeddings(
                trunk,
                backend.token_test_dataloader,
                accelerator,
                backend.pooling,
                backend.dtype,
                layers=backend.layers or [-1],
                autocast_dtype={
                    "float16": torch.float16,
                    "bfloat16": torch.bfloat16,
                    "float32": torch.float32,
                }[backend.autocast_dtype],
                normalize_before_pooling=backend.normalize_before_pooling,
                batch_transform=batch_transform,
                is_ragged=is_ragged,
                # token_test_dataloader yields the full dataset on every
                # rank already (see its docstring); gathering it again here
                # would double every row per additional rank.
                gather_across_processes=False,
            )
            if (
                cache is not None
                and cache_key is not None
                and backend.cache_policy != "off"
                and not is_ragged
                and cache.fits_budget(
                    embeddings.shape[0], embeddings.shape[-1], backend.dtype
                )
                and accelerator.is_main_process
            ):
                cache.write("test", cache_key, embeddings, ids, labels)
            # Only the main process writes the cache file above; without this
            # barrier, a later control that reuses this same cache_key could
            # have other ranks see cache.exists()==False (still racing ahead
            # of the write) while the main process, now past the write, sees
            # cache.exists()==True and takes the no-collective cache-hit
            # return path instead of joining the other ranks' extract_embeddings
            # call -- an inter-rank divergence that hangs the collective forever.
            accelerator.wait_for_everyone()
        return embeddings, ids, labels

    def _decode_scrambled_sequences(
        self, prepared: PreparedArtifacts, accelerator: Accelerator, seed: int
    ) -> list[str]:
        assert prepared.embedding_backend is not None
        transform = self._make_sequence_scrambler(prepared.tokenizer, seed)
        all_sequences: list[str] = []
        for batch in prepared.embedding_backend.token_test_dataloader:
            batch = dict(batch)
            batch.pop("id", None)
            batch.pop("labels", None)
            transformed = transform(batch)
            # token_test_dataloader yields the full dataset on every rank
            # already (see its docstring): decode locally instead of
            # gathering, which would duplicate every row once per rank.
            all_sequences.extend(
                prepared.tokenizer.batch_decode(
                    transformed["input_ids"].cpu().tolist(), skip_special_tokens=True
                )
            )
        return all_sequences

    @staticmethod
    def _run_local_head_inference(
        head_model: Any,
        dataloader: DataLoader,
        task_type: str,
        collect_probabilities: bool,
        accelerator: Accelerator,
        gather_across_processes: bool = True,
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
        with torch.inference_mode():
            for batch in dataloader:
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
                    key: (
                        value.to(accelerator.device)
                        if torch.is_tensor(value)
                        else value
                    )
                    for key, value in batch.items()
                }
                logits = head_model(**batch).logits
                preds = task.extract_predictions(logits, labels=labels)
                probabilities_batch = (
                    logits.softmax(dim=-1) if all_probabilities is not None else None
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
                        rest[0].cpu().tolist()
                        if probabilities_batch is not None
                        else None
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
        if len(all_predictions) != expected_examples:
            raise RuntimeError(
                f"Local head inference produced {len(all_predictions)} unique "
                f"predictions for {expected_examples} examples."
            )
        return all_ids, all_predictions, all_probabilities

    @staticmethod
    def _build_control_model(
        control: str,
        workspace: EvaluationWorkspace,
        prepared: PreparedArtifacts,
        accelerator: Accelerator,
        seed: int,
    ) -> Any:
        trained_model = accelerator.unwrap_model(prepared.model)

        # `from_pretrained`'s head init and `_init_weights` both draw from
        # torch's global RNG (there's no generator argument to inject), so
        # re-seed with `seed` here -- rather than once, up front, for the
        # whole run -- meaning each control's randomness depends only on
        # `seed`, not on the order controls happen to be built in.
        if control in RANDOMIZED_MODEL_CONTROLS:
            torch.manual_seed(seed)

        if control == UNTRAINED_CONTROL:
            model_id = workspace.model_repo_id or workspace.model_name
            task = get_task(workspace.task_type)
            num_labels = task.resolve_num_labels(
                workspace.num_labels, workspace.dataset_name
            )
            model = task.build_model(model_id, num_labels, workspace.model_kwargs)
            # `from_pretrained` always loads onto CPU; match the trained
            # model's device (already placed by accelerator.prepare() in
            # TuneStep or run_evaluate.py) so the (already-prepared)
            # test_dataloader's batches land on the same device as this
            # control model's parameters.
            model = model.to(next(trained_model.parameters()).device)
        elif control in RANDOMIZED_MODEL_CONTROLS - {UNTRAINED_CONTROL}:
            # Constructing from config lets each model class initialize all
            # parameters, including biases and normalization state that a
            # model-specific `_init_weights` may leave untouched.
            model = type(trained_model)(copy.deepcopy(trained_model.config)).to(
                next(trained_model.parameters()).device
            )
            base_prefix = model.base_model_prefix
            trained_state = trained_model.state_dict()
            if control == RANDOM_TRUNK_CONTROL:
                state_to_copy = {
                    name: value
                    for name, value in trained_state.items()
                    if name != base_prefix and not name.startswith(f"{base_prefix}.")
                }
                model.load_state_dict(state_to_copy, strict=False)
            elif control == RANDOM_HEAD_CONTROL:
                state_to_copy = {
                    name: value
                    for name, value in trained_state.items()
                    if name == base_prefix or name.startswith(f"{base_prefix}.")
                }
                model.load_state_dict(state_to_copy, strict=False)
        elif control == SCRAMBLED_SEQUENCES_CONTROL:
            # Same trained model; only the inputs are corrupted (see
            # `_make_sequence_scrambler`), so no reinitialization is needed.
            model = trained_model
        else:
            raise ValueError(f"Unknown predict control condition: {control}")

        model.eval()
        return model

    @staticmethod
    def _make_sequence_scrambler(
        tokenizer: Any, seed: int
    ) -> Callable[[dict[str, Any]], dict[str, Any]]:
        """Build a batch_transform that shuffles token order within each sequence
        (leaving padding/special-token positions, as well as each sequence's
        first amino acid, in place), destroying sequence/positional
        information while preserving token composition.

        Loops over rows in Python (batch_size is small relative to a training
        loop, so this isn't performance-critical) and permutes each row's
        shufflable positions in place via ``torch.randperm``, rather than a
        single vectorized argsort over the whole batch. Uses its own
        ``torch.Generator`` seeded from `seed` (rather than torch's global
        RNG) so the shuffling is fully deterministic and independent of any
        other randomness consumed elsewhere in the run.
        """
        special_ids_list = list(tokenizer.all_special_ids)
        # Lazily bound to the first batch's device and reused (its state
        # advances) across subsequent batches, so calls draw from a single
        # continuing random stream instead of repeating the same values.
        state: dict[str, Any] = {"generator": None, "device": None}

        def _scramble(batch: dict[str, Any]) -> dict[str, Any]:
            batch = dict(batch)
            input_ids = batch["input_ids"]
            device = input_ids.device
            batch_size, _ = input_ids.shape

            if state["generator"] is None or state["device"] != device:
                state["generator"] = torch.Generator(device=device).manual_seed(seed)
                state["device"] = device
            generator = state["generator"]
            special_ids = torch.tensor(
                special_ids_list, device=device, dtype=input_ids.dtype
            )
            movable = ~torch.isin(input_ids, special_ids)

            scrambled = input_ids.clone()
            if "cu_seqlens" in batch:
                offsets = batch["cu_seqlens"][: batch["num_sequences"] + 1].tolist()
                spans = [
                    (0, start, end) for start, end in zip(offsets[:-1], offsets[1:])
                ]
            else:
                spans = [(row, 0, input_ids.shape[1]) for row in range(batch_size)]
            for row, start, end in spans:
                movable_idx = movable[row, start:end].nonzero(as_tuple=True)[0] + start
                if movable_idx.numel() <= 1:
                    continue
                # Keep the first movable position (first amino acid) in
                # place; only the remaining movable positions are shuffled.
                shufflable_idx = movable_idx[1:]
                perm = shufflable_idx[
                    torch.randperm(
                        shufflable_idx.numel(), generator=generator, device=device
                    )
                ]
                scrambled[row, shufflable_idx] = input_ids[row, perm]

            batch["input_ids"] = scrambled
            return batch

        return _scramble
