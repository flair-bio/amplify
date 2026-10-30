"""HF-only CASP contact units, original-coordinate alignment and evaluation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np
import polars as pl
import torch
from transformers import AutoTokenizer

from modules.evaluate.src.contracts.artifacts import (
    ContactArtifacts,
    ContactEvaluationMetadata,
    ContactScoreOutput,
    ControlInputs,
    FeatureRequest,
    FitResult,
    PredictOutput,
    PreparedArtifacts,
)
from modules.evaluate.src.dataset.embedding_cache import weights_fingerprint
from modules.evaluate.src.metrics import contact_metrics
from modules.evaluate.src.model.contact_attention import (
    ATTENTION_PROTOCOL,
    ContactAttentionModel,
)
from modules.evaluate.src.model.contact_jacobian import ContactJacobianModel
from modules.evaluate.src.model.contact_model import (
    ContactPredictionModel,
    build_contact_model,
    load_contact_model,
)
from modules.evaluate.src.model.controls import (
    build_random_trunk,
    contact_random_trunk_seed,
)
from modules.evaluate.src.tasks.base import MetricFn, StructuredTask
from modules.evaluate.src.tasks.registry import register_task
from modules.evaluate.src.trainers.contact import (
    BILINEAR_PROTOCOL,
    fit_contact_attention_head,
    fit_contact_head,
)

if TYPE_CHECKING:
    from accelerate import Accelerator
    from datasets import DatasetDict

    from modules.evaluate.src.contracts.artifacts import (
        EvaluationMetadata,
    )
    from modules.evaluate.src.contracts.config import (
        EvaluationConfig,
        EvaluationWorkspaceConfig,
        PredictConfig,
        PrepareConfig,
        ScoreConfig,
        TuneConfig,
    )

from modules.evaluate.src.dataset.contact import (
    IGNORE_INDEX,
    PROTOCOL_VERSION,
    SCHEMA_VERSION,
    ContactChain,
    ContactDataset,
    ContactTokens,
    load_contact_dataset,
    prepare_contact_dataset,
    tokenize_contact_chain,
)


def contact_development_feature_identity(
    dataset: ContactDataset,
    tokens: Sequence[ContactTokens],
    *,
    weights: str,
    backbone_config: Mapping[str, Any],
    tf32: bool,
    matmul_precision: str,
) -> dict[str, Any]:
    """Scientific identity of deterministic full-chain development features."""
    chains = [*dataset.splits["train"], *dataset.splits["validation"]]
    if len(tokens) != len(chains):
        raise ValueError(
            "Development feature identity requires one token row per chain"
        )
    return {
        "extraction": "casp-final-embeddings-v1",
        "weights": weights,
        "backbone_config": dict(backbone_config),
        "dtype": "float32",
        "tf32": tf32,
        "matmul_precision": matmul_precision,
        "chains": [
            (
                chain.split,
                chain.record_id,
                chain.alignment.sequence,
                token.input_ids,
                token.position_ids,
                token.residue_token_indices,
            )
            for chain, token in zip(chains, tokens)
        ],
    }


def contact_development_attention_fingerprint(
    dataset: ContactDataset,
    tokens: Sequence[ContactTokens],
    *,
    weights: str,
    backbone_config: Mapping[str, Any],
    tf32: bool,
    matmul_precision: str,
) -> str:
    """Fingerprint train-only attention inputs without retaining attention maps."""

    training = dataset.splits["train"]
    if len(tokens) != len(training):
        raise ValueError("Attention identity requires one token row per training chain")
    identity = {
        "protocol": ATTENTION_PROTOCOL,
        "development_fingerprint": dataset.development_fingerprint,
        "weights": weights,
        "backbone_config": dict(backbone_config),
        "precision": "float32",
        "tf32": tf32,
        "matmul_precision": matmul_precision,
        "chains": [
            (
                chain.record_id,
                chain.alignment.sequence,
                token.input_ids,
                token.position_ids,
                token.residue_token_indices,
            )
            for chain, token in zip(training, tokens)
        ],
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


@register_task
class ContactPredictionTask(StructuredTask):
    """Fixed-split CASP contact prediction, alignment and scoring."""

    name = "contact_prediction"
    supports_packing = False
    fixed_protocol = True
    supported_head_types = frozenset(
        {"bilinear512", "attention_l1", "categorical_jacobian"}
    )
    supported_controls = frozenset({"random_trunk"})
    dataset_baselines = frozenset({"separation_prior"})
    uses_predictor_identity = True
    is_ragged = True
    pairwise = True
    aligns_labels = True
    label_dtype = torch.int8
    accepted_target_types = frozenset({"categorical"})
    accepted_label_formats = frozenset({"contact_pairs"})

    #: Tokenized-residue separation below which a pair is never scored (it's
    #: a sequence-adjacency contact, not a structural one).
    MIN_SEP: ClassVar[int] = 6

    metrics: ClassVar[dict[str, MetricFn]] = {
        **contact_metrics.PRECISION_METRICS,
        **contact_metrics.RANGE_AUC_METRICS,
    }
    default_metric_names = (
        *contact_metrics.PRECISION_METRICS,
        *contact_metrics.RANGE_AUC_METRICS,
    )

    def prepare_inputs(
        self,
        *,
        inputs: EvaluationWorkspaceConfig,
        config: PrepareConfig,
        device: torch.device,
        dataset: DatasetDict | None = None,
        predictor: str | None = None,
    ) -> PreparedArtifacts:

        source = inputs.dataset_repo_id
        model_id = inputs.model_repo_id
        if source is None or model_id is None:
            raise ValueError(
                "Contact prediction requires dataset_repo_id and model_repo_id"
            )
        if predictor is None:
            raise ValueError("Contact prediction requires a configured predictor.")
        if dataset is None:
            contacts = load_contact_dataset(
                source,
                revision=inputs.dataset_revision,
                config_name=inputs.dataset_config_name,
            )
        else:
            contacts = prepare_contact_dataset(
                dataset,
                source=source,
                revision=inputs.dataset_revision,
            )
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = build_contact_model(
            model_id,
            predictor=predictor,
            tokenizer=tokenizer,
        ).to(device)
        return PreparedArtifacts(
            model=model,
            tokenizer=tokenizer,
            train_dataloader=None,
            val_dataloader=None,
            test_dataloader=None,
            contact=ContactArtifacts(dataset=contacts),
        )

    def validate_pipeline(
        self,
        config: EvaluationConfig,
        *,
        workspace_seed: int | None,
        num_processes: int,
    ) -> None:
        if num_processes != 1:
            raise ValueError(
                "Contact embeddings require a single-process fixed-split run"
            )
        if workspace_seed != config.seed:
            raise ValueError(
                "Contact workspace must be created with the configured seed"
            )

    def validate_config(self, config: EvaluationConfig) -> None:
        """Guard unsupported execution capabilities and ineffective overrides."""
        steps = config.steps
        tune = steps.tune
        self.validate_dataset_config(
            config.workspace.target_type,
            config.workspace.label_format,
            config.workspace.dataset_name,
        )
        if (
            config.external_test_sets
            or steps.layerwise.enabled
            or steps.predict.baselines
        ):
            raise ValueError(
                "Contact prediction requires fixed CASP splits, without generic external tests/layers/inline baselines"
            )
        if steps.prepare.packed or steps.prepare.split.model_dump() != {
            "name": "default",
            "train": "train",
            "validation": "validation",
            "test": "test",
            "refit_additional_splits": [],
        }:
            raise ValueError(
                "Contact prediction requires unchanged fixed splits and unpadded full chains"
            )
        unsupported_controls = (
            set[str](steps.predict.controls) - self.supported_controls
        )
        if unsupported_controls:
            raise ValueError(
                "Contact prediction supports only the random_trunk control; "
                f"got {sorted(unsupported_controls)}"
            )
        if steps.predict.collect_probabilities:
            raise ValueError("CASP contact prediction does not collect probabilities")
        if steps.score.bootstrap_enabled:
            raise ValueError("Contact bootstrap requires complex-level resampling")
        if steps.score.metrics is not None and set(steps.score.metrics) != set(
            contact_metrics.CASP_METRICS
        ):
            raise ValueError("Contact prediction reports fixed CASP metrics")

        predictor = tune.head.head_type
        self.validate_head(predictor)
        expected_mode = (
            "evaluate_as_is" if predictor == "categorical_jacobian" else "tune_probe"
        )
        if steps.mode != expected_mode:
            raise ValueError(
                f"Contact predictor '{predictor}' requires steps.mode='{expected_mode}'"
            )
        if steps.prepare.embedding_cache.enabled:
            raise ValueError(
                "Contact predictors do not use the generic pooled embedding cache"
            )
        if (
            tune.hyperparameter_search.enabled
            or tune.refit_on_train_validation
            or tune.variance_seeds
            or tune.head.hyperparameters
            or tune.head.search_space
            or config.workspace.model_kwargs
        ):
            raise ValueError(
                "Contact predictors use fixed protocols without HPO, refit, "
                "seed sweeps or head/model overrides"
            )
        if predictor != "categorical_jacobian" and tune.standardize_features:
            raise ValueError("Contact predictors do not standardize features")
        if predictor == "bilinear512":
            supplied = tune.model_dump()
            conflicts = [
                name
                for name, value in asdict(BILINEAR_PROTOCOL).items()
                if supplied[name] != value
            ]
            if conflicts:
                raise ValueError(
                    "The bilinear512 protocol fixes these settings: "
                    f"{', '.join(conflicts)}. Only max_epochs is a fitting override."
                )
            if tune.optimizer_kwargs or tune.scheduler_kwargs:
                raise ValueError(
                    "The bilinear512 protocol has no optimizer/scheduler overrides"
                )
            if tune.max_epochs <= BILINEAR_PROTOCOL.early_stopping_patience:
                raise ValueError(
                    "Contact epoch limit must allow validation patience 10"
                )

    def feature_requests(
        self, prepared: PreparedArtifacts
    ) -> dict[str, FeatureRequest]:

        model, tokenizer = prepared.model, prepared.tokenizer
        if not isinstance(model, ContactPredictionModel):
            return {}
        if prepared.contact is None or tokenizer is None:
            raise ValueError("Contact fitting features require data and a tokenizer.")
        dataset = prepared.contact.dataset
        chains = [*dataset.splits["train"], *dataset.splits["validation"]]
        tokens = [tokenize_contact_chain(chain, tokenizer) for chain in chains]
        identity = contact_development_feature_identity(
            dataset,
            tokens,
            weights=weights_fingerprint(model.trunk),
            backbone_config=json.loads(model.trunk.config.to_json_string()),
            tf32=torch.backends.cuda.matmul.allow_tf32,
            matmul_precision=torch.get_float32_matmul_precision(),
        )

        def extract() -> dict[str, torch.Tensor]:
            return {
                str(index): model.extract(chain, tokenizer)
                for index, chain in enumerate(chains)
            }

        return {
            "development": FeatureRequest(
                namespace="contact-development",
                identity=identity,
                extract=extract,
                shapes={
                    str(index): (
                        len(chain.alignment.clean_sequence),
                        model.head.left.in_features,
                    )
                    for index, chain in enumerate(chains)
                },
            )
        }

    def fitting_identity(self, prepared: PreparedArtifacts) -> dict[str, Any]:

        model = prepared.model
        if prepared.contact is None or prepared.tokenizer is None:
            raise ValueError("Contact fitting identity requires data and a tokenizer.")
        dataset = prepared.contact.dataset
        if isinstance(model, ContactPredictionModel):
            key = prepared.features["development"].fingerprint
            protocol = PROTOCOL_VERSION
        elif isinstance(model, ContactAttentionModel):
            key = contact_development_attention_fingerprint(
                dataset,
                [
                    tokenize_contact_chain(chain, prepared.tokenizer)
                    for chain in dataset.splits["train"]
                ],
                weights=weights_fingerprint(model.trunk),
                backbone_config=json.loads(model.trunk.config.to_json_string()),
                tf32=torch.backends.cuda.matmul.allow_tf32,
                matmul_precision=torch.get_float32_matmul_precision(),
            )
            protocol = ATTENTION_PROTOCOL
        else:
            return {}
        return {
            "development_fingerprint": dataset.development_fingerprint,
            "development_features": key,
            "protocol": protocol,
        }

    def fit(
        self,
        prepared: PreparedArtifacts,
        config: TuneConfig,
        inputs: EvaluationWorkspaceConfig,
        accelerator: Accelerator,
        *,
        seed: int,
        mode: Literal["tune_probe", "finetune"] = "tune_probe",
    ) -> FitResult:

        if prepared.contact is None or prepared.tokenizer is None:
            raise ValueError("Contact fitting requires data and a tokenizer.")
        model, tokenizer, dataset = (
            prepared.model,
            prepared.tokenizer,
            prepared.contact.dataset,
        )
        if isinstance(model, ContactJacobianModel):
            return FitResult(
                model=model,
                seed=seed,
                checkpoint_settings={"fitting": "zero_shot"},
                statistics={"contact_training.json": {"fitting": "zero_shot"}},
                serialization="local",
            )
        if isinstance(model, ContactAttentionModel):
            settings = dict(
                fit_contact_attention_head(model, dataset, tokenizer, seed=seed)
            )
        elif isinstance(model, ContactPredictionModel):
            if "development" not in prepared.features:
                raise ValueError(
                    "Contact development features must be prepared before fitting."
                )
            development = prepared.features["development"]
            chains = [*dataset.splits["train"], *dataset.splits["validation"]]
            features = {
                chain.record_id: development.values[str(index)]
                for index, chain in enumerate(chains)
            }
            settings = dict(
                fit_contact_head(
                    model,
                    dataset,
                    features,
                    seed=seed,
                    max_epochs=config.max_epochs,
                    device=accelerator.device,
                )
            )
        else:
            raise TypeError("Unsupported contact predictor.")
        return FitResult(
            model=model,
            seed=seed,
            input_identity=prepared.input_identity,
            statistics={"contact_training.json": settings},
            checkpoint_settings=settings,
            serialization="local",
        )

    def independent_controls(
        self,
        prepared: PreparedArtifacts,
        controls: Sequence[str],
        *,
        seed: int,
        device: torch.device,
    ) -> dict[str, ControlInputs]:

        if "random_trunk" not in controls:
            return {}
        model = prepared.model
        if not isinstance(
            model, (ContactPredictionModel, ContactAttentionModel, ContactJacobianModel)
        ):
            raise TypeError("Contact controls require a contact predictor.")
        trunk_seed = contact_random_trunk_seed(seed, "trunk")
        head_seed = contact_random_trunk_seed(seed, "head")
        trunk = build_random_trunk(model.trunk, trunk_seed)
        if isinstance(model, ContactJacobianModel):
            random_model = ContactJacobianModel(
                trunk,
                prepared.tokenizer,
                mutant_batch_size=model.mutant_batch_size,
            ).to(device)
        else:
            devices = [device.index or 0] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(head_seed)
                random_model = type(model)(trunk).to(device)
        return {
            "random_trunk": ControlInputs(
                model=random_model,
                seed=head_seed,
                serialization_model=(
                    trunk if isinstance(model, ContactJacobianModel) else random_model
                ),
                metadata={
                    "control": "random_trunk",
                    "protocol": "contact-random-trunk-v1",
                    "trunk_initialization": "random_from_config",
                    "trunk_seed": trunk_seed,
                    "head_seed": (
                        None if isinstance(model, ContactJacobianModel) else head_seed
                    ),
                },
            )
        }

    def control_settings(
        self, inputs: ControlInputs, result: FitResult
    ) -> dict[str, Any]:

        return {
            **inputs.metadata,
            "development_features": (
                result.input_identity.get("development_features")
                if isinstance(inputs.model, ContactPredictionModel)
                else None
            ),
            "fit": result.checkpoint_settings,
        }

    def prediction_metadata(self, values: Mapping[str, Any]) -> EvaluationMetadata:

        return ContactEvaluationMetadata.model_validate(values)

    def predict(
        self,
        prepared: PreparedArtifacts,
        metadata: EvaluationMetadata,
        config: PredictConfig,
    ) -> PredictOutput:
        if prepared.contact is None:
            raise TypeError("Contact observations require contact data.")
        unsupported = set[str](config.controls) - self.supported_controls
        if unsupported:
            raise ValueError(
                "CASP contact prediction supports only random_trunk; "
                f"got {sorted(unsupported)}"
            )
        if config.collect_probabilities:
            raise ValueError(
                "CASP contacts consume pair scores, not class probabilities"
            )
        for name in config.controls:
            if name not in prepared.controls.independent_models:
                raise ValueError(
                    f"{name} was requested but no fitted contact control is available"
                )
        output = self.predict_units(
            prepared.contact.dataset,
            lambda chain: prepared.model.predict(chain, prepared.tokenizer),
            metadata,
        )
        for name in config.controls:
            control = prepared.controls.independent_models[name]
            output.controls[name] = self.predict_units(
                prepared.contact.dataset,
                lambda chain: control.predict(chain, prepared.tokenizer),
                metadata,
            ).predictions
        output.requested_controls = list(config.controls)
        return output

    def score(
        self,
        predictions: PredictOutput,
        config: ScoreConfig,
        *,
        seed: int = 710019,
    ) -> ContactScoreOutput:
        if config.bootstrap_enabled:
            raise ValueError("CASP bootstrap must resample complexes, not flat rows")
        if config.metrics is not None and set(config.metrics) != set(
            contact_metrics.CASP_METRICS
        ):
            raise ValueError("Contact scoring reports the fixed CASP metrics")
        return self.score_units(predictions)

    def default_metrics(self) -> list[str]:
        if self.name == "contact_prediction":
            return list(contact_metrics.CASP_METRICS)
        return super().default_metrics()

    def validate_metric_names(self, metric_names: Sequence[str]) -> None:
        if self.name != "contact_prediction":
            return super().validate_metric_names(metric_names)
        unknown = set(metric_names) - set(contact_metrics.CASP_METRICS)
        if unknown:
            raise KeyError(f"Unknown CASP contact metrics: {sorted(unknown)}")

    def predict_units(
        self,
        dataset: ContactDataset,
        predictor: Callable[[ContactChain], np.ndarray | torch.Tensor],
        metadata: EvaluationMetadata,
        *,
        split: str = "test",
    ) -> PredictOutput:
        """Coordinate full-chain extraction and preserve every unit's identity."""
        from modules.evaluate.src.model.inference import run_contact_inference

        if metadata.seed is None:
            raise ValueError("Contact predictions require an explicit run seed")
        contact_metadata = ContactEvaluationMetadata.model_validate(
            {
                **metadata.model_dump(),
                "dataset_revision": dataset.revision,
                "dataset_fingerprint": dataset.fingerprint,
                "schema_version": SCHEMA_VERSION,
                "protocol_version": PROTOCOL_VERSION,
            }
        )
        rows_by_chain = {}
        for chain, scores in run_contact_inference(dataset, predictor, split=split):
            rows_by_chain[chain.record_id] = [
                dict(
                    id=unit.id,
                    record_id=chain.record_id,
                    unit_id=unit.unit_id,
                    split=chain.split,
                    source=chain.source,
                    edition=chain.edition,
                    complex_id=chain.complex_id,
                    target=chain.target,
                    kind=unit.kind,
                    labels_available=unit.labels_available,
                    **contact_metrics.contact_prediction_row(
                        scores, unit, chain.alignment
                    ),
                )
                for unit in chain.units
            ]
        rows = [
            row
            for chain in dataset.splits[split]
            for row in rows_by_chain[chain.record_id]
        ]
        if not rows:
            raise ValueError(f"No contact scoring rows in split {split}")
        frame = pl.DataFrame(rows, infer_schema_length=None).with_columns(
            pl.col("original_i", "original_j").cast(pl.List(pl.Int32)),
            pl.col("labels").cast(pl.List(pl.Int8)),
            pl.col("scores").cast(pl.List(pl.Float64)),
        )
        return PredictOutput(metadata=contact_metadata, predictions=frame)

    def score_units(self, predictions: PredictOutput) -> ContactScoreOutput:
        """Score all CASP settings, then aggregate EUs, chains and complexes."""

        metadata = predictions.metadata
        if (
            not isinstance(metadata, ContactEvaluationMetadata)
            or metadata.schema_version != SCHEMA_VERSION
            or metadata.protocol_version != PROTOCOL_VERSION
            or not metadata.dataset_fingerprint
            or metadata.seed is None
        ):
            raise ValueError(
                "Contact scoring requires schema-6.0 CASP prediction metadata"
            )
        if set(predictions.requested_controls) != set(predictions.controls):
            raise ValueError("Every requested CASP control requires prediction rows")
        if set(predictions.controls) - (
            self.supported_controls | self.dataset_baselines
        ):
            raise ValueError(
                "CASP contact prediction supports only random_trunk and "
                "the configured dataset baselines"
            )
        if predictions.predictions.is_empty():
            raise ValueError("Contact scoring requires prediction rows")
        payload_columns = {
            "original_i",
            "original_j",
            "labels",
            "scores",
            "length",
            "status",
        }
        identity = {
            key: value
            for key, value in metadata.model_dump().items()
            if key not in {"split", "dataset_name", "dataset_repo_id"}
        }

        def score_rows(
            frame: pl.DataFrame, control: str | None = None
        ) -> tuple[pl.DataFrame, pl.DataFrame]:
            rows = []
            seen = set()
            for row in frame.to_dicts():
                if row["id"] in seen:
                    raise ValueError(f"Duplicate scoring id: {row['id']}")
                seen.add(row["id"])
                for minsep in contact_metrics.CASP_SEPARATIONS:
                    scored = dict(
                        {
                            key: value
                            for key, value in row.items()
                            if key not in payload_columns
                        },
                        **identity,
                        minsep=minsep,
                        **contact_metrics.score_contact_prediction(row, minsep),
                    )
                    if control is not None:
                        scored["control"] = control
                    rows.append(scored)
            summaries = contact_metrics.aggregate_contact_scores(rows)
            if control is not None:
                for summary in summaries:
                    summary["control"] = control
            return (
                pl.DataFrame(rows, infer_schema_length=None),
                pl.DataFrame(summaries, infer_schema_length=None),
            )

        unit_scores, summaries = score_rows(predictions.predictions)
        control_scores = {}
        control_summaries = {}
        primary_by_id = {row["id"]: row for row in predictions.predictions.to_dicts()}
        for name, frame in predictions.controls.items():
            if set(frame["id"].to_list()) != set(primary_by_id):
                raise ValueError(
                    f"{name} control rows do not match primary prediction IDs"
                )
            for row in frame.to_dicts():
                primary = primary_by_id[row["id"]]
                for key in ("original_i", "original_j", "labels", "length", "status"):
                    if row[key] != primary[key]:
                        raise ValueError(
                            f"{name} control candidates differ from primary predictions"
                        )
            control_scores[name], control_summaries[name] = score_rows(frame, name)
        return ContactScoreOutput(
            metadata,
            unit_scores,
            summaries,
            control_unit_scores=control_scores,
            control_summaries=control_summaries,
        )

    def model_kwargs(self) -> dict[str, Any]:
        return {}

    def build_model(
        self,
        model_id: str,
        num_labels: int,
        model_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        if model_kwargs:
            raise ValueError(
                "Contact predictor settings are fixed by the CASP protocol"
            )
        return build_contact_model(model_id)

    def load_finetuned_model(self, model_path: str) -> Any:
        return load_contact_model(model_path)

    def resolve_num_labels(self, num_labels: int | None, dataset_name: str) -> int:
        # Not a class count (binary per-pair score); nothing derives from it.
        return 2

    # -- data ----------------------------------------------------------------

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
    ) -> Any:
        """Crop+shift raw ``[i, j]`` contact pairs into tokenized coordinates.

        The aligned dict retains the cropped sequence length and resolved
        residue positions, excluding special tokens from candidate pairs.
        """
        end = start + kept
        if isinstance(label, dict):
            pairs = label["pairs"]
            source_valid_positions = label["valid_positions"] or []
        else:
            pairs = label
            source_valid_positions = range(num_residues)
        pairs = [
            [offset + (i - start), offset + (j - start)]
            for i, j in pairs
            if start <= i < end and start <= j < end
        ]
        valid_positions = [
            offset + (position - start)
            for position in source_valid_positions
            if start <= position < end
        ]
        return {
            "pairs": pairs,
            "valid_positions": valid_positions,
            "res_start": offset,
            "res_end": offset + kept,
            "length": kept,
        }

    def collate_labels(
        self, labels: list[Any], batch_size: int, seq_len: int
    ) -> torch.Tensor:
        idx = torch.arange(seq_len)
        base_valid = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() >= self.MIN_SEP
        collated = torch.full(
            (batch_size, seq_len, seq_len), IGNORE_INDEX, dtype=self.label_dtype
        )
        for b, entry in enumerate(labels):
            res_start, res_end = entry["res_start"], entry["res_end"]
            resolved = torch.zeros(seq_len, dtype=torch.bool)
            resolved[entry.get("valid_positions", range(res_start, res_end))] = True
            valid = base_valid.clone()
            # Exclude padding *and* special-token positions (before res_start,
            # at/after res_end): only real residues can be scored, positive
            # or implicit-negative.
            valid[res_end:, :] = False
            valid[:, res_end:] = False
            valid[:res_start, :] = False
            valid[:, :res_start] = False
            valid &= resolved[:, None] & resolved[None, :]
            collated[b][valid] = 0
            for i, j in entry["pairs"]:
                if valid[i, j]:
                    collated[b, i, j] = 1
                    collated[b, j, i] = 1
        return collated

    def extract_predictions(
        self, logits: torch.Tensor, *, labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        return logits.sigmoid()

    def split_batch_rows(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        *,
        residue_lengths: torch.Tensor | None = None,
    ) -> list[tuple[Any, Any]]:
        rows: list[tuple[Any, Any]] = []
        for row_index, (pred_mat, label_mat) in enumerate(zip(predictions, labels)):
            valid = label_mat != IGNORE_INDEX
            # Every scored pair appears twice (symmetric); count each once.
            upper = valid & torch.triu(torch.ones_like(valid), diagonal=1)
            i_idx, j_idx = upper.nonzero(as_tuple=True)
            scores = [float(score) for score in pred_mat[i_idx, j_idx].tolist()]
            seps = [int(sep) for sep in (j_idx - i_idx).tolist()]
            values = [int(value) for value in label_mat[i_idx, j_idx].tolist()]
            if residue_lengths is not None:
                length = int(residue_lengths[row_index])
            else:
                scored_cols = valid.any(dim=0).nonzero(as_tuple=True)[0]
                length = (
                    int(scored_cols.max() - scored_cols.min() + 1)
                    if scored_cols.numel()
                    else 0
                )
            pred_row = {"scores": scores, "separations": seps}
            label_row = {"length": length, "values": values}
            rows.append((pred_row, label_row))
        return rows

    def split_batch_rows_with_context(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        *,
        batch_metadata: Mapping[str, Any] | None = None,
    ) -> list[tuple[Any, Any]]:
        residue_lengths = (
            batch_metadata.get("residue_lengths")
            if batch_metadata is not None
            else None
        )
        return self.split_batch_rows(
            predictions, labels, residue_lengths=residue_lengths
        )

    def flatten_for_metrics(self, values: list[Any]) -> list[Any]:
        # Precision-at-L needs each sequence's own top-scored candidates and
        # length, so this task's metrics consume the per-sequence rows
        # directly instead of a pooled flat sequence.
        return values

    def compute_per_example_metrics(
        self,
        predictions: Sequence,
        labels: Sequence,
        metric_names: Sequence[str],
        average: str = "weighted",
        probabilities: Sequence | None = None,
    ) -> np.ndarray:
        """Compute contact scores once per protein for paired resampling.

        Rows follow the input order and columns follow metric_names. Each
        existing metric is a mean over proteins with defined contributions,
        so resampling these scores preserves its aggregation and NaN handling.
        """
        self.validate_metric_names(metric_names)
        return np.asarray(
            [
                [
                    self.metrics[name]([prediction], [label], average)
                    for name in metric_names
                ]
                for prediction, label in zip(predictions, labels)
            ],
            dtype=np.float64,
        ).reshape(len(predictions), len(metric_names))

    def scramble_labels(self, labels: list[Any], rng: np.random.Generator) -> list[Any]:
        """Permute each row's per-pair labels while retaining its length."""
        scrambled: list[Any] = []
        for row in labels:
            if not isinstance(row, dict):
                length, values = row[0], row[1:]
                permuted_idx = rng.permutation(len(values))
                scrambled.append([length] + [values[j] for j in permuted_idx])
                continue
            length, values = row["length"], row["values"]
            permuted_idx = rng.permutation(len(values))
            scrambled.append(
                {"length": length, "values": [values[j] for j in permuted_idx]}
            )
        return scrambled
