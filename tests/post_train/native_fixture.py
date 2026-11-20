"""Subprocess fixture: exercise real native code without global HF/op registries."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from post_train_fixtures import BASE_VOCAB, feature_columns

torch.set_num_threads(2)
root = Path(sys.argv[2])
checkpoint = root / "student"
output = root / "output"
vocab = BASE_VOCAB[1:] + BASE_VOCAB[:1] + ["<extra>"] if sys.argv[3] == "single" else BASE_VOCAB
token_ids = {token: index for index, token in enumerate(vocab)}
dimension = 7 if sys.argv[3] == "single" else 1024
features = [f"latent_{i}" for i in range(dimension)] if dimension == 7 else feature_columns(dimension)
length = 8 if sys.argv[3] == "single" else 64
probe = torch.tensor([[token_ids[token] for token in ("<bos>", "A", "M", "L", "<eos>")]])
attention = torch.ones_like(probe, dtype=torch.bool)


def build():
    from modules.pretrain.src.model.configuration_amplify import AMPLIFYConfig
    from modules.pretrain.src.model.modeling_amplify import (
        AMPLIFYForMaskedLM,
        register_amplify_auto_classes,
    )
    from modules.pretrain.src.model.tokenizer import ProteinTokenizer, TokenizerConfig

    config = AMPLIFYConfig(
        hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, max_position_embeddings=length + 2, nonlinear_mlm_head=True,
        vocab_size=len(vocab), pad_token_id=token_ids["<pad>"],
        bos_token_id=token_ids["<bos>"], eos_token_id=token_ids["<eos>"],
    )
    config.auto_map = register_amplify_auto_classes()
    AMPLIFYForMaskedLM(config).save_pretrained(checkpoint)
    tokenizer = ProteinTokenizer(TokenizerConfig(
        vocab=vocab, pad_token="<pad>", mask_token="<mask>", bos_token="<bos>",
        eos_token="<eos>", unk_token="<unk>",
    ))
    tokenizer.register_for_auto_class("AutoTokenizer")
    tokenizer.save_pretrained(checkpoint)
    if sys.argv[3] == "single":
        source = "\n".join(
            (checkpoint / name).read_text()
            for name in ("configuration_amplify.py", "fa4.py", "modeling_amplify.py")
        )
        source = source.replace("from .configuration_amplify import AMPLIFYConfig", "")
        source = source.replace("from .fa4 import get_fa4_varlen_func", "")
        (checkpoint / "amplify.py").write_text(source)
        saved_config = json.loads((checkpoint / "config.json").read_text())
        saved_config["auto_map"] = {
            key: "amplify." + value.rsplit(".", 1)[1]
            for key, value in saved_config["auto_map"].items()
        }
        (checkpoint / "config.json").write_text(json.dumps(saved_config))
        for name in ("configuration_amplify.py", "fa4.py", "modeling_amplify.py"):
            (checkpoint / name).unlink()
    (checkpoint / "unrelated.py").write_text("raise RuntimeError('Not a checkpoint dependency')\n")
    data = root / "data"
    data.mkdir()
    rows = []
    rng = np.random.default_rng(42)
    for i in range(65):
        sequence = "A" * (length - 4) + "".join("ACDEFGHIKLMNPQRSTVWY"[(i // 20**j) % 20] for j in range(4))
        path = data / f"{i}.parquet"
        row = {
            "id": f"fixture-{i:03d}", "protein_id": str(i), "chain": "A", "source": "atlas",
            "split": "train" if i < 64 else "validation", "sequence": sequence,
            "residue_ids": [f"{j + 1}:" for j in range(length)],
            "embedding_file": path.name,
        }
        values = rng.standard_normal((length, dimension)).astype(np.float32)
        table = pa.table({
            "record_id": pa.array([row["id"]] * length),
            **{name: pa.array([row.get(name)] * length, type=pa.string())
               for name in ("protein_id", "chain", "source", "split", "cluster_id")},
            "residue_index": pa.array(np.arange(length, dtype=np.int32)),
            "residue_id": pa.array(row["residue_ids"]),
            "amino_acid": pa.array(list(sequence)),
            **{name: pa.array(values[:, j]) for j, name in enumerate(features)},
        }).replace_schema_metadata({
            b"embedding_format": b"residue-by-dimension-float32",
            b"embedding_dimension": str(dimension).encode(),
        })
        pq.write_table(table, path)
        row["embedding_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(row)
    (data / "dataset.json").write_text(json.dumps({
        "metadata": {
            "embedding_format": "parquet_per_protein_v1",
            "embedding_schema": {"feature_columns": features},
            "preparation_note": "producer provenance is preserved",
        },
        "records": rows,
    }))


def train():
    from types import SimpleNamespace
    from unittest.mock import Mock

    from modules.post_train.src.contracts.config import PostTrainingConfig
    from modules.post_train.src.run_post_train import run
    from modules.post_train.src.runners import latent_alignment

    load = latent_alignment.load_student
    export = latent_alignment.export_student
    before = {}
    tracker = Mock()
    sdk = SimpleNamespace(init=Mock(return_value=tracker), Settings=Mock())

    def observed_load(*args, **kwargs):
        student = load(*args, **kwargs)
        before["encoder"] = next(student.amplify.parameters()).detach().clone()
        before["decoder"] = next(student.lm_head.parameters()).detach().clone()
        if sys.argv[3] == "single":
            sys.modules["wandb"] = sdk
        return student

    def observed_export(model, checkpoint, output):
        assert not torch.equal(before["encoder"], next(model.student.amplify.parameters()))
        assert not torch.equal(before["decoder"], next(model.student.lm_head.parameters()))
        model.student.eval()
        with torch.no_grad():
            logits = model.student(input_ids=probe, attention_mask=attention).logits
        torch.save({"state": model.student.state_dict(), "logits": logits}, root / "expected.pt")
        return export(model, checkpoint, output)

    latent_alignment.load_student = observed_load
    latent_alignment.export_student = observed_export
    config = PostTrainingConfig.model_validate({
        "method": "latent_alignment",
        "seed": 123,
        "student": {"checkpoint": str(checkpoint), "trust_remote_code": True},
        "dataset": {"path": str(root / "data")},
        "output_dir": str(output),
        "training": {"device": "cpu", "batch_size": 64, "microbatch_size": 16},
    })
    if sys.argv[3] == "single":
        config = PostTrainingConfig.model_validate({
            **config.model_dump(),
            "alignment": {
                "projection_dim": 12, "pool_proteins": 16, "loss_weight": 0.2,
                "initial_temperature": 0.2, "min_inverse_temperature": 0.5,
            },
            "mlm": {
                "mask_probability": 1.0, "mask_replace_probability": 1.0,
                "random_replace_probability": 0.0,
            },
            "training": {
                **config.training.model_dump(), "batch_size": 32, "microbatch_size": 8,
                "betas": (0.8, 0.9), "weight_decay": 0.02, "gradient_clip": 0.5,
                "validation_interval": 1,
            },
            "wandb": {"enabled": True, "mode": "offline"},
        })
    summary = run(config)
    assert summary["completed"] and summary["steps"] == 64 // config.training.batch_size
    assert summary["protein_presentations"] == 64
    assert summary["negative_pool_proteins"] == config.alignment.pool_proteins
    assert summary["independent_pools_per_update"] == config.training.batch_size // config.alignment.pool_proteins
    heads = torch.load(output / "alignment_heads.pt", weights_only=True)
    assert heads["sequence_projection"]["weight"].shape[0] == config.alignment.projection_dim
    assert heads["structure_projection"]["weight"].shape == (config.alignment.projection_dim, dimension)
    assert summary["method"]["vocabulary_size"] == len(vocab)
    assert summary["provenance"]["dataset_metadata"]["preparation_note"] == "producer provenance is preserved"
    assert summary["method"]["latent_weight"] == config.alignment.loss_weight
    assert summary["method"]["betas"] == list(config.training.betas)
    assert summary["method"]["gradient_clip"] == config.training.gradient_clip
    resolved = json.loads((output / "config.json").read_text())
    assert resolved["method"] == "latent_alignment"
    assert resolved["seed"] == 123
    assert resolved["training"]["device"] == summary["device"] == "cpu"
    assert resolved["dataset"]["path"] == str((root / "data").absolute())
    assert summary["provenance"]["hub_revision"] is None
    expected_source = "amplify.py" if sys.argv[3] == "single" else "modeling_amplify.py"
    assert expected_source in summary["provenance"]["student_files_sha256"]
    assert (output / "amplify" / expected_source).is_file()
    assert not (output / "amplify/unrelated.py").exists()
    logs = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert all(value > 0 for value in logs[1]["gradient_norms_before_clipping"].values())
    assert "validation" in logs[0] and "validation" in logs[-1]
    if sys.argv[3] == "single":
        from modules.post_train.src.tracking.progress import wandb_metrics

        assert [row["step"] for row in logs if "validation" in row] == [0, 1, 2]
        assert len(tracker.log.call_args_list) == len(logs)
        for call, row in zip(tracker.log.call_args_list, logs):
            assert call.args == (wandb_metrics(row),)
            assert call.kwargs == {"step": row["step"]}
        tracker.finish.assert_called_once_with(exit_code=0)
    try:
        run(config)
    except FileExistsError:
        pass
    else:
        raise AssertionError("Existing output was not protected.")
    del logs


def reload():
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    student = AutoModelForMaskedLM.from_pretrained(
        output / "amplify", trust_remote_code=True, local_files_only=True,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        output / "amplify", trust_remote_code=True, local_files_only=True,
    )
    assert tokenizer("AML")["input_ids"] == probe.tolist()[0]
    expected = torch.load(root / "expected.pt", weights_only=True)
    assert set(student.state_dict()) == set(expected["state"])
    for name, value in student.state_dict().items():
        torch.testing.assert_close(value, expected["state"][name], rtol=0, atol=0)
    with torch.no_grad():
        logits = student(input_ids=probe, attention_mask=attention).logits
    torch.testing.assert_close(logits, expected["logits"], rtol=0, atol=0)


if __name__ == "__main__":
    {"build": build, "train": train, "reload": reload}[sys.argv[1]]()
