import copy
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from post_train_fixtures import feature_columns, write_tokenizer

from modules.post_train.src.model.tokenization import load_vocabulary


@pytest.fixture(autouse=True, scope="module")
def bounded_cpu_threads():
    import torch

    original = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(original)


@pytest.fixture
def vocabulary(tmp_path):
    return load_vocabulary(write_tokenizer(tmp_path / "tokenizer"))


@pytest.fixture
def prepared_factory(tmp_path):
    def create(dimension=7, lengths=(64, 65, 66), teacher=True, columns=None):
        root = tmp_path / "parquet"
        root.mkdir()
        features = columns if columns is not None else feature_columns(dimension)
        teacher = (
            {"model_id": "fixture/teacher", "revision": "fixture-revision",
             "dimension": dimension}
            if teacher is True else copy.deepcopy(teacher)
        )
        records, arrays = [], []
        for index, (aa, length) in enumerate(zip("ACD", lengths)):
            values = np.arange(length * dimension, dtype=np.float32).reshape(length, dimension) + 1
            row = {
                "id": f"fixture-{index}:A", "protein_id": f"fixture-{index}", "chain": "A",
                "source": "atlas", "split": "train" if index == 0 else "validation",
                "sequence": aa * length, "residue_ids": [f"{i + 1}:" for i in range(length)],
                "cluster_id": f"cluster-{index}" if index == 0 else None,
                "embedding_file": f"{index}.parquet",
            }
            path = root / row["embedding_file"]
            columns = {
                "record_id": pa.array([row["id"]] * length),
                **{name: pa.array([row[name]] * length, type=pa.string())
                   for name in ("protein_id", "chain", "source", "split", "cluster_id")},
                "residue_index": pa.array(np.arange(length, dtype=np.int32)),
                "residue_id": pa.array(row["residue_ids"]),
                "amino_acid": pa.array(list(row["sequence"])),
                **{name: pa.array(values[:, i]) for i, name in enumerate(features)},
            }
            parquet_metadata = {
                b"embedding_format": b"residue-by-dimension-float32",
                b"embedding_dimension": str(dimension).encode(),
            }
            if teacher:
                for field, key in (("model_id", b"teacher_model"), ("revision", b"teacher_revision")):
                    if field in teacher:
                        parquet_metadata[key] = str(teacher[field]).encode()
            table = pa.table(columns).replace_schema_metadata(parquet_metadata)
            pq.write_table(table, path)
            records.append(row)
            arrays.append(values)
        metadata = {
            "embedding_format": "parquet_per_protein_v1",
            "embedding_schema": {"feature_columns": features},
        }
        if teacher is not None:
            metadata["structural_teacher"] = teacher
        payload = {"metadata": metadata, "records": records}
        manifest = root / "dataset.json"
        manifest.write_text(json.dumps(payload))
        return manifest, payload, arrays

    return create
