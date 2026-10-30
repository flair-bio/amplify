"""Map configured dataset sources onto a local ``--data-root`` directory.

Rules (plan resolver rule 9):

* The main dataset resolves to ``DIR/<local_path>`` where ``local_path``
  defaults to the dataset name.
* Each external test set resolves to ``DIR/<local_path>`` where ``local_path``
  defaults to the last component of its ``dataset_repo_id``.
* Missing local inputs are errors. There is never a silent Hub fallback, and
  the Hub revision is cleared for local main datasets (revisions only apply to
  Hub sources).

The mapping is computed from the dataset YAML alone; it only checks that
paths exist, so it works offline and in dry runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from modules.evaluate.src.config.namespaces import canonicalize_layer


class DataRootError(ValueError):
    """Raised when a configured dataset cannot be found under ``--data-root``."""


def _relative_local_path(value: str, label: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not value:
        raise DataRootError(
            f"local_path for '{label}' must be a relative path inside --data-root, "
            f"got '{value}'"
        )
    return path


def _existing_local_dataset(data_root: Path, relative: Path, label: str) -> str:
    path = (data_root / relative).resolve()
    if not path.exists():
        raise DataRootError(
            f"local dataset '{label}' not found at {path} (--data-root {data_root}); "
            "--data-root never falls back to the Hub"
        )
    return str(path)


def _load_raw_dataset_yaml(dataset_config: Path) -> dict[str, Any]:
    raw = yaml.safe_load(dataset_config.read_text()) or {}
    if not isinstance(raw, dict):
        raise DataRootError(f"dataset config {dataset_config} must be a mapping")
    return raw


def data_root_overrides(dataset_config: Path, data_root: Path) -> list[str]:
    """Return dotlist overrides mapping one dataset (and its externals) locally.

    The YAML is read through the canonical namespace mapping, so legacy
    ``workspace.dataset_*`` spellings resolve to the same ``dataset.*`` keys.
    """
    raw = _load_raw_dataset_yaml(dataset_config)
    canonical = canonicalize_layer(raw, str(dataset_config))
    dataset = canonical.get("dataset") or {}
    name = str(dataset.get("name") or dataset_config.stem)
    main_rel = _relative_local_path(str(dataset.get("local_path") or name), name)
    main_path = _existing_local_dataset(data_root, main_rel, name)

    overrides = [
        f"dataset.repo_id={main_path}",
        "dataset.revision=null",
    ]
    for index, external in enumerate(dataset.get("external_test_sets") or []):
        repo_id = str(external["dataset_repo_id"])
        label = str(external.get("name") or repo_id)
        default_local = repo_id.rsplit("/", 1)[-1]
        rel = _relative_local_path(
            str(external.get("local_path") or default_local), label
        )
        external_path = _existing_local_dataset(data_root, rel, label)
        overrides.append(
            f"dataset.external_test_sets.{index}.dataset_repo_id={external_path}"
        )
        overrides.append(f"dataset.external_test_sets.{index}.dataset_revision=null")
    return overrides
