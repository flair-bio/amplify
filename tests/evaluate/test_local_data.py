"""--data-root maps configured datasets to local directories, never to the Hub."""

from pathlib import Path

import pytest
import yaml

from modules.evaluate.src.dataset.local_data import DataRootError, data_root_overrides


def _write_dataset(path: Path, **workspace: str) -> Path:
    raw: dict = {"workspace": {"dataset_name": "main_ds", **workspace}}
    raw.setdefault("external_test_sets", [])
    path.write_text(yaml.safe_dump(raw))
    return path


def test_main_dataset_defaults_to_dataset_name(tmp_path: Path) -> None:
    config = _write_dataset(tmp_path / "cfg.yaml")
    data_root = tmp_path / "data"
    (data_root / "main_ds").mkdir(parents=True)
    overrides = data_root_overrides(config, data_root)
    assert overrides[0] == f"dataset.repo_id={(data_root / 'main_ds').resolve()}"
    assert "dataset.revision=null" in overrides


def test_explicit_local_path_is_used(tmp_path: Path) -> None:
    config = _write_dataset(tmp_path / "cfg.yaml", local_path="shared/stability")
    data_root = tmp_path / "data"
    (data_root / "shared" / "stability").mkdir(parents=True)
    overrides = data_root_overrides(config, data_root)
    assert overrides[0].endswith("shared/stability")


def test_missing_main_dataset_is_an_error_without_hub_fallback(tmp_path: Path) -> None:
    config = _write_dataset(tmp_path / "cfg.yaml")
    with pytest.raises(DataRootError, match="never falls back to the Hub"):
        data_root_overrides(config, tmp_path)


def test_external_test_set_maps_under_data_root(tmp_path: Path) -> None:
    config = tmp_path / "cfg.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "workspace": {"dataset_name": "main_ds"},
                "external_test_sets": [
                    {"name": "sethard", "dataset_repo_id": "flair-bio/sethard"}
                ],
            }
        )
    )
    data_root = tmp_path / "data"
    (data_root / "main_ds").mkdir(parents=True)
    with pytest.raises(DataRootError, match="sethard"):
        data_root_overrides(config, data_root)
    (data_root / "sethard").mkdir()
    overrides = data_root_overrides(config, data_root)
    assert (
        f"dataset.external_test_sets.0.dataset_repo_id={(data_root / 'sethard').resolve()}"
        in overrides
    )
    # A pinned Hub revision must not apply to a local directory.
    assert "dataset.external_test_sets.0.dataset_revision=null" in overrides


@pytest.mark.parametrize("bad", ["/abs/path", "../escape"])
def test_local_path_must_stay_under_data_root(tmp_path: Path, bad: str) -> None:
    config = _write_dataset(tmp_path / "cfg.yaml", local_path=bad)
    with pytest.raises(DataRootError, match="relative path"):
        data_root_overrides(config, tmp_path)
