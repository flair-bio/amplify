from unittest.mock import Mock

import pytest

from modules.post_train.src import run_post_train
from modules.post_train.src.config import PostTrainingConfig
from modules.post_train.src.runners import latent_alignment


def test_cli_dispatches_validated_config_to_latent_runner(monkeypatch):
    runner = Mock(return_value={"completed": True})
    monkeypatch.setattr(latent_alignment, "run", runner)
    run_post_train.main([
        "modules/post_train/configs/latent_alignment.yaml",
        "student.checkpoint=/unused/student",
        "dataset.path=/unused/data",
        "output_dir=/unused/output",
        "method=latent_alignment",
    ])
    runner.assert_called_once()
    config = runner.call_args.args[0]
    assert config.method == "latent_alignment"
    assert config.training.device == "auto"
    assert str(config.student.checkpoint) == "/unused/student"


def test_missing_gpu_fails_before_loading_data_or_creating_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(latent_alignment.torch.cuda, "is_available", lambda: False)
    load = Mock()
    monkeypatch.setattr(latent_alignment, "load_dataset", load)
    output = tmp_path / "output"
    config = PostTrainingConfig(
        student={"checkpoint": "/unused/student"},
        dataset={"path": "/unused/data"},
        output_dir=output,
    )
    with pytest.raises(ValueError, match="No CUDA GPU"):
        latent_alignment.run(config)
    load.assert_not_called()
    assert not output.exists()
