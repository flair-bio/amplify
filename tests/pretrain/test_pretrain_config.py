from pathlib import Path

import pytest
from pydantic import ValidationError
from omegaconf import DictConfig, OmegaConf

from modules.core.utils.config_loader import load_and_parse, load_config
from modules.pretrain.src.config import PretrainConfig
from modules.pretrain.src.model.modeling_amplify import AMPLIFYModelConfig
from modules.pretrain.src.model.tokenizer import TokenizerConfig
from modules.pretrain.src.optimizer import OptimizerConfig
from modules.pretrain.src.scheduler import SchedulerConfig

CONFIGS_DIR = Path(__file__).parents[2] / "modules" / "pretrain" / "configs"
MAIN_CONFIG = CONFIGS_DIR / "config.yaml"


@pytest.fixture
def valid_config_dict():
    """Return the validated pretrain config YAML as a plain dictionary."""
    cfg = load_config(MAIN_CONFIG)
    return OmegaConf.to_container(cfg, resolve=True)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def test_load_config_returns_dictconfig():
    """load_config returns an OmegaConf DictConfig for a valid YAML file."""
    cfg = load_config(MAIN_CONFIG)

    assert isinstance(cfg, DictConfig)
    assert "model" in cfg
    assert "tokenizer" in cfg
    assert "optimizer" in cfg
    assert "scheduler" in cfg


# ---------------------------------------------------------------------------
# Single YAML parsing
# ---------------------------------------------------------------------------


def test_pretrain_config_parsed_from_yaml():
    """Parsing config.yaml produces a fully validated PretrainConfig."""
    config = load_and_parse(MAIN_CONFIG, PretrainConfig)

    assert isinstance(config.model, AMPLIFYModelConfig)
    assert isinstance(config.tokenizer, TokenizerConfig)
    assert isinstance(config.optimizer, OptimizerConfig)
    assert isinstance(config.scheduler, SchedulerConfig)
    assert config.optimizer.type == "AdamW"
    assert config.scheduler.lr_scheduler_type == "cosine_with_min_lr"


# ---------------------------------------------------------------------------
# Plain dict parsing
# ---------------------------------------------------------------------------


def test_pretrain_config_accepts_plain_dict_inputs(valid_config_dict):
    """Test that nested dicts are parsed into the expected config models."""
    config = PretrainConfig(**valid_config_dict)

    assert isinstance(config.model, AMPLIFYModelConfig)
    assert isinstance(config.tokenizer, TokenizerConfig)
    assert config.optimizer.type == "AdamW"
    assert config.scheduler.lr_scheduler_type == "cosine_with_min_lr"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_pretrain_config_forbids_extra_fields(valid_config_dict):
    """Test that PretrainConfig strictly forbids extra fields (extra='forbid')."""
    with pytest.raises(ValidationError) as exc_info:
        PretrainConfig(**valid_config_dict, unexpected_key="should_fail")

    # Verify the error message mentions the extra field
    assert "unexpected_key" in str(exc_info.value)
    assert "Extra inputs are not permitted" in str(exc_info.value)


def test_pretrain_config_missing_required_fields():
    """Test that PretrainConfig fails if required sub-configs are missing."""
    with pytest.raises(ValidationError) as exc_info:
        PretrainConfig()  # Passing nothing

    error_msg = str(exc_info.value)
    # Verify all four missing fields are caught by Pydantic
    assert "model" in error_msg
    assert "tokenizer" in error_msg
    assert "optimizer" in error_msg
    assert "scheduler" in error_msg
    assert "Field required" in error_msg
