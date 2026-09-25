"""Unit tests for optimizer config sanitization and constructor behavior."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch
from torch import nn

# ---------------------------------------------------------------------------
# Module loading
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).parents[2]
_OPTIMIZER_PATH = _ROOT / "modules" / "pretrain" / "src" / "optimizer" / "optimizer.py"

_spec = importlib.util.spec_from_file_location("_optimizer", _OPTIMIZER_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

OptimizerConfig = _mod.OptimizerConfig
get_optimizer = _mod.get_optimizer


class _TinyModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.embed = nn.Embedding(16, 8)
		self.position_embeddings = nn.Embedding(16, 8)
		self.linear = nn.Linear(8, 8)
		self.norm = nn.LayerNorm(8)
		self.frozen = nn.Parameter(torch.ones(8), requires_grad=False)

	def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
		x = self.embed(token_ids) + self.position_embeddings(token_ids)
		x = self.linear(x)
		x = self.norm(x)
		return x.mean()


def _base_kwargs() -> dict[str, object]:
	return {
		"type": "AdamW",
		"lr": 1e-3,
		"betas": (0.9, 0.95),
		"eps": 1e-8,
		"weight_decay": 0.1,
		"fused": False,
	}


class TestSanitizeOptKwargs:
	def test_adamw_excludes_non_constructor_fields(self):
		cfg = OptimizerConfig(**_base_kwargs())

		kwargs = cfg.sanitize_opt_kwargs()

		assert "type" not in kwargs
		assert "weight_decay" not in kwargs
		assert kwargs["lr"] == 1e-3
		assert kwargs["betas"] == (0.9, 0.95)
		assert kwargs["eps"] == 1e-8
		assert kwargs["fused"] is False

	def test_adafactor_drops_fused_betas_and_float_eps(self):
		cfg = OptimizerConfig(
			type="Adafactor",
			lr=1e-3,
			betas=(0.9, 0.999),
			eps=1e-8,
			weight_decay=0.0,
			fused=True,
		)

		kwargs = cfg.sanitize_opt_kwargs()

		assert "fused" not in kwargs
		assert "betas" not in kwargs
		assert "eps" not in kwargs
		assert kwargs["lr"] == 1e-3

	def test_adafactor_keeps_tuple_eps(self):
		cfg = OptimizerConfig(
			type="Adafactor",
			lr=1e-3,
			betas=(0.9, 0.999),
			eps=(1e-30, 1e-3),
			weight_decay=0.0,
			fused=False,
		)

		kwargs = cfg.sanitize_opt_kwargs()

		assert kwargs["eps"] == (1e-30, 1e-3)
		assert "betas" not in kwargs
		assert "fused" not in kwargs

	def test_non_adam_types_drop_fused(self):
		cfg = OptimizerConfig(
			type="Lamb",
			lr=2e-4,
			betas=(0.9, 0.999),
			eps=1e-6,
			weight_decay=0.01,
			fused=True,
		)

		kwargs = cfg.sanitize_opt_kwargs()

		assert "fused" not in kwargs


class TestGetOptimizerConstructorBehavior:
	def test_adamw_groups_decay_and_no_decay_parameters(self):
		model = _TinyModel()
		cfg = OptimizerConfig(**_base_kwargs())

		optimizer = get_optimizer(model, cfg)

		assert isinstance(optimizer, torch.optim.AdamW)
		assert len(optimizer.param_groups) == 2

		group_by_wd = {group["weight_decay"]: group for group in optimizer.param_groups}
		assert 0.1 in group_by_wd
		assert 0.0 in group_by_wd

		decay_ids = {id(p) for p in group_by_wd[0.1]["params"]}
		no_decay_ids = {id(p) for p in group_by_wd[0.0]["params"]}

		# 2D trainable weights should decay.
		assert id(model.linear.weight) in decay_ids

		# Embeddings and 1D parameters should not decay.
		assert id(model.embed.weight) in no_decay_ids
		assert id(model.position_embeddings.weight) in no_decay_ids
		assert id(model.linear.bias) in no_decay_ids
		assert id(model.norm.weight) in no_decay_ids

		# Frozen parameters are excluded from all groups.
		assert id(model.frozen) not in decay_ids
		assert id(model.frozen) not in no_decay_ids

	def test_fused_on_cpu_raises_runtime_error(self, monkeypatch: pytest.MonkeyPatch):
		model = _TinyModel()
		cfg = OptimizerConfig(
			type="AdamW",
			lr=1e-3,
			betas=(0.9, 0.999),
			eps=1e-8,
			weight_decay=0.01,
			fused=True,
		)
		monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

		with pytest.raises(RuntimeError, match="Fused optimizer requested"):
			get_optimizer(model, cfg)

	def test_adam_constructor_returns_torch_adam(self):
		model = _TinyModel()
		cfg = OptimizerConfig(
			type="Adam",
			lr=3e-4,
			betas=(0.9, 0.98),
			eps=1e-6,
			weight_decay=0.0,
			fused=False,
		)

		optimizer = get_optimizer(model, cfg)

		assert isinstance(optimizer, torch.optim.Adam)
		# Group-level kwargs should be preserved across both parameter groups.
		assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-4)
		assert optimizer.param_groups[0]["betas"] == (0.9, 0.98)
		assert optimizer.param_groups[0]["eps"] == pytest.approx(1e-6)

	def test_unsupported_type_raises_value_error_with_model_construct(self):
		model = _TinyModel()
		cfg = OptimizerConfig.model_construct(
			type="NotReal",
			lr=1e-3,
			betas=(0.9, 0.999),
			eps=1e-8,
			weight_decay=0.01,
			fused=False,
		)

		with pytest.raises(ValueError, match="Unsupported optimizer type"):
			get_optimizer(model, cfg)


class TestExternalOptimizerImports:
	def test_adafactor_missing_dependency_raises_informative_import_error(
		self, monkeypatch: pytest.MonkeyPatch
	):
		model = _TinyModel()
		cfg = OptimizerConfig(
			type="Adafactor",
			lr=1e-3,
			betas=(0.9, 0.999),
			eps=(1e-30, 1e-3),
			weight_decay=0.0,
			fused=False,
		)

		fake_optim = types.ModuleType("transformers.optimization")
		monkeypatch.setitem(sys.modules, "transformers.optimization", fake_optim)

		with pytest.raises(ImportError, match="Adafactor requires the `transformers` library"):
			get_optimizer(model, cfg)

	def test_lamb_missing_dependency_raises_informative_import_error(
		self, monkeypatch: pytest.MonkeyPatch
	):
		model = _TinyModel()
		cfg = OptimizerConfig(
			type="Lamb",
			lr=1e-3,
			betas=(0.9, 0.999),
			eps=1e-6,
			weight_decay=0.01,
			fused=False,
		)

		monkeypatch.delitem(sys.modules, "torch_optimizer", raising=False)

		# Ensure import resolves to a module without Lamb.
		fake_torch_optim = types.ModuleType("torch_optimizer")
		monkeypatch.setitem(sys.modules, "torch_optimizer", fake_torch_optim)

		with pytest.raises(ImportError, match="Lamb requires the `torch_optimizer` package"):
			get_optimizer(model, cfg)

	def test_adafactor_constructor_uses_sanitized_kwargs(self, monkeypatch: pytest.MonkeyPatch):
		captured: dict[str, object] = {}

		class _FakeAdafactor:
			def __init__(self, params, **kwargs):
				self.param_groups = list(params)
				captured.update(kwargs)

		fake_optim = types.ModuleType("transformers.optimization")
		fake_optim.Adafactor = _FakeAdafactor
		monkeypatch.setitem(sys.modules, "transformers.optimization", fake_optim)

		model = _TinyModel()
		cfg = OptimizerConfig(
			type="Adafactor",
			lr=2e-3,
			betas=(0.8, 0.9),
			eps=1e-8,
			weight_decay=0.05,
			fused=True,
		)

		opt = get_optimizer(model, cfg)

		assert isinstance(opt, _FakeAdafactor)
		assert captured["lr"] == 2e-3
		assert "betas" not in captured
		assert "eps" not in captured
		assert "fused" not in captured

	def test_lamb_constructor_uses_sanitized_kwargs(self, monkeypatch: pytest.MonkeyPatch):
		captured: dict[str, object] = {}

		class _FakeLamb:
			def __init__(self, params, **kwargs):
				self.param_groups = list(params)
				captured.update(kwargs)

		fake_torch_optim = types.ModuleType("torch_optimizer")
		fake_torch_optim.Lamb = _FakeLamb
		monkeypatch.setitem(sys.modules, "torch_optimizer", fake_torch_optim)

		model = _TinyModel()
		cfg = OptimizerConfig(
			type="Lamb",
			lr=4e-4,
			betas=(0.9, 0.99),
			eps=1e-6,
			weight_decay=0.02,
			fused=True,
		)

		opt = get_optimizer(model, cfg)

		assert isinstance(opt, _FakeLamb)
		assert captured["lr"] == pytest.approx(4e-4)
		assert captured["betas"] == (0.9, 0.99)
		assert captured["eps"] == pytest.approx(1e-6)
		assert "fused" not in captured
