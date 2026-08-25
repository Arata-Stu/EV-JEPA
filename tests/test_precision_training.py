from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from event_window_jepa.config import ExperimentConfig, OptimizationConfig
from event_window_jepa.train.checkpoint import (
    load_training_checkpoint,
    save_checkpoint_atomic,
)
from event_window_jepa.train.pretrain import (
    _backward,
    _precision_support_error,
    _step_optimizer,
)


class _FakeGradScaler:
    def __init__(self, scale: float = 128.0) -> None:
        self.scale_value = scale
        self.step_calls = 0
        self.optimizer_step_calls = 0
        self.unscale_calls = 0
        self.update_values: list[float | None] = []
        self.found_nonfinite = False

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        self.unscale_calls += 1
        self.found_nonfinite = any(
            parameter.grad is not None
            and not bool(torch.isfinite(parameter.grad).all())
            for group in optimizer.param_groups
            for parameter in group["params"]
        )

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        self.step_calls += 1
        if not self.found_nonfinite:
            self.optimizer_step_calls += 1
            optimizer.step()

    def update(self, new_scale: float | None = None) -> None:
        self.update_values.append(new_scale)
        if new_scale is not None:
            self.scale_value = float(new_scale)
        elif self.found_nonfinite:
            self.scale_value *= 0.5
        self.found_nonfinite = False

    def get_scale(self) -> float:
        return self.scale_value

    def state_dict(self) -> dict[str, float]:
        return {
            "scale": self.scale_value,
            "backoff_factor": 0.5,
        }

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.scale_value = float(state["scale"])


def _tiny_model() -> SimpleNamespace:
    return SimpleNamespace(
        online_encoder=torch.nn.Linear(2, 2),
        target_encoder=torch.nn.Linear(2, 2),
        predictor=torch.nn.Linear(2, 2),
        scale_embedding=torch.nn.Linear(1, 2),
        target_scale_embedding=torch.nn.Linear(1, 2),
    )


def _parameters(model: SimpleNamespace):
    for name in (
        "online_encoder",
        "target_encoder",
        "predictor",
        "scale_embedding",
        "target_scale_embedding",
    ):
        yield from getattr(model, name).parameters()


def test_fp16_is_a_valid_explicit_precision_but_auto_is_runner_only() -> None:
    assert OptimizationConfig(precision="fp16").precision == "fp16"
    with pytest.raises(ValueError, match="fp32, fp16, or bf16"):
        OptimizationConfig(precision="auto")


def test_half_precision_is_not_silently_used_on_cpu() -> None:
    device = torch.device("cpu")
    assert _precision_support_error(device, "fp32") is None
    assert "requires a CUDA device" in str(
        _precision_support_error(device, "fp16")
    )
    assert "requires a CUDA device" in str(
        _precision_support_error(device, "bf16")
    )


def test_finite_fp16_gradient_steps_optimizer_once() -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = _FakeGradScaler()
    before = model.weight.detach().clone()

    loss = model(torch.ones(1, 2)).sum()
    _backward(loss, scaler)
    gradient_norm, skipped = _step_optimizer(
        model=model,
        optimizer=optimizer,
        grad_scaler=scaler,
        precision="fp16",
        gradient_clip=1.0,
        world_size=1,
    )

    assert torch.isfinite(gradient_norm)
    assert not skipped
    assert scaler.unscale_calls == 1
    assert scaler.step_calls == 1
    assert scaler.optimizer_step_calls == 1
    assert scaler.update_values == [None]
    assert not torch.equal(model.weight, before)


def test_nonfinite_fp16_gradient_skips_step_and_reduces_scale() -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = _FakeGradScaler(scale=128.0)
    before = model.weight.detach().clone()
    model.weight.grad = torch.full_like(model.weight, float("inf"))

    gradient_norm, skipped = _step_optimizer(
        model=model,
        optimizer=optimizer,
        grad_scaler=scaler,
        precision="fp16",
        gradient_clip=1.0,
        world_size=1,
    )

    assert not torch.isfinite(gradient_norm)
    assert skipped
    assert scaler.unscale_calls == 1
    assert scaler.step_calls == 1
    assert scaler.optimizer_step_calls == 0
    assert scaler.update_values == [None]
    assert scaler.get_scale() == 64.0
    assert torch.equal(model.weight, before)


def test_fp16_checkpoint_round_trip_restores_grad_scaler(tmp_path) -> None:
    config = ExperimentConfig.from_mapping(
        {"optimization": {"precision": "fp16"}}
    )
    model = _tiny_model()
    optimizer = torch.optim.AdamW(_parameters(model), lr=1e-3)
    saved_scaler = _FakeGradScaler(scale=512.0)
    path = tmp_path / "fp16.pt"
    save_checkpoint_atomic(
        path,
        model,
        optimizer,
        config,
        epoch=1,
        global_step=2,
        world_size=1,
        steps_per_epoch=2,
        grad_scaler=saved_scaler,
    )

    restored_model = _tiny_model()
    restored_optimizer = torch.optim.AdamW(_parameters(restored_model), lr=1e-3)
    restored_scaler = _FakeGradScaler(scale=1.0)
    epoch, global_step = load_training_checkpoint(
        path,
        restored_model,
        restored_optimizer,
        config,
        torch.device("cpu"),
        world_size=1,
        steps_per_epoch=2,
        grad_scaler=restored_scaler,
    )

    assert (epoch, global_step) == (1, 2)
    assert restored_scaler.get_scale() == 512.0


def test_fp16_checkpoint_save_requires_grad_scaler(tmp_path) -> None:
    config = ExperimentConfig.from_mapping(
        {"optimization": {"precision": "fp16"}}
    )
    model = _tiny_model()
    optimizer = torch.optim.AdamW(_parameters(model), lr=1e-3)
    path = tmp_path / "missing-scaler.pt"
    with pytest.raises(ValueError, match="saving requires a GradScaler"):
        save_checkpoint_atomic(
            path,
            model,
            optimizer,
            config,
            epoch=1,
            global_step=1,
            world_size=1,
            steps_per_epoch=1,
        )


def test_fp32_checkpoint_without_grad_scaler_remains_loadable(tmp_path) -> None:
    config = ExperimentConfig.from_mapping(
        {"optimization": {"precision": "fp32"}}
    )
    model = _tiny_model()
    optimizer = torch.optim.AdamW(_parameters(model), lr=1e-3)
    path = tmp_path / "legacy-fp32.pt"
    save_checkpoint_atomic(
        path,
        model,
        optimizer,
        config,
        epoch=1,
        global_step=1,
        world_size=1,
        steps_per_epoch=1,
    )

    restored_model = _tiny_model()
    restored_optimizer = torch.optim.AdamW(_parameters(restored_model), lr=1e-3)
    assert load_training_checkpoint(
        path,
        restored_model,
        restored_optimizer,
        config,
        torch.device("cpu"),
        world_size=1,
        steps_per_epoch=1,
        grad_scaler=_FakeGradScaler(),
    ) == (1, 1)
