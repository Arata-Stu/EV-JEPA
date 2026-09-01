from __future__ import annotations

import pytest
import torch

from event_window_jepa.config import ExperimentConfig, OptimizationConfig
from event_window_jepa.train.checkpoint import config_hash
from event_window_jepa.train.pretrain import (
    _accumulation_group_geometry,
    _optimizer_updates_per_epoch,
)


def _minimal_config(*, accumulation_steps: int | None = None) -> ExperimentConfig:
    optimization: dict[str, object] = {"precision": "fp32"}
    if accumulation_steps is not None:
        optimization["gradient_accumulation_steps"] = accumulation_steps
    return ExperimentConfig.from_mapping(
        {
            "data": {
                "manifest": "events.jsonl",
                "crop_size": [224, 224],
            },
            "optimization": optimization,
        }
    )


def test_gradient_accumulation_defaults_to_legacy_one_update_per_batch() -> None:
    implicit = _minimal_config()
    explicit = _minimal_config(accumulation_steps=1)

    assert implicit.optimization.gradient_accumulation_steps == 1
    assert explicit.optimization.gradient_accumulation_steps == 1
    assert config_hash(implicit) == config_hash(explicit)


def test_nondefault_gradient_accumulation_is_part_of_checkpoint_identity() -> None:
    assert config_hash(_minimal_config()) != config_hash(
        _minimal_config(accumulation_steps=4)
    )


@pytest.mark.parametrize("value", [0, -1])
def test_gradient_accumulation_requires_a_positive_value(value: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        OptimizationConfig(gradient_accumulation_steps=value)


@pytest.mark.parametrize("value", [True, 1.5, "4"])
def test_gradient_accumulation_rejects_non_integer_yaml_values(value: object) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        OptimizationConfig.from_mapping({"gradient_accumulation_steps": value})


def test_accumulation_groups_scale_an_epoch_remainder_by_its_actual_size() -> None:
    assert _optimizer_updates_per_epoch(10, 4) == 3
    assert _accumulation_group_geometry(0, 10, 4) == (0, 4, True, False)
    assert _accumulation_group_geometry(3, 10, 4) == (0, 4, False, True)
    assert _accumulation_group_geometry(4, 10, 4) == (1, 4, True, False)
    assert _accumulation_group_geometry(8, 10, 4) == (2, 2, True, False)
    assert _accumulation_group_geometry(9, 10, 4) == (2, 2, False, True)


def test_accumulation_geometry_rejects_invalid_steps() -> None:
    with pytest.raises(ValueError, match="inside the epoch"):
        _accumulation_group_geometry(4, 4, 2)
    with pytest.raises(ValueError, match="must be positive"):
        _optimizer_updates_per_epoch(4, 0)


def test_microbatch_accumulation_matches_group_mean_updates() -> None:
    reference = torch.nn.Linear(2, 1, bias=False)
    accumulated = torch.nn.Linear(2, 1, bias=False)
    accumulated.load_state_dict(reference.state_dict())
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)
    accumulated_optimizer = torch.optim.SGD(accumulated.parameters(), lr=0.05)
    inputs = torch.tensor(
        [[1.0, -1.0], [0.5, 2.0], [-2.0, 1.0], [3.0, 0.5]] * 2
        + [[-0.5, 1.5], [2.0, -3.0]]
    )
    targets = torch.linspace(-1.0, 1.0, inputs.shape[0]).unsqueeze(1)
    accumulation_steps = 4

    for start in range(0, inputs.shape[0], accumulation_steps):
        end = min(start + accumulation_steps, inputs.shape[0])
        reference_optimizer.zero_grad(set_to_none=True)
        reference_loss = torch.nn.functional.mse_loss(
            reference(inputs[start:end]), targets[start:end]
        )
        reference_loss.backward()
        reference_optimizer.step()

    optimizer_steps = 0
    for micro_step in range(inputs.shape[0]):
        _, group_size, is_first, is_last = _accumulation_group_geometry(
            micro_step, inputs.shape[0], accumulation_steps
        )
        if is_first:
            accumulated_optimizer.zero_grad(set_to_none=True)
        micro_loss = torch.nn.functional.mse_loss(
            accumulated(inputs[micro_step : micro_step + 1]),
            targets[micro_step : micro_step + 1],
        )
        (micro_loss / group_size).backward()
        if is_last:
            accumulated_optimizer.step()
            optimizer_steps += 1

    assert optimizer_steps == 3
    torch.testing.assert_close(accumulated.weight, reference.weight)
