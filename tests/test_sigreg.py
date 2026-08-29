from __future__ import annotations

import pytest
import torch

import event_window_jepa.losses.sigreg as sigreg_module
from event_window_jepa.losses.sigreg import (
    ProjectedSIGReg,
    SIGRegProjector,
    SlicedEppsPulleySIGReg,
)


def test_projector_seed_is_deterministic_without_consuming_global_rng() -> None:
    torch.manual_seed(31)
    expected_next_random = torch.rand(5)
    torch.manual_seed(31)
    first = SIGRegProjector(8, 6, hidden_dim=10, depth=3, seed=13)
    actual_next_random = torch.rand(5)
    second = SIGRegProjector(8, 6, hidden_dim=10, depth=3, seed=13)

    torch.testing.assert_close(actual_next_random, expected_next_random)
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name])


def test_projector_and_sigreg_run_in_fp32_and_backpropagate() -> None:
    module = ProjectedSIGReg(
        input_dim=8,
        projection_dim=6,
        hidden_dim=10,
        num_slices=16,
        num_frequencies=7,
        seed=17,
    )
    features = torch.randn(5, 8, dtype=torch.bfloat16, requires_grad=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = module(features)

    assert output.loss.dtype == torch.float32
    assert output.real_error.dtype == torch.float32
    assert output.imaginary_error.dtype == torch.float32
    assert output.effective_samples.item() == 5
    output.loss.backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in module.projector.parameters()
    )


def test_sigreg_calls_all_collectives_even_with_no_local_support(monkeypatch) -> None:
    calls: list[torch.Size] = []

    def recording_sum(value: torch.Tensor) -> torch.Tensor:
        calls.append(value.shape)
        return value

    monkeypatch.setattr(sigreg_module, "_autograd_all_reduce_sum", recording_sum)
    criterion = SlicedEppsPulleySIGReg(
        4,
        num_slices=8,
        num_frequencies=5,
        seed=19,
    )
    features = torch.full((3, 4), float("nan"), requires_grad=True)
    output = criterion(features, valid_mask=torch.zeros(3, dtype=torch.bool))

    assert calls == [torch.Size([]), torch.Size([8, 5]), torch.Size([8, 5])]
    assert output.loss.item() == 0.0
    assert output.effective_samples.item() == 0.0
    output.loss.backward()
    assert features.grad is not None
    assert torch.count_nonzero(features.grad) == 0


def test_standard_normal_has_lower_ecf_error_than_a_constant() -> None:
    criterion = SlicedEppsPulleySIGReg(
        12,
        num_slices=64,
        num_frequencies=11,
        seed=23,
    )
    generator = torch.Generator().manual_seed(29)
    gaussian = torch.randn(2048, 12, generator=generator)
    constant = torch.zeros_like(gaussian)

    gaussian_loss = criterion(gaussian).loss
    constant_loss = criterion(constant).loss
    assert gaussian_loss < constant_loss


def test_sigreg_directions_are_seeded_and_input_is_strictly_two_dimensional() -> None:
    first = SlicedEppsPulleySIGReg(5, num_slices=7, seed=37)
    second = SlicedEppsPulleySIGReg(5, num_slices=7, seed=37)
    different = SlicedEppsPulleySIGReg(5, num_slices=7, seed=41)

    torch.testing.assert_close(first.directions, second.directions)
    assert not torch.equal(first.directions, different.directions)
    with pytest.raises(ValueError, match=r"\[B,5\]"):
        first(torch.randn(2, 3, 5))
    with pytest.raises(ValueError, match="valid_mask"):
        first(torch.randn(2, 5), torch.ones(2, 1, dtype=torch.bool))
