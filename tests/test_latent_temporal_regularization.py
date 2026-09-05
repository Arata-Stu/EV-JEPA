from __future__ import annotations

import pytest
import torch

from event_window_jepa.losses.latent_temporal import (
    RATE_NORMALIZATION,
    window_level_latent_temporal_regularization,
)


def _regularize(
    tokens: torch.Tensor,
    activity: torch.Tensor,
    duration: torch.Tensor | None = None,
):
    if duration is None:
        duration = torch.full(tokens.shape[:2], 50.0)
    return window_level_latent_temporal_regularization(
        tokens,
        activity,
        duration,
        rate_gamma=1.0,
        rate_eps=1e-6,
        straightening_eps=1e-6,
        rate_normalization=RATE_NORMALIZATION,
    )


def test_rate_alignment_is_invariant_to_count_and_time_units_and_detached() -> None:
    tokens = torch.tensor(
        [[[[0.0, 0.0]], [[1.0, 2.0]], [[2.0, 4.0]]]],
        requires_grad=True,
    )
    activity = torch.tensor([[[10.0], [12.0], [18.0]]], requires_grad=True)
    duration = torch.tensor([[25.0, 50.0, 100.0]])

    baseline = _regularize(tokens, activity, duration)
    count_rescaled = _regularize(tokens, activity * 100.0, duration)
    time_rescaled = _regularize(tokens, activity, duration * 1_000.0)

    torch.testing.assert_close(
        baseline.rate_alignment_loss,
        count_rescaled.rate_alignment_loss,
    )
    torch.testing.assert_close(
        baseline.rate_alignment_loss,
        time_rescaled.rate_alignment_loss,
    )
    baseline.rate_alignment_loss.backward()
    assert tokens.grad is not None
    assert activity.grad is None


def test_rate_alignment_downweights_an_abrupt_rate_change() -> None:
    tokens = torch.tensor([[[[0.0]], [[1.0]]]])
    duration = torch.full((1, 2), 50.0)
    similar = _regularize(tokens, torch.tensor([[[10], [10]]]), duration)
    abrupt = _regularize(tokens, torch.tensor([[[10], [100]]]), duration)

    assert similar.rate_alignment_mean_weight > abrupt.rate_alignment_mean_weight
    assert similar.rate_alignment_loss > abrupt.rate_alignment_loss
    torch.testing.assert_close(similar.rate_alignment_mean_weight, torch.tensor(1.0))


def test_unsupported_patch_tokens_do_not_affect_either_loss() -> None:
    tokens = torch.tensor(
        [
            [
                [[0.0, 0.0], [10.0, -10.0]],
                [[1.0, 0.0], [-50.0, 50.0]],
                [[2.0, 0.0], [100.0, -100.0]],
            ]
        ]
    )
    activity = torch.tensor([[[5, 0], [5, 0], [5, 0]]])
    changed = tokens.clone()
    changed[:, :, 1] = torch.tensor(
        [[1_000.0, 2_000.0], [-3_000.0, 4_000.0], [5_000.0, -6_000.0]]
    )

    first = _regularize(tokens, activity)
    second = _regularize(changed, activity)

    torch.testing.assert_close(first.rate_alignment_loss, second.rate_alignment_loss)
    torch.testing.assert_close(
        first.latent_straightening_loss,
        second.latent_straightening_loss,
    )
    torch.testing.assert_close(first.rate_alignment_pairs, torch.tensor(2.0))
    torch.testing.assert_close(first.latent_straightening_pairs, torch.tensor(1.0))


def test_straightening_prefers_direction_consistency_not_zero_motion() -> None:
    activity = torch.tensor([[[3], [3], [3]]])
    straight = torch.tensor([[[[0.0, 0.0]], [[1.0, 0.0]], [[2.0, 0.0]]]])
    bent = torch.tensor([[[[0.0, 0.0]], [[1.0, 0.0]], [[0.0, 0.0]]]])
    constant = torch.zeros((1, 3, 1, 2), requires_grad=True)

    straight_output = _regularize(straight, activity)
    bent_output = _regularize(bent, activity)
    constant_output = _regularize(constant, activity)

    torch.testing.assert_close(
        straight_output.latent_straightening_loss,
        torch.tensor(0.0),
    )
    assert bent_output.latent_straightening_loss > 1.99
    torch.testing.assert_close(
        constant_output.latent_straightening_pairs,
        torch.tensor(0.0),
    )
    torch.testing.assert_close(
        constant_output.latent_straightening_loss,
        torch.tensor(0.0),
    )
    constant_output.latent_straightening_loss.backward()
    assert constant.grad is not None
    torch.testing.assert_close(constant.grad, torch.zeros_like(constant.grad))


def test_temporal_regularization_validates_shape_and_normalization() -> None:
    tokens = torch.zeros((1, 3, 2, 4))
    activity = torch.ones((1, 3, 2))
    duration = torch.ones((1, 3))

    with pytest.raises(ValueError, match="rate_normalization"):
        window_level_latent_temporal_regularization(
            tokens,
            activity,
            duration,
            rate_normalization="raw_hz",
        )
    with pytest.raises(ValueError, match=r"\[B,T,N\]"):
        _regularize(tokens, activity[:, :, :1], duration)
    with pytest.raises(ValueError, match="finite positive"):
        _regularize(tokens, activity, torch.zeros_like(duration))
