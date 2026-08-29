from __future__ import annotations

import pytest
import torch

from event_window_jepa.losses.latent_prediction import (
    balanced_event_support_latent_prediction_loss,
    latent_prediction_loss,
)


def test_event_support_loss_balances_classes_inside_each_sample() -> None:
    generator = torch.Generator().manual_seed(7)
    prediction = torch.randn(2, 4, 8, generator=generator, requires_grad=True)
    target = torch.randn(2, 4, 8, generator=generator, requires_grad=True)
    activity = torch.tensor(
        [
            [3, 0, 0, 0],
            [4, 2, 1, 0],
        ],
        dtype=torch.int64,
    )

    output = balanced_event_support_latent_prediction_loss(
        prediction,
        target,
        activity,
    )
    expected_samples = []
    for sample in range(2):
        active = activity[sample : sample + 1] > 0
        inactive = ~active
        active_loss = latent_prediction_loss(
            prediction[sample : sample + 1],
            target[sample : sample + 1],
            active,
        )
        inactive_loss = latent_prediction_loss(
            prediction[sample : sample + 1],
            target[sample : sample + 1],
            inactive,
        )
        expected_samples.append((active_loss + inactive_loss) * 0.5)

    torch.testing.assert_close(output.loss, torch.stack(expected_samples).mean())
    assert output.active_token_count.item() == 4
    assert output.inactive_token_count.item() == 4
    assert output.active_sample_count.item() == 2
    assert output.inactive_sample_count.item() == 2
    assert output.valid_sample_count.item() == 2

    output.loss.backward()
    assert prediction.grad is not None
    assert prediction.grad.abs().sum() > 0
    assert target.grad is None


def test_event_support_loss_uses_the_only_class_and_handles_empty_selection() -> None:
    prediction = torch.randn(2, 3, 4, requires_grad=True)
    target = torch.randn_like(prediction)
    activity = torch.zeros(2, 3, dtype=torch.int64)

    output = balanced_event_support_latent_prediction_loss(
        prediction,
        target,
        activity,
    )
    expected = latent_prediction_loss(prediction, target)
    torch.testing.assert_close(output.loss, expected)
    assert output.active_loss.item() == 0.0
    assert output.active_sample_count.item() == 0
    assert output.inactive_sample_count.item() == 2

    empty = balanced_event_support_latent_prediction_loss(
        prediction,
        target,
        activity,
        target_mask=torch.zeros(2, 3, dtype=torch.bool),
    )
    assert empty.loss.item() == 0.0
    assert empty.valid_sample_count.item() == 0
    empty.loss.backward()
    assert prediction.grad is not None
    assert torch.count_nonzero(prediction.grad) == 0


def test_event_support_loss_validates_activity_metadata() -> None:
    prediction = torch.randn(2, 3, 4)
    target = torch.randn_like(prediction)

    with pytest.raises(ValueError, match="shape"):
        balanced_event_support_latent_prediction_loss(
            prediction,
            target,
            torch.zeros(2, 4),
        )
    with pytest.raises(TypeError, match="event counts"):
        balanced_event_support_latent_prediction_loss(
            prediction,
            target,
            torch.zeros(2, 3, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="negative"):
        balanced_event_support_latent_prediction_loss(
            prediction,
            target,
            -torch.ones(2, 3),
        )
    with pytest.raises(ValueError, match="active_threshold"):
        balanced_event_support_latent_prediction_loss(
            prediction,
            target,
            torch.zeros(2, 3),
            active_threshold=-1,
        )

