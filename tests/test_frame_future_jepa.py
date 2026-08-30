from __future__ import annotations

import torch

from event_window_jepa.models.scale_embedding import LogFourierScaleEmbedding
from event_window_jepa.models.vjepa21_event_vit import (
    VJEPA21EventVisionTransformer,
)
from event_window_jepa.models.window_jepa import WindowJEPA
from event_window_jepa.models.window_predictor import WindowPredictor


def _frame_future_model() -> WindowJEPA:
    encoder = VJEPA21EventVisionTransformer(
        image_size=(16, 16),
        patch_size=8,
        input_channels=2,
        embed_dim=32,
        depth=2,
        num_heads=4,
        scale_dim=16,
        supervision_layers=(1,),
    )
    predictor = WindowPredictor(
        num_patches=4,
        encoder_dim=32,
        predictor_dim=16,
        depth=1,
        num_heads=4,
        scale_dim=16,
    )
    return WindowJEPA(
        encoder,
        predictor,
        LogFourierScaleEmbedding(output_dim=16, num_bands=2),
        frame_sigreg_weight=0.02,
        temporal_sigreg_weight=0.0,
        sigreg_projector_hidden_dim=32,
        sigreg_projector_output_dim=16,
        sigreg_num_slices=8,
        sigreg_num_points=5,
    )


def _inputs() -> tuple[torch.Tensor, ...]:
    batch_size, steps = 2, 3
    context = torch.randn(batch_size, steps, 2, 16, 16)
    future = torch.randn_like(context)
    duration = torch.full((batch_size, steps), 50.0)
    context_mask = torch.tensor([True, False, True, False]).reshape(
        1,
        1,
        4,
    ).expand(batch_size, steps, -1).clone()
    target_mask = ~context_mask
    context_activity = torch.tensor(
        [
            [[1, 0, 2, 0], [4, 0, 2, 0], [0, 3, 0, 1]],
            [[0, 1, 0, 2], [0, 2, 1, 0], [5, 0, 0, 2]],
        ]
    )
    future_activity = torch.tensor(
        [
            [[0, 1, 0, 2], [0, 3, 0, 2], [4, 0, 1, 0]],
            [[2, 0, 1, 0], [2, 0, 0, 1], [0, 5, 2, 0]],
        ]
    )
    loss_mask = torch.tensor(
        [[False, True, True], [False, True, True]],
        dtype=torch.bool,
    )
    return (
        context,
        future,
        duration,
        context_mask,
        target_mask,
        context_activity,
        future_activity,
        loss_mask,
    )


def _forward(model: WindowJEPA, values: tuple[torch.Tensor, ...]):
    (
        context,
        future,
        duration,
        context_mask,
        target_mask,
        context_activity,
        future_activity,
        loss_mask,
    ) = values
    return model(
        context,
        future,
        duration,
        duration,
        context_mask,
        target_mask,
        objective="frame_future_jepa",
        sequence_loss_mask=loss_mask,
        context_event_activity=context_activity,
        target_event_activity=future_activity,
    )


def test_frame_future_uses_aligned_future_and_ignores_jepa_masks() -> None:
    torch.manual_seed(31)
    model = _frame_future_model().eval()
    values = _inputs()
    first = _forward(model, values)

    changed_masks = list(values)
    changed_masks[3] = ~values[3]
    changed_masks[4] = ~values[4]
    masked = _forward(model, tuple(changed_masks))
    assert torch.equal(first.prediction, masked.prediction)
    assert torch.equal(first.target, masked.target)

    changed_future = list(values)
    changed_future[1] = values[1] + 7.0
    future = _forward(model, tuple(changed_future))
    assert torch.equal(first.prediction, future.prediction)
    assert not torch.equal(first.target, future.target)

    assert first.online_state is None
    assert first.prediction_sequence is not None
    assert first.prediction_sequence.shape == (2, 2, 4, 32)
    assert first.target_sequence is not None
    assert first.target_sequence.shape == (2, 2, 4, 32)
    assert first.temporal_sigreg_loss is not None
    assert first.temporal_sigreg_loss.item() == 0.0


def test_frame_future_sigreg_backpropagates_only_online() -> None:
    torch.manual_seed(37)
    model = _frame_future_model()
    output = _forward(model, _inputs())

    assert torch.isfinite(output.loss)
    assert output.frame_sigreg_loss is not None
    assert output.support_sigreg_loss is not None
    assert output.frame_sigreg_samples is not None
    assert output.frame_sigreg_samples >= 2

    output.loss.backward()
    assert model.online_encoder.patch_embed.weight.grad is not None
    assert model.predictor.output_projection.weight.grad is not None
    assert model.future_regularizers["frame"].projector.network[0].weight.grad is not None
    assert model.future_regularizers["support"].projector.network[0].weight.grad is not None
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())
