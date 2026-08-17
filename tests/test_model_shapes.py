from __future__ import annotations

import torch

from event_window_jepa.models.event_vit import EventVisionTransformer
from event_window_jepa.models.scale_embedding import LogFourierScaleEmbedding
from event_window_jepa.models.window_jepa import WindowJEPA
from event_window_jepa.models.window_predictor import WindowPredictor


def make_model() -> WindowJEPA:
    encoder = EventVisionTransformer(
        image_size=(16, 16),
        patch_size=8,
        input_channels=10,
        embed_dim=32,
        depth=1,
        num_heads=4,
        scale_dim=16,
    )
    predictor = WindowPredictor(
        num_patches=4,
        encoder_dim=32,
        predictor_dim=16,
        depth=1,
        num_heads=4,
        scale_dim=16,
    )
    scale = LogFourierScaleEmbedding(output_dim=16, num_bands=4)
    return WindowJEPA(encoder, predictor, scale)


def test_encoder_and_canonical_shapes() -> None:
    model = make_model()
    x = torch.randn(2, 10, 16, 16)
    duration = torch.tensor([10.0, 80.0])
    assert model.encode_only(x, duration).shape == (2, 4, 32)
    assert model.canonical_latent(x, duration, 40.0).shape == (2, 4, 32)


def test_target_encoder_is_frozen_and_receives_no_gradient() -> None:
    model = make_model()
    x_context = torch.randn(2, 10, 16, 16)
    x_target = torch.randn(2, 10, 16, 16)
    context_mask = torch.tensor([[1, 1, 0, 0], [1, 0, 1, 0]], dtype=torch.bool)
    target_mask = ~context_mask
    output = model(
        x_context,
        x_target,
        torch.tensor([10.0, 20.0]),
        torch.tensor([40.0, 40.0]),
        context_mask,
        target_mask,
    )
    output.loss.backward()
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())
    assert all(parameter.grad is None for parameter in model.target_scale_embedding.parameters())
    assert any(parameter.grad is not None for parameter in model.online_encoder.parameters())
