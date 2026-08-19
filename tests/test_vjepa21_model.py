from __future__ import annotations

import torch

from event_window_jepa.models.scale_embedding import LogFourierScaleEmbedding
from event_window_jepa.models.vjepa21_event_vit import VJEPA21EventVisionTransformer
from event_window_jepa.models.window_jepa import WindowJEPA
from event_window_jepa.models.window_predictor import WindowPredictor


def _model() -> WindowJEPA:
    encoder = VJEPA21EventVisionTransformer(
        image_size=(16, 16),
        patch_size=8,
        input_channels=10,
        embed_dim=32,
        depth=4,
        num_heads=4,
        scale_dim=16,
        supervision_layers=(0, 1, 2, 3),
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
        LogFourierScaleEmbedding(output_dim=16, num_bands=4),
    )


def test_vjepa21_encoder_returns_same_grid_at_selected_depths() -> None:
    model = _model()
    x = torch.randn(2, 10, 16, 16)
    scale = model.scale_embedding(torch.tensor([10.0, 40.0]))
    layers = model.online_encoder.forward_intermediates(x, scale)
    assert len(layers) == 4
    assert all(layer.shape == (2, 4, 32) for layer in layers)


def test_dense_objective_predicts_complete_grid_without_target_gradient() -> None:
    model = _model()
    context_mask = torch.tensor([[1, 1, 0, 0], [1, 0, 1, 0]], dtype=torch.bool)
    target_mask = ~context_mask
    output = model(
        torch.randn(2, 10, 16, 16),
        torch.randn(2, 10, 16, 16),
        torch.tensor([10.0, 20.0]),
        torch.tensor([40.0, 40.0]),
        context_mask,
        target_mask,
        objective="dense_window_jepa",
    )
    assert output.prediction.shape == (2, 4, 32)
    assert output.deep_supervision_loss.ndim == 0
    output.loss.backward()
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())
    assert any(parameter.grad is not None for parameter in model.online_encoder.parameters())
