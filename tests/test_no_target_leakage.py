from __future__ import annotations

import torch

from event_window_jepa.models.event_vit import EventVisionTransformer
from event_window_jepa.models.scale_embedding import LogFourierScaleEmbedding
from event_window_jepa.models.window_jepa import WindowJEPA
from event_window_jepa.models.window_predictor import WindowPredictor


class RecordingEncoder(EventVisionTransformer):
    def forward(self, x: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        self.last_input = x.detach().clone()
        return super().forward(x, *args, **kwargs)


def test_online_path_never_receives_target_input() -> None:
    encoder = RecordingEncoder(
        image_size=(16, 16),
        patch_size=8,
        input_channels=10,
        embed_dim=32,
        depth=1,
        num_heads=4,
        scale_dim=16,
    )
    predictor = WindowPredictor(4, 32, 16, 1, 4, 16)
    model = WindowJEPA(encoder, predictor, LogFourierScaleEmbedding(16, 4))
    context = torch.zeros(1, 10, 16, 16)
    target = torch.ones(1, 10, 16, 16)
    first = model(
        context,
        target,
        torch.tensor([10.0]),
        torch.tensor([40.0]),
        torch.tensor([[1, 1, 0, 0]], dtype=torch.bool),
        torch.tensor([[0, 0, 1, 1]], dtype=torch.bool),
    )
    second = model(
        context,
        target * 7.0,
        torch.tensor([10.0]),
        torch.tensor([40.0]),
        torch.tensor([[1, 1, 0, 0]], dtype=torch.bool),
        torch.tensor([[0, 0, 1, 1]], dtype=torch.bool),
    )
    assert torch.equal(model.online_encoder.last_input, context)
    assert torch.equal(model.target_encoder.last_input, target * 7.0)
    assert torch.equal(first.prediction, second.prediction)
