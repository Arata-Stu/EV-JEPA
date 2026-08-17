from __future__ import annotations

import torch

from event_window_jepa.models.window_jepa import WindowJEPA


@torch.no_grad()
def extract_patch_features(
    model: WindowJEPA,
    x: torch.Tensor,
    duration_ms: torch.Tensor,
    mode: str = "canonical",
    canonical_ms: float = 40.0,
) -> torch.Tensor:
    """Stable boundary between pretrained representations and task heads."""

    if mode == "encoder_only":
        return model.encode_only(x, duration_ms)
    if mode == "canonical":
        return model.canonical_latent(x, duration_ms, canonical_ms)
    raise ValueError("mode must be encoder_only or canonical")


def tokens_to_feature_map(tokens: torch.Tensor, grid_size: tuple[int, int]) -> torch.Tensor:
    if tokens.ndim != 3:
        raise ValueError("tokens must have shape [B, N, D]")
    height, width = grid_size
    if tokens.shape[1] != height * width:
        raise ValueError("token count does not match grid_size")
    return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], height, width)

