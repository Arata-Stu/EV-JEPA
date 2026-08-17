from __future__ import annotations

import torch
import torch.nn.functional as functional


def latent_prediction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor | None = None,
    kind: str = "smooth_l1",
) -> torch.Tensor:
    """Feature-normalized latent loss, averaged over selected target tokens."""

    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must share shape [B, N, D]")
    prediction = functional.layer_norm(prediction, (prediction.shape[-1],))
    target = functional.layer_norm(target.detach(), (target.shape[-1],))
    if kind == "smooth_l1":
        per_token = functional.smooth_l1_loss(prediction, target, reduction="none").mean(dim=-1)
    elif kind == "cosine":
        per_token = 1.0 - functional.cosine_similarity(prediction, target, dim=-1)
    else:
        raise ValueError("loss kind must be smooth_l1 or cosine")

    if target_mask is None:
        return per_token.mean()
    if target_mask.dtype is not torch.bool or target_mask.shape != per_token.shape:
        raise ValueError("target_mask must be boolean with shape [B, N]")
    count = target_mask.sum()
    if count == 0:
        raise ValueError("target_mask selects no tokens")
    return per_token.masked_select(target_mask).sum() / count

