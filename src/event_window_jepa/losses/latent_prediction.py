from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as functional


@dataclass(frozen=True)
class BalancedLatentPredictionOutput:
    """Sample-balanced future-latent loss and its support diagnostics.

    ``active_loss`` and ``inactive_loss`` average per-sample class means rather
    than individual tokens. Consequently, an event-rich sample cannot dominate
    another sample merely because it contains more active patches. Counts are
    tensors so callers can aggregate them without converting inside a model
    forward pass.
    """

    loss: torch.Tensor
    active_loss: torch.Tensor
    inactive_loss: torch.Tensor
    active_token_count: torch.Tensor
    inactive_token_count: torch.Tensor
    active_sample_count: torch.Tensor
    inactive_sample_count: torch.Tensor
    valid_sample_count: torch.Tensor


def _per_token_latent_prediction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    kind: str,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must share shape [B, N, D]")
    if prediction.shape[1] <= 0 or prediction.shape[2] <= 0:
        raise ValueError("latent prediction requires non-empty token and feature axes")
    prediction = functional.layer_norm(prediction, (prediction.shape[-1],))
    target = functional.layer_norm(target.detach(), (target.shape[-1],))
    if kind == "smooth_l1":
        return functional.smooth_l1_loss(
            prediction, target, reduction="none"
        ).mean(dim=-1)
    if kind == "cosine":
        return 1.0 - functional.cosine_similarity(prediction, target, dim=-1)
    raise ValueError("loss kind must be smooth_l1 or cosine")


def latent_prediction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor | None = None,
    kind: str = "smooth_l1",
) -> torch.Tensor:
    """Feature-normalized latent loss, averaged over selected target tokens."""

    per_token = _per_token_latent_prediction_loss(
        prediction,
        target,
        kind=kind,
    )

    if target_mask is None:
        return per_token.mean()
    if target_mask.dtype is not torch.bool or target_mask.shape != per_token.shape:
        raise ValueError("target_mask must be boolean with shape [B, N]")
    count = target_mask.sum()
    if count == 0:
        raise ValueError("target_mask selects no tokens")
    return per_token.masked_select(target_mask).sum() / count


def balanced_event_support_latent_prediction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    event_activity: torch.Tensor,
    target_mask: torch.Tensor | None = None,
    *,
    active_threshold: float = 0.0,
    kind: str = "smooth_l1",
) -> BalancedLatentPredictionOutput:
    """Balance active and inactive future-patch losses within every sample.

    The target encoder is expected to have processed the complete, unmasked
    future frame before this function is called. ``target_mask`` therefore only
    restricts which already-computed target tokens contribute to the loss; it
    must never be used to sparsify the teacher input.

    If both support classes occur in a sample they receive equal weight. If
    only one class occurs, that class receives the full sample weight. A sample
    with no eligible token is skipped. If every sample is empty, the returned
    loss is a differentiable zero rather than a NaN, and
    ``valid_sample_count`` reports zero.
    """

    per_token = _per_token_latent_prediction_loss(
        prediction,
        target,
        kind=kind,
    )
    if event_activity.shape != per_token.shape:
        raise ValueError("event_activity must have shape [B, N]")
    if event_activity.dtype == torch.bool or event_activity.is_complex():
        raise TypeError("event_activity must contain real-valued event counts")
    if event_activity.device != prediction.device:
        raise ValueError("event_activity and prediction must share a device")
    if not math.isfinite(active_threshold) or active_threshold < 0:
        raise ValueError("active_threshold must be finite and non-negative")
    if event_activity.is_floating_point() and not bool(
        torch.isfinite(event_activity).all()
    ):
        raise ValueError("event_activity must be finite")
    if bool((event_activity < 0).any()):
        raise ValueError("event_activity cannot contain negative counts")

    if target_mask is None:
        eligible = torch.ones_like(event_activity, dtype=torch.bool)
    else:
        if target_mask.dtype != torch.bool or target_mask.shape != per_token.shape:
            raise ValueError("target_mask must be boolean with shape [B, N]")
        if target_mask.device != prediction.device:
            raise ValueError("target_mask and prediction must share a device")
        eligible = target_mask

    active = eligible & (event_activity > active_threshold)
    inactive = eligible & ~active
    active_counts = active.sum(dim=1)
    inactive_counts = inactive.sum(dim=1)
    active_valid = active_counts > 0
    inactive_valid = inactive_counts > 0

    active_means = (per_token * active.to(per_token.dtype)).sum(dim=1)
    active_means = active_means / active_counts.clamp_min(1).to(per_token.dtype)
    inactive_means = (per_token * inactive.to(per_token.dtype)).sum(dim=1)
    inactive_means = inactive_means / inactive_counts.clamp_min(1).to(per_token.dtype)

    class_counts = active_valid.to(per_token.dtype) + inactive_valid.to(per_token.dtype)
    sample_losses = (
        active_means * active_valid.to(per_token.dtype)
        + inactive_means * inactive_valid.to(per_token.dtype)
    ) / class_counts.clamp_min(1.0)
    sample_valid = class_counts > 0
    valid_sample_count = sample_valid.sum()
    loss = (sample_losses * sample_valid.to(per_token.dtype)).sum()
    loss = loss / valid_sample_count.clamp_min(1).to(per_token.dtype)

    active_sample_count = active_valid.sum()
    inactive_sample_count = inactive_valid.sum()
    active_loss = (active_means * active_valid.to(per_token.dtype)).sum()
    active_loss = active_loss / active_sample_count.clamp_min(1).to(per_token.dtype)
    inactive_loss = (inactive_means * inactive_valid.to(per_token.dtype)).sum()
    inactive_loss = inactive_loss / inactive_sample_count.clamp_min(1).to(per_token.dtype)

    return BalancedLatentPredictionOutput(
        loss=loss,
        active_loss=active_loss,
        inactive_loss=inactive_loss,
        active_token_count=active_counts.sum(),
        inactive_token_count=inactive_counts.sum(),
        active_sample_count=active_sample_count,
        inactive_sample_count=inactive_sample_count,
        valid_sample_count=valid_sample_count,
    )
