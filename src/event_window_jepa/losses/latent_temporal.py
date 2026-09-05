from __future__ import annotations

import math
from dataclasses import dataclass

import torch


RATE_NORMALIZATION = "per_clip_mean_supported_patch_rate"


@dataclass(frozen=True)
class LatentTemporalRegularizationOutput:
    """Diagnostics for the window-level RA/LS adaptation.

    The source method regularizes event-level, pre-quantization logits.  This
    project has neither event-wise logits nor a codebook, so the adaptation is
    deliberately applied to causal recurrent patch tokens ``[B, T, N, D]``.
    It must therefore not be reported as an event-level reproduction.
    """

    rate_alignment_loss: torch.Tensor
    rate_alignment_pairs: torch.Tensor
    rate_alignment_mean_weight: torch.Tensor
    latent_straightening_loss: torch.Tensor
    latent_straightening_pairs: torch.Tensor


def _validate_inputs(
    latent_tokens: torch.Tensor,
    event_activity: torch.Tensor,
    duration_ms: torch.Tensor,
    *,
    minimum_events: int,
) -> None:
    if latent_tokens.ndim != 4:
        raise ValueError("latent_tokens must have shape [B,T,N,D]")
    if event_activity.shape != latent_tokens.shape[:3]:
        raise ValueError("event_activity must have shape [B,T,N]")
    if duration_ms.shape != latent_tokens.shape[:2]:
        raise ValueError("duration_ms must have shape [B,T]")
    if latent_tokens.shape[-1] <= 0:
        raise ValueError("latent token dimension must be positive")
    if (
        isinstance(minimum_events, bool)
        or not isinstance(minimum_events, int)
        or minimum_events <= 0
    ):
        raise ValueError("minimum_events must be a positive integer")
    if not bool(torch.isfinite(duration_ms).all()) or not bool(
        (duration_ms > 0).all()
    ):
        raise ValueError("duration_ms must contain finite positive values")
    if not bool(torch.isfinite(event_activity).all()) or not bool(
        (event_activity >= 0).all()
    ):
        raise ValueError("event_activity must contain finite non-negative values")


def window_level_latent_temporal_regularization(
    latent_tokens: torch.Tensor,
    event_activity: torch.Tensor,
    duration_ms: torch.Tensor,
    *,
    minimum_events: int = 1,
    rate_gamma: float = 1.0,
    rate_eps: float = 1e-6,
    straightening_eps: float = 1e-6,
    rate_normalization: str = RATE_NORMALIZATION,
) -> LatentTemporalRegularizationOutput:
    """Compute patchwise Rate Alignment and Latent Straightening.

    Rate Alignment first forms events per millisecond for every event-supported
    patch-window.  Patch area is constant on the model grid and therefore
    cancels in the following per-clip supported-rate normalization.  The
    dimensionless adjacent change is the symmetric relative difference

    ``|r[t]-r[t-1]| / (0.5*(r[t]+r[t-1]) + eps)``.

    Its exponential weight is detached from the graph.  Thus the result does
    not depend on whether time is expressed in Hz or events/ms, nor on a common
    rescaling of event counts.  Unsupported (including fully padded) patches
    are excluded rather than treated as zero-rate observations.
    The squared latent L2 term is divided by ``D`` before the valid-pair mean,
    keeping its scale comparable when the encoder width changes.

    Latent Straightening compares directions of two consecutive recurrent
    token differences.  Triples containing an unsupported patch or a
    near-zero difference are excluded.  In particular, an all-constant latent
    trajectory receives no straightening reward and no direct collapse
    gradient from this term.
    """

    _validate_inputs(
        latent_tokens,
        event_activity,
        duration_ms,
        minimum_events=minimum_events,
    )
    if rate_normalization != RATE_NORMALIZATION:
        raise ValueError(
            "rate_normalization must be "
            f"{RATE_NORMALIZATION!r}, got {rate_normalization!r}"
        )
    if not math.isfinite(rate_gamma) or rate_gamma < 0:
        raise ValueError("rate_gamma must be finite and non-negative")
    if not math.isfinite(rate_eps) or rate_eps <= 0:
        raise ValueError("rate_eps must be finite and positive")
    if not math.isfinite(straightening_eps) or straightening_eps <= 0:
        raise ValueError("straightening_eps must be finite and positive")

    # Keep reductions numerically stable under fp16/bf16 autocast.  Event
    # activity is metadata and every quantity derived from it is detached so
    # RA cannot learn to manipulate its own weighting signal.
    tokens = (
        latent_tokens
        if latent_tokens.dtype in {torch.float32, torch.float64}
        else latent_tokens.float()
    )
    activity = event_activity.detach().to(dtype=tokens.dtype)
    durations = duration_ms.detach().to(dtype=tokens.dtype).unsqueeze(-1)
    support = activity >= float(minimum_events)
    zero = tokens.sum() * 0.0

    if latent_tokens.shape[1] >= 2:
        with torch.no_grad():
            raw_rate = activity / durations
            support_float = support.to(dtype=raw_rate.dtype)
            clip_rate_scale = (
                (raw_rate * support_float).sum(dim=(1, 2), keepdim=True)
                / support_float.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
            )
            # Clips without any supported patch stay invalid below; the clamp
            # only makes their unused normalized values finite.  A dtype tiny
            # is used here (not the dimensionless RA epsilon) so changing the
            # time unit cannot change the normalization rule.
            normalized_rate = raw_rate / clip_rate_scale.clamp_min(
                torch.finfo(raw_rate.dtype).tiny
            )
            left_rate = normalized_rate[:, :-1]
            right_rate = normalized_rate[:, 1:]
            normalized_difference = (right_rate - left_rate).abs() / (
                0.5 * (right_rate.abs() + left_rate.abs()) + rate_eps
            )
            rate_weights = torch.exp(-float(rate_gamma) * normalized_difference)
            pair_support = support[:, :-1] & support[:, 1:]
            pair_support_float = pair_support.to(dtype=tokens.dtype)

        squared_change = (tokens[:, 1:] - tokens[:, :-1]).square().mean(dim=-1)
        rate_pair_count = pair_support_float.sum()
        rate_alignment_loss = (
            (squared_change * rate_weights * pair_support_float).sum()
            / rate_pair_count.clamp_min(1.0)
        )
        rate_mean_weight = (
            (rate_weights * pair_support_float).sum()
            / rate_pair_count.clamp_min(1.0)
        )
    else:
        rate_alignment_loss = zero
        rate_pair_count = zero.detach()
        rate_mean_weight = zero.detach()

    if latent_tokens.shape[1] >= 3:
        previous_delta = tokens[:, 1:-1] - tokens[:, :-2]
        next_delta = tokens[:, 2:] - tokens[:, 1:-1]
        previous_norm = torch.linalg.vector_norm(previous_delta, dim=-1)
        next_norm = torch.linalg.vector_norm(next_delta, dim=-1)
        triple_support = support[:, :-2] & support[:, 1:-1] & support[:, 2:]
        # Norm-based validity is a discrete selection rule and must not add a
        # gradient incentive to make already-small motion exactly zero.
        moving = (
            (previous_norm.detach() > straightening_eps)
            & (next_norm.detach() > straightening_eps)
        )
        straight_support = triple_support & moving
        straight_support_float = straight_support.to(dtype=tokens.dtype)
        cosine = torch.nn.functional.cosine_similarity(
            previous_delta,
            next_delta,
            dim=-1,
            eps=straightening_eps,
        ).clamp(min=-1.0, max=1.0)
        straight_pair_count = straight_support_float.sum()
        latent_straightening_loss = (
            ((1.0 - cosine) * straight_support_float).sum()
            / straight_pair_count.clamp_min(1.0)
        )
    else:
        latent_straightening_loss = zero
        straight_pair_count = zero.detach()

    return LatentTemporalRegularizationOutput(
        rate_alignment_loss=rate_alignment_loss,
        rate_alignment_pairs=rate_pair_count.detach(),
        rate_alignment_mean_weight=rate_mean_weight.detach(),
        latent_straightening_loss=latent_straightening_loss,
        latent_straightening_pairs=straight_pair_count.detach(),
    )
