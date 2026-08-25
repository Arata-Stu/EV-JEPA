from __future__ import annotations

from contextlib import nullcontext

import torch


def _fp32_context(features: torch.Tensor):
    if features.is_cuda:
        return torch.autocast(device_type="cuda", enabled=False)
    return nullcontext()


def variance_regularization(features: torch.Tensor, target_std: float = 1.0) -> torch.Tensor:
    """VICReg-style hinge penalty; intended only after collapse is observed."""

    if features.ndim < 2:
        raise ValueError("features must include samples and a feature dimension")
    with _fp32_context(features):
        flattened = features.reshape(-1, features.shape[-1]).float()
        if flattened.shape[0] < 2:
            raise ValueError("variance regularization requires at least two feature vectors")
        std = torch.sqrt(flattened.var(dim=0, unbiased=False) + 1e-4)
        return torch.relu(target_std - std).mean()


def covariance_regularization(features: torch.Tensor) -> torch.Tensor:
    """Penalize off-diagonal feature covariance as in VICReg."""

    with _fp32_context(features):
        flattened = features.reshape(-1, features.shape[-1]).float()
        if flattened.shape[0] < 2:
            raise ValueError("covariance regularization requires at least two samples")
        centered = flattened - flattened.mean(dim=0, keepdim=True)
        covariance = centered.transpose(0, 1) @ centered / (flattened.shape[0] - 1)
        off_diagonal = ~torch.eye(
            covariance.shape[0], dtype=torch.bool, device=covariance.device
        )
        off_diagonal_energy = covariance.masked_select(off_diagonal).square().sum()
        return off_diagonal_energy / covariance.shape[0]


@torch.no_grad()
def feature_standard_deviation(features: torch.Tensor) -> torch.Tensor:
    with _fp32_context(features):
        flattened = features.reshape(-1, features.shape[-1]).float()
        return flattened.std(dim=0, unbiased=False).mean()


def _masked_position_standard_deviations(
    features: torch.Tensor,
    mask: torch.Tensor,
    *,
    require_repeated_positions: bool = False,
) -> torch.Tensor:

    if features.ndim != 3 or mask.shape != features.shape[:2] or mask.dtype is not torch.bool:
        raise ValueError("features/mask must have shapes [B,N,D] and [B,N]")
    with _fp32_context(features):
        features = features.float()
        weights = mask.unsqueeze(-1).to(dtype=features.dtype)
        counts = weights.sum(dim=0)
        means = (features * weights).sum(dim=0) / counts.clamp_min(1.0)
        variance = ((features - means.unsqueeze(0)).square() * weights).sum(dim=0)
        variance = variance / counts.clamp_min(1.0)
        valid_positions = counts.squeeze(-1) >= 2
        if not bool(valid_positions.any()):
            if require_repeated_positions:
                raise ValueError("no patch position is selected by at least two batch items")
            return features.new_full((0, features.shape[-1]), float("nan"))
        return torch.sqrt(variance[valid_positions] + 1e-4)


def masked_position_standard_deviation(
    features: torch.Tensor,
    mask: torch.Tensor,
    *,
    require_repeated_positions: bool = False,
) -> torch.Tensor:
    """Average batch std at fixed spatial positions selected at least twice."""

    standard_deviations = _masked_position_standard_deviations(
        features,
        mask,
        require_repeated_positions=require_repeated_positions,
    )
    if standard_deviations.numel() == 0:
        return features.new_full((), float("nan"), dtype=torch.float32)
    return standard_deviations.mean()


def masked_position_variance_regularization(
    features: torch.Tensor, mask: torch.Tensor, target_std: float = 1.0
) -> torch.Tensor:
    std = _masked_position_standard_deviations(
        features, mask, require_repeated_positions=True
    )
    return torch.relu(target_std - std).mean()
