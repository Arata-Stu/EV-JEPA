from __future__ import annotations

import torch
import torch.nn.functional as functional


@torch.no_grad()
def latent_diagnostics(features: torch.Tensor) -> dict[str, float]:
    if features.ndim != 3:
        raise ValueError("features must have shape [B, N, D]")
    flattened = features.reshape(-1, features.shape[-1]).float()
    centered = flattened - flattened.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    probabilities = singular_values.square()
    probabilities = probabilities / probabilities.sum().clamp_min(1e-12)
    effective_rank = torch.exp(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
    normalized = functional.normalize(flattened, dim=-1)
    if len(normalized) > 1:
        adjacent_cosine = (normalized[:-1] * normalized[1:]).sum(dim=-1).mean()
    else:
        adjacent_cosine = torch.tensor(float("nan"), device=features.device)
    return {
        "feature_std": float(flattened.std(dim=0, unbiased=False).mean()),
        "mean_token_norm": float(flattened.norm(dim=-1).mean()),
        "adjacent_cosine": float(adjacent_cosine),
        "effective_rank": float(effective_rank),
    }


@torch.no_grad()
def event_count_norm_correlation(
    features: torch.Tensor, event_counts: torch.Tensor
) -> float:
    if features.shape[0] != event_counts.numel():
        raise ValueError("one event count is required per batch item")
    norms = features.float().mean(dim=1).norm(dim=-1)
    counts = event_counts.reshape(-1).float()
    norms = norms - norms.mean()
    counts = counts - counts.mean()
    denominator = torch.sqrt(norms.square().sum() * counts.square().sum()).clamp_min(1e-12)
    return float((norms * counts).sum() / denominator)

