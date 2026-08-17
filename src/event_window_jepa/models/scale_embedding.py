from __future__ import annotations

import math

import torch
from torch import nn


class LogFourierScaleEmbedding(nn.Module):
    """Fourier-feature MLP over log accumulation duration."""

    def __init__(
        self,
        output_dim: int = 128,
        num_bands: int = 16,
        maximum_frequency: float = 16.0,
        hidden_dim: int | None = None,
        minimum_duration_ms: float = 1e-3,
    ) -> None:
        super().__init__()
        if output_dim <= 0 or num_bands <= 0 or maximum_frequency <= 0:
            raise ValueError("embedding dimensions and frequency must be positive")
        if minimum_duration_ms <= 0:
            raise ValueError("minimum_duration_ms must be positive")
        hidden_dim = hidden_dim or output_dim * 2
        frequencies = torch.logspace(
            0.0,
            math.log10(maximum_frequency),
            num_bands,
            dtype=torch.float32,
        )
        self.register_buffer("frequencies", frequencies, persistent=True)
        self.minimum_duration_ms = minimum_duration_ms
        feature_dim = 1 + 2 * num_bands
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        self.output_dim = output_dim

    def embed_log_duration(self, log_duration: torch.Tensor) -> torch.Tensor:
        log_duration = log_duration.to(dtype=self.frequencies.dtype)
        angles = log_duration.unsqueeze(-1) * self.frequencies * math.pi
        features = torch.cat(
            [log_duration.unsqueeze(-1), torch.sin(angles), torch.cos(angles)], dim=-1
        )
        return self.mlp(features)

    def forward(self, duration_ms: torch.Tensor) -> torch.Tensor:
        if torch.any(duration_ms <= 0):
            raise ValueError("accumulation durations must be positive")
        log_duration = torch.log(duration_ms.clamp_min(self.minimum_duration_ms))
        return self.embed_log_duration(log_duration)

    def ratio(self, target_ms: torch.Tensor, context_ms: torch.Tensor) -> torch.Tensor:
        if torch.any(target_ms <= 0) or torch.any(context_ms <= 0):
            raise ValueError("accumulation durations must be positive")
        log_ratio = torch.log(target_ms) - torch.log(context_ms)
        return self.embed_log_duration(log_ratio)

