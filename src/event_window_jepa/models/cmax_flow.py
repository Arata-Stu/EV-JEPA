from __future__ import annotations

import math

import torch
from torch import nn


class RecurrentTokenFlowHead(nn.Module):
    """Decode recurrent patch tokens into bounded patch-grid optical flow.

    The returned channels are ``(dx, dy)`` in pixels per base event window.
    ``head_depth`` counts the output convolution, so a depth-one head is a
    single convolution from normalized tokens to two flow-logit channels.
    """

    def __init__(
        self,
        embed_dim: int,
        *,
        hidden_dim: int = 256,
        head_depth: int = 2,
        flow_scale: float = 0.01,
        max_displacement: float = 32.0,
    ) -> None:
        super().__init__()
        if isinstance(embed_dim, bool) or not isinstance(embed_dim, int):
            raise TypeError("embed_dim must be an integer")
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int):
            raise TypeError("hidden_dim must be an integer")
        if isinstance(head_depth, bool) or not isinstance(head_depth, int):
            raise TypeError("head_depth must be an integer")
        if embed_dim <= 0 or hidden_dim <= 0 or head_depth <= 0:
            raise ValueError("flow-head dimensions and depth must be positive")
        if not math.isfinite(flow_scale) or flow_scale <= 0:
            raise ValueError("flow_scale must be finite and positive")
        if not math.isfinite(max_displacement) or max_displacement <= 0:
            raise ValueError("max_displacement must be finite and positive")

        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.head_depth = int(head_depth)
        self.flow_scale = float(flow_scale)
        self.max_displacement = float(max_displacement)

        self.normalization = nn.LayerNorm(self.embed_dim)
        layers: list[nn.Module] = []
        input_channels = self.embed_dim
        for _ in range(self.head_depth - 1):
            layers.extend(
                (
                    nn.Conv2d(
                        input_channels,
                        self.hidden_dim,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.GELU(),
                )
            )
            input_channels = self.hidden_dim
        output_layer = nn.Conv2d(input_channels, 2, kernel_size=3, padding=1)
        layers.append(output_layer)
        self.network = nn.Sequential(*layers)

        # A tiny non-zero flow is intentional: raw event coordinates are often
        # integer-valued, where a zero-initialized bilinear splat can have a
        # locally flat average-timestamp objective. ``flow_scale`` keeps this
        # random initialization safely in the sub-pixel regime.
        nn.init.trunc_normal_(output_layer.weight, std=0.02)
        if output_layer.bias is not None:
            nn.init.zeros_(output_layer.bias)

    def forward(
        self,
        recurrent_tokens: torch.Tensor,
        grid_size: tuple[int, int],
    ) -> torch.Tensor:
        """Return ``[B,2,Hg,Wg]`` bounded flow for ``[B,Hg*Wg,D]`` tokens."""

        if recurrent_tokens.ndim != 3:
            raise ValueError(
                "recurrent_tokens must have shape [B,N,D], got "
                f"{tuple(recurrent_tokens.shape)}"
            )
        if not recurrent_tokens.is_floating_point():
            raise TypeError("recurrent_tokens must be floating point")
        if recurrent_tokens.shape[0] <= 0:
            raise ValueError("recurrent_tokens require a non-empty batch")
        if recurrent_tokens.shape[2] != self.embed_dim:
            raise ValueError(
                f"expected token dimension {self.embed_dim}, got "
                f"{recurrent_tokens.shape[2]}"
            )
        if (
            not isinstance(grid_size, tuple)
            or len(grid_size) != 2
            or any(isinstance(size, bool) or not isinstance(size, int) for size in grid_size)
        ):
            raise TypeError("grid_size must be a pair of integers")
        grid_height, grid_width = grid_size
        if grid_height <= 0 or grid_width <= 0:
            raise ValueError("grid_size entries must be positive")
        if recurrent_tokens.shape[1] != grid_height * grid_width:
            raise ValueError(
                "token count does not match grid_size: "
                f"{recurrent_tokens.shape[1]} != {grid_height}*{grid_width}"
            )

        features = self.normalization(recurrent_tokens)
        features = features.transpose(1, 2).reshape(
            recurrent_tokens.shape[0],
            self.embed_dim,
            grid_height,
            grid_width,
        )
        raw_logits = self.network(features)
        return self.max_displacement * torch.tanh(self.flow_scale * raw_logits)


__all__ = ["RecurrentTokenFlowHead"]
