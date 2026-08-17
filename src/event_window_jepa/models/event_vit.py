from __future__ import annotations

import torch
from torch import nn

from event_window_jepa.models.token_utils import gather_equal_count, validate_patch_mask


class EventVisionTransformer(nn.Module):
    """A compact 2-D ViT whose temporal voxel bins are input channels."""

    def __init__(
        self,
        image_size: tuple[int, int] = (224, 224),
        patch_size: int = 16,
        input_channels: int = 10,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        scale_dim: int = 128,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        height, width = image_size
        if height % patch_size or width % patch_size:
            raise ValueError("image dimensions must be divisible by patch_size")
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = (height // patch_size, width // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.embed_dim = embed_dim
        self.scale_dim = scale_dim
        self.patch_embed = nn.Conv2d(
            input_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.position_embedding = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.scale_projection = nn.Linear(scale_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=round(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.kaiming_normal_(self.patch_embed.weight, mode="fan_out")
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)
        nn.init.trunc_normal_(self.scale_projection.weight, std=0.02)
        nn.init.zeros_(self.scale_projection.bias)

    def forward(
        self,
        x: torch.Tensor,
        scale_embedding: torch.Tensor,
        context_keep_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 4 or tuple(x.shape[-2:]) != self.image_size:
            raise ValueError(
                f"expected images [B,C,{self.image_size[0]},{self.image_size[1]}]"
            )
        batch_size = x.shape[0]
        patches = self.patch_embed(x).flatten(2).transpose(1, 2)
        patches = patches + self.position_embedding.to(dtype=patches.dtype)
        if context_keep_mask is not None:
            validate_patch_mask(context_keep_mask, batch_size, self.num_patches)
            patches = gather_equal_count(patches, context_keep_mask)

        # The scale token is retained even when spatial patches are masked.
        scale_token = self.scale_projection(scale_embedding).unsqueeze(1)
        sequence = torch.cat([scale_token, patches], dim=1)
        sequence = self.blocks(sequence)
        sequence = self.norm(sequence)
        return sequence[:, 1:, :]
