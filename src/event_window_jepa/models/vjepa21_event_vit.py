from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn

from event_window_jepa.models.token_utils import gather_equal_count, validate_patch_mask


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    paired = x.reshape(*x.shape[:-1], -1, 2)
    first, second = paired.unbind(dim=-1)
    return torch.stack((-second, first), dim=-1).flatten(-2)


class RotaryEmbedding2D(nn.Module):
    """Two-dimensional RoPE for a flattened image-patch grid."""

    def __init__(self, head_dim: int, base: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 4:
            raise ValueError("RoPE attention head dimension must be divisible by four")
        axis_dim = head_dim // 2
        inverse_frequency = 1.0 / (
            base ** (torch.arange(0, axis_dim, 2, dtype=torch.float32) / axis_dim)
        )
        self.register_buffer("inverse_frequency", inverse_frequency, persistent=False)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if positions.ndim != 3 or positions.shape[-1] != 2:
            raise ValueError("RoPE positions must have shape [B, N, 2]")
        if q.shape != k.shape or q.shape[0] != positions.shape[0]:
            raise ValueError("RoPE query, key, and position batches do not match")
        if q.shape[-2] != positions.shape[1] + 1:
            raise ValueError("RoPE positions must describe every token except the scale token")

        frequencies = self.inverse_frequency.to(device=q.device)
        y_angles = positions[..., 0].float().unsqueeze(-1) * frequencies
        x_angles = positions[..., 1].float().unsqueeze(-1) * frequencies
        angles = torch.cat((y_angles, x_angles), dim=-1).repeat_interleave(2, dim=-1)
        cosine = angles.cos().to(dtype=q.dtype).unsqueeze(1)
        sine = angles.sin().to(dtype=q.dtype).unsqueeze(1)

        def apply(tensor: torch.Tensor) -> torch.Tensor:
            scale_token, patches = tensor[..., :1, :], tensor[..., 1:, :]
            patches = patches * cosine + _rotate_half(patches) * sine
            return torch.cat((scale_token, patches), dim=-2)

        return apply(q), apply(k)


class GlobalRoPEAttention(nn.Module):
    """Global multi-head self-attention using PyTorch SDPA and 2-D RoPE."""

    def __init__(self, dimension: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if dimension % num_heads:
            raise ValueError("attention dimension must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dimension // num_heads
        self.dropout = dropout
        self.qkv = nn.Linear(dimension, dimension * 3, bias=True)
        self.projection = nn.Linear(dimension, dimension)
        self.rope = RotaryEmbedding2D(self.head_dim)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, dimension = x.shape
        qkv = self.qkv(x).reshape(
            batch_size, token_count, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        q, k = self.rope(q, k, positions)
        attended = functional.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(batch_size, token_count, dimension)
        return self.projection(attended)


class VJEPA21Block(nn.Module):
    def __init__(
        self,
        dimension: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dimension = round(dimension * mlp_ratio)
        self.attention_norm = nn.LayerNorm(dimension, eps=1e-6)
        self.attention = GlobalRoPEAttention(dimension, num_heads, dropout)
        self.mlp_norm = nn.LayerNorm(dimension, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dimension, hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, dimension),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), positions)
        return x + self.mlp(self.mlp_norm(x))


def default_supervision_layers(depth: int) -> tuple[int, ...]:
    """Four approximately uniform, zero-based block indices, including the last."""

    if depth <= 0:
        raise ValueError("encoder depth must be positive")
    count = min(4, depth)
    layers = {
        round((index + 1) * depth / count) - 1
        for index in range(count)
    }
    layers.add(depth - 1)
    return tuple(sorted(layers))


class VJEPA21EventVisionTransformer(nn.Module):
    """Flat, global event ViT adapted from the V-JEPA 2.1 encoder design.

    Temporal event bins remain input channels. The architecture uses a Conv2d
    patch projection, pre-norm transformer blocks, global SDPA, and 2-D RoPE.
    Intermediate outputs retain one common spatial grid; they are not an FPN.
    """

    def __init__(
        self,
        image_size: tuple[int, int] = (224, 224),
        patch_size: int = 16,
        input_channels: int = 10,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        scale_dim: int = 128,
        supervision_layers: tuple[int, ...] = (),
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        height, width = image_size
        if height % patch_size or width % patch_size:
            raise ValueError("image dimensions must be divisible by patch_size")
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        if (embed_dim // num_heads) % 4:
            raise ValueError("V-JEPA 2.1 attention head dimension must be divisible by four")
        selected_layers = supervision_layers or default_supervision_layers(depth)
        if (
            len(set(selected_layers)) != len(selected_layers)
            or not selected_layers
            or min(selected_layers) < 0
            or max(selected_layers) >= depth
        ):
            raise ValueError("supervision layers must be unique block indices within depth")

        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = (height // patch_size, width // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.embed_dim = embed_dim
        self.scale_dim = scale_dim
        self.supervision_layers = tuple(sorted(selected_layers))
        self.patch_embed = nn.Conv2d(
            input_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.scale_projection = nn.Linear(scale_dim, embed_dim)
        self.blocks = nn.ModuleList(
            [
                VJEPA21Block(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.kaiming_normal_(self.patch_embed.weight, mode="fan_out")
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _patch_positions(
        batch_size: int,
        grid_size: tuple[int, int],
        device: torch.device,
    ) -> torch.Tensor:
        rows = torch.arange(grid_size[0], device=device)
        columns = torch.arange(grid_size[1], device=device)
        y, x = torch.meshgrid(rows, columns, indexing="ij")
        positions = torch.stack((y, x), dim=-1).reshape(1, -1, 2)
        return positions.expand(batch_size, -1, -1)

    def _forward_intermediates(
        self,
        x: torch.Tensor,
        scale_embedding: torch.Tensor,
        context_keep_mask: torch.Tensor | None,
        *,
        require_configured_size: bool,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[int, int]]:
        if x.ndim != 4:
            raise ValueError("expected event images with shape [B, C, H, W]")
        if require_configured_size and tuple(x.shape[-2:]) != self.image_size:
            raise ValueError(
                f"expected images [B,C,{self.image_size[0]},{self.image_size[1]}]"
            )
        if x.shape[-2] % self.patch_size or x.shape[-1] % self.patch_size:
            raise ValueError("input image dimensions must be divisible by patch_size")
        batch_size = x.shape[0]
        patches_2d = self.patch_embed(x)
        grid_size = (patches_2d.shape[-2], patches_2d.shape[-1])
        patches = patches_2d.flatten(2).transpose(1, 2)
        positions = self._patch_positions(batch_size, grid_size, x.device)
        if context_keep_mask is not None:
            validate_patch_mask(context_keep_mask, batch_size, patches.shape[1])
            patches = gather_equal_count(patches, context_keep_mask)
            positions = gather_equal_count(positions, context_keep_mask)

        scale_token = self.scale_projection(scale_embedding).unsqueeze(1)
        sequence = torch.cat((scale_token, patches), dim=1)
        outputs: list[torch.Tensor] = []
        selected = set(self.supervision_layers)
        for index, block in enumerate(self.blocks):
            sequence = block(sequence, positions)
            if index in selected:
                outputs.append(self.norm(sequence)[:, 1:, :])
        return tuple(outputs), grid_size

    def forward_intermediates(
        self,
        x: torch.Tensor,
        scale_embedding: torch.Tensor,
        context_keep_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        outputs, _ = self._forward_intermediates(
            x,
            scale_embedding,
            context_keep_mask,
            require_configured_size=True,
        )
        return outputs

    def forward(
        self,
        x: torch.Tensor,
        scale_embedding: torch.Tensor,
        context_keep_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_intermediates(x, scale_embedding, context_keep_mask)[-1]

    def forward_feature_map(
        self, x: torch.Tensor, scale_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Return the final token grid, allowing detection-time padded resolution."""

        outputs, grid_size = self._forward_intermediates(
            x,
            scale_embedding,
            None,
            require_configured_size=False,
        )
        tokens = outputs[-1]
        return tokens.transpose(1, 2).reshape(
            x.shape[0], self.embed_dim, grid_size[0], grid_size[1]
        )
