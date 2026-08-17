from __future__ import annotations

import torch
from torch import nn

from event_window_jepa.models.token_utils import gather_equal_count, validate_patch_mask


class CrossAttentionBlock(nn.Module):
    """Target queries attend to context without query-to-query communication."""

    def __init__(
        self,
        dimension: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dimension)
        self.memory_norm = nn.LayerNorm(dimension)
        self.cross_attention = nn.MultiheadAttention(
            dimension, num_heads, dropout=dropout, batch_first=True
        )
        self.feed_forward_norm = nn.LayerNorm(dimension)
        hidden_dimension = round(dimension * mlp_ratio)
        self.feed_forward = nn.Sequential(
            nn.Linear(dimension, hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, dimension),
            nn.Dropout(dropout),
        )

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        normalized_queries = self.query_norm(queries)
        normalized_memory = self.memory_norm(memory)
        update, _ = self.cross_attention(
            normalized_queries,
            normalized_memory,
            normalized_memory,
            need_weights=False,
        )
        queries = queries + update
        return queries + self.feed_forward(self.feed_forward_norm(queries))


class WindowPredictor(nn.Module):
    """Cross-attention decoder for accumulation-window latent conversion.

    It queries every spatial position and the objective selects target patches.
    This supports variable target-block area within a batch and enables a full
    canonical latent query without assuming context/target positions are unique.
    """

    def __init__(
        self,
        num_patches: int,
        encoder_dim: int = 384,
        predictor_dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        scale_dim: int = 128,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if predictor_dim % num_heads:
            raise ValueError("predictor_dim must be divisible by num_heads")
        self.num_patches = num_patches
        self.encoder_dim = encoder_dim
        self.scale_dim = scale_dim
        self.context_projection = nn.Linear(encoder_dim, predictor_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, num_patches, predictor_dim))
        self.query_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        condition_dim = scale_dim * 3
        self.context_condition = nn.Sequential(
            nn.Linear(condition_dim, predictor_dim),
            nn.SiLU(),
            nn.Linear(predictor_dim, predictor_dim),
        )
        self.query_condition = nn.Sequential(
            nn.Linear(condition_dim, predictor_dim),
            nn.SiLU(),
            nn.Linear(predictor_dim, predictor_dim),
        )
        self.blocks = nn.ModuleList(
            [
                CrossAttentionBlock(predictor_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(predictor_dim)
        self.output_projection = nn.Linear(predictor_dim, encoder_dim)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.query_token, std=0.02)

    def forward(
        self,
        context_tokens: torch.Tensor,
        context_keep_mask: torch.Tensor,
        target_mask: torch.Tensor,
        source_scale: torch.Tensor,
        target_scale: torch.Tensor,
        ratio_scale: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = context_tokens.shape[0]
        validate_patch_mask(context_keep_mask, batch_size, self.num_patches)
        validate_patch_mask(target_mask, batch_size, self.num_patches)
        projected_context = self.context_projection(context_tokens)
        position_embedding = self.position_embedding.to(dtype=projected_context.dtype)
        context_positions = gather_equal_count(
            position_embedding.expand(batch_size, -1, -1), context_keep_mask
        )
        condition = torch.cat([source_scale, target_scale, ratio_scale], dim=-1)
        memory = projected_context + context_positions
        memory = memory + self.context_condition(condition).unsqueeze(1)

        queries = self.query_token.to(dtype=memory.dtype).expand(batch_size, self.num_patches, -1)
        queries = queries + position_embedding
        queries = queries + self.query_condition(condition).unsqueeze(1)
        # Queries are independent. Therefore changing the number of requested
        # target patches cannot create a train/inference self-attention shift.
        for block in self.blocks:
            queries = block(queries, memory)
        return self.output_projection(self.norm(queries))
