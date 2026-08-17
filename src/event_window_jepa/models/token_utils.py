from __future__ import annotations

import torch


def validate_patch_mask(mask: torch.Tensor, batch_size: int, num_patches: int) -> None:
    if mask.dtype is not torch.bool:
        raise TypeError("patch masks must have dtype torch.bool")
    if mask.shape != (batch_size, num_patches):
        raise ValueError(
            f"expected patch mask {(batch_size, num_patches)}, got {tuple(mask.shape)}"
        )


def gather_equal_count(tokens: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
    """Gather a fixed count of tokens from every batch item."""

    batch_size, num_patches, feature_dim = tokens.shape
    validate_patch_mask(keep_mask, batch_size, num_patches)
    counts = keep_mask.sum(dim=1)
    if not torch.all(counts == counts[0]):
        raise ValueError("every batch item must retain the same number of context patches")
    return tokens[keep_mask].reshape(batch_size, int(counts[0].item()), feature_dim)

