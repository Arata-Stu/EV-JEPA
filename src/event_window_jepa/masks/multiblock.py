from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class MaskPair:
    """Boolean patch masks; True denotes a retained or queried patch."""

    context_keep: NDArray[np.bool_]
    target: NDArray[np.bool_]
    target_blocks: tuple[NDArray[np.bool_], ...]


class MultiBlockMaskGenerator:
    """Generate fixed-cardinality, disjoint context and target masks.

    ``target_area_range`` describes the *union* of all target blocks. This is
    explicit because interpreting it per block would make four 25% targets
    incompatible with a disjoint 60% context.
    """

    def __init__(
        self,
        grid_size: tuple[int, int],
        target_blocks: int = 4,
        target_area_range: tuple[float, float] = (0.15, 0.25),
        target_aspect_range: tuple[float, float] = (0.5, 2.0),
        context_keep_ratio: float = 0.60,
    ) -> None:
        self.grid_height, self.grid_width = grid_size
        self.num_patches = self.grid_height * self.grid_width
        self.target_blocks = target_blocks
        self.target_area_range = target_area_range
        self.target_aspect_range = target_aspect_range
        self.context_keep_ratio = context_keep_ratio
        if min(grid_size) <= 0 or target_blocks <= 0:
            raise ValueError("grid dimensions and target_blocks must be positive")
        if not 0 < target_area_range[0] <= target_area_range[1] < 1:
            raise ValueError("target_area_range must lie in (0, 1)")
        if not 0 < target_aspect_range[0] <= target_aspect_range[1]:
            raise ValueError("target_aspect_range must be positive")
        if not 0 < context_keep_ratio < 1:
            raise ValueError("context_keep_ratio must lie in (0, 1)")
        maximum_target = math.ceil(target_area_range[1] * self.num_patches)
        context_count = round(context_keep_ratio * self.num_patches)
        if math.floor(target_area_range[1] * self.num_patches) < target_blocks:
            raise ValueError("patch grid is too small for the requested number of target blocks")
        if maximum_target + context_count > self.num_patches:
            raise ValueError("context and target masks cannot be made disjoint")

    def _sample_rectangle(self, rng: random.Random, desired_area: int) -> set[int]:
        log_min = math.log(self.target_aspect_range[0])
        log_max = math.log(self.target_aspect_range[1])
        aspect = math.exp(rng.uniform(log_min, log_max))
        height = max(1, round(math.sqrt(desired_area / aspect)))
        width = max(1, round(math.sqrt(desired_area * aspect)))
        height = min(height, self.grid_height)
        width = min(width, self.grid_width)
        top = rng.randint(0, self.grid_height - height)
        left = rng.randint(0, self.grid_width - width)
        return {
            row * self.grid_width + column
            for row in range(top, top + height)
            for column in range(left, left + width)
        }

    def sample(self, rng: random.Random) -> MaskPair:
        minimum_count = math.ceil(self.target_area_range[0] * self.num_patches)
        maximum_count = math.floor(self.target_area_range[1] * self.num_patches)
        all_indices = list(range(self.num_patches))
        sampled_blocks: list[set[int]] | None = None
        for _ in range(100):
            desired_count = rng.randint(minimum_count, maximum_count)
            candidate_blocks: list[set[int]] = []
            occupied: set[int] = set()
            for block_index in range(self.target_blocks):
                remaining = max(1, desired_count - len(occupied))
                blocks_left = self.target_blocks - block_index
                desired_block_area = max(1, round(remaining / blocks_left))
                block = None
                for _ in range(100):
                    candidate = self._sample_rectangle(rng, desired_block_area)
                    if not (candidate & occupied):
                        block = candidate
                        break
                if block is None:
                    break
                candidate_blocks.append(block)
                occupied.update(block)
            if (
                len(candidate_blocks) == self.target_blocks
                and minimum_count <= len(occupied) <= maximum_count
            ):
                sampled_blocks = candidate_blocks
                break
        if sampled_blocks is None:
            raise RuntimeError("could not place disjoint target rectangles on the patch grid")
        target_indices = set().union(*sampled_blocks)

        context_count = round(self.context_keep_ratio * self.num_patches)
        context_candidates = [index for index in all_indices if index not in target_indices]
        context_indices = set(rng.sample(context_candidates, context_count))

        context_mask = np.zeros(self.num_patches, dtype=np.bool_)
        target_mask = np.zeros(self.num_patches, dtype=np.bool_)
        context_mask[list(context_indices)] = True
        target_mask[list(target_indices)] = True
        block_masks: list[NDArray[np.bool_]] = []
        for block in sampled_blocks:
            block_mask = np.zeros(self.num_patches, dtype=np.bool_)
            block_mask[list(block)] = True
            block_masks.append(block_mask)
        return MaskPair(
            context_keep=context_mask,
            target=target_mask,
            target_blocks=tuple(block_masks),
        )
