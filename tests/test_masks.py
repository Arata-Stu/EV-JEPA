from __future__ import annotations

import random

import numpy as np

from event_window_jepa.masks.multiblock import MultiBlockMaskGenerator


def test_context_and_target_masks_are_disjoint() -> None:
    generator = MultiBlockMaskGenerator((8, 8))
    masks = generator.sample(random.Random(0))
    assert masks.context_keep.dtype == np.bool_
    assert masks.target.dtype == np.bool_
    assert not np.any(masks.context_keep & masks.target)
    assert masks.context_keep.sum() == round(0.60 * 64)
    assert 0.15 * 64 <= masks.target.sum() <= 0.25 * 64
    assert len(masks.target_blocks) == 4
    assert np.array_equal(np.logical_or.reduce(masks.target_blocks), masks.target)
    for left_index, left in enumerate(masks.target_blocks):
        for right in masks.target_blocks[left_index + 1 :]:
            assert not np.any(left & right)
        rows, columns = np.where(left.reshape(8, 8))
        rectangle_area = (rows.max() - rows.min() + 1) * (columns.max() - columns.min() + 1)
        assert rectangle_area == left.sum()
