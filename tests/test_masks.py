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


def test_activity_aware_mask_prefers_candidates_with_active_targets() -> None:
    generator = MultiBlockMaskGenerator(
        (8, 8),
        activity_aware_probability=1.0,
        activity_candidates=4,
        minimum_active_target_ratio=0.5,
    )
    masks = generator.sample(
        random.Random(2), activity=np.ones((8, 8), dtype=np.int64)
    )

    assert masks.activity_aware is True
    assert masks.activity_fallback is False
    assert masks.selection_active_patch_ratio == 1.0
    assert 0.15 <= masks.selection_event_mass_coverage <= 0.25


def test_activity_aware_mask_falls_back_for_an_empty_context() -> None:
    generator = MultiBlockMaskGenerator(
        (8, 8), activity_aware_probability=1.0
    )
    masks = generator.sample(
        random.Random(3), activity=np.zeros((8, 8), dtype=np.int64)
    )

    assert masks.activity_aware is False
    assert masks.activity_fallback is True
    assert masks.selection_active_patch_ratio == 0.0
    assert masks.selection_event_mass_coverage == 0.0
