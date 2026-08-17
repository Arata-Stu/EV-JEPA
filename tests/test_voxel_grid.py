from __future__ import annotations

import numpy as np
import pytest

from event_window_jepa.data.types import EventWindow
from event_window_jepa.representations.event_image import EventImage
from event_window_jepa.representations.voxel_grid import VoxelGrid


def make_window(timestamps: list[int], polarities: list[int]) -> EventWindow:
    size = len(timestamps)
    return EventWindow(
        x=np.zeros(size, dtype=np.int16),
        y=np.zeros(size, dtype=np.int16),
        t_us=np.asarray(timestamps, dtype=np.int64),
        polarity=np.asarray(polarities, dtype=np.int8),
        t_start_us=0,
        t_end_us=100,
        height=2,
        width=2,
    )


def test_voxel_temporal_interpolation_and_polarity_order() -> None:
    voxel = VoxelGrid(temporal_bins=3, normalization="none")
    output = voxel(make_window([25, 100], [1, -1]))
    assert output.shape == (6, 2, 2)
    # ON channels are 3..5. t=25 lies halfway between bins 0 and 1.
    assert output[3, 0, 0] == 0.5
    assert output[4, 0, 0] == 0.5
    # OFF at the closed end belongs entirely to final temporal bin 2.
    assert output[2, 0, 0] == 1.0
    assert float(output.sum()) == 2.0


def test_empty_window_has_finite_zero_representations() -> None:
    window = make_window([], [])
    voxel = VoxelGrid(5)(window)
    image = EventImage()(window)
    assert voxel.shape == (10, 2, 2)
    assert image.shape == (2, 2, 2)
    assert np.isfinite(voxel).all() and not voxel.any()
    assert np.isfinite(image).all() and not image.any()


def test_log1p_does_not_normalize_by_total_event_count() -> None:
    output = EventImage(normalization="log1p")(make_window([10, 20], [1, 1]))
    assert np.isclose(output[1, 0, 0], np.log1p(2.0))


def test_inconsistent_polarity_encoding_is_rejected() -> None:
    with pytest.raises(ValueError, match="polarity encoding"):
        make_window([10, 20], [-1, 0])


def test_out_of_bounds_coordinates_are_rejected() -> None:
    with pytest.raises(ValueError, match="coordinates"):
        EventWindow(
            x=np.array([2], dtype=np.int16),
            y=np.array([0], dtype=np.int16),
            t_us=np.array([10], dtype=np.int64),
            polarity=np.array([1], dtype=np.int8),
            t_start_us=0,
            t_end_us=20,
            height=2,
            width=2,
        )
