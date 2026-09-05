from __future__ import annotations

import random

import numpy as np

from event_window_jepa.data.spatial_transforms import SharedRandomSpatialTransform
from event_window_jepa.data.types import EventWindow


def make_window(timestamps: list[int]) -> EventWindow:
    return EventWindow(
        x=np.array([1, 2, 3], dtype=np.int16),
        y=np.array([1, 2, 3], dtype=np.int16),
        t_us=np.asarray(timestamps, dtype=np.int64),
        polarity=np.ones(3, dtype=np.int8),
        t_start_us=0,
        t_end_us=10,
        height=4,
        width=4,
    )


def test_one_transform_state_is_shared_by_context_and_target() -> None:
    transform = SharedRandomSpatialTransform((4, 4), horizontal_flip_probability=1.0)
    params = transform.sample(random.Random(4), 4, 4)
    context = transform.apply(make_window([2, 4, 6]), params)
    target = transform.apply(make_window([3, 5, 7]), params)
    assert context.x.tolist() == target.x.tolist() == [2, 1, 0]
    assert context.y.tolist() == target.y.tolist() == [1, 2, 3]


def test_center_padding_shifts_events_without_inventing_events() -> None:
    transform = SharedRandomSpatialTransform(
        (8, 10),
        horizontal_flip_probability=0.0,
        allow_center_padding=True,
    )
    params = transform.sample(random.Random(0), 4, 4)
    padded = transform.apply(make_window([2, 4, 6]), params)

    assert (params.x0, params.y0) == (-3, -2)
    assert (padded.height, padded.width) == (8, 10)
    assert padded.x.tolist() == [4, 5, 6]
    assert padded.y.tolist() == [3, 4, 5]
    assert padded.event_count == 3


def test_padding_requires_explicit_opt_in() -> None:
    transform = SharedRandomSpatialTransform((8, 8))
    try:
        transform.sample(random.Random(0), 4, 4)
    except ValueError as error:
        assert "without padding" in str(error)
    else:
        raise AssertionError("oversized crop unexpectedly padded without opt-in")
