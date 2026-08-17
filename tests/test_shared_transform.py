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

