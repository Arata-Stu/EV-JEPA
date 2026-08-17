from __future__ import annotations

import numpy as np

from event_window_jepa.data.event_store import InMemoryEventStore
from event_window_jepa.data.types import SequenceInfo


def test_window_never_crosses_label_or_anchor_time() -> None:
    timestamps = np.arange(0, 101, 5, dtype=np.int64)
    store = InMemoryEventStore(
        {
            "s": {
                "x": np.zeros_like(timestamps),
                "y": np.zeros_like(timestamps),
                "t_us": timestamps,
                "polarity": np.ones_like(timestamps),
            }
        },
        {"s": SequenceInfo("s", None, 2, 2, 0, 100, "test")},
    )
    label_time_us = 70
    window = store.slice("s", label_time_us, 30)
    assert np.all(window.t_us <= label_time_us)
    assert 75 not in window.t_us

