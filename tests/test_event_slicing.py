from __future__ import annotations

import numpy as np
import pytest

from event_window_jepa.data.event_store import (
    InMemoryEventStore,
    _h5_bisect_right,
    _validate_h5_sequence,
)
from event_window_jepa.data.types import SequenceInfo


def make_store() -> InMemoryEventStore:
    arrays = {
        "sequence": {
            "x": np.arange(6, dtype=np.int16),
            "y": np.zeros(6, dtype=np.int16),
            "t_us": np.array([0, 10, 20, 20, 30, 40], dtype=np.int64),
            "polarity": np.array([0, 1, 0, 1, 0, 1], dtype=np.int8),
        }
    }
    metadata = {
        "sequence": SequenceInfo("sequence", None, 8, 8, 0, 40, "train")
    }
    return InMemoryEventStore(arrays, metadata)


def test_slice_uses_open_start_and_closed_end() -> None:
    window = make_store().slice("sequence", t_end_us=30, duration_us=20)
    assert window.t_us.tolist() == [20, 20, 30]
    assert window.t_start_us == 10
    assert window.t_end_us == 30


def test_duplicate_end_timestamps_are_all_included() -> None:
    window = make_store().slice("sequence", t_end_us=20, duration_us=20)
    assert window.t_us.tolist() == [10, 20, 20]


def test_hdf5_binary_search_has_right_boundary_semantics() -> None:
    timestamps = [0, 10, 20, 20, 30]
    assert _h5_bisect_right(timestamps, 10) == 2
    assert _h5_bisect_right(timestamps, 20) == 4


def test_hdf5_schema_rejects_fractional_timestamps() -> None:
    group = {
        "x": np.array([0, 1], dtype=np.int16),
        "y": np.array([0, 1], dtype=np.int16),
        "t_us": np.array([10.0, 20.9], dtype=np.float64),
        "polarity": np.array([0, 1], dtype=np.int8),
    }
    info = SequenceInfo("sequence", None, 8, 8, 0, 30, "train")
    with pytest.raises(TypeError, match="integer dtype"):
        _validate_h5_sequence(group, info)


def test_hdf5_schema_rejects_unsorted_timestamps() -> None:
    group = {
        "x": np.array([0, 1, 2], dtype=np.int16),
        "y": np.array([0, 1, 2], dtype=np.int16),
        "t_us": np.array([10, 30, 20], dtype=np.int64),
        "polarity": np.array([0, 1, 0], dtype=np.int8),
    }
    info = SequenceInfo("sequence", None, 8, 8, 0, 30, "train")
    with pytest.raises(ValueError, match="sorted"):
        _validate_h5_sequence(group, info)


def test_slice_rejects_window_before_sequence_boundary() -> None:
    with pytest.raises(ValueError, match="starts before"):
        make_store().slice("sequence", t_end_us=10, duration_us=20)
