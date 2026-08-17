from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


IntArray = NDArray[np.integer]


@dataclass(frozen=True)
class SequenceInfo:
    sequence_id: str
    path: Path | None
    height: int
    width: int
    t_start_us: int
    t_end_us: int
    split: str = "train"
    dataset: str = "unknown"
    source_time_origin_us: int = 0
    coordinate_frame: str = "unknown"
    source_width: int | None = None
    source_height: int | None = None
    spatial_downsample: int = 1
    camera: str = "unknown"
    timestamp_reference: str = "unknown"
    timestamp_synchronized: bool = False
    source_recording_id: str | None = None

    def __post_init__(self) -> None:
        if not self.sequence_id:
            raise ValueError("sequence_id cannot be empty")
        if self.height <= 0 or self.width <= 0:
            raise ValueError("sensor dimensions must be positive")
        if self.t_end_us < self.t_start_us:
            raise ValueError("sequence end cannot precede its start")
        if not self.dataset:
            raise ValueError("dataset cannot be empty")
        if not self.coordinate_frame:
            raise ValueError("coordinate_frame cannot be empty")
        if (self.source_width is None) != (self.source_height is None):
            raise ValueError("source_width and source_height must be provided together")
        if self.source_width is not None and min(self.source_width, self.source_height) <= 0:
            raise ValueError("source resolution must be positive")
        if self.spatial_downsample <= 0:
            raise ValueError("spatial_downsample must be positive")
        if not self.camera or not self.timestamp_reference:
            raise ValueError("camera and timestamp_reference cannot be empty")
        if not isinstance(self.timestamp_synchronized, bool):
            raise TypeError("timestamp_synchronized must be boolean")
        if self.source_recording_id is not None and not self.source_recording_id:
            raise ValueError("source_recording_id cannot be empty")

    def source_to_internal_time(self, timestamp_us: int) -> int:
        return int(timestamp_us) - self.source_time_origin_us

    def internal_to_source_time(self, timestamp_us: int) -> int:
        return int(timestamp_us) + self.source_time_origin_us


@dataclass(frozen=True)
class EventWindow:
    """A causal event interval with semantics ``(t_start_us, t_end_us]``."""

    x: IntArray
    y: IntArray
    t_us: IntArray
    polarity: IntArray
    t_start_us: int
    t_end_us: int
    height: int
    width: int

    def __post_init__(self) -> None:
        for name, values in (
            ("x", self.x),
            ("y", self.y),
            ("t_us", self.t_us),
            ("polarity", self.polarity),
        ):
            if not isinstance(values, np.ndarray) or values.ndim != 1:
                raise TypeError(f"{name} must be a one-dimensional NumPy array")
            is_integer = np.issubdtype(values.dtype, np.integer)
            if name == "polarity":
                is_integer = is_integer or np.issubdtype(values.dtype, np.bool_)
            if not is_integer:
                raise TypeError(f"{name} must use an integer dtype")
        size = len(self.x)
        if not (len(self.y) == len(self.t_us) == len(self.polarity) == size):
            raise ValueError("event fields must have the same length")
        if self.t_end_us <= self.t_start_us:
            raise ValueError("event window duration must be positive")
        if self.height <= 0 or self.width <= 0:
            raise ValueError("event window dimensions must be positive")
        if size:
            if np.any(self.t_us[1:] < self.t_us[:-1]):
                raise ValueError("events must be sorted by timestamp")
            if int(self.t_us[0]) <= self.t_start_us or int(self.t_us[-1]) > self.t_end_us:
                raise ValueError("events fall outside the causal interval (start, end]")
            if (
                np.any(self.x < 0)
                or np.any(self.x >= self.width)
                or np.any(self.y < 0)
                or np.any(self.y >= self.height)
            ):
                raise ValueError("event coordinates fall outside the sensor bounds")
            polarities = set(np.unique(self.polarity).tolist())
            if not polarities.issubset({-1, 0, 1}) or (-1 in polarities and 0 in polarities):
                raise ValueError("polarity encoding must be consistently {-1,+1} or {0,1}")

    @property
    def duration_us(self) -> int:
        return self.t_end_us - self.t_start_us

    @property
    def event_count(self) -> int:
        return len(self.x)
