from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TimestampIndex:
    """Binary-search index for sorted integer-microsecond timestamps."""

    timestamps_us: NDArray[np.integer]

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps_us)
        if timestamps.ndim != 1:
            raise ValueError("timestamps must be one-dimensional")
        if not np.issubdtype(timestamps.dtype, np.integer):
            raise TypeError("timestamps must use an integer dtype")
        if timestamps.size and np.any(timestamps[1:] < timestamps[:-1]):
            raise ValueError("timestamps must be sorted in ascending order")

    def bounds(self, t_start_us: int, t_end_us: int) -> tuple[int, int]:
        """Return indices for the causal interval ``(t_start_us, t_end_us]``.

        ``side='right'`` on both boundaries is intentional: an event exactly at
        the start is excluded, while an event exactly at the end is included.
        """

        if t_end_us <= t_start_us:
            raise ValueError("t_end_us must be greater than t_start_us")
        left = int(np.searchsorted(self.timestamps_us, t_start_us, side="right"))
        right = int(np.searchsorted(self.timestamps_us, t_end_us, side="right"))
        return left, right

