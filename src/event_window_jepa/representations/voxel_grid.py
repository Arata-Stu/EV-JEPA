from __future__ import annotations

import numpy as np

from event_window_jepa.data.types import EventWindow


def polarity_indices(polarity: np.ndarray) -> np.ndarray:
    """Map {-1, +1} or {0, 1} polarity to {0=OFF, 1=ON}."""

    values = np.asarray(polarity)
    if values.size == 0:
        return values.astype(np.int64)
    unique = set(np.unique(values).tolist())
    if not unique.issubset({-1, 0, 1}) or (-1 in unique and 0 in unique):
        raise ValueError("polarity must be encoded as {-1,+1} or {0,1}")
    return (values > 0).astype(np.int64, copy=False)


class VoxelGrid:
    """Polarity-major event-count voxel grid with temporal interpolation.

    Channel order is ``[OFF-bin-0 ... OFF-bin-(B-1), ON-bin-0 ...]``.
    Relative time is measured against the requested window boundaries, never
    against the first/last observed event. No per-sample count normalization is
    performed.
    """

    def __init__(self, temporal_bins: int = 5, normalization: str = "log1p") -> None:
        if temporal_bins <= 0:
            raise ValueError("temporal_bins must be positive")
        if normalization == "global_log1p":
            normalization = "log1p"
        if normalization not in {"none", "log1p"}:
            raise ValueError("normalization must be none or log1p")
        self.temporal_bins = temporal_bins
        self.normalization = normalization

    @property
    def channels(self) -> int:
        return 2 * self.temporal_bins

    def __call__(self, window: EventWindow) -> np.ndarray:
        grid = np.zeros(
            (2, self.temporal_bins, window.height, window.width), dtype=np.float32
        )
        if window.event_count == 0:
            return grid.reshape(self.channels, window.height, window.width)

        valid = (
            (window.x >= 0)
            & (window.x < window.width)
            & (window.y >= 0)
            & (window.y < window.height)
        )
        x = window.x[valid].astype(np.int64, copy=False)
        y = window.y[valid].astype(np.int64, copy=False)
        t_us = window.t_us[valid].astype(np.float64, copy=False)
        polarity = polarity_indices(window.polarity[valid])
        if x.size == 0:
            return grid.reshape(self.channels, window.height, window.width)

        relative = (t_us - window.t_start_us) / float(window.duration_us)
        relative = np.clip(relative, 0.0, 1.0)
        if self.temporal_bins == 1:
            np.add.at(grid, (polarity, 0, y, x), 1.0)
        else:
            continuous_bin = relative * (self.temporal_bins - 1)
            lower = np.floor(continuous_bin).astype(np.int64)
            upper = np.minimum(lower + 1, self.temporal_bins - 1)
            upper_weight = (continuous_bin - lower).astype(np.float32)
            lower_weight = 1.0 - upper_weight
            np.add.at(grid, (polarity, lower, y, x), lower_weight)
            np.add.at(grid, (polarity, upper, y, x), upper_weight)

        if self.normalization == "log1p":
            np.log1p(grid, out=grid)
        return grid.reshape(self.channels, window.height, window.width)
