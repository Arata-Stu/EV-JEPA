from __future__ import annotations

import numpy as np

from event_window_jepa.data.types import EventWindow
from event_window_jepa.representations.voxel_grid import polarity_indices


class EventImage:
    """Two-channel OFF/ON count image used as a representation ablation."""

    channels = 2

    def __init__(self, normalization: str = "log1p") -> None:
        if normalization == "global_log1p":
            normalization = "log1p"
        if normalization not in {"none", "log1p"}:
            raise ValueError("normalization must be none or log1p")
        self.normalization = normalization

    def __call__(self, window: EventWindow) -> np.ndarray:
        image = np.zeros((2, window.height, window.width), dtype=np.float32)
        if window.event_count == 0:
            return image
        valid = (
            (window.x >= 0)
            & (window.x < window.width)
            & (window.y >= 0)
            & (window.y < window.height)
        )
        x = window.x[valid].astype(np.int64, copy=False)
        y = window.y[valid].astype(np.int64, copy=False)
        polarity = polarity_indices(window.polarity[valid])
        np.add.at(image, (polarity, y, x), 1.0)
        if self.normalization == "log1p":
            np.log1p(image, out=image)
        return image

