from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from event_window_jepa.data.types import EventWindow


@dataclass(frozen=True)
class SpatialTransformParameters:
    x0: int
    y0: int
    output_height: int
    output_width: int
    horizontal_flip: bool


class SharedRandomSpatialTransform:
    """Samples geometry once and applies it identically to a window pair."""

    def __init__(
        self,
        crop_size: tuple[int, int] | None = None,
        horizontal_flip_probability: float = 0.5,
    ) -> None:
        if crop_size is not None and min(crop_size) <= 0:
            raise ValueError("crop dimensions must be positive")
        if not 0 <= horizontal_flip_probability <= 1:
            raise ValueError("horizontal_flip_probability must lie in [0, 1]")
        self.crop_size = crop_size
        self.horizontal_flip_probability = horizontal_flip_probability

    def sample(
        self, rng: random.Random, input_height: int, input_width: int
    ) -> SpatialTransformParameters:
        output_height, output_width = self.crop_size or (input_height, input_width)
        if output_height > input_height or output_width > input_width:
            raise ValueError("crop cannot exceed the sensor resolution")
        y0 = rng.randint(0, input_height - output_height)
        x0 = rng.randint(0, input_width - output_width)
        return SpatialTransformParameters(
            x0=x0,
            y0=y0,
            output_height=output_height,
            output_width=output_width,
            horizontal_flip=rng.random() < self.horizontal_flip_probability,
        )

    @staticmethod
    def apply(window: EventWindow, params: SpatialTransformParameters) -> EventWindow:
        x1 = params.x0 + params.output_width
        y1 = params.y0 + params.output_height
        keep = (
            (window.x >= params.x0)
            & (window.x < x1)
            & (window.y >= params.y0)
            & (window.y < y1)
        )
        x = window.x[keep].astype(np.int64, copy=True) - params.x0
        y = window.y[keep].astype(np.int64, copy=True) - params.y0
        if params.horizontal_flip:
            x = params.output_width - 1 - x
        return EventWindow(
            x=x,
            y=y,
            t_us=window.t_us[keep],
            polarity=window.polarity[keep],
            t_start_us=window.t_start_us,
            t_end_us=window.t_end_us,
            height=params.output_height,
            width=params.output_width,
        )

