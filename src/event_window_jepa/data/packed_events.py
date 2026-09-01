from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import default_collate

from event_window_jepa.data.types import EventWindow


RAW_EVENT_WINDOWS_KEY = "_raw_event_windows"
PACKED_EVENT_BATCH_KEY = "packed_events"


@dataclass(frozen=True)
class PackedEventBatch:
    """Padding-free raw events for a recurrent ``[B,T]`` window batch.

    Events are ordered by flattened window index ``batch * T + time`` and then
    by timestamp inside each window. ``window_offsets`` therefore has shape
    ``[B*T + 1]`` and slices every window, including empty ones, without
    padding. Coordinates are post-crop/post-flip pixel coordinates, ``t`` is
    normalized independently to ``[0, 1]`` in every causal window, and
    ``polarity`` always uses ``{-1, +1}``.
    """

    x: torch.Tensor
    y: torch.Tensor
    t: torch.Tensor
    polarity: torch.Tensor
    batch_index: torch.Tensor
    time_index: torch.Tensor
    window_offsets: torch.Tensor
    window_counts: torch.Tensor
    batch_size: int
    time_steps: int
    height: int
    width: int

    @property
    def event_count(self) -> int:
        return int(self.x.numel())

    @property
    def window_count(self) -> int:
        return self.batch_size * self.time_steps

    @property
    def device(self) -> torch.device:
        return self.x.device

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> PackedEventBatch:
        """Move every packed tensor together while preserving integer fields."""

        return PackedEventBatch(
            x=self.x.to(device=device, non_blocking=non_blocking),
            y=self.y.to(device=device, non_blocking=non_blocking),
            t=self.t.to(device=device, non_blocking=non_blocking),
            polarity=self.polarity.to(device=device, non_blocking=non_blocking),
            batch_index=self.batch_index.to(
                device=device, non_blocking=non_blocking
            ),
            time_index=self.time_index.to(device=device, non_blocking=non_blocking),
            window_offsets=self.window_offsets.to(
                device=device, non_blocking=non_blocking
            ),
            window_counts=self.window_counts.to(
                device=device, non_blocking=non_blocking
            ),
            batch_size=self.batch_size,
            time_steps=self.time_steps,
            height=self.height,
            width=self.width,
        )

    def pin_memory(self) -> PackedEventBatch:
        """Support PyTorch's custom-object DataLoader pin-memory walker."""

        return PackedEventBatch(
            x=self.x.pin_memory(),
            y=self.y.pin_memory(),
            t=self.t.pin_memory(),
            polarity=self.polarity.pin_memory(),
            batch_index=self.batch_index.pin_memory(),
            time_index=self.time_index.pin_memory(),
            window_offsets=self.window_offsets.pin_memory(),
            window_counts=self.window_counts.pin_memory(),
            batch_size=self.batch_size,
            time_steps=self.time_steps,
            height=self.height,
            width=self.width,
        )

    def select_time_range(self, start: int, end: int) -> PackedEventBatch:
        """Select ``[start,end)`` for every batch row and rebase time indices."""

        if isinstance(start, bool) or isinstance(end, bool):
            raise TypeError("time range bounds must be integers")
        if not isinstance(start, int) or not isinstance(end, int):
            raise TypeError("time range bounds must be integers")
        if not 0 <= start < end <= self.time_steps:
            raise ValueError("time range must satisfy 0 <= start < end <= time_steps")

        selected_steps = end - start
        keep = (self.time_index >= start) & (self.time_index < end)
        counts = self.window_counts.reshape(self.batch_size, self.time_steps)
        selected_counts = counts[:, start:end].reshape(-1)
        offsets = torch.empty(
            selected_counts.numel() + 1,
            dtype=torch.int64,
            device=self.window_offsets.device,
        )
        offsets[0] = 0
        torch.cumsum(selected_counts, dim=0, out=offsets[1:])
        return PackedEventBatch(
            x=self.x[keep],
            y=self.y[keep],
            t=self.t[keep],
            polarity=self.polarity[keep],
            batch_index=self.batch_index[keep],
            time_index=self.time_index[keep] - start,
            window_offsets=offsets,
            window_counts=selected_counts,
            batch_size=self.batch_size,
            time_steps=selected_steps,
            height=self.height,
            width=self.width,
        )


def _normalized_time(window: EventWindow) -> np.ndarray:
    if not window.event_count:
        return np.empty(0, dtype=np.float32)
    normalized = (
        window.t_us.astype(np.float64, copy=False) - float(window.t_start_us)
    ) / float(window.duration_us)
    return np.ascontiguousarray(normalized, dtype=np.float32)


def _concatenate(values: Sequence[np.ndarray], dtype: np.dtype[Any]) -> torch.Tensor:
    array = np.ascontiguousarray(np.concatenate(values), dtype=dtype)
    return torch.from_numpy(array)


def pack_event_windows(
    batch_windows: Sequence[Sequence[EventWindow]],
) -> PackedEventBatch:
    """Pack equal-length window sequences without padding variable event axes."""

    if not batch_windows:
        raise ValueError("event packing requires at least one batch row")
    time_steps = len(batch_windows[0])
    if time_steps <= 0:
        raise ValueError("event packing requires at least one online timestep")
    if any(len(windows) != time_steps for windows in batch_windows):
        raise ValueError("all event sequences must share the same online length")
    flattened = tuple(window for windows in batch_windows for window in windows)
    if any(not isinstance(window, EventWindow) for window in flattened):
        raise TypeError("raw event payloads must contain EventWindow objects")

    height = flattened[0].height
    width = flattened[0].width
    if any((window.height, window.width) != (height, width) for window in flattened):
        raise ValueError("all packed event windows must share transformed geometry")

    counts = torch.tensor(
        [window.event_count for window in flattened],
        dtype=torch.int64,
    )
    offsets = torch.empty(len(flattened) + 1, dtype=torch.int64)
    offsets[0] = 0
    torch.cumsum(counts, dim=0, out=offsets[1:])

    x = _concatenate([window.x for window in flattened], np.dtype(np.float32))
    y = _concatenate([window.y for window in flattened], np.dtype(np.float32))
    t = _concatenate(
        [_normalized_time(window) for window in flattened],
        np.dtype(np.float32),
    )
    polarity = _concatenate(
        [
            np.where(window.polarity > 0, 1.0, -1.0).astype(
                np.float32, copy=False
            )
            for window in flattened
        ],
        np.dtype(np.float32),
    )

    flat_window_index = torch.repeat_interleave(
        torch.arange(len(flattened), dtype=torch.int64),
        counts,
    )
    return PackedEventBatch(
        x=x,
        y=y,
        t=t,
        polarity=polarity,
        batch_index=flat_window_index // time_steps,
        time_index=flat_window_index % time_steps,
        window_offsets=offsets,
        window_counts=counts,
        batch_size=len(batch_windows),
        time_steps=time_steps,
        height=height,
        width=width,
    )


def collate_recurrent_samples(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Use default collation, except for optional padding-free raw events."""

    if not samples:
        raise ValueError("cannot collate an empty recurrent batch")
    raw_presence = [RAW_EVENT_WINDOWS_KEY in sample for sample in samples]
    if any(raw_presence) and not all(raw_presence):
        raise ValueError("raw event payload must be present in every batch row")
    if not any(raw_presence):
        return default_collate(samples)

    batch_windows: list[Sequence[EventWindow]] = []
    tensor_samples: list[dict[str, Any]] = []
    for sample in samples:
        windows = sample[RAW_EVENT_WINDOWS_KEY]
        if not isinstance(windows, (list, tuple)):
            raise TypeError("raw event payload must be a sequence of EventWindow objects")
        batch_windows.append(windows)
        tensor_samples.append(
            {
                key: value
                for key, value in sample.items()
                if key != RAW_EVENT_WINDOWS_KEY
            }
        )

    batch = default_collate(tensor_samples)
    batch[PACKED_EVENT_BATCH_KEY] = pack_event_windows(batch_windows)
    return batch


__all__ = [
    "PACKED_EVENT_BATCH_KEY",
    "RAW_EVENT_WINDOWS_KEY",
    "PackedEventBatch",
    "collate_recurrent_samples",
    "pack_event_windows",
]
