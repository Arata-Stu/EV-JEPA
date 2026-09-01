from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.distributed as distributed
from torch import nn

from event_window_jepa.data.packed_events import PackedEventBatch


# Algorithmic reference: Paredes-Valles et al., "Taming Contrast Maximization
# for Learning Sequential, Low-latency, Event-based Optical Flow", ICCV 2023.
# This packed-event implementation is independently written. The reference
# implementation is MIT licensed: https://github.com/tudelft/taming_event_flow
ReferenceMode = Literal["past", "future", "both"]


def _fp32_context(tensor: torch.Tensor) -> Any:
    """Disable an enclosing training autocast region for sparse CMax numerics."""

    if tensor.device.type in {"cpu", "cuda"}:
        return torch.autocast(device_type=tensor.device.type, enabled=False)
    return nullcontext()


def _autograd_all_reduce_sum(value: torch.Tensor) -> torch.Tensor:
    """Globally sum a tensor while preserving gradients on every DDP rank."""

    if (
        distributed.is_available()
        and distributed.is_initialized()
        and distributed.get_world_size() > 1
    ):
        from torch.distributed.nn.functional import all_reduce

        return all_reduce(value, op=distributed.ReduceOp.SUM)
    return value


def _detached_all_reduce_sum(value: torch.Tensor) -> torch.Tensor:
    """Globally sum diagnostic values without adding an autograd branch."""

    result = value.detach().clone()
    if (
        distributed.is_available()
        and distributed.is_initialized()
        and distributed.get_world_size() > 1
    ):
        distributed.all_reduce(result, op=distributed.ReduceOp.SUM)
    return result


def _validate_image_size(image_size: tuple[int, int]) -> tuple[int, int]:
    if (
        not isinstance(image_size, tuple)
        or len(image_size) != 2
        or any(isinstance(size, bool) or not isinstance(size, int) for size in image_size)
    ):
        raise TypeError("image_size must be a (height, width) integer pair")
    height, width = image_size
    if height <= 0 or width <= 0:
        raise ValueError("image_size entries must be positive")
    return height, width


def _validate_event_vectors(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_index: torch.Tensor,
    time_index: torch.Tensor,
    *,
    batch_size: int,
    time_steps: int,
) -> None:
    tensors = (x, y, batch_index, time_index)
    if any(tensor.ndim != 1 for tensor in tensors):
        raise ValueError("event coordinates and indices must be one-dimensional")
    if len({tensor.numel() for tensor in tensors}) != 1:
        raise ValueError("event coordinates and indices must have equal lengths")
    if not x.is_floating_point() or not y.is_floating_point():
        raise TypeError("event coordinates must be floating point")
    if batch_index.dtype != torch.int64 or time_index.dtype != torch.int64:
        raise TypeError("event batch/time indices must use int64")
    if any(tensor.device != x.device for tensor in tensors[1:]):
        raise ValueError("all event vectors must share a device")
    if x.numel() == 0:
        return
    if not bool(torch.isfinite(x).all() & torch.isfinite(y).all()):
        raise ValueError("event coordinates must be finite")
    if not bool(((batch_index >= 0) & (batch_index < batch_size)).all()):
        raise ValueError("event batch_index lies outside the flow batch")
    if not bool(((time_index >= 0) & (time_index < time_steps)).all()):
        raise ValueError("event time_index lies outside the flow sequence")


def _sample_patch_flow_at_events_unchecked(
    flow_maps: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_index: torch.Tensor,
    time_index: torch.Tensor,
    image_size: tuple[int, int],
) -> torch.Tensor:
    """FP32 sampling kernel for already validated event vectors."""

    _, time_steps, _, grid_height, grid_width = flow_maps.shape
    height, width = image_size
    if x.numel() == 0:
        return flow_maps.reshape(-1)[:0].reshape(0, 2)

    valid = (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
    safe_x = torch.where(valid, x, torch.zeros_like(x))
    safe_y = torch.where(valid, y, torch.zeros_like(y))
    # Recurrent tokens represent patch centers. This is align_corners=False
    # sampling with border padding, expressed sparsely without a dense resize.
    grid_x = (
        (safe_x + 0.5) * float(grid_width) / float(width) - 0.5
    ).clamp(0.0, float(grid_width - 1))
    grid_y = (
        (safe_y + 0.5) * float(grid_height) / float(height) - 0.5
    ).clamp(0.0, float(grid_height - 1))
    x0 = grid_x.floor().to(torch.int64).clamp(0, grid_width - 1)
    y0 = grid_y.floor().to(torch.int64).clamp(0, grid_height - 1)
    x1 = (x0 + 1).clamp(max=grid_width - 1)
    y1 = (y0 + 1).clamp(max=grid_height - 1)
    wx = grid_x - x0.to(torch.float32)
    wy = grid_y - y0.to(torch.float32)

    pixels_per_grid = grid_height * grid_width
    frame_index = batch_index * time_steps + time_index
    flow_by_pixel = flow_maps.permute(0, 1, 3, 4, 2).reshape(-1, 2)

    def gather(grid_row: torch.Tensor, grid_column: torch.Tensor) -> torch.Tensor:
        pixel_index = grid_row * grid_width + grid_column
        return flow_by_pixel[frame_index * pixels_per_grid + pixel_index]

    top_left = gather(y0, x0)
    top_right = gather(y0, x1)
    bottom_left = gather(y1, x0)
    bottom_right = gather(y1, x1)
    sampled = (
        top_left * ((1.0 - wx) * (1.0 - wy)).unsqueeze(1)
        + top_right * (wx * (1.0 - wy)).unsqueeze(1)
        + bottom_left * ((1.0 - wx) * wy).unsqueeze(1)
        + bottom_right * (wx * wy).unsqueeze(1)
    )
    return torch.where(valid.unsqueeze(1), sampled, torch.zeros_like(sampled))


def sample_patch_flow_at_events(
    flow_maps: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_index: torch.Tensor,
    time_index: torch.Tensor,
    image_size: tuple[int, int],
) -> torch.Tensor:
    """Bilinearly sample patch-grid flow at sparse full-resolution events.

    Args:
        flow_maps: ``[B,T,2,Hg,Wg]`` flow in ``(dx,dy)`` pixel units.
        x, y: Event coordinates in the full-resolution image frame.
        batch_index, time_index: Per-event indices into ``flow_maps``.
        image_size: Full-resolution ``(height,width)`` corresponding to events.

    Out-of-image events receive zero flow. The implementation is sparse and
    does not materialize a dense full-resolution flow field.
    """

    if flow_maps.ndim != 5 or flow_maps.shape[2] != 2:
        raise ValueError(
            "flow_maps must have shape [B,T,2,Hg,Wg], got "
            f"{tuple(flow_maps.shape)}"
        )
    if not flow_maps.is_floating_point():
        raise TypeError("flow_maps must be floating point")
    batch_size, time_steps, _, grid_height, grid_width = flow_maps.shape
    if min(batch_size, time_steps, grid_height, grid_width) <= 0:
        raise ValueError("flow_maps batch, time, and grid axes must be non-empty")
    height, width = _validate_image_size(image_size)
    if height % grid_height or width % grid_width:
        raise ValueError("image_size must be divisible by the flow patch grid")
    _validate_event_vectors(
        x,
        y,
        batch_index,
        time_index,
        batch_size=batch_size,
        time_steps=time_steps,
    )
    if flow_maps.device != x.device:
        raise ValueError("flow_maps and event vectors must share a device")

    with _fp32_context(flow_maps):
        return _sample_patch_flow_at_events_unchecked(
            flow_maps.float(),
            x.float(),
            y.float(),
            batch_index,
            time_index,
            (height, width),
        )


def warp_events_to_reference(
    x: torch.Tensor,
    y: torch.Tensor,
    event_time: torch.Tensor,
    event_flow: torch.Tensor,
    reference_time: float | torch.Tensor,
) -> torch.Tensor:
    """Linearly warp sparse events using ``xy + (t_ref - t) * flow``.

    This primitive is useful for one interval. ``TamingCMaxLoss`` uses its
    iterative counterpart internally so that every crossed base window samples
    that window's own recurrent flow map.
    """

    if x.ndim != 1 or y.ndim != 1 or event_time.ndim != 1:
        raise ValueError("x, y, and event_time must be one-dimensional")
    if x.shape != y.shape or x.shape != event_time.shape:
        raise ValueError("x, y, and event_time must have equal shapes")
    if event_flow.shape != (x.numel(), 2):
        raise ValueError("event_flow must have shape [E,2]")
    if not all(tensor.is_floating_point() for tensor in (x, y, event_time, event_flow)):
        raise TypeError("warp inputs must be floating point")
    if any(tensor.device != x.device for tensor in (y, event_time, event_flow)):
        raise ValueError("warp inputs must share a device")

    with _fp32_context(event_flow):
        time = event_time.float()
        if isinstance(reference_time, torch.Tensor):
            if reference_time.numel() != 1 or reference_time.device != x.device:
                raise ValueError("reference_time tensor must be a scalar on the event device")
            reference = reference_time.float()
        else:
            if not math.isfinite(float(reference_time)):
                raise ValueError("reference_time must be finite")
            reference = torch.as_tensor(
                float(reference_time),
                dtype=torch.float32,
                device=x.device,
            )
        coordinates = torch.stack((x.float(), y.float()), dim=1)
        return coordinates + (reference - time).unsqueeze(1) * event_flow.float()


def _bilinear_splat_iwe_unchecked(
    warped_xy: torch.Tensor,
    polarity: torch.Tensor,
    batch_index: torch.Tensor,
    image_size: tuple[int, int],
    *,
    batch_size: int,
    values: torch.Tensor | None,
) -> torch.Tensor:
    """FP32 splatting kernel for already validated packed-event vectors."""

    height, width = image_size
    output_size = batch_size * 2 * height * width
    if warped_xy.shape[0] == 0:
        connection = warped_xy.sum() * 0.0
        if values is not None:
            connection = connection + values.sum() * 0.0
        return torch.zeros(
            batch_size,
            2,
            height,
            width,
            dtype=torch.float32,
            device=warped_xy.device,
        ) + connection

    finite = torch.isfinite(warped_xy).all(dim=1)
    safe_xy = torch.where(
        finite.unsqueeze(1),
        warped_xy,
        torch.zeros_like(warped_xy),
    )
    x = safe_xy[:, 0]
    y = safe_xy[:, 1]
    x0 = x.floor().to(torch.int64)
    y0 = y.floor().to(torch.int64)
    neighbor_x = torch.stack((x0, x0 + 1, x0, x0 + 1), dim=1)
    neighbor_y = torch.stack((y0, y0, y0 + 1, y0 + 1), dim=1)
    weights = (1.0 - (x.unsqueeze(1) - neighbor_x).abs()) * (
        1.0 - (y.unsqueeze(1) - neighbor_y).abs()
    )
    in_bounds = (
        finite.unsqueeze(1)
        & (neighbor_x >= 0)
        & (neighbor_x < width)
        & (neighbor_y >= 0)
        & (neighbor_y < height)
    )
    weights = torch.where(in_bounds, weights, torch.zeros_like(weights))
    safe_neighbor_x = neighbor_x.clamp(0, width - 1)
    safe_neighbor_y = neighbor_y.clamp(0, height - 1)
    polarity_channel = (polarity < 0).to(torch.int64).unsqueeze(1)
    flat_index = (
        (
            (batch_index.unsqueeze(1) * 2 + polarity_channel) * height
            + safe_neighbor_y
        )
        * width
        + safe_neighbor_x
    )
    if values is not None:
        weights = weights * values.unsqueeze(1)
    output = torch.zeros(
        output_size,
        dtype=torch.float32,
        device=warped_xy.device,
    ).scatter_add(0, flat_index.reshape(-1), weights.reshape(-1))
    return output.reshape(batch_size, 2, height, width)


def bilinear_splat_iwe(
    warped_xy: torch.Tensor,
    polarity: torch.Tensor,
    batch_index: torch.Tensor,
    image_size: tuple[int, int],
    *,
    batch_size: int,
    values: torch.Tensor | None = None,
) -> torch.Tensor:
    """Polarity-aware bilinear splat into an ``[B,2,H,W]`` event image.

    Channel zero is positive polarity and channel one is negative polarity.
    Contributions whose bilinear neighbors lie outside the image are safely
    discarded. ``values=None`` splats unit event counts.
    """

    height, width = _validate_image_size(image_size)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    event_count = warped_xy.shape[0] if warped_xy.ndim == 2 else -1
    if warped_xy.ndim != 2 or warped_xy.shape[1] != 2:
        raise ValueError("warped_xy must have shape [E,2]")
    if polarity.ndim != 1 or batch_index.ndim != 1:
        raise ValueError("polarity and batch_index must be one-dimensional")
    if polarity.numel() != event_count or batch_index.numel() != event_count:
        raise ValueError("splat inputs must have the same event count")
    if not warped_xy.is_floating_point() or not polarity.is_floating_point():
        raise TypeError("warped coordinates and polarity must be floating point")
    if batch_index.dtype != torch.int64:
        raise TypeError("batch_index must use int64")
    if any(tensor.device != warped_xy.device for tensor in (polarity, batch_index)):
        raise ValueError("splat inputs must share a device")
    if values is not None:
        if values.ndim != 1 or values.numel() != event_count:
            raise ValueError("values must have shape [E]")
        if not values.is_floating_point():
            raise TypeError("values must be floating point")
        if values.device != warped_xy.device:
            raise ValueError("values and warped_xy must share a device")
    if event_count:
        if not bool(((batch_index >= 0) & (batch_index < batch_size)).all()):
            raise ValueError("batch_index lies outside the requested splat batch")
        if not bool(((polarity == 1) | (polarity == -1)).all()):
            raise ValueError("polarity must contain only -1 and +1")

    with _fp32_context(warped_xy):
        return _bilinear_splat_iwe_unchecked(
            warped_xy.float(),
            polarity.float(),
            batch_index,
            (height, width),
            batch_size=batch_size,
            values=None if values is None else values.float(),
        )


def average_timestamp_image(
    iwe: torch.Tensor,
    timestamp_sum: torch.Tensor,
    *,
    epsilon: float = 1e-9,
) -> torch.Tensor:
    """Convert bilinearly splatted timestamp sums to per-polarity averages."""

    if iwe.shape != timestamp_sum.shape or iwe.ndim != 4 or iwe.shape[1] != 2:
        raise ValueError("iwe and timestamp_sum must share shape [B,2,H,W]")
    if not iwe.is_floating_point() or not timestamp_sum.is_floating_point():
        raise TypeError("timestamp images must be floating point")
    if iwe.device != timestamp_sum.device:
        raise ValueError("iwe and timestamp_sum must share a device")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")

    with _fp32_context(iwe):
        counts = iwe.float()
        return timestamp_sum.float() / (counts + float(epsilon))


def taming_focus_loss(
    iwe: torch.Tensor,
    average_timestamps: torch.Tensor,
) -> torch.Tensor:
    """Return the Taming CMax average-timestamp focus value per batch row."""

    if iwe.shape != average_timestamps.shape or iwe.ndim != 4 or iwe.shape[1] != 2:
        raise ValueError("iwe and average_timestamps must share shape [B,2,H,W]")
    if not iwe.is_floating_point() or not average_timestamps.is_floating_point():
        raise TypeError("focus inputs must be floating point")
    if iwe.device != average_timestamps.device:
        raise ValueError("focus inputs must share a device")

    with _fp32_context(iwe):
        occupied = iwe.float().sum(dim=1) > 0
        occupied_count = occupied.flatten(1).sum(dim=1).to(torch.float32)
        numerator = average_timestamps.float().square().flatten(1).sum(dim=1)
        return numerator / occupied_count.clamp_min(1.0)


def charbonnier_spatial_smoothness(
    flow_maps: torch.Tensor,
    *,
    epsilon: float = 1e-3,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Charbonnier penalty over horizontal, vertical, and diagonal flow edges."""

    if flow_maps.ndim not in {4, 5} or flow_maps.shape[-3] != 2:
        raise ValueError("flow_maps must have shape [B,2,H,W] or [B,T,2,H,W]")
    if not flow_maps.is_floating_point():
        raise TypeError("flow_maps must be floating point")
    if flow_maps.numel() == 0:
        raise ValueError("flow_maps must be non-empty")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if not math.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("alpha must lie inside (0,1]")

    with _fp32_context(flow_maps):
        flow = flow_maps.float()
        differences: list[torch.Tensor] = []
        if flow.shape[-1] > 1:
            differences.append(flow[..., :, 1:] - flow[..., :, :-1])
        if flow.shape[-2] > 1:
            differences.append(flow[..., 1:, :] - flow[..., :-1, :])
        if flow.shape[-2] > 1 and flow.shape[-1] > 1:
            differences.extend(
                (
                    flow[..., 1:, 1:] - flow[..., :-1, :-1],
                    flow[..., 1:, :-1] - flow[..., :-1, 1:],
                )
            )
        if not differences:
            return flow.sum() * 0.0
        offset = float(epsilon) ** (2.0 * float(alpha))
        penalties = [
            (difference.square().sum(dim=-3) + float(epsilon) ** 2).pow(alpha)
            - offset
            for difference in differences
        ]
        return torch.stack([penalty.mean() for penalty in penalties]).mean()


@dataclass(frozen=True)
class CMaxOutput:
    """CMax objective and training diagnostics, all represented as FP32 scalars."""

    loss: torch.Tensor
    focus_loss: torch.Tensor
    forward_focus_loss: torch.Tensor
    backward_focus_loss: torch.Tensor
    smoothness_loss: torch.Tensor
    valid_event_count: torch.Tensor
    valid_partition_count: torch.Tensor
    valid_window_fraction: torch.Tensor
    mean_flow_magnitude: torch.Tensor
    occupied_pixel_fraction: torch.Tensor


def _validate_packed_events(events: PackedEventBatch, flow_maps: torch.Tensor) -> None:
    if not isinstance(events, PackedEventBatch):
        raise TypeError("events must be a PackedEventBatch")
    if isinstance(events.batch_size, bool) or not isinstance(events.batch_size, int):
        raise TypeError("PackedEventBatch.batch_size must be an integer")
    if isinstance(events.time_steps, bool) or not isinstance(events.time_steps, int):
        raise TypeError("PackedEventBatch.time_steps must be an integer")
    if events.batch_size <= 0 or events.time_steps <= 0:
        raise ValueError("PackedEventBatch batch and time dimensions must be positive")
    if events.height <= 0 or events.width <= 0:
        raise ValueError("PackedEventBatch image dimensions must be positive")
    if flow_maps.shape[:2] != (events.batch_size, events.time_steps):
        raise ValueError(
            "flow and packed-event batch/time dimensions differ: "
            f"{tuple(flow_maps.shape[:2])} != "
            f"{(events.batch_size, events.time_steps)}"
        )
    grid_height, grid_width = flow_maps.shape[-2:]
    if events.height % grid_height or events.width % grid_width:
        raise ValueError("packed image size must be divisible by the flow patch grid")

    event_vectors = {
        "x": events.x,
        "y": events.y,
        "t": events.t,
        "polarity": events.polarity,
        "batch_index": events.batch_index,
        "time_index": events.time_index,
    }
    if any(tensor.ndim != 1 for tensor in event_vectors.values()):
        raise ValueError("PackedEventBatch event tensors must be one-dimensional")
    if any(tensor.numel() != events.event_count for tensor in event_vectors.values()):
        raise ValueError("PackedEventBatch event tensors must have equal lengths")
    if any(tensor.device != flow_maps.device for tensor in event_vectors.values()):
        raise ValueError("flow_maps and packed event tensors must share a device")
    if any(
        tensor.device != flow_maps.device
        for tensor in (events.window_offsets, events.window_counts)
    ):
        raise ValueError("packed offsets/counts and flow_maps must share a device")
    if not all(
        event_vectors[name].is_floating_point()
        for name in ("x", "y", "t", "polarity")
    ):
        raise TypeError("packed event values must be floating point")
    if events.batch_index.dtype != torch.int64 or events.time_index.dtype != torch.int64:
        raise TypeError("packed batch/time indices must use int64")
    if events.window_offsets.dtype != torch.int64 or events.window_counts.dtype != torch.int64:
        raise TypeError("packed window offsets/counts must use int64")
    if events.window_offsets.shape != (events.window_count + 1,):
        raise ValueError("window_offsets must have shape [B*T+1]")
    if events.window_counts.shape != (events.window_count,):
        raise ValueError("window_counts must have shape [B*T]")
    if bool((events.window_counts < 0).any()):
        raise ValueError("window_counts cannot be negative")
    if not torch.equal(
        events.window_offsets[1:] - events.window_offsets[:-1],
        events.window_counts,
    ):
        raise ValueError("window_offsets and window_counts are inconsistent")
    if int(events.window_offsets[0].item()) != 0:
        raise ValueError("window_offsets must start at zero")
    if int(events.window_offsets[-1].item()) != events.event_count:
        raise ValueError("window_offsets must end at the packed event count")
    if not events.event_count:
        return

    if not bool(
        torch.isfinite(events.x).all()
        & torch.isfinite(events.y).all()
        & torch.isfinite(events.t).all()
        & torch.isfinite(events.polarity).all()
    ):
        raise ValueError("packed event values must be finite")
    in_image = (
        (events.x >= 0)
        & (events.x <= events.width - 1)
        & (events.y >= 0)
        & (events.y <= events.height - 1)
    )
    if not bool(in_image.all()):
        raise ValueError("packed event coordinates lie outside the transformed image")
    if not bool(((events.t >= -1e-6) & (events.t <= 1.0 + 1e-6)).all()):
        raise ValueError("packed event time must lie inside each window's [0,1]")
    if not bool(((events.polarity == 1) | (events.polarity == -1)).all()):
        raise ValueError("packed event polarity must contain only -1 and +1")
    if not bool(
        ((events.batch_index >= 0) & (events.batch_index < events.batch_size)).all()
    ):
        raise ValueError("packed batch_index lies outside the batch")
    if not bool(
        ((events.time_index >= 0) & (events.time_index < events.time_steps)).all()
    ):
        raise ValueError("packed time_index lies outside the sequence")

    flat_window_index = events.batch_index * events.time_steps + events.time_index
    observed_counts = torch.bincount(flat_window_index, minlength=events.window_count)
    if not torch.equal(observed_counts, events.window_counts):
        raise ValueError("packed event indices disagree with window_counts")
    if events.event_count > 1:
        if bool((flat_window_index[1:] < flat_window_index[:-1]).any()):
            raise ValueError("packed events must be ordered by flattened window index")
        same_window = flat_window_index[1:] == flat_window_index[:-1]
        if bool((same_window & (events.t[1:] < events.t[:-1])).any()):
            raise ValueError("packed events must be timestamp-sorted inside each window")


def _iterative_warp_to_endpoint(
    flow_maps: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    local_time: torch.Tensor,
    batch_index: torch.Tensor,
    source_time_index: torch.Tensor,
    image_size: tuple[int, int],
    *,
    partition_start: int,
    partition_end: int,
    reference: Literal["past", "future"],
) -> torch.Tensor:
    coordinates = torch.stack((x.float(), y.float()), dim=1)
    if coordinates.shape[0] == 0:
        return coordinates
    if reference == "future":
        steps = range(partition_start, partition_end)
    else:
        steps = range(partition_end - 1, partition_start - 1, -1)

    zero_time_index = torch.zeros_like(source_time_index)
    for step in steps:
        step_index = zero_time_index + step
        sampled_flow = _sample_patch_flow_at_events_unchecked(
            flow_maps,
            coordinates[:, 0],
            coordinates[:, 1],
            batch_index,
            step_index,
            image_size,
        )
        if reference == "future":
            active = source_time_index <= step
            duration = torch.where(
                source_time_index == step,
                1.0 - local_time.float(),
                torch.ones_like(local_time, dtype=torch.float32),
            )
        else:
            active = source_time_index >= step
            duration = torch.where(
                source_time_index == step,
                -local_time.float(),
                -torch.ones_like(local_time, dtype=torch.float32),
            )
        coordinates = coordinates + (
            active.to(torch.float32) * duration
        ).unsqueeze(1) * sampled_flow
    return coordinates


def _reference_statistics(
    flow_maps: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    local_time: torch.Tensor,
    polarity: torch.Tensor,
    batch_index: torch.Tensor,
    source_time_index: torch.Tensor,
    image_size: tuple[int, int],
    partition_valid: torch.Tensor,
    *,
    partition_start: int,
    partition_end: int,
    reference: Literal["past", "future"],
) -> tuple[torch.Tensor, torch.Tensor]:
    warped = _iterative_warp_to_endpoint(
        flow_maps,
        x,
        y,
        local_time,
        batch_index,
        source_time_index,
        image_size,
        partition_start=partition_start,
        partition_end=partition_end,
        reference=reference,
    )
    global_time = source_time_index.to(torch.float32) + local_time.float()
    duration = float(partition_end - partition_start)
    reference_time = float(partition_end if reference == "future" else partition_start)
    normalized_time = 1.0 - (global_time - reference_time).abs() / duration
    normalized_time = normalized_time.clamp(0.0, 1.0)
    iwe = _bilinear_splat_iwe_unchecked(
        warped,
        polarity,
        batch_index,
        image_size,
        batch_size=flow_maps.shape[0],
        values=None,
    )
    timestamp_sum = _bilinear_splat_iwe_unchecked(
        warped,
        polarity,
        batch_index,
        image_size,
        batch_size=flow_maps.shape[0],
        values=normalized_time,
    )
    average_timestamps = average_timestamp_image(iwe, timestamp_sum)
    focus = taming_focus_loss(iwe, average_timestamps)
    occupied_pixels = (iwe.sum(dim=1) > 0).flatten(1).sum(dim=1).to(torch.float32)
    image_area = float(image_size[0] * image_size[1])
    occupied_fraction = occupied_pixels / image_area
    valid = partition_valid.to(torch.float32)
    return (focus * valid).sum(), (occupied_fraction * valid).sum()


class TamingCMaxLoss(nn.Module):
    """Multi-timescale, bidirectional raw-event contrast-maximization loss.

    This is an independent packed-event implementation informed by the ICCV
    2023 Taming CMax formulation and MIT implementation:
    https://github.com/tudelft/taming_event_flow

    Event time is interpreted as ``time_index + t`` in base-window units. For
    endpoint warping, each crossed window bilinearly samples its own recurrent
    flow map. Events from low-support base windows are removed from the focus
    objective, while their predicted flow remains available when other events
    cross those windows. Each temporal scale is averaged separately, preventing
    scales with more partitions from receiving a larger weight.
    """

    def __init__(
        self,
        *,
        smoothness_weight: float = 0.0,
        reference_mode: ReferenceMode = "both",
        temporal_scales: tuple[int, ...] = (1, 2, 4),
        min_events: int = 128,
        charbonnier_epsilon: float = 1e-3,
        charbonnier_alpha: float = 0.5,
    ) -> None:
        super().__init__()
        if not math.isfinite(smoothness_weight) or smoothness_weight < 0:
            raise ValueError("smoothness_weight must be finite and non-negative")
        if reference_mode not in {"past", "future", "both"}:
            raise ValueError("reference_mode must be past, future, or both")
        if not isinstance(temporal_scales, tuple) or not temporal_scales:
            raise TypeError("temporal_scales must be a non-empty tuple")
        if any(
            isinstance(scale, bool) or not isinstance(scale, int) or scale <= 0
            for scale in temporal_scales
        ):
            raise ValueError("temporal_scales must contain positive integers")
        if tuple(sorted(set(temporal_scales))) != temporal_scales:
            raise ValueError("temporal_scales must be strictly increasing and unique")
        if isinstance(min_events, bool) or not isinstance(min_events, int):
            raise TypeError("min_events must be an integer")
        if min_events <= 0:
            raise ValueError("min_events must be positive")
        if not math.isfinite(charbonnier_epsilon) or charbonnier_epsilon <= 0:
            raise ValueError("charbonnier_epsilon must be finite and positive")
        if not math.isfinite(charbonnier_alpha) or not 0 < charbonnier_alpha <= 1:
            raise ValueError("charbonnier_alpha must lie inside (0,1]")

        self.smoothness_weight = float(smoothness_weight)
        self.reference_mode: ReferenceMode = reference_mode
        self.temporal_scales = temporal_scales
        self.min_events = int(min_events)
        self.charbonnier_epsilon = float(charbonnier_epsilon)
        self.charbonnier_alpha = float(charbonnier_alpha)

    def forward(
        self,
        flow_maps: torch.Tensor,
        events: PackedEventBatch,
    ) -> CMaxOutput:
        if flow_maps.ndim != 5 or flow_maps.shape[2] != 2:
            raise ValueError(
                "flow_maps must have shape [B,T,2,Hg,Wg], got "
                f"{tuple(flow_maps.shape)}"
            )
        if not flow_maps.is_floating_point():
            raise TypeError("flow_maps must be floating point")
        if min(flow_maps.shape) <= 0:
            raise ValueError("flow_maps dimensions must be non-empty")
        _validate_packed_events(events, flow_maps)
        time_steps = flow_maps.shape[1]
        if any(scale > time_steps or time_steps % scale for scale in self.temporal_scales):
            raise ValueError(
                "every temporal scale must divide the number of base windows"
            )

        with _fp32_context(flow_maps):
            flow = flow_maps.float()
            zero = flow.sum() * 0.0
            window_counts = events.window_counts.reshape(
                events.batch_size,
                events.time_steps,
            )
            window_is_valid = window_counts >= self.min_events
            valid_window_fraction = window_is_valid.to(torch.float32).mean()
            if events.event_count:
                source_is_valid = window_is_valid[
                    events.batch_index,
                    events.time_index,
                ]
            else:
                source_is_valid = torch.empty(
                    0,
                    dtype=torch.bool,
                    device=flow.device,
                )
            valid_event_count = source_is_valid.sum().to(torch.float32)

            forward_scale_losses: list[torch.Tensor] = []
            backward_scale_losses: list[torch.Tensor] = []
            scale_validities: list[torch.Tensor] = []
            occupied_sum = zero
            occupied_count = torch.zeros((), dtype=torch.float32, device=flow.device)
            valid_partition_count = torch.zeros(
                (),
                dtype=torch.float32,
                device=flow.device,
            )

            for partition_count in self.temporal_scales:
                partition_width = time_steps // partition_count
                scale_forward_sum = zero
                scale_backward_sum = zero
                scale_valid_count = torch.zeros(
                    (),
                    dtype=torch.float32,
                    device=flow.device,
                )
                for partition_index in range(partition_count):
                    start = partition_index * partition_width
                    end = start + partition_width
                    in_partition = (
                        source_is_valid
                        & (events.time_index >= start)
                        & (events.time_index < end)
                    )
                    batch_counts = torch.zeros(
                        events.batch_size,
                        dtype=torch.float32,
                        device=flow.device,
                    ).scatter_add(
                        0,
                        events.batch_index[in_partition],
                        torch.ones_like(
                            events.batch_index[in_partition],
                            dtype=torch.float32,
                        ),
                    )
                    partition_valid = batch_counts > 0
                    partition_valid_count = partition_valid.sum().to(torch.float32)
                    scale_valid_count = scale_valid_count + partition_valid_count
                    valid_partition_count = (
                        valid_partition_count + partition_valid_count
                    )

                    x = events.x[in_partition]
                    y = events.y[in_partition]
                    local_time = events.t[in_partition]
                    polarity = events.polarity[in_partition]
                    batch_index = events.batch_index[in_partition]
                    source_time_index = events.time_index[in_partition]

                    if self.reference_mode in {"future", "both"}:
                        focus_sum, partition_occupied_sum = _reference_statistics(
                            flow,
                            x,
                            y,
                            local_time,
                            polarity,
                            batch_index,
                            source_time_index,
                            (events.height, events.width),
                            partition_valid,
                            partition_start=start,
                            partition_end=end,
                            reference="future",
                        )
                        scale_forward_sum = scale_forward_sum + focus_sum
                        occupied_sum = occupied_sum + partition_occupied_sum
                        occupied_count = occupied_count + partition_valid_count
                    if self.reference_mode in {"past", "both"}:
                        focus_sum, partition_occupied_sum = _reference_statistics(
                            flow,
                            x,
                            y,
                            local_time,
                            polarity,
                            batch_index,
                            source_time_index,
                            (events.height, events.width),
                            partition_valid,
                            partition_start=start,
                            partition_end=end,
                            reference="past",
                        )
                        scale_backward_sum = scale_backward_sum + focus_sum
                        occupied_sum = occupied_sum + partition_occupied_sum
                        occupied_count = occupied_count + partition_valid_count

                # This packed collective is unconditional and ordered solely
                # by the static temporal_scales configuration. A rank with no
                # local support therefore still participates safely.
                global_statistics = _autograd_all_reduce_sum(
                    torch.stack(
                        (
                            scale_forward_sum,
                            scale_backward_sum,
                            scale_valid_count,
                        )
                    )
                )
                global_forward_sum = global_statistics[0]
                global_backward_sum = global_statistics[1]
                global_valid_count = global_statistics[2]
                denominator = global_valid_count.clamp_min(1.0)
                scale_validities.append(
                    (global_valid_count > 0).to(torch.float32)
                )
                if self.reference_mode in {"future", "both"}:
                    forward_scale_losses.append(global_forward_sum / denominator)
                if self.reference_mode in {"past", "both"}:
                    backward_scale_losses.append(global_backward_sum / denominator)

            validity = torch.stack(scale_validities)

            def valid_scale_mean(values: list[torch.Tensor]) -> torch.Tensor:
                if not values:
                    return zero
                return (torch.stack(values) * validity).sum() / validity.sum().clamp_min(1.0)

            forward_focus = valid_scale_mean(forward_scale_losses)
            backward_focus = valid_scale_mean(backward_scale_losses)
            if self.reference_mode == "future":
                focus_loss = forward_focus
            elif self.reference_mode == "past":
                focus_loss = backward_focus
            else:
                focus_loss = 0.5 * (forward_focus + backward_focus)
            smoothness_loss = charbonnier_spatial_smoothness(
                flow,
                epsilon=self.charbonnier_epsilon,
                alpha=self.charbonnier_alpha,
            )
            total_loss = focus_loss + self.smoothness_weight * smoothness_loss
            mean_flow_magnitude = flow.square().sum(dim=2).sqrt().mean()
            global_occupied = _detached_all_reduce_sum(
                torch.stack((occupied_sum, occupied_count))
            )
            occupied_pixel_fraction = (
                global_occupied[0] / global_occupied[1].clamp_min(1.0)
            )
            return CMaxOutput(
                loss=total_loss.float(),
                focus_loss=focus_loss.float(),
                forward_focus_loss=forward_focus.float(),
                backward_focus_loss=backward_focus.float(),
                smoothness_loss=smoothness_loss.float(),
                valid_event_count=valid_event_count.float(),
                valid_partition_count=valid_partition_count.float(),
                valid_window_fraction=valid_window_fraction.float(),
                mean_flow_magnitude=mean_flow_magnitude.float(),
                occupied_pixel_fraction=occupied_pixel_fraction.float(),
            )


__all__ = [
    "CMaxOutput",
    "TamingCMaxLoss",
    "average_timestamp_image",
    "bilinear_splat_iwe",
    "charbonnier_spatial_smoothness",
    "sample_patch_flow_at_events",
    "taming_focus_loss",
    "warp_events_to_reference",
]
