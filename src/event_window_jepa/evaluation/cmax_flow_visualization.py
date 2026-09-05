from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as functional

from event_window_jepa.config import ExperimentConfig
from event_window_jepa.data.packed_events import (
    RAW_EVENT_WINDOWS_KEY,
    PackedEventBatch,
    pack_event_windows,
)
from event_window_jepa.data.types import EventWindow
from event_window_jepa.evaluation.future_feature_data import (
    FutureFeatureMaterialization,
    materialize_future_feature_samples,
    validate_future_feature_config,
)
from event_window_jepa.evaluation.future_feature_visualization import (
    _asset_reference,
    _validate_report_compatibility,
    _write_png,
    _write_text,
    make_history_replacement_clip_permutation,
)
from event_window_jepa.inspection import _display_scale, _event_counts, _event_rgb
from event_window_jepa.losses.cmax import (
    CMaxOutput,
    TamingCMaxLoss,
    average_timestamp_image,
    bilinear_splat_iwe,
    sample_patch_flow_at_events,
    taming_focus_loss,
    warp_events_to_reference,
)
from event_window_jepa.models.window_jepa import WindowJEPA
from event_window_jepa.train.checkpoint import config_hash, load_pretrained_model


_FILE_STAT_KEYS = ("device", "inode", "bytes", "mtime_ns", "ctime_ns")


def _file_stat_identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _snapshot_file_identity(
    path: str | Path,
    *,
    label: str,
) -> dict[str, Any]:
    """Hash one regular file while proving its stat identity stayed stable."""

    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"{label} must be a regular file: {source}")
    before = _file_stat_identity(source)
    sha256 = _sha256_file(source)
    after = _file_stat_identity(source)
    if after != before:
        raise RuntimeError(f"{label} changed while its identity was captured: {source}")
    return {"path": str(source), **before, "sha256": sha256}


def _require_unchanged_stat_identity(
    path: str | Path,
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    try:
        source = Path(path).expanduser().resolve(strict=True)
        current = _file_stat_identity(source)
    except OSError as error:
        raise RuntimeError(f"{label} became unavailable during rendering") from error
    expected_stat = {key: int(expected[key]) for key in _FILE_STAT_KEYS}
    if str(source) != str(expected.get("path")) or current != expected_stat:
        raise RuntimeError(f"{label} changed while the CMax report was active: {source}")


def _require_unchanged_content_identity(
    path: str | Path,
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    current = _snapshot_file_identity(path, label=label)
    if current != dict(expected):
        raise RuntimeError(f"{label} content changed while it was being loaded")


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _validate_output_input_collisions(
    output: str | Path,
    *,
    checkpoint: str | Path,
    manifest: str | Path,
) -> None:
    """Keep every HTML/JSON/asset output disjoint from immutable inputs."""

    html_path = Path(output).expanduser().resolve(strict=False)
    json_path = html_path.with_suffix(".json")
    assets_path = (html_path.parent / f"{html_path.stem}_assets").resolve(
        strict=False
    )
    output_paths = (html_path, json_path, assets_path)
    for label, raw_input in (("checkpoint", checkpoint), ("manifest", manifest)):
        input_path = Path(raw_input).expanduser().resolve(strict=True)
        if (
            input_path in output_paths
            or _is_within(input_path, assets_path)
            or any(_is_within(candidate, input_path) for candidate in output_paths)
        ):
            raise ValueError(
                f"CMax HTML/JSON/assets would collide with the {label}: "
                f"{input_path}"
            )


@dataclass(frozen=True)
class CMaxStepWarpResult:
    """One-window IWE diagnostics, kept on the input flow device."""

    unwarped_iwe: torch.Tensor
    past_iwe: torch.Tensor
    future_iwe: torch.Tensor
    past_focus_loss: torch.Tensor
    future_focus_loss: torch.Tensor
    unwarped_occupied_pixel_fraction: torch.Tensor
    past_occupied_pixel_fraction: torch.Tensor
    future_occupied_pixel_fraction: torch.Tensor
    past_retained_event_fraction: torch.Tensor
    future_retained_event_fraction: torch.Tensor


@dataclass(frozen=True)
class CMaxFlowStepRecord:
    """CPU visualization tensors for one supervised recurrent timestep."""

    record_index: int
    clip_position: int
    sample_index: int
    online_step: int
    sequence_id: str
    t_start_us: int
    t_end_us: int
    raw_event_count: int
    cmax_event_count: int
    cmax_window_valid: bool
    flow_map: torch.Tensor
    unwarped_iwe: torch.Tensor
    past_iwe: torch.Tensor
    future_iwe: torch.Tensor
    past_focus_loss: float
    future_focus_loss: float
    unwarped_occupied_pixel_fraction: float
    past_occupied_pixel_fraction: float
    future_occupied_pixel_fraction: float
    past_retained_event_fraction: float
    future_retained_event_fraction: float


@dataclass(frozen=True)
class CMaxFlowClipRecord:
    """A complete supervised flow sequence and its source-event windows."""

    clip_position: int
    sample_index: int
    sequence_id: str
    supervised_steps: tuple[int, ...]
    flow_maps: torch.Tensor
    cmax_windows: tuple[EventWindow, ...]


@dataclass(frozen=True)
class CMaxFlowExtraction:
    """Flow records and sequence-level learned/control CMax comparisons."""

    clips: tuple[CMaxFlowClipRecord, ...]
    steps: tuple[CMaxFlowStepRecord, ...]
    flow_shuffle_permutation: tuple[int, ...]
    conditions: dict[str, dict[str, float]]
    focus_improvements: dict[str, float]


def validate_cmax_flow_config(
    config: ExperimentConfig,
    model: WindowJEPA | None = None,
) -> None:
    """Strictly require the trained post-encoder recurrent CMax contract."""

    validate_future_feature_config(config)
    if not config.cmax.enabled or config.cmax.weight <= 0:
        raise ValueError("flow visualization requires cmax.enabled=true")
    if config.recurrent.recurrent_placement != "post_encoder":
        raise ValueError("CMax flow visualization requires post_encoder recurrence")
    if config.recurrent.tbptt_steps != config.recurrent.sequence_length:
        raise ValueError("CMax visualization requires one complete supervised sequence")
    if model is None:
        return
    if not isinstance(model, WindowJEPA):
        raise TypeError("model must be a WindowJEPA")
    flow_head = model.cmax_flow_head
    criterion = model.cmax_criterion
    if flow_head is None or criterion is None:
        raise ValueError("model is missing the trained CMax flow head or criterion")
    if not math.isclose(
        model.cmax_weight,
        config.cmax.weight,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("model CMax weight differs from checkpoint configuration")
    if not math.isclose(
        flow_head.max_displacement,
        config.cmax.max_displacement,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("flow-head displacement bound differs from checkpoint config")
    if (
        flow_head.embed_dim != config.model.embed_dim
        or flow_head.hidden_dim != config.cmax.hidden_dim
        or flow_head.head_depth != config.cmax.head_depth
        or not math.isclose(
            flow_head.flow_scale,
            config.cmax.flow_scale,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("CMax flow-head structure differs from checkpoint config")
    if (
        criterion.reference_mode != config.cmax.reference_mode
        or criterion.temporal_scales != config.cmax.temporal_scales
        or criterion.min_events != config.cmax.min_events
        or not math.isclose(
            criterion.smoothness_weight,
            config.cmax.smoothness_weight,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("CMax criterion differs from checkpoint configuration")


def _validate_image_size(image_size: tuple[int, int]) -> tuple[int, int]:
    if (
        not isinstance(image_size, tuple)
        or len(image_size) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in image_size)
    ):
        raise TypeError("image_size must be an integer (height,width) pair")
    if min(image_size) <= 0:
        raise ValueError("image_size entries must be positive")
    return image_size


def _validate_flow_map(flow: torch.Tensor, name: str = "flow") -> None:
    if not isinstance(flow, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if flow.ndim != 3 or flow.shape[0] != 2 or min(flow.shape) <= 0:
        raise ValueError(f"{name} must have shape [2,Hg,Wg]")
    if not flow.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if not bool(torch.isfinite(flow).all()):
        raise FloatingPointError(f"{name} contains NaN or infinity")


def _validate_max_displacement(max_displacement: float) -> float:
    value = float(max_displacement)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("max_displacement must be finite and positive")
    return value


def _dense_flow(
    flow: torch.Tensor,
    image_size: tuple[int, int],
) -> np.ndarray:
    _validate_flow_map(flow)
    height, width = _validate_image_size(image_size)
    grid_height, grid_width = flow.shape[-2:]
    if height % grid_height or width % grid_width:
        raise ValueError("image_size must be divisible by the flow patch grid")
    dense = functional.interpolate(
        flow.detach().float().unsqueeze(0),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )[0]
    if not bool(torch.isfinite(dense).all()):
        raise FloatingPointError("dense flow interpolation produced non-finite values")
    return np.ascontiguousarray(dense.cpu().numpy(), dtype=np.float32)


def _hsv_to_rgb(hue: np.ndarray, saturation: np.ndarray, value: np.ndarray) -> np.ndarray:
    if hue.shape != saturation.shape or hue.shape != value.shape:
        raise ValueError("HSV channels must share a shape")
    h6 = np.mod(hue, 1.0) * 6.0
    sector = np.floor(h6).astype(np.int64) % 6
    fraction = h6 - np.floor(h6)
    p = value * (1.0 - saturation)
    q = value * (1.0 - saturation * fraction)
    t = value * (1.0 - saturation * (1.0 - fraction))
    rgb = np.empty((*hue.shape, 3), dtype=np.float32)
    choices = (
        (value, t, p),
        (q, value, p),
        (p, value, t),
        (p, q, value),
        (t, p, value),
        (value, p, q),
    )
    for index, channels in enumerate(choices):
        selected = sector == index
        for channel, values in enumerate(channels):
            rgb[..., channel][selected] = values[selected]
    return rgb


def flow_to_hsv_rgb(
    flow: torch.Tensor,
    *,
    max_displacement: float,
    image_size: tuple[int, int],
) -> np.ndarray:
    """Render fixed-scale flow: right=red, up=purple, left=cyan, down=yellow.

    Flow channels are ``(dx,dy)`` in pixels per base event window. Image
    coordinates use positive x to the right and positive y downward. The
    checkpoint bound applies independently to dx and dy. Hue is direction;
    HSV value clips vector magnitude at ``max_displacement`` and zero is black.
    """

    maximum = _validate_max_displacement(max_displacement)
    dense = _dense_flow(flow, image_size)
    dx, dy = dense
    magnitude = np.sqrt(dx * dx + dy * dy)
    hue = np.mod(np.arctan2(-dy, -dx) / (2.0 * np.pi) + 0.5, 1.0)
    value = np.clip(magnitude / maximum, 0.0, 1.0)
    rgb = _hsv_to_rgb(hue, np.ones_like(value), value)
    return np.clip(np.rint(255.0 * rgb), 0, 255).astype(np.uint8)


def flow_magnitude_rgb(
    flow: torch.Tensor,
    *,
    max_displacement: float,
    image_size: tuple[int, int],
) -> np.ndarray:
    """Render magnitude clipped to one checkpoint-fixed ``[0,max]`` scale."""

    maximum = _validate_max_displacement(max_displacement)
    dense = _dense_flow(flow, image_size)
    magnitude = np.sqrt(dense[0] * dense[0] + dense[1] * dense[1])
    normalized = np.clip(magnitude / maximum, 0.0, 1.0)
    anchors = np.asarray(
        (
            (5.0, 12.0, 35.0),
            (18.0, 111.0, 180.0),
            (20.0, 190.0, 160.0),
            (248.0, 205.0, 42.0),
            (218.0, 42.0, 45.0),
        ),
        dtype=np.float32,
    )
    positions = np.linspace(0.0, 1.0, len(anchors), dtype=np.float32)
    rgb = np.stack(
        [np.interp(normalized, positions, anchors[:, index]) for index in range(3)],
        axis=-1,
    )
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def flow_statistics(
    flow: torch.Tensor,
    *,
    max_displacement: float,
) -> dict[str, float]:
    """Return finite fixed-bound displacement statistics for any flow tensor."""

    maximum = _validate_max_displacement(max_displacement)
    if not isinstance(flow, torch.Tensor):
        raise TypeError("flow must be a torch.Tensor")
    if flow.ndim < 3 or flow.shape[-3] != 2 or flow.numel() == 0:
        raise ValueError("flow must have a non-empty [...,2,Hg,Wg] shape")
    if not flow.is_floating_point():
        raise TypeError("flow must be floating point")
    values = flow.detach().float()
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError("flow contains NaN or infinity")
    magnitude = values.square().sum(dim=-3).sqrt().reshape(-1)
    saturation = values.abs().amax(dim=-3) >= 0.95 * maximum
    return {
        "mean_dx": float(values.select(-3, 0).mean()),
        "mean_dy": float(values.select(-3, 1).mean()),
        "mean_magnitude": float(magnitude.mean()),
        "p95_magnitude": float(torch.quantile(magnitude, 0.95)),
        "max_magnitude": float(magnitude.max()),
        "saturation_fraction": float(saturation.float().mean()),
    }


def _draw_line(
    image: np.ndarray,
    start: tuple[float, float],
    end: tuple[float, float],
    color: np.ndarray,
    *,
    radius: int = 1,
    outline_color: np.ndarray | None = None,
    outline_radius: int = 2,
) -> None:
    x0, y0 = start
    x1, y1 = end
    count = max(int(math.ceil(max(abs(x1 - x0), abs(y1 - y0)))), 1) + 1
    x = np.rint(np.linspace(x0, x1, count)).astype(np.int64)
    y = np.rint(np.linspace(y0, y1, count)).astype(np.int64)
    def paint(paint_color: np.ndarray, paint_radius: int) -> None:
        for offset_y in range(-paint_radius, paint_radius + 1):
            for offset_x in range(-paint_radius, paint_radius + 1):
                xx = x + offset_x
                yy = y + offset_y
                valid = (
                    (xx >= 0)
                    & (xx < image.shape[1])
                    & (yy >= 0)
                    & (yy < image.shape[0])
                )
                image[yy[valid], xx[valid]] = paint_color

    if outline_color is not None:
        if outline_radius < radius:
            raise ValueError("outline_radius cannot be smaller than radius")
        paint(outline_color, outline_radius)
    paint(color, radius)


def quiver_overlay_rgb(
    base_image: np.ndarray,
    flow: torch.Tensor,
    *,
    stride: int = 1,
) -> np.ndarray:
    """Overlay patch-centre arrows using flow's native pixel displacement."""

    if (
        not isinstance(base_image, np.ndarray)
        or base_image.dtype != np.uint8
        or base_image.ndim != 3
        or base_image.shape[2] != 3
    ):
        raise TypeError("base_image must be an RGB uint8 NumPy array")
    if isinstance(stride, bool) or not isinstance(stride, int):
        raise TypeError("stride must be an integer")
    if stride <= 0:
        raise ValueError("stride must be positive")
    _validate_flow_map(flow)
    height, width = base_image.shape[:2]
    grid_height, grid_width = flow.shape[-2:]
    if height % grid_height or width % grid_width:
        raise ValueError("base image must be divisible by the flow patch grid")
    values = flow.detach().float().cpu().numpy()
    output = np.ascontiguousarray(base_image.copy())
    for row in range(0, grid_height, stride):
        for column in range(0, grid_width, stride):
            dx = float(values[0, row, column])
            dy = float(values[1, row, column])
            magnitude = math.hypot(dx, dy)
            if magnitude < 1e-6:
                continue
            start = (
                (column + 0.5) * width / grid_width,
                (row + 0.5) * height / grid_height,
            )
            end = (start[0] + dx, start[1] + dy)
            direction_hue = (
                math.atan2(-dy, -dx) / (2.0 * math.pi) + 0.5
            ) % 1.0
            color = np.rint(
                255.0
                * _hsv_to_rgb(
                    np.asarray([direction_hue], dtype=np.float32),
                    np.asarray([0.86], dtype=np.float32),
                    np.asarray([1.0], dtype=np.float32),
                )[0]
            ).astype(np.uint8)
            outline = np.asarray((0, 0, 0), dtype=np.uint8)
            _draw_line(
                output,
                start,
                end,
                color,
                radius=1,
                outline_color=outline,
                outline_radius=2,
            )
            angle = math.atan2(dy, dx)
            head_length = min(6.0, max(2.5, 0.25 * magnitude))
            for offset in (2.55, -2.55):
                tip = (
                    end[0] + head_length * math.cos(angle + offset),
                    end[1] + head_length * math.sin(angle + offset),
                )
                _draw_line(
                    output,
                    end,
                    tip,
                    color,
                    radius=1,
                    outline_color=outline,
                    outline_radius=2,
                )
    return output


def _occupied_fraction(iwe: torch.Tensor) -> torch.Tensor:
    if iwe.ndim != 4 or iwe.shape[0] != 1 or iwe.shape[1] != 2:
        raise ValueError("IWE must have shape [1,2,H,W]")
    return (iwe.sum(dim=1) > 0).float().mean()


def _retained_fraction(xy: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("warped coordinates must have shape [E,2]")
    if not xy.shape[0]:
        return xy.new_zeros(())
    height, width = image_size
    retained = (
        (xy[:, 0] >= 0)
        & (xy[:, 0] <= width - 1)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] <= height - 1)
    )
    return retained.float().mean()


@torch.no_grad()
def compute_step_warp_iwes(
    flow_map: torch.Tensor,
    events: PackedEventBatch,
) -> CMaxStepWarpResult:
    """Warp one packed base window to its past/future endpoints and splat IWEs."""

    _validate_flow_map(flow_map)
    if not isinstance(events, PackedEventBatch):
        raise TypeError("events must be a PackedEventBatch")
    if events.batch_size != 1 or events.time_steps != 1:
        raise ValueError("step visualization requires PackedEventBatch B=1,T=1")
    if events.device != flow_map.device:
        raise ValueError("flow_map and events must share a device")
    grid_height, grid_width = flow_map.shape[-2:]
    if events.height % grid_height or events.width % grid_width:
        raise ValueError("event image size must be divisible by the flow grid")
    flow_sequence = flow_map.float().reshape(1, 1, 2, grid_height, grid_width)
    event_flow = sample_patch_flow_at_events(
        flow_sequence,
        events.x,
        events.y,
        events.batch_index,
        events.time_index,
        (events.height, events.width),
    )
    unwarped_xy = torch.stack((events.x.float(), events.y.float()), dim=1)
    past_xy = warp_events_to_reference(
        events.x,
        events.y,
        events.t,
        event_flow,
        0.0,
    )
    future_xy = warp_events_to_reference(
        events.x,
        events.y,
        events.t,
        event_flow,
        1.0,
    )

    def splat(xy: torch.Tensor, values: torch.Tensor | None = None) -> torch.Tensor:
        return bilinear_splat_iwe(
            xy,
            events.polarity,
            events.batch_index,
            (events.height, events.width),
            batch_size=1,
            values=values,
        )

    unwarped_iwe = splat(unwarped_xy)
    past_iwe = splat(past_xy)
    future_iwe = splat(future_xy)
    past_average = average_timestamp_image(past_iwe, splat(past_xy, 1.0 - events.t))
    future_average = average_timestamp_image(future_iwe, splat(future_xy, events.t))
    tensors = (unwarped_iwe, past_iwe, future_iwe, past_average, future_average)
    if not all(bool(torch.isfinite(value).all()) for value in tensors):
        raise FloatingPointError("event warping or IWE splatting produced non-finite values")
    return CMaxStepWarpResult(
        unwarped_iwe=unwarped_iwe,
        past_iwe=past_iwe,
        future_iwe=future_iwe,
        past_focus_loss=taming_focus_loss(past_iwe, past_average)[0],
        future_focus_loss=taming_focus_loss(future_iwe, future_average)[0],
        unwarped_occupied_pixel_fraction=_occupied_fraction(unwarped_iwe),
        past_occupied_pixel_fraction=_occupied_fraction(past_iwe),
        future_occupied_pixel_fraction=_occupied_fraction(future_iwe),
        past_retained_event_fraction=_retained_fraction(
            past_xy, (events.height, events.width)
        ),
        future_retained_event_fraction=_retained_fraction(
            future_xy, (events.height, events.width)
        ),
    )


def _condition_record(
    output: CMaxOutput,
    flow: torch.Tensor,
    *,
    max_displacement: float,
) -> dict[str, float]:
    values = {
        "loss": float(output.loss),
        "focus_loss": float(output.focus_loss),
        "forward_focus_loss": float(output.forward_focus_loss),
        "backward_focus_loss": float(output.backward_focus_loss),
        "smoothness_loss": float(output.smoothness_loss),
        "valid_event_count": float(output.valid_event_count),
        "valid_partition_count": float(output.valid_partition_count),
        "valid_window_fraction": float(output.valid_window_fraction),
        "mean_flow_magnitude": float(output.mean_flow_magnitude),
        "occupied_pixel_fraction": float(output.occupied_pixel_fraction),
        "saturation_fraction": flow_statistics(
            flow, max_displacement=max_displacement
        )["saturation_fraction"],
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise FloatingPointError("CMax condition metrics contain NaN or infinity")
    return values


@torch.no_grad()
def compare_cmax_flow_conditions(
    flow_maps: torch.Tensor,
    packed_events: PackedEventBatch,
    criterion: TamingCMaxLoss,
    *,
    shuffled_flow_maps: torch.Tensor,
    max_displacement: float,
) -> dict[str, dict[str, float]]:
    """Evaluate learned, zero, and sample-shuffled flow with one criterion."""

    maximum = _validate_max_displacement(max_displacement)
    if not isinstance(criterion, TamingCMaxLoss):
        raise TypeError("criterion must be TamingCMaxLoss")
    if not isinstance(flow_maps, torch.Tensor) or flow_maps.ndim != 5:
        raise ValueError("flow_maps must have shape [B,T,2,Hg,Wg]")
    if flow_maps.shape[2] != 2 or min(flow_maps.shape) <= 0:
        raise ValueError("flow_maps must have shape [B,T,2,Hg,Wg]")
    if flow_maps.shape != shuffled_flow_maps.shape:
        raise ValueError("shuffled_flow_maps must match flow_maps")
    if not flow_maps.is_floating_point() or not shuffled_flow_maps.is_floating_point():
        raise TypeError("flow condition tensors must be floating point")
    if flow_maps.device != shuffled_flow_maps.device:
        raise ValueError("learned and shuffled flow must share a device")
    if packed_events.device != flow_maps.device:
        raise ValueError("flow maps and packed events must share a device")
    if packed_events.batch_size != flow_maps.shape[0] or (
        packed_events.time_steps != flow_maps.shape[1]
    ):
        raise ValueError("packed events do not match flow batch/time dimensions")
    if not bool(
        torch.isfinite(flow_maps).all() & torch.isfinite(shuffled_flow_maps).all()
    ):
        raise FloatingPointError("flow conditions contain NaN or infinity")
    conditions = {
        "learned": flow_maps,
        "zero": torch.zeros_like(flow_maps),
        "sample_shuffled": shuffled_flow_maps,
    }
    return {
        name: _condition_record(
            criterion(condition_flow, packed_events),
            condition_flow,
            max_displacement=maximum,
        )
        for name, condition_flow in conditions.items()
    }


def _focus_improvements(
    conditions: dict[str, dict[str, float]],
) -> dict[str, float]:
    learned = conditions["learned"]["focus_loss"]
    improvements = {
        "zero_minus_learned": conditions["zero"]["focus_loss"] - learned,
        "sample_shuffled_minus_learned": (
            conditions["sample_shuffled"]["focus_loss"] - learned
        ),
    }
    if not all(math.isfinite(value) for value in improvements.values()):
        raise FloatingPointError("CMax focus improvements contain NaN or infinity")
    return improvements


def _supervised_steps(sample: dict[str, Any]) -> tuple[int, ...]:
    loss_mask = sample.get("loss_mask")
    if not isinstance(loss_mask, torch.Tensor) or loss_mask.ndim != 1:
        raise ValueError("flow sample requires one-dimensional loss_mask")
    if loss_mask.dtype != torch.bool:
        raise TypeError("flow sample loss_mask must be boolean")
    supervised = tuple(int(value) for value in loss_mask.nonzero().flatten())
    if not supervised:
        raise ValueError("flow sample has no supervised timesteps")
    if supervised != tuple(range(supervised[0], len(loss_mask))):
        raise ValueError("flow sample supervision must be one temporal suffix")
    return supervised


def _cmax_windows(sample: dict[str, Any], online_steps: int) -> tuple[EventWindow, ...]:
    windows = sample.get(RAW_EVENT_WINDOWS_KEY)
    if not isinstance(windows, (tuple, list)) or len(windows) != online_steps:
        raise ValueError("CMax sample is missing aligned raw-event windows")
    if any(not isinstance(window, EventWindow) for window in windows):
        raise TypeError("CMax raw-event payload must contain EventWindow objects")
    return tuple(windows)


def _validate_cmax_event_subsequence(
    raw_window: EventWindow,
    cmax_window: EventWindow,
    *,
    max_events_per_window: int | None,
) -> None:
    """Verify that capped events preserve an exact ordered raw-event subset."""

    if (
        raw_window.t_start_us != cmax_window.t_start_us
        or raw_window.t_end_us != cmax_window.t_end_us
        or (raw_window.height, raw_window.width)
        != (cmax_window.height, cmax_window.width)
    ):
        raise ValueError("raw and CMax event windows are not temporally aligned")
    if cmax_window.event_count > raw_window.event_count:
        raise ValueError("CMax event count cannot exceed its raw event window")
    expected_count = (
        raw_window.event_count
        if max_events_per_window is None
        else min(raw_window.event_count, max_events_per_window)
    )
    if cmax_window.event_count != expected_count:
        raise ValueError("CMax event count does not match the configured deterministic cap")
    if not cmax_window.event_count:
        return
    selected_index = 0
    selected_count = cmax_window.event_count
    for raw_index in range(raw_window.event_count):
        if (
            int(raw_window.x[raw_index]) == int(cmax_window.x[selected_index])
            and int(raw_window.y[raw_index]) == int(cmax_window.y[selected_index])
            and int(raw_window.t_us[raw_index])
            == int(cmax_window.t_us[selected_index])
            and int(raw_window.polarity[raw_index])
            == int(cmax_window.polarity[selected_index])
        ):
            selected_index += 1
            if selected_index == selected_count:
                return
    raise ValueError("CMax events are not an ordered subsequence of raw transformed events")


def _model_device(model: WindowJEPA, requested: torch.device) -> torch.device:
    devices = {parameter.device for parameter in model.parameters()}
    if len(devices) != 1:
        raise ValueError("model parameters must reside on one device")
    actual = next(iter(devices))
    if actual.type != requested.type or (
        requested.index is not None and actual.index != requested.index
    ):
        raise ValueError("requested evaluation device differs from model device")
    return actual


@torch.inference_mode()
def extract_cmax_flow_records(
    model: WindowJEPA,
    materialization: FutureFeatureMaterialization,
    *,
    device: str | torch.device,
    flow_shuffle_seed: int = 0,
) -> CMaxFlowExtraction:
    """Replay checkpoint-matched clips and extract flow/IWE/control records."""

    if model.training:
        raise RuntimeError("flow extraction requires model.eval()")
    if flow_shuffle_seed < 0:
        raise ValueError("flow_shuffle_seed cannot be negative")
    validate_cmax_flow_config(materialization.config, model)
    requested = torch.device(device)
    model_device = _model_device(model, requested)
    flow_head = model.cmax_flow_head
    criterion = model.cmax_criterion
    if flow_head is None or criterion is None:
        raise RuntimeError("validated model lost its CMax modules")
    grid_size = getattr(model.online_encoder, "grid_size", None)
    if not isinstance(grid_size, tuple) or len(grid_size) != 2:
        raise ValueError("online encoder is missing a two-dimensional patch grid")

    clips: list[CMaxFlowClipRecord] = []
    steps: list[CMaxFlowStepRecord] = []
    gpu_flow_sequences: list[torch.Tensor] = []
    packed_window_rows: list[tuple[EventWindow, ...]] = []
    for clip_position, clip in enumerate(materialization.clips):
        sample = clip.sample
        x = sample["x"].to(device=model_device, dtype=torch.float32)
        x_future = sample["x_future"].to(device=model_device, dtype=torch.float32)
        duration = sample["dt_ms"].to(device=model_device, dtype=torch.float32)
        future_duration = sample["future_dt_ms"].to(
            device=model_device, dtype=torch.float32
        )
        if x.ndim != 4 or x.shape != x_future.shape:
            raise ValueError("flow extraction requires aligned [T,C,H,W] inputs")
        if duration.shape != (x.shape[0],) or future_duration.shape != duration.shape:
            raise ValueError("flow extraction requires one duration per online step")
        supervised = _supervised_steps(sample)
        cmax_windows = _cmax_windows(sample, x.shape[0])
        first_supervised = supervised[0]
        state = None
        if first_supervised:
            state = model.recurrent_burn_in(
                x[:first_supervised].unsqueeze(0),
                duration[:first_supervised].unsqueeze(0),
                None,
                online_state=None,
            )
        clip_flows: list[torch.Tensor] = []
        selected_windows = tuple(cmax_windows[index] for index in supervised)
        packed_window_rows.append(selected_windows)
        for sequence_step, online_step in enumerate(supervised):
            latent = model.extract_recurrent_future_step(
                x_context=x[online_step].unsqueeze(0),
                x_future=x_future[online_step].unsqueeze(0),
                context_duration_ms=duration[online_step].reshape(1),
                target_duration_ms=future_duration[online_step].reshape(1),
                online_state=state,
            )
            state = latent.online_state
            flow = flow_head(latent.recurrent_tokens, grid_size)
            if flow.shape != (1, 2, *grid_size):
                raise RuntimeError("CMax flow head returned an unexpected shape")
            if not bool(torch.isfinite(flow).all()):
                raise FloatingPointError("CMax flow head returned NaN or infinity")
            flow_map = flow[0]
            clip_flows.append(flow_map)
            raw_window = clip.debug.windows[online_step]
            cmax_window = selected_windows[sequence_step]
            _validate_cmax_event_subsequence(
                raw_window,
                cmax_window,
                max_events_per_window=(
                    materialization.config.cmax.max_events_per_window
                ),
            )
            step_events = pack_event_windows(((selected_windows[sequence_step],),)).to(
                model_device
            )
            warp = compute_step_warp_iwes(flow_map, step_events)
            steps.append(
                CMaxFlowStepRecord(
                    record_index=len(steps),
                    clip_position=clip_position,
                    sample_index=clip.sample_index,
                    online_step=online_step,
                    sequence_id=str(sample["sequence_id"]),
                    t_start_us=raw_window.t_start_us,
                    t_end_us=raw_window.t_end_us,
                    raw_event_count=raw_window.event_count,
                    cmax_event_count=cmax_window.event_count,
                    cmax_window_valid=(
                        cmax_window.event_count >= materialization.config.cmax.min_events
                    ),
                    flow_map=flow_map.detach().float().cpu(),
                    unwarped_iwe=warp.unwarped_iwe[0].detach().float().cpu(),
                    past_iwe=warp.past_iwe[0].detach().float().cpu(),
                    future_iwe=warp.future_iwe[0].detach().float().cpu(),
                    past_focus_loss=float(warp.past_focus_loss),
                    future_focus_loss=float(warp.future_focus_loss),
                    unwarped_occupied_pixel_fraction=float(
                        warp.unwarped_occupied_pixel_fraction
                    ),
                    past_occupied_pixel_fraction=float(
                        warp.past_occupied_pixel_fraction
                    ),
                    future_occupied_pixel_fraction=float(
                        warp.future_occupied_pixel_fraction
                    ),
                    past_retained_event_fraction=float(
                        warp.past_retained_event_fraction
                    ),
                    future_retained_event_fraction=float(
                        warp.future_retained_event_fraction
                    ),
                )
            )
        flow_sequence = torch.stack(clip_flows, dim=0)
        gpu_flow_sequences.append(flow_sequence)
        clips.append(
            CMaxFlowClipRecord(
                clip_position=clip_position,
                sample_index=clip.sample_index,
                sequence_id=str(sample["sequence_id"]),
                supervised_steps=supervised,
                flow_maps=flow_sequence.detach().float().cpu(),
                cmax_windows=selected_windows,
            )
        )

    if len(clips) < 2:
        raise ValueError("sample-shuffled flow control requires at least two clips")
    first_steps = clips[0].supervised_steps
    if any(clip.supervised_steps != first_steps for clip in clips[1:]):
        raise ValueError("calibration clips must share supervised timestep indices")
    flow_maps = torch.stack(gpu_flow_sequences, dim=0)
    packed_events = pack_event_windows(packed_window_rows).to(model_device)
    clip_identities = tuple(
        (
            clip.sequence_id,
            tuple(
                int(materialization.clips[clip.clip_position].sample["future_t_end_us"][step])
                for step in clip.supervised_steps
            ),
        )
        for clip in clips
    )
    permutation = make_history_replacement_clip_permutation(
        clip_identities,
        flow_shuffle_seed,
    )
    if any(
        clip_identities[source] == clip_identities[target]
        for source, target in enumerate(permutation)
    ):
        raise RuntimeError("sample-shuffled flow control reused an identical future anchor")
    permutation_tensor = torch.tensor(
        permutation, dtype=torch.int64, device=model_device
    )
    shuffled = flow_maps.index_select(0, permutation_tensor)
    conditions = compare_cmax_flow_conditions(
        flow_maps,
        packed_events,
        criterion,
        shuffled_flow_maps=shuffled,
        max_displacement=flow_head.max_displacement,
    )
    return CMaxFlowExtraction(
        clips=tuple(clips),
        steps=tuple(steps),
        flow_shuffle_permutation=permutation,
        conditions=conditions,
        focus_improvements=_focus_improvements(conditions),
    )


def _iwe_counts(iwe: torch.Tensor) -> np.ndarray:
    if iwe.shape[0] != 2 or iwe.ndim != 3:
        raise ValueError("IWE visualization expects [2,H,W]")
    if not bool(torch.isfinite(iwe).all()) or bool((iwe < 0).any()):
        raise ValueError("IWE visualization requires finite non-negative counts")
    values = iwe.detach().float().cpu().numpy()
    # CMax uses channel 0=positive and 1=negative. Inspection RGB expects
    # channel 0=OFF/negative and 1=ON/positive.
    return np.ascontiguousarray(np.stack((values[1], values[0])), dtype=np.float32)


def _format(value: float) -> str:
    return f"{value:.6f}"


def _condition_rows(conditions: dict[str, dict[str, float]]) -> str:
    labels = {
        "learned": "learned flow",
        "zero": "zero flow",
        "sample_shuffled": "sample-shuffled flow",
    }
    learned_focus = conditions["learned"]["focus_loss"]
    rows: list[str] = []
    for name in ("learned", "zero", "sample_shuffled"):
        values = conditions[name]
        difference = values["focus_loss"] - learned_focus
        rows.append(
            "<tr>"
            f"<td>{html.escape(labels[name])}</td>"
            f"<td>{_format(values['loss'])}</td>"
            f"<td>{_format(values['focus_loss'])}</td>"
            f"<td>{_format(difference) if name != 'learned' else '—'}</td>"
            f"<td>{_format(values['forward_focus_loss'])}</td>"
            f"<td>{_format(values['backward_focus_loss'])}</td>"
            f"<td>{_format(values['smoothness_loss'])}</td>"
            f"<td>{_format(values['occupied_pixel_fraction'])}</td>"
            f"<td>{_format(values['mean_flow_magnitude'])}</td>"
            f"<td>{_format(values['saturation_fraction'])}</td>"
            "</tr>"
        )
    return "".join(rows)


def _step_rows(
    steps: Sequence[CMaxFlowStepRecord],
    sample_index: int,
    *,
    max_displacement: float,
) -> str:
    rows: list[str] = []
    for record in steps:
        if record.sample_index != sample_index:
            continue
        stats = flow_statistics(
            record.flow_map,
            max_displacement=max_displacement,
        )
        rows.append(
            "<tr>"
            f"<td>{record.online_step}</td>"
            f"<td>{record.t_end_us:,}</td>"
            f"<td>{record.raw_event_count:,}</td>"
            f"<td>{record.cmax_event_count:,}</td>"
            f"<td>{'yes' if record.cmax_window_valid else 'no'}</td>"
            f"<td>{stats['mean_magnitude']:.3f}</td>"
            f"<td>{stats['p95_magnitude']:.3f}</td>"
            f"<td>{record.past_focus_loss:.6f}</td>"
            f"<td>{record.future_focus_loss:.6f}</td>"
            "</tr>"
        )
    return "".join(rows)


def _panel(key: str, label: str, caption: str, src: str) -> str:
    return (
        "<figure>"
        f'<img src="{html.escape(src, quote=True)}" alt="{html.escape(label)}">'
        f"<figcaption><b>{html.escape(label)}</b>"
        f"<span>{html.escape(caption)}</span></figcaption>"
        "</figure>"
    )


def _visual_indices(
    steps: Sequence[CMaxFlowStepRecord],
    sample_index: int,
    all_steps: bool,
) -> tuple[int, ...]:
    selected = [record.record_index for record in steps if record.sample_index == sample_index]
    if not selected:
        raise ValueError("display sample is absent from extracted flow records")
    return tuple(selected if all_steps else selected[-1:])


def _render_steps(
    extraction: CMaxFlowExtraction,
    materialization: FutureFeatureMaterialization,
    output: Path,
    *,
    display_sample_index: int,
    all_steps: bool,
    max_displacement: float,
    quiver_stride: int,
) -> tuple[str, list[dict[str, Any]]]:
    selected = _visual_indices(extraction.steps, display_sample_index, all_steps)
    assets = output.parent / f"{output.stem}_assets"
    assets.mkdir(parents=True, exist_ok=True)
    raw_counts = [
        _event_counts(clip.debug.windows[record.online_step])
        for record in extraction.steps
        for clip in (materialization.clips[record.clip_position],)
    ]
    cmax_counts = [
        _event_counts(
            extraction.clips[record.clip_position].cmax_windows[
                extraction.clips[record.clip_position].supervised_steps.index(
                    record.online_step
                )
            ]
        )
        for record in extraction.steps
    ]
    event_scale = _display_scale(
        [np.log1p(value) for value in (*raw_counts, *cmax_counts)]
    )
    iwe_counts = [
        _iwe_counts(value)
        for record in extraction.steps
        for value in (record.unwarped_iwe, record.past_iwe, record.future_iwe)
    ]
    iwe_scale = _display_scale([np.log1p(value) for value in iwe_counts])
    sections: list[str] = []
    visual_records: list[dict[str, Any]] = []
    for record_index in selected:
        record = extraction.steps[record_index]
        clip = materialization.clips[record.clip_position]
        clip_record = extraction.clips[record.clip_position]
        sequence_step = clip_record.supervised_steps.index(record.online_step)
        raw_window = clip.debug.windows[record.online_step]
        cmax_window = clip_record.cmax_windows[sequence_step]
        raw_rgb = _event_rgb(_event_counts(raw_window), event_scale)
        prefix = f"sample-{record.sample_index:05d}-step-{record.online_step:02d}"
        panels: list[str] = []
        images: dict[str, str] = {}

        def save(key: str, label: str, caption: str, image: np.ndarray) -> None:
            path = assets / f"{prefix}-{key}.png"
            _write_png(path, image)
            images[key] = str(path.relative_to(output.parent))
            panels.append(_panel(key, label, caption, _asset_reference(output, path)))

        stats = flow_statistics(
            record.flow_map, max_displacement=max_displacement
        )
        save(
            "raw-events",
            "全raw events",
            f"events={record.raw_event_count:,}（ViT表現の入力元）",
            raw_rgb,
        )
        save(
            "cmax-events",
            "CMax subsample events",
            (
                f"events={record.cmax_event_count:,} · "
                f"criterion-valid={'yes' if record.cmax_window_valid else 'no'}"
            ),
            _event_rgb(_event_counts(cmax_window), event_scale),
        )
        save(
            "flow-hsv",
            "Flow HSV",
            (
                f"fixed 0–{max_displacement:g} px/window · "
                "x右/y下 · 右=赤, 上=紫, 左=cyan, 下=黄緑"
            ),
            flow_to_hsv_rgb(
                record.flow_map,
                max_displacement=max_displacement,
                image_size=materialization.config.model.image_size,
            ),
        )
        save(
            "flow-magnitude",
            "Flow magnitude",
            (
                f"mean={stats['mean_magnitude']:.3f}, "
                f"p95={stats['p95_magnitude']:.3f}, "
                f"max={stats['max_magnitude']:.3f} px/window"
            ),
            flow_magnitude_rgb(
                record.flow_map,
                max_displacement=max_displacement,
                image_size=materialization.config.model.image_size,
            ),
        )
        save(
            "flow-quiver",
            "Flow quiver on raw events",
            f"native displacement · grid stride={quiver_stride}",
            quiver_overlay_rgb(raw_rgb, record.flow_map, stride=quiver_stride),
        )
        save(
            "iwe-unwarped",
            "Warp前 IWE",
            f"occupied={record.unwarped_occupied_pixel_fraction:.4f}",
            _event_rgb(_iwe_counts(record.unwarped_iwe), iwe_scale),
        )
        save(
            "iwe-past",
            "Past endpoint warp IWE",
            (
                f"focus={record.past_focus_loss:.6f}, "
                f"occupied={record.past_occupied_pixel_fraction:.4f}, "
                f"event-center retained={record.past_retained_event_fraction:.4f}"
            ),
            _event_rgb(_iwe_counts(record.past_iwe), iwe_scale),
        )
        save(
            "iwe-future",
            "Future endpoint warp IWE",
            (
                f"focus={record.future_focus_loss:.6f}, "
                f"occupied={record.future_occupied_pixel_fraction:.4f}, "
                f"event-center retained={record.future_retained_event_fraction:.4f}"
            ),
            _event_rgb(_iwe_counts(record.future_iwe), iwe_scale),
        )
        sections.append(
            f"""
            <section class="visual-step">
              <h2>sample {record.sample_index} · online step {record.online_step}</h2>
              <p class="metadata"><code>{html.escape(record.sequence_id)}</code>
                <span>window=({record.t_start_us:,}, {record.t_end_us:,}] μs</span></p>
              <div class="panel-grid">{''.join(panels)}</div>
            </section>
            """
        )
        visual_records.append(
            {
                "record_index": record.record_index,
                "sample_index": record.sample_index,
                "online_step": record.online_step,
                "images": images,
            }
        )
    return "".join(sections), visual_records


def _report_html(
    *,
    checkpoint: Path,
    config: ExperimentConfig,
    extraction: CMaxFlowExtraction,
    display_sample_index: int,
    visual_sections: str,
) -> str:
    grid = (
        config.model.image_size[0] // config.model.patch_size,
        config.model.image_size[1] // config.model.patch_size,
    )
    return f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CMax flow visualization</title>
<style>
:root {{ color-scheme:light dark; --bg:#f4f6fa; --surface:#fff; --text:#172033;
 --muted:#667085; --line:#d8dee9; --accent:#2457d6; }}
@media (prefers-color-scheme:dark) {{ :root {{ --bg:#10141c; --surface:#1a202b;
 --text:#edf2fb; --muted:#a6b1c4; --line:#303a4a; --accent:#91a8ff; }} }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--text);
 font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
main {{ max-width:1500px; margin:auto; padding:28px; }} h1 {{ margin:0 0 8px; font-size:1.75rem; }}
h2 {{ margin:0 0 12px; font-size:1.2rem; }} p {{ line-height:1.65; }}
.muted {{ color:var(--muted); }} .metadata {{ color:var(--muted); display:flex; gap:16px;
 flex-wrap:wrap; }}
.note {{ border-left:4px solid var(--accent); padding:4px 14px; margin:20px 0; }}
.summary {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
 gap:12px; margin:20px 0; }} .stat {{ background:var(--surface); border:1px solid var(--line);
 border-radius:12px; padding:14px; }} .stat span {{ display:block; color:var(--muted);
 font-size:.84rem; }} .stat strong {{ display:block; margin-top:5px; font-size:1.25rem; }}
.table-wrap {{ overflow:auto; margin:14px 0 34px; }} table {{ width:100%; border-collapse:collapse;
 background:var(--surface); }} th,td {{ padding:10px 12px; border-bottom:1px solid var(--line);
 text-align:right; white-space:nowrap; font-variant-numeric:tabular-nums; }}
th:first-child,td:first-child {{ text-align:left; }} .visual-step {{ margin:34px 0 48px; }}
.panel-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(250px,1fr)); gap:16px; }}
figure {{ margin:0; background:var(--surface); border:1px solid var(--line); border-radius:12px;
 overflow:hidden; }} figure img {{ display:block; width:100%; height:auto;
 image-rendering:pixelated; background:#050810; }} figcaption {{ display:flex;
 flex-direction:column; gap:4px; padding:11px 12px 13px; }}
figcaption span {{ color:var(--muted); font-size:.84rem; }} code {{ word-break:break-all; }}
@media(max-width:600px) {{ main {{ padding:17px; }} .panel-grid {{ grid-template-columns:1fr; }} }}
</style></head><body><main>
<h1>CMax flow 可視化</h1>
<p class="muted">checkpoint: <code>{html.escape(str(checkpoint))}</code></p>
<div class="note"><b>座標と単位:</b> flowは <code>(dx,dy)</code>、
x正方向は右、y正方向は下。
単位は <b>pixels / {config.recurrent.window_ms:g} ms base window</b> です。
Flow headのpatch gridは <b>{grid[0]}×{grid[1]}</b>、上限は
dx・dy各成分それぞれ<b>±{config.cmax.max_displacement:g} px/window</b>。
HSVとmagnitude表示はvector magnitudeを
<b>{config.cmax.max_displacement:g} px/window</b>でclipします。
これはGT flow評価ではなくCMax自己整合性診断です。</div>
<div class="note"><b>state scope:</b> 各sampleはdirect clipの先頭でstateをresetし、
checkpoint設定のburn-in {config.recurrent.burn_in_steps}窓だけを再生しています。
学習中のstream laneでclip以前から保持されていたstateは再現しません。</div>
<div class="note"><b>診断scope:</b> 各stepのpast/future IWEとfocusは、min-events判定や
training reference modeによらず両endpointを描く<b>1-window local診断</b>です。
一方、次のsequence比較だけが学習時と同じmulti-scale iterative warp、
reference mode、min-events、smoothnessを使います。
retainedはwarp後の<b>event center</b>が画像内に残る割合です。</div>
<div class="summary">
  <div class="stat"><span>zero focus − learned focus</span>
    <strong>{extraction.focus_improvements['zero_minus_learned']:+.6f}</strong></div>
  <div class="stat"><span>shuffled focus − learned focus</span>
    <strong>{extraction.focus_improvements['sample_shuffled_minus_learned']:+.6f}</strong></div>
</div>
<p class="muted">focusは<b>lower is better</b>です。
したがって上の差が正なら、learned flowが
それぞれのcontrolより低いfocus lossを達成しています。</p>
<h2>sequence全{config.recurrent.sequence_length}窓の同一criterion比較</h2>
<div class="table-wrap"><table><thead><tr><th>condition</th><th>total</th><th>focus</th>
<th>focus − learned</th><th>future focus</th><th>past focus</th><th>smoothness</th>
<th>occupied</th><th>mean |flow|</th><th>saturation</th>
</tr></thead><tbody>{_condition_rows(extraction.conditions)}</tbody></table></div>
<p class="muted">zeroとsample-shuffledも、学習時と同じreference mode・temporal scales・
min-events・smoothness設定で再計算しています。totalは
<code>focus + {config.cmax.smoothness_weight:g} × smoothness</code>という
外側のCMax weight {config.cmax.weight:g}を掛ける前のcriterion値です。
色やtotal lossだけで結論せず、focusとoccupiedも併記してください。</p>
<h2>sample {display_sample_index} の時間変化</h2>
<div class="table-wrap"><table><thead><tr><th>online step</th><th>t_end μs</th><th>raw events</th>
<th>CMax events</th><th>criterion-valid window</th><th>mean |flow|</th><th>p95 |flow|</th>
<th>past focus</th><th>future focus</th>
</tr></thead><tbody>{_step_rows(
    extraction.steps,
    display_sample_index,
    max_displacement=config.cmax.max_displacement,
)}</tbody></table></div>
{visual_sections}
</main></body></html>"""


def _step_json(
    record: CMaxFlowStepRecord,
    *,
    max_displacement: float,
) -> dict[str, Any]:
    return {
        "record_index": record.record_index,
        "clip_position": record.clip_position,
        "sample_index": record.sample_index,
        "online_step": record.online_step,
        "sequence_id": record.sequence_id,
        "t_start_us": record.t_start_us,
        "t_end_us": record.t_end_us,
        "raw_event_count": record.raw_event_count,
        "cmax_event_count": record.cmax_event_count,
        "cmax_window_valid": record.cmax_window_valid,
        "flow": flow_statistics(
            record.flow_map, max_displacement=max_displacement
        ),
        "past_focus_loss": record.past_focus_loss,
        "future_focus_loss": record.future_focus_loss,
        "unwarped_occupied_pixel_fraction": record.unwarped_occupied_pixel_fraction,
        "past_occupied_pixel_fraction": record.past_occupied_pixel_fraction,
        "future_occupied_pixel_fraction": record.future_occupied_pixel_fraction,
        "past_retained_event_fraction": record.past_retained_event_fraction,
        "future_retained_event_fraction": record.future_retained_event_fraction,
    }


def write_cmax_flow_report(
    model: WindowJEPA,
    checkpoint_config: ExperimentConfig,
    materialization: FutureFeatureMaterialization,
    checkpoint: str | Path,
    output: str | Path,
    *,
    device: str | torch.device,
    display_sample_index: int,
    all_steps: bool = False,
    flow_shuffle_seed: int = 0,
    quiver_stride: int = 1,
    checkpoint_identity: Mapping[str, Any] | None = None,
    manifest_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write an HTML+PNG+JSON flow/IWE report from a strict CMax checkpoint."""

    checkpoint_path = Path(checkpoint).expanduser().resolve(strict=True)
    output_path = Path(output).expanduser().resolve(strict=False)
    if output_path.suffix.lower() != ".html":
        raise ValueError("CMax flow report output must use an .html suffix")
    if display_sample_index < 0:
        raise ValueError("display_sample_index cannot be negative")
    if isinstance(quiver_stride, bool) or not isinstance(quiver_stride, int):
        raise TypeError("quiver_stride must be an integer")
    if quiver_stride <= 0:
        raise ValueError("quiver_stride must be positive")
    manifest_path = Path(
        materialization.config.data.manifest
    ).expanduser().resolve(strict=True)
    checkpoint_artifact = (
        _snapshot_file_identity(checkpoint_path, label="checkpoint")
        if checkpoint_identity is None
        else dict(checkpoint_identity)
    )
    manifest_artifact = (
        _snapshot_file_identity(manifest_path, label="manifest")
        if manifest_identity is None
        else dict(manifest_identity)
    )
    _require_unchanged_stat_identity(
        checkpoint_path, checkpoint_artifact, label="checkpoint"
    )
    _require_unchanged_stat_identity(
        manifest_path, manifest_artifact, label="manifest"
    )
    _validate_output_input_collisions(
        output_path,
        checkpoint=checkpoint_path,
        manifest=manifest_path,
    )
    validate_cmax_flow_config(checkpoint_config, model)
    _validate_report_compatibility(model, checkpoint_config, materialization)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    extraction = extract_cmax_flow_records(
        model,
        materialization,
        device=device,
        flow_shuffle_seed=flow_shuffle_seed,
    )
    visual_sections, visualized = _render_steps(
        extraction,
        materialization,
        output_path,
        display_sample_index=display_sample_index,
        all_steps=all_steps,
        max_displacement=checkpoint_config.cmax.max_displacement,
        quiver_stride=quiver_stride,
    )
    document = _report_html(
        checkpoint=checkpoint_path,
        config=checkpoint_config,
        extraction=extraction,
        display_sample_index=display_sample_index,
        visual_sections=visual_sections,
    )
    # Assets are now complete. Re-check both immutable inputs before binding
    # their recorded identities to the machine-readable report.
    _require_unchanged_stat_identity(
        checkpoint_path, checkpoint_artifact, label="checkpoint"
    )
    _require_unchanged_stat_identity(
        manifest_path, manifest_artifact, label="manifest"
    )
    payload: dict[str, Any] = {
        "schema": "event-window-jepa-cmax-flow-visualization-v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_artifact": checkpoint_artifact,
        "checkpoint_config_hash": config_hash(checkpoint_config),
        "effective_manifest": str(manifest_path),
        "manifest_artifact": manifest_artifact,
        "sampling_epoch": materialization.epoch,
        "sample_indices": [clip.sample_index for clip in materialization.clips],
        "flow_units": f"pixels/{checkpoint_config.recurrent.window_ms:g}ms_base_window",
        "coordinate_convention": {"x_positive": "right", "y_positive": "down"},
        "flow_grid": [
            checkpoint_config.model.image_size[0] // checkpoint_config.model.patch_size,
            checkpoint_config.model.image_size[1] // checkpoint_config.model.patch_size,
        ],
        "max_displacement": checkpoint_config.cmax.max_displacement,
        "max_displacement_semantics": "independent_absolute_bound_for_dx_and_dy",
        "magnitude_display_clip": checkpoint_config.cmax.max_displacement,
        "reference_mode": checkpoint_config.cmax.reference_mode,
        "temporal_scales": list(checkpoint_config.cmax.temporal_scales),
        "min_events": checkpoint_config.cmax.min_events,
        "max_events_per_window": checkpoint_config.cmax.max_events_per_window,
        "cmax_weight": checkpoint_config.cmax.weight,
        "smoothness_weight": checkpoint_config.cmax.smoothness_weight,
        "condition_total_semantics": (
            "unweighted_cmax_focus_plus_smoothness_weight_times_smoothness"
        ),
        "flow_shuffle_seed": flow_shuffle_seed,
        "flow_shuffle_permutation": list(extraction.flow_shuffle_permutation),
        "flow_shuffle_identity": "different_(sequence_id,supervised_future_t_end_us)",
        "conditions": extraction.conditions,
        "focus_improvements": extraction.focus_improvements,
        "metric_direction": {
            "focus_loss": "lower_is_better",
            "focus_improvements": "positive_means_learned_is_better",
        },
        "state_scope": {
            "mode": "direct_clip_reset_then_configured_burn_in",
            "burn_in_steps": checkpoint_config.recurrent.burn_in_steps,
            "reconstructs_prior_stream_lane_state": False,
        },
        "diagnostic_scope": {
            "step_iwe": (
                "single_base_window_local_both_endpoints_without_criterion_gating"
            ),
            "sequence_comparison": "training_identical_multiscale_iterative_criterion",
            "retained_fraction": (
                "fraction_of_warped_event_centers_inside_inclusive_image_bounds"
            ),
        },
        "clips": [
            {
                "clip_position": clip.clip_position,
                "sample_index": clip.sample_index,
                "sequence_id": clip.sequence_id,
                "supervised_steps": list(clip.supervised_steps),
                "shuffled_flow_donor_clip_position": (
                    extraction.flow_shuffle_permutation[clip.clip_position]
                ),
                "shuffled_flow_donor_sample_index": extraction.clips[
                    extraction.flow_shuffle_permutation[clip.clip_position]
                ].sample_index,
            }
            for clip in extraction.clips
        ],
        "steps": [
            _step_json(
                record,
                max_displacement=checkpoint_config.cmax.max_displacement,
            )
            for record in extraction.steps
        ],
        "visualized_steps": visualized,
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _write_text(output_path.with_suffix(".json"), serialized)
    _write_text(output_path, document)
    # The report must describe the files used for the entire render, not just
    # the identities observed before model/data loading.
    _require_unchanged_stat_identity(
        checkpoint_path, checkpoint_artifact, label="checkpoint"
    )
    _require_unchanged_stat_identity(
        manifest_path, manifest_artifact, label="manifest"
    )
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a trained CMax flow head, event warps, and controls"
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="override only the checkpoint dataset manifest path",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--calibration-samples", type=int, default=4)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--all-steps", action="store_true")
    parser.add_argument("--flow-shuffle-seed", type=int, default=0)
    parser.add_argument("--quiver-stride", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.sample_index < 0:
        raise ValueError("sample-index cannot be negative")
    if args.calibration_samples < 2:
        raise ValueError("calibration-samples must be at least two")
    if args.epoch < 0:
        raise ValueError("epoch cannot be negative")
    if args.flow_shuffle_seed < 0:
        raise ValueError("flow-shuffle-seed cannot be negative")
    if args.quiver_stride <= 0:
        raise ValueError("quiver-stride must be positive")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_artifact = _snapshot_file_identity(
        args.checkpoint, label="checkpoint"
    )
    checkpoint_path = Path(str(checkpoint_artifact["path"]))
    model, checkpoint_config = load_pretrained_model(checkpoint_path, device=device)
    _require_unchanged_content_identity(
        checkpoint_path, checkpoint_artifact, label="checkpoint"
    )
    validate_cmax_flow_config(checkpoint_config, model)
    manifest_path = Path(
        args.manifest
        if args.manifest is not None
        else checkpoint_config.data.manifest
    ).expanduser().resolve(strict=True)
    manifest_artifact = _snapshot_file_identity(manifest_path, label="manifest")
    output = args.output or checkpoint_path.with_name(
        f"{checkpoint_path.stem}-cmax-flow.html"
    )
    _validate_output_input_collisions(
        output,
        checkpoint=checkpoint_path,
        manifest=manifest_path,
    )
    materialization = materialize_future_feature_samples(
        checkpoint_config,
        range(args.sample_index, args.sample_index + args.calibration_samples),
        manifest_override=manifest_path,
        epoch=args.epoch,
    )
    _require_unchanged_stat_identity(
        checkpoint_path, checkpoint_artifact, label="checkpoint"
    )
    _require_unchanged_content_identity(
        manifest_path, manifest_artifact, label="manifest"
    )
    report = write_cmax_flow_report(
        model,
        checkpoint_config,
        materialization,
        checkpoint_path,
        output,
        device=device,
        display_sample_index=args.sample_index,
        all_steps=args.all_steps,
        flow_shuffle_seed=args.flow_shuffle_seed,
        quiver_stride=args.quiver_stride,
        checkpoint_identity=checkpoint_artifact,
        manifest_identity=manifest_artifact,
    )
    learned = report["conditions"]["learned"]
    zero = report["conditions"]["zero"]
    shuffled = report["conditions"]["sample_shuffled"]
    print(f"[window-jepa] CMax flow visualization: {Path(output).resolve()}")
    print(
        "[window-jepa] sequence CMax focus "
        f"learned={learned['focus_loss']:.6f}, "
        f"zero={zero['focus_loss']:.6f}, "
        f"sample-shuffled={shuffled['focus_loss']:.6f}"
    )


if __name__ == "__main__":
    main()


__all__ = [
    "CMaxFlowClipRecord",
    "CMaxFlowExtraction",
    "CMaxFlowStepRecord",
    "CMaxStepWarpResult",
    "compare_cmax_flow_conditions",
    "compute_step_warp_iwes",
    "extract_cmax_flow_records",
    "flow_magnitude_rgb",
    "flow_statistics",
    "flow_to_hsv_rgb",
    "quiver_overlay_rgb",
    "validate_cmax_flow_config",
    "write_cmax_flow_report",
]
