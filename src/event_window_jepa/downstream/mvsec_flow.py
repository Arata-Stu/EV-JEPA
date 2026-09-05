from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from event_window_jepa.data.event_store import H5EventStore
from event_window_jepa.downstream.mvsec_geometry import (
    EVFLOWNET_TEST_START_US,
    EVFLOWNET_TEST_STOP_US,
    MVSEC_F3_CONTEXT_WINDOW_US,
    MVSECGeometryDataset,
    MVSECGeometrySource,
    MVSECTargetReference,
    accumulate_metric_sums,
    build_mvsec_target_references,
    dense_patch_prediction,
    extract_frozen_mvsec_tokens,
    finalize_flow_metrics,
    flow_metric_sums,
    read_mvsec_geometry_sources,
    representation_from_config,
    split_mvsec_temporal_dev_references,
)
from event_window_jepa.models.cmax_flow import RecurrentTokenFlowHead
from event_window_jepa.models.recurrent_vjepa21_event_vit import (
    RecurrentVJEPA21EventVisionTransformer,
)
from event_window_jepa.train.checkpoint import (
    config_hash,
    load_pretrained_model,
)


FlowRate = Literal["native", "dt1_scaled_native_labels"]

NATIVE_FLOW_HZ = 20.0
DT1_FLOW_HZ = 45.0
NATIVE_FLOW_HORIZON_US = 50_000
DT1_FLOW_HORIZON_US = round(1_000_000 / DT1_FLOW_HZ)
DEFAULT_MINIMUM_VALID_PIXELS = 100
MVSEC_MODEL_IMAGE_SIZE = (272, 352)
CANONICAL_RANDOM_PROBE_HIDDEN_DIM = 256
CANONICAL_RANDOM_PROBE_HEAD_DEPTH = 2
CANONICAL_RANDOM_PROBE_FLOW_SCALE = 0.01
CANONICAL_RANDOM_PROBE_MAX_DISPLACEMENT = 32.0


@dataclass(frozen=True)
class FlowProtocol:
    """How stored 20 Hz MVSEC flow and head output are put on one horizon."""

    name: FlowRate
    requested_hz: float
    nominal_horizon_us: int
    ground_truth_scale: float

    def horizon_us(self, native_interval_us: int | float) -> float:
        if self.name == "native":
            interval = float(native_interval_us)
            if not math.isfinite(interval) or interval <= 0:
                raise ValueError("native MVSEC flow requires a positive label interval")
            return interval
        return 1_000_000.0 / self.requested_hz

    def prediction_scale(
        self,
        native_interval_us: int | float,
        *,
        base_window_us: int,
    ) -> float:
        if base_window_us <= 0:
            raise ValueError("base_window_us must be positive")
        return self.horizon_us(native_interval_us) / float(base_window_us)


def resolve_flow_protocol(value: str) -> FlowProtocol:
    """Resolve the intentionally small first-release MVSEC horizon surface."""

    if value == "native":
        return FlowProtocol(
            name="native",
            requested_hz=NATIVE_FLOW_HZ,
            nominal_horizon_us=NATIVE_FLOW_HORIZON_US,
            ground_truth_scale=1.0,
        )
    if value == "dt1":
        return FlowProtocol(
            name="dt1_scaled_native_labels",
            requested_hz=DT1_FLOW_HZ,
            nominal_horizon_us=DT1_FLOW_HORIZON_US,
            ground_truth_scale=NATIVE_FLOW_HZ / DT1_FLOW_HZ,
        )
    if value == "dt4":
        raise ValueError(
            "dt4 is intentionally unsupported: multi-interval flow must be composed "
            "on APS timestamps rather than obtained with a scalar shortcut"
        )
    raise ValueError("--dt must be native or dt1")


def reference_set_sha256(
    sources: Sequence[MVSECGeometrySource],
    references: Sequence[MVSECTargetReference],
) -> str:
    """Hash the ordered source, target-index, and internal-timestamp selection."""

    digest = hashlib.sha256()
    for reference in references:
        source = sources[reference.source_index]
        digest.update(
            (
                f"{source.sequence_id}\0{reference.target_index}\0"
                f"{reference.label_timestamp_us}\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _file_stat_identity(path: Path) -> dict[str, int]:
    stat_result = path.stat()
    return {
        "device": stat_result.st_dev,
        "inode": stat_result.st_ino,
        "bytes": stat_result.st_size,
        "mtime_ns": stat_result.st_mtime_ns,
        "ctime_ns": stat_result.st_ctime_ns,
    }


def _file_artifact_report(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    identity = _file_stat_identity(resolved)
    digest = _sha256_file(resolved)
    if _file_stat_identity(resolved) != identity:
        raise RuntimeError(f"artifact changed while SHA-256 was read: {resolved}")
    return {
        "path": str(resolved),
        **identity,
        "sha256": digest,
    }


def _numeric_summary(values: Sequence[int | float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    array = np.asarray(values, dtype=np.float64)
    if not bool(np.isfinite(array).all()):
        raise ValueError("summary values must be finite")
    return {
        "count": int(len(array)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
    }


def _selection_summary(
    sources: Sequence[MVSECGeometrySource],
    references: Sequence[MVSECTargetReference],
) -> dict[str, Any]:
    if not references:
        raise ValueError("cannot summarize an empty target selection")
    offsets = [
        reference.event_window_end_us - reference.label_timestamp_us
        for reference in references
    ]
    source_indices = sorted({reference.source_index for reference in references})
    target_artifacts = []
    for index in source_indices:
        source = sources[index]
        stat_result = source.ground_truth_path.stat()
        if source.target_format == "mvsec_gt_flow_npz_v1" and (
            stat_result.st_size != source.target_size_bytes
            or stat_result.st_mtime_ns != source.target_mtime_ns
        ):
            raise RuntimeError(
                "official MVSEC flow NPZ changed after manifest validation"
            )
        target_artifacts.append(
            {
                "path": str(source.ground_truth_path),
                "format": source.target_format,
                "file_id": source.target_file_id,
                "bytes": stat_result.st_size,
                "actual_size_bytes": stat_result.st_size,
                "mtime_ns": stat_result.st_mtime_ns,
                "manifest_expected_size_bytes": source.target_size_bytes,
                "manifest_source_mtime_ns": source.target_mtime_ns,
                "manifest_declared_sha256": source.target_sha256,
                "manifest_sha256_origin": source.target_sha256_origin,
                "sha256": source.target_sha256,
                "sha256_scope": (
                    "raw_npz_file_bytes"
                    if source.target_format == "mvsec_gt_flow_npz_v1"
                    else None
                ),
                "sha256_verification": (
                    (
                        "preprocessing_manifest_declaration_bound_to_current_"
                        "size_and_mtime"
                    )
                    if source.target_format == "mvsec_gt_flow_npz_v1"
                    else "unavailable_for_legacy_embedded_hdf5"
                ),
            }
        )
    return {
        "sequences": sorted(
            {sources[reference.source_index].sequence_id for reference in references}
        ),
        "target_formats": sorted(
            {sources[reference.source_index].target_format for reference in references}
        ),
        "target_artifacts": target_artifacts,
        "targets": len(references),
        "target_index_timestamp_sha256": reference_set_sha256(sources, references),
        "target_index_minimum": min(reference.target_index for reference in references),
        "target_index_maximum": max(reference.target_index for reference in references),
        "label_timestamp_us": _numeric_summary(
            [reference.label_timestamp_us for reference in references]
        ),
        "event_window_future_offset_us": _numeric_summary(offsets),
        "uses_events_after_label": any(offset > 0 for offset in offsets),
        "native_flow_interval_us": _numeric_summary(
            [reference.flow_interval_us for reference in references]
        ),
    }


@dataclass
class FlowMetricAccumulator:
    """Accumulate both F3-style per-frame means and global pixel means."""

    minimum_valid_pixels: int = DEFAULT_MINIMUM_VALID_PIXELS
    frames_seen: int = 0
    frames_evaluated: int = 0
    frames_skipped: int = 0
    sample_metric_sums: dict[str, float] = field(default_factory=dict)
    pixel_sums: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.minimum_valid_pixels <= 0:
            raise ValueError("minimum_valid_pixels must be positive")

    def update(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        valid: np.ndarray,
    ) -> bool:
        self.frames_seen += 1
        valid_pixels = int(np.count_nonzero(valid))
        if valid_pixels < self.minimum_valid_pixels:
            self.frames_skipped += 1
            return False
        current = flow_metric_sums(prediction, target, valid)
        finalized = finalize_flow_metrics(current)
        for name, value in finalized.items():
            self.sample_metric_sums[name] = self.sample_metric_sums.get(name, 0.0) + value
        accumulate_metric_sums(self.pixel_sums, current)
        self.frames_evaluated += 1
        return True

    def finalize(self) -> dict[str, Any]:
        if self.frames_evaluated <= 0:
            raise ValueError("no MVSEC frames have at least the required valid pixels")
        sample_average = {
            name: value / self.frames_evaluated
            for name, value in self.sample_metric_sums.items()
        }
        return {
            "sample_average": sample_average,
            "pixel_average": finalize_flow_metrics(self.pixel_sums),
            "frames_seen": self.frames_seen,
            "frames_evaluated": self.frames_evaluated,
            "frames_skipped_valid_below_threshold": self.frames_skipped,
            "minimum_valid_pixels_per_frame": self.minimum_valid_pixels,
            "valid_pixels_evaluated": int(self.pixel_sums["valid_pixels"]),
        }


@dataclass(frozen=True)
class ResolvedSchedule:
    window_ms: float
    stride_ms: float
    history_steps: int

    @property
    def window_us(self) -> int:
        return round(self.window_ms * 1_000)

    @property
    def stride_us(self) -> int:
        return round(self.stride_ms * 1_000)


def _resolve_schedule(args: argparse.Namespace, config: Any) -> ResolvedSchedule:
    recurrent = config.recurrent
    default_window = (
        recurrent.window_ms
        if recurrent.sequence_loader
        else config.windows.canonical_ms
    )
    default_stride = recurrent.stride_ms if recurrent.sequence_loader else default_window
    default_history = (
        recurrent.burn_in_steps + recurrent.sequence_length
        if recurrent.sequence_loader
        else 1
    )
    schedule = ResolvedSchedule(
        window_ms=float(
            default_window if args.window_ms is None else args.window_ms
        ),
        stride_ms=float(
            default_stride if args.stride_ms is None else args.stride_ms
        ),
        history_steps=int(
            default_history if args.history_steps is None else args.history_steps
        ),
    )
    if (
        not math.isfinite(schedule.window_ms)
        or not math.isfinite(schedule.stride_ms)
        or min(schedule.window_ms, schedule.stride_ms) <= 0
        or schedule.history_steps <= 0
    ):
        raise ValueError("window, stride, and history settings must be positive")
    return schedule


def _validate_alignment_schedule(
    alignment: str,
    schedule: ResolvedSchedule,
) -> None:
    if alignment not in {"causal", "f3_centered"}:
        raise ValueError("alignment must be causal or f3_centered")
    if (
        alignment == "f3_centered"
        and schedule.window_us != MVSEC_F3_CONTEXT_WINDOW_US
    ):
        raise ValueError(
            "f3_centered reproduces F3's fixed 50-ms ctx_flow support and "
            "therefore requires --window-ms 50"
        )


def _require_expected_sequence(
    sources: Sequence[MVSECGeometrySource], expected: str, role: str
) -> None:
    unexpected = [
        source.sequence_id
        for source in sources
        if expected not in source.sequence_id.split("__")
    ]
    if unexpected:
        raise ValueError(
            f"{role} must contain only {expected!r} left-camera sources, got {unexpected}"
        )


def _subsample_references(
    references: Sequence[MVSECTargetReference], maximum: int
) -> tuple[MVSECTargetReference, ...]:
    if maximum <= 0 or len(references) <= maximum:
        return tuple(references)
    indices = np.linspace(0, len(references) - 1, num=maximum, dtype=np.int64)
    return tuple(references[int(index)] for index in indices)


def _build_references(
    manifest: Path,
    *,
    split: str | None,
    expected_sequence: str,
    schedule: ResolvedSchedule,
    protocol: FlowProtocol,
    alignment: str,
    minimum_events: int,
    maximum_samples: int,
    evflownet_test: bool,
) -> tuple[tuple[MVSECGeometrySource, ...], tuple[MVSECTargetReference, ...]]:
    sources = read_mvsec_geometry_sources(manifest, kind="flow", split=split)
    _require_expected_sequence(sources, expected_sequence, "MVSEC flow protocol")
    store = H5EventStore(manifest)
    try:
        references = build_mvsec_target_references(
            store,
            sources,
            kind="flow",
            window_us=schedule.window_us,
            stride_us=schedule.stride_us,
            history_steps=schedule.history_steps,
            alignment=alignment,
            minimum_events=minimum_events,
            f3_evflownet_split=evflownet_test,
            maximum_samples=0,
        )
    finally:
        store.close()
    references = tuple(
        reference
        for reference in references
        if reference.flow_interval_us > 0
        and (
            alignment != "causal"
            or reference.label_timestamp_us
            - int(
                reference.flow_interval_us
                if protocol.name == "native"
                else protocol.nominal_horizon_us
            )
            >= sources[reference.source_index].t_start_us
        )
    )
    if not references:
        raise ValueError(
            "MVSEC flow selection has no labels with a positive prior interval "
            "and complete event-support history"
        )
    return sources, _subsample_references(references, maximum_samples)


def _model_image_size(config: Any) -> tuple[int, int]:
    image_size = tuple(config.model.image_size)
    if image_size != MVSEC_MODEL_IMAGE_SIZE:
        raise ValueError(
            "MVSEC flow protocol requires model.image_size [272,352] so the "
            "native 260x346 field of view is preserved by center padding"
        )
    return image_size


def _event_support_window_us(
    alignment: str,
    protocol: FlowProtocol,
) -> int | Literal["native_interval"] | None:
    if alignment == "causal":
        return (
            "native_interval"
            if protocol.name == "native"
            else protocol.nominal_horizon_us
        )
    if alignment == "f3_centered":
        # ``None`` makes MVSECGeometryDataset reuse the final centered model
        # input, matching F3's ctx_flow event-support mask.
        return None
    raise ValueError("alignment must be causal or f3_centered")


def _flow_additional_dependency_interval(
    reference: MVSECTargetReference,
    *,
    alignment: str,
    protocol: FlowProtocol,
) -> tuple[int, int]:
    """Return the event interval used to construct the flow validity mask."""

    if alignment == "causal":
        duration_us = (
            reference.flow_interval_us
            if protocol.name == "native"
            else protocol.nominal_horizon_us
        )
        if duration_us <= 0:
            raise ValueError("flow support dependency requires a positive duration")
        return reference.label_timestamp_us - duration_us, reference.label_timestamp_us
    if alignment == "f3_centered":
        return (
            reference.event_window_end_us - MVSEC_F3_CONTEXT_WINDOW_US,
            reference.event_window_end_us,
        )
    raise ValueError("alignment must be causal or f3_centered")


def _make_dataset(
    manifest: Path,
    sources: Sequence[MVSECGeometrySource],
    references: Sequence[MVSECTargetReference],
    *,
    config: Any,
    schedule: ResolvedSchedule,
    protocol: FlowProtocol,
    alignment: str,
) -> MVSECGeometryDataset:
    return MVSECGeometryDataset(
        manifest,
        sources,
        references,
        kind="flow",
        image_size=_model_image_size(config),
        window_us=schedule.window_us,
        stride_us=schedule.stride_us,
        history_steps=schedule.history_steps,
        representation=representation_from_config(config),
        flow_mask="f3",
        event_support_window_us=_event_support_window_us(alignment, protocol),
    )


def _make_loader(
    dataset: MVSECGeometryDataset,
    *,
    batch_size: int,
    workers: int,
    device: torch.device,
    shuffle: bool,
    seed: int,
) -> DataLoader[Mapping[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


def _resolve_device(value: str) -> torch.device:
    if value != "auto":
        device = torch.device(value)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def _autocast_context(device: torch.device, precision: str) -> Any:
    if precision == "fp32":
        return nullcontext()
    if precision == "bf16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    raise ValueError("bf16 evaluation is supported only on CUDA; use --precision fp32")


def _flow_batch_on_protocol(
    base_window_prediction: torch.Tensor,
    native_target: torch.Tensor,
    native_interval_us: torch.Tensor,
    *,
    protocol: FlowProtocol,
    base_window_us: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if base_window_prediction.shape != native_target.shape:
        raise ValueError("dense prediction and MVSEC target shapes differ")
    if native_interval_us.ndim != 1 or len(native_interval_us) != len(native_target):
        raise ValueError("flow intervals must have shape [B]")
    if base_window_us <= 0:
        raise ValueError("base_window_us must be positive")
    intervals = native_interval_us.to(
        device=base_window_prediction.device, dtype=torch.float32
    )
    if protocol.name == "native":
        if bool((intervals <= 0).any()) or not bool(torch.isfinite(intervals).all()):
            raise ValueError("native flow labels contain a non-positive interval")
        horizon_us = intervals
    else:
        horizon_us = torch.full_like(
            intervals,
            float(protocol.horizon_us(NATIVE_FLOW_HORIZON_US)),
        )
    prediction_scale = horizon_us / float(base_window_us)
    prediction = base_window_prediction.float() * prediction_scale[:, None, None, None]
    target = native_target.float() * protocol.ground_truth_scale
    return prediction, target, prediction_scale, horizon_us


def _reference_flow_scaling_summary(
    references: Sequence[MVSECTargetReference],
    *,
    protocol: FlowProtocol,
    base_window_us: int,
) -> dict[str, Any]:
    if not references:
        raise ValueError("cannot summarize flow scaling for an empty selection")
    horizons = [
        protocol.horizon_us(reference.flow_interval_us)
        for reference in references
    ]
    scales = [horizon / base_window_us for horizon in horizons]
    return {
        "stored_ground_truth_rate_hz": NATIVE_FLOW_HZ,
        "requested_rate_hz": protocol.requested_hz,
        "stored_ground_truth_multiplier": protocol.ground_truth_scale,
        "head_output_unit": "pixels_per_base_event_window",
        "base_event_window_us": base_window_us,
        "prediction_multiplier": _numeric_summary(scales),
        "evaluation_horizon_us": _numeric_summary(horizons),
    }


def _masked_endpoint_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    minimum_valid_pixels: int,
    epsilon: float = 1e-3,
) -> tuple[torch.Tensor | None, int]:
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("flow tensors must share shape [B,2,H,W]")
    if valid.shape != (len(prediction), *prediction.shape[-2:]) or valid.dtype != torch.bool:
        raise ValueError("valid flow mask must be boolean [B,H,W]")
    if minimum_valid_pixels <= 0 or epsilon <= 0:
        raise ValueError("loss thresholds must be positive")
    frame_keep = valid.flatten(1).sum(dim=1) >= minimum_valid_pixels
    selected = valid & frame_keep[:, None, None]
    kept_frames = int(frame_keep.sum().item())
    if kept_frames == 0:
        return None, 0
    if not bool(torch.isfinite(prediction).all()):
        raise FloatingPointError("flow head produced NaN or infinity")
    difference = prediction - target
    endpoint = torch.sqrt(torch.square(difference).sum(dim=1) + epsilon**2)
    return endpoint[selected].mean(), kept_frames


def _head_spec(head: RecurrentTokenFlowHead) -> dict[str, Any]:
    return {
        "class": "RecurrentTokenFlowHead",
        "embed_dim": head.embed_dim,
        "hidden_dim": head.hidden_dim,
        "head_depth": head.head_depth,
        "flow_scale": head.flow_scale,
        "max_displacement_pixels_per_base_window": head.max_displacement,
    }


def _head_state_sha256(head: RecurrentTokenFlowHead) -> str:
    """Hash tensor names, shapes, dtypes, and values independently of torch.save."""

    digest = hashlib.sha256()
    for name, value in sorted(head.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _make_probe_head(
    model: Any,
    config: Any,
    *,
    initialization: str,
    seed: int,
) -> RecurrentTokenFlowHead:
    if initialization not in {"random", "cmax"}:
        raise ValueError("head initialization must be random or cmax")
    source = model.cmax_flow_head
    if initialization == "cmax" and source is None:
        raise ValueError("--head-init cmax requires a checkpoint trained with CMax")
    if initialization == "random":
        hidden_dim = CANONICAL_RANDOM_PROBE_HIDDEN_DIM
        head_depth = CANONICAL_RANDOM_PROBE_HEAD_DEPTH
        flow_scale = CANONICAL_RANDOM_PROBE_FLOW_SCALE
        max_displacement = CANONICAL_RANDOM_PROBE_MAX_DISPLACEMENT
    else:
        assert source is not None
        hidden_dim = source.hidden_dim
        head_depth = source.head_depth
        flow_scale = source.flow_scale
        max_displacement = source.max_displacement
    torch.manual_seed(seed)
    head = RecurrentTokenFlowHead(
        config.model.embed_dim,
        hidden_dim=hidden_dim,
        head_depth=head_depth,
        flow_scale=flow_scale,
        max_displacement=max_displacement,
    )
    if initialization == "cmax":
        assert source is not None
        state = {name: value.detach().cpu() for name, value in source.state_dict().items()}
        head.load_state_dict(state, strict=True)
    return head


def _grid_size(model: Any) -> tuple[int, int]:
    grid = getattr(model.online_encoder, "grid_size", None)
    if (
        not isinstance(grid, tuple)
        or len(grid) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in grid)
        or min(grid) <= 0
    ):
        raise ValueError("checkpoint encoder has no fixed two-dimensional patch grid")
    return grid


def _train_probe_head(
    model: Any,
    head: RecurrentTokenFlowHead,
    dataset: MVSECGeometryDataset,
    *,
    schedule: ResolvedSchedule,
    protocol: FlowProtocol,
    device: torch.device,
    precision: str,
    epochs: int,
    batch_size: int,
    workers: int,
    learning_rate: float,
    weight_decay: float,
    minimum_valid_pixels: int,
    seed: int,
) -> list[dict[str, float | int]]:
    loader = _make_loader(
        dataset,
        batch_size=batch_size,
        workers=workers,
        device=device,
        shuffle=True,
        seed=seed,
    )
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    grid_size = _grid_size(model)
    history: list[dict[str, float | int]] = []
    head.train()
    for epoch in range(epochs):
        loss_sum = 0.0
        kept_sum = 0
        skipped_sum = 0
        for batch in tqdm(loader, desc=f"MVSEC probe epoch {epoch + 1}/{epochs}"):
            x = batch["x"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)
            intervals = batch["flow_interval_us"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, precision):
                tokens = extract_frozen_mvsec_tokens(
                    model,
                    x,
                    duration_ms=schedule.window_ms,
                )
                patch_flow = head(tokens, grid_size)
                dense = dense_patch_prediction(patch_flow, tuple(target.shape[-2:]))
            prediction, scaled_target, _, _ = _flow_batch_on_protocol(
                dense,
                target,
                intervals,
                protocol=protocol,
                base_window_us=schedule.window_us,
            )
            loss, kept = _masked_endpoint_loss(
                prediction,
                scaled_target,
                valid,
                minimum_valid_pixels=minimum_valid_pixels,
            )
            if loss is None:
                skipped_sum += len(x)
                continue
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * kept
            kept_sum += kept
            skipped_sum += len(x) - kept
        if kept_sum == 0:
            raise ValueError("no training frame reached the valid-pixel threshold")
        record: dict[str, float | int] = {
            "epoch": epoch + 1,
            "mean_endpoint_loss": loss_sum / kept_sum,
            "frames_used": kept_sum,
            "frames_skipped": skipped_sum,
        }
        history.append(record)
        print(
            f"[mvsec-flow] epoch {epoch + 1}/{epochs}: "
            f"loss={record['mean_endpoint_loss']:.6f}, frames={kept_sum}"
        )
    return history


@torch.no_grad()
def _evaluate_head(
    model: Any,
    head: RecurrentTokenFlowHead,
    dataset: MVSECGeometryDataset,
    *,
    schedule: ResolvedSchedule,
    protocol: FlowProtocol,
    device: torch.device,
    precision: str,
    batch_size: int,
    workers: int,
    minimum_valid_pixels: int,
    save_visualizations: int = 0,
    visualization_dir: Path | None = None,
    visualization_max_events: int = 200_000,
    visualization_context: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    loader = _make_loader(
        dataset,
        batch_size=batch_size,
        workers=workers,
        device=device,
        shuffle=False,
        seed=0,
    )
    metrics = FlowMetricAccumulator(minimum_valid_pixels=minimum_valid_pixels)
    prediction_scales: list[float] = []
    horizons: list[float] = []
    visualization_entries: list[dict[str, Any]] = []
    sample_cursor = 0
    grid_size = _grid_size(model)
    head.eval()
    for batch in tqdm(loader, desc="MVSEC flow evaluation"):
        x = batch["x"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)
        intervals = batch["flow_interval_us"].to(device, non_blocking=True)
        with _autocast_context(device, precision):
            tokens = extract_frozen_mvsec_tokens(
                model,
                x,
                duration_ms=schedule.window_ms,
            )
            patch_flow = head(tokens, grid_size)
            dense = dense_patch_prediction(patch_flow, tuple(target.shape[-2:]))
        prediction, scaled_target, scales, horizon_us = _flow_batch_on_protocol(
            dense,
            target,
            intervals,
            protocol=protocol,
            base_window_us=schedule.window_us,
        )
        prediction_scales.extend(scales.cpu().tolist())
        horizons.extend(horizon_us.cpu().tolist())
        for index in range(len(x)):
            included_in_metrics = metrics.update(
                prediction[index].float().cpu().numpy(),
                scaled_target[index].float().cpu().numpy(),
                valid[index].cpu().numpy(),
            )
            if (
                included_in_metrics
                and len(visualization_entries) < save_visualizations
            ):
                if visualization_dir is None:
                    raise RuntimeError("visualization directory was not resolved")
                from event_window_jepa.downstream.mvsec_visualize import (
                    extract_snapshot_events,
                    safe_component,
                    write_snapshot,
                )

                dataset_index = sample_cursor + index
                reference = dataset.references[dataset_index]
                source = dataset.sources[reference.source_index]
                # IWE must use the same physical horizon as the scaled GT and
                # prediction.  It is intentionally distinct from a centered
                # F3 model-input/support window.
                event_end_us = reference.label_timestamp_us
                event_duration_us = int(round(float(horizon_us[index].item())))
                event_arrays, event_metadata = extract_snapshot_events(
                    dataset,
                    dataset_index,
                    t_end_us=event_end_us,
                    duration_us=event_duration_us,
                    maximum_events=visualization_max_events,
                )
                filename = (
                    f"{dataset_index:06d}_"
                    f"{safe_component(source.sequence_id)}_"
                    f"target-{reference.target_index:06d}.npz"
                )
                entry = write_snapshot(
                    visualization_dir / filename,
                    kind="flow",
                    event_image=batch["x"][index, -1].numpy(),
                    target=scaled_target[index].float().cpu().numpy(),
                    prediction=prediction[index].float().cpu().numpy(),
                    valid=valid[index].cpu().numpy(),
                    metadata={
                        **dict(visualization_context or {}),
                        **event_metadata,
                        "dataset_index": dataset_index,
                        "sequence_id": source.sequence_id,
                        "target_index": reference.target_index,
                        "label_timestamp_us": reference.label_timestamp_us,
                        "event_alignment": "causal_evaluation_flow_horizon",
                        "event_representation_window_end_us": reference.event_window_end_us,
                        "event_representation_window_us": dataset.window_us,
                        "coordinate_frame": "native_distorted_left_DAVIS_center_padded",
                        "native_sensor_height_width": [260, 346],
                        "model_canvas_height_width": list(prediction.shape[-2:]),
                        "native_sensor_center_padding_yx": [6, 3],
                        "target_artifact_path": str(source.ground_truth_path),
                        "target_artifact_format": source.target_format,
                        "target_artifact_sha256": source.target_sha256,
                        "target_artifact_sha256_origin": source.target_sha256_origin,
                        "flow_protocol": protocol.name,
                        "flow_horizon_us": float(horizon_us[index].item()),
                        "flow_prediction_multiplier": float(scales[index].item()),
                        "ground_truth_multiplier": protocol.ground_truth_scale,
                        "flow_unit": "pixels_per_evaluation_horizon",
                        "included_in_aggregate_metrics": True,
                        "valid_pixels": int(valid[index].sum().item()),
                        "minimum_valid_pixels": minimum_valid_pixels,
                        "visualization_selection": (
                            "first_N_metric_eligible_in_evaluation_order"
                        ),
                    },
                    events=event_arrays,
                )
                entry["dataset_index"] = dataset_index
                visualization_entries.append(entry)
        sample_cursor += len(x)
    scaling = {
        "stored_ground_truth_rate_hz": NATIVE_FLOW_HZ,
        "requested_rate_hz": protocol.requested_hz,
        "stored_ground_truth_multiplier": protocol.ground_truth_scale,
        "head_output_unit": "pixels_per_base_event_window",
        "base_event_window_us": schedule.window_us,
        "prediction_multiplier": _numeric_summary(prediction_scales),
        "evaluation_horizon_us": _numeric_summary(horizons),
    }
    visualization_index = None
    if save_visualizations:
        if visualization_dir is None:
            raise RuntimeError("visualization directory was not resolved")
        from event_window_jepa.downstream.mvsec_visualize import write_snapshot_index

        visualization_index = write_snapshot_index(
            visualization_dir,
            visualization_entries,
            requested=save_visualizations,
            context=dict(visualization_context or {}),
        )
    return metrics.finalize(), scaling, visualization_index


def _checkpoint_report(
    artifact: Mapping[str, Any],
    config: Any,
) -> dict[str, Any]:
    resolved_config_hash = config_hash(config)
    return {
        "path": artifact["path"],
        "device": artifact["device"],
        "inode": artifact["inode"],
        "bytes": artifact["bytes"],
        "mtime_ns": artifact["mtime_ns"],
        "ctime_ns": artifact["ctime_ns"],
        "checkpoint_sha256": artifact["sha256"],
        "config_sha256": resolved_config_hash,
        "resolved_config_sha256": resolved_config_hash,
    }


def _manifest_report(path: Path) -> dict[str, Any]:
    return _file_artifact_report(path)


def _require_unchanged_manifest(
    path: Path,
    expected: Mapping[str, Any],
) -> None:
    if _manifest_report(path) != dict(expected):
        raise RuntimeError("MVSEC manifest changed while the flow run was active")


def _require_unchanged_checkpoint_identity(
    path: Path,
    expected: Mapping[str, Any],
) -> None:
    resolved = path.expanduser().resolve()
    expected_identity = {
        name: int(expected[name])
        for name in ("device", "inode", "bytes", "mtime_ns", "ctime_ns")
    }
    if str(resolved) != expected["path"] or (
        _file_stat_identity(resolved) != expected_identity
    ):
        raise RuntimeError("checkpoint changed while the flow run was active")


def _alignment_report(
    alignment: str,
    schedule: ResolvedSchedule,
) -> dict[str, Any]:
    _validate_alignment_schedule(alignment, schedule)
    return {
        "mode": alignment,
        "causal": alignment == "causal",
        "final_model_window_us": schedule.window_us,
        "f3_ctx_flow_fixed_window_us": (
            None
            if alignment == "causal"
            else MVSEC_F3_CONTEXT_WINDOW_US
        ),
        "nominal_future_event_use_us": (
            0 if alignment == "causal" else schedule.window_us // 2
        ),
        "f3_centered_warning": (
            None
            if alignment == "causal"
            else "the final event window includes post-label events"
        ),
    }


def _protocol_report(
    args: argparse.Namespace,
    schedule: ResolvedSchedule,
    protocol: FlowProtocol,
    *,
    temporal_dev_split: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _validate_alignment_schedule(args.alignment, schedule)
    if args.alignment == "causal":
        if protocol.name == "native":
            event_support = {
                "source": "per_reference_previous_to_current_flow_interval",
                "duration_us": "reference.flow_interval_us",
                "actual_duration_summary": (
                    "see each selection.native_flow_interval_us"
                ),
                "interval": "(label-flow_interval_us,label]",
                "uses_events_after_label": False,
            }
        else:
            event_support = {
                "source": "dedicated_label_ending_dt1_flow_horizon",
                "duration_us": protocol.nominal_horizon_us,
                "interval": (
                    f"(label-{protocol.nominal_horizon_us}us,label]"
                ),
                "uses_events_after_label": False,
            }
    elif args.alignment == "f3_centered":
        event_support = {
            "source": "fixed_50ms_f3_centered_model_input_ctx_flow",
            "duration_us": MVSEC_F3_CONTEXT_WINDOW_US,
            "interval": "(event_window_end-window,event_window_end]",
            "nominally_centered_on_label": True,
            "uses_events_after_label": True,
        }
    else:
        raise ValueError("alignment must be causal or f3_centered")
    protocol_stage = getattr(args, "protocol_stage", "final")
    if protocol_stage not in {"dev", "final"}:
        raise ValueError("protocol_stage must be dev or final")
    if protocol_stage == "dev" and temporal_dev_split is None:
        raise ValueError("dev protocol report requires its temporal split contract")
    if protocol_stage == "final" and temporal_dev_split is not None:
        raise ValueError("final protocol cannot include a temporal dev split")
    return {
        "name": "frozen_recurrent_mvsec_native_flow_labels_v1",
        "stage": protocol_stage,
        "target_timebase_contract": "mvsec_native_flow_label_time_interval_v1",
        "coordinate_frame": "native_distorted_left_DAVIS",
        "official_flow_target": {
            "format": "mvsec_gt_flow_npz_v1",
            "keys": ["timestamps", "x_flow_dist", "y_flow_dist"],
            "channel_order": ["x_flow_dist", "y_flow_dist"],
            "timestamps": "absolute_source_clock_seconds",
            "legacy_embedded_hdf5_fallback_requires_explicit_format": True,
        },
        "model_canvas_height_width": list(MVSEC_MODEL_IMAGE_SIZE),
        "native_sensor_center_padding_yx": [6, 3],
        "alignment": _alignment_report(args.alignment, schedule),
        "event_history": {
            "window_ms": schedule.window_ms,
            "stride_ms": schedule.stride_ms,
            "history_steps": schedule.history_steps,
            "state_reset": "once_per_target_sample",
            "history_span_us": (
                schedule.window_us
                + (schedule.history_steps - 1) * schedule.stride_us
            ),
        },
        "temporal_dev_split": (
            None if temporal_dev_split is None else dict(temporal_dev_split)
        ),
        "representation_pretraining_visibility_contract": (
            {
                "protocol_class": "inductive_cross_recording_final_evaluation",
                "dev_recording_events_visible": None,
                "geometry_labels_visible_to_pretraining": False,
            }
            if temporal_dev_split is None
            else dict(
                temporal_dev_split[
                    "representation_pretraining_visibility_contract"
                ]
            )
        ),
        "flow_rate": {
            "cli_value": args.dt,
            "protocol": protocol.name,
            "target_timestamps": "native_flow_npz",
            "ground_truth_resampling": "none",
            "exact_evflownet_800_frame_protocol": False,
            "benchmark_status": "diagnostic_not_published_800_frame_benchmark",
            "limitation": (
                "exact 800-frame reproduction requires APS-timestamp flow "
                "interpolation/composition and is not implemented"
            ),
        },
        "validity_mask": {
            "ground_truth_conditions": [
                "finite_ground_truth",
                "nonzero_ground_truth",
            ],
            "spatial_conditions": [
                "inside_original_260x346_sensor",
                "original_sensor_row_below_193",
            ],
            "event_support": event_support,
        },
        "minimum_valid_pixels_per_frame": args.minimum_valid_pixels,
        "evflownet_test_interval": {
            "applied": protocol_stage == "final",
            "recording": "outdoor_day1" if protocol_stage == "final" else None,
            "selection_basis": "label_internal_timestamp_us_not_fraction",
            "start_inclusive_us": EVFLOWNET_TEST_START_US,
            "stop_exclusive_us": EVFLOWNET_TEST_STOP_US,
            "exact_800_frame_reproduction": False,
            "selected_timestamp_grid": "native_flow_npz_not_45hz_APS_frames",
            "dev_stage_note": (
                None
                if protocol_stage == "final"
                else "not applied; dev evaluates the late outdoor_day2 partition"
            ),
        },
        "dt4_supported": False,
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _save_head_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def _validate_common_args(args: argparse.Namespace) -> FlowProtocol:
    protocol = resolve_flow_protocol(args.dt)
    positive = {
        "batch_size": args.batch_size,
        "minimum_valid_pixels": args.minimum_valid_pixels,
        "minimum_events": args.minimum_events,
    }
    if any(value <= 0 for value in positive.values()):
        raise ValueError(f"arguments must be positive: {positive}")
    if (
        args.workers < 0
        or args.max_eval_samples < 0
        or args.save_visualizations < 0
    ):
        raise ValueError("workers and sample limits cannot be negative")
    if args.visualization_max_events <= 0:
        raise ValueError("visualization_max_events must be positive")
    if args.visualization_dir is not None and args.save_visualizations == 0:
        raise ValueError("--visualization-dir requires --save-visualizations > 0")
    if args.precision not in {"fp32", "bf16"}:
        raise ValueError("precision must be fp32 or bf16")
    if args.protocol_stage not in {"dev", "final"}:
        raise ValueError("protocol_stage must be dev or final")
    if not math.isfinite(args.dev_fraction) or not 0.0 < args.dev_fraction < 1.0:
        raise ValueError("dev_fraction must be finite and strictly between 0 and 1")
    if args.dev_guard_ms is not None and (
        not math.isfinite(args.dev_guard_ms) or args.dev_guard_ms <= 0
    ):
        raise ValueError("dev_guard_ms must be finite and positive when provided")
    return protocol


def _requested_dev_guard_us(args: argparse.Namespace) -> int | None:
    if args.dev_guard_ms is None:
        return None
    return math.ceil(float(args.dev_guard_ms) * 1_000.0)


def _validate_output_paths(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError("--output-dir must be a directory path")
    outputs = {"report": output_dir / "report.json"}
    if args.command == "probe":
        outputs["flow head"] = output_dir / "flow_head.pt"
    protected = {
        "encoder checkpoint": args.checkpoint.expanduser().resolve(),
        "evaluation manifest": args.eval_manifest.expanduser().resolve(),
    }
    if args.command == "probe":
        protected["training manifest"] = args.train_manifest.expanduser().resolve()
    if getattr(args, "save_visualizations", 0):
        visualization_dir = _visualization_output_dir(args)
        if visualization_dir.exists() and not visualization_dir.is_dir():
            raise ValueError("--visualization-dir must be a directory path")
    for output_name, output_path in outputs.items():
        for protected_name, protected_path in protected.items():
            if output_path == protected_path:
                raise ValueError(
                    f"{output_name} output would overwrite the {protected_name}: "
                    f"{output_path}"
                )


def _visualization_output_dir(args: argparse.Namespace) -> Path:
    configured = getattr(args, "visualization_dir", None)
    if configured is None:
        return args.output_dir.expanduser().resolve() / "visualizations"
    return configured.expanduser().resolve()


def _load_frozen_checkpoint(
    checkpoint: Path,
    device: torch.device,
) -> tuple[Any, Any]:
    model, config = load_pretrained_model(checkpoint, device=device)
    if not isinstance(
        model.online_encoder, RecurrentVJEPA21EventVisionTransformer
    ):
        raise ValueError("MVSEC flow probing requires a recurrent encoder checkpoint")
    model.eval()
    model.requires_grad_(False)
    return model, config


def _run_probe(args: argparse.Namespace) -> dict[str, Any]:
    protocol = _validate_common_args(args)
    _validate_output_paths(args)
    if args.epochs <= 0 or args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("probe optimization arguments are invalid")
    if args.max_train_samples < 0:
        raise ValueError("max_train_samples cannot be negative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _resolve_device(args.device)
    checkpoint_artifact = _file_artifact_report(args.checkpoint)
    model, config = _load_frozen_checkpoint(args.checkpoint, device)
    if _file_artifact_report(args.checkpoint) != checkpoint_artifact:
        raise RuntimeError("checkpoint changed while it was being loaded")
    _model_image_size(config)
    checkpoint_report = _checkpoint_report(checkpoint_artifact, config)
    schedule = _resolve_schedule(args, config)
    _validate_alignment_schedule(args.alignment, schedule)
    head = _make_probe_head(
        model,
        config,
        initialization=args.head_init,
        seed=args.seed,
    ).to(device)
    train_manifest_path = args.train_manifest.expanduser().resolve()
    eval_manifest_path = args.eval_manifest.expanduser().resolve()
    train_manifest_artifact = _manifest_report(train_manifest_path)
    eval_manifest_artifact = (
        train_manifest_artifact
        if eval_manifest_path == train_manifest_path
        else _manifest_report(eval_manifest_path)
    )
    temporal_dev_split: dict[str, Any] | None = None
    if args.protocol_stage == "dev":
        if train_manifest_path != eval_manifest_path:
            raise ValueError(
                "flow dev protocol requires --train-manifest and "
                "--eval-manifest to resolve to the same outdoor_day2 manifest"
            )
        train_sources, all_day2_references = _build_references(
            train_manifest_path,
            split=args.train_split,
            expected_sequence="outdoor_day2",
            schedule=schedule,
            protocol=protocol,
            alignment=args.alignment,
            minimum_events=args.minimum_events,
            maximum_samples=0,
            evflownet_test=False,
        )
        train_references, eval_references, temporal_dev_split = (
            split_mvsec_temporal_dev_references(
                train_sources,
                all_day2_references,
                window_us=schedule.window_us,
                stride_us=schedule.stride_us,
                history_steps=schedule.history_steps,
                alignment=args.alignment,
                dev_fraction=args.dev_fraction,
                guard_us=_requested_dev_guard_us(args),
                maximum_train_samples=args.max_train_samples,
                maximum_dev_samples=args.max_eval_samples,
                additional_dependency_interval=lambda reference: (
                    _flow_additional_dependency_interval(
                        reference,
                        alignment=args.alignment,
                        protocol=protocol,
                    )
                ),
            )
        )
        temporal_dev_split["manifest_row_split"] = args.train_split
        temporal_dev_split["eval_split_cli_effect"] = (
            "ignored_in_dev; day2 training rows are split chronologically"
        )
        eval_sources = train_sources
        training_split_name = "chronological_early_train"
        evaluation_split_name = "chronological_late_dev"
        evaluation_role = "dev"
    else:
        train_sources, train_references = _build_references(
            train_manifest_path,
            split=args.train_split,
            expected_sequence="outdoor_day2",
            schedule=schedule,
            protocol=protocol,
            alignment=args.alignment,
            minimum_events=args.minimum_events,
            maximum_samples=args.max_train_samples,
            evflownet_test=False,
        )
        eval_sources, eval_references = _build_references(
            eval_manifest_path,
            split=args.eval_split,
            expected_sequence="outdoor_day1",
            schedule=schedule,
            protocol=protocol,
            alignment=args.alignment,
            minimum_events=args.minimum_events,
            maximum_samples=args.max_eval_samples,
            evflownet_test=True,
        )
        training_split_name = args.train_split
        evaluation_split_name = args.eval_split
        evaluation_role = "final_test"
    train_dataset = _make_dataset(
        args.train_manifest,
        train_sources,
        train_references,
        config=config,
        schedule=schedule,
        protocol=protocol,
        alignment=args.alignment,
    )
    eval_dataset = _make_dataset(
        args.eval_manifest,
        eval_sources,
        eval_references,
        config=config,
        schedule=schedule,
        protocol=protocol,
        alignment=args.alignment,
    )
    try:
        history = _train_probe_head(
            model,
            head,
            train_dataset,
            schedule=schedule,
            protocol=protocol,
            device=device,
            precision=args.precision,
            epochs=args.epochs,
            batch_size=args.batch_size,
            workers=args.workers,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            minimum_valid_pixels=args.minimum_valid_pixels,
            seed=args.seed,
        )
        metrics, flow_scaling, visualization_index = _evaluate_head(
            model,
            head,
            eval_dataset,
            schedule=schedule,
            protocol=protocol,
            device=device,
            precision=args.precision,
            batch_size=args.batch_size,
            workers=args.workers,
            minimum_valid_pixels=args.minimum_valid_pixels,
            save_visualizations=args.save_visualizations,
            visualization_dir=(
                _visualization_output_dir(args)
                if args.save_visualizations
                else None
            ),
            visualization_max_events=args.visualization_max_events,
            visualization_context={
                "evaluation_command": "probe",
                "checkpoint_sha256": checkpoint_report["checkpoint_sha256"],
                "eval_manifest_sha256": eval_manifest_artifact["sha256"],
                "evaluation_target_reference_sha256": reference_set_sha256(
                    eval_sources, eval_references
                ),
                "head_initialization": args.head_init,
                "head_spec": _head_spec(head),
                "head_state_sha256": _head_state_sha256(head),
                "alignment": args.alignment,
                "protocol_stage": args.protocol_stage,
                "evaluation_role": evaluation_role,
                "seed": args.seed,
            },
        )
    finally:
        train_dataset.close()
        eval_dataset.close()

    _require_unchanged_manifest(args.train_manifest, train_manifest_artifact)
    _require_unchanged_manifest(args.eval_manifest, eval_manifest_artifact)
    _require_unchanged_checkpoint_identity(args.checkpoint, checkpoint_artifact)
    train_selection = _selection_summary(train_sources, train_references)
    eval_selection = _selection_summary(eval_sources, eval_references)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    head_path = args.output_dir / "flow_head.pt"
    _save_head_atomic(
        head_path,
        {
            "schema_version": 1,
            "head": {
                name: value.detach().cpu()
                for name, value in head.state_dict().items()
            },
            "head_spec": _head_spec(head),
            "head_initialization": args.head_init,
            "head_architecture_source": (
                "canonical_random_mvsec_flow_probe_v1"
                if args.head_init == "random"
                else "checkpoint_cmax_head_spec"
            ),
            "source_checkpoint_sha256": checkpoint_report["checkpoint_sha256"],
            "source_config_sha256": checkpoint_report["resolved_config_sha256"],
            "flow_rate": protocol.name,
            "base_event_window_us": schedule.window_us,
            "protocol_stage": args.protocol_stage,
            "training_target_reference_sha256": reference_set_sha256(
                train_sources, train_references
            ),
        },
    )
    report = {
        "schema_version": 1,
        "command": "probe",
        "checkpoint": checkpoint_report,
        "encoder": {"frozen": True, "patch_grid": list(_grid_size(model))},
        "head": {
            **_head_spec(head),
            "initialization": args.head_init,
            "architecture_source": (
                "canonical_random_mvsec_flow_probe_v1"
                if args.head_init == "random"
                else "checkpoint_cmax_head_spec"
            ),
            "state_sha256": _head_state_sha256(head),
            "checkpoint": str(head_path.resolve()),
            "checkpoint_sha256": _sha256_file(head_path),
        },
        "protocol": _protocol_report(
            args,
            schedule,
            protocol,
            temporal_dev_split=temporal_dev_split,
        ),
        "training": {
            "manifest": str(args.train_manifest.expanduser().resolve()),
            "manifest_artifact": train_manifest_artifact,
            "split": training_split_name,
            "role": "dev_train" if args.protocol_stage == "dev" else "train",
            "selection": train_selection,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "precision": args.precision,
            "probe_seed": args.seed,
            "loss": "masked_mean_endpoint_charbonnier_epsilon_1e-3",
            "flow_scaling": _reference_flow_scaling_summary(
                train_references,
                protocol=protocol,
                base_window_us=schedule.window_us,
            ),
            "history": history,
        },
        "evaluation": {
            "manifest": str(args.eval_manifest.expanduser().resolve()),
            "manifest_artifact": eval_manifest_artifact,
            "split": evaluation_split_name,
            "role": evaluation_role,
            "used_for_model_selection": args.protocol_stage == "dev",
            "selection": eval_selection,
            "flow_scaling": flow_scaling,
            "metrics": metrics,
            "visualizations": visualization_index,
        },
        "runtime": {
            "device": str(device),
            "precision": args.precision,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
    }
    _write_json_atomic(args.output_dir / "report.json", report)
    return report


def _run_cmax_eval(args: argparse.Namespace) -> dict[str, Any]:
    protocol = _validate_common_args(args)
    _validate_output_paths(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _resolve_device(args.device)
    checkpoint_artifact = _file_artifact_report(args.checkpoint)
    model, config = _load_frozen_checkpoint(args.checkpoint, device)
    if _file_artifact_report(args.checkpoint) != checkpoint_artifact:
        raise RuntimeError("checkpoint changed while it was being loaded")
    _model_image_size(config)
    checkpoint_report = _checkpoint_report(checkpoint_artifact, config)
    if model.cmax_flow_head is None or not config.cmax.enabled:
        raise ValueError("cmax-eval requires a checkpoint with a trained CMax flow head")
    schedule = _resolve_schedule(args, config)
    _validate_alignment_schedule(args.alignment, schedule)
    eval_manifest_artifact = _manifest_report(args.eval_manifest)
    temporal_dev_split: dict[str, Any] | None = None
    if args.protocol_stage == "dev":
        eval_sources, all_day2_references = _build_references(
            args.eval_manifest,
            split="train",
            expected_sequence="outdoor_day2",
            schedule=schedule,
            protocol=protocol,
            alignment=args.alignment,
            minimum_events=args.minimum_events,
            maximum_samples=0,
            evflownet_test=False,
        )
        _, eval_references, temporal_dev_split = (
            split_mvsec_temporal_dev_references(
                eval_sources,
                all_day2_references,
                window_us=schedule.window_us,
                stride_us=schedule.stride_us,
                history_steps=schedule.history_steps,
                alignment=args.alignment,
                dev_fraction=args.dev_fraction,
                guard_us=_requested_dev_guard_us(args),
                maximum_train_samples=0,
                maximum_dev_samples=args.max_eval_samples,
                additional_dependency_interval=lambda reference: (
                    _flow_additional_dependency_interval(
                        reference,
                        alignment=args.alignment,
                        protocol=protocol,
                    )
                ),
            )
        )
        temporal_dev_split["manifest_row_split"] = "train"
        temporal_dev_split["eval_split_cli_effect"] = (
            "ignored_in_dev; day2 train rows are split chronologically"
        )
        temporal_dev_split["early_partition_usage"] = (
            "reported_for_boundary_audit_only; cmax-eval consumes late_dev_only"
        )
        evaluation_split_name = "chronological_late_dev"
        evaluation_role = "dev"
    else:
        eval_sources, eval_references = _build_references(
            args.eval_manifest,
            split=args.eval_split,
            expected_sequence="outdoor_day1",
            schedule=schedule,
            protocol=protocol,
            alignment=args.alignment,
            minimum_events=args.minimum_events,
            maximum_samples=args.max_eval_samples,
            evflownet_test=True,
        )
        evaluation_split_name = args.eval_split
        evaluation_role = "final_test"
    dataset = _make_dataset(
        args.eval_manifest,
        eval_sources,
        eval_references,
        config=config,
        schedule=schedule,
        protocol=protocol,
        alignment=args.alignment,
    )
    try:
        metrics, flow_scaling, visualization_index = _evaluate_head(
            model,
            model.cmax_flow_head,
            dataset,
            schedule=schedule,
            protocol=protocol,
            device=device,
            precision=args.precision,
            batch_size=args.batch_size,
            workers=args.workers,
            minimum_valid_pixels=args.minimum_valid_pixels,
            save_visualizations=args.save_visualizations,
            visualization_dir=(
                _visualization_output_dir(args)
                if args.save_visualizations
                else None
            ),
            visualization_max_events=args.visualization_max_events,
            visualization_context={
                "evaluation_command": "cmax-eval",
                "checkpoint_sha256": checkpoint_report["checkpoint_sha256"],
                "eval_manifest_sha256": eval_manifest_artifact["sha256"],
                "evaluation_target_reference_sha256": reference_set_sha256(
                    eval_sources, eval_references
                ),
                "head_initialization": "checkpoint_cmax",
                "head_spec": _head_spec(model.cmax_flow_head),
                "head_state_sha256": _head_state_sha256(model.cmax_flow_head),
                "alignment": args.alignment,
                "protocol_stage": args.protocol_stage,
                "evaluation_role": evaluation_role,
                "seed": args.seed,
            },
        )
    finally:
        dataset.close()
    _require_unchanged_manifest(args.eval_manifest, eval_manifest_artifact)
    _require_unchanged_checkpoint_identity(args.checkpoint, checkpoint_artifact)
    eval_selection = _selection_summary(eval_sources, eval_references)
    report = {
        "schema_version": 1,
        "command": "cmax-eval",
        "checkpoint": checkpoint_report,
        "encoder": {"frozen": True, "patch_grid": list(_grid_size(model))},
        "head": {
            **_head_spec(model.cmax_flow_head),
            "initialization": "checkpoint_cmax",
            "architecture_source": "checkpoint_cmax_head_spec",
            "state_sha256": _head_state_sha256(model.cmax_flow_head),
        },
        "protocol": _protocol_report(
            args,
            schedule,
            protocol,
            temporal_dev_split=temporal_dev_split,
        ),
        "evaluation": {
            "manifest": str(args.eval_manifest.expanduser().resolve()),
            "manifest_artifact": eval_manifest_artifact,
            "split": evaluation_split_name,
            "role": evaluation_role,
            "used_for_model_selection": args.protocol_stage == "dev",
            "selection": eval_selection,
            "flow_scaling": flow_scaling,
            "metrics": metrics,
            "visualizations": visualization_index,
        },
        "runtime": {
            "device": str(device),
            "precision": args.precision,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
    }
    _write_json_atomic(args.output_dir / "report.json", report)
    return report


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-split", default="test")
    parser.add_argument(
        "--protocol-stage",
        choices=("dev", "final"),
        default="final",
        help=(
            "dev uses a guarded early/late split of outdoor_day2; final keeps "
            "the outdoor_day2-to-outdoor_day1 benchmark protocol"
        ),
    )
    parser.add_argument(
        "--dev-fraction",
        type=float,
        default=0.2,
        help="late outdoor_day2 label fraction used only by --protocol-stage dev",
    )
    parser.add_argument(
        "--dev-guard-ms",
        type=float,
        help=(
            "label-time guard for dev; default auto is the complete recurrent "
            "history span plus any centered-alignment rounding allowance"
        ),
    )
    parser.add_argument(
        "--alignment",
        choices=("causal", "f3_centered"),
        default="causal",
        help="causal is the default; f3_centered explicitly permits future events",
    )
    parser.add_argument(
        "--dt",
        choices=("native", "dt1", "dt4"),
        default="native",
        help=(
            "dt1 is a 20/45 scalar diagnostic on native flow-label timestamps; "
            "dt4 is parsed only to return an explicit unsupported-protocol error"
        ),
    )
    parser.add_argument("--window-ms", type=float)
    parser.add_argument("--stride-ms", type=float)
    parser.add_argument("--history-steps", type=int)
    parser.add_argument("--minimum-events", type=int, default=1)
    parser.add_argument(
        "--minimum-valid-pixels",
        type=int,
        default=DEFAULT_MINIMUM_VALID_PIXELS,
    )
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument(
        "--save-visualizations",
        type=int,
        default=0,
        metavar="N",
        help=(
            "export the deterministic first N metric-eligible evaluation "
            "samples as bounded NPZ files"
        ),
    )
    parser.add_argument(
        "--visualization-dir",
        type=Path,
        help="snapshot directory (default: OUTPUT_DIR/visualizations)",
    )
    parser.add_argument(
        "--visualization-max-events",
        type=int,
        default=200_000,
        help="deterministic per-snapshot raw-event cap for IWE rendering",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--seed", type=int, default=0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen-head MVSEC optical-flow probes and CMax evaluation."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    probe = commands.add_parser(
        "probe",
        help="train RecurrentTokenFlowHead on outdoor_day2 and evaluate outdoor_day1",
    )
    _add_common_arguments(probe)
    probe.add_argument("--train-manifest", type=Path, required=True)
    probe.add_argument("--train-split", default="train")
    probe.add_argument("--head-init", choices=("random", "cmax"), default="random")
    probe.add_argument("--epochs", type=int, default=30)
    probe.add_argument("--learning-rate", type=float, default=1e-3)
    probe.add_argument("--weight-decay", type=float, default=1e-4)
    probe.add_argument("--max-train-samples", type=int, default=0)

    cmax_eval = commands.add_parser(
        "cmax-eval",
        help="evaluate the checkpoint's learned CMax flow head without fitting",
    )
    _add_common_arguments(cmax_eval)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "probe":
        return _run_probe(args)
    if args.command == "cmax-eval":
        return _run_cmax_eval(args)
    raise ValueError(f"unknown MVSEC flow command: {args.command}")


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()


__all__ = [
    "CANONICAL_RANDOM_PROBE_FLOW_SCALE",
    "CANONICAL_RANDOM_PROBE_HEAD_DEPTH",
    "CANONICAL_RANDOM_PROBE_HIDDEN_DIM",
    "CANONICAL_RANDOM_PROBE_MAX_DISPLACEMENT",
    "DEFAULT_MINIMUM_VALID_PIXELS",
    "DT1_FLOW_HORIZON_US",
    "FlowMetricAccumulator",
    "FlowProtocol",
    "MVSEC_MODEL_IMAGE_SIZE",
    "build_parser",
    "main",
    "reference_set_sha256",
    "resolve_flow_protocol",
    "run",
]
