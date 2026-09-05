from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import tempfile
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from event_window_jepa.data.event_store import H5EventStore
from event_window_jepa.downstream.mvsec_geometry import (
    Alignment,
    MVSECGeometryDataset,
    MVSECGeometrySource,
    MVSECTargetReference,
    accumulate_metric_sums,
    build_mvsec_target_references,
    dense_patch_prediction,
    depth_metric_sums,
    extract_frozen_mvsec_tokens,
    finalize_depth_metrics,
    read_mvsec_geometry_sources,
    representation_from_config,
    split_mvsec_temporal_dev_references,
)
from event_window_jepa.models.recurrent_vjepa21_event_vit import (
    RecurrentVJEPA21EventVisionTransformer,
)
from event_window_jepa.preprocessing.mvsec_labels import MVSEC_OFFICIAL_GT_ARTIFACTS
from event_window_jepa.train.checkpoint import config_hash, load_pretrained_model


MVSEC_DEPTH_IMAGE_SIZE = (272, 352)
MVSEC_DEPTH_MIN_METRIC_PIXELS = 10
MVSEC_DEPTH_CUTOFFS_METERS = (10.0, 20.0, 30.0)
_HASH_CHUNK_BYTES = 8 * 1024 * 1024
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_TRAIN_RECORDING = "outdoor_day2"
_EVAL_RECORDINGS = ("outdoor_day1", "outdoor_night1")
_KNOWN_RECORDINGS = (_TRAIN_RECORDING, *_EVAL_RECORDINGS)
_DepthIdentityCache = dict[
    tuple[Path, int, int, int, int, int], dict[str, Any]
]


def _stat_identity(stat_result: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
        stat_result.st_ctime_ns,
    )


def _sha256_file(path: Path) -> str:
    """Hash one stable file, failing if it changes during the read."""

    resolved = path.expanduser().resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        before = os.fstat(handle.fileno())
        while block := handle.read(_HASH_CHUNK_BYTES):
            digest.update(block)
        after_handle = os.fstat(handle.fileno())
    after_path = resolved.stat()
    if (
        _stat_identity(after_handle) != _stat_identity(before)
        or _stat_identity(after_path) != _stat_identity(after_handle)
    ):
        raise RuntimeError(f"file changed while hashing: {resolved}")
    return digest.hexdigest()


def _stable_file_identity(path: Path) -> dict[str, int | str]:
    """Return a content identity whose path target stayed fixed while hashing."""

    resolved = path.expanduser().resolve()
    before = resolved.stat()
    digest = _sha256_file(resolved)
    after = resolved.stat()
    if _stat_identity(after) != _stat_identity(before):
        raise RuntimeError(f"file changed while recording identity: {resolved}")
    return {
        "device": after.st_dev,
        "inode": after.st_ino,
        "bytes": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "ctime_ns": after.st_ctime_ns,
        "sha256": digest,
    }


def _require_unchanged_content_identity(
    path: Path,
    expected: Mapping[str, int | str],
    *,
    label: str,
) -> None:
    if _stable_file_identity(path) != dict(expected):
        raise RuntimeError(f"{label} changed while the depth probe was active: {path}")


def _require_unchanged_stat_identity(
    path: Path,
    expected: Mapping[str, int | str],
    *,
    label: str,
) -> None:
    current = _stat_identity(path.expanduser().resolve().stat())
    recorded = (
        int(expected["device"]),
        int(expected["inode"]),
        int(expected["bytes"]),
        int(expected["mtime_ns"]),
        int(expected["ctime_ns"]),
    )
    if current != recorded:
        raise RuntimeError(f"{label} changed while the depth probe was active: {path}")


def _verified_download_identity(
    path: Path,
    *,
    size_bytes: int,
    mtime_ns: int,
) -> dict[str, Any] | None:
    """Return a matching downloader sidecar identity, otherwise ``None``."""

    sidecar = path.with_name(path.name + ".verified.json")
    if not sidecar.is_file():
        return None
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    official = MVSEC_OFFICIAL_GT_ARTIFACTS.get(path.name)
    if official is None:
        return None
    official_file_id, official_expected_bytes = official
    file_id = payload.get("file_id")
    expected_bytes = payload.get("expected_bytes")
    sha256 = payload.get("sha256")
    required_matches = {
        "metadata_version": 1,
        "status": "verified",
        "filename": path.name,
        "kind": "gt_hdf5",
        "file_id": official_file_id,
        "expected_bytes": official_expected_bytes,
        "size_bytes": size_bytes,
        "mtime_ns": mtime_ns,
        "publisher_checksum_available": False,
    }
    if any(payload.get(key) != value for key, value in required_matches.items()):
        return None
    if (
        not isinstance(file_id, str)
        or file_id != official_file_id
        or isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes != official_expected_bytes
        or expected_bytes != size_bytes
        or not isinstance(sha256, str)
        or _SHA256_PATTERN.fullmatch(sha256) is None
    ):
        return None
    return {
        "verified_sidecar": str(sidecar.resolve()),
        "file_id": file_id,
        "expected_bytes": expected_bytes,
        "sha256": sha256,
    }


def _depth_target_identity(
    path: Path,
    cache: _DepthIdentityCache,
) -> dict[str, Any]:
    """Bind a raw depth HDF5 to a local SHA, reusing a valid download sidecar."""

    resolved = path.expanduser().resolve()
    stat_result = resolved.stat()
    cache_key = (resolved, *_stat_identity(stat_result))
    cached = cache.get(cache_key)
    if cached is not None:
        return dict(cached)
    sidecar = _verified_download_identity(
        resolved,
        size_bytes=stat_result.st_size,
        mtime_ns=stat_result.st_mtime_ns,
    )
    if sidecar is None:
        digest = _sha256_file(resolved)
        origin = "streamed_at_probe_start"
        source_metadata: dict[str, Any] = {}
    else:
        digest = str(sidecar["sha256"])
        origin = "download_verified_sidecar"
        source_metadata = {
            "verified_sidecar": sidecar["verified_sidecar"],
            "file_id": sidecar["file_id"],
            "expected_bytes": sidecar["expected_bytes"],
        }
    after = resolved.stat()
    if _stat_identity(after) != _stat_identity(stat_result):
        raise RuntimeError(f"depth target changed while recording identity: {resolved}")
    identity = {
        "path": str(resolved),
        "format": "mvsec_depth_hdf5",
        "bytes": stat_result.st_size,
        "mtime_ns": stat_result.st_mtime_ns,
        "sha256": digest,
        "sha256_origin": origin,
        **source_metadata,
    }
    cache[cache_key] = identity
    return dict(identity)


def _require_unchanged_depth_targets(
    expected: Mapping[Path, Mapping[str, Any]],
    cache: _DepthIdentityCache,
) -> None:
    for path, identity in expected.items():
        if _depth_target_identity(path, cache) != dict(identity):
            raise RuntimeError(
                f"depth target changed while the probe was active: {path}"
            )


class MVSECLogDepthHead(nn.Module):
    """Small patch decoder for absolute metric log-depth."""

    def __init__(
        self,
        embed_dim: int,
        *,
        hidden_dim: int = 128,
        initial_depth_m: float = 10.0,
    ) -> None:
        super().__init__()
        if isinstance(embed_dim, bool) or not isinstance(embed_dim, int):
            raise TypeError("embed_dim must be an integer")
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int):
            raise TypeError("hidden_dim must be an integer")
        if embed_dim <= 0 or hidden_dim <= 0:
            raise ValueError("depth-head dimensions must be positive")
        if not math.isfinite(initial_depth_m) or initial_depth_m <= 0:
            raise ValueError("initial_depth_m must be finite and positive")
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.normalization = nn.LayerNorm(self.embed_dim)
        self.spatial = nn.Conv2d(self.embed_dim, self.hidden_dim, 3, padding=1)
        self.activation = nn.GELU()
        self.log_depth = nn.Conv2d(self.hidden_dim, 1, 1)
        nn.init.trunc_normal_(self.spatial.weight, std=0.02)
        nn.init.zeros_(self.spatial.bias)
        nn.init.trunc_normal_(self.log_depth.weight, std=0.02)
        nn.init.constant_(self.log_depth.bias, math.log(initial_depth_m))

    def forward(
        self,
        tokens: torch.Tensor,
        grid_size: tuple[int, int],
    ) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError("tokens must have shape [B,N,D]")
        if not tokens.is_floating_point():
            raise TypeError("tokens must be floating point")
        if tokens.shape[2] != self.embed_dim:
            raise ValueError(
                f"expected token dimension {self.embed_dim}, got {tokens.shape[2]}"
            )
        if (
            not isinstance(grid_size, tuple)
            or len(grid_size) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in grid_size)
        ):
            raise TypeError("grid_size must be a pair of integers")
        grid_height, grid_width = grid_size
        if grid_height <= 0 or grid_width <= 0:
            raise ValueError("grid_size entries must be positive")
        if tokens.shape[1] != grid_height * grid_width:
            raise ValueError("token count does not match grid_size")
        features = self.normalization(tokens)
        features = features.transpose(1, 2).reshape(
            tokens.shape[0], self.embed_dim, grid_height, grid_width
        )
        return self.log_depth(self.activation(self.spatial(features)))


def masked_log_depth_smooth_l1(
    predicted_log_depth: torch.Tensor,
    target_depth: torch.Tensor,
    valid: torch.Tensor,
    *,
    beta: float,
    minimum_valid_pixels: int = MVSEC_DEPTH_MIN_METRIC_PIXELS,
) -> tuple[torch.Tensor | None, int, int]:
    """Return a pixel-weighted loss after dropping samples with too little GT."""

    if predicted_log_depth.ndim == 4 and predicted_log_depth.shape[1] == 1:
        predicted_log_depth = predicted_log_depth[:, 0]
    if predicted_log_depth.ndim != 3:
        raise ValueError("predicted_log_depth must have shape [B,H,W] or [B,1,H,W]")
    if target_depth.shape != predicted_log_depth.shape or valid.shape != target_depth.shape:
        raise ValueError("prediction, target, and valid mask must share shape [B,H,W]")
    if valid.dtype != torch.bool:
        raise TypeError("valid must be a boolean tensor")
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    if minimum_valid_pixels <= 0:
        raise ValueError("minimum_valid_pixels must be positive")
    valid_counts = valid.flatten(1).sum(dim=1)
    eligible = valid_counts >= minimum_valid_pixels
    eligible_count = int(eligible.sum().item())
    if eligible_count == 0:
        return None, 0, 0
    loss_mask = valid & eligible[:, None, None]
    target_values = target_depth[loss_mask]
    if bool(torch.any(~torch.isfinite(target_values))) or bool(
        torch.any(target_values <= 0)
    ):
        raise ValueError("valid target depths must be finite and positive")
    predicted_values = predicted_log_depth[loss_mask]
    if bool(torch.any(~torch.isfinite(predicted_values))):
        raise ValueError("valid predicted log-depth values must be finite")
    loss = functional.smooth_l1_loss(
        predicted_values,
        torch.log(target_values),
        beta=beta,
        reduction="mean",
    )
    return loss, eligible_count, int(loss_mask.sum().item())


@dataclass
class DepthEvaluationAccumulator:
    minimum_valid_pixels: int = MVSEC_DEPTH_MIN_METRIC_PIXELS
    pixel_sums: dict[str, float] = field(default_factory=dict)
    sample_metric_sums: dict[str, float] = field(default_factory=dict)
    cutoff_absolute_error_sums: dict[float, float] = field(default_factory=dict)
    cutoff_pixel_counts: dict[float, int] = field(default_factory=dict)
    cutoff_sample_mae_sums: dict[float, float] = field(default_factory=dict)
    cutoff_sample_counts: dict[float, int] = field(default_factory=dict)
    evaluated_samples: int = 0
    skipped_samples: int = 0

    def __post_init__(self) -> None:
        if self.minimum_valid_pixels <= 0:
            raise ValueError("minimum_valid_pixels must be positive")
        for cutoff in MVSEC_DEPTH_CUTOFFS_METERS:
            self.cutoff_absolute_error_sums.setdefault(cutoff, 0.0)
            self.cutoff_pixel_counts.setdefault(cutoff, 0)
            self.cutoff_sample_mae_sums.setdefault(cutoff, 0.0)
            self.cutoff_sample_counts.setdefault(cutoff, 0)

    def update(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        valid: np.ndarray,
    ) -> bool:
        prediction = np.asarray(prediction, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        valid = np.asarray(valid, dtype=np.bool_)
        if prediction.shape != target.shape or target.shape != valid.shape:
            raise ValueError("depth prediction, target, and mask must share shape [H,W]")
        if prediction.ndim != 2:
            raise ValueError("each depth sample must have shape [H,W]")
        if int(np.count_nonzero(valid)) < self.minimum_valid_pixels:
            self.skipped_samples += 1
            return False
        sample_sums = depth_metric_sums(prediction, target, valid)
        accumulate_metric_sums(self.pixel_sums, sample_sums)
        accumulate_metric_sums(
            self.sample_metric_sums, finalize_depth_metrics(sample_sums)
        )
        absolute_error = np.abs(prediction - target)
        for cutoff in MVSEC_DEPTH_CUTOFFS_METERS:
            cutoff_mask = valid & (target < cutoff)
            count = int(np.count_nonzero(cutoff_mask))
            if count == 0:
                continue
            error_sum = float(np.sum(absolute_error[cutoff_mask]))
            self.cutoff_absolute_error_sums[cutoff] += error_sum
            self.cutoff_pixel_counts[cutoff] += count
            self.cutoff_sample_mae_sums[cutoff] += error_sum / count
            self.cutoff_sample_counts[cutoff] += 1
        self.evaluated_samples += 1
        return True

    def merge(self, other: DepthEvaluationAccumulator) -> None:
        if self.minimum_valid_pixels != other.minimum_valid_pixels:
            raise ValueError("cannot merge depth accumulators with different thresholds")
        accumulate_metric_sums(self.pixel_sums, other.pixel_sums)
        accumulate_metric_sums(self.sample_metric_sums, other.sample_metric_sums)
        for cutoff in MVSEC_DEPTH_CUTOFFS_METERS:
            self.cutoff_absolute_error_sums[cutoff] += (
                other.cutoff_absolute_error_sums[cutoff]
            )
            self.cutoff_pixel_counts[cutoff] += other.cutoff_pixel_counts[cutoff]
            self.cutoff_sample_mae_sums[cutoff] += other.cutoff_sample_mae_sums[cutoff]
            self.cutoff_sample_counts[cutoff] += other.cutoff_sample_counts[cutoff]
        self.evaluated_samples += other.evaluated_samples
        self.skipped_samples += other.skipped_samples

    def finalize(self) -> dict[str, Any]:
        if self.evaluated_samples <= 0:
            raise ValueError("no depth samples have at least ten valid pixels")
        pixel_metrics = finalize_depth_metrics(self.pixel_sums)
        sample_metrics = {
            name: value / self.evaluated_samples
            for name, value in self.sample_metric_sums.items()
        }
        for cutoff in MVSEC_DEPTH_CUTOFFS_METERS:
            name = f"MAE_depth_lt_{int(cutoff)}m"
            pixel_count = self.cutoff_pixel_counts[cutoff]
            sample_count = self.cutoff_sample_counts[cutoff]
            pixel_metrics[name] = (
                self.cutoff_absolute_error_sums[cutoff] / pixel_count
                if pixel_count
                else None
            )
            sample_metrics[name] = (
                self.cutoff_sample_mae_sums[cutoff] / sample_count
                if sample_count
                else None
            )
        pixel_metrics["valid_pixels"] = int(self.pixel_sums["valid_pixels"])
        sample_metrics["evaluated_samples"] = self.evaluated_samples
        return {
            "pixel_average": pixel_metrics,
            "sample_average": sample_metrics,
            "samples_seen": self.evaluated_samples + self.skipped_samples,
            "minimum_valid_pixels_per_sample": self.minimum_valid_pixels,
            "skipped_samples_valid_lt_10": self.skipped_samples,
            "cutoff_pixel_counts": {
                f"depth_lt_{int(cutoff)}m": self.cutoff_pixel_counts[cutoff]
                for cutoff in MVSEC_DEPTH_CUTOFFS_METERS
            },
            "cutoff_sample_counts": {
                f"depth_lt_{int(cutoff)}m": self.cutoff_sample_counts[cutoff]
                for cutoff in MVSEC_DEPTH_CUTOFFS_METERS
            },
        }


@dataclass(frozen=True)
class _DepthSplit:
    role: str
    recording: str
    manifest: Path
    sources: tuple[MVSECGeometrySource, ...]
    references: tuple[MVSECTargetReference, ...]
    dataset: MVSECGeometryDataset


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a frozen-recurrent-backbone MVSEC absolute-depth probe on "
            "outdoor_day2 and evaluate outdoor_day1 (optionally outdoor_night1)."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--protocol-stage",
        choices=("dev", "final"),
        default="final",
        help=(
            "dev uses a guarded early/late split of outdoor_day2; final keeps "
            "outdoor_day1 as the sealed cross-recording test"
        ),
    )
    parser.add_argument(
        "--dev-fraction",
        type=float,
        default=0.2,
        help="late outdoor_day2 depth-label fraction reserved for development",
    )
    parser.add_argument(
        "--dev-guard-ms",
        type=float,
        help=(
            "label-time guard for dev; default auto is the complete recurrent "
            "history span plus centered-alignment rounding allowance"
        ),
    )
    parser.add_argument(
        "--alignment",
        choices=("causal", "f3_centered"),
        default="causal",
        help=(
            "causal ends at each depth timestamp; f3_centered explicitly uses "
            "the future half of the centered 50-ms-style event interval"
        ),
    )
    parser.add_argument(
        "--history-steps",
        type=int,
        help=(
            "fixed recurrent windows per target; default uses checkpoint "
            "burn_in_steps + sequence_length"
        ),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--smooth-l1-beta", type=float, default=0.1)
    parser.add_argument("--minimum-events", type=int, default=1)
    parser.add_argument("--max-train-targets", type=int, default=0)
    parser.add_argument("--max-eval-targets", type=int, default=0)
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
        help="deterministic per-snapshot raw-event cap",
    )
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "hidden_dim": args.hidden_dim,
        "smooth_l1_beta": args.smooth_l1_beta,
        "min_depth": args.min_depth,
        "max_depth": args.max_depth,
    }
    if any(not math.isfinite(float(value)) or value <= 0 for value in positive.values()):
        raise ValueError(f"arguments must be finite and positive: {positive}")
    if args.max_depth <= args.min_depth:
        raise ValueError("max_depth must exceed min_depth")
    if args.workers < 0 or args.minimum_events < 0:
        raise ValueError("workers and minimum_events cannot be negative")
    if args.history_steps is not None and args.history_steps <= 0:
        raise ValueError("history_steps must be positive when provided")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("weight_decay must be finite and non-negative")
    if (
        args.max_train_targets < 0
        or args.max_eval_targets < 0
        or args.save_visualizations < 0
    ):
        raise ValueError("target limits cannot be negative")
    if args.visualization_max_events <= 0:
        raise ValueError("visualization_max_events must be positive")
    if args.visualization_dir is not None and args.save_visualizations == 0:
        raise ValueError("--visualization-dir requires --save-visualizations > 0")
    if args.alignment not in {"causal", "f3_centered"}:
        raise ValueError("alignment must be causal or f3_centered")
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


def _requested_dev_guard_us(args: argparse.Namespace) -> int | None:
    if args.dev_guard_ms is None:
        return None
    return math.ceil(float(args.dev_guard_ms) * 1_000.0)


def _recording_name(sequence_id: str) -> str:
    components = sequence_id.split("__")
    matches = [name for name in _KNOWN_RECORDINGS if name in components]
    if len(matches) != 1:
        raise ValueError(f"cannot identify the MVSEC recording from {sequence_id!r}")
    return matches[0]


def _require_recordings(
    sources: Sequence[MVSECGeometrySource],
    *,
    allowed: Sequence[str],
    role: str,
) -> dict[str, tuple[MVSECGeometrySource, ...]]:
    allowed_set = set(allowed)
    grouped: dict[str, list[MVSECGeometrySource]] = {}
    sequence_ids: set[str] = set()
    for source in sources:
        if source.sequence_id in sequence_ids:
            raise ValueError(f"duplicate {role} sequence_id: {source.sequence_id}")
        sequence_ids.add(source.sequence_id)
        recording = _recording_name(source.sequence_id)
        if source.camera != "left" or source.target_dataset != "/davis/left/depth_image_raw" or (
            source.timestamp_dataset != "/davis/left/depth_image_raw_ts"
        ):
            raise ValueError(
                f"{role} source {source.sequence_id} does not use left raw/distorted depth"
            )
        if recording not in allowed_set:
            raise ValueError(
                f"{role} manifest contains disallowed recording {recording}; "
                f"expected one of {sorted(allowed_set)}"
            )
        grouped.setdefault(recording, []).append(source)
    if not grouped:
        raise ValueError(f"{role} manifest has no permitted MVSEC depth source")
    return {name: tuple(values) for name, values in grouped.items()}


def _target_reference_sha256(
    sources: Sequence[MVSECGeometrySource],
    references: Sequence[MVSECTargetReference],
) -> str:
    digest = hashlib.sha256()
    for reference in references:
        source = sources[reference.source_index]
        digest.update(source.sequence_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(reference.target_index).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(reference.label_timestamp_us).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _alignment_metadata(
    alignment: Alignment,
    references: Sequence[MVSECTargetReference],
) -> dict[str, Any]:
    future_use_us = np.asarray(
        [
            max(0, reference.event_window_end_us - reference.label_timestamp_us)
            for reference in references
        ],
        dtype=np.int64,
    )
    return {
        "alignment": alignment,
        "causal": bool(alignment == "causal" and not np.any(future_use_us)),
        "uses_future_events": bool(np.any(future_use_us > 0)),
        "future_event_use_us": {
            "minimum": int(future_use_us.min()) if len(future_use_us) else 0,
            "maximum": int(future_use_us.max()) if len(future_use_us) else 0,
            "mean": float(future_use_us.mean()) if len(future_use_us) else 0.0,
        },
    }


def _build_references(
    manifest: Path,
    sources: Sequence[MVSECGeometrySource],
    *,
    window_us: int,
    stride_us: int,
    history_steps: int,
    alignment: Alignment,
    minimum_events: int,
    maximum_samples: int,
) -> tuple[MVSECTargetReference, ...]:
    store = H5EventStore(manifest)
    try:
        return build_mvsec_target_references(
            store,
            sources,
            kind="depth",
            window_us=window_us,
            stride_us=stride_us,
            history_steps=history_steps,
            alignment=alignment,
            minimum_events=minimum_events,
            maximum_samples=maximum_samples,
        )
    finally:
        store.close()


def _make_split(
    *,
    role: str,
    recording: str,
    manifest: Path,
    sources: tuple[MVSECGeometrySource, ...],
    window_us: int,
    stride_us: int,
    history_steps: int,
    alignment: Alignment,
    minimum_events: int,
    maximum_samples: int,
    image_size: tuple[int, int],
    representation: Any,
    min_depth: float,
    max_depth: float,
) -> _DepthSplit:
    references = _build_references(
        manifest,
        sources,
        window_us=window_us,
        stride_us=stride_us,
        history_steps=history_steps,
        alignment=alignment,
        minimum_events=minimum_events,
        maximum_samples=maximum_samples,
    )
    return _make_split_from_references(
        role=role,
        recording=recording,
        manifest=manifest,
        sources=sources,
        references=references,
        window_us=window_us,
        stride_us=stride_us,
        history_steps=history_steps,
        image_size=image_size,
        representation=representation,
        min_depth=min_depth,
        max_depth=max_depth,
    )


def _make_split_from_references(
    *,
    role: str,
    recording: str,
    manifest: Path,
    sources: tuple[MVSECGeometrySource, ...],
    references: Sequence[MVSECTargetReference],
    window_us: int,
    stride_us: int,
    history_steps: int,
    image_size: tuple[int, int],
    representation: Any,
    min_depth: float,
    max_depth: float,
) -> _DepthSplit:
    selected_references = tuple(references)
    if not selected_references:
        raise ValueError("MVSEC depth split cannot be empty")
    dataset = MVSECGeometryDataset(
        manifest,
        sources,
        selected_references,
        kind="depth",
        image_size=image_size,
        window_us=window_us,
        stride_us=stride_us,
        history_steps=history_steps,
        representation=representation,
        min_depth=min_depth,
        max_depth=max_depth,
    )
    return _DepthSplit(
        role,
        recording,
        manifest,
        sources,
        selected_references,
        dataset,
    )


def _autocast_context(device: torch.device, precision: str) -> Any:
    if precision == "fp32":
        return nullcontext()
    if device.type in {"cpu", "cuda"}:
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    raise ValueError("bf16 encoding is supported only on CPU or CUDA")


def _loader(
    split: _DepthSplit,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
) -> DataLoader[Mapping[str, torch.Tensor]]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        split.dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
        generator=generator,
    )


def _predict_log_depth(
    model: Any,
    head: MVSECLogDepthHead,
    x: torch.Tensor,
    *,
    duration_ms: float,
    grid_size: tuple[int, int],
    image_size: tuple[int, int],
    device: torch.device,
    precision: str,
) -> torch.Tensor:
    with _autocast_context(device, precision):
        tokens = extract_frozen_mvsec_tokens(model, x, duration_ms=duration_ms)
    patch_log_depth = head(tokens.float(), grid_size)
    return dense_patch_prediction(patch_log_depth, image_size)[:, 0]


def _train_one_epoch(
    model: Any,
    head: MVSECLogDepthHead,
    loader: DataLoader[Mapping[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    *,
    duration_ms: float,
    grid_size: tuple[int, int],
    image_size: tuple[int, int],
    beta: float,
    device: torch.device,
    precision: str,
    epoch: int,
) -> dict[str, float | int]:
    model.eval()
    head.train()
    weighted_loss = 0.0
    optimized_pixels = 0
    optimized_samples = 0
    skipped_samples = 0
    skipped_batches = 0
    progress = tqdm(loader, desc=f"depth train {epoch}", leave=False)
    for batch in progress:
        x = batch["x"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)
        predicted_log_depth = _predict_log_depth(
            model,
            head,
            x,
            duration_ms=duration_ms,
            grid_size=grid_size,
            image_size=image_size,
            device=device,
            precision=precision,
        )
        loss, eligible_samples, valid_pixels = masked_log_depth_smooth_l1(
            predicted_log_depth,
            target,
            valid,
            beta=beta,
        )
        if loss is None:
            skipped_samples += len(x)
            skipped_batches += 1
            continue
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        weighted_loss += float(loss.detach().item()) * valid_pixels
        optimized_pixels += valid_pixels
        optimized_samples += eligible_samples
        skipped_samples += len(x) - eligible_samples
        progress.set_postfix(loss=f"{weighted_loss / optimized_pixels:.5f}")
    if optimized_pixels == 0:
        raise ValueError("no training sample has at least ten valid depth pixels")
    return {
        "valid_log_depth_smooth_l1": weighted_loss / optimized_pixels,
        "optimized_pixels": optimized_pixels,
        "optimized_samples": optimized_samples,
        "skipped_samples_valid_lt_10": skipped_samples,
        "skipped_batches": skipped_batches,
    }


@torch.no_grad()
def _evaluate(
    model: Any,
    head: MVSECLogDepthHead,
    loader: DataLoader[Mapping[str, torch.Tensor]],
    *,
    duration_ms: float,
    grid_size: tuple[int, int],
    image_size: tuple[int, int],
    device: torch.device,
    precision: str,
    description: str,
    save_visualizations: int = 0,
    visualization_dir: Path | None = None,
    visualization_start_index: int = 0,
    visualization_max_events: int = 200_000,
    visualization_context: Mapping[str, Any] | None = None,
) -> tuple[DepthEvaluationAccumulator, list[dict[str, Any]]]:
    model.eval()
    head.eval()
    accumulator = DepthEvaluationAccumulator()
    visualization_entries: list[dict[str, Any]] = []
    sample_cursor = 0
    dataset = loader.dataset
    if not isinstance(dataset, MVSECGeometryDataset):
        raise TypeError("MVSEC depth visualization requires MVSECGeometryDataset")
    for batch in tqdm(loader, desc=description, leave=False):
        x = batch["x"].to(device, non_blocking=True)
        predicted_log_depth = _predict_log_depth(
            model,
            head,
            x,
            duration_ms=duration_ms,
            grid_size=grid_size,
            image_size=image_size,
            device=device,
            precision=precision,
        )
        if not bool(torch.isfinite(predicted_log_depth).all()):
            raise FloatingPointError("depth head produced non-finite log-depth")
        prediction_tensor = torch.exp(predicted_log_depth.float())
        if not bool(
            torch.isfinite(prediction_tensor).all() & (prediction_tensor > 0).all()
        ):
            raise FloatingPointError("depth head produced invalid metric depth")
        prediction = prediction_tensor.cpu().numpy()
        target = batch["target"].numpy()
        valid = batch["valid"].numpy()
        for batch_index, (sample_prediction, sample_target, sample_valid) in enumerate(
            zip(prediction, target, valid, strict=True)
        ):
            included_in_metrics = accumulator.update(
                sample_prediction, sample_target, sample_valid
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

                dataset_index = sample_cursor + batch_index
                reference = dataset.references[dataset_index]
                source = dataset.sources[reference.source_index]
                event_arrays, event_metadata = extract_snapshot_events(
                    dataset,
                    dataset_index,
                    t_end_us=reference.event_window_end_us,
                    duration_us=dataset.window_us,
                    maximum_events=visualization_max_events,
                )
                output_index = visualization_start_index + len(visualization_entries)
                filename = (
                    f"{output_index:06d}_"
                    f"{safe_component(source.sequence_id)}_"
                    f"target-{reference.target_index:06d}.npz"
                )
                entry = write_snapshot(
                    visualization_dir / filename,
                    kind="depth",
                    event_image=batch["x"][batch_index, -1].numpy(),
                    target=sample_target,
                    prediction=sample_prediction,
                    valid=sample_valid,
                    metadata={
                        **dict(visualization_context or {}),
                        **event_metadata,
                        "dataset_index": dataset_index,
                        "sequence_id": source.sequence_id,
                        "target_index": reference.target_index,
                        "label_timestamp_us": reference.label_timestamp_us,
                        "event_representation_window_end_us": reference.event_window_end_us,
                        "event_representation_window_us": dataset.window_us,
                        "depth_unit": "meters",
                        "included_in_aggregate_metrics": True,
                        "valid_pixels": int(np.count_nonzero(sample_valid)),
                        "minimum_valid_pixels": accumulator.minimum_valid_pixels,
                        "visualization_selection": (
                            "first_N_metric_eligible_in_evaluation_order"
                        ),
                    },
                    events=event_arrays,
                )
                entry["dataset_index"] = dataset_index
                entry["visualization_index"] = output_index
                visualization_entries.append(entry)
        sample_cursor += len(prediction)
    return accumulator, visualization_entries


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".partial",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".partial",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _module_state_sha256(module: nn.Module) -> str:
    """Hash tensor names, shapes, dtypes, and values independently of torch.save."""

    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _checkpoint_payload(
    *,
    head: MVSECLogDepthHead,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    fixed_epochs: int,
    encoder_checkpoint: Path,
    encoder_checkpoint_bytes: int,
    encoder_checkpoint_sha256: str,
    encoder_config_hash: str,
    grid_size: tuple[int, int],
    protocol: Mapping[str, Any],
    training_history: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task": "mvsec_frozen_recurrent_absolute_depth_probe",
        "head": head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "fixed_epochs": fixed_epochs,
        "encoder_checkpoint": str(encoder_checkpoint),
        "encoder_checkpoint_bytes": encoder_checkpoint_bytes,
        "encoder_checkpoint_sha256": encoder_checkpoint_sha256,
        "encoder_checkpoint_config_sha256": encoder_config_hash,
        "grid_size": list(grid_size),
        "protocol": dict(protocol),
        "training_history": list(training_history),
        "metrics": None if metrics is None else dict(metrics),
    }


def _split_identity(
    split: _DepthSplit,
    alignment: Alignment,
    *,
    target_identity_cache: _DepthIdentityCache,
) -> dict[str, Any]:
    manifest = split.manifest.expanduser().resolve()
    manifest_identity = _stable_file_identity(manifest)
    target_artifacts: list[dict[str, Any]] = []
    sources_by_target: dict[Path, list[MVSECGeometrySource]] = {}
    for source in split.sources:
        path = source.ground_truth_path.expanduser().resolve()
        sources_by_target.setdefault(path, []).append(source)
    for path, sources in sorted(sources_by_target.items(), key=lambda item: str(item[0])):
        artifact = _depth_target_identity(path, target_identity_cache)
        artifact.update(
            {
                "sequence_ids": sorted(source.sequence_id for source in sources),
                "target_datasets": sorted(
                    {source.target_dataset for source in sources}
                ),
                "timestamp_datasets": sorted(
                    {source.timestamp_dataset for source in sources}
                ),
            }
        )
        target_artifacts.append(artifact)
    return {
        "role": split.role,
        "recording": split.recording,
        "manifest": str(manifest),
        "manifest_bytes": int(manifest_identity["bytes"]),
        "manifest_sha256": str(manifest_identity["sha256"]),
        "sequence_ids": [source.sequence_id for source in split.sources],
        "target_artifacts": target_artifacts,
        "target_count": len(split.references),
        "target_index_timestamp_sha256": _target_reference_sha256(
            split.sources, split.references
        ),
        **_alignment_metadata(alignment, split.references),
    }


def _visualization_output_dir(args: argparse.Namespace) -> Path:
    if args.visualization_dir is None:
        return args.output_dir.expanduser().resolve() / "visualizations"
    return args.visualization_dir.expanduser().resolve()


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is unavailable")
    output_dir = args.output_dir.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    probe_output_paths = {
        output_dir / "protocol.json",
        output_dir / "metrics.json",
        output_dir / "checkpoint-latest.pt",
        output_dir / "checkpoint-final.pt",
    }
    protected_inputs = {
        checkpoint_path,
        args.train_manifest.expanduser().resolve(),
        *(path.expanduser().resolve() for path in args.eval_manifest),
    }
    collisions = sorted(probe_output_paths & protected_inputs, key=str)
    if collisions:
        raise ValueError(
            "probe outputs would overwrite an input file; choose a separate output "
            f"directory: {[str(path) for path in collisions]}"
        )
    if args.save_visualizations:
        visualization_dir = _visualization_output_dir(args)
        if visualization_dir.exists() and not visualization_dir.is_dir():
            raise ValueError("--visualization-dir must be a directory path")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_identity_before = _stable_file_identity(checkpoint_path)
    model, checkpoint_config = load_pretrained_model(checkpoint_path, device=device)
    checkpoint_identity_after = _stable_file_identity(checkpoint_path)
    if checkpoint_identity_after != checkpoint_identity_before:
        raise RuntimeError(
            "encoder checkpoint changed while it was being loaded; stop the writer "
            "or probe an immutable checkpoint copy"
        )
    checkpoint_sha256 = str(checkpoint_identity_before["sha256"])
    checkpoint_bytes = int(checkpoint_identity_before["bytes"])
    if not isinstance(
        model.online_encoder, RecurrentVJEPA21EventVisionTransformer
    ):
        raise ValueError("MVSEC depth probing requires a recurrent encoder checkpoint")
    model.requires_grad_(False)
    model.eval()
    image_size = tuple(checkpoint_config.model.image_size)
    if image_size != MVSEC_DEPTH_IMAGE_SIZE:
        raise ValueError(
            "MVSEC raw-depth protocol requires model.image_size=[272,352] "
            "for centered zero padding of the native 260x346 sensor"
        )
    grid_size = tuple(model.online_encoder.grid_size)
    duration_ms = float(checkpoint_config.recurrent.window_ms)
    window_us = round(duration_ms * 1_000)
    stride_us = round(float(checkpoint_config.recurrent.stride_ms) * 1_000)
    checkpoint_history_steps = int(checkpoint_config.recurrent.burn_in_steps) + int(
        checkpoint_config.recurrent.sequence_length
    )
    history_steps = (
        checkpoint_history_steps
        if args.history_steps is None
        else int(args.history_steps)
    )
    if window_us <= 0 or stride_us <= 0 or history_steps <= 0:
        raise ValueError("checkpoint recurrent cadence is invalid")
    representation = representation_from_config(checkpoint_config)
    alignment: Alignment = args.alignment

    train_manifest = args.train_manifest.expanduser().resolve()
    eval_manifests = tuple(path.expanduser().resolve() for path in args.eval_manifest)
    if args.protocol_stage == "dev" and (
        len(eval_manifests) != 1 or eval_manifests[0] != train_manifest
    ):
        raise ValueError(
            "depth dev protocol requires exactly one --eval-manifest and it must "
            "resolve to the same outdoor_day2 manifest as --train-manifest"
        )
    manifest_identities = {
        manifest: _stable_file_identity(manifest)
        for manifest in {train_manifest, *eval_manifests}
    }
    train_sources = read_mvsec_geometry_sources(
        train_manifest,
        kind="depth",
        split="train" if args.protocol_stage == "dev" else None,
    )
    train_groups = _require_recordings(
        train_sources, allowed=(_TRAIN_RECORDING,), role="training"
    )
    eval_groups: list[tuple[Path, str, tuple[MVSECGeometrySource, ...]]] = []
    if args.protocol_stage == "dev":
        eval_groups.append(
            (train_manifest, _TRAIN_RECORDING, train_groups[_TRAIN_RECORDING])
        )
    else:
        seen_eval_sequence_ids: set[str] = set()
        for manifest in eval_manifests:
            grouped = _require_recordings(
                read_mvsec_geometry_sources(manifest, kind="depth"),
                allowed=_EVAL_RECORDINGS,
                role="evaluation",
            )
            for recording, sources in grouped.items():
                duplicate = seen_eval_sequence_ids.intersection(
                    source.sequence_id for source in sources
                )
                if duplicate:
                    raise ValueError(
                        f"evaluation manifests repeat sequence ids: {sorted(duplicate)}"
                    )
                seen_eval_sequence_ids.update(source.sequence_id for source in sources)
                eval_groups.append((manifest, recording, sources))
        if not any(recording == "outdoor_day1" for _, recording, _ in eval_groups):
            raise ValueError(
                "evaluation manifests must include outdoor_day1 left raw depth"
            )
        if seen_eval_sequence_ids.intersection(
            source.sequence_id for source in train_sources
        ):
            raise ValueError("training and evaluation sequence ids overlap")

    target_identity_cache: _DepthIdentityCache = {}
    initial_target_identities: dict[Path, dict[str, Any]] = {}
    for source in (
        *train_sources,
        *(source for _, _, sources in eval_groups for source in sources),
    ):
        target_path = source.ground_truth_path.expanduser().resolve()
        if target_path not in initial_target_identities:
            initial_target_identities[target_path] = _depth_target_identity(
                target_path, target_identity_cache
            )

    splits: list[_DepthSplit] = []
    try:
        temporal_dev_split: dict[str, Any] | None = None
        eval_splits: list[_DepthSplit] = []
        if args.protocol_stage == "dev":
            day2_sources = train_groups[_TRAIN_RECORDING]
            all_day2_references = _build_references(
                train_manifest,
                day2_sources,
                window_us=window_us,
                stride_us=stride_us,
                history_steps=history_steps,
                alignment=alignment,
                minimum_events=args.minimum_events,
                maximum_samples=0,
            )
            train_references, dev_references, temporal_dev_split = (
                split_mvsec_temporal_dev_references(
                    day2_sources,
                    all_day2_references,
                    window_us=window_us,
                    stride_us=stride_us,
                    history_steps=history_steps,
                    alignment=alignment,
                    dev_fraction=args.dev_fraction,
                    guard_us=_requested_dev_guard_us(args),
                    maximum_train_samples=args.max_train_targets,
                    maximum_dev_samples=args.max_eval_targets,
                )
            )
            temporal_dev_split["manifest_row_split"] = "train"
            train_split = _make_split_from_references(
                role="dev_train",
                recording=_TRAIN_RECORDING,
                manifest=train_manifest,
                sources=day2_sources,
                references=train_references,
                window_us=window_us,
                stride_us=stride_us,
                history_steps=history_steps,
                image_size=image_size,
                representation=representation,
                min_depth=args.min_depth,
                max_depth=args.max_depth,
            )
            dev_split = _make_split_from_references(
                role="dev",
                recording=_TRAIN_RECORDING,
                manifest=train_manifest,
                sources=day2_sources,
                references=dev_references,
                window_us=window_us,
                stride_us=stride_us,
                history_steps=history_steps,
                image_size=image_size,
                representation=representation,
                min_depth=args.min_depth,
                max_depth=args.max_depth,
            )
            splits.extend((train_split, dev_split))
            eval_splits.append(dev_split)
        else:
            train_split = _make_split(
                role="train",
                recording=_TRAIN_RECORDING,
                manifest=train_manifest,
                sources=train_groups[_TRAIN_RECORDING],
                window_us=window_us,
                stride_us=stride_us,
                history_steps=history_steps,
                alignment=alignment,
                minimum_events=args.minimum_events,
                maximum_samples=args.max_train_targets,
                image_size=image_size,
                representation=representation,
                min_depth=args.min_depth,
                max_depth=args.max_depth,
            )
            splits.append(train_split)
            for manifest, recording, sources in eval_groups:
                split = _make_split(
                    role="final_test" if recording == "outdoor_day1" else "ood",
                    recording=recording,
                    manifest=manifest,
                    sources=sources,
                    window_us=window_us,
                    stride_us=stride_us,
                    history_steps=history_steps,
                    alignment=alignment,
                    minimum_events=args.minimum_events,
                    maximum_samples=args.max_eval_targets,
                    image_size=image_size,
                    representation=representation,
                    min_depth=args.min_depth,
                    max_depth=args.max_depth,
                )
                splits.append(split)
                eval_splits.append(split)

        for manifest, identity in manifest_identities.items():
            _require_unchanged_content_identity(
                manifest, identity, label="manifest"
            )
        _require_unchanged_depth_targets(
            initial_target_identities, target_identity_cache
        )
        encoder_config_hash = config_hash(checkpoint_config)
        protocol = {
            "schema_version": 1,
            "name": "frozen_recurrent_raw_metric_depth_v1",
            "stage": args.protocol_stage,
            "task": "mvsec_frozen_recurrent_absolute_depth_probe",
            "encoder_checkpoint": str(checkpoint_path),
            "encoder_checkpoint_bytes": checkpoint_bytes,
            "encoder_checkpoint_sha256": checkpoint_sha256,
            "encoder_checkpoint_config_sha256": encoder_config_hash,
            "backbone": {
                "frozen": True,
                "feature": "final_recurrent_patch_token",
                "history_steps": history_steps,
                "history_policy": (
                    "full_pretraining_clip_ending_at_target"
                    if args.history_steps is None
                    else "fixed_downstream_history_ending_at_target"
                ),
                "history_source": (
                    "checkpoint_burn_in_plus_sequence_length"
                    if args.history_steps is None
                    else "explicit_cli_override"
                ),
                "checkpoint_default_history_steps": checkpoint_history_steps,
                "burn_in_steps": checkpoint_config.recurrent.burn_in_steps,
                "supervised_sequence_steps": checkpoint_config.recurrent.sequence_length,
                "state_reset": "once_per_target_sample",
                "window_us": window_us,
                "stride_us": stride_us,
                "history_span_us": window_us + (history_steps - 1) * stride_us,
            },
            "temporal_dev_split": temporal_dev_split,
            "representation_pretraining_visibility_contract": (
                {
                    "protocol_class": (
                        "inductive_cross_recording_final_evaluation"
                    ),
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
            "head": {
                "order": "LayerNorm-Conv2d-GELU-Conv2d(1ch log-depth)-bilinear",
                "embed_dim": checkpoint_config.model.embed_dim,
                "hidden_dim": args.hidden_dim,
                "initialization": "random_with_log_10m_output_bias",
                "initialization_seed": args.seed,
                "parameter_precision": "fp32",
                "patch_grid": list(grid_size),
            },
            "geometry": {
                "target_dataset": "/davis/left/depth_image_raw",
                "coordinate_frame": "distorted",
                "native_size": [260, 346],
                "model_size": list(image_size),
                "transform": "center zero-pad",
                "padding_valid": False,
            },
            "depth": {
                "validity_m": {
                    "finite": True,
                    "strict_minimum": args.min_depth,
                    "strict_maximum": args.max_depth,
                },
                "training_loss": "valid-pixel log-depth SmoothL1",
                "smooth_l1_beta": args.smooth_l1_beta,
                "prediction_clamp_for_metrics": None,
                "absolute_metric_scale_correction": "none",
                "minimum_valid_pixels_per_sample": MVSEC_DEPTH_MIN_METRIC_PIXELS,
                "metric_variants": {
                    "SqRel": "mean((prediction-target)^2/target)",
                    "SILog": "sqrt(mean(log_error^2)-mean(log_error)^2)",
                    "F3_SqRel": "mean((prediction-target)^2/target^2)",
                    "F3_SILog": "sqrt(mean(log_error^2)-0.5*mean(log_error)^2)",
                },
            },
            "training_policy": {
                "recording": _TRAIN_RECORDING,
                "fixed_epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "seed": args.seed,
                "device": str(device),
                "precision": args.precision,
                "evaluation_used_for_model_selection": (
                    args.protocol_stage == "dev"
                ),
                "model_selection": (
                    "late day2 dev metrics may select an ablation configuration"
                    if args.protocol_stage == "dev"
                    else "none; final fixed-epoch checkpoint"
                ),
            },
            "target_selection": {
                "minimum_events_in_final_window": args.minimum_events,
                "maximum_train_targets": args.max_train_targets,
                "maximum_eval_targets_per_recording": args.max_eval_targets,
            },
            "train_targets": _split_identity(
                train_split,
                alignment,
                target_identity_cache=target_identity_cache,
            ),
            "evaluation_targets": [
                _split_identity(
                    split,
                    alignment,
                    target_identity_cache=target_identity_cache,
                )
                for split in eval_splits
            ],
        }
        _atomic_json(protocol, output_dir / "protocol.json")
        print(
            json.dumps(
                {
                    "status": "ready",
                    "protocol_stage": args.protocol_stage,
                    "train_targets": len(train_split.references),
                    "evaluation_targets": {
                        split.recording: len(split.references) for split in eval_splits
                    },
                    "alignment": alignment,
                    "uses_future_events": any(
                        _alignment_metadata(alignment, split.references)[
                            "uses_future_events"
                        ]
                        for split in splits
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )

        # Reset after checkpoint reconstruction so JEPA-only and JEPA+CMax
        # checkpoints receive identical random probe heads for the same seed.
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        head = MVSECLogDepthHead(
            checkpoint_config.model.embed_dim,
            hidden_dim=args.hidden_dim,
        ).to(device)
        optimizer = torch.optim.AdamW(
            head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        train_loader = _loader(
            train_split,
            batch_size=args.batch_size,
            workers=args.workers,
            shuffle=True,
            seed=args.seed,
            pin_memory=device.type == "cuda",
        )
        training_history: list[dict[str, Any]] = []
        for epoch in range(1, args.epochs + 1):
            epoch_metrics = _train_one_epoch(
                model,
                head,
                train_loader,
                optimizer,
                duration_ms=duration_ms,
                grid_size=grid_size,
                image_size=image_size,
                beta=args.smooth_l1_beta,
                device=device,
                precision=args.precision,
                epoch=epoch,
            )
            epoch_record = {"epoch": epoch, **epoch_metrics}
            training_history.append(epoch_record)
            _atomic_torch_save(
                _checkpoint_payload(
                    head=head,
                    optimizer=optimizer,
                    epoch=epoch,
                    fixed_epochs=args.epochs,
                    encoder_checkpoint=checkpoint_path,
                    encoder_checkpoint_bytes=checkpoint_bytes,
                    encoder_checkpoint_sha256=checkpoint_sha256,
                    encoder_config_hash=encoder_config_hash,
                    grid_size=grid_size,
                    protocol=protocol,
                    training_history=training_history,
                ),
                output_dir / "checkpoint-latest.pt",
            )
            print(json.dumps(epoch_record, sort_keys=True), flush=True)

        per_recording_accumulators: dict[str, DepthEvaluationAccumulator] = {}
        combined = DepthEvaluationAccumulator()
        visualization_entries: list[dict[str, Any]] = []
        head_state_sha256 = _module_state_sha256(head)
        for index, split in enumerate(eval_splits):
            loader = _loader(
                split,
                batch_size=args.batch_size,
                workers=args.workers,
                shuffle=False,
                seed=args.seed + 1 + index,
                pin_memory=device.type == "cuda",
            )
            remaining_visualizations = max(
                0, args.save_visualizations - len(visualization_entries)
            )
            split_protocol = next(
                identity
                for identity in protocol["evaluation_targets"]
                if identity["recording"] == split.recording
                and identity["manifest"] == str(split.manifest.expanduser().resolve())
            )
            current, current_visualizations = _evaluate(
                model,
                head,
                loader,
                duration_ms=duration_ms,
                grid_size=grid_size,
                image_size=image_size,
                device=device,
                precision=args.precision,
                description=f"depth eval {split.recording}",
                save_visualizations=remaining_visualizations,
                visualization_dir=(
                    _visualization_output_dir(args)
                    if args.save_visualizations
                    else None
                ),
                visualization_start_index=len(visualization_entries),
                visualization_max_events=args.visualization_max_events,
                visualization_context={
                    "evaluation_command": "depth-probe",
                    "checkpoint_sha256": checkpoint_sha256,
                    "eval_manifest_sha256": split_protocol["manifest_sha256"],
                    "evaluation_target_reference_sha256": split_protocol[
                        "target_index_timestamp_sha256"
                    ],
                    "target_artifacts": split_protocol["target_artifacts"],
                    "head_state_sha256": head_state_sha256,
                    "head_spec": protocol["head"],
                    "coordinate_frame": "native_distorted_left_DAVIS_center_padded",
                    "native_sensor_height_width": [260, 346],
                    "model_canvas_height_width": list(image_size),
                    "native_sensor_center_padding_yx": [6, 3],
                    "alignment": alignment,
                    "recording": split.recording,
                    "split_role": split.role,
                    "protocol_stage": args.protocol_stage,
                    "seed": args.seed,
                },
            )
            visualization_entries.extend(current_visualizations)
            destination = per_recording_accumulators.setdefault(
                split.recording, DepthEvaluationAccumulator()
            )
            destination.merge(current)
            combined.merge(current)
        visualization_index = None
        if args.save_visualizations:
            from event_window_jepa.downstream.mvsec_visualize import (
                write_snapshot_index,
            )

            visualization_index = write_snapshot_index(
                _visualization_output_dir(args),
                visualization_entries,
                requested=args.save_visualizations,
                context={
                    "evaluation_command": "depth-probe",
                    "checkpoint_sha256": checkpoint_sha256,
                    "head_state_sha256": head_state_sha256,
                    "alignment": alignment,
                    "protocol_stage": args.protocol_stage,
                    "seed": args.seed,
                },
            )
        per_recording_metrics = {
            recording: accumulator.finalize()
            for recording, accumulator in sorted(per_recording_accumulators.items())
        }
        metrics: dict[str, Any] = {
            "protocol_stage": args.protocol_stage,
            "scale_correction": "none",
            "evaluation_used_for_model_selection": args.protocol_stage == "dev",
            "per_recording": per_recording_metrics,
            "combined_diagnostic_only": args.protocol_stage == "final",
            "combined": combined.finalize(),
        }
        if args.protocol_stage == "dev":
            metrics.update(
                {
                    "dev": per_recording_metrics[_TRAIN_RECORDING],
                    "final_test": None,
                    "ood": {},
                    "sealed_final_test_evaluated": False,
                }
            )
        else:
            metrics.update(
                {
                    "dev": None,
                    "final_test": per_recording_metrics["outdoor_day1"],
                    "ood": {
                        recording: result
                        for recording, result in per_recording_metrics.items()
                        if recording == "outdoor_night1"
                    },
                    "sealed_final_test_evaluated": True,
                }
            )
        for manifest, identity in manifest_identities.items():
            _require_unchanged_content_identity(
                manifest, identity, label="manifest"
            )
        _require_unchanged_depth_targets(
            initial_target_identities, target_identity_cache
        )
        _require_unchanged_stat_identity(
            checkpoint_path,
            checkpoint_identity_before,
            label="encoder checkpoint",
        )
        report = {
            "protocol": protocol,
            "head": {"state_sha256": head_state_sha256},
            "training_history": training_history,
            "metrics": metrics,
            "visualizations": visualization_index,
        }
        _atomic_json(report, output_dir / "metrics.json")
        _atomic_torch_save(
            _checkpoint_payload(
                head=head,
                optimizer=optimizer,
                epoch=args.epochs,
                fixed_epochs=args.epochs,
                encoder_checkpoint=checkpoint_path,
                encoder_checkpoint_bytes=checkpoint_bytes,
                encoder_checkpoint_sha256=checkpoint_sha256,
                encoder_config_hash=encoder_config_hash,
                grid_size=grid_size,
                protocol=protocol,
                training_history=training_history,
                metrics=metrics,
            ),
            output_dir / "checkpoint-final.pt",
        )
        print(json.dumps(metrics, sort_keys=True), flush=True)
        return report
    finally:
        for split in splits:
            split.dataset.close()


def main(argv: Sequence[str] | None = None) -> None:
    run(_parse_args(argv))


if __name__ == "__main__":
    main()


__all__ = [
    "DepthEvaluationAccumulator",
    "MVSECLogDepthHead",
    "main",
    "masked_log_depth_smooth_l1",
    "run",
]
