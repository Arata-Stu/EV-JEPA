from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from event_window_jepa.config import ExperimentConfig
from event_window_jepa.data.anchor_sampler import milliseconds_to_microseconds
from event_window_jepa.data.paired_window_dataset import _patch_event_activity
from event_window_jepa.data.recurrent_window_dataset import (
    RecurrentWindowDataset,
    RecurrentWindowDebugSample,
)
from event_window_jepa.train.pretrain import build_dataset


@dataclass(frozen=True)
class FutureFeatureClip:
    """One direct-index clip materialized for causal feature inspection."""

    sample_index: int
    sample: dict[str, Any]
    debug: RecurrentWindowDebugSample


@dataclass(frozen=True)
class FutureFeatureMaterialization:
    """Validated deterministic clips together with their effective data config."""

    config: ExperimentConfig
    dataset: RecurrentWindowDataset
    epoch: int
    clips: tuple[FutureFeatureClip, ...]


def validate_future_feature_config(config: ExperimentConfig) -> None:
    """Reject checkpoints that cannot provide the future-feature data contract."""

    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig")
    if config.optimization.objective != "recurrent_future_jepa":
        raise ValueError(
            "future feature materialization requires objective=recurrent_future_jepa"
        )
    if not config.recurrent.sequence_loader or not config.recurrent.enabled:
        raise ValueError("future feature materialization requires a recurrent sequence loader")
    if config.recurrent.recurrent_placement != "post_encoder":
        raise ValueError("future feature materialization requires post_encoder recurrence")
    if config.recurrent.prediction_horizon_steps < 1:
        raise ValueError("future feature materialization requires positive lookahead")
    if not config.recurrent.return_patch_event_activity:
        raise ValueError("future feature materialization requires patch event activity")


def _config_with_manifest(
    config: ExperimentConfig,
    manifest_override: str | Path | None,
) -> ExperimentConfig:
    if manifest_override is None:
        return config
    if isinstance(manifest_override, str) and not manifest_override.strip():
        raise ValueError("manifest_override cannot be empty")
    manifest = Path(manifest_override).expanduser().resolve()
    return replace(config, data=replace(config.data, manifest=str(manifest)))


def validate_future_feature_dataset(
    config: ExperimentConfig,
    dataset: RecurrentWindowDataset,
) -> None:
    """Validate configuration-dependent properties before reading any samples."""

    validate_future_feature_config(config)
    if not isinstance(dataset, RecurrentWindowDataset):
        raise TypeError("future feature materialization requires RecurrentWindowDataset")
    horizon = config.recurrent.prediction_horizon_steps
    online_steps = config.recurrent.burn_in_steps + config.recurrent.sequence_length
    if dataset.online_steps != online_steps:
        raise ValueError("dataset online length differs from the checkpoint configuration")
    if dataset.lookahead_steps != horizon:
        raise ValueError("dataset lookahead differs from prediction_horizon_steps")
    if dataset.total_steps != online_steps + horizon:
        raise ValueError("dataset sampled length does not include the configured lookahead")
    if not dataset.return_patch_event_activity:
        raise ValueError("dataset must return current and future patch event activity")


def build_future_feature_dataset(
    config: ExperimentConfig,
    *,
    manifest_override: str | Path | None = None,
) -> tuple[ExperimentConfig, RecurrentWindowDataset]:
    """Build the checkpoint-matched recurrent dataset without a batch sampler."""

    validate_future_feature_config(config)
    effective_config = _config_with_manifest(config, manifest_override)
    dataset = build_dataset(effective_config)
    if not isinstance(dataset, RecurrentWindowDataset):
        raise TypeError("checkpoint configuration did not build a recurrent dataset")
    validate_future_feature_dataset(effective_config, dataset)
    return effective_config, dataset


def _tensor(sample: Mapping[str, Any], name: str) -> torch.Tensor:
    value = sample.get(name)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"future feature sample requires tensor field {name!r}")
    return value.detach().cpu()


def _expected_representations(
    dataset: RecurrentWindowDataset,
    debug: RecurrentWindowDebugSample,
) -> torch.Tensor:
    representations = [
        np.ascontiguousarray(dataset.representation(window), dtype=np.float32)
        for window in debug.windows
    ]
    if not representations:
        raise ValueError("future feature debug clip contains no event windows")
    first_shape = representations[0].shape
    if any(value.shape != first_shape for value in representations[1:]):
        raise ValueError("debug event representations do not share one geometry")
    return torch.from_numpy(np.stack(representations, axis=0))


def _expected_patch_activity(
    dataset: RecurrentWindowDataset,
    debug: RecurrentWindowDebugSample,
) -> torch.Tensor:
    grid_size = (
        dataset.mask_generator.grid_height,
        dataset.mask_generator.grid_width,
    )
    activities = [
        np.ascontiguousarray(
            _patch_event_activity(window, grid_size).reshape(-1), dtype=np.int64
        )
        for window in debug.windows
    ]
    return torch.from_numpy(np.stack(activities, axis=0))


def validate_future_feature_sample(
    config: ExperimentConfig,
    dataset: RecurrentWindowDataset,
    sample: Mapping[str, Any],
    debug: RecurrentWindowDebugSample,
) -> None:
    """Validate that one direct sample preserves every future-target alignment."""

    validate_future_feature_dataset(config, dataset)
    if not isinstance(debug, RecurrentWindowDebugSample):
        raise TypeError("debug must be a RecurrentWindowDebugSample")

    online_steps = dataset.online_steps
    horizon = dataset.lookahead_steps
    total_steps = dataset.total_steps
    expected_image_shape = (
        online_steps,
        config.representation.channels,
        *config.model.image_size,
    )
    num_patches = (
        config.model.image_size[0] // config.model.patch_size
    ) * (config.model.image_size[1] // config.model.patch_size)

    x = _tensor(sample, "x")
    x_future = _tensor(sample, "x_future")
    dt_ms = _tensor(sample, "dt_ms")
    future_dt_ms = _tensor(sample, "future_dt_ms")
    t_end_us = _tensor(sample, "t_end_us")
    future_t_end_us = _tensor(sample, "future_t_end_us")
    activity = _tensor(sample, "patch_event_activity")
    future_activity = _tensor(sample, "future_patch_event_activity")

    if tuple(x.shape) != expected_image_shape or tuple(x_future.shape) != expected_image_shape:
        raise ValueError("x and x_future must share checkpoint representation geometry")
    if not bool(torch.isfinite(x).all() and torch.isfinite(x_future).all()):
        raise ValueError("x and x_future must contain only finite values")
    if dt_ms.shape != (online_steps,) or future_dt_ms.shape != (online_steps,):
        raise ValueError("current and future durations must have shape [online_steps]")
    expected_duration = torch.full_like(dt_ms, dataset.clip_sampler.base_window_ms)
    if not torch.equal(dt_ms, expected_duration) or not torch.equal(future_dt_ms, dt_ms):
        raise ValueError("current and future durations do not match the sampled window")
    if t_end_us.shape != (online_steps,) or future_t_end_us.shape != (online_steps,):
        raise ValueError("current and future timestamps must have shape [online_steps]")
    if activity.shape != (online_steps, num_patches) or future_activity.shape != (
        online_steps,
        num_patches,
    ):
        raise ValueError("current and future patch activity must have shape [T,P]")
    if activity.dtype != torch.int64 or future_activity.dtype != torch.int64:
        raise ValueError("current and future patch activity must use int64 counts")

    if len(debug.windows) != total_steps or len(debug.spatial_transforms) != total_steps:
        raise ValueError("debug clip does not contain online plus lookahead windows")
    if len(debug.sequence_ids) != total_steps or len(set(debug.sequence_ids)) != 1:
        raise ValueError("debug clip must remain inside one event sequence")
    if any(params != debug.spatial_transform for params in debug.spatial_transforms):
        raise ValueError("all current and future windows must share one spatial transform")
    if any(
        (window.height, window.width) != config.model.image_size
        for window in debug.windows
    ):
        raise ValueError("transformed event windows do not match checkpoint image geometry")
    if (
        str(sample.get("sequence_id", "")) != debug.clip.sequence_id
        or debug.sequence_info.sequence_id != debug.clip.sequence_id
    ):
        raise ValueError("sample and debug sequence identities differ")

    debug_timestamps = torch.tensor(
        [window.t_end_us for window in debug.windows], dtype=torch.int64
    )
    if tuple(int(value) for value in debug_timestamps) != debug.clip.t_end_us:
        raise ValueError("debug windows and clip timestamps differ")
    expected_current_timestamps = debug_timestamps[:online_steps]
    expected_future_timestamps = debug_timestamps[horizon : horizon + online_steps]
    if not torch.equal(t_end_us, expected_current_timestamps):
        raise ValueError("t_end_us is not aligned to the online debug windows")
    if not torch.equal(future_t_end_us, expected_future_timestamps):
        raise ValueError("future_t_end_us is not aligned to the lookahead debug windows")
    expected_delta_us = horizon * milliseconds_to_microseconds(
        config.recurrent.stride_ms
    )
    if not bool((future_t_end_us - t_end_us == expected_delta_us).all()):
        raise ValueError("future timestamps are not prediction_horizon_steps ahead")

    expected_x = _expected_representations(dataset, debug)
    if not torch.equal(x, expected_x[:online_steps]):
        raise ValueError("x is not aligned to the current debug event windows")
    if not torch.equal(x_future, expected_x[horizon : horizon + online_steps]):
        raise ValueError("x_future is not aligned to the future debug event windows")

    expected_activity = _expected_patch_activity(dataset, debug)
    if not torch.equal(activity, expected_activity[:online_steps]):
        raise ValueError("patch_event_activity is not aligned to current event windows")
    if not torch.equal(
        future_activity,
        expected_activity[horizon : horizon + online_steps],
    ):
        raise ValueError(
            "future_patch_event_activity is not aligned to future event windows"
        )

    if sample.get("sampling_mode") != "random" or str(sample.get("stream_id", "")):
        raise ValueError("feature clips must be selected through direct random indexing")
    state_reset = sample.get("state_reset")
    if not isinstance(state_reset, torch.Tensor) or not bool(state_reset.item()):
        raise ValueError("direct feature clips must reset recurrent state at clip start")


def materialize_future_feature_samples(
    config: ExperimentConfig,
    sample_indices: Iterable[int],
    *,
    manifest_override: str | Path | None = None,
    epoch: int = 0,
) -> FutureFeatureMaterialization:
    """Materialize validated clips by direct index, independent of mixed sampling."""

    if epoch < 0:
        raise ValueError("epoch cannot be negative")
    indices = tuple(sample_indices)
    if not indices:
        raise ValueError("sample_indices cannot be empty")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
        raise TypeError("sample_indices must contain integers")
    if len(set(indices)) != len(indices):
        raise ValueError("sample_indices must be unique")

    effective_config, dataset = build_future_feature_dataset(
        config,
        manifest_override=manifest_override,
    )
    invalid = [index for index in indices if not 0 <= index < len(dataset)]
    if invalid:
        raise ValueError(f"sample indices are outside the dataset: {invalid}")

    dataset.set_epoch(epoch)
    clips: list[FutureFeatureClip] = []
    for sample_index in indices:
        # Integer indexing deliberately bypasses MixedRecurrentBatchSampler even
        # when the checkpoint's training configuration selected mixed streams.
        sample, debug = dataset.sample_with_debug(sample_index)
        validate_future_feature_sample(effective_config, dataset, sample, debug)
        clips.append(FutureFeatureClip(sample_index, sample, debug))
    return FutureFeatureMaterialization(
        config=effective_config,
        dataset=dataset,
        epoch=epoch,
        clips=tuple(clips),
    )


__all__ = [
    "FutureFeatureClip",
    "FutureFeatureMaterialization",
    "build_future_feature_dataset",
    "materialize_future_feature_samples",
    "validate_future_feature_config",
    "validate_future_feature_dataset",
    "validate_future_feature_sample",
]
