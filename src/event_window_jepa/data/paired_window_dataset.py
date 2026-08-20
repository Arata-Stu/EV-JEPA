from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch
from torch.utils.data import Dataset

from event_window_jepa.data.anchor_sampler import (
    UniformTimeAnchorSampler,
    WindowPairSampler,
    deterministic_seed,
    milliseconds_to_microseconds,
)
from event_window_jepa.data.event_store import EventStore
from event_window_jepa.data.spatial_transforms import (
    SharedRandomSpatialTransform,
    SpatialTransformParameters,
)
from event_window_jepa.data.types import EventWindow, SequenceInfo
from event_window_jepa.masks.multiblock import MaskPair, MultiBlockMaskGenerator


class EventRepresentation(Protocol):
    def __call__(self, window: EventWindow) -> np.ndarray: ...


@dataclass(frozen=True)
class PairedWindowDebugSample:
    """Raw objects used to build one training sample for offline inspection."""

    context: EventWindow
    target: EventWindow
    spatial_transform: SpatialTransformParameters
    masks: MaskPair
    sequence_info: SequenceInfo


def _patch_event_activity(
    window: EventWindow, grid_size: tuple[int, int]
) -> np.ndarray:
    """Count cropped context events on the encoder's spatial patch grid."""

    grid_height, grid_width = grid_size
    if (
        min(grid_height, grid_width) <= 0
        or window.height % grid_height
        or window.width % grid_width
    ):
        raise ValueError("event window dimensions must be divisible by the mask grid")
    patch_height = window.height // grid_height
    patch_width = window.width // grid_width
    if window.event_count == 0:
        return np.zeros((grid_height, grid_width), dtype=np.int64)
    rows = window.y.astype(np.int64, copy=False) // patch_height
    columns = window.x.astype(np.int64, copy=False) // patch_width
    if (
        int(rows.min()) < 0
        or int(rows.max()) >= grid_height
        or int(columns.min()) < 0
        or int(columns.max()) >= grid_width
    ):
        raise ValueError("cropped context events exceed the mask grid")
    linear = rows * grid_width + columns
    return np.bincount(
        linear, minlength=grid_height * grid_width
    ).reshape(grid_height, grid_width)


def _masked_activity_metrics(
    activity: np.ndarray, target_mask: np.ndarray
) -> tuple[float, float]:
    flat_activity = np.asarray(activity).reshape(-1)
    flat_mask = np.asarray(target_mask, dtype=np.bool_).reshape(-1)
    if flat_activity.shape != flat_mask.shape or not bool(flat_mask.any()):
        raise ValueError("activity and target mask must share non-empty patch support")
    selected = flat_activity[flat_mask]
    active_patch_ratio = float(np.count_nonzero(selected) / len(selected))
    total_mass = float(flat_activity.sum())
    event_mass_coverage = (
        float(selected.sum()) / total_mass if total_mass > 0 else 0.0
    )
    return active_patch_ratio, event_mass_coverage


class PairedWindowDataset(Dataset[dict[str, Any]]):
    """Creates end-aligned context/target views without target-side leakage."""

    def __init__(
        self,
        store: EventStore,
        anchor_sampler: UniformTimeAnchorSampler,
        pair_sampler: WindowPairSampler,
        representation: EventRepresentation,
        mask_generator: MultiBlockMaskGenerator,
        spatial_transform: SharedRandomSpatialTransform,
        seed: int = 0,
    ) -> None:
        self.store = store
        self.anchor_sampler = anchor_sampler
        self.pair_sampler = pair_sampler
        self.representation = representation
        self.mask_generator = mask_generator
        self.spatial_transform = spatial_transform
        self.seed = seed
        self.epoch = 0
        self._metadata = {info.sequence_id: info for info in store.sequences()}

    def __len__(self) -> int:
        return len(self.anchor_sampler)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch cannot be negative")
        self.epoch = epoch

    def _build_sample(
        self, index: int
    ) -> tuple[dict[str, Any], PairedWindowDebugSample]:
        anchor = self.anchor_sampler.sample(index, self.epoch)
        rng = random.Random(deterministic_seed(self.seed, self.epoch, index, stream=1))
        pair = self.pair_sampler.sample(rng)
        context_duration_us = milliseconds_to_microseconds(pair.context_ms)
        target_duration_us = milliseconds_to_microseconds(pair.target_ms)
        maximum_duration_us = max(context_duration_us, target_duration_us)
        largest_window = self.store.slice(
            anchor.sequence_id,
            anchor.t_end_us,
            maximum_duration_us,
        )

        def tail_window(duration_us: int) -> EventWindow:
            if duration_us == maximum_duration_us:
                return largest_window
            t_start_us = anchor.t_end_us - duration_us
            left = int(np.searchsorted(largest_window.t_us, t_start_us, side="right"))
            return EventWindow(
                x=largest_window.x[left:],
                y=largest_window.y[left:],
                t_us=largest_window.t_us[left:],
                polarity=largest_window.polarity[left:],
                t_start_us=t_start_us,
                t_end_us=anchor.t_end_us,
                height=largest_window.height,
                width=largest_window.width,
            )

        context = tail_window(context_duration_us)
        target = tail_window(target_duration_us)
        if context.t_end_us != target.t_end_us:
            raise RuntimeError("paired windows do not share an end timestamp")

        info = self._metadata[anchor.sequence_id]
        params = self.spatial_transform.sample(rng, info.height, info.width)
        context = self.spatial_transform.apply(context, params)
        target = self.spatial_transform.apply(target, params)
        activity = _patch_event_activity(
            context,
            (self.mask_generator.grid_height, self.mask_generator.grid_width),
        )
        masks = self.mask_generator.sample(rng, activity=activity)
        target_activity = _patch_event_activity(
            target,
            (self.mask_generator.grid_height, self.mask_generator.grid_width),
        )
        target_active_ratio, target_mass_coverage = _masked_activity_metrics(
            target_activity, masks.target
        )

        x_context = np.ascontiguousarray(self.representation(context), dtype=np.float32)
        x_target = np.ascontiguousarray(self.representation(target), dtype=np.float32)
        sample = {
            "x_context": torch.from_numpy(x_context),
            "x_target": torch.from_numpy(x_target),
            "dt_context_ms": torch.tensor(pair.context_ms, dtype=torch.float32),
            "dt_target_ms": torch.tensor(pair.target_ms, dtype=torch.float32),
            "t_end_us": anchor.t_end_us,
            "sequence_id": anchor.sequence_id,
            # True means that a patch is retained/queried, respectively.
            "context_mask": torch.from_numpy(masks.context_keep),
            "target_mask": torch.from_numpy(masks.target),
            "mask_activity_aware": torch.tensor(
                float(masks.activity_aware), dtype=torch.float32
            ),
            "mask_activity_fallback": torch.tensor(
                float(masks.activity_fallback), dtype=torch.float32
            ),
            "mask_context_active_patch_ratio": torch.tensor(
                masks.selection_active_patch_ratio, dtype=torch.float32
            ),
            "mask_context_event_mass_coverage": torch.tensor(
                masks.selection_event_mass_coverage, dtype=torch.float32
            ),
            "mask_target_active_patch_ratio": torch.tensor(
                target_active_ratio, dtype=torch.float32
            ),
            "mask_target_event_mass_coverage": torch.tensor(
                target_mass_coverage, dtype=torch.float32
            ),
            "mask_empty_target": torch.tensor(
                float(target_active_ratio == 0.0),
                dtype=torch.float32,
            ),
        }
        debug = PairedWindowDebugSample(
            context=context,
            target=target,
            spatial_transform=params,
            masks=masks,
            sequence_info=info,
        )
        return sample, debug

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample, _ = self._build_sample(index)
        return sample

    def sample_with_debug(
        self, index: int
    ) -> tuple[dict[str, Any], PairedWindowDebugSample]:
        """Return the exact training sample plus raw windows and sampled geometry."""

        return self._build_sample(index)
