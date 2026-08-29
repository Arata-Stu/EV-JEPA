from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch
from torch.utils.data import Dataset

from event_window_jepa.data.anchor_sampler import deterministic_seed
from event_window_jepa.data.event_store import EventStore
from event_window_jepa.data.paired_window_dataset import (
    _masked_activity_metrics,
    _patch_event_activity,
)
from event_window_jepa.data.sequence_sampler import (
    RecurrentClipRequest,
    SequenceClip,
    UniformSequenceClipSampler,
)
from event_window_jepa.data.spatial_transforms import (
    SharedRandomSpatialTransform,
    SpatialTransformParameters,
)
from event_window_jepa.data.types import EventWindow, SequenceInfo
from event_window_jepa.masks.multiblock import MaskPair, MultiBlockMaskGenerator


class EventRepresentation(Protocol):
    def __call__(self, window: EventWindow) -> np.ndarray: ...


@dataclass(frozen=True)
class RecurrentWindowDebugSample:
    """Raw clip objects used to inspect temporal and spatial invariants.

    ``spatial_transforms`` and ``sequence_ids`` deliberately repeat the
    per-clip values at every timestep so offline tools can mechanically verify
    that no step-specific geometry or sequence switch entered the pipeline.
    """

    request: RecurrentClipRequest
    clip: SequenceClip
    windows: tuple[EventWindow, ...]
    spatial_transform: SpatialTransformParameters
    spatial_transforms: tuple[SpatialTransformParameters, ...]
    sequence_ids: tuple[str, ...]
    masks: tuple[MaskPair, ...]
    mask_step_seeds: tuple[int, ...]
    sequence_info: SequenceInfo


class RecurrentWindowDataset(Dataset[dict[str, Any]]):
    """Build same-sequence event clips for temporal pretraining.

    A dataset item has temporal-first image shape ``[T, C, H, W]``. PyTorch's
    default collator therefore produces ``[B, T, C, H, W]``. Geometry is
    sampled once per clip and reused for all ``T`` windows, while JEPA masks
    are sampled independently and deterministically at each timestep.

    ``sequence_length`` on the sampler is the number of loss-bearing steps.
    The leading ``burn_in_steps`` initialize recurrent state without a loss,
    giving ``T = burn_in_steps + sequence_length`` online steps.
    ``lookahead_steps`` additional windows are read only to build aligned future
    targets. They are exposed through ``x_future`` and related metadata without
    entering ``x`` or the recurrent control masks. ``loss_mask[t]`` selects
    supervised online steps. ``detach_mask[t]`` requests a state detach
    immediately *before* timestep ``t`` and marks the burn-in/TBPTT boundaries.

    When ``return_patch_event_activity`` is enabled, each sample additionally
    contains ``patch_event_activity`` with shape ``[T, P]`` and dtype int64.
    It stores raw transformed-window event counts on the same flattened patch
    grid as ``context_mask`` and ``target_mask``.  The key is omitted when the
    option is disabled to avoid retaining an otherwise unused batch tensor.
    """

    def __init__(
        self,
        store: EventStore,
        clip_sampler: UniformSequenceClipSampler,
        representation: EventRepresentation,
        mask_generator: MultiBlockMaskGenerator,
        spatial_transform: SharedRandomSpatialTransform,
        *,
        tbptt_steps: int | None = None,
        return_patch_event_activity: bool = False,
        seed: int = 0,
    ) -> None:
        if tbptt_steps is not None and tbptt_steps <= 0:
            raise ValueError("tbptt_steps must be positive when provided")
        if not isinstance(return_patch_event_activity, bool):
            raise TypeError("return_patch_event_activity must be a boolean")
        self.store = store
        self.clip_sampler = clip_sampler
        self.representation = representation
        self.mask_generator = mask_generator
        self.spatial_transform = spatial_transform
        self.tbptt_steps = tbptt_steps
        self.return_patch_event_activity = return_patch_event_activity
        self.seed = int(seed)
        self.epoch = 0
        self._metadata = {info.sequence_id: info for info in store.sequences()}
        missing = {
            info.sequence_id
            for info in self.clip_sampler.sequences
            if info.sequence_id not in self._metadata
        }
        if missing:
            raise ValueError(
                "clip sampler contains sequences absent from the event store: "
                f"{sorted(missing)[:5]}"
            )

    def __len__(self) -> int:
        return len(self.clip_sampler)

    @property
    def total_steps(self) -> int:
        """Number of sampled windows, including target-only lookahead."""

        return self.clip_sampler.total_steps

    @property
    def online_steps(self) -> int:
        """Number of windows consumed by the online recurrent encoder."""

        return self.clip_sampler.online_steps

    @property
    def lookahead_steps(self) -> int:
        return self.clip_sampler.lookahead_steps

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch cannot be negative")
        self.epoch = epoch

    def _training_masks(self) -> tuple[torch.Tensor, torch.Tensor]:
        burn_in = self.clip_sampler.burn_in_steps
        online_steps = self.clip_sampler.online_steps
        loss_mask = torch.zeros(online_steps, dtype=torch.bool)
        loss_mask[burn_in:] = True

        # True means: detach recurrent state immediately before this timestep.
        # The burn-in boundary is detached so initialization does not become an
        # unbounded gradient path. Further boundaries implement TBPTT, while
        # gradients remain full-BPTT inside every chunk.
        detach_mask = torch.zeros(online_steps, dtype=torch.bool)
        detach_mask[burn_in] = True
        if self.tbptt_steps is not None:
            for step in range(
                burn_in + self.tbptt_steps,
                online_steps,
                self.tbptt_steps,
            ):
                detach_mask[step] = True
        return loss_mask, detach_mask

    def _request_for_index(
        self, index: int | RecurrentClipRequest
    ) -> RecurrentClipRequest:
        if isinstance(index, RecurrentClipRequest):
            return index
        if not isinstance(index, int):
            raise TypeError("dataset index must be an integer or RecurrentClipRequest")

        clip = self.clip_sampler.sample(index, self.epoch)
        augmentation_seed = deterministic_seed(
            self.seed, self.epoch, index, stream=1
        ) & ((1 << 63) - 1)
        mask_seed = deterministic_seed(
            self.seed, self.epoch, index, stream=2
        ) & ((1 << 63) - 1)
        return RecurrentClipRequest(
            clip=clip,
            sampling_mode="random",
            stream_id="",
            state_reset=True,
            augmentation_seed=augmentation_seed,
            augmentation_id=(
                f"random:epoch-{self.epoch}:item-{index}:"
                f"augmentation-{augmentation_seed}"
            ),
            mask_seed=mask_seed,
        )

    def _build_sample(
        self, index: int | RecurrentClipRequest
    ) -> tuple[dict[str, Any], RecurrentWindowDebugSample]:
        request = self._request_for_index(index)
        clip = request.clip
        if len(clip.t_end_us) != self.total_steps:
            raise RuntimeError("clip sampler returned an unexpected number of timesteps")
        timestamps = np.asarray(clip.t_end_us, dtype=np.int64)
        if timestamps.size > 1 and bool(np.any(timestamps[1:] <= timestamps[:-1])):
            raise RuntimeError("recurrent clip timestamps are not strictly increasing")
        if timestamps.size > 1 and bool(
            np.any(np.diff(timestamps) != self.clip_sampler.stride_us)
        ):
            raise RuntimeError("recurrent clip does not use the configured fixed stride")

        if clip.sequence_id not in self._metadata:
            raise ValueError(
                f"clip request references an unknown sequence: {clip.sequence_id}"
            )
        info = self._metadata[clip.sequence_id]
        if (
            timestamps[0] - self.clip_sampler.base_window_us < info.t_start_us
            or timestamps[-1] > info.t_end_us
        ):
            raise RuntimeError("recurrent clip crosses a sequence boundary")

        augmentation_rng = random.Random(request.augmentation_seed)
        transform = self.spatial_transform.sample(
            augmentation_rng, info.height, info.width
        )
        windows: list[EventWindow] = []
        representations: list[np.ndarray] = []
        masks: list[MaskPair] = []
        mask_step_seeds: list[int] = []
        target_active_ratios: list[float] = []
        target_mass_coverages: list[float] = []
        patch_event_activities: list[np.ndarray] = []

        # Read the whole temporal span once. This is substantially cheaper for
        # HDF5-backed stores than issuing one overlapping disk read per step.
        first_start_us = clip.t_end_us[0] - self.clip_sampler.base_window_us
        complete_span = self.store.slice(
            clip.sequence_id,
            clip.t_end_us[-1],
            clip.t_end_us[-1] - first_start_us,
        )

        grid_size = (
            self.mask_generator.grid_height,
            self.mask_generator.grid_width,
        )
        for step, t_end_us in enumerate(clip.t_end_us):
            t_start_us = t_end_us - self.clip_sampler.base_window_us
            left = int(np.searchsorted(complete_span.t_us, t_start_us, side="right"))
            right = int(np.searchsorted(complete_span.t_us, t_end_us, side="right"))
            window = EventWindow(
                x=complete_span.x[left:right],
                y=complete_span.y[left:right],
                t_us=complete_span.t_us[left:right],
                polarity=complete_span.polarity[left:right],
                t_start_us=t_start_us,
                t_end_us=t_end_us,
                height=complete_span.height,
                width=complete_span.width,
            )
            window = self.spatial_transform.apply(window, transform)
            activity = _patch_event_activity(window, grid_size)
            mask_step_seed = deterministic_seed(
                request.mask_seed, 0, step, stream=0
            ) & ((1 << 63) - 1)
            mask = self.mask_generator.sample(
                random.Random(mask_step_seed), activity=activity
            )
            active_ratio, mass_coverage = _masked_activity_metrics(
                activity, mask.target
            )
            representation = np.ascontiguousarray(
                self.representation(window), dtype=np.float32
            )
            if representation.ndim != 3:
                raise ValueError("event representation must have shape [C, H, W]")

            windows.append(window)
            representations.append(representation)
            masks.append(mask)
            mask_step_seeds.append(mask_step_seed)
            target_active_ratios.append(active_ratio)
            target_mass_coverages.append(mass_coverage)
            if self.return_patch_event_activity:
                patch_event_activities.append(
                    np.ascontiguousarray(activity.reshape(-1), dtype=np.int64)
                )

        first_shape = representations[0].shape
        if any(value.shape != first_shape for value in representations[1:]):
            raise ValueError("all clip representations must share shape [C, H, W]")
        all_x = torch.from_numpy(np.stack(representations, axis=0))
        online_steps = self.online_steps
        online_slice = slice(0, online_steps)
        online_masks = masks[online_slice]
        online_target_active_ratios = target_active_ratios[online_slice]
        online_target_mass_coverages = target_mass_coverages[online_slice]
        loss_mask, detach_mask = self._training_masks()

        sample = {
            "x": all_x[online_slice],
            "dt_ms": torch.full(
                (online_steps,),
                self.clip_sampler.base_window_ms,
                dtype=torch.float32,
            ),
            "t_end_us": torch.from_numpy(timestamps[online_slice]),
            "sequence_id": clip.sequence_id,
            "sampling_mode": request.sampling_mode,
            "stream_id": request.stream_id,
            "state_reset": torch.tensor(request.state_reset, dtype=torch.bool),
            "augmentation_seed": request.augmentation_seed,
            "augmentation_id": request.augmentation_id,
            "mask_seed": request.mask_seed,
            "mask_step_seeds": torch.tensor(
                mask_step_seeds[online_slice], dtype=torch.int64
            ),
            "context_mask": torch.from_numpy(
                np.stack([mask.context_keep for mask in online_masks], axis=0)
            ),
            "target_mask": torch.from_numpy(
                np.stack([mask.target for mask in online_masks], axis=0)
            ),
            "loss_mask": loss_mask,
            "detach_mask": detach_mask,
            "mask_activity_aware": torch.tensor(
                [float(mask.activity_aware) for mask in online_masks],
                dtype=torch.float32,
            ),
            "mask_activity_fallback": torch.tensor(
                [float(mask.activity_fallback) for mask in online_masks],
                dtype=torch.float32,
            ),
            "mask_context_active_patch_ratio": torch.tensor(
                [mask.selection_active_patch_ratio for mask in online_masks],
                dtype=torch.float32,
            ),
            "mask_context_event_mass_coverage": torch.tensor(
                [mask.selection_event_mass_coverage for mask in online_masks],
                dtype=torch.float32,
            ),
            "mask_target_active_patch_ratio": torch.tensor(
                online_target_active_ratios, dtype=torch.float32
            ),
            "mask_target_event_mass_coverage": torch.tensor(
                online_target_mass_coverages, dtype=torch.float32
            ),
            "mask_empty_target": torch.tensor(
                [float(value == 0.0) for value in online_target_active_ratios],
                dtype=torch.float32,
            ),
        }
        if self.return_patch_event_activity:
            all_patch_event_activity = torch.from_numpy(
                np.stack(patch_event_activities, axis=0)
            )
            sample["patch_event_activity"] = all_patch_event_activity[online_slice]
        if self.lookahead_steps:
            future_slice = slice(
                self.lookahead_steps,
                self.lookahead_steps + online_steps,
            )
            sample["x_future"] = all_x[future_slice]
            sample["future_dt_ms"] = torch.full(
                (online_steps,),
                self.clip_sampler.base_window_ms,
                dtype=torch.float32,
            )
            sample["future_t_end_us"] = torch.from_numpy(timestamps[future_slice])
            if self.return_patch_event_activity:
                sample["future_patch_event_activity"] = all_patch_event_activity[
                    future_slice
                ]
        debug = RecurrentWindowDebugSample(
            request=request,
            clip=clip,
            windows=tuple(windows),
            spatial_transform=transform,
            spatial_transforms=tuple(transform for _ in windows),
            sequence_ids=tuple(clip.sequence_id for _ in windows),
            masks=tuple(masks),
            mask_step_seeds=tuple(mask_step_seeds),
            sequence_info=info,
        )
        return sample, debug

    def __getitem__(
        self, index: int | RecurrentClipRequest
    ) -> dict[str, Any]:
        sample, _ = self._build_sample(index)
        return sample

    def sample_with_debug(
        self, index: int | RecurrentClipRequest
    ) -> tuple[dict[str, Any], RecurrentWindowDebugSample]:
        """Return the training sample together with its transformed windows."""

        return self._build_sample(index)
