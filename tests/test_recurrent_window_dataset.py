from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from event_window_jepa.data.event_store import InMemoryEventStore
from event_window_jepa.data.recurrent_window_dataset import RecurrentWindowDataset
from event_window_jepa.data.sequence_sampler import (
    MixedRecurrentBatchSampler,
    RecurrentClipRequest,
    UniformSequenceClipSampler,
)
from event_window_jepa.data.spatial_transforms import (
    SharedRandomSpatialTransform,
    SpatialTransformParameters,
)
from event_window_jepa.data.types import EventWindow, SequenceInfo
from event_window_jepa.masks.multiblock import MultiBlockMaskGenerator
from event_window_jepa.representations.event_image import EventImage


class RecordingStore(InMemoryEventStore):
    def __init__(self) -> None:
        timestamps = np.arange(1_000, 300_001, 1_000, dtype=np.int64)
        indices = np.arange(len(timestamps), dtype=np.int64)
        super().__init__(
            {
                "sequence": {
                    # The center coordinate survives every valid 4x4 crop of
                    # this 6x6 sensor, keeping temporal-boundary tests exact.
                    "x": np.full_like(indices, 2),
                    "y": np.full_like(indices, 2),
                    "t_us": timestamps,
                    "polarity": indices % 2,
                }
            },
            {
                "sequence": SequenceInfo(
                    "sequence", None, 6, 6, 0, 300_000, "train", "events"
                )
            },
        )
        self.calls: list[tuple[str, int, int]] = []

    def slice(
        self, sequence_id: str, t_end_us: int, duration_us: int
    ) -> EventWindow:
        self.calls.append((sequence_id, t_end_us, duration_us))
        return super().slice(sequence_id, t_end_us, duration_us)


class RecordingTransform(SharedRandomSpatialTransform):
    def __init__(self) -> None:
        super().__init__((4, 4), horizontal_flip_probability=1.0)
        self.sample_calls = 0
        self.applied: list[SpatialTransformParameters] = []

    def sample(
        self, rng: random.Random, input_height: int, input_width: int
    ) -> SpatialTransformParameters:
        self.sample_calls += 1
        return super().sample(rng, input_height, input_width)

    def apply(
        self, window: EventWindow, params: SpatialTransformParameters
    ) -> EventWindow:
        self.applied.append(params)
        return super().apply(window, params)


def make_dataset(
    *,
    samples_per_epoch: int = 2,
    tbptt_steps: int | None = 2,
    stride_ms: float = 10,
    return_patch_event_activity: bool = False,
) -> tuple[RecurrentWindowDataset, RecordingStore, RecordingTransform]:
    store = RecordingStore()
    sampler = UniformSequenceClipSampler(
        store.sequences("train"),
        base_window_ms=50,
        stride_ms=stride_ms,
        sequence_length=4,
        burn_in_steps=2,
        samples_per_epoch=samples_per_epoch,
        seed=11,
        sampling_strategy="sequence_balanced",
    )
    transform = RecordingTransform()
    dataset = RecurrentWindowDataset(
        store=store,
        clip_sampler=sampler,
        representation=EventImage(normalization="none"),
        mask_generator=MultiBlockMaskGenerator(
            (2, 2),
            target_blocks=1,
            target_area_range=(0.25, 0.25),
            context_keep_ratio=0.5,
        ),
        spatial_transform=transform,
        tbptt_steps=tbptt_steps,
        return_patch_event_activity=return_patch_event_activity,
        seed=13,
    )
    return dataset, store, transform


def test_dataset_returns_temporal_tensors_masks_and_shared_geometry() -> None:
    dataset, store, transform = make_dataset(samples_per_epoch=1)
    sample, debug = dataset.sample_with_debug(0)

    assert sample["x"].shape == (6, 2, 4, 4)
    assert sample["x"].dtype == torch.float32
    assert sample["dt_ms"].tolist() == [50.0] * 6
    assert sample["context_mask"].shape == sample["target_mask"].shape == (6, 4)
    assert sample["context_mask"].dtype == sample["target_mask"].dtype == torch.bool
    for name in (
        "mask_activity_aware",
        "mask_activity_fallback",
        "mask_context_active_patch_ratio",
        "mask_context_event_mass_coverage",
        "mask_target_active_patch_ratio",
        "mask_target_event_mass_coverage",
        "mask_empty_target",
    ):
        assert sample[name].shape == (6,)

    assert sample["sequence_id"] == "sequence"
    assert sample["sampling_mode"] == "random"
    assert sample["stream_id"] == ""
    assert sample["state_reset"].dtype == torch.bool
    assert sample["state_reset"].ndim == 0
    assert sample["state_reset"].item() is True
    assert isinstance(sample["augmentation_seed"], int)
    assert isinstance(sample["augmentation_id"], str)
    assert isinstance(sample["mask_seed"], int)
    assert sample["mask_step_seeds"].shape == (6,)
    assert debug.request.sampling_mode == "random"
    assert debug.mask_step_seeds == tuple(sample["mask_step_seeds"].tolist())
    assert torch.diff(sample["t_end_us"]).tolist() == [10_000] * 5
    assert sample["loss_mask"].tolist() == [False, False, True, True, True, True]
    assert sample["detach_mask"].tolist() == [False, False, True, False, True, False]
    assert "patch_event_activity" not in sample

    assert transform.sample_calls == 1
    assert len(transform.applied) == 6
    assert all(params == debug.spatial_transform for params in transform.applied)
    assert debug.spatial_transforms == (debug.spatial_transform,) * 6
    assert debug.sequence_ids == ("sequence",) * 6
    assert len(debug.windows) == 6
    assert all(window.height == window.width == 4 for window in debug.windows)
    assert [window.t_end_us for window in debug.windows] == list(
        debug.clip.t_end_us
    )
    assert all(window.duration_us == 50_000 for window in debug.windows)

    # Overlapping windows are split from one complete-span EventStore read.
    assert len(store.calls) == 1
    assert {sequence_id for sequence_id, _, _ in store.calls} == {"sequence"}
    assert store.calls[0][1] == debug.clip.t_end_us[-1]
    assert store.calls[0][2] == 100_000


def test_optional_patch_event_activity_matches_transformed_raw_windows() -> None:
    dataset, _, _ = make_dataset(
        samples_per_epoch=1,
        return_patch_event_activity=True,
    )
    sample, debug = dataset.sample_with_debug(0)

    activity = sample["patch_event_activity"]
    assert activity.shape == sample["target_mask"].shape == (6, 4)
    assert activity.dtype == torch.int64
    assert torch.all(activity >= 0)
    assert activity.sum(dim=1).tolist() == [
        window.event_count for window in debug.windows
    ]


def test_default_collation_produces_batch_time_channel_height_width() -> None:
    dataset, _, _ = make_dataset(samples_per_epoch=2)
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))

    assert batch["x"].shape == (2, 6, 2, 4, 4)
    assert batch["dt_ms"].shape == batch["t_end_us"].shape == (2, 6)
    assert batch["context_mask"].shape == batch["target_mask"].shape == (2, 6, 4)
    assert batch["loss_mask"].shape == batch["detach_mask"].shape == (2, 6)
    assert batch["sequence_id"] == ["sequence", "sequence"]


def test_non_overlapping_windows_assign_boundary_events_exactly_once() -> None:
    dataset, _, _ = make_dataset(samples_per_epoch=1, stride_ms=50)
    sample, debug = dataset.sample_with_debug(0)

    assert sample["t_end_us"].tolist() == [
        50_000,
        100_000,
        150_000,
        200_000,
        250_000,
        300_000,
    ]
    flattened = np.concatenate([window.t_us for window in debug.windows])
    assert flattened.tolist() == list(range(1_000, 300_001, 1_000))
    assert 50_000 in debug.windows[0].t_us
    assert 50_000 not in debug.windows[1].t_us


def test_sampling_is_deterministic_and_full_bptt_has_no_detach_boundary() -> None:
    dataset, _, _ = make_dataset(samples_per_epoch=1)
    first = dataset[0]
    second = dataset[0]
    for key in (
        "x",
        "t_end_us",
        "context_mask",
        "target_mask",
        "loss_mask",
        "detach_mask",
    ):
        assert torch.equal(first[key], second[key])

    full_bptt, _, _ = make_dataset(samples_per_epoch=1, tbptt_steps=None)
    assert full_bptt[0]["detach_mask"].tolist() == [
        False,
        False,
        True,
        False,
        False,
        False,
    ]


def test_dataset_rejects_non_positive_tbptt_length() -> None:
    store = RecordingStore()
    sampler = UniformSequenceClipSampler(
        store.sequences(),
        base_window_ms=50,
        stride_ms=10,
        sequence_length=2,
        burn_in_steps=0,
        samples_per_epoch=1,
    )
    with pytest.raises(ValueError, match="tbptt_steps"):
        RecurrentWindowDataset(
            store,
            sampler,
            EventImage(),
            MultiBlockMaskGenerator(
                (2, 2),
                target_blocks=1,
                target_area_range=(0.25, 0.25),
                context_keep_ratio=0.5,
            ),
            SharedRandomSpatialTransform((4, 4)),
            tbptt_steps=0,
        )


def test_zero_burn_in_marks_the_first_step_as_a_tbptt_boundary() -> None:
    store = RecordingStore()
    sampler = UniformSequenceClipSampler(
        store.sequences(),
        base_window_ms=50,
        stride_ms=50,
        sequence_length=2,
        burn_in_steps=0,
        samples_per_epoch=1,
    )
    dataset = RecurrentWindowDataset(
        store,
        sampler,
        EventImage(normalization="none"),
        MultiBlockMaskGenerator(
            (2, 2),
            target_blocks=1,
            target_area_range=(0.25, 0.25),
            context_keep_ratio=0.5,
        ),
        SharedRandomSpatialTransform((4, 4), horizontal_flip_probability=0.0),
        tbptt_steps=1,
    )
    sample = dataset[0]
    assert sample["loss_mask"].tolist() == [True, True]
    assert sample["detach_mask"].tolist() == [True, True]


def test_explicit_stream_requests_share_augmentation_but_not_mask_rng() -> None:
    dataset, _, _ = make_dataset(samples_per_epoch=1)
    clip = dataset.clip_sampler.sample(0, epoch=0)
    first_request = RecurrentClipRequest(
        clip=clip,
        sampling_mode="stream",
        stream_id="rank-0:lane-0",
        state_reset=True,
        augmentation_seed=101,
        augmentation_id="recording:sequence:augmentation:101",
        mask_seed=201,
    )
    second_request = RecurrentClipRequest(
        clip=clip,
        sampling_mode="stream",
        stream_id="rank-0:lane-0",
        state_reset=False,
        augmentation_seed=101,
        augmentation_id="recording:sequence:augmentation:101",
        mask_seed=202,
    )

    first, first_debug = dataset.sample_with_debug(first_request)
    second, second_debug = dataset.sample_with_debug(second_request)

    assert first["sampling_mode"] == second["sampling_mode"] == "stream"
    assert first["stream_id"] == second["stream_id"] == "rank-0:lane-0"
    assert first["state_reset"].item() is True
    assert second["state_reset"].item() is False
    assert first["augmentation_seed"] == second["augmentation_seed"] == 101
    assert first["augmentation_id"] == second["augmentation_id"]
    assert first_debug.spatial_transform == second_debug.spatial_transform
    assert first_debug.request == first_request
    assert second_debug.request == second_request
    assert len(set(first_debug.mask_step_seeds)) == dataset.total_steps
    assert len(set(second_debug.mask_step_seeds)) == dataset.total_steps
    assert first_debug.mask_step_seeds != second_debug.mask_step_seeds
    assert first["mask_step_seeds"].tolist() == list(
        first_debug.mask_step_seeds
    )
    assert second["mask_step_seeds"].tolist() == list(
        second_debug.mask_step_seeds
    )


def _mixed_dataset() -> tuple[RecurrentWindowDataset, MixedRecurrentBatchSampler]:
    timestamps = np.arange(1_000, 300_001, 1_000, dtype=np.int64)
    indices = np.arange(len(timestamps), dtype=np.int64)
    events = {}
    metadata = {}
    for sequence_index in range(4):
        sequence_id = f"sequence-{sequence_index}"
        events[sequence_id] = {
            "x": np.full_like(indices, 2),
            "y": np.full_like(indices, 2),
            "t_us": timestamps,
            "polarity": indices % 2,
        }
        metadata[sequence_id] = SequenceInfo(
            sequence_id=sequence_id,
            path=None,
            height=6,
            width=6,
            t_start_us=0,
            t_end_us=300_000,
            split="train",
            dataset="events",
            source_recording_id=f"recording-{sequence_index}",
        )
    store = InMemoryEventStore(events, metadata)
    clip_sampler = UniformSequenceClipSampler(
        store.sequences("train"),
        base_window_ms=50,
        stride_ms=50,
        sequence_length=2,
        burn_in_steps=1,
        samples_per_epoch=8,
        seed=31,
        sampling_strategy="sequence_balanced",
    )
    dataset = RecurrentWindowDataset(
        store=store,
        clip_sampler=clip_sampler,
        representation=EventImage(normalization="none"),
        mask_generator=MultiBlockMaskGenerator(
            (2, 2),
            target_blocks=1,
            target_area_range=(0.25, 0.25),
            context_keep_ratio=0.5,
        ),
        spatial_transform=SharedRandomSpatialTransform(
            (4, 4), horizontal_flip_probability=0.5
        ),
        tbptt_steps=1,
        seed=37,
    )
    batch_sampler = MixedRecurrentBatchSampler(
        store.sequences("train"),
        base_window_ms=50,
        stride_ms=50,
        sequence_length=2,
        burn_in_steps=1,
        samples_per_epoch=8,
        batch_size=4,
        stream_ratio=(1, 1),
        seed=41,
        random_sampling_strategy="sequence_balanced",
    )
    return dataset, batch_sampler


def test_mixed_batch_sampler_requests_collate_to_recurrent_batch() -> None:
    dataset, batch_sampler = _mixed_dataset()
    batch = next(
        iter(DataLoader(dataset, batch_sampler=batch_sampler, num_workers=0))
    )

    assert batch["x"].shape == (4, 3, 2, 4, 4)
    assert batch["sampling_mode"] == ["stream", "stream", "random", "random"]
    assert batch["stream_id"] == ["rank-0:lane-0", "rank-0:lane-1", "", ""]
    assert batch["state_reset"].shape == (4,)
    assert batch["state_reset"].dtype == torch.bool
    assert batch["augmentation_seed"].shape == batch["mask_seed"].shape == (4,)
    assert batch["mask_step_seeds"].shape == (4, 3)
