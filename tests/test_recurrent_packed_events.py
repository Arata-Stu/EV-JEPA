from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from event_window_jepa.data.event_store import InMemoryEventStore
from event_window_jepa.data.packed_events import (
    PACKED_EVENT_BATCH_KEY,
    RAW_EVENT_WINDOWS_KEY,
    PackedEventBatch,
    collate_recurrent_samples,
    pack_event_windows,
)
from event_window_jepa.data.recurrent_window_dataset import RecurrentWindowDataset
from event_window_jepa.data.sequence_sampler import (
    RecurrentClipRequest,
    SequenceClip,
    UniformSequenceClipSampler,
)
from event_window_jepa.data.spatial_transforms import SharedRandomSpatialTransform
from event_window_jepa.data.types import EventWindow, SequenceInfo
from event_window_jepa.masks.multiblock import MultiBlockMaskGenerator
from event_window_jepa.representations.event_image import EventImage


def _window(
    *,
    t_start_us: int,
    t_end_us: int,
    x: list[int],
    y: list[int],
    t_us: list[int],
    polarity: list[int],
) -> EventWindow:
    return EventWindow(
        x=np.asarray(x, dtype=np.int64),
        y=np.asarray(y, dtype=np.int64),
        t_us=np.asarray(t_us, dtype=np.int64),
        polarity=np.asarray(polarity, dtype=np.int8),
        t_start_us=t_start_us,
        t_end_us=t_end_us,
        height=4,
        width=5,
    )


def _dataset(
    *,
    return_packed_events: bool = False,
    max_events_per_window: int | None = None,
) -> RecurrentWindowDataset:
    within_window = np.asarray([1_000, 3_000, 5_000, 7_000, 10_000])
    timestamps = np.concatenate(
        [within_window + 10_000 * step for step in range(4)]
    ).astype(np.int64)
    event_index = np.arange(len(timestamps), dtype=np.int64)
    info = SequenceInfo(
        "sequence",
        None,
        4,
        4,
        0,
        40_000,
        "train",
        "synthetic",
    )
    store = InMemoryEventStore(
        {
            "sequence": {
                "x": event_index % 4,
                "y": (event_index * 3) % 4,
                "t_us": timestamps,
                "polarity": event_index % 2,
            }
        },
        {"sequence": info},
    )
    sampler = UniformSequenceClipSampler(
        (info,),
        base_window_ms=10,
        stride_ms=10,
        sequence_length=2,
        burn_in_steps=1,
        lookahead_steps=1,
        samples_per_epoch=2,
        seed=7,
        sampling_strategy="sequence_balanced",
    )
    return RecurrentWindowDataset(
        store=store,
        clip_sampler=sampler,
        representation=EventImage(normalization="none"),
        mask_generator=MultiBlockMaskGenerator(
            (2, 2),
            target_blocks=1,
            target_area_range=(0.25, 0.25),
            context_keep_ratio=0.5,
        ),
        spatial_transform=SharedRandomSpatialTransform(
            (4, 4), horizontal_flip_probability=1.0
        ),
        return_patch_event_activity=True,
        return_packed_events=return_packed_events,
        max_events_per_window=max_events_per_window,
        seed=13,
    )


def _request(augmentation_seed: int, mask_seed: int) -> RecurrentClipRequest:
    return RecurrentClipRequest(
        clip=SequenceClip("sequence", (10_000, 20_000, 30_000, 40_000)),
        sampling_mode="random",
        stream_id="",
        state_reset=True,
        augmentation_seed=augmentation_seed,
        augmentation_id=f"augmentation-{augmentation_seed}",
        mask_seed=mask_seed,
    )


def test_pack_event_windows_is_padding_free_and_preserves_empty_windows() -> None:
    empty = _window(
        t_start_us=1_000,
        t_end_us=2_000,
        x=[],
        y=[],
        t_us=[],
        polarity=[],
    )
    packed = pack_event_windows(
        (
            (
                _window(
                    t_start_us=0,
                    t_end_us=1_000,
                    x=[1, 3],
                    y=[2, 0],
                    t_us=[250, 1_000],
                    polarity=[0, 1],
                ),
                empty,
            ),
            (
                _window(
                    t_start_us=0,
                    t_end_us=1_000,
                    x=[4],
                    y=[3],
                    t_us=[500],
                    polarity=[-1],
                ),
                _window(
                    t_start_us=1_000,
                    t_end_us=2_000,
                    x=[0, 2],
                    y=[1, 2],
                    t_us=[1_250, 2_000],
                    polarity=[1, -1],
                ),
            ),
        )
    )

    assert packed.event_count == 5
    assert packed.window_counts.tolist() == [2, 0, 1, 2]
    assert packed.window_offsets.tolist() == [0, 2, 2, 3, 5]
    assert packed.x.tolist() == [1.0, 3.0, 4.0, 0.0, 2.0]
    assert packed.y.tolist() == [2.0, 0.0, 3.0, 1.0, 2.0]
    assert torch.allclose(
        packed.t,
        torch.tensor([0.25, 1.0, 0.5, 0.25, 1.0]),
    )
    assert packed.polarity.tolist() == [-1.0, 1.0, -1.0, 1.0, -1.0]
    assert packed.batch_index.tolist() == [0, 0, 1, 1, 1]
    assert packed.time_index.tolist() == [0, 0, 0, 1, 1]

    selected = packed.select_time_range(1, 2)
    assert selected.time_steps == 1
    assert selected.window_counts.tolist() == [0, 2]
    assert selected.window_offsets.tolist() == [0, 0, 2]
    assert selected.batch_index.tolist() == [1, 1]
    assert selected.time_index.tolist() == [0, 0]
    assert selected.x.tolist() == [0.0, 2.0]
    assert selected.to("cpu").window_offsets.tolist() == [0, 0, 2]
    assert callable(packed.pin_memory)


def test_dataset_raw_payload_is_online_only_and_caps_no_training_inputs() -> None:
    dataset = _dataset(return_packed_events=True, max_events_per_window=2)
    first, first_debug = dataset.sample_with_debug(_request(1, 2))
    second, second_debug = dataset.sample_with_debug(_request(99, 101))

    first_windows = first[RAW_EVENT_WINDOWS_KEY]
    second_windows = second[RAW_EVENT_WINDOWS_KEY]
    assert len(first_debug.windows) == len(second_debug.windows) == 4
    assert len(first_windows) == len(second_windows) == dataset.online_steps == 3
    assert [window.event_count for window in first_windows] == [2, 2, 2]
    assert [window.t_us.tolist() for window in first_windows] == [
        window.t_us.tolist() for window in second_windows
    ]
    assert all(
        np.all(window.t_us[1:] >= window.t_us[:-1]) for window in first_windows
    )
    for packed_window, full_window in zip(
        first_windows,
        first_debug.windows[: dataset.online_steps],
        strict=True,
    ):
        selected = np.searchsorted(full_window.t_us, packed_window.t_us)
        assert np.array_equal(packed_window.x, full_window.x[selected])
        assert np.array_equal(packed_window.y, full_window.y[selected])
        assert np.array_equal(packed_window.polarity, full_window.polarity[selected])

    full_counts = [window.event_count for window in first_debug.windows[:3]]
    assert full_counts == [5, 5, 5]
    assert first["x"].sum(dim=(1, 2, 3)).tolist() == [5.0, 5.0, 5.0]
    assert first["patch_event_activity"].sum(dim=1).tolist() == full_counts


def test_recurrent_collate_packs_events_and_keeps_default_batch_contract() -> None:
    dataset = _dataset(return_packed_events=True, max_events_per_window=2)
    samples = [
        dataset.sample_with_debug(_request(1, 2))[0],
        dataset.sample_with_debug(_request(3, 4))[0],
    ]
    batch = collate_recurrent_samples(samples)
    packed = batch[PACKED_EVENT_BATCH_KEY]

    assert isinstance(packed, PackedEventBatch)
    assert RAW_EVENT_WINDOWS_KEY not in batch
    assert batch["x"].shape == (2, 3, 2, 4, 4)
    assert packed.batch_size == 2
    assert packed.time_steps == 3
    assert packed.window_counts.tolist() == [2] * 6
    assert packed.window_offsets.tolist() == list(range(0, 13, 2))
    assert packed.batch_index.tolist() == [0] * 6 + [1] * 6
    assert packed.time_index.tolist() == [0, 0, 1, 1, 2, 2] * 2
    assert bool(((packed.t >= 0.0) & (packed.t <= 1.0)).all())
    expected_polarity = [
        1.0 if int(value) > 0 else -1.0
        for sample in samples
        for window in sample[RAW_EVENT_WINDOWS_KEY]
        for value in window.polarity
    ]
    assert packed.polarity.tolist() == expected_polarity


def test_feature_is_off_by_default_and_default_collate_still_works() -> None:
    dataset = _dataset()
    sample = dataset[0]
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
    explicit_batch = collate_recurrent_samples([dataset[0], dataset[1]])

    assert RAW_EVENT_WINDOWS_KEY not in sample
    assert PACKED_EVENT_BATCH_KEY not in batch
    assert PACKED_EVENT_BATCH_KEY not in explicit_batch
    assert batch["x"].shape == (2, 3, 2, 4, 4)
    assert explicit_batch["x"].shape == batch["x"].shape

    with pytest.raises(ValueError, match="requires return_packed_events"):
        _dataset(max_events_per_window=2)
