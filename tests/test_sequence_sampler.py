from __future__ import annotations

import numpy as np
import pytest

from event_window_jepa.data.sequence_sampler import (
    MixedRecurrentBatchSampler,
    RecurrentClipRequest,
    SequenceClip,
    UniformSequenceClipSampler,
)
from event_window_jepa.data.types import SequenceInfo


def test_sampler_builds_strict_fixed_stride_clips_inside_one_sequence() -> None:
    sequences = (
        SequenceInfo("long", None, 4, 4, 0, 200_000, "train", "events"),
        # A complete clip needs 50 ms + four 10 ms strides = 90 ms.
        SequenceInfo("short", None, 4, 4, 0, 89_999, "train", "events"),
    )
    sampler = UniformSequenceClipSampler(
        sequences,
        base_window_ms=50,
        stride_ms=10,
        sequence_length=3,
        burn_in_steps=2,
        samples_per_epoch=32,
        seed=7,
        sampling_strategy="sequence_balanced",
    )

    assert sampler.total_steps == 5
    assert sampler.required_duration_us == 90_000
    assert [info.sequence_id for info in sampler.sequences] == ["long"]
    for index in range(len(sampler)):
        clip = sampler.sample(index, epoch=3)
        assert clip.sequence_id == "long"
        assert len(clip.t_end_us) == 5
        assert np.diff(clip.t_end_us).tolist() == [10_000] * 4
        assert clip.t_end_us[0] - sampler.base_window_us >= 0
        assert clip.t_end_us[-1] <= 200_000
        assert clip == sampler.sample(index, epoch=3)


def test_sampler_rejects_invalid_or_impossible_clip_geometry() -> None:
    sequence = SequenceInfo("s", None, 4, 4, 0, 80_000)
    with pytest.raises(ValueError, match="no sequence is long enough"):
        UniformSequenceClipSampler(
            (sequence,),
            base_window_ms=50,
            stride_ms=20,
            sequence_length=3,
            burn_in_steps=0,
            samples_per_epoch=1,
        )
    with pytest.raises(ValueError, match="sequence_length"):
        UniformSequenceClipSampler(
            (sequence,),
            base_window_ms=50,
            stride_ms=10,
            sequence_length=0,
            burn_in_steps=0,
            samples_per_epoch=1,
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        SequenceClip("s", (60_000, 60_000))


def _mixed_sequences(count: int = 8) -> tuple[SequenceInfo, ...]:
    return tuple(
        SequenceInfo(
            sequence_id=f"sequence-{index}",
            path=None,
            height=4,
            width=4,
            t_start_us=0,
            t_end_us=300_000,
            split="train",
            dataset="events",
            source_recording_id=f"recording-{index}",
        )
        for index in range(count)
    )


def _mixed_sampler(*, rank: int = 0, seed: int = 17) -> MixedRecurrentBatchSampler:
    return MixedRecurrentBatchSampler(
        _mixed_sequences(),
        base_window_ms=50,
        stride_ms=50,
        sequence_length=2,
        burn_in_steps=1,
        samples_per_epoch=64,
        batch_size=4,
        stream_ratio=(1, 1),
        world_size=2,
        rank=rank,
        seed=seed,
        random_sampling_strategy="sequence_balanced",
    )


def test_mixed_sampler_preserves_stream_lanes_and_global_epoch_accounting() -> None:
    rank_batches = [list(_mixed_sampler(rank=rank)) for rank in range(2)]

    assert [len(batches) for batches in rank_batches] == [8, 8]
    for rank, batches in enumerate(rank_batches):
        for batch in batches:
            assert len(batch) == 4
            assert [request.sampling_mode for request in batch] == [
                "stream",
                "stream",
                "random",
                "random",
            ]
            assert [request.stream_id for request in batch[:2]] == [
                f"rank-{rank}:lane-0",
                f"rank-{rank}:lane-1",
            ]
            assert all(request.state_reset for request in batch[2:])

        for local_lane in range(2):
            previous: RecurrentClipRequest | None = None
            for batch in batches:
                request = batch[local_lane]
                assert np.diff(request.clip.t_end_us).tolist() == [50_000, 50_000]
                if (
                    previous is not None
                    and previous.clip.sequence_id == request.clip.sequence_id
                ):
                    assert request.clip.t_end_us[0] == (
                        previous.clip.t_end_us[-1] + 50_000
                    )
                    assert request.state_reset is False
                    assert request.augmentation_seed == previous.augmentation_seed
                    assert request.augmentation_id == previous.augmentation_id
                else:
                    assert request.state_reset is True
                previous = request

    rank_recordings = [
        {
            request.clip.sequence_id
            for batch in batches
            for request in batch[:2]
        }
        for batches in rank_batches
    ]
    assert rank_recordings[0]
    assert rank_recordings[1]
    assert rank_recordings[0].isdisjoint(rank_recordings[1])

    sampler = _mixed_sampler()
    assert sampler.samples_per_epoch == 64
    assert sampler.effective_samples_per_epoch == 64
    assert sampler.global_batch_size == 8
    assert len(sampler) == 8


def test_mixed_sampler_epoch_shuffle_is_deterministic_and_shards_stay_fixed() -> None:
    first = _mixed_sampler(seed=23)
    second = _mixed_sampler(seed=23)
    first.set_epoch(4)
    second.set_epoch(4)

    first_batches = list(first)
    second_batches = list(second)
    assert first_batches == second_batches

    fixed_membership = {
        request.stream_id: {
            item.clip.sequence_id
            for batch in first_batches
            for item in batch[: first.stream_batch_size]
            if item.stream_id == request.stream_id
        }
        for batch in first_batches
        for request in batch[: first.stream_batch_size]
    }
    first.set_epoch(5)
    next_membership = {
        request.stream_id: {
            item.clip.sequence_id
            for batch in list(first)
            for item in batch[: first.stream_batch_size]
            if item.stream_id == request.stream_id
        }
        for request in first_batches[0][: first.stream_batch_size]
    }
    assert next_membership == fixed_membership


def test_mixed_sampler_drops_only_incomplete_global_batch_and_validates_streaming() -> None:
    sampler = MixedRecurrentBatchSampler(
        _mixed_sequences(),
        base_window_ms=50,
        stride_ms=50,
        sequence_length=2,
        burn_in_steps=1,
        samples_per_epoch=67,
        batch_size=4,
        world_size=2,
        rank=0,
    )
    assert len(sampler) == 8
    assert sampler.effective_samples_per_epoch == 64

    with pytest.raises(ValueError, match="stride_ms"):
        MixedRecurrentBatchSampler(
            _mixed_sequences(),
            base_window_ms=50,
            stride_ms=10,
            sequence_length=2,
            burn_in_steps=1,
            samples_per_epoch=64,
            batch_size=4,
        )


def test_stream_lane_resets_between_sequences_of_the_same_recording() -> None:
    sequences = (
        SequenceInfo(
            "part-a",
            None,
            4,
            4,
            0,
            150_000,
            source_recording_id="recording",
        ),
        SequenceInfo(
            "part-b",
            None,
            4,
            4,
            150_000,
            300_000,
            source_recording_id="recording",
        ),
    )
    sampler = MixedRecurrentBatchSampler(
        sequences,
        base_window_ms=50,
        stride_ms=50,
        sequence_length=2,
        burn_in_steps=1,
        samples_per_epoch=4,
        batch_size=2,
        seed=43,
        random_sampling_strategy="sequence_balanced",
    )

    stream_requests = [batch[0] for batch in sampler]
    assert [request.clip.sequence_id for request in stream_requests] == [
        "part-a",
        "part-b",
    ]
    assert [request.state_reset for request in stream_requests] == [True, True]
    assert len({request.augmentation_seed for request in stream_requests}) == 1
    assert len({request.augmentation_id for request in stream_requests}) == 1
    assert len({request.mask_seed for request in stream_requests}) == 2
