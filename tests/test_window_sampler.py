from __future__ import annotations

import random

from event_window_jepa.data.anchor_sampler import UniformTimeAnchorSampler, WindowPairSampler
from event_window_jepa.data.types import SequenceInfo


def test_pairs_respect_ratio_and_direction() -> None:
    sampler = WindowPairSampler(
        [10, 20, 40, 80], [10, 20, 40, 80], minimum_ratio=1.5, direction="short_to_long"
    )
    assert sampler.valid_pairs
    for pair in sampler.valid_pairs:
        assert pair.context_ms < pair.target_ms
        assert pair.target_ms / pair.context_ms >= 1.5


def test_pair_sampling_is_reproducible() -> None:
    sampler = WindowPairSampler([10, 20], [40], minimum_ratio=1.5)
    first = sampler.sample(random.Random(12))
    second = sampler.sample(random.Random(12))
    assert first == second


def test_dataset_balanced_sampling_is_not_sequence_count_weighted() -> None:
    sequences = [
        SequenceInfo(f"a_{index}", None, 240, 304, 0, 1_000_000, "train", "a")
        for index in range(10)
    ]
    sequences.append(SequenceInfo("b_0", None, 480, 640, 0, 1_000_000, "train", "b"))
    sampler = UniformTimeAnchorSampler(
        sequences,
        maximum_window_ms=120,
        samples_per_epoch=1_000,
        seed=7,
        sampling_strategy="dataset_balanced",
    )
    b_samples = sum(
        sampler.sample(index, epoch=0).sequence_id == "b_0" for index in range(1_000)
    )
    assert 400 <= b_samples <= 600
