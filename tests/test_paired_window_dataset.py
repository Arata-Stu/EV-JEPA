from __future__ import annotations

import numpy as np

from event_window_jepa.data.anchor_sampler import UniformTimeAnchorSampler, WindowPairSampler
from event_window_jepa.data.event_store import InMemoryEventStore
from event_window_jepa.data.paired_window_dataset import PairedWindowDataset
from event_window_jepa.data.spatial_transforms import SharedRandomSpatialTransform
from event_window_jepa.data.types import SequenceInfo
from event_window_jepa.masks.multiblock import MultiBlockMaskGenerator
from event_window_jepa.representations.event_image import EventImage


class RecordingStore(InMemoryEventStore):
    def __init__(self) -> None:
        timestamps = np.arange(1_000, 100_001, 1_000, dtype=np.int64)
        super().__init__(
            {
                "s": {
                    "x": np.zeros_like(timestamps),
                    "y": np.zeros_like(timestamps),
                    "t_us": timestamps,
                    "polarity": np.ones_like(timestamps),
                }
            },
            {"s": SequenceInfo("s", None, 4, 4, 0, 100_000, "train")},
        )
        self.calls: list[tuple[str, int, int]] = []

    def slice(self, sequence_id: str, t_end_us: int, duration_us: int):
        self.calls.append((sequence_id, t_end_us, duration_us))
        return super().slice(sequence_id, t_end_us, duration_us)


def test_dataset_uses_one_causal_end_for_both_windows() -> None:
    store = RecordingStore()
    anchors = UniformTimeAnchorSampler(store.sequences("train"), 40, 1, seed=3)
    pairs = WindowPairSampler([10], [40], minimum_ratio=1.5)
    dataset = PairedWindowDataset(
        store,
        anchors,
        pairs,
        EventImage(normalization="none"),
        MultiBlockMaskGenerator(
            (2, 2),
            target_blocks=1,
            target_area_range=(0.25, 0.25),
            context_keep_ratio=0.5,
        ),
        SharedRandomSpatialTransform((4, 4), horizontal_flip_probability=0.0),
        seed=4,
    )
    sample = dataset[0]
    assert len(store.calls) == 1
    assert store.calls[0][1] == sample["t_end_us"]
    assert store.calls[0][2] == 40_000
    assert sample["x_context"].shape == sample["x_target"].shape == (2, 4, 4)
