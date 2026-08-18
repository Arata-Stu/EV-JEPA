from __future__ import annotations

import base64

import numpy as np

from event_window_jepa.config import ExperimentConfig
from event_window_jepa.data.anchor_sampler import UniformTimeAnchorSampler, WindowPairSampler
from event_window_jepa.data.event_store import InMemoryEventStore
from event_window_jepa.data.paired_window_dataset import PairedWindowDataset
from event_window_jepa.data.spatial_transforms import SharedRandomSpatialTransform
from event_window_jepa.data.types import SequenceInfo
from event_window_jepa.inspection import _png_data_uri, write_inspection_report
from event_window_jepa.masks.multiblock import MultiBlockMaskGenerator
from event_window_jepa.representations.event_image import EventImage


def _config() -> ExperimentConfig:
    return ExperimentConfig.from_mapping(
        {
            "data": {
                "manifest": "unused.jsonl",
                "store": "npz",
                "samples_per_epoch": 2,
                "batch_size": 2,
                "workers": 0,
                "crop_size": [4, 4],
                "horizontal_flip_probability": 0.0,
            },
            "representation": {"type": "event_image", "normalization": "none"},
            "windows": {
                "train_ms": [10],
                "target_ms": [40],
                "eval_ms": [5, 10, 40],
                "unseen_eval_ms": [5],
                "canonical_ms": 40,
                "minimum_ratio": 1.5,
                "direction": "any",
                "allow_equal": False,
            },
            "model": {"image_size": [4, 4], "patch_size": 2},
            "mask": {
                "target_blocks": 1,
                "target_area_range": [0.25, 0.25],
                "context_keep_ratio": 0.5,
            },
        }
    )


def _dataset() -> PairedWindowDataset:
    timestamps = np.arange(1_000, 101_000, 1_000, dtype=np.int64)
    store = InMemoryEventStore(
        {
            "gen1-sequence": {
                "x": np.arange(timestamps.size, dtype=np.int64) % 4,
                "y": (np.arange(timestamps.size, dtype=np.int64) // 4) % 4,
                "t_us": timestamps,
                "polarity": np.where(np.arange(timestamps.size) % 2, 1, -1),
            }
        },
        {
            "gen1-sequence": SequenceInfo(
                "gen1-sequence",
                None,
                4,
                4,
                0,
                100_000,
                dataset="gen1",
            )
        },
    )
    return PairedWindowDataset(
        store,
        UniformTimeAnchorSampler(store.sequences(), 40, 2, seed=2),
        WindowPairSampler([10], [40]),
        EventImage(normalization="none"),
        MultiBlockMaskGenerator(
            (2, 2),
            target_blocks=1,
            target_area_range=(0.25, 0.25),
            context_keep_ratio=0.5,
        ),
        SharedRandomSpatialTransform((4, 4), horizontal_flip_probability=0.0),
        seed=3,
    )


def test_png_encoder_writes_png_signature() -> None:
    uri = _png_data_uri(np.zeros((2, 3, 3), dtype=np.uint8))
    payload = base64.b64decode(uri.partition(",")[2])
    assert payload.startswith(b"\x89PNG\r\n\x1a\n")


def test_inspection_report_contains_images_and_passes_checks(tmp_path) -> None:
    output = tmp_path / "samples.html"
    report = write_inspection_report(
        _dataset(),
        _config(),
        output,
        samples=1,
        expected_dataset="gen1",
    )

    assert report["passed"] is True
    assert "data:image/png;base64," in output.read_text(encoding="utf-8")
    assert output.with_suffix(".json").is_file()
