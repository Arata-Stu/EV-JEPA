from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from event_window_jepa.config import ExperimentConfig
from event_window_jepa.data.event_store import InMemoryEventStore
from event_window_jepa.data.recurrent_window_dataset import RecurrentWindowDataset
from event_window_jepa.data.sequence_sampler import (
    MixedRecurrentBatchSampler,
    UniformSequenceClipSampler,
)
from event_window_jepa.data.spatial_transforms import SharedRandomSpatialTransform
from event_window_jepa.data.types import SequenceInfo
from event_window_jepa.masks.multiblock import MultiBlockMaskGenerator
from event_window_jepa.recurrent_inspection import (
    assert_recurrent_clip_invariants,
    inspect_mixed_recurrent_batches,
    recurrent_clip_checks,
    write_recurrent_inspection_report,
)
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
                "train_ms": [50],
                "target_ms": [50],
                "canonical_ms": 50,
                "eval_ms": [25, 50],
                "unseen_eval_ms": [25],
                "minimum_ratio": 1.0,
                "allow_equal": True,
            },
            "model": {
                "architecture": "vjepa2_1",
                "image_size": [4, 4],
                "patch_size": 2,
                "embed_dim": 32,
                "encoder_depth": 2,
                "encoder_heads": 4,
                "predictor_dim": 16,
                "predictor_depth": 1,
                "predictor_heads": 4,
                "scale_dim": 16,
                "scale_fourier_bands": 2,
                "deep_supervision_layers": [0, 1],
            },
            "recurrent": {
                "enabled": True,
                "window_ms": 50,
                "stride_ms": 50,
                "sequence_length": 2,
                "burn_in_steps": 1,
                "tbptt_steps": 1,
            },
            "mask": {
                "target_blocks": 1,
                "target_area_range": [0.25, 0.25],
                "context_keep_ratio": 0.5,
            },
            "optimization": {
                "objective": "recurrent_dense_window_jepa",
                "canonical_query_weight": 0.0,
            },
        }
    )


def _dataset() -> RecurrentWindowDataset:
    timestamps = np.arange(1_000, 150_001, 1_000, dtype=np.int64)
    indices = np.arange(timestamps.size, dtype=np.int64)
    store = InMemoryEventStore(
        {
            "gen1-sequence": {
                "x": indices % 4,
                "y": (indices // 4) % 4,
                "t_us": timestamps,
                "polarity": indices % 2,
            }
        },
        {
            "gen1-sequence": SequenceInfo(
                "gen1-sequence",
                None,
                4,
                4,
                0,
                150_000,
                dataset="gen1",
            )
        },
    )
    return RecurrentWindowDataset(
        store,
        UniformSequenceClipSampler(
            store.sequences(),
            base_window_ms=50,
            stride_ms=50,
            sequence_length=2,
            burn_in_steps=1,
            samples_per_epoch=2,
            seed=3,
        ),
        EventImage(normalization="none"),
        MultiBlockMaskGenerator(
            (2, 2),
            target_blocks=1,
            target_area_range=(0.25, 0.25),
            context_keep_ratio=0.5,
        ),
        SharedRandomSpatialTransform((4, 4), horizontal_flip_probability=0.0),
        tbptt_steps=1,
        seed=5,
    )


def _mixed_config() -> ExperimentConfig:
    config = _config()
    return replace(
        config,
        data=replace(config.data, samples_per_epoch=8, batch_size=4),
        recurrent=replace(
            config.recurrent,
            sampling="mixed",
            stream_ratio=0.5,
            tbptt_steps=2,
        ),
    )


def _mixed_dataset_and_sampler(
    *, duration_us: int = 300_000,
) -> tuple[RecurrentWindowDataset, MixedRecurrentBatchSampler]:
    timestamps = np.arange(1_000, duration_us + 1, 1_000, dtype=np.int64)
    indices = np.arange(timestamps.size, dtype=np.int64)
    events = {}
    metadata = {}
    for sequence_index in range(4):
        sequence_id = f"gen1-sequence-{sequence_index}"
        events[sequence_id] = {
            "x": indices % 4,
            "y": (indices // 4) % 4,
            "t_us": timestamps,
            "polarity": indices % 2,
        }
        metadata[sequence_id] = SequenceInfo(
            sequence_id,
            None,
            4,
            4,
            0,
            duration_us,
            dataset="gen1",
            source_recording_id=f"recording-{sequence_index}",
        )
    store = InMemoryEventStore(events, metadata)
    clip_sampler = UniformSequenceClipSampler(
        store.sequences(),
        base_window_ms=50,
        stride_ms=50,
        sequence_length=2,
        burn_in_steps=1,
        samples_per_epoch=8,
        seed=47,
    )
    dataset = RecurrentWindowDataset(
        store,
        clip_sampler,
        EventImage(normalization="none"),
        MultiBlockMaskGenerator(
            (2, 2),
            target_blocks=1,
            target_area_range=(0.25, 0.25),
            context_keep_ratio=0.5,
        ),
        SharedRandomSpatialTransform((4, 4), horizontal_flip_probability=0.5),
        tbptt_steps=2,
        seed=53,
    )
    batch_sampler = MixedRecurrentBatchSampler(
        store.sequences(),
        base_window_ms=50,
        stride_ms=50,
        sequence_length=2,
        burn_in_steps=1,
        samples_per_epoch=8,
        batch_size=4,
        stream_ratio=(1, 1),
        seed=59,
    )
    return dataset, batch_sampler


def test_recurrent_report_saves_each_timestep_and_machine_checks(tmp_path) -> None:
    output = tmp_path / "recurrent-clip.html"
    report = write_recurrent_inspection_report(
        _dataset(),
        _config(),
        output,
        expected_dataset="gen1",
    )

    assert report["passed"] is True
    assert len(report["timesteps"]) == 3
    assert output.is_file()
    assert output.with_suffix(".json").is_file()
    serialized = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert serialized["timesteps"][0]["dt_ms"] == 50.0
    assert serialized["timesteps"][0]["loss_mask"] is False
    assets = tmp_path / "recurrent-clip_assets"
    images = sorted(assets.glob("*.png"))
    assert len(images) == 12
    assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in images)
    assert "step-00-mask-overlay.png" in output.read_text(encoding="utf-8")


def test_sequence_report_accepts_feedforward_temporal_model(tmp_path) -> None:
    config = _config()
    feedforward_config = replace(
        config,
        recurrent=replace(
            config.recurrent,
            enabled=False,
            sequence_loader=True,
            temporal_model="feedforward",
        ),
        optimization=replace(
            config.optimization,
            objective="sequence_dense_window_jepa",
        ),
    )

    report = write_recurrent_inspection_report(
        _dataset(),
        feedforward_config,
        tmp_path / "feedforward-sequence.html",
        expected_dataset="gen1",
    )

    assert report["passed"] is True


def test_assertions_detect_timestamp_damage_and_validate_future_sampler_metadata() -> None:
    dataset = _dataset()
    sample, debug = dataset.sample_with_debug(0)
    assert_recurrent_clip_invariants(sample, debug, _config(), "gen1")

    metadata_sample = dict(sample)
    metadata_sample.update(
        {
            "sampling_mode": "random",
            "state_reset": True,
            "augmentation_id": "shared-geometry-7",
        }
    )
    assert all(
        check.passed
        for check in recurrent_clip_checks(metadata_sample, debug, _config())
    )

    damaged = dict(sample)
    damaged["t_end_us"] = sample["t_end_us"].flip(0)
    with pytest.raises(AssertionError, match="timestamp"):
        assert_recurrent_clip_invariants(damaged, debug, _config())

    metadata_sample["state_reset"] = False
    checks = recurrent_clip_checks(metadata_sample, debug, _config())
    assert any(
        check.name == "random・stream sampler metadata" and not check.passed
        for check in checks
    )


def test_mixed_report_materializes_two_batches_and_checks_stream_continuity(
    tmp_path,
) -> None:
    dataset, batch_sampler = _mixed_dataset_and_sampler()
    config = _mixed_config()
    inspection = inspect_mixed_recurrent_batches(
        dataset,
        batch_sampler,
        config,
        epoch=2,
        expected_dataset="gen1",
    )

    assert all(check.passed for check in inspection.checks)
    assert len(inspection.requests) == 2
    assert all(len(batch) == 4 for batch in inspection.requests)
    for lane in range(batch_sampler.stream_batch_size):
        first = inspection.requests[0][lane]
        second = inspection.requests[1][lane]
        first_sample = inspection.samples[0][lane]
        second_sample = inspection.samples[1][lane]
        assert first.stream_id == second.stream_id
        assert first.state_reset is True
        assert second.state_reset is False
        assert first_sample["stream_id"] == second_sample["stream_id"]
        assert first_sample["state_reset"].item() is True
        assert second_sample["state_reset"].item() is False
        assert second.clip.t_end_us[0] == first.clip.t_end_us[-1] + 50_000
        assert first.augmentation_id == second.augmentation_id
        assert (
            inspection.debug_samples[0][lane].spatial_transform
            == inspection.debug_samples[1][lane].spatial_transform
        )
    assert all(
        request.state_reset
        for batch in inspection.requests
        for request in batch[batch_sampler.stream_batch_size :]
    )

    output = tmp_path / "mixed-recurrent.html"
    report = write_recurrent_inspection_report(
        dataset,
        config,
        output,
        epoch=2,
        expected_dataset="gen1",
        mixed_batch_sampler=batch_sampler,
    )
    assert report["passed"] is True
    assert report["inspection_mode"] == "mixed_two_batch"
    assert len(report["mixed_batches"]) == 2
    assert all(len(batch["items"]) == 4 for batch in report["mixed_batches"])
    serialized = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert serialized["inspection_mode"] == "mixed_two_batch"
    assert len(serialized["mixed_batches"]) == 2
    html_text = output.read_text(encoding="utf-8")
    assert "mixed sampler — 連続2 batch" in html_text
    assert "stream batch間timestamp・reset整合" in html_text


def test_mixed_checks_accept_a_legal_stream_boundary_between_batches() -> None:
    dataset, batch_sampler = _mixed_dataset_and_sampler(duration_us=150_000)
    inspection = inspect_mixed_recurrent_batches(
        dataset,
        batch_sampler,
        _mixed_config(),
    )

    assert all(check.passed for check in inspection.checks)
    stream_count = batch_sampler.stream_batch_size
    assert all(
        inspection.samples[1][lane]["state_reset"].item()
        for lane in range(stream_count)
    )
    assert all(
        inspection.samples[0][lane]["sequence_id"]
        != inspection.samples[1][lane]["sequence_id"]
        for lane in range(stream_count)
    )


def test_mixed_checks_reject_dataset_metadata_that_disagrees_with_request(
    monkeypatch,
) -> None:
    dataset, batch_sampler = _mixed_dataset_and_sampler()
    original = dataset.sample_with_debug

    def damaged_sample_with_debug(index):
        sample, debug = original(index)
        if index.sampling_mode == "stream" and not index.state_reset:
            sample = dict(sample)
            sample["state_reset"] = True
        return sample, debug

    monkeypatch.setattr(dataset, "sample_with_debug", damaged_sample_with_debug)
    inspection = inspect_mixed_recurrent_batches(
        dataset,
        batch_sampler,
        _mixed_config(),
    )
    failed_names = {check.name for check in inspection.checks if not check.passed}
    assert "sampler request・Dataset出力・debug一致" in failed_names
    assert "stream batch間timestamp・reset整合" in failed_names
