from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from event_window_jepa.config import ExperimentConfig
from event_window_jepa.evaluation.future_feature_data import (
    materialize_future_feature_samples,
    validate_future_feature_sample,
)
from event_window_jepa.evaluation.future_feature_visualization import (
    write_future_feature_report,
)
from event_window_jepa.train.pretrain import build_model


def _write_manifest(root: Path) -> Path:
    event_path = root / "events.npz"
    timestamps = np.arange(1_000, 100_000, 1_000, dtype=np.int64)
    np.savez(
        event_path,
        x=(np.arange(len(timestamps), dtype=np.int64) % 6),
        y=((np.arange(len(timestamps), dtype=np.int64) * 3) % 6),
        t_us=timestamps,
        polarity=(np.arange(len(timestamps), dtype=np.int8) % 2),
    )
    manifest = root / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sequence_id": "gen1__visualization",
                "path": event_path.name,
                "height": 6,
                "width": 6,
                "t_start_us": 0,
                "t_end_us": 100_000,
                "split": "train",
                "dataset": "gen1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def _config(manifest: str = "/checkpoint/host/original.jsonl") -> ExperimentConfig:
    return ExperimentConfig.from_mapping(
        {
            "data": {
                "manifest": manifest,
                "store": "npz",
                "split": "train",
                "samples_per_epoch": 8,
                "batch_size": 2,
                "workers": 0,
                "sequence_sampling": "sequence_balanced",
                "crop_size": [4, 4],
                "horizontal_flip_probability": 0.5,
            },
            "representation": {
                "type": "voxel_grid",
                "temporal_bins": 2,
                "split_polarity": True,
                "normalization": "log1p",
            },
            "windows": {
                "train_ms": [10],
                "eval_ms": [5, 10],
                "unseen_eval_ms": [5],
                "target_ms": [10],
                "canonical_ms": 10,
                "minimum_ratio": 1.0,
                "direction": "any",
                "allow_equal": True,
            },
            "model": {
                "architecture": "vjepa2_1",
                "image_size": [4, 4],
                "patch_size": 2,
                "embed_dim": 16,
                "encoder_depth": 1,
                "encoder_heads": 1,
                "deep_supervision_layers": [0],
                "predictor_dim": 8,
                "predictor_depth": 1,
                "predictor_heads": 1,
                "scale_dim": 8,
                "scale_fourier_bands": 2,
                "condition_on_scale": True,
            },
            "recurrent": {
                "sequence_loader": True,
                "temporal_model": "conv_lstm",
                "return_patch_event_activity": True,
                "recurrent_placement": "post_encoder",
                "prediction_horizon_steps": 1,
                "kernel_size": 3,
                # This deliberately exercises the direct-index path despite a
                # mixed training configuration.
                "sampling": "mixed",
                "stream_ratio": 0.5,
                "window_ms": 10,
                "stride_ms": 10,
                "sequence_length": 2,
                "burn_in_steps": 1,
                "tbptt_steps": 2,
            },
            "mask": {
                "target_blocks": 1,
                "target_area_range": [0.25, 0.25],
                "target_aspect_range": [1.0, 1.0],
                "context_keep_ratio": 0.5,
                "activity_aware_probability": 0.0,
            },
            "future_prediction": {
                "frame_sigreg_weight": 0.0,
                "temporal_sigreg_weight": 0.0,
                "allow_unregularized": True,
            },
            "optimization": {
                "objective": "recurrent_future_jepa",
                "epochs": 2,
                "learning_rate": 0.001,
                "minimum_learning_rate": 0.0001,
                "warmup_epochs": 1,
                "weight_decay": 0.0,
                "target_ema_start": 0.9,
                "target_ema_end": 0.99,
                "precision": "fp32",
                "gradient_clip": 1.0,
                "variance_weight": 0.0,
                "covariance_weight": 0.0,
                "canonical_query_weight": 0.0,
            },
            "runtime": {"seed": 23, "output_dir": "outputs/test"},
        }
    )


def test_materialization_uses_manifest_override_and_direct_deterministic_indices(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path)
    source_config = _config()

    first = materialize_future_feature_samples(
        source_config,
        [3, 1],
        manifest_override=manifest,
        epoch=2,
    )
    second = materialize_future_feature_samples(
        source_config,
        [3, 1],
        manifest_override=manifest,
        epoch=2,
    )

    assert source_config.data.manifest == "/checkpoint/host/original.jsonl"
    assert first.config.data.manifest == str(manifest.resolve())
    assert first.epoch == 2
    assert [clip.sample_index for clip in first.clips] == [3, 1]
    assert first.dataset.online_steps == 3
    assert first.dataset.total_steps == 4
    for left, right in zip(first.clips, second.clips, strict=True):
        assert left.sample["sampling_mode"] == "random"
        assert left.sample["stream_id"] == ""
        assert left.sample["state_reset"].item() is True
        assert left.sample["sequence_id"] == right.sample["sequence_id"]
        assert torch.equal(left.sample["t_end_us"], right.sample["t_end_us"])
        assert torch.equal(left.sample["x"], right.sample["x"])
        assert torch.equal(left.sample["x_future"], right.sample["x_future"])
        assert torch.equal(
            left.sample["future_patch_event_activity"],
            right.sample["future_patch_event_activity"],
        )
        assert left.debug.spatial_transform == right.debug.spatial_transform


def test_materialized_future_fields_follow_debug_lookahead(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    result = materialize_future_feature_samples(
        _config(),
        [0],
        manifest_override=manifest,
    )
    clip = result.clips[0]
    sample = clip.sample
    debug_timestamps = torch.tensor(
        [window.t_end_us for window in clip.debug.windows], dtype=torch.int64
    )

    assert torch.equal(sample["t_end_us"], debug_timestamps[:3])
    assert torch.equal(sample["future_t_end_us"], debug_timestamps[1:4])
    assert torch.equal(sample["future_t_end_us"] - sample["t_end_us"], torch.full((3,), 10_000))
    assert torch.equal(sample["x_future"][:-1], sample["x"][1:])
    assert torch.equal(
        sample["future_patch_event_activity"][:-1],
        sample["patch_event_activity"][1:],
    )


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("future_t_end_us", "future_t_end_us"),
        ("x_future", "x_future"),
        ("future_patch_event_activity", "future_patch_event_activity"),
    ),
)
def test_validation_rejects_misaligned_future_fields(
    tmp_path: Path,
    field: str,
    message: str,
) -> None:
    manifest = _write_manifest(tmp_path)
    result = materialize_future_feature_samples(
        _config(),
        [0],
        manifest_override=manifest,
    )
    clip = result.clips[0]
    damaged = dict(clip.sample)
    damaged[field] = clip.sample[field].clone()
    damaged[field][-1].add_(1)

    with pytest.raises(ValueError, match=message):
        validate_future_feature_sample(result.config, result.dataset, damaged, clip.debug)


def test_materialization_rejects_invalid_index_requests(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    config = _config()

    with pytest.raises(ValueError, match="cannot be empty"):
        materialize_future_feature_samples(config, [], manifest_override=manifest)
    with pytest.raises(ValueError, match="unique"):
        materialize_future_feature_samples(config, [1, 1], manifest_override=manifest)
    with pytest.raises(ValueError, match="outside"):
        materialize_future_feature_samples(config, [8], manifest_override=manifest)


def test_future_feature_report_runs_the_full_tiny_pipeline(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    checkpoint_config = _config()
    materialization = materialize_future_feature_samples(
        checkpoint_config,
        [0, 1, 2, 3],
        manifest_override=manifest,
    )
    model = build_model(checkpoint_config).eval()
    output = tmp_path / "future-features.html"

    report = write_future_feature_report(
        model,
        checkpoint_config,
        materialization,
        tmp_path / "checkpoint.pt",
        output,
        device="cpu",
        display_sample_index=0,
    )

    assert report["schema"] == "event-window-jepa-future-feature-visualization-v2"
    assert set(report["conditions"]) == {
        "correct",
        "history_shuffled",
        "history_reversed",
        "history_replaced",
        "reset",
        "unrelated_target",
    }
    assert len(report["steps"]) == 8
    assert report["steps"][0]["history_shuffled_cosine_error"] is None
    assert report["steps"][0]["history_reversed_cosine_error"] is None
    assert report["steps"][0]["history_replaced_cosine_error"] is not None
    assert sorted(report["history_replacement_clip_permutation"]) == [0, 1, 2, 3]
    assert output.is_file()
    assert "共通PCA" in output.read_text(encoding="utf-8")
    json_report = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert json_report["sample_set_id"] == report["sample_set_id"]
    assets = sorted((tmp_path / "future-features_assets").rglob("*.png"))
    assert assets
    assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in assets)
