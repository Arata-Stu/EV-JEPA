from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from event_window_jepa.config import ExperimentConfig
from event_window_jepa.data.packed_events import PackedEventBatch
from event_window_jepa.evaluation.cmax_flow_visualization import (
    _require_unchanged_content_identity,
    _snapshot_file_identity,
    _validate_output_input_collisions,
    compare_cmax_flow_conditions,
    compute_step_warp_iwes,
    extract_cmax_flow_records,
    flow_statistics,
    flow_to_hsv_rgb,
    quiver_overlay_rgb,
    write_cmax_flow_report,
)
from event_window_jepa.evaluation.future_feature_data import (
    materialize_future_feature_samples,
)
from event_window_jepa.losses.cmax import TamingCMaxLoss
from event_window_jepa.train.pretrain import build_model


def _moving_events() -> PackedEventBatch:
    """Two windows from a one-pixel-per-window horizontal translation."""

    local_time = torch.tensor([0.125, 0.375, 0.625, 0.875] * 2)
    time_index = torch.tensor([0] * 4 + [1] * 4, dtype=torch.int64)
    global_time = time_index.float() + local_time
    counts = torch.tensor([4, 4], dtype=torch.int64)
    return PackedEventBatch(
        x=3.0 + global_time,
        y=torch.full((8,), 3.0),
        t=local_time,
        polarity=torch.ones(8),
        batch_index=torch.zeros(8, dtype=torch.int64),
        time_index=time_index,
        window_offsets=torch.tensor([0, 4, 8], dtype=torch.int64),
        window_counts=counts,
        batch_size=1,
        time_steps=2,
        height=8,
        width=8,
    )


def _write_manifest(root: Path) -> Path:
    event_path = root / "events.npz"
    timestamps = np.arange(1_000, 100_000, 1_000, dtype=np.int64)
    indices = np.arange(len(timestamps), dtype=np.int64)
    np.savez(
        event_path,
        x=indices % 6,
        y=(indices * 3) % 6,
        t_us=timestamps,
        polarity=(indices.astype(np.int8) % 2),
    )
    manifest = root / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sequence_id": "gen1__cmax_visualization",
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


def test_cmax_report_identity_and_input_collision_guards(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint-v1")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")

    identity = _snapshot_file_identity(checkpoint, label="checkpoint")
    _require_unchanged_content_identity(
        checkpoint, identity, label="checkpoint"
    )
    checkpoint.write_bytes(b"checkpoint-v2")
    with pytest.raises(RuntimeError, match="content changed"):
        _require_unchanged_content_identity(
            checkpoint, identity, label="checkpoint"
        )

    checkpoint_json = tmp_path / "report.json"
    checkpoint_json.write_bytes(b"checkpoint")
    with pytest.raises(ValueError, match="collide with the checkpoint"):
        _validate_output_input_collisions(
            tmp_path / "report.html",
            checkpoint=checkpoint_json,
            manifest=manifest,
        )

    assets = tmp_path / "qualitative_assets"
    assets.mkdir()
    nested_manifest = assets / "manifest.jsonl"
    nested_manifest.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="collide with the manifest"):
        _validate_output_input_collisions(
            tmp_path / "qualitative.html",
            checkpoint=checkpoint,
            manifest=nested_manifest,
        )


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
                "horizontal_flip_probability": 0.0,
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
                "sampling": "random",
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
            "cmax": {
                "enabled": True,
                "weight": 0.05,
                "smoothness_weight": 0.0,
                "hidden_dim": 8,
                "head_depth": 1,
                "reference_mode": "both",
                "temporal_scales": [1],
                "min_events": 1,
                "max_events_per_window": 8,
                "flow_scale": 0.1,
                "max_displacement": 4.0,
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
            "runtime": {"seed": 29, "output_dir": "outputs/test"},
        }
    )


def test_flow_hsv_uses_fixed_checkpoint_scale_and_documented_directions() -> None:
    flow = torch.tensor(
        [
            [[1.0, 0.0, -1.0, 0.0, 0.5]],
            [[0.0, -1.0, 0.0, 1.0, 0.0]],
        ]
    )

    rgb = flow_to_hsv_rgb(flow, max_displacement=1.0, image_size=(1, 5))

    assert rgb.shape == (1, 5, 3)
    assert rgb.dtype == np.uint8
    right, up, left, down, half_right = rgb[0]
    assert tuple(right) == (255, 0, 0)
    assert up[2] > up[0] > up[1]
    assert left[1] == left[2] > left[0]
    assert down[1] > down[0] > down[2]
    assert tuple(half_right) == (128, 0, 0)


def test_flow_statistics_are_finite_and_use_component_saturation() -> None:
    flow = torch.tensor(
        [
            [[0.0, 5.0, 10.0]],
            [[0.0, 0.0, 0.0]],
        ]
    )

    statistics = flow_statistics(flow, max_displacement=10.0)

    assert statistics["mean_magnitude"] == pytest.approx(5.0)
    assert statistics["p95_magnitude"] == pytest.approx(9.5)
    assert statistics["max_magnitude"] == pytest.approx(10.0)
    assert statistics["saturation_fraction"] == pytest.approx(1.0 / 3.0)
    assert all(np.isfinite(value) for value in statistics.values())

    damaged = flow.clone()
    damaged[0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="finite|NaN|infinity"):
        flow_statistics(damaged, max_displacement=10.0)


def test_quiver_nonzero_flow_draws_without_mutating_the_event_image() -> None:
    event_image = np.zeros((12, 12, 3), dtype=np.uint8)
    flow = torch.tensor([[[3.0]], [[0.0]]])

    overlay = quiver_overlay_rgb(event_image, flow)

    assert overlay.shape == event_image.shape
    assert overlay.dtype == np.uint8
    assert np.any(overlay != 0)
    assert np.array_equal(event_image, np.zeros_like(event_image))


def test_known_translation_improves_cmax_over_zero_and_wrong_flow() -> None:
    events = _moving_events()
    learned = torch.zeros(1, 2, 2, 1, 1)
    learned[:, :, 0] = 1.0
    wrong = -learned
    criterion = TamingCMaxLoss(
        reference_mode="both",
        temporal_scales=(1,),
        min_events=4,
    )

    conditions = compare_cmax_flow_conditions(
        learned,
        events,
        criterion,
        shuffled_flow_maps=wrong,
        max_displacement=4.0,
    )

    assert set(conditions) == {"learned", "zero", "sample_shuffled"}
    assert conditions["learned"]["focus_loss"] < conditions["zero"]["focus_loss"]
    assert conditions["learned"]["focus_loss"] < conditions["sample_shuffled"][
        "focus_loss"
    ]
    for metrics in conditions.values():
        assert all(np.isfinite(value) for value in metrics.values())
        assert metrics["valid_event_count"] == 8
        assert metrics["valid_window_fraction"] == 1


def test_known_translation_concentrates_single_window_iwes() -> None:
    events = _moving_events().select_time_range(0, 1)
    flow = torch.zeros(2, 1, 1)
    flow[0] = 1.0

    result = compute_step_warp_iwes(flow, events)

    torch.testing.assert_close(result.unwarped_iwe.sum(), torch.tensor(4.0))
    torch.testing.assert_close(result.past_iwe.sum(), torch.tensor(4.0))
    torch.testing.assert_close(result.future_iwe.sum(), torch.tensor(4.0))
    assert (
        result.past_occupied_pixel_fraction
        < result.unwarped_occupied_pixel_fraction
    )
    assert (
        result.future_occupied_pixel_fraction
        < result.unwarped_occupied_pixel_fraction
    )
    assert result.past_retained_event_fraction == 1
    assert result.future_retained_event_fraction == 1


def test_cmax_flow_report_runs_tiny_pipeline_and_preserves_alignment(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path)
    checkpoint_config = _config()
    materialization = materialize_future_feature_samples(
        checkpoint_config,
        [0, 1],
        manifest_override=manifest,
    )
    model = build_model(checkpoint_config).eval()

    extraction = extract_cmax_flow_records(
        model,
        materialization,
        device="cpu",
        flow_shuffle_seed=3,
    )

    assert len(extraction.clips) == 2
    assert extraction.flow_shuffle_permutation == (1, 0)
    assert set(extraction.conditions) == {"learned", "zero", "sample_shuffled"}
    assert extraction.focus_improvements["zero_minus_learned"] == pytest.approx(
        extraction.conditions["zero"]["focus_loss"]
        - extraction.conditions["learned"]["focus_loss"]
    )
    assert extraction.focus_improvements[
        "sample_shuffled_minus_learned"
    ] == pytest.approx(
        extraction.conditions["sample_shuffled"]["focus_loss"]
        - extraction.conditions["learned"]["focus_loss"]
    )
    clip_identities = tuple(
        (
            record.sequence_id,
            tuple(
                int(clip.sample["future_t_end_us"][step])
                for step in record.supervised_steps
            ),
        )
        for record, clip in zip(
            extraction.clips,
            materialization.clips,
            strict=True,
        )
    )
    assert len(set(clip_identities)) == len(clip_identities)
    assert all(
        clip_identities[source] != clip_identities[target]
        for source, target in enumerate(extraction.flow_shuffle_permutation)
    )
    for record, clip in zip(extraction.clips, materialization.clips, strict=True):
        assert record.sample_index == clip.sample_index
        assert record.sequence_id == clip.sample["sequence_id"]
        assert record.supervised_steps == (1, 2)
        assert record.flow_maps.shape == (2, 2, 2, 2)
        assert len(record.cmax_windows) == 2
        for sequence_step, online_step in enumerate(record.supervised_steps):
            raw_window = clip.debug.windows[online_step]
            cmax_window = record.cmax_windows[sequence_step]
            assert (cmax_window.t_start_us, cmax_window.t_end_us) == (
                raw_window.t_start_us,
                raw_window.t_end_us,
            )
            assert cmax_window.event_count == min(raw_window.event_count, 8)

    assert len(extraction.steps) == 4
    for step in extraction.steps:
        clip = materialization.clips[step.clip_position]
        raw_window = clip.debug.windows[step.online_step]
        assert step.sample_index == clip.sample_index
        assert step.sequence_id == clip.sample["sequence_id"]
        assert (step.t_start_us, step.t_end_us) == (
            raw_window.t_start_us,
            raw_window.t_end_us,
        )
        assert step.raw_event_count == raw_window.event_count
        assert step.cmax_event_count == min(step.raw_event_count, 8)
        assert step.cmax_window_valid == (step.cmax_event_count >= 1)

    reversed_materialization = replace(
        materialization,
        clips=tuple(reversed(materialization.clips)),
    )
    reversed_extraction = extract_cmax_flow_records(
        model,
        reversed_materialization,
        device="cpu",
        flow_shuffle_seed=3,
    )
    forward_flow = {
        clip.sample_index: clip.flow_maps for clip in extraction.clips
    }
    reversed_flow = {
        clip.sample_index: clip.flow_maps for clip in reversed_extraction.clips
    }
    assert forward_flow.keys() == reversed_flow.keys()
    for sample_index in forward_flow:
        torch.testing.assert_close(
            forward_flow[sample_index],
            reversed_flow[sample_index],
        )

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"fixed checkpoint identity for report test")
    output = tmp_path / "cmax-flow.html"
    report = write_cmax_flow_report(
        model,
        checkpoint_config,
        materialization,
        checkpoint,
        output,
        device="cpu",
        display_sample_index=0,
        all_steps=True,
        flow_shuffle_seed=3,
    )

    assert report["schema"] == "event-window-jepa-cmax-flow-visualization-v1"
    assert report["checkpoint_artifact"]["path"] == str(checkpoint.resolve())
    assert report["checkpoint_artifact"]["bytes"] == checkpoint.stat().st_size
    assert report["checkpoint_artifact"]["sha256"] == hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest()
    manifest = Path(materialization.config.data.manifest).resolve()
    assert report["manifest_artifact"]["path"] == str(manifest)
    assert report["manifest_artifact"]["bytes"] == manifest.stat().st_size
    assert report["manifest_artifact"]["sha256"] == hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest()
    assert report["sample_indices"] == [0, 1]
    assert set(report["conditions"]) == {"learned", "zero", "sample_shuffled"}
    assert report["focus_improvements"]["zero_minus_learned"] == pytest.approx(
        report["conditions"]["zero"]["focus_loss"]
        - report["conditions"]["learned"]["focus_loss"]
    )
    assert report["focus_improvements"][
        "sample_shuffled_minus_learned"
    ] == pytest.approx(
        report["conditions"]["sample_shuffled"]["focus_loss"]
        - report["conditions"]["learned"]["focus_loss"]
    )
    assert report["metric_direction"]["focus_loss"] == "lower_is_better"
    assert report["state_scope"] == {
        "mode": "direct_clip_reset_then_configured_burn_in",
        "burn_in_steps": 1,
        "reconstructs_prior_stream_lane_state": False,
    }
    assert all(step["cmax_window_valid"] for step in report["steps"])
    assert output.is_file()
    document = output.read_text(encoding="utf-8")
    assert "CMax flow" in document
    assert "zero flow" in document
    json_report = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert json_report == report
    assets = sorted((tmp_path / "cmax-flow_assets").rglob("*.png"))
    assert assets
    assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in assets)
