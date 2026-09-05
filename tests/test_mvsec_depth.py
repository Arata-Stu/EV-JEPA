from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional

from event_window_jepa.downstream import mvsec_depth as mvsec_depth_module
from event_window_jepa.downstream.mvsec_depth import (
    DepthEvaluationAccumulator,
    MVSECLogDepthHead,
    _alignment_metadata,
    _atomic_json,
    _checkpoint_payload,
    _depth_target_identity,
    _parse_args,
    _require_recordings,
    _split_identity,
    _target_reference_sha256,
    masked_log_depth_smooth_l1,
)
from event_window_jepa.downstream.mvsec_geometry import (
    MVSECGeometrySource,
    MVSECTargetReference,
    _project_native_map,
    center_crop_parameters,
    dense_patch_prediction,
    depth_metric_sums,
    finalize_depth_metrics,
)


def _source(recording: str = "outdoor_day2") -> MVSECGeometrySource:
    return MVSECGeometrySource(
        sequence_id=f"mvsec__{recording}__left",
        ground_truth_path=Path(f"/{recording}_gt.hdf5"),
        target_dataset="/davis/left/depth_image_raw",
        timestamp_dataset="/davis/left/depth_image_raw_ts",
        source_time_origin_us=100_000_000,
        t_start_us=0,
        t_end_us=60_000_000,
        camera="left",
    )


def _reference(
    target_index: int,
    label_timestamp_us: int,
    event_window_end_us: int | None = None,
) -> MVSECTargetReference:
    return MVSECTargetReference(
        source_index=0,
        target_index=target_index,
        label_timestamp_us=label_timestamp_us,
        event_window_end_us=(
            label_timestamp_us
            if event_window_end_us is None
            else event_window_end_us
        ),
        flow_interval_us=0,
    )


def test_log_depth_head_decodes_patch_tokens_then_upsamples() -> None:
    head = MVSECLogDepthHead(8, hidden_dim=4, initial_depth_m=5.0)
    tokens = torch.zeros(2, 17 * 22, 8)

    patch_log_depth = head(tokens, (17, 22))
    dense_log_depth = dense_patch_prediction(patch_log_depth, (272, 352))

    assert patch_log_depth.shape == (2, 1, 17, 22)
    assert dense_log_depth.shape == (2, 1, 272, 352)
    assert torch.allclose(
        patch_log_depth,
        torch.full_like(patch_log_depth, math.log(5.0)),
    )
    with pytest.raises(ValueError, match="token count"):
        head(tokens[:, :-1], (17, 22))


def test_log_depth_loss_drops_samples_with_fewer_than_ten_valid_pixels() -> None:
    predicted = torch.zeros(2, 1, 3, 4)
    target = torch.full((2, 3, 4), 2.0)
    valid = torch.zeros(2, 3, 4, dtype=torch.bool)
    valid[0].flatten()[:10] = True
    valid[1].flatten()[:9] = True

    loss, sample_count, pixel_count = masked_log_depth_smooth_l1(
        predicted,
        target,
        valid,
        beta=0.1,
    )

    assert loss is not None
    expected = functional.smooth_l1_loss(
        torch.zeros(10),
        torch.full((10,), math.log(2.0)),
        beta=0.1,
    )
    assert torch.allclose(loss, expected)
    assert sample_count == 1
    assert pixel_count == 10


def test_depth_metrics_report_distinct_pixel_and_sample_means_without_scaling() -> None:
    accumulator = DepthEvaluationAccumulator()
    accumulator.update(
        prediction=np.full((2, 5), 2.0),
        target=np.ones((2, 5)),
        valid=np.ones((2, 5), dtype=np.bool_),
    )
    accumulator.update(
        prediction=np.full((2, 10), 2.0),
        target=np.full((2, 10), 2.0),
        valid=np.ones((2, 10), dtype=np.bool_),
    )
    accumulator.update(
        prediction=np.ones((3, 3)),
        target=np.ones((3, 3)),
        valid=np.ones((3, 3), dtype=np.bool_),
    )

    result = accumulator.finalize()
    pixel = result["pixel_average"]
    sample = result["sample_average"]

    assert pixel["AbsRel"] == pytest.approx(1.0 / 3.0)
    assert sample["AbsRel"] == pytest.approx(0.5)
    assert pixel["RMSE"] == pytest.approx(math.sqrt(1.0 / 3.0))
    assert sample["RMSE"] == pytest.approx(0.5)
    assert pixel["RMSE_log"] == pytest.approx(math.log(2.0) / math.sqrt(3.0))
    assert sample["RMSE_log"] == pytest.approx(math.log(2.0) / 2.0)
    assert pixel["SILog"] == pytest.approx(math.sqrt(2.0) * math.log(2.0) / 3.0)
    assert sample["SILog"] == pytest.approx(0.0)
    assert pixel["F3_SILog"] == pytest.approx(math.sqrt(5.0 / 18.0) * math.log(2.0))
    assert sample["F3_SILog"] == pytest.approx(math.log(2.0) / (2.0 * math.sqrt(2.0)))
    assert pixel["F3_SqRel"] == pytest.approx(1.0 / 3.0)
    assert sample["F3_SqRel"] == pytest.approx(0.5)
    assert pixel["delta1"] == pytest.approx(2.0 / 3.0)
    assert sample["delta1"] == pytest.approx(0.5)
    assert pixel["MAE_depth_lt_10m"] == pytest.approx(1.0 / 3.0)
    assert sample["MAE_depth_lt_10m"] == pytest.approx(0.5)
    assert pixel["valid_pixels"] == 30
    assert sample["evaluated_samples"] == 2
    assert result["skipped_samples_valid_lt_10"] == 1


def test_f3_legacy_depth_metric_definitions_are_reported_separately() -> None:
    target = np.full((2, 5), 2.0)
    prediction = np.full((2, 5), 4.0)
    valid = np.ones((2, 5), dtype=np.bool_)

    metrics = finalize_depth_metrics(depth_metric_sums(prediction, target, valid))

    assert metrics["SqRel"] == pytest.approx(2.0)
    assert metrics["F3_SqRel"] == pytest.approx(1.0)
    assert metrics["SILog"] == pytest.approx(0.0)
    assert metrics["F3_SILog"] == pytest.approx(math.log(2.0) / math.sqrt(2.0))


def test_center_padding_cannot_become_valid_raw_depth() -> None:
    transform = center_crop_parameters((272, 352))
    projected = _project_native_map(np.ones((260, 346), np.float32), transform)
    valid = np.isfinite(projected) & (projected > 0.1) & (projected < 80.0)

    assert (transform.y0, transform.x0) == (-6, -3)
    assert projected.shape == (272, 352)
    assert int(valid.sum()) == 260 * 346
    assert not bool(valid[:6].any())
    assert not bool(valid[266:].any())
    assert not bool(valid[:, :3].any())
    assert not bool(valid[:, 349:].any())
    assert bool(valid[6:266, 3:349].all())


def test_target_hash_and_alignment_metadata_are_explicit_and_stable() -> None:
    sources = (_source(),)
    causal = (_reference(3, 2_000_000), _reference(4, 2_050_000))
    centered = (
        _reference(3, 2_000_000, 2_025_000),
        _reference(4, 2_050_000, 2_075_000),
    )

    digest = _target_reference_sha256(sources, causal)
    assert len(digest) == 64
    assert digest == _target_reference_sha256(sources, causal)
    assert digest != _target_reference_sha256(sources, causal[:1])
    assert _alignment_metadata("causal", causal) == {
        "alignment": "causal",
        "causal": True,
        "uses_future_events": False,
        "future_event_use_us": {"minimum": 0, "maximum": 0, "mean": 0.0},
    }
    centered_metadata = _alignment_metadata("f3_centered", centered)
    assert centered_metadata["causal"] is False
    assert centered_metadata["uses_future_events"] is True
    assert centered_metadata["future_event_use_us"] == {
        "minimum": 25_000,
        "maximum": 25_000,
        "mean": 25_000.0,
    }


def test_recording_protocol_and_cli_defaults() -> None:
    train = _require_recordings(
        (_source("outdoor_day2"),),
        allowed=("outdoor_day2",),
        role="training",
    )
    assert tuple(train) == ("outdoor_day2",)
    with pytest.raises(ValueError, match="disallowed recording"):
        _require_recordings(
            (_source("outdoor_day1"),),
            allowed=("outdoor_day2",),
            role="training",
        )

    args = _parse_args(
        [
            "--checkpoint",
            "encoder.pt",
            "--train-manifest",
            "day2.jsonl",
            "--eval-manifest",
            "day1.jsonl",
            "--output-dir",
            "outputs/depth",
        ]
    )
    assert args.alignment == "causal"
    assert args.min_depth == 0.1
    assert args.max_depth == 80.0
    assert args.protocol_stage == "final"
    assert args.dev_fraction == 0.2
    assert args.dev_guard_ms is None
    assert args.history_steps is None


def test_protocol_json_is_written_atomically(tmp_path: Path) -> None:
    destination = tmp_path / "protocol.json"
    _atomic_json({"alignment": "causal", "uses_future_events": False}, destination)

    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "alignment": "causal",
        "uses_future_events": False,
    }
    assert not list(tmp_path.glob("*.partial"))


def test_depth_target_identity_reuses_only_matching_download_sidecar(
    tmp_path: Path,
    monkeypatch,
) -> None:
    target = tmp_path / "outdoor_day1_gt.hdf5"
    target.write_bytes(b"synthetic depth target")
    stat_result = target.stat()
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    monkeypatch.setitem(
        mvsec_depth_module.MVSEC_OFFICIAL_GT_ARTIFACTS,
        target.name,
        ("synthetic-file-id", stat_result.st_size),
    )
    sidecar = target.with_name(target.name + ".verified.json")
    sidecar.write_text(
        json.dumps(
            {
                "metadata_version": 1,
                "status": "verified",
                "filename": target.name,
                "kind": "gt_hdf5",
                "file_id": "synthetic-file-id",
                "expected_bytes": stat_result.st_size,
                "size_bytes": stat_result.st_size,
                "mtime_ns": stat_result.st_mtime_ns,
                "sha256": digest,
                "publisher_checksum_available": False,
            }
        ),
        encoding="utf-8",
    )

    def unexpected_stream_hash(path: Path) -> str:
        raise AssertionError(f"matching sidecar should avoid rehashing {path}")

    monkeypatch.setattr(mvsec_depth_module, "_sha256_file", unexpected_stream_hash)
    identity = _depth_target_identity(target, {})

    assert identity["sha256"] == digest
    assert identity["sha256_origin"] == "download_verified_sidecar"
    assert identity["file_id"] == "synthetic-file-id"
    assert identity["expected_bytes"] == len(b"synthetic depth target")


def test_depth_target_identity_hashes_when_sidecar_is_stale(
    tmp_path: Path,
    monkeypatch,
) -> None:
    target = tmp_path / "outdoor_day2_gt.hdf5"
    target.write_bytes(b"fallback streamed identity")
    stat_result = target.stat()
    monkeypatch.setitem(
        mvsec_depth_module.MVSEC_OFFICIAL_GT_ARTIFACTS,
        target.name,
        ("synthetic-file-id", stat_result.st_size),
    )
    target.with_name(target.name + ".verified.json").write_text(
        json.dumps(
            {
                "metadata_version": 1,
                "status": "verified",
                "filename": target.name,
                "kind": "gt_hdf5",
                "file_id": "synthetic-file-id",
                "expected_bytes": stat_result.st_size,
                "size_bytes": stat_result.st_size,
                "mtime_ns": stat_result.st_mtime_ns + 1,
                "sha256": "0" * 64,
                "publisher_checksum_available": False,
            }
        ),
        encoding="utf-8",
    )

    identity = _depth_target_identity(target, {})

    assert identity["sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert identity["sha256_origin"] == "streamed_at_probe_start"
    assert "file_id" not in identity


def test_depth_split_and_probe_checkpoint_bind_all_input_files(tmp_path: Path) -> None:
    manifest = tmp_path / "test.jsonl"
    manifest.write_text(
        '{"sequence_id":"mvsec__outdoor_day1__left"}\n', encoding="utf-8"
    )
    target = tmp_path / "outdoor_day1_gt.hdf5"
    target.write_bytes(b"depth values")
    source = MVSECGeometrySource(
        sequence_id="mvsec__outdoor_day1__left",
        ground_truth_path=target,
        target_dataset="/davis/left/depth_image_raw",
        timestamp_dataset="/davis/left/depth_image_raw_ts",
        source_time_origin_us=100_000_000,
        t_start_us=0,
        t_end_us=60_000_000,
        camera="left",
    )
    split = SimpleNamespace(
        role="final_test",
        recording="outdoor_day1",
        manifest=manifest,
        sources=(source,),
        references=(_reference(3, 2_000_000),),
    )

    identity = _split_identity(
        split,
        "causal",
        target_identity_cache={},
    )

    assert identity["manifest_sha256"] == hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest()
    assert identity["target_artifacts"][0]["sha256"] == hashlib.sha256(
        target.read_bytes()
    ).hexdigest()

    head = MVSECLogDepthHead(2, hidden_dim=2)
    optimizer = torch.optim.AdamW(head.parameters())
    payload = _checkpoint_payload(
        head=head,
        optimizer=optimizer,
        epoch=1,
        fixed_epochs=1,
        encoder_checkpoint=tmp_path / "encoder.pt",
        encoder_checkpoint_bytes=123,
        encoder_checkpoint_sha256="a" * 64,
        encoder_config_hash="b" * 64,
        grid_size=(1, 1),
        protocol={},
        training_history=(),
    )
    assert payload["encoder_checkpoint_bytes"] == 123
    assert payload["encoder_checkpoint_sha256"] == "a" * 64
