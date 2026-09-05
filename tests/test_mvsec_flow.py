from __future__ import annotations

import hashlib
import json
import struct
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from event_window_jepa.downstream.mvsec_flow import (
    CANONICAL_RANDOM_PROBE_FLOW_SCALE,
    CANONICAL_RANDOM_PROBE_HEAD_DEPTH,
    CANONICAL_RANDOM_PROBE_HIDDEN_DIM,
    CANONICAL_RANDOM_PROBE_MAX_DISPLACEMENT,
    DT1_FLOW_HORIZON_US,
    FlowMetricAccumulator,
    ResolvedSchedule,
    _event_support_window_us,
    _file_artifact_report,
    _flow_batch_on_protocol,
    _manifest_report,
    _make_probe_head,
    _masked_endpoint_loss,
    _model_image_size,
    _protocol_report,
    _require_expected_sequence,
    _require_unchanged_checkpoint_identity,
    _resolve_schedule,
    _selection_summary,
    _validate_alignment_schedule,
    _validate_output_paths,
    build_parser,
    reference_set_sha256,
    resolve_flow_protocol,
)
from event_window_jepa.models.cmax_flow import RecurrentTokenFlowHead
from event_window_jepa.downstream.mvsec_geometry import (
    MVSECGeometrySource,
    MVSECTargetReference,
    build_mvsec_target_references,
    open_mvsec_official_flow_npz,
    read_mvsec_geometry_sources,
)


def _source(sequence_id: str) -> MVSECGeometrySource:
    return MVSECGeometrySource(
        sequence_id=sequence_id,
        ground_truth_path=Path("unused.h5"),
        target_dataset="/davis/left/flow_dist",
        timestamp_dataset="/davis/left/flow_dist_ts",
        source_time_origin_us=0,
        t_start_us=0,
        t_end_us=1_000_000,
        camera="left",
    )


def _reference(
    target_index: int,
    timestamp_us: int,
    *,
    flow_interval_us: int = 50_000,
) -> MVSECTargetReference:
    return MVSECTargetReference(
        source_index=0,
        target_index=target_index,
        label_timestamp_us=timestamp_us,
        event_window_end_us=timestamp_us,
        flow_interval_us=flow_interval_us,
    )


def test_flow_protocol_accepts_only_native_and_dt1() -> None:
    native = resolve_flow_protocol("native")
    dt1 = resolve_flow_protocol("dt1")

    assert native.ground_truth_scale == 1.0
    assert native.horizon_us(49_999) == 49_999
    assert dt1.nominal_horizon_us == DT1_FLOW_HORIZON_US == 22_222
    assert dt1.name == "dt1_scaled_native_labels"
    assert dt1.ground_truth_scale == pytest.approx(20.0 / 45.0)
    with pytest.raises(ValueError, match="dt4 is intentionally unsupported"):
        resolve_flow_protocol("dt4")


def test_flow_scaling_reports_prediction_and_ground_truth_horizons() -> None:
    prediction = torch.ones((2, 2, 1, 1))
    target = torch.full_like(prediction, 2.0)
    intervals = torch.tensor([50_000, 40_000])

    native_prediction, native_target, native_scale, native_horizon = (
        _flow_batch_on_protocol(
            prediction,
            target,
            intervals,
            protocol=resolve_flow_protocol("native"),
            base_window_us=50_000,
        )
    )
    assert native_prediction[:, 0, 0, 0].tolist() == pytest.approx([1.0, 0.8])
    assert torch.equal(native_target, target)
    assert native_scale.tolist() == pytest.approx([1.0, 0.8])
    assert native_horizon.tolist() == pytest.approx([50_000, 40_000])

    dt1_prediction, dt1_target, dt1_scale, dt1_horizon = _flow_batch_on_protocol(
        prediction,
        target,
        intervals,
        protocol=resolve_flow_protocol("dt1"),
        base_window_us=50_000,
    )
    assert dt1_prediction[:, 0, 0, 0].tolist() == pytest.approx(
        [20.0 / 45.0] * 2
    )
    assert dt1_target[:, 0, 0, 0].tolist() == pytest.approx([40.0 / 45.0] * 2)
    assert dt1_scale.tolist() == pytest.approx(
        [20.0 / 45.0] * 2
    )
    assert dt1_horizon.tolist() == pytest.approx([1_000_000.0 / 45.0] * 2)


def test_metric_accumulator_keeps_sample_and_pixel_averages_distinct() -> None:
    accumulator = FlowMetricAccumulator(minimum_valid_pixels=1)
    target_one = np.zeros((2, 1, 1), dtype=np.float32)
    prediction_one = target_one.copy()
    prediction_one[0, 0, 0] = 2.0
    target_three = np.zeros((2, 1, 3), dtype=np.float32)

    assert accumulator.update(
        prediction_one,
        target_one,
        np.ones((1, 1), dtype=np.bool_),
    )
    assert accumulator.update(
        target_three,
        target_three,
        np.ones((1, 3), dtype=np.bool_),
    )
    metrics = accumulator.finalize()

    assert metrics["sample_average"]["AEPE"] == pytest.approx(1.0)
    assert metrics["pixel_average"]["AEPE"] == pytest.approx(0.5)
    assert metrics["sample_average"]["1PE_percent"] == pytest.approx(50.0)
    assert metrics["pixel_average"]["1PE_percent"] == pytest.approx(25.0)
    assert metrics["sample_average"]["2PE_percent"] == 0.0
    assert metrics["valid_pixels_evaluated"] == 4


def test_metric_accumulator_skips_frames_below_valid_threshold() -> None:
    accumulator = FlowMetricAccumulator(minimum_valid_pixels=2)
    flow = np.zeros((2, 1, 2), dtype=np.float32)

    assert not accumulator.update(
        flow,
        flow,
        np.array([[True, False]], dtype=np.bool_),
    )
    assert accumulator.update(
        flow,
        flow,
        np.ones((1, 2), dtype=np.bool_),
    )
    metrics = accumulator.finalize()
    assert metrics["frames_seen"] == 2
    assert metrics["frames_evaluated"] == 1
    assert metrics["frames_skipped_valid_below_threshold"] == 1


def test_masked_endpoint_loss_drops_only_underfilled_frames() -> None:
    prediction = torch.tensor(
        [
            [[[1.0, 1.0]], [[0.0, 0.0]]],
            [[[2.0, 2.0]], [[0.0, 0.0]]],
        ],
        requires_grad=True,
    )
    target = torch.zeros_like(prediction)
    valid = torch.tensor([[[True, False]], [[True, True]]])

    loss, kept = _masked_endpoint_loss(
        prediction,
        target,
        valid,
        minimum_valid_pixels=2,
    )

    assert loss is not None
    assert kept == 1
    assert float(loss.detach()) == pytest.approx((4.0 + 1e-6) ** 0.5)
    loss.backward()
    assert prediction.grad is not None
    assert not bool(prediction.grad[0].any())
    assert bool(prediction.grad[1].any())


def test_reference_hash_tracks_ordered_target_index_and_timestamp() -> None:
    sources = (_source("mvsec__outdoor_day1__left"),)
    references = (_reference(4, 222_400_000), _reference(5, 222_450_000))

    digest = reference_set_sha256(sources, references)
    assert digest == reference_set_sha256(sources, references)
    assert digest != reference_set_sha256(sources, tuple(reversed(references)))
    assert digest != reference_set_sha256(
        sources,
        (_reference(4, 222_400_001), references[1]),
    )
    # Flow interval and alignment metadata do not redefine the selected labels.
    assert digest == reference_set_sha256(
        sources,
        (
            _reference(4, 222_400_000, flow_interval_us=40_000),
            references[1],
        ),
    )


def test_checkpoint_artifact_snapshot_detects_atomic_replacement(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"stable checkpoint")
    artifact = _file_artifact_report(checkpoint)

    assert artifact["sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    _require_unchanged_checkpoint_identity(checkpoint, artifact)

    replacement = tmp_path / "replacement.pt"
    replacement.write_bytes(checkpoint.read_bytes())
    replacement.replace(checkpoint)
    with pytest.raises(RuntimeError, match="checkpoint changed"):
        _require_unchanged_checkpoint_identity(checkpoint, artifact)


def test_cli_defaults_to_causal_native_protocol() -> None:
    args = build_parser().parse_args(
        [
            "cmax-eval",
            "--checkpoint",
            "checkpoint.pt",
            "--eval-manifest",
            "day1.jsonl",
            "--output-dir",
            "results",
        ]
    )

    assert args.command == "cmax-eval"
    assert args.alignment == "causal"
    assert args.dt == "native"
    assert args.minimum_valid_pixels == 100
    assert args.protocol_stage == "final"
    assert args.dev_fraction == 0.2
    assert args.dev_guard_ms is None


def test_random_probe_head_uses_canonical_spec_not_checkpoint_cmax_spec() -> None:
    checkpoint_head = RecurrentTokenFlowHead(
        8,
        hidden_dim=13,
        head_depth=3,
        flow_scale=0.2,
        max_displacement=7.0,
    )
    model = SimpleNamespace(cmax_flow_head=checkpoint_head)
    config = SimpleNamespace(
        model=SimpleNamespace(embed_dim=8),
        cmax=SimpleNamespace(
            hidden_dim=99,
            head_depth=4,
            flow_scale=0.3,
            max_displacement=9.0,
        ),
    )

    random_head = _make_probe_head(model, config, initialization="random", seed=5)
    assert random_head.hidden_dim == CANONICAL_RANDOM_PROBE_HIDDEN_DIM
    assert random_head.head_depth == CANONICAL_RANDOM_PROBE_HEAD_DEPTH
    assert random_head.flow_scale == CANONICAL_RANDOM_PROBE_FLOW_SCALE
    assert (
        random_head.max_displacement
        == CANONICAL_RANDOM_PROBE_MAX_DISPLACEMENT
    )

    cmax_head = _make_probe_head(model, config, initialization="cmax", seed=5)
    assert cmax_head.hidden_dim == 13
    assert cmax_head.head_depth == 3
    assert cmax_head.flow_scale == 0.2
    assert cmax_head.max_displacement == 7.0
    for name, value in checkpoint_head.state_dict().items():
        assert torch.equal(cmax_head.state_dict()[name], value)


def test_flow_outputs_cannot_overwrite_checkpoint_or_manifest(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    protected_checkpoint = output_dir / "flow_head.pt"
    args = SimpleNamespace(
        command="probe",
        checkpoint=protected_checkpoint,
        train_manifest=tmp_path / "train.jsonl",
        eval_manifest=tmp_path / "eval.jsonl",
        output_dir=output_dir,
    )
    with pytest.raises(ValueError, match="overwrite the encoder checkpoint"):
        _validate_output_paths(args)

    args.checkpoint = tmp_path / "encoder.pt"
    args.eval_manifest = output_dir / "report.json"
    with pytest.raises(ValueError, match="overwrite the evaluation manifest"):
        _validate_output_paths(args)


def test_event_support_distinguishes_causal_and_f3_centered_masks() -> None:
    native = resolve_flow_protocol("native")
    dt1 = resolve_flow_protocol("dt1")

    assert _event_support_window_us("causal", native) == "native_interval"
    assert _event_support_window_us("causal", dt1) == 22_222
    assert _event_support_window_us("f3_centered", dt1) is None


def test_report_marks_dt1_as_scaled_native_label_diagnostic() -> None:
    args = SimpleNamespace(
        alignment="causal",
        dt="dt1",
        minimum_valid_pixels=100,
    )
    report = _protocol_report(
        args,
        ResolvedSchedule(window_ms=50.0, stride_ms=50.0, history_steps=10),
        resolve_flow_protocol("dt1"),
    )

    assert report["flow_rate"] == {
        "cli_value": "dt1",
        "protocol": "dt1_scaled_native_labels",
        "target_timestamps": "native_flow_npz",
        "ground_truth_resampling": "none",
        "exact_evflownet_800_frame_protocol": False,
        "benchmark_status": "diagnostic_not_published_800_frame_benchmark",
        "limitation": (
            "exact 800-frame reproduction requires APS-timestamp flow "
            "interpolation/composition and is not implemented"
        ),
    }
    support = report["validity_mask"]["event_support"]
    assert support["duration_us"] == 22_222
    assert support["uses_events_after_label"] is False
    assert report["evflownet_test_interval"]["selected_timestamp_grid"] == (
        "native_flow_npz_not_45hz_APS_frames"
    )

    args.alignment = "f3_centered"
    centered = _protocol_report(
        args,
        ResolvedSchedule(window_ms=50.0, stride_ms=50.0, history_steps=10),
        resolve_flow_protocol("dt1"),
    )
    centered_support = centered["validity_mask"]["event_support"]
    centered_alignment = centered["alignment"]
    assert centered_alignment["final_model_window_us"] == 50_000
    assert centered_alignment["f3_ctx_flow_fixed_window_us"] == 50_000
    assert centered_support["source"] == "fixed_50ms_f3_centered_model_input_ctx_flow"
    assert centered_support["duration_us"] == 50_000
    assert centered_support["uses_events_after_label"] is True


def test_zero_schedule_override_is_rejected_instead_of_using_default() -> None:
    args = SimpleNamespace(window_ms=0.0, stride_ms=None, history_steps=None)
    recurrent = SimpleNamespace(
        sequence_loader=True,
        window_ms=50.0,
        stride_ms=50.0,
        burn_in_steps=2,
        sequence_length=8,
    )
    config = SimpleNamespace(
        recurrent=recurrent,
        windows=SimpleNamespace(canonical_ms=50.0),
    )

    with pytest.raises(ValueError, match="must be positive"):
        _resolve_schedule(args, config)


def test_f3_centered_rejects_non_reference_context_duration() -> None:
    _validate_alignment_schedule(
        "f3_centered",
        ResolvedSchedule(window_ms=50.0, stride_ms=25.0, history_steps=2),
    )
    with pytest.raises(ValueError, match="fixed 50-ms"):
        _validate_alignment_schedule(
            "f3_centered",
            ResolvedSchedule(window_ms=25.0, stride_ms=25.0, history_steps=2),
        )
    _validate_alignment_schedule(
        "causal",
        ResolvedSchedule(window_ms=25.0, stride_ms=25.0, history_steps=2),
    )
    args = SimpleNamespace(
        alignment="f3_centered",
        dt="native",
        minimum_valid_pixels=100,
    )
    with pytest.raises(ValueError, match="fixed 50-ms"):
        _protocol_report(
            args,
            ResolvedSchedule(window_ms=25.0, stride_ms=25.0, history_steps=2),
            resolve_flow_protocol("native"),
        )


def test_mvsec_protocol_requires_full_center_padded_canvas() -> None:
    good = SimpleNamespace(model=SimpleNamespace(image_size=(272, 352)))
    bad = SimpleNamespace(model=SimpleNamespace(image_size=(224, 224)))

    assert _model_image_size(good) == (272, 352)
    with pytest.raises(ValueError, match="field of view"):
        _model_image_size(bad)


def test_sequence_protocol_matches_identifier_component_not_substring() -> None:
    _require_expected_sequence(
        (_source("mvsec__outdoor_day2__left"),),
        "outdoor_day2",
        "train",
    )
    with pytest.raises(ValueError, match="outdoor_day2"):
        _require_expected_sequence(
            (_source("mvsec__outdoor_day20__left"),),
            "outdoor_day2",
            "train",
        )


def test_official_mvsec_flow_npz_is_validated_and_memory_mapped(tmp_path: Path) -> None:
    flow_path = tmp_path / "outdoor_day1_gt_flow_dist.npz"
    x_flow = np.zeros((2, 260, 346), dtype=np.float32)
    y_flow = np.ones_like(x_flow)
    np.savez(
        flow_path,
        timestamps=np.array([100.0, 100.05], dtype=np.float64),
        x_flow_dist=x_flow,
        y_flow_dist=y_flow,
    )

    flow = open_mvsec_official_flow_npz(flow_path)
    try:
        assert isinstance(flow.x_flow_dist, np.memmap)
        assert isinstance(flow.y_flow_dist, np.memmap)
        assert flow.timestamps_seconds.tolist() == [100.0, 100.05]
        assert flow.x_flow_dist.shape == (2, 260, 346)
        assert float(flow.y_flow_dist[1, 259, 345]) == 1.0
    finally:
        flow.close()


def _write_forced_zip64_npz(path: Path) -> None:
    arrays = {
        "timestamps": np.array([100.0, 100.05], dtype=np.float64),
        "x_flow_dist": np.zeros((2, 260, 346), dtype=np.float32),
        "y_flow_dist": np.ones((2, 260, 346), dtype=np.float32),
    }
    with zipfile.ZipFile(path, "w", allowZip64=True) as archive:
        for key, array in arrays.items():
            with archive.open(f"{key}.npy", "w", force_zip64=True) as stream:
                np.lib.format.write_array(stream, array, allow_pickle=False)


def test_official_mvsec_flow_npz_supports_zip64_local_headers(
    tmp_path: Path,
) -> None:
    flow_path = tmp_path / "forced_zip64.npz"
    _write_forced_zip64_npz(flow_path)

    with zipfile.ZipFile(flow_path, "r") as archive:
        member = archive.getinfo("x_flow_dist.npy")
    with flow_path.open("rb") as handle:
        handle.seek(member.header_offset)
        local_header = handle.read(30)
    assert struct.unpack_from("<II", local_header, 18) == (0xFFFFFFFF, 0xFFFFFFFF)

    flow = open_mvsec_official_flow_npz(flow_path)
    try:
        assert isinstance(flow.x_flow_dist, np.memmap)
        assert flow.x_flow_dist.shape == (2, 260, 346)
        assert float(flow.y_flow_dist[1, 259, 345]) == 1.0
    finally:
        flow.close()


def test_official_mvsec_flow_npz_rejects_crc_corruption(tmp_path: Path) -> None:
    flow_path = tmp_path / "corrupted.npz"
    _write_forced_zip64_npz(flow_path)
    with zipfile.ZipFile(flow_path, "r") as archive:
        member = archive.getinfo("y_flow_dist.npy")
    with flow_path.open("r+b") as handle:
        handle.seek(member.header_offset)
        local_header = handle.read(30)
        filename_length, extra_length = struct.unpack_from("<HH", local_header, 26)
        member_start = member.header_offset + 30 + filename_length + extra_length
        handle.seek(member_start + member.file_size - 1)
        original = handle.read(1)
        handle.seek(-1, 1)
        handle.write(bytes([original[0] ^ 0xFF]))

    with pytest.raises(ValueError, match="CRC"):
        open_mvsec_official_flow_npz(flow_path)


def test_mvsec_manifest_declares_official_npz_flow_contract(tmp_path: Path) -> None:
    flow_path = tmp_path / "outdoor_day1_gt_flow_dist.npz"
    shape = (2, 260, 346)
    np.savez(
        flow_path,
        timestamps=np.array([100.0, 100.05], dtype=np.float64),
        x_flow_dist=np.zeros(shape, dtype=np.float32),
        y_flow_dist=np.zeros(shape, dtype=np.float32),
    )
    manifest = tmp_path / "test.jsonl"
    flow_sha256 = hashlib.sha256(flow_path.read_bytes()).hexdigest()
    manifest.write_text(
        json.dumps(
            {
                "sequence_id": "mvsec__outdoor_day1__left",
                "dataset": "mvsec",
                "camera": "left",
                "split": "test",
                "height": 260,
                "width": 346,
                "source_height": 260,
                "source_width": 346,
                "spatial_downsample": 1,
                "coordinate_frame": "distorted",
                "flow_coordinate_frame": "distorted",
                "flow_timestamps_relative": False,
                "flow_path": str(flow_path),
                "flow_format": "mvsec_gt_flow_npz_v1",
                "flow_dataset": "x_flow_dist,y_flow_dist",
                "flow_x_key": "x_flow_dist",
                "flow_y_key": "y_flow_dist",
                "flow_timestamp_dataset": "timestamps",
                "flow_timestamp_key": "timestamps",
                "flow_channel_order": "x,y",
                "flow_shape": [2, 260, 346],
                "flow_dtype": np.dtype(np.float32).str,
                "flow_count": 2,
                "flow_source_metadata_version": 1,
                "flow_source_file_id": "synthetic-test-id",
                "flow_source_expected_bytes": flow_path.stat().st_size,
                "flow_source_size_bytes": flow_path.stat().st_size,
                "flow_source_mtime_ns": flow_path.stat().st_mtime_ns,
                "flow_source_sha256": flow_sha256,
                "flow_source_sha256_origin": "computed_during_test",
                "source_time_origin_us": 100_000_000,
                "t_start_us": 0,
                "t_end_us": 1_000_000,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    sources = read_mvsec_geometry_sources(manifest, kind="flow", split="test")
    assert len(sources) == 1
    assert sources[0].target_format == "mvsec_gt_flow_npz_v1"
    assert sources[0].target_dataset == "x_flow_dist,y_flow_dist"
    assert sources[0].timestamp_dataset == "timestamps"
    assert sources[0].target_file_id == "synthetic-test-id"
    assert sources[0].target_sha256 == flow_sha256
    assert sources[0].target_mtime_ns == flow_path.stat().st_mtime_ns

    summary = _selection_summary(sources, (_reference(1, 50_000),))
    artifact = summary["target_artifacts"][0]
    assert artifact["bytes"] == flow_path.stat().st_size
    assert artifact["actual_size_bytes"] == flow_path.stat().st_size
    assert artifact["manifest_source_mtime_ns"] == flow_path.stat().st_mtime_ns
    assert artifact["manifest_declared_sha256"] == flow_sha256
    assert artifact["sha256"] == flow_sha256
    assert artifact["sha256_verification"] == (
        "preprocessing_manifest_declaration_bound_to_current_size_and_mtime"
    )

    manifest_artifact = _manifest_report(manifest)
    assert manifest_artifact["path"] == str(manifest.resolve())
    assert manifest_artifact["bytes"] == manifest.stat().st_size
    assert manifest_artifact["mtime_ns"] == manifest.stat().st_mtime_ns
    assert manifest_artifact["sha256"] == hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest()

    stale_row = json.loads(manifest.read_text(encoding="utf-8"))
    del stale_row["flow_source_mtime_ns"]
    manifest.write_text(json.dumps(stale_row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mtime metadata"):
        read_mvsec_geometry_sources(manifest, kind="flow", split="test")


def test_compressed_mvsec_flow_npz_is_rejected_for_bounded_memory(
    tmp_path: Path,
) -> None:
    flow_path = tmp_path / "compressed.npz"
    shape = (1, 260, 346)
    np.savez_compressed(
        flow_path,
        timestamps=np.array([1.0]),
        x_flow_dist=np.zeros(shape, dtype=np.float32),
        y_flow_dist=np.zeros(shape, dtype=np.float32),
    )

    with pytest.raises(ValueError, match="uncompressed"):
        open_mvsec_official_flow_npz(flow_path)


def test_official_flow_reference_zero_is_always_skipped(tmp_path: Path) -> None:
    flow_path = tmp_path / "flow.npz"
    shape = (2, 260, 346)
    np.savez(
        flow_path,
        timestamps=np.array([100.1, 100.15], dtype=np.float64),
        x_flow_dist=np.zeros(shape, dtype=np.float32),
        y_flow_dist=np.zeros(shape, dtype=np.float32),
    )
    source = MVSECGeometrySource(
        sequence_id="mvsec__outdoor_day2__left",
        ground_truth_path=flow_path,
        target_dataset="x_flow_dist,y_flow_dist",
        timestamp_dataset="timestamps",
        source_time_origin_us=100_000_000,
        t_start_us=0,
        t_end_us=1_000_000,
        camera="left",
        target_format="mvsec_gt_flow_npz_v1",
    )

    class Store:
        @staticmethod
        def slice(sequence_id: str, t_end_us: int, duration_us: int):
            assert sequence_id == source.sequence_id
            assert t_end_us >= duration_us
            return SimpleNamespace(event_count=1)

    references = build_mvsec_target_references(
        Store(),
        (source,),
        kind="flow",
        window_us=50_000,
        stride_us=50_000,
        history_steps=1,
        alignment="causal",
        minimum_events=1,
    )

    assert [reference.target_index for reference in references] == [1]
    assert references[0].flow_interval_us == 50_000
