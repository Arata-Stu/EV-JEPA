from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from event_window_jepa.downstream.mvsec_geometry import (
    MVSECGeometrySource,
    MVSECTargetReference,
    _project_native_map,
    build_mvsec_target_references,
    center_crop_parameters,
    depth_metric_sums,
    finalize_depth_metrics,
    finalize_flow_metrics,
    flow_metric_sums,
    split_mvsec_temporal_dev_references,
)


class _UnusedStore:
    def slice(self, sequence_id: str, t_end_us: int, duration_us: int) -> None:
        raise AssertionError("invalid alignment must fail before reading events")


def _temporal_source() -> MVSECGeometrySource:
    return MVSECGeometrySource(
        sequence_id="mvsec__outdoor_day2__left",
        ground_truth_path=Path("unused.hdf5"),
        target_dataset="/davis/left/depth_image_raw",
        timestamp_dataset="/davis/left/depth_image_raw_ts",
        source_time_origin_us=0,
        t_start_us=0,
        t_end_us=2_000_000,
        camera="left",
    )


def _temporal_references() -> tuple[MVSECTargetReference, ...]:
    return tuple(
        MVSECTargetReference(
            source_index=0,
            target_index=index,
            label_timestamp_us=100_000 + index * 50_000,
            event_window_end_us=100_000 + index * 50_000,
            flow_interval_us=50_000,
        )
        for index in range(20)
    )


def test_temporal_dev_split_guards_full_history_and_hashes_partitions() -> None:
    sources = (_temporal_source(),)
    train, dev, report = split_mvsec_temporal_dev_references(
        sources,
        _temporal_references(),
        window_us=50_000,
        stride_us=50_000,
        history_steps=3,
        alignment="causal",
        dev_fraction=0.2,
    )

    assert len(dev) == 4
    assert report["history_span_us"] == 150_000
    assert report["effective_guard_us"] == 150_000
    assert dev[0].label_timestamp_us - train[-1].label_timestamp_us >= 150_000
    assert report["boundary"]["actual_input_dependency_nonoverlap"] is True
    assert report["selected"]["actual_input_dependency_nonoverlap"] is True
    assert report["hashes"]["selected_train_targets_sha256"] != report[
        "hashes"
    ]["selected_dev_targets_sha256"]
    contract = report["representation_pretraining_visibility_contract"]
    assert contract["protocol_class"] == (
        "transductive_event_only_representation_pretraining"
    )
    assert contract["geometry_labels_visible_to_pretraining"] is False


def test_temporal_dev_split_checks_task_specific_event_dependencies() -> None:
    sources = (_temporal_source(),)
    references = tuple(
        MVSECTargetReference(0, index, index * 10_000, index * 10_000, 50_000)
        for index in range(1, 21)
    )
    train, dev, report = split_mvsec_temporal_dev_references(
        sources,
        references,
        window_us=10_000,
        stride_us=10_000,
        history_steps=1,
        alignment="causal",
        dev_fraction=0.2,
        additional_dependency_interval=lambda reference: (
            reference.label_timestamp_us - reference.flow_interval_us,
            reference.label_timestamp_us,
        ),
    )

    assert train[-1].label_timestamp_us <= dev[0].label_timestamp_us - 50_000
    assert report["counts"]["dropped_by_dependency_assertion"] > 0
    assert report["selected"]["dependency_gap_us"] >= 0


def test_centered_temporal_dev_guard_includes_ceil_allowance() -> None:
    with pytest.raises(ValueError, match="rounding allowance"):
        split_mvsec_temporal_dev_references(
            (_temporal_source(),),
            _temporal_references(),
            window_us=50_000,
            stride_us=50_000,
            history_steps=2,
            alignment="f3_centered",
            guard_us=100_000,
            dev_fraction=0.2,
        )


def test_mvsec_center_pad_preserves_complete_native_map() -> None:
    transform = center_crop_parameters((272, 352))
    native = np.arange(260 * 346, dtype=np.float32).reshape(260, 346)
    padded = _project_native_map(native, transform)

    assert (transform.x0, transform.y0) == (-3, -6)
    assert padded.shape == (272, 352)
    assert np.array_equal(padded[6:266, 3:349], native)
    assert not np.any(padded[:6])
    assert not np.any(padded[:, :3])


def test_flow_metrics_report_endpoint_thresholds_and_angle() -> None:
    target = np.zeros((2, 1, 4), dtype=np.float32)
    prediction = np.zeros_like(target)
    prediction[0, 0] = [0.0, 1.5, 2.5, 4.0]
    valid = np.ones((1, 4), dtype=np.bool_)
    metrics = finalize_flow_metrics(flow_metric_sums(prediction, target, valid))

    assert metrics["AEPE"] == 2.0
    assert metrics["1PE_percent"] == 75.0
    assert metrics["2PE_percent"] == 50.0
    assert metrics["3PE_percent"] == 25.0


def test_depth_metrics_are_zero_for_exact_prediction() -> None:
    target = np.array([[1.0, 2.0], [4.0, 8.0]], dtype=np.float32)
    valid = np.ones_like(target, dtype=np.bool_)
    metrics = finalize_depth_metrics(depth_metric_sums(target, target, valid))

    assert metrics["AbsRel"] == 0.0
    assert metrics["RMSE"] == 0.0
    assert metrics["delta1"] == 1.0


def test_shared_f3_alignment_requires_the_fixed_50ms_context() -> None:
    with pytest.raises(ValueError, match="fixed 50-ms"):
        build_mvsec_target_references(
            _UnusedStore(),
            (),
            kind="flow",
            window_us=25_000,
            stride_us=25_000,
            history_steps=1,
            alignment="f3_centered",
            minimum_events=0,
        )
