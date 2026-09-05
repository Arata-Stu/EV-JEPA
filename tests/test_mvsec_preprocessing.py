from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from event_window_jepa.preprocessing import mvsec_labels
from event_window_jepa.preprocessing.mvsec_labels import (
    inspect_mvsec_flow_ground_truth,
    inspect_mvsec_ground_truth,
    matching_mvsec_flow_ground_truth,
    matching_mvsec_ground_truth,
)
from event_window_jepa.preprocessing.sources import MVSECEventSource


def _hdf5_modules():
    h5py = pytest.importorskip("h5py")
    pytest.importorskip("hdf5plugin")
    return h5py


def test_mvsec_source_keeps_camera_specific_absolute_origins(tmp_path) -> None:
    h5py = _hdf5_modules()
    path = tmp_path / "outdoor_day2_data.hdf5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "davis/left/events",
            data=np.array(
                [[0, 1, 100.000001, -1], [345, 259, 100.000003, 1]],
                dtype=np.float64,
            ),
        )
        handle.create_dataset(
            "davis/right/events",
            data=np.array(
                [[2, 3, 100.500001, 1], [4, 5, 100.500004, -1]],
                dtype=np.float64,
            ),
        )

    left = MVSECEventSource(path, camera="left")
    right = MVSECEventSource(path, camera="right")
    try:
        assert left.metadata.first_timestamp_us == 100_000_001
        assert right.metadata.first_timestamp_us == 100_500_001
        assert left.metadata.sequence_id == "mvsec__outdoor_day2__left"
        assert right.metadata.sequence_id == "mvsec__outdoor_day2__right"
        assert next(left.iter_event_chunks(10))["t_us"].tolist() == [
            100_000_001,
            100_000_003,
        ]
        assert next(right.iter_event_chunks(10))["polarity"].tolist() == [1, -1]
    finally:
        left.close()
        right.close()


def test_mvsec_hdf5_inspector_requires_explicit_embedded_flow(
    tmp_path, monkeypatch
) -> None:
    h5py = _hdf5_modules()
    data_path = tmp_path / "outdoor_day1_data.hdf5"
    data_path.write_bytes(b"placeholder")
    gt_path = tmp_path / "outdoor_day1_gt.hdf5"
    with h5py.File(gt_path, "w") as handle:
        handle.create_dataset(
            "davis/left/flow_dist", data=np.zeros((2, 2, 260, 346), np.float32)
        )
        handle.create_dataset(
            "davis/left/flow_dist_ts", data=np.array([100.0, 100.05])
        )
        handle.create_dataset(
            "davis/left/depth_image_raw",
            data=np.ones((2, 260, 346), np.float32),
        )
        handle.create_dataset(
            "davis/left/depth_image_raw_ts", data=np.array([100.0, 100.05])
        )

    stat_result = gt_path.stat()
    monkeypatch.setitem(
        mvsec_labels.MVSEC_OFFICIAL_GT_ARTIFACTS,
        gt_path.name,
        ("synthetic-gt-test-id", stat_result.st_size),
    )
    gt_sha256 = hashlib.sha256(gt_path.read_bytes()).hexdigest()
    gt_path.with_name(gt_path.name + ".verified.json").write_text(
        json.dumps(
            {
                "metadata_version": 1,
                "status": "verified",
                "filename": gt_path.name,
                "kind": "gt_hdf5",
                "file_id": "synthetic-gt-test-id",
                "expected_bytes": stat_result.st_size,
                "size_bytes": stat_result.st_size,
                "mtime_ns": stat_result.st_mtime_ns,
                "sha256": gt_sha256,
                "publisher_checksum_available": False,
            }
        ),
        encoding="utf-8",
    )

    assert matching_mvsec_ground_truth(data_path) == gt_path
    metadata = inspect_mvsec_ground_truth(gt_path, camera="left")
    assert "flow_path" not in metadata
    assert metadata["depth_dataset"] == "/davis/left/depth_image_raw"
    assert metadata["depth_count"] == 2
    assert metadata["mvsec_gt_source_file_id"] == "synthetic-gt-test-id"
    assert metadata["mvsec_gt_source_expected_bytes"] == stat_result.st_size
    assert metadata["mvsec_gt_source_size_bytes"] == stat_result.st_size
    assert metadata["mvsec_gt_source_mtime_ns"] == stat_result.st_mtime_ns
    assert metadata["mvsec_gt_source_sha256"] == gt_sha256
    assert (
        metadata["mvsec_gt_source_sha256_origin"]
        == "download_verified_sidecar"
    )
    embedded = inspect_mvsec_ground_truth(
        gt_path, camera="left", include_embedded_flow=True
    )
    assert embedded["flow_dataset"] == "/davis/left/flow_dist"
    assert embedded["flow_format"] == "mvsec_embedded_hdf5_flow_dist"
    assert embedded["flow_coordinate_frame"] == "distorted"


def test_mvsec_official_flow_npz_inspector_records_identity(tmp_path, monkeypatch) -> None:
    data_path = tmp_path / "outdoor_day1_data.hdf5"
    data_path.write_bytes(b"placeholder")
    flow_path = tmp_path / "outdoor_day1_gt_flow_dist.npz"
    np.savez(
        flow_path,
        timestamps=np.array([100.0, 100.05], dtype=np.float64),
        x_flow_dist=np.zeros((2, 260, 346), dtype=np.float32),
        y_flow_dist=np.ones((2, 260, 346), dtype=np.float32),
    )
    monkeypatch.setitem(
        mvsec_labels.MVSEC_OFFICIAL_FLOW_ARTIFACTS,
        flow_path.name,
        ("synthetic-test-id", flow_path.stat().st_size),
    )

    assert matching_mvsec_flow_ground_truth(data_path) == flow_path
    metadata = inspect_mvsec_flow_ground_truth(flow_path)
    assert metadata["flow_format"] == "mvsec_gt_flow_npz_v1"
    assert metadata["flow_x_key"] == "x_flow_dist"
    assert metadata["flow_y_key"] == "y_flow_dist"
    assert metadata["flow_timestamp_key"] == "timestamps"
    assert metadata["flow_shape"] == [2, 260, 346]
    assert metadata["flow_source_file_id"] == "synthetic-test-id"
    assert metadata["flow_source_mtime_ns"] == flow_path.stat().st_mtime_ns
    assert metadata["flow_source_sha256"] == hashlib.sha256(
        flow_path.read_bytes()
    ).hexdigest()


def test_mvsec_gt_identity_ignores_stale_sidecar(tmp_path, monkeypatch) -> None:
    gt_path = tmp_path / "outdoor_day2_gt.hdf5"
    gt_path.write_bytes(b"synthetic GT")
    stat_result = gt_path.stat()
    monkeypatch.setitem(
        mvsec_labels.MVSEC_OFFICIAL_GT_ARTIFACTS,
        gt_path.name,
        ("synthetic-gt-test-id", stat_result.st_size),
    )
    gt_path.with_name(gt_path.name + ".verified.json").write_text(
        json.dumps(
            {
                "metadata_version": 1,
                "status": "verified",
                "filename": gt_path.name,
                "kind": "gt_hdf5",
                "file_id": "synthetic-gt-test-id",
                "expected_bytes": stat_result.st_size,
                "size_bytes": stat_result.st_size,
                "mtime_ns": stat_result.st_mtime_ns + 1,
                "sha256": "0" * 64,
                "publisher_checksum_available": False,
            }
        ),
        encoding="utf-8",
    )

    identity = mvsec_labels._gt_source_identity(gt_path)
    assert identity == {
        "mvsec_gt_source_size_bytes": stat_result.st_size,
        "mvsec_gt_source_mtime_ns": stat_result.st_mtime_ns,
    }


def test_mvsec_source_accepts_zero_one_polarity(tmp_path) -> None:
    h5py = _hdf5_modules()
    path = tmp_path / "bad_data.hdf5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "davis/left/events",
            data=np.array([[0, 0, 1.0, 0], [1, 1, 1.1, 1]], dtype=np.float64),
        )
    source = MVSECEventSource(path)
    try:
        assert next(source.iter_event_chunks(10))["polarity"].tolist() == [0, 1]
    finally:
        source.close()


def test_mvsec_source_rejects_mixed_polarity_encodings_across_chunks(
    tmp_path,
) -> None:
    h5py = _hdf5_modules()
    path = tmp_path / "mixed_data.hdf5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "davis/left/events",
            data=np.array(
                [[0, 0, 1.0, 0], [1, 1, 1.1, 1], [2, 2, 1.2, -1]],
                dtype=np.float64,
            ),
        )
    source = MVSECEventSource(path)
    try:
        chunks = source.iter_event_chunks(1)
        next(chunks)
        next(chunks)
        with pytest.raises(ValueError, match="mixes polarity encodings"):
            next(chunks)
    finally:
        source.close()
