from __future__ import annotations

import json

import numpy as np
import pytest

from event_window_jepa.data.types import SequenceInfo
from event_window_jepa.preprocessing.common import (
    EventSourceMetadata,
    _exclusive_output_lock,
    _repair_timestamp_regressions,
    _timestamp_numpy_dtype,
    _transform_coordinates,
    _validate_identity_attributes,
    _validate_source_chunk,
    coarse_index_entries,
    write_manifest,
)
from event_window_jepa.preprocessing.cli import (
    _copy_bbox_atomic,
    _dat_path_split,
    _matching_bbox,
    _m3ed_official_names,
    _resolved_spatial_downsample,
    _resolved_spatial_downsample_method,
    _validate_bbox,
)
from event_window_jepa.preprocessing.merge_manifests import merge_manifests
from event_window_jepa.preprocessing.m3ed_labels import (
    copy_file_atomic,
    discover_m3ed_labels,
)
from event_window_jepa.preprocessing.m3ed_splits import create_m3ed_split_manifests
from event_window_jepa.preprocessing.sources import sequence_identifier


def test_auto_timestamp_dtype_uses_uint32_only_when_safe() -> None:
    assert _timestamp_numpy_dtype(10_000_000, "auto") == np.dtype(np.uint32)
    assert _timestamp_numpy_dtype(1 << 32, "auto") == np.dtype(np.uint64)


def test_coarse_index_is_first_ge_across_streaming_chunks() -> None:
    first, next_boundary = coarse_index_entries(
        np.array([0, 500, 2_100], dtype=np.uint32),
        next_boundary_us=0,
        step_us=1_000,
        global_event_offset=0,
    )
    second, next_boundary = coarse_index_entries(
        np.array([5_000, 5_000, 7_100], dtype=np.uint32),
        next_boundary_us=next_boundary,
        step_us=1_000,
        global_event_offset=3,
    )
    assert np.concatenate((first, second)).tolist() == [0, 2, 2, 3, 3, 3, 5, 5]
    assert next_boundary == 8_000


def test_rvt_timestamp_repair_is_running_max_across_chunks() -> None:
    first = {
        "x": np.arange(4),
        "y": np.arange(4),
        "t_us": np.array([10, 9, 8, 11]),
        "polarity": np.zeros(4, dtype=np.uint8),
    }
    repaired, count, maximum = _repair_timestamp_regressions(first, None)
    arrays, previous = _validate_source_chunk(repaired, None)
    assert arrays["t_us"].tolist() == [10, 10, 10, 11]
    assert (count, maximum, previous) == (2, 2, 11)

    second = {
        "x": np.arange(2),
        "y": np.arange(2),
        "t_us": np.array([7, 12]),
        "polarity": np.ones(2, dtype=np.uint8),
    }
    repaired, count, maximum = _repair_timestamp_regressions(second, previous)
    arrays, previous = _validate_source_chunk(repaired, previous)
    assert arrays["t_us"].tolist() == [11, 12]
    assert (count, maximum, previous) == (1, 4, 12)


def test_rvt_coordinate_repair_drops_out_of_bounds_without_clipping() -> None:
    x, y, valid = _transform_coordinates(
        np.array([0, 1279, 1280, 20]),
        np.array([0, 719, 10, -1]),
        source_width=1280,
        source_height=720,
        downsample=2,
        drop_out_of_bounds=True,
    )
    assert valid.tolist() == [True, True, False, False]
    assert x[valid].tolist() == [0, 639]
    assert y[valid].tolist() == [0, 359]


def test_coordinate_repair_remains_strict_by_default() -> None:
    with pytest.raises(ValueError, match="invalid=1/2"):
        _transform_coordinates(
            np.array([0, 1280]),
            np.array([0, 10]),
            source_width=1280,
            source_height=720,
            downsample=1,
        )


def test_legacy_rvt_output_without_duration_extension_metadata_is_accepted(
    tmp_path,
) -> None:
    source_path = tmp_path / "source.h5"
    source_path.write_bytes(b"source")
    metadata = EventSourceMetadata(
        sequence_id="gen4__recording__left",
        dataset="gen4",
        source_path=source_path,
        camera="left",
        width=1280,
        height=720,
        event_count=3,
        first_timestamp_us=100,
        last_timestamp_us=200,
        attributes={
            "timestamp_reference": "RVT recording clock",
            "timestamp_synchronized": True,
            "timestamp_repair_policy": "running_max",
        },
    )

    class LegacyHandle:
        attrs = {
            "sequence_id": metadata.sequence_id,
            "source_recording_id": "gen4__recording",
            "source_dataset": metadata.dataset,
            "source_path": str(source_path.resolve()),
            "source_file_size": source_path.stat().st_size,
            "source_mtime_ns": source_path.stat().st_mtime_ns,
            "camera": metadata.camera,
            "source_width": metadata.width,
            "source_height": metadata.height,
            "source_time_origin_us": metadata.first_timestamp_us,
            "duration_us": 100,
            "source_event_count": metadata.event_count,
            "logical_split": "train",
            "converter_config_sha256": "config",
            "complete": True,
        }

    _validate_identity_attributes(LegacyHandle(), metadata, "train", "config")


def test_sequence_identifier_is_dataset_and_camera_qualified() -> None:
    assert sequence_identifier("m3ed", "car urban/01", "left") == (
        "m3ed__car_urban_01__left"
    )


def test_sequence_time_origin_round_trip() -> None:
    info = SequenceInfo(
        "dsec__zurich_city_00_a__left",
        None,
        480,
        640,
        0,
        1_000,
        source_time_origin_us=42_000,
        coordinate_frame="distorted",
        source_width=640,
        source_height=480,
    )
    assert info.source_to_internal_time(42_123) == 123
    assert info.internal_to_source_time(123) == 42_123


def test_m3ed_official_split_parser_keeps_test_boundary(tmp_path) -> None:
    dataset_list = tmp_path / "dataset_list.yaml"
    dataset_list.write_text(
        "- file: camera_calibration\n  filetype: camera_calib\n"
        "- file: car_train\n  filetype: data\n  is_test_file: false\n"
        "- file: falcon_test\n  filetype: data\n  is_test_file: true\n",
        encoding="utf-8",
    )
    assert _m3ed_official_names(dataset_list, "train") == {"car_train"}
    assert _m3ed_official_names(dataset_list, "val") == {"car_train"}
    assert _m3ed_official_names(dataset_list, "test") == {"falcon_test"}


def test_dat_split_comes_from_nearest_official_directory(tmp_path) -> None:
    assert _dat_path_split(tmp_path / "train" / "recording_td.dat") == "train"
    assert _dat_path_split(tmp_path / "validation" / "recording_td.dat") == "val"
    assert _dat_path_split(tmp_path / "unassigned" / "recording_td.dat") is None


def test_1mpx_default_downsample_and_bbox_pairing(tmp_path) -> None:
    dat = tmp_path / "recording_td.dat"
    bbox = tmp_path / "recording_bbox.npy"
    dat.write_bytes(b"events")
    bbox.write_bytes(b"labels")

    assert _resolved_spatial_downsample("prophesee_1mpx", None) == 2
    assert _resolved_spatial_downsample("gen4", None) == 2
    assert _resolved_spatial_downsample("m3ed", None) == 2
    assert _resolved_spatial_downsample("gen1", None) == 1
    assert _resolved_spatial_downsample_method("gen4", 2, None) == (
        "area_accumulate"
    )
    assert _resolved_spatial_downsample_method("m3ed", 2, None) == (
        "area_accumulate"
    )
    assert _resolved_spatial_downsample_method("dsec", 1, None) == "coordinate"
    assert _resolved_spatial_downsample_method("gen1", 1, None) == "coordinate"
    assert _resolved_spatial_downsample_method("gen4", 2, "coordinate") == (
        "coordinate"
    )
    assert _matching_bbox(dat) == bbox

    gen4_h5 = tmp_path / "recording_td.h5"
    gen1_h5 = tmp_path / "recording_td.dat.h5"
    gen4_h5.write_bytes(b"hdf5")
    gen1_h5.write_bytes(b"hdf5")
    assert _matching_bbox(gen4_h5) == bbox
    assert _matching_bbox(gen1_h5) == bbox


def test_bbox_copy_and_manifest_paths_are_portable(tmp_path) -> None:
    source_bbox = tmp_path / "raw" / "recording_bbox.npy"
    source_bbox.parent.mkdir()
    source_bbox.write_bytes(b"npy payload")
    bundle = tmp_path / "bundle"
    copied_bbox = bundle / "labels" / "train" / "recording_bbox.npy"
    _copy_bbox_atomic(source_bbox, copied_bbox)
    assert copied_bbox.read_bytes() == b"npy payload"

    event_path = bundle / "events" / "train" / "recording.h5"
    event_path.parent.mkdir(parents=True)
    event_path.write_bytes(b"hdf5 placeholder")
    manifest = bundle / "manifests" / "train.jsonl"
    write_manifest(
        [
            {
                "sequence_id": "prophesee_1mpx__recording_td__left",
                "source_recording_id": "prophesee_1mpx__recording_td",
                "path": str(event_path),
                "bbox_path": str(copied_bbox),
                "split": "train",
            }
        ],
        manifest,
    )
    row = json.loads(manifest.read_text(encoding="utf-8"))
    assert row["path"] == "../events/train/recording.h5"
    assert row["bbox_path"] == "../labels/train/recording_bbox.npy"

    merged = bundle / "merged.jsonl"
    merge_manifests([manifest], merged)
    merged_row = json.loads(merged.read_text(encoding="utf-8"))
    assert merged_row["path"] == "events/train/recording.h5"
    assert merged_row["bbox_path"] == "labels/train/recording_bbox.npy"

    second_event = bundle / "events" / "train" / "second.h5"
    second_event.write_bytes(b"hdf5 placeholder 2")
    write_manifest(
        [
            {
                "sequence_id": "prophesee_1mpx__second_td__left",
                "source_recording_id": "prophesee_1mpx__second_td",
                "path": str(second_event),
                "split": "train",
            }
        ],
        manifest,
        merge_existing=True,
    )
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
    ]
    assert {row["sequence_id"] for row in rows} == {
        "prophesee_1mpx__recording_td__left",
        "prophesee_1mpx__second_td__left",
    }

    with pytest.raises(ValueError, match="changes identity field split"):
        write_manifest(
            [
                {
                    "sequence_id": "prophesee_1mpx__recording_td__left",
                    "source_recording_id": "prophesee_1mpx__recording_td",
                    "path": str(event_path),
                    "split": "val",
                }
            ],
            manifest,
            merge_existing=True,
        )


def test_m3ed_label_discovery_and_atomic_copy(tmp_path) -> None:
    raw = tmp_path / "raw" / "car_urban_day_horse"
    raw.mkdir(parents=True)
    data = raw / "car_urban_day_horse_data.h5"
    depth = raw / "car_urban_day_horse_depth_gt.h5"
    pose = raw / "car_urban_day_horse_pose_gt.h5"
    data.write_bytes(b"events")
    depth.write_bytes(b"depth")
    pose.write_bytes(b"pose")

    assert discover_m3ed_labels(data) == {"depth": depth, "pose": pose}
    destination = tmp_path / "bundle" / "labels" / "depth" / depth.name
    copy_file_atomic(depth, destination)
    copy_file_atomic(depth, destination)
    assert destination.read_bytes() == b"depth"


def test_m3ed_experiment_split_preserves_storage_split_and_portable_paths(
    tmp_path,
) -> None:
    bundle = tmp_path / "bundle"
    events = bundle / "events" / "storage_train"
    labels = bundle / "labels" / "depth"
    calibration = bundle / "calibration"
    events.mkdir(parents=True)
    labels.mkdir(parents=True)
    calibration.mkdir(parents=True)
    source_names = ("train_recording", "val_recording", "test_recording")
    records = []
    for name in source_names:
        event_path = events / f"m3ed__{name}__left.h5"
        depth_path = labels / f"{name}_depth_gt.h5"
        calibration_path = calibration / f"{name}_calibration.h5"
        event_path.write_bytes(b"event")
        depth_path.write_bytes(b"depth")
        calibration_path.write_bytes(b"calibration")
        records.append(
            {
                "sequence_id": f"m3ed__{name}__left",
                "source_recording_id": f"m3ed__{name}",
                "source_sequence_name": name,
                "path": str(event_path),
                "depth_path": str(depth_path),
                "calibration_path": str(calibration_path),
                "dataset": "m3ed",
                "camera": "left",
                "split": "train",
                "storage_split": "train",
            }
        )
    storage_manifest = bundle / "manifests" / "storage_train.jsonl"
    write_manifest(records, storage_manifest)

    dataset_list = tmp_path / "dataset_list.yaml"
    dataset_list.write_text(
        "".join(
            f"- file: {name}\n  filetype: data\n  is_test_file: false\n"
            for name in source_names
        ),
        encoding="utf-8",
    )
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text(
        "name: unit_test\nsplits:\n"
        "  train: [train_recording]\n"
        "  val: [val_recording]\n"
        "  test: [test_recording]\n",
        encoding="utf-8",
    )
    output_dir = bundle / "manifests" / "protocol"
    assert create_m3ed_split_manifests(
        storage_manifest, protocol, dataset_list, output_dir
    ) == {"train": 1, "val": 1, "test": 1}

    val_row = json.loads((output_dir / "val.jsonl").read_text(encoding="utf-8"))
    assert val_row["split"] == "val"
    assert val_row["storage_split"] == "train"
    assert val_row["path"] == "../../events/storage_train/m3ed__val_recording__left.h5"
    assert val_row["depth_path"] == "../../labels/depth/val_recording_depth_gt.h5"
    assert val_row["calibration_path"] == "../../calibration/val_recording_calibration.h5"


def test_bbox_validation_checks_structure_and_native_bounds(tmp_path) -> None:
    bbox = tmp_path / "recording_bbox.npy"
    values = np.array(
        [(100, 10.0, 20.0, 30.0, 40.0)],
        dtype=[
            ("t", "<i8"),
            ("x", "<f4"),
            ("y", "<f4"),
            ("w", "<f4"),
            ("h", "<f4"),
        ],
    )
    np.save(bbox, values, allow_pickle=False)
    assert _validate_bbox(bbox, source_width=1280, source_height=720) == {
        "bbox_count": 1,
        "bbox_out_of_fov_count": 0,
        "bbox_requires_fov_clip": False,
        "bbox_timestamp_field": "t",
    }

    values["w"] = 2_000
    np.save(bbox, values, allow_pickle=False)
    metadata = _validate_bbox(bbox, source_width=1280, source_height=720)
    assert metadata["bbox_out_of_fov_count"] == 1
    assert metadata["bbox_requires_fov_clip"] is True

    values["w"] = 0
    np.save(bbox, values, allow_pickle=False)
    with pytest.raises(ValueError, match="width and height must be positive"):
        _validate_bbox(bbox, source_width=1280, source_height=720)


def test_output_lock_rejects_concurrent_writer_and_recovers(tmp_path) -> None:
    output = tmp_path / "recording.h5"
    with _exclusive_output_lock(output):
        with pytest.raises(RuntimeError, match="another converter owns"):
            with _exclusive_output_lock(output):
                pass
    with _exclusive_output_lock(output):
        pass


def test_manifest_rejects_cameras_of_one_recording_in_different_splits(
    tmp_path,
) -> None:
    base = {
        "path": str(tmp_path / "events.h5"),
        "height": 720,
        "width": 1280,
        "t_start_us": 0,
        "t_end_us": 1_000,
        "dataset": "m3ed",
        "source_recording_id": "m3ed__car_urban_day",
    }
    with pytest.raises(ValueError, match="cannot cross logical splits"):
        write_manifest(
            [
                {**base, "sequence_id": "m3ed__car_urban_day__left", "split": "train"},
                {**base, "sequence_id": "m3ed__car_urban_day__right", "split": "val"},
            ],
            tmp_path / "manifest.jsonl",
        )
    assert not (tmp_path / "manifest.jsonl").exists()
