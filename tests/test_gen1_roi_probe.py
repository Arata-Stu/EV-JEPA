from __future__ import annotations

from pathlib import Path

import numpy as np

from event_window_jepa.data.spatial_transforms import SpatialTransformParameters
from event_window_jepa.downstream.gen1_roi_probe import (
    LabelSource,
    _crop_boxes,
    _frame_references,
    _window_group,
)


def _source(path: Path) -> LabelSource:
    return LabelSource(
        sequence_id="gen1__sample",
        path=path,
        timestamp_field="t",
        class_field="class_id",
        timestamps_relative=False,
        source_time_origin_us=1_000,
        bbox_width=304,
        bbox_height=240,
        event_width=304,
        event_height=240,
        t_start_us=0,
        t_end_us=20_000,
    )


def test_frame_references_group_timestamps_and_convert_origin(tmp_path: Path) -> None:
    path = tmp_path / "labels.npy"
    labels = np.array(
        [(2_000, 0), (7_000, 0), (7_000, 1), (22_000, 0)],
        dtype=[("t", "<i8"), ("class_id", "<i2")],
    )
    np.save(path, labels, allow_pickle=False)
    references = _frame_references(
        (_source(path),), maximum_window_us=5_000, maximum_frames=0, seed=0
    )
    assert [(item.start, item.stop, item.t_end_us) for item in references] == [
        (1, 3, 6_000),
    ]


def test_crop_boxes_maps_to_center_crop_and_removes_invisible_boxes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "labels.npy"
    labels = np.array(
        [
            (1_000, 10.0, 20.0, 20.0, 10.0, 0),
            (1_000, 100.0, 40.0, 32.0, 16.0, 1),
        ],
        dtype=[
            ("t", "<i8"),
            ("x", "<f4"),
            ("y", "<f4"),
            ("w", "<f4"),
            ("h", "<f4"),
            ("class_id", "<i2"),
        ],
    )
    crop = SpatialTransformParameters(40, 8, 224, 224, False)
    boxes, classes = _crop_boxes(labels, _source(path), crop)
    np.testing.assert_allclose(boxes, [[60.0, 32.0, 92.0, 48.0]])
    np.testing.assert_array_equal(classes, [1])


def test_window_group_distinguishes_interpolation_and_extrapolation() -> None:
    trained = (10.0, 20.0, 40.0, 80.0)
    assert _window_group(40.0, trained) == "seen"
    assert _window_group(30.0, trained) == "unseen_interpolation"
    assert _window_group(5.0, trained) == "unseen_extrapolation"
    assert _window_group(120.0, trained) == "unseen_extrapolation"
