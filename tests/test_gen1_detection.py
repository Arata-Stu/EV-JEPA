from __future__ import annotations

from pathlib import Path

import numpy as np

from event_window_jepa.downstream.gen1_detection import (
    _ground_truth_array,
    _scaled_full_boxes,
)
from event_window_jepa.downstream.gen1_roi_probe import LabelSource


def _source(path: Path) -> LabelSource:
    return LabelSource(
        sequence_id="gen1__sample",
        path=path,
        timestamp_field="t",
        class_field="class_id",
        timestamps_relative=True,
        source_time_origin_us=0,
        bbox_width=608,
        bbox_height=480,
        event_width=304,
        event_height=240,
        t_start_us=0,
        t_end_us=1_000_000,
    )


def test_scaled_full_boxes_maps_label_resolution_and_clips(tmp_path: Path) -> None:
    labels = np.array(
        [(600_000, -10.0, 20.0, 80.0, 40.0, 1)],
        dtype=[
            ("t", "<i8"),
            ("x", "<f4"),
            ("y", "<f4"),
            ("w", "<f4"),
            ("h", "<f4"),
            ("class_id", "<i2"),
        ],
    )
    boxes, classes = _scaled_full_boxes(labels, _source(tmp_path / "labels.npy"))
    np.testing.assert_allclose(boxes, [[0.0, 10.0, 35.0, 30.0]])
    np.testing.assert_array_equal(classes, [1])


def test_ground_truth_uses_internal_timestamp() -> None:
    boxes = np.array([[1.0, 2.0, 11.0, 22.0]], dtype=np.float32)
    values = _ground_truth_array(boxes, np.array([0]), timestamp_us=700_000)
    assert values["t"].tolist() == [700_000]
    assert values["w"].tolist() == [10.0]
    assert values["h"].tolist() == [20.0]
    assert values["class_confidence"].tolist() == [1.0]
