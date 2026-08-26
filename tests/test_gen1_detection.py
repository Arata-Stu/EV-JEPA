from __future__ import annotations

from pathlib import Path

import numpy as np

from event_window_jepa.downstream.gen1_detection import (
    _ground_truth_array,
    _scaled_full_boxes,
    _stream_references,
)
from event_window_jepa.downstream.gen1_roi_probe import FrameReference, LabelSource


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


def test_stream_references_fill_gaps_and_reset_only_at_sequence_start(
    tmp_path: Path,
) -> None:
    first = _source(tmp_path / "first.npy")
    second = LabelSource(
        **{
            **first.__dict__,
            "sequence_id": "gen1__second",
            "path": tmp_path / "second.npy",
        }
    )
    labeled = (
        FrameReference(0, 0, 1, 100_000),
        FrameReference(0, 1, 2, 250_000),
        FrameReference(1, 0, 1, 150_000),
    )
    references = _stream_references(
        (first, second),
        labeled,
        duration_us=50_000,
        maximum_labeled_frames=0,
    )
    assert [value.t_end_us for value in references] == [
        50_000,
        100_000,
        150_000,
        200_000,
        250_000,
        50_000,
        100_000,
        150_000,
    ]
    assert [value.has_labels for value in references] == [
        False,
        True,
        False,
        False,
        True,
        False,
        False,
        True,
    ]
    assert [value.state_reset for value in references] == [
        True,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
    ]


def test_stream_reference_limit_keeps_ordered_labeled_prefix(tmp_path: Path) -> None:
    source = _source(tmp_path / "labels.npy")
    references = _stream_references(
        (source,),
        (
            FrameReference(0, 0, 1, 100_000),
            FrameReference(0, 1, 2, 250_000),
        ),
        duration_us=50_000,
        maximum_labeled_frames=1,
    )
    assert [value.t_end_us for value in references] == [50_000, 100_000]
    assert sum(value.has_labels for value in references) == 1
