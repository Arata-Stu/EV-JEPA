from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest
import torch

from event_window_jepa.downstream.gen1_detection import (
    DirectStreamLaneIndex,
    MixedStatefulBatchSampler,
    RandomLabelClipBatchSampler,
    StreamFrameReference,
    StreamLaneBatchSampler,
    StreamStateManager,
    WindowJEPAYOLOX,
    _collate_stream_sequence_detection,
    _collate_stream_detection,
    _concatenate_padded_targets,
    _forward_stateful_sequence,
    _ground_truth_array,
    _manifest_identity,
    _scaled_full_boxes,
    _stream_references,
    _stateful_batch_partition,
    _stateful_sampling_counts,
    _training_stream_references,
    _validate_args,
    _validate_detection_resume_metadata,
    _validate_stateful_window_duration,
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


def test_stream_reference_limit_exercises_multiple_lanes(tmp_path: Path) -> None:
    sources = tuple(
        LabelSource(
            **{
                **_source(tmp_path / f"{index}.npy").__dict__,
                "sequence_id": f"gen1__{index}",
            }
        )
        for index in range(3)
    )
    labeled = tuple(
        FrameReference(source_index, step, step + 1, (step + 2) * 50_000)
        for source_index in range(3)
        for step in range(3)
    )
    references = _stream_references(
        sources,
        labeled,
        duration_us=50_000,
        maximum_labeled_frames=4,
        stream_lanes=2,
    )
    labeled_output = [reference for reference in references if reference.has_labels]
    assert len(labeled_output) == 4
    assert {reference.source_index for reference in labeled_output} == {0, 1}
    for source_index in (0, 1):
        timestamps = [
            reference.t_end_us
            for reference in labeled_output
            if reference.source_index == source_index
        ]
        assert timestamps == [100_000, 150_000]


def test_stream_reference_limit_refills_quota_from_longer_recording(
    tmp_path: Path,
) -> None:
    sources = tuple(
        LabelSource(
            **{
                **_source(tmp_path / f"{index}.npy").__dict__,
                "sequence_id": f"gen1__{index}",
            }
        )
        for index in range(2)
    )
    labeled = (
        FrameReference(0, 0, 1, 100_000),
        FrameReference(1, 0, 1, 100_000),
        FrameReference(1, 1, 2, 150_000),
        FrameReference(1, 2, 3, 200_000),
    )
    references = _stream_references(
        sources,
        labeled,
        duration_us=50_000,
        maximum_labeled_frames=4,
        stream_lanes=2,
    )
    assert sum(reference.has_labels for reference in references) == 4


def test_stream_references_reject_misaligned_label_timestamp(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not aligned"):
        _stream_references(
            (_source(tmp_path / "labels.npy"),),
            (
                FrameReference(0, 0, 1, 125_000),
                FrameReference(0, 1, 2, 160_000),
            ),
            duration_us=50_000,
            maximum_labeled_frames=0,
        )


def test_stream_references_accept_gen1_inclusive_boundary_phase(
    tmp_path: Path,
) -> None:
    references = _stream_references(
        (_source(tmp_path / "labels.npy"),),
        (
            FrameReference(0, 0, 1, 99_999),
            FrameReference(0, 1, 2, 199_999),
        ),
        duration_us=50_000,
        maximum_labeled_frames=0,
    )
    assert [reference.t_end_us for reference in references] == [
        99_999,
        149_999,
        199_999,
    ]
    assert [reference.has_labels for reference in references] == [True, False, True]


def test_training_streams_split_label_gaps_and_guarantee_each_chunk_has_labels(
    tmp_path: Path,
) -> None:
    references = _training_stream_references(
        (_source(tmp_path / "labels.npy"),),
        (
            FrameReference(0, 0, 1, 150_000),
            FrameReference(0, 1, 2, 250_000),
            FrameReference(0, 2, 3, 450_000),
        ),
        duration_us=50_000,
        sequence_length=3,
        maximum_labeled_frames=0,
        stream_lanes=1,
    )
    by_segment: dict[int, list[StreamFrameReference]] = {}
    for reference in references:
        by_segment.setdefault(reference.stream_segment_id, []).append(reference)

    assert [value.t_end_us for value in by_segment[0]] == [
        50_000,
        100_000,
        150_000,
        200_000,
        250_000,
    ]
    assert [value.t_end_us for value in by_segment[1]] == [
        350_000,
        400_000,
        450_000,
    ]
    for segment in by_segment.values():
        assert segment[0].state_reset is True
        assert all(not value.state_reset for value in segment[1:])
        for start in range(0, len(segment), 3):
            assert any(value.has_labels for value in segment[start : start + 3])


def test_stateful_window_must_match_sequence_checkpoint_cadence() -> None:
    _validate_stateful_window_duration(
        50.0,
        stateful=True,
        sequence_loader=True,
        checkpoint_ms=50.0,
    )
    with pytest.raises(ValueError, match="checkpoint cadence"):
        _validate_stateful_window_duration(
            40.0,
            stateful=True,
            sequence_loader=True,
            checkpoint_ms=50.0,
        )
    _validate_stateful_window_duration(
        40.0,
        stateful=True,
        sequence_loader=False,
        checkpoint_ms=50.0,
    )


def test_stateful_arguments_allow_multiple_recording_lanes() -> None:
    args = argparse.Namespace(
        window_ms=50.0,
        batch_size=8,
        sequence_length=21,
        stateful_sampling="mixed",
        stream_batch_size=0,
        epochs=30,
        learning_rate=2e-4,
        eval_every=5,
        workers=4,
        max_train_frames=0,
        max_val_frames=0,
        confidence_threshold=0.001,
        nms_threshold=0.45,
        stateful=True,
        unfreeze_backbone=False,
    )
    _validate_args(args)
    args.sequence_length = 0
    with pytest.raises(ValueError, match="sequence length"):
        _validate_args(args)
    args.sequence_length = 21
    args.max_train_frames = 4
    with pytest.raises(ValueError, match="complete batch"):
        _validate_args(args)


def test_stateful_batch_partition_defaults_to_rvt_one_to_one() -> None:
    args = argparse.Namespace(
        batch_size=8,
        stateful_sampling="mixed",
        stream_batch_size=0,
    )
    assert _stateful_batch_partition(args) == (4, 4)
    args.stateful_sampling = "stream"
    assert _stateful_batch_partition(args) == (8, 0)
    args.stateful_sampling = "stream_reset"
    assert _stateful_batch_partition(args) == (8, 0)
    args.stateful_sampling = "random"
    assert _stateful_batch_partition(args) == (0, 8)
    args.stateful_sampling = "mixed"
    args.stream_batch_size = 4
    assert _stateful_batch_partition(args) == (4, 4)
    args.stream_batch_size = 3
    with pytest.raises(ValueError, match="exact 1:1"):
        _stateful_batch_partition(args)
    args.stream_batch_size = 0
    args.batch_size = 7
    with pytest.raises(ValueError, match="even batch size"):
        _stateful_batch_partition(args)


def test_detection_resume_validates_backbone_and_lane_identity(tmp_path: Path) -> None:
    pretrain = tmp_path / "pretrain.pt"
    train_manifest = tmp_path / "train.jsonl"
    val_manifest = tmp_path / "val.jsonl"
    train_manifest.write_text('{"split":"train"}\n', encoding="utf-8")
    val_manifest.write_text('{"split":"val"}\n', encoding="utf-8")
    args = argparse.Namespace(
        checkpoint=pretrain,
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        backbone_init="pretrained",
        window_ms=50.0,
        batch_size=8,
        sequence_length=21,
        stateful_sampling="mixed",
        stream_batch_size=0,
        stateful=True,
        seed=17,
        precision="fp32",
        max_train_frames=0,
        max_val_frames=0,
    )
    resumed = {
        "schema": "event-window-jepa-gen1-yolox-v2",
        "pretrain_checkpoint": str(pretrain.resolve()),
        "pretrain_config_hash": "config-a",
        "backbone_fingerprint": "weights-a",
        "backbone_init": "pretrained",
        "window_ms": 50.0,
        "batch_size": 8,
        "stateful": True,
        "stateful_lane_schema": "rvt-label-aware-sampling-v3",
        "sequence_length": 21,
        "stateful_sampling": "mixed",
        "stream_batch_size": 4,
        "seed": 17,
        "precision": "fp32",
        "max_train_frames": 0,
        "max_val_frames": 0,
        "train_manifest_identity": _manifest_identity(train_manifest),
        "val_manifest_identity": _manifest_identity(val_manifest),
    }
    _validate_detection_resume_metadata(
        resumed,
        args,
        pretrain_config_hash="config-a",
        backbone_fingerprint="weights-a",
    )
    with pytest.raises(ValueError, match="backbone_fingerprint"):
        _validate_detection_resume_metadata(
            resumed,
            args,
            pretrain_config_hash="config-a",
            backbone_fingerprint="weights-b",
        )
    resumed["sequence_length"] = 7
    with pytest.raises(ValueError, match="sequence_length"):
        _validate_detection_resume_metadata(
            resumed,
            args,
            pretrain_config_hash="config-a",
            backbone_fingerprint="weights-a",
        )
    resumed["sequence_length"] = 21
    resumed["stateful_sampling"] = "stream"
    with pytest.raises(ValueError, match="stateful_sampling"):
        _validate_detection_resume_metadata(
            resumed,
            args,
            pretrain_config_hash="config-a",
            backbone_fingerprint="weights-a",
        )
    resumed["stateful_sampling"] = "mixed"
    resumed["stream_batch_size"] = 3
    with pytest.raises(ValueError, match="stream_batch_size"):
        _validate_detection_resume_metadata(
            resumed,
            args,
            pretrain_config_hash="config-a",
            backbone_fingerprint="weights-a",
        )
    resumed["stream_batch_size"] = 4
    train_manifest.write_text('{"split":"different"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="train_manifest_identity"):
        _validate_detection_resume_metadata(
            resumed,
            args,
            pretrain_config_hash="config-a",
            backbone_fingerprint="weights-a",
        )


def test_legacy_stateful_resume_is_rejected_without_sampling_identity(
    tmp_path: Path,
) -> None:
    pretrain = tmp_path / "pretrain.pt"
    args = argparse.Namespace(
        checkpoint=pretrain,
        backbone_init="pretrained",
        window_ms=50.0,
        batch_size=8,
        sequence_length=1,
        stateful_sampling="stream",
        stream_batch_size=0,
        stateful=True,
        seed=17,
        precision="fp32",
        max_train_frames=0,
        max_val_frames=0,
    )
    resumed = {
        "schema": "event-window-jepa-gen1-yolox-v1",
        "pretrain_checkpoint": str(pretrain.resolve()),
        "backbone_init": "pretrained",
        "window_ms": 50.0,
        "stateful": True,
    }
    with pytest.raises(ValueError, match="predates sampling identity"):
        _validate_detection_resume_metadata(
            resumed,
            args,
            pretrain_config_hash="config-a",
            backbone_fingerprint="weights-a",
        )


def test_legacy_stateless_resume_keeps_v1_compatibility(tmp_path: Path) -> None:
    args = argparse.Namespace(
        checkpoint=tmp_path / "different-pretrain.pt",
        backbone_init="random",
        window_ms=80.0,
        batch_size=16,
        sequence_length=21,
        stateful_sampling="mixed",
        stream_batch_size=0,
        stateful=False,
        seed=99,
        precision="bf16",
        max_train_frames=0,
        max_val_frames=0,
    )
    _validate_detection_resume_metadata(
        {
            "schema": "event-window-jepa-gen1-yolox-v1",
            "stateful": False,
        },
        args,
        pretrain_config_hash="config-new",
        backbone_fingerprint="weights-new",
    )


def _stream_reference(
    source_index: int,
    timestamp: int,
    *,
    reset: bool,
    labeled: bool = False,
) -> StreamFrameReference:
    return StreamFrameReference(
        source_index=source_index,
        start=0,
        stop=1 if labeled else 0,
        t_end_us=timestamp,
        has_labels=labeled,
        state_reset=reset,
    )


def test_stream_lane_sampler_refills_stable_lanes_and_keeps_a_short_tail() -> None:
    references = (
        _stream_reference(0, 50_000, reset=True),
        _stream_reference(0, 100_000, reset=False),
        _stream_reference(1, 50_000, reset=True),
        _stream_reference(2, 50_000, reset=True),
        _stream_reference(2, 100_000, reset=False),
    )
    sampler = StreamLaneBatchSampler(references, batch_size=2, sequence_length=1)
    batches = list(sampler)

    assert len(sampler) == len(batches) == 3
    assert [
        [(item.reference_index, item.lane_id) for item in batch]
        for batch in batches
    ] == [
        [(0, 0), (3, 1)],
        [(1, 0), (4, 1)],
        [(2, 0)],
    ]
    assert sorted(
        item.reference_index for batch in batches for item in batch
    ) == list(range(len(references)))


def test_stream_lane_sampler_preserves_batch_one_behavior() -> None:
    references = (
        _stream_reference(0, 50_000, reset=True),
        _stream_reference(0, 100_000, reset=False),
        _stream_reference(1, 50_000, reset=True),
    )
    batches = list(
        StreamLaneBatchSampler(references, batch_size=1, sequence_length=1)
    )
    assert [[item.reference_index for item in batch] for batch in batches] == [
        [0],
        [1],
        [2],
    ]
    assert all(batch[0].lane_id == 0 for batch in batches)


def test_stream_lane_sampler_epoch_shuffle_is_deterministic_and_causal() -> None:
    references = tuple(
        _stream_reference(source, step * 50_000, reset=step == 1)
        for source in range(6)
        for step in range(1, 4)
    )
    first = StreamLaneBatchSampler(
        references, batch_size=2, sequence_length=1, shuffle=True, seed=17
    )
    second = StreamLaneBatchSampler(
        references, batch_size=2, sequence_length=1, shuffle=True, seed=17
    )
    first.set_epoch(3)
    second.set_epoch(3)
    first_batches = list(first)
    assert first_batches == list(second)
    assert len(first) == len(first_batches)

    previous_by_lane: dict[int, StreamFrameReference] = {}
    visited: list[int] = []
    for batch in first_batches:
        for item in batch:
            reference = references[item.reference_index]
            previous = previous_by_lane.get(item.lane_id)
            if previous is not None and previous.source_index == reference.source_index:
                assert reference.state_reset is False
                assert reference.t_end_us == previous.t_end_us + 50_000
            else:
                assert reference.state_reset is True
            previous_by_lane[item.lane_id] = reference
            visited.append(item.reference_index)
    assert sorted(visited) == list(range(len(references)))

    first.set_epoch(4)
    assert list(first) != first_batches


@pytest.mark.parametrize("shuffle", [False, True])
def test_stream_lane_sampler_length_matches_iteration_for_uneven_groups(
    shuffle: bool,
) -> None:
    references = tuple(
        _stream_reference(source, step * 50_000, reset=step == 1)
        for source, length in enumerate((7, 6, 5, 4, 3, 2, 1))
        for step in range(1, length + 1)
    )
    sampler = StreamLaneBatchSampler(
        references,
        batch_size=3,
        sequence_length=1,
        shuffle=shuffle,
        seed=29,
    )
    for epoch in range(3):
        sampler.set_epoch(epoch)
        assert len(sampler) == len(list(sampler))


def test_stream_lane_sampler_yields_time_major_chunks_and_refills_at_boundary() -> None:
    references = tuple(
        _stream_reference(source, step * 50_000, reset=step == 1)
        for source, length in enumerate((5, 3, 2))
        for step in range(1, length + 1)
    )
    sampler = StreamLaneBatchSampler(
        references,
        batch_size=2,
        sequence_length=2,
    )
    batches = list(sampler)

    assert len(sampler) == len(batches) == 3
    assert [
        [(item.reference_index, item.lane_id) for item in batch]
        for batch in batches
    ] == [
        [(0, 0), (5, 1), (1, 0), (6, 1)],
        [(2, 0), (7, 1), (3, 0)],
        [(4, 0), (8, 1), (9, 1)],
    ]
    assert sorted(item.reference_index for batch in batches for item in batch) == list(
        range(len(references))
    )


def test_stream_sampler_never_crosses_label_gap_segments() -> None:
    references = (
        _stream_reference(0, 50_000, reset=True, labeled=True),
        StreamFrameReference(
            source_index=0,
            start=1,
            stop=2,
            t_end_us=250_000,
            has_labels=True,
            state_reset=True,
            stream_segment_id=1,
        ),
    )
    batches = list(
        StreamLaneBatchSampler(references, batch_size=1, sequence_length=2)
    )
    assert [[item.reference_index for item in batch] for batch in batches] == [
        [0],
        [1],
    ]


def test_stream_and_stream_reset_have_identical_chunks_but_different_boundaries() -> None:
    references = tuple(
        _stream_reference(source, step * 50_000, reset=step == 1, labeled=True)
        for source in range(2)
        for step in range(1, 5)
    )
    carry = StreamLaneBatchSampler(
        references,
        batch_size=2,
        sequence_length=2,
        shuffle=True,
        seed=31,
    )
    reset = StreamLaneBatchSampler(
        references,
        batch_size=2,
        sequence_length=2,
        reset_every_chunk=True,
        shuffle=True,
        seed=31,
    )
    carry.set_epoch(4)
    reset.set_epoch(4)
    carry_batches = list(carry)
    reset_batches = list(reset)
    assert [
        [(item.reference_index, item.lane_id) for item in batch]
        for batch in carry_batches
    ] == [
        [(item.reference_index, item.lane_id) for item in batch]
        for batch in reset_batches
    ]
    assert all(not item.force_state_reset for batch in carry_batches for item in batch)
    for batch in reset_batches:
        first_by_lane: dict[int, StreamLaneIndex] = {}
        for item in batch:
            first_by_lane.setdefault(item.lane_id, item)
        assert all(item.force_state_reset for item in first_by_lane.values())


def test_random_sampler_builds_complete_reset_clips_ending_at_labels(
    tmp_path: Path,
) -> None:
    sampler = RandomLabelClipBatchSampler(
        (_source(tmp_path / "labels.npy"),),
        (
            FrameReference(0, 0, 1, 100_000),
            FrameReference(0, 1, 2, 150_000),
            FrameReference(0, 2, 3, 200_000),
            FrameReference(0, 3, 4, 300_000),
        ),
        duration_us=50_000,
        sequence_length=3,
        batch_size=2,
        seed=13,
    )
    batches = list(sampler)
    assert len(sampler.anchors) == 3
    assert len(batches) == 2
    observed_endpoints: set[int] = set()
    observed_internal_label = False
    for batch in batches:
        by_lane: dict[int, list[StreamFrameReference]] = {}
        for item in batch:
            by_lane.setdefault(item.lane_id, []).append(item.reference)
        for clip in by_lane.values():
            assert len(clip) == 3
            assert [value.state_reset for value in clip] == [True, False, False]
            assert all(value.sampling_mode == "random" for value in clip)
            assert clip[-1].has_labels is True
            observed_endpoints.add(clip[-1].t_end_us)
            observed_internal_label |= any(value.has_labels for value in clip[:-1])
    assert observed_endpoints == {150_000, 200_000, 300_000}
    assert observed_internal_label is True


def test_mixed_sampler_is_one_to_one_and_keeps_real_stream_tail(
    tmp_path: Path,
) -> None:
    stream_references = tuple(
        _stream_reference(source, step * 50_000, reset=step == 1, labeled=True)
        for source, length in enumerate((3, 3, 1))
        for step in range(1, length + 1)
    )
    stream = StreamLaneBatchSampler(
        stream_references,
        batch_size=2,
        sequence_length=2,
        shuffle=False,
    )
    random_sampler = RandomLabelClipBatchSampler(
        (_source(tmp_path / "labels.npy"),),
        tuple(
            FrameReference(0, index, index + 1, timestamp)
            for index, timestamp in enumerate((100_000, 150_000, 200_000, 250_000))
        ),
        duration_us=50_000,
        sequence_length=2,
        batch_size=2,
        lane_offset=2,
        drop_last=True,
        seed=19,
    )
    mixed = MixedStatefulBatchSampler(stream, random_sampler)
    batches = list(mixed)
    assert len(batches) == len(mixed) == 3
    real_stream_indices: set[int] = set()
    for batch in batches:
        assert {item.lane_id for item in batch} == {0, 1, 2, 3}
        random_items = [
            item for item in batch if isinstance(item, DirectStreamLaneIndex)
        ]
        assert {item.lane_id for item in random_items} == {2, 3}
        first_random: dict[int, DirectStreamLaneIndex] = {}
        for item in random_items:
            first_random.setdefault(item.lane_id, item)
        assert all(item.reference.state_reset for item in first_random.values())
        real_stream_indices.update(
            item.reference_index
            for item in batch
            if isinstance(item, StreamLaneIndex) and not item.force_state_reset
        )
    # Every real tail chunk appears before a shorter lane cycles its own shard.
    assert real_stream_indices == set(range(len(stream_references)))


def test_stream_state_is_not_detached_until_sequence_boundary() -> None:
    manager = StreamStateManager(recurrent=True, stride_us=50_000)
    manager.update(
        lane_ids=(0,),
        source_indices=(3,),
        timestamps=(50_000,),
        state=torch.ones(1, 1, 1, 1, requires_grad=True),
        detach=False,
    )
    continued = manager.prepare(
        lane_ids=(0,),
        source_indices=(3,),
        timestamps=(100_000,),
        state_reset=torch.tensor([False]),
    )
    assert isinstance(continued, torch.Tensor)
    assert continued.requires_grad is True

    manager.detach()
    detached = manager.prepare(
        lane_ids=(0,),
        source_indices=(3,),
        timestamps=(100_000,),
        state_reset=torch.tensor([False]),
    )
    assert isinstance(detached, torch.Tensor)
    assert detached.requires_grad is False


def test_stream_state_manager_gathers_resets_and_detaches_gru_lanes() -> None:
    manager = StreamStateManager(recurrent=True, stride_us=50_000)
    assert manager.prepare(
        lane_ids=(0, 1),
        source_indices=(10, 11),
        timestamps=(50_000, 50_000),
        state_reset=torch.tensor([True, True]),
    ) is None

    first_state = torch.tensor([1.0, 2.0], requires_grad=True).reshape(2, 1, 1, 1)
    manager.update(
        lane_ids=(0, 1),
        source_indices=(10, 11),
        timestamps=(50_000, 50_000),
        state=first_state,
    )
    continued = manager.prepare(
        lane_ids=(0, 1),
        source_indices=(10, 11),
        timestamps=(100_000, 100_000),
        state_reset=torch.tensor([False, False]),
    )
    assert isinstance(continued, torch.Tensor)
    torch.testing.assert_close(continued.flatten(), torch.tensor([1.0, 2.0]))
    assert continued.requires_grad is False

    manager.update(
        lane_ids=(0, 1),
        source_indices=(10, 11),
        timestamps=(100_000, 100_000),
        state=torch.tensor([3.0, 4.0]).reshape(2, 1, 1, 1),
    )
    switched = manager.prepare(
        lane_ids=(0, 1),
        source_indices=(10, 12),
        timestamps=(150_000, 50_000),
        state_reset=torch.tensor([False, True]),
    )
    assert isinstance(switched, torch.Tensor)
    torch.testing.assert_close(switched.flatten(), torch.tensor([3.0, 0.0]))

    manager.update(
        lane_ids=(0, 1),
        source_indices=(10, 12),
        timestamps=(150_000, 50_000),
        state=torch.tensor([5.0, 6.0]).reshape(2, 1, 1, 1),
    )
    short_tail = manager.prepare(
        lane_ids=(1,),
        source_indices=(12,),
        timestamps=(100_000,),
        state_reset=torch.tensor([False]),
    )
    assert isinstance(short_tail, torch.Tensor)
    torch.testing.assert_close(short_tail.flatten(), torch.tensor([6.0]))


def test_stream_state_manager_supports_convlstm_and_feedforward() -> None:
    recurrent = StreamStateManager(recurrent=True, stride_us=50_000)
    reset = torch.tensor([True, True])
    assert recurrent.prepare(
        lane_ids=(0, 1),
        source_indices=(0, 1),
        timestamps=(50_000, 50_000),
        state_reset=reset,
    ) is None
    hidden = torch.tensor([1.0, 2.0]).reshape(2, 1, 1, 1)
    cell = torch.tensor([3.0, 4.0]).reshape(2, 1, 1, 1)
    recurrent.update(
        lane_ids=(0, 1),
        source_indices=(0, 1),
        timestamps=(50_000, 50_000),
        state=(hidden, cell),
    )
    gathered = recurrent.prepare(
        lane_ids=(0, 1),
        source_indices=(0, 2),
        timestamps=(100_000, 50_000),
        state_reset=torch.tensor([False, True]),
    )
    assert isinstance(gathered, tuple)
    torch.testing.assert_close(gathered[0].flatten(), torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(gathered[1].flatten(), torch.tensor([3.0, 0.0]))

    feedforward = StreamStateManager(recurrent=False, stride_us=50_000)
    assert feedforward.prepare(
        lane_ids=(0,),
        source_indices=(4,),
        timestamps=(50_000,),
        state_reset=torch.tensor([True]),
    ) is None
    feedforward.update(
        lane_ids=(0,), source_indices=(4,), timestamps=(50_000,), state=None
    )
    assert feedforward.prepare(
        lane_ids=(0,),
        source_indices=(4,),
        timestamps=(100_000,),
        state_reset=torch.tensor([False]),
    ) is None


def test_stream_state_manager_allows_independent_lanes_from_same_recording() -> None:
    manager = StreamStateManager(recurrent=False, stride_us=50_000)
    assert manager.prepare(
        lane_ids=(0, 1),
        source_indices=(4, 4),
        timestamps=(50_000, 150_000),
        state_reset=torch.tensor([True, True]),
    ) is None
    manager.update(
        lane_ids=(0, 1),
        source_indices=(4, 4),
        timestamps=(50_000, 150_000),
        state=None,
    )
    assert manager.prepare(
        lane_ids=(0, 1),
        source_indices=(4, 4),
        timestamps=(100_000, 200_000),
        state_reset=torch.tensor([False, False]),
    ) is None


def test_stream_state_manager_rejects_lane_discontinuity() -> None:
    manager = StreamStateManager(recurrent=False, stride_us=50_000)
    manager.update(
        lane_ids=(0,), source_indices=(1,), timestamps=(50_000,), state=None
    )
    with pytest.raises(ValueError, match="recording"):
        manager.prepare(
            lane_ids=(0,),
            source_indices=(2,),
            timestamps=(100_000,),
            state_reset=torch.tensor([False]),
        )
    with pytest.raises(ValueError, match="causal"):
        manager.prepare(
            lane_ids=(0,),
            source_indices=(1,),
            timestamps=(125_000,),
            state_reset=torch.tensor([False]),
        )


def test_stream_collate_keeps_per_row_masks_sources_and_lanes() -> None:
    empty_gt = np.zeros(0, dtype=np.dtype([("t", "<i8")]))
    batch = (
        (
            torch.zeros(2, 4, 4),
            torch.zeros(0, 5),
            empty_gt,
            50_000,
            False,
            True,
            3,
            0,
        ),
        (
            torch.ones(2, 4, 4),
            torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]),
            empty_gt,
            100_000,
            True,
            False,
            4,
            2,
        ),
    )
    images, targets, _, timestamps, labels, resets, sources, lanes = (
        _collate_stream_detection(batch)
    )
    assert images.shape == (2, 2, 4, 4)
    assert targets.shape == (2, 1, 5)
    assert timestamps == (50_000, 100_000)
    assert labels.tolist() == [False, True]
    assert resets.tolist() == [True, False]
    assert sources == (3, 4)
    assert lanes == (0, 2)


def test_stream_sequence_collate_transposes_lanes_into_time_steps() -> None:
    empty_gt = np.zeros(0, dtype=np.dtype([("t", "<i8")]))

    def sample(value: float, timestamp: int, lane: int) -> tuple[object, ...]:
        return (
            torch.full((2, 4, 4), value),
            torch.zeros(0, 5),
            empty_gt,
            timestamp,
            False,
            timestamp == 50_000,
            lane,
            lane,
        )

    sequence = _collate_stream_sequence_detection(
        (
            sample(0.0, 50_000, 0),
            sample(1.0, 50_000, 1),
            sample(2.0, 100_000, 0),
        )  # type: ignore[arg-type]
    )
    assert len(sequence) == 2
    assert sequence[0][0].shape == (2, 2, 4, 4)
    assert sequence[0][-1] == (0, 1)
    assert sequence[1][0].shape == (1, 2, 4, 4)
    assert sequence[1][3] == (100_000,)
    assert sequence[1][-1] == (0,)


def test_stateful_sampling_counts_report_sequence_not_label_ratio() -> None:
    empty_gt = np.zeros(0, dtype=np.dtype([("t", "<i8")]))

    def sample(
        lane: int, timestamp: int, labeled: bool
    ) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, int, bool, bool, int, int]:
        return (
            torch.zeros(2, 4, 4),
            torch.zeros(1 if labeled else 0, 5),
            empty_gt,
            timestamp,
            labeled,
            timestamp == 50_000,
            lane,
            lane,
        )

    sequence = _collate_stream_sequence_detection(
        (
            sample(0, 50_000, True),
            sample(1, 50_000, False),
            sample(2, 50_000, True),
            sample(3, 50_000, True),
            sample(0, 100_000, True),
            sample(1, 100_000, True),
            sample(2, 100_000, False),
            sample(3, 100_000, True),
        )
    )
    assert _stateful_sampling_counts(
        sequence, sampling="mixed", stream_batch_size=2
    ) == (3, 3, 2, 2)


def test_concatenate_padded_targets_preserves_all_labeled_rows() -> None:
    first = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]])
    second = torch.tensor(
        [
            [[6.0, 7.0, 8.0, 9.0, 10.0], [11.0, 12.0, 13.0, 14.0, 15.0]],
            [[16.0, 17.0, 18.0, 19.0, 20.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
        ]
    )
    output = _concatenate_padded_targets((first, second))
    assert output.shape == (3, 2, 5)
    torch.testing.assert_close(output[0, 0], first[0, 0])
    torch.testing.assert_close(output[0, 1], torch.zeros(5))
    torch.testing.assert_close(output[1:], second)


class _TestScaleEmbedding(torch.nn.Module):
    def forward(self, duration_ms: torch.Tensor) -> torch.Tensor:
        return duration_ms.reshape(-1, 1)


class _TestFeatureEncoder(torch.nn.Module):
    embed_dim = 4

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward_feature_map(
        self, images: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        del scale
        self.calls += 1
        return images.new_ones((len(images), self.embed_dim, 16, 20))


class _TestBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.online_encoder = _TestFeatureEncoder()
        self.scale_embedding = _TestScaleEmbedding()
        self.condition_on_scale = False


class _TestDetectionHead(torch.nn.Module):
    def __init__(self, **_: object) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []
        self.selected_targets: torch.Tensor | None = None

    def forward(
        self,
        features: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        batch_size = len(features[0])
        self.batch_sizes.append(batch_size)
        self.selected_targets = targets
        decoded = features[0].new_zeros((batch_size, 1, 7))
        losses = None if targets is None else {"loss": features[0].mean()}
        return decoded, losses


def test_stateful_forward_runs_head_only_for_labeled_rows() -> None:
    model = WindowJEPAYOLOX(
        _TestBackbone(), _TestDetectionHead, freeze_backbone=True
    )
    images = torch.zeros(3, 2, 240, 304)
    duration = torch.full((3,), 50.0)
    targets = torch.arange(3 * 2 * 5, dtype=torch.float32).reshape(3, 2, 5)

    decoded, losses, state = model.forward_stateful(
        images,
        duration,
        targets,
        detection_mask=torch.tensor([False, True, False]),
    )
    assert decoded is not None and decoded.shape[0] == 1
    assert losses is not None
    assert state is None
    assert model.head.batch_sizes == [1]
    torch.testing.assert_close(model.head.selected_targets, targets[1:2])

    decoded, losses, state = model.forward_stateful(
        images,
        duration,
        targets,
        detection_mask=torch.zeros(3, dtype=torch.bool),
    )
    assert decoded is losses is state is None
    assert model.head.batch_sizes == [1]


def test_sequence_features_from_multiple_steps_run_head_once() -> None:
    model = WindowJEPAYOLOX(
        _TestBackbone(), _TestDetectionHead, freeze_backbone=True
    )
    duration = torch.full((2,), 50.0)
    first, first_state = model.forward_stateful_backbone(
        torch.zeros(2, 2, 240, 304), duration
    )
    second, second_state = model.forward_stateful_backbone(
        torch.ones(2, 2, 240, 304), duration, state=first_state
    )
    assert first_state is second_state is None

    selected = torch.cat((first[0:1], second[1:2]), dim=0)
    decoded, losses = model.forward_detection_features(
        selected, torch.zeros(2, 1, 5)
    )
    assert decoded.shape[0] == 2
    assert losses is not None
    assert model.head.batch_sizes == [2]


def test_stateful_sequence_helper_loops_backbone_and_batches_head_once() -> None:
    empty_gt = np.zeros(0, dtype=np.dtype([("t", "<i8")]))

    def sample(
        step: int, lane: int, *, labeled: bool, target_value: float
    ) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, int, bool, bool, int, int]:
        targets = (
            torch.tensor([[target_value, 0.0, 0.0, 1.0, 1.0]])
            if labeled
            else torch.zeros(0, 5)
        )
        return (
            torch.full((2, 240, 304), float(step * 2 + lane)),
            targets,
            empty_gt,
            (step + 1) * 50_000,
            labeled,
            step == 0,
            lane,
            lane,
        )

    sequence = _collate_stream_sequence_detection(
        (
            sample(0, 0, labeled=False, target_value=0.0),
            sample(0, 1, labeled=True, target_value=10.0),
            sample(1, 0, labeled=True, target_value=20.0),
            sample(1, 1, labeled=True, target_value=30.0),
        )
    )
    model = WindowJEPAYOLOX(
        _TestBackbone(), _TestDetectionHead, freeze_backbone=True
    )
    manager = StreamStateManager(recurrent=False, stride_us=50_000)
    output = _forward_stateful_sequence(
        model,
        sequence,
        manager,
        duration_ms=50.0,
        device=torch.device("cpu"),
        precision="fp32",
        compute_losses=True,
    )

    assert model.backbone.online_encoder.calls == 2
    assert model.head.batch_sizes == [3]
    assert output.decoded is not None and output.decoded.shape[0] == 3
    assert output.losses is not None
    assert output.labeled_frames == 3
    assert output.timestamps == (50_000, 100_000, 100_000)
    assert model.head.selected_targets is not None
    torch.testing.assert_close(
        model.head.selected_targets[:, 0, 0], torch.tensor([10.0, 20.0, 30.0])
    )
