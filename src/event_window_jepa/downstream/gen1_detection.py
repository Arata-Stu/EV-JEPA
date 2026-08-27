from __future__ import annotations

import argparse
import hashlib
import json
import random
import tempfile
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, Sequence

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm.auto import tqdm

from event_window_jepa.data.event_store import H5EventStore
from event_window_jepa.downstream.features import require_feedforward_feature_model
from event_window_jepa.downstream.gen1_roi_probe import (
    FrameReference,
    LabelSource,
    _frame_references,
    _read_label_sources,
    _representation,
)
from event_window_jepa.models.recurrent_vjepa21_event_vit import (
    RecurrentState,
    RecurrentVJEPA21EventVisionTransformer,
    detach_recurrent_state,
    reset_recurrent_state,
)
from event_window_jepa.train.checkpoint import config_hash, load_pretrained_model


BBOX_DTYPE = np.dtype(
    {
        "names": (
            "t",
            "x",
            "y",
            "w",
            "h",
            "class_id",
            "track_id",
            "class_confidence",
        ),
        "formats": ("<i8", "<f4", "<f4", "<f4", "<f4", "<u4", "<u4", "<f4"),
    }
)
STATEFUL_LANE_SCHEMA = "rvt-label-aware-sampling-v3"
SEQUENCE_STATEFUL_LANE_SCHEMA = "stable-lanes-sequence-v2"
LEGACY_STATEFUL_LANE_SCHEMA = "stable-lanes-v1"


def _autocast_context(device: torch.device, precision: str):  # type: ignore[no-untyped-def]
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


@dataclass(frozen=True)
class RVTComponents:
    head_type: type[nn.Module]
    postprocess: Callable[..., list[torch.Tensor | None]]
    evaluate_list: Callable[..., dict[str, float]]


@dataclass(frozen=True)
class StreamFrameReference:
    source_index: int
    start: int
    stop: int
    t_end_us: int
    has_labels: bool
    state_reset: bool
    stream_segment_id: int = 0
    sampling_mode: Literal["stream", "random"] = "stream"


@dataclass(frozen=True)
class StreamLaneIndex:
    """Dataset index annotated with the stable recurrent lane that owns it."""

    reference_index: int
    lane_id: int
    force_state_reset: bool = False


@dataclass(frozen=True)
class DirectStreamLaneIndex:
    """Worker-safe direct reference used by random-access training clips."""

    reference: StreamFrameReference
    lane_id: int


@dataclass(frozen=True)
class _StreamLaneState:
    source_index: int
    t_end_us: int
    state: RecurrentState | None


@dataclass(frozen=True)
class StatefulSequenceForward:
    decoded: torch.Tensor | None
    losses: dict[str, torch.Tensor] | None
    ground_truth: tuple[np.ndarray, ...]
    timestamps: tuple[int, ...]
    labeled_frames: int
    input_frames: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a Gen1 YOLOX detector on full-frame Window-JEPA features and "
            "evaluate with the official Prophesee COCO protocol."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--val-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--backbone-init", choices=("pretrained", "random"), default="pretrained"
    )
    parser.add_argument("--window-ms", type=float, default=40.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=21,
        help=(
            "Number of consecutive stream windows unrolled inside one optimizer "
            "step in --stateful mode (RVT-style recurrent chunk execution). The "
            "current frozen-backbone detector carries state but does not backpropagate "
            "into the backbone."
        ),
    )
    parser.add_argument(
        "--stateful-sampling",
        choices=("mixed", "stream", "stream_reset", "random"),
        default="mixed",
        help=(
            "Training sampling protocol for --stateful. mixed is the RVT default: "
            "half stable stream lanes and half independent random label-anchored "
            "clips. stream carries state between label-guaranteed chunks; "
            "stream_reset uses the same chunks but resets every chunk as a "
            "state-carry ablation; random resets every clip. Validation always "
            "uses the complete causal stream. This option does not imply backbone "
            "BPTT while the detector backbone is frozen."
        ),
    )
    parser.add_argument(
        "--stream-batch-size",
        type=int,
        default=0,
        help=(
            "Number of stream lanes in a mixed stateful training batch. Zero uses "
            "the required equal 1:1 split. An explicit value is accepted only when "
            "it equals half of an even --batch-size."
        ),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--max-train-frames", type=int, default=0)
    parser.add_argument("--max-val-frames", type=int, default=0)
    parser.add_argument("--confidence-threshold", type=float, default=0.001)
    parser.add_argument("--nms-threshold", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument(
        "--unfreeze-backbone",
        action="store_true",
        help="Fine-tune the encoder; the default is a frozen-backbone detector",
    )
    parser.add_argument(
        "--stateful",
        action="store_true",
        help=(
            "Process recordings in parallel causal lanes, carry lane-local recurrent "
            "state, and update it on unlabeled windows. Training uses the selected "
            "RVT-style stream/random sampling protocol; validation always traverses "
            "the complete stream. Batch size counts concurrent clips. Requires a frozen "
            "backbone; use this for fair FF/ConvGRU/ConvLSTM comparison. Window "
            "duration must match recurrent.window_ms in a sequence-loader checkpoint."
        ),
    )
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def _load_rvt_components() -> RVTComponents:
    try:
        from event_window_jepa.third_party.rvt_detection import (
            YOLOXHead,
            evaluate_list,
            postprocess,
        )
    except ImportError as error:
        raise ImportError(
            "Gen1 detection requires torchvision and pycocotools; install "
            "event-window-jepa[detection,hdf5]"
        ) from error
    return RVTComponents(
        head_type=YOLOXHead,
        postprocess=postprocess,
        evaluate_list=evaluate_list,
    )


def _scaled_full_boxes(
    labels: np.ndarray, source: LabelSource
) -> tuple[np.ndarray, np.ndarray]:
    scale_x = source.event_width / source.bbox_width
    scale_y = source.event_height / source.bbox_height
    x1 = np.asarray(labels["x"], dtype=np.float32) * scale_x
    y1 = np.asarray(labels["y"], dtype=np.float32) * scale_y
    x2 = x1 + np.asarray(labels["w"], dtype=np.float32) * scale_x
    y2 = y1 + np.asarray(labels["h"], dtype=np.float32) * scale_y
    x1 = np.clip(x1, 0, source.event_width)
    y1 = np.clip(y1, 0, source.event_height)
    x2 = np.clip(x2, 0, source.event_width)
    y2 = np.clip(y2, 0, source.event_height)
    keep = (x2 > x1) & (y2 > y1)
    boxes = np.stack((x1[keep], y1[keep], x2[keep], y2[keep]), axis=1)
    classes = np.asarray(labels[source.class_field], dtype=np.int64)[keep]
    return boxes.astype(np.float32, copy=False), classes


def _ground_truth_array(
    boxes: np.ndarray, classes: np.ndarray, timestamp_us: int
) -> np.ndarray:
    result = np.zeros(len(boxes), dtype=BBOX_DTYPE)
    result["t"] = timestamp_us
    result["x"] = boxes[:, 0]
    result["y"] = boxes[:, 1]
    result["w"] = boxes[:, 2] - boxes[:, 0]
    result["h"] = boxes[:, 3] - boxes[:, 1]
    result["class_id"] = classes
    result["class_confidence"] = 1.0
    return result


class Gen1DetectionDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, np.ndarray, int]]
):
    def __init__(
        self,
        manifest: Path,
        sources: Sequence[LabelSource],
        references: Sequence[FrameReference],
        *,
        duration_ms: float,
        representation: Any,
    ) -> None:
        self.store = H5EventStore(manifest)
        self.sources = tuple(sources)
        self.references = tuple(references)
        self.duration_us = round(duration_ms * 1_000)
        self.representation = representation
        self._label_arrays: OrderedDict[Path, np.ndarray] = OrderedDict()

    def __len__(self) -> int:
        return len(self.references)

    def _labels(self, path: Path) -> np.ndarray:
        if path in self._label_arrays:
            labels = self._label_arrays.pop(path)
            self._label_arrays[path] = labels
            return labels
        labels = np.load(path, mmap_mode="r", allow_pickle=False)
        self._label_arrays[path] = labels
        while len(self._label_arrays) > 8:
            self._label_arrays.popitem(last=False)
        return labels

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, int]:
        reference = self.references[index]
        source = self.sources[reference.source_index]
        window = self.store.slice(source.sequence_id, reference.t_end_us, self.duration_us)
        image = torch.from_numpy(self.representation(window))
        labels = self._labels(source.path)[reference.start : reference.stop]
        boxes, classes = _scaled_full_boxes(labels, source)
        targets = np.zeros((len(boxes), 5), dtype=np.float32)
        targets[:, 0] = classes
        targets[:, 1] = (boxes[:, 0] + boxes[:, 2]) * 0.5
        targets[:, 2] = (boxes[:, 1] + boxes[:, 3]) * 0.5
        targets[:, 3] = boxes[:, 2] - boxes[:, 0]
        targets[:, 4] = boxes[:, 3] - boxes[:, 1]
        ground_truth = _ground_truth_array(boxes, classes, reference.t_end_us)
        return image, torch.from_numpy(targets), ground_truth, reference.t_end_us


def _limited_labeled_references_by_source(
    sources: Sequence[LabelSource],
    labeled_references: Sequence[FrameReference],
    *,
    maximum_labeled_frames: int,
    stream_lanes: int,
) -> dict[int, list[FrameReference]]:
    """Keep a causal, lane-balanced prefix for bounded stateful smoke runs."""

    by_source: dict[int, list[FrameReference]] = {
        index: [] for index in range(len(sources))
    }
    for reference in labeled_references:
        by_source[reference.source_index].append(reference)
    for references in by_source.values():
        references.sort(key=lambda value: value.t_end_us)
    if maximum_labeled_frames <= 0:
        return by_source

    active_sources = [
        source_index for source_index, references in by_source.items() if references
    ]
    selected_sources = active_sources[
        : min(stream_lanes, maximum_labeled_frames, len(active_sources))
    ]
    if not selected_sources:
        return {source_index: [] for source_index in by_source}
    limits = {source_index: 0 for source_index in selected_sources}
    remaining = maximum_labeled_frames
    while remaining:
        changed = False
        for source_index in selected_sources:
            if limits[source_index] >= len(by_source[source_index]):
                continue
            limits[source_index] += 1
            remaining -= 1
            changed = True
            if not remaining:
                break
        if not changed:
            break
    return {
        source_index: references[: limits.get(source_index, 0)]
        for source_index, references in by_source.items()
    }


def _cadence_indices(
    source: LabelSource,
    references: Sequence[FrameReference],
    *,
    duration_us: int,
) -> tuple[int, dict[int, FrameReference]]:
    """Map labels to the recording-local representation cadence."""

    if not references:
        raise ValueError("cannot infer an event cadence without labels")
    ordered = sorted(references, key=lambda value: value.t_end_us)
    phase = (ordered[0].t_end_us - source.t_start_us) % duration_us
    signed_phase = phase if phase <= duration_us // 2 else phase - duration_us
    first_end = source.t_start_us + duration_us + signed_phase
    while first_end - duration_us < source.t_start_us:
        first_end += duration_us
    indexed: dict[int, FrameReference] = {}
    for reference in ordered:
        delta = reference.t_end_us - first_end
        if delta < 0 or delta % duration_us:
            raise ValueError(
                f"stateful label timestamp {reference.t_end_us} for "
                f"{source.sequence_id} is not aligned to the {duration_us} us "
                "window cadence; use the checkpoint and annotation cadence"
            )
        representation_index = delta // duration_us
        if representation_index in indexed:
            raise ValueError("stateful labels contain a duplicate representation index")
        indexed[representation_index] = reference
    return first_end, indexed


def _stream_references(
    sources: Sequence[LabelSource],
    labeled_references: Sequence[FrameReference],
    *,
    duration_us: int,
    maximum_labeled_frames: int,
    stream_lanes: int = 1,
) -> tuple[StreamFrameReference, ...]:
    """Expand labeled timestamps into an ordered causal stream.

    Every gap larger than one window receives label-free state-update steps.
    Label timestamps must share the fixed window cadence within each recording.
    Gen1 annotation clocks can have a recording-local phase offset from the event
    recording origin, so that phase is inferred from the first label and
    preserved. Frame limits keep an ordered prefix in up to ``stream_lanes``
    recordings instead of random frame sampling because random subsampling would
    invalidate recurrent state.
    """

    if duration_us <= 0:
        raise ValueError("stateful window duration must be at least one microsecond")
    if stream_lanes <= 0:
        raise ValueError("stateful stream_lanes must be positive")

    by_source = _limited_labeled_references_by_source(
        sources,
        labeled_references,
        maximum_labeled_frames=maximum_labeled_frames,
        stream_lanes=stream_lanes,
    )
    output: list[StreamFrameReference] = []
    for source_index, source in enumerate(sources):
        references = by_source[source_index]
        if not references:
            continue
        next_end, _ = _cadence_indices(
            source, references, duration_us=duration_us
        )
        first = True
        for reference in references:
            while next_end < reference.t_end_us:
                output.append(
                    StreamFrameReference(
                        source_index=source_index,
                        start=0,
                        stop=0,
                        t_end_us=next_end,
                        has_labels=False,
                        state_reset=first,
                    )
                )
                first = False
                next_end += duration_us
            if next_end != reference.t_end_us:
                raise ValueError(
                    f"stateful label timestamp {reference.t_end_us} for "
                    f"{source.sequence_id} is not aligned to the {duration_us} us "
                    "window cadence; use the checkpoint and annotation cadence"
                )
            output.append(
                StreamFrameReference(
                    source_index=source_index,
                    start=reference.start,
                    stop=reference.stop,
                    t_end_us=reference.t_end_us,
                    has_labels=True,
                    state_reset=first,
                )
            )
            first = False
            next_end += duration_us
    return tuple(output)


def _training_stream_references(
    sources: Sequence[LabelSource],
    labeled_references: Sequence[FrameReference],
    *,
    duration_us: int,
    sequence_length: int,
    maximum_labeled_frames: int,
    stream_lanes: int,
) -> tuple[StreamFrameReference, ...]:
    """Build RVT training streams whose every chunk contains a label.

    Label-index gaps larger than ``sequence_length`` start a new independent
    segment. Each segment begins at most ``T - 1`` representations before its
    first label and stops immediately after its final label. Consequently, every
    consecutive T-sized chunk (including the final partial chunk) has at least
    one supervised timestep, exactly as RVT's training stream construction.
    """

    if duration_us <= 0 or sequence_length <= 0 or stream_lanes <= 0:
        raise ValueError("training stream cadence, length, and lanes must be positive")
    by_source = _limited_labeled_references_by_source(
        sources,
        labeled_references,
        maximum_labeled_frames=maximum_labeled_frames,
        stream_lanes=stream_lanes,
    )
    output: list[StreamFrameReference] = []
    segment_id = 0
    for source_index, source in enumerate(sources):
        references = by_source[source_index]
        if not references:
            continue
        first_end, indexed = _cadence_indices(
            source, references, duration_us=duration_us
        )
        label_indices = sorted(indexed)
        group_start = 0
        split_stops = [
            index
            for index in range(len(label_indices) - 1)
            if label_indices[index + 1] - label_indices[index] > sequence_length
        ]
        for group_stop in (*split_stops, len(label_indices) - 1):
            group = label_indices[group_start : group_stop + 1]
            start_index = max(group[0] - sequence_length + 1, 0)
            stop_index = group[-1] + 1
            for representation_index in range(start_index, stop_index):
                labeled = indexed.get(representation_index)
                output.append(
                    StreamFrameReference(
                        source_index=source_index,
                        start=0 if labeled is None else labeled.start,
                        stop=0 if labeled is None else labeled.stop,
                        t_end_us=first_end + representation_index * duration_us,
                        has_labels=labeled is not None,
                        state_reset=representation_index == start_index,
                        stream_segment_id=segment_id,
                        sampling_mode="stream",
                    )
                )
            segment_id += 1
            group_start = group_stop + 1
    return tuple(output)


class Gen1StreamDetectionDataset(Gen1DetectionDataset):
    references: tuple[StreamFrameReference, ...]

    def __init__(
        self,
        manifest: Path,
        sources: Sequence[LabelSource],
        references: Sequence[StreamFrameReference],
        *,
        duration_ms: float,
        representation: Any,
    ) -> None:
        super().__init__(
            manifest,
            sources,
            references,  # type: ignore[arg-type]
            duration_ms=duration_ms,
            representation=representation,
        )
        self.references = tuple(references)

    def __getitem__(
        self, index: int | StreamLaneIndex | DirectStreamLaneIndex
    ) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, int, bool, bool, int, int]:
        if isinstance(index, DirectStreamLaneIndex):
            reference = index.reference
            lane_id = index.lane_id
            image, targets, ground_truth, timestamp = self._load_reference(reference)
            return (
                image,
                targets,
                ground_truth,
                timestamp,
                reference.has_labels,
                reference.state_reset,
                reference.source_index,
                lane_id,
            )
        if isinstance(index, StreamLaneIndex):
            reference_index = index.reference_index
            lane_id = index.lane_id
        elif isinstance(index, int):
            # Direct indexing remains useful for inspection and preserves the old
            # single-stream behavior. Stateful DataLoaders always provide an
            # explicit StreamLaneIndex through StreamLaneBatchSampler.
            reference_index = index
            lane_id = 0
        else:
            raise TypeError("stream dataset index must be int or StreamLaneIndex")
        reference = self.references[reference_index]
        image, targets, ground_truth, timestamp = super().__getitem__(reference_index)
        return (
            image,
            targets,
            ground_truth,
            timestamp,
            reference.has_labels,
            reference.state_reset or (
                isinstance(index, StreamLaneIndex) and index.force_state_reset
            ),
            reference.source_index,
            lane_id,
        )

    def _load_reference(
        self, reference: StreamFrameReference
    ) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, int]:
        """Load a direct reference without materializing every random clip."""

        source = self.sources[reference.source_index]
        window = self.store.slice(source.sequence_id, reference.t_end_us, self.duration_us)
        image = torch.from_numpy(self.representation(window))
        labels = self._labels(source.path)[reference.start : reference.stop]
        boxes, classes = _scaled_full_boxes(labels, source)
        targets = np.zeros((len(boxes), 5), dtype=np.float32)
        targets[:, 0] = classes
        targets[:, 1] = (boxes[:, 0] + boxes[:, 2]) * 0.5
        targets[:, 2] = (boxes[:, 1] + boxes[:, 3]) * 0.5
        targets[:, 3] = boxes[:, 2] - boxes[:, 0]
        targets[:, 4] = boxes[:, 3] - boxes[:, 1]
        ground_truth = _ground_truth_array(boxes, classes, reference.t_end_us)
        return image, torch.from_numpy(targets), ground_truth, reference.t_end_us


class StreamLaneBatchSampler(Sampler[list[StreamLaneIndex]]):
    """Yield causal ``B x T`` chunks and refill lanes only at chunk boundaries."""

    def __init__(
        self,
        references: Sequence[StreamFrameReference],
        batch_size: int,
        *,
        sequence_length: int = 21,
        lane_offset: int = 0,
        reset_every_chunk: bool = False,
        shuffle: bool = False,
        world_size: int = 1,
        rank: int = 0,
        seed: int = 0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("stream lane batch size must be positive")
        if sequence_length <= 0:
            raise ValueError("stream sequence length must be positive")
        if lane_offset < 0:
            raise ValueError("stream lane offset cannot be negative")
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("stream sampler rank must lie inside a positive world size")
        self.batch_size = int(batch_size)
        self.sequence_length = int(sequence_length)
        self.lane_offset = int(lane_offset)
        self.reset_every_chunk = bool(reset_every_chunk)
        groups: OrderedDict[tuple[int, int], list[int]] = OrderedDict()
        for index, reference in enumerate(references):
            groups.setdefault(
                (reference.source_index, reference.stream_segment_id), []
            ).append(index)
        all_groups = tuple(tuple(indices) for indices in groups.values())
        self.groups = all_groups[rank::world_size]
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("stream sampler epoch cannot be negative")
        self.epoch = int(epoch)

    def _ordered_groups(self) -> tuple[tuple[int, ...], ...]:
        groups = list(self.groups)
        if self.shuffle:
            # Segment order changes across epochs, while the references inside
            # each segment remain untouched and causal. The explicit epoch also
            # makes a resumed run reproduce the same ordering at epoch boundaries.
            rng = random.Random(self.seed + self.epoch * 1_000_003)
            rng.shuffle(groups)
        else:
            # Longest-processing-time-first refill is the same scheduling heuristic
            # used by RVT's evaluation streams to reduce under-filled tail batches.
            groups.sort(key=lambda indices: (-len(indices), indices[0]))
        return tuple(groups)

    def __iter__(self):  # type: ignore[no-untyped-def]
        pending = iter(self._ordered_groups())
        active: list[tuple[int, tuple[int, ...], int]] = []
        for local_lane_id in range(self.batch_size):
            try:
                active.append(
                    (self.lane_offset + local_lane_id, next(pending), 0)
                )
            except StopIteration:
                break
        while active:
            lane_chunks = [
                (
                    lane_id,
                    indices[
                        position : min(position + self.sequence_length, len(indices))
                    ],
                )
                for lane_id, indices, position in active
            ]
            # Time-major ordering lets collate build one variable-B batch per
            # recurrent step without padding images or inventing fake windows.
            maximum_steps = max(len(chunk) for _, chunk in lane_chunks)
            yield [
                StreamLaneIndex(
                    reference_index=chunk[step],
                    lane_id=lane_id,
                    force_state_reset=self.reset_every_chunk and step == 0,
                )
                for step in range(maximum_steps)
                for lane_id, chunk in lane_chunks
                if step < len(chunk)
            ]
            next_active: list[tuple[int, tuple[int, ...], int]] = []
            for lane_id, indices, position in active:
                next_position = min(position + self.sequence_length, len(indices))
                if next_position < len(indices):
                    next_active.append((lane_id, indices, next_position))
                    continue
                try:
                    next_active.append((lane_id, next(pending), 0))
                except StopIteration:
                    pass
            active = next_active

    def __len__(self) -> int:
        # The iterator refills whichever lane finishes first. Computing the same
        # list-scheduling makespan by recording lengths avoids walking every frame
        # merely to size the chunk progress bar.
        groups = self._ordered_groups()
        chunk_counts = [
            (len(group) + self.sequence_length - 1) // self.sequence_length
            for group in groups
        ]
        lane_loads = chunk_counts[: self.batch_size]
        for chunk_count in chunk_counts[self.batch_size :]:
            lane_id = min(
                range(len(lane_loads)),
                key=lambda index: (lane_loads[index], index),
            )
            lane_loads[lane_id] += chunk_count
        return max(lane_loads, default=0)

    def iter_cycled(self) -> Iterator[list[StreamLaneIndex]]:
        """Repeat lane-local segment shards without losing a partial tail.

        Mixed training needs a fixed number of stream clips on every optimizer
        step while the random and stream datasets can have different lengths.
        Groups are assigned to stable lanes with the same least-loaded heuristic
        as ``__len__``. Each lane then cycles only its own causal chunks; wraparound
        lands on a segment start and therefore resets state.
        """

        groups = self._ordered_groups()
        if len(groups) < self.batch_size:
            raise ValueError("cyclic stream sampling needs one segment per lane")
        lane_chunks: list[list[tuple[int, ...]]] = [
            [] for _ in range(self.batch_size)
        ]
        lane_loads = [0 for _ in range(self.batch_size)]
        for group_index, group in enumerate(groups):
            lane = (
                group_index
                if group_index < self.batch_size
                else min(
                    range(self.batch_size),
                    key=lambda index: (lane_loads[index], index),
                )
            )
            chunks = [
                group[start : start + self.sequence_length]
                for start in range(0, len(group), self.sequence_length)
            ]
            lane_chunks[lane].extend(chunks)
            lane_loads[lane] += len(chunks)
        batch_index = 0
        while True:
            selected = [
                chunks[batch_index % len(chunks)] for chunks in lane_chunks
            ]
            maximum_steps = max(len(chunk) for chunk in selected)
            yield [
                StreamLaneIndex(
                    reference_index=chunk[step],
                    lane_id=self.lane_offset + local_lane_id,
                    force_state_reset=self.reset_every_chunk and step == 0,
                )
                for step in range(maximum_steps)
                for local_lane_id, chunk in enumerate(selected)
                if step < len(chunk)
            ]
            batch_index += 1


class RandomLabelClipBatchSampler(Sampler[list[DirectStreamLaneIndex]]):
    """Yield independent complete T-clips ending at shuffled label timestamps.

    Earlier labeled timesteps inside a clip remain supervised, matching RVT's
    Gen1 ``only_load_end_labels=False`` setting. Anchors without T complete event
    windows of history are excluded. Every clip resets its lane at step zero.
    """

    def __init__(
        self,
        sources: Sequence[LabelSource],
        labeled_references: Sequence[FrameReference],
        *,
        duration_us: int,
        sequence_length: int,
        batch_size: int,
        lane_offset: int = 0,
        maximum_labeled_frames: int = 0,
        drop_last: bool = False,
        world_size: int = 1,
        rank: int = 0,
        seed: int = 0,
    ) -> None:
        if min(duration_us, sequence_length, batch_size) <= 0:
            raise ValueError("random clip cadence, length, and batch must be positive")
        if lane_offset < 0 or maximum_labeled_frames < 0:
            raise ValueError("random lane offset and frame limit cannot be negative")
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("random sampler rank must lie inside a positive world size")
        self.sources = tuple(sources)
        self.duration_us = int(duration_us)
        self.sequence_length = int(sequence_length)
        self.batch_size = int(batch_size)
        self.lane_offset = int(lane_offset)
        self.drop_last = bool(drop_last)
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0

        by_source: dict[int, list[FrameReference]] = {
            index: [] for index in range(len(sources))
        }
        for reference in labeled_references:
            by_source[reference.source_index].append(reference)
        cadence: dict[int, tuple[int, dict[int, FrameReference]]] = {}
        anchors: list[tuple[int, int]] = []
        for source_index, source in enumerate(sources):
            references = by_source[source_index]
            if not references:
                continue
            first_end, indexed = _cadence_indices(
                source, references, duration_us=self.duration_us
            )
            cadence[source_index] = (first_end, indexed)
            anchors.extend(
                (source_index, representation_index)
                for representation_index in sorted(indexed)
                if representation_index >= self.sequence_length - 1
            )
        if maximum_labeled_frames and len(anchors) > maximum_labeled_frames:
            rng = random.Random(self.seed + 7_919)
            selected = sorted(
                rng.sample(range(len(anchors)), maximum_labeled_frames)
            )
            anchors = [anchors[index] for index in selected]
        anchors = anchors[self.rank :: self.world_size]
        if not anchors:
            raise ValueError(
                "no labeled frame has enough history for a complete random clip"
            )
        if self.drop_last and len(anchors) < self.batch_size:
            raise ValueError(
                "random sampling has fewer valid anchors than its batch share"
            )
        self._cadence = cadence
        self.anchors = tuple(anchors)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("random sampler epoch cannot be negative")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.anchors) // self.batch_size
        return (len(self.anchors) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[list[DirectStreamLaneIndex]]:
        order = list(range(len(self.anchors)))
        rng = random.Random(self.seed + self.epoch * 1_000_003)
        rng.shuffle(order)
        stop = (
            len(order) - len(order) % self.batch_size
            if self.drop_last
            else len(order)
        )
        for batch_start in range(0, stop, self.batch_size):
            selected = order[batch_start : batch_start + self.batch_size]
            if not selected:
                continue
            clips: list[tuple[int, tuple[StreamFrameReference, ...]]] = []
            for local_lane_id, anchor_offset in enumerate(selected):
                source_index, anchor_index = self.anchors[anchor_offset]
                first_end, indexed = self._cadence[source_index]
                first_index = anchor_index - self.sequence_length + 1
                references: list[StreamFrameReference] = []
                for step in range(self.sequence_length):
                    representation_index = first_index + step
                    labeled = indexed.get(representation_index)
                    references.append(
                        StreamFrameReference(
                            source_index=source_index,
                            start=0 if labeled is None else labeled.start,
                            stop=0 if labeled is None else labeled.stop,
                            t_end_us=(
                                first_end
                                + representation_index * self.duration_us
                            ),
                            has_labels=labeled is not None,
                            state_reset=step == 0,
                            stream_segment_id=-1,
                            sampling_mode="random",
                        )
                    )
                clips.append(
                    (
                        self.lane_offset + local_lane_id,
                        tuple(references),
                    )
                )
            yield [
                DirectStreamLaneIndex(reference=clip[step], lane_id=lane_id)
                for step in range(self.sequence_length)
                for lane_id, clip in clips
            ]


StatefulTrainingIndex = StreamLaneIndex | DirectStreamLaneIndex


class MixedStatefulBatchSampler(Sampler[list[StatefulTrainingIndex]]):
    """Cycle two samplers into fixed RVT stream/random batch shares."""

    def __init__(
        self,
        stream_sampler: StreamLaneBatchSampler,
        random_sampler: RandomLabelClipBatchSampler,
    ) -> None:
        if stream_sampler.batch_size <= 0 or random_sampler.batch_size <= 0:
            raise ValueError("mixed sampling requires both stream and random clips")
        self.stream_sampler = stream_sampler
        self.random_sampler = random_sampler
        self.stream_batch_size = stream_sampler.batch_size
        self.random_batch_size = random_sampler.batch_size
        self.epoch = 0
        if len(stream_sampler.groups) < self.stream_batch_size:
            raise ValueError(
                "mixed sampling needs at least one training segment per stream lane"
            )

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("mixed sampler epoch cannot be negative")
        self.epoch = int(epoch)
        self.stream_sampler.set_epoch(epoch)
        self.random_sampler.set_epoch(epoch)

    def __len__(self) -> int:
        return max(len(self.stream_sampler), len(self.random_sampler))

    @staticmethod
    def _next_cycled(
        iterator: Iterator[list[Any]], factory: Callable[[], Iterator[list[Any]]]
    ) -> tuple[list[Any], Iterator[list[Any]]]:
        try:
            return next(iterator), iterator
        except StopIteration:
            iterator = factory()
            try:
                return next(iterator), iterator
            except StopIteration as error:
                raise RuntimeError("mixed sampling component is empty") from error

    def __iter__(self) -> Iterator[list[StatefulTrainingIndex]]:
        stream_factory = self.stream_sampler.iter_cycled
        random_factory = lambda: iter(self.random_sampler)
        stream_iterator: Iterator[list[Any]] = stream_factory()
        random_iterator: Iterator[list[Any]] = random_factory()
        for _ in range(len(self)):
            stream_batch, stream_iterator = self._next_cycled(
                stream_iterator, stream_factory
            )
            random_batch, random_iterator = self._next_cycled(
                random_iterator, random_factory
            )
            yield [*stream_batch, *random_batch]


class StreamStateManager:
    """Own recurrent state for stable lanes and detach it at chunk boundaries.

    DataLoader workers only construct CPU samples. State lives here, next to the
    model, and is keyed by the explicit lane id emitted by StreamLaneBatchSampler.
    The common fixed-lane path reuses one detached batched state directly; the
    lane cache avoids state swaps when exhausted lanes disappear from a short tail.
    """

    def __init__(self, *, recurrent: bool, stride_us: int) -> None:
        if stride_us <= 0:
            raise ValueError("stream state stride must be positive")
        self.recurrent = bool(recurrent)
        self.stride_us = int(stride_us)
        self._lanes: dict[int, _StreamLaneState] = {}
        self._batched_lane_ids: tuple[int, ...] = ()
        self._batched_state: RecurrentState | None = None

    @staticmethod
    def _validate_batch_metadata(
        lane_ids: Sequence[int],
        source_indices: Sequence[int],
        timestamps: Sequence[int],
        state_reset: torch.Tensor,
    ) -> None:
        batch_size = len(lane_ids)
        if (
            len(source_indices) != batch_size
            or len(timestamps) != batch_size
            or tuple(state_reset.shape) != (batch_size,)
            or state_reset.dtype != torch.bool
        ):
            raise ValueError("stream metadata must contain one reset/source/time per lane")
        if len(set(lane_ids)) != batch_size:
            raise ValueError("a stream batch cannot contain a lane more than once")
        if any(lane_id < 0 for lane_id in lane_ids):
            raise ValueError("stream lane ids must be non-negative")

    @staticmethod
    def _zero_like(state: RecurrentState) -> RecurrentState:
        if isinstance(state, torch.Tensor):
            return torch.zeros_like(state)
        return torch.zeros_like(state[0]), torch.zeros_like(state[1])

    @staticmethod
    def _concatenate(states: Sequence[RecurrentState]) -> RecurrentState:
        first = states[0]
        if isinstance(first, torch.Tensor):
            if not all(isinstance(state, torch.Tensor) for state in states):
                raise TypeError("stream lanes contain incompatible recurrent state types")
            return torch.cat(list(states), dim=0)  # type: ignore[arg-type]
        if not all(isinstance(state, tuple) and len(state) == 2 for state in states):
            raise TypeError("stream lanes contain incompatible recurrent state types")
        return (
            torch.cat([state[0] for state in states], dim=0),  # type: ignore[index]
            torch.cat([state[1] for state in states], dim=0),  # type: ignore[index]
        )

    @staticmethod
    def _batch_size(state: RecurrentState) -> int:
        if isinstance(state, torch.Tensor):
            return int(state.shape[0])
        if state[0].shape[0] != state[1].shape[0]:
            raise ValueError("ConvLSTM hidden and cell states have different batch sizes")
        return int(state[0].shape[0])

    @staticmethod
    def _row(state: RecurrentState, row: int) -> RecurrentState:
        if isinstance(state, torch.Tensor):
            return state[row : row + 1]
        return state[0][row : row + 1], state[1][row : row + 1]

    def prepare(
        self,
        *,
        lane_ids: Sequence[int],
        source_indices: Sequence[int],
        timestamps: Sequence[int],
        state_reset: torch.Tensor,
    ) -> RecurrentState | None:
        """Gather lane states into current batch-row order and apply resets."""

        self._validate_batch_metadata(
            lane_ids, source_indices, timestamps, state_reset
        )
        rows: list[RecurrentState | None] = []
        for row, (lane_id, source_index, timestamp) in enumerate(
            zip(lane_ids, source_indices, timestamps, strict=True)
        ):
            reset = bool(state_reset[row])
            previous = self._lanes.get(lane_id)
            if reset:
                rows.append(None)
                continue
            if previous is None:
                raise ValueError(
                    f"stream lane {lane_id} continued without an initialized state"
                )
            if previous.source_index != source_index:
                raise ValueError(
                    f"stream lane {lane_id} changed recording without state reset"
                )
            expected_timestamp = previous.t_end_us + self.stride_us
            if timestamp != expected_timestamp:
                raise ValueError(
                    f"stream lane {lane_id} is not causal at {timestamp}; "
                    f"expected {expected_timestamp}"
                )
            if self.recurrent and previous.state is None:
                raise RuntimeError(f"stream lane {lane_id} has no recurrent state")
            rows.append(previous.state)

        if not self.recurrent or all(state is None for state in rows):
            return None
        if tuple(lane_ids) == self._batched_lane_ids:
            if self._batched_state is None:
                raise RuntimeError("stream batch has no cached recurrent state")
            if not bool(state_reset.any()):
                return self._batched_state
            reset_mask = state_reset.to(
                device=(
                    self._batched_state.device
                    if isinstance(self._batched_state, torch.Tensor)
                    else self._batched_state[0].device
                )
            )
            return reset_recurrent_state(self._batched_state, reset_mask)
        prototype = next(state for state in rows if state is not None)
        gathered = [
            self._zero_like(prototype) if state is None else state for state in rows
        ]
        return self._concatenate(gathered)

    def update(
        self,
        *,
        lane_ids: Sequence[int],
        source_indices: Sequence[int],
        timestamps: Sequence[int],
        state: RecurrentState | None,
        detach: bool = True,
    ) -> None:
        """Split model state into lane rows, optionally detaching the graph."""

        batch_size = len(lane_ids)
        if len(source_indices) != batch_size or len(timestamps) != batch_size:
            raise ValueError("stream state update metadata has inconsistent lengths")
        if self.recurrent:
            if state is None:
                raise RuntimeError("recurrent stream forward returned no state")
            if self._batch_size(state) != batch_size:
                raise ValueError("recurrent stream state has the wrong batch size")
        elif state is not None:
            raise ValueError("feedforward stream unexpectedly returned recurrent state")

        stored_state = detach_recurrent_state(state) if detach else state
        active_lanes = set(lane_ids)
        self._lanes = {
            lane_id: lane_state
            for lane_id, lane_state in self._lanes.items()
            if lane_id in active_lanes
        }
        for row, (lane_id, source_index, timestamp) in enumerate(
            zip(lane_ids, source_indices, timestamps, strict=True)
        ):
            lane_state = None
            if stored_state is not None:
                lane_state = self._row(stored_state, row)
            self._lanes[lane_id] = _StreamLaneState(
                source_index=source_index,
                t_end_us=timestamp,
                state=lane_state,
            )
        self._batched_lane_ids = tuple(lane_ids)
        self._batched_state = stored_state

    def detach(self) -> None:
        """Cut recurrent history after a complete sequence chunk."""

        self._batched_state = detach_recurrent_state(self._batched_state)
        self._lanes = {
            lane_id: _StreamLaneState(
                source_index=lane_state.source_index,
                t_end_us=lane_state.t_end_us,
                state=detach_recurrent_state(lane_state.state),
            )
            for lane_id, lane_state in self._lanes.items()
        }


def _collate_detection(
    batch: Sequence[tuple[torch.Tensor, torch.Tensor, np.ndarray, int]],
) -> tuple[torch.Tensor, torch.Tensor, tuple[np.ndarray, ...], tuple[int, ...]]:
    images, labels, ground_truth, timestamps = zip(*batch, strict=True)
    maximum = max(len(value) for value in labels)
    padded = torch.zeros((len(labels), maximum, 5), dtype=torch.float32)
    for index, value in enumerate(labels):
        padded[index, : len(value)] = value
    return torch.stack(images), padded, tuple(ground_truth), tuple(timestamps)


def _collate_stream_detection(
    batch: Sequence[
        tuple[torch.Tensor, torch.Tensor, np.ndarray, int, bool, bool, int, int]
    ],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    tuple[np.ndarray, ...],
    tuple[int, ...],
    torch.Tensor,
    torch.Tensor,
    tuple[int, ...],
    tuple[int, ...],
]:
    (
        images,
        labels,
        ground_truth,
        timestamps,
        has_labels,
        state_reset,
        source_ids,
        lane_ids,
    ) = zip(*batch, strict=True)
    maximum = max(len(value) for value in labels)
    padded = torch.zeros((len(labels), maximum, 5), dtype=torch.float32)
    for index, value in enumerate(labels):
        padded[index, : len(value)] = value
    return (
        torch.stack(images),
        padded,
        tuple(ground_truth),
        tuple(timestamps),
        torch.tensor(has_labels, dtype=torch.bool),
        torch.tensor(state_reset, dtype=torch.bool),
        tuple(source_ids),
        tuple(lane_ids),
    )


def _collate_stream_sequence_detection(
    batch: Sequence[
        tuple[torch.Tensor, torch.Tensor, np.ndarray, int, bool, bool, int, int]
    ],
) -> tuple[
    tuple[
        torch.Tensor,
        torch.Tensor,
        tuple[np.ndarray, ...],
        tuple[int, ...],
        torch.Tensor,
        torch.Tensor,
        tuple[int, ...],
        tuple[int, ...],
    ],
    ...,
]:
    """Transpose flattened lane chunks into a tuple of variable-B time steps."""

    if not batch:
        raise ValueError("stream sequence batch cannot be empty")
    steps: list[list[Any]] = []
    next_step_by_lane: dict[int, int] = {}
    for sample in batch:
        lane_id = int(sample[-1])
        step = next_step_by_lane.get(lane_id, 0)
        next_step_by_lane[lane_id] = step + 1
        if step == len(steps):
            steps.append([])
        elif step > len(steps):
            raise RuntimeError("stream sequence contains a lane timestep gap")
        steps[step].append(sample)
    return tuple(_collate_stream_detection(step) for step in steps)


def _concatenate_padded_targets(targets: Sequence[torch.Tensor]) -> torch.Tensor:
    """Concatenate target rows whose per-timestep box padding may differ."""

    if not targets:
        raise ValueError("cannot concatenate an empty target sequence")
    maximum = max(int(value.shape[1]) for value in targets)
    total = sum(int(value.shape[0]) for value in targets)
    output = targets[0].new_zeros((total, maximum, 5))
    offset = 0
    for value in targets:
        if value.ndim != 3 or value.shape[2] != 5:
            raise ValueError("detection targets must have shape [B,N,5]")
        count = int(value.shape[0])
        output[offset : offset + count, : value.shape[1]] = value
        offset += count
    return output


def _interpolated_position_embedding(
    encoder: Any, grid_size: tuple[int, int], dtype: torch.dtype
) -> torch.Tensor:
    source_height, source_width = encoder.grid_size
    target_height, target_width = grid_size
    position = encoder.position_embedding
    if (source_height, source_width) == grid_size:
        return position.to(dtype=dtype)
    position_map = position.reshape(1, source_height, source_width, encoder.embed_dim)
    position_map = position_map.permute(0, 3, 1, 2).float()
    position_map = functional.interpolate(
        position_map,
        size=(target_height, target_width),
        mode="bicubic",
        align_corners=False,
    )
    return position_map.permute(0, 2, 3, 1).reshape(
        1, target_height * target_width, encoder.embed_dim
    ).to(dtype=dtype)


def _dynamic_encoder_feature_map(
    model: Any, images: torch.Tensor, duration_ms: torch.Tensor
) -> torch.Tensor:
    encoder = model.online_encoder
    scale = model.scale_embedding(duration_ms.reshape(len(images)))
    if not model.condition_on_scale:
        scale = torch.zeros_like(scale)
    if hasattr(encoder, "forward_feature_map"):
        return encoder.forward_feature_map(images, scale)
    patches_2d = encoder.patch_embed(images)
    grid_size = tuple(patches_2d.shape[-2:])
    patches = patches_2d.flatten(2).transpose(1, 2)
    patches = patches + _interpolated_position_embedding(
        encoder, grid_size, patches.dtype
    )
    scale_token = encoder.scale_projection(scale).unsqueeze(1)
    tokens = encoder.norm(encoder.blocks(torch.cat((scale_token, patches), dim=1)))
    tokens = tokens[:, 1:]
    return tokens.transpose(1, 2).reshape(
        len(images), encoder.embed_dim, *grid_size
    )


def _dynamic_encoder_feature_map_stateful(
    model: Any,
    images: torch.Tensor,
    duration_ms: torch.Tensor,
    state: Any,
    *,
    detach_state: bool,
) -> tuple[torch.Tensor, Any]:
    encoder = model.online_encoder
    if not isinstance(encoder, RecurrentVJEPA21EventVisionTransformer):
        return _dynamic_encoder_feature_map(model, images, duration_ms), None
    scale = model.scale_embedding(duration_ms.reshape(len(images)))
    if not model.condition_on_scale:
        scale = torch.zeros_like(scale)
    return encoder.forward_feature_map_recurrent(
        images,
        scale,
        state=state,
        detach_state=detach_state,
    )


class WindowJEPAYOLOX(nn.Module):
    def __init__(
        self,
        backbone: Any,
        head_type: type[nn.Module],
        *,
        freeze_backbone: bool,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        feature_dim = backbone.online_encoder.embed_dim
        channels = (96, 192, 384)
        self.p3 = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, channels[0], 2, 2),
            nn.BatchNorm2d(channels[0]),
            nn.SiLU(),
        )
        self.p4 = nn.Sequential(
            nn.Conv2d(feature_dim, channels[1], 1),
            nn.BatchNorm2d(channels[1]),
            nn.SiLU(),
        )
        self.p5 = nn.Sequential(
            nn.Conv2d(feature_dim, channels[2], 3, 2, 1),
            nn.BatchNorm2d(channels[2]),
            nn.SiLU(),
        )
        self.head = head_type(
            num_classes=2,
            strides=(8, 16, 32),
            in_channels=channels,
            act="silu",
            depthwise=False,
            compile_cfg=None,
        )
        if freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    @property
    def recurrent(self) -> bool:
        return isinstance(
            self.backbone.online_encoder,
            RecurrentVJEPA21EventVisionTransformer,
        )

    def train(self, mode: bool = True) -> WindowJEPAYOLOX:
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(
        self,
        images: torch.Tensor,
        duration_ms: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        # Gen1 is 240x304. Bottom/right zero padding gives a 256x320 grid whose
        # three detection levels align exactly at strides 8, 16, and 32.
        images = functional.pad(images, (0, 16, 0, 16))
        if self.freeze_backbone:
            with torch.no_grad():
                base = _dynamic_encoder_feature_map(self.backbone, images, duration_ms)
        else:
            base = _dynamic_encoder_feature_map(self.backbone, images, duration_ms)
        return self.forward_detection_features(base, targets)

    def forward_detection_features(
        self,
        base: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        """Run the trainable pyramid/head on selected backbone feature rows."""

        features = (self.p3(base), self.p4(base), self.p5(base))
        return self.head(features, targets)

    def forward_stateful_backbone(
        self,
        images: torch.Tensor,
        duration_ms: torch.Tensor,
        *,
        state: RecurrentState | None = None,
        detach_state: bool = False,
    ) -> tuple[torch.Tensor, RecurrentState | None]:
        """Advance every stream lane once without invoking the detection head."""

        images = functional.pad(images, (0, 16, 0, 16))
        if self.freeze_backbone:
            with torch.no_grad():
                return _dynamic_encoder_feature_map_stateful(
                    self.backbone,
                    images,
                    duration_ms,
                    state,
                    detach_state=detach_state,
                )
        return _dynamic_encoder_feature_map_stateful(
            self.backbone,
            images,
            duration_ms,
            state,
            detach_state=detach_state,
        )

    def forward_stateful(
        self,
        images: torch.Tensor,
        duration_ms: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        state: RecurrentState | None = None,
        detection_mask: torch.Tensor | None = None,
        features_only: bool = False,
        detach_state: bool = True,
    ) -> tuple[
        torch.Tensor | None,
        dict[str, torch.Tensor] | None,
        RecurrentState | None,
    ]:
        base, next_state = self.forward_stateful_backbone(
            images,
            duration_ms,
            state=state,
            detach_state=detach_state,
        )
        if detection_mask is None:
            has_detection_rows = not features_only
            detection_mask = torch.full(
                (len(images),),
                has_detection_rows,
                dtype=torch.bool,
                device=base.device,
            )
        else:
            if (
                detection_mask.dtype != torch.bool
                or tuple(detection_mask.shape) != (len(images),)
            ):
                raise ValueError("detection_mask must be a boolean [B] tensor")
            has_detection_rows = bool(detection_mask.any())
            if features_only and has_detection_rows:
                raise ValueError("features_only cannot select detection rows")
            detection_mask = detection_mask.to(device=base.device)
        if not has_detection_rows:
            return None, None, next_state
        selected_base = base[detection_mask]
        selected_targets = targets
        if targets is not None:
            if targets.ndim != 3 or targets.shape[0] != len(images):
                raise ValueError("stateful targets must have shape [B,N,5]")
            selected_targets = targets[detection_mask]
        decoded, losses = self.forward_detection_features(
            selected_base, selected_targets
        )
        return decoded, losses, next_state


def _forward_stateful_sequence(
    model: WindowJEPAYOLOX,
    sequence: Sequence[Any],
    state_manager: StreamStateManager,
    *,
    duration_ms: float,
    device: torch.device,
    precision: str,
    compute_losses: bool,
) -> StatefulSequenceForward:
    """Unroll one causal chunk, then invoke the head once on all labeled rows."""

    selected_features: list[torch.Tensor] = []
    selected_targets: list[torch.Tensor] = []
    selected_ground_truth: list[np.ndarray] = []
    selected_timestamps: list[int] = []
    input_frames = 0
    for (
        images,
        targets,
        batch_ground_truth,
        timestamps,
        has_labels,
        state_reset,
        source_indices,
        lane_ids,
    ) in sequence:
        input_frames += len(images)
        state = state_manager.prepare(
            lane_ids=lane_ids,
            source_indices=source_indices,
            timestamps=timestamps,
            state_reset=state_reset,
        )
        images = images.to(device, non_blocking=True)
        duration = torch.full((len(images),), duration_ms, device=device)
        context = _autocast_context(device, precision)
        with context:
            base, next_state = model.forward_stateful_backbone(
                images, duration, state=state, detach_state=False
            )
        state_manager.update(
            lane_ids=lane_ids,
            source_indices=source_indices,
            timestamps=timestamps,
            state=next_state,
            detach=False,
        )
        selected_indices = torch.nonzero(
            has_labels, as_tuple=False
        ).flatten().tolist()
        if not selected_indices:
            continue
        detection_mask = has_labels.to(device=base.device)
        selected_features.append(base[detection_mask])
        if compute_losses:
            targets = targets.to(device, non_blocking=True)
            selected_targets.append(targets[detection_mask])
        selected_ground_truth.extend(
            batch_ground_truth[index] for index in selected_indices
        )
        selected_timestamps.extend(timestamps[index] for index in selected_indices)

    # The recurrent state persists across the next chunk, but its history does
    # not. With the currently required frozen backbone this is a state boundary,
    # not an active backbone-gradient boundary.
    state_manager.detach()
    labeled_frames = len(selected_timestamps)
    if not selected_features:
        return StatefulSequenceForward(None, None, (), (), 0, input_frames)
    head_targets = (
        _concatenate_padded_targets(selected_targets) if compute_losses else None
    )
    context = _autocast_context(device, precision)
    with context:
        decoded, losses = model.forward_detection_features(
            torch.cat(selected_features, dim=0), head_targets
        )
    return StatefulSequenceForward(
        decoded=decoded,
        losses=losses,
        ground_truth=tuple(selected_ground_truth),
        timestamps=tuple(selected_timestamps),
        labeled_frames=labeled_frames,
        input_frames=input_frames,
    )


def _stateful_sampling_counts(
    sequence: Sequence[Any],
    *,
    sampling: str,
    stream_batch_size: int,
) -> tuple[int, int, int, int]:
    """Return labeled stream/random frames and stream/random clip counts."""

    stream_labeled = 0
    random_labeled = 0
    stream_lanes: set[int] = set()
    random_lanes: set[int] = set()
    for step in sequence:
        has_labels: torch.Tensor = step[4]
        lane_ids: tuple[int, ...] = step[7]
        for labeled, lane_id in zip(has_labels.tolist(), lane_ids, strict=True):
            is_stream = sampling in {"stream", "stream_reset"} or (
                sampling == "mixed" and lane_id < stream_batch_size
            )
            if is_stream:
                stream_lanes.add(lane_id)
                stream_labeled += int(labeled)
            else:
                random_lanes.add(lane_id)
                random_labeled += int(labeled)
    return stream_labeled, random_labeled, len(stream_lanes), len(random_lanes)


def _predictions_to_prophesee(
    detections: Sequence[torch.Tensor | None],
    timestamps: Sequence[int],
    *,
    height: int,
    width: int,
) -> list[np.ndarray]:
    output: list[np.ndarray] = []
    for detection, timestamp in zip(detections, timestamps, strict=True):
        if detection is None:
            output.append(np.zeros(0, dtype=BBOX_DTYPE))
            continue
        values = detection.detach().float().cpu().numpy()
        x1 = np.clip(values[:, 0], 0, width)
        y1 = np.clip(values[:, 1], 0, height)
        x2 = np.clip(values[:, 2], 0, width)
        y2 = np.clip(values[:, 3], 0, height)
        keep = (x2 > x1) & (y2 > y1)
        result = np.zeros(int(np.count_nonzero(keep)), dtype=BBOX_DTYPE)
        result["t"] = timestamp
        result["x"] = x1[keep]
        result["y"] = y1[keep]
        result["w"] = x2[keep] - x1[keep]
        result["h"] = y2[keep] - y1[keep]
        result["class_id"] = values[keep, 6].astype(np.uint32)
        # This matches RVT's to_prophesee conversion after product-threshold NMS.
        result["class_confidence"] = values[keep, 5]
        output.append(result)
    return output


@torch.no_grad()
def _evaluate(
    model: WindowJEPAYOLOX,
    loader: DataLoader[Any],
    components: RVTComponents,
    *,
    duration_ms: float,
    confidence_threshold: float,
    nms_threshold: float,
    device: torch.device,
    precision: str,
) -> dict[str, float]:
    model.eval()
    ground_truth: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for images, _, batch_ground_truth, timestamps in tqdm(loader, desc="detection val"):
        images = images.to(device, non_blocking=True)
        duration = torch.full((len(images),), duration_ms, device=device)
        context = _autocast_context(device, precision)
        with context:
            decoded, _ = model(images, duration)
        processed = components.postprocess(
            decoded.float().clone(),
            num_classes=2,
            conf_thre=confidence_threshold,
            nms_thre=nms_threshold,
        )
        ground_truth.extend(batch_ground_truth)
        predictions.extend(
            _predictions_to_prophesee(
                processed, timestamps, height=240, width=304
            )
        )
    metrics = components.evaluate_list(
        result_boxes_list=predictions,
        gt_boxes_list=ground_truth,
        height=240,
        width=304,
        camera="gen1",
        apply_bbox_filters=True,
        downsampled_by_2=False,
        return_aps=True,
    )
    return {str(key): float(value) for key, value in metrics.items()}


@torch.no_grad()
def _evaluate_stateful(
    model: WindowJEPAYOLOX,
    loader: DataLoader[Any],
    components: RVTComponents,
    *,
    duration_ms: float,
    confidence_threshold: float,
    nms_threshold: float,
    device: torch.device,
    precision: str,
) -> dict[str, float]:
    model.eval()
    ground_truth: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    state_manager = StreamStateManager(
        recurrent=model.recurrent,
        stride_us=round(duration_ms * 1_000),
    )
    for sequence in tqdm(loader, desc="stateful detection val"):
        forward = _forward_stateful_sequence(
            model,
            sequence,
            state_manager,
            duration_ms=duration_ms,
            device=device,
            precision=precision,
            compute_losses=False,
        )
        if forward.decoded is None:
            continue
        processed = components.postprocess(
            forward.decoded.float().clone(),
            num_classes=2,
            conf_thre=confidence_threshold,
            nms_thre=nms_threshold,
        )
        ground_truth.extend(forward.ground_truth)
        predictions.extend(
            _predictions_to_prophesee(
                processed, forward.timestamps, height=240, width=304
            )
        )
    metrics = components.evaluate_list(
        result_boxes_list=predictions,
        gt_boxes_list=ground_truth,
        height=240,
        width=304,
        camera="gen1",
        apply_bbox_filters=True,
        downsampled_by_2=False,
        return_aps=True,
    )
    return {str(key): float(value) for key, value in metrics.items()}


def _save_checkpoint(
    path: Path,
    model: WindowJEPAYOLOX,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    args: argparse.Namespace,
    metrics: dict[str, float],
    best_ap: float,
    best_epoch: int,
    pretrain_config_hash: str,
    backbone_fingerprint: str,
) -> None:
    detector_state = {
        name: value
        for name, value in model.state_dict().items()
        if not name.startswith("backbone.")
        or (
            args.unfreeze_backbone
            and (
                name.startswith("backbone.online_encoder.")
                or name.startswith("backbone.scale_embedding.")
            )
        )
    }
    payload = {
        "schema": "event-window-jepa-gen1-yolox-v2",
        "model": detector_state,
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "backbone_init": args.backbone_init,
        "pretrain_checkpoint": str(args.checkpoint.resolve()),
        "pretrain_config_hash": pretrain_config_hash,
        "backbone_fingerprint": backbone_fingerprint,
        "window_ms": args.window_ms,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length if args.stateful else 1,
        "stateful_sampling": args.stateful_sampling if args.stateful else None,
        "stream_batch_size": (
            _stateful_batch_partition(args)[0] if args.stateful else 0
        ),
        "max_train_frames": args.max_train_frames,
        "max_val_frames": args.max_val_frames,
        "train_manifest_identity": (
            _manifest_identity(args.train_manifest) if args.stateful else None
        ),
        "val_manifest_identity": (
            _manifest_identity(args.val_manifest) if args.stateful else None
        ),
        "seed": args.seed,
        "precision": args.precision,
        "stateful": args.stateful,
        "stateful_lane_schema": STATEFUL_LANE_SCHEMA if args.stateful else None,
        "metrics": metrics,
        "best_ap": best_ap,
        "best_epoch": best_epoch,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".partial", delete=False
        ) as handle:
            temporary_name = handle.name
        torch.save(payload, temporary_name)
        Path(temporary_name).replace(path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _stateful_batch_partition(args: argparse.Namespace) -> tuple[int, int]:
    """Resolve (stream, random) clip counts for one stateful training batch."""

    sampling = args.stateful_sampling
    if sampling not in {"mixed", "stream", "stream_reset", "random"}:
        raise ValueError("unsupported stateful sampling protocol")
    requested_stream = int(args.stream_batch_size)
    if requested_stream < 0:
        raise ValueError("stream batch size cannot be negative")
    if sampling == "mixed":
        if args.batch_size % 2:
            raise ValueError(
                "equal mixed sampling requires an even batch size"
            )
        equal_stream = args.batch_size // 2
        if requested_stream not in {0, equal_stream}:
            raise ValueError(
                "mixed sampling requires an exact 1:1 stream/random split; "
                "omit --stream-batch-size or set it to half of batch size"
            )
        requested_stream = equal_stream
        return requested_stream, args.batch_size - requested_stream
    if sampling in {"stream", "stream_reset"}:
        if requested_stream not in {0, args.batch_size}:
            raise ValueError(
                "stream sampling uses the full batch; omit --stream-batch-size"
            )
        return args.batch_size, 0
    if requested_stream:
        raise ValueError("random sampling cannot reserve stream lanes")
    return 0, args.batch_size


def _validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.window_ms,
        args.batch_size,
        args.sequence_length,
        args.epochs,
        args.learning_rate,
        args.eval_every,
    )
    if any(value <= 0 for value in positive) or round(args.window_ms * 1_000) <= 0:
        raise ValueError(
            "window, batch, sequence length, epoch, learning-rate, and eval "
            "cadence must be positive"
        )
    if args.workers < 0 or min(args.max_train_frames, args.max_val_frames) < 0:
        raise ValueError("workers and frame limits cannot be negative")
    if not 0 < args.confidence_threshold < 1 or not 0 < args.nms_threshold < 1:
        raise ValueError("confidence and NMS thresholds must lie inside (0, 1)")
    if args.stateful and args.unfreeze_backbone:
        raise ValueError(
            "stateful detection currently requires a frozen backbone; updating "
            "recurrent weights while carrying old state would be inconsistent"
        )
    if args.stateful:
        _stateful_batch_partition(args)
        if (
            args.stateful_sampling == "mixed"
            and 0 < args.max_train_frames < args.batch_size
        ):
            raise ValueError(
                "mixed max_train_frames must fit at least one complete batch"
            )
    elif args.stream_batch_size:
        raise ValueError("--stream-batch-size requires --stateful")


def _validate_stateful_window_duration(
    requested_ms: float,
    *,
    stateful: bool,
    sequence_loader: bool,
    checkpoint_ms: float,
) -> None:
    """Prevent a sequence checkpoint from silently changing its frame cadence."""

    if not stateful or not sequence_loader:
        return
    requested_us = round(requested_ms * 1_000)
    checkpoint_us = round(checkpoint_ms * 1_000)
    if requested_us != checkpoint_us:
        raise ValueError(
            f"stateful --window-ms {requested_ms:g} does not match checkpoint "
            f"recurrent.window_ms={checkpoint_ms:g}; use the checkpoint cadence"
        )


def _feature_backbone_fingerprint(backbone: nn.Module) -> str:
    """Hash weights used by the full-frame feature path, independent of file path."""

    digest = hashlib.sha256()
    for namespace in ("online_encoder", "scale_embedding"):
        module = getattr(backbone, namespace)
        for name, tensor in sorted(module.state_dict().items()):
            value = tensor.detach().cpu().contiguous()
            identity = (
                f"{namespace}.{name}\0{value.dtype}\0{tuple(value.shape)}\0"
            ).encode("utf-8")
            digest.update(identity)
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _manifest_identity(path: Path) -> dict[str, str]:
    """Identify a dataset manifest by canonical location and file contents."""

    resolved = path.expanduser().resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(resolved), "sha256": digest.hexdigest()}


def _validate_detection_resume_metadata(
    resumed: dict[str, Any],
    args: argparse.Namespace,
    *,
    pretrain_config_hash: str,
    backbone_fingerprint: str,
) -> None:
    """Reject resume combinations that would silently mix experiments."""

    schema = resumed.get("schema")
    if schema not in {
        "event-window-jepa-gen1-yolox-v1",
        "event-window-jepa-gen1-yolox-v2",
    }:
        raise ValueError("unsupported detection checkpoint")
    if bool(resumed.get("stateful", False)) != args.stateful:
        raise ValueError("detection resume stateful mode does not match checkpoint")
    if schema == "event-window-jepa-gen1-yolox-v1":
        # Stateless v1 keeps its historical resume behavior. Stateful v1 has no
        # sampling identity and traversed a different full-stream train set.
        if args.stateful:
            raise ValueError(
                "legacy stateful detection checkpoint predates sampling identity "
                "and cannot safely resume the RVT sampling protocol"
            )
        return

    required_identity = {
        "pretrain_config_hash": pretrain_config_hash,
        "backbone_fingerprint": backbone_fingerprint,
        "backbone_init": args.backbone_init,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "precision": args.precision,
    }
    if args.stateful:
        required_identity.update(
            max_train_frames=args.max_train_frames,
            max_val_frames=args.max_val_frames,
            train_manifest_identity=_manifest_identity(args.train_manifest),
            val_manifest_identity=_manifest_identity(args.val_manifest),
        )
        saved_lane_schema = resumed.get("stateful_lane_schema")
        if saved_lane_schema in {
            LEGACY_STATEFUL_LANE_SCHEMA,
            SEQUENCE_STATEFUL_LANE_SCHEMA,
        }:
            raise ValueError(
                "stateful detection checkpoint predates label-aware sampling "
                "identity and cannot safely resume"
            )
        if saved_lane_schema == STATEFUL_LANE_SCHEMA:
            if resumed.get("sequence_length") != args.sequence_length:
                raise ValueError(
                    "detection resume sequence_length does not match checkpoint"
                )
            if resumed.get("stateful_sampling") != args.stateful_sampling:
                raise ValueError(
                    "detection resume stateful_sampling does not match checkpoint"
                )
            if resumed.get("stream_batch_size") != _stateful_batch_partition(args)[0]:
                raise ValueError(
                    "detection resume stream_batch_size does not match checkpoint"
                )
        else:
            raise ValueError("detection resume stateful_lane_schema does not match")
    elif resumed.get("stateful_lane_schema") is not None:
        raise ValueError("stateless detection checkpoint has a lane schema")
    for name, expected in required_identity.items():
        if resumed.get(name) != expected:
            raise ValueError(f"detection resume {name} does not match checkpoint")
    if round(float(resumed.get("window_ms", -1.0)) * 1_000) != round(
        args.window_ms * 1_000
    ):
        raise ValueError("detection resume window_ms does not match checkpoint")


def train(args: argparse.Namespace) -> None:
    _validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    backbone, config = load_pretrained_model(args.checkpoint, device=device)
    _validate_stateful_window_duration(
        args.window_ms,
        stateful=args.stateful,
        sequence_loader=bool(config.recurrent.sequence_loader),
        checkpoint_ms=config.recurrent.window_ms,
    )
    if args.backbone_init == "random":
        from event_window_jepa.train.pretrain import build_model

        torch.manual_seed(args.seed)
        backbone = build_model(config).to(device)
    pretrain_config_hash = config_hash(config)
    backbone_fingerprint = _feature_backbone_fingerprint(backbone)
    recurrent_checkpoint = isinstance(
        backbone.online_encoder, RecurrentVJEPA21EventVisionTransformer
    )
    if recurrent_checkpoint and not args.stateful:
        require_feedforward_feature_model(backbone, caller="Gen1 detection")
    resumed: dict[str, Any] | None = None
    if args.resume is not None:
        resumed = torch.load(args.resume, map_location="cpu", weights_only=False)
        _validate_detection_resume_metadata(
            resumed,
            args,
            pretrain_config_hash=pretrain_config_hash,
            backbone_fingerprint=backbone_fingerprint,
        )
    components = _load_rvt_components()
    representation = _representation(config)
    train_sources = _read_label_sources(args.train_manifest)
    val_sources = _read_label_sources(args.val_manifest)
    duration_us = round(args.window_ms * 1_000)
    train_labeled_references = _frame_references(
        train_sources,
        maximum_window_us=duration_us,
        maximum_frames=0 if args.stateful else args.max_train_frames,
        seed=args.seed,
    )
    val_labeled_references = _frame_references(
        val_sources,
        maximum_window_us=duration_us,
        maximum_frames=0 if args.stateful else args.max_val_frames,
        seed=args.seed + 1,
    )
    if args.stateful:
        stream_batch_size, random_batch_size = _stateful_batch_partition(args)
        train_stream_label_limit = args.max_train_frames
        train_random_anchor_limit = args.max_train_frames
        if args.stateful_sampling == "mixed" and args.max_train_frames:
            train_stream_label_limit = (
                args.max_train_frames * stream_batch_size // args.batch_size
            )
            train_random_anchor_limit = (
                args.max_train_frames - train_stream_label_limit
            )
        train_references: Sequence[FrameReference | StreamFrameReference] = (
            _training_stream_references(
                train_sources,
                train_labeled_references,
                duration_us=duration_us,
                sequence_length=args.sequence_length,
                maximum_labeled_frames=train_stream_label_limit,
                stream_lanes=stream_batch_size,
            )
            if stream_batch_size
            else ()
        )
        val_references: Sequence[FrameReference | StreamFrameReference] = (
            _stream_references(
                val_sources,
                val_labeled_references,
                duration_us=duration_us,
                maximum_labeled_frames=args.max_val_frames,
                stream_lanes=args.batch_size,
            )
        )
    else:
        stream_batch_size, random_batch_size = 0, 0
        train_stream_label_limit = args.max_train_frames
        train_random_anchor_limit = 0
        train_references = train_labeled_references
        val_references = val_labeled_references
    if (
        (not args.stateful and not train_references)
        or (stream_batch_size and not train_references)
        or not val_references
    ):
        raise ValueError("no valid labeled frames remain")
    model = WindowJEPAYOLOX(
        backbone,
        components.head_type,
        freeze_backbone=not args.unfreeze_backbone,
    ).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    start_epoch = 0
    best_ap = float("-inf")
    best_epoch = 0
    if resumed is not None:
        incompatible = model.load_state_dict(resumed["model"], strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing = [
            name
            for name in incompatible.missing_keys
            if not name.startswith("backbone.")
        ]
        if unexpected or missing:
            raise ValueError(
                f"incompatible detection checkpoint: missing={missing}, "
                f"unexpected={unexpected}"
            )
        optimizer.load_state_dict(resumed["optimizer"])
        start_epoch = int(resumed["epoch"])
        best_ap = float(resumed.get("best_ap", float("-inf")))
        best_epoch = int(resumed.get("best_epoch", 0))

    dataset_type = Gen1StreamDetectionDataset if args.stateful else Gen1DetectionDataset
    train_dataset = dataset_type(
        args.train_manifest,
        train_sources,
        train_references,
        duration_ms=args.window_ms,
        representation=representation,
    )
    val_dataset = dataset_type(
        args.val_manifest,
        val_sources,
        val_references,
        duration_ms=args.window_ms,
        representation=representation,
    )
    train_generator = torch.Generator().manual_seed(args.seed)
    loader_options = {
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    if args.stateful:
        train_stream_sampler: StreamLaneBatchSampler | None = None
        train_random_sampler: RandomLabelClipBatchSampler | None = None
        if stream_batch_size:
            train_stream_sampler = StreamLaneBatchSampler(
                train_dataset.references,
                stream_batch_size,
                sequence_length=args.sequence_length,
                reset_every_chunk=args.stateful_sampling == "stream_reset",
                shuffle=True,
                seed=args.seed,
            )
        if random_batch_size:
            train_random_sampler = RandomLabelClipBatchSampler(
                train_sources,
                train_labeled_references,
                duration_us=duration_us,
                sequence_length=args.sequence_length,
                batch_size=random_batch_size,
                lane_offset=stream_batch_size,
                maximum_labeled_frames=train_random_anchor_limit,
                drop_last=args.stateful_sampling == "mixed",
                seed=args.seed + 17,
            )
        if args.stateful_sampling == "mixed":
            if train_stream_sampler is None or train_random_sampler is None:
                raise RuntimeError("mixed stateful samplers were not constructed")
            train_batch_sampler: Any = MixedStatefulBatchSampler(
                train_stream_sampler, train_random_sampler
            )
        elif args.stateful_sampling in {"stream", "stream_reset"}:
            if train_stream_sampler is None:
                raise RuntimeError("stream sampler was not constructed")
            train_batch_sampler = train_stream_sampler
        else:
            if train_random_sampler is None:
                raise RuntimeError("random sampler was not constructed")
            train_batch_sampler = train_random_sampler
        val_lane_sampler = StreamLaneBatchSampler(
            val_dataset.references,
            args.batch_size,
            sequence_length=args.sequence_length,
            shuffle=False,
            seed=args.seed + 1,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            collate_fn=_collate_stream_sequence_detection,
            **loader_options,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_lane_sampler,
            collate_fn=_collate_stream_sequence_detection,
            **loader_options,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=train_generator,
            collate_fn=_collate_detection,
            **loader_options,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=_collate_detection,
            **loader_options,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "train.jsonl"
    effective_sequence_length = args.sequence_length if args.stateful else 1
    random_anchor_count = (
        len(train_random_sampler.anchors)
        if args.stateful and train_random_sampler is not None
        else 0
    )
    print(
        f"[gen1-detect] train_stream_valid_windows="
        f"{len(train_dataset) if args.stateful else 0}, "
        f"train_random_valid_anchors={random_anchor_count}, "
        f"train_labeled_frames={len(train_dataset) if not args.stateful else 'n/a'}, "
        f"val_full_stream_windows={len(val_dataset) if args.stateful else 0}, "
        f"val_labeled_frames={len(val_dataset) if not args.stateful else 'n/a'}, "
        f"init={args.backbone_init}, frozen={not args.unfreeze_backbone}, "
        f"stateful={args.stateful}, recurrent={recurrent_checkpoint}, "
        f"train_chunks={len(train_loader)}, batch_size={args.batch_size}, "
        f"sequence_length={effective_sequence_length}, "
        f"max_windows_per_chunk={args.batch_size * effective_sequence_length}, "
        f"sampling={args.stateful_sampling if args.stateful else 'labeled-frames'}, "
        f"stream_clips_per_batch={stream_batch_size}, "
        f"random_clips_per_batch={random_batch_size}, "
        f"stream_label_limit={train_stream_label_limit}, "
        f"random_anchor_limit={train_random_anchor_limit}, "
        f"eval_sampling={'full-stream' if args.stateful else 'labeled-frames'}, "
        "augmentation=none(identity-consistent-across-time), "
        f"backbone_bptt={'disabled-frozen' if args.stateful else 'not-applicable'}",
        flush=True,
    )
    for epoch in range(start_epoch, args.epochs):
        if args.stateful:
            train_batch_sampler.set_epoch(epoch)
        model.train()
        running: dict[str, float] = {}
        samples = 0
        optimizer_updates = 0
        train_input_frames = 0
        train_stream_labeled_frames = 0
        train_random_labeled_frames = 0
        train_stream_clips = 0
        train_random_clips = 0
        progress = tqdm(train_loader, desc=f"detection epoch {epoch + 1}/{args.epochs}")
        state_manager = StreamStateManager(
            recurrent=model.recurrent,
            stride_us=duration_us,
        )
        for batch in progress:
            if args.stateful:
                batch_counts = _stateful_sampling_counts(
                    batch,
                    sampling=args.stateful_sampling,
                    stream_batch_size=stream_batch_size,
                )
                optimizer.zero_grad(set_to_none=True)
                forward = _forward_stateful_sequence(
                    model,
                    batch,
                    state_manager,
                    duration_ms=args.window_ms,
                    device=device,
                    precision=args.precision,
                    compute_losses=True,
                )
                train_input_frames += forward.input_frames
                train_stream_labeled_frames += batch_counts[0]
                train_random_labeled_frames += batch_counts[1]
                train_stream_clips += batch_counts[2]
                train_random_clips += batch_counts[3]
                labeled_batch_size = forward.labeled_frames
                if batch_counts[0] + batch_counts[1] != labeled_batch_size:
                    raise RuntimeError(
                        "stateful sampling accounting disagrees with head labels"
                    )
                if not labeled_batch_size:
                    continue
                losses = forward.losses
            else:
                images, targets, _, _ = batch
                labeled_batch_size = len(images)
                train_input_frames += labeled_batch_size
                optimizer.zero_grad(set_to_none=True)
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                duration = torch.full((len(images),), args.window_ms, device=device)
                context = _autocast_context(device, args.precision)
                with context:
                    _, losses = model(images, duration, targets)
            if losses is None:
                raise RuntimeError("YOLOX returned no training losses")
            loss = losses["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 10.0)
            optimizer.step()
            optimizer_updates += 1
            samples += labeled_batch_size
            for name, value in losses.items():
                numeric = float(value.detach()) if torch.is_tensor(value) else float(value)
                running[name] = (
                    running.get(name, 0.0) + numeric * labeled_batch_size
                )
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}", refresh=False)
        if samples == 0:
            raise RuntimeError("detection epoch contained no labeled frames")
        train_metrics = {name: value / samples for name, value in running.items()}
        should_evaluate = (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs
        validation_metrics: dict[str, float] = {}
        if should_evaluate:
            evaluator = _evaluate_stateful if args.stateful else _evaluate
            validation_metrics = evaluator(
                model,
                val_loader,
                components,
                duration_ms=args.window_ms,
                confidence_threshold=args.confidence_threshold,
                nms_threshold=args.nms_threshold,
                device=device,
                precision=args.precision,
            )
        record = {
            "epoch": epoch + 1,
            "train": train_metrics,
            "validation": validation_metrics,
            "train_labeled_frames": samples,
            "labeled_head_frames": samples,
            "train_input_frames": train_input_frames,
            "train_stream_labeled_frames": (
                train_stream_labeled_frames if args.stateful else None
            ),
            "train_random_labeled_frames": (
                train_random_labeled_frames if args.stateful else None
            ),
            "train_stream_clips": train_stream_clips if args.stateful else None,
            "train_random_clips": train_random_clips if args.stateful else None,
            "train_batches": len(train_loader),
            "train_chunks": len(train_loader) if args.stateful else None,
            "sequence_length": effective_sequence_length,
            "stateful_sampling": args.stateful_sampling if args.stateful else None,
            "stream_batch_size": stream_batch_size if args.stateful else None,
            "optimizer_updates": optimizer_updates,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(f"[gen1-detect] {json.dumps(record, sort_keys=True)}", flush=True)
        current_ap = validation_metrics.get("AP")
        if current_ap is not None and current_ap > best_ap:
            best_ap = current_ap
            best_epoch = epoch + 1
            _save_checkpoint(
                args.output_dir / "checkpoint-best.pt",
                model,
                optimizer,
                epoch=epoch + 1,
                args=args,
                metrics=validation_metrics,
                best_ap=best_ap,
                best_epoch=best_epoch,
                pretrain_config_hash=pretrain_config_hash,
                backbone_fingerprint=backbone_fingerprint,
            )
        _save_checkpoint(
            args.output_dir / "checkpoint-latest.pt",
            model,
            optimizer,
            epoch=epoch + 1,
            args=args,
            metrics=validation_metrics,
            best_ap=best_ap,
            best_epoch=best_epoch,
            pretrain_config_hash=pretrain_config_hash,
            backbone_fingerprint=backbone_fingerprint,
        )


def main() -> None:
    train(_parse_args())


if __name__ == "__main__":
    main()
