from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from event_window_jepa.data.timestamp_index import TimestampIndex
from event_window_jepa.data.types import EventWindow, SequenceInfo


class EventStore(ABC):
    """Random-access source of sorted event sequences."""

    @abstractmethod
    def sequences(self, split: str | None = None) -> tuple[SequenceInfo, ...]:
        raise NotImplementedError

    @abstractmethod
    def slice(self, sequence_id: str, t_end_us: int, duration_us: int) -> EventWindow:
        raise NotImplementedError


def _manifest_boolean(row: Mapping[str, Any], name: str, default: bool) -> bool:
    value = row.get(name, default)
    if not isinstance(value, bool):
        raise TypeError(f"manifest field {name} must be boolean")
    return value


def _validate_arrays(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    aliases = {
        "x": ("x",),
        "y": ("y",),
        "t_us": ("t_us", "t", "timestamp", "timestamps"),
        "polarity": ("polarity", "p"),
    }
    result: dict[str, np.ndarray] = {}
    for canonical, candidates in aliases.items():
        key = next((candidate for candidate in candidates if candidate in arrays), None)
        if key is None:
            raise KeyError(f"missing event field {canonical!r}")
        result[canonical] = np.asarray(arrays[key])

    size = len(result["x"])
    if any(value.ndim != 1 or len(value) != size for value in result.values()):
        raise ValueError("event arrays must be one-dimensional and equally sized")
    for key in ("x", "y", "t_us"):
        if not np.issubdtype(result[key].dtype, np.integer):
            raise TypeError(f"event field {key} must use an integer dtype")
    polarity_dtype = result["polarity"].dtype
    if not (
        np.issubdtype(polarity_dtype, np.integer)
        or np.issubdtype(polarity_dtype, np.bool_)
    ):
        raise TypeError("event field polarity must use an integer or boolean dtype")
    return result


def _validate_sequence_values(arrays: Mapping[str, np.ndarray], info: SequenceInfo) -> None:
    timestamps = arrays["t_us"]
    if timestamps.size and (
        int(timestamps[0]) < info.t_start_us or int(timestamps[-1]) > info.t_end_us
    ):
        raise ValueError(f"timestamps for {info.sequence_id!r} exceed manifest bounds")
    if (
        np.any(arrays["x"] < 0)
        or np.any(arrays["x"] >= info.width)
        or np.any(arrays["y"] < 0)
        or np.any(arrays["y"] >= info.height)
    ):
        raise ValueError(f"sequence {info.sequence_id!r} contains out-of-bounds coordinates")
    polarities = set(np.unique(arrays["polarity"]).tolist())
    if not polarities.issubset({-1, 0, 1}) or (-1 in polarities and 0 in polarities):
        raise ValueError(f"sequence {info.sequence_id!r} mixes polarity encodings")


class InMemoryEventStore(EventStore):
    def __init__(
        self,
        sequences: Mapping[str, Mapping[str, np.ndarray]],
        metadata: Mapping[str, SequenceInfo],
    ) -> None:
        if set(sequences) != set(metadata):
            raise ValueError("sequences and metadata must contain identical ids")
        self._arrays = {key: _validate_arrays(value) for key, value in sequences.items()}
        self._metadata = dict(metadata)
        for key, arrays in self._arrays.items():
            _validate_sequence_values(arrays, self._metadata[key])
        self._indices = {
            key: TimestampIndex(value["t_us"]) for key, value in self._arrays.items()
        }

    def sequences(self, split: str | None = None) -> tuple[SequenceInfo, ...]:
        values = self._metadata.values()
        if split is not None:
            values = (value for value in values if value.split == split)
        return tuple(sorted(values, key=lambda item: item.sequence_id))

    def slice(self, sequence_id: str, t_end_us: int, duration_us: int) -> EventWindow:
        if duration_us <= 0:
            raise ValueError("duration_us must be positive")
        info = self._metadata[sequence_id]
        if t_end_us > info.t_end_us:
            raise ValueError("requested end exceeds the sequence end")
        t_start_us = t_end_us - duration_us
        if t_start_us < info.t_start_us:
            raise ValueError("requested window starts before the sequence")
        arrays = self._arrays[sequence_id]
        left, right = self._indices[sequence_id].bounds(t_start_us, t_end_us)
        return EventWindow(
            x=arrays["x"][left:right],
            y=arrays["y"][left:right],
            t_us=arrays["t_us"][left:right],
            polarity=arrays["polarity"][left:right],
            t_start_us=t_start_us,
            t_end_us=t_end_us,
            height=info.height,
            width=info.width,
        )


class NpzEventStore(EventStore):
    """Manifest-backed NPZ event store with a small per-process LRU cache.

    Each JSONL row must contain ``sequence_id``, ``path``, ``height``,
    ``width``, ``t_start_us`` and ``t_end_us``. ``path`` is resolved relative
    to the manifest. The NPZ file contains integer arrays ``x``, ``y``,
    ``t_us`` (or ``t``), and ``polarity`` (or ``p``).
    """

    def __init__(self, manifest: str | Path, cache_size: int = 2) -> None:
        self.manifest = Path(manifest).expanduser().resolve()
        if cache_size <= 0:
            raise ValueError("cache_size must be positive")
        self.cache_size = cache_size
        self._metadata = self._read_manifest(self.manifest)
        self._cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
        self._indices: dict[str, TimestampIndex] = {}

    @staticmethod
    def _read_manifest(path: Path) -> dict[str, SequenceInfo]:
        metadata: dict[str, SequenceInfo] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row: dict[str, Any] = json.loads(line)
                sequence_id = str(row["sequence_id"])
                if sequence_id in metadata:
                    raise ValueError(f"duplicate sequence_id on line {line_number}")
                event_path = Path(row["path"])
                if not event_path.is_absolute():
                    event_path = (path.parent / event_path).resolve()
                metadata[sequence_id] = SequenceInfo(
                    sequence_id=sequence_id,
                    path=event_path,
                    height=int(row["height"]),
                    width=int(row["width"]),
                    t_start_us=int(row["t_start_us"]),
                    t_end_us=int(row["t_end_us"]),
                    split=str(row.get("split", "train")),
                    dataset=str(row.get("dataset", "unknown")),
                    source_time_origin_us=int(row.get("source_time_origin_us", 0)),
                    coordinate_frame=str(row.get("coordinate_frame", "unknown")),
                    source_width=(
                        None if row.get("source_width") is None else int(row["source_width"])
                    ),
                    source_height=(
                        None
                        if row.get("source_height") is None
                        else int(row["source_height"])
                    ),
                    spatial_downsample=int(row.get("spatial_downsample", 1)),
                    camera=str(row.get("camera", "unknown")),
                    timestamp_reference=str(
                        row.get("timestamp_reference", "unknown")
                    ),
                    timestamp_synchronized=_manifest_boolean(
                        row, "timestamp_synchronized", False
                    ),
                    source_recording_id=(
                        None
                        if row.get("source_recording_id") is None
                        else str(row["source_recording_id"])
                    ),
                )
        if not metadata:
            raise ValueError("manifest contains no sequences")
        path_splits: dict[Path, set[str]] = {}
        for info in metadata.values():
            if info.path is not None:
                path_splits.setdefault(info.path, set()).add(info.split)
        leaked_paths = [path for path, splits in path_splits.items() if len(splits) > 1]
        if leaked_paths:
            raise ValueError("an NPZ sequence path is assigned to multiple splits")
        recording_splits: dict[str, set[str]] = {}
        for info in metadata.values():
            recording_id = info.source_recording_id or info.sequence_id
            recording_splits.setdefault(recording_id, set()).add(info.split)
        if any(len(splits) > 1 for splits in recording_splits.values()):
            raise ValueError("an NPZ source recording is assigned to multiple splits")
        return metadata

    def _load(self, sequence_id: str) -> dict[str, np.ndarray]:
        if sequence_id in self._cache:
            arrays = self._cache.pop(sequence_id)
            self._cache[sequence_id] = arrays
            return arrays
        info = self._metadata[sequence_id]
        if info.path is None:
            raise RuntimeError("NPZ sequence has no path")
        with np.load(info.path, allow_pickle=False) as archive:
            arrays = _validate_arrays({key: archive[key] for key in archive.files})
        _validate_sequence_values(arrays, info)
        self._cache[sequence_id] = arrays
        self._indices[sequence_id] = TimestampIndex(arrays["t_us"])
        while len(self._cache) > self.cache_size:
            evicted_id, _ = self._cache.popitem(last=False)
            self._indices.pop(evicted_id, None)
        return arrays

    def sequences(self, split: str | None = None) -> tuple[SequenceInfo, ...]:
        values = self._metadata.values()
        if split is not None:
            values = (value for value in values if value.split == split)
        return tuple(sorted(values, key=lambda item: item.sequence_id))

    def slice(self, sequence_id: str, t_end_us: int, duration_us: int) -> EventWindow:
        if duration_us <= 0:
            raise ValueError("duration_us must be positive")
        info = self._metadata[sequence_id]
        if t_end_us > info.t_end_us:
            raise ValueError("requested end exceeds the sequence end")
        t_start_us = t_end_us - duration_us
        if t_start_us < info.t_start_us:
            raise ValueError("requested window starts before the sequence")
        arrays = self._load(sequence_id)
        left, right = self._indices[sequence_id].bounds(t_start_us, t_end_us)
        return EventWindow(
            x=arrays["x"][left:right],
            y=arrays["y"][left:right],
            t_us=arrays["t_us"][left:right],
            polarity=arrays["polarity"][left:right],
            t_start_us=t_start_us,
            t_end_us=t_end_us,
            height=info.height,
            width=info.width,
        )

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_cache"] = OrderedDict()
        state["_indices"] = {}
        return state


def _h5_bisect_right(
    dataset: Any, value: int, low: int = 0, high: int | None = None
) -> int:
    """bisect_right without materializing a large HDF5 timestamp array."""

    if high is None:
        high = len(dataset)
    if not 0 <= low <= high <= len(dataset):
        raise ValueError("HDF5 binary-search bounds are invalid")
    while low < high:
        middle = (low + high) // 2
        if int(dataset[middle]) <= value:
            low = middle + 1
        else:
            high = middle
    return low


def _h5_dataset(group: Any, *names: str) -> Any:
    for name in names:
        if name in group:
            return group[name]
    raise KeyError(f"none of the HDF5 datasets exist: {names}")


def _h5_filter_ids(dataset: Any) -> set[int]:
    property_list = dataset.id.get_create_plist()
    return {
        int(property_list.get_filter(index)[0])
        for index in range(property_list.get_nfilters())
    }


def _validate_h5_sequence(group: Any, info: SequenceInfo) -> None:
    """Validate an HDF5 sequence once without materializing it in memory.

    Binary search is only causal when timestamps are sorted integers.  The
    chunked scan also validates the coordinate and polarity contracts used by
    the in-memory and NPZ stores, while keeping peak host memory bounded.
    """

    datasets = {
        "x": _h5_dataset(group, "x"),
        "y": _h5_dataset(group, "y"),
        "t_us": _h5_dataset(group, "t_us", "t", "timestamp", "timestamps"),
        "polarity": _h5_dataset(group, "polarity", "p"),
    }
    lengths: set[int] = set()
    for name, dataset in datasets.items():
        if dataset.ndim != 1:
            raise ValueError(f"HDF5 event field {name} must be one-dimensional")
        lengths.add(len(dataset))
        is_integer = np.issubdtype(dataset.dtype, np.integer)
        if name == "polarity":
            is_integer = is_integer or np.issubdtype(dataset.dtype, np.bool_)
        if not is_integer:
            raise TypeError(f"HDF5 event field {name} must use an integer dtype")
    if len(lengths) != 1:
        raise ValueError("HDF5 event fields must have the same length")

    size = lengths.pop()
    chunk_size = 1_000_000
    previous_timestamp: int | None = None
    polarities: set[int] = set()
    for start in range(0, size, chunk_size):
        stop = min(start + chunk_size, size)
        timestamps = np.asarray(datasets["t_us"][start:stop])
        x = np.asarray(datasets["x"][start:stop])
        y = np.asarray(datasets["y"][start:stop])
        polarity = np.asarray(datasets["polarity"][start:stop])

        if timestamps.size:
            if np.any(timestamps[1:] < timestamps[:-1]):
                raise ValueError("HDF5 timestamps must be sorted in ascending order")
            if previous_timestamp is not None and int(timestamps[0]) < previous_timestamp:
                raise ValueError("HDF5 timestamps must be sorted in ascending order")
            previous_timestamp = int(timestamps[-1])
            if (
                int(timestamps[0]) < info.t_start_us
                or int(timestamps[-1]) > info.t_end_us
            ):
                raise ValueError("HDF5 timestamps exceed manifest bounds")
        if (
            np.any(x < 0)
            or np.any(x >= info.width)
            or np.any(y < 0)
            or np.any(y >= info.height)
        ):
            raise ValueError("HDF5 sequence contains out-of-bounds coordinates")
        polarities.update(int(value) for value in np.unique(polarity).tolist())

    if not polarities.issubset({-1, 0, 1}) or (-1 in polarities and 0 in polarities):
        raise ValueError("HDF5 sequence mixes polarity encodings")


class H5EventStore(EventStore):
    """Worker-safe, lazy HDF5 store for large event sequences.

    Manifest rows use the same metadata as :class:`NpzEventStore` and may add
    ``group`` (default ``/``) plus ``events_group`` (default ``events``).
    Expected datasets below that location are ``x``, ``y``, ``t``/``t_us``
    and ``p``/``polarity``.
    """

    def __init__(
        self,
        manifest: str | Path,
        *,
        max_open_files: int = 8,
        index_cache_size: int = 16,
    ) -> None:
        if max_open_files <= 0 or index_cache_size <= 0:
            raise ValueError("HDF5 cache sizes must be positive")
        try:
            import hdf5plugin  # noqa: F401
            import h5py  # noqa: F401
        except ImportError as error:
            raise ImportError("install event-window-jepa[hdf5] to use H5EventStore") from error
        self.manifest = Path(manifest).expanduser().resolve()
        self._metadata: dict[str, SequenceInfo] = {}
        self._groups: dict[str, tuple[str, str]] = {}
        with self.manifest.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                sequence_id = str(row["sequence_id"])
                if sequence_id in self._metadata:
                    raise ValueError(f"duplicate sequence_id on line {line_number}")
                event_path = Path(row["path"])
                if not event_path.is_absolute():
                    event_path = (self.manifest.parent / event_path).resolve()
                self._metadata[sequence_id] = SequenceInfo(
                    sequence_id=sequence_id,
                    path=event_path,
                    height=int(row["height"]),
                    width=int(row["width"]),
                    t_start_us=int(row["t_start_us"]),
                    t_end_us=int(row["t_end_us"]),
                    split=str(row.get("split", "train")),
                    dataset=str(row.get("dataset", "unknown")),
                    source_time_origin_us=int(row.get("source_time_origin_us", 0)),
                    coordinate_frame=str(row.get("coordinate_frame", "unknown")),
                    source_width=(
                        None if row.get("source_width") is None else int(row["source_width"])
                    ),
                    source_height=(
                        None
                        if row.get("source_height") is None
                        else int(row["source_height"])
                    ),
                    spatial_downsample=int(row.get("spatial_downsample", 1)),
                    camera=str(row.get("camera", "unknown")),
                    timestamp_reference=str(
                        row.get("timestamp_reference", "unknown")
                    ),
                    timestamp_synchronized=_manifest_boolean(
                        row, "timestamp_synchronized", False
                    ),
                    source_recording_id=(
                        None
                        if row.get("source_recording_id") is None
                        else str(row["source_recording_id"])
                    ),
                )
                self._groups[sequence_id] = (
                    str(row.get("group", "/")),
                    str(row.get("events_group", "events")),
                )
        if not self._metadata:
            raise ValueError("manifest contains no sequences")
        assignments: dict[tuple[Path | None, str, str], str] = {}
        for sequence_id, info in self._metadata.items():
            key = (info.path, *self._groups[sequence_id])
            if key in assignments:
                raise ValueError("an HDF5 event group is assigned to multiple sequence ids")
            assignments[key] = info.split
        recording_splits: dict[str, set[str]] = {}
        for info in self._metadata.values():
            recording_id = info.source_recording_id or info.sequence_id
            recording_splits.setdefault(recording_id, set()).add(info.split)
        if any(len(splits) > 1 for splits in recording_splits.values()):
            raise ValueError("an HDF5 source recording is assigned to multiple splits")
        self._validate_manifest_sequences()
        self.max_open_files = max_open_files
        self.index_cache_size = index_cache_size
        self._process_id: int | None = None
        self._handles: OrderedDict[Path, Any] = OrderedDict()
        self._index_cache: OrderedDict[str, tuple[np.ndarray, int]] = OrderedDict()

    @staticmethod
    def _is_canonical_file(handle: Any) -> bool:
        return (
            handle.attrs.get("schema_name") == "event-window-jepa-events"
            and int(handle.attrs.get("schema_version", -1)) == 1
            and bool(handle.attrs.get("complete", False))
        )

    def _validate_canonical_sequence(
        self, handle: Any, group: Any, info: SequenceInfo
    ) -> None:
        if str(handle.attrs.get("sequence_id", "")) != info.sequence_id:
            raise ValueError("canonical HDF5 sequence_id disagrees with manifest")
        if (
            int(handle.attrs.get("width", -1)) != info.width
            or int(handle.attrs.get("height", -1)) != info.height
            or int(handle.attrs.get("duration_us", -1)) != info.t_end_us
            or info.t_start_us != 0
        ):
            raise ValueError("canonical HDF5 geometry/time bounds disagree with manifest")
        metadata_pairs = {
            "source_dataset": info.dataset,
            "source_time_origin_us": info.source_time_origin_us,
            "coordinate_frame": info.coordinate_frame,
            "source_width": info.source_width,
            "source_height": info.source_height,
            "spatial_downsample": info.spatial_downsample,
            "camera": info.camera,
            "source_timestamp_reference": info.timestamp_reference,
            "source_timestamp_synchronized": info.timestamp_synchronized,
            "source_recording_id": info.source_recording_id,
            "logical_split": info.split,
        }
        for attribute, expected in metadata_pairs.items():
            if expected is None:
                continue
            actual = handle.attrs.get(attribute)
            if isinstance(expected, int):
                matches = actual is not None and int(actual) == expected
            else:
                matches = str(actual) == expected
            if not matches:
                raise ValueError(
                    f"canonical HDF5 attribute {attribute} disagrees with manifest"
                )
        datasets = {
            "x": _h5_dataset(group, "x"),
            "y": _h5_dataset(group, "y"),
            "t_us": _h5_dataset(group, "t_us"),
            "polarity": _h5_dataset(group, "polarity"),
        }
        lengths = {len(dataset) for dataset in datasets.values()}
        if len(lengths) != 1 or not lengths or lengths == {0}:
            raise ValueError("canonical HDF5 event arrays are empty or unequal")
        for name, dataset in datasets.items():
            if dataset.ndim != 1:
                raise ValueError(f"canonical HDF5 field {name} is not one-dimensional")
            is_integer = np.issubdtype(dataset.dtype, np.integer)
            if name == "polarity":
                is_integer = is_integer or np.issubdtype(dataset.dtype, np.bool_)
            if not is_integer:
                raise TypeError(f"canonical HDF5 field {name} is not integer")
        size = lengths.pop()
        if int(handle.attrs.get("event_count", -1)) != size:
            raise ValueError("canonical HDF5 event_count attribute is inconsistent")
        required_filters = {2, 3, 32015}
        for dataset in datasets.values():
            if not required_filters.issubset(_h5_filter_ids(dataset)):
                raise ValueError(
                    "canonical event arrays must use shuffle, Fletcher32, and Zstd"
                )
        timestamps = datasets["t_us"]
        if (
            int(timestamps[0]) < info.t_start_us
            or int(timestamps[-1]) > info.t_end_us
        ):
            raise ValueError("canonical HDF5 timestamps exceed manifest coverage")
        root_group, _ = self._groups[info.sequence_id]
        root = handle[root_group]
        if "index/ms_to_event_idx" not in root:
            raise ValueError("canonical HDF5 file has no millisecond event index")
        index_dataset = root["index/ms_to_event_idx"]
        index = np.asarray(index_dataset)
        step_us = int(index_dataset.attrs.get("step_us", 0))
        expected_index_length = info.t_end_us // step_us + 2 if step_us > 0 else -1
        if (
            index_dataset.ndim != 1
            or not np.issubdtype(index_dataset.dtype, np.integer)
            or len(index) != expected_index_length
            or int(index[0]) != 0
            or int(index[-1]) != size
            or np.any(index[1:] < index[:-1])
            or np.any(index > size)
            or not required_filters.issubset(_h5_filter_ids(index_dataset))
        ):
            raise ValueError("canonical HDF5 millisecond event index is invalid")

    def _validate_manifest_sequences(self) -> None:
        """Fail before worker startup if an input cannot support exact slicing."""

        import h5py

        for sequence_id, info in self._metadata.items():
            if info.path is None:
                raise RuntimeError("HDF5 sequence has no path")
            with h5py.File(info.path, "r") as handle:
                root_group, events_group = self._groups[sequence_id]
                root = handle[root_group]
                group = root[events_group] if events_group else root
                try:
                    if self._is_canonical_file(handle):
                        self._validate_canonical_sequence(handle, group, info)
                    else:
                        _validate_h5_sequence(group, info)
                except (KeyError, TypeError, ValueError) as error:
                    raise type(error)(
                        f"invalid HDF5 sequence {sequence_id!r}: {error}"
                    ) from error

    def _handle(self, path: Path) -> Any:
        import h5py

        process_id = os.getpid()
        if self._process_id != process_id:
            self.close()
            self._process_id = process_id
        if path in self._handles:
            handle = self._handles.pop(path)
            self._handles[path] = handle
            return handle
        handle = h5py.File(path, "r")
        self._handles[path] = handle
        while len(self._handles) > self.max_open_files:
            _, evicted = self._handles.popitem(last=False)
            evicted.close()
        return handle

    def _sequence_root(self, sequence_id: str) -> Any:
        info = self._metadata[sequence_id]
        if info.path is None:
            raise RuntimeError("HDF5 sequence has no path")
        root_group, _ = self._groups[sequence_id]
        return self._handle(info.path)[root_group]

    def _event_group(self, sequence_id: str) -> Any:
        root = self._sequence_root(sequence_id)
        _, events_group = self._groups[sequence_id]
        return root[events_group] if events_group else root

    def _coarse_index(self, sequence_id: str) -> tuple[np.ndarray, int] | None:
        if sequence_id in self._index_cache:
            cached = self._index_cache.pop(sequence_id)
            self._index_cache[sequence_id] = cached
            return cached
        root = self._sequence_root(sequence_id)
        if "index/ms_to_event_idx" not in root:
            return None
        dataset = root["index/ms_to_event_idx"]
        cached = (np.asarray(dataset), int(dataset.attrs["step_us"]))
        self._index_cache[sequence_id] = cached
        while len(self._index_cache) > self.index_cache_size:
            self._index_cache.popitem(last=False)
        return cached

    def _right_bound(self, sequence_id: str, timestamps: Any, value: int) -> int:
        coarse = self._coarse_index(sequence_id)
        if coarse is None or value < 0:
            return _h5_bisect_right(timestamps, value)
        index, step_us = coarse
        bucket = value // step_us
        if bucket + 1 >= len(index):
            return _h5_bisect_right(timestamps, value)
        low = int(index[bucket])
        high = int(index[bucket + 1])
        if low == high:
            return low
        block = np.asarray(timestamps[low:high])
        return low + int(np.searchsorted(block, value, side="right"))

    @staticmethod
    def _dataset(group: Any, *names: str) -> Any:
        return _h5_dataset(group, *names)

    def sequences(self, split: str | None = None) -> tuple[SequenceInfo, ...]:
        values = self._metadata.values()
        if split is not None:
            values = (value for value in values if value.split == split)
        return tuple(sorted(values, key=lambda item: item.sequence_id))

    def slice(self, sequence_id: str, t_end_us: int, duration_us: int) -> EventWindow:
        if duration_us <= 0:
            raise ValueError("duration_us must be positive")
        info = self._metadata[sequence_id]
        t_start_us = t_end_us - duration_us
        if t_end_us > info.t_end_us or t_start_us < info.t_start_us:
            raise ValueError("requested window exceeds the sequence time range")
        group = self._event_group(sequence_id)
        timestamps = self._dataset(group, "t_us", "t", "timestamp", "timestamps")
        left = self._right_bound(sequence_id, timestamps, t_start_us)
        right = self._right_bound(sequence_id, timestamps, t_end_us)
        return EventWindow(
            x=np.asarray(self._dataset(group, "x")[left:right]),
            y=np.asarray(self._dataset(group, "y")[left:right]),
            t_us=np.asarray(timestamps[left:right]),
            polarity=np.asarray(self._dataset(group, "polarity", "p")[left:right]),
            t_start_us=t_start_us,
            t_end_us=t_end_us,
            height=info.height,
            width=info.width,
        )

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles = OrderedDict()
        self._index_cache = OrderedDict()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handles"] = OrderedDict()
        state["_index_cache"] = OrderedDict()
        state["_process_id"] = None
        return state

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
