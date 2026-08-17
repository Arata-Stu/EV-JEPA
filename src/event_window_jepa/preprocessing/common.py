from __future__ import annotations

import hashlib
import json
import math
import os
import socket
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Protocol

import numpy as np


SCHEMA_NAME = "event-window-jepa-events"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EventSourceMetadata:
    sequence_id: str
    dataset: str
    source_path: Path
    camera: str
    width: int
    height: int
    event_count: int
    first_timestamp_us: int
    last_timestamp_us: int
    coordinate_frame: str = "raw"
    attributes: Mapping[str, str | int | float | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sequence_id or not self.dataset or not self.camera:
            raise ValueError("sequence_id, dataset, and camera cannot be empty")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("source resolution must be positive")
        if self.event_count <= 0:
            raise ValueError("an event sequence cannot be empty")
        if self.last_timestamp_us < self.first_timestamp_us:
            raise ValueError("source timestamps are not monotonic")
        timestamp_reference = self.attributes.get("timestamp_reference", "unknown")
        if not isinstance(timestamp_reference, str) or not timestamp_reference:
            raise TypeError("timestamp_reference must be a non-empty string")
        timestamp_synchronized = self.attributes.get("timestamp_synchronized", False)
        if not isinstance(timestamp_synchronized, bool):
            raise TypeError("timestamp_synchronized must be boolean")


class EventSource(Protocol):
    metadata: EventSourceMetadata

    def iter_event_chunks(
        self, chunk_events: int, start_event: int = 0
    ) -> Iterator[Mapping[str, np.ndarray]]:
        """Yield equally-sized 1-D arrays named x, y, t_us, and polarity."""

    def close(self) -> None: ...


@dataclass(frozen=True)
class PreprocessOptions:
    spatial_downsample: int = 1
    read_chunk_events: int = 1_000_000
    hdf5_chunk_events: int = 262_144
    zstd_level: int = 5
    index_step_us: int = 1_000
    timestamp_dtype: str = "auto"
    overwrite: bool = False
    skip_existing: bool = False
    resume_partial: bool = True

    def __post_init__(self) -> None:
        if self.spatial_downsample <= 0:
            raise ValueError("spatial_downsample must be a positive integer")
        if self.read_chunk_events <= 0 or self.hdf5_chunk_events <= 0:
            raise ValueError("event chunk sizes must be positive")
        if not 1 <= self.zstd_level <= 22:
            raise ValueError("zstd_level must lie in [1, 22]")
        if self.index_step_us <= 0:
            raise ValueError("index_step_us must be positive")
        if self.timestamp_dtype not in {"auto", "uint32", "uint64"}:
            raise ValueError("timestamp_dtype must be auto, uint32, or uint64")
        if self.overwrite and self.skip_existing:
            raise ValueError("overwrite and skip_existing are mutually exclusive")


def _require_hdf5() -> tuple[Any, Any]:
    try:
        import hdf5plugin
        import h5py
    except ImportError as error:
        raise ImportError(
            "install event-window-jepa[hdf5] to preprocess event datasets"
        ) from error
    return h5py, hdf5plugin


def _append(dataset: Any, values: np.ndarray) -> int:
    start = len(dataset)
    stop = start + len(values)
    dataset.resize((stop,))
    dataset[start:stop] = values
    return start


def _timestamp_numpy_dtype(duration_us: int, requested: str) -> np.dtype[Any]:
    if requested == "uint64":
        return np.dtype(np.uint64)
    if requested == "uint32":
        if duration_us > np.iinfo(np.uint32).max:
            raise ValueError("sequence duration does not fit uint32 timestamps")
        return np.dtype(np.uint32)
    return np.dtype(np.uint32 if duration_us <= np.iinfo(np.uint32).max else np.uint64)


def coarse_index_entries(
    timestamps_us: np.ndarray,
    *,
    next_boundary_us: int,
    step_us: int,
    global_event_offset: int,
) -> tuple[np.ndarray, int]:
    """Return first-ge index entries whose boundaries reach this sorted chunk."""

    timestamps = np.asarray(timestamps_us)
    if timestamps.ndim != 1 or not np.issubdtype(timestamps.dtype, np.integer):
        raise TypeError("timestamps_us must be a one-dimensional integer array")
    if step_us <= 0 or next_boundary_us < 0 or global_event_offset < 0:
        raise ValueError("coarse index coordinates are invalid")
    if not timestamps.size or next_boundary_us > int(timestamps[-1]):
        return np.empty(0, dtype=np.uint64), next_boundary_us
    boundary_count = (int(timestamps[-1]) - next_boundary_us) // step_us + 1
    boundaries = next_boundary_us + step_us * np.arange(
        boundary_count, dtype=np.uint64
    )
    positions = np.searchsorted(timestamps, boundaries, side="left").astype(np.uint64)
    return positions + global_event_offset, next_boundary_us + boundary_count * step_us


def _validate_source_chunk(
    chunk: Mapping[str, np.ndarray], previous_timestamp_us: int | None
) -> tuple[dict[str, np.ndarray], int | None]:
    missing = {"x", "y", "t_us", "polarity"} - set(chunk)
    if missing:
        raise KeyError(f"source event chunk is missing fields: {sorted(missing)}")
    arrays = {name: np.asarray(chunk[name]) for name in ("x", "y", "t_us", "polarity")}
    size = len(arrays["x"])
    if any(values.ndim != 1 or len(values) != size for values in arrays.values()):
        raise ValueError("source event arrays must be one-dimensional and equally sized")
    for name, values in arrays.items():
        is_integer = np.issubdtype(values.dtype, np.integer)
        if name == "polarity":
            is_integer = is_integer or np.issubdtype(values.dtype, np.bool_)
        if not is_integer:
            raise TypeError(f"source event field {name} must use an integer dtype")
    timestamps = arrays["t_us"]
    if timestamps.size:
        if np.any(timestamps[1:] < timestamps[:-1]):
            raise ValueError("source timestamps must be sorted")
        first = int(timestamps[0])
        if previous_timestamp_us is not None and first < previous_timestamp_us:
            raise ValueError("source timestamps decrease across chunks")
        previous_timestamp_us = int(timestamps[-1])
    return arrays, previous_timestamp_us


def _transform_coordinates(
    x: np.ndarray,
    y: np.ndarray,
    *,
    source_width: int,
    source_height: int,
    downsample: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x64 = x.astype(np.int64, copy=False)
    y64 = y.astype(np.int64, copy=False)
    valid = (x64 >= 0) & (x64 < source_width) & (y64 >= 0) & (y64 < source_height)
    if not bool(valid.all()):
        raise ValueError("source event coordinates exceed the declared resolution")
    x64 = x64 // downsample
    y64 = y64 // downsample
    return x64, y64, valid


def _create_event_dataset(
    group: Any,
    name: str,
    dtype: np.dtype[Any] | type[np.generic],
    chunk_events: int,
    compression: Mapping[str, Any],
) -> Any:
    return group.create_dataset(
        name,
        shape=(0,),
        maxshape=(None,),
        dtype=dtype,
        chunks=(chunk_events,),
        shuffle=True,
        fletcher32=True,
        **compression,
    )


def hdf5_filter_ids(dataset: Any) -> tuple[int, ...]:
    property_list = dataset.id.get_create_plist()
    return tuple(
        int(property_list.get_filter(index)[0])
        for index in range(property_list.get_nfilters())
    )


def _validate_compressed_structure(handle: Any) -> int:
    if "events" not in handle or "index/ms_to_event_idx" not in handle:
        raise ValueError("preprocessing output has no events/index")
    event_group = handle["events"]
    required = ("x", "y", "t_us", "polarity")
    if any(name not in event_group for name in required):
        raise ValueError("preprocessing output has missing event fields")
    datasets = [event_group[name] for name in required]
    lengths = {len(dataset) for dataset in datasets}
    if len(lengths) != 1 or not lengths or lengths == {0}:
        raise ValueError("preprocessing output has inconsistent event lengths")
    event_count = lengths.pop()
    if event_count != int(handle.attrs.get("event_count", -1)):
        raise ValueError("preprocessing output event_count attribute is inconsistent")
    for dataset in (*datasets, handle["index/ms_to_event_idx"]):
        filters = set(hdf5_filter_ids(dataset))
        if not {2, 3, 32015}.issubset(filters):
            raise ValueError(
                "preprocessing output must use shuffle, Fletcher32, and Zstd filters"
            )
    index_dataset = handle["index/ms_to_event_idx"]
    index = np.asarray(index_dataset)
    step_us = int(index_dataset.attrs.get("step_us", 0))
    duration_us = int(handle.attrs.get("duration_us", -1))
    expected_length = duration_us // step_us + 2 if step_us > 0 else -1
    if (
        index.ndim != 1
        or len(index) != expected_length
        or int(index[0]) != 0
        or int(index[-1]) != event_count
        or np.any(index[1:] < index[:-1])
        or np.any(index > event_count)
    ):
        raise ValueError("preprocessing output has an invalid coarse event index")
    return event_count


def _metadata_identity(
    metadata: EventSourceMetadata, split: str, config_hash: str
) -> dict[str, str | int]:
    source_stat = metadata.source_path.stat()
    recording_prefix, separator, _ = metadata.sequence_id.rpartition("__")
    source_recording_id = recording_prefix if separator else metadata.sequence_id
    return {
        "sequence_id": metadata.sequence_id,
        "source_recording_id": source_recording_id,
        "source_dataset": metadata.dataset,
        "source_path": str(metadata.source_path.resolve()),
        "source_file_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "camera": metadata.camera,
        "source_width": metadata.width,
        "source_height": metadata.height,
        "source_time_origin_us": metadata.first_timestamp_us,
        "duration_us": metadata.last_timestamp_us - metadata.first_timestamp_us,
        "source_event_count": metadata.event_count,
        "logical_split": split,
        "converter_config_sha256": config_hash,
    }


def _validate_identity_attributes(
    handle: Any, metadata: EventSourceMetadata, split: str, config_hash: str
) -> None:
    for name, expected in _metadata_identity(metadata, split, config_hash).items():
        actual = handle.attrs.get(name)
        if isinstance(expected, int):
            matches = actual is not None and int(actual) == expected
        else:
            matches = str(actual) == expected
        if not matches:
            raise ValueError(f"preprocessing identity mismatch for {name}")


def _write_json_atomic(path: Path, values: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(values, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


@contextmanager
def _exclusive_output_lock(output: Path) -> Iterator[None]:
    """Serialize writers of one output without leaving a stale crash lock.

    The lock file intentionally remains on disk. ``flock`` is attached to the
    open file description, so the operating system releases it if a converter
    is interrupted or killed. Removing the pathname on normal exit would open
    an inode race between a waiting process and a newly started process.
    """

    try:
        import fcntl
    except ImportError as error:  # pragma: no cover - supported targets are POSIX
        raise RuntimeError("preprocessing requires a POSIX advisory file lock") from error

    lock_path = output.with_name(f".{output.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            lock_handle.seek(0)
            owner = lock_handle.read().strip() or "unknown process"
            raise RuntimeError(
                f"another converter owns {output}; lock owner: {owner}"
            ) from error
        lock_handle.seek(0)
        lock_handle.truncate()
        json.dump(
            {"hostname": socket.gethostname(), "pid": os.getpid()},
            lock_handle,
            sort_keys=True,
        )
        lock_handle.write("\n")
        lock_handle.flush()
        os.fsync(lock_handle.fileno())
        yield
    finally:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        finally:
            lock_handle.close()


def _conversion_config_hash(options: PreprocessOptions) -> str:
    values: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "spatial_downsample": options.spatial_downsample,
        "hdf5_chunk_events": options.hdf5_chunk_events,
        "zstd_level": options.zstd_level,
        "index_step_us": options.index_step_us,
        "timestamp_dtype": options.timestamp_dtype,
    }
    payload = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _existing_record(
    path: Path,
    split: str,
    *,
    expected_metadata: EventSourceMetadata | None = None,
    expected_config_hash: str | None = None,
) -> dict[str, Any]:
    h5py, _ = _require_hdf5()
    with h5py.File(path, "r") as handle:
        if handle.attrs.get("schema_name") != SCHEMA_NAME:
            raise ValueError(f"existing output is not a {SCHEMA_NAME} file: {path}")
        if int(handle.attrs.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError(f"unsupported preprocessing schema in {path}")
        if not bool(handle.attrs.get("complete", False)):
            raise ValueError(f"existing preprocessing output is incomplete: {path}")
        if str(handle.attrs.get("logical_split", "")) != split:
            raise ValueError("existing preprocessing output belongs to a different split")
        _validate_compressed_structure(handle)
        if expected_metadata is not None:
            if expected_config_hash is None:
                raise ValueError("expected_config_hash is required with expected_metadata")
            _validate_identity_attributes(
                handle, expected_metadata, split, expected_config_hash
            )
        return {
            "sequence_id": str(handle.attrs["sequence_id"]),
            "path": str(path.resolve()),
            "group": "/",
            "events_group": "events",
            "height": int(handle.attrs["height"]),
            "width": int(handle.attrs["width"]),
            "t_start_us": 0,
            "t_end_us": int(handle.attrs["duration_us"]),
            "split": split,
            "dataset": str(handle.attrs["source_dataset"]),
            "source_recording_id": str(handle.attrs["source_recording_id"]),
            "camera": str(handle.attrs["camera"]),
            "source_time_origin_us": int(handle.attrs["source_time_origin_us"]),
            "coordinate_frame": str(handle.attrs["coordinate_frame"]),
            "source_width": int(handle.attrs["source_width"]),
            "source_height": int(handle.attrs["source_height"]),
            "spatial_downsample": int(handle.attrs["spatial_downsample"]),
            "timestamp_reference": str(handle.attrs["source_timestamp_reference"]),
            "timestamp_synchronized": bool(
                handle.attrs["source_timestamp_synchronized"]
            ),
            "event_count": int(handle.attrs["event_count"]),
            "output_file_size": path.stat().st_size,
            "source_file_size": int(handle.attrs["source_file_size"]),
        }


def _preprocess_sequence_unlocked(
    source: EventSource,
    output_path: str | Path,
    *,
    split: str,
    options: PreprocessOptions,
) -> dict[str, Any]:
    """Stream one raw sequence into an atomic, Zstandard-compressed HDF5 file."""

    try:
        h5py, hdf5plugin = _require_hdf5()
        metadata = source.metadata
        if metadata.dataset == "dsec" and options.spatial_downsample != 1:
            raise ValueError(
                "DSEC must remain at native resolution so its official label and "
                "rectification coordinates remain recoverable"
            )
        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        config_hash = _conversion_config_hash(options)
        if output.exists():
            if options.skip_existing:
                return _existing_record(
                    output,
                    split,
                    expected_metadata=metadata,
                    expected_config_hash=config_hash,
                )
            if not options.overwrite:
                raise FileExistsError(f"output already exists: {output}")

        duration_us = metadata.last_timestamp_us - metadata.first_timestamp_us
        if duration_us <= 0:
            raise ValueError("source sequence must span a positive time interval")
        timestamp_dtype = _timestamp_numpy_dtype(duration_us, options.timestamp_dtype)
        output_width = math.ceil(metadata.width / options.spatial_downsample)
        output_height = math.ceil(metadata.height / options.spatial_downsample)
        if max(output_width, output_height) > np.iinfo(np.uint16).max:
            raise ValueError("output coordinates do not fit uint16")

        compression = dict(hdf5plugin.Zstd(clevel=options.zstd_level))
        temporary = output.with_name(f".{output.name}.partial")
        checkpoint_path = temporary.with_name(f"{temporary.name}.json")
        identity = _metadata_identity(metadata, split, config_hash)
        resume = temporary.exists()
        if checkpoint_path.exists() != resume:
            if options.overwrite:
                temporary.unlink(missing_ok=True)
                checkpoint_path.unlink(missing_ok=True)
                resume = False
            else:
                raise ValueError("partial HDF5/checkpoint pair is incomplete")
        if resume and not options.resume_partial:
            if not options.overwrite:
                raise FileExistsError(
                    f"partial output exists; use resume or --overwrite: {temporary}"
                )
            temporary.unlink()
            checkpoint_path.unlink()
            resume = False

        checkpoint: dict[str, Any] | None = None
        if resume:
            try:
                checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                for name, expected in identity.items():
                    if checkpoint.get(name) != expected:
                        raise ValueError(f"partial checkpoint mismatch for {name}")
                with h5py.File(temporary, "r") as partial_handle:
                    _validate_identity_attributes(
                        partial_handle, metadata, split, config_hash
                    )
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                if not options.overwrite:
                    raise
                temporary.unlink(missing_ok=True)
                checkpoint_path.unlink(missing_ok=True)
                checkpoint = None
                resume = False

        mode = "r+" if resume else "w"
        with h5py.File(temporary, mode) as handle:
            if not resume:
                handle.attrs.update(
                    {
                        "schema_name": SCHEMA_NAME,
                        "schema_version": SCHEMA_VERSION,
                        "complete": False,
                        **identity,
                        "width": output_width,
                        "height": output_height,
                        "spatial_downsample": options.spatial_downsample,
                        "coordinate_frame": metadata.coordinate_frame,
                        "time_unit": "microseconds",
                        "time_origin": "first_source_event",
                        "compression": "hdf5plugin.Zstd",
                        "zstd_level": options.zstd_level,
                        "event_count": 0,
                        "dropped_event_count": 0,
                        "polarity_encoding": "0=OFF,1=ON",
                        "source_timestamp_reference": str(
                            metadata.attributes.get("timestamp_reference", "unknown")
                        ),
                        "source_timestamp_synchronized": bool(
                            metadata.attributes.get("timestamp_synchronized", False)
                        ),
                    }
                )
                for name, value in metadata.attributes.items():
                    handle.attrs[f"source_{name}"] = value
                events = handle.create_group("events")
                _create_event_dataset(
                    events, "x", np.uint16, options.hdf5_chunk_events, compression
                )
                _create_event_dataset(
                    events, "y", np.uint16, options.hdf5_chunk_events, compression
                )
                _create_event_dataset(
                    events,
                    "t_us",
                    timestamp_dtype,
                    options.hdf5_chunk_events,
                    compression,
                )
                _create_event_dataset(
                    events,
                    "polarity",
                    np.uint8,
                    options.hdf5_chunk_events,
                    compression,
                )
                index_group = handle.create_group("index")
                index_dtype = (
                    np.uint32
                    if metadata.event_count <= np.iinfo(np.uint32).max
                    else np.uint64
                )
                index_dataset = _create_event_dataset(
                    index_group,
                    "ms_to_event_idx",
                    index_dtype,
                    min(options.hdf5_chunk_events, 262_144),
                    compression,
                )
                index_dataset.attrs["step_us"] = options.index_step_us
                index_dataset.attrs["semantics"] = (
                    "searchsorted(t_us, boundary, side=left)"
                )
                checkpoint = {
                    **identity,
                    "source_events_read": 0,
                    "written_events": 0,
                    "dropped_events": 0,
                    "index_entries": 0,
                    "next_index_boundary_us": 0,
                    "previous_timestamp_us": None,
                    "seen_polarities": [],
                    "complete": False,
                }
                handle.flush()
                _write_json_atomic(checkpoint_path, checkpoint)

            assert checkpoint is not None
            events = handle["events"]
            x_dataset = events["x"]
            y_dataset = events["y"]
            t_dataset = events["t_us"]
            p_dataset = events["polarity"]
            index_dataset = handle["index/ms_to_event_idx"]
            index_dtype = index_dataset.dtype
            source_events_read = int(checkpoint["source_events_read"])
            written_events = int(checkpoint["written_events"])
            dropped_events = int(checkpoint["dropped_events"])
            index_entries = int(checkpoint["index_entries"])
            next_index_boundary_us = int(checkpoint["next_index_boundary_us"])
            previous_value = checkpoint["previous_timestamp_us"]
            previous_timestamp = None if previous_value is None else int(previous_value)
            seen_polarities = {int(value) for value in checkpoint["seen_polarities"]}
            minimum_event_length = min(
                map(len, (x_dataset, y_dataset, t_dataset, p_dataset))
            )
            if not (
                0 <= source_events_read <= metadata.event_count
                and 0 <= written_events <= minimum_event_length
                and 0 <= index_entries <= len(index_dataset)
            ):
                raise ValueError("partial checkpoint lengths are invalid")
            for dataset in (x_dataset, y_dataset, t_dataset, p_dataset):
                dataset.resize((written_events,))
            index_dataset.resize((index_entries,))
            handle.attrs["complete"] = False
            handle.attrs["event_count"] = written_events
            handle.attrs["dropped_event_count"] = dropped_events

            for chunk in source.iter_event_chunks(
                options.read_chunk_events, start_event=source_events_read
            ):
                arrays, previous_timestamp = _validate_source_chunk(
                    chunk, previous_timestamp
                )
                if not len(arrays["x"]):
                    continue
                source_events_read += len(arrays["x"])
                seen_polarities.update(
                    int(value) for value in np.unique(arrays["polarity"]).tolist()
                )
                if not seen_polarities.issubset({-1, 0, 1}):
                    raise ValueError("source polarity must use {-1,+1} or {0,1}")
                if -1 in seen_polarities and 0 in seen_polarities:
                    raise ValueError("source sequence mixes polarity encodings")

                relative_timestamps = (
                    arrays["t_us"].astype(np.int64, copy=False)
                    - metadata.first_timestamp_us
                )
                if relative_timestamps[0] < 0 or relative_timestamps[-1] > duration_us:
                    raise ValueError("source timestamps exceed declared metadata bounds")
                x, y, valid = _transform_coordinates(
                    arrays["x"],
                    arrays["y"],
                    source_width=metadata.width,
                    source_height=metadata.height,
                    downsample=options.spatial_downsample,
                )
                dropped_events += int((~valid).sum())
                if not bool(valid.any()):
                    continue
                x = x[valid]
                y = y[valid]
                relative_timestamps = relative_timestamps[valid]
                polarity = (arrays["polarity"][valid] > 0).astype(np.uint8, copy=False)
                if relative_timestamps[-1] > np.iinfo(timestamp_dtype).max:
                    raise ValueError(
                        f"timestamp exceeds {timestamp_dtype}; use --timestamp-dtype uint64"
                    )

                chunk_start = written_events
                _append(x_dataset, x.astype(np.uint16, copy=False))
                _append(y_dataset, y.astype(np.uint16, copy=False))
                _append(t_dataset, relative_timestamps.astype(timestamp_dtype, copy=False))
                _append(p_dataset, polarity)
                written_events += len(relative_timestamps)

                last_timestamp = int(relative_timestamps[-1])
                if next_index_boundary_us <= last_timestamp:
                    entries, next_index_boundary_us = coarse_index_entries(
                        relative_timestamps,
                        next_boundary_us=next_index_boundary_us,
                        step_us=options.index_step_us,
                        global_event_offset=chunk_start,
                    )
                    _append(index_dataset, entries.astype(index_dtype, copy=False))
                index_entries = len(index_dataset)
                checkpoint.update(
                    {
                        "source_events_read": source_events_read,
                        "written_events": written_events,
                        "dropped_events": dropped_events,
                        "index_entries": index_entries,
                        "next_index_boundary_us": next_index_boundary_us,
                        "previous_timestamp_us": previous_timestamp,
                        "seen_polarities": sorted(seen_polarities),
                        "complete": False,
                    }
                )
                handle.attrs["event_count"] = written_events
                handle.attrs["dropped_event_count"] = dropped_events
                handle.flush()
                _write_json_atomic(checkpoint_path, checkpoint)

            if source_events_read != metadata.event_count:
                raise ValueError(
                    "source event count changed while preprocessing: "
                    f"expected {metadata.event_count}, read {source_events_read}"
                )
            if _metadata_identity(metadata, split, config_hash) != identity:
                raise ValueError("source file metadata changed while preprocessing")
            if written_events == 0:
                raise ValueError("all source events were removed during preprocessing")
            terminal_boundary = (duration_us // options.index_step_us + 1) * options.index_step_us
            if next_index_boundary_us <= terminal_boundary:
                remaining = (
                    (terminal_boundary - next_index_boundary_us) // options.index_step_us + 1
                )
                _append(index_dataset, np.full(remaining, written_events, dtype=index_dtype))
                next_index_boundary_us = terminal_boundary + options.index_step_us

            handle.attrs["event_count"] = written_events
            handle.attrs["dropped_event_count"] = dropped_events
            handle.attrs["complete"] = True
            handle.flush()
            _validate_compressed_structure(handle)
            checkpoint.update(
                {
                    "source_events_read": source_events_read,
                    "written_events": written_events,
                    "dropped_events": dropped_events,
                    "index_entries": len(index_dataset),
                    "next_index_boundary_us": next_index_boundary_us,
                    "previous_timestamp_us": previous_timestamp,
                    "seen_polarities": sorted(seen_polarities),
                    "complete": True,
                }
            )
            _write_json_atomic(checkpoint_path, checkpoint)

        with temporary.open("r+b") as file_handle:
            os.fsync(file_handle.fileno())
        os.replace(temporary, output)
        checkpoint_path.unlink(missing_ok=True)
        return _existing_record(output, split)
    finally:
        source.close()


def preprocess_sequence(
    source: EventSource,
    output_path: str | Path,
    *,
    split: str,
    options: PreprocessOptions,
) -> dict[str, Any]:
    """Stream one source under a per-output process lock.

    Completed outputs and resumable ``.partial`` files use the same lock, so
    two launchers cannot mutate one HDF5/checkpoint pair concurrently.
    """

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_acquired = False
    try:
        with _exclusive_output_lock(output):
            lock_acquired = True
            return _preprocess_sequence_unlocked(
                source,
                output,
                split=split,
                options=options,
            )
    finally:
        # The unlocked implementation owns the source after lock acquisition.
        # A lock conflict happens before it is entered, so close in that case.
        if not lock_acquired:
            source.close()


_MANIFEST_IDENTITY_FIELDS = (
    "source_recording_id",
    "dataset",
    "split",
    "camera",
    "source_width",
    "source_height",
    "width",
    "height",
    "spatial_downsample",
    "source_time_origin_us",
    "coordinate_frame",
    "timestamp_reference",
)


def _resolved_manifest_records(manifest: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row: dict[str, Any] = json.loads(line)
            if "sequence_id" not in row or "path" not in row:
                raise ValueError(
                    f"{manifest}:{line_number} has no sequence_id/path"
                )
            for path_field in ("path", "bbox_path"):
                if row.get(path_field) is None:
                    continue
                artifact_path = Path(str(row[path_field]))
                if not artifact_path.is_absolute():
                    artifact_path = (manifest.parent / artifact_path).resolve()
                row[path_field] = str(artifact_path)
            records.append(row)
    return records


def _merge_manifest_records(
    existing: list[dict[str, Any]], new: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for record in existing:
        sequence_id = str(record["sequence_id"])
        if sequence_id in merged:
            raise ValueError(f"existing manifest has duplicate {sequence_id}")
        merged[sequence_id] = record
    for record in new:
        sequence_id = str(record["sequence_id"])
        previous = merged.get(sequence_id)
        if previous is not None:
            if Path(str(previous["path"])).resolve() != Path(
                str(record["path"])
            ).resolve():
                raise ValueError(
                    f"manifest sequence {sequence_id} points to a different event file"
                )
            for field in _MANIFEST_IDENTITY_FIELDS:
                if (
                    field in previous
                    and field in record
                    and previous[field] != record[field]
                ):
                    raise ValueError(
                        f"manifest sequence {sequence_id} changes identity field {field}"
                    )
            # Preserve optional artifacts from an older row when a label-free
            # rerun validates only the event file.
            merged[sequence_id] = {**previous, **record}
        else:
            merged[sequence_id] = record
    return list(merged.values())


def _validate_manifest_artifacts(records: list[Mapping[str, Any]]) -> None:
    for record in records:
        for path_field in ("path", "bbox_path"):
            if record.get(path_field) is None:
                continue
            artifact = Path(str(record[path_field])).resolve()
            if not artifact.is_file():
                raise FileNotFoundError(
                    f"manifest artifact does not exist for "
                    f"{record['sequence_id']}: {artifact}"
                )


def _write_manifest_unlocked(
    records: list[Mapping[str, Any]], manifest: Path
) -> None:
    sequence_ids = [str(record["sequence_id"]) for record in records]
    if len(sequence_ids) != len(set(sequence_ids)):
        raise ValueError("manifest would contain duplicate sequence ids")
    recording_splits: dict[str, set[str]] = {}
    for record in records:
        recording_id = str(record.get("source_recording_id", record["sequence_id"]))
        recording_splits.setdefault(recording_id, set()).add(str(record["split"]))
    leaked_recordings = sorted(
        recording_id
        for recording_id, splits in recording_splits.items()
        if len(splits) > 1
    )
    if leaked_recordings:
        raise ValueError(
            "source recordings cannot cross logical splits: "
            f"{leaked_recordings}"
        )
    temporary = manifest.with_name(f".{manifest.name}.{uuid.uuid4().hex}.partial")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in sorted(records, key=lambda item: str(item["sequence_id"])):
                row = dict(record)
                for path_field in ("path", "bbox_path"):
                    if row.get(path_field) is None:
                        continue
                    artifact_path = Path(str(row[path_field])).resolve()
                    row[path_field] = os.path.relpath(artifact_path, manifest.parent)
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, manifest)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def write_manifest(
    records: list[Mapping[str, Any]],
    path: str | Path,
    *,
    merge_existing: bool = False,
) -> None:
    manifest = Path(path).expanduser().resolve()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    new_records = [dict(record) for record in records]
    with _exclusive_output_lock(manifest):
        output_records = new_records
        if merge_existing and manifest.exists():
            output_records = _merge_manifest_records(
                _resolved_manifest_records(manifest), new_records
            )
        if merge_existing:
            _validate_manifest_artifacts(output_records)
        _write_manifest_unlocked(output_records, manifest)
