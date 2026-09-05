from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

from event_window_jepa.preprocessing.common import EventSourceMetadata


def _require_hdf5() -> Any:
    try:
        # Importing registers both the Blosc-ZSTD filter used by DSEC and the
        # standalone Zstd filter used by our canonical output.
        import hdf5plugin  # noqa: F401
        import h5py
    except ImportError as error:
        raise ImportError("install event-window-jepa[hdf5] to read source HDF5") from error
    return h5py


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    if not cleaned:
        raise ValueError("sequence name becomes empty after sanitization")
    return cleaned


def sequence_identifier(dataset: str, source_name: str, camera: str) -> str:
    return "__".join(
        (_safe_component(dataset), _safe_component(source_name), _safe_component(camera))
    )


def _validate_h5_event_fields(group: Any) -> int:
    names = ("x", "y", "t", "p")
    missing = [name for name in names if name not in group]
    if missing:
        raise KeyError(f"source HDF5 event group is missing {missing}")
    lengths: set[int] = set()
    for name in names:
        dataset = group[name]
        if dataset.ndim != 1:
            raise ValueError(f"source HDF5 event field {name} must be one-dimensional")
        is_integer = np.issubdtype(dataset.dtype, np.integer)
        if name == "p":
            is_integer = is_integer or np.issubdtype(dataset.dtype, np.bool_)
        if not is_integer:
            raise TypeError(f"source HDF5 event field {name} must be integer")
        lengths.add(len(dataset))
    if len(lengths) != 1:
        raise ValueError("source HDF5 event fields do not have the same length")
    size = lengths.pop()
    if size <= 0:
        raise ValueError("source HDF5 event stream is empty")
    return size


def _validate_native_event_index(dataset: Any, event_count: int, name: str) -> None:
    if dataset.ndim != 1 or not np.issubdtype(dataset.dtype, np.integer):
        raise TypeError(f"{name} must be a one-dimensional integer dataset")
    if len(dataset) == 0:
        raise ValueError(f"{name} cannot be empty")
    previous: int | None = None
    for start in range(0, len(dataset), 1_000_000):
        values = np.asarray(dataset[start : min(start + 1_000_000, len(dataset))])
        if (
            np.any(values < 0)
            or np.any(values > event_count)
            or np.any(values[1:] < values[:-1])
        ):
            raise ValueError(f"{name} is not a valid monotonic event index")
        if previous is not None and int(values[0]) < previous:
            raise ValueError(f"{name} is not a valid monotonic event index")
        previous = int(values[-1])


class DSECEventSource:
    """Streaming adapter for an official DSEC camera-side events.h5 file."""

    def __init__(
        self,
        path: str | Path,
        *,
        camera: str = "left",
        sequence_name: str | None = None,
    ) -> None:
        if camera not in {"left", "right"}:
            raise ValueError("DSEC camera must be left or right")
        h5py = _require_hdf5()
        self.path = Path(path).expanduser().resolve()
        if self.path.parent.name in {"left", "right"} and self.path.parent.name != camera:
            raise ValueError(
                f"DSEC camera={camera!r} disagrees with source path {self.path.parent.name!r}"
            )
        self._handle = h5py.File(self.path, "r")
        try:
            self._events = self._handle["events"]
            event_count = _validate_h5_event_fields(self._events)
            if "t_offset" not in self._handle:
                raise KeyError("official DSEC events.h5 must contain /t_offset")
            if "ms_to_idx" not in self._handle:
                raise KeyError("official DSEC events.h5 must contain /ms_to_idx")
            _validate_native_event_index(
                self._handle["ms_to_idx"], event_count, "/ms_to_idx"
            )
            offset_values = np.asarray(self._handle["t_offset"][()])
            if offset_values.size != 1 or not np.issubdtype(
                offset_values.dtype, np.integer
            ):
                raise TypeError("DSEC /t_offset must contain one integer microsecond value")
            t_offset = int(offset_values.reshape(-1)[0])
            first = int(self._events["t"][0]) + t_offset
            last = int(self._events["t"][-1]) + t_offset
        except BaseException:
            self._handle.close()
            raise
        if sequence_name is None:
            if self.path.parent.name in {"left", "right"}:
                sequence_name = self.path.parents[2].name
            else:
                sequence_name = self.path.stem
        self._t_offset = t_offset
        self.metadata = EventSourceMetadata(
            sequence_id=sequence_identifier("dsec", sequence_name, camera),
            dataset="dsec",
            source_path=self.path,
            camera=camera,
            width=640,
            height=480,
            event_count=event_count,
            first_timestamp_us=first,
            last_timestamp_us=last,
            coordinate_frame="distorted",
            attributes={
                "timestamp_offset_us": t_offset,
                "timestamp_reference": "events/t + t_offset",
                "timestamp_synchronized": True,
            },
        )

    def iter_event_chunks(
        self, chunk_events: int, start_event: int = 0
    ) -> Iterator[Mapping[str, np.ndarray]]:
        if not 0 <= start_event <= self.metadata.event_count:
            raise ValueError("start_event is outside the source event stream")
        for start in range(start_event, self.metadata.event_count, chunk_events):
            stop = min(start + chunk_events, self.metadata.event_count)
            timestamps = np.asarray(self._events["t"][start:stop], dtype=np.int64)
            timestamps += self._t_offset
            yield {
                "x": np.asarray(self._events["x"][start:stop]),
                "y": np.asarray(self._events["y"][start:stop]),
                "t_us": timestamps,
                "polarity": np.asarray(self._events["p"][start:stop]),
            }

    def close(self) -> None:
        if self._handle:
            self._handle.close()
            self._handle = None


class M3EDEventSource:
    """Streaming adapter for one Prophesee camera in an M3ED processed HDF5."""

    def __init__(
        self,
        path: str | Path,
        *,
        camera: str = "left",
        sequence_name: str | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        if camera not in {"left", "right"}:
            raise ValueError("M3ED camera must be left or right")
        if (width is None) != (height is None):
            raise ValueError("M3ED width and height overrides must be provided together")
        h5py = _require_hdf5()
        self.path = Path(path).expanduser().resolve()
        self._handle = h5py.File(self.path, "r")
        try:
            self._events = self._handle[f"prophesee/{camera}"]
            event_count = _validate_h5_event_fields(self._events)
            if "ms_map_idx" not in self._events:
                raise KeyError(
                    f"official M3ED event group /prophesee/{camera} has no ms_map_idx"
                )
            _validate_native_event_index(
                self._events["ms_map_idx"],
                event_count,
                f"/prophesee/{camera}/ms_map_idx",
            )
            resolution_path = f"prophesee/{camera}/calib/resolution"
            if resolution_path in self._handle:
                resolution = np.asarray(self._handle[resolution_path]).reshape(-1)
                if resolution.size != 2 or not np.issubdtype(
                    resolution.dtype, np.integer
                ):
                    raise TypeError(
                        "M3ED calibration resolution must contain integer width and height"
                    )
                inferred_width, inferred_height = int(resolution[0]), int(resolution[1])
                if width is not None and (width, height) != (
                    inferred_width,
                    inferred_height,
                ):
                    raise ValueError(
                        "M3ED resolution override disagrees with official calibration"
                    )
                width, height = inferred_width, inferred_height
            else:
                width = 1280 if width is None else width
                height = 720 if height is None else height
            first = int(self._events["t"][0])
            last = int(self._events["t"][-1])
        except BaseException:
            self._handle.close()
            raise
        if sequence_name is None:
            sequence_name = self.path.stem.removesuffix("_data")
        self.metadata = EventSourceMetadata(
            sequence_id=sequence_identifier("m3ed", sequence_name, camera),
            dataset="m3ed",
            source_path=self.path,
            camera=camera,
            width=width,
            height=height,
            event_count=event_count,
            first_timestamp_us=first,
            last_timestamp_us=last,
            coordinate_frame="distorted",
            attributes={
                "timestamp_reference": "M3ED synchronized global clock (microseconds)",
                "timestamp_synchronized": True,
                "native_ms_index": f"/prophesee/{camera}/ms_map_idx",
            },
        )

    def iter_event_chunks(
        self, chunk_events: int, start_event: int = 0
    ) -> Iterator[Mapping[str, np.ndarray]]:
        if not 0 <= start_event <= self.metadata.event_count:
            raise ValueError("start_event is outside the source event stream")
        for start in range(start_event, self.metadata.event_count, chunk_events):
            stop = min(start + chunk_events, self.metadata.event_count)
            yield {
                "x": np.asarray(self._events["x"][start:stop]),
                "y": np.asarray(self._events["y"][start:stop]),
                "t_us": np.asarray(self._events["t"][start:stop], dtype=np.int64),
                "polarity": np.asarray(self._events["p"][start:stop]),
            }

    def close(self) -> None:
        if self._handle:
            self._handle.close()
            self._handle = None


class MVSECEventSource:
    """Read one camera from an official MVSEC ``*_data.hdf5`` file.

    MVSEC stores events as an ``[N, 4]`` floating-point matrix containing
    ``x, y, timestamp_seconds, polarity``.  The source file is kept strictly
    read-only and timestamps are rounded once onto the repository-wide integer
    microsecond clock.  Left and right cameras therefore retain their common
    recording clock; each canonical output records its own first timestamp as
    ``source_time_origin_us``.
    """

    _WIDTH = 346
    _HEIGHT = 260

    def __init__(
        self,
        path: str | Path,
        *,
        camera: str = "left",
        sequence_name: str | None = None,
    ) -> None:
        if camera not in {"left", "right"}:
            raise ValueError("MVSEC camera must be left or right")
        h5py = _require_hdf5()
        self.path = Path(path).expanduser().resolve()
        self._handle = h5py.File(self.path, "r")
        try:
            event_path = f"davis/{camera}/events"
            if event_path not in self._handle:
                raise KeyError(f"official MVSEC data has no /{event_path}")
            self._events = self._handle[event_path]
            if self._events.ndim != 2 or self._events.shape[1] != 4:
                raise ValueError(
                    f"MVSEC /{event_path} must have shape [N,4] (x,y,t_seconds,p)"
                )
            if not np.issubdtype(self._events.dtype, np.number):
                raise TypeError(f"MVSEC /{event_path} must be numeric")
            event_count = len(self._events)
            if event_count <= 0:
                raise ValueError("MVSEC event stream is empty")
            endpoints = np.asarray(self._events[[0, event_count - 1], 2], dtype=np.float64)
            if not bool(np.isfinite(endpoints).all()) or endpoints[1] < endpoints[0]:
                raise ValueError("MVSEC event timestamps are not finite and monotonic")
            first, last = np.rint(endpoints * 1_000_000.0).astype(np.int64).tolist()
        except BaseException:
            self._handle.close()
            raise

        if sequence_name is None:
            name = self.path.name
            if name.endswith("_data.hdf5"):
                sequence_name = name.removesuffix("_data.hdf5")
            elif name.endswith("_data.h5"):
                sequence_name = name.removesuffix("_data.h5")
            else:
                sequence_name = self.path.stem.removesuffix("_data")
        self.metadata = EventSourceMetadata(
            sequence_id=sequence_identifier("mvsec", sequence_name, camera),
            dataset="mvsec",
            source_path=self.path,
            camera=camera,
            width=self._WIDTH,
            height=self._HEIGHT,
            event_count=event_count,
            first_timestamp_us=int(first),
            last_timestamp_us=int(last),
            coordinate_frame="distorted",
            attributes={
                "timestamp_reference": "MVSEC synchronized recording clock (seconds)",
                "timestamp_synchronized": True,
                "source_format": "official MVSEC Nx4 DAVIS events",
                "source_events_dataset": f"/{event_path}",
                "timestamp_conversion": "round(seconds * 1e6)",
            },
        )

    def iter_event_chunks(
        self, chunk_events: int, start_event: int = 0
    ) -> Iterator[Mapping[str, np.ndarray]]:
        if chunk_events <= 0:
            raise ValueError("chunk_events must be positive")
        if not 0 <= start_event <= self.metadata.event_count:
            raise ValueError("start_event is outside the source event stream")
        previous_timestamp: float | None = None
        seen_polarities: set[int] = set()
        for start in range(start_event, self.metadata.event_count, chunk_events):
            stop = min(start + chunk_events, self.metadata.event_count)
            values = np.asarray(self._events[start:stop], dtype=np.float64)
            if not bool(np.isfinite(values).all()):
                raise ValueError("MVSEC events contain NaN or infinity")
            x_raw, y_raw, timestamps_seconds, polarity_raw = values.T
            if (
                np.any(x_raw != np.rint(x_raw))
                or np.any(y_raw != np.rint(y_raw))
                or np.any(polarity_raw != np.rint(polarity_raw))
            ):
                raise ValueError("MVSEC x, y, and polarity values must be integral")
            if (
                np.any(x_raw < 0)
                or np.any(x_raw >= self._WIDTH)
                or np.any(y_raw < 0)
                or np.any(y_raw >= self._HEIGHT)
            ):
                raise ValueError("MVSEC event coordinates exceed the 346x260 sensor")
            seen_polarities.update(
                int(value) for value in np.unique(polarity_raw).tolist()
            )
            if not seen_polarities.issubset({-1, 0, 1}):
                raise ValueError("MVSEC polarity must use {-1,+1} or {0,1}")
            if -1 in seen_polarities and 0 in seen_polarities:
                raise ValueError("MVSEC event stream mixes polarity encodings")
            if (
                np.any(timestamps_seconds[1:] < timestamps_seconds[:-1])
                or (
                    previous_timestamp is not None
                    and len(timestamps_seconds)
                    and timestamps_seconds[0] < previous_timestamp
                )
            ):
                raise ValueError("MVSEC event timestamps decrease")
            if len(timestamps_seconds):
                previous_timestamp = float(timestamps_seconds[-1])
            timestamps_us = np.rint(timestamps_seconds * 1_000_000.0)
            if np.any(np.abs(timestamps_us) > np.iinfo(np.int64).max):
                raise OverflowError("MVSEC timestamps do not fit signed microseconds")
            yield {
                "x": np.rint(x_raw).astype(np.int64),
                "y": np.rint(y_raw).astype(np.int64),
                "t_us": timestamps_us.astype(np.int64),
                "polarity": np.rint(polarity_raw).astype(np.int8),
            }

    def close(self) -> None:
        if self._handle:
            self._handle.close()
            self._handle = None


class RVTGenXH5EventSource:
    """Streaming adapter for RVT's original-event Gen1/Gen4 HDF5 files."""

    def __init__(
        self,
        path: str | Path,
        *,
        dataset: str,
        camera: str = "left",
        sequence_name: str | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        if dataset not in {"gen1", "gen4"}:
            raise ValueError("RVT HDF5 dataset must be gen1 or gen4")
        if camera != "left":
            raise ValueError("RVT Gen1/Gen4 detection recordings are monocular")
        if (width is None) != (height is None):
            raise ValueError("RVT HDF5 width and height overrides must be paired")

        h5py = _require_hdf5()
        self.path = Path(path).expanduser().resolve()
        self._handle = h5py.File(self.path, "r")
        try:
            self._events = self._handle["events"]
            event_count = _validate_h5_event_fields(self._events)
            defaults = (304, 240) if dataset == "gen1" else (1280, 720)
            has_width = "width" in self._events
            has_height = "height" in self._events
            if has_width != has_height:
                raise ValueError("RVT /events has only one of width and height")
            if has_width:
                width_value = np.asarray(self._events["width"][()])
                height_value = np.asarray(self._events["height"][()])
                if (
                    width_value.size != 1
                    or height_value.size != 1
                    or not np.issubdtype(width_value.dtype, np.integer)
                    or not np.issubdtype(height_value.dtype, np.integer)
                ):
                    raise TypeError("RVT event width/height must be integer scalars")
                inferred = (
                    int(width_value.reshape(-1)[0]),
                    int(height_value.reshape(-1)[0]),
                )
            else:
                inferred = defaults
            if inferred != defaults:
                raise ValueError(
                    f"RVT {dataset} resolution must be {defaults[0]}x{defaults[1]}, "
                    f"got {inferred[0]}x{inferred[1]}"
                )
            if width is not None and (width, height) != inferred:
                raise ValueError("resolution override disagrees with RVT HDF5 metadata")
            width, height = inferred
            first = int(self._events["t"][0])
            last = int(self._events["t"][-1])
        except BaseException:
            self._handle.close()
            raise

        if sequence_name is None:
            filename = self.path.name
            suffix = "_td.dat.h5" if dataset == "gen1" else "_td.h5"
            sequence_name = filename.removesuffix(suffix)
        self.metadata = EventSourceMetadata(
            sequence_id=sequence_identifier(dataset, sequence_name, camera),
            dataset=dataset,
            source_path=self.path,
            camera=camera,
            width=width,
            height=height,
            event_count=event_count,
            first_timestamp_us=first,
            last_timestamp_us=last,
            coordinate_frame="distorted",
            attributes={
                "timestamp_reference": "RVT original-event HDF5 recording clock",
                "timestamp_synchronized": True,
                "source_format": "RVT GenX original-event HDF5",
                "source_events_group": "/events",
                "timestamp_repair_policy": "running_max",
                "timestamp_repair_scope": "RVT Gen1/Gen4 source only",
                "coordinate_repair_policy": "drop_out_of_bounds",
                "coordinate_repair_scope": "RVT Gen1/Gen4 source only",
            },
        )

    def iter_event_chunks(
        self, chunk_events: int, start_event: int = 0
    ) -> Iterator[Mapping[str, np.ndarray]]:
        if not 0 <= start_event <= self.metadata.event_count:
            raise ValueError("start_event is outside the source event stream")
        for start in range(start_event, self.metadata.event_count, chunk_events):
            stop = min(start + chunk_events, self.metadata.event_count)
            arrays: dict[str, np.ndarray] = {}
            for output_name, source_name, dtype in (
                ("x", "x", None),
                ("y", "y", None),
                ("t_us", "t", np.int64),
                ("polarity", "p", None),
            ):
                try:
                    arrays[output_name] = np.asarray(
                        self._events[source_name][start:stop], dtype=dtype
                    )
                except OSError as error:
                    raise OSError(
                        "failed to read RVT source HDF5 chunk: "
                        f"path={self.path}, field=/events/{source_name}, "
                        f"events=[{start},{stop})"
                    ) from error
            yield arrays

    def close(self) -> None:
        if self._handle:
            self._handle.close()
            self._handle = None


def _parse_dat_header(handle: Any) -> tuple[int, int, int, int, int]:
    handle.seek(0, os.SEEK_SET)
    width: int | None = None
    height: int | None = None
    comment_lines = 0
    while True:
        position = handle.tell()
        prefix = handle.read(2)
        if prefix != b"% ":
            handle.seek(position, os.SEEK_SET)
            break
        line = prefix + handle.readline()
        comment_lines += 1
        words = line.decode("latin-1", errors="replace").strip().split()
        if len(words) >= 3 and words[1].lower() == "height":
            height = int(words[2])
        if len(words) >= 3 and words[1].lower() == "width":
            width = int(words[2])
    if comment_lines:
        type_bytes = handle.read(1)
        size_bytes = handle.read(1)
        if len(type_bytes) != 1 or len(size_bytes) != 1:
            raise ValueError("truncated Prophesee DAT header")
        event_type = type_bytes[0]
        event_size = size_bytes[0]
    else:
        event_type = 0
        event_size = 8
    if event_type != 0 or event_size != 8:
        raise ValueError(
            f"only 8-byte Event2D DAT records are supported, got type={event_type}, "
            f"size={event_size}"
        )
    return handle.tell(), event_type, event_size, height or 0, width or 0


class PropheseeDatEventSource:
    """Chunked Event2D DAT adapter for Gen1 and 1 Megapixel recordings."""

    _dtype = np.dtype([("t", "<u4"), ("packed", "<u4")])
    _wrap = 1 << 32

    def __init__(
        self,
        path: str | Path,
        *,
        dataset: str = "prophesee_1mpx",
        camera: str = "left",
        sequence_name: str | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        if dataset in {"prophesee_1mpx", "gen1"} and camera != "left":
            raise ValueError("official Prophesee 1Mpx and Gen1 recordings are monocular")
        if (width is None) != (height is None):
            raise ValueError("DAT width and height overrides must be provided together")
        self.path = Path(path).expanduser().resolve()
        self._handle = self.path.open("rb")
        start, _, event_size, header_height, header_width = _parse_dat_header(self._handle)
        file_size = self.path.stat().st_size
        payload_size = file_size - start
        if payload_size <= 0 or payload_size % event_size:
            self._handle.close()
            raise ValueError("Prophesee DAT payload is empty or truncated")
        self._data_start = start
        event_count = payload_size // event_size
        self._handle.seek(start)
        first_record = np.fromfile(self._handle, dtype=self._dtype, count=1)
        self._handle.seek(start + (event_count - 1) * event_size)
        last_record = np.fromfile(self._handle, dtype=self._dtype, count=1)
        first_raw = int(first_record["t"][0])
        last_raw = int(last_record["t"][0])
        last_unwrapped = last_raw + (self._wrap if last_raw < first_raw else 0)
        if bool(header_width) != bool(header_height):
            self._handle.close()
            raise ValueError("Prophesee DAT header contains an incomplete resolution")
        if header_width and header_height:
            if width is not None and (width, height) != (header_width, header_height):
                self._handle.close()
                raise ValueError(
                    "DAT resolution override disagrees with the recording header"
                )
            width, height = header_width, header_height
        else:
            if width is None:
                width = 1280 if dataset == "prophesee_1mpx" else 304
                height = 720 if dataset == "prophesee_1mpx" else 240
        if sequence_name is None:
            sequence_name = self.path.stem
        self.metadata = EventSourceMetadata(
            sequence_id=sequence_identifier(dataset, sequence_name, camera),
            dataset=dataset,
            source_path=self.path,
            camera=camera,
            width=width,
            height=height,
            event_count=event_count,
            first_timestamp_us=first_raw,
            last_timestamp_us=last_unwrapped,
            coordinate_frame="distorted",
            attributes={
                "timestamp_reference": "DAT event/annotation recording clock",
                "timestamp_synchronized": True,
                "dat_event_type": 0,
                "dat_event_size": event_size,
            },
        )

    def iter_event_chunks(
        self, chunk_events: int, start_event: int = 0
    ) -> Iterator[Mapping[str, np.ndarray]]:
        if not 0 <= start_event <= self.metadata.event_count:
            raise ValueError("start_event is outside the source event stream")
        self._handle.seek(self._data_start + start_event * self._dtype.itemsize)
        previous_raw: int | None = None
        wrap_count = 0
        if start_event:
            first_at_resume = np.fromfile(self._handle, dtype=self._dtype, count=1)
            if not len(first_at_resume):
                raise ValueError("resume offset exceeds the Prophesee DAT payload")
            wrap_count = int(
                int(first_at_resume["t"][0]) < self.metadata.first_timestamp_us
            )
            self._handle.seek(self._data_start + start_event * self._dtype.itemsize)
        remaining = self.metadata.event_count - start_event
        while remaining:
            records = np.fromfile(
                self._handle, dtype=self._dtype, count=min(chunk_events, remaining)
            )
            if not len(records):
                raise ValueError("unexpected end of Prophesee DAT event stream")
            raw_timestamps = records["t"].astype(np.uint64)
            initial_wrap = int(previous_raw is not None and int(raw_timestamps[0]) < previous_raw)
            internal_wraps = np.concatenate(
                (
                    np.zeros(1, dtype=np.uint64),
                    np.cumsum(raw_timestamps[1:] < raw_timestamps[:-1], dtype=np.uint64),
                )
            )
            internal_wraps += wrap_count + initial_wrap
            timestamps = raw_timestamps + internal_wraps * self._wrap
            wrap_count = int(internal_wraps[-1])
            previous_raw = int(raw_timestamps[-1])
            packed = records["packed"]
            yield {
                "x": np.bitwise_and(packed, 0x3FFF).astype(np.uint16),
                "y": np.right_shift(np.bitwise_and(packed, 0x0FFFC000), 14).astype(
                    np.uint16
                ),
                "t_us": timestamps.astype(np.int64),
                "polarity": np.right_shift(
                    np.bitwise_and(packed, 0x10000000), 28
                ).astype(np.uint8),
            }
            remaining -= len(records)

    def close(self) -> None:
        if self._handle:
            self._handle.close()
            self._handle = None


def discover_sequence_paths(dataset: str, input_path: str | Path, camera: str) -> list[Path]:
    root = Path(input_path).expanduser().resolve()
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(root)
    if dataset == "dsec":
        paths = sorted(root.glob(f"**/events/{camera}/events.h5"))
        if not paths:
            paths = sorted(root.glob("**/events.h5"))
    elif dataset == "m3ed":
        h5py = _require_hdf5()
        paths = []
        for candidate in sorted(root.glob("**/*.h5")):
            try:
                with h5py.File(candidate, "r") as handle:
                    if f"prophesee/{camera}/t" in handle:
                        paths.append(candidate)
            except OSError:
                continue
    elif dataset == "mvsec":
        paths = sorted(root.glob("**/*_data.hdf5"))
        paths.extend(sorted(root.glob("**/*_data.h5")))
        paths = sorted(set(paths))
    elif dataset == "gen4":
        paths = sorted(root.glob("**/*_td.h5"))
    elif dataset == "prophesee_1mpx":
        paths = sorted(root.glob("**/*.dat"))
    elif dataset == "gen1":
        paths = sorted(root.glob("**/*.dat"))
        paths.extend(sorted(root.glob("**/*_td.dat.h5")))
        paths = sorted(set(paths))
    else:
        raise ValueError(f"unsupported source dataset: {dataset}")
    if root.is_dir() and dataset in {"gen1", "gen4"}:
        paths = [
            path
            for path in paths
            if "_excluded_failed_validation" not in path.relative_to(root).parts
        ]
    if not paths:
        raise ValueError(f"no {dataset} event sequences found below {root}")
    return paths


def make_event_source(
    dataset: str,
    path: str | Path,
    *,
    camera: str,
    width: int | None = None,
    height: int | None = None,
) -> (
    DSECEventSource
    | M3EDEventSource
    | MVSECEventSource
    | RVTGenXH5EventSource
    | PropheseeDatEventSource
):
    if dataset == "dsec":
        if width is not None or height is not None:
            raise ValueError("DSEC resolution is fixed at 640x480")
        return DSECEventSource(path, camera=camera)
    if dataset == "m3ed":
        return M3EDEventSource(
            path, camera=camera, width=width, height=height
        )
    if dataset == "mvsec":
        if width is not None or height is not None:
            raise ValueError("MVSEC resolution is fixed at 346x260")
        return MVSECEventSource(path, camera=camera)
    source_path = Path(path)
    if dataset == "gen4" or (dataset == "gen1" and source_path.suffix == ".h5"):
        return RVTGenXH5EventSource(
            path,
            dataset=dataset,
            camera=camera,
            width=width,
            height=height,
        )
    if dataset in {"prophesee_1mpx", "gen1"}:
        return PropheseeDatEventSource(
            path,
            dataset=dataset,
            camera=camera,
            width=width,
            height=height,
        )
    raise ValueError(f"unsupported source dataset: {dataset}")
