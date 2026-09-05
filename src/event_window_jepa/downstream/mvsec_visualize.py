from __future__ import annotations

import argparse
import binascii
import hashlib
import html
import json
import math
import os
import re
import struct
import tempfile
import zipfile
import zlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SNAPSHOT_SCHEMA = "event-window-jepa-mvsec-visualization-sample-v1"
INDEX_SCHEMA = "event-window-jepa-mvsec-visualization-index-v1"
REPORT_SCHEMA = "event-window-jepa-mvsec-visualization-report-v1"
COMPARISON_SCHEMA = "event-window-jepa-mvsec-comparison-v1"
DEFAULT_MAX_SNAPSHOT_MIB = 256
_COLORS = np.asarray(
    [
        (36, 99, 235),
        (239, 68, 68),
        (16, 185, 129),
        (245, 158, 11),
        (139, 92, 246),
        (8, 145, 178),
        (236, 72, 153),
        (101, 116, 139),
    ],
    dtype=np.uint8,
)
_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")
_SEED_LABEL = re.compile(r"^(?P<condition>.+)__seed(?P<seed>[0-9]+)$")
_HIERARCHICAL_SEED_LABEL = re.compile(
    r"^(?P<condition>.+)__encoder_seed(?P<encoder>[0-9]+)"
    r"__probe_seed(?P<probe>[0-9]+)$"
)


def safe_component(value: str, *, fallback: str = "sample") -> str:
    cleaned = _SAFE_COMPONENT.sub("-", value.strip()).strip(".-")
    return cleaned[:96] or fallback


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".partial",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _atomic_text(path: Path, value: str) -> None:
    _atomic_bytes(path, value.encode("utf-8"))


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    serialized = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    _atomic_text(path, serialized + "\n")


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = binascii.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def png_bytes(image: np.ndarray) -> bytes:
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("PNG input must be RGB uint8 [H,W,3]")
    height, width, _ = image.shape
    if height <= 0 or width <= 0:
        raise ValueError("PNG dimensions must be positive")
    scanlines = b"".join(
        b"\x00" + row.tobytes() for row in np.ascontiguousarray(image)
    )
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0),
        )
        + _png_chunk(b"IDAT", zlib.compress(scanlines, level=6))
        + _png_chunk(b"IEND", b"")
    )


def write_png(path: str | Path, image: np.ndarray) -> None:
    _atomic_bytes(Path(path), png_bytes(image))


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o600 << 16
    return info


def _write_npz_deterministic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write a deterministic, compressed NPZ without building a ZIP in RAM."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".partial",
            delete=False,
        ) as raw:
            temporary_name = raw.name
            with zipfile.ZipFile(raw, "w", allowZip64=True) as archive:
                for name in sorted(arrays):
                    value = np.asarray(arrays[name])
                    if value.dtype.hasobject:
                        raise TypeError(f"snapshot array {name!r} cannot contain objects")
                    with archive.open(_zip_info(f"{name}.npy"), "w", force_zip64=True) as member:
                        np.lib.format.write_array(
                            member,
                            value if value.ndim == 0 else np.ascontiguousarray(value),
                            allow_pickle=False,
                        )
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _finite_array(value: Any, name: str, *, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be numeric")
    if not bool(np.isfinite(array).all()):
        raise ValueError(f"{name} contains NaN or infinity")
    return array


def _validate_snapshot_arrays(
    kind: str,
    arrays: Mapping[str, np.ndarray],
) -> tuple[int, int]:
    if kind not in {"flow", "depth"}:
        raise ValueError("snapshot kind must be flow or depth")
    required = {"event_image", "target", "prediction", "valid"}
    missing = required - set(arrays)
    if missing:
        raise KeyError(f"snapshot is missing arrays {sorted(missing)}")
    target = _finite_array(arrays["target"], "target")
    prediction = _finite_array(arrays["prediction"], "prediction")
    valid = np.asarray(arrays["valid"])
    if valid.ndim != 2:
        raise ValueError("valid must have shape [H,W]")
    if valid.dtype != np.bool_:
        raise TypeError("valid must use boolean dtype")
    if kind == "flow":
        if target.ndim != 3 or target.shape[0] != 2:
            raise ValueError("flow target must have shape [2,H,W]")
        if prediction.shape != target.shape:
            raise ValueError("flow prediction must match target [2,H,W]")
        height, width = target.shape[1:]
    else:
        if target.ndim != 2 or prediction.shape != target.shape:
            raise ValueError("depth target and prediction must share [H,W]")
        height, width = target.shape
    if valid.shape != (height, width):
        raise ValueError("valid mask does not match target spatial dimensions")
    if kind == "depth" and (
        bool(np.any(target[valid] <= 0)) or bool(np.any(prediction[valid] <= 0))
    ):
        raise ValueError("valid metric-depth target and prediction must be positive")
    event_image = _finite_array(arrays["event_image"], "event_image")
    if event_image.ndim == 2:
        event_shape = event_image.shape
    elif event_image.ndim == 3:
        if event_image.shape[0] <= 0:
            raise ValueError("event_image must have at least one channel")
        event_shape = event_image.shape[1:]
    else:
        raise ValueError("event_image must have shape [H,W] or [C,H,W]")
    if event_shape != (height, width):
        raise ValueError("event_image and target spatial dimensions differ")
    event_keys = {"event_x", "event_y", "event_t_us", "event_polarity"}
    present = event_keys & set(arrays)
    if present and present != event_keys:
        raise KeyError("raw events must provide x, y, t_us, and polarity together")
    if present:
        event_values = [np.asarray(arrays[name]) for name in sorted(event_keys)]
        if any(value.ndim != 1 for value in event_values):
            raise ValueError("raw event fields must be one-dimensional")
        if len({len(value) for value in event_values}) != 1:
            raise ValueError("raw event fields must have identical lengths")
        x = np.asarray(arrays["event_x"])
        y = np.asarray(arrays["event_y"])
        t_us = np.asarray(arrays["event_t_us"])
        polarity = np.asarray(arrays["event_polarity"])
        if any(not np.issubdtype(value.dtype, np.integer) for value in (x, y, t_us)):
            raise TypeError("event coordinates and timestamps must use integer dtype")
        if not (
            np.issubdtype(polarity.dtype, np.integer)
            or np.issubdtype(polarity.dtype, np.bool_)
        ):
            raise TypeError("event polarity must use integer or boolean dtype")
        if len(x) and (
            bool(np.any(x < 0))
            or bool(np.any(x >= width))
            or bool(np.any(y < 0))
            or bool(np.any(y >= height))
        ):
            raise ValueError("raw event coordinates fall outside the snapshot image")
        if len(t_us) and bool(np.any(t_us[1:] < t_us[:-1])):
            raise ValueError("raw event timestamps must be sorted")
        polarities = set(np.unique(polarity).tolist())
        if not polarities.issubset({-1, 0, 1}) or (
            -1 in polarities and 0 in polarities
        ):
            raise ValueError("event polarity must consistently use {-1,+1} or {0,1}")
    return height, width


def write_snapshot(
    path: str | Path,
    *,
    kind: str,
    event_image: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    valid: np.ndarray,
    metadata: Mapping[str, Any],
    events: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Write one bounded, pickle-free visualization artifact."""

    destination = Path(path).expanduser().resolve()
    if destination.suffix.lower() != ".npz":
        raise ValueError("MVSEC visualization snapshots must use an .npz suffix")
    record = dict(metadata)
    record.update({"schema": SNAPSHOT_SCHEMA, "kind": kind})
    # Validate JSON before opening the destination.
    metadata_json = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    arrays: dict[str, np.ndarray] = {
        "event_image": np.asarray(event_image, dtype=np.float32),
        "target": np.asarray(target, dtype=np.float32),
        "prediction": np.asarray(prediction, dtype=np.float32),
        "valid": np.asarray(valid, dtype=np.bool_),
        "metadata_json": np.asarray(metadata_json),
    }
    if events is not None:
        required = {"event_x", "event_y", "event_t_us", "event_polarity"}
        if set(events) != required:
            raise KeyError(f"events must contain exactly {sorted(required)}")
        arrays.update({name: np.asarray(value) for name, value in events.items()})
        if kind == "flow" and not {
            "event_window_start_us",
            "event_window_end_us",
        }.issubset(record):
            raise KeyError("flow snapshots with raw events require event window bounds")
    height, width = _validate_snapshot_arrays(kind, arrays)
    _write_npz_deterministic(destination, arrays)
    return {
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": _sha256_file(destination),
        "kind": kind,
        "height": height,
        "width": width,
        "target_index": record.get("target_index"),
        "label_timestamp_us": record.get("label_timestamp_us"),
        "sequence_id": record.get("sequence_id"),
    }


def extract_snapshot_events(
    dataset: Any,
    dataset_index: int,
    *,
    t_end_us: int,
    duration_us: int,
    maximum_events: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read only one requested event interval from an MVSECGeometryDataset."""

    if dataset_index < 0 or dataset_index >= len(dataset.references):
        raise IndexError("visualization dataset index is out of range")
    if duration_us <= 0 or maximum_events <= 0:
        raise ValueError("event duration and maximum_events must be positive")
    reference = dataset.references[dataset_index]
    source = dataset.sources[reference.source_index]
    window = dataset.store.slice(source.sequence_id, int(t_end_us), int(duration_us))
    x = np.asarray(window.x, dtype=np.int64) - int(dataset.crop.x0)
    y = np.asarray(window.y, dtype=np.int64) - int(dataset.crop.y0)
    t_us = np.asarray(window.t_us, dtype=np.int64)
    polarity = np.asarray(window.polarity)
    raw_count = len(x)
    retained = (
        (x >= 0)
        & (x < int(dataset.crop.output_width))
        & (y >= 0)
        & (y < int(dataset.crop.output_height))
    )
    x = x[retained]
    y = y[retained]
    t_us = t_us[retained]
    polarity = polarity[retained]
    transformed_count = len(x)
    if transformed_count > maximum_events:
        selected = np.linspace(
            0,
            transformed_count - 1,
            num=maximum_events,
            dtype=np.int64,
        )
        x = x[selected]
        y = y[selected]
        t_us = t_us[selected]
        polarity = polarity[selected]
    arrays = {
        "event_x": np.ascontiguousarray(x),
        "event_y": np.ascontiguousarray(y),
        "event_t_us": np.ascontiguousarray(t_us),
        "event_polarity": np.ascontiguousarray(polarity),
    }
    metadata = {
        "event_window_start_us": int(t_end_us) - int(duration_us),
        "event_window_end_us": int(t_end_us),
        "event_count_raw": raw_count,
        "event_count_total": transformed_count,
        "event_count_stored": len(x),
        "event_subsampling": (
            "none"
            if transformed_count <= maximum_events
            else "deterministic_uniform_index"
        ),
        "maximum_stored_events": maximum_events,
    }
    return arrays, metadata


def write_snapshot_index(
    output_dir: str | Path,
    entries: Sequence[Mapping[str, Any]],
    *,
    requested: int,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    directory = Path(output_dir).expanduser().resolve()
    indexed_paths = {Path(str(entry["path"])).resolve() for entry in entries}
    unindexed = sorted(
        str(path.resolve())
        for path in directory.glob("*.npz")
        if path.resolve() not in indexed_paths
    )
    payload = {
        "schema": INDEX_SCHEMA,
        "requested": int(requested),
        "saved": len(entries),
        "selection": "first_N_metric_eligible_in_deterministic_evaluation_order",
        "context": dict(context),
        "samples": [dict(entry) for entry in entries],
        "unindexed_npz_ignored": unindexed,
        "consumer_contract": "consume_only_samples_listed_in_this_index",
    }
    _atomic_json(directory / "index.json", payload)
    return payload


def load_snapshot(
    path: str | Path,
    *,
    maximum_mib: int = DEFAULT_MAX_SNAPSHOT_MIB,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    source = Path(path).expanduser().resolve()
    if maximum_mib <= 0:
        raise ValueError("maximum_mib must be positive")
    if source.suffix.lower() != ".npz" or not source.is_file():
        raise ValueError("snapshot must be an existing .npz file")
    if expected_bytes is not None and (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes < 0
    ):
        raise ValueError("expected snapshot byte size must be non-negative")
    if expected_sha256 is not None and re.fullmatch(
        r"[0-9a-f]{64}", expected_sha256
    ) is None:
        raise ValueError("expected snapshot SHA-256 must be 64 lowercase hex digits")
    initial_stat = source.stat()
    initial_identity = (
        initial_stat.st_dev,
        initial_stat.st_ino,
        initial_stat.st_size,
        initial_stat.st_mtime_ns,
        initial_stat.st_ctime_ns,
    )
    if expected_bytes is not None and initial_stat.st_size != expected_bytes:
        raise ValueError("snapshot byte size does not match its visualization index")
    if initial_stat.st_size > maximum_mib * 1024 * 1024:
        raise ValueError(
            f"snapshot exceeds the {maximum_mib} MiB safety limit; "
            "do not pass an official multi-gigabyte MVSEC flow NPZ"
        )
    if expected_sha256 is not None and _sha256_file(source) != expected_sha256:
        raise ValueError("snapshot SHA-256 does not match its visualization index")
    with zipfile.ZipFile(source, "r") as archive:
        members = archive.infolist()
        if len(members) > 16 or len({member.filename for member in members}) != len(
            members
        ):
            raise ValueError("snapshot NPZ has too many or duplicate members")
        if any(
            not member.filename.endswith(".npy")
            or "/" in member.filename
            or "\\" in member.filename
            for member in members
        ):
            raise ValueError("snapshot NPZ contains an invalid member name")
        uncompressed_bytes = sum(member.file_size for member in members)
        if uncompressed_bytes > maximum_mib * 1024 * 1024:
            raise ValueError(
                f"snapshot arrays exceed the {maximum_mib} MiB decompression limit"
            )
    with np.load(source, allow_pickle=False) as archive:
        if "metadata_json" not in archive.files:
            raise KeyError("snapshot lacks metadata_json")
        metadata_value = np.asarray(archive["metadata_json"])
        if metadata_value.ndim != 0 or metadata_value.dtype.kind not in {"U", "S"}:
            raise TypeError("metadata_json must be a scalar string")
        raw_metadata = metadata_value.item()
        if isinstance(raw_metadata, bytes):
            raw_metadata = raw_metadata.decode("utf-8")
        metadata = json.loads(str(raw_metadata))
        if not isinstance(metadata, dict) or metadata.get("schema") != SNAPSHOT_SCHEMA:
            raise ValueError("unsupported MVSEC visualization snapshot schema")
        arrays = {
            name: np.asarray(archive[name])
            for name in archive.files
            if name != "metadata_json"
        }
    final_stat = source.stat()
    final_identity = (
        final_stat.st_dev,
        final_stat.st_ino,
        final_stat.st_size,
        final_stat.st_mtime_ns,
        final_stat.st_ctime_ns,
    )
    if final_identity != initial_identity:
        raise RuntimeError("snapshot changed while it was being validated and loaded")
    _validate_snapshot_arrays(str(metadata.get("kind")), arrays)
    return metadata, arrays


def _robust_max(values: np.ndarray, *, percentile: float = 99.0) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite) & (finite > 0)]
    if not finite.size:
        return 1.0
    return max(float(np.percentile(finite, percentile)), 1e-12)


def event_image_rgb(event_image: np.ndarray) -> np.ndarray:
    values = _finite_array(event_image, "event_image")
    if values.ndim == 2:
        positive = np.maximum(values, 0.0)
        negative = np.maximum(-values, 0.0)
    elif values.ndim == 3 and values.shape[0] >= 2:
        midpoint = values.shape[0] // 2
        if values.shape[0] % 2:
            negative = np.maximum(values[:-1], 0.0).sum(axis=0)
            positive = np.maximum(values[-1], 0.0)
        else:
            negative = np.maximum(values[:midpoint], 0.0).sum(axis=0)
            positive = np.maximum(values[midpoint:], 0.0).sum(axis=0)
    elif values.ndim == 3:
        positive = np.maximum(values[0], 0.0)
        negative = np.maximum(-values[0], 0.0)
    else:
        raise ValueError("event_image must have shape [H,W] or [C,H,W]")
    positive = np.log1p(positive)
    negative = np.log1p(negative)
    scale = _robust_max(np.concatenate((positive.ravel(), negative.ravel())))
    positive = np.clip(positive / scale, 0.0, 1.0)
    negative = np.clip(negative / scale, 0.0, 1.0)
    rgb = np.full((*positive.shape, 3), 9.0, dtype=np.float64)
    rgb += positive[..., None] * np.asarray((246.0, 92.0, 64.0))
    rgb += negative[..., None] * np.asarray((48.0, 156.0, 246.0))
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def _hsv_to_rgb(hue: np.ndarray, saturation: np.ndarray, value: np.ndarray) -> np.ndarray:
    h6 = np.mod(hue, 1.0) * 6.0
    sector = np.floor(h6).astype(np.int64) % 6
    fraction = h6 - np.floor(h6)
    p = value * (1.0 - saturation)
    q = value * (1.0 - saturation * fraction)
    t = value * (1.0 - saturation * (1.0 - fraction))
    output = np.empty((*hue.shape, 3), dtype=np.float64)
    choices = (
        (value, t, p),
        (q, value, p),
        (p, value, t),
        (p, q, value),
        (t, p, value),
        (value, p, q),
    )
    for index, channels in enumerate(choices):
        selected = sector == index
        for channel, channel_values in enumerate(channels):
            output[..., channel][selected] = channel_values[selected]
    return output


def flow_rgb(
    flow: np.ndarray,
    *,
    valid: np.ndarray | None = None,
    maximum: float | None = None,
) -> tuple[np.ndarray, float]:
    values = _finite_array(flow, "flow", ndim=3)
    if values.shape[0] != 2:
        raise ValueError("flow must have shape [2,H,W]")
    magnitude = np.sqrt(np.square(values[0]) + np.square(values[1]))
    if valid is None:
        mask = np.ones_like(magnitude, dtype=np.bool_)
    else:
        mask = np.asarray(valid, dtype=np.bool_)
        if mask.shape != magnitude.shape:
            raise ValueError("flow mask shape differs from flow")
    scale = _robust_max(magnitude[mask]) if maximum is None else float(maximum)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("flow color scale must be finite and positive")
    hue = np.mod(np.arctan2(-values[1], -values[0]) / (2.0 * np.pi) + 0.5, 1.0)
    value = np.clip(magnitude / scale, 0.0, 1.0)
    rgb = _hsv_to_rgb(hue, np.ones_like(value), value)
    rgb[~mask] = np.asarray((0.035, 0.035, 0.035))
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8), scale


def _heat_rgb(values: np.ndarray, *, valid: np.ndarray, maximum: float) -> np.ndarray:
    if not math.isfinite(maximum) or maximum <= 0:
        maximum = 1.0
    level = np.clip(np.asarray(values, dtype=np.float64) / maximum, 0.0, 1.0)
    anchors = np.asarray(
        [
            (12, 18, 48),
            (40, 84, 176),
            (24, 190, 184),
            (242, 211, 64),
            (226, 48, 44),
        ],
        dtype=np.float64,
    )
    position = level * (len(anchors) - 1)
    lower = np.floor(position).astype(np.int64)
    upper = np.minimum(lower + 1, len(anchors) - 1)
    weight = position - lower
    rgb = anchors[lower] * (1.0 - weight[..., None]) + anchors[upper] * weight[..., None]
    rgb[~valid] = np.asarray((9.0, 9.0, 9.0))
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def mask_rgb(valid: np.ndarray) -> np.ndarray:
    mask = np.asarray(valid, dtype=np.bool_)
    output = np.full((*mask.shape, 3), 18, dtype=np.uint8)
    output[mask] = np.asarray((238, 244, 252), dtype=np.uint8)
    return output


def iwe_counts(
    events: Mapping[str, np.ndarray],
    flow: np.ndarray,
    *,
    t_start_us: int,
    t_end_us: int,
) -> np.ndarray:
    """Bilinearly splat events to the interval end using dense full-window flow."""

    dense = _finite_array(flow, "flow", ndim=3)
    if dense.shape[0] != 2 or t_end_us <= t_start_us:
        raise ValueError("IWE requires [2,H,W] flow and a positive event interval")
    height, width = dense.shape[1:]
    x = np.asarray(events["event_x"], dtype=np.int64)
    y = np.asarray(events["event_y"], dtype=np.int64)
    t_us = np.asarray(events["event_t_us"], dtype=np.int64)
    polarity = np.asarray(events["event_polarity"])
    if not (len(x) == len(y) == len(t_us) == len(polarity)):
        raise ValueError("IWE event fields have inconsistent lengths")
    output = np.zeros((2, height, width), dtype=np.float64)
    if not len(x):
        return output.astype(np.float32)
    if (
        bool(np.any(x < 0))
        or bool(np.any(x >= width))
        or bool(np.any(y < 0))
        or bool(np.any(y >= height))
        or bool(np.any(t_us <= t_start_us))
        or bool(np.any(t_us > t_end_us))
    ):
        raise ValueError("events fall outside the IWE spatial or temporal interval")
    tau = (t_us.astype(np.float64) - t_start_us) / float(t_end_us - t_start_us)
    warped_x = x + (1.0 - tau) * dense[0, y, x]
    warped_y = y + (1.0 - tau) * dense[1, y, x]
    x0 = np.floor(warped_x).astype(np.int64)
    y0 = np.floor(warped_y).astype(np.int64)
    dx = warped_x - x0
    dy = warped_y - y0
    channel = (polarity > 0).astype(np.int64)
    for offset_x, offset_y, weight in (
        (0, 0, (1.0 - dx) * (1.0 - dy)),
        (1, 0, dx * (1.0 - dy)),
        (0, 1, (1.0 - dx) * dy),
        (1, 1, dx * dy),
    ):
        destination_x = x0 + offset_x
        destination_y = y0 + offset_y
        retained = (
            (destination_x >= 0)
            & (destination_x < width)
            & (destination_y >= 0)
            & (destination_y < height)
        )
        np.add.at(
            output,
            (
                channel[retained],
                destination_y[retained],
                destination_x[retained],
            ),
            weight[retained],
        )
    return output.astype(np.float32)


def iwe_rgb(iwe: np.ndarray) -> np.ndarray:
    values = _finite_array(iwe, "IWE", ndim=3)
    if values.shape[0] != 2 or bool(np.any(values < 0)):
        raise ValueError("IWE must be non-negative [2,H,W]")
    return event_image_rgb(values)


def _iwe_focus(iwe: np.ndarray) -> float:
    values = np.asarray(iwe, dtype=np.float64)
    return float(np.var(values[0]) + np.var(values[1]))


def _flow_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float | int | None]:
    count = int(np.count_nonzero(valid))
    if not count:
        return {"valid_pixels": 0, "AEPE": None, "1PE_percent": None, "3PE_percent": None}
    endpoint = np.sqrt(np.square(prediction - target).sum(axis=0))[valid]
    return {
        "valid_pixels": count,
        "AEPE": float(endpoint.mean()),
        "1PE_percent": float(100.0 * np.mean(endpoint > 1.0)),
        "2PE_percent": float(100.0 * np.mean(endpoint > 2.0)),
        "3PE_percent": float(100.0 * np.mean(endpoint > 3.0)),
    }


def _depth_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float | int | None]:
    count = int(np.count_nonzero(valid))
    if not count:
        return {"valid_pixels": 0, "MAE": None, "RMSE": None, "AbsRel": None, "delta1": None}
    estimate = prediction[valid].astype(np.float64)
    truth = target[valid].astype(np.float64)
    if bool(np.any(estimate <= 0)) or bool(np.any(truth <= 0)):
        raise ValueError("valid metric depths must be positive")
    error = estimate - truth
    ratio = np.maximum(estimate / truth, truth / estimate)
    return {
        "valid_pixels": count,
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(np.square(error)))),
        "AbsRel": float(np.mean(np.abs(error) / truth)),
        "delta1": float(np.mean(ratio < 1.25)),
    }


def _panel_html(panels: Sequence[tuple[str, str]]) -> str:
    return "".join(
        "<figure>"
        f'<img src="{html.escape(filename, quote=True)}" alt="{html.escape(label)}">'
        f"<figcaption>{html.escape(label)}</figcaption>"
        "</figure>"
        for filename, label in panels
    )


def render_snapshot(
    snapshot: str | Path,
    output_dir: str | Path,
    *,
    maximum_mib: int = DEFAULT_MAX_SNAPSHOT_MIB,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    source = Path(snapshot).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    metadata, arrays = load_snapshot(
        source,
        maximum_mib=maximum_mib,
        expected_bytes=expected_bytes,
        expected_sha256=expected_sha256,
    )
    destination.mkdir(parents=True, exist_ok=True)
    kind = str(metadata["kind"])
    target = arrays["target"].astype(np.float64, copy=False)
    prediction = arrays["prediction"].astype(np.float64, copy=False)
    valid = arrays["valid"].astype(np.bool_, copy=False)
    panels: list[tuple[str, str]] = []

    def save(name: str, label: str, image: np.ndarray) -> None:
        write_png(destination / name, image)
        panels.append((name, label))

    save(
        "events.png",
        "Event representation (OFF=blue, ON=red)",
        event_image_rgb(arrays["event_image"]),
    )
    save("valid.png", "Evaluation valid mask", mask_rgb(valid))
    diagnostics: dict[str, Any]
    color_scales: dict[str, float] = {}
    if kind == "flow":
        magnitudes = np.concatenate(
            (
                np.sqrt(np.square(target).sum(axis=0))[valid],
                np.sqrt(np.square(prediction).sum(axis=0))[valid],
            )
        )
        flow_scale = _robust_max(magnitudes)
        target_rgb, _ = flow_rgb(target, valid=valid, maximum=flow_scale)
        prediction_rgb, _ = flow_rgb(prediction, valid=valid, maximum=flow_scale)
        endpoint = np.sqrt(np.square(prediction - target).sum(axis=0))
        epe_scale = _robust_max(endpoint[valid], percentile=99.0)
        save("target_flow.png", "Ground-truth flow (shared HSV scale)", target_rgb)
        save("prediction_flow.png", "Predicted flow (shared HSV scale)", prediction_rgb)
        save("epe.png", "Endpoint error", _heat_rgb(endpoint, valid=valid, maximum=epe_scale))
        diagnostics = _flow_metrics(prediction, target, valid)
        color_scales.update({"flow_p99_pixels": flow_scale, "epe_p99_pixels": epe_scale})
        event_keys = {"event_x", "event_y", "event_t_us", "event_polarity"}
        if event_keys.issubset(arrays):
            start = int(metadata["event_window_start_us"])
            end = int(metadata["event_window_end_us"])
            event_arrays = {name: arrays[name] for name in event_keys}
            zero_iwe = iwe_counts(
                event_arrays,
                np.zeros_like(target),
                t_start_us=start,
                t_end_us=end,
            )
            predicted_iwe = iwe_counts(
                event_arrays,
                prediction,
                t_start_us=start,
                t_end_us=end,
            )
            target_iwe = iwe_counts(
                event_arrays,
                target,
                t_start_us=start,
                t_end_us=end,
            )
            save("cmax_before.png", "CMax diagnostic: unwarped IWE", iwe_rgb(zero_iwe))
            save("cmax_after.png", "CMax diagnostic: prediction-warped IWE", iwe_rgb(predicted_iwe))
            save("cmax_ground_truth.png", "CMax diagnostic: GT-warped IWE", iwe_rgb(target_iwe))
            before_focus = _iwe_focus(zero_iwe)
            after_focus = _iwe_focus(predicted_iwe)
            target_focus = _iwe_focus(target_iwe)
            diagnostics["cmax_visual_diagnostic"] = {
                "definition": "sum_of_per_polarity_spatial_variances_end_reference_IWE",
                "not_training_objective": True,
                "before_focus": before_focus,
                "prediction_focus": after_focus,
                "ground_truth_focus": target_focus,
                "prediction_minus_before": after_focus - before_focus,
                "event_subsampling_affects_focus": bool(
                    metadata.get("event_count_stored") != metadata.get("event_count_total")
                ),
            }
    else:
        valid_values = np.concatenate((target[valid], prediction[valid]))
        depth_min = float(np.percentile(valid_values, 1.0)) if valid_values.size else 0.0
        depth_max = float(np.percentile(valid_values, 99.0)) if valid_values.size else 1.0
        if depth_max <= depth_min:
            depth_max = depth_min + 1.0
        normalized_target = (target - depth_min) / (depth_max - depth_min)
        normalized_prediction = (prediction - depth_min) / (depth_max - depth_min)
        absolute_error = np.abs(prediction - target)
        error_scale = _robust_max(absolute_error[valid])
        save(
            "target_depth.png",
            "Ground-truth depth (shared scale)",
            _heat_rgb(normalized_target, valid=valid, maximum=1.0),
        )
        save(
            "prediction_depth.png",
            "Predicted depth (shared scale)",
            _heat_rgb(normalized_prediction, valid=valid, maximum=1.0),
        )
        save(
            "absolute_error.png",
            "Absolute depth error",
            _heat_rgb(absolute_error, valid=valid, maximum=error_scale),
        )
        diagnostics = _depth_metrics(prediction, target, valid)
        color_scales.update(
            {
                "depth_p01_m": depth_min,
                "depth_p99_m": depth_max,
                "absolute_error_p99_m": error_scale,
            }
        )
    summary = {
        "schema": REPORT_SCHEMA,
        "kind": kind,
        "snapshot": {
            "path": str(source),
            "bytes": source.stat().st_size,
            "sha256": _sha256_file(source),
        },
        "metadata": metadata,
        "metrics": diagnostics,
        "color_scales": color_scales,
        "panels": [{"path": name, "label": label} for name, label in panels],
    }
    _atomic_json(destination / "summary.json", summary)
    metric_rows = "".join(
        f"<tr><th>{html.escape(str(name))}</th><td>{html.escape(str(value))}</td></tr>"
        for name, value in diagnostics.items()
        if not isinstance(value, Mapping)
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>MVSEC {html.escape(kind)} visualization</title><style>
body{{font:15px system-ui,sans-serif;margin:28px;background:#0b1020;color:#e5edf8}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px}}
figure{{margin:0;padding:12px;background:#151d30;border-radius:10px}}
img{{width:100%;image-rendering:auto}}figcaption{{margin-top:8px}}
table{{border-collapse:collapse;margin:18px 0}}
th,td{{padding:6px 12px;border-bottom:1px solid #334155;text-align:left}}
code{{overflow-wrap:anywhere}}</style></head><body>
<h1>MVSEC {html.escape(kind)} visualization</h1>
<p>Snapshot: <code>{html.escape(str(source))}</code></p><table>{metric_rows}</table>
<div class="grid">{_panel_html(panels)}</div>
</body></html>"""
    _atomic_text(destination / "index.html", document)
    return summary


def _read_records(path: Path) -> tuple[Any, list[Mapping[str, Any]]]:
    if path.is_dir():
        candidates = (path / "report.json", path / "metrics.json", path / "train.jsonl")
        existing = [candidate for candidate in candidates if candidate.is_file()]
        if not existing:
            raise ValueError(f"run directory contains no supported report: {path}")
        path = existing[0]
    if path.suffix.lower() == ".jsonl":
        records: list[Mapping[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"JSONL line {line_number} is not an object")
                records.append(value)
        return records, records
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("comparison report JSON must contain an object")
    history: Any = payload.get("training_history")
    if history is None and isinstance(payload.get("training"), dict):
        history = payload["training"].get("history")
    if history is None:
        history = []
    if not isinstance(history, list) or any(not isinstance(item, dict) for item in history):
        raise TypeError("training history must be a list of objects")
    return payload, history


def _get_path(value: Any, path: str) -> Any:
    current = value
    for component in path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            raise KeyError(path)
        current = current[component]
    return current


def _numeric_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _draw_line(
    canvas: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: np.ndarray,
    *,
    thickness: int = 2,
) -> None:
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    xs = np.rint(np.linspace(x0, x1, steps + 1)).astype(np.int64)
    ys = np.rint(np.linspace(y0, y1, steps + 1)).astype(np.int64)
    radius = max(0, thickness // 2)
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            xx = np.clip(xs + dx, 0, canvas.shape[1] - 1)
            yy = np.clip(ys + dy, 0, canvas.shape[0] - 1)
            canvas[yy, xx] = color


def _chart_grid(count: int) -> tuple[int, int]:
    columns = min(3, max(1, count))
    return math.ceil(count / columns), columns


def _bar_chart(metric_values: Sequence[Sequence[float | None]]) -> np.ndarray:
    rows, columns = _chart_grid(len(metric_values))
    panel_height, panel_width = 230, 320
    canvas = np.full((rows * panel_height, columns * panel_width, 3), 250, dtype=np.uint8)
    for metric_index, values in enumerate(metric_values):
        row, column = divmod(metric_index, columns)
        top, left = row * panel_height, column * panel_width
        plot_left, plot_right = left + 34, left + panel_width - 18
        plot_top, plot_bottom = top + 18, top + panel_height - 28
        canvas[plot_top:plot_bottom + 1, plot_left] = 75
        canvas[plot_bottom, plot_left:plot_right + 1] = 75
        finite = [value for value in values if value is not None]
        maximum = max(finite, default=1.0)
        minimum = min(0.0, min(finite, default=0.0))
        if maximum <= minimum:
            maximum = minimum + 1.0
        width = max(2, (plot_right - plot_left) // max(1, len(values)))
        zero_y = int(
            round(
                plot_bottom
                - (-minimum / (maximum - minimum)) * (plot_bottom - plot_top)
            )
        )
        for index, value in enumerate(values):
            if value is None:
                continue
            y = int(
                round(
                    plot_bottom
                    - ((value - minimum) / (maximum - minimum))
                    * (plot_bottom - plot_top)
                )
            )
            x0 = plot_left + index * width + 3
            x1 = min(plot_right, x0 + width - 6)
            canvas[min(y, zero_y):max(y, zero_y) + 1, x0:x1 + 1] = _COLORS[index % len(_COLORS)]
    return canvas


def _curve_chart(
    curves: Sequence[Sequence[tuple[np.ndarray, np.ndarray] | None]],
) -> np.ndarray:
    rows, columns = _chart_grid(len(curves))
    panel_height, panel_width = 250, 360
    canvas = np.full((rows * panel_height, columns * panel_width, 3), 250, dtype=np.uint8)
    for curve_index, run_curves in enumerate(curves):
        row, column = divmod(curve_index, columns)
        top, left = row * panel_height, column * panel_width
        plot_left, plot_right = left + 38, left + panel_width - 18
        plot_top, plot_bottom = top + 18, top + panel_height - 30
        canvas[plot_top:plot_bottom + 1, plot_left] = 75
        canvas[plot_bottom, plot_left:plot_right + 1] = 75
        present = [pair for pair in run_curves if pair is not None and len(pair[0])]
        if not present:
            continue
        all_x = np.concatenate([pair[0] for pair in present])
        all_y = np.concatenate([pair[1] for pair in present])
        x_min, x_max = float(all_x.min()), float(all_x.max())
        y_min, y_max = float(all_y.min()), float(all_y.max())
        if x_max <= x_min:
            x_max = x_min + 1.0
        if y_max <= y_min:
            padding = max(abs(y_min) * 0.05, 0.5)
            y_min -= padding
            y_max += padding
        for run_index, pair in enumerate(run_curves):
            if pair is None or not len(pair[0]):
                continue
            x, y = pair
            px = plot_left + np.rint(
                (x - x_min) / (x_max - x_min) * (plot_right - plot_left)
            ).astype(np.int64)
            py = plot_bottom - np.rint(
                (y - y_min) / (y_max - y_min) * (plot_bottom - plot_top)
            ).astype(np.int64)
            for point in range(1, len(px)):
                _draw_line(
                    canvas,
                    int(px[point - 1]),
                    int(py[point - 1]),
                    int(px[point]),
                    int(py[point]),
                    _COLORS[run_index % len(_COLORS)],
                    thickness=3,
                )
    return canvas


def _auto_metrics(payloads: Sequence[Any]) -> list[str]:
    candidates = (
        "evaluation.metrics.sample_average.AEPE",
        "evaluation.metrics.pixel_average.AEPE",
        "evaluation.metrics.sample_average.3PE_percent",
        "evaluation.metrics.pixel_average.3PE_percent",
        "metrics.dev.pixel_average.AbsRel",
        "metrics.dev.pixel_average.RMSE",
        "metrics.dev.pixel_average.SILog",
        "metrics.final_test.pixel_average.AbsRel",
        "metrics.final_test.pixel_average.RMSE",
        "metrics.final_test.pixel_average.SILog",
        "final_test.pixel_average.AbsRel",
        "final_test.pixel_average.RMSE",
        "final_test.pixel_average.SILog",
    )
    selected = []
    for candidate in candidates:
        for payload in payloads:
            try:
                value = _get_path(payload, candidate)
            except KeyError:
                continue
            if _numeric_or_none(value) is not None:
                selected.append(candidate)
                break
    return selected


def _auto_curves(histories: Sequence[Sequence[Mapping[str, Any]]]) -> list[str]:
    candidates = (
        "loss",
        "mean_endpoint_loss",
        "mean_loss",
        "valid_log_depth_smooth_l1",
        "future_prediction_loss",
        "cmax_weighted_loss",
        "rate_alignment_weighted_loss",
        "latent_straightening_weighted_loss",
        "latent_dynamics_weighted_loss",
        "rate_alignment_mean_weight",
        "prediction_std",
        "target_std",
    )
    return [
        candidate
        for candidate in candidates
        if any(
            any(_numeric_or_none(record.get(candidate)) is not None for record in history)
            for history in histories
        )
    ]


def _artifact_contract(artifact: Any) -> dict[str, Any] | None:
    if not isinstance(artifact, Mapping):
        return None
    return {
        key: artifact.get(key)
        for key in ("format", "bytes", "sha256", "manifest_declared_sha256")
        if artifact.get(key) is not None
    }


def _comparison_contract(payload: Any) -> dict[str, Any] | None:
    """Extract only fields that must match for a paired metric comparison."""

    if not isinstance(payload, Mapping):
        return None
    evaluation = payload.get("evaluation")
    protocol = payload.get("protocol")
    if isinstance(evaluation, Mapping) and isinstance(protocol, Mapping):
        selection = evaluation.get("selection")
        if not isinstance(selection, Mapping):
            return None
        command = payload.get("command")
        if command not in {"probe", "cmax-eval"}:
            return None
        head = payload.get("head")
        if not isinstance(head, Mapping):
            return None
        runtime = payload.get("runtime")
        if not isinstance(runtime, Mapping):
            runtime = {}
        target_artifacts = selection.get("target_artifacts", [])
        contract: dict[str, Any] = {
            "task": "flow",
            "command": command,
            "head": {
                key: head.get(key)
                for key in (
                    "class",
                    "embed_dim",
                    "hidden_dim",
                    "head_depth",
                    "flow_scale",
                    "max_displacement_pixels_per_base_window",
                    "initialization",
                    "architecture_source",
                )
            },
            "protocol": {
                key: protocol.get(key)
                for key in (
                    "name",
                    "stage",
                    "target_timebase_contract",
                    "coordinate_frame",
                    "model_canvas_height_width",
                    "native_sensor_center_padding_yx",
                    "alignment",
                    "event_history",
                    "temporal_dev_split",
                    "representation_pretraining_visibility_contract",
                    "flow_rate",
                    "validity_mask",
                    "minimum_valid_pixels_per_frame",
                    "evflownet_test_interval",
                )
            },
            "evaluation": {
                "split": evaluation.get("split"),
                "role": evaluation.get("role"),
                "manifest_sha256": (
                    evaluation.get("manifest_artifact", {}).get("sha256")
                    if isinstance(evaluation.get("manifest_artifact"), Mapping)
                    else None
                ),
                "target_index_timestamp_sha256": selection.get(
                    "target_index_timestamp_sha256"
                ),
                "targets": selection.get("targets"),
                "target_artifacts": [
                    _artifact_contract(artifact) for artifact in target_artifacts
                ],
            },
            "runtime": {
                "precision": runtime.get("precision"),
                "batch_size": runtime.get("batch_size"),
            },
        }
        if command == "probe":
            training = payload.get("training")
            if not isinstance(training, Mapping):
                return None
            training_selection = training.get("selection")
            if not isinstance(training_selection, Mapping):
                return None
            contract["training"] = {
                "split": training.get("split"),
                "role": training.get("role"),
                "manifest_sha256": (
                    training.get("manifest_artifact", {}).get("sha256")
                    if isinstance(training.get("manifest_artifact"), Mapping)
                    else None
                ),
                "selection": {
                    "targets": training_selection.get("targets"),
                    "target_index_timestamp_sha256": training_selection.get(
                        "target_index_timestamp_sha256"
                    ),
                    "target_artifacts": [
                        _artifact_contract(artifact)
                        for artifact in training_selection.get("target_artifacts", [])
                    ],
                },
                "epochs": training.get("epochs"),
                "batch_size": training.get("batch_size"),
                "learning_rate": training.get("learning_rate"),
                "weight_decay": training.get("weight_decay"),
                "loss": training.get("loss"),
                "flow_scaling": training.get("flow_scaling"),
            }
        return contract
    if isinstance(protocol, Mapping) and protocol.get("task") == (
        "mvsec_frozen_recurrent_absolute_depth_probe"
    ):
        training_policy = protocol.get("training_policy", {})
        if not isinstance(training_policy, Mapping):
            training_policy = {}
        head = protocol.get("head", {})
        if not isinstance(head, Mapping):
            head = {}
        backbone = protocol.get("backbone", {})
        if not isinstance(backbone, Mapping):
            backbone = {}

        def split_contract(split: Any) -> Any:
            if not isinstance(split, Mapping):
                return None
            result = {
                key: split.get(key)
                for key in (
                    "role",
                    "recording",
                    "manifest_sha256",
                    "sequence_ids",
                    "target_count",
                    "target_index_timestamp_sha256",
                    "alignment",
                    "causal",
                    "uses_future_events",
                )
            }
            result["target_artifacts"] = [
                {
                    **(_artifact_contract(artifact) or {}),
                    "sequence_ids": artifact.get("sequence_ids"),
                    "target_datasets": artifact.get("target_datasets"),
                    "timestamp_datasets": artifact.get("timestamp_datasets"),
                }
                for artifact in split.get("target_artifacts", [])
                if isinstance(artifact, Mapping)
            ]
            return result

        return {
            "task": "depth",
            "name": protocol.get("name"),
            "stage": protocol.get("stage"),
            "backbone": {
                key: backbone.get(key)
                for key in (
                    "frozen",
                    "feature",
                    "history_steps",
                    "history_policy",
                    "history_source",
                    "state_reset",
                    "window_us",
                    "stride_us",
                    "history_span_us",
                )
            },
            "head": {
                key: head.get(key)
                for key in (
                    "order",
                    "embed_dim",
                    "hidden_dim",
                    "initialization",
                    "parameter_precision",
                    "patch_grid",
                )
            },
            "temporal_dev_split": protocol.get("temporal_dev_split"),
            "representation_pretraining_visibility_contract": protocol.get(
                "representation_pretraining_visibility_contract"
            ),
            "geometry": protocol.get("geometry"),
            "depth": protocol.get("depth"),
            "training_policy": {
                key: training_policy.get(key)
                for key in (
                    "recording",
                    "fixed_epochs",
                    "batch_size",
                    "learning_rate",
                    "weight_decay",
                    "precision",
                    "evaluation_used_for_model_selection",
                    "model_selection",
                )
            },
            "target_selection": protocol.get("target_selection"),
            "train_targets": split_contract(protocol.get("train_targets")),
            "evaluation_targets": [
                split_contract(split) for split in protocol.get("evaluation_targets", [])
            ],
        }
    return None


def _canonical_sha256(value: Any) -> str:
    serialized = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def compare_reports(
    runs: Sequence[tuple[str, str | Path]],
    output_dir: str | Path,
    *,
    metrics: Sequence[str] = (),
    curves: Sequence[str] = (),
    aggregate_seeds: bool = False,
    allow_incompatible: bool = False,
) -> dict[str, Any]:
    if len(runs) < 2:
        raise ValueError("comparison requires at least two runs")
    labels = [label for label, _ in runs]
    if len(set(labels)) != len(labels) or any(not label.strip() for label in labels):
        raise ValueError("comparison run labels must be unique and non-empty")
    payloads: list[Any] = []
    histories: list[list[Mapping[str, Any]]] = []
    artifacts: list[dict[str, Any]] = []
    for label, raw_path in runs:
        path = Path(raw_path).expanduser().resolve()
        if path.is_dir():
            candidates = (path / "report.json", path / "metrics.json", path / "train.jsonl")
            source = next((candidate for candidate in candidates if candidate.is_file()), None)
            if source is None:
                raise ValueError(f"run directory contains no supported report: {path}")
        else:
            source = path
        payload, history = _read_records(source)
        payloads.append(payload)
        histories.append(list(history))
        artifacts.append(
            {
                "label": label,
                "path": str(source),
                "format": source.suffix.lower().lstrip("."),
                "bytes": source.stat().st_size,
                "sha256": _sha256_file(source),
            }
        )
    contracts = [_comparison_contract(payload) for payload in payloads]
    contract_hashes = [
        _canonical_sha256(contract) if contract is not None else None
        for contract in contracts
    ]
    known_hashes = {value for value in contract_hashes if value is not None}
    untyped_json_labels = [
        labels[index]
        for index, contract in enumerate(contracts)
        if contract is None and artifacts[index]["format"] != "jsonl"
    ]
    if untyped_json_labels and not allow_incompatible:
        raise ValueError(
            "JSON reports lack a recognized MVSEC evaluation contract: "
            + ", ".join(untyped_json_labels)
        )
    if len(known_hashes) > 1 and not allow_incompatible:
        raise ValueError(
            "MVSEC evaluation contracts differ; pass --allow-incompatible only "
            "for an explicitly exploratory comparison"
        )
    if known_hashes and any(value is None for value in contract_hashes) and not allow_incompatible:
        raise ValueError(
            "cannot mix contract-bearing MVSEC reports with untyped logs without "
            "--allow-incompatible"
        )
    compatibility = {
        "status": (
            "matched"
            if len(known_hashes) == 1 and all(value is not None for value in contract_hashes)
            else (
                "exploratory_incompatible_override"
                if len(known_hashes) > 1
                or (
                    known_hashes
                    and any(value is None for value in contract_hashes)
                )
                else "exploratory_no_evaluation_contract"
            )
        ),
        "allow_incompatible": allow_incompatible,
        "contract_sha256_by_run": dict(zip(labels, contract_hashes, strict=True)),
        "contracts_by_run": dict(zip(labels, contracts, strict=True)),
    }
    metric_names = list(metrics) or _auto_metrics(payloads)
    curve_names = list(curves) or _auto_curves(histories)
    if not metric_names and not curve_names:
        raise ValueError("no comparable numeric metrics or training curves were found")
    metric_table: dict[str, dict[str, float | None]] = {}
    for name in metric_names:
        values: dict[str, float | None] = {}
        for label, payload in zip(labels, payloads, strict=True):
            try:
                raw_value = _get_path(payload, name)
            except KeyError:
                raw_value = None
            values[label] = _numeric_or_none(raw_value)
        if all(value is None for value in values.values()):
            raise KeyError(f"metric path was absent or non-numeric in every run: {name}")
        if aggregate_seeds and any(value is None for value in values.values()):
            raise ValueError(
                f"seed aggregation requires metric {name!r} in every run"
            )
        metric_table[name] = values
    curve_table: dict[str, dict[str, dict[str, list[float]] | None]] = {}
    chart_curves: list[list[tuple[np.ndarray, np.ndarray] | None]] = []
    for name in curve_names:
        per_run: dict[str, dict[str, list[float]] | None] = {}
        rendered_run: list[tuple[np.ndarray, np.ndarray] | None] = []
        for label, history in zip(labels, histories, strict=True):
            points: list[tuple[float, float]] = []
            for ordinal, record in enumerate(history, start=1):
                try:
                    raw_y = _get_path(record, name)
                except KeyError:
                    continue
                y = _numeric_or_none(raw_y)
                if y is None:
                    continue
                raw_x = record.get("global_step", record.get("epoch", ordinal))
                x = _numeric_or_none(raw_x)
                if x is not None:
                    points.append((x, y))
            if points:
                x_array = np.asarray([point[0] for point in points], dtype=np.float64)
                y_array = np.asarray([point[1] for point in points], dtype=np.float64)
                per_run[label] = {"x": x_array.tolist(), "y": y_array.tolist()}
                rendered_run.append((x_array, y_array))
            else:
                per_run[label] = None
                rendered_run.append(None)
        if all(value is None for value in per_run.values()):
            raise KeyError(f"curve path was absent or non-numeric in every run: {name}")
        curve_table[name] = per_run
        chart_curves.append(rendered_run)
    flat_seed_groups: dict[str, list[tuple[str, int | None]]] = {}
    hierarchical_seed_groups: dict[
        str, dict[int, list[tuple[str, int]]]
    ] = {}
    condition_order: list[str] = []
    for label in labels:
        hierarchical = _HIERARCHICAL_SEED_LABEL.fullmatch(label)
        if hierarchical:
            condition = hierarchical.group("condition")
            encoder_seed = int(hierarchical.group("encoder"))
            probe_seed = int(hierarchical.group("probe"))
            if condition in flat_seed_groups:
                raise ValueError(
                    f"cannot mix flat and hierarchical seed labels for {condition!r}"
                )
            if condition not in hierarchical_seed_groups:
                hierarchical_seed_groups[condition] = {}
                condition_order.append(condition)
            members = hierarchical_seed_groups[condition].setdefault(
                encoder_seed, []
            )
            if any(existing_probe == probe_seed for _, existing_probe in members):
                raise ValueError(
                    f"duplicate probe seed {probe_seed} for {condition!r}, "
                    f"encoder seed {encoder_seed}"
                )
            members.append((label, probe_seed))
            continue
        match = _SEED_LABEL.fullmatch(label)
        condition = match.group("condition") if match else label
        seed = int(match.group("seed")) if match else None
        if condition in hierarchical_seed_groups:
            raise ValueError(
                f"cannot mix flat and hierarchical seed labels for {condition!r}"
            )
        if condition not in flat_seed_groups:
            flat_seed_groups[condition] = []
            condition_order.append(condition)
        if any(
            existing_seed == seed
            for _, existing_seed in flat_seed_groups[condition]
        ):
            raise ValueError(f"duplicate seed label within condition {condition!r}")
        flat_seed_groups[condition].append((label, seed))
    if aggregate_seeds and len(condition_order) > 1:
        signatures: dict[str, Any] = {}
        for condition in condition_order:
            if condition in hierarchical_seed_groups:
                signatures[condition] = {
                    "mode": "probe_then_encoder",
                    "seeds": [
                        [encoder_seed, sorted(probe for _, probe in members)]
                        for encoder_seed, members in sorted(
                            hierarchical_seed_groups[condition].items()
                        )
                    ],
                }
            else:
                signatures[condition] = {
                    "mode": "flat",
                    "seeds": sorted(
                        seed for _, seed in flat_seed_groups[condition]
                        if seed is not None
                    ),
                }
        canonical_signatures = {
            _canonical_sha256(signature) for signature in signatures.values()
        }
        if len(canonical_signatures) > 1:
            raise ValueError(
                "seed sets differ across conditions; paired aggregation requires "
                "the same encoder/probe seed hierarchy"
            )
    seed_aggregates: dict[str, dict[str, dict[str, Any]]] = {}
    if aggregate_seeds:
        for metric_name, values in metric_table.items():
            by_condition: dict[str, dict[str, Any]] = {}
            for condition in condition_order:
                if condition in hierarchical_seed_groups:
                    encoder_statistics: dict[str, dict[str, Any]] = {}
                    encoder_means: list[float] = []
                    probe_run_count = 0
                    for encoder_seed, members in sorted(
                        hierarchical_seed_groups[condition].items()
                    ):
                        numeric = [
                            float(values[label])
                            for label, _ in members
                            if values[label] is not None
                        ]
                        if not numeric:
                            continue
                        probe_array = np.asarray(numeric, dtype=np.float64)
                        probe_mean = float(probe_array.mean())
                        encoder_means.append(probe_mean)
                        probe_run_count += len(numeric)
                        encoder_statistics[str(encoder_seed)] = {
                            "probe_mean": probe_mean,
                            "probe_std_population": float(
                                probe_array.std(ddof=0)
                            ),
                            "probe_count": int(len(probe_array)),
                            "probe_seeds": [probe for _, probe in members],
                        }
                    if not encoder_means:
                        continue
                    encoder_array = np.asarray(encoder_means, dtype=np.float64)
                    by_condition[condition] = {
                        "mean": float(encoder_array.mean()),
                        "std_population": float(encoder_array.std(ddof=0)),
                        "count": int(len(encoder_array)),
                        "aggregation_order": "probe_mean_then_encoder_mean",
                        "encoder_count": int(len(encoder_array)),
                        "probe_run_count": int(probe_run_count),
                        "per_encoder": encoder_statistics,
                    }
                    continue
                members = flat_seed_groups[condition]
                numeric = [
                    float(values[label])
                    for label, _ in members
                    if values[label] is not None
                ]
                if not numeric:
                    continue
                array = np.asarray(numeric, dtype=np.float64)
                by_condition[condition] = {
                    "mean": float(array.mean()),
                    "std_population": float(array.std(ddof=0)),
                    "count": int(len(array)),
                    "aggregation_order": "flat_seed_mean",
                }
            seed_aggregates[metric_name] = by_condition
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    panels: list[tuple[str, str]] = []
    if metric_table:
        metric_values = [
            [values[label] for label in labels] for values in metric_table.values()
        ]
        write_png(destination / "metrics.png", _bar_chart(metric_values))
        panels.append(("metrics.png", "Final metrics (panel order follows comparison.json)"))
        for metric_index, (name, values) in enumerate(metric_table.items()):
            filename = f"metric-{metric_index:02d}-{safe_component(name)}.png"
            write_png(
                destination / filename,
                _bar_chart([[values[label] for label in labels]]),
            )
            panels.append((filename, f"Final metric: {name}"))
    if curve_table:
        write_png(destination / "curves.png", _curve_chart(chart_curves))
        panels.append(("curves.png", "Training curves (panel order follows comparison.json)"))
        for curve_index, name in enumerate(curve_table):
            filename = f"curve-{curve_index:02d}-{safe_component(name)}.png"
            write_png(destination / filename, _curve_chart([chart_curves[curve_index]]))
            panels.append((filename, f"Training curve: {name}"))
    if aggregate_seeds and seed_aggregates:
        conditions = list(condition_order)
        aggregate_values = [
            [
                (
                    values[condition]["mean"]
                    if condition in values
                    else None
                )
                for condition in conditions
            ]
            for values in seed_aggregates.values()
        ]
        write_png(destination / "seed-aggregate-metrics.png", _bar_chart(aggregate_values))
        panels.append(
            (
                "seed-aggregate-metrics.png",
                "Condition means with declared seed hierarchy; std is tabulated below",
            )
        )
    result = {
        "schema": COMPARISON_SCHEMA,
        "runs": artifacts,
        "compatibility": compatibility,
        "run_colors_rgb": {
            label: _COLORS[index % len(_COLORS)].tolist()
            for index, label in enumerate(labels)
        },
        "metrics": metric_table,
        "curves": curve_table,
        "seed_aggregation": {
            "enabled": aggregate_seeds,
            "label_patterns": [
                "CONDITION__seedN",
                "CONDITION__encoder_seedN__probe_seedM",
            ],
            "std_definition": "population_ddof_0",
            "groups": {
                condition: (
                    {
                        "mode": "probe_then_encoder",
                        "encoders": {
                            str(encoder_seed): [
                                {"label": label, "probe_seed": probe_seed}
                                for label, probe_seed in members
                            ]
                            for encoder_seed, members in sorted(
                                hierarchical_seed_groups[condition].items()
                            )
                        },
                    }
                    if condition in hierarchical_seed_groups
                    else {
                        "mode": "flat_seed_mean",
                        "members": [
                            {"label": label, "seed": seed}
                            for label, seed in flat_seed_groups[condition]
                        ],
                    }
                )
                for condition in condition_order
            },
            "metrics": seed_aggregates,
        },
        "panels": [{"path": name, "label": label} for name, label in panels],
    }
    _atomic_json(destination / "comparison.json", result)
    legend = "".join(
        (
            f'<span><i style="background:rgb({color[0]},{color[1]},{color[2]})">'
            f"</i>{html.escape(label)}</span>"
        )
        for index, label in enumerate(labels)
        for color in (_COLORS[index % len(_COLORS)],)
    )
    metric_rows = "".join(
        "<tr><th>" + html.escape(name) + "</th>" + "".join(
            f"<td>{'—' if values[label] is None else f'{values[label]:.6g}'}</td>"
            for label in labels
        ) + "</tr>"
        for name, values in metric_table.items()
    )
    header = "".join(f"<th>{html.escape(label)}</th>" for label in labels)
    aggregate_rows = "".join(
        f"<tr><th>{html.escape(metric_name)}</th><th>{html.escape(condition)}</th>"
        f"<td>{values['mean']:.6g} ± {values['std_population']:.6g}</td>"
        f"<td>{values['count']}</td></tr>"
        for metric_name, conditions in seed_aggregates.items()
        for condition, values in conditions.items()
    )
    aggregate_table = (
        "<h2>Seed aggregates</h2><table><tr><th>Metric</th><th>Condition</th>"
        "<th>Mean ± population std</th><th>N</th></tr>"
        f"{aggregate_rows}</table>"
        if aggregate_seeds
        else ""
    )
    compatibility_note = (
        "Evaluation contract matched across every run."
        if compatibility["status"] == "matched"
        else (
            "Exploratory comparison: a strict shared MVSEC evaluation "
            "contract was not established."
        )
    )
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width"><title>MVSEC run comparison</title><style>
body{{font:15px system-ui,sans-serif;margin:28px;color:#172033}}.grid{{display:grid;gap:20px}}
figure{{margin:0}}img{{max-width:100%;border:1px solid #cbd5e1}}
i{{display:inline-block;width:14px;height:14px;margin:0 5px 0 14px}}
table{{border-collapse:collapse}}th,td{{padding:7px 12px;border-bottom:1px solid #cbd5e1;
text-align:right}}th:first-child{{text-align:left}}</style>
</head><body><h1>MVSEC run comparison</h1>
<p><strong>{html.escape(compatibility_note)}</strong></p><p>{legend}</p>
<table><tr><th>Metric</th>{header}</tr>{metric_rows}</table>{aggregate_table}
<div class="grid">{_panel_html(panels)}</div>
</body></html>"""
    _atomic_text(destination / "index.html", document)
    return result


def _synthetic_snapshot(kind: str, output: Path, seed: int) -> Path:
    generator = np.random.default_rng(seed)
    height, width = 72, 96
    yy, xx = np.mgrid[:height, :width]
    event_image = np.zeros((2, height, width), dtype=np.float32)
    event_image[0] = np.exp(-np.square((xx - 34) / 4.0) - np.square((yy - 36) / 14.0)) * 8
    event_image[1] = np.exp(-np.square((xx - 58) / 5.0) - np.square((yy - 36) / 14.0)) * 8
    valid = ((xx - width / 2) ** 2 / 42**2 + (yy - height / 2) ** 2 / 30**2) < 1
    if kind == "flow":
        target = np.stack(
            (
                2.0 + 0.02 * (yy - height / 2),
                -0.7 + 0.01 * (xx - width / 2),
            )
        ).astype(np.float32)
        prediction = target + generator.normal(0.0, 0.28, target.shape).astype(np.float32)
        event_count = 5000
        event_x = generator.integers(4, width - 4, event_count, dtype=np.int64)
        event_y = generator.integers(4, height - 4, event_count, dtype=np.int64)
        event_t = np.sort(generator.integers(1, 50_001, event_count, dtype=np.int64))
        events = {
            "event_x": event_x,
            "event_y": event_y,
            "event_t_us": event_t,
            "event_polarity": generator.integers(0, 2, event_count, dtype=np.int8),
        }
        extra = {
            "event_window_start_us": 0,
            "event_window_end_us": 50_000,
            "event_count_total": event_count,
            "event_count_stored": event_count,
        }
    else:
        target = (4.0 + 0.08 * xx + 0.04 * yy).astype(np.float32)
        prediction = (target * (1.0 + 0.03 * np.sin(xx / 8.0))).astype(np.float32)
        events = None
        extra = {}
    path = output / f"synthetic_{kind}.npz"
    write_snapshot(
        path,
        kind=kind,
        event_image=event_image,
        target=np.where(valid, target, 0.0),
        prediction=prediction,
        valid=valid,
        metadata={
            "synthetic": True,
            "seed": seed,
            "sequence_id": "synthetic",
            "target_index": 0,
            "label_timestamp_us": 50_000,
            **extra,
        },
        events=events,
    )
    return path


def _parse_run(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("--run must use LABEL=PATH")
    return label.strip(), Path(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render reproducible MVSEC flow/depth snapshots and compare reports."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    sample = commands.add_parser("sample", help="render one exported flow/depth snapshot")
    sample.add_argument("--snapshot", type=Path, required=True)
    sample.add_argument("--output-dir", type=Path, required=True)
    sample.add_argument("--maximum-snapshot-mib", type=int, default=DEFAULT_MAX_SNAPSHOT_MIB)
    sample.add_argument(
        "--expected-bytes",
        type=int,
        help="optional byte size pinned by a visualization index",
    )
    sample.add_argument(
        "--expected-sha256",
        help="optional lowercase SHA-256 pinned by a visualization index",
    )
    compare = commands.add_parser("compare", help="compare two or more JSON/JSONL run reports")
    compare.add_argument("--run", action="append", type=_parse_run, required=True)
    compare.add_argument("--metric", action="append", default=[])
    compare.add_argument("--curve", action="append", default=[])
    compare.add_argument(
        "--aggregate-seeds",
        action="store_true",
        help=(
            "aggregate CONDITION__seedN labels directly, or "
            "CONDITION__encoder_seedN__probe_seedM labels as probe mean then "
            "encoder mean/std"
        ),
    )
    compare.add_argument(
        "--allow-incompatible",
        action="store_true",
        help="permit an explicitly exploratory plot when evaluation contracts differ",
    )
    compare.add_argument("--output-dir", type=Path, required=True)
    synthetic = commands.add_parser("synthetic", help="create dependency-light reference reports")
    synthetic.add_argument("--kind", choices=("flow", "depth", "both"), default="both")
    synthetic.add_argument("--seed", type=int, default=0)
    synthetic.add_argument("--output-dir", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "sample":
        return render_snapshot(
            args.snapshot,
            args.output_dir,
            maximum_mib=args.maximum_snapshot_mib,
            expected_bytes=args.expected_bytes,
            expected_sha256=args.expected_sha256,
        )
    if args.command == "compare":
        return compare_reports(
            args.run,
            args.output_dir,
            metrics=args.metric,
            curves=args.curve,
            aggregate_seeds=args.aggregate_seeds,
            allow_incompatible=args.allow_incompatible,
        )
    if args.command == "synthetic":
        if args.seed < 0:
            raise ValueError("seed must be non-negative")
        kinds = ("flow", "depth") if args.kind == "both" else (args.kind,)
        results = {}
        for kind in kinds:
            snapshot = _synthetic_snapshot(kind, args.output_dir, args.seed)
            results[kind] = render_snapshot(
                snapshot,
                args.output_dir / kind,
            )
        return {"schema": REPORT_SCHEMA, "synthetic": True, "reports": results}
    raise ValueError(f"unknown visualization command: {args.command}")


def main(argv: Sequence[str] | None = None) -> None:
    result = run(build_parser().parse_args(argv))
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = [
    "COMPARISON_SCHEMA",
    "INDEX_SCHEMA",
    "REPORT_SCHEMA",
    "SNAPSHOT_SCHEMA",
    "build_parser",
    "compare_reports",
    "event_image_rgb",
    "extract_snapshot_events",
    "flow_rgb",
    "iwe_counts",
    "load_snapshot",
    "main",
    "png_bytes",
    "render_snapshot",
    "run",
    "safe_component",
    "write_png",
    "write_snapshot",
    "write_snapshot_index",
]
