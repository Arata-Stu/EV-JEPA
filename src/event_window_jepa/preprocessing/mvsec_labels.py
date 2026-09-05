from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import struct
import zipfile
from pathlib import Path
from typing import Any

import numpy as np


MVSEC_SENSOR_SIZE = (260, 346)
MVSEC_FLOW_FORMAT = "mvsec_gt_flow_npz_v1"
MVSEC_EMBEDDED_FLOW_FORMAT = "mvsec_embedded_hdf5_flow_dist"
MVSEC_FLOW_SOURCE_METADATA_VERSION = 1
# The official download page links to the Flow GT root folder. The day-driving
# files live one level below it; preserve that parent/child relationship in
# manifest provenance so it agrees with the downloader sidecar.
MVSEC_FLOW_PARENT_FOLDER_ID = "1XS0AQTuCwUaWOmtjyJWRHkbXjj_igJLp"
MVSEC_FLOW_DAY_FOLDER_ID = "1BSzN0E8Kcp_08ezRbWyAbjZqiEdjtp3Q"
MVSEC_OFFICIAL_FLOW_ARTIFACTS: dict[str, tuple[str, int]] = {
    "outdoor_day1_gt_flow_dist.npz": (
        "1XjJnriPh3k0FJo11or7X02myWARqVt7S",
        7_389_716_086,
    ),
    "outdoor_day2_gt_flow_dist.npz": (
        "1RIP-Fp0s7z9QtJTbsyqn_EMEiNwA7l1Y",
        17_555_972_270,
    ),
}
MVSEC_OFFICIAL_GT_ARTIFACTS: dict[str, tuple[str, int]] = {
    "outdoor_day1_gt.hdf5": (
        "1wzUmTBxQ5wtSpB0KBogliB2IGTrCtJ7e",
        6_267_453_264,
    ),
    "outdoor_day2_gt.hdf5": (
        "1zWOA92-Bw4xz1y5CzIROXWFymTFFwBBH",
        17_790_444_169,
    ),
    "outdoor_night1_gt.hdf5": (
        "139dZNXHNUtSul0ZLmPu6N39gbvciQZme",
        10_005_821_320,
    ),
}


def matching_mvsec_ground_truth(data_path: str | Path) -> Path:
    """Return the official sibling ``*_gt.hdf5`` path for an MVSEC data file."""

    path = Path(data_path).expanduser().resolve()
    if path.name.endswith("_data.hdf5"):
        name = path.name.removesuffix("_data.hdf5") + "_gt.hdf5"
    elif path.name.endswith("_data.h5"):
        name = path.name.removesuffix("_data.h5") + "_gt.h5"
    else:
        raise ValueError("MVSEC source filename must end in _data.hdf5 or _data.h5")
    result = path.with_name(name)
    if not result.is_file():
        raise FileNotFoundError(
            f"MVSEC ground truth must be the sibling file {result.name}: {result}"
        )
    return result


def matching_mvsec_flow_ground_truth(data_path: str | Path) -> Path:
    """Return the official sibling ``*_gt_flow_dist.npz`` path."""

    path = Path(data_path).expanduser().resolve()
    if path.name.endswith("_data.hdf5"):
        stem = path.name.removesuffix("_data.hdf5")
    elif path.name.endswith("_data.h5"):
        stem = path.name.removesuffix("_data.h5")
    else:
        raise ValueError("MVSEC source filename must end in _data.hdf5 or _data.h5")
    result = path.with_name(stem + "_gt_flow_dist.npz")
    if not result.is_file():
        raise FileNotFoundError(
            f"official MVSEC flow GT must be the sibling file {result.name}: {result}"
        )
    return result


def _read_npy_header(
    stream: Any, *, member_name: str
) -> tuple[tuple[int, ...], np.dtype[Any], bool, int]:
    if stream.read(6) != b"\x93NUMPY":
        raise ValueError(f"NPZ member is not an NPY array: {member_name}")
    version = stream.read(2)
    if len(version) != 2:
        raise ValueError(f"truncated NPY version: {member_name}")
    major, minor = version
    if (major, minor) == (1, 0):
        length_bytes = stream.read(2)
        if len(length_bytes) != 2:
            raise ValueError(f"truncated NPY header length: {member_name}")
        header_length = struct.unpack("<H", length_bytes)[0]
        length_field_bytes = 2
        encoding = "latin1"
    elif (major, minor) in {(2, 0), (3, 0)}:
        length_bytes = stream.read(4)
        if len(length_bytes) != 4:
            raise ValueError(f"truncated NPY header length: {member_name}")
        header_length = struct.unpack("<I", length_bytes)[0]
        length_field_bytes = 4
        encoding = "utf-8" if major == 3 else "latin1"
    else:
        raise ValueError(f"unsupported NPY version {major}.{minor}: {member_name}")
    if not 0 < header_length <= 1_000_000:
        raise ValueError(f"invalid NPY header length: {member_name}")
    header_bytes = stream.read(header_length)
    if len(header_bytes) != header_length:
        raise ValueError(f"truncated NPY header: {member_name}")
    try:
        header = ast.literal_eval(header_bytes.decode(encoding).strip())
    except (SyntaxError, UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"invalid NPY header: {member_name}") from error
    if not isinstance(header, dict) or set(header) != {
        "descr",
        "fortran_order",
        "shape",
    }:
        raise ValueError(f"unexpected NPY header fields: {member_name}")
    shape = header["shape"]
    descriptor = header["descr"]
    fortran_order = header["fortran_order"]
    if (
        not isinstance(shape, tuple)
        or any(not isinstance(value, int) or value <= 0 for value in shape)
        or not isinstance(descriptor, str)
        or not isinstance(fortran_order, bool)
    ):
        raise ValueError(f"invalid NPY array metadata: {member_name}")
    try:
        dtype = np.dtype(descriptor)
    except TypeError as error:
        raise TypeError(f"invalid NPY dtype: {member_name}") from error
    if dtype.hasobject or not np.issubdtype(dtype, np.number):
        raise TypeError(f"MVSEC flow NPZ member must be numeric: {member_name}")
    return shape, dtype, fortran_order, 8 + length_field_bytes + header_length


def _inspect_flow_npz_headers(path: Path) -> dict[str, tuple[tuple[int, ...], np.dtype[Any]]]:
    expected_members = {
        "timestamps.npy",
        "x_flow_dist.npy",
        "y_flow_dist.npy",
    }
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            if len(names) != len(set(names)) or set(names) != expected_members:
                raise ValueError(
                    "MVSEC flow NPZ must contain exactly timestamps, x_flow_dist, "
                    f"and y_flow_dist; got {sorted(names)}"
                )
            result: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {}
            for name in sorted(expected_members):
                member = archive.getinfo(name)
                if member.compress_type != zipfile.ZIP_STORED:
                    raise ValueError(
                        f"official MVSEC flow member must be ZIP_STORED: {name}"
                    )
                with archive.open(member, "r") as stream:
                    shape, dtype, fortran_order, data_offset = _read_npy_header(
                        stream, member_name=name
                    )
                if fortran_order:
                    raise ValueError(f"MVSEC flow member must be C-contiguous: {name}")
                if member.file_size != data_offset + math.prod(shape) * dtype.itemsize:
                    raise ValueError(f"NPY member size does not match header: {name}")
                result[name] = shape, dtype
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError(f"invalid MVSEC flow NPZ: {path}") from error
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _flow_source_identity(
    path: Path, *, file_id: str, expected_bytes: int
) -> dict[str, object]:
    stat_result = path.stat()
    if stat_result.st_size != expected_bytes:
        raise ValueError(
            f"official MVSEC flow GT size mismatch for {path.name}: "
            f"expected {expected_bytes}, got {stat_result.st_size}"
        )
    digest: str | None = None
    digest_origin = "computed_during_preprocessing"
    sidecar = path.with_name(path.name + ".verified.json")
    if sidecar.is_file():
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict):
            candidate = payload.get("sha256")
            required = {
                "metadata_version": MVSEC_FLOW_SOURCE_METADATA_VERSION,
                "status": "verified",
                "filename": path.name,
                "kind": "flow_npz",
                "file_id": file_id,
                "expected_bytes": expected_bytes,
                "size_bytes": stat_result.st_size,
                "mtime_ns": stat_result.st_mtime_ns,
                "publisher_checksum_available": False,
            }
            if (
                all(payload.get(key) == value for key, value in required.items())
                and isinstance(candidate, str)
                and re.fullmatch(r"[0-9a-f]{64}", candidate) is not None
            ):
                digest = candidate
                digest_origin = "download_verified_sidecar"
    if digest is None:
        digest = _sha256(path)
    return {
        "flow_source_metadata_version": MVSEC_FLOW_SOURCE_METADATA_VERSION,
        "flow_source_release": "official_mvsec_generated_optical_flow_gt",
        "flow_source_folder_id": MVSEC_FLOW_DAY_FOLDER_ID,
        "flow_source_parent_folder_id": MVSEC_FLOW_PARENT_FOLDER_ID,
        "flow_source_file_id": file_id,
        "flow_source_expected_bytes": expected_bytes,
        "flow_source_size_bytes": stat_result.st_size,
        "flow_source_mtime_ns": stat_result.st_mtime_ns,
        "flow_source_sha256": digest,
        "flow_source_sha256_origin": digest_origin,
        "flow_publisher_checksum_available": False,
    }


def _gt_source_identity(path: Path) -> dict[str, object]:
    """Preserve a verified downloader identity without rehashing a large GT file."""

    stat_result = path.stat()
    result: dict[str, object] = {
        "mvsec_gt_source_size_bytes": stat_result.st_size,
        "mvsec_gt_source_mtime_ns": stat_result.st_mtime_ns,
    }
    official = MVSEC_OFFICIAL_GT_ARTIFACTS.get(path.name)
    if official is None:
        return result
    file_id, expected_bytes = official
    sidecar = path.with_name(path.name + ".verified.json")
    if not sidecar.is_file():
        return result
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return result
    if not isinstance(payload, dict):
        return result
    digest = payload.get("sha256")
    required = {
        "metadata_version": MVSEC_FLOW_SOURCE_METADATA_VERSION,
        "status": "verified",
        "filename": path.name,
        "kind": "gt_hdf5",
        "file_id": file_id,
        "expected_bytes": expected_bytes,
        "size_bytes": stat_result.st_size,
        "mtime_ns": stat_result.st_mtime_ns,
        "publisher_checksum_available": False,
    }
    if (
        any(payload.get(key) != value for key, value in required.items())
        or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
    ):
        return result
    result.update(
        {
            "mvsec_gt_source_file_id": file_id,
            "mvsec_gt_source_expected_bytes": expected_bytes,
            "mvsec_gt_source_sha256": digest,
            "mvsec_gt_source_sha256_origin": "download_verified_sidecar",
        }
    )
    return result


def inspect_mvsec_flow_ground_truth(path: str | Path) -> dict[str, object]:
    """Validate the official generated left-camera distorted-flow NPZ."""

    flow_path = Path(path).expanduser().resolve()
    try:
        file_id, expected_bytes = MVSEC_OFFICIAL_FLOW_ARTIFACTS[flow_path.name]
    except KeyError as error:
        raise ValueError(
            f"unsupported official MVSEC flow filename: {flow_path.name}"
        ) from error
    headers = _inspect_flow_npz_headers(flow_path)
    timestamp_shape, timestamp_dtype = headers["timestamps.npy"]
    x_shape, x_dtype = headers["x_flow_dist.npy"]
    y_shape, y_dtype = headers["y_flow_dist.npy"]
    if len(timestamp_shape) != 1:
        raise ValueError("MVSEC flow timestamps must have shape [N]")
    expected_shape = (timestamp_shape[0], *MVSEC_SENSOR_SIZE)
    if x_shape != expected_shape or y_shape != expected_shape:
        raise ValueError(
            "MVSEC x/y flow must both have shape [N,260,346] matching timestamps"
        )
    if not np.issubdtype(timestamp_dtype, np.floating):
        raise TypeError("MVSEC flow timestamps must be floating-point seconds")
    if not np.issubdtype(x_dtype, np.floating) or x_dtype != y_dtype:
        raise TypeError("MVSEC x/y flow arrays must have one shared floating dtype")
    with np.load(flow_path, allow_pickle=False) as archive:
        timestamps = np.asarray(archive["timestamps"], dtype=np.float64)
    if (
        timestamps.ndim != 1
        or len(timestamps) != timestamp_shape[0]
        or not bool(np.isfinite(timestamps).all())
        or np.any(timestamps[1:] < timestamps[:-1])
    ):
        raise ValueError("MVSEC flow timestamps are invalid or decreasing")
    return {
        "flow_path": str(flow_path),
        "flow_format": MVSEC_FLOW_FORMAT,
        "flow_dataset": "x_flow_dist,y_flow_dist",
        "flow_x_key": "x_flow_dist",
        "flow_y_key": "y_flow_dist",
        "flow_timestamp_dataset": "timestamps",
        "flow_timestamp_key": "timestamps",
        "flow_count": len(timestamps),
        "flow_shape": [len(timestamps), *MVSEC_SENSOR_SIZE],
        "flow_dtype": x_dtype.str,
        "flow_first_timestamp_seconds": float(timestamps[0]),
        "flow_last_timestamp_seconds": float(timestamps[-1]),
        "flow_coordinate_frame": "distorted",
        "flow_channel_order": "x,y",
        "flow_semantics": (
            "previous pose to current pose; timestamp is current pose; "
            "first interval is zero"
        ),
        "flow_timestamp_reference": "MVSEC synchronized recording clock (seconds)",
        "flow_timestamps_relative": False,
        **_flow_source_identity(
            flow_path, file_id=file_id, expected_bytes=expected_bytes
        ),
    }


def _require_hdf5() -> Any:
    try:
        import h5py
    except ImportError as error:
        raise ImportError("install event-window-jepa[hdf5] to inspect MVSEC labels") from error
    return h5py


def _validate_timestamps(dataset: Any, expected_count: int, name: str) -> tuple[float, float]:
    if dataset.ndim != 1 or len(dataset) != expected_count:
        raise ValueError(f"MVSEC {name} must be one-dimensional with {expected_count} entries")
    if not np.issubdtype(dataset.dtype, np.number):
        raise TypeError(f"MVSEC {name} must be numeric seconds")
    first: float | None = None
    previous: float | None = None
    for start in range(0, len(dataset), 1_000_000):
        values = np.asarray(dataset[start : min(start + 1_000_000, len(dataset))])
        if not bool(np.isfinite(values).all()) or np.any(values[1:] < values[:-1]):
            raise ValueError(f"MVSEC {name} contains invalid or decreasing timestamps")
        if previous is not None and len(values) and float(values[0]) < previous:
            raise ValueError(f"MVSEC {name} decreases across chunks")
        if len(values):
            first = float(values[0]) if first is None else first
            previous = float(values[-1])
    if first is None or previous is None:
        raise ValueError(f"MVSEC {name} cannot be empty")
    return first, previous


def _inspect_map_pair(
    handle: Any,
    *,
    data_name: str,
    timestamp_name: str,
    kind: str,
) -> dict[str, object] | None:
    has_data = data_name in handle
    has_timestamps = timestamp_name in handle
    if has_data != has_timestamps:
        raise ValueError(
            f"MVSEC GT contains only one of /{data_name} and /{timestamp_name}"
        )
    if not has_data:
        return None
    data = handle[data_name]
    if kind == "flow":
        expected_tail = (2, *MVSEC_SENSOR_SIZE)
        if data.ndim != 4 or tuple(data.shape[1:]) != expected_tail:
            raise ValueError(
                f"MVSEC /{data_name} must have shape [N,2,260,346], got {data.shape}"
            )
    else:
        if data.ndim != 3 or tuple(data.shape[1:]) != MVSEC_SENSOR_SIZE:
            raise ValueError(
                f"MVSEC /{data_name} must have shape [N,260,346], got {data.shape}"
            )
    if len(data) <= 0 or not np.issubdtype(data.dtype, np.number):
        raise TypeError(f"MVSEC /{data_name} must be a non-empty numeric dataset")
    first, last = _validate_timestamps(handle[timestamp_name], len(data), timestamp_name)
    return {
        f"{kind}_dataset": f"/{data_name}",
        f"{kind}_timestamp_dataset": f"/{timestamp_name}",
        f"{kind}_count": len(data),
        f"{kind}_first_timestamp_seconds": first,
        f"{kind}_last_timestamp_seconds": last,
        f"{kind}_coordinate_frame": "distorted",
        f"{kind}_timestamp_reference": "MVSEC synchronized recording clock (seconds)",
        f"{kind}_timestamps_relative": False,
    }


def inspect_mvsec_ground_truth(
    path: str | Path,
    *,
    camera: str = "left",
    include_embedded_flow: bool = False,
) -> dict[str, object]:
    """Validate native depth/pose HDF5 and, only if requested, embedded flow."""

    if camera not in {"left", "right"}:
        raise ValueError("MVSEC camera must be left or right")
    gt_path = Path(path).expanduser().resolve()
    h5py = _require_hdf5()
    base = f"davis/{camera}"
    result: dict[str, object] = {
        "mvsec_gt_path": str(gt_path),
        "mvsec_gt_format": "mvsec_depth_pose_hdf5_v1",
        **_gt_source_identity(gt_path),
    }
    with h5py.File(gt_path, "r") as handle:
        depth = _inspect_map_pair(
            handle,
            data_name=f"{base}/depth_image_raw",
            timestamp_name=f"{base}/depth_image_raw_ts",
            kind="depth",
        )
        if depth is not None:
            result.update(depth)
            result["depth_path"] = str(gt_path)

        if include_embedded_flow:
            flow = _inspect_map_pair(
                handle,
                data_name=f"{base}/flow_dist",
                timestamp_name=f"{base}/flow_dist_ts",
                kind="flow",
            )
            if flow is None:
                raise ValueError(
                    f"MVSEC GT has no explicitly requested embedded flow for camera={camera}"
                )
            result.update(flow)
            result.update(
                {
                    "flow_path": str(gt_path),
                    "flow_format": MVSEC_EMBEDDED_FLOW_FORMAT,
                    "flow_channel_order": "x,y",
                    "flow_semantics": (
                        "alternate embedded flow_dist; verify interval semantics at source"
                    ),
                }
            )

        pose_pair: tuple[str, str] | None = None
        for pose_leaf in ("pose", "odometry"):
            candidate = f"{base}/{pose_leaf}"
            timestamp_candidate = f"{candidate}_ts"
            if candidate in handle or timestamp_candidate in handle:
                if candidate not in handle or timestamp_candidate not in handle:
                    raise ValueError(
                        f"MVSEC GT contains an incomplete /{candidate} pose pair"
                    )
                pose_pair = candidate, timestamp_candidate
                break
        if pose_pair is not None:
            pose_name, pose_timestamp_name = pose_pair
            pose = handle[pose_name]
            if pose.ndim < 2 or len(pose) <= 0 or not np.issubdtype(pose.dtype, np.number):
                raise TypeError(f"MVSEC /{pose_name} must be a non-empty numeric array")
            first, last = _validate_timestamps(
                handle[pose_timestamp_name], len(pose), pose_timestamp_name
            )
            result.update(
                {
                    "pose_path": str(gt_path),
                    "pose_dataset": f"/{pose_name}",
                    "pose_timestamp_dataset": f"/{pose_timestamp_name}",
                    "pose_count": len(pose),
                    "pose_first_timestamp_seconds": first,
                    "pose_last_timestamp_seconds": last,
                    "pose_timestamp_reference": (
                        "MVSEC synchronized recording clock (seconds)"
                    ),
                    "pose_timestamps_relative": False,
                }
            )

    if not any(key in result for key in ("depth_path", "pose_path", "flow_path")):
        raise ValueError(f"MVSEC GT has no raw depth or pose for camera={camera}")
    return result
