from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import numpy as np


M3ED_LABEL_SUFFIXES = {
    "depth": "_depth_gt.h5",
    "semantics": "_semantics.h5",
    "pose": "_pose_gt.h5",
}
M3ED_CALIBRATION_GROUPS = (
    "prophesee/left/calib",
    "prophesee/right/calib",
    "ovc/left/calib",
    "ovc/right/calib",
    "ovc/rgb/calib",
    "ovc/imu/calib",
    "ouster/calib",
)


def discover_m3ed_labels(data_path: str | Path) -> dict[str, Path]:
    """Return available official/derived label files beside one M3ED data HDF5."""

    path = Path(data_path).expanduser().resolve()
    sequence_name = path.stem.removesuffix("_data")
    labels: dict[str, Path] = {}
    for kind, suffix in M3ED_LABEL_SUFFIXES.items():
        candidate = path.with_name(f"{sequence_name}{suffix}")
        if candidate.is_file():
            labels[kind] = candidate
    return labels


def copy_file_atomic(source: str | Path, destination: str | Path) -> None:
    """Copy a large artifact atomically and make identical reruns inexpensive."""

    source_path = Path(source).expanduser().resolve()
    destination_path = Path(destination).expanduser().resolve()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    source_stat = source_path.stat()
    if destination_path.is_file():
        destination_stat = destination_path.stat()
        if (
            destination_stat.st_size == source_stat.st_size
            and destination_stat.st_mtime_ns == source_stat.st_mtime_ns
        ):
            return
    temporary = destination_path.with_name(
        f".{destination_path.name}.{uuid.uuid4().hex}.partial"
    )
    try:
        with source_path.open("rb") as source_handle, temporary.open(
            "xb"
        ) as target_handle:
            shutil.copyfileobj(
                source_handle, target_handle, length=16 * 1024 * 1024
            )
            target_handle.flush()
            os.fsync(target_handle.fileno())
        shutil.copystat(source_path, temporary)
        os.replace(temporary, destination_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _require_hdf5() -> Any:
    try:
        import hdf5plugin  # noqa: F401
        import h5py
    except ImportError as error:
        raise ImportError(
            "install event-window-jepa[hdf5] to inspect M3ED labels"
        ) from error
    return h5py


def _frame_geometry(dataset: Any, path: Path) -> tuple[int, int, int]:
    shape = tuple(int(value) for value in dataset.shape)
    if len(shape) not in {3, 4} or (len(shape) == 4 and shape[-1] != 1):
        raise ValueError(
            f"M3ED frame dataset must have shape [N,H,W] or [N,H,W,1]: {path}"
        )
    if min(shape[:3]) <= 0:
        raise ValueError(f"M3ED frame dataset is empty: {path}")
    return shape[0], shape[1], shape[2]


def _monotonic_timestamp_bounds(dataset: Any, path: Path) -> tuple[int, int]:
    """Validate one M3ED timestamp vector without loading it all into memory."""

    if dataset.ndim != 1 or not np.issubdtype(dataset.dtype, np.integer):
        raise TypeError(
            f"M3ED label /ts must be one-dimensional integers: {path}"
        )
    if len(dataset) == 0:
        raise ValueError(f"M3ED label /ts cannot be empty: {path}")

    first: int | None = None
    previous: int | None = None
    for start in range(0, len(dataset), 1_000_000):
        stop = min(start + 1_000_000, len(dataset))
        values = np.asarray(dataset[start:stop])
        if np.any(values < 0) or np.any(values[1:] < values[:-1]):
            raise ValueError(f"M3ED label timestamps are invalid: {path}")
        current_first = int(values[0])
        if previous is not None and current_first < previous:
            raise ValueError(f"M3ED label timestamps are invalid: {path}")
        if first is None:
            first = current_first
        previous = int(values[-1])

    assert first is not None and previous is not None
    return first, previous


def _timestamp_metadata(handle: Any, frame_count: int, path: Path) -> dict[str, Any]:
    if "ts" not in handle:
        raise KeyError(f"M3ED label HDF5 has no /ts dataset: {path}")
    timestamps = handle["ts"]
    if len(timestamps) != frame_count:
        raise ValueError(
            f"M3ED label frame count and /ts length disagree: {path}"
        )
    first, last = _monotonic_timestamp_bounds(timestamps, path)
    return {
        "timestamp_dataset": "/ts",
        "first_timestamp_us": first,
        "last_timestamp_us": last,
        "timestamp_reference": "M3ED synchronized global clock (microseconds)",
    }


def inspect_m3ed_label(
    path: str | Path, *, kind: str, event_camera: str
) -> dict[str, Any]:
    """Validate an M3ED label container without decoding all image frames."""

    label_path = Path(path).expanduser().resolve()
    if kind not in M3ED_LABEL_SUFFIXES:
        raise ValueError(f"unsupported M3ED label kind: {kind}")
    h5py = _require_hdf5()
    with h5py.File(label_path, "r") as handle:
        if kind == "depth":
            preferred = f"depth/prophesee/{event_camera}"
            dataset_name = preferred if preferred in handle else "depth/prophesee/left"
            if dataset_name not in handle:
                raise KeyError(
                    f"M3ED depth HDF5 has no /{preferred} or /depth/prophesee/left: "
                    f"{label_path}"
                )
            frames = handle[dataset_name]
            frame_count, height, width = _frame_geometry(frames, label_path)
            if not np.issubdtype(frames.dtype, np.floating):
                raise TypeError(f"M3ED depth frames must use a floating dtype: {label_path}")
            camera = dataset_name.rsplit("/", 1)[-1]
            return {
                "depth_dataset": f"/{dataset_name}",
                "depth_frame_count": frame_count,
                "depth_height": height,
                "depth_width": width,
                "depth_camera": camera,
                "depth_matches_event_camera": camera == event_camera,
                "depth_spatial_downsample": 1,
                "depth_invalid_policy": (
                    "non-finite or <=0 is invalid; >200m is excluded by F3 evaluation"
                ),
                **{
                    f"depth_{key}": value
                    for key, value in _timestamp_metadata(
                        handle, frame_count, label_path
                    ).items()
                },
            }

        if kind == "semantics":
            if "predictions" not in handle:
                raise KeyError(
                    f"M3ED semantics HDF5 has no /predictions dataset: {label_path}"
                )
            frames = handle["predictions"]
            frame_count, height, width = _frame_geometry(frames, label_path)
            if not np.issubdtype(frames.dtype, np.integer):
                raise TypeError(
                    f"M3ED semantic predictions must use an integer dtype: {label_path}"
                )
            return {
                "semantics_dataset": "/predictions",
                "semantics_frame_count": frame_count,
                "semantics_height": height,
                "semantics_width": width,
                "semantics_camera": "left",
                "semantics_matches_event_camera": event_camera == "left",
                "semantics_spatial_downsample": 1,
                "semantics_ignore_index": 255,
                "semantics_is_pseudo_label": True,
                "semantics_source_model": "InternImage",
                **{
                    f"semantics_{key}": value
                    for key, value in _timestamp_metadata(
                        handle, frame_count, label_path
                    ).items()
                },
            }

        dataset_paths: list[str] = []

        def collect(name: str, value: Any) -> None:
            if isinstance(value, h5py.Dataset):
                dataset_paths.append(f"/{name}")

        handle.visititems(collect)
        if not dataset_paths:
            raise ValueError(f"M3ED pose HDF5 has no datasets: {label_path}")
        metadata: dict[str, Any] = {
            "pose_dataset_paths": sorted(dataset_paths),
            "pose_coordinate_system": "official M3ED pose convention",
        }
        if "ts" in handle:
            timestamps = handle["ts"]
            if len(timestamps):
                first, last = _monotonic_timestamp_bounds(
                    timestamps, label_path
                )
                metadata.update(
                    {
                        "pose_timestamp_dataset": "/ts",
                        "pose_count": len(timestamps),
                        "pose_first_timestamp_us": first,
                        "pose_last_timestamp_us": last,
                        "pose_timestamp_reference": (
                            "M3ED synchronized global clock (microseconds)"
                        ),
                    }
                )
        return metadata


def copy_m3ed_calibration(
    data_path: str | Path, output_root: str | Path
) -> dict[str, Any]:
    """Extract only compact calibration groups from a large M3ED data HDF5."""

    source_path = Path(data_path).expanduser().resolve()
    destination_root = Path(output_root).expanduser().resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    sequence_name = source_path.stem.removesuffix("_data")
    destination = destination_root / f"{sequence_name}_calibration.h5"
    source_stat = source_path.stat()
    h5py = _require_hdf5()
    if destination.is_file():
        try:
            with h5py.File(destination, "r") as existing:
                if (
                    bool(existing.attrs.get("complete", False))
                    and int(existing.attrs.get("source_file_size", -1))
                    == source_stat.st_size
                    and int(existing.attrs.get("source_mtime_ns", -1))
                    == source_stat.st_mtime_ns
                ):
                    groups = [
                        group
                        for group in M3ED_CALIBRATION_GROUPS
                        if group in existing
                    ]
                    return {
                        "calibration_path": str(destination),
                        "calibration_groups": groups,
                        "calibration_file_size": destination.stat().st_size,
                    }
        except OSError:
            pass

    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.partial"
    )
    try:
        with h5py.File(source_path, "r") as source, h5py.File(
            temporary, "x"
        ) as target:
            groups = [group for group in M3ED_CALIBRATION_GROUPS if group in source]
            if not groups:
                raise ValueError(
                    f"M3ED data HDF5 has no supported calibration groups: {source_path}"
                )
            target.attrs.update(
                {
                    "schema_name": "event-window-jepa-m3ed-calibration",
                    "schema_version": 1,
                    "complete": False,
                    "source_sequence_name": sequence_name,
                    "source_file_size": source_stat.st_size,
                    "source_mtime_ns": source_stat.st_mtime_ns,
                }
            )
            for name in ("creation_date", "raw_bag_name", "version"):
                if name in source.attrs:
                    target.attrs[f"m3ed_{name}"] = source.attrs[name]
            for group in groups:
                parent_name, leaf = group.rsplit("/", 1)
                parent = target.require_group(parent_name)
                source.copy(source[group], parent, name=leaf)
            target.attrs["complete"] = True
            target.flush()
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "calibration_path": str(destination),
        "calibration_groups": groups,
        "calibration_file_size": destination.stat().st_size,
    }


def copy_m3ed_labels(
    data_path: str | Path,
    output_root: str | Path,
    *,
    event_camera: str,
) -> dict[str, Any]:
    """Validate and copy all available M3ED labels into a portable bundle."""

    destination_root = Path(output_root).expanduser().resolve()
    metadata: dict[str, Any] = {
        "has_depth": False,
        "has_semantics": False,
        "has_pose": False,
    }
    for kind, source in discover_m3ed_labels(data_path).items():
        details = inspect_m3ed_label(
            source, kind=kind, event_camera=event_camera
        )
        destination = destination_root / kind / source.name
        copy_file_atomic(source, destination)
        metadata[f"has_{kind}"] = True
        metadata[f"{kind}_path"] = str(destination)
        metadata[f"{kind}_file_size"] = destination.stat().st_size
        metadata.update(details)
    return metadata
