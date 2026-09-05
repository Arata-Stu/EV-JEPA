#!/usr/bin/env python3
"""Safely download the official MVSEC HDF5 and flow-GT releases.

Planning and local validation use only the Python standard library.  ``gdown``
is imported only when a missing file actually needs to be transferred, and
``h5py`` is used for schema validation only when it is already installed.

The MVSEC publisher does not publish cryptographic checksums for these Google
Drive objects.  The pinned Drive file IDs and exact byte counts therefore
identify the intended objects, while the locally generated SHA-256 sidecars
only protect the local cache after its first successful verification.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import importlib
import json
import math
import os
import re
import shutil
import struct
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Iterator, Sequence


HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
HASH_CHUNK_BYTES = 8 * 1024 * 1024
FREE_SPACE_MARGIN_BYTES = 1024**3
METADATA_VERSION = 1
OFFICIAL_DOWNLOAD_PAGE = "https://daniilidis-group.github.io/mvsec/download/"
OFFICIAL_HDF5_FOLDER_ID = "1rwyRk26wtWeRgrAx_fgPc-ubUzTFThkV"
OFFICIAL_FLOW_FOLDER_ID = "1XS0AQTuCwUaWOmtjyJWRHkbXjj_igJLp"
OFFICIAL_FLOW_DAY_FOLDER_ID = "1BSzN0E8Kcp_08ezRbWyAbjZqiEdjtp3Q"


class MVSECDownloadError(RuntimeError):
    """A safe-download invariant was not satisfied."""


@dataclass(frozen=True)
class Artifact:
    scene: str
    filename: str
    file_id: str
    expected_bytes: int
    kind: str

    @property
    def relative_path(self) -> Path:
        return Path("raw") / self.scene / self.filename


ARTIFACTS: dict[str, Artifact] = {
    "outdoor_day_calib": Artifact(
        scene="outdoor_day",
        filename="outdoor_day_calib.zip",
        file_id="1Y0sPP0ebX_cEKUCVVhJLej9TZgCxS3ME",
        expected_bytes=1_290_038,
        kind="calibration_zip",
    ),
    "outdoor_day1_data": Artifact(
        scene="outdoor_day",
        filename="outdoor_day1_data.hdf5",
        file_id="1JLIrw2L24zIQBmqaWvef7G2t9tsMY3H0",
        expected_bytes=12_136_423_279,
        kind="data_hdf5",
    ),
    "outdoor_day1_gt": Artifact(
        scene="outdoor_day",
        filename="outdoor_day1_gt.hdf5",
        file_id="1wzUmTBxQ5wtSpB0KBogliB2IGTrCtJ7e",
        expected_bytes=6_267_453_264,
        kind="gt_hdf5",
    ),
    "outdoor_day1_flow": Artifact(
        scene="outdoor_day",
        filename="outdoor_day1_gt_flow_dist.npz",
        file_id="1XjJnriPh3k0FJo11or7X02myWARqVt7S",
        expected_bytes=7_389_716_086,
        kind="flow_npz",
    ),
    "outdoor_day2_data": Artifact(
        scene="outdoor_day",
        filename="outdoor_day2_data.hdf5",
        file_id="1fu9GhjYcET00mMN-YbAp3eBK1YMCd3Ox",
        expected_bytes=41_506_282_485,
        kind="data_hdf5",
    ),
    "outdoor_day2_gt": Artifact(
        scene="outdoor_day",
        filename="outdoor_day2_gt.hdf5",
        file_id="1zWOA92-Bw4xz1y5CzIROXWFymTFFwBBH",
        expected_bytes=17_790_444_169,
        kind="gt_hdf5",
    ),
    "outdoor_day2_flow": Artifact(
        scene="outdoor_day",
        filename="outdoor_day2_gt_flow_dist.npz",
        file_id="1RIP-Fp0s7z9QtJTbsyqn_EMEiNwA7l1Y",
        expected_bytes=17_555_972_270,
        kind="flow_npz",
    ),
    "outdoor_night_calib": Artifact(
        scene="outdoor_night",
        filename="outdoor_night_calib.zip",
        file_id="1NGUBQ8b41b9murJualNeaO7M3nrIPjfm",
        expected_bytes=1_292_482,
        kind="calibration_zip",
    ),
    "outdoor_night1_data": Artifact(
        scene="outdoor_night",
        filename="outdoor_night1_data.hdf5",
        file_id="1z8b00gWoZnGuzAOSD49KFaX03q1UuKxc",
        expected_bytes=9_575_081_506,
        kind="data_hdf5",
    ),
    "outdoor_night1_gt": Artifact(
        scene="outdoor_night",
        filename="outdoor_night1_gt.hdf5",
        file_id="139dZNXHNUtSul0ZLmPu6N39gbvciQZme",
        expected_bytes=10_005_821_320,
        kind="gt_hdf5",
    ),
}

PROFILE_KEYS: dict[str, tuple[str, ...]] = {
    "stage1": (
        "outdoor_day1_data",
        "outdoor_day1_gt",
        "outdoor_day1_flow",
        "outdoor_day2_data",
        "outdoor_day2_gt",
        "outdoor_day2_flow",
    ),
    "stage1-ood": (
        "outdoor_day1_data",
        "outdoor_day1_gt",
        "outdoor_day1_flow",
        "outdoor_day2_data",
        "outdoor_day2_gt",
        "outdoor_day2_flow",
        "outdoor_night1_data",
        "outdoor_night1_gt",
    ),
}

PROFILE_TOTALS = {
    "stage1": 102_646_291_553,
    "stage1-ood": 122_227_194_379,
}


def artifacts_for_profile(profile: str, include_calibration: bool) -> tuple[Artifact, ...]:
    try:
        keys = list(PROFILE_KEYS[profile])
    except KeyError as error:
        raise ValueError(f"unknown MVSEC profile: {profile}") from error
    if include_calibration:
        keys.append("outdoor_day_calib")
        if profile == "stage1-ood":
            keys.append("outdoor_night_calib")
    artifacts = tuple(ARTIFACTS[key] for key in keys)
    names = [artifact.filename.casefold() for artifact in artifacts]
    ids = [artifact.file_id for artifact in artifacts]
    if len(names) != len(set(names)) or len(ids) != len(set(ids)):
        raise AssertionError("MVSEC profile contains a duplicate filename or Drive ID")
    if (
        not include_calibration
        and sum(artifact.expected_bytes for artifact in artifacts)
        != PROFILE_TOTALS[profile]
    ):
        raise AssertionError("MVSEC pinned profile total is internally inconsistent")
    return artifacts


def profile_total_bytes(profile: str, include_calibration: bool = False) -> int:
    return sum(
        artifact.expected_bytes
        for artifact in artifacts_for_profile(profile, include_calibration)
    )


def validate_root(raw_root: str | os.PathLike[str]) -> Path:
    raw_text = os.fspath(raw_root).strip()
    if not raw_text:
        raise ValueError("--root must not be empty")
    candidate = Path(raw_text).expanduser()
    if not candidate.is_absolute():
        raise ValueError("--root must be an absolute, dedicated dataset directory")
    root = candidate.resolve(strict=False)
    anchor = Path(root.anchor).resolve(strict=False)
    home = Path.home().resolve(strict=False)
    cwd = Path.cwd().resolve(strict=False)
    if root == anchor:
        raise ValueError("refusing to use the filesystem root as --root")
    if root == home:
        raise ValueError("refusing to use the home directory itself as --root")
    if root == cwd or root in cwd.parents:
        raise ValueError("refusing to use the workspace or one of its parents as --root")
    if len(root.parts) < 3:
        raise ValueError("--root is too broad; choose a dedicated MVSEC subdirectory")
    if root.exists() and not root.is_dir():
        raise ValueError(f"--root is not a directory: {root}")
    return root


def output_path(root: Path, artifact: Artifact) -> Path:
    resolved_root = root.resolve(strict=False)
    result = (resolved_root / artifact.relative_path).resolve(strict=False)
    try:
        result.relative_to(resolved_root)
    except ValueError as error:
        raise MVSECDownloadError(f"unsafe artifact path escaped --root: {result}") from error
    return result


def _part_path(final_path: Path) -> Path:
    return final_path.with_name(final_path.name + ".part")


def _gdown_partial_candidates(part_path: Path) -> list[Path]:
    if not part_path.parent.is_dir():
        return []
    return sorted(
        candidate
        for candidate in part_path.parent.iterdir()
        if candidate != part_path
        and candidate.is_file()
        and candidate.name.startswith(part_path.name)
        and candidate.name.endswith(".part")
    )


def _partial_progress(final_path: Path, expected_bytes: int) -> int:
    if final_path.exists():
        if not final_path.is_file():
            raise MVSECDownloadError(f"output exists but is not a file: {final_path}")
        actual = final_path.stat().st_size
        if actual != expected_bytes:
            raise MVSECDownloadError(
                f"existing output has wrong size: {final_path} "
                f"(expected {expected_bytes}, got {actual})"
            )
        return expected_bytes

    part_path = _part_path(final_path)
    candidates = _gdown_partial_candidates(part_path)
    if part_path.exists():
        if not part_path.is_file():
            raise MVSECDownloadError(f"partial output is not a file: {part_path}")
        if candidates:
            raise MVSECDownloadError(
                f"both completed and resumable partials exist for {final_path.name}; "
                "retain them and resolve the ambiguity manually"
            )
        progress = part_path.stat().st_size
    elif len(candidates) == 1:
        progress = candidates[0].stat().st_size
    elif len(candidates) > 1:
        raise MVSECDownloadError(
            f"multiple gdown partials exist for {final_path.name}; retain them and "
            "resolve the ambiguity manually"
        )
    else:
        progress = 0
    if progress > expected_bytes:
        raise MVSECDownloadError(
            f"partial file exceeds the expected size for {final_path.name}: "
            f"{progress} > {expected_bytes}"
        )
    return progress


def remaining_download_bytes(root: Path, artifacts: Sequence[Artifact]) -> int:
    remaining = 0
    for artifact in artifacts:
        final_path = output_path(root, artifact)
        progress = _partial_progress(final_path, artifact.expected_bytes)
        remaining += artifact.expected_bytes - progress
    return remaining


def _nearest_existing_directory(path: Path) -> Path:
    current = path
    while not current.exists():
        parent = current.parent
        if parent == current:
            raise MVSECDownloadError(f"cannot find an existing parent for {path}")
        current = parent
    return current if current.is_dir() else current.parent


def preflight_disk_space(root: Path, artifacts: Sequence[Artifact]) -> tuple[int, int]:
    remaining = remaining_download_bytes(root, artifacts)
    if remaining == 0:
        return 0, shutil.disk_usage(_nearest_existing_directory(root)).free
    probe = _nearest_existing_directory(root)
    available = shutil.disk_usage(probe).free
    required = remaining + FREE_SPACE_MARGIN_BYTES
    if available < required:
        raise MVSECDownloadError(
            "insufficient free disk space: "
            f"need {required} bytes ({remaining} download + "
            f"{FREE_SPACE_MARGIN_BYTES} safety margin), have {available} at {probe}"
        )
    return remaining, available


def _read_magic(path: Path) -> bytes:
    with path.open("rb") as handle:
        return handle.read(len(HDF5_MAGIC))


def _load_h5py_if_available() -> ModuleType | None:
    try:
        return importlib.import_module("h5py")
    except ModuleNotFoundError as error:
        if error.name == "h5py":
            return None
        raise


def _require_hdf5_dataset(handle: object, name: str) -> object:
    if name not in handle:  # type: ignore[operator]
        raise MVSECDownloadError(f"MVSEC HDF5 is missing required dataset /{name}")
    dataset = handle[name]  # type: ignore[index]
    if not hasattr(dataset, "shape") or not hasattr(dataset, "dtype"):
        raise MVSECDownloadError(f"MVSEC HDF5 path /{name} is not a dataset")
    return dataset


def _numeric_dataset(dataset: object, name: str) -> None:
    dtype = getattr(dataset, "dtype")
    if getattr(dtype, "kind", "") not in "biufc":
        raise MVSECDownloadError(f"MVSEC HDF5 dataset /{name} is not numeric")


def _validate_timestamp_pair(data: object, timestamps: object, name: str) -> None:
    data_shape = tuple(getattr(data, "shape"))
    timestamp_shape = tuple(getattr(timestamps, "shape"))
    if not data_shape or data_shape[0] <= 0:
        raise MVSECDownloadError(f"MVSEC HDF5 dataset /{name} is empty")
    if len(timestamp_shape) != 1 or timestamp_shape[0] != data_shape[0]:
        raise MVSECDownloadError(
            f"MVSEC timestamps for /{name} do not match its leading dimension"
        )
    _numeric_dataset(data, name)
    _numeric_dataset(timestamps, name + "_ts")
    first = float(timestamps[0])  # type: ignore[index]
    last = float(timestamps[-1])  # type: ignore[index]
    if not math.isfinite(first) or not math.isfinite(last) or last < first:
        raise MVSECDownloadError(f"MVSEC timestamps for /{name} have invalid endpoints")


def _validate_data_hdf5(handle: object) -> None:
    for camera in ("left", "right"):
        name = f"davis/{camera}/events"
        events = _require_hdf5_dataset(handle, name)
        shape = tuple(getattr(events, "shape"))
        if len(shape) != 2 or shape[0] <= 0 or shape[1] != 4:
            raise MVSECDownloadError(
                f"MVSEC /{name} must have shape [N,4], got {shape}"
            )
        _numeric_dataset(events, name)
        first_timestamp = float(events[0, 2])  # type: ignore[index]
        last_timestamp = float(events[-1, 2])  # type: ignore[index]
        if (
            not math.isfinite(first_timestamp)
            or not math.isfinite(last_timestamp)
            or last_timestamp < first_timestamp
        ):
            raise MVSECDownloadError(f"MVSEC /{name} has invalid timestamp endpoints")

    images = _require_hdf5_dataset(handle, "davis/left/image_raw")
    image_timestamps = _require_hdf5_dataset(handle, "davis/left/image_raw_ts")
    image_event_indices = _require_hdf5_dataset(
        handle, "davis/left/image_raw_event_inds"
    )
    _validate_timestamp_pair(images, image_timestamps, "davis/left/image_raw")
    image_count = tuple(getattr(images, "shape"))[0]
    index_shape = tuple(getattr(image_event_indices, "shape"))
    if len(index_shape) != 1 or index_shape[0] != image_count:
        raise MVSECDownloadError(
            "MVSEC /davis/left/image_raw_event_inds does not match image count"
        )
    _numeric_dataset(image_event_indices, "davis/left/image_raw_event_inds")


def _validate_gt_hdf5(handle: object) -> None:
    base = "davis/left"
    for leaf in ("depth_image_raw", "depth_image_rect"):
        name = f"{base}/{leaf}"
        data = _require_hdf5_dataset(handle, name)
        timestamps = _require_hdf5_dataset(handle, name + "_ts")
        shape = tuple(getattr(data, "shape"))
        if len(shape) != 3 or tuple(shape[1:]) != (260, 346):
            raise MVSECDownloadError(
                f"MVSEC /{name} must have shape [N,260,346], got {shape}"
            )
        _validate_timestamp_pair(data, timestamps, name)

    pose_pairs = []
    for leaf in ("pose", "odometry"):
        name = f"{base}/{leaf}"
        timestamp_name = name + "_ts"
        has_data = name in handle  # type: ignore[operator]
        has_timestamps = timestamp_name in handle  # type: ignore[operator]
        if has_data != has_timestamps:
            raise MVSECDownloadError(f"MVSEC GT has an incomplete /{name} pair")
        if has_data:
            data = _require_hdf5_dataset(handle, name)
            timestamps = _require_hdf5_dataset(handle, timestamp_name)
            _validate_timestamp_pair(data, timestamps, name)
            pose_pairs.append(name)
    if not pose_pairs:
        raise MVSECDownloadError("MVSEC GT contains neither pose nor odometry")

    flow_name = f"{base}/flow_dist"
    flow_timestamp_name = flow_name + "_ts"
    has_flow = flow_name in handle  # type: ignore[operator]
    has_flow_timestamps = flow_timestamp_name in handle  # type: ignore[operator]
    if has_flow != has_flow_timestamps:
        raise MVSECDownloadError("MVSEC GT has an incomplete flow_dist pair")
    if has_flow:
        flow = _require_hdf5_dataset(handle, flow_name)
        timestamps = _require_hdf5_dataset(handle, flow_timestamp_name)
        shape = tuple(getattr(flow, "shape"))
        if len(shape) != 4 or tuple(shape[1:]) != (2, 260, 346):
            raise MVSECDownloadError(
                f"MVSEC /{flow_name} must have shape [N,2,260,346], got {shape}"
            )
        _validate_timestamp_pair(flow, timestamps, flow_name)


def _validate_hdf5(path: Path, kind: str) -> str:
    if _read_magic(path) != HDF5_MAGIC:
        raise MVSECDownloadError(f"not an HDF5 file (bad signature): {path}")
    h5py = _load_h5py_if_available()
    if h5py is None:
        return "magic-only (h5py unavailable)"
    try:
        with h5py.File(path, "r") as handle:
            if kind == "data_hdf5":
                _validate_data_hdf5(handle)
            elif kind == "gt_hdf5":
                _validate_gt_hdf5(handle)
            else:
                raise AssertionError(f"unexpected HDF5 artifact kind: {kind}")
    except MVSECDownloadError:
        raise
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise MVSECDownloadError(f"cannot validate MVSEC HDF5 schema: {path}") from error
    return "h5py-schema"


def _validate_calibration_zip(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as archive:
            if not archive.infolist():
                raise MVSECDownloadError(f"empty calibration ZIP: {path}")
            for member in archive.infolist():
                name = member.filename.replace("\\", "/")
                pure = PurePosixPath(name)
                if pure.is_absolute() or ".." in pure.parts:
                    raise MVSECDownloadError(
                        f"unsafe path in calibration ZIP {path.name}: {member.filename}"
                    )
            bad_member = archive.testzip()
            if bad_member is not None:
                raise MVSECDownloadError(
                    f"CRC failure in calibration ZIP {path.name}: {bad_member}"
                )
    except MVSECDownloadError:
        raise
    except (OSError, zipfile.BadZipFile) as error:
        raise MVSECDownloadError(f"invalid calibration ZIP: {path}") from error
    return "zip-crc"


def _read_npy_header(
    stream: object, *, member_name: str
) -> tuple[tuple[int, ...], str, int]:
    """Read an NPY header without importing NumPy or materializing its array."""

    read = getattr(stream, "read")
    if read(6) != b"\x93NUMPY":
        raise MVSECDownloadError(f"NPZ member is not an NPY array: {member_name}")
    version = read(2)
    if len(version) != 2:
        raise MVSECDownloadError(f"truncated NPY version: {member_name}")
    major, minor = version
    if (major, minor) == (1, 0):
        length_bytes = read(2)
        if len(length_bytes) != 2:
            raise MVSECDownloadError(f"truncated NPY header length: {member_name}")
        header_length = struct.unpack("<H", length_bytes)[0]
        encoding = "latin1"
    elif (major, minor) in {(2, 0), (3, 0)}:
        length_bytes = read(4)
        if len(length_bytes) != 4:
            raise MVSECDownloadError(f"truncated NPY header length: {member_name}")
        header_length = struct.unpack("<I", length_bytes)[0]
        encoding = "utf-8" if major == 3 else "latin1"
    else:
        raise MVSECDownloadError(
            f"unsupported NPY version {major}.{minor}: {member_name}"
        )
    if not 0 < header_length <= 1_000_000:
        raise MVSECDownloadError(f"invalid NPY header length: {member_name}")
    header_bytes = read(header_length)
    if len(header_bytes) != header_length:
        raise MVSECDownloadError(f"truncated NPY header: {member_name}")
    try:
        header = ast.literal_eval(header_bytes.decode(encoding).strip())
    except (SyntaxError, UnicodeDecodeError, ValueError) as error:
        raise MVSECDownloadError(f"invalid NPY header: {member_name}") from error
    if not isinstance(header, dict) or set(header) != {
        "descr",
        "fortran_order",
        "shape",
    }:
        raise MVSECDownloadError(f"unexpected NPY header fields: {member_name}")
    shape = header["shape"]
    descriptor = header["descr"]
    if (
        not isinstance(shape, tuple)
        or any(not isinstance(value, int) or value <= 0 for value in shape)
        or not isinstance(descriptor, str)
        or not isinstance(header["fortran_order"], bool)
    ):
        raise MVSECDownloadError(f"invalid NPY array metadata: {member_name}")
    if re.fullmatch(r"[<>=|]?[fiu][1248]", descriptor) is None:
        raise MVSECDownloadError(
            f"MVSEC flow NPZ member must use a numeric scalar dtype: {member_name}"
        )
    if header["fortran_order"]:
        raise MVSECDownloadError(
            f"official MVSEC flow NPZ member must use C order: {member_name}"
        )
    length_field_bytes = 2 if major == 1 else 4
    return shape, descriptor, 8 + length_field_bytes + header_length


def _validate_flow_npz(path: Path) -> str:
    expected_members = {
        "timestamps.npy",
        "x_flow_dist.npy",
        "y_flow_dist.npy",
    }
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            if len(names) != len(set(names)):
                raise MVSECDownloadError(f"duplicate member in flow NPZ: {path}")
            if set(names) != expected_members:
                raise MVSECDownloadError(
                    "MVSEC flow NPZ must contain exactly timestamps, x_flow_dist, "
                    f"and y_flow_dist; got {sorted(names)}"
                )
            headers: dict[str, tuple[tuple[int, ...], str, int]] = {}
            for name in sorted(expected_members):
                member = archive.getinfo(name)
                # np.savez (used by the official generator) stores NPY members
                # without compression.  Requiring ZIP_STORED lets downstream
                # readers memory-map x/y instead of allocating multi-GiB arrays.
                if member.compress_type != zipfile.ZIP_STORED:
                    raise MVSECDownloadError(
                        f"official MVSEC flow member is unexpectedly compressed: {name}"
                    )
                with archive.open(member, "r") as stream:
                    headers[name] = _read_npy_header(stream, member_name=name)
                shape, descriptor, data_offset = headers[name]
                item_size = int(descriptor[-1])
                expected_member_bytes = data_offset + math.prod(shape) * item_size
                if member.file_size != expected_member_bytes:
                    raise MVSECDownloadError(
                        f"NPY member size does not match its header: {name}"
                    )
            bad_member = archive.testzip()
            if bad_member is not None:
                raise MVSECDownloadError(
                    f"CRC failure in MVSEC flow NPZ member: {bad_member}"
                )
    except MVSECDownloadError:
        raise
    except (OSError, KeyError, zipfile.BadZipFile) as error:
        raise MVSECDownloadError(f"invalid MVSEC flow NPZ: {path}") from error

    timestamp_shape, timestamp_descriptor, _ = headers["timestamps.npy"]
    x_shape, x_descriptor, _ = headers["x_flow_dist.npy"]
    y_shape, y_descriptor, _ = headers["y_flow_dist.npy"]
    if len(timestamp_shape) != 1:
        raise MVSECDownloadError("MVSEC flow timestamps must have shape [N]")
    expected_flow_shape = (timestamp_shape[0], 260, 346)
    if x_shape != expected_flow_shape or y_shape != expected_flow_shape:
        raise MVSECDownloadError(
            "MVSEC x/y flow must both have shape [N,260,346] matching timestamps"
        )
    if (
        "f" not in timestamp_descriptor
        or "f" not in x_descriptor
        or x_descriptor != y_descriptor
    ):
        raise MVSECDownloadError(
            "MVSEC timestamps and x/y flow must use floating dtypes; x/y must match"
        )
    return "npz-stored-npy-headers-crc"


def validate_artifact_file(path: Path, artifact: Artifact) -> str:
    if not path.is_file():
        raise MVSECDownloadError(f"downloaded artifact is not a regular file: {path}")
    actual_bytes = path.stat().st_size
    if actual_bytes != artifact.expected_bytes:
        raise MVSECDownloadError(
            f"size mismatch for {artifact.filename}: expected {artifact.expected_bytes}, "
            f"got {actual_bytes}"
        )
    if artifact.kind in {"data_hdf5", "gt_hdf5"}:
        return _validate_hdf5(path, artifact.kind)
    if artifact.kind == "flow_npz":
        return _validate_flow_npz(path)
    if artifact.kind == "calibration_zip":
        return _validate_calibration_zip(path)
    raise AssertionError(f"unknown MVSEC artifact kind: {artifact.kind}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(HASH_CHUNK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _sidecar_path(path: Path) -> Path:
    return path.with_name(path.name + ".verified.json")


def _cached_sha256(path: Path, artifact: Artifact) -> str | None:
    sidecar = _sidecar_path(path)
    if not sidecar.is_file():
        return None
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    stat_result = path.stat()
    expected = {
        "metadata_version": METADATA_VERSION,
        "status": "verified",
        "file_id": artifact.file_id,
        "expected_bytes": artifact.expected_bytes,
        "size_bytes": stat_result.st_size,
        "mtime_ns": stat_result.st_mtime_ns,
        "publisher_checksum_available": False,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        return None
    digest = payload.get("sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        return None
    return digest


def _write_sidecar(
    path: Path,
    artifact: Artifact,
    digest: str,
    validation: str,
) -> None:
    stat_result = path.stat()
    if artifact.kind == "flow_npz":
        source = "official MVSEC generated optical-flow GT release"
        source_folder_id = OFFICIAL_FLOW_DAY_FOLDER_ID
        source_parent_folder_id = OFFICIAL_FLOW_FOLDER_ID
    else:
        source = "official MVSEC HDF5/calibration release"
        source_folder_id = OFFICIAL_HDF5_FOLDER_ID
        source_parent_folder_id = None
    payload = {
        "metadata_version": METADATA_VERSION,
        "status": "verified",
        "source": source,
        "official_download_page": OFFICIAL_DOWNLOAD_PAGE,
        "source_folder_id": source_folder_id,
        "source_parent_folder_id": source_parent_folder_id,
        "file_id": artifact.file_id,
        "filename": artifact.filename,
        "kind": artifact.kind,
        "expected_bytes": artifact.expected_bytes,
        "size_bytes": stat_result.st_size,
        "mtime_ns": stat_result.st_mtime_ns,
        "sha256": digest,
        "publisher_checksum": None,
        "publisher_checksum_available": False,
        "publisher_checksum_note": (
            "MVSEC publishes no cryptographic checksum for this object; "
            "SHA-256 is a local cache identity only"
        ),
        "content_validation": validation,
        "verified_unix_ns": time.time_ns(),
    }
    sidecar = _sidecar_path(path)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{sidecar.name}.", suffix=".part", dir=sidecar.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, sidecar)
    finally:
        temporary.unlink(missing_ok=True)


def verify_and_record(path: Path, artifact: Artifact) -> str:
    validation = validate_artifact_file(path, artifact)
    digest = _cached_sha256(path, artifact)
    if digest is None:
        digest = sha256_file(path)
    _write_sidecar(path, artifact, digest, validation)
    return digest


@contextlib.contextmanager
def _output_lock(final_path: Path) -> Iterator[None]:
    lock_path = final_path.with_name(final_path.name + ".lock")
    try:
        lock_path.mkdir()
    except FileExistsError as error:
        raise MVSECDownloadError(
            f"output is locked: {final_path}; if no downloader is running, remove "
            f"only the stale lock directory {lock_path}"
        ) from error
    try:
        yield
    finally:
        try:
            lock_path.rmdir()
        except OSError:
            pass


def _prepare_resumable_partial(part_path: Path, expected_bytes: int) -> None:
    candidates = _gdown_partial_candidates(part_path)
    if part_path.exists():
        size = part_path.stat().st_size
        if size > expected_bytes:
            raise MVSECDownloadError(
                f"partial file exceeds expected size: {part_path} ({size})"
            )
        if size < expected_bytes:
            if candidates:
                raise MVSECDownloadError(
                    f"ambiguous partial state for {part_path}; files were retained"
                )
            resumable = part_path.with_name(part_path.name + ".resume.part")
            if resumable.exists():
                raise MVSECDownloadError(
                    f"resumable partial already exists: {resumable}; files were retained"
                )
            os.replace(part_path, resumable)
    elif len(candidates) > 1:
        raise MVSECDownloadError(
            f"multiple gdown partials exist for {part_path.name}; files were retained"
        )
    elif len(candidates) == 1:
        candidate_size = candidates[0].stat().st_size
        if candidate_size > expected_bytes:
            raise MVSECDownloadError(
                f"gdown partial exceeds expected size: {candidates[0]}"
            )
        if candidate_size == expected_bytes:
            os.replace(candidates[0], part_path)


def _load_gdown() -> ModuleType:
    try:
        return importlib.import_module("gdown")
    except ModuleNotFoundError as error:
        if error.name != "gdown":
            raise
        raise MVSECDownloadError(
            "gdown is required for an actual MVSEC transfer; install the project's "
            "download extra, for example: python -m pip install -e '.[download]'"
        ) from error


def _download_one(root: Path, artifact: Artifact, use_cookies: bool) -> None:
    final_path = output_path(root, artifact)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    with _output_lock(final_path):
        if final_path.exists():
            digest = verify_and_record(final_path, artifact)
            print(f"ready: {final_path} (sha256={digest})")
            return

        part_path = _part_path(final_path)
        _prepare_resumable_partial(part_path, artifact.expected_bytes)
        if not part_path.exists():
            gdown = _load_gdown()
            print(f"downloading/resuming: {artifact.filename}", file=sys.stderr)
            try:
                result = gdown.download(
                    id=artifact.file_id,
                    output=str(part_path),
                    quiet=False,
                    resume=True,
                    verify=True,
                    use_cookies=use_cookies,
                )
            except Exception as error:
                raise MVSECDownloadError(
                    f"gdown failed for {artifact.filename}; partial data was retained"
                ) from error
            if result is None:
                raise MVSECDownloadError(
                    f"gdown returned no output for {artifact.filename}; partial data was retained"
                )
            returned_path = Path(os.fspath(result)).resolve(strict=False)
            if returned_path != part_path.resolve(strict=False):
                raise MVSECDownloadError(
                    f"gdown returned an unexpected output path: {returned_path}"
                )

        validation = validate_artifact_file(part_path, artifact)
        digest = sha256_file(part_path)
        os.replace(part_path, final_path)
        _write_sidecar(final_path, artifact, digest, validation)
        print(f"ready: {final_path} (sha256={digest})")


def _format_gib(size_bytes: int) -> str:
    return f"{size_bytes / 1024**3:.3f} GiB"


def print_plan(
    root: Path,
    profile: str,
    artifacts: Sequence[Artifact],
    include_calibration: bool,
) -> None:
    total = sum(artifact.expected_bytes for artifact in artifacts)
    print("MVSEC download plan")
    print(f"profile: {profile}")
    print(f"root: {root}")
    print(f"include_calibration: {str(include_calibration).lower()}")
    for artifact in artifacts:
        print(
            f"- {artifact.relative_path.as_posix()} "
            f"id={artifact.file_id} bytes={artifact.expected_bytes}"
        )
    print(f"total_bytes: {total}")
    print(f"total_size: {_format_gib(total)}")
    print("publisher_checksum: unavailable")
    print("local_sha256: generated after byte and content validation")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download pinned official MVSEC HDF5 and generated flow-GT NPZ files "
            "safely. Planning requires only the Python standard library."
        )
    )
    parser.add_argument(
        "--root",
        required=True,
        help="absolute dedicated dataset root; files are stored below ROOT/raw",
    )
    parser.add_argument(
        "--profile",
        required=True,
        choices=tuple(PROFILE_KEYS),
        help=(
            "stage1 is outdoor day1/2 data, depth/pose GT, and flow GT; "
            "stage1-ood additionally includes night1 data and depth/pose GT"
        ),
    )
    parser.add_argument(
        "--include-calibration",
        action="store_true",
        help="also download the official calibration ZIP for each selected scene",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="print the exact file/size plan without network access or filesystem writes",
    )
    parser.add_argument(
        "--use-cookies",
        action="store_true",
        help=(
            "allow gdown's cookie cache; disabled by default because publisher files "
            "are public and cookie files are sensitive"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        root = validate_root(args.root)
        artifacts = artifacts_for_profile(args.profile, args.include_calibration)
    except ValueError as error:
        parser.error(str(error))
    assert isinstance(root, Path)

    if args.plan_only:
        print_plan(root, args.profile, artifacts, args.include_calibration)
        return 0

    print_plan(root, args.profile, artifacts, args.include_calibration)
    print(
        "warning: MVSEC publishes no checksum for these objects; generated SHA-256 "
        "values are local cache identities only.",
        file=sys.stderr,
    )
    remaining, available = preflight_disk_space(root, artifacts)
    print(
        f"disk preflight: remaining={remaining} available={available} "
        f"margin={FREE_SPACE_MARGIN_BYTES}",
        file=sys.stderr,
    )
    for artifact in artifacts:
        _download_one(root, artifact, args.use_cookies)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MVSECDownloadError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
