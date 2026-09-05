from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
import zipfile
import zlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from event_window_jepa.data.event_store import H5EventStore
from event_window_jepa.data.spatial_transforms import (
    SharedRandomSpatialTransform,
    SpatialTransformParameters,
)
from event_window_jepa.models.recurrent_vjepa21_event_vit import (
    RecurrentVJEPA21EventVisionTransformer,
)
from event_window_jepa.representations.event_image import EventImage
from event_window_jepa.representations.voxel_grid import VoxelGrid


MVSEC_SENSOR_SIZE = (260, 346)
MVSEC_CAR_HOOD_START_ROW = 193
MVSEC_F3_CONTEXT_WINDOW_US = 50_000
EVFLOWNET_TEST_START_US = 222_400_000
EVFLOWNET_TEST_STOP_US = 240_400_000
_OFFICIAL_FLOW_NPZ_KEYS = ("timestamps", "x_flow_dist", "y_flow_dist")

FlowNPZIdentity = tuple[Path, int, int, int, int, int]
_CRC_VERIFIED_FLOW_NPZ: set[FlowNPZIdentity] = set()

TargetKind = Literal["flow", "depth"]
Alignment = Literal["causal", "f3_centered"]


@dataclass(frozen=True)
class MVSECGeometrySource:
    sequence_id: str
    ground_truth_path: Path
    target_dataset: str
    timestamp_dataset: str
    source_time_origin_us: int
    t_start_us: int
    t_end_us: int
    camera: str
    target_format: str = "hdf5"
    target_file_id: str | None = None
    target_size_bytes: int | None = None
    target_sha256: str | None = None
    target_sha256_origin: str | None = None
    target_mtime_ns: int | None = None


@dataclass(frozen=True)
class MVSECTargetReference:
    source_index: int
    target_index: int
    label_timestamp_us: int
    event_window_end_us: int
    flow_interval_us: int


@dataclass(frozen=True)
class MVSECOfficialFlowNPZ:
    """Memory-mapped views of the official raw/distorted left-camera flow."""

    path: Path
    timestamps_seconds: np.memmap
    x_flow_dist: np.memmap
    y_flow_dist: np.memmap

    def close(self) -> None:
        for array in (
            self.timestamps_seconds,
            self.x_flow_dist,
            self.y_flow_dist,
        ):
            memory_map = getattr(array, "_mmap", None)
            if memory_map is not None:
                memory_map.close()


def _flow_npz_identity(path: Path) -> FlowNPZIdentity:
    """Return a process-independent identity suitable for worker hand-off."""

    stat_result = path.stat()
    return (
        path,
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
        stat_result.st_ctime_ns,
    )


def _stored_npz_member_memmap(path: Path, key: str) -> np.memmap:
    """Map an uncompressed NPY member without materializing the whole NPZ."""

    member_name = f"{key}.npy"
    with zipfile.ZipFile(path, "r") as archive:
        try:
            member = archive.getinfo(member_name)
        except KeyError as error:
            raise KeyError(f"official MVSEC flow NPZ lacks {key!r}") from error
        if member.compress_type != zipfile.ZIP_STORED:
            raise ValueError(
                "MVSEC flow NPZ members must be uncompressed (official np.savez "
                "format) for bounded-memory random access"
            )
        if member.flag_bits & 0x1:
            raise ValueError("encrypted MVSEC flow NPZ members are unsupported")
        local_header_offset = member.header_offset
        member_size = member.file_size

    with path.open("rb") as handle:
        handle.seek(local_header_offset)
        local_header = handle.read(30)
        if len(local_header) != 30 or local_header[:4] != b"PK\x03\x04":
            raise ValueError("MVSEC flow NPZ has an invalid local ZIP header")
        local_flags = struct.unpack_from("<H", local_header, 6)[0]
        local_compression = struct.unpack_from("<H", local_header, 8)[0]
        local_crc = struct.unpack_from("<I", local_header, 14)[0]
        if local_flags != member.flag_bits or local_compression != member.compress_type:
            raise ValueError("MVSEC flow NPZ ZIP headers disagree")
        if local_crc not in {0, member.CRC}:
            raise ValueError("MVSEC flow NPZ local and central CRC metadata disagree")
        filename_length, extra_length = struct.unpack_from("<HH", local_header, 26)
        local_filename = handle.read(filename_length)
        expected_filename = member_name.encode("utf-8")
        if local_filename != expected_filename:
            raise ValueError("MVSEC flow NPZ member filename header is inconsistent")
        npy_offset = local_header_offset + 30 + filename_length + extra_length
        handle.seek(npy_offset)
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
                handle
            )
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(
                handle
            )
        else:
            raise ValueError(
                f"MVSEC flow NPZ member {key!r} uses unsupported NPY {version}"
            )
        data_offset = handle.tell()

    dtype = np.dtype(dtype)
    if dtype.hasobject:
        raise TypeError("MVSEC flow NPZ arrays cannot contain Python objects")
    element_count = math.prod(shape)
    expected_member_size = data_offset - npy_offset + element_count * dtype.itemsize
    if expected_member_size != member_size:
        raise ValueError(f"MVSEC flow NPZ member {key!r} has an invalid byte size")
    return np.memmap(
        path,
        mode="r",
        dtype=dtype,
        offset=data_offset,
        shape=shape,
        order="F" if fortran_order else "C",
    )


def _verify_official_flow_npz_crc(path: Path) -> None:
    """Stream each required member once and verify its central-directory CRC."""

    identity = _flow_npz_identity(path)
    if identity in _CRC_VERIFIED_FLOW_NPZ:
        return
    try:
        with zipfile.ZipFile(path, "r") as archive:
            for key in _OFFICIAL_FLOW_NPZ_KEYS:
                member = archive.getinfo(f"{key}.npy")
                checksum = 0
                byte_count = 0
                with archive.open(member, "r") as handle:
                    while block := handle.read(8 * 1024 * 1024):
                        checksum = zlib.crc32(block, checksum)
                        byte_count += len(block)
                if byte_count != member.file_size or checksum != member.CRC:
                    raise ValueError(
                        f"MVSEC flow NPZ member {key!r} failed CRC validation"
                    )
    except zipfile.BadZipFile as error:
        raise ValueError("MVSEC flow NPZ failed ZIP CRC validation") from error
    if _flow_npz_identity(path) != identity:
        raise RuntimeError("MVSEC flow NPZ changed while it was being validated")
    _CRC_VERIFIED_FLOW_NPZ.add(identity)


def open_mvsec_official_flow_npz(
    path: str | Path,
    *,
    verify_crc: bool = True,
) -> MVSECOfficialFlowNPZ:
    """Open and strictly validate official MVSEC dense flow without copying it."""

    resolved = Path(path).expanduser().resolve()
    if resolved.suffix.lower() != ".npz" or not resolved.is_file():
        raise ValueError("official MVSEC dense flow must be an existing .npz file")
    with np.load(resolved, allow_pickle=False) as archive:
        required = set(_OFFICIAL_FLOW_NPZ_KEYS)
        actual = set(archive.files)
        missing = required - actual
        if missing:
            raise KeyError(f"official MVSEC flow NPZ lacks keys {sorted(missing)}")
        unexpected = actual - required
        if unexpected:
            raise ValueError(
                f"official MVSEC flow NPZ has unexpected keys {sorted(unexpected)}"
            )
    with zipfile.ZipFile(resolved, "r") as archive:
        member_names = [member.filename for member in archive.infolist()]
        expected_members = {f"{key}.npy" for key in _OFFICIAL_FLOW_NPZ_KEYS}
        if set(member_names) != expected_members:
            raise ValueError("MVSEC flow NPZ contains unexpected ZIP members")
        for key in _OFFICIAL_FLOW_NPZ_KEYS:
            if member_names.count(f"{key}.npy") != 1:
                raise ValueError(
                    f"MVSEC flow NPZ must contain exactly one {key!r} member"
                )
    arrays: list[np.memmap] = []
    try:
        timestamps = _stored_npz_member_memmap(resolved, "timestamps")
        arrays.append(timestamps)
        x_flow = _stored_npz_member_memmap(resolved, "x_flow_dist")
        arrays.append(x_flow)
        y_flow = _stored_npz_member_memmap(resolved, "y_flow_dist")
        arrays.append(y_flow)
        if verify_crc:
            _verify_official_flow_npz_crc(resolved)
        if timestamps.ndim != 1 or not np.issubdtype(
            timestamps.dtype, np.floating
        ):
            raise TypeError(
                "MVSEC flow timestamps must be a floating-point seconds vector"
            )
        expected_shape = (len(timestamps), *MVSEC_SENSOR_SIZE)
        if x_flow.shape != expected_shape or y_flow.shape != expected_shape:
            raise ValueError(
                "MVSEC x_flow_dist/y_flow_dist must have shape [N,260,346]"
            )
        if not np.issubdtype(x_flow.dtype, np.floating) or not np.issubdtype(
            y_flow.dtype, np.floating
        ):
            raise TypeError("MVSEC distorted flow arrays must be floating point")
        if x_flow.dtype != y_flow.dtype:
            raise TypeError("MVSEC x/y distorted flow arrays must share a dtype")
        if not (
            timestamps.flags.c_contiguous
            and x_flow.flags.c_contiguous
            and y_flow.flags.c_contiguous
        ):
            raise ValueError("official MVSEC flow NPZ arrays must use C order")
        previous: float | None = None
        for start in range(0, len(timestamps), 1_000_000):
            values = np.asarray(timestamps[start : start + 1_000_000])
            if not bool(np.isfinite(values).all()) or np.any(
                values[1:] <= values[:-1]
            ):
                raise ValueError(
                    "MVSEC flow timestamps are invalid or not strictly increasing"
                )
            if previous is not None and len(values) and float(values[0]) <= previous:
                raise ValueError("MVSEC flow timestamps do not increase across chunks")
            if len(values):
                previous = float(values[-1])
        if previous is None:
            raise ValueError("MVSEC flow NPZ cannot be empty")
        return MVSECOfficialFlowNPZ(
            path=resolved,
            timestamps_seconds=timestamps,
            x_flow_dist=x_flow,
            y_flow_dist=y_flow,
        )
    except Exception:
        for array in arrays:
            memory_map = getattr(array, "_mmap", None)
            if memory_map is not None:
                memory_map.close()
        raise


def _resolve_path(value: str | Path, parent: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (parent / path).resolve()


def read_mvsec_geometry_sources(
    manifest: str | Path,
    *,
    kind: TargetKind,
    split: str | None = None,
) -> tuple[MVSECGeometrySource, ...]:
    """Read label-bearing MVSEC rows while enforcing native distorted geometry."""

    manifest_path = Path(manifest).expanduser().resolve()
    path_field = f"{kind}_path"
    data_field = f"{kind}_dataset"
    timestamp_field = f"{kind}_timestamp_dataset"
    default_data = f"/davis/left/{'flow_dist' if kind == 'flow' else 'depth_image_raw'}"
    default_timestamps = default_data + "_ts"
    sources: list[MVSECGeometrySource] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row: dict[str, Any] = json.loads(line)
            if split is not None and str(row.get("split", "train")) != split:
                continue
            if str(row.get("dataset", "")) != "mvsec":
                raise ValueError(
                    f"{manifest_path}:{line_number} is not an MVSEC manifest row"
                )
            if path_field not in row:
                continue
            camera = str(row.get("camera", "unknown"))
            if camera != "left":
                raise ValueError("official MVSEC flow/raw-depth evaluation requires left camera")
            geometry = (
                int(row.get("height", -1)),
                int(row.get("width", -1)),
            )
            source_geometry = (
                int(row.get("source_height", geometry[0])),
                int(row.get("source_width", geometry[1])),
            )
            if geometry != MVSEC_SENSOR_SIZE or source_geometry != MVSEC_SENSOR_SIZE:
                raise ValueError("MVSEC geometry labels require native 260x346 events")
            if int(row.get("spatial_downsample", 1)) != 1:
                raise ValueError("MVSEC geometry evaluation rejects spatially downsampled events")
            if str(row.get("coordinate_frame", "")) != "distorted":
                raise ValueError("MVSEC raw-depth/flow evaluation requires distorted events")
            if str(row.get(f"{kind}_coordinate_frame", "distorted")) != "distorted":
                raise ValueError(f"MVSEC {kind} target is not in distorted coordinates")
            if bool(row.get(f"{kind}_timestamps_relative", False)):
                raise ValueError(f"MVSEC {kind} timestamps must remain on the source clock")
            gt_path = _resolve_path(row[path_field], manifest_path.parent)
            if not gt_path.is_file():
                raise FileNotFoundError(gt_path)
            target_format = "mvsec_depth_hdf5"
            target_file_id: str | None = None
            target_size_bytes: int | None = None
            target_sha256: str | None = None
            target_sha256_origin: str | None = None
            target_mtime_ns: int | None = None
            target_dataset = str(row.get(data_field, default_data))
            timestamp_dataset = str(
                row.get(timestamp_field, default_timestamps)
            )
            if kind == "flow":
                target_format = str(row.get("flow_format", ""))
                if target_format == "mvsec_gt_flow_npz_v1":
                    if gt_path.suffix.lower() != ".npz":
                        raise ValueError(
                            "mvsec_gt_flow_npz_v1 requires a .npz flow_path"
                        )
                    target_dataset = str(
                        row.get(data_field, "x_flow_dist,y_flow_dist")
                    )
                    timestamp_dataset = str(
                        row.get(timestamp_field, "timestamps")
                    )
                    if target_dataset != "x_flow_dist,y_flow_dist":
                        raise ValueError(
                            "official MVSEC flow NPZ requires "
                            "flow_dataset='x_flow_dist,y_flow_dist'"
                        )
                    if timestamp_dataset != "timestamps":
                        raise ValueError(
                            "official MVSEC flow NPZ requires "
                            "flow_timestamp_dataset='timestamps'"
                        )
                    declared_keys = {
                        "flow_x_key": "x_flow_dist",
                        "flow_y_key": "y_flow_dist",
                        "flow_timestamp_key": "timestamps",
                        "flow_channel_order": "x,y",
                    }
                    for field_name, expected_value in declared_keys.items():
                        if row.get(field_name) != expected_value:
                            raise ValueError(
                                f"official MVSEC flow NPZ requires "
                                f"{field_name}={expected_value!r}"
                            )
                    source_metadata_version = row.get(
                        "flow_source_metadata_version"
                    )
                    if (
                        isinstance(source_metadata_version, bool)
                        or source_metadata_version != 1
                    ):
                        raise ValueError(
                            "official MVSEC flow source metadata version must be 1"
                        )
                    target_file_id = row.get("flow_source_file_id")
                    if not isinstance(target_file_id, str) or not target_file_id:
                        raise ValueError(
                            "official MVSEC flow manifest requires a source file ID"
                        )
                    expected_bytes = row.get("flow_source_expected_bytes")
                    recorded_bytes = row.get("flow_source_size_bytes")
                    if (
                        isinstance(expected_bytes, bool)
                        or not isinstance(expected_bytes, int)
                        or isinstance(recorded_bytes, bool)
                        or not isinstance(recorded_bytes, int)
                        or expected_bytes <= 0
                        or recorded_bytes != expected_bytes
                        or gt_path.stat().st_size != expected_bytes
                    ):
                        raise ValueError(
                            "official MVSEC flow source byte metadata disagrees "
                            "with flow_path"
                        )
                    target_size_bytes = expected_bytes
                    recorded_mtime_ns = row.get("flow_source_mtime_ns")
                    if (
                        isinstance(recorded_mtime_ns, bool)
                        or not isinstance(recorded_mtime_ns, int)
                        or recorded_mtime_ns <= 0
                        or gt_path.stat().st_mtime_ns != recorded_mtime_ns
                    ):
                        raise ValueError(
                            "official MVSEC flow source mtime metadata disagrees "
                            "with flow_path"
                        )
                    target_mtime_ns = recorded_mtime_ns
                    target_sha256 = row.get("flow_source_sha256")
                    if not isinstance(target_sha256, str) or re.fullmatch(
                        r"[0-9a-f]{64}", target_sha256
                    ) is None:
                        raise ValueError(
                            "official MVSEC flow manifest requires a lowercase "
                            "SHA-256 source identity"
                        )
                    sha256_origin = row.get("flow_source_sha256_origin")
                    if not isinstance(sha256_origin, str) or not sha256_origin:
                        raise ValueError(
                            "official MVSEC flow manifest requires SHA-256 provenance"
                        )
                    target_sha256_origin = sha256_origin
                    flow_npz = open_mvsec_official_flow_npz(gt_path)
                    try:
                        expected_shape = [
                            len(flow_npz.timestamps_seconds),
                            *MVSEC_SENSOR_SIZE,
                        ]
                        if row.get("flow_shape") != expected_shape:
                            raise ValueError(
                                "manifest flow_shape disagrees with official NPZ"
                            )
                        if row.get("flow_dtype") != flow_npz.x_flow_dist.dtype.str:
                            raise ValueError(
                                "manifest flow_dtype disagrees with official NPZ"
                            )
                        if int(row.get("flow_count", -1)) != expected_shape[0]:
                            raise ValueError(
                                "manifest flow_count disagrees with official NPZ"
                            )
                    finally:
                        flow_npz.close()
                elif target_format == "mvsec_embedded_hdf5_flow_dist":
                    if gt_path.suffix.lower() not in {".h5", ".hdf5"}:
                        raise ValueError(
                            "mvsec_embedded_hdf5_flow_dist requires an HDF5 path"
                        )
                else:
                    raise ValueError(
                        "MVSEC flow rows require flow_format="
                        "'mvsec_gt_flow_npz_v1'; the legacy HDF5 fallback must "
                        "be explicitly declared as "
                        "'mvsec_embedded_hdf5_flow_dist'"
                    )
            sources.append(
                MVSECGeometrySource(
                    sequence_id=str(row["sequence_id"]),
                    ground_truth_path=gt_path,
                    target_dataset=target_dataset,
                    timestamp_dataset=timestamp_dataset,
                    source_time_origin_us=int(row["source_time_origin_us"]),
                    t_start_us=int(row["t_start_us"]),
                    t_end_us=int(row["t_end_us"]),
                    camera=camera,
                    target_format=target_format,
                    target_file_id=target_file_id,
                    target_size_bytes=target_size_bytes,
                    target_sha256=target_sha256,
                    target_sha256_origin=target_sha256_origin,
                    target_mtime_ns=target_mtime_ns,
                )
            )
    if not sources:
        requested = "all splits" if split is None else f"split={split!r}"
        raise ValueError(f"manifest has no {kind}-bearing MVSEC rows for {requested}")
    return tuple(sources)


def _ceil_to_step(value: int, step: int) -> int:
    if step <= 0:
        raise ValueError("step must be positive")
    return ((value + step - 1) // step) * step


def _subsample_evenly(
    references: Sequence[MVSECTargetReference], maximum: int
) -> tuple[MVSECTargetReference, ...]:
    if maximum <= 0 or len(references) <= maximum:
        return tuple(references)
    indices = np.linspace(0, len(references) - 1, num=maximum, dtype=np.int64)
    return tuple(references[int(index)] for index in indices)


def temporal_reference_set_sha256(
    sources: Sequence[MVSECGeometrySource],
    references: Sequence[MVSECTargetReference],
) -> str:
    """Hash an ordered temporal target selection and all input-time metadata."""

    digest = hashlib.sha256()
    for reference in references:
        if not 0 <= reference.source_index < len(sources):
            raise IndexError("MVSEC target reference has an invalid source_index")
        source = sources[reference.source_index]
        digest.update(
            (
                f"{source.sequence_id}\0{reference.target_index}\0"
                f"{reference.label_timestamp_us}\0"
                f"{reference.event_window_end_us}\0"
                f"{reference.flow_interval_us}\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()


def split_mvsec_temporal_dev_references(
    sources: Sequence[MVSECGeometrySource],
    references: Sequence[MVSECTargetReference],
    *,
    window_us: int,
    stride_us: int,
    history_steps: int,
    alignment: Alignment,
    dev_fraction: float = 0.2,
    guard_us: int | None = None,
    maximum_train_samples: int = 0,
    maximum_dev_samples: int = 0,
    additional_dependency_interval: (
        Callable[[MVSECTargetReference], tuple[int, int]] | None
    ) = None,
) -> tuple[
    tuple[MVSECTargetReference, ...],
    tuple[MVSECTargetReference, ...],
    dict[str, Any],
]:
    """Create a guarded early-train/late-dev split within one recording.

    Event slices use ``(start, end]`` semantics.  The dependency interval for
    every target is the union of its complete recurrent input history and an
    optional task-specific interval (for example the causal flow-mask support).
    The returned partitions are asserted not to share any input event time.
    """

    if min(window_us, stride_us, history_steps) <= 0:
        raise ValueError("window_us, stride_us, and history_steps must be positive")
    if alignment not in {"causal", "f3_centered"}:
        raise ValueError("alignment must be causal or f3_centered")
    if not math.isfinite(dev_fraction) or not 0.0 < dev_fraction < 1.0:
        raise ValueError("dev_fraction must be finite and strictly between 0 and 1")
    if maximum_train_samples < 0 or maximum_dev_samples < 0:
        raise ValueError("temporal dev sample limits cannot be negative")
    if guard_us is not None and (
        isinstance(guard_us, bool) or not isinstance(guard_us, int) or guard_us <= 0
    ):
        raise ValueError("guard_us must be a positive integer or None for auto")
    if len(references) < 2:
        raise ValueError("temporal dev splitting requires at least two targets")

    sequence_ids: set[str] = set()
    seen_targets: set[tuple[int, int]] = set()
    for reference in references:
        if not 0 <= reference.source_index < len(sources):
            raise IndexError("MVSEC target reference has an invalid source_index")
        sequence_ids.add(sources[reference.source_index].sequence_id)
        target_key = (reference.source_index, reference.target_index)
        if target_key in seen_targets:
            raise ValueError("temporal dev input contains a duplicate target reference")
        seen_targets.add(target_key)
    if len(sequence_ids) != 1:
        raise ValueError(
            "temporal dev splitting requires exactly one label-bearing MVSEC sequence"
        )

    ordered = tuple(
        sorted(
            references,
            key=lambda reference: (
                reference.label_timestamp_us,
                reference.source_index,
                reference.target_index,
            ),
        )
    )
    history_span_us = window_us + (history_steps - 1) * stride_us
    ceil_allowance_us = 999 if alignment == "f3_centered" else 0
    automatic_minimum_guard_us = history_span_us + ceil_allowance_us
    effective_guard_us = (
        automatic_minimum_guard_us if guard_us is None else guard_us
    )
    if effective_guard_us < automatic_minimum_guard_us:
        raise ValueError(
            "dev guard is shorter than the recurrent history plus alignment "
            f"rounding allowance ({effective_guard_us} < "
            f"{automatic_minimum_guard_us} us)"
        )

    def dependency_interval(
        reference: MVSECTargetReference,
    ) -> tuple[int, int]:
        model_start_us = reference.event_window_end_us - history_span_us
        model_end_us = reference.event_window_end_us
        start_us, end_us = model_start_us, model_end_us
        if additional_dependency_interval is not None:
            extra_start_us, extra_end_us = additional_dependency_interval(reference)
            if (
                isinstance(extra_start_us, bool)
                or isinstance(extra_end_us, bool)
                or not isinstance(extra_start_us, int)
                or not isinstance(extra_end_us, int)
                or extra_start_us >= extra_end_us
            ):
                raise ValueError(
                    "additional dependency interval must be integer (start, end]"
                )
            start_us = min(start_us, extra_start_us)
            end_us = max(end_us, extra_end_us)
        if start_us >= end_us:
            raise ValueError("MVSEC input dependency interval must have positive duration")
        return start_us, end_us

    dependencies = {
        reference: dependency_interval(reference) for reference in ordered
    }
    nominal_dev_count = max(1, math.ceil(len(ordered) * dev_fraction))
    nominal_boundary_index = len(ordered) - nominal_dev_count
    if nominal_boundary_index <= 0:
        raise ValueError("dev_fraction leaves no target before the dev partition")
    dev_eligible = ordered[nominal_boundary_index:]
    first_dev_label_us = dev_eligible[0].label_timestamp_us
    dev_dependency_start_us = min(
        dependencies[reference][0] for reference in dev_eligible
    )
    before_boundary = ordered[:nominal_boundary_index]
    label_guard_eligible = tuple(
        reference
        for reference in before_boundary
        if first_dev_label_us - reference.label_timestamp_us >= effective_guard_us
    )
    train_eligible = tuple(
        reference
        for reference in label_guard_eligible
        if dependencies[reference][1] <= dev_dependency_start_us
    )
    if not train_eligible:
        raise ValueError(
            "temporal dev guard leaves no early training target; reduce dev_fraction "
            "or use a longer recording"
        )

    eligible_label_gap_us = (
        first_dev_label_us - train_eligible[-1].label_timestamp_us
    )
    train_dependency_end_us = max(
        dependencies[reference][1] for reference in train_eligible
    )
    eligible_dependency_gap_us = dev_dependency_start_us - train_dependency_end_us
    if eligible_label_gap_us < effective_guard_us:
        raise AssertionError("temporal dev label guard invariant failed")
    if eligible_dependency_gap_us < 0:
        raise AssertionError("temporal dev input dependencies overlap")

    train_selected = _subsample_evenly(train_eligible, maximum_train_samples)
    dev_selected = _subsample_evenly(dev_eligible, maximum_dev_samples)
    if not train_selected or not dev_selected:
        raise AssertionError("temporal dev subsampling produced an empty partition")
    selected_dev_dependency_start_us = min(
        dependencies[reference][0] for reference in dev_selected
    )
    selected_train_dependency_end_us = max(
        dependencies[reference][1] for reference in train_selected
    )
    selected_label_gap_us = (
        dev_selected[0].label_timestamp_us
        - train_selected[-1].label_timestamp_us
    )
    selected_dependency_gap_us = (
        selected_dev_dependency_start_us - selected_train_dependency_end_us
    )
    if selected_label_gap_us < effective_guard_us:
        raise AssertionError("subsampled temporal dev label guard invariant failed")
    if selected_dependency_gap_us < 0:
        raise AssertionError("subsampled temporal dev input dependencies overlap")

    report: dict[str, Any] = {
        "schema_version": 1,
        "method": "chronological_early_train_late_dev_guarded_v1",
        "recording_sequence_id": next(iter(sequence_ids)),
        "requested_dev_fraction": float(dev_fraction),
        "history_span_us": history_span_us,
        "alignment_ceil_allowance_us": ceil_allowance_us,
        "automatic_minimum_guard_us": automatic_minimum_guard_us,
        "requested_guard_us": guard_us,
        "effective_guard_us": effective_guard_us,
        "interval_semantics": "(start_us,end_us]",
        "boundary": {
            "nominal_dev_start_order_index": nominal_boundary_index,
            "first_dev_target_index": dev_eligible[0].target_index,
            "first_dev_label_timestamp_us": first_dev_label_us,
            "last_train_target_index": train_eligible[-1].target_index,
            "last_train_label_timestamp_us": train_eligible[-1].label_timestamp_us,
            "eligible_label_gap_us": eligible_label_gap_us,
            "train_dependency_end_us": train_dependency_end_us,
            "dev_dependency_start_us": dev_dependency_start_us,
            "eligible_dependency_gap_us": eligible_dependency_gap_us,
            "actual_input_dependency_nonoverlap": True,
        },
        "counts": {
            "all_targets": len(ordered),
            "nominal_early_targets_before_guard": len(before_boundary),
            "nominal_late_dev_targets": len(dev_eligible),
            "dropped_by_label_guard": len(before_boundary) - len(label_guard_eligible),
            "dropped_by_dependency_assertion": (
                len(label_guard_eligible) - len(train_eligible)
            ),
            "eligible_train_targets": len(train_eligible),
            "eligible_dev_targets": len(dev_eligible),
            "selected_train_targets": len(train_selected),
            "selected_dev_targets": len(dev_selected),
        },
        "hashes": {
            "all_targets_sha256": temporal_reference_set_sha256(sources, ordered),
            "eligible_train_targets_sha256": temporal_reference_set_sha256(
                sources, train_eligible
            ),
            "eligible_dev_targets_sha256": temporal_reference_set_sha256(
                sources, dev_eligible
            ),
            "selected_train_targets_sha256": temporal_reference_set_sha256(
                sources, train_selected
            ),
            "selected_dev_targets_sha256": temporal_reference_set_sha256(
                sources, dev_selected
            ),
        },
        "selected": {
            "maximum_train_samples": maximum_train_samples,
            "maximum_dev_samples": maximum_dev_samples,
            "label_gap_us": selected_label_gap_us,
            "train_dependency_end_us": selected_train_dependency_end_us,
            "dev_dependency_start_us": selected_dev_dependency_start_us,
            "dependency_gap_us": selected_dependency_gap_us,
            "actual_input_dependency_nonoverlap": True,
        },
        "representation_pretraining_visibility_contract": {
            "protocol_class": (
                "transductive_event_only_representation_pretraining"
            ),
            "recording": "outdoor_day2",
            "event_visibility": (
                "full_recording_including_the_late_dev_time_range"
            ),
            "geometry_labels_visible_to_pretraining": False,
            "late_dev_geometry_labels_visible_to_probe_head_training": False,
        },
    }
    return train_selected, dev_selected, report


def build_mvsec_target_references(
    store: H5EventStore,
    sources: Sequence[MVSECGeometrySource],
    *,
    kind: TargetKind,
    window_us: int,
    stride_us: int,
    history_steps: int,
    alignment: Alignment,
    minimum_events: int,
    f3_evflownet_split: bool = False,
    maximum_samples: int = 0,
) -> tuple[MVSECTargetReference, ...]:
    """Align source-clock GT to event windows without hiding future-event use.

    ``causal`` ends the final event window at the GT timestamp.  The optional
    ``f3_centered`` mode reproduces F3's 50-ms interval centered on that
    timestamp (rounded up to a millisecond); it is intentionally labelled
    non-causal in reports.
    """

    if kind not in {"flow", "depth"}:
        raise ValueError("kind must be flow or depth")
    if min(window_us, stride_us, history_steps) <= 0 or minimum_events < 0:
        raise ValueError("window, stride, and history must be positive")
    if alignment not in {"causal", "f3_centered"}:
        raise ValueError("alignment must be causal or f3_centered")
    if (
        alignment == "f3_centered"
        and window_us != MVSEC_F3_CONTEXT_WINDOW_US
    ):
        raise ValueError(
            "f3_centered reproduces F3's fixed 50-ms MVSEC context and "
            "therefore requires window_us=50000"
        )
    if f3_evflownet_split and kind != "flow":
        raise ValueError("the F3/EvFlowNet test interval applies only to flow")

    all_references: list[MVSECTargetReference] = []
    for source_index, source in enumerate(sources):
        if kind == "flow" and source.target_format == "mvsec_gt_flow_npz_v1":
            flow_npz = open_mvsec_official_flow_npz(source.ground_truth_path)
            try:
                timestamps_seconds = np.array(
                    flow_npz.timestamps_seconds,
                    dtype=np.float64,
                    copy=True,
                )
                target_count = len(flow_npz.x_flow_dist)
            finally:
                flow_npz.close()
        else:
            try:
                import h5py
            except ImportError as error:
                raise ImportError(
                    "install event-window-jepa[hdf5] for MVSEC targets"
                ) from error
            with h5py.File(source.ground_truth_path, "r") as handle:
                if (
                    source.timestamp_dataset not in handle
                    or source.target_dataset not in handle
                ):
                    raise KeyError(
                        f"MVSEC GT lacks {source.target_dataset} or "
                        f"{source.timestamp_dataset}"
                    )
                timestamps_seconds = np.asarray(
                    handle[source.timestamp_dataset], dtype=np.float64
                )
                target_count = len(handle[source.target_dataset])
        if timestamps_seconds.ndim != 1 or len(timestamps_seconds) != target_count:
            raise ValueError("MVSEC target timestamps and maps have different lengths")
        if not bool(np.isfinite(timestamps_seconds).all()) or np.any(
            timestamps_seconds[1:] < timestamps_seconds[:-1]
        ):
            raise ValueError("MVSEC target timestamps are invalid")
        source_timestamps_us = np.rint(timestamps_seconds * 1_000_000.0).astype(
            np.int64
        )
        per_source: list[MVSECTargetReference] = []
        for target_index, source_timestamp_us in enumerate(source_timestamps_us.tolist()):
            # Official dense flow i uses the previous pose to the current pose.
            # Index zero therefore has neither a positive interval nor a
            # meaningful motion target, even when event history happens to fit.
            if kind == "flow" and target_index == 0:
                continue
            label_timestamp_us = int(source_timestamp_us) - source.source_time_origin_us
            if f3_evflownet_split and not (
                EVFLOWNET_TEST_START_US
                <= label_timestamp_us
                < EVFLOWNET_TEST_STOP_US
            ):
                continue
            if alignment == "causal":
                event_window_end_us = label_timestamp_us
            else:
                event_window_end_us = _ceil_to_step(
                    label_timestamp_us + window_us // 2, 1_000
                )
            earliest_end = event_window_end_us - (history_steps - 1) * stride_us
            if (
                earliest_end - window_us < source.t_start_us
                or event_window_end_us > source.t_end_us
            ):
                continue
            final_window = store.slice(
                source.sequence_id, event_window_end_us, window_us
            )
            if final_window.event_count < minimum_events:
                continue
            flow_interval_us = 0
            if target_index > 0:
                flow_interval_us = int(
                    source_timestamps_us[target_index]
                    - source_timestamps_us[target_index - 1]
                )
            per_source.append(
                MVSECTargetReference(
                    source_index=source_index,
                    target_index=target_index,
                    label_timestamp_us=label_timestamp_us,
                    event_window_end_us=event_window_end_us,
                    flow_interval_us=flow_interval_us,
                )
            )
        all_references.extend(per_source)
    if not all_references:
        raise ValueError("no MVSEC targets have sufficient causal event history")
    return _subsample_evenly(all_references, maximum_samples)


def center_crop_parameters(image_size: tuple[int, int]) -> SpatialTransformParameters:
    height, width = image_size
    sensor_height, sensor_width = MVSEC_SENSOR_SIZE
    return SpatialTransformParameters(
        x0=(sensor_width - width) // 2,
        y0=(sensor_height - height) // 2,
        output_height=height,
        output_width=width,
        horizontal_flip=False,
    )


def _project_native_map(
    values: np.ndarray,
    transform: SpatialTransformParameters,
) -> np.ndarray:
    """Apply the same deterministic center crop/pad as the event window."""

    if tuple(values.shape[-2:]) != MVSEC_SENSOR_SIZE:
        raise ValueError("MVSEC target map must end in native shape [260,346]")
    source_y0 = max(0, transform.y0)
    source_x0 = max(0, transform.x0)
    source_y1 = min(
        MVSEC_SENSOR_SIZE[0], transform.y0 + transform.output_height
    )
    source_x1 = min(
        MVSEC_SENSOR_SIZE[1], transform.x0 + transform.output_width
    )
    destination_y0 = source_y0 - transform.y0
    destination_x0 = source_x0 - transform.x0
    destination_y1 = destination_y0 + max(0, source_y1 - source_y0)
    destination_x1 = destination_x0 + max(0, source_x1 - source_x0)
    output = np.zeros(
        (*values.shape[:-2], transform.output_height, transform.output_width),
        dtype=values.dtype,
    )
    if source_y1 > source_y0 and source_x1 > source_x0:
        output[
            ...,
            destination_y0:destination_y1,
            destination_x0:destination_x1,
        ] = values[..., source_y0:source_y1, source_x0:source_x1]
    return output


def representation_from_config(config: Any) -> Any:
    if config.representation.kind == "voxel_grid":
        return VoxelGrid(
            temporal_bins=config.representation.temporal_bins,
            normalization=config.representation.normalization,
        )
    return EventImage(normalization=config.representation.normalization)


class MVSECGeometryDataset(Dataset[Mapping[str, torch.Tensor]]):
    """Deterministic event histories paired with native-coordinate MVSEC GT."""

    def __init__(
        self,
        manifest: str | Path,
        sources: Sequence[MVSECGeometrySource],
        references: Sequence[MVSECTargetReference],
        *,
        kind: TargetKind,
        image_size: tuple[int, int],
        window_us: int,
        stride_us: int,
        history_steps: int,
        representation: Any,
        flow_mask: str = "f3",
        event_support_window_us: int | Literal["native_interval"] | None = None,
        min_depth: float = 0.1,
        max_depth: float = 80.0,
    ) -> None:
        if kind not in {"flow", "depth"}:
            raise ValueError("kind must be flow or depth")
        if flow_mask not in {"f3", "gt"}:
            raise ValueError("flow_mask must be f3 or gt")
        if min_depth <= 0 or max_depth <= min_depth:
            raise ValueError("depth bounds are invalid")
        self.store = H5EventStore(manifest)
        self.sources = tuple(sources)
        self.references = tuple(references)
        self.kind = kind
        self.crop = center_crop_parameters(image_size)
        self.window_us = int(window_us)
        self.stride_us = int(stride_us)
        self.history_steps = int(history_steps)
        self.representation = representation
        self.flow_mask = flow_mask
        if event_support_window_us is not None and (
            event_support_window_us != "native_interval"
            and (
                isinstance(event_support_window_us, bool)
                or not isinstance(event_support_window_us, int)
                or event_support_window_us <= 0
            )
        ):
            raise ValueError(
                "event_support_window_us must be positive, native_interval, or None"
            )
        self.event_support_window_us = event_support_window_us
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self._process_id: int | None = None
        self._gt_handles: OrderedDict[Path, Any] = OrderedDict()
        self._flow_npz_handles: OrderedDict[
            Path, MVSECOfficialFlowNPZ
        ] = OrderedDict()
        self._flow_npz_identities: dict[Path, FlowNPZIdentity] = {}
        if self.kind == "flow":
            flow_paths = {
                source.ground_truth_path
                for source in self.sources
                if source.target_format == "mvsec_gt_flow_npz_v1"
            }
            for path in flow_paths:
                # Main-process validation is cached after manifest inspection.
                # Capturing the identity lets spawned workers avoid rescanning
                # multi-gigabyte members without silently trusting a changed file.
                flow_npz = open_mvsec_official_flow_npz(path, verify_crc=True)
                try:
                    validated_identity = _flow_npz_identity(path)
                    if validated_identity not in _CRC_VERIFIED_FLOW_NPZ:
                        raise RuntimeError(
                            "MVSEC flow NPZ changed immediately after CRC validation"
                        )
                    self._flow_npz_identities[path] = validated_identity
                finally:
                    flow_npz.close()

    def __len__(self) -> int:
        return len(self.references)

    def _handle(self, path: Path) -> Any:
        import h5py

        process_id = os.getpid()
        if self._process_id != process_id:
            self.close()
            self._process_id = process_id
        if path in self._gt_handles:
            handle = self._gt_handles.pop(path)
            self._gt_handles[path] = handle
            return handle
        handle = h5py.File(path, "r")
        self._gt_handles[path] = handle
        while len(self._gt_handles) > 4:
            _, evicted = self._gt_handles.popitem(last=False)
            evicted.close()
        return handle

    def _flow_npz_handle(self, path: Path) -> MVSECOfficialFlowNPZ:
        process_id = os.getpid()
        if self._process_id != process_id:
            self.close()
            self._process_id = process_id
        expected_identity = self._flow_npz_identities.get(path)
        if expected_identity is None:
            raise RuntimeError("MVSEC flow NPZ was not validated before worker use")
        if _flow_npz_identity(path) != expected_identity:
            raise RuntimeError("MVSEC flow NPZ changed after dataset construction")
        if path in self._flow_npz_handles:
            flow_npz = self._flow_npz_handles.pop(path)
            self._flow_npz_handles[path] = flow_npz
            return flow_npz
        # Workers reopen the main-process-validated file without rescanning
        # multi-gigabyte flow members.  Check again after mapping to close the
        # replacement race between the identity check and the open calls.
        flow_npz = open_mvsec_official_flow_npz(path, verify_crc=False)
        if _flow_npz_identity(path) != expected_identity:
            flow_npz.close()
            raise RuntimeError("MVSEC flow NPZ changed while a worker opened it")
        self._flow_npz_handles[path] = flow_npz
        while len(self._flow_npz_handles) > 4:
            _, evicted = self._flow_npz_handles.popitem(last=False)
            evicted.close()
        return flow_npz

    def __getitem__(self, index: int) -> Mapping[str, torch.Tensor]:
        reference = self.references[index]
        source = self.sources[reference.source_index]
        windows = []
        final_window = None
        first_end = reference.event_window_end_us - (
            self.history_steps - 1
        ) * self.stride_us
        for step in range(self.history_steps):
            window = self.store.slice(
                source.sequence_id,
                first_end + step * self.stride_us,
                self.window_us,
            )
            cropped = SharedRandomSpatialTransform.apply(window, self.crop)
            windows.append(
                np.ascontiguousarray(self.representation(cropped), dtype=np.float32)
            )
            final_window = cropped
        assert final_window is not None

        support_window = final_window
        if self.kind == "flow" and self.event_support_window_us is not None:
            support_duration_us = (
                reference.flow_interval_us
                if self.event_support_window_us == "native_interval"
                else self.event_support_window_us
            )
            if support_duration_us <= 0:
                raise ValueError("flow event-support interval must be positive")
            support_native = self.store.slice(
                source.sequence_id,
                reference.label_timestamp_us,
                support_duration_us,
            )
            support_window = SharedRandomSpatialTransform.apply(
                support_native, self.crop
            )
        event_support = np.zeros(
            (self.crop.output_height, self.crop.output_width), dtype=np.bool_
        )
        event_support[support_window.y, support_window.x] = True

        if self.kind == "flow" and source.target_format == "mvsec_gt_flow_npz_v1":
            flow_npz = self._flow_npz_handle(source.ground_truth_path)
            target = np.stack(
                (
                    flow_npz.x_flow_dist[reference.target_index],
                    flow_npz.y_flow_dist[reference.target_index],
                ),
                axis=0,
            ).astype(np.float32, copy=False)
        else:
            target = np.asarray(
                self._handle(source.ground_truth_path)[source.target_dataset][
                    reference.target_index
                ],
                dtype=np.float32,
            )
        if self.kind == "flow":
            target = np.ascontiguousarray(_project_native_map(target, self.crop))
            finite = np.isfinite(target).all(axis=0)
            nonzero = np.linalg.norm(target, axis=0) > 0
            valid = finite & nonzero
            if self.flow_mask == "f3":
                sensor_rows = (
                    np.arange(self.crop.output_height)[:, None] + self.crop.y0
                )
                in_sensor = (sensor_rows >= 0) & (
                    sensor_rows < MVSEC_SENSOR_SIZE[0]
                )
                valid &= (
                    event_support
                    & in_sensor
                    & (sensor_rows < MVSEC_CAR_HOOD_START_ROW)
                )
        else:
            target = np.ascontiguousarray(_project_native_map(target, self.crop))
            valid = np.isfinite(target) & (target > self.min_depth) & (
                target < self.max_depth
            )
        target = np.where(valid, target, 0.0).astype(np.float32, copy=False)
        return {
            "x": torch.from_numpy(np.stack(windows)),
            "target": torch.from_numpy(target),
            "valid": torch.from_numpy(np.ascontiguousarray(valid)),
            "flow_interval_us": torch.tensor(
                reference.flow_interval_us, dtype=torch.int64
            ),
            "target_index": torch.tensor(reference.target_index, dtype=torch.int64),
            "label_timestamp_us": torch.tensor(
                reference.label_timestamp_us, dtype=torch.int64
            ),
        }

    def close(self) -> None:
        for handle in self._gt_handles.values():
            handle.close()
        self._gt_handles = OrderedDict()
        for flow_npz in self._flow_npz_handles.values():
            flow_npz.close()
        self._flow_npz_handles = OrderedDict()
        self.store.close()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_gt_handles"] = OrderedDict()
        state["_flow_npz_handles"] = OrderedDict()
        state["_process_id"] = None
        return state

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


@torch.no_grad()
def extract_frozen_mvsec_tokens(
    model: Any,
    x: torch.Tensor,
    *,
    duration_ms: float,
) -> torch.Tensor:
    """Extract the final causal recurrent state (or final feed-forward frame)."""

    if x.ndim != 5:
        raise ValueError("MVSEC histories must have shape [B,T,C,H,W]")
    duration = torch.full(
        (x.shape[0],), duration_ms, device=x.device, dtype=torch.float32
    )
    if isinstance(
        model.online_encoder, RecurrentVJEPA21EventVisionTransformer
    ):
        state = None
        tokens = None
        for step in range(x.shape[1]):
            tokens, state = model.encode_recurrent(
                x[:, step], duration, online_state=state, detach_state=True
            )
        assert tokens is not None
        return tokens
    return model.encode_only(x[:, -1], duration)


def dense_patch_prediction(
    prediction: torch.Tensor, image_size: tuple[int, int]
) -> torch.Tensor:
    if prediction.ndim != 4:
        raise ValueError("patch prediction must have shape [B,C,Hg,Wg]")
    return torch.nn.functional.interpolate(
        prediction,
        size=image_size,
        mode="bilinear",
        align_corners=False,
    )


def flow_metric_sums(
    prediction: np.ndarray, target: np.ndarray, valid: np.ndarray
) -> dict[str, float]:
    """Return additive MVSEC flow statistics, including F3-compatible metrics."""

    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    valid = np.asarray(valid, dtype=np.bool_)
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[0] != 2:
        raise ValueError("flow prediction and target must share shape [2,H,W]")
    if valid.shape != target.shape[1:]:
        raise ValueError("flow valid mask has the wrong shape")
    count = int(np.count_nonzero(valid))
    if count == 0:
        raise ValueError("flow metric mask is empty")
    pred = prediction[:, valid].T
    gt = target[:, valid].T
    if not bool(np.isfinite(pred).all() and np.isfinite(gt).all()):
        raise ValueError("valid flow vectors must be finite")
    endpoint = np.linalg.norm(pred - gt, axis=1)
    pred3 = np.concatenate((pred, np.ones((count, 1))), axis=1)
    gt3 = np.concatenate((gt, np.ones((count, 1))), axis=1)
    cosine = np.sum(pred3 * gt3, axis=1) / (
        np.linalg.norm(pred3, axis=1) * np.linalg.norm(gt3, axis=1)
    )
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return {
        "valid_pixels": float(count),
        "epe_sum": float(endpoint.sum()),
        "one_pe_count": float(np.count_nonzero(endpoint > 1.0)),
        "two_pe_count": float(np.count_nonzero(endpoint > 2.0)),
        "three_pe_count": float(np.count_nonzero(endpoint > 3.0)),
        "angular_error_sum": float(angle.sum()),
    }


def depth_metric_sums(
    prediction: np.ndarray, target: np.ndarray, valid: np.ndarray
) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    valid = np.asarray(valid, dtype=np.bool_)
    if prediction.shape != target.shape or prediction.shape != valid.shape:
        raise ValueError("depth prediction, target, and mask must share shape [H,W]")
    count = int(np.count_nonzero(valid))
    if count == 0:
        raise ValueError("depth metric mask is empty")
    pred = prediction[valid]
    gt = target[valid]
    if not bool(np.isfinite(pred).all() and np.isfinite(gt).all()):
        raise ValueError("valid depth values must be finite")
    if np.any(pred <= 0) or np.any(gt <= 0):
        raise ValueError("valid depth values must be positive")
    difference = pred - gt
    ratio = np.maximum(pred / gt, gt / pred)
    log_difference = np.log(pred) - np.log(gt)
    return {
        "valid_pixels": float(count),
        "abs_rel_sum": float(np.sum(np.abs(difference) / gt)),
        "sq_rel_sum": float(np.sum(np.square(difference) / gt)),
        "f3_sq_rel_sum": float(np.sum(np.square(difference) / np.square(gt))),
        "squared_error_sum": float(np.sum(np.square(difference))),
        "log_error_sum": float(np.sum(log_difference)),
        "squared_log_error_sum": float(np.sum(np.square(log_difference))),
        "log10_error_sum": float(np.sum(np.abs(np.log10(pred) - np.log10(gt)))),
        "delta1_count": float(np.count_nonzero(ratio < 1.25)),
        "delta2_count": float(np.count_nonzero(ratio < 1.25**2)),
        "delta3_count": float(np.count_nonzero(ratio < 1.25**3)),
    }


def finalize_flow_metrics(sums: Mapping[str, float]) -> dict[str, float]:
    count = float(sums["valid_pixels"])
    if count <= 0:
        raise ValueError("cannot finalize empty flow metrics")
    return {
        "AEPE": float(sums["epe_sum"]) / count,
        "1PE_percent": 100.0 * float(sums["one_pe_count"]) / count,
        "2PE_percent": 100.0 * float(sums["two_pe_count"]) / count,
        "3PE_percent": 100.0 * float(sums["three_pe_count"]) / count,
        "AAE_degrees": float(sums["angular_error_sum"]) / count,
    }


def finalize_depth_metrics(sums: Mapping[str, float]) -> dict[str, float]:
    count = float(sums["valid_pixels"])
    if count <= 0:
        raise ValueError("cannot finalize empty depth metrics")
    mean_log_error = float(sums.get("log_error_sum", 0.0)) / count
    mean_squared_log_error = float(sums["squared_log_error_sum"]) / count
    return {
        "AbsRel": float(sums["abs_rel_sum"]) / count,
        "SqRel": float(sums["sq_rel_sum"]) / count,
        "F3_SqRel": float(sums["f3_sq_rel_sum"]) / count,
        "RMSE": math.sqrt(float(sums["squared_error_sum"]) / count),
        "RMSE_log": math.sqrt(mean_squared_log_error),
        "SILog": math.sqrt(
            max(0.0, mean_squared_log_error - mean_log_error**2)
        ),
        "F3_SILog": math.sqrt(
            max(0.0, mean_squared_log_error - 0.5 * mean_log_error**2)
        ),
        "log10": float(sums["log10_error_sum"]) / count,
        "delta1": float(sums["delta1_count"]) / count,
        "delta2": float(sums["delta2_count"]) / count,
        "delta3": float(sums["delta3_count"]) / count,
    }


def accumulate_metric_sums(
    total: dict[str, float], current: Mapping[str, float]
) -> None:
    for name, value in current.items():
        total[name] = total.get(name, 0.0) + float(value)


__all__ = [
    "Alignment",
    "EVFLOWNET_TEST_START_US",
    "EVFLOWNET_TEST_STOP_US",
    "FlowNPZIdentity",
    "MVSECGeometryDataset",
    "MVSEC_F3_CONTEXT_WINDOW_US",
    "MVSECGeometrySource",
    "MVSECOfficialFlowNPZ",
    "MVSECTargetReference",
    "TargetKind",
    "accumulate_metric_sums",
    "build_mvsec_target_references",
    "center_crop_parameters",
    "dense_patch_prediction",
    "depth_metric_sums",
    "extract_frozen_mvsec_tokens",
    "finalize_depth_metrics",
    "finalize_flow_metrics",
    "flow_metric_sums",
    "open_mvsec_official_flow_npz",
    "read_mvsec_geometry_sources",
    "representation_from_config",
    "split_mvsec_temporal_dev_references",
    "temporal_reference_set_sha256",
]
