from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from event_window_jepa.preprocessing.common import (
    PreprocessOptions,
    preprocess_sequence,
    write_manifest,
)
from event_window_jepa.preprocessing.sources import (
    discover_sequence_paths,
    make_event_source,
    sequence_identifier,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert raw DSEC, M3ED, Gen1, or Gen4/1Mpx event streams into "
            "arbitrary-window Zstd HDF5 files"
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("dsec", "m3ed", "prophesee_1mpx", "gen1", "gen4"),
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--split", required=True, choices=("train", "val", "test"), help="Logical split"
    )
    parser.add_argument("--camera", default="left", choices=("left", "right"))
    parser.add_argument(
        "--sequence-list",
        type=Path,
        help="Optional text file of sequence directory/stem names to include",
    )
    parser.add_argument(
        "--m3ed-dataset-list",
        type=Path,
        help="Official M3ED dataset_list.yaml used to enforce is_test_file",
    )
    parser.add_argument(
        "--spatial-downsample",
        type=int,
        help=(
            "Integer coordinate downsample. Defaults to 2 for M3ED/1Mpx and 1 "
            "for DSEC/Gen1. This does not discard events or create time windows."
        ),
    )
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--read-chunk-events", type=int, default=1_000_000)
    parser.add_argument("--hdf5-chunk-events", type=int, default=262_144)
    parser.add_argument("--zstd-level", type=int, default=5)
    parser.add_argument("--index-step-us", type=int, default=1_000)
    parser.add_argument(
        "--timestamp-dtype", choices=("auto", "uint32", "uint64"), default="auto"
    )
    parser.add_argument("--limit", type=int, help="Convert only the first N sequences")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Inspect inputs and print the conversion plan without writing HDF5",
    )
    parser.add_argument(
        "--bbox-output-root",
        type=Path,
        help=(
            "Directory for portable copies of matching *_bbox.npy files. "
            "Defaults to OUTPUT_ROOT/labels for Prophesee DAT or RVT HDF5 inputs."
        ),
    )
    parser.add_argument(
        "--allow-missing-bboxes",
        action="store_true",
        help=(
            "Allow Gen1/Gen4/1Mpx train or val event files without a matching sibling "
            "*_bbox.npy (intended only for self-supervised pretraining)"
        ),
    )
    parser.add_argument(
        "--merge-manifest",
        action="store_true",
        help=(
            "Under a manifest lock, retain valid existing rows and add/refresh "
            "the selected sequences"
        ),
    )
    replacement = parser.add_mutually_exclusive_group()
    replacement.add_argument("--overwrite", action="store_true")
    replacement.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--no-resume-partial",
        action="store_true",
        help="Restart instead of resuming a compatible per-sequence .partial file",
    )
    return parser.parse_args()


def _selected_names(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    names = {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not names:
        raise ValueError("sequence-list does not contain any sequence names")
    return names


def _source_sequence_name(dataset: str, path: Path) -> str:
    if dataset == "dsec" and path.parent.name in {"left", "right"}:
        return path.parents[2].name
    if dataset == "m3ed":
        return path.stem.removesuffix("_data")
    if dataset == "gen4" and path.name.endswith("_td.h5"):
        return path.name.removesuffix("_td.h5")
    if dataset == "gen1" and path.name.endswith("_td.dat.h5"):
        return path.name.removesuffix("_td.dat.h5")
    return path.stem


def _resolved_spatial_downsample(dataset: str, requested: int | None) -> int:
    if requested is not None:
        value = requested
    elif dataset in {"m3ed", "prophesee_1mpx", "gen4"}:
        value = 2
    else:
        value = 1
    if value <= 0:
        raise ValueError("--spatial-downsample must be positive")
    return value


def _matching_bbox(path: Path) -> Path | None:
    name = path.name
    if name.endswith("_td.dat.h5"):
        bbox_stem = name.removesuffix("_td.dat.h5") + "_bbox"
    elif name.endswith("_td.h5"):
        bbox_stem = name.removesuffix("_td.h5") + "_bbox"
    elif path.suffix.lower() == ".dat":
        stem = path.stem
        bbox_stem = f"{stem[:-3]}_bbox" if stem.endswith("_td") else f"{stem}_bbox"
    else:
        return None
    candidate = path.with_name(f"{bbox_stem}.npy")
    return candidate if candidate.is_file() else None


def _copy_bbox_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_stat = source.stat()
    if destination.is_file():
        destination_stat = destination.stat()
        if (
            destination_stat.st_size == source_stat.st_size
            and destination_stat.st_mtime_ns == source_stat.st_mtime_ns
        ):
            return
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.partial"
    )
    try:
        with source.open("rb") as source_handle, temporary.open("xb") as target_handle:
            shutil.copyfileobj(source_handle, target_handle, length=16 * 1024 * 1024)
            target_handle.flush()
            os.fsync(target_handle.fileno())
        shutil.copystat(source, temporary)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _validate_bbox(
    path: Path, *, source_width: int, source_height: int
) -> dict[str, bool | int | str]:
    """Validate a Prophesee bbox NPY without loading the complete array."""

    boxes = np.load(path, mmap_mode="r", allow_pickle=False)
    if not isinstance(boxes, np.ndarray) or boxes.ndim != 1:
        raise ValueError(f"bbox file must contain a one-dimensional array: {path}")
    names = set(boxes.dtype.names or ())
    timestamp_name = "t" if "t" in names else "ts" if "ts" in names else None
    missing = {"x", "y", "w", "h"} - names
    if timestamp_name is None or missing:
        raise ValueError(
            f"bbox file {path} requires t/ts,x,y,w,h fields; missing "
            f"{sorted(missing | ({'t/ts'} if timestamp_name is None else set()))}"
        )
    if not np.issubdtype(boxes.dtype.fields[timestamp_name][0], np.integer):
        raise TypeError(f"bbox timestamp field must be integer microseconds in {path}")
    required_names = (timestamp_name, "x", "y", "w", "h")
    for name in ("x", "y", "w", "h"):
        if not np.issubdtype(boxes.dtype.fields[name][0], np.number):
            raise TypeError(f"bbox field {name} must be numeric in {path}")

    previous_timestamp: float | int | None = None
    out_of_fov_count = 0
    for start in range(0, len(boxes), 1_000_000):
        stop = min(start + 1_000_000, len(boxes))
        values = {
            name: np.asarray(boxes[name][start:stop]) for name in required_names
        }
        if any(not bool(np.isfinite(value).all()) for value in values.values()):
            raise ValueError(f"bbox file contains non-finite values: {path}")
        timestamps = values[timestamp_name]
        if len(timestamps):
            if np.any(timestamps < 0) or np.any(timestamps[1:] < timestamps[:-1]):
                raise ValueError(f"bbox timestamps must be sorted and non-negative: {path}")
            if previous_timestamp is not None and timestamps[0] < previous_timestamp:
                raise ValueError(f"bbox timestamps decrease across chunks: {path}")
            previous_timestamp = timestamps[-1].item()
        # Cast coordinate arithmetic so narrow unsigned fields cannot overflow
        # while checking x+w/y+h at the sensor boundary.
        x = values["x"].astype(np.float64, copy=False)
        y = values["y"].astype(np.float64, copy=False)
        width = values["w"].astype(np.float64, copy=False)
        height = values["h"].astype(np.float64, copy=False)
        if (
            np.any(width <= 0)
            or np.any(height <= 0)
        ):
            raise ValueError(f"bbox width and height must be positive: {path}")
        out_of_fov = (
            (x < 0)
            | (y < 0)
            | (x + width > source_width)
            | (y + height > source_height)
        )
        out_of_fov_count += int(np.count_nonzero(out_of_fov))
    return {
        "bbox_count": len(boxes),
        "bbox_out_of_fov_count": out_of_fov_count,
        "bbox_requires_fov_clip": out_of_fov_count > 0,
        "bbox_timestamp_field": timestamp_name,
    }


def _dat_path_split(path: Path) -> str | None:
    aliases = {"train": "train", "val": "val", "validation": "val", "test": "test"}
    for parent in path.parents:
        if parent.name.lower() in aliases:
            return aliases[parent.name.lower()]
    return None


def _m3ed_official_names(path: Path, logical_split: str) -> set[str]:
    payload: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("M3ED dataset_list.yaml must contain a non-empty list")
    assignments: dict[str, bool] = {}
    for row in payload:
        if not isinstance(row, dict) or "file" not in row or "filetype" not in row:
            raise ValueError("invalid entry in M3ED dataset_list.yaml")
        if str(row["filetype"]).lower() != "data":
            continue
        if "is_test_file" not in row:
            raise ValueError("M3ED data entries must define is_test_file")
        name = str(row["file"])
        is_test = row["is_test_file"]
        if not isinstance(is_test, bool):
            raise TypeError("M3ED is_test_file values must be boolean")
        if name in assignments and assignments[name] != is_test:
            raise ValueError(f"conflicting M3ED split assignment for {name}")
        assignments[name] = is_test
    expected_test = logical_split == "test"
    return {name for name, is_test in assignments.items() if is_test == expected_test}


def main() -> None:
    args = _parse_args()
    spatial_downsample = _resolved_spatial_downsample(
        args.dataset, args.spatial_downsample
    )
    if (args.width is None) != (args.height is None):
        raise ValueError("--width and --height must be provided together")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.dataset == "dsec" and spatial_downsample != 1:
        raise ValueError(
            "DSEC must remain at native resolution; apply rectification or label-aware "
            "resizing in the downstream adapter"
        )
    if args.bbox_output_root is not None and args.dataset not in {
        "prophesee_1mpx",
        "gen1",
        "gen4",
    }:
        raise ValueError("--bbox-output-root is only valid for Prophesee/RVT datasets")
    if args.allow_missing_bboxes and args.dataset not in {
        "prophesee_1mpx",
        "gen1",
        "gen4",
    }:
        raise ValueError(
            "--allow-missing-bboxes is only valid for Prophesee/RVT datasets"
        )

    input_path = args.input.expanduser().resolve()
    paths = discover_sequence_paths(args.dataset, input_path, args.camera)
    if args.dataset in {"prophesee_1mpx", "gen1", "gen4"}:
        declared_splits = {path: _dat_path_split(path) for path in paths}
        if input_path.is_dir():
            unassigned = [path for path, split in declared_splits.items() if split is None]
            if unassigned:
                raise ValueError(
                    "directory-wide Prophesee/RVT conversion requires train/val/test "
                    "directories; "
                    f"could not determine the split of {unassigned[:3]}"
                )
            paths = [
                path for path, declared_split in declared_splits.items()
                if declared_split == args.split
            ]
        else:
            declared_split = declared_splits[paths[0]]
            if declared_split is not None and declared_split != args.split:
                raise ValueError(
                    f"source path belongs to {declared_split}, not requested {args.split}"
                )
    if args.dataset == "dsec" and input_path.is_dir() and args.sequence_list is None:
        raise ValueError(
            "directory-wide DSEC conversion requires --sequence-list so logical "
            "validation sequences cannot leak into train"
        )
    if args.dataset != "m3ed" and args.m3ed_dataset_list is not None:
        raise ValueError("--m3ed-dataset-list can only be used with --dataset m3ed")
    if args.dataset == "m3ed" and args.m3ed_dataset_list is None:
        raise ValueError(
            "M3ED conversion requires --m3ed-dataset-list so the official "
            "train/test boundary is enforced"
        )

    requested_names = _selected_names(args.sequence_list)
    if args.dataset == "m3ed" and args.split in {"train", "val"}:
        if requested_names is None:
            raise ValueError(
                "M3ED train/val conversion requires an explicit --sequence-list; "
                "use disjoint recording-level lists because M3ED has no official val split"
            )
    official_names: set[str] | None = None
    if args.m3ed_dataset_list is not None:
        official_names = _m3ed_official_names(
            args.m3ed_dataset_list.expanduser().resolve(), args.split
        )
        if requested_names is not None:
            outside_official_split = requested_names - official_names
            if outside_official_split:
                raise ValueError(
                    "M3ED sequence-list crosses the official test boundary: "
                    f"{sorted(outside_official_split)}"
                )

    selected = requested_names if requested_names is not None else official_names
    if selected is not None:
        paths = [
            path
            for path in paths
            if _source_sequence_name(args.dataset, path) in selected
        ]
        found = {_source_sequence_name(args.dataset, path) for path in paths}
        missing = (requested_names or set()) - found
        if missing:
            raise ValueError(f"sequence-list entries were not found: {sorted(missing)}")
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise ValueError("no input sequences remain after filtering")

    expected_sequence_ids = [
        sequence_identifier(
            args.dataset,
            _source_sequence_name(args.dataset, path),
            args.camera,
        )
        for path in paths
    ]
    duplicates = sorted(
        sequence_id
        for sequence_id, count in Counter(expected_sequence_ids).items()
        if count > 1
    )
    if duplicates:
        raise ValueError(
            "input paths collapse to duplicate output sequence ids: "
            f"{duplicates}"
        )
    bbox_sources = (
        {path: _matching_bbox(path) for path in paths}
        if args.dataset in {"prophesee_1mpx", "gen1", "gen4"}
        else {}
    )
    if (
        args.dataset in {"prophesee_1mpx", "gen1", "gen4"}
        and args.split in {"train", "val"}
        and not args.allow_missing_bboxes
    ):
        missing_bboxes = [path for path, bbox in bbox_sources.items() if bbox is None]
        if missing_bboxes:
            raise FileNotFoundError(
                "selected train/val event files have no matching sibling *_bbox.npy: "
                f"{missing_bboxes[:3]}; use --allow-missing-bboxes only for "
                "label-free self-supervised data"
            )
    bbox_metadata: dict[Path, dict[str, bool | int | str]] = {}
    if bbox_sources:
        expected_width = args.width or (
            1280 if args.dataset in {"prophesee_1mpx", "gen4"} else 304
        )
        expected_height = args.height or (
            720 if args.dataset in {"prophesee_1mpx", "gen4"} else 240
        )
        for path, bbox in bbox_sources.items():
            if bbox is not None:
                bbox_metadata[path] = _validate_bbox(
                    bbox,
                    source_width=expected_width,
                    source_height=expected_height,
                )

    output_root = args.output_root.expanduser().resolve()
    bbox_output_root = (
        args.bbox_output_root.expanduser().resolve()
        if args.bbox_output_root is not None
        else output_root / "labels"
    )
    manifest = (
        args.manifest.expanduser().resolve()
        if args.manifest is not None
        else output_root / f"{args.dataset}_{args.split}.jsonl"
    )
    options = PreprocessOptions(
        spatial_downsample=spatial_downsample,
        read_chunk_events=args.read_chunk_events,
        hdf5_chunk_events=args.hdf5_chunk_events,
        zstd_level=args.zstd_level,
        index_step_us=args.index_step_us,
        timestamp_dtype=args.timestamp_dtype,
        overwrite=args.overwrite,
        skip_existing=args.skip_existing,
        resume_partial=not args.no_resume_partial,
    )

    records: list[dict[str, object]] = []
    planned_source_bytes = 0
    planned_events = 0
    for number, path in enumerate(paths, start=1):
        source = make_event_source(
            args.dataset,
            path,
            camera=args.camera,
            width=args.width,
            height=args.height,
        )
        output = output_root / f"{source.metadata.sequence_id}.h5"
        output_width = math.ceil(source.metadata.width / spatial_downsample)
        output_height = math.ceil(source.metadata.height / spatial_downsample)
        bbox_source = bbox_sources.get(path)
        bbox_output = (
            bbox_output_root / f"{source.metadata.sequence_id}__bbox.npy"
            if bbox_source is not None
            else None
        )
        if args.plan_only:
            planned_source_bytes += path.stat().st_size
            planned_events += source.metadata.event_count
            print(
                json.dumps(
                    {
                        "status": "planned",
                        "sequence": source.metadata.sequence_id,
                        "number": number,
                        "total": len(paths),
                        "source": str(path),
                        "source_bytes": path.stat().st_size,
                        "source_events": source.metadata.event_count,
                        "source_resolution": [
                            source.metadata.width,
                            source.metadata.height,
                        ],
                        "stored_resolution": [output_width, output_height],
                        "spatial_downsample": spatial_downsample,
                        "output": str(output),
                        "bbox_source": (
                            None if bbox_source is None else str(bbox_source)
                        ),
                        "bbox_output": (
                            None if bbox_output is None else str(bbox_output)
                        ),
                        "bbox_count": (
                            None
                            if bbox_source is None
                            else bbox_metadata[path]["bbox_count"]
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            source.close()
            continue
        print(
            json.dumps(
                {
                    "status": "converting",
                    "sequence": source.metadata.sequence_id,
                    "number": number,
                    "total": len(paths),
                    "source_events": source.metadata.event_count,
                    "source_resolution": [source.metadata.width, source.metadata.height],
                    "stored_resolution": [output_width, output_height],
                    "spatial_downsample": spatial_downsample,
                    "output": str(output),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        try:
            record = preprocess_sequence(
                source,
                output,
                split=args.split,
                options=options,
            )
        finally:
            source.close()
        if bbox_source is not None and bbox_output is not None:
            _copy_bbox_atomic(bbox_source, bbox_output)
            record.update(
                {
                    "bbox_path": str(bbox_output.resolve()),
                    "bbox_spatial_downsample": 1,
                    "bbox_width": source.metadata.width,
                    "bbox_height": source.metadata.height,
                    "bbox_coordinate_frame": source.metadata.coordinate_frame,
                    "bbox_timestamp_reference": str(
                        source.metadata.attributes.get(
                            "timestamp_reference", "unknown"
                        )
                    ),
                    "bbox_timestamps_relative": False,
                    **bbox_metadata[path],
                }
            )
        records.append(record)
        print(
            json.dumps(
                {
                    "status": "complete",
                    "sequence": record["sequence_id"],
                    "event_count": record["event_count"],
                    "output_bytes": record["output_file_size"],
                    "source_file_bytes": record["source_file_size"],
                    "bbox": record.get("bbox_path"),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if args.plan_only:
        print(
            json.dumps(
                {
                    "status": "plan_complete",
                    "sequences": len(paths),
                    "source_events": planned_events,
                    "source_bytes": planned_source_bytes,
                    "spatial_downsample": spatial_downsample,
                    "note": "event count is preserved; output size is data dependent",
                },
                sort_keys=True,
            )
        )
        return

    write_manifest(records, manifest, merge_existing=args.merge_manifest)
    print(
        json.dumps(
            {"status": "manifest_complete", "manifest": str(manifest), "sequences": len(records)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
