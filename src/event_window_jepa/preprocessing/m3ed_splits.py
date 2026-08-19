from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from event_window_jepa.preprocessing.common import (
    MANIFEST_ARTIFACT_PATH_FIELDS,
    write_manifest,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create M3ED experiment train/val/test manifests without rewriting "
            "preprocessed event or label artifacts"
        )
    )
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--m3ed-dataset-list", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--allow-unassigned",
        action="store_true",
        help="Ignore preprocessed recordings omitted from the selected protocol",
    )
    return parser.parse_args()


def _official_test_assignments(path: Path) -> dict[str, bool]:
    payload: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("M3ED dataset_list.yaml must contain a non-empty list")
    assignments: dict[str, bool] = {}
    for row in payload:
        if (
            not isinstance(row, dict)
            or str(row.get("filetype", "")).lower() != "data"
        ):
            continue
        name = str(row.get("file", ""))
        is_test = row.get("is_test_file")
        if not name or not isinstance(is_test, bool):
            raise ValueError("invalid M3ED data entry in dataset_list.yaml")
        if name in assignments and assignments[name] != is_test:
            raise ValueError(f"conflicting official split assignment for {name}")
        assignments[name] = is_test
    if not assignments:
        raise ValueError("M3ED dataset_list.yaml contains no data entries")
    return assignments


def _protocol(path: Path) -> tuple[str, dict[str, set[str]]]:
    payload: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("M3ED protocol must be a YAML mapping")
    protocol_name = str(payload.get("name", path.stem))
    raw_splits = payload.get("splits")
    if not isinstance(raw_splits, dict):
        raise ValueError("M3ED protocol requires a splits mapping")
    splits: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        values = raw_splits.get(split)
        if not isinstance(values, list) or not values:
            raise ValueError(f"M3ED protocol split {split} must be a non-empty list")
        names = {str(value) for value in values}
        if len(names) != len(values) or not all(names):
            raise ValueError(f"M3ED protocol split {split} has invalid duplicates/names")
        splits[split] = names
    intersections = {
        f"{left}/{right}": sorted(splits[left] & splits[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
        if splits[left] & splits[right]
    }
    if intersections:
        raise ValueError(f"M3ED protocol recordings cross splits: {intersections}")
    return protocol_name, splits


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row: dict[str, Any] = json.loads(line)
            if row.get("dataset") != "m3ed" or "sequence_id" not in row:
                raise ValueError(f"{path}:{line_number} is not an M3ED manifest row")
            if "source_sequence_name" not in row:
                sequence_id = str(row["sequence_id"])
                prefix = "m3ed__"
                suffix = f"__{row.get('camera', 'left')}"
                if not sequence_id.startswith(prefix) or not sequence_id.endswith(
                    suffix
                ):
                    raise ValueError(
                        f"{path}:{line_number} has no source_sequence_name and cannot be parsed"
                    )
                row["source_sequence_name"] = sequence_id[
                    len(prefix) : -len(suffix)
                ]
            row["storage_split"] = str(
                row.get("storage_split", row.get("split", "train"))
            )
            for field in MANIFEST_ARTIFACT_PATH_FIELDS:
                if row.get(field) is None:
                    continue
                artifact = Path(str(row[field]))
                if not artifact.is_absolute():
                    artifact = (path.parent / artifact).resolve()
                if not artifact.is_file():
                    raise FileNotFoundError(
                        f"{path}:{line_number} references missing {field}: {artifact}"
                    )
                row[field] = str(artifact)
            records.append(row)
    if not records:
        raise ValueError(f"M3ED input manifest is empty: {path}")
    return records


def create_m3ed_split_manifests(
    input_manifest: str | Path,
    protocol_path: str | Path,
    dataset_list: str | Path,
    output_dir: str | Path,
    *,
    allow_unassigned: bool = False,
) -> dict[str, int]:
    manifest = Path(input_manifest).expanduser().resolve()
    protocol_file = Path(protocol_path).expanduser().resolve()
    dataset_list_file = Path(dataset_list).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    protocol_name, splits = _protocol(protocol_file)
    official = _official_test_assignments(dataset_list_file)
    requested_names = set().union(*splits.values())
    unknown = requested_names - set(official)
    if unknown:
        raise ValueError(
            "protocol recordings are absent from official M3ED list: "
            f"{sorted(unknown)}"
        )
    official_test_leakage = (splits["train"] | splits["val"]) & {
        name for name, is_test in official.items() if is_test
    }
    if official_test_leakage:
        raise ValueError(
            "official M3ED test recordings cannot enter experiment train/val: "
            f"{sorted(official_test_leakage)}"
        )

    records = _read_manifest(manifest)
    available = {str(row["source_sequence_name"]) for row in records}
    missing = requested_names - available
    if missing:
        raise ValueError(
            "protocol recordings are absent from input manifest: "
            f"{sorted(missing)}"
        )
    unassigned = available - requested_names
    if unassigned and not allow_unassigned:
        raise ValueError(
            "input manifest contains recordings absent from protocol; use "
            f"--allow-unassigned to ignore them: {sorted(unassigned)}"
        )

    counts: dict[str, int] = {}
    destination.mkdir(parents=True, exist_ok=True)
    for split, names in splits.items():
        output_records: list[dict[str, Any]] = []
        for original in records:
            if str(original["source_sequence_name"]) not in names:
                continue
            row = dict(original)
            row["split"] = split
            row["split_protocol"] = protocol_name
            output_records.append(row)
        write_manifest(output_records, destination / f"{split}.jsonl")
        counts[split] = len(output_records)
    return counts


def main() -> None:
    args = _parse_args()
    counts = create_m3ed_split_manifests(
        args.input_manifest,
        args.protocol,
        args.m3ed_dataset_list,
        args.output_dir,
        allow_unassigned=args.allow_unassigned,
    )
    print(json.dumps({"status": "complete", "counts": counts}, sort_keys=True))


if __name__ == "__main__":
    main()
