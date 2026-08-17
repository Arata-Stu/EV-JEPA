from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from event_window_jepa.preprocessing.common import write_manifest


def merge_manifests(inputs: list[str | Path], output: str | Path) -> None:
    records: list[dict[str, Any]] = []
    for input_path in inputs:
        manifest = Path(input_path).expanduser().resolve()
        with manifest.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
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
    write_manifest(records, output)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge dataset manifests while preserving resolved artifact paths"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("inputs", nargs="+", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    merge_manifests(args.inputs, args.output)


if __name__ == "__main__":
    main()
