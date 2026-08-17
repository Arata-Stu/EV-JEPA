from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from event_window_jepa.data.anchor_sampler import UniformTimeAnchorSampler
from event_window_jepa.data.types import SequenceInfo
from event_window_jepa.evaluation.robustness_metrics import (
    mean_and_sample_std,
    summarize_curve,
)


@dataclass(frozen=True, order=True)
class EvaluationAnchor:
    sequence_id: str
    t_end_us: int

    def __post_init__(self) -> None:
        if not self.sequence_id or self.t_end_us < 0:
            raise ValueError("evaluation anchors require an id and non-negative timestamp")


def evaluation_anchor_set_id(anchors: Sequence[EvaluationAnchor]) -> str:
    if not anchors:
        raise ValueError("evaluation anchors cannot be empty")
    serialized = json.dumps(
        [(anchor.sequence_id, anchor.t_end_us) for anchor in sorted(anchors)],
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def make_evaluation_anchors(
    sequences: Sequence[SequenceInfo],
    maximum_window_ms: float,
    number_of_anchors: int,
    seed: int,
) -> tuple[EvaluationAnchor, ...]:
    """Create one time-uniform anchor population shared by every window."""

    if number_of_anchors <= 0:
        raise ValueError("number_of_anchors must be positive")
    maximum_attempts = number_of_anchors * 100
    sampler = UniformTimeAnchorSampler(
        sequences,
        maximum_window_ms=maximum_window_ms,
        samples_per_epoch=maximum_attempts,
        seed=seed,
    )
    anchors: list[EvaluationAnchor] = []
    seen: set[EvaluationAnchor] = set()
    for index in range(maximum_attempts):
        sampled = sampler.sample(index, epoch=0)
        anchor = EvaluationAnchor(sampled.sequence_id, sampled.t_end_us)
        if anchor in seen:
            continue
        seen.add(anchor)
        anchors.append(anchor)
        if len(anchors) == number_of_anchors:
            return tuple(anchors)
    raise ValueError("sequence time ranges contain too few unique evaluation anchors")


def run_window_sweep(
    evaluate: Callable[[float, Sequence[EvaluationAnchor]], float],
    windows_ms: Iterable[float],
    anchors: Sequence[EvaluationAnchor],
) -> tuple[dict[float, float], str]:
    """Evaluate every duration on one immutable, hashed anchor population."""

    windows = tuple(float(window) for window in windows_ms)
    anchors = tuple(anchors)
    if (
        not windows
        or any(not math.isfinite(window) or window <= 0 for window in windows)
        or len(set(windows)) != len(windows)
    ):
        raise ValueError("windows must be unique finite positive durations")
    if len(set(anchors)) != len(anchors):
        raise ValueError("evaluation anchors must be unique")
    sample_set_id = evaluation_anchor_set_id(anchors)
    curve = {window: float(evaluate(window, anchors)) for window in windows}
    return curve, sample_set_id


def _aggregate_seed_records(
    records: Sequence[dict[str, Any]], minimum_seeds: int
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["method"], record["metric"])].append(record)
    seeds_by_metric: dict[str, set[int]] = {}
    for (_, metric), method_records in grouped.items():
        seeds = {int(record["seed"]) for record in method_records}
        if metric in seeds_by_metric and seeds_by_metric[metric] != seeds:
            raise ValueError(f"metric {metric!r} uses different seed sets across methods")
        seeds_by_metric[metric] = seeds
    aggregate: list[dict[str, Any]] = []
    for (method, metric), method_records in sorted(grouped.items()):
        seeds = {record["seed"] for record in method_records}
        if len(seeds) < minimum_seeds:
            raise ValueError(
                f"{method}/{metric} has {len(seeds)} seeds; at least {minimum_seeds} required"
            )
        scopes = method_records[0]["summaries"].keys()
        summarized_scopes: dict[str, Any] = {}
        for scope in scopes:
            fields = method_records[0]["summaries"][scope].keys()
            summarized_scopes[scope] = {}
            for field in fields:
                values = [float(record["summaries"][scope][field]) for record in method_records]
                mean, sample_std = mean_and_sample_std(values)
                summarized_scopes[scope][field] = {"mean": mean, "std": sample_std}
        aggregate.append(
            {
                "method": method,
                "metric": metric,
                "seeds": sorted(seeds),
                "summaries": summarized_scopes,
            }
        )
    return aggregate


def summarize_jsonl(
    path: str | Path,
    reference_window_ms: float,
    minimum_seeds: int = 3,
) -> dict[str, Any]:
    if minimum_seeds <= 0:
        raise ValueError("minimum_seeds must be positive")
    required = {
        "method",
        "metric",
        "seed",
        "window_ms",
        "value",
        "higher_is_better",
        "sample_set_id",
        "window_group",
    }
    grouped: dict[tuple[str, str, int], dict[float, float]] = defaultdict(dict)
    group_labels: dict[tuple[str, str, int], dict[float, str]] = defaultdict(dict)
    metadata: dict[tuple[str, str, int], tuple[bool, str]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not required.issubset(row):
                missing = sorted(required - set(row))
                raise ValueError(f"line {line_number} lacks fields: {missing}")
            key = (str(row["method"]), str(row["metric"]), int(row["seed"]))
            window = float(row["window_ms"])
            if window in grouped[key]:
                raise ValueError(f"duplicate window on line {line_number}")
            direction = row["higher_is_better"]
            if not isinstance(direction, bool):
                raise TypeError(f"higher_is_better must be boolean on line {line_number}")
            sample_set_id = str(row["sample_set_id"])
            if not sample_set_id:
                raise ValueError(f"sample_set_id cannot be empty on line {line_number}")
            current_metadata = (direction, sample_set_id)
            if key in metadata and metadata[key] != current_metadata:
                raise ValueError(f"inconsistent direction/sample population on line {line_number}")
            metadata[key] = current_metadata
            group = str(row["window_group"])
            if group not in {"seen", "unseen_interpolation", "unseen_extrapolation"}:
                raise ValueError(f"invalid window_group on line {line_number}")
            grouped[key][window] = float(row["value"])
            group_labels[key][window] = group

    if not grouped:
        raise ValueError("metric file contains no rows")

    protocol_by_metric: dict[str, tuple[Any, ...]] = {}
    for key, curve in grouped.items():
        _, metric, _ = key
        direction, sample_set_id = metadata[key]
        protocol = (
            tuple(sorted(curve)),
            tuple(sorted(group_labels[key].items())),
            direction,
            sample_set_id,
        )
        if metric in protocol_by_metric and protocol_by_metric[metric] != protocol:
            raise ValueError(f"metric {metric!r} uses mismatched windows or sample populations")
        protocol_by_metric[metric] = protocol

    per_seed: list[dict[str, Any]] = []
    for (method, metric, seed), curve in sorted(grouped.items()):
        direction, sample_set_id = metadata[(method, metric, seed)]
        full = summarize_curve(curve, reference_window_ms, direction)
        reference_value = full.reference
        summaries: dict[str, Any] = {"all": full.to_dict()}
        labels = group_labels[(method, metric, seed)]
        scope_windows = {
            "seen": [window for window, group in labels.items() if group == "seen"],
            "unseen_interpolation": [
                window for window, group in labels.items() if group == "unseen_interpolation"
            ],
            "unseen_extrapolation": [
                window for window, group in labels.items() if group == "unseen_extrapolation"
            ],
            "unseen_all": [window for window, group in labels.items() if group != "seen"],
        }
        for scope, windows in scope_windows.items():
            if not windows:
                continue
            subset = {window: curve[window] for window in windows}
            summaries[scope] = summarize_curve(
                subset,
                reference_window_ms,
                direction,
                reference_value=reference_value,
            ).to_dict()
        per_seed.append(
            {
                "method": method,
                "metric": metric,
                "seed": seed,
                "higher_is_better": direction,
                "sample_set_id": sample_set_id,
                "summaries": summaries,
            }
        )
    return {
        "per_seed": per_seed,
        "across_seeds": _aggregate_seed_records(per_seed, minimum_seeds),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize matched window-sweep JSONL metrics")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--reference-window-ms", type=float, default=40.0)
    parser.add_argument("--minimum-seeds", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    records = summarize_jsonl(
        args.input,
        reference_window_ms=args.reference_window_ms,
        minimum_seeds=args.minimum_seeds,
    )
    payload = json.dumps(records, indent=2, sort_keys=True)
    if args.output is None:
        print(payload)
    else:
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
