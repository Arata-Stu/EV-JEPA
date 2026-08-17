from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class RobustnessSummary:
    average: float
    worst: float
    reference: float
    maximum_degradation: float
    relative_degradation: float
    area_over_log_window: float
    number_of_windows: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _log_trapezoid_average(points: Sequence[tuple[float, float]]) -> float:
    if len(points) == 1:
        return points[0][1]
    ordered = sorted(points)
    logs = [math.log(window) for window, _ in ordered]
    span = logs[-1] - logs[0]
    if span <= 0:
        raise ValueError("window durations must be distinct")
    area = 0.0
    for index in range(len(ordered) - 1):
        width = logs[index + 1] - logs[index]
        area += width * (ordered[index][1] + ordered[index + 1][1]) / 2.0
    return area / span


def summarize_curve(
    metrics_by_window: Mapping[float, float],
    reference_window_ms: float = 40.0,
    higher_is_better: bool = True,
    reference_value: float | None = None,
) -> RobustnessSummary:
    """Summarize a window-performance curve for either metric direction."""

    if not metrics_by_window:
        raise ValueError("metrics_by_window cannot be empty")
    points = [(float(window), float(metric)) for window, metric in metrics_by_window.items()]
    if any(not math.isfinite(window) or window <= 0 for window, _ in points):
        raise ValueError("window durations must be finite and positive")
    if not math.isfinite(reference_window_ms) or reference_window_ms <= 0:
        raise ValueError("reference_window_ms must be finite and positive")
    if any(not math.isfinite(metric) for _, metric in points):
        raise ValueError("metrics must be finite")
    if reference_value is None:
        reference_matches = [
            metric for window, metric in points if math.isclose(window, reference_window_ms)
        ]
        if len(reference_matches) != 1:
            raise ValueError("reference window must occur exactly once")
        reference = reference_matches[0]
    else:
        reference = float(reference_value)
        if not math.isfinite(reference):
            raise ValueError("reference_value must be finite")
    values = [metric for _, metric in points]
    worst = min(values) if higher_is_better else max(values)
    raw_degradation = reference - worst if higher_is_better else worst - reference
    maximum_degradation = max(0.0, raw_degradation)
    denominator = max(abs(reference), 1e-12)
    return RobustnessSummary(
        average=statistics.fmean(values),
        worst=worst,
        reference=reference,
        maximum_degradation=maximum_degradation,
        relative_degradation=maximum_degradation / denominator,
        area_over_log_window=_log_trapezoid_average(points),
        number_of_windows=len(points),
    )


def subset_curve(
    metrics_by_window: Mapping[float, float], windows_ms: Sequence[float]
) -> dict[float, float]:
    requested = {float(window) for window in windows_ms}
    result = {
        float(window): float(metric)
        for window, metric in metrics_by_window.items()
        if float(window) in requested
    }
    missing = requested - set(result)
    if missing:
        raise ValueError(f"curve is missing windows: {sorted(missing)}")
    return result


def mean_and_sample_std(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("values cannot be empty")
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std
