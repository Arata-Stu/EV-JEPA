from __future__ import annotations

from event_window_jepa.evaluation.robustness_metrics import summarize_curve


def test_higher_is_better_degradation() -> None:
    summary = summarize_curve({10: 0.4, 40: 0.6, 80: 0.5}, 40, True)
    assert summary.worst == 0.4
    assert abs(summary.maximum_degradation - 0.2) < 1e-12


def test_lower_is_better_degradation() -> None:
    summary = summarize_curve({10: 4.0, 40: 2.0, 80: 3.0}, 40, False)
    assert summary.worst == 4.0
    assert summary.maximum_degradation == 2.0

