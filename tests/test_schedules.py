from __future__ import annotations

from event_window_jepa.train.callbacks import ema_momentum_at_step


def test_ema_schedule_increases_toward_one() -> None:
    values = [ema_momentum_at_step(step, 100, 0.996, 1.0) for step in (0, 50, 99)]
    assert values[0] == 0.996
    assert values[0] < values[1] < values[2]
    assert values[2] == 1.0
