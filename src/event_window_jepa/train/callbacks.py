from __future__ import annotations

import math


def cosine_value(start: float, end: float, step: int, total_steps: int) -> float:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    progress = min(max(step / total_steps, 0.0), 1.0)
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))


def learning_rate_at_step(
    step: int,
    total_steps: int,
    warmup_steps: int,
    peak: float,
    final: float,
) -> float:
    if warmup_steps < 0 or warmup_steps >= total_steps:
        raise ValueError("warmup_steps must be in [0, total_steps)")
    if step < warmup_steps:
        return peak * (step + 1) / max(warmup_steps, 1)
    decay_steps = total_steps - warmup_steps
    if decay_steps == 1:
        return final
    return cosine_value(peak, final, step - warmup_steps, decay_steps - 1)


def ema_momentum_at_step(
    step: int, total_steps: int, start: float = 0.996, end: float = 1.0
) -> float:
    return cosine_value(start, end, step, max(total_steps - 1, 1))
