from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

from event_window_jepa.data.types import SequenceInfo


_MASK_64 = (1 << 64) - 1


def deterministic_seed(seed: int, epoch: int, sample_index: int, stream: int = 0) -> int:
    """Mix integer coordinates without relying on Python's randomized hash()."""

    value = (
        (seed & _MASK_64)
        ^ ((epoch + 0x9E3779B97F4A7C15) & _MASK_64)
        ^ (((sample_index + 1) * 0xBF58476D1CE4E5B9) & _MASK_64)
        ^ (((stream + 1) * 0x94D049BB133111EB) & _MASK_64)
    )
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK_64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK_64
    return (value ^ (value >> 31)) & _MASK_64


def milliseconds_to_microseconds(duration_ms: float) -> int:
    duration_us = int(round(float(duration_ms) * 1_000.0))
    if duration_us <= 0:
        raise ValueError("duration must round to at least one microsecond")
    return duration_us


@dataclass(frozen=True)
class Anchor:
    sequence_id: str
    t_end_us: int


class UniformTimeAnchorSampler:
    """Stateless, sequence-balanced sampler whose timestamps are time-uniform."""

    def __init__(
        self,
        sequences: Sequence[SequenceInfo],
        maximum_window_ms: float,
        samples_per_epoch: int,
        seed: int = 0,
        sampling_strategy: str = "dataset_balanced",
    ) -> None:
        if samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be positive")
        self.maximum_window_us = milliseconds_to_microseconds(maximum_window_ms)
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        if sampling_strategy not in {"dataset_balanced", "sequence_balanced"}:
            raise ValueError("unsupported sequence sampling strategy")
        self.sampling_strategy = sampling_strategy
        self._sequences = tuple(
            info
            for info in sequences
            if info.t_end_us - info.t_start_us >= self.maximum_window_us
        )
        if not self._sequences:
            raise ValueError("no sequence is long enough for the maximum window")
        grouped: dict[str, list[SequenceInfo]] = {}
        for info in self._sequences:
            grouped.setdefault(info.dataset, []).append(info)
        self._datasets = tuple(
            (dataset, tuple(sorted(values, key=lambda item: item.sequence_id)))
            for dataset, values in sorted(grouped.items())
        )

    def __len__(self) -> int:
        return self.samples_per_epoch

    def sample(self, sample_index: int, epoch: int) -> Anchor:
        if not 0 <= sample_index < self.samples_per_epoch:
            raise IndexError(sample_index)
        rng = random.Random(deterministic_seed(self.seed, epoch, sample_index, stream=0))
        if self.sampling_strategy == "dataset_balanced":
            _, sequences = self._datasets[rng.randrange(len(self._datasets))]
            info = sequences[rng.randrange(len(sequences))]
        else:
            info = self._sequences[rng.randrange(len(self._sequences))]
        earliest_end = info.t_start_us + self.maximum_window_us
        # randint is inclusive at both ends, which lets an anchor coincide with
        # the final event/label timestamp without ever crossing it.
        t_end_us = rng.randint(earliest_end, info.t_end_us)
        return Anchor(sequence_id=info.sequence_id, t_end_us=t_end_us)


@dataclass(frozen=True)
class WindowPair:
    context_ms: float
    target_ms: float


class WindowPairSampler:
    """Sample valid accumulation-window pairs independently of event density."""

    def __init__(
        self,
        context_windows_ms: Sequence[float],
        target_windows_ms: Sequence[float],
        minimum_ratio: float = 1.5,
        direction: str = "any",
        allow_equal: bool = False,
    ) -> None:
        if minimum_ratio < 1.0:
            raise ValueError("minimum_ratio must be at least one")
        if direction not in {"any", "short_to_long", "long_to_short"}:
            raise ValueError("unsupported pair direction")
        context = tuple(float(value) for value in context_windows_ms)
        target = tuple(float(value) for value in target_windows_ms)
        if not context or not target or min(context + target) <= 0:
            raise ValueError("window sets must contain positive durations")

        valid: list[WindowPair] = []
        for context_ms in context:
            for target_ms in target:
                equal = math.isclose(context_ms, target_ms, rel_tol=0.0, abs_tol=1e-9)
                if equal:
                    if allow_equal:
                        valid.append(WindowPair(context_ms, target_ms))
                    continue
                if max(context_ms, target_ms) / min(context_ms, target_ms) < minimum_ratio:
                    continue
                if direction == "short_to_long" and not context_ms < target_ms:
                    continue
                if direction == "long_to_short" and not context_ms > target_ms:
                    continue
                valid.append(WindowPair(context_ms, target_ms))
        if not valid:
            raise ValueError("window constraints leave no valid context/target pair")
        self._valid_pairs = tuple(valid)

    @property
    def valid_pairs(self) -> tuple[WindowPair, ...]:
        return self._valid_pairs

    def sample(self, rng: random.Random) -> WindowPair:
        return self._valid_pairs[rng.randrange(len(self._valid_pairs))]
