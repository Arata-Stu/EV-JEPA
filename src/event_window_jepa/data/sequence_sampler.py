from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator, Literal, Sequence

from torch.utils.data import Sampler

from event_window_jepa.data.anchor_sampler import (
    deterministic_seed,
    milliseconds_to_microseconds,
)
from event_window_jepa.data.types import SequenceInfo


_SIGNED_SEED_MASK = (1 << 63) - 1


@dataclass(frozen=True)
class SequenceClip:
    """One strictly ordered clip drawn from a single event sequence."""

    sequence_id: str
    t_end_us: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.sequence_id:
            raise ValueError("sequence_id cannot be empty")
        if not self.t_end_us:
            raise ValueError("a sequence clip must contain at least one window")
        if any(
            current <= previous
            for previous, current in zip(self.t_end_us, self.t_end_us[1:], strict=False)
        ):
            raise ValueError("clip end timestamps must be strictly increasing")


@dataclass(frozen=True)
class RecurrentClipRequest:
    """Complete, worker-safe description of one random or stream clip."""

    clip: SequenceClip
    sampling_mode: Literal["random", "stream"]
    stream_id: str
    state_reset: bool
    augmentation_seed: int
    augmentation_id: str
    mask_seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.clip, SequenceClip):
            raise TypeError("clip must be a SequenceClip")
        if self.sampling_mode not in {"random", "stream"}:
            raise ValueError("sampling_mode must be random or stream")
        if not isinstance(self.stream_id, str):
            raise TypeError("stream_id must be a string")
        if not isinstance(self.state_reset, bool):
            raise TypeError("state_reset must be boolean")
        if self.sampling_mode == "stream" and not self.stream_id:
            raise ValueError("stream requests require a non-empty stream_id")
        if self.sampling_mode == "random" and not self.state_reset:
            raise ValueError("random requests must reset recurrent state")
        if not isinstance(self.augmentation_id, str) or not self.augmentation_id:
            raise ValueError("augmentation_id cannot be empty")
        for name, value in (
            ("augmentation_seed", self.augmentation_seed),
            ("mask_seed", self.mask_seed),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= _SIGNED_SEED_MASK
            ):
                raise ValueError(f"{name} must be a signed-64-bit-safe non-negative integer")


@dataclass(frozen=True)
class _StreamChunk:
    recording_id: str
    sequence_id: str
    chunk_index: int
    clip: SequenceClip
    state_reset: bool


class UniformSequenceClipSampler:
    """Stateless sampler for fixed-stride clips that never cross a sequence.

    ``sequence_length`` counts timesteps that contribute a training loss.
    ``burn_in_steps`` are prepended to those timesteps, so every sampled clip
    contains ``burn_in_steps + sequence_length`` windows in total.
    """

    def __init__(
        self,
        sequences: Sequence[SequenceInfo],
        *,
        base_window_ms: float,
        stride_ms: float,
        sequence_length: int,
        burn_in_steps: int,
        samples_per_epoch: int,
        seed: int = 0,
        sampling_strategy: str = "dataset_balanced",
    ) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if burn_in_steps < 0:
            raise ValueError("burn_in_steps cannot be negative")
        if samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be positive")
        if sampling_strategy not in {"dataset_balanced", "sequence_balanced"}:
            raise ValueError("unsupported sequence sampling strategy")

        self.base_window_us = milliseconds_to_microseconds(base_window_ms)
        self.stride_us = milliseconds_to_microseconds(stride_ms)
        self.sequence_length = int(sequence_length)
        self.burn_in_steps = int(burn_in_steps)
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed = int(seed)
        self.sampling_strategy = sampling_strategy
        self.total_steps = self.burn_in_steps + self.sequence_length
        self.required_duration_us = self.base_window_us + (
            self.total_steps - 1
        ) * self.stride_us

        self._sequences = tuple(
            info
            for info in sequences
            if info.t_end_us - info.t_start_us >= self.required_duration_us
        )
        if not self._sequences:
            raise ValueError("no sequence is long enough for a complete recurrent clip")
        grouped: dict[str, list[SequenceInfo]] = {}
        for info in self._sequences:
            grouped.setdefault(info.dataset, []).append(info)
        self._datasets = tuple(
            (dataset, tuple(sorted(values, key=lambda item: item.sequence_id)))
            for dataset, values in sorted(grouped.items())
        )

    @property
    def base_window_ms(self) -> float:
        return self.base_window_us / 1_000.0

    @property
    def stride_ms(self) -> float:
        return self.stride_us / 1_000.0

    @property
    def sequences(self) -> tuple[SequenceInfo, ...]:
        """Sequences long enough to produce a complete clip."""

        return self._sequences

    def __len__(self) -> int:
        return self.samples_per_epoch

    def sample(self, sample_index: int, epoch: int) -> SequenceClip:
        if not 0 <= sample_index < self.samples_per_epoch:
            raise IndexError(sample_index)
        if epoch < 0:
            raise ValueError("epoch cannot be negative")
        rng = random.Random(
            deterministic_seed(self.seed, epoch, sample_index, stream=0)
        )
        if self.sampling_strategy == "dataset_balanced":
            _, sequences = self._datasets[rng.randrange(len(self._datasets))]
            info = sequences[rng.randrange(len(sequences))]
        else:
            info = self._sequences[rng.randrange(len(self._sequences))]

        earliest_first_end = info.t_start_us + self.base_window_us
        latest_first_end = info.t_end_us - (self.total_steps - 1) * self.stride_us
        first_end = rng.randint(earliest_first_end, latest_first_end)
        end_timestamps = tuple(
            first_end + step * self.stride_us for step in range(self.total_steps)
        )
        if end_timestamps[-1] > info.t_end_us:
            raise RuntimeError("sampled recurrent clip exceeds its sequence boundary")
        return SequenceClip(sequence_id=info.sequence_id, t_end_us=end_timestamps)


class MixedRecurrentBatchSampler(Sampler[list[RecurrentClipRequest]]):
    """RVT-style stable stream lanes with optional random rows.

    ``samples_per_epoch`` is the requested number of items across all ranks.
    Complete global batches are retained, so ``len(self)`` is
    ``samples_per_epoch // (batch_size * world_size)``. Stream recordings are
    assigned once to global ``(rank, lane)`` shards. Their order is shuffled
    deterministically per epoch, while every recording's chunks remain causal
    and consecutive inside its lane. A stream recording shares one spatial
    augmentation seed for all of its chunks in an epoch. ``stream_ratio`` may
    be ``(1, 0)`` for stream-only TBPTT. ``force_stream_reset`` keeps those
    exact stream chunks and augmentations but resets every chunk, providing a
    sampling-matched full-BPTT control. The historical class name is retained
    because mixed 1:1 sampling remains its default and primary use.
    """

    def __init__(
        self,
        sequences: Sequence[SequenceInfo],
        *,
        base_window_ms: float,
        stride_ms: float,
        sequence_length: int,
        burn_in_steps: int,
        samples_per_epoch: int,
        batch_size: int,
        stream_ratio: tuple[int, int] = (1, 1),
        world_size: int = 1,
        rank: int = 0,
        seed: int = 0,
        random_sampling_strategy: str = "dataset_balanced",
        force_stream_reset: bool = False,
    ) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if burn_in_steps < 0:
            raise ValueError("burn_in_steps cannot be negative")
        if samples_per_epoch <= 0 or batch_size <= 0:
            raise ValueError("epoch and batch sizes must be positive")
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("rank must lie inside a positive world_size")
        if not isinstance(force_stream_reset, bool):
            raise TypeError("force_stream_reset must be boolean")
        if (
            len(stream_ratio) != 2
            or stream_ratio[0] <= 0
            or stream_ratio[1] < 0
        ):
            raise ValueError(
                "stream_ratio must contain positive stream and non-negative random weights"
            )
        if stream_ratio[1] and stream_ratio[0] != stream_ratio[1]:
            raise ValueError("mixed stream/random sampling requires a 1:1 ratio")
        if batch_size % sum(stream_ratio):
            raise ValueError("batch_size must be divisible by the random/stream ratio sum")

        self.base_window_us = milliseconds_to_microseconds(base_window_ms)
        self.stride_us = milliseconds_to_microseconds(stride_ms)
        if self.base_window_us != self.stride_us:
            raise ValueError("mixed streaming requires stride_ms == base_window_ms")
        self.sequence_length = int(sequence_length)
        self.burn_in_steps = int(burn_in_steps)
        self.total_steps = self.burn_in_steps + self.sequence_length
        self.samples_per_epoch = int(samples_per_epoch)
        self.batch_size = int(batch_size)
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.seed = int(seed)
        self.force_stream_reset = force_stream_reset
        self.epoch = 0
        ratio_sum = sum(stream_ratio)
        self.stream_batch_size = self.batch_size * stream_ratio[0] // ratio_sum
        self.random_batch_size = self.batch_size - self.stream_batch_size
        if self.stream_batch_size <= 0:
            raise ValueError("stream-aware batches require at least one stream item")

        self.global_batch_size = self.batch_size * self.world_size
        self.batches_per_epoch = self.samples_per_epoch // self.global_batch_size
        if self.batches_per_epoch <= 0:
            raise ValueError("samples_per_epoch is smaller than one complete global batch")
        self.effective_samples_per_epoch = (
            self.batches_per_epoch * self.global_batch_size
        )

        sequence_values = tuple(sequences)
        if not sequence_values:
            raise ValueError("mixed sampler requires event sequences")
        self._metadata = {info.sequence_id: info for info in sequence_values}
        if len(self._metadata) != len(sequence_values):
            raise ValueError("mixed sampler sequence ids must be unique")
        self._recording_chunks = self._build_recording_chunks(sequence_values)
        self._recording_indices = {
            recording_id: index
            for index, recording_id in enumerate(sorted(self._recording_chunks))
        }

        global_stream_lanes = self.world_size * self.stream_batch_size
        recording_ids = list(sorted(self._recording_chunks))
        assignment_rng = random.Random(
            deterministic_seed(self.seed, 0, 0, stream=101)
        )
        assignment_rng.shuffle(recording_ids)
        if len(recording_ids) < global_stream_lanes:
            raise ValueError(
                "stream sampling needs at least one recording per global stream lane"
            )
        shards: list[list[str]] = [[] for _ in range(global_stream_lanes)]
        for index, recording_id in enumerate(recording_ids):
            shards[index % global_stream_lanes].append(recording_id)
        self._lane_recordings = tuple(tuple(values) for values in shards)

        random_items = self.batches_per_epoch * self.world_size * self.random_batch_size
        self._random_sampler = (
            UniformSequenceClipSampler(
                sequence_values,
                base_window_ms=self.base_window_us / 1_000.0,
                stride_ms=self.stride_us / 1_000.0,
                sequence_length=self.sequence_length,
                burn_in_steps=self.burn_in_steps,
                samples_per_epoch=random_items,
                seed=self.seed,
                sampling_strategy=random_sampling_strategy,
            )
            if random_items
            else None
        )

    def _build_recording_chunks(
        self, sequences: Sequence[SequenceInfo]
    ) -> dict[str, tuple[_StreamChunk, ...]]:
        grouped: dict[str, list[SequenceInfo]] = {}
        for info in sequences:
            recording_id = info.source_recording_id or info.sequence_id
            grouped.setdefault(recording_id, []).append(info)

        recordings: dict[str, tuple[_StreamChunk, ...]] = {}
        for recording_id, recording_sequences in sorted(grouped.items()):
            chunks: list[_StreamChunk] = []
            ordered = sorted(
                recording_sequences,
                key=lambda info: (info.t_start_us, info.t_end_us, info.sequence_id),
            )
            for info in ordered:
                available_steps = (
                    (info.t_end_us - info.t_start_us - self.base_window_us)
                    // self.stride_us
                    + 1
                )
                complete_chunks = max(0, available_steps) // self.total_steps
                for chunk_index in range(complete_chunks):
                    first_end = (
                        info.t_start_us
                        + self.base_window_us
                        + chunk_index * self.total_steps * self.stride_us
                    )
                    end_timestamps = tuple(
                        first_end + step * self.stride_us
                        for step in range(self.total_steps)
                    )
                    chunks.append(
                        _StreamChunk(
                            recording_id=recording_id,
                            sequence_id=info.sequence_id,
                            chunk_index=chunk_index,
                            clip=SequenceClip(info.sequence_id, end_timestamps),
                            state_reset=chunk_index == 0,
                        )
                    )
            if chunks:
                recordings[recording_id] = tuple(chunks)
        if not recordings:
            raise ValueError("no recording contains a complete recurrent stream chunk")
        return recordings

    def __len__(self) -> int:
        return self.batches_per_epoch

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch cannot be negative")
        self.epoch = epoch

    def _ordered_lane_chunks(self, global_lane: int) -> tuple[_StreamChunk, ...]:
        recording_ids = list(self._lane_recordings[global_lane])
        rng = random.Random(
            deterministic_seed(self.seed, self.epoch, global_lane, stream=102)
        )
        rng.shuffle(recording_ids)
        chunks: list[_StreamChunk] = []
        for recording_id in recording_ids:
            chunks.extend(self._recording_chunks[recording_id])
        if not chunks:
            raise RuntimeError("a fixed stream lane unexpectedly contains no chunks")
        return tuple(chunks)

    def _stream_request(
        self,
        chunk: _StreamChunk,
        *,
        global_lane: int,
        batch_index: int,
        lane_cycle: int,
    ) -> RecurrentClipRequest:
        recording_index = self._recording_indices[chunk.recording_id]
        augmentation_seed = deterministic_seed(
            self.seed, self.epoch, recording_index, stream=103
        ) & _SIGNED_SEED_MASK
        mask_coordinate = (
            batch_index * self.world_size * self.stream_batch_size
            + self.rank * self.stream_batch_size
            + global_lane % self.stream_batch_size
        )
        mask_seed = deterministic_seed(
            self.seed,
            self.epoch + lane_cycle,
            mask_coordinate,
            stream=104,
        ) & _SIGNED_SEED_MASK
        return RecurrentClipRequest(
            clip=chunk.clip,
            sampling_mode="stream",
            stream_id=f"rank-{self.rank}:lane-{global_lane % self.stream_batch_size}",
            state_reset=chunk.state_reset or self.force_stream_reset,
            augmentation_seed=augmentation_seed,
            augmentation_id=(
                f"recording:{chunk.recording_id}:epoch-{self.epoch}:"
                f"augmentation-{augmentation_seed}"
            ),
            mask_seed=mask_seed,
        )

    def _random_request(self, global_random_index: int) -> RecurrentClipRequest:
        if self._random_sampler is None:
            raise RuntimeError("a stream-only sampler cannot create random requests")
        clip = self._random_sampler.sample(global_random_index, self.epoch)
        augmentation_seed = deterministic_seed(
            self.seed, self.epoch, global_random_index, stream=105
        ) & _SIGNED_SEED_MASK
        mask_seed = deterministic_seed(
            self.seed, self.epoch, global_random_index, stream=106
        ) & _SIGNED_SEED_MASK
        return RecurrentClipRequest(
            clip=clip,
            sampling_mode="random",
            stream_id="",
            state_reset=True,
            augmentation_seed=augmentation_seed,
            augmentation_id=(
                f"random:epoch-{self.epoch}:item-{global_random_index}:"
                f"augmentation-{augmentation_seed}"
            ),
            mask_seed=mask_seed,
        )

    def __iter__(self) -> Iterator[list[RecurrentClipRequest]]:
        lane_chunks: list[tuple[_StreamChunk, ...]] = []
        for local_lane in range(self.stream_batch_size):
            global_lane = self.rank * self.stream_batch_size + local_lane
            lane_chunks.append(self._ordered_lane_chunks(global_lane))

        for batch_index in range(self.batches_per_epoch):
            batch: list[RecurrentClipRequest] = []
            for local_lane, chunks in enumerate(lane_chunks):
                global_lane = self.rank * self.stream_batch_size + local_lane
                position = batch_index % len(chunks)
                lane_cycle = batch_index // len(chunks)
                chunk = chunks[position]
                request = self._stream_request(
                    chunk,
                    global_lane=global_lane,
                    batch_index=batch_index,
                    lane_cycle=lane_cycle,
                )
                if position == 0 and lane_cycle > 0 and not request.state_reset:
                    request = RecurrentClipRequest(
                        clip=request.clip,
                        sampling_mode=request.sampling_mode,
                        stream_id=request.stream_id,
                        state_reset=True,
                        augmentation_seed=request.augmentation_seed,
                        augmentation_id=request.augmentation_id,
                        mask_seed=request.mask_seed,
                    )
                batch.append(request)

            random_base = (
                batch_index * self.world_size * self.random_batch_size
                + self.rank * self.random_batch_size
            )
            batch.extend(
                self._random_request(random_base + offset)
                for offset in range(self.random_batch_size)
            )
            yield batch
