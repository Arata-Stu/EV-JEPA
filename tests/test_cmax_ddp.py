from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from event_window_jepa.data.packed_events import PackedEventBatch
from event_window_jepa.losses.cmax import CMaxOutput, TamingCMaxLoss


_TIMESTAMP_PATTERNS = (
    (0.05, 0.10, 0.20, 0.40),
    (0.10, 0.35, 0.65, 0.90),
    (0.55, 0.70, 0.82, 0.95),
    (0.05, 0.45, 0.55, 0.95),
)


def _packed_events(
    row_ids: tuple[int, ...],
    valid_rows: tuple[bool, ...],
) -> PackedEventBatch:
    if len(row_ids) != len(valid_rows) or not row_ids:
        raise ValueError("row_ids and valid_rows must have equal non-zero lengths")
    time_steps = 2
    x: list[float] = []
    y: list[float] = []
    timestamp: list[float] = []
    polarity: list[float] = []
    batch_index: list[int] = []
    time_index: list[int] = []
    counts: list[int] = []
    for local_row, (row_id, is_valid) in enumerate(zip(row_ids, valid_rows)):
        for step in range(time_steps):
            counts.append(4 if is_valid else 0)
            if not is_valid:
                continue
            local_timestamps = _TIMESTAMP_PATTERNS[row_id]
            for event_index, local_timestamp in enumerate(local_timestamps):
                x.append(float(2 + 3 * row_id + step + event_index // 2))
                y.append(float(2 + event_index % 2))
                timestamp.append(local_timestamp)
                polarity.append(1.0 if event_index % 2 == 0 else -1.0)
                batch_index.append(local_row)
                time_index.append(step)

    window_counts = torch.tensor(counts, dtype=torch.int64)
    window_offsets = torch.empty(window_counts.numel() + 1, dtype=torch.int64)
    window_offsets[0] = 0
    torch.cumsum(window_counts, dim=0, out=window_offsets[1:])
    return PackedEventBatch(
        x=torch.tensor(x, dtype=torch.float32),
        y=torch.tensor(y, dtype=torch.float32),
        t=torch.tensor(timestamp, dtype=torch.float32),
        polarity=torch.tensor(polarity, dtype=torch.float32),
        batch_index=torch.tensor(batch_index, dtype=torch.int64),
        time_index=torch.tensor(time_index, dtype=torch.int64),
        window_offsets=window_offsets,
        window_counts=window_counts,
        batch_size=len(row_ids),
        time_steps=time_steps,
        height=8,
        width=16,
    )


class _CMaxFixture(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        initial_flow = torch.linspace(-0.08, 0.08, 32).reshape(1, 2, 2, 2, 4)
        self.flow = nn.Parameter(initial_flow)
        self.objective = TamingCMaxLoss(
            reference_mode="both",
            temporal_scales=(1, 2),
            min_events=4,
        )

    def forward(self, events: PackedEventBatch) -> CMaxOutput:
        flow = self.flow.expand(events.batch_size, -1, -1, -1, -1)
        return self.objective(flow, events)


def _valid_rows(rank: int, scenario: str) -> tuple[bool, bool]:
    if scenario == "uneven":
        return (True, True) if rank == 0 else (True, False)
    if scenario == "zero-support":
        return (True, True) if rank == 0 else (False, False)
    raise ValueError(f"unknown CMax DDP scenario: {scenario}")


def _ddp_worker(
    rank: int,
    world_size: int,
    initialization_file: str,
    output_directory: str,
    scenario: str,
) -> None:
    distributed.init_process_group(
        "gloo",
        init_method=f"file://{initialization_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        module = DistributedDataParallel(_CMaxFixture())
        events = _packed_events(
            (rank * 2, rank * 2 + 1),
            _valid_rows(rank, scenario),
        )
        output = module(events)
        output.loss.backward()
        torch.save(
            {
                "loss": output.loss.detach(),
                "focus": output.focus_loss.detach(),
                "forward": output.forward_focus_loss.detach(),
                "backward": output.backward_focus_loss.detach(),
                "occupied": output.occupied_pixel_fraction.detach(),
                "valid_events": output.valid_event_count.detach(),
                "valid_partitions": output.valid_partition_count.detach(),
                "valid_windows": output.valid_window_fraction.detach(),
                "gradient": module.module.flow.grad.detach(),
            },
            Path(output_directory) / f"{scenario}-rank-{rank}.pt",
        )
    finally:
        distributed.destroy_process_group()


def _reference(scenario: str) -> tuple[_CMaxFixture, CMaxOutput]:
    valid_rows = _valid_rows(0, scenario) + _valid_rows(1, scenario)
    module = _CMaxFixture()
    output = module(_packed_events((0, 1, 2, 3), valid_rows))
    output.loss.backward()
    return module, output


def _run_and_compare(tmp_path: Path, scenario: str) -> None:
    world_size = 2
    initialization_file = tmp_path / f"gloo-{scenario}-init"
    multiprocessing.spawn(
        _ddp_worker,
        args=(
            world_size,
            str(initialization_file),
            str(tmp_path),
            scenario,
        ),
        nprocs=world_size,
        join=True,
    )

    reference_module, reference_output = _reference(scenario)
    assert reference_module.flow.grad is not None
    assert reference_module.flow.grad.abs().sum() > 0
    local_outputs: list[CMaxOutput] = []
    local_weights: list[float] = []
    for rank in range(world_size):
        valid_rows = _valid_rows(rank, scenario)
        local_outputs.append(
            _CMaxFixture()(
                _packed_events(
                    (rank * 2, rank * 2 + 1),
                    valid_rows,
                )
            )
        )
        local_weights.append(float(sum(valid_rows)))
    total_weight = sum(local_weights)

    def weighted_local_mean(field: str) -> torch.Tensor:
        return sum(
            getattr(output, field) * weight
            for output, weight in zip(local_outputs, local_weights)
        ) / total_weight

    torch.testing.assert_close(
        reference_output.focus_loss,
        weighted_local_mean("focus_loss"),
    )
    torch.testing.assert_close(
        reference_output.forward_focus_loss,
        weighted_local_mean("forward_focus_loss"),
    )
    torch.testing.assert_close(
        reference_output.backward_focus_loss,
        weighted_local_mean("backward_focus_loss"),
    )
    expected_valid_partitions = (
        (6.0, 3.0) if scenario == "uneven" else (6.0, 0.0)
    )
    expected_valid_events = (
        (16.0, 8.0) if scenario == "uneven" else (16.0, 0.0)
    )
    expected_valid_windows = (
        (1.0, 0.5) if scenario == "uneven" else (1.0, 0.0)
    )

    for rank in range(world_size):
        result = torch.load(
            tmp_path / f"{scenario}-rank-{rank}.pt",
            map_location="cpu",
            weights_only=True,
        )
        torch.testing.assert_close(result["loss"], reference_output.loss)
        torch.testing.assert_close(result["focus"], reference_output.focus_loss)
        torch.testing.assert_close(
            result["forward"],
            reference_output.forward_focus_loss,
        )
        torch.testing.assert_close(
            result["backward"],
            reference_output.backward_focus_loss,
        )
        torch.testing.assert_close(
            result["occupied"],
            reference_output.occupied_pixel_fraction,
        )
        torch.testing.assert_close(
            result["gradient"],
            reference_module.flow.grad,
            rtol=2e-5,
            atol=1e-6,
        )
        assert result["valid_partitions"].item() == expected_valid_partitions[rank]
        assert result["valid_events"].item() == expected_valid_events[rank]
        assert result["valid_windows"].item() == expected_valid_windows[rank]


@pytest.mark.skipif(
    not distributed.is_available() or not distributed.is_gloo_available(),
    reason="CPU gloo is required for the CMax DDP contract",
)
def test_cmax_ddp_matches_global_valid_partition_mean(tmp_path: Path) -> None:
    _run_and_compare(tmp_path, "uneven")


@pytest.mark.skipif(
    not distributed.is_available() or not distributed.is_gloo_available(),
    reason="CPU gloo is required for the CMax DDP contract",
)
def test_cmax_ddp_zero_support_rank_has_matching_global_gradient(
    tmp_path: Path,
) -> None:
    _run_and_compare(tmp_path, "zero-support")
