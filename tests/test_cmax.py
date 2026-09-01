from __future__ import annotations

import pytest
import torch

from event_window_jepa.data.packed_events import PackedEventBatch
from event_window_jepa.losses.cmax import (
    TamingCMaxLoss,
    average_timestamp_image,
    bilinear_splat_iwe,
    charbonnier_spatial_smoothness,
    sample_patch_flow_at_events,
    taming_focus_loss,
    warp_events_to_reference,
)
from event_window_jepa.models.cmax_flow import RecurrentTokenFlowHead


def _packed_events(
    *,
    x: list[float],
    y: list[float],
    t: list[float],
    polarity: list[float],
    batch_index: list[int],
    time_index: list[int],
    batch_size: int,
    time_steps: int,
    height: int,
    width: int,
) -> PackedEventBatch:
    batch_tensor = torch.tensor(batch_index, dtype=torch.int64)
    time_tensor = torch.tensor(time_index, dtype=torch.int64)
    flat_window = batch_tensor * time_steps + time_tensor
    counts = torch.bincount(flat_window, minlength=batch_size * time_steps)
    offsets = torch.empty(batch_size * time_steps + 1, dtype=torch.int64)
    offsets[0] = 0
    torch.cumsum(counts, dim=0, out=offsets[1:])
    return PackedEventBatch(
        x=torch.tensor(x, dtype=torch.float32),
        y=torch.tensor(y, dtype=torch.float32),
        t=torch.tensor(t, dtype=torch.float32),
        polarity=torch.tensor(polarity, dtype=torch.float32),
        batch_index=batch_tensor,
        time_index=time_tensor,
        window_offsets=offsets,
        window_counts=counts,
        batch_size=batch_size,
        time_steps=time_steps,
        height=height,
        width=width,
    )


def test_recurrent_token_flow_head_is_bounded_and_grid_shaped() -> None:
    head = RecurrentTokenFlowHead(
        8,
        hidden_dim=12,
        head_depth=2,
        flow_scale=0.5,
        max_displacement=3.0,
    )
    tokens = torch.randn(2, 6, 8, requires_grad=True)

    initial_flow = head(tokens, (2, 3))

    assert initial_flow.shape == (2, 2, 2, 3)
    assert torch.isfinite(initial_flow).all()
    assert torch.any(initial_flow != 0)
    with torch.no_grad():
        output_layer = head.network[-1]
        assert isinstance(output_layer, torch.nn.Conv2d)
        output_layer.bias.fill_(100.0)
    bounded_flow = head(tokens, (2, 3))
    assert torch.all(bounded_flow <= 3.0)
    assert torch.all(bounded_flow >= -3.0)
    bounded_flow.sum().backward()
    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()


def test_patch_flow_sampling_is_sparse_dynamic_and_differentiable() -> None:
    flow = torch.zeros(1, 2, 2, 2, 3, requires_grad=True)
    with torch.no_grad():
        flow[:, :, 0].fill_(2.0)
        flow[:, :, 1].fill_(-1.0)
    x = torch.tensor([0.0, 5.0, 10.0, -1.0])
    y = torch.tensor([0.0, 3.0, 6.0, 2.0])
    batch_index = torch.zeros(4, dtype=torch.int64)
    time_index = torch.tensor([0, 0, 1, 1], dtype=torch.int64)

    sampled = sample_patch_flow_at_events(
        flow,
        x,
        y,
        batch_index,
        time_index,
        (8, 12),
    )

    torch.testing.assert_close(
        sampled[:3],
        torch.tensor([[2.0, -1.0], [2.0, -1.0], [2.0, -1.0]]),
    )
    torch.testing.assert_close(sampled[3], torch.zeros(2))
    sampled.sum().backward()
    assert flow.grad is not None
    assert torch.isfinite(flow.grad).all()


def test_patch_flow_sampling_uses_patch_center_coordinates() -> None:
    flow = torch.zeros(1, 1, 2, 1, 2)
    flow[0, 0, 0, 0] = torch.tensor([1.0, 3.0])
    sampled = sample_patch_flow_at_events(
        flow,
        torch.tensor([1.5, 3.5, 5.5]),
        torch.tensor([1.5, 1.5, 1.5]),
        torch.zeros(3, dtype=torch.int64),
        torch.zeros(3, dtype=torch.int64),
        (4, 8),
    )

    torch.testing.assert_close(sampled[:, 0], torch.tensor([1.0, 2.0, 3.0]))
    torch.testing.assert_close(sampled[:, 1], torch.zeros(3))

    with pytest.raises(ValueError, match="divisible by the flow patch grid"):
        sample_patch_flow_at_events(
            flow,
            torch.tensor([1.0]),
            torch.tensor([1.0]),
            torch.zeros(1, dtype=torch.int64),
            torch.zeros(1, dtype=torch.int64),
            (5, 8),
        )


def test_linear_warp_uses_pixels_per_base_window() -> None:
    warped = warp_events_to_reference(
        torch.tensor([2.0, 4.0]),
        torch.tensor([3.0, 3.0]),
        torch.tensor([0.25, 1.5]),
        torch.tensor([[2.0, -1.0], [2.0, -1.0]]),
        2.0,
    )

    torch.testing.assert_close(
        warped,
        torch.tensor([[5.5, 1.25], [5.0, 2.5]]),
    )


def test_polarity_splat_and_average_timestamp_discard_far_outliers() -> None:
    warped = torch.tensor([[1.5, 2.5], [1.5, 2.5], [100.0, 100.0]])
    polarity = torch.tensor([1.0, -1.0, 1.0])
    batch_index = torch.zeros(3, dtype=torch.int64)
    timestamp = torch.tensor([0.25, 0.75, 0.9])

    iwe = bilinear_splat_iwe(
        warped,
        polarity,
        batch_index,
        (5, 6),
        batch_size=1,
    )
    timestamp_sum = bilinear_splat_iwe(
        warped,
        polarity,
        batch_index,
        (5, 6),
        batch_size=1,
        values=timestamp,
    )
    average = average_timestamp_image(iwe, timestamp_sum)
    focus = taming_focus_loss(iwe, average)

    torch.testing.assert_close(iwe[:, 0].sum(), torch.tensor(1.0))
    torch.testing.assert_close(iwe[:, 1].sum(), torch.tensor(1.0))
    torch.testing.assert_close(
        average[0, 0, 2:4, 1:3],
        torch.full((2, 2), 0.25),
        rtol=1e-4,
        atol=1e-6,
    )
    torch.testing.assert_close(
        average[0, 1, 2:4, 1:3],
        torch.full((2, 2), 0.75),
        rtol=1e-4,
        atol=1e-6,
    )
    # Positive and negative events share four pixels. Taming's scaling uses
    # polarity-union occupied pixels, while timestamp averages stay separate.
    torch.testing.assert_close(
        focus,
        torch.tensor([0.625]),
        rtol=1e-4,
        atol=1e-6,
    )


def test_charbonnier_smoothness_is_zero_for_constant_flow() -> None:
    constant = torch.ones(2, 3, 2, 4, 5)
    varying = constant.clone()
    varying[..., 2:, 2:] = 4.0

    constant_loss = charbonnier_spatial_smoothness(constant)
    varying_loss = charbonnier_spatial_smoothness(varying)

    torch.testing.assert_close(constant_loss, torch.tensor(0.0))
    assert varying_loss > constant_loss
    assert constant_loss.dtype == torch.float32


def test_correct_constant_motion_reduces_multiscale_bidirectional_focus() -> None:
    time_steps = 4
    local_times = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]
    x: list[float] = []
    y: list[float] = []
    t: list[float] = []
    polarity: list[float] = []
    batch_index: list[int] = []
    time_index: list[int] = []
    for step in range(time_steps):
        for local_time in local_times:
            x.append(2.0 + step + local_time)
            y.append(3.0)
            t.append(local_time)
            polarity.append(1.0)
            batch_index.append(0)
            time_index.append(step)
    events = _packed_events(
        x=x,
        y=y,
        t=t,
        polarity=polarity,
        batch_index=batch_index,
        time_index=time_index,
        batch_size=1,
        time_steps=time_steps,
        height=8,
        width=16,
    )
    correct_flow = torch.zeros(1, time_steps, 2, 2, 4)
    correct_flow[:, :, 0].fill_(1.0)
    zero_flow = torch.zeros_like(correct_flow, requires_grad=True)
    objective = TamingCMaxLoss(
        reference_mode="both",
        temporal_scales=(1, 2, 4),
        min_events=4,
    )

    correct = objective(correct_flow, events)
    stationary = objective(zero_flow, events)

    assert correct.focus_loss < stationary.focus_loss
    assert correct.valid_event_count == 16
    assert correct.valid_partition_count == 7
    assert correct.valid_window_fraction == 1
    assert 0 < correct.occupied_pixel_fraction <= 1
    torch.testing.assert_close(correct.mean_flow_magnitude, torch.tensor(1.0))
    stationary.loss.backward()
    assert zero_flow.grad is not None
    assert torch.isfinite(zero_flow.grad).all()
    assert zero_flow.grad.abs().sum() > 0


def test_integer_events_start_flow_head_with_nonzero_cmax_gradient() -> None:
    torch.manual_seed(17)
    head = RecurrentTokenFlowHead(
        8,
        hidden_dim=12,
        head_depth=2,
        flow_scale=0.01,
        max_displacement=4.0,
    )
    recurrent_tokens = torch.randn(1, 4, 8)
    flow = head(recurrent_tokens, (2, 2)).unsqueeze(1)
    events = _packed_events(
        x=[2.0, 2.0, 3.0, 3.0],
        y=[3.0, 3.0, 3.0, 3.0],
        t=[0.1, 0.3, 0.7, 0.9],
        polarity=[1.0, 1.0, 1.0, 1.0],
        batch_index=[0, 0, 0, 0],
        time_index=[0, 0, 0, 0],
        batch_size=1,
        time_steps=1,
        height=8,
        width=8,
    )
    objective = TamingCMaxLoss(
        reference_mode="both",
        temporal_scales=(1,),
        min_events=4,
    )

    objective(flow, events).loss.backward()

    output_layer = head.network[-1]
    assert isinstance(output_layer, torch.nn.Conv2d)
    assert output_layer.weight.grad is not None
    assert torch.isfinite(output_layer.weight.grad).all()
    assert output_layer.weight.grad.abs().sum() > 0


def test_min_events_filters_source_windows_not_whole_partitions() -> None:
    events = _packed_events(
        x=[2.0, 3.0, 3.5],
        y=[2.0, 2.0, 2.0],
        t=[0.5, 0.25, 0.75],
        polarity=[1.0, 1.0, -1.0],
        batch_index=[0, 0, 0],
        time_index=[0, 1, 1],
        batch_size=1,
        time_steps=2,
        height=6,
        width=8,
    )
    flow = torch.zeros(1, 2, 2, 2, 2)
    objective = TamingCMaxLoss(
        reference_mode="both",
        temporal_scales=(1, 2),
        min_events=2,
    )

    output = objective(flow, events)

    assert output.valid_event_count == 2
    assert output.valid_partition_count == 2
    assert output.valid_window_fraction == 0.5


def test_empty_packed_events_are_finite_under_autocast() -> None:
    events = _packed_events(
        x=[],
        y=[],
        t=[],
        polarity=[],
        batch_index=[],
        time_index=[],
        batch_size=1,
        time_steps=4,
        height=8,
        width=12,
    )
    flow = torch.zeros(1, 4, 2, 2, 3, requires_grad=True)
    objective = TamingCMaxLoss(
        smoothness_weight=0.1,
        reference_mode="both",
        temporal_scales=(1, 2, 4),
        min_events=1,
    )

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = objective(flow, events)

    assert output.loss.dtype == torch.float32
    assert output.loss == 0
    assert output.valid_event_count == 0
    assert output.valid_partition_count == 0
    assert output.valid_window_fraction == 0
    assert output.occupied_pixel_fraction == 0
    output.loss.backward()
    assert flow.grad is not None
    assert torch.isfinite(flow.grad).all()


def test_top_level_cmax_rejects_out_of_image_raw_coordinates() -> None:
    events = _packed_events(
        x=[8.0],
        y=[2.0],
        t=[0.5],
        polarity=[1.0],
        batch_index=[0],
        time_index=[0],
        batch_size=1,
        time_steps=1,
        height=6,
        width=8,
    )
    objective = TamingCMaxLoss(temporal_scales=(1,), min_events=1)

    with pytest.raises(ValueError, match="outside the transformed image"):
        objective(torch.zeros(1, 1, 2, 2, 2), events)
