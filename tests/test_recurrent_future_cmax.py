from __future__ import annotations

import torch

from event_window_jepa.config import ExperimentConfig
from event_window_jepa.data.packed_events import PackedEventBatch
from event_window_jepa.train.checkpoint import (
    load_pretrained_model,
    save_checkpoint_atomic,
)
from event_window_jepa.train.pretrain import build_model


def _config() -> ExperimentConfig:
    return ExperimentConfig.from_mapping(
        {
            "data": {
                "manifest": "unused.jsonl",
                "batch_size": 2,
                "crop_size": [16, 16],
            },
            "representation": {"type": "event_image"},
            "windows": {
                "train_ms": [50],
                "target_ms": [50],
                "canonical_ms": 50,
                "eval_ms": [25, 50],
                "unseen_eval_ms": [25],
                "minimum_ratio": 1.0,
                "allow_equal": True,
            },
            "model": {
                "architecture": "vjepa2_1",
                "image_size": [16, 16],
                "patch_size": 8,
                "embed_dim": 32,
                "encoder_depth": 2,
                "encoder_heads": 4,
                "predictor_dim": 16,
                "predictor_depth": 1,
                "predictor_heads": 4,
                "scale_dim": 16,
                "scale_fourier_bands": 2,
                "deep_supervision_layers": [1],
            },
            "recurrent": {
                "sequence_loader": True,
                "temporal_model": "conv_lstm",
                "return_patch_event_activity": True,
                "recurrent_placement": "post_encoder",
                "prediction_horizon_steps": 1,
                "window_ms": 50,
                "stride_ms": 50,
                "sequence_length": 2,
                "burn_in_steps": 1,
                "tbptt_steps": 2,
            },
            "future_prediction": {
                "frame_sigreg_weight": 0.02,
                "temporal_sigreg_weight": 0.0,
                "projector_hidden_dim": 32,
                "projector_output_dim": 16,
                "sigreg_num_slices": 8,
                "sigreg_num_points": 5,
            },
            "cmax": {
                "enabled": True,
                "weight": 0.05,
                "smoothness_weight": 0.001,
                "hidden_dim": 16,
                "head_depth": 2,
                "reference_mode": "both",
                "temporal_scales": [1, 2],
                "min_events": 4,
                "flow_scale": 0.1,
                "max_displacement": 4.0,
            },
            "optimization": {
                "objective": "recurrent_future_jepa",
                "epochs": 2,
                "warmup_epochs": 0,
                "precision": "fp32",
                "canonical_query_weight": 0,
            },
        }
    )


def _packed_events() -> PackedEventBatch:
    events_per_window = 4
    batch_size = 2
    time_steps = 2
    batch_index = torch.repeat_interleave(
        torch.arange(batch_size, dtype=torch.int64),
        time_steps * events_per_window,
    )
    time_index = torch.tensor(
        [0] * events_per_window
        + [1] * events_per_window
        + [0] * events_per_window
        + [1] * events_per_window,
        dtype=torch.int64,
    )
    local_time = torch.tensor([0.1, 0.35, 0.65, 0.9] * 4)
    global_time = time_index.float() + local_time
    x = 3.0 + global_time + 0.5 * batch_index.float()
    y = 4.0 + torch.tensor([0.0, 1.0, 0.0, 1.0] * 4)
    polarity = torch.tensor([1.0, -1.0, 1.0, -1.0] * 4)
    counts = torch.full(
        (batch_size * time_steps,),
        events_per_window,
        dtype=torch.int64,
    )
    offsets = torch.arange(
        0,
        batch_size * time_steps * events_per_window + 1,
        events_per_window,
        dtype=torch.int64,
    )
    return PackedEventBatch(
        x=x,
        y=y,
        t=local_time,
        polarity=polarity,
        batch_index=batch_index,
        time_index=time_index,
        window_offsets=offsets,
        window_counts=counts,
        batch_size=batch_size,
        time_steps=time_steps,
        height=16,
        width=16,
    )


def _forward(model: torch.nn.Module):
    batch_size, steps = 2, 2
    context = torch.randn(batch_size, steps, 2, 16, 16)
    future = torch.randn_like(context)
    duration = torch.full((batch_size, steps), 50.0)
    context_mask = torch.ones(batch_size, steps, 4, dtype=torch.bool)
    target_mask = context_mask.clone()
    context_activity = torch.tensor(
        [[[4, 0, 2, 0], [0, 3, 0, 1]], [[0, 2, 1, 0], [5, 0, 0, 2]]]
    )
    future_activity = torch.tensor(
        [[[0, 3, 0, 2], [4, 0, 1, 0]], [[2, 0, 0, 1], [0, 5, 2, 0]]]
    )
    return model(
        context,
        future,
        duration,
        duration,
        context_mask,
        target_mask,
        objective="recurrent_future_jepa",
        context_event_activity=context_activity,
        target_event_activity=future_activity,
        packed_events=_packed_events(),
    )


def test_cmax_is_integrated_into_recurrent_future_loss_and_gradients() -> None:
    torch.manual_seed(41)
    model = build_model(_config())

    output = _forward(model)

    assert output.cmax_loss is not None and torch.isfinite(output.cmax_loss)
    assert output.cmax_focus_loss is not None
    assert output.cmax_smoothness_loss is not None
    assert output.cmax_weighted_loss is not None
    assert output.cmax_valid_event_count == 16
    assert output.cmax_valid_partition_count == 6
    assert output.cmax_valid_window_fraction == 1
    assert output.cmax_occupied_pixel_fraction is not None
    assert output.cmax_occupied_pixel_fraction > 0
    assert output.cmax_flow_saturation_fraction == 0
    assert output.future_prediction_loss is not None
    torch.testing.assert_close(
        output.loss,
        output.future_prediction_loss
        + output.sigreg_loss
        + output.cmax_weighted_loss,
    )

    output.loss.backward()
    assert model.cmax_flow_head is not None
    output_layer = model.cmax_flow_head.network[-1]
    assert isinstance(output_layer, torch.nn.Conv2d)
    assert output_layer.weight.grad is not None
    assert torch.isfinite(output_layer.weight.grad).all()
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())


def test_cmax_flow_head_checkpoint_roundtrip(tmp_path) -> None:
    torch.manual_seed(43)
    config = _config()
    model = build_model(config)
    assert model.cmax_flow_head is not None
    with torch.no_grad():
        model.cmax_flow_head.network[-1].bias.fill_(0.75)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "cmax.pt"
    save_checkpoint_atomic(
        path,
        model,
        optimizer,
        config,
        epoch=1,
        global_step=1,
        world_size=1,
        steps_per_epoch=1,
    )

    loaded, loaded_config = load_pretrained_model(path)

    assert loaded_config.cmax.enabled
    assert loaded.cmax_flow_head is not None
    for name, expected in model.cmax_flow_head.state_dict().items():
        assert torch.equal(loaded.cmax_flow_head.state_dict()[name], expected)
