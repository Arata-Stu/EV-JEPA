from __future__ import annotations

import copy

import pytest
import torch

from event_window_jepa.config import ExperimentConfig
from event_window_jepa.downstream.features import extract_patch_features
from event_window_jepa.models.recurrent_vjepa21_event_vit import (
    RecurrentState,
    RecurrentVJEPA21EventVisionTransformer,
    detach_recurrent_state,
)
from event_window_jepa.models.scale_embedding import LogFourierScaleEmbedding
from event_window_jepa.models.window_jepa import WindowJEPA
from event_window_jepa.models.window_predictor import WindowPredictor
from event_window_jepa.train.checkpoint import (
    load_pretrained_model,
    save_checkpoint_atomic,
)
from event_window_jepa.train.pretrain import (
    _recurrent_backward,
    _recurrent_chunk_ranges,
    _validate_mixed_recurrent_batch,
)


def _model() -> WindowJEPA:
    encoder = RecurrentVJEPA21EventVisionTransformer(
        image_size=(16, 16),
        patch_size=8,
        input_channels=2,
        embed_dim=32,
        depth=2,
        num_heads=4,
        scale_dim=16,
        supervision_layers=(0, 1),
        recurrent_cell="conv_lstm",
    )
    predictor = WindowPredictor(
        num_patches=4,
        encoder_dim=32,
        predictor_dim=16,
        depth=1,
        num_heads=4,
        scale_dim=16,
    )
    return WindowJEPA(
        encoder,
        predictor,
        LogFourierScaleEmbedding(output_dim=16, num_bands=2),
        canonical_query_weight=0.0,
    )


def _masks(batch_size: int, steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    context = torch.tensor([True, False, True, False]).reshape(1, 1, 4)
    target = ~context
    return (
        context.expand(batch_size, steps, -1).clone(),
        target.expand(batch_size, steps, -1).clone(),
    )


def _r0_config() -> ExperimentConfig:
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
                "deep_supervision_layers": [0, 1],
            },
            "recurrent": {
                "enabled": True,
                "window_ms": 50,
                "stride_ms": 50,
                "sequence_length": 3,
                "burn_in_steps": 1,
                "tbptt_steps": 2,
            },
            "optimization": {
                "objective": "recurrent_dense_window_jepa",
                "epochs": 2,
                "warmup_epochs": 0,
                "precision": "fp32",
                "canonical_query_weight": 0,
            },
        }
    )


def _state_equal(first: RecurrentState | None, second: RecurrentState | None) -> bool:
    if isinstance(first, tuple) and isinstance(second, tuple):
        return torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])
    return isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor) and (
        torch.equal(first, second)
    )


@pytest.mark.parametrize(
    "objective", ["recurrent_window_jepa", "recurrent_dense_window_jepa"]
)
def test_target_input_cannot_change_prediction_or_online_state(objective: str) -> None:
    torch.manual_seed(3)
    model = _model().eval()
    batch_size, steps = 2, 3
    context = torch.randn(batch_size, steps, 2, 16, 16)
    first_target = torch.randn_like(context)
    changed_target = first_target + 20.0
    duration = torch.full((batch_size, steps), 50.0)
    context_mask, target_mask = _masks(batch_size, steps)

    first = model(
        context,
        first_target,
        duration,
        duration,
        context_mask,
        target_mask,
        objective=objective,
    )
    changed = model(
        context,
        changed_target,
        duration,
        duration,
        context_mask,
        target_mask,
        objective=objective,
    )

    assert torch.equal(first.prediction, changed.prediction)
    assert _state_equal(first.online_state, changed.online_state)
    assert not torch.equal(first.target, changed.target)


@pytest.mark.parametrize(
    "objective", ["recurrent_window_jepa", "recurrent_dense_window_jepa"]
)
def test_target_encoder_resets_state_at_every_current_time_step(objective: str) -> None:
    torch.manual_seed(5)
    model = _model().eval()
    batch_size, steps = 2, 3
    first_context = torch.randn(batch_size, steps, 2, 16, 16)
    second_context = first_context.clone()
    second_context[:, :-1] += 10.0
    first_target = torch.randn_like(first_context)
    second_target = first_target.clone()
    second_target[:, :-1] -= 10.0
    duration = torch.full((batch_size, steps), 50.0)
    context_mask, target_mask = _masks(batch_size, steps)

    first = model(
        first_context,
        first_target,
        duration,
        duration,
        context_mask,
        target_mask,
        objective=objective,
    )
    second = model(
        second_context,
        second_target,
        duration,
        duration,
        context_mask,
        target_mask,
        objective=objective,
    )

    # Outputs are concatenated time-major: [t0 batch, t1 batch, t2 batch].
    assert torch.equal(first.target[-batch_size:], second.target[-batch_size:])


def test_bptt_reaches_earlier_steps_but_detached_tbptt_state_does_not() -> None:
    torch.manual_seed(7)
    model = _model()
    one_context_mask, one_target_mask = _masks(batch_size=2, steps=1)
    one_step_duration = torch.full((2, 1), 50.0)
    first_context = torch.randn(2, 1, 2, 16, 16, requires_grad=True)
    second_context = torch.randn(2, 1, 2, 16, 16, requires_grad=True)
    first = model(
        first_context,
        torch.randn_like(first_context),
        one_step_duration,
        one_step_duration,
        one_context_mask,
        one_target_mask,
        objective="recurrent_window_jepa",
    )
    second = model(
        second_context,
        torch.randn_like(second_context),
        one_step_duration,
        one_step_duration,
        one_context_mask,
        one_target_mask,
        objective="recurrent_window_jepa",
        online_state=first.online_state,
    )
    # Only the future loss is differentiated. A gradient on the first input
    # therefore proves propagation through recurrent state across timesteps.
    second.loss.backward()
    assert first_context.grad is not None
    assert first_context.grad.abs().sum() > 0
    assert second_context.grad is not None
    assert second_context.grad.abs().sum() > 0

    model.zero_grad(set_to_none=True)
    first_context = torch.randn(2, 1, 2, 16, 16, requires_grad=True)
    second_context = torch.randn(2, 1, 2, 16, 16, requires_grad=True)
    first = model(
        first_context,
        torch.randn_like(first_context),
        one_step_duration,
        one_step_duration,
        one_context_mask,
        one_target_mask,
        objective="recurrent_window_jepa",
    )
    second = model(
        second_context,
        torch.randn_like(second_context),
        one_step_duration,
        one_step_duration,
        one_context_mask,
        one_target_mask,
        objective="recurrent_window_jepa",
        online_state=detach_recurrent_state(first.online_state),
    )
    second.loss.backward()
    assert first_context.grad is None
    assert second_context.grad is not None
    assert second_context.grad.abs().sum() > 0


def test_tbptt_ranges_support_a_short_tail_chunk() -> None:
    loss_mask = torch.tensor(
        [[False, False, True, True, True, True, True]] * 2,
        dtype=torch.bool,
    )
    detach_mask = torch.tensor(
        [[False, False, True, False, False, True, False]] * 2,
        dtype=torch.bool,
    )
    assert _recurrent_chunk_ranges(loss_mask, detach_mask) == ((2, 5), (5, 7))


def _mixed_contract_batch(
    *,
    first_stream_end: int,
    reset_stream: bool,
    stream_augmentation: str = "recording-0-geometry",
    stream_sequence: str = "stream-sequence",
    reset_random: bool = True,
) -> dict[str, object]:
    return {
        "sampling_mode": ["stream", "random"],
        "state_reset": torch.tensor(
            [reset_stream, reset_random], dtype=torch.bool
        ),
        "t_end_us": torch.tensor(
            [
                [first_stream_end, first_stream_end + 50_000],
                [600_000, 650_000],
            ],
            dtype=torch.int64,
        ),
        "sequence_id": [stream_sequence, "random-sequence"],
        "stream_id": ["rank-0:lane-0", ""],
        "augmentation_id": [stream_augmentation, "random-geometry"],
    }


def test_mixed_contract_enforces_cross_batch_tbptt_and_random_resets() -> None:
    first = _validate_mixed_recurrent_batch(
        _mixed_contract_batch(first_stream_end=50_000, reset_stream=True),
        batch_size=2,
        stream_batch_size=1,
        stride_us=50_000,
        previous_streams=None,
    )
    second = _validate_mixed_recurrent_batch(
        _mixed_contract_batch(first_stream_end=150_000, reset_stream=False),
        batch_size=2,
        stream_batch_size=1,
        stride_us=50_000,
        previous_streams=first,
    )
    assert second == (
        ("rank-0:lane-0", "stream-sequence", "recording-0-geometry", 200_000),
    )

    boundary = _validate_mixed_recurrent_batch(
        _mixed_contract_batch(
            first_stream_end=50_000,
            reset_stream=True,
            stream_augmentation="recording-1-geometry",
            stream_sequence="next-stream-sequence",
        ),
        batch_size=2,
        stream_batch_size=1,
        stride_us=50_000,
        previous_streams=second,
    )
    assert boundary == (
        (
            "rank-0:lane-0",
            "next-stream-sequence",
            "recording-1-geometry",
            100_000,
        ),
    )

    with pytest.raises(ValueError, match="augmentation"):
        _validate_mixed_recurrent_batch(
            _mixed_contract_batch(
                first_stream_end=250_000,
                reset_stream=False,
                stream_augmentation="changed-geometry",
            ),
            batch_size=2,
            stream_batch_size=1,
            stride_us=50_000,
            previous_streams=second,
        )
    with pytest.raises(ValueError, match="random rows must reset"):
        _validate_mixed_recurrent_batch(
            _mixed_contract_batch(
                first_stream_end=250_000,
                reset_stream=False,
                reset_random=False,
            ),
            batch_size=2,
            stream_batch_size=1,
            stride_us=50_000,
            previous_streams=second,
        )


def test_stream_reset_contract_preserves_lanes_but_allows_continuous_reset() -> None:
    first = {
        "sampling_mode": ["stream", "stream"],
        "state_reset": torch.tensor([True, True], dtype=torch.bool),
        "t_end_us": torch.tensor(
            [[50_000, 100_000], [250_000, 300_000]], dtype=torch.int64
        ),
        "sequence_id": ["sequence-a", "sequence-b"],
        "stream_id": ["rank-0:lane-0", "rank-0:lane-1"],
        "augmentation_id": ["geometry-a", "geometry-b"],
    }
    previous = _validate_mixed_recurrent_batch(
        first,
        batch_size=2,
        stream_batch_size=2,
        stride_us=50_000,
        previous_streams=None,
        stream_reset_every_batch=True,
    )
    second = {
        **first,
        "t_end_us": torch.tensor(
            [[150_000, 200_000], [350_000, 400_000]], dtype=torch.int64
        ),
    }
    current = _validate_mixed_recurrent_batch(
        second,
        batch_size=2,
        stream_batch_size=2,
        stride_us=50_000,
        previous_streams=previous,
        stream_reset_every_batch=True,
    )
    assert current[-1][-1] == 400_000

    second["state_reset"] = torch.tensor([False, True], dtype=torch.bool)
    with pytest.raises(ValueError, match="stream_reset rows must reset"):
        _validate_mixed_recurrent_batch(
            second,
            batch_size=2,
            stream_batch_size=2,
            stride_us=50_000,
            previous_streams=previous,
            stream_reset_every_batch=True,
        )


def test_dense_r0_backward_weights_an_uneven_tail_by_timestep_count() -> None:
    torch.manual_seed(11)
    config = _r0_config()
    model = _model()
    reference = copy.deepcopy(model)
    batch_size, total_steps = 2, 4
    context_mask, target_mask = _masks(batch_size, total_steps)
    batch = {
        "x": torch.randn(batch_size, total_steps, 2, 16, 16),
        "dt_ms": torch.full((batch_size, total_steps), 50.0),
        "context_mask": context_mask,
        "target_mask": target_mask,
        "loss_mask": torch.tensor(
            [[False, True, True, True]] * batch_size, dtype=torch.bool
        ),
        "detach_mask": torch.tensor(
            [[False, True, False, True]] * batch_size, dtype=torch.bool
        ),
    }

    reference_state = reference.recurrent_burn_in(
        batch["x"][:, :1], batch["dt_ms"][:, :1], context_mask[:, :1]
    )
    first_chunk = reference(
        batch["x"][:, 1:3],
        batch["x"][:, 1:3],
        batch["dt_ms"][:, 1:3],
        batch["dt_ms"][:, 1:3],
        context_mask[:, 1:3],
        target_mask[:, 1:3],
        objective="recurrent_dense_window_jepa",
        online_state=reference_state,
    )
    tail_chunk = reference(
        batch["x"][:, 3:4],
        batch["x"][:, 3:4],
        batch["dt_ms"][:, 3:4],
        batch["dt_ms"][:, 3:4],
        context_mask[:, 3:4],
        target_mask[:, 3:4],
        objective="recurrent_dense_window_jepa",
        online_state=detach_recurrent_state(first_chunk.online_state),
    )
    expected_loss = first_chunk.loss * (2.0 / 3.0) + (
        tail_chunk.loss * (1.0 / 3.0)
    )
    expected_metric_loss = expected_loss.detach()
    expected_loss.backward()

    class RecordingScaler:
        def __init__(self) -> None:
            self.scale_calls = 0

        def scale(self, loss: torch.Tensor) -> torch.Tensor:
            self.scale_calls += 1
            return loss

    scaler = RecordingScaler()
    metrics, state_rms, final_state = _recurrent_backward(
        model=model,
        core_model=model,
        batch=batch,
        config=config,
        device=torch.device("cpu"),
        world_size=1,
        grad_scaler=scaler,
    )
    assert scaler.scale_calls == 2
    assert torch.allclose(metrics[0], expected_metric_loss)
    assert state_rms > 0
    assert final_state is not None
    for (name, parameter), (reference_name, reference_parameter) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert name == reference_name
        if not parameter.requires_grad:
            continue
        assert parameter.grad is not None, name
        assert reference_parameter.grad is not None, name
        torch.testing.assert_close(
            parameter.grad,
            reference_parameter.grad,
            rtol=1e-5,
            atol=1e-6,
            msg=lambda message, parameter_name=name: (
                f"{parameter_name} gradient differs from weighted reference: {message}"
            ),
        )


def test_recurrent_checkpoint_round_trip_keeps_weights_but_not_runtime_state(
    tmp_path,
) -> None:
    config = _r0_config()
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    path = tmp_path / "recurrent.pt"
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
    assert loaded_config == config
    assert isinstance(
        loaded.online_encoder, RecurrentVJEPA21EventVisionTransformer
    )
    for name, expected in model.online_encoder.state_dict().items():
        assert torch.equal(loaded.online_encoder.state_dict()[name], expected)
    assert all("state" not in name for name in loaded.online_encoder.state_dict())


def test_stateless_feature_extraction_rejects_a_recurrent_checkpoint() -> None:
    model = _model().eval()
    with pytest.raises(ValueError, match="frames independently"):
        extract_patch_features(
            model,
            torch.randn(2, 2, 16, 16),
            torch.full((2,), 50.0),
            mode="encoder_only",
        )
