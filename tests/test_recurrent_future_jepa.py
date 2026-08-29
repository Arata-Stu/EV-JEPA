from __future__ import annotations

import torch

from event_window_jepa.config import ExperimentConfig
from event_window_jepa.models.recurrent_vjepa21_event_vit import (
    RecurrentState,
    RecurrentVJEPA21EventVisionTransformer,
)
from event_window_jepa.models.scale_embedding import LogFourierScaleEmbedding
from event_window_jepa.models.window_jepa import WindowJEPA
from event_window_jepa.models.window_predictor import WindowPredictor
from event_window_jepa.train.checkpoint import (
    load_pretrained_model,
    save_checkpoint_atomic,
)


def _future_model() -> WindowJEPA:
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
        recurrent_placement="post_encoder",
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
        frame_sigreg_weight=0.02,
        temporal_sigreg_weight=0.02,
        sigreg_projector_hidden_dim=32,
        sigreg_projector_output_dim=16,
        sigreg_num_slices=8,
        sigreg_num_points=5,
    )


def _future_config() -> ExperimentConfig:
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
                "temporal_sigreg_weight": 0.02,
                "projector_hidden_dim": 32,
                "projector_output_dim": 16,
                "sigreg_num_slices": 8,
                "sigreg_num_points": 5,
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


def _state_equal(first: RecurrentState | None, second: RecurrentState | None) -> bool:
    if isinstance(first, tuple) and isinstance(second, tuple):
        return torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])
    return isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor) and (
        torch.equal(first, second)
    )


def _inputs() -> tuple[torch.Tensor, ...]:
    batch_size, steps = 2, 2
    context = torch.randn(batch_size, steps, 2, 16, 16)
    future = torch.randn_like(context)
    duration = torch.full((batch_size, steps), 50.0)
    first_mask = torch.tensor([True, False, True, False]).reshape(1, 1, 4)
    context_mask = first_mask.expand(batch_size, steps, -1).clone()
    target_mask = ~context_mask
    context_activity = torch.tensor(
        [[[4, 0, 2, 0], [0, 3, 0, 1]], [[0, 2, 1, 0], [5, 0, 0, 2]]]
    )
    future_activity = torch.tensor(
        [[[0, 3, 0, 2], [4, 0, 1, 0]], [[2, 0, 0, 1], [0, 5, 2, 0]]]
    )
    return (
        context,
        future,
        duration,
        context_mask,
        target_mask,
        context_activity,
        future_activity,
    )


def _forward(model: WindowJEPA, values: tuple[torch.Tensor, ...]):
    (
        context,
        future,
        duration,
        context_mask,
        target_mask,
        context_activity,
        future_activity,
    ) = values
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
    )


def test_future_objective_ignores_spatial_masks_and_teacher_recurrent_cell() -> None:
    torch.manual_seed(23)
    model = _future_model().eval()
    values = _inputs()
    first = _forward(model, values)

    changed_masks = list(values)
    changed_masks[3] = ~values[3]
    changed_masks[4] = ~values[4]
    masked = _forward(model, tuple(changed_masks))
    assert torch.equal(first.prediction, masked.prediction)
    assert torch.equal(first.target, masked.target)
    assert _state_equal(first.online_state, masked.online_state)

    with torch.no_grad():
        target_cell = model.target_encoder.recurrent_cell
        for parameter in target_cell.parameters():
            parameter.normal_(mean=100.0, std=10.0)
    changed_teacher_cell = _forward(model, values)
    assert torch.equal(first.prediction, changed_teacher_cell.prediction)
    assert torch.equal(first.target, changed_teacher_cell.target)
    assert _state_equal(first.online_state, changed_teacher_cell.online_state)


def test_future_input_changes_teacher_only_and_sigreg_backpropagates_online() -> None:
    torch.manual_seed(29)
    model = _future_model()
    values = _inputs()
    first = _forward(model, values)
    changed_values = list(values)
    changed_values[1] = values[1] + 7.0
    changed = _forward(model, tuple(changed_values))

    assert torch.equal(first.prediction, changed.prediction)
    assert _state_equal(first.online_state, changed.online_state)
    assert not torch.equal(first.target, changed.target)
    assert first.future_prediction_loss is not None
    assert first.frame_sigreg_loss is not None
    assert first.support_sigreg_loss is not None
    assert first.temporal_sigreg_loss is not None
    assert first.support_sigreg_samples is not None
    assert first.support_sigreg_samples >= 2
    assert torch.isfinite(first.loss)

    first.loss.backward()
    assert model.online_encoder.patch_embed.weight.grad is not None
    assert model.predictor.output_projection.weight.grad is not None
    assert model.future_regularizers["frame"].projector.network[0].weight.grad is not None
    assert model.future_regularizers["support"].projector.network[0].weight.grad is not None
    assert model.future_regularizers["temporal"].projector.network[0].weight.grad is not None
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())


def test_event_support_uses_raw_presence_and_only_meaningful_count_splits() -> None:
    model = _future_model()
    model.future_active_min_events = 2
    tokens = torch.randn(3, 4, 32)
    activity = torch.tensor(
        [
            [1, 1, 1, 1],
            [0, 1, 0, 1],
            [0, 0, 0, 0],
        ]
    )

    _, frame_valid = model._event_support_pool(tokens, activity)
    _, support_valid = model._event_support_contrast(tokens, activity)

    assert frame_valid.tolist() == [True, True, False]
    # Equal non-zero counts have no event-derived spatial partition. The second
    # row has genuine count variation, so its below-threshold class is split.
    assert support_valid.tolist() == [False, True, False]


def test_future_collapse_std_is_fixed_position_batch_std() -> None:
    prediction = torch.tensor(
        [
            [[0.0, 2.0], [10.0, 12.0]],
            [[2.0, 4.0], [12.0, 14.0]],
        ]
    )
    target = prediction * 2.0

    prediction_std, target_std = (
        WindowJEPA._global_fixed_position_standard_deviations(
            prediction,
            target,
        )
    )

    torch.testing.assert_close(prediction_std, torch.tensor(1.0))
    torch.testing.assert_close(target_std, torch.tensor(2.0))


def test_future_sigreg_projectors_survive_checkpoint_round_trip(tmp_path) -> None:
    model = _future_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    path = tmp_path / "future.pt"
    save_checkpoint_atomic(
        path,
        model,
        optimizer,
        _future_config(),
        epoch=1,
        global_step=1,
        world_size=1,
        steps_per_epoch=1,
    )

    loaded, _ = load_pretrained_model(path)
    for name, expected in model.future_regularizers.state_dict().items():
        assert torch.equal(loaded.future_regularizers.state_dict()[name], expected)
