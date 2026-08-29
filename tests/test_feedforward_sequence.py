from __future__ import annotations

import pytest
import torch

from event_window_jepa.config import ExperimentConfig
from event_window_jepa.models.scale_embedding import LogFourierScaleEmbedding
from event_window_jepa.models.vjepa21_event_vit import VJEPA21EventVisionTransformer
from event_window_jepa.models.window_jepa import WindowJEPA
from event_window_jepa.models.window_predictor import WindowPredictor
from event_window_jepa.train.pretrain import (
    OUTPUT_METRIC_NAMES,
    _feedforward_sequence_backward,
)


def _model() -> WindowJEPA:
    encoder = VJEPA21EventVisionTransformer(
        image_size=(16, 16),
        patch_size=8,
        input_channels=2,
        embed_dim=32,
        depth=2,
        num_heads=4,
        scale_dim=16,
        supervision_layers=(0, 1),
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
    )


def _batch(batch_size: int = 2, steps: int = 4) -> dict[str, torch.Tensor]:
    context_mask = torch.tensor(
        [True, True, False, False], dtype=torch.bool
    ).reshape(1, 1, 4).expand(batch_size, steps, -1).clone()
    return {
        "x": torch.randn(batch_size, steps, 2, 16, 16),
        "dt_ms": torch.full((batch_size, steps), 50.0),
        "context_mask": context_mask,
        "target_mask": ~context_mask,
        "loss_mask": torch.tensor(
            [False, False, True, True], dtype=torch.bool
        ).reshape(1, steps).expand(batch_size, -1).clone(),
        # Feedforward sequence training deliberately ignores this TBPTT metadata.
        "detach_mask": torch.tensor(
            [False, False, True, False], dtype=torch.bool
        ).reshape(1, steps).expand(batch_size, -1).clone(),
    }


@pytest.mark.parametrize(
    ("objective", "flat_objective"),
    [
        ("sequence_window_jepa", "window_jepa"),
        ("sequence_dense_window_jepa", "dense_window_jepa"),
    ],
)
def test_feedforward_sequence_matches_supervised_flat_batch(
    objective: str, flat_objective: str
) -> None:
    torch.manual_seed(13)
    model = _model().eval()
    batch = _batch()

    sequence = model(
        batch["x"],
        batch["x"],
        batch["dt_ms"],
        batch["dt_ms"],
        batch["context_mask"],
        batch["target_mask"],
        objective=objective,
        sequence_loss_mask=batch["loss_mask"],
    )
    supervised = batch["loss_mask"][0]

    def flatten(value: torch.Tensor) -> torch.Tensor:
        return value[:, supervised].flatten(0, 1)

    flat = model(
        flatten(batch["x"]),
        flatten(batch["x"]),
        flatten(batch["dt_ms"]),
        flatten(batch["dt_ms"]),
        flatten(batch["context_mask"]),
        flatten(batch["target_mask"]),
        objective=flat_objective,
    )
    assert torch.equal(sequence.prediction, flat.prediction)
    assert torch.equal(sequence.target, flat.target)
    assert sequence.prediction_sequence is not None
    assert sequence.target_sequence is not None
    assert sequence.prediction_sequence.shape == (2, 2, 4, 32)
    assert sequence.target_sequence.shape == (2, 2, 4, 32)
    assert torch.equal(
        sequence.prediction_sequence, sequence.prediction.reshape(2, 2, 4, 32)
    )


def test_feedforward_sequence_omits_burn_in_and_uses_one_encoder_batch() -> None:
    torch.manual_seed(17)
    model = _model().eval()
    batch = _batch()
    changed = batch["x"].clone()
    changed[:, :2] += 100.0
    encoder_batch_sizes: list[int] = []

    def record_batch_size(
        _module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]
    ) -> None:
        encoder_batch_sizes.append(int(inputs[0].shape[0]))

    handle = model.online_encoder.register_forward_pre_hook(record_batch_size)
    try:
        first = model(
            batch["x"],
            batch["x"],
            batch["dt_ms"],
            batch["dt_ms"],
            batch["context_mask"],
            batch["target_mask"],
            objective="sequence_window_jepa",
            sequence_loss_mask=batch["loss_mask"],
        )
        second = model(
            changed,
            changed,
            batch["dt_ms"],
            batch["dt_ms"],
            batch["context_mask"],
            batch["target_mask"],
            objective="sequence_window_jepa",
            sequence_loss_mask=batch["loss_mask"],
        )
    finally:
        handle.remove()

    assert encoder_batch_sizes == [4, 4]
    assert torch.equal(first.prediction, second.prediction)
    assert torch.equal(first.target, second.target)


def test_feedforward_sequence_rejects_row_dependent_loss_mask() -> None:
    model = _model()
    batch = _batch()
    batch["loss_mask"][1] = torch.tensor([False, True, True, True])
    with pytest.raises(ValueError, match="share sequence_loss_mask"):
        model(
            batch["x"],
            batch["x"],
            batch["dt_ms"],
            batch["dt_ms"],
            batch["context_mask"],
            batch["target_mask"],
            objective="sequence_window_jepa",
            sequence_loss_mask=batch["loss_mask"],
        )


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
                "deep_supervision_layers": [0, 1],
            },
            "recurrent": {
                "sequence_loader": True,
                "temporal_model": "feedforward",
                "window_ms": 50,
                "stride_ms": 50,
                "sequence_length": 2,
                "burn_in_steps": 2,
                "tbptt_steps": 2,
            },
            "optimization": {
                "objective": "sequence_window_jepa",
                "epochs": 2,
                "warmup_epochs": 0,
                "precision": "fp32",
            },
        }
    )


def test_feedforward_sequence_training_ignores_detach_state_metadata() -> None:
    torch.manual_seed(19)
    model = _model()
    metrics = _feedforward_sequence_backward(
        model=model,
        batch=_batch(),
        config=_config(),
        device=torch.device("cpu"),
        world_size=1,
    )
    assert metrics.shape == (len(OUTPUT_METRIC_NAMES),)
    assert any(parameter.grad is not None for parameter in model.online_encoder.parameters())
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())
