from __future__ import annotations

import pytest

from event_window_jepa.config import ModelConfig, WindowsConfig


def test_unseen_windows_exclude_context_and_target_training_durations() -> None:
    with pytest.raises(ValueError, match="overlap"):
        WindowsConfig(
            train_ms=(10.0, 40.0),
            target_ms=(20.0, 40.0),
            eval_ms=(5.0, 10.0, 20.0, 40.0),
            unseen_eval_ms=(5.0, 20.0),
            canonical_ms=40.0,
        )


def test_vjepa21_model_accepts_zero_based_deep_supervision_layers() -> None:
    config = ModelConfig.from_mapping(
        {
            "architecture": "vjepa2_1",
            "embed_dim": 384,
            "encoder_heads": 6,
            "encoder_depth": 12,
            "deep_supervision_layers": [2, 5, 8, 11],
        }
    )
    assert config.architecture == "vjepa2_1"
    assert config.deep_supervision_layers == (2, 5, 8, 11)
