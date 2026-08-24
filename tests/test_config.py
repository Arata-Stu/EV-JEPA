from __future__ import annotations

import pytest

from event_window_jepa.config import (
    ExperimentConfig,
    MaskConfig,
    ModelConfig,
    RecurrentConfig,
    WindowsConfig,
)


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


def test_event_aware_mask_configuration_is_validated() -> None:
    config = MaskConfig.from_mapping(
        {
            "activity_aware_probability": 0.7,
            "activity_candidates": 32,
            "minimum_active_target_ratio": 0.25,
            "activity_selection_strategy": "topk_enrichment",
            "activity_topk_fraction": 0.25,
        }
    )
    assert config.activity_aware_probability == 0.7
    assert config.activity_candidates == 32
    assert config.minimum_active_target_ratio == 0.25
    assert config.activity_selection_strategy == "topk_enrichment"
    assert config.activity_topk_fraction == 0.25

    with pytest.raises(ValueError, match="activity_aware_probability"):
        MaskConfig(activity_aware_probability=1.1)

    with pytest.raises(ValueError, match="activity_selection_strategy"):
        MaskConfig(activity_selection_strategy="maximum_mass")

    with pytest.raises(ValueError, match="activity_topk_fraction"):
        MaskConfig(activity_topk_fraction=0.0)


def test_recurrent_config_validates_bptt_geometry() -> None:
    config = RecurrentConfig(
        enabled=True,
        cell="conv_lstm",
        window_ms=50,
        stride_ms=50,
        sequence_length=8,
        burn_in_steps=2,
        tbptt_steps=4,
    )
    assert config.sequence_length == 8
    assert config.burn_in_steps == 2
    assert config.tbptt_steps == 4

    with pytest.raises(ValueError, match="positive odd"):
        RecurrentConfig(kernel_size=4)
    with pytest.raises(ValueError, match="sequence_length"):
        RecurrentConfig(sequence_length=2, tbptt_steps=3)


def test_r0_requires_equal_non_overlapping_fifty_ms_windows() -> None:
    base = {
        "data": {
            "manifest": "events.jsonl",
            "batch_size": 2,
            "crop_size": [16, 16],
        },
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
            "predictor_heads": 4,
            "scale_dim": 16,
            "deep_supervision_layers": [0, 1],
        },
        "recurrent": {
            "enabled": True,
            "sampling": "mixed",
            "stream_ratio": 0.5,
            "window_ms": 50,
            "stride_ms": 50,
            "sequence_length": 4,
            "burn_in_steps": 1,
            "tbptt_steps": 4,
        },
        "optimization": {
            "objective": "recurrent_dense_window_jepa",
            "canonical_query_weight": 0,
        },
    }
    assert ExperimentConfig.from_mapping(base).recurrent.enabled

    overlapping = {**base, "recurrent": {**base["recurrent"], "stride_ms": 25}}
    with pytest.raises(ValueError, match="stride_ms == window_ms"):
        ExperimentConfig.from_mapping(overlapping)

    missing_final_supervision = {
        **base,
        "model": {**base["model"], "deep_supervision_layers": [0]},
    }
    with pytest.raises(ValueError, match="final encoder block"):
        ExperimentConfig.from_mapping(missing_final_supervision)

    plain_missing_final_supervision = {
        **missing_final_supervision,
        "optimization": {
            **base["optimization"],
            "objective": "recurrent_window_jepa",
        },
    }
    with pytest.raises(ValueError, match="final encoder block"):
        ExperimentConfig.from_mapping(plain_missing_final_supervision)

    truncated_mixed = {
        **base,
        "recurrent": {**base["recurrent"], "tbptt_steps": 2},
    }
    with pytest.raises(ValueError, match="tbptt_steps == sequence_length"):
        ExperimentConfig.from_mapping(truncated_mixed)
