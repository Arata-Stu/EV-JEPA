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


def test_temporal_loader_and_model_switches_resolve_legacy_and_explicit_forms() -> None:
    legacy = RecurrentConfig.from_mapping(
        {"enabled": True, "cell": "conv_gru"}
    )
    assert legacy.sequence_loader is True
    assert legacy.temporal_model == "conv_gru"
    assert legacy.enabled is True
    assert legacy.cell == "conv_gru"

    feedforward = RecurrentConfig.from_mapping(
        {
            "sequence_loader": True,
            "temporal_model": "feedforward",
            "return_patch_event_activity": True,
        }
    )
    assert feedforward.sequence_loader is True
    assert feedforward.temporal_model == "feedforward"
    assert feedforward.enabled is False
    assert feedforward.return_patch_event_activity is True

    conv_gru = RecurrentConfig.from_mapping({"temporal_model": "conv_gru"})
    assert conv_gru.sequence_loader is True
    assert conv_gru.enabled is True
    assert conv_gru.cell == "conv_gru"


def test_temporal_loader_and_legacy_alias_contradictions_are_rejected() -> None:
    with pytest.raises(ValueError, match="enabled contradicts"):
        RecurrentConfig(enabled=False, temporal_model="conv_lstm")
    with pytest.raises(ValueError, match="enabled contradicts"):
        RecurrentConfig(enabled=True, temporal_model="feedforward")
    with pytest.raises(ValueError, match="cell contradicts"):
        RecurrentConfig(cell="conv_lstm", temporal_model="conv_gru")
    with pytest.raises(ValueError, match="sequence_loader=true"):
        RecurrentConfig(sequence_loader=False, temporal_model="conv_gru")
    with pytest.raises(ValueError, match="return_patch_event_activity"):
        RecurrentConfig(return_patch_event_activity=True)
    with pytest.raises(TypeError, match="YAML boolean"):
        RecurrentConfig.from_mapping({"sequence_loader": "true"})


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

    feedforward_sequence = {
        **base,
        "recurrent": {
            key: value
            for key, value in base["recurrent"].items()
            if key != "enabled"
        }
        | {
            "sequence_loader": True,
            "temporal_model": "feedforward",
            "return_patch_event_activity": True,
        },
        "optimization": {
            **base["optimization"],
            "objective": "sequence_dense_window_jepa",
        },
    }
    resolved_feedforward = ExperimentConfig.from_mapping(feedforward_sequence)
    assert resolved_feedforward.recurrent.sequence_loader is True
    assert resolved_feedforward.recurrent.enabled is False
    assert resolved_feedforward.recurrent.temporal_model == "feedforward"
    assert resolved_feedforward.recurrent.return_patch_event_activity is True

    sequence_default_supervision = {
        **feedforward_sequence,
        "model": {
            key: value
            for key, value in feedforward_sequence["model"].items()
            if key != "deep_supervision_layers"
        },
    }
    assert (
        ExperimentConfig.from_mapping(
            sequence_default_supervision
        ).model.deep_supervision_layers
        == ()
    )

    paired_default_supervision = {
        key: value for key, value in base.items() if key != "recurrent"
    }
    paired_default_supervision["model"] = {
        key: value
        for key, value in base["model"].items()
        if key != "deep_supervision_layers"
    }
    paired_default_supervision["optimization"] = {
        **base["optimization"],
        "objective": "dense_window_jepa",
    }
    assert (
        ExperimentConfig.from_mapping(
            paired_default_supervision
        ).model.deep_supervision_layers
        == ()
    )

    feedforward_chunk_metadata = {
        **feedforward_sequence,
        "recurrent": {
            **feedforward_sequence["recurrent"],
            "tbptt_steps": 2,
        },
    }
    assert (
        ExperimentConfig.from_mapping(
            feedforward_chunk_metadata
        ).recurrent.tbptt_steps
        == 2
    )

    feedforward_missing_final = {
        **feedforward_sequence,
        "model": {
            **feedforward_sequence["model"],
            "deep_supervision_layers": [0],
        },
    }
    with pytest.raises(ValueError, match="final encoder block"):
        ExperimentConfig.from_mapping(feedforward_missing_final)

    feedforward_plain_missing_final = {
        **feedforward_missing_final,
        "optimization": {
            **feedforward_missing_final["optimization"],
            "objective": "sequence_window_jepa",
        },
    }
    with pytest.raises(ValueError, match="final encoder block"):
        ExperimentConfig.from_mapping(feedforward_plain_missing_final)

    paired_missing_final = {
        **paired_default_supervision,
        "model": {**base["model"], "deep_supervision_layers": [0]},
        "optimization": {
            **base["optimization"],
            "objective": "window_jepa",
        },
    }
    with pytest.raises(ValueError, match="final encoder block"):
        ExperimentConfig.from_mapping(paired_missing_final)

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

    for sampling in ("random", "stream_reset", "stream", "mixed"):
        resolved = ExperimentConfig.from_mapping(
            {
                **base,
                "recurrent": {**base["recurrent"], "sampling": sampling},
            }
        )
        assert resolved.recurrent.sampling == sampling

    truncated_explicit = {
        **base,
        "recurrent": {
            **base["recurrent"],
            "sampling": "stream_reset",
            "tbptt_steps": 2,
        },
    }
    with pytest.raises(ValueError, match="tbptt_steps == sequence_length"):
        ExperimentConfig.from_mapping(truncated_explicit)

    # stream_ratio is irrelevant outside mixed sampling.
    stream_without_ratio = {
        **base,
        "recurrent": {
            **base["recurrent"],
            "sampling": "stream",
            "stream_ratio": 1.0,
        },
    }
    assert ExperimentConfig.from_mapping(stream_without_ratio).recurrent.sampling == "stream"

    non_rvt_mixed_ratio = {
        **base,
        "recurrent": {
            **base["recurrent"],
            "sampling": "mixed",
            "stream_ratio": 0.25,
        },
    }
    with pytest.raises(ValueError, match="mixed 1:1"):
        ExperimentConfig.from_mapping(non_rvt_mixed_ratio)

    odd_mixed_batch = {
        **base,
        "data": {**base["data"], "batch_size": 3},
    }
    with pytest.raises(ValueError, match="even per-rank batch"):
        ExperimentConfig.from_mapping(odd_mixed_batch)
