from __future__ import annotations

import numpy as np
import pytest
import torch

from event_window_jepa.evaluation.future_feature_visualization import (
    FutureFeatureStepRecord,
    cosine_error_rgb,
    fit_shared_target_pca,
    fixed_support_latent_diagnostics,
    make_history_permutation,
    make_history_replacement_clip_permutation,
    make_unrelated_clip_permutation,
    make_unrelated_record_permutation,
    pca_patch_rgb,
    project_with_shared_pca,
    step_conditioned_latent_diagnostics,
    summarize_prediction_condition,
    token_cosine_error,
)


def test_shared_target_pca_is_deterministic_and_shared_across_panels() -> None:
    generator = torch.Generator().manual_seed(7)
    target = torch.randn(5, 6, 8, generator=generator)
    first = fit_shared_target_pca(target)
    second = fit_shared_target_pca(target.clone())

    assert first.basis_id == second.basis_id
    assert torch.equal(first.mean, second.mean)
    assert torch.equal(first.components, second.components)
    assert torch.equal(first.scale, second.scale)
    assert torch.allclose(
        first.components.transpose(0, 1) @ first.components,
        torch.eye(3, dtype=torch.float64),
        atol=1e-10,
        rtol=0,
    )
    for index in range(3):
        column = first.components[:, index]
        assert column[column.abs().argmax()] >= 0

    projected_target, _ = project_with_shared_pca(target, first)
    projected_copy, _ = project_with_shared_pca(target.clone(), first)
    assert torch.equal(projected_target, projected_copy)


def test_constant_target_pca_stays_gray_instead_of_inventing_color() -> None:
    target = torch.full((3, 4, 6), 2.0)
    basis = fit_shared_target_pca(target)
    image, clip_fraction = pca_patch_rgb(
        target[0],
        basis,
        grid_size=(2, 2),
        image_size=(4, 4),
    )

    assert basis.valid_rank == 0
    assert np.array_equal(image, np.full((4, 4, 3), 128, dtype=np.uint8))
    assert clip_fraction == 0.0


def test_visual_diagnostics_reject_non_finite_latents() -> None:
    target = torch.zeros(2, 4, 6)
    target[0, 0, 0] = float("nan")

    with pytest.raises(FloatingPointError, match="NaN"):
        fit_shared_target_pca(target)
    with pytest.raises(FloatingPointError, match="NaN"):
        token_cosine_error(target[0], torch.zeros_like(target[0]))


def test_cosine_error_and_fixed_heatmap_have_known_endpoints() -> None:
    prediction = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
    )
    target = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
    )
    error = token_cosine_error(prediction, target)

    assert torch.allclose(error, torch.tensor([0.0, 2.0, 2.0]))
    heatmap = cosine_error_rgb(
        torch.tensor([0.0, 0.5, 1.0, 2.0]),
        grid_size=(2, 2),
        image_size=(2, 2),
    )
    assert np.array_equal(heatmap[0, 0], np.array([13, 25, 60]))
    assert np.array_equal(heatmap[0, 1], np.array([20, 184, 166]))
    assert np.array_equal(heatmap[1, 0], np.array([250, 204, 21]))
    assert np.array_equal(heatmap[1, 1], np.array([220, 38, 38]))


def test_position_only_features_are_zero_after_fixed_position_centering() -> None:
    position_features = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    features = position_features.unsqueeze(0).repeat(5, 1, 1)
    diagnostics = fixed_support_latent_diagnostics(features)

    assert diagnostics["global_token_rank"] > 0
    assert diagnostics["fixed_position_std"] == 0.0
    assert diagnostics["fixed_position_centered_rank"] == 0.0
    assert diagnostics["frame_pooled_rank"] == 0.0


def test_fixed_position_normalized_rank_uses_positionwise_degrees_of_freedom() -> None:
    features = torch.tensor(
        [
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            [[-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0]],
        ]
    )
    diagnostics = fixed_support_latent_diagnostics(features)

    assert torch.isclose(
        torch.tensor(diagnostics["fixed_position_centered_rank"]),
        torch.tensor(2.0),
    )
    assert torch.isclose(
        torch.tensor(diagnostics["fixed_position_centered_normalized_rank"]),
        torch.tensor(1.0),
    )


def test_prediction_summary_reports_event_balanced_cosine_error() -> None:
    generator = torch.Generator().manual_seed(19)
    target = torch.randn(2, 4, 6, generator=generator)
    activity = torch.tensor([[3, 0, 2, 0], [0, 4, 0, 1]])
    correct = summarize_prediction_condition(
        target,
        target,
        activity,
        active_min_events=1,
        loss_kind="smooth_l1",
    )
    opposite = summarize_prediction_condition(
        -target,
        target,
        activity,
        active_min_events=1,
        loss_kind="smooth_l1",
    )

    assert abs(correct["balanced_cosine_error"]) < 1e-6
    assert abs(opposite["balanced_cosine_error"] - 2.0) < 1e-5


def test_step_conditioning_does_not_mistake_a_time_code_for_sample_variance() -> None:
    early = torch.ones(3, 2, 4)
    late = -torch.ones(3, 2, 4)
    features = torch.cat((early, late), dim=0)
    steps = torch.tensor([2, 2, 2, 3, 3, 3], dtype=torch.int64)
    global_diagnostics = fixed_support_latent_diagnostics(features)
    conditioned = step_conditioned_latent_diagnostics(features, steps)

    assert global_diagnostics["fixed_position_std"] > 0
    assert conditioned["step_conditioned_fixed_position_std"] == 0.0
    assert conditioned["step_conditioned_fixed_position_centered_rank"] == 0.0


def test_history_and_unrelated_permutations_are_deterministic_derangements() -> None:
    first = make_history_permutation(
        6,
        seed=11,
        epoch=2,
        sample_index=7,
        online_step=6,
    )
    second = make_history_permutation(
        6,
        seed=11,
        epoch=2,
        sample_index=7,
        online_step=6,
    )
    unrelated = make_unrelated_clip_permutation(8, seed=3)

    assert first == second
    assert sorted(first) == list(range(6))
    assert first != tuple(range(6))
    assert sorted(unrelated) == list(range(8))
    assert all(source != target for source, target in enumerate(unrelated))


def test_history_replacement_avoids_duplicate_clip_anchors() -> None:
    identities = (
        ("sequence-a", (10, 20, 30)),
        ("sequence-a", (10, 20, 30)),
        ("sequence-b", (10, 20, 30)),
        ("sequence-c", (40, 50, 60)),
    )

    permutation = make_history_replacement_clip_permutation(identities, seed=0)

    assert sorted(permutation) == list(range(len(identities)))
    assert all(
        identities[source] != identities[target]
        for source, target in enumerate(permutation)
    )


def test_unrelated_records_keep_online_step_and_change_clip() -> None:
    records: list[FutureFeatureStepRecord] = []
    tokens = torch.zeros(4, 6)
    activity = torch.ones(4, dtype=torch.int64)
    for clip_position in range(3):
        for online_step in (2, 3):
            records.append(
                FutureFeatureStepRecord(
                    record_index=len(records),
                    clip_position=clip_position,
                    sample_index=10 + clip_position,
                    online_step=online_step,
                    sequence_id=f"sequence-{clip_position}",
                    context_t_end_us=10_000 * online_step,
                    target_t_end_us=10_000 * (online_step + 1),
                    frame_tokens=tokens,
                    recurrent_tokens=tokens,
                    prediction=tokens,
                    target_tokens=tokens,
                    reset_recurrent_tokens=tokens,
                    reset_prediction=tokens,
                    shuffled_recurrent_tokens=tokens,
                    shuffled_prediction=tokens,
                    target_activity=activity,
                    history_permutation=(1, 0),
                )
            )

    permutation = make_unrelated_record_permutation(records, seed=0)

    assert sorted(permutation) == list(range(len(records)))
    for source_index, target_index in enumerate(permutation):
        assert records[source_index].clip_position != records[target_index].clip_position
        assert records[source_index].online_step == records[target_index].online_step
