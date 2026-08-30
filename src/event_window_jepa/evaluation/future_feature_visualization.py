from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import quote

import numpy as np
import torch
import torch.nn.functional as functional

from event_window_jepa.config import ExperimentConfig
from event_window_jepa.data.anchor_sampler import deterministic_seed
from event_window_jepa.evaluation.future_feature_data import (
    FutureFeatureMaterialization,
    materialize_future_feature_samples,
    validate_future_feature_config,
)
from event_window_jepa.inspection import (
    _display_scale,
    _event_counts,
    _event_rgb,
    _png_data_uri,
)
from event_window_jepa.losses.latent_prediction import (
    balanced_event_support_latent_prediction_loss,
)
from event_window_jepa.models.window_jepa import WindowJEPA
from event_window_jepa.train.checkpoint import config_hash, load_pretrained_model


@dataclass(frozen=True)
class SharedPCABasis:
    """One target-calibrated three-channel basis shared by every panel."""

    mean: torch.Tensor
    components: torch.Tensor
    scale: torch.Tensor
    explained_variance_ratio: torch.Tensor
    valid_rank: int
    calibration_tokens: int
    basis_id: str

    def to_record(self) -> dict[str, Any]:
        return {
            "basis_id": self.basis_id,
            "feature_transform": "token_layer_norm",
            "valid_rank": self.valid_rank,
            "calibration_tokens": self.calibration_tokens,
            "explained_variance_ratio": self.explained_variance_ratio.tolist(),
            "rgb_scale": self.scale.tolist(),
            "mean": self.mean.tolist(),
            "components": self.components.tolist(),
        }


@dataclass(frozen=True)
class FutureFeatureStepRecord:
    """CPU latents for one supervised future-prediction anchor."""

    record_index: int
    clip_position: int
    sample_index: int
    online_step: int
    sequence_id: str
    context_t_end_us: int
    target_t_end_us: int
    frame_tokens: torch.Tensor
    recurrent_tokens: torch.Tensor
    prediction: torch.Tensor
    target_tokens: torch.Tensor
    reset_recurrent_tokens: torch.Tensor
    reset_prediction: torch.Tensor
    shuffled_recurrent_tokens: torch.Tensor | None
    shuffled_prediction: torch.Tensor | None
    target_activity: torch.Tensor
    history_permutation: tuple[int, ...]
    reversed_recurrent_tokens: torch.Tensor | None = None
    reversed_prediction: torch.Tensor | None = None
    replaced_recurrent_tokens: torch.Tensor | None = None
    replaced_prediction: torch.Tensor | None = None
    replacement_clip_position: int | None = None

    @property
    def history_shuffle_available(self) -> bool:
        return self.shuffled_prediction is not None

    @property
    def history_reverse_available(self) -> bool:
        return self.reversed_prediction is not None

    @property
    def history_replacement_available(self) -> bool:
        return self.replaced_prediction is not None


def _token_layer_norm(features: torch.Tensor) -> torch.Tensor:
    if features.ndim < 2 or features.shape[-1] <= 0:
        raise ValueError("features must have a non-empty final dimension")
    if not bool(torch.isfinite(features).all()):
        raise FloatingPointError("features contain NaN or infinity")
    normalized = functional.layer_norm(features.float(), (features.shape[-1],))
    if not bool(torch.isfinite(normalized).all()):
        raise FloatingPointError("token LayerNorm produced NaN or infinity")
    return normalized


@torch.no_grad()
def fit_shared_target_pca(target_features: torch.Tensor) -> SharedPCABasis:
    """Fit one deterministic CPU RGB PCA basis from EMA target tokens only."""

    if target_features.ndim != 3:
        raise ValueError("target_features must have shape [S,N,D]")
    sample_count, patch_count, feature_dim = target_features.shape
    if sample_count <= 0 or patch_count <= 0 or feature_dim <= 0:
        raise ValueError("PCA calibration requires non-empty feature axes")

    flattened = _token_layer_norm(
        target_features.detach().float().cpu()
    ).reshape(-1, feature_dim).double()
    mean = flattened.mean(dim=0)
    centered = flattened - mean
    denominator = max(flattened.shape[0] - 1, 1)
    covariance = centered.transpose(0, 1).matmul(centered) / denominator
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min(0.0)
    eigenvectors = eigenvectors[:, order]

    maximum = float(eigenvalues[0]) if eigenvalues.numel() else 0.0
    tolerance = (
        torch.finfo(torch.float64).eps
        * max(flattened.shape[0], feature_dim)
        * max(maximum, 1.0)
    )
    valid_rank = int((eigenvalues > tolerance).sum().item())
    component_count = min(3, feature_dim, valid_rank)
    components = torch.zeros((feature_dim, 3), dtype=torch.float64)
    if component_count:
        components[:, :component_count] = eigenvectors[:, :component_count]
        for index in range(component_count):
            column = components[:, index]
            pivot = int(column.abs().argmax())
            if float(column[pivot]) < 0:
                components[:, index].neg_()

    scores = centered.matmul(components)
    scale = torch.ones(3, dtype=torch.float64)
    if component_count:
        calibrated = torch.quantile(
            scores[:, :component_count].abs(),
            0.995,
            dim=0,
        )
        scale[:component_count] = calibrated.clamp_min(1e-12)
    total_variance = eigenvalues.sum()
    explained = torch.zeros(3, dtype=torch.float64)
    if float(total_variance) > tolerance:
        explained[:component_count] = (
            eigenvalues[:component_count] / total_variance
        )

    digest = hashlib.sha256()
    for value in (mean, components, scale):
        digest.update(value.contiguous().numpy().tobytes())
    digest.update(f"{valid_rank}:{flattened.shape[0]}".encode("ascii"))
    return SharedPCABasis(
        mean=mean,
        components=components,
        scale=scale,
        explained_variance_ratio=explained,
        valid_rank=valid_rank,
        calibration_tokens=flattened.shape[0],
        basis_id=digest.hexdigest(),
    )


@torch.no_grad()
def project_with_shared_pca(
    features: torch.Tensor,
    basis: SharedPCABasis,
) -> tuple[torch.Tensor, float]:
    """Project to CPU without changing the target-calibrated display scale."""

    if features.ndim < 2 or features.shape[-1] != basis.mean.numel():
        raise ValueError("features do not match the shared PCA feature dimension")
    normalized = _token_layer_norm(
        features.detach().float().cpu()
    ).double()
    scores = (normalized - basis.mean).matmul(basis.components)
    scaled = scores / basis.scale
    active_components = min(3, basis.valid_rank)
    clip_fraction = (
        float(
            (scaled[..., :active_components].abs() > 1.0).float().mean()
        )
        if active_components
        else 0.0
    )
    return scaled.clamp(-1.0, 1.0), clip_fraction


def pca_patch_rgb(
    features: torch.Tensor,
    basis: SharedPCABasis,
    grid_size: tuple[int, int],
    image_size: tuple[int, int],
) -> tuple[np.ndarray, float]:
    """Render one [N,D] token grid with nearest-neighbour patch expansion."""

    if features.ndim != 2:
        raise ValueError("one PCA panel requires features with shape [N,D]")
    grid_height, grid_width = grid_size
    image_height, image_width = image_size
    if features.shape[0] != grid_height * grid_width:
        raise ValueError("feature token count does not match the patch grid")
    if image_height % grid_height or image_width % grid_width:
        raise ValueError("image_size must be divisible by grid_size")
    projected, clip_fraction = project_with_shared_pca(features, basis)
    rgb = torch.round(127.5 * (projected + 1.0)).to(torch.uint8)
    image = rgb.reshape(grid_height, grid_width, 3).cpu().numpy()
    image = np.repeat(image, image_height // grid_height, axis=0)
    image = np.repeat(image, image_width // grid_width, axis=1)
    return np.ascontiguousarray(image), clip_fraction


def token_cosine_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-token cosine error after the same token LayerNorm used by training."""

    if prediction.shape != target.shape or prediction.ndim not in {2, 3}:
        raise ValueError("prediction and target must share [N,D] or [B,N,D]")
    prediction = _token_layer_norm(prediction)
    target = _token_layer_norm(target)
    return 1.0 - functional.cosine_similarity(prediction, target, dim=-1)


def cosine_error_rgb(
    error: torch.Tensor,
    grid_size: tuple[int, int],
    image_size: tuple[int, int],
) -> np.ndarray:
    """Render a fixed [0,2] cosine-error heatmap; lower is better."""

    if error.ndim != 1 or error.numel() != math.prod(grid_size):
        raise ValueError("error must contain one scalar per patch")
    if not bool(torch.isfinite(error).all()):
        raise FloatingPointError("cosine error contains NaN or infinity")
    values = error.detach().float().clamp(0.0, 2.0).cpu().numpy()
    anchors = np.asarray(
        [
            [13.0, 25.0, 60.0],
            [20.0, 184.0, 166.0],
            [250.0, 204.0, 21.0],
            [220.0, 38.0, 38.0],
        ],
        dtype=np.float32,
    )
    positions = np.asarray([0.0, 0.5, 1.0, 2.0], dtype=np.float32)
    channels = [np.interp(values, positions, anchors[:, index]) for index in range(3)]
    rgb = np.stack(channels, axis=-1).reshape(*grid_size, 3)
    image_height, image_width = image_size
    grid_height, grid_width = grid_size
    if image_height % grid_height or image_width % grid_width:
        raise ValueError("image_size must be divisible by grid_size")
    rgb = np.repeat(rgb, image_height // grid_height, axis=0)
    rgb = np.repeat(rgb, image_width // grid_width, axis=1)
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def make_history_permutation(
    history_steps: int,
    *,
    seed: int,
    epoch: int,
    sample_index: int,
    online_step: int,
) -> tuple[int, ...]:
    """Return a deterministic non-identity permutation of the past prefix."""

    if history_steps < 0:
        raise ValueError("history_steps cannot be negative")
    if history_steps < 2:
        return tuple(range(history_steps))
    mixed_seed = deterministic_seed(
        seed,
        epoch,
        sample_index,
        stream=20_000 + online_step,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(mixed_seed & ((1 << 63) - 1))
    permutation = torch.randperm(history_steps, generator=generator)
    identity = torch.arange(history_steps)
    if torch.equal(permutation, identity):
        permutation = torch.roll(identity, shifts=1)
    return tuple(int(value) for value in permutation)


def make_unrelated_clip_permutation(clip_count: int, seed: int) -> tuple[int, ...]:
    """Create a deterministic clip-level derangement."""

    if clip_count < 2:
        raise ValueError("at least two clips are required for unrelated targets")
    if seed < 0:
        raise ValueError("unrelated-target seed cannot be negative")
    base_shift = max(1, (clip_count + 1) // 2)
    shift = 1 + (base_shift - 1 + int(seed)) % (clip_count - 1)
    return tuple((index + shift) % clip_count for index in range(clip_count))


def make_history_replacement_clip_permutation(
    clip_identities: Sequence[tuple[str, tuple[int, ...]]],
    seed: int,
) -> tuple[int, ...]:
    """Map each clip to a distinct, non-duplicate history donor clip."""

    if seed < 0:
        raise ValueError("history-replacement seed cannot be negative")
    clip_count = len(clip_identities)
    if clip_count < 2:
        raise ValueError("history replacement requires at least two clips")
    first_shift = make_unrelated_clip_permutation(clip_count, seed)[0]
    candidate_shifts = tuple(
        1 + (first_shift - 1 + offset) % (clip_count - 1)
        for offset in range(clip_count - 1)
    )
    for shift in candidate_shifts:
        mapping = tuple((index + shift) % clip_count for index in range(clip_count))
        if all(
            clip_identities[source] != clip_identities[target]
            for source, target in enumerate(mapping)
        ):
            return mapping
    raise ValueError(
        "calibration clips contain duplicate history anchors; select more samples "
        "or change sample-index"
    )


def make_unrelated_record_permutation(
    records: Sequence[FutureFeatureStepRecord],
    seed: int,
) -> tuple[int, ...]:
    """Map every record to the same step in a distinct deterministic clip."""

    if seed < 0:
        raise ValueError("unrelated-target seed cannot be negative")
    by_clip: dict[int, dict[int, int]] = {}
    for record in records:
        steps = by_clip.setdefault(record.clip_position, {})
        if record.online_step in steps:
            raise ValueError("clip contains a duplicate supervised online step")
        steps[record.online_step] = record.record_index
    clip_positions = tuple(sorted(by_clip))
    if len(clip_positions) < 2:
        raise ValueError(
            "unrelated-target control requires at least two calibration clips"
        )
    expected_steps = set(by_clip[clip_positions[0]])
    if not expected_steps or any(
        set(by_clip[position]) != expected_steps for position in clip_positions[1:]
    ):
        raise ValueError("calibration clips do not share the same supervised steps")
    if any(
        record.record_index != position
        for position, record in enumerate(records)
    ):
        raise ValueError("future feature records must be ordered by record_index")

    clip_count = len(clip_positions)
    first_shift = make_unrelated_clip_permutation(clip_count, seed)[0]
    candidate_shifts = tuple(
        1 + (first_shift - 1 + offset) % (clip_count - 1)
        for offset in range(clip_count - 1)
    )
    for shift in candidate_shifts:
        mapping = [-1] * len(records)
        collision = False
        for source_position, source_clip in enumerate(clip_positions):
            target_clip = clip_positions[(source_position + shift) % clip_count]
            for online_step in expected_steps:
                source_index = by_clip[source_clip][online_step]
                target_index = by_clip[target_clip][online_step]
                source = records[source_index]
                target = records[target_index]
                if (
                    source.sequence_id,
                    source.target_t_end_us,
                ) == (
                    target.sequence_id,
                    target.target_t_end_us,
                ):
                    collision = True
                    break
                mapping[source_index] = target_index
            if collision:
                break
        if not collision and all(index >= 0 for index in mapping):
            return tuple(mapping)
    raise ValueError(
        "calibration clips contain duplicate future anchors; select more samples "
        "or change sample-index"
    )


def _detached_tokens(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.ndim != 3 or value.shape[0] != 1:
        raise RuntimeError("future feature extraction expected one [1,N,D] latent")
    result = value[0].detach().float().cpu()
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError(f"{name} contains NaN or infinity")
    return result


def _require_same_latent(
    first: torch.Tensor,
    second: torch.Tensor,
    name: str,
) -> None:
    if first.shape != second.shape or not torch.allclose(
        first,
        second,
        rtol=1e-5,
        atol=1e-6,
    ):
        raise RuntimeError(
            f"{name} changed across history controls; the evaluation is not paired"
        )


@torch.inference_mode()
def extract_future_feature_records(
    model: WindowJEPA,
    materialization: FutureFeatureMaterialization,
    *,
    device: torch.device,
    history_shuffle_seed: int = 0,
    history_replacement_seed: int = 0,
) -> tuple[FutureFeatureStepRecord, ...]:
    """Replay clips and extract paired history counterfactual latents."""

    if model.training:
        raise RuntimeError("feature extraction requires model.eval()")
    validate_future_feature_config(materialization.config)
    if history_shuffle_seed < 0:
        raise ValueError("history_shuffle_seed cannot be negative")
    if history_replacement_seed < 0:
        raise ValueError("history_replacement_seed cannot be negative")

    clip_identities = tuple(
        (
            str(clip.sample["sequence_id"]),
            tuple(int(value) for value in clip.sample["t_end_us"]),
        )
        for clip in materialization.clips
    )
    replacement_clip_permutation = make_history_replacement_clip_permutation(
        clip_identities,
        history_replacement_seed,
    )

    records: list[FutureFeatureStepRecord] = []
    for clip_position, clip in enumerate(materialization.clips):
        sample = clip.sample
        x = sample["x"].to(device=device, dtype=torch.float32)
        x_future = sample["x_future"].to(device=device, dtype=torch.float32)
        duration = sample["dt_ms"].to(device=device, dtype=torch.float32)
        future_duration = sample["future_dt_ms"].to(
            device=device,
            dtype=torch.float32,
        )
        replacement_clip_position = replacement_clip_permutation[clip_position]
        replacement_sample = materialization.clips[
            replacement_clip_position
        ].sample
        replacement_x = replacement_sample["x"].to(
            device=device,
            dtype=torch.float32,
        )
        replacement_duration = replacement_sample["dt_ms"].to(
            device=device,
            dtype=torch.float32,
        )
        if replacement_x.shape != x.shape or replacement_duration.shape != duration.shape:
            raise ValueError(
                "history replacement clips must share sequence and duration shapes"
            )
        loss_mask = sample["loss_mask"].detach().cpu().bool()
        supervised = tuple(int(index) for index in loss_mask.nonzero().flatten())
        if not supervised:
            raise ValueError("future feature clip has no supervised timestep")
        first_supervised = supervised[0]
        expected_supervised = tuple(range(first_supervised, len(loss_mask)))
        if supervised != expected_supervised:
            raise ValueError("future feature loss_mask must select one temporal suffix")

        state = None
        replacement_state = None
        if first_supervised:
            state = model.recurrent_burn_in(
                x[:first_supervised].unsqueeze(0),
                duration[:first_supervised].unsqueeze(0),
                None,
                online_state=None,
            )
            replacement_state = model.recurrent_burn_in(
                replacement_x[:first_supervised].unsqueeze(0),
                replacement_duration[:first_supervised].unsqueeze(0),
                None,
                online_state=None,
            )

        for online_step in supervised:
            step_arguments = {
                "x_context": x[online_step].unsqueeze(0),
                "x_future": x_future[online_step].unsqueeze(0),
                "context_duration_ms": duration[online_step].reshape(1),
                "target_duration_ms": future_duration[online_step].reshape(1),
            }
            correct = model.extract_recurrent_future_step(
                **step_arguments,
                online_state=state,
            )
            state = correct.online_state
            reset = model.extract_recurrent_future_step(
                **step_arguments,
                online_state=None,
            )

            permutation = make_history_permutation(
                online_step,
                seed=history_shuffle_seed,
                epoch=materialization.epoch,
                sample_index=clip.sample_index,
                online_step=online_step,
            )
            shuffled = None
            if online_step >= 2:
                permutation_tensor = torch.tensor(
                    permutation,
                    dtype=torch.long,
                    device=device,
                )
                shuffled_state = model.recurrent_burn_in(
                    x.index_select(0, permutation_tensor).unsqueeze(0),
                    duration.index_select(0, permutation_tensor).unsqueeze(0),
                    None,
                    online_state=None,
                )
                shuffled = model.extract_recurrent_future_step(
                    **step_arguments,
                    online_state=shuffled_state,
                )

            reversed_step = None
            if online_step >= 2:
                reverse_indices = torch.arange(
                    online_step - 1,
                    -1,
                    -1,
                    dtype=torch.long,
                    device=device,
                )
                reversed_state = model.recurrent_burn_in(
                    x.index_select(0, reverse_indices).unsqueeze(0),
                    duration.index_select(0, reverse_indices).unsqueeze(0),
                    None,
                    online_state=None,
                )
                reversed_step = model.extract_recurrent_future_step(
                    **step_arguments,
                    online_state=reversed_state,
                )

            replaced = None
            if online_step >= 1:
                replaced = model.extract_recurrent_future_step(
                    **step_arguments,
                    online_state=replacement_state,
                )

            correct_frame = _detached_tokens(correct.frame_tokens, "correct frame")
            correct_target = _detached_tokens(correct.target_tokens, "EMA target")
            _require_same_latent(
                correct_frame,
                _detached_tokens(reset.frame_tokens, "reset frame"),
                "frame latent",
            )
            _require_same_latent(
                correct_target,
                _detached_tokens(reset.target_tokens, "reset EMA target"),
                "EMA target",
            )
            if shuffled is not None:
                _require_same_latent(
                    correct_frame,
                    _detached_tokens(shuffled.frame_tokens, "shuffled frame"),
                    "frame latent",
                )
                _require_same_latent(
                    correct_target,
                    _detached_tokens(
                        shuffled.target_tokens,
                        "shuffled EMA target",
                    ),
                    "EMA target",
                )
            if reversed_step is not None:
                _require_same_latent(
                    correct_frame,
                    _detached_tokens(reversed_step.frame_tokens, "reversed frame"),
                    "frame latent",
                )
                _require_same_latent(
                    correct_target,
                    _detached_tokens(
                        reversed_step.target_tokens,
                        "reversed EMA target",
                    ),
                    "EMA target",
                )
            if replaced is not None:
                _require_same_latent(
                    correct_frame,
                    _detached_tokens(replaced.frame_tokens, "replaced-history frame"),
                    "frame latent",
                )
                _require_same_latent(
                    correct_target,
                    _detached_tokens(
                        replaced.target_tokens,
                        "replaced-history EMA target",
                    ),
                    "EMA target",
                )

            records.append(
                FutureFeatureStepRecord(
                    record_index=len(records),
                    clip_position=clip_position,
                    sample_index=clip.sample_index,
                    online_step=online_step,
                    sequence_id=str(sample["sequence_id"]),
                    context_t_end_us=int(sample["t_end_us"][online_step]),
                    target_t_end_us=int(sample["future_t_end_us"][online_step]),
                    frame_tokens=correct_frame,
                    recurrent_tokens=_detached_tokens(
                        correct.recurrent_tokens,
                        "correct recurrent state",
                    ),
                    prediction=_detached_tokens(
                        correct.prediction,
                        "correct prediction",
                    ),
                    target_tokens=correct_target,
                    reset_recurrent_tokens=_detached_tokens(
                        reset.recurrent_tokens,
                        "reset recurrent state",
                    ),
                    reset_prediction=_detached_tokens(
                        reset.prediction,
                        "reset prediction",
                    ),
                    shuffled_recurrent_tokens=(
                        None
                        if shuffled is None
                        else _detached_tokens(
                            shuffled.recurrent_tokens,
                            "shuffled recurrent state",
                        )
                    ),
                    shuffled_prediction=(
                        None
                        if shuffled is None
                        else _detached_tokens(
                            shuffled.prediction,
                            "shuffled prediction",
                        )
                    ),
                    target_activity=sample["future_patch_event_activity"][
                        online_step
                    ]
                    .detach()
                    .cpu(),
                    history_permutation=permutation,
                    reversed_recurrent_tokens=(
                        None
                        if reversed_step is None
                        else _detached_tokens(
                            reversed_step.recurrent_tokens,
                            "reversed recurrent state",
                        )
                    ),
                    reversed_prediction=(
                        None
                        if reversed_step is None
                        else _detached_tokens(
                            reversed_step.prediction,
                            "reversed prediction",
                        )
                    ),
                    replaced_recurrent_tokens=(
                        None
                        if replaced is None
                        else _detached_tokens(
                            replaced.recurrent_tokens,
                            "replaced-history recurrent state",
                        )
                    ),
                    replaced_prediction=(
                        None
                        if replaced is None
                        else _detached_tokens(
                            replaced.prediction,
                            "replaced-history prediction",
                        )
                    ),
                    replacement_clip_position=(
                        None if replaced is None else replacement_clip_position
                    ),
                )
            )
            if online_step != supervised[-1]:
                replacement_state = model.recurrent_burn_in(
                    replacement_x[online_step : online_step + 1].unsqueeze(0),
                    replacement_duration[
                        online_step : online_step + 1
                    ].unsqueeze(0),
                    None,
                    online_state=replacement_state,
                )
    if len({record.clip_position for record in records}) < 2:
        raise ValueError(
            "at least two calibration clips are required for unrelated targets"
        )
    return tuple(records)


def _effective_rank(
    matrix: torch.Tensor,
    *,
    maximum_rank: int | None = None,
) -> dict[str, float]:
    if matrix.ndim != 2 or matrix.shape[1] <= 0:
        raise ValueError("effective rank requires a [samples,features] matrix")
    values = matrix.detach().cpu().double()
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError("effective-rank features contain NaN or infinity")
    if values.shape[0] == 0:
        return {"effective_rank": 0.0, "normalized_effective_rank": 0.0}
    values = values - values.mean(dim=0, keepdim=True)
    if values.shape[0] <= values.shape[1]:
        eigenvalues = torch.linalg.eigvalsh(values.matmul(values.transpose(0, 1)))
    else:
        eigenvalues = torch.linalg.eigvalsh(values.transpose(0, 1).matmul(values))
    eigenvalues = eigenvalues.clamp_min(0.0)
    total = eigenvalues.sum()
    tolerance = torch.finfo(torch.float64).eps * max(matrix.shape) * max(
        float(eigenvalues.max()) if eigenvalues.numel() else 0.0,
        1.0,
    )
    if float(total) <= tolerance:
        effective = 0.0
    else:
        probabilities = eigenvalues / total
        positive = probabilities > 0
        entropy = -(probabilities[positive] * probabilities[positive].log()).sum()
        effective = float(entropy.exp())
    if maximum_rank is None:
        maximum_rank = min(matrix.shape[1], max(matrix.shape[0] - 1, 0))
    if maximum_rank < 0 or maximum_rank > min(matrix.shape):
        raise ValueError("maximum_rank is inconsistent with the feature matrix")
    return {
        "effective_rank": effective,
        "normalized_effective_rank": (
            effective / maximum_rank if maximum_rank else 0.0
        ),
    }


@torch.no_grad()
def fixed_support_latent_diagnostics(features: torch.Tensor) -> dict[str, float]:
    """Collapse diagnostics that separate position and spatial shortcuts."""

    if features.ndim != 3:
        raise ValueError("features must have shape [S,N,D]")
    sample_count, patch_count, feature_dim = features.shape
    if min(sample_count, patch_count, feature_dim) <= 0:
        raise ValueError("latent diagnostics require non-empty axes")
    values = features.detach().cpu().double()
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError("latent diagnostics contain NaN or infinity")
    fixed_position = values - values.mean(dim=0, keepdim=True)
    spatial_residual = values - values.mean(dim=1, keepdim=True)
    frame_pooled = values.mean(dim=1)
    diagnostics: dict[str, float] = {
        "fixed_position_std": float(
            values.std(dim=0, unbiased=False).mean()
        ),
        "spatial_std": float(values.std(dim=1, unbiased=False).mean()),
        "mean_token_norm": float(values.norm(dim=-1).mean()),
    }
    for name, matrix, maximum_rank in (
        (
            "global_token",
            values.reshape(-1, feature_dim),
            min(feature_dim, sample_count * patch_count - 1),
        ),
        (
            "fixed_position_centered",
            fixed_position.reshape(-1, feature_dim),
            min(feature_dim, patch_count * max(sample_count - 1, 0)),
        ),
        (
            "spatial_residual",
            spatial_residual.reshape(-1, feature_dim),
            min(feature_dim, sample_count * max(patch_count - 1, 0)),
        ),
        (
            "frame_pooled",
            frame_pooled,
            min(feature_dim, max(sample_count - 1, 0)),
        ),
    ):
        rank = _effective_rank(matrix, maximum_rank=maximum_rank)
        diagnostics[f"{name}_rank"] = rank["effective_rank"]
        diagnostics[f"{name}_normalized_rank"] = rank[
            "normalized_effective_rank"
        ]
    return diagnostics


@torch.no_grad()
def step_conditioned_latent_diagnostics(
    features: torch.Tensor,
    online_steps: torch.Tensor,
) -> dict[str, float]:
    """Average collapse diagnostics within fixed online-step populations."""

    if features.ndim != 3:
        raise ValueError("features must have shape [S,N,D]")
    if online_steps.ndim != 1 or online_steps.shape[0] != features.shape[0]:
        raise ValueError("online_steps must contain one value per feature sample")
    if online_steps.dtype == torch.bool or online_steps.is_floating_point():
        raise TypeError("online_steps must contain integer step indices")
    features = features.detach().cpu()
    online_steps = online_steps.detach().cpu()
    unique_steps = torch.unique(online_steps, sorted=True)
    if unique_steps.numel() == 0:
        raise ValueError("step-conditioned diagnostics require at least one step")
    grouped = [
        fixed_support_latent_diagnostics(
            features[online_steps == step]
        )
        for step in unique_steps
    ]
    keys = grouped[0].keys()
    return {
        f"step_conditioned_{key}": float(
            np.mean([diagnostics[key] for diagnostics in grouped])
        )
        for key in keys
    }


def _combined_latent_diagnostics(
    features: torch.Tensor,
    online_steps: torch.Tensor,
) -> dict[str, float]:
    raw = {
        **fixed_support_latent_diagnostics(features),
        **step_conditioned_latent_diagnostics(features, online_steps),
    }
    token_normalized = _token_layer_norm(features)
    loss_space = {
        **fixed_support_latent_diagnostics(token_normalized),
        **step_conditioned_latent_diagnostics(token_normalized, online_steps),
    }
    return {
        "anchor_count": float(features.shape[0]),
        "online_step_count": float(torch.unique(online_steps).numel()),
        **{f"raw_{key}": value for key, value in raw.items()},
        **{f"token_ln_{key}": value for key, value in loss_space.items()},
    }


def _mean_or_nan(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


@torch.no_grad()
def summarize_prediction_condition(
    prediction: torch.Tensor,
    target: torch.Tensor,
    activity: torch.Tensor,
    *,
    active_min_events: int,
    loss_kind: str,
    correct_prediction: torch.Tensor | None = None,
    correct_target: torch.Tensor | None = None,
) -> dict[str, float]:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("condition latents must share shape [S,N,D]")
    if activity.shape != prediction.shape[:2]:
        raise ValueError("condition activity must have shape [S,N]")
    if active_min_events <= 0:
        raise ValueError("active_min_events must be positive")
    balanced = balanced_event_support_latent_prediction_loss(
        prediction,
        target,
        activity,
        active_threshold=float(active_min_events - 1),
        kind=loss_kind,
    )
    balanced_cosine = balanced_event_support_latent_prediction_loss(
        prediction,
        target,
        activity,
        active_threshold=float(active_min_events - 1),
        kind="cosine",
    )
    error = token_cosine_error(prediction, target)
    active = activity >= active_min_events
    inactive = ~active

    active_means: list[float] = []
    inactive_means: list[float] = []
    for index in range(error.shape[0]):
        if bool(active[index].any()):
            active_means.append(float(error[index][active[index]].mean()))
        if bool(inactive[index].any()):
            inactive_means.append(float(error[index][inactive[index]].mean()))
    normalized_prediction = _token_layer_norm(prediction)
    normalized_target = _token_layer_norm(target)
    result = {
        "balanced_prediction_loss": float(balanced.loss),
        "balanced_active_loss": float(balanced.active_loss),
        "balanced_inactive_loss": float(balanced.inactive_loss),
        "balanced_cosine_error": float(balanced_cosine.loss),
        "balanced_active_cosine_error": float(balanced_cosine.active_loss),
        "balanced_inactive_cosine_error": float(
            balanced_cosine.inactive_loss
        ),
        "cosine_error_mean": float(error.mean(dim=1).mean()),
        "cosine_error_median": float(error.median(dim=1).values.mean()),
        "cosine_error_p90": float(torch.quantile(error, 0.9, dim=1).mean()),
        "active_cosine_error_mean": _mean_or_nan(active_means),
        "inactive_cosine_error_mean": _mean_or_nan(inactive_means),
        "prediction_zero_norm_fraction": float(
            (normalized_prediction.norm(dim=-1) <= 1e-8).float().mean()
        ),
        "target_zero_norm_fraction": float(
            (normalized_target.norm(dim=-1) <= 1e-8).float().mean()
        ),
        "anchors": float(prediction.shape[0]),
    }
    if correct_prediction is not None or correct_target is not None:
        if correct_prediction is None or correct_target is None:
            raise ValueError("both correct_prediction and correct_target are required")
        if correct_prediction.shape != prediction.shape or correct_target.shape != target.shape:
            raise ValueError("correct reference latents must match the condition")
        correct_error = token_cosine_error(correct_prediction, correct_target)
        result["paired_cosine_penalty"] = float(
            (error.mean(dim=1) - correct_error.mean(dim=1)).mean()
        )
        result["prediction_drift_from_correct"] = float(
            token_cosine_error(prediction, correct_prediction).mean()
        )
    return result


@torch.no_grad()
def analyze_future_feature_records(
    records: Sequence[FutureFeatureStepRecord],
    config: ExperimentConfig,
    *,
    unrelated_seed: int,
) -> tuple[dict[str, Any], tuple[int, ...]]:
    if len({record.clip_position for record in records}) < 2:
        raise ValueError("future feature analysis requires at least two clips")
    prediction = torch.stack([record.prediction for record in records])
    target = torch.stack([record.target_tokens for record in records])
    activity = torch.stack([record.target_activity for record in records])
    reset = torch.stack([record.reset_prediction for record in records])
    unrelated_permutation = make_unrelated_record_permutation(
        records,
        unrelated_seed,
    )
    unrelated_indices = torch.tensor(unrelated_permutation, dtype=torch.long)
    unrelated_target = target.index_select(0, unrelated_indices)
    unrelated_activity = activity.index_select(0, unrelated_indices)

    conditions: dict[str, dict[str, float]] = {
        "correct": summarize_prediction_condition(
            prediction,
            target,
            activity,
            active_min_events=config.future_prediction.active_min_events,
            loss_kind="smooth_l1",
        ),
        "reset": summarize_prediction_condition(
            reset,
            target,
            activity,
            active_min_events=config.future_prediction.active_min_events,
            loss_kind="smooth_l1",
            correct_prediction=prediction,
            correct_target=target,
        ),
        "unrelated_target": summarize_prediction_condition(
            prediction,
            unrelated_target,
            unrelated_activity,
            active_min_events=config.future_prediction.active_min_events,
            loss_kind="smooth_l1",
            correct_prediction=prediction,
            correct_target=target,
        ),
    }
    conditions["reset"]["paired_balanced_cosine_penalty"] = (
        conditions["reset"]["balanced_cosine_error"]
        - conditions["correct"]["balanced_cosine_error"]
    )
    conditions["unrelated_target"]["paired_balanced_cosine_penalty"] = (
        conditions["unrelated_target"]["balanced_cosine_error"]
        - conditions["correct"]["balanced_cosine_error"]
    )
    history_controls: dict[
        str,
        tuple[list[tuple[int, FutureFeatureStepRecord]], torch.Tensor, torch.Tensor],
    ] = {}
    for condition_name, prediction_field in (
        ("history_shuffled", "shuffled_prediction"),
        ("history_reversed", "reversed_prediction"),
        ("history_replaced", "replaced_prediction"),
    ):
        available = [
            (index, record)
            for index, record in enumerate(records)
            if getattr(record, prediction_field) is not None
        ]
        if not available:
            continue
        selected = torch.tensor(
            [index for index, _ in available],
            dtype=torch.long,
        )
        control_prediction = torch.stack(
            [getattr(record, prediction_field) for _, record in available]
        )
        selected_correct = summarize_prediction_condition(
            prediction.index_select(0, selected),
            target.index_select(0, selected),
            activity.index_select(0, selected),
            active_min_events=config.future_prediction.active_min_events,
            loss_kind="smooth_l1",
        )
        conditions[condition_name] = summarize_prediction_condition(
            control_prediction,
            target.index_select(0, selected),
            activity.index_select(0, selected),
            active_min_events=config.future_prediction.active_min_events,
            loss_kind="smooth_l1",
            correct_prediction=prediction.index_select(0, selected),
            correct_target=target.index_select(0, selected),
        )
        conditions[condition_name]["paired_balanced_cosine_penalty"] = (
            conditions[condition_name]["balanced_cosine_error"]
            - selected_correct["balanced_cosine_error"]
        )
        history_controls[condition_name] = (
            available,
            selected,
            control_prediction,
        )

    all_steps = torch.tensor(
        [record.online_step for record in records],
        dtype=torch.int64,
    )
    diagnostic_features: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
        "ema_target": (target, all_steps),
        "online_frame": (
            torch.stack([record.frame_tokens for record in records]),
            all_steps,
        ),
        "recurrent": (
            torch.stack([record.recurrent_tokens for record in records]),
            all_steps,
        ),
        "prediction": (prediction, all_steps),
        "reset_recurrent": (
            torch.stack([record.reset_recurrent_tokens for record in records]),
            all_steps,
        ),
        "reset_prediction": (reset, all_steps),
    }
    for condition_name, recurrent_field in (
        ("history_shuffled", "shuffled_recurrent_tokens"),
        ("history_reversed", "reversed_recurrent_tokens"),
        ("history_replaced", "replaced_recurrent_tokens"),
    ):
        if condition_name not in history_controls:
            continue
        available, selected, control_prediction = history_controls[condition_name]
        control_steps = all_steps.index_select(0, selected)
        diagnostic_features[f"{condition_name}_recurrent"] = (
            torch.stack(
                [getattr(record, recurrent_field) for _, record in available]
            ),
            control_steps,
        )
        diagnostic_features[f"{condition_name}_prediction"] = (
            control_prediction,
            control_steps,
        )
    diagnostics = {
        name: _combined_latent_diagnostics(features, steps)
        for name, (features, steps) in diagnostic_features.items()
    }
    return {
        "conditions": conditions,
        "latent_diagnostics": diagnostics,
    }, unrelated_permutation


def _write_png(path: Path, image: np.ndarray) -> None:
    encoded = _png_data_uri(image).partition(",")[2]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_bytes(base64.b64decode(encoded))
    temporary.replace(path)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _asset_reference(output: Path, asset_path: Path) -> str:
    relative = asset_path.relative_to(output.parent)
    return "/".join(quote(part) for part in relative.parts)


def _clip_sampling_records(
    materialization: FutureFeatureMaterialization,
) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for clip in materialization.clips:
        transform = clip.debug.spatial_transform
        values.append(
            {
                "sample_index": clip.sample_index,
                "augmentation_seed": int(clip.sample["augmentation_seed"]),
                "augmentation_id": str(clip.sample["augmentation_id"]),
                "crop": [
                    transform.x0,
                    transform.y0,
                    transform.output_height,
                    transform.output_width,
                ],
                "horizontal_flip": transform.horizontal_flip,
            }
        )
    return values


def _sample_set_id(
    records: Sequence[FutureFeatureStepRecord],
    materialization: FutureFeatureMaterialization,
) -> str:
    anchors = [
        {
            "sample_index": record.sample_index,
            "sequence_id": record.sequence_id,
            "context_t_end_us": record.context_t_end_us,
            "target_t_end_us": record.target_t_end_us,
        }
        for record in records
    ]
    values = {
        "anchors": anchors,
        "augmentations": _clip_sampling_records(materialization),
    }
    serialized = json.dumps(values, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _validate_report_compatibility(
    model: WindowJEPA,
    checkpoint_config: ExperimentConfig,
    materialization: FutureFeatureMaterialization,
) -> None:
    restored_manifest = replace(
        materialization.config,
        data=replace(
            materialization.config.data,
            manifest=checkpoint_config.data.manifest,
        ),
    )
    if restored_manifest != checkpoint_config:
        raise ValueError(
            "materialization config differs from the checkpoint beyond manifest path"
        )
    expected_patches = (
        checkpoint_config.model.image_size[0] // checkpoint_config.model.patch_size
    ) * (
        checkpoint_config.model.image_size[1] // checkpoint_config.model.patch_size
    )
    if model.num_patches != expected_patches:
        raise ValueError("model patch grid differs from the checkpoint configuration")
    if model.online_encoder.embed_dim != checkpoint_config.model.embed_dim:
        raise ValueError("model feature dimension differs from checkpoint configuration")
    if getattr(model.online_encoder, "recurrent_placement", None) != (
        checkpoint_config.recurrent.recurrent_placement
    ):
        raise ValueError("model recurrent placement differs from checkpoint configuration")


def _condition_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    error = token_cosine_error(prediction, target)
    return error, float(error.mean())


def _step_records(
    records: Sequence[FutureFeatureStepRecord],
    unrelated_permutation: Sequence[int],
    *,
    active_min_events: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        unrelated = records[unrelated_permutation[index]]
        _, correct_error = _condition_error(record.prediction, record.target_tokens)
        _, reset_error = _condition_error(
            record.reset_prediction,
            record.target_tokens,
        )
        _, unrelated_error = _condition_error(
            record.prediction,
            unrelated.target_tokens,
        )
        shuffled_error = float("nan")
        if record.shuffled_prediction is not None:
            _, shuffled_error = _condition_error(
                record.shuffled_prediction,
                record.target_tokens,
            )
        reversed_error = float("nan")
        if record.reversed_prediction is not None:
            _, reversed_error = _condition_error(
                record.reversed_prediction,
                record.target_tokens,
            )
        replaced_error = float("nan")
        replacement_sample_index: int | None = None
        replacement_sequence_id: str | None = None
        if record.replaced_prediction is not None:
            _, replaced_error = _condition_error(
                record.replaced_prediction,
                record.target_tokens,
            )
            if record.replacement_clip_position is None:
                raise RuntimeError("replaced history is missing its donor clip")
            replacement_record = next(
                donor
                for donor in records
                if donor.clip_position == record.replacement_clip_position
            )
            replacement_sample_index = replacement_record.sample_index
            replacement_sequence_id = replacement_record.sequence_id
        rows.append(
            {
                "record_index": record.record_index,
                "sample_index": record.sample_index,
                "online_step": record.online_step,
                "sequence_id": record.sequence_id,
                "context_t_end_us": record.context_t_end_us,
                "target_t_end_us": record.target_t_end_us,
                "active_patch_fraction": float(
                    (
                        record.target_activity
                        >= active_min_events
                    ).float().mean()
                ),
                "correct_cosine_error": correct_error,
                "history_shuffled_cosine_error": shuffled_error,
                "history_shuffle_penalty": shuffled_error - correct_error,
                "history_reversed_cosine_error": reversed_error,
                "history_reverse_penalty": reversed_error - correct_error,
                "history_replaced_cosine_error": replaced_error,
                "history_replacement_penalty": replaced_error - correct_error,
                "history_replacement_clip_position": (
                    record.replacement_clip_position
                ),
                "history_replacement_sample_index": replacement_sample_index,
                "history_replacement_sequence_id": replacement_sequence_id,
                "reset_cosine_error": reset_error,
                "reset_penalty": reset_error - correct_error,
                "unrelated_target_cosine_error": unrelated_error,
                "unrelated_target_penalty": unrelated_error - correct_error,
                "unrelated_record_index": unrelated.record_index,
                "history_permutation": list(record.history_permutation),
                "history_reverse_order": list(
                    range(record.online_step - 1, -1, -1)
                ),
            }
        )
    return rows


def _format_metric(value: float, digits: int = 4) -> str:
    if not math.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _diagnostic_rows(diagnostics: dict[str, dict[str, float]]) -> str:
    labels = {
        "ema_target": "EMA target",
        "online_frame": "online frame",
        "recurrent": "recurrent state",
        "prediction": "prediction",
        "history_shuffled_recurrent": "history shuffled state",
        "history_shuffled_prediction": "history shuffled",
        "history_reversed_recurrent": "history reversed state",
        "history_reversed_prediction": "history reversed",
        "history_replaced_recurrent": "history replaced state",
        "history_replaced_prediction": "history replaced",
        "reset_recurrent": "state reset recurrent",
        "reset_prediction": "state reset",
    }
    rows: list[str] = []
    for name, values in diagnostics.items():
        raw_std = values["raw_step_conditioned_fixed_position_std"]
        token_std = values["token_ln_step_conditioned_fixed_position_std"]
        fixed_rank = values[
            "token_ln_step_conditioned_fixed_position_centered_rank"
        ]
        spatial_rank = values[
            "token_ln_step_conditioned_spatial_residual_rank"
        ]
        pooled_rank = values["token_ln_step_conditioned_frame_pooled_rank"]
        rows.append(
            "<tr>"
            f"<th>{html.escape(labels.get(name, name))}</th>"
            f"<td>{int(values['anchor_count'])}</td>"
            f"<td>{_format_metric(raw_std)}</td>"
            f"<td>{_format_metric(token_std)}</td>"
            f"<td>{_format_metric(fixed_rank, 2)}</td>"
            f"<td>{_format_metric(spatial_rank, 2)}</td>"
            f"<td>{_format_metric(pooled_rank, 2)}</td>"
            "</tr>"
        )
    return "".join(rows)


def _temporal_rows(
    step_records: Sequence[dict[str, Any]],
    sample_index: int,
) -> str:
    rows: list[str] = []
    for record in step_records:
        if int(record["sample_index"]) != sample_index:
            continue
        rows.append(
            "<tr>"
            f"<td>{int(record['online_step'])}</td>"
            f"<td>{int(record['context_t_end_us']):,}</td>"
            f"<td>{_format_metric(float(record['correct_cosine_error']))}</td>"
            f"<td>{_format_metric(float(record['history_shuffled_cosine_error']))}</td>"
            f"<td>{_format_metric(float(record['history_reversed_cosine_error']))}</td>"
            f"<td>{_format_metric(float(record['history_replaced_cosine_error']))}</td>"
            f"<td>{_format_metric(float(record['reset_cosine_error']))}</td>"
            f"<td>{_format_metric(float(record['unrelated_target_cosine_error']))}</td>"
            "</tr>"
        )
    return "".join(rows)


def _panel_figure(panel: dict[str, Any]) -> str:
    return (
        "<figure>"
        f"<img src=\"{html.escape(str(panel['src']))}\" "
        f"alt=\"{html.escape(str(panel['label']))}\">"
        f"<figcaption><b>{html.escape(str(panel['label']))}</b>"
        f"<span>{html.escape(str(panel['caption']))}</span></figcaption>"
        "</figure>"
    )


def _visual_record_indices(
    records: Sequence[FutureFeatureStepRecord],
    sample_index: int,
    all_steps: bool,
) -> tuple[int, ...]:
    matching = [
        record.record_index
        for record in records
        if record.sample_index == sample_index
    ]
    if not matching:
        raise ValueError("display sample is absent from the calibration records")
    return tuple(matching if all_steps else matching[-1:])


def _render_visual_sections(
    records: Sequence[FutureFeatureStepRecord],
    materialization: FutureFeatureMaterialization,
    basis: SharedPCABasis,
    unrelated_permutation: Sequence[int],
    output: Path,
    visual_record_indices: Sequence[int],
) -> tuple[str, list[dict[str, Any]]]:
    config = materialization.config
    grid_size = (
        config.model.image_size[0] // config.model.patch_size,
        config.model.image_size[1] // config.model.patch_size,
    )
    image_size = config.model.image_size
    recurrent_label = (
        "ConvLSTM"
        if config.recurrent.temporal_model == "conv_lstm"
        else "ConvGRU"
    )
    assets = (
        output.parent
        / f"{output.stem}_assets"
        / basis.basis_id[:12]
    )
    assets.mkdir(parents=True, exist_ok=True)

    all_event_counts = [
        _event_counts(window)
        for clip in materialization.clips
        for window in clip.debug.windows
    ]
    event_scale = _display_scale([np.log1p(counts) for counts in all_event_counts])
    sections: list[str] = []
    visual_records: list[dict[str, Any]] = []
    for record_index in visual_record_indices:
        record = records[record_index]
        unrelated = records[unrelated_permutation[record_index]]
        clip = materialization.clips[record.clip_position]
        unrelated_clip = materialization.clips[unrelated.clip_position]
        replacement_clip = (
            None
            if record.replacement_clip_position is None
            else materialization.clips[record.replacement_clip_position]
        )
        horizon = materialization.dataset.lookahead_steps
        current_window = clip.debug.windows[record.online_step]
        future_window = clip.debug.windows[record.online_step + horizon]
        unrelated_window = unrelated_clip.debug.windows[
            unrelated.online_step + horizon
        ]

        prefix = f"sample-{record.sample_index:05d}-step-{record.online_step:02d}"
        panels: list[dict[str, Any]] = []
        image_records: dict[str, str] = {}

        def save_panel(
            key: str,
            label: str,
            caption: str,
            image: np.ndarray,
            *,
            clip_fraction: float | None = None,
        ) -> None:
            path = assets / f"{prefix}-{key}.png"
            _write_png(path, image)
            relative = str(path.relative_to(output.parent))
            image_records[key] = relative
            panel: dict[str, Any] = {
                "key": key,
                "label": label,
                "caption": caption,
                "src": _asset_reference(output, path),
            }
            if clip_fraction is not None:
                panel["clip_fraction"] = clip_fraction
            panels.append(panel)

        save_panel(
            "current-events",
            "現在のevents  E_t",
            f"events={current_window.event_count:,}",
            _event_rgb(_event_counts(current_window), event_scale),
        )
        save_panel(
            "future-events",
            "正しい未来events  E_t+k",
            f"events={future_window.event_count:,}",
            _event_rgb(_event_counts(future_window), event_scale),
        )
        save_panel(
            "unrelated-events",
            "無関係な未来events",
            f"sample={unrelated.sample_index}, step={unrelated.online_step}",
            _event_rgb(_event_counts(unrelated_window), event_scale),
        )
        if replacement_clip is not None and record.online_step >= 1:
            replacement_history_window = replacement_clip.debug.windows[
                record.online_step - 1
            ]
            save_panel(
                "replacement-history-events",
                "別clip履歴の末尾events",
                (
                    f"sample={replacement_clip.sample_index}, "
                    f"step={record.online_step - 1}"
                ),
                _event_rgb(
                    _event_counts(replacement_history_window),
                    event_scale,
                ),
            )

        feature_panels = (
            ("frame", "Frame ViT  f_t", record.frame_tokens),
            ("recurrent", f"{recurrent_label} state  h_t", record.recurrent_tokens),
            ("prediction", "正しい履歴からの予測", record.prediction),
            ("target", "EMA future target", record.target_tokens),
            (
                "history-shuffled-recurrent",
                "履歴順を崩した state",
                record.shuffled_recurrent_tokens,
            ),
            (
                "history-shuffled-prediction",
                "履歴順を崩した予測",
                record.shuffled_prediction,
            ),
            (
                "history-reversed-recurrent",
                "履歴を逆順にした state",
                record.reversed_recurrent_tokens,
            ),
            (
                "history-reversed-prediction",
                "履歴を逆順にした予測",
                record.reversed_prediction,
            ),
            (
                "history-replaced-recurrent",
                "別clip履歴の state",
                record.replaced_recurrent_tokens,
            ),
            (
                "history-replaced-prediction",
                "別clip履歴からの予測",
                record.replaced_prediction,
            ),
            (
                "reset-recurrent",
                "履歴なしの state",
                record.reset_recurrent_tokens,
            ),
            ("reset-prediction", "履歴なしの予測", record.reset_prediction),
            ("unrelated-target", "無関係なEMA target", unrelated.target_tokens),
        )
        for key, label, features in feature_panels:
            if features is None:
                continue
            image, clip_fraction = pca_patch_rgb(
                features,
                basis,
                grid_size,
                image_size,
            )
            save_panel(
                key,
                label,
                f"shared PCA RGB · clip={clip_fraction:.3f}",
                image,
                clip_fraction=clip_fraction,
            )

        error_panels: list[tuple[str, str, torch.Tensor, torch.Tensor]] = [
            (
                "correct-error",
                "正しい予測の誤差",
                record.prediction,
                record.target_tokens,
            )
        ]
        if record.shuffled_prediction is not None:
            error_panels.append(
                (
                    "history-shuffled-error",
                    "履歴順を崩した誤差",
                    record.shuffled_prediction,
                    record.target_tokens,
                ),
            )
        if record.reversed_prediction is not None:
            error_panels.append(
                (
                    "history-reversed-error",
                    "履歴を逆順にした誤差",
                    record.reversed_prediction,
                    record.target_tokens,
                ),
            )
        if record.replaced_prediction is not None:
            error_panels.append(
                (
                    "history-replaced-error",
                    "別clip履歴での誤差",
                    record.replaced_prediction,
                    record.target_tokens,
                ),
            )
        error_panels.extend(
            (
                (
                    "reset-error",
                    "履歴なし予測の誤差",
                    record.reset_prediction,
                    record.target_tokens,
                ),
                (
                    "unrelated-error",
                    "無関係targetとの誤差",
                    record.prediction,
                    unrelated.target_tokens,
                ),
            )
        )
        panel_errors: dict[str, float] = {}
        for key, label, prediction, target in error_panels:
            error, mean_error = _condition_error(prediction, target)
            panel_errors[key] = mean_error
            save_panel(
                key,
                label,
                f"token-LN cosine error · mean={mean_error:.4f}",
                cosine_error_rgb(error, grid_size, image_size),
            )

        panel_html = "".join(_panel_figure(panel) for panel in panels)
        sections.append(
            f"""
            <section class="visual-step">
              <h2>sample {record.sample_index} · online step {record.online_step}</h2>
              <p class="metadata"><code>{html.escape(record.sequence_id)}</code>
                <span>context={record.context_t_end_us:,} μs</span>
                <span>future={record.target_t_end_us:,} μs</span></p>
              <div class="panel-grid">{panel_html}</div>
            </section>
            """
        )
        visual_records.append(
            {
                "record_index": record.record_index,
                "sample_index": record.sample_index,
                "online_step": record.online_step,
                "images": image_records,
                "mean_cosine_errors": panel_errors,
                "pca_clip_fractions": {
                    str(panel["key"]): float(panel["clip_fraction"])
                    for panel in panels
                    if "clip_fraction" in panel
                },
            }
        )
    return "".join(sections), visual_records


def _report_html(
    *,
    checkpoint: Path,
    sample_index: int,
    basis: SharedPCABasis,
    analysis: dict[str, Any],
    step_records: Sequence[dict[str, Any]],
    visual_sections: str,
) -> str:
    conditions = analysis["conditions"]
    correct = conditions["correct"]
    reset = conditions["reset"]
    unrelated = conditions["unrelated_target"]
    shuffled = conditions.get("history_shuffled", {})
    reversed_condition = conditions.get("history_reversed", {})
    replaced = conditions.get("history_replaced", {})
    pca_variance = 100.0 * float(basis.explained_variance_ratio.sum())
    return f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Causal Future Event JEPA 特徴可視化</title>
<style>
:root {{ color-scheme: light dark; --bg:#f5f7fb; --surface:#fff; --text:#172033;
  --muted:#647089; --line:#dbe1eb; --accent:#3157d5; --good:#0b7a55; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#10131a; --surface:#191e28;
  --text:#eef2fb; --muted:#a9b3c6; --line:#303848; --accent:#8aa4ff; --good:#5ed6a7; }} }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--text);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
main {{ max-width:1440px; margin:auto; padding:28px; }} h1 {{ margin:0 0 8px; font-size:1.7rem; }}
h2 {{ margin:0 0 14px; font-size:1.2rem; }} p {{ line-height:1.65; }} code {{ word-break:break-all; }}
.muted,.metadata {{ color:var(--muted); }} .metadata {{ display:flex; gap:14px; flex-wrap:wrap; }}
.summary {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px;
  margin:22px 0; }} .stat {{ background:var(--surface); border:1px solid var(--line);
  border-radius:12px; padding:15px; }} .stat span {{ display:block; color:var(--muted); font-size:.85rem; }}
.stat strong {{ display:block; margin-top:6px; font-size:1.35rem; font-variant-numeric:tabular-nums; }}
.explanation {{ border-left:4px solid var(--accent); padding:4px 0 4px 14px; margin:20px 0 28px; }}
.table-wrap {{ overflow-x:auto; margin:12px 0 30px; }} table {{ width:100%; border-collapse:collapse;
  background:var(--surface); }} th,td {{ padding:10px 12px; border-bottom:1px solid var(--line);
  text-align:right; white-space:nowrap; font-variant-numeric:tabular-nums; }} th:first-child,td:first-child {{ text-align:left; }}
.visual-step {{ margin:34px 0 46px; }} .panel-grid {{ display:grid;
  grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:16px; }} figure {{ margin:0;
  background:var(--surface); border:1px solid var(--line); border-radius:12px; overflow:hidden; }}
figure img {{ display:block; width:100%; height:auto; image-rendering:pixelated;
  background:#080b12; }} figcaption {{ display:flex; flex-direction:column; gap:4px;
  padding:11px 12px 13px; }} figcaption span {{ color:var(--muted); font-size:.84rem; }}
.legend {{ display:flex; gap:18px; flex-wrap:wrap; color:var(--muted); margin:8px 0 20px; }}
.swatch {{ display:inline-block; width:76px; height:10px; margin-right:7px; vertical-align:middle;
  background:linear-gradient(90deg,#0d193c,#14b8a6,#facc15,#dc2626); }}
@media (max-width:600px) {{ main {{ padding:18px; }} .panel-grid {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body><main>
<h1>Causal Future Event JEPA 特徴可視化</h1>
<p class="muted">checkpoint: <code>{html.escape(str(checkpoint))}</code></p>
<div class="summary">
  <div class="stat"><span>correct balanced cosine error</span><strong>{_format_metric(float(correct['balanced_cosine_error']))}</strong></div>
  <div class="stat"><span>history shuffle penalty</span><strong>{_format_metric(float(shuffled.get('paired_balanced_cosine_penalty', float('nan'))))}</strong></div>
  <div class="stat"><span>history reverse penalty</span><strong>{_format_metric(float(reversed_condition.get('paired_balanced_cosine_penalty', float('nan'))))}</strong></div>
  <div class="stat"><span>history replacement penalty</span><strong>{_format_metric(float(replaced.get('paired_balanced_cosine_penalty', float('nan'))))}</strong></div>
  <div class="stat"><span>state reset penalty</span><strong>{_format_metric(float(reset['paired_balanced_cosine_penalty']))}</strong></div>
  <div class="stat"><span>unrelated target penalty</span><strong>{_format_metric(float(unrelated['paired_balanced_cosine_penalty']))}</strong></div>
</div>
<div class="explanation"><b>読み方:</b> penaltyが正なら、履歴内容・順序・state継続・未来対応が
  controlより予測に役立っています。色の違いだけで判断せず、誤差mapと数値を併用してください。</div>
<p class="legend"><span>特徴RGB: EMA targetだけでfitした共通PCA
  （上位3成分の説明率={pca_variance:.1f}%、有効rank={basis.valid_rank}）</span>
  <span><i class="swatch"></i>cosine error 0 → 2（低いほど良い）</span></p>
<h2>collapse診断</h2>
<div class="table-wrap"><table><thead><tr><th>latent</th><th>anchors</th>
<th>raw same-step std</th><th>token-LN same-step std</th><th>token-LN fixed rank</th>
<th>token-LN spatial rank</th><th>token-LN pooled rank</th></tr></thead><tbody>
{_diagnostic_rows(analysis['latent_diagnostics'])}
</tbody></table></div>
<h2>sample {sample_index} の時間変化</h2>
<div class="table-wrap"><table><thead><tr><th>online step</th><th>context t_end μs</th>
<th>correct</th><th>history shuffled</th><th>history reversed</th>
<th>history replaced</th><th>state reset</th><th>unrelated target</th>
</tr></thead><tbody>{_temporal_rows(step_records, sample_index)}</tbody></table></div>
{visual_sections}
</main></body></html>
"""


def write_future_feature_report(
    model: WindowJEPA,
    checkpoint_config: ExperimentConfig,
    materialization: FutureFeatureMaterialization,
    checkpoint: str | Path,
    output: str | Path,
    *,
    device: str | torch.device,
    display_sample_index: int,
    all_steps: bool = False,
    history_shuffle_seed: int = 0,
    history_replacement_seed: int = 0,
    unrelated_seed: int = 0,
) -> dict[str, Any]:
    """Generate a paired qualitative feature report plus machine-readable JSON."""

    output = Path(output)
    checkpoint = Path(checkpoint)
    if output.suffix.lower() != ".html":
        raise ValueError("future feature report output must use an .html suffix")
    if unrelated_seed < 0:
        raise ValueError("unrelated_seed cannot be negative")
    if history_replacement_seed < 0:
        raise ValueError("history_replacement_seed cannot be negative")
    validate_future_feature_config(checkpoint_config)
    _validate_report_compatibility(model, checkpoint_config, materialization)
    requested_device = torch.device(device)
    model_devices = {parameter.device for parameter in model.parameters()}
    if len(model_devices) != 1:
        raise ValueError("model parameters must reside on one device")
    model_device = next(iter(model_devices))
    if model_device.type != requested_device.type or (
        requested_device.index is not None
        and model_device.index != requested_device.index
    ):
        raise ValueError("requested evaluation device differs from the model device")
    output.parent.mkdir(parents=True, exist_ok=True)

    records = extract_future_feature_records(
        model,
        materialization,
        device=model_device,
        history_shuffle_seed=history_shuffle_seed,
        history_replacement_seed=history_replacement_seed,
    )
    replacement_clip_permutation = make_history_replacement_clip_permutation(
        tuple(
            (
                str(clip.sample["sequence_id"]),
                tuple(int(value) for value in clip.sample["t_end_us"]),
            )
            for clip in materialization.clips
        ),
        history_replacement_seed,
    )
    target_features = torch.stack([record.target_tokens for record in records])
    basis = fit_shared_target_pca(target_features)
    analysis, unrelated_permutation = analyze_future_feature_records(
        records,
        materialization.config,
        unrelated_seed=unrelated_seed,
    )
    steps = _step_records(
        records,
        unrelated_permutation,
        active_min_events=materialization.config.future_prediction.active_min_events,
    )
    visual_indices = _visual_record_indices(
        records,
        display_sample_index,
        all_steps,
    )
    visual_sections, visual_records = _render_visual_sections(
        records,
        materialization,
        basis,
        unrelated_permutation,
        output,
        visual_indices,
    )
    document = _report_html(
        checkpoint=checkpoint,
        sample_index=display_sample_index,
        basis=basis,
        analysis=analysis,
        step_records=steps,
        visual_sections=visual_sections,
    )
    payload: dict[str, Any] = {
        "schema": "event-window-jepa-future-feature-visualization-v2",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_config_hash": config_hash(checkpoint_config),
        "effective_manifest": materialization.config.data.manifest,
        "sampling_epoch": materialization.epoch,
        "sample_indices": [clip.sample_index for clip in materialization.clips],
        "sample_set_id": _sample_set_id(records, materialization),
        "clips": _clip_sampling_records(materialization),
        "prediction_horizon_steps": materialization.dataset.lookahead_steps,
        "history_shuffle_seed": history_shuffle_seed,
        "history_replacement_seed": history_replacement_seed,
        "history_replacement_clip_permutation": list(
            replacement_clip_permutation
        ),
        "unrelated_seed": unrelated_seed,
        "unrelated_target_permutation": list(unrelated_permutation),
        "pca": basis.to_record(),
        **analysis,
        "steps": steps,
        "visualized_steps": visual_records,
    }
    payload = _json_safe(payload)
    json_path = output.with_suffix(".json")
    _write_text(
        json_path,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    _write_text(output, document)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize causal future-JEPA latents with shared target PCA and "
            "history controls"
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="override only the checkpoint dataset manifest path",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--calibration-samples", type=int, default=4)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--all-steps", action="store_true")
    parser.add_argument("--history-shuffle-seed", type=int, default=None)
    parser.add_argument(
        "--history-replacement-seed",
        type=int,
        default=0,
        help="seed for the one-to-one donor-clip history mapping",
    )
    parser.add_argument("--unrelated-seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.sample_index < 0:
        raise ValueError("sample-index cannot be negative")
    if args.calibration_samples < 2:
        raise ValueError("calibration-samples must be at least two")
    if args.epoch < 0:
        raise ValueError("epoch cannot be negative")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint_config = load_pretrained_model(
        args.checkpoint,
        device=device,
    )
    sample_indices = range(
        args.sample_index,
        args.sample_index + args.calibration_samples,
    )
    materialization = materialize_future_feature_samples(
        checkpoint_config,
        sample_indices,
        manifest_override=args.manifest,
        epoch=args.epoch,
    )
    output = args.output or args.checkpoint.with_name(
        f"{args.checkpoint.stem}-future-features.html"
    )
    history_seed = (
        checkpoint_config.runtime.seed
        if args.history_shuffle_seed is None
        else args.history_shuffle_seed
    )
    report = write_future_feature_report(
        model,
        checkpoint_config,
        materialization,
        args.checkpoint,
        output,
        device=device,
        display_sample_index=args.sample_index,
        all_steps=args.all_steps,
        history_shuffle_seed=history_seed,
        history_replacement_seed=args.history_replacement_seed,
        unrelated_seed=args.unrelated_seed,
    )
    conditions = report["conditions"]
    shuffled_penalty = conditions.get("history_shuffled", {}).get(
        "paired_balanced_cosine_penalty",
        float("nan"),
    )
    reversed_penalty = conditions.get("history_reversed", {}).get(
        "paired_balanced_cosine_penalty",
        float("nan"),
    )
    replacement_penalty = conditions.get("history_replaced", {}).get(
        "paired_balanced_cosine_penalty",
        float("nan"),
    )
    print(f"[window-jepa] feature visualization: {Path(output).resolve()}")
    print(
        "[window-jepa] balanced cosine "
        f"correct={conditions['correct']['balanced_cosine_error']:.4f}, "
        f"shuffle-penalty={shuffled_penalty:.4f}, "
        f"reverse-penalty={reversed_penalty:.4f}, "
        f"replacement-penalty={replacement_penalty:.4f}, "
        f"reset-penalty="
        f"{conditions['reset']['paired_balanced_cosine_penalty']:.4f}, "
        f"unrelated-penalty="
        f"{conditions['unrelated_target']['paired_balanced_cosine_penalty']:.4f}"
    )


if __name__ == "__main__":
    main()


__all__ = [
    "FutureFeatureStepRecord",
    "SharedPCABasis",
    "analyze_future_feature_records",
    "cosine_error_rgb",
    "extract_future_feature_records",
    "fit_shared_target_pca",
    "fixed_support_latent_diagnostics",
    "make_history_permutation",
    "make_history_replacement_clip_permutation",
    "make_unrelated_clip_permutation",
    "make_unrelated_record_permutation",
    "pca_patch_rgb",
    "project_with_shared_pca",
    "step_conditioned_latent_diagnostics",
    "summarize_prediction_condition",
    "token_cosine_error",
    "write_future_feature_report",
]
