from __future__ import annotations

import argparse
import json
import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as distributed
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm

from event_window_jepa.config import ExperimentConfig
from event_window_jepa.data.anchor_sampler import (
    UniformTimeAnchorSampler,
    WindowPairSampler,
    milliseconds_to_microseconds,
)
from event_window_jepa.data.event_store import H5EventStore, NpzEventStore
from event_window_jepa.data.paired_window_dataset import PairedWindowDataset
from event_window_jepa.data.recurrent_window_dataset import RecurrentWindowDataset
from event_window_jepa.data.sequence_sampler import (
    MixedRecurrentBatchSampler,
    UniformSequenceClipSampler,
)
from event_window_jepa.data.spatial_transforms import SharedRandomSpatialTransform
from event_window_jepa.masks.multiblock import MultiBlockMaskGenerator
from event_window_jepa.models.event_vit import EventVisionTransformer
from event_window_jepa.models.recurrent_vjepa21_event_vit import (
    RecurrentState,
    RecurrentVJEPA21EventVisionTransformer,
    detach_recurrent_state,
    reset_recurrent_state,
)
from event_window_jepa.models.scale_embedding import LogFourierScaleEmbedding
from event_window_jepa.models.vjepa21_event_vit import VJEPA21EventVisionTransformer
from event_window_jepa.models.window_jepa import WindowJEPA
from event_window_jepa.models.window_predictor import WindowPredictor
from event_window_jepa.representations.event_image import EventImage
from event_window_jepa.representations.voxel_grid import VoxelGrid
from event_window_jepa.train.callbacks import ema_momentum_at_step, learning_rate_at_step
from event_window_jepa.train.checkpoint import (
    collect_rng_states,
    load_training_checkpoint,
    save_checkpoint_atomic,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

OUTPUT_METRIC_NAMES = (
    "loss",
    "masked_loss",
    "canonical_loss",
    "dense_loss",
    "visible_loss",
    "deep_supervision_loss",
    "prediction_std",
    "target_std",
    "future_prediction_loss",
    "active_prediction_loss",
    "inactive_prediction_loss",
    "frame_sigreg_loss",
    "support_sigreg_loss",
    "temporal_sigreg_loss",
    "sigreg_loss",
    "active_patch_fraction",
    "context_active_patch_fraction",
    "frame_sigreg_samples",
    "support_sigreg_samples",
    "temporal_sigreg_samples",
    "frame_sigreg_real_error",
    "frame_sigreg_imaginary_error",
    "support_sigreg_real_error",
    "support_sigreg_imaginary_error",
    "temporal_sigreg_real_error",
    "temporal_sigreg_imaginary_error",
    "active_prediction_sum",
    "active_prediction_count",
    "inactive_prediction_sum",
    "inactive_prediction_count",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain Event Window-JEPA")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--milestone-epochs",
        type=int,
        nargs="*",
        default=(),
        help="Also preserve named checkpoints at these completed epochs",
    )
    return parser.parse_args()


def _distributed_context() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    if world_size > 1 and not distributed.is_initialized():
        distributed.init_process_group(backend=backend)
    return world_size, rank, local_rank, device


def _seed_everything(seed: int, rank: int) -> None:
    effective_seed = seed + rank
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_seed)


def build_model(config: ExperimentConfig) -> WindowJEPA:
    model_config = config.model
    if config.recurrent.enabled:
        encoder = RecurrentVJEPA21EventVisionTransformer(
            image_size=model_config.image_size,
            patch_size=model_config.patch_size,
            input_channels=config.representation.channels,
            embed_dim=model_config.embed_dim,
            depth=model_config.encoder_depth,
            num_heads=model_config.encoder_heads,
            scale_dim=model_config.scale_dim,
            supervision_layers=model_config.deep_supervision_layers,
            recurrent_cell=config.recurrent.cell,
            recurrent_kernel_size=config.recurrent.kernel_size,
            recurrent_placement=config.recurrent.recurrent_placement,
        )
    elif model_config.architecture == "vjepa2_1":
        encoder = VJEPA21EventVisionTransformer(
            image_size=model_config.image_size,
            patch_size=model_config.patch_size,
            input_channels=config.representation.channels,
            embed_dim=model_config.embed_dim,
            depth=model_config.encoder_depth,
            num_heads=model_config.encoder_heads,
            scale_dim=model_config.scale_dim,
            supervision_layers=model_config.deep_supervision_layers,
        )
    else:
        encoder = EventVisionTransformer(
            image_size=model_config.image_size,
            patch_size=model_config.patch_size,
            input_channels=config.representation.channels,
            embed_dim=model_config.embed_dim,
            depth=model_config.encoder_depth,
            num_heads=model_config.encoder_heads,
            scale_dim=model_config.scale_dim,
        )
    predictor = WindowPredictor(
        num_patches=encoder.num_patches,
        encoder_dim=model_config.embed_dim,
        predictor_dim=model_config.predictor_dim,
        depth=model_config.predictor_depth,
        num_heads=model_config.predictor_heads,
        scale_dim=model_config.scale_dim,
    )
    scale_embedding = LogFourierScaleEmbedding(
        output_dim=model_config.scale_dim,
        num_bands=model_config.scale_fourier_bands,
    )
    model = WindowJEPA(
        encoder=encoder,
        predictor=predictor,
        scale_embedding=scale_embedding,
        condition_on_scale=model_config.condition_on_scale,
        variance_weight=config.optimization.variance_weight,
        covariance_weight=config.optimization.covariance_weight,
        canonical_query_weight=config.optimization.canonical_query_weight,
        future_active_min_events=config.future_prediction.active_min_events,
        future_activity_floor=config.future_prediction.activity_floor,
        frame_sigreg_weight=config.future_prediction.frame_sigreg_weight,
        temporal_sigreg_weight=config.future_prediction.temporal_sigreg_weight,
        sigreg_projector_hidden_dim=(
            config.future_prediction.projector_hidden_dim
        ),
        sigreg_projector_output_dim=(
            config.future_prediction.projector_output_dim
        ),
        sigreg_num_slices=config.future_prediction.sigreg_num_slices,
        sigreg_t_max=config.future_prediction.sigreg_t_max,
        sigreg_num_points=config.future_prediction.sigreg_num_points,
        sigreg_projection_seed=config.future_prediction.projection_seed,
    )
    # Keep DDP's expected-gradient set exact for ablations that intentionally
    # remove a component from the objective.
    if not model_config.condition_on_scale:
        model.scale_embedding.requires_grad_(False)
    if config.optimization.objective == "feature_consistency":
        model.predictor.requires_grad_(False)
    return model


def build_dataset(
    config: ExperimentConfig,
) -> PairedWindowDataset | RecurrentWindowDataset:
    store = (
        NpzEventStore(config.data.manifest)
        if config.data.store == "npz"
        else H5EventStore(config.data.manifest)
    )
    sequences = store.sequences(config.data.split)
    if not sequences:
        raise ValueError(f"manifest has no sequences for split {config.data.split!r}")
    crop_height, crop_width = config.data.crop_size
    too_small = [
        info.sequence_id
        for info in sequences
        if info.height < crop_height or info.width < crop_width
    ]
    if too_small:
        raise ValueError(
            "preprocessed resolutions are smaller than data.crop_size: "
            f"{too_small[:5]}"
        )
    if config.representation.kind == "voxel_grid":
        representation = VoxelGrid(
            temporal_bins=config.representation.temporal_bins,
            normalization=config.representation.normalization,
        )
    else:
        representation = EventImage(normalization=config.representation.normalization)
    transform = SharedRandomSpatialTransform(
        crop_size=config.data.crop_size,
        horizontal_flip_probability=config.data.horizontal_flip_probability,
    )
    grid_size = (
        config.model.image_size[0] // config.model.patch_size,
        config.model.image_size[1] // config.model.patch_size,
    )
    mask = MultiBlockMaskGenerator(
        grid_size=grid_size,
        target_blocks=config.mask.target_blocks,
        target_area_range=config.mask.target_area_range,
        target_aspect_range=config.mask.target_aspect_range,
        context_keep_ratio=config.mask.context_keep_ratio,
        activity_aware_probability=config.mask.activity_aware_probability,
        activity_candidates=config.mask.activity_candidates,
        minimum_active_target_ratio=config.mask.minimum_active_target_ratio,
        activity_selection_strategy=config.mask.activity_selection_strategy,
        activity_topk_fraction=config.mask.activity_topk_fraction,
    )
    if config.recurrent.sequence_loader:
        clip_sampler = UniformSequenceClipSampler(
            sequences,
            base_window_ms=config.recurrent.window_ms,
            stride_ms=config.recurrent.stride_ms,
            sequence_length=config.recurrent.sequence_length,
            burn_in_steps=config.recurrent.burn_in_steps,
            lookahead_steps=config.recurrent.prediction_horizon_steps,
            samples_per_epoch=config.data.samples_per_epoch,
            seed=config.runtime.seed,
            sampling_strategy=config.data.sequence_sampling,
        )
        return RecurrentWindowDataset(
            store=store,
            clip_sampler=clip_sampler,
            representation=representation,
            mask_generator=mask,
            spatial_transform=transform,
            tbptt_steps=config.recurrent.tbptt_steps,
            return_patch_event_activity=(
                config.recurrent.return_patch_event_activity
            ),
            seed=config.runtime.seed,
        )

    maximum_window = max(config.windows.train_ms + config.windows.target_ms)
    anchor_sampler = UniformTimeAnchorSampler(
        sequences,
        maximum_window_ms=maximum_window,
        samples_per_epoch=config.data.samples_per_epoch,
        seed=config.runtime.seed,
        sampling_strategy=config.data.sequence_sampling,
    )
    pair_sampler = WindowPairSampler(
        config.windows.train_ms,
        config.windows.target_ms,
        minimum_ratio=config.windows.minimum_ratio,
        direction=config.windows.direction,
        allow_equal=config.windows.allow_equal,
    )
    return PairedWindowDataset(
        store=store,
        anchor_sampler=anchor_sampler,
        pair_sampler=pair_sampler,
        representation=representation,
        mask_generator=mask,
        spatial_transform=transform,
        seed=config.runtime.seed,
    )


def build_recurrent_batch_sampler(
    config: ExperimentConfig,
    dataset: RecurrentWindowDataset,
    *,
    world_size: int,
    rank: int,
) -> MixedRecurrentBatchSampler:
    """Build stable stream lanes, optionally mixed with random clips, per rank."""

    if (
        not config.recurrent.sequence_loader
        or config.recurrent.sampling not in {"stream_reset", "stream", "mixed"}
    ):
        raise ValueError("stream-aware sequence batch sampling is not enabled")
    stream_batch_size = (
        config.data.batch_size
        if config.recurrent.sampling in {"stream_reset", "stream"}
        else round(config.data.batch_size * config.recurrent.stream_ratio)
    )
    random_batch_size = config.data.batch_size - stream_batch_size
    return MixedRecurrentBatchSampler(
        dataset.clip_sampler.sequences,
        base_window_ms=config.recurrent.window_ms,
        stride_ms=config.recurrent.stride_ms,
        sequence_length=config.recurrent.sequence_length,
        burn_in_steps=config.recurrent.burn_in_steps,
        lookahead_steps=config.recurrent.prediction_horizon_steps,
        samples_per_epoch=config.data.samples_per_epoch,
        batch_size=config.data.batch_size,
        stream_ratio=(stream_batch_size, random_batch_size),
        world_size=world_size,
        rank=rank,
        seed=config.runtime.seed,
        random_sampling_strategy=config.data.sequence_sampling,
        force_stream_reset=config.recurrent.sampling == "stream_reset",
    )


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


MixedStreamContract = tuple[tuple[str, str, str, int], ...]


def _validate_mixed_recurrent_batch(
    batch: dict[str, Any],
    *,
    batch_size: int,
    stream_batch_size: int,
    stride_us: int,
    previous_streams: MixedStreamContract | None,
    stream_reset_every_batch: bool = False,
) -> MixedStreamContract:
    """Validate lane stability and causality before a stream-aware GPU batch."""

    if not 0 < stream_batch_size <= batch_size:
        raise ValueError("stream-aware batches require at least one stream row")
    expected_modes = ["stream"] * stream_batch_size + ["random"] * (
        batch_size - stream_batch_size
    )
    modes = batch.get("sampling_mode")
    if not isinstance(modes, (list, tuple)) or list(modes) != expected_modes:
        raise ValueError(
            "stream-aware recurrent rows must be ordered as fixed stream lanes followed "
            "by random clips"
        )

    resets = batch.get("state_reset")
    timestamps = batch.get("t_end_us")
    if (
        not isinstance(resets, torch.Tensor)
        or resets.dtype != torch.bool
        or tuple(resets.shape) != (batch_size,)
    ):
        raise ValueError("stream-aware recurrent state_reset must be a boolean [B] tensor")
    if (
        not isinstance(timestamps, torch.Tensor)
        or timestamps.ndim != 2
        or timestamps.shape[0] != batch_size
        or timestamps.shape[1] == 0
    ):
        raise ValueError("stream-aware recurrent t_end_us must have shape [B,T]")

    sequence_ids = batch.get("sequence_id")
    stream_ids = batch.get("stream_id")
    augmentation_ids = batch.get("augmentation_id")
    for name, values in (
        ("sequence_id", sequence_ids),
        ("stream_id", stream_ids),
        ("augmentation_id", augmentation_ids),
    ):
        if not isinstance(values, (list, tuple)) or len(values) != batch_size:
            raise ValueError(
                f"stream-aware recurrent {name} must contain one value per row"
            )

    if previous_streams is not None and len(previous_streams) != stream_batch_size:
        raise ValueError("previous stream contract has the wrong lane count")
    current_streams: list[tuple[str, str, str, int]] = []
    for row in range(stream_batch_size):
        stream_id = str(stream_ids[row])
        sequence_id = str(sequence_ids[row])
        augmentation_id = str(augmentation_ids[row])
        first_end_us = int(timestamps[row, 0])
        last_end_us = int(timestamps[row, -1])
        reset = bool(resets[row])
        if not stream_id:
            raise ValueError("stream rows require stable non-empty stream_id values")
        if stream_reset_every_batch and not reset:
            raise ValueError("stream_reset rows must reset recurrent state every batch")
        if previous_streams is None:
            if not reset:
                raise ValueError(
                    "the first stream-aware batch must reset every stream lane"
                )
        else:
            (
                previous_stream_id,
                previous_sequence_id,
                previous_augmentation_id,
                previous_last_end_us,
            ) = previous_streams[row]
            if stream_id != previous_stream_id:
                raise ValueError("a stream lane changed position between batches")
            is_adjacent = first_end_us == previous_last_end_us + stride_us
            if not reset and (
                sequence_id != previous_sequence_id
                or augmentation_id != previous_augmentation_id
                or not is_adjacent
            ):
                raise ValueError(
                    "a non-reset stream row changed sequence, augmentation, or causal "
                    "timestamp continuity"
                )
            if (
                reset
                and not stream_reset_every_batch
                and sequence_id == previous_sequence_id
                and is_adjacent
            ):
                raise ValueError("a continuous stream row was reset before its next chunk")
        current_streams.append(
            (stream_id, sequence_id, augmentation_id, last_end_us)
        )

    for row in range(stream_batch_size, batch_size):
        if str(stream_ids[row]):
            raise ValueError("random rows must not carry a stream_id")
        if not bool(resets[row]):
            raise ValueError("random rows must reset recurrent state every batch")
    return tuple(current_streams)


def _unwrap(model: WindowJEPA | DistributedDataParallel) -> WindowJEPA:
    return model.module if isinstance(model, DistributedDataParallel) else model


def _precision_support_error(device: torch.device, precision: str) -> str | None:
    if precision == "fp32":
        return None
    if device.type != "cuda":
        return f"precision={precision} requires a CUDA device"
    if precision == "fp16":
        return None
    if precision != "bf16":
        return f"unsupported precision: {precision}"
    properties = torch.cuda.get_device_properties(device)
    if properties.major < 8:
        return (
            "precision=bf16 requires native Ampere-or-newer support; "
            f"found {properties.name} capability={properties.major}.{properties.minor}"
        )
    checker = getattr(torch.cuda, "is_bf16_supported", None)
    if checker is None:
        return "precision=bf16 is unsupported by this PyTorch build"
    try:
        supported = bool(checker(including_emulation=False))
    except TypeError:
        supported = bool(checker())
    if not supported:
        return "precision=bf16 is not natively supported by this CUDA/PyTorch device"
    return None


def _validate_precision_support(
    device: torch.device, precision: str, world_size: int
) -> None:
    local_error = _precision_support_error(device, precision)
    errors: list[str | None] = [local_error]
    if world_size > 1:
        errors = [None] * world_size
        distributed.all_gather_object(errors, local_error)
    failures = [
        f"rank {rank}: {error}"
        for rank, error in enumerate(errors)
        if error is not None
    ]
    if failures:
        raise RuntimeError("; ".join(failures))


def _make_grad_scaler(device: torch.device, precision: str) -> Any:
    enabled = device.type == "cuda" and precision == "fp16"
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        # torch>=2.2 compatibility; newer releases prefer torch.amp.GradScaler.
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast_context(device: torch.device, precision: str) -> Any:
    if device.type == "cuda" and precision in {"fp16", "bf16"}:
        dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def _backward(loss: torch.Tensor, grad_scaler: Any | None) -> None:
    if grad_scaler is None:
        loss.backward()
    else:
        grad_scaler.scale(loss).backward()


def _step_optimizer(
    *,
    model: WindowJEPA | DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    grad_scaler: Any,
    precision: str,
    gradient_clip: float,
    world_size: int,
) -> tuple[torch.Tensor, bool]:
    """Unscale once, clip, and synchronously step or skip every DDP rank."""

    grad_scaler.unscale_(optimizer)
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        gradient_clip,
        error_if_nonfinite=(precision != "fp16"),
    )
    gradient_finite = torch.isfinite(gradient_norm).to(dtype=torch.int32)
    if world_size > 1:
        distributed.all_reduce(gradient_finite, op=distributed.ReduceOp.MIN)
    optimizer_step_skipped = not bool(gradient_finite)
    if optimizer_step_skipped and precision != "fp16":
        raise FloatingPointError("non-finite gradient")
    # Enabled GradScaler skips optimizer.step and resets its growth tracker when
    # unscale_ observed a non-finite gradient. DDP-reduced gradients make that
    # decision identical on every rank; the explicit flag controls EMA/step logs.
    grad_scaler.step(optimizer)
    grad_scaler.update()
    return gradient_norm, optimizer_step_skipped


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _project_output_path(configured_path: str) -> Path:
    path = Path(configured_path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _create_summary_writer(output_dir: Path, global_step: int) -> Any:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as error:
        raise ImportError(
            "TensorBoard logging requires the tensorboard package; reinstall the "
            "project with `python -m pip install -e '.[hdf5]'`"
        ) from error
    options: dict[str, Any] = {
        "log_dir": str(output_dir / "tensorboard"),
        "max_queue": 20,
        "flush_secs": 30,
    }
    if global_step > 0:
        options["purge_step"] = global_step
    return SummaryWriter(**options)


def _write_tensorboard_metrics(writer: Any, record: dict[str, Any]) -> None:
    step = int(record["global_step"])
    # Keep the dashboard intentionally small: the objective components, two
    # collapse diagnostics, and the learning-rate schedule are sufficient for
    # routine monitoring. Full diagnostics remain available in train.jsonl.
    for name, value in (
        ("loss/total", record["loss"]),
        ("loss/masked", record["masked_loss"]),
        ("loss/dense", record["dense_loss"]),
        ("loss/visible", record["visible_loss"]),
        ("loss/deep_supervision", record["deep_supervision_loss"]),
        ("loss/canonical", record["canonical_loss"]),
        ("representation/prediction_std", record["prediction_std"]),
        ("representation/target_std", record["target_std"]),
        ("future/prediction_loss", record["future_prediction_loss"]),
        ("future/active_prediction_loss", record["active_prediction_loss"]),
        ("future/inactive_prediction_loss", record["inactive_prediction_loss"]),
        ("future/frame_sigreg_loss", record["frame_sigreg_loss"]),
        ("future/support_sigreg_loss", record["support_sigreg_loss"]),
        ("future/temporal_sigreg_loss", record["temporal_sigreg_loss"]),
        ("future/weighted_sigreg_loss", record["sigreg_loss"]),
        (
            "future/sigreg_to_prediction_ratio",
            record["sigreg_to_prediction_ratio"],
        ),
        ("future/active_patch_fraction", record["active_patch_fraction"]),
        (
            "future/context_active_patch_fraction",
            record["context_active_patch_fraction"],
        ),
        ("future/frame_sigreg_samples", record["frame_sigreg_samples"]),
        ("future/support_sigreg_samples", record["support_sigreg_samples"]),
        ("future/temporal_sigreg_samples", record["temporal_sigreg_samples"]),
        ("future/frame_sigreg_real", record["frame_sigreg_real_error"]),
        (
            "future/frame_sigreg_imaginary",
            record["frame_sigreg_imaginary_error"],
        ),
        ("future/support_sigreg_real", record["support_sigreg_real_error"]),
        (
            "future/support_sigreg_imaginary",
            record["support_sigreg_imaginary_error"],
        ),
        ("future/temporal_sigreg_real", record["temporal_sigreg_real_error"]),
        (
            "future/temporal_sigreg_imaginary",
            record["temporal_sigreg_imaginary_error"],
        ),
        ("mask/activity_aware_fraction", record["mask_activity_aware_fraction"]),
        ("mask/activity_fallback_fraction", record["mask_activity_fallback_fraction"]),
        ("mask/context_active_patch_ratio", record["mask_context_active_patch_ratio"]),
        (
            "mask/context_event_mass_coverage",
            record["mask_context_event_mass_coverage"],
        ),
        ("mask/target_active_patch_ratio", record["mask_target_active_patch_ratio"]),
        (
            "mask/target_event_mass_coverage",
            record["mask_target_event_mass_coverage"],
        ),
        ("mask/empty_target_fraction", record["mask_empty_target_fraction"]),
        ("optimization/learning_rate", record["learning_rate"]),
        ("optimization/loss_scale", record["loss_scale"]),
        (
            "optimization/optimizer_step_skipped",
            float(record["optimizer_step_skipped"]),
        ),
    ):
        writer.add_scalar(name, value, step)
    if "recurrent_state_rms" in record:
        writer.add_scalar(
            "recurrent/state_rms", record["recurrent_state_rms"], step
        )


def _output_metric_tensor(output: Any) -> torch.Tensor:
    values: list[torch.Tensor] = []
    for name in OUTPUT_METRIC_NAMES:
        value = getattr(output, name, None)
        if value is None:
            value = output.loss.new_zeros(())
        values.append(value.detach().float())
    return torch.stack(values)


def _recurrent_chunk_ranges(
    loss_mask: torch.Tensor, detach_mask: torch.Tensor
) -> tuple[tuple[int, int], ...]:
    """Validate collated temporal control masks and return loss chunk ranges."""

    if (
        loss_mask.ndim != 2
        or detach_mask.shape != loss_mask.shape
        or loss_mask.dtype is not torch.bool
        or detach_mask.dtype is not torch.bool
    ):
        raise ValueError("recurrent loss/detach masks must be boolean [B,T] tensors")
    if not bool((loss_mask == loss_mask[:1]).all()) or not bool(
        (detach_mask == detach_mask[:1]).all()
    ):
        raise ValueError("all clips in a batch must share BPTT control masks")
    supervised = torch.nonzero(loss_mask[0], as_tuple=False).flatten().tolist()
    if not supervised:
        raise ValueError("recurrent clip has no loss-bearing timesteps")
    expected = list(range(supervised[0], supervised[-1] + 1))
    if supervised != expected:
        raise ValueError("loss-bearing recurrent timesteps must form one suffix")
    if bool(loss_mask[0, : supervised[0]].any()) or not bool(
        loss_mask[0, supervised[0] :].all()
    ):
        raise ValueError("burn-in must be a prefix followed by supervised timesteps")
    if not bool(detach_mask[0, supervised[0]]):
        raise ValueError("recurrent state must detach at the burn-in boundary")

    boundaries = [
        index
        for index in supervised
        if index == supervised[0] or bool(detach_mask[0, index])
    ]
    ranges: list[tuple[int, int]] = []
    for offset, start in enumerate(boundaries):
        end = boundaries[offset + 1] if offset + 1 < len(boundaries) else supervised[-1] + 1
        if end <= start:
            raise ValueError("TBPTT boundaries must be strictly increasing")
        ranges.append((start, end))
    return tuple(ranges)


def _recurrent_state_rms(state: Any) -> torch.Tensor:
    values = state if isinstance(state, tuple) else (state,)
    tensors = [value.detach().float() for value in values if isinstance(value, torch.Tensor)]
    if not tensors:
        raise RuntimeError("recurrent encoder did not return a tensor state")
    mean_square = torch.stack([value.square().mean() for value in tensors]).mean()
    return mean_square.sqrt()


def _recurrent_backward(
    *,
    model: WindowJEPA | DistributedDataParallel,
    core_model: WindowJEPA,
    batch: dict[str, Any],
    config: ExperimentConfig,
    device: torch.device,
    world_size: int,
    initial_state: RecurrentState | None = None,
    grad_scaler: Any | None = None,
) -> tuple[torch.Tensor, torch.Tensor, RecurrentState]:
    """Run BPTT inside a batch and return state for optional cross-batch TBPTT."""

    ranges = _recurrent_chunk_ranges(batch["loss_mask"], batch["detach_mask"])
    future_objective = config.optimization.objective == "recurrent_future_jepa"
    if future_objective:
        required_future = {
            "x_future",
            "future_dt_ms",
            "future_t_end_us",
            "patch_event_activity",
            "future_patch_event_activity",
        }
        missing = sorted(required_future - batch.keys())
        if missing:
            raise ValueError(
                f"future recurrent batch is missing aligned fields: {missing}"
            )
        timestamps = batch.get("t_end_us")
        future_timestamps = batch["future_t_end_us"]
        expected_delta_us = (
            config.recurrent.prediction_horizon_steps
            * milliseconds_to_microseconds(config.recurrent.stride_ms)
        )
        if (
            not isinstance(timestamps, torch.Tensor)
            or not isinstance(future_timestamps, torch.Tensor)
            or future_timestamps.shape != timestamps.shape
            or not bool(
                (future_timestamps - timestamps == expected_delta_us).all()
            )
        ):
            raise ValueError(
                "future_t_end_us must be prediction_horizon_steps strides "
                "after t_end_us"
            )
    first_supervised = ranges[0][0]
    state = detach_recurrent_state(initial_state)
    if first_supervised:
        with _autocast_context(device, config.optimization.precision):
            state = core_model.recurrent_burn_in(
                x=batch["x"][:, :first_supervised],
                duration_ms=batch["dt_ms"][:, :first_supervised],
                context_mask=(
                    None
                    if future_objective
                    else batch["context_mask"][:, :first_supervised]
                ),
                online_state=state,
            )
    total_supervised = sum(end - start for start, end in ranges)
    if total_supervised != config.recurrent.sequence_length:
        raise ValueError(
            "dataset supervised length differs from recurrent.sequence_length"
        )

    accumulated_metrics = torch.zeros(
        len(OUTPUT_METRIC_NAMES), device=device, dtype=torch.float32
    )
    for chunk_index, (start, end) in enumerate(ranges):
        state = detach_recurrent_state(state)
        is_final_chunk = chunk_index + 1 == len(ranges)
        synchronization = (
            model.no_sync()
            if isinstance(model, DistributedDataParallel) and not is_final_chunk
            else nullcontext()
        )
        with synchronization:
            with _autocast_context(device, config.optimization.precision):
                output = model(
                    x_context=batch["x"][:, start:end],
                    x_target=(
                        batch["x_future"][:, start:end]
                        if future_objective
                        else batch["x"][:, start:end]
                    ),
                    dt_context_ms=batch["dt_ms"][:, start:end],
                    dt_target_ms=(
                        batch["future_dt_ms"][:, start:end]
                        if future_objective
                        else batch["dt_ms"][:, start:end]
                    ),
                    context_mask=batch["context_mask"][:, start:end],
                    target_mask=batch["target_mask"][:, start:end],
                    objective=config.optimization.objective,
                    online_state=state,
                    context_event_activity=(
                        batch["patch_event_activity"][:, start:end]
                        if future_objective
                        else None
                    ),
                    target_event_activity=(
                        batch["future_patch_event_activity"][:, start:end]
                        if future_objective
                        else None
                    ),
                )
            finite_flag = torch.isfinite(output.loss).to(dtype=torch.int32)
            if world_size > 1:
                distributed.all_reduce(finite_flag, op=distributed.ReduceOp.MIN)
            if not bool(finite_flag):
                raise FloatingPointError("non-finite loss in recurrent BPTT chunk")
            chunk_steps = end - start
            weight = chunk_steps / total_supervised
            _backward(output.loss * weight, grad_scaler)
        state = output.online_state
        accumulated_metrics += _output_metric_tensor(output) * chunk_steps

    if state is None:
        raise RuntimeError("recurrent training did not produce online state")
    state_rms = _recurrent_state_rms(state)
    state_finite = torch.isfinite(state_rms).to(dtype=torch.int32)
    if world_size > 1:
        distributed.all_reduce(state_finite, op=distributed.ReduceOp.MIN)
    if not bool(state_finite):
        raise FloatingPointError("non-finite recurrent state after TBPTT")
    return (
        accumulated_metrics / total_supervised,
        state_rms,
        state,
    )


def _feedforward_sequence_backward(
    *,
    model: WindowJEPA | DistributedDataParallel,
    batch: dict[str, Any],
    config: ExperimentConfig,
    device: torch.device,
    world_size: int,
    grad_scaler: Any | None = None,
) -> torch.Tensor:
    """Train independent clip frames with one DDP forward/backward operation."""

    future_objective = config.optimization.objective == "frame_future_jepa"
    loss_mask = batch.get("loss_mask")
    if (
        not isinstance(loss_mask, torch.Tensor)
        or loss_mask.dtype != torch.bool
        or loss_mask.ndim != 2
    ):
        raise ValueError("feedforward sequence batches require loss_mask [B,T]")
    supervised_counts = loss_mask.sum(dim=1)
    if not bool((supervised_counts == config.recurrent.sequence_length).all()):
        raise ValueError(
            "dataset supervised length differs from recurrent.sequence_length"
        )
    if future_objective:
        required_future = {
            "x_future",
            "future_dt_ms",
            "future_t_end_us",
            "patch_event_activity",
            "future_patch_event_activity",
        }
        missing = sorted(required_future - batch.keys())
        if missing:
            raise ValueError(
                f"frame future batch is missing aligned fields: {missing}"
            )
        timestamps = batch.get("t_end_us")
        future_timestamps = batch["future_t_end_us"]
        expected_delta_us = (
            config.recurrent.prediction_horizon_steps
            * milliseconds_to_microseconds(config.recurrent.stride_ms)
        )
        if (
            not isinstance(timestamps, torch.Tensor)
            or not isinstance(future_timestamps, torch.Tensor)
            or future_timestamps.shape != timestamps.shape
            or not bool(
                (future_timestamps - timestamps == expected_delta_us).all()
            )
        ):
            raise ValueError(
                "future_t_end_us must be prediction_horizon_steps strides "
                "after t_end_us"
            )
    with _autocast_context(device, config.optimization.precision):
        output = model(
            x_context=batch["x"],
            x_target=(batch["x_future"] if future_objective else batch["x"]),
            dt_context_ms=batch["dt_ms"],
            dt_target_ms=(
                batch["future_dt_ms"] if future_objective else batch["dt_ms"]
            ),
            context_mask=batch["context_mask"],
            target_mask=batch["target_mask"],
            objective=config.optimization.objective,
            sequence_loss_mask=loss_mask,
            context_event_activity=(
                batch["patch_event_activity"] if future_objective else None
            ),
            target_event_activity=(
                batch["future_patch_event_activity"]
                if future_objective
                else None
            ),
        )
    finite_flag = torch.isfinite(output.loss).to(dtype=torch.int32)
    if world_size > 1:
        distributed.all_reduce(finite_flag, op=distributed.ReduceOp.MIN)
    if not bool(finite_flag):
        raise FloatingPointError("non-finite loss in feedforward sequence batch")
    if output.prediction_sequence is None or output.target_sequence is None:
        raise RuntimeError("feedforward sequence objective did not retain latent sequences")
    _backward(output.loss, grad_scaler)
    return _output_metric_tensor(output)


def train(
    config: ExperimentConfig,
    resume_override: Path | None = None,
    milestone_epochs: tuple[int, ...] = (),
) -> None:
    world_size, rank, local_rank, device = _distributed_context()
    _seed_everything(config.runtime.seed, rank)
    _validate_precision_support(device, config.optimization.precision, world_size)
    milestones = tuple(sorted(set(milestone_epochs)))
    if any(epoch <= 0 or epoch > config.optimization.epochs for epoch in milestones):
        raise ValueError("milestone epochs must lie inside the configured training run")
    if rank == 0:
        print(
            f"[window-jepa] validating dataset: {config.data.manifest}",
            flush=True,
        )
    dataset = build_dataset(config)
    mixed_batch_sampler: MixedRecurrentBatchSampler | None = None
    sampler: DistributedSampler | None = None
    loader_options: dict[str, Any] = {
        "dataset": dataset,
        "num_workers": config.data.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": False,
    }
    if (
        config.recurrent.sequence_loader
        and config.recurrent.sampling in {"stream_reset", "stream", "mixed"}
        and isinstance(dataset, RecurrentWindowDataset)
    ):
        mixed_batch_sampler = build_recurrent_batch_sampler(
            config,
            dataset,
            world_size=world_size,
            rank=rank,
        )
        loader_options["batch_sampler"] = mixed_batch_sampler
    else:
        sampler = (
            DistributedSampler(dataset, shuffle=True, seed=config.runtime.seed)
            if world_size > 1
            else None
        )
        loader_options.update(
            {
                "batch_size": config.data.batch_size,
                "sampler": sampler,
                "shuffle": sampler is None,
                "drop_last": True,
            }
        )
    loader = DataLoader(**loader_options)
    if not loader:
        raise ValueError("data loader is empty; lower batch_size or increase samples_per_epoch")

    core_model = build_model(config).to(device)
    model: WindowJEPA | DistributedDataParallel
    if world_size > 1:
        model = DistributedDataParallel(
            core_model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            broadcast_buffers=False,
        )
    else:
        model = core_model
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
    )
    grad_scaler = _make_grad_scaler(device, config.optimization.precision)

    output_dir = _project_output_path(config.runtime.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
    if world_size > 1:
        distributed.barrier()

    start_epoch = 0
    global_step = 0
    resume_path = resume_override or (
        Path(config.runtime.resume) if config.runtime.resume is not None else None
    )
    if resume_path is not None:
        start_epoch, global_step = load_training_checkpoint(
            resume_path,
            core_model,
            optimizer,
            config,
            device,
            world_size=world_size,
            steps_per_epoch=len(loader),
            rank=rank,
            grad_scaler=grad_scaler,
        )
        completed_attempts = start_epoch * len(loader)
        if not 0 <= global_step <= completed_attempts:
            raise ValueError(
                "checkpoint optimizer-update count is incompatible with its epoch"
            )

    total_steps = config.optimization.epochs * len(loader)
    warmup_steps = config.optimization.warmup_epochs * len(loader)
    if warmup_steps >= total_steps:
        raise ValueError("warmup duration must be shorter than total training")
    metrics_path = output_dir / "train.jsonl"
    writer = _create_summary_writer(output_dir, global_step) if rank == 0 else None
    if rank == 0:
        trainable_parameters = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        print(
            "[window-jepa] training ready: "
            f"device={device}, precision={config.optimization.precision}, "
            f"epochs={config.optimization.epochs}, "
            f"steps_per_epoch={len(loader)}, trainable_parameters={trainable_parameters:,}, "
            f"tensorboard={output_dir / 'tensorboard'}",
            flush=True,
        )
        if config.recurrent.sequence_loader:
            chunks = (
                config.recurrent.sequence_length + config.recurrent.tbptt_steps - 1
            ) // config.recurrent.tbptt_steps
            effective_clips = len(loader) * config.data.batch_size * world_size
            processed_frames = (
                effective_clips * config.recurrent.sequence_length
            )
            input_frames = effective_clips * (
                config.recurrent.burn_in_steps
                + config.recurrent.sequence_length
                + config.recurrent.prediction_horizon_steps
            )
            mixture = ""
            if mixed_batch_sampler is not None:
                mixture = (
                    f", stream_per_rank={mixed_batch_sampler.stream_batch_size}, "
                    f"random_per_rank={mixed_batch_sampler.random_batch_size}"
                )
            elif config.recurrent.sampling in {"clip", "random"}:
                mixture = (
                    f", stream_per_rank=0, random_per_rank={config.data.batch_size}"
                )
            temporal_execution = (
                f"tbptt_chunks_per_update={chunks}, "
                if config.recurrent.enabled
                else "independent_frame_forward=true, "
            )
            cross_batch_state = (
                "not_applicable"
                if not config.recurrent.enabled
                else (
                    "carry_detached_stream_rows"
                    if config.recurrent.sampling in {"stream", "mixed"}
                    else "reset_all_rows"
                )
            )
            print(
                "[window-jepa] temporal sequence: "
                f"model={config.recurrent.temporal_model}, "
                f"sampling={config.recurrent.sampling}{mixture}, "
                f"window={config.recurrent.window_ms:g}ms, "
                f"burn_in={config.recurrent.burn_in_steps}, "
                f"supervised_steps={config.recurrent.sequence_length}, "
                f"future_horizon={config.recurrent.prediction_horizon_steps}, "
                f"{temporal_execution}"
                f"cross_batch_state={cross_batch_state}, "
                f"effective_clips_per_epoch={effective_clips}, "
                f"input_frames_per_epoch={input_frames}, "
                f"supervised_frames_per_epoch={processed_frames}",
                flush=True,
            )

    try:
        for epoch in range(start_epoch, config.optimization.epochs):
            dataset.set_epoch(epoch)
            if sampler is not None:
                sampler.set_epoch(epoch)
            if mixed_batch_sampler is not None:
                mixed_batch_sampler.set_epoch(epoch)
            model.train()
            progress = tqdm(
                loader,
                desc=f"epoch {epoch + 1}/{config.optimization.epochs}",
                disable=rank != 0,
                dynamic_ncols=True,
                mininterval=1.0,
                leave=True,
            )
            carried_recurrent_state: RecurrentState | None = None
            previous_stream_contract: MixedStreamContract | None = None
            try:
                for step_in_epoch, raw_batch in enumerate(progress):
                    attempt_step = epoch * len(loader) + step_in_epoch + 1
                    if mixed_batch_sampler is not None:
                        previous_stream_contract = _validate_mixed_recurrent_batch(
                            raw_batch,
                            batch_size=config.data.batch_size,
                            stream_batch_size=mixed_batch_sampler.stream_batch_size,
                            stride_us=mixed_batch_sampler.stride_us,
                            previous_streams=previous_stream_contract,
                            stream_reset_every_batch=(
                                config.recurrent.sampling == "stream_reset"
                            ),
                        )
                    batch = _move_batch(raw_batch, device)
                    learning_rate = learning_rate_at_step(
                        global_step,
                        total_steps,
                        warmup_steps,
                        config.optimization.learning_rate,
                        config.optimization.minimum_learning_rate,
                    )
                    for group in optimizer.param_groups:
                        group["lr"] = learning_rate
                    optimizer.zero_grad(set_to_none=True)
                    recurrent_state_rms = torch.zeros((), device=device)
                    if config.recurrent.enabled:
                        initial_state = None
                        if config.recurrent.sampling in {"stream", "mixed"}:
                            reset_mask = batch.get("state_reset")
                            if not isinstance(reset_mask, torch.Tensor):
                                raise ValueError(
                                    "stream-aware recurrent batches require state_reset [B]"
                                )
                            if carried_recurrent_state is None and not bool(
                                reset_mask.all()
                            ):
                                raise ValueError(
                                    "the first stream-aware batch of an epoch must reset every "
                                    "stream and random lane"
                                )
                            initial_state = reset_recurrent_state(
                                carried_recurrent_state,
                                reset_mask,
                            )
                        (
                            output_metrics,
                            recurrent_state_rms,
                            final_recurrent_state,
                        ) = _recurrent_backward(
                            model=model,
                            core_model=core_model,
                            batch=batch,
                            config=config,
                            device=device,
                            world_size=world_size,
                            initial_state=initial_state,
                            grad_scaler=grad_scaler,
                        )
                        carried_recurrent_state = (
                            detach_recurrent_state(final_recurrent_state)
                            if config.recurrent.sampling in {"stream", "mixed"}
                            else None
                        )
                    elif config.recurrent.sequence_loader:
                        output_metrics = _feedforward_sequence_backward(
                            model=model,
                            batch=batch,
                            config=config,
                            device=device,
                            world_size=world_size,
                            grad_scaler=grad_scaler,
                        )
                    else:
                        with _autocast_context(device, config.optimization.precision):
                            output = model(
                                x_context=batch["x_context"],
                                x_target=batch["x_target"],
                                dt_context_ms=batch["dt_context_ms"],
                                dt_target_ms=batch["dt_target_ms"],
                                context_mask=batch["context_mask"],
                                target_mask=batch["target_mask"],
                                objective=config.optimization.objective,
                            )
                        finite_flag = torch.isfinite(output.loss).to(dtype=torch.int32)
                        if world_size > 1:
                            distributed.all_reduce(
                                finite_flag, op=distributed.ReduceOp.MIN
                            )
                        if not bool(finite_flag):
                            raise FloatingPointError(
                                f"non-finite loss at global step {global_step}"
                            )
                        _backward(output.loss, grad_scaler)
                        output_metrics = _output_metric_tensor(output)
                    gradient_norm, optimizer_step_skipped = _step_optimizer(
                        model=model,
                        optimizer=optimizer,
                        grad_scaler=grad_scaler,
                        precision=config.optimization.precision,
                        gradient_clip=config.optimization.gradient_clip,
                        world_size=world_size,
                    )
                    momentum = ema_momentum_at_step(
                        global_step,
                        total_steps,
                        config.optimization.target_ema_start,
                        config.optimization.target_ema_end,
                    )
                    if not optimizer_step_skipped:
                        _unwrap(model).update_target_encoder(momentum)
                        global_step += 1

                    should_log = (
                        global_step % config.runtime.log_every_steps == 0
                        or step_in_epoch + 1 == len(loader)
                        or optimizer_step_skipped
                    )
                    if should_log:
                        temporal_selection = (
                            batch["loss_mask"]
                            if config.recurrent.sequence_loader
                            else None
                        )

                        def mask_mean(name: str) -> torch.Tensor:
                            values = batch[name]
                            if temporal_selection is not None:
                                values = values.masked_select(temporal_selection)
                            return values.mean().detach().float()

                        mask_metrics = torch.stack(
                            [
                                mask_mean("mask_activity_aware"),
                                mask_mean("mask_activity_fallback"),
                                mask_mean("mask_context_active_patch_ratio"),
                                mask_mean("mask_context_event_mass_coverage"),
                                mask_mean("mask_target_active_patch_ratio"),
                                mask_mean("mask_target_event_mass_coverage"),
                                mask_mean("mask_empty_target"),
                            ]
                        )
                        metric_values = torch.cat(
                            (
                                output_metrics,
                                gradient_norm.detach().float().reshape(1),
                                mask_metrics,
                                recurrent_state_rms.detach().float().reshape(1),
                            )
                        )
                        if world_size > 1:
                            distributed.all_reduce(
                                metric_values, op=distributed.ReduceOp.SUM
                            )
                            metric_values /= world_size
                        if rank == 0:
                            output_count = len(OUTPUT_METRIC_NAMES)
                            gradient_index = output_count
                            mask_start = gradient_index + 1
                            recurrent_index = mask_start + 7
                            record = {
                                "epoch": epoch,
                                "step_in_epoch": step_in_epoch,
                                "attempt_step": attempt_step,
                                "global_step": global_step,
                                **{
                                    name: float(metric_values[index])
                                    for index, name in enumerate(OUTPUT_METRIC_NAMES)
                                },
                                "gradient_norm": float(
                                    metric_values[gradient_index]
                                ),
                                "mask_activity_aware_fraction": float(
                                    metric_values[mask_start]
                                ),
                                "mask_activity_fallback_fraction": float(
                                    metric_values[mask_start + 1]
                                ),
                                "mask_context_active_patch_ratio": float(
                                    metric_values[mask_start + 2]
                                ),
                                "mask_context_event_mass_coverage": float(
                                    metric_values[mask_start + 3]
                                ),
                                "mask_target_active_patch_ratio": float(
                                    metric_values[mask_start + 4]
                                ),
                                "mask_target_event_mass_coverage": float(
                                    metric_values[mask_start + 5]
                                ),
                                "mask_empty_target_fraction": float(
                                    metric_values[mask_start + 6]
                                ),
                                "learning_rate": learning_rate,
                                "ema_momentum": momentum,
                                "loss_scale": float(grad_scaler.get_scale()),
                                "optimizer_step_skipped": optimizer_step_skipped,
                            }
                            if config.recurrent.enabled:
                                record["recurrent_state_rms"] = float(
                                    metric_values[recurrent_index]
                                )
                            if record["active_prediction_count"] > 0:
                                record["active_prediction_loss"] = record[
                                    "active_prediction_sum"
                                ] / record["active_prediction_count"]
                            if record["inactive_prediction_count"] > 0:
                                record["inactive_prediction_loss"] = record[
                                    "inactive_prediction_sum"
                                ] / record["inactive_prediction_count"]
                            record["sigreg_to_prediction_ratio"] = record[
                                "sigreg_loss"
                            ] / max(record["future_prediction_loss"], 1e-12)
                            _append_jsonl(metrics_path, record)
                            if writer is not None:
                                _write_tensorboard_metrics(writer, record)
                            progress.set_postfix(
                                loss=f"{record['loss']:.4f}",
                                pred_std=f"{record['prediction_std']:.3f}",
                                target_std=f"{record['target_std']:.3f}",
                                active=(
                                    f"{record['active_patch_fraction']:.2f}"
                                    if config.optimization.objective
                                    in {"frame_future_jepa", "recurrent_future_jepa"}
                                    else f"{record['mask_target_active_patch_ratio']:.2f}"
                                ),
                                lr=f"{record['learning_rate']:.2e}",
                                refresh=False,
                            )
            finally:
                progress.close()

            if writer is not None:
                writer.flush()
            should_checkpoint = (
                (epoch + 1) % config.runtime.checkpoint_every_epochs == 0
                or epoch + 1 == config.optimization.epochs
                or epoch + 1 in milestones
            )
            if should_checkpoint:
                rng_states = collect_rng_states(world_size)
                if rank == 0:
                    save_checkpoint_atomic(
                        output_dir / "checkpoint-latest.pt",
                        _unwrap(model),
                        optimizer,
                        config,
                        epoch=epoch + 1,
                        global_step=global_step,
                        world_size=world_size,
                        steps_per_epoch=len(loader),
                        rng_states=rng_states,
                        grad_scaler=grad_scaler,
                    )
                    if epoch + 1 in milestones:
                        save_checkpoint_atomic(
                            output_dir / f"checkpoint-epoch{epoch + 1:04d}.pt",
                            _unwrap(model),
                            optimizer,
                            config,
                            epoch=epoch + 1,
                            global_step=global_step,
                            world_size=world_size,
                            steps_per_epoch=len(loader),
                            rng_states=rng_states,
                            grad_scaler=grad_scaler,
                        )
                if world_size > 1:
                    distributed.barrier()
    finally:
        if writer is not None:
            writer.close()
    if world_size > 1:
        distributed.barrier()
        distributed.destroy_process_group()


def main() -> None:
    args = _parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    train(
        config,
        resume_override=args.resume,
        milestone_epochs=tuple(args.milestone_epochs),
    )


if __name__ == "__main__":
    main()
