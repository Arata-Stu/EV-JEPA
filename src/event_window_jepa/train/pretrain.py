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
from event_window_jepa.data.anchor_sampler import UniformTimeAnchorSampler, WindowPairSampler
from event_window_jepa.data.event_store import H5EventStore, NpzEventStore
from event_window_jepa.data.paired_window_dataset import PairedWindowDataset
from event_window_jepa.data.spatial_transforms import SharedRandomSpatialTransform
from event_window_jepa.masks.multiblock import MultiBlockMaskGenerator
from event_window_jepa.models.event_vit import EventVisionTransformer
from event_window_jepa.models.scale_embedding import LogFourierScaleEmbedding
from event_window_jepa.models.vjepa21_event_vit import VJEPA21EventVisionTransformer
from event_window_jepa.models.window_jepa import WindowJEPA
from event_window_jepa.models.window_predictor import WindowPredictor
from event_window_jepa.representations.event_image import EventImage
from event_window_jepa.representations.voxel_grid import VoxelGrid
from event_window_jepa.train.callbacks import ema_momentum_at_step, learning_rate_at_step
from event_window_jepa.train.checkpoint import (
    load_training_checkpoint,
    save_checkpoint_atomic,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


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
    if model_config.architecture == "vjepa2_1":
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
    )
    # Keep DDP's expected-gradient set exact for ablations that intentionally
    # remove a component from the objective.
    if not model_config.condition_on_scale:
        model.scale_embedding.requires_grad_(False)
    if config.optimization.objective == "feature_consistency":
        model.predictor.requires_grad_(False)
    return model


def build_dataset(config: ExperimentConfig) -> PairedWindowDataset:
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
    return PairedWindowDataset(
        store=store,
        anchor_sampler=anchor_sampler,
        pair_sampler=pair_sampler,
        representation=representation,
        mask_generator=mask,
        spatial_transform=transform,
        seed=config.runtime.seed,
    )


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _unwrap(model: WindowJEPA | DistributedDataParallel) -> WindowJEPA:
    return model.module if isinstance(model, DistributedDataParallel) else model


def _autocast_context(device: torch.device, precision: str) -> Any:
    if precision == "bf16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


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
    ):
        writer.add_scalar(name, value, step)


def train(
    config: ExperimentConfig,
    resume_override: Path | None = None,
    milestone_epochs: tuple[int, ...] = (),
) -> None:
    world_size, rank, local_rank, device = _distributed_context()
    _seed_everything(config.runtime.seed, rank)
    milestones = tuple(sorted(set(milestone_epochs)))
    if any(epoch <= 0 or epoch > config.optimization.epochs for epoch in milestones):
        raise ValueError("milestone epochs must lie inside the configured training run")
    if rank == 0:
        print(
            f"[window-jepa] validating dataset: {config.data.manifest}",
            flush=True,
        )
    dataset = build_dataset(config)
    sampler = (
        DistributedSampler(dataset, shuffle=True, seed=config.runtime.seed)
        if world_size > 1
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=config.data.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=False,
    )
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
        )

    total_steps = config.optimization.epochs * len(loader)
    warmup_steps = config.optimization.warmup_epochs * len(loader)
    if warmup_steps >= total_steps:
        raise ValueError("warmup duration must be shorter than total training")
    metrics_path = output_dir / "train.jsonl"
    writer = _create_summary_writer(output_dir, global_step) if rank == 0 else None
    if rank == 0:
        print(
            "[window-jepa] training ready: "
            f"device={device}, epochs={config.optimization.epochs}, "
            f"steps_per_epoch={len(loader)}, tensorboard={output_dir / 'tensorboard'}",
            flush=True,
        )

    try:
        for epoch in range(start_epoch, config.optimization.epochs):
            dataset.set_epoch(epoch)
            if sampler is not None:
                sampler.set_epoch(epoch)
            model.train()
            progress = tqdm(
                loader,
                desc=f"epoch {epoch + 1}/{config.optimization.epochs}",
                disable=rank != 0,
                dynamic_ncols=True,
                mininterval=1.0,
                leave=True,
            )
            try:
                for step_in_epoch, raw_batch in enumerate(progress):
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
                        distributed.all_reduce(finite_flag, op=distributed.ReduceOp.MIN)
                    if not bool(finite_flag):
                        raise FloatingPointError(
                            f"non-finite loss at global step {global_step}"
                        )
                    output.loss.backward()
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        (
                            parameter
                            for parameter in model.parameters()
                            if parameter.requires_grad
                        ),
                        config.optimization.gradient_clip,
                        error_if_nonfinite=True,
                    )
                    optimizer.step()
                    momentum = ema_momentum_at_step(
                        global_step,
                        total_steps,
                        config.optimization.target_ema_start,
                        config.optimization.target_ema_end,
                    )
                    _unwrap(model).update_target_encoder(momentum)
                    global_step += 1

                    should_log = (
                        global_step % config.runtime.log_every_steps == 0
                        or step_in_epoch + 1 == len(loader)
                    )
                    if should_log:
                        metric_values = torch.stack(
                            [
                                output.loss.detach().float(),
                                output.masked_loss.detach().float(),
                                output.canonical_loss.detach().float(),
                                output.dense_loss.detach().float(),
                                output.visible_loss.detach().float(),
                                output.deep_supervision_loss.detach().float(),
                                output.prediction_std.detach().float(),
                                output.target_std.detach().float(),
                                gradient_norm.detach().float(),
                                batch["mask_activity_aware"].mean().detach().float(),
                                batch["mask_activity_fallback"].mean().detach().float(),
                                batch["mask_context_active_patch_ratio"]
                                .mean()
                                .detach()
                                .float(),
                                batch["mask_context_event_mass_coverage"]
                                .mean()
                                .detach()
                                .float(),
                                batch["mask_target_active_patch_ratio"]
                                .mean()
                                .detach()
                                .float(),
                                batch["mask_target_event_mass_coverage"]
                                .mean()
                                .detach()
                                .float(),
                                batch["mask_empty_target"].mean().detach().float(),
                            ]
                        )
                        if world_size > 1:
                            distributed.all_reduce(
                                metric_values, op=distributed.ReduceOp.SUM
                            )
                            metric_values /= world_size
                        if rank == 0:
                            record = {
                                "epoch": epoch,
                                "step_in_epoch": step_in_epoch,
                                "global_step": global_step,
                                "loss": float(metric_values[0]),
                                "masked_loss": float(metric_values[1]),
                                "canonical_loss": float(metric_values[2]),
                                "dense_loss": float(metric_values[3]),
                                "visible_loss": float(metric_values[4]),
                                "deep_supervision_loss": float(metric_values[5]),
                                "prediction_std": float(metric_values[6]),
                                "target_std": float(metric_values[7]),
                                "gradient_norm": float(metric_values[8]),
                                "mask_activity_aware_fraction": float(
                                    metric_values[9]
                                ),
                                "mask_activity_fallback_fraction": float(
                                    metric_values[10]
                                ),
                                "mask_context_active_patch_ratio": float(
                                    metric_values[11]
                                ),
                                "mask_context_event_mass_coverage": float(
                                    metric_values[12]
                                ),
                                "mask_target_active_patch_ratio": float(
                                    metric_values[13]
                                ),
                                "mask_target_event_mass_coverage": float(
                                    metric_values[14]
                                ),
                                "mask_empty_target_fraction": float(
                                    metric_values[15]
                                ),
                                "learning_rate": learning_rate,
                                "ema_momentum": momentum,
                            }
                            _append_jsonl(metrics_path, record)
                            if writer is not None:
                                _write_tensorboard_metrics(writer, record)
                            progress.set_postfix(
                                loss=f"{record['loss']:.4f}",
                                pred_std=f"{record['prediction_std']:.3f}",
                                target_std=f"{record['target_std']:.3f}",
                                mask_active=(
                                    f"{record['mask_target_active_patch_ratio']:.2f}"
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
