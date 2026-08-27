"""Three-GPU RVT-style ConvLSTM scratch sanity training for Gen1 detection.

The checkpoint passed to this entry point supplies only the resolved input and
model architecture.  Its learned weights are deliberately not loaded.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler
from tqdm.auto import tqdm

from event_window_jepa.config import ExperimentConfig
from event_window_jepa.downstream.gen1_detection import (
    Gen1StreamDetectionDataset,
    MixedStatefulBatchSampler,
    RandomLabelClipBatchSampler,
    StreamLaneBatchSampler,
    StreamStateManager,
    WindowJEPAYOLOX,
    _collate_stream_sequence_detection,
    _evaluate_stateful,
    _frame_references,
    _load_rvt_components,
    _manifest_identity,
    _read_label_sources,
    _representation,
    _stateful_sampling_counts,
    _stream_references,
    _training_stream_references,
    _validate_stateful_window_duration,
    _forward_stateful_sequence,
)
from event_window_jepa.models.recurrent_vjepa21_event_vit import (
    RecurrentVJEPA21EventVisionTransformer,
)
from event_window_jepa.train.checkpoint import SCHEMA_VERSION, config_hash
from event_window_jepa.train.pretrain import build_model


SCRATCH_SCHEMA = "event-window-jepa-gen1-convlstm-scratch-ddp-v1"


class _FeatureBackbone(nn.Module):
    """Keep only modules used by detection, excluding teacher and predictor."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.online_encoder = source.online_encoder
        self.scale_embedding = source.scale_embedding
        self.condition_on_scale = bool(source.condition_on_scale)


class _StatefulScratchTrainer(nn.Module):
    """Make one complete BxT chunk one top-level DDP forward."""

    def __init__(self, detector: WindowJEPAYOLOX) -> None:
        super().__init__()
        self.detector = detector

    def forward(  # type: ignore[override]
        self,
        sequence: Sequence[Any],
        state_manager: StreamStateManager,
        duration_ms: float,
        precision: str,
    ) -> dict[str, torch.Tensor]:
        result = _forward_stateful_sequence(
            self.detector,
            sequence,
            state_manager,
            duration_ms=duration_ms,
            device=next(self.parameters()).device,
            precision=precision,
            compute_losses=True,
        )
        if result.losses is None or result.labeled_frames <= 0:
            raise RuntimeError("scratch mixed chunk contains no supervised frame")
        return result.losses


class _LimitedBatchSampler(Sampler[list[Any]]):
    def __init__(self, sampler: Sampler[list[Any]], length: int) -> None:
        if length <= 0:
            raise ValueError("DDP scratch training needs at least one common batch")
        self.sampler = sampler
        self.length = int(length)

    def set_epoch(self, epoch: int) -> None:
        setter = getattr(self.sampler, "set_epoch", None)
        if setter is not None:
            setter(epoch)

    def set_length(self, length: int) -> None:
        if length <= 0:
            raise ValueError("DDP scratch training needs a common batch")
        self.length = int(length)

    def __len__(self) -> int:
        return self.length

    def __iter__(self) -> Iterator[list[Any]]:
        for index, batch in enumerate(self.sampler):
            if index >= self.length:
                break
            yield batch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train ConvLSTM+YOLOX from random initialization with RVT-style "
            "mixed stream/random clips on one or more GPUs."
        )
    )
    parser.add_argument("--architecture-checkpoint", required=True, type=Path)
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--val-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--window-ms", type=float, default=50.0)
    parser.add_argument("--sequence-length", type=int, default=21)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--max-train-frames", type=int, default=0)
    parser.add_argument("--max-val-frames", type=int, default=0)
    parser.add_argument("--precision", choices=("fp16", "fp32", "bf16"), default="fp16")
    parser.add_argument("--confidence-threshold", type=float, default=0.001)
    parser.add_argument("--nms-threshold", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def _distributed() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("scratch ConvLSTM detection requires CUDA")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(hours=6),
        )
    return rank, world_size, local_rank, device


def _architecture(path: Path) -> ExperimentConfig:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported architecture checkpoint schema")
    config = ExperimentConfig.from_mapping(checkpoint["resolved_config"])
    if checkpoint.get("config_hash") != config_hash(config):
        raise ValueError("architecture checkpoint has inconsistent config metadata")
    if not config.recurrent.enabled or config.recurrent.cell != "conv_lstm":
        raise ValueError("scratch sanity run requires a ConvLSTM pretrain architecture")
    return config


def _validate(args: argparse.Namespace, world_size: int) -> None:
    positive = (
        args.window_ms,
        args.sequence_length,
        args.batch_size,
        args.epochs,
        args.learning_rate,
        args.eval_every,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("window, T, batch, epochs, learning rate, and eval cadence must be positive")
    if args.batch_size % 2:
        raise ValueError("per-rank mixed batch size must be even for an exact 1:1 split")
    if args.workers < 0 or min(args.max_train_frames, args.max_val_frames) < 0:
        raise ValueError("workers and frame limits cannot be negative")
    if world_size <= 0:
        raise ValueError("world size must be positive")


def _common_steps(local_steps: int, device: torch.device) -> int:
    value = torch.tensor(local_steps, dtype=torch.int64, device=device)
    if dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.MIN)
    return int(value.item())


def _reduce_metrics(
    running: dict[str, float], labeled: int, device: torch.device
) -> dict[str, float]:
    keys = sorted(running)
    values = torch.tensor(
        [float(labeled), *(running[key] for key in keys)],
        dtype=torch.float64,
        device=device,
    )
    if dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    total = max(float(values[0].item()), 1.0)
    return {key: float(values[index + 1].item()) / total for index, key in enumerate(keys)}


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".partial", delete=False) as handle:
            temporary = handle.name
        torch.save(payload, temporary)
        Path(temporary).replace(path)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def _save(
    path: Path,
    module: _StatefulScratchTrainer,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    *,
    epoch: int,
    best_ap: float,
    args: argparse.Namespace,
    config: ExperimentConfig,
    world_size: int,
) -> None:
    _atomic_save(
        {
            "schema": SCRATCH_SCHEMA,
            "model": module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "grad_scaler": None if scaler is None else scaler.state_dict(),
            "epoch": epoch,
            "best_ap": best_ap,
            "config_hash": config_hash(config),
            "world_size": world_size,
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "window_ms": args.window_ms,
            "precision": args.precision,
            "seed": args.seed,
            "train_manifest": _manifest_identity(args.train_manifest),
            "val_manifest": _manifest_identity(args.val_manifest),
        },
        path,
    )


def _resume(
    path: Path,
    module: _StatefulScratchTrainer,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    *,
    args: argparse.Namespace,
    config: ExperimentConfig,
    world_size: int,
) -> tuple[int, float]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "schema": SCRATCH_SCHEMA,
        "config_hash": config_hash(config),
        "world_size": world_size,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "window_ms": args.window_ms,
        "precision": args.precision,
        "seed": args.seed,
        "train_manifest": _manifest_identity(args.train_manifest),
        "val_manifest": _manifest_identity(args.val_manifest),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"scratch resume {key} does not match")
    module.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None:
        if payload.get("grad_scaler") is None:
            raise ValueError("FP16 scratch resume has no GradScaler state")
        scaler.load_state_dict(payload["grad_scaler"])
    return int(payload["epoch"]), float(payload.get("best_ap", float("-inf")))


def train(args: argparse.Namespace) -> None:
    rank, world_size, local_rank, device = _distributed()
    _validate(args, world_size)
    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(args.seed)
    config = _architecture(args.architecture_checkpoint)
    _validate_stateful_window_duration(
        args.window_ms,
        stateful=True,
        sequence_loader=bool(config.recurrent.sequence_loader),
        checkpoint_ms=config.recurrent.window_ms,
    )
    if args.precision == "bf16" and torch.cuda.get_device_capability(device)[0] < 8:
        raise ValueError("BF16 scratch training requires an Ampere-or-newer GPU; use FP16 on V100")

    source_model = build_model(config)
    feature_backbone = _FeatureBackbone(source_model)
    del source_model
    components = _load_rvt_components()
    detector = WindowJEPAYOLOX(
        feature_backbone,
        components.head_type,
        freeze_backbone=False,
    ).to(device)
    if not isinstance(detector.backbone.online_encoder, RecurrentVJEPA21EventVisionTransformer):
        raise ValueError("scratch detector did not build a recurrent encoder")
    trainer = _StatefulScratchTrainer(detector).to(device)
    if world_size > 1:
        trainer = nn.SyncBatchNorm.convert_sync_batchnorm(trainer)
    parameters = [value for value in trainer.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda") if args.precision == "fp16" else None
    start_epoch = 0
    best_ap = float("-inf")
    if args.resume is not None:
        start_epoch, best_ap = _resume(
            args.resume,
            trainer,
            optimizer,
            scaler,
            args=args,
            config=config,
            world_size=world_size,
        )
    ddp: nn.Module = trainer
    if world_size > 1:
        ddp = DistributedDataParallel(
            trainer,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    train_sources = _read_label_sources(args.train_manifest)
    val_sources = _read_label_sources(args.val_manifest)
    duration_us = round(args.window_ms * 1_000)
    train_labels = _frame_references(
        train_sources,
        maximum_window_us=duration_us,
        maximum_frames=0,
        seed=args.seed,
    )
    val_labels = _frame_references(
        val_sources,
        maximum_window_us=duration_us,
        maximum_frames=0,
        seed=args.seed + 1,
    )
    stream_per_rank = args.batch_size // 2
    random_per_rank = args.batch_size // 2
    stream_limit = args.max_train_frames // 2 if args.max_train_frames else 0
    random_limit = args.max_train_frames - stream_limit if args.max_train_frames else 0
    train_references = _training_stream_references(
        train_sources,
        train_labels,
        duration_us=duration_us,
        sequence_length=args.sequence_length,
        maximum_labeled_frames=stream_limit,
        stream_lanes=stream_per_rank * world_size,
    )
    val_references = _stream_references(
        val_sources,
        val_labels,
        duration_us=duration_us,
        maximum_labeled_frames=args.max_val_frames,
        stream_lanes=args.batch_size,
    )
    representation = _representation(config)
    train_dataset = Gen1StreamDetectionDataset(
        args.train_manifest,
        train_sources,
        train_references,
        duration_ms=args.window_ms,
        representation=representation,
    )
    stream_sampler = StreamLaneBatchSampler(
        train_dataset.references,
        stream_per_rank,
        sequence_length=args.sequence_length,
        shuffle=True,
        world_size=world_size,
        rank=rank,
        seed=args.seed,
    )
    random_sampler = RandomLabelClipBatchSampler(
        train_sources,
        train_labels,
        duration_us=duration_us,
        sequence_length=args.sequence_length,
        batch_size=random_per_rank,
        lane_offset=stream_per_rank,
        maximum_labeled_frames=random_limit,
        drop_last=True,
        world_size=world_size,
        rank=rank,
        seed=args.seed + 17,
    )
    mixed_sampler = MixedStatefulBatchSampler(stream_sampler, random_sampler)
    common_steps = _common_steps(len(mixed_sampler), device)
    train_sampler = _LimitedBatchSampler(mixed_sampler, common_steps)
    loader_options = {
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=_collate_stream_sequence_detection,
        **loader_options,
    )
    val_loader = None
    if rank == 0:
        val_dataset = Gen1StreamDetectionDataset(
            args.val_manifest,
            val_sources,
            val_references,
            duration_ms=args.window_ms,
            representation=representation,
        )
        val_sampler = StreamLaneBatchSampler(
            val_dataset.references,
            args.batch_size,
            sequence_length=args.sequence_length,
            shuffle=False,
            seed=args.seed + 1,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_sampler,
            collate_fn=_collate_stream_sequence_detection,
            **loader_options,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[gen1-scratch-ddp] GPUs={world_size}, per_rank_batch={args.batch_size}, "
            f"global_batch={args.batch_size * world_size}, T={args.sequence_length}, "
            f"mixed=1:1, steps_per_epoch={common_steps}, precision={args.precision}, "
            f"trainable_parameters={sum(value.numel() for value in parameters)}, "
            "init=random, encoder_trainable=true, augmentation=none, "
            "eval=full-stream-rank0",
            flush=True,
        )

    metrics_path = args.output_dir / "train.jsonl"
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        common_steps = _common_steps(len(mixed_sampler), device)
        train_sampler.set_length(common_steps)
        ddp.train()
        state_manager = StreamStateManager(recurrent=True, stride_us=duration_us)
        running: dict[str, float] = {}
        labeled_total = 0
        progress: Any = tqdm(
            train_loader,
            desc=f"scratch detection epoch {epoch + 1}/{args.epochs}",
            disable=rank != 0,
        )
        for sequence in progress:
            stream_labels, random_labels, _, _ = _stateful_sampling_counts(
                sequence,
                sampling="mixed",
                stream_batch_size=stream_per_rank,
            )
            local_labeled = stream_labels + random_labels
            global_labeled = torch.tensor(local_labeled, dtype=torch.float32, device=device)
            if dist.is_initialized():
                dist.all_reduce(global_labeled, op=dist.ReduceOp.SUM)
            optimizer.zero_grad(set_to_none=True)
            losses = ddp(sequence, state_manager, args.window_ms, args.precision)
            if not isinstance(losses, dict):
                raise RuntimeError("scratch DDP model returned an invalid loss payload")
            # DDP averages rank gradients. This factor recovers an exact global
            # labeled-frame-weighted objective when ranks contain different counts.
            weight = world_size * local_labeled / max(float(global_labeled.item()), 1.0)
            loss = losses["loss"] * weight
            if scaler is None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 10.0)
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, 10.0, error_if_nonfinite=False)
                scaler.step(optimizer)
                scaler.update()
            labeled_total += local_labeled
            for name, value in losses.items():
                numeric = float(value.detach()) if torch.is_tensor(value) else float(value)
                running[name] = running.get(name, 0.0) + numeric * local_labeled
            if rank == 0:
                progress.set_postfix(loss=f"{float(losses['loss'].detach()):.4f}", refresh=False)

        train_metrics = _reduce_metrics(running, labeled_total, device)
        validation: dict[str, float] = {}
        should_evaluate = (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs
        if dist.is_initialized():
            dist.barrier()
        if rank == 0 and should_evaluate:
            if val_loader is None:
                raise RuntimeError("rank zero validation loader is missing")
            validation = _evaluate_stateful(
                trainer.detector,
                val_loader,
                components,
                duration_ms=args.window_ms,
                confidence_threshold=args.confidence_threshold,
                nms_threshold=args.nms_threshold,
                device=device,
                precision=args.precision,
            )
        if dist.is_initialized():
            payload: list[dict[str, float]] = [validation]
            dist.broadcast_object_list(payload, src=0)
            validation = payload[0]
        if rank == 0:
            record = {
                "epoch": epoch + 1,
                "train": train_metrics,
                "validation": validation,
                "optimizer_updates": common_steps,
                "global_batch_size": args.batch_size * world_size,
                "sequence_length": args.sequence_length,
                "sampling": "mixed-1:1",
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(f"[gen1-scratch-ddp] {json.dumps(record, sort_keys=True)}", flush=True)
            current_ap = validation.get("AP")
            if current_ap is not None and current_ap > best_ap:
                best_ap = current_ap
                _save(
                    args.output_dir / "checkpoint-best.pt",
                    trainer,
                    optimizer,
                    scaler,
                    epoch=epoch + 1,
                    best_ap=best_ap,
                    args=args,
                    config=config,
                    world_size=world_size,
                )
            _save(
                args.output_dir / "checkpoint-latest.pt",
                trainer,
                optimizer,
                scaler,
                epoch=epoch + 1,
                best_ap=best_ap,
                args=args,
                config=config,
                world_size=world_size,
            )
        if dist.is_initialized():
            dist.barrier()

    if dist.is_initialized():
        dist.destroy_process_group()


def main() -> None:
    train(_parse_args())


if __name__ == "__main__":
    main()
