from __future__ import annotations

import argparse
import json
import random
import tempfile
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from event_window_jepa.data.event_store import H5EventStore
from event_window_jepa.downstream.features import require_feedforward_feature_model
from event_window_jepa.downstream.gen1_roi_probe import (
    FrameReference,
    LabelSource,
    _frame_references,
    _read_label_sources,
    _representation,
)
from event_window_jepa.train.checkpoint import load_pretrained_model


BBOX_DTYPE = np.dtype(
    {
        "names": (
            "t",
            "x",
            "y",
            "w",
            "h",
            "class_id",
            "track_id",
            "class_confidence",
        ),
        "formats": ("<i8", "<f4", "<f4", "<f4", "<f4", "<u4", "<u4", "<f4"),
    }
)


@dataclass(frozen=True)
class RVTComponents:
    head_type: type[nn.Module]
    postprocess: Callable[..., list[torch.Tensor | None]]
    evaluate_list: Callable[..., dict[str, float]]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a Gen1 YOLOX detector on full-frame Window-JEPA features and "
            "evaluate with the official Prophesee COCO protocol."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--val-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--backbone-init", choices=("pretrained", "random"), default="pretrained"
    )
    parser.add_argument("--window-ms", type=float, default=40.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--max-train-frames", type=int, default=0)
    parser.add_argument("--max-val-frames", type=int, default=0)
    parser.add_argument("--confidence-threshold", type=float, default=0.001)
    parser.add_argument("--nms-threshold", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument(
        "--unfreeze-backbone",
        action="store_true",
        help="Fine-tune the encoder; the default is a frozen-backbone detector",
    )
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def _load_rvt_components() -> RVTComponents:
    try:
        from event_window_jepa.third_party.rvt_detection import (
            YOLOXHead,
            evaluate_list,
            postprocess,
        )
    except ImportError as error:
        raise ImportError(
            "Gen1 detection requires torchvision and pycocotools; install "
            "event-window-jepa[detection,hdf5]"
        ) from error
    return RVTComponents(
        head_type=YOLOXHead,
        postprocess=postprocess,
        evaluate_list=evaluate_list,
    )


def _scaled_full_boxes(
    labels: np.ndarray, source: LabelSource
) -> tuple[np.ndarray, np.ndarray]:
    scale_x = source.event_width / source.bbox_width
    scale_y = source.event_height / source.bbox_height
    x1 = np.asarray(labels["x"], dtype=np.float32) * scale_x
    y1 = np.asarray(labels["y"], dtype=np.float32) * scale_y
    x2 = x1 + np.asarray(labels["w"], dtype=np.float32) * scale_x
    y2 = y1 + np.asarray(labels["h"], dtype=np.float32) * scale_y
    x1 = np.clip(x1, 0, source.event_width)
    y1 = np.clip(y1, 0, source.event_height)
    x2 = np.clip(x2, 0, source.event_width)
    y2 = np.clip(y2, 0, source.event_height)
    keep = (x2 > x1) & (y2 > y1)
    boxes = np.stack((x1[keep], y1[keep], x2[keep], y2[keep]), axis=1)
    classes = np.asarray(labels[source.class_field], dtype=np.int64)[keep]
    return boxes.astype(np.float32, copy=False), classes


def _ground_truth_array(
    boxes: np.ndarray, classes: np.ndarray, timestamp_us: int
) -> np.ndarray:
    result = np.zeros(len(boxes), dtype=BBOX_DTYPE)
    result["t"] = timestamp_us
    result["x"] = boxes[:, 0]
    result["y"] = boxes[:, 1]
    result["w"] = boxes[:, 2] - boxes[:, 0]
    result["h"] = boxes[:, 3] - boxes[:, 1]
    result["class_id"] = classes
    result["class_confidence"] = 1.0
    return result


class Gen1DetectionDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, np.ndarray, int]]
):
    def __init__(
        self,
        manifest: Path,
        sources: Sequence[LabelSource],
        references: Sequence[FrameReference],
        *,
        duration_ms: float,
        representation: Any,
    ) -> None:
        self.store = H5EventStore(manifest)
        self.sources = tuple(sources)
        self.references = tuple(references)
        self.duration_us = round(duration_ms * 1_000)
        self.representation = representation
        self._label_arrays: OrderedDict[Path, np.ndarray] = OrderedDict()

    def __len__(self) -> int:
        return len(self.references)

    def _labels(self, path: Path) -> np.ndarray:
        if path in self._label_arrays:
            labels = self._label_arrays.pop(path)
            self._label_arrays[path] = labels
            return labels
        labels = np.load(path, mmap_mode="r", allow_pickle=False)
        self._label_arrays[path] = labels
        while len(self._label_arrays) > 8:
            self._label_arrays.popitem(last=False)
        return labels

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, int]:
        reference = self.references[index]
        source = self.sources[reference.source_index]
        window = self.store.slice(source.sequence_id, reference.t_end_us, self.duration_us)
        image = torch.from_numpy(self.representation(window))
        labels = self._labels(source.path)[reference.start : reference.stop]
        boxes, classes = _scaled_full_boxes(labels, source)
        targets = np.zeros((len(boxes), 5), dtype=np.float32)
        targets[:, 0] = classes
        targets[:, 1] = (boxes[:, 0] + boxes[:, 2]) * 0.5
        targets[:, 2] = (boxes[:, 1] + boxes[:, 3]) * 0.5
        targets[:, 3] = boxes[:, 2] - boxes[:, 0]
        targets[:, 4] = boxes[:, 3] - boxes[:, 1]
        ground_truth = _ground_truth_array(boxes, classes, reference.t_end_us)
        return image, torch.from_numpy(targets), ground_truth, reference.t_end_us


def _collate_detection(
    batch: Sequence[tuple[torch.Tensor, torch.Tensor, np.ndarray, int]],
) -> tuple[torch.Tensor, torch.Tensor, tuple[np.ndarray, ...], tuple[int, ...]]:
    images, labels, ground_truth, timestamps = zip(*batch, strict=True)
    maximum = max(len(value) for value in labels)
    padded = torch.zeros((len(labels), maximum, 5), dtype=torch.float32)
    for index, value in enumerate(labels):
        padded[index, : len(value)] = value
    return torch.stack(images), padded, tuple(ground_truth), tuple(timestamps)


def _interpolated_position_embedding(
    encoder: Any, grid_size: tuple[int, int], dtype: torch.dtype
) -> torch.Tensor:
    source_height, source_width = encoder.grid_size
    target_height, target_width = grid_size
    position = encoder.position_embedding
    if (source_height, source_width) == grid_size:
        return position.to(dtype=dtype)
    position_map = position.reshape(1, source_height, source_width, encoder.embed_dim)
    position_map = position_map.permute(0, 3, 1, 2).float()
    position_map = functional.interpolate(
        position_map,
        size=(target_height, target_width),
        mode="bicubic",
        align_corners=False,
    )
    return position_map.permute(0, 2, 3, 1).reshape(
        1, target_height * target_width, encoder.embed_dim
    ).to(dtype=dtype)


def _dynamic_encoder_feature_map(
    model: Any, images: torch.Tensor, duration_ms: torch.Tensor
) -> torch.Tensor:
    encoder = model.online_encoder
    scale = model.scale_embedding(duration_ms.reshape(len(images)))
    if not model.condition_on_scale:
        scale = torch.zeros_like(scale)
    if hasattr(encoder, "forward_feature_map"):
        return encoder.forward_feature_map(images, scale)
    patches_2d = encoder.patch_embed(images)
    grid_size = tuple(patches_2d.shape[-2:])
    patches = patches_2d.flatten(2).transpose(1, 2)
    patches = patches + _interpolated_position_embedding(
        encoder, grid_size, patches.dtype
    )
    scale_token = encoder.scale_projection(scale).unsqueeze(1)
    tokens = encoder.norm(encoder.blocks(torch.cat((scale_token, patches), dim=1)))
    tokens = tokens[:, 1:]
    return tokens.transpose(1, 2).reshape(
        len(images), encoder.embed_dim, *grid_size
    )


class WindowJEPAYOLOX(nn.Module):
    def __init__(
        self,
        backbone: Any,
        head_type: type[nn.Module],
        *,
        freeze_backbone: bool,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        feature_dim = backbone.online_encoder.embed_dim
        channels = (96, 192, 384)
        self.p3 = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, channels[0], 2, 2),
            nn.BatchNorm2d(channels[0]),
            nn.SiLU(),
        )
        self.p4 = nn.Sequential(
            nn.Conv2d(feature_dim, channels[1], 1),
            nn.BatchNorm2d(channels[1]),
            nn.SiLU(),
        )
        self.p5 = nn.Sequential(
            nn.Conv2d(feature_dim, channels[2], 3, 2, 1),
            nn.BatchNorm2d(channels[2]),
            nn.SiLU(),
        )
        self.head = head_type(
            num_classes=2,
            strides=(8, 16, 32),
            in_channels=channels,
            act="silu",
            depthwise=False,
            compile_cfg=None,
        )
        if freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode: bool = True) -> WindowJEPAYOLOX:
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(
        self,
        images: torch.Tensor,
        duration_ms: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        # Gen1 is 240x304. Bottom/right zero padding gives a 256x320 grid whose
        # three detection levels align exactly at strides 8, 16, and 32.
        images = functional.pad(images, (0, 16, 0, 16))
        if self.freeze_backbone:
            with torch.no_grad():
                base = _dynamic_encoder_feature_map(self.backbone, images, duration_ms)
        else:
            base = _dynamic_encoder_feature_map(self.backbone, images, duration_ms)
        features = (self.p3(base), self.p4(base), self.p5(base))
        return self.head(features, targets)


def _predictions_to_prophesee(
    detections: Sequence[torch.Tensor | None],
    timestamps: Sequence[int],
    *,
    height: int,
    width: int,
) -> list[np.ndarray]:
    output: list[np.ndarray] = []
    for detection, timestamp in zip(detections, timestamps, strict=True):
        if detection is None:
            output.append(np.zeros(0, dtype=BBOX_DTYPE))
            continue
        values = detection.detach().float().cpu().numpy()
        x1 = np.clip(values[:, 0], 0, width)
        y1 = np.clip(values[:, 1], 0, height)
        x2 = np.clip(values[:, 2], 0, width)
        y2 = np.clip(values[:, 3], 0, height)
        keep = (x2 > x1) & (y2 > y1)
        result = np.zeros(int(np.count_nonzero(keep)), dtype=BBOX_DTYPE)
        result["t"] = timestamp
        result["x"] = x1[keep]
        result["y"] = y1[keep]
        result["w"] = x2[keep] - x1[keep]
        result["h"] = y2[keep] - y1[keep]
        result["class_id"] = values[keep, 6].astype(np.uint32)
        # This matches RVT's to_prophesee conversion after product-threshold NMS.
        result["class_confidence"] = values[keep, 5]
        output.append(result)
    return output


@torch.no_grad()
def _evaluate(
    model: WindowJEPAYOLOX,
    loader: DataLoader[Any],
    components: RVTComponents,
    *,
    duration_ms: float,
    confidence_threshold: float,
    nms_threshold: float,
    device: torch.device,
    precision: str,
) -> dict[str, float]:
    model.eval()
    ground_truth: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for images, _, batch_ground_truth, timestamps in tqdm(loader, desc="detection val"):
        images = images.to(device, non_blocking=True)
        duration = torch.full((len(images),), duration_ms, device=device)
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if precision == "bf16" and device.type == "cuda"
            else nullcontext()
        )
        with context:
            decoded, _ = model(images, duration)
        processed = components.postprocess(
            decoded.float().clone(),
            num_classes=2,
            conf_thre=confidence_threshold,
            nms_thre=nms_threshold,
        )
        ground_truth.extend(batch_ground_truth)
        predictions.extend(
            _predictions_to_prophesee(
                processed, timestamps, height=240, width=304
            )
        )
    metrics = components.evaluate_list(
        result_boxes_list=predictions,
        gt_boxes_list=ground_truth,
        height=240,
        width=304,
        camera="gen1",
        apply_bbox_filters=True,
        downsampled_by_2=False,
        return_aps=True,
    )
    return {str(key): float(value) for key, value in metrics.items()}


def _save_checkpoint(
    path: Path,
    model: WindowJEPAYOLOX,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    args: argparse.Namespace,
    metrics: dict[str, float],
) -> None:
    detector_state = {
        name: value
        for name, value in model.state_dict().items()
        if not name.startswith("backbone.")
        or (
            args.unfreeze_backbone
            and (
                name.startswith("backbone.online_encoder.")
                or name.startswith("backbone.scale_embedding.")
            )
        )
    }
    payload = {
        "schema": "event-window-jepa-gen1-yolox-v1",
        "model": detector_state,
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "backbone_init": args.backbone_init,
        "pretrain_checkpoint": str(args.checkpoint.resolve()),
        "window_ms": args.window_ms,
        "metrics": metrics,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".partial", delete=False
        ) as handle:
            temporary_name = handle.name
        torch.save(payload, temporary_name)
        Path(temporary_name).replace(path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.window_ms,
        args.batch_size,
        args.epochs,
        args.learning_rate,
        args.eval_every,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("window, batch, epoch, learning-rate, and eval cadence must be positive")
    if args.workers < 0 or min(args.max_train_frames, args.max_val_frames) < 0:
        raise ValueError("workers and frame limits cannot be negative")
    if not 0 < args.confidence_threshold < 1 or not 0 < args.nms_threshold < 1:
        raise ValueError("confidence and NMS thresholds must lie inside (0, 1)")


def train(args: argparse.Namespace) -> None:
    _validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    backbone, config = load_pretrained_model(args.checkpoint, device=device)
    if args.backbone_init == "random":
        from event_window_jepa.train.pretrain import build_model

        torch.manual_seed(args.seed)
        backbone = build_model(config).to(device)
    require_feedforward_feature_model(backbone, caller="Gen1 detection")
    components = _load_rvt_components()
    representation = _representation(config)
    train_sources = _read_label_sources(args.train_manifest)
    val_sources = _read_label_sources(args.val_manifest)
    duration_us = round(args.window_ms * 1_000)
    train_references = _frame_references(
        train_sources,
        maximum_window_us=duration_us,
        maximum_frames=args.max_train_frames,
        seed=args.seed,
    )
    val_references = _frame_references(
        val_sources,
        maximum_window_us=duration_us,
        maximum_frames=args.max_val_frames,
        seed=args.seed + 1,
    )
    if not train_references or not val_references:
        raise ValueError("no valid labeled frames remain")
    model = WindowJEPAYOLOX(
        backbone,
        components.head_type,
        freeze_backbone=not args.unfreeze_backbone,
    ).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    start_epoch = 0
    if args.resume is not None:
        resumed = torch.load(args.resume, map_location="cpu", weights_only=False)
        if resumed.get("schema") != "event-window-jepa-gen1-yolox-v1":
            raise ValueError("unsupported detection checkpoint")
        incompatible = model.load_state_dict(resumed["model"], strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing = [
            name
            for name in incompatible.missing_keys
            if not name.startswith("backbone.")
        ]
        if unexpected or missing:
            raise ValueError(
                f"incompatible detection checkpoint: missing={missing}, "
                f"unexpected={unexpected}"
            )
        optimizer.load_state_dict(resumed["optimizer"])
        start_epoch = int(resumed["epoch"])

    train_dataset = Gen1DetectionDataset(
        args.train_manifest,
        train_sources,
        train_references,
        duration_ms=args.window_ms,
        representation=representation,
    )
    val_dataset = Gen1DetectionDataset(
        args.val_manifest,
        val_sources,
        val_references,
        duration_ms=args.window_ms,
        representation=representation,
    )
    train_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        collate_fn=_collate_detection,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        collate_fn=_collate_detection,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "train.jsonl"
    print(
        f"[gen1-detect] train={len(train_dataset)}, val={len(val_dataset)}, "
        f"init={args.backbone_init}, frozen={not args.unfreeze_backbone}"
    )
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running: dict[str, float] = {}
        samples = 0
        progress = tqdm(train_loader, desc=f"detection epoch {epoch + 1}/{args.epochs}")
        for images, targets, _, _ in progress:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            duration = torch.full((len(images),), args.window_ms, device=device)
            optimizer.zero_grad(set_to_none=True)
            context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if args.precision == "bf16" and device.type == "cuda"
                else nullcontext()
            )
            with context:
                _, losses = model(images, duration, targets)
            if losses is None:
                raise RuntimeError("YOLOX returned no training losses")
            loss = losses["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 10.0)
            optimizer.step()
            batch_size = len(images)
            samples += batch_size
            for name, value in losses.items():
                numeric = float(value.detach()) if torch.is_tensor(value) else float(value)
                running[name] = running.get(name, 0.0) + numeric * batch_size
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}", refresh=False)
        train_metrics = {name: value / samples for name, value in running.items()}
        should_evaluate = (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs
        validation_metrics: dict[str, float] = {}
        if should_evaluate:
            validation_metrics = _evaluate(
                model,
                val_loader,
                components,
                duration_ms=args.window_ms,
                confidence_threshold=args.confidence_threshold,
                nms_threshold=args.nms_threshold,
                device=device,
                precision=args.precision,
            )
        record = {
            "epoch": epoch + 1,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(f"[gen1-detect] {json.dumps(record, sort_keys=True)}")
        _save_checkpoint(
            args.output_dir / "checkpoint-latest.pt",
            model,
            optimizer,
            epoch=epoch + 1,
            args=args,
            metrics=validation_metrics,
        )


def main() -> None:
    train(_parse_args())


if __name__ == "__main__":
    main()
