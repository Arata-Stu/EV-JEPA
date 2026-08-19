from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm.auto import tqdm

from event_window_jepa.data.event_store import H5EventStore
from event_window_jepa.data.spatial_transforms import (
    SharedRandomSpatialTransform,
    SpatialTransformParameters,
)
from event_window_jepa.downstream.features import extract_patch_features
from event_window_jepa.representations.event_image import EventImage
from event_window_jepa.representations.voxel_grid import VoxelGrid
from event_window_jepa.train.checkpoint import load_pretrained_model


@dataclass(frozen=True)
class LabelSource:
    sequence_id: str
    path: Path
    timestamp_field: str
    class_field: str
    timestamps_relative: bool
    source_time_origin_us: int
    bbox_width: int
    bbox_height: int
    event_width: int
    event_height: int
    t_start_us: int
    t_end_us: int


@dataclass(frozen=True)
class FrameReference:
    source_index: int
    start: int
    stop: int
    t_end_us: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Frozen Gen1 ROI classification probe. This diagnoses spatial class "
            "separability; it is not the official detection mAP protocol."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--val-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--mode", choices=("encoder_only", "canonical"), default="encoder_only"
    )
    parser.add_argument("--train-window-ms", type=float, default=40.0)
    parser.add_argument(
        "--eval-window-ms", type=float, nargs="+", default=(40.0,)
    )
    parser.add_argument("--canonical-ms", type=float, default=40.0)
    parser.add_argument("--feature-batch-size", type=int, default=16)
    parser.add_argument("--head-batch-size", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-train-frames", type=int, default=0)
    parser.add_argument("--max-val-frames", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--precision", choices=("fp32", "bf16"), default="bf16"
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="Do not reuse extracted ROI features"
    )
    return parser.parse_args()


def _resolve_path(value: str | Path, parent: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (parent / path).resolve()


def _field_name(dtype: np.dtype[Any], candidates: Sequence[str], kind: str) -> str:
    names = dtype.names or ()
    name = next((candidate for candidate in candidates if candidate in names), None)
    if name is None:
        raise ValueError(f"bbox labels have no {kind} field; tried {tuple(candidates)}")
    return name


def _read_label_sources(manifest: Path) -> tuple[LabelSource, ...]:
    manifest = manifest.expanduser().resolve()
    sources: list[LabelSource] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row: dict[str, Any] = json.loads(line)
            if "bbox_path" not in row:
                raise ValueError(
                    f"manifest line {line_number} has no bbox_path: {manifest}"
                )
            bbox_path = _resolve_path(row["bbox_path"], manifest.parent)
            labels = np.load(bbox_path, mmap_mode="r", allow_pickle=False)
            if labels.dtype.names is None or labels.ndim != 1:
                raise ValueError(f"bbox labels must be a structured 1-D NPY: {bbox_path}")
            timestamp_field = str(
                row.get(
                    "bbox_timestamp_field",
                    _field_name(
                        labels.dtype, ("t", "t_us", "timestamp", "timestamps"), "timestamp"
                    ),
                )
            )
            class_field = _field_name(
                labels.dtype, ("class_id", "class", "label", "category_id"), "class"
            )
            event_width = int(row["width"])
            event_height = int(row["height"])
            sources.append(
                LabelSource(
                    sequence_id=str(row["sequence_id"]),
                    path=bbox_path,
                    timestamp_field=timestamp_field,
                    class_field=class_field,
                    timestamps_relative=bool(row.get("bbox_timestamps_relative", False)),
                    source_time_origin_us=int(row.get("source_time_origin_us", 0)),
                    bbox_width=int(row.get("bbox_width", event_width)),
                    bbox_height=int(row.get("bbox_height", event_height)),
                    event_width=event_width,
                    event_height=event_height,
                    t_start_us=int(row["t_start_us"]),
                    t_end_us=int(row["t_end_us"]),
                )
            )
    if not sources:
        raise ValueError(f"manifest contains no label sources: {manifest}")
    return tuple(sources)


def _frame_references(
    sources: Sequence[LabelSource],
    *,
    maximum_window_us: int,
    maximum_frames: int,
    seed: int,
) -> tuple[FrameReference, ...]:
    references: list[FrameReference] = []
    for source_index, source in enumerate(sources):
        labels = np.load(source.path, mmap_mode="r", allow_pickle=False)
        timestamps = np.asarray(labels[source.timestamp_field])
        if len(timestamps) == 0:
            continue
        if not np.issubdtype(timestamps.dtype, np.integer):
            raise TypeError(f"bbox timestamps must be integers: {source.path}")
        if len(timestamps) and np.any(timestamps[1:] < timestamps[:-1]):
            raise ValueError(f"bbox timestamps are not sorted: {source.path}")
        starts = np.flatnonzero(np.r_[True, timestamps[1:] != timestamps[:-1]])
        stops = np.r_[starts[1:], len(timestamps)]
        for start, stop in zip(starts.tolist(), stops.tolist(), strict=True):
            source_timestamp = int(timestamps[start])
            timestamp = (
                source_timestamp
                if source.timestamps_relative
                else source_timestamp - source.source_time_origin_us
            )
            if (
                timestamp - maximum_window_us >= source.t_start_us
                and timestamp <= source.t_end_us
            ):
                references.append(
                    FrameReference(source_index, int(start), int(stop), timestamp)
                )
    if maximum_frames > 0 and len(references) > maximum_frames:
        rng = random.Random(seed)
        selected = sorted(rng.sample(range(len(references)), maximum_frames))
        references = [references[index] for index in selected]
    return tuple(references)


def _center_crop_parameters(
    input_height: int, input_width: int, output_size: tuple[int, int]
) -> SpatialTransformParameters:
    output_height, output_width = output_size
    if output_height > input_height or output_width > input_width:
        raise ValueError(
            f"model input {output_width}x{output_height} exceeds event frame "
            f"{input_width}x{input_height}"
        )
    return SpatialTransformParameters(
        x0=(input_width - output_width) // 2,
        y0=(input_height - output_height) // 2,
        output_height=output_height,
        output_width=output_width,
        horizontal_flip=False,
    )


def _crop_boxes(
    labels: np.ndarray,
    source: LabelSource,
    crop: SpatialTransformParameters,
) -> tuple[np.ndarray, np.ndarray]:
    for field in ("x", "y", "w", "h", source.class_field):
        if field not in (labels.dtype.names or ()):
            raise ValueError(f"bbox labels have no {field!r} field: {source.path}")
    scale_x = source.event_width / source.bbox_width
    scale_y = source.event_height / source.bbox_height
    x1 = np.asarray(labels["x"], dtype=np.float32) * scale_x - crop.x0
    y1 = np.asarray(labels["y"], dtype=np.float32) * scale_y - crop.y0
    x2 = x1 + np.asarray(labels["w"], dtype=np.float32) * scale_x
    y2 = y1 + np.asarray(labels["h"], dtype=np.float32) * scale_y
    x1 = np.clip(x1, 0, crop.output_width)
    y1 = np.clip(y1, 0, crop.output_height)
    x2 = np.clip(x2, 0, crop.output_width)
    y2 = np.clip(y2, 0, crop.output_height)
    keep = (x2 > x1) & (y2 > y1)
    boxes = np.stack((x1[keep], y1[keep], x2[keep], y2[keep]), axis=1)
    classes = np.asarray(labels[source.class_field], dtype=np.int64)[keep]
    return boxes.astype(np.float32, copy=False), classes


class Gen1FrameDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        manifest: Path,
        sources: Sequence[LabelSource],
        references: Sequence[FrameReference],
        *,
        duration_ms: float,
        image_size: tuple[int, int],
        representation: Any,
    ) -> None:
        self.store = H5EventStore(manifest)
        self.sources = tuple(sources)
        self.references = tuple(references)
        self.duration_us = round(duration_ms * 1_000)
        self.image_size = image_size
        self.representation = representation
        self._labels: OrderedDict[Path, np.ndarray] = OrderedDict()

    def __len__(self) -> int:
        return len(self.references)

    def _label_array(self, path: Path) -> np.ndarray:
        if path in self._labels:
            labels = self._labels.pop(path)
            self._labels[path] = labels
            return labels
        labels = np.load(path, mmap_mode="r", allow_pickle=False)
        self._labels[path] = labels
        while len(self._labels) > 8:
            self._labels.popitem(last=False)
        return labels

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        reference = self.references[index]
        source = self.sources[reference.source_index]
        window = self.store.slice(source.sequence_id, reference.t_end_us, self.duration_us)
        crop = _center_crop_parameters(window.height, window.width, self.image_size)
        cropped_window = SharedRandomSpatialTransform.apply(window, crop)
        image = torch.from_numpy(self.representation(cropped_window))
        labels = self._label_array(source.path)[reference.start : reference.stop]
        boxes, classes = _crop_boxes(labels, source, crop)
        return image, torch.from_numpy(boxes), torch.from_numpy(classes)


def _collate_frames(
    batch: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    images, boxes, classes = zip(*batch, strict=True)
    return torch.stack(images), tuple(boxes), tuple(classes)


def _roi_pool_tokens(
    tokens: torch.Tensor,
    boxes: Sequence[torch.Tensor],
    *,
    image_size: tuple[int, int],
    grid_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, _, feature_dim = tokens.shape
    grid_height, grid_width = grid_size
    if tokens.shape[1] != grid_height * grid_width:
        raise ValueError("token count does not match the encoder patch grid")
    maps = tokens.reshape(batch_size, grid_height, grid_width, feature_dim)
    pooled: list[torch.Tensor] = []
    owners: list[int] = []
    scale_x = grid_width / image_size[1]
    scale_y = grid_height / image_size[0]
    for owner, frame_boxes in enumerate(boxes):
        for box in frame_boxes:
            x0 = max(0, min(grid_width - 1, math.floor(float(box[0]) * scale_x)))
            y0 = max(0, min(grid_height - 1, math.floor(float(box[1]) * scale_y)))
            x1 = max(x0 + 1, min(grid_width, math.ceil(float(box[2]) * scale_x)))
            y1 = max(y0 + 1, min(grid_height, math.ceil(float(box[3]) * scale_y)))
            pooled.append(maps[owner, y0:y1, x0:x1].mean(dim=(0, 1)))
            owners.append(owner)
    if not pooled:
        return tokens.new_empty((0, feature_dim)), torch.empty(
            0, dtype=torch.long, device=tokens.device
        )
    return torch.stack(pooled), torch.tensor(owners, dtype=torch.long, device=tokens.device)


def _representation(config: Any) -> Any:
    if config.representation.kind == "voxel_grid":
        return VoxelGrid(
            temporal_bins=config.representation.temporal_bins,
            normalization=config.representation.normalization,
        )
    return EventImage(normalization=config.representation.normalization)


def _feature_cache_path(
    output_dir: Path,
    *,
    split: str,
    mode: str,
    duration_ms: float,
    canonical_ms: float,
    checkpoint: Path,
    sample_hash: str,
) -> Path:
    identity = hashlib.sha256(
        f"{checkpoint.resolve()}:{checkpoint.stat().st_size}:{checkpoint.stat().st_mtime_ns}"
        .encode("utf-8")
    ).hexdigest()[:12]
    duration = f"{duration_ms:g}".replace(".", "p")
    canonical = f"{canonical_ms:g}".replace(".", "p")
    return output_dir / "features" / (
        f"{split}_{mode}_{duration}ms_c{canonical}ms_{sample_hash[:12]}_{identity}.pt"
    )


def _sample_hash(
    sources: Sequence[LabelSource], references: Sequence[FrameReference]
) -> str:
    digest = hashlib.sha256()
    for reference in references:
        source = sources[reference.source_index]
        digest.update(
            f"{source.sequence_id}\0{reference.t_end_us}\0{reference.start}\0"
            f"{reference.stop}\n".encode("utf-8")
        )
    return digest.hexdigest()


@torch.no_grad()
def _extract_features(
    model: Any,
    dataset: Gen1FrameDataset,
    *,
    duration_ms: float,
    mode: str,
    canonical_ms: float,
    device: torch.device,
    precision: str,
    batch_size: int,
    workers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        collate_fn=_collate_frames,
    )
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for images, frame_boxes, frame_classes in tqdm(loader, desc=f"features {duration_ms:g}ms"):
        images = images.to(device, non_blocking=True)
        duration = torch.full((len(images),), duration_ms, device=device)
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if precision == "bf16" and device.type == "cuda"
            else nullcontext()
        )
        with context:
            tokens = extract_patch_features(
                model, images, duration, mode=mode, canonical_ms=canonical_ms
            )
            pooled, owners = _roi_pool_tokens(
                tokens,
                frame_boxes,
                image_size=model.online_encoder.image_size,
                grid_size=model.online_encoder.grid_size,
            )
        if pooled.numel() == 0:
            continue
        batch_labels = torch.cat(frame_classes).to(device=device)
        # Empty boxes were already removed per frame; owner order therefore matches cat.
        if len(batch_labels) != len(owners):
            raise RuntimeError("ROI feature and class counts disagree")
        features.append(pooled.float().cpu())
        labels.append(batch_labels.cpu())
    if not features:
        raise ValueError("no bounding boxes intersect the deterministic center crop")
    return torch.cat(features), torch.cat(labels)


def _load_or_extract(
    cache_path: Path,
    *,
    no_cache: bool,
    extract: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cache_path.exists() and not no_cache:
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        return payload["features"].float(), payload["labels"].long()
    features, labels = extract()
    if not no_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(cache_path.suffix + ".partial")
        torch.save({"features": features.half(), "labels": labels}, temporary)
        temporary.replace(cache_path)
    return features, labels


def _class_mapping(labels: torch.Tensor) -> dict[int, int]:
    return {int(value): index for index, value in enumerate(sorted(labels.unique().tolist()))}


def _remap_labels(
    labels: torch.Tensor, mapping: Mapping[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    keep = torch.tensor([int(value) in mapping for value in labels.tolist()])
    mapped = torch.tensor(
        [mapping[int(value)] for value in labels[keep].tolist()], dtype=torch.long
    )
    return mapped, keep


def _train_head(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    class_count: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> nn.Linear:
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(features, labels),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    head = nn.Linear(features.shape[1], class_count).to(device)
    counts = torch.bincount(labels, minlength=class_count).float()
    weights = (counts.sum() / counts.clamp_min(1)).sqrt()
    weights = (weights / weights.mean()).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    head.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total = 0
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(head(batch_features), batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_labels)
            total += len(batch_labels)
        print(f"[gen1-probe] head epoch {epoch + 1}/{epochs}: loss={total_loss / total:.5f}")
    return head.eval()


@torch.no_grad()
def _metrics(
    head: nn.Linear,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    class_count: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    confusion = torch.zeros((class_count, class_count), dtype=torch.long)
    loader = DataLoader(TensorDataset(features, labels), batch_size=batch_size)
    for batch_features, batch_labels in loader:
        predictions = head(batch_features.to(device)).argmax(dim=1).cpu()
        indices = batch_labels * class_count + predictions
        confusion += torch.bincount(indices, minlength=class_count**2).reshape(
            class_count, class_count
        )
    true_positive = confusion.diag().float()
    precision = true_positive / confusion.sum(dim=0).clamp_min(1)
    recall = true_positive / confusion.sum(dim=1).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {
        "accuracy": float(true_positive.sum() / confusion.sum().clamp_min(1)),
        "balanced_accuracy": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class_f1": f1.tolist(),
        "confusion_matrix": confusion.tolist(),
        "samples": int(confusion.sum()),
    }


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "train_window_ms": args.train_window_ms,
        "canonical_ms": args.canonical_ms,
        "feature_batch_size": args.feature_batch_size,
        "head_batch_size": args.head_batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
    }
    if any(value <= 0 for value in positive.values()):
        raise ValueError(f"arguments must be positive: {positive}")
    if any(value <= 0 for value in args.eval_window_ms):
        raise ValueError("eval window durations must be positive")
    if args.workers < 0 or args.max_train_frames < 0 or args.max_val_frames < 0:
        raise ValueError("worker and frame limits cannot be negative")


def _window_group(duration_ms: float, trained_windows_ms: Sequence[float]) -> str:
    trained = {float(value) for value in trained_windows_ms}
    if duration_ms in trained:
        return "seen"
    return (
        "unseen_extrapolation"
        if duration_ms < min(trained) or duration_ms > max(trained)
        else "unseen_interpolation"
    )


def run(args: argparse.Namespace) -> None:
    _validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, config = load_pretrained_model(args.checkpoint, device=device)
    model.requires_grad_(False)
    image_size = tuple(config.model.image_size)
    representation = _representation(config)

    train_sources = _read_label_sources(args.train_manifest)
    val_sources = _read_label_sources(args.val_manifest)
    maximum_window_us = round(max(args.train_window_ms, *args.eval_window_ms) * 1_000)
    train_references = _frame_references(
        train_sources,
        maximum_window_us=maximum_window_us,
        maximum_frames=args.max_train_frames,
        seed=args.seed,
    )
    val_references = _frame_references(
        val_sources,
        maximum_window_us=maximum_window_us,
        maximum_frames=args.max_val_frames,
        seed=args.seed + 1,
    )
    train_hash = _sample_hash(train_sources, train_references)
    val_hash = _sample_hash(val_sources, val_references)
    if not train_references or not val_references:
        raise ValueError("no valid labeled frames remain after the maximum-window filter")
    print(
        f"[gen1-probe] frames: train={len(train_references)}, "
        f"val={len(val_references)}, mode={args.mode}"
    )

    def features_for(
        *,
        split: str,
        manifest: Path,
        sources: Sequence[LabelSource],
        references: Sequence[FrameReference],
        duration_ms: float,
        sample_hash: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dataset = Gen1FrameDataset(
            manifest,
            sources,
            references,
            duration_ms=duration_ms,
            image_size=image_size,
            representation=representation,
        )
        cache_path = _feature_cache_path(
            args.output_dir,
            split=split,
            mode=args.mode,
            duration_ms=duration_ms,
            canonical_ms=args.canonical_ms,
            checkpoint=args.checkpoint,
            sample_hash=sample_hash,
        )
        return _load_or_extract(
            cache_path,
            no_cache=args.no_cache,
            extract=lambda: _extract_features(
                model,
                dataset,
                duration_ms=duration_ms,
                mode=args.mode,
                canonical_ms=args.canonical_ms,
                device=device,
                precision=args.precision,
                batch_size=args.feature_batch_size,
                workers=args.workers,
            ),
        )

    train_features, raw_train_labels = features_for(
        split="train",
        manifest=args.train_manifest,
        sources=train_sources,
        references=train_references,
        duration_ms=args.train_window_ms,
        sample_hash=train_hash,
    )
    mapping = _class_mapping(raw_train_labels)
    if len(mapping) < 2:
        raise ValueError(f"the training probe requires at least two classes, found {mapping}")
    train_labels, train_keep = _remap_labels(raw_train_labels, mapping)
    train_features = train_features[train_keep]
    print(
        f"[gen1-probe] train ROIs={len(train_labels)}, classes={sorted(mapping)}"
    )
    head = _train_head(
        train_features,
        train_labels,
        class_count=len(mapping),
        epochs=args.epochs,
        batch_size=args.head_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=device,
    )
    torch.save(
        {"head": head.cpu().state_dict(), "class_mapping": mapping},
        args.output_dir / f"head_{args.mode}.pt",
    )
    head.to(device)

    results: list[dict[str, Any]] = []
    metrics_path = args.output_dir / f"window_metrics_{args.mode}.jsonl"
    with metrics_path.open("w", encoding="utf-8") as handle:
        for duration_ms in args.eval_window_ms:
            val_features, raw_val_labels = features_for(
                split="val",
                manifest=args.val_manifest,
                sources=val_sources,
                references=val_references,
                duration_ms=duration_ms,
                sample_hash=val_hash,
            )
            val_labels, keep = _remap_labels(raw_val_labels, mapping)
            val_features = val_features[keep]
            values = _metrics(
                head,
                val_features,
                val_labels,
                class_count=len(mapping),
                batch_size=args.head_batch_size,
                device=device,
            )
            record = {
                "method": f"gen1_roi_probe_{args.mode}",
                "metric": "macro_f1",
                "seed": args.seed,
                "window_ms": float(duration_ms),
                "value": values["macro_f1"],
                "higher_is_better": True,
                "sample_set_id": val_hash,
                "window_group": _window_group(
                    duration_ms,
                    (*config.windows.train_ms, *config.windows.target_ms),
                ),
                **values,
            }
            results.append(record)
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(
                f"[gen1-probe] {duration_ms:g} ms: macro-F1={values['macro_f1']:.4f}, "
                f"balanced-acc={values['balanced_accuracy']:.4f}, "
                f"accuracy={values['accuracy']:.4f}"
            )
    summary = {
        "protocol": "frozen_roi_classification_probe_not_detection_map",
        "checkpoint": str(args.checkpoint.resolve()),
        "mode": args.mode,
        "train_window_ms": args.train_window_ms,
        "class_mapping": mapping,
        "train_frames": len(train_references),
        "val_frames": len(val_references),
        "train_rois": len(train_labels),
        "results": results,
    }
    (args.output_dir / f"summary_{args.mode}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
