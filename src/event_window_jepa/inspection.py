from __future__ import annotations

import argparse
import base64
import binascii
import html
import json
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from event_window_jepa.config import ExperimentConfig
from event_window_jepa.data.paired_window_dataset import (
    PairedWindowDataset,
    PairedWindowDebugSample,
)
from event_window_jepa.data.types import EventWindow
from event_window_jepa.representations.voxel_grid import polarity_indices
from event_window_jepa.train.pretrain import build_dataset


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render deterministic context/target samples and masks as an HTML report"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--expected-dataset", type=str, default=None)
    return parser.parse_args()


def _event_counts(window: EventWindow) -> np.ndarray:
    counts = np.zeros((2, window.height, window.width), dtype=np.float32)
    if window.event_count:
        polarity = polarity_indices(window.polarity)
        np.add.at(counts, (polarity, window.y, window.x), 1.0)
    return counts


def _display_scale(arrays: list[np.ndarray]) -> float:
    values = np.concatenate([array[array > 0].reshape(-1) for array in arrays])
    if not values.size:
        return 1.0
    return max(float(np.quantile(values, 0.995)), 1e-6)


def _polarity_rgb(off: np.ndarray, on: np.ndarray, scale: float) -> np.ndarray:
    off_level = np.clip(off / scale, 0.0, 1.0)
    on_level = np.clip(on / scale, 0.0, 1.0)
    rgb = np.zeros((*off.shape, 3), dtype=np.float32)
    rgb += off_level[..., None] * np.array([0.05, 0.72, 1.0], dtype=np.float32)
    rgb += on_level[..., None] * np.array([1.0, 0.30, 0.06], dtype=np.float32)
    rgb += 0.025
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def _event_rgb(counts: np.ndarray, scale: float) -> np.ndarray:
    return _polarity_rgb(np.log1p(counts[0]), np.log1p(counts[1]), scale)


def _representation_bins(
    representation: torch.Tensor, temporal_bins: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    array = representation.detach().cpu().numpy()
    if array.shape[0] == 2:
        return [(array[0], array[1])]
    if array.shape[0] != 2 * temporal_bins:
        raise ValueError(
            f"expected 2 or {2 * temporal_bins} representation channels, got {array.shape[0]}"
        )
    return [(array[index], array[temporal_bins + index]) for index in range(temporal_bins)]


def _mask_rgb(
    base: np.ndarray,
    context_mask: np.ndarray,
    target_mask: np.ndarray,
    target_blocks: tuple[np.ndarray, ...],
    grid_size: tuple[int, int],
) -> np.ndarray:
    image = base.astype(np.float32)
    height, width = image.shape[:2]
    grid_height, grid_width = grid_size
    if context_mask.size != grid_height * grid_width or target_mask.size != context_mask.size:
        raise ValueError("mask size does not match the configured patch grid")
    context = context_mask.reshape(grid_height, grid_width)
    target = target_mask.reshape(grid_height, grid_width)
    context_color = np.array([42.0, 210.0, 112.0])
    unused_color = np.array([126.0, 132.0, 148.0])
    target_colors = (
        np.array([239.0, 77.0, 166.0]),
        np.array([181.0, 97.0, 255.0]),
        np.array([255.0, 102.0, 92.0]),
        np.array([255.0, 184.0, 76.0]),
    )
    for row in range(grid_height):
        y0 = round(row * height / grid_height)
        y1 = round((row + 1) * height / grid_height)
        for column in range(grid_width):
            x0 = round(column * width / grid_width)
            x1 = round((column + 1) * width / grid_width)
            flat_index = row * grid_width + column
            block_index = next(
                (
                    index
                    for index, block in enumerate(target_blocks)
                    if bool(block[flat_index])
                ),
                None,
            )
            if target[row, column]:
                color = target_colors[(block_index or 0) % len(target_colors)]
            elif context[row, column]:
                color = context_color
            else:
                color = unused_color
            image[y0:y1, x0:x1] = image[y0:y1, x0:x1] * 0.76 + color * 0.24
            image[y0 : min(y0 + 1, y1), x0:x1] = color
            image[max(y1 - 1, y0) : y1, x0:x1] = color
            image[y0:y1, x0 : min(x0 + 1, x1)] = color
            image[y0:y1, max(x1 - 1, x0) : x1] = color
    return np.clip(image, 0, 255).astype(np.uint8)


def _tail_inclusion(shorter: EventWindow, longer: EventWindow) -> bool:
    if shorter.t_end_us != longer.t_end_us or shorter.duration_us > longer.duration_us:
        return False
    left = int(np.searchsorted(longer.t_us, shorter.t_start_us, side="right"))
    for name in ("x", "y", "t_us", "polarity"):
        if not np.array_equal(getattr(shorter, name), getattr(longer, name)[left:]):
            return False
    return True


def _overlap_rgb(context: EventWindow, target: EventWindow) -> tuple[np.ndarray, str]:
    context_counts = _event_counts(context).sum(axis=0)
    target_counts = _event_counts(target).sum(axis=0)
    if context.duration_us <= target.duration_us:
        shorter, longer = context_counts, target_counts
        longer_name = "target"
    else:
        shorter, longer = target_counts, context_counts
        longer_name = "context"
    common = np.minimum(shorter, longer)
    added = np.maximum(longer - shorter, 0.0)
    inconsistent = np.maximum(shorter - longer, 0.0)
    transformed = [np.log1p(common), np.log1p(added), np.log1p(inconsistent)]
    scale = _display_scale(transformed)
    rgb = np.zeros((*common.shape, 3), dtype=np.float32)
    rgb += (transformed[0] / scale)[..., None] * np.array([0.18, 0.88, 0.42])
    rgb += (transformed[1] / scale)[..., None] * np.array([0.72, 0.28, 1.0])
    rgb += (transformed[2] / scale)[..., None] * np.array([1.0, 0.05, 0.05])
    rgb += 0.025
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8), longer_name


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = binascii.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def _png_data_uri(image: np.ndarray) -> str:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("PNG input must be an RGB uint8 array")
    height, width, _ = image.shape
    scanlines = b"".join(b"\x00" + row.tobytes() for row in np.ascontiguousarray(image))
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(scanlines, level=6))
        + _png_chunk(b"IEND", b"")
    )
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _sample_checks(
    sample: dict[str, Any],
    debug: PairedWindowDebugSample,
    config: ExperimentConfig,
    expected_dataset: str | None,
) -> list[Check]:
    x_context = sample["x_context"]
    x_target = sample["x_target"]
    context_mask = sample["context_mask"].numpy()
    target_mask = sample["target_mask"].numpy()
    if debug.context.duration_us <= debug.target.duration_us:
        inclusion = _tail_inclusion(debug.context, debug.target)
    else:
        inclusion = _tail_inclusion(debug.target, debug.context)
    num_patches = context_mask.size
    minimum_target = int(np.ceil(config.mask.target_area_range[0] * num_patches))
    maximum_target = int(np.floor(config.mask.target_area_range[1] * num_patches))
    checks = [
        Check(
            "テンソル形状",
            tuple(x_context.shape) == tuple(x_target.shape)
            and tuple(x_context.shape[-2:]) == config.model.image_size,
            f"context={tuple(x_context.shape)}, target={tuple(x_target.shape)}",
        ),
        Check(
            "有限・非負",
            bool(torch.isfinite(x_context).all() and torch.isfinite(x_target).all())
            and bool((x_context >= 0).all() and (x_target >= 0).all()),
            f"max={float(x_context.max()):.3f}/{float(x_target.max()):.3f}",
        ),
        Check(
            "空でない時間窓",
            debug.context.event_count > 0 and debug.target.event_count > 0,
            f"events={debug.context.event_count}/{debug.target.event_count}",
        ),
        Check(
            "終了時刻・幾何変換の共有",
            debug.context.t_end_us == debug.target.t_end_us
            and (debug.context.height, debug.context.width)
            == (debug.target.height, debug.target.width),
            f"t_end={debug.context.t_end_us}, size={debug.context.width}x{debug.context.height}",
        ),
        Check(
            "短時間窓の包含関係",
            inclusion,
            "短時間窓のraw eventが長時間窓の末尾と一致",
        ),
        Check(
            "maskの非重複",
            not bool(np.any(context_mask & target_mask)),
            f"context={int(context_mask.sum())}, target={int(target_mask.sum())}",
        ),
        Check(
            "maskのパッチ数",
            int(context_mask.sum()) == round(config.mask.context_keep_ratio * num_patches)
            and minimum_target <= int(target_mask.sum()) <= maximum_target,
            f"expected context={round(config.mask.context_keep_ratio * num_patches)}, "
            f"target={minimum_target}..{maximum_target}",
        ),
    ]
    if expected_dataset is not None:
        checks.insert(
            0,
            Check(
                "データセット種別",
                debug.sequence_info.dataset.lower() == expected_dataset.lower(),
                f"expected={expected_dataset}, actual={debug.sequence_info.dataset}",
            ),
        )
    return checks


def _sample_section(
    sample_index: int,
    sample: dict[str, Any],
    debug: PairedWindowDebugSample,
    config: ExperimentConfig,
    expected_dataset: str | None,
) -> tuple[str, dict[str, Any]]:
    context_counts = _event_counts(debug.context)
    target_counts = _event_counts(debug.target)
    raw_scale = _display_scale([np.log1p(context_counts), np.log1p(target_counts)])
    context_rgb = _event_rgb(context_counts, raw_scale)
    target_rgb = _event_rgb(target_counts, raw_scale)
    context_mask = sample["context_mask"].numpy()
    target_mask = sample["target_mask"].numpy()
    grid_size = (
        config.model.image_size[0] // config.model.patch_size,
        config.model.image_size[1] // config.model.patch_size,
    )
    mask_rgb = _mask_rgb(
        context_rgb,
        context_mask,
        target_mask,
        debug.masks.target_blocks,
        grid_size,
    )
    overlap_rgb, longer_name = _overlap_rgb(debug.context, debug.target)

    context_bins = _representation_bins(
        sample["x_context"], config.representation.temporal_bins
    )
    target_bins = _representation_bins(sample["x_target"], config.representation.temporal_bins)
    representation_scale = _display_scale(
        [array for pair in context_bins + target_bins for array in pair]
    )
    bin_panels: list[str] = []
    for label, bins in (("context", context_bins), ("target", target_bins)):
        for index, (off, on) in enumerate(bins):
            rgb = _polarity_rgb(off, on, representation_scale)
            bin_panels.append(
                '<figure><img loading="lazy" src="{}" alt="{} bin {}">'
                "<figcaption>{} bin {}</figcaption></figure>".format(
                    _png_data_uri(rgb), label, index, label, index
                )
            )

    checks = _sample_checks(sample, debug, config, expected_dataset)
    passed = all(check.passed for check in checks)
    check_html = "".join(
        '<li class="{}"><span>{}</span><small>{}</small></li>'.format(
            "pass" if check.passed else "fail",
            html.escape(check.name),
            html.escape(check.detail),
        )
        for check in checks
    )
    params = debug.spatial_transform
    section = f"""
    <section class="sample {'ok' if passed else 'problem'}">
      <h2>sample {sample_index} <span>{'合格' if passed else '要確認'}</span></h2>
      <div class="metadata">
        <code>{html.escape(str(sample['sequence_id']))}</code>
        <b>{float(sample['dt_context_ms']):g} ms → {float(sample['dt_target_ms']):g} ms</b>
        <span>events {debug.context.event_count:,} / {debug.target.event_count:,}</span>
        <span>crop x={params.x0}, y={params.y0}, flip={str(params.horizontal_flip).lower()}</span>
      </div>
      <ul class="checks">{check_html}</ul>
      <div class="main-grid">
        <figure>
          <img src="{_png_data_uri(context_rgb)}" alt="context event image">
          <figcaption>context — OFF: cyan / ON: orange</figcaption>
        </figure>
        <figure>
          <img src="{_png_data_uri(target_rgb)}" alt="target event image">
          <figcaption>target — contextと共通の表示スケール</figcaption>
        </figure>
        <figure>
          <img src="{_png_data_uri(overlap_rgb)}" alt="window overlap">
          <figcaption>共通: green / {longer_name}の追加分: violet / 不整合: red</figcaption>
        </figure>
        <figure>
          <img src="{_png_data_uri(mask_rgb)}" alt="patch mask">
          <figcaption>context: green / target block: multicolor / 未使用: gray</figcaption>
        </figure>
      </div>
      <h3>時間表現bin</h3>
      <div class="bin-grid">{''.join(bin_panels)}</div>
    </section>
    """
    record = {
        "index": sample_index,
        "passed": passed,
        "sequence_id": str(sample["sequence_id"]),
        "dataset": debug.sequence_info.dataset,
        "t_end_us": int(sample["t_end_us"]),
        "context_ms": float(sample["dt_context_ms"]),
        "target_ms": float(sample["dt_target_ms"]),
        "context_events": debug.context.event_count,
        "target_events": debug.target.event_count,
        "crop": {
            "x0": params.x0,
            "y0": params.y0,
            "height": params.output_height,
            "width": params.output_width,
            "horizontal_flip": params.horizontal_flip,
        },
        "checks": [check.__dict__ for check in checks],
    }
    return section, record


def write_inspection_report(
    dataset: PairedWindowDataset,
    config: ExperimentConfig,
    output: Path,
    *,
    samples: int = 8,
    start_index: int = 0,
    epoch: int = 0,
    expected_dataset: str | None = None,
) -> dict[str, Any]:
    if samples <= 0:
        raise ValueError("samples must be positive")
    if start_index < 0 or start_index + samples > len(dataset):
        raise ValueError("requested sample range is outside the dataset")
    dataset.set_epoch(epoch)
    sections: list[str] = []
    records: list[dict[str, Any]] = []
    for index in range(start_index, start_index + samples):
        sample, debug = dataset.sample_with_debug(index)
        section, record = _sample_section(
            index,
            sample,
            debug,
            config,
            expected_dataset,
        )
        sections.append(section)
        records.append(record)

    passed = all(record["passed"] for record in records)
    report = {
        "passed": passed,
        "config": config.to_dict(),
        "epoch": epoch,
        "start_index": start_index,
        "expected_dataset": expected_dataset,
        "samples": records,
    }
    document = f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Event Window-JEPA サンプル検査</title>
<style>
:root {{
  color-scheme: dark; font-family: ui-sans-serif, system-ui, sans-serif;
  background:#0a0c11; color:#eef1f7;
}}
body {{ max-width:1440px; margin:0 auto; padding:28px; }}
header {{ margin-bottom:24px; }} h1 {{ margin:0 0 8px; font-size:28px; }}
header p {{ color:#aeb6c7; margin:4px 0; }}
.sample {{ border:1px solid #2b3240; padding:20px; margin:0 0 28px; background:#10141c; }}
.sample.problem {{ border-color:#d94a5c; }}
h2 {{ display:flex; justify-content:space-between; margin:0 0 12px; }}
h2 span {{ font-size:13px; color:#65df91; }} .problem h2 span {{ color:#ff697a; }}
.metadata {{ display:flex; flex-wrap:wrap; gap:10px 20px; color:#c8cfdb; margin-bottom:14px; }}
.checks {{
  display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr));
  gap:7px 18px; padding:0; list-style:none;
}}
.checks li {{
  display:flex; justify-content:space-between; gap:10px;
  border-bottom:1px solid #252b37; padding:5px 0;
}}
.checks li::before {{ content:'✓'; color:#65df91; }}
.checks li.fail::before {{ content:'×'; color:#ff697a; }}
.checks small {{ color:#8f99ac; text-align:right; }}
.main-grid {{
  display:grid; grid-template-columns:repeat(2,minmax(0,1fr));
  gap:14px; margin-top:18px;
}}
.bin-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; }}
figure {{ margin:0; }}
img {{
  display:block; width:100%; height:auto; image-rendering:pixelated;
  background:#05070a; border:1px solid #303849;
}}
figcaption {{ color:#aeb6c7; font-size:12px; margin-top:5px; }}
h3 {{ margin:22px 0 10px; font-size:15px; }}
@media(max-width:700px) {{ body {{ padding:14px; }} .main-grid {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<header><h1>Event Window-JEPA サンプル検査</h1>
<p>{html.escape(str(config.data.manifest))}</p>
<p>
  epoch={epoch}, indices={start_index}..{start_index + samples - 1},
  overall={'合格' if passed else '要確認'}
</p></header>
{''.join(sections)}
</body>
</html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    args = _parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    output = args.output or Path(config.runtime.output_dir) / "inspection" / "samples.html"
    dataset = build_dataset(config)
    report = write_inspection_report(
        dataset,
        config,
        output,
        samples=args.samples,
        start_index=args.start_index,
        epoch=args.epoch,
        expected_dataset=args.expected_dataset,
    )
    print(
        f"[window-jepa] inspection {'passed' if report['passed'] else 'needs attention'}: "
        f"{output.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
