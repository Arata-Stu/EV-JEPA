from __future__ import annotations

import argparse
import base64
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import torch

from event_window_jepa.config import ExperimentConfig
from event_window_jepa.data.recurrent_window_dataset import (
    RecurrentWindowDataset,
    RecurrentWindowDebugSample,
)
from event_window_jepa.data.sequence_sampler import (
    MixedRecurrentBatchSampler,
    RecurrentClipRequest,
)
from event_window_jepa.data.spatial_transforms import SpatialTransformParameters
from event_window_jepa.inspection import (
    Check,
    _display_scale,
    _event_counts,
    _event_rgb,
    _mask_rgb,
    _png_data_uri,
    _polarity_rgb,
    _representation_bins,
)
from event_window_jepa.train.pretrain import (
    build_dataset,
    build_recurrent_batch_sampler,
)


OPTIONAL_SAMPLING_KEYS = (
    "sampling_mode",
    "stream_id",
    "state_reset",
    "augmentation_id",
)


@dataclass(frozen=True)
class MixedRecurrentInspection:
    """Two adjacent mixed batches and their fully materialized dataset items."""

    requests: tuple[
        tuple[RecurrentClipRequest, ...],
        tuple[RecurrentClipRequest, ...],
    ]
    samples: tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]
    debug_samples: tuple[
        tuple[RecurrentWindowDebugSample, ...],
        tuple[RecurrentWindowDebugSample, ...],
    ]
    checks: tuple[Check, ...]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render and validate one recurrent R0 event clip"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="dataset index, or local item index in the first mixed batch",
    )
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--expected-dataset", type=str, default=None)
    return parser.parse_args()


def _crop_record(params: SpatialTransformParameters) -> dict[str, Any]:
    return {
        "x0": params.x0,
        "y0": params.y0,
        "height": params.output_height,
        "width": params.output_width,
        "horizontal_flip": params.horizontal_flip,
    }


def _expected_control_masks(
    config: ExperimentConfig, total_steps: int
) -> tuple[list[bool], list[bool]]:
    burn_in = config.recurrent.burn_in_steps
    if total_steps != burn_in + config.recurrent.sequence_length:
        raise ValueError("clip length disagrees with recurrent configuration")
    loss_mask = [False] * burn_in + [True] * config.recurrent.sequence_length
    detach_mask = [False] * total_steps
    detach_mask[burn_in] = True
    for index in range(
        burn_in + config.recurrent.tbptt_steps,
        total_steps,
        config.recurrent.tbptt_steps,
    ):
        detach_mask[index] = True
    return loss_mask, detach_mask


def _json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, np.ndarray):
        return value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sampling_metadata_by_step(
    sample: dict[str, Any], total_steps: int
) -> dict[str, list[Any]]:
    """Normalize future random/stream sampler metadata to one value per step."""

    result: dict[str, list[Any]] = {}
    for key in OPTIONAL_SAMPLING_KEYS:
        if key not in sample:
            continue
        value = _json_value(sample[key])
        if isinstance(value, (list, tuple)):
            if len(value) != total_steps:
                raise ValueError(f"optional sample field {key} must be scalar or length T")
            result[key] = [_json_value(item) for item in value]
        else:
            result[key] = [value] * total_steps
    return result


def _sampling_metadata_check(metadata: dict[str, list[Any]]) -> Check | None:
    if not metadata:
        return None

    failures: list[str] = []
    mode_values = metadata.get("sampling_mode")
    mode: str | None = None
    if mode_values is not None:
        modes = {str(value) for value in mode_values}
        if len(modes) != 1 or not modes.issubset({"random", "stream"}):
            failures.append(f"sampling_mode={sorted(modes)}")
        else:
            mode = next(iter(modes))

    reset_values = metadata.get("state_reset")
    if reset_values is not None and not all(
        isinstance(value, bool) for value in reset_values
    ):
        failures.append("state_reset must be boolean")
    if mode == "random" and (not reset_values or reset_values[0] is not True):
        failures.append("random sample must reset state at clip start")

    stream_values = metadata.get("stream_id")
    if mode == "stream":
        if stream_values is None or len({str(value) for value in stream_values}) != 1:
            failures.append("stream sample requires one stream_id")
        elif not str(stream_values[0]):
            failures.append("stream_id cannot be empty")

    augmentation_values = metadata.get("augmentation_id")
    if augmentation_values is not None and len(
        {json.dumps(value, sort_keys=True) for value in augmentation_values}
    ) != 1:
        failures.append("augmentation_id must be shared across the clip")
    return Check(
        "random・stream sampler metadata",
        not failures,
        "; ".join(failures) if failures else json.dumps(metadata, ensure_ascii=False),
    )


def recurrent_clip_checks(
    sample: dict[str, Any],
    debug: RecurrentWindowDebugSample,
    config: ExperimentConfig,
    expected_dataset: str | None = None,
) -> tuple[Check, ...]:
    """Return machine-readable invariants for one recurrent dataset item."""

    x = sample["x"]
    timestamps = sample["t_end_us"].detach().cpu().numpy().astype(np.int64)
    durations = sample["dt_ms"].detach().cpu().numpy()
    context_mask = sample["context_mask"].detach().cpu().numpy()
    target_mask = sample["target_mask"].detach().cpu().numpy()
    loss_mask = sample["loss_mask"].detach().cpu().tolist()
    detach_mask = sample["detach_mask"].detach().cpu().tolist()
    total_steps = config.recurrent.burn_in_steps + config.recurrent.sequence_length
    expected_loss, expected_detach = _expected_control_masks(config, total_steps)
    sampling_metadata = _sampling_metadata_by_step(sample, total_steps)

    sequence_ids = (
        set(debug.sequence_ids)
        | {str(sample["sequence_id"])}
        | {debug.clip.sequence_id, debug.sequence_info.sequence_id}
    )
    debug_timestamps = np.asarray(
        [window.t_end_us for window in debug.windows], dtype=np.int64
    )
    strictly_increasing = bool(
        timestamps.size == total_steps
        and (timestamps.size == 1 or np.all(timestamps[1:] > timestamps[:-1]))
        and np.array_equal(timestamps, debug_timestamps)
        and tuple(int(value) for value in timestamps) == debug.clip.t_end_us
    )

    fifty_ms_us = 50_000
    adjacent_intervals = all(
        previous.t_end_us == current.t_start_us
        for previous, current in zip(debug.windows, debug.windows[1:], strict=False)
    )
    boundary_events_are_disjoint = all(
        (previous.t_us.size == 0 or bool(np.all(previous.t_us <= previous.t_end_us)))
        and (current.t_us.size == 0 or bool(np.all(current.t_us > previous.t_end_us)))
        for previous, current in zip(debug.windows, debug.windows[1:], strict=False)
    )
    nonoverlap_50ms = bool(
        config.recurrent.window_ms == 50.0
        and config.recurrent.stride_ms == 50.0
        and all(window.duration_us == fifty_ms_us for window in debug.windows)
        and (timestamps.size == 1 or np.all(np.diff(timestamps) == fifty_ms_us))
        and adjacent_intervals
        and boundary_events_are_disjoint
    )

    transforms_shared = bool(
        len(debug.spatial_transforms) == total_steps
        and all(
            params == debug.spatial_transform for params in debug.spatial_transforms
        )
        and all(
            (window.height, window.width)
            == (
                debug.spatial_transform.output_height,
                debug.spatial_transform.output_width,
            )
            for window in debug.windows
        )
    )
    num_patches = (
        config.model.image_size[0] // config.model.patch_size
    ) * (config.model.image_size[1] // config.model.patch_size)
    masks_valid = bool(
        context_mask.shape
        == target_mask.shape
        == (total_steps, num_patches)
        and not np.any(context_mask & target_mask)
    )
    checks = [
        Check(
            "時系列テンソル形状",
            tuple(x.shape)
            == (
                total_steps,
                config.representation.channels,
                *config.model.image_size,
            )
            and len(debug.windows) == total_steps
            and len(debug.masks) == total_steps,
            f"x={tuple(x.shape)}, T={total_steps}",
        ),
        Check(
            "有限・非負表現",
            bool(torch.isfinite(x).all()) and bool((x >= 0).all()),
            f"min={float(x.min()):.3f}, max={float(x.max()):.3f}",
        ),
        Check(
            "sequence ID単一",
            len(sequence_ids) == 1 and len(debug.sequence_ids) == total_steps,
            f"ids={sorted(sequence_ids)}",
        ),
        Check(
            "終了timestamp厳密昇順",
            strictly_increasing,
            f"t_end_us={timestamps.tolist()}",
        ),
        Check(
            "50 ms非重複区間",
            nonoverlap_50ms,
            "各区間は(previous_end, current_end]で隣接",
        ),
        Check(
            "dtとEventWindowの一致",
            bool(
                durations.shape == (total_steps,)
                and np.allclose(durations, 50.0, rtol=0.0, atol=1e-6)
            ),
            f"dt_ms={durations.tolist()}",
        ),
        Check(
            "crop・flipの全step共有",
            transforms_shared,
            json.dumps(_crop_record(debug.spatial_transform), ensure_ascii=False),
        ),
        Check(
            "JEPA mask形状・非重複",
            masks_valid,
            f"context/target={context_mask.shape}",
        ),
        Check(
            "burn-in・TBPTT制御mask",
            loss_mask == expected_loss and detach_mask == expected_detach,
            f"loss={loss_mask}, detach={detach_mask}",
        ),
    ]
    sampling_check = _sampling_metadata_check(sampling_metadata)
    if sampling_check is not None:
        checks.append(sampling_check)
    if expected_dataset is not None:
        checks.insert(
            0,
            Check(
                "データセット種別",
                debug.sequence_info.dataset.lower() == expected_dataset.lower(),
                f"expected={expected_dataset}, actual={debug.sequence_info.dataset}",
            ),
        )
    return tuple(checks)


def assert_recurrent_clip_invariants(
    sample: dict[str, Any],
    debug: RecurrentWindowDebugSample,
    config: ExperimentConfig,
    expected_dataset: str | None = None,
) -> tuple[Check, ...]:
    """Raise with all failed invariants, suitable for preprocessing smoke tests."""

    checks = recurrent_clip_checks(sample, debug, config, expected_dataset)
    failed = [check for check in checks if not check.passed]
    if failed:
        details = "; ".join(f"{check.name}: {check.detail}" for check in failed)
        raise AssertionError(details)
    return checks


def _materialized_mixed_metadata(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "sampling_mode": str(sample["sampling_mode"]),
        "stream_id": str(sample["stream_id"]),
        "state_reset": bool(_json_value(sample["state_reset"])),
        "sequence_id": str(sample["sequence_id"]),
        "t_end_us": tuple(int(value) for value in _json_value(sample["t_end_us"])),
        "augmentation_seed": int(_json_value(sample["augmentation_seed"])),
        "augmentation_id": str(sample["augmentation_id"]),
        "mask_seed": int(_json_value(sample["mask_seed"])),
        "mask_step_seeds": tuple(
            int(value) for value in _json_value(sample["mask_step_seeds"])
        ),
    }


def inspect_mixed_recurrent_batches(
    dataset: RecurrentWindowDataset,
    batch_sampler: MixedRecurrentBatchSampler,
    config: ExperimentConfig,
    *,
    epoch: int = 0,
    expected_dataset: str | None = None,
) -> MixedRecurrentInspection:
    """Materialize and validate the first two adjacent mixed batches."""

    if epoch < 0:
        raise ValueError("epoch cannot be negative")
    if len(batch_sampler) < 2:
        raise ValueError("mixed inspection requires at least two complete batches")
    dataset.set_epoch(epoch)
    batch_sampler.set_epoch(epoch)
    iterator = iter(batch_sampler)
    requests = (tuple(next(iterator)), tuple(next(iterator)))
    materialized = tuple(
        tuple(dataset.sample_with_debug(request) for request in batch)
        for batch in requests
    )
    samples = (materialized[0], materialized[1])
    sample_batches = (
        tuple(item[0] for item in samples[0]),
        tuple(item[0] for item in samples[1]),
    )
    debug_batches = (
        tuple(item[1] for item in samples[0]),
        tuple(item[1] for item in samples[1]),
    )
    metadata_batches = (
        tuple(_materialized_mixed_metadata(sample) for sample in sample_batches[0]),
        tuple(_materialized_mixed_metadata(sample) for sample in sample_batches[1]),
    )

    stream_count = batch_sampler.stream_batch_size
    random_count = batch_sampler.random_batch_size
    expected_modes = ("stream",) * stream_count + ("random",) * random_count
    actual_modes = tuple(
        tuple(str(metadata["sampling_mode"]) for metadata in batch)
        for batch in metadata_batches
    )
    ratio_valid = all(
        len(batch) == batch_sampler.batch_size
        and tuple(str(metadata["sampling_mode"]) for metadata in batch)
        == expected_modes
        for batch in metadata_batches
    )

    stream_metadata_pairs = tuple(
        (metadata_batches[0][lane], metadata_batches[1][lane])
        for lane in range(stream_count)
    )
    stream_debug_pairs = tuple(
        (debug_batches[0][lane], debug_batches[1][lane])
        for lane in range(stream_count)
    )
    stream_ids_fixed = all(
        bool(first["stream_id"]) and first["stream_id"] == second["stream_id"]
        for first, second in stream_metadata_pairs
    )
    stream_continuity = tuple(
        first["sequence_id"] == second["sequence_id"]
        and second["t_end_us"][0]
        == first["t_end_us"][-1] + batch_sampler.stride_us
        for first, second in stream_metadata_pairs
    )
    first_batch_resets = all(
        bool(metadata["state_reset"]) for metadata in metadata_batches[0]
    )
    stream_reset_contract = all(
        bool(second["state_reset"]) != is_continuous
        for (_, second), is_continuous in zip(
            stream_metadata_pairs, stream_continuity, strict=True
        )
    )
    augmentation_contract = all(
        bool(second["state_reset"])
        or (
            first["augmentation_seed"] == second["augmentation_seed"]
            and first["augmentation_id"] == second["augmentation_id"]
            and first_debug.spatial_transform == second_debug.spatial_transform
        )
        for (first, second), (first_debug, second_debug) in zip(
            stream_metadata_pairs, stream_debug_pairs, strict=True
        )
    )
    random_resets = all(
        bool(metadata["state_reset"])
        for batch in metadata_batches
        for metadata in batch[stream_count:]
    )
    masks_independent = all(
        first["mask_seed"] != second["mask_seed"]
        and len(set(first["mask_step_seeds"])) == dataset.total_steps
        and len(set(second["mask_step_seeds"])) == dataset.total_steps
        for first, second in stream_metadata_pairs
    )

    request_mismatches: list[str] = []
    for batch_index, (request_batch, metadata_batch, debug_batch) in enumerate(
        zip(requests, metadata_batches, debug_batches, strict=True)
    ):
        for item_index, (request, metadata, debug) in enumerate(
            zip(request_batch, metadata_batch, debug_batch, strict=True)
        ):
            expected = {
                "sampling_mode": request.sampling_mode,
                "stream_id": request.stream_id,
                "state_reset": request.state_reset,
                "sequence_id": request.clip.sequence_id,
                "t_end_us": request.clip.t_end_us,
                "augmentation_seed": request.augmentation_seed,
                "augmentation_id": request.augmentation_id,
                "mask_seed": request.mask_seed,
            }
            actual = {key: metadata[key] for key in expected}
            if (
                actual != expected
                or debug.request != request
                or debug.clip != request.clip
                or tuple(debug.mask_step_seeds) != metadata["mask_step_seeds"]
            ):
                request_mismatches.append(f"batch={batch_index}, item={item_index}")

    failed_clip_checks: list[str] = []
    for batch_index, (sample_batch, debug_batch) in enumerate(
        zip(sample_batches, debug_batches, strict=True)
    ):
        for item_index, (sample, debug) in enumerate(
            zip(sample_batch, debug_batch, strict=True)
        ):
            failed = [
                check.name
                for check in recurrent_clip_checks(
                    sample, debug, config, expected_dataset
                )
                if not check.passed
            ]
            if failed:
                failed_clip_checks.append(
                    f"batch={batch_index}, item={item_index}: {','.join(failed)}"
                )

    checks = (
        Check(
            "mixed batch内stream・random比固定",
            ratio_valid,
            f"expected={list(expected_modes)}, actual={actual_modes}",
        ),
        Check(
            "sampler request・Dataset出力・debug一致",
            not request_mismatches,
            ", ".join(request_mismatches) if request_mismatches else "all items match",
        ),
        Check(
            "epoch先頭batchは全lane reset",
            first_batch_resets,
            f"state_reset={[item['state_reset'] for item in metadata_batches[0]]}",
        ),
        Check(
            "stream lane ID固定",
            stream_ids_fixed,
            str(
                [
                    (first["stream_id"], second["stream_id"])
                    for first, second in stream_metadata_pairs
                ]
            ),
        ),
        Check(
            "stream batch間timestamp・reset整合",
            stream_reset_contract,
            str(
                [
                    {
                        "previous_end": first["t_end_us"][-1],
                        "current_start": second["t_end_us"][0],
                        "state_reset": second["state_reset"],
                        "continuous": is_continuous,
                    }
                    for (first, second), is_continuous in zip(
                        stream_metadata_pairs, stream_continuity, strict=True
                    )
                ]
            ),
        ),
        Check(
            "stream継続中のaugmentation・transform固定",
            augmentation_contract,
            str(
                [
                    {
                        "previous": first["augmentation_id"],
                        "current": second["augmentation_id"],
                        "state_reset": second["state_reset"],
                    }
                    for first, second in stream_metadata_pairs
                ]
            ),
        ),
        Check(
            "random clipは毎回state reset",
            random_resets,
            str(
                [
                    metadata["state_reset"]
                    for batch in metadata_batches
                    for metadata in batch[stream_count:]
                ]
            ),
        ),
        Check(
            "mixed mask seedはchunk・step独立",
            masks_independent,
            str(
                [
                    (first["mask_seed"], second["mask_seed"])
                    for first, second in stream_metadata_pairs
                ]
            ),
        ),
        Check(
            "mixed全clipの内部不変条件",
            not failed_clip_checks,
            "; ".join(failed_clip_checks) if failed_clip_checks else "all clips passed",
        ),
    )
    return MixedRecurrentInspection(
        requests=requests,
        samples=sample_batches,
        debug_samples=debug_batches,
        checks=checks,
    )


def _mixed_inspection_record(
    inspection: MixedRecurrentInspection,
) -> list[dict[str, Any]]:
    batches: list[dict[str, Any]] = []
    for batch_index, (sample_batch, debug_batch) in enumerate(
        zip(inspection.samples, inspection.debug_samples, strict=True)
    ):
        items = []
        for item_index, (sample, debug) in enumerate(
            zip(sample_batch, debug_batch, strict=True)
        ):
            metadata = _materialized_mixed_metadata(sample)
            items.append(
                {
                    "item_index": item_index,
                    "sampling_mode": metadata["sampling_mode"],
                    "stream_id": metadata["stream_id"],
                    "state_reset": metadata["state_reset"],
                    "sequence_id": metadata["sequence_id"],
                    "t_end_us": list(metadata["t_end_us"]),
                    "augmentation_seed": metadata["augmentation_seed"],
                    "augmentation_id": metadata["augmentation_id"],
                    "mask_seed": metadata["mask_seed"],
                    "mask_step_seeds": list(metadata["mask_step_seeds"]),
                    "spatial_transform": _crop_record(debug.spatial_transform),
                }
            )
        batches.append({"batch_index": batch_index, "items": items})
    return batches


def _mixed_inspection_html(inspection: MixedRecurrentInspection) -> str:
    rows: list[str] = []
    for batch_index, (sample_batch, debug_batch) in enumerate(
        zip(inspection.samples, inspection.debug_samples, strict=True)
    ):
        for item_index, (sample, debug) in enumerate(
            zip(sample_batch, debug_batch, strict=True)
        ):
            metadata = _materialized_mixed_metadata(sample)
            params = debug.spatial_transform
            rows.append(
                "<tr>"
                f"<td>{batch_index}</td><td>{item_index}</td>"
                f"<td>{html.escape(metadata['sampling_mode'])}</td>"
                f"<td>{html.escape(metadata['stream_id'] or '—')}</td>"
                f"<td>{str(metadata['state_reset']).lower()}</td>"
                f"<td>{html.escape(metadata['sequence_id'])}</td>"
                f"<td>{metadata['t_end_us'][0]:,}–{metadata['t_end_us'][-1]:,}</td>"
                f"<td>{html.escape(metadata['augmentation_id'])}</td>"
                f"<td>x={params.x0}, y={params.y0}, "
                f"flip={str(params.horizontal_flip).lower()}</td>"
                "</tr>"
            )
    return (
        '<section class="mixed"><h2>mixed sampler — 連続2 batch</h2>'
        "<div class=table-wrap><table><thead><tr>"
        "<th>batch</th><th>item</th><th>mode</th><th>stream</th>"
        "<th>reset</th><th>sequence</th><th>t_end μs</th>"
        "<th>augmentation</th><th>transform</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>"
    )


def _write_png(path: Path, image: np.ndarray) -> None:
    encoded = _png_data_uri(image).partition(",")[2]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(encoded))


def _asset_reference(output: Path, asset_path: Path) -> str:
    relative = asset_path.relative_to(output.parent)
    return "/".join(quote(part) for part in relative.parts)


def write_recurrent_inspection_report(
    dataset: RecurrentWindowDataset,
    config: ExperimentConfig,
    output: Path,
    *,
    sample_index: int = 0,
    epoch: int = 0,
    expected_dataset: str | None = None,
    mixed_batch_sampler: MixedRecurrentBatchSampler | None = None,
) -> dict[str, Any]:
    """Save clip images plus optional adjacent mixed-batch diagnostics."""

    if not config.recurrent.sequence_loader:
        raise ValueError(
            "sequence inspection requires recurrent.sequence_loader=true"
        )
    if output.suffix.lower() != ".html":
        raise ValueError("recurrent inspection output must use an .html suffix")
    if epoch < 0:
        raise ValueError("epoch cannot be negative")

    mixed_inspection: MixedRecurrentInspection | None = None
    if mixed_batch_sampler is None:
        if not 0 <= sample_index < len(dataset):
            raise ValueError("sample_index is outside the dataset")
        dataset.set_epoch(epoch)
        sample, debug = dataset.sample_with_debug(sample_index)
        checks = recurrent_clip_checks(sample, debug, config, expected_dataset)
    else:
        if not 0 <= sample_index < mixed_batch_sampler.batch_size:
            raise ValueError("sample_index is outside the per-rank mixed batch")
        mixed_inspection = inspect_mixed_recurrent_batches(
            dataset,
            mixed_batch_sampler,
            config,
            epoch=epoch,
            expected_dataset=expected_dataset,
        )
        sample = mixed_inspection.samples[0][sample_index]
        debug = mixed_inspection.debug_samples[0][sample_index]
        checks = (
            *recurrent_clip_checks(sample, debug, config, expected_dataset),
            *mixed_inspection.checks,
        )
    passed = all(check.passed for check in checks)
    output.parent.mkdir(parents=True, exist_ok=True)
    assets = output.parent / f"{output.stem}_assets"
    assets.mkdir(parents=True, exist_ok=True)

    event_counts = [_event_counts(window) for window in debug.windows]
    event_scale = _display_scale([np.log1p(value) for value in event_counts])
    representation_bins = [
        _representation_bins(sample["x"][index], config.representation.temporal_bins)
        for index in range(len(debug.windows))
    ]
    representation_scale = _display_scale(
        [array for bins in representation_bins for pair in bins for array in pair]
    )
    aggregate_pairs = [
        (
            np.sum([pair[0] for pair in bins], axis=0),
            np.sum([pair[1] for pair in bins], axis=0),
        )
        for bins in representation_bins
    ]
    aggregate_scale = _display_scale(
        [array for pair in aggregate_pairs for array in pair]
    )
    grid_size = (
        config.model.image_size[0] // config.model.patch_size,
        config.model.image_size[1] // config.model.patch_size,
    )
    sampling_metadata = _sampling_metadata_by_step(sample, len(debug.windows))
    sampling_summary = {
        key: values[0] if all(value == values[0] for value in values) else values
        for key, values in sampling_metadata.items()
    }
    sampling_header = "".join(
        f"<span>{html.escape(key)}={html.escape(str(value))}</span>"
        for key, value in sampling_summary.items()
    )

    records: list[dict[str, Any]] = []
    sections: list[str] = []
    for index, (window, mask, counts, bins, aggregate) in enumerate(
        zip(
            debug.windows,
            debug.masks,
            event_counts,
            representation_bins,
            aggregate_pairs,
            strict=True,
        )
    ):
        prefix = f"step-{index:02d}"
        event_path = assets / f"{prefix}-events.png"
        representation_path = assets / f"{prefix}-representation.png"
        overlay_path = assets / f"{prefix}-mask-overlay.png"
        event_rgb = _event_rgb(counts, event_scale)
        representation_rgb = _polarity_rgb(*aggregate, aggregate_scale)
        overlay_rgb = _mask_rgb(
            event_rgb,
            sample["context_mask"][index].numpy(),
            sample["target_mask"][index].numpy(),
            mask.target_blocks,
            grid_size,
        )
        _write_png(event_path, event_rgb)
        _write_png(representation_path, representation_rgb)
        _write_png(overlay_path, overlay_rgb)

        bin_paths: list[Path] = []
        bin_figures: list[str] = []
        for bin_index, (off, on) in enumerate(bins):
            bin_path = assets / f"{prefix}-representation-bin-{bin_index:02d}.png"
            _write_png(bin_path, _polarity_rgb(off, on, representation_scale))
            bin_paths.append(bin_path)
            bin_figures.append(
                '<figure><img loading="lazy" src="{}" alt="step {} bin {}">'
                "<figcaption>representation bin {}</figcaption></figure>".format(
                    _asset_reference(output, bin_path),
                    index,
                    bin_index,
                    bin_index,
                )
            )

        loss_enabled = bool(sample["loss_mask"][index])
        detach_enabled = bool(sample["detach_mask"][index])
        step_sampling = {
            key: values[index] for key, values in sampling_metadata.items()
        }
        step_sampling_html = "".join(
            f"<span>{html.escape(key)}={html.escape(str(value))}</span>"
            for key, value in step_sampling.items()
        )
        sections.append(
            f"""
            <section class="step">
              <h2>step {index:02d}</h2>
              <div class="metadata">
                <code>{html.escape(debug.sequence_ids[index])}</code>
                <code>{window.t_start_us:,} &lt; t ≤ {window.t_end_us:,} μs</code>
                <b>{float(sample['dt_ms'][index]):g} ms</b>
                <span>events={window.event_count:,}</span>
                <span>loss={str(loss_enabled).lower()}</span>
                <span>detach-before={str(detach_enabled).lower()}</span>
                {step_sampling_html}
              </div>
              <div class="main-grid">
                <figure><img src="{_asset_reference(output, event_path)}" alt="events">
                  <figcaption>raw events — OFF: cyan / ON: orange</figcaption></figure>
                <figure><img src="{_asset_reference(output, representation_path)}"
                  alt="representation">
                  <figcaption>event representation aggregate</figcaption></figure>
                <figure><img src="{_asset_reference(output, overlay_path)}" alt="mask overlay">
                  <figcaption>context: green / target: multicolor /
                    unused: gray</figcaption></figure>
              </div>
              <div class="bin-grid">{''.join(bin_figures)}</div>
            </section>
            """
        )
        records.append(
            {
                "index": index,
                "sequence_id": debug.sequence_ids[index],
                "t_start_us": window.t_start_us,
                "t_end_us": window.t_end_us,
                "dt_ms": float(sample["dt_ms"][index]),
                "event_count": window.event_count,
                "loss_mask": loss_enabled,
                "detach_mask": detach_enabled,
                "spatial_transform": _crop_record(debug.spatial_transforms[index]),
                "sampling": step_sampling,
                "mask": {
                    "context_indices": np.flatnonzero(
                        sample["context_mask"][index].numpy()
                    ).tolist(),
                    "target_indices": np.flatnonzero(
                        sample["target_mask"][index].numpy()
                    ).tolist(),
                    "target_blocks": [
                        np.flatnonzero(block).tolist() for block in mask.target_blocks
                    ],
                },
                "images": {
                    "events": str(event_path.relative_to(output.parent)),
                    "representation": str(
                        representation_path.relative_to(output.parent)
                    ),
                    "mask_overlay": str(overlay_path.relative_to(output.parent)),
                    "representation_bins": [
                        str(path.relative_to(output.parent)) for path in bin_paths
                    ],
                },
            }
        )

    check_html = "".join(
        '<li class="{}"><span>{}</span><small>{}</small></li>'.format(
            "pass" if check.passed else "fail",
            html.escape(check.name),
            html.escape(check.detail),
        )
        for check in checks
    )
    params = debug.spatial_transform
    mixed_html = (
        _mixed_inspection_html(mixed_inspection)
        if mixed_inspection is not None
        else ""
    )
    selection_label = "per-rank row" if mixed_inspection is not None else "sample"
    selected_mode = str(sample.get("sampling_mode", "clip"))
    document = f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Recurrent EV-JEPA clip検査</title>
<style>
:root {{ color-scheme:dark; font-family:ui-sans-serif,system-ui,sans-serif;
  background:#0a0c11; color:#eef1f7; }}
body {{ max-width:1500px; margin:0 auto; padding:28px; }}
header,.step,.mixed {{ border:1px solid #2b3240; padding:20px; margin:0 0 24px;
  background:#10141c; }}
header.problem {{ border-color:#d94a5c; }} h1,h2 {{ margin-top:0; }}
.metadata {{ display:flex; flex-wrap:wrap; gap:10px 20px; color:#c8cfdb; }}
.checks {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr));
  gap:7px 18px; padding:0; list-style:none; }}
.checks li {{ display:flex; justify-content:space-between; gap:12px;
  border-bottom:1px solid #252b37; padding:5px 0; }}
.checks li::before {{ content:'✓'; color:#65df91; }}
.checks li.fail::before {{ content:'×'; color:#ff697a; }}
.checks small {{ color:#8f99ac; text-align:right; }}
.main-grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr));
  gap:14px; margin-top:18px; }}
.bin-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
  gap:10px; margin-top:14px; }}
figure {{ margin:0; }} img {{ display:block; width:100%; height:auto;
  image-rendering:pixelated; background:#05070a; border:1px solid #303849; }}
figcaption {{ color:#aeb6c7; font-size:12px; margin-top:5px; }}
.table-wrap {{ overflow-x:auto; margin-top:14px; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
th,td {{ border-bottom:1px solid #2b3240; padding:8px; text-align:left;
  white-space:nowrap; }} th {{ color:#9ea9bd; }}
@media(max-width:800px) {{ body {{ padding:12px; }}
  .main-grid {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<header class="{'ok' if passed else 'problem'}">
  <h1>Recurrent EV-JEPA clip検査 — {'合格' if passed else '要確認'}</h1>
  <div class="metadata">
    <code>{html.escape(str(sample['sequence_id']))}</code>
    <span>{selection_label}={sample_index}, epoch={epoch}, T={len(debug.windows)}</span>
    <b>selected mode={html.escape(selected_mode)}</b>
    <span>crop x={params.x0}, y={params.y0}, {params.output_width}×{params.output_height}</span>
    <span>flip={str(params.horizontal_flip).lower()}</span>
    {sampling_header}
  </div>
  <ul class="checks">{check_html}</ul>
</header>
{mixed_html}
{''.join(sections)}
</body>
</html>
"""
    report = {
        "passed": passed,
        "config": config.to_dict(),
        "sample_index": sample_index,
        "epoch": epoch,
        "sequence_id": str(sample["sequence_id"]),
        "dataset": debug.sequence_info.dataset,
        "inspection_mode": (
            "mixed_two_batch" if mixed_inspection is not None else "single_clip"
        ),
        "spatial_transform": _crop_record(debug.spatial_transform),
        "sampling": sampling_summary,
        "checks": [check.__dict__ for check in checks],
        "timesteps": records,
        "mixed_batches": (
            _mixed_inspection_record(mixed_inspection)
            if mixed_inspection is not None
            else None
        ),
    }
    output.write_text(document, encoding="utf-8")
    output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    args = _parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    dataset = build_dataset(config)
    if not isinstance(dataset, RecurrentWindowDataset):
        raise ValueError("the inspection config must build a recurrent dataset")
    mixed_batch_sampler = (
        build_recurrent_batch_sampler(config, dataset, world_size=1, rank=0)
        if config.recurrent.sampling == "mixed"
        else None
    )
    output = args.output or (
        Path(config.runtime.output_dir) / "inspection" / "recurrent-clip.html"
    )
    report = write_recurrent_inspection_report(
        dataset,
        config,
        output,
        sample_index=args.sample_index,
        epoch=args.epoch,
        expected_dataset=args.expected_dataset,
        mixed_batch_sampler=mixed_batch_sampler,
    )
    print(
        f"[window-jepa] recurrent inspection "
        f"{'passed' if report['passed'] else 'failed'}: {output.resolve()}",
        flush=True,
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
