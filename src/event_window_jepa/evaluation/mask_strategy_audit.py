from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from event_window_jepa.config import ExperimentConfig
from event_window_jepa.train.pretrain import build_dataset


METRICS = (
    "context_active_ratio",
    "context_event_mass",
    "context_enrichment",
    "target_active_ratio",
    "target_event_mass",
    "target_enrichment",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare uniform and top-k event-enrichment masks on identical Gen1 samples "
            "without training a model."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--topk-fraction", type=float, default=0.25)
    parser.add_argument("--minimum-context-mass-lift", type=float, default=0.15)
    parser.add_argument("--minimum-target-mass-lift", type=float, default=0.05)
    parser.add_argument("--minimum-context-enrichment", type=float, default=1.15)
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.samples <= 0 or args.epoch < 0:
        raise ValueError("samples must be positive and epoch must be non-negative")
    if not 0 < args.topk_fraction <= 1:
        raise ValueError("topk-fraction must lie inside (0, 1]")
    if min(args.minimum_context_mass_lift, args.minimum_target_mass_lift) < 0:
        raise ValueError("minimum mass lifts cannot be negative")
    if args.minimum_context_enrichment <= 0:
        raise ValueError("minimum-context-enrichment must be positive")


def _float(value: Any) -> float:
    return float(value.detach().cpu()) if hasattr(value, "detach") else float(value)


def _sample_metrics(sample: dict[str, Any]) -> dict[str, float]:
    area_fraction = _float(sample["target_mask"].float().mean())
    context_mass = _float(sample["mask_context_event_mass_coverage"])
    target_mass = _float(sample["mask_target_event_mass_coverage"])
    return {
        "context_active_ratio": _float(sample["mask_context_active_patch_ratio"]),
        "context_event_mass": context_mass,
        "context_enrichment": context_mass / area_fraction,
        "target_active_ratio": _float(sample["mask_target_active_patch_ratio"]),
        "target_event_mass": target_mass,
        "target_enrichment": target_mass / area_fraction,
    }


def _distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "q05": float(np.quantile(values, 0.05)),
        "median": float(np.quantile(values, 0.50)),
        "q95": float(np.quantile(values, 0.95)),
    }


def _paired_comparison(random_values: np.ndarray, event_values: np.ndarray) -> dict[str, Any]:
    delta = event_values - random_values
    mean_delta = float(delta.mean())
    standard_error = (
        float(delta.std(ddof=1) / math.sqrt(len(delta))) if len(delta) > 1 else 0.0
    )
    random_mean = float(random_values.mean())
    relative_lift = (
        float(event_values.mean() / random_mean - 1.0) if random_mean > 0 else float("nan")
    )
    return {
        "random": _distribution(random_values),
        "event_aware": _distribution(event_values),
        "mean_delta": mean_delta,
        "relative_lift": relative_lift,
        "delta_ci95": [
            mean_delta - 1.96 * standard_error,
            mean_delta + 1.96 * standard_error,
        ],
        "event_aware_win_fraction": float(np.mean(event_values > random_values)),
    }


def _fmt(value: float, *, signed: bool = False) -> str:
    if not math.isfinite(value):
        return "-"
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def audit(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    source_config = ExperimentConfig.from_yaml(args.config)
    event_mask = replace(
        source_config.mask,
        activity_selection_strategy="topk_enrichment",
        activity_topk_fraction=args.topk_fraction,
    )
    if event_mask.activity_aware_probability <= 0:
        raise ValueError(
            "the audit config must set mask.activity_aware_probability above zero"
        )
    random_config = replace(
        source_config,
        mask=replace(event_mask, activity_aware_probability=0.0),
    )
    event_config = replace(source_config, mask=event_mask)

    random_dataset = build_dataset(random_config)
    event_dataset = build_dataset(event_config)
    if len(random_dataset) != len(event_dataset):
        raise RuntimeError("audit datasets do not share length")
    if args.samples > len(random_dataset):
        raise ValueError(
            f"requested {args.samples} samples but the configured epoch has "
            f"only {len(random_dataset)}"
        )
    random_dataset.set_epoch(args.epoch)
    event_dataset.set_epoch(args.epoch)

    collected = {
        "random": {metric: [] for metric in METRICS},
        "event_aware": {metric: [] for metric in METRICS},
    }
    aware_values: list[float] = []
    fallback_values: list[float] = []
    for index in range(args.samples):
        random_sample = random_dataset[index]
        event_sample = event_dataset[index]
        identity = ("sequence_id", "t_end_us", "dt_context_ms", "dt_target_ms")
        for key in identity:
            left = random_sample[key]
            right = event_sample[key]
            if str(left) != str(right):
                raise RuntimeError(f"paired audit sample differs at index={index}, key={key}")
        random_metrics = _sample_metrics(random_sample)
        event_metrics = _sample_metrics(event_sample)
        for metric in METRICS:
            collected["random"][metric].append(random_metrics[metric])
            collected["event_aware"][metric].append(event_metrics[metric])
        aware_values.append(_float(event_sample["mask_activity_aware"]))
        fallback_values.append(_float(event_sample["mask_activity_fallback"]))

    comparisons = {}
    for metric in METRICS:
        comparisons[metric] = _paired_comparison(
            np.asarray(collected["random"][metric], dtype=np.float64),
            np.asarray(collected["event_aware"][metric], dtype=np.float64),
        )

    aware_fraction = float(np.mean(aware_values))
    fallback_fraction = float(np.mean(fallback_values))
    checks = {
        "context_mass_lift": {
            "value": comparisons["context_event_mass"]["relative_lift"],
            "minimum": args.minimum_context_mass_lift,
        },
        "target_mass_lift": {
            "value": comparisons["target_event_mass"]["relative_lift"],
            "minimum": args.minimum_target_mass_lift,
        },
        "context_enrichment": {
            "value": comparisons["context_enrichment"]["event_aware"]["mean"],
            "minimum": args.minimum_context_enrichment,
        },
        "positive_context_mass_ci": {
            "value": comparisons["context_event_mass"]["delta_ci95"][0],
            "minimum": 0.0,
        },
        "aware_fraction": {
            "value": aware_fraction,
            "minimum": max(0.0, event_mask.activity_aware_probability - 0.05),
        },
    }
    for check in checks.values():
        check["passed"] = bool(check["value"] >= check["minimum"])
    passed = all(check["passed"] for check in checks.values())

    report = {
        "passed": passed,
        "config": str(args.config.resolve()),
        "epoch": args.epoch,
        "samples": args.samples,
        "activity_aware_probability": event_mask.activity_aware_probability,
        "activity_candidates": event_mask.activity_candidates,
        "activity_selection_strategy": event_mask.activity_selection_strategy,
        "activity_topk_fraction": event_mask.activity_topk_fraction,
        "aware_fraction": aware_fraction,
        "fallback_fraction": fallback_fraction,
        "checks": checks,
        "metrics": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Gen1 mask-strategy audit",
        "",
        f"Overall: {'PASS' if passed else 'FAIL'}",
        "",
        f"Samples: {args.samples}, epoch: {args.epoch}, top-k fraction: {args.topk_fraction}",
        "",
        "| Metric | Random mean | Event-aware mean | Relative lift | Delta 95% CI | Win rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric in METRICS:
        result = comparisons[metric]
        ci = result["delta_ci95"]
        lines.append(
            f"| {metric} | {_fmt(result['random']['mean'])} "
            f"| {_fmt(result['event_aware']['mean'])} "
            f"| {_fmt(result['relative_lift'], signed=True)} "
            f"| [{_fmt(ci[0], signed=True)}, {_fmt(ci[1], signed=True)}] "
            f"| {_fmt(result['event_aware_win_fraction'])} |"
        )
    lines.extend(
        [
            "",
            "## Acceptance checks",
            "",
            "| Check | Value | Minimum | Result |",
            "|---|---:|---:|---|",
        ]
    )
    for name, check in checks.items():
        lines.append(
            f"| {name} | {_fmt(check['value'])} | {_fmt(check['minimum'])} "
            f"| {'PASS' if check['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            f"Activity-aware fraction: {_fmt(aware_fraction)}",
            "",
            f"Fallback fraction: {_fmt(fallback_fraction)}",
        ]
    )
    markdown_path = args.output.with_suffix(".md")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"\nJSON: {args.output.resolve()}", flush=True)
    print(f"Markdown: {markdown_path.resolve()}", flush=True)
    return report


def main() -> None:
    args = _parse_args()
    report = audit(args)
    if args.require_pass and not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
