#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BASE_CONFIG="$PROJECT_ROOT/configs/pretrain/window_jepa21_vits_gen1.yaml"
RANDOM_CONFIG="$PROJECT_ROOT/configs/pretrain/window_jepa21_vits_gen1_random_mask_rerun_v2.yaml"
EVENT_CONFIG="$PROJECT_ROOT/configs/pretrain/window_jepa21_vits_gen1_event_aware_rerun_v2.yaml"
PRETRAIN_ROOT="$PROJECT_ROOT/outputs/pretrain/rerun_v2"
DETECTION_ROOT="$PROJECT_ROOT/outputs/downstream/rerun_v2"
DATA_ROOT=/media/noah-22/AT_SSD/dataset/evjepa/gen1_304x240
ACTION=all

usage() {
  printf '%s\n' \
    'Usage: bash scripts/experiments/run_gen1_mask_rerun_v2.sh [options]' \
    '' \
    '  --action ACTION   prepare, pretrain, detection, summary, or all (default: all)' \
    '  --data-root DIR   Gen1 processed dataset root' \
    '  -h, --help        Show this help' \
    '' \
    'The script intentionally refuses to train into an output directory that' \
    'already contains train.jsonl or checkpoint-latest.pt.'
}

while (($#)); do
  case "$1" in
    --action)
      ACTION=$2
      shift 2
      ;;
    --data-root)
      DATA_ROOT=$2
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$ACTION" in
  prepare|pretrain|detection|summary|all) ;;
  *)
    printf 'Unsupported action: %s\n' "$ACTION" >&2
    usage >&2
    exit 2
    ;;
esac

TRAIN_MANIFEST="$DATA_ROOT/manifests/train.jsonl"
VAL_MANIFEST="$DATA_ROOT/manifests/val.jsonl"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf 'Required command is unavailable: %s\n' "$1" >&2
    exit 1
  fi
}

require_file() {
  if [[ ! -f "$1" ]]; then
    printf 'Required file does not exist: %s\n' "$1" >&2
    exit 1
  fi
}

require_fresh_output() {
  local output_dir=$1
  if [[ -e "$output_dir/train.jsonl" || -e "$output_dir/checkpoint-latest.pt" ]]; then
    printf 'Refusing to mix with an existing run: %s\n' "$output_dir" >&2
    printf 'Move the existing directory or choose a new rerun name.\n' >&2
    exit 1
  fi
}

prepare_configs() {
  require_file "$BASE_CONFIG"
  mkdir -p "$(dirname "$RANDOM_CONFIG")"

  cp "$BASE_CONFIG" "$RANDOM_CONFIG"
  cp "$BASE_CONFIG" "$EVENT_CONFIG"

  perl -0pi -e \
    's/activity_aware_probability: 0\.70/activity_aware_probability: 0.0/' \
    "$RANDOM_CONFIG"
  perl -0pi -e \
    's#output_dir: outputs/pretrain/vjepa21_vits_gen1_event_aware_seed0#output_dir: outputs/pretrain/rerun_v2/random_mask_seed0#' \
    "$RANDOM_CONFIG"
  perl -0pi -e \
    's#output_dir: outputs/pretrain/vjepa21_vits_gen1_event_aware_seed0#output_dir: outputs/pretrain/rerun_v2/event_aware_seed0#' \
    "$EVENT_CONFIG"

  grep -q 'activity_aware_probability: 0.0' "$RANDOM_CONFIG"
  grep -q 'output_dir: outputs/pretrain/rerun_v2/random_mask_seed0' "$RANDOM_CONFIG"
  grep -q 'activity_aware_probability: 0.70' "$EVENT_CONFIG"
  grep -q 'output_dir: outputs/pretrain/rerun_v2/event_aware_seed0' "$EVENT_CONFIG"
  grep -q 'epochs: 100' "$RANDOM_CONFIG"
  grep -q 'epochs: 100' "$EVENT_CONFIG"

  printf 'Prepared and validated:\n  %s\n  %s\n' "$RANDOM_CONFIG" "$EVENT_CONFIG"
}

run_pretrain() {
  require_command window-jepa-pretrain
  require_fresh_output "$PRETRAIN_ROOT/random_mask_seed0"
  require_fresh_output "$PRETRAIN_ROOT/event_aware_seed0"
  mkdir -p "$PRETRAIN_ROOT/logs"

  (
    cd "$PROJECT_ROOT"
    PYTHONUNBUFFERED=1 window-jepa-pretrain --config "$RANDOM_CONFIG"
  ) 2>&1 | tee "$PRETRAIN_ROOT/logs/random_mask_seed0.log"

  (
    cd "$PROJECT_ROOT"
    PYTHONUNBUFFERED=1 window-jepa-pretrain --config "$EVENT_CONFIG"
  ) 2>&1 | tee "$PRETRAIN_ROOT/logs/event_aware_seed0.log"

  verify_pretrain_checkpoints
}

verify_pretrain_checkpoints() {
  local random_checkpoint="$PRETRAIN_ROOT/random_mask_seed0/checkpoint-latest.pt"
  local event_checkpoint="$PRETRAIN_ROOT/event_aware_seed0/checkpoint-latest.pt"
  require_file "$random_checkpoint"
  require_file "$event_checkpoint"

  PROJECT_ROOT_FOR_CHECK="$PROJECT_ROOT" python - <<'PY'
import os
from pathlib import Path

import torch

from event_window_jepa.config import ExperimentConfig
from event_window_jepa.train.checkpoint import config_hash

root = Path(os.environ["PROJECT_ROOT_FOR_CHECK"])
paths = (
    root / "outputs/pretrain/rerun_v2/random_mask_seed0/checkpoint-latest.pt",
    root / "outputs/pretrain/rerun_v2/event_aware_seed0/checkpoint-latest.pt",
)
for path in paths:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = ExperimentConfig.from_mapping(checkpoint["resolved_config"])
    if checkpoint.get("config_hash") != config_hash(config):
        raise RuntimeError(f"inconsistent checkpoint configuration metadata: {path}")
    if int(checkpoint.get("epoch", -1)) != 100:
        raise RuntimeError(f"pretraining did not reach epoch 100: {path}")
    print(
        f"verified: {path} "
        f"activity_aware_probability={config.mask.activity_aware_probability}"
    )
PY
}

run_detection_one() {
  local name=$1
  local checkpoint=$2
  local backbone_init=$3
  local unfreeze=$4
  local output_dir="$DETECTION_ROOT/$name"
  local command=(
    window-jepa-gen1-detect
    --checkpoint "$checkpoint"
    --train-manifest "$TRAIN_MANIFEST"
    --val-manifest "$VAL_MANIFEST"
    --output-dir "$output_dir"
    --backbone-init "$backbone_init"
    --window-ms 40
    --batch-size 8
    --workers 4
    --epochs 50
    --learning-rate 0.0002
    --eval-every 5
    --seed 0
    --precision fp32
  )

  if [[ "$unfreeze" == 1 ]]; then
    command+=(--unfreeze-backbone)
  fi

  require_fresh_output "$output_dir"
  mkdir -p "$DETECTION_ROOT/logs"
  (
    cd "$PROJECT_ROOT"
    PYTHONUNBUFFERED=1 "${command[@]}"
  ) 2>&1 | tee "$DETECTION_ROOT/logs/$name.log"
}

run_detection() {
  require_command window-jepa-gen1-detect
  require_file "$TRAIN_MANIFEST"
  require_file "$VAL_MANIFEST"
  verify_pretrain_checkpoints

  local random_checkpoint="$PRETRAIN_ROOT/random_mask_seed0/checkpoint-latest.pt"
  local event_checkpoint="$PRETRAIN_ROOT/event_aware_seed0/checkpoint-latest.pt"

  # Run the most important end-to-end comparisons first.
  run_detection_one random_mask_finetune_seed0 "$random_checkpoint" pretrained 1
  run_detection_one event_aware_finetune_seed0 "$event_checkpoint" pretrained 1
  run_detection_one random_scratch_seed0 "$random_checkpoint" random 1

  # Frozen-feature diagnostics follow after the primary comparisons.
  run_detection_one random_mask_frozen_seed0 "$random_checkpoint" pretrained 0
  run_detection_one event_aware_frozen_seed0 "$event_checkpoint" pretrained 0
  run_detection_one random_frozen_seed0 "$random_checkpoint" random 0

  printf 'All Gen1 detection reruns completed: %s\n' "$DETECTION_ROOT"
}

summarize_results() {
  mkdir -p "$DETECTION_ROOT"
  PROJECT_ROOT_FOR_SUMMARY="$PROJECT_ROOT" python - <<'PY'
import csv
import json
import math
import os
from pathlib import Path
from statistics import fmean

root = Path(os.environ["PROJECT_ROOT_FOR_SUMMARY"])
pretrain_root = root / "outputs/pretrain/rerun_v2"
detection_root = root / "outputs/downstream/rerun_v2"

pretrain_runs = {
    "Random mask": pretrain_root / "random_mask_seed0/train.jsonl",
    "Event-aware": pretrain_root / "event_aware_seed0/train.jsonl",
}
detection_runs = [
    ("Random mask", "Pretrained / Frozen", "random_mask_frozen_seed0"),
    ("Event-aware", "Pretrained / Frozen", "event_aware_frozen_seed0"),
    ("Random mask", "Pretrained / Fine-tune", "random_mask_finetune_seed0"),
    ("Event-aware", "Pretrained / Fine-tune", "event_aware_finetune_seed0"),
    ("Control", "Random / Frozen", "random_frozen_seed0"),
    ("Control", "Random / Scratch", "random_scratch_seed0"),
]
detection_metrics = ("AP", "AP_50", "AP_75", "AP_S", "AP_M", "AP_L")
pretrain_metrics = (
    "loss",
    "masked_loss",
    "prediction_std",
    "target_std",
    "mask_activity_aware_fraction",
    "mask_activity_fallback_fraction",
    "mask_target_active_patch_ratio",
    "mask_target_event_mass_coverage",
    "mask_empty_target_fraction",
)


def read_jsonl(path):
    if not path.is_file():
        raise FileNotFoundError(f"required result log does not exist: {path}")
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSON at {path}:{line_number}") from exc
    if not records:
        raise RuntimeError(f"result log is empty: {path}")
    return records


def fmt(value):
    if value is None or not math.isfinite(float(value)):
        return "-"
    return f"{float(value):.4f}"


pretrain_results = []
for strategy, path in pretrain_runs.items():
    records = read_jsonl(path)
    last_epoch_index = max(int(record["epoch"]) for record in records)
    last_epoch = [
        record for record in records if int(record["epoch"]) == last_epoch_index
    ]
    means = {}
    for metric in pretrain_metrics:
        values = [float(record[metric]) for record in last_epoch if metric in record]
        means[metric] = fmean(values) if values else float("nan")
    pretrain_results.append(
        {
            "strategy": strategy,
            "epoch": last_epoch_index + 1,
            "global_step": max(int(record["global_step"]) for record in last_epoch),
            "path": path,
            **means,
        }
    )

detection_results = []
for strategy, condition, directory in detection_runs:
    path = detection_root / directory / "train.jsonl"
    records = read_jsonl(path)
    evaluated = [record for record in records if record.get("validation", {}).get("AP") is not None]
    if not evaluated:
        raise RuntimeError(f"no validation result in: {path}")
    best = max(evaluated, key=lambda record: float(record["validation"]["AP"]))
    final = evaluated[-1]
    last_epoch = max(int(record["epoch"]) for record in records)
    detection_results.append(
        {
            "strategy": strategy,
            "condition": condition,
            "directory": directory,
            "path": path,
            "last_epoch": last_epoch,
            "best_epoch": int(best["epoch"]),
            "final_epoch": int(final["epoch"]),
            "best": {metric: float(best["validation"][metric]) for metric in detection_metrics},
            "final": {metric: float(final["validation"][metric]) for metric in detection_metrics},
        }
    )

pretrain_csv = detection_root / "pretrain_summary.csv"
with pretrain_csv.open("w", newline="", encoding="utf-8") as handle:
    fields = ["strategy", "epoch", "global_step", *pretrain_metrics, "log_path"]
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for result in pretrain_results:
        writer.writerow(
            {
                **{key: result[key] for key in fields if key not in {"log_path"}},
                "log_path": str(result["path"].relative_to(root)),
            }
        )

detection_csv = detection_root / "detection_summary.csv"
with detection_csv.open("w", newline="", encoding="utf-8") as handle:
    fields = [
        "strategy",
        "condition",
        "last_epoch",
        "best_epoch",
        *[f"best_{metric}" for metric in detection_metrics],
        "final_epoch",
        *[f"final_{metric}" for metric in detection_metrics],
        "best_at_final_evaluation",
        "log_path",
    ]
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for result in detection_results:
        row = {
            "strategy": result["strategy"],
            "condition": result["condition"],
            "last_epoch": result["last_epoch"],
            "best_epoch": result["best_epoch"],
            "final_epoch": result["final_epoch"],
            "best_at_final_evaluation": result["best_epoch"] == result["final_epoch"],
            "log_path": str(result["path"].relative_to(root)),
        }
        row.update({f"best_{key}": value for key, value in result["best"].items()})
        row.update({f"final_{key}": value for key, value in result["final"].items()})
        writer.writerow(row)

lines = [
    "# Gen1 mask-strategy rerun v2 summary",
    "",
    "## Pretrain final-epoch averages",
    "",
    "| Strategy | Epoch | Loss | Masked loss | Prediction std | Target std | "
    "Aware fraction | Target active ratio | Event-mass coverage | Empty target |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for result in pretrain_results:
    lines.append(
        f"| {result['strategy']} | {result['epoch']} "
        f"| {fmt(result['loss'])} | {fmt(result['masked_loss'])} "
        f"| {fmt(result['prediction_std'])} | {fmt(result['target_std'])} "
        f"| {fmt(result['mask_activity_aware_fraction'])} "
        f"| {fmt(result['mask_target_active_patch_ratio'])} "
        f"| {fmt(result['mask_target_event_mass_coverage'])} "
        f"| {fmt(result['mask_empty_target_fraction'])} |"
    )

lines.extend(
    [
        "",
        "## Detection best-AP results",
        "",
        "| Strategy | Condition | Best epoch | AP | AP50 | AP75 | AP_S | AP_M | AP_L | Final AP | Status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
)
for result in detection_results:
    best = result["best"]
    at_boundary = result["best_epoch"] == result["final_epoch"]
    status = "best at final eval" if at_boundary else "internal peak"
    lines.append(
        f"| {result['strategy']} | {result['condition']} | {result['best_epoch']} "
        f"| {fmt(best['AP'])} | {fmt(best['AP_50'])} | {fmt(best['AP_75'])} "
        f"| {fmt(best['AP_S'])} | {fmt(best['AP_M'])} | {fmt(best['AP_L'])} "
        f"| {fmt(result['final']['AP'])} | {status} |"
    )

by_key = {
    (result["strategy"], result["condition"]): result
    for result in detection_results
}
lines.extend(
    [
        "",
        "## Event-aware minus random-mask pretraining",
        "",
        "| Condition | dAP | dAP50 | dAP75 | dAP_S | dAP_M | dAP_L |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
)
for condition in ("Pretrained / Frozen", "Pretrained / Fine-tune"):
    random_result = by_key[("Random mask", condition)]
    event_result = by_key[("Event-aware", condition)]
    delta = {
        metric: event_result["best"][metric] - random_result["best"][metric]
        for metric in detection_metrics
    }
    lines.append(
        f"| {condition} | {delta['AP']:+.4f} | {delta['AP_50']:+.4f} "
        f"| {delta['AP_75']:+.4f} | {delta['AP_S']:+.4f} "
        f"| {delta['AP_M']:+.4f} | {delta['AP_L']:+.4f} |"
    )

random_frozen = by_key[("Control", "Random / Frozen")]
random_scratch = by_key[("Control", "Random / Scratch")]
lines.extend(
    [
        "",
        "## AP improvement over controls",
        "",
        "| Pretraining | Frozen vs random frozen | Fine-tune vs random scratch |",
        "|---|---:|---:|",
    ]
)
for strategy in ("Random mask", "Event-aware"):
    frozen = by_key[(strategy, "Pretrained / Frozen")]
    finetune = by_key[(strategy, "Pretrained / Fine-tune")]
    lines.append(
        f"| {strategy} "
        f"| {frozen['best']['AP'] - random_frozen['best']['AP']:+.4f} "
        f"| {finetune['best']['AP'] - random_scratch['best']['AP']:+.4f} |"
    )

boundary_runs = [
    f"{result['strategy']} / {result['condition']}"
    for result in detection_results
    if result["best_epoch"] == result["final_epoch"]
]
lines.extend(
    [
        "",
        "## Convergence check",
        "",
        "Best APがepoch 50にある条件は、追加学習候補です。",
        "",
    ]
)
if boundary_runs:
    lines.extend(f"- {name}" for name in boundary_runs)
else:
    lines.append("- 全条件でBest APはepoch 50より前です。")
lines.extend(
    [
        "",
        "注: seed 0のみの結果であり、統計的な優位性ではなく傾向として解釈します。",
    ]
)

markdown = detection_root / "experiment_summary.md"
markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print(f"\nMarkdown: {markdown}")
print(f"Detection CSV: {detection_csv}")
print(f"Pretrain CSV: {pretrain_csv}")
PY
}

require_command perl
require_command grep
require_command python
require_command tee

case "$ACTION" in
  prepare)
    prepare_configs
    ;;
  pretrain)
    prepare_configs
    run_pretrain
    ;;
  detection)
    run_detection
    ;;
  summary)
    summarize_results
    ;;
  all)
    prepare_configs
    run_pretrain
    run_detection
    summarize_results
    ;;
esac
