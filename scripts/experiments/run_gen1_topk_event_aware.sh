#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BASE_CONFIG="$PROJECT_ROOT/configs/pretrain/window_jepa21_vits_gen1.yaml"
AUDIT_ROOT="$PROJECT_ROOT/outputs/mask_audit/topk_enrichment_seed0"
SELECTED_FRACTION_FILE="$AUDIT_ROOT/selected_topk_fraction.txt"
PRETRAIN_ROOT="$PROJECT_ROOT/outputs/pretrain/topk_v3"
DETECTION_ROOT="$PROJECT_ROOT/outputs/downstream/topk_v3"
RANDOM_BASELINE_ROOT="$PROJECT_ROOT/outputs/downstream/rerun_v2"
DATA_ROOT=/media/noah-22/AT_SSD/dataset/evjepa/gen1_304x240
ACTION=all

usage() {
  printf '%s\n' \
    'Usage: bash scripts/experiments/run_gen1_topk_event_aware.sh [options]' \
    '' \
    '  --action ACTION   prepare, pretrain, detection, summary, or all (default: all)' \
    '  --data-root DIR   Gen1 processed dataset root' \
    '  -h, --help        Show this help'
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

selected_fraction() {
  require_file "$SELECTED_FRACTION_FILE"
  local fraction
  fraction=$(tr -d '[:space:]' < "$SELECTED_FRACTION_FILE")
  case "$fraction" in
    0.25|0.125|0.0625)
      printf '%s\n' "$fraction"
      ;;
    *)
      printf 'Unsupported audited top-k fraction: %s\n' "$fraction" >&2
      exit 1
      ;;
  esac
}

fraction_label() {
  case "$1" in
    0.25) printf 'top25\n' ;;
    0.125) printf 'top12_5\n' ;;
    0.0625) printf 'top6_25\n' ;;
    *) return 1 ;;
  esac
}

require_fresh_output() {
  local output_dir=$1
  if [[ -e "$output_dir/train.jsonl" || -e "$output_dir/checkpoint-latest.pt" ]]; then
    printf 'Refusing to mix with an existing run: %s\n' "$output_dir" >&2
    printf 'Use --action summary, or move the existing directory before retraining.\n' >&2
    exit 1
  fi
}

topk_config_path() {
  local label=$1
  printf '%s/configs/pretrain/window_jepa21_vits_gen1_event_aware_%s_v3.yaml\n' \
    "$PROJECT_ROOT" "$label"
}

prepare_config() {
  require_file "$BASE_CONFIG"
  local fraction
  local label
  local config
  fraction=$(selected_fraction)
  label=$(fraction_label "$fraction")
  config=$(topk_config_path "$label")

  cp "$BASE_CONFIG" "$config"
  TOPK_FRACTION="$fraction" perl -0pi -e \
    's/(  activity_candidates: 32\n)/$1  activity_selection_strategy: topk_enrichment\n  activity_topk_fraction: $ENV{TOPK_FRACTION}\n/' \
    "$config"
  TOPK_OUTPUT="outputs/pretrain/topk_v3/event_aware_${label}_seed0" perl -0pi -e \
    's#output_dir: outputs/pretrain/vjepa21_vits_gen1_event_aware_seed0#output_dir: $ENV{TOPK_OUTPUT}#' \
    "$config"

  grep -q 'activity_selection_strategy: topk_enrichment' "$config"
  grep -q "activity_topk_fraction: $fraction" "$config"
  grep -q "output_dir: outputs/pretrain/topk_v3/event_aware_${label}_seed0" "$config"
  grep -q 'epochs: 100' "$config"
  printf 'Prepared top-k pretrain config: %s\n' "$config"
}

verify_pretrain_checkpoint() {
  local fraction
  local label
  local checkpoint
  fraction=$(selected_fraction)
  label=$(fraction_label "$fraction")
  checkpoint="$PRETRAIN_ROOT/event_aware_${label}_seed0/checkpoint-latest.pt"
  require_file "$checkpoint"

  TOPK_CHECKPOINT="$checkpoint" TOPK_FRACTION="$fraction" python - <<'PY'
import os
from pathlib import Path

import torch

from event_window_jepa.config import ExperimentConfig
from event_window_jepa.train.checkpoint import config_hash

path = Path(os.environ["TOPK_CHECKPOINT"])
expected_fraction = float(os.environ["TOPK_FRACTION"])
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
config = ExperimentConfig.from_mapping(checkpoint["resolved_config"])
if checkpoint.get("config_hash") != config_hash(config):
    raise RuntimeError(f"inconsistent checkpoint configuration metadata: {path}")
if int(checkpoint.get("epoch", -1)) != 100:
    raise RuntimeError(f"pretraining did not reach epoch 100: {path}")
if config.mask.activity_selection_strategy != "topk_enrichment":
    raise RuntimeError("checkpoint did not use topk_enrichment")
if config.mask.activity_topk_fraction != expected_fraction:
    raise RuntimeError("checkpoint top-k fraction differs from the audited value")
print(f"verified: {path}")
PY
}

run_pretrain() {
  require_command window-jepa-pretrain
  local fraction
  local label
  local config
  local output_dir
  local checkpoint
  local resume=()
  fraction=$(selected_fraction)
  label=$(fraction_label "$fraction")
  config=$(topk_config_path "$label")
  require_file "$config"
  output_dir="$PRETRAIN_ROOT/event_aware_${label}_seed0"
  checkpoint="$output_dir/checkpoint-latest.pt"
  if [[ -e "$checkpoint" ]]; then
    require_file "$output_dir/train.jsonl"
    resume=(--resume "$checkpoint")
    printf 'Resuming top-k pretraining from: %s\n' "$checkpoint"
  else
    require_fresh_output "$output_dir"
  fi
  mkdir -p "$PRETRAIN_ROOT/logs"

  (
    cd "$PROJECT_ROOT"
    PYTHONUNBUFFERED=1 window-jepa-pretrain --config "$config" "${resume[@]}"
  ) 2>&1 | tee "$PRETRAIN_ROOT/logs/event_aware_${label}_seed0.log"

  verify_pretrain_checkpoint
}

run_detection_one() {
  local name=$1
  local checkpoint=$2
  local unfreeze=$3
  local output_dir="$DETECTION_ROOT/$name"
  local detection_checkpoint="$output_dir/checkpoint-latest.pt"
  local command=(
    window-jepa-gen1-detect
    --checkpoint "$checkpoint"
    --train-manifest "$TRAIN_MANIFEST"
    --val-manifest "$VAL_MANIFEST"
    --output-dir "$output_dir"
    --backbone-init pretrained
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

  if [[ -e "$detection_checkpoint" ]]; then
    require_file "$output_dir/train.jsonl"
    command+=(--resume "$detection_checkpoint")
    printf 'Resuming detection from: %s\n' "$detection_checkpoint"
  else
    require_fresh_output "$output_dir"
  fi
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
  verify_pretrain_checkpoint
  local fraction
  local label
  local checkpoint
  fraction=$(selected_fraction)
  label=$(fraction_label "$fraction")
  checkpoint="$PRETRAIN_ROOT/event_aware_${label}_seed0/checkpoint-latest.pt"

  run_detection_one "event_aware_${label}_finetune_seed0" "$checkpoint" 1
  run_detection_one "event_aware_${label}_frozen_seed0" "$checkpoint" 0
}

summarize_results() {
  local fraction
  local label
  fraction=$(selected_fraction)
  label=$(fraction_label "$fraction")
  TOPK_PROJECT_ROOT="$PROJECT_ROOT" TOPK_LABEL="$label" python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["TOPK_PROJECT_ROOT"])
label = os.environ["TOPK_LABEL"]
runs = [
    ("Random mask", "Frozen", root / "outputs/downstream/rerun_v2/random_mask_frozen_seed0/train.jsonl"),
    ("Top-k event-aware", "Frozen", root / f"outputs/downstream/topk_v3/event_aware_{label}_frozen_seed0/train.jsonl"),
    ("Random mask", "Fine-tune", root / "outputs/downstream/rerun_v2/random_mask_finetune_seed0/train.jsonl"),
    ("Top-k event-aware", "Fine-tune", root / f"outputs/downstream/topk_v3/event_aware_{label}_finetune_seed0/train.jsonl"),
    ("Control", "Random Frozen", root / "outputs/downstream/rerun_v2/random_frozen_seed0/train.jsonl"),
    ("Control", "Random Scratch", root / "outputs/downstream/rerun_v2/random_scratch_seed0/train.jsonl"),
]
metrics = ("AP", "AP_50", "AP_75", "AP_S", "AP_M", "AP_L")
results = []
for strategy, condition, path in runs:
    if not path.is_file():
        raise FileNotFoundError(f"required detection log does not exist: {path}")
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    evaluated = [record for record in records if record.get("validation", {}).get("AP") is not None]
    if not evaluated:
        raise RuntimeError(f"no validation result in: {path}")
    best = max(evaluated, key=lambda record: float(record["validation"]["AP"]))
    final = evaluated[-1]
    results.append(
        {
            "strategy": strategy,
            "condition": condition,
            "best_epoch": int(best["epoch"]),
            "best": {metric: float(best["validation"][metric]) for metric in metrics},
            "final_epoch": int(final["epoch"]),
            "final_ap": float(final["validation"]["AP"]),
        }
    )

lines = [
    "# Top-k event-aware detection comparison",
    "",
    "| Strategy | Condition | Best epoch | AP | AP50 | AP75 | AP_S | AP_M | AP_L | Final AP |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for result in results:
    best = result["best"]
    lines.append(
        f"| {result['strategy']} | {result['condition']} | {result['best_epoch']} "
        f"| {best['AP']:.4f} | {best['AP_50']:.4f} | {best['AP_75']:.4f} "
        f"| {best['AP_S']:.4f} | {best['AP_M']:.4f} | {best['AP_L']:.4f} "
        f"| {result['final_ap']:.4f} |"
    )

by_key = {(result["strategy"], result["condition"]): result for result in results}
lines.extend(
    [
        "",
        "## Top-k event-aware minus random mask",
        "",
        "| Condition | dAP | dAP50 | dAP75 | dAP_S | dAP_M | dAP_L |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
)
for condition in ("Frozen", "Fine-tune"):
    random_result = by_key[("Random mask", condition)]["best"]
    topk_result = by_key[("Top-k event-aware", condition)]["best"]
    delta = {metric: topk_result[metric] - random_result[metric] for metric in metrics}
    lines.append(
        f"| {condition} | {delta['AP']:+.4f} | {delta['AP_50']:+.4f} "
        f"| {delta['AP_75']:+.4f} | {delta['AP_S']:+.4f} "
        f"| {delta['AP_M']:+.4f} | {delta['AP_L']:+.4f} |"
    )

output = root / "outputs/downstream/topk_v3/topk_comparison.md"
output.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print(f"\nMarkdown: {output}")
PY
}

require_command grep
require_command perl
require_command python
require_command tee
require_command tr

case "$ACTION" in
  prepare)
    prepare_config
    ;;
  pretrain)
    prepare_config
    run_pretrain
    ;;
  detection)
    run_detection
    ;;
  summary)
    summarize_results
    ;;
  all)
    prepare_config
    run_pretrain
    run_detection
    summarize_results
    ;;
esac
