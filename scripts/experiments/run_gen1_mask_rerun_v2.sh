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
    '  --action ACTION   prepare, pretrain, detection, or all (default: all)' \
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
  prepare|pretrain|detection|all) ;;
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
  all)
    prepare_configs
    run_pretrain
    run_detection
    ;;
esac
