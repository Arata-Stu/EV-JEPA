#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/preprocess/preprocess_mvsec.sh \
    --raw-root /datasets/mvsec \
    --bundle-root /datasets/mvsec/processed [OPTIONS]

Required:
  --raw-root PATH       MVSEC dataset root, or downloader root containing raw/
  --bundle-root PATH    Destination for canonical events and JSONL manifests

Options:
  --include-night       Also convert outdoor_night1 left as ood_test
  --plan-only           Validate sources/GT and print the conversion plan only
  --progress MODE       auto, tqdm, json, or none (default: auto)
  --python-bin PATH     Python from the target environment (default: python)
  -h, --help            Show this help

Stage 1 keeps raw/distorted MVSEC events at native 346x260. The training
manifest contains outdoor_day2 left+right in one recording-level split; only
left receives official NPZ flow and HDF5 depth/pose references. outdoor_day1 is
a separate final test manifest and is never used for training or checkpoint
selection. The raw root may use either ROOT/outdoor_day (GUI/direct download) or
ROOT/raw/outdoor_day (this repository's downloader). If both layouts exist the
wrapper stops instead of choosing one. Plan mode reports all missing Stage 1
data/GT HDF5 and flow NPZ files together.
EOF
}

raw_root=""
bundle_root=""
python_bin="python"
progress="auto"
include_night=0
plan_only=0

while (($#)); do
  case "$1" in
    --raw-root)
      (($# >= 2)) || { echo "missing value for --raw-root" >&2; exit 2; }
      raw_root=$2
      shift 2
      ;;
    --bundle-root)
      (($# >= 2)) || { echo "missing value for --bundle-root" >&2; exit 2; }
      bundle_root=$2
      shift 2
      ;;
    --python-bin)
      (($# >= 2)) || { echo "missing value for --python-bin" >&2; exit 2; }
      python_bin=$2
      shift 2
      ;;
    --progress)
      (($# >= 2)) || { echo "missing value for --progress" >&2; exit 2; }
      progress=$2
      shift 2
      ;;
    --include-night)
      include_night=1
      shift
      ;;
    --plan-only)
      plan_only=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$progress" in
  auto|tqdm|json|none) ;;
  *) echo "invalid --progress mode: $progress" >&2; exit 2 ;;
esac

[[ -n "$raw_root" ]] || { echo "--raw-root is required" >&2; exit 2; }
[[ -n "$bundle_root" ]] || { echo "--bundle-root is required" >&2; exit 2; }
[[ -d "$raw_root" ]] || { echo "raw root does not exist: $raw_root" >&2; exit 1; }

direct_day_root="$raw_root/outdoor_day"
managed_day_root="$raw_root/raw/outdoor_day"
direct_layout=0
managed_layout=0
[[ -d "$direct_day_root" ]] && direct_layout=1
[[ -d "$managed_day_root" ]] && managed_layout=1

if ((direct_layout && managed_layout)); then
  printf '%s\n' \
    "ambiguous MVSEC raw root: both direct and downloader layouts exist:" \
    "  - $direct_day_root" \
    "  - $managed_day_root" \
    "Pass an unambiguous dataset root; use its raw/ directory for the downloader layout." >&2
  exit 1
fi

layout="direct"
dataset_root=$raw_root
if ((managed_layout)); then
  layout="downloader"
  dataset_root="$raw_root/raw"
fi

required_relative_paths=(
  "outdoor_day/outdoor_day1_data.hdf5"
  "outdoor_day/outdoor_day1_gt.hdf5"
  "outdoor_day/outdoor_day1_gt_flow_dist.npz"
  "outdoor_day/outdoor_day2_data.hdf5"
  "outdoor_day/outdoor_day2_gt.hdf5"
  "outdoor_day/outdoor_day2_gt_flow_dist.npz"
)
if ((include_night)); then
  required_relative_paths+=(
    "outdoor_night/outdoor_night1_data.hdf5"
    "outdoor_night/outdoor_night1_gt.hdf5"
  )
fi

missing_paths=()
for relative_path in "${required_relative_paths[@]}"; do
  candidate="$dataset_root/$relative_path"
  [[ -f "$candidate" ]] || missing_paths+=("$candidate")
done
if ((${#missing_paths[@]})); then
  printf 'required MVSEC Stage 1 files are missing from the %s layout:\n' \
    "$layout" >&2
  printf '  - %s\n' "${missing_paths[@]}" >&2
  printf '%s\n' \
    'Dense flow NPZ files are a separate official download:' \
    '  https://daniilidis-group.github.io/mvsec/download/' >&2
  exit 1
fi

# Resolve the selected root without relying on GNU-only `readlink -f`; this is
# also compatible with the Bash 3.2 shipped by macOS.
dataset_root=$(cd "$dataset_root" && pwd -P)
printf 'Resolved MVSEC raw root (%s layout): %s\n' "$layout" "$dataset_root"

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
train_list="$project_root/configs/datasets/mvsec_stage1_train.txt"
test_list="$project_root/configs/datasets/mvsec_stage1_test.txt"
ood_list="$project_root/configs/datasets/mvsec_stage1_ood.txt"

common=(
  --dataset mvsec
  --input "$dataset_root"
  --spatial-downsample 1
  --spatial-downsample-method coordinate
  --zstd-level 5
  --progress "$progress"
  --skip-existing
  --merge-manifest
)
if ((plan_only)); then
  common+=(--plan-only)
fi

run_conversion() {
  local output_name=$1
  local split=$2
  local camera=$3
  local sequence_list=$4
  local gt_mode=$5
  local flow_mode=$6
  "$python_bin" -m event_window_jepa.preprocessing.cli \
    "${common[@]}" \
    --output-root "$bundle_root/events/$output_name" \
    --manifest "$bundle_root/manifests/$output_name.jsonl" \
    --split "$split" \
    --camera "$camera" \
    --sequence-list "$sequence_list" \
    --mvsec-gt "$gt_mode" \
    --mvsec-flow "$flow_mode"
}

run_conversion train train left "$train_list" reference official-npz
run_conversion train train right "$train_list" none none
run_conversion test test left "$test_list" reference official-npz
if ((include_night)); then
  run_conversion ood_test test left "$ood_list" reference none
fi

if ((plan_only)); then
  printf 'MVSEC preprocessing plan validated.\n'
else
  printf 'MVSEC preprocessing complete: %s\n' "$bundle_root"
  printf 'Train manifest: %s\n' "$bundle_root/manifests/train.jsonl"
  printf 'Test manifest:  %s\n' "$bundle_root/manifests/test.jsonl"
fi
