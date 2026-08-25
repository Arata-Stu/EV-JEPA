#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/preprocess/preprocess_dsec.sh \
    --input /datasets/downloads/dsec/raw \
    --bundle-root /datasets/evjepa/dsec_640x480 \
    --split train

Required:
  --input PATH           DSEC events.h5 file or extracted DSEC root
  --bundle-root PATH     Movable output bundle root
  --split SPLIT          train, val, or test

Selection:
  --sequence-list FILE   One DSEC sequence name per line
                         (default: official DSEC-Detection list for SPLIT)
  --limit N              Convert only the first N selected sequences

Mode:
  --plan-only            Validate inputs and print a no-write plan
  --python-bin PATH      Python from the target venv (default: python)
  -h, --help             Show this help

DSEC is intentionally kept at its native 640x480 resolution. Spatial factor 1
and coordinate-preserving conversion are fixed so distorted event coordinates
remain aligned with the official detection labels. Reruns validate completed
files and resume compatible per-sequence .partial files.
EOF
}

input_path=""
bundle_root=""
logical_split=""
sequence_list=""
limit=""
python_bin="python"
plan_only=0

while (($#)); do
  case "$1" in
    --input)
      (($# >= 2)) || { echo "missing value for --input" >&2; exit 2; }
      input_path=$2
      shift 2
      ;;
    --bundle-root)
      (($# >= 2)) || { echo "missing value for --bundle-root" >&2; exit 2; }
      bundle_root=$2
      shift 2
      ;;
    --split)
      (($# >= 2)) || { echo "missing value for --split" >&2; exit 2; }
      logical_split=$2
      shift 2
      ;;
    --sequence-list)
      (($# >= 2)) || { echo "missing value for --sequence-list" >&2; exit 2; }
      sequence_list=$2
      shift 2
      ;;
    --limit)
      (($# >= 2)) || { echo "missing value for --limit" >&2; exit 2; }
      limit=$2
      shift 2
      ;;
    --python-bin)
      (($# >= 2)) || { echo "missing value for --python-bin" >&2; exit 2; }
      python_bin=$2
      shift 2
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

[[ -n "$input_path" ]] || { echo "--input is required" >&2; exit 2; }
[[ -n "$bundle_root" ]] || { echo "--bundle-root is required" >&2; exit 2; }
case "$logical_split" in
  train|val|test) ;;
  *) echo "--split must be train, val, or test" >&2; exit 2 ;;
esac

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
if [[ -z "$sequence_list" && -d "$input_path" ]]; then
  sequence_list="$project_root/configs/datasets/dsec_detection_${logical_split}.txt"
fi
if [[ -n "$sequence_list" && ! -f "$sequence_list" ]]; then
  echo "sequence list does not exist: $sequence_list" >&2
  exit 1
fi

arguments=(
  --dataset dsec
  --input "$input_path"
  --output-root "$bundle_root/events/$logical_split"
  --manifest "$bundle_root/manifests/$logical_split.jsonl"
  --split "$logical_split"
  --camera left
  --spatial-downsample 1
  --spatial-downsample-method coordinate
  --zstd-level 5
  --skip-existing
  --merge-manifest
)

if [[ -n "$sequence_list" ]]; then
  arguments+=(--sequence-list "$sequence_list")
fi
if [[ -n "$limit" ]]; then
  arguments+=(--limit "$limit")
fi
if ((plan_only)); then
  arguments+=(--plan-only)
fi

exec "$python_bin" -m event_window_jepa.preprocessing.cli "${arguments[@]}"
