#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/preprocess/preprocess_gen1.sh \
    --input /datasets/downloads/gen1/raw \
    --bundle-root /datasets/evjepa/gen1_304x240 \
    --split train \
    --all

Required:
  --input PATH           Gen1 DAT/RVT HDF5 file or dataset root
  --bundle-root PATH     Movable output bundle root
  --split SPLIT          train, val, or test

Selection:
  --sequence-list FILE   One selected recording name per line
                         (recommended and required for directories unless --all)
  --all                  Explicitly convert every recording in the selected split
  --limit N              Convert only the first N selected recordings

Mode:
  --plan-only            Validate inputs and print a no-write plan
  --self-supervised      Allow train/val recordings without sibling *_bbox.npy
  --python-bin PATH      Python from the target venv (default: python)
  -h, --help             Show this help

Gen1 is intentionally kept at its native 304x240 resolution. Spatial factor 1
and coordinate-preserving conversion are fixed; bbox coordinates and timestamps
are copied without rewriting. Reruns validate completed files and resume
compatible per-recording .partial files.
EOF
}

input_path=""
bundle_root=""
logical_split=""
sequence_list=""
limit=""
python_bin="python"
select_all=0
plan_only=0
self_supervised=0

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
    --all)
      select_all=1
      shift
      ;;
    --plan-only)
      plan_only=1
      shift
      ;;
    --self-supervised)
      self_supervised=1
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
if [[ -n "$sequence_list" && "$select_all" -eq 1 ]]; then
  echo "--sequence-list and --all are mutually exclusive" >&2
  exit 2
fi
if [[ -d "$input_path" && -z "$sequence_list" && "$select_all" -ne 1 ]]; then
  echo "directory input requires --sequence-list; use --all only after a pilot" >&2
  exit 2
fi
if [[ -n "$sequence_list" && ! -f "$sequence_list" ]]; then
  echo "sequence list does not exist: $sequence_list" >&2
  exit 1
fi

arguments=(
  --dataset gen1
  --input "$input_path"
  --output-root "$bundle_root/events/$logical_split"
  --bbox-output-root "$bundle_root/labels/$logical_split"
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
if ((self_supervised)); then
  arguments+=(--allow-missing-bboxes)
fi

exec "$python_bin" -m event_window_jepa.preprocessing.cli "${arguments[@]}"
