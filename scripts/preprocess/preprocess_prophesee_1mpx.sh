#!/usr/bin/env bash
set -eu

usage() {
  cat <<'EOF'
Usage:
  bash scripts/preprocess/preprocess_prophesee_1mpx.sh \
    --input /datasets/1mpx/train \
    --bundle-root /datasets/evjepa/1mpx_640x360 \
    --split train \
    --sequence-list /path/to/1mpx_train_subset.txt

Required:
  --input PATH           Raw DAT file or directory containing one logical split
  --bundle-root PATH     Movable output bundle root
  --split SPLIT          train, val, or test

Selection:
  --sequence-list FILE   One DAT stem per line, including `_td` and excluding `.dat`
                         (recommended and required for directories)
  --all                  Explicitly convert every DAT in the selected split
  --limit N              Convert only the first N selected recordings

Mode:
  --plan-only            Validate inputs and print a no-write plan
  --self-supervised      Allow train/val recordings without sibling *_bbox.npy
  --python-bin PATH      Python from the target venv (default: python)
  -h, --help             Show this help

The spatial factor is intentionally fixed at 2 (1280x720 -> 640x360). Event
timestamps and event rows are retained; accumulation windows are sampled later.
Rerunning the same command validates completed files and resumes a compatible
per-recording .partial file.
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

while [ "$#" -gt 0 ]; do
  case "$1" in
    --input)
      [ "$#" -ge 2 ] || { echo "missing value for --input" >&2; exit 2; }
      input_path=$2
      shift 2
      ;;
    --bundle-root)
      [ "$#" -ge 2 ] || { echo "missing value for --bundle-root" >&2; exit 2; }
      bundle_root=$2
      shift 2
      ;;
    --split)
      [ "$#" -ge 2 ] || { echo "missing value for --split" >&2; exit 2; }
      logical_split=$2
      shift 2
      ;;
    --sequence-list)
      [ "$#" -ge 2 ] || { echo "missing value for --sequence-list" >&2; exit 2; }
      sequence_list=$2
      shift 2
      ;;
    --limit)
      [ "$#" -ge 2 ] || { echo "missing value for --limit" >&2; exit 2; }
      limit=$2
      shift 2
      ;;
    --python-bin)
      [ "$#" -ge 2 ] || { echo "missing value for --python-bin" >&2; exit 2; }
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

[ -n "$input_path" ] || { echo "--input is required" >&2; exit 2; }
[ -n "$bundle_root" ] || { echo "--bundle-root is required" >&2; exit 2; }
case "$logical_split" in
  train|val|test) ;;
  *) echo "--split must be train, val, or test" >&2; exit 2 ;;
esac
if [ -n "$sequence_list" ] && [ "$select_all" -eq 1 ]; then
  echo "--sequence-list and --all are mutually exclusive" >&2
  exit 2
fi
if [ -d "$input_path" ] && [ -z "$sequence_list" ] && [ "$select_all" -ne 1 ]; then
  echo "directory input requires --sequence-list; use --all only after a pilot" >&2
  exit 2
fi

arguments=(
  --dataset prophesee_1mpx
  --input "$input_path"
  --output-root "$bundle_root/events/$logical_split"
  --bbox-output-root "$bundle_root/labels/$logical_split"
  --manifest "$bundle_root/manifests/$logical_split.jsonl"
  --split "$logical_split"
  --camera left
  --spatial-downsample 2
  --zstd-level 5
  --skip-existing
  --merge-manifest
)

if [ -n "$sequence_list" ]; then
  arguments+=(--sequence-list "$sequence_list")
fi
if [ -n "$limit" ]; then
  arguments+=(--limit "$limit")
fi
if [ "$plan_only" -eq 1 ]; then
  arguments+=(--plan-only)
fi
if [ "$self_supervised" -eq 1 ]; then
  arguments+=(--allow-missing-bboxes)
fi

exec "$python_bin" -m event_window_jepa.preprocessing.cli "${arguments[@]}"
