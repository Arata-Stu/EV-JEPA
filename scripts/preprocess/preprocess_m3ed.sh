#!/usr/bin/env bash
set -euo pipefail

RAW_ROOT=/mnt/ssd-4tb/dataset/m3ed
OUTPUT_ROOT=/mnt/ssd-4tb/dataset/evjepa/m3ed_640x360_dagr
DATASET_LIST=
INCLUDE_OFFICIAL_TEST=0

usage() {
  printf '%s\n' \
    'Usage: bash scripts/preprocess/preprocess_m3ed.sh [options]' \
    '' \
    '  --raw-root DIR             M3ED root (default: /mnt/ssd-4tb/dataset/m3ed)' \
    '  --output-root DIR          Portable processed bundle root' \
    '  --dataset-list FILE        Official dataset_list.yaml' \
    '  --include-official-test    Also convert the five label-hidden car test recordings' \
    '  -h, --help                 Show this help'
}

while (($#)); do
  case "$1" in
    --raw-root)
      RAW_ROOT=$2
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT=$2
      shift 2
      ;;
    --dataset-list)
      DATASET_LIST=$2
      shift 2
      ;;
    --include-official-test)
      INCLUDE_OFFICIAL_TEST=1
      shift
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

if [[ -z "$DATASET_LIST" ]]; then
  DATASET_LIST="$RAW_ROOT/metadata/dataset_list.yaml"
fi

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
NON_TEST_LIST="$PROJECT_ROOT/configs/datasets/m3ed_car_storage_non_test.txt"
OFFICIAL_TEST_LIST="$PROJECT_ROOT/configs/datasets/m3ed_car_storage_official_test.txt"
PROTOCOL="$PROJECT_ROOT/configs/datasets/m3ed_f3_route_holdout_v1.yaml"

for required in "$RAW_ROOT" "$DATASET_LIST" "$NON_TEST_LIST" "$PROTOCOL"; do
  if [[ ! -e "$required" ]]; then
    printf 'Required path does not exist: %s\n' "$required" >&2
    exit 1
  fi
done

mkdir -p \
  "$OUTPUT_ROOT/events/storage_train" \
  "$OUTPUT_ROOT/labels" \
  "$OUTPUT_ROOT/manifests" \
  "$OUTPUT_ROOT/logs"

window-jepa-preprocess \
  --dataset m3ed \
  --input "$RAW_ROOT" \
  --output-root "$OUTPUT_ROOT/events/storage_train" \
  --manifest "$OUTPUT_ROOT/manifests/storage_train.jsonl" \
  --split train \
  --camera left \
  --m3ed-dataset-list "$DATASET_LIST" \
  --sequence-list "$NON_TEST_LIST" \
  --m3ed-labels copy \
  --m3ed-label-output-root "$OUTPUT_ROOT/labels" \
  --spatial-downsample 2 \
  --spatial-downsample-method area_accumulate \
  --skip-existing \
  --merge-manifest \
  2>&1 | tee "$OUTPUT_ROOT/logs/preprocess_storage_train.log"

MANIFEST_INPUTS=("$OUTPUT_ROOT/manifests/storage_train.jsonl")
if ((INCLUDE_OFFICIAL_TEST)); then
  mkdir -p "$OUTPUT_ROOT/events/storage_test"
  window-jepa-preprocess \
    --dataset m3ed \
    --input "$RAW_ROOT" \
    --output-root "$OUTPUT_ROOT/events/storage_test" \
    --manifest "$OUTPUT_ROOT/manifests/storage_test.jsonl" \
    --split test \
    --camera left \
    --m3ed-dataset-list "$DATASET_LIST" \
    --sequence-list "$OFFICIAL_TEST_LIST" \
    --m3ed-labels copy \
    --m3ed-label-output-root "$OUTPUT_ROOT/labels" \
    --spatial-downsample 2 \
    --spatial-downsample-method area_accumulate \
    --skip-existing \
    --merge-manifest \
    2>&1 | tee "$OUTPUT_ROOT/logs/preprocess_storage_test.log"
  MANIFEST_INPUTS+=("$OUTPUT_ROOT/manifests/storage_test.jsonl")
  window-jepa-merge-manifests \
    --output "$OUTPUT_ROOT/manifests/official_test.jsonl" \
    "$OUTPUT_ROOT/manifests/storage_test.jsonl"
fi

window-jepa-merge-manifests \
  --output "$OUTPUT_ROOT/manifests/all.jsonl" \
  "${MANIFEST_INPUTS[@]}"

window-jepa-m3ed-splits \
  --input-manifest "$OUTPUT_ROOT/manifests/all.jsonl" \
  --protocol "$PROTOCOL" \
  --m3ed-dataset-list "$DATASET_LIST" \
  --output-dir "$OUTPUT_ROOT/manifests" \
  --allow-unassigned

for split in train val test; do
  rows=$(wc -l < "$OUTPUT_ROOT/manifests/$split.jsonl")
  printf '%s manifest rows: %s\n' "$split" "$rows"
done
printf 'M3ED preprocessing complete: %s\n' "$OUTPUT_ROOT"
