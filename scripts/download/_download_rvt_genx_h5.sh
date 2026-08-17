#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

DATASET="${1:-}"
shift || true

case "${DATASET}" in
  gen4) BASE_URL='https://download.ifi.uzh.ch/rpg/RVT/datasets/gen4_tar' ;;
  gen1) BASE_URL='https://download.ifi.uzh.ch/rpg/RVT/datasets/gen1_tar' ;;
  *) download_die "internal error: expected gen1 or gen4" ;;
esac

rvt_crc32() {
  case "${DATASET}:$1" in
    gen4:train) printf '%s\n' 'd677488a' ;;
    gen4:val) printf '%s\n' '72f13c3e' ;;
    gen4:test) printf '%s\n' '643e61ef' ;;
    gen1:train) printf '%s\n' '3d23bd30' ;;
    gen1:val) printf '%s\n' 'cc802022' ;;
    gen1:test) printf '%s\n' 'cdd4fd69' ;;
    *) download_die "internal error: unsupported split: $1" ;;
  esac
}

usage() {
  cat <<EOF
Usage: ${RVT_DOWNLOAD_PROGRAM_NAME:-$(basename "$0")} [--root DIRECTORY] [--split train|val|test|all] [--extract]
       ${RVT_DOWNLOAD_PROGRAM_NAME:-$(basename "$0")} --extracted-root DIRECTORY [--split train|val|test|all]

Download RVT's original-event ${DATASET} HDF5 distribution. These are continuous
event streams, not RVT's fixed event-representation files.

Options:
  --root DIRECTORY            Put TARs in ROOT/archives and data in ROOT/raw
  --split SPLIT               train, val, test, or all (default: all)
  --extract                   Safely extract after download
  --extracted-root DIRECTORY  Validate an existing train/val/test HDF5 tree only
  --allow-orphan-bboxes       Report but allow bbox files whose corrupt HDF5 was removed
  --help                      Show this help

Downloads are resumable. Publisher CRC32 is checked before a TAR is published.
EOF
}

ROOT=""
SPLIT="all"
EXTRACT=0
EXTRACTED_ROOT=""
ALLOW_ORPHAN_BBOXES=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --root)
      [ "$#" -ge 2 ] || download_die "--root requires a value"
      ROOT="$2"
      shift 2
      ;;
    --split)
      [ "$#" -ge 2 ] || download_die "--split requires a value"
      SPLIT="$2"
      shift 2
      ;;
    --extract)
      EXTRACT=1
      shift
      ;;
    --extracted-root)
      [ "$#" -ge 2 ] || download_die "--extracted-root requires a value"
      EXTRACTED_ROOT="$2"
      shift 2
      ;;
    --allow-orphan-bboxes)
      ALLOW_ORPHAN_BBOXES=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *) download_die "unknown argument: $1" ;;
  esac
done

case "${SPLIT}" in
  train|val|test|all) ;;
  *) download_die "--split must be train, val, test, or all" ;;
esac

download_require_runtime

VALIDATE_EXTRA_ARGS=()
if [ "${ALLOW_ORPHAN_BBOXES}" -eq 1 ]; then
  VALIDATE_EXTRA_ARGS+=(--allow-orphan-bboxes)
fi

if [ -n "${EXTRACTED_ROOT}" ]; then
  [ -z "${ROOT}" ] || download_die "--root and --extracted-root cannot be combined"
  [ "${EXTRACT}" -eq 0 ] || download_die "--extract is invalid with --extracted-root"
  RAW_ROOT="$(cd "${EXTRACTED_ROOT}" && pwd)" || \
    download_die "extracted root does not exist: ${EXTRACTED_ROOT}"
else
  [ -n "${ROOT}" ] || download_die "--root is required"
  [ "${ALLOW_ORPHAN_BBOXES}" -eq 0 ] || \
    download_die "--allow-orphan-bboxes is only valid with --extracted-root"
  mkdir -p "${ROOT}/archives" "${ROOT}/raw" "${ROOT}/state/extract"
  RAW_ROOT="${ROOT}/raw"
fi

if [ "${SPLIT}" = all ]; then
  SPLITS=(train val test)
else
  SPLITS=("${SPLIT}")
fi

for current_split in "${SPLITS[@]}"; do
  if [ -n "${EXTRACTED_ROOT}" ]; then
    "${DOWNLOAD_PYTHON}" "${DOWNLOAD_ARCHIVE_TOOL}" validate-rvt-genx \
      --root "${RAW_ROOT}" --dataset "${DATASET}" --split "${current_split}" \
      "${VALIDATE_EXTRA_ARGS[@]}"
    continue
  fi

  archive="${ROOT}/archives/${current_split}.tar"
  crc32="$(rvt_crc32 "${current_split}")"
  download_url \
    "${archive}" "${BASE_URL}/${current_split}.tar" archive - - "${crc32}"

  if [ "${EXTRACT}" -eq 1 ]; then
    download_extract_archive "${archive}" "${RAW_ROOT}" "${ROOT}/state/extract"
    "${DOWNLOAD_PYTHON}" "${DOWNLOAD_ARCHIVE_TOOL}" validate-rvt-genx \
      --root "${RAW_ROOT}" --dataset "${DATASET}" --split "${current_split}" \
      "${VALIDATE_EXTRA_ARGS[@]}"
  fi
done

if [ -n "${EXTRACTED_ROOT}" ]; then
  download_note "RVT ${DATASET} existing HDF5 tree is structurally valid: ${RAW_ROOT}"
elif [ "${EXTRACT}" -eq 1 ]; then
  download_note "RVT ${DATASET} HDF5 data are ready below ${RAW_ROOT}"
else
  download_note "RVT ${DATASET} TAR download is complete; add --extract when space is ready"
fi
