#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

DSEC_BASE='https://download.ifi.uzh.ch/rpg/DSEC'
DSEC_LABEL_URL="${DSEC_BASE}/detection/dsec-det_left_object_detections.zip"
DSEC_LABEL_BYTES='4655023'
DSEC_EXTRA_VAL_URL="${DSEC_BASE}/train_object_detection_coarse/train_events.zip"
DSEC_EXTRA_VAL_BYTES='21906940828'
DSEC_EXTRA_TEST_URL="${DSEC_BASE}/test_object_detection_coarse/test_events.zip"
DSEC_EXTRA_TEST_BYTES='7450129771'

usage() {
  cat <<EOF
Usage: $(basename "$0") --root DIRECTORY --profile PROFILE [OPTIONS]

Profiles:
  detection-train  41 original train sequences, one left-event ZIP per sequence
  detection-val    Extra ZIP containing logical val 16_a through 21_a
  detection-test   12 original test sequences plus extra thun_02_a ZIP
  custom           Use --physical-split and --sequence-list

Options:
  --root DIRECTORY          Download state root (required)
  --profile PROFILE         One profile above (required)
  --sequence-list FILE      Required only for custom
  --physical-split SPLIT    train or test; required only for custom
  --include-calibration     Also download calibration ZIPs
  --without-labels          Do not fetch the 4.7 MB Detection label archive
  --extract                 Safely merge archives into ROOT/raw (off by default)
  --extract-to DIRECTORY    Extract into DIRECTORY and imply --extract
  --help                    Show this help

Authentication and GUI are not required. Downloads resume from .part files.
Archives are retained after extraction and are never deleted automatically.
EOF
}

ROOT=""
PROFILE=""
SEQUENCE_LIST=""
PHYSICAL_SPLIT=""
INCLUDE_CALIBRATION=0
INCLUDE_LABELS=1
EXTRACT=0
EXTRACT_TO=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --root)
      [ "$#" -ge 2 ] || download_die "--root requires a value"
      ROOT="$2"
      shift 2
      ;;
    --profile)
      [ "$#" -ge 2 ] || download_die "--profile requires a value"
      PROFILE="$2"
      shift 2
      ;;
    --sequence-list)
      [ "$#" -ge 2 ] || download_die "--sequence-list requires a value"
      SEQUENCE_LIST="$2"
      shift 2
      ;;
    --physical-split)
      [ "$#" -ge 2 ] || download_die "--physical-split requires a value"
      PHYSICAL_SPLIT="$2"
      shift 2
      ;;
    --include-calibration)
      INCLUDE_CALIBRATION=1
      shift
      ;;
    --without-labels)
      INCLUDE_LABELS=0
      shift
      ;;
    --extract)
      EXTRACT=1
      shift
      ;;
    --extract-to)
      [ "$#" -ge 2 ] || download_die "--extract-to requires a value"
      EXTRACT=1
      EXTRACT_TO="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *) download_die "unknown argument: $1" ;;
  esac
done

[ -n "${ROOT}" ] || download_die "--root is required"
case "${PROFILE}" in
  detection-train)
    PHYSICAL_SPLIT=train
    SEQUENCE_LIST="${PROJECT_ROOT}/configs/datasets/dsec_detection_train.txt"
    ;;
  detection-val)
    PHYSICAL_SPLIT=train
    ;;
  detection-test)
    PHYSICAL_SPLIT=test
    SEQUENCE_LIST="${PROJECT_ROOT}/configs/datasets/dsec_detection_test.txt"
    ;;
  custom)
    case "${PHYSICAL_SPLIT}" in train|test) ;; *)
      download_die "custom profile requires --physical-split train or test"
    esac
    [ -f "${SEQUENCE_LIST}" ] || download_die "custom profile requires --sequence-list"
    ;;
  *) download_die "--profile must be detection-train, detection-val, detection-test, or custom" ;;
esac

download_require_runtime
ARCHIVE_ROOT="${ROOT}/archives"
STATE_ROOT="${ROOT}/.download-state/extract"
if [ -z "${EXTRACT_TO}" ]; then
  EXTRACT_TO="${ROOT}/raw"
fi
mkdir -p "${ARCHIVE_ROOT}/${PROFILE}" "${ROOT}/.download-state"

record_archive() {
  local archive="$1"
  local output="${2:-${EXTRACT_TO}}"
  if [ "${EXTRACT}" -eq 1 ]; then
    download_extract_archive "${archive}" "${output}" "${STATE_ROOT}"
  fi
}

extract_detection_labels() {
  local outer_archive="$1"
  [ "${EXTRACT}" -eq 1 ] || return 0

  # The small all-labels archive is an archive-of-archives.  Expanding it
  # directly below raw/ leaves train_object_detections.zip and
  # test_object_detections.zip unopened, so stage those archives beside the
  # downloaded artifacts and then merge their contents into the DSEC tree.
  local nested_root="${ARCHIVE_ROOT}/detection-labels"
  mkdir -p "${nested_root}"
  download_extract_archive "${outer_archive}" "${nested_root}" "${STATE_ROOT}"

  local split nested_archive
  for split in train test; do
    nested_archive="${nested_root}/${split}_object_detections.zip"
    [ -f "${nested_archive}" ] || \
      download_die "DSEC label bundle lacks $(basename "${nested_archive}")"
    download_extract_archive "${nested_archive}" "${EXTRACT_TO}" "${STATE_ROOT}"
  done
}

download_sequence() {
  local sequence="$1"
  local events_name="${sequence}_events_left.zip"
  local sequence_base="${DSEC_BASE}/${PHYSICAL_SPLIT}/${sequence}"
  local events_output="${ARCHIVE_ROOT}/${PROFILE}/${events_name}"
  download_url "${events_output}" "${sequence_base}/${events_name}" archive - -
  # Individual DSEC event ZIPs contain bare events.h5/rectify_map.h5 members.
  # They must be extracted into the sequence/camera directory; extracting all
  # of them into raw/ would overwrite the preceding sequence.
  record_archive \
    "${events_output}" \
    "${EXTRACT_TO}/${PHYSICAL_SPLIT}/${sequence}/events/left"
  if [ "${INCLUDE_CALIBRATION}" -eq 1 ]; then
    local calibration_name="${sequence}_calibration.zip"
    local calibration_output="${ARCHIVE_ROOT}/${PROFILE}/${calibration_name}"
    download_url \
      "${calibration_output}" "${sequence_base}/${calibration_name}" archive - -
    record_archive \
      "${calibration_output}" \
      "${EXTRACT_TO}/${PHYSICAL_SPLIT}/${sequence}/calibration"
  fi
}

if [ "${PROFILE}" = "detection-val" ]; then
  extra_output="${ARCHIVE_ROOT}/${PROFILE}/train_events.zip"
  download_url \
    "${extra_output}" "${DSEC_EXTRA_VAL_URL}" archive - "${DSEC_EXTRA_VAL_BYTES}"
  record_archive "${extra_output}"
  if [ "${INCLUDE_CALIBRATION}" -eq 1 ]; then
    calibration_output="${ARCHIVE_ROOT}/${PROFILE}/train_calibration.zip"
    download_url "${calibration_output}" \
      "${DSEC_BASE}/train_object_detection_coarse/train_calibration.zip" archive - -
    record_archive "${calibration_output}"
  fi
else
  [ -f "${SEQUENCE_LIST}" ] || download_die "sequence list not found: ${SEQUENCE_LIST}"
  sequence_count=0
  while IFS= read -r raw_sequence || [ -n "${raw_sequence}" ]; do
    sequence="${raw_sequence#"${raw_sequence%%[![:space:]]*}"}"
    sequence="${sequence%"${sequence##*[![:space:]]}"}"
    case "${sequence}" in
      ""|\#*) continue ;;
      *[!A-Za-z0-9._+-]*) download_die "unsafe DSEC sequence name: ${sequence}" ;;
    esac
    if [ "${PROFILE}" = "detection-test" ] && [ "${sequence}" = "thun_02_a" ]; then
      continue
    fi
    download_sequence "${sequence}"
    sequence_count=$((sequence_count + 1))
  done < "${SEQUENCE_LIST}"
  [ "${sequence_count}" -gt 0 ] || download_die "sequence list is empty"

  if [ "${PROFILE}" = "detection-test" ]; then
    extra_output="${ARCHIVE_ROOT}/${PROFILE}/test_events.zip"
    download_url \
      "${extra_output}" "${DSEC_EXTRA_TEST_URL}" archive - "${DSEC_EXTRA_TEST_BYTES}"
    record_archive "${extra_output}"
    if [ "${INCLUDE_CALIBRATION}" -eq 1 ]; then
      calibration_output="${ARCHIVE_ROOT}/${PROFILE}/test_calibration.zip"
      download_url "${calibration_output}" \
        "${DSEC_BASE}/test_object_detection_coarse/test_calibration.zip" archive - -
      record_archive "${calibration_output}"
    fi
  fi
fi

if [ "${INCLUDE_LABELS}" -eq 1 ]; then
  labels_output="${ARCHIVE_ROOT}/dsec-det_left_object_detections.zip"
  download_url "${labels_output}" "${DSEC_LABEL_URL}" archive - "${DSEC_LABEL_BYTES}"
  extract_detection_labels "${labels_output}"
fi

if [ "${EXTRACT}" -eq 1 ]; then
  download_note "DSEC ${PROFILE} was merged below ${EXTRACT_TO}"
else
  download_note "DSEC ${PROFILE} archives are ready below ${ARCHIVE_ROOT}"
  download_note "rerun the same command with --extract when enough working space is available"
fi
