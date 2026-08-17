#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

M3ED_BASE_URL='https://m3ed-dist.s3.us-west-2.amazonaws.com'
M3ED_LIST_COMMIT='df739f20fba41ac6da8c22f4260c305875e391ed'
M3ED_LIST_URL="https://raw.githubusercontent.com/daniilidis-group/m3ed/${M3ED_LIST_COMMIT}/dataset_list.yaml"
M3ED_LIST_SHA256='92e3fd3e0f316459a30a61b894e7510744d44d7008623b0a74a19fb4233da8c0'
M3ED_LIST_BYTES='36090'

usage() {
  cat <<EOF
Usage: $(basename "$0") --root DIRECTORY --split SPLIT --sequence-list FILE

Options:
  --root DIRECTORY       Store official HDF5 files below ROOT/raw (required)
  --split train|val|test Logical split; checked against official is_test_file
  --sequence-list FILE   One official M3ED recording name per line (required)
  --dataset-list FILE    Use an existing official dataset_list.yaml
  --help                 Show this help

Only processed *_data.h5 files are downloaded. Videos, ROS bags, depth, pose,
and point clouds are intentionally omitted. Re-running resumes each .part file.
EOF
}

ROOT=""
SPLIT=""
SEQUENCE_LIST=""
DATASET_LIST=""
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
    --sequence-list)
      [ "$#" -ge 2 ] || download_die "--sequence-list requires a value"
      SEQUENCE_LIST="$2"
      shift 2
      ;;
    --dataset-list)
      [ "$#" -ge 2 ] || download_die "--dataset-list requires a value"
      DATASET_LIST="$2"
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
case "${SPLIT}" in
  train|val|test) ;;
  *) download_die "--split must be train, val, or test" ;;
esac
[ -f "${SEQUENCE_LIST}" ] || download_die "sequence list not found: ${SEQUENCE_LIST}"

download_require_runtime
mkdir -p "${ROOT}/raw" "${ROOT}/metadata"
if [ -z "${DATASET_LIST}" ]; then
  DATASET_LIST="${ROOT}/metadata/dataset_list.yaml"
  download_url "${DATASET_LIST}" "${M3ED_LIST_URL}" text \
    "${M3ED_LIST_SHA256}" "${M3ED_LIST_BYTES}"
else
  [ -f "${DATASET_LIST}" ] || download_die "dataset list not found: ${DATASET_LIST}"
  download_verify_file "${DATASET_LIST}" text - -
fi

"${DOWNLOAD_PYTHON}" "${DOWNLOAD_ARCHIVE_TOOL}" validate-m3ed-plan \
  --dataset-list "${DATASET_LIST}" --sequence-list "${SEQUENCE_LIST}" --split "${SPLIT}"

count=0
while IFS= read -r raw_name || [ -n "${raw_name}" ]; do
  name="${raw_name#"${raw_name%%[![:space:]]*}"}"
  name="${name%"${name##*[![:space:]]}"}"
  case "${name}" in
    ""|\#*) continue ;;
    *[!A-Za-z0-9._+-]*) download_die "unsafe M3ED sequence name: ${name}" ;;
  esac
  output="${ROOT}/raw/${name}/${name}_data.h5"
  url="${M3ED_BASE_URL}/processed/${name}/${name}_data.h5"
  download_url "${output}" "${url}" hdf5 - -
  count=$((count + 1))
done < "${SEQUENCE_LIST}"

[ "${count}" -gt 0 ] || download_die "sequence list is empty"
download_note "M3ED ${SPLIT}: ${count} processed data file(s) are ready below ${ROOT}/raw"
download_note "official split metadata: ${DATASET_LIST}"
