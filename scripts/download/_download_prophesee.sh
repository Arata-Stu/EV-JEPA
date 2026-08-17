#!/usr/bin/env bash
set -euo pipefail
set +x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

DATASET_ID="$1"
EXPECTED_WIDTH="$2"
EXPECTED_HEIGHT="$3"
LANDING_URL="$4"
FORM_URL="$5"
shift 5

case "${DATASET_ID}" in
  gen1) DISPLAY_SCRIPT='download_prophesee_gen1_dat.sh' ;;
  prophesee_1mpx) DISPLAY_SCRIPT='download_prophesee_1mpx.sh' ;;
  *) DISPLAY_SCRIPT='_download_prophesee.sh' ;;
esac

usage() {
  cat <<EOF
Usage: ${DISPLAY_SCRIPT} --root DIRECTORY [SOURCE] [--extract]

Sources (one or more):
  --url-file FILE   Private rows: filename URL [sha256|-] [bytes|-]
  --inbox DIRECTORY Archives downloaded manually in a browser
  --extracted-root DIRECTORY  One split already extracted manually

Options:
  --root DIRECTORY  Download state root (required)
  --split SPLIT     train, val, or test (required)
  --extract         Safely extract archives into ROOT/raw (off by default)
  --extract-to DIR  Use DIR as raw base and imply --extract
  --help            Show this help

The official form/CAPTCHA must be completed manually first:
  ${LANDING_URL}
  ${FORM_URL}

Neither archives nor source files are deleted by this script.
EOF
}

ROOT=""
URL_FILE=""
INBOX=""
EXTRACTED_ROOT=""
SPLIT=""
EXTRACT=0
EXTRACT_TO=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --root)
      [ "$#" -ge 2 ] || download_die "--root requires a value"
      ROOT="$2"
      shift 2
      ;;
    --url-file)
      [ "$#" -ge 2 ] || download_die "--url-file requires a value"
      URL_FILE="$2"
      shift 2
      ;;
    --inbox)
      [ "$#" -ge 2 ] || download_die "--inbox requires a value"
      INBOX="$2"
      shift 2
      ;;
    --extracted-root)
      [ "$#" -ge 2 ] || download_die "--extracted-root requires a value"
      EXTRACTED_ROOT="$2"
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
    *)
      download_die "unknown argument: $1"
      ;;
  esac
done

[ -n "${ROOT}" ] || {
  usage >&2
  download_die "--root is required"
}
case "${SPLIT}" in
  train|val|test) ;;
  *) download_die "--split must be train, val, or test" ;;
esac
if [ -z "${URL_FILE}" ] && [ -z "${INBOX}" ] && [ -z "${EXTRACTED_ROOT}" ]; then
  usage >&2
  download_die "complete the official form, then provide a download or extracted source"
fi

download_require_runtime
ARCHIVE_ROOT="${ROOT}/archives/${SPLIT}"
STATE_ROOT="${ROOT}/.download-state/extract"
if [ -z "${EXTRACT_TO}" ]; then
  EXTRACT_TO="${ROOT}/raw/${SPLIT}"
else
  EXTRACT_TO="${EXTRACT_TO}/${SPLIT}"
fi
mkdir -p "${ARCHIVE_ROOT}" "${ROOT}/.download-state"

if [ -n "${URL_FILE}" ]; then
  [ -f "${URL_FILE}" ] || download_die "URL file does not exist: ${URL_FILE}"
  permission_bits="$("${DOWNLOAD_PYTHON}" -c \
    'import os, stat, sys; print(oct(stat.S_IMODE(os.stat(sys.argv[1]).st_mode)))' \
    "${URL_FILE}")"
  case "${permission_bits}" in
    0o600|0o400) ;;
    *) download_note "warning: private URL file permissions are ${permission_bits}; chmod 600 is recommended" ;;
  esac
  download_process_private_url_file "${URL_FILE}" "${ARCHIVE_ROOT}"
fi

PROTECTED_EXTRACT="${EXTRACT}"
PROTECTED_EXTRACT_TO="${EXTRACT_TO}"
PROTECTED_STATE_ROOT="${STATE_ROOT}"
process_protected_archive() {
  local archive="$1"
  download_verify_file "${archive}" archive - -
  if [ "${PROTECTED_EXTRACT}" -eq 1 ]; then
    download_extract_archive \
      "${archive}" "${PROTECTED_EXTRACT_TO}" "${PROTECTED_STATE_ROOT}"
  fi
}

if [ -n "${URL_FILE}" ]; then
  download_for_each_inbox_archive "${ARCHIVE_ROOT}" process_protected_archive
fi
if [ -n "${INBOX}" ]; then
  download_for_each_inbox_archive "${INBOX}" process_protected_archive
fi

if [ "${EXTRACT}" -eq 1 ]; then
  "${DOWNLOAD_PYTHON}" "${DOWNLOAD_ARCHIVE_TOOL}" validate-prophesee \
    --root "${EXTRACT_TO}" --width "${EXPECTED_WIDTH}" --height "${EXPECTED_HEIGHT}" \
    --split "${SPLIT}"
fi
if [ -n "${EXTRACTED_ROOT}" ]; then
  [ -d "${EXTRACTED_ROOT}" ] || download_die \
    "extracted root does not exist: ${EXTRACTED_ROOT}"
  "${DOWNLOAD_PYTHON}" "${DOWNLOAD_ARCHIVE_TOOL}" validate-prophesee \
    --root "${EXTRACTED_ROOT}" --width "${EXPECTED_WIDTH}" --height "${EXPECTED_HEIGHT}" \
    --split "${SPLIT}"
fi

download_note "${DATASET_ID} acquisition step completed"
if [ "${EXTRACT}" -eq 0 ]; then
  download_note "archives were retained without extraction; rerun with --extract when ready"
fi
