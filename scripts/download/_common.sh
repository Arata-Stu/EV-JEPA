#!/usr/bin/env bash

# Shared, Bash 3.2-compatible helpers for large resumable downloads.

umask 077

DOWNLOAD_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOWNLOAD_ARCHIVE_TOOL="${DOWNLOAD_SCRIPT_DIR}/archive_tool.py"
DOWNLOAD_PYTHON="${PYTHON_BIN:-python3}"
DOWNLOAD_TEMP_FILES=("")
DOWNLOAD_LOCK_DIRS=("")
DOWNLOAD_LAST_TEMP=""
DOWNLOAD_LAST_LOCK=""
DOWNLOAD_REMOTE_BYTES=""

_download_cleanup_runtime() {
  local path
  for path in "${DOWNLOAD_TEMP_FILES[@]}"; do
    [ -n "${path}" ] && rm -f -- "${path}"
  done
  for path in "${DOWNLOAD_LOCK_DIRS[@]}"; do
    [ -n "${path}" ] && rmdir -- "${path}" 2>/dev/null || true
  done
}
trap _download_cleanup_runtime EXIT

_download_make_temp_file() {
  DOWNLOAD_LAST_TEMP="$(mktemp "${TMPDIR:-/tmp}/event-window-jepa-download.XXXXXX")"
  DOWNLOAD_TEMP_FILES+=("${DOWNLOAD_LAST_TEMP}")
}

_download_acquire_lock() {
  DOWNLOAD_LAST_LOCK="$1.lock"
  if ! mkdir "${DOWNLOAD_LAST_LOCK}" 2>/dev/null; then
    download_die "output is locked by another process or a stale lock: ${DOWNLOAD_LAST_LOCK}"
  fi
  DOWNLOAD_LOCK_DIRS+=("${DOWNLOAD_LAST_LOCK}")
}

_download_release_lock() {
  local lock_dir="$1"
  rmdir -- "${lock_dir}" 2>/dev/null || \
    download_die "could not release output lock: $1"
  local index
  for index in "${!DOWNLOAD_LOCK_DIRS[@]}"; do
    if [ "${DOWNLOAD_LOCK_DIRS[$index]}" = "${lock_dir}" ]; then
      unset "DOWNLOAD_LOCK_DIRS[$index]"
    fi
  done
}

download_die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

download_note() {
  printf '[dataset-download] %s\n' "$*" >&2
}

download_require_runtime() {
  command -v curl >/dev/null 2>&1 || download_die "curl is required"
  command -v "${DOWNLOAD_PYTHON}" >/dev/null 2>&1 || \
    download_die "${DOWNLOAD_PYTHON} is required (Python standard library only)"
  [ -f "${DOWNLOAD_ARCHIVE_TOOL}" ] || \
    download_die "missing helper: ${DOWNLOAD_ARCHIVE_TOOL}"
}

download_validate_filename() {
  local name="$1"
  case "${name}" in
    ""|.*|*/*|*\\*|*[!A-Za-z0-9._+-]*)
      download_die "unsafe filename '${name}'; use only A-Z, a-z, 0-9, dot, underscore, plus, and dash"
      ;;
  esac
  case "${name}" in
    *.part|*.verified.json|*.remote.json|*.sha256|*.crdownload|*.download)
      download_die "filename uses a reserved download-state suffix: ${name}"
      ;;
  esac
}

download_validate_url() {
  case "$1" in
    https://*) ;;
    *) download_die "only HTTPS download URLs are accepted" ;;
  esac
}

_download_make_curl_config() {
  local escaped="$1"
  escaped="${escaped//\\/\\\\}"
  escaped="${escaped//\"/\\\"}"
  _download_make_temp_file
  printf 'url = "%s"\n' "${escaped}" > "${DOWNLOAD_LAST_TEMP}"
}

_download_curl_retry_args() {
  if curl --help all 2>/dev/null | grep -q -- '--retry-all-errors'; then
    printf '%s\n' '--retry-all-errors'
  fi
}

_download_remote_identity() {
  local url="$1"
  local header_file="$2"
  local identity_file="$3"
  local retry_all=""
  retry_all="$(_download_curl_retry_args)"
  _download_make_curl_config "${url}"
  local curl_config="${DOWNLOAD_LAST_TEMP}"
  local args
  args=(
    --fail
    --silent
    --show-error
    --location
    --head
    --proto '=https'
    --proto-redir '=https'
    --connect-timeout 30
    --retry 4
    --retry-delay 3
    --dump-header "${header_file}"
    --output /dev/null
  )
  if [ -n "${retry_all}" ]; then
    args+=("${retry_all}")
  fi
  if ! curl -q "${args[@]}" --config "${curl_config}"; then
    args=(
      --fail
      --silent
      --show-error
      --location
      --range 0-0
      --max-filesize 1
      --max-time 30
      --proto '=https'
      --proto-redir '=https'
      --connect-timeout 30
      --retry 4
      --retry-delay 3
      --dump-header "${header_file}"
      --output /dev/null
    )
    if [ -n "${retry_all}" ]; then
      args+=("${retry_all}")
    fi
    if ! curl -q "${args[@]}" --config "${curl_config}"; then
      rm -f -- "${curl_config}"
      return 1
    fi
  fi
  rm -f -- "${curl_config}"
  DOWNLOAD_REMOTE_BYTES="$(
    "${DOWNLOAD_PYTHON}" "${DOWNLOAD_ARCHIVE_TOOL}" http-identity \
      --headers "${header_file}" --metadata "${identity_file}"
  )"
}

_download_file_size() {
  "${DOWNLOAD_PYTHON}" -c \
    'import os, sys; print(os.path.getsize(sys.argv[1]))' "$1"
}

_download_if_range_value() {
  local metadata="$1"
  "${DOWNLOAD_PYTHON}" -c \
    'import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
etag = str(data.get("etag", ""))
print(etag if etag and not etag.startswith("W/") else data.get("last_modified", ""))' \
    "${metadata}"
}

_download_compare_remote_identity() {
  local previous="$1"
  local current="$2"
  local allow_weak="$3"
  if [ "${allow_weak}" -eq 1 ]; then
    "${DOWNLOAD_PYTHON}" "${DOWNLOAD_ARCHIVE_TOOL}" compare-http-identity \
      --previous "${previous}" --current "${current}" --allow-weak
  else
    "${DOWNLOAD_PYTHON}" "${DOWNLOAD_ARCHIVE_TOOL}" compare-http-identity \
      --previous "${previous}" --current "${current}"
  fi
}

download_verify_file() {
  local path="$1"
  local kind="$2"
  local expected_sha256="${3:--}"
  local expected_bytes="${4:--}"
  local metadata="${path}.verified.json"
  local expected_args
  expected_args=()
  if [ -n "${expected_sha256}" ] && [ "${expected_sha256}" != "-" ]; then
    expected_args+=(--expected-sha256 "${expected_sha256}")
  fi
  if [ -n "${expected_bytes}" ] && [ "${expected_bytes}" != "-" ]; then
    expected_args+=(--expected-bytes "${expected_bytes}")
  fi

  if "${DOWNLOAD_PYTHON}" "${DOWNLOAD_ARCHIVE_TOOL}" check \
      --path "${path}" --kind "${kind}" --metadata "${metadata}" \
      "${expected_args[@]}" >/dev/null 2>&1; then
    download_note "verified marker is current: $(basename "${path}")"
    return 0
  fi

  download_note "verifying size, content, and SHA-256: $(basename "${path}")"
  "${DOWNLOAD_PYTHON}" "${DOWNLOAD_ARCHIVE_TOOL}" verify \
    --path "${path}" --kind "${kind}" --metadata "${metadata}" \
    "${expected_args[@]}"
}

download_url() {
  local output="$1"
  local url="$2"
  local kind="$3"
  local expected_sha256="${4:--}"
  local expected_bytes="${5:--}"
  local allow_weak_identity=0
  if [ -n "${expected_sha256}" ] && [ "${expected_sha256}" != "-" ]; then
    if [ "${#expected_sha256}" -ne 64 ] || \
        [ -n "${expected_sha256//[[:xdigit:]]/}" ]; then
      download_die "expected SHA-256 must contain exactly 64 hexadecimal digits"
    fi
    allow_weak_identity=1
  fi

  download_validate_url "${url}"
  mkdir -p "$(dirname "${output}")"
  _download_acquire_lock "${output}"
  local lock_dir="${DOWNLOAD_LAST_LOCK}"
  local partial="${output}.part"
  local partial_metadata="${partial}.verified.json"
  local final_metadata="${output}.verified.json"
  local partial_remote="${partial}.remote.json"
  local final_remote="${output}.remote.json"

  if [ -f "${output}" ]; then
    download_verify_file "${output}" "${kind}" "${expected_sha256}" "${expected_bytes}"
    download_note "already complete: $(basename "${output}")"
    _download_release_lock "${lock_dir}"
    return 0
  fi

  _download_make_temp_file
  local header_file="${DOWNLOAD_LAST_TEMP}"
  _download_make_temp_file
  local current_remote="${DOWNLOAD_LAST_TEMP}"
  local remote_bytes=""
  if _download_remote_identity \
      "${url}" "${header_file}" "${current_remote}" 2>/dev/null; then
    remote_bytes="${DOWNLOAD_REMOTE_BYTES}"
    if [ -f "${partial}" ] && [ -s "${partial}" ]; then
      [ -f "${partial_remote}" ] || {
        rm -f "${header_file}" "${current_remote}"
        download_die "cannot safely resume ${partial}: remote identity sidecar is missing"
      }
      if ! _download_compare_remote_identity \
          "${partial_remote}" "${current_remote}" "${allow_weak_identity}" \
          >/dev/null; then
        rm -f "${header_file}" "${current_remote}"
        download_die "remote object changed; retained partial without appending: ${partial}"
      fi
    else
      if ! _download_compare_remote_identity \
          "${current_remote}" "${current_remote}" "${allow_weak_identity}" \
          >/dev/null; then
        rm -f "${header_file}" "${current_remote}"
        download_die "server lacks a strong ETag/provider checksum and no publisher SHA-256 was supplied"
      fi
      mv "${current_remote}" "${partial_remote}"
    fi
    if [ -n "${remote_bytes}" ]; then
      if [ "${expected_bytes}" != "-" ] && [ -n "${expected_bytes}" ] && \
          [ "${expected_bytes}" != "${remote_bytes}" ]; then
        rm -f "${header_file}"
        download_die "publisher size and URL Content-Length disagree for $(basename "${output}")"
      fi
      expected_bytes="${remote_bytes}"
    fi
  else
    rm -f "${header_file}" "${current_remote}"
    download_die "server did not expose a stable object identity; use a refreshed URL or browser --inbox mode"
  fi
  rm -f "${header_file}" "${current_remote}"

  if [ -f "${partial}" ] && [ -s "${partial}" ] && [ ! -f "${partial_remote}" ]; then
    download_die "cannot safely resume without Content-Length/ETag identity: ${partial}"
  fi

  if [ -f "${partial}" ] && [ "${expected_bytes}" != "-" ] && \
      [ -n "${expected_bytes}" ]; then
    local partial_bytes
    partial_bytes="$(_download_file_size "${partial}")"
    if [ "${partial_bytes}" -gt "${expected_bytes}" ]; then
      download_die "partial file is larger than the remote object: ${partial}"
    fi
    if [ "${partial_bytes}" -eq "${expected_bytes}" ]; then
      download_note "download bytes were already complete; verifying retained .part"
      download_verify_file "${partial}" "${kind}" "${expected_sha256}" "${expected_bytes}"
      mv "${partial}" "${output}"
      mv "${partial_metadata}" "${final_metadata}"
      if [ -f "${partial_remote}" ]; then
        mv "${partial_remote}" "${final_remote}"
      fi
      _download_release_lock "${lock_dir}"
      return 0
    fi
  fi

  download_note "downloading/resuming: $(basename "${output}")"
  local retry_all=""
  retry_all="$(_download_curl_retry_args)"
  local args
  args=(
    --fail
    --location
    --show-error
    --continue-at -
    --proto '=https'
    --proto-redir '=https'
    --connect-timeout 30
    --retry 12
    --retry-delay 5
    --speed-limit 1024
    --speed-time 120
    --output "${partial}"
  )
  if [ -n "${retry_all}" ]; then
    args+=("${retry_all}")
  fi
  if [ -f "${partial}" ] && [ -s "${partial}" ] && [ -f "${partial_remote}" ]; then
    local if_range
    if_range="$(_download_if_range_value "${partial_remote}")"
    if [ -n "${if_range}" ]; then
      args+=(--header "If-Range: ${if_range}")
    fi
  fi
  _download_make_curl_config "${url}"
  local curl_config="${DOWNLOAD_LAST_TEMP}"
  if ! curl -q "${args[@]}" --config "${curl_config}"; then
    download_die "download failed; retained for the next run: ${partial}"
  fi
  rm -f -- "${curl_config}"

  _download_make_temp_file
  local final_header_file="${DOWNLOAD_LAST_TEMP}"
  _download_make_temp_file
  local final_remote_check="${DOWNLOAD_LAST_TEMP}"
  if ! _download_remote_identity \
      "${url}" "${final_header_file}" "${final_remote_check}" >/dev/null 2>&1; then
    rm -f "${final_header_file}" "${final_remote_check}"
    download_die "could not re-check remote identity after download; retained: ${partial}"
  fi
  if ! _download_compare_remote_identity \
      "${partial_remote}" "${final_remote_check}" "${allow_weak_identity}" \
      >/dev/null; then
    rm -f "${final_header_file}" "${final_remote_check}"
    download_die "remote object changed during download; retained without publishing: ${partial}"
  fi
  rm -f "${final_header_file}" "${final_remote_check}"

  if ! download_verify_file \
      "${partial}" "${kind}" "${expected_sha256}" "${expected_bytes}"; then
    download_die "verification failed; retained for inspection: ${partial}"
  fi
  mv "${partial}" "${output}"
  mv "${partial_metadata}" "${final_metadata}"
  if [ -f "${partial_remote}" ]; then
    mv "${partial_remote}" "${final_remote}"
  fi
  download_note "complete: $(basename "${output}")"
  _download_release_lock "${lock_dir}"
}

download_extract_archive() {
  local archive="$1"
  local output="$2"
  local state_root="$3"
  download_verify_file "${archive}" archive - -
  "${DOWNLOAD_PYTHON}" "${DOWNLOAD_ARCHIVE_TOOL}" extract \
    --archive "${archive}" --output "${output}" --state-root "${state_root}"
}

download_validate_private_url_file() {
  local url_file="$1"
  [ -f "${url_file}" ] || download_die "URL file does not exist: ${url_file}"
  local line_number=0
  local rows=0
  local seen_filenames='|'
  local raw_line trimmed filename filename_key url sha bytes extra
  while IFS= read -r raw_line || [ -n "${raw_line}" ]; do
    line_number=$((line_number + 1))
    trimmed="${raw_line#"${raw_line%%[![:space:]]*}"}"
    case "${trimmed}" in
      ""|\#*) continue ;;
    esac
    IFS=' ' read -r filename url sha bytes extra <<< "${trimmed}"
    if [ -z "${filename:-}" ] || [ -z "${url:-}" ] || [ -n "${extra:-}" ]; then
      download_die "invalid URL file row ${line_number}; expected: filename URL [sha256|-] [bytes|-]"
    fi
    sha="${sha:--}"
    bytes="${bytes:--}"
    download_validate_filename "${filename}"
    download_validate_url "${url}"
    if [ "${sha}" != "-" ]; then
      case "${sha}" in *[!0-9A-Fa-f]*) download_die "invalid SHA-256 at row ${line_number}" ;; esac
      [ "${#sha}" -eq 64 ] || download_die "invalid SHA-256 at row ${line_number}"
    fi
    if [ "${bytes}" != "-" ]; then
      case "${bytes}" in ""|*[!0-9]*) download_die "invalid byte count at row ${line_number}" ;; esac
      [ "${bytes}" -gt 0 ] || download_die "invalid byte count at row ${line_number}"
    fi
    filename_key="$(printf '%s' "${filename}" | tr '[:upper:]' '[:lower:]')"
    case "${seen_filenames}" in
      *"|${filename_key}|"*) download_die "duplicate filename in URL file: ${filename}" ;;
    esac
    seen_filenames="${seen_filenames}${filename_key}|"
    rows=$((rows + 1))
  done < "${url_file}"
  [ "${rows}" -gt 0 ] || download_die "URL file has no download rows: ${url_file}"
}

download_process_private_url_file() {
  local url_file="$1"
  local archive_root="$2"
  download_validate_private_url_file "${url_file}"
  local line_number=0
  local processed=0
  local seen_filenames='|'
  local raw_line trimmed filename filename_key url sha bytes extra
  while IFS= read -r raw_line || [ -n "${raw_line}" ]; do
    line_number=$((line_number + 1))
    trimmed="${raw_line#"${raw_line%%[![:space:]]*}"}"
    case "${trimmed}" in
      ""|\#*) continue ;;
    esac
    filename=""
    url=""
    sha=""
    bytes=""
    extra=""
    IFS=' ' read -r filename url sha bytes extra <<< "${trimmed}"
    if [ -z "${filename}" ] || [ -z "${url}" ] || [ -n "${extra}" ]; then
      download_die "invalid URL file row ${line_number}; expected: filename URL [sha256|-] [bytes|-]"
    fi
    sha="${sha:--}"
    bytes="${bytes:--}"
    download_validate_filename "${filename}"
    filename_key="$(printf '%s' "${filename}" | tr '[:upper:]' '[:lower:]')"
    case "${seen_filenames}" in
      *"|${filename_key}|"*) download_die "duplicate filename in URL file: ${filename}" ;;
    esac
    seen_filenames="${seen_filenames}${filename_key}|"
    download_url "${archive_root}/${filename}" "${url}" archive "${sha}" "${bytes}"
    processed=$((processed + 1))
  done < "${url_file}"
  [ "${processed}" -gt 0 ] || download_die "URL file has no download rows: ${url_file}"
}

download_for_each_inbox_archive() {
  local inbox="$1"
  local callback="$2"
  [ -d "${inbox}" ] || download_die "inbox directory does not exist: ${inbox}"
  local found=0
  local candidate
  for candidate in "${inbox}"/*; do
    [ -f "${candidate}" ] || continue
    case "${candidate}" in
      */.*|*.part|*.crdownload|*.download|*.verified.json|*.remote.json|*.sha256|*.headers.*) continue ;;
    esac
    found=$((found + 1))
    "${callback}" "${candidate}"
  done
  [ "${found}" -gt 0 ] || download_die "no archive files found in inbox: ${inbox}"
}
