#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RVT_DOWNLOAD_PROGRAM_NAME="$(basename "$0")"
exec "${SCRIPT_DIR}/_download_rvt_genx_h5.sh" gen4 "$@"
