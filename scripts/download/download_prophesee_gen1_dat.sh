#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/_download_prophesee.sh" \
  gen1 \
  304 \
  240 \
  'https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/' \
  'https://forms.zohopublic.com/itdesk175/form/DatasetAtisAutomotiveDetection/formperma/c8fMk4X9Y2P5f-H8kXUHULiDXyGp4a04i027OnpQePQ' \
  "$@"
