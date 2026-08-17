#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/_download_prophesee.sh" \
  prophesee_1mpx \
  1280 \
  720 \
  'https://www.prophesee.ai/2020/11/24/automotive-megapixel-event-based-dataset/' \
  'https://forms.zohopublic.com/itdesk175/form/Dataset1MegapixelAutomotiveDetection/formperma/m8gOxbwaLFXc2PaLpalHNXeKpq4Tdci1DL0Ynx8q_FE' \
  "$@"

