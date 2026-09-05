#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
HELPER="$PROJECT_ROOT/scripts/experiments/mvsec_ablation_config.py"

ACTION=plan
MODE=compare
STAGE=dev
SUITE=core
SUITE_EXPLICIT=0
SEEDS=0,1,2
SEEDS_EXPLICIT=0
PROBE_SEEDS=0
SELECTED_RUN_ID=""
PRETRAIN_EPOCH=100
TASK=flow-random
ALIGNMENT=causal
DT=native
PYTHON_BIN=${PYTHON_BIN:-python3}
DEVICE=cuda
DATA_ROOT="$PROJECT_ROOT/data/mvsec/processed"
TRAIN_MANIFEST=""
ARTIFACT_ROOT="$PROJECT_ROOT/outputs/experiments/mvsec_ablation"
EVAL_ROOT=""
CHECKPOINT_ROOT=""
OUTPUT_ROOT=""
SAMPLE_LIMIT=2
MAXIMUM_SNAPSHOT_MIB=256
CMAX_SAMPLE_INDEX=0
CMAX_CALIBRATION_SAMPLES=4
QUIVER_STRIDE=1
ALLOW_INCOMPATIBLE=0
ALLOW_LARGE_MATRIX=0

usage() {
  printf '%s\n' \
    'Usage: bash scripts/experiments/visualize_mvsec_ablation.sh [options]' \
    '' \
    '  --action plan|run         Default plan never writes files' \
    '  --mode MODE               compare, samples, cmax-raw, or all' \
    '  --stage dev|final         Match the evaluation artifact stage' \
    '  --suite NAME              Named suite for dev visualization' \
    '  --seeds CSV               Pretrain seeds for dev visualization' \
    '  --probe-seeds CSV         Head seeds present in eval outputs' \
    '  --selected-run-id ID      Required for final-stage visualization' \
    '  --pretrain-epoch N        Evaluated milestone epoch (default: 100)' \
    '  --task TASK               flow-random, flow-cmax-init, cmax-direct, depth' \
    '  --alignment MODE          causal or f3_centered' \
    '  --dt RATE                 native or dt1; flow tasks only' \
    '  --data-root DIR           Processed MVSEC root' \
    '  --train-manifest FILE     day2 manifest for CMax raw-warp views' \
    '  --artifact-root DIR       Parent used by train/eval runners' \
    '  --eval-root DIR           Evaluation root override' \
    '  --checkpoint-root DIR     Pretrain run root override' \
    '  --output-root DIR         New visualization root' \
    '  --python-bin PATH         Target Python executable' \
    '  --device DEVICE           Device for CMax raw-warp views' \
    '  --sample-limit N          Snapshots rendered per eval job (default: 2)' \
    '  --maximum-snapshot-mib N  Safe NPZ read ceiling (default: 256)' \
    '  --cmax-sample-index N     CMax raw-warp dataset sample (default: 0)' \
    '  --cmax-calibration-samples N  Must be at least 2 (default: 4)' \
    '  --quiver-stride N         CMax quiver subsampling (default: 1)' \
    '  --allow-incompatible      Explicit exploratory report comparison' \
    '  --allow-large-matrix      Permit more than 48 render jobs' \
    '  -h, --help                Show this help' \
    '' \
    'compare never scans a root recursively: it passes only one explicit' \
    'task/alignment/dt contract to the fail-closed comparison CLI. samples' \
    'consumes only NPZ paths listed in each visualizations/index.json.' \
    'Outputs include the named suite/selected run and an explicit report-set' \
    'hash, so staged comparisons cannot overwrite another comparison.' \
    'Multiple probe seeds are averaged within each encoder seed before the' \
    'encoder-seed mean/std is computed; mismatched seed sets are rejected.' \
    'Named RA/LS suites: rate_alignment, rate_gamma, straightening, latent,' \
    'and latent_cmax. Temporal SIGReg remains a separate suite.'
}

need_value() { if (($# < 2)); then printf 'Missing value for %s\n' "$1" >&2; exit 2; fi; }

while (($#)); do
  case "$1" in
    --action) need_value "$@"; ACTION=$2; shift 2 ;;
    --mode) need_value "$@"; MODE=$2; shift 2 ;;
    --stage) need_value "$@"; STAGE=$2; shift 2 ;;
    --suite) need_value "$@"; SUITE=$2; SUITE_EXPLICIT=1; shift 2 ;;
    --seeds) need_value "$@"; SEEDS=$2; SEEDS_EXPLICIT=1; shift 2 ;;
    --probe-seeds) need_value "$@"; PROBE_SEEDS=$2; shift 2 ;;
    --selected-run-id) need_value "$@"; SELECTED_RUN_ID=$2; shift 2 ;;
    --pretrain-epoch) need_value "$@"; PRETRAIN_EPOCH=$2; shift 2 ;;
    --task) need_value "$@"; TASK=$2; shift 2 ;;
    --alignment) need_value "$@"; ALIGNMENT=$2; shift 2 ;;
    --dt) need_value "$@"; DT=$2; shift 2 ;;
    --data-root) need_value "$@"; DATA_ROOT=$2; shift 2 ;;
    --train-manifest) need_value "$@"; TRAIN_MANIFEST=$2; shift 2 ;;
    --artifact-root) need_value "$@"; ARTIFACT_ROOT=$2; shift 2 ;;
    --eval-root) need_value "$@"; EVAL_ROOT=$2; shift 2 ;;
    --checkpoint-root) need_value "$@"; CHECKPOINT_ROOT=$2; shift 2 ;;
    --output-root) need_value "$@"; OUTPUT_ROOT=$2; shift 2 ;;
    --python-bin) need_value "$@"; PYTHON_BIN=$2; shift 2 ;;
    --device) need_value "$@"; DEVICE=$2; shift 2 ;;
    --sample-limit) need_value "$@"; SAMPLE_LIMIT=$2; shift 2 ;;
    --maximum-snapshot-mib) need_value "$@"; MAXIMUM_SNAPSHOT_MIB=$2; shift 2 ;;
    --cmax-sample-index) need_value "$@"; CMAX_SAMPLE_INDEX=$2; shift 2 ;;
    --cmax-calibration-samples) need_value "$@"; CMAX_CALIBRATION_SAMPLES=$2; shift 2 ;;
    --quiver-stride) need_value "$@"; QUIVER_STRIDE=$2; shift 2 ;;
    --allow-incompatible) ALLOW_INCOMPATIBLE=1; shift ;;
    --allow-large-matrix) ALLOW_LARGE_MATRIX=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$ACTION" in plan|run) ;; *) printf 'Invalid action: %s\n' "$ACTION" >&2; exit 2 ;; esac
case "$MODE" in compare|samples|cmax-raw|all) ;; *) printf 'Invalid mode: %s\n' "$MODE" >&2; exit 2 ;; esac
case "$STAGE" in dev|final) ;; *) printf 'Invalid stage: %s\n' "$STAGE" >&2; exit 2 ;; esac
case "$SUITE" in core|temporal_sigreg|cmax|context|interaction|reference|scales|frame|rate_alignment|rate_gamma|straightening|latent|latent_cmax|all) ;; *) printf 'Invalid suite: %s\n' "$SUITE" >&2; exit 2 ;; esac
case "$TASK" in flow-random|flow-cmax-init|cmax-direct|depth) ;; *) printf 'Invalid task: %s\n' "$TASK" >&2; exit 2 ;; esac
case "$ALIGNMENT" in causal|f3_centered) ;; *) printf 'Invalid alignment: %s\n' "$ALIGNMENT" >&2; exit 2 ;; esac
case "$DT" in native|dt1) ;; *) printf 'Invalid flow rate: %s\n' "$DT" >&2; exit 2 ;; esac
if [[ "$STAGE" == final ]]; then
  [[ -n "$SELECTED_RUN_ID" ]] || { printf '%s\n' '--selected-run-id is required for final stage.' >&2; exit 2; }
  if ((SUITE_EXPLICIT || SEEDS_EXPLICIT)); then
    printf '%s\n' 'Final visualization accepts one --selected-run-id, not --suite/--seeds.' >&2
    exit 2
  fi
else
  [[ -z "$SELECTED_RUN_ID" ]] || { printf '%s\n' '--selected-run-id is final-stage only.' >&2; exit 2; }
fi
if [[ "$TASK" == depth && "$DT" != native ]]; then
  printf '%s\n' 'Depth has no dt axis; use --dt native.' >&2
  exit 2
fi
for numeric in "$PRETRAIN_EPOCH" "$MAXIMUM_SNAPSHOT_MIB" "$CMAX_CALIBRATION_SAMPLES" "$QUIVER_STRIDE"; do
  case "$numeric" in ''|0|0[0-9]*|*[!0-9]*) printf 'Expected a positive integer, got: %s\n' "$numeric" >&2; exit 2 ;; esac
done
for numeric in "$SAMPLE_LIMIT" "$CMAX_SAMPLE_INDEX"; do
  case "$numeric" in ''|0[0-9]*|*[!0-9]*) printf 'Expected a non-negative integer, got: %s\n' "$numeric" >&2; exit 2 ;; esac
done
if ((CMAX_CALIBRATION_SAMPLES < 2)); then
  printf '%s\n' '--cmax-calibration-samples must be at least 2.' >&2
  exit 2
fi
[[ -f "$HELPER" ]] || { printf 'Missing helper: %s\n' "$HELPER" >&2; exit 1; }
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  printf 'Python executable is unavailable: %s\n' "$PYTHON_BIN" >&2
  exit 1
fi
PYTHON_BIN=$(command -v "$PYTHON_BIN")
# Preserve an env/bin/python symlink so Python retains virtual-environment
# discovery instead of being invoked through its base interpreter path.
PYTHON_BIN_DIR=$(cd "$(dirname "$PYTHON_BIN")" && pwd -P)
PYTHON_BIN="$PYTHON_BIN_DIR/$(basename "$PYTHON_BIN")"

absolute_path() { "$PYTHON_BIN" "$HELPER" path "$1"; }
DATA_ROOT=$(absolute_path "$DATA_ROOT")
ARTIFACT_ROOT=$(absolute_path "$ARTIFACT_ROOT")
if [[ -z "$TRAIN_MANIFEST" ]]; then TRAIN_MANIFEST="$DATA_ROOT/manifests/train.jsonl"; fi
if [[ -z "$EVAL_ROOT" ]]; then EVAL_ROOT="$ARTIFACT_ROOT/eval"; fi
if [[ -z "$CHECKPOINT_ROOT" ]]; then CHECKPOINT_ROOT="$ARTIFACT_ROOT/pretrain"; fi
if [[ -z "$OUTPUT_ROOT" ]]; then OUTPUT_ROOT="$ARTIFACT_ROOT/visualized"; fi
TRAIN_MANIFEST=$(absolute_path "$TRAIN_MANIFEST")
EVAL_ROOT=$(absolute_path "$EVAL_ROOT")
CHECKPOINT_ROOT=$(absolute_path "$CHECKPOINT_ROOT")
OUTPUT_ROOT=$(absolute_path "$OUTPUT_ROOT")
case "$OUTPUT_ROOT/" in "$EVAL_ROOT/"*|"$CHECKPOINT_ROOT/"*) printf '%s\n' 'Visualization output cannot be inside an input root.' >&2; exit 2 ;; esac
case "$EVAL_ROOT/" in "$OUTPUT_ROOT/"*) printf '%s\n' 'Evaluation input cannot be inside visualization output.' >&2; exit 2 ;; esac
case "$CHECKPOINT_ROOT/" in "$OUTPUT_ROOT/"*) printf '%s\n' 'Checkpoint input cannot be inside visualization output.' >&2; exit 2 ;; esac

PROBE_SEED_VALUES=()
while IFS= read -r seed; do [[ -n "$seed" ]] && PROBE_SEED_VALUES+=("$seed"); done \
  < <("$PYTHON_BIN" "$HELPER" seeds --values "$PROBE_SEEDS")
if ((${#PROBE_SEED_VALUES[@]} == 0)); then
  printf '%s\n' 'Probe seed list is empty.' >&2
  exit 2
fi
RUN_IDS=()
RUN_SEEDS=()
CMAX_WEIGHTS=()
if [[ "$STAGE" == dev ]]; then
  while IFS=$'\t' read -r run_id _condition seed cmax_weight _rest; do
    RUN_IDS+=("$run_id"); RUN_SEEDS+=("$seed"); CMAX_WEIGHTS+=("$cmax_weight")
  done < <("$PYTHON_BIN" "$HELPER" matrix --suite "$SUITE" --seeds "$SEEDS")
else
  while IFS=$'\t' read -r run_id _condition seed cmax_weight _rest; do
    RUN_IDS+=("$run_id"); RUN_SEEDS+=("$seed"); CMAX_WEIGHTS+=("$cmax_weight")
  done < <("$PYTHON_BIN" "$HELPER" lookup --run-id "$SELECTED_RUN_ID")
fi

is_cmax_task() { [[ "$1" == flow-cmax-init || "$1" == cmax-direct ]]; }
evaluation_dir() {
  local run_id=$1
  local probe_seed=$2
  local path="$EVAL_ROOT/$STAGE/$run_id/epoch$(printf '%04d' "$PRETRAIN_EPOCH")/$TASK/$ALIGNMENT"
  if [[ "$TASK" != depth ]]; then path="$path/$DT"; fi
  if [[ "$TASK" != cmax-direct ]]; then path="$path/probe_seed$probe_seed"; fi
  printf '%s\n' "$path"
}
checkpoint_for() {
  printf '%s/%s/checkpoint-epoch%04d.pt\n' "$CHECKPOINT_ROOT" "$1" "$PRETRAIN_EPOCH"
}

REPORT_LABELS=()
REPORT_PATHS=()
INDEX_RUN_IDS=()
INDEX_PATHS=()
if [[ "$MODE" == compare || "$MODE" == samples || "$MODE" == all ]]; then
  for ((run_index=0; run_index<${#RUN_IDS[@]}; run_index++)); do
    if is_cmax_task "$TASK" && [[ "${CMAX_WEIGHTS[$run_index]}" == 0 ]]; then continue; fi
    if [[ "$TASK" == cmax-direct ]]; then
      probe_values=("${RUN_SEEDS[$run_index]}")
    else
      probe_values=("${PROBE_SEED_VALUES[@]}")
    fi
    for probe_seed in "${probe_values[@]}"; do
      directory=$(evaluation_dir "${RUN_IDS[$run_index]}" "$probe_seed")
      if [[ "$TASK" == depth ]]; then report="$directory/metrics.json"; else report="$directory/report.json"; fi
      if [[ "$TASK" != cmax-direct ]]; then
        condition=${RUN_IDS[$run_index]#mvsec_}
        condition=${condition%__seed${RUN_SEEDS[$run_index]}}
        label="${condition}__encoder_seed${RUN_SEEDS[$run_index]}__probe_seed$probe_seed"
      else
        label=${RUN_IDS[$run_index]#mvsec_}
      fi
      REPORT_LABELS+=("$label")
      REPORT_PATHS+=("$report")
      INDEX_RUN_IDS+=("${RUN_IDS[$run_index]}__probe_seed$probe_seed")
      INDEX_PATHS+=("$directory/visualizations/index.json")
    done
  done
fi
if [[ "$MODE" == samples || "$MODE" == all ]]; then
  if ((${#INDEX_PATHS[@]} == 0)); then
    printf '%s\n' 'No applicable snapshot indexes were selected.' >&2
    exit 2
  fi
fi
if [[ "$MODE" == cmax-raw ]]; then
  cmax_run_count=0
  for ((index=0; index<${#CMAX_WEIGHTS[@]}; index++)); do
    if [[ "${CMAX_WEIGHTS[$index]}" != 0 ]]; then
      cmax_run_count=$((cmax_run_count + 1))
    fi
  done
  if ((cmax_run_count == 0)); then
    printf '%s\n' 'CMax raw-warp visualization requires a CMax checkpoint.' >&2
    exit 2
  fi
fi

if [[ "$STAGE" == dev ]]; then
  VISUALIZATION_SCOPE="suite_$SUITE"
else
  VISUALIZATION_SCOPE="selected_$SELECTED_RUN_ID"
fi
COMPARE_SET_ID=""
if [[ "$MODE" == compare || "$MODE" == all ]]; then
  SET_ID_COMMAND=("$PYTHON_BIN" "$HELPER" set-id)
  for ((index=0; index<${#REPORT_PATHS[@]}; index++)); do
    SET_ID_COMMAND+=(--item "${REPORT_LABELS[$index]}=${REPORT_PATHS[$index]}")
  done
  COMPARE_SET_ID=$("${SET_ID_COMMAND[@]}")
fi

COMPARE_OUTPUT="$OUTPUT_ROOT/compare/$STAGE/$VISUALIZATION_SCOPE/set_$COMPARE_SET_ID/epoch$(printf '%04d' "$PRETRAIN_EPOCH")/$TASK/$ALIGNMENT"
if [[ "$TASK" != depth ]]; then COMPARE_OUTPUT="$COMPARE_OUTPUT/$DT"; fi
if [[ "$TASK" != cmax-direct ]]; then
  if ((${#PROBE_SEED_VALUES[@]} == 1)); then
    COMPARE_OUTPUT="$COMPARE_OUTPUT/probe_seed${PROBE_SEED_VALUES[0]}"
  else
    COMPARE_OUTPUT="$COMPARE_OUTPUT/probe_seed_hierarchy"
  fi
fi

COMPARE_COMMAND=()
if [[ "$MODE" == compare || "$MODE" == all ]]; then
  if ((${#REPORT_PATHS[@]} < 2)); then
    printf '%s\n' 'Report comparison requires at least two matched reports.' >&2
    exit 2
  fi
  COMPARE_COMMAND=("$PYTHON_BIN" -m event_window_jepa.downstream.mvsec_visualize compare)
  for ((index=0; index<${#REPORT_PATHS[@]}; index++)); do
    COMPARE_COMMAND+=(--run "${REPORT_LABELS[$index]}=${REPORT_PATHS[$index]}")
  done
  COMPARE_COMMAND+=(--aggregate-seeds --output-dir "$COMPARE_OUTPUT")
  if ((ALLOW_INCOMPATIBLE)); then COMPARE_COMMAND+=(--allow-incompatible); fi
fi

print_command() { printf '  '; printf '%q ' "$@"; printf '\n'; }

printf 'MVSEC visualization: action=%s mode=%s stage=%s task=%s alignment=%s dt=%s\n' \
  "$ACTION" "$MODE" "$STAGE" "$TASK" "$ALIGNMENT" "$DT"
if [[ "$MODE" == compare || "$MODE" == all ]]; then
  printf '\n[matched report comparison: %s reports]\n' "${#REPORT_PATHS[@]}"
  print_command env "PYTHONPATH=$PROJECT_ROOT/src" "${COMPARE_COMMAND[@]}"
fi
if [[ "$MODE" == samples || "$MODE" == all ]]; then
  printf '\n[snapshot indexes]\n'
  for ((index=0; index<${#INDEX_PATHS[@]}; index++)); do
    printf '  %s -> %s (limit=%s)\n' \
      "${INDEX_RUN_IDS[$index]}" "${INDEX_PATHS[$index]}" "$SAMPLE_LIMIT"
  done
fi
if [[ "$MODE" == cmax-raw || "$MODE" == all ]]; then
  printf '\n[CMax raw-warp reports]\n'
  for ((index=0; index<${#RUN_IDS[@]}; index++)); do
    [[ "${CMAX_WEIGHTS[$index]}" != 0 ]] || continue
    output="$OUTPUT_ROOT/cmax-raw/$STAGE/$VISUALIZATION_SCOPE/${RUN_IDS[$index]}/epoch$(printf '%04d' "$PRETRAIN_EPOCH")/sample${CMAX_SAMPLE_INDEX}.html"
    print_command env "PYTHONPATH=$PROJECT_ROOT/src" "$PYTHON_BIN" -m \
      event_window_jepa.evaluation.cmax_flow_visualization \
      --checkpoint "$(checkpoint_for "${RUN_IDS[$index]}")" \
      --manifest "$TRAIN_MANIFEST" --output "$output" --device "$DEVICE" \
      --sample-index "$CMAX_SAMPLE_INDEX" \
      --calibration-samples "$CMAX_CALIBRATION_SAMPLES" \
      --flow-shuffle-seed "${RUN_SEEDS[$index]}" --quiver-stride "$QUIVER_STRIDE"
  done
fi
if [[ "$ACTION" == plan ]]; then exit 0; fi

estimated_jobs=0
if [[ "$MODE" == compare || "$MODE" == all ]]; then estimated_jobs=1; fi
if [[ "$MODE" == samples || "$MODE" == all ]]; then
  if ((SAMPLE_LIMIT == 0 && ALLOW_LARGE_MATRIX == 0)); then
    printf '%s\n' '--sample-limit 0 requires --allow-large-matrix for execution.' >&2
    exit 2
  fi
  estimated_jobs=$((estimated_jobs + ${#INDEX_PATHS[@]} * SAMPLE_LIMIT))
fi
if [[ "$MODE" == cmax-raw || "$MODE" == all ]]; then
  for weight in "${CMAX_WEIGHTS[@]}"; do
    if [[ "$weight" != 0 ]]; then estimated_jobs=$((estimated_jobs + 1)); fi
  done
fi
if ((estimated_jobs > 48 && ALLOW_LARGE_MATRIX == 0)); then
  printf 'Refusing about %s visualization jobs without --allow-large-matrix.\n' \
    "$estimated_jobs" >&2
  exit 2
fi

SAMPLE_SNAPSHOTS=()
SAMPLE_OUTPUTS=()
SAMPLE_EXPECTED_BYTES=()
SAMPLE_EXPECTED_SHA256=()
if [[ "$MODE" == samples || "$MODE" == all ]]; then
  for ((index=0; index<${#INDEX_PATHS[@]}; index++)); do
    [[ -f "${INDEX_PATHS[$index]}" ]] || { printf 'Snapshot index is missing: %s\n' "${INDEX_PATHS[$index]}" >&2; exit 1; }
    while IFS=$'\t' read -r ordinal snapshot _kind expected_bytes expected_sha256; do
      SAMPLE_SNAPSHOTS+=("$snapshot")
      SAMPLE_OUTPUTS+=("$OUTPUT_ROOT/samples/$STAGE/$VISUALIZATION_SCOPE/epoch$(printf '%04d' "$PRETRAIN_EPOCH")/$TASK/$ALIGNMENT/$DT/${INDEX_RUN_IDS[$index]}/sample$ordinal")
      SAMPLE_EXPECTED_BYTES+=("$expected_bytes")
      SAMPLE_EXPECTED_SHA256+=("$expected_sha256")
    done < <("$PYTHON_BIN" "$HELPER" snapshot-index \
      --path "${INDEX_PATHS[$index]}" --limit "$SAMPLE_LIMIT")
  done
fi

RAW_RUN_IDS=()
RAW_RUN_SEEDS=()
RAW_OUTPUTS=()
if [[ "$MODE" == cmax-raw || "$MODE" == all ]]; then
  [[ -f "$TRAIN_MANIFEST" ]] || { printf 'Train manifest is missing: %s\n' "$TRAIN_MANIFEST" >&2; exit 1; }
  "$PYTHON_BIN" "$HELPER" manifest --path "$TRAIN_MANIFEST" \
    --expected-recording outdoor_day2 --expected-split train \
    --expected-cameras left,right --require-artifacts >/dev/null
  for ((index=0; index<${#RUN_IDS[@]}; index++)); do
    [[ "${CMAX_WEIGHTS[$index]}" != 0 ]] || continue
    checkpoint=$(checkpoint_for "${RUN_IDS[$index]}")
    [[ -f "$checkpoint" ]] || { printf 'Checkpoint is missing: %s\n' "$checkpoint" >&2; exit 1; }
    RAW_RUN_IDS+=("${RUN_IDS[$index]}")
    RAW_RUN_SEEDS+=("${RUN_SEEDS[$index]}")
    RAW_OUTPUTS+=("$OUTPUT_ROOT/cmax-raw/$STAGE/$VISUALIZATION_SCOPE/${RUN_IDS[$index]}/epoch$(printf '%04d' "$PRETRAIN_EPOCH")/sample${CMAX_SAMPLE_INDEX}.html")
  done
fi

if [[ "$MODE" == compare || "$MODE" == all ]]; then
  for report in "${REPORT_PATHS[@]}"; do
    [[ -f "$report" ]] || { printf 'Evaluation report is missing: %s\n' "$report" >&2; exit 1; }
  done
  [[ ! -e "$COMPARE_OUTPUT" ]] || { printf 'Refusing existing comparison output: %s\n' "$COMPARE_OUTPUT" >&2; exit 1; }
fi
for ((index=0; index<${#SAMPLE_OUTPUTS[@]}; index++)); do
  output=${SAMPLE_OUTPUTS[$index]}
  [[ ! -e "$output" ]] || { printf 'Refusing existing sample render: %s\n' "$output" >&2; exit 1; }
done
for ((index=0; index<${#RAW_OUTPUTS[@]}; index++)); do
  output=${RAW_OUTPUTS[$index]}
  [[ ! -e "$output" && ! -e "${output%.html}.json" && ! -e "${output%.html}_assets" ]] || {
    printf 'Refusing existing CMax visualization output: %s\n' "$output" >&2; exit 1;
  }
done
job_count=${#SAMPLE_OUTPUTS[@]}
job_count=$((job_count + ${#RAW_OUTPUTS[@]}))
if [[ "$MODE" == compare || "$MODE" == all ]]; then job_count=$((job_count + 1)); fi
if ((job_count > 48 && ALLOW_LARGE_MATRIX == 0)); then
  printf 'Refusing %s visualization jobs without --allow-large-matrix.\n' "$job_count" >&2
  exit 2
fi

if [[ "$MODE" == compare || "$MODE" == all ]]; then
  PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "${COMPARE_COMMAND[@]}"
fi
for ((index=0; index<${#SAMPLE_SNAPSHOTS[@]}; index++)); do
  PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" -m event_window_jepa.downstream.mvsec_visualize sample \
    --snapshot "${SAMPLE_SNAPSHOTS[$index]}" \
    --output-dir "${SAMPLE_OUTPUTS[$index]}" \
    --maximum-snapshot-mib "$MAXIMUM_SNAPSHOT_MIB" \
    --expected-bytes "${SAMPLE_EXPECTED_BYTES[$index]}" \
    --expected-sha256 "${SAMPLE_EXPECTED_SHA256[$index]}"
done
for ((index=0; index<${#RAW_RUN_IDS[@]}; index++)); do
  mkdir -p "$(dirname "${RAW_OUTPUTS[$index]}")"
  PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" -m event_window_jepa.evaluation.cmax_flow_visualization \
    --checkpoint "$(checkpoint_for "${RAW_RUN_IDS[$index]}")" \
    --manifest "$TRAIN_MANIFEST" --output "${RAW_OUTPUTS[$index]}" \
    --device "$DEVICE" --sample-index "$CMAX_SAMPLE_INDEX" \
    --calibration-samples "$CMAX_CALIBRATION_SAMPLES" \
    --flow-shuffle-seed "${RAW_RUN_SEEDS[$index]}" --quiver-stride "$QUIVER_STRIDE"
done

printf 'Completed %s visualization jobs under %s.\n' "$job_count" "$OUTPUT_ROOT"
