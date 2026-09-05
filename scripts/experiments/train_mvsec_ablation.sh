#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
HELPER="$PROJECT_ROOT/scripts/experiments/mvsec_ablation_config.py"
JEPA_TEMPLATE="$PROJECT_ROOT/configs/pretrain/recurrent_future_convlstm_vits_mvsec.yaml"
CMAX_TEMPLATE="$PROJECT_ROOT/configs/pretrain/recurrent_future_convlstm_vits_mvsec_cmax.yaml"

ACTION=plan
SUITE=core
SEEDS=0,1,2
PYTHON_BIN=${PYTHON_BIN:-python3}
DATA_ROOT="$PROJECT_ROOT/data/mvsec/processed"
TRAIN_MANIFEST=""
OUTPUT_ROOT="$PROJECT_ROOT/outputs/experiments/mvsec_ablation"
NPROC_PER_NODE=1
PRECISION=fp16
EPOCHS=100
WARMUP_EPOCHS=10
SAMPLES_PER_EPOCH=6250
BATCH_SIZE=8
WORKERS=4
MILESTONE_EPOCHS=10,25,50,75,100
SMOKE=0
RESUME=0
SKIP_COMPLETE=0
ALLOW_LARGE_MATRIX=0

usage() {
  printf '%s\n' \
    'Usage: bash scripts/experiments/train_mvsec_ablation.sh [options]' \
    '' \
    '  --action plan|prepare|run  Default is plan; only run starts training' \
    '  --suite NAME              See named suites below' \
    '  --seeds CSV               Matched non-negative seeds (default: 0,1,2)' \
    '  --data-root DIR           Processed MVSEC bundle root' \
    '  --train-manifest FILE     Explicit day2 manifest override' \
    '  --output-root DIR         Config/run/log artifact root' \
    '  --python-bin PATH         Target Python executable' \
    '  --nproc-per-node N        DDP processes (default: 1)' \
    '  --precision MODE          fp16, bf16, or fp32' \
    '  --epochs N                Total pretrain epochs (default: 100)' \
    '  --warmup-epochs N         Warmup epochs (default: 10)' \
    '  --samples-per-epoch N     Global clips per epoch (default: 6250)' \
    '  --batch-size N            Per-rank batch size (default: 8)' \
    '  --workers N               DataLoader workers per rank (default: 4)' \
    '  --milestone-epochs CSV    Preserved checkpoints' \
    '  --smoke                   Distinct 1-epoch IDs; T=8 uses 2 global batches' \
    '  --resume                  Resume every selected run from checkpoint-latest.pt' \
    '  --skip-complete           Reuse only hash-verified completed runs' \
    '  --allow-large-matrix      Permit more than 12 sequential training jobs' \
    '  -h, --help                Show this help' \
    '' \
    'Named suites change one axis at a time:' \
    '  core      JEPA-only vs JEPA+CMax(0.05)' \
    '  temporal_sigreg  Temporal SIGReg weights 0, 0.01, 0.02, 0.05; not RA/LS' \
    '  cmax      CMax weights 0, 0.01, 0.05, 0.10; Temporal SIGReg off' \
    '  context   T=4,8,16; equal supervised frames but variable updates' \
    '  interaction  full JEPA/CMax x Temporal SIGReg(0/0.02) 2x2' \
    '  reference CMax reference mode past, future, both' \
    '  scales    CMax temporal scales [1], [1,2], [1,2,4]' \
    '  frame     Frame/support SIGReg 0 vs 0.02; unregularized is explicit' \
    '  rate_alignment  RA weights 0, 0.001, 0.01, 0.05; gamma=1' \
    '  rate_gamma  RA gamma 0.5, 1, 2 at weight=0.01' \
    '  straightening  LS weights 0, 0.001, 0.01, 0.05' \
    '  latent    full RA(0.01) x LS(0.01) 2x2' \
    '  latent_cmax  full selected RA+LS x CMax(0.05) 2x2' \
    '  all       union of the above (not a confounded Cartesian product)' \
    '' \
    'Safety: plan has no filesystem writes. Generated configs are immutable;' \
    'run preflights the complete matrix before starting its first job, and' \
    'existing outputs fail unless --resume or strict --skip-complete is explicit.'
}

need_value() {
  if (($# < 2)); then
    printf 'Missing value for %s\n' "$1" >&2
    exit 2
  fi
}

while (($#)); do
  case "$1" in
    --action) need_value "$@"; ACTION=$2; shift 2 ;;
    --suite) need_value "$@"; SUITE=$2; shift 2 ;;
    --seeds) need_value "$@"; SEEDS=$2; shift 2 ;;
    --data-root) need_value "$@"; DATA_ROOT=$2; shift 2 ;;
    --train-manifest) need_value "$@"; TRAIN_MANIFEST=$2; shift 2 ;;
    --output-root) need_value "$@"; OUTPUT_ROOT=$2; shift 2 ;;
    --python-bin) need_value "$@"; PYTHON_BIN=$2; shift 2 ;;
    --nproc-per-node) need_value "$@"; NPROC_PER_NODE=$2; shift 2 ;;
    --precision) need_value "$@"; PRECISION=$2; shift 2 ;;
    --epochs) need_value "$@"; EPOCHS=$2; shift 2 ;;
    --warmup-epochs) need_value "$@"; WARMUP_EPOCHS=$2; shift 2 ;;
    --samples-per-epoch) need_value "$@"; SAMPLES_PER_EPOCH=$2; shift 2 ;;
    --batch-size) need_value "$@"; BATCH_SIZE=$2; shift 2 ;;
    --workers) need_value "$@"; WORKERS=$2; shift 2 ;;
    --milestone-epochs) need_value "$@"; MILESTONE_EPOCHS=$2; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    --resume) RESUME=1; shift ;;
    --skip-complete) SKIP_COMPLETE=1; shift ;;
    --allow-large-matrix) ALLOW_LARGE_MATRIX=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$ACTION" in plan|prepare|run) ;; *) printf 'Invalid action: %s\n' "$ACTION" >&2; exit 2 ;; esac
case "$SUITE" in core|temporal_sigreg|cmax|context|interaction|reference|scales|frame|rate_alignment|rate_gamma|straightening|latent|latent_cmax|all) ;; *) printf 'Invalid suite: %s\n' "$SUITE" >&2; exit 2 ;; esac
case "$PRECISION" in fp16|bf16|fp32) ;; *) printf 'Invalid precision: %s\n' "$PRECISION" >&2; exit 2 ;; esac

for numeric in "$NPROC_PER_NODE" "$EPOCHS" "$SAMPLES_PER_EPOCH" "$BATCH_SIZE"; do
  case "$numeric" in ''|0|0[0-9]*|*[!0-9]*) printf 'Expected a positive integer, got: %s\n' "$numeric" >&2; exit 2 ;; esac
done
case "$WARMUP_EPOCHS" in ''|0[0-9]*|*[!0-9]*) printf 'Invalid warmup epochs: %s\n' "$WARMUP_EPOCHS" >&2; exit 2 ;; esac
case "$WORKERS" in ''|0[0-9]*|*[!0-9]*) printf 'Invalid workers: %s\n' "$WORKERS" >&2; exit 2 ;; esac
if ((WARMUP_EPOCHS >= EPOCHS)); then
  printf 'Warmup epochs must be less than total epochs.\n' >&2
  exit 2
fi
if ((SMOKE && RESUME)); then
  printf '%s\n' '--smoke cannot be combined with --resume.' >&2
  exit 2
fi
if ((SKIP_COMPLETE && RESUME)); then
  printf '%s\n' '--skip-complete cannot be combined with --resume.' >&2
  exit 2
fi
if [[ ! -f "$HELPER" || ! -f "$JEPA_TEMPLATE" || ! -f "$CMAX_TEMPLATE" ]]; then
  printf '%s\n' 'MVSEC helper or templates are missing from the repository.' >&2
  exit 1
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  printf 'Python executable is unavailable: %s\n' "$PYTHON_BIN" >&2
  exit 1
fi
PYTHON_BIN=$(command -v "$PYTHON_BIN")
PYTHON_BIN=$("$PYTHON_BIN" "$HELPER" path "$PYTHON_BIN")

absolute_path() {
  "$PYTHON_BIN" "$HELPER" path "$1"
}

DATA_ROOT=$(absolute_path "$DATA_ROOT")
OUTPUT_ROOT=$(absolute_path "$OUTPUT_ROOT")
if [[ -z "$TRAIN_MANIFEST" ]]; then
  TRAIN_MANIFEST="$DATA_ROOT/manifests/train.jsonl"
fi
TRAIN_MANIFEST=$(absolute_path "$TRAIN_MANIFEST")

# Never place generated artifacts in, or above, the processed data tree.
case "$OUTPUT_ROOT/" in "$DATA_ROOT/"*) printf 'Output root cannot be inside the data root.\n' >&2; exit 2 ;; esac
case "$DATA_ROOT/" in "$OUTPUT_ROOT/"*) printf 'Data root cannot be inside the output root.\n' >&2; exit 2 ;; esac

if ((SMOKE)); then
  EPOCHS=1
  WARMUP_EPOCHS=0
  SAMPLES_PER_EPOCH=$((2 * NPROC_PER_NODE * BATCH_SIZE))
  MILESTONE_EPOCHS=1
fi

IFS=',' read -r -a MILESTONES <<< "$MILESTONE_EPOCHS"
if ((${#MILESTONES[@]} == 0)); then
  printf '%s\n' '--milestone-epochs cannot be empty.' >&2
  exit 2
fi
previous_epoch=0
for epoch in "${MILESTONES[@]}"; do
  case "$epoch" in ''|0|0[0-9]*|*[!0-9]*) printf 'Invalid milestone epoch: %s\n' "$epoch" >&2; exit 2 ;; esac
  if ((epoch <= previous_epoch)); then
    printf '%s\n' 'Milestone epochs must be unique and strictly increasing.' >&2
    exit 2
  fi
  if ((epoch > EPOCHS)); then
    printf 'Milestone %s exceeds total epochs %s.\n' "$epoch" "$EPOCHS" >&2
    exit 2
  fi
  previous_epoch=$epoch
done
if ((previous_epoch != EPOCHS)); then
  printf 'The final epoch (%s) must be the last milestone checkpoint.\n' "$EPOCHS" >&2
  exit 2
fi

MATRIX_ARGS=(matrix --suite "$SUITE" --seeds "$SEEDS")
if ((SMOKE)); then MATRIX_ARGS+=(--smoke); fi
RUN_IDS=()
CONDITIONS=()
RUN_SEEDS=()
CMAX_WEIGHTS=()
TEMPORAL_WEIGHTS=()
FRAME_WEIGHTS=()
SEQUENCE_LENGTHS=()
CMAX_REFERENCE_MODES=()
CMAX_TEMPORAL_SCALES=()
RATE_ALIGNMENT_WEIGHTS=()
RATE_ALIGNMENT_GAMMAS=()
RATE_ALIGNMENT_EPSILONS=()
RATE_ALIGNMENT_NORMALIZATIONS=()
LATENT_STRAIGHTENING_WEIGHTS=()
LATENT_STRAIGHTENING_EPSILONS=()
while IFS=$'\t' read -r run_id condition seed cmax_weight temporal_weight frame_weight sequence_length reference_mode temporal_scales rate_weight rate_gamma rate_eps rate_normalization straightening_weight straightening_eps; do
  [[ -n "$run_id" ]] || continue
  RUN_IDS+=("$run_id")
  CONDITIONS+=("$condition")
  RUN_SEEDS+=("$seed")
  CMAX_WEIGHTS+=("$cmax_weight")
  TEMPORAL_WEIGHTS+=("$temporal_weight")
  FRAME_WEIGHTS+=("$frame_weight")
  SEQUENCE_LENGTHS+=("$sequence_length")
  CMAX_REFERENCE_MODES+=("$reference_mode")
  CMAX_TEMPORAL_SCALES+=("$temporal_scales")
  RATE_ALIGNMENT_WEIGHTS+=("$rate_weight")
  RATE_ALIGNMENT_GAMMAS+=("$rate_gamma")
  RATE_ALIGNMENT_EPSILONS+=("$rate_eps")
  RATE_ALIGNMENT_NORMALIZATIONS+=("$rate_normalization")
  LATENT_STRAIGHTENING_WEIGHTS+=("$straightening_weight")
  LATENT_STRAIGHTENING_EPSILONS+=("$straightening_eps")
done < <("$PYTHON_BIN" "$HELPER" "${MATRIX_ARGS[@]}")
if ((${#RUN_IDS[@]} == 0)); then
  printf '%s\n' 'The selected matrix is empty.' >&2
  exit 1
fi
if [[ "$ACTION" == run ]] && ((${#RUN_IDS[@]} > 12)) && ((ALLOW_LARGE_MATRIX == 0)); then
  printf 'Refusing to start %s training jobs without --allow-large-matrix.\n' \
    "${#RUN_IDS[@]}" >&2
  exit 2
fi

config_path() { printf '%s/configs/%s.yaml\n' "$OUTPUT_ROOT" "$1"; }
run_path() { printf '%s/pretrain/%s\n' "$OUTPUT_ROOT" "$1"; }
log_path() { printf '%s/logs/pretrain/%s.log\n' "$OUTPUT_ROOT" "$1"; }
completion_path() { printf '%s/pretrain/%s/.ablation-complete.json\n' "$OUTPUT_ROOT" "$1"; }

template_for() {
  if [[ "$1" == 0 ]]; then printf '%s\n' "$JEPA_TEMPLATE"; else printf '%s\n' "$CMAX_TEMPLATE"; fi
}

render_one() {
  local index=$1
  local command=("$PYTHON_BIN" "$HELPER" render \
    --template "$(template_for "${CMAX_WEIGHTS[$index]}")" \
    --output "$(config_path "${RUN_IDS[$index]}")" \
    --run-id "${RUN_IDS[$index]}" \
    --manifest "$TRAIN_MANIFEST" \
    --run-output "$(run_path "${RUN_IDS[$index]}")" \
    --seed "${RUN_SEEDS[$index]}" \
    --cmax-weight "${CMAX_WEIGHTS[$index]}" \
    --temporal-sigreg-weight "${TEMPORAL_WEIGHTS[$index]}" \
    --frame-sigreg-weight "${FRAME_WEIGHTS[$index]}" \
    --rate-alignment-weight "${RATE_ALIGNMENT_WEIGHTS[$index]}" \
    --rate-alignment-gamma "${RATE_ALIGNMENT_GAMMAS[$index]}" \
    --rate-alignment-eps "${RATE_ALIGNMENT_EPSILONS[$index]}" \
    --rate-alignment-normalization "${RATE_ALIGNMENT_NORMALIZATIONS[$index]}" \
    --latent-straightening-weight "${LATENT_STRAIGHTENING_WEIGHTS[$index]}" \
    --latent-straightening-eps "${LATENT_STRAIGHTENING_EPSILONS[$index]}" \
    --sequence-length "${SEQUENCE_LENGTHS[$index]}" \
    --cmax-reference-mode "${CMAX_REFERENCE_MODES[$index]}" \
    --cmax-temporal-scales "${CMAX_TEMPORAL_SCALES[$index]}" \
    --epochs "$EPOCHS" \
    --warmup-epochs "$WARMUP_EPOCHS" \
    --samples-per-epoch "$(samples_for_index "$index")" \
    --batch-size "$BATCH_SIZE" \
    --workers "$WORKERS" \
    --precision "$PRECISION")
  if [[ "$2" == check ]]; then command+=(--check-only); fi
  "${command[@]}"
}

samples_for_index() {
  local index=$1
  local numerator=$((SAMPLES_PER_EPOCH * 8))
  local length=${SEQUENCE_LENGTHS[$index]}
  if ((numerator % length != 0)); then
    printf 'Base samples (%s) cannot equalize supervised frames for T=%s.\n' \
      "$SAMPLES_PER_EPOCH" "$length" >&2
    exit 2
  fi
  printf '%s\n' "$((numerator / length))"
}

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

MILESTONE_ID=""
for epoch in "${MILESTONES[@]}"; do
  if [[ -n "$MILESTONE_ID" ]]; then MILESTONE_ID="$MILESTONE_ID,"; fi
  MILESTONE_ID="$MILESTONE_ID$epoch"
done

build_completion_command() {
  local action=$1
  local index=$2
  local run_id=${RUN_IDS[$index]}
  local output
  local final_checkpoint
  output=$(run_path "$run_id")
  final_checkpoint="$output/checkpoint-epoch$(printf '%04d' "$EPOCHS").pt"
  COMPLETION_COMMAND=("$PYTHON_BIN" "$HELPER" completion \
    --action "$action" --path "$(completion_path "$run_id")" --kind pretrain \
    --identity "run_id=$run_id" \
    --identity "seed=${RUN_SEEDS[$index]}" \
    --identity "final_epoch=$EPOCHS" \
    --identity "milestone_epochs=$MILESTONE_ID" \
    --identity "nproc_per_node=$NPROC_PER_NODE" \
    --identity "precision=$PRECISION" \
    --identity "exposure_policy=equal_supervised_frames_variable_updates" \
    --artifact "generated_config=$(config_path "$run_id")" \
    --artifact "resolved_config=$output/resolved_config.yaml" \
    --artifact "checkpoint=$final_checkpoint" \
    --artifact "metrics=$output/train.jsonl" \
    --artifact "log=$(log_path "$run_id")" \
    --artifact "train_manifest=$TRAIN_MANIFEST")
}

printf 'MVSEC pretrain ablation: action=%s suite=%s runs=%s\n' \
  "$ACTION" "$SUITE" "${#RUN_IDS[@]}"
printf '  train manifest: %s\n  output root:    %s\n' "$TRAIN_MANIFEST" "$OUTPUT_ROOT"
printf '%s\n' '  exposure policy: equal_supervised_frames_variable_updates'
if ((SKIP_COMPLETE)); then printf '%s\n' '  completed-run policy: strict hash-verified reuse'; fi

for ((index=0; index<${#RUN_IDS[@]}; index++)); do
  render_one "$index" check
done

if [[ "$ACTION" == plan ]]; then
  for ((index=0; index<${#RUN_IDS[@]}; index++)); do
    config=$(config_path "${RUN_IDS[$index]}")
    output=$(run_path "${RUN_IDS[$index]}")
    printf '\n[%s] condition=%s seed=%s CMax=%s TemporalSIGReg=%s FrameSupportSIGReg=%s RA=%s gamma=%s LS=%s T=%s clips=%s ref=%s scales=%s\n' \
      "${RUN_IDS[$index]}" "${CONDITIONS[$index]}" "${RUN_SEEDS[$index]}" \
      "${CMAX_WEIGHTS[$index]}" "${TEMPORAL_WEIGHTS[$index]}" \
      "${FRAME_WEIGHTS[$index]}" "${RATE_ALIGNMENT_WEIGHTS[$index]}" \
      "${RATE_ALIGNMENT_GAMMAS[$index]}" \
      "${LATENT_STRAIGHTENING_WEIGHTS[$index]}" \
      "${SEQUENCE_LENGTHS[$index]}" \
      "$(samples_for_index "$index")" \
      "${CMAX_REFERENCE_MODES[$index]}" "${CMAX_TEMPORAL_SCALES[$index]}"
    print_command "$PYTHON_BIN" "$HELPER" render --template \
      "$(template_for "${CMAX_WEIGHTS[$index]}")" --output "$config" \
      --run-id "${RUN_IDS[$index]}" --manifest "$TRAIN_MANIFEST" \
      --run-output "$output" --seed "${RUN_SEEDS[$index]}" \
      --cmax-weight "${CMAX_WEIGHTS[$index]}" \
      --temporal-sigreg-weight "${TEMPORAL_WEIGHTS[$index]}" \
      --frame-sigreg-weight "${FRAME_WEIGHTS[$index]}" \
      --rate-alignment-weight "${RATE_ALIGNMENT_WEIGHTS[$index]}" \
      --rate-alignment-gamma "${RATE_ALIGNMENT_GAMMAS[$index]}" \
      --rate-alignment-eps "${RATE_ALIGNMENT_EPSILONS[$index]}" \
      --rate-alignment-normalization "${RATE_ALIGNMENT_NORMALIZATIONS[$index]}" \
      --latent-straightening-weight "${LATENT_STRAIGHTENING_WEIGHTS[$index]}" \
      --latent-straightening-eps "${LATENT_STRAIGHTENING_EPSILONS[$index]}" \
      --sequence-length "${SEQUENCE_LENGTHS[$index]}" \
      --cmax-reference-mode "${CMAX_REFERENCE_MODES[$index]}" \
      --cmax-temporal-scales "${CMAX_TEMPORAL_SCALES[$index]}" \
      --epochs "$EPOCHS" \
      --warmup-epochs "$WARMUP_EPOCHS" \
      --samples-per-epoch "$(samples_for_index "$index")" \
      --batch-size "$BATCH_SIZE" --workers "$WORKERS" --precision "$PRECISION"
    command=("$PYTHON_BIN" -m torch.distributed.run --standalone \
      "--nproc-per-node=$NPROC_PER_NODE" -m event_window_jepa.train.pretrain \
      --config "$config" --milestone-epochs "${MILESTONES[@]}")
    if ((RESUME)); then command+=(--resume "$output/checkpoint-latest.pt"); fi
    print_command env "PYTHONPATH=$PROJECT_ROOT/src" PYTHONUNBUFFERED=1 "${command[@]}"
  done
  exit 0
fi

# Config compatibility is checked for the entire suite before creating any file.
for ((index=0; index<${#RUN_IDS[@]}; index++)); do
  render_one "$index" write
done
if [[ "$ACTION" == prepare ]]; then
  printf 'Prepared %s immutable configs under %s/configs\n' "${#RUN_IDS[@]}" "$OUTPUT_ROOT"
  exit 0
fi

[[ -f "$TRAIN_MANIFEST" ]] || { printf 'Train manifest is missing: %s\n' "$TRAIN_MANIFEST" >&2; exit 1; }
"$PYTHON_BIN" "$HELPER" manifest --path "$TRAIN_MANIFEST" \
  --expected-recording outdoor_day2 --expected-split train \
  --expected-cameras left,right --require-artifacts >/dev/null

# Preflight every destination before starting the first expensive job.
RUN_SKIPS=()
for ((index=0; index<${#RUN_IDS[@]}; index++)); do
  output=$(run_path "${RUN_IDS[$index]}")
  log=$(log_path "${RUN_IDS[$index]}")
  if ((RESUME)); then
    [[ -f "$output/checkpoint-latest.pt" ]] || { printf 'Resume checkpoint is missing: %s\n' "$output/checkpoint-latest.pt" >&2; exit 1; }
    [[ -f "$output/train.jsonl" ]] || { printf 'Resume metrics are missing: %s\n' "$output/train.jsonl" >&2; exit 1; }
    RUN_SKIPS+=(0)
  elif ((SKIP_COMPLETE)) && [[ -e "$output" || -e "$log" ]]; then
    [[ -d "$output" && -f "$log" ]] || {
      printf 'Existing pretrain job is partial: %s\n' "${RUN_IDS[$index]}" >&2
      exit 1
    }
    build_completion_command verify "$index"
    if ! "${COMPLETION_COMMAND[@]}" >/dev/null; then
      printf 'Existing pretrain job failed strict completion verification: %s\n' \
        "${RUN_IDS[$index]}" >&2
      exit 1
    fi
    RUN_SKIPS+=(1)
  else
    [[ ! -e "$output" ]] || { printf 'Refusing existing run output: %s\n' "$output" >&2; exit 1; }
    [[ ! -e "$log" ]] || { printf 'Refusing existing log: %s\n' "$log" >&2; exit 1; }
    RUN_SKIPS+=(0)
  fi
done

mkdir -p "$OUTPUT_ROOT/logs/pretrain"
for ((index=0; index<${#RUN_IDS[@]}; index++)); do
  config=$(config_path "${RUN_IDS[$index]}")
  output=$(run_path "${RUN_IDS[$index]}")
  log=$(log_path "${RUN_IDS[$index]}")
  if ((${RUN_SKIPS[$index]})); then
    printf '\nSkipping verified completed run %s\n' "${RUN_IDS[$index]}"
    continue
  fi
  command=("$PYTHON_BIN" -m torch.distributed.run --standalone \
    "--nproc-per-node=$NPROC_PER_NODE" -m event_window_jepa.train.pretrain \
    --config "$config" --milestone-epochs "${MILESTONES[@]}")
  if ((RESUME)); then command+=(--resume "$output/checkpoint-latest.pt"); fi
  printf '\nStarting %s\n' "${RUN_IDS[$index]}"
  print_command env "PYTHONPATH=$PROJECT_ROOT/src" PYTHONUNBUFFERED=1 "${command[@]}"
  if ((RESUME)); then printf '\n[resume %s]\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$log"; fi
  (
    cd "$PROJECT_ROOT"
    PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
      PYTHONUNBUFFERED=1 "${command[@]}"
  ) 2>&1 | tee -a "$log"
  build_completion_command record "$index"
  "${COMPLETION_COMMAND[@]}"
done

printf 'Completed all %s selected pretrain runs.\n' "${#RUN_IDS[@]}"
