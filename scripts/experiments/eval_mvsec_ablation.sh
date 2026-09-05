#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
HELPER="$PROJECT_ROOT/scripts/experiments/mvsec_ablation_config.py"

ACTION=plan
STAGE=dev
SUITE=core
SUITE_EXPLICIT=0
SEEDS=0,1,2
SEEDS_EXPLICIT=0
PROBE_SEEDS=0
SELECTED_RUN_ID=""
PRETRAIN_EPOCH=100
HISTORY_STEPS=10
TASKS=primary
PROTOCOL_SUITE=primary
DEV_FRACTION=0.2
DEV_GUARD_MS=""
PYTHON_BIN=${PYTHON_BIN:-python3}
DATA_ROOT="$PROJECT_ROOT/data/mvsec/processed"
TRAIN_MANIFEST=""
TEST_MANIFEST=""
OOD_MANIFEST=""
INCLUDE_OOD=0
ARTIFACT_ROOT="$PROJECT_ROOT/outputs/experiments/mvsec_ablation"
CHECKPOINT_ROOT=""
OUTPUT_ROOT=""
DEVICE=cuda
PRECISION=fp32
EPOCHS=30
BATCH_SIZE=4
WORKERS=4
LEARNING_RATE=0.001
WEIGHT_DECAY=0.0001
MAX_TRAIN_SAMPLES=0
MAX_EVAL_SAMPLES=0
SAVE_VISUALIZATIONS=0
VISUALIZATION_MAX_EVENTS=200000
SMOKE=0
SKIP_COMPLETE=0
ALLOW_LARGE_MATRIX=0

usage() {
  printf '%s\n' \
    'Usage: bash scripts/experiments/eval_mvsec_ablation.sh [options]' \
    '' \
    '  --action plan|run         Default is plan; only run starts evaluation' \
    '  --stage dev|final         day2 temporal dev (default) or sealed day1 final' \
    '  --suite NAME              Pretrain suite; dev stage only' \
    '  --seeds CSV               Pretrain checkpoint seeds; dev stage only' \
    '  --probe-seeds CSV         Independent frozen-head seeds (default: 0)' \
    '  --selected-run-id ID      Required single preselected run for final stage' \
    '  --pretrain-epoch N        Immutable milestone checkpoint (default: 100)' \
    '  --history-steps N         Fixed downstream history for every run (default: 10)' \
    '  --tasks SET               primary, all, or a comma-separated task list' \
    '  --protocol-suite SET      primary, rate, alignment, or all' \
    '  --dev-fraction FLOAT      Tail fraction of day2 reserved for dev' \
    '  --dev-guard-ms FLOAT      Optional explicit temporal guard before dev' \
    '  --data-root DIR           Processed MVSEC bundle root' \
    '  --train-manifest FILE     Explicit outdoor_day2 manifest' \
    '  --test-manifest FILE      Explicit outdoor_day1 final-test manifest' \
    '  --include-ood             Add outdoor_night1 to depth final reporting' \
    '  --ood-manifest FILE       Explicit night1 manifest; implies --include-ood' \
    '  --artifact-root DIR       Parent produced by the train runner' \
    '  --checkpoint-root DIR     Run directories containing checkpoints' \
    '  --output-root DIR         New evaluation artifact root' \
    '  --python-bin PATH         Target Python executable' \
    '  --device DEVICE           Default: cuda' \
    '  --precision fp32|bf16     Probe/evaluation precision (default: fp32)' \
    '  --epochs N                Frozen-head training epochs (default: 30)' \
    '  --batch-size N            Probe batch size (default: 4)' \
    '  --workers N               DataLoader workers (default: 4)' \
    '  --learning-rate FLOAT     Probe learning rate (default: 0.001)' \
    '  --weight-decay FLOAT      Probe weight decay (default: 0.0001)' \
    '  --max-train-samples N     0 means all available targets' \
    '  --max-eval-samples N      0 means all final-test targets' \
    '  --save-visualizations N   Deterministic snapshots per job (default: 0)' \
    '  --visualization-max-events N  Raw-event cap per snapshot' \
    '  --smoke                   Use smoke checkpoints, 1 epoch, max 8 targets' \
    '  --skip-complete           Reuse only hash-verified completed eval jobs' \
    '  --allow-large-matrix      Permit more than 24 sequential eval jobs' \
    '  -h, --help                Show this help' \
    '' \
    'Task sets:' \
    '  primary = flow-random,cmax-direct,depth' \
    '  all     = flow-random,flow-cmax-init,cmax-direct,depth' \
    'The random flow head is the encoder-quality comparison. CMax-init and' \
    'direct CMax are diagnostics and are generated only for CMax checkpoints.' \
    '' \
    'Pretrain suites include core, Temporal SIGReg, CMax, context, RA/LS,' \
    'their named 2x2 interactions, protocol controls, and all.' \
    '' \
    'Protocol suites:' \
    '  primary    causal + native flow (and causal depth)' \
    '  rate       causal/native vs causal/dt1' \
    '  alignment  causal/native vs f3_centered/native' \
    '  all        causal/f3_centered x native/dt1 exploratory grid' \
    'dt1 remains the repository diagnostic, not the exact 800-frame protocol.' \
    '' \
    'Probe seeds are separate from encoder seeds. The default dev stage fits' \
    'and scores only disjoint temporal blocks of outdoor_day2. outdoor_day1 is' \
    'opened only by --stage final with one explicitly selected run ID.' \
    'Downstream history is fixed across context-length ablations. With' \
    '--skip-complete, missing markers, partial outputs, or identity/hash' \
    'differences fail before any new evaluation job starts.'
}

need_value() {
  if (($# < 2)); then printf 'Missing value for %s\n' "$1" >&2; exit 2; fi
}

while (($#)); do
  case "$1" in
    --action) need_value "$@"; ACTION=$2; shift 2 ;;
    --stage) need_value "$@"; STAGE=$2; shift 2 ;;
    --suite) need_value "$@"; SUITE=$2; SUITE_EXPLICIT=1; shift 2 ;;
    --seeds) need_value "$@"; SEEDS=$2; SEEDS_EXPLICIT=1; shift 2 ;;
    --probe-seeds) need_value "$@"; PROBE_SEEDS=$2; shift 2 ;;
    --selected-run-id) need_value "$@"; SELECTED_RUN_ID=$2; shift 2 ;;
    --pretrain-epoch) need_value "$@"; PRETRAIN_EPOCH=$2; shift 2 ;;
    --history-steps) need_value "$@"; HISTORY_STEPS=$2; shift 2 ;;
    --tasks) need_value "$@"; TASKS=$2; shift 2 ;;
    --protocol-suite) need_value "$@"; PROTOCOL_SUITE=$2; shift 2 ;;
    --dev-fraction) need_value "$@"; DEV_FRACTION=$2; shift 2 ;;
    --dev-guard-ms) need_value "$@"; DEV_GUARD_MS=$2; shift 2 ;;
    --data-root) need_value "$@"; DATA_ROOT=$2; shift 2 ;;
    --train-manifest) need_value "$@"; TRAIN_MANIFEST=$2; shift 2 ;;
    --test-manifest) need_value "$@"; TEST_MANIFEST=$2; shift 2 ;;
    --include-ood) INCLUDE_OOD=1; shift ;;
    --ood-manifest) need_value "$@"; OOD_MANIFEST=$2; INCLUDE_OOD=1; shift 2 ;;
    --artifact-root) need_value "$@"; ARTIFACT_ROOT=$2; shift 2 ;;
    --checkpoint-root) need_value "$@"; CHECKPOINT_ROOT=$2; shift 2 ;;
    --output-root) need_value "$@"; OUTPUT_ROOT=$2; shift 2 ;;
    --python-bin) need_value "$@"; PYTHON_BIN=$2; shift 2 ;;
    --device) need_value "$@"; DEVICE=$2; shift 2 ;;
    --precision) need_value "$@"; PRECISION=$2; shift 2 ;;
    --epochs) need_value "$@"; EPOCHS=$2; shift 2 ;;
    --batch-size) need_value "$@"; BATCH_SIZE=$2; shift 2 ;;
    --workers) need_value "$@"; WORKERS=$2; shift 2 ;;
    --learning-rate) need_value "$@"; LEARNING_RATE=$2; shift 2 ;;
    --weight-decay) need_value "$@"; WEIGHT_DECAY=$2; shift 2 ;;
    --max-train-samples) need_value "$@"; MAX_TRAIN_SAMPLES=$2; shift 2 ;;
    --max-eval-samples) need_value "$@"; MAX_EVAL_SAMPLES=$2; shift 2 ;;
    --save-visualizations) need_value "$@"; SAVE_VISUALIZATIONS=$2; shift 2 ;;
    --visualization-max-events) need_value "$@"; VISUALIZATION_MAX_EVENTS=$2; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    --skip-complete) SKIP_COMPLETE=1; shift ;;
    --allow-large-matrix) ALLOW_LARGE_MATRIX=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$ACTION" in plan|run) ;; *) printf 'Invalid action: %s\n' "$ACTION" >&2; exit 2 ;; esac
case "$STAGE" in dev|final) ;; *) printf 'Invalid stage: %s\n' "$STAGE" >&2; exit 2 ;; esac
case "$SUITE" in core|temporal_sigreg|cmax|context|interaction|reference|scales|frame|rate_alignment|rate_gamma|straightening|latent|latent_cmax|all) ;; *) printf 'Invalid suite: %s\n' "$SUITE" >&2; exit 2 ;; esac
case "$PROTOCOL_SUITE" in primary|rate|alignment|all) ;; *) printf 'Invalid protocol suite: %s\n' "$PROTOCOL_SUITE" >&2; exit 2 ;; esac
case "$PRECISION" in fp32|bf16) ;; *) printf 'Evaluation precision must be fp32 or bf16.\n' >&2; exit 2 ;; esac
for numeric in "$EPOCHS" "$BATCH_SIZE" "$PRETRAIN_EPOCH" "$HISTORY_STEPS"; do
  case "$numeric" in ''|0|0[0-9]*|*[!0-9]*) printf 'Expected a positive integer, got: %s\n' "$numeric" >&2; exit 2 ;; esac
done
for numeric in "$WORKERS" "$MAX_TRAIN_SAMPLES" "$MAX_EVAL_SAMPLES" "$SAVE_VISUALIZATIONS"; do
  case "$numeric" in ''|0[0-9]*|*[!0-9]*) printf 'Expected a non-negative integer, got: %s\n' "$numeric" >&2; exit 2 ;; esac
done
case "$VISUALIZATION_MAX_EVENTS" in ''|*[!0-9]*|0) printf 'Invalid visualization event cap: %s\n' "$VISUALIZATION_MAX_EVENTS" >&2; exit 2 ;; esac
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
if ! LEARNING_RATE=$("$PYTHON_BIN" "$HELPER" number "$LEARNING_RATE"); then
  printf '%s\n' 'Learning rate must be a finite positive decimal.' >&2
  exit 2
fi
if ! WEIGHT_DECAY=$("$PYTHON_BIN" "$HELPER" number "$WEIGHT_DECAY" --allow-zero); then
  printf '%s\n' 'Weight decay must be a finite non-negative decimal.' >&2
  exit 2
fi
if ! DEV_FRACTION=$("$PYTHON_BIN" "$HELPER" fraction "$DEV_FRACTION"); then
  printf '%s\n' 'Dev fraction must be strictly between zero and one.' >&2
  exit 2
fi
if [[ -n "$DEV_GUARD_MS" ]]; then
  if ! DEV_GUARD_MS=$("$PYTHON_BIN" "$HELPER" number "$DEV_GUARD_MS"); then
    printf '%s\n' 'Dev guard must be a finite positive duration.' >&2
    exit 2
  fi
fi
if [[ "$STAGE" == final ]]; then
  [[ -n "$SELECTED_RUN_ID" ]] || { printf '%s\n' '--selected-run-id is required for final stage.' >&2; exit 2; }
  if ((SUITE_EXPLICIT || SEEDS_EXPLICIT)); then
    printf '%s\n' 'Final stage accepts one --selected-run-id, not --suite/--seeds.' >&2
    exit 2
  fi
else
  [[ -z "$SELECTED_RUN_ID" ]] || { printf '%s\n' '--selected-run-id is final-stage only.' >&2; exit 2; }
  if ((INCLUDE_OOD)); then
    printf '%s\n' '--include-ood is final-stage only.' >&2
    exit 2
  fi
fi

absolute_path() { "$PYTHON_BIN" "$HELPER" path "$1"; }
DATA_ROOT=$(absolute_path "$DATA_ROOT")
ARTIFACT_ROOT=$(absolute_path "$ARTIFACT_ROOT")
if [[ -z "$CHECKPOINT_ROOT" ]]; then CHECKPOINT_ROOT="$ARTIFACT_ROOT/pretrain"; fi
if [[ -z "$OUTPUT_ROOT" ]]; then OUTPUT_ROOT="$ARTIFACT_ROOT/eval"; fi
CHECKPOINT_ROOT=$(absolute_path "$CHECKPOINT_ROOT")
OUTPUT_ROOT=$(absolute_path "$OUTPUT_ROOT")
if [[ -z "$TRAIN_MANIFEST" ]]; then TRAIN_MANIFEST="$DATA_ROOT/manifests/train.jsonl"; fi
if [[ -z "$TEST_MANIFEST" ]]; then TEST_MANIFEST="$DATA_ROOT/manifests/test.jsonl"; fi
if [[ -z "$OOD_MANIFEST" ]]; then OOD_MANIFEST="$DATA_ROOT/manifests/ood_test.jsonl"; fi
TRAIN_MANIFEST=$(absolute_path "$TRAIN_MANIFEST")
TEST_MANIFEST=$(absolute_path "$TEST_MANIFEST")
OOD_MANIFEST=$(absolute_path "$OOD_MANIFEST")

case "$OUTPUT_ROOT/" in "$DATA_ROOT/"*) printf 'Output root cannot be inside the data root.\n' >&2; exit 2 ;; esac
case "$DATA_ROOT/" in "$OUTPUT_ROOT/"*) printf 'Data root cannot be inside the output root.\n' >&2; exit 2 ;; esac
case "$OUTPUT_ROOT/" in "$CHECKPOINT_ROOT/"*) printf 'Evaluation output cannot be inside the checkpoint root.\n' >&2; exit 2 ;; esac
case "$CHECKPOINT_ROOT/" in "$OUTPUT_ROOT/"*) printf 'Checkpoint root cannot be inside evaluation output.\n' >&2; exit 2 ;; esac

if ((SMOKE)); then
  PRETRAIN_EPOCH=1
  EPOCHS=1
  if ((MAX_TRAIN_SAMPLES == 0)); then MAX_TRAIN_SAMPLES=8; fi
  if ((MAX_EVAL_SAMPLES == 0)); then MAX_EVAL_SAMPLES=8; fi
fi

case "$TASKS" in
  primary) TASK_LIST=(flow-random cmax-direct depth) ;;
  all) TASK_LIST=(flow-random flow-cmax-init cmax-direct depth) ;;
  *)
    IFS=',' read -r -a TASK_LIST <<< "$TASKS"
    ((${#TASK_LIST[@]} > 0)) || { printf 'Task list is empty.\n' >&2; exit 2; }
    ;;
esac
SEEN_TASKS=' '
for task in "${TASK_LIST[@]}"; do
  case "$task" in flow-random|flow-cmax-init|cmax-direct|depth) ;;
    *) printf 'Invalid task: %s\n' "$task" >&2; exit 2 ;;
  esac
  case "$SEEN_TASKS" in *" $task "*) printf 'Duplicate task: %s\n' "$task" >&2; exit 2 ;; esac
  SEEN_TASKS="$SEEN_TASKS$task "
done

FLOW_ALIGNMENTS=()
FLOW_DTS=()
DEPTH_ALIGNMENTS=()
add_flow_protocol() { FLOW_ALIGNMENTS+=("$1"); FLOW_DTS+=("$2"); }
case "$PROTOCOL_SUITE" in
  primary) add_flow_protocol causal native; DEPTH_ALIGNMENTS=(causal) ;;
  rate) add_flow_protocol causal native; add_flow_protocol causal dt1; DEPTH_ALIGNMENTS=(causal) ;;
  alignment) add_flow_protocol causal native; add_flow_protocol f3_centered native; DEPTH_ALIGNMENTS=(causal f3_centered) ;;
  all)
    add_flow_protocol causal native
    add_flow_protocol causal dt1
    add_flow_protocol f3_centered native
    add_flow_protocol f3_centered dt1
    DEPTH_ALIGNMENTS=(causal f3_centered)
    ;;
esac

PROBE_SEED_VALUES=()
while IFS= read -r seed; do
  [[ -n "$seed" ]] && PROBE_SEED_VALUES+=("$seed")
done < <("$PYTHON_BIN" "$HELPER" seeds --values "$PROBE_SEEDS")
if ((${#PROBE_SEED_VALUES[@]} == 0)); then
  printf '%s\n' 'Probe seed matrix is empty.' >&2
  exit 2
fi

RUN_IDS=()
RUN_SEEDS=()
CMAX_WEIGHTS=()
if [[ "$STAGE" == dev ]]; then
  MATRIX_ARGS=(matrix --suite "$SUITE" --seeds "$SEEDS")
  if ((SMOKE)); then MATRIX_ARGS+=(--smoke); fi
  while IFS=$'\t' read -r run_id _condition seed cmax_weight _rest; do
    [[ -n "$run_id" ]] || continue
    RUN_IDS+=("$run_id")
    RUN_SEEDS+=("$seed")
    CMAX_WEIGHTS+=("$cmax_weight")
  done < <("$PYTHON_BIN" "$HELPER" "${MATRIX_ARGS[@]}")
else
  case "$SELECTED_RUN_ID" in
    *__smoke) ((SMOKE)) || { printf '%s\n' 'A smoke run ID requires --smoke.' >&2; exit 2; } ;;
    *) ((SMOKE == 0)) || { printf '%s\n' '--smoke requires a smoke run ID.' >&2; exit 2; } ;;
  esac
  while IFS=$'\t' read -r run_id _condition seed cmax_weight _rest; do
    RUN_IDS+=("$run_id")
    RUN_SEEDS+=("$seed")
    CMAX_WEIGHTS+=("$cmax_weight")
  done < <("$PYTHON_BIN" "$HELPER" lookup --run-id "$SELECTED_RUN_ID")
fi

JOB_RUN_IDS=()
JOB_KINDS=()
JOB_ALIGNMENTS=()
JOB_DTS=()
JOB_PROBE_SEEDS=()
JOB_OUTPUTS=()
JOB_LOGS=()

add_job() {
  local run_id=$1
  local kind=$2
  local alignment=$3
  local dt=$4
  local probe_seed=$5
  local leaf="$kind/$alignment"
  local log_leaf="${run_id}__${kind}__${alignment}"
  if [[ "$kind" != depth ]]; then
    leaf="$leaf/$dt"
    log_leaf="${log_leaf}__${dt}"
  fi
  if [[ "$kind" != cmax-direct ]]; then
    leaf="$leaf/probe_seed$probe_seed"
    log_leaf="${log_leaf}__probe_seed$probe_seed"
  fi
  JOB_RUN_IDS+=("$run_id")
  JOB_KINDS+=("$kind")
  JOB_ALIGNMENTS+=("$alignment")
  JOB_DTS+=("$dt")
  JOB_PROBE_SEEDS+=("$probe_seed")
  JOB_OUTPUTS+=("$OUTPUT_ROOT/$STAGE/$run_id/epoch$(printf '%04d' "$PRETRAIN_EPOCH")/$leaf")
  JOB_LOGS+=("$OUTPUT_ROOT/logs/$STAGE/epoch$(printf '%04d' "$PRETRAIN_EPOCH")/$log_leaf.log")
}

for ((run_index=0; run_index<${#RUN_IDS[@]}; run_index++)); do
  for task in "${TASK_LIST[@]}"; do
    if [[ "$task" == cmax-direct || "$task" == flow-cmax-init ]]; then
      [[ "${CMAX_WEIGHTS[$run_index]}" != 0 ]] || continue
    fi
    if [[ "$task" == depth ]]; then
      for alignment in "${DEPTH_ALIGNMENTS[@]}"; do
        for probe_seed in "${PROBE_SEED_VALUES[@]}"; do
          add_job "${RUN_IDS[$run_index]}" depth "$alignment" none "$probe_seed"
        done
      done
    elif [[ "$task" == cmax-direct ]]; then
      for ((protocol_index=0; protocol_index<${#FLOW_ALIGNMENTS[@]}; protocol_index++)); do
        add_job "${RUN_IDS[$run_index]}" "$task" \
          "${FLOW_ALIGNMENTS[$protocol_index]}" "${FLOW_DTS[$protocol_index]}" \
          "${RUN_SEEDS[$run_index]}"
      done
    else
      for ((protocol_index=0; protocol_index<${#FLOW_ALIGNMENTS[@]}; protocol_index++)); do
        for probe_seed in "${PROBE_SEED_VALUES[@]}"; do
          add_job "${RUN_IDS[$run_index]}" "$task" \
            "${FLOW_ALIGNMENTS[$protocol_index]}" "${FLOW_DTS[$protocol_index]}" \
            "$probe_seed"
        done
      done
    fi
  done
done
if ((${#JOB_RUN_IDS[@]} == 0)); then
  printf '%s\n' 'No applicable evaluation jobs were selected.' >&2
  exit 2
fi
if [[ "$ACTION" == run ]] && ((${#JOB_RUN_IDS[@]} > 24)) && ((ALLOW_LARGE_MATRIX == 0)); then
  printf 'Refusing to start %s evaluation jobs without --allow-large-matrix.\n' \
    "${#JOB_RUN_IDS[@]}" >&2
  exit 2
fi

checkpoint_for() {
  printf '%s/%s/checkpoint-epoch%04d.pt\n' \
    "$CHECKPOINT_ROOT" "$1" "$PRETRAIN_EPOCH"
}

completion_marker_for() {
  printf '%s/.ablation-complete.json\n' "${JOB_OUTPUTS[$1]}"
}

report_for() {
  if [[ "${JOB_KINDS[$1]}" == depth ]]; then
    printf '%s/metrics.json\n' "${JOB_OUTPUTS[$1]}"
  else
    printf '%s/report.json\n' "${JOB_OUTPUTS[$1]}"
  fi
}

build_command() {
  local index=$1
  local run_id=${JOB_RUN_IDS[$index]}
  local kind=${JOB_KINDS[$index]}
  local alignment=${JOB_ALIGNMENTS[$index]}
  local dt=${JOB_DTS[$index]}
  local output=${JOB_OUTPUTS[$index]}
  local checkpoint
  local seed=${JOB_PROBE_SEEDS[$index]}
  local eval_manifest=$TEST_MANIFEST
  local eval_split=test
  if [[ "$STAGE" == dev ]]; then eval_manifest=$TRAIN_MANIFEST; fi
  if [[ "$STAGE" == dev ]]; then eval_split=train; fi
  checkpoint=$(checkpoint_for "$run_id")
  COMMAND=()
  if [[ "$kind" == flow-random || "$kind" == flow-cmax-init ]]; then
    local head_init=random
    [[ "$kind" == flow-cmax-init ]] && head_init=cmax
    COMMAND=("$PYTHON_BIN" -m event_window_jepa.downstream.mvsec_flow probe \
      --checkpoint "$checkpoint" --train-manifest "$TRAIN_MANIFEST" \
      --eval-manifest "$eval_manifest" --output-dir "$output" \
      --eval-split "$eval_split" \
      --head-init "$head_init" --alignment "$alignment" --dt "$dt" \
      --history-steps "$HISTORY_STEPS" \
      --protocol-stage "$STAGE" --dev-fraction "$DEV_FRACTION" \
      --epochs "$EPOCHS" --learning-rate "$LEARNING_RATE" \
      --weight-decay "$WEIGHT_DECAY" --batch-size "$BATCH_SIZE" \
      --workers "$WORKERS" --device "$DEVICE" --precision "$PRECISION" \
      --seed "$seed" --max-train-samples "$MAX_TRAIN_SAMPLES" \
      --max-eval-samples "$MAX_EVAL_SAMPLES")
  elif [[ "$kind" == cmax-direct ]]; then
    COMMAND=("$PYTHON_BIN" -m event_window_jepa.downstream.mvsec_flow cmax-eval \
      --checkpoint "$checkpoint" --eval-manifest "$eval_manifest" \
      --eval-split "$eval_split" --output-dir "$output" \
      --alignment "$alignment" --dt "$dt" \
      --history-steps "$HISTORY_STEPS" \
      --protocol-stage "$STAGE" --dev-fraction "$DEV_FRACTION" \
      --batch-size "$BATCH_SIZE" --workers "$WORKERS" --device "$DEVICE" \
      --precision "$PRECISION" --seed "$seed" \
      --max-eval-samples "$MAX_EVAL_SAMPLES")
  else
    COMMAND=("$PYTHON_BIN" -m event_window_jepa.downstream.mvsec_depth \
      --checkpoint "$checkpoint" --train-manifest "$TRAIN_MANIFEST" \
      --eval-manifest "$eval_manifest")
    if ((INCLUDE_OOD)); then COMMAND+=("$OOD_MANIFEST"); fi
    COMMAND+=(--output-dir "$output" --alignment "$alignment" \
      --history-steps "$HISTORY_STEPS" \
      --protocol-stage "$STAGE" --dev-fraction "$DEV_FRACTION" \
      --epochs "$EPOCHS" \
      --learning-rate "$LEARNING_RATE" --weight-decay "$WEIGHT_DECAY" \
      --batch-size "$BATCH_SIZE" --workers "$WORKERS" --device "$DEVICE" \
      --precision "$PRECISION" --seed "$seed" \
      --max-train-targets "$MAX_TRAIN_SAMPLES" \
      --max-eval-targets "$MAX_EVAL_SAMPLES")
  fi
  if [[ -n "$DEV_GUARD_MS" ]]; then COMMAND+=(--dev-guard-ms "$DEV_GUARD_MS"); fi
  if ((SAVE_VISUALIZATIONS > 0)); then
    COMMAND+=(--save-visualizations "$SAVE_VISUALIZATIONS" \
      --visualization-dir "$output/visualizations" \
      --visualization-max-events "$VISUALIZATION_MAX_EVENTS")
  fi
}

build_completion_command() {
  local action=$1
  local index=$2
  local run_id=${JOB_RUN_IDS[$index]}
  local kind=${JOB_KINDS[$index]}
  local alignment=${JOB_ALIGNMENTS[$index]}
  local dt=${JOB_DTS[$index]}
  local seed=${JOB_PROBE_SEEDS[$index]}
  local output=${JOB_OUTPUTS[$index]}
  local checkpoint
  local eval_manifest=$TEST_MANIFEST
  local eval_split=test
  local guard_identity=auto
  checkpoint=$(checkpoint_for "$run_id")
  if [[ "$STAGE" == dev ]]; then
    eval_manifest=$TRAIN_MANIFEST
    eval_split=train
  fi
  if [[ -n "$DEV_GUARD_MS" ]]; then guard_identity=$DEV_GUARD_MS; fi
  COMPLETION_COMMAND=("$PYTHON_BIN" "$HELPER" completion \
    --action "$action" --path "$(completion_marker_for "$index")" \
    --kind evaluation \
    --identity "run_id=$run_id" \
    --identity "task=$kind" \
    --identity "stage=$STAGE" \
    --identity "pretrain_epoch=$PRETRAIN_EPOCH" \
    --identity "alignment=$alignment" \
    --identity "dt=$dt" \
    --identity "probe_seed=$seed" \
    --identity "history_steps=$HISTORY_STEPS" \
    --identity "dev_fraction=$DEV_FRACTION" \
    --identity "dev_guard_ms=$guard_identity" \
    --identity "batch_size=$BATCH_SIZE" \
    --identity "workers=$WORKERS" \
    --identity "device=$DEVICE" \
    --identity "precision=$PRECISION" \
    --identity "max_eval_samples=$MAX_EVAL_SAMPLES" \
    --identity "save_visualizations=$SAVE_VISUALIZATIONS" \
    --identity "visualization_max_events=$VISUALIZATION_MAX_EVENTS" \
    --identity "eval_split=$eval_split" \
    --artifact "checkpoint=$checkpoint" \
    --artifact "train_manifest=$TRAIN_MANIFEST" \
    --artifact "eval_manifest=$eval_manifest" \
    --artifact "report=$(report_for "$index")" \
    --artifact "log=${JOB_LOGS[$index]}")
  if ((INCLUDE_OOD)) && [[ "$kind" == depth ]]; then
    COMPLETION_COMMAND+=(--artifact "ood_manifest=$OOD_MANIFEST")
  fi
  if ((SAVE_VISUALIZATIONS > 0)); then
    COMPLETION_COMMAND+=(--artifact \
      "visualization_index=$output/visualizations/index.json")
  fi
  if [[ "$kind" == flow-random || "$kind" == flow-cmax-init ]]; then
    local head_init=random
    [[ "$kind" == flow-cmax-init ]] && head_init=cmax
    COMPLETION_COMMAND+=(--identity "head_initialization=$head_init" \
      --identity "epochs=$EPOCHS" \
      --identity "learning_rate=$LEARNING_RATE" \
      --identity "weight_decay=$WEIGHT_DECAY" \
      --identity "max_train_samples=$MAX_TRAIN_SAMPLES" \
      --artifact "head=$output/flow_head.pt" \
      --report-field "command=probe" \
      --report-field "checkpoint.path=$checkpoint" \
      --report-field "checkpoint.checkpoint_sha256=@sha256:checkpoint" \
      --report-field "protocol.stage=$STAGE" \
      --report-field "protocol.alignment.mode=$alignment" \
      --report-field "protocol.flow_rate.cli_value=$dt" \
      --report-field "protocol.event_history.history_steps=$HISTORY_STEPS" \
      --report-field "head.initialization=$head_init" \
      --report-field "head.checkpoint=$output/flow_head.pt" \
      --report-field "head.checkpoint_sha256=@sha256:head" \
      --report-field "training.manifest=$TRAIN_MANIFEST" \
      --report-field "training.manifest_artifact.sha256=@sha256:train_manifest" \
      --report-field "training.epochs=$EPOCHS" \
      --report-field "training.batch_size=$BATCH_SIZE" \
      --report-field "training.learning_rate=$LEARNING_RATE" \
      --report-field "training.weight_decay=$WEIGHT_DECAY" \
      --report-field "training.precision=$PRECISION" \
      --report-field "training.probe_seed=$seed" \
      --report-field "evaluation.manifest=$eval_manifest" \
      --report-field "evaluation.manifest_artifact.sha256=@sha256:eval_manifest" \
      --report-field "runtime.precision=$PRECISION" \
      --report-field "runtime.seed=$seed")
  elif [[ "$kind" == cmax-direct ]]; then
    COMPLETION_COMMAND+=(--identity "head_initialization=checkpoint_cmax" \
      --report-field "command=cmax-eval" \
      --report-field "checkpoint.path=$checkpoint" \
      --report-field "checkpoint.checkpoint_sha256=@sha256:checkpoint" \
      --report-field "protocol.stage=$STAGE" \
      --report-field "protocol.alignment.mode=$alignment" \
      --report-field "protocol.flow_rate.cli_value=$dt" \
      --report-field "protocol.event_history.history_steps=$HISTORY_STEPS" \
      --report-field "head.initialization=checkpoint_cmax" \
      --report-field "evaluation.manifest=$eval_manifest" \
      --report-field "evaluation.manifest_artifact.sha256=@sha256:eval_manifest" \
      --report-field "runtime.precision=$PRECISION" \
      --report-field "runtime.seed=$seed")
  else
    COMPLETION_COMMAND+=(--identity "head_initialization=random" \
      --identity "epochs=$EPOCHS" \
      --identity "learning_rate=$LEARNING_RATE" \
      --identity "weight_decay=$WEIGHT_DECAY" \
      --identity "max_train_samples=$MAX_TRAIN_SAMPLES" \
      --identity "include_ood=$INCLUDE_OOD" \
      --artifact "head=$output/checkpoint-final.pt" \
      --artifact "protocol=$output/protocol.json" \
      --report-field "protocol.stage=$STAGE" \
      --report-field "protocol.encoder_checkpoint=$checkpoint" \
      --report-field "protocol.encoder_checkpoint_sha256=@sha256:checkpoint" \
      --report-field "protocol.backbone.history_steps=$HISTORY_STEPS" \
      --report-field "protocol.head.initialization_seed=$seed" \
      --report-field "protocol.training_policy.fixed_epochs=$EPOCHS" \
      --report-field "protocol.training_policy.batch_size=$BATCH_SIZE" \
      --report-field "protocol.training_policy.learning_rate=$LEARNING_RATE" \
      --report-field "protocol.training_policy.weight_decay=$WEIGHT_DECAY" \
      --report-field "protocol.training_policy.seed=$seed" \
      --report-field "protocol.training_policy.precision=$PRECISION" \
      --report-field "protocol.train_targets.manifest=$TRAIN_MANIFEST" \
      --report-field "protocol.train_targets.manifest_sha256=@sha256:train_manifest" \
      --report-field "protocol.train_targets.alignment=$alignment" \
      --report-field "protocol.evaluation_targets.0.manifest=$eval_manifest" \
      --report-field "protocol.evaluation_targets.0.manifest_sha256=@sha256:eval_manifest" \
      --report-field "metrics.protocol_stage=$STAGE")
    if ((INCLUDE_OOD)); then
      COMPLETION_COMMAND+=( \
        --report-field "protocol.evaluation_targets.1.manifest=$OOD_MANIFEST" \
        --report-field \
        "protocol.evaluation_targets.1.manifest_sha256=@sha256:ood_manifest")
    fi
  fi
}

print_command() { printf '  '; printf '%q ' "$@"; printf '\n'; }

if [[ "$STAGE" == dev ]]; then
  printf 'MVSEC evaluation ablation: action=%s stage=dev suite=%s jobs=%s\n' \
    "$ACTION" "$SUITE" "${#JOB_RUN_IDS[@]}"
else
  printf 'MVSEC evaluation ablation: action=%s stage=final selection=%s jobs=%s\n' \
    "$ACTION" "$SELECTED_RUN_ID" "${#JOB_RUN_IDS[@]}"
fi
printf '  checkpoint root: %s\n  pretrain epoch:  %s\n  probe seeds:     %s\n  train manifest:  %s\n  output root:     %s\n' \
  "$CHECKPOINT_ROOT" "$PRETRAIN_EPOCH" "$PROBE_SEEDS" "$TRAIN_MANIFEST" "$OUTPUT_ROOT"
printf '  downstream history: %s fixed steps\n' "$HISTORY_STEPS"
if ((SKIP_COMPLETE)); then printf '%s\n' '  completed-job policy: strict hash-verified reuse'; fi
if [[ "$STAGE" == dev ]]; then
  printf '  dev source:      outdoor_day2 tail fraction=%s\n' "$DEV_FRACTION"
else
  printf '  selected run:    %s\n  final test:      %s\n' "$SELECTED_RUN_ID" "$TEST_MANIFEST"
fi
if ((INCLUDE_OOD)); then printf '  OOD report:      %s\n' "$OOD_MANIFEST"; fi

if [[ "$ACTION" == plan ]]; then
  for ((index=0; index<${#JOB_RUN_IDS[@]}; index++)); do
    build_command "$index"
    printf '\n[%s] %s alignment=%s dt=%s probe_seed=%s\n' \
      "${JOB_RUN_IDS[$index]}" "${JOB_KINDS[$index]}" \
      "${JOB_ALIGNMENTS[$index]}" "${JOB_DTS[$index]}" \
      "${JOB_PROBE_SEEDS[$index]}"
    print_command env "PYTHONPATH=$PROJECT_ROOT/src" PYTHONUNBUFFERED=1 "${COMMAND[@]}"
  done
  exit 0
fi

[[ -f "$TRAIN_MANIFEST" ]] || { printf 'Train manifest is missing: %s\n' "$TRAIN_MANIFEST" >&2; exit 1; }
"$PYTHON_BIN" "$HELPER" manifest --path "$TRAIN_MANIFEST" \
  --expected-recording outdoor_day2 --expected-split train \
  --expected-cameras left,right --require-artifacts >/dev/null
if [[ "$STAGE" == final ]]; then
  [[ -f "$TEST_MANIFEST" ]] || { printf 'Final-test manifest is missing: %s\n' "$TEST_MANIFEST" >&2; exit 1; }
  "$PYTHON_BIN" "$HELPER" manifest --path "$TEST_MANIFEST" \
    --expected-recording outdoor_day1 --expected-split test \
    --expected-cameras left --require-artifacts >/dev/null
  if ((INCLUDE_OOD)); then
    [[ -f "$OOD_MANIFEST" ]] || { printf 'OOD manifest is missing: %s\n' "$OOD_MANIFEST" >&2; exit 1; }
    "$PYTHON_BIN" "$HELPER" manifest --path "$OOD_MANIFEST" \
      --expected-recording outdoor_night1 --expected-split test \
      --expected-cameras left --require-artifacts >/dev/null
  fi
fi

# Validate every input and destination before starting the first expensive job.
SEEN_CHECKPOINTS=' '
for run_id in "${JOB_RUN_IDS[@]}"; do
  case "$SEEN_CHECKPOINTS" in *" $run_id "*) continue ;; esac
  SEEN_CHECKPOINTS="$SEEN_CHECKPOINTS$run_id "
  checkpoint=$(checkpoint_for "$run_id")
  [[ -f "$checkpoint" ]] || { printf 'Checkpoint is missing: %s\n' "$checkpoint" >&2; exit 1; }
done
JOB_SKIPS=()
for ((index=0; index<${#JOB_RUN_IDS[@]}; index++)); do
  output=${JOB_OUTPUTS[$index]}
  log=${JOB_LOGS[$index]}
  if ((SKIP_COMPLETE)) && [[ -e "$output" || -e "$log" ]]; then
    [[ -d "$output" && -f "$log" ]] || {
      printf 'Existing evaluation job is partial: %s / %s\n' \
        "${JOB_RUN_IDS[$index]}" "${JOB_KINDS[$index]}" >&2
      exit 1
    }
    build_completion_command verify "$index"
    if ! "${COMPLETION_COMMAND[@]}" >/dev/null; then
      printf 'Existing evaluation job failed strict completion verification: %s / %s\n' \
        "${JOB_RUN_IDS[$index]}" "${JOB_KINDS[$index]}" >&2
      exit 1
    fi
    if ((SAVE_VISUALIZATIONS > 0)); then
      "$PYTHON_BIN" "$HELPER" snapshot-index \
        --path "$output/visualizations/index.json" --limit 0 >/dev/null
    fi
    JOB_SKIPS+=(1)
  else
    [[ ! -e "$output" ]] || { printf 'Refusing existing evaluation output: %s\n' "$output" >&2; exit 1; }
    [[ ! -e "$log" ]] || { printf 'Refusing existing evaluation log: %s\n' "$log" >&2; exit 1; }
    JOB_SKIPS+=(0)
  fi
done

mkdir -p "$OUTPUT_ROOT/logs/$STAGE/epoch$(printf '%04d' "$PRETRAIN_EPOCH")"
for ((index=0; index<${#JOB_RUN_IDS[@]}; index++)); do
  if ((${JOB_SKIPS[$index]})); then
    printf '\nSkipping verified completed evaluation %s / %s / %s / %s / probe_seed=%s\n' \
      "${JOB_RUN_IDS[$index]}" "${JOB_KINDS[$index]}" \
      "${JOB_ALIGNMENTS[$index]}" "${JOB_DTS[$index]}" \
      "${JOB_PROBE_SEEDS[$index]}"
    continue
  fi
  build_command "$index"
  printf '\nStarting %s / %s / %s / %s / probe_seed=%s\n' \
    "${JOB_RUN_IDS[$index]}" "${JOB_KINDS[$index]}" \
    "${JOB_ALIGNMENTS[$index]}" "${JOB_DTS[$index]}" \
    "${JOB_PROBE_SEEDS[$index]}"
  print_command env "PYTHONPATH=$PROJECT_ROOT/src" PYTHONUNBUFFERED=1 "${COMMAND[@]}"
  (
    cd "$PROJECT_ROOT"
    PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
      PYTHONUNBUFFERED=1 "${COMMAND[@]}"
  ) 2>&1 | tee "${JOB_LOGS[$index]}"
  if ((SAVE_VISUALIZATIONS > 0)); then
    "$PYTHON_BIN" "$HELPER" snapshot-index \
      --path "${JOB_OUTPUTS[$index]}/visualizations/index.json" --limit 0 >/dev/null
  fi
  build_completion_command record "$index"
  "${COMPLETION_COMMAND[@]}"
done

printf 'Completed all %s selected evaluation jobs.\n' "${#JOB_RUN_IDS[@]}"
