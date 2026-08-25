#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
FEEDFORWARD_TEMPLATE="$PROJECT_ROOT/configs/pretrain/sequence_r0_feedforward_vits_gen1.yaml"
RECURRENT_TEMPLATE="$PROJECT_ROOT/configs/pretrain/recurrent_r0_convlstm_vits_gen1.yaml"

STAGE=ready
ACTION=plan
SEED=0
SELECTED_INPUT=10ch
SELECTED_INPUT_EXPLICIT=0
SELECTED_MODEL=clstm
NPROC_PER_NODE=3
REQUESTED_NPROC_PER_NODE=$NPROC_PER_NODE
SAMPLE_INDEX=0
DATA_ROOT=/home/iASL/Arata_repo/dataset/gen1_304x240
OUTPUT_ROOT="$PROJECT_ROOT/outputs/pretrain/sequence_sigreg"
PYTHON_BIN=${PYTHON_BIN:-python}
PRECISION=fp32
REQUESTED_PRECISION=$PRECISION
BATCH_SIZE=4
WORKERS=4
SMOKE=0
RESUME=0

usage() {
  printf '%s\n' \
    'Usage: bash scripts/experiments/run_sequence_sigreg_plan.sh [options]' \
    '' \
    '  --stage STAGE            1, 2, 3, ready (plan 1+2), or all (default: ready)' \
    '  --action ACTION          plan, prepare, inspect, run, or all (default: plan)' \
    '  --seed N                 Non-negative training seed (default: 0)' \
    '  --selected-input INPUT   2ch or 10ch; required to execute Stage 2' \
    '  --selected-model MODEL   ff, cgru, or clstm for the Stage 3 plan' \
    '  --nproc-per-node N|auto  GPUs/processes for training (default: 3)' \
    '  --precision MODE         auto, fp32, fp16, or bf16 (default: fp32)' \
    '  --batch-size N           Per-rank batch; even and >=2 (default: 4)' \
    '  --workers N              DataLoader workers per rank (default: 4)' \
    '  --smoke                  Isolated 1-epoch, 2-global-batch hardware check' \
    '  --sample-index N         Dataset sample used by inspect (default: 0)' \
    '  --data-root DIR          Processed Gen1 dataset root' \
    '  --output-root DIR        Experiment artifact root' \
    '  --python PATH            Python executable (default: $PYTHON_BIN or python)' \
    '  --resume                 Resume only from each run checkpoint-latest.pt' \
    '  -h, --help               Show this help' \
    '' \
    'Safety rules:' \
    '  * The default action only prints the matrix and commands.' \
    '  * prepare never overwrites a run directory.' \
    '  * run refuses an existing run unless --resume is explicit.' \
    '  * ready is plan-only: Stage 2 must wait for the selected Stage 1 input.' \
    '  * executing Stage 2 requires an explicit --selected-input.' \
    '  * Stage 3 can only be planned until SIGReg is implemented.' \
    '  * smoke runs use a distinct run ID and cannot be resumed.' \
    '  * nproc=auto uses all visible GPUs; precision=auto uses their common mode.' \
    '  * resume requires concrete GPU count and precision, not auto.' \
    '  * --resume is accepted only with --action plan or --action run.'
}

require_option_value() {
  if (($# < 2)); then
    printf 'Missing value for %s\n' "$1" >&2
    exit 2
  fi
}

resolve_auto_runtime() {
  local resolution
  local resolved_nproc
  local resolved_precision
  local extra

  if [[ "$REQUESTED_NPROC_PER_NODE" != auto && \
        "$REQUESTED_PRECISION" != auto ]]; then
    return
  fi
  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    printf 'Python is required to resolve automatic GPU settings: %s\n' \
      "$PYTHON_BIN" >&2
    exit 1
  fi
  if ! resolution=$("$PYTHON_BIN" - \
      "$REQUESTED_NPROC_PER_NODE" "$REQUESTED_PRECISION" <<'PY'
from __future__ import annotations

import sys

try:
    import torch
except Exception as error:
    raise SystemExit(
        f"could not import PyTorch to resolve automatic GPU settings: {error}"
    ) from error


requested_nproc = sys.argv[1]
requested_precision = sys.argv[2]
if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA is unavailable; pass concrete values for offline planning, or run "
        "automatic detection on the training server"
    )
visible_devices = torch.cuda.device_count()
if visible_devices < 1:
    raise SystemExit("no CUDA devices are visible")
nproc = visible_devices if requested_nproc == "auto" else int(requested_nproc)
if nproc > visible_devices:
    raise SystemExit(
        f"requested {nproc} processes but only {visible_devices} CUDA devices are visible"
    )


def has_native_bf16(index: int) -> bool:
    properties = torch.cuda.get_device_properties(index)
    if properties.major < 8:
        return False
    checker = getattr(torch.cuda, "is_bf16_supported", None)
    if checker is None:
        return False
    with torch.cuda.device(index):
        try:
            return bool(checker(including_emulation=False))
        except TypeError:
            # Older PyTorch releases do not expose including_emulation. The
            # compute-capability guard above still excludes emulated V100 BF16.
            return bool(checker())


properties = [torch.cuda.get_device_properties(index) for index in range(nproc)]
for index, item in enumerate(properties):
    print(
        "[runtime-auto] "
        f"device={index} name={item.name} capability={item.major}.{item.minor}",
        file=sys.stderr,
    )
all_native_bf16 = all(has_native_bf16(index) for index in range(nproc))
all_native_fp16 = all(item.major >= 7 for item in properties)
if requested_precision == "auto":
    if all_native_bf16:
        precision = "bf16"
    elif all_native_fp16:
        precision = "fp16"
    else:
        precision = "fp32"
else:
    precision = requested_precision

if precision == "bf16" and not all_native_bf16:
    details = ", ".join(
        f"device {index} ({item.name}, capability={item.major}.{item.minor})"
        for index, item in enumerate(properties)
        if not has_native_bf16(index)
    )
    raise SystemExit(
        "precision=bf16 requires native BF16 on every selected GPU; " + details
    )
if precision == "fp16" and not all_native_fp16:
    details = ", ".join(
        f"device {index} ({item.name}, capability={item.major}.{item.minor})"
        for index, item in enumerate(properties)
        if item.major < 7
    )
    raise SystemExit(
        "precision=fp16 requires Volta-or-newer GPUs in this runner; " + details
    )

print(f"{nproc}\t{precision}")
PY
  ); then
    printf '%s\n' 'Failed to resolve automatic GPU settings.' >&2
    exit 1
  fi
  resolution=${resolution##*$'\n'}
  IFS=$'\t' read -r resolved_nproc resolved_precision extra <<< "$resolution"
  case "$resolved_nproc" in
    ''|*[!0-9]*|0)
      printf 'Invalid automatically resolved GPU count: %s\n' "$resolved_nproc" >&2
      exit 1
      ;;
  esac
  case "$resolved_precision" in
    fp32|fp16|bf16) ;;
    *)
      printf 'Invalid automatically resolved precision: %s\n' \
        "$resolved_precision" >&2
      exit 1
      ;;
  esac
  if [[ -n "${extra:-}" ]]; then
    printf 'Unexpected automatic runtime result: %s\n' "$resolution" >&2
    exit 1
  fi
  NPROC_PER_NODE=$resolved_nproc
  PRECISION=$resolved_precision
  printf '[runtime-auto] nproc_per_node=%s precision=%s\n' \
    "$NPROC_PER_NODE" "$PRECISION" >&2
}

while (($#)); do
  case "$1" in
    --stage)
      require_option_value "$@"
      STAGE=$2
      shift 2
      ;;
    --action)
      require_option_value "$@"
      ACTION=$2
      shift 2
      ;;
    --seed)
      require_option_value "$@"
      SEED=$2
      shift 2
      ;;
    --selected-input)
      require_option_value "$@"
      SELECTED_INPUT=$2
      SELECTED_INPUT_EXPLICIT=1
      shift 2
      ;;
    --selected-model)
      require_option_value "$@"
      SELECTED_MODEL=$2
      shift 2
      ;;
    --nproc-per-node)
      require_option_value "$@"
      NPROC_PER_NODE=$2
      REQUESTED_NPROC_PER_NODE=$2
      shift 2
      ;;
    --precision)
      require_option_value "$@"
      PRECISION=$2
      REQUESTED_PRECISION=$2
      shift 2
      ;;
    --batch-size)
      require_option_value "$@"
      BATCH_SIZE=$2
      shift 2
      ;;
    --workers)
      require_option_value "$@"
      WORKERS=$2
      shift 2
      ;;
    --smoke)
      SMOKE=1
      shift
      ;;
    --sample-index)
      require_option_value "$@"
      SAMPLE_INDEX=$2
      shift 2
      ;;
    --data-root)
      require_option_value "$@"
      DATA_ROOT=$2
      shift 2
      ;;
    --output-root)
      require_option_value "$@"
      OUTPUT_ROOT=$2
      shift 2
      ;;
    --python)
      require_option_value "$@"
      PYTHON_BIN=$2
      shift 2
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$STAGE" in
  1|2|3|ready|all) ;;
  *)
    printf 'Unsupported stage: %s\n' "$STAGE" >&2
    exit 2
    ;;
esac
case "$ACTION" in
  plan|prepare|inspect|run|all) ;;
  *)
    printf 'Unsupported action: %s\n' "$ACTION" >&2
    exit 2
    ;;
esac
case "$SELECTED_INPUT" in
  2ch|10ch) ;;
  *)
    printf 'Unsupported selected input: %s (expected 2ch or 10ch)\n' \
      "$SELECTED_INPUT" >&2
    exit 2
    ;;
esac
case "$SELECTED_MODEL" in
  ff|cgru|clstm) ;;
  *)
    printf 'Unsupported selected model: %s (expected ff, cgru, or clstm)\n' \
      "$SELECTED_MODEL" >&2
    exit 2
    ;;
esac
case "$SEED" in
  ''|*[!0-9]*)
    printf 'seed must be a non-negative integer: %s\n' "$SEED" >&2
    exit 2
    ;;
esac
case "$NPROC_PER_NODE" in
  auto) ;;
  ''|*[!0-9]*|0)
    printf 'nproc-per-node must be auto or a positive integer: %s\n' \
      "$NPROC_PER_NODE" >&2
    exit 2
    ;;
esac
case "$PRECISION" in
  auto|fp32|fp16|bf16) ;;
  *)
    printf 'Unsupported precision: %s (expected auto, fp32, fp16, or bf16)\n' \
      "$PRECISION" >&2
    exit 2
    ;;
esac
case "$BATCH_SIZE" in
  ''|*[!0-9]*|0|1)
    printf 'batch-size must be an even per-rank integer >=2: %s\n' \
      "$BATCH_SIZE" >&2
    exit 2
    ;;
esac
if ((10#$BATCH_SIZE % 2 != 0)); then
  printf '%s\n' \
    'batch-size must be even because mixed sampling uses a 0.5 stream ratio.' >&2
  exit 2
fi
case "$WORKERS" in
  ''|*[!0-9]*)
    printf 'workers must be a non-negative integer per rank: %s\n' \
      "$WORKERS" >&2
    exit 2
    ;;
esac
case "$SAMPLE_INDEX" in
  ''|*[!0-9]*)
    printf 'sample-index must be a non-negative integer: %s\n' "$SAMPLE_INDEX" >&2
    exit 2
    ;;
esac
if [[ "$RESUME" == 1 && "$ACTION" != plan && "$ACTION" != run ]]; then
  printf '%s\n' '--resume is accepted only with --action plan or --action run.' >&2
  exit 2
fi
if [[ "$RESUME" == 1 && "$SMOKE" == 1 ]]; then
  printf '%s\n' '--smoke cannot be combined with --resume.' >&2
  exit 2
fi
if [[ "$RESUME" == 1 && \
      ("$REQUESTED_NPROC_PER_NODE" == auto || "$REQUESTED_PRECISION" == auto) ]]; then
  printf '%s\n' \
    '--resume requires concrete --nproc-per-node and --precision values.' \
    'Read them from the original run ID or launch_metadata.txt.' >&2
  exit 2
fi
if [[ "$STAGE" == ready && "$ACTION" != plan ]]; then
  printf '%s\n' \
    'stage=ready is plan-only. Run Stage 1 first, then pass its selected input' \
    'explicitly with --stage 2 --selected-input {2ch,10ch}.' >&2
  exit 2
fi
if [[ "$STAGE" == 2 && "$ACTION" != plan && "$SELECTED_INPUT_EXPLICIT" != 1 ]]; then
  printf '%s\n' \
    'Stage 2 requires an explicit --selected-input 2ch or --selected-input 10ch' \
    'chosen from the completed Stage 1 comparison.' >&2
  exit 2
fi

case "$DATA_ROOT" in
  /*) ;;
  *) DATA_ROOT="$PROJECT_ROOT/$DATA_ROOT" ;;
esac
case "$OUTPUT_ROOT" in
  /*) ;;
  *) OUTPUT_ROOT="$PROJECT_ROOT/$OUTPUT_ROOT" ;;
esac
DATA_ROOT=${DATA_ROOT%/}
OUTPUT_ROOT=${OUTPUT_ROOT%/}
if [[ -z "$OUTPUT_ROOT" || "$OUTPUT_ROOT" == / || "$OUTPUT_ROOT" == "$PROJECT_ROOT" ]]; then
  printf 'Unsafe output root: %s\n' "$OUTPUT_ROOT" >&2
  exit 2
fi
if [[ "$DATA_ROOT" == *$'\n'* || "$OUTPUT_ROOT" == *$'\n'* ]]; then
  printf '%s\n' 'Paths containing newlines are unsupported.' >&2
  exit 2
fi

resolve_auto_runtime

TRAIN_MANIFEST="$DATA_ROOT/manifests/train.jsonl"
GLOBAL_BATCH_SIZE=$((10#$NPROC_PER_NODE * 10#$BATCH_SIZE))
RUN_SUFFIX="np${NPROC_PER_NODE}_bs${BATCH_SIZE}_${PRECISION}_seed${SEED}"
if [[ "$SMOKE" == 1 ]]; then
  RUN_SUFFIX="${RUN_SUFFIX}_smoke"
  CONFIG_SAMPLES_PER_EPOCH=$((2 * GLOBAL_BATCH_SIZE))
  CONFIG_EPOCHS=1
  CONFIG_WARMUP_EPOCHS=0
else
  CONFIG_SAMPLES_PER_EPOCH=6250
  CONFIG_EPOCHS=100
  CONFIG_WARMUP_EPOCHS=10
fi
SPEC_STAGES=()
SPEC_RUN_IDS=()
SPEC_INPUTS=()
SPEC_MODELS=()

add_spec() {
  SPEC_STAGES+=("$1")
  SPEC_RUN_IDS+=("$2")
  SPEC_INPUTS+=("$3")
  SPEC_MODELS+=("$4")
}

add_stage1_specs() {
  add_spec 1 "s1_input_2ch_ff_nosig_${RUN_SUFFIX}" 2ch ff
  add_spec 1 "s1_input_10ch_ff_nosig_${RUN_SUFFIX}" 10ch ff
}

add_stage2_specs() {
  add_spec 2 "s2_input_${SELECTED_INPUT}_ff_nosig_${RUN_SUFFIX}" \
    "$SELECTED_INPUT" ff
  add_spec 2 "s2_input_${SELECTED_INPUT}_cgru_nosig_${RUN_SUFFIX}" \
    "$SELECTED_INPUT" cgru
  add_spec 2 "s2_input_${SELECTED_INPUT}_clstm_nosig_${RUN_SUFFIX}" \
    "$SELECTED_INPUT" clstm
}

case "$STAGE" in
  1) add_stage1_specs ;;
  2) add_stage2_specs ;;
  3) ;;
  ready)
    add_stage1_specs
    add_stage2_specs
    ;;
  all)
    add_stage1_specs
    add_stage2_specs
    ;;
esac

stage3_selected() {
  [[ "$STAGE" == 3 || "$STAGE" == all ]]
}

print_stage3_plan() {
  local prefix="s3_input_${SELECTED_INPUT}_${SELECTED_MODEL}"
  printf '%s\n' \
    '' \
    'Stage 3 (planned, BLOCKED):' \
    "  ${prefix}_nosig_${RUN_SUFFIX}" \
    "  ${prefix}_sigreg_global_${RUN_SUFFIX}" \
    "  ${prefix}_sigreg_tc_${RUN_SUFFIX}" \
    "  ${prefix}_sigreg_event_tc_${RUN_SUFFIX}" \
    '  BLOCKED: the SIGReg loss/projector/objective is not implemented yet.' \
    '  No placeholder config is generated, so these conditions cannot silently' \
    '  run as the current EMA-JEPA baseline.'
}

if stage3_selected && [[ "$ACTION" != plan ]]; then
  print_stage3_plan >&2
  printf '\nRefusing action=%s before any files or training runs are started.\n' \
    "$ACTION" >&2
  exit 3
fi

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf 'Required command is unavailable: %s\n' "$1" >&2
    exit 1
  fi
}

require_file() {
  if [[ ! -f "$1" ]]; then
    printf 'Required file does not exist: %s\n' "$1" >&2
    exit 1
  fi
}

preflight_cuda_environment() {
  "$PYTHON_BIN" - "$NPROC_PER_NODE" "$PRECISION" <<'PY'
from __future__ import annotations

import sys
from contextlib import nullcontext

import torch


required_devices = int(sys.argv[1])
precision = sys.argv[2]
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the selected Python environment")
visible_devices = torch.cuda.device_count()
if visible_devices < required_devices:
    raise SystemExit(
        f"requested {required_devices} processes but only {visible_devices} CUDA devices are visible"
    )


def has_native_bf16(index: int) -> bool:
    properties = torch.cuda.get_device_properties(index)
    if properties.major < 8:
        return False
    checker = getattr(torch.cuda, "is_bf16_supported", None)
    if checker is None:
        return False
    with torch.cuda.device(index):
        try:
            return bool(checker(including_emulation=False))
        except TypeError:
            return bool(checker())


if precision == "bf16":
    non_native_bf16 = []
    for index in range(required_devices):
        properties = torch.cuda.get_device_properties(index)
        if not has_native_bf16(index):
            non_native_bf16.append(
                f"device {index} ({properties.name}, capability="
                f"{properties.major}.{properties.minor})"
            )
    if non_native_bf16:
        details = ", ".join(non_native_bf16)
        raise SystemExit(
            "precision=bf16 requires native Ampere-or-newer BF16 support; "
            f"{details}. Use fp16 on Volta/Turing or fp32"
        )
elif precision == "fp16":
    non_native_fp16 = []
    for index in range(required_devices):
        properties = torch.cuda.get_device_properties(index)
        if properties.major < 7:
            non_native_fp16.append(
                f"device {index} ({properties.name}, capability="
                f"{properties.major}.{properties.minor})"
            )
    if non_native_fp16:
        details = ", ".join(non_native_fp16)
        raise SystemExit(
            "precision=fp16 requires Volta-or-newer GPUs in this runner; "
            f"{details}. Use fp32"
        )

autocast_dtype = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}.get(precision)

print(
    "[cuda-preflight] "
    f"torch={torch.__version__} cuda_runtime={torch.version.cuda} "
    f"compiled_arches={torch.cuda.get_arch_list()}"
)
for index in range(required_devices):
    properties = torch.cuda.get_device_properties(index)
    try:
        with torch.cuda.device(index):
            value = torch.ones((16, 16), device=f"cuda:{index}")
            context = (
                torch.autocast(device_type="cuda", dtype=autocast_dtype)
                if autocast_dtype is not None
                else nullcontext()
            )
            with context:
                result = value @ value
            torch.cuda.synchronize()
        if float(result[0, 0]) != 16.0:
            raise RuntimeError("unexpected CUDA matmul result")
    except Exception as error:
        raise SystemExit(
            f"CUDA kernel check failed on device {index} "
            f"({properties.name}, capability={properties.major}.{properties.minor}): {error}"
        ) from error
    print(
        "[cuda-preflight] "
        f"device={index} name={properties.name} "
        f"capability={properties.major}.{properties.minor} ok"
    )
PY
}

run_dir() {
  printf '%s/stage%s/%s\n' "$OUTPUT_ROOT" "$1" "$2"
}

config_path() {
  printf '%s/launch_config.yaml\n' "$(run_dir "$1" "$2")"
}

inspection_path() {
  printf '%s/inspection/recurrent-clip.html\n' "$(run_dir "$1" "$2")"
}

output_path_for_config() {
  local path
  path=$(run_dir "$1" "$2")
  case "$path" in
    "$PROJECT_ROOT"/*) printf '%s\n' "${path#"$PROJECT_ROOT"/}" ;;
    *) printf '%s\n' "$path" ;;
  esac
}

temporal_bins() {
  case "$1" in
    2ch) printf '1\n' ;;
    10ch) printf '5\n' ;;
    *) return 1 ;;
  esac
}

model_template() {
  case "$1" in
    ff) printf '%s\n' "$FEEDFORWARD_TEMPLATE" ;;
    cgru|clstm) printf '%s\n' "$RECURRENT_TEMPLATE" ;;
    *) return 1 ;;
  esac
}

expected_objective() {
  case "$1" in
    ff) printf 'sequence_dense_window_jepa\n' ;;
    cgru|clstm) printf 'recurrent_dense_window_jepa\n' ;;
    *) return 1 ;;
  esac
}

expected_model_setting() {
  case "$1" in
    ff) printf '  temporal_model: feedforward\n' ;;
    cgru) printf '  cell: conv_gru\n' ;;
    clstm) printf '  cell: conv_lstm\n' ;;
    *) return 1 ;;
  esac
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    printf '%s\n' 'sha256sum or shasum is required.' >&2
    return 1
  fi
}

validate_config_content() {
  local config=$1
  local stage=$2
  local run_id=$3
  local input=$4
  local model=$5
  local bins
  local output_path
  local objective
  local model_setting
  bins=$(temporal_bins "$input")
  output_path=$(output_path_for_config "$stage" "$run_id")
  objective=$(expected_objective "$model")
  model_setting=$(expected_model_setting "$model")

  grep -Fqx "  manifest: $TRAIN_MANIFEST" "$config"
  grep -Fqx "  samples_per_epoch: $CONFIG_SAMPLES_PER_EPOCH" "$config"
  grep -Fqx "  batch_size: $BATCH_SIZE" "$config"
  grep -Fqx "  workers: $WORKERS" "$config"
  grep -Fqx "  temporal_bins: $bins" "$config"
  grep -Fqx '  split_polarity: true' "$config"
  grep -Fqx "  objective: $objective" "$config"
  grep -Fqx "  epochs: $CONFIG_EPOCHS" "$config"
  grep -Fqx "  warmup_epochs: $CONFIG_WARMUP_EPOCHS" "$config"
  grep -Fqx "  seed: $SEED" "$config"
  grep -Fqx "  output_dir: $output_path" "$config"
  grep -Fqx "  precision: $PRECISION" "$config"
  grep -Fqx "$model_setting" "$config"
  grep -Fqx '  variance_weight: 0.0' "$config"
  grep -Fqx '  covariance_weight: 0.0' "$config"
  if grep -Eq '^[[:space:]]*sigreg:' "$config"; then
    printf 'Unexpected SIGReg section in no-SIGReg config: %s\n' "$config" >&2
    return 1
  fi
  if grep -Fqx '  return_patch_event_activity: true' "$config"; then
    printf 'Unexpected patch activity payload in no-SIGReg config: %s\n' \
      "$config" >&2
    return 1
  fi
}

prepare_one() {
  local stage=$1
  local run_id=$2
  local input=$3
  local model=$4
  local directory
  local config
  local template
  local bins
  local output_path
  local checksum
  local temporary
  directory=$(run_dir "$stage" "$run_id")
  config=$(config_path "$stage" "$run_id")
  template=$(model_template "$model")
  bins=$(temporal_bins "$input")
  output_path=$(output_path_for_config "$stage" "$run_id")

  require_file "$template"
  require_command perl
  if [[ -e "$directory" ]]; then
    printf 'Refusing to overwrite an existing run directory: %s\n' "$directory" >&2
    exit 1
  fi

  temporary=$(mktemp "${TMPDIR:-/tmp}/evjepa-sequence-plan.XXXXXX")
  cp "$template" "$temporary"
  CONFIG_MANIFEST="$TRAIN_MANIFEST" \
  CONFIG_SAMPLES="$CONFIG_SAMPLES_PER_EPOCH" \
  CONFIG_BINS="$bins" \
  CONFIG_BATCH_SIZE="$BATCH_SIZE" \
  CONFIG_WORKERS="$WORKERS" \
  CONFIG_PRECISION="$PRECISION" \
  CONFIG_EPOCHS_VALUE="$CONFIG_EPOCHS" \
  CONFIG_WARMUP="$CONFIG_WARMUP_EPOCHS" \
  CONFIG_SEED="$SEED" \
  CONFIG_OUTPUT="$output_path" \
    perl -0pi -e '
      s{(?m)^  manifest:.*$}{  manifest: $ENV{CONFIG_MANIFEST}};
      s{(?m)^  samples_per_epoch:.*$}{  samples_per_epoch: $ENV{CONFIG_SAMPLES}};
      s{(?m)^  batch_size:.*$}{  batch_size: $ENV{CONFIG_BATCH_SIZE}};
      s{(?m)^  workers:.*$}{  workers: $ENV{CONFIG_WORKERS}};
      s{(?m)^  temporal_bins:.*$}{  temporal_bins: $ENV{CONFIG_BINS}};
      s{(?m)^  epochs:.*$}{  epochs: $ENV{CONFIG_EPOCHS_VALUE}};
      s{(?m)^  warmup_epochs:.*$}{  warmup_epochs: $ENV{CONFIG_WARMUP}};
      s{(?m)^  precision:.*$}{  precision: $ENV{CONFIG_PRECISION}};
      s{(?m)^  seed:.*$}{  seed: $ENV{CONFIG_SEED}};
      s{(?m)^  output_dir:.*$}{  output_dir: $ENV{CONFIG_OUTPUT}};
      s{(?m)^  # Sequence mode counts clips\.[^\n]*\n  # nominal signal[^\n]*}
       {  # Sequence mode counts clips; samples_per_epoch is global across ranks.\n  # Each clip contains eight supervised steps.};
      s{(?m)^  # Recurrent mode counts clips, not individual windows\.[^\n]*\n  # supervised steps[^\n]*}
       {  # Recurrent mode counts clips; samples_per_epoch is global across ranks.\n  # Each clip contains eight supervised steps.};
    ' "$temporary"
  if [[ "$model" == cgru ]]; then
    perl -0pi -e 's/conv_lstm/conv_gru/g; s/ConvLSTM/ConvGRU/g' "$temporary"
  fi
  perl -0pi -e '
    s{# RVT-style mixed batching: per-rank batch 4 = 2 stream \+ 2 random clips\.}
     {# RVT-style mixed batching: half stream and half random clips per rank.};
  ' "$temporary"
  validate_config_content "$temporary" "$stage" "$run_id" "$input" "$model"

  mkdir -p "$(dirname "$directory")"
  if ! mkdir "$directory"; then
    printf 'Could not reserve fresh run directory: %s\n' "$directory" >&2
    exit 1
  fi
  mv "$temporary" "$config"
  checksum=$(sha256_file "$config")
  printf '%s  %s\n' "$checksum" "$(basename "$config")" > "$directory/launch_config.sha256"
  {
    printf 'format=2\n'
    printf 'stage=%s\n' "$stage"
    printf 'run_id=%s\n' "$run_id"
    printf 'seed=%s\n' "$SEED"
    printf 'input=%s\n' "$input"
    printf 'model=%s\n' "$model"
    printf 'nproc_per_node_request=%s\n' "$REQUESTED_NPROC_PER_NODE"
    printf 'nproc_per_node=%s\n' "$NPROC_PER_NODE"
    printf 'batch_size_per_rank=%s\n' "$BATCH_SIZE"
    printf 'global_batch_size=%s\n' "$GLOBAL_BATCH_SIZE"
    printf 'workers_per_rank=%s\n' "$WORKERS"
    printf 'precision_request=%s\n' "$REQUESTED_PRECISION"
    printf 'precision=%s\n' "$PRECISION"
    printf 'smoke=%s\n' "$SMOKE"
    printf 'samples_per_epoch=%s\n' "$CONFIG_SAMPLES_PER_EPOCH"
    printf 'epochs=%s\n' "$CONFIG_EPOCHS"
    printf 'warmup_epochs=%s\n' "$CONFIG_WARMUP_EPOCHS"
    printf 'train_manifest=%s\n' "$TRAIN_MANIFEST"
  } > "$directory/launch_metadata.txt"
  printf 'Prepared: %s\n' "$config"
}

require_prepared() {
  local stage=$1
  local run_id=$2
  local input=$3
  local model=$4
  local directory
  local config
  local checksum_file
  local expected_checksum
  local actual_checksum
  directory=$(run_dir "$stage" "$run_id")
  config=$(config_path "$stage" "$run_id")
  checksum_file="$directory/launch_config.sha256"
  require_file "$config"
  require_file "$checksum_file"
  require_file "$directory/launch_metadata.txt"
  expected_checksum=$(awk 'NR == 1 {print $1}' "$checksum_file")
  actual_checksum=$(sha256_file "$config")
  if [[ -z "$expected_checksum" || "$actual_checksum" != "$expected_checksum" ]]; then
    printf 'Prepared config was modified; refusing to run: %s\n' "$config" >&2
    exit 1
  fi
  validate_config_content "$config" "$stage" "$run_id" "$input" "$model"
  grep -Fqx 'format=2' "$directory/launch_metadata.txt"
  grep -Fqx "stage=$stage" "$directory/launch_metadata.txt"
  grep -Fqx "run_id=$run_id" "$directory/launch_metadata.txt"
  grep -Fqx "seed=$SEED" "$directory/launch_metadata.txt"
  grep -Fqx "input=$input" "$directory/launch_metadata.txt"
  grep -Fqx "model=$model" "$directory/launch_metadata.txt"
  grep -Fqx "nproc_per_node=$NPROC_PER_NODE" "$directory/launch_metadata.txt"
  grep -Fqx "batch_size_per_rank=$BATCH_SIZE" "$directory/launch_metadata.txt"
  grep -Fqx "global_batch_size=$GLOBAL_BATCH_SIZE" "$directory/launch_metadata.txt"
  grep -Fqx "workers_per_rank=$WORKERS" "$directory/launch_metadata.txt"
  grep -Fqx "precision=$PRECISION" "$directory/launch_metadata.txt"
  grep -Fqx "smoke=$SMOKE" "$directory/launch_metadata.txt"
  grep -Fqx "samples_per_epoch=$CONFIG_SAMPLES_PER_EPOCH" \
    "$directory/launch_metadata.txt"
  grep -Fqx "epochs=$CONFIG_EPOCHS" "$directory/launch_metadata.txt"
  grep -Fqx "warmup_epochs=$CONFIG_WARMUP_EPOCHS" \
    "$directory/launch_metadata.txt"
  grep -Fqx "train_manifest=$TRAIN_MANIFEST" "$directory/launch_metadata.txt"
}

build_inspect_command() {
  local config=$1
  local output=$2
  COMMAND=(
    "$PYTHON_BIN"
    -m event_window_jepa.recurrent_inspection
    --config "$config"
    --expected-dataset gen1
    --sample-index "$SAMPLE_INDEX"
    --output "$output"
  )
}

build_train_command() {
  local config=$1
  local checkpoint=$2
  if [[ "$NPROC_PER_NODE" == 1 ]]; then
    COMMAND=(
      "$PYTHON_BIN"
      -m event_window_jepa.train.pretrain
      --config "$config"
    )
  else
    COMMAND=(
      "$PYTHON_BIN"
      -m torch.distributed.run
      --standalone
      "--nproc-per-node=$NPROC_PER_NODE"
      -m event_window_jepa.train.pretrain
      --config "$config"
    )
  fi
  if [[ "$RESUME" == 1 ]]; then
    COMMAND+=(--resume "$checkpoint")
  fi
}

print_command() {
  local path_value="$PROJECT_ROOT/src"
  if [[ -n "${PYTHONPATH:-}" ]]; then
    path_value="$path_value:$PYTHONPATH"
  fi
  printf '    PYTHONPATH=%q PYTHONUNBUFFERED=1 ' "$path_value"
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
}

inspect_one() {
  local stage=$1
  local run_id=$2
  local input=$3
  local model=$4
  local config
  local output
  config=$(config_path "$stage" "$run_id")
  output=$(inspection_path "$stage" "$run_id")
  require_prepared "$stage" "$run_id" "$input" "$model"
  require_file "$TRAIN_MANIFEST"
  require_command "$PYTHON_BIN"
  if [[ -e "$output" ]]; then
    printf 'Refusing to overwrite an existing inspection: %s\n' "$output" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$output")"
  build_inspect_command "$config" "$output"
  (
    cd "$PROJECT_ROOT"
    export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
    export PYTHONUNBUFFERED=1
    "${COMMAND[@]}"
  )
  require_file "$output"
  printf 'Inspected: %s\n' "$output"
}

run_one() {
  local stage=$1
  local run_id=$2
  local input=$3
  local model=$4
  local directory
  local config
  local checkpoint
  local log
  directory=$(run_dir "$stage" "$run_id")
  config=$(config_path "$stage" "$run_id")
  checkpoint="$directory/checkpoint-latest.pt"
  log="$directory/train.log"
  require_prepared "$stage" "$run_id" "$input" "$model"
  require_file "$TRAIN_MANIFEST"
  require_command "$PYTHON_BIN"

  if [[ "$RESUME" == 1 ]]; then
    require_file "$checkpoint"
    require_file "$directory/train.jsonl"
    require_file "$directory/resolved_config.yaml"
  else
    if [[ -e "$directory/.run-attempted" || -e "$checkpoint" || \
          -e "$directory/train.jsonl" || -e "$directory/resolved_config.yaml" || \
          -e "$log" ]]; then
      printf 'Refusing to mix with an existing training attempt: %s\n' "$directory" >&2
      printf '%s\n' 'Use --resume only when checkpoint-latest.pt exists.' >&2
      exit 1
    fi
    printf 'started\n' > "$directory/.run-attempted"
  fi

  build_train_command "$config" "$checkpoint"
  printf 'Training: %s\n' "$run_id"
  if [[ "$RESUME" == 1 ]]; then
    (
      cd "$PROJECT_ROOT"
      export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
      export PYTHONUNBUFFERED=1
      "${COMMAND[@]}"
    ) 2>&1 | tee -a "$log"
  else
    (
      cd "$PROJECT_ROOT"
      export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
      export PYTHONUNBUFFERED=1
      "${COMMAND[@]}"
    ) 2>&1 | tee "$log"
  fi
  require_file "$checkpoint"
  printf 'Completed: %s\n' "$checkpoint"
}

preflight_prepare_one() {
  local directory
  directory=$(run_dir "$1" "$2")
  require_file "$(model_template "$4")"
  if [[ -e "$directory" ]]; then
    printf 'Refusing to overwrite an existing run directory: %s\n' "$directory" >&2
    exit 1
  fi
}

preflight_inspect_one() {
  local output
  require_prepared "$1" "$2" "$3" "$4"
  output=$(inspection_path "$1" "$2")
  if [[ -e "$output" ]]; then
    printf 'Refusing to overwrite an existing inspection: %s\n' "$output" >&2
    exit 1
  fi
}

preflight_run_one() {
  local directory
  local checkpoint
  local log
  directory=$(run_dir "$1" "$2")
  checkpoint="$directory/checkpoint-latest.pt"
  log="$directory/train.log"
  require_prepared "$1" "$2" "$3" "$4"
  if [[ "$RESUME" == 1 ]]; then
    require_file "$checkpoint"
    require_file "$directory/train.jsonl"
    require_file "$directory/resolved_config.yaml"
  elif [[ -e "$directory/.run-attempted" || -e "$checkpoint" || \
          -e "$directory/train.jsonl" || -e "$directory/resolved_config.yaml" || \
          -e "$log" ]]; then
    printf 'Refusing to mix with an existing training attempt: %s\n' "$directory" >&2
    printf '%s\n' 'Use --resume only when checkpoint-latest.pt exists.' >&2
    exit 1
  fi
}

print_plan() {
  local index
  local stage
  local run_id
  local input
  local model
  local config
  local inspection
  local checkpoint
  printf '%s\n' \
    'Sequence/SIGReg staged experiment plan' \
    "  seed=$SEED selected_input=$SELECTED_INPUT selected_model=$SELECTED_MODEL" \
    "  nproc_per_node=$NPROC_PER_NODE precision=$PRECISION" \
    "  batch_size_per_rank=$BATCH_SIZE global_batch_size=$GLOBAL_BATCH_SIZE" \
    "  workers_per_rank=$WORKERS total_worker_processes=$((10#$NPROC_PER_NODE * 10#$WORKERS))" \
    "  smoke=$SMOKE samples_per_epoch=$CONFIG_SAMPLES_PER_EPOCH epochs=$CONFIG_EPOCHS warmup_epochs=$CONFIG_WARMUP_EPOCHS" \
    "  manifest=$TRAIN_MANIFEST" \
    "  output_root=$OUTPUT_ROOT"
  if [[ "$REQUESTED_NPROC_PER_NODE" != "$NPROC_PER_NODE" || \
        "$REQUESTED_PRECISION" != "$PRECISION" ]]; then
    printf '  auto_request: nproc_per_node=%s precision=%s\n' \
      "$REQUESTED_NPROC_PER_NODE" "$REQUESTED_PRECISION"
  fi
  for ((index = 0; index < ${#SPEC_RUN_IDS[@]}; index++)); do
    stage=${SPEC_STAGES[$index]}
    run_id=${SPEC_RUN_IDS[$index]}
    input=${SPEC_INPUTS[$index]}
    model=${SPEC_MODELS[$index]}
    config=$(config_path "$stage" "$run_id")
    inspection=$(inspection_path "$stage" "$run_id")
    checkpoint="$(run_dir "$stage" "$run_id")/checkpoint-latest.pt"
    printf '\n  [%s] input=%s model=%s sigreg=none\n' "$run_id" "$input" "$model"
    printf '    output=%s\n' "$(run_dir "$stage" "$run_id")"
    build_inspect_command "$config" "$inspection"
    printf '%s\n' '    inspect:'
    print_command
    build_train_command "$config" "$checkpoint"
    printf '%s\n' '    train:'
    print_command
  done
  if stage3_selected; then
    print_stage3_plan
  fi
}

run_for_all_specs() {
  local operation=$1
  local index
  for ((index = 0; index < ${#SPEC_RUN_IDS[@]}; index++)); do
    "$operation" \
      "${SPEC_STAGES[$index]}" \
      "${SPEC_RUN_IDS[$index]}" \
      "${SPEC_INPUTS[$index]}" \
      "${SPEC_MODELS[$index]}"
  done
}

case "$ACTION" in
  plan)
    print_plan
    ;;
  prepare)
    run_for_all_specs preflight_prepare_one
    run_for_all_specs prepare_one
    ;;
  inspect)
    require_file "$TRAIN_MANIFEST"
    require_command "$PYTHON_BIN"
    run_for_all_specs preflight_inspect_one
    run_for_all_specs inspect_one
    ;;
  run)
    require_file "$TRAIN_MANIFEST"
    require_command "$PYTHON_BIN"
    preflight_cuda_environment
    run_for_all_specs preflight_run_one
    run_for_all_specs run_one
    ;;
  all)
    require_file "$TRAIN_MANIFEST"
    require_command "$PYTHON_BIN"
    require_command perl
    preflight_cuda_environment
    run_for_all_specs preflight_prepare_one
    run_for_all_specs prepare_one
    run_for_all_specs preflight_inspect_one
    run_for_all_specs inspect_one
    run_for_all_specs preflight_run_one
    run_for_all_specs run_one
    ;;
esac
