#!/usr/bin/env bash
set -euo pipefail

ROOT="${EV_JEPA_ROOT:-/home/iASL/Arata_repo/EV-JEPA}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/env/bin/python}"
DATA_ROOT="${GEN1_DATA_ROOT:-/home/iASL/Arata_repo/dataset/gen1_304x240}"
ARCHITECTURE_CHECKPOINT=""
OUTPUT_DIR="$ROOT/outputs/downstream/gen1_convlstm_scratch_mixed_ddp"
NPROC=3
BATCH_SIZE=2
WORKERS=4
PRECISION=fp16
EPOCHS=30
EVAL_EVERY=5
SEQUENCE_LENGTH=21
SMOKE=0
RESUME=""

usage() {
  printf '%s\n' \
    "Usage: bash scripts/experiments/run_gen1_scratch_convlstm_ddp.sh [options]" \
    "" \
    "  --architecture-checkpoint PATH  ConvLSTM pretrain checkpoint (config only)" \
    "  --output-dir DIR               Output directory" \
    "  --nproc-per-node N             GPUs used by one DDP job (default: 3)" \
    "  --batch-size N                 Per-GPU mixed batch, must be even (default: 2)" \
    "  --workers N                    DataLoader workers per GPU (default: 4)" \
    "  --precision fp16|fp32|bf16     Training precision (default: fp16)" \
    "  --epochs N                     Formal epochs (default: 30)" \
    "  --eval-every N                 Full-stream evaluation cadence (default: 5)" \
    "  --sequence-length N            Frames per clip (default: 21)" \
    "  --smoke                        1 epoch with bounded train/validation labels" \
    "  --resume PATH                  Strict epoch-boundary resume" \
    "  -h, --help                     Show this help"
}

while (($#)); do
  case "$1" in
    --architecture-checkpoint) ARCHITECTURE_CHECKPOINT=$2; shift 2 ;;
    --output-dir) OUTPUT_DIR=$2; shift 2 ;;
    --nproc-per-node) NPROC=$2; shift 2 ;;
    --batch-size) BATCH_SIZE=$2; shift 2 ;;
    --workers) WORKERS=$2; shift 2 ;;
    --precision) PRECISION=$2; shift 2 ;;
    --epochs) EPOCHS=$2; shift 2 ;;
    --eval-every) EVAL_EVERY=$2; shift 2 ;;
    --sequence-length) SEQUENCE_LENGTH=$2; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    --resume) RESUME=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$ARCHITECTURE_CHECKPOINT" ]]; then
  printf '%s\n' "--architecture-checkpoint is required." >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  printf 'Python is not executable: %s\n' "$PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -f "$ARCHITECTURE_CHECKPOINT" ]]; then
  printf 'Architecture checkpoint not found: %s\n' "$ARCHITECTURE_CHECKPOINT" >&2
  exit 2
fi
if ((BATCH_SIZE <= 0 || BATCH_SIZE % 2 != 0)); then
  printf '%s\n' "--batch-size must be a positive even per-GPU value." >&2
  exit 2
fi
if ((NPROC <= 0 || WORKERS < 0 || EPOCHS <= 0 || EVAL_EVERY <= 0)); then
  printf '%s\n' "GPU count, epochs, and eval cadence must be positive; workers cannot be negative." >&2
  exit 2
fi
if [[ "$PRECISION" != fp16 && "$PRECISION" != fp32 && "$PRECISION" != bf16 ]]; then
  printf 'Unsupported precision: %s\n' "$PRECISION" >&2
  exit 2
fi

EXTRA_ARGS=()
if ((SMOKE)); then
  OUTPUT_DIR="${OUTPUT_DIR%/}_smoke"
  EPOCHS=1
  EVAL_EVERY=1
  EXTRA_ARGS+=(--max-train-frames 192 --max-val-frames 64)
fi
if [[ -n "$RESUME" ]]; then
  if ((SMOKE)); then
    printf '%s\n' "--smoke cannot be combined with --resume." >&2
    exit 2
  fi
  if [[ ! -f "$RESUME" ]]; then
    printf 'Resume checkpoint not found: %s\n' "$RESUME" >&2
    exit 2
  fi
  EXTRA_ARGS+=(--resume "$RESUME")
elif [[ -e "$OUTPUT_DIR/train.jsonl" || -e "$OUTPUT_DIR/checkpoint-latest.pt" ]]; then
  printf 'Output already contains a run; use --resume explicitly: %s\n' "$OUTPUT_DIR" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
printf '%s\n' \
  "ConvLSTM scratch sanity: one DDP model" \
  "GPUs=$NPROC, per-GPU batch=$BATCH_SIZE, global batch=$((NPROC * BATCH_SIZE)), T=$SEQUENCE_LENGTH" \
  "sampling=mixed 1:1, validation=full causal stream, precision=$PRECISION" \
  "output=$OUTPUT_DIR"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nproc-per-node="$NPROC" \
  -m event_window_jepa.downstream.gen1_detection_scratch_ddp \
  --architecture-checkpoint "$ARCHITECTURE_CHECKPOINT" \
  --train-manifest "$DATA_ROOT/manifests/train.jsonl" \
  --val-manifest "$DATA_ROOT/manifests/val.jsonl" \
  --output-dir "$OUTPUT_DIR" \
  --window-ms 50 \
  --sequence-length "$SEQUENCE_LENGTH" \
  --batch-size "$BATCH_SIZE" \
  --workers "$WORKERS" \
  --precision "$PRECISION" \
  --epochs "$EPOCHS" \
  --eval-every "$EVAL_EVERY" \
  --seed 0 \
  "${EXTRA_ARGS[@]}"
