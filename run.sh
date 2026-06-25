#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-odb-enable}"
shift || true

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

DATA="${ODB_MM_MIX_DATA:-data/mm-mix-tmdb}"
MODEL="${ODB_MM_MIX_MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
OUTPUT_ROOT="${ODB_MM_MIX_OUTPUT_ROOT:-outputs/hf-trainer-real}"
MAX_STEPS="${ODB_MM_MIX_MAX_STEPS:-20}"
MAX_LENGTH="${ODB_MM_MIX_MAX_LENGTH:-16384}"
IMAGE_MAX_PIXELS="${ODB_MM_MIX_IMAGE_MAX_PIXELS:-589824}"
TOKEN_BUDGET="${ODB_MM_MIX_TOKEN_BUDGET:-12288}"
BUFFER_SIZE="${ODB_MM_MIX_BUFFER_SIZE:-1024}"
NUM_PROCESSES="${ODB_MM_MIX_NUM_PROCESSES:-1}"

COMMON_ARGS=(
  --data "${DATA}"
  --model "${MODEL}"
  --split-mode lf_val_size
  --val-size 0.05
  --split-seed 42
  --max-length "${MAX_LENGTH}"
  --image-max-pixels "${IMAGE_MAX_PIXELS}"
  --max-steps "${MAX_STEPS}"
)

run_train() {
  local loader="$1"
  local integration="${2:-enable}"
  local output_dir="${OUTPUT_ROOT}/${loader}-${integration}"
  local cmd=(
    python scripts/train_hf_trainer_real_processor.py
    "${COMMON_ARGS[@]}"
    --loader "${loader}"
    --output-dir "${output_dir}"
  )

  if [[ "${loader}" == "odb" ]]; then
    cmd+=(
      --odb-integration "${integration}"
      --token-budget "${TOKEN_BUDGET}"
      --buffer-size "${BUFFER_SIZE}"
      --loss-scaling exact
      --join
    )
  else
    cmd+=(--fixed-batch-size 1)
  fi

  if [[ "${NUM_PROCESSES}" != "1" ]]; then
    torchrun --nproc_per_node="${NUM_PROCESSES}" "${cmd[@]}" "$@"
  else
    "${cmd[@]}" "$@"
  fi
}

case "${MODE}" in
  odb|odb-enable|enable)
    run_train odb enable "$@"
    ;;
  odb-manual|manual)
    run_train odb manual "$@"
    ;;
  standard)
    run_train standard none "$@"
    ;;
  inspect)
    python scripts/inspect_hf_processor_mm_tokens.py \
      --data "${DATA}" \
      --model "${MODEL}" \
      --image-max-pixels "${IMAGE_MAX_PIXELS}" \
      "$@"
    ;;
  help|-h|--help)
    cat <<'EOF'
Usage:
  ./run.sh [mode] [extra args passed to the underlying script]

Modes:
  odb-enable   Recommended ODB path: enable_odb(...). Default.
  odb-manual   Advanced ODB path: odb.apply(...) + configure_trainer(...).
  standard     Fixed-batch baseline.
  inspect      Inspect HF processor multimodal tensor output.

Useful environment variables:
  ODB_MM_MIX_DATA=data/mm-mix-tmdb
  ODB_MM_MIX_MODEL=Qwen/Qwen3-VL-2B-Instruct
  ODB_MM_MIX_MAX_STEPS=20
  ODB_MM_MIX_NUM_PROCESSES=8
EOF
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    echo "Run ./run.sh help for usage." >&2
    exit 2
    ;;
esac
