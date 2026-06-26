#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-odb-enable}"
shift || true

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

DATA="${ODB_MM_MIX_DATA:-data/mm-mix-tmdb}"
MODEL="${ODB_MM_MIX_MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
OUTPUT_ROOT="${ODB_MM_MIX_OUTPUT_ROOT:-outputs/hf-trainer-real}"
EVAL_CHECKPOINT="${ODB_HF_EVAL_CHECKPOINT:-${OUTPUT_ROOT}/odb-enable}"
EVAL_OUTPUT_ROOT="${ODB_HF_EVAL_OUTPUT_ROOT:-${EVAL_CHECKPOINT}}"
MAX_STEPS="${ODB_MM_MIX_MAX_STEPS:-20}"
MAX_LENGTH="${ODB_MM_MIX_MAX_LENGTH:-16384}"
IMAGE_MAX_PIXELS="${ODB_MM_MIX_IMAGE_MAX_PIXELS:-589824}"
TOKEN_BUDGET="${ODB_MM_MIX_TOKEN_BUDGET:-12288}"
BUFFER_SIZE="${ODB_MM_MIX_BUFFER_SIZE:-1024}"
NUM_PROCESSES="${ODB_MM_MIX_NUM_PROCESSES:-1}"
ODB_PREFETCH_FACTOR="${ODB_MM_MIX_ODB_PREFETCH_FACTOR:-512}"
STANDARD_PREFETCH_FACTOR="${ODB_MM_MIX_STANDARD_PREFETCH_FACTOR:-2}"
TRAINABLE_KEYWORDS="${ODB_MM_MIX_TRAINABLE_KEYWORDS:-full}"
DEEPSPEED="${ODB_MM_MIX_DEEPSPEED:-configs/ds_z2.json}"
PYTHON_BIN="${PYTHON:-python}"

COMMON_ARGS=(
  --data "${DATA}"
  --model "${MODEL}"
  --split-mode lf_val_size
  --val-size 0.05
  --split-seed 42
  --max-length "${MAX_LENGTH}"
  --image-max-pixels "${IMAGE_MAX_PIXELS}"
  --max-steps "${MAX_STEPS}"
  --trainable-keywords "${TRAINABLE_KEYWORDS}"
)

if [[ -n "${DEEPSPEED}" ]]; then
  COMMON_ARGS+=(--deepspeed "${DEEPSPEED}")
fi

run_train() {
  local loader="$1"
  local integration="${2:-enable}"
  shift 2
  local output_dir="${OUTPUT_ROOT}/${loader}-${integration}"
  local cmd=(
    scripts/train_hf_trainer.py
    "${COMMON_ARGS[@]}"
    --loader "${loader}"
    --output-dir "${output_dir}"
  )

  if [[ "${loader}" == "odb" ]]; then
    cmd+=(
      --odb-integration "${integration}"
      --token-budget "${TOKEN_BUDGET}"
      --buffer-size "${BUFFER_SIZE}"
      --prefetch-factor "${ODB_PREFETCH_FACTOR}"
      --loss-scaling exact
      --join
    )
  else
    cmd+=(
      --fixed-batch-size 1
      --prefetch-factor "${STANDARD_PREFETCH_FACTOR}"
    )
  fi

  if [[ "${NUM_PROCESSES}" != "1" ]]; then
    "${PYTHON_BIN}" -m torch.distributed.run --nproc_per_node="${NUM_PROCESSES}" "${cmd[@]}" "$@"
  else
    "${PYTHON_BIN}" "${cmd[@]}" "$@"
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
    "${PYTHON_BIN}" scripts/inspect_data_pipeline.py \
      --data "${DATA}" \
      --model "${MODEL}" \
      --image-max-pixels "${IMAGE_MAX_PIXELS}" \
      "$@"
    ;;
  eval-valloss|valloss)
    "${PYTHON_BIN}" scripts/eval_valloss.py \
      --checkpoint "${EVAL_CHECKPOINT}" \
      --data "${DATA}" \
      --output-dir "${EVAL_OUTPUT_ROOT}/eval_out_hf_valloss" \
      --split-mode lf_val_size \
      --val-size 0.05 \
      --split-seed 42 \
      --max-length "${MAX_LENGTH}" \
      --image-max-pixels "${IMAGE_MAX_PIXELS}" \
      "$@"
    ;;
  eval-benchmark|benchmark)
    "${PYTHON_BIN}" scripts/eval_benchmark.py \
      --checkpoint "${EVAL_CHECKPOINT}" \
      --output-dir "${EVAL_OUTPUT_ROOT}/mmmu_mc_likelihood_hf" \
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
  eval-valloss Evaluate validation loss for a saved checkpoint.
  benchmark    Run the built-in MMMU-MC benchmark for a saved checkpoint.

Useful environment variables:
  ODB_MM_MIX_DATA=data/mm-mix-tmdb
  ODB_MM_MIX_MODEL=Qwen/Qwen3-VL-2B-Instruct
  ODB_MM_MIX_MAX_STEPS=20        # set to -1 for a full-epoch run
  ODB_MM_MIX_NUM_PROCESSES=8
  ODB_MM_MIX_IMAGE_MAX_PIXELS=589824
  ODB_MM_MIX_GRADIENT_CHECKPOINTING=1
  ODB_MM_MIX_ODB_PREFETCH_FACTOR=512
  ODB_MM_MIX_STANDARD_PREFETCH_FACTOR=2
  ODB_MM_MIX_TRAINABLE_KEYWORDS=full
  ODB_MM_MIX_DEEPSPEED=configs/ds_z2.json
  ODB_HF_EVAL_CHECKPOINT=outputs/hf-trainer-real/odb-enable
EOF
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    echo "Run ./run.sh help for usage." >&2
    exit 2
    ;;
esac
