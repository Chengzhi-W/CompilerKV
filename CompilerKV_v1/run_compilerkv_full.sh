#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${MODEL_PATH:?Set MODEL_PATH to a local checkpoint or Hugging Face model id}"
: "${TABLES_DIR:?Set TABLES_DIR to artifacts produced by compilerkv-compile}"

MODEL_NAME="${MODEL_NAME:-$(basename "${MODEL_PATH}")}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/data/LongBench}"
SAVE_DIR="${SAVE_DIR:-${PROJECT_DIR}/results}"
PROMPT_PPL_FILE="${PROMPT_PPL_FILE:-}"

PPL_ARGS=(--missing_ppl error)
if [[ -n "${PROMPT_PPL_FILE}" ]]; then
  PPL_ARGS+=(--prompt_ppl_file "${PROMPT_PPL_FILE}")
elif [[ -n "${PROMPT_PPL:-}" ]]; then
  PPL_ARGS+=(--prompt_ppl "${PROMPT_PPL}")
else
  echo "Set PROMPT_PPL_FILE (preferred) or PROMPT_PPL for the semantic-risk gate." >&2
  exit 2
fi

cd "${PROJECT_DIR}"
python run/longbench/pred.py \
  --model_path "${MODEL_PATH}" \
  --model_name "${MODEL_NAME}" \
  --method compilerkv \
  --tables_dir "${TABLES_DIR}" \
  --dataset_file "${DATA_DIR}" \
  --save_dir "${SAVE_DIR}" \
  --max_capacity_prompts 512 \
  --window_size 64 \
  --attn_implementation flash_attention_2 \
  --eval_batch_size 1 \
  "${PPL_ARGS[@]}"
