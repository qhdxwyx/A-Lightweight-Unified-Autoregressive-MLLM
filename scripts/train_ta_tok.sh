#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$(pwd)/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$(pwd)/.cache/torch_extensions}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$(pwd)/.cache/triton}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR"

NUM_PROCESSES="${NUM_PROCESSES:-1}"
IMAGE_ROOT="${IMAGE_ROOT:-}"
MANIFEST="${MANIFEST:-}"
PARQUET_ROOT="${PARQUET_ROOT:-}"
TA_TOK_OUTPUT_DIR="${TA_TOK_OUTPUT_DIR:-output_dir/ta_tok_train}"
TA_TOK_INIT_FROM="${TA_TOK_INIT_FROM:-}"
TA_TOK_BATCH_SIZE="${TA_TOK_BATCH_SIZE:-2}"
TA_TOK_GRAD_ACCUM_STEPS="${TA_TOK_GRAD_ACCUM_STEPS:-8}"
TA_TOK_NUM_WORKERS="${TA_TOK_NUM_WORKERS:-${DATALOADER_NUM_WORKERS:-8}}"
TA_TOK_PREFETCH_FACTOR="${TA_TOK_PREFETCH_FACTOR:-${DATALOADER_PREFETCH_FACTOR:-4}}"
TA_TOK_WARMUP_STEPS="${TA_TOK_WARMUP_STEPS:-2000}"
TA_TOK_MAX_STEPS="${TA_TOK_MAX_STEPS:-100000}"

if [ -z "$IMAGE_ROOT" ] && [ -z "$MANIFEST" ] && [ -z "$PARQUET_ROOT" ]; then
  echo "Set IMAGE_ROOT, MANIFEST, or PARQUET_ROOT before running."
  exit 1
fi

CMD=(
  accelerate launch
  --num_processes "$NUM_PROCESSES"
  scripts/train_ta_tok.py
  --config scripts/ta_tok_384.yaml
  --output-dir "$TA_TOK_OUTPUT_DIR"
  --batch-size "$TA_TOK_BATCH_SIZE"
  --grad-accum-steps "$TA_TOK_GRAD_ACCUM_STEPS"
  --num-workers "$TA_TOK_NUM_WORKERS"
  --prefetch-factor "$TA_TOK_PREFETCH_FACTOR"
  --warmup-steps "$TA_TOK_WARMUP_STEPS"
  --max-steps "$TA_TOK_MAX_STEPS"
)

if [ -n "$TA_TOK_INIT_FROM" ]; then
  CMD+=(--init-from "$TA_TOK_INIT_FROM")
fi
if [ -n "$IMAGE_ROOT" ]; then
  CMD+=(--image-root "$IMAGE_ROOT")
fi
if [ -n "$MANIFEST" ]; then
  CMD+=(--manifest "$MANIFEST")
fi
if [ -n "$PARQUET_ROOT" ]; then
  CMD+=(--parquet-root "$PARQUET_ROOT")
fi

"${CMD[@]}"
