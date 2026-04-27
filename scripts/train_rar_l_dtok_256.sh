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
ASSET_ROOT="${ASSET_ROOT:-$(pwd)/asset}"
COMMON_ASSET_ROOT="${COMMON_ASSET_ROOT:-$ASSET_ROOT}"
AUTO_DOWNLOAD="${AUTO_DOWNLOAD:-0}"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$ASSET_ROOT" "$COMMON_ASSET_ROOT"

NUM_PROCESSES="${NUM_PROCESSES:-1}"
IMAGE_ROOT="${IMAGE_ROOT:-}"
MANIFEST="${MANIFEST:-}"
PARQUET_ROOT="${PARQUET_ROOT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-output_dir/rar_l_dtok_256}"
TA_TOK_PATH="${TA_TOK_PATH:-$COMMON_ASSET_ROOT/ta_tok.pth}"
VQVAE_PATH="${VQVAE_PATH:-$COMMON_ASSET_ROOT/vq_ds16_t2i.pt}"
RAR_BATCH_SIZE="${RAR_BATCH_SIZE:-8}"
RAR_GRAD_ACCUM_STEPS="${RAR_GRAD_ACCUM_STEPS:-2}"
RAR_NUM_WORKERS="${RAR_NUM_WORKERS:-${DATALOADER_NUM_WORKERS:-8}}"
RAR_PREFETCH_FACTOR="${RAR_PREFETCH_FACTOR:-${DATALOADER_PREFETCH_FACTOR:-2}}"
RAR_WARMUP_STEPS="${RAR_WARMUP_STEPS:-2000}"
RAR_WARMUP_RATIO="${RAR_WARMUP_RATIO:-}"
RAR_MAX_STEPS="${RAR_MAX_STEPS:-250000}"
RAR_INIT_FROM="${RAR_INIT_FROM:-}"
RAR_START_STEP="${RAR_START_STEP:-0}"

resolve_hf_cli() {
  if command -v hf >/dev/null 2>&1; then
    echo "hf"
    return 0
  fi
  if command -v huggingface-cli >/dev/null 2>&1; then
    echo "huggingface-cli"
    return 0
  fi
  echo ""
}

HF_CLI_BIN="${HF_CLI_BIN:-$(resolve_hf_cli)}"
if [ -z "$HF_CLI_BIN" ]; then
  echo "Missing Hugging Face CLI. Install it with:"
  echo "  python -m pip install -U huggingface_hub"
  echo "Then run: hash -r"
  exit 1
fi

if [ -z "$IMAGE_ROOT" ] && [ -z "$MANIFEST" ] && [ -z "$PARQUET_ROOT" ]; then
  echo "Set IMAGE_ROOT, MANIFEST, or PARQUET_ROOT before running."
  exit 1
fi

download_file_if_needed() {
  local repo_id="$1"
  local filename="$2"
  local local_dir="$3"
  if [ -f "$local_dir/$filename" ]; then
    return 0
  fi
  mkdir -p "$local_dir"
  "$HF_CLI_BIN" download "$repo_id" "$filename" --local-dir "$local_dir"
}

require_path() {
  local path="$1"
  if [ ! -e "$path" ]; then
    echo "Missing required path: $path"
    exit 1
  fi
}

if [ "$AUTO_DOWNLOAD" = "1" ]; then
  download_file_if_needed "csuhan/TA-Tok" "ta_tok.pth" "$COMMON_ASSET_ROOT"
  download_file_if_needed "peizesun/llamagen_t2i" "vq_ds16_t2i.pt" "$COMMON_ASSET_ROOT"
fi

require_path "$TA_TOK_PATH"
require_path "$VQVAE_PATH"

CMD=(
  accelerate launch
  --num_processes "$NUM_PROCESSES"
  scripts/train_rar_dtok.py
  --config scripts/rar_l_dtok_256.yaml
  --output-dir "$OUTPUT_DIR"
  --ta-tok-path "$TA_TOK_PATH"
  --vqvae-path "$VQVAE_PATH"
  --batch-size "$RAR_BATCH_SIZE"
  --grad-accum-steps "$RAR_GRAD_ACCUM_STEPS"
  --num-workers "$RAR_NUM_WORKERS"
  --prefetch-factor "$RAR_PREFETCH_FACTOR"
  --warmup-steps "$RAR_WARMUP_STEPS"
  --max-steps "$RAR_MAX_STEPS"
  --start-step "$RAR_START_STEP"
)

if [ -n "$RAR_INIT_FROM" ]; then
  CMD+=(--init-from "$RAR_INIT_FROM")
fi

if [ -n "$RAR_WARMUP_RATIO" ]; then
  CMD+=(--warmup-ratio "$RAR_WARMUP_RATIO")
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
