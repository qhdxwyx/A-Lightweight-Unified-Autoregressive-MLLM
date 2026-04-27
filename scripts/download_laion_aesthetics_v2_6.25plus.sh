#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

DATASET_REPO="${DATASET_REPO:-xingjianleng/laion_aesthetics_v2_6.5plus}"
ASSET_ROOT="${ASSET_ROOT:-$REPO_ROOT/assets}"
RAR_STAGE_ROOT="${RAR_STAGE_ROOT:-$ASSET_ROOT/rar256_stage}"
META_ROOT="${META_ROOT:-$RAR_STAGE_ROOT/laion_aesthetics_v2_6.5plus}"
IMAGE_ROOT="${IMAGE_ROOT:-$RAR_STAGE_ROOT/laion_aesthetics_v2_6.5plus_images}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
PROCESSES="${PROCESSES:-1}"
THREADS="${THREADS:-8}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"

mkdir -p "$META_ROOT" "$IMAGE_ROOT"

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

if ! python -c "import img2dataset, pyarrow" >/dev/null 2>&1; then
  python -m pip install -U img2dataset pyarrow
fi

if [ -z "$(find "$META_ROOT" -name '*.parquet' -print -quit 2>/dev/null)" ]; then
  "$HF_CLI_BIN" download "$DATASET_REPO" --local-dir "$META_ROOT" --repo-type dataset
fi

img2dataset \
  --url_list "$META_ROOT" \
  --input_format parquet \
  --url_col URL \
  --caption_col TEXT \
  --output_format files \
  --output_folder "$IMAGE_ROOT" \
  --processes_count "$PROCESSES" \
  --thread_count "$THREADS" \
  --image_size "$IMAGE_SIZE" \
  --resize_mode keep_ratio \
  --resize_only_if_bigger True \
  --skip_reencode True

du -sh "$META_ROOT"
du -sh "$IMAGE_ROOT"
find "$IMAGE_ROOT" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.webp" -o -iname "*.bmp" \) | wc -l
