#!/bin/bash
# -*- coding: utf-8 -*-

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [ "${WANDB_API_KEY:-}" = "YOUR_WANDB_KEY_HERE" ]; then
  unset WANDB_API_KEY
fi
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$(pwd)/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
unset TRANSFORMERS_CACHE
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$(pwd)/.cache/torch_extensions}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$(pwd)/.cache/triton}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ASSET_ROOT="${ASSET_ROOT:-$(pwd)/assets}"
COMMON_ASSET_ROOT="${COMMON_ASSET_ROOT:-$ASSET_ROOT/common}"
QWEN3_STAGE_ROOT="${QWEN3_STAGE_ROOT:-$ASSET_ROOT/qwen3_stage}"
RECAP118K_ROOT="${RECAP118K_ROOT:-$QWEN3_STAGE_ROOT/LLaVA-ReCap-118K}"
RECAP558K_ROOT="${RECAP558K_ROOT:-$QWEN3_STAGE_ROOT/LLaVA-ReCap-558K}"
SFT665K_ROOT="${SFT665K_ROOT:-$QWEN3_STAGE_ROOT/LLaVA-SFT-665K}"
IMAGENET_T2I_ROOT="${IMAGENET_T2I_ROOT:-$QWEN3_STAGE_ROOT/ImageNet1K-T2I-QwenVL-FLUX}"
MAGPIE_ROOT="${MAGPIE_ROOT:-$QWEN3_STAGE_ROOT/Magpie-Qwen2.5-Pro-1M-v0.1}"
QWEN3_MODEL_ROOT="${QWEN3_MODEL_ROOT:-$QWEN3_STAGE_ROOT/Qwen3-0.6B}"
RECAP118K_PATH="${RECAP118K_PATH:-$RECAP118K_ROOT/data}"
RECAP558K_PATH="${RECAP558K_PATH:-$RECAP558K_ROOT/data}"
SFT665K_PATH="${SFT665K_PATH:-$SFT665K_ROOT}"
IMAGENET_T2I_PATH="${IMAGENET_T2I_PATH:-$IMAGENET_T2I_ROOT}"
MAGPIE_PATH="${MAGPIE_PATH:-$MAGPIE_ROOT/data}"
AUTO_DOWNLOAD="${AUTO_DOWNLOAD:-0}"

mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$ASSET_ROOT" "$COMMON_ASSET_ROOT" "$QWEN3_STAGE_ROOT"

NUM_GPUS="${NUM_GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-29505}"
DATA_PATH="${DATA_PATH:-$QWEN3_STAGE_ROOT/data_qwen3_full.yaml}"
MAX_STEPS="${MAX_STEPS:-60000}"
LR="${LR:-5e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
TRAIN_PARTS="${TRAIN_PARTS:-mm_language_model}"
RUN_NAME="${RUN_NAME:-tar_qwen3_0.6B_pretrain}"
OUTPUT_DIR="${OUTPUT_DIR:-output_dir/${RUN_NAME}}"
VISION_MODEL="${VISION_MODEL:-$COMMON_ASSET_ROOT/ta_tok.pth}"
PREV_STAGE_CHECKPOINT="${PREV_STAGE_CHECKPOINT:-$QWEN3_MODEL_ROOT}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-8}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-True}"
TORCH_COMPILE="${TORCH_COMPILE:-False}"
TORCH_COMPILE_BACKEND="${TORCH_COMPILE_BACKEND:-inductor}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
LABEL_DEBUG_STEPS="${LABEL_DEBUG_STEPS:-0}"

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

download_repo_if_needed() {
  local repo_id="$1"
  local local_dir="$2"
  local repo_type="${3:-model}"
  if [ -d "$local_dir" ] && [ -n "$(find "$local_dir" -mindepth 1 -print -quit 2>/dev/null)" ]; then
    return 0
  fi
  mkdir -p "$local_dir"
  if [ "$repo_type" = "dataset" ]; then
    "$HF_CLI_BIN" download "$repo_id" --local-dir "$local_dir" --max-workers 64 --repo-type dataset
  else
    "$HF_CLI_BIN" download "$repo_id" --local-dir "$local_dir"
  fi
}

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
  download_repo_if_needed "lmms-lab/LLaVA-ReCap-118K" "$RECAP118K_ROOT" "dataset"
  download_repo_if_needed "lmms-lab/LLaVA-ReCap-558K" "$RECAP558K_ROOT" "dataset"
  download_repo_if_needed "csuhan/LLaVA-SFT-665K" "$SFT665K_ROOT" "dataset"
  download_repo_if_needed "csuhan/ImageNet1K-T2I-QwenVL-FLUX" "$IMAGENET_T2I_ROOT" "dataset"
  download_repo_if_needed "Magpie-Align/Magpie-Qwen2.5-Pro-1M-v0.1" "$MAGPIE_ROOT" "dataset"
  download_file_if_needed "csuhan/TA-Tok" "ta_tok.pth" "$COMMON_ASSET_ROOT"
  download_repo_if_needed "Qwen/Qwen3-0.6B" "$QWEN3_MODEL_ROOT" "model"
fi

require_path "$RECAP118K_PATH"
require_path "$RECAP558K_PATH"
require_path "$SFT665K_PATH"
require_path "$IMAGENET_T2I_PATH"
require_path "$MAGPIE_PATH"
require_path "$VISION_MODEL"
require_path "$PREV_STAGE_CHECKPOINT"

cat > "$DATA_PATH" <<EOF
datasets:
  - json_path:
      - $RECAP118K_PATH
      - $RECAP558K_PATH
      - $SFT665K_PATH
    name: parquet
    ratio: 2
  - json_path: $IMAGENET_T2I_PATH
    name: parquet
    ratio: 4
  - json_path: $MAGPIE_PATH
    name: parquet
    ratio: 1
EOF

echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "RUN_NAME: ${RUN_NAME}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "DATA_PATH: ${DATA_PATH}"
echo "AUTO_DOWNLOAD: ${AUTO_DOWNLOAD}"
echo "HF_ENDPOINT: ${HF_ENDPOINT}"
echo "HF_CLI_BIN: ${HF_CLI_BIN}"
echo "NUM_GPUS: ${NUM_GPUS}"
echo "PER_DEVICE_TRAIN_BATCH_SIZE: ${PER_DEVICE_TRAIN_BATCH_SIZE}"
echo "GRADIENT_ACCUMULATION_STEPS: ${GRADIENT_ACCUMULATION_STEPS}"
echo "WEIGHT_DECAY: ${WEIGHT_DECAY}"
echo "WARMUP_RATIO: ${WARMUP_RATIO}"
echo "DATALOADER_NUM_WORKERS: ${DATALOADER_NUM_WORKERS}"
echo "GRADIENT_CHECKPOINTING: ${GRADIENT_CHECKPOINTING}"
echo "TORCH_COMPILE: ${TORCH_COMPILE}"
echo "LABEL_DEBUG_STEPS: ${LABEL_DEBUG_STEPS}"

torchrun \
--nproc_per_node="${NUM_GPUS}" \
--nnodes=1 \
--node_rank=0 \
--master_addr=127.0.0.1 \
--master_port="${MASTER_PORT}" \
llava/train/train.py \
--deepspeed scripts/zero1.json \
--num_image_tokens 65536 \
--num_scale_tokens 3 \
--load_embeddings_from_vision True \
--model_name_or_path "${PREV_STAGE_CHECKPOINT}" \
--version "qwen_3" \
--data_path "${DATA_PATH}" \
--dataset_cls 'weighted_parquet' \
--dispatch_batches False \
--max_steps "${MAX_STEPS}" \
--mm_tunable_parts "${TRAIN_PARTS}" \
--vision_tower "${VISION_MODEL}" \
--mm_vision_select_layer -2 \
--mm_use_im_start_end True \
--group_by_modality_length True \
--image_aspect_ratio square \
--mm_patch_merge_type flat \
--bf16 True \
--run_name "${RUN_NAME}" \
--output_dir "${OUTPUT_DIR}" \
--num_train_epochs 1 \
--per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
--per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
--gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
--save_strategy "steps" \
--save_steps 10000 \
--save_total_limit 1 \
--learning_rate "${LR}" \
--weight_decay "${WEIGHT_DECAY}" \
--warmup_ratio "${WARMUP_RATIO}" \
--lr_scheduler_type "cosine" \
--logging_steps 10 \
--tf32 True \
--model_max_length 2048 \
--gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
--dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
--dataloader_prefetch_factor "${DATALOADER_PREFETCH_FACTOR}" \
--lazy_preprocess True \
--report_to wandb \
--torch_compile "${TORCH_COMPILE}" \
--torch_compile_backend "${TORCH_COMPILE_BACKEND}" \
--dataloader_drop_last True \
--attn_implementation "${ATTN_IMPLEMENTATION}" \
--label_debug_steps "${LABEL_DEBUG_STEPS}"
