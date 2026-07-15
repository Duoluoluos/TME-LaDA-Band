#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 /path/to/vocal.wav \"style description\" [output_dir]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VOCAL_FP="$1"
TEXT="$2"
OUTPUT_DIR="${3:-$ROOT_DIR/outputs/demo}"

: "${LADA_PRETRAINED_ROOT:?Set LADA_PRETRAINED_ROOT to the auxiliary-assets directory.}"
CHECKPOINT_FP="${LADA_CHECKPOINT_PATH:-${LADA_PRETRAINED_ROOT%/}/../checkpoints/lada_band_lm1B_total3B.ckpt}"
CONFIG_FP="${LADA_INFER_CONFIG:-$ROOT_DIR/codes/lada_band/conf/infer_1B.yaml}"

if [[ ! -f "$VOCAL_FP" ]]; then
  echo "Vocal input not found: $VOCAL_FP" >&2
  exit 1
fi
if [[ ! -f "$CHECKPOINT_FP" ]]; then
  echo "Checkpoint not found: $CHECKPOINT_FP" >&2
  exit 1
fi

# Set LADA_OFFLINE=1 after the Hugging Face assets have been cached locally.
if [[ "${LADA_OFFLINE:-0}" == "1" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

export LADA_CODE_ROOT="$ROOT_DIR/codes"
cd "$ROOT_DIR/codes"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python infer.py \
  --model_fp "$CHECKPOINT_FP" \
  --config "$CONFIG_FP" \
  --output_dir "$OUTPUT_DIR" \
  --seg_time "${LADA_SEGMENT_START:-0}" "${LADA_SEGMENT_END:-30}" \
  --vocal_fps "$VOCAL_FP" \
  --texts "$TEXT" \
  --sampling_steps "${LADA_SAMPLING_STEPS:-60}" \
  --save_format "${LADA_SAVE_FORMAT:-wav}" \
  --acc_only
