#!/bin/bash
set -euo pipefail

ROOT_DIR="/home/matin/convert_dir/Expedition44"
TMP_DIR="/home/matin/convert_dir/Expedition44/tmp"

export MAMBA_ROOT_PREFIX=/home/matin/micromamba
export MAMBA_EXE=/home/matin/.local/bin/micromamba
eval "$($MAMBA_EXE shell hook --shell bash)"
micromamba activate dev_eval_acc_flash

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin${PATH:+:$PATH}"
export CPATH="${CUDA_HOME}/include${CPATH:+:$CPATH}"

DCP_DIR="/mnt/b300-runner/checkpoints-v2"
STEP="589999"
CONVERT_SCRIPT="$ROOT_DIR/src/evals/convert_dcp_to_pt.py"
HF_CONVERT_SCRIPT="$ROOT_DIR/src/evals/hf_model/convert_checkpoint.py"

mkdir -p "$TMP_DIR"

# ── 1. DCP → .pt ─────────────────────────────────────────────────────────────
PT_FILE="$TMP_DIR/ckpt-step-${STEP}.pt"
echo "=== [1/2] Converting DCP → $PT_FILE ==="
python "$CONVERT_SCRIPT" \
    "$DCP_DIR" "$TMP_DIR" \
    --steps "$STEP" \
    --name "ckpt-step"

if [[ ! -f "$PT_FILE" ]]; then
    echo "ERROR: DCP conversion produced no file at $PT_FILE"
    exit 1
fi

# ── 2. .pt → HuggingFace model dir ───────────────────────────────────────────
HF_DIR="$TMP_DIR/hf-step-${STEP}"
echo "=== [2/2] Converting .pt → HF dir at $HF_DIR ==="
python "$HF_CONVERT_SCRIPT" \
    "$PT_FILE" "$HF_DIR" \
    --attn_backend pytorch \
    --nvfp4

if [[ ! -f "$HF_DIR/config.json" ]]; then
    echo "ERROR: HF conversion failed (no config.json)"
    exit 1
fi

rm -f "$PT_FILE"

echo ""
echo "=== Conversion complete ==="
echo "HF model dir: $HF_DIR"
