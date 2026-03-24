#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# setup_dev_eval_acc_flash.sh — reproducible setup for the accelerate-based
# eval env with Flash Attention 2
#
# Creates (or recreates) the `dev_eval_acc_flash` micromamba environment with
# all the packages needed for the accelerate + lm_eval CLI pipeline using
# flash_attention_2 as the transformer attention backend:
#   convert_dcp_to_pt.py → convert_checkpoint.py → accelerate launch lm_eval
#   → upload_wandb.py
#
# Prerequisites:
#   - micromamba installed at ~/.local/bin/micromamba
#   - CUDA module system (module load cuda/13.1.0)
#
# Usage:
#   bash setup_dev_eval_acc_flash.sh              # full setup from scratch
#   bash setup_dev_eval_acc_flash.sh --skip-create  # skip env creation, just install packages
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── Micromamba & CUDA ─────────────────────────────────────────────────────────
export MAMBA_ROOT_PREFIX=~/micromamba
export MAMBA_EXE=~/.local/bin/micromamba
eval "$($MAMBA_EXE shell hook --shell bash)"

# module load cuda/13.1.0
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin${PATH:+:$PATH}"
export CPATH="${CUDA_HOME}/include${CPATH:+:$CPATH}"
export LD_LIBRARY_PATH="/mnt/nfs/clustersw/Debian/bookworm/gcc/14.1.0/lib64:$ROOT_DIR/nvml_wrapper${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

SKIP_CREATE=false
if [[ "${1:-}" == "--skip-create" ]]; then
    SKIP_CREATE=true
fi

# ── Create environment ────────────────────────────────────────────────────────
ENV_NAME="dev_eval_acc_flash"
PYTHON_VERSION="3.12"

if [[ "$SKIP_CREATE" == false ]]; then
    echo "Removing old environment (if any) ..."
    set +u; micromamba deactivate 2>/dev/null || true; set -u
    micromamba env remove -n "$ENV_NAME" -y 2>/dev/null || true
    rm -rf "${MAMBA_ROOT_PREFIX}/envs/${ENV_NAME}"

    echo "Creating micromamba environment: ${ENV_NAME} (python=${PYTHON_VERSION})"
    micromamba create -n "$ENV_NAME" python="$PYTHON_VERSION" -c conda-forge -y
fi

set +u; micromamba activate "$ENV_NAME"; set -u
echo "Activated: $(python --version)  at  $(which python)"

# ── Core conda-forge deps ────────────────────────────────────────────────────
echo ""
echo "Installing scipy, matplotlib, datasets, safetensors, transformers, accelerate ..."
set +u
micromamba install -c conda-forge -y \
    scipy matplotlib datasets safetensors transformers accelerate
set -u

# ── PyTorch (CUDA 12.8+) ─────────────────────────────────────────────────────
echo ""
echo "Installing PyTorch + torchvision (cu130) ..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
# ── Flash-attn build deps ─────────────────────────────────────────────────────
echo ""
echo "Installing packaging, psutil, ninja (needed by flash-attn build) ..."
pip install packaging psutil ninja
pip install cuda-toolkit

# ── Flash Attention 2 ─────────────────────────────────────────────────────────
echo ""
echo "Installing flash-attn (Flash Attention 2) ..."
pip install flash-attn --no-build-isolation

# ── PyPI-only packages ───────────────────────────────────────────────────────
echo ""
echo "Installing PyPI-only packages (tokenmonster, lm_eval, wandb) ..."
pip install tokenmonster "lm_eval[hf]" wandb
# ── Sanity check ──────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Sanity checks"
echo "════════════════════════════════════════════════════════════"
python -c "import torch;         print(f'torch        {torch.__version__}  cuda={torch.cuda.is_available()}')"
python -c "import lm_eval;       print(f'lm_eval      {lm_eval.__version__}')"
python -c "import tokenmonster;  print('tokenmonster  OK')"
python -c "import safetensors;   print(f'safetensors  {safetensors.__version__}')"
python -c "import transformers;  print(f'transformers {transformers.__version__}')"
python -c "import accelerate;    print(f'accelerate   {accelerate.__version__}')"
python -c "import wandb;         print(f'wandb        {wandb.__version__}')"
python -c "import triton;        print(f'triton       {triton.__version__}')"
python -c "import flash_attn;    print(f'flash_attn   {flash_attn.__version__}')"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Setup complete."
echo ""
echo "  Activate with:"
echo "    export MAMBA_ROOT_PREFIX=~/micromamba"
echo "    export MAMBA_EXE=~/.local/bin/micromamba"
echo '    eval "$($MAMBA_EXE shell hook --shell bash)"'
echo "    module load cuda/13.1.0"
echo "    micromamba activate dev_eval_acc_flash"
echo '    export CUDA_HOME=/usr/local/cuda   # needed by Triton (fake_quartet)'
echo ""
echo "  Hide broken GPU 9 (this machine):"
echo '    export LD_LIBRARY_PATH="/mnt/nfs/clustersw/Debian/bookworm/gcc/14.1.0/lib64:$ROOT_DIR/nvml_wrapper${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"'
echo "    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
echo ""
echo "  Run accelerate eval pipeline:"
echo "    bash src/evals/auto_eval_4b_acc.sh"
echo "════════════════════════════════════════════════════════════"
