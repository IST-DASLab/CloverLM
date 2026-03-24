
set -euo pipefail
cd "$(dirname "$0")"
ROOT_DIR="/home/matin/Expedition44"

# ── Environment ───────────────────────────────────────────────────────────────
export MAMBA_ROOT_PREFIX=~/micromamba
export MAMBA_EXE=~/.local/bin/micromamba
eval "$($MAMBA_EXE shell hook --shell bash)"
micromamba activate dev_eval_acc_flash

# export LD_LIBRARY_PATH="/mnt/nfs/clustersw/Debian/bookworm/gcc/14.1.0/lib64:$ROOT_DIR/nvml_wrapper${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin${PATH:+:$PATH}"
export CPATH="${CUDA_HOME}/include${CPATH:+:$CPATH}"
export TRITON_PTXAS_BLACKWELL_PATH="${CUDA_HOME}/bin/ptxas"
export CUDA_VISIBLE_DEVICES=0,1

# ── Configuration ─────────────────────────────────────────────────────────────
DCP_DIR="/mnt/b300-runner/checkpoints/"
TMP_DIR="/home/matin/Expedition44/tmp/"
PROCESSED_FILE="$ROOT_DIR/.auto_eval_acc_processed_steps"
EVAL_MODE="fast"

NGPUS="2"
if [ "$EVAL_MODE" == "fast" ]; then
    TASKS="arc_easy_mi,arc_challenge_mi,hellaswag,piqa,wikitext,lambada_openai_norm,nq_open,coqa"
else
    TASKS="arc_easy_mi,arc_challenge_mi,hellaswag,piqa,wikitext,lambada_openai_norm,triviaqa,nq_open,coqa,drop"
fi
WANDB_PROJECT="expedition44_test"
WANDB_RUN_NAME="4b-28h-29d-cm310b"
WANDB_RUN_ID="eval-4b-28h-29d-cm310b"

QUARTET_II_IMPL="pseudoquant"
# QUARTET_II_IMPL="quartet2" # Eval with real kernels. Requires installing the kernels from https://github.com/IST-DASLab/Quartet-II
ATTN_BACKEND="pytorch"
POLL_INTERVAL=300  # seconds between checks

CONVERT_SCRIPT="$ROOT_DIR/src/evals/convert_dcp_to_pt.py"
HF_CONVERT_SCRIPT="$ROOT_DIR/src/evals/hf_model/convert_checkpoint.py"
UPLOAD_SCRIPT="$ROOT_DIR/src/evals/upload_wandb.py"
CLOVERLM_LM="$ROOT_DIR/src/evals/hf_model/cloverlm_lm.py"
INCLUDE_PATH="$ROOT_DIR/src/evals/lm_tasks_edit"

# ── Helpers ───────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

discover_steps() {
    local dir="$1"
    if [[ ! -d "$dir" ]]; then
        return
    fi
    for entry in "$dir"/*/; do
        step="$(basename "$entry")"
        if [[ "$step" =~ ^[0-9]+$ ]] && [[ -f "$entry/.metadata" ]]; then
            echo "$step"
        fi
    done | sort -n
}

is_processed() {
    local step="$1"
    touch "$PROCESSED_FILE"
    grep -qx "$step" "$PROCESSED_FILE"
}

mark_processed() {
    local step="$1"
    echo "$step" >> "$PROCESSED_FILE"
}

find_results_json() {
    # lm_eval writes results to <output_path>/<model_name>/results_*.json
    # Find the most recent results file under the given directory.
    local dir="$1"
    find "$dir" -name 'results_*.json' -type f -printf '%T@ %p\n' \
        | sort -rn | head -1 | cut -d' ' -f2-
}

# ── Main loop ─────────────────────────────────────────────────────────────────
log "Auto-eval loop started (accelerate pipeline)"
log "  DCP_DIR        : $DCP_DIR"
log "  TMP_DIR        : $TMP_DIR"
log "  PROCESSED_FILE : $PROCESSED_FILE"
log "  POLL_INTERVAL  : ${POLL_INTERVAL}s"
log "  WANDB_PROJECT  : $WANDB_PROJECT"
log "  WANDB_RUN_ID   : $WANDB_RUN_ID"
log "  NGPUS          : $NGPUS"
log ""

while true; do
    log "Scanning for new checkpoints in $DCP_DIR ..."

    if [[ ! -d "$DCP_DIR" ]]; then
        log "  Checkpoint dir does not exist yet, waiting ..."
        sleep "$POLL_INTERVAL"
        continue
    fi

    NEW_STEPS=()
    while IFS= read -r step; do
        if ! is_processed "$step"; then
            NEW_STEPS+=("$step")
        fi
    done < <(discover_steps "$DCP_DIR")

    if [[ ${#NEW_STEPS[@]} -eq 0 ]]; then
        log "  No new checkpoints found. Sleeping ${POLL_INTERVAL}s ..."
        sleep "$POLL_INTERVAL"
        continue
    fi

    log "  Found ${#NEW_STEPS[@]} new step(s): ${NEW_STEPS[*]}"

    for STEP in "${NEW_STEPS[@]}"; do
        STEP_DIR="$DCP_DIR/$STEP"
        log "────────────────────────────────────────────────────────"
        log "  Processing step $STEP"
        log "────────────────────────────────────────────────────────"

        # ── 1. Convert DCP → .pt ─────────────────────────────────
        mkdir -p "$TMP_DIR"
        PT_FILE="$TMP_DIR/ckpt-step-${STEP}.pt"

        log "  [1/4] Converting DCP → $PT_FILE ..."
        python "$CONVERT_SCRIPT" \
            "$DCP_DIR" "$TMP_DIR" \
            --steps "$STEP" \
            --name "ckpt-step"

        CONVERTED="$TMP_DIR/ckpt-step-${STEP}.pt"
        if [[ ! -f "$CONVERTED" ]]; then
            log "  ERROR: DCP conversion produced no file at $CONVERTED, skipping"
            continue
        fi

        # ── 2. Convert .pt → HuggingFace model dir ──────────────
        HF_DIR="$TMP_DIR/hf-step-${STEP}"
        log "  [2/4] Converting .pt → HF dir at $HF_DIR ..."
        python "$HF_CONVERT_SCRIPT" \
            "$CONVERTED" "$HF_DIR" \
            --quartet_2_impl "$QUARTET_II_IMPL" \
            --attn_backend "$ATTN_BACKEND"

        if [[ ! -f "$HF_DIR/config.json" ]]; then
            log "  ERROR: HF conversion failed (no config.json), skipping"
            rm -f "$CONVERTED"
            continue
        fi

        # Done with the .pt file
        rm -f "$CONVERTED"

        # ── 3. Run eval with accelerate + lm_eval CLI ───────────
        EVAL_OUTPUT="$TMP_DIR/eval-output-step-${STEP}"
        log "  [3/4] Running accelerate launch lm_eval ($NGPUS GPUs) ..."

        accelerate launch --num_processes "$NGPUS" \
            "$CLOVERLM_LM" \
            --model cloverlm \
            --model_args "pretrained=$HF_DIR,dtype=bfloat16" \
            --tasks "$TASKS" \
            --num_fewshot 0 \
            --include_path "$INCLUDE_PATH" \
            --trust_remote_code \
            --confirm_run_unsafe_code \
            --batch_size auto \
            --output_path "$EVAL_OUTPUT" \
            --log_samples \
            --write_out

        RESULTS_JSON="$(find_results_json "$EVAL_OUTPUT")"
        if [[ -z "$RESULTS_JSON" ]]; then
            log "  ERROR: No results JSON found in $EVAL_OUTPUT, skipping wandb upload"
            rm -rf "$HF_DIR" "$EVAL_OUTPUT"
            mark_processed "$STEP"
            continue
        fi

        log "  Results file: $RESULTS_JSON"

        # ── 4. Upload results to wandb ───────────────────────────
        log "  [4/4] Uploading results to wandb ..."
        if ! python "$UPLOAD_SCRIPT" "$RESULTS_JSON" \
            --wandb_project "$WANDB_PROJECT" \
            --wandb_run_name "$WANDB_RUN_NAME" \
            --wandb_run_id "$WANDB_RUN_ID" \
            --wandb_step "$STEP" \
            --checkpoint "$DCP_DIR/$STEP"; then
            log "  WARNING: wandb upload failed for step $STEP, continuing anyway"
        fi

        # ── Cleanup tmp ──────────────────────────────────────────
        log "  Cleaning up temp files ..."
        rm -rf "$HF_DIR" "$EVAL_OUTPUT"

        # ── Mark as done ─────────────────────────────────────────
        mark_processed "$STEP"
        log "  Step $STEP complete."
        log ""
    done

    log "All new steps processed. Sleeping ${POLL_INTERVAL}s ..."
    sleep "$POLL_INTERVAL"
done
