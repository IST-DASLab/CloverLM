# CloverLM Evaluation Scripts

This directory contains scripts for evaluating [CloverLM](https://huggingface.co/daslab-testing/CloverLM) checkpoints during training using the [lm-eval harness](https://github.com/EleutherAI/lm-evaluation-harness) with Accelerate.

## Quick Start (Recommended)

For a ready-to-use evaluation setup with `uv` and locked dependencies, see the
[lm_eval README on HuggingFace](https://huggingface.co/daslab-testing/CloverLM/blob/main/lm_eval/README.md).
That setup requires only three commands to get running and is the easiest way to
reproduce the published results.

## Development Evaluation Pipeline

The scripts below are used internally for continuous evaluation of checkpoints
as they are produced during training.

### `setup_dev_eval_acc_flash.sh`

Creates (or recreates) the `dev_eval_acc_flash` micromamba environment with all
packages needed for the accelerate + lm_eval pipeline using Flash Attention 2:

```
convert_dcp_to_pt.py → convert_checkpoint.py → accelerate launch lm_eval → upload_wandb.py
```

**Prerequisites:** micromamba at `~/.local/bin/micromamba`, CUDA toolkit available.

```bash
bash setup_dev_eval_acc_flash.sh
```

### `auto_eval_4b_acc.sh`

Polling loop that watches a DCP checkpoint directory, and for each new step:

1. Converts the DCP checkpoint to a single `.pt` file
2. Converts the `.pt` file to a HuggingFace model directory
3. Runs zero-shot evaluation via `accelerate launch` with lm_eval
4. Uploads results to Weights & Biases

Configure the checkpoint directory, tasks, GPU count, and W&B project by editing
the variables at the top of the script. Runs indefinitely, polling every
`POLL_INTERVAL` seconds for new checkpoints.

```bash
bash auto_eval_4b_acc.sh
```
