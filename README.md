# CloverLM

## Description
PyTorch codebase used for the training and evaluation of [CloverLM](https://github.com/IST-DASLab/CloverLM/blob/main/REPORT.md). Training code is heavily modified [ant](https://github.com/gvlassis/ant), with NVFP4 kernels from [Quartet-II](https://github.com/IST-DASLab/Quartet-II), and evaluation code by Matin Ansaripour(@matinansaripour) and Andrei Panferov(@BlackSamorez).

## Getting started

### Training
1) Clone the repo:

```bash
git clone https://github.com/IST-DASLab/CloverLM.git
```

2) Use uv to install dependencies from `pyproject.toml`

```bash
uv sync
```

3) Install [FlashAttention](https://github.com/Dao-AILab/flash-attention)

4) [Download](https://drive.google.com/drive/folders/104QlBZRcPAF3PQV2W1GuZIDTdV4dOIZ9?usp=sharing) pretokenized ClimbMix (305B tokens/610GB)

5) Train CloverLM

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 ./src/train.py  4b-28h-29d-cm310b-v3 --opt adam --micro_batch_size 32 --train_batches 590000  --k_input 3e-3 --momentum 0.9 --beta2 0.95 --eps 1e-6 --quartet true --info false --extra_freq 200 --backend flash2 --dataset=climbmix10m --num_blocks=29 --heads=28 --ratio=4 --checkpoint_freq 20000 --dataset_seed=654356 --dataset_path=climbmix --wandb_kwargs='{"project": "expedition44"}' --warmup 2000 --cooldown 20000 --model_stats_freq=5000
```

### Evaluation
See [./src/evals/README.md](src/evals/README.md)
