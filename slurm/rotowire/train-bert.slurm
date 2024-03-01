#!/bin/bash
#SBATCH --cpus-per-gpu 30
#SBATCH --gpus=2
#SBATCH -e slurm/output/slurm-%j.err
#SBATCH -o slurm/output/slurm-%j.out

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate eleet

MODEL="bert-base-uncased"

# FINETUNE_DATASET="rotowire" FINETUNE_MODEL="eleet" MASTER_PORT=29504 python -m debugpy --listen 5681 --wait-for-client scripts/finetune.py $MODEL \
FINETUNE_DATASET="rotowire" FINETUNE_MODEL="eleet" MASTER_PORT=29504 python scripts/finetune.py $MODEL \
    --log-level debug \
    --max-steps 8000 \
    --debug-fraction 0.01 \
    --per-device-train-batch-size 12 \
    --train-batch-size 48 \
    --learning-rate 1e-4 \
    --learning-rate-schedule linear \
    --num-eval 5 \
    --logging-steps 10 \
    --dataloader-num-workers 30 \
    --eval-split-limit 5000 \
    --finetune-split-sizes 16384 4096 1024 256 64 16 4
