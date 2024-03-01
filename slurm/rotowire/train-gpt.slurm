#!/bin/bash
#SBATCH --cpus-per-task 3
#SBATCH --gpus=0
#SBATCH -e slurm/output/slurm-%j.err
#SBATCH -o slurm/output/slurm-%j.out

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate eleet

# FINETUNE_DATASET="rotowire" FINETUNE_MODEL="gpt-3.5-turbo-1106" python -m debugpy --listen 5687 --wait-for-client scripts/finetune.py \
FINETUNE_DATASET="rotowire" FINETUNE_MODEL="gpt-3.5-turbo-1106" python scripts/finetune.py \
    --finetune-split-sizes 1024 256 64 16 4  # 4096
