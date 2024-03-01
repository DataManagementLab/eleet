#!/bin/bash
#SBATCH --cpus-per-gpu 30
#SBATCH --gpus=2
#SBATCH -e slurm/output/slurm-%j.err
#SBATCH -o slurm/output/slurm-%j.out

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate eleet

# python -m debugpy --listen 5681 --wait-for-client scripts/rotowire/train_ours.py $MODEL \
FINETUNE_DATASET="corona" FINETUNE_MODEL="t2t" python scripts/finetune.py \
    --finetune-split-sizes 256 64 16 4
