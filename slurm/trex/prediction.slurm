#!/bin/bash
#SBATCH --cpus-per-gpu 30
#SBATCH --gpus=2
#SBATCH -e slurm/output/slurm-%j.err
#SBATCH -o slurm/output/slurm-%j.out

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate eleet

# python -m debugpy --listen 5681 --wait-for-client scripts/trex/prediction.py \
python scripts/trex/prediction.py \
    --experiment quality \
    --models our=~/multimodal-database/models/trex/ours/finetuned/union/2023-02-24_14-36-34_0 \
    --split test
    # t2t=~/multimodal-database/models/trex/text_to_table/current \
