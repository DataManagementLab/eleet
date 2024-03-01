#!/bin/bash
#SBATCH --cpus-per-gpu 30
#SBATCH --gpus=2
#SBATCH -e slurm/output/slurm-%j.err
#SBATCH -o slurm/output/slurm-%j.out

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate eleet

cd text_to_table/
bash scripts/rotowire/test_constraint.sh ../datasets/rotowire/data/ /home/murban/multimodal-database/models/rotowire/text_to_table/2022-12-19_12-20-37/checkpoints.3395/checkpoint_average_best-3.pt
