#!/bin/bash
#SBATCH --cpus-per-gpu 15
#SBATCH --gpus=2
#SBATCH -e slurm/output/slurm-%j.err
#SBATCH -o slurm/output/slurm-%j.out

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate eleet

MODEL="models/pretrained/current/pytorch_model.bin"
DATASET="/mnt/labstore/murban/preprocessed_data_ready/preprocessed_trex-wikidata_v9*/data.h5"

# python -m debugpy --listen 5681 --wait-for-client scripts/finetune_hp_tuning.py \
python scripts/finetune_hp_tuning.py \
    $MODEL finetune-hpo \
    --study-storage-url postgresql://postgres:FfoORxeeYwl5xeH6ziMh@10.0.2.16:65432/optuna \
    --log-level debug \
    --min-steps 150 \
    --max-steps 1350 \
    --per-device-train-batch-size 6 \
    --logging-steps 10 \
    --dataloader-num-workers 30 \
    --dataset $DATASET \
    --eval-split-limit 5000 \
    --finetune-split-size 128 \
    --force-restart
