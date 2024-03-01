#!/bin/bash
#SBATCH --cpus-per-gpu 10
#SBATCH --gpus=1
#SBATCH -e slurm/output/slurm-%j.err
#SBATCH -o slurm/output/slurm-%j.out

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate llamaft

# FINETUNE_DATASET="corona" FINETUNE_MODEL="llama" python -m debugpy --listen 5687 --wait-for-client scripts/finetune.py \
FINETUNE_DATASET="corona" FINETUNE_MODEL="llama" python scripts/finetune.py \
    --finetune-split-sizes 64 \
    --learning-rate 4e-4 \
    --per-device-train-batch-size 4 \
    --gradient-accumulation-steps 32 \
    --quantization-num-bits 4 \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-dropout 0.05 \
    --max-steps 250

#4096 1024 256 64 16 4 \

