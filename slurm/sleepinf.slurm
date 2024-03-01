#!/bin/bash
#SBATCH --cpus-per-task 1
#SBATCH --gpus=0
#SBATCH -e slurm/output/slurm-%j.err
#SBATCH -o slurm/output/slurm-%j.out

sleep infinity
