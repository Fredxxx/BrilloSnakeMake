#!/bin/bash
#SBATCH -A prevedel
#SBATCH -p gpu-el8
#SBATCH -C gpu=A100
#SBATCH -c 224
#SBATCH -G 2
#SBATCH --mem-per-gpu 128789
#SBATCH -t 24:00:00
#SBATCH -o my-log_00_20251128.log
#SBATCH -e my-log_00_20251128.log

#module load Python/3.12.3-GCCcore-13.3.0

#source ~/projectsHPC/brillo/venv312brillo/bin/activate

python -u scatLoop_v06_cluster.py
