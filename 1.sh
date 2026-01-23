#!/bin/bash
#SBATCH -A p_zhu                     # Project name
#SBATCH --partition=gpu              # GPU partition
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --job-name=pysgg_train       # Job name
#SBATCH --time=03:00:00              # Time limit
#SBATCH --mail-type=ALL              # Email notifications
#SBATCH --mail-user=drgck8@inf.elte.hu
#SBATCH --mem-per-gpu=50000
#SBATCH --nodes=1

# Set project directory
PROJECT_DIR=/home/p_zhuzy/p_zhu/PySGG-main

# Option 1: Using uv (recommended)
cd $PROJECT_DIR
uv run python -u train.py

# Option 2: Using venv directly (alternative)
# source $PROJECT_DIR/.venv/bin/activate
# python -u train.py
