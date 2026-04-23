#!/bin/bash
# ==============================================================================
# Example SLURM launcher for PETRG-3D training.
#
# This is a reference template -- please adjust ``-p``, ``-w``, ``--gres``,
# the conda/venv activation command and the output directory to match your own
# cluster before using it.
# ==============================================================================

#SBATCH -J PETRG-3D-train
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o ./outputs/petrg3d_train_%j.log
#SBATCH -t 10-00:00:00
#SBATCH --gres=gpu:2

# Activate your Python environment.
source activate petrg

# Run from the scripts/ directory so that the config sourcing uses the correct
# relative path (../configs/train/<name>.sh).
cd "$(dirname "$0")"

bash train.sh qwen3_joint_template
