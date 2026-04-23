#!/bin/bash
# ==============================================================================
# Example SLURM launcher for PETRG-3D inference. See ``srun.sh`` for
# per-cluster customization notes.
# ==============================================================================

#SBATCH -J PETRG-3D-infer
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -o ./results/petrg3d_infer_%j.log
#SBATCH -t 1-00:00:00
#SBATCH --gres=gpu:1

source activate petrg

cd "$(dirname "$0")"

bash test.sh qwen3_template_autopet
