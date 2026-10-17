#!/bin/bash
#SBATCH --job-name=emerge-baseline
#SBATCH --account=mpensky
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=results/baseline-%j.log

set -euo pipefail
module load python/python-3.11.4-gcc-12.2.0
cd "${SLURM_SUBMIT_DIR:?Submit this job from the eMerge project directory}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
mkdir -p results/baseline
.venv/bin/python -m pip freeze > results/baseline/environment.txt
.venv/bin/python scripts/challenge6_features.py
.venv/bin/python scripts/challenge6_baseline.py
