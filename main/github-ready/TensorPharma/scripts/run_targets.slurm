#!/bin/bash
#SBATCH --job-name=emerge-targets
#SBATCH --account=mpensky
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=results/targets-%j.log

set -euo pipefail
module load python/python-3.11.4-gcc-12.2.0
cd "${SLURM_SUBMIT_DIR:?Submit this job from the eMerge project directory}"
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/match_chembl.py
.venv/bin/python scripts/challenge6_targets.py
