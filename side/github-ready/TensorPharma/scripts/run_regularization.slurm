#!/bin/bash
#SBATCH --job-name=emerge-regularize
#SBATCH --account=mpensky
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=results/regularization-%j.log

set -euo pipefail
module load python/python-3.11.4-gcc-12.2.0
cd "${SLURM_SUBMIT_DIR:?Submit this job from the eMerge project directory}"
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
for strength in 100 1000 10000; do
    .venv/bin/python scripts/challenge6_joint.py --validation-only \
        --compound-regularization "$strength" --output-regularization "$strength" \
        --output "results/regularization/$strength"
done
.venv/bin/python scripts/summarize_regularization.py
