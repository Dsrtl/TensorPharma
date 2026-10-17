#!/bin/bash
#SBATCH --job-name=emerge-chembl
#SBATCH --account=mpensky
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=results/chembl-extract-%j.log

set -euo pipefail
module load python/python-3.11.4-gcc-12.2.0
module load apptainer/apptainer-1.3.3-go-1.22.5
cd "${SLURM_SUBMIT_DIR:?Submit this job from the eMerge project directory}"
export GOMAXPROCS=2
export LC_ALL=C
export LANG=C
export LC_CTYPE=C
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/extract_chembl.py
