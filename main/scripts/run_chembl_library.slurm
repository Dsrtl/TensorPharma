#!/bin/bash
#SBATCH --job-name=emerge-target-library
#SBATCH --account=mpensky
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=results/chembl-library-%j.log

set -euo pipefail
module load python/python-3.11.4-gcc-12.2.0
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TMPDIR="${TP_WORK:?}/tmp-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
if [[ ! -f "$TP_WORK/library/manifest.json" ]]; then
    .venv/bin/python scripts/chembl_library.py build \
        --source "$TP_WORK/input/drug_targets.csv.gz" \
        --frozen "$TP_WORK/frozen" --output "$TP_WORK/library" \
        --workers "$SLURM_CPUS_PER_TASK"
fi
.venv/bin/python scripts/chembl_library.py evaluate \
    --library "$TP_WORK/library" --output "$TP_WORK/metrics.json"
mkdir -p results/chembl_library
cp "$TP_WORK/metrics.json" "$TP_WORK/library/manifest.json" results/chembl_library/
.venv/bin/python -m pip freeze > results/chembl_library/environment.txt
