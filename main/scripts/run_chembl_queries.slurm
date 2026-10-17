#!/bin/bash
#SBATCH --job-name=emerge-target-queries
#SBATCH --account=mpensky
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=results/chembl-queries-%A_%a.log

set -euo pipefail
module load python/python-3.11.4-gcc-12.2.0
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
export TMPDIR="${TP_WORK:?}/tmp-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
.venv/bin/python scripts/chembl_library.py score \
    --library "$TP_WORK/library" --selection "$TP_WORK/metrics.json" \
    --queries "$TP_WORK/input/queries.csv" \
    --shard "$SLURM_ARRAY_TASK_ID" --shards "${TP_SHARDS:?}" \
    --output "$TP_WORK/predictions/shard-${SLURM_ARRAY_TASK_ID}.csv"
