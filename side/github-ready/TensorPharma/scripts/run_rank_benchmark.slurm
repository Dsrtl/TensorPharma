#!/bin/bash
#SBATCH --job-name=emerge-rank-benchmark
#SBATCH --account=mpensky
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=results/rank-benchmark-%j.log

set -euo pipefail
module load python/python-3.11.4-gcc-12.2.0
cd "${SLURM_SUBMIT_DIR:?Submit this job from the eMerge project directory}"
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
if [[ -f results/rank_benchmark/benchmark_report.json ]]; then
    printf '%s\n' 'Completed benchmark exists; choose a new output directory for another experiment.' >&2
    exit 1
fi
.venv/bin/python -m pip install --no-deps "lightgbm>=4,<5" narwhals
.venv/bin/python -m pip check
mkdir -p results/rank_benchmark
.venv/bin/python -m pip freeze > results/rank_benchmark/environment.txt
.venv/bin/python scripts/rank_benchmark.py --threads 8
