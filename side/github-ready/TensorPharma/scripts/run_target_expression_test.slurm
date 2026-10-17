#!/bin/bash
#SBATCH --job-name=emerge-target-expression
#SBATCH --account=mpensky
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=results/target-expression-%j.log

set -euo pipefail
module load python/python-3.11.4-gcc-12.2.0
cd "${SLURM_SUBMIT_DIR:?Submit from the eMerge project directory}"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export BLIS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1
library="$(cat results/chembl-library-workdir.txt)/library"
test -f "$library/manifest.json"
test -f results/rank_benchmark_cancer_only/benchmark_report.json
test -f results/rank_benchmark_cancer_only/input/gene_expression.csv.gz
test -f results/chembl_library/metrics.json
.venv/bin/python -c 'import lightgbm, narwhals, numpy, pandas, scipy, rdkit; print("Existing environment available")'
.venv/bin/python -m pip check
arguments=()
output=results/target_expression_test
if [[ -n "${TP_EXCLUDE_GENES:-}" ]]; then
    test -f "$TP_EXCLUDE_GENES"
    output=results/target_expression_filtered_test
    arguments+=(--exclude-genes "$TP_EXCLUDE_GENES")
fi
if [[ -f "$output/report.json" ]]; then
    printf '%s\n' 'Completed experiment exists; preserve it rather than overwrite.' >&2
    exit 1
fi
mkdir -p "$output"
.venv/bin/python -m pip freeze > "$output/environment.txt"
.venv/bin/python scripts/target_expression_test.py \
    --library "$library" --selection results/chembl_library/metrics.json \
    --output "$output" --threads "$SLURM_CPUS_PER_TASK" "${arguments[@]}"
