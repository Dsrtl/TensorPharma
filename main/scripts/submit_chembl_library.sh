#!/bin/bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
cd "$(dirname "$0")/.."
module load python/python-3.11.4-gcc-12.2.0
work_base="${TP_WORK_BASE:-${SCRATCH:-}}"
if [[ -z "$work_base" ]]; then
    printf '%s\n' 'Set SCRATCH to assigned scratch storage, or TP_WORK_BASE to an existing project work directory (counts against its quota).' >&2
    exit 1
fi
test -d "$work_base"
work_base=$(cd "$work_base" && pwd -P)
.venv/bin/python -c 'import rdkit, pandas, numpy, scipy; print("Using existing frozen environment")'
export TP_WORK="$work_base/emerge-targets-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$TP_WORK/input" "$TP_WORK/frozen" results
cp TensorPharma-main/data/processed/drug_targets.csv.gz "$TP_WORK/input/"
cp data/features/compound_split.csv data/features/feature_report.json "$TP_WORK/frozen/"
if [[ $# -gt 0 ]]; then
    cp "$1" "$TP_WORK/input/queries.csv"
    export TP_SHARDS
    TP_SHARDS=$(.venv/bin/python - "$1" <<'PY'
import sys
import pandas as pd

queries = pd.read_csv(sys.argv[1], keep_default_na=False)
if not {"query_id", "smiles"} <= set(queries) or queries.empty or not queries["query_id"].is_unique:
    raise ValueError("Queries must have unique query_id and smiles columns, and at least one row")
print(min(20, len(queries)))
PY
)
fi
build_job=$(sbatch --parsable scripts/run_chembl_library.slurm)
build_job="${build_job%%;*}"
printf 'Build/evaluation job: %s\nScratch artifacts: %s\n' "$build_job" "$TP_WORK"
printf '%s\n' "$TP_WORK" > results/chembl-library-workdir.txt
if [[ $# -gt 0 ]]; then
    array_job=$(sbatch --parsable --dependency="afterok:$build_job" --kill-on-invalid-dep=yes \
        --array="0-$((TP_SHARDS - 1))%20" scripts/run_chembl_queries.slurm)
    array_job="${array_job%%;*}"
    sbatch --dependency="afterok:$array_job" --kill-on-invalid-dep=yes scripts/merge_chembl_queries.slurm
    printf 'Query array: %s\n' "$array_job"
fi
