#!/bin/bash
#SBATCH --job-name=emerge-cancer-only
#SBATCH --account=mpensky
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=results/cancer-only-rank-%j.log

set -euo pipefail
module load python/python-3.11.4-gcc-12.2.0
cd "${SLURM_SUBMIT_DIR:?Submit from the eMerge project directory}"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export BLIS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1
output="results/rank_benchmark_cancer_only"
if [[ -f "$output/benchmark_report.json" ]]; then
    printf '%s\n' 'Completed cancer-only benchmark exists; retain it rather than overwrite.' >&2
    exit 1
fi
for name in cells gene_expression responses; do
    test -f "data/updated_cancer_cell_data/$name.csv.gz"
done
for name in drugs drug_ids drug_targets; do
    test -f "TensorPharma-main/data/processed/$name.csv.gz"
done
test -f data/features/compound_split.csv
test -f data/features/fingerprints.npz
test -f data/features/feature_report.json
.venv/bin/python -c 'import lightgbm, narwhals, numpy, scipy, pandas, rdkit; print("Existing benchmark environment available")'
.venv/bin/python -m pip check
mkdir -p "$output/input"
for name in cells gene_expression responses; do
    cp "data/updated_cancer_cell_data/$name.csv.gz" "$output/input/"
done
for name in drugs drug_ids drug_targets; do
    cp "TensorPharma-main/data/processed/$name.csv.gz" "$output/input/"
done
.venv/bin/python -m pip freeze > "$output/environment.txt"
.venv/bin/python - <<'PY'
import json
from pathlib import Path

import numpy as np
import pandas as pd

original = Path("TensorPharma-main/data/processed")
updated = Path("results/rank_benchmark_cancer_only/input")
before = pd.read_csv(original / "cells.csv.gz", keep_default_na=False)
after = pd.read_csv(updated / "cells.csv.gz", keep_default_na=False)
if not before["cell_id"].is_unique or not after["cell_id"].is_unique:
    raise ValueError("Duplicate cell metadata")
removed = before[~before["cell_id"].isin(after["cell_id"])]
if removed.empty or not removed["cancer_subtype"].str.lower().eq("non-cancerous").all():
    raise ValueError("Expected only removal of cells labelled Non-Cancerous")
retained = before[before["cell_id"].isin(after["cell_id"])].sort_values("cell_id").reset_index(drop=True)
if not retained.equals(after.sort_values("cell_id").reset_index(drop=True)):
    raise ValueError("Retained cell metadata changed")
old_responses = pd.read_csv(original / "responses.csv.gz")
new_responses = pd.read_csv(updated / "responses.csv.gz")
expected = old_responses[old_responses["cell_id"].isin(after["cell_id"])]
keys = ["drug_id", "cell_id"]
if new_responses.duplicated(keys).any() or expected.duplicated(keys).any():
    raise ValueError("Duplicate drug/cell observations")
joined = expected.merge(new_responses, on=keys, how="outer", suffixes=("_old", "_new"), indicator=True, validate="one_to_one")
if not joined["_merge"].eq("both").all():
    raise ValueError("Response changes exceed removed cell IDs")
for column in ("sensitivity_raw", "n_sources"):
    if not joined[column + "_old"].equals(joined[column + "_new"]):
        raise ValueError("Retained response values changed: " + column)
ranks = new_responses.groupby("drug_id")["sensitivity_raw"].rank(pct=True, method="average")
if not np.allclose(ranks, new_responses["sensitivity_rank"], atol=1e-8, rtol=0):
    raise ValueError("Updated ranks do not reproduce")
report = {
    "variant": "cancer_only_scope_correction",
    "reason": "Exclude source cell lines labelled Non-Cancerous, not a claim of general bias removal",
    "old_cells": len(before), "new_cells": len(after),
    "removed_cells": removed.to_dict("records"),
    "removed_response_rows": len(old_responses) - len(new_responses),
    "retained_raw_response_and_source_counts_unchanged": True,
    "within_drug_ranks_recomputed": True,
    "chemical_split": "Original frozen assignments retained",
    "selection": "Same six-family grid; validation only; previous test inspection makes results exploratory",
    "previous_results": "results/rank_benchmark (not modified)",
}
Path("results/rank_benchmark_cancer_only/dataset_change_report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
PY
.venv/bin/python scripts/rank_benchmark.py \
    --source "$output/input" --frozen data/features \
    --output "$output" --threads "$SLURM_CPUS_PER_TASK"
