#!/bin/bash
#SBATCH --job-name=emerge-target-merge
#SBATCH --account=mpensky
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=results/chembl-merge-%j.log

set -euo pipefail
module load python/python-3.11.4-gcc-12.2.0
cd "${SLURM_SUBMIT_DIR:?}"
.venv/bin/python - <<'PY'
import os
from pathlib import Path
import pandas as pd

work = Path(os.environ["TP_WORK"])
shards = int(os.environ["TP_SHARDS"])
frames = [pd.read_csv(work / "predictions" / ("shard-" + str(index) + ".csv")) for index in range(shards)]
pd.concat(frames, ignore_index=True).to_csv("results/chembl_library/query_predictions.csv", index=False)
PY
