"""Check which ChEMBL test types point at real drug targets.

For drugs with known targets, measures how often a ChEMBL row of each activity type
(IC50, Ki, Potency, ...) names one of the drug's known targets. This is the evidence
for KEEP_ACTIVITY_TYPES in build_dataset.py. Run on two random halves of the drugs
to show the result does not depend on which drugs are used.

Needs data/processed/chembl_activity.csv (from scripts/export_chembl.py, all types).

Usage (from the repo root):
    .venv/bin/python scripts/activity_type_check.py
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data/processed"
SEED = 0


def hit_rates(rows):
    return (rows.groupby("activity_type")["hit"].agg(rows="size", hit_rate="mean")
            .assign(hit_rate=lambda s: s["hit_rate"].round(3)).to_dict("index"))


def main():
    drugs = pd.read_csv(PROCESSED / "drugs.csv.gz").dropna(subset=["known_targets"])
    known = drugs.set_index("drug_id")["known_targets"].str.split("; ").apply(set).to_dict()

    ch = pd.read_csv(PROCESSED / "chembl_activity.csv", usecols=["inchikey", "target_gene", "activity_type"])
    rows = ch[ch["inchikey"].isin(known)].copy()
    rows["hit"] = [g in known[d] for d, g in zip(rows["inchikey"], rows["target_gene"])]

    ids = pd.Series(sorted(known)).sample(frac=1, random_state=SEED)
    half_a = set(ids[: len(ids) // 2])
    result = {
        "all_drugs": hit_rates(rows),
        "half_a": hit_rates(rows[rows["inchikey"].isin(half_a)]),
        "half_b": hit_rates(rows[~rows["inchikey"].isin(half_a)]),
    }
    (PROCESSED / "activity_type_check.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
