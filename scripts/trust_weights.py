"""Measure how reliable ChEMBL evidence is, and turn that into trust weights.

For drugs with known targets, checks how often a ChEMBL row points at one of the
drug's known targets, split by number of lab results (1 vs 2+). The weight for each
group is its hit rate divided by the best group's hit rate.

With a split file, only training drugs are used, so held-out drugs never influence
the weights.

Usage (from the repo root):
    python3 scripts/trust_weights.py                     # all drugs (preview)
    python3 scripts/trust_weights.py data/processed/split.csv   # training drugs only
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data/processed"


def main():
    drugs = pd.read_csv(PROCESSED / "drugs.csv.gz")
    targets = pd.read_csv(PROCESSED / "drug_targets.csv.gz",
                          usecols=["drug_id", "target_gene", "n_measurements"])

    scope = "all drugs"
    if len(sys.argv) > 1:
        split = pd.read_csv(sys.argv[1])
        drugs = drugs[drugs["drug_id"].isin(split.loc[split["split"] == "train", "drug_id"])]
        scope = "training drugs only"

    known = drugs.dropna(subset=["known_targets"]).set_index("drug_id")["known_targets"].str.split("; ")
    known = known.apply(set).to_dict()

    rows = targets[targets["drug_id"].isin(known)].copy()
    rows["hits_known_target"] = [g in known[d] for d, g in zip(rows["drug_id"], rows["target_gene"])]
    rows["group"] = rows["n_measurements"].map(lambda n: "1 result" if n == 1 else "2+ results")

    stats = rows.groupby("group")["hits_known_target"].agg(rows="size", hits="sum")
    stats["hit_rate"] = stats["hits"] / stats["rows"]
    stats["weight"] = stats["hit_rate"] / stats["hit_rate"].max()

    result = {
        "scope": scope,
        "drugs_used": int(rows["drug_id"].nunique()),
        "groups": {g: {"rows": int(r.rows), "hits": int(r.hits),
                       "hit_rate": round(float(r.hit_rate), 4), "weight": round(float(r.weight), 3)}
                   for g, r in stats.iterrows()},
    }
    (PROCESSED / "trust_weights.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
