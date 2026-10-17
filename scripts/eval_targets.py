"""Evaluate target prediction on drugs the predictor has never seen.

For every drug with known targets, the drug and its stereo variants are hidden
from the ChEMBL library, its targets are predicted from the remaining compounds,
and we check whether a known target appears in the top 1 / 5 / 10. Safety and
metabolism proteins are removed from both the predictions and the answer key.

Compared against:
    popularity   always guess the most common known targets
    1-NN         copy the single most similar compound's targets, strongest first
    count / similarity / full / full_popularity   vote weightings (see targets.py)
    ceiling      look the drug itself up in ChEMBL (the answer key is partly built from ChEMBL),
                 on the drugs that are in ChEMBL

Also reported: accuracy by similarity of the closest library compound, precision@5,
and the same test with near-copies (similarity >= 0.95) hidden as well.

Usage (from the repo root):
    .venv/bin/python scripts/eval_targets.py
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tensorpharma.targets import PROCESSED, SAFETY_GENES, WEIGHTINGS, TargetPredictor  # noqa: E402

K = 25
TOP = [1, 5, 10]
NEAR_COPY = 0.95
BANDS = [(0.95, 1.01, ">=0.95"), (0.8, 0.95, "0.8-0.95"), (0.7, 0.8, "0.7-0.8"), (0.5, 0.7, "0.5-0.7"), (0, 0.5, "<0.5")]
OUT = ROOT / "outputs"


def clean(genes):
    return [g for g in genes if g not in SAFETY_GENES]


def score(ranked, known):
    hits = {f"top{n}": any(g in known for g in ranked[:n]) for n in TOP}
    hits["precision5"] = sum(g in known for g in ranked[:5]) / 5
    return hits


def summarise(rows, method):
    df = pd.DataFrame([r[method] for r in rows])
    return {c: round(float(df[c].mean()), 3) for c in df.columns}


def main():
    t0 = time.time()
    predictor = TargetPredictor()
    print(f"library ready: {len(predictor.ids)} compounds ({time.time() - t0:.0f}s)", flush=True)

    drugs = pd.read_csv(PROCESSED / "drugs.csv.gz").dropna(subset=["known_targets"])
    drugs["known"] = drugs["known_targets"].str.split("; ").apply(lambda s: set(s) - SAFETY_GENES)
    dropped_safety_only = int((drugs["known"].str.len() == 0).sum())
    drugs = drugs[drugs["known"].str.len() > 0]
    all_counts = Counter(g for s in drugs["known"] for g in s)

    rows = []
    for _, d in drugs.iterrows():
        sims = predictor.similarities(d["smiles"], exclude_drug_id=d["drug_id"])
        top, top_sims = predictor.top_k(sims, K)
        r = {"drug_id": d["drug_id"], "name": d["name"], "known_targets": "; ".join(sorted(d["known"])),
             "nearest_similarity": round(float(top_sims[0]), 3), "in_chembl": d["drug_id"] in predictor.row_of}

        for w in WEIGHTINGS:
            ranked = clean(t["gene"] for t in predictor.vote(top, top_sims, K, w))
            r[w] = score(ranked, d["known"])
            if w == "full":
                r["predicted_top5"] = "; ".join(ranked[:5])

        r["1-NN"] = score(clean(predictor.own_targets(predictor.ids[top[0]])), d["known"])
        r["ceiling"] = score(clean(predictor.own_targets(d["drug_id"])), d["known"])

        # popularity: most common known targets among the other drugs
        others = all_counts - Counter(d["known"])
        r["popularity"] = score([g for g, _ in others.most_common(max(TOP))], d["known"])

        near = sims.copy()
        near[near >= NEAR_COPY] = -1.0
        ntop, nsims = predictor.top_k(near, K)
        r["full_no_near_copies"] = score(clean(t["gene"] for t in predictor.vote(ntop, nsims, K, "full")), d["known"])
        rows.append(r)

    methods = ["popularity", "1-NN", *WEIGHTINGS, "full_no_near_copies"]
    in_chembl = [r for r in rows if r["in_chembl"]]
    by_band = {}
    for lo, hi, label in BANDS:
        group = [r for r in rows if lo <= r["nearest_similarity"] < hi]
        by_band[label] = {"drugs": len(group), **({m: summarise(group, m)["top5"] for m in ["1-NN", "full"]}
                                                   if group else {})}

    report = {
        "drugs_evaluated": len(rows),
        "drugs_dropped_only_safety_targets": dropped_safety_only,
        "safety_genes_excluded": sorted(SAFETY_GENES),
        "k": K,
        "methods": {m: summarise(rows, m) for m in methods},
        "ceiling_on_drugs_in_chembl": {"drugs": len(in_chembl), "look_up_drug_itself": summarise(in_chembl, "ceiling"),
                                       "full": summarise(in_chembl, "full")},
        "top5_by_nearest_similarity": by_band,
        "median_nearest_similarity": round(float(pd.Series([r["nearest_similarity"] for r in rows]).median()), 3),
        "notes": ["k=25 was chosen earlier on these same drugs (k=10/25/50 spread under 2 points)",
                  "the answer key (CMap/Repurposing Hub) is partly curated from ChEMBL, so 'ceiling' is the "
                  "score from looking the drug itself up; the predictor never sees the drug",
                  "trust weights were fitted on all labeled drugs (no split file yet)"],
        "seconds": round(time.time() - t0),
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "target_eval.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame([{**{k: v for k, v in r.items() if not isinstance(v, dict)}, "hit_top5": r["full"]["top5"]}
                  for r in rows]).to_csv(OUT / "target_predictions.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
