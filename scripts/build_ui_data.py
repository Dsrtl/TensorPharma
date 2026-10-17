"""Build the data behind the results library UI.

For every drug: predicted targets (with the drug hidden from the library, the same
way it is evaluated), evidence for each target, safety flags, a confidence level from
the closest known relative, and the measured cell lines, cancer types and subtypes
it kills most.

If outputs/step2/predicted_responses.csv exists (drug_id, cell_id, predicted_rank: predicted
within-drug kill rank, 0 = least, 1 = most), each drug also gets predicted top cell lines,
cancer types and subtypes, ranked the same way as the measured ones.

Kill scores (DSS3) are put on one 0-1 scale; a cell line counts as responsive at
DSS3 >= 0.2. Drugs with no responsive line get no top cancer type.

Outputs (outputs/ui/):
    library.json           one record per drug, nested
    drugs.csv              one row per drug (flat tables, same content)
    predicted_targets.csv  drug x top-5 targets
    evidence.csv           drug x target x supporting compound
    top_cells.csv          drug x top-10 measured cell lines
    cancer_types.csv       drug x cancer type (average kill rank, share responsive, number of lines)
    cancer_subtypes.csv    drug x cancer subtype (same columns)
    safety_flags.csv       drug x safety protein (hERG, CYPs, drug pumps)
    validation.json        headline numbers for the validation page

Usage (from the repo root):
    .venv/bin/python scripts/build_ui_data.py
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tensorpharma.targets import PROCESSED, SAFETY_GENES, TargetPredictor  # noqa: E402

OUT = ROOT / "outputs/ui"
STEP2 = ROOT / "outputs/step2/predicted_responses.csv"
K = 25
TOP_TARGETS = 5
TOP_EVIDENCE = 3
TOP_CELLS = 10
MIN_LINES_FOR_CONFIDENT_TYPE = 5
RESPONSIVE_DSS3 = 0.2
# a drug's breast lines over-represented in its top 10% far beyond chance (10%): likely a merged
# breast-only screen on a different scale (imatinib, ixabepilone, GSK-923295, NSC-663284)
BREAST_ARTIFACT_SHARE = 0.5
# small groups win "top cancer type" by chance (neuroblastoma: median 5 lines per drug), so the
# ranking score pulls each group's average toward 0.5 as if it had SHRINK extra average lines
SHRINK = 10


def confidence(similarity):
    # bands measured in scripts/eval_targets.py: ~84% / 58% / 26% top-5 accuracy
    if similarity >= 0.7:
        return "High"
    if similarity >= 0.5:
        return "Medium"
    return "Low"


def by_group(r, column, rank="sensitivity_rank"):
    g = (r.groupby(column).agg(avg_kill_rank=(rank, "mean"),
                               share_responsive=("responsive", "mean"), n_cell_lines=("responsive", "size"))
         .reset_index().rename(columns={column: "name"}))
    g["rank_score"] = (g["avg_kill_rank"] * g["n_cell_lines"] + 0.5 * SHRINK) / (g["n_cell_lines"] + SHRINK)
    g = g.sort_values("rank_score", ascending=False)
    g[["avg_kill_rank", "share_responsive", "rank_score"]] = g[["avg_kill_rank", "share_responsive", "rank_score"]].round(3)
    g["low_confidence"] = g["n_cell_lines"] < MIN_LINES_FOR_CONFIDENT_TYPE
    return g.to_dict("records")


def main():
    t0 = time.time()
    predictor = TargetPredictor()
    print(f"library ready ({time.time() - t0:.0f}s)", flush=True)

    chembl_ids = (pd.read_csv(PROCESSED / "drug_targets.csv.gz", usecols=["drug_id", "chembl_molecule_id"])
                  .drop_duplicates("drug_id").set_index("drug_id")["chembl_molecule_id"].to_dict())
    drugs = pd.read_csv(PROCESSED / "drugs.csv.gz")
    cells = pd.read_csv(PROCESSED / "cells.csv.gz").set_index("cell_id")
    responses = pd.read_csv(PROCESSED / "responses.csv.gz").join(cells, on="cell_id")
    # 20 drugs are scored 0-100, the rest 0-1
    scale = responses.groupby("drug_id")["sensitivity_raw"].transform("max").gt(1.5).map({True: 100, False: 1})
    responses["dss3"] = responses["sensitivity_raw"] / scale
    responses["responsive"] = responses["dss3"] >= RESPONSIVE_DSS3
    by_drug = dict(tuple(responses.groupby("drug_id")))
    predicted = {}
    if STEP2.exists():
        pr = pd.read_csv(STEP2).join(cells, on="cell_id")
        pr["responsive"] = False  # step 2 predicts rank, not absolute kill
        predicted = dict(tuple(pr.groupby("drug_id")))
        print(f"step 2 predictions: {len(pr)} rows, {len(predicted)} drugs", flush=True)

    records, flat = [], {"drugs": [], "predicted_targets": [], "evidence": [], "top_cells": [],
                         "cancer_types": [], "cancer_subtypes": [], "safety_flags": [], "predicted_cells": []}
    for _, d in drugs.iterrows():
        known = set(d["known_targets"].split("; ")) if isinstance(d["known_targets"], str) else set()
        known_targets = known - SAFETY_GENES  # what the accuracy check counts as a hit
        try:
            top, sims = predictor.neighbours(d["smiles"], exclude_drug_id=d["drug_id"], k=K)
            voted = predictor.vote(top, sims, K)
            ranked = [t for t in voted if t["gene"] not in SAFETY_GENES][:TOP_TARGETS]
            safety = [t["gene"] for t in voted[:10] if t["gene"] in SAFETY_GENES]
            nearest = round(float(sims[0]), 3)
        except ValueError:
            ranked, safety, nearest = [], [], 0.0
        for g in safety:
            flat["safety_flags"].append({"drug_id": d["drug_id"], "gene": g})

        targets = []
        for rank, t in enumerate(ranked, 1):
            ev = sorted(t["evidence"], key=lambda e: -e["similarity"])[:TOP_EVIDENCE]
            for e in ev:
                e["similar_compound"] = chembl_ids.get(e["similar_compound"], e["similar_compound"])
                flat["evidence"].append({"drug_id": d["drug_id"], "target_rank": rank, "gene": t["gene"], **e})
            targets.append({"rank": rank, "gene": t["gene"], "score": t["score"],
                            "is_known_target": t["gene"] in known_targets, "evidence": ev})
            flat["predicted_targets"].append({"drug_id": d["drug_id"], "rank": rank, "gene": t["gene"],
                                              "score": t["score"], "is_known_target": t["gene"] in known_targets})

        r = by_drug.get(d["drug_id"], pd.DataFrame(columns=responses.columns))
        top_cells = (r[r["responsive"]].sort_values("sensitivity_rank", ascending=False).head(TOP_CELLS)
                     [["cell_id", "cell_name", "cancer_type", "cancer_subtype", "sensitivity_rank", "dss3"]])
        cells_list = top_cells.round({"sensitivity_rank": 3, "dss3": 3}).to_dict("records")
        types_list = by_group(r, "cancer_type")
        subtypes_list = by_group(r, "cancer_subtype")
        any_responsive = bool(r["responsive"].any())
        breast = r[r["cancer_type"] == "Breast"]
        breast_artifact = bool(len(breast) and (breast["sensitivity_rank"] >= 0.9).mean() > BREAST_ARTIFACT_SHARE)

        for c in cells_list:
            flat["top_cells"].append({"drug_id": d["drug_id"], **c})
        for t in types_list:
            flat["cancer_types"].append({"drug_id": d["drug_id"], **t})
        for t in subtypes_list:
            flat["cancer_subtypes"].append({"drug_id": d["drug_id"], **t})
        confident = [t for t in types_list if not t["low_confidence"]
                     and not (breast_artifact and t["name"] == "Breast")]
        breast_subtypes = set(breast["cancer_subtype"]) if breast_artifact else set()
        confident_sub = [t for t in subtypes_list if not t["low_confidence"] and t["name"] not in breast_subtypes]

        base = {
            "drug_id": d["drug_id"], "name": d["name"], "smiles": d["smiles"],
            "known_targets": sorted(known), "moa": d["moa"] if isinstance(d["moa"], str) else None,
            "labeled": bool(known), "structure_conflict": bool(d["structure_conflict"]),
            "nearest_similarity": nearest, "confidence": confidence(nearest),
            "top_target": ranked[0]["gene"] if ranked else None,
            "top_cancer_type": confident[0]["name"] if any_responsive and confident else None,
            "top_cancer_subtype": confident_sub[0]["name"] if any_responsive and confident_sub else None,
            "responsive_cell_lines": int(r["responsive"].sum()), "tested_cell_lines": len(r),
            "breast_screen_artifact": breast_artifact,
            "safety_flags": safety,
            "top5_hits_known": any(t["is_known_target"] for t in targets) if known_targets else None,
        }
        p = predicted.get(d["drug_id"])
        if p is not None:
            pcells = (p.sort_values("predicted_rank", ascending=False).head(TOP_CELLS)
                      [["cell_id", "cell_name", "cancer_type", "cancer_subtype", "predicted_rank"]]
                      .round({"predicted_rank": 3}).to_dict("records"))
            pred = {"top_cells": pcells,
                    "cancer_types": [{k: v for k, v in t.items() if k != "share_responsive"}
                                     for t in by_group(p, "cancer_type", "predicted_rank")],
                    "cancer_subtypes": [{k: v for k, v in t.items() if k != "share_responsive"}
                                        for t in by_group(p, "cancer_subtype", "predicted_rank")]}
            conf_sub = [t for t in pred["cancer_subtypes"] if not t["low_confidence"]]
            base["predicted_top_cancer_subtype"] = conf_sub[0]["name"] if conf_sub else None
            for c in pcells:
                flat["predicted_cells"].append({"drug_id": d["drug_id"], **c})
        else:
            pred = None
            base["predicted_top_cancer_subtype"] = None

        flat["drugs"].append({**base, "known_targets": "; ".join(sorted(known)), "safety_flags": "; ".join(safety)})
        records.append({**base, "predicted_targets": targets, "measured_top_cells": cells_list,
                        "measured_cancer_types": types_list, "measured_cancer_subtypes": subtypes_list,
                        "predicted": pred, "why": None})

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "library.json").write_text(json.dumps(records, indent=1, default=str))
    for name, rows in flat.items():
        pd.DataFrame(rows).to_csv(OUT / f"{name}.csv", index=False)

    labeled = [r for r in records if r["top5_hits_known"] is not None]
    bands = {}
    for level in ["High", "Medium", "Low"]:
        group = [r for r in labeled if r["confidence"] == level]
        bands[level] = {"drugs": len(group),
                        "top5_accuracy": round(sum(r["top5_hits_known"] for r in group) / len(group), 3) if group else None}
    eval_path = ROOT / "outputs/target_eval.json"
    check_path = PROCESSED / "activity_type_check.json"
    validation = {
        "target_prediction": json.loads(eval_path.read_text()) if eval_path.exists() else None,
        "accuracy_by_confidence": bands,
        "activity_type_check": json.loads(check_path.read_text()) if check_path.exists() else None,
        "dataset": {"drugs": len(records), "labeled_drugs": len(labeled), "cell_lines": int(len(cells)),
                    "cancer_types": int(cells["cancer_type"].nunique()),
                    "responsive_dss3_cutoff": RESPONSIVE_DSS3,
                    "drugs_with_no_responsive_line": sum(r["responsive_cell_lines"] == 0 for r in records),
                    "breast_screen_artifact_drugs": [r["name"] for r in records if r["breast_screen_artifact"]]},
    }
    (OUT / "validation.json").write_text(json.dumps(validation, indent=2))
    print(f"wrote {len(records)} drugs to {OUT.relative_to(ROOT)} ({time.time() - t0:.0f}s)")
    print(json.dumps(bands, indent=2))


if __name__ == "__main__":
    main()
