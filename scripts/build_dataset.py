"""Build the clean, joined dataset from the raw files.

Inputs (data/raw/):
    DrugSensitivity/L1000 level4 data/StandardAnnotations/compoundinfo.txt
    DrugSensitivity/L1000_cp_pertid_Cellosaurus_celllineid_PharmacoDB_dss3.csv
    DrugSensitivity/CCLE/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv
    DepMap/Model.csv
and data/processed/chembl_activity.csv (from scripts/export_chembl.py).

Outputs (data/processed/):
    drugs.csv.gz            one row per drug, keyed by InChIKey
    drug_ids.csv.gz         LINCS pert_id -> drug_id
    drug_targets.csv.gz     cleaned ChEMBL drug-protein potency library
    cells.csv.gz            one row per cancer cell line with cancer type (non-cancerous lines dropped)
    gene_expression.csv.gz  cell line x gene expression (default entries only)
    responses.csv.gz        drug x cell line sensitivity, raw and ranked within drug
    build_audit.json     rows in / kept / dropped at every step

Usage (from the repo root):
    python3 scripts/build_dataset.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw"
OUT = ROOT / "data/processed"

COMPOUNDINFO = RAW / "DrugSensitivity/L1000 level4 data/StandardAnnotations/compoundinfo.txt"
SENSITIVITY = RAW / "DrugSensitivity/L1000_cp_pertid_Cellosaurus_celllineid_PharmacoDB_dss3.csv"
EXPRESSION = RAW / "DrugSensitivity/CCLE/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
MODELS = RAW / "DepMap/Model.csv"
CHEMBL = OUT / "chembl_activity.csv"

KEEP_ACTIVITY_TYPES = ["IC50", "Ki", "Kd", "EC50"]  # Potency, ED50, AC50 tested as noise
EXPRESSION_META = ["Unnamed: 0", "SequencingID", "ModelConditionID", "ModelID",
                   "IsDefaultEntryForMC", "IsDefaultEntryForModel"]

# compoundinfo and Small_Molecule_Metadata disagree on these structures; compoundinfo is used
STRUCTURE_CONFLICTS = {"BRD-K35329391", "BRD-K65924316", "BRD-K20696416", "BRD-K76011241", "BRD-M16762496"}

audit = {}


def check_unique(df, key, name):
    dups = df.duplicated(key).sum()
    if dups:
        sys.exit(f"{name}: {dups} duplicate keys on {key}")


def join_set(values):
    vals = sorted({v for v in values if isinstance(v, str) and v})
    return "; ".join(vals) if vals else None


def build_drugs(pert_ids):
    ci = pd.read_csv(COMPOUNDINFO, sep="\t", dtype=str)
    ci = ci[ci["pert_id"].isin(pert_ids)]
    missing = pert_ids - set(ci["pert_id"])
    if missing:
        sys.exit(f"compoundinfo is missing {len(missing)} sensitivity pert_ids")
    if ci.groupby("pert_id")["inchi_key"].nunique().gt(1).any():
        sys.exit("a pert_id has more than one InChIKey in compoundinfo")

    drug_ids = (ci.drop_duplicates("pert_id")[["pert_id", "inchi_key"]]
                .rename(columns={"inchi_key": "drug_id"}).sort_values("pert_id"))
    check_unique(drug_ids, "pert_id", "drug_ids")

    rows = []
    for drug_id, g in ci.groupby("inchi_key"):
        perts = sorted(g["pert_id"].unique())
        rows.append({
            "drug_id": drug_id,
            "name": sorted(g["cmap_name"].dropna().unique())[0],
            "smiles": g["canonical_smiles"].iloc[0],
            "known_targets": join_set(g["target"]),
            "moa": join_set(g["moa"]),
            "pert_ids": "; ".join(perts),
            "structure_conflict": any(p in STRUCTURE_CONFLICTS for p in perts),
        })
    drugs = pd.DataFrame(rows).sort_values("drug_id")
    check_unique(drugs, "drug_id", "drugs")
    if drugs.groupby("drug_id")["smiles"].nunique().gt(1).any():
        sys.exit("one drug_id has different SMILES")

    audit["drugs"] = {"pert_ids": len(drug_ids), "unique_drugs": len(drugs),
                      "merged_duplicate_ids": len(drug_ids) - len(drugs),
                      "with_known_targets": int(drugs["known_targets"].notna().sum()),
                      "structure_conflicts_flagged": int(drugs["structure_conflict"].sum())}
    return drugs, drug_ids


def build_drug_targets():
    ch = pd.read_csv(CHEMBL)
    n_in = len(ch)
    ch = ch[ch["activity_type"].isin(KEEP_ACTIVITY_TYPES)]
    ch = ch.rename(columns={"inchikey": "drug_id"}).drop(columns=["target_uniprot"])
    check_unique(ch, ["drug_id", "target_chembl_id", "activity_type"], "drug_targets")
    audit["drug_targets"] = {"rows_in": n_in, "rows_kept": len(ch),
                             "activity_types_kept": KEEP_ACTIVITY_TYPES,
                             "compounds": int(ch["drug_id"].nunique()),
                             "targets": int(ch["target_chembl_id"].nunique())}
    return ch


def build_cells(cell_ids):
    models = pd.read_csv(MODELS, dtype=str)
    expr_meta = pd.read_csv(EXPRESSION, usecols=["ModelID", "IsDefaultEntryForModel"], dtype=str)
    with_expr = set(expr_meta.loc[expr_meta["IsDefaultEntryForModel"] == "Yes", "ModelID"])

    cells = models[models["ModelID"].isin(cell_ids & with_expr)]
    cells = cells[["ModelID", "CellLineName", "OncotreeLineage", "OncotreePrimaryDisease"]].rename(
        columns={"ModelID": "cell_id", "CellLineName": "cell_name",
                 "OncotreeLineage": "cancer_type", "OncotreePrimaryDisease": "cancer_subtype"})
    cells = cells.dropna(subset=["cancer_type"])
    # the task predicts cancer cells: drop normal fibroblasts and benign lines
    non_cancer = cells["cancer_subtype"] == "Non-Cancerous"
    cells = cells[~non_cancer].sort_values("cell_id")
    check_unique(cells, "cell_id", "cells")

    expr = pd.read_csv(EXPRESSION)
    expr = expr[(expr["IsDefaultEntryForModel"] == "Yes") & expr["ModelID"].isin(set(cells["cell_id"]))]
    expr = expr.drop(columns=[c for c in EXPRESSION_META if c != "ModelID"]).rename(columns={"ModelID": "cell_id"})
    check_unique(expr, "cell_id", "gene_expression")
    expr = expr.sort_values("cell_id")

    audit["cells"] = {"cell_lines_in_responses": len(cell_ids),
                      "with_expression": len(cell_ids & with_expr),
                      "non_cancerous_dropped": int(non_cancer.sum()),
                      "cancer_cell_lines_kept": len(cells),
                      "cancer_types": int(cells["cancer_type"].nunique()),
                      "genes": expr.shape[1] - 1}
    return cells, expr


def build_responses(sens, drug_ids, cells):
    n_in = len(sens)
    r = sens.merge(drug_ids, on="pert_id", how="inner").rename(columns={"cell_line": "cell_id"})
    r = r[r["cell_id"].isin(set(cells["cell_id"]))]
    n_matched = len(r)
    # merged duplicate drugs tested on the same cell line: average their scores
    r = (r.groupby(["drug_id", "cell_id"], as_index=False)
         .agg(sensitivity_raw=("sensitivity_score", "mean"), n_sources=("pert_id", "nunique")))
    # rank within each drug so drugs on different score scales are comparable (0 = least, 1 = most)
    r["sensitivity_rank"] = r.groupby("drug_id")["sensitivity_raw"].rank(pct=True, method="average")
    check_unique(r, ["drug_id", "cell_id"], "responses")
    audit["responses"] = {"rows_in": n_in, "rows_after_drug_and_cell_match": n_matched,
                          "rows_after_merging_duplicate_drugs": len(r),
                          "pairs_averaged": int((r["n_sources"] > 1).sum()),
                          "drugs": int(r["drug_id"].nunique()), "cells": int(r["cell_id"].nunique())}
    return r


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sens = pd.read_csv(SENSITIVITY, dtype={"pert_id": str, "cell_line": str})

    drugs, drug_ids = build_drugs(set(sens["pert_id"]))
    drug_targets = build_drug_targets()
    cells, expr = build_cells(set(sens["cell_line"]))
    responses = build_responses(sens, drug_ids, cells)

    drugs["in_chembl"] = drugs["drug_id"].isin(set(drug_targets["drug_id"]))
    audit["drugs"]["in_chembl"] = int(drugs["in_chembl"].sum())

    drugs.to_csv(OUT / "drugs.csv.gz", index=False)
    drug_ids.to_csv(OUT / "drug_ids.csv.gz", index=False)
    drug_targets.to_csv(OUT / "drug_targets.csv.gz", index=False)
    cells.to_csv(OUT / "cells.csv.gz", index=False)
    expr.to_csv(OUT / "gene_expression.csv.gz", index=False)
    responses.to_csv(OUT / "responses.csv.gz", index=False)
    (OUT / "build_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
