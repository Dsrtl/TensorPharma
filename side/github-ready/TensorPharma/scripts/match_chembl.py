import argparse
import hashlib
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import pandas as pd


MATCH_COLUMNS = ["pert_id", "molregno", "chembl_id", "standardized_smiles", "inchi_key", "split"]


def match_candidates(records, compounds, identity, prefixes, source_smiles):
    by_key = defaultdict(list)
    by_connectivity = defaultdict(list)
    for compound in compounds.to_dict("records"):
        by_key[compound["inchi_key"]].append(compound)
        by_connectivity[compound["connectivity_key"]].append(compound)
    matches = []
    connectivity_only = []
    failures = []
    scanned = 0
    candidates = 0
    for record in records:
        scanned += 1
        key = record["standard_inchi_key"] or ""
        smiles = record["canonical_smiles"] or ""
        if key[:14] not in prefixes and smiles not in source_smiles:
            continue
        candidates += 1
        try:
            canonical, standardized_key = identity(smiles)
        except (ValueError, RuntimeError) as error:
            failures.append({"molregno": record["molregno"], "reason": str(error)})
            continue
        exact = by_key.get(standardized_key, [])
        for compound in exact:
            matches.append({"pert_id": compound["pert_id"], "molregno": record["molregno"],
                            "chembl_id": record["chembl_id"], "standardized_smiles": canonical,
                            "inchi_key": standardized_key, "split": compound["split"]})
        if not exact:
            for compound in by_connectivity.get(standardized_key[:14], []):
                connectivity_only.append({"pert_id": compound["pert_id"], "chembl_id": record["chembl_id"],
                                          "molregno": record["molregno"], "chembl_inchi_key": standardized_key,
                                          "lincs_inchi_key": compound["inchi_key"]})
    return (pd.DataFrame(matches, columns=MATCH_COLUMNS),
            pd.DataFrame(connectivity_only, columns=["pert_id", "chembl_id", "molregno", "chembl_inchi_key", "lincs_inchi_key"]),
            {"reference_molecules_scanned": scanned, "reference_candidates_standardized": candidates,
             "standardization_failures": failures})


def aggregate_labels(activities, threshold=6.0):
    columns = ["pert_id", "target_id", "target_name", "standard_type", "assay_type", "split"]
    endpoints = activities.groupby(columns, dropna=False)["pchembl_value"].agg(
        median_pchembl="median", measurements="size", minimum_pchembl="min", maximum_pchembl="max").reset_index()
    endpoints["endpoint_active"] = endpoints["median_pchembl"] >= threshold
    labels = endpoints.groupby(["pert_id", "target_id", "target_name", "split"], dropna=False)["endpoint_active"].agg(
        known_active="max", all_endpoints_active="min", endpoint_groups="size").reset_index()
    labels["endpoint_disagreement"] = labels["known_active"] != labels["all_endpoints_active"]
    return endpoints, labels


def prepare(root):
    from rdkit import Chem, rdBase

    from challenge6_features import standardize

    feature_dir = root / "data" / "features"
    feature_report = json.loads((feature_dir / "feature_report.json").read_text())
    if feature_report["configuration"]["rdkit_version"] != rdBase.rdkitVersion:
        raise ValueError("Use the same RDKit version as the frozen features")
    split_path = feature_dir / "compound_split.csv"
    compounds = pd.read_csv(split_path, keep_default_na=False)
    for column in ("inchi_key", "connectivity_key", "scaffold"):
        if compounds.groupby(column)["split"].nunique().max() != 1:
            raise ValueError("Split leakage in " + column)
    prefixes = set(compounds["connectivity_key"])
    source_smiles = set(compounds["smiles"]) | set(compounds["standardized_smiles"])
    for smiles in compounds["smiles"]:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is not None:
            key = Chem.MolToInchiKey(molecule)
            if key:
                prefixes.add(key[:14])

    def identity(smiles):
        molecule = standardize(smiles)
        key = Chem.MolToInchiKey(molecule)
        if not key:
            raise ValueError("InChIKey generation failed")
        return Chem.MolToSmiles(molecule, isomericSmiles=True), key

    output = root / "data" / "chembl_matched"
    output.mkdir(parents=True, exist_ok=True)
    database = root / "data" / "chembl" / "chembl_filtered.sqlite"
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        records = connection.execute("SELECT molregno, chembl_id, canonical_smiles, standard_inchi_key FROM molecules")
        matches, candidates, summary = match_candidates(records, compounds, identity, prefixes, source_smiles)
        matches = matches.drop_duplicates(["pert_id", "molregno"])
        matches.to_csv(output / "compound_matches.csv", index=False)
        candidates.to_csv(output / "connectivity_only_excluded.csv", index=False)
        if matches.empty:
            raise ValueError("No full standardized InChIKey matches; inspect connectivity_only_excluded.csv")
        connection.execute("CREATE TEMP TABLE compound_matches (pert_id TEXT, molregno INTEGER, split TEXT)")
        connection.executemany("INSERT INTO compound_matches VALUES (?, ?, ?)",
                               matches[["pert_id", "molregno", "split"]].itertuples(index=False, name=None))
        activities = pd.read_sql_query("""
            SELECT compound_matches.pert_id, compound_matches.split, activities.*,
                   targets.target_id, targets.target_name, assays.confidence_score, assays.assay_type
            FROM compound_matches JOIN activities USING (molregno)
            JOIN targets USING (tid) JOIN assays USING (assay_id)
        """, connection).drop_duplicates(["pert_id", "activity_id"])
    finally:
        connection.close()
    if activities.empty:
        raise ValueError("Matched compounds have no eligible target activities")
    endpoints, labels = aggregate_labels(activities)
    activities.to_csv(output / "matched_activities.csv.gz", index=False)
    endpoints.to_csv(output / "endpoint_activity.csv", index=False)
    labels.to_csv(output / "target_labels.csv", index=False)
    compound_coverage = compounds[["pert_id", "name", "split", "inchi_key"]].copy()
    compound_coverage["matched"] = compound_coverage["pert_id"].isin(matches["pert_id"])
    compound_coverage["has_known_active_target"] = compound_coverage["pert_id"].isin(labels.loc[labels["known_active"], "pert_id"])
    compound_coverage.to_csv(output / "compound_coverage.csv", index=False)
    report = {**summary, "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
              "lincs_compounds": len(compounds), "exact_matched_compounds": int(matches["pert_id"].nunique()),
              "matched_chembl_molecules": int(matches["molregno"].nunique()),
              "matched_activity_rows": len(activities), "endpoint_groups": len(endpoints),
              "measured_compound_target_pairs": len(labels),
              "known_active_pairs": int(labels["known_active"].sum()),
              "endpoint_disagreement_pairs": int(labels["endpoint_disagreement"].sum()),
              "coverage_by_split": compound_coverage.groupby("split")[["matched", "has_known_active_target"]].sum().astype(int).to_dict("index"),
              "notes": ["Only full InChIKey matches after the frozen standardization pipeline are accepted.",
                        "Connectivity-only matches are exported for inspection and excluded from labels.",
                        "Source connectivity/SMILES prefilter can miss structures whose connectivity changes on standardization; coverage is a conservative first pass.",
                        "Medians are computed within endpoint and assay type. A target is known active if any such median is at least pChEMBL 6.",
                        "This active-target rule is a screening convention, not a universal biological truth or calibrated probability.",
                        "Missing targets are unknown, not negative. Repeated LINCS aliases are deduplicated by structure during target evaluation.",
                        "All partitions are retained for label evaluation; held-out activities must never enter training/retrieval evidence."]}
    (output / "matching_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    arguments = parser.parse_args()
    prepare(arguments.root.resolve())
