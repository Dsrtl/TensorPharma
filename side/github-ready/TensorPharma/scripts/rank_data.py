import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import svd


def table_path(source, name):
    for suffix in (".csv.gz", ".csv"):
        path = source / (name + suffix)
        if path.is_file():
            return path
    raise FileNotFoundError("Missing processed table: " + name)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gene_features(source, cells, gene_count):
    path = table_path(source, "gene_expression")
    header = pd.read_csv(path, nrows=0).columns.tolist()
    genes = [column for column in header if column != "cell_id"]
    expression = pd.read_csv(path, dtype={gene: np.float32 for gene in genes})
    if not expression["cell_id"].is_unique:
        raise ValueError("Duplicate expression cell IDs")
    if not set(cells) <= set(expression["cell_id"]):
        raise ValueError("Missing expression for benchmark cells")
    values = expression.set_index("cell_id").loc[cells, genes].to_numpy(dtype=np.float64)
    if np.isinf(values).any():
        raise ValueError("Infinite expression values")
    valid = np.isfinite(values).any(axis=0)
    values = values[:, valid]
    genes = np.array(genes, dtype=str)[valid]
    medians = np.nanmedian(values, axis=0)
    values = np.where(np.isfinite(values), values, medians)
    variance = values.var(axis=0)
    selected = np.argsort(-variance, kind="stable")[:min(gene_count, int((variance > 1e-12).sum()))]
    if not len(selected):
        raise ValueError("No variable genes")
    values = values[:, selected]
    center = values.mean(axis=0)
    scale = values.std(axis=0)
    normalized = (values - center) / scale
    _, singular_values, directions = svd(normalized, full_matrices=False)
    dimensions = min(32, int((singular_values > 1e-8).sum()))
    if dimensions < 1:
        raise ValueError("No expression components")
    projection = directions[:dimensions].T
    embedded = normalized @ projection
    embedding_scale = np.maximum(embedded.std(axis=0), 1e-8)
    design = np.column_stack([np.ones(len(cells)), embedded / embedding_scale])
    return {"gene_names": genes[selected], "gene_values": values.astype(np.float32),
            "gene_medians": medians[selected], "gene_center": center, "gene_scale": scale,
            "gene_projection": projection, "gene_embedding_scale": embedding_scale,
            "expression_design": design}


def resolve_mapping(bridge, drugs, output, identity_from_smiles=None):
    if not drugs["drug_id"].is_unique:
        raise ValueError("Duplicate cleaned drug metadata IDs")
    if not set(bridge["drug_id"]) <= set(drugs["drug_id"]):
        raise ValueError("Missing cleaned drug structure metadata")
    if identity_from_smiles is None:
        from rdkit import Chem
        from challenge6_features import standardize

        def identity_from_smiles(smiles):
            return Chem.MolToInchiKey(standardize(smiles))

    metadata = drugs.set_index("drug_id")
    reasons = {}
    for identifier, records in bridge.groupby("drug_id", sort=True):
        problems = []
        if records["inchi_key"].nunique() != 1:
            problems.append("merged_source_id_has_multiple_frozen_identities")
        if records["split"].nunique() != 1:
            problems.append("merged_source_id_crosses_frozen_partitions")
        row = metadata.loc[identifier]
        if str(row.get("structure_conflict", "false")).lower() in ("true", "1", "yes"):
            problems.append("source_structure_conflict_flag")
        try:
            key = identity_from_smiles(row["smiles"])
            if not key or set(records["inchi_key"]) != {key}:
                problems.append("source_structure_disagrees_with_frozen_identity")
        except (ValueError, RuntimeError):
            problems.append("source_structure_could_not_be_standardized")
        if problems:
            reasons[identifier] = ";".join(problems)
    exclusions = bridge[bridge["drug_id"].isin(reasons)].copy()
    exclusions["exclusion_reason"] = exclusions["drug_id"].map(reasons)
    exclusions = exclusions.merge(drugs[["drug_id", "name", "smiles"]], on="drug_id", validate="many_to_one")
    output.mkdir(parents=True, exist_ok=True)
    exclusions.to_csv(output / "mapping_exclusions.csv", index=False)
    print("Excluded unresolved source drug IDs:", len(reasons), "; audit:", output / "mapping_exclusions.csv", flush=True)
    accepted = bridge[~bridge["drug_id"].isin(reasons)].copy()
    if accepted.empty:
        raise ValueError("No unambiguous cleaned-to-frozen mappings survive")
    return accepted, exclusions


def prepare(source, frozen, output, gene_count=1000):
    if not source.is_dir():
        raise FileNotFoundError(source)
    manifest_path = frozen / "compound_split.csv"
    compounds = pd.read_csv(manifest_path, keep_default_na=False)
    for column in ("inchi_key", "connectivity_key", "scaffold"):
        if compounds.groupby(column)["split"].nunique().max() != 1:
            raise ValueError("Frozen split leakage in " + column)
    with np.load(frozen / "fingerprints.npz", allow_pickle=False) as archive:
        if not np.array_equal(archive["pert_ids"], compounds["pert_id"].to_numpy()):
            raise ValueError("Frozen fingerprints and manifest differ")
        fingerprints = archive["fingerprints"].copy()
    feature_lookup = {identifier: index for index, identifier in enumerate(compounds["pert_id"])}
    identifiers = pd.read_csv(table_path(source, "drug_ids"), keep_default_na=False)
    if not identifiers["pert_id"].is_unique:
        raise ValueError("Duplicate LINCS IDs in cleaned drug mapping")
    if not set(identifiers["pert_id"]) <= set(compounds["pert_id"]):
        raise ValueError("Cleaned data contains compounds absent from the frozen split")
    bridge = identifiers.merge(compounds[["pert_id", "inchi_key", "split"]], on="pert_id", validate="one_to_one")
    drugs = pd.read_csv(table_path(source, "drugs"), keep_default_na=False)
    bridge, mapping_exclusions = resolve_mapping(bridge, drugs, output)
    mapping = bridge.drop_duplicates("drug_id")[["drug_id", "inchi_key", "split"]].copy()
    known_keys = set(mapping["inchi_key"])
    representatives = compounds[compounds["inchi_key"].isin(known_keys)].sort_values("pert_id").drop_duplicates("inchi_key").reset_index(drop=True)
    representative_ids = dict(zip(representatives["inchi_key"], representatives["pert_id"]))
    mapping["pert_id"] = mapping["inchi_key"].map(representative_ids)
    for key, records in compounds[compounds["inchi_key"].isin(known_keys)].groupby("inchi_key"):
        vectors = fingerprints[[feature_lookup[identifier] for identifier in records["pert_id"]]]
        if not np.all(vectors == vectors[0]):
            raise ValueError("Same chemical identity has different fingerprints: " + key)
    responses = pd.read_csv(table_path(source, "responses"))
    if responses.duplicated(["drug_id", "cell_id"]).any():
        raise ValueError("Duplicate cleaned drug/cell rows")
    if not responses["sensitivity_rank"].between(0, 1).all():
        raise ValueError("Rank labels outside [0, 1]")
    original_ranks = responses.groupby("drug_id")["sensitivity_raw"].rank(pct=True, method="average")
    if not np.allclose(original_ranks, responses["sensitivity_rank"], atol=1e-8, rtol=0):
        raise ValueError("Cleaned rank labels do not reproduce from raw scores")
    if not np.isfinite(responses["sensitivity_raw"]).all() or not np.isfinite(responses["n_sources"]).all() or not responses["n_sources"].ge(1).all():
        raise ValueError("Invalid raw response values or source counts")
    if not responses["n_sources"].mod(1).eq(0).all():
        raise ValueError("Source counts must be integers")
    if not set(responses["drug_id"]) <= set(identifiers["drug_id"]):
        raise ValueError("Missing cleaned-to-frozen drug mappings")
    excluded_responses = responses[~responses["drug_id"].isin(mapping["drug_id"])]
    joined = responses.merge(mapping[["drug_id", "pert_id", "split"]], on="drug_id", validate="many_to_one")
    training_cells = set(joined.loc[joined["split"] == "train", "cell_id"])
    excluded_cells = sorted(set(joined["cell_id"]) - training_cells)
    joined = joined[joined["cell_id"].isin(training_cells)].copy()
    joined["weighted_raw"] = joined["sensitivity_raw"] * joined["n_sources"]
    merged = joined.groupby(["pert_id", "cell_id"], as_index=False).agg(
        weighted_raw=("weighted_raw", "sum"), n_sources=("n_sources", "sum"), cleaned_rows=("drug_id", "size"))
    merged["sensitivity_raw"] = merged["weighted_raw"] / merged["n_sources"]
    merged["sensitivity_rank"] = merged.groupby("pert_id")["sensitivity_raw"].rank(pct=True, method="average")
    representatives = representatives[representatives["pert_id"].isin(merged["pert_id"])].reset_index(drop=True)
    cells = np.array(sorted(training_cells), dtype=str)
    cell_metadata = pd.read_csv(table_path(source, "cells"), keep_default_na=False)
    if not cell_metadata["cell_id"].is_unique or not set(cells) <= set(cell_metadata["cell_id"]):
        raise ValueError("Missing or duplicated cell metadata")
    cell_metadata = cell_metadata.set_index("cell_id").loc[cells].reset_index()
    matrix = merged.pivot(index="pert_id", columns="cell_id", values="sensitivity_rank").reindex(
        index=representatives["pert_id"], columns=cells).to_numpy(dtype=np.float64)
    partitions = {name: np.flatnonzero(representatives["split"].to_numpy() == name) for name in ("train", "validation", "test")}
    if any(not len(indices) for indices in partitions.values()):
        raise ValueError("Empty benchmark partition")
    features = fingerprints[[feature_lookup[identifier] for identifier in representatives["pert_id"]]]
    expression = gene_features(source, cells, gene_count)
    raw_to_representative = dict(zip(mapping["drug_id"], mapping["pert_id"]))
    kept = []
    target_columns = ["drug_id", "target_chembl_id", "target_name", "activity_type", "pchembl_median"]
    for chunk in pd.read_csv(table_path(source, "drug_targets"), usecols=target_columns, chunksize=50000):
        chunk = chunk[chunk["drug_id"].isin(raw_to_representative)].copy()
        if len(chunk):
            chunk["pert_id"] = chunk["drug_id"].map(raw_to_representative)
            kept.append(chunk)
    if not kept:
        raise ValueError("No cleaned ChEMBL activities match the frozen compounds")
    targets = pd.concat(kept, ignore_index=True).rename(columns={"target_chembl_id": "target_id", "activity_type": "standard_type", "pchembl_median": "median_pchembl"})
    if not np.isfinite(targets["median_pchembl"]).all():
        raise ValueError("Non-finite cleaned target activity")
    targets = targets[targets["standard_type"].isin(["IC50", "Ki", "Kd", "EC50"])]
    targets = targets.groupby(["pert_id", "target_id", "standard_type"], as_index=False)["median_pchembl"].median()
    targets["assay_type"] = "source_pooled"
    targets["split"] = targets["pert_id"].map(representatives.set_index("pert_id")["split"])
    targets = targets.dropna(subset=["split"])
    output.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(output / "drug_mapping.csv", index=False)
    representatives.to_csv(output / "compound_split.csv", index=False)
    merged.drop(columns="weighted_raw").to_csv(output / "rank_responses.csv.gz", index=False)
    targets.to_csv(output / "endpoint_activity.csv", index=False)
    cell_metadata.to_csv(output / "cells.csv", index=False)
    feature_report = frozen / "feature_report.json"
    feature_configuration = json.loads(feature_report.read_text())["configuration"] if feature_report.exists() else {}
    report = {"source": str(source), "frozen_split_sha256": file_hash(manifest_path),
              "frozen_fingerprints_sha256": file_hash(frozen / "fingerprints.npz"),
              "frozen_feature_configuration": feature_configuration,
              "benchmark_split_sha256": file_hash(output / "compound_split.csv"),
              "source_response_rows": len(responses), "canonical_response_rows": len(merged),
              "source_drugs": int(responses["drug_id"].nunique()), "canonical_drugs": len(representatives),
              "excluded_source_drug_ids": sorted(mapping_exclusions["drug_id"].unique().tolist()),
              "excluded_response_rows_for_unresolved_structures": len(excluded_responses),
              "mapping_exclusion_report": "mapping_exclusions.csv",
              "cells": len(cells), "genes_selected": len(expression["gene_names"]), "fingerprint_bits": features.shape[1],
              "excluded_cells_without_training_labels": excluded_cells,
              "compound_counts": {name: len(indices) for name, indices in partitions.items()},
              "response_counts": {name: int(np.isfinite(matrix[indices]).sum()) for name, indices in partitions.items()},
              "cleaned_target_rows_in_benchmark": len(targets),
              "source_files": {name: {"bytes": table_path(source, name).stat().st_size, "sha256": file_hash(table_path(source, name))}
                               for name in ("responses", "drug_ids", "drugs", "gene_expression", "drug_targets", "cells")},
              "notes": ["Original scaffold assignments are preserved; canonical duplicate drugs are never split across partitions.",
                        "Entire source drug IDs with unresolved identity/split mappings, flagged structure conflicts, or source/frozen structure disagreement are excluded for every model before fitting.",
                        "Already-averaged response labels for ambiguous merged IDs cannot safely be reassigned to one of several structures; no first-match identity is selected.",
                        "Additional canonical duplicates are merged using source-count-weighted raw scores, then within-drug ranks are recomputed.",
                        "All models predict relative within-drug sensitivity ranks, not raw response or cell-kill probabilities.",
                        "Gene selection and projection use only cell lines with training labels; evaluation is compound-cold, cell-warm.",
                        "Known targets, MOA annotations, and the all-drug trust weights are excluded from predictive inputs.",
                        "Target supervision uses source-aggregated pChEMBL values; assay types were pooled upstream and cannot be recovered.",
                        "If canonical aliases have different source medians, their median is a representative summary, not a reconstruction of raw measurements."]}
    (output / "data_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return {"compounds": representatives, "fingerprints": features, "cells": cell_metadata, "matrix": matrix,
            "partitions": partitions, "expression": expression, "endpoint_path": output / "endpoint_activity.csv", "audit": report}
