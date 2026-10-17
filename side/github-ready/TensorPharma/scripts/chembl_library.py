import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rank_data import file_hash


def describe(smiles):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from challenge6_features import standardize, fingerprint

    try:
        molecule = standardize(smiles)
        key = Chem.MolToInchiKey(molecule)
        if not key:
            raise ValueError("Missing InChIKey")
        return (key, Chem.MolToSmiles(molecule, isomericSmiles=True),
                MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False) or "__ACYCLIC__",
                np.packbits(fingerprint(molecule)))
    except (ValueError, RuntimeError):
        return None


def build(source, frozen, output, workers):
    from rdkit import rdBase

    if (output / "manifest.json").exists():
        raise ValueError("Completed library exists; use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    split = pd.read_csv(frozen / "compound_split.csv", keep_default_na=False)
    for column in ("inchi_key", "connectivity_key", "scaffold"):
        if split.groupby(column)["split"].nunique().max() != 1:
            raise ValueError("Frozen split leakage: " + column)
    configuration = json.loads((frozen / "feature_report.json").read_text())["configuration"]
    if configuration["rdkit_version"] != rdBase.rdkitVersion:
        raise ValueError("Use the RDKit version from the frozen split")
    table = pd.read_csv(source, keep_default_na=False)
    potency = pd.to_numeric(table["pchembl_median"], errors="coerce")
    count = pd.to_numeric(table["n_measurements"], errors="coerce")
    valid = np.isfinite(potency) & (potency >= 6) & table["target_gene"].str.strip().ne("")
    if not (np.isfinite(count[valid]) & (count[valid] >= 1) & (count[valid] % 1 == 0)).all():
        raise ValueError("Invalid measurement counts")
    table = table.loc[valid].copy()
    table["pchembl_median"] = potency[valid]
    table["n_measurements"] = count[valid].astype(int)
    table["target_gene"] = table["target_gene"].str.strip()
    smiles = table["smiles"].drop_duplicates().tolist()
    descriptions = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for position, (text, result) in enumerate(zip(smiles, pool.map(describe, smiles, chunksize=64))):
            if result is not None:
                descriptions[text] = result
            if (position + 1) % 10000 == 0:
                print("Standardized", position + 1, "/", len(smiles), flush=True)
    table["inchi_key"] = table["smiles"].map(lambda value: descriptions[value][0] if value in descriptions else None)
    table = table.dropna(subset=["inchi_key"])
    known = {key: sorted(set(records["target_gene"])) for key, records in table.groupby("inchi_key")}
    queries = split.sort_values("pert_id").drop_duplicates("inchi_key").copy()
    queries["known_genes"] = queries["inchi_key"].map(lambda key: json.dumps(known.get(key, [])))
    queries.to_csv(output / "evaluation_queries.csv", index=False)
    heldout = split[split["split"].isin(["validation", "test"])]
    excluded_keys = set(split["inchi_key"])
    excluded_connectivity = set(split["connectivity_key"])
    excluded_scaffolds = set(heldout["scaffold"])
    reference = {}
    for result in descriptions.values():
        key, canonical, scaffold, packed = result
        if key not in excluded_keys and key[:14] not in excluded_connectivity and scaffold not in excluded_scaffolds:
            reference[key] = (canonical, packed)
    keys = sorted(reference)
    if not keys:
        raise ValueError("No reference molecules survive exclusions")
    key_index = {key: index for index, key in enumerate(keys)}
    evidence = table[table["inchi_key"].isin(key_index)].copy()
    evidence["compound_index"] = evidence["inchi_key"].map(key_index)
    evidence = evidence.drop_duplicates()
    evidence.to_csv(output / "evidence.csv.gz", index=False)
    labels = evidence.groupby(["compound_index", "target_gene"], sort=True).agg(
        pchembl_max=("pchembl_median", "max"), n_measurements_max=("n_measurements", "max"),
        source_rows=("target_gene", "size")).reset_index()
    labels.to_csv(output / "labels.csv.gz", index=False)
    np.save(output / "fingerprints.npy", np.stack([reference[key][1] for key in keys]))
    pd.DataFrame({"inchi_key": keys, "canonical_smiles": [reference[key][0] for key in keys]}).to_csv(output / "compounds.csv.gz", index=False)
    manifest = {"source_sha256": file_hash(source), "split_sha256": file_hash(frozen / "compound_split.csv"),
                "rdkit_version": rdBase.rdkitVersion, "standardization": configuration["standardization"],
                "fingerprint_bits": 2048, "radius": 2, "active_pchembl_minimum": 6,
                "source_unique_active_smiles": len(smiles), "unparseable_or_failed_smiles": len(smiles) - len(descriptions),
                "reference_compounds": len(keys), "reference_compound_gene_pairs": len(labels),
                "exclusion": "All frozen LINCS identities/connectivities, plus validation/test scaffolds",
                "notes": ["Source endpoints/assays were already pooled upstream; preserved evidence is not raw assay provenance.",
                          "Evaluation positives come from ChEMBL, not incomplete free-text known_targets annotations.",
                          "All LINCS structures are held out, including training structures. Test scaffolds are also excluded."]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


class Library:
    def __init__(self, path):
        self.manifest = json.loads((path / "manifest.json").read_text())
        self.fingerprints = np.load(path / "fingerprints.npy", mmap_mode="r", allow_pickle=False)
        self.popcount = np.array([value.bit_count() if hasattr(value, "bit_count") else bin(value).count("1") for value in range(256)])
        self.counts = self.popcount[self.fingerprints].sum(axis=1)
        self.labels = pd.read_csv(path / "labels.csv.gz", keep_default_na=False)
        self.genes = sorted(self.labels["target_gene"].unique())
        lookup = {gene: index for index, gene in enumerate(self.genes)}
        self.columns = self.labels["target_gene"].map(lookup).to_numpy()
        self.rows = self.labels["compound_index"].to_numpy()
        self.gene_indices = self.labels.groupby("target_gene").indices

    def similarity(self, query):
        values = np.empty(len(self.fingerprints), dtype=np.float32)
        query_count = self.popcount[query].sum()
        for start in range(0, len(values), 8192):
            stop = min(start + 8192, len(values))
            intersection = self.popcount[np.bitwise_and(self.fingerprints[start:stop], query)].sum(axis=1)
            union = self.counts[start:stop] + query_count - intersection
            values[start:stop] = np.divide(intersection, union, out=np.zeros(len(union), dtype=float), where=union > 0)
        return values

    def score(self, similarity, cutoff, trust=False, limit=20):
        weights = np.where(similarity >= cutoff, similarity, 0)
        reliability = np.where(self.labels["n_measurements_max"].to_numpy() >= 2, 1.0, 0.26) if trust else np.ones(len(self.rows))
        votes = weights[self.rows] * reliability
        totals = np.bincount(self.columns, weights=votes, minlength=len(self.genes))
        order = sorted(np.flatnonzero(totals > 0), key=lambda index: (-totals[index], self.genes[index]))[:limit]
        records = []
        for index in order:
            indices = self.gene_indices[self.genes[index]]
            supported = indices[votes[indices] > 0]
            records.append({"gene": self.genes[index], "score": float(totals[index]),
                            "nearest_neighbor_similarity": float(similarity[self.rows[supported]].max()),
                            "supporting_pchembl_max": float(self.labels.iloc[supported]["pchembl_max"].max()),
                            "supporting_compounds": len(supported),
                            "source_rows": int(self.labels.iloc[supported]["source_rows"].sum()), "abstained": False})
        return records


def query_fingerprint(text, manifest):
    from rdkit import rdBase

    if rdBase.rdkitVersion != manifest["rdkit_version"]:
        raise ValueError("Query RDKit version differs from library")
    result = describe(text)
    if result is None:
        raise ValueError("Invalid query SMILES: " + text)
    return result[3]


def metrics(records):
    if not records:
        raise ValueError("No ChEMBL-labelled evaluation queries")
    return {"labelled_structures": len(records), "hit_at_5": float(np.mean([record[0] for record in records])),
            "macro_known_target_recall_at_5": float(np.mean([record[1] for record in records])),
            "abstentions": sum(record[2] for record in records)}


def evaluate(path, output):
    if output.exists():
        raise ValueError("Evaluation output exists; do not retune on test results")
    library = Library(path)
    queries = pd.read_csv(path / "evaluation_queries.csv", keep_default_na=False)
    candidates = [(cutoff, trust) for cutoff in (0.3, 0.4, 0.5, 0.6) for trust in (False, True)]
    validation = {str(candidate): [] for candidate in candidates}
    coverage = {}
    chosen = None
    test = []
    for partition in ("validation", "test"):
        subset = queries[queries["split"] == partition]
        coverage[partition] = {"total_structures": len(subset), "with_known_active_genes": 0}
        for record in subset.to_dict("records"):
            truth = set(json.loads(record["known_genes"]))
            if not truth:
                continue
            coverage[partition]["with_known_active_genes"] += 1
            similarity = library.similarity(query_fingerprint(record["standardized_smiles"], library.manifest))
            for candidate in candidates if partition == "validation" else [chosen]:
                predictions = library.score(similarity, *candidate, limit=5)
                genes = {prediction["gene"] for prediction in predictions}
                result = (bool(genes & truth), len(genes & truth) / len(truth), not genes)
                if partition == "validation":
                    validation[str(candidate)].append(result)
                else:
                    test.append(result)
        if partition == "validation":
            scores = {name: metrics(records) for name, records in validation.items()}
            chosen = max(candidates, key=lambda candidate: (scores[str(candidate)]["macro_known_target_recall_at_5"], scores[str(candidate)]["hit_at_5"]))
            print("Validation-selected cutoff/trust:", chosen, flush=True)
    report = {"selected_cutoff": chosen[0], "selected_trust": chosen[1], "validation_candidates": scores,
              "test": metrics(test), "coverage": coverage, "library_manifest": library.manifest,
              "notes": ["Missing labels are unknown, not negatives. Precision is not estimated.",
                        "Scores are similarity votes, not probabilities or binding predictions.",
                        "0.26 is an optional heuristic, not a validated reliability probability.",
                        "Test compounds were inspected in previous experiments; these results are exploratory."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def score_file(path, queries, output, selection, shard, shards):
    library = Library(path)
    selected = json.loads(selection.read_text())
    if selected["library_manifest"] != library.manifest:
        raise ValueError("Selected configuration belongs to another library")
    frame = pd.read_csv(queries, keep_default_na=False)
    if not frame["query_id"].is_unique:
        raise ValueError("Query IDs must be unique")
    records = []
    for row in frame.iloc[shard::shards].to_dict("records"):
        similarity = library.similarity(query_fingerprint(row["smiles"], library.manifest))
        predictions = library.score(similarity, selected["selected_cutoff"], selected["selected_trust"])
        if not predictions:
            predictions = [{"gene": "", "score": 0, "abstained": True}]
        records.extend({"query_id": row["query_id"], **prediction} for prediction in predictions)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records, columns=["query_id", "gene", "score", "nearest_neighbor_similarity", "supporting_pchembl_max", "supporting_compounds", "source_rows", "abstained"]).to_csv(output, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    builder = commands.add_parser("build")
    builder.add_argument("--source", type=Path, required=True)
    builder.add_argument("--frozen", type=Path, required=True)
    builder.add_argument("--output", type=Path, required=True)
    builder.add_argument("--workers", type=int, default=8)
    evaluator = commands.add_parser("evaluate")
    evaluator.add_argument("--library", type=Path, required=True)
    evaluator.add_argument("--output", type=Path, required=True)
    scorer = commands.add_parser("score")
    scorer.add_argument("--library", type=Path, required=True)
    scorer.add_argument("--queries", type=Path, required=True)
    scorer.add_argument("--output", type=Path, required=True)
    scorer.add_argument("--selection", type=Path, required=True)
    scorer.add_argument("--shard", type=int, default=0)
    scorer.add_argument("--shards", type=int, default=1)
    arguments = parser.parse_args()
    if arguments.command == "build":
        build(arguments.source, arguments.frozen, arguments.output, arguments.workers)
    elif arguments.command == "evaluate":
        evaluate(arguments.library, arguments.output)
    else:
        if not 0 <= arguments.shard < arguments.shards:
            parser.error("Require 0 <= shard < shards")
        score_file(arguments.library, arguments.queries, arguments.output, arguments.selection, arguments.shard, arguments.shards)
