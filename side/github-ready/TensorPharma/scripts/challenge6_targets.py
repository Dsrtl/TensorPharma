import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from challenge6_baseline import similarities


def score_targets(query, references, positive_labels, prior, neighbors):
    if neighbors == 0:
        return np.broadcast_to(prior, (len(query), len(prior))).copy()
    similarity = similarities(query, references)
    scores = np.zeros((len(query), positive_labels.shape[1]), dtype=np.float64)
    for index, values in enumerate(similarity):
        selected = np.argsort(-values, kind="stable")[:neighbors]
        weights = values[selected] ** 2
        if weights.sum() > 0:
            scores[index] = weights @ positive_labels[selected] / weights.sum()
    return scores


def evaluate_targets(truth, scores, target_ids):
    if not truth:
        raise ValueError("No labelled evaluation structures")
    hits = []
    recalls = []
    top_one = []
    reachable = []
    abstentions = 0
    vocabulary = set(target_ids)
    predictions = []
    for known, values in zip(truth, scores):
        if not known:
            raise ValueError("Evaluation requires at least one known active target per query")
        ranking = np.argsort(-values, kind="stable")
        ranking = [index for index in ranking[:5] if values[index] > 0]
        predicted = [str(target_ids[index]) for index in ranking]
        overlap = len(known & set(predicted))
        hits.append(float(overlap > 0))
        recalls.append(overlap / len(known))
        top_one.append(float(bool(predicted) and predicted[0] in known))
        reachable.append(min(5, len(known & vocabulary)) / len(known))
        abstentions += not predicted
        predictions.append({"known_active_targets": ";".join(sorted(known)), "top_five_targets": ";".join(predicted)})
    return {"unique_labelled_structures": len(truth), "hit_at_1": float(np.mean(top_one)),
            "hit_at_5": float(np.mean(hits)), "macro_known_target_recall_at_5": float(np.mean(recalls)),
            "maximum_recall_at_5_given_training_vocabulary": float(np.mean(reachable)),
            "queries_without_positive_votes": int(abstentions)}, predictions


def train(root):
    features = root / "data" / "features"
    matched = root / "data" / "chembl_matched"
    split_path = features / "compound_split.csv"
    split_hash = hashlib.sha256(split_path.read_bytes()).hexdigest()
    matching_report = json.loads((matched / "matching_report.json").read_text())
    if matching_report["split_sha256"] != split_hash:
        raise ValueError("Target matching and frozen split differ")
    compounds = pd.read_csv(split_path, keep_default_na=False)
    for column in ("inchi_key", "connectivity_key", "scaffold"):
        if compounds.groupby(column)["split"].nunique().max() != 1:
            raise ValueError("Cross-partition leakage in " + column)
    with np.load(features / "fingerprints.npz", allow_pickle=False) as archive:
        if not np.array_equal(archive["pert_ids"], compounds["pert_id"].to_numpy()):
            raise ValueError("Fingerprint and compound ordering differ")
        fingerprints = archive["fingerprints"].copy()
    feature_lookup = {identifier: index for index, identifier in enumerate(compounds["pert_id"])}
    labels = pd.read_csv(matched / "target_labels.csv", keep_default_na=False)
    labels = labels.merge(compounds[["pert_id", "inchi_key"]], on="pert_id", validate="many_to_one")
    active = labels[labels["known_active"].astype(str).str.lower().isin(["true", "1"])]
    known_targets = {key: set(records["target_id"]) for key, records in active.groupby("inchi_key")}
    unique = compounds.sort_values("pert_id").drop_duplicates("inchi_key").reset_index(drop=True)
    unique = unique[unique["inchi_key"].isin(known_targets)].copy()
    training = unique[unique["split"] == "train"]
    if len(training) < 2:
        raise ValueError("At least two training structures with known active targets are required")
    target_ids = np.array(sorted(set().union(*(known_targets[key] for key in training["inchi_key"]))), dtype=str)
    target_lookup = {identifier: index for index, identifier in enumerate(target_ids)}
    positive_labels = np.zeros((len(training), len(target_ids)), dtype=np.float32)
    for row_index, key in enumerate(training["inchi_key"]):
        for target in known_targets[key]:
            positive_labels[row_index, target_lookup[target]] = 1
    training_features = fingerprints[[feature_lookup[identifier] for identifier in training["pert_id"]]]
    prior = positive_labels.mean(axis=0)
    partitions = {}
    for name in ("validation", "test"):
        queries = unique[unique["split"] == name]
        if queries.empty:
            raise ValueError("No " + name + " structures have known active target labels")
        partitions[name] = {
            "compounds": queries,
            "features": fingerprints[[feature_lookup[identifier] for identifier in queries["pert_id"]]],
            "truth": [known_targets[key] for key in queries["inchi_key"]],
        }
    candidates = {"most_common_targets": 0, "tanimoto_targets_1": 1, "tanimoto_targets_3": 3,
                  "tanimoto_targets_5": 5, "tanimoto_targets_10": 10}
    validation = {}
    for name, neighbors in candidates.items():
        scores = score_targets(partitions["validation"]["features"], training_features, positive_labels, prior, neighbors)
        validation[name], _ = evaluate_targets(partitions["validation"]["truth"], scores, target_ids)
    selected = max(candidates, key=lambda name: (validation[name]["macro_known_target_recall_at_5"], validation[name]["hit_at_5"]))
    output = root / "results" / "targets"
    output.mkdir(parents=True, exist_ok=True)
    compared = list(dict.fromkeys(["most_common_targets", selected]))
    evaluation = {}
    for name in compared:
        scores = score_targets(partitions["test"]["features"], training_features, positive_labels, prior, candidates[name])
        test_metrics, predictions = evaluate_targets(partitions["test"]["truth"], scores, target_ids)
        evaluation[name] = {"validation": validation[name], "test": test_metrics}
        if name == selected:
            frame = pd.DataFrame(predictions)
            frame.insert(0, "pert_id", partitions["test"]["compounds"]["pert_id"].to_numpy())
            frame.to_csv(output / "test_target_predictions.csv", index=False)
    names = labels.drop_duplicates("target_id").set_index("target_id")["target_name"].to_dict()
    np.savez_compressed(
        output / "target_baseline.npz", selected_model=np.array(selected), neighbors=np.array(candidates[selected]),
        target_ids=target_ids, target_names=np.array([names.get(identifier, identifier) for identifier in target_ids], dtype=str),
        training_ids=training["pert_id"].to_numpy(dtype=str), training_names=training["name"].to_numpy(dtype=str),
        training_smiles=training["standardized_smiles"].to_numpy(dtype=str),
        training_fingerprints=training_features, positive_labels=positive_labels, prior_scores=prior,
        split_sha256=np.array(split_hash))
    evidence = pd.read_csv(matched / "matched_activities.csv.gz")
    evidence = evidence[evidence["split"] == "train"]
    evidence.to_csv(output / "training_activity_evidence.csv.gz", index=False)
    report = {"selected_model": selected, "split_sha256": split_hash,
              "selection_rule": "Highest validation macro known-target recall@5; ties use hit@5, then fixed candidate order",
              "training_unique_structures": len(training), "training_target_vocabulary": len(target_ids),
              "coverage_by_split": matching_report["coverage_by_split"],
              "validation_candidates": validation, "models": evaluation,
              "notes": ["References contain only matched training structures; validation/test scaffolds are excluded.",
                        "LINCS aliases are collapsed by full standardized InChIKey for training and evaluation.",
                        "Evaluation covers only structures with at least one known active target; inspect coverage separately.",
                        "Targets outside the training vocabulary count as misses, not silently excluded labels.",
                        "Unmeasured targets are unknown. Hit/recall use incomplete known-positive labels; precision is not estimated.",
                        "Scores are similarity-weighted votes, not binding affinities or calibrated probabilities.",
                        "The prior baseline is molecule-independent. Similarity uses the nearest positive-labelled training structures.",
                        "All 655,004 reference structures are not used as neighbours; this initial baseline uses the matched training subset.",
                        "Target test labels are not used for model selection. These compounds were previously evaluated for sensitivity."]}
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    arguments = parser.parse_args()
    train(arguments.root.resolve())
