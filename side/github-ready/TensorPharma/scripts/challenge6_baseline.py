import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import solve, svd
from scipy.stats import spearmanr


def cell_means(responses):
    fallback = float(np.nanmean(responses))
    counts = np.isfinite(responses).sum(axis=0)
    return np.divide(np.nansum(responses, axis=0), counts,
                     out=np.full(responses.shape[1], fallback, dtype=np.float64), where=counts > 0)


def similarities(query, reference):
    query = np.asarray(query, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    intersection = query @ reference.T
    union = query.sum(axis=1)[:, None] + reference.sum(axis=1)[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def neighbor_predictions(query, reference, responses, means, neighbors=5):
    similarity = similarities(query, reference)
    predictions = np.empty((len(query), responses.shape[1]), dtype=np.float64)
    for index, weights in enumerate(similarity):
        order = np.argsort(-weights, kind="stable")
        observed = np.isfinite(responses[order])
        selected = observed & (np.cumsum(observed, axis=0) <= neighbors)
        weighted = selected * weights[order, None] ** 2
        denominator = weighted.sum(axis=0)
        numerator = (weighted * np.nan_to_num(responses[order], nan=0.0)).sum(axis=0)
        predictions[index] = np.divide(numerator, denominator, out=means.copy(), where=denominator > 0)
    return predictions


def fit_projection(features, dimensions=64):
    center = features.mean(axis=0, dtype=np.float64)
    _, singular_values, directions = svd(features - center, full_matrices=False, check_finite=True)
    rank = min(dimensions, int((singular_values > 1e-8).sum()))
    if rank == 0:
        raise ValueError("Training fingerprints have no variation")
    return center, directions[:rank].T


def fit_ridge(embedding, responses, means, regularization):
    coefficients = np.zeros((embedding.shape[1], responses.shape[1]), dtype=np.float64)
    intercepts = means.copy()
    identity = np.eye(embedding.shape[1])
    for cell_index in range(responses.shape[1]):
        observed = np.isfinite(responses[:, cell_index])
        if not observed.any():
            continue
        design = embedding[observed]
        values = responses[observed, cell_index]
        design_mean = design.mean(axis=0)
        value_mean = float(values.mean())
        centered = design - design_mean
        coefficients[:, cell_index] = solve(
            centered.T @ centered + regularization * identity,
            centered.T @ (values - value_mean), assume_a="pos")
        intercepts[cell_index] = value_mean - design_mean @ coefficients[:, cell_index]
    return coefficients, intercepts


def metrics(truth, predictions):
    observed = np.isfinite(truth)
    if not observed.any() or not np.isfinite(predictions).all():
        raise ValueError("Missing evaluation labels or non-finite predictions")
    errors = predictions[observed] - truth[observed]
    correlations = []
    for cell_index in range(truth.shape[1]):
        selected = observed[:, cell_index]
        actual = truth[selected, cell_index]
        predicted = predictions[selected, cell_index]
        if len(actual) >= 5 and np.ptp(actual) > 0 and np.ptp(predicted) > 1e-10:
            correlation = float(spearmanr(actual, predicted).correlation)
            if np.isfinite(correlation):
                correlations.append(correlation)
    return {"observations": int(observed.sum()), "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "mae": float(np.mean(np.abs(errors))),
            "median_within_cell_spearman": float(np.median(correlations)) if correlations else None,
            "cells_with_defined_spearman": len(correlations)}


def train(root, feature_dir, output):
    compounds = pd.read_csv(feature_dir / "compound_split.csv", keep_default_na=False)
    with np.load(feature_dir / "fingerprints.npz", allow_pickle=False) as archive:
        if not np.array_equal(archive["pert_ids"], compounds["pert_id"].to_numpy()):
            raise ValueError("Fingerprint and split compound ordering differs")
        fingerprints = archive["fingerprints"].astype(np.float64)
    responses = pd.read_csv(root / "data" / "prepared" / "matched_responses.csv")
    responses = responses[responses["pert_id"].isin(compounds["pert_id"])]
    if responses.duplicated(["pert_id", "model_id"]).any():
        raise ValueError("Duplicate compound/cell pairs require an explicit aggregation policy")
    cells = np.array(sorted(responses["model_id"].unique()), dtype=str)
    matrix = responses.pivot(index="pert_id", columns="model_id", values="sensitivity_score")
    matrix = matrix.reindex(index=compounds["pert_id"], columns=cells).to_numpy(dtype=np.float64)
    partitions = {name: np.flatnonzero(compounds["split"].to_numpy() == name)
                  for name in ("train", "validation", "test")}
    for name, selected in partitions.items():
        if not len(selected) or not np.isfinite(matrix[selected]).any():
            raise ValueError("Empty partition: " + name)
    training = partitions["train"]
    validation = partitions["validation"]
    testing = partitions["test"]
    seen_cells = np.isfinite(matrix[training]).any(axis=0)
    excluded_cells = cells[~seen_cells].tolist()
    cells = cells[seen_cells]
    matrix = matrix[:, seen_cells]
    if not len(cells):
        raise ValueError("No cell lines have training observations")
    training_responses = matrix[training]
    means = cell_means(training_responses)
    print("Fitting fingerprint projection on training compounds only", flush=True)
    center, projection = fit_projection(fingerprints[training])
    embedded = (fingerprints - center) @ projection
    candidates = {}
    fitted = {}
    candidates["cell_mean"] = np.broadcast_to(means, matrix[validation].shape).copy()
    candidates["tanimoto_knn_5"] = neighbor_predictions(
        fingerprints[validation], fingerprints[training], training_responses, means)
    ridge_validation = {}
    for regularization in (1.0, 10.0, 100.0):
        name = "fingerprint_ridge_" + str(int(regularization))
        fitted[name] = fit_ridge(embedded[training], training_responses, means, regularization)
        coefficients, intercepts = fitted[name]
        predictions = embedded[validation] @ coefficients + intercepts
        candidates[name] = predictions
        ridge_validation[str(regularization)] = metrics(matrix[validation], predictions)
        print(name, json.dumps(ridge_validation[str(regularization)]), flush=True)
    validation_metrics = {name: metrics(matrix[validation], prediction) for name, prediction in candidates.items()}
    best_ridge = min(fitted, key=lambda name: validation_metrics[name]["rmse"])
    compared = ["cell_mean", "tanimoto_knn_5", best_ridge]
    selected_model = min(compared, key=lambda name: validation_metrics[name]["rmse"])
    test_predictions = {
        "cell_mean": np.broadcast_to(means, matrix[testing].shape).copy(),
        "tanimoto_knn_5": neighbor_predictions(
            fingerprints[testing], fingerprints[training], training_responses, means),
        best_ridge: embedded[testing] @ fitted[best_ridge][0] + fitted[best_ridge][1],
    }
    evaluation = {name: {"validation": validation_metrics[name], "test": metrics(matrix[testing], prediction)}
                  for name, prediction in test_predictions.items()}
    output.mkdir(parents=True, exist_ok=True)
    coefficients, intercepts = fitted[best_ridge]
    split_hash = hashlib.sha256((feature_dir / "compound_split.csv").read_bytes()).hexdigest()
    model_path = output / "sensitivity_baseline.npz"
    np.savez_compressed(
        model_path, selected_model=np.array(selected_model), ridge_model=np.array(best_ridge),
        model_ids=cells, training_ids=compounds.iloc[training]["pert_id"].to_numpy(dtype=str),
        training_names=compounds.iloc[training]["name"].to_numpy(dtype=str),
        training_smiles=compounds.iloc[training]["standardized_smiles"].to_numpy(dtype=str),
        training_fingerprints=fingerprints[training].astype(np.uint8), training_responses=training_responses,
        cell_means=means, fingerprint_center=center, projection=projection,
        coefficients=coefficients, intercepts=intercepts, split_sha256=np.array(split_hash))
    selected_predictions = test_predictions[selected_model]
    row_indices, cell_indices = np.where(np.isfinite(matrix[testing]))
    pd.DataFrame({"pert_id": compounds.iloc[testing[row_indices]]["pert_id"].to_numpy(),
                  "model_id": cells[cell_indices], "observed_score": matrix[testing][row_indices, cell_indices],
                  "predicted_score": selected_predictions[row_indices, cell_indices]}).to_csv(
                      output / "test_predictions.csv", index=False)
    report = {
        "selected_model": selected_model, "selection_rule": "Lowest validation RMSE; test labels not used for selection",
        "split_sha256": split_hash, "compound_counts": {name: len(indices) for name, indices in partitions.items()},
        "observations": {name: int(np.isfinite(matrix[indices]).sum()) for name, indices in partitions.items()},
        "cell_lines": len(cells), "excluded_cells_without_training_labels": excluded_cells,
        "projection_dimensions": projection.shape[1],
        "ridge_tuning_validation_only": ridge_validation, "models": evaluation,
        "notes": ["Compound/scaffold-cold, cell-warm evaluation; not an unseen-cell-line test.",
                  "Projection, regression, and mean estimates use training labels/compounds only.",
                  "Model artifact retains train-only fit so deployed predictions match the reported evaluation.",
                  "Sensitivity score units/direction require dataset confirmation; scores are not probabilities.",
                  "Protein-target prediction, cancer labels, and the multilayer comparison are pending.",
                  "Constant per-cell predictions have undefined within-cell Spearman, reported as null."]}
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--features", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    train(arguments.root, arguments.features or arguments.root / "data" / "features",
          arguments.output or arguments.root / "results" / "baseline")
