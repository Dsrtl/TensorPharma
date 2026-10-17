import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import solve

from challenge6_baseline import metrics
from challenge6_multilayer import expression_features


def target_heads(root, compounds, training, endpoint_path=None):
    endpoints = pd.read_csv(endpoint_path or root / "data" / "chembl_matched" / "endpoint_activity.csv", keep_default_na=False)
    endpoints = endpoints.merge(compounds[["pert_id", "inchi_key", "split"]], on="pert_id", suffixes=("_label", ""), validate="many_to_one")
    if not (endpoints["split_label"] == endpoints["split"]).all():
        raise ValueError("Target labels and compound partitions differ")
    columns = ["inchi_key", "target_id", "standard_type", "assay_type"]
    endpoints = endpoints.groupby(columns, dropna=False)["median_pchembl"].median().reset_index()
    training_keys = set(compounds.iloc[training]["inchi_key"])
    reference = endpoints[endpoints["inchi_key"].isin(training_keys)]
    head_columns = ["target_id", "standard_type", "assay_type"]
    summary = reference.groupby(head_columns)["median_pchembl"].agg(["count", "std"])
    retained = summary[(summary["count"] >= 3) & (summary["std"] > 1e-6)].sort_index()
    heads = list(retained.index)
    if not heads:
        raise ValueError("No variable target/endpoint heads with at least three unique training structures")
    lookup = {head: index for index, head in enumerate(heads)}
    representatives = compounds.reset_index().sort_values("pert_id").drop_duplicates("inchi_key")
    row_lookup = dict(zip(representatives["inchi_key"], representatives["index"]))
    matrix = np.full((len(compounds), len(heads)), np.nan)
    for record in endpoints.to_dict("records"):
        head = (record["target_id"], record["standard_type"], record["assay_type"])
        if head in lookup:
            matrix[row_lookup[record["inchi_key"]], lookup[head]] = float(record["median_pchembl"])
    training_matrix = matrix[training]
    means = np.nanmean(training_matrix, axis=0)
    scales = np.nanstd(training_matrix, axis=0)
    if not np.isfinite(means).all() or not np.all(scales > 1e-6):
        raise ValueError("Invalid target training normalization")
    return matrix, means, scales, np.array(["|".join(head) for head in heads], dtype=str)


def fit_joint(features, expression, normalized_labels, weights, validation_features,
              validation_truth, means, response_scale, cell_count, rank, epochs, name, seed=2026,
              compound_regularization=10.0, output_regularization=1.0, validation_objective=None):
    if not np.isfinite([compound_regularization, output_regularization]).all() or min(compound_regularization, output_regularization) <= 0:
        raise ValueError("Regularization must be positive and finite")
    generator = np.random.default_rng(seed)
    observed = np.isfinite(normalized_labels)
    labels = np.nan_to_num(normalized_labels, nan=0.0)
    observation_weights = observed * weights
    compound_map = generator.normal(0, 0.05, (features.shape[1], rank))
    output_factors = generator.normal(0, 0.05, (labels.shape[1], rank))
    output_factors[:, 0] = 1
    expression_map = np.zeros((expression.shape[1], rank))
    best = None
    best_error = np.inf
    stalled = 0
    history = []
    for epoch in range(1, epochs + 1):
        embedding = features @ compound_map
        residual = observation_weights * (labels - embedding @ output_factors.T)
        for factor in range(rank):
            partial = residual + observation_weights * np.outer(embedding[:, factor], output_factors[:, factor])
            row_weights = observation_weights @ (output_factors[:, factor] ** 2)
            gram = features.T @ (row_weights[:, None] * features)
            right = features.T @ (partial @ output_factors[:, factor])
            compound_map[:, factor] = solve(gram + compound_regularization * np.eye(gram.shape[0]), right, assume_a="pos")
            embedding[:, factor] = features @ compound_map[:, factor]
            residual = partial - observation_weights * np.outer(embedding[:, factor], output_factors[:, factor])
        prior = expression @ expression_map
        for output_index in range(labels.shape[1]):
            selected = observed[:, output_index]
            design = embedding[selected]
            weight = weights[output_index]
            coupling = 10.0 if output_index < cell_count else 0.0
            gram = weight * design.T @ design + (output_regularization + coupling) * np.eye(rank)
            right = weight * design.T @ labels[selected, output_index]
            if coupling:
                right += coupling * prior[output_index]
            output_factors[output_index] = solve(gram, right, assume_a="pos")
        expression_map = solve(expression.T @ expression + 0.1 * np.eye(expression.shape[1]),
                               expression.T @ output_factors[:cell_count], assume_a="pos")
        residual = labels - embedding @ output_factors.T
        objective = float(np.sum(observation_weights * residual ** 2) + compound_regularization * np.sum(compound_map ** 2)
                          + output_regularization * np.sum(output_factors ** 2) + np.sum(expression_map ** 2)
                          + 10 * np.sum((output_factors[:cell_count] - expression @ expression_map) ** 2))
        predictions = validation_features @ compound_map @ output_factors[:cell_count].T * response_scale + means
        valid = np.isfinite(validation_truth)
        error = float(np.sqrt(np.mean((predictions[valid] - validation_truth[valid]) ** 2)))
        selection_loss = error if validation_objective is None else float(validation_objective(predictions))
        if not np.isfinite(error) or not np.isfinite(objective) or not np.isfinite(selection_loss):
            raise ValueError("Non-finite joint optimization result")
        history.append({"epoch": epoch, "training_objective": objective, "validation_rmse": error})
        if validation_objective is not None:
            history[-1]["selection_loss"] = selection_loss
        print(json.dumps({"model": name, **history[-1]}), flush=True)
        if selection_loss < best_error - 1e-8:
            best_error = selection_loss
            best = {"compound_map": compound_map.copy(), "output_factors": output_factors.copy(),
                    "expression_map": expression_map.copy(), "epoch": epoch}
            stalled = 0
        else:
            stalled += 1
        if stalled >= 5:
            break
    if best is None:
        raise ValueError("No finite joint checkpoint")
    return best, history


def train(root, rank=8, epochs=25, compound_regularization=10.0, output_regularization=1.0,
          validation_only=False, output=None):
    feature_dir = root / "data" / "features"
    baseline_dir = root / "results" / "baseline"
    split_path = feature_dir / "compound_split.csv"
    split_hash = hashlib.sha256(split_path.read_bytes()).hexdigest()
    baseline_metrics = json.loads((baseline_dir / "metrics.json").read_text())
    matching_report = json.loads((root / "data" / "chembl_matched" / "matching_report.json").read_text())
    if split_hash != baseline_metrics["split_sha256"] or split_hash != matching_report["split_sha256"]:
        raise ValueError("Baseline, target labels, and joint model splits differ")
    compounds = pd.read_csv(split_path, keep_default_na=False)
    for column in ("inchi_key", "connectivity_key", "scaffold"):
        if compounds.groupby(column)["split"].nunique().max() != 1:
            raise ValueError("Split leakage in " + column)
    with np.load(feature_dir / "fingerprints.npz", allow_pickle=False) as archive:
        if not np.array_equal(archive["pert_ids"], compounds["pert_id"].to_numpy()):
            raise ValueError("Fingerprint order differs")
        fingerprints = archive["fingerprints"].astype(np.float64)
    with np.load(baseline_dir / "sensitivity_baseline.npz", allow_pickle=False) as archive:
        baseline = {key: archive[key] for key in archive.files}
    if str(baseline["split_sha256"].item()) != split_hash:
        raise ValueError("Baseline artifact split differs")
    partitions = {name: np.flatnonzero(compounds["split"].to_numpy() == name) for name in ("train", "validation", "test")}
    training = partitions["train"]
    if not np.array_equal(compounds.iloc[training]["pert_id"].to_numpy(), baseline["training_ids"]):
        raise ValueError("Baseline training ordering differs")
    responses = pd.read_csv(root / "data" / "prepared" / "matched_responses.csv")
    if responses.duplicated(["pert_id", "model_id"]).any():
        raise ValueError("Duplicate compound/cell observations")
    sensitivity = responses.pivot(index="pert_id", columns="model_id", values="sensitivity_score").reindex(
        index=compounds["pert_id"], columns=baseline["model_ids"]).to_numpy(dtype=np.float64)
    if not np.allclose(sensitivity[training], baseline["training_responses"], equal_nan=True):
        raise ValueError("Baseline training labels changed")
    means = baseline["cell_means"]
    training_sensitivity = sensitivity[training]
    response_scale = float(np.sqrt(np.nanmean((training_sensitivity - means) ** 2)))
    if response_scale <= 1e-12:
        raise ValueError("Training sensitivity residuals have no variation")
    embedded = (fingerprints - baseline["fingerprint_center"]) @ baseline["projection"]
    projection_scale = np.maximum(embedded[training].std(axis=0), 1e-8)
    embedded /= projection_scale
    expression, expression_state = expression_features(root / "data" / "expression" / "ccle_matched.npz", baseline["model_ids"])
    activity, target_means, target_scales, head_ids = target_heads(root, compounds, training)
    cell_count = sensitivity.shape[1]
    normalized = np.column_stack([(training_sensitivity - means) / response_scale,
                                  (activity[training] - target_means) / target_scales])
    activity_weight = float(np.isfinite(training_sensitivity).sum() / np.isfinite(activity[training]).sum())
    print(json.dumps({"target_heads": len(head_ids), "training_target_observations": int(np.isfinite(activity[training]).sum()),
                      "target_loss_weight": activity_weight}), flush=True)
    fits = {}
    histories = {}
    validation_metrics = {}
    for name, target_weight in (("expression_without_targets", 0.0), ("joint_targets_expression", activity_weight)):
        weights = np.concatenate([np.ones(cell_count), np.full(len(head_ids), target_weight)])
        fitted, history = fit_joint(embedded[training], expression, normalized, weights,
                                    embedded[partitions["validation"]], sensitivity[partitions["validation"]],
                                    means, response_scale, cell_count, rank, epochs, name,
                                    compound_regularization=compound_regularization, output_regularization=output_regularization)
        fits[name] = fitted
        histories[name] = history
        predicted = embedded[partitions["validation"]] @ fitted["compound_map"] @ fitted["output_factors"][:cell_count].T * response_scale + means
        validation_metrics[name] = metrics(sensitivity[partitions["validation"]], predicted)
    selected = min(fits, key=lambda name: validation_metrics[name]["rmse"])
    baseline_name = baseline_metrics["selected_model"]
    baseline_error = baseline_metrics["models"][baseline_name]["validation"]["rmse"]
    candidate_only = validation_metrics[selected]["rmse"] >= baseline_error
    if validation_only:
        directory = output or root / "results" / "joint_validation"
        directory.mkdir(parents=True, exist_ok=True)
        report = {"selected_model": selected, "split_sha256": split_hash, "validation_only": True,
                  "baseline_validation_rmse": baseline_error, "validation_winner_including_baseline": baseline_name if candidate_only else selected,
                  "configuration": {"rank": rank, "max_epochs": epochs, "patience": 5, "seed": 2026,
                                    "compound_regularization": compound_regularization, "output_regularization": output_regularization,
                                    "expression_coupling": 10.0, "expression_map_regularization": 1.0, "target_loss_weight": activity_weight},
                  "models": {name: {"validation": validation_metrics[name], "best_epoch": fitted["epoch"]} for name, fitted in fits.items()},
                  "notes": ["No test metrics are computed and no prediction artifact is published in validation-only mode.",
                            "Only regularization strengths change; the frozen split, target heads, and loss weighting stay fixed."]}
        (directory / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
        (directory / "optimization_history.json").write_text(json.dumps(histories, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        return
    evaluation = {}
    for name, fitted in fits.items():
        validation_prediction = embedded[partitions["validation"]] @ fitted["compound_map"] @ fitted["output_factors"][cell_count:].T * target_scales + target_means
        test_prediction = embedded[partitions["test"]] @ fitted["compound_map"] @ fitted["output_factors"][cell_count:].T * target_scales + target_means
        test_sensitivity = embedded[partitions["test"]] @ fitted["compound_map"] @ fitted["output_factors"][:cell_count].T * response_scale + means
        evaluation[name] = {"validation": validation_metrics[name], "test": metrics(sensitivity[partitions["test"]], test_sensitivity)}
        for partition, prediction in (("validation", validation_prediction), ("test", test_prediction)):
            truth = activity[partitions[partition]]
            evaluation[name]["target_pchembl_" + partition] = metrics(truth, prediction) if np.isfinite(truth).any() else None
    output = output or root / "results" / "joint"
    output.mkdir(parents=True, exist_ok=True)
    chosen = fits[selected]
    shared = {key: baseline[key] for key in ("model_ids", "training_ids", "training_names", "training_smiles",
              "training_fingerprints", "training_responses", "cell_means", "fingerprint_center", "projection", "split_sha256")}
    np.savez_compressed(output / "sensitivity_joint.npz", **shared, **expression_state,
                        model_kind=np.array("inductive_multilayer"), selected_model=np.array(selected), candidate_only=np.array(candidate_only),
                        projection_scale=projection_scale, compound_map=chosen["compound_map"],
                        cell_factors=chosen["output_factors"][:cell_count], expression_map=chosen["expression_map"],
                        target_factors=chosen["output_factors"][cell_count:], target_head_ids=head_ids,
                        target_means=target_means, target_scales=target_scales, response_scale=np.array(response_scale))
    report = {"selected_model": selected, "split_sha256": split_hash, "candidate_only": candidate_only,
              "validation_winner_including_baseline": baseline_name if candidate_only else selected,
              "baseline_validation_rmse": baseline_error, "models": evaluation,
              "target_heads": len(head_ids), "target_proteins": len(set(head.split("|")[0] for head in head_ids)),
              "target_observations": {name: int(np.isfinite(activity[indices]).sum()) for name, indices in partitions.items()},
              "configuration": {"rank": rank, "max_epochs": epochs, "patience": 5, "seed": 2026,
                                "target_loss_weight": activity_weight, "minimum_training_structures_per_target_endpoint": 3,
                                "compound_regularization": compound_regularization, "output_regularization": output_regularization,
                                "expression_coupling": 10.0, "expression_map_regularization": 1.0},
              "best_epochs": {name: fitted["epoch"] for name, fitted in fits.items()}, "optimization_history": histories,
              "notes": ["Shared compound factors connect cell response and measured protein activity; expression regularizes cell factors.",
                        "Target supervision uses only training structures. Target means, scales, and head vocabulary are train-only.",
                        "Protein/endpoint/assay-type heads stay separate; missing measurements are masked, not negative labels.",
                        "Target aliases are collapsed by full chemical identity; each chemical contributes target labels once.",
                        "Target loss weight balances the total observed normalized sensitivity and activity losses.",
                        "Early stopping and candidate selection use sensitivity validation RMSE only.",
                        "The zero-target-weight control isolates the effect of target supervision; the deployed baseline stays if it wins.",
                        "Auxiliary pChEMBL errors cover supported measured heads only; they are not target identification accuracy.",
                        "Previously inspected test results make these follow-up test comparisons exploratory."]}
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "optimization_history"}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--compound-regularization", type=float, default=10.0)
    parser.add_argument("--output-regularization", type=float, default=1.0)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.rank < 1 or arguments.epochs < 1:
        parser.error("rank and epochs must be positive")
    if not np.isfinite([arguments.compound_regularization, arguments.output_regularization]).all() or min(arguments.compound_regularization, arguments.output_regularization) <= 0:
        parser.error("regularization must be positive and finite")
    train(arguments.root.resolve(), arguments.rank, arguments.epochs, arguments.compound_regularization,
          arguments.output_regularization, arguments.validation_only, arguments.output)
