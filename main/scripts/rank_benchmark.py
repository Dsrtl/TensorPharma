import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from challenge6_baseline import cell_means, fit_projection, fit_ridge, neighbor_predictions
from challenge6_joint import fit_joint, target_heads
from rank_data import prepare


class RankMetrics:
    def __init__(self, truth):
        self.truth = truth
        self.observed = np.isfinite(truth)
        self.values = truth[self.observed]
        self.groups = []
        cursor = 0
        for row in truth:
            values = row[np.isfinite(row)]
            ranks = rankdata(values)
            centered = ranks - ranks.mean() if len(ranks) else ranks
            norm = float(np.linalg.norm(centered))
            self.groups.append((cursor, cursor + len(values), centered, norm))
            cursor += len(values)
        if not len(self.values):
            raise ValueError("No labelled observations for ranking evaluation")

    def vector(self, predictions):
        values = np.clip(np.asarray(predictions, dtype=np.float64), 0, 1)
        if values.shape != self.values.shape or not np.isfinite(values).all():
            raise ValueError("Invalid ranking predictions")
        correlations = []
        constants = 0
        excluded = 0
        per_drug_errors = []
        for start, end, centered, norm in self.groups:
            if start == end:
                continue
            per_drug_errors.append(float(np.sqrt(np.mean((values[start:end] - self.values[start:end]) ** 2))))
            if end - start < 3 or norm <= 1e-12:
                excluded += 1
                continue
            predicted = rankdata(values[start:end])
            predicted -= predicted.mean()
            predicted_norm = float(np.linalg.norm(predicted))
            if predicted_norm <= 1e-12:
                constants += 1
                correlations.append(0.0)
            else:
                correlations.append(float(np.clip(centered @ predicted / (norm * predicted_norm), -1, 1)))
        errors = values - self.values
        return {"observations": len(values), "macro_per_drug_spearman": float(np.mean(correlations)) if correlations else None,
                "ranking_eligible_drugs": len(correlations), "constant_prediction_drugs_scored_zero": constants,
                "drugs_excluded_from_ranking": excluded, "rmse": float(np.sqrt(np.mean(errors ** 2))),
                "mae": float(np.mean(np.abs(errors))), "macro_per_drug_rmse": float(np.mean(per_drug_errors))}

    def matrix(self, predictions):
        if predictions.shape != self.truth.shape:
            raise ValueError("Prediction matrix shape differs from labels")
        return self.vector(predictions[self.observed])

    def loss(self, predictions):
        score = self.matrix(predictions)["macro_per_drug_spearman"]
        if score is None:
            raise ValueError("No nonconstant validation rank labels")
        return -score


def selection_key(metrics):
    score = metrics["macro_per_drug_spearman"]
    return (score if score is not None else -np.inf, -metrics["rmse"])


def pair_features(data, compound_indices, cell_indices):
    fingerprints = data["fingerprints"]
    genes = data["expression"]["gene_values"]
    features = np.empty((len(compound_indices), fingerprints.shape[1] + genes.shape[1]), dtype=np.float32)
    features[:, :fingerprints.shape[1]] = fingerprints[compound_indices]
    features[:, fingerprints.shape[1]:] = genes[cell_indices]
    return features


def predict(candidate, indices, data, common):
    kind = candidate["kind"]
    state = candidate["state"]
    if kind == "mean":
        values = np.broadcast_to(state["prediction_means"], (len(indices), len(common["cell_means"]))).copy()
    elif kind == "tanimoto":
        values = neighbor_predictions(data["fingerprints"][indices], common["training_fingerprints"],
                                      common["training_responses"], common["cell_means"], state["neighbors"])
    elif kind == "ridge":
        embedded = (data["fingerprints"][indices] - common["fingerprint_center"]) @ common["projection"]
        values = embedded @ state["coefficients"] + state["intercepts"]
    elif kind == "coupled":
        embedded = (data["fingerprints"][indices] - common["fingerprint_center"]) @ common["projection"] / common["projection_scale"]
        values = embedded @ state["compound_map"] @ state["cell_factors"].T * common["response_scale"] + common["cell_means"]
    elif kind == "lightgbm":
        rows, cells = np.where(np.isfinite(data["matrix"][indices]))
        features = pair_features(data, indices[rows], cells)
        predictions = state["booster"].predict(features, num_iteration=state["booster"].best_iteration)
        values = np.full((len(indices), len(common["cell_means"])), np.nan)
        values[rows, cells] = predictions
        del features
    else:
        raise ValueError("Unknown prediction kind: " + kind)
    return np.clip(values, 0, 1)


def lightgbm_candidates(data, common, evaluator, rounds, threads):
    import lightgbm as lgb

    training = data["partitions"]["train"]
    validation = data["partitions"]["validation"]
    training_rows, training_cells = np.where(np.isfinite(data["matrix"][training]))
    validation_rows, validation_cells = np.where(np.isfinite(data["matrix"][validation]))
    training_features = pair_features(data, training[training_rows], training_cells)
    validation_features = pair_features(data, validation[validation_rows], validation_cells)
    dataset_parameters = {"feature_pre_filter": False, "max_bin": 63, "num_threads": threads}
    training_set = lgb.Dataset(training_features, label=data["matrix"][training][training_rows, training_cells],
                               free_raw_data=True, params=dataset_parameters)
    validation_set = lgb.Dataset(validation_features, label=data["matrix"][validation][validation_rows, validation_cells],
                                 reference=training_set, free_raw_data=True, params=dataset_parameters)
    training_set.construct()
    validation_set.construct()
    del training_features, validation_features
    gc.collect()

    def evaluate(predictions, dataset):
        report = evaluator.vector(predictions)
        return [("macro_drug_spearman", report["macro_per_drug_spearman"], True), ("rank_rmse", report["rmse"], False)]

    candidates = []
    for leaves, penalty in ((15, 10.0), (31, 100.0)):
        parameters = {**dataset_parameters, "objective": "regression", "metric": "None", "learning_rate": 0.05,
                      "num_leaves": leaves, "min_data_in_leaf": 200, "lambda_l2": penalty,
                      "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
                      "seed": 2026, "feature_fraction_seed": 2026, "bagging_seed": 2026,
                      "deterministic": True, "force_col_wise": True, "histogram_pool_size": 512, "verbosity": -1}
        booster = lgb.train(parameters, training_set, num_boost_round=rounds, valid_sets=[validation_set],
                            valid_names=["validation"], feval=evaluate,
                            callbacks=[lgb.early_stopping(40, first_metric_only=True, min_delta=0.0001), lgb.log_evaluation(25)])
        candidate = {"name": "lightgbm_leaves_" + str(leaves), "family": "lightgbm", "kind": "lightgbm",
                     "configuration": {**parameters, "best_iteration": booster.best_iteration}, "state": {"booster": booster}}
        candidate["validation"] = evaluator.matrix(predict(candidate, validation, data, common))
        candidates.append(candidate)
        print(json.dumps({"candidate": candidate["name"], "validation": candidate["validation"]}), flush=True)
    return candidates


def run(root, source, frozen, output, gene_count=1000, epochs=25, rounds=600, threads=8, include_lightgbm=True):
    if (output / "benchmark_report.json").exists():
        raise FileExistsError("Completed benchmark exists; choose a new --output directory")
    data = prepare(source, frozen, output / "prepared", gene_count)
    partitions = data["partitions"]
    training = partitions["train"]
    validation = partitions["validation"]
    truth = data["matrix"]
    training_labels = truth[training]
    means = cell_means(training_labels)
    response_scale = float(np.sqrt(np.nanmean((training_labels - means) ** 2)))
    if response_scale <= 1e-12:
        raise ValueError("No training rank variation")
    center, projection = fit_projection(data["fingerprints"][training])
    embedded = (data["fingerprints"] - center) @ projection
    projection_scale = np.maximum(embedded[training].std(axis=0), 1e-8)
    normalized_embedding = embedded / projection_scale
    evaluators = {name: RankMetrics(truth[indices]) for name, indices in partitions.items() if name != "train"}
    if not evaluators["validation"].matrix(np.broadcast_to(means, truth[validation].shape))["ranking_eligible_drugs"]:
        raise ValueError("No usable validation ranking labels")
    common = {"cell_means": means, "training_fingerprints": data["fingerprints"][training],
              "training_responses": training_labels, "fingerprint_center": center, "projection": projection,
              "projection_scale": projection_scale, "response_scale": np.array(response_scale)}
    candidates = []

    def add(name, family, kind, configuration, state):
        candidate = {"name": name, "family": family, "kind": kind, "configuration": configuration, "state": state}
        candidate["validation"] = evaluators["validation"].matrix(predict(candidate, validation, data, common))
        candidates.append(candidate)
        print(json.dumps({"candidate": name, "validation": candidate["validation"]}), flush=True)

    add("global_mean", "mean_baseline", "mean", {"variant": "global"}, {"prediction_means": np.full(len(means), np.nanmean(training_labels))})
    add("cell_mean", "mean_baseline", "mean", {"variant": "per_cell"}, {"prediction_means": means})
    for neighbors in (1, 5, 10):
        add("tanimoto_" + str(neighbors), "similarity", "tanimoto", {"neighbors": neighbors}, {"neighbors": neighbors})
    for regularization in (10.0, 100.0, 1000.0):
        coefficients, intercepts = fit_ridge(embedded[training], training_labels, means, regularization)
        add("ridge_" + str(int(regularization)), "fingerprint_ridge", "ridge", {"regularization": regularization},
            {"coefficients": coefficients, "intercepts": intercepts})
    activity, target_means, target_scales, head_ids = target_heads(root, data["compounds"], training, data["endpoint_path"])
    normalized_labels = np.column_stack([(training_labels - means) / response_scale, (activity[training] - target_means) / target_scales])
    target_weight = float(np.isfinite(training_labels).sum() / np.isfinite(activity[training]).sum())
    cell_count = truth.shape[1]
    for family, activity_weight in (("expression_coupled", 0.0), ("joint_targets_expression", target_weight)):
        for regularization in (10.0, 100.0):
            name = family + "_penalty_" + str(int(regularization))
            weights = np.concatenate([np.ones(cell_count), np.full(len(head_ids), activity_weight)])
            fitted, history = fit_joint(
                normalized_embedding[training], data["expression"]["expression_design"], normalized_labels, weights,
                normalized_embedding[validation], truth[validation], means, response_scale, cell_count, 8, epochs, name,
                compound_regularization=regularization, output_regularization=regularization,
                validation_objective=evaluators["validation"].loss)
            (output / (name + "_history.json")).write_text(json.dumps(history, indent=2) + "\n")
            add(name, family, "coupled", {"rank": 8, "regularization": regularization, "target_weight": activity_weight,
                                          "best_epoch": fitted["epoch"], "target_heads": len(head_ids)},
                {"compound_map": fitted["compound_map"], "cell_factors": fitted["output_factors"][:cell_count],
                 "target_factors": fitted["output_factors"][cell_count:], "expression_map": fitted["expression_map"]})
    if include_lightgbm:
        candidates.extend(lightgbm_candidates(data, common, evaluators["validation"], rounds, threads))
    families = sorted({candidate["family"] for candidate in candidates})
    finalists = {family: max((candidate for candidate in candidates if candidate["family"] == family),
                            key=lambda candidate: selection_key(candidate["validation"])) for family in families}
    winner = max(finalists.values(), key=lambda candidate: selection_key(candidate["validation"]))
    print("Selected on validation before test evaluation:", winner["name"], flush=True)
    models = {}
    test_rows, test_cells = np.where(np.isfinite(truth[partitions["test"]]))
    test_records = pd.DataFrame({"pert_id": data["compounds"].iloc[partitions["test"][test_rows]]["pert_id"].to_numpy(),
                                 "cell_id": data["cells"].iloc[test_cells]["cell_id"].to_numpy(),
                                 "observed_rank": truth[partitions["test"]][test_rows, test_cells]})
    for family, candidate in finalists.items():
        predictions = predict(candidate, partitions["test"], data, common)
        models[family] = {"candidate": candidate["name"], "configuration": candidate["configuration"],
                          "validation": candidate["validation"], "test": evaluators["test"].matrix(predictions)}
        test_records[family] = predictions[test_rows, test_cells]
    test_records.to_csv(output / "test_predictions.csv.gz", index=False)
    training_compounds = data["compounds"].iloc[training]
    artifact = {**common, "model_kind": np.array(winner["kind"]), "selected_model": np.array(winner["name"]),
                "model_ids": data["cells"]["cell_id"].to_numpy(dtype=str), "cell_names": data["cells"]["cell_name"].to_numpy(dtype=str),
                "cancer_types": data["cells"]["cancer_type"].to_numpy(dtype=str),
                "training_ids": training_compounds["pert_id"].to_numpy(dtype=str), "training_names": training_compounds["name"].to_numpy(dtype=str),
                "training_smiles": training_compounds["standardized_smiles"].to_numpy(dtype=str),
                "target_head_ids": head_ids, "target_means": target_means, "target_scales": target_scales,
                "fingerprint_bits": np.array(data["fingerprints"].shape[1]), **data["expression"]}
    if winner["kind"] == "lightgbm":
        winner["state"]["booster"].save_model(str(output / "best_model.txt"))
    else:
        artifact.update({key: np.asarray(value) for key, value in winner["state"].items()})
    np.savez_compressed(output / "best_model.npz", **artifact)
    if "lightgbm" in finalists:
        booster = finalists["lightgbm"]["state"]["booster"]
        feature_names = ["fingerprint_" + str(index) for index in range(data["fingerprints"].shape[1])] + data["expression"]["gene_names"].tolist()
        pd.DataFrame({"feature": feature_names, "gain": booster.feature_importance(importance_type="gain")}).sort_values("gain", ascending=False).to_csv(output / "lightgbm_feature_importance.csv", index=False)
    report = {"selected_model": winner["name"], "selected_family": winner["family"], "benchmark_complete": include_lightgbm,
              "label": "sensitivity_rank", "selection_rule": "Highest validation macro per-drug Spearman; ties use lower validation RMSE",
              "data": data["audit"], "validation_candidates": [{key: candidate[key] for key in ("name", "family", "configuration", "validation")} for candidate in candidates],
              "models": models, "selected_configuration": winner["configuration"],
              "notes": ["Six response model families are compared; protein-target retrieval is a separate task." if include_lightgbm else "Diagnostic run omits LightGBM; this is not the complete six-family comparison.",
                        "All models use identical cleaned rank labels, canonical compounds/cells, and frozen scaffold partitions.",
                        "Predicted ranks are clipped to [0, 1] before selection, evaluation, and serving.",
                        "Ranking is within each drug across cell lines; this differs from earlier raw-score within-cell metrics.",
                        "Constant predictions receive ranking score zero when true ranks vary. Constant true labels or fewer than three observations are excluded from ranking, but included in errors.",
                        "Hyperparameters, checkpoints, and the winner are chosen before test evaluation; each family's fixed finalist is evaluated once.",
                        "Test compounds were inspected in earlier raw-score experiments; this is not a pristine external validation set.",
                        "Relative sensitivity does not establish absolute killing, treatment efficacy, or calibrated probabilities.",
                        "Known target/MOA annotations and all-drug trust weights are never predictive features."]}
    (output / "benchmark_report.json").write_text(json.dumps(report, indent=2) + "\n")
    summary = {"selected_model": winner["name"], "selected_family": winner["family"], "benchmark_complete": include_lightgbm,
               "compound_counts": data["audit"]["compound_counts"], "response_counts": data["audit"]["response_counts"],
               "models": {family: {"candidate": model["candidate"], "validation_spearman": model["validation"]["macro_per_drug_spearman"],
                                     "test_spearman": model["test"]["macro_per_drug_spearman"], "validation_rmse": model["validation"]["rmse"],
                                     "test_rmse": model["test"]["rmse"]} for family, model in models.items()}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source", type=Path)
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--genes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--rounds", type=int, default=600)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--skip-lightgbm", action="store_true")
    arguments = parser.parse_args()
    if min(arguments.genes, arguments.epochs, arguments.rounds, arguments.threads) < 1:
        parser.error("genes, epochs, rounds, and threads must be positive")
    root = arguments.root.resolve()
    run(root, (arguments.source or root / "TensorPharma-main" / "data" / "processed").resolve(),
        (arguments.frozen or root / "data" / "features").resolve(),
        (arguments.output or root / "results" / "rank_benchmark").resolve(),
        arguments.genes, arguments.epochs, arguments.rounds, arguments.threads, not arguments.skip_lightgbm)
