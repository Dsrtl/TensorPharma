import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import solve, svd

from challenge6_baseline import metrics


def expression_features(path, model_ids, gene_count=1000, dimensions=32):
    with np.load(path, allow_pickle=False) as archive:
        lookup = {identifier: index for index, identifier in enumerate(archive["model_ids"])}
        missing = sorted(set(model_ids) - set(lookup))
        if missing:
            raise ValueError("Missing expression models: " + str(missing))
        expression = archive["expression"][[lookup[identifier] for identifier in model_ids]].astype(np.float64)
        gene_names = archive["genes"]
    valid = np.isfinite(expression).any(axis=0)
    expression = expression[:, valid]
    gene_names = gene_names[valid]
    medians = np.nanmedian(expression, axis=0)
    expression = np.where(np.isfinite(expression), expression, medians)
    variance = expression.var(axis=0)
    selected = np.argsort(-variance, kind="stable")[:min(gene_count, int((variance > 1e-12).sum()))]
    if not len(selected):
        raise ValueError("No variable expression genes")
    expression = expression[:, selected]
    center = expression.mean(axis=0)
    scale = expression.std(axis=0)
    normalized = (expression - center) / scale
    _, singular_values, directions = svd(normalized, full_matrices=False, check_finite=True)
    rank = min(dimensions, int((singular_values > 1e-8).sum()))
    if rank == 0:
        raise ValueError("Expression projection has no variation")
    projection = directions[:rank].T
    embedding = normalized @ projection
    embedding_scale = embedding.std(axis=0)
    embedding /= embedding_scale
    design = np.column_stack([np.ones(len(model_ids)), embedding])
    return design, {"expression_genes": gene_names[selected], "expression_medians": medians[selected],
                    "expression_center": center, "expression_scale": scale,
                    "expression_projection": projection, "expression_embedding_scale": embedding_scale}


def fit_coupled(compound_features, expression, training_scores, validation_features,
                validation_truth, cell_means, response_scale, coupling, rank=8, epochs=25, seed=2026):
    generator = np.random.default_rng(seed)
    observed = np.isfinite(training_scores)
    targets = np.where(observed, (training_scores - cell_means) / response_scale, 0.0)
    compound_map = generator.normal(0, 0.05, (compound_features.shape[1], rank))
    cell_factors = generator.normal(0, 0.05, (training_scores.shape[1], rank))
    cell_factors[:, 0] = 1.0
    expression_map = np.zeros((expression.shape[1], rank))
    compound_regularization = 10.0
    cell_regularization = 1.0
    expression_regularization = 1.0
    best_error = np.inf
    best = None
    history = []
    stalled = 0
    for epoch in range(1, epochs + 1):
        embedding = compound_features @ compound_map
        residual = np.where(observed, targets - embedding @ cell_factors.T, 0.0)
        for factor in range(rank):
            old_component = np.outer(embedding[:, factor], cell_factors[:, factor])
            partial = residual + observed * old_component
            row_weights = observed @ (cell_factors[:, factor] ** 2)
            gram = compound_features.T @ (row_weights[:, None] * compound_features)
            right = compound_features.T @ (partial @ cell_factors[:, factor])
            compound_map[:, factor] = solve(
                gram + compound_regularization * np.eye(gram.shape[0]), right, assume_a="pos")
            embedding[:, factor] = compound_features @ compound_map[:, factor]
            residual = partial - observed * np.outer(embedding[:, factor], cell_factors[:, factor])
        prior = expression @ expression_map
        for cell_index in range(training_scores.shape[1]):
            selected = observed[:, cell_index]
            design = embedding[selected]
            gram = design.T @ design + (cell_regularization + coupling) * np.eye(rank)
            right = design.T @ targets[selected, cell_index] + coupling * prior[cell_index]
            cell_factors[cell_index] = solve(gram, right, assume_a="pos")
        if coupling > 0:
            expression_map = solve(
                expression.T @ expression + expression_regularization / coupling * np.eye(expression.shape[1]),
                expression.T @ cell_factors, assume_a="pos")
        residual = np.where(observed, targets - embedding @ cell_factors.T, 0.0)
        objective = float(np.sum(residual ** 2) + compound_regularization * np.sum(compound_map ** 2)
                          + cell_regularization * np.sum(cell_factors ** 2)
                          + coupling * np.sum((cell_factors - expression @ expression_map) ** 2)
                          + expression_regularization * np.sum(expression_map ** 2))
        validation_prediction = (validation_features @ compound_map @ cell_factors.T) * response_scale + cell_means
        validation_mask = np.isfinite(validation_truth)
        validation_error = float(np.sqrt(np.mean((validation_prediction[validation_mask] - validation_truth[validation_mask]) ** 2)))
        if not np.isfinite(objective) or not np.isfinite(validation_error):
            raise ValueError("Non-finite optimization result")
        history.append({"epoch": epoch, "training_objective": objective, "validation_rmse": validation_error})
        print(json.dumps({"coupling": coupling, **history[-1]}), flush=True)
        if validation_error < best_error - 1e-8:
            best_error = validation_error
            best = {"compound_map": compound_map.copy(), "cell_factors": cell_factors.copy(),
                    "expression_map": expression_map.copy(), "epoch": epoch}
            stalled = 0
        else:
            stalled += 1
        if stalled >= 5:
            break
    if best is None:
        raise ValueError("Optimization did not produce a checkpoint")
    return best, history


def train(root, rank, epochs):
    output = root / "results" / "multilayer"
    baseline_dir = root / "results" / "baseline"
    feature_dir = root / "data" / "features"
    baseline_metrics = json.loads((baseline_dir / "metrics.json").read_text())
    split_path = feature_dir / "compound_split.csv"
    split_hash = hashlib.sha256(split_path.read_bytes()).hexdigest()
    if split_hash != baseline_metrics["split_sha256"]:
        raise ValueError("Baseline and multilayer splits differ")
    compounds = pd.read_csv(split_path, keep_default_na=False)
    with np.load(feature_dir / "fingerprints.npz", allow_pickle=False) as archive:
        if not np.array_equal(archive["pert_ids"], compounds["pert_id"].to_numpy()):
            raise ValueError("Fingerprint order differs from split")
        fingerprints = archive["fingerprints"].astype(np.float64)
    with np.load(baseline_dir / "sensitivity_baseline.npz", allow_pickle=False) as archive:
        baseline = {key: archive[key] for key in archive.files}
    partitions = {name: np.flatnonzero(compounds["split"].to_numpy() == name)
                  for name in ("train", "validation", "test")}
    training = partitions["train"]
    if not np.array_equal(compounds.iloc[training]["pert_id"].to_numpy(), baseline["training_ids"]):
        raise ValueError("Baseline training compound order differs")
    responses = pd.read_csv(root / "data" / "prepared" / "matched_responses.csv")
    if responses.duplicated(["pert_id", "model_id"]).any():
        raise ValueError("Duplicate compound/cell observations")
    matrix = responses.pivot(index="pert_id", columns="model_id", values="sensitivity_score").reindex(
        index=compounds["pert_id"], columns=baseline["model_ids"]).to_numpy(dtype=np.float64)
    means = baseline["cell_means"]
    training_scores = matrix[training]
    observed = np.isfinite(training_scores)
    if not np.allclose(training_scores, baseline["training_responses"], equal_nan=True):
        raise ValueError("Baseline training responses changed")
    response_scale = float(np.sqrt(np.mean(((training_scores - means)[observed]) ** 2)))
    if response_scale <= 1e-12:
        raise ValueError("Training response residuals have no variation")
    embedded = (fingerprints - baseline["fingerprint_center"]) @ baseline["projection"]
    projection_scale = embedded[training].std(axis=0)
    projection_scale = np.maximum(projection_scale, 1e-8)
    embedded /= projection_scale
    print("Preparing gene features from cell lines with training labels", flush=True)
    expression, expression_state = expression_features(root / "data" / "expression" / "ccle_matched.npz", baseline["model_ids"])
    fits = {}
    histories = {}
    validation_metrics = {}
    for name, coupling in (("fingerprint_lowrank", 0.0), ("expression_coupled_lowrank", 10.0)):
        fitted, history = fit_coupled(
            embedded[training], expression, training_scores, embedded[partitions["validation"]],
            matrix[partitions["validation"]], means, response_scale, coupling, rank, epochs)
        fits[name] = fitted
        histories[name] = history
        predictions = embedded[partitions["validation"]] @ fitted["compound_map"] @ fitted["cell_factors"].T * response_scale + means
        validation_metrics[name] = metrics(matrix[partitions["validation"]], predictions)
    selected = min(fits, key=lambda name: validation_metrics[name]["rmse"])
    original = baseline_metrics["selected_model"]
    baseline_error = baseline_metrics["models"][original]["validation"]["rmse"]
    candidate_only = validation_metrics[selected]["rmse"] >= baseline_error
    evaluation = {}
    for name, fitted in fits.items():
        predictions = embedded[partitions["test"]] @ fitted["compound_map"] @ fitted["cell_factors"].T * response_scale + means
        evaluation[name] = {"validation": validation_metrics[name], "test": metrics(matrix[partitions["test"]], predictions)}
    output.mkdir(parents=True, exist_ok=True)
    shared = {key: baseline[key] for key in ("model_ids", "training_ids", "training_names", "training_smiles",
              "training_fingerprints", "training_responses", "cell_means", "fingerprint_center", "projection", "split_sha256")}
    chosen = fits[selected]
    np.savez_compressed(output / "sensitivity_multilayer.npz", **shared, **expression_state,
                        model_kind=np.array("inductive_multilayer"), selected_model=np.array(selected),
                        candidate_only=np.array(candidate_only), projection_scale=projection_scale,
                        compound_map=chosen["compound_map"], cell_factors=chosen["cell_factors"],
                        expression_map=chosen["expression_map"], response_scale=np.array(response_scale))
    report = {"selected_model": selected, "split_sha256": split_hash, "candidate_only": candidate_only,
              "validation_winner_including_baseline": original if candidate_only else selected,
              "baseline_validation_rmse": baseline_error, "models": evaluation,
              "configuration": {"rank": rank, "max_epochs": epochs, "patience": 5, "seed": 2026,
                                "compound_regularization": 10.0, "cell_regularization": 1.0,
                                "expression_coupling": 10.0, "expression_map_regularization": 1.0},
              "best_epochs": {name: fitted["epoch"] for name, fitted in fits.items()},
              "optimization_history": histories,
              "notes": ["Shared fingerprint encoder and cell factors; expression regularizes cell factors.",
                        "The uncoupled low-rank control isolates the effect of adding expression.",
                        "Epoch and candidate selection use validation RMSE only; the baseline is retained if it wins.",
                        "Baseline test results were already inspected; these follow-up test comparisons are exploratory.",
                        "This is the pre-ChEMBL sensitivity component, not a protein-target predictor.",
                        "Cell features are fitted only on cells with training response labels; evaluation remains cell-warm.",
                        "Scores remain on the original scale. No large-score observations are removed or clipped."]}
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "optimization_history"}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=25)
    arguments = parser.parse_args()
    if arguments.rank < 1 or arguments.epochs < 1:
        parser.error("rank and epochs must be positive")
    train(arguments.root, arguments.rank, arguments.epochs)
