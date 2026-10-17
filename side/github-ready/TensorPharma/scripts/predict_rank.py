import argparse
import json
from pathlib import Path

import numpy as np

from challenge6_baseline import neighbor_predictions, similarities


def predict_fingerprint(query, output, limit=20, canonical_smiles=None):
    if limit < 1:
        raise ValueError("limit must be positive")
    report = json.loads((output / "benchmark_report.json").read_text())
    with np.load(output / "best_model.npz", allow_pickle=False) as model:
        if str(model["selected_model"].item()) != report["selected_model"]:
            raise ValueError("Saved model and benchmark report name different winners")
        query = np.asarray(query, dtype=np.float64).reshape(1, -1)
        if query.shape[1] != int(model["fingerprint_bits"]):
            raise ValueError("Query fingerprint length differs from the trained model")
        kind = str(model["model_kind"].item())
        if kind == "mean":
            scores = model["prediction_means"]
        elif kind == "tanimoto":
            scores = neighbor_predictions(query, model["training_fingerprints"], model["training_responses"],
                                          model["cell_means"], int(model["neighbors"]))[0]
        elif kind == "ridge":
            embedding = (query - model["fingerprint_center"]) @ model["projection"]
            scores = (embedding @ model["coefficients"] + model["intercepts"])[0]
        elif kind == "coupled":
            embedding = (query - model["fingerprint_center"]) @ model["projection"] / model["projection_scale"]
            scores = (embedding @ model["compound_map"] @ model["cell_factors"].T)[0] * float(model["response_scale"]) + model["cell_means"]
        elif kind == "lightgbm":
            import lightgbm as lgb

            booster = lgb.Booster(model_file=str(output / "best_model.txt"))
            values = np.empty((len(model["model_ids"]), query.shape[1] + model["gene_values"].shape[1]), dtype=np.float32)
            values[:, :query.shape[1]] = query
            values[:, query.shape[1]:] = model["gene_values"]
            scores = booster.predict(values)
        else:
            raise ValueError("Unknown saved model kind: " + kind)
        scores = np.clip(scores, 0, 1)
        if not np.isfinite(scores).all():
            raise ValueError("Non-finite saved-model prediction")
        ranked = np.argsort(-scores, kind="stable")[:limit]
        cells = [{"cell_id": str(model["model_ids"][index]), "cell_name": str(model["cell_names"][index]),
                  "cancer_type": str(model["cancer_types"][index]), "predicted_sensitivity_rank": float(scores[index])}
                 for index in ranked]
        types = []
        for cancer_type in sorted(set(model["cancer_types"])):
            selected = model["cancer_types"] == cancer_type
            types.append({"cancer_type": str(cancer_type), "median_predicted_rank": float(np.median(scores[selected])),
                          "cell_lines": int(selected.sum())})
        types.sort(key=lambda record: record["median_predicted_rank"], reverse=True)
        similarity = similarities(query, model["training_fingerprints"])[0]
        nearest = np.argsort(-similarity, kind="stable")[:5]
        evidence = [{"pert_id": str(model["training_ids"][index]), "name": str(model["training_names"][index]),
                     "tanimoto_similarity": float(similarity[index])} for index in nearest]
        family = report["selected_family"]
        return {"schema_version": "rank-1", "selected_model": report["selected_model"], "model_family": family,
                "benchmark_complete": report["benchmark_complete"], "standardized_smiles": canonical_smiles,
                "molecule_specific": kind != "mean",
                "structure_present_in_training": bool(np.any(model["training_smiles"] == canonical_smiles)) if canonical_smiles else None,
                "cell_lines": cells, "cancer_type_summaries": types, "similar_training_compounds": evidence,
                "held_out_performance": report["models"][family]["test"],
                "notes": ["Predicted ranks describe relative cell-line sensitivity for this drug, not absolute cell killing or probabilities.",
                          "Cancer-type summaries aggregate model scores from cell lines; they do not establish clinical indications.",
                          "The mean baseline, if selected, gives the same cell-line ranking for every query molecule.",
                          "Structural neighbours provide context, not a causal mechanism or calibrated uncertainty."]}


if __name__ == "__main__":
    from rdkit import Chem, rdBase

    from challenge6_features import fingerprint, standardize

    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--benchmark", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "rank_benchmark")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    metadata = json.loads((arguments.benchmark / "benchmark_report.json").read_text())
    expected_version = metadata["data"].get("frozen_feature_configuration", {}).get("rdkit_version")
    if expected_version and expected_version != rdBase.rdkitVersion:
        parser.error("Use the frozen feature RDKit version: " + expected_version)
    molecule = standardize(arguments.smiles)
    result = predict_fingerprint(fingerprint(molecule), arguments.benchmark, arguments.limit,
                                 Chem.MolToSmiles(molecule, isomericSmiles=True))
    rendered = json.dumps(result, indent=2)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n")
    else:
        print(rendered)
