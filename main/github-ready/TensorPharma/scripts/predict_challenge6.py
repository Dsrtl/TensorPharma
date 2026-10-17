import argparse
import json
from pathlib import Path

import numpy as np
from rdkit import Chem

from challenge6_baseline import neighbor_predictions, similarities
from challenge6_features import fingerprint, standardize


def predict(smiles, model_path, limit=20):
    if limit < 1:
        raise ValueError("limit must be positive")
    molecule = standardize(smiles)
    canonical = Chem.MolToSmiles(molecule, isomericSmiles=True)
    query = fingerprint(molecule)[None, :]
    with np.load(model_path, allow_pickle=False) as model:
        selected = str(model["selected_model"].item())
        similarity = similarities(query, model["training_fingerprints"])[0]
        is_multilayer = "model_kind" in model and str(model["model_kind"].item()) == "inductive_multilayer"
        if is_multilayer:
            embedding = (query - model["fingerprint_center"]) @ model["projection"] / model["projection_scale"]
            scores = (embedding @ model["compound_map"] @ model["cell_factors"].T)[0] * float(model["response_scale"]) + model["cell_means"]
        elif selected == "cell_mean":
            scores = model["cell_means"]
        elif selected == "tanimoto_knn_5":
            scores = neighbor_predictions(query, model["training_fingerprints"],
                                          model["training_responses"], model["cell_means"])[0]
        elif selected == str(model["ridge_model"].item()):
            embedding = (query - model["fingerprint_center"]) @ model["projection"]
            scores = (embedding @ model["coefficients"] + model["intercepts"])[0]
        else:
            raise ValueError("Unsupported model: " + selected)
        if not np.isfinite(scores).all():
            raise ValueError("Prediction produced non-finite scores")
        nearest = np.argsort(-similarity, kind="stable")[:5]
        evidence = [{"pert_id": str(model["training_ids"][index]),
                     "name": str(model["training_names"][index]),
                     "standardized_smiles": str(model["training_smiles"][index]),
                     "tanimoto_similarity": float(similarity[index])} for index in nearest]
        ranked = np.argsort(-scores, kind="stable")[:limit]
        cells = []
        for cell_index in ranked:
            cell_evidence = []
            for reference_index in nearest:
                measured = model["training_responses"][reference_index, cell_index]
                if np.isfinite(measured):
                    cell_evidence.append({"pert_id": str(model["training_ids"][reference_index]),
                                          "measured_score": float(measured)})
            cells.append({"model_id": str(model["model_ids"][cell_index]),
                          "predicted_sensitivity_score": float(scores[cell_index]),
                          "nearby_compound_measurements": cell_evidence})
        result = {"schema_version": "0.1", "standardized_smiles": canonical,
                  "sensitivity_model": selected, "split_sha256": str(model["split_sha256"].item()),
                  "molecule_specific_response_model": selected != "cell_mean",
                  "structure_present_in_training": bool(np.any(model["training_smiles"] == canonical)),
                  "maximum_training_tanimoto": float(similarity.max()),
                  "similar_training_compounds": evidence, "cell_lines": cells,
                  "targets": [], "cancer_types": [],
                  "pending_components": ["ChEMBL target predictor", "DepMap cancer-type labels", "mechanism hypothesis"],
                  "notes": ["Cell lines are sorted by descending raw predicted score.",
                            "Score meaning/direction requires confirmation; these are not response probabilities.",
                            "Nearby measurements provide context; they are not causal explanations.",
                            "Low training similarity indicates weaker structural support; no calibrated uncertainty is available."]}
        if selected == "cell_mean":
            result["notes"].append("These response scores do not depend on the query molecule; molecule-specific response prediction has not beaten this baseline.")
        if is_multilayer:
            result["candidate_only"] = bool(model["candidate_only"].item())
            if result["candidate_only"]:
                result["notes"].append("This candidate did not beat the deployed baseline on validation RMSE.")
    metrics_path = model_path.parent / "metrics.json"
    if metrics_path.exists():
        evaluation = json.loads(metrics_path.read_text())
        if evaluation["split_sha256"] != result["split_sha256"]:
            raise ValueError("Model and metrics refer to different splits")
        result["held_out_performance"] = evaluation["models"][selected]["test"]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--model", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results" / "baseline" / "sensitivity_baseline.npz")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = json.dumps(predict(arguments.smiles, arguments.model, arguments.limit), indent=2)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(result + "\n")
    else:
        print(result)
