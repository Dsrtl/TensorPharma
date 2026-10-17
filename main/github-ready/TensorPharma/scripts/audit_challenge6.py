import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def audit(root):
    responses = pd.read_csv(root / "data" / "prepared" / "matched_responses.csv")
    compounds = pd.read_csv(root / "data" / "prepared" / "compounds.csv", keep_default_na=False)
    output = root / "results" / "diagnostics"
    output.mkdir(parents=True, exist_ok=True)
    split_path = root / "data" / "features" / "compound_split.csv"
    if split_path.exists():
        splits = pd.read_csv(split_path, keep_default_na=False)[["pert_id", "split"]]
        responses = responses.merge(splits, on="pert_id", validate="many_to_one")
    else:
        responses["split"] = "all"
    summary = {}
    for split, records in responses.groupby("split"):
        values = records["sensitivity_score"]
        summary[split] = {
            "observations": len(records), "compounds": int(records["pert_id"].nunique()),
            "fraction_zero": float((values == 0).mean()), "mean": float(values.mean()),
            "standard_deviation": float(values.std()),
            "quantiles": {str(key): float(value) for key, value in values.quantile([0, .5, .9, .95, .99, .999, 1]).items()},
            "fraction_above_10": float((values > 10).mean()),
        }
    per_compound = responses.groupby(["split", "pert_id"])["sensitivity_score"].agg(["count", "mean", "median", "max"]).reset_index()
    per_compound = per_compound.merge(compounds[["pert_id", "name"]], on="pert_id", validate="many_to_one")
    per_compound.to_csv(output / "compound_score_summary.csv", index=False)
    prediction_path = root / "results" / "baseline" / "test_predictions.csv"
    if prediction_path.exists():
        predictions = pd.read_csv(prediction_path)
        predictions["squared_error"] = (predictions["predicted_score"] - predictions["observed_score"]) ** 2
        errors = predictions.groupby("pert_id").agg(
            observations=("squared_error", "size"), squared_error=("squared_error", "sum"),
            mean_observed=("observed_score", "mean"), max_observed=("observed_score", "max"))
        total_error = float(errors["squared_error"].sum())
        errors["fraction_of_test_squared_error"] = errors["squared_error"] / total_error if total_error > 0 else 0.0
        errors = errors.reset_index().merge(compounds[["pert_id", "name"]], on="pert_id", validate="one_to_one")
        errors = errors.sort_values("squared_error", ascending=False)
        errors.to_csv(output / "test_error_by_compound.csv", index=False)
        summary["test_error_diagnostic"] = {
            "top_five_fraction_of_squared_error": float(errors.head(5)["fraction_of_test_squared_error"].sum()),
            "top_five_compounds": errors.head(5).to_dict("records"),
            "note": "Diagnostic only. Do not remove difficult test compounds or tune on these test errors.",
        }
    summary["interpretation"] = [
        "Large-score observations can dominate squared error even when most scores are small.",
        "Different scaffold partitions can have different response distributions.",
        "Confirm uniform score units and direction with the data provider; this audit cannot establish them.",
        "A cell-mean predictor cannot distinguish compounds within the same cell line.",
        "Report both score error and ranking metrics. Do not select models using test performance.",
    ]
    (output / "score_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    arguments = parser.parse_args()
    audit(arguments.root)
