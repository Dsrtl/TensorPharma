import argparse
import json
from pathlib import Path


def summarize(root):
    baseline = json.loads((root / "results" / "baseline" / "metrics.json").read_text())
    baseline_name = baseline["selected_model"]
    baseline_error = baseline["models"][baseline_name]["validation"]["rmse"]
    rows = []
    for strength in (100, 1000, 10000):
        path = root / "results" / "regularization" / str(strength) / "validation_report.json"
        report = json.loads(path.read_text())
        if report["split_sha256"] != baseline["split_sha256"] or not report["validation_only"]:
            raise ValueError("Invalid sweep report: " + str(path))
        configuration = report["configuration"]
        if configuration["compound_regularization"] != strength or configuration["output_regularization"] != strength:
            raise ValueError("Sweep regularization does not match its directory: " + str(path))
        for name, model in report["models"].items():
            rows.append({"model": name, "regularization": strength, "validation_rmse": model["validation"]["rmse"],
                         "validation_mae": model["validation"]["mae"],
                         "validation_spearman": model["validation"]["median_within_cell_spearman"],
                         "best_epoch": model["best_epoch"]})
    best = min(rows, key=lambda record: record["validation_rmse"])
    beats_baseline = best["validation_rmse"] < baseline_error
    strength = best["regularization"]
    command = (".venv/bin/python scripts/challenge6_joint.py --compound-regularization " + str(strength)
               + " --output-regularization " + str(strength) + " --output results/joint_regularized")
    summary = {"baseline_model": baseline_name, "baseline_validation_rmse": baseline_error,
               "candidates": rows, "best_candidate": best, "candidate_beats_baseline_on_validation": beats_baseline,
               "final_evaluation_command": command if beats_baseline else None,
               "notes": ["Selection uses validation only; no new test metrics were computed in this sweep.",
                         "A small validation improvement is not proof of a meaningful generalization gain.",
                         "If the baseline still wins, retain it and report the multilayer experiment as a negative result."]}
    output = root / "results" / "regularization" / "summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    arguments = parser.parse_args()
    summarize(arguments.root.resolve())
