import argparse
import gc
import json
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd

from chembl_library import Library
from rank_benchmark import RankMetrics, pair_features, selection_key
from rank_data import file_hash, prepare, table_path


FEATURE_NAMES = ["target_weighted_log_expression", "target_weighted_gene_zscore",
                 "target_max_gene_zscore", "top_target_log_expression", "top_target_gene_zscore",
                 "target_expression_mapped_weight", "target_expression_mapped_count"]


def exclusions(path):
    if path is None:
        return set()
    frame = pd.read_csv(path, keep_default_na=False)
    if "gene" not in frame or frame.empty:
        raise ValueError("Exclusion CSV must have a nonempty gene column")
    genes = frame["gene"].str.strip()
    if genes.eq("").any() or genes.duplicated().any():
        raise ValueError("Exclusion genes must be nonempty and unique")
    if not genes.str.fullmatch(r"[A-Z][A-Z0-9-]*").all():
        raise ValueError("Malformed gene symbols in exclusion CSV; resolve the original names first")
    return set(genes)


def retrieve(data, library_path, selection_path, output, excluded):
    metadata = json.loads((library_path / "manifest.json").read_text())
    selected = json.loads(selection_path.read_text())
    audit = data["audit"]
    if selected["library_manifest"] != metadata:
        raise ValueError("Target settings and library manifest differ")
    if metadata["split_sha256"] != audit["frozen_split_sha256"]:
        raise ValueError("Target reference and response chemical split differ")
    if metadata["rdkit_version"] != audit["frozen_feature_configuration"]["rdkit_version"]:
        raise ValueError("Fingerprint RDKit versions differ")
    if metadata["fingerprint_bits"] != data["fingerprints"].shape[1] or metadata["radius"] != 2:
        raise ValueError("Target/reference fingerprint settings differ")
    required_exclusion = "All frozen LINCS identities/connectivities, plus validation/test scaffolds"
    if metadata["exclusion"] != required_exclusion:
        raise ValueError("Require the original LINCS-excluded target reference")
    references = pd.read_csv(library_path / "compounds.csv.gz", usecols=["inchi_key"])
    query_keys = set(data["compounds"]["inchi_key"])
    if query_keys & set(references["inchi_key"]):
        raise ValueError("Query structures leaked into the target reference")
    if {key[:14] for key in query_keys} & {key[:14] for key in references["inchi_key"]}:
        raise ValueError("Query connectivity leaked into the target reference")
    signature = {"library_manifest": metadata, "cutoff": selected["selected_cutoff"],
                 "trust": selected["selected_trust"], "top_targets": 20,
                 "excluded_genes": sorted(excluded), "data": audit,
                 "feature_names": FEATURE_NAMES}
    configuration = output / "configuration.json"
    if configuration.exists() and json.loads(configuration.read_text()) != signature:
        raise ValueError("Partial-run inputs changed; use a separate output directory")
    configuration.write_text(json.dumps(signature, indent=2) + "\n")
    library = Library(library_path, excluded_genes=excluded)
    print(json.dumps({"stage": "reference", "excluded_genes_requested": len(excluded),
                      "reference_gene_labels_retained": len(library.genes)}), flush=True)
    cache = output / "target_predictions.jsonl"
    predictions = {}
    if cache.exists():
        for line in cache.read_text().splitlines():
            record = json.loads(line)
            if record["pert_id"] in predictions:
                raise ValueError("Duplicate cached target query")
            predictions[record["pert_id"]] = record["targets"]
    start = time.monotonic()
    processed = 0
    remaining = len(data["compounds"]) - len(predictions)
    with cache.open("a") as stream:
        for index, row in data["compounds"].iterrows():
            identifier = row["pert_id"]
            if identifier in predictions:
                continue
            query = np.packbits(data["fingerprints"][index].astype(np.uint8))
            similarity = library.similarity(query)
            targets = library.score(similarity, selected["selected_cutoff"], selected["selected_trust"], limit=20)
            predictions[identifier] = targets
            stream.write(json.dumps({"pert_id": identifier, "targets": targets}) + "\n")
            stream.flush()
            processed += 1
            if processed % 25 == 0 or processed == remaining:
                seconds = time.monotonic() - start
                print(json.dumps({"stage": "target_features", "done": len(predictions),
                                  "total": len(data["compounds"]), "estimated_seconds_remaining":
                                  round(seconds / processed * (remaining - processed), 1)}), flush=True)
    return predictions, signature


def expression_features(source, cells, compounds, predictions):
    desired = {record["gene"] for records in predictions.values() for record in records}
    columns = pd.read_csv(table_path(source, "gene_expression"), nrows=0).columns.tolist()
    symbols = {column: re.sub(r"\s+\(\d+\)$", "", column).strip() for column in columns if column != "cell_id"}
    selected = [column for column, gene in symbols.items() if gene in desired]
    if not selected:
        raise ValueError("No predicted target genes match the expression matrix")
    expression = pd.read_csv(table_path(source, "gene_expression"), usecols=["cell_id"] + selected)
    if not expression["cell_id"].is_unique or not set(cells) <= set(expression["cell_id"]):
        raise ValueError("Missing or duplicated expression cells")
    values = expression.set_index("cell_id").loc[cells, selected].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite target gene expression")
    genes = sorted(set(symbols[column] for column in selected))
    grouped = np.column_stack([values[:, [index for index, column in enumerate(selected) if symbols[column] == gene]].mean(axis=1) for gene in genes])
    center = grouped.mean(axis=0)
    scale = np.maximum(grouped.std(axis=0), 1e-8)
    standardized = (grouped - center) / scale
    lookup = {gene: index for index, gene in enumerate(genes)}
    features = np.zeros((len(compounds), len(cells), len(FEATURE_NAMES)), dtype=np.float32)
    missing = []
    for index, identifier in enumerate(compounds["pert_id"]):
        targets = predictions[identifier]
        matched = [record for record in targets if record["gene"] in lookup]
        if not matched:
            missing.append(identifier)
            continue
        indices = [lookup[record["gene"]] for record in matched]
        weights = np.array([record["score"] for record in matched], dtype=float)
        total_score = sum(record["score"] for record in targets)
        mapped_weight = weights.sum() / total_score
        weights /= weights.sum()
        features[index, :, 0] = grouped[:, indices] @ weights
        features[index, :, 1] = standardized[:, indices] @ weights
        features[index, :, 2] = standardized[:, indices].max(axis=1)
        features[index, :, 3] = grouped[:, indices[0]]
        features[index, :, 4] = standardized[:, indices[0]]
        features[index, :, 5] = mapped_weight
        features[index, :, 6] = len(matched)
    state = {"target_gene_names": np.array(genes, dtype=str), "target_gene_values": grouped.astype(np.float32),
             "target_gene_center": center, "target_gene_scale": scale, "feature_names": np.array(FEATURE_NAMES)}
    return features, state, missing


def run(arguments):
    import lightgbm as lgb

    root = arguments.root.resolve()
    baseline = (arguments.baseline or root / "results/rank_benchmark_cancer_only").resolve()
    output = (arguments.output or root / "results/target_expression_test").resolve()
    if (output / "report.json").exists():
        raise ValueError("Completed experiment exists; do not overwrite or tune against test results")
    output.mkdir(parents=True, exist_ok=True)
    source = baseline / "input"
    data = prepare(source, root / "data/features", output / "prepared", gene_count=1000)
    incumbent = json.loads((baseline / "benchmark_report.json").read_text())
    previous = incumbent["data"]
    for key in ("source_files", "benchmark_split_sha256", "frozen_split_sha256", "frozen_fingerprints_sha256", "response_counts", "compound_counts"):
        if data["audit"][key] != previous[key]:
            raise ValueError("Response inputs differ from the completed cancer-only experiment: " + key)
    excluded = exclusions(arguments.exclude_genes)
    predictions, configuration = retrieve(data, arguments.library.resolve(), arguments.selection.resolve(), output, excluded)
    extra, state, missing = expression_features(source, data["cells"]["cell_id"].tolist(), data["compounds"], predictions)
    np.savez_compressed(output / "target_expression_features.npz", features=extra, **state)
    partitions = data["partitions"]
    evaluators = {name: RankMetrics(data["matrix"][indices]) for name, indices in partitions.items() if name != "train"}
    parameters = dict(incumbent["models"]["lightgbm"]["configuration"])
    parameters.pop("best_iteration", None)
    parameters["num_threads"] = arguments.threads
    parameters["feature_pre_filter"] = False
    candidates = {}
    training_panel_counts = np.isfinite(data["matrix"][partitions["train"]]).sum(axis=0)

    def features_for(indices, augmented):
        rows, cells = np.where(np.isfinite(data["matrix"][indices]))
        features = pair_features(data, indices[rows], cells)
        if augmented:
            features = np.column_stack([features, extra[indices[rows], cells]])
        return features, rows, cells

    def evaluate(predicted, dataset):
        metrics = evaluators["validation"].vector(predicted)
        return [("macro_drug_spearman", metrics["macro_per_drug_spearman"], True), ("rank_rmse", metrics["rmse"], False)]

    for name, augmented in (("lightgbm_control", False), ("lightgbm_target_expression", True)):
        print("Training", name, flush=True)
        training, rows, cells = features_for(partitions["train"], augmented)
        validation, validation_rows, validation_cells = features_for(partitions["validation"], augmented)
        dataset_parameters = {key: parameters[key] for key in ("max_bin", "num_threads", "feature_pre_filter")}
        training_set = lgb.Dataset(training, label=data["matrix"][partitions["train"]][rows, cells], params=dataset_parameters, free_raw_data=True)
        validation_set = lgb.Dataset(validation, label=data["matrix"][partitions["validation"]][validation_rows, validation_cells], reference=training_set, params=dataset_parameters, free_raw_data=True)
        training_set.construct()
        validation_set.construct()
        del training, validation
        gc.collect()
        model = lgb.train(parameters, training_set, num_boost_round=600, valid_sets=[validation_set], valid_names=["validation"], feval=evaluate,
                          callbacks=[lgb.early_stopping(40, first_metric_only=True, min_delta=0.0001), lgb.log_evaluation(25)])
        validation, _, _ = features_for(partitions["validation"], augmented)
        metrics = evaluators["validation"].vector(model.predict(validation, num_iteration=model.best_iteration))
        candidates[name] = {"validation": metrics, "best_iteration": model.best_iteration, "model": model, "augmented": augmented}
        model.save_model(str(output / (name + ".txt")))
        del validation, training_set, validation_set
        gc.collect()
    control_previous = incumbent["models"]["lightgbm"]["validation"]
    if not np.isclose(candidates["lightgbm_control"]["validation"]["macro_per_drug_spearman"], control_previous["macro_per_drug_spearman"], atol=1e-6, rtol=0):
        raise ValueError("Control does not reproduce original validation ranking; investigate before evaluating test")
    chosen = max(candidates, key=lambda name: selection_key(candidates[name]["validation"]))
    incumbent_metrics = incumbent["models"][incumbent["selected_family"]]["validation"]
    beats_incumbent = selection_key(candidates[chosen]["validation"]) > selection_key(incumbent_metrics)
    selection = {"selected_new_candidate": chosen, "beats_incumbent_on_validation": beats_incumbent,
                 "incumbent": incumbent["selected_model"], "automatic_deployment": False}
    (output / "validation_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print("Frozen selection before test:", json.dumps(selection), flush=True)
    for name, candidate in candidates.items():
        test_features, _, _ = features_for(partitions["test"], candidate["augmented"])
        candidate["test"] = evaluators["test"].vector(candidate["model"].predict(test_features, num_iteration=candidate["model"].best_iteration))
        del test_features
    report = {**selection, "parameters": parameters,
              "models": {name: {key: value for key, value in candidate.items() if key != "model"} for name, candidate in candidates.items()},
              "incumbent_validation": incumbent_metrics,
              "configuration": configuration, "exclusion_csv_sha256": file_hash(arguments.exclude_genes) if arguments.exclude_genes else None,
              "queries_without_expression_mapped_targets": missing,
              "training_observations_per_cell_quantiles": {str(quantile): float(np.quantile(training_panel_counts, quantile)) for quantile in (0, 0.25, 0.5, 0.75, 1)},
              "notes": ["Exploratory test on previously inspected compounds; no independent confirmation or significance claim.",
                        "All query targets are predicted from the LINCS-excluded reference; no true held-out target annotations are features.",
                        "Top-20 target votes are normalized, and expression is summarized by fixed seven features.",
                        "Expression symbols strip Entrez suffixes; duplicate symbol columns are averaged; gene z-scores use cells with training labels only.",
                        "This is a feature-addition ablation, not a causal mechanism or general bias correction. No panel adjustment is claimed.",
                        "An optional excluded-gene list changes the feature definition only; original safety evidence remains in the unmodified reference.",
                        "Existing target hit@5 does not validate a filtered target ranking. Target cutoff/trust settings were previously validation-selected.",
                        "No cancer enrichment p-values or assay-derived agonist/antagonist claims are produced. Deployment remains frozen until inference integration is verified."]}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({**selection, "models": report["models"]}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--exclude-genes", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int, default=8)
    arguments = parser.parse_args()
    if arguments.threads < 1:
        parser.error("threads must be positive")
    run(arguments)
