import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from rank_data import file_hash


def correlation(first, second):
    first = rankdata(first)
    second = rankdata(second)
    first -= first.mean()
    second -= second.mean()
    scale = np.linalg.norm(first) * np.linalg.norm(second)
    return float(first @ second / scale) if scale > 1e-12 else None


def association(frame, protocol, seed):
    grouped = frame.groupby("scaffold", sort=True).agg(score=("target_score", "mean"), effect=("observed_effect", "mean"))
    positive = int(grouped["score"].gt(0).sum())
    zero = int(grouped["score"].eq(0).sum())
    result = {"eligible_test_drugs": len(frame), "scaffolds": len(grouped),
              "positive_vote_scaffolds": positive, "zero_vote_scaffolds": zero,
              "status": "insufficient_coverage", "spearman": None, "p_value": None,
              "ci95_low": None, "ci95_high": None}
    if len(grouped) < protocol["minimum_scaffolds"] or positive < protocol["minimum_positive_vote_scaffolds"] or zero < protocol["minimum_zero_vote_scaffolds"]:
        return result
    score = grouped["score"].to_numpy()
    effect = grouped["effect"].to_numpy()
    observed = correlation(score, effect)
    if observed is None:
        result["status"] = "constant_scores_or_effects"
        return result
    generator = np.random.default_rng(seed)
    centered_score = rankdata(score) - rankdata(score).mean()
    centered_effect = rankdata(effect) - rankdata(effect).mean()
    scale = np.linalg.norm(centered_score) * np.linalg.norm(centered_effect)
    extreme = sum(abs(generator.permutation(centered_score) @ centered_effect / scale) >= abs(observed) - 1e-12 for _ in range(protocol["permutations"]))
    bootstrap = []
    for _ in range(protocol["bootstrap_samples"]):
        indices = generator.integers(0, len(grouped), len(grouped))
        value = correlation(score[indices], effect[indices])
        if value is not None:
            bootstrap.append(value)
    result.update(status="tested_exploratory", spearman=observed,
                  p_value=(extreme + 1) / (protocol["permutations"] + 1),
                  bootstrap_defined=len(bootstrap))
    if len(bootstrap) >= 0.9 * protocol["bootstrap_samples"]:
        result["ci95_low"], result["ci95_high"] = [float(value) for value in np.quantile(bootstrap, [0.025, 0.975])]
    return result


def adjust(results):
    values = np.array([record["p_value"] if record["p_value"] is not None else 1.0 for record in results])
    order = np.argsort(values, kind="stable")
    adjusted = np.minimum.accumulate((values[order] * len(values) / np.arange(1, len(values) + 1))[::-1])[::-1]
    restored = np.empty(len(values))
    restored[order] = np.clip(adjusted, 0, 1)
    for index, record in enumerate(results):
        record["q_value"] = float(restored[index]) if record["p_value"] is not None else None
        record["fdr_association"] = bool(record["q_value"] is not None and record["q_value"] < 0.05)


def cell_group(cells, link):
    if "cell_ids" in link:
        return set(cells.loc[cells["cell_id"].isin(link["cell_ids"]), "cell_id"])
    column = "cancer_subtype" if "cancer_subtype" in link else "cancer_type"
    return set(cells.loc[cells[column] == link[column], "cell_id"])


def contrasts(responses, group, training_means, prediction_column=None):
    records = []
    for identifier, rows in responses.groupby("pert_id", sort=True):
        inside = rows["cell_id"].isin(group)
        outside = ~inside
        if not inside.any() or not outside.any():
            continue
        residual = rows["sensitivity_rank"].to_numpy() - rows["cell_id"].map(training_means).to_numpy()
        record = {"pert_id": identifier, "group_cells": int(inside.sum()), "other_cells": int(outside.sum()),
                  "observed_effect": float(residual[inside].mean() - residual[outside].mean()),
                  "observed_group_mean_rank": float(rows.loc[inside, "sensitivity_rank"].mean())}
        if prediction_column and rows[prediction_column].notna().all():
            predicted = rows[prediction_column].to_numpy() - rows["cell_id"].map(training_means).to_numpy()
            record["predicted_effect"] = float(predicted[inside].mean() - predicted[outside].mean())
            record["predicted_group_mean_rank"] = float(rows.loc[inside, prediction_column].mean())
        records.append(record)
    return pd.DataFrame(records, columns=["pert_id", "group_cells", "other_cells", "observed_effect", "observed_group_mean_rank", "predicted_effect", "predicted_group_mean_rank"])


def run(arguments):
    baseline = arguments.baseline
    target_results = arguments.targets
    output = arguments.output
    if (output / "report.json").exists():
        raise ValueError("Completed Step 3 exists; preserve results and the fixed protocol")
    configuration = json.loads(arguments.links.read_text())
    protocol = configuration["protocol"]
    audit = json.loads((baseline / "prepared/data_audit.json").read_text())
    target_report = json.loads((target_results / "report.json").read_text())
    target_configuration = json.loads((target_results / "configuration.json").read_text())
    if target_configuration != target_report["configuration"]:
        raise ValueError("Target cache configuration does not match the completed experiment")
    for key in ("source_files", "frozen_split_sha256", "benchmark_split_sha256"):
        if target_configuration["data"][key] != audit[key]:
            raise ValueError("Target cache and response source differ: " + key)
    split = pd.read_csv(baseline / "prepared/compound_split.csv", keep_default_na=False)
    if file_hash(baseline / "prepared/compound_split.csv") != audit["benchmark_split_sha256"]:
        raise ValueError("Prepared split hash differs from audit")
    for column in ("inchi_key", "connectivity_key", "scaffold"):
        if split.groupby(column)["split"].nunique().max() != 1:
            raise ValueError("Chemical split leakage")
    targets = {}
    for line in (target_results / "target_predictions.jsonl").read_text().splitlines():
        record = json.loads(line)
        identifier = record["pert_id"]
        if identifier in targets:
            raise ValueError("Duplicate target query")
        if any(not np.isfinite(item["score"]) or item["score"] <= 0 for item in record["targets"]):
            raise ValueError("Invalid target vote")
        if len({item["gene"] for item in record["targets"]}) != len(record["targets"]):
            raise ValueError("Duplicate predicted target genes")
        targets[identifier] = record["targets"]
    if set(targets) != set(split["pert_id"]):
        raise ValueError("Target queries do not match canonical compounds")
    responses = pd.read_csv(baseline / "prepared/rank_responses.csv.gz", usecols=["pert_id", "cell_id", "sensitivity_rank"])
    if responses.duplicated(["pert_id", "cell_id"]).any() or not responses["sensitivity_rank"].between(0, 1).all():
        raise ValueError("Invalid response matrix")
    responses = responses.merge(split[["pert_id", "split", "scaffold", "name"]], on="pert_id", validate="many_to_one")
    cells = pd.read_csv(baseline / "prepared/cells.csv", keep_default_na=False)
    training = responses[responses["split"] == "train"]
    training_means = training.groupby("cell_id")["sensitivity_rank"].mean()
    training_count = int(split["split"].eq("train").sum())
    counts = training.groupby("cell_id")["pert_id"].nunique()
    panel = set(counts[counts >= protocol["training_cell_coverage"] * training_count].index)
    if not panel:
        raise ValueError("Empty training-defined shared panel")
    predicted = pd.read_csv(baseline / "test_predictions.csv.gz", usecols=["pert_id", "cell_id", "observed_rank", "fingerprint_ridge"])
    heldout = responses[responses["split"] == "test"].merge(predicted, on=["pert_id", "cell_id"], how="left", validate="one_to_one")
    if heldout["observed_rank"].isna().any() or len(heldout) != len(predicted) or not np.allclose(heldout["sensitivity_rank"], heldout["observed_rank"], atol=1e-8, rtol=0):
        raise ValueError("Observed/test prediction table mismatch")
    selected = heldout[heldout["cell_id"].isin(panel)].copy()
    observed_counts = selected.groupby("pert_id")["cell_id"].nunique()
    eligible = set(observed_counts[observed_counts >= protocol["query_panel_coverage"] * len(panel)].index)
    selected = selected[selected["pert_id"].isin(eligible)]
    results = []
    details = []
    all_contrasts = {}
    for index, link in enumerate(configuration["links"]):
        group = cell_group(cells, link)
        panel_group = panel & group
        effect = contrasts(selected, panel_group, training_means, "fingerprint_ridge")
        effect = effect[(effect["group_cells"] >= protocol["minimum_observed_group_cells"]) & (effect["other_cells"] >= protocol["minimum_observed_other_cells"])]
        score = {}
        for identifier, records in targets.items():
            total = sum(record["score"] for record in records)
            score[identifier] = sum(record["score"] for record in records if record["gene"] in link["target_genes"]) / total if total else np.nan
        effect["target_score"] = effect["pert_id"].map(score)
        effect = effect.dropna(subset=["target_score"])
        effect = effect.merge(split[["pert_id", "scaffold"]], on="pert_id", validate="one_to_one")
        result = association(effect, protocol, protocol["seed"] + index) if len(panel_group) >= protocol["minimum_group_panel_cells"] else {
            "status": "insufficient_group_cells_in_shared_panel", "eligible_test_drugs": len(effect), "scaffolds": 0,
            "positive_vote_scaffolds": 0, "zero_vote_scaffolds": 0,
            "spearman": None, "p_value": None, "ci95_low": None, "ci95_high": None}
        result.update(link_id=link["id"], kind=link["kind"], group_cells_full_panel=len(group),
                      group_cells_shared_panel=len(panel_group), context_status=link["context_status"])
        results.append(result)
        effect["link_id"] = link["id"]
        details.append(effect)
        all_contrasts[link["id"]] = contrasts(responses, group, training_means).set_index("pert_id")
    adjust(results)
    source_drugs = pd.read_csv(arguments.drugs, keep_default_na=False)
    if file_hash(arguments.drugs) != audit["source_files"]["drugs"]["sha256"]:
        raise ValueError("Drug annotations differ from the benchmark source")
    mapping = pd.read_csv(baseline / "prepared/drug_mapping.csv")
    annotations = mapping.merge(source_drugs[["drug_id", "known_targets", "moa"]], on="drug_id", validate="many_to_one").groupby("pert_id")
    annotated = {identifier: {"known_target_annotations": sorted(set(value for text in rows["known_targets"] for value in text.split("; ") if value)),
                              "moa_annotations": sorted(set(value for text in rows["moa"] for value in text.split("; ") if value))}
                 for identifier, rows in annotations}
    test_contrasts = {link["id"]: contrasts(heldout, cell_group(cells, link), training_means, "fingerprint_ridge").set_index("pert_id") for link in configuration["links"]}
    examples = {name for link in configuration["links"] for name in link["example_drugs"]}
    chains = []
    for row in split.to_dict("records"):
        if row["split"] != "test" and row["name"] not in examples:
            continue
        records = targets[row["pert_id"]]
        links = []
        for link in configuration["links"]:
            matching = [record for record in records if record["gene"] in link["target_genes"]]
            exemplar = row["name"] in link["example_drugs"]
            if not matching and not exemplar:
                continue
            full = all_contrasts[link["id"]]
            observed = full.loc[row["pert_id"]].dropna().to_dict() if row["pert_id"] in full.index else None
            prediction = test_contrasts[link["id"]]
            model = prediction.loc[row["pert_id"]][["predicted_effect", "predicted_group_mean_rank"]].dropna().to_dict() if row["pert_id"] in prediction.index else None
            links.append({"link_id": link["id"], "kind": link["kind"], "literature_evidence_tag": "external_evidence",
                          "target_evidence_tag": "predicted_association", "predicted_targets": matching,
                          "observed_evidence_tag": "measured_response_derived_summary", "observed_full_panel": observed,
                          "model_evidence_tag": "predicted_association", "ridge_prediction_on_observed_test_cells": model,
                          "context_status": link["context_status"],
                          "chain_status": "qualified_hypothesis_not_causal" if matching else "external_example_without_matching_target_prediction",
                          "caution": "Full-panel summaries are descriptive, not additional significance tests. No response means zero observed rows, not inactivity. Binding/action/clinical context must not be inferred from the target vote."})
        if links:
            chains.append({"pert_id": row["pert_id"], "name": row["name"], "split": row["split"],
                           "formal_association_partition": row["split"] == "test", "annotations": annotated.get(row["pert_id"], {}),
                           "annotation_evidence_tag": "source_annotation_not_model_input", "links": links})
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output / "association_results.csv", index=False)
    pd.concat(details, ignore_index=True).to_csv(output / "per_drug_association_inputs.csv", index=False)
    (output / "literature_cards.json").write_text(json.dumps(configuration["links"], indent=2) + "\n")
    (output / "drug_chains.json").write_text(json.dumps(chains, indent=2) + "\n")
    report = {"protocol": protocol, "panel_cells": len(panel), "panel_cell_ids": sorted(panel),
              "test_compounds": int(split["split"].eq("test").sum()), "test_compounds_meeting_panel_coverage": len(eligible),
              "links": results, "chain_compounds": len(chains),
              "source_hashes": {"link_configuration": file_hash(arguments.links), "target_cache": file_hash(target_results / "target_predictions.jsonl"),
                                "target_report": file_hash(target_results / "report.json"), "split": file_hash(baseline / "prepared/compound_split.csv"),
                                "responses": file_hash(baseline / "prepared/rank_responses.csv.gz"), "test_predictions": file_hash(baseline / "test_predictions.csv.gz")},
              "notes": ["Six literature-backed clinical contexts plus two prespecified exploratory repurposing hypotheses; none are model discoveries.",
                        "Primary tests correlate predicted top-20 class vote mass with observed rank residual contrast (group versus other cells); Ridge predictions are not used in hypothesis tests.",
                        "Training-only cell means remove the average cell-response tendency; a training-defined 80%-coverage shared panel and 80% query coverage reduce but do not eliminate screen-panel bias.",
                        "Scaffold-mean correlation, two-sided scaffold permutation, and scaffold bootstrap condition on this fixed cell panel. Permutation requires exchangeability and cannot rule out other observational confounding.",
                        "Top-20 absent target votes are retrieval zeros, not experimentally confirmed non-binding. Entirely unscorable queries are excluded and coverage is reported.",
                        "The eight hypotheses, coverage rules, and minimum scaffold counts were fixed before computing these associations, but these compounds were previously inspected; all results are exploratory.",
                        "BH adjustment uses eight hypotheses, including untested links with p=1. No significance filtering or omission of negative/underpowered links.",
                        "Response direction is not provider-confirmed: effects concern higher/lower source ranks, not demonstrated killing or benefit.",
                        "Anatomical proxies lack several biomarker/action/modality conditions; an association does not validate the corresponding clinical indication or causal mechanism.",
                        "Imatinib is in validation, not test. Training/validation exemplars are tagged and never enter formal test association statistics.",
                        "Statin-glioma and glucocorticoid-mesothelioma ideas have prior literature; they remain unconfirmed repurposing hypotheses, not novel discoveries."]}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"panel_cells": len(panel), "eligible_test_compounds": len(eligible), "links": results}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=Path("partner-handoff/rank_benchmark_cancer_only"))
    parser.add_argument("--targets", type=Path, default=Path("partner-handoff/target_expression_filtered_test"))
    parser.add_argument("--drugs", type=Path, default=Path("TensorPharma-main/data/processed/drugs.csv.gz"))
    parser.add_argument("--links", type=Path, default=Path("config/step3_links.json"))
    parser.add_argument("--output", type=Path, default=Path("results/step3_evidence"))
    run(parser.parse_args())
