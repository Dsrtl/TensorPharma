import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def read_rows(path, delimiter=",", encoding="utf-8-sig"):
    with path.open(newline="", encoding=encoding) as stream:
        yield from csv.DictReader(stream, delimiter=delimiter)


def prepare(root, output):
    annotations = root / "StandardAnnotations"
    structures = defaultdict(set)
    names = {}
    for record in read_rows(annotations / "compoundinfo.txt", "\t"):
        identifier = record["pert_id"].strip()
        smiles = record["canonical_smiles"].strip()
        if smiles and smiles.lower() not in {"nan", "na", "-666"}:
            structures[identifier].add(smiles)
        names[identifier] = record["cmap_name"]

    fallback = defaultdict(set)
    for record in read_rows(annotations / "Small_Molecule_Metadata.txt", "\t", "latin-1"):
        smiles = record["sm_smiles_parent"].strip()
        if smiles and smiles.lower() not in {"nan", "na", "-666"}:
            for identifier in {record["sm_name"], record["sm_center_canonical_id"]}:
                if identifier.startswith("BRD-"):
                    fallback[identifier].add(smiles)

    expression_path = root / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
    expression_models = set()
    default_models = Counter()
    expression_rows = 0
    for record in read_rows(expression_path):
        expression_rows += 1
        identifier = record["ModelID"].strip()
        expression_models.add(identifier)
        if record["IsDefaultEntryForModel"].strip().lower() in {"yes", "true", "1"}:
            default_models[identifier] += 1

    output.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    compounds = set()
    cells = set()
    matched_compounds = set()
    matched_cells = set()
    pairs = Counter()
    missing_structures = set()
    missing_expression = set()
    resolved_compounds = {}
    minimum_score = math.inf
    maximum_score = -math.inf
    response_path = root / "L1000_cp_pertid_Cellosaurus_celllineid_PharmacoDB_dss3.csv"
    with (output / "matched_responses.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "pert_id", "model_id", "sensitivity_score"
        ])
        writer.writeheader()
        for record in read_rows(response_path):
            counts["response_rows"] += 1
            compound = record["pert_id"].strip()
            cell = record["cell_line"].strip()
            compounds.add(compound)
            cells.add(cell)
            pairs[(compound, cell)] += 1
            candidates = structures.get(compound) or fallback.get(compound, set())
            has_structure = len(candidates) == 1
            has_expression = cell in expression_models
            if has_structure:
                counts["rows_with_unambiguous_smiles"] += 1
            else:
                missing_structures.add(compound)
            if has_expression:
                counts["rows_with_expression"] += 1
            else:
                missing_expression.add(cell)
            try:
                score = float(record["sensitivity_score"])
            except ValueError:
                counts["invalid_score_rows"] += 1
                continue
            if not math.isfinite(score):
                counts["invalid_score_rows"] += 1
                continue
            minimum_score = min(minimum_score, score)
            maximum_score = max(maximum_score, score)
            if not (has_structure and has_expression):
                continue
            resolved_compounds[compound] = {
                "pert_id": compound,
                "name": names.get(compound, ""),
                "smiles": next(iter(candidates)),
                "structure_source": "compoundinfo" if structures.get(compound) else "Small_Molecule_Metadata",
            }
            writer.writerow({"pert_id": compound, "model_id": cell, "sensitivity_score": score})
            counts["matched_response_rows"] += 1
            matched_compounds.add(compound)
            matched_cells.add(cell)

    with (output / "compounds.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["pert_id", "name", "smiles", "structure_source"])
        writer.writeheader()
        writer.writerows(resolved_compounds[key] for key in sorted(resolved_compounds))

    report = {
        **dict(counts),
        "response_compounds": len(compounds),
        "response_cell_lines": len(cells),
        "matched_compounds": len(matched_compounds),
        "matched_cell_lines": len(matched_cells),
        "expression_rows": expression_rows,
        "expression_models": len(expression_models),
        "matched_models_without_exactly_one_default_expression_row": sorted(
            identifier for identifier in matched_cells if default_models[identifier] != 1
        ),
        "duplicate_response_pair_extra_rows": sum(count - 1 for count in pairs.values()),
        "missing_or_ambiguous_structure_ids": sorted(missing_structures),
        "missing_expression_ids": sorted(missing_expression),
        "observed_score_min": minimum_score if math.isfinite(minimum_score) else None,
        "observed_score_max": maximum_score if math.isfinite(maximum_score) else None,
        "notes": [
            "Response cell_line already contains DepMap ACH IDs; join to expression ModelID.",
            "Use IsDefaultEntryForModel to select expression rows; inspect exceptions in this report.",
            "SMILES are metadata strings, not yet chemically validated or standardized.",
            "Conflicting SMILES strings within the preferred source are excluded pending chemical validation.",
            "Response observations are preserved; duplicate pairs must stay together in validation splits.",
            "Cancer-type labels and ChEMBL target activity are not joined by this script.",
            "Confirm response score meaning before interpreting predictions.",
        ],
    }
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if not isinstance(value, list)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    prepare(arguments.root, arguments.output or arguments.root / "data" / "prepared")
