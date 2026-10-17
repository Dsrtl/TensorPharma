import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def prepare(root):
    source = root / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
    output = root / "data" / "expression"
    desired = set(pd.read_csv(root / "data" / "prepared" / "matched_responses.csv", usecols=["model_id"])["model_id"])
    columns = pd.read_csv(source, nrows=0).columns.tolist()
    genes = columns[6:]
    types = {gene: np.float32 for gene in genes}
    model_ids = []
    matrices = []
    for chunk in pd.read_csv(source, chunksize=64, dtype=types):
        selected = chunk["ModelID"].isin(desired) & chunk["IsDefaultEntryForModel"].astype(str).str.lower().isin(["yes", "true", "1"])
        retained = chunk.loc[selected]
        model_ids.extend(retained["ModelID"].tolist())
        matrices.append(retained[genes].to_numpy(dtype=np.float32))
    if len(model_ids) != len(set(model_ids)):
        raise ValueError("Multiple default expression profiles for a model")
    if set(model_ids) != desired:
        raise ValueError("Missing default expression profiles: " + str(sorted(desired - set(model_ids))))
    matrix = np.concatenate(matrices)
    order = np.argsort(model_ids)
    model_ids = np.array(model_ids, dtype=str)[order]
    matrix = matrix[order]
    if np.isinf(matrix).any():
        raise ValueError("Infinite expression values")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "ccle_matched.npz", model_ids=model_ids,
                        genes=np.array(genes, dtype=str), expression=matrix)
    report = {"cell_lines": len(model_ids), "genes": len(genes),
              "missing_values": int(np.isnan(matrix).sum()),
              "notes": ["Default expression profiles only; input values preserved without applying another log transform.",
                        "This pack contains raw features only. Fit gene selection, imputation, and projection inside training."]}
    (output / "expression_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    arguments = parser.parse_args()
    prepare(arguments.root)
