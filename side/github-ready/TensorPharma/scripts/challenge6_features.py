import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold


def standardize(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("SMILES could not be parsed")
    molecule = rdMolStandardize.Cleanup(molecule)
    molecule = rdMolStandardize.FragmentParent(molecule)
    molecule = rdMolStandardize.Uncharger().uncharge(molecule)
    molecule = rdMolStandardize.TautomerEnumerator().Canonicalize(molecule)
    Chem.SanitizeMol(molecule)
    if molecule.GetNumHeavyAtoms() == 0:
        raise ValueError("No heavy atoms after standardization")
    return molecule


def fingerprint(molecule):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    values = np.zeros(2048, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(generator.GetFingerprint(molecule), values)
    return values


def assign_splits(compounds, seed):
    groups = sorted(compounds["scaffold"].unique())
    if len(groups) < 3:
        raise ValueError("At least three scaffold groups are required")
    generator = np.random.default_rng(seed)
    generator.shuffle(groups)
    counts = compounds["scaffold"].value_counts()
    desired = {"test": 0.15 * len(compounds), "validation": 0.15 * len(compounds)}
    assignment = {}
    cursor = 0
    for split in ("test", "validation"):
        selected = 0
        while selected < desired[split] and cursor < len(groups) - (2 if split == "test" else 1):
            group = groups[cursor]
            assignment[group] = split
            selected += int(counts[group])
            cursor += 1
    for group in groups[cursor:]:
        assignment[group] = "train"
    compounds["split"] = compounds["scaffold"].map(assignment)
    if set(compounds["split"]) != {"train", "validation", "test"}:
        raise ValueError("Split construction produced an empty partition")
    if compounds.groupby("scaffold")["split"].nunique().max() != 1:
        raise AssertionError("Scaffold leakage")
    if compounds.groupby("connectivity_key")["split"].nunique().max() != 1:
        raise AssertionError("Equivalent connectivity appears in different partitions")
    return compounds


def prepare_features(root, output, seed):
    source = root / "data" / "prepared" / "compounds.csv"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    signature = {"input_sha256": source_hash, "seed": seed, "rdkit_version": rdBase.rdkitVersion,
                 "standardization": "cleanup_fragment_parent_uncharge_canonical_tautomer_v1"}
    report_path = output / "feature_report.json"
    if report_path.exists():
        previous = json.loads(report_path.read_text())
        if previous["configuration"] != signature:
            raise ValueError("Frozen feature configuration changed; use a new output directory")
        if all((output / name).exists() for name in ("compound_split.csv", "fingerprints.npz")):
            print("Existing frozen features retained:", report_path)
            return
    records = []
    vectors = []
    rejected = []
    for record in pd.read_csv(source, keep_default_na=False).to_dict("records"):
        try:
            molecule = standardize(record["smiles"])
            canonical = Chem.MolToSmiles(molecule, isomericSmiles=True)
            inchi_key = Chem.MolToInchiKey(molecule)
            if not inchi_key:
                raise ValueError("InChIKey generation failed")
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False)
            records.append({**record, "standardized_smiles": canonical, "inchi_key": inchi_key,
                            "connectivity_key": inchi_key[:14], "scaffold": scaffold or "__ACYCLIC__"})
            vectors.append(fingerprint(molecule))
        except (ValueError, RuntimeError) as error:
            rejected.append({"pert_id": record["pert_id"], "reason": str(error)})
    if not records:
        raise ValueError("No valid molecules")
    compounds = assign_splits(pd.DataFrame(records), seed)
    output.mkdir(parents=True, exist_ok=True)
    compounds.to_csv(output / "compound_split.csv", index=False)
    np.savez_compressed(output / "fingerprints.npz", pert_ids=compounds["pert_id"].to_numpy(dtype=str),
                        fingerprints=np.stack(vectors))
    pd.DataFrame(rejected, columns=["pert_id", "reason"]).to_csv(output / "rejected_compounds.csv", index=False)
    report = {"configuration": signature, "valid_compounds": len(compounds), "rejected_compounds": rejected,
              "unique_standardized_structures": int(compounds["standardized_smiles"].nunique()),
              "scaffold_groups": int(compounds["scaffold"].nunique()),
              "split_counts": compounds["split"].value_counts().to_dict(),
              "notes": ["All acyclic molecules share one scaffold group.",
                        "Scaffold and connectivity groups are disjoint across splits.",
                        "Split is compound-cold, cell-warm; it does not evaluate unseen cell lines.",
                        "Standardization can discard salts, charge, and tautomer distinctions."]}
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    arguments = parser.parse_args()
    prepare_features(arguments.root, arguments.output or arguments.root / "data" / "features", arguments.seed)
