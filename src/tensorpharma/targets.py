"""Target prediction by nearest-neighbour retrieval over ChEMBL.

A query SMILES is fingerprinted, compared with every compound in the ChEMBL
library by Tanimoto similarity, and the most similar compounds vote for their
protein targets. Each vote is similarity x strength (pChEMBL) x trust, where trust
comes from data/processed/trust_weights.json (repeated results count more than a
single result).

Safety and metabolism proteins (hERG, liver CYPs, drug pumps) are routinely tested
on most compounds, so they are reported as safety flags, not as targets.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data/processed"

FP_BITS = 2048
FP_RADIUS = 2

# screened for heart risk (KCNH2 = hERG), liver metabolism (CYPs) and drug pumps (ABC transporters)
SAFETY_GENES = {"KCNH2", "CYP3A4", "CYP2D6", "CYP2C9", "CYP2C19", "CYP1A2", "CYP2C8", "CYP2B6",
                "ABCB1", "ABCG2", "ABCC1"}

# vote weightings compared in scripts/eval_targets.py
WEIGHTINGS = ["count", "similarity", "full", "full_popularity"]

_generator = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)


def fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    return _generator.GetFingerprint(mol) if mol is not None else None


class TargetPredictor:
    def __init__(self):
        weights = json.loads((PROCESSED / "trust_weights.json").read_text())["groups"]
        single_trust = weights["1 result"]["weight"]

        rows = pd.read_csv(PROCESSED / "drug_targets.csv.gz",
                           usecols=["drug_id", "smiles", "target_gene", "pchembl_median", "n_measurements"])
        rows = rows.dropna(subset=["target_gene"])
        # one entry per compound-target: average over test types, count all results;
        # trust needs one test type repeated, the way trust_weights.py calibrates it
        per_pair = (rows.groupby(["drug_id", "target_gene"], as_index=False)
                    .agg(strength=("pchembl_median", "mean"), n=("n_measurements", "sum"),
                         n_best=("n_measurements", "max")))
        per_pair["trust"] = np.where(per_pair["n_best"] >= 2, 1.0, single_trust)
        per_pair["weight"] = per_pair["strength"] * per_pair["trust"]
        # how many library compounds were tested on each protein (for the popularity correction)
        self.library_count = per_pair["target_gene"].value_counts().to_dict()

        compounds = rows.drop_duplicates("drug_id")[["drug_id", "smiles"]]
        fps, ids = [], []
        for drug_id, smiles in zip(compounds["drug_id"], compounds["smiles"]):
            fp = fingerprint(smiles)
            if fp is not None:
                fps.append(fp)
                ids.append(drug_id)
        self.fps = fps
        self.ids = np.array(ids)
        self.connectivity = np.array([i[:14] for i in ids])
        index = {d: i for i, d in enumerate(ids)}

        per_pair = per_pair[per_pair["drug_id"].isin(index)]
        per_pair["row"] = per_pair["drug_id"].map(index)
        # targets of each library compound, as (gene, weight, strength, n) lists
        self.targets = {r: g[["target_gene", "weight", "strength", "n"]].to_numpy()
                        for r, g in per_pair.groupby("row")}
        self.row_of = index

    def similarities(self, smiles, exclude_drug_id=None, max_similarity=None):
        """Similarity to every library compound, optionally hiding a drug, its stereo variants and near-copies."""
        fp = fingerprint(smiles)
        if fp is None:
            raise ValueError(f"could not read SMILES: {smiles}")
        sims = np.array(DataStructs.BulkTanimotoSimilarity(fp, self.fps))
        if exclude_drug_id is not None:
            sims[self.connectivity == exclude_drug_id[:14]] = -1.0
        if max_similarity is not None:
            sims[sims >= max_similarity] = -1.0
        return sims

    @staticmethod
    def top_k(sims, k):
        top = np.argpartition(-sims, k)[:k]
        top = top[np.argsort(-sims[top])]
        return top, sims[top]

    def neighbours(self, smiles, exclude_drug_id=None, k=50, max_similarity=None):
        """Top-k most similar library compounds."""
        return self.top_k(self.similarities(smiles, exclude_drug_id, max_similarity), k)

    def vote(self, top, sims, k, weighting="full"):
        """Rank proteins by summed votes over the k nearest compounds.

        count: 1 per neighbour; similarity: similarity; full: similarity x strength x trust;
        full_popularity: full, divided by log of how many library compounds hit the protein.
        """
        scores, evidence = {}, {}
        for row, sim in zip(top[:k], sims[:k]):
            if sim <= 0:
                continue
            for gene, weight, strength, n in self.targets.get(row, ()):
                v = 1.0 if weighting == "count" else sim if weighting == "similarity" else sim * weight
                scores[gene] = scores.get(gene, 0.0) + v
                evidence.setdefault(gene, []).append(
                    {"similar_compound": self.ids[row], "similarity": round(float(sim), 3),
                     "pchembl": round(float(strength), 2), "n_tests": int(n)})
        if weighting == "full_popularity":
            scores = {g: s / np.log(2 + self.library_count.get(g, 0)) for g, s in scores.items()}
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        return [{"gene": g, "score": round(s, 3), "evidence": evidence[g]} for g, s in ranked]

    def own_targets(self, drug_id):
        """The compound's own ChEMBL record, strongest first (the 'look it up' ceiling)."""
        row = self.row_of.get(drug_id)
        if row is None:
            return []
        rows = sorted(self.targets.get(row, ()), key=lambda t: -t[2])
        return [t[0] for t in rows]

    def predict(self, smiles, k=25, top_n=5, exclude_drug_id=None, weighting="full"):
        """Top targets plus safety flags (safety proteins that would have ranked in the top 10)."""
        top, sims = self.neighbours(smiles, exclude_drug_id=exclude_drug_id, k=max(k, 1))
        ranked = self.vote(top, sims, k, weighting)
        targets = [t for t in ranked if t["gene"] not in SAFETY_GENES][:top_n]
        safety = [t for t in ranked[:10] if t["gene"] in SAFETY_GENES]
        return targets, safety
