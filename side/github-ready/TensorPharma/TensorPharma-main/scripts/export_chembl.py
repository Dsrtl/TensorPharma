"""Export a clean drug-protein potency table from the ChEMBL 37 PostgreSQL dump.

Reads tables straight out of the dump with pg_restore (no database server needed),
keeps human single-protein targets with direct assay assignments and exact pChEMBL
values, maps salts to their parent compound, and writes one row per
(compound, target, activity type) with the median pChEMBL.

Usage (from the repo root):
    python3 scripts/export_chembl.py
"""

import csv
import json
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DUMP = ROOT / "data/raw/Bioactivity/chembl_37/chembl_37_postgresql/chembl_37_postgresql.dmp"
OUT_CSV = ROOT / "data/processed/chembl_activity.csv"
OUT_AUDIT = ROOT / "data/processed/chembl_activity_audit.json"
PG_RESTORE = "/Applications/Postgres.app/Contents/Versions/latest/bin/pg_restore"

ORGANISM = "Homo sapiens"
TARGET_TYPE = "SINGLE PROTEIN"
MIN_CONFIDENCE = 9  # 9 = assay assigned directly to a single protein

_UNESCAPE = {"b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t", "v": "\v", "\\": "\\"}


def _unescape(field):
    if field == "\\N":
        return None
    if "\\" not in field:
        return field
    out, i = [], 0
    while i < len(field):
        c = field[i]
        if c == "\\" and i + 1 < len(field):
            out.append(_UNESCAPE.get(field[i + 1], field[i + 1]))
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def read_table(name, columns):
    """Yield dicts of the requested columns for every row of a dumped table."""
    proc = subprocess.Popen(
        [PG_RESTORE, "--data-only", "--table", name, "--file", "-", str(DUMP)],
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        bufsize=1 << 20,
    )
    idx = None
    for line in proc.stdout:
        if idx is None:
            if line.startswith(f"COPY public.{name} "):
                header = line[line.index("(") + 1 : line.index(")")]
                names = [c.strip().strip('"') for c in header.split(",")]
                idx = [names.index(c) for c in columns]
            continue
        if line.startswith("\\."):
            break
        parts = line.rstrip("\n").split("\t")
        yield {c: _unescape(parts[i]) for c, i in zip(columns, idx)}
    proc.stdout.close()
    if proc.wait() != 0:
        sys.exit(f"pg_restore failed reading table {name}")
    if idx is None:
        sys.exit(f"table {name} not found in dump")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    audit = {}

    # Targets: human single proteins only
    targets = {}
    for r in read_table("target_dictionary", ["tid", "chembl_id", "pref_name", "target_type", "organism"]):
        if r["target_type"] == TARGET_TYPE and r["organism"] == ORGANISM:
            targets[r["tid"]] = {"chembl_id": r["chembl_id"], "name": r["pref_name"]}
    audit["targets_kept"] = len(targets)
    log(f"targets kept: {len(targets)}")

    tid_component = {}
    for r in read_table("target_components", ["tid", "component_id"]):
        if r["tid"] in targets:
            tid_component.setdefault(r["tid"], r["component_id"])
    components = set(tid_component.values())

    accession = {}
    for r in read_table("component_sequences", ["component_id", "accession"]):
        if r["component_id"] in components:
            accession[r["component_id"]] = r["accession"]

    gene = {}
    for r in read_table("component_synonyms", ["component_id", "component_synonym", "syn_type"]):
        if r["syn_type"] == "GENE_SYMBOL" and r["component_id"] in components:
            current = gene.get(r["component_id"])
            if current is None or r["component_synonym"] < current:
                gene[r["component_id"]] = r["component_synonym"]

    # Assays: direct single-protein assignments to kept targets
    assays = {}
    for r in read_table("assays", ["assay_id", "tid", "confidence_score"]):
        if r["tid"] in targets and r["confidence_score"] and int(r["confidence_score"]) >= MIN_CONFIDENCE:
            assays[r["assay_id"]] = r["tid"]
    audit["assays_kept"] = len(assays)
    log(f"assays kept: {len(assays)}")

    # Salts and other forms point to their parent compound
    parent = {}
    for r in read_table("molecule_hierarchy", ["molregno", "parent_molregno"]):
        if r["parent_molregno"]:
            parent[r["molregno"]] = r["parent_molregno"]

    # Activities: exact pChEMBL values with no validity flags
    values = defaultdict(list)
    seen = kept = 0
    drop = defaultdict(int)
    for r in read_table(
        "activities",
        ["assay_id", "molregno", "standard_relation", "standard_type",
         "pchembl_value", "data_validity_comment", "potential_duplicate"],
    ):
        seen += 1
        if r["pchembl_value"] is None:
            drop["no_pchembl"] += 1
            continue
        if r["assay_id"] not in assays:
            drop["target_or_assay_filtered"] += 1
            continue
        if r["standard_relation"] != "=":
            drop["not_exact"] += 1
            continue
        if r["data_validity_comment"] is not None:
            drop["validity_flag"] += 1
            continue
        if r["potential_duplicate"] == "1":
            drop["potential_duplicate"] += 1
            continue
        mol = parent.get(r["molregno"], r["molregno"])
        values[(mol, assays[r["assay_id"]], r["standard_type"])].append(float(r["pchembl_value"]))
        kept += 1
    audit["activities_seen"] = seen
    audit["activities_kept"] = kept
    audit["activities_dropped"] = dict(drop)
    log(f"activities seen {seen}, kept {kept}")

    needed = {k[0] for k in values}
    chembl_id = {}
    for r in read_table("molecule_dictionary", ["molregno", "chembl_id"]):
        if r["molregno"] in needed:
            chembl_id[r["molregno"]] = r["chembl_id"]

    structure = {}
    for r in read_table("compound_structures", ["molregno", "standard_inchi_key", "canonical_smiles"]):
        if r["molregno"] in needed:
            structure[r["molregno"]] = (r["standard_inchi_key"], r["canonical_smiles"])

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    written = no_structure = 0
    with OUT_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["inchikey", "chembl_molecule_id", "smiles", "target_chembl_id", "target_gene",
                    "target_uniprot", "target_name", "activity_type", "pchembl_median", "n_measurements"])
        for (mol, tid, atype), vals in sorted(values.items()):
            inchikey, smiles = structure.get(mol, (None, None))
            if not inchikey:
                no_structure += 1
                continue
            comp = tid_component.get(tid)
            w.writerow([inchikey, chembl_id.get(mol), smiles, targets[tid]["chembl_id"], gene.get(comp),
                        accession.get(comp), targets[tid]["name"], atype,
                        round(statistics.median(vals), 3), len(vals)])
            written += 1

    audit["rows_written"] = written
    audit["rows_dropped_no_structure"] = no_structure
    audit["unique_compounds"] = len({k[0] for k in values} & set(structure))
    audit["unique_targets"] = len({k[1] for k in values})
    audit["filters"] = {"organism": ORGANISM, "target_type": TARGET_TYPE, "min_confidence": MIN_CONFIDENCE,
                        "relation": "=", "validity_comment": "none", "salts": "mapped to parent"}
    OUT_AUDIT.write_text(json.dumps(audit, indent=2))
    log(f"wrote {written} rows to {OUT_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
