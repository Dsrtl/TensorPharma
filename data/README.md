# Data

## Processed tables (committed)

Built by the scripts in `scripts/`, in this order:

```
python3 scripts/export_chembl.py    # ChEMBL dump -> data/processed/chembl_activity.csv (local only, 256 MB)
python3 scripts/build_dataset.py    # joins everything -> the tables below + build_audit.json
python3 scripts/trust_weights.py    # evidence reliability -> trust_weights.json
python3 scripts/activity_type_check.py  # why Potency/ED50/AC50 are cut -> activity_type_check.json
```

| File | One row = | Built from |
|---|---|---|
| `processed/drugs.csv.gz` | one drug (961), keyed by InChIKey | compoundinfo |
| `processed/drug_ids.csv.gz` | LINCS `pert_id` -> `drug_id` (972) | compoundinfo |
| `processed/drug_targets.csv.gz` | compound + protein + test type, median pChEMBL (1.02M) | ChEMBL 37 |
| `processed/cells.csv.gz` | one cell line with cancer type (985) | DepMap Model.csv + CCLE |
| `processed/gene_expression.csv.gz` | one cell line x 19,215 genes | CCLE |
| `processed/responses.csv.gz` | drug x cell line sensitivity, raw and ranked within drug (485,662) | PharmacoDB |
| `processed/build_audit.json` | rows in / kept / dropped at every step | |
| `processed/trust_weights.json` | evidence reliability weights | |
| `processed/activity_type_check.json` | share of ChEMBL rows hitting a known target, per test type | |

`.csv.gz` files are compressed CSV: pandas reads them directly, and double-clicking one on macOS unzips it.

## Raw data (not committed)

```
data/raw/DrugSensitivity/   from the hackathon Data.zip: LINCS StandardAnnotations, CCLE expression,
                            PharmacoDB sensitivity file (L1000 .gctx not used)
data/raw/Bioactivity/       from the hackathon Data.zip: ChEMBL 37 PostgreSQL dump
data/raw/DepMap/Model.csv   DepMap 24Q4 Public, https://ndownloader.figshare.com/files/51065297
                            (MD5 675210d17675f3517b0ce39a3c274f16)
```

## Sources and licenses

- ChEMBL 37 (EMBL-EBI), CC BY-SA 3.0: https://www.ebi.ac.uk/chembl/
- DepMap 24Q4 Public and CCLE expression (Broad Institute), CC BY 4.0: https://depmap.org/
- LINCS L1000 compound annotations: https://lincsportal.ccs.miami.edu/
- PharmacoDB drug sensitivity scores: https://pharmacodb.ca/
