# Inputs excluded from the GitHub handoff

The original frozen chemical feature directory must be obtained from the modeling lead:

```text
data/features/compound_split.csv
data/features/fingerprints.npz
data/features/feature_report.json
```

Preserve all three together and use the recorded RDKit version. Original split SHA256: `1400e5c0fbab24ed66d5253a1b042dd97876d5e885177a2413e5cce00c00e1a3`. Do not regenerate a different split and compare its scores as if they were the same experiment.

The current rank benchmark also needs the partner's processed CSV tables described in `TensorPharma-main/data/README.md`.

Historical scripts additionally use `data/prepared/`, `data/expression/`, `data/chembl/`, and `data/chembl_matched/`. They are not prerequisites for querying the already-trained rank-response model. See `MODELING.md` for those earlier experiments. The historical raw downloads and 2 GB ChEMBL dump are not included.
