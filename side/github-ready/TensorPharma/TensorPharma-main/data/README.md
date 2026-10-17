# Partner dataset layout

Place the separately shared cleaned files here:

```text
processed/
  drugs.csv.gz
  drug_ids.csv.gz
  responses.csv.gz
  cells.csv.gz
  gene_expression.csv.gz
  drug_targets.csv.gz
  build_audit.json
  trust_weights.json
```

No dataset files are included in this GitHub-ready handoff. Use the same corrected dataset as the completed reports; source hashes are recorded in `results/rank_benchmark/prepared/data_audit.json` and the ChEMBL manifest.

The source dataset contains 961 raw drug identities, 485,662 response pairs, 985 cell lines, 28 cancer types, and 19,215 genes. The adapter excludes unresolved source identities and merges additional canonical aliases, yielding the 954 canonical drugs reported by the current benchmark.

The all-drug `trust_weights.json` is not used as a learned predictor input. The full-library target experiment separately compared a fixed optional count heuristic on validation; its selected configuration does not use that heuristic.

The original preparation scripts under `TensorPharma-main/scripts/` expect additional `data/raw/` inputs and an intermediate ChEMBL activity CSV, with source locations specified in those scripts. They are included for transparency, not as a claim that raw data is bundled. Rebuilding from raw sources requires their access permissions and the corresponding schema.
