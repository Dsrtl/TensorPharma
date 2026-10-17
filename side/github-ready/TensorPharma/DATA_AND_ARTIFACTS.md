# Runtime artifacts and partner handoff

## Already included

- Scientific and data-preparation source code, Slurm templates, and dependency specifications.
- Target metrics/manifest/environment from completed ARCC job 949350.
- Response summary/full report/environment from completed ARCC job 949920.
- Response data audit, mapping exclusions, four low-rank optimization histories, and descriptive LightGBM feature importance.

Source-path strings in copied JSON reports are normalized to a repository-relative path. All metrics, source hashes, and split hashes remain unchanged. No dataset archive, response-label table, test-prediction table, model binary, or target reference is included.

## To run the selected response predictor

Ask the modeling lead for:

```text
results/rank_benchmark/best_model.npz
```

It belongs with the included `benchmark_report.json` and the recorded environment. The selected winner is `expression_coupled_penalty_100`, so it does **not** need a LightGBM text model. A future LightGBM winner would require both NPZ preprocessing and `best_model.txt`.

The NPZ contains model state and data-derived preprocessing/training context. It is excluded from Git by default; agree on redistribution permissions before attaching it to a public release. Private shared storage is sufficient for team integration.

## To run the target predictor

Ask the modeling lead for the completed library directory containing:

```text
manifest.json
fingerprints.npy
labels.csv.gz
```

These are required by the scorer. For readable compound lookup, detailed source evidence, and evaluation reproduction, also obtain:

```text
compounds.csv.gz
evidence.csv.gz
evaluation_queries.csv
```

The selected threshold/configuration is already included as `results/chembl_library/metrics.json`; its embedded manifest must match the library. Use RDKit 2026.09.1.

On the original ARCC project, the library is under the work directory recorded in:

```bash
cat results/chembl-library-workdir.txt
```

Append `/library` to that directory. The original work base was explicitly chosen under project storage because no dedicated scratch directory was configured. Preserve the library before storage cleanup; the small `results/chembl_library/` report folder alone cannot answer queries.

## To reproduce training

Obtain all corrected partner input tables and the original `data/features/` triplet. Verify their hashes against the audit. Six source IDs were excluded before fitting because of unresolved/flagged chemical mappings; do not restore their averaged labels by assigning the first matching structure.

Existing result folders contain completed evidence. Use a new experiment output directory and keep the original reports unchanged. More repeated tuning on this already-inspected test set does not create independent validation.

## Before publishing this folder

1. Review contributors' code ownership and agree on a license; none is assumed.
2. Check `.gitignore` and the staged file list. The included small audit CSVs intentionally contain chemical identifiers and structure strings, not personal records.
3. Keep raw/processed datasets and data-derived runtime artifacts in approved storage unless redistribution terms allow publication.
4. No personal resume/contact details, SSH keys, passwords, or API tokens are bundled. ARCC account/module settings in the runbooks are configuration examples, not credentials; adjust them for your authorized account.
5. Push this folder as the repository root, not its parent project directory. There is no need to push unrelated hackathon experiments or local download archives.
