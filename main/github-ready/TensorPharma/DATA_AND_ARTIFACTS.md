# Runtime artifacts and partner handoff

## Already included

- Scientific and data-preparation source code, Slurm templates, and dependency specifications.
- Target metrics/manifest/environment from completed ARCC job 949350.
- Final cancer-only response summary/full report/environment and dataset-change report from completed ARCC job 950090.
- Earlier mixed-cell response reports from ARCC job 949920, retained separately as historical evidence.
- Completed target-expression test (job 950385): compact summary, full machine-produced report, validation selection, and environment. The 45-gene filter hash was verified locally. Added features did not improve the matched control; 124 of 954 drugs had no expression-mapped target features.
- Completed local Step 3: source-backed literature cards, eight observed-response association outcomes, 47 evidence-tagged drug records, fixed configuration, script, and documented examples. See `STEP3.md`; this is not causal MoA validation.
- Response data audit, mapping exclusions, four low-rank optimization histories, and descriptive LightGBM feature importance.

Source-path strings in copied JSON reports are normalized to a repository-relative path. All metrics, source hashes, and split hashes remain unchanged. No dataset archive, response-label table, test-prediction table, model binary, target reference, or cancer-only `input/` directory is included.

## To run the selected response predictor

Ask the modeling lead for:

```text
results/rank_benchmark_cancer_only/best_model.npz
```

It belongs with the included cancer-only `benchmark_report.json` and recorded environment. The final selected winner is **`ridge_1000`**, so it does **not** need a LightGBM text model. The earlier mixed-cell `expression_coupled_penalty_100` artifact is not the final integration model. A future LightGBM winner would require both NPZ preprocessing and `best_model.txt`.

Always use `predict_rank.py --benchmark results/rank_benchmark_cancer_only`; the CLI default still selects the earlier directory. Do not copy an artifact between reports: the exporter checks selected-model agreement.

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

Obtain all corrected partner input tables and the original `data/features/` triplet. For final reproduction, obtain the six-file snapshot in `results/rank_benchmark_cancer_only/input/`, or reconstruct it with the updated `cells.csv.gz`, `responses.csv.gz`, and `gene_expression.csv.gz` plus the original unchanged `drugs.csv.gz`, `drug_ids.csv.gz`, and `drug_targets.csv.gz`. Verify hashes against the cancer-only audit. The three updated tables can be staged under `data/updated_cancer_cell_data/` for the supplied Slurm script.

This update removes nine `Non-Cancerous` cells and recomputes within-drug ranks. It does not demonstrate general bias removal or cancer-versus-normal selectivity. The final panel contains 976 cells. Six source IDs and 3,720 response rows were separately excluded because of unresolved/flagged chemical mappings; do not restore their averaged labels by assigning the first matching structure.

Existing result folders contain completed evidence. Use a new experiment output directory and keep the original reports unchanged. More repeated tuning on this already-inspected test set does not create independent validation.

## Before publishing this folder

For the additional target-expression test, the small `report.json`, `validation_selection.json`, `environment.txt`, and reconciled `summary.json` are already included under `results/target_expression_filtered_test/`. Do not copy the entire experimental output folder into Git: it also contains model binaries, feature arrays, cached predictions, and prepared labels. Scheduler completion details are drawn from the modeling lead's shared terminal output, not fabricated as part of the machine report.

1. Review contributors' code ownership and agree on a license; none is assumed.
2. Check `.gitignore` and the staged file list. The included small audit CSVs intentionally contain chemical identifiers and structure strings, not personal records.
3. Keep raw/processed datasets and data-derived runtime artifacts in approved storage unless redistribution terms allow publication.
4. No personal resume/contact details, SSH keys, passwords, or API tokens are bundled. ARCC account/module settings in the runbooks are configuration examples, not credentials; adjust them for your authorized account.
5. Push this folder as the repository root, not its parent project directory. There is no need to push unrelated hackathon experiments or local download archives.
