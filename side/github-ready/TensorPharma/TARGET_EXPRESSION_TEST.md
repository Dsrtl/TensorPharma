# One additional experiment: predicted target × expression

## Completed outcome

ARCC job **950385** completed successfully in **12 minutes 43 seconds**, exit `0:0`. The matched LightGBM control reproduced validation Spearman **0.307368** and test Spearman **0.325227**. Adding the seven target-expression summaries gave **0.305685** validation and **0.323724** test Spearman; test RMSE also worsened slightly (0.262837 versus 0.262547). Validation retained the control among these two candidates, and neither displaced the incumbent **Ridge (`ridge_1000`)**.

No performance improvement, statistical significance, causal mechanism, or general failure of biological features is established. The result is specific to this fixed feature addition on previously inspected compounds. Stage C enrichment and Stage D mechanism validation were not run.

The included `results/target_expression_filtered_test/summary.json` is reconciled with the downloaded machine-produced `report.json`. Validation selection and environment files are included too. The recorded 45-gene exclusion-file SHA256 matches `config/target_exclusions.csv`. Of 954 compounds, **124 had no expression-mapped predicted target genes** and therefore received zero target-expression summaries; these are missing-support cases, not confirmed inactive compounds. Absolute input paths in the copied report are normalized; metrics and hashes are preserved.

## What is new

The completed six-family benchmark did not feed Stage-A **predicted** target scores into the response model. Its joint model used measured training activity as auxiliary supervision; that is a different idea.

This bounded experiment tests whether seven fixed target/expression summaries improve LightGBM beyond the same fingerprint + 1,000-expression-gene baseline. It does not rebuild ChEMBL, reset the split, retrain an LLM, run a broad hyperparameter search, or replace the final selected Ridge artifact automatically.

## Corrections to the partner runbook

- The final cancer-only winner is **Ridge (`ridge_1000`)**, not expression coupling. Retain it as the incumbent.
- Nine non-cancer lines were already removed using subtype, including BPH-1. The final panel has 976 cells. Do not remove all fibroblast lineages or apply a second rank transformation.
- The three updated tables plus the unchanged drug/target tables are already snapshotted under `results/rank_benchmark_cancer_only/input/`. Preserve their hashes and frozen chemical assignments.
- Do not recreate `.venv`, install a different RDKit, assume `$SCRATCH` exists, or modify read-only source tables. Use the existing ARCC environment/work directory.
- The selected Stage-A settings are **cutoff 0.4 with no trust weighting**. Retain that selection instead of silently enabling weights. Its reference excludes every frozen LINCS identity/connectivity plus validation/test scaffolds, not merely 702 raw overlaps.
- The partner supplied the exact 45-gene exclusion list separately, with the corrupted CYP2E1 entry confirmed. It is stored in `config/target_exclusions.csv`. The original reference and safety/off-target evidence remain intact; the list filters this experiment's feature votes only. A filtered target ranking needs its own target evaluation before reusing the earlier hit@5 claim.
- Target affinity endpoints alone do not establish agonism, antagonism, or inhibition of a particular biological function. ChEMBL exposes curated mechanism/action information separately. Do not turn IC50/Ki/Kd/EC50 directly into an MoA action label. See [ChEMBL mechanisms and assays](https://chembl.gitbook.io/chembl-interface-documentation/web-services/chembl-data-web-services).
- The test compounds have already been inspected. This follow-up is exploratory; do not call the test untouched or claim independent confirmation.

## Fixed design

1. Load exactly the cancer-only dataset and verify input hashes/counts and compound assignments against the completed benchmark.
2. Predict top-20 target genes for all 954 benchmark compounds using **only the LINCS-excluded reference**. True query target/MOA annotations never become features. Check identity/connectivity exclusion again against the saved reference.
3. Strip trailing Entrez suffixes such as ` (1234)` from expression columns. Average duplicate-symbol columns. Match predicted genes to the stored known cell panel.
4. Normalize query target votes across the top 20. For mapped genes, compute expression summaries; absent-target queries get zeros and explicit coverage reporting.
5. Fit one LightGBM control and one augmented model with the same settings as the completed cancer-only LightGBM: 31 leaves, L2 100, 600-round maximum, validation early stopping, fixed seed. No new grid.
6. Require the control's validation correlation to reproduce the old result within 1e-6. If not, stop before evaluating test and investigate.
7. Freeze candidate selection using validation before computing test metrics. Compare validation with the incumbent Ridge, but do not automatically deploy the experimental model.

Seven added features per compound/cell pair:

| Feature | Meaning |
|---|---|
| Target-weighted log expression | Weighted mean of the predicted target genes' supplied expression values |
| Target-weighted gene z-score | Weighted relative over/under-expression of those genes |
| Maximum target-gene z-score | Most relatively over-expressed matched target |
| Top mapped target expression | Expression of the highest-scoring target that has a gene match |
| Top mapped target z-score | Relative expression of that target |
| Mapped vote mass | Fraction of top-20 target votes represented by expression genes |
| Mapped target count | Number of top-20 target genes matched to expression |

Gene means/scales use only cells with training response labels. Evaluation remains compound-cold, cell-warm. This experiment does **not** claim to correct screened-panel imbalance, prove causality, or establish target dependence. A useful result would be a validation improvement over the matched control; uncertainty and previously used test data limit stronger claims.

## ARCC run

On the Mac, transfer current scripts:

```bash
rsync -avP --exclude=__pycache__ scripts \
  <user>@<cluster-host>:~/projects/eMerge/
```

On ARCC:

```bash
cd ~/projects/eMerge
mkdir -p results
sbatch scripts/run_target_expression_test.slurm
squeue -u "$USER"
```

Requests 8 CPUs / 32 GB / a two-hour maximum, no GPU. Uses the library location recorded in `results/chembl-library-workdir.txt`. Target scoring across all compounds adds work beyond the earlier five-minute response-only fit; no measured duration is claimed before it runs.

Logs: `results/target-expression-JOBID.log`. Every 25 new query predictions, the log reports done/total and an estimated number of seconds left in **target scoring only**. The cached JSONL predictions permit a restart with identical inputs after a failed run; completed reports are protected from overwrite. A truncated cache line fails fast rather than silently omitting a molecule.

Results: `results/target_expression_test/report.json`, plus validation selection, provenance, target-score cache, target-expression features, and experimental LightGBM models. Keep these data-derived artifacts out of Git until a curated report is prepared. For a filtered experiment, supply the exact CSV first:

```bash
export TP_EXCLUDE_GENES="$HOME/projects/eMerge/data/anti_target_genes.csv"
sbatch scripts/run_target_expression_test.slurm
```

That uses a separate `results/target_expression_filtered_test/` folder. Without the list, the unfiltered experiment still tests the new cross-data feature idea. Do not compare filtered/unfiltered test metrics to repeatedly choose exclusions.

### Partner list received

`config/target_exclusions.csv` contains the **45 symbols** supplied by the partner. One entry arrived as `20,00 CYP`; the source counts (98 compounds / 107 rows) matched `CYP2E1`, and the partner explicitly confirmed that correction. `config/target_exclusions_notes.json` records its provenance.

For this supplied feature-scope experiment, use `TP_EXCLUDE_GENES="$HOME/projects/eMerge/config/target_exclusions.csv"`. Listed genes remain available as safety/off-target evidence in the unchanged original reference. Steroidogenic CYPs and CA9/CA12 are not included in the supplied exclusion file. Excluding a gene from oncology features does not establish that it is biologically irrelevant to cancer.

## What this does not implement

Stage C enrichment and Stage D MoA/explanations remain separate integration work. Fisher/Mann–Whitney p-values on predicted responders are not independent evidence of drug efficacy; multiple-testing correction does not fix correlated cell lines, screen-panel selection, or circular expression-based explanations. Cancer-type summaries can be descriptive and explicitly tagged as predictions until a suitable validation protocol exists.

MoA learning would need a curated gene-class/action mapping, training-only annotation fitting, identity/scaffold-safe neighbour voting, coverage accounting, and ambiguity handling. ChEMBL-absent compounds are not automatically structurally novel or a clean independent test set. Endpoint-derived action labels and expression correlations must not be presented as experimentally established mechanism.

The safe output chain is **predicted target association → predicted cell-response pattern → descriptive cancer grouping → evidence-tagged mechanism hypothesis**, not a claim that a compound treats a cancer.
