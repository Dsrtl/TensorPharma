# Challenge 6 modelling

The current shared-data comparison is described in `RANK_BENCHMARK.md`. It uses Nasir's cleaned `sensitivity_rank` labels and compares six response families, including LightGBM. The earlier raw-score experiments below remain historical results and use different target values and ranking axes.

Owner: math/model lead, with ARCC access. Nasir owns ChEMBL preparation, cancer metadata, API, and UI integration.

## Run locally or on ARCC

From the project root, create an isolated environment and install dependencies:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-challenge6.txt
.venv/bin/python scripts/challenge6_features.py
.venv/bin/python scripts/challenge6_baseline.py
```

Use an available Python 3.10+ interpreter on ARCC. The small baseline runs on a CPU; it does not need a GPU allocation. Package installation needs network access. Baseline fitting uses the prepared data and does not require ChEMBL or a copy of the expression matrix.

The feature step standardizes structures, calculates 2,048-bit radius-2 Morgan fingerprints, and freezes approximately 70/15/15 train/validation/test compound splits by scaffold (seed 2026). Fractions can differ because whole scaffolds stay together. All acyclic molecules share one group. Connectivity keys and scaffold groups must not cross partitions. A rerun retains existing features when configuration matches and refuses to silently replace a different configuration. Preserve `data/features/` when moving to ARCC.

## Baselines

- Training-set cell-line mean, with a global training mean fallback.
- Tanimoto nearest-neighbour response prediction: up to five observed training compounds per cell line, weighted by squared similarity; training cell means are the fallback.
- Independent per-cell ridge regression on up to 64 fingerprint principal components, fitted on training compounds only. Choose ridge regularization from 1, 10, and 100 using validation RMSE.

Select the deployed baseline using validation RMSE across those approaches, then report untouched test performance. All models use the same compound split. There is no requirement that a more complex model win. Test predictions, all comparison metrics, and the selected model are written to `results/baseline/`. The saved fit uses training compounds only, so it matches the reported test evaluation.

Metrics: RMSE, MAE, and median within-cell Spearman over cells with at least five observations and nonconstant labels/predictions. Constant cell-mean predictions have undefined within-cell correlation; this is reported as null, not zero. This tests unseen compounds in known cell lines, not unseen cell lines. Cell lines with no training labels are excluded and listed in the report. The aggregate RMSE weights each observed compound/cell pair equally.

The prepared file's score semantics still need confirmation before interpreting descending scores as stronger sensitivity. Do not call raw regression scores probabilities. No target annotations or MOA strings are included as features.

## Prediction handoff to Nasir

After training:

```bash
.venv/bin/python scripts/predict_challenge6.py --smiles 'CCO' --output results/example_prediction.json
```

The SMILES above is only a command example, not a proposed cancer drug. Use a held-out compound from `data/features/compound_split.csv` for an actual demo evaluation.

Python entry point: `predict(smiles, model_path, limit=20)` in `scripts/predict_challenge6.py`, with `scripts/` on the Python module search path. Output includes:

- Standardized query SMILES, model name, split hash, and whether the standardized structure occurs in training.
- Ranked `cell_lines`: DepMap `model_id`, `predicted_sensitivity_score`, and available nearby-compound measurements.
- `similar_training_compounds`: LINCS IDs, names, standardized SMILES, and Tanimoto similarities.
- Held-out model performance separately from individual scores.
- Explicitly pending target, cancer-type, and mechanism components.

Similarity examples explain structural proximity, not causation. The current predictions are sorted by descending raw score, pending confirmation of score direction. Models are not uncertainty-calibrated.

## Next: multilayer comparison

Reuse the frozen compound split. Add CCLE cell features and a coupled model linking compound fingerprints, compound/cell response, and eventually ChEMBL target activity. Compare against these baselines on the same validation/test partitions; keep target records for held-out compounds out of training and retrieval. Nasir must receive the standardized structures and scaffold assignments when constructing the ChEMBL training/reference set. A shared embedding does not by itself establish a biological mechanism.

Still needed from Nasir: filtered compound/target activity with standardized structures and source IDs, plus `ModelID` to cancer-type metadata.

## ARCC deployment

SSH: `<user>@<cluster-host>`. Project: `~/projects/eMerge`. Slurm account: `mpensky`; partition: `normal`.

`<user>`, `<cluster-host>` and the Slurm account are placeholders for the cluster this work ran on. Substitute the values for your own authorized account before running anything below.

Create the project directory on ARCC, then transfer from the Mac project directory:

```bash
rsync -av --progress scripts data requirements-challenge6.txt MODELING.md <user>@<cluster-host>:~/projects/eMerge/
```

The baseline uses `data/prepared/` and optional frozen `data/features/`. It does not need the large raw downloads or ChEMBL yet. Do not copy the Mac virtual environment to Linux.

On ARCC, load the available GCC Python 3.11.4 module before creating the Linux environment. The batch script loads the same module:

```bash
cd ~/projects/eMerge
module load python/python-3.11.4-gcc-12.2.0
python3 --version
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-challenge6.txt
mkdir -p results
sbatch scripts/run_baseline.slurm
squeue -u "$USER"
```

The supplied job requests four CPUs, 8 GB RAM, and one hour on `normal`, with no GPU. It runs feature preparation and baseline fitting, records package versions, and writes a log to `results/baseline-JOBID.log`. Create `results/` before submitting so Slurm can open its log. If the cluster rejects the account or resource request, inspect the scheduler error rather than changing the allocation arbitrarily.

## First expression-coupled experiment

The first ARCC baseline selected the cell-line mean by validation RMSE. Similarity predictions had modest within-cell ranking signal. This does not establish useful molecule-specific prediction, and the mean predictor cannot rank compounds within a cell line.

Before the next model, `scripts/audit_challenge6.py` reports score quantiles by partition and the compounds contributing most test squared error. These are diagnostics, not a reason to remove difficult observations. The local full-data audit found a median score of approximately 0.086, a maximum of 96, and only 0.94% of observations above 10. Confirm score units and direction with the provider.

Prepare a portable expression pack from the original CCLE CSV on the Mac:

```bash
python3 scripts/prepare_expression.py
rsync -av scripts MODELING.md <user>@<cluster-host>:~/projects/eMerge/
rsync -av --relative data/expression/ <user>@<cluster-host>:~/projects/eMerge/
```

On ARCC, preserve the existing frozen split and baseline artifacts, then submit:

```bash
cd ~/projects/eMerge
sbatch scripts/run_multilayer.slurm
```

The model predicts the training-cell mean plus `fingerprint_features @ compound_map @ cell_factors.T`. Compound features use the baseline's train-fitted projection, normalized with training-only scales. The cell factors are regularized toward `expression_features @ expression_map`. Gene selection, imputation, and expression projection use only cells with training response labels. Default settings are rank 8, at most 25 alternating sweeps, and validation-based early stopping with patience 5.

The objective is the sum of squared observed normalized response residuals, plus `10 * ||compound_map||²`, `||cell_factors||²`, `coupling * ||cell_factors - expression_features @ expression_map||²`, and `||expression_map||²`. Missing responses are masked throughout. Optimization alternates coordinate ridge updates for the compound map, per-cell ridge updates for cell factors, and a ridge expression map.

Compare a zero-coupling low-rank control with expression coupling 10. Choose epochs and the candidate model using validation RMSE only. The original baseline stays deployed if it wins. The candidate artifact is `results/multilayer/sensitivity_multilayer.npz`, and its report is `results/multilayer/metrics.json`. Prediction accepts that artifact through `--model`; a candidate that failed to beat the baseline is explicitly labelled `candidate_only`.

This is the sensitivity component before the ChEMBL target layer is available. It is not yet the full compound–target–cancer model. The already-inspected test results make follow-up test comparisons exploratory; do not repeatedly tune against them or reset the split to get a better score.

## ChEMBL extraction without a PostgreSQL server

`scripts/run_chembl_extract.slurm` runs the PostgreSQL 16 tools inside `containers/postgres16.sif` using Apptainer, with Go parallelism limited to two. It reads `chembl_37_postgresql.tar.gz` from the project root and extracts the dump into `data/chembl/source/`. It needs two CPUs, 8 GB RAM, and requests up to two hours. It does not request GPUs or run a database server.

The extractor uses `pg_restore --data-only --table=... --file=-` to stream individual tables. It parses PostgreSQL COPY text and stores selected records in a new SQLite database; source SQL is never executed. Locale variables are set to C for both host and container processes.

Filters: human single-protein targets, assay confidence at least 9, exact IC50/Ki/Kd/EC50 measurements with finite pChEMBL values, no validity comments, and no flagged potential duplicates. Activities without a usable source SMILES are removed. Repeated measurements and endpoint types remain separate, retaining activity and assay IDs. No standardization or training/test split is applied to these raw reference structures yet.

Outputs under `data/chembl/`:

- `chembl_filtered.sqlite`: normalized targets, eligible assays, activities, molecules, target gene symbols, and protein accessions.
- `molecules.csv.gz`: `molregno`, ChEMBL ID, compound name, canonical SMILES, standard InChIKey.
- `activities.csv.gz`: source activity ID, `molregno`, assay ID, target `tid`, endpoint, pChEMBL, standard value/units, assay confidence/type. All retained relations are exact equality.
- `targets.csv.gz`, `target_genes.csv.gz`, `target_accessions.csv.gz`: target identifier crosswalks and biological annotations.
- `extraction_report.json`: counts, source archive size, filters, and limitations.

After transferring the two extraction scripts from the Mac:

```bash
cd ~/projects/eMerge
mkdir -p results
sbatch scripts/run_chembl_extract.slurm
```

Monitor `results/chembl-extract-JOBID.log`. The extractor reuses a same-size extracted dump with a valid PostgreSQL header, builds filtered files in a temporary directory, and publishes them after successful filtering/export. It refuses to overwrite an existing filtered database. A failed extraction can be resubmitted; use a different `--output` directory to intentionally rebuild a completed database.

Next, standardize reference structures with the same RDKit pipeline, match to LINCS compounds, and exclude all validation/test scaffold groups from any training/retrieval reference. The full filtered export is raw data, not an evaluation-safe training set. Nasir can consume the CSV exports without installing PostgreSQL.

## Match ChEMBL and establish target prediction

The first ARCC extraction retained 1,188,949 measurements for 655,004 molecules and 4,516 measured human protein targets. `scripts/run_targets.slurm` next runs `match_chembl.py` and `challenge6_targets.py` on the existing frozen split.

Matching first screens source connectivity keys and SMILES against the LINCS source/standardized structures, then standardizes candidate ChEMBL structures with the same RDKit version and pipeline. Only full standardized InChIKey matches are accepted. Connectivity-only candidates are exported separately and excluded; the prefilter may miss transformations that change connectivity. Report match coverage rather than assuming all 972 compounds have target labels.

Outputs in `data/chembl_matched/` include the accepted compound crosswalk, excluded connectivity-only candidates, source activity records, endpoint-specific medians, target labels, compound coverage, and `matching_report.json`. A target is marked known active when at least one median within an endpoint/assay-type group reaches pChEMBL 6. Endpoint disagreement is recorded. Raw measurements remain available; different endpoints are not averaged together.

The first target baseline deliberately uses only matched training structures, not all 655,004 ChEMBL molecules. This is a quick, evaluation-safe starting point. LINCS aliases are collapsed by full standardized InChIKey. A common-target prior is compared with similarity-weighted target voting using 1, 3, 5, or 10 positive-labelled neighbours. Validation known-target recall@5 selects the model; hit@5 breaks ties. Target test labels do not affect selection or training artifacts.

Report hit@1, hit@5, and macro known-target recall@5 on unique held-out structures with known active labels, alongside label coverage. Targets absent from the training vocabulary count as misses. Unknown/unmeasured targets do not become negatives; these metrics cannot establish precision against every possible target. Prior scores do not depend on the molecule, and similarity votes are not calibrated probabilities or affinities.

Submit on ARCC after transferring the new scripts:

```bash
cd ~/projects/eMerge
sbatch scripts/run_targets.slurm
```

Results go to `results/targets/metrics.json` and `target_baseline.npz`. `training_activity_evidence.csv.gz` contains train-only source records for explanations. `test_target_predictions.csv` is evaluation material and must not be used as retrieval evidence. Next, use the matched measured activity relationships to build the joint target/sensitivity experiment without exposing held-out labels during training.

## Joint target and sensitivity experiment

The matching run found exact standardized identities for 652 of 972 LINCS compounds, with 19,221 source activity rows. The target retrieval baseline selected 10 neighbours on validation labels. On 84 unique labelled test structures it retrieved at least one known active target in the top five for 32 structures (38.1%), compared with 12 (14.3%) for the common-target prior. This is hit@5 on a labelled subset, not precision or overall cancer-prediction accuracy. Known-target macro recall@5 was approximately 19.9%.

`scripts/challenge6_joint.py` now shares a fingerprint-to-latent map between sensitivity and protein-activity prediction. In normalized units, its relationships are:

```text
cell response       ≈ fingerprint_features @ compound_map @ cell_factors.T
protein activity    ≈ fingerprint_features @ compound_map @ target_factors.T
cell factors        ≈ expression_features @ expression_map
```

Each target head corresponds to one protein, endpoint type, and assay type; IC50, Ki, Kd, and EC50 are not pooled. Keep heads with at least three unique training structures and variable training pChEMBL values. Collapse LINCS aliases for target supervision, preserving all original sensitivity observations. Target means, scales, and head vocabulary are fitted on training labels only. All missing measurements stay masked.

The loss sums observed normalized sensitivity squared errors and weighted observed normalized activity squared errors. The activity weight is the number of observed training sensitivity labels divided by the number of observed training activity labels, balancing the two aggregate losses. Penalties are `10 * ||compound_map||²`, `||output_factors||²`, `10 * ||cell_factors - expression_features @ expression_map||²`, and `||expression_map||²`. The expression penalty applies only to cell factors. Fit alternating ridge updates, with default rank 8, at most 25 sweeps, and validation patience 5.

Run both zero activity weight (expression-only control) and the balanced activity weight (joint model). Select checkpoints and the candidate using sensitivity validation RMSE, then compare with the deployed baseline. No target validation/test labels enter optimization, preprocessing, or checkpoint selection. Auxiliary held-out pChEMBL errors are reported only on supported measured heads; they do not replace the separately validated target-retrieval metrics.

Submit `sbatch scripts/run_joint.slurm` from the ARCC project root after transferring the updated scripts. It requests four CPUs, 8 GB RAM, and one hour. Outputs are `results/joint/metrics.json` and `results/joint/sensitivity_joint.npz`; prior experiment files are preserved. The existing prediction script can read the candidate through `--model`, and the original baseline stays deployed if validation RMSE remains better. Follow-up test comparisons remain exploratory because those test compounds have already been inspected.

## Bounded regularization check

The initial joint model improved sensitivity validation RMSE from 2.276 to 2.171 versus the expression-only control, but remained worse than the cell-mean baseline (1.436). It also had worse auxiliary pChEMBL error than a target-mean prediction. Both sensitivity fits selected epoch one while training error continued falling; this is consistent with overfitting, not evidence that additional training will help.

`run_regularization.slurm` checks three stronger penalty settings: compound and output penalties both set to 100, 1,000, or 10,000. Rank, split, expression coupling, target loss weight, and the two-model comparison stay fixed. The modified solver retains the previous 10/1 defaults when flags are absent. This is a bounded follow-up experiment, not a claim that shrinkage will solve the problem.

Each run uses `--validation-only`, computes no test metrics, and publishes no model artifact. Reports and optimization histories go to separate directories under `results/regularization/`. `summarize_regularization.py` checks split/configuration consistency and writes a compact `summary.json`. Selection uses validation RMSE only. If a candidate improves on the baseline, the summary includes a command to reproduce that setting for final evaluation in `results/joint_regularized/`. If the baseline still wins, retain it and report the negative multilayer result alongside the separately validated target-retrieval result. Small validation differences do not establish meaningful generalization gains.
