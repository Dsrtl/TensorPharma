# Common ranked-response benchmark

This experiment compares six response model families on Nasir's cleaned data in `TensorPharma-main/data/processed/`. It refits every candidate on the same labels and preserves the original scaffold assignments. Older raw-score RMSE values are not comparable with these rank-label results. Protein-target retrieval is a separate task and stays outside this response leaderboard.

## Choices

- **Label:** `sensitivity_rank`, a percentile among measured cell lines for the same drug. It describes relative sensitivity, not absolute killing, viability, or a probability of response.
- **Chemical features:** the existing 2,048-bit Morgan fingerprints. Reusing them keeps chemical representation and splits consistent and avoids another standardization run.
- **Cell features:** 1,000 genes selected by variance among cells with training response labels. Imputation and the expression projection use those cells only. The cells are known at prediction time; the held-out entities are whole compounds/scaffolds.
- **Target information:** auxiliary measured activity for the joint model, with training-only normalization and target-head selection. Query compounds' true known targets and MOA annotations are never input features. Per-query target-expression features would need a separate leakage-safe target predictor and are excluded from this first benchmark.
- **Trust weights:** Nasir's supplied file says `scope: all drugs`; it is excluded. Any future learned weighting must be fitted on training data only.
- **Winner:** highest validation mean within-drug Spearman, with lower validation RMSE breaking ties. Each drug contributes equally to the ranking metric. Test results are produced after selecting the candidate and family winner.

## Compared models

| Family | Fixed candidate choices |
|---|---|
| Mean baseline | Global mean and training cell-line mean |
| Similarity | 1, 5, or 10 training neighbours; squared Tanimoto weights |
| Fingerprint ridge | Train-fitted fingerprint projection (up to 64 components), penalties 10/100/1,000 |
| Expression-coupled | Rank-8 shared model, compound/output penalties 10 or 100, expression coupling 10 |
| Joint targets + expression | Same shared model with observed target activity added; missing activities remain masked |
| LightGBM | Fingerprints + 1,000 gene values, regression on ranks; 15 or 31 leaves, L2 10 or 100, minimum leaf size 200 |

The coupled models stop on validation ranking performance with patience five. LightGBM uses a native custom ranking metric, up to 600 trees, learning rate 0.05, and early stopping after 40 rounds without a ranking improvement of 0.0001. Its default RMSE evaluation is disabled so early stopping follows the intended ranking metric. All predictions are clipped to the known [0, 1] label range before selection, scoring, and serving.

The target auxiliary heads keep IC50, Ki, Kd, and EC50 separate, with at least three variable unique training structures per head. Nasir's exported activities already pool assay types, so the benchmark records the assay type as `source_pooled`; that lost distinction cannot be recovered here. If canonical aliases have different reported medians, their median is a representative summary, not the original raw-measurement median.

## Data checks

The provided source has 485,662 drug/cell pairs, 961 source drug IDs, and 985 cell lines. Its percentile ranks reproduce from the raw response scores. The adapter maps source IDs through LINCS IDs into our frozen standardized identities. It excludes entire source IDs with ambiguous identity/partition assignments, flagged structure conflicts, invalid source structures, or source/frozen standardized-structure disagreement. Already-averaged labels cannot safely be reassigned to an arbitrary one of several structures. All exclusions and their reasons are written to `prepared/mapping_exclusions.csv`, and excluded ID/response counts appear in `data_audit.json`. The same filter applies before fitting every model, without consulting response performance. Missing mappings and inconsistent fingerprints still fail the job. Original split assignments and input files are never changed.

Additional aliases of the same standardized compound are merged using source-count-weighted raw scores, and ranks are recomputed afterwards. The resulting canonical drug count can be slightly smaller than 961. New cleaned files are never written over. The adapter excludes cells without training labels, records exclusions, and hashes the source tables, frozen manifest, and fingerprints.

For Spearman evaluation, drugs with constant true labels or fewer than three observations are excluded from ranking and counted. They remain in the RMSE/MAE calculations. A constant prediction for a drug with varying labels receives ranking score zero as an explicit abstention convention; counts are reported. No model gains by silently omitting its constant predictions.

## Run on ARCC

From the Mac project directory, transfer the corrected source folder and scripts. SSH keepalives and partial-file retention help with interrupted transfers:

```bash
rsync -avP --exclude=__pycache__ --exclude=.DS_Store \
  -e "ssh -o ServerAliveInterval=15 -o ServerAliveCountMax=6" \
  TensorPharma-main scripts <user>@<cluster-host>:~/projects/eMerge/
```

On ARCC:

```bash
cd ~/projects/eMerge
mkdir -p results
sbatch scripts/run_rank_benchmark.slurm
squeue -u <user>
```

The job uses account `mpensky`, partition `normal`, eight CPUs, 32 GB RAM, and a two-hour limit. It adds LightGBM 4.x and its Narwhals dependency to the existing environment without replacing the frozen RDKit/numerical packages, then checks dependencies and records installed package versions. No GPU is requested.

The terminal for this coding session cannot reach ARCC directly; submit from the working ARCC terminal. A completed benchmark is protected from accidental overwrite: use a new `--output` directory for a new experiment. `--skip-lightgbm` is for diagnostic checks only and produces `benchmark_complete: false`.

## Outputs and handoff

Under `results/rank_benchmark/`:

- `summary.json`: compact comparison of each family's validation-selected finalist and the overall winner.
- `benchmark_report.json`: full configuration grid, metrics, data hashes, and limitations.
- `prepared/`: canonical ID mapping, frozen split, cleaned rank labels, cell labels, target summaries, and data audit.
- `best_model.npz`: selected numerical model state and preprocessing. `best_model.txt` is also required if LightGBM wins.
- `test_predictions.csv.gz`: all finalist predictions on the same held-out observations; evaluation material, not training data.
- `lightgbm_feature_importance.csv`: descriptive feature importance for the selected LightGBM candidate. Importance is not causal evidence.
- `environment.txt`: installed package versions.

After completion, produce a result for Nasir's API using the same ARCC environment:

```bash
.venv/bin/python scripts/predict_rank.py --smiles 'CCO' --output results/rank_prediction.json
```

The SMILES above is a command example, not a proposed cancer drug. Use a held-out molecule for a real evaluated example. Prediction returns ranked cell lines, cancer-type summaries, training neighbours, and separately reported held-out metrics. It explicitly identifies a molecule-independent mean model if that wins. Cancer-type summaries aggregate model predictions; they do not establish treatment indications.

The test compounds were already inspected in earlier raw-score experiments. These comparisons help choose a hackathon prototype, but they are not a pristine external validation. If the mean baseline remains strongest, retain and report it rather than declaring a more complex model superior.
