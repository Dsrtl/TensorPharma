# TensorPharma

**eMerge AI Hackathon 2026 — chemical structure → protein targets → cancer-cell response**

Given a molecule's SMILES string, TensorPharma retrieves likely protein targets and predicts relative response ranks across a known panel of cancer cell lines. Cancer-type summaries group those cell-line predictions. A planned application will present the results and evidence together.

This folder is a **code, documentation, and results handoff**, not a complete deployment bundle. Large datasets, fitted model binaries, reference-library artifacts, and private files are deliberately excluded. The API, UI, local-LLM explanations, MCP integration, and molecule generation are not implemented here.

## Start here

| Partner task | Read |
|---|---|
| Understand the project and completed results | This README |
| Review the final cancer-only comparison | [Response summary](results/rank_benchmark_cancer_only/summary.json), [full report](results/rank_benchmark_cancer_only/benchmark_report.json) |
| Review the earlier mixed-cell comparison | [Historical summary](results/rank_benchmark/summary.json) |
| Review protein-target retrieval | [Target metrics](results/chembl_library/metrics.json), [reference manifest](results/chembl_library/manifest.json) |
| Get the missing data/models for integration | [DATA_AND_ARTIFACTS.md](DATA_AND_ARTIFACTS.md) |
| Reproduce the response experiment | [RANK_BENCHMARK.md](RANK_BENCHMARK.md) |
| Build/score the full target library | [CHEMBL_LIBRARY.md](CHEMBL_LIBRARY.md) |
| Review earlier experiments | [MODELING.md](MODELING.md), which includes historical—not current—results |

## What has actually completed

All four ARCC jobs below finished successfully on October 17, 2026. **The cancer-only Ridge model remains the final response model for integration**; the additional feature test did not improve its matched control.

| Experiment | Job | Runtime | Outcome |
|---|---:|---:|---|
| Expanded ChEMBL target retrieval | 949350 | 57 min 56 sec | Completed, exit `0:0` |
| Earlier mixed-cell response comparison | 949920 | 5 min 29 sec | Completed, exit `0:0`; historical |
| Final cancer-only response comparison | 950090 | 5 min 22 sec | Completed, exit `0:0`; selected `ridge_1000` |
| Target-expression feature test | 950385 | 12 min 43 sec | Completed, exit `0:0`; added features did not help |

### 1. Protein-target retrieval

The reference contains **498,671 standardized compounds** and **674,883 compound/gene pairs**. Queries are compared with radius-2, 2,048-bit Morgan fingerprints using Tanimoto similarity. Qualifying neighbours vote for protein genes.

- **67/84 labelled test drugs (79.8%)** had at least one known target among the top five predictions.
- **47.9% macro known-target recall@5**: average fraction of a drug's known positive target genes recovered in its top five.
- **3/84 abstentions** because no neighbour passed the selected cutoff.
- **84/145 test structures** had known active-gene labels and could enter this evaluation; 81/141 validation structures were labelled.
- Validation selected **similarity cutoff 0.4, without trust weighting**.
- All frozen LINCS identities/connectivities and validation/test scaffolds were excluded from references.

**This is hit@5 on labelled compounds, not 79.8% accuracy over all drugs or proteins.** Unknown target labels are not negatives. Scores are similarity votes, not affinities or probabilities. Previously inspected test compounds make these results exploratory.

### 2. Cancer-cell-response comparison

Six model families used the same cancer-only data, structure handling, and original frozen compound/scaffold assignments. Nine cell lines labelled `Non-Cancerous` and their 1,957 source response rows were removed. Retained raw response and expression values were unchanged, and within-drug percentile ranks were recomputed. This is a scope correction, not proof of general bias removal; it prevents a cancer-versus-normal selectivity assessment on this panel. See the [dataset change report](results/rank_benchmark_cancer_only/dataset_change_report.json).

Candidate selection used validation macro within-drug Spearman, with validation RMSE breaking ties. Final results:

| Family | Selected candidate | Validation Spearman | Test Spearman | Test RMSE |
|---|---|---:|---:|---:|
| **Fingerprint ridge — selected overall** | `ridge_1000` | **0.307394** | 0.318915 | 0.263563 |
| LightGBM | `lightgbm_leaves_31` | 0.307368 | 0.325227 | 0.262547 |
| Expression-coupled | `expression_coupled_penalty_100` | 0.304637 | 0.309855 | 0.265274 |
| Cell-average baseline | `cell_mean` | 0.296002 | 0.309785 | 0.264650 |
| Joint targets + expression | `joint_targets_expression_penalty_100` | 0.295086 | 0.307355 | 0.265244 |
| Chemical similarity | `tanimoto_10` | 0.270241 | 0.278410 | 0.271808 |

**Interpretation:** Ridge won validation over LightGBM by only about **0.000026 Spearman**—an essentially tied result, not demonstrated scientific superiority. LightGBM scored highest on test, but we do not change the selected winner using test results. Ridge's improvement over the cell-average baseline is modest. The multilayer models did not win, and adding protein-target supervision did not help under this configuration. No statistical significance or clinical usefulness has been established.

The previous mixed-cell run selected expression coupling by a narrow validation margin. Its original reports remain under `results/rank_benchmark/`; they are not overwritten. Different cell panels define different ranking tasks, so the cancer-only version is chosen for its intended scope, not because its score happens to be better.

Spearman is ranking correlation, **not percentage accuracy**. Prediction targets are relative cell-line response percentiles within a drug—not absolute cell killing, viability, or clinical response probabilities. Higher predicted ranks follow the source score direction; its biological interpretation still needs provider confirmation.

Final benchmark data:

| Partition | Canonical compounds | Response observations |
|---|---:|---:|
| Training | 670 | 333,114 |
| Validation | 140 | 71,817 |
| Test | 144 | 74,640 |

There are 954 canonical compounds, 976 known cell lines, and 1,000 selected gene features. Six unresolved/flagged source drug IDs and their 3,720 cancer-only response rows were excluded before fitting every model; further canonical duplicate aggregation also changes row counts. See the [audit](results/rank_benchmark_cancer_only/prepared/data_audit.json) and [exclusion records](results/rank_benchmark_cancer_only/prepared/mapping_exclusions.csv). Source files and original scaffold assignments were not rewritten.

## How it works

```text
SMILES → RDKit cleanup / parent fragment / uncharge / canonical tautomer
       → Morgan fingerprint
           ├── ChEMBL chemical neighbours → likely protein genes + evidence
           └── selected response model → ranked known cell lines
                                      → cancer-type summaries

Planned API/UI joins both outputs; a local LLM may explain supplied evidence.
```

The multilayer implementation is a coupled low-rank model, not a graph neural network. A fingerprint encoder supplies compound factors; cell factors are regularized toward gene-expression features. The joint variant shares compound factors with observed protein-activity heads. Missing observations remain masked, not negative labels.

The LightGBM candidate uses **2,048 fingerprint bits + 1,000 gene values**. True query target and MOA annotations are not input features. The large L1000 `.gctx` perturbation file is not used.

## Responsibilities

- **Math/modeling lead:** mathematical models, evaluation, ARCC execution, and prediction artifacts.
- **Nasir:** data preparation/integration, cancer metadata, application/API, and evidence-grounded local-model explanations.
- **Both:** interpretation, presentation, and reproducibility.

Local LLMs explain evidence; they do not generate the numerical predictions. APIs serve the scientific tools. MCP is optional integration after those tools work. Challenge 7 molecule generation remains a future extension, not a completed feature.

## Layout

```text
scripts/                       Modeling, preparation, prediction, Slurm jobs
TensorPharma-main/scripts/      Partner's original data-preparation scripts
TensorPharma-main/data/README.md Required private/local input-table layout
data/README.md                 Frozen-feature and historical-input requirements
results/rank_benchmark_cancer_only/ Final response metrics, audit, histories
results/rank_benchmark/         Earlier mixed-cell results, preserved
results/chembl_library/         Completed target metrics and provenance
requirements-challenge6.txt    Base dependency ranges
requirements-reproducible.txt  Core package versions from the completed ARCC job
DATA_AND_ARTIFACTS.md          How to obtain missing runtime artifacts
.gitignore                    Dataset/model/cache exclusions
```

Raw datasets, training labels, test-prediction tables, fitted model binaries, reference evidence, and environment directories are intentionally not included. Reports' absolute source paths are normalized to their repository-relative input locations; numerical results and original input hashes are preserved.

## Environment

The completed jobs used Linux Python 3.11.4 and RDKit 2026.09.1. On a supported machine, create a fresh environment:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-reproducible.txt
.venv/bin/python -m pip check
```

This is a convenience pin set for the core runtime, not a universal cross-platform lockfile. The recorded `environment.txt` files contain the fuller actual installed environment. macOS LightGBM may require a system OpenMP runtime. Do not replace the existing frozen ARCC environment just to use this handoff folder.

Before numerical imports on an ARCC login node, prevent excess library threads:

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
```

The supplied Slurm templates use account `mpensky`, partition `normal`, and module `python/python-3.11.4-gcc-12.2.0`. These are cluster-specific settings, not portable defaults. The templates request CPUs; no GPU is required for the current models.

## Reproduction and inference

First obtain the cancer-only input tables and original frozen feature artifacts described in [DATA_AND_ARTIFACTS.md](DATA_AND_ARTIFACTS.md). A clone alone cannot retrain or serve predictions. The selected final model is **Ridge (`ridge_1000`)**, not the earlier expression-coupled artifact.

The included `results/` directories are completed evidence, protected from overwrite. For a deliberately new response run, use a different output directory inside an appropriate compute allocation:

```bash
.venv/bin/python scripts/rank_benchmark.py \
  --source results/rank_benchmark_cancer_only/input \
  --output results/rank_benchmark_cancer_only_reproduction --threads 8
```

Then score a query against that run:

```bash
.venv/bin/python scripts/predict_rank.py \
  --benchmark results/rank_benchmark_cancer_only_reproduction \
  --smiles 'CCO' --output results/example_response.json
```

Alternatively, obtain the actual final `best_model.npz`, place it in `results/rank_benchmark_cancer_only/`, and score without retraining:

```bash
.venv/bin/python scripts/predict_rank.py \
  --benchmark results/rank_benchmark_cancer_only \
  --smiles 'CCO' --output results/example_response.json
```

Always pass `--benchmark results/rank_benchmark_cancer_only` for final integration: the script's default still points at the earlier mixed-cell directory. `CCO` is only a command example, not a candidate cancer drug.

For an existing target library, with its directory supplied as `LIBRARY_PATH`:

```bash
.venv/bin/python scripts/chembl_library.py score \
  --library "$LIBRARY_PATH" \
  --selection results/chembl_library/metrics.json \
  --queries queries.csv --output results/example_targets.csv
```

`queries.csv` needs unique `query_id` values and a `smiles` column. See [CHEMBL_LIBRARY.md](CHEMBL_LIBRARY.md) for fresh builds and dependent Slurm arrays. Preserve original handoff reports before running scripts that copy new results to their default paths.

## Integration handoff

The response exporter returns standardized SMILES, ranked cell IDs/names, cancer types, predicted ranks, cancer-type summaries, similar training compounds, model identity, and held-out metrics. It supports the stored cell panel, not arbitrary new cell lines.

Target CSVs include gene, vote score, nearest similarity, supporting potency maximum, evidence counts, and abstentions. Full evidence records live in the separately transferred library. The maximum supporting potency need not come from the nearest neighbour; do not present them as one assay.

The planned application should distinguish **measured evidence**, **predicted associations**, and **mechanism hypotheses**. A query identical to a reference compound produces known-compound evidence, not proof of novel generalization.

## Scientific and sharing limits

### Additional experiment completed: no improvement

The [target-expression test](TARGET_EXPRESSION_TEST.md) compared the fixed cancer-only LightGBM control with the same model plus seven summaries of **predicted target votes × cell expression**. The protocol reuses the LINCS-excluded reference and frozen split, with the partner-confirmed 45-gene feature exclusion in `config/target_exclusions.csv`, while preserving the original safety/off-target reference. This differs from the earlier joint auxiliary-target model.

| Model | Validation Spearman | Test Spearman | Test RMSE |
|---|---:|---:|---:|
| LightGBM matched control | **0.307368** | **0.325227** | **0.262547** |
| + seven target-expression features | 0.305685 | 0.323724 | 0.262837 |

The control reproduced the previous cancer-only LightGBM metrics. The added features performed slightly worse on both partitions, so **Ridge remains selected**. This does not show biological information is generally unhelpful; it concerns this feature design and tested configuration only. No statistical significance or Stage C/Stage D mechanism validation is implied.

The [summary](results/target_expression_filtered_test/summary.json), [full machine-produced report](results/target_expression_filtered_test/report.json), validation selection, and recorded environment are included. Numerical results were reconciled with the downloaded report; the 45-gene exclusion-file hash matches the local configuration. **124 of 954 compounds** had no expression-mapped predicted targets and received zero added features; absence of retrieved/mapped targets is not evidence of biological inactivity. Absolute input paths were normalized without changing metrics or hashes. Scheduler status comes from the modeling lead's shared `sacct` output.

- Compounds/scaffolds are held out, but cell lines are known during training; there is no unseen-patient validation.
- Test compounds were previously inspected. These are exploratory hackathon results, not an untouched external benchmark.
- Incomplete protein labels prevent a general precision/false-positive claim.
- Source activity medians pool some assay information; original assay IDs cannot be reconstructed from the partner table.
- Cancer-type summaries do not establish clinical indications, safety, selectivity, or efficacy.
- No normal-cell toxicity evaluation, prospective experiment, or causal-mechanism validation is provided.
- Dataset access does not imply redistribution rights. Verify source terms before publishing data-derived model/reference artifacts.

No repository license is invented here. Agree on a code license with all contributors before public redistribution.
