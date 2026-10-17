# TensorPharma — partner handoff

## Read this first

**Final architecture:** use the validation-selected **Ridge (`ridge_1000`)** for relative cancer-cell response, the separate ChEMBL similarity library for protein targets, and the evidence-tagged Step 3 outputs for qualified explanations. No additional model tuning is required for this handoff.

## Completed work

| Component | Result |
|---|---|
| Protein-target retrieval | 498,671 reference compounds; top five contain a known target for 67/84 labelled test drugs (79.8% hit@5), 47.9% average known-target recall@5 |
| Six-model cancer-only comparison | Ridge selected by validation, essentially tied with LightGBM; 0.3189 test Spearman, not 31.89% accuracy |
| Target-expression feature addition | Did not improve the matched LightGBM control; Ridge retained |
| Step 3 | Six external clinical-context cards, two exploratory hypothesis cards, eight reported association outcomes, and 47 per-drug evidence records |

Step 3 found one ERBB2–breast **anatomical-proxy association** surviving the eight-link correction (rho 0.337, q 0.00560), but HER2 status is absent, only seven positive-vote scaffolds support the test, and confounding remains possible. This is not proof of a causal mechanism or clinical usefulness. EGFR–head-and-neck did not survive correction. Other links were nonsignificant or underpowered. Neither repurposing hypothesis was validated. See [STEP3.md](STEP3.md) for all outcomes, context, and sources.

## What each partner should use

### Nasir / app and API

1. Use the **final cancer-only** Ridge artifact, not the earlier mixed-cell expression-coupled artifact.
2. Run `scripts/predict_rank.py --benchmark results/rank_benchmark_cancer_only --smiles ...` in a compatible environment.
3. Show protein-target votes separately from cell-response ranks. Neither is a calibrated probability or proof of cell killing.
4. Read `results/step3_evidence/literature_cards.json` for the source-backed context and `drug_chains.json` for separate measured/predicted/annotation fields. `examples/` contains a validation imatinib example and a test-fold example.
5. An empty Ridge field for a train/validation example means it was not in the saved test-prediction export. Do not fill it with observed measurements or imply it was a held-out test prediction.
6. Local-LLM text may summarize supplied evidence, but must preserve the context, uncertainty, and evidence tags. A shared association is not a causal MoA finding.

### Modeling / scientific presentation

- Keep the original scaffold split and source hashes. No row-level random split or test-driven model switch.
- Present the final comparison, including simple baselines and unsuccessful feature/multilayer additions.
- Use Step 3's observed-response associations, not a correlation between two predicted outputs.
- Do not imply clinical applicability without receptor/genotype/treatment-context data. Do not call the statin/glucocorticoid hypotheses novel literature discoveries.

## Sharing formats

### GitHub-ready folder

The code/report version excludes datasets, fitted binaries, reference matrices, and target caches. It contains scripts, configurations, requirements, full experiment reports, Step 3 outputs, documented examples, and limitations. Check contributor ownership and source-data redistribution terms before making anything public; no license is assumed.

### Private partner bundle

The separately assembled private bundle additionally includes:

- Final `results/rank_benchmark_cancer_only/best_model.npz`.
- Its six-file cleaned input snapshot, prepared mappings/responses, and saved held-out predictions.
- The computed target-query cache and matching configuration from `results/target_expression_filtered_test/`.
- Byte-preserved original metadata under `provenance/original_reports/`, because public-facing reports normalize absolute input paths.

**Do not upload the private archive to a public GitHub repository.** It contains data-derived model state and datasets. Its `.gitignore` protects standard artifact locations, but an archive bypasses those rules.

## What is not bundled

The full ~499K-compound **runtime target-reference library is still on ARCC**, not on this Mac. The cached target predictions cover the 954 benchmark compounds only. They do not support arbitrary new SMILES queries offline.

The original full `data/features/` triplet is also still on ARCC; the canonical prepared split in this package does not replace those original features for a full training/reference rebuild. The final Ridge predictor can run from its saved model without retraining.

To obtain those remaining assets, an authorized modeling lead can locate them on ARCC:

```bash
cd ~/projects/eMerge
cat results/chembl-library-workdir.txt
ls data/features/{compound_split.csv,fingerprints.npz,feature_report.json}
```

Append `/library` to the printed work directory. Minimum target-scoring runtime files are `manifest.json`, `fingerprints.npy`, and `labels.csv.gz`; obtain `compounds.csv.gz`, `evidence.csv.gz`, and `evaluation_queries.csv` too for lookup, audit, or evaluation. Use the unmodified manifest with the matching metrics configuration. Keep the 45-gene oncology-feature filter distinct from the original reference's safety evidence.

## Reproduce Step 3 from the private bundle

Use Python 3.11 and the recorded package versions. No GPU or new target-library build is needed:

```bash
.venv/bin/python scripts/step3_evidence.py \
  --baseline results/rank_benchmark_cancer_only \
  --targets results/target_expression_filtered_test \
  --drugs results/rank_benchmark_cancer_only/input/drugs.csv.gz \
  --output results/step3_reproduction
```

It should reproduce the association statistics. Hashes of path-normalized metadata files can differ from original byte-preserved reports; the original numerical data/cache and original metadata are retained separately for audit. The computation uses observed test ranks, not the Ridge prediction values.

## Final scope

Ready: data/model experiments, final Ridge artifact, target retrieval results, source-backed evidence cards, observational Step 3 analysis, example chains, and reproducible scripts.

Still integration work: live API/UI, arbitrary-query target-library serving, local-LLM presentation, and any future curated action/genotype-aware mechanism model. No molecule generation, prospective biological validation, clinical indication prediction, or demonstrated causal MoA is claimed.
