# Step 3 — literature, observed response, and qualified drug chains

## Completed result

This analysis was run locally on the downloaded ARCC outputs; it did **not** retrain a model or spend another cluster allocation. It uses the cancer-only split, the previously predicted target cache, and observed held-out response ranks. Configuration: `config/step3_links.json`. Script: `scripts/step3_evidence.py`. Outputs: `results/step3_evidence/`.

Six externally supported clinical contexts and two exploratory repurposing hypotheses were fixed before computing these associations. The compounds were already inspected in earlier experiments, so this is **exploratory analysis**, not preregistered or independent validation.

| Link | Dataset result | Key qualification |
|---|---|---|
| BCR::ABL1 → CML | Not statistically testable on the shared panel | Two externally checked CML anchors exist, but neither is in the dense screening panel |
| MEK → melanoma | Insufficient target-vote coverage | Only one positive-vote scaffold; BRAF mutation status is missing |
| EGFR → head-and-neck squamous cancer | rho 0.244; p 0.0166; BH q 0.0664 | Does not survive eight-link correction; clinical antibody evidence is not equivalent to our small-molecule panel |
| ERBB2 → breast cancer | rho 0.337; 95% scaffold-bootstrap interval 0.185–0.467; p 0.000700; BH q 0.00560 | Exploratory **anatomical-proxy association**, only seven positive-vote scaffolds; HER2 status is missing |
| ESR1 → breast cancer | rho 0.035; p 0.736; BH q 1.0 | No association detected; ER status and agonist/antagonist distinctions are missing |
| mTOR → renal-cell carcinoma | Insufficient target-vote coverage | Three positive-vote scaffolds |
| HMGCR/statins → diffuse glioma | Insufficient target-vote coverage | Zero positive-vote scaffolds remain after the fixed primary coverage filter; this is not evidence against the hypothesis |
| NR3C1/glucocorticoids → mesothelioma | Insufficient target-vote coverage | Three positive-vote scaffolds; no clinical efficacy claim |

**Interpretation:** one target–lineage proxy association survived the specified correction. It does not prove that ERBB2 causes drug response or that these drugs treat breast cancer. The other clinical links remain external evidence, not five new discoveries. Neither repurposing hypothesis is validated by this dataset analysis.

## What is being tested

For each drug, compute a lineage-specific observed response contrast:

```text
residual(drug, cell) = observed within-drug response rank
                      − that cell's mean rank across training drugs

effect(drug, cancer group) = mean residual within that group
                            − mean residual in other observed panel cells
```

This subtracts the training-estimated tendency for certain cells to have high ranks across many drugs. It is not an estimate of absolute killing, clinical efficacy, or causal target dependence. Response direction is still awaiting provider confirmation; effect signs describe higher/lower source ranks only.

The target predictor supplies a separate exposure:

```text
target score = summed similarity-vote scores for the specified gene(s)
               / total vote scores across the query's top 20 target genes
```

These are predictions, not measured query affinities. Zero means the target was not retrieved in the top 20, not confirmed non-binding. Queries with no retrieved targets are excluded from the association test, not assigned negative biological labels. Known target/MoA annotations are displayed separately and are not used to manufacture the target predictor's input.

The primary test correlates **predicted target scores with observed responses**, not response-model scores with other response-model scores. Ridge predictions appear only as a separate descriptive field in the chain export.

## Fixed coverage and statistics

- Select cells measured for at least 80% of the **670 training compounds**: **440 cells**.
- Require a test drug to have observations for at least 80% of that panel: **136/144 test compounds** pass this initial filter.
- Require at least five panel cells in the cancer group; per drug require at least three observed group cells and ten other cells.
- Exclude entirely unscorable target queries. For the non-CML groups, **117 drugs across 96 scaffolds** enter the coverage accounting.
- Average target scores and response contrasts within scaffold so chemical aliases/related scaffold members do not become independent statistical replicates.
- Require at least 20 scaffolds, including five with positive target votes and five with zero votes. No p-value is forced for sparse links.
- Use two-sided Spearman correlation, 10,000 scaffold-unit label permutations, and 2,000 paired scaffold-bootstrap samples, seed 2026 with a fixed link-order offset.
- Apply Benjamini–Hochberg over **all eight prespecified links**, including untested links as p=1 in the correction. Display q only for tested links.

Permutation assumes exchangeability of scaffold-level scores under the null. The panel restriction and cell-mean subtraction reduce some biases but do not remove confounding, assay-panel selection, promiscuity, cell-line correlations, or missing biomarker context. Intervals condition on the observed fixed cell panel; they are not patient-level uncertainty or per-query confidence.

## Literature cards: external evidence, not our discoveries

1. **BCR::ABL1-positive CML and imatinib:** supported externally, with the fusion-positive context retained. Do not equate every myeloid cell line or high ABL1 expression with CML. [NCI](https://www.cancer.gov/research/progress/discovery/gleevec).
2. **MEK inhibition in BRAF V600-selected melanoma:** clinical evidence includes defined mutation and combination-treatment contexts. Do not generalize this to all melanoma, colorectal, or pancreatic cancer. [NCI](https://www.cancer.gov/types/skin/research/dabrafenib-trametinib).
3. **EGFR targeting in head-and-neck squamous-cell cancer:** cetuximab is a clinical anchor, but an EGFR antibody is not interchangeable with small-molecule kinase inhibitors. [NCI](https://www.cancer.gov/about-cancer/treatment/drugs/cetuximab).
4. **HER2 targeting in HER2-positive breast cancer:** receptor status matters; the table's broad breast label is not a HER2 assay. [NCI](https://www.cancer.gov/types/breast/research/altto-qa).
5. **Estrogen-receptor blockade/degradation in receptor-positive breast cancer:** binding, agonism, antagonism, and receptor degradation are different. Our target votes do not infer the action verb. [NCI](https://www.cancer.gov/types/breast/treatment/hormone-therapy).
6. **mTOR inhibition in renal-cell carcinoma:** everolimus is a clinical anchor with treatment-context requirements, and its FKBP12-complex mechanism is not fully described by a generic MTOR vote. [NCI](https://www.cancer.gov/about-cancer/treatment/drugs/everolimus).

The machine-readable cards preserve context, sources, and evidence tiers. We mapped only two CML anchors from external Cellosaurus records: [K-562](https://www.cellosaurus.org/CVCL_0004) and [KU812](https://www.cellosaurus.org/CVCL_0379). This is intentionally not an exhaustive CML panel.

## The two hypotheses

### Statins → glioma

This is **not a new literature discovery**. Preclinical simvastatin experiments exist, while a secondary analysis of glioblastoma trials did not find a survival association for statin co-medication. Preserve both directions of evidence. [Preclinical study](https://pmc.ncbi.nlm.nih.gov/articles/PMC3711082/), [clinical secondary analysis](https://pubmed.ncbi.nlm.nih.gov/29423540/).

The held-out pitavastatin example has descriptive glioma response observations, but it fails the primary shared-panel coverage requirement. One drug must not rescue or validate a class-wide hypothesis after looking at results. It is shown separately, not added back into the significance test.

### Glucocorticoids → mesothelioma

Use **glucocorticoids**, not all steroids. A published 1995 experiment included a dexamethasone-responsive mesothelioma line, so this is also not a novel hypothesis. That limited cell-line finding is not established clinical efficacy, and NR3C1 association does not by itself prove receptor agonism. [Original study](https://www.sciencedirect.com/science/article/pii/0959804995004319).

## Per-drug chains

`drug_chains.json` contains **47 compounds**: test compounds with relevant predicted targets, plus explicitly tagged literature exemplars from other partitions. Every entry separates:

- `external_evidence`: literature card and context requirements.
- `predicted_association`: target votes and, where available, final Ridge response predictions.
- `measured_response_derived_summary`: actual observed response-rank summary, counts, and training-baseline-adjusted contrast.
- `source_annotation_not_model_input`: known target/MoA metadata, not a new experimental measurement or model input.
- `qualified_hypothesis_not_causal`: the overall proposed chain, never a claim of demonstrated causal action.

Supporting pChEMBL in target retrieval belongs to **reference neighbours**, not necessarily the query compound. An external exemplar with no matching predicted target is marked as such rather than inventing a model-supported chain. Full-panel example summaries are descriptive and are not additional significance tests.

**Imatinib:** it is in the **validation** fold, not the test fold. Its ABL1 prediction and two externally mapped CML anchors can be shown beside the known literature mechanism. The two cells' mean observed within-drug rank is approximately **0.994**, but that is not 99.4% killing or a held-out test success rate. The Ridge test-prediction table does not contain this validation molecule, so the export leaves its model-response field empty rather than substituting observations as predictions.

## Run and outputs

From the original project root, using the downloaded handoff inputs:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 scripts/step3_evidence.py
```

The script verifies source/split consistency, refuses duplicate query IDs and mismatched observation tables, and refuses to overwrite a completed report. In a standalone partner package, pass paths explicitly:

```bash
.venv/bin/python scripts/step3_evidence.py \
  --baseline results/rank_benchmark_cancer_only \
  --targets results/target_expression_filtered_test \
  --drugs results/rank_benchmark_cancer_only/input/drugs.csv.gz \
  --output results/step3_reproduction
```

Files:

- `report.json`: all eight results, coverage, protocol, limitations, and input hashes.
- `association_results.csv`: compact table, including null/underpowered cases.
- `literature_cards.json`: six clinical-context cards and two hypothesis cards.
- `drug_chains.json`: separate measured/predicted/external evidence for 47 compounds.
- `per_drug_association_inputs.csv`: audit table of the target-score/observed-response contrasts; not raw response measurements.

Model deployment is unchanged: final cancer-only **Ridge**, plus separate ChEMBL target retrieval. A significant observational link does not override the model-selection protocol or establish a treatment indication.
