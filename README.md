# TensorPharma

eMerge AI Hackathon 2026 · Healthcare track · Challenge 6 (Sylvester / University of Miami)

Given a compound as a SMILES string, predict its likely protein targets, the cancer cell lines that respond, the cancer types those point to, and an evidence-tagged mechanism hypothesis. Every prediction carries its supporting evidence.

**This is a research prototype. Predictions are experimental associations, not clinical advice.**

## Where things are

| Path | Contents |
|---|---|
| **[`main/github-ready/TensorPharma/`](main/github-ready/TensorPharma/)** | **The modeling work: all scripts, experiment reports, protocols and results. Start here.** |
| `scripts/` | Dataset build and export utilities for the app pipeline |
| `src/tensorpharma/` | Target-retrieval module used by the app |
| `data/processed/` | Cleaned, joined tables, committed as compressed CSV (see the caveat below) |

Start at [`main/github-ready/TensorPharma/`](main/github-ready/TensorPharma/) and read its own [README](main/github-ready/TensorPharma/README.md). That folder holds the substantive work: every script, protocol, experiment report and result.

## Headline result

The selected model is **Ridge (`ridge_1000`)**, chosen on **validation** macro within-drug Spearman (0.3074) across six model families compared on one frozen scaffold split. LightGBM scored higher on **test** (0.3252 against 0.3189) and was deliberately not selected, because the selection rule was frozen before test results were seen.

Simple baselines are competitive: the cell-mean baseline reaches 0.2960 validation Spearman. Complexity did not guarantee better performance, and the unsuccessful additions are reported alongside the successful ones.

| Document | What it covers |
|---|---|
| [RANK_BENCHMARK.md](main/github-ready/TensorPharma/RANK_BENCHMARK.md) | Six-family response comparison protocol |
| [CHEMBL_LIBRARY.md](main/github-ready/TensorPharma/CHEMBL_LIBRARY.md) | Protein-target retrieval from ChEMBL similarity |
| [STEP3.md](main/github-ready/TensorPharma/STEP3.md) | Literature contexts and observed-response associations |
| [TARGET_EXPRESSION_TEST.md](main/github-ready/TensorPharma/TARGET_EXPRESSION_TEST.md) | A feature addition that did not help; incumbent retained |
| [DATA_AND_ARTIFACTS.md](main/github-ready/TensorPharma/DATA_AND_ARTIFACTS.md) | Which artifacts exist, where, and what is excluded |
| [MODELING.md](main/github-ready/TensorPharma/MODELING.md) | Historical runbooks |

## Reproducing

A clone alone cannot retrain or serve predictions. Fitted model binaries, reference matrices and target caches are **not** in this repository; see [DATA_AND_ARTIFACTS.md](main/github-ready/TensorPharma/DATA_AND_ARTIFACTS.md) for how to obtain them.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
brew install libomp   # macOS only, needed by LightGBM
```

For the modeling environment, use `main/github-ready/TensorPharma/requirements-reproducible.txt`, which pins the versions the recorded results were produced with. Frozen experiments must use their original RDKit version; an arbitrary fresh install is not a substitute.

Cluster runbooks use `<user>@<cluster-host>` as placeholders. Substitute your own authorized account.

## Data and licensing

⚠️ `data/processed/` contains committed compressed tables totalling roughly 109 MB, derived from **ChEMBL 37, CCLE/DepMap, PharmacoDB and LINCS**. Availability here does not imply redistribution rights. Check each source's access and licensing terms before reusing or redistributing these files. No license is asserted for this repository, and contributor ownership has not been cleared.

Raw downloads (including the 83 GB LINCS level-4 `.gctx`) are not committed and are not required by the current pipeline.

## Limitations

Predictions are experimental associations, not laboratory confirmation, clinical indications or treatment recommendations. Cell lines do not capture patient biology, toxicity, resistance or treatment context. Similarity votes and response ranks are not calibrated probabilities or binding affinities. Evaluation is compound/scaffold-cold and cell-warm: unseen chemicals in known cell lines, which does not demonstrate performance on unseen cells or patients.

Full limitations are listed in [the modeling README](main/github-ready/TensorPharma/README.md#limitations).
