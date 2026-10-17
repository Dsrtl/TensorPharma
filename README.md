# TensorPharma

Given a compound (SMILES), predict its protein targets, the cancer cell lines that respond, the cancer types those point to, and a mechanism hypothesis. Every prediction carries its evidence.

Optional second mode: given a target and/or cell line, generate and rank new candidate molecules using the same models as the scorer.

## Layout

```
data/raw/         downloaded datasets (not committed)
data/processed/   canonicalized, joined tables (not committed)
src/tensorpharma/ pipeline code
app/              demo interface
```

## Setup

```bash
mamba env create -f environment.yml
conda activate tensorpharma
```

Then place the datasets under `data/raw/` as described in [data/README.md](data/README.md).
