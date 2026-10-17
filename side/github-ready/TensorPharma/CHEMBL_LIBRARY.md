# Full ChEMBL target retrieval on ARCC

This is Q1 (SMILES → likely protein genes), separate from the six-model cell-response benchmark. No retraining of the response models is required for this job.

## Changes to the partner runbook

- Reuse the existing `.venv`, including frozen RDKit, rather than installing `rdkit-pypi` or changing dependency versions. NumPy/pandas/SciPy/RDKit are sufficient; no parquet dependency is needed.
- Use the real `drug_targets.csv.gz` filename. Keep finite pChEMBL ≥ 6 and nonempty gene labels. This threshold is a screening convention, not universal biological truth.
- Use our existing salt/charge/tautomer standardization, not just canonical SMILES. Compute 2048-bit radius-2 Morgan fingerprints once; store packed `.npy`, compressed CSV evidence, and a hashed manifest on scratch.
- Exclude every frozen LINCS identity/connectivity from references, plus validation/test scaffolds. This is stricter than holding out only the reported 702 raw overlaps; standardized matching determines actual coverage. Build reference fingerprints only for surviving active compounds.
- Obtain evaluation positives from the ChEMBL table before exclusions. Do not use free-text `known_targets` annotations as ground truth. Preserve the old scaffold partitions and deduplicate aliases.
- Tune cutoff 0.3/0.4/0.5/0.6 and optional single-measurement weight 0.26 using validation recall@5, breaking ties with hit@5. Default candidate order prefers no trust weighting. Freeze the winner before evaluating test. Do not import the all-drug `trust_weights.json`.
- Aggregate one vote per standardized compound/gene (not one per assay/endpoint). A row with ≥2 measurements allows full weight in the optional heuristic. Measurement counts are upstream pooled counts, not independent replication. Missing labels are unknown; scores are not probabilities. Source rows/endpoints are preserved separately in evidence; source assay IDs were already lost upstream.
- Report hit@5, known-positive recall@5, coverage, and abstentions. Earlier inspection of these test drugs makes this exploratory, not an untouched external evaluation. This gene-level task is not numerically interchangeable with the earlier target-ID task.

## Run

On the Mac, upload updated scripts and the corrected partner folder:

```bash
cd /path/to/TensorPharma
rsync -avP --exclude=__pycache__ --exclude=.DS_Store \
  -e "ssh -o ServerAliveInterval=15 -o ServerAliveCountMax=6" \
  scripts TensorPharma-main <user>@<cluster-host>:~/projects/eMerge/
```

On ARCC, check `echo "$SCRATCH"`. If empty, set it to the **actual existing scratch directory assigned to your account**, not an invented path. Then:

If no separate scratch directory is assigned, explicitly use project storage instead. This is **not dedicated scratch** and counts against your home/project quota. Check your quota (filesystem free space is not your personal allowance), and allow several GB of headroom:

```bash
lfs quota -u "$USER" /lustre/fs1
mkdir -p "$HOME/projects/eMerge/work"
export TP_WORK_BASE="$HOME/projects/eMerge/work"
```

`TP_WORK_BASE` takes precedence over `SCRATCH`. The scripts work with either storage location; printed scratch/artifact paths refer to the selected work location.

```bash
cd ~/projects/eMerge
bash scripts/submit_chembl_library.sh
squeue -u <user>
```

The submission script stages small inputs on scratch and submits an 8 CPU / 64 GB / 8-hour maximum build-and-evaluation job. Actual completion depends on standardization time and queue policy. No GPU, PostgreSQL server, new dependency install, or copy of the original 2 GB dump is required. Failures leave their scratch files available for diagnosis. Completed artifacts are never silently rebuilt.

After completion:

```bash
cat results/chembl_library/metrics.json
tail -n 40 results/chembl-library-*.log
cat results/chembl-library-workdir.txt
```

For external query molecules, create a CSV with `query_id,smiles` columns and pass its path:

```bash
bash scripts/submit_chembl_library.sh queries.csv
```

This submits the build/evaluation, then a dependent array of up to 20 shards (4 CPUs / 8 GB each), then a merge job. Each query must parse successfully; failures stop that shard and prevent merging. Empty predictions have an explicit abstention row. The final file is `results/chembl_library/query_predictions.csv`. The full reference/evidence remains at the printed scratch location; retain it before the cluster's scratch purge deadline.

Reference: [RDKit fingerprint API](https://www.rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html), [Slurm array dependencies](https://slurm.schedmd.com/job_array.html).
