# Analyses added in revision 1 (PONE-D-26-48338)

Every computation of the revision is listed, in the order it was run, in
`run_revision.sh`:

```bash
bash review/run_revision.sh all     # or: main | sensitivity | stability | labels | review
```

The main analysis (`uav_ehfs.py`, results in `../results/`) uses the cleaned
feature set (identifier and clock columns removed by content, S3 headers
realigned), the ANOVA F statistic in Stage 2 and median imputation. The scripts
below reuse `uav_ehfs.prepare()`, so they see exactly the partitions, selection
sample, folds and seeds of the main analysis.

| Script | Output | Reviewer comment | Paper |
|---|---|---|---|
| `feature_audit.py` | `feature_audit/` (status of every column, S3 realignment check) | R1.7, R2.5 | Feature audit, Table 4, S4 and S5 Tables |
| `duplicate_origin.py` | `duplicate_origin/` | R1.6 | Preprocessing, S6 Table |
| `smote_placement.py` | `smote_placement/` (E1/E2 placements) | R2.4 | Sensitivity to the placement of SMOTE, S3 Table |
| `baselines.py` | `baselines/` (other detectors, other selectors, size-constrained trees) | R1.2, R1.9, R1.11, R2.3 | Tables 14, 15, 20; S9 and S10 Tables |
| `evaluation_checks.py` | `evaluation_checks/` (Wilson intervals, test-row overlap, capture-order split, S3 confusions, SMOTE on S1) | R1.3, R1.4, R1.12, R2.6 | Tables 11-13, 17, 22 |
| `deployment_cost.py` | `deployment_cost/summary.json` (single-core selection cost, inference cost) | R1.5, R1.10 | Table 21 |
| `tree_features_used.py` | `deployment_cost/features_used.json` | R1.10 | Table 20 (last column) |
| `reproduce.py` | `reproduction/`, `reproduction_check.json` | — | Re-run of the earlier configuration on a second machine |

Sensitivity analyses (`../results/sensitivity/`), resampling runs with seeds 1-10
(`../results/stability/`) and the Pearson Stage 2 score under random class codings
(`../results/label_coding/`) are produced by `uav_ehfs.py` options, as listed in
`run_revision.sh`. `../results/round1/` holds the outputs of the configuration in
which identifiers are removed only by name and IP-address content (clock columns
retained), with the Pearson score in Stage 2 and mean imputation.

All tables, in-text numbers and Fig 4-7 are generated from these files by
`python make_tables.py`.
