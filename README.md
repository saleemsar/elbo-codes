# UAV-EHFS: task-aware elbow feature selection for UAV intrusion detection

Code and outputs for *"UAV-EHFS: A task-aware elbow feature-selection framework
for multi-topology drone intrusion detection"* (S. Alsaraireh, M. Al-Fayoumi),
PLOS ONE manuscript PONE-D-26-48338.

Archived on Zenodo: https://doi.org/10.5281/zenodo.22964985 (this DOI always
resolves to the latest version). Code: https://github.com/saleemsar/elbo-codes.

Every number in the manuscript is produced by the pipeline (`uav_ehfs.py`) and the
analysis scripts in `review/`, stored as JSON in `results/` and `review/`, and turned
into LaTeX tables, in-text macros and figures by `make_tables.py` (which runs
`make_tables_r1.py`). No reported number is typed by hand. `review/run_revision.sh`
lists every command in the order it was run.

## Layout

| Path | Content |
|---|---|
| `uav_ehfs.py` | The pipeline: feature audit rules, preprocessing, duplicate removal, 80/20 split, 5-fold CV sweep over K, one-SE stopping rule, WCSS baseline, ablation, sealed-test evaluation |
| `make_tables.py`, `make_tables_r1.py` | Build `paper/generated/*.tex` (tables, `macros.tex`) and `paper/figures/Fig4-7` from the result files |
| `results/<S>_<task>.json` | Main analysis: all metrics, confusion matrices, CV curves (per fold), selected features, preprocessing log, environment |
| `results/<S>_<task>_test_predictions.csv.gz` | Test-set true and predicted labels for every configuration |
| `results/sensitivity/` | One change at a time: header-validated features, Stage 2 score, imputation, duplicates kept |
| `results/stability/seed_<s>/` | The whole pipeline with seeds 1-10 (resampling stability) |
| `results/label_coding/perm_<p>/` | Pearson Stage 2 score under random class codings |
| `results/round1/` | Identifiers removed only by name and IP-address content (clock columns retained), Pearson Stage 2, mean imputation |
| `diagnostics.py`, `results/diagnostics.json` | Duplicate structure of S3 |
| `review/` | Feature audit, duplicate origin, SMOTE placement, other detectors and selectors, constrained trees, cost, evaluation checks (see `review/README.md`) |
| `logs/` | Console logs of the runs |
| `paper/generated/`, `paper/figures/`, `paper/figure_sources/` | Generated tables and numbers, Fig 1-7, LaTeX sources of Fig 1-3 |
| `dataset/` | Place the UAV-NIDD CSV archives here (see Data below) |

## Reproduce

```bash
pip install -r requirements.txt          # Python 3.12 (versions pinned)
bash review/run_revision.sh all          # audit, main runs, sensitivity, stability,
                                         # label coding, detectors/selectors, cost, checks
python diagnostics.py                    # S3 duplicate diagnostics -> results/diagnostics.json
python make_tables.py                    # tables, macros, Fig4-Fig7
cd paper/figure_sources && for f in Fig1 Fig2 Fig3; do pdflatex $f; done
```

Seeds are fixed at 42. The environment variable `UAV_EHFS_JOBS` sets how many CV
folds run in parallel (default 4). Timing figures (training time, inference
latency) depend on the hardware; the hardware used is recorded in each JSON file.

## Method in one paragraph

For every K = 2..n, Stage 1 clusters features by their Pearson correlation
profiles (K-means), and Stage 2 keeps the feature in each cluster with the
largest one-way ANOVA F statistic with the (nominal) class label. The subset is scored by the
weighted F1 of a CART decision tree, averaged over 5 stratified folds of a
selection sample taken from the training partition. Imputation, scaling,
SMOTE, correlations, K-means and the detector are fitted on the training fold
only. The stopping rule is the one-standard-error rule:
K* = min{K : mean F1(K) >= mean F1(K_hat) - SE(K_hat)}, with K_hat = argmax mean F1.
The 20% test partition is used once, after K* is fixed.

## Data

UAV-NIDD: Hadi HJ, Cao Y, Alshara MA, *UAV-NIDD: a dynamic dataset for
cybersecurity and intrusion detection in UAV networks*, Zenodo 2025,
https://zenodo.org/records/15125851. Place the three CSV zip archives in
`dataset/` under these names: `UAV-Case1-Label (1).zip`,
`Access Point Case2 Label.zip`, `GSC Case3 Label .zip`. MD5 of the
uncompressed CSVs used here:

| File | MD5 |
|---|---|
| UAV-Case1-Label (1).csv | 0e21d927c70382261c38e21512d23a78 |
| Access Point Case2 Label.csv | 58581e4204fa04b5c4c4fc9d22bf6c2f |
| GSC Case3 Label .csv | 047640a9fcb139ce271096cb20511501 |

Three properties of the released files are handled explicitly and reported in the
paper. First, exact duplicate rows are removed before splitting (98% of S3 rows
are duplicates once identifiers are dropped). Second, the column headers are
misaligned with the values; the S3 headers are realigned by one column, and every
column's status is listed by `review/feature_audit.py` (output in
`review/feature_audit/`). Third, several S1 and S2 columns hold addresses or
capture clocks under unrelated headers; identifier and clock columns are removed
by content, whatever their header.

## License

Code: MIT (see `LICENSE`). Generated results and figures: CC BY 4.0. The
UAV-NIDD dataset keeps the license set by its authors on Zenodo.
