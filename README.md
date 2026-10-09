# UAV-EHFS: task-aware elbow feature selection for UAV intrusion detection

Code and results for *"UAV-EHFS: A task-aware elbow
feature-selection framework for multi-topology drone intrusion detection"*
(S. Alsaraireh, M. Al-Fayoumi), PLOS ONE manuscript **PONE-D-26-48338**.

This repository holds the code and outputs that correspond to the submitted
manuscript (tag `v2.0.0-manuscript`). Every number in the manuscript is produced
by one script (`uav_ehfs.py`), stored as JSON in `results/`, and turned into the
manuscript's LaTeX tables, in-text numbers and data figures by `make_tables.py`.
No reported number is typed by hand. The manuscript itself is not included.

## Layout

| Path | Content |
|---|---|
| `uav_ehfs.py` | The single pipeline: preprocessing, duplicate removal, 80/20 split, 5-fold CV sweep over K, one-SE stopping rule, WCSS baseline, ablation, sealed-test evaluation |
| `make_tables.py` | Builds every results table of the manuscript (`paper/generated/tab_*.tex`), every number quoted in its text (`paper/generated/macros.tex`) and Fig 4-7 (`paper/figures/`) from `results/*.json` |
| `diagnostics.py` | Duplicate structure of S3 used in the sensitivity paragraph -> `results/diagnostics.json` |
| `results/<S>_<task>.json` | All metrics, confusion matrices, per-fold CV curves, ablation (`ablation_cv`), selected features, preprocessing log, environment |
| `results/<S>_<task>_test_predictions.csv.gz` | Test-set row id, true label and predicted label for every configuration (task-aware elbow, all features, WCSS knee) |
| `results/with_duplicates/` | Sensitivity analysis without duplicate removal |
| `logs/` | Console logs of the two runs |
| `paper/generated/` | Output of `make_tables.py`: the tables and in-text numbers exactly as they appear in the manuscript |
| `paper/figures/` | Output of `make_tables.py`: Fig 4-7 of the manuscript (Fig 1-3 are schematic diagrams, not data) |
| `review/` | Added after submission in response to the reviewer (tag `v2.0.1-review`): independent re-run and SMOTE-placement analysis. See `review/README.md` |

`S` is `S1` (compromised UAV), `S2` (rogue access point) or `S3` (compromised
ground control station); `task` is `binary` or `multiclass`.

## Environment

Python 3.11 with the exact versions in `requirements.txt`:
numpy 2.4.6, pandas 3.0.6, scikit-learn 1.9.1, imbalanced-learn 0.14.2,
kneed 0.8.6, joblib 1.6.0, matplotlib 3.11.2. The interpreter, library versions,
platform and finish time of each run are recorded under `environment` in every
result file.

## Reproduce

```bash
pip install -r requirements.txt
python uav_ehfs.py all                   # 6 runs -> results/
python uav_ehfs.py all --keep-duplicates # sensitivity -> results/with_duplicates/
python diagnostics.py                    # -> results/diagnostics.json
python make_tables.py                    # paper/generated/*.tex and paper/figures/Fig4-Fig7
```

Comparing the regenerated `paper/generated/` with the committed copy (e.g.
`git diff paper/generated`) shows directly that the tables follow from the
result files.

A single case can be run with, e.g., `python uav_ehfs.py S3 multiclass`.
The environment variable `UAV_EHFS_JOBS` sets how many CV folds run in parallel
(default 4); it does not change the results. Timing figures (training time,
inference latency) depend on the hardware.

## Partitions and folds

All randomness is seeded with 42, so partitions and folds are reproduced exactly
by the script:

1. Preprocessing and exact-duplicate removal (`preprocess`); rows keep their order.
2. Stratified 80/20 split: `train_test_split(test_size=0.2, stratify=y, random_state=42)`.
   The `row_id` column of each predictions file gives the index of every test row
   in the de-duplicated data.
3. Selection sample from the training partition (`selection_sample`, cap 60,000
   rows, classes kept whole up to 50 rows, `RandomState(42)`).
4. Folds: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on the
   selection sample.

SMOTE, K-means (k-means++, 10 restarts) and the decision tree also use seed 42.
Split sizes and per-class counts are stored under `split` and `selection` in each
result file.

## Data

UAV-NIDD: Hadi HJ, Cao Y, Alshara MA, *UAV-NIDD: a dynamic dataset for
cybersecurity and intrusion detection in UAV networks*, Zenodo 2025,
https://zenodo.org/records/15125851. Download the three CSV zip archives and
place them in `dataset/` under these names:

| Archive | MD5 of archive | CSV inside | MD5 of CSV |
|---|---|---|---|
| `UAV-Case1-Label (1).zip` | 78ec5c744001bfead0a007f82bf1de7c | `UAV-Case1-Label (1).csv` | 0e21d927c70382261c38e21512d23a78 |
| `Access Point Case2 Label.zip` | 2f574c933ef6a0b9c20b6dd4beef6734 | `Access Point Case2 Label.csv` | 58581e4204fa04b5c4c4fc9d22bf6c2f |
| `GSC Case3 Label .zip` | f17f542f7aeb4fbda60ca1f3799d67d8 | `GSC Case3 Label .csv` | 047640a9fcb139ce271096cb20511501 |

Two properties of the released files are handled explicitly and reported in the
paper. First, exact duplicate rows are removed before splitting (98% of S3 rows
are duplicates once identifiers are dropped). Second, some columns are
misaligned with their headers and contain IP addresses; such columns are
dropped by a content rule (P1).

## Provenance of the reported results

The files in `results/` were produced on 2026-09-25 (15:31-15:43 UTC, Linux,
Python 3.11.15) by `uav_ehfs.py` exactly as it appears in this repository. The
`git_commit` field recorded in those files (`d4c2289`) is the repository head at
the time of the run; the script had not yet been committed and was committed
unchanged immediately afterwards (`b7c554d` in the authors' development
repository). The same script is archived on Zenodo
(MD5 `5a8ec5826ba19577e534166a610e1bf7`, LF line endings).

## License

Code: MIT (see `LICENSE`). Generated results and figures: CC BY 4.0. The
UAV-NIDD dataset keeps the license set by its authors on Zenodo.
