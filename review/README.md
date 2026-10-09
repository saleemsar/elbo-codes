# Analyses added in response to the reviewer (PONE-D-26-48338)

Everything in this folder was added after the manuscript was submitted (tag
`v2.0.1-review`). Nothing outside `review/` was changed: the code and results
that correspond to the manuscript are those of tag `v2.0.0-manuscript`.

Both scripts import `uav_ehfs.py` unchanged, so data preparation, the
train/test split, the selection sample, the folds and all seeds are those of the
manuscript. They were run on Windows with Python 3.14 and the library versions
in `requirements.txt` (the reported results were produced on Linux with
Python 3.11.15).

## 1. Independent re-run (`reproduce.py`)

```bash
python review/reproduce.py            # re-runs uav_ehfs.py -> review/reproduction/, then compares
python review/reproduce.py --compare  # comparison only
```

Writes `review/reproduction/` (same format as `results/`) and
`review/reproduction_check.json`, which compares each scenario and task with
`results/`: K\*, selected features, WCSS knee, per-fold CV curves, ablation, test
metrics, confusion matrices and test-set predictions, with the number of test
rows whose prediction differs for each configuration.

Result: in all six cases, K\*, the selected features and the WCSS knee are
identical and the ablation results are identical. Task-aware elbow test-set
predictions are identical in five cases and differ in 3 of 169,116 rows in the
sixth (S1 multi-class). No reported test metric changes by more than 0.00013.
These differences also appear for the all-features detector and come from
platform floating-point and tie-breaking differences.

## 2. SMOTE placement (`smote_placement.py`)

In `uav_ehfs.py`, SMOTE is applied in the full feature space in the CV sweep
(`run_fold`) and in the selected-feature subspace for the final model
(`fit_final`). This script makes the two stages consistent in each direction:

- **E1**: full-space SMOTE at both stages (K\* and F\* unchanged; final detector
  trained on the full-space SMOTE-balanced training partition restricted to F\*).
- **E2**: subset-space SMOTE at both stages (for every K the CV detector is
  trained on SMOTE applied after restricting the fold to F_K; the one-SE rule
  gives K\*_E2; final model as reported).

```bash
python review/smote_placement.py all      # or: python review/smote_placement.py S3 multiclass
```

Outputs per scenario and task in `review/smote_placement/`: K\*, subsets, CV
curves (per fold for E2), test metrics and confusion matrices for the reported
configuration (recomputed on the same machine), E1 and E2, and test-set
predictions for all of them. `summary.csv` is Table R1 of the response letter.

Result: in five of six cases E2 selects the same K\* and features, and test
weighted and macro F1 differ by at most 0.0001 between the placements. In S3
multi-class, where the CV curve is nearly flat, E2 selects K\* = 19 (the 11
reported features plus 8); test weighted F1 is unchanged (0.8968 / 0.8970 / 0.8969
for reported / E1 / E2) and macro F1 is 0.6535 / 0.6775 / 0.6697. No consistent
placement lowers any reported test metric by more than 0.0002.

`smote_placement.log` holds the console output of the runs (S1 binary was run
first, separately, and its line is not in the log; its output file is complete).
