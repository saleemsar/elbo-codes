"""Re-run the unchanged uav_ehfs.py and compare against the reported results/.

Writes fresh outputs to review/reproduction/ (results/ is never touched), then
checks, for every scenario and task, that K*, the selected subset, the WCSS knee,
the per-fold CV curves, every test metric and every test-set prediction match
the files the manuscript was built from.

Usage:  python review/reproduce.py            (run + compare)
        python review/reproduce.py --compare  (compare only)
"""
import gzip
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import uav_ehfs as u  # noqa: E402

REF = os.path.join(ROOT, "results")
NEW = os.path.join(HERE, "reproduction")
JOBS = [(s, t) for s in ["S1", "S2", "S3"] for t in ["binary", "multiclass"]]


def max_abs_diff(a, b):
    """Largest absolute difference between two nested JSON structures of numbers."""
    if isinstance(a, dict):
        return max([max_abs_diff(a[k], b[k]) for k in a] or [0.0])
    if isinstance(a, list):
        return max([max_abs_diff(x, y) for x, y in zip(a, b)] or [0.0])
    if isinstance(a, (int, float)) and not isinstance(a, bool):
        return abs(float(a) - float(b))
    return 0.0 if a == b else float("inf")


def compare(scn, task):
    r = json.load(open(os.path.join(REF, f"{scn}_{task}.json")))
    n = json.load(open(os.path.join(NEW, f"{scn}_{task}.json")))
    rs, ns = r["selection"], n["selection"]
    feats = lambda s: [f["feature"] for f in s["selected_features"]]
    pr = pd.read_csv(gzip.open(os.path.join(REF, f"{scn}_{task}_test_predictions.csv.gz")))
    pn = pd.read_csv(gzip.open(os.path.join(NEW, f"{scn}_{task}_test_predictions.csv.gz")))
    out = {
        "K_star_equal": rs["K_star"] == ns["K_star"],
        "K_wcss_equal": rs["K_wcss_kneedle"] == ns["K_wcss_kneedle"],
        "selected_features_equal": feats(rs) == feats(ns),
        "cv_per_fold_max_abs_diff": max_abs_diff(rs["cv_f1_per_fold"], ns["cv_f1_per_fold"]),
        "ablation_max_abs_diff": max_abs_diff(r["ablation_cv"], n["ablation_cv"]),
        "test_metrics_max_abs_diff": max_abs_diff(r["test"], n["test"]),
        "confusion_matrices_equal": all(
            r["test"][c]["confusion_matrix"] == n["test"][c]["confusion_matrix"] for c in r["test"]),
        "test_predictions_identical": pr.equals(pn),
        "n_test_rows": int(len(pr)),
        "test_rows_with_different_prediction": {
            c[len("y_pred_"):]: int((pr[c] != pn[c]).sum()) for c in pr.columns if c.startswith("y_pred_")},
        "metrics_paper_vs_rerun": {
            c: {k: [r["test"][c][k], n["test"][c][k]] if k == "accuracy"
                else [r["test"][c][k]["f1"], n["test"][c][k]["f1"]]
                for k in ["accuracy", "weighted", "macro"]} for c in r["test"]},
        "rerun_environment": n["environment"],
    }
    out["identical"] = (out["K_star_equal"] and out["K_wcss_equal"] and out["selected_features_equal"]
                        and out["confusion_matrices_equal"] and out["test_predictions_identical"]
                        and out["cv_per_fold_max_abs_diff"] < 1e-12
                        and out["test_metrics_max_abs_diff"] < 1e-12)
    return out


if __name__ == "__main__":
    if "--compare" not in sys.argv:
        u.OUT_DIR = NEW
        for s, t in JOBS:
            if not os.path.exists(os.path.join(NEW, f"{s}_{t}_test_predictions.csv.gz")):
                u.run(s, t)
    report = {f"{s}_{t}": compare(s, t) for s, t in JOBS}
    report["all_identical"] = all(v["identical"] for v in report.values())
    with open(os.path.join(HERE, "reproduction_check.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    for k, v in report.items():
        if isinstance(v, bool):
            print(k, v)
        else:
            print(k, "IDENTICAL" if v["identical"] else
                  f"K* equal={v['K_star_equal']} features equal={v['selected_features_equal']} "
                  f"WCSS equal={v['K_wcss_equal']} max metric diff={v['test_metrics_max_abs_diff']:.1e} "
                  f"rows differing={v['test_rows_with_different_prediction']}")
