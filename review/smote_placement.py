"""Sensitivity of the reported results to where SMOTE is applied (reviewer query).

In uav_ehfs.py SMOTE is applied at two points:
  * CV sweep (run_fold): on the training fold in the FULL feature space; the
    oversampled fold feeds Stages 1-2 and the detector is then trained on the
    columns F_K of that oversampled fold.
  * Final model (fit_final): AFTER restricting the training partition to F*,
    i.e. nearest neighbours are found in the SUBSET space.
Both stages use training data only. This script measures what the difference does,
by making the pipeline consistent in each of the two directions:

  E1  full-space SMOTE at both stages.  CV unchanged (K*, F* as reported); the final
      detector is trained on the full-space SMOTE training partition, restricted to F.
  E2  subset-space SMOTE at both stages.  Stages 1-2 still use the full-space
      balanced fold (they need all columns), but for every K the detector is trained
      on SMOTE applied to the fold restricted to F_K.  The one-SE rule then gives
      K*_E2; F*_E2 is built exactly as in the main pipeline; the final model is
      fit_final (subset-space SMOTE, as reported).

Data preparation, split, selection sample, folds, seeds and all fitted components
are those of uav_ehfs.py (imported, not re-implemented). As a built-in check, the
original CV curve is recomputed on the same folds and compared with results/.

Usage:  python review/smote_placement.py all      (or: S3 multiclass)
Output: review/smote_placement/<S>_<task>.json (+ test predictions) and summary.json
"""
import argparse
import gzip
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import uav_ehfs as u  # noqa: E402

OUT = os.path.join(HERE, "smote_placement")
REF = os.path.join(ROOT, "results")


def prepare(scn, task):
    """Identical to the first part of uav_ehfs.run()."""
    df, label_col = u.load_raw(scn)
    Xdf, y_str, _ = u.preprocess(df, label_col, task)
    del df
    class_names = ["Normal", "Attack"] if task == "binary" else sorted(y_str.unique())
    code = {c: i for i, c in enumerate(class_names)}
    y = y_str.map(code).to_numpy()
    X = Xdf.to_numpy(dtype=np.float32)
    feat_names = list(Xdf.columns)
    row_ids = np.arange(len(y))
    Xtr, Xte, ytr, yte, idtr, idte = u.train_test_split(
        X, y, row_ids, test_size=u.TEST_SIZE, stratify=y, random_state=u.SEED)
    keep = [j for j in range(X.shape[1])
            if np.nanmax(Xtr[:, j]) != np.nanmin(Xtr[:, j]) and not np.all(np.isnan(Xtr[:, j]))]
    Xtr, Xte = Xtr[:, keep], Xte[:, keep]
    feat_names = [feat_names[j] for j in keep]
    sel = u.selection_sample(Xtr, ytr)
    Xs, ys = Xtr[sel], ytr[sel]
    folds = list(u.StratifiedKFold(n_splits=u.N_FOLDS, shuffle=True, random_state=u.SEED).split(Xs, ys))
    return dict(class_names=class_names, feat_names=feat_names, Xtr=Xtr, Xte=Xte, ytr=ytr,
                yte=yte, idte=idte, Xs=Xs, ys=ys, folds=folds)


def fold_curves(fold, Xs, ys, tr, va, Ks):
    """One CV fold: original curve (as run_fold) and E2 curve (subset-space SMOTE)."""
    A, B, _ = u.fit_transform_fold(Xs[tr], Xs[va])
    ytr, yva = ys[tr], ys[va]
    As, ytrs = u.safe_smote(A, ytr)
    R, rho = u.correlations(As, ytrs)
    orig, e2 = {}, {}
    for K in Ks:
        lab, _ = u.kmeans_labels(R, K)
        F = u.representatives(lab, rho, K)
        orig[K] = u.wf1(yva, u.detector().fit(As[:, F], ytrs).predict(B[:, F]))
        Asub, ysub = u.safe_smote(A[:, F], ytr)
        e2[K] = u.wf1(yva, u.detector().fit(Asub, ysub).predict(B[:, F]))
    return orig, e2


def final_subset(Xs, ys, K):
    """Stages 1-2 on the whole selection sample at K (as in uav_ehfs.run)."""
    A, _, _ = u.fit_transform_fold(Xs)
    As, yss = u.safe_smote(A, ys)
    R, rho = u.correlations(As, yss)
    return u.representatives(u.kmeans_labels(R, K)[0], rho, K)


def fit_final_fullspace(Xtr, ytr, Xte, F):
    """E1: SMOTE on the full-space training partition, then restrict to F."""
    A, B, _ = u.fit_transform_fold(Xtr, Xte)
    As, ys = u.safe_smote(A, ytr)
    h = u.detector().fit(As[:, F], ys)
    return h.predict(B[:, F]), int(len(ys))


def curve_stats(per_fold):
    m = per_fold.mean(0)
    se = per_fold.std(0, ddof=1) / np.sqrt(per_fold.shape[0])
    return m, se


def headline(m):
    return {"accuracy": m["accuracy"], "weighted_f1": m["weighted"]["f1"],
            "macro_f1": m["macro"]["f1"], "balanced_accuracy": m.get("balanced_accuracy")}


def run(scn, task):
    t0 = time.time()
    ref = json.load(open(os.path.join(REF, f"{scn}_{task}.json")))
    d = prepare(scn, task)
    names, n = d["feat_names"], len(d["feat_names"])
    assert names == ref["feature_names"], "feature space differs from results/"
    Ks = list(range(2, n + 1))
    outs = Parallel(n_jobs=u.N_JOBS)(
        delayed(fold_curves)(i, d["Xs"], d["ys"], tr, va, Ks) for i, (tr, va) in enumerate(d["folds"]))
    P_orig = np.array([[o[0][K] for K in Ks] for o in outs])
    P_e2 = np.array([[o[1][K] for K in Ks] for o in outs])

    # Built-in reproduction check of the reported CV curve.
    cv_diff = float(np.abs(P_orig - np.array(ref["selection"]["cv_f1_per_fold"])).max())
    m0, se0 = curve_stats(P_orig)
    K0, _, _ = u.one_se_rule(Ks, m0, se0)
    assert K0 == ref["selection"]["K_star"], "reported K* not reproduced"

    m2, se2 = curve_stats(P_e2)
    K2, K2_hat, thr2 = u.one_se_rule(Ks, m2, se2)

    F0 = final_subset(d["Xs"], d["ys"], K0)
    assert sorted(names[f] for f in F0) == sorted(
        f["feature"] for f in ref["selection"]["selected_features"]), "reported F* not reproduced"
    F2 = F0 if K2 == K0 else final_subset(d["Xs"], d["ys"], K2)
    Fw = [names.index(f) for f in ref["selection"]["wcss_features"]] if ref["selection"]["wcss_features"] else None

    cn, yte = d["class_names"], d["yte"]
    pos = "Attack" if task == "binary" else None
    preds = pd.DataFrame({"row_id": d["idte"], "y_true": [cn[c] for c in yte]})
    test = {}

    # Reported configurations (subset-space SMOTE at the final stage), recomputed.
    for tag, F in [("task_aware", F0), ("wcss_kneedle", Fw)]:
        if F is None:
            continue
        p, _ = u.fit_final(d["Xtr"], d["ytr"], d["Xte"], F)
        test[f"reported_{tag}"] = u.metrics(yte, p, cn, pos)
        preds[f"y_pred_reported_{tag}"] = [cn[c] for c in p]
    # E1: full-space SMOTE at the final stage, same subsets.
    for tag, F in [("task_aware", F0), ("wcss_kneedle", Fw), ("all_features", list(range(n)))]:
        if F is None:
            continue
        p, rows = fit_final_fullspace(d["Xtr"], d["ytr"], d["Xte"], F)
        test[f"E1_{tag}"] = u.metrics(yte, p, cn, pos)
        preds[f"y_pred_E1_{tag}"] = [cn[c] for c in p]
    # E2: subset-space SMOTE in CV -> K*_E2 -> F*_E2 -> final model as reported.
    p, _ = u.fit_final(d["Xtr"], d["ytr"], d["Xte"], F2)
    test["E2_task_aware"] = u.metrics(yte, p, cn, pos)
    preds["y_pred_E2_task_aware"] = [cn[c] for c in p]

    # Check that the recomputed reported configuration equals results/.
    rep_match = all(test[f"reported_{t}"]["confusion_matrix"] == ref["test"][t]["confusion_matrix"]
                    for t in ["task_aware", "wcss_kneedle"] if f"reported_{t}" in test)
    n_changed_E1 = int((preds["y_pred_E1_task_aware"] != preds["y_pred_reported_task_aware"]).sum())

    res = {
        "scenario": scn, "task": task, "n_features": n, "class_names": cn,
        "checks": {"cv_curve_max_abs_diff_vs_results": cv_diff, "K_star_reproduced": True,
                   "F_star_reproduced": True, "reported_test_confusion_matrices_reproduced": rep_match},
        "reported": {"K_star": K0, "n_selected": len(F0), "features": [names[f] for f in F0],
                     "cv_f1_mean": m0.tolist(), "cv_f1_se": se0.tolist()},
        "E1": {"description": "full-space SMOTE at final stage (consistent with CV); K*, F* unchanged",
               "test_predictions_changed_vs_reported_task_aware": n_changed_E1},
        "E2": {"description": "subset-space SMOTE in CV for every K (consistent with final stage)",
               "K_star": K2, "K_hat_argmax": K2_hat, "one_se_threshold": thr2, "n_selected": len(F2),
               "features": [names[f] for f in F2],
               "same_subset_as_reported": sorted(F2) == sorted(F0),
               "cv_f1_mean": m2.tolist(), "cv_f1_se": se2.tolist(), "cv_f1_per_fold": P_e2.tolist(),
               "cv_f1_at_reported_K_star": float(m2[Ks.index(K0)]),
               "cv_f1_at_K_star_E2": float(m2[Ks.index(K2)])},
        "K": Ks,
        "test": test,
        "headline": {k: headline(v) for k, v in test.items()},
        "runtime_s": time.time() - t0,
    }
    os.makedirs(OUT, exist_ok=True)
    stem = os.path.join(OUT, f"{scn}_{task}")
    with open(stem + ".json", "w") as fh:
        json.dump(res, fh, indent=1)
    with gzip.open(stem + "_test_predictions.csv.gz", "wt") as fh:
        preds.to_csv(fh, index=False)
    h = res["headline"]
    print(f"[{scn}/{task}] K*={K0} -> E2 K*={K2} (same subset: {res['E2']['same_subset_as_reported']}) | "
          f"mF1 reported={h['reported_task_aware']['macro_f1']:.4f} E1={h['E1_task_aware']['macro_f1']:.4f} "
          f"E2={h['E2_task_aware']['macro_f1']:.4f} | checks cv_diff={cv_diff:.1e} cm={rep_match} "
          f"{res['runtime_s']:.0f}s", flush=True)
    return res


def summary():
    rows = []
    for s in ["S1", "S2", "S3"]:
        for t in ["binary", "multiclass"]:
            p = os.path.join(OUT, f"{s}_{t}.json")
            if not os.path.exists(p):
                continue
            r = json.load(open(p))
            h = r["headline"]
            row = {"scenario": s, "task": t, "n": r["n_features"],
                   "K_star_reported": r["reported"]["K_star"], "K_star_E2": r["E2"]["K_star"],
                   "E2_same_subset": r["E2"]["same_subset_as_reported"],
                   "E1_predictions_changed": r["E1"]["test_predictions_changed_vs_reported_task_aware"],
                   "cv_curve_max_abs_diff_vs_paper": r["checks"]["cv_curve_max_abs_diff_vs_results"],
                   "paper_task_aware_wF1": json.load(open(os.path.join(REF, f"{s}_{t}.json")))["test"]["task_aware"]["weighted"]["f1"],
                   "paper_task_aware_mF1": json.load(open(os.path.join(REF, f"{s}_{t}.json")))["test"]["task_aware"]["macro"]["f1"]}
            for k in ["reported_task_aware", "E1_task_aware", "E2_task_aware",
                      "reported_wcss_kneedle", "E1_wcss_kneedle", "E1_all_features"]:
                if k in h:
                    row[f"{k}_wF1"] = h[k]["weighted_f1"]
                    row[f"{k}_mF1"] = h[k]["macro_f1"]
            rows.append(row)
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "summary.csv"), index=False)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", choices=["S1", "S2", "S3", "all"])
    ap.add_argument("task", nargs="?", choices=["binary", "multiclass"])
    a = ap.parse_args()
    jobs = ([(s, t) for s in ["S1", "S2", "S3"] for t in ["binary", "multiclass"]]
            if a.scenario == "all" else [(a.scenario, a.task)])
    for s, t in jobs:
        if not os.path.exists(os.path.join(OUT, f"{s}_{t}_test_predictions.csv.gz")):
            run(s, t)
    summary()
