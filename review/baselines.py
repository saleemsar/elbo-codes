"""Other detectors, other feature selectors and size-constrained trees
(reviewer comments R1.2, R1.9, R1.11 and R2.3).

All three parts reuse uav_ehfs.prepare(), so they see exactly the partitions,
selection sample and folds of the main analysis, and read K*, F*, F_K_hat and the
WCSS subset from results/<S>_<task>.json.

detectors   The subsets (task-aware F*, K_hat subset, WCSS subset, all features) are
            used to train logistic regression, a linear SVM and a random forest in
            addition to the CART detector. Training follows fit_final: imputation and
            scaling fitted on the training partition, SMOTE in the subset space.

selectors   Each alternative selector produces a feature ranking on every CV training
            fold (after the same SMOTE balancing as the main sweep). For k = 1..n the
            top-k features train the CART detector and are scored on the validation
            fold; the one-SE rule picks k*, exactly as for K*. The final ranking is
            computed on the whole selection sample and the top-k* (own k) and top-K*
            (matched size) subsets are evaluated once on the sealed test partition.
            PCA is included with k components (it still needs every raw feature).

pruning     A grid of CART depth / leaf-size constraints is cross-validated on the
            selection sample for F* and for all features. Following Breiman et al.,
            the smallest tree whose CV weighted F1 is within one SE of the best is
            chosen and evaluated on the test partition, with its size as a pickled
            model and as a flat C array (NODE_BYTES per node).

Usage:  python review/baselines.py all [--part detectors|selectors|pruning|all]
Output: review/baselines/<S>_<task>_<part>.json
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, chi2, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import uav_ehfs as U  # noqa: E402

OUT = os.path.join(HERE, "baselines")
NODE_BYTES = 12   # feature id (2) + threshold float32 (4) + left/right child (2+2) + leaf class (2)
RANK_ROWS = 20_000  # rows used for the slower rankings (MI, L1 logistic, forest importance)


# ------------------------------------------------------------------ helpers
def load_result(scn, task):
    with open(os.path.join(U.OUT_DIR, f"{scn}_{task}.json")) as fh:
        return json.load(fh)


def subsets(res, feat_names):
    idx = {f: j for j, f in enumerate(feat_names)}
    sel = res["selection"]
    out = {"task_aware": sorted(d["index"] for d in sel["selected_features"]),
           "k_hat": sorted(idx[f] for f in sel["k_hat_features"]),
           "all_features": list(range(len(feat_names)))}
    if sel.get("wcss_features"):
        out["wcss_kneedle"] = sorted(idx[f] for f in sel["wcss_features"])
    assert sorted(feat_names[j] for j in out["task_aware"]) == sorted(
        d["feature"] for d in sel["selected_features"]), "feature index mismatch with results JSON"
    return out


def fit_eval(d, F, model, transform=None):
    """fit_final with an arbitrary model (and optional fitted transform, e.g. PCA)."""
    A, B, _ = U.fit_transform_fold(d["Xtr"][:, F], d["Xte"][:, F])
    if transform is not None:
        tr = transform.fit(A)
        A, B = tr.transform(A).astype(np.float32), tr.transform(B).astype(np.float32)
    As, ys = U.safe_smote(A, d["ytr"])
    t0 = time.perf_counter(); model.fit(As, ys); t_train = time.perf_counter() - t0
    t0 = time.perf_counter(); pred = model.predict(B); t_inf = time.perf_counter() - t0
    positive = "Attack" if d["class_names"] == ["Normal", "Attack"] else None
    m = U.metrics(d["yte"], pred, d["class_names"], positive)
    eff = {"n_features": len(F), "train_time_s": t_train,
           "inference_ms_per_sample": 1000 * t_inf / len(B),
           "model_size_kb": len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)) / 1024}
    if isinstance(model, DecisionTreeClassifier):
        eff.update(tree_depth=int(model.get_depth()), tree_leaves=int(model.get_n_leaves()),
                   tree_nodes=int(model.tree_.node_count),
                   c_array_kb=model.tree_.node_count * NODE_BYTES / 1024)
    return {"metrics": m, "efficiency": eff}


def summary(r):
    m = r["metrics"]
    return {"accuracy": m["accuracy"], "balanced_accuracy": m["balanced_accuracy"],
            "weighted_f1": m["weighted"]["f1"], "macro_f1": m["macro"]["f1"],
            "per_class": m["per_class"], **r["efficiency"]}


# ------------------------------------------------------------------ part 1: detectors
DETECTORS = {
    "cart": lambda: U.detector(),
    "logistic_regression": lambda: LogisticRegression(max_iter=1000, class_weight="balanced",
                                                      random_state=U.SEED),
    "linear_svm": lambda: LinearSVC(class_weight="balanced", dual=False, max_iter=5000,
                                    random_state=U.SEED),
    "random_forest": lambda: RandomForestClassifier(n_estimators=100, class_weight="balanced_subsample",
                                                    n_jobs=U.N_JOBS, random_state=U.SEED),
}


def part_detectors(d, res):
    S = subsets(res, d["feat_names"])
    out = {}
    for sname, F in S.items():
        for dname, mk in DETECTORS.items():
            r = fit_eval(d, F, mk())
            out[f"{sname}|{dname}"] = summary(r)
            print(f"   {sname:13s} {dname:20s} wF1={r['metrics']['weighted']['f1']:.4f} "
                  f"mF1={r['metrics']['macro']['f1']:.4f}", flush=True)
    return out


# ------------------------------------------------------------------ part 2: selectors
def rank(name, A, y):
    """Feature ranking (best first) computed on one balanced training fold."""
    n = A.shape[1]
    rng = np.random.RandomState(U.SEED)
    sub = rng.choice(len(y), size=min(RANK_ROWS, len(y)), replace=False)
    if name == "variance":
        score = A.var(0)
    elif name == "anova_f":
        with np.errstate(invalid="ignore", divide="ignore"):
            score = np.nan_to_num(f_classif(A, y)[0])
    elif name == "mutual_info":
        score = mutual_info_classif(A[sub], y[sub], random_state=U.SEED)
    elif name == "chi2":
        score = np.nan_to_num(chi2(np.clip(A, 0, None), y)[0])
    elif name == "mrmr":
        # greedy F-test relevance / mean |correlation| redundancy (FCQ variant of mRMR)
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = np.nan_to_num(f_classif(A, y)[0])
            C = np.abs(np.nan_to_num(np.corrcoef(A, rowvar=False)))
        chosen = [int(np.argmax(rel))]
        while len(chosen) < n:
            rest = [j for j in range(n) if j not in chosen]
            red = C[np.ix_(rest, chosen)].mean(1)
            q = rel[rest] / np.maximum(red, 1e-6)
            chosen.append(rest[int(np.argmax(q))])
        return chosen
    elif name == "forest_importance":
        rf = RandomForestClassifier(n_estimators=50, random_state=U.SEED, n_jobs=1)
        score = rf.fit(A[sub], y[sub]).feature_importances_
    elif name == "l1_logistic":
        lr = LogisticRegression(penalty="l1", solver="saga", C=0.1, max_iter=2000, random_state=U.SEED)
        score = np.abs(lr.fit(A[sub], y[sub]).coef_).sum(0)
    elif name == "rfe_cart":
        rfe = RFE(DecisionTreeClassifier(random_state=U.SEED), n_features_to_select=1, step=1)
        rfe.fit(A[sub], y[sub])
        return list(np.argsort(rfe.ranking_, kind="stable"))
    else:
        raise ValueError(name)
    return list(np.argsort(-np.asarray(score), kind="stable"))


SELECTORS = ["variance", "anova_f", "mutual_info", "chi2", "mrmr", "forest_importance", "l1_logistic", "rfe_cart"]


def selector_fold(i, Xs, ys, tr, va, ks, cfg):
    U.set_config(cfg)
    A, B, _ = U.fit_transform_fold(Xs[tr], Xs[va])
    As, ytrs = U.safe_smote(A, ys[tr])
    out = {}
    for name in SELECTORS:
        order = rank(name, As, ytrs)
        out[name] = [U.wf1(ys[va], U.detector().fit(As[:, order[:k]], ytrs).predict(B[:, order[:k]])) for k in ks]
    pca = PCA(random_state=U.SEED).fit(As)
    Pa, Pb = pca.transform(As), pca.transform(B)
    out["pca"] = [U.wf1(ys[va], U.detector().fit(Pa[:, :k], ytrs).predict(Pb[:, :k])) for k in ks]
    return out


def part_selectors(d, res):
    n = len(d["feat_names"])
    ks = list(range(1, n + 1))
    Xs, ys, folds = d["Xs"], d["ys"], d["folds"]
    fo = Parallel(n_jobs=U.N_JOBS, max_nbytes=None)(delayed(selector_fold)(i, Xs, ys, tr, va, ks, U.config())
                                   for i, (tr, va) in enumerate(folds))
    A, _, _ = U.fit_transform_fold(Xs)
    As, yss = U.safe_smote(A, ys)
    K_star = res["selection"]["K_star"]
    out = {"k_grid": ks, "K_star": K_star}
    for name in SELECTORS + ["pca"]:
        curves = np.array([f[name] for f in fo])
        mean, se = curves.mean(0), curves.std(0, ddof=1) / np.sqrt(len(folds))
        k_own, k_best, thr = U.one_se_rule(ks, mean, se)
        entry = {"cv_mean": mean.tolist(), "cv_se": se.tolist(), "k_own": k_own, "k_best": k_best,
                 "cv_at_k_own": float(mean[ks.index(k_own)]), "cv_at_K_star": float(mean[ks.index(K_star)])}
        if name == "pca":
            for tag, k in [("own_k", k_own), ("matched_K_star", K_star)]:
                r = fit_eval(d, list(range(n)), U.detector(), transform=PCA(n_components=k, random_state=U.SEED))
                entry[tag] = summary(r)
        else:
            order = rank(name, As, yss)
            entry["final_ranking"] = [d["feat_names"][j] for j in order]
            for tag, k in [("own_k", k_own), ("matched_K_star", K_star)]:
                entry[tag] = summary(fit_eval(d, sorted(order[:k]), U.detector()))
                entry[tag]["features"] = [d["feat_names"][j] for j in sorted(order[:k])]
        out[name] = entry
        print(f"   {name:18s} k*={k_own:3d} test wF1={entry['own_k']['weighted_f1']:.4f} "
              f"mF1={entry['own_k']['macro_f1']:.4f} | at K*={K_star}: wF1={entry['matched_K_star']['weighted_f1']:.4f} "
              f"mF1={entry['matched_K_star']['macro_f1']:.4f}", flush=True)
    return out


# ------------------------------------------------------------------ part 3: pruning
GRID = [(md, msl) for md in [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, None] for msl in [1, 10, 50]]


def make_tree(md, msl):
    return DecisionTreeClassifier(criterion="gini", class_weight="balanced", random_state=U.SEED,
                                  max_depth=md, min_samples_leaf=msl)


def pruning_fold(i, Xs, ys, tr, va, F, cfg):
    U.set_config(cfg)
    A, B, _ = U.fit_transform_fold(Xs[tr][:, F], Xs[va][:, F])
    As, ytrs = U.safe_smote(A, ys[tr])
    out = []
    for md, msl in GRID:
        h = make_tree(md, msl).fit(As, ytrs)
        out.append((U.wf1(ys[va], h.predict(B)), int(h.tree_.node_count)))
    return out


def part_pruning(d, res):
    S = subsets(res, d["feat_names"])
    out = {"grid": [[md, msl] for md, msl in GRID], "node_bytes": NODE_BYTES}
    for sname in ["task_aware", "all_features"]:
        F = S[sname]
        fo = Parallel(n_jobs=U.N_JOBS, max_nbytes=None)(delayed(pruning_fold)(i, d["Xs"], d["ys"], tr, va, F, U.config())
                                       for i, (tr, va) in enumerate(d["folds"]))
        f1 = np.array([[g[0] for g in f] for f in fo])
        nodes = np.array([[g[1] for g in f] for f in fo]).mean(0)
        mean, se = f1.mean(0), f1.std(0, ddof=1) / np.sqrt(len(fo))
        best = int(np.argmax(mean))
        ok = np.flatnonzero(mean >= mean[best] - se[best])
        pick = int(ok[np.argmin(nodes[ok])])
        md, msl = GRID[pick]
        entry = {"cv_mean": mean.tolist(), "cv_se": se.tolist(), "cv_nodes": nodes.tolist(),
                 "best": {"max_depth": GRID[best][0], "min_samples_leaf": GRID[best][1]},
                 "chosen": {"max_depth": md, "min_samples_leaf": msl,
                            "cv_wf1": float(mean[pick]), "cv_nodes": float(nodes[pick])},
                 "constrained": summary(fit_eval(d, F, make_tree(md, msl))),
                 "unconstrained": summary(fit_eval(d, F, U.detector()))}
        out[sname] = entry
        c, u = entry["constrained"], entry["unconstrained"]
        print(f"   {sname:13s} chosen depth={md} leaf={msl}: wF1 {c['weighted_f1']:.4f} (unconstr. {u['weighted_f1']:.4f}), "
              f"mF1 {c['macro_f1']:.4f} ({u['macro_f1']:.4f}), nodes {c['tree_nodes']} ({u['tree_nodes']}), "
              f"{c['model_size_kb']:.0f} KB ({u['model_size_kb']:.0f} KB)", flush=True)
    return out


PARTS = {"detectors": part_detectors, "selectors": part_selectors, "pruning": part_pruning}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", choices=["S1", "S2", "S3", "all"])
    ap.add_argument("task", nargs="?", choices=["binary", "multiclass"])
    ap.add_argument("--part", choices=list(PARTS) + ["all"], default="all")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    jobs = ([(s, t) for s in ["S1", "S2", "S3"] for t in ["binary", "multiclass"]]
            if a.scenario == "all" else [(a.scenario, a.task)])
    parts = list(PARTS) if a.part == "all" else [a.part]
    for s, t in jobs:
        d = U.prepare(s, t)
        res = load_result(s, t)
        for p in parts:
            print(f"[{s}/{t}] {p}", flush=True)
            t0 = time.time()
            out = PARTS[p](d, res)
            out["runtime_s"] = time.time() - t0
            out["config"] = U.config()
            with open(os.path.join(OUT, f"{s}_{t}_{p}.json"), "w") as fh:
                json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
