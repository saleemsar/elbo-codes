"""Checks on the evaluation itself (reviewer comments R1.3, R1.4, R1.12 and R2.6).

minority    Wilson 95% intervals for per-class recall and precision of the task-aware
            detector; classes with fewer than MIN_SUPPORT test flows are flagged as not
            estimable, and macro F1 is recomputed without them (R1.4, R2.6).
overlap     Exact duplicates are removed before the split, but a test row can still
            repeat a training row on the selected features. For F* and for all
            features: share of test rows whose vector occurs in training, and test
            scores on seen vs unseen vectors. Nearest-neighbour distance (scaled F*
            space) from test rows to training rows, against training-to-training
            distances (R2.6).
blocked     Group-type evaluation: each class is a contiguous block of capture in the
            CSV, so the first 80% (file order) of every class is used for training and
            the last 20% for testing. F* and all features, same preprocessing and SMOTE
            as fit_final (R2.6).
confusion   S3 multi-class: the largest confusions, the pairs Scanning/Reconnaissance
            and DoS/De-authentication, the number of F* vectors (and full vectors) that
            occur in both classes of a pair, and the pairwise balanced accuracy of a
            CART trained on just that pair with F* and with all features (R1.12).
smote_s1    S1 multi-class per-class test recall of F* with SMOTE + class weights (main),
            class weights only, SMOTE only, neither (R1.3).

Usage:  python review/evaluation_checks.py
Output: review/evaluation_checks/<check>.json
"""
import gzip
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import uav_ehfs as U  # noqa: E402

OUT = os.path.join(HERE, "evaluation_checks")
MIN_SUPPORT = 30
NN_ROWS = 5000
RUNS = [(s, t) for s in ["S1", "S2", "S3"] for t in ["binary", "multiclass"]]


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(max(0.0, c - h)), float(min(1.0, c + h))]


def load(scn, task):
    with open(os.path.join(U.OUT_DIR, f"{scn}_{task}.json")) as fh:
        res = json.load(fh)
    with gzip.open(os.path.join(U.OUT_DIR, f"{scn}_{task}_test_predictions.csv.gz"), "rt") as fh:
        pred = pd.read_csv(fh)
    return res, pred


def f_star(res):
    return sorted(d["index"] for d in res["selection"]["selected_features"])


def scores(y, p, labels):
    return {"n": int(len(y)), "accuracy": float(np.mean(y == p)) if len(y) else None,
            "weighted_f1": float(f1_score(y, p, labels=labels, average="weighted", zero_division=0)) if len(y) else None,
            "macro_f1": float(f1_score(y, p, labels=labels, average="macro", zero_division=0)) if len(y) else None}


# ------------------------------------------------------------------ minority
def check_minority():
    out = {}
    for scn, task in RUNS:
        res, _ = load(scn, task)
        m = res["test"]["task_aware"]
        rows, keep = [], []
        for c in m["per_class"]:
            n = c["support"]
            rows.append({"class": c["class"], "support": n, "recall": c["recall"],
                         "recall_ci95": wilson(c["tp"], n), "precision": c["precision"],
                         "precision_ci95": wilson(c["tp"], c["tp"] + c["fp"]),
                         "estimable": n >= MIN_SUPPORT})
            if n >= MIN_SUPPORT:
                keep.append(c["f1"])
        out[f"{scn}_{task}"] = {"per_class": rows, "macro_f1_all": m["macro"]["f1"],
                                "macro_f1_estimable_only": float(np.mean(keep)),
                                "n_not_estimable": sum(not r["estimable"] for r in rows)}
    return out


# ------------------------------------------------------------------ overlap
def check_overlap():
    out = {}
    for scn, task in RUNS:
        res, pred = load(scn, task)
        d = U.prepare(scn, task)
        assert np.array_equal(pred["row_id"].to_numpy(), d["idte"]), "prediction rows out of order"
        y = pred["y_true"].to_numpy()
        labels = d["class_names"]
        entry = {}
        for tag, F in [("task_aware", f_star(res)), ("all_features", list(range(len(d["feat_names"]))))]:
            htr = set(pd.util.hash_pandas_object(pd.DataFrame(d["Xtr"][:, F]), index=False))
            hte = pd.util.hash_pandas_object(pd.DataFrame(d["Xte"][:, F]), index=False).to_numpy()
            seen = np.array([h in htr for h in hte])
            p = pred[f"y_pred_{tag}"].to_numpy()
            entry[tag] = {"share_test_vectors_seen_in_train": float(seen.mean()),
                          "seen": scores(y[seen], p[seen], labels), "unseen": scores(y[~seen], p[~seen], labels)}
        # nearest-neighbour distances in the scaled F* space
        F = f_star(res)
        A, B, _ = U.fit_transform_fold(d["Xtr"][:, F], d["Xte"][:, F])
        rng = np.random.RandomState(U.SEED)
        tr = rng.choice(len(A), size=min(NN_ROWS, len(A)), replace=False)
        te = rng.choice(len(B), size=min(NN_ROWS, len(B)), replace=False)
        nn = NearestNeighbors(n_neighbors=2).fit(A)
        d_te = nn.kneighbors(B[te], n_neighbors=1)[0][:, 0]
        d_tr = nn.kneighbors(A[tr], n_neighbors=2)[0][:, 1]   # nearest other training row
        q = [0.25, 0.5, 0.75, 0.9]
        entry["nn_distance_task_aware"] = {"quantiles": q,
                                           "test_to_train": np.quantile(d_te, q).tolist(),
                                           "train_to_train": np.quantile(d_tr, q).tolist(),
                                           "share_zero_test_to_train": float((d_te == 0).mean()),
                                           "share_zero_train_to_train": float((d_tr == 0).mean())}
        out[f"{scn}_{task}"] = entry
        print(scn, task, {k: round(v["share_test_vectors_seen_in_train"], 3) for k, v in entry.items() if k != "nn_distance_task_aware"}, flush=True)
    return out


# ------------------------------------------------------------------ blocked split
def check_blocked():
    out = {}
    for scn, task in RUNS:
        res, _ = load(scn, task)
        df, label_col = U.load_raw(scn)
        Xdf, y_str, _ = U.preprocess(df, label_col, task, scn)
        del df
        class_names = res["class_names"]
        code = {c: i for i, c in enumerate(class_names)}
        y = y_str.map(code).to_numpy()
        names = res["feature_names"]
        X = Xdf[names].to_numpy(dtype=np.float32)
        tr = np.zeros(len(y), bool)
        for c in np.unique(y):
            idx = np.flatnonzero(y == c)            # file order = capture order within the class
            tr[idx[: int(np.floor(0.8 * len(idx)))]] = True
        d = {"Xtr": X[tr], "Xte": X[~tr], "ytr": y[tr], "yte": y[~tr]}
        entry = {"n_train": int(tr.sum()), "n_test": int((~tr).sum())}
        for tag, F in [("task_aware", f_star(res)), ("all_features", list(range(len(names))))]:
            A, B, _ = U.fit_transform_fold(d["Xtr"][:, F], d["Xte"][:, F])
            As, ys = U.safe_smote(A, d["ytr"])
            p = U.detector().fit(As, ys).predict(B)
            positive = "Attack" if task == "binary" else None
            m = U.metrics(d["yte"], p, class_names, positive)
            entry[tag] = {"accuracy": m["accuracy"], "weighted_f1": m["weighted"]["f1"],
                          "macro_f1": m["macro"]["f1"], "per_class": m["per_class"]}
            rnd = res["test"][tag]
            entry[tag]["random_split_weighted_f1"] = rnd["weighted"]["f1"]
            entry[tag]["random_split_macro_f1"] = rnd["macro"]["f1"]
        out[f"{scn}_{task}"] = entry
        print(scn, task, "blocked wF1/mF1", round(entry["task_aware"]["weighted_f1"], 4), round(entry["task_aware"]["macro_f1"], 4),
              "random", round(entry["task_aware"]["random_split_weighted_f1"], 4), round(entry["task_aware"]["random_split_macro_f1"], 4), flush=True)
    return out


# ------------------------------------------------------------------ S3 confusion
PAIRS = [("Scanning", "Reconnaissance"), ("DoS", "De-authentication"), ("DoS", "DDoS")]


def check_confusion():
    res, pred = load("S3", "multiclass")
    d = U.prepare("S3", "multiclass")
    names = d["class_names"]
    cm = np.array(res["test"]["task_aware"]["confusion_matrix"]["rows_true_cols_pred"])
    off = [(names[i], names[j], int(cm[i, j])) for i in range(len(names)) for j in range(len(names)) if i != j and cm[i, j]]
    off.sort(key=lambda t: -t[2])
    X = np.vstack([d["Xtr"], d["Xte"]])
    y = np.concatenate([d["ytr"], d["yte"]])
    F = f_star(res)
    out = {"largest_confusions": off[:12], "pairs": {}}
    skf = StratifiedKFold(5, shuffle=True, random_state=U.SEED)
    for a, b in PAIRS:
        ia, ib = names.index(a), names.index(b)
        m = (y == ia) | (y == ib)
        e = {"n_a": int((y == ia).sum()), "n_b": int((y == ib).sum()),
             "test_a_pred_b": int(cm[ia, ib]), "test_b_pred_a": int(cm[ib, ia])}
        for tag, cols in [("task_aware", F), ("all_features", list(range(X.shape[1])))]:
            Z = pd.DataFrame(X[m][:, cols])
            h = pd.util.hash_pandas_object(Z, index=False).to_numpy()
            ya = y[m]
            ha, hb = set(h[ya == ia]), set(h[ya == ib])
            shared = ha & hb
            e[tag] = {"distinct_vectors_a": len(ha), "distinct_vectors_b": len(hb), "shared_vectors": len(shared),
                      "rows_a_on_shared_vectors": int(sum(1 for v, l in zip(h, ya) if l == ia and v in shared)),
                      "rows_b_on_shared_vectors": int(sum(1 for v, l in zip(h, ya) if l == ib and v in shared))}
            Zf = np.nan_to_num(Z.to_numpy(), nan=0.0)
            if min((ya == ia).sum(), (ya == ib).sum()) >= 5:
                p = cross_val_predict(DecisionTreeClassifier(class_weight="balanced", random_state=U.SEED), Zf, ya, cv=skf)
                e[tag]["pairwise_cv_balanced_accuracy"] = float(balanced_accuracy_score(ya, p))
        out["pairs"][f"{a}|{b}"] = e
        print("S3 pair", a, b, e, flush=True)
    return out


# ------------------------------------------------------------------ S1 SMOTE
def check_smote_s1():
    res, _ = load("S1", "multiclass")
    d = U.prepare("S1", "multiclass")
    F = f_star(res)
    A, B, _ = U.fit_transform_fold(d["Xtr"][:, F], d["Xte"][:, F])
    out = {}
    for tag, smote, cw in [("smote_and_class_weight", True, "balanced"), ("class_weight_only", False, "balanced"),
                           ("smote_only", True, None), ("neither", False, None)]:
        Xa, ya = U.safe_smote(A, d["ytr"]) if smote else (A, d["ytr"])
        h = DecisionTreeClassifier(criterion="gini", class_weight=cw, random_state=U.SEED).fit(Xa, ya)
        m = U.metrics(d["yte"], h.predict(B), d["class_names"])
        out[tag] = {"weighted_f1": m["weighted"]["f1"], "macro_f1": m["macro"]["f1"],
                    "recall": {c["class"]: c["recall"] for c in m["per_class"]},
                    "support": {c["class"]: c["support"] for c in m["per_class"]},
                    "tree_leaves": int(h.get_n_leaves())}
        print("S1 smote", tag, round(out[tag]["weighted_f1"], 4), round(out[tag]["macro_f1"], 4), flush=True)
    return out


def check_ceiling():
    """Accuracy ceiling set by conflicting labels: on the deduplicated multi-class data,
    predict the most frequent label of every distinct feature vector."""
    out = {}
    for scn in ["S1", "S2", "S3"]:
        df, label_col = U.load_raw(scn)
        X, y, _ = U.preprocess(df, label_col, "multiclass", scn)
        h = pd.util.hash_pandas_object(X, index=False).values
        g = pd.DataFrame({"h": h, "y": y.values}).groupby(["h", "y"]).size().unstack(fill_value=0)
        conflicting = (g > 0).sum(axis=1) > 1
        out[scn] = {"rows": int(len(y)), "max_accuracy": float(g.max(axis=1).sum() / len(y)),
                    "share_rows_on_conflicting_vectors": float(g[conflicting].sum(axis=1).sum() / len(y))}
        print(scn, out[scn], flush=True)
    return out


CHECKS = {"ceiling": check_ceiling, "minority": check_minority, "overlap": check_overlap, "blocked": check_blocked,
          "confusion": check_confusion, "smote_s1": check_smote_s1}


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    todo = sys.argv[1:] or list(CHECKS)
    for name in todo:
        r = CHECKS[name]()
        with open(os.path.join(OUT, f"{name}.json"), "w") as fh:
            json.dump(r, fh, indent=1)
        print("wrote", name, flush=True)
