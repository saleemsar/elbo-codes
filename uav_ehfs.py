"""UAV-EHFS: task-aware elbow feature selection -- single shared pipeline.

One script, one stopping rule, used for every scenario (S1, S2, S3) and task
(binary, multiclass). Every number reported in the manuscript is written by this
script to results/<scenario>_<task>.json (+ test-set predictions), and the
manuscript tables are generated from those files by make_tables.py.

Stopping rule (manuscript Eq 2, Algorithm 1)
--------------------------------------------
For every K = 2..n (full sweep, no early stopping) the candidate subset F_K is
built by Stage 1 + Stage 2 and scored by the weighted F1 of the detector,
averaged over 5 stratified CV folds of the selection sample:  mean F1(K), SE(K).
    K_hat = argmax_K mean F1(K)
    K*    = min { K : mean F1(K) >= mean F1(K_hat) - SE(K_hat) }   (one-SE rule)
Everything that is fitted (imputer, scaler, SMOTE, correlations, K-means,
detector) is fitted on the CV training fold only; the 20% test partition is used
exactly once, after K* has been fixed.

Usage:  python uav_ehfs.py S1 multiclass      (or: python uav_ehfs.py all)
"""
import argparse
import gzip
import json
import os
import pickle
import platform
import re
import subprocess
import sys
import time
import warnings
import zipfile

import numpy as np
import pandas as pd
import sklearn
import imblearn
from imblearn.over_sampling import SMOTE
from joblib import Parallel, delayed
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

SEED = 42
TEST_SIZE = 0.20
N_FOLDS = 5
N_SEL = 60_000          # cap on the selection sample drawn from the training partition
MIN_PER_CLASS_SEL = 50  # rare classes are kept whole (up to this many rows) in the selection sample
KMEANS_N_INIT = 10
N_JOBS = int(os.environ.get("UAV_EHFS_JOBS", "4"))

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "dataset")
OUT_DIR = os.path.join(HERE, "results")
KEEP_DUPLICATES = False  # set by --keep-duplicates (sensitivity analysis only)

SCENARIOS = {
    "S1": {"name": "Compromised UAV", "zip": "UAV-Case1-Label (1).zip", "label": "Label"},
    "S2": {"name": "Rogue access point", "zip": "Access Point Case2 Label.zip", "label": "Normal"},
    "S3": {"name": "Compromised GCS", "zip": "GSC Case3 Label .zip", "label": "Class"},
}

# P1: identifiers / capture artefacts dropped by name (addresses, ports,
# flow ids, absolute timestamps, frame and sequence counters).
ID_COLUMNS = {
    "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p",
    "ip.src", "ip.dst", "udp.srcport", "udp.dstport", "wlan.bssid",
    "frame.number", "frame.time_epoch", "frame.time_relative",
    "radiotap.mactime", "radiotap.timestamp.ts", "wlan_radio.start_tsf",
    "wlan_radio.end_tsf", "wlan_radio.timestamp", "wlan.seq",
}
IP_FRACTION_DROP = 0.001  # P1 (content rule): drop a column if >0.1% of its values are IP addresses

LABEL_CANON = {
    "normal": "Normal", "benign": "Normal",
    "ddos": "DDoS", "dos": "DoS",
    "udp flooding": "UDP Flooding", "icmp flooding": "ICMP Flooding",
    "de-authentication": "De-authentication",
    "bruteforce": "Brute-Force", "brute-force": "Brute-Force",
    "evil twin": "Evil Twin", "ewil twin": "Evil Twin",
    "scanning": "Scanning",
    "reconnaissance": "Reconnaissance", "reconnassiance": "Reconnaissance",
    "jamming": "GPS Jamming",
    "fakelanding": "Fake Landing", "fake landing": "Fake Landing",
    "mitm": "MITM", "replay": "Replay",
}

_IP_RE = re.compile(r"^(\d{1,3}(\.\d{1,3}){3}|[0-9a-fA-F:]*:[0-9a-fA-F:]+)$")
_HEX_RE = re.compile(r"^0x[0-9a-fA-F]+$")


# --------------------------------------------------------------------------- data
def load_raw(scn):
    cfg = SCENARIOS[scn]
    with zipfile.ZipFile(os.path.join(DATA_DIR, cfg["zip"])) as z:
        name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        df = pd.read_csv(z.open(name), low_memory=False, dtype=str, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    return df, cfg["label"]


def canon_label(v):
    key = str(v).strip().lower()
    if key not in LABEL_CANON:
        raise ValueError(f"unknown label {v!r}")
    return LABEL_CANON[key]


def parse_numeric(col):
    """Decimal or hexadecimal strings -> float; anything else -> NaN (imputed later)."""
    s = col.astype(str).str.strip()
    num = pd.to_numeric(s, errors="coerce")
    hexm = s.str.match(_HEX_RE) & num.isna()
    if hexm.any():
        num[hexm] = s[hexm].map(lambda v: float(int(v, 16)))
    return num.astype("float64")


def preprocess(df, label_col, task):
    """P1 (identifier removal), P3 (label), P5 (numeric parsing), duplicate removal.
    None of these steps learns from labels or from value distributions across rows,
    except that duplicate removal is applied before the split so that no identical
    (features, label) row can appear in both training and test partitions."""
    log = {"n_rows_raw": int(len(df)), "n_columns_raw": int(df.shape[1] - 1)}
    y_raw = df[label_col].map(canon_label)
    log["class_counts_raw"] = {k: int(v) for k, v in y_raw.value_counts().items()}
    X = df.drop(columns=[label_col])

    dropped_name = [c for c in X.columns if c in ID_COLUMNS]
    X = X.drop(columns=dropped_name)
    dropped_ip = []
    for c in X.columns:
        s = X[c].dropna().astype(str).str.strip()
        if len(s) and s.str.match(_IP_RE).mean() > IP_FRACTION_DROP:
            dropped_ip.append(c)
    X = X.drop(columns=dropped_ip)
    log["p1_dropped_by_name"] = sorted(dropped_name)
    log["p1_dropped_ip_content"] = sorted(dropped_ip)

    Xn = pd.DataFrame({c: parse_numeric(X[c]) for c in X.columns})
    nonparse = {c: float((Xn[c].isna() & X[c].notna()).mean()) for c in X.columns}
    log["p5_unparsable_fraction"] = {c: round(v, 6) for c, v in nonparse.items() if v > 0}

    if task == "binary":
        y = y_raw.map(lambda v: "Normal" if v == "Normal" else "Attack")
    else:
        y = y_raw

    full = Xn.copy()
    full["__y__"] = y.values
    dup = full.duplicated(keep="first")
    fhash = pd.util.hash_pandas_object(Xn, index=False)
    conflict = pd.Series(y.values).groupby(fhash.values).nunique()
    log["n_duplicate_rows"] = int(dup.sum())
    log["n_feature_vectors_with_conflicting_labels"] = int((conflict > 1).sum())
    log["duplicates_removed"] = not KEEP_DUPLICATES
    if not KEEP_DUPLICATES:
        full = full[~dup].reset_index(drop=True)
    log["n_rows_used"] = int(len(full))
    y = full.pop("__y__")
    log["class_counts_used"] = {k: int(v) for k, v in y.value_counts().items()}
    return full, y, log


# ------------------------------------------------------------------------ helpers
def safe_smote(X, y, seed=SEED):
    """SMOTE every class with >= 2 samples up to the majority count.
    k_neighbors = min(5, smallest oversampled class - 1). Singleton classes are left as is."""
    counts = np.bincount(y)
    maj = counts.max()
    targets = {c: maj for c in np.flatnonzero(counts) if 2 <= counts[c] < maj}
    if not targets:
        return X, y
    k = int(min(5, min(counts[c] for c in targets) - 1))
    sm = SMOTE(sampling_strategy=targets, k_neighbors=k, random_state=seed)
    Xs, ys = sm.fit_resample(X, y)
    return Xs.astype(np.float32), ys


def fit_transform_fold(Xtr, Xva=None):
    imp = SimpleImputer(strategy="mean", keep_empty_features=True)
    sc = MinMaxScaler()
    A = sc.fit_transform(imp.fit_transform(Xtr)).astype(np.float32)
    if Xva is None:
        return A, None, (imp, sc)
    B = np.clip(sc.transform(imp.transform(Xva)), -1e6, 1e6).astype(np.float32)
    return A, B, (imp, sc)


def correlations(X, y):
    """Stage 1 matrix R (Eq 3) and Stage 2 scores rho (Eq 5)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        R = np.corrcoef(X, rowvar=False)
    R = np.nan_to_num(R)
    np.fill_diagonal(R, 1.0)
    yc = (y - y.mean()).astype(np.float64)
    Xc = X - X.mean(axis=0)
    denom = np.sqrt((Xc.astype(np.float64) ** 2).sum(0) * (yc ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = np.abs(np.nan_to_num((Xc.T.astype(np.float64) @ yc) / denom))
    return R, rho


def kmeans_labels(R, K):
    km = KMeans(n_clusters=K, n_init=KMEANS_N_INIT, random_state=SEED)
    lab = km.fit_predict(R)
    return lab, float(km.inertia_)


def representatives(labels, rho, K, rng=None):
    """Eq 6: highest-|rho| feature per cluster (ties -> lowest index).
    With rng given, a random member is taken instead (ablation variant B)."""
    reps = []
    for k in range(K):
        idx = np.flatnonzero(labels == k)
        if len(idx) == 0:
            continue
        reps.append(int(rng.choice(idx)) if rng is not None else int(idx[np.argmax(rho[idx])]))
    return sorted(set(reps))


def detector():
    return DecisionTreeClassifier(criterion="gini", class_weight="balanced", random_state=SEED)


def wf1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def selection_sample(X, y):
    """Stratified cap of the training partition for the selection loop; classes smaller
    than MIN_PER_CLASS_SEL are kept whole so that every class is represented."""
    rng = np.random.RandomState(SEED)
    n = len(y)
    frac = min(1.0, N_SEL / n)
    idx = []
    for c in np.unique(y):
        ci = np.flatnonzero(y == c)
        take = max(int(round(len(ci) * frac)), min(len(ci), MIN_PER_CLASS_SEL))
        idx.append(rng.choice(ci, size=take, replace=False) if take < len(ci) else ci)
    return np.sort(np.concatenate(idx))


# ----------------------------------------------------------------- CV machinery
def run_fold(fold, Xs, ys, tr, va, Ks, K_star=None):
    """Full K sweep on one fold. If K_star is given, also the ablation variants."""
    A, B, _ = fit_transform_fold(Xs[tr], Xs[va])
    ytr, yva = ys[tr], ys[va]
    As, ytrs = safe_smote(A, ytr)
    R, rho = correlations(As, ytrs)
    out = {"fold": fold, "curve": {}, "subsets": {}}
    labels_cache = {}
    for K in Ks:
        lab, _ = kmeans_labels(R, K)
        labels_cache[K] = lab
        F = representatives(lab, rho, K)
        h = detector().fit(As[:, F], ytrs)
        out["curve"][K] = wf1(yva, h.predict(B[:, F]))
        out["subsets"][K] = F
    if K_star is not None:
        ab = {"A_full": out["curve"][K_star]}
        rng = np.random.RandomState(SEED + fold)
        Fb = representatives(labels_cache[K_star], rho, K_star, rng=rng)
        ab["B_random_representative"] = wf1(yva, detector().fit(As[:, Fb], ytrs).predict(B[:, Fb]))
        ab["C_all_features"] = wf1(yva, detector().fit(As, ytrs).predict(B))
        R0, rho0 = correlations(A, ytr)
        lab0, _ = kmeans_labels(R0, K_star)
        F0 = representatives(lab0, rho0, K_star)
        ab["D_no_smote"] = wf1(yva, detector().fit(A[:, F0], ytr).predict(B[:, F0]))
        out["ablation"] = ab
    return out


def one_se_rule(Ks, mean, se):
    i_hat = int(np.argmax(mean))
    thr = mean[i_hat] - se[i_hat]
    i_star = int(np.flatnonzero(mean >= thr)[0])
    return Ks[i_star], Ks[i_hat], float(thr)


# --------------------------------------------------------------------- metrics
def metrics(y_true, y_pred, class_names, positive=None):
    L = list(range(len(class_names)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=L, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=L)
    mp, mr, mf, _ = precision_recall_fscore_support(y_true, y_pred, labels=L, average="macro", zero_division=0)
    wp, wr, wf, _ = precision_recall_fscore_support(y_true, y_pred, labels=L, average="weighted", zero_division=0)
    m = {
        "n_test": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro": {"precision": float(mp), "recall": float(mr), "f1": float(mf)},
        "weighted": {"precision": float(wp), "recall": float(wr), "f1": float(wf)},
        "per_class": [{"class": class_names[i], "precision": float(p[i]), "recall": float(r[i]),
                       "f1": float(f[i]), "support": int(s[i]),
                       "tp": int(cm[i, i]), "fp": int(cm[:, i].sum() - cm[i, i]),
                       "fn": int(cm[i, :].sum() - cm[i, i])} for i in L],
        "confusion_matrix": {"labels": class_names, "rows_true_cols_pred": cm.tolist()},
    }
    # Consistency checks: every aggregate recomputed from the confusion matrix.
    n = cm.sum()
    sup = cm.sum(1)
    pc = np.divide(np.diag(cm), cm.sum(0), out=np.zeros(len(L)), where=cm.sum(0) > 0)
    rc = np.divide(np.diag(cm), sup, out=np.zeros(len(L)), where=sup > 0)
    fc = np.divide(2 * pc * rc, pc + rc, out=np.zeros(len(L)), where=(pc + rc) > 0)
    chk = {
        "accuracy_from_cm": float(np.trace(cm) / n),
        "macro_f1_from_cm": float(fc.mean()),
        "weighted_f1_from_cm": float((fc * sup).sum() / n),
    }
    chk["all_consistent"] = bool(
        abs(chk["accuracy_from_cm"] - m["accuracy"]) < 1e-12
        and abs(chk["macro_f1_from_cm"] - mf) < 1e-9
        and abs(chk["weighted_f1_from_cm"] - wf) < 1e-9)
    m["consistency_checks"] = chk
    if positive is not None:
        pi = class_names.index(positive)
        ni = 1 - pi
        m["binary_counts"] = {"positive_class": positive,
                              "TP": int(cm[pi, pi]), "FN": int(cm[pi, ni]),
                              "FP": int(cm[ni, pi]), "TN": int(cm[ni, ni])}
    return m


def fit_final(Xtr, ytr, Xte, F):
    A, B, _ = fit_transform_fold(Xtr[:, F], Xte[:, F])
    As, ys = safe_smote(A, ytr)
    h = detector()
    t0 = time.perf_counter(); h.fit(As, ys); t_train = time.perf_counter() - t0
    t0 = time.perf_counter(); pred = h.predict(B); t_inf = time.perf_counter() - t0
    eff = {"n_features": len(F), "train_rows_after_smote": int(len(ys)),
           "train_time_s": t_train, "inference_ms_per_sample": 1000 * t_inf / len(B),
           "model_size_kb": len(pickle.dumps(h, protocol=pickle.HIGHEST_PROTOCOL)) / 1024,
           "tree_depth": int(h.get_depth()), "tree_leaves": int(h.get_n_leaves())}
    return pred, eff


# ------------------------------------------------------------------------ main
def git_commit():
    try:
        return subprocess.check_output(["git", "-C", HERE, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def run(scn, task):
    t_start = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    df, label_col = load_raw(scn)
    Xdf, y_str, prep = preprocess(df, label_col, task)
    del df

    if task == "binary":
        class_names = ["Normal", "Attack"]   # Attack (=1) is the positive class
    else:
        class_names = sorted(y_str.unique())
    code = {c: i for i, c in enumerate(class_names)}
    y = y_str.map(code).to_numpy()
    X = Xdf.to_numpy(dtype=np.float32)
    feat_names = list(Xdf.columns)
    row_ids = np.arange(len(y))

    Xtr, Xte, ytr, yte, idtr, idte = train_test_split(
        X, y, row_ids, test_size=TEST_SIZE, stratify=y, random_state=SEED)

    # P2: zero-variance columns, decided on the training partition only.
    keep = [j for j in range(X.shape[1])
            if np.nanmax(Xtr[:, j]) != np.nanmin(Xtr[:, j]) and not np.all(np.isnan(Xtr[:, j]))]
    prep["p2_dropped_zero_variance"] = [feat_names[j] for j in range(X.shape[1]) if j not in keep]
    Xtr, Xte = Xtr[:, keep], Xte[:, keep]
    feat_names = [feat_names[j] for j in keep]
    n = len(feat_names)
    prep["n_features_after_preprocessing"] = n

    split = {"n_train": int(len(ytr)), "n_test": int(len(yte)),
             "train_counts": {class_names[c]: int((ytr == c).sum()) for c in range(len(class_names))},
             "test_counts": {class_names[c]: int((yte == c).sum()) for c in range(len(class_names))}}

    # ---------------- selection sample + 5-fold CV sweep over K = 2..n
    sel = selection_sample(Xtr, ytr)
    Xs, ys = Xtr[sel], ytr[sel]
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(Xs, ys))
    Ks = list(range(2, n + 1))
    t0 = time.time()
    fold_out = Parallel(n_jobs=N_JOBS)(
        delayed(run_fold)(i, Xs, ys, tr, va, Ks) for i, (tr, va) in enumerate(folds))
    t_sweep = time.time() - t0
    per_fold = np.array([[fo["curve"][K] for K in Ks] for fo in fold_out])  # folds x Ks
    mean = per_fold.mean(0)
    sd = per_fold.std(0, ddof=1)
    se = sd / np.sqrt(N_FOLDS)
    K_star, K_hat, thr = one_se_rule(Ks, mean, se)

    # Ablation on the same folds at K*.
    ab_out = Parallel(n_jobs=N_JOBS)(
        delayed(run_fold)(i, Xs, ys, tr, va, [K_star], K_star) for i, (tr, va) in enumerate(folds))
    ablation = {}
    for v in ["A_full", "B_random_representative", "C_all_features", "D_no_smote"]:
        vals = [a["ablation"][v] for a in ab_out]
        ablation[v] = {"per_fold": vals, "mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1))}
    ablation["n_features"] = {"A_full": K_star, "B_random_representative": K_star,
                              "C_all_features": n, "D_no_smote": K_star}

    # ---------------- final subset: Stages 1-2 on the whole selection sample at K*
    A, _, _ = fit_transform_fold(Xs)
    As, yss = safe_smote(A, ys)
    R, rho = correlations(As, yss)
    lab_star, _ = kmeans_labels(R, K_star)
    F_star = representatives(lab_star, rho, K_star)
    wcss = []
    for K in Ks:
        wcss.append(kmeans_labels(R, K)[1])
    try:
        knee = KneeLocator(Ks, wcss, curve="convex", direction="decreasing").knee
    except Exception:
        knee = None
    K_wcss = int(knee) if knee is not None else None
    if K_wcss is not None:
        lab_w, _ = kmeans_labels(R, K_wcss)
        F_wcss = representatives(lab_w, rho, K_wcss)
    else:
        F_wcss = None

    selected = []
    for f in F_star:
        cl = int(lab_star[f])
        selected.append({"feature": feat_names[f], "index": int(f), "abs_rho": float(rho[f]),
                         "cluster": cl, "cluster_size": int((lab_star == cl).sum()),
                         "cluster_members": [feat_names[j] for j in np.flatnonzero(lab_star == cl)]})
    selected.sort(key=lambda d: -d["abs_rho"])

    # ---------------- sealed test evaluation (single use)
    positive = "Attack" if task == "binary" else None
    configs = {"task_aware": F_star, "all_features": list(range(n))}
    if F_wcss is not None:
        configs["wcss_kneedle"] = F_wcss
    test, efficiency, preds = {}, {}, {}
    for tag, F in configs.items():
        pred, eff = fit_final(Xtr, ytr, Xte, F)
        test[tag] = metrics(yte, pred, class_names, positive)
        efficiency[tag] = eff
        preds[tag] = pred

    res = {
        "scenario": scn, "scenario_name": SCENARIOS[scn]["name"], "task": task,
        "class_names": class_names,
        "protocol": {"seed": SEED, "test_size": TEST_SIZE, "n_folds": N_FOLDS,
                     "selection_sample_cap": N_SEL, "min_per_class_selection": MIN_PER_CLASS_SEL,
                     "kmeans_n_init": KMEANS_N_INIT, "K_range": [Ks[0], Ks[-1]],
                     "stopping_rule": "one-SE: K* = min{K : meanF1(K) >= meanF1(K_hat) - SE(K_hat)}, "
                                      "K_hat = argmax meanF1, full sweep K=2..n, 5-fold CV weighted F1",
                     "detector": "DecisionTreeClassifier(criterion=gini, class_weight=balanced, random_state=42)",
                     "smote": "all classes with >=2 samples to majority count, k=min(5, min class-1), training folds only"},
        "preprocessing": prep,
        "split": split,
        "selection": {"n_selection_rows": int(len(ys)),
                      "selection_counts": {class_names[c]: int((ys == c).sum()) for c in range(len(class_names))},
                      "K": Ks, "cv_f1_mean": mean.tolist(), "cv_f1_sd": sd.tolist(), "cv_f1_se": se.tolist(),
                      "cv_f1_per_fold": per_fold.tolist(),
                      "K_hat_argmax": K_hat, "one_se_threshold": thr, "K_star": K_star,
                      "n_selected": len(F_star), "reduction_pct": 100 * (1 - len(F_star) / n),
                      "cv_f1_at_K_star": float(mean[Ks.index(K_star)]),
                      "cv_f1_at_all_features": float(mean[-1]),
                      "K_wcss_kneedle": K_wcss,
                      "cv_f1_at_K_wcss": float(mean[Ks.index(K_wcss)]) if K_wcss else None,
                      "wcss_curve": wcss,
                      "selected_features": selected,
                      "wcss_features": [feat_names[f] for f in F_wcss] if F_wcss else None,
                      "sweep_runtime_s": t_sweep},
        "ablation_cv": ablation,
        "test": test,
        "efficiency": efficiency,
        "feature_names": feat_names,
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                        "imbalanced_learn": imblearn.__version__,
                        "platform": platform.platform(), "cpu_count": os.cpu_count(),
                        "n_jobs": N_JOBS, "git_commit": git_commit(),
                        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        "runtime_s": time.time() - t_start,
    }
    stem = os.path.join(OUT_DIR, f"{scn}_{task}")
    with open(stem + ".json", "w") as fh:
        json.dump(res, fh, indent=1)
    pdf = pd.DataFrame({"row_id": idte, "y_true": [class_names[c] for c in yte]})
    for tag, p in preds.items():
        pdf[f"y_pred_{tag}"] = [class_names[c] for c in p]
    with gzip.open(stem + "_test_predictions.csv.gz", "wt") as fh:
        pdf.to_csv(fh, index=False)
    t = test["task_aware"]
    print(f"[{scn}/{task}] n={n} K*={K_star} (K_hat={K_hat}, K_wcss={K_wcss}) "
          f"acc={t['accuracy']:.4f} wF1={t['weighted']['f1']:.4f} mF1={t['macro']['f1']:.4f} "
          f"consistent={t['consistency_checks']['all_consistent']} {res['runtime_s']:.0f}s", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", choices=["S1", "S2", "S3", "all"])
    ap.add_argument("task", nargs="?", choices=["binary", "multiclass"])
    ap.add_argument("--keep-duplicates", action="store_true",
                    help="sensitivity analysis: do not remove duplicate rows (results/with_duplicates/)")
    a = ap.parse_args()
    if a.keep_duplicates:
        KEEP_DUPLICATES = True
        OUT_DIR = os.path.join(OUT_DIR, "with_duplicates")
    jobs = ([(s, t) for s in ["S1", "S2", "S3"] for t in ["binary", "multiclass"]]
            if a.scenario == "all" else [(a.scenario, a.task)])
    for s, t in jobs:
        run(s, t)
