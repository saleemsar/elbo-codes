"""Cost of running the selection and the detector (reviewer comments R1.5 and R1.10).

selection   Single-core cost of the task-aware elbow for each scenario and task:
            one CV fold of the full K sweep is timed with all BLAS/OpenMP pools
            limited to one thread, split into K-means and detector fits; the full
            selection is 5 folds plus the final Stage 1-2 pass. The number of model
            fits is V(n-1) K-means runs (each with n_init restarts) and V(n-1) trees.

inference   For the task-aware subset F* and all features, with the unconstrained and
            the size-constrained tree (review/baselines.py, part pruning):
              fields to extract per flow, feature-vector bytes (float32),
              tree nodes and flat C-array size (NODE_BYTES per node), mean and maximum
              decision-path length on the test partition (comparisons per inference),
              and measured single-thread batch latency on this machine.
            Energy per inference is reported as a formula
              E = path_length x CYCLES_PER_NODE / f_clk x P_active
            with the assumed constants below stated in the output; it is an estimate,
            not a measurement on flight hardware.

Usage:  python review/deployment_cost.py
Output: review/deployment_cost/summary.json
"""
import json
import os
import platform
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits
from sklearn.tree import DecisionTreeClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import uav_ehfs as U  # noqa: E402

OUT = os.path.join(HERE, "deployment_cost")
NODE_BYTES = 12
# Assumptions for the energy estimate (stated in the manuscript as such)
CYCLES_PER_NODE = 20          # load feature + load threshold + compare + branch, compiled C on a Cortex-M class core
F_CLK_HZ = 400e6              # flight-controller class MCU clock
P_ACTIVE_W = 0.5              # active power of such an MCU at that clock
RUNS = [(s, t) for s in ["S1", "S2", "S3"] for t in ["binary", "multiclass"]]


def time_selection(d):
    Xs, ys = d["Xs"], d["ys"]
    tr, va = d["folds"][0]
    n = len(d["feat_names"])
    Ks = list(range(2, n + 1))
    with threadpool_limits(limits=1):
        t0 = time.perf_counter()
        A, B, _ = U.fit_transform_fold(Xs[tr], Xs[va])
        As, ytrs = U.safe_smote(A, ys[tr])
        R, rho = U.correlations(As, ytrs)
        t_prep = time.perf_counter() - t0
        t_km = t_tree = 0.0
        for K in Ks:
            t0 = time.perf_counter()
            lab, _ = U.kmeans_labels(R, K)
            F = U.representatives(lab, rho, K)
            t_km += time.perf_counter() - t0
            t0 = time.perf_counter()
            U.detector().fit(As[:, F], ytrs).predict(B[:, F])
            t_tree += time.perf_counter() - t0
    fold = t_prep + t_km + t_tree
    return {"n": n, "n_K": len(Ks), "balanced_fold_rows": int(len(ytrs)),
            "fold_seconds": {"preprocess_smote_correlation": t_prep, "kmeans": t_km, "detector": t_tree, "total": fold},
            "estimated_single_core_selection_s": U.N_FOLDS * fold + t_prep + t_km,
            "n_kmeans_runs": U.N_FOLDS * len(Ks), "n_detector_fits": U.N_FOLDS * len(Ks)}


def tree_cost(d, F, model):
    A, B, _ = U.fit_transform_fold(d["Xtr"][:, F], d["Xte"][:, F])
    As, ys = U.safe_smote(A, d["ytr"])
    model.fit(As, ys)
    path = np.asarray(model.decision_path(B).sum(1)).ravel() - 1   # internal nodes visited
    with threadpool_limits(limits=1):
        t0 = time.perf_counter()
        model.predict(B)
        ms = 1000 * (time.perf_counter() - t0) / len(B)
    nodes = int(model.tree_.node_count)
    mean_path = float(path.mean())
    e_j = mean_path * CYCLES_PER_NODE / F_CLK_HZ * P_ACTIVE_W
    return {"n_features": len(F), "feature_vector_bytes": 4 * len(F), "tree_nodes": nodes,
            "c_array_kb": nodes * NODE_BYTES / 1024, "depth": int(model.get_depth()),
            "mean_path_length": mean_path, "max_path_length": int(path.max()),
            "measured_ms_per_sample_batch_single_thread": ms,
            "estimated_energy_nJ_per_inference": 1e9 * e_j,
            "estimated_time_us_per_inference_on_mcu": 1e6 * mean_path * CYCLES_PER_NODE / F_CLK_HZ}


def main():
    os.makedirs(OUT, exist_ok=True)
    out = {"assumptions": {"node_bytes": NODE_BYTES, "cycles_per_node": CYCLES_PER_NODE,
                           "f_clk_hz": F_CLK_HZ, "p_active_w": P_ACTIVE_W},
           "machine": {"platform": platform.platform(), "processor": platform.processor(),
                       "cpu_count": os.cpu_count()}}
    for scn, task in RUNS:
        with open(os.path.join(U.OUT_DIR, f"{scn}_{task}.json")) as fh:
            res = json.load(fh)
        prune_path = os.path.join(HERE, "baselines", f"{scn}_{task}_pruning.json")
        prune = json.load(open(prune_path)) if os.path.exists(prune_path) else None
        d = U.prepare(scn, task)
        F_star = sorted(x["index"] for x in res["selection"]["selected_features"])
        F_all = list(range(len(d["feat_names"])))
        entry = {"selection": time_selection(d),
                 "sweep_runtime_s_parallel": res["selection"]["sweep_runtime_s"]}
        for tag, F in [("task_aware", F_star), ("all_features", F_all)]:
            entry[f"{tag}|unconstrained"] = tree_cost(d, F, U.detector())
            if prune is not None:
                c = prune[tag]["chosen"]
                m = DecisionTreeClassifier(criterion="gini", class_weight="balanced", random_state=U.SEED,
                                           max_depth=c["max_depth"], min_samples_leaf=c["min_samples_leaf"])
                entry[f"{tag}|constrained"] = tree_cost(d, F, m)
        out[f"{scn}_{task}"] = entry
        s = entry["selection"]
        print(f"[{scn}/{task}] single-core selection ~{s['estimated_single_core_selection_s']:.0f}s "
              f"(fold: kmeans {s['fold_seconds']['kmeans']:.1f}s, trees {s['fold_seconds']['detector']:.1f}s) | "
              + " | ".join(f"{k}: {v['n_features']} feat, {v['tree_nodes']} nodes, path {v['mean_path_length']:.1f}"
                           for k, v in entry.items() if "|" in k), flush=True)
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
